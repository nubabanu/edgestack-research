"""Compact JSON toolbelt so AI agents can drive EdgeStack in one call each.

Every command prints a small, token-budgeted JSON document to stdout and never
raises: anything unavailable degrades to a ``NOT_AVAILABLE`` section carrying
the reason. Projections reuse the exact payload conventions the nightly job
and mobile service already ship, and every underlying status stamp,
watermark, and disclaimer is preserved verbatim — this module summarizes,
it never re-labels.

Usage: ``python -m edgestack.agenttools <command>`` or ``edgestack agent
<command>``; start with ``describe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
from datetime import date as date_type
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    add_completion=False,
    help="Compact JSON access to EdgeStack for AI agents; run `describe` first.",
)

_TOM_CONFIG = "configs/spy-tom-edge-v1.yaml"
_ALIGNMENT_ROW_KEYS = (
    "session",
    "weekday",
    "expected_daily_bp",
    "win_score_0_100",
    "active_calendar_conditions",
)
_CONDITION_BRIEF_KEYS = (
    "name",
    "kind",
    "classification",
    "shrunk_mean_daily",
    "hit_rate",
    "hac_t",
    "dsr_probability",
    "n",
    "regime_down_mean",
    "regime_up_mean",
    "worst_session",
    "decayed_in_recent_third",
)


def _emit(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _not_available(reason: str) -> dict[str, Any]:
    return {"status": "NOT_AVAILABLE", "reason": reason}


def _guarded(build: Any) -> dict[str, Any]:
    """Run a zero-argument section builder, degrading errors to a section."""

    try:
        result: dict[str, Any] = build()
    except Exception as error:  # a missing artifact must not kill the report
        return _not_available(f"{type(error).__name__}: {error}")
    return result


def _fetch_bars(symbol: str, years: int) -> tuple[Any, tuple[str, ...]]:
    """Free-chain daily bars for one symbol (Stooq first, Yahoo fallback)."""

    from edgestack.data.sources import (
        FallbackDailyBarSource,
        StooqDailyBarSource,
        YahooDailyBarSource,
        bars_to_frame,
    )
    from edgestack.models import AssetKey, BarRequest

    async def _fetch() -> tuple[Any, tuple[str, ...]]:
        chain = FallbackDailyBarSource((StooqDailyBarSource(), YahooDailyBarSource()))
        batch = await chain.fetch_bars(
            BarRequest(
                AssetKey(symbol.upper()),
                date_type.today() - timedelta(days=365 * years),
                date_type.today(),
                adjusted=True,
            )
        )
        return bars_to_frame(batch), tuple(batch.warnings)

    return asyncio.run(_fetch())


def _condition_brief(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: entry.get(key) for key in _CONDITION_BRIEF_KEYS}


def _alignment_brief(row: dict[str, Any]) -> dict[str, Any]:
    brief = {key: row.get(key) for key in _ALIGNMENT_ROW_KEYS}
    expected = brief.get("expected_daily_bp")
    if isinstance(expected, float):
        brief["expected_daily_bp"] = round(expected, 2)
    return brief


def _window_brief(window: dict[str, Any]) -> dict[str, Any]:
    """Keep the human instructions, drop the full evidence blobs."""

    kept: dict[str, Any] = {}
    for key, value in window.items():
        if key.endswith("evidence"):
            continue
        if key == "ranking":
            kept[key] = value[:3]
        elif isinstance(value, dict) and "evidence" in value:
            kept[key] = {k: v for k, v in value.items() if k != "evidence"}
            evidence = value["evidence"]
            kept[key]["classification"] = evidence.get("classification")
        else:
            kept[key] = value
    return kept


def _advise_report(
    symbol: str,
    *,
    years: int,
    sessions: int,
    buy_date: str | None,
    root: Path,
) -> dict[str, Any]:
    from edgestack.advisor import advise

    frame, warnings = _fetch_bars(symbol, years)
    buy_session = date_type.fromisoformat(buy_date) if buy_date else None
    return advise(
        frame,
        symbol=symbol.upper(),
        scan_sessions=sessions,
        buy_session=buy_session,
        provenance_warnings=warnings,
        root=root,
    )


def _advise_payload(report: dict[str, Any], *, top: int) -> dict[str, Any]:
    alignment = report["alignment"]
    payload: dict[str, Any] = {
        "status": report["status"],
        "symbol": report["symbol"],
        "as_of_session": report["as_of_session"],
        "history_sessions": report["history_sessions"],
        "validated_edges": report["validated_edges"],
        "current_year_context": report["current_year_context"],
        "tailwinds": [_condition_brief(entry) for entry in report["tailwinds"]],
        "headwinds": [_condition_brief(entry) for entry in report["headwinds"]],
        "timing": {
            "anchors": report["timing"]["anchors"],
            "week": _window_brief(report["timing"]["week"]),
            "month": _window_brief(report["timing"]["month"]),
            "year": _window_brief(report["timing"]["year"]),
            "execution": report["timing"]["execution"],
        },
        "best_upcoming_sessions": [
            _alignment_brief(row) for row in alignment["all_stars_aligned"][:top]
        ],
        "worst_upcoming_sessions": [
            _alignment_brief(row) for row in alignment["worst_sessions"][:top]
        ],
        "provenance_warnings": report["provenance_warnings"],
        "disclaimer": report["disclaimer"],
    }
    assessment = report.get("buy_time_assessment")
    if assessment is not None:
        if "choice_review" in assessment:
            review = assessment["choice_review"]
            assessment = {
                key: value
                for key, value in assessment.items()
                if key != "choice_review"
            }
            assessment["choice_review"] = {
                "weekday_rank": review.get("weekday_rank"),
                "better_upcoming_sessions": review.get("better_upcoming_sessions", [])[
                    :top
                ],
                "sell_plan_by_horizon": review.get("sell_plan_by_horizon"),
            }
        payload["buy_time_assessment"] = assessment
    return payload


def _calendar_payload(report: dict[str, Any]) -> dict[str, Any]:
    """The exact projection the nightly job publishes for the mobile app."""

    return {
        "status": report["status"],
        "symbol": report["symbol"],
        "as_of_session": report["as_of_session"],
        "policy": report["alignment"]["policy"],
        "anchors": report["timing"]["anchors"],
        "calendar": report["alignment"]["calendar"],
        "validated_edges": report["validated_edges"],
        "provenance_warnings": report["provenance_warnings"],
        "disclaimer": report["disclaimer"],
    }


def _tom_section(root: Path) -> dict[str, Any]:
    def _build() -> dict[str, Any]:
        from edgestack.edges.turn_of_month import next_trade

        # Catalog's constructor creates the sqlite store; probing must not.
        if not (root / "artifacts" / "edgestack.sqlite").is_file():
            return _not_available("no campaign catalog at artifacts/edgestack.sqlite")
        with contextlib.redirect_stdout(io.StringIO()):
            plan = next_trade(_TOM_CONFIG, root=root)
        return {"status": "AVAILABLE", **plan}

    return _guarded(_build)


def _gate_rows(root: Path, campaign: str | None) -> dict[str, Any]:
    def _build() -> dict[str, Any]:
        from edgestack.storage.catalog import Catalog

        catalog_path = root / "artifacts" / "edgestack.sqlite"
        if not catalog_path.is_file():
            return _not_available(f"no catalog at {catalog_path}")
        query = "SELECT campaign_id, phase, status, checked_at, summary FROM gates"
        parameters: tuple[str, ...] = ()
        if campaign is not None:
            query += " WHERE campaign_id=?"
            parameters = (campaign,)
        with Catalog(catalog_path).connect() as connection:
            rows = connection.execute(
                query + " ORDER BY campaign_id, checked_at", parameters
            ).fetchall()
        return {
            "status": "AVAILABLE",
            "gates": [
                {
                    "campaign_id": row["campaign_id"],
                    "phase": row["phase"],
                    "status": row["status"],
                    "checked_at": row["checked_at"],
                    "summary": row["summary"],
                }
                for row in rows
            ],
        }

    return _guarded(_build)


def _forward_section(root: Path, campaign: str) -> dict[str, Any]:
    def _build() -> dict[str, Any]:
        from edgestack.live.forward_ledger import ForwardLedger

        ledger_path = (
            root / "artifacts" / "campaigns" / campaign / "forward" / "ledger.sqlite"
        )
        if not ledger_path.is_file():
            return _not_available(f"no forward ledger at {ledger_path}")
        return {
            "status": "AVAILABLE",
            "campaign_id": campaign,
            "scorecard": ForwardLedger(ledger_path).scorecard(),
        }

    return _guarded(_build)


def _calendars_section(root: Path) -> dict[str, Any]:
    def _build() -> dict[str, Any]:
        advisor_dir = root / "artifacts" / "advisor"
        published = []
        for path in sorted(advisor_dir.glob("tailwind-calendar-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            published.append(
                {
                    "symbol": payload.get("symbol"),
                    "as_of_session": payload.get("as_of_session"),
                    "path": str(path),
                }
            )
        if not published:
            return _not_available(
                "no published calendars; run `edgestack agent calendar <SYMBOL>"
                " --publish` or the nightly post-close job"
            )
        return {"status": "AVAILABLE", "published": published}

    return _guarded(_build)


def _oil_section(root: Path) -> dict[str, Any]:
    def _build() -> dict[str, Any]:
        latest_path = root / "artifacts" / "oil" / "latest.json"
        if not latest_path.is_file():
            return _not_available(
                f"no persisted oil snapshot at {latest_path}; refresh with"
                " `edgestack oil-decision`"
            )
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        section: dict[str, Any] = {"status": "AVAILABLE", "latest": latest}
        ledger_path = root / "artifacts" / "oil" / "forward.sqlite"
        if ledger_path.is_file():
            from edgestack.oil.ledger import OilLedger

            section["scorecard"] = OilLedger(ledger_path).scorecard()
        return section

    return _guarded(_build)


@app.command("describe")
def describe() -> None:
    """Machine-readable command list; the entry point for a new agent."""

    _emit(
        {
            "tool": "edgestack.agenttools",
            "purpose": (
                "compact JSON access to EdgeStack research output; diagnostic"
                " evidence only, never orders or advice"
            ),
            "commands": {
                "describe": {"args": {}, "network": False},
                "overview": {
                    "args": {"--root": "repo root (default .)"},
                    "network": False,
                    "returns": "gates, TOM plan, forward scorecard, calendars, oil",
                },
                "advise": {
                    "args": {
                        "SYMBOL": "any US ticker",
                        "--buy-date": "rate one intended session (YYYY-MM-DD)",
                        "--years": "history depth (default 20)",
                        "--sessions": "forward scan length (default 63)",
                        "--top": "rows per best/worst list (default 5)",
                    },
                    "network": True,
                    "returns": "compact diagnostic timing report",
                },
                "calendar": {
                    "args": {
                        "SYMBOL": "any US ticker",
                        "--sessions": "scan length (default 63)",
                        "--rows": "stdout rows (default 21; publish keeps all)",
                        "--publish": "write artifacts/advisor calendar for the app",
                    },
                    "network": True,
                },
                "compare": {
                    "args": {"SYMBOLS": "comma-separated tickers"},
                    "network": True,
                    "returns": "symbols ranked by best upcoming session",
                },
                "tom": {"args": {}, "network": False},
                "oil": {"args": {"--root": "repo root"}, "network": False},
                "gates": {
                    "args": {"--campaign": "optional campaign id filter"},
                    "network": False,
                },
                "telegram-test": {
                    "args": {},
                    "network": True,
                    "returns": "SENT / SKIPPED_NO_CREDENTIALS with setup steps",
                },
                "entry-check": {
                    "args": {
                        "SYMBOL": "any US ticker",
                        "--sessions": "look-ahead sessions (default 42)",
                        "--threshold-bp": "GOOD verdict threshold (default 15)",
                    },
                    "network": True,
                    "returns": (
                        "per-session GOOD/WAIT/CAUTION_EARNINGS verdicts fusing"
                        " alignment, calm regime, dip state, and the estimated"
                        " earnings window"
                    ),
                },
                "leverage-check": {
                    "args": {
                        "SYMBOL": "any US ticker",
                        "--leverage": "leverage to stress (default 5)",
                        "--horizon": "sessions per position (default 60)",
                        "--years": "history depth (default 20)",
                    },
                    "network": True,
                    "returns": (
                        "liquidation rate, median outcome, and max leverage at"
                        " 95% survival under any/calm/calm+dip entries"
                    ),
                },
            },
            "honesty_contract": (
                "outputs keep every status stamp, watermark, and disclaimer;"
                " NOT_AVAILABLE sections are reported instead of guesses;"
                " validated-edge tier is SPY turn-of-month only"
            ),
        }
    )


@app.command("overview")
def overview(
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path("."),
) -> None:
    """Offline one-call system status assembled from persisted artifacts."""

    from edgestack.disclaimer import DISCLAIMER
    from edgestack.live.daily_job import DEFAULT_CAMPAIGN

    base = root.resolve()
    _emit(
        {
            "status": "DIAGNOSTIC_SNAPSHOT_FROM_PERSISTED_ARTIFACTS",
            "root": str(base),
            "gates": _gate_rows(base, None),
            "turn_of_month": _tom_section(base),
            "forward": _forward_section(base, DEFAULT_CAMPAIGN),
            "calendars": _calendars_section(base),
            "oil": _oil_section(base),
            "disclaimer": DISCLAIMER,
        }
    )


@app.command("advise")
def advise_command(
    symbol: Annotated[str, typer.Argument(help="Ticker, e.g. ACN.")],
    buy_date: Annotated[
        str | None, typer.Option("--buy-date", help="Rate this session.")
    ] = None,
    years: Annotated[int, typer.Option("--years", min=2, max=60)] = 20,
    sessions: Annotated[int, typer.Option("--sessions", min=5, max=252)] = 63,
    top: Annotated[int, typer.Option("--top", min=1, max=21)] = 5,
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path("."),
) -> None:
    """Compact diagnostic timing report for one symbol (fetches bars)."""

    def _build() -> dict[str, Any]:
        report = _advise_report(
            symbol,
            years=years,
            sessions=sessions,
            buy_date=buy_date,
            root=root.resolve(),
        )
        return _advise_payload(report, top=top)

    _emit(_guarded(_build))


@app.command("calendar")
def calendar_command(
    symbol: Annotated[str, typer.Argument(help="Ticker, e.g. ACN.")],
    sessions: Annotated[int, typer.Option("--sessions", min=5, max=252)] = 63,
    rows: Annotated[int, typer.Option("--rows", min=1, max=252)] = 21,
    publish: Annotated[
        bool,
        typer.Option(
            "--publish",
            help="Write artifacts/advisor/tailwind-calendar-<SYMBOL>.json.",
        ),
    ] = False,
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path("."),
) -> None:
    """Forward tailwind calendar in the nightly-job payload shape."""

    def _build() -> dict[str, Any]:
        base = root.resolve()
        report = _advise_report(
            symbol, years=20, sessions=sessions, buy_date=None, root=base
        )
        payload = _calendar_payload(report)
        if publish:
            advisor_dir = base / "artifacts" / "advisor"
            advisor_dir.mkdir(parents=True, exist_ok=True)
            target = advisor_dir / f"tailwind-calendar-{payload['symbol']}.json"
            target.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            payload["published_to"] = str(target)
        payload["calendar"] = payload["calendar"][:rows]
        return payload

    _emit(_guarded(_build))


@app.command("compare")
def compare_command(
    symbols: Annotated[
        str, typer.Argument(help="Comma-separated tickers, e.g. ACN,CTSH,SPY.")
    ],
    sessions: Annotated[int, typer.Option("--sessions", min=5, max=252)] = 42,
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path("."),
) -> None:
    """Rank symbols by their single best upcoming alignment session."""

    base = root.resolve()
    ranked: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    for raw in symbols.split(","):
        ticker = raw.strip().upper()
        if not ticker:
            continue

        def _build(ticker: str = ticker) -> dict[str, Any]:
            report = _advise_report(
                ticker, years=20, sessions=sessions, buy_date=None, root=base
            )
            best = report["alignment"]["all_stars_aligned"][0]
            return {
                "symbol": ticker,
                "as_of_session": report["as_of_session"],
                "best_session": _alignment_brief(best),
                "trend_state": report["current_year_context"]["trend_state"],
            }

        entry = _guarded(_build)
        if entry.get("status") == "NOT_AVAILABLE":
            failures[ticker] = entry["reason"]
        else:
            ranked.append(entry)
    ranked.sort(
        key=lambda item: item["best_session"]["expected_daily_bp"] or 0.0,
        reverse=True,
    )
    _emit(
        {
            "status": "DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER",
            "policy": (
                "ranked by the DSR-reliability-weighted expected daily basis"
                " points of each symbol's best upcoming session"
            ),
            "ranked": ranked,
            "failures": failures,
        }
    )


@app.command("tom")
def tom_command(
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path("."),
) -> None:
    """The validated SPY turn-of-month plan, or why it is unavailable."""

    _emit(_tom_section(root.resolve()))


@app.command("oil")
def oil_command(
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path("."),
) -> None:
    """Latest persisted oil paper decision and scorecard (offline)."""

    _emit(_oil_section(root.resolve()))


@app.command("gates")
def gates_command(
    campaign: Annotated[str | None, typer.Option("--campaign")] = None,
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path("."),
) -> None:
    """Campaign gate results from the catalog (read-only)."""

    _emit(_gate_rows(root.resolve(), campaign))


def _leverage_assessment(
    frame: Any, symbol: str, *, leverage: float, horizon: int
) -> dict[str, Any]:
    """Path-based liquidation analysis for one symbol at one leverage.

    Assumptions are declared, not hidden: entry at the NEXT session's close
    after a signal day; a position is closed out when the intraday adjusted
    low breaches a 0.5/leverage excursion (a 50% maintenance close-out,
    executed cleanly at the threshold — real overnight gaps make the true
    tail WORSE); entry windows overlap, so samples are dependent.
    """

    import numpy as np
    import pandas as pd

    bars = (
        frame.loc[frame["symbol"] == symbol.upper()].set_index("session").sort_index()
    )
    close = bars["close"].astype(float)
    adjusted = bars["adjusted_close"].astype(float)
    low = bars["low"].astype(float)
    high = bars["high"].astype(float)
    returns = adjusted.pct_change()
    adjusted_low = low * adjusted / close
    sma200 = close.rolling(200, min_periods=200).mean()
    vol20 = returns.rolling(20, min_periods=20).std() * np.sqrt(252.0)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi2 = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))
    span = (high - low).replace(0.0, np.nan)
    ibs = ((close - low) / span).fillna(0.5)
    down3 = (returns < 0) & (returns.shift(1) < 0) & (returns.shift(2) < 0)
    dip = (rsi2 < 10.0) | down3 | (ibs < 0.2)
    calm = (close > sma200) & (vol20 < 0.30)
    conditions = {
        "any_entry": sma200.notna(),
        "calm_regime": calm.fillna(False),
        "calm_and_dip": (calm & dip).fillna(False),
    }
    liq_threshold = 0.5 / leverage
    report: dict[str, Any] = {}
    for name, mask in conditions.items():
        indices = np.flatnonzero(mask.to_numpy())
        outcomes: list[float] = []
        excursions: list[float] = []
        liquidated = 0
        for i in indices:
            if i + 1 + horizon >= len(bars):
                continue
            entry = float(adjusted.iloc[i + 1])
            # Excursions strictly AFTER the entry close; the entry session's
            # own low happened before the fill and cannot stop us out.
            path_low = float(adjusted_low.iloc[i + 2 : i + 2 + horizon].min())
            excursion = 1.0 - path_low / entry
            excursions.append(excursion)
            if excursion >= liq_threshold:
                liquidated += 1
                outcomes.append(-0.5)
            else:
                outcomes.append(
                    leverage * (float(adjusted.iloc[i + 1 + horizon]) / entry - 1.0)
                )
        if len(outcomes) < 8:
            report[name] = {"status": "TOO_FEW_ENTRIES", "n": len(outcomes)}
            continue
        excursion_95 = float(np.quantile(excursions, 0.95))
        report[name] = {
            "n": len(outcomes),
            "liquidated_fraction": round(liquidated / len(outcomes), 3),
            "median_levered_outcome": round(float(np.median(outcomes)), 3),
            "excursion_95pct": round(excursion_95, 4),
            "max_leverage_95pct_survival": (
                round(0.5 / excursion_95, 2) if excursion_95 > 0 else None
            ),
            "small_sample_warning": len(outcomes) < 50,
        }
    currently_calm = bool(calm.iloc[-1]) if len(calm) else False
    return {
        "status": "DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER",
        "symbol": symbol.upper(),
        "as_of_session": str(pd.Timestamp(bars.index.max()).date()),
        "leverage_tested": leverage,
        "horizon_sessions": horizon,
        "close_out_rule": (
            f"intraday adjusted-low excursion >= {liq_threshold:.1%}"
            " (50% maintenance, clean fill assumed; real gaps are worse)"
        ),
        "conditions": report,
        "calm_regime_now": currently_calm,
        "calm_regime_definition": "close > 200-session SMA and 20-session vol < 30%",
        "caveats": [
            "OVERLAPPING_WINDOWS_DEPENDENT_SAMPLES",
            "CLEAN_LIQUIDATION_ASSUMED_GAPS_MAKE_TAILS_WORSE",
            "95PCT_PER_POSITION_SURVIVAL_STILL_IMPLIES_REGULAR_CLOSE_OUTS_OVER_YEARS",
            "ENTRY_RULES_MOVE_MEDIANS_NOT_PATH_RISK",
        ],
    }


def entry_state(frame: Any, symbol: str) -> dict[str, Any]:
    """Current regime and dip state for one symbol, latest session.

    Same formulas the leverage study used: calm = close above the
    200-session SMA with 20-session vol under 30%; dip = RSI(2) < 10, or
    IBS < 0.2, or three consecutive down closes. Values describe the LAST
    completed session only — path states are unknowable in advance.
    """

    import numpy as np

    bars = (
        frame.loc[frame["symbol"] == symbol.upper()].set_index("session").sort_index()
    )
    close = bars["close"].astype(float)
    low = bars["low"].astype(float)
    high = bars["high"].astype(float)
    returns = bars["adjusted_close"].astype(float).pct_change()
    sma200 = close.rolling(200, min_periods=200).mean()
    vol20 = returns.rolling(20, min_periods=20).std() * np.sqrt(252.0)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=0.5, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=0.5, adjust=False).mean()
    rsi2 = float((100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))).iloc[-1])
    span = float(high.iloc[-1] - low.iloc[-1])
    ibs = float((close.iloc[-1] - low.iloc[-1]) / span) if span > 0 else 0.5
    down3 = bool((returns.iloc[-3:] < 0).all())
    above_ma200 = bool(close.iloc[-1] > sma200.iloc[-1])
    vol_now = float(vol20.iloc[-1])
    dip = bool(rsi2 < 10.0 or ibs < 0.2 or down3)
    return {
        "as_of_session": str(bars.index.max().date()),
        "trend_above_ma200": above_ma200,
        "vol20_annualized": round(vol_now, 3),
        "calm_regime": bool(above_ma200 and vol_now < 0.30),
        "rsi2": round(rsi2, 1),
        "ibs": round(ibs, 2),
        "three_down_days": down3,
        "dip": dip,
    }


def _earnings_estimate(symbol: str, root: Path) -> dict[str, Any]:
    """Estimated next earnings window from the EDGAR 8-K 2.02 cadence.

    An ESTIMATE, stamped as such — the true date is set by the company.
    """

    import numpy as np
    import pandas as pd

    earnings_dir = root / "artifacts" / "earnings"
    tables = [
        pd.read_parquet(path, columns=["symbol", "acceptance"])
        for path in (
            earnings_dir / "announcements.parquet",
            earnings_dir / "live-announcements.parquet",  # nightly append-only
        )
        if path.is_file()
    ]
    if not tables:
        return _not_available(
            "no EDGAR announcements crawl; run"
            " `python -m edgestack.data.edgar_earnings`"
        )
    table = pd.concat(tables, ignore_index=True)
    mine = table.loc[table["symbol"] == symbol.upper()]
    if len(mine) < 4:
        return _not_available(f"too few 8-K 2.02 filings for {symbol.upper()}")
    dates = (
        pd.to_datetime(mine["acceptance"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.normalize()
        .dt.tz_localize(None)
        .sort_values()
        .drop_duplicates()
    )
    gaps = dates.diff().dropna().dt.days
    quarterly = gaps[(gaps > 60) & (gaps < 130)]
    median_gap = float(np.median(quarterly)) if len(quarterly) else 91.0
    estimated = dates.iloc[-1] + pd.Timedelta(days=round(median_gap))
    return {
        "status": "EARNINGS_WINDOW_ESTIMATED_NOT_CONFIRMED",
        "last_announcement": str(dates.iloc[-1].date()),
        "median_gap_days": round(median_gap),
        "estimated_next": str(estimated.date()),
        "window_start": str((estimated - pd.Timedelta(days=7)).date()),
        "window_end": str((estimated + pd.Timedelta(days=7)).date()),
    }


@app.command("telegram-test")
def telegram_test_command() -> None:
    """Send one test push through the configured Telegram credentials."""

    def _build() -> dict[str, Any]:
        import os
        from datetime import UTC, datetime

        token = os.environ.get("EDGESTACK_TELEGRAM_TOKEN", "").strip()
        chat = os.environ.get("EDGESTACK_TELEGRAM_CHAT", "").strip()
        if not token or not chat:
            return {
                "status": "SKIPPED_NO_CREDENTIALS",
                "setup": [
                    "1. In Telegram, message @BotFather -> /newbot -> copy the token",
                    "2. Message your new bot once (any text), then open"
                    " https://api.telegram.org/bot<TOKEN>/getUpdates and copy"
                    " your chat id",
                    '3. setx EDGESTACK_TELEGRAM_TOKEN "<token>"',
                    '4. setx EDGESTACK_TELEGRAM_CHAT "<chat id>"',
                    "5. Open a NEW terminal and re-run telegram-test",
                ],
            }
        from edgestack.live.notify import TelegramChannel
        from edgestack.models import AlertEvent

        now = datetime.now(UTC)
        event = AlertEvent(
            event_id=f"telegram-test-{now:%Y%m%d%H%M%S}",
            recommendation_id="telegram-test",
            revision=1,
            event_type="TEST",
            message=(
                "EdgeStack telegram-test: alerts are wired to this chat."
                " PAPER ONLY - not financial advice."
            ),
            created_at=now,
        )
        receipt = asyncio.run(
            TelegramChannel(token=token, chat_id=chat).send(event, event.event_id)
        )
        return {"status": "SENT", "receipt": str(receipt)}

    _emit(_guarded(_build))


@app.command("entry-check")
def entry_check_command(
    symbol: Annotated[str, typer.Argument(help="Ticker, e.g. EPAM.")],
    sessions: Annotated[int, typer.Option("--sessions", min=5, max=252)] = 42,
    threshold_bp: Annotated[float, typer.Option("--threshold-bp", min=0.0)] = 15.0,
    years: Annotated[int, typer.Option("--years", min=2, max=60)] = 20,
    root: Annotated[Path, typer.Option("--root", file_okay=False)] = Path("."),
) -> None:
    """Fused entry verdicts: alignment + regime + dip + estimated earnings."""

    def _build() -> dict[str, Any]:
        from edgestack.disclaimer import DISCLAIMER

        base = root.resolve()
        frame, warnings = _fetch_bars(symbol, years)
        report = _advise_report(
            symbol, years=years, sessions=sessions, buy_date=None, root=base
        )
        now = entry_state(frame, symbol)
        earnings = _guarded(lambda: _earnings_estimate(symbol, base))
        window_start = earnings.get("window_start")
        window_end = earnings.get("window_end")
        upcoming = []
        for row in report["alignment"]["calendar"]:
            session = str(row["session"])
            expected = float(row.get("expected_daily_bp") or 0.0)
            in_window = bool(
                window_start and window_end and window_start <= session <= window_end
            )
            if in_window:
                verdict = "CAUTION_EARNINGS"
            elif expected >= threshold_bp:
                verdict = "GOOD"
            else:
                verdict = "WAIT"
            upcoming.append(
                {
                    "session": session,
                    "weekday": row.get("weekday"),
                    "expected_daily_bp": round(expected, 2),
                    "win_score_0_100": row.get("win_score_0_100"),
                    "verdict": verdict,
                }
            )
        return {
            "status": "DIAGNOSTIC_NOT_A_VALIDATED_EDGE_NOT_AN_ORDER",
            "symbol": symbol.upper(),
            "now": now,
            "regime_note": (
                "calm regime — normal entry rules apply"
                if now["calm_regime"]
                else "NO_CALM_REGIME: below 200-DMA or elevated vol; entries"
                " here historically carry the widest downside paths"
            ),
            "earnings": earnings,
            "threshold_bp": threshold_bp,
            "upcoming": upcoming,
            "best_upcoming": [u for u in upcoming if u["verdict"] == "GOOD"][:5],
            "provenance_warnings": list(warnings),
            "disclaimer": DISCLAIMER,
        }

    _emit(_guarded(_build))


@app.command("leverage-check")
def leverage_check_command(
    symbol: Annotated[str, typer.Argument(help="Ticker, e.g. MU.")],
    leverage: Annotated[float, typer.Option("--leverage", min=1.0, max=25.0)] = 5.0,
    horizon: Annotated[int, typer.Option("--horizon", min=5, max=252)] = 60,
    years: Annotated[int, typer.Option("--years", min=2, max=60)] = 20,
) -> None:
    """Liquidation risk at a given leverage, under the known entry filters."""

    def _build() -> dict[str, Any]:
        from edgestack.disclaimer import DISCLAIMER

        frame, warnings = _fetch_bars(symbol, years)
        payload = _leverage_assessment(
            frame, symbol, leverage=leverage, horizon=horizon
        )
        payload["provenance_warnings"] = list(warnings)
        payload["disclaimer"] = DISCLAIMER
        return payload

    _emit(_guarded(_build))


if __name__ == "__main__":
    app()
