"""EdgeStack command-line interface with hard campaign-gate ordering."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from edgestack.config import load_config
from edgestack.disclaimer import DISCLAIMER
from edgestack.live.demo import run as run_demo
from edgestack.logging import configure_logging
from edgestack.models import GateStatus

app = typer.Typer(
    name="edgestack",
    no_args_is_help=True,
    help="Reproducible statistical-edge research and paper-assistant engine.",
    epilog=DISCLAIMER,
)
console = Console()


@app.callback()
def main(
    json_logs: Annotated[
        bool, typer.Option("--json-logs", help="Emit structured JSON logs.")
    ] = False,
) -> None:
    """Print the mandatory disclosure and configure structured logging."""

    configure_logging(json_output=json_logs)
    console.print(f"[bold red]{DISCLAIMER}[/bold red]")


@app.command()
def ingest(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            metavar="YYYY-MM-DD",
            help="Last market date considered by the campaign.",
        ),
    ] = None,
    campaign_id: Annotated[str | None, typer.Option("--campaign-id")] = None,
) -> None:
    """Download/cache bars, run data QA, and create a campaign snapshot."""

    from edgestack.pipeline.runner import CampaignRunner

    resolved = load_config(config)
    runner = CampaignRunner.create(
        resolved,
        campaign_id=campaign_id,
        as_of=_parse_iso_date(as_of, "--as-of"),
    )
    result = runner.ingest()
    console.print(
        f"Campaign [bold]{runner.campaign_id}[/bold]: data gate {result.status.value}"
    )
    console.print(result.summary)
    _exit_on_failed_gate(result.status)


@app.command()
def replicate(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
) -> None:
    """Run the frozen known-effects pipeline-correctness suite."""

    _run_phase("replicate", campaign, config)


@app.command()
def discover(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
) -> None:
    """Enumerate and backtest the preregistered real/placebo grid."""

    _run_phase("discover", campaign, config)


@app.command("reversal-study")
def reversal_study(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/reversal-study.yaml"),
    run_ml: Annotated[
        bool,
        typer.Option(
            "--run-ml",
            help="Also run purged ridge/elastic-net/XGBoost rank diagnostics.",
        ),
    ] = False,
    gpu: Annotated[
        bool,
        typer.Option(
            "--gpu",
            help="Assign XGBoost trials across the configured independent CUDA devices.",
        ),
    ] = False,
) -> None:
    """Run the opt-in, non-holdout top-K and residual-reversal study."""

    from edgestack.pipeline.runner import CampaignRunner

    if gpu and not run_ml:
        raise typer.BadParameter("--gpu requires --run-ml")
    runner = CampaignRunner.open(load_config(config), campaign)
    result = runner.reversal_research(run_ml=run_ml, use_gpu=gpu)
    console.print(f"reversal-study: {result.status.value} — {result.summary}")
    _exit_on_failed_gate(result.status)


@app.command()
def validate(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
) -> None:
    """Run walk-forward, CPCV/PBO, stability, decay, and confirmation."""

    _run_phase("validate", campaign, config)


@app.command("report")
def report_command(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
    provisional: Annotated[bool, typer.Option("--provisional")] = False,
    finalize_holdout: Annotated[bool, typer.Option("--finalize-holdout")] = False,
) -> None:
    """Render the provisional report or consume/finalize the one-use holdout."""

    if provisional == finalize_holdout:
        raise typer.BadParameter(
            "choose exactly one of --provisional or --finalize-holdout"
        )
    phase = "finalize_holdout" if finalize_holdout else "report"
    _run_phase(phase, campaign, config)


@app.command()
def score(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/smoke.yaml"),
    freeze: Annotated[bool, typer.Option("--freeze")] = False,
) -> None:
    """Build the shrunk, clustered composite and freeze its full definition."""

    if not freeze:
        raise typer.BadParameter(
            "score requires --freeze to protect holdout governance"
        )
    _run_phase("score", campaign, config)


@app.command()
def live(
    campaign: Annotated[str, typer.Option("--campaign")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/full.yaml"),
    once: Annotated[
        bool, typer.Option("--once", help="Run one scan instead of scheduling.")
    ] = False,
    v2_database: Annotated[
        Path | None,
        typer.Option(
            "--v2-database",
            help="Loss-aware V2 forward ledger; bypasses V1 holdout access.",
        ),
    ] = None,
    marks: Annotated[
        Path | None,
        typer.Option(
            "--marks",
            exists=True,
            dir_okay=False,
            help="Recorded causal marks JSON for a V2 --once replay.",
        ),
    ] = None,
) -> None:
    """Start the paper-only scanner/monitor after final holdout promotion."""

    if v2_database is not None:
        if not once:
            raise typer.BadParameter("V2 forward marking currently requires --once")
        from edgestack.live.state import StateStore

        store = StateStore(v2_database)
        if marks is not None:
            payload = json.loads(marks.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise typer.BadParameter("--marks must contain a JSON list")
            for item in payload:
                store.record_paper_mark(
                    str(item["decision_id"]),
                    mark_at=datetime.fromisoformat(str(item["mark_at"])),
                    available_at=datetime.fromisoformat(str(item["available_at"])),
                    price=float(item["price"]),
                    causal_data_hash=str(item["causal_data_hash"]),
                )
        console.print_json(data=store.paper_scorecard())
        return

    from edgestack.pipeline.runner import CampaignRunner

    runner = CampaignRunner.open(load_config(config), campaign)
    result = runner.live(once=once)
    console.print(result)


@app.command("loss-aware-v2")
def loss_aware_v2(
    campaign_id: Annotated[str, typer.Option("--campaign-id")],
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/loss-aware-v2.yaml"),
    artifacts: Annotated[Path, typer.Option(file_okay=False)] = Path("artifacts"),
) -> None:
    """Create the isolated free-only V2 diagnostic and forward declaration."""

    from edgestack.v2.campaign import create_free_only_diagnostic

    output = create_free_only_diagnostic(
        artifacts, campaign_id=campaign_id, config_path=config
    )
    console.print(f"V2 diagnostic written to [bold]{output}[/bold]")
    console.print(
        "PIT membership, estimate vintages, and auction execution are "
        "DATA_UNAVAILABLE until hash-pinned entitled files are imported."
    )


@app.command()
def advise(
    symbol: Annotated[str, typer.Option("--symbol", help="Ticker, e.g. GLD, USO.")],
    years: Annotated[int, typer.Option("--years", min=2, max=60)] = 20,
    buy_date: Annotated[
        str | None,
        typer.Option(
            "--buy-date",
            metavar="YYYY-MM-DD",
            help="Rate this intended buy session against active conditions.",
        ),
    ] = None,
    buy_hour: Annotated[
        str | None,
        typer.Option(
            "--buy-hour",
            metavar="HH:MM",
            help="ET clock time to rate; only the auction anchors are "
            "measurable (open ~09:30, close 15:45-16:00).",
        ),
    ] = None,
    bars: Annotated[
        Path | None,
        typer.Option(
            "--bars",
            exists=True,
            dir_okay=False,
            help="Offline bars parquet (symbol/session/adjusted_close); "
            "skips the network fetch.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, help="Write the JSON report here."),
    ] = None,
) -> None:
    """Diagnostic per-instrument timing report: tailwinds, headwinds, windows.

    NOT a validated edge and NOT an order. Daily bars only: execution anchors
    are the opening/closing auctions; news is DATA_UNAVAILABLE by design.
    """

    import asyncio
    from datetime import date as date_type
    from datetime import timedelta

    import pandas as pd

    from edgestack.advisor import advise as build_report

    warnings: tuple[str, ...] = ()
    if bars is not None:
        frame = pd.read_parquet(bars)
    else:
        from edgestack.data.sources import (
            FallbackDailyBarSource,
            StooqDailyBarSource,
            YahooDailyBarSource,
            bars_to_frame,
        )
        from edgestack.models import AssetKey, BarRequest

        async def _fetch() -> tuple[pd.DataFrame, tuple[str, ...]]:
            chain = FallbackDailyBarSource(
                (StooqDailyBarSource(), YahooDailyBarSource())
            )
            batch = await chain.fetch_bars(
                BarRequest(
                    AssetKey(symbol.upper()),
                    date_type.today() - timedelta(days=365 * years),
                    date_type.today(),
                    adjusted=True,
                )
            )
            return bars_to_frame(batch), tuple(batch.warnings)

        frame, warnings = asyncio.run(_fetch())
    report = build_report(
        frame,
        symbol=symbol.upper(),
        buy_session=_parse_iso_date(buy_date, "--buy-date"),
        buy_hour=buy_hour,
        provenance_warnings=warnings,
        root=Path.cwd(),
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        console.print(f"Advisor report written to [bold]{output}[/bold]")
    console.print_json(data=report)


@app.command("tailwind-calendar")
def tailwind_calendar(
    symbol: Annotated[str, typer.Option("--symbol", help="Ticker, e.g. SPY, QQQ, GLD, USO.")],
    sessions: Annotated[int, typer.Option("--sessions", min=5, max=252)] = 63,
    years: Annotated[int, typer.Option("--years", min=2, max=60)] = 20,
    bars: Annotated[
        Path | None,
        typer.Option("--bars", exists=True, dir_okay=False, help="Offline bars parquet."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, help="Write the JSON calendar here."),
    ] = None,
) -> None:
    """Forward tailwind calendar with per-session win scores and anchors.

    Daily granularity plus the two auction anchors; hourly and 15-minute
    calendars are DATA_UNAVAILABLE by construction, never estimated.
    """

    import asyncio
    from datetime import date as date_type
    from datetime import timedelta

    import pandas as pd

    from edgestack.advisor import advise as build_report

    warnings: tuple[str, ...] = ()
    if bars is not None:
        frame = pd.read_parquet(bars)
    else:
        from edgestack.data.sources import (
            FallbackDailyBarSource,
            StooqDailyBarSource,
            YahooDailyBarSource,
            bars_to_frame,
        )
        from edgestack.models import AssetKey, BarRequest

        async def _fetch() -> tuple[pd.DataFrame, tuple[str, ...]]:
            chain = FallbackDailyBarSource(
                (StooqDailyBarSource(), YahooDailyBarSource())
            )
            batch = await chain.fetch_bars(
                BarRequest(
                    AssetKey(symbol.upper()),
                    date_type.today() - timedelta(days=365 * years),
                    date_type.today(),
                    adjusted=True,
                )
            )
            return bars_to_frame(batch), tuple(batch.warnings)

        frame, warnings = asyncio.run(_fetch())
    report = build_report(
        frame,
        symbol=symbol.upper(),
        scan_sessions=sessions,
        provenance_warnings=warnings,
        root=Path.cwd(),
    )
    calendar = {
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
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(calendar, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        console.print(f"Tailwind calendar written to [bold]{output}[/bold]")
    console.print_json(data=calendar)


@app.command("universe-pit-audit")
def universe_pit_audit(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False)
    ] = Path("configs/full.yaml"),
    start: Annotated[
        str, typer.Option("--start", metavar="YYYY-MM-DD")
    ] = "1996-01-01",
    end: Annotated[str | None, typer.Option("--end", metavar="YYYY-MM-DD")] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", dir_okay=False, help="Write the JSON report here."),
    ] = None,
) -> None:
    """Audit delisted-name price coverage for PIT universe reconstruction.

    Crosses the Wikipedia S&P 500 change log against the hash-pinned Stooq
    bulk archive member index. Report-only; decides how far back a
    PIT_APPROXIMATION universe is honest.
    """

    import asyncio
    from datetime import date as date_type

    from edgestack.data.pit_audit import summarize_pit_coverage
    from edgestack.data.sources import StooqBulkArchiveDailyBarSource
    from edgestack.data.universe import WikipediaSP500UniverseSource

    resolved = load_config(config)
    providers = resolved.data.providers
    if providers.stooq_bulk_archive is None or providers.stooq_bulk_sha256 is None:
        raise typer.BadParameter(
            "the config must pin data.providers.stooq_bulk_archive and "
            "stooq_bulk_sha256",
            param_hint="--config",
        )
    start_date = _parse_iso_date(start, "--start")
    end_date = _parse_iso_date(end, "--end") or date_type.today()
    assert start_date is not None
    bulk = StooqBulkArchiveDailyBarSource(
        providers.stooq_bulk_archive,
        expected_sha256=providers.stooq_bulk_sha256,
    )
    changes = asyncio.run(
        WikipediaSP500UniverseSource().membership_changes(start_date, end_date)
    )
    report = summarize_pit_coverage(
        changes, bulk._members.keys(), start=start_date, end=end_date
    )
    report["archive_sha256"] = bulk.archive_sha256
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        console.print(f"PIT coverage report written to [bold]{output}[/bold]")
    console.print_json(data=report)


@app.command("universe-bias-delta")
def universe_bias_delta_command(
    campaign: Annotated[str, typer.Option("--campaign")],
    artifacts: Annotated[
        Path, typer.Option(file_okay=False, help="EdgeStack artifact directory.")
    ] = Path("artifacts"),
) -> None:
    """Measure the survivorship-bias inflation of two replicated signals.

    Report-only: reruns gross reversal/momentum decile streams on the
    campaign's persisted bars with and without the PIT membership mask.
    """

    import pandas as pd

    from edgestack.data.pit_audit import universe_bias_delta

    campaign_root = artifacts / "campaigns" / campaign
    bars_path = campaign_root / "data" / "bars.parquet"
    universe_path = campaign_root / "data" / "universe.parquet"
    for path in (bars_path, universe_path):
        if not path.is_file():
            raise typer.BadParameter(
                f"missing campaign artifact: {path}", param_hint="--campaign"
            )
    report = universe_bias_delta(
        pd.read_parquet(bars_path), pd.read_parquet(universe_path)
    )
    output = campaign_root / "diagnostics" / "universe_bias_delta.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    console.print(f"Bias-delta report written to [bold]{output}[/bold]")
    console.print_json(data=report)


@app.command("holdout-diagnostic")
def holdout_diagnostic(
    campaign: Annotated[str, typer.Option("--campaign")],
    artifacts: Annotated[
        Path, typer.Option(file_okay=False, help="EdgeStack artifact directory.")
    ] = Path("artifacts"),
) -> None:
    """Report whether a sealed holdout result would pass the CI_V2 evaluator.

    Reads only the persisted result document; the sealed verdict never changes.
    """

    from edgestack.pipeline.holdout import retro_ci_diagnostic

    result_path = artifacts / "campaigns" / campaign / "holdout" / "result.json"
    if not result_path.is_file():
        raise typer.BadParameter(
            f"no sealed holdout result at {result_path}", param_hint="--campaign"
        )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    console.print_json(data=retro_ci_diagnostic(payload))


@app.command("paper-scorecard")
def paper_scorecard(
    database: Annotated[Path, typer.Option("--database", exists=True, dir_okay=False)],
) -> None:
    """Replay the V2 paper scorecard without network access or backfills."""

    from edgestack.live.state import StateStore

    console.print_json(data=StateStore(database).paper_scorecard())


@app.command("live-demo")
def live_demo(
    database: Annotated[Path, typer.Option("--database")] = Path(
        "artifacts/live_demo.sqlite"
    ),
) -> None:
    """Run an accelerated recorded day with a forced restart."""

    counts = run_demo(database)
    console.print(counts)
    if counts.get("sent") != counts.get("receiver_unique"):
        raise typer.Exit(1)


@app.command("mobile-api")
def mobile_api(
    host: Annotated[
        str,
        typer.Option(help="Bind address; use 0.0.0.0 only on a trusted network."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
    token: Annotated[
        str | None,
        typer.Option(
            envvar="EDGESTACK_MOBILE_TOKEN",
            help="Bearer token. Required unless --demo is used.",
        ),
    ] = None,
    campaign: Annotated[
        str | None,
        typer.Option(
            help="Targeted campaign whose sealed mobile artifacts are served."
        ),
    ] = None,
    artifacts: Annotated[
        Path, typer.Option(file_okay=False, help="EdgeStack artifact directory.")
    ] = Path("artifacts"),
    demo: Annotated[
        bool,
        typer.Option(help="Serve packaged offline demonstration data."),
    ] = False,
) -> None:
    """Serve the read-only Android companion API; never accepts orders."""

    import uvicorn

    from edgestack.mobile.api import create_mobile_app

    if not demo and (token is None or len(token) < 24):
        raise typer.BadParameter(
            "a bearer token of at least 24 characters is required outside demo mode",
            param_hint="--token / EDGESTACK_MOBILE_TOKEN",
        )
    application = create_mobile_app(
        artifact_root=artifacts,
        campaign_id=campaign,
        bearer_token=token,
        demo=demo,
    )
    uvicorn.run(application, host=host, port=port, access_log=False)


def _run_phase(phase: str, campaign: str, config: Path) -> None:
    from edgestack.pipeline.runner import CampaignRunner

    runner = CampaignRunner.open(load_config(config), campaign)
    method = getattr(runner, phase)
    result = method()
    if hasattr(result, "status"):
        console.print(f"{phase}: {result.status.value} — {result.summary}")
        _exit_on_failed_gate(result.status)
    else:
        console.print(result)


def _exit_on_failed_gate(status: GateStatus) -> None:
    """Return a failing process status after a persisted gate has been reported."""

    if status in {GateStatus.FAIL, GateStatus.BLOCKED}:
        raise typer.Exit(code=1)


def _parse_iso_date(value: str | None, option_name: str) -> date | None:
    """Parse an ISO calendar date without relying on Typer date conversion.

    Typer/Click does not expose a native ``datetime.date`` parameter type. Keeping
    the command-line representation as text also makes the accepted format
    explicit and produces a stable user-facing error on every supported Typer
    release.
    """

    if value is None:
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise typer.BadParameter(
            "expected an ISO date in YYYY-MM-DD format",
            param_hint=option_name,
        ) from error
    if parsed.isoformat() != value:
        raise typer.BadParameter(
            "expected an ISO date in YYYY-MM-DD format",
            param_hint=option_name,
        )
    return parsed


if __name__ == "__main__":
    app()
