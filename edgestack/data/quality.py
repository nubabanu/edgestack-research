"""Causal market-data quality checks and immutable correction evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from edgestack.data.calendars import NYSECalendar


class Severity(StrEnum):
    """Quality finding severity."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One traceable data-quality observation."""

    code: str
    severity: Severity
    symbol: str
    message: str
    sessions: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InstrumentQuality:
    """Coverage metrics and issues for one instrument."""

    symbol: str
    eligible_start: date
    eligible_end: date
    expected_sessions: int
    observed_sessions: int
    missing_sessions: int
    missing_fraction: float
    zero_volume_sessions: int
    stale_runs: int
    causal_outliers: int
    unexplained_jumps: int
    issues: tuple[QualityIssue, ...]


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Rebased total-return comparison for one symbol/provider pair."""

    symbol: str
    source_a: str
    source_b: str
    common_sessions: int
    agreement_fraction: float
    tolerance: float
    max_relative_difference: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SurvivorshipAudit:
    """Coverage and mandatory bias stamp for an intended universe."""

    intended_assets: int
    available_assets: int
    missing_assets: tuple[str, ...]
    missing_fraction: float
    point_in_time: bool
    bias_tier: str
    warning: str | None


@dataclass(frozen=True, slots=True)
class QAReport:
    """Complete immutable evidence emitted before research begins."""

    created_at: datetime
    instruments: tuple[InstrumentQuality, ...]
    reconciliations: tuple[ReconciliationResult, ...] = ()
    survivorship: SurvivorshipAudit | None = None
    missing_bar_threshold: float = 0.001

    @property
    def aggregate_missing_fraction(self) -> float:
        """Missing eligible bars divided by all eligible instrument sessions."""

        expected = sum(item.expected_sessions for item in self.instruments)
        missing = sum(item.missing_sessions for item in self.instruments)
        return missing / expected if expected else 0.0

    @property
    def passed(self) -> bool:
        """Whether coverage and all supplied reconciliation gates pass."""

        return self.aggregate_missing_fraction < self.missing_bar_threshold and all(
            item.passed for item in self.reconciliations
        )


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    """A non-destructive, causal transformation of one research observation."""

    symbol: str
    session: str
    field: str
    original: float
    corrected: float
    reason: str
    method: str
    prior_observations: int


def audit_instrument(
    frame: pd.DataFrame,
    *,
    symbol: str | None = None,
    nyse: NYSECalendar | None = None,
    listing_date: date | None = None,
    delisting_date: date | None = None,
    stale_sessions: int = 3,
    outlier_sigma: float = 10.0,
) -> InstrumentQuality:
    """Run coverage, volume, stale, split, and causal outlier checks.

    Missing-bar eligibility begins at a verified listing date when supplied,
    otherwise at the first observation.  It ends at a verified delisting date when
    supplied, otherwise at the last observation, so pre-listing/post-delisting days
    never inflate the denominator.
    """

    required = {"session", "close", "volume"}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"quality frame missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("cannot audit an empty instrument frame")
    data = frame.copy(deep=True)
    data["session"] = pd.to_datetime(data["session"]).dt.normalize()
    data = data.sort_values("session", kind="stable").drop_duplicates(
        "session", keep=False
    )
    inferred_symbol = symbol or (
        str(frame["symbol"].iloc[0])
        if "symbol" in frame and not frame.empty
        else "UNKNOWN"
    )
    first = data["session"].iloc[0].date()
    last = data["session"].iloc[-1].date()
    eligible_start = max(first, listing_date) if listing_date else first
    eligible_end = min(last, delisting_date) if delisting_date else last
    if eligible_end < eligible_start:
        raise ValueError("verified listing/delisting dates exclude every observation")
    exchange = nyse or NYSECalendar()
    expected = exchange.sessions(eligible_start, eligible_end)
    observed = pd.DatetimeIndex(
        data.loc[
            data["session"].between(
                pd.Timestamp(eligible_start), pd.Timestamp(eligible_end)
            ),
            "session",
        ]
    )
    missing = expected.difference(observed)
    issues: list[QualityIssue] = []
    if len(missing):
        issues.append(
            QualityIssue(
                "MISSING_BAR",
                Severity.WARNING,
                inferred_symbol,
                f"{len(missing)} eligible NYSE sessions have no bar",
                tuple(item.date().isoformat() for item in missing),
            )
        )
    zero_volume = data.loc[data["volume"].fillna(0.0) <= 0, "session"]
    if len(zero_volume):
        issues.append(
            QualityIssue(
                "ZERO_VOLUME",
                Severity.WARNING,
                inferred_symbol,
                f"{len(zero_volume)} sessions have non-positive volume",
                tuple(item.date().isoformat() for item in zero_volume),
            )
        )
    stale = stale_price_runs(data, minimum_sessions=stale_sessions)
    for run in stale:
        issues.append(
            QualityIssue(
                "STALE_PRICE",
                Severity.WARNING,
                inferred_symbol,
                f"close repeated for {len(run)} consecutive sessions",
                tuple(item.date().isoformat() for item in run),
            )
        )
    outliers = causal_outlier_mask(data["close"], sigma=outlier_sigma)
    outlier_sessions = data.loc[outliers, "session"]
    if len(outlier_sessions):
        issues.append(
            QualityIssue(
                "CAUSAL_OUTLIER",
                Severity.WARNING,
                inferred_symbol,
                f"{len(outlier_sessions)} returns exceed the causal {outlier_sigma:g}-sigma bound",
                tuple(item.date().isoformat() for item in outlier_sessions),
            )
        )
    jump_issues = split_dividend_issues(data, symbol=inferred_symbol)
    issues.extend(jump_issues)
    return InstrumentQuality(
        inferred_symbol,
        eligible_start,
        eligible_end,
        len(expected),
        len(observed),
        len(missing),
        len(missing) / len(expected) if len(expected) else 0.0,
        len(zero_volume),
        len(stale),
        int(outliers.sum()),
        sum(issue.code == "UNEXPLAINED_JUMP" for issue in jump_issues),
        tuple(issues),
    )


def stale_price_runs(
    frame: pd.DataFrame, *, minimum_sessions: int = 3, price_column: str = "close"
) -> tuple[tuple[pd.Timestamp, ...], ...]:
    """Return runs of unchanged prices of at least ``minimum_sessions``."""

    if minimum_sessions < 2:
        raise ValueError("minimum_sessions must be at least two")
    if price_column not in frame or "session" not in frame:
        raise ValueError("frame must contain session and selected price column")
    ordered = frame.sort_values("session", kind="stable")
    prices = pd.to_numeric(ordered[price_column], errors="coerce")
    groups = prices.ne(prices.shift()).cumsum()
    runs: list[tuple[pd.Timestamp, ...]] = []
    for _, group in ordered.groupby(groups, sort=False):
        if len(group) >= minimum_sessions and pd.notna(group[price_column].iloc[0]):
            runs.append(tuple(pd.Timestamp(value) for value in group["session"]))
    return tuple(runs)


def causal_outlier_mask(
    prices: pd.Series,
    *,
    sigma: float = 10.0,
    minimum_history: int = 20,
) -> pd.Series:
    """Flag log returns against mean/volatility estimated from prior returns only."""

    if sigma <= 0 or minimum_history < 2:
        raise ValueError("sigma must be positive and minimum_history at least two")
    values = pd.to_numeric(prices, errors="coerce")
    returns = np.log(values).diff()
    prior_mean = returns.expanding(min_periods=minimum_history).mean().shift(1)
    prior_std = returns.expanding(min_periods=minimum_history).std(ddof=1).shift(1)
    z_score = (returns - prior_mean).div(prior_std.replace(0.0, np.nan))
    return cast(pd.Series, z_score.abs().gt(sigma).fillna(False))


def causal_winsorize_prices(
    frame: pd.DataFrame,
    *,
    symbol: str | None = None,
    price_column: str = "close",
    sigma: float = 10.0,
    minimum_history: int = 20,
) -> tuple[pd.DataFrame, tuple[CorrectionRecord, ...]]:
    """Add ``research_<price>`` using only prior returns and log corrections.

    Source fields are never changed.  Bounds are computed sequentially from the
    previously accepted/winsorized return history, preventing future observations
    from changing an earlier correction.
    """

    if "session" not in frame or price_column not in frame:
        raise ValueError("frame must contain session and selected price column")
    if sigma <= 0 or minimum_history < 2:
        raise ValueError("invalid causal winsorization parameters")
    data = frame.sort_values("session", kind="stable").copy(deep=True)
    output_column = f"research_{price_column}"
    values = pd.to_numeric(data[price_column], errors="raise").to_numpy(float)
    corrected = values.copy()
    accepted_returns: list[float] = []
    records: list[CorrectionRecord] = []
    instrument = symbol or (
        str(data["symbol"].iloc[0])
        if "symbol" in data and not data.empty
        else "UNKNOWN"
    )
    for index in range(1, len(values)):
        raw_return = math.log(values[index] / corrected[index - 1])
        accepted = raw_return
        if len(accepted_returns) >= minimum_history:
            mean = float(np.mean(accepted_returns))
            std = float(np.std(accepted_returns, ddof=1))
            if std > 0:
                lower, upper = mean - sigma * std, mean + sigma * std
                accepted = min(max(raw_return, lower), upper)
        corrected[index] = corrected[index - 1] * math.exp(accepted)
        accepted_returns.append(accepted)
        if accepted != raw_return:
            records.append(
                CorrectionRecord(
                    instrument,
                    pd.Timestamp(data["session"].iloc[index]).date().isoformat(),
                    price_column,
                    float(values[index]),
                    float(corrected[index]),
                    f"causal return exceeded {sigma:g}-sigma prior-history bound",
                    "sequential_log_return_winsorization",
                    len(accepted_returns) - 1,
                )
            )
    data[output_column] = corrected
    return data, tuple(records)


def split_dividend_issues(
    frame: pd.DataFrame,
    *,
    symbol: str = "UNKNOWN",
    jump_threshold: float = 0.40,
) -> tuple[QualityIssue, ...]:
    """Flag >40% raw moves not explained by split/action adjustment evidence."""

    if jump_threshold <= 0:
        raise ValueError("jump_threshold must be positive")
    required = {"session", "close"}
    if not required.issubset(frame):
        raise ValueError("frame must contain session and close")
    ordered = frame.sort_values("session", kind="stable")
    raw_return = pd.to_numeric(ordered["close"], errors="coerce").pct_change()
    adjusted_return = (
        pd.to_numeric(ordered["adjusted_close"], errors="coerce").pct_change()
        if "adjusted_close" in ordered
        else raw_return
    )
    split_factor = (
        pd.to_numeric(ordered["split_factor"], errors="coerce").fillna(1.0)
        if "split_factor" in ordered
        else pd.Series(1.0, index=ordered.index)
    )
    dividend = (
        pd.to_numeric(ordered["dividend"], errors="coerce").fillna(0.0)
        if "dividend" in ordered
        else pd.Series(0.0, index=ordered.index)
    )
    raw_jump = raw_return.abs() > jump_threshold
    explained = (
        split_factor.ne(1.0)
        | dividend.ne(0.0)
        | (raw_return.abs().sub(adjusted_return.abs()).abs() > 0.10)
    )
    issues: list[QualityIssue] = []
    for position in np.flatnonzero((raw_jump & ~explained).to_numpy()):
        session = pd.Timestamp(ordered.iloc[position]["session"])
        issues.append(
            QualityIssue(
                "UNEXPLAINED_JUMP",
                Severity.ERROR,
                symbol,
                f"raw close moved {raw_return.iloc[position]:+.1%} without action evidence",
                (session.date().isoformat(),),
                {"raw_return": float(raw_return.iloc[position])},
            )
        )
    return tuple(issues)


def reconcile_adjusted_series(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    symbol: str,
    source_a: str,
    source_b: str,
    tolerance: float = 0.005,
    required_fraction: float = 0.99,
    price_column: str = "close",
) -> ReconciliationResult:
    """Compare provider total-return indices after rebasing the common history."""

    if tolerance <= 0 or not 0 < required_fraction <= 1:
        raise ValueError("invalid reconciliation tolerances")
    required = {"session", price_column}
    if not required.issubset(frame_a) or not required.issubset(frame_b):
        raise ValueError("both frames require session and the selected price column")
    left = frame_a.loc[:, ["session", price_column]].copy()
    right = frame_b.loc[:, ["session", price_column]].copy()
    left["session"] = pd.to_datetime(left["session"]).dt.normalize()
    right["session"] = pd.to_datetime(right["session"]).dt.normalize()
    common = left.merge(
        right, on="session", suffixes=("_a", "_b"), validate="one_to_one"
    )
    common = common.dropna().sort_values("session", kind="stable")
    if common.empty:
        return ReconciliationResult(
            symbol, source_a, source_b, 0, 0.0, tolerance, math.inf, False
        )
    a = pd.to_numeric(common[f"{price_column}_a"], errors="coerce")
    b = pd.to_numeric(common[f"{price_column}_b"], errors="coerce")
    valid = (a > 0) & (b > 0)
    a, b = a[valid], b[valid]
    if a.empty:
        return ReconciliationResult(
            symbol, source_a, source_b, 0, 0.0, tolerance, math.inf, False
        )
    rebased_a = a / a.iloc[0]
    rebased_b = b / b.iloc[0]
    relative = rebased_a.sub(rebased_b).abs().div(rebased_b.abs())
    fraction = float((relative <= tolerance).mean())
    return ReconciliationResult(
        symbol,
        source_a,
        source_b,
        len(relative),
        fraction,
        tolerance,
        float(relative.max()),
        fraction >= required_fraction,
    )


def audit_survivorship(
    intended_symbols: Sequence[str],
    available_symbols: Sequence[str],
    *,
    point_in_time: bool,
) -> SurvivorshipAudit:
    """Calculate missing-universe coverage and the mandatory bias watermark."""

    intended = {symbol.upper() for symbol in intended_symbols}
    available = {symbol.upper() for symbol in available_symbols}
    missing = tuple(sorted(intended - available))
    count = len(intended)
    biased = not point_in_time
    warning = (
        "SURVIVORSHIP_BIASED: current constituents/present-day ticker histories "
        "were used; historical performance may be overstated."
        if biased
        else None
    )
    return SurvivorshipAudit(
        count,
        len(intended & available),
        missing,
        len(missing) / count if count else 0.0,
        point_in_time,
        "SURVIVORSHIP_BIASED" if biased else "POINT_IN_TIME",
        warning,
    )


def run_quality_audit(
    bars: pd.DataFrame,
    *,
    nyse: NYSECalendar | None = None,
    listing_dates: Mapping[str, date] | None = None,
    delisting_dates: Mapping[str, date] | None = None,
    stale_sessions: int = 3,
    outlier_sigma: float = 10.0,
    reconciliations: Sequence[ReconciliationResult] = (),
    survivorship: SurvivorshipAudit | None = None,
    missing_bar_threshold: float = 0.001,
) -> QAReport:
    """Audit every symbol in a canonical multi-instrument frame."""

    if "symbol" not in bars:
        raise ValueError("bars must contain symbol")
    exchange = nyse or NYSECalendar()
    instruments = tuple(
        audit_instrument(
            group,
            symbol=str(symbol),
            nyse=exchange,
            listing_date=(listing_dates or {}).get(str(symbol)),
            delisting_date=(delisting_dates or {}).get(str(symbol)),
            stale_sessions=stale_sessions,
            outlier_sigma=outlier_sigma,
        )
        for symbol, group in bars.groupby("symbol", sort=True)
    )
    return QAReport(
        datetime.now(UTC),
        instruments,
        tuple(reconciliations),
        survivorship,
        missing_bar_threshold,
    )


def write_qa_report(report: QAReport, path: str | Path) -> str:
    """Write deterministic JSON once and return its content hash."""

    payload = _jsonable(asdict(report))
    body = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _immutable_write(Path(path), body)
    return hashlib.sha256(body).hexdigest()


def write_correction_log(records: Sequence[CorrectionRecord], path: str | Path) -> str:
    """Persist an immutable, sorted JSON correction ledger and return its hash."""

    payload = [
        asdict(record)
        for record in sorted(
            records, key=lambda item: (item.symbol, item.session, item.field)
        )
    ]
    body = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _immutable_write(Path(path), body)
    return hashlib.sha256(body).hexdigest()


def _immutable_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise RuntimeError(f"refusing to overwrite immutable artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "CorrectionRecord",
    "InstrumentQuality",
    "QAReport",
    "QualityIssue",
    "ReconciliationResult",
    "Severity",
    "SurvivorshipAudit",
    "audit_instrument",
    "audit_survivorship",
    "causal_outlier_mask",
    "causal_winsorize_prices",
    "reconcile_adjusted_series",
    "run_quality_audit",
    "split_dividend_issues",
    "stale_price_runs",
    "write_correction_log",
    "write_qa_report",
]
