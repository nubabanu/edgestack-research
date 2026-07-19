"""Causal market-regime interaction evidence for finalist validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats  # type: ignore[import-untyped]

from edgestack.entrytiming.regime import TrendRegime, ma_regime
from edgestack.stats._types import FloatArray
from edgestack.stats.tests import hac_lag, hac_mean_test


@dataclass(frozen=True, slots=True)
class CausalTrendRegimes:
    """Lagged return-date labels and the latest observable trend state."""

    labels: pd.Series
    current_regime: str | None
    available: bool
    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class RegimeInteractionResult:
    """HAC interaction and active-regime evidence for one return stream."""

    available: bool
    active_regime: str | None
    current_regime: str | None
    active_observations: int
    inactive_observations: int
    active_mean: float | None
    inactive_mean: float | None
    active_t: float | None
    interaction_t: float | None
    p_value: float | None
    adjusted_p_value: float | None
    currently_active: bool
    source: str
    reason: str

    def with_adjusted_p(self, adjusted_p_value: float) -> RegimeInteractionResult:
        """Attach the global FDR result without mutating raw evidence."""

        if not 0.0 <= adjusted_p_value <= 1.0:
            raise ValueError("adjusted p-value must lie in [0, 1]")
        return replace(self, adjusted_p_value=float(adjusted_p_value))


def causal_spy_ma200_regimes(close: pd.DataFrame) -> CausalTrendRegimes:
    """Label each return date from the previous completed SPY MA(200) state.

    The unshifted final state is observable after the latest completed close and
    is retained only to decide whether a validated regime is currently active.
    It is never joined to the same date's return.
    """

    source = "SPY adjusted close / trailing MA200; return labels lagged one session"
    if "SPY" not in close.columns:
        return CausalTrendRegimes(
            pd.Series(TrendRegime.UNKNOWN, index=close.index, dtype="object"),
            None,
            False,
            source,
            "SPY is absent from PreparedResearch.close",
        )
    prices = pd.to_numeric(close["SPY"], errors="coerce")
    observed = ma_regime(prices, window=200)
    labels = observed.shift(1).fillna(TrendRegime.UNKNOWN).rename("trend_regime")
    known = observed.loc[observed != TrendRegime.UNKNOWN]
    if known.empty:
        return CausalTrendRegimes(
            labels,
            None,
            False,
            source,
            "fewer than 200 usable SPY sessions for a causal MA200 state",
        )
    return CausalTrendRegimes(
        labels,
        str(known.iloc[-1]),
        True,
        source,
        "available",
    )


def causal_realized_vol_terciles(
    prices: pd.Series,
    *,
    window: int = 21,
    breakpoint_end: pd.Timestamp | None = None,
) -> pd.Series:
    """Label each return date by the prior session's trailing realized-vol tercile.

    Volatility on date ``t`` uses close-to-close returns through ``t-1`` only.
    Tercile breakpoints come from sessions strictly before ``breakpoint_end``
    (typically the holdout start) so labels applied inside a sealed window use
    no information from that window; without a boundary the full sample sets
    the breakpoints and the labels are descriptive only.
    """

    values = pd.to_numeric(prices, errors="coerce")
    volatility = values.pct_change().rolling(window).std().shift(1)
    volatility = volatility.rename("volatility_regime")
    if breakpoint_end is not None:
        reference = volatility.loc[volatility.index < breakpoint_end]
    else:
        reference = volatility
    reference = reference.dropna()
    if len(reference) < 3 * window:
        return pd.Series("UNKNOWN", index=volatility.index, dtype="object").rename(
            "volatility_regime"
        )
    low, high = reference.quantile([1.0 / 3.0, 2.0 / 3.0])
    labels = pd.Series("UNKNOWN", index=volatility.index, dtype="object")
    known = volatility.notna()
    labels[known & (volatility <= low)] = "VOL_LOW"
    labels[known & (volatility > low) & (volatility <= high)] = "VOL_MID"
    labels[known & (volatility > high)] = "VOL_HIGH"
    return labels.rename("volatility_regime")


def trend_regime_interaction(
    returns: FloatArray | list[float],
    regimes: CausalTrendRegimes,
    *,
    holding_period: int = 1,
    minimum_observations: int = 100,
    minimum_per_regime: int = 2,
) -> RegimeInteractionResult:
    """Test an UP/DOWN mean interaction using a Newey-West OLS covariance.

    The higher-mean state is reported as the active regime only after the
    two-sided interaction is tested.  The caller must globally adjust the raw
    interaction p-value before it can qualify as ``REGIME_DEPENDENT``.
    """

    if holding_period < 1:
        raise ValueError("holding_period must be positive")
    if minimum_observations < 2 or minimum_per_regime < 2:
        raise ValueError("regime sample minima must be at least two")
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) != len(regimes.labels):
        raise ValueError("returns and regime labels must be aligned one-dimensional")
    if not regimes.available:
        return _unavailable(regimes, regimes.reason)
    label_values = regimes.labels.astype(str).to_numpy()
    eligible = np.isfinite(values) & np.isin(
        label_values, (TrendRegime.UP.value, TrendRegime.DOWN.value)
    )
    sample = values[eligible]
    labels = label_values[eligible]
    if sample.size < minimum_observations:
        return _unavailable(
            regimes,
            f"only {sample.size} known-regime observations; {minimum_observations} required",
        )
    up = sample[labels == TrendRegime.UP.value]
    down = sample[labels == TrendRegime.DOWN.value]
    if min(up.size, down.size) < minimum_per_regime:
        return _unavailable(
            regimes,
            "both UP and DOWN states need at least "
            f"{minimum_per_regime} observations",
        )
    up_mean = float(np.mean(up))
    down_mean = float(np.mean(down))
    if up_mean >= down_mean:
        active_name = TrendRegime.UP.value
        active = up
        inactive = down
    else:
        active_name = TrendRegime.DOWN.value
        active = down
        inactive = up
    active_dummy = (labels == active_name).astype(float)
    interaction_t, p_value = _hac_binary_interaction(
        sample,
        active_dummy,
        holding_period=holding_period,
    )
    active_test = hac_mean_test(active, holding_period=holding_period)
    return RegimeInteractionResult(
        True,
        active_name,
        regimes.current_regime,
        int(active.size),
        int(inactive.size),
        float(np.mean(active)),
        float(np.mean(inactive)),
        active_test.t_stat,
        interaction_t,
        p_value,
        None,
        regimes.current_regime == active_name,
        regimes.source,
        "raw interaction available; global FDR adjustment pending",
    )


def _hac_binary_interaction(
    returns: FloatArray,
    active_dummy: FloatArray,
    *,
    holding_period: int,
) -> tuple[float, float]:
    """Return HAC t and two-sided p for the active-state OLS coefficient."""

    n_observations = len(returns)
    design = np.column_stack((np.ones(n_observations), active_dummy))
    inverse_cross = np.linalg.inv(design.T @ design)
    coefficients = inverse_cross @ design.T @ returns
    residuals = returns - design @ coefficients
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    lags = hac_lag(n_observations, holding_period=holding_period)
    for lag in range(1, lags + 1):
        covariance = scores[lag:].T @ scores[:-lag]
        weight = 1.0 - lag / (lags + 1.0)
        meat += weight * (covariance + covariance.T)
    covariance_matrix = inverse_cross @ meat @ inverse_cross
    covariance_matrix *= n_observations / (n_observations - design.shape[1])
    variance = max(float(covariance_matrix[1, 1]), 0.0)
    difference = float(coefficients[1])
    if variance == 0.0:
        t_stat = math.copysign(math.inf, difference) if difference else 0.0
    else:
        t_stat = difference / math.sqrt(variance)
    p_value = float(
        2.0 * scipy_stats.t.sf(abs(t_stat), df=n_observations - design.shape[1])
    )
    return float(t_stat), p_value


def _unavailable(regimes: CausalTrendRegimes, reason: str) -> RegimeInteractionResult:
    return RegimeInteractionResult(
        False,
        None,
        regimes.current_regime,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        regimes.source,
        reason,
    )
