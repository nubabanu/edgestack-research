"""Causal cross-sectional features and net residual labels for ML research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from edgestack.config import ReversalResearchConfig
from edgestack.features.reversal import reversal_signal_set

if TYPE_CHECKING:
    from edgestack.pipeline.research import PreparedResearch


@dataclass(frozen=True, slots=True)
class CrossSectionalDataset:
    """Long-form date/symbol observations with explicit label horizons."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    long_target: str
    short_target: str
    group_column: str
    bias_tier: str
    event_features_available: bool
    intraday_decision_features_available: bool

    def model_frame(self, *, side: str, candidates_only: bool = False) -> pd.DataFrame:
        """Return a deterministic side-specific view sorted by ranking group."""

        normalized = side.upper()
        if normalized not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        output = self.frame
        if candidates_only:
            output = output.loc[output[f"candidate_{normalized.lower()}"]]
        target = self.long_target if normalized == "LONG" else self.short_target
        columns = (
            "date",
            "symbol",
            self.group_column,
            *self.feature_columns,
            target,
            "label_end",
        )
        return output.loc[:, columns].sort_values(
            [self.group_column, "symbol"], kind="stable"
        )


def _forward_compound(
    daily_returns: pd.DataFrame | pd.Series,
    *,
    execution_delay: int,
    holding_sessions: int,
) -> pd.DataFrame | pd.Series:
    result: pd.DataFrame | pd.Series = daily_returns.shift(-(execution_delay + 1)) * 0.0
    result = result + 1.0
    for offset in range(execution_delay + 1, execution_delay + holding_sessions + 1):
        result = result * (1.0 + daily_returns.shift(-offset))
    return result - 1.0


def _atr_percent(prepared: PreparedResearch, window: int = 14) -> pd.DataFrame:
    previous = prepared.close.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce(
            [
                (prepared.high - prepared.low).abs().to_numpy(dtype=float),
                (prepared.high - previous).abs().to_numpy(dtype=float),
                (prepared.low - previous).abs().to_numpy(dtype=float),
            ]
        ),
        index=prepared.close.index,
        columns=prepared.close.columns,
    )
    return true_range.rolling(window, min_periods=window).mean().div(prepared.close)


def _stack(values: pd.DataFrame, name: str) -> pd.Series:
    result = cast(pd.Series, values.stack(dropna=False))
    result.index.names = ["date", "symbol"]
    return result.rename(name)


def build_cross_sectional_dataset(
    prepared: PreparedResearch,
    config: ReversalResearchConfig,
    *,
    membership: pd.DataFrame | None = None,
    estimated_round_trip_cost_bps: float = 8.0,
    short_borrow_annual: float = 0.003,
) -> CrossSectionalDataset:
    """Build close-available predictors and next-close-to-horizon net labels.

    The target begins after the next-session closing fill.  Consequently a row
    dated ``t`` never contains any component of its own future label.  Event and
    15:45 decision features remain explicitly unavailable until timestamped
    historical inputs are supplied; they are not reconstructed from final bars.
    """

    if estimated_round_trip_cost_bps < 0.0 or short_borrow_annual < 0.0:
        raise ValueError("cost and borrow assumptions cannot be negative")
    close = prepared.close
    if membership is not None:
        membership_mask = membership.reindex(index=close.index, columns=close.columns)
        if membership_mask.isna().any(axis=None):
            raise ValueError("point-in-time membership must cover the complete panel")
        membership_mask = membership_mask.astype(bool)
    else:
        membership_mask = None
    equity = pd.Series(
        [asset_type == "equity" for asset_type in prepared.asset_types],
        index=close.columns,
        dtype=bool,
    )
    eligible = (
        pd.DataFrame(
            np.broadcast_to(equity.to_numpy(), close.shape),
            index=close.index,
            columns=close.columns,
        )
        & close.notna()
    )
    if membership_mask is not None:
        eligible &= membership_mask
    spy = next(
        (column for column in close.columns if str(column).upper() == "SPY"), None
    )
    market_returns = (
        prepared.close_returns[spy]
        if spy is not None
        else prepared.close_returns.where(eligible).mean(axis=1)
    )
    signals = reversal_signal_set(
        close,
        prepared.sector_by_symbol,
        market_returns=market_returns,
        membership=membership_mask,
        lookback=config.lookback_sessions,
        beta_window=config.beta_window,
        beta_min_observations=config.beta_min_observations,
        residual_vol_window=config.residual_vol_window,
    )
    features: dict[str, pd.DataFrame] = {}
    for horizon in (1, 2, 3, 5, 10, 20):
        features[f"return_{horizon}d"] = close.div(close.shift(horizon)) - 1.0
    features.update(
        {
            "reversal_raw": signals.raw,
            "reversal_sector_neutral": signals.sector_neutral,
            "reversal_market_sector_residual": signals.market_sector_residual,
            "market_beta": signals.market_beta,
            "sector_beta": signals.sector_beta,
            "residual_volatility_20d": signals.residual_volatility,
            "atr_14_pct": _atr_percent(prepared),
            "realized_volatility_20d": prepared.close_returns.rolling(
                20, min_periods=20
            ).std(ddof=1),
            "abnormal_volume_20d": prepared.volume.div(
                prepared.volume.rolling(20, min_periods=20).mean().shift(1)
            ),
            "overnight_1d": prepared.overnight_returns,
            "intraday_1d": prepared.intraday_returns,
            "gap_fraction_asof_close": (prepared.open - close.shift(1))
            .abs()
            .div((close - close.shift(1)).abs().clip(lower=1e-12)),
            "sector_relative_5d": signals.sector_neutral * -1.0,
            "log_adv_20d": pd.DataFrame(
                np.log1p(
                    close.mul(prepared.volume)
                    .rolling(20, min_periods=20)
                    .mean()
                    .shift(1)
                    .to_numpy()
                ),
                index=close.index,
                columns=close.columns,
            ),
        }
    )
    short_vol = prepared.close_returns.rolling(5, min_periods=5).std(ddof=1)
    long_vol = prepared.close_returns.rolling(20, min_periods=20).std(ddof=1).shift(1)
    features["volatility_expansion_5v20"] = short_vol.div(long_vol)
    market_regime = market_returns.rolling(20, min_periods=20).std(ddof=1).shift(1)
    features["market_volatility_20d"] = pd.DataFrame(
        np.broadcast_to(market_regime.to_numpy()[:, None], close.shape),
        index=close.index,
        columns=close.columns,
    )
    future_asset = (
        close.shift(-(config.holding_sessions + 1)).div(close.shift(-1)) - 1.0
    )
    future_market = _forward_compound(
        market_returns,
        execution_delay=1,
        holding_sessions=config.holding_sessions,
    )
    future_sector = _forward_compound(
        signals.sector_returns,
        execution_delay=1,
        holding_sessions=config.holding_sessions,
    )
    if not isinstance(future_market, pd.Series) or not isinstance(
        future_sector, pd.DataFrame
    ):
        raise TypeError("forward target compounding returned unexpected types")
    residual_target = (
        future_asset
        - signals.market_beta.mul(future_market, axis=0)
        - signals.sector_beta.mul(future_sector)
    )
    trading_cost = estimated_round_trip_cost_bps / 10_000.0
    short_borrow = short_borrow_annual * config.holding_sessions / 365.0
    target_long = residual_target - trading_cost
    target_short = -residual_target - trading_cost - short_borrow
    raw_percentile = signals.raw.rank(axis=1, pct=True, method="average")
    candidate_long = raw_percentile > 0.90
    candidate_short = raw_percentile <= 0.10
    pieces = [_stack(values.where(eligible), name) for name, values in features.items()]
    pieces.extend(
        [
            _stack(target_long.where(eligible), "target_long_net_residual"),
            _stack(target_short.where(eligible), "target_short_net_residual"),
            _stack(candidate_long.where(eligible, False), "candidate_long"),
            _stack(candidate_short.where(eligible, False), "candidate_short"),
            _stack(eligible, "eligible"),
        ]
    )
    frame = pd.concat(pieces, axis=1).reset_index()
    frame = frame.loc[frame["eligible"].astype(bool)].drop(columns="eligible")
    label_end_by_date = pd.Series(
        close.index.to_series().shift(-(config.holding_sessions + 1)).to_numpy(),
        index=close.index,
    )
    frame["label_end"] = frame["date"].map(label_end_by_date)
    frame["group_id"] = pd.factorize(frame["date"], sort=True)[0]
    frame["sector"] = frame["symbol"].map(
        lambda symbol: prepared.sector_by_symbol.get(str(symbol), "UNKNOWN")
    )
    frame = frame.dropna(
        subset=["target_long_net_residual", "target_short_net_residual", "label_end"]
    )
    feature_columns = tuple(features)
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return CrossSectionalDataset(
        frame.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True),
        feature_columns,
        "target_long_net_residual",
        "target_short_net_residual",
        "group_id",
        "POINT_IN_TIME" if membership_mask is not None else "SURVIVORSHIP_BIASED",
        False,
        False,
    )
