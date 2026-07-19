from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from edgestack.data.calendars import NYSECalendar
from edgestack.edges import (
    high52_study,
    momentum_xs_study,
    preholiday_study,
    volshock_study,
)

_ETFS = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI"]


def _config(name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _panel(symbols: list[str], *, start: str, end: str, seed: int) -> dict[str, Any]:
    sessions = NYSECalendar().sessions(start, end)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, (len(sessions), len(symbols)))
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    adjusted = pd.DataFrame(prices, index=sessions, columns=symbols)
    volume = pd.DataFrame(
        rng.uniform(5e6, 2e7, (len(sessions), len(symbols))),
        index=sessions,
        columns=symbols,
    )
    asset_types = pd.Series(
        {symbol: ("etf" if symbol in _ETFS else "equity") for symbol in symbols}
    )
    return {
        "adjusted_close": adjusted,
        "close": adjusted.copy(),
        "open": adjusted.copy(),
        "volume": volume,
        "asset_types": asset_types,
    }


@pytest.mark.parametrize(
    "module, config_name, expected_trials",
    [
        (momentum_xs_study, "momentum-xs-study-v1.yaml", 3),
        (high52_study, "high52-study-v1.yaml", 2),
        (volshock_study, "volshock-study-v1.yaml", 2),
    ],
)
def test_cross_sectional_families_build_valid_streams(
    module: Any, config_name: str, expected_trials: int
) -> None:
    config = module._load_config(
        Path(__file__).resolve().parents[1] / "configs" / config_name
    )
    symbols = _ETFS + [f"EQ{i:03d}" for i in range(400)]
    panel = _panel(symbols, start="2019-01-02", end="2022-12-30", seed=11)
    gross, net, definitions, benchmark = module.build_streams(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    assert len(definitions) == expected_trials
    for trial_id, meta in definitions.items():
        assert meta["average_names"] >= 30
        assert meta["rebalances"] > 0
        both = gross[trial_id].notna() & net[trial_id].notna()
        assert (net[trial_id][both] <= gross[trial_id][both] + 1e-12).all()
    assert len(benchmark) == len(gross)


def test_momentum_feature_ignores_the_last_month() -> None:
    symbols = _ETFS + [f"EQ{i:03d}" for i in range(2)]
    panel = _panel(symbols, start="2019-01-02", end="2021-12-30", seed=12)
    feature = momentum_xs_study._momentum_feature(21, 252)(
        panel, [s for s in symbols if s.startswith("EQ")]
    )
    # Perturb the final 20 sessions: the feature at the last session must
    # not change, because the last month is skipped.
    perturbed = {
        key: (value.copy() if hasattr(value, "copy") else value)
        for key, value in panel.items()
    }
    perturbed["adjusted_close"].iloc[-20:, -1] *= 1.5
    feature_perturbed = momentum_xs_study._momentum_feature(21, 252)(
        perturbed, [s for s in symbols if s.startswith("EQ")]
    )
    assert np.isclose(
        feature.iloc[-1, -1], feature_perturbed.iloc[-1, -1], equal_nan=True
    )


def test_pre_holiday_mask_finds_real_holidays_not_weekends() -> None:
    sessions = NYSECalendar().sessions("2021-06-01", "2021-07-15")
    mask = preholiday_study.pre_holiday_mask(pd.DatetimeIndex(sessions))
    # 2021-07-05 (Monday) was the Independence Day holiday; the prior
    # session 2021-07-02 (Friday) is pre-holiday.
    assert bool(mask.loc["2021-07-02"])
    # An ordinary Friday is NOT pre-holiday (plain weekend gap).
    assert not bool(mask.loc["2021-06-11"])
    # The final session is never classified.
    assert not bool(mask.iloc[-1])


def test_pre_holiday_streams_expose_only_masked_sessions() -> None:
    config = preholiday_study._load_config(
        Path(__file__).resolve().parents[1] / "configs" / "preholiday-study-v1.yaml"
    )
    panel = _panel(_ETFS, start="2018-01-02", end="2022-12-30", seed=13)
    gross, net, definitions, _ = preholiday_study.build_streams(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    mask = preholiday_study.pre_holiday_mask(
        pd.DatetimeIndex(panel["adjusted_close"].index)
    )
    for trial_id, meta in definitions.items():
        active = gross[trial_id].dropna()
        assert meta["events"] == len(active)
        assert mask.loc[active.index].all()
        # Roughly nine holidays per year across five years.
        assert 35 <= meta["events"] <= 55
        both = gross[trial_id].notna() & net[trial_id].notna()
        assert (net[trial_id][both] <= gross[trial_id][both] + 1e-12).all()


@pytest.mark.parametrize(
    "config_name",
    [
        "momentum-xs-study-v1.yaml",
        "high52-study-v1.yaml",
        "volshock-study-v1.yaml",
        "preholiday-study-v1.yaml",
    ],
)
def test_round_b_configs_are_declared_and_forward_only(config_name: str) -> None:
    config = _config(config_name)
    assert config["status"] == "DECLARED_AWAITING_IMPLEMENTATION"
    assert config["holdout"]["policy"] == "FORWARD_ONLY"
    assert config["holdout"]["start"] == "2026-07-19"
