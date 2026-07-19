from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from edgestack.data.calendars import NYSECalendar
from edgestack.edges import lowvol_study, pairs_study, seasonal_study

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
    volume = pd.DataFrame(1e7, index=sessions, columns=symbols)
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


def test_seasonal_streams_hold_winter_and_flip_twice_a_year() -> None:
    config = _config("seasonal-study-v1.yaml")
    panel = _panel(_ETFS, start="2015-01-02", end="2023-12-29", seed=1)
    gross, net, definitions, benchmark = seasonal_study.build_streams(
        config, panel, pd.Timestamp("2024-01-01").date()
    )
    assert len(definitions) == 9
    for trial_id, meta in definitions.items():
        # Roughly half the year is in season; two flips per full year.
        assert 0.4 < meta["active_fraction"] < 0.6
        assert 15 <= meta["flips"] <= 19
        active = gross[trial_id].dropna()
        assert set(pd.DatetimeIndex(active.index).month) <= {11, 12, 1, 2, 3, 4, 5}
    assert (net.fillna(0.0) <= gross.fillna(0.0) + 1e-12).all().all()
    assert len(benchmark) == len(gross)


def test_lowvol_goes_flat_when_the_decile_is_too_small() -> None:
    config = _config("lowvol-study-v1.yaml")
    symbols = _ETFS + [f"EQ{i:03d}" for i in range(40)]  # decile of 40 is 4 < 30
    panel = _panel(symbols, start="2018-01-02", end="2022-12-30", seed=2)
    gross, net, definitions, _ = lowvol_study.build_streams(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    assert len(definitions) == 3
    assert gross.isna().all().all()
    assert (net.dropna(how="all").fillna(0.0) <= 0.0).all().all()  # costs only


def test_lowvol_builds_portfolios_with_enough_names() -> None:
    config = _config("lowvol-study-v1.yaml")
    symbols = _ETFS + [f"EQ{i:03d}" for i in range(400)]
    panel = _panel(symbols, start="2019-01-02", end="2022-12-30", seed=3)
    gross, net, definitions, _ = lowvol_study.build_streams(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    for trial_id, meta in definitions.items():
        assert meta["average_names"] >= 30
        assert meta["rebalances"] > 0
        assert gross[trial_id].notna().sum() > 200
        # Net never exceeds gross on sessions where both are defined.
        both = gross[trial_id].notna() & net[trial_id].notna()
        assert (net[trial_id][both] <= gross[trial_id][both] + 1e-12).all()


def test_pairs_trades_within_declared_cycles_and_charges_costs() -> None:
    config = _config("pairs-study-v1.yaml")
    panel = _panel(_ETFS, start="2018-01-02", end="2022-12-30", seed=4)
    gross, net, definitions, _ = pairs_study.build_streams(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    assert set(definitions) == {
        "pairs|pairs_top5_z20",
        "pairs|pairs_top5_z15",
        "pairs|pairs_top10_z20",
    }
    for trial_id, meta in definitions.items():
        assert meta["cycles"] >= 7
        # The looser threshold must trade at least as often as the tighter one.
        both = gross[trial_id].notna() & net[trial_id].notna()
        assert (net[trial_id][both] <= gross[trial_id][both] + 1e-12).all()
    assert (
        definitions["pairs|pairs_top5_z15"]["round_trips"]
        >= definitions["pairs|pairs_top5_z20"]["round_trips"]
    )
    # No exposure before the first trading period ends its formation window.
    first_trade_session = panel["adjusted_close"].index[252]
    for trial_id in definitions:
        active = gross[trial_id].dropna()
        if len(active):
            assert active.index.min() >= first_trade_session


def test_pairs_truncated_rebuild_matches_on_the_shared_window() -> None:
    config = _config("pairs-study-v1.yaml")
    panel = _panel(_ETFS, start="2018-01-02", end="2022-12-30", seed=5)
    _, net_full, _, _ = pairs_study.build_streams(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    _, net_trunc, _, _ = pairs_study.build_streams(
        config, panel, pd.Timestamp("2022-06-01").date()
    )
    compare_end = pd.Timestamp("2022-04-01")
    for trial_id in net_full.columns:
        full = net_full[trial_id].loc[net_full.index < compare_end]
        trunc = net_trunc[trial_id].reindex(full.index)
        assert np.allclose(
            full.to_numpy(dtype=float),
            trunc.to_numpy(dtype=float),
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "module, config_name",
    [
        (seasonal_study, "seasonal-study-v1.yaml"),
        (lowvol_study, "lowvol-study-v1.yaml"),
        (pairs_study, "pairs-study-v1.yaml"),
    ],
)
def test_configs_match_module_trial_counts(module: Any, config_name: str) -> None:
    config = module._load_config(
        Path(__file__).resolve().parents[1] / "configs" / config_name
    )
    assert config["status"] == "DECLARED_AWAITING_IMPLEMENTATION"
    assert config["holdout"]["policy"] == "FORWARD_ONLY"
