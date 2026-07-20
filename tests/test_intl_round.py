from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from edgestack.data.calendars import NYSECalendar
from edgestack.data.intl_panel import BENCHMARK, INSTRUMENTS
from edgestack.edges import seasonal_intl_study, tom_intl_study
from edgestack.edges.seasonal_study import build_streams as seasonal_build


def _config(name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _panel(*, start: str, end: str, seed: int) -> dict[str, Any]:
    symbols = [*INSTRUMENTS, BENCHMARK]
    sessions = NYSECalendar().sessions(start, end)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, (len(sessions), len(symbols)))
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    adjusted = pd.DataFrame(prices, index=sessions, columns=symbols)
    volume = pd.DataFrame(1e7, index=sessions, columns=symbols)
    return {
        "adjusted_close": adjusted,
        "close": adjusted.copy(),
        "open": adjusted.copy(),
        "volume": volume,
        "asset_types": pd.Series(dict.fromkeys(symbols, "etf")),
    }


def test_intl_configs_declare_the_frozen_rules() -> None:
    seasonal = _config("seasonal-intl-v1.yaml")
    tom = _config("tom-intl-v1.yaml")
    for config in (seasonal, tom):
        assert config["status"] == "DECLARED_AWAITING_IMPLEMENTATION"
        assert config["holdout"]["policy"] == "FORWARD_ONLY"
        assert config["declared_family"]["instruments"] == list(INSTRUMENTS)
        assert config["declared_family"]["real_trial_count"] == 12
    assert "halloween" in seasonal["declared_family"]["signals"]
    assert "turn_of_month_last1_first3" in tom["declared_family"]["signals"]


def test_seasonal_intl_reuses_the_frozen_us_rule_verbatim() -> None:
    config = _config("seasonal-intl-v1.yaml")
    panel = _panel(start="2015-01-02", end="2022-12-30", seed=21)
    gross, _net, definitions, _benchmark = seasonal_build(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    assert len(definitions) == 12
    for trial_id, meta in definitions.items():
        assert 0.4 < meta["active_fraction"] < 0.6
        active = gross[trial_id].dropna()
        assert set(pd.DatetimeIndex(active.index).month) <= {11, 12, 1, 2, 3, 4, 5}
    # The intl module's config loader accepts exactly this declaration.
    loaded = seasonal_intl_study._load_config(
        Path(__file__).resolve().parents[1] / "configs" / "seasonal-intl-v1.yaml"
    )
    assert loaded["campaign_id"].startswith("seasonal-intl-v1")


def test_tom_mask_is_last1_first3() -> None:
    sessions = pd.DatetimeIndex(NYSECalendar().sessions("2021-01-04", "2021-12-31"))
    mask = tom_intl_study.tom_exposure_mask(sessions)
    # March 2021: last session 03-31; first three of April: 04-01, 04-05, 04-06.
    for day in ("2021-03-31", "2021-04-01", "2021-04-05", "2021-04-06"):
        assert bool(mask.loc[day]), day
    for day in ("2021-03-30", "2021-04-07", "2021-03-15"):
        assert not bool(mask.loc[day]), day
    # Roughly 4 exposed sessions per completed month.
    assert 40 <= int(mask.sum()) <= 48


def test_tom_intl_streams_charge_two_fills_per_month(tmp_path: Path) -> None:
    config = _config("tom-intl-v1.yaml")
    panel = _panel(start="2018-01-02", end="2022-12-30", seed=22)
    gross, net, definitions, _ = tom_intl_study.build_streams(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    assert len(definitions) == 12
    mask = tom_intl_study.tom_exposure_mask(
        pd.DatetimeIndex(panel["adjusted_close"].index)
    )
    for trial_id, meta in definitions.items():
        active = gross[trial_id].dropna()
        assert mask.loc[active.index].all()
        # ~12 events per full year over five years.
        assert 55 <= meta["events"] <= 62
        both = gross[trial_id].notna() & net[trial_id].notna()
        assert (net[trial_id][both] <= gross[trial_id][both] + 1e-12).all()


def test_tom_intl_truncated_rebuild_matches_shared_window() -> None:
    config = _config("tom-intl-v1.yaml")
    panel = _panel(start="2018-01-02", end="2022-12-30", seed=23)
    _, net_full, _, _ = tom_intl_study.build_streams(
        config, panel, pd.Timestamp("2023-01-01").date()
    )
    _, net_trunc, _, _ = tom_intl_study.build_streams(
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
