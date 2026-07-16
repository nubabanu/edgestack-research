"""Selection, causality, validation, and experiment-ledger tests for reversal research."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from edgestack.config import ReversalResearchConfig
from edgestack.entrytiming.causal_filters import (
    DecisionSnapshot,
    freeze_loc_decision,
    simulate_loc_auction,
)
from edgestack.features.reversal import (
    leave_one_out_sector_returns,
    reversal_signal_set,
)
from edgestack.models import Direction, TimingVerdict
from edgestack.reversal import study as reversal_study
from edgestack.reversal.dataset import CrossSectionalDataset
from edgestack.reversal.models import (
    LinearRanker,
    ModelSpec,
    XGBoostMetaLabeler,
    XGBoostRanker,
    assigned_gpu_device,
)
from edgestack.reversal.portfolio import reversal_trial_specs, top_k_side_weights
from edgestack.reversal.validation import purged_panel_splits, ranking_diagnostics
from edgestack.storage.catalog import Catalog


def test_reversal_grid_declares_every_breadth_variant_and_side() -> None:
    config = ReversalResearchConfig(enabled=True)

    specs = reversal_trial_specs(config, point_in_time_universe=False)

    assert len(specs) == 3 * 5 * 2
    assert len({spec.hypothesis_id for spec in specs}) == len(specs)
    assert {int(spec.parameters["top_k"]) for spec in specs} == {3, 5, 10, 20, 50}
    assert {spec.direction for spec in specs} == {Direction.LONG, Direction.SHORT}
    assert all(spec.universe == "sp500_current" for spec in specs)


def test_top_k_portfolios_match_the_extreme_tail_actually_traded() -> None:
    signal = pd.DataFrame(
        [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
        index=[pd.Timestamp("2024-01-02")],
        columns=list("ABCDEF"),
    )

    long = top_k_side_weights(signal, top_k=2, direction=Direction.LONG)
    short = top_k_side_weights(signal, top_k=2, direction=Direction.SHORT)

    assert long.loc[:, ["A", "B"]].to_numpy().tolist() == [[0.5, 0.5]]
    assert float(long.loc[:, ["C", "D", "E", "F"]].abs().to_numpy().sum()) == 0.0
    assert short.loc[:, ["E", "F"]].to_numpy().tolist() == [[-0.5, -0.5]]
    assert float(short.loc[:, ["A", "B", "C", "D"]].abs().to_numpy().sum()) == 0.0


def test_residual_reversal_is_prefix_invariant_and_leave_one_out() -> None:
    dates = pd.bdate_range("2022-01-03", periods=90)
    base = np.arange(len(dates), dtype=float)
    changes = np.column_stack(
        [
            0.0005 + 0.004 * np.sin(base / 5.0),
            0.0002 + 0.003 * np.cos(base / 7.0),
            -0.0001 + 0.005 * np.sin(base / 9.0 + 0.4),
            0.0003 + 0.002 * np.cos(base / 4.0 + 0.2),
        ]
    )
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(changes, axis=0)),
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    sectors = {"A": "TECH", "B": "TECH", "C": "ENERGY", "D": "ENERGY"}
    cutoff = 65
    changed = close.copy()
    changed.iloc[cutoff + 1 :] *= np.array([3.0, 0.2, 5.0, 0.4])

    kwargs = {
        "lookback": 5,
        "beta_window": 20,
        "beta_min_observations": 10,
        "residual_vol_window": 10,
    }
    original = reversal_signal_set(close, sectors, **kwargs)
    mutated = reversal_signal_set(changed, sectors, **kwargs)
    for field in (
        "raw",
        "sector_neutral",
        "market_sector_residual",
        "market_beta",
        "sector_beta",
        "residual_volatility",
    ):
        pdt.assert_frame_equal(
            getattr(original, field).iloc[: cutoff + 1],
            getattr(mutated, field).iloc[: cutoff + 1],
            check_exact=True,
        )

    returns = close.pct_change(fill_method=None)
    sector_returns = leave_one_out_sector_returns(returns, sectors)
    pdt.assert_series_equal(
        sector_returns["A"], returns["B"], check_names=False, check_exact=True
    )
    pdt.assert_series_equal(
        sector_returns["C"], returns["D"], check_names=False, check_exact=True
    )


def _decision_snapshot(*, quote_minute: int = 44) -> DecisionSnapshot:
    eastern = ZoneInfo("America/New_York")
    return DecisionSnapshot(
        symbol="TEST",
        direction=Direction.LONG,
        signal_available_at=datetime(2024, 7, 14, 16, 1, tzinfo=eastern),
        quote_available_at=datetime(2024, 7, 15, 15, quote_minute, tzinfo=eastern),
        decision_time=datetime(2024, 7, 15, 15, 45, tzinfo=eastern),
        order_cutoff=datetime(2024, 7, 15, 15, 50, tzinfo=eastern),
        auction_time=datetime(2024, 7, 15, 16, 0, tzinfo=eastern),
        previous_close=100.0,
        session_open=97.0,
        decision_price=98.0,
        atr14=2.0,
        event_proximity_sessions=10,
        event_available_at=datetime(2024, 7, 15, 8, 0, tzinfo=eastern),
    )


def test_pre_cutoff_loc_decision_cannot_reselect_from_the_closing_price() -> None:
    snapshot = _decision_snapshot()
    frozen = freeze_loc_decision(snapshot)

    assert frozen.passed
    assert frozen.entry_plan.verdict is TimingVerdict.ACT_NOW
    assert frozen.entry_plan.limit_price == pytest.approx(98.5)
    fill = simulate_loc_auction(
        frozen, auction_price=98.4, auction_time=snapshot.auction_time
    )
    miss = simulate_loc_auction(
        frozen, auction_price=98.6, auction_time=snapshot.auction_time
    )
    assert fill.filled and fill.fill_price == pytest.approx(98.4)
    assert not miss.filled

    with pytest.raises(ValueError, match="quote was not available"):
        _decision_snapshot(quote_minute=46)


def test_purged_panel_folds_exclude_overlapping_labels() -> None:
    dates = pd.bdate_range("2014-01-02", "2023-12-29")
    frame = pd.DataFrame(
        {
            "date": dates,
            "label_end": pd.Series(dates).shift(-5),
        }
    ).dropna()

    folds = purged_panel_splits(frame, min_train_years=5, purge_sessions=5)

    assert folds
    for fold in folds:
        train = frame.iloc[fold.train_indices]
        test = frame.iloc[fold.test_indices]
        assert pd.Timestamp(train["label_end"].max()) < pd.Timestamp(test["date"].min())
        assert fold.purged_rows > 0


def test_rankers_and_side_specific_meta_model_share_grouped_diagnostics() -> None:
    pytest.importorskip("xgboost")
    dates = np.repeat(pd.bdate_range("2024-01-02", periods=8), 8)
    symbol = np.tile([f"S{value}" for value in range(8)], 8)
    feature = np.tile(np.linspace(-1.0, 1.0, 8), 8)
    target = feature + 0.01 * np.sin(np.arange(len(feature)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "group_id": np.repeat(np.arange(8), 8),
            "feature": feature,
            "target": target,
        }
    )

    linear = LinearRanker(kind="ridge", alpha=0.1).fit(
        frame, features=("feature",), target="target", group="group_id"
    )
    linear_prediction = linear.predict(frame, features=("feature",))
    evidence = ranking_diagnostics(target, linear_prediction, frame["date"], top_k=2)
    assert evidence.mean_ic > 0.95
    assert evidence.top_bottom_spread > 0.0

    ranker = XGBoostRanker(n_estimators=8, max_depth=2, learning_rate=0.2).fit(
        frame, features=("feature",), target="target", group="group_id"
    )
    assert np.isfinite(ranker.predict(frame, features=("feature",))).all()
    meta = XGBoostMetaLabeler(n_estimators=8, max_depth=2, learning_rate=0.2).fit(
        frame, features=("feature",), target="target"
    )
    probabilities = meta.predict_probability(frame, features=("feature",))
    assert bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())
    trial = "0" * 63 + "1"
    assert assigned_gpu_device(trial, (0, 1)) in {"cuda:0", "cuda:1"}


def test_shared_experiment_catalog_claims_once_and_seals_metrics(tmp_path) -> None:
    catalog = Catalog(tmp_path / "catalog.sqlite")
    spec = {"algorithm": "ridge", "side": "LONG", "alpha": 1.0}

    assert catalog.claim_experiment("study", "trial", spec, device="cpu")
    assert not catalog.claim_experiment("study", "trial", spec, device="cpu")
    catalog.complete_experiment("trial", {"mean_ic": 0.01})
    catalog.complete_experiment("trial", {"mean_ic": 0.01})
    with pytest.raises(RuntimeError, match="immutable"):
        catalog.complete_experiment("trial", {"mean_ic": 0.02})

    records = catalog.experiments("study")
    assert len(records) == 1
    assert records[0].status == "COMPLETE"
    assert records[0].metrics == {"mean_ic": 0.01}


def test_gpu_study_refuses_an_invisible_device_without_claiming_work(
    tmp_path, monkeypatch
) -> None:
    dataset = CrossSectionalDataset(
        pd.DataFrame(),
        ("feature",),
        "long_target",
        "short_target",
        "group_id",
        "SURVIVORSHIP_BIASED",
        False,
        False,
    )
    spec = ModelSpec("ridge", "LONG", ("feature",), {"alpha": 1.0}, 42)
    catalog = Catalog(tmp_path / "catalog.sqlite")
    monkeypatch.setattr(reversal_study, "visible_cuda_devices", lambda: ())

    with pytest.raises(RuntimeError, match="refusing silent CPU fallback"):
        reversal_study.run_model_study(
            dataset,
            (spec,),
            catalog=catalog,
            study_id="gpu-study",
            use_gpu=True,
            gpu_devices=(0, 1),
        )

    assert catalog.experiments("gpu-study") == ()


def test_model_study_is_resumable_and_bound_to_its_data_study(tmp_path) -> None:
    dates = pd.date_range("2018-01-31", periods=48, freq="ME")
    symbols = [f"S{value}" for value in range(6)]
    rows: list[dict[str, object]] = []
    for group_id, date in enumerate(dates):
        for symbol_id, symbol in enumerate(symbols):
            feature = (symbol_id - 2.5) / 2.5
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "group_id": group_id,
                    "feature": feature,
                    "long_target": feature + 0.01 * np.sin(group_id),
                    "short_target": -feature + 0.01 * np.cos(group_id),
                    "label_end": date + pd.offsets.MonthEnd(1),
                }
            )
    dataset = CrossSectionalDataset(
        pd.DataFrame(rows),
        ("feature",),
        "long_target",
        "short_target",
        "group_id",
        "POINT_IN_TIME",
        False,
        False,
    )
    spec = ModelSpec("ridge", "LONG", ("feature",), {"alpha": 0.1}, 42)
    catalog = Catalog(tmp_path / "catalog.sqlite")

    first = reversal_study.run_model_study(
        dataset,
        (spec,),
        catalog=catalog,
        study_id="snapshot-a",
        min_train_years=1,
        purge_sessions=1,
        diagnostic_top_k=2,
    )
    replay = reversal_study.run_model_study(
        dataset,
        (spec,),
        catalog=catalog,
        study_id="snapshot-a",
        min_train_years=1,
        purge_sessions=1,
        diagnostic_top_k=2,
    )
    second_snapshot = reversal_study.run_model_study(
        dataset,
        (spec,),
        catalog=catalog,
        study_id="snapshot-b",
        min_train_years=1,
        purge_sessions=1,
        diagnostic_top_k=2,
    )

    assert first.failed_trial_ids == ()
    assert replay.declared_trial_ids == first.declared_trial_ids
    assert second_snapshot.declared_trial_ids != first.declared_trial_ids
    assert (
        len(first.metrics) == len(replay.metrics) == len(second_snapshot.metrics) == 1
    )
    assert len(catalog.experiments("snapshot-a")) == 1
    assert len(catalog.experiments("snapshot-b")) == 1
