"""Crash-safe purged studies for side-specific rankers and meta-labelers."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from edgestack.reversal.dataset import CrossSectionalDataset
from edgestack.reversal.models import (
    LinearRanker,
    ModelSpec,
    XGBoostMetaLabeler,
    XGBoostRanker,
    assigned_gpu_device,
    visible_cuda_devices,
)
from edgestack.reversal.validation import purged_panel_splits, ranking_diagnostics
from edgestack.storage.catalog import Catalog


@dataclass(frozen=True, slots=True)
class ModelStudyResult:
    """All completed/failed declarations from one chronological model study."""

    study_id: str
    metrics: pd.DataFrame
    declared_trial_ids: tuple[str, ...]
    failed_trial_ids: tuple[str, ...]
    bias_tier: str


def default_model_specs(
    dataset: CrossSectionalDataset,
    *,
    seed: int = 42,
) -> tuple[ModelSpec, ...]:
    """Declare linear, LambdaMART, and meta-label trials for both sides."""

    specs: list[ModelSpec] = []
    for side in ("LONG", "SHORT"):
        specs.extend(
            [
                ModelSpec("ridge", side, dataset.feature_columns, {"alpha": 1.0}, seed),
                ModelSpec(
                    "elastic_net",
                    side,
                    dataset.feature_columns,
                    {"alpha": 0.001, "l1_ratio": 0.25},
                    seed,
                ),
                ModelSpec(
                    "xgboost_ranker",
                    side,
                    dataset.feature_columns,
                    {
                        "n_estimators": 500,
                        "max_depth": 6,
                        "learning_rate": 0.03,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                    },
                    seed,
                ),
                ModelSpec(
                    "xgboost_meta",
                    side,
                    dataset.feature_columns,
                    {
                        "n_estimators": 300,
                        "max_depth": 4,
                        "learning_rate": 0.03,
                    },
                    seed,
                ),
            ]
        )
    return tuple(specs)


def _model(spec: ModelSpec, *, device: str) -> Any:
    parameters = dict(spec.parameters)
    if spec.algorithm == "ridge":
        return LinearRanker(kind="ridge", seed=spec.seed, **parameters)
    if spec.algorithm == "elastic_net":
        return LinearRanker(kind="elastic_net", seed=spec.seed, **parameters)
    if spec.algorithm == "xgboost_ranker":
        return XGBoostRanker(device=device, seed=spec.seed, **parameters)
    if spec.algorithm == "xgboost_meta":
        return XGBoostMetaLabeler(device=device, seed=spec.seed, **parameters)
    raise ValueError(f"unsupported model algorithm {spec.algorithm}")


def _experiment_id(study_id: str, spec: ModelSpec) -> str:
    """Bind a model declaration to its campaign/data study identity."""

    return hashlib.sha256(f"{study_id}:{spec.trial_id}".encode()).hexdigest()


def _run_declared_trial(
    dataset: CrossSectionalDataset,
    spec: ModelSpec,
    *,
    catalog: Catalog,
    study_id: str,
    device: str,
    declared_family_trials: int,
    min_train_years: int,
    test_years: int,
    step_years: int,
    purge_sessions: int,
    diagnostic_top_k: int,
) -> tuple[dict[str, Any] | None, bool]:
    """Run or replay one atomically claimed model declaration."""

    trial_id = _experiment_id(study_id, spec)
    claimed = catalog.claim_experiment(study_id, trial_id, asdict(spec), device=device)
    if not claimed:
        existing = next(
            (
                item
                for item in catalog.experiments(study_id)
                if item.trial_id == trial_id
            ),
            None,
        )
        if existing is not None and existing.status == "COMPLETE" and existing.metrics:
            return existing.metrics, False
        return None, existing is not None and existing.status == "FAILED"
    try:
        side = spec.side
        target = dataset.long_target if side == "LONG" else dataset.short_target
        candidates_only = spec.algorithm == "xgboost_meta"
        frame = dataset.model_frame(side=side, candidates_only=candidates_only)
        frame = frame.dropna(subset=[target, "label_end"])
        folds = purged_panel_splits(
            frame,
            min_train_years=min_train_years,
            test_years=test_years,
            step_years=step_years,
            purge_sessions=purge_sessions,
        )
        if not folds:
            raise ValueError("no eligible purged chronological folds")
        predictions = np.full(len(frame), np.nan, dtype=float)
        fold_rows: list[dict[str, Any]] = []
        for fold in folds:
            train = frame.iloc[fold.train_indices]
            test = frame.iloc[fold.test_indices]
            estimator = _model(spec, device=device)
            if spec.algorithm == "xgboost_meta":
                estimator.fit(train, features=spec.features, target=target)
                forecast = estimator.predict_probability(test, features=spec.features)
            else:
                estimator.fit(
                    train,
                    features=spec.features,
                    target=target,
                    group=dataset.group_column,
                )
                forecast = estimator.predict(test, features=spec.features)
            unassigned = ~np.isfinite(predictions[fold.test_indices])
            predictions[fold.test_indices[unassigned]] = forecast[unassigned]
            evidence = ranking_diagnostics(
                test[target].to_numpy(dtype=float),
                forecast,
                pd.to_datetime(test["date"]),
                top_k=diagnostic_top_k,
            )
            fold_rows.append({"fold": fold.fold, **asdict(evidence)})
        selected = np.isfinite(predictions)
        aggregate = ranking_diagnostics(
            frame.loc[selected, target].to_numpy(dtype=float),
            predictions[selected],
            pd.to_datetime(frame.loc[selected, "date"]),
            top_k=diagnostic_top_k,
        )
        metrics: dict[str, Any] = {
            "study_id": study_id,
            "trial_id": trial_id,
            "model_spec_id": spec.trial_id,
            "algorithm": spec.algorithm,
            "side": side,
            "device": device,
            "declared_family_trials": declared_family_trials,
            "bias_tier": dataset.bias_tier,
            "fold_count": len(folds),
            "event_features_available": dataset.event_features_available,
            "intraday_decision_features_available": (
                dataset.intraday_decision_features_available
            ),
            **asdict(aggregate),
            "folds": fold_rows,
            "promotion_eligible": False,
            "promotion_blocker": (
                "model ranking diagnostics are not a causal portfolio backtest"
            ),
        }
        catalog.complete_experiment(trial_id, metrics)
        return metrics, False
    except Exception as error:
        catalog.fail_experiment(trial_id, f"{type(error).__name__}: {error}")
        return None, True


def run_model_study(
    dataset: CrossSectionalDataset,
    specs: tuple[ModelSpec, ...],
    *,
    catalog: Catalog,
    study_id: str,
    use_gpu: bool = False,
    gpu_devices: tuple[int, ...] = (0, 1),
    min_train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
    purge_sessions: int = 5,
    diagnostic_top_k: int = 5,
) -> ModelStudyResult:
    """Run every declared model under identical purged chronological folds.

    GPU mode creates one worker per configured device and runs that device's
    deterministic bucket sequentially.  The feature frame is shared read-only,
    while the SQLite claims prevent duplicate work across local or external
    workers. Ranking diagnostics remain non-promotable until a complete causal
    portfolio backtest, independent confirmation, and holdout evaluation exist.
    """

    if not specs:
        raise ValueError("at least one model trial must be declared")
    model_spec_ids = tuple(spec.trial_id for spec in specs)
    if len(set(model_spec_ids)) != len(model_spec_ids):
        raise ValueError("duplicate model trial declarations")
    identifiers = tuple(_experiment_id(study_id, spec) for spec in specs)
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("study-bound model trial identity collision")
    if use_gpu:
        visible = set(visible_cuda_devices())
        requested = set(gpu_devices)
        missing = sorted(requested.difference(visible))
        if missing:
            raise RuntimeError(
                "requested CUDA devices are not visible; refusing silent CPU "
                f"fallback for devices {missing}"
            )

    def execute(spec: ModelSpec, device: str) -> tuple[dict[str, Any] | None, bool]:
        return _run_declared_trial(
            dataset,
            spec,
            catalog=catalog,
            study_id=study_id,
            device=device,
            declared_family_trials=len(specs),
            min_train_years=min_train_years,
            test_years=test_years,
            step_years=step_years,
            purge_sessions=purge_sessions,
            diagnostic_top_k=diagnostic_top_k,
        )

    outcomes: dict[str, tuple[dict[str, Any] | None, bool]] = {}
    if use_gpu:
        buckets: dict[str, list[ModelSpec]] = {
            f"cuda:{device}": [] for device in gpu_devices
        }
        for spec in specs:
            buckets[
                assigned_gpu_device(_experiment_id(study_id, spec), gpu_devices)
            ].append(spec)

        def execute_bucket(
            device: str, declared: list[ModelSpec]
        ) -> list[tuple[str, tuple[dict[str, Any] | None, bool]]]:
            return [
                (_experiment_id(study_id, spec), execute(spec, device))
                for spec in declared
            ]

        with ThreadPoolExecutor(max_workers=len(buckets)) as workers:
            futures = [
                workers.submit(execute_bucket, device, declared)
                for device, declared in buckets.items()
            ]
            for future in futures:
                outcomes.update(future.result())
    else:
        outcomes = {
            _experiment_id(study_id, spec): execute(spec, "cpu") for spec in specs
        }

    rows = [
        outcome[0]
        for trial_id in identifiers
        if (outcome := outcomes[trial_id])[0] is not None
    ]
    failed = tuple(trial_id for trial_id in identifiers if outcomes[trial_id][1])
    return ModelStudyResult(
        study_id,
        pd.DataFrame(rows),
        identifiers,
        failed,
        dataset.bias_tier,
    )
