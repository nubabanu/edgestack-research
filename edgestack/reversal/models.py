"""Interpretable baselines, LambdaMART ranking, and reversal meta-labeling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, Self

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import ElasticNet, Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]


class Ranker(Protocol):
    """Minimal common interface used by purged chronological studies."""

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        features: tuple[str, ...],
        target: str,
        group: str,
    ) -> Self:
        """Fit one side-specific model."""

    def predict(
        self, frame: pd.DataFrame, *, features: tuple[str, ...]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return an ordering score for each row."""


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Hash-addressed declaration of one model trial."""

    algorithm: Literal["ridge", "elastic_net", "xgboost_ranker", "xgboost_meta"]
    side: Literal["LONG", "SHORT"]
    features: tuple[str, ...]
    parameters: dict[str, Any]
    seed: int

    @property
    def trial_id(self) -> str:
        """Return a stable identity for duplicate suppression and trial counting."""

        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assigned_gpu_device(trial_id: str, devices: tuple[int, ...]) -> str:
    """Assign independent trials deterministically across separate CUDA devices."""

    if not devices or any(device < 0 for device in devices):
        raise ValueError("devices must contain non-negative CUDA indices")
    bucket = int(trial_id[:16], 16) % len(devices)
    return f"cuda:{devices[bucket]}"


def visible_cuda_devices() -> tuple[int, ...]:
    """Return CUDA indices visible to this process, or an empty tuple.

    XGBoost can silently warn and fall back to CPU when a requested CUDA device
    is unavailable.  The study runner calls this guard first so `--gpu` is a
    strict execution request rather than a mislabeled CPU run.
    """

    try:
        from numba import cuda  # type: ignore[import-untyped]

        if not cuda.is_available():
            return ()
        return tuple(range(len(cuda.gpus)))
    except Exception:  # pragma: no cover - depends on host CUDA driver state
        return ()


def _model_matrix(
    frame: pd.DataFrame, features: tuple[str, ...]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    missing = set(features).difference(frame.columns)
    if missing:
        raise ValueError(f"model frame is missing features {sorted(missing)}")
    values = frame.loc[:, features].to_numpy(dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def _within_group_rank_target(
    frame: pd.DataFrame, *, target: str, group: str
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if target not in frame or group not in frame:
        raise ValueError("target and group columns are required")
    ranks = frame.groupby(group, sort=False)[target].rank(method="average", pct=True)
    return ranks.to_numpy(dtype=float)


class LinearRanker:
    """Ridge or elastic-net baseline fitted to within-date target ranks."""

    def __init__(
        self,
        *,
        kind: Literal["ridge", "elastic_net"] = "ridge",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        seed: int = 0,
    ) -> None:
        if alpha <= 0.0 or not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("alpha must be positive and l1_ratio in [0, 1]")
        estimator: Ridge | ElasticNet
        if kind == "ridge":
            estimator = Ridge(alpha=alpha)
        elif kind == "elastic_net":
            estimator = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=seed,
                max_iter=10_000,
            )
        else:
            raise ValueError("unsupported linear ranker")
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        features: tuple[str, ...],
        target: str,
        group: str,
    ) -> LinearRanker:
        """Fit the additive benchmark to cross-sectional relevance ranks."""

        matrix = _model_matrix(frame, features)
        relevance = _within_group_rank_target(frame, target=target, group=group)
        self.pipeline.fit(matrix, relevance)
        return self

    def predict(
        self, frame: pd.DataFrame, *, features: tuple[str, ...]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return continuous linear ordering scores."""

        return np.asarray(
            self.pipeline.predict(_model_matrix(frame, features)), dtype=float
        )


class XGBoostRanker:
    """LambdaMART cross-sectional ranker with one query group per trading date."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        seed: int = 0,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.03,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ) -> None:
        try:
            from xgboost import XGBRanker
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install EdgeStack's ml extra to use XGBoost") from error
        if n_estimators < 1 or max_depth < 1 or learning_rate <= 0.0:
            raise ValueError("tree count/depth/rate must be positive")
        self.model = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            tree_method="hist",
            device=device,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=seed,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=10,
            n_jobs=1 if device.startswith("cuda") else -1,
        )

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        features: tuple[str, ...],
        target: str,
        group: str,
    ) -> XGBoostRanker:
        """Fit LambdaMART using sorted date IDs as query identifiers."""

        ordered = frame.sort_values([group, "symbol"], kind="stable")
        qid = ordered[group].to_numpy(dtype=np.int64)
        if np.any(np.diff(qid) < 0):
            raise ValueError("ranking groups must be non-decreasing")
        percentile = ordered.groupby(group, sort=False)[target].rank(
            method="average", pct=True
        )
        relevance = np.minimum(np.floor(percentile.to_numpy() * 31.0), 30.0).astype(
            np.int32
        )
        self.model.fit(
            _model_matrix(ordered, features), relevance, qid=qid, verbose=False
        )
        return self

    def predict(
        self, frame: pd.DataFrame, *, features: tuple[str, ...]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return LambdaMART ranking scores in the supplied row order."""

        return np.asarray(
            self.model.predict(_model_matrix(frame, features)), dtype=float
        )


class XGBoostMetaLabeler:
    """Take/skip probability model conditioned on economic reversal candidates."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        seed: int = 0,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.03,
    ) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install EdgeStack's ml extra to use XGBoost") from error
        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1 if device.startswith("cuda") else -1,
        )

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        features: tuple[str, ...],
        target: str,
    ) -> XGBoostMetaLabeler:
        """Fit a profitable-after-cost binary label on preselected candidates."""

        labels = (pd.to_numeric(frame[target], errors="coerce") > 0.0).astype(int)
        if labels.nunique() < 2:
            raise ValueError(
                "meta-label training needs both profitable and losing trades"
            )
        self.model.fit(_model_matrix(frame, features), labels.to_numpy(), verbose=False)
        return self

    def predict_probability(
        self, frame: pd.DataFrame, *, features: tuple[str, ...]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return take probabilities; these are model outputs, not confidence claims."""

        probabilities = self.model.predict_proba(_model_matrix(frame, features))
        return np.asarray(probabilities[:, 1], dtype=float)
