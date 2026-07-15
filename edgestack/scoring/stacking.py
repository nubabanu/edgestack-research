"""Correlation-clustered equal-weight composites."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from edgestack.models import StackArtifact
from edgestack.scoring.shrinkage import empirical_bayes_shrinkage


@dataclass(frozen=True, slots=True)
class StackResult:
    """Frozen stack definition and historical composite stream."""

    artifact: StackArtifact
    returns: pd.Series


def correlation_clusters(
    returns: pd.DataFrame, threshold: float = 0.70
) -> dict[str, int]:
    """Find deterministic connected components at an absolute-correlation cutoff."""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if returns.columns.has_duplicates:
        raise ValueError("return-stream columns must be unique")
    if any(not isinstance(column, str) for column in returns.columns):
        raise TypeError("return-stream columns must be edge-ID strings")
    columns = sorted(returns.columns)
    if not columns:
        return {}
    correlation = returns[columns].corr(min_periods=20).abs().fillna(0.0)
    unseen = set(columns)
    result: dict[str, int] = {}
    cluster = 0
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            neighbors = [
                candidate
                for candidate in unseen
                if candidate != current
                and float(cast(Any, correlation.at[current, candidate])) >= threshold
            ]
            stack.extend(neighbors)
        for edge in sorted(component):
            result[edge] = cluster
        unseen.difference_update(component)
        cluster += 1
    return result


def equal_cluster_weights(cluster_by_edge: Mapping[str, int]) -> dict[str, float]:
    """Equal-weight clusters and equal-weight their members."""

    if not cluster_by_edge:
        return {}
    members: dict[int, list[str]] = {}
    for edge, cluster in cluster_by_edge.items():
        members.setdefault(cluster, []).append(edge)
    cluster_weight = 1.0 / len(members)
    return {
        edge: cluster_weight / len(edges)
        for _, edges in sorted(members.items())
        for edge in sorted(edges)
    }


def build_stack(
    return_streams: pd.DataFrame,
    net_means: Mapping[str, float],
    sampling_variances: Mapping[str, float],
    dsr_reliability: float,
    *,
    correlation_threshold: float = 0.70,
) -> StackResult:
    """Shrink, cluster, and build a frozen equal-weight composite."""

    if return_streams.columns.has_duplicates:
        raise ValueError("return-stream columns must be unique")
    if set(net_means) != set(sampling_variances):
        raise ValueError("net means and sampling variances need identical edge IDs")
    if not set(net_means).issubset(return_streams.columns):
        raise ValueError("every edge estimate needs a return stream")
    if not 0 <= dsr_reliability <= 1:
        raise ValueError("dsr_reliability must be in [0, 1]")
    edge_ids = tuple(sorted(net_means))
    if not edge_ids:
        empty = StackArtifact("empty", (), {}, {}, {}, dsr_reliability, False)
        return StackResult(empty, pd.Series(dtype=float, name="composite"))
    selected = return_streams.loc[:, list(edge_ids)].astype(float)
    shrinkage = empirical_bayes_shrinkage(net_means, sampling_variances)
    clusters = correlation_clusters(selected, correlation_threshold)
    weights = equal_cluster_weights(clusters)
    composite = selected.mul(pd.Series(weights), axis=1).sum(axis=1, min_count=1)
    composite.name = "composite"
    payload = {
        "edges": edge_ids,
        "clusters": clusters,
        "weights": weights,
        "shrunk": shrinkage.shrunk,
        "threshold": correlation_threshold,
    }
    stack_id = (
        "stack-"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    )
    artifact = StackArtifact(
        stack_id=stack_id,
        edge_ids=edge_ids,
        cluster_by_edge=clusters,
        weights=weights,
        shrunk_means=shrinkage.shrunk,
        dsr_reliability=dsr_reliability,
        promoted=False,
    )
    return StackResult(artifact, composite)


def confidence_score(dsr_probability: float, magnitude_percentile: float) -> int:
    """Map composite reliability and current forecast magnitude to an ordinal score."""

    if not 0 <= dsr_probability <= 1 or not 0 <= magnitude_percentile <= 1:
        raise ValueError("inputs must be in [0, 1]")
    return round(100.0 * dsr_probability * magnitude_percentile)
