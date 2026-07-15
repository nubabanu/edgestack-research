"""Deterministic matched placebo controls.

Each real rule receives a shuffled-date return control and an
exposure/turnover-matched random-signal control. Seeds are derived from the
campaign seed and stable parent hypothesis ID, so batching and worker count do
not affect results.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pandas as pd

from edgestack.models import HypothesisSpec
from edgestack.stats._types import FloatArray


def control_seed(campaign_seed: int, hypothesis_id: str, kind: str) -> int:
    """Derive an unsigned 64-bit seed from stable control identity."""

    payload = f"{campaign_seed}:{hypothesis_id}:{kind}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def control_specs(
    spec: HypothesisSpec, *, campaign_seed: int = 0
) -> tuple[HypothesisSpec, ...]:
    """Declare the two controls paired with a real hypothesis."""

    if spec.placebo_kind is not None:
        raise ValueError("controls cannot be generated from another control")
    parent = spec.hypothesis_id
    declarations: list[HypothesisSpec] = []
    for kind in ("SHUFFLED_DATE", "MATCHED_RANDOM"):
        seed = control_seed(campaign_seed, parent, kind)
        declarations.append(
            replace(
                spec,
                description=f"{kind} placebo for {spec.description}",
                placebo_kind=kind,
                parameters={
                    **spec.parameters,
                    "parent_id": parent,
                    "control_seed": seed,
                },
            )
        )
    return tuple(declarations)


def shuffled_date_returns(
    returns: FloatArray | pd.Series | pd.DataFrame,
    *,
    seed: int,
) -> FloatArray | pd.Series | pd.DataFrame:
    """Shuffle complete dates while preserving cross-sectional dependence."""

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(returns))
    if isinstance(returns, pd.Series):
        values = returns.to_numpy()[order]
        return pd.Series(values, index=returns.index, name=returns.name)
    if isinstance(returns, pd.DataFrame):
        values = returns.to_numpy()[order]
        return pd.DataFrame(values, index=returns.index, columns=returns.columns)
    values = np.asarray(returns)
    if values.ndim not in (1, 2):
        raise ValueError("returns must be one- or two-dimensional")
    return values[order]


def matched_random_signal(
    signal: FloatArray | pd.DataFrame, *, seed: int
) -> FloatArray | pd.DataFrame:
    """Randomize asset labels within each date, exactly preserving exposures.

    A global asset-label permutation preserves every row's gross/net exposure,
    position count, and total turnover exactly while removing asset identity.
    Scalar event signals use a circular date rotation with the same invariants.
    """

    values = (
        signal.to_numpy(dtype=float)
        if isinstance(signal, pd.DataFrame)
        else np.asarray(signal, dtype=float)
    )
    if values.ndim not in (1, 2):
        raise ValueError("signal must be one- or two-dimensional")
    rng = np.random.default_rng(seed)
    if values.ndim == 1:
        # A nonzero circular rotation preserves exposure and every cyclic
        # transition count, while breaking alignment with calendar returns.
        shift = int(rng.integers(1, len(values))) if len(values) > 1 else 0
        output = np.roll(values, shift)
    else:
        # One global asset-label permutation preserves each date's gross/net
        # exposure *and* total turnover exactly.
        output = values[:, rng.permutation(values.shape[1])]
    if isinstance(signal, pd.DataFrame):
        return pd.DataFrame(output, index=signal.index, columns=signal.columns)
    return output


def turnover(signal: FloatArray, *, axis: int = 0) -> float:
    """Return mean one-way absolute weight change."""

    values = np.asarray(signal, dtype=float)
    if values.shape[axis] < 2:
        return 0.0
    changes = np.diff(values, axis=axis)
    if values.ndim == 2 and axis == 0:
        return float(np.nanmean(np.nansum(np.abs(changes), axis=1)) / 2.0)
    return float(np.nanmean(np.abs(changes)))
