"""Cross-sectional empirical-Bayes shrinkage toward zero.

The estimator follows the shrinkage logic used in broad anomaly-replication work:
the cross-edge prior variance is the non-negative excess of observed estimate
variance over average sampling variance.  It is deliberately not tuned against
strategy performance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ShrinkageResult:
    """Shrunk estimates and the common prior variance."""

    raw: Mapping[str, float]
    shrunk: Mapping[str, float]
    factors: Mapping[str, float]
    tau_squared: float


def empirical_bayes_shrinkage(
    estimates: Mapping[str, float], sampling_variances: Mapping[str, float]
) -> ShrinkageResult:
    """Shrink noisy edge means toward zero.

    Parameters
    ----------
    estimates
        Raw net expected return estimates keyed by edge ID.
    sampling_variances
        Estimated variance of each mean (``sigma² / T``), not return variance.

    Returns
    -------
    ShrinkageResult
        Shrunk estimates bounded between zero and their raw values.
    """

    if set(estimates) != set(sampling_variances):
        raise ValueError("estimates and sampling_variances must have identical keys")
    if not estimates:
        return ShrinkageResult({}, {}, {}, 0.0)
    keys = sorted(estimates)
    raw = np.asarray([float(estimates[key]) for key in keys], dtype=float)
    sampling = np.asarray([float(sampling_variances[key]) for key in keys], dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(sampling)):
        raise ValueError("estimates and variances must be finite")
    if np.any(sampling < 0):
        raise ValueError("sampling variances cannot be negative")
    observed_variance = float(np.var(raw, ddof=1)) if len(raw) > 1 else 0.0
    tau_squared = max(0.0, observed_variance - float(np.mean(sampling)))
    denominator = tau_squared + sampling
    factors = np.divide(
        tau_squared,
        denominator,
        out=np.zeros_like(sampling),
        where=denominator > 0,
    )
    shrunk = raw * np.clip(factors, 0.0, 1.0)
    return ShrinkageResult(
        raw=dict(zip(keys, raw.tolist(), strict=True)),
        shrunk=dict(zip(keys, shrunk.tolist(), strict=True)),
        factors=dict(zip(keys, factors.tolist(), strict=True)),
        tau_squared=tau_squared,
    )
