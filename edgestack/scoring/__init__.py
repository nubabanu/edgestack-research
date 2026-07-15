"""Empirical-Bayes shrinkage and robust edge stacking."""

from edgestack.scoring.shrinkage import ShrinkageResult, empirical_bayes_shrinkage
from edgestack.scoring.stacking import build_stack, confidence_score

__all__ = [
    "ShrinkageResult",
    "build_stack",
    "confidence_score",
    "empirical_bayes_shrinkage",
]
