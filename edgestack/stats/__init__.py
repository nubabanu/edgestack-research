"""Research-grade statistical primitives for EdgeStack."""

from edgestack.stats.deflated_sharpe import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from edgestack.stats.multiple_testing import benjamini_hochberg, bonferroni
from edgestack.stats.tests import hac_mean_test, summarize_returns

__all__ = [
    "benjamini_hochberg",
    "bonferroni",
    "deflated_sharpe_ratio",
    "hac_mean_test",
    "probabilistic_sharpe_ratio",
    "summarize_returns",
]
