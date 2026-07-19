"""Loss-aware V2 research contracts and paper-forward tooling."""

from edgestack.v2.gates import CapabilityGate, CapabilityReport, evaluate_capabilities
from edgestack.v2.metrics import LossMetrics, loss_metrics
from edgestack.v2.research import TrialSpec, declared_trials

__all__ = [
    "CapabilityGate",
    "CapabilityReport",
    "LossMetrics",
    "TrialSpec",
    "declared_trials",
    "evaluate_capabilities",
    "loss_metrics",
]
