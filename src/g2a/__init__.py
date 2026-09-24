"""G2A: Guard-and-Guide runtime safety for long-horizon agents."""

from .conditions import Condition, ConditionGroup
from .runtime import Decision, DecisionKind, G2ARuntime, RuntimeConfig
from .skills import SafetySkill, SkillPool
from .tcsd import TCSDEvidence, compute_tcsd_evidence

__all__ = [
    "Condition",
    "ConditionGroup",
    "Decision",
    "DecisionKind",
    "G2ARuntime",
    "RuntimeConfig",
    "SafetySkill",
    "SkillPool",
    "TCSDEvidence",
    "compute_tcsd_evidence",
]

__version__ = "0.1.0"
