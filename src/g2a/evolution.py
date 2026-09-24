"""Risk localization and verified, component-specific Safety Skill evolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Protocol, Sequence

from .skills import SafetySkill


class SkillComponent(str, Enum):
    ACTIVATION = "activation"
    CORRECTION = "correction"
    TERMINATION = "termination"


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    action: str
    risk: float


@dataclass(frozen=True, slots=True)
class ValidationRollout:
    steps: tuple[TrajectoryStep, ...]
    utility: float
    probability_ratio: float = 1.0
    advantage: float = 0.0


@dataclass(frozen=True, slots=True)
class SkillCandidate:
    parent: SafetySkill
    candidate: SafetySkill
    component: SkillComponent
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    accepted: bool
    reason: str
    average_utility: float
    maximum_risk: float
    ppo_surrogate: float


class IndependentVerifier(Protocol):
    def verify_skill(
        self, parent: SafetySkill, candidate: SafetySkill, rollouts: Sequence[ValidationRollout]
    ) -> bool: ...


def first_violation_step(risks: Sequence[float], *, threshold: float = 0.45) -> int | None:
    """Return the zero-based first threshold-crossing step t*, or ``None``."""

    return next((index for index, risk in enumerate(risks) if risk > threshold), None)


def clipped_utility_surrogate(
    rollouts: Sequence[ValidationRollout], *, clip_epsilon: float = 0.2
) -> float:
    """PPO-style clipped utility gate reported in Appendix B.4."""

    if not rollouts:
        raise ValueError("at least one validation rollout is required")
    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be positive")
    values = []
    for rollout in rollouts:
        ratio = rollout.probability_ratio
        clipped = min(max(ratio, 1.0 - clip_epsilon), 1.0 + clip_epsilon)
        values.append(min(ratio * rollout.advantage, clipped * rollout.advantage))
    return sum(values) / len(values)


def verify_candidate(
    proposal: SkillCandidate,
    rollouts: Sequence[ValidationRollout],
    verifier: IndependentVerifier,
    *,
    parent_utility: float,
    threshold: float = 0.45,
    clip_epsilon: float = 0.2,
    required_rollouts: int = 8,
) -> VerificationResult:
    """Apply utility-surrogate, Guard, utility, and independent-verifier gates."""

    _validate_component_specific(proposal)
    if len(rollouts) < required_rollouts:
        return VerificationResult(False, "insufficient rollouts", -math.inf, math.inf, -math.inf)
    utility = sum(item.utility for item in rollouts) / len(rollouts)
    risks = [step.risk for rollout in rollouts for step in rollout.steps]
    maximum_risk = max(risks, default=math.inf)
    surrogate = clipped_utility_surrogate(rollouts, clip_epsilon=clip_epsilon)
    if surrogate < 0:
        return VerificationResult(
            False, "negative clipped utility surrogate", utility, maximum_risk, surrogate
        )
    if not risks or maximum_risk > threshold:
        return VerificationResult(
            False, "Guard threshold violation", utility, maximum_risk, surrogate
        )
    if utility < parent_utility:
        return VerificationResult(
            False, "task utility regression", utility, maximum_risk, surrogate
        )
    if not verifier.verify_skill(proposal.parent, proposal.candidate, rollouts):
        return VerificationResult(
            False, "independent verifier rejection", utility, maximum_risk, surrogate
        )
    return VerificationResult(
        True, "all verification gates passed", utility, maximum_risk, surrogate
    )


def _validate_component_specific(proposal: SkillCandidate) -> None:
    parent = proposal.parent
    candidate = proposal.candidate
    changed: set[SkillComponent] = set()
    if parent.activation != candidate.activation:
        changed.add(SkillComponent.ACTIVATION)
    if parent.correction_instructions != candidate.correction_instructions:
        changed.add(SkillComponent.CORRECTION)
    if parent.termination != candidate.termination:
        changed.add(SkillComponent.TERMINATION)
    if changed != {proposal.component}:
        raise ValueError("each candidate must modify exactly its declared Skill component")


@dataclass(frozen=True, slots=True)
class InterventionPair:
    state: Mapping[str, object]
    skill: str
    corrected_action: str
    original_action: str


class PairVerifier(Protocol):
    def __call__(self, pair: InterventionPair) -> bool: ...


class GuideExperienceBuffer:
    """Stores only independently verified lower-risk correction pairs."""

    def __init__(self) -> None:
        self._pairs: list[InterventionPair] = []

    @property
    def pairs(self) -> tuple[InterventionPair, ...]:
        return tuple(self._pairs)

    def add(
        self,
        pair: InterventionPair,
        *,
        corrected_risk: float,
        original_risk: float,
        threshold: float,
        task_utility_preserved: bool,
        verifier: PairVerifier,
    ) -> bool:
        accepted = (
            corrected_risk <= threshold
            and corrected_risk < original_risk
            and task_utility_preserved
            and verifier(pair)
        )
        if accepted:
            self._pairs.append(pair)
        return accepted
