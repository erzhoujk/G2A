import pytest

from g2a.conditions import Condition, ConditionGroup
from g2a.evolution import (
    SkillCandidate,
    SkillComponent,
    TrajectoryStep,
    ValidationRollout,
    first_violation_step,
    verify_candidate,
)
from g2a.skills import SafetySkill


class AcceptVerifier:
    def verify_skill(self, _parent, _candidate, _rollouts) -> bool:
        return True


def skills() -> tuple[SafetySkill, SafetySkill]:
    parent = SafetySkill(
        "privacy", "Privacy", ConditionGroup((Condition("private", "eq", True),)), "old"
    )
    return parent, parent.with_component("correction", "new")


def safe_rollouts(count: int = 8) -> list[ValidationRollout]:
    return [
        ValidationRollout((TrajectoryStep("safe", 0.2),), 1.0, 1.0, 0.1)
        for _ in range(count)
    ]


def test_first_violation_localization() -> None:
    assert first_violation_step([0.1, 0.3, 0.7], threshold=0.45) == 2
    assert first_violation_step([0.1, 0.3], threshold=0.45) is None


def test_candidate_passes_all_gates() -> None:
    parent, candidate = skills()
    result = verify_candidate(
        SkillCandidate(parent, candidate, SkillComponent.CORRECTION),
        safe_rollouts(),
        AcceptVerifier(),
        parent_utility=1.0,
    )
    assert result.accepted


def test_any_unsafe_validation_action_rejects_candidate() -> None:
    parent, candidate = skills()
    rollouts = safe_rollouts()
    rollouts[-1] = ValidationRollout((TrajectoryStep("unsafe", 0.8),), 1.0, 1.0, 0.1)
    result = verify_candidate(
        SkillCandidate(parent, candidate, SkillComponent.CORRECTION),
        rollouts,
        AcceptVerifier(),
        parent_utility=1.0,
    )
    assert not result.accepted
    assert result.reason == "Guard threshold violation"


def test_multi_component_edit_is_rejected() -> None:
    parent, candidate = skills()
    candidate = candidate.with_component("termination", ConditionGroup((Condition("done"),)))
    with pytest.raises(ValueError, match="exactly"):
        verify_candidate(
            SkillCandidate(parent, candidate, SkillComponent.CORRECTION),
            safe_rollouts(),
            AcceptVerifier(),
            parent_utility=1.0,
        )
