from typing import Any, Mapping

from g2a.conditions import Condition, ConditionGroup
from g2a.policy import PolicyEngine, deny_substrings
from g2a.runtime import DecisionKind, G2ARuntime
from g2a.skills import SafetySkill, SkillPool


class Guard:
    def score(self, _state: Mapping[str, Any], _skill: str, action: str) -> float:
        return 0.9 if "public" in action else 0.1


def pool() -> SkillPool:
    result = SkillPool()
    result.add(SafetySkill(
        "privacy",
        "Privacy",
        ConditionGroup((Condition("private", "eq", True),)),
        "replace public access with private access",
    ))
    return result


def correct(_state, action, _skill):
    return action.replace("public", "private")


def test_safe_action_executes_without_intervention() -> None:
    runtime = G2ARuntime(Guard(), pool(), correct)
    decision = runtime.inspect({"private": True}, "create private link")
    assert decision.kind is DecisionKind.EXECUTE
    assert decision.action == "create private link"


def test_risky_action_is_corrected_and_rescored() -> None:
    runtime = G2ARuntime(Guard(), pool(), correct)
    decision = runtime.inspect({"private": True}, "create public link")
    assert decision.kind is DecisionKind.CORRECTED
    assert decision.action == "create private link"
    assert decision.original_risk == 0.9
    assert decision.final_risk == 0.1
    assert runtime.audit.verify()


def test_no_skill_fails_closed() -> None:
    runtime = G2ARuntime(Guard(), pool(), correct)
    decision = runtime.inspect({"private": False}, "create public link")
    assert decision.kind is DecisionKind.BLOCK
    assert decision.action is None


def test_hard_policy_overrides_low_model_risk() -> None:
    runtime = G2ARuntime(
        Guard(), pool(), correct, policy=PolicyEngine([deny_substrings("secret", "raw password")])
    )
    decision = runtime.inspect({"private": True}, "upload raw password")
    assert decision.kind is DecisionKind.BLOCK
    assert decision.original_risk is None


def test_unsafe_correction_is_never_executed() -> None:
    runtime = G2ARuntime(Guard(), pool(), lambda _s, action, _skill: action)
    decision = runtime.inspect({"private": True}, "create public link")
    assert decision.kind is DecisionKind.BLOCK


def test_guard_and_policy_errors_fail_closed() -> None:
    class BrokenGuard:
        def score(self, _state, _skill, _action):
            raise RuntimeError("offline")

    def broken_rule(_state, _action):
        raise RuntimeError("invalid policy state")

    guard_decision = G2ARuntime(BrokenGuard(), pool(), correct).inspect(
        {"private": True}, "create private link"
    )
    assert guard_decision.kind is DecisionKind.BLOCK

    policy_decision = G2ARuntime(
        Guard(), pool(), correct, policy=PolicyEngine([broken_rule])
    ).inspect({"private": True}, "create private link")
    assert policy_decision.kind is DecisionKind.BLOCK


def test_deployment_freezes_skill_pool() -> None:
    skills = pool()
    G2ARuntime(Guard(), skills, correct)
    assert skills.frozen
