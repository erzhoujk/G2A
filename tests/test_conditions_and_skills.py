from g2a.conditions import Condition, ConditionGroup
from g2a.skills import SafetySkill, SkillPool


def make_skill(skill_id: str, priority: int, termination: bool = False) -> SafetySkill:
    return SafetySkill(
        skill_id=skill_id,
        name=skill_id,
        activation=ConditionGroup((Condition("risk", "eq", "privacy"),)),
        correction_instructions="correct safely",
        termination=ConditionGroup((Condition("done", "eq", termination),)),
        priority=priority,
    )


def test_router_prefers_active_skill_then_switches_on_termination() -> None:
    pool = SkillPool()
    pool.add(make_skill("low", 1, termination=True))
    pool.add(make_skill("high", 10))
    assert [skill.skill_id for skill in pool.route({"risk": "privacy"})] == ["high", "low"]
    active = pool.route({"risk": "privacy", "done": False}, active_skill_id="low")
    assert active[0].skill_id == "low"
    switched = pool.route({"risk": "privacy", "done": True}, active_skill_id="low")
    assert switched[0].skill_id == "high"


def test_nested_condition_without_eval() -> None:
    condition = Condition("authorization.scope", "contains", "write")
    assert condition.matches({"authorization": {"scope": ["read", "write"]}})
