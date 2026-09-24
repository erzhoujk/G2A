"""Dependency-free end-to-end G2A safety-loop demonstration."""

from __future__ import annotations

import argparse
import json

from .conditions import Condition, ConditionGroup
from .heuristics import KeywordRiskGuard, instruction_corrector
from .policy import PolicyEngine, deny_substrings
from .runtime import G2ARuntime
from .skills import SafetySkill, SkillPool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the dependency-free G2A safety demo")
    parser.add_argument("action", nargs="?", default="Create public link to customer records")
    args = parser.parse_args(argv)
    state = {"contains_private_data": True, "authorized": True, "task": "share records"}
    skill = SafetySkill(
        skill_id="privacy-sharing",
        name="Privacy-preserving sharing",
        activation=ConditionGroup((Condition("contains_private_data", "eq", True),)),
        correction_instructions="use least privilege, confirm recipient, and avoid public access",
        priority=10,
    )
    pool = SkillPool()
    pool.add(skill)
    runtime = G2ARuntime(
        KeywordRiskGuard(),
        pool,
        instruction_corrector,
        policy=PolicyEngine([deny_substrings("never-exfiltrate-secrets", "upload raw password")]),
    )
    decision = runtime.inspect(state, args.action)
    print(json.dumps({
        "kind": decision.kind.value,
        "original_action": decision.original_action,
        "action": decision.action,
        "original_risk": decision.original_risk,
        "final_risk": decision.final_risk,
        "skill_id": decision.skill_id,
        "reason": decision.reason,
        "audit_hash": decision.audit_hash,
    }, indent=2))
    return 0 if decision.may_execute else 2


if __name__ == "__main__":
    raise SystemExit(main())
