"""Deterministic hard constraints that complement the learned Guard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class PolicyViolation:
    rule_id: str
    message: str
    severity: str = "block"


class PolicyRule(Protocol):
    def __call__(self, state: Mapping[str, Any], action: str) -> PolicyViolation | None: ...


@dataclass(slots=True)
class PolicyEngine:
    """Fail-closed hard rules for invariants unsuitable for probabilistic scoring."""

    rules: list[PolicyRule] = field(default_factory=list)

    def check(self, state: Mapping[str, Any], action: str) -> tuple[PolicyViolation, ...]:
        violations: list[PolicyViolation] = []
        for rule in self.rules:
            result = rule(state, action)
            if result is not None:
                violations.append(result)
        return tuple(violations)


def deny_substrings(rule_id: str, *needles: str) -> PolicyRule:
    lowered = tuple(needle.casefold() for needle in needles)

    def check(_state: Mapping[str, Any], action: str) -> PolicyViolation | None:
        match = next((needle for needle in lowered if needle in action.casefold()), None)
        if match is None:
            return None
        return PolicyViolation(rule_id=rule_id, message=f"action contains denied pattern: {match}")

    return check


def require_state_flag(rule_id: str, action_substring: str, flag: str) -> PolicyRule:
    """Require a top-level Boolean state flag for a matching action."""

    def check(state: Mapping[str, Any], action: str) -> PolicyViolation | None:
        if action_substring.casefold() not in action.casefold() or state.get(flag) is True:
            return None
        return PolicyViolation(rule_id=rule_id, message=f"state flag '{flag}' is required")

    return check
