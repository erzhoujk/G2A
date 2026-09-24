"""Declarative, non-eval activation and termination conditions for Safety Skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_MISSING = object()


@dataclass(frozen=True, slots=True)
class Condition:
    field: str
    op: str = "eq"
    value: Any = None

    def matches(self, state: Mapping[str, Any]) -> bool:
        actual = _lookup(state, self.field)
        if self.op == "exists":
            return (actual is not _MISSING) is bool(self.value)
        if actual is _MISSING:
            return False
        if self.op == "eq":
            return actual == self.value
        if self.op == "ne":
            return actual != self.value
        if self.op == "in":
            return actual in self.value
        if self.op == "not_in":
            return actual not in self.value
        if self.op == "contains":
            return self.value in actual
        if self.op == "truthy":
            return bool(actual) is bool(self.value)
        if self.op == "gt":
            return actual > self.value
        if self.op == "gte":
            return actual >= self.value
        if self.op == "lt":
            return actual < self.value
        if self.op == "lte":
            return actual <= self.value
        raise ValueError(f"unsupported condition operator: {self.op}")


@dataclass(frozen=True, slots=True)
class ConditionGroup:
    conditions: tuple[Condition, ...] = ()
    mode: str = "all"

    @classmethod
    def from_dicts(
        cls, items: Sequence[Mapping[str, Any]], *, mode: str = "all"
    ) -> "ConditionGroup":
        return cls(
            conditions=tuple(
                Condition(
                    field=str(item["field"]),
                    op=str(item.get("op", "eq")),
                    value=item.get("value"),
                )
                for item in items
            ),
            mode=mode,
        )

    def matches(self, state: Mapping[str, Any]) -> bool:
        if self.mode == "all":
            return all(condition.matches(state) for condition in self.conditions)
        if self.mode == "any":
            return any(condition.matches(state) for condition in self.conditions)
        raise ValueError("condition-group mode must be 'all' or 'any'")


def _lookup(state: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = state
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current
