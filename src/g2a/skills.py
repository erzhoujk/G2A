"""Structured Safety Skills and deterministic routing."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .conditions import ConditionGroup

Corrector = Callable[[Mapping[str, Any], str, "SafetySkill"], str]


@dataclass(frozen=True, slots=True)
class SafetySkill:
    """omega = <I_omega, pi_omega, beta_omega>."""

    skill_id: str
    name: str
    activation: ConditionGroup
    correction_instructions: str
    termination: ConditionGroup = field(default_factory=ConditionGroup)
    priority: int = 0
    version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fallback_instructions: str = "request clarification or abort the subtask"

    def applicable(self, state: Mapping[str, Any]) -> bool:
        return self.activation.matches(state)

    def should_terminate(self, state: Mapping[str, Any]) -> bool:
        return bool(self.termination.conditions) and self.termination.matches(state)

    def revise(self, state: Mapping[str, Any], action: str, corrector: Corrector) -> str:
        revised = corrector(state, action, self)
        if not isinstance(revised, str) or not revised.strip():
            raise ValueError(f"skill {self.skill_id} produced an empty correction")
        return revised

    def with_component(self, component: str, value: Any) -> "SafetySkill":
        if component == "activation":
            return replace(self, activation=value, version=self.version + 1)
        if component == "correction":
            return replace(self, correction_instructions=str(value), version=self.version + 1)
        if component == "termination":
            return replace(self, termination=value, version=self.version + 1)
        raise ValueError(f"unknown skill component: {component}")

    def as_guard_context(self) -> str:
        return f"{self.name}: {self.correction_instructions}"

    def fallback(self) -> str:
        """Return the conservative fallback mandated by the runtime protocol."""
        return self.fallback_instructions


@dataclass(slots=True)
class SkillPool:
    skills: dict[str, SafetySkill] = field(default_factory=dict)
    frozen: bool = False

    def add(self, skill: SafetySkill, *, replace_existing: bool = False) -> None:
        if self.frozen:
            raise RuntimeError("Skill pool is frozen for deployment")
        if skill.skill_id in self.skills and not replace_existing:
            raise ValueError(f"duplicate skill id: {skill.skill_id}")
        self.skills[skill.skill_id] = skill

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze_for_training(self) -> None:
        self.frozen = False

    def applicable(self, state: Mapping[str, Any]) -> list[SafetySkill]:
        matches = [skill for skill in self.skills.values() if skill.applicable(state)]
        return sorted(matches, key=lambda skill: (-skill.priority, skill.skill_id))

    def route(
        self, state: Mapping[str, Any], *, active_skill_id: str | None = None
    ) -> list[SafetySkill]:
        """Return ordered candidates, preserving a still-valid active skill first."""

        candidates = self.applicable(state)
        if active_skill_id is None:
            return candidates
        active = self.skills.get(active_skill_id)
        if active is None or active.should_terminate(state) or not active.applicable(state):
            return candidates
        return [active, *(skill for skill in candidates if skill.skill_id != active_skill_id)]

    @classmethod
    def from_json(cls, path: str | Path) -> "SkillPool":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        pool = cls()
        for item in payload["skills"]:
            pool.add(
                SafetySkill(
                    skill_id=item["skill_id"],
                    name=item["name"],
                    activation=ConditionGroup.from_dicts(
                        item.get("activation", []), mode=item.get("activation_mode", "all")
                    ),
                    correction_instructions=item["correction_instructions"],
                    fallback_instructions=item.get(
                        "fallback_instructions", "request clarification or abort the subtask"
                    ),
                    termination=ConditionGroup.from_dicts(
                        item.get("termination", []), mode=item.get("termination_mode", "all")
                    ),
                    priority=int(item.get("priority", 0)),
                    version=int(item.get("version", 1)),
                    metadata=item.get("metadata", {}),
                )
            )
        return pool

    def to_json(self, path: str | Path) -> None:
        """Persist the schema-valid Skill pool for deployment or review."""
        payload = {
            "skills": [
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "activation": [
                        {"field": c.field, "op": c.op, "value": c.value}
                        for c in skill.activation.conditions
                    ],
                    "activation_mode": skill.activation.mode,
                    "correction_instructions": skill.correction_instructions,
                    "fallback_instructions": skill.fallback_instructions,
                    "termination": [
                        {"field": c.field, "op": c.op, "value": c.value}
                        for c in skill.termination.conditions
                    ],
                    "termination_mode": skill.termination.mode,
                    "priority": skill.priority,
                    "version": skill.version,
                    "metadata": dict(skill.metadata),
                }
                for skill in self.skills.values()
            ]
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
