"""Closed-loop Guard-and-Guide runtime intervention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping, Protocol

from .audit import AuditLog
from .policy import PolicyEngine, PolicyViolation
from .skills import Corrector, SafetySkill, SkillPool


class RiskGuard(Protocol):
    def score(self, state: Mapping[str, Any], skill: str, action: str) -> float: ...


class DecisionKind(str, Enum):
    EXECUTE = "execute"
    CORRECTED = "corrected"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    threshold: float = 0.45
    max_skill_attempts: int = 3
    fail_closed: bool = True
    freeze_skills: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.max_skill_attempts < 1:
            raise ValueError("max_skill_attempts must be positive")


@dataclass(frozen=True, slots=True)
class InterventionAttempt:
    skill_id: str
    revised_action: str | None
    risk: float | None
    violations: tuple[PolicyViolation, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Decision:
    kind: DecisionKind
    action: str | None
    original_action: str
    original_risk: float | None
    final_risk: float | None
    skill_id: str | None
    reason: str
    attempts: tuple[InterventionAttempt, ...] = ()
    audit_hash: str | None = None

    @property
    def may_execute(self) -> bool:
        return self.kind in {DecisionKind.EXECUTE, DecisionKind.CORRECTED}


class G2ARuntime:
    """Screens every pending action and executes only a re-checked safe action.

    Hard policy violations always block. Learned-Guard failures block by default.
    A risky proposal is never executed merely because correction failed.
    """

    def __init__(
        self,
        guard: RiskGuard,
        skills: SkillPool,
        corrector: Corrector,
        *,
        policy: PolicyEngine | None = None,
        config: RuntimeConfig | None = None,
        audit_log: AuditLog | None = None,
    ) -> None:
        self.guard = guard
        self.skills = skills
        self.corrector = corrector
        self.policy = policy or PolicyEngine()
        self.config = config or RuntimeConfig()
        self.audit = audit_log or AuditLog()
        self.active_skill_id: str | None = None
        if self.config.freeze_skills:
            self.skills.freeze()

    def inspect(self, state: Mapping[str, Any], proposed_action: str) -> Decision:
        if not isinstance(proposed_action, str) or not proposed_action.strip():
            return self._block(str(proposed_action), None, "empty action", ())

        try:
            hard_violations = self.policy.check(state, proposed_action)
        except Exception as exc:
            if not self.config.fail_closed:
                raise
            return self._block(proposed_action, None, f"hard policy error: {exc}", ())
        if hard_violations:
            return self._block(proposed_action, None, "hard policy violation", (), hard_violations)

        try:
            original_risk = self._score(state, self._active_context(), proposed_action)
        except Exception as exc:
            if not self.config.fail_closed:
                raise
            return self._block(proposed_action, None, f"guard error: {exc}", ())

        if original_risk <= self.config.threshold:
            if self._active_skill_should_end(state):
                self.active_skill_id = None
            return self._decision(
                DecisionKind.EXECUTE,
                action=proposed_action,
                original_action=proposed_action,
                original_risk=original_risk,
                final_risk=original_risk,
                skill_id=self.active_skill_id,
                reason="proposal passed Guard and hard policy",
                attempts=(),
            )

        attempts: list[InterventionAttempt] = []
        try:
            candidates = self.skills.route(state, active_skill_id=self.active_skill_id)
        except Exception as exc:
            if not self.config.fail_closed:
                raise
            return self._block(
                proposed_action, original_risk, f"Guide routing error: {exc}", tuple(attempts)
            )
        for skill in candidates[: self.config.max_skill_attempts]:
            attempt = self._try_skill(state, proposed_action, skill)
            attempts.append(attempt)
            if attempt.error or attempt.violations or attempt.risk is None:
                continue
            if attempt.risk <= self.config.threshold and attempt.revised_action is not None:
                self.active_skill_id = skill.skill_id
                return self._decision(
                    DecisionKind.CORRECTED,
                    action=attempt.revised_action,
                    original_action=proposed_action,
                    original_risk=original_risk,
                    final_risk=attempt.risk,
                    skill_id=skill.skill_id,
                    reason="Guide correction passed hard policy and second Guard check",
                    attempts=tuple(attempts),
                )

        reason = (
            "no applicable Safety Skill" if not candidates else "all corrections remained unsafe"
        )
        return self._block(proposed_action, original_risk, reason, tuple(attempts))

    def _try_skill(
        self, state: Mapping[str, Any], proposed_action: str, skill: SafetySkill
    ) -> InterventionAttempt:
        try:
            revised = skill.revise(state, proposed_action, self.corrector)
            violations = self.policy.check(state, revised)
            if violations:
                return InterventionAttempt(skill.skill_id, revised, None, violations)
            risk = self._score(state, skill.as_guard_context(), revised)
            return InterventionAttempt(skill.skill_id, revised, risk, ())
        except Exception as exc:
            if not self.config.fail_closed:
                raise
            return InterventionAttempt(skill.skill_id, None, None, (), str(exc))

    def _score(self, state: Mapping[str, Any], skill: str, action: str) -> float:
        risk = float(self.guard.score(state, skill, action))
        if not math.isfinite(risk):
            raise ValueError("Guard returned a non-finite score")
        return risk

    def _active_context(self) -> str:
        active = self.skills.skills.get(self.active_skill_id) if self.active_skill_id else None
        return active.as_guard_context() if active else "no active safety skill"

    def _active_skill_should_end(self, state: Mapping[str, Any]) -> bool:
        active = self.skills.skills.get(self.active_skill_id) if self.active_skill_id else None
        return active is not None and active.should_terminate(state)

    def _block(
        self,
        action: str,
        original_risk: float | None,
        reason: str,
        attempts: tuple[InterventionAttempt, ...],
        violations: tuple[PolicyViolation, ...] = (),
    ) -> Decision:
        if violations:
            attempts = (*attempts, InterventionAttempt("hard-policy", action, None, violations))
        return self._decision(
            DecisionKind.BLOCK,
            action=None,
            original_action=action,
            original_risk=original_risk,
            final_risk=None,
            skill_id=None,
            reason=reason,
            attempts=tuple(attempts),
        )

    def _decision(self, kind: DecisionKind, **kwargs: Any) -> Decision:
        payload = {"kind": kind.value, **kwargs, "attempts": [str(x) for x in kwargs["attempts"]]}
        event = self.audit.append("runtime_decision", payload)
        return Decision(kind=kind, audit_hash=event.event_hash, **kwargs)
