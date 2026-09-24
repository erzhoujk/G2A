"""Dependency-free demo Guard and corrector; not a substitute for trained TCSD."""

from __future__ import annotations

from typing import Any, Mapping

from .skills import SafetySkill


class KeywordRiskGuard:
    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self.weights = dict(
            weights
            or {
                "public link": 0.8,
                "delete": 0.7,
                "password": 0.65,
                "secret": 0.65,
                "external": 0.5,
                "private link": -0.45,
                "redact": -0.4,
                "confirm": -0.35,
            }
        )

    def score(self, state: Mapping[str, Any], skill: str, action: str) -> float:
        text = f"{state} {skill} {action}".casefold()
        return max(0.0, min(1.0, 0.1 + sum(v for key, v in self.weights.items() if key in text)))


def instruction_corrector(
    _state: Mapping[str, Any], action: str, skill: SafetySkill
) -> str:
    """Toy corrector used by the CLI example."""

    if "public link" in action.casefold():
        return action.lower().replace("public link", "private link after confirm")
    return f"Apply safety procedure ({skill.correction_instructions}); then: {action}"
