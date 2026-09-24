"""Trajectory-Contrastive Safety Distillation evidence (paper equation 2)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class TCSDEvidence:
    """Risk evidence for one fixed candidate action.

    Positive values mean that the action is more compatible with the matched
    unsafe trajectory context than with the safe context.
    """

    risk: float
    token_count: int
    token_differences: tuple[float, ...]


def compute_tcsd_evidence(
    unsafe_context_logprobs: Sequence[float] | Iterable[float],
    safe_context_logprobs: Sequence[float] | Iterable[float],
) -> TCSDEvidence:
    """Average log p(action|unsafe context) - log p(action|safe context).

    Both inputs must score exactly the same realized action token IDs. Prompt,
    padding, and reference-trajectory tokens must already be excluded.
    """

    unsafe = tuple(float(value) for value in unsafe_context_logprobs)
    safe = tuple(float(value) for value in safe_context_logprobs)
    if not unsafe:
        raise ValueError("an action must contain at least one scored token")
    if len(unsafe) != len(safe):
        raise ValueError("safe and unsafe scores must cover identical action tokens")
    if not all(math.isfinite(value) for value in unsafe + safe):
        raise ValueError("token log-probabilities must be finite")
    differences = tuple(u - s for u, s in zip(unsafe, safe, strict=True))
    return TCSDEvidence(
        risk=math.fsum(differences) / len(differences),
        token_count=len(differences),
        token_differences=differences,
    )
