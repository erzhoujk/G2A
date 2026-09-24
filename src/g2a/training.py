"""Small, explicit training loops for initial and refinement-stage Guard fitting."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from .guard import guard_huber_loss, guard_refinement_loss, serialize_guard_input


@dataclass(frozen=True, slots=True)
class GuardExample:
    state: Mapping[str, Any] | str
    skill: str
    action: str
    target_risk: float


@dataclass(frozen=True, slots=True)
class GuidePair:
    state: Mapping[str, Any] | str
    skill: str
    corrected_action: str
    original_action: str


def train_initial_guard(
    guard: object,
    examples: Sequence[GuardExample],
    *,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    steps: int = 1000,
    seed: int = 0,
) -> list[float]:
    import torch

    if not examples:
        raise ValueError("at least one TCSD example is required")
    optimizer = torch.optim.AdamW(
        guard.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = random.Random(seed)
    losses: list[float] = []
    guard.train(True)
    for _ in range(steps):
        batch = generator.choices(examples, k=min(batch_size, len(examples)))
        texts = [serialize_guard_input(item.state, item.skill, item.action) for item in batch]
        targets = torch.tensor(
            [item.target_risk for item in batch], dtype=torch.float32, device=guard.device
        )
        loss = guard_huber_loss(guard.forward_texts(texts), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def refine_guard(
    guard: object,
    pairs: Sequence[GuidePair],
    *,
    margin: float = 0.1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    steps: int = 100,
    seed: int = 0,
) -> list[float]:
    import torch

    if not pairs:
        raise ValueError("at least one verified Guide pair is required")
    optimizer = torch.optim.AdamW(
        guard.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = random.Random(seed)
    losses: list[float] = []
    guard.train(True)
    for _ in range(steps):
        batch = generator.choices(pairs, k=min(batch_size, len(pairs)))
        corrected_texts = [
            serialize_guard_input(item.state, item.skill, item.corrected_action) for item in batch
        ]
        original_texts = [
            serialize_guard_input(item.state, item.skill, item.original_action) for item in batch
        ]
        loss = guard_refinement_loss(
            guard.forward_texts(corrected_texts),
            guard.forward_texts(original_texts),
            margin=margin,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses
