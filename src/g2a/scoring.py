"""Teacher-forced fixed-action scoring for TCSD."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def score_action_tokens(
    model: object,
    input_ids: "torch.Tensor",
    attention_mask: "torch.Tensor",
    action_mask: "torch.Tensor",
) -> list[list[float]]:
    """Return causal token log-probabilities only where ``action_mask`` is true.

    Call once with the unsafe reference context and once with the matched safe
    context, while preserving identical action token IDs in both batches.
    """

    import torch
    import torch.nn.functional as functional

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must have shape [batch, tokens]")
    if action_mask.shape != input_ids.shape:
        raise ValueError("action_mask must match input_ids")
    if torch.any(action_mask[:, 0]):
        raise ValueError("a causal LM cannot score the first input token")
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logprobs = functional.log_softmax(outputs.logits[:, :-1], dim=-1)
        target_ids = input_ids[:, 1:]
        selected = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        selected_mask = action_mask[:, 1:].bool() & attention_mask[:, 1:].bool()
    rows: list[list[float]] = []
    for values, mask in zip(selected, selected_mask, strict=True):
        if not torch.any(mask):
            raise ValueError("every row must contain at least one action token")
        rows.append(values[mask].detach().cpu().tolist())
    return rows
