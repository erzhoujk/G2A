"""Qwen teacher integration for trajectory-contrastive supervision.

The teacher is deliberately isolated from the Guard.  It scores the *same
realized action token ids* under matched safe and unsafe trajectory contexts,
then emits the length-normalized TCSD evidence used by MiniLM training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .scoring import score_action_tokens
from .tcsd import TCSDEvidence, compute_tcsd_evidence


@dataclass(frozen=True, slots=True)
class TeacherConfig:
    model_name: str = "Qwen/Qwen3-32B"
    device: str | None = None
    torch_dtype: str = "bfloat16"
    max_context_tokens: int = 8192


class QwenTCSDTeacher:
    """Frozen Hugging Face causal LM used as the paper's TCSD scorer.

    Loading is lazy with respect to the optional ``torch`` and ``transformers``
    dependencies.  The class therefore remains importable in the dependency-free
    runtime, while production training can install ``g2a-safety[train]``.
    """

    def __init__(self, config: TeacherConfig = TeacherConfig(), *, model: object | None = None) -> None:
        self.config = config
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "Qwen TCSD scoring requires the optional train dependencies: "
                "pip install -e '.[train]'"
            ) from exc
        self.torch = torch
        dtype = getattr(torch, config.torch_dtype, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {config.torch_dtype}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        self.model = model or AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=dtype, device_map="auto"
        )
        self.model.eval()

    def _joined_inputs(self, contexts: Sequence[str], action: str):
        """Build padded context+action batches with an identical action suffix."""
        if not action:
            raise ValueError("candidate action must not be empty")
        rows: list[list[int]] = []
        masks: list[list[bool]] = []
        for context in contexts:
            context_ids = self.tokenizer.encode(
                context, add_special_tokens=True, truncation=True,
                max_length=self.config.max_context_tokens,
            )
            action_ids = self.tokenizer.encode(action, add_special_tokens=False)
            if not action_ids:
                raise ValueError("candidate action produced no tokens")
            rows.append(context_ids + action_ids)
            masks.append([False] * len(context_ids) + [True] * len(action_ids))
        max_len = max(map(len, rows))
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = self.torch.full((len(rows), max_len), pad_id, dtype=self.torch.long)
        attention = self.torch.zeros_like(input_ids)
        action_mask = self.torch.zeros_like(input_ids, dtype=self.torch.bool)
        for index, (row, mask) in enumerate(zip(rows, masks, strict=True)):
            input_ids[index, : len(row)] = self.torch.tensor(row)
            attention[index, : len(row)] = 1
            action_mask[index, : len(mask)] = self.torch.tensor(mask)
        device = next(self.model.parameters()).device
        return input_ids.to(device), attention.to(device), action_mask.to(device)

    def score_action(self, context: str, action: str) -> list[float]:
        input_ids, attention, action_mask = self._joined_inputs([context], action)
        return score_action_tokens(self.model, input_ids, attention, action_mask)[0]

    def evidence(self, *, state: Mapping[str, Any] | str, action: str,
                 safe_reference: str, unsafe_reference: str) -> TCSDEvidence:
        state_text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, default=str)
        safe_scores, unsafe_scores = self.score_action(
            f"{state_text}\nSAFE_REFERENCE:\n{safe_reference}", action
        ), self.score_action(f"{state_text}\nUNSAFE_REFERENCE:\n{unsafe_reference}", action)
        return compute_tcsd_evidence(unsafe_scores, safe_scores)


def write_tcsd_jsonl(
    teacher: QwenTCSDTeacher, records: Iterable[Mapping[str, Any]], output: str | Path
) -> int:
    """Materialize paper-style TCSD supervision from trajectory records.

    Each input record must contain ``state``, ``action``, ``safe_reference`` and
    ``unsafe_reference``; ``skill`` is optional.  References are privileged teacher
    context and are never passed to the MiniLM Guard at training or deployment.
    """
    count = 0
    with Path(output).open("w", encoding="utf-8") as handle:
        for record in records:
            evidence = teacher.evidence(
                state=record["state"], action=str(record["action"]),
                safe_reference=str(record["safe_reference"]),
                unsafe_reference=str(record["unsafe_reference"]),
            )
            payload = {
                "state": record["state"], "skill": str(record.get("skill", "no active safety skill")),
                "action": str(record["action"]), "target_risk": evidence.risk,
                "token_count": evidence.token_count,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count


def iter_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"line {line_number} must contain an object")
                yield record
