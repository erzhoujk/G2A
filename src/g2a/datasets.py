"""JSONL adapters for TCSD supervision and verified Guide correction pairs."""

from __future__ import annotations

import json
from pathlib import Path

from .tcsd import compute_tcsd_evidence
from .training import GuardExample, GuidePair


def load_tcsd_examples(path: str | Path) -> list[GuardExample]:
    examples: list[GuardExample] = []
    for line_number, record in _records(path):
        try:
            if "target_risk" in record:
                target = float(record["target_risk"])
            else:
                target = compute_tcsd_evidence(
                    record["unsafe_context_logprobs"], record["safe_context_logprobs"]
                ).risk
            examples.append(
                GuardExample(
                    state=record["state"],
                    skill=str(record.get("skill", "no active safety skill")),
                    action=str(record["action"]),
                    target_risk=target,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid TCSD record on line {line_number}: {exc}") from exc
    return examples


def load_guide_pairs(path: str | Path) -> list[GuidePair]:
    pairs: list[GuidePair] = []
    for line_number, record in _records(path):
        try:
            pairs.append(
                GuidePair(
                    state=record["state"],
                    skill=str(record["skill"]),
                    corrected_action=str(record["corrected_action"]),
                    original_action=str(record["original_action"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Guide pair on line {line_number}: {exc}") from exc
    return pairs


def _records(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            yield line_number, record
