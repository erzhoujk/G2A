"""Optional MiniLM Guard and training objectives."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    import torch


def serialize_guard_input(state: Mapping[str, Any] | str, skill: str, action: str) -> str:
    """Serialize [CLS] state [SEP] Skill [SEP] action without adding model tokens."""

    state_text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, default=str)
    return f"state: {state_text}\nskill: {skill}\naction: {action}"


class MiniLMGuard:
    """Six-layer MiniLM encoder with a scalar risk head.

    ``sentence-transformers/all-MiniLM-L6-v2`` matches the six-layer architecture
    reported by the paper. The model is loaded lazily, so the core runtime stays
    dependency-free.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.risk_head = torch.nn.Linear(hidden_size, 1)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder.to(self.device)
        self.risk_head.to(self.device)

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.risk_head.parameters()

    def train(self, mode: bool = True) -> "MiniLMGuard":
        self.encoder.train(mode)
        self.risk_head.train(mode)
        return self

    def eval(self) -> "MiniLMGuard":
        return self.train(False)

    def forward_texts(self, texts: Sequence[str]) -> "torch.Tensor":
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        outputs = self.encoder(**batch)
        cls = outputs.last_hidden_state[:, 0]
        return self.risk_head(cls).squeeze(-1)

    def score(self, state: Mapping[str, Any], skill: str, action: str) -> float:
        text = serialize_guard_input(state, skill, action)
        self.eval()
        with self.torch.no_grad():
            return float(self.forward_texts([text]).item())

    def state_dict(self) -> dict[str, Any]:
        return {"encoder": self.encoder.state_dict(), "risk_head": self.risk_head.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.risk_head.load_state_dict(state["risk_head"])


def guard_huber_loss(
    predicted_risk: "torch.Tensor", tcsd_targets: "torch.Tensor"
) -> "torch.Tensor":
    """Equation (3): Huber regression from Guard scores to TCSD evidence."""

    import torch.nn.functional as functional

    if predicted_risk.shape != tcsd_targets.shape:
        raise ValueError("predictions and TCSD targets must have equal shape")
    return functional.huber_loss(predicted_risk, tcsd_targets)


def guard_refinement_loss(
    corrected_risk: "torch.Tensor", original_risk: "torch.Tensor", *, margin: float = 0.1
) -> "torch.Tensor":
    """Equation (6): corrected actions must be lower-risk by at least ``margin``."""

    import torch

    if corrected_risk.shape != original_risk.shape:
        raise ValueError("corrected and original risk tensors must have equal shape")
    if margin <= 0:
        raise ValueError("margin must be positive")
    return torch.clamp(corrected_risk - original_risk + margin, min=0.0).mean()
