import json

import pytest

from g2a.datasets import load_tcsd_examples


def test_load_tcsd_examples(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({
        "state": {"private": True},
        "skill": "privacy",
        "action": "share",
        "unsafe_context_logprobs": [-0.2],
        "safe_context_logprobs": [-0.8],
    }) + "\n", encoding="utf-8")
    examples = load_tcsd_examples(path)
    assert examples[0].target_risk == pytest.approx(0.6)
