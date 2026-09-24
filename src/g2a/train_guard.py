"""Train the MiniLM Guard from precomputed or token-level TCSD JSONL data."""

from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import load_tcsd_examples
from .guard import MiniLMGuard
from .training import train_initial_guard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a G2A MiniLM Guard")
    parser.add_argument("input", help="TCSD supervision JSONL")
    parser.add_argument("-o", "--output", default="checkpoints/g2a-guard.pt")
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2", help="encoder name"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    import torch

    examples = load_tcsd_examples(args.input)
    guard = MiniLMGuard(args.model, device=args.device)
    losses = train_initial_guard(
        guard,
        examples,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        steps=args.steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "state_dict": guard.state_dict(),
            "steps": args.steps,
            "final_loss": losses[-1],
        },
        output,
    )
    print(f"saved Guard checkpoint to {output} (final loss: {losses[-1]:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
