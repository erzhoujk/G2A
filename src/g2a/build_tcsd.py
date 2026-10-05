"""CLI for materializing TCSD labels with the frozen Qwen teacher."""

from __future__ import annotations

import argparse

from .teacher import QwenTCSDTeacher, TeacherConfig, iter_jsonl, write_tcsd_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build TCSD labels with Qwen3-32B")
    parser.add_argument("input", help="JSONL records with safe/unsafe teacher references")
    parser.add_argument("output", help="output JSONL consumed by g2a-train-guard")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--device")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    teacher = QwenTCSDTeacher(TeacherConfig(
        model_name=args.model, device=args.device, torch_dtype=args.dtype,
        max_context_tokens=args.max_context_tokens,
    ))
    count = write_tcsd_jsonl(teacher, iter_jsonl(args.input), args.output)
    print(f"wrote {count} TCSD examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
