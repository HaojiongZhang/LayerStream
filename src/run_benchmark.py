"""
Evaluate LayerStream on open multiple-choice benchmarks (e.g. GPQA, ARC).

Example::

    export HF_TOKEN=hf_...   # GPQA is gated; accept the license on the Hub first
    python -m src.run_benchmark --dataset gpqa_diamond --model Qwen/Qwen2.5-7B-Instruct --limit 50

Requires ``datasets`` (see src/requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from .benchmark_datasets import build_prompt_for_example, iter_arc_challenge_mc, iter_gpqa_mc
from .mcq_utils import letter_choice_token_ids, predict_letters_from_logits
from .profiler import Profiler
from .run_infer import build_inference_stack


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LayerStream — accuracy on GPQA / ARC-Challenge style MC sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        choices=["gpqa_main", "gpqa_diamond", "gpqa_extended", "arc_challenge"],
        default="gpqa_diamond",
        help="Open MC benchmark to run.",
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-2-13b-chat-hf",
        help="HuggingFace model id or local path.",
    )
    p.add_argument("--mode", choices=["layered", "vram"], default="layered")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--buffer-depth", type=int, default=2)
    p.add_argument("--max-gpu-kv-gb", type=float, default=2.0)
    p.add_argument("--kv-cache-bits", type=int, choices=[16, 8, 4], default=16)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--no-profile", action="store_true", help="Disable CUDA event profiling in the engine.")
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable the sampling profiler during the eval loop: GPU memory, KV-cache "
            "bytes, compute utilization (via pynvml), and per-prediction latency. "
            "Writes logs/profile_samples.jsonl + logs/profile_summary.json."
        ),
    )
    p.add_argument(
        "--profile-hz",
        type=float,
        default=20.0,
        help="Sampling rate for --profile (Hz).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="B",
        help="Micro-batch size for forward (right-padded tokenization).",
    )
    p.add_argument("--limit", type=int, default=None, help="Evaluate at most this many items.")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed for dataset order.")
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write aggregate metrics and per-item predictions to this path.",
    )
    p.add_argument(
        "--system-preamble",
        default=None,
        help="Optional text prepended to each item (e.g. short system instruction).",
    )
    p.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help=(
            "Hugging Face read token for gated datasets (GPQA). "
            "If unset, uses HF_TOKEN / HUGGING_FACE_HUB_TOKEN env or ``huggingface-cli login`` cache."
        ),
    )
    ns = p.parse_args()
    if ns.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    return ns


def _examples(
    args: argparse.Namespace,
) -> Iterator[tuple[str, str, list[str], str]]:
    if args.dataset == "arc_challenge":
        yield from iter_arc_challenge_mc(seed=args.seed)
        return
    yield from iter_gpqa_mc(subset=args.dataset, seed=args.seed)


def main() -> None:
    args = parse_args()
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")

    generator, parts, _logger, _cfg, _ = build_inference_stack(
        model_name_or_path=args.model,
        mode=args.mode,
        dtype=args.dtype,
        buffer_depth=args.buffer_depth,
        max_gpu_kv_gb=args.max_gpu_kv_gb,
        kv_cache_bits=args.kv_cache_bits,
        max_seq_len=args.max_seq_len,
        log_dir=Path(args.log_dir),
        no_profile=args.no_profile,
        batch_size=args.batch_size,
    )

    tok = parts.tokenizer
    letter_ids = letter_choice_token_ids(tok)
    letters = ("A", "B", "C", "D")

    correct = 0
    total = 0
    predictions: list[dict[str, Any]] = []
    prompt_tokens_total = 0

    buf: list[tuple[str, str, list[str], str]] = []
    req_id = 0

    profiler: Profiler | None = None
    if args.profile:
        profiler = Profiler(
            log_dir=Path(args.log_dir),
            kv_manager=generator.kv_manager,
            sample_hz=args.profile_hz,
            request_id=0,
        )
        profiler.start()

    def flush_batch(batch: list[tuple[str, str, list[str], str]]) -> None:
        nonlocal correct, total, req_id, prompt_tokens_total
        if not batch:
            return
        prompts = [
            build_prompt_for_example(
                question,
                choices,
                system_preamble=args.system_preamble,
            )
            for _ex_id, _gold, choices, question in batch
        ]
        req_id += 1
        if profiler is not None:
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                padding_side="right",
            )
            prompt_tokens_total += int(enc["input_ids"].numel())
        logits = generator.logits_after_prompts(request_id=req_id, prompts=prompts)
        preds = predict_letters_from_logits(logits, letter_ids, letters)
        for (_ex_id, gold, _choices, _question), pred in zip(batch, preds):
            is_correct = pred == gold
            correct += int(is_correct)
            total += 1
            predictions.append(
                {
                    "id": _ex_id,
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                }
            )

    try:
        for row in _examples(args):
            if args.limit is not None and total + len(buf) >= args.limit:
                break
            buf.append(row)
            if len(buf) >= args.batch_size:
                flush_batch(buf)
                buf.clear()

        flush_batch(buf)
    finally:
        if profiler is not None:
            profiler.stop()

    acc = correct / total if total else 0.0
    print(
        f"Dataset: {args.dataset}  model: {args.model}  "
        f"batch_size={args.batch_size}  n={total}  accuracy={acc:.4f}"
    )

    if profiler is not None:
        # Benchmark is prefill-only: one forward per batch yielding letter-logits.
        # Count each prediction as 1 "generated" token so tokens_per_s reports
        # predictions/sec; prompt_len is the aggregate prefill token count.
        summary = profiler.summary(
            prompt_len=prompt_tokens_total,
            generated_tokens=total,
        )
        profiler.write_summary(summary)
        profiler.print_summary(summary)

    if args.output_json is not None:
        payload = {
            "dataset": args.dataset,
            "model": args.model,
            "batch_size": args.batch_size,
            "n": total,
            "correct": correct,
            "accuracy": acc,
            "items": predictions,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
