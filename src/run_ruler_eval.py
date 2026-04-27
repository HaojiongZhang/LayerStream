from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from .config import (
    AppConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    OffloadConfig,
)
from .generation import Generator
from .logger import ExperimentLogger
from .model_loader import load_model_parts
from .profiler import Profiler

# Reuse your existing engine builders from run_infer.py
from .run_infer import _build_layered_engine, _build_vram_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate LayerStream on RULER/NIAH JSONL tasks."
    )

    p.add_argument("--model", default="meta-llama/Llama-2-13b-chat-hf")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--task", default="niah")
    p.add_argument("--output-path", default=None)

    p.add_argument("--mode", choices=["layered", "vram"], default="layered")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")

    p.add_argument("--buffer-depth", type=int, default=2)
    p.add_argument("--max-gpu-kv-gb", type=float, default=2.0)
    p.add_argument("--kv-cache-bits", type=int, choices=[16, 8, 4], default=16)
    p.add_argument("--max-seq-len", type=int, default=8192)

    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--no-profile", action="store_true")

    p.add_argument(
        "--profile",
        action="store_true",
        help="Enable your sampling profiler for the whole eval run.",
    )
    p.add_argument("--profile-hz", type=float, default=20.0)

    return p.parse_args()


def load_jsonl(path: str | Path, num_samples: int | None = None) -> list[dict[str, Any]]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
            if num_samples is not None and len(examples) >= num_samples:
                break
    return examples


def get_prompt(example: dict[str, Any]) -> str:
    """
    Handles common RULER/NIAH JSONL field names.
    Adjust this if your generated RULER files use different keys.
    """
    for key in ["input", "prompt", "context", "text"]:
        if key in example:
            return str(example[key])

    raise KeyError(
        f"Could not find prompt field. Available keys: {list(example.keys())}"
    )


def get_answers(example: dict[str, Any]) -> list[str]:
    """
    Handles common RULER/NIAH answer formats.
    """
    for key in ["answer", "answers", "target", "targets", "outputs"]:
        if key in example:
            value = example[key]
            if isinstance(value, list):
                return [str(x) for x in value]
            return [str(value)]

    raise KeyError(
        f"Could not find answer field. Available keys: {list(example.keys())}"
    )


def normalize_text(s: str) -> str:
    return " ".join(s.lower().strip().split())


def exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize_text(prediction)

    for ans in answers:
        ans_norm = normalize_text(ans)

        # RULER/NIAH often only needs the needle value to appear.
        if ans_norm == pred:
            return True
        if ans_norm in pred:
            return True

    return False


def build_generator(args: argparse.Namespace):
    cfg = AppConfig(
        model=ModelConfig(
            model_name_or_path=args.model,
            dtype=args.dtype,
        ),
        offload=OffloadConfig(
            buffer_depth=args.buffer_depth,
            max_gpu_kv_bytes=int(args.max_gpu_kv_gb * 1024**3),
            kv_cache_bits=args.kv_cache_bits,
            vram_only=(args.mode == "vram"),
            profile_cuda=not args.no_profile,
        ),
        generation=GenerationConfig(max_new_tokens=args.max_new_tokens),
        logging=LoggingConfig(log_dir=Path(args.log_dir)),
    )

    print(
        f"Loading model: {cfg.model.model_name_or_path} "
        f"(dtype={cfg.model.dtype}, mode={args.mode}, kv_bits={args.kv_cache_bits})"
    )

    logger = ExperimentLogger(cfg.logging.log_dir)

    load_device = cfg.model.device if cfg.offload.vram_only else "cpu"
    parts = load_model_parts(
        cfg.model.model_name_or_path,
        dtype=cfg.model.dtype,
        device=load_device,
    )

    torch_dtype = getattr(torch, cfg.model.dtype)

    if cfg.offload.vram_only:
        engine, kv_manager = _build_vram_engine(
            parts,
            cfg,
            logger,
            args.max_seq_len,
        )
    else:
        engine, kv_manager = _build_layered_engine(
            parts,
            cfg,
            logger,
            torch_dtype,
            args.max_seq_len,
        )

    generator = Generator(
        tokenizer=parts.tokenizer,
        engine=engine,
        kv_manager=kv_manager,
        logger=logger,
        eos_token_id=parts.tokenizer.eos_token_id,
    )

    return cfg, parts, generator, kv_manager


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")

    dataset_path = Path(args.dataset_path)
    examples = load_jsonl(dataset_path, args.num_samples)

    print(f"Loaded {len(examples)} examples from {dataset_path}")

    if args.output_path is None:
        output_path = (
            Path(args.log_dir)
            / f"ruler_{args.task}_kv{args.kv_cache_bits}_seq{args.max_seq_len}.csv"
        )
    else:
        output_path = Path(args.output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg, parts, generator, kv_manager = build_generator(args)

    profiler: Profiler | None = None
    if args.profile:
        profiler = Profiler(
            log_dir=cfg.logging.log_dir,
            kv_manager=kv_manager,
            sample_hz=args.profile_hz,
            request_id=0,
        )
        profiler.start()

    rows = []
    num_correct = 0

    try:
        for i, ex in enumerate(examples):
            prompt = get_prompt(ex)
            answers = get_answers(ex)

            prompt_tokens = int(
                parts.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            )

            start = time.perf_counter()

            result = generator.generate(
                request_id=i + 1,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
            )

            elapsed = time.perf_counter() - start

            prediction = result.text
            generated_tokens = len(result.generated_ids)
            correct = exact_match(prediction, answers)

            num_correct += int(correct)

            row = {
                "task": args.task,
                "sample_id": i,
                "model": args.model,
                "mode": args.mode,
                "dtype": args.dtype,
                "kv_cache_bits": args.kv_cache_bits,
                "max_seq_len": args.max_seq_len,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "latency_s": elapsed,
                "tokens_per_s": generated_tokens / elapsed if elapsed > 0 else 0.0,
                "correct": int(correct),
                "prediction": prediction,
                "answers": json.dumps(answers, ensure_ascii=False),
            }

            rows.append(row)

            print(
                f"[{i + 1}/{len(examples)}] "
                f"correct={correct} "
                f"prompt_tokens={prompt_tokens} "
                f"gen_tokens={generated_tokens} "
                f"latency={elapsed:.2f}s"
            )

    finally:
        if profiler is not None:
            profiler.stop()

    accuracy = num_correct / len(examples) if examples else 0.0

    print("\n=== Eval Summary ===")
    print(f"Task: {args.task}")
    print(f"KV cache bits: {args.kv_cache_bits}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Examples: {len(examples)}")
    print(f"Correct: {num_correct}")
    print(f"Accuracy: {accuracy:.4f}")

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote per-example results to: {output_path}")

    if profiler is not None:
        total_prompt_tokens = sum(r["prompt_tokens"] for r in rows)
        total_generated_tokens = sum(r["generated_tokens"] for r in rows)

        summary = profiler.summary(
            prompt_len=total_prompt_tokens,
            generated_tokens=total_generated_tokens,
        )
        profiler.write_summary(summary)
        profiler.print_summary(summary)


if __name__ == "__main__":
    main()