from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import (
    AppConfig,
    GenerationConfig,
    LoggingConfig,
    ModelConfig,
    OffloadConfig,
)
from .cpu_weight_store import MmapWeightStore
from .forward_engine import ForwardEngine
from .generation import Generator
from .gpu_buffer_pool import GPUBufferPool
from .kv_manager import KVCacheManager
from .logger import ExperimentLogger
from .model_loader import LoadedModelParts, load_model_parts
from .prefetcher import LayerPrefetcher
from .profiler import Profiler
from .streams import StreamManager


def build_inference_stack(
    *,
    model_name_or_path: str,
    mode: str,
    dtype: str,
    buffer_depth: int,
    max_gpu_kv_gb: float,
    kv_cache_bits: int,
    max_seq_len: int,
    log_dir: Path,
    no_profile: bool,
    batch_size: int = 1,
) -> tuple[Generator, LoadedModelParts, ExperimentLogger, AppConfig, torch.dtype]:
    """
    Load weights, construct the forward engine + KV manager, and return a
    ``Generator`` ready for ``generate()`` / ``logits_after_prompt()``.
    """
    cfg = AppConfig(
        model=ModelConfig(
            model_name_or_path=model_name_or_path,
            dtype=dtype,
        ),
        offload=OffloadConfig(
            buffer_depth=buffer_depth,
            max_gpu_kv_bytes=int(max_gpu_kv_gb * 1024**3),
            kv_cache_bits=kv_cache_bits,
            vram_only=(mode == "vram"),
            profile_cuda=not no_profile,
        ),
        generation=GenerationConfig(max_new_tokens=64, batch_size=batch_size),
        logging=LoggingConfig(log_dir=log_dir),
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
        engine, kv_manager = _build_vram_engine(parts, cfg, logger, max_seq_len)
    else:
        engine, kv_manager = _build_layered_engine(
            parts,
            cfg,
            logger,
            torch_dtype,
            max_seq_len,
        )

    eos_id = parts.tokenizer.eos_token_id
    generator = Generator(
        tokenizer=parts.tokenizer,
        engine=engine,
        kv_manager=kv_manager,
        logger=logger,
        eos_token_id=eos_id,
        batch_size=batch_size,
    )
    return generator, parts, logger, cfg, torch_dtype


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LayerStream inference — double-buffered layer streaming or full-VRAM mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-2-13b-chat-hf",
        help=(
            "HuggingFace model name or path to a local checkpoint. "
            "For Llama-2 you must first accept the license and run "
            "'huggingface-cli login'."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["layered", "vram"],
        default="layered",
        help=(
            "layered: stream decoder layers from CPU with double-buffered "
            "weight + KV prefetch (low VRAM, e.g. 8–16 GB). "
            "vram: keep all layers on GPU at once (needs full model VRAM)."
        ),
    )
    p.add_argument("--prompt", default="Hello, I am a language model,")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="B",
        help="Decode the same prompt in parallel on B rows (uniform batch).",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument(
        "--buffer-depth",
        type=int,
        default=2,
        help="Number of GPU weight-buffer slots (layered mode only). 2 = classic double-buffer.",
    )
    p.add_argument(
        "--max-gpu-kv-gb",
        type=float,
        default=2.0,
        help="Maximum GPU memory reserved for KV cache (GB, layered mode only).",
    )
    p.add_argument(
        "--kv-cache-bits",
        type=int,
        choices=[16, 8, 4],
        default=16,
        help=(
            "Quantize CPU-offloaded KV cache to this bit-width. "
            "16 = no KV quantization, 8 = symmetric int8 KV on CPU."
        ),
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length for static KV cache allocation.",
    )
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--no-profile", action="store_true", help="Disable CUDA event profiling.")
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Enable the sampling profiler: GPU memory, KV-cache bytes, "
            "compute utilization (via pynvml), and token throughput/latency. "
            "Writes logs/profile_samples.jsonl + logs/profile_summary.json."
        ),
    )
    p.add_argument(
        "--profile-hz",
        type=float,
        default=20.0,
        help="Sampling rate for --profile (Hz).",
    )
    ns = p.parse_args()
    if ns.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    return ns


# ---------------------------------------------------------------------------
# Engine builders
# ---------------------------------------------------------------------------

def _build_vram_engine(
    parts,
    cfg: AppConfig,
    logger: ExperimentLogger,
    max_seq_len: int,
) -> tuple[ForwardEngine, KVCacheManager]:
    """All decoder layers already on GPU (loaded there by load_model_parts)."""
    print(f"[vram] Using {len(parts.layers)} layers already on GPU.")
    vram_layers = [layer.eval() for layer in parts.layers]

    kv_manager = KVCacheManager(
        num_layers=len(vram_layers),
        max_gpu_kv_bytes=cfg.offload.max_gpu_kv_bytes,
        max_seq_len=max_seq_len,
        enable_cpu_offload=False,
        pin_cpu_memory=False,
        kv_cache_bits=cfg.offload.kv_cache_bits,
    )

    engine = ForwardEngine(
        embed_tokens=parts.embed_tokens,
        final_norm=parts.final_norm,
        lm_head=parts.lm_head,
        kv_manager=kv_manager,
        logger=logger,
        vram_layers=vram_layers,
        rotary_emb=parts.rotary_emb,
        profile_cuda=cfg.offload.profile_cuda,
    )
    return engine, kv_manager



def _build_layered_engine(
    parts,
    cfg: AppConfig,
    logger: ExperimentLogger,
    torch_dtype: torch.dtype,
    max_seq_len: int,
) -> tuple[ForwardEngine, KVCacheManager]:
    """
    Double-buffered streaming mode.

    Weight double-buffer:
        two GPU slots (buffer_depth=2) alternate so one is being computed
        while the next is copied from the safetensors mmap.

    KV double-buffer:
        the KV prefetch for layer N+1 is issued concurrently with compute for
        layer N, so the KV H2D copy overlaps with compute.

    Weight store:
        MmapWeightStore reads directly from the on-disk safetensors files via mmap.
        Tensors are file-backed pages (evictable under pressure) rather than
        anonymous/pinned shmem, so the model size does not count against the
        cgroup anonymous memory limit.
    """
    # GPUBufferPool deep-copies the prototype layer for the buffer shapes.
    # Must happen BEFORE we free layer params below.
    gpu_pool = GPUBufferPool(
        prototype_layer=parts.layers[0],
        buffer_depth=cfg.offload.buffer_depth,
        device=cfg.model.device,
        dtype=torch_dtype,
    )

    # Free the in-memory layer params now that GPUBufferPool has its copy.
    # from_pretrained with low_cpu_mem_usage already loads weights as
    # file-backed pages; this just drops those references explicitly.
    for layer in parts.layers:
        for p in layer.parameters():
            p.data = torch.empty(0, dtype=p.dtype)

    print(f"[layered] Opening {len(parts.layers)} layers via mmap from {parts.model_path} …")
    weight_store = MmapWeightStore(parts.model_path)
    streams = StreamManager()
    prefetcher = LayerPrefetcher(
        weight_store=weight_store,
        gpu_pool=gpu_pool,
        copy_stream=streams.copy_stream,
        logger=logger,
        profile_cuda=cfg.offload.profile_cuda,
    )

    kv_manager = KVCacheManager(
        num_layers=weight_store.num_layers(),
        max_gpu_kv_bytes=cfg.offload.max_gpu_kv_bytes,
        max_seq_len=max_seq_len,
        enable_cpu_offload=cfg.offload.enable_kv_cpu_offload,
        pin_cpu_memory=False,  # no shmem for KV offload on constrained cgroups
        kv_cache_bits=cfg.offload.kv_cache_bits,
    )

    engine = ForwardEngine(
        embed_tokens=parts.embed_tokens,
        final_norm=parts.final_norm,
        lm_head=parts.lm_head,
        gpu_pool=gpu_pool,
        prefetcher=prefetcher,
        streams=streams,
        kv_manager=kv_manager,
        logger=logger,
        rotary_emb=parts.rotary_emb,
        profile_cuda=cfg.offload.profile_cuda,
    )
    return engine, kv_manager


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    print(
        f"Loading model: {args.model} "
        f"(dtype={args.dtype}, mode={args.mode})"
    )
    generator, parts, logger, cfg, _torch_dtype = build_inference_stack(
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
    cfg.generation.max_new_tokens = args.max_new_tokens
    cfg.generation.batch_size = args.batch_size

    print(f"Generating (mode={args.mode}, max_new_tokens={args.max_new_tokens}) …")

    request_id = 1
    profiler: Profiler | None = None
    if args.profile:
        profiler = Profiler(
            log_dir=cfg.logging.log_dir,
            kv_manager=generator.kv_manager,
            sample_hz=args.profile_hz,
            request_id=request_id,
        )
        profiler.start()

    try:
        result = generator.generate(
            request_id=request_id,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        if profiler is not None:
            profiler.stop()

    print("\n--- Output ---")
    print(result.text)

    if profiler is not None:
        enc = parts.tokenizer(
            [args.prompt] * args.batch_size,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        prompt_len = int(enc["input_ids"].shape[1])
        gen_per_row = len(result.generated_ids[0]) if result.generated_ids else 0
        summary = profiler.summary(
            prompt_len=prompt_len,
            generated_tokens=gen_per_row * args.batch_size,
        )
        profiler.write_summary(summary)
        profiler.print_summary(summary)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    main()
