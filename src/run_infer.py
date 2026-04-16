from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import AppConfig, GenerationConfig, ModelConfig, OffloadConfig, LoggingConfig
from .cpu_weight_store import CPUWeightStore
from .forward_engine import ForwardEngine
from .generation import Generator
from .gpu_buffer_pool import GPUBufferPool
from .kv_manager import KVCacheManager
from .logger import ExperimentLogger
from .model_loader import load_model_parts
from .prefetcher import LayerPrefetcher
from .streams import StreamManager


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LayerStream inference — double-buffered layer streaming or full-VRAM mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-2-13b-chat-hf",
        help="HuggingFace model name or path to a local checkpoint. "
             "For Llama-2 you must first accept the license and run "
             "'huggingface-cli login'.",
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
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument(
        "--buffer-depth",
        type=int,
        default=2,
        help="Number of GPU weight-buffer slots (layered mode only). "
             "2 = classic double-buffer.",
    )
    p.add_argument(
        "--max-gpu-kv-gb",
        type=float,
        default=2.0,
        help="Maximum GPU memory reserved for KV cache (GB, layered mode only).",
    )
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--no-profile", action="store_true", help="Disable CUDA event profiling.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Engine builders
# ---------------------------------------------------------------------------

def _build_vram_engine(
    parts,
    cfg: AppConfig,
    logger: ExperimentLogger,
) -> tuple[ForwardEngine, KVCacheManager]:
    """All decoder layers already on GPU (loaded there by load_model_parts)."""
    print(f"[vram] Using {len(parts.layers)} layers already on GPU.")
    vram_layers = [layer.eval() for layer in parts.layers]

    kv_manager = KVCacheManager(
        num_layers=len(vram_layers),
        max_gpu_kv_bytes=cfg.offload.max_gpu_kv_bytes,
        enable_cpu_offload=False,
        pin_cpu_memory=False,
    )
    engine = ForwardEngine(
        embed_tokens=parts.embed_tokens,
        final_norm=parts.final_norm,
        lm_head=parts.lm_head,
        kv_manager=kv_manager,
        logger=logger,
        vram_layers=vram_layers,
        profile_cuda=cfg.offload.profile_cuda,
    )
    return engine, kv_manager


def _build_layered_engine(
    parts,
    cfg: AppConfig,
    logger: ExperimentLogger,
    torch_dtype: torch.dtype,
) -> tuple[ForwardEngine, KVCacheManager]:
    """
    Double-buffered streaming mode.

    Weight double-buffer: two GPU slots (buffer_depth=2) alternate so one
    is being computed while the next is copied from pinned CPU memory.

    KV double-buffer: the KV prefetch for layer N+1 is issued concurrently
    with compute for layer N, so the KV H2D copy overlaps with compute.
    """
    # GPUBufferPool deep-copies the prototype layer, so it MUST be created
    # before CPUWeightStore, which frees each layer's parameters after pinning.
    gpu_pool = GPUBufferPool(
        prototype_layer=parts.layers[0],
        buffer_depth=cfg.offload.buffer_depth,
        device=cfg.model.device,
        dtype=torch_dtype,
    )
    print(f"[layered] Pinning {len(parts.layers)} layers in CPU memory …")
    weight_store = CPUWeightStore(
        layers=parts.layers,
        pin_memory=cfg.offload.pin_cpu_memory,
    )
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
        enable_cpu_offload=cfg.offload.enable_kv_cpu_offload,
        pin_cpu_memory=cfg.offload.pin_cpu_memory,
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
        profile_cuda=cfg.offload.profile_cuda,
    )
    return engine, kv_manager


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = AppConfig(
        model=ModelConfig(
            model_name_or_path=args.model,
            dtype=args.dtype,
        ),
        offload=OffloadConfig(
            buffer_depth=args.buffer_depth,
            max_gpu_kv_bytes=int(args.max_gpu_kv_gb * 1024 ** 3),
            vram_only=(args.mode == "vram"),
            profile_cuda=not args.no_profile,
        ),
        generation=GenerationConfig(max_new_tokens=args.max_new_tokens),
        logging=LoggingConfig(log_dir=Path(args.log_dir)),
    )

    print(f"Loading model: {cfg.model.model_name_or_path}  (dtype={cfg.model.dtype}, mode={args.mode})")
    logger = ExperimentLogger(cfg.logging.log_dir)
    # vram mode: load directly to GPU so weights never stage as a full copy in CPU RAM.
    # layered mode: load to CPU so CPUWeightStore can pin them layer-by-layer.
    load_device = cfg.model.device if cfg.offload.vram_only else "cpu"
    parts = load_model_parts(cfg.model.model_name_or_path, dtype=cfg.model.dtype, device=load_device)
    torch_dtype = getattr(torch, cfg.model.dtype)

    if cfg.offload.vram_only:
        engine, kv_manager = _build_vram_engine(parts, cfg, logger)
    else:
        engine, kv_manager = _build_layered_engine(parts, cfg, logger, torch_dtype)

    eos_id = parts.tokenizer.eos_token_id
    generator = Generator(
        tokenizer=parts.tokenizer,
        engine=engine,
        kv_manager=kv_manager,
        logger=logger,
        eos_token_id=eos_id,
    )

    print(f"Generating (mode={args.mode}, max_new_tokens={args.max_new_tokens}) …")
    result = generator.generate(
        request_id=1,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n--- Output ---")
    print(result.text)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    main()
