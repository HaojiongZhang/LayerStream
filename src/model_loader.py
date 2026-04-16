from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModelParts:
    tokenizer: Any
    embed_tokens: torch.nn.Module
    layers: list[torch.nn.Module]
    final_norm: torch.nn.Module | None
    lm_head: torch.nn.Module
    hf_config: Any
    model_path: Path  # resolved local directory (used by MmapWeightStore)
    # Model-level rotary embedding module (transformers >= 4.40).
    # None on older versions where each attention layer had its own rotary_emb.
    rotary_emb: torch.nn.Module | None = None


def _resolve_local_path(model_name_or_path: str) -> Path:
    """
    Return the local directory that contains the model's safetensors files.

    If model_name_or_path is already a local directory, return it directly.
    Otherwise treat it as a HuggingFace Hub repo ID and resolve via the
    local snapshot cache (downloads if not yet cached).
    """
    p = Path(model_name_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_name_or_path))


def load_model_parts(
    model_name_or_path: str,
    dtype: str = "float16",
    device: str = "cpu",
) -> LoadedModelParts:
    torch_dtype = getattr(torch, dtype)

    # Resolve to local path before loading — needed later for MmapWeightStore.
    local_path = _resolve_local_path(model_name_or_path)

    # When targeting GPU, load directly onto the device so weights never
    # stage as a full second copy in CPU RAM.
    device_map = None if device == "cpu" else device

    model = AutoModelForCausalLM.from_pretrained(
        str(local_path),
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(local_path), use_fast=True)

    base = model.model
    embed_tokens = base.embed_tokens
    layers = list(base.layers)
    final_norm = getattr(base, "norm", None)
    lm_head = model.lm_head
    # Present in transformers >= 4.40 (moved from per-attention to model level).
    rotary_emb = getattr(base, "rotary_emb", None)

    model.eval()

    if device == "cpu":
        model.cpu()

    return LoadedModelParts(
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
        hf_config=model.config,
        model_path=local_path,
        rotary_emb=rotary_emb,
    )
