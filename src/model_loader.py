from __future__ import annotations

from dataclasses import dataclass
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


def load_model_parts(
    model_name_or_path: str,
    dtype: str = "float16",
    device: str = "cpu",
) -> LoadedModelParts:
    torch_dtype = getattr(torch, dtype)

    # When targeting GPU, load directly onto the device so weights never
    # stage as a full second copy in CPU RAM.
    if device == "cpu":
        device_map = None
    else:
        device_map = device  # e.g. "cuda:0" or "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    base = model.model
    embed_tokens = base.embed_tokens
    layers = list(base.layers)
    final_norm = getattr(base, "norm", None)
    lm_head = model.lm_head

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
    )