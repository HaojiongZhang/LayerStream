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


def load_model_parts(model_name_or_path: str, dtype: str = "float16") -> LoadedModelParts:
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    # These attribute names vary by model family.
    # Adjust once you decide on a target family.
    base = model.model
    embed_tokens = base.embed_tokens
    layers = list(base.layers)
    final_norm = getattr(base, "norm", None)
    lm_head = model.lm_head

    model.eval()

    # Keep everything on CPU initially.
    model.cpu()

    return LoadedModelParts(
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
        hf_config=model.config,
    )