from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    model_name_or_path: str
    dtype: str = "float16"
    device: str = "cuda"


@dataclass
class OffloadConfig:
    buffer_depth: int = 2
    pin_cpu_memory: bool = True
    async_prefetch: bool = True
    gpu_margin_bytes: int = 512 * 1024 * 1024
    max_gpu_kv_bytes: int = 2 * 1024 * 1024 * 1024
    enable_kv_cpu_offload: bool = True
    profile_cuda: bool = True
    profile_wall: bool = True
    # When True: all layers are kept on GPU; no CPU weight streaming or KV offload.
    # When False: layers are double-buffered from CPU (streaming/offload mode).
    vram_only: bool = False


@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    eos_token_id: Optional[int] = None


@dataclass
class LoggingConfig:
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    flush_every: int = 1


@dataclass
class AppConfig:
    model: ModelConfig
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)