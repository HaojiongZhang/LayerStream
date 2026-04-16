from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LayerKV:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    device: str = "none"
    bytes_total: int = 0
    ready_event: torch.cuda.Event | None = None


def _tensor_pair_nbytes(k: torch.Tensor | None, v: torch.Tensor | None) -> int:
    total = 0
    if k is not None:
        total += k.numel() * k.element_size()
    if v is not None:
        total += v.numel() * v.element_size()
    return total


class KVCacheManager:
    def __init__(
        self,
        num_layers: int,
        max_gpu_kv_bytes: int,
        max_seq_len: int,
        enable_cpu_offload: bool = True,
        pin_cpu_memory: bool = True,
    ) -> None:
        self.max_gpu_kv_bytes = max_gpu_kv_bytes
        self.max_seq_len = max_seq_len
        self.enable_cpu_offload = enable_cpu_offload
        self.pin_cpu_memory = pin_cpu_memory
        self.layer_kv: list[LayerKV] = [LayerKV() for _ in range(num_layers)]
        self.current_seq_len = 0 

    def get_layer_kv(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        rec = self.layer_kv[layer_idx]
        return rec.key, rec.value

    def update_layer_kv(
        self,
        layer_idx: int,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        rec = self.layer_kv[layer_idx]
        rec.key = key
        rec.value = value
        
        if key is None or value is None:
            rec.device = "none"
            rec.bytes_total = 0
            rec.ready_event = None
            return

        rec.device = str(key.device)
        rec.bytes_total = _tensor_pair_nbytes(key, value)
        
        if stream is None:
            stream = torch.cuda.current_stream()
            
        if rec.ready_event is None:
            rec.ready_event = torch.cuda.Event()
            
        rec.ready_event.record(stream)

    def prefetch_layer_kv(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        rec = self.layer_kv[layer_idx]
        if rec.key is None or rec.value is None:
            return
        if rec.device.startswith("cuda"):
            return

        with torch.cuda.stream(stream):
            key_cuda = rec.key.to("cuda", non_blocking=True)
            val_cuda = rec.value.to("cuda", non_blocking=True)
            rec.key = key_cuda
            rec.value = val_cuda
            rec.device = "cuda"
            rec.bytes_total = _tensor_pair_nbytes(key_cuda, val_cuda)
            ready_event = torch.cuda.Event()
            ready_event.record(stream)
            rec.ready_event = ready_event

    def wait_for_layer_kv(
        self,
        layer_idx: int,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        rec = self.layer_kv[layer_idx]
        if rec.key is None or rec.value is None:
            return
        if not rec.device.startswith("cuda"):
            return
        if rec.ready_event is None:
            return

        target_stream = torch.cuda.current_stream() if stream is None else stream
        target_stream.wait_event(rec.ready_event)
        rec.ready_event = None

    def total_gpu_kv_bytes(self) -> int:
        total = 0
        for rec in self.layer_kv:
            if rec.device.startswith("cuda"):
                total += rec.bytes_total
        return total

    def total_cpu_kv_bytes(self) -> int:
        total = 0
        for rec in self.layer_kv:
            if rec.device == "cpu":
                total += rec.bytes_total
        return total

    def maybe_offload_old_layers(self, protected_layer: int | None = None) -> None:
        if not self.enable_cpu_offload:
            return

        total_gpu = self.total_gpu_kv_bytes()
        if total_gpu <= self.max_gpu_kv_bytes:
            return

        for i, rec in enumerate(self.layer_kv):
            if total_gpu <= self.max_gpu_kv_bytes:
                break
            if i == protected_layer:
                continue
            if rec.key is None or rec.value is None:
                continue
            if not rec.device.startswith("cuda"):
                continue

            key_cpu = rec.key.detach().cpu()
            val_cpu = rec.value.detach().cpu()
            if self.pin_cpu_memory:
                key_cpu = key_cpu.pin_memory()
                val_cpu = val_cpu.pin_memory()

            rec.key = key_cpu
            rec.value = val_cpu
            rec.device = "cpu"
            rec.bytes_total = _tensor_pair_nbytes(key_cpu, val_cpu)
            rec.ready_event = None
            total_gpu = self.total_gpu_kv_bytes()

    def ensure_layer_on_gpu(self, layer_idx: int) -> None:
        rec = self.layer_kv[layer_idx]
        if rec.key is None or rec.value is None:
            return
        if rec.device.startswith("cuda"):
            return

        rec.key = rec.key.to("cuda", non_blocking=True)
        rec.value = rec.value.to("cuda", non_blocking=True)
        rec.device = "cuda"
        rec.bytes_total = _tensor_pair_nbytes(rec.key, rec.value)
        rec.ready_event = None