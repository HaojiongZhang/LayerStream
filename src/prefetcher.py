from __future__ import annotations

from dataclasses import dataclass

import torch

from .cpu_weight_store import CPUWeightStore
from .gpu_buffer_pool import GPUBufferPool, LayerSlot
from .logger import ExperimentLogger
from .timing import CudaTimer


@dataclass
class PrefetchStats:
    layer_idx: int
    slot_idx: int
    copy_ms: float
    weight_bytes: int


class LayerPrefetcher:
    def __init__(
        self,
        weight_store: CPUWeightStore,
        gpu_pool: GPUBufferPool,
        copy_stream: torch.cuda.Stream,
        logger: ExperimentLogger,
        profile_cuda: bool = True,
    ) -> None:
        self.weight_store = weight_store
        self.gpu_pool = gpu_pool
        self.copy_stream = copy_stream
        self.logger = logger
        self.profile_cuda = profile_cuda

    def prefetch_layer(
        self,
        request_id: int,
        token_idx: int,
        layer_idx: int,
        slot_idx: int,
    ) -> PrefetchStats:
        slot = self.gpu_pool.get_slot(slot_idx)
        cpu_state = self.weight_store.get_layer_state(layer_idx)

        timer = CudaTimer(enabled=self.profile_cuda)
        ready_event = torch.cuda.Event()

        with torch.cuda.stream(self.copy_stream):
            timer.start(self.copy_stream)
            module_state = slot.module.state_dict()
            for name, cpu_tensor in cpu_state.items():
                module_state[name].copy_(cpu_tensor, non_blocking=True)
            timer.stop(self.copy_stream)
            ready_event.record(self.copy_stream)

        slot.loaded_layer_idx = layer_idx
        slot.ready_event = ready_event

        copy_ms = timer.elapsed_ms()
        stats = PrefetchStats(
            layer_idx=layer_idx,
            slot_idx=slot_idx,
            copy_ms=copy_ms,
            weight_bytes=self.weight_store.layer_nbytes(layer_idx),
        )

        self.logger.log_layer(
            request_id=request_id,
            token_idx=token_idx,
            layer_idx=layer_idx,
            buffer_slot=slot_idx,
            event="prefetch",
            copy_ms=copy_ms,
            weight_bytes=stats.weight_bytes,
        )
        return stats

    def wait_until_ready(self, slot: LayerSlot, stream: torch.cuda.Stream | None = None) -> None:
        if slot.ready_event is None:
            raise RuntimeError(f"Slot {slot.slot_idx} has no ready event.")
        target_stream = torch.cuda.current_stream() if stream is None else stream
        target_stream.wait_event(slot.ready_event)

    def get_slot_for_layer(self, slot_idx: int, expected_layer_idx: int) -> LayerSlot:
        slot = self.gpu_pool.get_slot(slot_idx)
        if slot.loaded_layer_idx != expected_layer_idx:
            raise RuntimeError(
                f"Slot {slot_idx} has layer {slot.loaded_layer_idx}, expected {expected_layer_idx}"
            )
        return slot