from __future__ import annotations

import copy
from dataclasses import dataclass

import torch


@dataclass
class LayerSlot:
    slot_idx: int
    module: torch.nn.Module
    loaded_layer_idx: int | None = None
    ready_event: torch.cuda.Event | None = None


class GPUBufferPool:
    def __init__(
        self,
        prototype_layer: torch.nn.Module,
        buffer_depth: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.slots: list[LayerSlot] = []
        for i in range(buffer_depth):
            module = copy.deepcopy(prototype_layer).to(device=device, dtype=dtype)
            module.eval()
            self.slots.append(LayerSlot(slot_idx=i, module=module))

    def get_slot(self, slot_idx: int) -> LayerSlot:
        return self.slots[slot_idx]

    def num_slots(self) -> int:
        return len(self.slots)