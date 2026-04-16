from __future__ import annotations

from typing import Any

import torch


def _pin_tensor_tree(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory()
    if isinstance(obj, dict):
        return {k: _pin_tensor_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_pin_tensor_tree(v) for v in obj]
    return obj


def _state_dict_nbytes(state: dict[str, torch.Tensor]) -> int:
    total = 0
    for v in state.values():
        total += v.numel() * v.element_size()
    return total


class CPUWeightStore:
    def __init__(self, layers: list[torch.nn.Module], pin_memory: bool = True) -> None:
        self._layers: list[dict[str, torch.Tensor]] = []
        self._layer_bytes: list[int] = []
        self.pin_memory = pin_memory

        for layer in layers:
            # Build the pinned state one key at a time.
            # .clone() ensures the new tensor owns its storage independently
            # of the layer's parameter storage, so we can free the original.
            state: dict[str, torch.Tensor] = {}
            for k, v in layer.state_dict().items():
                t = v.detach().cpu().clone()
                if pin_memory:
                    t = t.pin_memory()
                state[k] = t

            # Release the original parameter storage immediately.
            # Without this, both the model copy (~26 GB for 13B) and the
            # pinned copy live in RAM at the same time, causing OOM.
            for p in layer.parameters():
                p.data = torch.empty(0, dtype=p.dtype)

            self._layers.append(state)
            self._layer_bytes.append(_state_dict_nbytes(state))

    def num_layers(self) -> int:
        return len(self._layers)

    def get_layer_state(self, layer_idx: int) -> dict[str, torch.Tensor]:
        return self._layers[layer_idx]

    def layer_nbytes(self, layer_idx: int) -> int:
        return self._layer_bytes[layer_idx]

    def max_layer_nbytes(self) -> int:
        return max(self._layer_bytes) if self._layer_bytes else 0