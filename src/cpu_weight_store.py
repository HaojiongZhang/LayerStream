from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _state_dict_nbytes(state: dict[str, torch.Tensor]) -> int:
    total = 0
    for v in state.values():
        total += v.numel() * v.element_size()
    return total


# ---------------------------------------------------------------------------
# MmapWeightStore  (default, memory-safe)
# ---------------------------------------------------------------------------

class MmapWeightStore:
    """
    Reads layer weights directly from safetensors files via mmap.

    No weights are copied into anonymous or pinned (shmem) memory.
    Tensors returned by get_layer_state() are file-backed pages that the
    kernel can evict under memory pressure — so this class operates safely
    inside tight cgroup limits regardless of model size.

    H2D copies in LayerPrefetcher use CUDA's automatic pageable-to-pinned
    staging path, which allocates only a small transient buffer per transfer
    rather than pinning all 26 GB at once.
    """

    def __init__(
        self,
        model_path: str | Path,
        layer_prefix: str = "model.layers",
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise ValueError(f"model_path must be a local directory, got: {model_path}")

        # Resolve the safetensors shard index.
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                data = json.load(f)
            weight_map: dict[str, str] = data["weight_map"]  # param_key → shard filename
        elif (model_path / "model.safetensors").exists():
            from safetensors import safe_open as _sf_open
            tmp = _sf_open(str(model_path / "model.safetensors"), framework="pt", device="cpu")
            weight_map = {k: "model.safetensors" for k in tmp.keys()}
            del tmp
        else:
            raise FileNotFoundError(
                f"No safetensors files found in {model_path}.\n"
                "Layered mmap mode requires the model to be saved in safetensors format.\n"
                "Download it with: huggingface-cli download <model> --local-dir <path>"
            )

        self._model_path = model_path
        self._weight_map = weight_map
        self._layer_prefix = layer_prefix

        # Open each shard once; keep the handle alive so the mmap stays valid.
        from safetensors import safe_open as _sf_open
        shard_names = sorted(set(weight_map.values()))
        self._shards: dict[str, Any] = {
            name: _sf_open(str(model_path / name), framework="pt", device="cpu")
            for name in shard_names
        }

        # Count layers by scanning key names.
        pat = re.compile(rf"^{re.escape(layer_prefix)}\.(\d+)\.")
        indices: set[int] = set()
        for key in weight_map:
            m = pat.match(key)
            if m:
                indices.add(int(m.group(1)))
        self._num_layers = max(indices) + 1 if indices else 0

        # Lazy cache for per-layer byte sizes.
        self._layer_bytes_cache: dict[int, int] = {}

    def num_layers(self) -> int:
        return self._num_layers

    def get_layer_state(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """
        Return a dict keyed by local param names (layer prefix stripped).

        Tensors are mmap-backed views of the safetensors file — no RAM copy.
        """
        prefix = f"{self._layer_prefix}.{layer_idx}."
        state: dict[str, torch.Tensor] = {}
        for full_key, shard_name in self._weight_map.items():
            if full_key.startswith(prefix):
                local_key = full_key[len(prefix):]
                state[local_key] = self._shards[shard_name].get_tensor(full_key)
        return state

    def layer_nbytes(self, layer_idx: int) -> int:
        if layer_idx not in self._layer_bytes_cache:
            self._layer_bytes_cache[layer_idx] = _state_dict_nbytes(
                self.get_layer_state(layer_idx)
            )
        return self._layer_bytes_cache[layer_idx]

    def max_layer_nbytes(self) -> int:
        return max(self.layer_nbytes(i) for i in range(self._num_layers)) if self._num_layers else 0


# ---------------------------------------------------------------------------
# CPUWeightStore  (legacy / high-RAM environments)
# ---------------------------------------------------------------------------

class CPUWeightStore:
    """
    Clones all layer weights into pinned CPU memory up front.

    Faster H2D transfers than MmapWeightStore (DMA from page-locked memory),
    but requires the full model to fit in RAM as shmem.  Use MmapWeightStore
    in memory-constrained environments (cgroup < model size).
    """

    def __init__(self, layers: list[torch.nn.Module], pin_memory: bool = True) -> None:
        self._layers: list[dict[str, torch.Tensor]] = []
        self._layer_bytes: list[int] = []
        self.pin_memory = pin_memory

        for layer in layers:
            state: dict[str, torch.Tensor] = {}
            for k, v in layer.state_dict().items():
                t = v.detach().cpu().clone()
                if pin_memory:
                    t = t.pin_memory()
                state[k] = t

            # Release original parameter storage to halve peak RAM.
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
