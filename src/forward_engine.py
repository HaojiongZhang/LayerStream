from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .gpu_buffer_pool import GPUBufferPool
from .kv_manager import KVCacheManager
from .logger import ExperimentLogger
from .prefetcher import LayerPrefetcher
from .streams import StreamManager
from .timing import CudaTimer, WallTimer


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    total_copy_ms: float
    total_compute_ms: float
    total_wait_ms: float


class ForwardEngine:
    """
    Two operating modes, selected at construction time:

    vram_only  (vram_layers is not None)
        All decoder layers are already resident on GPU.  No streaming,
        no prefetcher, no buffer pool.  Fastest if VRAM is large enough.

    layered / streaming  (gpu_pool + prefetcher + streams are provided)
        Layers are kept on CPU and double-buffered onto two small GPU slots.
        KV prefetch for the next layer is kicked off alongside the next weight
        prefetch so both H2D copies overlap with the current layer's compute.
    """

    def __init__(
        self,
        embed_tokens: torch.nn.Module,
        final_norm: torch.nn.Module | None,
        lm_head: torch.nn.Module,
        kv_manager: KVCacheManager,
        logger: ExperimentLogger,
        # --- streaming mode ---
        gpu_pool: GPUBufferPool | None = None,
        prefetcher: LayerPrefetcher | None = None,
        streams: StreamManager | None = None,
        # --- vram-only mode ---
        vram_layers: list[torch.nn.Module] | None = None,
        profile_cuda: bool = True,
    ) -> None:
        if vram_layers is not None and (gpu_pool is not None or prefetcher is not None):
            raise ValueError(
                "Specify either vram_layers (vram-only mode) "
                "OR gpu_pool+prefetcher+streams (streaming mode), not both."
            )
        if vram_layers is None and (gpu_pool is None or prefetcher is None or streams is None):
            raise ValueError(
                "Streaming mode requires gpu_pool, prefetcher, and streams."
            )

        self.embed_tokens = embed_tokens.to("cuda")
        self.final_norm = final_norm.to("cuda") if final_norm is not None else None
        self.lm_head = lm_head.to("cuda")
        self.kv_manager = kv_manager
        self.logger = logger
        self.gpu_pool = gpu_pool
        self.prefetcher = prefetcher
        self.streams = streams
        self.vram_layers = vram_layers
        self.profile_cuda = profile_cuda

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _run_layer(
        self,
        layer_module: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        layer_idx: int,
    ) -> tuple[torch.Tensor, Any | None]:
        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

        if past_key_value is not None:
            if "layer_past" in layer_module.forward.__code__.co_varnames:
                kwargs["layer_past"] = past_key_value
            elif "past_key_value" in layer_module.forward.__code__.co_varnames:
                kwargs["past_key_value"] = past_key_value
            if "use_cache" in layer_module.forward.__code__.co_varnames:
                kwargs["use_cache"] = True
        elif "use_cache" in layer_module.forward.__code__.co_varnames:
            kwargs["use_cache"] = False

        output = layer_module(hidden_states, **kwargs)

        hidden: torch.Tensor
        presents: Any | None = None
        if isinstance(output, tuple):
            hidden = output[0]
            presents = output[1] if len(output) > 1 else None
        else:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None:
                hidden = getattr(output, "hidden_states", output)
            presents = getattr(output, "past_key_values", None)

        return hidden, presents

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def forward_token(
        self,
        request_id: int,
        token_idx: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> ForwardOutput:
        if self.vram_layers is not None:
            return self._forward_vram_only(
                request_id, token_idx, input_ids, attention_mask, position_ids
            )
        return self._forward_streaming(
            request_id, token_idx, input_ids, attention_mask, position_ids
        )

    # ------------------------------------------------------------------
    # VRAM-only mode
    # ------------------------------------------------------------------

    def _forward_vram_only(
        self,
        request_id: int,
        token_idx: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> ForwardOutput:
        total_compute_ms = 0.0

        hidden_states = self.embed_tokens(input_ids)

        for layer_idx, layer in enumerate(self.vram_layers):
            key, value = self.kv_manager.get_layer_kv(layer_idx)

            compute_timer = CudaTimer(enabled=self.profile_cuda)
            compute_timer.start()
            hidden_states, presents = self._run_layer(
                layer,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(key, value) if key is not None and value is not None else None,
                layer_idx=layer_idx,
            )
            compute_timer.stop()
            compute_ms = compute_timer.elapsed_ms()
            total_compute_ms += compute_ms

            if presents is not None:
                if isinstance(presents, (tuple, list)) and len(presents) == 2:
                    self.kv_manager.update_layer_kv(layer_idx, presents[0], presents[1])

            self.logger.log_layer(
                request_id=request_id,
                token_idx=token_idx,
                layer_idx=layer_idx,
                buffer_slot=0,
                event="compute",
                compute_ms=compute_ms,
                wait_ms=0.0,
                gpu_kv_bytes=self.kv_manager.total_gpu_kv_bytes(),
                cpu_kv_bytes=self.kv_manager.total_cpu_kv_bytes(),
            )

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)
        return ForwardOutput(
            logits=logits,
            total_copy_ms=0.0,
            total_compute_ms=total_compute_ms,
            total_wait_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Streaming (double-buffered) mode
    # ------------------------------------------------------------------

    def _forward_streaming(
        self,
        request_id: int,
        token_idx: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> ForwardOutput:
        total_copy_ms = 0.0
        total_compute_ms = 0.0
        total_wait_ms = 0.0

        hidden_states = self.embed_tokens(input_ids)

        true_num_layers = self.prefetcher.weight_store.num_layers()

        # Prime layer 0: start both the weight H2D copy and the KV H2D copy
        # (KV is a no-op on the very first token when the cache is empty).
        stats = self.prefetcher.prefetch_layer(
            request_id=request_id,
            token_idx=token_idx,
            layer_idx=0,
            slot_idx=0,
        )
        total_copy_ms += stats.copy_ms
        self.kv_manager.prefetch_layer_kv(0, self.streams.kv_stream)

        for layer_idx in range(true_num_layers):
            cur_slot_idx = layer_idx % self.gpu_pool.num_slots()
            next_layer_idx = layer_idx + 1
            next_slot_idx = next_layer_idx % self.gpu_pool.num_slots()

            slot = self.prefetcher.get_slot_for_layer(cur_slot_idx, layer_idx)

            # Block until the current layer's weights AND KV are both on GPU.
            wait_timer = WallTimer()
            wait_timer.start()
            self.prefetcher.wait_until_ready(slot, self.streams.compute_stream)
            self.kv_manager.wait_for_layer_kv(layer_idx, self.streams.compute_stream)
            wait_timer.stop()
            total_wait_ms += wait_timer.elapsed_ms

            key, value = self.kv_manager.get_layer_kv(layer_idx)

            # Kick off the NEXT layer's weight copy AND KV copy before computing
            # the current layer so both H2D transfers overlap with compute.
            if next_layer_idx < true_num_layers:
                stats = self.prefetcher.prefetch_layer(
                    request_id=request_id,
                    token_idx=token_idx,
                    layer_idx=next_layer_idx,
                    slot_idx=next_slot_idx,
                )
                total_copy_ms += stats.copy_ms
                self.kv_manager.prefetch_layer_kv(next_layer_idx, self.streams.kv_stream)

            # Compute current layer (overlaps with H2D copies issued above).
            compute_timer = CudaTimer(enabled=self.profile_cuda)
            with torch.cuda.stream(self.streams.compute_stream):
                compute_timer.start(self.streams.compute_stream)
                hidden_states, presents = self._run_layer(
                    slot.module,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=(key, value) if key is not None and value is not None else None,
                    layer_idx=layer_idx,
                )
                compute_timer.stop(self.streams.compute_stream)
            compute_ms = compute_timer.elapsed_ms()
            total_compute_ms += compute_ms

            if presents is not None:
                if isinstance(presents, (tuple, list)) and len(presents) == 2:
                    self.kv_manager.update_layer_kv(layer_idx, presents[0], presents[1])

            # Protect next_layer_idx from eviction — its KV was just prefetched.
            self.kv_manager.maybe_offload_old_layers(protected_layer=next_layer_idx)

            self.logger.log_layer(
                request_id=request_id,
                token_idx=token_idx,
                layer_idx=layer_idx,
                buffer_slot=cur_slot_idx,
                event="compute",
                compute_ms=compute_ms,
                wait_ms=wait_timer.elapsed_ms,
                gpu_kv_bytes=self.kv_manager.total_gpu_kv_bytes(),
                cpu_kv_bytes=self.kv_manager.total_cpu_kv_bytes(),
            )

        if self.final_norm is not None:
            with torch.cuda.stream(self.streams.compute_stream):
                hidden_states = self.final_norm(hidden_states)

        with torch.cuda.stream(self.streams.compute_stream):
            logits = self.lm_head(hidden_states)

        return ForwardOutput(
            logits=logits,
            total_copy_ms=total_copy_ms,
            total_compute_ms=total_compute_ms,
            total_wait_ms=total_wait_ms,
        )
