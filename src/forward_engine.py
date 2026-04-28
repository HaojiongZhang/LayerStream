from __future__ import annotations

import torch

try:
    from transformers.cache_utils import Cache as _CacheBase
except ImportError:
    _CacheBase = object  # type: ignore[assignment,misc]


class _SingleLayerCache(_CacheBase):  # type: ignore[misc]
    is_sliding_window: bool = False
    is_static: bool = True 

    def __init__(
        self,
        max_seq_len: int,
        current_seq_len: int,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
    ) -> None:
  
        self.max_seq_len = max_seq_len
        self.k_cache = past_key
        self.v_cache = past_value
        self._seen = current_seq_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = key_states.shape[-2]

        if self.k_cache is None:
            batch_size, num_heads, _, head_dim = key_states.shape
            self.k_cache = torch.empty(
                (batch_size, num_heads, self.max_seq_len, head_dim), 
                dtype=key_states.dtype, device=key_states.device
            )
            self.v_cache = torch.empty(
                (batch_size, num_heads, self.max_seq_len, head_dim), 
                dtype=value_states.dtype, device=value_states.device
            )

        self.k_cache[:, :, self._seen : self._seen + seq_len, :] = key_states
        self.v_cache[:, :, self._seen : self._seen + seq_len, :] = value_states
        
        self._seen += seq_len
        
        return self.k_cache[:, :, :self._seen, :], self.v_cache[:, :, :self._seen, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen

    def get_max_cache_shape(self) -> int | None:
        return self.max_seq_len

    def get_max_length(self) -> int | None:
        return self.max_seq_len

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.max_seq_len - self._seen


from dataclasses import dataclass
from typing import Any

import inspect
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
    def __init__(
        self,
        embed_tokens: torch.nn.Module,
        final_norm: torch.nn.Module | None,
        lm_head: torch.nn.Module,
        kv_manager: KVCacheManager,
        logger: ExperimentLogger,
        gpu_pool: GPUBufferPool | None = None,
        prefetcher: LayerPrefetcher | None = None,
        streams: StreamManager | None = None,
        vram_layers: list[torch.nn.Module] | None = None,
        rotary_emb: torch.nn.Module | None = None,
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
        self.rotary_emb = rotary_emb.to("cuda") if rotary_emb is not None else None
        self.kv_manager = kv_manager
        self.logger = logger
        self.gpu_pool = gpu_pool
        self.prefetcher = prefetcher
        self.streams = streams
        self.vram_layers = vram_layers
        self.profile_cuda = profile_cuda

        self._new_llama_api: bool | None = None

    def _detect_new_api(self, layer_module: torch.nn.Module) -> bool:
        if self._new_llama_api is None:
            params = inspect.signature(layer_module.forward).parameters
            self._new_llama_api = "position_embeddings" in params
        return self._new_llama_api

    def _make_position_info(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len = input_ids.shape[0], input_ids.shape[1]

        past_len = self.kv_manager.current_seq_len

        cache_position = torch.arange(
            past_len, past_len + seq_len,
            device=input_ids.device, dtype=torch.long,
        )
        position_ids = cache_position.unsqueeze(0).expand(batch, -1)

        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        return position_ids, cache_position, position_embeddings

    def _run_layer(
        self,
        layer_module: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        layer_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        if self._detect_new_api(layer_module):
            return self._run_layer_new(
                layer_module, hidden_states, past_key_value,
                position_embeddings, cache_position,
            )
        return self._run_layer_old(
            layer_module, hidden_states, attention_mask, position_ids, past_key_value,
        )

    def _run_layer_new(
        self,
        layer_module: torch.nn.Module,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        cache_position: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        
        params = inspect.signature(layer_module.forward).parameters
        kv_kwarg = "past_key_values" if "past_key_values" in params else "past_key_value"

        cache = _SingleLayerCache(
            max_seq_len=self.kv_manager.max_seq_len,
            current_seq_len=self.kv_manager.current_seq_len,
            past_key=past_key_value[0] if past_key_value is not None else None,
            past_value=past_key_value[1] if past_key_value is not None else None,
        )

        kwargs: dict[str, Any] = {
            kv_kwarg: cache,
            "use_cache": True,
        }
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if cache_position is not None:
            kwargs["cache_position"] = cache_position

        output = layer_module(hidden_states, **kwargs)
        hidden = output[0] if isinstance(output, tuple) else output

        presents: tuple[torch.Tensor, torch.Tensor] | None = (
            (cache.k_cache, cache.v_cache) if cache.k_cache is not None else None
        )
        return hidden, presents

    def _run_layer_old(
        self,
        layer_module: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        kwargs: dict[str, Any] = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

        if past_key_value is not None:
            co_vars = layer_module.forward.__code__.co_varnames
            if "layer_past" in co_vars:
                kwargs["layer_past"] = past_key_value
            elif "past_key_value" in co_vars:
                kwargs["past_key_value"] = past_key_value
            if "use_cache" in co_vars:
                kwargs["use_cache"] = True
        elif "use_cache" in layer_module.forward.__code__.co_varnames:
            kwargs["use_cache"] = False

        output = layer_module(hidden_states, **kwargs)

        if isinstance(output, tuple):
            hidden = output[0]
            raw = output[1] if len(output) > 1 else None
        else:
            hidden = getattr(output, "last_hidden_state", None) or output
            raw = getattr(output, "past_key_values", None)

        presents: tuple[torch.Tensor, torch.Tensor] | None = None
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            presents = (raw[0], raw[1])

        return hidden, presents

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

        pos_ids, cache_pos, pos_emb = self._make_position_info(
            hidden_states, input_ids
        )
        if position_ids is None:
            position_ids = pos_ids

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
                position_embeddings=pos_emb,
                cache_position=cache_pos,
            )
            compute_timer.stop()
            compute_ms = compute_timer.elapsed_ms()
            total_compute_ms += compute_ms

            if presents is not None:
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
        
        self.kv_manager.current_seq_len += input_ids.shape[1]
        
        return ForwardOutput(
            logits=logits,
            total_copy_ms=0.0,
            total_compute_ms=total_compute_ms,
            total_wait_ms=0.0,
        )

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

        pos_ids, cache_pos, pos_emb = self._make_position_info(
            hidden_states, input_ids
        )
        if position_ids is None:
            position_ids = pos_ids

        true_num_layers = self.prefetcher.weight_store.num_layers()

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

            wait_timer = WallTimer()
            wait_timer.start()
            self.prefetcher.wait_until_ready(slot, self.streams.compute_stream)
            self.kv_manager.wait_for_layer_kv(layer_idx, self.streams.compute_stream)
            wait_timer.stop()
            total_wait_ms += wait_timer.elapsed_ms

            key, value = self.kv_manager.get_layer_kv(layer_idx)

            if next_layer_idx < true_num_layers:
                stats = self.prefetcher.prefetch_layer(
                    request_id=request_id,
                    token_idx=token_idx,
                    layer_idx=next_layer_idx,
                    slot_idx=next_slot_idx,
                )
                total_copy_ms += stats.copy_ms
                self.kv_manager.prefetch_layer_kv(next_layer_idx, self.streams.kv_stream)

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
                    position_embeddings=pos_emb,
                    cache_position=cache_pos,
                )
                compute_timer.stop(self.streams.compute_stream)
                
                if presents is not None:
                    self.kv_manager.update_layer_kv(
                        layer_idx, presents[0], presents[1], stream=self.streams.compute_stream
                    )

            compute_ms = compute_timer.elapsed_ms()
            total_compute_ms += compute_ms

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

        self.kv_manager.current_seq_len += input_ids.shape[1]

        return ForwardOutput(
            logits=logits,
            total_copy_ms=total_copy_ms,
            total_compute_ms=total_compute_ms,
            total_wait_ms=total_wait_ms,
        )