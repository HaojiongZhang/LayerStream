from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LayerKV:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    key_scale: torch.Tensor | None = None
    value_scale: torch.Tensor | None = None
    orig_dtype: torch.dtype | None = None
    key_orig_shape: tuple[int, ...] | None = None
    value_orig_shape: tuple[int, ...] | None = None
    device: str = "none"
    bytes_total: int = 0
    ready_event: torch.cuda.Event | None = None
    quantized_bits: int = 16


def _tensor_nbytes(x: torch.Tensor | None) -> int:
    if x is None:
        return 0
    return x.numel() * x.element_size()


def _record_nbytes(rec: LayerKV) -> int:
    return (
        _tensor_nbytes(rec.key)
        + _tensor_nbytes(rec.value)
        + _tensor_nbytes(rec.key_scale)
        + _tensor_nbytes(rec.value_scale)
    )


def _quantize_tensor_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-vector int8 quantization along the last dimension.

    For LLM KV cache tensors this is typically applied to tensors shaped like:
        [batch, num_heads, seq_len, head_dim]

    The scale is therefore shaped like:
        [batch, num_heads, seq_len, 1]

    Quantization is performed on the tensor's current device, so this works for
    both CPU and GPU tensors.
    """
    x_detached = x.detach()
    max_abs = x_detached.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / 127.0).clamp_min(1e-8).to(torch.float32)
    q = torch.round(x_detached / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale



def _dequantize_tensor_int8(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
) -> torch.Tensor:
    q_dev = q.to(device, non_blocking=True)
    s_dev = scale.to(device, non_blocking=True)
    return (q_dev.to(torch.float32) * s_dev).to(dtype)



def _quantize_tensor_int4(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """
    Symmetric per-vector packed int4 quantization along the last dimension.

    Returns:
        packed: uint8 tensor containing two signed int4 values per byte
        scale:  float32 tensor shaped [..., 1]
        orig_shape: original tensor shape needed for unpacking/dequantization
    """
    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)

    max_abs = x_detached.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / 7.0).clamp_min(1e-8).to(torch.float32)
    q = torch.round(x_detached / scale).clamp_(-8, 7).to(torch.int8)

    if q.shape[-1] % 2 != 0:
        q = F.pad(q, (0, 1), mode="constant", value=0)

    q_i16 = q.to(torch.int16)
    low = (q_i16[..., 0::2] & 0x0F).to(torch.uint8)
    high = ((q_i16[..., 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = low | high
    return packed, scale, orig_shape



def _dequantize_tensor_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
) -> torch.Tensor:
    packed_dev = packed.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)

    packed_i16 = packed_dev.to(torch.int16)
    low = packed_i16 & 0x0F
    high = (packed_i16 >> 4) & 0x0F

    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    unpacked = torch.stack((low, high), dim=-1).reshape(
        *packed_dev.shape[:-1], packed_dev.shape[-1] * 2
    )
    unpacked = unpacked[..., : orig_shape[-1]]
    unpacked = unpacked.reshape(orig_shape)

    return (unpacked.to(torch.float32) * scale_dev).to(dtype)


class KVCacheManager:
    def __init__(
        self,
        num_layers: int,
        max_gpu_kv_bytes: int,
        max_seq_len: int,
        enable_cpu_offload: bool = True,
        pin_cpu_memory: bool = True,
        kv_cache_bits: int = 16,
    ) -> None:
        if kv_cache_bits not in (4, 8, 16):
            raise ValueError(
                f"Unsupported kv_cache_bits={kv_cache_bits}; expected 4, 8 or 16"
            )

        self.max_gpu_kv_bytes = max_gpu_kv_bytes
        self.max_seq_len = max_seq_len
        self.enable_cpu_offload = enable_cpu_offload
        self.pin_cpu_memory = pin_cpu_memory
        self.kv_cache_bits = kv_cache_bits
        self.layer_kv: list[LayerKV] = [LayerKV() for _ in range(num_layers)]
        self.current_seq_len = 0

    def _pin_if_needed(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        if x.device.type == "cpu" and self.pin_cpu_memory and not x.is_pinned():
            return x.pin_memory()
        return x

    def _quantize_record_in_place(self, rec: LayerKV) -> None:
        if rec.key is None or rec.value is None:
            return
        if rec.quantized_bits == self.kv_cache_bits:
            return

        if self.kv_cache_bits == 8:
            qk, sk = _quantize_tensor_int8(rec.key)
            qv, sv = _quantize_tensor_int8(rec.value)
            rec.key = self._pin_if_needed(qk)
            rec.value = self._pin_if_needed(qv)
            rec.key_scale = self._pin_if_needed(sk)
            rec.value_scale = self._pin_if_needed(sv)
            rec.key_orig_shape = tuple(rec.key.shape)
            rec.value_orig_shape = tuple(rec.value.shape)
            rec.quantized_bits = 8
        elif self.kv_cache_bits == 4:
            qk, sk, kshape = _quantize_tensor_int4(rec.key)
            qv, sv, vshape = _quantize_tensor_int4(rec.value)
            rec.key = self._pin_if_needed(qk)
            rec.value = self._pin_if_needed(qv)
            rec.key_scale = self._pin_if_needed(sk)
            rec.value_scale = self._pin_if_needed(sv)
            rec.key_orig_shape = kshape
            rec.value_orig_shape = vshape
            rec.quantized_bits = 4
        else:
            rec.key_scale = None
            rec.value_scale = None
            rec.key_orig_shape = tuple(rec.key.shape)
            rec.value_orig_shape = tuple(rec.value.shape)
            rec.quantized_bits = 16

        rec.bytes_total = _record_nbytes(rec)

    def _move_record_to_device(
        self,
        rec: LayerKV,
        device: str | torch.device,
        *,
        non_blocking: bool,
    ) -> None:
        if rec.key is None or rec.value is None:
            return

        rec.key = rec.key.to(device, non_blocking=non_blocking)
        rec.value = rec.value.to(device, non_blocking=non_blocking)
        if rec.key_scale is not None:
            rec.key_scale = rec.key_scale.to(device, non_blocking=non_blocking)
        if rec.value_scale is not None:
            rec.value_scale = rec.value_scale.to(device, non_blocking=non_blocking)

        rec.key = self._pin_if_needed(rec.key)
        rec.value = self._pin_if_needed(rec.value)
        rec.key_scale = self._pin_if_needed(rec.key_scale)
        rec.value_scale = self._pin_if_needed(rec.value_scale)

        rec.device = str(torch.device(device))
        rec.bytes_total = _record_nbytes(rec)

    def get_layer_kv(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Return tensors suitable for passing to model attention.

        When kv_cache_bits == 8 or 4, the resident cache is stored quantized on
        either CPU or GPU. This method materializes temporary dequantized tensors
        on the cache's current device for use by the layer forward pass.
        """
        rec = self.layer_kv[layer_idx]
        if rec.key is None or rec.value is None:
            return None, None

        if rec.quantized_bits == 8:
            assert rec.key_scale is not None and rec.value_scale is not None
            assert rec.orig_dtype is not None
            key = _dequantize_tensor_int8(
                rec.key, rec.key_scale, rec.orig_dtype, rec.key.device
            )
            value = _dequantize_tensor_int8(
                rec.value, rec.value_scale, rec.orig_dtype, rec.value.device
            )
            return key, value

        if rec.quantized_bits == 4:
            assert rec.key_scale is not None and rec.value_scale is not None
            assert rec.orig_dtype is not None
            assert rec.key_orig_shape is not None and rec.value_orig_shape is not None
            key = _dequantize_tensor_int4(
                rec.key,
                rec.key_scale,
                rec.orig_dtype,
                rec.key.device,
                rec.key_orig_shape,
            )
            value = _dequantize_tensor_int4(
                rec.value,
                rec.value_scale,
                rec.orig_dtype,
                rec.value.device,
                rec.value_orig_shape,
            )
            return key, value

        return rec.key, rec.value

    def update_layer_kv(
        self,
        layer_idx: int,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        rec = self.layer_kv[layer_idx]

        if key is None or value is None:
            rec.key = None
            rec.value = None
            rec.key_scale = None
            rec.value_scale = None
            rec.orig_dtype = None
            rec.key_orig_shape = None
            rec.value_orig_shape = None
            rec.device = "none"
            rec.bytes_total = 0
            rec.ready_event = None
            rec.quantized_bits = 16
            return

        rec.orig_dtype = key.dtype
        rec.device = str(key.device)
        rec.key_orig_shape = tuple(key.shape)
        rec.value_orig_shape = tuple(value.shape)

        is_cuda = key.device.type == "cuda"
        if is_cuda:
            if stream is None:
                stream = torch.cuda.current_stream(device=key.device)
            with torch.cuda.stream(stream):
                if self.kv_cache_bits == 8:
                    qk, sk = _quantize_tensor_int8(key)
                    qv, sv = _quantize_tensor_int8(value)
                    rec.key = qk
                    rec.value = qv
                    rec.key_scale = sk
                    rec.value_scale = sv
                    rec.quantized_bits = 8
                elif self.kv_cache_bits == 4:
                    qk, sk, _ = _quantize_tensor_int4(key)
                    qv, sv, _ = _quantize_tensor_int4(value)
                    rec.key = qk
                    rec.value = qv
                    rec.key_scale = sk
                    rec.value_scale = sv
                    rec.quantized_bits = 4
                else:
                    rec.key = key
                    rec.value = value
                    rec.key_scale = None
                    rec.value_scale = None
                    rec.quantized_bits = 16
        else:
            if self.kv_cache_bits == 8:
                qk, sk = _quantize_tensor_int8(key)
                qv, sv = _quantize_tensor_int8(value)
                rec.key = self._pin_if_needed(qk)
                rec.value = self._pin_if_needed(qv)
                rec.key_scale = self._pin_if_needed(sk)
                rec.value_scale = self._pin_if_needed(sv)
                rec.quantized_bits = 8
            elif self.kv_cache_bits == 4:
                qk, sk, _ = _quantize_tensor_int4(key)
                qv, sv, _ = _quantize_tensor_int4(value)
                rec.key = self._pin_if_needed(qk)
                rec.value = self._pin_if_needed(qv)
                rec.key_scale = self._pin_if_needed(sk)
                rec.value_scale = self._pin_if_needed(sv)
                rec.quantized_bits = 4
            else:
                rec.key = self._pin_if_needed(key)
                rec.value = self._pin_if_needed(value)
                rec.key_scale = None
                rec.value_scale = None
                rec.quantized_bits = 16

        rec.bytes_total = _record_nbytes(rec)

        if is_cuda:
            if rec.ready_event is None:
                rec.ready_event = torch.cuda.Event()
            rec.ready_event.record(stream)
        else:
            rec.ready_event = None

    def prefetch_layer_kv(self, layer_idx: int, stream: torch.cuda.Stream) -> None:
        rec = self.layer_kv[layer_idx]
        if rec.key is None or rec.value is None:
            return
        if rec.device.startswith("cuda"):
            return

        with torch.cuda.stream(stream):
            self._move_record_to_device(rec, "cuda", non_blocking=True)
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

            if self.kv_cache_bits in (4, 8) and rec.quantized_bits != self.kv_cache_bits:
                self._quantize_record_in_place(rec)

            self._move_record_to_device(rec, "cpu", non_blocking=False)
            rec.ready_event = None
            total_gpu = self.total_gpu_kv_bytes()

    def ensure_layer_on_gpu(self, layer_idx: int) -> None:
        rec = self.layer_kv[layer_idx]
        if rec.key is None or rec.value is None:
            return
        if rec.device.startswith("cuda"):
            return

        self._move_record_to_device(rec, "cuda", non_blocking=True)
        rec.ready_event = None
