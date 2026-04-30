from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


QUANT_MODE_SEQ_CHANNEL = "seq_channel"
QUANT_MODE_HEAD_BLOCK = "head_block"
QUANT_MODE_SINGLE_SCALE = "single_scale"
QUANT_MODE_PER_HEAD = "per_head"

VALID_QUANT_MODES = (
    QUANT_MODE_SEQ_CHANNEL,
    QUANT_MODE_HEAD_BLOCK,
    QUANT_MODE_SINGLE_SCALE,
    QUANT_MODE_PER_HEAD,
)

_BLOCK_BASED_QUANT_MODES = (QUANT_MODE_SEQ_CHANNEL, QUANT_MODE_HEAD_BLOCK)

# Toggle the quantization mode here. KVCacheManager uses these when the
# constructor is not given explicit k_quant_mode/v_quant_mode arguments.
DEFAULT_K_QUANT_MODE = QUANT_MODE_SEQ_CHANNEL
DEFAULT_V_QUANT_MODE = QUANT_MODE_HEAD_BLOCK

# Backward-compatible aliases.
KEY_QUANT_MODE_SEQ_CHANNEL = QUANT_MODE_SEQ_CHANNEL
VALUE_QUANT_MODE_HEAD_BLOCK = QUANT_MODE_HEAD_BLOCK


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

    # K and V intentionally use different block layouts.
    #   K: one scale per channel over a block of sequence positions.
    #   V: one scale per token over a block of head-dimension channels.
    key_quant_mode: str | None = None
    value_quant_mode: str | None = None
    key_quant_block_size: int | None = None
    value_quant_block_size: int | None = None


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


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _effective_block_size(dim_size: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError(f"quant block size must be positive, got {block_size}")
    return min(block_size, dim_size)


# -----------------------------------------------------------------------------
# V quantization: per-token, head-dimension blocks.
# Tensor shape convention: [..., seq_len, head_dim]
# For standard HF KV cache this is usually [batch, num_heads, seq_len, head_dim].
# -----------------------------------------------------------------------------


def _pad_last_dim_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    pad = (-x.shape[-1]) % multiple
    if pad == 0:
        return x
    return F.pad(x, (0, pad), mode="constant", value=0)


def _head_block_view(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """View a [..., padded_head_dim] tensor as [..., num_blocks, block_size]."""
    return x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)


def _quantize_tensor_int8_head_block(
    x: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """
    Symmetric int8 quantization over contiguous head-dimension blocks.

    This is the recommended V-cache path: values are quantized per token, with
    each token vector split into smaller head-dimension blocks.

    Returns:
        q:          int8 tensor in the original shape
        scale:      float32 tensor shaped [..., num_head_blocks, 1]
        orig_shape: original tensor shape for dequantization
    """
    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)
    bsz = _effective_block_size(orig_shape[-1], block_size)

    x_padded = _pad_last_dim_to_multiple(x_detached, bsz)
    x_blocks = _head_block_view(x_padded, bsz)

    max_abs = x_blocks.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / 127.0).clamp_min(1e-8).to(torch.float32)
    q_blocks = torch.round(x_blocks / scale).clamp_(-127, 127).to(torch.int8)
    q_padded = q_blocks.reshape(*x_padded.shape)
    q = q_padded[..., : orig_shape[-1]].reshape(orig_shape)
    return q, scale, orig_shape


def _dequantize_tensor_int8_head_block(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
    block_size: int,
) -> torch.Tensor:
    q_dev = q.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)

    bsz = _effective_block_size(orig_shape[-1], block_size)
    q_padded = _pad_last_dim_to_multiple(q_dev, bsz)
    q_blocks = _head_block_view(q_padded, bsz)
    deq_blocks = q_blocks.to(torch.float32) * scale_dev
    deq_padded = deq_blocks.reshape(*q_padded.shape)
    deq = deq_padded[..., : orig_shape[-1]].reshape(orig_shape)
    return deq.to(dtype)


def _quantize_tensor_int4_head_block(
    x: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """
    Symmetric packed int4 quantization over contiguous head-dimension blocks.

    Returns:
        packed:     uint8 tensor containing two signed int4 values per byte
        scale:      float32 tensor shaped [..., num_head_blocks, 1]
        orig_shape: original tensor shape needed for unpacking/dequantization
    """
    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)
    bsz = _effective_block_size(orig_shape[-1], block_size)

    x_padded = _pad_last_dim_to_multiple(x_detached, bsz)
    x_blocks = _head_block_view(x_padded, bsz)

    max_abs = x_blocks.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / 7.0).clamp_min(1e-8).to(torch.float32)
    q_blocks = torch.round(x_blocks / scale).clamp_(-8, 7).to(torch.int8)
    q_padded = q_blocks.reshape(*x_padded.shape)

    # Packing needs an even number of int4 entries.
    if q_padded.shape[-1] % 2 != 0:
        q_padded = F.pad(q_padded, (0, 1), mode="constant", value=0)

    q_i16 = q_padded.to(torch.int16)
    low = (q_i16[..., 0::2] & 0x0F).to(torch.uint8)
    high = ((q_i16[..., 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = low | high
    return packed, scale, orig_shape


def _unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    packed_i16 = packed.to(torch.int16)
    low = packed_i16 & 0x0F
    high = (packed_i16 >> 4) & 0x0F

    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    return torch.stack((low, high), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def _dequantize_tensor_int4_head_block(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
    block_size: int,
) -> torch.Tensor:
    packed_dev = packed.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)

    unpacked = _unpack_signed_int4(packed_dev)

    bsz = _effective_block_size(orig_shape[-1], block_size)
    padded_last_dim = _ceil_div(orig_shape[-1], bsz) * bsz
    unpacked = unpacked[..., :padded_last_dim]

    q_blocks = _head_block_view(unpacked, bsz)
    deq_blocks = q_blocks.to(torch.float32) * scale_dev
    deq_padded = deq_blocks.reshape(*unpacked.shape)
    deq = deq_padded[..., : orig_shape[-1]].reshape(orig_shape)
    return deq.to(dtype)


# -----------------------------------------------------------------------------
# K quantization: per-channel, sequence blocks.
# Tensor shape convention: [..., seq_len, head_dim]
# This creates one scale for each channel over a block of sequence positions.
# -----------------------------------------------------------------------------


def _pad_seq_dim_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad the second-to-last dimension: [..., seq_len, head_dim]."""
    pad = (-x.shape[-2]) % multiple
    if pad == 0:
        return x
    return F.pad(x, (0, 0, 0, pad), mode="constant", value=0)


def _slice_seq_dim(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Slice the second-to-last dimension: [..., seq_len, head_dim]."""
    return x[..., :seq_len, :]


def _seq_channel_block_view(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    View [..., padded_seq_len, head_dim] as
    [..., num_seq_blocks, block_size, head_dim].
    """
    return x.reshape(
        *x.shape[:-2],
        x.shape[-2] // block_size,
        block_size,
        x.shape[-1],
    )


def _quantize_tensor_int8_seq_channel(
    x: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """
    Symmetric int8 K-cache quantization: per channel over sequence blocks.

    For x shaped [B, H, S, D], scale is shaped [B, H, S_blocks, 1, D].
    This reduces the effect of key-channel outliers versus quantizing each full
    token vector with one scale.
    """
    if x.ndim < 2:
        raise ValueError("sequence/channel quantization expects at least 2 dims")

    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)
    bsz = _effective_block_size(orig_shape[-2], block_size)

    x_padded = _pad_seq_dim_to_multiple(x_detached, bsz)
    x_blocks = _seq_channel_block_view(x_padded, bsz)

    max_abs = x_blocks.abs().amax(dim=-2, keepdim=True)
    scale = (max_abs / 127.0).clamp_min(1e-8).to(torch.float32)
    q_blocks = torch.round(x_blocks / scale).clamp_(-127, 127).to(torch.int8)
    q_padded = q_blocks.reshape(*x_padded.shape)
    q = _slice_seq_dim(q_padded, orig_shape[-2]).reshape(orig_shape)
    return q, scale, orig_shape


def _dequantize_tensor_int8_seq_channel(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
    block_size: int,
) -> torch.Tensor:
    q_dev = q.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)

    bsz = _effective_block_size(orig_shape[-2], block_size)
    q_padded = _pad_seq_dim_to_multiple(q_dev, bsz)
    q_blocks = _seq_channel_block_view(q_padded, bsz)
    deq_blocks = q_blocks.to(torch.float32) * scale_dev
    deq_padded = deq_blocks.reshape(*q_padded.shape)
    deq = _slice_seq_dim(deq_padded, orig_shape[-2]).reshape(orig_shape)
    return deq.to(dtype)


def _quantize_tensor_int4_seq_channel(
    x: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """
    Symmetric packed int4 K-cache quantization: per channel over sequence blocks.

    For x shaped [B, H, S, D], scale is shaped [B, H, S_blocks, 1, D].
    The int4 values are still byte-packed along head_dim for compact storage.
    """
    if x.ndim < 2:
        raise ValueError("sequence/channel quantization expects at least 2 dims")

    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)
    bsz = _effective_block_size(orig_shape[-2], block_size)

    x_padded = _pad_seq_dim_to_multiple(x_detached, bsz)
    x_blocks = _seq_channel_block_view(x_padded, bsz)

    max_abs = x_blocks.abs().amax(dim=-2, keepdim=True)
    scale = (max_abs / 7.0).clamp_min(1e-8).to(torch.float32)
    q_blocks = torch.round(x_blocks / scale).clamp_(-8, 7).to(torch.int8)
    q_padded = q_blocks.reshape(*x_padded.shape)

    # Store only the real sequence positions; dequantization pads the sequence
    # dimension back to a multiple of the K block size before applying scales.
    q = _slice_seq_dim(q_padded, orig_shape[-2]).reshape(orig_shape)

    # Packing is still along the head/channel dimension.
    if q.shape[-1] % 2 != 0:
        q = F.pad(q, (0, 1), mode="constant", value=0)

    q_i16 = q.to(torch.int16)
    low = (q_i16[..., 0::2] & 0x0F).to(torch.uint8)
    high = ((q_i16[..., 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = low | high
    return packed, scale, orig_shape


def _dequantize_tensor_int4_seq_channel(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
    block_size: int,
) -> torch.Tensor:
    packed_dev = packed.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)

    unpacked = _unpack_signed_int4(packed_dev)
    unpacked = unpacked[..., : orig_shape[-1]]

    bsz = _effective_block_size(orig_shape[-2], block_size)
    q_padded = _pad_seq_dim_to_multiple(unpacked, bsz)
    q_blocks = _seq_channel_block_view(q_padded, bsz)

    deq_blocks = q_blocks.to(torch.float32) * scale_dev
    deq_padded = deq_blocks.reshape(*q_padded.shape)
    deq = _slice_seq_dim(deq_padded, orig_shape[-2]).reshape(orig_shape)
    return deq.to(dtype)


# -----------------------------------------------------------------------------
# Single-scale quantization: one scale value for the entire tensor.
# Cheapest to compute and store; coarsest accuracy.
# -----------------------------------------------------------------------------


def _quantize_tensor_int8_single_scale(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)
    max_abs = x_detached.abs().amax().reshape(1)
    scale = (max_abs / 127.0).clamp_min(1e-8).to(torch.float32)
    q = torch.round(x_detached / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale, orig_shape


def _dequantize_tensor_int8_single_scale(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
) -> torch.Tensor:
    q_dev = q.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)
    deq = q_dev.to(torch.float32) * scale_dev
    return deq.reshape(orig_shape).to(dtype)


def _quantize_tensor_int4_single_scale(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)
    max_abs = x_detached.abs().amax().reshape(1)
    scale = (max_abs / 7.0).clamp_min(1e-8).to(torch.float32)
    q = torch.round(x_detached / scale).clamp_(-8, 7).to(torch.int8)

    if q.shape[-1] % 2 != 0:
        q = F.pad(q, (0, 1), mode="constant", value=0)

    q_i16 = q.to(torch.int16)
    low = (q_i16[..., 0::2] & 0x0F).to(torch.uint8)
    high = ((q_i16[..., 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = low | high
    return packed, scale, orig_shape


def _dequantize_tensor_int4_single_scale(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
) -> torch.Tensor:
    packed_dev = packed.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)
    unpacked = _unpack_signed_int4(packed_dev)
    unpacked = unpacked[..., : orig_shape[-1]]
    deq = unpacked.to(torch.float32) * scale_dev
    return deq.reshape(orig_shape).to(dtype)


# -----------------------------------------------------------------------------
# Per-head quantization: one scale per head index.
# Tensor shape convention: [..., H, S, D] (e.g. [B, H, S, D] for HF KV cache).
# Scale is reduced over the seq and head_dim axes only, so the head dim at -3
# keeps an independent scale.
# -----------------------------------------------------------------------------


def _quantize_tensor_int8_per_head(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if x.ndim < 3:
        raise ValueError(
            "per-head quantization requires at least 3 dims [..., H, S, D]"
        )
    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)
    max_abs = x_detached.abs().amax(dim=(-2, -1), keepdim=True)
    scale = (max_abs / 127.0).clamp_min(1e-8).to(torch.float32)
    q = torch.round(x_detached / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale, orig_shape


def _dequantize_tensor_int8_per_head(
    q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
) -> torch.Tensor:
    q_dev = q.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)
    deq = q_dev.to(torch.float32) * scale_dev
    return deq.reshape(orig_shape).to(dtype)


def _quantize_tensor_int4_per_head(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if x.ndim < 3:
        raise ValueError(
            "per-head quantization requires at least 3 dims [..., H, S, D]"
        )
    x_detached = x.detach()
    orig_shape = tuple(x_detached.shape)
    max_abs = x_detached.abs().amax(dim=(-2, -1), keepdim=True)
    scale = (max_abs / 7.0).clamp_min(1e-8).to(torch.float32)
    q = torch.round(x_detached / scale).clamp_(-8, 7).to(torch.int8)

    if q.shape[-1] % 2 != 0:
        q = F.pad(q, (0, 1), mode="constant", value=0)

    q_i16 = q.to(torch.int16)
    low = (q_i16[..., 0::2] & 0x0F).to(torch.uint8)
    high = ((q_i16[..., 1::2] & 0x0F) << 4).to(torch.uint8)
    packed = low | high
    return packed, scale, orig_shape


def _dequantize_tensor_int4_per_head(
    packed: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
) -> torch.Tensor:
    packed_dev = packed.to(device, non_blocking=True)
    scale_dev = scale.to(device, non_blocking=True)
    unpacked = _unpack_signed_int4(packed_dev)
    unpacked = unpacked[..., : orig_shape[-1]]
    deq = unpacked.to(torch.float32) * scale_dev
    return deq.reshape(orig_shape).to(dtype)


# -----------------------------------------------------------------------------
# Mode dispatch.
# -----------------------------------------------------------------------------


def _quantize_dispatch(
    x: torch.Tensor,
    mode: str,
    bits: int,
    block_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if bits == 8:
        if mode == QUANT_MODE_SEQ_CHANNEL:
            assert block_size is not None
            return _quantize_tensor_int8_seq_channel(x, block_size)
        if mode == QUANT_MODE_HEAD_BLOCK:
            assert block_size is not None
            return _quantize_tensor_int8_head_block(x, block_size)
        if mode == QUANT_MODE_SINGLE_SCALE:
            return _quantize_tensor_int8_single_scale(x)
        if mode == QUANT_MODE_PER_HEAD:
            return _quantize_tensor_int8_per_head(x)
    elif bits == 4:
        if mode == QUANT_MODE_SEQ_CHANNEL:
            assert block_size is not None
            return _quantize_tensor_int4_seq_channel(x, block_size)
        if mode == QUANT_MODE_HEAD_BLOCK:
            assert block_size is not None
            return _quantize_tensor_int4_head_block(x, block_size)
        if mode == QUANT_MODE_SINGLE_SCALE:
            return _quantize_tensor_int4_single_scale(x)
        if mode == QUANT_MODE_PER_HEAD:
            return _quantize_tensor_int4_per_head(x)
    raise RuntimeError(f"Unsupported quantization config: mode={mode}, bits={bits}")


def _dequantize_dispatch(
    q: torch.Tensor,
    scale: torch.Tensor,
    mode: str,
    bits: int,
    dtype: torch.dtype,
    device: str | torch.device,
    orig_shape: tuple[int, ...],
    block_size: int | None,
) -> torch.Tensor:
    if bits == 8:
        if mode == QUANT_MODE_SEQ_CHANNEL:
            assert block_size is not None
            return _dequantize_tensor_int8_seq_channel(
                q, scale, dtype, device, orig_shape, block_size
            )
        if mode == QUANT_MODE_HEAD_BLOCK:
            assert block_size is not None
            return _dequantize_tensor_int8_head_block(
                q, scale, dtype, device, orig_shape, block_size
            )
        if mode == QUANT_MODE_SINGLE_SCALE:
            return _dequantize_tensor_int8_single_scale(
                q, scale, dtype, device, orig_shape
            )
        if mode == QUANT_MODE_PER_HEAD:
            return _dequantize_tensor_int8_per_head(
                q, scale, dtype, device, orig_shape
            )
    elif bits == 4:
        if mode == QUANT_MODE_SEQ_CHANNEL:
            assert block_size is not None
            return _dequantize_tensor_int4_seq_channel(
                q, scale, dtype, device, orig_shape, block_size
            )
        if mode == QUANT_MODE_HEAD_BLOCK:
            assert block_size is not None
            return _dequantize_tensor_int4_head_block(
                q, scale, dtype, device, orig_shape, block_size
            )
        if mode == QUANT_MODE_SINGLE_SCALE:
            return _dequantize_tensor_int4_single_scale(
                q, scale, dtype, device, orig_shape
            )
        if mode == QUANT_MODE_PER_HEAD:
            return _dequantize_tensor_int4_per_head(
                q, scale, dtype, device, orig_shape
            )
    raise RuntimeError(f"Unsupported quantization config: mode={mode}, bits={bits}")


class KVCacheManager:
    def __init__(
        self,
        num_layers: int,
        max_gpu_kv_bytes: int,
        max_seq_len: int,
        enable_cpu_offload: bool = True,
        pin_cpu_memory: bool = True,
        kv_cache_bits: int = 16,
        # Backward-compatible default. If the K/V-specific block sizes below are
        # not supplied, both use this value.
        kv_quant_block_size: int = 32,
        # Asymmetric K/V defaults.
        k_seq_block_size: int | None = None,
        v_head_block_size: int | None = None,
        # Quantization mode per K/V. Block sizes above only apply when the mode
        # is one of the block-based modes (seq_channel, head_block). Defaults
        # come from the module-level DEFAULT_{K,V}_QUANT_MODE constants so the
        # mode can be toggled in one place without touching call sites.
        k_quant_mode: str | None = None,
        v_quant_mode: str | None = None,
    ) -> None:
        if k_quant_mode is None:
            k_quant_mode = DEFAULT_K_QUANT_MODE
        if v_quant_mode is None:
            v_quant_mode = DEFAULT_V_QUANT_MODE
        if kv_cache_bits not in (4, 8, 16):
            raise ValueError(
                f"Unsupported kv_cache_bits={kv_cache_bits}; expected 4, 8 or 16"
            )
        if kv_quant_block_size <= 0:
            raise ValueError(
                f"Unsupported kv_quant_block_size={kv_quant_block_size}; expected > 0"
            )
        if k_quant_mode not in VALID_QUANT_MODES:
            raise ValueError(
                f"Unsupported k_quant_mode={k_quant_mode}; expected one of {VALID_QUANT_MODES}"
            )
        if v_quant_mode not in VALID_QUANT_MODES:
            raise ValueError(
                f"Unsupported v_quant_mode={v_quant_mode}; expected one of {VALID_QUANT_MODES}"
            )

        self.max_gpu_kv_bytes = max_gpu_kv_bytes
        self.max_seq_len = max_seq_len
        self.enable_cpu_offload = enable_cpu_offload
        self.pin_cpu_memory = pin_cpu_memory
        self.kv_cache_bits = kv_cache_bits

        self.k_quant_mode = k_quant_mode
        self.v_quant_mode = v_quant_mode

        self.k_seq_block_size = (
            kv_quant_block_size if k_seq_block_size is None else k_seq_block_size
        )
        self.v_head_block_size = (
            kv_quant_block_size if v_head_block_size is None else v_head_block_size
        )
        if self.k_seq_block_size <= 0:
            raise ValueError(
                f"Unsupported k_seq_block_size={self.k_seq_block_size}; expected > 0"
            )
        if self.v_head_block_size <= 0:
            raise ValueError(
                f"Unsupported v_head_block_size={self.v_head_block_size}; expected > 0"
            )

        self.layer_kv: list[LayerKV] = [LayerKV() for _ in range(num_layers)]
        self.current_seq_len = 0

    def _k_block_size(self) -> int | None:
        return (
            self.k_seq_block_size
            if self.k_quant_mode in _BLOCK_BASED_QUANT_MODES
            else None
        )

    def _v_block_size(self) -> int | None:
        return (
            self.v_head_block_size
            if self.v_quant_mode in _BLOCK_BASED_QUANT_MODES
            else None
        )

    def reset(self) -> None:
        """Drop all per-layer KV state and reset the seq-len counter.

        Call this between independent requests / batches so the next prefill
        starts at position 0 with no stale K/V from prior runs. Without this,
        ``current_seq_len`` keeps growing across requests and old K/V is fed
        in as ``past_key_value``, corrupting subsequent generations.
        """
        for rec in self.layer_kv:
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
            rec.key_quant_mode = None
            rec.value_quant_mode = None
            rec.key_quant_block_size = None
            rec.value_quant_block_size = None
        self.current_seq_len = 0

    def _pin_if_needed(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        if x.device.type == "cpu" and self.pin_cpu_memory and not x.is_pinned():
            return x.pin_memory()
        return x

    def _record_matches_current_quant_config(self, rec: LayerKV) -> bool:
        if rec.quantized_bits != self.kv_cache_bits:
            return False
        if self.kv_cache_bits == 16:
            return True
        if rec.key_quant_mode != self.k_quant_mode:
            return False
        if rec.value_quant_mode != self.v_quant_mode:
            return False
        if rec.key_quant_block_size != self._k_block_size():
            return False
        if rec.value_quant_block_size != self._v_block_size():
            return False
        return True

    def _materialize_record(self, rec: LayerKV) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dequantized key/value on the record's current device."""
        assert rec.key is not None and rec.value is not None
        if rec.quantized_bits == 16:
            return rec.key, rec.value
        if rec.orig_dtype is None:
            raise RuntimeError("Cannot materialize quantized KV without orig_dtype")
        if rec.key_orig_shape is None or rec.value_orig_shape is None:
            raise RuntimeError("Cannot materialize quantized KV without original shapes")
        if rec.key_quant_mode is None or rec.value_quant_mode is None:
            raise RuntimeError("Cannot materialize quantized KV without quant modes")
        if rec.key_quant_mode in _BLOCK_BASED_QUANT_MODES and rec.key_quant_block_size is None:
            raise RuntimeError("Cannot materialize block-quantized K without block size")
        if rec.value_quant_mode in _BLOCK_BASED_QUANT_MODES and rec.value_quant_block_size is None:
            raise RuntimeError("Cannot materialize block-quantized V without block size")

        assert rec.key_scale is not None and rec.value_scale is not None
        key = _dequantize_dispatch(
            rec.key,
            rec.key_scale,
            rec.key_quant_mode,
            rec.quantized_bits,
            rec.orig_dtype,
            rec.key.device,
            rec.key_orig_shape,
            rec.key_quant_block_size,
        )
        value = _dequantize_dispatch(
            rec.value,
            rec.value_scale,
            rec.value_quant_mode,
            rec.quantized_bits,
            rec.orig_dtype,
            rec.value.device,
            rec.value_orig_shape,
            rec.value_quant_block_size,
        )
        return key, value

    def _store_quantized_record(
        self,
        rec: LayerKV,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        pin_cpu_result: bool,
    ) -> None:
        rec.orig_dtype = key.dtype

        if self.kv_cache_bits in (8, 4):
            k_block_size = self._k_block_size()
            v_block_size = self._v_block_size()
            qk, sk, kshape = _quantize_dispatch(
                key, self.k_quant_mode, self.kv_cache_bits, k_block_size
            )
            qv, sv, vshape = _quantize_dispatch(
                value, self.v_quant_mode, self.kv_cache_bits, v_block_size
            )
            rec.key = self._pin_if_needed(qk) if pin_cpu_result else qk
            rec.value = self._pin_if_needed(qv) if pin_cpu_result else qv
            rec.key_scale = self._pin_if_needed(sk) if pin_cpu_result else sk
            rec.value_scale = self._pin_if_needed(sv) if pin_cpu_result else sv
            rec.key_orig_shape = kshape
            rec.value_orig_shape = vshape
            rec.quantized_bits = self.kv_cache_bits
            rec.key_quant_mode = self.k_quant_mode
            rec.value_quant_mode = self.v_quant_mode
            rec.key_quant_block_size = k_block_size
            rec.value_quant_block_size = v_block_size
        else:
            rec.key = self._pin_if_needed(key) if pin_cpu_result else key
            rec.value = self._pin_if_needed(value) if pin_cpu_result else value
            rec.key_scale = None
            rec.value_scale = None
            rec.key_orig_shape = tuple(key.shape)
            rec.value_orig_shape = tuple(value.shape)
            rec.quantized_bits = 16
            rec.key_quant_mode = None
            rec.value_quant_mode = None
            rec.key_quant_block_size = None
            rec.value_quant_block_size = None

    def _quantize_record_in_place(self, rec: LayerKV) -> None:
        if rec.key is None or rec.value is None:
            return
        if self._record_matches_current_quant_config(rec):
            return

        key_src, value_src = self._materialize_record(rec)
        pin_cpu_result = key_src.device.type == "cpu"
        self._store_quantized_record(
            rec,
            key_src,
            value_src,
            pin_cpu_result=pin_cpu_result,
        )
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

        When kv_cache_bits == 8 or 4, resident cache tensors are stored
        quantized on either CPU or GPU. This method materializes temporary
        dequantized tensors on the cache's current device for the layer forward.
        """
        rec = self.layer_kv[layer_idx]
        if rec.key is None or rec.value is None:
            return None, None
        if rec.quantized_bits == 16:
            return rec.key, rec.value
        return self._materialize_record(rec)

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
            rec.key_quant_mode = None
            rec.value_quant_mode = None
            rec.key_quant_block_size = None
            rec.value_quant_block_size = None
            return

        rec.device = str(key.device)
        is_cuda = key.device.type == "cuda"

        if is_cuda:
            if stream is None:
                stream = torch.cuda.current_stream(device=key.device)
            with torch.cuda.stream(stream):
                self._store_quantized_record(
                    rec,
                    key,
                    value,
                    pin_cpu_result=False,
                )
        else:
            self._store_quantized_record(
                rec,
                key,
                value,
                pin_cpu_result=True,
            )

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

            if self.kv_cache_bits in (4, 8) and not self._record_matches_current_quant_config(rec):
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
