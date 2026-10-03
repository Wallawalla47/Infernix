"""Exact block-scaled FP8 (``fp8_e4m3fn_block128_f32``) in the ``block128_scale_v1`` layout.

The representation of Qwen's FP8 checkpoints, which NVIDIA's NVFP4 checkpoint copies for the MTP
routed experts: E4M3FN codes with one FP32 multiplier (``weight_scale_inv``) per 128 x 128 tile,
edge tiles truncated. ``W[n,k] = e4m3fn(c[n,k]) * s[n // 128, k // 128]``, exact in binary64.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import block128_geometry
from ._tensor_bytes import Payload, _payload_length, _payload_tensor


def _validate(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if bool(((codes & 0x7F) == 0x7F).any()):
        raise ValueError("block-scaled FP8 codes must be finite E4M3FN words")
    words = scales.view(torch.int32)
    if bool(((words < 0) | ((words & 0x7F800000) == 0x7F800000)).any()):
        raise ValueError("block-scaled FP8 multipliers must be nonnegative finite FP32")


def encode_fp8_block128(
    code_words: torch.Tensor, block_scales: torch.Tensor, shape: Sequence[int]
) -> bytes:
    """Encode exact E4M3FN code words ``[..., N, K]`` and FP32 tile multipliers."""

    g = block128_geometry("fp8_e4m3fn_block128_f32", shape)
    lead = tuple(shape[:-2])
    if code_words.dtype != torch.uint8 or tuple(code_words.shape) != (*lead, g.n, g.k):
        raise TypeError(f"FP8 codes must be uint8 with shape {(*lead, g.n, g.k)}")
    if block_scales.dtype != torch.float32 or tuple(block_scales.shape) != (*lead, g.scale_rows, g.scale_cols):
        raise TypeError(f"FP8 block multipliers must be float32 with shape {(*lead, g.scale_rows, g.scale_cols)}")
    codes = code_words.detach().contiguous().cpu()
    scales = block_scales.detach().contiguous().cpu()
    _validate(codes, scales)
    payload = bytearray(g.payload_bytes)
    payload[: g.code_plane_bytes] = codes.numpy().tobytes()
    payload[g.scale_plane_offset :] = scales.numpy().tobytes()
    return bytes(payload)


def decode_fp8_block128_words(
    payload: Payload, shape: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    g = block128_geometry("fp8_e4m3fn_block128_f32", shape)
    if _payload_length(payload) != g.payload_bytes:
        raise ValueError(f"FP8 block payload has {_payload_length(payload)} bytes, expected {g.payload_bytes}")
    raw = _payload_tensor(payload, torch.device("cpu"))
    lead = tuple(shape[:-2])
    codes = raw[: g.code_plane_bytes].clone().reshape(*lead, g.n, g.k)
    scale_bytes = bytearray(raw[g.scale_plane_offset :].numpy().tobytes())
    scales = torch.frombuffer(scale_bytes, dtype=torch.float32).reshape(*lead, g.scale_rows, g.scale_cols).clone()
    _validate(codes, scales)
    return codes, scales


def dequantize_fp8_block128(codes: torch.Tensor, scales: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Exact represented weights in binary64 (4 + 24 significant bits)."""

    n, k = codes.shape[-2:]
    expanded = scales.to(torch.float64).repeat_interleave(block, dim=-2)[..., :n, :]
    expanded = expanded.repeat_interleave(block, dim=-1)[..., :k]
    return codes.view(torch.float8_e4m3fn).to(torch.float64) * expanded
