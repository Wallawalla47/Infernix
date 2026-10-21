"""Exact E4M3FN code words and UE8M0 group scales (``fp8_e4m3fn_g32_e8m0``) in g32_scale_rows16_v1.

``W[n, k] = e4m3fn(c[n, k]) * 2^(e[n, k // 32] - 127)``: OCP MXFP8 weights as ModelOpt stores them
(``weight`` E4M3FN ``[N, K]`` and ``weight_scale`` UE8M0 bytes ``[N, K / 32]``). The format admits
finite codes and scale words 3 <= e <= 246, the range in which every code times its scale is an
exact, finite BF16 value (the engine's tensor-core route widens each weight to BF16 exactly; real
weight scales lie near 2^-14). UE8M0 0xFF is NaN.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import group_scale_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)

FORMAT = "fp8_e4m3fn_g32_e8m0"
GROUP = 32
MIN_SCALE_WORD = 3    # 2^-9 (the least E4M3FN value) * 2^(3 - 127) = 2^-133, BF16's least subnormal
MAX_SCALE_WORD = 246  # 448 * 2^(246 - 127) < BF16's largest finite value


def validate_fp8_group_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    if bool(((codes & 0x7F) == 0x7F).any()):
        raise ValueError("group-scaled FP8 codes must be finite E4M3FN words")
    if bool(((scales < MIN_SCALE_WORD) | (scales > MAX_SCALE_WORD)).any()):
        raise ValueError(
            f"group-scaled FP8 scales must be UE8M0 words in [{MIN_SCALE_WORD}, {MAX_SCALE_WORD}] (every weight a BF16 value)"
        )


def rows16_scale_plane(scales: torch.Tensor) -> torch.Tensor:
    """Row-major UE8M0 scales ``uint8 [n, groups]`` (n a multiple of 16) in the layout's tile order:
    ``[n / 16][groups][16]``."""

    n, groups = scales.shape
    return scales.reshape(n // 16, 16, groups).permute(0, 2, 1).contiguous().reshape(-1)


def rows16_scale_rows(plane: torch.Tensor, n: int, groups: int) -> torch.Tensor:
    """The inverse of :func:`rows16_scale_plane`: row-major ``uint8 [n, groups]``."""

    return plane.reshape(n // 16, groups, 16).permute(0, 2, 1).contiguous().reshape(n, groups)


def encode_fp8_group_scaled(code_words: torch.Tensor, scale_words: torch.Tensor, shape: Sequence[int]) -> bytes:
    """Encode exact E4M3FN code words ``[n, k]`` and UE8M0 scale words ``[n, k / 32]``."""

    geometry = group_scale_geometry(FORMAT, shape)
    codes = _exact_uint8_matrix(code_words, (geometry.n, geometry.k), "group-scaled FP8 codes")
    scales = _exact_uint8_matrix(scale_words, (geometry.n, geometry.groups), "group-scaled FP8 scales")
    validate_fp8_group_words(codes, scales)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    begin = geometry.scale_plane_offset
    payload[begin : begin + geometry.scale_plane_bytes] = rows16_scale_plane(scales).numpy().tobytes()
    return bytes(payload)


def decode_fp8_group_scaled_words(payload: Payload, shape: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact E4M3FN code words ``uint8 [n, k]`` and UE8M0 scale words ``uint8 [n, k / 32]``."""

    geometry = group_scale_geometry(FORMAT, shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"group-scaled FP8 payload has {_payload_length(payload)} bytes, expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].clone().reshape(geometry.n, geometry.k)
    begin = geometry.scale_plane_offset
    scales = rows16_scale_rows(raw[begin : begin + geometry.scale_plane_bytes].clone(), geometry.n, geometry.groups)
    validate_fp8_group_words(codes, scales)
    return codes, scales


def ue8m0_values(scales: torch.Tensor) -> torch.Tensor:
    """``2^(e - 127)`` of UE8M0 words as binary64 (exact for every finite word, 2^-127 included)."""

    return torch.ldexp(torch.ones(scales.shape, dtype=torch.float64), scales.to(torch.int32) - 127)


def fp8_group_values(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Exact represented weights in binary64 (4 significant bits times a power of two)."""

    groups = scales.to(torch.int64).shape[-1]
    if codes.shape[-1] != groups * GROUP:
        raise ValueError("group-scaled FP8 codes and scales disagree on K")
    widened = ue8m0_values(scales).repeat_interleave(GROUP, dim=-1)
    return codes.view(torch.float8_e4m3fn).to(torch.float64) * widened


def dequantize_fp8_group_scaled(payload: Payload, shape: Sequence[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Reconstruct a group-scaled FP8 matrix from its exact stored words."""

    codes, scales = decode_fp8_group_scaled_words(payload, shape)
    return fp8_group_values(codes, scales).to(dtype)
