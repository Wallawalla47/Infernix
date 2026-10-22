"""Exact E4M3FN code words and per-row multipliers in row_scale_v1 layout.

``fp8_e4m3fn_row_bf16`` stores one BF16 multiplier per row and ``fp8_e4m3fn_row_f32`` one FP32
multiplier per row; the code plane is the same.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..formats import Fp8RowFormat, get_format
from ..layouts import row_scale_geometry
from ._tensor_bytes import (
    Payload,
    _exact_uint8_matrix,
    _payload_length,
    _payload_tensor,
)
from .direct import decode_direct, encode_direct

_SCALE_DTYPES = {2: (torch.bfloat16, "bf16"), 4: (torch.float32, "fp32")}


def _row_format(format: str) -> Fp8RowFormat:
    spec = get_format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError(f"{format} is not a row-scaled FP8 format")
    return spec


def _exact_scale_vector(tensor: torch.Tensor, length: int, spec: Fp8RowFormat) -> torch.Tensor:
    dtype, name = _SCALE_DTYPES[spec.scale_bytes]
    if tensor.dtype != dtype or tuple(tensor.shape) != (length,):
        raise TypeError(f"{spec.name} scales must be {name.upper()} with shape ({length},)")
    return tensor.detach().contiguous().cpu()


def validate_fp8_row_words(codes: torch.Tensor, scales: torch.Tensor) -> None:
    """Finite E4M3FN codes and nonnegative finite multipliers (BF16 or FP32 by the scales' dtype);
    a zero multiplier admits only signed-zero codes."""

    if bool(((codes & 0x7F) == 0x7F).any()):
        raise ValueError("row-scaled FP8 codes must be finite E4M3FN words")
    if scales.dtype == torch.bfloat16:
        words = scales.view(torch.int16).to(torch.int32) & 0xFFFF
        invalid = ((words & 0x8000) != 0) | ((words & 0x7F80) == 0x7F80)
        name = "BF16"
    elif scales.dtype == torch.float32:
        words = scales.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        invalid = ((words & 0x80000000) != 0) | ((words & 0x7F800000) == 0x7F800000)
        name = "FP32"
    else:
        raise TypeError("row-scaled FP8 scales must be BF16 or FP32")
    if bool(invalid.any()):
        raise ValueError(f"row-scaled FP8 scales must be nonnegative finite {name} words")
    zero_scale = words == 0
    nonzero_code = (codes & 0x7F) != 0
    if bool((zero_scale.unsqueeze(1) & nonzero_code).any()):
        raise ValueError("a zero row scale requires only signed-zero FP8 codes")


def encode_fp8_row_scaled(
    code_words: torch.Tensor,
    row_scales: torch.Tensor,
    shape: Sequence[int],
    format: str = "fp8_e4m3fn_row_bf16",
) -> bytes:
    """Encode exact E4M3FN code words and the format's row multipliers."""

    spec = _row_format(format)
    geometry = row_scale_geometry(spec, shape)
    codes = _exact_uint8_matrix(code_words, (geometry.n, geometry.k), "row-scaled FP8 codes")
    scales = _exact_scale_vector(row_scales, geometry.n, spec)
    validate_fp8_row_words(codes, scales)
    payload = bytearray(geometry.payload_bytes)
    payload[: geometry.code_plane_bytes] = codes.numpy().tobytes()
    scale_begin = geometry.scale_plane_offset
    payload[scale_begin : scale_begin + geometry.scale_plane_bytes] = encode_direct(
        scales, _SCALE_DTYPES[spec.scale_bytes][1]
    )
    return bytes(payload)


def decode_fp8_row_scaled_words(
    payload: Payload,
    shape: Sequence[int],
    format: str = "fp8_e4m3fn_row_bf16",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode exact E4M3FN code words and the format's row multipliers."""

    spec = _row_format(format)
    geometry = row_scale_geometry(spec, shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"row-scaled FP8 payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    codes = raw[: geometry.code_plane_bytes].clone().reshape(geometry.n, geometry.k)
    scale_begin = geometry.scale_plane_offset
    scale_bytes = raw[scale_begin : scale_begin + geometry.scale_plane_bytes]
    scales = decode_direct(scale_bytes, _SCALE_DTYPES[spec.scale_bytes][1], (geometry.n,))
    validate_fp8_row_words(codes, scales)
    return codes, scales


def dequantize_fp8_row_scaled(
    payload: Payload,
    shape: Sequence[int],
    dtype: torch.dtype = torch.float32,
    format: str = "fp8_e4m3fn_row_bf16",
) -> torch.Tensor:
    """Reconstruct a row-scaled FP8 matrix from its exact stored words. Binary64 holds every
    product exactly (4 + 24 significant bits); a narrower dtype rounds once."""

    codes, scales = decode_fp8_row_scaled_words(payload, shape, format)
    values = codes.view(torch.float8_e4m3fn).double() * scales.double().unsqueeze(1)
    return values.to(dtype)
