"""Floating-point-to-NVFP4 weight quantization (E2M1 codes, E4M3FN group scales, FP32 divisor).

A matrix ``W[N,K]`` is represented as ``W[n,k] = e2m1(c[n,k]) * e4m3fn(s[n,k//16]) / d_w``
(``docs/maintainer/tensor-formats.md`` section 3.3). Every 16-value K group maps its largest
magnitude to one of the requested ``targets`` (6 maps it to the largest E2M1 code); with several
targets the group keeps the one with the least squared reconstruction error, ties keeping the
earlier target. One divisor ``d_w`` serves the whole matrix, so the parent's largest group fits
the E4M3FN range at the smallest target.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Sequence

import torch

GROUP = 16
E2M1_MAX = 6.0
E4M3FN_MAX = 448.0
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between consecutive magnitudes; a tie rounds to the even code.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@dataclass(frozen=True, slots=True)
class Nvfp4Words:
    codes: torch.Tensor  # uint8 [N, K/2]; the low nibble holds the even K coordinate
    scales: torch.Tensor  # uint8 [N, K/16] natural E4M3FN words
    weight_divisor: bytes  # little-endian binary32


def weight_divisor(amax: float, targets: Sequence[float]) -> float:
    """Return the binary32 matrix divisor for a parent whose largest magnitude is ``amax``."""

    if not targets or any(not 0.0 < t <= E2M1_MAX for t in targets):
        raise ValueError("NVFP4 targets must lie in (0, 6]")
    if not amax > 0.0 or amax != amax or amax == float("inf"):
        return 1.0
    divisor = struct.unpack(
        "<f", struct.pack("<f", E4M3FN_MAX * min(targets) / float(amax))
    )[0]
    if not 0.0 < divisor < float("inf"):
        raise ValueError("NVFP4 weight divisor is not a finite positive binary32 value")
    return divisor


def round_e2m1(magnitude: torch.Tensor) -> torch.Tensor:
    """Round nonnegative values in ``[0, 6]`` to E2M1 magnitude codes, ties to even."""

    code = torch.zeros(magnitude.shape, dtype=torch.int32, device=magnitude.device)
    for index, midpoint in enumerate(_MIDPOINTS):
        code += (magnitude > midpoint).to(torch.int32)
        if index % 2 == 1:  # a tie between codes `index` (odd) and `index+1` (even)
            code += (magnitude == midpoint).to(torch.int32)
    return code


def _decode_e4m3fn(words: torch.Tensor) -> torch.Tensor:
    return words.view(torch.float8_e4m3fn).to(torch.float32)


def quantize_nvfp4(
    values: torch.Tensor, divisor: float, targets: Sequence[float] = (E2M1_MAX,)
) -> Nvfp4Words:
    """Quantize floating ``[rows, K]`` values with a fixed matrix divisor."""

    if values.dim() != 2 or values.shape[1] % GROUP or values.shape[0] <= 0:
        raise ValueError("NVFP4 quantization needs a rank-two matrix with K % 16 == 0")
    if not values.dtype.is_floating_point:
        raise TypeError("NVFP4 quantization source must be floating point")
    x = values.to(torch.float32)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("NVFP4 quantization source contains NaN or infinity")
    rows, k = x.shape
    groups = x.reshape(rows, k // GROUP, GROUP)
    magnitude = groups.abs()
    amax = magnitude.amax(dim=-1)
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=x.device)
    d = torch.tensor(divisor, dtype=torch.float32, device=x.device)

    best_error = None
    best_codes = None
    best_scales = None
    for target in targets:
        raw_scale = torch.clamp(amax * d / target, max=E4M3FN_MAX)
        scale_words = raw_scale.to(torch.float8_e4m3fn).view(torch.uint8)
        scale = _decode_e4m3fn(scale_words)
        live = scale > 0
        step = torch.where(live, scale / d, torch.ones_like(scale))
        normalized = torch.clamp(magnitude / step[..., None], max=E2M1_MAX)
        code = torch.where(live[..., None], round_e2m1(normalized), 0)
        error = (
            (magnitudes[code.long()] * step[..., None] * live[..., None] - magnitude)
            .square()
            .sum(dim=-1)
        )
        if best_error is None:
            best_error, best_codes, best_scales = error, code, scale_words
        else:
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_codes = torch.where(better[..., None], code, best_codes)
            best_scales = torch.where(better, scale_words, best_scales)

    negative = (groups < 0) & (best_codes > 0)
    words = (best_codes | (negative.to(torch.int32) << 3)).to(torch.uint8).reshape(rows, k)
    packed = (words[:, 0::2] | (words[:, 1::2] << 4)).contiguous()
    return Nvfp4Words(
        packed.cpu(),
        best_scales.reshape(rows, k // GROUP).contiguous().cpu(),
        struct.pack("<f", divisor),
    )
