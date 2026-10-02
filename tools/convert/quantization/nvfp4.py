"""Floating-point-to-NVFP4 weight quantization (E2M1 codes, E4M3FN group scales, FP32 divisor).

A matrix ``W[N,K]`` is represented as ``W[n,k] = e2m1(c[n,k]) * e4m3fn(s[n,k//16]) / d_w``
(``docs/maintainer/tensor-formats.md`` section 3.3). One divisor ``d_w`` serves the whole matrix.
``quantize_nvfp4`` maps every 16-value K group's largest magnitude to the largest E2M1 code;
``quantize_nvfp4_mse`` instead gives each group the E4M3FN scale, among all 126 positive finite
ones, with the least squared reconstruction error (ModelOpt's NVFP4 weight-MSE scale sweep).
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

GROUP = 16
E2M1_MAX = 6.0
E4M3FN_MAX = 448.0
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Midpoints between consecutive magnitudes; a tie rounds to the even code.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Every positive finite E4M3FN scale word, in increasing value (0x7F is NaN).
_E4M3FN_POSITIVE_WORDS = range(0x01, 0x7F)


@dataclass(frozen=True, slots=True)
class Nvfp4Words:
    codes: torch.Tensor  # uint8 [N, K/2]; the low nibble holds the even K coordinate
    scales: torch.Tensor  # uint8 [N, K/16] natural E4M3FN words
    weight_divisor: bytes  # little-endian binary32


def weight_divisor(amax: float, target: float = E2M1_MAX) -> float:
    """Return the binary32 matrix divisor for a parent whose largest magnitude is ``amax``.

    The parent's largest group reaches the largest E4M3FN scale when its largest magnitude maps to
    ``target`` (6 is the largest E2M1 code); a smaller target leaves that group room for larger
    scales.
    """

    if not 0.0 < target <= E2M1_MAX:
        raise ValueError("NVFP4 divisor target must lie in (0, 6]")
    if not amax > 0.0 or amax != amax or amax == float("inf"):
        return 1.0
    divisor = struct.unpack("<f", struct.pack("<f", E4M3FN_MAX * target / float(amax)))[0]
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


def _groups(values: torch.Tensor) -> torch.Tensor:
    if values.dim() != 2 or values.shape[1] % GROUP or values.shape[0] <= 0:
        raise ValueError("NVFP4 quantization needs a rank-two matrix with K % 16 == 0")
    if not values.dtype.is_floating_point:
        raise TypeError("NVFP4 quantization source must be floating point")
    x = values.to(torch.float32)
    if not bool(torch.isfinite(x).all()):
        raise ValueError("NVFP4 quantization source contains NaN or infinity")
    rows, k = x.shape
    return x.reshape(rows, k // GROUP, GROUP)


def _encode(groups: torch.Tensor, scale_words: torch.Tensor, divisor: float) -> Nvfp4Words:
    """Round every group to E2M1 codes at its stored scale, ties to even."""

    rows, blocks, _ = groups.shape
    scale = _decode_e4m3fn(scale_words)
    live = scale > 0
    step = torch.where(live, scale / divisor, torch.ones_like(scale))
    normalized = torch.clamp(groups.abs() / step[..., None], max=E2M1_MAX)
    code = torch.where(live[..., None], round_e2m1(normalized), 0)
    negative = (groups < 0) & (code > 0)
    words = (code | (negative.to(torch.int32) << 3)).to(torch.uint8).reshape(rows, blocks * GROUP)
    packed = (words[:, 0::2] | (words[:, 1::2] << 4)).contiguous()
    return Nvfp4Words(
        packed.cpu(),
        scale_words.reshape(rows, blocks).contiguous().cpu(),
        struct.pack("<f", divisor),
    )


def quantize_nvfp4(values: torch.Tensor, divisor: float) -> Nvfp4Words:
    """Quantize ``[rows, K]`` values with each group's largest magnitude at the largest code."""

    groups = _groups(values)
    amax = groups.abs().amax(dim=-1)
    raw_scale = torch.clamp(amax * divisor / E2M1_MAX, max=E4M3FN_MAX)
    return _encode(groups, raw_scale.to(torch.float8_e4m3fn).view(torch.uint8), divisor)


def quantize_nvfp4_mse(values: torch.Tensor, divisor: float) -> Nvfp4Words:
    """Quantize ``[rows, K]`` values with each group's least-squared-error E4M3FN scale.

    Every positive finite E4M3FN scale is evaluated for every group; ties keep the smaller scale.
    An all-zero group stores the zero scale.
    """

    groups = _groups(values)
    magnitude = groups.abs()
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=groups.device)
    midpoints = torch.tensor(_MIDPOINTS, dtype=torch.float32, device=groups.device)
    best_error = None
    best_words = None
    for word in _E4M3FN_POSITIVE_WORDS:
        step = float(_decode_e4m3fn(torch.tensor([word], dtype=torch.uint8))) / divisor
        normalized = torch.clamp(magnitude / step, max=E2M1_MAX)
        # A value on a midpoint has the same error at either neighbouring code, so the search
        # rounds ties down; the stored codes round them to even (_encode).
        code = torch.bucketize(normalized, midpoints)
        error = (magnitudes[code] * step - magnitude).square().sum(dim=-1)
        if best_error is None:
            best_error = error
            best_words = torch.full(error.shape, word, dtype=torch.uint8, device=groups.device)
        else:
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_words = torch.where(better, word, best_words)
    zero = magnitude.amax(dim=-1) == 0
    best_words = torch.where(zero, 0, best_words).to(torch.uint8)
    return _encode(groups, best_words, divisor)
