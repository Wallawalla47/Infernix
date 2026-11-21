"""Exact MXFP8 expert banks in the ``mxfp8_expert_rg16_v1`` layout.

A bank ``[experts, hidden, intermediate]`` (``fp8_e4m3fn_g32_e8m0``: E4M3FN codes with one UE8M0 power-of-two scale
per 32 K-axis elements, ``W = e4m3fn(c) * 2^(e - 127)``) stores, per expert, the gate/up matrix with rows interleaved
(row 2i = gate_i, row 2i+1 = up_i) and then the down matrix. Both are 16-row groups of 528-byte units, one per
32-element K block, ordered row group major and block ascending, the record padded to 4 KiB. A unit is the A operand
of one ``mma.sync.m16n8k32`` E4M3 product in fragment order, so lane ``l = 4 g + t`` loads its four registers as the
16 bytes at ``16 l``:

- bytes ``16 l + 0..3``: row ``g`` at ``k = 4 t .. 4 t + 3``; ``+4..7``: row ``g + 8`` at the same ``k``;
- bytes ``16 l + 8..11``: row ``g`` at ``k = 16 + 4 t ..``; ``+12..15``: row ``g + 8`` at those ``k``;
- bytes ``512 + r``: the UE8M0 scale of the group's row ``r``.

There is no global scale. The words obey the format's rule (``fp8_g32.validate_fp8_group_words``): no NaN code
(``S.1111.111``) and scale words in ``[3, 246]``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..layouts import mxfp8_expert_geometry
from ._tensor_bytes import Payload, _payload_length, _payload_tensor
from .fp8_g32 import validate_fp8_group_words

UNIT = 528


def _matrix(tensor: torch.Tensor, shape: tuple[int, ...], label: str) -> torch.Tensor:
    if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
        raise TypeError(f"{label} must be uint8 with shape {shape}")
    return tensor.detach().contiguous().cpu()


def _check(codes: torch.Tensor, scales: torch.Tensor, label: str) -> None:
    try:
        validate_fp8_group_words(codes, scales)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from None


def _units(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``[E, R, K]`` codes and ``[E, R, K/32]`` scales -> ``[E, R/16 * K/32 * 528]`` bytes."""

    e, r, k = codes.shape
    g, b = r // 16, k // 32
    # e, group, row half (r // 8), row in half (g of the fragment), block, k half, t, k in quad
    c = codes.reshape(e, g, 2, 8, b, 2, 4, 4)
    lanes = c.permute(0, 1, 4, 3, 6, 5, 2, 7).reshape(e, g, b, 512)  # e, group, block, (g, t, khalf, half, kk)
    scale_part = scales.reshape(e, g, 16, b).permute(0, 1, 3, 2)
    return torch.cat((lanes, scale_part), dim=-1).reshape(e, -1)


def _from_units(raw: torch.Tensor, rows: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`_units`: ``[E, bytes]`` -> codes ``[E, R, K]``, scales ``[E, R, K/32]``."""

    e = raw.shape[0]
    g, b = rows // 16, k // 32
    units = raw.reshape(e, g, b, UNIT)
    lanes = units[..., :512].reshape(e, g, b, 8, 4, 2, 2, 4)  # e, group, block, g, t, khalf, half, kk
    codes = lanes.permute(0, 1, 6, 3, 2, 5, 4, 7).reshape(e, rows, k)
    scales = units[..., 512:].permute(0, 1, 3, 2).reshape(e, rows, b)
    return codes, scales


def encode_mxfp8_expert_records(
    gate_codes: torch.Tensor,
    gate_scales: torch.Tensor,
    up_codes: torch.Tensor,
    up_scales: torch.Tensor,
    down_codes: torch.Tensor,
    down_scales: torch.Tensor,
    shape: Sequence[int],
) -> bytes:
    """The records of ``shape[0]`` experts (the whole payload when that is the bank's expert count).

    ``*_codes`` are E4M3FN words: gate/up ``[E, I, H]``, down ``[E, H, I]``; ``*_scales`` UE8M0 words
    ``[E, I, H/32]`` and ``[E, H, I/32]``. No value is converted."""

    geometry = mxfp8_expert_geometry("fp8_e4m3fn_g32_e8m0", shape)
    e, h, i = geometry.experts, geometry.hidden, geometry.intermediate
    gc = _matrix(gate_codes, (e, i, h), "gate codes")
    uc = _matrix(up_codes, (e, i, h), "up codes")
    dc = _matrix(down_codes, (e, h, i), "down codes")
    gs = _matrix(gate_scales, (e, i, h // 32), "gate scales")
    us = _matrix(up_scales, (e, i, h // 32), "up scales")
    ds = _matrix(down_scales, (e, h, i // 32), "down scales")
    for codes, scales, label in ((gc, gs, "gate"), (uc, us, "up"), (dc, ds, "down")):
        _check(codes, scales, label)
    gate_up = torch.stack((gc, uc), dim=2).reshape(e, 2 * i, h)
    gate_up_scales = torch.stack((gs, us), dim=2).reshape(e, 2 * i, h // 32)
    records = torch.cat((_units(gate_up, gate_up_scales), _units(dc, ds)), dim=1)
    payload = torch.zeros(e, geometry.record_stride, dtype=torch.uint8)
    payload[:, : geometry.record_bytes] = records
    return payload.numpy().tobytes()


def decode_mxfp8_expert_bank_words(payload: Payload, shape: Sequence[int]) -> dict[str, torch.Tensor]:
    """Recover the exact source words: the inverse of :func:`encode_mxfp8_expert_records`."""

    geometry = mxfp8_expert_geometry("fp8_e4m3fn_g32_e8m0", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"expert bank payload has {_payload_length(payload)} bytes, expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    e, h, i = geometry.experts, geometry.hidden, geometry.intermediate
    records = raw.reshape(e, geometry.record_stride)
    gate_up, gate_up_scales = _from_units(records[:, : geometry.gate_up_bytes], 2 * i, h)
    down, down_scales = _from_units(records[:, geometry.gate_up_bytes : geometry.record_bytes], h, i)
    gate_up = gate_up.reshape(e, i, 2, h)
    gate_up_scales = gate_up_scales.reshape(e, i, 2, h // 32)
    return {
        "gate_codes": gate_up[:, :, 0].contiguous(),
        "gate_scales": gate_up_scales[:, :, 0].contiguous(),
        "up_codes": gate_up[:, :, 1].contiguous(),
        "up_scales": gate_up_scales[:, :, 1].contiguous(),
        "down_codes": down.contiguous(),
        "down_scales": down_scales.contiguous(),
    }
