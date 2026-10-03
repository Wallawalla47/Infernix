"""Exact ModelOpt NVFP4 expert banks in the ``nvfp4_expert_rg16_v1`` layout.

A bank ``[experts, hidden, intermediate]`` stores, per expert, the gate/up matrix with rows
interleaved (row 2i = gate_i, row 2i+1 = up_i) and then the down matrix. Both are 16-row groups
of 144-byte units, one per 16-element K block, ordered row group major (design
docs/maintainer/qwen3_8-flash-next-design.md §6.2):

- 128 code bytes as four 32-byte quads; in quad q, byte j holds row ``j // 4`` (low nibble) and
  row ``8 + j // 4`` (high nibble) at ``k = 16 b + 4 q + j % 4``;
- 16 E4M3FN scale bytes, row i of the group at byte i.

After the records comes one plane of FP32 ``weight_scale_2`` multipliers, ``[experts, 3]`` in
(gate, up, down) order. Activation input scales are model-role tensors, not part of this format.
"""

from __future__ import annotations

import struct
from typing import Sequence

import torch

from ..formats import valid_positive_fp32_word
from ..layouts import ExpertBankGeometry, expert_bank_geometry
from ._tensor_bytes import Payload, _payload_length, _payload_tensor


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """``[..., R, K/2]`` packed bytes (even element in the low nibble) -> ``[..., R, K]``."""

    return torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4)


def _check_scales(scales: torch.Tensor, label: str) -> None:
    if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
        raise ValueError(f"{label}: NVFP4 scales must be nonnegative finite E4M3FN words")


def _units(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """``[E, R, K]`` codes and ``[E, R, K/16]`` scales -> ``[E, R/16 * K/16 * 144]`` bytes."""

    e, r, k = codes.shape
    g, b = r // 16, k // 16
    c = codes.reshape(e, g, 2, 8, b, 4, 4)  # half, row in half, block, quad, k in quad
    quad_bytes = (c[:, :, 0] | (c[:, :, 1] << 4)).permute(0, 1, 3, 4, 2, 5)  # e,g,b,q,row,kk
    code_part = quad_bytes.reshape(e, g, b, 128)
    scale_part = scales.reshape(e, g, 16, b).permute(0, 1, 3, 2)
    return torch.cat((code_part, scale_part), dim=-1).reshape(e, -1)


def _from_units(raw: torch.Tensor, rows: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`_units`: ``[E, bytes]`` -> codes ``[E, R, K]``, scales ``[E, R, K/16]``."""

    e = raw.shape[0]
    g, b = rows // 16, k // 16
    units = raw.reshape(e, g, b, 144)
    quad_bytes = units[..., :128].reshape(e, g, b, 4, 8, 4).permute(0, 1, 4, 2, 3, 5)  # e,g,row,b,q,kk
    low, high = quad_bytes & 0xF, quad_bytes >> 4
    codes = torch.stack((low, high), dim=2).reshape(e, rows, k)
    scales = units[..., 128:].permute(0, 1, 3, 2).reshape(e, rows, b)
    return codes, scales


def _matrix(tensor: torch.Tensor, shape: tuple[int, ...], label: str) -> torch.Tensor:
    if tensor.dtype != torch.uint8 or tuple(tensor.shape) != shape:
        raise TypeError(f"{label} must be uint8 with shape {shape}")
    return tensor.detach().contiguous().cpu()


def _fp32_words(multipliers: torch.Tensor, experts: int) -> torch.Tensor:
    if multipliers.dtype != torch.float32 or tuple(multipliers.shape) != (experts, 3):
        raise TypeError(f"weight multipliers must be float32 with shape ({experts}, 3)")
    words = multipliers.detach().contiguous().cpu().view(torch.int32).reshape(-1).tolist()
    for word in words:
        if not valid_positive_fp32_word(word & 0xFFFFFFFF):
            raise ValueError("weight multipliers must be finite and positive")
    return multipliers.detach().contiguous().cpu()


def encode_nvfp4_expert_bank(
    gate_codes: torch.Tensor,
    gate_scales: torch.Tensor,
    up_codes: torch.Tensor,
    up_scales: torch.Tensor,
    down_codes: torch.Tensor,
    down_scales: torch.Tensor,
    multipliers: torch.Tensor,
    shape: Sequence[int],
) -> bytes:
    """Encode exact ModelOpt words of a bank ``[E, H, I]`` without numerical conversion.

    ``*_codes`` are the checkpoint's packed E2M1 bytes (even element in the low nibble):
    gate/up ``[E, I, H/2]`` and down ``[E, H, I/2]``. ``*_scales`` are E4M3FN words
    ``[E, I, H/16]`` and ``[E, H, I/16]``. ``multipliers`` is ``[E, 3]`` FP32
    ``weight_scale_2`` in (gate, up, down) order, stored bit-exactly.
    """

    geometry = expert_bank_geometry("nvfp4_mul", shape)
    e, h, i = geometry.experts, geometry.hidden, geometry.intermediate
    gc = _matrix(gate_codes, (e, i, h // 2), "gate codes")
    uc = _matrix(up_codes, (e, i, h // 2), "up codes")
    dc = _matrix(down_codes, (e, h, i // 2), "down codes")
    gs = _matrix(gate_scales, (e, i, h // 16), "gate scales")
    us = _matrix(up_scales, (e, i, h // 16), "up scales")
    ds = _matrix(down_scales, (e, h, i // 16), "down scales")
    for scales, label in ((gs, "gate"), (us, "up"), (ds, "down")):
        _check_scales(scales, label)
    mult = _fp32_words(multipliers, e)

    gate_up_codes = torch.stack((_unpack_nibbles(gc), _unpack_nibbles(uc)), dim=2).reshape(e, 2 * i, h)
    gate_up_scales = torch.stack((gs, us), dim=2).reshape(e, 2 * i, h // 16)
    records = torch.cat(
        (_units(gate_up_codes, gate_up_scales), _units(_unpack_nibbles(dc), ds)), dim=1
    )
    payload = bytearray(geometry.payload_bytes)
    view = torch.frombuffer(payload, dtype=torch.uint8)
    stride = geometry.record_stride
    view[: e * stride].view(e, stride)[:, : geometry.record_bytes] = records
    payload[geometry.multiplier_offset :] = mult.numpy().tobytes()
    return bytes(payload)


def decode_nvfp4_expert_bank_words(
    payload: Payload, shape: Sequence[int]
) -> dict[str, torch.Tensor]:
    """Recover the exact source words: the inverse of :func:`encode_nvfp4_expert_bank`."""

    geometry = expert_bank_geometry("nvfp4_mul", shape)
    if _payload_length(payload) != geometry.payload_bytes:
        raise ValueError(
            f"expert bank payload has {_payload_length(payload)} bytes, "
            f"expected {geometry.payload_bytes}"
        )
    raw = _payload_tensor(payload, torch.device("cpu"))
    e, h, i = geometry.experts, geometry.hidden, geometry.intermediate
    records = raw[: e * geometry.record_stride].reshape(e, geometry.record_stride)
    gate_up, gate_up_scales = _from_units(records[:, : geometry.gate_up_bytes], 2 * i, h)
    down, down_scales = _from_units(
        records[:, geometry.gate_up_bytes : geometry.record_bytes], h, i
    )
    for scales, label in ((gate_up_scales, "gate/up"), (down_scales, "down")):
        _check_scales(scales, label)
    gate_up = gate_up.reshape(e, i, 2, h)
    gate_up_scales = gate_up_scales.reshape(e, i, 2, h // 16)
    mult_bytes = bytearray(raw[geometry.multiplier_offset :].numpy().tobytes())
    multipliers = torch.frombuffer(mult_bytes, dtype=torch.float32).reshape(e, 3).clone()
    return {
        "gate_codes": _pack_nibbles(gate_up[:, :, 0]).contiguous(),
        "gate_scales": gate_up_scales[:, :, 0].contiguous(),
        "up_codes": _pack_nibbles(gate_up[:, :, 1]).contiguous(),
        "up_scales": gate_up_scales[:, :, 1].contiguous(),
        "down_codes": _pack_nibbles(down).contiguous(),
        "down_scales": down_scales.contiguous(),
        "multipliers": multipliers,
    }


_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
)


def _e4m3(words: torch.Tensor) -> torch.Tensor:
    return words.view(torch.float8_e4m3fn).to(torch.float64)


def dequantize_nvfp4_mul(
    packed_codes: torch.Tensor, scales: torch.Tensor, multiplier: float
) -> torch.Tensor:
    """Exact represented weights of one ``nvfp4_mul`` matrix, in binary64.

    ``W[n,k] = e2m1(c[n,k]) * e4m3fn(s[n,k/16]) * m_w``; binary64 holds every product exactly.
    """

    values = _E2M1[_unpack_nibbles(packed_codes).long()]
    return values * _e4m3(scales).repeat_interleave(16, dim=-1) * float(multiplier)


def expert_bank_record_offset(geometry: ExpertBankGeometry, expert: int) -> int:
    """Byte offset of one expert's record: what the loader DMAs into a frame."""

    if not 0 <= expert < geometry.experts:
        raise IndexError("expert index out of range")
    return expert * geometry.record_stride


def fp32_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]
