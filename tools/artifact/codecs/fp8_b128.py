"""Block-scaled FP8 (``fp8_e4m3fn_b128``) and its expert record (``fp8_expert_b128_v1``), exactly.

The format is the official Qwen3.8-Flash-Next-FP8 checkpoint's (docs/maintainer/tensor-formats.md §3.7,
docs/maintainer/storage-layouts.md §9): ``W[n, k] = e4m3fn(c[n, k]) * s[n // 128, k // 128]``
with one E4M3FN code byte per weight and one BF16 scale per 128 x 128 tile. Every E4M3FN value is an
integer multiple of 2^-9, so the exact arithmetic (design §16.2.2) uses ``w_int = e4m3fn(c) * 2^9``:

    normal (e in 1..15):  w_int = (-1)^s * (8 + m) * 2^(e - 1)     |w_int| <= 14 * 2^14 < 2^18
    subnormal (e = 0):    w_int = (-1)^s * m

The NaN codes (S.1111.111) have no value and are refused. A scale must be a positive, normal, finite
BF16 value. Within a row's tiles the scale exponents span at most :data:`MAX_EXPONENT_SPREAD` binades,
so a row's exact sum fits the int128 accumulator of the production routes.

One record per expert, 4 KiB aligned (``record_stride``):

    [ gate rows, row-major, k bytes each ] [ up rows ] [ down rows ]
    [ scale plane: gate tiles, up tiles, down tiles, each row-major [n/128][k/128], BF16 words ]
    [ zero padding to 4 KiB ]

Every row is addressed as ``base + r * k`` like a GGUF bank's (ops/offloaded_sparse_moe), so routes
read rows in place; the scale plane follows the codes.
"""

from __future__ import annotations

import numpy as np

from ..layouts import Fp8ExpertGeometry, fp8_expert_geometry

TILE = 128
MAX_EXPONENT_SPREAD = 62  # a tile term's shift in the exact sum (ops/common/exact_combine.h add_shifted)
NAN_CODES = (0x7F, 0xFF)


def e4m3fn_int(codes: np.ndarray) -> np.ndarray:
    """The exact integer ``e4m3fn(c) * 2^9`` of each code byte (int32); refuses NaN codes."""

    c = np.asarray(codes, dtype=np.uint8)
    if np.isin(c, NAN_CODES).any():
        raise ValueError("an E4M3FN NaN code has no value")
    c = c.astype(np.int32)
    sign = np.where(c & 0x80, -1, 1)
    e = (c >> 3) & 15
    m = c & 7
    magnitude = np.where(e == 0, m, (8 + m) << np.maximum(e - 1, 0))
    return (sign * magnitude).astype(np.int32)


def e4m3fn_values(codes: np.ndarray) -> np.ndarray:
    """The exact binary64 value of each code (the converter's oracle)."""

    return np.ldexp(e4m3fn_int(codes).astype(np.float64), -9)


def bf16_values(words: np.ndarray) -> np.ndarray:
    """BF16 words (uint16) as exact binary64 values."""

    return (np.asarray(words, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def check_scales(words: np.ndarray, label: str) -> None:
    """Every scale word is a positive normal finite BF16 value and each row of tiles spans at most
    MAX_EXPONENT_SPREAD binades."""

    w = np.asarray(words, dtype=np.uint16)
    exponent = (w >> 7) & 0xFF
    if (w & 0x8000).any() or (exponent == 0).any() or (exponent == 0xFF).any():
        raise ValueError(f"{label}: a block scale is not a positive normal finite BF16 value")
    rows = exponent.reshape(w.shape[0], -1).astype(np.int32)
    spread = int((rows.max(axis=1) - rows.min(axis=1)).max())
    if spread > MAX_EXPONENT_SPREAD:
        raise ValueError(f"{label}: a row's block scales span {spread} binades (at most {MAX_EXPONENT_SPREAD})")


def dequantize(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Exact binary64 W[n, k] of a block-scaled matrix (scales: BF16 words [n/128, k/128])."""

    n, k = codes.shape
    if n % TILE or k % TILE or scales.shape != (n // TILE, k // TILE):
        raise ValueError(f"block-FP8 {codes.shape} with scales {scales.shape}: not whole 128 x 128 tiles")
    s = np.repeat(np.repeat(bf16_values(scales), TILE, axis=0), TILE, axis=1)
    return e4m3fn_values(codes) * s  # each product is exact: 4 + 8 significant bits


def expert_geometry(hidden: int, intermediate: int) -> Fp8ExpertGeometry:
    """One record's geometry (the layout's, for a one-expert bank)."""

    return fp8_expert_geometry("fp8_e4m3fn_b128", (1, hidden, intermediate))


def encode_expert(gate: tuple[np.ndarray, np.ndarray], up: tuple[np.ndarray, np.ndarray],
                  down: tuple[np.ndarray, np.ndarray], label: str = "expert") -> bytes:
    """One record from (codes uint8, scale words uint16) of gate [I, H], up [I, H] and down [H, I]."""

    (gc, gs), (uc, us), (dc, ds) = gate, up, down
    inter, hidden = gc.shape
    g = expert_geometry(hidden, inter)
    for name, codes, scales, shape in (("gate", gc, gs, (inter, hidden)), ("up", uc, us, (inter, hidden)),
                                       ("down", dc, ds, (hidden, inter))):
        if codes.shape != shape or codes.dtype != np.uint8:
            raise ValueError(f"{label} {name}: codes {codes.shape} {codes.dtype}, expected {shape} uint8")
        if scales.shape != (shape[0] // TILE, shape[1] // TILE):
            raise ValueError(f"{label} {name}: scales {scales.shape} do not tile {shape}")
        if np.isin(codes, NAN_CODES).any():
            raise ValueError(f"{label} {name}: an E4M3FN NaN code has no value")
        check_scales(scales, f"{label} {name}")
    out = bytearray(g.record_stride)
    at = 0
    for codes in (gc, uc, dc):
        raw = np.ascontiguousarray(codes).tobytes()
        out[at : at + len(raw)] = raw
        at += len(raw)
    for scales in (gs, us, ds):
        raw = np.ascontiguousarray(scales, dtype="<u2").tobytes()
        out[at : at + len(raw)] = raw
        at += len(raw)
    assert at == g.record_bytes
    return bytes(out)


def decode_expert(record: bytes | np.ndarray, hidden: int, intermediate: int):
    """(gate, up, down) as (codes, scale words) from one record, the inverse of :func:`encode_expert`."""

    g = expert_geometry(hidden, intermediate)
    raw = np.frombuffer(record, dtype=np.uint8) if not isinstance(record, np.ndarray) else record.reshape(-1)
    if raw.size < g.record_bytes:
        raise ValueError("fp8_expert_b128_v1 record is truncated")
    gc = raw[: g.gate_bytes].reshape(intermediate, hidden)
    uc = raw[g.gate_bytes : 2 * g.gate_bytes].reshape(intermediate, hidden)
    dc = raw[2 * g.gate_bytes : g.scale_offset].reshape(hidden, intermediate)
    words = raw[g.scale_offset : g.record_bytes].copy().view("<u2")
    t = g.tiles
    gs = words[:t].reshape(intermediate // TILE, hidden // TILE)
    us = words[t : 2 * t].reshape(intermediate // TILE, hidden // TILE)
    ds = words[2 * t : 3 * t].reshape(hidden // TILE, intermediate // TILE)
    return (gc, gs), (uc, us), (dc, ds)
