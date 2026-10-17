"""GGUF (ggml) block formats Infernix stores bit-exactly, and their exact decoders.

Every format here encodes an element as ``kappa * (d * sub * code - dmin * min_sub)``: an integer
code, an integer sub-scale, a binary16 block scale ``d`` (and, for Q4_K/Q5_K, a second term with
``dmin``) and a power-of-two format constant ``kappa`` (docs/maintainer/tensor-formats.md §ggml).
:func:`decode_terms` returns those integer and binary16 components; :func:`decode_values` their
exact binary64 values (each term has at most 26 significant bits, so binary64 holds it exactly).
These are the converter's oracles: production kernels reproduce the components, never the values.

The byte layouts and formulas are llama.cpp's (``ggml-common.h``, ``ggml-quants.c``
``dequantize_row_*``), MIT. Q2_0 is llama.cpp #24448: blocks of 64, codes ``c - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from . import ggml_tables as tables


@dataclass(frozen=True, slots=True)
class GgmlType:
    name: str          # Infernix format name
    ggml_id: int       # GGML_TYPE_* value in GGUF files
    block_elems: int
    block_bytes: int
    kappa: float       # power-of-two constant of the value formula
    has_min: bool = False


Q2_0 = GgmlType("ggml_q2_0", 42, 64, 18, 1.0)
Q4_K = GgmlType("ggml_q4_k", 12, 256, 144, 1.0, has_min=True)
Q5_K = GgmlType("ggml_q5_k", 13, 256, 176, 1.0, has_min=True)
Q6_K = GgmlType("ggml_q6_k", 14, 256, 210, 1.0)
IQ3_XXS = GgmlType("ggml_iq3_xxs", 18, 256, 98, 0.25)
IQ4_NL = GgmlType("ggml_iq4_nl", 20, 32, 18, 1.0)
IQ3_S = GgmlType("ggml_iq3_s", 21, 256, 110, 1.0)
IQ2_S = GgmlType("ggml_iq2_s", 22, 256, 82, 0.125)
IQ4_XS = GgmlType("ggml_iq4_xs", 23, 256, 136, 1.0)

GGML_TYPES = MappingProxyType({t.name: t for t in (Q2_0, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS)})
BY_GGML_ID = MappingProxyType({t.ggml_id: t for t in GGML_TYPES.values()})

# Plain GGUF types the converter reads as values (not stored as ggml blocks).
GGUF_F32, GGUF_F16, GGUF_Q8_0, GGUF_BF16 = 0, 1, 8, 30
PLAIN_BLOCKS = MappingProxyType({GGUF_F32: (1, 4), GGUF_F16: (1, 2), GGUF_Q8_0: (32, 34), GGUF_BF16: (1, 2)})


def block_geometry(ggml_id: int) -> tuple[int, int]:
    """(elements, bytes) of one block of a GGUF type id."""

    if ggml_id in BY_GGML_ID:
        t = BY_GGML_ID[ggml_id]
        return t.block_elems, t.block_bytes
    if ggml_id in PLAIN_BLOCKS:
        return PLAIN_BLOCKS[ggml_id]
    raise ValueError(f"unsupported GGUF tensor type {ggml_id}")


def row_bytes(fmt: GgmlType, k: int) -> int:
    if k % fmt.block_elems:
        raise ValueError(f"{fmt.name}: K={k} is not a multiple of {fmt.block_elems}")
    return k // fmt.block_elems * fmt.block_bytes


@dataclass(frozen=True, slots=True)
class Terms:
    """Per-element components: value = kappa * (d * sub * code - dmin * min_sub)."""

    code: np.ndarray     # int32 [n, k]
    sub: np.ndarray      # int32 [n, k]
    d: np.ndarray        # float64 [n, k] (binary16 values)
    min_sub: np.ndarray | None = None  # int32 [n, k], Q4_K/Q5_K
    dmin: np.ndarray | None = None     # float64 [n, k]


def _f16(raw: np.ndarray) -> np.ndarray:
    """Binary16 words (uint8 pairs, little-endian) to float64."""

    return np.ascontiguousarray(raw).view("<f2").astype(np.float64)[..., 0]


def _u32(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(raw).view("<u4")


_GRID2S = np.array(tables.IQ2S_GRID, dtype=np.uint64).view(np.uint8).reshape(1024, 8).astype(np.int32)
_GRID3XXS = np.array(tables.IQ3XXS_GRID, dtype=np.uint32).view(np.uint8).reshape(256, 4).astype(np.int32)
_GRID3S = np.array(tables.IQ3S_GRID, dtype=np.uint32).view(np.uint8).reshape(512, 4).astype(np.int32)
_KVALUES = np.array(tables.KVALUES_IQ4NL, dtype=np.int32)
_KSIGNS = np.array(tables.KSIGNS_IQ2XS, dtype=np.uint8)
_BITS8 = (np.uint8(1) << np.arange(8, dtype=np.uint8)).astype(np.uint8)


def _signs(byte: np.ndarray) -> np.ndarray:
    """[..., 8] of +-1 from sign bytes [...]: bit j set means negative."""

    bits = (byte[..., None] & _BITS8) != 0
    return np.where(bits, -1, 1).astype(np.int32)


def _blocks(fmt: GgmlType, raw: np.ndarray, n: int, k: int) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw.reshape(-1)
    nb = k // fmt.block_elems
    if data.size != n * nb * fmt.block_bytes:
        raise ValueError(f"{fmt.name}: {data.size} bytes for [{n}, {k}]")
    return data.reshape(n * nb, fmt.block_bytes)


def _scale_offsets(fmt: GgmlType) -> tuple[int, ...]:
    """Byte offsets of a block's binary16 scale words (d, and dmin for Q4_K/Q5_K)."""

    return (208,) if fmt is Q6_K else (0, 2) if fmt.has_min else (0,)


def require_finite_scales(fmt: GgmlType, raw, n: int, k: int, label: str) -> None:
    """Rejects rows whose binary16 scale words are infinite or NaN: the exact arithmetic reads a
    scale as mantissa * 2^(e - 25) and is defined for finite words only (tensor-formats §3.6)."""

    b = _blocks(fmt, raw, n, k)
    for offset in _scale_offsets(fmt):
        words = np.ascontiguousarray(b[:, offset:offset + 2]).view("<u2")[:, 0]
        bad = np.flatnonzero((words & 0x7C00) == 0x7C00)
        if bad.size:
            block = int(bad[0])
            raise ValueError(f"{label}: {bad.size} non-finite {fmt.name} scale words (first at row "
                             f"{block // (k // fmt.block_elems)}, byte {offset})")


def _k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q4_K/Q5_K get_scale_min_k4 for j = 0..7: [blocks, 8] scales and mins."""

    s = scales.astype(np.int32)
    sc = np.empty((s.shape[0], 8), np.int32)
    mn = np.empty((s.shape[0], 8), np.int32)
    for j in range(8):
        if j < 4:
            sc[:, j] = s[:, j] & 63
            mn[:, j] = s[:, j + 4] & 63
        else:
            sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
            mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


def decode_terms(fmt: GgmlType, raw, n: int, k: int) -> Terms:
    """The exact components of ``n`` rows of ``k`` elements stored as ``fmt`` blocks."""

    b = _blocks(fmt, raw, n, k)
    nb = b.shape[0]
    e = fmt.block_elems
    code = np.empty((nb, e), np.int32)
    sub = np.ones((nb, e), np.int32)
    min_sub = None
    dmin = None
    d = _f16(b[:, 0:2])
    if fmt is Q2_0:
        qs = b[:, 2:18].astype(np.int32)
        j = np.arange(64)
        code[:] = ((qs[:, j // 4] >> (2 * (j % 4))) & 3) - 1
    elif fmt is IQ4_NL:
        qs = b[:, 2:18].astype(np.int32)
        code[:, :16] = _KVALUES[qs & 0xF]
        code[:, 16:] = _KVALUES[qs >> 4]
    elif fmt is IQ4_XS:
        scales_h = b[:, 2:4].copy().view("<u2")[:, 0].astype(np.int32)
        scales_l = b[:, 4:8].astype(np.int32)
        qs = b[:, 8:136].astype(np.int32)
        for ib in range(8):
            ls = ((scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)
            q = qs[:, 16 * ib : 16 * ib + 16]
            code[:, 32 * ib : 32 * ib + 16] = _KVALUES[q & 0xF]
            code[:, 32 * ib + 16 : 32 * ib + 32] = _KVALUES[q >> 4]
            sub[:, 32 * ib : 32 * ib + 32] = (ls - 32)[:, None]
    elif fmt is Q4_K or fmt is Q5_K:
        dmin_b = _f16(b[:, 2:4])
        sc, mn = _k_scale_min(b[:, 4:16])
        if fmt is Q4_K:
            qs = b[:, 16:144].astype(np.int32)
            qh = None
        else:
            qh = b[:, 16:48].astype(np.int32)
            qs = b[:, 48:176].astype(np.int32)
        min_sub = np.empty((nb, e), np.int32)
        for chunk in range(4):
            q = qs[:, 32 * chunk : 32 * chunk + 32]
            lo, hi = q & 0xF, q >> 4
            if qh is not None:
                lo = lo + np.where((qh & (1 << (2 * chunk))) != 0, 16, 0)
                hi = hi + np.where((qh & (1 << (2 * chunk + 1))) != 0, 16, 0)
            base = 64 * chunk
            code[:, base : base + 32] = lo
            code[:, base + 32 : base + 64] = hi
            sub[:, base : base + 32] = sc[:, 2 * chunk][:, None]
            sub[:, base + 32 : base + 64] = sc[:, 2 * chunk + 1][:, None]
            min_sub[:, base : base + 32] = mn[:, 2 * chunk][:, None]
            min_sub[:, base + 32 : base + 64] = mn[:, 2 * chunk + 1][:, None]
        dmin = np.repeat(dmin_b[:, None], e, axis=1)
    elif fmt is Q6_K:
        ql = b[:, 0:128].astype(np.int32)
        qh = b[:, 128:192].astype(np.int32)
        scales = b[:, 192:208].copy().view(np.int8).astype(np.int32)
        d = _f16(b[:, 208:210])
        for half in range(2):
            l = np.arange(32)
            qlh, qhh, sch = ql[:, 64 * half :], qh[:, 32 * half :], scales[:, 8 * half :]
            base = 128 * half
            q1 = (qlh[:, l] & 0xF) | (((qhh[:, l] >> 0) & 3) << 4)
            q2 = (qlh[:, l + 32] & 0xF) | (((qhh[:, l] >> 2) & 3) << 4)
            q3 = (qlh[:, l] >> 4) | (((qhh[:, l] >> 4) & 3) << 4)
            q4 = (qlh[:, l + 32] >> 4) | (((qhh[:, l] >> 6) & 3) << 4)
            for i, q in enumerate((q1, q2, q3, q4)):
                code[:, base + 32 * i : base + 32 * i + 32] = q - 32
                for part in range(2):  # is = l // 16
                    sub[:, base + 32 * i + 16 * part : base + 32 * i + 16 * part + 16] = sch[:, part + 2 * i][:, None]
    elif fmt is IQ2_S:
        qs = b[:, 2:34].astype(np.int32)
        signs = b[:, 34:66]
        qh = b[:, 66:74].astype(np.int32)
        scales = b[:, 74:82].astype(np.int32)
        for ib in range(8):
            for l in range(4):
                index = qs[:, 4 * ib + l] | ((qh[:, ib] << (8 - 2 * l)) & 0x300)
                s = _signs(signs[:, 4 * ib + l])
                base = 32 * ib + 8 * l
                code[:, base : base + 8] = _GRID2S[index] * s
                nibble = (scales[:, ib] >> (4 * (l // 2))) & 0xF
                sub[:, base : base + 8] = (2 * nibble + 1)[:, None]
    elif fmt is IQ3_XXS:
        qs = b[:, 2:66].astype(np.int32)
        aux = _u32(b[:, 66:98]).astype(np.int64)
        for ib in range(8):
            a = aux[:, ib]
            s32 = (2 * (a >> 28) + 1).astype(np.int32)
            for l in range(4):
                signs = _KSIGNS[((a >> (7 * l)) & 127).astype(np.int64)]
                s = _signs(signs)
                g1 = _GRID3XXS[qs[:, 8 * ib + 2 * l]]
                g2 = _GRID3XXS[qs[:, 8 * ib + 2 * l + 1]]
                base = 32 * ib + 8 * l
                code[:, base : base + 4] = g1 * s[:, 0:4]
                code[:, base + 4 : base + 8] = g2 * s[:, 4:8]
                sub[:, base : base + 8] = s32[:, None]
    elif fmt is IQ3_S:
        qs = b[:, 2:66].astype(np.int32)
        qh = b[:, 66:74].astype(np.int32)
        signs = b[:, 74:106]
        scales = b[:, 106:110].astype(np.int32)
        for sb in range(8):
            nibble = (scales[:, sb // 2] >> (4 * (sb % 2))) & 0xF
            for l in range(4):
                i1 = qs[:, 8 * sb + 2 * l] | ((qh[:, sb] << (8 - 2 * l)) & 256)
                i2 = qs[:, 8 * sb + 2 * l + 1] | ((qh[:, sb] << (7 - 2 * l)) & 256)
                s = _signs(signs[:, 4 * sb + l])
                base = 32 * sb + 8 * l
                code[:, base : base + 4] = _GRID3S[i1] * s[:, 0:4]
                code[:, base + 4 : base + 8] = _GRID3S[i2] * s[:, 4:8]
                sub[:, base : base + 8] = (2 * nibble + 1)[:, None]
    else:  # pragma: no cover - closed registry
        raise ValueError(fmt.name)
    shape = (n, k)
    return Terms(
        code.reshape(shape),
        sub.reshape(shape),
        np.repeat(d[:, None], e, axis=1).reshape(shape),
        None if min_sub is None else min_sub.reshape(shape),
        None if dmin is None else dmin.reshape(shape),
    )


def decode_values(fmt: GgmlType, raw, n: int, k: int) -> np.ndarray:
    """Exact binary64 values [n, k]."""

    t = decode_terms(fmt, raw, n, k)
    value = t.d * (t.sub.astype(np.float64) * t.code.astype(np.float64))
    if t.min_sub is not None:
        value = value - t.dmin * t.min_sub.astype(np.float64)
    return value * fmt.kappa


def decode_q8_0(raw, n: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """GGUF Q8_0 rows as (int8 codes [n, k], binary16 scales [n, k / 32] as uint16 words)."""

    data = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) else raw.reshape(-1)
    if k % 32 or data.size != n * k // 32 * 34:
        raise ValueError("Q8_0 rows must be whole 32-element blocks")
    b = data.reshape(n * k // 32, 34)
    scales = b[:, 0:2].copy().view("<u2")[:, 0].reshape(n, k // 32)
    codes = b[:, 2:34].copy().view(np.int8).reshape(n, k)
    return codes, scales
