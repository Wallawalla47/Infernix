"""Logical sources over a raw GGUF store, and the exact inverses of llama.cpp's value-head order.

Quantized tensors expose their stored block rows (``read_encoded``) so recipes copy them verbatim,
and their exact decoded values; plain tensors (BF16, F32, F16) their exact values. llama.cpp's
gated delta-net exports (``_LinearAttentionVReorderBase``) store the value heads in ggml's tiled
order; ``head_rows``, ``permute_vector`` and ``value_head_columns`` restore Hugging Face order.
"""

from __future__ import annotations

from math import prod

import numpy as np
import torch

from tools.artifact.codecs.ggml import BY_GGML_ID, GGUF_BF16, GGUF_F16, GGUF_F32, GGUF_Q8_0, decode_q8_0, decode_values

from .gguf_raw import GgufStore
from .logical import EncodedRows, LogicalSource, select_rows


def ggml_matrix(store: GgufStore, name: str, shape: tuple[int, int]) -> LogicalSource:
    """A quantized 2-D GGUF tensor: exact values, and its stored rows as EncodedRows."""

    t = store.tensor(name)
    if t.shape != tuple(shape):
        raise ValueError(f"{name}: GGUF shape {t.shape} differs from {shape}")
    k = shape[1]

    if t.ggml_type == GGUF_Q8_0:
        def encoded(begin: int, end: int) -> EncodedRows:
            codes, scales = decode_q8_0(np.ascontiguousarray(store.raw(name, begin, end)), end - begin, k)
            if (codes == -128).any():
                raise ValueError(f"{name}: Q8_0 code -128 has no q8_g32_fp16 encoding")
            return EncodedRows("q8_g32_fp16", torch.from_numpy(codes.copy()).reshape(end - begin, k // 32, 32),
                               torch.from_numpy(scales.copy()).view(torch.float16))

        def values(begin: int, end: int) -> torch.Tensor:
            rows = encoded(begin // k, -(-end // k))
            v = rows.codes.to(torch.float64) * rows.scales.to(torch.float64)[..., None]
            flat = v.reshape(-1)
            start = begin - begin // k * k
            return flat[start : start + end - begin]

        return LogicalSource(tuple(shape), f"{store.path}:{name}", values, encoded)

    if t.ggml_type not in BY_GGML_ID:
        raise ValueError(f"{name}: GGUF type {t.ggml_type} is not a ggml block format this importer stores")
    fmt = BY_GGML_ID[t.ggml_type]

    def encoded(begin: int, end: int) -> EncodedRows:
        rows = np.array(store.raw(name, begin, end))  # a private copy of the stored bytes
        return EncodedRows(fmt.name, torch.from_numpy(rows), torch.empty(0, dtype=torch.uint8))

    def values(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        rows = np.ascontiguousarray(store.raw(name, first, last))
        flat = torch.from_numpy(decode_values(fmt, rows, last - first, k)).reshape(-1)
        start = begin - first * k
        return flat[start : start + end - begin]

    return LogicalSource(tuple(shape), f"{store.path}:{name}", values, encoded)


def plain(store: GgufStore, name: str, shape: tuple[int, ...], *, minus_one: bool = False,
          bf16_exact: bool = True) -> LogicalSource:
    """An F32/F16/BF16 GGUF tensor's exact values (binary64), optionally g - 1; when ``bf16_exact``
    every value must be a BF16 value, so a BF16 binding stores exactly what the GGUF holds."""

    t = store.tensor(name)
    if t.ggml_type not in (GGUF_F32, GGUF_F16, GGUF_BF16):
        raise ValueError(f"{name}: expected an F32, F16 or BF16 tensor")
    if prod(t.shape) != prod(shape):
        raise ValueError(f"{name}: GGUF shape {t.shape} differs from {shape}")
    cache: list[torch.Tensor] = []

    def read(begin: int, end: int) -> torch.Tensor:
        if not cache:
            v = torch.from_numpy(store.plain(name).reshape(-1).astype(np.float64))
            if minus_one:
                v = v - 1.0  # exact in binary64
            if bf16_exact and not torch.equal(v.to(torch.bfloat16).to(torch.float64), v):
                raise ValueError(f"{name}: a value is not exact in BF16" + (" after undoing +1" if minus_one else ""))
            cache.append(v)
        return cache[0][begin:end]

    return LogicalSource(tuple(shape), f"{store.path}:{name}" + (" - 1" if minus_one else ""), read)


def ggml_or_plain(store: GgufStore, name: str, shape: tuple[int, int]) -> LogicalSource:
    """A 2-D GGUF tensor: stored rows for ggml formats and Q8_0, exact values for BF16/F32/F16 (a
    BF16 binding: every value must be a BF16 value, so an F32 matrix is never rounded silently)."""

    t = store.tensor(name)
    if t.ggml_type in (GGUF_F32, GGUF_F16, GGUF_BF16):
        return plain(store, name, shape)
    return ggml_matrix(store, name, shape)


def value_head_slots(heads: int, key_heads: int) -> list[int]:
    # Hugging Face head h = kk * R + r (key head kk, R value heads per key head) is ggml slot
    # g = r * key_heads + kk (llama.cpp ``_LinearAttentionVReorderBase``; its graph pairs value slot
    # g with key head g % key_heads).
    per_key = heads // key_heads
    return [(h % per_key) * key_heads + h // per_key for h in range(heads)]


def head_rows(source: LogicalSource, heads: int, rows_per_head: int, key_heads: int, row_offset: int = 0) -> LogicalSource:
    """Rows of ``heads`` value heads in Hugging Face order from ggml's tiled order."""

    ranges = []
    for g in value_head_slots(heads, key_heads):
        first = row_offset + g * rows_per_head
        ranges.append((first, first + rows_per_head))
    return select_rows(source, tuple(ranges))


def permute_vector(source: LogicalSource, heads: int, key_heads: int) -> LogicalSource:
    """A per-value-head vector (one element per head) in Hugging Face order."""

    order = value_head_slots(heads, key_heads)

    def read(begin: int, end: int) -> torch.Tensor:
        return source.values()[order][begin:end]

    return LogicalSource(source.shape, f"heads({source.label})", read)


def value_head_columns(source: LogicalSource, heads: int, head_width: int, key_heads: int) -> LogicalSource:
    """A matrix whose columns are value heads in ggml's tiled order, with the columns in Hugging Face
    order. Whole 32-column blocks move, so a Q8_0 source's stored blocks stay exact; each head must
    span whole blocks (a block spanning two heads cannot be split without re-quantizing)."""

    block = 32
    n, k = source.shape
    if k != heads * head_width or head_width % block:
        raise ValueError(f"{source.label}: {heads} value heads of {head_width} columns are not whole "
                         f"{block}-column blocks of {k}")
    per_head = head_width // block
    index = torch.tensor([g * per_head + b for g in value_head_slots(heads, key_heads) for b in range(per_head)])

    def read(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        rows = source.values(first * k, last * k).reshape(last - first, k // block, block)
        flat = rows[:, index].reshape(-1)
        start = begin - first * k
        return flat[start : start + end - begin]

    def encoded(begin: int, end: int) -> EncodedRows:
        rows = source.read_encoded(begin, end)
        if rows.format != "q8_g32_fp16":
            raise ValueError(f"{source.label}: {rows.format} rows cannot move column blocks exactly")
        return EncodedRows(rows.format, rows.codes[:, index].contiguous(), rows.scales[:, index].contiguous())

    return LogicalSource((n, k), f"value_head_columns({source.label})", read,
                         encoded if source.read_encoded is not None else None)
