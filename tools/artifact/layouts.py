"""Persistent tensor layouts shared by Infernix converters and reference models.

This module maps already-selected numeric words to their registered byte layout.
It deliberately does not quantize floating-point source weights or assign model
roles to tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
import operator
from types import MappingProxyType
from typing import Sequence


from .formats import (
    GGML_FORMATS,
    DirectFormat,
    Fp8BlockFormat,
    Fp8RowFormat,
    GgmlFormat,
    Nvfp4Format,
    Nvfp4MulFormat,
    NumericFormat,
    QuantFormat,
    get_format,
)

PLANE_ALIGNMENT = 256
K_ALIGNMENT = 128


@dataclass(frozen=True, slots=True)
class Layout:
    name: str
    alignment: int
    formats: frozenset[str]


@dataclass(frozen=True, slots=True)
class RowSplitGeometry:
    n: int
    k: int
    k_pad: int
    groups_per_row: int
    base_bytes_per_group: int
    high_bytes_per_group: int
    base_row_bytes: int
    high_row_bytes: int
    scale_row_bytes: int
    base_offset: int
    base_bytes: int
    high_offset: int
    high_bytes: int
    scale_offset: int
    scale_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class BlockScaleGeometry:
    n: int
    k: int
    groups_per_row: int
    k_tiles: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    weight_divisor_offset: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class RowScaleGeometry:
    n: int
    k: int
    code_plane_bytes: int
    scale_plane_offset: int
    scale_plane_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class ExpertBankGeometry:
    """``nvfp4_expert_rg16_v1``: one record per expert, then the FP32 multiplier plane."""

    experts: int
    hidden: int
    intermediate: int
    gate_up_row_groups: int
    gate_up_blocks: int
    down_row_groups: int
    down_blocks: int
    gate_up_bytes: int
    record_bytes: int
    record_stride: int
    multiplier_offset: int
    multiplier_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class GgmlRowsGeometry:
    """``ggml_rows_v1``: [n, k] as n rows of whole GGUF blocks, verbatim and contiguous."""

    n: int
    k: int
    blocks_per_row: int
    row_bytes: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class GgmlExpertGeometry:
    """``ggml_expert_*_v1``: one 4 KiB-aligned record per expert of [experts, hidden, intermediate]:
    the gate rows, the up rows ([intermediate, hidden] each, in the bank's format), then the down
    rows ([hidden, intermediate], in the layout's down format), every row verbatim GGUF blocks."""

    experts: int
    hidden: int
    intermediate: int
    gate_up_format: str
    down_format: str
    gate_row_bytes: int
    down_row_bytes: int
    gate_bytes: int     # one of gate, up; up starts at gate_bytes
    down_offset: int    # 2 * gate_bytes
    record_bytes: int
    record_stride: int
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class Fp8ExpertGeometry:
    """``fp8_expert_b128_v1``: one 4 KiB-aligned record per expert of [experts, hidden, intermediate]:
    the gate rows, the up rows ([intermediate, hidden] each), the down rows ([hidden, intermediate]),
    one E4M3FN byte per weight, then the scale plane: each matrix's BF16 128 x 128 tile scales,
    row-major [n/128][k/128], gate, up, down (tools/artifact/codecs/fp8_b128.py)."""

    experts: int
    hidden: int
    intermediate: int
    gate_bytes: int     # one of gate, up; up starts at gate_bytes
    down_offset: int    # 2 * gate_bytes
    scale_offset: int   # 2 * gate_bytes + down bytes
    tiles: int          # scale words per matrix
    record_bytes: int
    record_stride: int
    payload_bytes: int


EXPERT_UNIT_BYTES = 144
EXPERT_RECORD_ALIGNMENT = 4096

CONTIGUOUS_LE_V1 = Layout("contiguous_le_v1", 256, frozenset(("bf16", "fp32", "int32")))
ROW_SPLIT_K128_V1 = Layout(
    "row_split_k128_v1",
    256,
    frozenset(("q2_g64_fp16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")),
)
BLOCK_SCALE_K16_M128X4_V1 = Layout(
    "block_scale_k16_m128x4_v1",
    256,
    frozenset(("nvfp4",)),
)
ROW_SCALE_V1 = Layout(
    "row_scale_v1",
    256,
    frozenset(("fp8_e4m3fn_row_bf16",)),
)

NVFP4_EXPERT_RG16_V1 = Layout(
    "nvfp4_expert_rg16_v1",
    EXPERT_RECORD_ALIGNMENT,
    frozenset(("nvfp4_mul",)),
)

GGML_ROWS_V1 = Layout("ggml_rows_v1", 256, frozenset(GGML_FORMATS))

# The dense formats the tiled narrow kernels decode (ops/ggml/ggml_linear.cu, 256-element blocks).
GGML_TILES16_FORMATS = frozenset(("ggml_q4_k", "ggml_q5_k", "ggml_q6_k", "ggml_iq4_xs"))
GGML_TILES16_V1 = Layout("ggml_tiles16_v1", 256, GGML_TILES16_FORMATS)

# Expert banks of GGUF formats: the layout names the down matrices' format; the bank's own format
# is the gate/up format.
_GGML_EXPERT_GATE_UP = frozenset(("ggml_iq2_s", "ggml_iq3_xxs", "ggml_iq3_s", "ggml_iq4_xs"))
GGML_EXPERT_IQ4_NL_V1 = Layout("ggml_expert_iq4_nl_v1", EXPERT_RECORD_ALIGNMENT, _GGML_EXPERT_GATE_UP)
GGML_EXPERT_Q2_0_V1 = Layout("ggml_expert_q2_0_v1", EXPERT_RECORD_ALIGNMENT, _GGML_EXPERT_GATE_UP)
GGML_EXPERT_DOWN_FORMATS = MappingProxyType(
    {GGML_EXPERT_IQ4_NL_V1.name: "ggml_iq4_nl", GGML_EXPERT_Q2_0_V1.name: "ggml_q2_0"}
)

FP8_EXPERT_B128_V1 = Layout("fp8_expert_b128_v1", EXPERT_RECORD_ALIGNMENT, frozenset(("fp8_e4m3fn_b128",)))

LAYOUTS = MappingProxyType(
    {
        layout.name: layout
        for layout in (
            CONTIGUOUS_LE_V1,
            ROW_SPLIT_K128_V1,
            BLOCK_SCALE_K16_M128X4_V1,
            ROW_SCALE_V1,
            NVFP4_EXPERT_RG16_V1,
            GGML_ROWS_V1,
            GGML_TILES16_V1,
            GGML_EXPERT_IQ4_NL_V1,
            GGML_EXPERT_Q2_0_V1,
            FP8_EXPERT_B128_V1,
        )
    }
)


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires a nonnegative value and positive alignment")
    return (value + alignment - 1) // alignment * alignment


def get_layout(name: str) -> Layout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"unknown tensor layout: {name!r}") from None


def _format(value: str | NumericFormat) -> NumericFormat:
    if isinstance(value, str):
        return get_format(value)
    registered = get_format(value.name)
    if value != registered:
        raise ValueError(f"numeric format does not match registered {value.name!r}")
    return registered


def _layout(value: str | Layout) -> Layout:
    if isinstance(value, str):
        return get_layout(value)
    registered = get_layout(value.name)
    if value != registered:
        raise ValueError(f"layout does not match registered {value.name!r}")
    return registered


def _shape(value: Sequence[int], *, rank: int | None = None) -> tuple[int, ...]:
    dims = []
    for dim in value:
        if isinstance(dim, bool):
            raise ValueError("shape dimensions must be positive integers")
        try:
            item = operator.index(dim)
        except TypeError:
            raise ValueError("shape dimensions must be positive integers") from None
        if item <= 0:
            raise ValueError("shape dimensions must be positive integers")
        dims.append(item)
    result = tuple(dims)
    if rank is not None and len(result) != rank:
        raise ValueError(f"layout requires rank {rank}, got rank {len(result)}")
    return result


def row_split_geometry(
    format: str | QuantFormat, shape: Sequence[int]
) -> RowSplitGeometry:
    spec = _format(format)
    if not isinstance(spec, QuantFormat):
        raise ValueError("row_split_k128_v1 requires a grouped quantized format")
    n, k = _shape(shape, rank=2)
    k_pad = align_up(k, K_ALIGNMENT)
    groups_per_row = k_pad // spec.group_size
    # The base plane: whole bytes (8), low nibbles (4-6), or the 2-bit codes four to a byte (2).
    base_bytes_per_group = (
        spec.group_size if spec.bits == 8 else spec.group_size // 4 if spec.bits == 2 else spec.group_size // 2
    )
    high_bytes_per_group = (
        0 if spec.bits in (2, 4, 8) else spec.group_size * (spec.bits - 4) // 8
    )
    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2
    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes
    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    return RowSplitGeometry(
        n=n,
        k=k,
        k_pad=k_pad,
        groups_per_row=groups_per_row,
        base_bytes_per_group=base_bytes_per_group,
        high_bytes_per_group=high_bytes_per_group,
        base_row_bytes=base_row_bytes,
        high_row_bytes=high_row_bytes,
        scale_row_bytes=scale_row_bytes,
        base_offset=0,
        base_bytes=base_bytes,
        high_offset=high_offset,
        high_bytes=high_bytes,
        scale_offset=scale_offset,
        scale_bytes=scale_bytes,
        payload_bytes=scale_offset + scale_bytes,
    )


def block_scale_geometry(
    format: str | Nvfp4Format, shape: Sequence[int]
) -> BlockScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Nvfp4Format):
        raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
    n, k = _shape(shape, rank=2)
    if n % 128 != 0 or k % 64 != 0:
        raise ValueError(
            "block_scale_k16_m128x4_v1 requires N divisible by 128 "
            "and K divisible by 64"
        )
    code_plane_bytes = n * k // 2
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * k // spec.group_size
    weight_divisor_offset = scale_plane_offset + scale_plane_bytes
    return BlockScaleGeometry(
        n=n,
        k=k,
        groups_per_row=k // spec.group_size,
        k_tiles=k // 64,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        weight_divisor_offset=weight_divisor_offset,
        payload_bytes=weight_divisor_offset + 4,
    )


def row_scale_geometry(
    format: str | Fp8RowFormat, shape: Sequence[int]
) -> RowScaleGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8RowFormat):
        raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
    n, k = _shape(shape, rank=2)
    code_plane_bytes = n * k
    scale_plane_offset = align_up(code_plane_bytes, PLANE_ALIGNMENT)
    scale_plane_bytes = n * 2
    return RowScaleGeometry(
        n=n,
        k=k,
        code_plane_bytes=code_plane_bytes,
        scale_plane_offset=scale_plane_offset,
        scale_plane_bytes=scale_plane_bytes,
        payload_bytes=scale_plane_offset + scale_plane_bytes,
    )


def expert_bank_geometry(
    format: str | Nvfp4MulFormat, shape: Sequence[int]
) -> ExpertBankGeometry:
    """Geometry of a routed-expert bank ``[experts, hidden, intermediate]``.

    Each record holds the gate/up matrix (rows interleaved gate_i, up_i) and then the down
    matrix, as 16-row groups of 144-byte units (design §6.2). Records are padded to 4 KiB so
    each one is a single aligned DMA and ``O_DIRECT`` range.
    """

    spec = _format(format)
    if not isinstance(spec, Nvfp4MulFormat):
        raise ValueError("nvfp4_expert_rg16_v1 requires nvfp4_mul")
    experts, hidden, intermediate = _shape(shape, rank=3)
    if hidden % 16 or intermediate % 16:
        raise ValueError("nvfp4_expert_rg16_v1 requires hidden and intermediate divisible by 16")
    gate_up_row_groups = 2 * intermediate // 16
    gate_up_blocks = hidden // 16
    down_row_groups = hidden // 16
    down_blocks = intermediate // 16
    gate_up_bytes = gate_up_row_groups * gate_up_blocks * EXPERT_UNIT_BYTES
    record_bytes = gate_up_bytes + down_row_groups * down_blocks * EXPERT_UNIT_BYTES
    record_stride = align_up(record_bytes, EXPERT_RECORD_ALIGNMENT)
    multiplier_offset = experts * record_stride
    multiplier_bytes = experts * 3 * 4
    return ExpertBankGeometry(
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_row_groups=gate_up_row_groups,
        gate_up_blocks=gate_up_blocks,
        down_row_groups=down_row_groups,
        down_blocks=down_blocks,
        gate_up_bytes=gate_up_bytes,
        record_bytes=record_bytes,
        record_stride=record_stride,
        multiplier_offset=multiplier_offset,
        multiplier_bytes=multiplier_bytes,
        payload_bytes=multiplier_offset + multiplier_bytes,
    )


def ggml_rows_geometry(format: str | GgmlFormat, shape: Sequence[int]) -> GgmlRowsGeometry:
    spec = _format(format)
    if not isinstance(spec, GgmlFormat):
        raise ValueError("ggml_rows_v1 requires a ggml block format")
    n, k = _shape(shape, rank=2)
    if k % spec.block_elems:
        raise ValueError(f"ggml_rows_v1: K={k} is not a multiple of {spec.name}'s {spec.block_elems}")
    blocks = k // spec.block_elems
    row_bytes = blocks * spec.block_bytes
    return GgmlRowsGeometry(n, k, blocks, row_bytes, n * row_bytes)


def ggml_tiles16_geometry(format: str | GgmlFormat, shape: Sequence[int]) -> GgmlRowsGeometry:
    """``ggml_tiles16_v1``: ``ggml_rows_v1``'s rows in tiles of 16 (tile i at 16 i row_bytes; block b
    of the tile's row r at (16 b + r) block_bytes), the last tile padded with zero blocks."""
    spec = _format(format)
    if not isinstance(spec, GgmlFormat) or spec.name not in GGML_TILES16_FORMATS:
        raise ValueError(f"ggml_tiles16_v1 holds {sorted(GGML_TILES16_FORMATS)} only")
    rows = ggml_rows_geometry(spec, shape)
    return GgmlRowsGeometry(rows.n, rows.k, rows.blocks_per_row, rows.row_bytes, (rows.n + 15) // 16 * 16 * rows.row_bytes)


def ggml_expert_geometry(
    layout: str | Layout, format: str | GgmlFormat, shape: Sequence[int]
) -> GgmlExpertGeometry:
    layout_spec = _layout(layout)
    spec = _format(format)
    if layout_spec.name not in GGML_EXPERT_DOWN_FORMATS or spec.name not in layout_spec.formats:
        raise ValueError(f"{layout_spec.name} does not accept {spec.name}")
    down = _format(GGML_EXPERT_DOWN_FORMATS[layout_spec.name])
    experts, hidden, intermediate = _shape(shape, rank=3)
    if hidden % spec.block_elems or intermediate % down.block_elems:
        raise ValueError("ggml expert rows must be whole blocks")
    gate_row = hidden // spec.block_elems * spec.block_bytes
    down_row = intermediate // down.block_elems * down.block_bytes
    gate_bytes = intermediate * gate_row
    record = 2 * gate_bytes + hidden * down_row
    stride = align_up(record, EXPERT_RECORD_ALIGNMENT)
    return GgmlExpertGeometry(
        experts=experts, hidden=hidden, intermediate=intermediate, gate_up_format=spec.name,
        down_format=down.name, gate_row_bytes=gate_row, down_row_bytes=down_row, gate_bytes=gate_bytes,
        down_offset=2 * gate_bytes, record_bytes=record, record_stride=stride, payload_bytes=experts * stride,
    )


def fp8_expert_geometry(format: str | Fp8BlockFormat, shape: Sequence[int]) -> Fp8ExpertGeometry:
    spec = _format(format)
    if not isinstance(spec, Fp8BlockFormat):
        raise ValueError("fp8_expert_b128_v1 requires a block-scaled FP8 format")
    experts, hidden, intermediate = _shape(shape, rank=3)
    if hidden % spec.block or intermediate % spec.block:
        raise ValueError(f"fp8_expert_b128_v1 needs whole {spec.block} x {spec.block} tiles")
    gate_bytes = intermediate * hidden
    scale_offset = 2 * gate_bytes + hidden * intermediate
    tiles = (intermediate // spec.block) * (hidden // spec.block)
    record = scale_offset + 3 * tiles * 2
    stride = align_up(record, EXPERT_RECORD_ALIGNMENT)
    return Fp8ExpertGeometry(
        experts=experts, hidden=hidden, intermediate=intermediate, gate_bytes=gate_bytes, down_offset=2 * gate_bytes,
        scale_offset=scale_offset, tiles=tiles, record_bytes=record, record_stride=stride, payload_bytes=experts * stride,
    )


def encoded_size(
    layout: str | Layout,
    format: str | NumericFormat,
    shape: Sequence[int],
) -> int:
    layout_spec = _layout(layout)
    numeric_spec = _format(format)
    if numeric_spec.name not in layout_spec.formats:
        raise ValueError(
            f"layout {layout_spec.name!r} does not accept format {numeric_spec.name!r}"
        )
    if layout_spec is CONTIGUOUS_LE_V1:
        if not isinstance(numeric_spec, DirectFormat):
            raise ValueError("contiguous_le_v1 requires a direct format")
        dims = _shape(shape)
        if len(dims) > 16:
            raise ValueError("contiguous_le_v1 supports rank 0 through 16")
        return prod(dims) * numeric_spec.word_bytes
    if layout_spec is ROW_SPLIT_K128_V1:
        if not isinstance(numeric_spec, QuantFormat):
            raise ValueError("row_split_k128_v1 requires a grouped quantized format")
        return row_split_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is BLOCK_SCALE_K16_M128X4_V1:
        if not isinstance(numeric_spec, Nvfp4Format):
            raise ValueError("block_scale_k16_m128x4_v1 requires NVFP4")
        return block_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is ROW_SCALE_V1:
        if not isinstance(numeric_spec, Fp8RowFormat):
            raise ValueError("row_scale_v1 requires a row-scaled FP8 format")
        return row_scale_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is NVFP4_EXPERT_RG16_V1:
        return expert_bank_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is GGML_ROWS_V1:
        return ggml_rows_geometry(numeric_spec, shape).payload_bytes
    if layout_spec is GGML_TILES16_V1:
        return ggml_tiles16_geometry(numeric_spec, shape).payload_bytes
    if layout_spec.name in GGML_EXPERT_DOWN_FORMATS:
        return ggml_expert_geometry(layout_spec, numeric_spec, shape).payload_bytes
    if layout_spec is FP8_EXPERT_B128_V1:
        return fp8_expert_geometry(numeric_spec, shape).payload_bytes
    raise ValueError(f"unsupported tensor layout: {layout_spec.name!r}")
