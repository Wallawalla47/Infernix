"""Method requests/results and the built-in adapters to quantization or encoded import.

Methods own input traversal, chunking, and auxiliary values. Numerical algorithms
return codes/scales; artifact.tensor_output owns their physical byte placement.
User methods accept the same PrepareRequest and return a PreparedMethod.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import prod
import struct
from typing import Callable, Mapping

import torch

from tools.artifact.formats import (
    DirectFormat,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput

from .quantization.fp8_row import quantize_bf16_rows
from .quantization.groupwise import quantize_matrix, quantize_matrix_mse
from .quantization.nvfp4 import E2M1_MAX, quantize_nvfp4, quantize_nvfp4_mse, weight_divisor
from .sources.logical import EncodedRows, LogicalSource

UseKey = tuple[str, str]
AuxiliaryKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class MethodInput:
    parameter: str
    source: LogicalSource
    uses: tuple[UseKey, ...]


@dataclass(frozen=True, slots=True)
class AuxiliaryValue:
    format: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def activation_divisor(cls, value: bytes | float) -> AuxiliaryValue:
        raw = value if isinstance(value, bytes) else struct.pack("<f", value)
        if len(raw) != 4 or not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
            raise ValueError("activation input divisor must be positive finite FP32")
        return cls("fp32", (), raw)

    @classmethod
    def activation_scale(cls, value: bytes | float) -> AuxiliaryValue:
        """A static FP8 input's FP32 activation scale s (codes ``e4m3(x * fl32(1 / s))``), as stored."""
        raw = value if isinstance(value, bytes) else struct.pack("<f", value)
        if len(raw) != 4 or not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
            raise ValueError("activation input scale must be positive finite FP32")
        return cls("fp32", (), raw)


# Use auxiliary roles: an NVFP4 A4 input's divisor and a static FP8 A8 input's scale.
AUXILIARY_ROLES = {
    "activation_input_divisor": AuxiliaryValue.activation_divisor,
    "activation_input_scale": AuxiliaryValue.activation_scale,
}


@dataclass(frozen=True, slots=True)
class PreparedMethod:
    produce: Callable[[TensorOutput], None]
    auxiliaries: Mapping[AuxiliaryKey, AuxiliaryValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    target: TensorSpec
    inputs: tuple[MethodInput, ...]
    policies: Mapping[UseKey, str]
    parameters: Mapping[str, object]
    device: str = "cuda"
    rows_per_chunk: int = 512
    auxiliary_overrides: Mapping[AuxiliaryKey, AuxiliaryValue] = field(
        default_factory=dict
    )
    source_offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = [0]
        for item in self.inputs:
            values.append(values[-1] + prod(item.source.shape))
        object.__setattr__(self, "source_offsets", tuple(values))

    @property
    def source(self) -> LogicalSource:
        if len(self.inputs) != 1:
            raise ValueError("this method has multiple inputs; use request.inputs")
        return self.inputs[0].source

    def require_input_shape(self, shape: tuple[int, ...]) -> None:
        if self.source.shape != tuple(shape):
            raise ValueError(f"source shape {self.source.shape} differs from {shape}")

    def job(self, *, produce, auxiliaries=None) -> PreparedMethod:
        return PreparedMethod(produce, {} if auxiliaries is None else dict(auxiliaries))

    def values(self, begin: int, end: int) -> torch.Tensor:
        """Read the ordered logical inputs as one C-order parent element sequence."""
        pieces = []
        if not 0 <= begin <= end <= self.source_offsets[-1]:
            raise ValueError(f"{self.target.id}: requested values exceed method inputs")
        index = bisect_right(self.source_offsets, begin) - 1
        while begin < end:
            high = min(end, self.source_offsets[index + 1])
            pieces.append(
                self.inputs[index].source.values(
                    begin - self.source_offsets[index],
                    high - self.source_offsets[index],
                )
            )
            begin = high
            index += 1
        if not pieces:
            return torch.empty(0)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def encoded_rows(self, begin: int, end: int) -> EncodedRows:
        pieces = []
        cursor = 0
        for item in self.inputs:
            source = item.source
            if len(source.shape) != 2 or source.shape[1] != self.target.shape[1]:
                raise ValueError(
                    f"{item.parameter}: encoded grouping requires complete rows"
                )
            low, high = max(begin, cursor), min(end, cursor + source.shape[0])
            if low < high:
                if source.read_encoded is None:
                    raise ValueError(f"{item.parameter}: encoded rows are unavailable")
                pieces.append(source.read_encoded(low - cursor, high - cursor))
            cursor += source.shape[0]
        if not pieces or end > cursor:
            raise ValueError("encoded method input range is invalid")
        first = pieces[0]
        if any(
            (p.format, p.weight_divisor, p.weight_multiplier)
            != (first.format, first.weight_divisor, first.weight_multiplier)
            for p in pieces
        ):
            raise ValueError("encoded inputs cannot share one parent format/divisor/multiplier")
        if len(pieces) == 1:
            return first
        return EncodedRows(
            first.format,
            torch.cat([p.codes for p in pieces]),
            torch.cat([p.scales for p in pieces]),
            first.weight_divisor,
            first.weight_multiplier,
        )


Method = Callable[[PrepareRequest], PreparedMethod]
_DIRECT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}


def _preflight(request: PrepareRequest, *, values: bool = True) -> None:
    if request.parameters:
        raise ValueError(
            f"{request.target.id}: this method accepts no numerical parameters"
        )
    if prod(request.target.shape) != sum(
        prod(item.source.shape) for item in request.inputs
    ):
        raise ValueError(
            f"{request.target.id}: parent and logical source counts differ"
        )
    if type(request.rows_per_chunk) is not int or request.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    if values:
        for item in request.inputs:
            item.source.values(0, 1)


def bf16_from_float64(values: torch.Tensor) -> torch.Tensor:
    """Round binary64 values to BF16 once, to nearest even. PyTorch's conversion passes through
    binary32, a double rounding that differs at BF16 ties (1 + 2^-8 + 2^-30 becomes 1, not
    1 + 2^-7). Results below BF16's normal range are refused: no exact source here produces them."""

    bits = values.contiguous().view(torch.int64)
    sign = (bits >> 63) & 1
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & ((1 << 52) - 1)
    zero = (exponent == 0) & (mantissa == 0)
    if bool((exponent == 0x7FF).any()):
        raise ValueError("BF16 conversion source holds a non-finite value")
    biased = exponent - 1023 + 127
    if bool(((biased < 1) & ~zero).any()):
        raise ValueError("BF16 conversion source holds a value below BF16's normal range")
    kept = (biased << 7) | (mantissa >> 45)
    round_bit = (mantissa >> 44) & 1
    sticky = (mantissa & ((1 << 44) - 1)) != 0
    kept = kept + (round_bit & (sticky | (kept & 1)).to(torch.int64))
    if bool((kept >= 0xFF << 7).any()):
        raise ValueError("BF16 conversion source holds a value beyond BF16's range")
    words = torch.where(zero, 0, kept) | (sign << 15)
    words = torch.where(words >= 0x8000, words - 0x10000, words)  # the signed 16-bit view of each word
    return words.to(torch.int16).view(torch.bfloat16).reshape(values.shape)


def cast_direct(request: PrepareRequest) -> PreparedMethod:
    """Convert values at the explicit target BF16/FP32/INT32 boundary (one rounding from binary64)."""
    if not isinstance(get_format(request.target.format), DirectFormat):
        raise ValueError("cast_direct requires a direct target format")
    _preflight(request)
    dtype = _DIRECT_DTYPES[request.target.format]
    chunk = request.rows_per_chunk * (
        request.target.shape[-1] if len(request.target.shape) > 1 else 1
    )
    elements = prod(request.target.shape)

    def produce(output):
        for begin in range(0, elements, chunk):
            values = request.values(begin, min(elements, begin + chunk))
            if dtype == torch.int32 and (
                bool((values < -(1 << 31)).any())
                or bool((values > (1 << 31) - 1).any())
            ):
                raise ValueError(
                    "int32 conversion source is outside the representable range"
                )
            if values.dtype == torch.float64 and dtype == torch.bfloat16:
                values = bf16_from_float64(values)
            else:
                values = values.to(dtype=dtype)
            output.write_values(begin, values)

    return request.job(produce=produce)


def grouped_absmax(request: PrepareRequest) -> PreparedMethod:
    """Use the existing grouped max-abs, FP16-scale and code-rounding algorithm."""
    if (
        not isinstance(get_format(request.target.format), QuantFormat)
        or len(request.target.shape) != 2
    ):
        raise ValueError("grouped_absmax requires a grouped-integer matrix target")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError(
                    "grouped_absmax source must provide floating-point values"
                )
            encoded = quantize_matrix(
                values, request.target.format, device=request.device
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def grouped_mse(request: PrepareRequest) -> PreparedMethod:
    """Grouped quantisation with the per-group scale that minimises the
    encoded squared error (ties keep the max-abs scale)."""
    if (
        not isinstance(get_format(request.target.format), QuantFormat)
        or len(request.target.shape) != 2
    ):
        raise ValueError("grouped_mse requires a grouped-integer matrix target")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError(
                    "grouped_mse source must provide floating-point values"
                )
            encoded = quantize_matrix_mse(
                values, request.target.format, device=request.device
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def fp8_row_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Round inputs to BF16, then quantize to FP8 codes with BF16 row scales."""
    if request.target.format != "fp8_e4m3fn_row_bf16" or len(request.target.shape) != 2:
        raise ValueError("fp8_row_maxabs requires the row-scaled FP8 matrix format")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = (
                request.values(begin * k, end * k)
                .reshape(end - begin, k)
                .to(torch.bfloat16)
            )
            encoded = quantize_bf16_rows(values)
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


_IMPORT_FORMATS = (
    "nvfp4", "nvfp4_mul", "fp8_e4m3fn_row_bf16", "fp8_e4m3fn_row_f32", "fp8_e4m3fn_g32_e8m0", "q8_g32_fp16",
)


def import_encoded(request: PrepareRequest) -> PreparedMethod:
    """Preserve the current FP8/NVFP4 source codes, scales and weight divisor (or a GGUF Q8_0
    tensor's codes and scales as q8_g32_fp16, the same words; or ModelOpt MXFP8 codes and UE8M0
    group scales as fp8_e4m3fn_g32_e8m0).

    ``fp8_e4m3fn_row_f32`` keeps ModelOpt's FP32 weight scales exactly; an ``AllowA8`` input records
    the source's static FP32 ``input_scale`` as its ``activation_input_scale``, and inputs fused into
    one parent must share it (they share one activation encoding). ``nvfp4_mul`` in
    ``nvfp4_rg16_v1`` keeps ModelOpt's FP32 multiplier exactly."""
    if request.target.format not in _IMPORT_FORMATS or len(request.target.shape) != 2:
        raise ValueError("import_encoded requires a known encoded matrix target")
    if request.target.format == "nvfp4_mul" and request.target.layout != "nvfp4_rg16_v1":
        raise ValueError("a dense nvfp4_mul matrix is stored in nvfp4_rg16_v1")
    _preflight(request, values=False)
    auxiliaries = {}
    weight_divisor = None
    weight_multiplier = None
    input_scale = None
    for item in request.inputs:
        source = item.source
        if source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        first = source.read_encoded(0, 1)
        if first.format != request.target.format:
            raise ValueError(
                f"{item.parameter}: source {first.format} differs from target {request.target.format}"
            )
        if first.format == "nvfp4":
            if weight_divisor is None:
                weight_divisor = first.weight_divisor
            elif first.weight_divisor != weight_divisor:
                raise ValueError(
                    "NVFP4 weight divisors differ; choose separate parents or a conversion method"
                )
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_divisor")
                if key in request.auxiliary_overrides:
                    auxiliaries[key] = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] == "AllowA4":
                    if source.input_divisor is None:
                        raise ValueError(
                            f"{parameter}: supply an activation divisor for AllowA4"
                        )
                    auxiliaries[key] = AuxiliaryValue.activation_divisor(
                        source.input_divisor()
                    )
        elif first.format == "nvfp4_mul":
            if weight_multiplier is None:
                weight_multiplier = first.weight_multiplier
            elif first.weight_multiplier != weight_multiplier:
                raise ValueError("nvfp4_mul multipliers differ; choose separate parents")
        elif first.format == "fp8_e4m3fn_row_f32":
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_scale")
                if key in request.auxiliary_overrides:
                    value = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] in ("AllowA8", "AllowA4"):
                    if source.input_scale is None:
                        raise ValueError(f"{parameter}: AllowA8 needs the source's static input_scale")
                    value = AuxiliaryValue.activation_scale(source.input_scale())
                else:
                    continue
                if input_scale is None:
                    input_scale = value.data
                elif value.data != input_scale:
                    raise ValueError(
                        f"{parameter}: fused FP8 inputs need one static activation scale; separate the parents"
                    )
                auxiliaries[key] = value
    n = request.target.shape[0]
    if request.target.format == "nvfp4":
        chunk = max(128, request.rows_per_chunk // 128 * 128)
    elif request.target.format in ("fp8_e4m3fn_g32_e8m0", "nvfp4_mul"):
        chunk = max(16, request.rows_per_chunk // 16 * 16)  # whole 16-row scale tiles / row groups
    else:
        chunk = request.rows_per_chunk

    def produce(output):
        for begin in range(0, n, chunk):
            words = request.encoded_rows(begin, min(n, begin + chunk))
            if words.format == "nvfp4_mul":
                output.write_codes(begin, words.codes, words.scales, weight_multiplier=words.weight_multiplier)
            else:
                output.write_codes(begin, words.codes, words.scales, words.weight_divisor)

    return request.job(produce=produce, auxiliaries=auxiliaries)


def ggml_tiles16_blocks(codes: torch.Tensor, block_bytes: int) -> torch.Tensor:
    """``ggml_rows_v1`` rows (uint8 [rows, row_bytes], rows a multiple of 16 or the matrix's last
    ones) in ``ggml_tiles16_v1`` order: tile i, then block b, then row r within the tile, the last
    tile padded with zero blocks. Exact: a permutation of whole blocks."""
    rows, row_bytes = codes.shape
    tiles = (rows + 15) // 16
    padded = torch.zeros((tiles * 16, row_bytes), dtype=torch.uint8)
    padded[:rows] = codes
    blocks = row_bytes // block_bytes
    return padded.view(tiles, 16, blocks, block_bytes).permute(0, 2, 1, 3).contiguous().view(-1)


def ggml_tiles16_rows(payload: torch.Tensor, rows: int, row_bytes: int, block_bytes: int) -> torch.Tensor:
    """The ``ggml_rows_v1`` rows of a ``ggml_tiles16_v1`` payload (uint8): ggml_tiles16_blocks' inverse."""
    tiles = (rows + 15) // 16
    blocks = row_bytes // block_bytes
    return payload.view(tiles, blocks, 16, block_bytes).permute(0, 2, 1, 3).contiguous().view(tiles * 16, row_bytes)[:rows]


def import_ggml_rows(request: PrepareRequest) -> PreparedMethod:
    """Copy GGUF block rows verbatim into a ``ggml_rows_v1`` parent, or rearranged into a
    ``ggml_tiles16_v1`` one (bit-exact import: the same blocks).

    Each input's ``read_encoded`` returns its rows' stored bytes (EncodedRows.codes, uint8
    [rows, row_bytes]) in the target's ggml format; grouped inputs concatenate by rows.
    """
    from tools.artifact.codecs.ggml import GGML_TYPES, require_finite_scales
    from tools.artifact.formats import GgmlFormat
    from tools.artifact.layouts import ggml_rows_geometry, ggml_tiles16_geometry

    fmt = get_format(request.target.format)
    layout = request.target.layout
    if not isinstance(fmt, GgmlFormat) or layout not in ("ggml_rows_v1", "ggml_tiles16_v1"):
        raise ValueError("import_ggml_rows requires a ggml format in ggml_rows_v1 or ggml_tiles16_v1")
    _preflight(request, values=False)
    tiles = layout == "ggml_tiles16_v1"
    geometry = ggml_tiles16_geometry(fmt, request.target.shape) if tiles else ggml_rows_geometry(fmt, request.target.shape)
    for item in request.inputs:
        if item.source.read_encoded is None:
            raise ValueError(f"{item.parameter}: stored GGUF rows are unavailable")
        first = item.source.read_encoded(0, 1)
        if first.format != fmt.name:
            raise ValueError(f"{item.parameter}: source {first.format} differs from target {fmt.name}")
    n = geometry.n
    chunk = max(1, request.rows_per_chunk)
    if tiles:  # whole tiles per chunk: tile i lands at 16 i row_bytes, as row 16 i of the rows layout
        chunk = (chunk + 15) // 16 * 16

    def produce(output):
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            words = request.encoded_rows(begin, end)
            codes = words.codes
            if codes.dtype != torch.uint8 or tuple(codes.shape) != (end - begin, geometry.row_bytes):
                raise ValueError(f"{request.target.id}: GGUF rows have the wrong byte shape")
            require_finite_scales(GGML_TYPES[fmt.name], codes.numpy(), end - begin, geometry.k, request.target.id)
            data = ggml_tiles16_blocks(codes.contiguous(), fmt.block_bytes) if tiles else codes.contiguous().view(-1)
            output.write_bytes(begin * geometry.row_bytes, data.numpy().tobytes())

    return request.job(produce=produce)


# The MSE divisor maps the parent's largest magnitude to 4 at the largest scale, so its largest
# group may still take scales up to 1.5x its max-magnitude scale.
_NVFP4_MSE_DIVISOR_TARGET = 4.0


def _nvfp4_quantized(request: PrepareRequest, quantize, divisor_target: float) -> PreparedMethod:
    if request.target.format != "nvfp4" or len(request.target.shape) != 2:
        raise ValueError("NVFP4 quantization requires an nvfp4 matrix target")
    _preflight(request)
    n, k = request.target.shape
    auxiliaries = {}
    for item in request.inputs:
        for parameter, input_name in item.uses:
            key = (parameter, input_name, "activation_input_divisor")
            if key in request.auxiliary_overrides:
                auxiliaries[key] = request.auxiliary_overrides[key]
            elif request.policies[(parameter, input_name)] == "AllowA4":
                raise ValueError(
                    f"{parameter}: AllowA4 needs an activation divisor; supply one with "
                    "recipe.use(..., auxiliaries=...) or use A16Only"
                )
    rows = max(128, request.rows_per_chunk // 128 * 128)

    def produce(output):
        amax = 0.0
        for begin in range(0, n, rows):
            end = min(n, begin + rows)
            values = request.values(begin * k, end * k)
            if not values.dtype.is_floating_point:
                raise TypeError("NVFP4 quantization source must be floating point")
            amax = max(amax, float(values.abs().max()))
        divisor = weight_divisor(amax, divisor_target)
        for begin in range(0, n, rows):
            end = min(n, begin + rows)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            words = quantize(values.to(request.device), divisor)
            output.write_codes(begin, words.codes, words.scales, words.weight_divisor)

    return request.job(produce=produce, auxiliaries=auxiliaries)


def nvfp4_absmax(request: PrepareRequest) -> PreparedMethod:
    """NVFP4 with every 16-value group's largest magnitude at the largest E2M1 code."""
    return _nvfp4_quantized(request, quantize_nvfp4, E2M1_MAX)


def nvfp4_mse(request: PrepareRequest) -> PreparedMethod:
    """NVFP4 with each group's least-squared-error scale among every E4M3FN value."""
    return _nvfp4_quantized(request, quantize_nvfp4_mse, _NVFP4_MSE_DIVISOR_TARGET)


METHODS: dict[str, Method] = {
    "cast_direct": cast_direct,
    "grouped_absmax": grouped_absmax,
    "grouped_mse": grouped_mse,
    "fp8_row_maxabs": fp8_row_maxabs,
    "import_encoded": import_encoded,
    "import_ggml_rows": import_ggml_rows,
    "nvfp4_absmax": nvfp4_absmax,
    "nvfp4_mse": nvfp4_mse,
}
