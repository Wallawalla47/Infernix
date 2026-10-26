"""Interpret the current compressed-tensors FP8/NVFP4 fields and scale semantics.

The matrix resolver also accepts direct tensors. Resolution remains lazy so a
recipe can replace an unused checkpoint source before its weights are inspected.
"""

from __future__ import annotations

from math import prod
import struct

import torch

from tools.artifact.codecs.fp8_row import validate_fp8_row_words
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .modelopt import MXFP8_FORMAT, fp8_block_values, is_fp8_block_matrix, is_mxfp8_matrix, mxfp8_matrix_source
from .safetensors import SafetensorsSource, tensor_source

_E2M1_F64 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
)


def _divisor_word(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    tensor = store.read_flat(name)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: divisor must be finite and positive")
    return raw


def _reciprocal_f32_word(raw: bytes) -> bytes:
    """Store the exact reciprocal (rounded to one FP32 word) of a source scale.

    The v3 NVFP4 codec reconstructs values as ``code * block_scale /
    weight_divisor``.  ModelOpt AutoQuant checkpoints instead store
    *multiplicative* global scales, so their ``weight_scale_2`` /
    ``input_scale`` scalars must be inverted before they can be consumed as
    divisors.
    """
    word = struct.pack("<f", 1.0 / struct.unpack("<f", raw)[0])
    if not valid_positive_fp32_word(struct.unpack("<I", word)[0]):
        raise ValueError(f"reciprocal of {raw!r} is not finite and positive")
    return word


_ENCODED_FORMATS = ("nvfp4", "nvfp4_mul", "fp8_e4m3fn_row_bf16", "fp8_e4m3fn_row_f32")


def compressed_matrix_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], format: str
) -> LogicalSource:
    """Interpret the current compressed-tensors or ModelOpt NVFP4 / per-row FP8 representation.

    ``nvfp4`` divides by the matrix divisor (a ModelOpt multiplier becomes its rounded reciprocal);
    ``nvfp4_mul`` keeps ModelOpt's FP32 ``weight_scale_2`` multiplier exactly. ``fp8_e4m3fn_row_bf16``
    rounds an FP32 scale to BF16 once; ``fp8_e4m3fn_row_f32`` keeps it exactly (a per-tensor scale is
    written into every row) and exposes a static FP32 ``input_scale`` when the matrix has one."""
    if format not in _ENCODED_FORMATS:
        raise ValueError(f"unsupported encoded source format {format}")
    n, k = shape
    if format in ("nvfp4", "nvfp4_mul") and k % 16:
        raise ValueError(f"{prefix}: NVFP4 source K must be divisible by 16")

    def signature(name: str, expected: tuple[int, ...], dtype: str) -> None:
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    def divisor(suffix: str, reciprocal: bool = False) -> bytes:
        raw = _divisor_word(store, f"{prefix}.{suffix}")
        return _reciprocal_f32_word(raw) if reciprocal else raw

    def nvfp4_names() -> tuple[str, str, str, bool]:
        # GPTQ-style checkpoints keep the packed codes in weight_packed with
        # the divisors in weight_global_scale/input_global_scale.  ModelOpt
        # AutoQuant checkpoints store the packed codes in weight with
        # multiplicative global scales in weight_scale_2/input_scale; the v3
        # codec divides, so those are exposed as reciprocals (see the
        # trailing flag).
        if store.has(f"{prefix}.weight_packed"):
            return "weight_packed", "weight_global_scale", "input_global_scale", False
        if store.describe(f"{prefix}.weight").dtype == "U8":
            return "weight", "weight_scale_2", "input_scale", True
        raise ValueError(f"{prefix}: no NVFP4 encoded source found")

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        if format == "nvfp4_mul":
            # ModelOpt's convention as stored: value = e2m1 * e4m3fn(scale) * weight_scale_2.
            packed, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
            signature(packed, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(end - begin, k // 2)
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            if bool((scales > 0x7E).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(format, codes, scales, None, divisor("weight_scale_2"))
        if format == "nvfp4":
            packed_name, weight_divisor_name, _, reciprocal = nvfp4_names()
            packed, scale = f"{prefix}.{packed_name}", f"{prefix}.weight_scale"
            signature(packed, (n, k // 2), "U8")
            signature(scale, (n, k // 16), "F8_E4M3")
            codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(
                end - begin, k // 2
            )
            scales = (
                store.read_flat(scale, begin * (k // 16), end * (k // 16))
                .view(torch.uint8)
                .reshape(end - begin, k // 16)
            )
            if bool((scales > 0x7E).any()):
                raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
            return EncodedRows(
                format, codes, scales, divisor(weight_divisor_name, reciprocal))
        weight, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
        signature(weight, shape, "F8_E4M3")
        info = store.describe(scale)
        if info.dtype not in ("BF16", "F32"):
            raise ValueError(f"{scale}: expected BF16 or F32 scales")
        if prod(info.shape) not in (1, n):
            raise ValueError(
                f"{scale}: expected one scale per row or one global scale"
            )
        codes = (
            store.read_flat(weight, begin * k, end * k)
            .view(torch.uint8)
            .reshape(end - begin, k)
        )
        if prod(info.shape) == n:
            scales = store.read_flat(scale, begin, end)
        else:
            # Per-tensor FP8 (ModelOpt): the global scale written into every requested row.
            scales = store.read_flat(scale).repeat(end - begin)
        if format == "fp8_e4m3fn_row_bf16" and scales.dtype == torch.float32:
            # The BF16 row format: an FP32 checkpoint scale is rounded once at this boundary.
            scales = scales.to(torch.bfloat16)
        elif format == "fp8_e4m3fn_row_f32" and scales.dtype == torch.bfloat16:
            scales = scales.to(torch.float32)  # exact widening
        validate_fp8_row_words(codes, scales)
        return EncodedRows(format, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        if format == "fp8_e4m3fn_row_bf16":
            values = (
                words.codes.view(torch.float8_e4m3fn).float()
                * words.scales.float()[:, None]
            )
        elif format == "fp8_e4m3fn_row_f32":
            # Binary64 holds every code x FP32 scale product exactly (4 + 24 significant bits).
            values = (
                words.codes.view(torch.float8_e4m3fn).double()
                * words.scales.double()[:, None]
            )
        elif format == "nvfp4_mul":
            codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(last - first, k)
            values = (
                _E2M1_F64[codes.long()]
                * words.scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)
                * float(struct.unpack("<f", words.weight_multiplier)[0])
            )
        else:
            codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
                last - first, k
            )
            values_table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            values = values_table[codes.long()]
            scales = (
                words.scales.view(torch.float8_e4m3fn)
                .float()
                .repeat_interleave(16, dim=1)
            )
            values = values * scales / struct.unpack("<f", words.weight_divisor)[0]
        return values.reshape(-1)[begin - first * k : end - first * k]

    def input_scale() -> bytes:
        name = f"{prefix}.input_scale"
        if not store.has(name):
            raise ValueError(f"{prefix}: no static activation scale (input_scale)")
        return _divisor_word(store, name)

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} ({format})",
        read,
        encoded,
        (lambda: divisor(nvfp4_names()[1], nvfp4_names()[3]))
        if format == "nvfp4"
        else None,
        (lambda: divisor(nvfp4_names()[2], nvfp4_names()[3]))
        if format == "nvfp4"
        else None,
        input_scale if format == "fp8_e4m3fn_row_f32" and store.has(f"{prefix}.input_scale") else None,
    )


def matrix_source(
    store: SafetensorsSource,
    name: str,
    shape: tuple[int, int],
    format: str | None = None,
) -> LogicalSource:
    """Resolve the selected matrix's encoding lazily, after recipe source overrides."""
    prefix = name.removesuffix(".weight")
    resolved: LogicalSource | None = None

    def resolve() -> LogicalSource:
        nonlocal resolved
        if resolved is None:
            actual = format
            if actual is None and store.has(prefix + ".weight_packed"):
                actual = "nvfp4"
            if actual is None and store.describe(name).dtype == "U8":
                actual = "nvfp4"
            if actual is None and is_mxfp8_matrix(store, prefix, shape):
                actual = MXFP8_FORMAT  # ModelOpt MXFP8: UE8M0 scales per 32 K elements
            if actual is None and is_fp8_block_matrix(store, prefix):
                # Block FP8 (ModelOpt FP8_PB_WO): values only, code x the 128 x 128 tile's FP32 multiplier.
                resolved = fp8_block_values(store, prefix, shape)
                return resolved
            if actual is None and store.describe(name).dtype == "F8_E4M3":
                # Per-row FP32 scales (ModelOpt FP8_PER_CHANNEL_PER_TOKEN) are read exactly; a BF16 or
                # per-tensor scale takes the BF16 row format, as before.
                scale = store.describe(prefix + ".weight_scale") if store.has(prefix + ".weight_scale") else None
                per_row_f32 = scale is not None and scale.dtype == "F32" and prod(scale.shape) == shape[0] > 1
                actual = "fp8_e4m3fn_row_f32" if per_row_f32 else "fp8_e4m3fn_row_bf16"
            if actual is None:
                resolved = tensor_source(store, name, shape)
            elif actual == MXFP8_FORMAT:
                resolved = mxfp8_matrix_source(store, prefix, shape)
            else:
                resolved = compressed_matrix_source(store, prefix, shape, actual)
        return resolved

    def encoded(begin: int, end: int) -> EncodedRows:
        reader = resolve().read_encoded
        if reader is None:
            raise ValueError(f"{name}: selected source does not provide encoded rows")
        return reader(begin, end)

    def divisor(which: str) -> bytes:
        read = getattr(resolve(), which)
        if read is None:
            raise ValueError(f"{name}: selected source does not provide {which}")
        return read()

    return LogicalSource(
        shape,
        f"{store.path}:{name}",
        lambda begin, end: resolve().values(begin, end),
        encoded,
        lambda: divisor("weight_divisor"),
        lambda: divisor("input_divisor"),
        # A static activation scale only where the matrix stores one (a per-channel export has none).
        (lambda: divisor("input_scale"))
        if format == "fp8_e4m3fn_row_f32" and store.has(prefix + ".input_scale")
        else None,
    )
