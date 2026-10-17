"""Closed registry of persistent Infernix tensor numeric formats."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from types import MappingProxyType
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class DirectFormat:
    """One fixed-width word per logical tensor element."""

    name: str
    word_bytes: int


@dataclass(frozen=True, slots=True)
class QuantFormat:
    """Signed grouped codes with one binary16 multiplier per group."""

    name: str
    bits: int
    group_size: int
    qmin: int
    qmax: int


@dataclass(frozen=True, slots=True)
class Nvfp4Format:
    """E2M1 weights with one E4M3FN scale word per K-axis group."""

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class Fp8RowFormat:
    """E4M3FN weights with one BF16 multiplier per logical row."""

    name: str


@dataclass(frozen=True, slots=True)
class Nvfp4MulFormat:
    """E2M1 weights, one E4M3FN scale per K-axis group, one FP32 multiplier per matrix.

    This is ModelOpt's NVFP4 convention (``weight_scale_2`` multiplies). It differs from
    ``nvfp4``, whose matrix coefficient is a divisor, so neither is a variant of the other.
    """

    name: str
    group_size: int


@dataclass(frozen=True, slots=True)
class GgmlFormat:
    """A GGUF (ggml) block format stored verbatim: ``block_bytes`` per ``block_elems`` K-axis
    elements (tools/artifact/codecs/ggml.py holds each one's exact decoder)."""

    name: str
    block_elems: int
    block_bytes: int


@dataclass(frozen=True, slots=True)
class Fp8BlockFormat:
    """E4M3FN weights with one BF16 scale per ``block`` x ``block`` tile (the official
    Qwen3.8-Flash-Next-FP8 experts; tools/artifact/codecs/fp8_b128.py holds the exact decoder)."""

    name: str
    block: int


NumericFormat: TypeAlias = (
    DirectFormat | QuantFormat | Nvfp4Format | Fp8RowFormat | Nvfp4MulFormat | GgmlFormat | Fp8BlockFormat
)


BF16 = DirectFormat("bf16", 2)
FP32 = DirectFormat("fp32", 4)
INT32 = DirectFormat("int32", 4)

# Two-bit codes {-2, -1, 0, 1}: draft-only weights (the Qwen4Exp MTP drafter's experts), whose
# precision moves only the drafts' acceptance.
Q2_G64_FP16 = QuantFormat("q2_g64_fp16", 2, 64, -2, 1)
Q4_G64_FP16 = QuantFormat("q4_g64_fp16", 4, 64, -8, 7)
Q5_G64_FP16 = QuantFormat("q5_g64_fp16", 5, 64, -16, 15)
Q6_G64_FP16 = QuantFormat("q6_g64_fp16", 6, 64, -32, 31)
Q8_G32_FP16 = QuantFormat("q8_g32_fp16", 8, 32, -127, 127)
NVFP4 = Nvfp4Format("nvfp4", 16)
FP8_E4M3FN_ROW_BF16 = Fp8RowFormat("fp8_e4m3fn_row_bf16")
NVFP4_MUL = Nvfp4MulFormat("nvfp4_mul", 16)

GGML_Q2_0 = GgmlFormat("ggml_q2_0", 64, 18)
GGML_Q4_K = GgmlFormat("ggml_q4_k", 256, 144)
GGML_Q5_K = GgmlFormat("ggml_q5_k", 256, 176)
GGML_Q6_K = GgmlFormat("ggml_q6_k", 256, 210)
GGML_IQ2_S = GgmlFormat("ggml_iq2_s", 256, 82)
GGML_IQ3_XXS = GgmlFormat("ggml_iq3_xxs", 256, 98)
GGML_IQ3_S = GgmlFormat("ggml_iq3_s", 256, 110)
GGML_IQ4_XS = GgmlFormat("ggml_iq4_xs", 256, 136)
GGML_IQ4_NL = GgmlFormat("ggml_iq4_nl", 32, 18)
FP8_E4M3FN_B128 = Fp8BlockFormat("fp8_e4m3fn_b128", 128)


DIRECT_FORMATS = MappingProxyType({item.name: item for item in (BF16, FP32, INT32)})
QUANT_FORMATS = MappingProxyType(
    {item.name: item for item in (Q2_G64_FP16, Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, Q8_G32_FP16)}
)
NVFP4_FORMATS = MappingProxyType({NVFP4.name: NVFP4})
FP8_ROW_FORMATS = MappingProxyType({FP8_E4M3FN_ROW_BF16.name: FP8_E4M3FN_ROW_BF16})
NVFP4_MUL_FORMATS = MappingProxyType({NVFP4_MUL.name: NVFP4_MUL})
GGML_FORMATS = MappingProxyType(
    {
        item.name: item
        for item in (
            GGML_Q2_0, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K, GGML_IQ2_S, GGML_IQ3_XXS, GGML_IQ3_S,
            GGML_IQ4_XS, GGML_IQ4_NL,
        )
    }
)
FP8_BLOCK_FORMATS = MappingProxyType({FP8_E4M3FN_B128.name: FP8_E4M3FN_B128})
NUMERIC_FORMATS = MappingProxyType(
    {
        **DIRECT_FORMATS,
        **QUANT_FORMATS,
        **NVFP4_FORMATS,
        **FP8_ROW_FORMATS,
        **NVFP4_MUL_FORMATS,
        **GGML_FORMATS,
        **FP8_BLOCK_FORMATS,
    }
)


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def decode_e2m1_word(word: int) -> float:
    """Decode one exact four-bit E2M1 word, including signed zero."""

    if type(word) is not int or not 0 <= word <= 0xF:
        raise ValueError("E2M1 word must be an integer in [0, 15]")
    magnitude = _E2M1_MAGNITUDES[word & 0x7]
    return math.copysign(magnitude, -1.0 if word & 0x8 else 1.0)


def decode_e4m3fn_word(word: int) -> float:
    """Decode one exact eight-bit E4M3FN word."""

    if type(word) is not int or not 0 <= word <= 0xFF:
        raise ValueError("E4M3FN word must be an integer in [0, 255]")
    sign = -1.0 if word & 0x80 else 1.0
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    if exponent == 0:
        if fraction == 0:
            return math.copysign(0.0, sign)
        return sign * fraction * (2.0**-9)
    if exponent == 0xF and fraction == 0x7:
        return math.copysign(math.nan, sign)
    return sign * (1.0 + fraction / 8.0) * (2.0 ** (exponent - 7))


def valid_nvfp4_scale_word(word: int) -> bool:
    """Return whether *word* is an admitted nonnegative finite E4M3FN scale."""

    return type(word) is int and 0 <= word <= 0xFF and word & 0x80 == 0 and word != 0x7F


def valid_fp8_weight_word(word: int) -> bool:
    """Return whether *word* is a finite E4M3FN weight code."""

    return type(word) is int and 0 <= word <= 0xFF and (word & 0x7F) != 0x7F


def valid_fp8_row_scale_word(word: int) -> bool:
    """Return whether *word* is a nonnegative finite BF16 multiplier."""

    if type(word) is not int or not 0 <= word <= 0xFFFF or word & 0x8000:
        return False
    value = struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return math.isfinite(value)


def valid_positive_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word represents a finite positive value."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF:
        return False
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    return math.isfinite(value) and value > 0.0


def valid_nonnegative_fp32_word(word: int) -> bool:
    """Return whether an IEEE binary32 word is finite with a clear sign bit (zero allowed)."""

    if type(word) is not int or not 0 <= word <= 0xFFFFFFFF or word & 0x80000000:
        return False
    return (word & 0x7F800000) != 0x7F800000


def get_format(name: str) -> NumericFormat:
    """Return the registered format named *name*."""

    try:
        return NUMERIC_FORMATS[name]
    except KeyError:
        raise ValueError(f"unknown numeric format: {name!r}") from None


__all__ = [
    "BF16",
    "FP32",
    "INT32",
    "Q2_G64_FP16",
    "Q4_G64_FP16",
    "Q5_G64_FP16",
    "Q6_G64_FP16",
    "Q8_G32_FP16",
    "NVFP4",
    "FP8_E4M3FN_ROW_BF16",
    "NVFP4_MUL",
    "FP8_E4M3FN_B128",
    "DIRECT_FORMATS",
    "QUANT_FORMATS",
    "NVFP4_FORMATS",
    "FP8_ROW_FORMATS",
    "NVFP4_MUL_FORMATS",
    "GGML_FORMATS",
    "FP8_BLOCK_FORMATS",
    "NUMERIC_FORMATS",
    "GgmlFormat",
    "Fp8BlockFormat",
    "DirectFormat",
    "QuantFormat",
    "Nvfp4Format",
    "Fp8RowFormat",
    "Nvfp4MulFormat",
    "NumericFormat",
    "get_format",
    "decode_e2m1_word",
    "decode_e4m3fn_word",
    "valid_fp8_row_scale_word",
    "valid_fp8_weight_word",
    "valid_nvfp4_scale_word",
    "valid_nonnegative_fp32_word",
    "valid_positive_fp32_word",
]
