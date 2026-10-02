from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words, encode_nvfp4
from tools.artifact.formats import decode_e2m1_word, decode_e4m3fn_word
from tools.convert.quantization.nvfp4 import (
    quantize_nvfp4,
    quantize_nvfp4_mse,
    round_e2m1,
    weight_divisor,
)

MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MSE_DIVISOR_TARGET = 4.0


def _decode(words, shape) -> torch.Tensor:
    """Independent scalar decode of packed words: e2m1 * e4m3fn(scale) / divisor."""
    n, k = shape
    divisor = struct.unpack("<f", words.weight_divisor)[0]
    out = torch.empty((n, k), dtype=torch.float64)
    codes = words.codes.tolist()
    scales = words.scales.tolist()
    for row in range(n):
        for col in range(k):
            byte = codes[row][col // 2]
            word = byte & 0xF if col % 2 == 0 else byte >> 4
            scale = decode_e4m3fn_word(scales[row][col // 16])
            out[row, col] = decode_e2m1_word(word) * scale / divisor
    return out


def test_round_e2m1_nearest_with_ties_to_even():
    values = torch.tensor(
        [0.0, 0.2, 0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 6.0, 0.74, 2.6]
    )
    expected = [0, 0, 0, 1, 2, 2, 4, 4, 6, 6, 7, 7, 1, 5]
    assert round_e2m1(values).tolist() == expected


def test_divisor_puts_largest_group_at_top_scale():
    assert weight_divisor(3.0) == pytest.approx(448.0 * 6.0 / 3.0)
    assert weight_divisor(3.0, MSE_DIVISOR_TARGET) == pytest.approx(448.0 * 4.0 / 3.0)
    assert weight_divisor(0.0) == 1.0


def test_exactly_representable_matrix_round_trips():
    generator = torch.Generator().manual_seed(7)
    n, k = 128, 64
    divisor = 37.5
    scale_words = torch.randint(0x30, 0x60, (n, k // 16), generator=generator)
    codes = torch.randint(0, 16, (n, k), generator=generator)
    codes[:, ::16] = 7  # every group holds +6, so its absmax scale is exact
    values = torch.empty(n, k, dtype=torch.float64)
    for row in range(n):
        for col in range(k):
            values[row, col] = (
                decode_e2m1_word(int(codes[row, col]))
                * decode_e4m3fn_word(int(scale_words[row, col // 16]))
                / divisor
            )
    words = quantize_nvfp4(values.float(), divisor)
    assert torch.equal(words.scales, scale_words.to(torch.uint8))
    assert torch.equal(_decode(words, (n, k)), values)


def test_codec_round_trip_and_valid_words():
    generator = torch.Generator().manual_seed(11)
    n, k = 256, 128
    values = torch.randn(n, k, generator=generator) * torch.logspace(-3, 1, k)
    divisor = weight_divisor(float(values.abs().max()), MSE_DIVISOR_TARGET)
    words = quantize_nvfp4_mse(values, divisor)
    assert words.codes.shape == (n, k // 2) and words.codes.dtype == torch.uint8
    assert words.scales.shape == (n, k // 16) and words.scales.dtype == torch.uint8
    assert not bool(((words.scales & 0x80) != 0).any())
    assert not bool((words.scales == 0x7F).any())
    payload = encode_nvfp4(words.codes, words.scales, words.weight_divisor, (n, k))
    codes, scales, stored_divisor = decode_nvfp4_words(payload, (n, k))
    assert torch.equal(codes, words.codes)
    assert torch.equal(scales, words.scales)
    assert struct.pack("<f", float(stored_divisor)) == words.weight_divisor


def test_reconstruction_error_is_bounded_by_half_a_code_step():
    generator = torch.Generator().manual_seed(3)
    n, k = 128, 256
    values = torch.randn(n, k, generator=generator, dtype=torch.float64)
    divisor = weight_divisor(float(values.abs().max()))
    words = quantize_nvfp4(values.float(), divisor)
    decoded = _decode(words, (n, k))
    scales = torch.tensor(
        [[decode_e4m3fn_word(int(w)) for w in row] for row in words.scales.tolist()],
        dtype=torch.float64,
    )
    step = (scales / divisor).repeat_interleave(16, dim=1)
    # The widest E2M1 gap is 2 (between 4 and 6), so rounding moves at most one step unit.
    assert bool(((decoded - values.float().double()).abs() <= step + 1e-12).all())


def _group_error(decoded: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    n, k = values.shape
    return (decoded - values.double()).square().reshape(n, k // 16, 16).sum(-1)


def _oracle_least_group_error(values: torch.Tensor, divisor: float) -> torch.Tensor:
    """Least squared error of every group over each positive E4M3FN scale and signed code."""
    n, k = values.shape
    scales = torch.tensor(
        [decode_e4m3fn_word(word) for word in range(1, 127)], dtype=torch.float64
    )
    codes = torch.tensor([decode_e2m1_word(word) for word in range(16)], dtype=torch.float64)
    representable = scales[:, None] * codes[None, :] / divisor  # [scale, code]
    groups = values.double().reshape(n, k // 16, 16, 1, 1)
    per_scale = (groups - representable).square().amin(dim=-1).sum(dim=-2)
    return per_scale.amin(dim=-1)


@pytest.mark.parametrize("divisor_target", [6.0, MSE_DIVISOR_TARGET])
def test_mse_scale_is_the_least_error_scale(divisor_target):
    generator = torch.Generator().manual_seed(5)
    n, k = 16, 128
    # Groups span four decades, so some take scales well away from their max-magnitude scale.
    decades = torch.logspace(-4, 0, k // 16).repeat_interleave(16)
    values = torch.randn(n, k, generator=generator) * decades
    divisor = weight_divisor(float(values.abs().max()), divisor_target)
    mse_error = _group_error(_decode(quantize_nvfp4_mse(values, divisor), (n, k)), values)
    least = _oracle_least_group_error(values, divisor)
    assert bool((mse_error <= least * (1 + 1e-5) + 1e-30).all())


def test_mse_never_loses_to_absmax_at_the_same_divisor():
    generator = torch.Generator().manual_seed(5)
    n, k = 128, 512
    values = torch.randn(n, k, generator=generator) * 0.02
    divisor = weight_divisor(float(values.abs().max()), MSE_DIVISOR_TARGET)
    absmax_error = _group_error(_decode(quantize_nvfp4(values, divisor), (n, k)), values)
    mse_error = _group_error(_decode(quantize_nvfp4_mse(values, divisor), (n, k)), values)
    assert bool((mse_error <= absmax_error * (1 + 1e-6) + 1e-30).all())
    assert float(mse_error.sum()) < float(absmax_error.sum())


@pytest.mark.parametrize("quantize", [quantize_nvfp4, quantize_nvfp4_mse])
def test_zero_groups_and_signs(quantize):
    values = torch.zeros(128, 32)
    values[0, 16:] = torch.tensor([-6.0, 6.0] * 8)
    words = quantize(values, weight_divisor(6.0))
    assert int(words.scales[1, 0]) == 0 and int(words.scales[0, 0]) == 0
    assert words.codes[1].tolist() == [0] * 16
    first = words.codes[0, 8:].tolist()
    assert all(byte == 0x7F for byte in first)  # -6 (0xF) in the low nibble, +6 (0x7) high
