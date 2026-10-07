from __future__ import annotations

from fractions import Fraction

import pytest
import torch
from safetensors.torch import save_file

from tools.convert.qwen4_exp import fp8_block_values
from tools.convert.sources.safetensors import SafetensorsSource


def _e4m3fn(word: int) -> Fraction:
    """E4M3FN by its bit fields: bias 7, subnormals at exponent 0, no infinities."""
    sign = -1 if word & 0x80 else 1
    exponent, mantissa = (word >> 3) & 0xF, word & 0x7
    if exponent == 0:
        return sign * Fraction(mantissa, 8) * Fraction(1, 2**6)
    return sign * (1 + Fraction(mantissa, 8)) * Fraction(2) ** (exponent - 7)


@pytest.mark.parametrize("scale_dtype", [torch.bfloat16, torch.float32])  # NVIDIA's MTP / Qwen's FP8
@pytest.mark.parametrize("shape", [(200, 130), (128, 256), (300, 70)])
def test_mtp_expert_values_are_code_times_tile_multiplier(tmp_path, shape, scale_dtype):
    n, k = shape
    generator = torch.Generator().manual_seed(n * 1000 + k)
    codes = torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
    codes[(codes & 0x7F) == 0x7F] = 0x01  # NaN words are invalid in a checkpoint
    tiles = (-(-n // 128), -(-k // 128))
    scales = (torch.rand(tiles, generator=generator, dtype=torch.float32) * 0.01 + 1e-4).to(scale_dtype)
    leaf = "mtp.layers.0.mlp.experts.3.gate_proj"
    save_file({leaf + ".weight": codes.view(torch.float8_e4m3fn), leaf + ".weight_scale_inv": scales},
              tmp_path / "model.safetensors")
    source = fp8_block_values(SafetensorsSource(tmp_path), leaf, shape)

    words, multipliers = codes.tolist(), scales.tolist()
    expected = [
        float(_e4m3fn(words[r][c]) * Fraction(multipliers[r // 128][c // 128]))
        for r in range(n)
        for c in range(k)
    ]
    assert source.values().tolist() == expected
    # Row-chunked and unaligned reads see the same values.
    for begin, end in ((0, 1), (k - 3, 3 * k + 5), (min(129 * k + 7, n * k - 1), n * k), (5 * k, 5 * k)):
        assert source.values(begin, end).tolist() == expected[begin:end]
