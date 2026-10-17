from __future__ import annotations

import numpy as np
import torch

from tools.artifact.codecs.row_split import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
)
from tools.convert.quantization.groupwise import quantize_matrix, quantize_matrix_mse


def test_quantization_uses_stored_fp16_scale_and_zero_padding() -> None:
    weight = torch.tensor(
        [[-7.0, -1.0, 0.0, 1.0, 7.0, 3.5, -3.5, 0.25] + [0.0] * 57],
        dtype=torch.bfloat16,
    )
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert quantized.codes.shape == (1, 2, 64)
    assert quantized.scales.shape == (1, 2)
    expected_codes = torch.zeros((1, 2, 64), dtype=torch.int8)
    expected_codes[0, 0, :8] = torch.tensor([-7, -1, 0, 1, 7, 4, -4, 0])
    assert torch.equal(quantized.codes, expected_codes)
    assert quantized.scales.tolist() == [[1.0, 0.0]]
    assert torch.count_nonzero(quantized.codes[0, 1]) == 0
    assert quantized.scales[0, 1] == 0

    payload = encode_row_split(
        quantized.codes, quantized.scales, "q4_g64_fp16", weight.shape
    )
    scales, codes = decode_row_split_codes(payload, "q4_g64_fp16", tuple(weight.shape))
    assert torch.equal(scales, quantized.scales)
    assert torch.equal(codes, quantized.codes)
    decoded = dequantize_row_split(
        payload, "q4_g64_fp16", tuple(weight.shape), dtype=torch.float32
    )
    expected = expected_codes.float().reshape(1, 128)[:, :65]
    assert torch.equal(decoded, expected)


def test_quantization_uses_canonical_scale_rounding_on_cuda() -> None:
    # BF16 word 0x3636 divided by Q4 qmax is a known CUDA division boundary:
    # approximate device division rounds the FP16 scale to word 7, while the
    # ordered RNE oracle requires word 6.
    value = torch.tensor([0x3636], dtype=torch.int16).view(torch.bfloat16)
    weight = value.repeat(64).reshape(1, 64)
    cpu = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(cpu.scales.view(torch.int16)[0, 0]) == 6
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "q4_g64_fp16", device="cuda")
        assert torch.equal(
            cuda.scales.cpu().view(torch.int16), cpu.scales.view(torch.int16)
        )
        assert torch.equal(cuda.codes.cpu(), cpu.codes)


def test_quantization_uses_reciprocal_multiply_and_ties_to_even() -> None:
    words = torch.tensor([0x41A0B334, 0x417C7BFF], dtype=torch.int32).view(
        torch.float32
    )
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, :2] = words
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 0x41BD
    assert quantized.codes[0, 0, :2].tolist() == [7, 6]
    if torch.cuda.is_available():
        cuda = quantize_matrix(weight, "q4_g64_fp16", device="cuda")
        assert torch.equal(cuda.scales.cpu(), quantized.scales)
        assert torch.equal(cuda.codes.cpu(), quantized.codes)

    tie_weight = torch.zeros((1, 64), dtype=torch.float32)
    tie_weight[0, :7] = torch.tensor([7.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
    ties = quantize_matrix(tie_weight, "q4_g64_fp16", device="cpu")
    assert ties.codes[0, 0, :7].tolist() == [7, 0, 2, 2, 0, -2, -2]


def test_q2_mse_puts_the_larger_side_on_the_wider_code_half() -> None:
    # Codes {-2, -1, 0, 1}: a group whose largest magnitude is positive takes a negative scale,
    # one whose largest is negative a positive one, and either way the max maps to code -2.
    rng = np.random.default_rng(7)
    base = rng.standard_normal(64).astype(np.float32) * 0.1
    rows = np.stack([base, -base])
    rows[0, 3] = 1.0
    rows[1, 3] = -1.0
    weight = torch.from_numpy(rows)
    quantized = quantize_matrix_mse(weight, "q2_g64_fp16", device="cpu")
    scales = quantized.scales.float()
    assert scales[0, 0] < 0 < scales[1, 0]
    assert quantized.codes[0, 0, 3] == -2 and quantized.codes[1, 0, 3] == -2
    # The encoded error is no worse than the max-abs scale's, and the codec round-trips it.
    absmax = quantize_matrix(weight, "q2_g64_fp16", device="cpu")

    def values(q) -> torch.Tensor:
        # K = 64 pads to 128: two groups per row, the second all zero.
        return (q.codes.float() * q.scales.float().unsqueeze(-1)).reshape(2, 128)[:, :64]

    def sse(q) -> float:
        return float(((values(q) - weight) ** 2).sum())

    assert sse(quantized) < sse(absmax)
    payload = encode_row_split(quantized.codes, quantized.scales, "q2_g64_fp16", weight.shape)
    decoded = dequantize_row_split(payload, "q2_g64_fp16", tuple(weight.shape), dtype=torch.float32)
    assert torch.equal(decoded, values(quantized))


def test_nonzero_scale_underflow_uses_smallest_fp16_subnormal() -> None:
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, 0] = torch.finfo(torch.float32).tiny
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 1
