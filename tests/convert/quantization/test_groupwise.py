from __future__ import annotations

import numpy as np
import torch

from tools.artifact.codecs.row_split import (
    decode_row_split_codes,
    dequantize_row_split,
    encode_row_split,
)
from tools.convert.quantization.groupwise import (
    quantize_matrix,
    quantize_matrix_mse,
    quantize_matrix_mse_norm,
)


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


def test_q2_screen_keeps_the_mse_codes_and_stretches_each_scale() -> None:
    # The screen copy (grouped_mse_norm): grouped_mse's codes, each group's scale times
    # sqrt(|w|^2 / <w, w_hat>) rounded once to binary16 (an independent FP64 evaluation here), its sign
    # kept; a zero group keeps its zero scale; near the least-squares scale the copy keeps the group's
    # norm; the codec round-trips it.
    rng = np.random.default_rng(11)
    rows = (rng.standard_normal((6, 2560)) * 0.02).astype(np.float32)
    rows[2, 64:128] = 0.0
    weight = torch.from_numpy(rows)
    mse = quantize_matrix_mse(weight, "q2_g64_fp16", device="cpu")
    screen = quantize_matrix_mse_norm(weight, "q2_g64_fp16", device="cpu")
    assert torch.equal(screen.codes, mse.codes)
    assert screen.scales.dtype == torch.float16
    codes = screen.codes.double().numpy()
    grouped = rows.astype(np.float64).reshape(6, 40, 64)
    least_squares = mse.scales.double().numpy()
    overlap = (grouped * codes).sum(axis=2) * least_squares
    energy = (grouped**2).sum(axis=2)
    expected = np.where(overlap > 0, least_squares * np.sqrt(energy / np.where(overlap > 0, overlap, 1.0)), least_squares)
    assert np.array_equal(screen.scales.numpy(), expected.astype(np.float16))
    stretched = screen.scales.double().numpy()
    assert stretched[2, 1] == 0.0 and least_squares[2, 1] == 0.0
    live = energy > 0
    assert np.all(np.sign(stretched[live]) == np.sign(least_squares[live]))
    # Gaussian groups at the MSE scale: the copy's norm near the group's.
    copy_norm = np.sqrt(((codes * stretched[..., None]) ** 2).sum(axis=2))
    assert np.all(np.abs(copy_norm[live] / np.sqrt(energy[live]) - 1.0) <= 0.15)
    payload = encode_row_split(screen.codes, screen.scales, "q2_g64_fp16", weight.shape)
    decoded = dequantize_row_split(payload, "q2_g64_fp16", tuple(weight.shape), dtype=torch.float32)
    expected_values = torch.from_numpy((codes * stretched[..., None]).reshape(6, 2560)).float()
    assert torch.equal(decoded, expected_values)


def test_nonzero_scale_underflow_uses_smallest_fp16_subnormal() -> None:
    weight = torch.zeros((1, 64), dtype=torch.float32)
    weight[0, 0] = torch.finfo(torch.float32).tiny
    quantized = quantize_matrix(weight, "q4_g64_fp16", device="cpu")
    assert int(quantized.scales.view(torch.int16)[0, 0]) == 1
