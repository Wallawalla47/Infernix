"""The shadow's exact arithmetic (tools/quality/shadow): roundings, certified reductions and functions, and
operations on Shadow tensors, against rational and 160-bit references."""

from __future__ import annotations

from fractions import Fraction

import mpmath
import pytest
import torch

from tools.quality.shadow import exact
from tools.quality.shadow.tensor import Shadow, shadow

F64 = torch.float64


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_round_to_is_nearest_even(dtype):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4000, generator=g, dtype=F64) * torch.exp2(torch.randint(-140, 130, (4000,), generator=g).to(F64))
    got = exact.round_to(x, dtype)
    want = [exact.round_fraction(Fraction(v), dtype) for v in x.tolist()]
    assert got.tolist() == want


def test_round_to_bf16_avoids_double_rounding():
    # 1 + 2^-8 + 2^-30: binary32 rounds it to the BF16 midpoint 1 + 2^-8, which ties to even (1.0);
    # its correct BF16 rounding is up.
    x = torch.tensor([1 + 2.0 ** -8 + 2.0 ** -30], dtype=F64)
    assert exact.round_to(x, torch.bfloat16).item() == 1 + 2.0 ** -7


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_matmul_is_the_rounded_exact_sum(dtype):
    g = torch.Generator().manual_seed(2)
    a = exact.round_to(torch.randn(7, 300, generator=g, dtype=F64), torch.bfloat16)
    b = exact.round_to(torch.randn(300, 5, generator=g, dtype=F64) * 3, torch.bfloat16)
    got = exact.matmul(a, b, dtype)
    for i in range(7):
        for j in range(5):
            want = exact.round_fraction(exact.exact_dot(a[i].tolist(), b[:, j].tolist()), dtype)
            assert got[i, j].item() == want


def test_matmul_falls_back_where_binary64_cannot_decide():
    # 1 + 2^-8 (a BF16 midpoint) plus terms far below binary64's last bit decide the rounding.
    a = torch.tensor([[1.0, 2.0 ** -8, 2.0 ** -80, -(2.0 ** -81)]], dtype=F64)
    b = torch.ones(4, 1, dtype=F64)
    assert exact.matmul(a, b, torch.bfloat16).item() == 1 + 2.0 ** -7  # exact sum 1 + 2^-8 + 2^-81: up


def test_mean_and_sum_round_once():
    x = torch.tensor([[1.0, 2.0 ** -8, 2.0 ** -60]], dtype=F64)
    assert exact.reduce_sum(x, (1,), False, torch.bfloat16).item() == 1.0 + 2.0 ** -7  # just above the midpoint
    assert exact.reduce_sum(x, (1,), False, torch.bfloat16, divisor=3).item() == exact.round_fraction(
        (Fraction(1) + Fraction(2) ** -8 + Fraction(2) ** -60) / 3, torch.bfloat16)


@pytest.mark.parametrize("name,ref", [("exp", mpmath.exp), ("sigmoid", lambda v: 1 / (1 + mpmath.exp(-v))),
                                      ("silu", lambda v: v / (1 + mpmath.exp(-v))),
                                      ("rsqrt", lambda v: 1 / mpmath.sqrt(v)), ("cos", mpmath.cos)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_functions_are_correctly_rounded(name, ref, dtype):
    g = torch.Generator().manual_seed(3)
    x = exact.round_to(torch.rand(500, generator=g, dtype=F64) * 12 - (0 if name == "rsqrt" else 6), torch.float32)
    if name == "rsqrt":
        x = x + 0.01
    got = exact.transcendental(name, x, dtype)
    want = [exact.round_mpf(ref(mpmath.mpf(v)), dtype) for v in x.tolist()]
    assert got.tolist() == want


def test_softmax_is_correctly_rounded():
    g = torch.Generator().manual_seed(4)
    x = exact.round_to(torch.randn(3, 40, generator=g, dtype=F64) * 4, torch.bfloat16)
    got = exact.softmax(x, -1, torch.float32)
    for row in range(3):
        values = [mpmath.mpf(v) for v in x[row].tolist()]
        total = mpmath.fsum(mpmath.exp(v) for v in values)
        assert got[row].tolist() == [exact.round_mpf(mpmath.exp(v) / total, torch.float32) for v in values]


def test_shadow_ops_follow_pytorch_dtypes_and_round_once():
    g = torch.Generator().manual_seed(5)
    x = shadow(torch.randn(4, 64, generator=g), torch.bfloat16)
    w = shadow(torch.randn(64, generator=g), torch.bfloat16)
    # Qwen3.5's RMSNorm, operation by operation.
    h = x.float()
    assert isinstance(h, Shadow) and h.dtype == torch.float32
    r = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    out = (r * (1.0 + w.float())).type_as(x)
    assert out.dtype == torch.bfloat16
    xs = x.inner
    for row in range(4):
        m = exact.round_fraction(sum(Fraction(v) ** 2 for v in xs[row].tolist()) / 64, torch.float32)
        eps = float(torch.tensor(1e-6, dtype=torch.float32))
        s = exact.round_fraction(Fraction(m) + Fraction(eps), torch.float32)
        rs = exact.round_mpf(1 / mpmath.sqrt(mpmath.mpf(s)), torch.float32)
        for k in range(64):
            y = exact.round_fraction(Fraction(xs[row, k].item()) * Fraction(rs), torch.float32)
            one = exact.round_fraction(1 + Fraction(w.inner[k].item()), torch.float32)
            z = exact.round_fraction(Fraction(y) * Fraction(one), torch.float32)
            assert out.inner[row, k].item() == exact.round_fraction(Fraction(z), torch.bfloat16)


def test_shadow_views_alias_and_inplace_rounds():
    s = shadow(torch.zeros(2, 3), torch.bfloat16)
    s[:, 1] = shadow(torch.tensor([1.0 + 2 ** -9, 3.0]), torch.float32)
    assert s.inner[:, 1].tolist() == [1.0, 3.0]  # 1 + 2^-9 is a BF16 tie: to even
    s.index_add_(0, torch.tensor([1]), shadow(torch.full((1, 3), 2.0 ** -8), torch.bfloat16))
    assert s.inner[1].tolist() == [2.0 ** -8, 3.0 + 0.0, 2.0 ** -8]


def test_topk_breaks_ties_toward_the_lower_index():
    x = shadow(torch.tensor([[0.25, 0.5, 0.5, 0.125]]), torch.float32)
    values, indices = torch.topk(x, 2, dim=-1)
    assert indices.tolist() == [[1, 2]] and values.inner.tolist() == [[0.5, 0.5]]

def test_the_causal_convolution_rounds_its_sum_to_binary32_before_silu():
    # The 27B's layer 28, token 4, channel 5230 (first divergence of the lossless route from the shadow, found by
    # tools.quality.shadow.bisect): the exact sum is -0.0626295320689678, silu of it -0.03033447187 lies 7.8e-10
    # above the BF16 midpoint -0.03033447265625 (correct single rounding: -0.0302734375), but the kernel rounds the
    # sum to binary32 first (-0.06262953579) and silu of that is below the midpoint: -0.0303955078125, which is
    # what the engine's causal_conv kernel returns.
    from tools.quality.shadow.overrides import exact_causal_conv
    weights = [0.0020904541015625, -0.006317138671875, -0.07275390625, -0.007354736328125]
    inputs = [0.016845703125, 0.515625, 0.73046875, 0.8515625]
    out = exact_causal_conv(shadow(torch.tensor([[inputs]]), torch.bfloat16),
                            shadow(torch.tensor([weights]), torch.bfloat16), activation="silu")
    assert float(out.inner[0, 0, 3]) == -0.0303955078125
