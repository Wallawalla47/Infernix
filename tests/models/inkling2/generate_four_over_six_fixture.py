"""The four-over-six NVFP4 activation fixture (tests/fixtures/inkling2/four_over_six.json) for test_inkling2_canon.cu:
rows of BF16 values and, per 16-element block, the scale word and codes the rule gives, computed here independently
of canonical_math.h in torch binary32 arithmetic (each operation rounded as vLLM's and TokenSpeed's kernels round it):

  amax = max|x|; enc = min(1536 / amax, FP32_MAX); dec = 1 / enc; outer = amax * fl32(1/1536)
  s6 = e4m3(min(bmax * (enc * fl32(1/6)), 448)); s4 = e4m3(min(bmax * (enc * 0.25), 448))
  codes = e2m1(x * min(1 / (s * dec), FP32_MAX)); error = pairwise sum of |(c * s) * outer - x| over the 16 columns
  the block takes s4 exactly when its error is smaller than s6's.

Rows: Gaussian activations at several magnitudes, heavy tails, rows with zero blocks, an all-zero row, single
outliers, values on E2M1 rounding boundaries, subnormal-scale blocks, and BF16 extremes.

  py -3.11 generate_four_over_six_fixture.py <out.json>
"""

from __future__ import annotations

import json
import sys

import torch

FP32_MAX = torch.finfo(torch.float32).max
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def e4m3_word(v: torch.Tensor) -> torch.Tensor:
    return v.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int64)


def e4m3_value(words: torch.Tensor) -> torch.Tensor:
    return words.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)


def e2m1_code(v: torch.Tensor) -> torch.Tensor:
    """Round to nearest even E2M1 code with saturation (cvt.rn.satfinite.e2m1): the sign in bit 3."""
    a = v.abs()
    code = torch.zeros_like(a, dtype=torch.int64)
    code = torch.where(a > 0.25, 1, code)
    code = torch.where(a >= 0.75, 2, code)
    code = torch.where(a > 1.25, 3, code)
    code = torch.where(a >= 1.75, 4, code)
    code = torch.where(a > 2.5, 5, code)
    code = torch.where(a >= 3.5, 6, code)
    code = torch.where(a > 5.0, 7, code)
    return code | torch.where(torch.signbit(v), 8, 0)


def pairwise16(err: torch.Tensor) -> torch.Tensor:
    while err.shape[-1] > 1:
        err = err[..., 0::2] + err[..., 1::2]
    return err[..., 0]


def quantize(row: torch.Tensor):
    blocks = row.reshape(-1, 16)
    amax = row.abs().max()
    enc = torch.minimum(torch.tensor(1536.0) / amax, torch.tensor(FP32_MAX))
    dec = torch.tensor(1.0) / enc
    outer = amax * torch.tensor(1.0 / 1536.0, dtype=torch.float32)
    bmax = blocks.abs().amax(-1)
    results = []
    for factor in (torch.tensor(1.0 / 6.0, dtype=torch.float32), torch.tensor(0.25)):
        words = e4m3_word(torch.minimum(bmax * (enc * factor), torch.tensor(448.0)))
        s = e4m3_value(words)
        inv = torch.minimum(torch.tensor(1.0) / (s * dec), torch.tensor(FP32_MAX))
        codes = e2m1_code(blocks * inv[:, None])
        c = torch.where((codes & 8) != 0, -E2M1[codes & 7], E2M1[codes & 7])
        err = pairwise16(((c * s[:, None]) * outer - blocks).abs())
        results.append((words, codes, err))
    (w6, c6, e6), (w4, c4, e4) = results
    four = e4 < e6
    words = torch.where(four, w4, w6)
    codes = torch.where(four[:, None], c4, c6)
    return words, codes, four


def rows() -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(20261121)
    out = []
    for scale in (1e-3, 0.05, 1.0, 7.0, 300.0):
        for _ in range(12):
            out.append(torch.randn(128, generator=g) * scale)
    for _ in range(12):  # heavy tails
        out.append(torch.randn(128, generator=g) ** 3)
    blocky = torch.randn(256, generator=g)
    blocky[16:48] = 0.0
    out.append(blocky)
    out.append(torch.zeros(64))
    for position in (0, 7, 63):
        spike = torch.randn(64, generator=g) * 0.01
        spike[position] = 40.0
        out.append(spike)
    # values on E2M1 rounding boundaries relative to a block maximum of 6
    edges = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 0.0])
    out.append(torch.cat([edges, edges * 0.5, edges * 4.0, edges * 1.5]))
    tiny = torch.randn(64, generator=g) * 1e-30
    tiny[0] = 50.0  # a large row amax with blocks whose scales underflow E4M3
    out.append(tiny)
    out.append(torch.tensor([3.3895e38, -3.3895e38] + [1.0] * 14))
    out.append(torch.tensor([1e-38, -2e-38] + [3e-39] * 14))
    return [r.to(torch.bfloat16).to(torch.float32) for r in out]


def main() -> int:
    cases = []
    for row in rows():
        words, codes, four = quantize(row)
        cases.append({"row": [float(v) for v in row], "scale_words": words.tolist(), "codes": codes.reshape(-1).tolist(),
                      "four": four.tolist()})
    blocks = sum(len(c["four"]) for c in cases)
    fours = sum(sum(c["four"]) for c in cases)
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump({"rule": "four_over_six", "torch": torch.__version__, "cases": cases}, handle)
    print(f"{len(cases)} rows, {blocks} blocks, {fours} took the 4 scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
