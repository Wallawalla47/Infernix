"""Exact oracle for the lossless route's elementwise and row operations (include/infernix/ops/lossless_ops.h).

Writes test cases and runs the CUDA test on them: python lossless_ops_oracle.py <family executable> lossless_ops.
Every expected value follows docs/maintainer/lossless-route.md §2 and §2a step by step: sums exact (fractions),
transcendental functions correctly rounded (decimal at 60 digits, lossless_cr_oracle.py), every
intermediate rounded to the dtype the specification gives it. Standard library only.
"""
from __future__ import annotations

import itertools
import math
import random
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossless_cr_oracle as cr  # noqa: E402

(RMSNORM, RMSNORM_GATED, ADD, SILU_MUL, SIGMOID_MUL, GDN_GATES, CAUSAL_CONV, ATTENTION, DELTA_RULE, ROPE, ROUTER,
 MOE_COMBINE, MOE_ROUTE, RMSNORM_GROUPED, RMSNORM_GATED_SIGMOID, HC_GATES, HC_MIX, HC_INJECT, PLE_GATE, PLE_CONV,
 QSA_POOL, QSA_SELECT, ATTENTION_SELECT, ROUTER_BF16, MOE_COMBINE_ENTRIES, CONV_SLOTS, DELTA_RULE_SLOTS, ATTENTION_PAGED,
 PLE_CONV_SLOTS, QSA_CACHE, QSA_SELECT_POSITIONS, ATTENTION_PAGED_SELECT, QUANTIZE_NVFP4, QUANTIZE_NVFP4_SCALED,
 QUANTIZE_FP8_TOKEN, ROUND_FP16, QUANTIZE_FP8_STATIC, CODEC_ATTENTION) = range(38)


def bf16_value(bits: int) -> Fraction:
    return Fraction(struct.unpack("<f", struct.pack("<I", bits << 16))[0])


def f32_value(bits: int) -> Fraction:
    return Fraction(struct.unpack("<f", struct.pack("<I", bits))[0])


def f32_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def round_fraction(a: Fraction, prec: int) -> int:
    """Correctly rounded bits of the exact rational a (binary32: prec 24; BF16: prec 8, low 16 bits)."""
    num, den = a.numerator, a.denominator
    if num == 0:
        return 0
    if den & (den - 1) == 0:
        return round_scaled(num, den.bit_length() - 1, prec)
    sign = 1 if num < 0 else 0
    mag = -num if sign else num
    # floor(log2 a) is e or e - 1 for this e.
    e = mag.bit_length() - den.bit_length()
    if (mag << max(0, -e)) < (den << max(0, e)):
        e -= 1
    e = max(e, -126)
    shift = e - prec + 1
    divisor = den << max(0, shift)
    n, rem = divmod(mag << max(0, -shift), divisor)
    if 2 * rem > divisor or (2 * rem == divisor and n % 2 == 1):
        n += 1
    return pack_rounded(sign, e, n, prec)


def round_scaled(value: int, scale: int, prec: int) -> int:
    """round_fraction(value / 2^scale, prec), in integers."""
    if value == 0:
        return 0
    sign = 1 if value < 0 else 0
    mag = -value if sign else value
    e = max(mag.bit_length() - 1 - scale, -126)
    drop = scale + e - prec + 1
    if drop <= 0:
        n = mag << -drop
    else:
        n = mag >> drop
        rem = mag & ((1 << drop) - 1)
        half = 1 << (drop - 1)
        if rem > half or (rem == half and n & 1):
            n += 1
    return pack_rounded(sign, e, n, prec)


def pack_rounded(sign: int, e: int, n: int, prec: int) -> int:
    """The bits of (-1)^sign n 2^(e - prec + 1), n the significand rounded to prec bits (2^prec when it carried)."""
    total = 16 if prec == 8 else 32
    if n == 2 ** prec:
        n //= 2
        e += 1
    if e > 127:
        return (sign << (total - 1)) | (255 << (prec - 1))
    if n < 2 ** (prec - 1):
        return (sign << (total - 1)) | n
    return (sign << (total - 1)) | ((e + 127) << (prec - 1)) | (n - 2 ** (prec - 1))


def sign_bit(bits: int, prec: int) -> int:
    return (bits >> (15 if prec == 8 else 31)) & 1


def signed_round(a: Fraction, prec: int, zero_negative: bool) -> int:
    """round_fraction, with an exact zero taking the sign IEEE gives it (zero_negative)."""
    bits = round_fraction(a, prec)
    if a == 0 and zero_negative:
        bits |= 1 << (15 if prec == 8 else 31)
    return bits


def float_of(bits: int, prec: int) -> float:
    """The signed Python float of BF16 (prec 8) or binary32 bits."""
    return struct.unpack("<f", struct.pack("<I", bits << 16 if prec == 8 else bits))[0]


def f32_of(v: Fraction) -> Fraction:
    return f32_value(round_fraction(v, 24))


def bf16_of(v: Fraction) -> Fraction:
    return bf16_value(round_fraction(v, 8))


def cr32(func: int, x: Fraction) -> Fraction:
    return f32_value(cr.evaluate(func, float(x)))


def crbf(func: int, x: Fraction) -> Fraction:
    return bf16_value(cr.evaluate(func, float(x)))


def random_bf16(rng: random.Random, spread: int = 8) -> int:
    v = rng.gauss(0.0, 1.0) * 2.0 ** rng.randint(-spread, spread)
    if rng.random() < 0.02:
        v = 0.0
    if rng.random() < 0.01:
        v = rng.gauss(0.0, 1.0) * 2.0 ** -130  # subnormal
    return f32_bits(v) >> 16


def rmsnorm_rows(xs: list[int], ws: list[int], w_fp32: bool, eps: float, width: int, gated: bool,
                 zs: list[int] | None, groups: int = 1, sigmoid: bool = False) -> list[int]:
    out = []
    eps_v = f32_value(f32_bits(eps))
    for row in range(len(xs) // width):
        xr = xs[row * width:(row + 1) * width]
        s = sum((bf16_value(b) ** 2 for b in xr), Fraction(0))
        m = f32_of(s / width)
        t = f32_value(round_fraction(m + eps_v, 24))
        r = cr32(cr.RSQRT, t)
        w0 = (row % groups) * width
        for i, b in enumerate(xr):
            w = f32_value(ws[w0 + i]) if w_fp32 else bf16_value(ws[w0 + i])
            ws_ = (ws[w0 + i] >> 31) & 1 if w_fp32 else (ws[w0 + i] >> 15) & 1
            sx = (b >> 15) & 1
            if not gated:
                y_bits = signed_round(bf16_value(b) * r, 24, sx == 1)
                y = f32_value(y_bits)
                gain = f32_of(1 + w)
                sg = 1 if gain < 0 else 0  # 1 + w is +0 when w = -1
                prod = f32_of(y * gain)
                out.append(signed_round(prod, 8, (sign_bit(y_bits, 24) ^ sg) == 1 or (prod == 0 and y * gain < 0)))
            else:
                n_bits = signed_round(f32_of(bf16_value(b) * r), 8, sx == 1)
                n = bf16_value(n_bits)
                sn = sign_bit(n_bits, 8)
                if w_fp32:
                    y_bits = signed_round(w * n, 24, (ws_ ^ sn) == 1)
                    y, sy = f32_value(y_bits), sign_bit(y_bits, 24)
                else:
                    y_bits = signed_round(w * n, 8, (ws_ ^ sn) == 1)
                    y, sy = bf16_value(y_bits), sign_bit(y_bits, 8)
                g_bits = cr.evaluate(cr.SIGMOID if sigmoid else cr.SILU, float_of(zs[row * width + i], 8))
                s_ = f32_value(g_bits)
                prod = f32_of(y * s_)
                out.append(signed_round(prod, 8, (sy ^ sign_bit(g_bits, 24)) == 1 or (prod == 0 and y * s_ < 0)))
    return out


def write_u32(f, values):
    f.write(struct.pack(f"<{len(values)}I", *values))


def cases(rng: random.Random):
    """Yields (op, dims, inputs (lists of u32), expected u32 list)."""
    # RMSNorm, BF16 and FP32 weights, widths 256 (q/k norms) and 5120 (hidden).
    for width, rows, w_fp32 in ((256, 48, False), (5120, 6, False), (5120, 4, True)):
        xs = [random_bf16(rng) for _ in range(width * rows)]
        # A row with outliers 2^12 apart and a row of tiny values.
        for i in range(0, width, 7):
            xs[i] = f32_bits(4096.0 * (1 if i % 14 == 0 else -1)) >> 16
        for i in range(width, 2 * width):
            xs[i] = f32_bits(rng.gauss(0.0, 1.0) * 2.0 ** -120) >> 16
        ws = [f32_bits(rng.gauss(0.0, 0.1)) if w_fp32 else f32_bits(rng.gauss(0.0, 0.1)) >> 16 for _ in range(width)]
        yield RMSNORM, [width, rows, int(w_fp32)], [xs, ws], rmsnorm_rows(xs, ws, w_fp32, 1e-6, width, False, None)
    # Gated RMSNorm (width 128).
    for w_fp32 in (False, True):
        width, rows = 128, 64
        os_ = [random_bf16(rng) for _ in range(width * rows)]
        zs = [random_bf16(rng, 4) for _ in range(width * rows)]
        ws = [f32_bits(rng.gauss(1.0, 0.2)) if w_fp32 else f32_bits(rng.gauss(1.0, 0.2)) >> 16 for _ in range(width)]
        yield (RMSNORM_GATED, [width, rows, int(w_fp32)], [os_, zs, ws],
               rmsnorm_rows(os_, ws, w_fp32, 1e-6, width, True, zs))
    # Elementwise pairs.
    n = 20000
    a = [random_bf16(rng, 40) for _ in range(n)]
    b = [random_bf16(rng, 40) for _ in range(n)]
    yield ADD, [n], [a, b], [round_fraction(bf16_value(x) + bf16_value(y), 8) for x, y in zip(a, b)]
    g = [random_bf16(rng, 5) for _ in range(n)]
    u = [random_bf16(rng, 5) for _ in range(n)]
    silu_mul = []
    for x, y in zip(g, u):
        act = cr.evaluate(cr.SILU_BF16, float_of(x, 8))
        silu_mul.append(signed_round(bf16_value(act) * bf16_value(y), 8, (sign_bit(act, 8) ^ sign_bit(y, 8)) == 1))
    yield SILU_MUL, [n], [g, u], silu_mul
    sigmoid_mul = []
    for x, y in zip(u, g):
        sig = cr.evaluate(cr.SIGMOID_BF16, float_of(y, 8))
        sigmoid_mul.append(signed_round(bf16_value(x) * bf16_value(sig), 8, (sign_bit(x, 8) ^ sign_bit(sig, 8)) == 1))
    yield SIGMOID_MUL, [n], [u, g], sigmoid_mul
    # GDN gates: H heads, T columns.
    heads, cols = 48, 64
    av = [random_bf16(rng, 3) for _ in range(heads * cols)]
    bv = [random_bf16(rng, 3) for _ in range(heads * cols)]
    a_log = [f32_bits(rng.uniform(-3.0, 2.0)) for _ in range(heads)]
    dt = [f32_bits(rng.uniform(-6.0, 2.0)) for _ in range(heads)]
    gs, betas = [], []
    for i in range(heads * cols):
        h = i % heads
        betas.append(cr.evaluate(cr.SIGMOID_BF16, float(bf16_value(bv[i]))))
        t = f32_of(bf16_value(av[i]) + f32_value(dt[h]))
        sp = cr32(cr.SOFTPLUS, t)
        e = cr32(cr.EXP, f32_value(a_log[h]))
        gs.append(round_fraction(-(e * sp), 24))
    yield GDN_GATES, [heads, cols], [av, bv, a_log, dt], gs + betas
    # Causal conv: C channels, T columns, 4 taps.
    channels, cols, taps = 96, 40, 4
    x = [random_bf16(rng, 20) for _ in range(channels * cols)]
    w = [random_bf16(rng, 2) for _ in range(channels * taps)]
    expected = []
    for t in range(cols):
        for c in range(channels):
            s = Fraction(0)
            for tap in range(taps):
                src = t - taps + 1 + tap
                if src >= 0:
                    s += bf16_value(x[src * channels + c]) * bf16_value(w[tap * channels + c])
            expected.append(cr.evaluate(cr.SILU_BF16, float_of(round_fraction(s, 24), 24)))
    yield CAUSAL_CONV, [channels, cols, taps], [x, w], expected
    # Attention: [256, heads, T] q, [256, kv heads, T] k and v. One case of ordinary values, one with
    # products far below the largest (q and k elements 2^-80 against 2^20: the exact fallback).
    for spread in (False, True):
        heads, kv_heads, cols = 4, 2, 24
        q = [random_bf16(rng, 2) for _ in range(256 * heads * cols)]
        k = [random_bf16(rng, 2) for _ in range(256 * kv_heads * cols)]
        v = [random_bf16(rng, 3) for _ in range(256 * kv_heads * cols)]
        if spread:
            for idx in range(0, len(q), 37):
                q[idx] = f32_bits(rng.choice([1.0, -1.0]) * 2.0 ** 20) >> 16
            for idx in range(5, len(q), 41):
                q[idx] = f32_bits(rng.choice([1.0, -1.0]) * 2.0 ** -80) >> 16
            for idx in range(3, len(k), 29):
                k[idx] = f32_bits(rng.choice([1.0, -1.0]) * 2.0 ** -70) >> 16
            # Value elements far below their channel's largest (rounded to the value grid).
            for idx in range(7, len(v), 23):
                v[idx] = f32_bits(rng.gauss(0.0, 1.0) * 2.0 ** -40) >> 16
        yield ATTENTION, [heads, kv_heads, cols], [q, k, v], attention(q, k, v, heads, kv_heads, cols)
    # Gated delta rule: Hk = 1, Hv = 2; g negative (decay below one), beta in (0, 1). 8 tokens, and 70
    # (Hk = 1, Hv = 2: two of the kernel's 32-token passes and a partial one).
    for hk, hv, cols in ((1, 2, 8), (1, 2, 70)):
        q = [random_bf16(rng, 2) for _ in range(128 * hk * cols)]
        k = [random_bf16(rng, 2) for _ in range(128 * hk * cols)]
        v = [random_bf16(rng, 2) for _ in range(128 * hv * cols)]
        g = [f32_bits(-rng.uniform(0.0, 2.0)) for _ in range(hv * cols)]
        beta = [f32_bits(rng.uniform(0.05, 0.95)) >> 16 for _ in range(hv * cols)]
        yield DELTA_RULE, [hk, hv, cols], [q, k, v, g, beta], delta_rule(q, k, v, g, beta, hk, hv, cols)
    # RoPE (theta 1e7, rotary 64 of 256): 1-D positions up to 262,143 and MRoPE positions differing per
    # axis.
    for axes in (1, 3):
        heads, cols = 3, 20
        positions = [rng.randrange(0, 262144) if i % 4 else i for i in range(cols * axes)]
        x = [random_bf16(rng, 6) for _ in range(256 * heads * cols)]
        yield ROPE, [heads, cols, axes], [positions, x], rope(positions, x, heads, cols, axes)
    # Router: 256 experts (logits [257, T], the last row the shared gate's), top 8; a column of equal
    # logits (ties to the lower expert) and one of widely spread logits.
    experts, top_k, cols = 256, 8, 12
    logits = [random_bf16(rng, 2) for _ in range((experts + 1) * cols)]
    for e in range(experts + 1):
        logits[1 * (experts + 1) + e] = f32_bits(0.5) >> 16
        logits[2 * (experts + 1) + e] = f32_bits(rng.uniform(-90.0, 30.0)) >> 16
    yield ROUTER, [experts, top_k, cols], [logits], router(logits, experts, top_k, cols)
    # Combine: width 64, 9 columns, top 8 slots placed at scattered gathered columns, a shared expert.
    width, cols, top_k, gathered = 64, 9, 8, 128
    expert_out = [random_bf16(rng, 6) for _ in range(width * gathered)]
    ids, weights, slot_column = [], [], []
    columns_free = list(range(gathered))
    rng.shuffle(columns_free)
    for t in range(cols):
        chosen = rng.sample(range(256), top_k)
        ids += chosen
        weights += [f32_bits(rng.uniform(0.0, 0.4)) for _ in range(top_k)]
        slot_column += [columns_free.pop() for _ in range(top_k)]
    shared_out = [random_bf16(rng, 6) for _ in range(width * cols)]
    shared_logits = [random_bf16(rng, 2) for _ in range(3 * cols)]
    residual = [random_bf16(rng, 8) for _ in range(width * cols)]
    yield (MOE_COMBINE, [width, cols, top_k, gathered],
           [expert_out, weights, ids, slot_column, shared_out, shared_logits, residual],
           moe_combine(expert_out, weights, ids, slot_column, shared_out, shared_logits, residual, width, cols, top_k))
    # Routing (structure, checked by the test itself): 256 experts, top 8, 70 columns.
    cols = 70
    ids = []
    for t in range(cols):
        ids += rng.sample(range(256) if t % 3 else range(12), 8)
    x = [random_bf16(rng, 4) for _ in range(64 * cols)]
    yield MOE_ROUTE, [256, 8, cols], [ids, x], [1]
    # The gathered-column bound at its extremes: one decoded column (8 runs of one pair: 512 columns, all
    # used by runs), every expert hit once (256 runs of one pair fill all 256 tiles), and runs of whole
    # tiles (8 experts, 64 pairs each).
    for experts, top_k, cols, pick in ((256, 8, 1, None), (256, 8, 32, "distinct"), (8, 8, 64, "all")):
        if pick == "distinct":
            ids = [e for e in range(256)]
        elif pick == "all":
            ids = [e for _ in range(cols) for e in range(8)]
        else:
            ids = rng.sample(range(experts), top_k)
        x = [random_bf16(rng, 4) for _ in range(64 * cols)]
        yield MOE_ROUTE, [experts, top_k, cols], [ids, x], [1]
    yield from flash_cases(rng)
    yield from state_cases(rng)
    yield from flash_state_cases(rng)


def bf16_add_bits(a_bits: int, b_bits: int) -> int:
    """bf16(a + b) of two BF16 values with IEEE's zero signs."""
    return signed_round(bf16_value(a_bits) + bf16_value(b_bits), 8,
                        math.copysign(1.0, float_of(a_bits, 8) + float_of(b_bits, 8)) < 0)


def flash_state_cases(rng: random.Random):
    """Qwen3.8-Flash-Next's cache forms (generation)."""
    # PLE convolution with history: 160 channels, 4 taps, dilation 3 (span 9); runs of 5 columns (shorter
    # than the span) in 2 sequences and of 12 in 1; 3 slots, one updated in place.
    channels, taps, dilation, slots = 160, 4, 3, 3
    span = (taps - 1) * dilation
    for width, seqs, src, dst in ((5, 2, [2, 0], [2, 1]), (12, 1, [1], [0])):
        cols = width * seqs
        gt = [random_bf16(rng, 4) for _ in range(channels * cols)]
        nz = [random_bf16(rng, 20) for _ in range(channels * cols)]
        w = [random_bf16(rng, 2) for _ in range(channels * taps)]
        res = [random_bf16(rng, 4) for _ in range(channels * cols)]
        hist = [random_bf16(rng, 20) for _ in range(channels * span * slots)]
        out = [0] * (channels * cols)
        final = list(hist)
        for b in range(seqs):
            for c in range(channels):
                u = [hist[(src[b] * span + k) * channels + c] for k in range(span)]
                u += [nz[(b * width + t) * channels + c] for t in range(width)]
                for t in range(width):
                    s = sum((bf16_value(w[tap * channels + c]) * bf16_value(u[span + t - (taps - 1 - tap) * dilation])
                             for tap in range(taps)), Fraction(0))
                    act = cr.evaluate(cr.SILU_BF16, float_of(round_fraction(s, 8), 8))
                    i = (b * width + t) * channels + c
                    out[i] = bf16_add_bits(res[i], bf16_add_bits(gt[i], act))
                for k in range(span):
                    final[(dst[b] * span + k) * channels + c] = u[width + k]
        yield (PLE_CONV_SLOTS, [channels, width, seqs, taps, dilation, slots], [gt, nz, w, res, hist, src, dst],
               out + final)
    # The QSA indexer cache: dim 128, ratio 4, 2 sequences of 7 columns from positions 6 and 9 (each
    # completes two blocks, one of them begun before the call), tails in 3 slots, one page per row.
    dim, ratio, width, seqs, slots = 128, 4, 7, 2, 3
    starts, tail_slots, rows = [6, 9], [2, 0], [1, 0]
    pages, stride = 3, 1
    page_of_row = [[2], [0]]  # row r's page table
    cols = width * seqs
    positions = [starts[b] + i for b in range(seqs) for i in range(width)]
    keys = [random_bf16(rng, 30) for _ in range(dim * cols)]
    tails = [random_bf16(rng, 30) for _ in range(dim * (ratio - 1) * slots)]
    rope = [rng.randrange(0, 100000) for _ in range(3 * cols)]
    block_start = [rng.randrange(0, 100000) for _ in range(3 * seqs)]
    plane = [random_bf16(rng, 30) for _ in range((dim // ratio) * 64 * pages)]
    tables = [page_of_row[r][0] for r in range(2)]

    def raw_key(b, q, d):
        p0 = starts[b]
        if q >= p0:
            return keys[(b * width + q - p0) * dim + d]
        return tails[(tail_slots[b] * (ratio - 1) + q % ratio) * dim + d]

    pooled = [0] * (dim * cols)
    block_rope = [0] * (3 * cols)
    plane_final = list(plane)
    for b in range(seqs):
        for i in range(width):
            t, p = b * width + i, starts[b] + i
            complete = (p + 1) % ratio == 0
            first = p + 1 - ratio
            for a in range(3):
                if not complete:
                    block_rope[a * cols + t] = rope[a * cols + t]
                elif first >= starts[b]:
                    block_rope[a * cols + t] = rope[a * cols + t - (p - first)]
                else:
                    block_rope[a * cols + t] = block_start[a * seqs + b]
            if complete:
                page = tables[rows[b]]
                for d in range(dim):
                    s = sum((bf16_value(raw_key(b, first + k, d)) for k in range(ratio)), Fraction(0))
                    v = round_fraction(f32_of(s / ratio), 8)
                    pooled[t * dim + d] = v
                    plane_final[(page * 64 + first % 64) * (dim // ratio) + d] = v
    tails_final = list(tails)
    for b in range(seqs):
        last = starts[b] + width - 1
        open_from = (last + 1) // ratio * ratio
        for p in range(open_from, last + 1):
            for d in range(dim):
                tails_final[(tail_slots[b] * (ratio - 1) + p % ratio) * dim + d] = keys[(b * width + p - starts[b]) * dim + d]
    gathered = []
    for b in range(seqs):
        n = (starts[b] + width) // ratio
        page = tables[rows[b]]
        for blk in range(n):
            for d in range(dim):
                gathered.append(plane_final[(page * 64 + (blk * ratio) % 64) * (dim // ratio) + d])
    yield (QSA_CACHE, [dim, ratio, width, seqs, slots, stride, pages],
           [keys, positions, tails, tail_slots, rope, block_start, plane, tables, rows],
           pooled + block_rope + plane_final + tails_final + gathered)
    # Selection by position: 4 heads, ratio 4, budget 5, queries at positions 26..33 (6 to 8 blocks).
    heads, ratio, budget, dim = 4, 4, 5, 128
    qpos = list(range(26, 34))
    cols, blocks = len(qpos), 8
    sc = [f32_bits(rng.choice([0.5, 0.25, -1.0, rng.gauss(0.0, 1.0)])) for _ in range(cols * heads * blocks)]
    words = 1
    divisor = f32_value(f32_bits(math.sqrt(dim)))
    expected = []
    for t in range(cols):
        n = (qpos[t] + 1) // ratio
        chosen = set(range(n))
        if n > budget:
            vals = []
            for b in range(n):
                s = sum((max(f32_value(sc[(t * heads + h) * blocks + b]), Fraction(0)) for h in range(heads)), Fraction(0))
                vals.append(f32_of(f32_of(s) / divisor))
            chosen = set(sorted(range(n), key=lambda b: (-vals[b], b))[:budget])
        bits = 0
        for b in chosen:
            bits |= 1 << b
        expected.append(bits)
    yield QSA_SELECT_POSITIONS, [heads, ratio, budget, dim, cols, blocks], [sc, qpos], expected
    # Paged attention with a selection: 2 sequences of 3 columns from positions 40 and 25, ratio 4.
    heads, kv_heads, width, ratio = 4, 2, 3, 4
    starts = [40, 25]
    seqs = len(starts)
    lens = [s + width for s in starts]
    ks = [[random_bf16(rng, 2) for _ in range(256 * kv_heads * n)] for n in lens]
    vs = [[random_bf16(rng, 3) for _ in range(256 * kv_heads * n)] for n in lens]
    q = [random_bf16(rng, 2) for _ in range(256 * heads * width * seqs)]
    positions = [starts[b] + w for b in range(seqs) for w in range(width)]
    sel = []
    for p in positions:
        bits = 1
        for b in range(1, (p + 1) // ratio):
            if rng.random() < 0.5:
                bits |= 1 << b
        sel.append(bits)
    stride = max((n + 63) // 64 for n in lens)
    pages = sum((n + 63) // 64 for n in lens) + 2
    free = list(range(pages))
    rng.shuffle(free)
    tables = []
    for n in lens:
        row = [free.pop() for _ in range((n + 63) // 64)]
        tables += row + [0] * (stride - len(row))
    expected = attention_paged(q, ks, vs, heads, kv_heads, width, seqs, positions, sel, ratio)
    yield (ATTENTION_PAGED_SELECT, [heads, kv_heads, width, seqs, stride, pages, max(lens), ratio],
           [q, [x for kk in ks for x in kk], [x for vv in vs for x in vv], positions, lens, tables, sel], expected)


def state_cases(rng: random.Random):
    """The slot forms (generation): runs of columns continuing from and leaving recurrent states."""
    # Causal conv: 3 sequences of 5 columns (and of 2, shorter than the state), 4 taps, 5 slots; one
    # sequence updates its slot in place.
    channels, taps, slots = 96, 4, 5
    for width, seqs, src, dst in ((5, 3, [1, 4, 0], [2, 4, 3]), (2, 2, [3, 1], [3, 0])):
        x = [random_bf16(rng, 20) for _ in range(channels * width * seqs)]
        w = [random_bf16(rng, 2) for _ in range(channels * taps)]
        states = [random_bf16(rng, 20) for _ in range(channels * (taps - 1) * slots)]
        out, final = conv_slots(x, w, states, src, dst, channels, width, seqs, taps)
        yield CONV_SLOTS, [channels, width, seqs, taps, slots], [x, w, states, src, dst], out + final
    # Gated delta rule: Hk = 1, Hv = 2, 2 sequences of 3 tokens from random FP32 states (4 slots), and 2
    # of 37 tokens (a 32-token pass and a partial one each).
    for hk, hv, width, seqs, slots in ((1, 2, 3, 2, 4), (1, 2, 37, 2, 4)):
        cols = width * seqs
        q = [random_bf16(rng, 2) for _ in range(128 * hk * cols)]
        k = [random_bf16(rng, 2) for _ in range(128 * hk * cols)]
        v = [random_bf16(rng, 2) for _ in range(128 * hv * cols)]
        g = [f32_bits(-rng.uniform(0.0, 2.0)) for _ in range(hv * cols)]
        beta = [f32_bits(rng.uniform(0.05, 0.95)) >> 16 for _ in range(hv * cols)]
        states = [f32_bits(rng.gauss(0.0, 0.05)) for _ in range(128 * 128 * hv * slots)]
        src, dst = [2, 0], [2, 3]
        out, final = delta_rule_slots(q, k, v, g, beta, hk, hv, width, seqs, states, src, dst)
        yield DELTA_RULE_SLOTS, [hk, hv, width, seqs, slots], [q, k, v, g, beta, states, src, dst], out + final
    # Attention over the paged cache: (a) two sequences of 2 query columns (300 and 150 keys), with
    # key ranges split over CTAs and some elements far below their rows' largest (the exact fallback);
    # (b) one sequence of 24 columns from position 70 (one key range).
    for heads, kv_heads, width, starts, spread in ((4, 2, 2, [298, 148], True), (4, 1, 24, [70], False)):
        seqs = len(starts)
        lens = [s + width for s in starts]
        ks = [[random_bf16(rng, 2) for _ in range(256 * kv_heads * n)] for n in lens]
        vs = [[random_bf16(rng, 3) for _ in range(256 * kv_heads * n)] for n in lens]
        q = [random_bf16(rng, 2) for _ in range(256 * heads * width * seqs)]
        if spread:
            for idx in range(5, len(q), 41):
                q[idx] = f32_bits(rng.choice([1.0, -1.0]) * 2.0 ** -80) >> 16
            for kk in ks:
                for idx in range(3, len(kk), 97):
                    kk[idx] = f32_bits(rng.choice([1.0, -1.0]) * 2.0 ** -70) >> 16
            for vv in vs:
                for idx in range(7, len(vv), 89):
                    vv[idx] = f32_bits(rng.gauss(0.0, 1.0) * 2.0 ** -40) >> 16
        positions = [starts[b] + w for b in range(seqs) for w in range(width)]
        stride = max((n + 63) // 64 for n in lens)
        pages = sum((n + 63) // 64 for n in lens) + 3
        free = list(range(pages))
        rng.shuffle(free)
        tables = []
        for n in lens:
            row = [free.pop() for _ in range((n + 63) // 64)]
            tables += row + [0] * (stride - len(row))
        max_keys = max(lens)
        expected = attention_paged(q, ks, vs, heads, kv_heads, width, seqs, positions)
        yield (ATTENTION_PAGED, [heads, kv_heads, width, seqs, stride, pages, max_keys],
               [q, [x for kk in ks for x in kk], [x for vv in vs for x in vv], positions, lens, tables], expected)


def attention_paged(q, ks, vs, heads, kv_heads, width, seqs, positions, select=None, ratio=0):
    """Query column t = b width + w at positions[t] over keys 0..positions[t] of sequence b (ks[b], vs[b]:
    [len, kv_heads, 256]); with a selection, the keys of its tail and its selected blocks only."""
    group = heads // kv_heads
    out = [0] * (256 * heads * width * seqs)
    for b in range(seqs):
        kf = [bf16_value(x) for x in ks[b]]
        vf = [bf16_value(x) for x in vs[b]]
        for w in range(width):
            t = b * width + w
            pos = positions[t]
            for h in range(heads):
                kvh = h // group
                qr = [bf16_value(q[(t * heads + h) * 256 + d]) for d in range(256)]
                if select is None:
                    keys = list(range(pos + 1))
                else:
                    tail = (pos + 1) // ratio * ratio
                    keys = [j for j in range(pos + 1) if j >= tail or (select[t] >> (j // ratio)) & 1]
                scores = []
                for j in keys:
                    base = (j * kv_heads + kvh) * 256
                    scores.append(sum((a * kf[base + d] for d, a in enumerate(qr)), Fraction(0)) / 16)
                m = max(scores)
                ps = [f32_value(cr.evaluate(cr.EXP, float(f32_of(s - m)))) for s in scores]
                den = sum(ps, Fraction(0))
                for c in range(256):
                    num = sum((p * vf[(j * kv_heads + kvh) * 256 + c] for j, p in zip(keys, ps)), Fraction(0))
                    out[(t * heads + h) * 256 + c] = round_fraction(num / den, 8)
    return out


def conv_slots(x, w, states, src, dst, channels, width, seqs, taps):
    keep = taps - 1
    out = [0] * (channels * width * seqs)
    final = list(states)
    for b in range(seqs):
        for c in range(channels):
            hist = [states[(src[b] * keep + h) * channels + c] for h in range(keep)]
            run = hist + [x[(b * width + t) * channels + c] for t in range(width)]
            for t in range(width):
                s = sum((bf16_value(run[t + tap]) * bf16_value(w[tap * channels + c]) for tap in range(taps)),
                        Fraction(0))
                out[(b * width + t) * channels + c] = cr.evaluate(cr.SILU_BF16, float_of(round_fraction(s, 24), 24))
            for h in range(keep):
                final[(dst[b] * keep + h) * channels + c] = run[width + h]
    return out, final


# The delta rule's oracle computes in integers: a binary32 value v is the integer v 2^149 (its smallest step is
# 2^-149), so a product of two is exact at 2^298 and sums are exact; each rounding is round_scaled's.
F32_SCALE = 149


def f32_int(bits: int) -> int:
    """The finite binary32 value of bits, times 2^149."""
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    mag = m if e == 0 else (m | 0x800000) << (e - 1)
    return -mag if bits >> 31 else mag


def f32_round(value: int, scale: int) -> int:
    """value / 2^scale rounded to binary32, times 2^149."""
    if value == 0:
        return 0
    mag = -value if value < 0 else value
    e = max(mag.bit_length() - 1 - scale, -126)
    drop = scale + e - 23
    if drop <= 0:
        n = mag << -drop
    else:
        n = mag >> drop
        rem = mag & ((1 << drop) - 1)
        half = 1 << (drop - 1)
        if rem > half or (rem == half and n & 1):
            n += 1
    if n == 1 << 24:
        n >>= 1
        e += 1
    if e > 127:
        raise OverflowError("binary32 overflow in the delta rule oracle")
    n <<= e + 126
    return -n if value < 0 else n


def delta_rule_column(S, qb, kb, vb, g_bits, beta_bits) -> list[int]:
    """One column of the gated delta rule on the head's state S (S[i][j], f32_int values), in place, from its
    128 BF16 bits of q, k and v, FP32 bits of g and BF16 bits of beta; returns the column's BF16 output bits."""
    eps = f32_int(f32_bits(1e-6))
    scale = f32_int(0x3DB504F3)
    s2 = 2 * F32_SCALE
    qv = [f32_int(x << 16) for x in qb]
    kv = [f32_int(x << 16) for x in kb]
    rq_in = round_scaled(f32_round(sum(x * x for x in qv), s2) + eps, F32_SCALE, 24)
    rk_in = round_scaled(f32_round(sum(x * x for x in kv), s2) + eps, F32_SCALE, 24)
    rq = f32_int(cr.evaluate(cr.RSQRT, float_of(rq_in, 24)))
    rk = f32_int(cr.evaluate(cr.RSQRT, float_of(rk_in, 24)))
    qn = [f32_round(f32_round(x * rq, s2) * scale, s2) for x in qv]
    kn = [f32_round(x * rk, s2) for x in kv]
    decay = f32_int(cr.evaluate(cr.EXP, float(f32_value(g_bits))))
    bt = f32_int(beta_bits << 16)
    for row in S:
        for j in range(128):
            row[j] = f32_round(row[j] * decay, s2)
    out = []
    for j in range(128):
        u = f32_round(sum(f32_round(S[i][j] * kn[i], s2) for i in range(128)), F32_SCALE)
        delta = f32_round(f32_round(f32_int(vb[j] << 16) - u, F32_SCALE) * bt, s2)
        for i in range(128):
            S[i][j] = f32_round(S[i][j] + f32_round(kn[i] * delta, s2), F32_SCALE)
        o = sum(f32_round(S[i][j] * qn[i], s2) for i in range(128))
        out.append(round_scaled(f32_round(o, F32_SCALE), F32_SCALE, 8))
    return out


def delta_rule_slots(q, k, v, g, beta, hk, hv, width, seqs, states, src, dst):
    """The rule per sequence from its source slot; states FP32 [128 (k), 128 (v), Hv, slots]."""
    out = [0] * (128 * hv * width * seqs)
    final = list(states)
    for b in range(seqs):
        for head in range(hv):
            kh = head // (hv // hk)
            base = (src[b] * hv + head) * 128 * 128
            S = [[f32_int(states[base + j * 128 + i]) for j in range(128)] for i in range(128)]  # S[i][j]
            for w_ in range(width):
                t = b * width + w_
                o = (t * hv + head) * 128
                out[o:o + 128] = delta_rule_column(S, q[(t * hk + kh) * 128:(t * hk + kh + 1) * 128],
                                                   k[(t * hk + kh) * 128:(t * hk + kh + 1) * 128], v[o:o + 128],
                                                   g[t * hv + head], beta[t * hv + head])
            dbase = (dst[b] * hv + head) * 128 * 128
            for i in range(128):
                for j in range(128):
                    final[dbase + j * 128 + i] = round_scaled(S[i][j], F32_SCALE, 24)
    return out, final


def round_sqrt(a: Fraction, prec: int) -> int:
    """Correctly rounded bits of sqrt(a) (a >= 0)."""
    if a == 0:
        return 0
    k = 120 + max(0, -(a.numerator.bit_length() - a.denominator.bit_length()))
    scaled = a * (1 << (2 * k))
    n = scaled.numerator // scaled.denominator
    s = math.isqrt(n)
    exact = s * s == n and scaled.denominator == 1
    v = Fraction(2 * s + (0 if exact else 1), 1 << (k + 1))
    return round_fraction(v, prec)


def flash_cases(rng: random.Random):
    """Qwen3.8-Flash-Next's operations (§2a)."""
    # Grouped RMSNorm: 4 streams of 2560 (BF16 weights) and of 128 (FP32 weights).
    for width, groups, cols, w_fp32 in ((2560, 4, 3, False), (128, 4, 9, True)):
        xs = [random_bf16(rng) for _ in range(width * groups * cols)]
        ws = [f32_bits(rng.gauss(0.0, 0.1)) if w_fp32 else f32_bits(rng.gauss(0.0, 0.1)) >> 16
              for _ in range(width * groups)]
        yield (RMSNORM_GROUPED, [width, groups, cols, int(w_fp32)], [xs, ws],
               rmsnorm_rows(xs, ws, w_fp32, 1e-6, width, False, None, groups))
    # Gated RMSNorm with the sigmoid gate (Flash-Next's output_gate_type).
    width, rows = 128, 48
    os_ = [random_bf16(rng) for _ in range(width * rows)]
    zs = [random_bf16(rng, 4) for _ in range(width * rows)]
    ws = [f32_bits(rng.gauss(1.0, 0.2)) >> 16 for _ in range(width)]
    yield (RMSNORM_GATED_SIGMOID, [width, rows, 0], [os_, zs, ws],
           rmsnorm_rows(os_, ws, False, 1e-6, width, True, zs, 1, True))
    # Hyper-connection gates: rank 320 + 4 injection rows, 7 columns (one large, one tiny value).
    rank, streams, cols = 320, 4, 7
    rows = rank + streams
    dj = [random_bf16(rng, 6) for _ in range(rows * cols)]
    dj[3] = f32_bits(2.0 ** -133) >> 16  # d / 4 rounds (subnormal)
    mix, inject = [0] * (rank * cols), [0] * (streams * cols)
    for t in range(cols):
        for r in range(rows):
            d = dj[t * rows + r]
            q_bits = signed_round(bf16_value(d) / streams, 8, sign_bit(d, 8) == 1)
            if r < rank:
                mix[t * rank + r] = cr.evaluate(cr.SILU_BF16, float_of(q_bits, 8))
            else:
                inject[t * streams + r - rank] = round_fraction(2 * crbf(cr.SIGMOID_BF16, bf16_value(q_bits)), 8)
    yield HC_GATES, [rank, streams, cols], [dj], mix + inject
    # Mix: x_i = bf16(sum_c bf16(bf16(sigmoid(u)) n) / S); width 192, 4 streams, 5 columns.
    width, streams, cols = 192, 4, 5
    up = [random_bf16(rng, 4) for _ in range(width * streams * cols)]
    nm = [random_bf16(rng, 40) for _ in range(width * streams * cols)]
    x = []
    for t in range(cols):
        for i in range(width):
            s = sum((bf16_of(crbf(cr.SIGMOID_BF16, bf16_value(up[(t * streams + c) * width + i])) *
                             bf16_value(nm[(t * streams + c) * width + i])) for c in range(streams)), Fraction(0))
            x.append(round_fraction(s / streams, 8))
    yield HC_MIX, [width, streams, cols], [up, nm], x
    # Injection: R_ci = bf16(R_ci + bf16(y_i j_c)).
    y = [random_bf16(rng, 6) for _ in range(width * cols)]
    j = [round_fraction(2 * bf16_value(random_bf16(rng, 0)) ** 2 / (1 + bf16_value(random_bf16(rng, 0)) ** 2), 8)
         for _ in range(streams * cols)]
    res = [random_bf16(rng, 6) for _ in range(width * streams * cols)]
    out = []
    for t in range(cols):
        for c in range(streams):
            for i in range(width):
                yb, jb = y[t * width + i], j[t * streams + c]
                term = signed_round(bf16_value(yb) * bf16_value(jb), 8, (sign_bit(yb, 8) ^ sign_bit(jb, 8)) == 1)
                rb = res[(t * streams + c) * width + i]
                out.append(signed_round(bf16_value(rb) + bf16_value(term), 8,
                                        math.copysign(1.0, float_of(rb, 8) + float_of(term, 8)) < 0))
    yield HC_INJECT, [width, streams, cols], [y, j, res], out
    # PLE gate: width 2560, 4 streams, 4 columns; a column whose scores fall below the 1e-6 clamp.
    width, streams, cols = 2560, 4, 4
    kn = [random_bf16(rng, 2) for _ in range(width * streams * cols)]
    qn = [random_bf16(rng, 2) for _ in range(width * streams * cols)]
    for i in range(width * streams * 3, width * streams * 4):
        qn[i] = f32_bits(rng.gauss(0.0, 1.0) * 2.0 ** -40) >> 16
    val = [random_bf16(rng, 3) for _ in range(width * cols)]
    divisor = f32_value(f32_bits(math.sqrt(width)))
    floor_ = bf16_of(f32_value(f32_bits(1e-6)))
    scores, gated = [], []
    for t in range(cols):
        for c in range(streams):
            base = (t * streams + c) * width
            g = bf16_of(sum((bf16_of(bf16_value(kn[base + i]) * bf16_value(qn[base + i])) for i in range(width)),
                            Fraction(0)))
            g = bf16_of(g / divisor)
            a = abs(g) if abs(g) >= f32_value(f32_bits(1e-6)) else floor_
            root = bf16_value(round_sqrt(a, 8))
            s = root if g > 0 else (-root if g < 0 else Fraction(0))
            scores.append(round_fraction(s, 8))
    for t in range(cols):
        for c in range(streams):
            sg = crbf(cr.SIGMOID_BF16, bf16_value(scores[t * streams + c]))  # positive
            for i in range(width):
                vb = val[t * width + i]
                gated.append(signed_round(sg * bf16_value(vb), 8, sign_bit(vb, 8) == 1))
    yield PLE_GATE, [width, streams, cols], [kn, qn, val], scores + gated
    # PLE convolution: 160 channels, 20 columns, 4 taps, dilation 3.
    channels, cols, taps, dilation = 160, 20, 4, 3
    gt = [random_bf16(rng, 4) for _ in range(channels * cols)]
    nz = [random_bf16(rng, 20) for _ in range(channels * cols)]
    w = [random_bf16(rng, 2) for _ in range(channels * taps)]
    res = [random_bf16(rng, 4) for _ in range(channels * cols)]
    out = []
    for t in range(cols):
        for c in range(channels):
            s = Fraction(0)
            for tap in range(taps):
                src = t - (taps - 1 - tap) * dilation
                if src >= 0:
                    s += bf16_value(w[tap * channels + c]) * bf16_value(nz[src * channels + c])
            act_bits = cr.evaluate(cr.SILU_BF16, float_of(round_fraction(s, 8), 8))
            g_bits = gt[t * channels + c]
            o_bits = signed_round(bf16_value(g_bits) + bf16_value(act_bits), 8,
                                  math.copysign(1.0, float_of(g_bits, 8) + float_of(act_bits, 8)) < 0)
            r_bits = res[t * channels + c]
            out.append(signed_round(bf16_value(r_bits) + bf16_value(o_bits), 8,
                                    math.copysign(1.0, float_of(r_bits, 8) + float_of(o_bits, 8)) < 0))
    yield PLE_CONV, [channels, cols, taps, dilation], [gt, nz, w, res], out
    # Pooled keys: dim 128, ratio 4, 30 columns (7 blocks).
    dim, ratio, cols = 128, 4, 30
    keys = [random_bf16(rng, 30) for _ in range(dim * cols)]
    pooled = []
    for b in range(cols // ratio):
        for d in range(dim):
            s = sum((bf16_value(keys[(b * ratio + i) * dim + d]) for i in range(ratio)), Fraction(0))
            pooled.append(round_fraction(f32_of(s / ratio), 8))
    yield QSA_POOL, [dim, ratio, cols], [keys], pooled
    # Selection: 4 heads, ratio 4, budget 5 blocks, 64 columns (16 blocks); ties and negative scores.
    heads, ratio, budget, dim, cols = 4, 4, 5, 128, 64
    blocks = cols // ratio
    sc = []
    for t in range(cols):
        for h in range(heads):
            for b in range(blocks):
                v = rng.choice([0.5, 0.25, -1.0, rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0) * 2.0 ** -30])
                sc.append(f32_bits(v))
    words = (blocks + 31) // 32
    divisor = f32_value(f32_bits(math.sqrt(dim)))
    expected = []
    for t in range(cols):
        n = (t + 1) // ratio
        chosen = set(range(n))
        if n > budget:
            vals = []
            for b in range(n):
                s = sum((max(f32_value(sc[(t * heads + h) * blocks + b]), Fraction(0)) for h in range(heads)), Fraction(0))
                vals.append(f32_of(f32_of(s) / divisor))
            chosen = set(sorted(range(n), key=lambda b: (-vals[b], b))[:budget])
        row = [0] * words
        for b in chosen:
            row[b // 32] |= 1 << (b % 32)
        expected += row
    yield QSA_SELECT, [heads, ratio, budget, dim, cols], [sc], expected
    # Attention with a selection: ratio 4, 28 columns, random selected blocks per query.
    heads, kv_heads, cols, ratio = 4, 2, 28, 4
    q = [random_bf16(rng, 2) for _ in range(256 * heads * cols)]
    k = [random_bf16(rng, 2) for _ in range(256 * kv_heads * cols)]
    v = [random_bf16(rng, 3) for _ in range(256 * kv_heads * cols)]
    sel = []
    for t in range(cols):
        bits = 0
        for b in range((t + 1) // ratio):
            if rng.random() < 0.5 or b == 0:
                bits |= 1 << b
        sel.append(bits)
    yield ATTENTION_SELECT, [heads, kv_heads, cols, ratio], [q, k, v, sel], attention(q, k, v, heads, kv_heads, cols, sel, ratio)
    # Router with BF16 weights: 512 experts, top 10 (Flash-Next).
    experts, top_k, cols = 512, 10, 6
    logits = [random_bf16(rng, 2) for _ in range((experts + 1) * cols)]
    ref = router(logits, experts, top_k, cols)
    weights = [(round_fraction(f32_value(b), 8) << 16) for b in ref[:top_k * cols]]
    yield ROUTER_BF16, [experts, top_k, cols], [logits], weights + ref[top_k * cols:]
    # Combine by entry (the offloaded MoE's outputs at column t k + slot), BF16 weights, no residual add.
    width, cols, top_k = 64, 7, 10
    expert_out = [random_bf16(rng, 6) for _ in range(width * cols * top_k)]
    ids, weights = [], []
    for t in range(cols):
        ids += rng.sample(range(512), top_k)
        weights += [f32_bits(rng.uniform(0.0, 0.3)) & 0xFFFF0000 for _ in range(top_k)]
    shared_out = [random_bf16(rng, 6) for _ in range(width * cols)]
    shared_logits = [random_bf16(rng, 2) for _ in range(cols)]
    out = []
    for t in range(cols):
        slots = sorted(range(top_k), key=lambda s: ids[t * top_k + s])
        gate = crbf(cr.SIGMOID_BF16, bf16_value(shared_logits[t]))
        for h in range(width):
            s = Fraction(0)
            for slot in slots:
                at = t * top_k + slot
                s = bf16_of(s + bf16_of(bf16_value(expert_out[at * width + h]) * f32_value(weights[at])))
            out.append(round_fraction(s + bf16_of(bf16_value(shared_out[t * width + h]) * gate), 8))
    yield MOE_COMBINE_ENTRIES, [width, cols, top_k], [expert_out, weights, ids, shared_out, shared_logits], out


def rope(positions, x, heads, cols, axes):
    pairs = 32
    inv = []
    for i in range(pairs):
        p = cr.round_binary(cr.Decimal(10000000) ** (cr.Decimal(2 * i) / 64), 24)
        inv.append(f32_of(1 / f32_value(p)))
    out = list(x)
    for t in range(cols):
        for i in range(pairs):
            axis = i % 3 if axes == 3 else 0
            angle = f32_of(positions[axis * cols + t] * inv[i])
            c = bf16_of(cr32(cr.COS, angle))
            s = bf16_of(cr32(cr.SIN, angle))
            c_bits = round_fraction(c, 8)
            s_bits = round_fraction(s, 8)
            for h in range(heads):
                base = (t * heads + h) * 256
                xa, xb = x[base + i], x[base + i + pairs]
                a, b = bf16_value(xa), bf16_value(xb)
                ac = signed_round(a * c, 8, (sign_bit(xa, 8) ^ sign_bit(c_bits, 8)) == 1)
                bs = signed_round(b * s, 8, (sign_bit(xb, 8) ^ sign_bit(s_bits, 8)) == 1)
                bc = signed_round(b * c, 8, (sign_bit(xb, 8) ^ sign_bit(c_bits, 8)) == 1)
                as_ = signed_round(a * s, 8, (sign_bit(xa, 8) ^ sign_bit(s_bits, 8)) == 1)
                d1 = float_of(ac, 8) - float_of(bs, 8)
                d2 = float_of(bc, 8) + float_of(as_, 8)
                out[base + i] = signed_round(bf16_value(ac) - bf16_value(bs), 8, math.copysign(1.0, d1) < 0)
                out[base + i + pairs] = signed_round(bf16_value(bc) + bf16_value(as_), 8, math.copysign(1.0, d2) < 0)
    return out


def router(logits, experts, top_k, cols):
    from decimal import Decimal
    weights, ids = [], []
    for t in range(cols):
        ls = [bf16_value(logits[t * (experts + 1) + e]) for e in range(experts)]
        m = max(ls)
        es = [(Decimal(l.numerator) / Decimal(l.denominator) - Decimal(m.numerator) / Decimal(m.denominator)).exp()
              for l in ls]
        total = sum(es, Decimal(0))
        p = [f32_value(cr.round_binary(e / total, 24)) for e in es]
        order = sorted(range(experts), key=lambda e: (-p[e], e))[:top_k]
        denominator = f32_of(sum((p[e] for e in order), Fraction(0)))
        weights += [round_fraction(p[e] / denominator, 24) for e in order]
        ids += order
    return weights + ids


def moe_combine(expert_out, weights, ids, slot_column, shared_out, shared_logits, residual, width, cols, top_k):
    out = []
    for t in range(cols):
        slots = sorted(range(top_k), key=lambda s: ids[t * top_k + s])
        gate = crbf(cr.SIGMOID_BF16, bf16_value(shared_logits[t * 3 + 2]))
        for h in range(width):
            s = Fraction(0)
            for slot in slots:
                at = t * top_k + slot
                y = bf16_value(expert_out[slot_column[at] * width + h])
                s = bf16_of(s + bf16_of(f32_of(y * f32_value(weights[at]))))
            s = bf16_of(s + bf16_of(bf16_value(shared_out[t * width + h]) * gate))
            out.append(round_fraction(bf16_value(residual[t * width + h]) + s, 8))
    return out


def delta_rule(q, k, v, g, beta, hk, hv, cols):
    out = [0] * (128 * hv * cols)
    for head in range(hv):
        kh = head // (hv // hk)
        S = [[0] * 128 for _ in range(128)]  # S[i][j]
        for t in range(cols):
            o = (t * hv + head) * 128
            out[o:o + 128] = delta_rule_column(S, q[(t * hk + kh) * 128:(t * hk + kh + 1) * 128],
                                               k[(t * hk + kh) * 128:(t * hk + kh + 1) * 128], v[o:o + 128],
                                               g[t * hv + head], beta[t * hv + head])
    return out


def attention(q, k, v, heads, kv_heads, cols, select=None, ratio=0):
    group = heads // kv_heads
    out = [0] * (256 * heads * cols)
    for t in range(cols):
        if select is None:
            keys = list(range(t + 1))
        else:
            tail = (t + 1) // ratio * ratio
            keys = [j for j in range(t + 1) if j >= tail or (select[t] >> (j // ratio)) & 1]
        for h in range(heads):
            kvh = h // group
            qr = [bf16_value(q[(t * heads + h) * 256 + d]) for d in range(256)]
            scores = []
            for j in keys:
                kr = k[(j * kv_heads + kvh) * 256:(j * kv_heads + kvh + 1) * 256]
                scores.append(sum((a * bf16_value(b) for a, b in zip(qr, kr)), Fraction(0)) / 16)
            m = max(scores)
            ps = [f32_value(cr.evaluate(cr.EXP, float(f32_of(s - m)))) for s in scores]
            den = sum(ps, Fraction(0))
            for c in range(256):
                num = sum((p * bf16_value(v[(j * kv_heads + kvh) * 256 + c]) for j, p in zip(keys, ps)), Fraction(0))
                out[(t * heads + h) * 256 + c] = round_fraction(num / den, 8)
    return out


# ----------------------------------------------------------------- the design reference's quantizers
# (lossless_quantize.h; docs/maintainer/lossless-route.md §3b): the reproducible route's rules, each FP32
# operation rounded once, E4M3 and E2M1 rounded to nearest even and saturated (cvt.rn.satfinite).


def round_half_even(q: Fraction) -> int:
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n % 2 == 1):
        n += 1
    return n


def e4m3_magnitude(a: Fraction) -> Fraction:
    """The E4M3FN value of a non-negative a: subnormals in steps of 2^-9, three mantissa bits, 448 at most."""
    if a >= 448:
        return Fraction(448)
    if a < Fraction(1, 64):
        return Fraction(round_half_even(a * 512), 512)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    while Fraction(2) ** (e + 1) <= a:
        e += 1
    step = Fraction(2) ** (e - 3)
    return min(Fraction(448), round_half_even(a / step) * step)


def e2m1_magnitude(a: Fraction) -> Fraction:
    """The E2M1 value of a non-negative a ({0, .5, 1, 1.5, 2, 3, 4, 6}; ties to the even code, 6 at most)."""
    for limit, value, inclusive in ((Fraction(1, 4), 0, True), (Fraction(3, 4), Fraction(1, 2), False),
                                    (Fraction(5, 4), 1, True), (Fraction(7, 4), Fraction(3, 2), False),
                                    (Fraction(5, 2), 2, True), (Fraction(7, 2), 3, False), (Fraction(5), 4, True)):
        if a < limit or (inclusive and a == limit):
            return Fraction(value)
    return Fraction(6)


def bf16_signed(value: Fraction, negative: bool) -> int:
    """BF16 bits of a value exact in BF16, -0 when negative and zero."""
    bits = round_fraction(abs(value), 8)
    return bits | (0x8000 if negative else 0)


def nvfp4_divisor_block(xs: list[int], divisor: Fraction) -> list[int]:
    """quantize_nvfp4_k16 (the divisor form) on 16 BF16 values: c e4m3(w) each."""
    vs = [bf16_value(x) for x in xs]
    amax = max(abs(v) for v in vs)
    w = e4m3_magnitude(f32_of(f32_of(divisor * amax) / 6))
    if w == 0:
        return [0] * 16
    out = []
    for x, v in zip(xs, vs):
        q = f32_of(f32_of(v * divisor) / w)
        out.append(bf16_signed(e2m1_magnitude(abs(q)) * w, (x >> 15) != 0))
    return out


def nvfp4_scaled_block(xs: list[int], g: Fraction) -> list[int]:
    """quantize_a4_codes (ModelOpt's rule) on 16 BF16 values with the input scale g: c e4m3(s) each (a code
    whose magnitude rounds to zero gives +0: the doubled-value form carries no sign for zero)."""
    vs = [bf16_value(x) for x in xs]
    amax = max(abs(v) for v in vs)
    s = e4m3_magnitude(f32_of(amax / f32_of(6 * g)))
    if s == 0:
        return [0] * 16
    d = f32_of(s * g)
    out = []
    for v in vs:
        c = e2m1_magnitude(abs(f32_of(v / d)))
        out.append(bf16_signed(c * s, v < 0 and c != 0))
    return out


def fp8_token_column(xs: list[int]) -> tuple[list[int], int]:
    """fp8_a8's per-token rule: the E4M3 values (BF16 bits) and the scale's FP32 bits."""
    vs = [bf16_value(x) for x in xs]
    amax = max(abs(v) for v in vs)
    s = f32_of(amax / 448) if amax > 0 else Fraction(0)
    inv = f32_of(1 / s) if s > 0 else Fraction(0)
    out = []
    for x, v in zip(xs, vs):
        scaled = f32_of(v * inv)
        out.append(bf16_signed(e4m3_magnitude(abs(scaled)), (x >> 15) != 0))
    return out, round_fraction(s, 24)


def fp8_static_values(xs: list[int], s: Fraction) -> list[int]:
    """fp8_a8's static rule with the input scale s: the E4M3 values (BF16 bits) of fl32(v fl32(1 / s))."""
    inv = f32_of(1 / s)
    out = []
    for x in xs:
        scaled = f32_of(bf16_value(x) * inv)
        out.append(bf16_signed(e4m3_magnitude(abs(scaled)), (x >> 15) != 0))
    return out


def fp16_round_bits(x: int) -> int:
    """BF16 -> FP16 (round to nearest even, subnormals, infinity from 65520) -> BF16 bits."""
    v = bf16_value(x)
    negative = (x >> 15) != 0
    a = abs(v)
    if a >= 65520:
        return 0xFF80 if negative else 0x7F80
    if a < Fraction(1, 2 ** 14):
        r = Fraction(round_half_even(a * 2 ** 24), 2 ** 24)
    else:
        e = a.numerator.bit_length() - a.denominator.bit_length()
        if Fraction(2) ** e > a:
            e -= 1
        while Fraction(2) ** (e + 1) <= a:
            e += 1
        step = Fraction(2) ** (e - 10)
        r = round_half_even(a / step) * step
    return bf16_signed(r, negative)


def quantize_cases(rng: random.Random):
    # The NVFP4 divisor rule: blocks of mixed magnitudes, a zero block, a block whose scale rounds to zero,
    # tiny negatives, a saturating block (amax D / 6 above 448).
    k, cols = 256, 24
    xs = [random_bf16(rng, 10) for _ in range(k * cols)]
    for i in range(16):
        xs[i] = 0
        xs[16 + i] = f32_bits(rng.gauss(0.0, 1.0) * 2.0 ** -40) >> 16
        xs[32 + i] = f32_bits(-abs(rng.gauss(0.0, 1.0)) * 2.0 ** -12) >> 16 if i % 2 else xs[32 + i]
        xs[48 + i] = f32_bits(rng.gauss(0.0, 1.0) * 2.0 ** 20) >> 16
    for divisor_f in (1.0, 0.0371, 2741.5):
        divisor = f32_value(f32_bits(divisor_f))
        expected = []
        for b in range(0, k * cols, 16):
            expected += nvfp4_divisor_block(xs[b:b + 16], divisor)
        yield QUANTIZE_NVFP4, [k, cols, f32_bits(divisor_f)], [xs], expected
    # ModelOpt's rule, each output column with its own scale and source column (some -1: zero columns).
    k, src, cols = 512, 20, 64
    xs = [random_bf16(rng, 8) for _ in range(k * src)]
    for i in range(16):
        xs[i] = 0
    in_columns = [(c * 7) % src if c % 9 != 4 else 0xFFFFFFFF for c in range(cols)]
    scales_f = [abs(rng.gauss(0.0, 0.02)) + 1e-6 for _ in range(cols)]
    scales = [f32_bits(s) for s in scales_f]
    expected = []
    for c in range(cols):
        source = in_columns[c]
        if source == 0xFFFFFFFF:
            expected += [0] * k
            continue
        col = xs[source * k:(source + 1) * k]
        g = f32_value(scales[c])
        for b in range(0, k, 16):
            expected += nvfp4_scaled_block(col[b:b + 16], g)
    yield QUANTIZE_NVFP4_SCALED, [k, src, cols], [xs, in_columns, scales], expected
    # The per-token E4M3 rule: K 5120, columns of different spreads and a zero column.
    k, cols = 5120, 6
    xs = [random_bf16(rng, 6 + 6 * (i // 5120)) for i in range(k * cols)]
    for i in range(k):
        xs[3 * k + i] = 0
    values, scale_bits = [], []
    for c in range(cols):
        out, s = fp8_token_column(xs[c * k:(c + 1) * k])
        values += out
        scale_bits.append(s)
    yield QUANTIZE_FP8_TOKEN, [k, cols], [xs], values + scale_bits
    # FP16 storage of BF16 values: normal, subnormal, below the subnormals and overflowing magnitudes.
    n = 20000
    xs = [random_bf16(rng, 60) for _ in range(n)]
    yield ROUND_FP16, [n], [xs], [fp16_round_bits(x) for x in xs]
    # The static E4M3 rule (Qwen3.5-122B-A10B's FP8 sites): K 3072, magnitudes from subnormal to saturating
    # (above 448 s), signed zeros, with a model-like scale and one whose reciprocal is inexact.
    k, cols = 3072, 5
    xs = [random_bf16(rng, 14) for _ in range(k * cols)]
    xs[7] = 0x8000
    for scale_f in (0.0217285, 3.0):
        s = f32_value(f32_bits(scale_f))
        yield QUANTIZE_FP8_STATIC, [k, cols, f32_bits(scale_f)], [xs], fp8_static_values(xs, s) + [f32_bits(scale_f)] * cols


# ----------------------------------------------------------------- the design reference's KV codecs
# (lossless_codec_attention, include/infernix/ops/lossless_ops.h; docs/maintainer/lossless-route.md §3b):
# attention over each KV format's stored codes, every row decoded exactly (its codes times its scales,
# src/ops/kv_cache/*_codec.cuh) and taken back from the rotated basis exactly (x = R y, R = H_256 / 16,
# Sylvester order), the scores, exponentials and sums as attention()'s. The VQ formats' exact window rows
# (rotated INT8-G64, kv_window.cuh) serve key j of query t when j < sinks or j >= t - recent.

KV_INT8, KV_FP8, KV_NVFP4, KV_Q4, KV_VQ2 = range(5)
# u32 entries per 256-element row: code bytes (VQ2: 16-bit words) and scales (FP16 bits; NVFP4: E4M3 bytes).
CODE_UNITS = {KV_INT8: 256, KV_FP8: 256, KV_NVFP4: 128, KV_Q4: 128, KV_VQ2: 32}
SCALE_UNITS = {KV_INT8: 4, KV_FP8: 1, KV_NVFP4: 16, KV_Q4: 1, KV_VQ2: 1}
Q4_LEVELS = (4, 12, 21, 30, 40, 52, 66, 87)
E2M1_VALUES = tuple(Fraction(v) for v in ("0", "1/2", "1", "3/2", "2", "3", "4", "6"))
_VQ2_CODEBOOK: list[int] = []


def vq2_codebook() -> list[int]:
    if not _VQ2_CODEBOOK:
        inc = Path(__file__).resolve().parents[2] / "src" / "ops" / "kv_cache" / "vq2_codebook.inc"
        lines = [line for line in inc.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("//")]
        _VQ2_CODEBOOK.extend(int(t) for t in " ".join(lines).split(",") if t.strip())
        assert len(_VQ2_CODEBOOK) == 512 * 8
    return _VQ2_CODEBOOK


def fp16_value(bits: int) -> Fraction:
    sign = -1 if bits & 0x8000 else 1
    e, m = (bits >> 10) & 0x1F, bits & 0x3FF
    assert e != 31
    return sign * (Fraction(m, 1 << 24) if e == 0 else Fraction(1024 + m) * Fraction(2) ** (e - 25))


def e4m3_value(bits: int) -> Fraction:
    assert bits & 0x7F != 0x7F
    sign = -1 if bits & 0x80 else 1
    e, m = (bits >> 3) & 0xF, bits & 7
    return sign * (Fraction(m, 512) if e == 0 else Fraction(8 + m) * Fraction(2) ** (e - 10))


def signed_byte(b: int) -> int:
    return b - 256 if b >= 128 else b


def plane_row(codec: int, codes: list[int], scales: list[int]) -> list[Fraction]:
    """A stored row's 256 values (rotated basis where the format rotates), exactly."""
    row = []
    for d in range(256):
        if codec == KV_INT8:
            row.append(signed_byte(codes[d]) * fp16_value(scales[d // 64]))
        elif codec == KV_FP8:
            row.append(e4m3_value(codes[d]) * fp16_value(scales[0]))
        elif codec in (KV_NVFP4, KV_Q4):
            n = (codes[d // 2] >> (4 * (d % 2))) & 0xF
            if codec == KV_NVFP4:
                row.append((-1 if n & 8 else 1) * E2M1_VALUES[n & 7] * e4m3_value(scales[d // 16]))
            else:
                level = Q4_LEVELS[n ^ 8] if n >= 8 else -Q4_LEVELS[n ^ 7]
                row.append(level * fp16_value(scales[0]))
        else:
            word = codes[d // 8]
            low7 = (word >> 9) & 0x7F
            sign = low7 | ((bin(low7).count("1") & 1) << 7)
            mag = vq2_codebook()[8 * (word & 0x1FF) + d % 8]
            row.append((-mag if (sign >> (d % 8)) & 1 else mag) * fp16_value(scales[0]))
    return row


def hadamard_exact(y: list[Fraction]) -> list[Fraction]:
    """R y with R = H_256 / 16 (H Sylvester's, H_ij = (-1)^popcount(i & j)), exactly."""
    x = list(y)
    span = 1
    while span < 256:
        for base in range(0, 256, 2 * span):
            for i in range(base, base + span):
                x[i], x[i + span] = x[i] + x[i + span], x[i] - x[i + span]
        span *= 2
    return [v / 16 for v in x]


CODEC_SHIFT = 40   # every decoded K/V value (after R) is an integer multiple of 2^-37
Q_SHIFT = 140      # every BF16 value is an integer multiple of 2^-133
P_SHIFT = 149      # every binary32 value is an integer multiple of 2^-149


def codec_rows(codec: int, rotated: bool, codes: list[int], scales: list[int], rows: int) -> list[list[int]]:
    """Each row's exact values times 2^CODEC_SHIFT, as integers (the sums below stay exact and fast)."""
    cu, su = CODE_UNITS[codec], SCALE_UNITS[codec]
    out = []
    for r in range(rows):
        row = plane_row(codec, codes[r * cu:(r + 1) * cu], scales[r * su:(r + 1) * su])
        scaled = [v * (1 << CODEC_SHIFT) for v in (hadamard_exact(row) if rotated else row)]
        assert all(v.denominator == 1 for v in scaled)
        out.append([int(v) for v in scaled])
    return out


def codec_attention(q, kp, vp, window, heads, kv_heads, cols, sinks, recent, select=None, ratio=0):
    """kp, vp: (codec, rotated, codes, scales), rows [key][kv head]; window: None or the exact rows'
    (k codes, k scales, v codes, v scales), INT8-G64 of the rotated row."""
    rows = cols * kv_heads
    ks, vs = codec_rows(*kp, rows), codec_rows(*vp, rows)
    if window is not None:
        wk = codec_rows(KV_INT8, True, window[0], window[1], rows)
        wv = codec_rows(KV_INT8, True, window[2], window[3], rows)
    group = heads // kv_heads
    out = [0] * (256 * heads * cols)
    score_unit = Fraction(1, 1 << (Q_SHIFT + CODEC_SHIFT + 4))  # s = q k / 16
    for t in range(cols):
        if select is None:
            keys = list(range(t + 1))
        else:
            tail = (t + 1) // ratio * ratio
            keys = [j for j in range(t + 1) if j >= tail or (select[t] >> (j // ratio)) & 1]
        for h in range(heads):
            kvh = h // group
            qr = [int(bf16_value(q[(t * heads + h) * 256 + d]) * (1 << Q_SHIFT)) for d in range(256)]
            krs, vrs = [], []
            for j in keys:
                exact = window is not None and (j < sinks or j >= t - recent)
                krs.append((wk if exact else ks)[j * kv_heads + kvh])
                vrs.append((wv if exact else vs)[j * kv_heads + kvh])
            scores = [sum(a * b for a, b in zip(qr, kr)) for kr in krs]
            m = max(scores)
            ps = [int(f32_value(cr.evaluate(cr.EXP, float(f32_of((s - m) * score_unit)))) * (1 << P_SHIFT))
                  for s in scores]
            den = sum(ps) << CODEC_SHIFT
            for c in range(256):
                num = sum(p * vr[c] for vr, p in zip(vrs, ps))
                out[(t * heads + h) * 256 + c] = round_fraction(Fraction(num, den), 8)
    return out


def random_codec_plane(rng: random.Random, codec: int, rows: int, spread: int) -> tuple[list[int], list[int]]:
    """Codes and scales as the encoders can store them (finite, E4M3 codes not NaN), scales over `spread`
    binades."""
    codes, scales = [], []
    for _ in range(rows):
        if codec in (KV_INT8,):
            codes += [rng.randint(-127, 127) & 0xFF for _ in range(256)]
        elif codec == KV_FP8:
            codes += [b if b & 0x7F != 0x7F else 0x7E for b in (rng.getrandbits(8) for _ in range(256))]
        elif codec in (KV_NVFP4, KV_Q4):
            codes += [rng.getrandbits(8) for _ in range(128)]
        else:
            codes += [rng.getrandbits(16) for _ in range(32)]
        for _ in range(SCALE_UNITS[codec]):
            if codec == KV_NVFP4:
                scales.append(rng.randint(0x08, 0x7E) if rng.random() > 0.05 else 0)
            else:
                v = abs(rng.gauss(0.0, 1.0)) * 2.0 ** rng.randint(-spread, spread // 2) * 0.01
                scales.append(0 if rng.random() < 0.03 else struct.unpack("<H", struct.pack("<e", v))[0])
    return codes, scales


def codec_cases(rng: random.Random):
    # Every KV format's K and V codecs (kv_cache/append: K rotated in all, V in NVFP4, K8V4 and the VQ formats),
    # the VQ formats with their exact window (small bounds so both reads occur), a QSA selection, scales over a
    # wide range of binades (the exact fallback).
    formats = [("int8", KV_INT8, True, KV_INT8, False, False), ("fp8", KV_FP8, True, KV_FP8, False, False),
               ("nvfp4", KV_NVFP4, True, KV_NVFP4, True, False), ("k8v4", KV_FP8, True, KV_NVFP4, True, False),
               ("vq2", KV_VQ2, True, KV_VQ2, True, True), ("k4v2", KV_Q4, True, KV_VQ2, True, True)]
    heads, kv_heads, cols, sinks, recent = 4, 2, 26, 3, 9
    for name, kc, kr, vc, vr, has_window in formats:
        for spread in (6, 40):
            q = [random_bf16(rng, 6) for _ in range(256 * heads * cols)]
            kcodes, kscales = random_codec_plane(rng, kc, cols * kv_heads, spread)
            vcodes, vscales = random_codec_plane(rng, vc, cols * kv_heads, spread)
            window = None
            if has_window:
                window = (*random_codec_plane(rng, KV_INT8, cols * kv_heads, spread),
                          *random_codec_plane(rng, KV_INT8, cols * kv_heads, spread))
            inputs = [q, kcodes, kscales, vcodes, vscales] + (list(window) if window else [])
            expected = codec_attention(q, (kc, kr, kcodes, kscales), (vc, vr, vcodes, vscales), window, heads,
                                       kv_heads, cols, sinks, recent)
            yield (CODEC_ATTENTION, [kc, int(kr), vc, int(vr), int(has_window), heads, kv_heads, cols, sinks, recent, 0],
                   inputs, expected)
    # QSA: selected blocks of 4 and the tail, VQ2 with its window.
    ratio, cols = 4, 26
    q = [random_bf16(rng, 6) for _ in range(256 * heads * cols)]
    kcodes, kscales = random_codec_plane(rng, KV_VQ2, cols * kv_heads, 6)
    vcodes, vscales = random_codec_plane(rng, KV_VQ2, cols * kv_heads, 6)
    window = (*random_codec_plane(rng, KV_INT8, cols * kv_heads, 6), *random_codec_plane(rng, KV_INT8, cols * kv_heads, 6))
    # Production selects at least one block once one is complete, so no query is left without keys.
    select = [(rng.getrandbits((t + 1) // ratio) | 1 << ((t + 1) // ratio - 1)) if t + 1 >= ratio else 0
              for t in range(cols)]
    expected = codec_attention(q, (KV_VQ2, True, kcodes, kscales), (KV_VQ2, True, vcodes, vscales), window, heads,
                               kv_heads, cols, sinks, recent, select, ratio)
    yield (CODEC_ATTENTION, [KV_VQ2, 1, KV_VQ2, 1, 1, heads, kv_heads, cols, sinks, recent, ratio],
           [q, kcodes, kscales, vcodes, vscales, *window, select], expected)


def write_cases(path: Path) -> None:
    rng = random.Random(20261021)
    count = 0
    with path.open("wb") as f:
        for op, dims, inputs, expected in itertools.chain(cases(rng), quantize_cases(rng), codec_cases(rng)):
            write_u32(f, [op, len(dims)] + dims + [len(inputs)] + [len(v) for v in inputs] + [len(expected)])
            for v in inputs:
                write_u32(f, v)
            write_u32(f, expected)
            count += 1
    print(f"{count} cases", flush=True)


def main() -> int:
    """lossless_ops_oracle.py <family executable> <test name> [cache directory]: with a directory, the cases are
    written there once per version of this file and lossless_cr_oracle.py and reused afterwards (they are
    deterministic)."""
    command = sys.argv[1:3]  # the family executable and the test name (tests/ops/op_family.h)
    sources = [Path(__file__), Path(cr.__file__)]
    path, tmp = cr.cached_file(sys.argv[3] if len(sys.argv) > 3 else None, "ops_cases", sources, write_cases)
    try:
        return subprocess.run([*command, str(path)], check=False).returncode
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
