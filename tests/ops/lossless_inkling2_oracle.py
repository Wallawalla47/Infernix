"""Exact oracle for the lossless route's Inkling 2 operations (include/infernix/ops/lossless_inkling2.h).

Writes test cases and runs the CUDA test on them: python lossless_inkling2_oracle.py <family executable>
lossless_inkling2 [cache directory]. Every expected value follows docs/maintainer/lossless-route.md §2b step by step:
binary32 operations on binary32 operands are Python's binary64 operation rounded to binary32 (exact for +, -, *, /
and sqrt: binary64 has more than 2p + 2 bits), sums exact (fractions) and rounded once, transcendental functions
correctly rounded (lossless_cr_oracle.py's 60-digit evaluation), a softmax element its exact quotient rounded once
(60 digits), every intermediate rounded to the dtype the specification gives it. The quantizers follow
src/ops/common/canonical_math.h's rules, transcribed here in binary32 steps. Standard library only.
"""
from __future__ import annotations

import math
import random
import struct
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossless_cr_oracle as cr  # noqa: E402

(RMSNORM, MIX, MESSAGE_GATE, CLIP, CONV, DYNAMIC_CONV, SIGMOID_GATE, ADD, PROBS, ROUTER, FOUR_OVER_SIX, MXFP8,
 SWIGLU, COMBINE, SOFTCAP, SQRT, RING_LOAD, RING_STORE, KEYS_LOAD, KEYS_STORE, PLANES) = range(21)

F32_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]


# ------------------------------------------------------------------------------------------- formats

def f32(x: float) -> float:
    """x rounded to binary32 (round to nearest even, subnormals, overflow to infinity, signed zeros)."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_of_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b))[0]


def bf_of_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b << 16))[0]


def round_bits(v: Fraction, prec: int, negative_zero: bool = False) -> int:
    """Bits of the exact v rounded to nearest even with binary32's exponent range (prec 24) or BF16's (prec 8, low
    16 bits). A zero result keeps v's sign (or negative_zero's for v == 0)."""
    total = 16 if prec == 8 else 32
    sign = 1 if v < 0 or (v == 0 and negative_zero) else 0
    a = -v if v < 0 else v
    if a == 0:
        return sign << (total - 1)
    num, den = a.numerator, a.denominator
    e = num.bit_length() - den.bit_length()
    if (num << max(0, -e)) < (den << max(0, e)):
        e -= 1
    e = max(e, -126)
    shift = e - prec + 1
    divisor = den << max(0, shift)
    n, rem = divmod(num << max(0, -shift), divisor)
    if 2 * rem > divisor or (2 * rem == divisor and n % 2 == 1):
        n += 1
    if n == 2 ** prec:
        n //= 2
        e += 1
    if e > 127:
        field, mantissa = 255, 0
    elif n < 2 ** (prec - 1):
        field, mantissa = 0, n
    else:
        field, mantissa = e + 127, n - 2 ** (prec - 1)
    return (sign << (total - 1)) | (field << (prec - 1)) | mantissa


def bf(x: float | Fraction) -> float:
    """x (a binary64 value taken exactly, or a fraction) rounded to BF16."""
    if isinstance(x, float):
        if math.isnan(x):
            return x
        if math.isinf(x):
            return x
        return bf_of_bits(round_bits(Fraction(x), 8, math.copysign(1.0, x) < 0))
    return bf_of_bits(round_bits(x, 8))


def bf_bits(x: float) -> int:
    return f32_bits(x) >> 16


def f32_exact(v: Fraction) -> float:
    return f32_of_bits(round_bits(v, 24))


def cr32(func: int, x: float) -> float:
    return f32_of_bits(cr.evaluate(func, x))


def random_bf16(rng: random.Random, spread: int = 6) -> float:
    v = rng.gauss(0.0, 1.0) * 2.0 ** rng.randint(-spread, spread)
    if rng.random() < 0.02:
        v = 0.0
    return bf(v)


def random_f32(rng: random.Random, spread: int = 6) -> float:
    return f32(rng.gauss(0.0, 1.0) * 2.0 ** rng.randint(-spread, spread))


def sqrt_f32(s: Fraction) -> float:
    """sqrt(s) correctly rounded to binary32, s >= 0 exact: a 60-digit estimate, then the exact midpoint tests."""
    if s == 0:
        return 0.0
    est = (Decimal(s.numerator) / Decimal(s.denominator)).sqrt()
    f = f32_of_bits(cr.round_binary(est, 24))
    # Move f until s lies between the squares of its two midpoints.
    while True:
        up32 = f32_of_bits(f32_bits(f) + 1)
        down32 = f32_of_bits(f32_bits(f) - 1) if f32_bits(f) > 0 else 0.0
        hi = (Fraction(f) + Fraction(up32)) / 2
        lo = (Fraction(f) + Fraction(down32)) / 2
        if s > hi * hi:
            f = up32
        elif s < lo * lo:
            f = down32
        elif s == hi * hi:
            return f if f32_bits(f) % 2 == 0 else up32
        elif s == lo * lo:
            return f if f32_bits(f) % 2 == 0 else down32
        else:
            return f


def softmax_f32(logits: list[float]) -> list[float]:
    """Each element's exact quotient exp(l_j) / sum_k exp(l_k) rounded to binary32 (60 digits)."""
    m = max(logits)
    e = [(Decimal(x) - Decimal(m)).exp() for x in logits]
    s = sum(e, Decimal(0))
    return [f32_of_bits(cr.round_binary(v / s, 24)) for v in e]


def mean_square_f32(row: list[float]) -> float:
    s = sum((Fraction(f32(x * x)) for x in row), Fraction(0))
    return f32_exact(s / len(row))


def rms_factor(row: list[float], eps: float) -> float:
    return cr32(cr.RSQRT, f32(mean_square_f32(row) + f32(eps)))


# ------------------------------------------------------------------------------------------- cases

def u16(values: list[float]) -> list[int]:
    return [bf_bits(v) for v in values]


def u32(values: list[float]) -> list[int]:
    return [f32_bits(v) for v in values]


def rmsnorm_cases(rng):
    for width, rows, x_fp32, weighted, out_fp32, out_ld in ((128, 5, False, False, False, 128),
                                                           (2304, 3, True, True, False, 4608),
                                                           (2304, 2, True, False, True, 2304),
                                                           (2304, 2, False, True, True, 2304),
                                                           (64, 4, True, True, False, 64)):
        x = [(random_f32 if x_fp32 else random_bf16)(rng) for _ in range(width * rows)]
        w = [random_bf16(rng, 1) for _ in range(width)] if weighted else []
        out = []
        for r in range(rows):
            row = x[r * width:(r + 1) * width]
            f = rms_factor(row, 1e-6)
            for i, v in enumerate(row):
                y = f32(v * f)
                if weighted:
                    y = f32(w[i] * y)
                out.append(y if out_fp32 else bf(y))
        yield (RMSNORM, [width, rows, int(x_fp32), int(weighted), int(out_fp32), out_ld],
               [u32(x) if x_fp32 else u16(x), u16(w)], u32(out) if out_fp32 else u16(out))


def mix_case(H, T, kinds, queries, sources, q):
    scale = f32(H ** -0.5)
    outs = [[] for _ in range(queries)]
    for t in range(T):
        n = []
        for s in sources:
            row = s[t * H:(t + 1) * H]
            f = rms_factor(row, 1e-6)
            n.append([f32(v * f) for v in row])
        for qi in range(queries):
            qq = q[qi * H:(qi + 1) * H]
            logits = [f32(f32_exact(sum((Fraction(a) * Fraction(b) for a, b in zip(qq, nj)), Fraction(0))) * scale)
                      for nj in n]
            p = softmax_f32(logits)
            for i in range(H):
                outs[qi].append(f32_exact(sum((Fraction(p[j]) * Fraction(n[j][i]) for j in range(len(n))),
                                              Fraction(0))))
    inputs = [u32(s) if fp32 else u16(s) for s, fp32 in zip(sources, kinds)] + [u32(q)]
    expected = []
    for o in outs:
        expected += u32(o)
    return MIX, [H, T, len(kinds), queries] + [int(k) for k in kinds], inputs, expected


def mix_cases(rng):
    for H, T, kinds, queries in ((2304, 2, [False, False, False, True], 1), (2304, 1, [False, False, True], 3),
                                 (256, 3, [False, False, False, False, False, False], 4), (2304, 1, [False, True], 4)):
        sources = [[(random_f32 if fp32 else random_bf16)(rng, 3) for _ in range(H * T)] for fp32 in kinds]
        q = [f32(rng.gauss(0.0, 1.0) * 0.5) for _ in range(H * queries)]
        yield mix_case(H, T, kinds, queries, sources, q)
    # A softmax whose largest logit is a tiny nonzero value (about 2^-45, a full 24-bit significand) beside logits
    # of -8..-100: l_j - m is not exact in binary64, so the kernel takes its exponents from 0 there. Source j is one
    # at element j and zero elsewhere, so logit j is set by query element j alone (n_j[j] * scale is about 1).
    H, wanted = 256, [2.0 ** -45 * 1.3710938, -8.0, -20.5, -41.25, -70.125, -100.0625]
    sources = [[1.0 if i == j else 0.0 for i in range(H)] for j in range(len(wanted))]
    q = [f32(wanted[i]) if i < len(wanted) else 0.0 for i in range(H)]
    yield mix_case(H, 1, [False] * len(wanted), 1, sources, q)


def message_gate_cases(rng):
    n = 3000
    fl = [random_f32(rng, 2) for _ in range(n)]
    gl = [random_f32(rng, 2) for _ in range(n)]
    x = [random_f32(rng) for _ in range(n)]
    m = [random_f32(rng) for _ in range(n)]
    out = []
    for a, b, xv, mv in zip(fl, gl, x, m):
        f, g = cr32(cr.TANH, a), cr32(cr.TANH, b)
        out.append(f32(f32(f32(1.0 - f) * xv) + f32(g * mv)))
    yield MESSAGE_GATE, [n], [u32(fl), u32(gl), u32(x), u32(m)], u32(out)


def clip_cases(rng):
    for D, heads, T, scaled, logscale, threshold in ((128, 3, 4, True, False, 22.6274166),
                                                     (128, 2, 3, False, False, 22.6274166),
                                                     (16, 4, 5, True, True, 8.0), (128, 2, 3, True, True, 22.6274166)):
        x = [random_bf16(rng, 3) for _ in range(D * heads * T)]
        s = [bf(1.0 + 0.3 * rng.gauss(0.0, 1.0)) for _ in range(D * heads)] if scaled else []
        positions = [rng.choice([0, 5, 8191, 8192, 9000, 40000, 1048575]) for _ in range(T)]
        t32 = f32(threshold)
        out = []
        for t in range(T):
            tau = 1.0
            if logscale:
                q = f32(f32(positions[t] + 1) / 8192.0)
                q = max(q, 1.0)
                tau = f32(1.0 + f32(f32(0.1) * cr32(cr.LOG, q)))
            for h in range(heads):
                row = x[(t * heads + h) * D:(t * heads + h + 1) * D]
                y = [f32(v * s[h * D + i]) if scaled else v for i, v in enumerate(row)]
                nu = sqrt_f32(sum((Fraction(v) * Fraction(v) for v in y), Fraction(0)))
                c = f32(f32(1.0 / f32(nu + f32(1e-6))) * t32)
                c = 1.0 if c > 1.0 else c
                for v in y:
                    o = bf(f32(v * c))
                    out.append(bf(f32(o * tau)) if logscale else o)
        inputs = [u16(x), u16(s), positions]
        yield CLIP, [D, heads, T, int(scaled), int(logscale), f32_bits(t32), f32_bits(f32(0.1)), 8192], inputs, u16(out)


def conv_cases(rng):
    for C, T, k in ((64, 7, 4), (2304, 3, 4)):
        x = [random_bf16(rng, 2) for _ in range(C * T)]
        taps = [random_bf16(rng, 1) for _ in range(k * C)]
        div = f32(math.sqrt(k + 1))
        out = []
        for t in range(T):
            for c in range(C):
                s = sum((Fraction(taps[d * C + c]) * Fraction(x[(t - d) * C + c]) for d in range(k) if d <= t), Fraction(0))
                out.append(bf(f32(f32(f32_exact(s) + x[t * C + c]) / div)))
        yield CONV, [C, T, k], [u16(x), u16(taps)], u16(out)


def dynamic_conv_cases(rng):
    for C, heads, T, k in ((1024, 8, 6, 4), (128, 2, 3, 4)):
        x = [random_bf16(rng, 2) for _ in range(C * T)]
        dyn = [random_bf16(rng, 1) for _ in range(heads * k * T)]
        taps = [random_bf16(rng, 1) for _ in range(k * C)]
        div = f32(math.sqrt(2 * k + 1))
        out = []
        for t in range(T):
            for c in range(C):
                h = c // (C // heads)
                s = Fraction(0)
                for d in range(k):
                    w = f32(taps[d * C + c] + dyn[t * heads * k + h * k + d])
                    v = x[(t - d) * C + c] if d <= t else 0.0
                    s += Fraction(f32(v * w))
                out.append(bf(f32(f32(f32_exact(s) + x[t * C + c]) / div)))
        yield DYNAMIC_CONV, [C, heads, T, k], [u16(x), u16(dyn), u16(taps)], u16(out)


def sigmoid_gate_cases(rng):
    n = 4000
    o = [random_bf16(rng) for _ in range(n)]
    g = [random_bf16(rng, 3) for _ in range(n)]
    out = [bf(f32(a * cr32(cr.SIGMOID, b))) for a, b in zip(o, g)]
    yield SIGMOID_GATE, [n], [u16(o), u16(g)], u16(out)


def add_cases(rng):
    n = 3000
    for a32, b32, o32, via in ((False, True, True, False), (True, False, False, True), (True, True, False, False),
                              (False, False, False, False), (True, True, True, False)):
        a = [(random_f32 if a32 else random_bf16)(rng, 10) for _ in range(n)]
        b = [(random_f32 if b32 else random_bf16)(rng, 10) for _ in range(n)]
        if a32 and b32 and not o32:
            # Sums near BF16 midpoints with a far smaller second term (double rounding would misround them).
            for i in range(0, n, 7):
                a[i] = f32(bf(a[i]) * (1.0 + 2.0 ** -8))
                b[i] = f32(a[i] * 2.0 ** -40)
        out = []
        for x, y in zip(a, b):
            if o32 or via:
                s = f32(x + y)
                out.append(s if o32 else bf(s))
            else:
                out.append(bf(Fraction(x) + Fraction(y)) if x + y != 0 else bf(x + y))
        yield (ADD, [n, int(a32), int(b32), int(o32), int(via)], [u32(a) if a32 else u16(a), u32(b) if b32 else u16(b)],
               u32(out) if o32 else u16(out))


def probs_cases(rng):
    rel = 16
    # keys, ld, query columns T, first query position, window, first head, heads in the call, heads, extent: the small
    # shapes, then the real geometry (24 heads, extent 1,024; a sliding visit's 1,024-key window and a full visit's keys,
    # one and eight query columns), 256 query columns, visible keys past one block with a partial last warp, lo > 0.
    for keys, ld, T, begin, window, head, count, heads, extent in (
            (70, 128, 5, 60, 0, 1, 1, 3, 40), (70, 128, 4, 30, 16, 2, 1, 3, 40), (256, 256, 3, 250, 0, 0, 1, 3, 40),
            (2048, 2048, 1, 2047, 1024, 0, 24, 24, 1024), (2048, 2048, 1, 1500, 0, 5, 3, 24, 1024),
            (1088, 1088, 8, 1000, 1024, 0, 3, 24, 1024), (320, 320, 256, 40, 0, 2, 1, 3, 40),
            (1000, 1100, 2, 700, 300, 1, 2, 3, 40), (513, 600, 1, 512, 0, 0, 1, 3, 40), (257, 300, 1, 256, 0, 0, 3, 3, 40)):
        columns = T * count
        scores = [random_bf16(rng, 2) for _ in range(ld * columns)]
        r = [random_bf16(rng, 1) for _ in range(rel * heads * T)]
        table = [random_bf16(rng, 1) for _ in range(rel * extent)]
        sinks = [random_bf16(rng, 1) for _ in range(heads * T)]
        scale = f32(128 ** -0.5)
        out = []
        for column in range(columns):
            t, head_c = column % T, head + column // T
            P = begin + t
            lo = max(0, P - window + 1) if window > 0 else 0
            hi = min(P, keys - 1)
            logits = []
            for j in range(lo, hi + 1):
                a = bf(Fraction(scores[column * ld + j]) * Fraction(scale))
                d = P - j
                if d < extent:
                    rr = r[(t * heads + head_c) * rel:(t * heads + head_c + 1) * rel]
                    b = bf(sum((Fraction(f32(rc * table[c * extent + d])) for c, rc in enumerate(rr)), Fraction(0)))
                    a = bf(Fraction(a) + Fraction(b))
                logits.append(a)
            p = softmax_f32(logits + [sinks[t * heads + head_c]])[:-1]
            row = [0.0] * ld
            for j, pj in zip(range(lo, hi + 1), p):
                row[j] = bf(pj)
            out += row[:keys] + scores[column * ld + keys:(column + 1) * ld]
        yield (PROBS, [keys, ld, T, begin, window, head, heads, rel, extent, f32_bits(scale), count],
               [u16(scores), u16(r), u16(table), u16(sinks)], u16(out))
    # A row whose maximum is a tiny nonzero logit (2^-45) beside logits of -8..-700: s_j - m is not exact in binary64,
    # so the kernel takes its exponents from 0 there. No relative bias (zero factors), one head, one query column.
    keys, ld, heads, extent, scale = 64, 64, 1, 40, f32(128 ** -0.5)
    logits_wanted = [-(8.0 + 692.0 * j / (keys - 2)) for j in range(keys - 1)] + [2.0 ** -45]
    scores = [bf(Fraction(v) / Fraction(scale)) for v in logits_wanted]
    r, table, sinks = [0.0] * rel, [0.0] * (rel * extent), [bf(-750.0)]
    logits = [bf(Fraction(s) * Fraction(scale)) for s in scores]
    p = softmax_f32(logits + sinks)[:-1]
    yield (PROBS, [keys, ld, 1, keys - 1, 0, 0, heads, rel, extent, f32_bits(scale), 1],
           [u16(scores), u16(r), u16(table), u16(sinks)], u16([bf(pj) for pj in p]))


def router_cases(rng):
    E, k, T = 128, 8, 6
    logits = [random_f32(rng, 2) for _ in range(E * T)]
    bias = [f32(rng.gauss(0.0, 0.1)) for _ in range(E)]
    # Ties: equal keys in one column.
    for e in range(10, 20):
        logits[e] = logits[9]
        bias[e] = bias[9]
    w_out, id_out = [], []
    for t in range(T):
        l = logits[t * E:(t + 1) * E]
        s = [cr32(cr.SIGMOID, v) for v in l]
        keys = [f32(si + bi) for si, bi in zip(s, bias)]
        order = sorted(range(E), key=lambda e: (-keys[e], e))[:k]
        total = f32_exact(sum((Fraction(s[e]) for e in order), Fraction(0)))
        den = f32(total + f32(1e-20))
        for e in order:
            w_out.append(f32(f32(s[e] / den) * 8.0))
            id_out.append(e)
    yield ROUTER, [E, k, T, f32_bits(8.0)], [u32(logits), u32(bias)], u32(w_out) + id_out


# canonical_math.h's rules in binary32 steps.

def e4m3_rn_satfinite(x: float) -> int:
    """E4M3FN word of a nonnegative binary32 (round to nearest even, saturation to 448; NaN 0x7F)."""
    if x != x:
        return 0x7F
    if x >= 448.0:
        return 0x7E
    best, best_word = None, 0
    # Exhaustive over the 127 finite nonnegative words: the nearest, ties to the even word.
    for word in range(0x7F):
        v = e4m3_value(word)
        d = abs(Fraction(v) - Fraction(x))
        if best is None or d < best or (d == best and word % 2 == 0):
            best, best_word = d, word
    return best_word


def e4m3_value(word: int) -> float:
    e, m = (word >> 3) & 0xF, word & 7
    if e == 0:
        return m * 2.0 ** -9
    return (1 + m / 8) * 2.0 ** (e - 7)


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_rn_satfinite(x: float) -> int:
    sign = 8 if math.copysign(1.0, x) < 0 else 0
    a = abs(x)
    if a != a:
        return 7
    best, code = None, 0
    for c, v in enumerate(E2M1):
        d = abs(Fraction(v) - Fraction(min(a, 6.0)))
        if best is None or d < best or (d == best and c % 2 == 0):
            best, code = d, c
    return sign | code


def e2m1_value(code: int) -> float:
    v = E2M1[code & 7]
    return -v if code & 8 else v


def four_over_six(row: list[float]) -> tuple[list[float], float]:
    amax = max(abs(v) for v in row)
    enc = f32(1536.0 / amax) if amax != 0 else math.inf
    enc = enc if enc < F32_MAX else F32_MAX
    dec = f32(1.0 / enc)
    outer = f32(amax * f32(1.0 / 1536.0))
    out = []
    for b in range(len(row) // 16):
        x = row[16 * b:16 * b + 16]
        bmax = max(abs(v) for v in x)
        v6 = f32(bmax * f32(enc * f32(1.0 / 6.0)))
        v4 = f32(bmax * f32(enc * 0.25))
        words = [e4m3_rn_satfinite(min(v6, 448.0)), e4m3_rn_satfinite(min(v4, 448.0))]
        cand = []
        for w in words:
            s = e4m3_value(w)
            inv = f32(1.0 / f32(s * dec)) if f32(s * dec) != 0 else math.inf
            kk = inv if inv < F32_MAX else F32_MAX
            codes, errs = [], []
            for v in x:
                c = e2m1_rn_satfinite(f32(v * kk))
                cv = e2m1_value(c)
                d = f32(f32(f32(cv * s) * outer) - v)
                codes.append(c)
                errs.append(abs(d))
            halves = []
            for half in (errs[:8], errs[8:]):
                e = list(half)
                while len(e) > 1:
                    e = [f32(e[2 * i] + e[2 * i + 1]) for i in range(len(e) // 2)]
                halves.append(e[0])
            cand.append((f32(halves[0] + halves[1]), codes, s))
        err6, codes6, s6 = cand[0]
        err4, codes4, s4 = cand[1]
        codes, s = (codes4, s4) if err4 < err6 else (codes6, s6)
        out += [e2m1_value(c) * s for c in codes]
    return out, outer


def mxfp8(row: list[float]) -> list[float]:
    out = []
    for g in range(len(row) // 32):
        x = row[32 * g:32 * g + 32]
        amax = max(abs(v) for v in x)
        raw = f32(amax * f32(1.0 / 448.0))
        e = ((f32_bits(raw) + 0x7FFFFF) & 0x7F800000) >> 23
        e = 0 if raw <= 2.0 ** -127 else min(e, 254)
        scale = 2.0 ** -127 if e == 0 else 2.0 ** (e - 127)
        inverse = 2.0 ** 127 if e == 0 else 1.0 / scale
        for v in x:
            q = f32(v * inverse)
            q = max(-448.0, min(448.0, q))
            word = e4m3_rn_satfinite(abs(q))
            val = e4m3_value(word)
            val = -val if math.copysign(1.0, q) < 0 else val
            out.append(bf(val * scale))
    return out


def quantize_cases(rng):
    K, T = 256, 5
    x = [random_bf16(rng, 4) for _ in range(K * T)]
    x[K * 2:K * 3] = [0.0] * K  # an all-zero row
    in_columns = [3, 0, 2, 4, 1, 3]
    out, outer = [], []
    for c in in_columns:
        q, o = four_over_six(x[c * K:(c + 1) * K])
        out += q
        outer.append(o)
    yield FOUR_OVER_SIX, [K, T, len(in_columns)], [u16(x), in_columns], u16(out) + u32(outer)
    mx = []
    for c in in_columns:
        mx += mxfp8(x[c * K:(c + 1) * K])
    yield MXFP8, [K, T, len(in_columns)], [u16(x), in_columns], u16(mx)


def swiglu_cases(rng):
    I, C = 96, 5
    gu = [random_bf16(rng, 4) for _ in range(2 * I * C)]
    gamma = [f32(abs(rng.gauss(0.0, 1.0))) for _ in range(C)]
    out = []
    for c in range(C):
        for e in range(I):
            g, u = gu[c * 2 * I + 2 * e], gu[c * 2 * I + 2 * e + 1]
            scg = f32(cr32(cr.TANH, f32(g / 4.0)) * 4.0)
            scu = f32(cr32(cr.TANH, f32(u / 25.0)) * 25.0)
            a = f32(scg * cr32(cr.SIGMOID, g))
            out.append(bf(f32(f32(a * scu) * gamma[c])))
    yield SWIGLU, [I, C, f32_bits(4.0), f32_bits(25.0)], [u16(gu), u32(gamma)], u16(out)


def combine_cases(rng):
    H, T, k = 64, 3, 8
    entries = T * k
    y = [random_bf16(rng, 3) for _ in range(H * entries)]
    perm = list(range(entries))
    rng.shuffle(perm)
    out = []
    for t in range(T):
        for e in range(H):
            v = 0.0
            for j in range(k):
                v = f32(v + y[perm[t * k + j] * H + e])
            out.append(bf(v))
    yield COMBINE, [H, T, k, entries], [u16(y), perm], u16(out)


def softcap_cases(rng):
    n = 4000
    x = [random_f32(rng, 5) for _ in range(n)]
    out = [f32(cr32(cr.TANH, f32(v / 30.0)) * 30.0) for v in x]
    yield SOFTCAP, [n, f32_bits(30.0)], [u32(x)], u32(out)


def sqrt_cases(rng):
    """sqrt_magnitude (wide.cuh) on magnitudes mag 2^unit: exact rounding midpoints m^2 of binary32 neighbours (ties to
    even), one unit above and below them, on grids finer than, equal to and coarser than m^2's, subnormal results,
    and random magnitudes."""
    words, expected = [], []

    def add(mag: int, unit: int) -> None:
        if mag < 0 or mag >= 1 << 640:
            return
        words.extend((mag >> (32 * i)) & 0xFFFFFFFF for i in range(20))
        words.append(unit & 0xFFFFFFFF)
        s = Fraction(mag) * (Fraction(2) ** unit)
        expected.append(f32_bits(sqrt_f32(s)))

    for _ in range(300):
        kind = rng.random()
        if kind < 0.15:
            f = f32_of_bits(rng.randrange(1, 0x00800000))  # a subnormal root
        else:
            f = f32(rng.uniform(1.0, 2.0) * 2.0 ** rng.randint(-60, 60))
        nxt = f32_of_bits(f32_bits(f) + 1)
        m = (Fraction(f) + Fraction(nxt)) / 2
        num, den = m.numerator, m.denominator  # den a power of two
        em = -(den.bit_length() - 1)
        M = num  # m = M 2^em
        for shift in (0, 7, 40):
            unit = 2 * em - shift
            base = (M * M) << shift
            for delta in (0, 1, -1):
                add(base + delta, unit)
        # A coarser grid than m^2's: the nearest multiples below and above.
        for coarse in (1, 3):
            unit = 2 * em + coarse
            add((M * M) >> coarse, unit)
            add(((M * M) >> coarse) + 1, unit)
    for _ in range(300):
        bits = rng.randint(1, 600)
        add(rng.getrandbits(bits) | (1 << (bits - 1)), rng.randint(-297 - bits, 250 - bits))  # 2^-298 .. 2^251
    yield SQRT, [len(expected)], [words], expected


# The cache form's state copies: exact, so their inputs are arbitrary 16-bit words (NaN patterns and -0 included), every
# destination starts as random words and the whole of it is compared (what a copy must leave alone stays).
def words16(rng, n: int) -> list[int]:
    return [rng.getrandbits(16) for _ in range(n)]


def ring_load_cases(rng):
    # Real rings (32 positions, 1024 and 2304 wide), positions before the sequence, the ring's wrap, strided columns.
    for rows, C, first, count, columns, ld in ((32, 1024, -3, 3, 4, 1024), (32, 2304, -1, 3, 5, 2310), (32, 64, 30, 3, 4, 70),
                                               (32, 64, 64, 3, 3, 64), (8, 16, -20, 30, 32, 19)):
        ring, out = words16(rng, rows * C), words16(rng, ld * columns)
        expected = list(out)
        for c in range(count):
            p = first + c
            for e in range(C):
                expected[c * ld + e] = 0 if p < 0 else ring[(p % rows) * C + e]
        yield RING_LOAD, [rows, C, first & 0xFFFFFFFF, count, columns, ld], [ring, out], expected


def ring_store_cases(rng):
    # One column, the wrap, exactly the ring's rows, more than them (only the last rows), strided columns.
    for rows, C, first, T, ld in ((32, 1024, 7, 1, 1024), (32, 2304, 29, 6, 2304), (32, 64, 100, 32, 70), (32, 64, 5, 40, 64),
                                  (8, 16, 0, 8, 19)):
        x, ring = words16(rng, ld * T), words16(rng, rows * C)
        expected = list(ring)
        for t in range(max(0, T - rows), T):
            slot = (first + t) % rows
            for e in range(C):
                expected[slot * C + e] = x[t * ld + e]
        yield RING_STORE, [rows, C, first, T, ld], [x, ring], expected


def key_row(table: list[int], ring_pages: int, heads: int, position: int, head: int) -> int:
    unit, row = position // 64, position % 64
    if table:
        return ((table[unit] * heads + head) * 64 + row) * 128
    return ((head * ring_pages + unit % ring_pages) * 64 + row) * 128


def keys_load_cases(rng):
    # A real sliding visit's ring and window (17 pages, 8 heads, 1,023 keys across its wrap), a small ring's wrap, one
    # key, and pools through a permuted block table with an unaligned start and a partial last tile.
    for heads, ring_pages, pages, first, count, stride in ((8, 17, 0, 2000, 1023, 1088), (2, 3, 0, 100, 130, 192),
                                                           (2, 3, 0, 0, 1, 64), (8, 0, 7, 0, 1, 64), (2, 0, 7, 30, 200, 256),
                                                           (2, 0, 7, 64, 64, 128)):
        n = (ring_pages or pages) * heads * 64 * 128
        k, v = words16(rng, n), words16(rng, n)
        table = rng.sample(range(pages), pages) if pages else []
        kh, vt = words16(rng, heads * 128 * stride), words16(rng, heads * 128 * stride)
        out_k, out_v = list(kh), list(vt)
        for h in range(heads):
            plane = h * 128 * stride
            for j in range(count):
                r = key_row(table, ring_pages, heads, first + j, h)
                for d in range(128):
                    out_k[plane + j * 128 + d] = k[r + d]
                    out_v[plane + d * stride + j] = v[r + d]
        yield KEYS_LOAD, [heads, ring_pages, pages, first, count, stride], [k, v, table or [0], kh, vt], out_k + out_v


def keys_store_cases(rng):
    # A real ring's one column, more columns than a ring's positions (only the last), the wrap, and pools across a page.
    for heads, ring_pages, pages, first, T, ld in ((8, 17, 0, 2047, 1, 1024), (2, 2, 0, 10, 150, 256), (2, 2, 0, 120, 20, 260),
                                                   (8, 0, 7, 63, 2, 1024), (2, 0, 7, 60, 130, 256)):
        n = (ring_pages or pages) * heads * 64 * 128
        x_k, x_v = words16(rng, ld * T), words16(rng, ld * T)
        table = rng.sample(range(pages), pages) if pages else []
        k, v = words16(rng, n), words16(rng, n)
        out_k, out_v = list(k), list(v)
        start = max(0, T - ring_pages * 64) if ring_pages else 0
        for t in range(start, T):
            for h in range(heads):
                r = key_row(table, ring_pages, heads, first + t, h)
                for d in range(128):
                    out_k[r + d] = x_k[t * ld + h * 128 + d]
                    out_v[r + d] = x_v[t * ld + h * 128 + d]
        yield KEYS_STORE, [heads, ring_pages, pages, first, T, ld], [x_k, x_v, table or [0], k, v], out_k + out_v


def planes_cases(rng):
    # lossless_linear_planes: each plane's rows against its own columns, exact sums rounded once. Attention's shapes
    # (eight planes of three columns: scores over keys, values over a transposed plane), a full tile a plane, partial
    # rows of a plane, and exponent spreads that take the lo grids (spread 12) and the exact fallback (spread 40).
    for K, rows, stride, planes, plane_columns, fp32, spread in (
            (128, 40, 64, 8, 3, False, 2), (256, 128, 128, 8, 3, False, 2), (128, 33, 64, 8, 24, False, 12),
            (64, 1, 32, 2, 64, False, 2), (128, 64, 64, 3, 5, True, 2), (128, 20, 32, 4, 3, False, 40),
            (192, 16, 32, 8, 1, True, 12)):
        columns = planes * plane_columns
        out_rows = rows + 3
        x = [random_bf16(rng, spread) for _ in range(K * columns)]
        w = [random_bf16(rng, spread) for _ in range(planes * stride * K)]
        before = [rng.getrandbits(32 if fp32 else 16) for _ in range(out_rows * columns)]
        expected = list(before)
        for column in range(columns):
            plane = column // plane_columns
            xc = [Fraction(v) for v in x[column * K:(column + 1) * K]]
            for r in range(rows):
                row = w[(plane * stride + r) * K:(plane * stride + r + 1) * K]
                total = sum((Fraction(a) * b for a, b in zip(row, xc) if a != 0.0), Fraction(0))
                expected[column * out_rows + r] = f32_bits(f32_exact(total)) if fp32 else bf_bits(bf(total))
        yield (PLANES, [K, rows, stride, planes, plane_columns, int(fp32), out_rows],
               [u16(x), u16(w), list(range(planes)), before], expected)


def cases(rng):
    for generator in (rmsnorm_cases, mix_cases, message_gate_cases, clip_cases, conv_cases, dynamic_conv_cases,
                      sigmoid_gate_cases, add_cases, probs_cases, router_cases, quantize_cases, swiglu_cases,
                      combine_cases, softcap_cases, sqrt_cases, ring_load_cases, ring_store_cases, keys_load_cases,
                      keys_store_cases, planes_cases):
        yield from generator(rng)


def write_u32(f, values):
    f.write(struct.pack(f"<{len(values)}I", *values))


def write_cases(path: Path) -> None:
    rng = random.Random(20261122)
    count = 0
    with path.open("wb") as f:
        for op, dims, inputs, expected in cases(rng):
            write_u32(f, [op, len(dims)] + dims + [len(inputs)] + [len(v) for v in inputs] + [len(expected)])
            for v in inputs:
                write_u32(f, v)
            write_u32(f, expected)
            count += 1
    print(f"{count} cases", flush=True)


def main() -> int:
    """lossless_inkling2_oracle.py <family executable> <test name> [cache directory]."""
    command = sys.argv[1:3]
    sources = [Path(__file__), Path(cr.__file__)]
    path, tmp = cr.cached_file(sys.argv[3] if len(sys.argv) > 3 else None, "inkling2_cases", sources, write_cases)
    try:
        return subprocess.run([*command, str(path)], check=False).returncode
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
