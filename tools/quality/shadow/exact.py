"""Correct rounding in binary64 with certificates, and exact fallbacks.

Every value the shadow holds is a binary64 number that is exactly representable in its nominal dtype
(BF16 or FP32). An operation's result is computed in binary64 together with a bound on its error;
where the bound cannot decide the rounding to the nominal dtype (the interval holds a rounding
boundary), the result is recomputed exactly: sums and products with Python integers, transcendental
functions with mpmath at 160 bits, then rounded exactly.

Single IEEE operations (+, -, *, /, sqrt) on BF16 or FP32 inputs need no certificate: binary64 has more
than 2p + 2 bits for both, so rounding the binary64 result again is the correctly rounded result.
"""

from __future__ import annotations

from fractions import Fraction
import math

import mpmath
import torch

F64 = torch.float64
U = 2.0 ** -53
PRECISION = {torch.bfloat16: 8, torch.float32: 24, torch.float16: 11}
MIN_EXPONENT = {torch.bfloat16: -126, torch.float32: -126, torch.float16: -14}
MAX_FINITE = {torch.bfloat16: float.fromhex("0x1.fep127"), torch.float32: float.fromhex("0x1.fffffep127"),
              torch.float16: 65504.0}
mpmath.mp.prec = 160


def round_to(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Binary64 values rounded to nearest even in dtype, held in binary64."""
    if dtype == F64 or not dtype.is_floating_point:
        return x
    if dtype == torch.float32:
        return x.to(torch.float32).to(F64)
    if dtype == torch.bfloat16:
        # Round to odd into binary32 (24 bits >= 8 + 2), then to nearest even into BF16: one correct rounding.
        t = x.to(torch.float32)
        back = t.to(F64)
        overshoot = back.abs() > x.abs()
        t = torch.where(overshoot, torch.nextafter(t, torch.zeros_like(t)), t)
        inexact = t.to(F64) != x
        bits = t.view(torch.int32)
        bits = torch.where(inexact & torch.isfinite(t), bits | 1, bits)
        return bits.view(torch.float32).to(torch.bfloat16).to(F64)
    if dtype == torch.float16:
        t = x.to(torch.float32)
        back = t.to(F64)
        overshoot = back.abs() > x.abs()
        t = torch.where(overshoot, torch.nextafter(t, torch.zeros_like(t)), t)
        inexact = t.to(F64) != x
        bits = t.view(torch.int32)
        bits = torch.where(inexact & torch.isfinite(t), bits | 1, bits)
        return bits.view(torch.float32).to(torch.float16).to(F64)
    raise TypeError(f"no rounding to {dtype}")


def certain(value: torch.Tensor, error: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Where every real within error of value rounds to the same dtype value."""
    return round_to(value - error, dtype) == round_to(value + error, dtype)


# ----------------------------------------------------------------------------------------------- exact values

def exact(value: float) -> Fraction:
    return Fraction(value)


def round_fraction(q: Fraction, dtype: torch.dtype) -> float:
    """The nearest-even dtype value of the exact rational q."""
    if q == 0:
        return 0.0
    p, emin = PRECISION[dtype], MIN_EXPONENT[dtype]
    sign = -1.0 if q < 0 else 1.0
    a = abs(q)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, emin)  # subnormal range: fixed quantum
    quantum = Fraction(2) ** (e - p + 1)
    n = a / quantum
    whole = n.numerator // n.denominator
    rest = n - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    result = float(whole * quantum)
    if result > MAX_FINITE[dtype]:
        return sign * math.inf
    return sign * result


def round_mpf(value, dtype: torch.dtype) -> float:
    """The nearest-even dtype value of an mpmath number known to 160 bits (far beyond any tie)."""
    if value == 0:
        return 0.0
    sign, man, e, _ = mpmath.mpf(value)._mpf_
    q = Fraction(int(man)) * (Fraction(2) ** e if e >= 0 else Fraction(1, 2 ** -e))
    return round_fraction(-q if sign else q, dtype)


def exact_dot(a: list[float], b: list[float], extra: float = 0.0) -> Fraction:
    """sum a_i b_i + extra, exactly (binary64 inputs)."""
    terms = [Fraction(x) * Fraction(y) for x, y in zip(a, b) if x != 0.0 and y != 0.0]
    return sum(terms, Fraction(extra))


# ------------------------------------------------------------------------------------------- certified ops

def matmul(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype, bias: torch.Tensor | None = None) -> torch.Tensor:
    """(a @ b [+ bias]) correctly rounded to dtype; a [..., M, K], b [..., K, N] (batch dims broadcast)."""
    r = torch.matmul(a, b)
    bound = torch.matmul(a.abs(), b.abs())
    if bias is not None:
        r = r + bias
        bound = bound + bias.abs()
    k = a.shape[-1]
    error = bound * ((k + 4) * U * 1.01) + 2.0 ** -1070
    out = round_to(r, dtype)
    ok = certain(r, error, dtype)
    if not bool(ok.all()):
        ab, bb = torch.broadcast_tensors(a[..., :, :, None], b[..., None, :, :])  # [..., M, K, N] views
        bias_b = None if bias is None else torch.broadcast_to(bias, r.shape)
        for index in torch.nonzero(~ok).tolist():
            *batch, i, j = index
            lhs = ab[tuple(batch)][i, :, j].tolist()
            rhs = bb[tuple(batch)][i, :, j].tolist()
            extra = 0.0 if bias_b is None else float(bias_b[tuple(index)])
            out[tuple(index)] = round_fraction(exact_dot(lhs, rhs, extra), dtype)
    return out


def reduce_sum(x: torch.Tensor, dims, keepdim: bool, dtype: torch.dtype, divisor: int = 1) -> torch.Tensor:
    """sum (or mean, with divisor) over dims, exact and rounded once to dtype."""
    dims = tuple(d % x.dim() for d in dims) if dims else tuple(range(x.dim()))
    r = x.sum(dim=dims, keepdim=True)
    n = math.prod(x.shape[d] for d in dims)
    error = x.abs().sum(dim=dims, keepdim=True) * ((n + 4) * U * 1.01) + 2.0 ** -1070
    value, error = r / divisor, error / divisor + (r / divisor).abs() * U
    out = round_to(value, dtype)
    ok = certain(value, error, dtype)
    if not bool(ok.all()):
        moved = x.permute(*[d for d in range(x.dim()) if d not in dims], *dims)
        flat = moved.reshape(*moved.shape[:x.dim() - len(dims)], -1)
        keep_shape = [x.shape[d] for d in range(x.dim()) if d not in dims]
        okk = ok.reshape(keep_shape) if keep_shape else ok.reshape(())
        outk = out.reshape(keep_shape) if keep_shape else out.reshape(())
        for index in torch.nonzero(~okk).tolist() if keep_shape else [[]]:
            terms = flat[tuple(index)].tolist()
            total = sum((Fraction(t) for t in terms), Fraction(0)) / divisor
            outk[tuple(index)] = round_fraction(total, dtype)
        out = outk.reshape(out.shape)
    return out if keepdim else out.squeeze(dims) if dims else out.reshape(())


_MP = {
    "exp": mpmath.exp, "sigmoid": lambda x: 1 / (1 + mpmath.exp(-x)), "silu": lambda x: x / (1 + mpmath.exp(-x)),
    "rsqrt": lambda x: 1 / mpmath.sqrt(x), "cos": mpmath.cos, "sin": mpmath.sin, "tanh": mpmath.tanh,
    "log": mpmath.log, "log1p": lambda x: mpmath.log(1 + x), "expm1": mpmath.expm1, "sqrt": mpmath.sqrt,
}
_TORCH = {
    "exp": torch.exp, "sigmoid": torch.sigmoid, "silu": torch.nn.functional.silu, "rsqrt": torch.rsqrt,
    "cos": torch.cos, "sin": torch.sin, "tanh": torch.tanh, "log": torch.log, "log1p": torch.log1p,
    "expm1": torch.expm1, "sqrt": torch.sqrt,
}


def transcendental(name: str, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """name(x) for binary64 inputs, correctly rounded to dtype."""
    r = _TORCH[name](x)
    error = r.abs() * (32 * U) + 2.0 ** -1070
    out = round_to(r, dtype)
    ok = certain(r, error, dtype) | ~torch.isfinite(r)
    if not bool(ok.all()):
        f = _MP[name]
        for index in torch.nonzero(~ok).tolist():
            out[tuple(index)] = round_mpf(f(mpmath.mpf(float(x[tuple(index)]))), dtype)
    return out


def softplus(x: torch.Tensor, beta: float, threshold: float, dtype: torch.dtype) -> torch.Tensor:
    """torch.nn.functional.softplus: x where beta x > threshold, else log1p(exp(beta x)) / beta, correctly rounded."""
    linear = x * beta > threshold
    r = torch.log1p(torch.exp(x * beta)) / beta
    error = r.abs() * (32 * U) + 2.0 ** -1070
    out = torch.where(linear, round_to(x, dtype), round_to(r, dtype))
    ok = linear | certain(r, error, dtype)
    if not bool(ok.all()):
        for index in torch.nonzero(~ok).tolist():
            v = mpmath.mpf(float(x[tuple(index)]))
            out[tuple(index)] = round_mpf(mpmath.log(1 + mpmath.exp(v * beta)) / beta, dtype)
    return out


def power_scalar_base(base: float, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """base ** x (a scalar base, binary64 exponents), correctly rounded."""
    r = torch.pow(torch.tensor(base, dtype=F64), x)
    error = r.abs() * (64 * U) + 2.0 ** -1070
    out = round_to(r, dtype)
    ok = certain(r, error, dtype)
    if not bool(ok.all()):
        for index in torch.nonzero(~ok).tolist():
            out[tuple(index)] = round_mpf(mpmath.power(mpmath.mpf(base), mpmath.mpf(float(x[tuple(index)]))), dtype)
    return out


def softmax(x: torch.Tensor, dim: int, dtype: torch.dtype) -> torch.Tensor:
    """softmax(x) along dim, each element the correctly rounded exact value."""
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - m)
    s = e.sum(dim=dim, keepdim=True)
    r = e / s
    n = x.shape[dim]
    error = r * ((n + 64) * U) + 2.0 ** -1070
    out = round_to(r, dtype)
    ok = certain(r, error, dtype)
    if not bool(ok.all()):
        moved = x.movedim(dim, -1)
        okm, outm = ok.movedim(dim, -1), out.movedim(dim, -1)
        for index in torch.nonzero(~okm).tolist():
            *row, j = index
            values = [mpmath.mpf(v) for v in moved[tuple(row)].tolist()]
            top = max(values)
            total = mpmath.fsum(mpmath.exp(v - top) for v in values)
            outm[tuple(index)] = round_mpf(mpmath.exp(values[j] - top) / total, dtype)
        out = outm.movedim(-1, dim)
    return out
