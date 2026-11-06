"""Exact oracle for the lossless route's correctly rounded functions (src/ops/lossless/cr_math.cuh).

Writes test vectors and runs the CUDA test on them: python lossless_cr_oracle.py <family executable> lossless_cr_math.
Each expected value is the function evaluated with Python's decimal module at 60 significant digits
(its exp, ln and sqrt are correctly rounded at that precision) and rounded to binary32 or BF16 (round to
nearest, ties to even, subnormals, overflow to infinity) with exact rational arithmetic. 60 digits put
the evaluation far closer to the exact value than any binary32 argument's result lies to a rounding
midpoint. Inputs: every finite BF16 value, and random binary32 values across each function's range.
Standard library only.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import random
import struct
import subprocess
import sys
import tempfile
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 60

EXP, SIGMOID, SIGMOID_BF16, SILU, SILU_BF16, SOFTPLUS, RSQRT, COS, SIN = range(9)

# pi to 110 digits: arguments below 2^20 reduce with an error far below the 60-digit evaluation's.
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651")


def float_of_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def round_binary(value: Decimal, prec: int) -> int:
    """Bits of value rounded to nearest even in the format with `prec` significant bits and binary32's
    exponent range (BF16: prec 8, returned in the low 16 bits; binary32: prec 24). The sign is the value's,
    zeros included (-0, and a negative value rounding to zero, give -0)."""
    sign = 1 if value.is_signed() else 0
    # |value| = digits 10^exponent exactly: num / den in integers (no gcd: the rounding needs none).
    _, digits, exponent = value.as_tuple()
    num = int("".join(map(str, digits))) if digits else 0
    if num == 0:
        return (sign << (15 if prec == 8 else 31))
    den = 1
    if exponent >= 0:
        num *= 10 ** exponent
    else:
        den = 10 ** -exponent
    # floor(log2 |value|) is e or e - 1 for this e.
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
    mant_bits = prec - 1
    total_bits = 16 if prec == 8 else 32
    if e > 127:
        field, mantissa = 255, 0
    elif n < 2 ** (prec - 1):
        field, mantissa = 0, n
    else:
        field, mantissa = e + 127, n - 2 ** (prec - 1)
    return (sign << (total_bits - 1)) | (field << mant_bits) | mantissa


def sigmoid(d: Decimal) -> Decimal:
    # exp of a negative argument only, so nothing overflows (it underflows to 0 far below binary32).
    if d >= 0:
        return Decimal(1) / (Decimal(1) + (-d).exp())
    e = d.exp()
    return e / (Decimal(1) + e)


def cos_sin(d: Decimal) -> tuple[Decimal, Decimal]:
    """cos and sin of d (|d| < 2^20) by reduction modulo 2 pi at 120 digits and Taylor series."""
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = 120
        two_pi = 2 * PI
        k = (d / two_pi).to_integral_value()
        r = d - k * two_pi
        c, s = Decimal(0), Decimal(0)
        term, n = Decimal(1), 0
        eps = Decimal(10) ** -110
        while True:
            # term = r^n / n!
            if n % 4 == 0:
                c += term
            elif n % 4 == 1:
                s += term
            elif n % 4 == 2:
                c -= term
            else:
                s -= term
            n += 1
            term = term * r / n
            if abs(term) < eps and n > 2:
                break
    return +c, +s


def evaluate(func: int, x: float) -> int:
    d = Decimal(x)
    if func == SIN and d == 0:
        return round_binary(d, 24)  # sin(+-0) = +-0
    if func in (COS, SIN):
        c, s = cos_sin(d)
        return round_binary(c if func == COS else s, 24)
    if func == EXP:
        return round_binary(d.exp(), 24)
    if func in (SIGMOID, SIGMOID_BF16):
        return round_binary(sigmoid(d), 24 if func == SIGMOID else 8)
    if func in (SILU, SILU_BF16):
        return round_binary(d * sigmoid(d), 24 if func == SILU else 8)
    if func == SOFTPLUS:
        if d > 0:
            return round_binary(d + (Decimal(1) + (-d).exp()).ln(), 24)
        return round_binary((Decimal(1) + d.exp()).ln(), 24)
    if func == RSQRT:
        return round_binary(Decimal(1) / d.sqrt(), 24)
    raise ValueError(func)


def evaluate_record(task: tuple[int, int]) -> tuple[int, int, int]:
    func, bits = task
    return func, bits, evaluate(func, float_of_bits(bits))


def evaluate_records(tasks: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """evaluate_record over the tasks, in their order, on every core (each evaluation is independent)."""
    with multiprocessing.Pool() as pool:
        return pool.map(evaluate_record, tasks, chunksize=max(1, len(tasks) // (64 * (os.cpu_count() or 1))))


def cached_file(directory: str | None, name: str, sources: list[Path], write) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    """The file `write(path)` produces, kept in `directory` under a key of the generating sources and the
    Python version (the output is deterministic given both), so later runs reuse it; without a directory, a
    temporary file. Returns the path and the temporary directory to clean up (None for a cached file)."""
    if directory is None:
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / f"{name}.bin"
        write(path)
        return path, tmp
    key = hashlib.sha256(sys.version.encode())
    for source in sources:
        key.update(source.read_bytes())
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-{key.hexdigest()[:16]}.bin"
    if not path.is_file():
        for stale in folder.glob(f"{name}-*.bin"):
            stale.unlink(missing_ok=True)
        partial = folder / f"{path.name}.{os.getpid()}.partial"
        write(partial)
        os.replace(partial, path)
    return path, None


def vectors() -> list[tuple[int, int, int]]:
    rng = random.Random(20261021)
    tasks = []
    bf16_inputs = []
    for bits in range(0x10000):
        field = (bits >> 7) & 0xFF
        if field == 0xFF:
            continue  # infinities and NaN
        bf16_inputs.append(bits << 16)
    ranges = {EXP: (-110.0, 92.0), SIGMOID: (-120.0, 120.0), SILU: (-120.0, 120.0), SOFTPLUS: (-120.0, 120.0)}
    for func in (EXP, SIGMOID, SIGMOID_BF16, SILU, SILU_BF16, SOFTPLUS, RSQRT):
        inputs = list(bf16_inputs)
        lo, hi = ranges.get(func if func not in (SIGMOID_BF16, SILU_BF16) else func - 1, (0.0, 0.0))
        for _ in range(100_000):
            if func == RSQRT:
                bits = rng.randrange(0x00000001, 0x7F800000)  # every positive finite binary32
            else:
                bits = struct.unpack("<I", struct.pack("<f", rng.uniform(lo, hi)))[0]
                # Random low bits too, so arguments are not all short decimals.
                bits = (bits & 0xFFFF0000) | rng.randrange(0x10000)
            inputs.append(bits)
        for bits in inputs:
            x = float_of_bits(bits)
            if func == RSQRT and (x <= 0.0):
                continue
            if func == EXP and x > 89.0:
                continue
            tasks.append((func, bits))
    # cos and sin over RoPE's arguments (|x| < 2^20): the BF16 values in range, random binary32 across
    # the binades, and the binary32 values nearest multiples of pi / 2 (the reduction's hard cases).
    trig_inputs = [b for b in bf16_inputs if abs(float_of_bits(b)) < 2.0 ** 20]
    for _ in range(20_000):
        e = rng.uniform(-30.0, 20.0)
        x = rng.choice((-1.0, 1.0)) * 2.0 ** e
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        trig_inputs.append((bits & 0xFFFF0000) | rng.randrange(0x10000))
    for k in range(1, 4000):
        target = float(k * PI / 2) * rng.choice((1.0, 41.0, 677.0)) if k % 7 == 0 else float(k * PI / 2)
        if abs(target) >= 2.0 ** 20:
            continue
        bits = struct.unpack("<I", struct.pack("<f", target))[0]
        trig_inputs += [bits, bits + 1, bits - 1]
    trig_inputs = [b for b in trig_inputs if abs(float_of_bits(b)) < 2.0 ** 20]
    for func in (COS, SIN):
        for bits in trig_inputs:
            tasks.append((func, bits))
    return evaluate_records(tasks)


def write_vectors(path: Path) -> None:
    records = vectors()
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(records)))
        for func, x, want in records:
            f.write(struct.pack("<III", func, x, want))
    print(f"{len(records)} vectors", flush=True)


def main() -> int:
    """lossless_cr_oracle.py <family executable> <test name> [cache directory]: with a directory, the vectors are
    written there once per version of this file and reused afterwards (they are deterministic)."""
    command = sys.argv[1:3]  # the family executable and the test name (tests/ops/op_family.h)
    path, tmp = cached_file(sys.argv[3] if len(sys.argv) > 3 else None, "cr_vectors", [Path(__file__)], write_vectors)
    try:
        return subprocess.run([*command, str(path)], check=False).returncode
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
