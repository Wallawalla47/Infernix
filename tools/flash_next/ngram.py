"""Exact PLE n-gram row ids of Qwen3.8-Flash-Next (design §12).

Each position p hashes the n-grams ending at p (n = 2 .. ngram_size). The context token at distance
s is ``history[p - s]`` when none of ``history[p - s .. p - 1]`` is EOS, and EOS otherwise, so an EOS
closes every window. For n-gram order n:

    mixed       = XOR_{k < n} context[k] * multiplier[k]        (non-negative, < 2^63)
    row[head j] = mixed mod size[j] + offset[j]                  (heads_per_ngram heads per order)

The multipliers come from splitmix64 of a per-layer seed, and the head sizes are consecutive primes
after ``ngram_vocab_size_base - 1``, continued across PLE layers. This matches the published
implementations (SGLang ``qwen4_exp.py``); here it is restated as a plain specification.

    python3 -m tools.flash_next.ngram --fixture tests/fixtures/qwen4_exp/ngram_rows.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_PRIME_1 = 10007


@dataclass(frozen=True)
class NgramConfig:
    vocab_size: int
    eos_token_id: int
    ngram_size: int
    heads_per_ngram: int
    ngram_vocab_size_base: int
    make_ngram_vocab_size_divisible_by: int
    seed: int = 1234


def _splitmix64(x: int) -> int:
    x = (x + _GAMMA) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def layer_multipliers(c: NgramConfig, layer: int) -> list[int]:
    half_bound = max(1, ((1 << 63) - 1) // max(c.vocab_size, 1) // 2)
    base = c.seed + _PRIME_1 * layer
    return [2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half_bound) + 1 for i in range(c.ngram_size)]


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def head_tables(c: NgramConfig, layer: int) -> tuple[list[int], list[int], int]:
    """Per-head prime modulus and row offset, and the padded row count of the layer's table."""

    heads = (c.ngram_size - 1) * c.heads_per_ngram
    prime, sizes, offsets, total = c.ngram_vocab_size_base - 1, [], [], 0
    for i in range((layer + 1) * heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        if i >= layer * heads:
            sizes.append(prime)
            offsets.append(total)
            total += prime
    d = c.make_ngram_vocab_size_divisible_by
    return sizes, offsets, -(-total // d) * d


def row_ids(c: NgramConfig, layer: int, history: list[int], count: int) -> list[list[int]]:
    """Rows ``[count][heads]`` for the last ``count`` positions of ``history``.

    ``history`` holds at least ``ngram_size - 1`` tokens before those positions (EOS at sequence start).
    """

    if len(history) < count + c.ngram_size - 1:
        raise ValueError("history must cover ngram_size - 1 tokens before the first position")
    mult = layer_multipliers(c, layer)
    sizes, offsets, _ = head_tables(c, layer)
    out = []
    for p in range(len(history) - count, len(history)):
        context = [history[p - s] if c.eos_token_id not in history[p - s : p] else c.eos_token_id
                   for s in range(c.ngram_size)]
        row = []
        for n in range(2, c.ngram_size + 1):
            mixed = 0
            for k in range(n):
                mixed ^= context[k] * mult[k]
            base = (n - 2) * c.heads_per_ngram
            row.extend(mixed % sizes[j] + offsets[j] for j in range(base, base + c.heads_per_ngram))
        out.append(row)
    return out


# A synthetic configuration with the published structure (small primes keep the fixture readable).
FIXTURE_CONFIG = NgramConfig(vocab_size=151936, eos_token_id=151645, ngram_size=3, heads_per_ngram=4,
                             ngram_vocab_size_base=1000003, make_ngram_vocab_size_divisible_by=128)


def fixture_history(seed: int = 7, length: int = 400) -> list[int]:
    state, out = seed, [FIXTURE_CONFIG.eos_token_id] * (FIXTURE_CONFIG.ngram_size - 1)
    for i in range(length):
        state = _splitmix64(state)
        out.append(FIXTURE_CONFIG.eos_token_id if state % 23 == 0 else int(state % FIXTURE_CONFIG.vocab_size))
    return out


def write_fixture(path: Path) -> None:
    c = FIXTURE_CONFIG
    history = fixture_history()
    lines = [f"# vocab {c.vocab_size} eos {c.eos_token_id} n {c.ngram_size} heads {c.heads_per_ngram} "
             f"base {c.ngram_vocab_size_base} div {c.make_ngram_vocab_size_divisible_by} seed {c.seed}",
             "H " + " ".join(map(str, history))]
    for layer in (0, 1):
        sizes, offsets, padded = head_tables(c, layer)
        lines.append(f"L {layer} M " + " ".join(map(str, layer_multipliers(c, layer))))
        lines.append(f"L {layer} S " + " ".join(map(str, sizes)) + f" P {padded}")
        for p, row in enumerate(row_ids(c, layer, history, len(history) - (c.ngram_size - 1))):
            lines.append(f"R {layer} {p} " + " ".join(map(str, row)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, required=True)
    write_fixture(parser.parse_args().fixture)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
