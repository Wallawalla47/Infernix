"""Statistics over llama.cpp-format arms (infernix-certify --llama-base, llama-compare).

An arm's rows (PREFIX.positions.csv: chunk, position, kld, p_diff, same_top, nll, nll_base) hold llama.cpp's
per-position values against one base file. This tool reports, per arm, the statistics
(median KL, clipped mean KL, top-1 flips, mean dNLL) and llama.cpp's headline ones, each with a 95 % percentile
bootstrap interval over chunks (the resampling unit: 10,000 resamples, seed fixed), and paired differences of
arms scored on the same positions. It also reads llama-perplexity's final summary, so its printout can be set
beside ours line by line (the statistics port's check).

  table ARM... [--json OUT]           one row per arm
  paired A B                          A - B per statistic, resampling both on the same chunks
  printout-diff OURS THEIRS           llama-perplexity summaries (text files) side by side, value by value
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CLIP = 0.1
RESAMPLES = 10_000
SEED = 20261025
GATED = ("median_kld", "clipped_kld", "flip_rate", "mean_dnll")
REPORTED = ("mean_kld", "p99_kld", "p999_kld", "rms_p_diff")


@dataclass
class Arm:
    prefix: Path
    meta: dict
    chunk: np.ndarray
    position: np.ndarray
    kld: np.ndarray
    p_diff: np.ndarray
    same_top: np.ndarray
    nll: np.ndarray
    nll_base: np.ndarray

    @property
    def name(self) -> str:
        return self.prefix.name


def load(prefix: Path) -> Arm:
    prefix = Path(prefix)
    meta = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    data = np.loadtxt(Path(str(prefix) + ".positions.csv"), delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        raise ValueError(f"{prefix}: no positions")
    return Arm(prefix, meta, data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2], data[:, 3],
               data[:, 4].astype(bool), data[:, 5], data[:, 6])


def _values(arm: Arm) -> dict[str, np.ndarray]:
    return {"kld": arm.kld, "clipped": np.minimum(arm.kld, CLIP), "flip": (~arm.same_top).astype(np.float64),
            "dnll": arm.nll - arm.nll_base, "p_diff2": arm.p_diff.astype(np.float64) ** 2}


def point(arm: Arm) -> dict[str, float]:
    v = _values(arm)
    return {"median_kld": float(np.median(arm.kld)), "clipped_kld": float(v["clipped"].mean()),
            "flip_rate": float(v["flip"].mean()), "mean_dnll": float(v["dnll"].mean()),
            "mean_kld": float(arm.kld.mean()), "p99_kld": float(np.quantile(arm.kld, 0.99)),
            "p999_kld": float(np.quantile(arm.kld, 0.999)), "rms_p_diff": float(np.sqrt(v["p_diff2"].mean()))}


def _weights(chunk: np.ndarray, resamples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Per resample, how often each chunk is drawn (rows), and each position's chunk index."""
    units, inverse = np.unique(chunk, return_inverse=True)
    rng = np.random.default_rng(seed)
    return rng.multinomial(len(units), np.full(len(units), 1.0 / len(units)), size=resamples).astype(np.float64), inverse


def _weighted_quantiles(values: np.ndarray, inverse: np.ndarray, weights: np.ndarray, q: float,
                        block: int = 256) -> np.ndarray:
    """Per resample, the q-quantile of the values with each position weighted by its chunk's draw count (the
    lower value at an exact split, as the median of a duplicated sample takes either middle value)."""
    order = np.argsort(values, kind="stable")
    ordered, owner = values[order], inverse[order]
    out = np.empty(len(weights))
    for start in range(0, len(weights), block):
        w = weights[start:start + block][:, owner]  # resamples x positions, in ascending value order
        cumulative = np.cumsum(w, axis=1)
        target = q * cumulative[:, -1]
        index = (cumulative < target[:, None]).sum(axis=1)
        out[start:start + block] = ordered[np.minimum(index, len(ordered) - 1)]
    return out


def resampled(arm: Arm, weights: np.ndarray, inverse: np.ndarray) -> dict[str, np.ndarray]:
    """Every statistic per resample."""
    n = weights.shape[1]
    counts = weights @ np.bincount(inverse, minlength=n).astype(np.float64)
    v = _values(arm)

    def mean(x):
        return (weights @ np.bincount(inverse, weights=x, minlength=n)) / counts

    return {"median_kld": _weighted_quantiles(arm.kld, inverse, weights, 0.5), "clipped_kld": mean(v["clipped"]),
            "flip_rate": mean(v["flip"]), "mean_dnll": mean(v["dnll"]), "mean_kld": mean(arm.kld),
            "p99_kld": _weighted_quantiles(arm.kld, inverse, weights, 0.99),
            "p999_kld": _weighted_quantiles(arm.kld, inverse, weights, 0.999),
            "rms_p_diff": np.sqrt(mean(v["p_diff2"]))}


def table(arm: Arm, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, dict[str, float]]:
    weights, inverse = _weights(arm.chunk, resamples, seed)
    draws = resampled(arm, weights, inverse)
    estimate = point(arm)
    return {m: {"value": estimate[m], "low": float(np.quantile(draws[m], 0.025)),
                "high": float(np.quantile(draws[m], 0.975))} for m in GATED + REPORTED}


def paired(a: Arm, b: Arm, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, dict[str, float]]:
    """A - B on the same positions: both resampled with the same chunk draws."""
    if not (np.array_equal(a.chunk, b.chunk) and np.array_equal(a.position, b.position)):
        raise ValueError(f"{a.name} and {b.name} hold different positions")
    weights, inverse = _weights(a.chunk, resamples, seed)
    da, db = resampled(a, weights, inverse), resampled(b, weights, inverse)
    pa, pb = point(a), point(b)
    return {m: {"value": pa[m] - pb[m], "low": float(np.quantile(da[m] - db[m], 0.025)),
                "high": float(np.quantile(da[m] - db[m], 0.975))} for m in GATED + REPORTED}


# ------------------------------------------------------------------------- llama-perplexity's printout

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_printout(text: str) -> dict[str, tuple[float, float | None]]:
    """The final summary's lines ("Mean    KLD:   0.001234 ±   0.000012") as label -> (value, uncertainty),
    values as printed (percent lines in percent); everything before "====== Perplexity statistics" (the
    per-chunk table, the log) is skipped."""
    out: dict[str, tuple[float, float | None]] = {}
    started = False
    for line in text.splitlines():
        if line.startswith("====== Perplexity statistics"):
            started = True
            continue
        if not started or ":" not in line or line.startswith("======"):
            continue
        label, rest = line.split(":", 1)
        numbers = _NUMBER.findall(rest)
        if not numbers:
            continue
        unc = float(numbers[1]) if "±" in rest and len(numbers) > 1 else None
        out[" ".join(label.split())] = (float(numbers[0]), unc)
    return out


def printout_diff(ours: str, theirs: str) -> list[tuple[str, tuple | None, tuple | None, bool]]:
    a, b = parse_printout(ours), parse_printout(theirs)
    rows = []
    for label in list(dict.fromkeys(list(b) + list(a))):
        rows.append((label, a.get(label), b.get(label), a.get(label) == b.get(label)))
    return rows


def _format(row: dict[str, float]) -> str:
    return f"{row['value']:.6g} [{row['low']:.6g}, {row['high']:.6g}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("table")
    t.add_argument("arms", nargs="+", type=Path)
    t.add_argument("--json", type=Path)
    p = sub.add_parser("paired")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    d = sub.add_parser("printout-diff")
    d.add_argument("ours", type=Path)
    d.add_argument("theirs", type=Path)
    args = parser.parse_args(argv)
    if args.command == "table":
        results = {}
        for prefix in args.arms:
            arm = load(prefix)
            results[arm.name] = {"positions": int(len(arm.kld)), "chunks": int(np.unique(arm.chunk).size),
                                 "route": arm.meta.get("route"), "arm": arm.meta.get("arm"), **table(arm)}
            row = results[arm.name]
            print(f"{arm.name} ({row['positions']} positions, {row['chunks']} chunks)")
            for m in GATED + REPORTED:
                print(f"  {m:12s} {_format(row[m])}")
        if args.json:
            args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "paired":
        a, b = load(args.a), load(args.b)
        print(f"{a.name} - {b.name}")
        for m, row in paired(a, b).items():
            print(f"  {m:12s} {_format(row)}")
        return 0
    rows = printout_diff(args.ours.read_text(encoding="utf-8"), args.theirs.read_text(encoding="utf-8"))
    for label, ours, theirs, same in rows:
        print(f"{'same' if same else 'DIFF'}  {label:32s} ours {ours}  llama.cpp {theirs}")
    return 0 if all(same for *_, same in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
