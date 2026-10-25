"""Statistics over llama.cpp-format arms (infernix-certify --llama-base, llama-rows, llama-compare).

An arm's rows (PREFIX.positions.csv: chunk, position, kld, p_diff, same_top, nll, nll_base, kld_top) hold
llama.cpp's per-position values against one base file. This tool reports, per arm, the statistics people
publish from llama-perplexity --kl-divergence: perplexity (and the ln ratio to the base's), mean KLD, KLD at
the 99.9th percentile and the top-1 agreement ("Same top p"), each with a 95 % percentile bootstrap interval
over chunks (the resampling unit: 10,000 resamples, seed fixed), and paired differences of arms scored on the
same positions. kld_top, the decode arms' form of the KL (the observed tokens plus one bucket), is reported
as mean_kld_top where an arm has it. It also reads llama-perplexity's final summary, so its printout can be
set beside ours line by line (the statistics port's check) or stand for an engine that only prints one.

  table ARM... [--json OUT]           one row per arm
  paired A B                          A - B per statistic, resampling both on the same chunks
  printout-diff OURS THEIRS           llama-perplexity summaries (text files) side by side, value by value
  head BASE N OUT                     a base file of BASE's first N chunks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESAMPLES = 10_000
SEED = 20261025
# The published statistics, in the order a report shows them.
METRICS = ("ppl", "ln_ppl_ratio", "mean_kld", "p999_kld", "same_top")
EXTRA = ("mean_kld_top",)
COLUMNS = ("chunk", "position", "kld", "p_diff", "same_top", "nll", "nll_base", "kld_top")


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
    kld_top: np.ndarray  # NaN where the source has no observed form (a full-vocabulary engine tool)

    @property
    def name(self) -> str:
        return self.prefix.name


def load(prefix: Path) -> Arm:
    prefix = Path(prefix)
    meta = json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))
    path = Path(str(prefix) + ".positions.csv")
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if tuple(header) != COLUMNS:
        raise ValueError(f"{path}: columns {header}, expected {list(COLUMNS)}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        raise ValueError(f"{prefix}: no positions")
    return Arm(prefix, meta, data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2], data[:, 3],
               data[:, 4].astype(bool), data[:, 5], data[:, 6], data[:, 7])


def write_positions(prefix: Path, chunk, position, kld, p_diff, same_top, nll, nll_base, kld_top=None) -> None:
    """An arm's rows from another tool's per-position values (an adapter's), in the certify format."""
    n = len(kld)
    kld_top = np.full(n, np.nan) if kld_top is None else kld_top
    rows = np.column_stack([chunk, position, kld, p_diff, np.asarray(same_top, dtype=int), nll, nll_base, kld_top])
    np.savetxt(str(prefix) + ".positions.csv", rows, delimiter=",", header=",".join(COLUMNS), comments="",
               fmt=["%d", "%d", "%.9g", "%.9g", "%d", "%.9g", "%.9g", "%.9g"])


def point(arm: Arm) -> dict[str, float]:
    """The statistics as llama-perplexity prints them (its 99.9 % interpolates between order statistics)."""
    out = {"ppl": float(np.exp(arm.nll.mean())), "ln_ppl_ratio": float((arm.nll - arm.nll_base).mean()),
           "mean_kld": float(arm.kld.mean()), "p999_kld": float(np.quantile(arm.kld, 0.999)),
           "same_top": float(arm.same_top.mean())}
    if not np.isnan(arm.kld_top).any():
        out["mean_kld_top"] = float(arm.kld_top.mean())
    return out


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

    def mean(x):
        return (weights @ np.bincount(inverse, weights=x, minlength=n)) / counts

    out = {"ppl": np.exp(mean(arm.nll)), "ln_ppl_ratio": mean(arm.nll - arm.nll_base), "mean_kld": mean(arm.kld),
           "p999_kld": _weighted_quantiles(arm.kld, inverse, weights, 0.999),
           "same_top": mean(arm.same_top.astype(np.float64))}
    if not np.isnan(arm.kld_top).any():
        out["mean_kld_top"] = mean(arm.kld_top)
    return out


def _interval(estimate: float, draws: np.ndarray) -> dict[str, float]:
    return {"value": estimate, "low": float(np.quantile(draws, 0.025)), "high": float(np.quantile(draws, 0.975))}


def table(arm: Arm, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, dict[str, float]]:
    weights, inverse = _weights(arm.chunk, resamples, seed)
    draws = resampled(arm, weights, inverse)
    return {m: _interval(v, draws[m]) for m, v in point(arm).items()}


def same_positions(a: Arm, b: Arm) -> bool:
    return np.array_equal(a.chunk, b.chunk) and np.array_equal(a.position, b.position)


def paired(a: Arm, b: Arm, resamples: int = RESAMPLES, seed: int = SEED) -> dict[str, dict[str, float]]:
    """A - B on the same positions: both resampled with the same chunk draws."""
    if not same_positions(a, b):
        raise ValueError(f"{a.name} and {b.name} hold different positions")
    weights, inverse = _weights(a.chunk, resamples, seed)
    da, db = resampled(a, weights, inverse), resampled(b, weights, inverse)
    pa, pb = point(a), point(b)
    return {m: _interval(pa[m] - pb[m], da[m] - db[m]) for m in pa if m in pb}


def identical(a: Arm, b: Arm) -> bool:
    """Whether two arms scored every position to the same values (a path that changed nothing there)."""
    return same_positions(a, b) and all(np.array_equal(x, y, equal_nan=True) for x, y in
                                        ((a.kld, b.kld), (a.nll, b.nll), (a.same_top, b.same_top), (a.kld_top, b.kld_top)))


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


def printout_point(text: str) -> dict[str, dict[str, float]]:
    """An engine that only prints llama-perplexity's summary: its statistics with llama.cpp's own ± (one
    standard error, shown as value +- 1.96 se), the 99.9 % without an interval."""
    parsed = parse_printout(text)
    if "Mean KLD" not in parsed:
        raise ValueError("no KL-divergence summary in the printout")

    def row(value: float, se: float | None) -> dict[str, float]:
        if se is None:
            return {"value": value, "low": float("nan"), "high": float("nan")}
        return {"value": value, "low": value - 1.96 * se, "high": value + 1.96 * se}

    ppl, ppl_se = parsed["Mean PPL(Q)"]
    ratio, ratio_se = parsed["Mean ln(PPL(Q)/PPL(base))"]
    kld, kld_se = parsed["Mean KLD"]
    top, top_se = parsed["Same top p"]
    return {"ppl": row(ppl, ppl_se), "ln_ppl_ratio": row(ratio, ratio_se), "mean_kld": row(kld, kld_se),
            "p999_kld": row(parsed["99.9% KLD"][0], None), "same_top": row(top / 100, (top_se or 0) / 100)}


def printout_diff(ours: str, theirs: str) -> list[tuple[str, tuple | None, tuple | None, bool]]:
    a, b = parse_printout(ours), parse_printout(theirs)
    rows = []
    for label in list(dict.fromkeys(list(b) + list(a))):
        rows.append((label, a.get(label), b.get(label), a.get(label) == b.get(label)))
    return rows


def head(base: Path, chunks: int, out: Path) -> None:
    """A base file of the first `chunks` chunks (llama-perplexity's KL mode scores every chunk its base holds)."""
    with Path(base).open("rb") as f:
        magic = f.read(8)
        if magic != b"_logits_":
            raise ValueError(f"{base} is not a llama.cpp _logits_ file")
        context, vocab, total = np.frombuffer(f.read(12), dtype="<i4")
        if not 0 < chunks <= total:
            raise ValueError(f"{base} holds {total} chunks")
        tokens = f.read(4 * int(context) * int(total))
        row_bytes = 2 * (2 * ((int(vocab) + 1) // 2) + 4)
        rows = int(context) - 1 - int(context) // 2
        with Path(out).open("wb") as g:
            g.write(magic + np.array([context, vocab, chunks], dtype="<i4").tobytes())
            g.write(tokens[: 4 * int(context) * chunks])
            remaining = rows * row_bytes * chunks
            while remaining:
                block = f.read(min(remaining, 1 << 26))
                if not block:
                    raise ValueError(f"{base} is truncated")
                g.write(block)
                remaining -= len(block)


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
    h = sub.add_parser("head")
    h.add_argument("base", type=Path)
    h.add_argument("chunks", type=int)
    h.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "head":
        head(args.base, args.chunks, args.out)
        return 0
    if args.command == "table":
        results = {}
        for prefix in args.arms:
            arm = load(prefix)
            results[arm.name] = {"positions": int(len(arm.kld)), "chunks": int(np.unique(arm.chunk).size),
                                 "route": arm.meta.get("route"), "arm": arm.meta.get("arm"), **table(arm)}
            row = results[arm.name]
            print(f"{arm.name} ({row['positions']} positions, {row['chunks']} chunks)")
            for m in METRICS + EXTRA:
                if m in row:
                    print(f"  {m:13s} {_format(row[m])}")
        if args.json:
            args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "paired":
        a, b = load(args.a), load(args.b)
        print(f"{a.name} - {b.name}")
        for m, row in paired(a, b).items():
            print(f"  {m:13s} {_format(row)}")
        return 0
    rows = printout_diff(args.ours.read_text(encoding="utf-8"), args.theirs.read_text(encoding="utf-8"))
    for label, ours, theirs, same in rows:
        print(f"{'same' if same else 'DIFF'}  {label:32s} ours {ours}  llama.cpp {theirs}")
    return 0 if all(same for *_, same in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
