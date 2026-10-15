"""Fast-path numerical acceptance gate.

Decides whether an artifact's fast paths keep it within Infernix's numerical acceptance policy
(docs/maintainer/model-quality.md, section 6). The policy compares two Engine routes against one
fixed reference distribution: the exact route (--disable-fast-paths), in the default role, and the
fast-path route (every path qualified for the configuration plus any --fast-path candidates, so a
new path is always measured together with the paths it will run beside). A fast path may move
the output away from the reference by at most a fixed fraction of what the artifact already loses
(its floor: the default route's own divergence from the reference, which its activation and KV
quantization and rounding set), so the budget is small where the arithmetic is precise and larger
with 4-bit activations or KV.
Every decision is a one-sided 95 % upper confidence bound over independent units (scoring windows
or decode segments), so noise makes a fast path harder to pass, never easier: a noisier comparison
needs more data.

Subcommands:
  prefill          run (or reuse) the reference and the default and fast-path
                   infernix-perplexity runs; decide KL, top-1, NLL, tail and context criteria
  decode           run (or reuse) the reference's top tokens at teacher-forced decode positions and
                   the default and fast-path teacher-forced decode runs observing them (and, with
                   --free-running, greedy generations and the judge); decide the decode KL, top-1,
                   NLL, tail and prompt-length criteria
  evaluate-prefill decide an existing prefill directory
  compare-prefill  decide any two sets of perplexity runs of one reference (a numerics change to
                   the default route, or a control)
  evaluate-decode  decide two existing teacher-forced decode runs of one reference
  compare-decode   decide any two teacher-forced runs of one reference (controls)
  evaluate-free-running  decide an existing judge report
  identical        check that two builds' default routes agree bit for bit: two infernix-perplexity
                   run directories (every window's NLL and divergence), two teacher-forced decode
                   runs, or two generation files

Exit status: 0 every criterion passes, 1 a criterion fails, 2 the measurement is unusable.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Policy:
    """The acceptance policy's constants (docs/maintainer/model-quality.md, section 6.3)."""

    # A fast path's excess over the default route, as a fraction of the artifact's floor.
    relative_budget: float = 0.10
    # KL budget bounds (nats): the smallest the measurement resolves, the most any fast path may
    # cost whatever the artifact's floor. NLL excess (prefill and teacher-forced decode) uses the
    # same budget: in expectation it is the same quantity measured on the corpus's own tokens.
    kl_budget_minimum: float = 1.0e-4
    kl_budget_maximum: float = 3.0e-3
    # Decode's bounded criterion is total variation (bounded by 1 per position, so a position whose
    # most probable token flips cannot dominate the mean as an unbounded KL does): budget bounds.
    tv_budget_minimum: float = 1.0e-4
    tv_budget_maximum: float = 5.0e-3
    # Top-1 disagreement budget bounds (fractions of positions or tokens).
    top1_budget_minimum: float = 5.0e-4
    top1_budget_maximum: float = 5.0e-3
    # Free-running decode regret budget lower bound (nats per token).
    regret_budget_minimum: float = 1.0e-4
    # Gross caps on point estimates: 99.9th-percentile KL ratio, and each context bucket's (or
    # decode prompt length's) excess as a fraction of its default divergence or of the floor.
    tail_ratio: float = 1.25
    bucket_relative: float = 0.25
    bucket_min_positions: int = 20000
    # The KL lower bound must resolve at least this share of the reference distribution.
    coverage_minimum: float = 0.90
    # One-sided 95 % normal quantile and the fewest independent units it is applied to.
    z: float = 1.6448536269514722
    min_units: int = 30


POLICY = Policy()

# fast_path_name of every infernix::FastPath.
FAST_PATHS = ("prompt-pv8", "decode-split64", "batched-decode", "batched-a8-decode",
              "batched-a4-decode", "batched-a4-residual-decode", "batched-a8-residual-decode",
              "batch-split-attention", "adaptive-speculation", "free-cache-resume", "preemption")


class MeasurementError(RuntimeError):
    """The runs cannot support a decision (missing, mismatched or too few units)."""


@dataclass
class Criterion:
    name: str
    estimate: float
    bound: float
    budget: float
    passed: bool
    detail: str = ""

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "estimate": self.estimate,
            "upper_bound": self.bound,
            "budget": self.budget,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class Verdict:
    scope: str
    criteria: list[Criterion] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.criteria)

    def to_json(self) -> dict:
        return {
            "scope": self.scope,
            "passed": self.passed,
            "facts": self.facts,
            "criteria": [c.to_json() for c in self.criteria],
        }


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def kl_budget(floor_kl: float, policy: Policy) -> float:
    return clamp(policy.relative_budget * floor_kl, policy.kl_budget_minimum,
                 policy.kl_budget_maximum)


@dataclass(frozen=True)
class PairedUnit:
    """One independent unit: its weight and the paired default/fast totals of a quantity."""

    weight: float
    default: float
    fast: float


def paired_excess(units: Sequence[PairedUnit], z: float) -> tuple[float, float]:
    """Weighted mean of (fast - default) per unit weight and its one-sided upper bound.

    The ratio estimator sum(fast - default) / sum(weight) with a cluster-robust standard error:
    each unit is one independent sample (a scoring window or a decode segment).
    """
    m = len(units)
    total = sum(u.weight for u in units)
    if m < 2 or total <= 0:
        raise MeasurementError("too few units for a confidence bound")
    mean = sum(u.fast - u.default for u in units) / total
    residual = sum(((u.fast - u.default) - mean * u.weight) ** 2 for u in units)
    standard_error = math.sqrt(residual * m / (m - 1)) / total
    return mean, mean + z * standard_error


def bounded(name: str, units: Sequence[PairedUnit], budget: float, policy: Policy,
            detail: str) -> Criterion:
    mean, bound = paired_excess(units, policy.z)
    return Criterion(name, mean, bound, budget, bound <= budget, detail)


# --- prefill -------------------------------------------------------------------------------------


def load_report(path: Path) -> dict:
    report_path = path / "report.json"
    if not report_path.is_file():
        raise MeasurementError(f"missing report: {report_path}")
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if "kl_divergence" not in report:
        raise MeasurementError(f"{report_path} was not scored against a reference (--kl-reference)")
    return report


def window_records(report: dict) -> dict[tuple[str, int], dict]:
    windows = {}
    for stream in report["streams"]:
        for window in stream["windows"]:
            divergence = window.get("kl_divergence")
            if divergence is None:
                raise MeasurementError(
                    "report has no per-window divergence: rebuild infernix-perplexity"
                )
            windows[(stream["id"], window["index"])] = {
                "positions": divergence["positions"],
                "mean_kl": divergence["mean_kl"],
                "top1_agreement": divergence["top1_agreement"],
                "total_nll": window["total_nll"],
                "scored_tokens": window["scored_tokens"],
            }
    return windows


@dataclass
class ChunkPair:
    label: str
    default: dict
    fast: dict


def check_pair(pair: ChunkPair, *, fast_route: bool) -> None:
    a, b = pair.default, pair.fast
    for key in ("context_tokens", "stride_tokens"):
        if a["execution"][key] != b["execution"][key]:
            raise MeasurementError(f"{pair.label}: the runs differ in {key}")
    if a["kl_divergence"]["reference"] != b["kl_divergence"]["reference"]:
        raise MeasurementError(f"{pair.label}: the runs used different references")
    if fast_route:
        for key in ("kv_dtype", "prefill_chunk_tokens"):
            if a["execution"][key] != b["execution"][key]:
                raise MeasurementError(f"{pair.label}: default and fast differ in {key}")
        if a["execution"].get("fast_paths") or not b["execution"].get("fast_paths"):
            raise MeasurementError(
                f"{pair.label}: expected an exact-route run and a run with fast paths active")


def prefill_pairs(directory: Path) -> list[ChunkPair]:
    pairs = []
    for default_dir in sorted(d for d in directory.glob("default-c*") if d.is_dir()):
        chunk = default_dir.name.removeprefix("default-c")
        pairs.append(ChunkPair(f"chunk {chunk}", load_report(default_dir),
                               load_report(directory / f"fast-c{chunk}")))
    if not pairs:
        raise MeasurementError(f"no default-c<chunk> runs in {directory}")
    for pair in pairs:
        check_pair(pair, fast_route=True)
    return pairs


def compared_pairs(base: Sequence[Path], candidate: Sequence[Path]) -> list[ChunkPair]:
    if len(base) != len(candidate) or not base:
        raise MeasurementError("compare equal, nonempty numbers of base and candidate runs")
    pairs = [ChunkPair(f"{b.name} vs {c.name}", load_report(b), load_report(c))
             for b, c in zip(base, candidate)]
    for pair in pairs:
        check_pair(pair, fast_route=False)
    return pairs


def evaluate_prefill_pairs(pairs: Sequence[ChunkPair], scope: str,
                           policy: Policy = POLICY) -> Verdict:
    verdict = Verdict(scope=scope)
    # Each window is one unit; its runs (chunk sizes) are summed into it.
    # totals: positions, KL default, KL fast, disagreement default/fast, NLL default/fast
    units: dict[tuple[str, int], list[float]] = {}
    for pair in pairs:
        default = window_records(pair.default)
        fast = window_records(pair.fast)
        if default.keys() != fast.keys():
            raise MeasurementError(f"{pair.label}: the runs scored different windows")
        for key, d in default.items():
            f = fast[key]
            if d["positions"] != f["positions"] or d["scored_tokens"] != f["scored_tokens"]:
                raise MeasurementError(f"{pair.label}: window {key} position counts differ")
            n = d["positions"]
            totals = units.setdefault(key, [0.0] * 7)
            totals[0] += n
            totals[1] += n * d["mean_kl"]
            totals[2] += n * f["mean_kl"]
            totals[3] += n * (1.0 - d["top1_agreement"])
            totals[4] += n * (1.0 - f["top1_agreement"])
            totals[5] += d["total_nll"]
            totals[6] += f["total_nll"]
    if len(units) < policy.min_units:
        raise MeasurementError(
            f"{len(units)} scoring windows; the bound needs at least {policy.min_units}")
    coverage = min(p.default["kl_divergence"]["mean_reference_top_mass"] for p in pairs)
    if coverage < policy.coverage_minimum:
        raise MeasurementError(
            f"the reference's saved tokens cover {coverage:.3f} of its distribution; the KL bound "
            f"needs at least {policy.coverage_minimum}")
    positions = sum(t[0] for t in units.values())
    # The floor: the default route's own divergence from the reference, which its activation and
    # KV quantization and rounding set.
    floor_kl = sum(t[1] for t in units.values()) / positions
    floor_top1 = sum(t[3] for t in units.values()) / positions
    verdict.facts = {
        "runs": [p.label for p in pairs],
        "kv_dtype": pairs[0].default["execution"]["kv_dtype"],
        "windows": len(units),
        "positions_per_route": positions,
        "reference_top_mass": coverage,
        "floor_kl": floor_kl,
        "floor_top1_disagreement": floor_top1,
        "fast_mean_kl": sum(t[2] for t in units.values()) / positions,
        "fast_top1_disagreement": sum(t[4] for t in units.values()) / positions,
        "default_mean_nll": sum(t[5] for t in units.values()) / positions,
        "fast_mean_nll": sum(t[6] for t in units.values()) / positions,
        "fast_identical_to_default": all(
            t[1] == t[2] and t[3] == t[4] and t[5] == t[6] for t in units.values()),
    }

    budget = kl_budget(floor_kl, policy)
    verdict.criteria.append(bounded(
        "mean KL excess", [PairedUnit(t[0], t[1], t[2]) for t in units.values()], budget, policy,
        f"floor {floor_kl:.6g}"))
    top1_budget = clamp(policy.relative_budget * floor_top1, policy.top1_budget_minimum,
                        policy.top1_budget_maximum)
    verdict.criteria.append(bounded(
        "top-1 disagreement excess", [PairedUnit(t[0], t[3], t[4]) for t in units.values()],
        top1_budget, policy, f"floor {floor_top1:.6g}"))
    # Corpus NLL moves at first order under any perturbation of the logits, where the KL excess
    # moves at second order: a pure reordering shifted it by 1.7e-3 on the 27B NVFP4 artifact
    # (section 6.4). It is therefore a gross cap on the point estimate, for what truncated KL cannot
    # see (probability moved within the unsaved tail); its bound is reported.
    nll_mean, nll_bound = paired_excess(
        [PairedUnit(t[0], t[5], t[6]) for t in units.values()], policy.z)
    nll_cap = max(policy.bucket_relative * floor_kl, policy.kl_budget_minimum)
    verdict.criteria.append(
        Criterion("corpus NLL excess", nll_mean, nll_mean, nll_cap, nll_mean <= nll_cap,
                  f"95% bound {nll_bound:.4e}; perplexity ratio {math.exp(nll_mean):.5f}"))

    # Gross caps on point estimates: the far tail and each long-enough context bucket.
    default_tail = sum(p.default["kl_divergence"]["p999_kl"] for p in pairs) / len(pairs)
    fast_tail = sum(p.fast["kl_divergence"]["p999_kl"] for p in pairs) / len(pairs)
    if default_tail > 0:
        tail_ratio = fast_tail / default_tail
    else:
        tail_ratio = 0.0 if fast_tail == 0 else math.inf
    verdict.criteria.append(
        Criterion("p99.9 KL ratio", tail_ratio, tail_ratio, policy.tail_ratio,
                  tail_ratio <= policy.tail_ratio,
                  f"default {default_tail:.6g}, fast {fast_tail:.6g}"))
    buckets: dict[int, list[float]] = {}
    for pair in pairs:
        for route, report in ((1, pair.default), (2, pair.fast)):
            for item in report["kl_divergence"]["by_context"]:
                totals = buckets.setdefault(item["context_begin"],
                                            [0.0, 0.0, 0.0, item["context_end"]])
                if route == 1:
                    totals[0] += item["positions"]
                totals[route] += item["positions"] * item["mean_kl"]
    for begin in sorted(buckets):
        n, d, f, end = buckets[begin]
        if n < policy.bucket_min_positions:
            continue
        excess = (f - d) / n
        bucket_budget = max(policy.bucket_relative * d / n, policy.kl_budget_minimum)
        verdict.criteria.append(
            Criterion(f"context [{begin},{end}) KL excess", excess, excess, bucket_budget,
                      excess <= bucket_budget, f"{int(n)} positions, default {d / n:.6g}"))
    return verdict


def evaluate_prefill(directory: Path, policy: Policy = POLICY) -> Verdict:
    return evaluate_prefill_pairs(prefill_pairs(directory), f"prefill {directory}", policy)


# --- decode --------------------------------------------------------------------------------------


def load_forced(path: Path) -> dict:
    if not path.is_file():
        raise MeasurementError(f"missing teacher-forced run: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# infernix_decode_quality_forced --kv-dtype names.
KV_FORMATS = ("bf16", "fp8", "int8", "nvfp4", "k8v4", "vq2", "k4v2")


def position_divergence(reference: Sequence[float],
                        evaluated: Sequence[float]) -> tuple[float, float, float]:
    """KL(reference || evaluated) over the reference's top tokens plus one bucket for the rest (a
    lower bound of the full divergence, as infernix-perplexity computes it), the total variation
    distance over the same coarsening (also a lower bound), and the reference's top mass."""
    kl = 0.0
    tv = 0.0
    reference_mass = 0.0
    evaluated_mass = 0.0
    for r, e in zip(reference, evaluated, strict=True):
        if not (math.isfinite(r) and math.isfinite(e)):
            raise MeasurementError("a divergence received a non-finite log-probability")
        p, q = math.exp(r), math.exp(e)
        kl += p * (r - e)
        tv += abs(p - q)
        reference_mass += p
        evaluated_mass += q
    reference_rest = 1.0 - reference_mass
    if reference_rest > 0.0:
        evaluated_rest = max(1.0 - evaluated_mass, 1.0e-12)
        kl += reference_rest * (math.log(reference_rest) - math.log(evaluated_rest))
    tv += abs(max(reference_rest, 0.0) - max(1.0 - evaluated_mass, 0.0))
    return kl, 0.5 * tv, min(reference_mass, 1.0)


@dataclass
class ForcedSegment:
    """One teacher-forced segment of one run, scored against the reference."""

    prompt_tokens: int
    kl: list[float]          # per forced position
    tv: list[float]          # per forced position
    disagreement: list[int]  # 1 where the run's most probable token is not the reference's
    nll: float               # summed over the forced tokens


def forced_segments(run: dict, reference: dict) -> tuple[list[ForcedSegment], float]:
    """A run's segments scored against the reference, and the reference's mean top mass."""
    if run.get("observed_reference") in (None, ""):
        raise MeasurementError("the teacher-forced run did not observe a reference (--observe)")
    top = reference["top_tokens"]
    if len(run["segments"]) != len(reference["segments"]):
        raise MeasurementError("the run and the reference hold different segments")
    out = []
    mass_total = 0.0
    positions = 0
    for segment, ref in zip(run["segments"], reference["segments"]):
        if (segment["offset"], segment["prompt_tokens"]) != (ref["offset"], ref["prompt_tokens"]):
            raise MeasurementError("the run and the reference forced different segments")
        forced = len(segment["logprobs"])
        observed, best = segment.get("observed"), segment.get("top_logprobs")
        if observed is None or len(observed) != forced * top or len(best) != forced:
            raise MeasurementError(f"segment at {segment['offset']} lacks its observations")
        kl, tv, disagreement = [], [], []
        for step in range(forced):
            window = slice(step * top, (step + 1) * top)
            divergence, variation, mass = position_divergence(ref["top_logprobs"][window],
                                                              observed[window])
            kl.append(divergence)
            tv.append(variation)
            mass_total += mass
            positions += 1
            # Observed[0] is the reference's most probable token; the run agrees when that token
            # holds the run's largest log-probability.
            disagreement.append(0 if observed[step * top] == best[step] else 1)
        out.append(ForcedSegment(segment["prompt_tokens"], kl, tv, disagreement,
                                 -sum(segment["logprobs"])))
    return out, mass_total / positions


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def evaluate_forced(default_path: Path, fast_path: Path, reference_path: Path,
                    policy: Policy = POLICY, *, fast_route: bool = True) -> Verdict:
    """The decode criteria: the prefill criteria at teacher-forced decode positions.

    The floor is the default run's own decode divergence from the reference. With fast_route=False
    (compare-decode) any two runs of one reference are compared, for controls.
    """
    default, fast = load_forced(default_path), load_forced(fast_path)
    reference = load_forced(reference_path)
    for key in ("mode", "kv_dtype", "forced_tokens"):
        if default[key] != fast[key] and fast_route:
            raise MeasurementError(f"the teacher-forced runs differ in {key}")
    if fast_route and (default["fast_paths"] or not fast["fast_paths"]):
        raise MeasurementError("expected an exact-route run and a run with fast paths active")
    if reference.get("kind") != "reference" or reference["forced_tokens"] != default["forced_tokens"]:
        raise MeasurementError(f"{reference_path} is not a reference for these runs")
    a, coverage = forced_segments(default, reference)
    b, _ = forced_segments(fast, reference)
    if len(a) < policy.min_units:
        raise MeasurementError(f"{len(a)} segments; the bound needs at least {policy.min_units}")
    if coverage < policy.coverage_minimum:
        raise MeasurementError(
            f"the reference's top tokens cover {coverage:.3f} of its distribution; the KL bound "
            f"needs at least {policy.coverage_minimum}")
    kl_units = [PairedUnit(len(d.kl), sum(d.kl), sum(f.kl)) for d, f in zip(a, b)]
    tv_units = [PairedUnit(len(d.tv), sum(d.tv), sum(f.tv)) for d, f in zip(a, b)]
    top1_units = [PairedUnit(len(d.kl), sum(d.disagreement), sum(f.disagreement))
                  for d, f in zip(a, b)]
    nll_units = [PairedUnit(len(d.kl), d.nll, f.nll) for d, f in zip(a, b)]
    positions = sum(u.weight for u in kl_units)
    floor_kl = sum(u.default for u in kl_units) / positions
    floor_tv = sum(u.default for u in tv_units) / positions
    floor_top1 = sum(u.default for u in top1_units) / positions
    verdict = Verdict(scope=f"teacher-forced decode {default['mode']} {fast_path.parent}")
    verdict.facts = {
        "mode": default["mode"],
        "kv_dtype": default["kv_dtype"],
        "segments": len(a),
        "forced_positions_per_route": positions,
        "reference_top_mass": coverage,
        "floor_kl": floor_kl,
        "floor_tv": floor_tv,
        "floor_top1_disagreement": floor_top1,
        "fast_mean_kl": sum(u.fast for u in kl_units) / positions,
        "fast_mean_tv": sum(u.fast for u in tv_units) / positions,
        "fast_top1_disagreement": sum(u.fast for u in top1_units) / positions,
        "default_mean_nll": sum(u.default for u in nll_units) / positions,
        "fast_mean_nll": sum(u.fast for u in nll_units) / positions,
        "underflows": [default.get("underflows", 0), fast.get("underflows", 0)],
        "fast_identical_to_default": all(
            d["logprobs"] == f["logprobs"] for d, f in zip(default["segments"], fast["segments"])),
    }
    # The bounded criteria. Total variation, not KL, carries the distribution criterion: KL is
    # unbounded per position, so the few positions whose most probable token flips under any
    # rounding change (a few nats each) set its standard error, and no feasible sample resolves a
    # 10 % budget (section 6.4). Total variation counts each such position at most once.
    verdict.criteria.append(bounded(
        "decode mean TV excess", tv_units,
        clamp(policy.relative_budget * floor_tv, policy.tv_budget_minimum,
              policy.tv_budget_maximum), policy, f"floor {floor_tv:.6g}"))
    verdict.criteria.append(bounded(
        "decode top-1 disagreement excess", top1_units,
        clamp(policy.relative_budget * floor_top1, policy.top1_budget_minimum,
              policy.top1_budget_maximum), policy, f"floor {floor_top1:.6g}"))
    # Gross caps on point estimates, their bounds reported: KL and NLL (NLL moves at first order,
    # as in prefill), the far tail, and each prompt length.
    cap = max(policy.bucket_relative * floor_kl, policy.kl_budget_minimum)
    for name, units in (("decode mean KL excess", kl_units), ("decode NLL excess", nll_units)):
        mean, bound = paired_excess(units, policy.z)
        verdict.criteria.append(Criterion(name, mean, mean, cap, mean <= cap,
                                          f"95% bound {bound:.4e}"))
    default_tail = quantile([k for s in a for k in s.kl], 0.999)
    fast_tail = quantile([k for s in b for k in s.kl], 0.999)
    tail_ratio = fast_tail / default_tail if default_tail > 0 else (0.0 if fast_tail == 0 else math.inf)
    verdict.criteria.append(Criterion("decode p99.9 KL ratio", tail_ratio, tail_ratio,
                                      policy.tail_ratio, tail_ratio <= policy.tail_ratio,
                                      f"default {default_tail:.6g}, fast {fast_tail:.6g}"))
    by_prompt: dict[int, list[float]] = {}
    for d, f in zip(a, b):
        totals = by_prompt.setdefault(d.prompt_tokens, [0.0, 0.0, 0.0])
        totals[0] += len(d.tv)
        totals[1] += sum(d.tv)
        totals[2] += sum(f.tv)
    for prompt in sorted(by_prompt):
        n, d, f = by_prompt[prompt]
        excess = (f - d) / n
        bucket_cap = max(policy.bucket_relative * d / n, policy.tv_budget_minimum)
        verdict.criteria.append(
            Criterion(f"prompt {prompt} decode TV excess", excess, excess, bucket_cap,
                      excess <= bucket_cap, f"{int(n)} positions, default {d / n:.6g}"))
    return verdict


def evaluate_free_running(report_path: Path, policy: Policy = POLICY) -> Verdict:
    """Greedy continuations, each judged on its own prefix against the reference."""
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    runs = report["runs"]
    if len(runs) != 2:
        raise MeasurementError("the judge report must hold exactly the default and the fast run")
    default, fast = runs[0]["segments"], runs[1]["segments"]
    if [s["offset"] for s in default] != [s["offset"] for s in fast]:
        raise MeasurementError("default and fast runs generated from different segments")
    if len(default) < policy.min_units:
        raise MeasurementError(
            f"{len(default)} segments; the bound needs at least {policy.min_units}")
    for s in default + fast:
        if s["tokens"] <= 0:
            raise MeasurementError("a segment generated no tokens")
    verdict = Verdict(scope=f"free-running decode {report_path}")
    # Per-token rates of each build, weighted by the segment's mean length.
    disagreement = [
        PairedUnit(
            (d["tokens"] + f["tokens"]) / 2,
            (1.0 - d["agree"] / d["tokens"]) * (d["tokens"] + f["tokens"]) / 2,
            (1.0 - f["agree"] / f["tokens"]) * (d["tokens"] + f["tokens"]) / 2,
        )
        for d, f in zip(default, fast)
    ]
    regret = [
        PairedUnit(
            (d["tokens"] + f["tokens"]) / 2,
            d["regret"] / d["tokens"] * (d["tokens"] + f["tokens"]) / 2,
            f["regret"] / f["tokens"] * (d["tokens"] + f["tokens"]) / 2,
        )
        for d, f in zip(default, fast)
    ]
    tokens = sum(u.weight for u in disagreement)
    floor_disagreement = sum(u.default for u in disagreement) / tokens
    floor_regret = sum(u.default for u in regret) / tokens
    verdict.facts = {
        "reference": report.get("reference"),
        "segments": len(default),
        "tokens_per_route": tokens,
        "default_disagreement": floor_disagreement,
        "fast_disagreement": sum(u.fast for u in disagreement) / tokens,
        "default_regret": floor_regret,
        "fast_regret": sum(u.fast for u in regret) / tokens,
    }
    budget = clamp(policy.relative_budget * floor_disagreement, policy.top1_budget_minimum,
                   policy.top1_budget_maximum)
    verdict.criteria.append(bounded("reference-top disagreement excess", disagreement, budget,
                                    policy, f"floor {floor_disagreement:.6g}"))
    budget = max(policy.relative_budget * floor_regret, policy.regret_budget_minimum)
    verdict.criteria.append(bounded("mean regret excess", regret, budget, policy,
                                    f"floor {floor_regret:.6g} nats"))
    return verdict


# --- default-route identity ----------------------------------------------------------------------


def window_values(report: dict) -> dict[tuple[str, int], tuple]:
    values = {}
    for stream in report["streams"]:
        for window in stream["windows"]:
            divergence = window.get("kl_divergence") or {}
            values[(stream["id"], window["index"])] = (
                window["scored_tokens"],
                window["total_nll"],
                divergence.get("mean_kl"),
                divergence.get("top1_agreement"),
            )
    return values


def identical(first: Path, second: Path) -> tuple[bool, str]:
    """Whether two default-route measurements agree exactly, and the first difference."""
    if first.is_dir() and second.is_dir():
        with (first / "report.json").open(encoding="utf-8") as handle:
            a = json.load(handle)
        with (second / "report.json").open(encoding="utf-8") as handle:
            b = json.load(handle)
        for key in ("context_tokens", "stride_tokens", "kv_dtype", "prefill_chunk_tokens"):
            if a["execution"][key] != b["execution"][key]:
                raise MeasurementError(f"the runs differ in {key}; compare like with like")
        if a["execution"].get("fast_paths") or b["execution"].get("fast_paths"):
            raise MeasurementError("identity applies to the default route; a run used fast paths")
        va, vb = window_values(a), window_values(b)
        if va.keys() != vb.keys():
            return False, "the runs scored different windows"
        for key in sorted(va):
            if va[key] != vb[key]:
                return False, f"window {key}: {va[key]} != {vb[key]}"
        return True, f"{len(va)} windows identical"
    if first.suffix == ".json" and second.suffix == ".json":
        a, b = load_forced(first), load_forced(second)
        if a["fast_paths"] or b["fast_paths"]:
            raise MeasurementError("identity applies to the default route; a run used fast paths")
        if len(a["segments"]) != len(b["segments"]):
            return False, "the runs forced different numbers of segments"
        for index, (x, y) in enumerate(zip(a["segments"], b["segments"])):
            if x != y:
                return False, f"segment {index} differs"
        return True, f"{len(a['segments'])} segments identical"
    if first.is_file() and second.is_file():
        a_lines = first.read_text(encoding="utf-8").splitlines()
        b_lines = second.read_text(encoding="utf-8").splitlines()
        if len(a_lines) != len(b_lines):
            return False, f"{len(a_lines)} != {len(b_lines)} segments"
        for index, (x, y) in enumerate(zip(a_lines, b_lines)):
            if x != y:
                return False, f"segment {index} differs"
        return True, f"{len(a_lines)} segments identical"
    raise MeasurementError("compare two run directories, two forced runs or two generation files")


# --- running -------------------------------------------------------------------------------------


def default_tool(name: str) -> str:
    for candidate in (
        Path("build-windows/apps/Release") / f"{name}.exe",
        Path("build-windows/bench/Release") / f"{name}.exe",
        Path("build/apps") / name,
        Path("build/bench") / name,
    ):
        if candidate.is_file():
            return str(candidate)
    return name


def run(command: list[str], log: Path) -> None:
    print("  $ " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise MeasurementError(f"command failed ({result.returncode}); see {log}")


def run_prefill(args: argparse.Namespace) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workload = ["--context", str(args.context), "--stride", str(args.stride), "--log-level",
                "warning"]
    workload += ["--corpus", args.corpus] if args.corpus else ["--text", args.text]
    if args.quick:
        workload.append("--quick")
    if args.ngram_volume:
        workload += ["--ngram-volume", args.ngram_volume]
    common = [args.perplexity, args.artifact] + workload
    # 16-bit activations, BF16 KV, one pass per window.
    one_pass = ["--kv-dtype", "bf16", "--prefill-chunk", str(args.context), "--a16-activations"]
    reference = out / "reference.top"
    if not (out / "reference" / "report.json").is_file():
        run(common + one_pass + ["--save-top-tokens", str(reference), "--output",
                                 str(out / "reference")],
            out / "reference.log")
    for kv in args.kv:
        for chunk in args.chunk:
            for route, extra in (("default", ["--disable-fast-paths"]),
                                 ("fast", candidate_options(args, "--fast-path"))):
                target = out / kv / f"{route}-c{chunk}"
                if (target / "report.json").is_file():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # A schedule fast path's fast run may prefill at another width than the default
                # run it pairs with (--fast-chunk); the pair keeps the default run's chunk name.
                width = args.fast_chunk if route == "fast" and args.fast_chunk else chunk
                run(common + ["--kv-dtype", kv, "--prefill-chunk", str(width), "--kl-reference",
                              str(reference), "--output", str(target)] + extra,
                    out / kv / f"{route}-c{chunk}.log")
    return out


def candidate_options(args: argparse.Namespace, option: str) -> list[str]:
    """The fast run's options: every qualified path, plus each --fast-path candidate."""
    out = []
    for name in args.fast_path:
        out += [option, name]
    return out


def corpus_text(manifest: Path, out: Path) -> Path:
    """The manifest's streams joined into one UTF-8 file, the decode tools' segment source."""
    target = out / "corpus.txt"
    if not target.is_file():
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        parts = [(manifest.parent / s["path"]).read_text(encoding="utf-8")
                 for s in spec["streams"]]
        target.write_text("\n\n".join(parts), encoding="utf-8")
    return target


def run_decode(args: argparse.Namespace) -> list[Verdict]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = args.text or str(corpus_text(Path(args.corpus), out))
    volume = ["--ngram-volume", args.ngram_volume] if args.ngram_volume else []
    segments = ["--segments", str(args.segments), "--prompts",
                ",".join(str(p) for p in args.prompts)]
    # The arm: both routes run the same Engine conditions, so the request-coupling paths act in
    # the fast run (other requests in the round, a short KV pool, prefix-cache resumes).
    arm = ["--concurrency", str(args.concurrency)]
    if args.kv_tokens:
        arm += ["--kv-tokens", str(args.kv_tokens)]
    if args.prefix_reuse:
        arm += ["--prefix-reuse"]
    if args.warm_tokens:
        arm += ["--warm-tokens", str(args.warm_tokens)]
    # The reference: the 16-bit-activation, BF16-KV scoring route at the same positions.
    reference = out / "reference.json"
    if not reference.is_file():
        run([args.forced, args.artifact, text, str(reference), "--reference"] + segments + volume,
            out / "reference.log")
    for route, extra in (("default", ["--disable-fast-paths"]),
                         ("fast", candidate_options(args, "--fast-path"))):
        target = out / f"{route}-forced.json"
        if target.is_file():
            continue
        run([args.forced, args.artifact, text, str(target), "--observe", str(reference),
             "--mode", args.mode, "--kv-dtype", args.kv] + segments + arm + volume + extra,
            out / f"{route}-forced.log")
    verdicts = [evaluate_forced(out / "default-forced.json", out / "fast-forced.json", reference)]
    if args.free_running:
        modes = [] if args.mode == "dflash2" else [args.mode]
        for route, extra in (("default", ["exact", f"kv={args.kv}"]),
                             ("fast", [f"path={p}" for p in args.fast_path] + [f"kv={args.kv}"])):
            target = out / f"{route}.txt"
            if target.is_file():
                continue
            volume = [f"ngram-volume={args.ngram_volume}"] if args.ngram_volume else []
            run([args.gen, args.artifact, text, str(target), str(args.free_running_segments),
                 "1536", "384"] + modes + extra + volume,
                out / f"{route}.log")
        report = out / "judge.json"
        if not report.is_file():
            volume = ["--ngram-volume", args.ngram_volume] if args.ngram_volume else []
            run([args.judge, "--report", str(report)] + volume + [args.artifact, text,
                 str(out / "default.txt"), str(out / "fast.txt")],
                out / "judge.log")
        verdicts.append(evaluate_free_running(report))
    return verdicts


# --- reporting -----------------------------------------------------------------------------------


def print_verdict(verdict: Verdict) -> None:
    print(f"\n{verdict.scope}")
    for key, value in verdict.facts.items():
        print(f"  {key}: {value}")
    print(f"  {'criterion':<36}{'estimate':>13}{'95% bound':>13}{'budget':>13}  result")
    for c in verdict.criteria:
        print(f"  {c.name:<36}{c.estimate:>13.4e}{c.bound:>13.4e}{c.budget:>13.4e}  "
              f"{'PASS' if c.passed else 'FAIL'}  {c.detail}")
    print(f"  => {'PASS' if verdict.passed else 'FAIL'}")


def finish(verdicts: Iterable[Verdict], output: Path | None) -> int:
    verdicts = list(verdicts)
    for verdict in verdicts:
        print_verdict(verdict)
    if output is not None:
        output.write_text(json.dumps([v.to_json() for v in verdicts], indent=2) + "\n",
                          encoding="utf-8")
    return 0 if all(v.passed for v in verdicts) else 1


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    prefill = commands.add_parser("prefill", help="run and decide the prefill criteria")
    prefill.add_argument("artifact")
    source = prefill.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus", help="perplexity corpus manifest")
    source.add_argument("--text", help="one UTF-8 file")
    prefill.add_argument("--quick", action="store_true")
    prefill.add_argument("--context", type=int, default=14336)
    prefill.add_argument("--stride", type=int, default=7168)
    prefill.add_argument("--kv", nargs="+", default=["int8"], help="KV formats (default int8)")
    prefill.add_argument("--chunk", nargs="+", type=int, default=[896, 1792, 3584, 7168],
                         help="prefill chunk sizes (default 896 1792 3584 7168)")
    prefill.add_argument("--perplexity", default=default_tool("infernix-perplexity"))
    prefill.add_argument("--ngram-volume", help="Qwen3.8-Flash-Next n-gram volume")
    prefill.add_argument("--fast-path", action="append", default=[], choices=FAST_PATHS,
                         help="a candidate to qualify, run with the qualified paths")
    prefill.add_argument("--fast-chunk", type=int, default=0,
                         help="prefill chunk of every fast run, each paired with the default run "
                              "at its --chunk (a schedule fast path that widens prefill steps); "
                              "use a separate --out")
    prefill.add_argument("--out", required=True, help="run directory (reused runs are skipped)")

    decode = commands.add_parser("decode", help="run and decide the decode criteria")
    decode.add_argument("artifact")
    decode_source = decode.add_mutually_exclusive_group()
    decode_source.add_argument("--corpus", default="eval/corpora/decode-4m/manifest.json",
                               help="corpus manifest whose streams, joined, the segments are cut "
                                    "from (default: the decode corpus infernix-decode-v1)")
    decode_source.add_argument("--text", help="one UTF-8 file the segments are cut from")
    decode.add_argument("--mode", choices=("plain", "dflash2", "mtp", "ngram"), default="plain")
    decode.add_argument("--kv", default="int8", choices=KV_FORMATS)
    decode.add_argument("--segments", type=int, default=1024)
    decode.add_argument("--concurrency", type=int, default=1,
                        help="requests per round in both runs (the batch-coupled paths act above 1)")
    decode.add_argument("--kv-tokens", type=int, default=0,
                        help="KV pool tokens (default: every lane's context); less preempts")
    decode.add_argument("--prefix-reuse", action="store_true",
                        help="resume each segment from a warm request's prefix-cache entry")
    decode.add_argument("--warm-tokens", type=int, default=0,
                        help="the warm request's prompt length (default: the whole prompt)")
    decode.add_argument("--prompts", type=int, nargs="+", default=[512, 2048, 8192],
                        help="prompt lengths the segments cycle through")
    decode.add_argument("--free-running", action="store_true",
                        help="also judge greedy continuations on their own prefixes")
    decode.add_argument("--free-running-segments", type=int, default=128)
    decode.add_argument("--fast-path", action="append", default=[], choices=FAST_PATHS,
                        help="a candidate to qualify, run with the qualified paths")
    decode.add_argument("--ngram-volume", help="Qwen3.8-Flash-Next n-gram volume")
    decode.add_argument("--forced", default=default_tool("infernix_decode_quality_forced"))
    decode.add_argument("--gen", default=default_tool("infernix_decode_quality_gen"))
    decode.add_argument("--judge", default=default_tool("infernix_decode_quality_judge"))
    decode.add_argument("--out", required=True)

    evaluate = commands.add_parser("evaluate-prefill", help="decide an existing prefill run")
    evaluate.add_argument("directory",
                          help="a prefill run directory, or one KV format's subdirectory")

    compare = commands.add_parser("compare-prefill",
                                  help="decide two sets of runs of one reference, paired in order")
    compare.add_argument("--base", nargs="+", type=Path, required=True)
    compare.add_argument("--candidate", nargs="+", type=Path, required=True)

    forced = commands.add_parser("evaluate-decode", help="decide two teacher-forced decode runs")
    forced.add_argument("default", type=Path)
    forced.add_argument("fast", type=Path)
    forced.add_argument("--reference", type=Path, required=True,
                        help="the reference both runs observed (--reference of the forced tool)")

    compare_forced = commands.add_parser(
        "compare-decode", help="decide any two teacher-forced runs of one reference (controls)")
    compare_forced.add_argument("--base", type=Path, required=True)
    compare_forced.add_argument("--candidate", type=Path, required=True)
    compare_forced.add_argument("--reference", type=Path, required=True)

    judged = commands.add_parser("evaluate-free-running", help="decide an existing judge report")
    judged.add_argument("report")

    for sub in (prefill, decode, evaluate, compare, forced, compare_forced, judged):
        sub.add_argument("--json", type=Path, help="also write the verdicts as JSON")

    same = commands.add_parser("identical", help="check two default-route runs for bit-exactness")
    same.add_argument("first")
    same.add_argument("second")
    return parser.parse_args(argv)


def prefill_verdicts(directory: Path) -> list[Verdict]:
    if any(d.is_dir() for d in directory.glob("default-c*")):
        return [evaluate_prefill(directory)]
    kv_dirs = sorted(d for d in directory.iterdir()
                     if d.is_dir() and any(c.is_dir() for c in d.glob("default-c*")))
    if not kv_dirs:
        raise MeasurementError(f"no prefill runs under {directory}")
    return [evaluate_prefill(d) for d in kv_dirs]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "prefill":
            return finish(prefill_verdicts(run_prefill(args)), args.json)
        if args.command == "decode":
            return finish(run_decode(args), args.json)
        if args.command == "evaluate-prefill":
            return finish(prefill_verdicts(Path(args.directory)), args.json)
        if args.command == "compare-prefill":
            pairs = compared_pairs(args.base, args.candidate)
            return finish([evaluate_prefill_pairs(pairs, "compare-prefill")], args.json)
        if args.command == "evaluate-decode":
            return finish([evaluate_forced(args.default, args.fast, args.reference)], args.json)
        if args.command == "compare-decode":
            return finish([evaluate_forced(args.base, args.candidate, args.reference,
                                           fast_route=False)], args.json)
        if args.command == "evaluate-free-running":
            return finish([evaluate_free_running(Path(args.report))], args.json)
        same, detail = identical(Path(args.first), Path(args.second))
        print(f"{'IDENTICAL' if same else 'DIFFERENT'}: {detail}")
        return 0 if same else 1
    except MeasurementError as error:
        print(f"fast_path_gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
