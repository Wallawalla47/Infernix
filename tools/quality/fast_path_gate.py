"""Fast-path numerical acceptance gate.

Decides whether an artifact's fast paths keep it within Infernix's numerical acceptance policy
(docs/maintainer/model-quality.md, section 6). The policy compares two Engine routes against one
fixed reference distribution: the reproducible route (--reproducible), in the default role, and the
fast-path route (every path qualified for the configuration plus any --fast-path candidates, so a
new path is always measured together with the paths it will run beside). The fast paths of a
configuration together may move the output away from the reference by at most one fixed budget per
measure, the same for every artifact and KV format (about what llama.cpp's q8_0 costs against
FP16); the floor (the default route's own divergence from the reference) is reported, not used.
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
  concat-forced    join teacher-forced runs or references of consecutive segment ranges (the forced
                   tool's --first-segment), so a 1,024-segment gate extends a 256-segment one

Every deciding subcommand takes --policy NAME=VALUE (any Policy constant, repeatable) to decide under
another budget; runs found on disk are reused, so stored runs are decided again without measuring.

Exit status: 0 every criterion passes, 1 a criterion fails, 2 the measurement is unusable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Policy:
    """The acceptance policy's constants (docs/maintainer/model-quality.md, section 6.3)."""

    # One budget per measure for every configuration (since 2026-10-16), for the excess of all its
    # fast paths together over the exact route. Mean KL excess (nats): 1.5e-3, about what
    # llama.cpp's q8_0 costs against FP16 on Llama 3 8B (1.36e-3) and below Strata's adopted
    # defaults (4e-3, 5.8e-3) and the "distribution-lossless" band (2e-3 to 4e-3).
    kl_budget: float = 1.5e-3
    # Decode's bounded criterion is total variation (bounded by 1 per position, so a position whose
    # most probable token flips cannot dominate the mean as an unbounded KL does): the KL budget at
    # the decode excess TV / KL ratio measured on recipe C (about 2.3), rounded up.
    tv_budget: float = 4.0e-3
    # Top-1 disagreement excess (fractions of positions or tokens): 0.5 points.
    top1_budget: float = 5.0e-3
    # Free-running decode regret (nats per token): the KL budget.
    regret_budget: float = 1.5e-3
    # Gross caps on point estimates: the corpus NLL excess and each prefill context bucket at twice
    # the KL budget; each decode prompt length's TV excess at twice the TV budget; the decode KL
    # and forced-token NLL excess at decode_kl_cap, since their standard error is about 2-4e-3 even
    # over 1,024 segments (knife-edge positions; total variation carries decode's decision), so a
    # cap must sit about three standard errors above zero; the 99.9th-percentile KL at most 1.25
    # times the default's, or a fast p99.9 of at most tail_minimum (the tail of a route at the KL
    # budget: p99.9 / mean measured 48 in prefill and 21 in decode on recipe C), so a default whose
    # tail is 0 does not fail every path.
    point_cap_factor: float = 2.0
    decode_kl_cap: float = 1.0e-2
    tail_ratio: float = 1.25
    tail_minimum: float = 0.05
    bucket_min_positions: int = 20000
    # The KL lower bound must resolve at least this share of the reference distribution.
    coverage_minimum: float = 0.90
    # One-sided 95 % normal quantile and the fewest independent units it is applied to.
    z: float = 1.6448536269514722
    min_units: int = 30


POLICY = Policy()


def policy_with(overrides: Sequence[str]) -> Policy:
    """POLICY with --policy NAME=VALUE overrides, so stored runs are decided again under other
    budgets without being measured again."""
    kinds = {f.name: f.type for f in fields(Policy)}
    changes: dict[str, float | int] = {}
    for item in overrides:
        name, separator, value = item.partition("=")
        if not separator or name not in kinds:
            raise MeasurementError(f"--policy {item}: expected NAME=VALUE, NAME one of "
                                   f"{', '.join(kinds)}")
        try:
            changes[name] = int(value) if kinds[name] == "int" else float(value)
        except ValueError as error:
            raise MeasurementError(f"--policy {item}: {error}") from error
    return replace(POLICY, **changes)

# fast_path_name of every infernix::FastPath.
FAST_PATHS = ("prompt-pv8", "decode-split64", "batched-decode", "batched-a8-decode",
              "batched-a4-decode", "batched-a4-residual-decode", "batched-a8-residual-decode",
              "single-column-decode", "expert-a4-residual", "expert-a4-residual-3", "batch-split-attention",
              "adaptive-speculation", "free-cache-resume", "preemption", "q8-sliced-k", "gdn-conv-fused",
              "qsa-prepare-fused", "adaptive-mtp", "adaptive-dflash", "idle-wide-prefill", "fp8-a8-pair", "q4-w4a8",
              "sampled-mtp-drafts", "gdn-f16-accumulate", "ggml-bf16-prefill")


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


def same_artifact(first: str, second: str) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return (os.path.normcase(os.path.abspath(first)) ==
                os.path.normcase(os.path.abspath(second)))


def check_pair(pair: ChunkPair, *, fast_route: bool, fast_chunk: int = 0) -> None:
    """fast_chunk: the fast run prefills at this width instead of its default run's chunk
    (prefill --fast-chunk, a schedule fast path that widens prefill steps)."""
    a, b = pair.default, pair.fast
    for key in ("context_tokens", "stride_tokens"):
        if a["execution"][key] != b["execution"][key]:
            raise MeasurementError(f"{pair.label}: the runs differ in {key}")
    if a["kl_divergence"]["reference"] != b["kl_divergence"]["reference"]:
        raise MeasurementError(f"{pair.label}: the runs used different references")
    if fast_route:
        # Both routes of one artifact (compare-prefill may set two artifacts against one reference).
        if (not same_artifact(a["artifact"]["path"], b["artifact"]["path"]) or
                a["artifact"]["prefill_signature"] != b["artifact"]["prefill_signature"]):
            raise MeasurementError(f"{pair.label}: the default and fast runs scored other artifacts")
        if a["execution"]["kv_dtype"] != b["execution"]["kv_dtype"]:
            raise MeasurementError(f"{pair.label}: default and fast differ in kv_dtype")
        expected = fast_chunk or a["execution"]["prefill_chunk_tokens"]
        if b["execution"]["prefill_chunk_tokens"] != expected:
            raise MeasurementError(
                f"{pair.label}: the fast run prefills at {b['execution']['prefill_chunk_tokens']}"
                f" tokens, expected {expected}")
        if a["execution"].get("fast_paths") or not b["execution"].get("fast_paths"):
            raise MeasurementError(
                f"{pair.label}: expected an exact-route run and a run with fast paths active")


def prefill_pairs(directory: Path, fast_chunk: int = 0) -> list[ChunkPair]:
    pairs = []
    for default_dir in sorted(d for d in directory.glob("default-c*") if d.is_dir()):
        chunk = default_dir.name.removeprefix("default-c")
        pairs.append(ChunkPair(f"chunk {chunk}", load_report(default_dir),
                               load_report(directory / f"fast-c{chunk}")))
    if not pairs:
        raise MeasurementError(f"no default-c<chunk> runs in {directory}")
    for pair in pairs:
        check_pair(pair, fast_route=True, fast_chunk=fast_chunk)
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

    verdict.criteria.append(bounded(
        "mean KL excess", [PairedUnit(t[0], t[1], t[2]) for t in units.values()], policy.kl_budget,
        policy, f"floor {floor_kl:.6g}"))
    verdict.criteria.append(bounded(
        "top-1 disagreement excess", [PairedUnit(t[0], t[3], t[4]) for t in units.values()],
        policy.top1_budget, policy, f"floor {floor_top1:.6g}"))
    # Corpus NLL moves at first order under any perturbation of the logits, where the KL excess
    # moves at second order: a pure reordering shifted it by 1.7e-3 on the 27B NVFP4 artifact
    # (section 6.4). It is therefore a gross cap on the point estimate, for what truncated KL cannot
    # see (probability moved within the unsaved tail); its bound is reported.
    nll_mean, nll_bound = paired_excess(
        [PairedUnit(t[0], t[5], t[6]) for t in units.values()], policy.z)
    nll_cap = policy.point_cap_factor * policy.kl_budget
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
                  tail_ratio <= policy.tail_ratio or fast_tail <= policy.tail_minimum,
                  f"default {default_tail:.6g}, fast {fast_tail:.6g}; "
                  f"a fast p99.9 to {policy.tail_minimum:g} passes"))
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
        bucket_budget = policy.point_cap_factor * policy.kl_budget
        verdict.criteria.append(
            Criterion(f"context [{begin},{end}) KL excess", excess, excess, bucket_budget,
                      excess <= bucket_budget, f"{int(n)} positions, default {d / n:.6g}"))
    return verdict


def evaluate_prefill(directory: Path, policy: Policy = POLICY, fast_chunk: int = 0) -> Verdict:
    return evaluate_prefill_pairs(prefill_pairs(directory, fast_chunk), f"prefill {directory}",
                                  policy)


# --- decode --------------------------------------------------------------------------------------


def load_forced(path: Path) -> dict:
    if not path.is_file():
        raise MeasurementError(f"missing teacher-forced run: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


# The fields every part of a joined run or reference shares (the forced tool's header).
CONCAT_FIELDS = {"reference": ("artifact", "kind", "forced_tokens", "top_tokens"),
                 "run": ("artifact", "mode", "kv_dtype", "fast_paths", "forced_tokens")}
CONCAT_COUNTERS = ("underflows", "reused_prompt_tokens", "preemptions", "replayed_tokens")


def concat_forced(parts: Sequence[Path], out: Path) -> dict:
    """Consecutive ranges of one segment set (the forced tool's --first-segment) as one file.

    Runs and references alike: each part must share the first part's header and start where the
    previous one ended (the corpus token after its last segment's prompt and forced tokens), so the
    result holds exactly the segments, offsets and observations of one run over the whole range; a
    run's counters are summed. A larger gate set extends a smaller one this way (the first 256 of
    1,024 segments are the 256-segment set) instead of computing its first part again.
    """
    loaded = [load_forced(path) for path in parts]
    kind = "reference" if loaded[0].get("kind") == "reference" else "run"
    joined = dict(loaded[0])
    joined["segments"] = []
    for path, part in zip(parts, loaded):
        for name in CONCAT_FIELDS[kind]:
            if part.get(name) != loaded[0].get(name):
                raise MeasurementError(f"{path} differs from {parts[0]} in {name}")
        if not part["segments"]:
            raise MeasurementError(f"{path} holds no segments")
        if joined["segments"]:
            last = joined["segments"][-1]
            expected = last["offset"] + last["prompt_tokens"] + loaded[0]["forced_tokens"]
            if part["segments"][0]["offset"] != expected:
                raise MeasurementError(f"{path} starts at corpus token {part['segments'][0]['offset']}, "
                                       f"not {expected} after the previous part")
        joined["segments"] += part["segments"]
    if kind == "run":
        for counter in CONCAT_COUNTERS:
            if counter in joined:
                joined[counter] = sum(part.get(counter, 0) for part in loaded)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(joined, handle)
        handle.write("\n")
    return joined


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
    if fast_route and not all(same_artifact(run["artifact"], reference["artifact"])
                              for run in (default, fast)):
        raise MeasurementError("the reference and the teacher-forced runs scored other artifacts")
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
    # KL budget (section 6.4). Total variation counts each such position at most once.
    verdict.criteria.append(bounded(
        "decode mean TV excess", tv_units, policy.tv_budget, policy, f"floor {floor_tv:.6g}"))
    verdict.criteria.append(bounded(
        "decode top-1 disagreement excess", top1_units, policy.top1_budget, policy,
        f"floor {floor_top1:.6g}"))
    # Gross caps on point estimates, their bounds reported: KL and NLL (NLL moves at first order,
    # as in prefill), the far tail, and each prompt length.
    cap = policy.decode_kl_cap
    for name, units in (("decode mean KL excess", kl_units), ("decode NLL excess", nll_units)):
        mean, bound = paired_excess(units, policy.z)
        verdict.criteria.append(Criterion(name, mean, mean, cap, mean <= cap,
                                          f"95% bound {bound:.4e}"))
    default_tail = quantile([k for s in a for k in s.kl], 0.999)
    fast_tail = quantile([k for s in b for k in s.kl], 0.999)
    tail_ratio = fast_tail / default_tail if default_tail > 0 else (0.0 if fast_tail == 0 else math.inf)
    verdict.criteria.append(Criterion("decode p99.9 KL ratio", tail_ratio, tail_ratio,
                                      policy.tail_ratio,
                                      tail_ratio <= policy.tail_ratio or fast_tail <= policy.tail_minimum,
                                      f"default {default_tail:.6g}, fast {fast_tail:.6g}; "
                                      f"a fast p99.9 to {policy.tail_minimum:g} passes"))
    by_prompt: dict[int, list[float]] = {}
    for d, f in zip(a, b):
        totals = by_prompt.setdefault(d.prompt_tokens, [0.0, 0.0, 0.0])
        totals[0] += len(d.tv)
        totals[1] += sum(d.tv)
        totals[2] += sum(f.tv)
    for prompt in sorted(by_prompt):
        n, d, f = by_prompt[prompt]
        excess = (f - d) / n
        bucket_cap = policy.point_cap_factor * policy.tv_budget
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
    verdict.criteria.append(bounded("reference-top disagreement excess", disagreement,
                                    policy.top1_budget, policy, f"floor {floor_disagreement:.6g}"))
    verdict.criteria.append(bounded("mean regret excess", regret, policy.regret_budget, policy,
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


DEFAULT_REFERENCE_CACHE = Path("profiles/quality/reference-cache")


def file_digest(*paths: Path) -> str:
    """SHA-256 of the files' contents, in order."""
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_identity(artifact: str) -> dict:
    """The artifact as a reference depends on it: its file, and the size and modification time
    that change whenever it is rewritten (published artifacts are never rewritten in place)."""
    path = Path(artifact).resolve()
    stat = path.stat()
    return {"path": os.path.normcase(str(path)), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# The lossless route's definition (docs/maintainer/lossless-route.md), every reference's route: its bits
# depend on the request and the artifact alone, so a reference is keyed by this version instead of the
# build, and survives rebuilds. Bump it with any change to what the route computes.
LOSSLESS_DEFINITION = "2026-10-21"


def corpus_identity(corpus: str | None, text: str | None) -> str:
    """The content of the text a reference scores (a manifest with its streams, or one file)."""
    if text:
        return file_digest(Path(text))
    manifest = Path(corpus)
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    return file_digest(manifest, *[manifest.parent / s["path"] for s in spec["streams"]])


def ensure_reference(out: Path, names: Sequence[str], key: dict, produce: Callable[[], None],
                     cache: Path | None, recompute: bool, dependents: Sequence[Path] = ()) -> str:
    """Puts the reference files `names` (relative to `out`) in place and says where they came from.

    A reference is deterministic in its key (the artifact, the corpus content and every setting
    that shapes it, never the KV format, mode or candidate), so a run directory takes it from the
    shared cache when an entry for the key exists, linked (or copied) in, and stores what it
    computes there. recompute (--recompute-reference) recomputes it and replaces the cache entry,
    for a change to the lossless route that the key's definition does not record; it refuses while
    `dependents` (runs already scored against the directory's reference) exist.
    """
    present = all((out / name).exists() for name in names)
    if present and not recompute:
        return "kept"
    if recompute:
        if dependents:
            raise MeasurementError(
                f"{out} holds runs scored against its reference ({', '.join(map(str, dependents))}); "
                "remove them or use a fresh --out to recompute the reference")
        for name in names:
            remove_path(out / name)
    entry = None
    if cache is not None:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:32]
        entry = cache / digest
        if not recompute and (entry / "key.json").is_file():
            for name in names:
                place(entry / name, out / name)
            return f"reused {entry}"
    produce()
    if entry is not None:
        staging = entry.with_name(entry.name + f".tmp{os.getpid()}")
        remove_path(staging)
        staging.mkdir(parents=True)
        for name in names:
            place(out / name, staging / name)
        (staging / "key.json").write_text(json.dumps(key, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
        remove_path(entry)
        staging.rename(entry)
        return f"computed, stored in {entry}"
    return "computed"


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def place(source: Path, target: Path) -> None:
    """target becomes source's content: hard links (references are large and never modified),
    a copy where linking fails (another volume)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        target.mkdir(exist_ok=True)
        for child in source.iterdir():
            place(child, target / child.name)
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def claim_out(out: Path, settings: dict, fast_path: Sequence[str],
              fast_runs: Sequence[str]) -> None:
    """Reused runs are skipped, so a run directory (gate.json) serves one artifact and one set of
    run settings: another artifact or workload needs a fresh --out. Another candidate set reuses
    the reference and default runs once the fast runs (the fast_runs globs) are removed."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "gate.json"
    settings = dict(settings, artifact=os.path.abspath(settings["artifact"]))
    candidates = sorted(fast_path)
    if not manifest.is_file():
        if any(out.iterdir()):
            raise MeasurementError(f"{out} holds runs without a gate.json; use a fresh --out")
    else:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        differ = sorted(key for key in settings.keys() | recorded["settings"].keys()
                        if key != "artifact" and recorded["settings"].get(key) != settings.get(key))
        if not same_artifact(recorded["settings"].get("artifact", ""), settings["artifact"]):
            differ.insert(0, "artifact")
        if differ:
            raise MeasurementError(f"{out} holds runs of other settings ({', '.join(differ)}); "
                                   "use a fresh --out")
        if recorded["fast_path"] == candidates:
            return
        stale = [str(p) for pattern in fast_runs for p in out.glob(pattern)]
        if stale:
            raise MeasurementError(
                f"{out}'s fast runs were made with --fast-path {recorded['fast_path'] or 'none'}; "
                f"remove {', '.join(stale)} to rerun them with this candidate set")
    manifest.write_text(json.dumps({"settings": settings, "fast_path": candidates}, indent=2) + "\n",
                        encoding="utf-8")


def run_prefill(args: argparse.Namespace) -> Path:
    out = Path(args.out)
    claim_out(out, {"command": "prefill", "artifact": args.artifact, "corpus": args.corpus,
                    "text": args.text, "quick": args.quick, "context": args.context,
                    "stride": args.stride, "ngram_volume": args.ngram_volume,
                    "fast_chunk": args.fast_chunk},
              args.fast_path, ["*/fast-c*"])
    workload = ["--context", str(args.context), "--stride", str(args.stride), "--log-level",
                "warning"]
    workload += ["--corpus", args.corpus] if args.corpus else ["--text", args.text]
    if args.quick:
        workload.append("--quick")
    if args.ngram_volume:
        workload += ["--ngram-volume", args.ngram_volume]
    common = [args.perplexity, args.artifact] + workload
    # The lossless route, BF16 KV, one pass per window.
    one_pass = ["--kv-dtype", "bf16", "--prefill-chunk", str(args.context), "--lossless"]
    reference = out / "reference.top"
    key = {"kind": "prefill", "artifact": artifact_identity(args.artifact),
           "corpus": corpus_identity(args.corpus, args.text), "quick": args.quick,
           "context": args.context, "stride": args.stride,
           "ngram_volume": os.path.abspath(args.ngram_volume) if args.ngram_volume else None,
           "route": one_pass, "lossless_definition": LOSSLESS_DEFINITION}
    source = ensure_reference(
        out, ["reference.top", "reference"], key,
        lambda: run(common + one_pass + ["--save-top-tokens", str(reference), "--output",
                                         str(out / "reference")], out / "reference.log"),
        args.reference_cache, args.recompute_reference,
        [p for p in out.glob("*/*-c*") if p.is_dir()])
    print(f"  reference: {source}", flush=True)
    for kv in args.kv:
        for chunk in args.chunk:
            for route, extra in (("default", ["--reproducible"]),
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
    claim_out(out, {"command": "decode", "artifact": args.artifact, "corpus": args.corpus,
                    "text": args.text, "mode": args.mode, "kv": args.kv,
                    "segments": args.segments, "prompts": args.prompts,
                    "concurrency": args.concurrency, "kv_tokens": args.kv_tokens,
                    "prefix_reuse": args.prefix_reuse, "warm_tokens": args.warm_tokens,
                    "ngram_volume": args.ngram_volume,
                    "free_running_segments": args.free_running_segments},
              args.fast_path, ["fast-forced.json", "fast.txt", "judge.json"])
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
    # The reference: the lossless route at the same positions, one pass per segment.
    reference = out / "reference.json"
    key = {"kind": "decode", "artifact": artifact_identity(args.artifact),
           "corpus": corpus_identity(None, text), "segments": args.segments,
           "prompts": list(args.prompts),
           "ngram_volume": os.path.abspath(args.ngram_volume) if args.ngram_volume else None,
           "route": "lossless", "lossless_definition": LOSSLESS_DEFINITION}
    source = ensure_reference(
        out, ["reference.json"], key,
        lambda: run([args.forced, args.artifact, text, str(reference), "--lossless"] + segments
                    + volume, out / "reference.log"),
        args.reference_cache, args.recompute_reference,
        [p for p in (out / "default-forced.json", out / "fast-forced.json") if p.exists()])
    print(f"  reference: {source}", flush=True)
    for route, extra in (("default", ["--reproducible"]),
                         ("fast", candidate_options(args, "--fast-path"))):
        target = out / f"{route}-forced.json"
        if target.is_file():
            continue
        run([args.forced, args.artifact, text, str(target), "--observe", str(reference),
             "--mode", args.mode, "--kv-dtype", args.kv] + segments + arm + volume + extra,
            out / f"{route}-forced.log")
    # The preemption path acts only on a preempted request: a run that never preempted is no
    # evidence for it.
    if "preemption" in args.fast_path and load_forced(out / "fast-forced.json")["preemptions"] == 0:
        raise MeasurementError("the fast run never preempted; lower --kv-tokens or raise "
                               "--concurrency so the preemption candidate acts")
    verdicts = [evaluate_forced(out / "default-forced.json", out / "fast-forced.json", reference,
                                args.policy)]
    if args.free_running:
        modes = [] if args.mode == "dflash2" else [args.mode]
        for route, extra in (("default", ["reproducible", f"kv={args.kv}"]),
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
        verdicts.append(evaluate_free_running(report, args.policy))
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


def add_reference_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE,
                         help="directory of references shared between run directories, keyed by "
                              "artifact, corpus content and reference settings (default "
                              f"{DEFAULT_REFERENCE_CACHE}; 'none' disables it)")
    command.add_argument("--recompute-reference", action="store_true",
                         help="recompute the reference and replace its cache entry (after a "
                              "change to the lossless route that the key's definition does not "
                              "record)")


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
    add_reference_options(prefill)

    decode = commands.add_parser("decode", help="run and decide the decode criteria")
    decode.add_argument("artifact")
    decode_source = decode.add_mutually_exclusive_group()
    decode_source.add_argument("--corpus", default="eval/corpora/decode-4m/manifest.json",
                               help="corpus manifest whose streams, joined, the segments are cut "
                                    "from (default: the decode corpus infernix-decode-v1)")
    decode_source.add_argument("--text", help="one UTF-8 file the segments are cut from")
    decode.add_argument("--mode", choices=("plain", "dflash2", "mtp", "ngram", "mtp-ngram"), default="plain")
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
    add_reference_options(decode)

    evaluate = commands.add_parser("evaluate-prefill", help="decide an existing prefill run")
    evaluate.add_argument("directory",
                          help="a prefill run directory, or one KV format's subdirectory")
    evaluate.add_argument("--fast-chunk", type=int, default=0,
                          help="the width the fast runs prefilled at (prefill --fast-chunk)")

    compare = commands.add_parser("compare-prefill",
                                  help="decide two sets of runs of one reference, paired in order")
    compare.add_argument("--base", nargs="+", type=Path, required=True)
    compare.add_argument("--candidate", nargs="+", type=Path, required=True)

    forced = commands.add_parser("evaluate-decode", help="decide two teacher-forced decode runs")
    forced.add_argument("default", type=Path)
    forced.add_argument("fast", type=Path)
    forced.add_argument("--reference", type=Path, required=True,
                        help="the reference both runs observed (--lossless of the forced tool)")

    compare_forced = commands.add_parser(
        "compare-decode", help="decide any two teacher-forced runs of one reference (controls)")
    compare_forced.add_argument("--base", type=Path, required=True)
    compare_forced.add_argument("--candidate", type=Path, required=True)
    compare_forced.add_argument("--reference", type=Path, required=True)

    judged = commands.add_parser("evaluate-free-running", help="decide an existing judge report")
    judged.add_argument("report")

    for sub in (prefill, decode, evaluate, compare, forced, compare_forced, judged):
        sub.add_argument("--json", type=Path, help="also write the verdicts as JSON")
        sub.add_argument("--policy", action="append", default=[], metavar="NAME=VALUE",
                         help="decide under another value of a Policy constant (e.g. "
                              "kl_budget=1e-2), repeatable; stored runs are reused as they are")

    same = commands.add_parser("identical", help="check two default-route runs for bit-exactness")
    same.add_argument("first")
    same.add_argument("second")

    concat = commands.add_parser(
        "concat-forced", help="join teacher-forced runs or references of consecutive segment ranges")
    concat.add_argument("parts", nargs="+", type=Path)
    concat.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if getattr(args, "reference_cache", None) is not None and str(args.reference_cache) == "none":
        args.reference_cache = None
    return args


def prefill_verdicts(directory: Path, policy: Policy, fast_chunk: int = 0) -> list[Verdict]:
    if any(d.is_dir() for d in directory.glob("default-c*")):
        return [evaluate_prefill(directory, policy, fast_chunk)]
    kv_dirs = sorted(d for d in directory.iterdir()
                     if d.is_dir() and any(c.is_dir() for c in d.glob("default-c*")))
    if not kv_dirs:
        raise MeasurementError(f"no prefill runs under {directory}")
    return [evaluate_prefill(d, policy, fast_chunk) for d in kv_dirs]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        if hasattr(args, "policy"):
            args.policy = policy_with(args.policy)
        if args.command == "prefill":
            return finish(prefill_verdicts(run_prefill(args), args.policy, args.fast_chunk),
                          args.json)
        if args.command == "decode":
            return finish(run_decode(args), args.json)
        if args.command == "evaluate-prefill":
            return finish(prefill_verdicts(Path(args.directory), args.policy, args.fast_chunk),
                          args.json)
        if args.command == "compare-prefill":
            pairs = compared_pairs(args.base, args.candidate)
            return finish([evaluate_prefill_pairs(pairs, "compare-prefill", args.policy)],
                          args.json)
        if args.command == "evaluate-decode":
            return finish([evaluate_forced(args.default, args.fast, args.reference, args.policy)],
                          args.json)
        if args.command == "compare-decode":
            return finish([evaluate_forced(args.base, args.candidate, args.reference, args.policy,
                                           fast_route=False)], args.json)
        if args.command == "evaluate-free-running":
            return finish([evaluate_free_running(Path(args.report), args.policy)], args.json)
        if args.command == "concat-forced":
            joined = concat_forced(args.parts, args.out)
            print(f"{args.out}: {len(joined['segments'])} segments from {len(args.parts)} parts")
            return 0
        same, detail = identical(Path(args.first), Path(args.second))
        print(f"{'IDENTICAL' if same else 'DIFFERENT'}: {detail}")
        return 0 if same else 1
    except MeasurementError as error:
        print(f"fast_path_gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
