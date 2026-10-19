from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from tools.quality import fast_path_gate as gate


def window(positions: int, mean_kl: float, agreement: float, nll: float) -> dict:
    return {"positions": positions, "mean_kl": mean_kl, "top1_agreement": agreement,
            "nll": nll}


def write_report(directory: Path, *, chunk: int, fast: bool, windows: list[dict],
                 p999: float, reference: str = "ref.top", coverage: float = 0.95,
                 artifact: str = "model.infernix") -> None:
    directory.mkdir(parents=True)
    positions = sum(w["positions"] for w in windows)
    mean_kl = sum(w["positions"] * w["mean_kl"] for w in windows) / positions
    report = {
        "artifact": {"path": artifact, "prefill_signature": "sig"},
        "execution": {"context_tokens": 14336, "stride_tokens": 7168, "kv_dtype": "int8-g64",
                      "prefill_chunk_tokens": chunk,
                      "fast_paths": ["decode-split64"] if fast else []},
        "streams": [{"id": "s", "windows": [
            {"index": i, "scored_tokens": w["positions"], "total_nll": w["nll"],
             "kl_divergence": {"positions": w["positions"], "mean_kl": w["mean_kl"],
                               "top1_agreement": w["top1_agreement"]}}
            for i, w in enumerate(windows)]}],
        "kl_divergence": {
            "reference": reference,
            "p999_kl": p999,
            "mean_reference_top_mass": coverage,
            "by_context": [{"context_begin": 0, "context_end": 1024, "positions": positions,
                            "mean_kl": mean_kl}],
        },
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")


def make_prefill(root: Path, *, windows: int, floor: float, excess: float, noise: float,
                 top1_floor: float = 0.04, top1_excess: float = 0.0,
                 nll_excess: float | None = None, chunks: tuple[int, ...] = (896, 3584),
                 tail_ratio: float = 1.0, alternating: float = 0.0, coverage: float = 0.95,
                 fast_flag: bool = True, fast_chunk: int = 0, seed: int = 1,
                 default_p999: float = 1.0) -> Path:
    """Synthetic default and fast runs: the fast route's KL and NLL move by excess per position
    (NLL by nll_excess when given), plus per-window noise."""
    rng = random.Random(seed)
    nll_excess = excess if nll_excess is None else nll_excess
    for chunk in chunks:
        default, fast = [], []
        for index in range(windows):
            base = floor * (1.0 + rng.uniform(-0.3, 0.3))
            nll = 7168 * rng.uniform(1.0, 2.0)
            # Alternating signs: an exactly zero-mean difference of fixed spread.
            swing = alternating if index % 2 == 0 else -alternating
            shift = swing + rng.gauss(0.0, noise)
            default.append(window(7168, base, 1.0 - top1_floor, nll))
            fast.append(window(7168, base + excess + shift, 1.0 - top1_floor - top1_excess,
                               nll + 7168 * (nll_excess + shift)))
        write_report(root / f"default-c{chunk}", chunk=chunk, fast=False, windows=default,
                     p999=default_p999, coverage=coverage)
        write_report(root / f"fast-c{chunk}", chunk=fast_chunk or chunk, fast=fast_flag, windows=fast,
                     p999=tail_ratio, coverage=coverage)
    return root


def test_a_widened_fast_run_pairs_only_under_its_fast_chunk(tmp_path):
    # prefill --fast-chunk: the fast runs prefill at one width, each paired with a default chunk.
    root = make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0, noise=1e-4, fast_chunk=3584,
                        chunks=(896, 1792))
    assert gate.evaluate_prefill(root, fast_chunk=3584).passed
    with pytest.raises(gate.MeasurementError, match="prefills at 3584"):
        gate.evaluate_prefill(root)
    with pytest.raises(gate.MeasurementError, match="expected 7168"):
        gate.evaluate_prefill(root, fast_chunk=7168)


def criterion(verdict: gate.Verdict, name: str) -> gate.Criterion:
    return next(c for c in verdict.criteria if c.name == name)


def test_paired_excess_matches_the_cluster_robust_ratio_estimator():
    units = [gate.PairedUnit(2.0, 1.0, 1.5), gate.PairedUnit(1.0, 2.0, 1.8),
             gate.PairedUnit(3.0, 0.0, 0.9)]
    mean, bound = gate.paired_excess(units, 1.0)
    differences = [0.5, -0.2, 0.9]
    expected_mean = sum(differences) / 6.0
    residual = sum((d - expected_mean * u.weight) ** 2 for d, u in zip(differences, units))
    assert mean == pytest.approx(expected_mean)
    assert bound == pytest.approx(expected_mean + math.sqrt(residual * 3 / 2) / 6.0)


def test_a_fast_path_identical_to_the_default_passes(tmp_path):
    root = make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0, noise=0.0)
    verdict = gate.evaluate_prefill(root)
    assert verdict.passed
    assert verdict.facts["fast_identical_to_default"]
    assert criterion(verdict, "mean KL excess").bound == 0.0
    assert criterion(verdict, "corpus NLL excess").bound == 0.0


def test_one_kl_budget_for_every_configuration(tmp_path):
    # The same excess gets the same verdict whatever the configuration's floor: a 4-bit-like, an
    # 8-bit-like and an exact route that is the reference itself (floor 0).
    for floor in (0.03, 0.003, 0.0):
        passing = gate.evaluate_prefill(
            make_prefill(tmp_path / f"pass-{floor}", windows=60, floor=floor, excess=0.0012,
                         noise=1e-5))
        failing = gate.evaluate_prefill(
            make_prefill(tmp_path / f"fail-{floor}", windows=60, floor=floor, excess=0.0018,
                         noise=1e-5))
        kl = criterion(passing, "mean KL excess")
        assert kl.budget == gate.POLICY.kl_budget
        assert kl.passed
        assert not criterion(failing, "mean KL excess").passed


def test_a_zero_floor_configuration_can_pass(tmp_path):
    # Floor 0 (the exact route is the reference): the default's tail is 0, so the tail cap admits
    # a fast p99.9 up to tail_minimum.
    small = gate.evaluate_prefill(
        make_prefill(tmp_path / "a", windows=40, floor=0.0, excess=5e-4, noise=0.0, top1_floor=0.0,
                     top1_excess=0.004, default_p999=0.0, tail_ratio=0.03))
    assert small.passed
    assert criterion(small, "p99.9 KL ratio").estimate == math.inf
    large = gate.evaluate_prefill(
        make_prefill(tmp_path / "b", windows=40, floor=0.0, excess=2e-3, noise=0.0, top1_floor=0.0,
                     default_p999=0.0, tail_ratio=0.2))
    assert not criterion(large, "mean KL excess").passed
    assert not criterion(large, "p99.9 KL ratio").passed


def test_noise_widens_the_bound_instead_of_the_budget(tmp_path):
    # A zero-mean but noisy difference must not pass on its point estimate.
    verdict = gate.evaluate_prefill(
        make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0, noise=0.0, alternating=0.01))
    kl = criterion(verdict, "mean KL excess")
    assert kl.estimate == pytest.approx(0.0, abs=1e-12)
    assert kl.bound > kl.budget
    assert not kl.passed


def test_top1_and_tail_criteria(tmp_path):
    verdict = gate.evaluate_prefill(
        make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0, noise=0.0,
                     top1_floor=0.04, top1_excess=0.006, tail_ratio=1.5))
    assert not criterion(verdict, "top-1 disagreement excess").passed
    assert not criterion(verdict, "p99.9 KL ratio").passed
    assert criterion(verdict, "mean KL excess").passed


def test_a_corpus_nll_loss_fails_even_with_kl_inside_budget(tmp_path):
    # A change the truncated KL cannot see (mass moved within the unsaved tail) still raises the
    # corpus tokens' NLL; the cap is twice the KL budget, on the point estimate.
    verdict = gate.evaluate_prefill(
        make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0, noise=0.0, nll_excess=0.006))
    assert criterion(verdict, "mean KL excess").passed
    nll = criterion(verdict, "corpus NLL excess")
    assert nll.estimate == pytest.approx(0.006)
    assert nll.budget == pytest.approx(gate.POLICY.point_cap_factor * gate.POLICY.kl_budget)
    assert not nll.passed
    noisy = gate.evaluate_prefill(
        make_prefill(tmp_path / "noisy", windows=40, floor=0.02, excess=0.0, noise=0.0,
                     nll_excess=0.002))
    assert criterion(noisy, "corpus NLL excess").passed


def test_a_reference_that_resolves_too_little_cannot_decide(tmp_path):
    with pytest.raises(gate.MeasurementError, match="cover"):
        gate.evaluate_prefill(make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0,
                                           noise=0.0, coverage=0.8))


def test_too_few_windows_cannot_decide(tmp_path):
    with pytest.raises(gate.MeasurementError):
        gate.evaluate_prefill(make_prefill(tmp_path, windows=10, floor=0.02, excess=0.0,
                                           noise=0.0, chunks=(896,)))


def test_mismatched_runs_cannot_decide(tmp_path):
    root = make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0, noise=0.0, chunks=(896,))
    report_path = root / "fast-c896" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["kl_divergence"]["reference"] = "other.top"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(gate.MeasurementError):
        gate.evaluate_prefill(root)


def test_a_fast_route_pairs_runs_of_one_artifact(tmp_path):
    root = make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0, noise=0.0, chunks=(896,))
    report_path = root / "fast-c896" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact"]["prefill_signature"] = "other"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(gate.MeasurementError, match="other artifacts"):
        gate.evaluate_prefill(root)
    # A comparison may set two artifacts against one reference.
    pairs = gate.compared_pairs([root / "default-c896"], [root / "fast-c896"])
    assert gate.evaluate_prefill_pairs(pairs, "control").passed


def test_a_run_directory_serves_one_artifact_and_settings(tmp_path):
    settings = {"command": "decode", "artifact": "a.infernix", "kv_tokens": 4096}
    fast_runs = ["fast-forced.json"]
    out = tmp_path / "out"
    gate.claim_out(out, settings, ["preemption"], fast_runs)
    (out / "default-forced.json").write_text("{}", encoding="utf-8")
    (out / "fast-forced.json").write_text("{}", encoding="utf-8")
    gate.claim_out(out, settings, ["preemption"], fast_runs)  # reuse under the same settings
    with pytest.raises(gate.MeasurementError, match="artifact"):
        gate.claim_out(out, dict(settings, artifact="b.infernix"), ["preemption"], fast_runs)
    with pytest.raises(gate.MeasurementError, match="kv_tokens"):
        gate.claim_out(out, dict(settings, kv_tokens=0), ["preemption"], fast_runs)
    # Another candidate set keeps the reference and default runs but not the fast run.
    with pytest.raises(gate.MeasurementError, match="fast-forced.json"):
        gate.claim_out(out, settings, [], fast_runs)
    (out / "fast-forced.json").unlink()
    gate.claim_out(out, settings, [], fast_runs)
    assert json.loads((out / "gate.json").read_text(encoding="utf-8"))["fast_path"] == []
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "reference.json").write_text("{}", encoding="utf-8")
    with pytest.raises(gate.MeasurementError, match="gate.json"):
        gate.claim_out(legacy, settings, [], fast_runs)


def test_a_fast_route_needs_the_fast_path_flag_but_a_comparison_does_not(tmp_path):
    root = make_prefill(tmp_path, windows=40, floor=0.02, excess=0.0, noise=0.0, chunks=(896,),
                        fast_flag=False)
    with pytest.raises(gate.MeasurementError, match="fast paths active"):
        gate.evaluate_prefill(root)
    pairs = gate.compared_pairs([root / "default-c896"], [root / "fast-c896"])
    assert gate.evaluate_prefill_pairs(pairs, "control").passed


def write_forced(path: Path, segments: list[tuple[int, list[float]]], *, fast: bool,
                 observed: list[tuple[list[float], list[float]]] | None = None) -> Path:
    run = {
        "artifact": "model.infernix", "mode": "plain", "kv_dtype": "int8", "forced_tokens": 64,
        "fast_paths": ["decode-split64"] if fast else [],
        "underflows": 0,
        "observed_reference": "reference.json" if observed is not None else "",
        "segments": [{"offset": 1000 * i, "prompt_tokens": prompt, "logprobs": logprobs}
                     for i, (prompt, logprobs) in enumerate(segments)]}
    if observed is not None:
        for segment, (values, best) in zip(run["segments"], observed):
            segment["observed"] = values
            segment["top_logprobs"] = best
    path.write_text(json.dumps(run), encoding="utf-8")
    return path


def forced_segments(count: int, shift: float, rng: random.Random, noise: float = 0.0):
    out = []
    for index in range(count):
        base = [-rng.uniform(0.0, 5.0) for _ in range(64)]
        out.append(([512, 2048, 8192][index % 3], base))
    shifted = [(prompt, [lp - shift + rng.gauss(0.0, noise) for lp in logprobs])
               for prompt, logprobs in out]
    return out, shifted


# The reference's top tokens at every position: four tokens holding 92 % of its distribution.
REFERENCE_TOP = [math.log(p) for p in (0.55, 0.22, 0.10, 0.05)]


def write_reference(path: Path, count: int, coverage_scale: float = 1.0) -> Path:
    top = [lp + math.log(coverage_scale) for lp in REFERENCE_TOP]
    path.write_text(json.dumps({
        "artifact": "model.infernix", "kind": "reference", "forced_tokens": 64,
        "top_tokens": len(top),
        "segments": [{"offset": 1000 * i, "prompt_tokens": [512, 2048, 8192][i % 3],
                      "logprobs": [-1.0] * 64, "top_ids": list(range(4)) * 64,
                      "top_logprobs": top * 64} for i in range(count)]}), encoding="utf-8")
    return path


def observations(count: int, noise: float, seed: int, *, swap: float = 0.0):
    """Each segment's observed log-probabilities of the reference's top tokens: the reference's
    distribution (its top tokens and the rest) with Gaussian noise on every logit, renormalized;
    with swap > 0, that share of positions favor the second token."""
    rng = random.Random(seed)
    rest = math.log(1.0 - sum(math.exp(lp) for lp in REFERENCE_TOP))
    out = []
    for _ in range(count):
        values, best = [], []
        for _ in range(64):
            logits = [lp + rng.gauss(0.0, noise) for lp in REFERENCE_TOP + [rest]]
            normalizer = math.log(sum(math.exp(v) for v in logits))
            step = [v - normalizer for v in logits[:-1]]
            if rng.random() < swap:
                step[0], step[1] = step[1], step[0]
            values += step
            best.append(max(step))
        out.append((values, best))
    return out


def make_decode(tmp_path: Path, *, count: int = 60, default_noise: float = 0.05,
                fast_noise: float | None = None, fast_seed: int = 1, swap: float = 0.0,
                fast_flag: bool = True) -> tuple[Path, Path, Path]:
    reference = write_reference(tmp_path / "reference.json", count)
    logprobs, _ = forced_segments(count, 0.0, random.Random(4))
    default = write_forced(tmp_path / "default.json", logprobs, fast=False,
                           observed=observations(count, default_noise, 1))
    fast_observed = observations(count, default_noise if fast_noise is None else fast_noise,
                                 fast_seed, swap=swap)
    fast = write_forced(tmp_path / "fast.json", logprobs, fast=fast_flag, observed=fast_observed)
    return reference, default, fast


def test_the_divergence_matches_infernix_perplexity():
    reference = [math.log(0.6), math.log(0.3)]
    kl, tv, mass = gate.position_divergence(reference, reference)
    assert kl == pytest.approx(0.0, abs=1e-15)
    assert tv == pytest.approx(0.0, abs=1e-15)
    assert mass == pytest.approx(0.9)
    evaluated = [math.log(0.5), math.log(0.3)]
    kl, tv, _ = gate.position_divergence(reference, evaluated)
    assert kl == pytest.approx(0.6 * math.log(0.6 / 0.5) + 0.1 * math.log(0.1 / 0.2))
    # Total variation over the top tokens and the rest: (|0.6-0.5| + |0.3-0.3| + |0.1-0.2|) / 2.
    assert tv == pytest.approx(0.1)


def test_teacher_forced_decode_measures_the_distribution_against_the_reference(tmp_path):
    reference, default, fast = make_decode(tmp_path)
    verdict = gate.evaluate_forced(default, fast, reference)
    assert verdict.passed
    assert verdict.facts["fast_identical_to_default"]
    assert verdict.facts["reference_top_mass"] == pytest.approx(0.92)
    floor_tv, floor_kl = verdict.facts["floor_tv"], verdict.facts["floor_kl"]
    assert floor_tv > 0 and floor_kl > 0
    tv = criterion(verdict, "decode mean TV excess")
    assert tv.estimate == 0.0
    assert tv.budget == gate.POLICY.tv_budget
    # KL is a gross cap on its point estimate, like NLL.
    assert criterion(verdict, "decode mean KL excess").budget == gate.POLICY.decode_kl_cap
    assert {c.name for c in verdict.criteria} >= {
        "decode top-1 disagreement excess", "decode NLL excess", "decode p99.9 KL ratio",
        "prompt 512 decode TV excess", "prompt 8192 decode TV excess"}


def test_a_wider_decode_perturbation_fails(tmp_path):
    reference, default, fast = make_decode(tmp_path, fast_noise=0.15)
    verdict = gate.evaluate_forced(default, fast, reference)
    assert not criterion(verdict, "decode mean TV excess").passed
    assert not verdict.passed


def test_an_equal_perturbation_from_other_rounding_passes(tmp_path):
    # A/A: the same noise level, independently drawn, like two equally precise orderings.
    reference, default, fast = make_decode(tmp_path, fast_seed=2)
    verdict = gate.evaluate_forced(default, fast, reference)
    assert criterion(verdict, "decode mean TV excess").passed


def test_a_few_flipped_positions_cannot_dominate_the_distribution_criterion(tmp_path):
    # Knife-edge positions: in the fast run, three positions move the reference's top token to
    # almost nothing (KL of about 18 nats each). Total variation counts each at most once.
    count = 60
    reference = write_reference(tmp_path / "reference.json", count)
    logprobs, _ = forced_segments(count, 0.0, random.Random(4))
    observed = observations(count, 0.05, 1)
    default = write_forced(tmp_path / "default.json", logprobs, fast=False, observed=observed)
    flipped = [(list(values), list(best)) for values, best in observed]
    for segment in (3, 17, 41):
        values, best = flipped[segment]
        values[0] = math.log(1e-8)
        best[0] = max(values[1:4])
    fast = write_forced(tmp_path / "fast.json", logprobs, fast=True, observed=flipped)
    verdict = gate.evaluate_forced(default, fast, reference)
    # Three positions of 3,840 move the mean KL by several times the KL budget ...
    assert verdict.facts["fast_mean_kl"] - verdict.facts["floor_kl"] > 4 * gate.POLICY.kl_budget
    # ... but total variation by at most 3/3840.
    assert criterion(verdict, "decode mean TV excess").passed


def test_decode_top1_follows_the_run_maximum(tmp_path):
    reference, default, fast = make_decode(tmp_path, fast_noise=0.05, swap=0.2)
    verdict = gate.evaluate_forced(default, fast, reference)
    assert verdict.facts["fast_top1_disagreement"] == pytest.approx(0.2, abs=0.03)
    assert not criterion(verdict, "decode top-1 disagreement excess").passed


def test_teacher_forced_runs_must_match(tmp_path):
    reference, default, fast = make_decode(tmp_path, count=40)
    (tmp_path / "u").mkdir()
    unflagged = make_decode(tmp_path / "u", count=40, fast_flag=False)[2]
    with pytest.raises(gate.MeasurementError, match="fast paths active"):
        gate.evaluate_forced(default, unflagged, reference)
    # A control compares any two runs of one reference.
    assert gate.evaluate_forced(default, unflagged, reference, fast_route=False).passed
    short = write_reference(tmp_path / "short.json", 39)
    with pytest.raises(gate.MeasurementError):
        gate.evaluate_forced(default, fast, short)
    logprobs, _ = forced_segments(40, 0.0, random.Random(4))
    blind = write_forced(tmp_path / "blind.json", logprobs, fast=True)
    with pytest.raises(gate.MeasurementError, match="observe"):
        gate.evaluate_forced(default, blind, reference)
    other = json.loads(reference.read_text(encoding="utf-8"))
    other["artifact"] = "other.infernix"
    (tmp_path / "other.json").write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(gate.MeasurementError, match="other artifacts"):
        gate.evaluate_forced(default, fast, tmp_path / "other.json")


def test_decode_needs_reference_coverage(tmp_path):
    reference, default, fast = make_decode(tmp_path)
    write_reference(reference, 60, coverage_scale=0.9)
    with pytest.raises(gate.MeasurementError, match="cover"):
        gate.evaluate_forced(default, fast, reference)


def write_judge(path: Path, default: list[tuple[int, int, float]],
                fast: list[tuple[int, int, float]]) -> Path:
    def segments(values):
        return [{"offset": 100 * i, "tokens": t, "agree": a, "logprob": -r, "regret": r}
                for i, (t, a, r) in enumerate(values)]

    path.write_text(json.dumps({"reference": "a16_activations", "runs": [
        {"file": "default.txt", "segments": segments(default)},
        {"file": "fast.txt", "segments": segments(fast)}]}), encoding="utf-8")
    return path


def test_free_running_criteria(tmp_path):
    rng = random.Random(3)
    default = [(384, 375, 3.0) for _ in range(64)]
    same = gate.evaluate_free_running(write_judge(tmp_path / "a.json", default, default))
    assert same.passed
    worse = [(384, 375 - rng.choice((1, 2, 3)), 3.0 + rng.uniform(0.5, 1.0)) for _ in range(64)]
    verdict = gate.evaluate_free_running(write_judge(tmp_path / "b.json", default, worse))
    assert not criterion(verdict, "reference-top disagreement excess").passed
    assert not criterion(verdict, "mean regret excess").passed


def test_free_running_needs_paired_segments(tmp_path):
    report = write_judge(tmp_path / "a.json", [(384, 380, 1.0)] * 40, [(384, 380, 1.0)] * 39)
    with pytest.raises(gate.MeasurementError):
        gate.evaluate_free_running(report)


def write_identity_report(directory: Path, nll: list[float], *, fast: bool = False) -> Path:
    directory.mkdir(parents=True)
    report = {
        "execution": {"context_tokens": 14336, "stride_tokens": 7168, "kv_dtype": "int8-g64",
                      "prefill_chunk_tokens": 3584,
                      "fast_paths": ["decode-split64"] if fast else []},
        "streams": [{"id": "s", "windows": [
            {"index": i, "scored_tokens": 7168, "total_nll": value}
            for i, value in enumerate(nll)]}],
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


def test_identity_compares_every_window_exactly(tmp_path):
    a = write_identity_report(tmp_path / "a", [1.25, 2.5])
    b = write_identity_report(tmp_path / "b", [1.25, 2.5])
    c = write_identity_report(tmp_path / "c", [1.25, 2.5000000000000004])
    assert gate.identical(a, b)[0]
    assert not gate.identical(a, c)[0]
    with pytest.raises(gate.MeasurementError):
        gate.identical(a, write_identity_report(tmp_path / "d", [1.25, 2.5], fast=True))


def test_identity_compares_forced_runs_and_generations(tmp_path):
    rng = random.Random(7)
    default, _ = forced_segments(3, 0.0, rng)
    a = write_forced(tmp_path / "a.json", default, fast=False)
    changed = [(p, list(lp)) for p, lp in default]
    changed[1][1][5] -= 1e-6
    b = write_forced(tmp_path / "b.json", changed, fast=False)
    assert gate.identical(a, a)[0]
    assert not gate.identical(a, b)[0]
    x = tmp_path / "a.txt"
    y = tmp_path / "b.txt"
    x.write_text("0 4 1 2 3\n100 4 5 6\n", encoding="utf-8")
    y.write_text("0 4 1 2 3\n100 4 5 7\n", encoding="utf-8")
    assert gate.identical(x, x)[0]
    assert not gate.identical(x, y)[0]


# --- shared reference cache ----------------------------------------------------------------------


class Producer:
    """Stands for the reference tool: writes a reference file and a report directory."""

    def __init__(self, out: Path, content: str) -> None:
        self.out, self.content, self.calls = out, content, 0

    def __call__(self) -> None:
        self.calls += 1
        (self.out / "reference.json").write_text(self.content, encoding="utf-8")
        (self.out / "reference").mkdir(exist_ok=True)
        (self.out / "reference" / "report.json").write_text(self.content, encoding="utf-8")


NAMES = ["reference.json", "reference"]


def reference_key(artifact: Path, corpus: Path, segments: int = 256) -> dict:
    return {"kind": "decode", "artifact": gate.artifact_identity(str(artifact)),
            "corpus": gate.corpus_identity(None, str(corpus)), "segments": segments}


@pytest.fixture
def inputs(tmp_path):
    artifact = tmp_path / "model.infernix"
    artifact.write_bytes(b"weights")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("some text", encoding="utf-8")
    return artifact, corpus, tmp_path / "cache"


def test_a_reference_is_computed_once_and_reused_by_other_run_directories(tmp_path, inputs):
    artifact, corpus, cache = inputs
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    produce = Producer(first, "reference A")
    assert gate.ensure_reference(first, NAMES, reference_key(artifact, corpus), produce, cache,
                                 False).startswith("computed")
    reuse = Producer(second, "never written")
    assert gate.ensure_reference(second, NAMES, reference_key(artifact, corpus), reuse, cache,
                                 False).startswith("reused")
    assert (produce.calls, reuse.calls) == (1, 0)
    assert (second / "reference.json").read_text(encoding="utf-8") == "reference A"
    assert (second / "reference" / "report.json").read_text(encoding="utf-8") == "reference A"
    # A directory that has its reference keeps it.
    assert gate.ensure_reference(second, NAMES, reference_key(artifact, corpus), reuse, cache,
                                 False) == "kept"


def test_another_corpus_artifact_or_setting_computes_its_own_reference(tmp_path, inputs):
    artifact, corpus, cache = inputs
    base = tmp_path / "base"
    base.mkdir()
    gate.ensure_reference(base, NAMES, reference_key(artifact, corpus), Producer(base, "A"), cache,
                          False)
    other_corpus = tmp_path / "other.txt"
    other_corpus.write_text("other text", encoding="utf-8")
    rewritten = tmp_path / "rewritten.infernix"
    rewritten.write_bytes(b"weights, rewritten")
    for index, key in enumerate((reference_key(artifact, other_corpus),
                                 reference_key(rewritten, corpus),
                                 reference_key(artifact, corpus, segments=1024))):
        out = tmp_path / f"run{index}"
        out.mkdir()
        produce = Producer(out, f"reference {index}")
        gate.ensure_reference(out, NAMES, key, produce, cache, False)
        assert produce.calls == 1
    # The corpus key follows content, not the file name.
    renamed = tmp_path / "renamed.txt"
    renamed.write_text("some text", encoding="utf-8")
    assert reference_key(artifact, renamed) == reference_key(artifact, corpus)


def test_recompute_replaces_the_entry_and_refuses_with_dependent_runs(tmp_path, inputs):
    artifact, corpus, cache = inputs
    key = reference_key(artifact, corpus)
    first = tmp_path / "a"
    first.mkdir()
    gate.ensure_reference(first, NAMES, key, Producer(first, "old"), cache, False)
    (first / "default-forced.json").write_text("{}", encoding="utf-8")
    with pytest.raises(gate.MeasurementError):
        gate.ensure_reference(first, NAMES, key, Producer(first, "new"), cache, True,
                              [first / "default-forced.json"])
    fresh = tmp_path / "b"
    fresh.mkdir()
    produce = Producer(fresh, "new")
    gate.ensure_reference(fresh, NAMES, key, produce, cache, True)
    assert produce.calls == 1
    later = tmp_path / "c"
    later.mkdir()
    gate.ensure_reference(later, NAMES, key, Producer(later, "unused"), cache, False)
    assert (later / "reference.json").read_text(encoding="utf-8") == "new"
    # The earlier directory's linked files kept their own content.
    assert (first / "reference.json").read_text(encoding="utf-8") == "old"


def test_without_a_cache_every_directory_computes_its_reference(tmp_path, inputs):
    artifact, corpus, _ = inputs
    for name in ("a", "b"):
        out = tmp_path / name
        out.mkdir()
        produce = Producer(out, name)
        assert gate.ensure_reference(out, NAMES, reference_key(artifact, corpus), produce, None,
                                     False) == "computed"
        assert produce.calls == 1
    args = gate.parse_arguments(["decode", "model", "--out", "x", "--reference-cache", "none"])
    assert args.reference_cache is None
