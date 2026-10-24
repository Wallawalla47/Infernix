"""IQS-2 decisions (tools/quality/iqs.py) on synthetic arms: what passes, what fails, how the noise floor
moves a budget, and how a run plan becomes a certificate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.quality import iqs

DOMAINS = ("prose", "code", "math", "chat")


def write_arm(directory: Path, name: str, kl: np.ndarray, flip: np.ndarray, dnll: np.ndarray, windows: int,
              route: str = "default", nonfinite: int = 0, first: int = 0) -> Path:
    """An infernix-certify arm (prefix.json, prefix.positions.csv) of windows first .. first+windows-1, and its
    reference's window list."""
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / f"ref{first}.bin"
    meta_windows = [{"index": w, "id": f"doc{w}", "domain": DOMAINS[w % len(DOMAINS)]}
                    for w in range(first, first + windows)]
    Path(str(reference) + ".json").write_text(json.dumps({"windows": meta_windows}), encoding="utf-8")
    prefix = directory / name
    per = kl.size // windows
    lines = ["window,segment,position,kl,top_mass,flip,dnll"]
    for i in range(kl.size):
        lines.append(f"{first + i // per},-1,{i % per + 1},{float(kl[i]):.9g},0.95,{int(flip[i])},{float(dnll[i]):.9g}")
    Path(str(prefix) + ".positions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {"route": route, "kv_dtype": "bf16", "reference": str(reference), "nonfinite": nonfinite,
            "load": {"fast_paths": ["prompt-pv8"]}}
    Path(str(prefix) + ".json").write_text(json.dumps(meta), encoding="utf-8")
    return prefix


def synthetic(rng: np.random.Generator, windows: int, per: int, kl_mean: float, flip_rate: float,
              dnll_mean: float = 0.0):
    n = windows * per
    kl = rng.exponential(kl_mean, n)
    flip = (rng.random(n) < flip_rate).astype(float)
    dnll = rng.normal(dnll_mean, 0.01, n)
    return kl, flip, dnll


FAST = iqs.replace(iqs.POLICY, resamples=400)


def test_a_faithful_path_passes(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    arm = write_arm(tmp_path, "good", *synthetic(rng, 32, 200, 2e-4, 0.002), windows=32)
    decision = iqs.decide(iqs.load_arm(arm), FAST)
    assert decision["passed"], decision["criteria"]
    assert decision["bound"]["mean_kl"] >= decision["estimate"]["mean_kl"]
    assert set(decision["domains"]) == set(DOMAINS)


def test_mean_kl_above_q8_fails(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    arm = write_arm(tmp_path, "bad", *synthetic(rng, 32, 200, 3e-3, 0.002), windows=32)
    decision = iqs.decide(iqs.load_arm(arm), FAST)
    failing = {c["name"] for c in decision["criteria"] if not c["passed"]}
    assert "C1 mean KL" in failing and not decision["passed"]


def test_per_domain_values_and_perplexity_are_disclosed_not_gated(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    kl, flip, dnll = synthetic(rng, 32, 200, 1e-4, 0.001, dnll_mean=0.004)
    # Every code window (w % 4 == 1) far worse; the overall mean stays below C1.
    for w in range(1, 32, 4):
        kl[w * 200:(w + 1) * 200] *= 10
    arm = write_arm(tmp_path, "domain", kl, flip, dnll, windows=32)
    decision = iqs.decide(iqs.load_arm(arm), FAST)
    assert decision["passed"], decision["criteria"]
    assert {c["name"] for c in decision["criteria"]} == {"C0 finite", "C1 mean KL", "C2 flip rate", "C3 p99.9 KL"}
    assert decision["domains"]["code"]["mean_kl"] > 5 * decision["domains"]["prose"]["mean_kl"]
    assert decision["perplexity"]["ratio"] == pytest.approx(np.exp(decision["estimate"]["mean_dnll"]))
    assert decision["perplexity"]["ratio_bound"] >= decision["perplexity"]["ratio"] > 1.0


def test_noise_floor_raises_a_budget_by_kappa(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    path = write_arm(tmp_path, "moe", *synthetic(rng, 32, 200, 3e-3, 0.004), windows=32)
    twin = write_arm(tmp_path, "twin", *synthetic(rng, 8, 200, 2.5e-3, 0.004), windows=8, route="noise_twin")
    assert not iqs.decide(iqs.load_arm(path), FAST)["passed"]
    floor = iqs.floor_of(twin)
    decision = iqs.decide(iqs.load_arm(path), FAST, floor)
    assert decision["budget"]["mean_kl"] == pytest.approx(2.0 * floor["mean_kl"])
    assert decision["passed"], decision["criteria"]


def test_a_floor_below_the_fixed_budget_changes_nothing() -> None:
    floor = {"mean_kl": 1e-5, "flip_rate": 1e-4, "p999_kl": 1e-3, "clipped_kl": 1e-5, "mean_dnll": 0.0}
    with_floor = iqs.budgets(iqs.POLICY, floor)
    assert {m: with_floor[m] for m in ("mean_kl", "flip_rate", "p999_kl")} == iqs.budgets(iqs.POLICY, None)
    assert with_floor["clipped_kl"] == pytest.approx(2e-5)  # C4 has no fixed budget


def test_nonfinite_fails_outright(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    arm = write_arm(tmp_path, "nan", *synthetic(rng, 32, 50, 1e-4, 0.001), windows=32, nonfinite=1)
    decision = iqs.decide(iqs.load_arm(arm), FAST)
    assert not decision["criteria"][0]["passed"] and not decision["passed"]


def test_too_few_windows_is_unusable(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    arm = write_arm(tmp_path, "small", *synthetic(rng, 4, 50, 1e-4, 0.001), windows=4)
    with pytest.raises(iqs.Unusable):
        iqs.decide(iqs.load_arm(arm), FAST)


def test_policy_overrides_decide_again_without_measuring(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    arm = iqs.load_arm(write_arm(tmp_path, "edge", *synthetic(rng, 32, 200, 1.0e-3, 0.002), windows=32))
    strict = iqs.policy_with(["mean_kl=5e-4", "resamples=400"])
    assert not iqs.decide(arm, strict)["passed"]
    loose = iqs.policy_with(["mean_kl=1e-2", "resamples=400"])
    assert iqs.decide(arm, loose)["passed"]


def test_kv_disclosure_uses_its_class(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    arm = iqs.load_arm(write_arm(tmp_path, "vq2", *synthetic(rng, 16, 200, 6e-3, 0.02), windows=16, route="design"))
    assert iqs.disclose(arm, "vq2", FAST)["passed"]
    assert not iqs.disclose(arm, "int8", FAST)["passed"]


def test_weighted_quantile_matches_the_cumulative_definition() -> None:
    """The batched p99.9 equals the first ascending value whose cumulative weight reaches 99.9 %, including
    resamples that weight the top values nearly out (the whole-column fallback)."""
    rng = np.random.default_rng(3)
    windows, per = 16, 400
    values = rng.exponential(1e-3, windows * per)
    values[:per] *= 50.0  # window 0 holds the tail
    window = np.repeat(np.arange(windows), per)
    weights = rng.multinomial(windows, np.full(windows, 1.0 / windows), size=300).astype(np.float64)
    weights[:20, 0] = 0.0
    total = weights @ np.bincount(window, minlength=windows).astype(np.float64)
    got = iqs.weighted_upper_quantile(values, window, weights, total, 0.999, chunk=64)
    order = np.argsort(values, kind="stable")
    for b in range(len(weights)):
        cumulative = np.cumsum(weights[b][window[order]])
        index = min(np.searchsorted(cumulative, 0.999 * cumulative[-1]), len(values) - 1)
        assert got[b] == values[order][index]


def test_certificate_collects_decisions_disclosures_and_checks(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    write_arm(tmp_path, "good", *synthetic(rng, 16, 100, 2e-4, 0.002), windows=16)
    (tmp_path / "invariance.json").write_text(json.dumps({"passed": True, "detail": "4 arms identical"}), encoding="utf-8")
    plan = {"provenance": {"build": "test"},
            "decisions": [{"label": "q27gw bf16 default prefill", "arm": "good"}],
            "checks": [{"name": "invariance", "result": "invariance.json"},
                       {"name": "missing check", "result": "absent.json"}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    status = iqs.run_certificate(tmp_path / "plan.json", tmp_path / "cert", FAST)
    certificate = json.loads((tmp_path / "cert" / "certificate.json").read_text(encoding="utf-8"))
    assert status == 1 and not certificate["passed"]
    assert certificate["decisions"][0]["passed"]
    assert [c["passed"] for c in certificate["checks"]] == [True, False]
    assert "IQS-2 certificate" in (tmp_path / "cert" / "certificate.html").read_text(encoding="utf-8")


def write_scored(path: Path, windows: dict[int, tuple[int, list[list[int]]]]) -> Path:
    """infernix-certify mask's output: per window its position count and the inclusive runs it scores."""
    entries = [{"index": w, "id": f"doc{w}", "domain": "d", "positions": n, "scored": sum(b - a + 1 for a, b in runs),
                "runs": runs} for w, (n, runs) in windows.items()]
    path.write_text(json.dumps({"windows": entries}), encoding="utf-8")
    return path


def test_only_the_positions_the_model_writes_are_scored(tmp_path: Path) -> None:
    # 16 windows of 100 positions; the first half of each is prompt text with a huge KL, the second half is
    # assistant output with a small one
    n, per = 16, 100
    kl = np.where(np.tile(np.arange(per), n) < 50, 5.0, 1e-4)
    arm = write_arm(tmp_path, "chat", kl, np.zeros(kl.size), np.zeros(kl.size), windows=n)
    scored = write_scored(tmp_path / "scored.json", {w: (per, [[51, per]]) for w in range(n)})
    everything = iqs.decide(iqs.load_arm(arm), FAST)
    only = iqs.decide(iqs.load_arm(arm, scored), FAST)
    assert not everything["passed"] and only["passed"]
    assert only["positions"] == n * 50 and only["unscored"]["positions"] == n * 50
    assert only["unscored"]["estimate"]["mean_kl"] > 4.0  # the prompt text's KL stays reported
    assert only["estimate"]["mean_kl"] < 2e-4


def test_a_scored_list_must_cover_the_arm(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    arm = write_arm(tmp_path, "cover", *synthetic(rng, 8, 40, 1e-4, 0.0), windows=8)
    with pytest.raises(iqs.Unusable):  # window 7 is missing
        iqs.load_arm(arm, write_scored(tmp_path / "missing.json", {w: (40, [[1, 40]]) for w in range(7)}))
    with pytest.raises(iqs.Unusable):  # nothing is scored
        iqs.load_arm(arm, write_scored(tmp_path / "none.json", {w: (40, []) for w in range(8)}))


def test_a_set_of_few_long_windows_resamples_blocks(tmp_path: Path) -> None:
    # 4 windows of 16,384 positions: too few windows to bootstrap, 16 blocks of 4,096 enough
    rng = np.random.default_rng(12)
    arm = write_arm(tmp_path, "long", *synthetic(rng, 4, 16384, 2e-4, 0.002), windows=4)
    loaded = iqs.load_arm(arm)
    assert np.unique(loaded.unit).size == 16
    assert iqs.decide(loaded, FAST)["bound"]["mean_kl"] > 0
    # four short windows have nothing to resample
    small = write_arm(tmp_path / "s", "small", *synthetic(rng, 4, 500, 2e-4, 0.002), windows=4)
    with pytest.raises(iqs.Unusable):
        iqs.decide(iqs.load_arm(small), FAST)

def test_a_long_set_arm_reads_its_main_set_twin_with_the_main_scored_list(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    long_arm = write_arm(tmp_path / "long", "arm", *synthetic(rng, 4, 16384, 2e-4, 0.002), windows=4)
    twin = write_arm(tmp_path / "main", "twin", *synthetic(rng, 8, 100, 2e-4, 0.002), windows=8, route="noise_twin")
    long_scored = write_scored(tmp_path / "long-scored.json", {w: (16384, [[1, 16384]]) for w in range(4)})
    main_scored = write_scored(tmp_path / "main-scored.json", {w: (100, [[1, 100]]) for w in range(8)})
    # the twin's windows (0..7) are not the long set's (0..3): its own list must be the one that reads it
    plan = {"decisions": [{"label": "long", "arm": "long/arm", "twin": "main/twin", "scored": "long-scored.json",
                           "twin_scored": "main-scored.json"}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert iqs.run_certificate(tmp_path / "plan.json", tmp_path / "cert", FAST) in (0, 1)
    certificate = json.loads((tmp_path / "cert" / "certificate.json").read_text(encoding="utf-8"))
    assert not certificate["unusable"] and len(certificate["decisions"]) == 1
    assert long_scored.exists() and main_scored.exists()


def test_an_arm_in_parts_is_the_arm_whole(tmp_path: Path) -> None:
    # A corpus layer scored later: windows 0-31 in one part, 32-63 in another, decided as one arm.
    rng = np.random.default_rng(14)
    kl, flip, dnll = synthetic(rng, 64, 100, 2e-4, 0.002)
    whole = write_arm(tmp_path / "whole", "arm", kl, flip, dnll, windows=64)
    low = write_arm(tmp_path / "parts", "low", kl[:3200], flip[:3200], dnll[:3200], windows=32)
    high = write_arm(tmp_path / "parts", "high", kl[3200:], flip[3200:], dnll[3200:], windows=32, first=32)
    one, two = iqs.decide(iqs.load_arm(whole), FAST), iqs.decide(iqs.load_arm([low, high]), FAST)
    assert two["windows"] == 64 and two["positions"] == one["positions"]
    assert two["estimate"] == one["estimate"] and two["bound"] == one["bound"]
    assert two["domains"] == one["domains"]


def test_parts_must_hold_disjoint_windows_of_one_route(tmp_path: Path) -> None:
    rng = np.random.default_rng(15)
    a = write_arm(tmp_path, "a", *synthetic(rng, 32, 50, 2e-4, 0.002), windows=32)
    b = write_arm(tmp_path, "b", *synthetic(rng, 32, 50, 2e-4, 0.002), windows=32, first=16)
    with pytest.raises(iqs.Unusable, match="repeats window"):
        iqs.load_arm([a, b])
    c = write_arm(tmp_path, "c", *synthetic(rng, 32, 50, 2e-4, 0.002), windows=32, first=32, route="reproducible")
    with pytest.raises(iqs.Unusable, match="differs"):
        iqs.load_arm([a, c])


def test_clipped_kl_catches_a_typical_position_error_a_heavy_tailed_floor_hides(tmp_path: Path) -> None:
    # A floor whose mean a few near-tie positions decide: 2 x its mean is generous, 2 x its clipped mean is not.
    rng = np.random.default_rng(16)
    kl, flip, dnll = synthetic(rng, 32, 200, 1e-4, 0.002)
    kl[rng.choice(kl.size, 10, replace=False)] = 5.0
    twin = write_arm(tmp_path, "twin", kl, flip, dnll, windows=32, route="noise_twin")
    arm = write_arm(tmp_path, "arm", *synthetic(rng, 32, 200, 1e-3, 0.002), windows=32)
    decision = iqs.decide(iqs.load_arm(arm), FAST, iqs.floor_of(twin))
    failing = {c["name"] for c in decision["criteria"] if not c["passed"]}
    assert failing == {"C4 clipped KL"}, decision["criteria"]
