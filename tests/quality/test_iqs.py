"""IQS-1 decisions (tools/quality/iqs.py) on synthetic arms: what passes, what fails, how the noise floor
moves a budget, and how a run plan becomes a certificate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.quality import iqs

DOMAINS = ("prose", "code", "math", "chat")


def write_arm(directory: Path, name: str, kl: np.ndarray, flip: np.ndarray, dnll: np.ndarray, windows: int,
              route: str = "default", nonfinite: int = 0) -> Path:
    """An infernix-certify arm (prefix.json, prefix.positions.csv) and its reference's window list."""
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / "ref.bin"
    meta_windows = [{"index": w, "id": f"doc{w}", "domain": DOMAINS[w % len(DOMAINS)]} for w in range(windows)]
    Path(str(reference) + ".json").write_text(json.dumps({"windows": meta_windows}), encoding="utf-8")
    prefix = directory / name
    per = kl.size // windows
    lines = ["window,segment,position,kl,top_mass,flip,dnll"]
    for i in range(kl.size):
        lines.append(f"{i // per},-1,{i % per + 1},{float(kl[i]):.9g},0.95,{int(flip[i])},{float(dnll[i]):.9g}")
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
    assert "F1 mean KL" in failing and not decision["passed"]


def test_one_bad_domain_fails_on_its_own(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    kl, flip, dnll = synthetic(rng, 32, 200, 1e-4, 0.001)
    # Every code window (w % 4 == 1) far worse; the overall mean stays below F1.
    for w in range(1, 32, 4):
        kl[w * 200:(w + 1) * 200] *= 40
    arm = write_arm(tmp_path, "domain", kl, flip, dnll, windows=32)
    decision = iqs.decide(iqs.load_arm(arm), FAST)
    failing = {c["name"] for c in decision["criteria"] if not c["passed"]}
    assert "F5 code mean_kl" in failing
    assert "F5 prose mean_kl" not in failing


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
    floor = {"mean_kl": 1e-5, "flip_rate": 1e-4, "p999_kl": 1e-3, "mean_dnll": 0.0}
    assert iqs.budgets(iqs.POLICY, floor) == iqs.budgets(iqs.POLICY, None)


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
    assert "IQS-1 certificate" in (tmp_path / "cert" / "certificate.html").read_text(encoding="utf-8")
