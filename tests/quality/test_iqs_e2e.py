"""The statistics and scoring behind IQS-1's end-to-end checks (tools/quality/iqs_e2e.py): the chi-square tail,
the homogeneity test speculation is judged by, loop detection and answer extraction."""

from __future__ import annotations

import json
import random

import pytest

from tools.quality import iqs_e2e


@pytest.mark.parametrize("x,k,p", [(3.841459, 1, 0.05), (18.307038, 10, 0.05), (6.634897, 1, 0.01),
                                   (0.0, 3, 1.0), (124.342113, 100, 0.05), (2.0, 4, 0.7357589)])
def test_chi2_tail_matches_tables(x, k, p):
    assert iqs_e2e.chi2_sf(x, k) == pytest.approx(p, rel=1e-5, abs=1e-12)


def test_homogeneity_accepts_one_distribution_and_rejects_two():
    rng = random.Random(5)
    words = ["blue", "red", "green", "grey", "teal"]
    weights = [0.4, 0.3, 0.15, 0.1, 0.05]
    a = rng.choices(words, weights, k=400)
    b = rng.choices(words, weights, k=400)
    assert iqs_e2e.homogeneity(a, b)[0] > 1e-3
    shifted = rng.choices(words, [0.25, 0.45, 0.15, 0.1, 0.05], k=400)
    assert iqs_e2e.homogeneity(a, shifted)[0] < 1e-3


def test_homogeneity_merges_rare_categories():
    a = ["x"] * 50 + [f"rare{i}" for i in range(50)]
    b = ["x"] * 50 + [f"other{i}" for i in range(50)]
    p, dof = iqs_e2e.homogeneity(a, b)
    assert dof == 1 and p == pytest.approx(1.0)


def test_looping_and_answers():
    assert iqs_e2e.looping(" ".join(["so the total is twelve and we check again"] * 20))
    assert not iqs_e2e.looping(" ".join(f"step {i} adds {i * 3}" for i in range(200)))
    assert iqs_e2e.gsm_answer("First 3 + 4 = 7, then $1,250.\nAnswer: 1,250.") == "1250"
    assert iqs_e2e.gsm_answer("no number here") is None


def test_the_comparison_is_proven_by_any_artifacts_negative_control(tmp_path: Path) -> None:
    from types import SimpleNamespace

    def result(name: str, control: int | None) -> str:
        path = tmp_path / f"{name}.json"
        body = {"passed": True} if control is None else {"passed": True, "negative_control": {"differences": control}}
        path.write_text(json.dumps(body), encoding="utf-8")
        return str(path)

    seen = iqs_e2e.check_sensitivity(SimpleNamespace(results=[result("q27", 5), result("q35", 0), result("fp8", None)]))
    assert seen["passed"] and seen["controls"] == {"q27": 5, "q35": 0}
    blind = iqs_e2e.check_sensitivity(SimpleNamespace(results=[result("a", 0), result("b", 0)]))
    assert not blind["passed"]
    assert not iqs_e2e.check_sensitivity(SimpleNamespace(results=[result("c", None)]))["passed"]
