"""The statistics and scoring behind IQS-2's end-to-end checks (tools/quality/iqs_e2e.py): the chi-square tail,
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


def test_task_audit_is_paired_non_inferiority_on_the_items_that_differ() -> None:
    # 250 items, 16 answered differently, 2 more lost than gained (the 35B run of 2026-10-24): within 5 points
    base = [True] * 200 + [False] * 50
    fast = list(base)
    for i in range(9):
        fast[i] = False           # only the reproducible route answers these
    for i in range(200, 207):
        fast[i] = True            # only the default route answers these
    result = iqs_e2e.paired_loss(base, fast, 0.05)
    assert (result["only_reproducible"], result["only_default"], result["flips"]) == (9, 7, 16)
    assert result["loss"] == pytest.approx(2 / 250) and result["passed"]
    # a route that loses 25 items it used to answer fails, whatever it gains back by chance
    worse = [False] * 25 + base[25:]
    assert not iqs_e2e.paired_loss(base, worse, 0.05)["passed"]


def test_golden_hashes_are_recorded_then_disclosed(tmp_path) -> None:
    # A build may change the reproducible route's bits by a declared numerics change (model-quality.md 6.1):
    # a changed hash is reported with the new hashes, never failed.
    store = tmp_path / "golden.json"
    first = iqs_e2e.golden(store, "a.infernix", ["x", "y"])["golden"]
    assert first["recorded"] and store.is_file()
    assert iqs_e2e.golden(store, "a.infernix", ["x", "y"])["golden"]["equal"]
    changed = iqs_e2e.golden(store, "a.infernix", ["x", "z"])["golden"]
    assert not changed["equal"] and changed["changed"] == [1] and changed["hashes"] == ["x", "z"]
    assert "passed" not in changed
