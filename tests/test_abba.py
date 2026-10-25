"""tools/bench/abba.py: statistics, schedule and per-sample evidence checks (no GPU)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))

import abba  # noqa: E402


def test_wilcoxon_exact_matches_known_values():
    # Five positive differences: the most extreme of 2^5 sign patterns, two-sided p = 2/32.
    assert math.isclose(abba.wilcoxon_signed_rank([1, 2, 3, 4, 5]), 2 / 32)
    # Symmetric differences give p = 1.
    assert abba.wilcoxon_signed_rank([1, -1, 2, -2]) == 1.0
    # Zeros are dropped; nothing left means no evidence.
    assert abba.wilcoxon_signed_rank([0.0, 0.0]) == 1.0


def test_wilcoxon_ties_and_normal_branch_agree_in_direction():
    small = abba.wilcoxon_signed_rank([1.0, 1.0, 2.0, 3.0, -0.5, 4.0])
    assert 0.0 < small < 0.2
    large = abba.wilcoxon_signed_rank([1.0 + 0.01 * i for i in range(30)])
    assert large < 1e-5


def test_holm_is_monotone_and_bounded():
    adjusted = abba.holm([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]
    assert all(p <= 1.0 for p in abba.holm([0.9, 0.8, 0.7]))


def test_schedule_balances_positions():
    order = abba.schedule(["a", "b"], 4)
    assert order == [["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]]
    three = abba.schedule(["a", "b", "c"], 6)
    firsts = [block[0] for block in three]
    assert set(firsts) == {"a", "b", "c"}


def test_parse_switches_and_sample_checks():
    record = "INFERNIX_TMP_NGRAM_AGREE=1 | INFERNIX_TMP_QSA_PROMPT_V2=unset"
    assert abba.parse_switches(record) == {"INFERNIX_TMP_NGRAM_AGREE": "1",
                                           "INFERNIX_TMP_QSA_PROMPT_V2": "unset"}
    arm = abba.Arm("agree", env={"INFERNIX_TMP_NGRAM_AGREE": "1"},
                   args=["--fast-path", "dflash2-pairs"],
                   expect_switches={"INFERNIX_TMP_NGRAM_AGREE": "1"})
    good = abba.Sample("agree", 0, 0, {"config": {"temporary_switches": record,
                                                       "fast_paths": "dflash2-pairs"}},
                       "PCIe link to the GPU: 27.9 GB/s\n", 0, 1.0)
    abba.check_sample(good, arm, 20.0)
    assert good.problems == []
    bad = abba.Sample("agree", 0, 0, {"config": {"temporary_switches": "INFERNIX_TMP_NGRAM_AGREE=unset",
                                                      "fast_paths": ""}},
                      "PCIe link to the GPU: 3.0 GB/s\n", 0, 1.0)
    abba.check_sample(bad, arm, 20.0)
    assert len(bad.problems) == 3  # switch unset, fast path missing, link probe flake


def test_analyse_pairs_by_block():
    def report(value):
        return {"tests": [{"label": "t", "decode_output_tok_s_mean": value}]}

    arms = {"base": abba.Arm("base"), "new": abba.Arm("new")}
    samples = []
    for block, (base, new) in enumerate([(100, 104), (101, 105), (99, 103), (100, 103), (102, 106)]):
        samples.append(abba.Sample("base", block, 0, report(base), "", 0, 1.0))
        samples.append(abba.Sample("new", block, 1, report(new), "", 0, 1.0))
    results = abba.analyse(samples, arms, "base", [{"name": "decode_output_tok_s"}], None, 0.15)
    row = results["metrics"]["decode_output_tok_s"]["arms"][0]
    assert row["pairs"] == 5 and row["better"] == 5
    assert math.isclose(row["p"], 2 / 32)
    assert row["pooled_mean_ratio"] > 1.03


def test_replay_bench_metrics_read_the_last_release_line():
    def log(per_round):
        return ("replay bench: 10 rounds, 120.000000 ms device time, 20.000000 ms CPU waits, "
                "100.000000 ms less waits (11.000000 ms a round); residency frozen\n"
                f"replay bench: 20 rounds, 240.000000 ms device time, 40.000000 ms CPU waits, "
                f"200.000000 ms less waits ({per_round:.6f} ms a round); residency frozen\n")

    assert abba.replay_value(log(10.0), "replay_ms_per_round") == 10.0
    assert abba.replay_value(log(10.0), "replay_wait_ms") == 40.0
    assert abba.replay_value("no line", "replay_ms_per_round") is None
    arms = {"base": abba.Arm("base"), "new": abba.Arm("new")}
    samples = []
    for block, (base, new) in enumerate([(10.0, 9.9), (10.1, 10.0), (10.0, 9.95), (10.05, 9.97), (10.02, 9.94)]):
        samples.append(abba.Sample("base", block, 0, {"tests": []}, log(base), 0, 1.0))
        samples.append(abba.Sample("new", block, 1, {"tests": []}, log(new), 0, 1.0))
    results = abba.analyse(samples, arms, "base", [{"name": "replay_ms_per_round", "higher_is_better": False}],
                           None, 0.15)
    row = results["metrics"]["replay_ms_per_round"]["arms"][0]
    assert row["pairs"] == 5 and row["better"] == 5
