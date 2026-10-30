"""tools/bench/abba.py: statistics, schedule and per-sample evidence checks (no GPU)."""

import json
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
    record = "INFERNIX_TMP_STAGING_SLOTS=64 | INFERNIX_TMP_QSA_PROMPT_V2=unset"
    assert abba.parse_switches(record) == {"INFERNIX_TMP_STAGING_SLOTS": "64",
                                           "INFERNIX_TMP_QSA_PROMPT_V2": "unset"}
    arm = abba.Arm("slots", env={"INFERNIX_TMP_STAGING_SLOTS": "64"},
                   args=["--fast-path", "dflash2-pairs"],
                   expect_switches={"INFERNIX_TMP_STAGING_SLOTS": "64"})
    good = abba.Sample("slots", 0, 0, {"config": {"temporary_switches": record,
                                                       "fast_paths": "dflash2-pairs"}},
                       "PCIe link to the GPU: 27.9 GB/s\n", 0, 1.0)
    abba.check_sample(good, arm, 20.0)
    assert good.problems == []
    bad = abba.Sample("slots", 0, 0, {"config": {"temporary_switches": "INFERNIX_TMP_STAGING_SLOTS=unset",
                                                      "fast_paths": ""}},
                      "PCIe link to the GPU: 3.0 GB/s\n", 0, 1.0)
    abba.check_sample(bad, arm, 20.0)
    assert len(bad.problems) == 3  # switch unset, fast path missing, link probe flake


def machine_report(width=16, *, generation=5, max_width=16, decode=100.0, probe=True):
    """A report with the machine probe and one test, as infernix_bench writes them."""
    machine = {"pcie_generation": generation,
               "pcie_width": width if probe else None,  # null when the probe measured nothing
               "pcie_max_generation": 5, "pcie_max_width": max_width, "h2d_gbps": 57.8}
    return {"machine": machine, "config": {"temporary_switches": "", "fast_paths": ""},
            "tests": [{"label": "t", "decode_output_tok_s_mean": decode}]}


def test_link_of_reads_the_report_probe_and_says_what_it_cannot():
    link = abba.link_of(machine_report(width=8, max_width=16))
    assert (link.generation, link.width, link.max_width) == (5, 8, 16)
    assert str(link) == "gen5 x8"
    assert link.full == "gen5 x8 of gen5 x16"
    assert link.order < abba.link_of(machine_report(width=16)).order  # narrowest sorts first

    # A report whose probe measured nothing still shows the generation it read; one with no machine
    # block (an older schema) shows neither.
    unprobed = abba.link_of(machine_report(probe=False))
    assert unprobed.width is None and str(unprobed) == "gen5 x?" and unprobed.order == (0, 5)
    for report in ({"tests": []}, None):
        blank = abba.link_of(report)
        assert blank.width is None and str(blank) == "gen? x?" and blank.order == (0, 0)


def test_the_link_guard_says_what_a_width_fails_to_show():
    assert abba.link_problem(abba.link_of(machine_report(width=8)), None) is None
    assert abba.link_problem(abba.link_of(machine_report(width=8)), 8) is None
    assert abba.link_problem(abba.link_of(machine_report(width=8)), 16) == "link width gen5 x8 (expected x16)"
    assert "cannot be checked against x16" in abba.link_problem(abba.link_of({"tests": []}), 16)


def test_check_sample_drops_a_sample_that_ran_on_the_wrong_link():
    arm = abba.Arm("base")
    report = {"config": {"temporary_switches": "", "fast_paths": ""},
              "machine": {"pcie_width": 8, "pcie_generation": 5}}
    wrong = abba.Sample("base", 0, 0, report, "PCIe link to the GPU: 28.9 GB/s\n", 0, 1.0)
    abba.check_sample(wrong, arm, 20.0, expect_width=16)
    assert wrong.problems == ["link width gen5 x8 (expected x16)"]
    # The bandwidth probe alone cannot see this: an x8 card still reads a plausible 28.9 GB/s.
    same_but_unguarded = abba.Sample("base", 0, 0, report, "PCIe link to the GPU: 28.9 GB/s\n", 0, 1.0)
    abba.check_sample(same_but_unguarded, arm, 20.0)
    assert same_but_unguarded.problems == []

    unprobed = abba.Sample("base", 0, 0, {"config": {}, "machine": {}}, "", 0, 1.0)
    abba.check_sample(unprobed, arm, 20.0, expect_width=16)
    assert len(unprobed.problems) == 1 and "no PCIe link width in the report" in unprobed.problems[0]

    right = abba.Sample("base", 0, 0, machine_report(width=16), "", 0, 1.0)
    abba.check_sample(right, arm, 20.0, expect_width=16)
    assert right.problems == []


def test_a_link_that_changed_mid_run_excludes_its_samples_instead_of_pairing_them():
    arms = {"base": abba.Arm("base"), "new": abba.Arm("new")}
    widths = [16, 16, 8, 16, 16]

    def build():
        samples = []
        for block, (base, new, width) in enumerate(zip([100, 101, 99, 100, 102], [104, 105, 103, 103, 106],
                                                       widths)):
            for arm, value in (("base", base), ("new", new)):
                samples.append(abba.Sample(arm, block, block, machine_report(width=width, decode=value),
                                           "", 0, 1.0))
        return samples

    metric = [{"name": "decode_output_tok_s"}]
    samples = build()
    results = abba.analyse(samples, arms, "base", metric, None, 0.15)
    assert results["link"]["width"] == 16 and results["link"]["widths"] == [8, 16]
    assert len(results["excluded"]) == 2  # both of the x8 block's samples
    assert all("differs from this run's x16" in problem for item in results["excluded"]
               for problem in item["problems"])
    assert results["metrics"]["decode_output_tok_s"]["arms"][0]["pairs"] == 4

    # With an explicit expectation the width is that expectation, not the run's own majority, so the
    # four x16 blocks leave and the one x8 block is the only pair left.
    samples = build()
    for sample in samples:  # what the runner does for every process it starts
        abba.check_sample(sample, arms[sample.arm], 0.0, 8)
    results = abba.analyse(samples, arms, "base", metric, None, 0.15, expect_width=8)
    assert results["link"]["width"] == 8 and results["link"]["run"] == "gen5 x8"
    assert len(results["excluded"]) == 8
    assert all("expected x8" in problem for item in results["excluded"] for problem in item["problems"])
    assert results["metrics"]["decode_output_tok_s"]["arms"][0]["pairs"] == 1


def test_a_report_without_a_probe_is_counted_not_quietly_ignored():
    samples = [abba.Sample("base", 0, 0, machine_report(probe=False), "", 0, 1.0),
               abba.Sample("base", 1, 0, machine_report(width=8), "", 0, 1.0),
               abba.Sample("base", 2, 0, {"tests": []}, "", 0, 1.0)]
    link = abba.apply_link_guard(samples, None)
    assert link["width"] == 8 and link["without_probe"] == 2 and link["run"] == "gen5 x8"
    assert all(not s.problems for s in samples)  # with no expectation, an old report is not a fault

    # A run asked for one width and given another: the evidence says so instead of naming a link it
    # never saw.
    link = abba.apply_link_guard(samples, 16)
    assert link == {"run": None, "width": 16, "expected": 16, "widths": [8], "without_probe": 2}


def test_the_run_refuses_to_start_on_a_link_it_did_not_ask_for(tmp_path, monkeypatch, capsys):
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps({"bench": "infernix_bench", "control": "base",
                                     "arms": [{"name": "base"}, {"name": "new"}],
                                     "metrics": [{"name": "decode_output_tok_s"}]}), encoding="utf-8")
    calls = []

    def stub(arm, config, out, tag, extra, timeout):
        calls.append(arm.name)
        return abba.Sample(arm.name, -1, -1, machine_report(width=8, decode=100.0), "", 0, 1.0)

    monkeypatch.setattr(abba, "run_sample", stub)
    code = abba.main([str(arms_file), "--out", str(tmp_path / "out"), "--expect-width", "16"])
    assert code == 4
    assert calls == ["base"]  # only the discarded warm-up ran; the measured blocks never started
    printed = capsys.readouterr()
    assert "refusing to start: link width gen5 x8 (expected x16)" in printed.err
    assert "link probe in the warm-up report: gen5 x8 of gen5 x16" in printed.out


def test_the_arm_file_can_carry_the_link_guard_for_everyone_who_uses_it(tmp_path, monkeypatch, capsys):
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps({"bench": "infernix_bench", "control": "base", "expect_width": 16,
                                     "arms": [{"name": "base"}, {"name": "new"}],
                                     "metrics": [{"name": "decode_output_tok_s"}]}), encoding="utf-8")
    monkeypatch.setattr(abba, "run_sample",
                        lambda arm, config, out, tag, extra, timeout: abba.Sample(
                            arm.name, -1, -1, machine_report(width=8), "", 0, 1.0))
    assert abba.main([str(arms_file), "--out", str(tmp_path / "out")]) == 4
    capsys.readouterr()


def test_a_run_starts_and_reports_the_link_when_the_probe_agrees(tmp_path, monkeypatch, capsys):
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps({"bench": "infernix_bench", "control": "base",
                                     "arms": [{"name": "base"}, {"name": "new"}],
                                     "metrics": [{"name": "decode_output_tok_s"}]}), encoding="utf-8")
    calls: list[str] = []

    def stub(arm, config, out, tag, extra, timeout):
        calls.append(tag)
        block = int(tag.split("-")[0][1:]) if tag.startswith("b") else -1
        decode = 100.0 + block + (4.0 if arm.name != "base" else 0.0)
        return abba.Sample(arm.name, block, 0, machine_report(width=16, decode=decode), "", 0, 1.0)

    monkeypatch.setattr(abba, "run_sample", stub)
    out = tmp_path / "out"
    assert abba.main([str(arms_file), "--out", str(out), "--expect-width", "16", "--blocks", "3",
                     "--no-noise-arm"]) == 0
    printed = capsys.readouterr().out
    assert "link probe in the warm-up report: gen5 x16 of gen5 x16" in printed
    assert len(calls) == 7  # the warm-up plus three blocks of two arms
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["link"] == {"run": "gen5 x16", "width": 16, "expected": 16, "widths": [16],
                              "without_probe": 0}
    row = summary["metrics"]["decode_output_tok_s"]["arms"][0]
    assert row["pairs"] == 3 and row["better"] == 3


def test_report_and_summary_record_the_link_the_run_ran_on(tmp_path):
    arms = {"base": abba.Arm("base"), "new": abba.Arm("new")}
    samples = []
    for block, (base, new) in enumerate([(100, 104), (101, 105), (99, 103)]):
        samples.append(abba.Sample("base", block, 0, machine_report(width=8, decode=base), "", 0, 1.0))
        samples.append(abba.Sample("new", block, 1, machine_report(width=8, decode=new), "", 0, 1.0))
    results = abba.analyse(samples, arms, "base", [{"name": "decode_output_tok_s"}], None, 0.15)
    out = tmp_path / "out"
    out.mkdir()
    abba.write_report(out, results, {"control": "base"}, None, 0.05)
    assert "PCIe link: `gen5 x8`" in (out / "report.md").read_text(encoding="utf-8")
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["link"]["width"] == 8

    # A run that was guarded to one width and probed another says so in the same line.
    results["link"] = abba.apply_link_guard(
        [abba.Sample("base", 0, 0, machine_report(width=8), "", 0, 1.0)], 16)
    abba.write_report(out, results, {"control": "base"}, None, 0.05)
    assert "PCIe link: no report probed the guarded x16" in (out / "report.md").read_text(
        encoding="utf-8")

    results["link"] = abba.apply_link_guard(
        [abba.Sample("base", 0, 0, machine_report(width=16), "", 0, 1.0)], 16)
    abba.write_report(out, results, {"control": "base"}, None, 0.05)
    assert "PCIe link: `gen5 x16`, guarded at x16" in (out / "report.md").read_text(encoding="utf-8")


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
