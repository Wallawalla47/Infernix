"""tools/bench/join_reports.py: report join, link-width guards and comparability notes (no GPU)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))

import join_reports  # noqa: E402


def report(width=8, *, generation=5, decode=100.0, stddev=0.5, prefill=None, load=13.5, h2d=28.9,
           gpu="RTX 5090", kv="int8", label="tg512", concurrency=1, backend="none", model="Flash-Next",
           switches="INFERNIX_TMP_A=unset | INFERNIX_TMP_B=unset", probe=True, max_width=16):
    """A schema-v26 infernix_bench report, with only what the join reads."""
    machine = {"pcie_generation": generation, "pcie_max_generation": 5, "pcie_max_width": max_width,
               "h2d_gbps": h2d, "ram_read_gbps": 74.0, "ram_read_threads": 32}
    if probe:
        machine["pcie_width"] = width
    else:  # what infernix_bench writes when the probe could not measure anything
        machine["pcie_width"] = None
    return {
        "schema_version": 26, "artifact_type": "infernix_bench_report", "tool": "infernix_bench",
        "machine": machine,
        "artifact": {"path": "/models/dense8.infernix", "file_size_bytes": 32_000_000_000},
        "environment": {"gpu_name": gpu, "cuda_runtime_version": "13.3",
                        "cuda_driver_version": "610.57.04"},
        "load": {"architecture": "Qwen4ExpForCausalLM", "name": model, "load_seconds": load},
        "config": {"concurrency": concurrency, "kv_cache": kv, "repetitions": 5, "warmup": 2,
                   "speculative_backend": backend, "fast_paths": "q8-sliced-k",
                   "temporary_switches": switches, "expert_state": "/per-run/expert.cache"},
        "tests": [{"label": label, "decode_output_tok_s_mean": decode,
                   "decode_output_tok_s_stddev": stddev, "output_tok_s_mean": decode * 0.97,
                   "output_tok_s_stddev": stddev, "prefill_tok_s_mean": prefill,
                   "prefill_tok_s_stddev": None}],
    }


def write_run(root: Path, label: str, reports: dict[str, dict]) -> Path:
    directory = root / label / "report"
    directory.mkdir(parents=True)
    for name, payload in reports.items():
        (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return root / label


def read_out(out: Path) -> tuple[str, dict]:
    return (out / "join.md").read_text(encoding="utf-8"), json.loads(
        (out / "join.json").read_text(encoding="utf-8"))


def test_link_join_ratios_the_wider_link_over_the_narrower_one(tmp_path, capsys):
    write_run(tmp_path, "run-x8", {"010_plain": report(width=8, decode=100.0)})
    write_run(tmp_path, "run-x16", {"010_plain": report(width=16, decode=110.0, h2d=57.8)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    markdown, payload = read_out(out)
    assert "base `gen5 x8`" in markdown
    assert "1.100x" in markdown  # the width ratio, read off the reports' own probes
    row = [r for r in payload["rows"] if r["metric"] == "decode_output_tok_s"]
    assert {r["group"]: r["mean"] for r in row} == {"gen5 x8": 100.0, "gen5 x16": 110.0}
    assert "## Machine probes" in markdown and "h2d_gbps" in markdown
    assert "2.00x" in markdown
    assert (out / "join.csv").read_text(encoding="utf-8").startswith("metric,arm,test,")
    capsys.readouterr()


def test_join_groups_reports_by_probe_not_by_run_so_two_runs_of_one_width_pooled(tmp_path, capsys):
    write_run(tmp_path, "a-x16", {"010_plain": report(width=16, decode=100.0)})
    write_run(tmp_path, "b-x16", {"010_plain": report(width=16, decode=120.0)})
    write_run(tmp_path, "c-x8", {"010_plain": report(width=8, decode=100.0)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    _, payload = read_out(out)
    assert payload["meta"]["groups"] == ["gen5 x8", "gen5 x16"]  # narrowest first, so the ratio is up
    x16 = [r for r in payload["rows"]
           if r["metric"] == "decode_output_tok_s" and r["group"] == "gen5 x16"][0]
    assert x16["reports"] == 2 and x16["mean"] == 110.0
    assert sorted(x16["runs"]) == ["a-x16", "b-x16"]
    capsys.readouterr()


def test_join_drops_a_run_whose_reports_disagree_on_width(tmp_path, capsys):
    write_run(tmp_path, "moved", {"010_plain": report(width=16, decode=110.0),
                                  "020_mtp": report(width=8, decode=60.0)})
    write_run(tmp_path, "clean", {"010_plain": report(width=8, decode=100.0)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    markdown, payload = read_out(out)
    assert "the card moved, or the link retrained, mid-run" in markdown
    assert "--allow-mixed-link to keep it" in markdown
    # Only the clean run's x8 reports are left, and one group forms no ratio.
    assert payload["rows"] == []
    assert "Only one group (`gen5 x8`)" in markdown
    assert any("moved" in note for note in payload["meta"]["guards"])
    capsys.readouterr()

    kept = tmp_path / "kept"
    assert join_reports.main([str(tmp_path), "--allow-mixed-link", "--out", str(kept)]) == 0
    markdown, payload = read_out(kept)
    assert "kept, each report joining on its own probe" in markdown
    assert sorted({run for r in payload["rows"] for run in r["runs"]}) == ["clean", "moved"]
    # Each report joins on its own probe: the moved run's x8 arm lands in the x8 group, its x16 arm
    # in the x16 group, and each key keeps its own reports.
    decoded = [r for r in payload["rows"] if r["metric"] == "decode_output_tok_s"]
    assert {(r["arm"], r["group"], r["mean"]) for r in decoded} == {
        ("010_plain", "gen5 x8", 100.0), ("010_plain", "gen5 x16", 110.0),
        ("020_mtp", "gen5 x8", 60.0)}
    capsys.readouterr()


def test_join_drops_and_counts_reports_that_recorded_no_link_width(tmp_path, capsys):
    write_run(tmp_path, "x8", {"010_plain": report(width=8), "020_mtp": report(probe=False)})
    write_run(tmp_path, "x16", {"010_plain": report(width=16, decode=110.0)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    markdown, payload = read_out(out)
    assert "1 of 2 report(s) recorded no PCIe link width" in markdown
    arms = {r["arm"] for r in payload["rows"]}
    assert arms == {"010_plain"}  # the unprobed report is not placed on the link axis
    capsys.readouterr()


def test_join_reports_only_what_both_links_measured_and_names_the_rest(tmp_path, capsys):
    write_run(tmp_path, "x8", {"010_plain": report(width=8), "030_only_here": report(width=8)})
    write_run(tmp_path, "x16", {"010_plain": report(width=16, decode=101.0)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    markdown, _ = read_out(out)
    assert "## Not measured in every group" in markdown
    assert "`030_only_here`" in markdown
    capsys.readouterr()


def test_join_refuses_a_ratio_from_one_link_and_says_so(tmp_path, capsys):
    write_run(tmp_path, "x8", {"010_plain": report(width=8)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    markdown, payload = read_out(out)
    assert "no ratio" in markdown
    assert payload["rows"] == []
    assert (out / "join.csv").read_text(encoding="utf-8") == ""
    capsys.readouterr()


def test_join_lists_what_differs_between_the_runs_but_not_what_varies_between_arms(tmp_path, capsys):
    write_run(tmp_path, "x8", {"010_plain": report(width=8, backend="none"),
                               "040_dflash2": report(width=8, backend="dflash2")})
    write_run(tmp_path, "x16", {"010_plain": report(width=16, gpu="RTX 5090 Ti", backend="none"),
                                "040_dflash2": report(width=16, gpu="RTX 5090 Ti", backend="dflash2")})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    markdown, _ = read_out(out)
    assert "- gpu: x16 `RTX 5090 Ti`, x8 `RTX 5090`" in markdown
    assert "spec backend" not in markdown  # it differs between arms inside a run, not between runs
    assert "expert_state" not in markdown  # one state file per run directory, by construction
    capsys.readouterr()


def test_join_warns_when_a_group_pools_runs_that_are_not_the_same_build(tmp_path, capsys):
    write_run(tmp_path, "x8", {"010_plain": report(width=8)})
    write_run(tmp_path, "x16-old", {"010_plain": report(width=16, decode=110.0)})
    write_run(tmp_path, "x16-new", {"010_plain": report(width=16, decode=90.0,
                                                        switches="INFERNIX_TMP_A=unset | "
                                                                 "INFERNIX_TMP_B=unset | "
                                                                 "INFERNIX_TMP_C=unset")})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    markdown, payload = read_out(out)
    assert "pools 2 runs that differ in temporary switches" in markdown
    assert "its ± is a difference between those runs, not measurement spread" in markdown
    # The record itself is counted and diffed by name, not dumped in full.
    assert "`3 switches`" in markdown and "`2 switches`" in markdown
    assert "in x16-old against x16-new: lacks 1 (INFERNIX_TMP_C)" in markdown
    assert "different builds" in markdown
    assert any("pools 2 runs" in note for note in payload["meta"]["guards"])  # and it is in the json
    assert "INFERNIX_TMP_B=unset | INFERNIX_TMP_C" not in markdown
    capsys.readouterr()


def test_join_reads_an_abba_out_directory_and_pairs_its_blocks(tmp_path, capsys):
    out_dir = tmp_path / "abba-x8"
    out_dir.mkdir()
    (out_dir / "summary.json").write_text(json.dumps({"noisy": False}), encoding="utf-8")
    for block, decode in enumerate([100.0, 104.0]):
        (out_dir / f"b{block:02d}-0-base.json").write_text(json.dumps(report(width=8, decode=decode)),
                                                           encoding="utf-8")
        (out_dir / f"b{block:02d}-1-new.json").write_text(json.dumps(report(width=8, decode=decode + 4)),
                                                          encoding="utf-8")
    other = tmp_path / "abba-x16"
    other.mkdir()
    for block, decode in enumerate([110.0, 114.0]):
        (other / f"b{block:02d}-0-base.json").write_text(json.dumps(report(width=16, decode=decode)),
                                                         encoding="utf-8")

    out = tmp_path / "join"
    assert join_reports.main(["--run", f"a={out_dir}", "--run", f"b={other}", "--out",
                              str(out)]) == 0
    _, payload = read_out(out)
    rows = [r for r in payload["rows"] if r["metric"] == "decode_output_tok_s"]
    assert {(r["arm"], r["group"], r["reports"]) for r in rows} == {
        ("base", "gen5 x8", 2), ("new", "gen5 x8", 2), ("base", "gen5 x16", 2)}
    assert "summary" not in {r["arm"] for r in rows}  # abba's summary.json is not a report
    capsys.readouterr()


def test_join_drops_warm_up_arms_unless_asked(tmp_path, capsys):
    write_run(tmp_path, "x8", {"warmup": report(width=8, decode=42.0),
                               "000_warm": report(width=8, decode=43.0),
                               "010_plain": report(width=8, decode=100.0)})
    write_run(tmp_path, "x16", {"warmup": report(width=16, decode=44.0),
                                "010_plain": report(width=16, decode=110.0)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    _, payload = read_out(out)
    assert {r["arm"] for r in payload["rows"]} == {"010_plain"}
    assert "excluded arms as not measurements" in payload["meta"]["notes"][0]

    kept = tmp_path / "kept"
    assert join_reports.main([str(tmp_path), "--exclude", "nothing", "--out", str(kept)]) == 0
    _, payload = read_out(kept)
    assert {r["arm"] for r in payload["rows"]} == {"000_warm", "010_plain", "warmup"}
    capsys.readouterr()


def test_join_marks_a_lower_is_better_metric_and_takes_direction_overrides(tmp_path, capsys):
    write_run(tmp_path, "x8", {"010_plain": report(width=8, load=14.0)})
    write_run(tmp_path, "x16", {"010_plain": report(width=16, load=12.6)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--out", str(out)]) == 0
    markdown, payload = read_out(out)
    assert "## load_seconds (lower is better; base `gen5 x8`)" in markdown
    assert "0.900x" in markdown
    assert "## Machine probes" in markdown

    flipped = tmp_path / "flipped"
    assert join_reports.main([str(tmp_path), "--higher-is-better", "load_seconds", "--no-probes",
                              "--out", str(flipped)]) == 0
    markdown, payload = read_out(flipped)
    assert "## load_seconds (higher is better; base `gen5 x8`)" in markdown
    assert "## Machine probes" not in markdown
    assert payload["meta"]["probes"] == []
    capsys.readouterr()


def test_join_groups_by_run_when_the_link_is_not_the_question(tmp_path, capsys):
    write_run(tmp_path, "base-build", {"010_plain": report(width=8, decode=100.0)})
    write_run(tmp_path, "new-build", {"010_plain": report(width=8, decode=103.0)})
    out = tmp_path / "join"

    assert join_reports.main([str(tmp_path), "--group-by", "run", "--base", "new-build", "--out",
                              str(out)]) == 0
    markdown, payload = read_out(out)
    assert payload["meta"]["groups"] == ["base-build", "new-build"]
    assert payload["meta"]["base"] == "new-build"
    assert "grouped by run: each report's link is listed but not enforced" in markdown
    ratio = [r for r in payload["rows"] if r["group"] == "base-build"][0]
    assert ratio["ratio_vs_base"] == 100.0 / 103.0
    capsys.readouterr()


def test_join_rejects_a_bad_base_a_bad_run_and_an_empty_root(tmp_path, capsys):
    write_run(tmp_path, "x8", {"010_plain": report(width=8)})
    write_run(tmp_path, "x16", {"010_plain": report(width=16)})
    for argv in ([str(tmp_path), "--base", "gen9 x32"], ["--run", "novalue"],
                 ["--run", f"x={tmp_path}/absent"], [str(tmp_path / "empty")]):
        try:
            join_reports.main(argv)
        except SystemExit as exit_request:
            assert str(exit_request)
        else:
            raise AssertionError(f"--base/bad run accepted: {argv}")
    capsys.readouterr()


def test_explicit_runs_win_over_the_root_and_keep_their_labels(tmp_path, capsys):
    narrow = write_run(tmp_path, "20261028-095535Z-x8", {"010_plain": report(width=8)})
    write_run(tmp_path, "20261028-103019Z-x16", {"010_plain": report(width=16, decode=118.0)})
    inner = write_run(tmp_path / "chosen", "inner", {"010_plain": report(width=16, decode=120.0)})

    out = tmp_path / "join"
    assert join_reports.main([str(tmp_path), "--run", f"x16again={inner}", "--run", f"narrow={narrow}",
                              "--group-by", "run", "--out", str(out)]) == 0
    markdown, payload = read_out(out)
    assert payload["meta"]["runs"] == [{"run": "x16again", "dir": str(inner)},
                                       {"run": "narrow", "dir": str(narrow)}]
    decoded = [r for r in payload["rows"] if r["metric"] == "decode_output_tok_s"]
    assert {r["mean"]: r["group"] for r in decoded} == {120.0: "x16again", 100.0: "narrow"}
    assert "20261028-103019Z-x16" not in markdown  # the root's other run was not joined
    capsys.readouterr()
