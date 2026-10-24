"""The IQS-2 run plan (tools/quality/certify.py): every reference a step reads is made by an earlier step, every
decision has its arm and noise twin planned, and the declared designs use their design references."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.quality import certify


def plan(tmp_path: Path, keys: tuple[str, ...], mode: str = "full", extra: tuple[str, ...] = ()) -> tuple[list[dict], dict]:
    local = {k: {"path": f"/models/{k}.infernix", "ngram_volume": f"/models/{k}.ngram",
                 "draft_model": "/models/dflash2.infernix"} for k in keys}
    (tmp_path / "local.json").write_text(json.dumps(local), encoding="utf-8")
    run = tmp_path / "run"
    assert certify.main(["plan", "--artifacts", str(tmp_path / "local.json"), "--build", str(tmp_path / "build"),
                         "--store", str(tmp_path / "store"), "--out", str(run), "--mode", mode, *extra]) == 0
    return (json.loads((run / "steps.json").read_text(encoding="utf-8")),
            json.loads((run / "plan.json").read_text(encoding="utf-8")))


def test_every_input_is_made_before_it_is_read(tmp_path: Path) -> None:
    steps, decisions = plan(tmp_path, ("q27", "q27gw", "rc", "fp4l", "d8"), mode="release")
    made: set[str] = set()
    for step in steps:
        argv = step["argv"]
        for i, arg in enumerate(argv[:-1]):
            if arg == "--reference":
                assert argv[i + 1] + ".json" in made, f"{step['name']} reads {argv[i + 1]} before it is made"
        made.update(step["outputs"])
    for d in decisions["decisions"]:
        assert all(p + ".json" in made for p in d["arm"] + d["twin"]), d["label"]
    for d in decisions["disclosures"]:
        assert all(p + ".json" in made for p in d["arm"]), d["label"]


def test_declared_designs_use_their_design_reference(tmp_path: Path) -> None:
    steps, _ = plan(tmp_path, ("q27", "q27gw", "fp4l"))
    by_name = {s["name"]: s["argv"] for s in steps}
    reference = lambda name: by_name[name][by_name[name].index("--reference") + 1]
    assert reference("arm-q27-main-bf16-prefill-default").endswith("main-design-bf16.ref")
    assert reference("arm-q27gw-main-bf16-prefill-default").endswith("main-lossless.ref")
    assert reference("arm-q27gw-main-int8-prefill-default").endswith("main-design-int8.ref")
    assert "--expert-a4-activations" in by_name["arm-fp4l-a4-main-bf16-prefill-default"]
    assert Path(reference("arm-fp4l-a4-main-bf16-prefill-default")).parts[-2:] == ("fp4l-a4", "main-design-bf16.ref")
    # fp4l and fp4l-a4 share one lossless reference
    assert sum(1 for s in steps if s["name"] == "ref-fp4l-lossless") == 1


def test_missing_artifacts_are_left_out_and_said(tmp_path: Path) -> None:
    _, decisions = plan(tmp_path, ("q27gw",))
    assert "rc" in decisions["provenance"]["left_out"]
    assert all(d["label"].startswith("q27gw") for d in decisions["decisions"])


def test_a_step_finds_its_output_directory_made(tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    output = tmp_path / "store" / "key" / "made.ref"
    step = certify.Step(block="C1", name="probe", outputs=[str(output)], minutes=1,
                        argv=[sys.executable, "-c", "import sys, pathlib; pathlib.Path(sys.argv[1]).write_text('x')",
                              str(output)])
    code, _ = certify.run_step(step, tmp_path)
    assert code == 0 and output.exists()


def test_every_decision_names_its_scored_positions_and_they_are_made_first(tmp_path: Path) -> None:
    steps, plan_json = plan(tmp_path, ("q27gw", "rc"))
    made = {out for s in steps for out in s["outputs"]}
    names = [s["name"] for s in steps]
    for item in plan_json["decisions"] + plan_json["disclosures"]:
        assert item["scored"] in made, f"{item['label']} reads {item['scored']} that no step makes"
    assert names.index("mask-q27gw-main") < names.index("ref-q27gw-lossless")
    assert "mask-q27gw-long" in names and "mask-rc-long" in names
    assert any(i["scored"].endswith("long-scored.json") for i in plan_json["decisions"])
    assert "e2e-sensitivity" in names  # the identity comparison's sensitivity is a check of its own


def windows_of(argv: list[str]) -> set[int]:
    return set(map(int, argv[argv.index("--windows") + 1].split(",")))


def test_stored_items_are_made_a_layer_at_a_time_and_cover_the_main_set(tmp_path: Path) -> None:
    # References and noise twins are stored per 32-window layer, so a larger corpus makes only its new layers;
    # the arms run on every window.
    steps, plan_json = plan(tmp_path, ("q27gw",))
    by_name = {s["name"]: s for s in steps}
    count = certify.main_windows()
    layers = count // certify.LAYER
    assert layers >= 2
    for stem in ("ref-q27gw-lossless", "twin-q27gw", "twin-q27gw-int8", "ref-q27gw-design-nvfp4"):
        names = [stem] + [f"{stem}-L{k}" for k in range(1, layers)]
        covered = [windows_of(by_name[n]["argv"]) for n in names]
        assert set().union(*covered) == set(range(count)) and sum(map(len, covered)) == count, stem
    store = str(tmp_path / "store")
    default = by_name["arm-q27gw-main-bf16-prefill-default"]
    assert not default["outputs"][0].startswith(store) and "--windows" not in default["argv"]
    # every part of the reference is passed to an arm, layer 0 under its old name
    references = [a for i, a in enumerate(default["argv"][1:]) if default["argv"][i] == "--reference"]
    assert len(references) == layers and references[0].endswith("main-lossless.ref")


def test_a_kv_format_is_judged_against_the_twin_of_its_own_design(tmp_path: Path) -> None:
    _, plan_json = plan(tmp_path, ("q27gw",))
    by_label = {d["label"]: d for d in plan_json["decisions"]}
    assert all("main-twin-nvfp4" in t for t in by_label["q27gw main nvfp4 prefill-default"]["twin"])
    assert all("main-twin.w" in t for t in by_label["q27gw main bf16 prefill-default"]["twin"])
    assert all("main-twin.w" in t for t in by_label["q27gw long bf16 prefill-default"]["twin"])


def test_only_the_default_route_is_measured_and_one_request_decode_arms_chain_their_segments(tmp_path: Path) -> None:
    steps, plan_json = plan(tmp_path, ("q27", "q27gw"))
    by_name = {s["name"]: s["argv"] for s in steps}
    assert not any("--reproducible" in argv for argv in by_name.values())
    assert not any("reproducible" in d["label"] for d in plan_json["decisions"] + plan_json["disclosures"])
    for name in ("arm-q27gw-main-bf16-decode-default-one", "arm-q27-main-bf16-decode-default-tree",
                 "arm-q27gw-long-bf16-decode-default-one"):
        argv = by_name[name]
        assert "--prefix-reuse" in argv and argv[argv.index("--concurrency") + 1] == "1", name


def test_reuse_takes_the_unchanged_checks_and_the_single_request_arms_earlier_windows(tmp_path: Path) -> None:
    earlier = tmp_path / "earlier"
    (earlier / "arms").mkdir(parents=True)
    (earlier / "checks").mkdir()
    for name in ("q27gw-main-bf16-prefill-default", "q27gw-main-bf16-decode-default-one",
                 "q27gw-main-bf16-decode-default-stress"):
        (earlier / "arms" / f"{name}.json").write_text("{}", encoding="utf-8")
    (earlier / "checks" / "e2e-q27gw-identity.json").write_text('{"passed": true}', encoding="utf-8")
    steps, plan_json = plan(tmp_path, ("q27gw",), mode="release", extra=("--reuse", str(earlier)))
    by_name = {s["name"]: s for s in steps}
    decisions = {d["label"]: d for d in plan_json["decisions"]}
    # a single-request default arm: windows 0-31 from the earlier run, 32 onwards made here
    arm = decisions["q27gw main bf16 prefill-default"]["arm"]
    assert arm[0] == str(earlier / "arms" / "q27gw-main-bf16-prefill-default") and len(arm) == certify.main_windows() // 32
    assert "arm-q27gw-main-bf16-prefill-default" not in by_name
    assert min(windows_of(by_name["arm-q27gw-main-bf16-prefill-default-L1"]["argv"])) == 32
    # the stress arm always runs on every window
    assert decisions["q27gw main bf16 decode-default-stress"]["arm"] == [
        str(tmp_path / "run" / "arms" / "q27gw-main-bf16-decode-default-stress")]
    assert "--windows" not in by_name["arm-q27gw-main-bf16-decode-default-stress"]["argv"]
    # a check whose result the earlier run holds is read from there, and says so
    assert "e2e-q27gw-identity" not in by_name
    names = {c["name"]: c["result"] for c in plan_json["checks"]}
    assert names["F q27gw identity (from earlier)"] == str(earlier / "checks" / "e2e-q27gw-identity.json")
    # release spot checks: one lossless reference per attention family
    assert [n for n in by_name if n.startswith("verify-")] == ["verify-q27gw-main-lossless"]


def test_a_declared_design_discloses_its_shipped_routes_against_lossless(tmp_path: Path) -> None:
    steps, plan_json = plan(tmp_path, ("q27", "q27gw"))
    by_name = {s["name"]: s["argv"] for s in steps}
    disclosures = {d["label"]: d for d in plan_json["disclosures"]}
    item = disclosures["q27 default route against lossless (engine and design error)"]
    assert "kv" not in item  # reported, no ceiling
    argv = by_name["arm-q27-main-bf16-prefill-default-vs-lossless"]
    assert argv[argv.index("--reference") + 1].endswith("main-lossless.ref")
    assert not any(label.startswith("q27gw") and "against lossless" in label for label in disclosures)
