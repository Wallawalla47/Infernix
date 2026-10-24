"""The IQS-2 run plan (tools/quality/certify.py): every reference a step reads is made by an earlier step, every
decision has its arm and noise twin planned, and the declared designs use their design references."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.quality import certify


def plan(tmp_path: Path, keys: tuple[str, ...], mode: str = "full") -> tuple[list[dict], dict]:
    local = {k: {"path": f"/models/{k}.infernix", "ngram_volume": f"/models/{k}.ngram",
                 "draft_model": "/models/dflash2.infernix"} for k in keys}
    (tmp_path / "local.json").write_text(json.dumps(local), encoding="utf-8")
    run = tmp_path / "run"
    assert certify.main(["plan", "--artifacts", str(tmp_path / "local.json"), "--build", str(tmp_path / "build"),
                         "--store", str(tmp_path / "store"), "--out", str(run), "--mode", mode]) == 0
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
    # References, noise twins and reproducible arms are stored per 32-window layer, so a larger corpus makes only
    # its new layers; the default-route arms run on every window.
    steps, plan_json = plan(tmp_path, ("q27gw",))
    by_name = {s["name"]: s for s in steps}
    count = certify.main_windows()
    layers = count // certify.LAYER
    assert layers >= 2
    for stem in ("ref-q27gw-lossless", "twin-q27gw", "twin-q27gw-int8", "ref-q27gw-design-nvfp4",
                 "arm-q27gw-main-bf16-prefill-reproducible", "arm-q27gw-main-bf16-decode-reproducible"):
        names = [stem] + [f"{stem}-L{k}" for k in range(1, layers)]
        covered = [windows_of(by_name[n]["argv"]) for n in names]
        assert set().union(*covered) == set(range(count)) and sum(map(len, covered)) == count, stem
    store = str(tmp_path / "store")
    reproducible = by_name["arm-q27gw-main-bf16-prefill-reproducible"]["outputs"][0]
    default = by_name["arm-q27gw-main-bf16-prefill-default"]
    assert reproducible.startswith(store) and not default["outputs"][0].startswith(store)
    assert "--windows" not in default["argv"]
    # every part of the reference is passed to an arm, layer 0 under its old name
    references = [a for i, a in enumerate(default["argv"][1:]) if default["argv"][i] == "--reference"]
    assert len(references) == layers and references[0].endswith("main-lossless.ref")


def test_a_kv_format_is_judged_against_the_twin_of_its_own_design(tmp_path: Path) -> None:
    _, plan_json = plan(tmp_path, ("q27gw",))
    by_label = {d["label"]: d for d in plan_json["decisions"]}
    assert all("main-twin-nvfp4" in t for t in by_label["q27gw main nvfp4 prefill-default"]["twin"])
    assert all("main-twin.w" in t for t in by_label["q27gw main bf16 prefill-default"]["twin"])
    assert all("main-twin.w" in t for t in by_label["q27gw long bf16 prefill-default"]["twin"])


def test_every_default_route_arm_is_gated_against_its_reproducible_arm(tmp_path: Path) -> None:
    steps, plan_json = plan(tmp_path, ("q27", "q27gw"))
    made = {out for s in steps for out in s["outputs"]}
    gates = {g["label"]: g for g in plan_json["gates"]}
    defaults = [d["label"] for d in plan_json["decisions"] if " main " in d["label"] and "default" in d["label"]]
    assert defaults and set(defaults) == set(gates)
    assert gates["q27 main bf16 decode-default-tree"]["baseline"][0].endswith("main-bf16-decode-reproducible.w0-31")
    assert gates["q27gw main nvfp4 prefill-default"]["baseline"][0].endswith("main-nvfp4-prefill-reproducible.w0-31")
    for g in gates.values():
        assert all(p + ".json" in made for p in g["arm"] + g["baseline"] + g["twin"]), g["label"]
