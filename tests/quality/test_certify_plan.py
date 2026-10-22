"""The IQS-1 run plan (tools/quality/certify.py): every reference a step reads is made by an earlier step, every
decision has its arm and noise twin planned, and the declared designs use their design references."""

from __future__ import annotations

import json
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
        assert d["arm"] + ".json" in made and d["twin"] + ".json" in made
    for d in decisions["disclosures"]:
        assert d["arm"] + ".json" in made


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
