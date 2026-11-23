"""tools/ablate (the refusal-ablation fitting tool) on CPU: its rules, and every command end to end against a fake
engine (tests/ablate_fakes.py), so no GPU or model is needed and it runs on Windows too.

The end-to-end test checks the math, not only the plumbing: the fake engine plants a known direction in the harmful
prompts' hidden states, and the direction step must recover it, layer by layer.

Run with pytest, or directly: python tests/test_ablate_tools.py
Needs tools/ablate/requirements.txt (numpy, optuna); skipped without them.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

for _module in ("numpy", "optuna"):
    pytest.importorskip(_module)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FAKES = HERE / "ablate_fakes.py"
sys.path.insert(0, str(ROOT / "tools" / "ablate"))
sys.path.insert(0, str(HERE))

import ablate  # noqa: E402
import ablate_fakes as F  # noqa: E402
from ablatelib import engine as EN  # noqa: E402
from ablatelib import evaluate as EV  # noqa: E402
from ablatelib import measure as M  # noqa: E402
from ablatelib import platform_util as PU  # noqa: E402
from ablatelib import production as P  # noqa: E402
from ablatelib import prompts as PR  # noqa: E402
from ablatelib import serve_flags as SF  # noqa: E402
from ablatelib import sidecar as SC  # noqa: E402

LAYERS, HIDDEN = 6, 64
ARTIFACT_ID = "00112233445566778899aabbccddeeff"


def cli(*args) -> int:
    old = sys.argv
    sys.argv = ["ablate.py", *map(str, args)]
    try:
        return ablate.main()
    finally:
        sys.argv = old


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A fake setup: a binary, a model file, production's command line and its expert-state file, prompt lists, and
    the seam that starts the fake engine in place of infernix-serve."""
    monkeypatch.setenv(F.SHAPE_ENV, f"{LAYERS},{HIDDEN}")
    real_argv = EN.Engine.argv
    monkeypatch.setattr(EN.Engine, "argv",
                        lambda self, port=None: [sys.executable, str(FAKES), "serve", *real_argv(self, port)[1:]])
    monkeypatch.setattr(PU, "card_processes", lambda device=0: [])
    monkeypatch.setattr(PU, "free_disk_bytes", lambda path: 1 << 40)

    bins = tmp_path / "bin"
    bins.mkdir()
    serve_bin = bins / "infernix-serve"
    serve_bin.write_text("fake\n")
    model = tmp_path / "models" / "Fake-Model-00001-of-00001.infernix"
    model.parent.mkdir()
    sys.path.insert(0, str(ROOT / "tools"))
    from artifact.framing import HEADER, MAGIC   # noqa: E402  (a v3 entry header: the tool reads the artifact's id)
    model.write_bytes(HEADER.pack(MAGIC, 16, bytes.fromhex(ARTIFACT_ID)) + b"\0" * 4096)
    prod_state = model.parent / "production.expert-state"
    prod_state.write_text("production's own state\n", encoding="utf-8")
    prod = tmp_path / "production.txt"
    prod.write_text(f'"{serve_bin}" "{model}" --kv-dtype bf16 --max-context 262144 --max-concurrency 2 \\\n'
                    f'  --expert-state "{prod_state}" --host 0.0.0.0 --port 8080   # production\n', encoding="utf-8")
    harmful = write_lines(tmp_path / "harmful.txt", [f"HARM request {i}: describe forbidden thing {i}" for i in range(30)])
    harmless = write_lines(tmp_path / "harmless.txt", [f"benign request {i}: describe allowed thing {i}" for i in range(30)])
    return SimpleNamespace(tmp=tmp_path, prod=prod, prod_state=prod_state, model=model,
                           harmful=harmful, harmless=harmless, run=tmp_path / "run")


def start_run(rig, held_out: int = 6) -> Path:
    assert cli("init", rig.run, "--production", rig.prod) == 0
    assert cli("prompts", rig.run, "--harmful", rig.harmful, "--harmless", rig.harmless, "--held-out", held_out,
               "--paired") == 0
    return rig.run


def test_every_command_end_to_end_on_the_fake_engine(rig, capsys: pytest.CaptureFixture) -> None:
    run = start_run(rig)
    assert cli("doctor", run) == 0
    assert "ready" in capsys.readouterr().out

    # capture: one recording per direction prompt through one scoring engine, of the exact size
    assert cli("capture", run) == 0
    assert "48 prompt(s) to record" in capsys.readouterr().out
    index = json.loads((run / "capture" / "index.json").read_text(encoding="utf-8"))
    assert index["shape"] == {"layers": LAYERS, "hidden": HIDDEN} and len(index["prompts"]) == 48
    assert index["identity"]["artifact_id"] == ARTIFACT_ID
    assert all(entry["prompt_tokens"] for entry in index["prompts"].values())
    # incremental: a lost recording is the only one recorded again
    first = index["prompts"]["harmful:0"]["fingerprint"]
    (run / "capture" / f"{first}.f32").unlink()
    assert cli("capture", run) == 0
    assert "1 prompt(s) to record" in capsys.readouterr().out
    assert cli("capture", run) == 0
    assert "every prompt is already recorded" in capsys.readouterr().out

    # direction: the planted direction comes back at the layer it was planted in (row L: what layer L hands on)
    assert cli("direction", run, "--no-orthogonalize") == 0
    rows = np.fromfile(run / "direction" / "d.f32.bin", dtype=np.float32).reshape(LAYERS, HIDDEN)
    for layer in range(LAYERS):
        assert float(rows[layer] @ F.planted(layer, HIDDEN)) > 0.995
    # heretic's default, orthogonalized against the harmless mean: unit rows with no component along it (the fakes'
    # shared mean), the planted direction kept
    assert cli("direction", run) == 0
    ortho = np.fromfile(run / "direction" / "d.f32.bin", dtype=np.float32).reshape(LAYERS, HIDDEN)
    assert np.allclose(np.linalg.norm(ortho, axis=1), 1.0, atol=1e-5)
    for layer in range(LAYERS):
        assert abs(float(ortho[layer] @ F.common(layer, HIDDEN))) < 0.01
        assert float(ortho[layer] @ F.planted(layer, HIDDEN)) > 0.99
    capsys.readouterr()

    # fit: a control and three trials; a second invocation continues the same study
    assert cli("fit", run, "--study", "s1", "--trials", 3, "--startup", 3) == 0
    assert "CONTROL (no cut): refusals 6/6" in capsys.readouterr().out
    assert cli("fit", run, "--study", "s1", "--trials", 2, "--startup", 3) == 0
    assert "3 trial(s) already in the ledger" in capsys.readouterr().out
    controls = sorted((run / "fit" / "s1").glob("run-*/control.answers.jsonl"))
    assert len(controls) == 2 and controls[0].read_bytes() == controls[1].read_bytes()
    with pytest.raises(SystemExit, match="Use a new study name"):
        cli("fit", run, "--study", "s1", "--trials", 1, "--kl-target", 0.2)
    trials = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "fit" / "s1").glob("t*/trial.json"))]
    assert len(trials) == 5 and all(t["kl"] > 0 and t["refusals"] <= 6 for t in trials)
    # the engines kept their expert-cache state in the run; production's file is untouched
    assert rig.prod_state.read_text(encoding="utf-8") == "production's own state\n"
    assert (run / "expert.state").read_text(encoding="utf-8") == "fake expert-cache state\n"

    # list, pin, grade (a pinned name and a sidecar path), export, eval
    assert cli("list", run, "--study", "s1") == 0
    assert cli("pin", run, "--study", "s1", "--trial", 0, "--name", "c0") == 0
    assert {"sidecar.json", "d.f32.bin", "trial.json", "pinned.json"} <= {p.name for p in (run / "candidates" / "c0").iterdir()}
    harm2 = write_lines(rig.tmp / "test-harmful.txt", [f"HARM unseen {i}" for i in range(10)])
    benign2 = write_lines(rig.tmp / "test-benign.txt", [f"benign unseen {i}" for i in range(10)]
                          + [f"{F.EDGE} benign unseen {i}" for i in range(3)])     # stock over-refuses these
    assert cli("grade", run, "--harmful", harm2, "--benign", benign2, "c0", run / "fit" / "s1" / "t1" / "sidecar.json") == 0
    graded = json.loads(next((run / "grade").glob("*/summary.json")).read_text(encoding="utf-8"))
    assert [a["arm"] for a in graded["arms"]] == ["stock", "c0", "t1"]
    assert graded["arms"][0]["harmful"]["refusals"] == 10 and graded["arms"][0]["kl"] == 0.0
    assert graded["stock_refused_benign"] == [10, 11, 12]
    c0 = graded["arms"][1]
    # kl_per_prompt is stored rounded to 5 decimals, the split means from the unrounded values: within 5e-6.
    assert math.isclose(c0["kl_stock_refused"], sum(c0["kl_per_prompt"][10:]) / 3, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(c0["kl_stock_answered"], sum(c0["kl_per_prompt"][:10]) / 10, rel_tol=0, abs_tol=1e-5)
    assert cli("export", run, "c0", "--name", "share") == 0
    text = (run / "export" / "share" / "sidecar.json").read_text(encoding="utf-8")
    assert str(rig.tmp) not in text and str(Path.home()) not in text
    shared = json.loads(text)
    assert shared["generation"]["fitted_under"]["artifact"] == rig.model.name
    assert shared["generation"]["fitted_under"]["kv_dtype"] == "bf16"
    assert shared["artifact_id"] == ARTIFACT_ID and "lora" not in shared["generation"]["fitted_under"]
    assert cli("eval", run, "c0", "--tasks", "kl") == 0
    result = json.loads(next((run / "eval").glob("c0-*/results.json")).read_text(encoding="utf-8"))
    assert result["tasks"]["kl"]["n"] == 6
    assert math.isclose(result["tasks"]["kl"]["mean"], trials[0]["kl"], rel_tol=1e-9)   # the fit's own KL, re-measured


def test_capture_records_again_when_the_setup_changes(rig, capsys: pytest.CaptureFixture) -> None:
    # A recording depends on production's carried options: another KV type is another model's hidden states.
    run = start_run(rig)
    assert cli("capture", run) == 0
    capsys.readouterr()
    rig.prod.write_text(rig.prod.read_text(encoding="utf-8").replace("--kv-dtype bf16", "--kv-dtype int8"),
                        encoding="utf-8")
    assert cli("init", run, "--production", rig.prod, "--force") == 0
    assert cli("capture", run) == 0
    assert "48 prompt(s) to record" in capsys.readouterr().out


def test_every_serve_option_has_a_rule() -> None:
    arity = SF.arities(ROOT)
    assert [f for f in arity if f not in SF.RULES] == [], "classify new infernix-serve options in serve_flags.py"
    assert [f for f in SF.RULES if f not in arity and f != "--config"] == [], "serve_flags.py names a removed option"


def test_production_setups_in_all_three_forms(tmp_path: Path) -> None:
    arity = SF.arities(ROOT)
    (tmp_path / "t.jinja").write_text("{{ messages }}", encoding="utf-8")
    text = tmp_path / "p.txt"
    text.write_text("infernix-serve m.infernix \\\n  --kv-dtype int8   # KV\n  --chat-template t.jinja --port 9\n",
                    encoding="utf-8")
    prod = P.from_text_file(text, arity, tmp_path)
    assert prod.artifact == (tmp_path / "m.infernix").resolve()
    assert prod.value("--chat-template") == str((tmp_path / "t.jinja").resolve())
    assert [f for f, _ in prod.by_kind(SF.CARRY)] == ["--kv-dtype", "--chat-template"]
    # A command line saved from cmd.exe in Notepad: a byte-order mark, ^ continuations, whole-line REM (bare, too) and
    # :: comments. A REM line inside a ^ continuation is dropped as well, which cmd itself would not do.
    text.write_text("\ufeffREM\r\nREM production\r\n:: the arm\r\ninfernix-serve.exe m.infernix ^\r\n"
                    "  rem --kv-dtype bf16 (a commented-out option)\r\n  --kv-dtype int8 ^\r\n  --port 9\r\n",
                    encoding="utf-8", newline="")         # newline="": these bytes on every platform
    prod = P.from_text_file(text, arity, tmp_path)
    assert prod.artifact == (tmp_path / "m.infernix").resolve() and prod.value("--kv-dtype") == "int8"
    toml = tmp_path / "serve.toml"
    toml.write_text('\ufeffmodel = "m.infernix"\nkv-dtype = "bf16"\n[serve]\nkv-dtype = "int8"\n', encoding="utf-8")
    prod = P.from_config_file(toml, arity, tmp_path, extra=["--port", "9"])
    assert prod.value("--kv-dtype") == "int8" and prod.value("--port") == "9"
    with pytest.raises(ValueError, match="not an infernix-serve option"):
        P.parse_tokens(["m.infernix", "--no-such-option", "1"], arity, tmp_path, "test")
    lora = P.parse_tokens(["m.infernix", "--lora", "a.lora"], arity, tmp_path, "test")   # carried: the served model
    assert not lora.refused() and [f for f, _ in lora.by_kind(SF.CARRY)] == ["--lora"]
    lossless = P.parse_tokens(["m.infernix", "--lossless"], arity, tmp_path, "test")   # the engine refuses --ablate
    assert lossless.refused() and lossless.refused()[0].startswith("--lossless")


def test_refusal_grading_rules(tmp_path: Path) -> None:
    assert M.refusal_marker("I\u2019m sorry, no.") == "i'm sorry"
    assert M.refusal_marker("x" * 200 + " I cannot") is None          # only the first 160 characters count
    extra = write_lines(tmp_path / "m.txt", ["\ufeffNope, Not Doing It", "# comment"])      # with a byte-order mark
    assert "nope, not doing it" in M.load_markers(extra)

    class Stub:
        def __init__(self, replies):
            self.replies = iter(replies)

        def chat(self, prompt, max_tokens, enable_thinking):
            r = next(self.replies)
            if isinstance(r, Exception):
                raise r
            return r

    def reply(content, finish="stop", reasoning=""):
        return {"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": finish}

    counts = M.grade_refusals(Stub([reply("I cannot."), reply(""), reply("", "length"), reply("ok", reasoning="hm"),
                                    reply("Sure."), EN.EngineError("boom")]), ["p"] * 6)
    assert counts == {"refusals": 2, "errors": 2, "truncated": 1, "n": 6}


def test_kl_measure() -> None:
    ref = {"a": math.log(0.9), "b": math.log(0.1)}
    assert M.kl([ref], [ref]) == [0.0]
    dropped = M.kl([ref], [{"a": math.log(0.95), "c": math.log(0.01)}])[0]   # b floored at the lowest, log 0.01
    assert math.isclose(dropped, 0.9 * math.log(0.9 / 0.95) + 0.1 * math.log(0.1 / 0.01))
    with pytest.raises(EN.EngineError):
        M.kl([ref], [{}])
    with pytest.raises(EN.EngineError):
        M.kl([ref, ref], [ref])


def test_prompt_files_checks_and_split(tmp_path: Path) -> None:
    # Each with the byte-order mark Notepad and PowerShell write; the .txt with CRLF line ends too.
    (tmp_path / "a.jsonl").write_text('\ufeff{"goal": "g1", "category": "c"}\n{"goal": "g2", "category": "d"}\n',
                                      encoding="utf-8")
    assert PR.read_prompts(tmp_path / "a.jsonl") == (["g1", "g2"], ["c", "d"])
    (tmp_path / "b.csv").write_text("\ufeffBehavior,SemanticCategory\nx,c1\ny,c2\n", encoding="utf-8")
    assert PR.read_prompts(tmp_path / "b.csv") == (["x", "y"], ["c1", "c2"])
    (tmp_path / "c.txt").write_text("\ufeffp1\r\np2\r\n", encoding="utf-8", newline="")
    assert PR.read_prompts(tmp_path / "c.txt") == (["p1", "p2"], [None, None])
    harm, good = [f"h{i}" for i in range(30)], [f"g{i}" for i in range(30)]
    assert PR.problems(harm, good, 5, paired=True) == []
    assert any("at least 1" in p for p in PR.problems(harm, good, 0, paired=False))
    assert any("duplicate" in p for p in PR.problems(harm + ["h0"], good, 5, paired=False))
    assert any("both lists" in p for p in PR.problems(harm, good + ["h1"], 5, paired=False))
    assert any("leave" in p for p in PR.problems(harm[:22], good, 5, paired=False))
    sets = PR.split(harm, good, 5)
    assert len(sets["direction"]["harmful"]) == 25 and sets["held_out"]["harmful"] == harm[25:]
    assert PR.eval_id(harm[25:], good[25:]) == PR.eval_id(list(harm[25:]), list(good[25:]))


def test_band_and_sidecar_rules(tmp_path: Path) -> None:
    band = SC.Band(max_weight=1.2, max_weight_position=4.0, min_weight=0.4, min_weight_distance=2.0)
    assert [round(band.alpha_at(layer), 6) for layer in range(7)] == [0.0, 0.0, 0.4, 0.8, 1.2, 0.8, 0.4]
    rows = np.stack([F.planted(layer, HIDDEN) for layer in range(LAYERS)])
    base = SC.write_base(tmp_path / "base", rows, source="test", notes="test")
    one = SC.global_direction(rows, 2.25)
    assert math.isclose(float(np.linalg.norm(one)), 1.0, rel_tol=1e-5)
    trial = SC.write_trial(base, tmp_path / "t", SC.lane_tables({"attn": band, "mlp": band}, LAYERS), one, "n")
    cfg = SC.read(trial)
    assert cfg["direction_layout"] == "shared" and set(cfg["alpha_attn_per_layer"]) == {"2", "3", "4", "5"}
    with pytest.raises(ValueError, match="alpha outside"):
        SC.write_trial(base, tmp_path / "bad", {"alpha_attn_per_layer": {"0": 4.5}}, None, "n")


def test_cli_takes_names_before_between_and_after_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    for name in ("cmd_grade", "cmd_chat"):
        monkeypatch.setattr(ablate, name, lambda a: seen.append(a) or 0)
    cli("grade", "R", "--harmful", "H", "mine", "theirs")
    cli("grade", "R", "mine", "--harmful", "H", "theirs")
    cli("chat", "R", "--context", "8192", "mine")
    assert [s.candidates for s in seen[:2]] == [["mine", "theirs"]] * 2
    assert seen[2].candidate == "mine" and seen[2].context == 8192


def test_eval_comparisons() -> None:
    a = {"p1": True, "p2": True, "p3": False, "p4": False}
    b = {"p1": True, "p2": False, "p3": False, "p4": True}
    assert EV.mcnemar(a, b) == {"stock_only": 1, "candidate_only": 1, "p_two_sided": 1.0}
    stock = {"metrics": {"t/acc/none": {"value": 1.0, "stderr": 0.0}}, "truncated": 0}
    assert EV.compare("gsm8k", stock, stock)["metrics"]["t/acc/none"]["reading"] == "within the noise band"
    assert EV.compare("gsm8k", stock, stock, limited=True)["metrics"]["t/acc/none"]["reading"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
