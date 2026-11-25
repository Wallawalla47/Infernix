"""After the fit: pick (list), keep (pin), test out of sample (grade), talk to (chat) and share (export) candidates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from . import measure as M
from . import sidecar as SC
from .engine import MODEL_ID, Engine, is_inkling
from .fit import front, open_study
from .prompts import read_prompts
from .run import REPO_ROOT, artifact_id, sha256_file, write_json


def list_front(run, study_name: str, n: int = 15) -> list[dict]:
    return front(open_study(run, study_name))[:n]


def pin(run, study_name: str, trial: int, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise SystemExit(f"candidate name {name!r}: use letters, digits, '.', '_' and '-'")
    tdir = run.dir / "fit" / study_name / f"t{trial}"
    if not (tdir / "trial.json").is_file():
        raise SystemExit(f"{tdir} has no trial.json (did trial {trial} complete?)")
    dest = run.dir / "candidates" / name
    if dest.exists():
        raise SystemExit(f"{dest} exists; pick another name")
    SC.copy(tdir / "sidecar.json", dest)
    for extra in ("trial.json", "answers.jsonl"):
        if (tdir / extra).is_file():
            shutil.copy2(tdir / extra, dest / extra)
    write_json(dest / "pinned.json", {"study": study_name, "trial": trial, "pinned": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      "d_sha256": hashlib.sha256((dest / "d.f32.bin").read_bytes()).hexdigest()})
    return dest


def candidate_sidecar(run, name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".json" and p.is_file():
        return p.resolve()
    pinned = run.dir / "candidates" / name_or_path / "sidecar.json"
    if not pinned.is_file():
        raise SystemExit(f"no candidate {name_or_path!r} (pin one first, or pass a sidecar.json path)")
    return pinned


def grade(run, harmful_file: Path, benign_file: Path | None, candidates: list[str], field: str | None = None,
          markers: tuple[str, ...] = M.MARKERS, log=print) -> Path:
    """Once, on lists the fit never saw: refusals on the harmful list, over-refusals on the benign list, and KL on the
    benign list against the plain model. Two engines per arm.

    KL is also split by what the stock model did with each benign prompt. Where it answered, KL is collateral drift.
    Where it refused, a moved first token is the cut working ("I can't" became an answer), so a high KL there is
    expected and a mean over the whole list overstates the damage."""
    harm, harm_cat = read_prompts(harmful_file, field)
    benign, _ = read_prompts(benign_file, field) if benign_file else ([], [])
    arms = [("stock", None)] + [(Path(c).parent.name if c.endswith(".json") else c, candidate_sidecar(run, c))
                                for c in candidates]
    out = run.sub("grade", time.strftime("%Y%m%dT%H%M%S"))
    rows, anchor, stock_refused = [], None, set()
    for name, sc in arms:
        t0, d = time.time(), out / name
        if benign and sc is None:
            with Engine(run, "score", log_path=d / "score.log") as e:
                anchor = M.first_token_distributions(e, benign)
        with Engine(run, "generate", sidecar=sc, log_path=d / "generate.log") as e:
            h = M.grade_refusals(e, harm, record=d / "harmful.answers.jsonl", markers=markers)
            b = M.grade_refusals(e, benign, record=d / "benign.answers.jsonl", markers=markers) if benign else None
        row = {"arm": name, "sidecar": str(sc) if sc else None, "harmful": h, "benign": b,
               "harmful_by_category": _by_category(d / "harmful.answers.jsonl", harm_cat)}
        if benign and sc is None:
            stock_refused = _refused(d / "benign.answers.jsonl")
            row.update(kl=0.0, kl_stock_answered=0.0, kl_stock_refused=0.0)
        elif benign:
            assert anchor is not None   # the stock arm runs first and sets it
            with Engine(run, "score", sidecar=sc, log_path=d / "score.log") as e:
                kls = M.kl(anchor, M.first_token_distributions(e, benign))
            answered = [k for i, k in enumerate(kls) if i not in stock_refused]
            refused = [k for i, k in enumerate(kls) if i in stock_refused]
            row.update(kl=M.mean(kls), kl_stock_answered=M.mean(answered) if answered else None,
                       kl_stock_refused=M.mean(refused) if refused else None,
                       kl_per_prompt=[round(k, 5) for k in kls])
        row["seconds"] = round(time.time() - t0)
        rows.append(row)
        log(f"  {name:<16} harmful refused {_frac(h)}" + (f"   benign refused {_frac(b)}" if b else "")
            + (f"   KL {row['kl']:.4f} (where stock answered: {_num(row['kl_stock_answered'])})" if "kl" in row else "")
            + f"   ({row['seconds']} s)")
        write_json(out / "summary.json", {"harmful": str(harmful_file), "benign": str(benign_file or ""),
                                          "stock_refused_benign": sorted(stock_refused), "arms": rows})
    title = f"# grade: {Path(harmful_file).name}" + (f" + {Path(benign_file).name}" if benign_file else "")
    n_ref = len(stock_refused)
    lines = [title, "", "| arm | harmful refused | benign refused | KL on benign, all | "
             f"KL where stock answered ({len(benign) - n_ref}) | KL where stock refused ({n_ref}) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {_frac(r['harmful'])} | {_frac(r['benign']) if r['benign'] else '-'} | "
                     f"{_num(r.get('kl'))} | {_num(r.get('kl_stock_answered'))} | {_num(r.get('kl_stock_refused'))} |")
    cats = sorted({c for r in rows for c in r["harmful_by_category"]})
    if cats:
        lines += ["", "Harmful refusals by category (refused/graded):", "",
                  "| category | " + " | ".join(r["arm"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
        for c in cats:
            cells = [r["harmful_by_category"].get(c) for r in rows]
            lines.append(f"| {c} | " + " | ".join(f"{x[0]}/{x[1]}" if x else "-" for x in cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _frac(counts: dict) -> str:
    return f"{counts['refusals']}/{counts['n'] - counts['errors'] - counts['truncated']}"


def _num(x) -> str:
    return "-" if x is None else f"{x:.4f}"


def _refused(record: Path) -> set[int]:
    return {json.loads(line)["i"] for line in record.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["verdict"].startswith("refusal")}


def _by_category(record: Path, cats: list) -> dict:
    if not record.is_file() or not any(cats):
        return {}
    out: dict = {}
    for line in record.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        c = cats[row["i"]] or "(none)"
        refused, graded = out.get(c, [0, 0])
        if row["verdict"] not in ("error", "error_thinking_on", "truncated"):
            out[c] = [refused + (row["verdict"].startswith("refusal")), graded + 1]
    return out


def chat(run, candidate: str | None, context: int | None, thinking: bool = True) -> None:
    """Talk to a candidate (or the stock model) on production's carried settings. Blank lines are ignored;
    /reset clears the conversation, /exit leaves."""
    sc = candidate_sidecar(run, candidate) if candidate else None
    with Engine(run, "generate", sidecar=sc, context=context, log_path=run.sub("logs") / "chat.log") as e:
        print(f"talking to {candidate or 'the stock model'} ({e.context} tokens of context). /reset clears, /exit leaves.\n")
        msgs: list[dict] = []
        while True:
            try:
                user = input("you: ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not user:
                continue
            if user == "/exit":
                return
            if user == "/reset":
                msgs = []
                print("(cleared)\n")
                continue
            msgs.append({"role": "user", "content": user})
            reply = e.post("/v1/chat/completions", {"model": MODEL_ID, "messages": msgs, "enable_thinking": thinking,
                                                    "max_tokens": max(256, e.context // 2)})
            content = reply["choices"][0]["message"].get("content") or ""
            msgs.append({"role": "assistant", "content": content})
            print(f"\nmodel: {content}\n")


def export(run, candidate: str, name: str, source: str | None = None, out: Path | None = None) -> Path:
    """A shareable copy in out/name (default: the run's export/ folder): no local paths, plain-words provenance, and
    what it was fitted under."""
    sc = candidate_sidecar(run, candidate)
    cfg = SC.read(sc)
    sets = json.loads((run.dir / "prompts" / "sets.json").read_text(encoding="utf-8"))
    trial = json.loads((sc.parent / "trial.json").read_text(encoding="utf-8")) if (sc.parent / "trial.json").is_file() else {}
    base = SC.read(run.dir / "direction" / "sidecar.json")
    prod = run.production
    template = prod.value("--chat-template")
    template_text = Path(template).read_text(encoding="utf-8") if template else None
    numerics = [(f, v) for f, v in prod.options if f in ("--kv-dtype", "--kv-window-dtype", "--a8-activations",
                                                          "--a16-activations",
                                                          "--expert-a4-activations", "--reproducible", "--lossless",
                                                          "--fast-path", "--no-fast-path", "--rope-yarn-factor",
                                                          "--logits-dtype", "--expert-tail-skip",
                                                          "--use-original-int8-prefill-kernel",
                                                          "--use-original-nvfp4-prefill-kernel")]
    src = source or (f"{sets['source']['harmful']['name']} against {sets['source']['harmless']['name']}; "
                     f"{base['generation']['notes'].removeprefix('direction: ')}")
    pinned = (json.loads((sc.parent / "pinned.json").read_text(encoding="utf-8"))
              if (sc.parent / "pinned.json").is_file() else {})
    notes = ""
    if trial:
        which = f"trial {pinned['trial']} of study {pinned['study']}: " if pinned else ""
        notes = (f"alphas: per-lane bands fitted with tools/ablate (heretic 1.4's band and direction scope; two-objective "
                 f"Optuna TPE over refusals and first-token KL); this candidate, {which}{trial['refusals']}/"
                 f"{trial['n_harmful']} held-out harmful prompts refused (stock: {trial['control_refusals']}), "
                 f"first-token KL {trial['kl']:.3f} on the held-out harmless prompts; {trial['what']}")
    fitted_under = {
        "artifact": prod.artifact.name, "kv_dtype": prod.value("--kv-dtype") or "bf16 (default)",
        "chat_template": ({"file": Path(template).name, "sha256": hashlib.sha256(template_text.encode()).hexdigest()}
                          if template_text else "the artifact's own"),
        "numerics_options": [f if v is None else f"{f} {v}" for f, v in numerics],
        "thinking": M.thinking_protocol(is_inkling(prod.artifact)), "tool": "tools/ablate " + tool_version(),
        "engine": f"{run.serve_binary.name} sha256:{sha256_file(run.serve_binary)[:16]}"}
    lora = prod.value("--lora")
    if lora:
        # The adapter the model was served with: its folder (or file) name and its weights' sha256.
        weights = Path(lora) / "adapter_model.safetensors" if Path(lora).is_dir() else Path(lora)
        fitted_under["lora"] = {"adapter": Path(lora).name, "sha256": sha256_file(weights)}
    cfg["generation"] = {"source": src, "notes": notes, "fitted_under": fitted_under}
    # The engine warns when the sidecar starts on another artifact.
    cfg["artifact_id"] = artifact_id(prod.artifact)
    dest = (Path(out) if out else run.dir / "export") / name
    if dest.exists():
        raise SystemExit(f"{dest} exists; pick another name")
    dest.mkdir(parents=True)
    shutil.copy2(sc.parent / cfg["direction_file"], dest / cfg["direction_file"])
    text = json.dumps(cfg, indent=1) + "\n"
    leaks = [p for p in (str(run.dir), str(Path.home()), str(prod.artifact.parent)) if p and p in text]
    if leaks:
        raise SystemExit(f"export would carry local paths {leaks}; pass --source to describe the prompt sets")
    (dest / "sidecar.json").write_text(text, encoding="utf-8")
    return dest


def same_cut(a: Path, b: Path) -> bool:
    """Whether two sidecar.json files cut the same: the same alpha tables and the same direction bytes (their
    provenance text may differ)."""
    ca, cb = SC.read(a), SC.read(b)
    tables = ("alpha_attn_per_layer", "alpha_mlp_per_layer", "alpha_per_layer", "default_alpha")
    if any(ca.get(k) != cb.get(k) for k in tables):
        return False
    return (a.parent / ca["direction_file"]).read_bytes() == (b.parent / cb["direction_file"]).read_bytes()


def tool_version() -> str:
    """The tool's version: what the launcher that installed it says (INFERNIX_ABLATE_TOOL_VERSION), else its checkout's
    commit ('-dirty' when tools/ablate has uncommitted changes), else '(unknown)'."""
    given = os.environ.get("INFERNIX_ABLATE_TOOL_VERSION", "").strip()
    if given:
        return given
    if not (REPO_ROOT / ".git").exists():
        return "(unknown)"

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    try:
        head = git("rev-parse", "--short=8", "HEAD")
        dirty = git("status", "--porcelain", "--", "tools/ablate")
    except (OSError, subprocess.TimeoutExpired):
        return "(unknown)"
    return (head + ("-dirty" if dirty else "")) if head else "(unknown)"
