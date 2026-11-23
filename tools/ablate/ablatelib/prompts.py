"""Prompt sets: read the user's files, check them, split them the way the fit uses them.

A run has one harmful list and one harmless list. The first part of each builds the direction; the held-out tail
grades every trial (refusals on the harmful half, KL on the harmless half), so no trial is graded on a prompt the
direction was built from. A test pair (for example JBB-Behaviors' harmful and benign halves) is scored once, at the
end, by `grade`: that is the out-of-sample number. The tool ships no prompts: `datasets` downloads and converts public
sets, and docs/ablate-fitting.md says where to get others and how to build matched harmless counterparts.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

FIELDS = ("prompt", "text", "goal", "Goal", "Behavior", "behavior", "instruction", "question")


def read_prompts(path: Path, field: str | None = None) -> tuple[list[str], list[str | None]]:
    """(prompts, categories) from .txt (one per line), .jsonl or .csv. `field` names the prompt column; without it
    the first of FIELDS present is used. A `category` (or `Category`, `SemanticCategory`) column is kept if present.
    A leading UTF-8 byte-order mark (Notepad, PowerShell) is dropped."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".text", ""):
        lines = [line.rstrip("\r\n") for line in path.read_text(encoding="utf-8-sig").splitlines()]
        return [line for line in lines if line.strip()], [None] * sum(1 for line in lines if line.strip())
    rows: list[dict] = []
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = data if isinstance(data, list) else data.get("prompts") or []
        rows = [{"prompt": r} if isinstance(r, str) else r for r in rows]
    else:
        raise SystemExit(f"{path}: unknown prompt file type {suffix!r} (use .txt, .jsonl, .json or .csv)")
    if not rows:
        raise SystemExit(f"{path}: no prompts")
    name = field or next((f for f in FIELDS if f in rows[0]), None)
    if name is None or name not in rows[0]:
        raise SystemExit(f"{path}: no prompt column; pass --field (columns: {sorted(rows[0])})")
    cat = next((c for c in ("category", "Category", "SemanticCategory") if c in rows[0]), None)
    prompts = [str(r[name]).strip() for r in rows]
    if any(not p for p in prompts):
        raise SystemExit(f"{path}: {sum(1 for p in prompts if not p)} empty prompt(s) in column {name!r}")
    return prompts, [str(r[cat]) if cat else None for r in rows]


def problems(harmful: list[str], harmless: list[str], held_out: int, paired: bool) -> list[str]:
    out = []
    if held_out < 1:
        out.append(f"--held-out {held_out}: the fit grades every trial on held-out prompts, so hold out at least 1")
    for label, items in (("harmful", harmful), ("harmless", harmless)):
        dup = len(items) - len(set(items))
        if dup:
            out.append(f"{label}: {dup} duplicate prompt(s)")
        if any("\n" in p for p in items):
            out.append(f"{label}: prompts may not contain line breaks")
    overlap = set(harmful) & set(harmless)
    if overlap:
        out.append(f"{len(overlap)} prompt(s) appear in both lists")
    if paired and len(harmful) != len(harmless):
        out.append(f"--paired needs lists of equal length (harmful {len(harmful)}, harmless {len(harmless)})")
    for label, items in (("harmful", harmful), ("harmless", harmless)):
        if len(items) - held_out < 20:
            out.append(f"{label}: {len(items)} prompts leave {len(items) - held_out} for the direction after "
                       f"holding out {held_out}; the docs suggest 150-200 prompts per list")
    return out


def split(harmful: list[str], harmless: list[str], held_out: int) -> dict:
    """The last `held_out` prompts of each list are held out (order your files, or pre-split them, to choose)."""
    if held_out < 1:
        raise ValueError("held_out must be at least 1")
    h, g = len(harmful) - held_out, len(harmless) - held_out
    return {"direction": {"harmful": harmful[:h], "harmless": harmless[:g]},
            "held_out": {"harmful": harmful[h:], "harmless": harmless[g:]}}


def prepare(run, harmful_file: Path, harmless_file: Path, held_out: int, paired: bool, field: str | None) -> dict:
    """Reads, checks and splits the two lists into run/prompts/sets.json; returns the sets."""
    from .run import sha256_file, write_json
    harmful, hcat = read_prompts(Path(harmful_file), field)
    harmless, gcat = read_prompts(Path(harmless_file), field)
    issues = problems(harmful, harmless, held_out, paired)
    if issues:
        raise SystemExit("prompt sets refused:\n  " + "\n  ".join(issues))
    sets = split(harmful, harmless, held_out)
    sets["source"] = {k: {"name": Path(f).name, "sha256": sha256_file(Path(f)), "n": n}
                      for k, f, n in (("harmful", harmful_file, len(harmful)), ("harmless", harmless_file, len(harmless)))}
    sets["categories"] = {"harmful": hcat, "harmless": gcat}
    sets["paired"] = paired
    write_json(run.sub("prompts") / "sets.json", sets)
    return sets


def eval_id(harmful: list[str], harmless: list[str]) -> str:
    """12 hex characters naming the exact graded prompts; part of a study's identity."""
    return hashlib.sha256(json.dumps([harmful, harmless]).encode()).hexdigest()[:12]
