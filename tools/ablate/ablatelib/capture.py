"""Capture: every direction prompt's hidden states at its last token, from the model as production serves it.

One scoring engine (infernix-serve --causal-scoring with production's carried options, engine.py) renders each prompt
as one user turn with production's chat template, thinking off, as heretic renders it, and POST /v1/score with
"hidden_states" returns every layer's hidden state at the last prompt token: where the answer starts, after the
template's empty thinking block. Row L is the hidden state layer L hands on (docs/serving.md, "Teacher-forced
scoring"); the direction step reads them and nothing else does.

A recording is FP32 [layers, hidden] and is named by a fingerprint of its prompt and of everything that decides its
values (the artifact, the chat template, production's carried options, the engine build), so adding prompts records
only those, an interrupted capture resumes, and a changed setup records again.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import serve_flags as SF
from .engine import Engine
from .run import artifact_id, files_stamp, replace_file, sha256_file, write_json


def identity(run) -> dict:
    """What decides a recording's values besides its prompt."""
    prod = run.production
    template = prod.value("--chat-template")
    carried = prod.by_kind(SF.CARRY)
    return {"artifact": prod.artifact.name, "artifact_id": artifact_id(prod.artifact),
            "carried": [f if v is None else f"{f} {v}" for f, v in carried],
            # A path-valued option's file as it is now (--lora's adapter, --ngram-volume): one replaced in place
            # records again.
            "carried_files": {f: stamp for f, v in carried if (stamp := files_stamp(v)) is not None},
            "chat_template": sha256_file(Path(template)) if template else "the artifact's own",
            "serve_binary": sha256_file(run.serve_binary)}


def fingerprint(prompt: str, ident: dict) -> str:
    return hashlib.sha256(json.dumps([prompt, ident], sort_keys=True).encode()).hexdigest()[:20]


def run_capture(run, kinds: dict[str, list[str]], log=print) -> dict:
    """Record every prompt not yet recorded. kinds: {'harmful': [...], 'harmless': [...]}. Returns the index."""
    out = run.sub("capture")
    ident = identity(run)
    previous_path = out / "index.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    known = {entry["fingerprint"]: entry for entry in (previous.get("prompts") or {}).values()}
    shape = previous.get("shape") if previous.get("identity") == ident else None
    index: dict = {"identity": ident, "shape": shape, "prompts": {}}
    todo: list[tuple[str, str]] = []
    for kind, prompts in kinds.items():
        for i, prompt in enumerate(prompts):
            fp = fingerprint(prompt, ident)
            entry = {"fingerprint": fp, "prompt_tokens": (known.get(fp) or {}).get("prompt_tokens")}
            index["prompts"][f"{kind}:{i}"] = entry
            if shape is None or entry["prompt_tokens"] is None or not _complete(out / f"{fp}.f32", shape):
                todo.append((f"{kind}:{i}", prompt))
    if not todo:
        log("  every prompt is already recorded")
        write_json(previous_path, index)
        return index
    log(f"  {len(todo)} prompt(s) to record through one scoring engine")
    started = time.monotonic()
    log_path = run.sub("logs") / f"capture-{time.strftime('%Y%m%dT%H%M%S')}.score.log"
    with Engine(run, "score", log_path=log_path) as engine:
        for n, (key, prompt) in enumerate(todo, 1):
            rows, prompt_tokens = engine.hidden_states(prompt)
            got = {"layers": int(rows.shape[0]), "hidden": int(rows.shape[1])}
            if index["shape"] is None:
                index["shape"] = got
            elif got != index["shape"]:
                raise SystemExit(f"{key}: the engine returned {got} hidden states, the capture holds "
                                 f"{index['shape']}: one capture is one model")
            entry = index["prompts"][key]
            target = out / f"{entry['fingerprint']}.f32"
            tmp = target.with_suffix(".f32.tmp")
            rows.astype("<f4").tofile(tmp)
            replace_file(tmp, target)
            entry["prompt_tokens"] = prompt_tokens
            if n % 40 == 0 or n == len(todo):
                log(f"  recorded {n}/{len(todo)} ({time.monotonic() - started:.0f} s with the engine's start)")
                write_json(previous_path, index)     # an interruption keeps what is recorded
    write_json(previous_path, index)
    return index


def _complete(path: Path, shape: dict) -> bool:
    return path.is_file() and path.stat().st_size == int(shape["layers"]) * int(shape["hidden"]) * 4
