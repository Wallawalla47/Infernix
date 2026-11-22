"""doctor: everything a long run depends on, checked before it starts. Nothing here loads the model."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

from . import platform_util as PU
from . import serve_flags as SF
from .capture import choose_mode, shard_bytes
from .engine import Engine

REQUIRED = {"numpy": "numpy", "optuna": "optuna", "jinja2": "jinja2", "tokenizers": "tokenizers"}


def check(run, log=print) -> int:
    """Prints a line per check; returns the number of problems (0 = ready)."""
    problems = 0

    def say(ok: bool | None, text: str) -> None:
        nonlocal problems
        mark = {True: "ok  ", False: "FAIL", None: "note"}[ok]
        problems += ok is False
        log(f"  {mark} {text}")

    say(sys.version_info >= (3, 11), f"Python {sys.version.split()[0]} (3.11 or newer: tomllib)")
    for module, label in REQUIRED.items():
        try:
            m = importlib.import_module(module)
            say(True, f"{label} {getattr(m, '__version__', '')}".strip())
        except ImportError:
            say(False, f"{label} is not installed (pip install -r tools/ablate/requirements.txt)")

    prod = run.production
    say(run.serve_binary.is_file(), f"infernix-serve: {run.serve_binary}")
    say(run.capture_binary is not None and Path(run.capture_binary).is_file(),
        f"capture test: {run.capture_binary or '(not set: build with -DBUILD_TESTING=ON and pass --capture-binary)'}")
    try:
        size = shard_bytes(prod.artifact)
        say(True, f"model: {prod.artifact.name} ({size / PU.GIB:.1f} GiB over its shards)")
    except OSError as e:
        say(False, f"model: {e}")
    for flag in ("--ngram-volume", "--chat-template", "--expert-stripe"):
        v = prod.value(flag)
        if v:
            say(Path(v).is_file(), f"{flag}: {v}")
    for r in prod.refused():
        say(False, f"production uses {r}")
    carried = [f for f, _ in prod.by_kind(SF.CARRY)]
    dropped = [f for f, _ in prod.by_kind(SF.DROP)]
    say(None, f"carried from production: {' '.join(carried) or '(nothing)'}")
    say(None, f"set by the tool: --max-context {run.context} --max-concurrency 1 --spec off --expert-state <run>/expert.state "
              f"(production's {' '.join(f for f, _ in prod.by_kind(SF.OWNED)) or 'none'} replaced)")
    if dropped:
        say(None, f"left out: {' '.join(dropped)}")

    for purpose in ("generate", "score"):
        with tempfile.TemporaryDirectory() as td:
            plan = Path(td) / "plan.json"
            argv = Engine(run, purpose).argv(port=1) + ["--plan", str(plan)]
            try:
                rc = subprocess.run(argv, capture_output=True, text=True, timeout=300).returncode
            except (OSError, subprocess.TimeoutExpired) as e:
                say(False, f"{purpose} engine --plan could not run: {e}")
                continue
            say(rc == 0, f"{purpose} engine fits this machine (infernix-serve --plan: exit {rc}"
                         + {0: ", fits)", 3: ", does not fit)", 1: ", no plan)"}.get(rc, ")"))

    try:
        busy = PU.card_processes(int(prod.value("--device") or 0))
        say(not busy, "the card is free" if not busy else
            f"the card is busy ({len(busy)} process(es), {sum(m for _, m in busy)} MiB): stop production first")
    except RuntimeError as e:
        say(False, str(e))
    mode, why = choose_mode(run, "auto")
    say(None, f"capture would run in {mode} mode: {why}")
    free = PU.free_disk_bytes(run.dir)
    say(free >= 25 * PU.GIB, f"{free / PU.GIB:.0f} GiB free disk at {run.dir} (a capture of 400 prompts writes ~20 GiB)")
    log("ready" if not problems else f"{problems} problem(s) to fix first")
    return problems
