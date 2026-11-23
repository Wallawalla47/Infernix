"""doctor: everything a long run depends on, checked before it starts. Nothing here loads the model."""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

from . import platform_util as PU
from . import serve_flags as SF
from .engine import Engine
from .run import artifact_id

REQUIRED = {"numpy": "numpy", "optuna": "optuna"}
OPTIONAL_EVAL = {"lm_eval": "lm-eval (only for `eval`)"}


def check(run, log=print) -> int:
    """Prints a line per check; returns the number of problems (0 = ready)."""
    problems = 0

    def say(ok: bool | None, text: str) -> None:
        nonlocal problems
        mark = {True: "ok  ", False: "FAIL", None: "note"}[ok]
        problems += ok is False
        log(f"  {mark} {text}")

    say(sys.version_info >= (3, 11), f"Python {sys.version.split()[0]} (3.11 or newer: tomllib)")
    for module, label in {**REQUIRED, **OPTIONAL_EVAL}.items():
        try:
            m = importlib.import_module(module)
            say(True, f"{label} {getattr(m, '__version__', '')}".strip())
        except ImportError:
            say(None if module in OPTIONAL_EVAL else False,
                f"{label} is not installed (pip install -r tools/ablate/requirements"
                + ("-eval" if module in OPTIONAL_EVAL else "") + ".txt)")

    prod = run.production
    say(run.serve_binary.is_file(), f"infernix-serve: {run.serve_binary}")
    try:
        say(True, f"model: {prod.artifact.name} (artifact {artifact_id(prod.artifact)})")
    except (OSError, SystemExit) as e:
        say(False, f"model: {e}")
    for flag in ("--ngram-volume", "--chat-template", "--expert-stripe", "--lora"):
        v = prod.value(flag)
        if v:
            say(Path(v).exists(), f"{flag}: {v}")
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
        busy = PU.card_busy(int(prod.value("--device") or 0))
        say(busy is None, "the card is free" if busy is None else f"the card is busy ({busy}): stop production first")
    except RuntimeError as e:
        say(False, str(e))
    free = PU.free_disk_bytes(run.dir)
    say(free >= 2 * PU.GIB, f"{free / PU.GIB:.0f} GiB free disk at {run.dir} (a capture of 400 prompts writes up to "
                            "0.5 GiB, a 300-trial fit about 120 MB)")
    log("ready" if not problems else f"{problems} problem(s) to fix first")
    return problems
