"""A stand-in for tools/ablate (launcher_check's ablation group): the commands the launcher's Ablation page runs, with
no GPU, no model and no download. launcher_check copies it as ablate.py into a tool folder of its own and starts the
launcher with --ablate-tool on that folder.

    datasets list --json | datasets fetch PRESET --out DIR | init RUN --production-argv F --launch-dir D | auto RUN ...

The configuration's environment (which the launcher passes to the tool) picks what `auto` does:
    STUB_AUTO=done       (default) trials of STUB_TRIAL_S seconds (0.3), then an exported sidecar
    STUB_AUTO=no_sidecar the run ends without a sidecar
    STUB_AUTO=fail       the run fails
    STUB_AUTO=stubborn   ignores the stop request (the launcher's "End now" ends it), with its engine running
STUB_REFUSE=1 makes `init` refuse the configuration. Each `auto` starts a stand-in engine (this Python sleeping, in a
process group of its own) recorded in RUN/engines/<pid>.json while it runs, as the real tool's engines are, and adds a
line to RUN/invocations.txt.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_json(path: Path, data) -> None:
    """As the real tool writes (tools/ablate/ablatelib/run.py): whole files by a rename, tried again while Windows
    refuses a rename over the file the launcher has open for a moment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    for attempt in range(60):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.05)


def option(argv: list[str], name: str, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


PRESET = {"name": "stub-set", "title": "Stub prompts", "description": "two tiny lists",
          "sources": [{"title": "Stub harmful", "licence": "MIT", "licence_note": "free to use", "home": "https://example.invalid/h"},
                      {"title": "Stub harmless", "licence": "CC BY-NC 4.0", "licence_note": "non-commercial use only",
                       "home": "https://example.invalid/g"}]}


def datasets(argv: list[str]) -> int:
    if argv[0] == "list":
        print(json.dumps([PRESET]))
        return 0
    name, out = argv[1], Path(option(argv, "--out"))
    if name != PRESET["name"]:
        print(f"unknown prompt set {name!r}", file=sys.stderr)
        return 1
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    for part, n in (("harmful", 12), ("harmless", 12), ("test-harmful", 4), ("test-benign", 4)):
        (folder / f"{part}.txt").write_text("".join(f"{part} prompt {i}\n" for i in range(n)), encoding="utf-8")
    write_json(folder / "sets.json", {"preset": name, "title": PRESET["title"], "fetched": "2026-11-23T00:00:00Z",
                                      "counts": {"harmful": 12, "harmless": 12, "test-harmful": 4, "test-benign": 4},
                                      "sources": {}, "source_text": "stub prompts (MIT; CC BY-NC 4.0)"})
    print(f"{name}: harmful 12, harmless 12 (in {folder})")
    return 0


def init(argv: list[str]) -> int:
    run = Path(argv[0])
    words = json.loads(Path(option(argv, "--production-argv")).read_text(encoding="utf-8"))
    if os.environ.get("STUB_REFUSE") == "1":
        print("this setup cannot be fitted as it stands:\n  --stub-option: the stub refuses it", file=sys.stderr)
        return 1
    write_json(run / "run.json", {"schema": "infernix-ablate-run-1", "argv": words,
                                  "launch_dir": option(argv, "--launch-dir"),
                                  "tool_version": os.environ.get("INFERNIX_ABLATE_TOOL_VERSION", "")})
    print(f"run {run}")
    return 0


def auto(argv: list[str]) -> int:
    import stubpkg  # noqa: F401  (the packages the launcher installed are on the path)
    run = Path(argv[0])
    mode = os.environ.get("STUB_AUTO", "done")
    trials, out, name = int(option(argv, "--trials")), Path(option(argv, "--out")), option(argv, "--name")
    with open(run / "invocations.txt", "a", encoding="utf-8") as fh:
        fh.write(" ".join(argv[1:]) + "\n")
    flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    engine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"], **flags)
    record = run / "engines" / f"{engine.pid}.json"
    write_json(record, {"pid": engine.pid, "executable": os.path.realpath(sys.executable), "purpose": "score"})
    done_file = run / "trials-done.txt"
    done = int(done_file.read_text()) if done_file.is_file() else 0
    progress = {"state": "running", "phase": "fit", "phases_done": ["doctor", "prompts", "capture", "direction"],
                "message": "measuring the stock model", "trials": {"done": done, "target": trials},
                "control": {"refusals": 4, "n": 4}, "best": None, "grade": None, "export": None,
                "started": "2026-11-23T00:00:00Z", "updated": "2026-11-23T00:00:00Z"}
    write_json(run / "progress.json", progress)
    print("stub auto: started", flush=True)

    def stop_engine() -> None:
        engine.terminate()
        engine.wait(30)
        record.unlink(missing_ok=True)

    if mode == "stubborn":
        for sig in ("SIGTERM", "SIGBREAK"):
            if hasattr(signal, sig):
                signal.signal(getattr(signal, sig), signal.SIG_IGN)
        while True:
            time.sleep(1)
    try:
        if mode == "fail":
            stop_engine()
            progress.update(state="failed", message="the stub failed on purpose")
            write_json(run / "progress.json", progress)
            return 1
        while done < trials:
            time.sleep(float(os.environ.get("STUB_TRIAL_S", "0.3")))
            done += 1
            done_file.write_text(str(done))
            progress.update(message=f"trial {done} of {trials}", trials={"done": done, "target": trials, "seconds": 0.3},
                            best={"trial": done - 1, "refusals": 1, "kl": 0.1, "what": "stub"})
            write_json(run / "progress.json", progress)
            print(f"trial {done}", flush=True)
        stop_engine()
        if mode == "no_sidecar":
            progress.update(state="no_sidecar", phase="pick", message="no trial kept the KL below the target")
            write_json(run / "progress.json", progress)
            return 0
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "d.f32.bin").write_bytes(b"\0" * 16)
        write_json(folder / "sidecar.json", {"format": "ablate-direction-v2", "direction_dtype": "float32",
                                             "direction_dim": 4, "direction_file": "d.f32.bin", "default_alpha": 0.0,
                                             "alpha_attn_per_layer": {"1": 1.0},
                                             "generation": {"source": option(argv, "--source", "")}})
        progress.update(state="done", phase="export", message="sidecar exported: stub",
                        export=str(folder / "sidecar.json"))
        write_json(run / "progress.json", progress)
        return 0
    except KeyboardInterrupt:
        stop_engine()
        progress.update(state="stopped", message="stopped; run it again to continue (finished work is kept)")
        write_json(run / "progress.json", progress)
        return 130


def main() -> int:
    # The real tool's stop handling: the launcher's Ctrl+Break (SIGTERM elsewhere) unwinds like Ctrl+C.
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)
    argv = sys.argv[1:]
    return {"datasets": datasets, "init": init, "auto": auto}[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
