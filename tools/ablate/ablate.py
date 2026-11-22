#!/usr/bin/env python3
"""Fit a refusal-ablation sidecar (--ablate) for Qwen3.8-Flash-Next on your own setup.

    python tools/ablate/ablate.py init RUN --production prod.txt      # your infernix-serve command line, once
    python tools/ablate/ablate.py all RUN --harmful H --harmless G --study s1
    python tools/ablate/ablate.py list RUN --study s1                  # pick a trial from the front
    python tools/ablate/ablate.py pin RUN --study s1 --trial 115 --name mine
    python tools/ablate/ablate.py grade RUN --harmful T --benign B mine
    python tools/ablate/ablate.py export RUN mine --name my-cut

`all` is doctor, prompts, capture, direction and fit in order; each also runs on its own. docs/ablate-fitting.md is
the guide (what to install, where to get prompt sets, how long each step takes, how to read the results).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablatelib import platform_util as PU  # noqa: E402
from ablatelib import production as P  # noqa: E402
from ablatelib import serve_flags as SF  # noqa: E402
from ablatelib.run import REPO_ROOT, Run, sha256_file, write_json  # noqa: E402


def cmd_init(a) -> int:
    arity = SF.arities(REPO_ROOT)
    launch = Path(a.launch_dir).resolve() if a.launch_dir else Path.cwd()
    if sum(x is not None and x is not False for x in (a.production, a.production_config, a.from_running or None)) != 1:
        raise SystemExit("give exactly one of --production FILE, --production-config TOML, --from-running")
    if a.production:
        prod = P.from_text_file(Path(a.production), arity, launch)
    elif a.production_config:
        prod = P.from_config_file(Path(a.production_config), arity, launch, extra=a.extra)
    else:
        prod = P.from_running(arity)
    serve = Path(a.serve_binary).resolve() if a.serve_binary else prod.serve_binary
    if serve is None:
        raise SystemExit("which infernix-serve? pass --serve-binary (the production command line did not name one)")
    capture = Path(a.capture_binary).resolve() if a.capture_binary else None
    if capture is None:
        guess = serve.parent.parent / "tests" / ("infernix_qwen4_exp_forward_real_test" + (".exe" if sys.platform == "win32" else ""))
        capture = guess if guess.is_file() else None
    refused = prod.refused()
    if refused:
        raise SystemExit("this setup cannot be fitted as it stands:\n  " + "\n  ".join(refused))
    run = Run(dir=Path(a.run).resolve(), production=prod, serve_binary=serve, capture_binary=capture, context=a.context)
    if run.path.exists() and not a.force:
        raise SystemExit(f"{run.path} exists (pass --force to replace the setup; prompts and results are kept)")
    run.save()
    print(f"run {run.dir}\n  model {prod.artifact}\n  serve {serve}\n  capture {capture or '(none yet)'}\n"
          f"  from {prod.source}\nnext: ablate.py doctor {a.run}")
    return 0


def cmd_doctor(a) -> int:
    from ablatelib import doctor
    return 1 if doctor.check(Run.load(Path(a.run))) else 0


def need_free_card(run) -> None:
    busy = PU.card_processes(int(run.production.value("--device") or 0))
    if busy:
        raise SystemExit(f"the card is busy ({len(busy)} process(es)): every step starts its own engine on the whole "
                         "card, so stop your production server first")


def cmd_prompts(a) -> int:
    from ablatelib import prompts as PR
    run = Run.load(Path(a.run))
    harmful, hcat = PR.read_prompts(Path(a.harmful), a.field)
    harmless, gcat = PR.read_prompts(Path(a.harmless), a.field)
    issues = PR.problems(harmful, harmless, a.held_out, a.paired)
    if issues:
        raise SystemExit("prompt sets refused:\n  " + "\n  ".join(issues))
    sets = PR.split(harmful, harmless, a.held_out)
    sets["source"] = {k: {"name": Path(f).name, "sha256": sha256_file(Path(f)), "n": n}
                      for k, f, n in (("harmful", a.harmful, len(harmful)), ("harmless", a.harmless, len(harmless)))}
    sets["categories"] = {"harmful": hcat, "harmless": gcat}
    sets["paired"] = a.paired
    write_json(run.sub("prompts") / "sets.json", sets)
    d, h = sets["direction"], sets["held_out"]
    print(f"direction: {len(d['harmful'])} harmful / {len(d['harmless'])} harmless;  held out (graded every trial): "
          f"{len(h['harmful'])} / {len(h['harmless'])}  eval id {PR.eval_id(h['harmful'], h['harmless'])}")
    return 0


def cmd_capture(a) -> int:
    from ablatelib import capture as C
    from ablatelib.fit import load_sets
    run = Run.load(Path(a.run))
    need_free_card(run)
    d = load_sets(run)["direction"]
    kinds = {"harmful": d["harmful"], "harmless": d["harmless"]}
    C.run_capture(run, kinds, mode_request=a.capture_mode)
    C.check_against_engine(run, kinds)
    print(f"recordings: {run.dir / 'capture'}")
    return 0


def cmd_direction(a) -> int:
    from ablatelib import direction as D
    run = Run.load(Path(a.run))
    index = json.loads((run.dir / "capture" / "index.json").read_text(encoding="utf-8"))
    path, report = D.build(run, index, run.dir / "direction", measure_at=a.measure_at, token=a.token,
                           orthogonalize=not a.no_orthogonalize)
    write_json(run.dir / "direction" / "report.json", report)
    print(f"base sidecar {path}  d_sha256 {report['d_sha256'][:16]}  ({report['recipe']})")
    return 0


def markers_of(a):
    from ablatelib import measure as M
    return M.load_markers(Path(a.refusal_markers) if getattr(a, "refusal_markers", None) else None)


def cmd_fit(a) -> int:
    from ablatelib import fit as F
    run = Run.load(Path(a.run))
    need_free_card(run)
    s = F.fit(run, a.study, a.trials, kl_target=a.kl_target, startup=a.startup, markers=markers_of(a))
    print_front(s["front"], s["control_refusals"])
    return 0 if s["status"] in ("complete", "interrupted") else 1


def print_front(rows, control=None) -> None:
    print("front by (refusals, KL); the pick is yours:" + (f"  (stock refuses {control})" if control is not None else ""))
    for r in rows:
        print(f"  trial {r['trial']:>4}: refusals {r['refusals']}/{r['n']}  KL {r['kl']:.4f}  {r['what']}")


def cmd_list(a) -> int:
    from ablatelib import candidates as CD
    print_front(CD.list_front(Run.load(Path(a.run)), a.study, a.n))
    return 0


def cmd_pin(a) -> int:
    from ablatelib import candidates as CD
    print(f"pinned {CD.pin(Run.load(Path(a.run)), a.study, a.trial, a.name)}")
    return 0


def cmd_grade(a) -> int:
    from ablatelib import candidates as CD
    run = Run.load(Path(a.run))
    need_free_card(run)
    out = CD.grade(run, Path(a.harmful), Path(a.benign) if a.benign else None, a.candidates, a.field, markers_of(a))
    print(f"summary: {out / 'summary.md'}")
    return 0


def cmd_chat(a) -> int:
    from ablatelib import candidates as CD
    run = Run.load(Path(a.run))
    need_free_card(run)
    CD.chat(run, a.candidate, a.context, thinking=not a.no_thinking)
    return 0


def cmd_export(a) -> int:
    from ablatelib import candidates as CD
    print(f"exported {CD.export(Run.load(Path(a.run)), a.candidate, a.name, a.source)}")
    return 0


def cmd_all(a) -> int:
    from ablatelib import doctor
    run = Run.load(Path(a.run))
    if doctor.check(run):
        return 1
    for step in (cmd_prompts, cmd_capture, cmd_direction, cmd_fit):
        print(f"\n== {step.__name__.removeprefix('cmd_')}")
        rc = step(a)
        if rc:
            return rc
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="ablate.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def command(name, fn, help_text):
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("run", help="the run directory")
        p.set_defaults(fn=fn)
        parsers[name] = p
        return p

    parsers: dict[str, argparse.ArgumentParser] = {}

    def prompt_args(p):
        p.add_argument("--harmful", required=True, help="harmful prompts (.txt one per line, .jsonl, .json or .csv)")
        p.add_argument("--harmless", required=True, help="harmless prompts, same formats")
        p.add_argument("--field", help="the prompt column in .jsonl/.csv files (default: prompt, text, goal, Behavior...)")
        p.add_argument("--held-out", type=int, default=40, help="prompts held out of each list to grade trials (40)")
        p.add_argument("--paired", action="store_true", help="the lists are matched counterparts (line i with line i)")

    def fit_args(p):
        p.add_argument("--study", required=True, help="the study name; rerun with the same name to continue it")
        p.add_argument("--trials", type=int, default=300, help="trials to add to the study (300)")
        p.add_argument("--kl-target", type=float, default=0.4, help="first-token KL at which a trial is judged on drift "
                                                                    "alone (0.4; a cliff, so keep it loose)")
        p.add_argument("--startup", type=int, default=60, help="random trials before TPE starts modelling (60)")
        p.add_argument("--refusal-markers", help="extra refusal phrases, one per line")

    p = command("init", cmd_init, "record your production setup in a new run directory")
    p.add_argument("--production", help="a text file holding your infernix-serve command line")
    p.add_argument("--production-config", help="your --config TOML (options after `--` are added on top)")
    p.add_argument("--from-running", action="store_true", help="Linux: read the running infernix-serve's command line")
    p.add_argument("--launch-dir", help="the directory production is started from (relative paths; default: here)")
    p.add_argument("--serve-binary", help="the infernix-serve to run (default: the one the command line names)")
    p.add_argument("--capture-binary", help="infernix_qwen4_exp_forward_real_test (a -DBUILD_TESTING=ON build)")
    p.add_argument("--context", type=int, default=16384, help="context of the tool's engines (16384)")
    p.add_argument("--force", action="store_true")
    p.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    command("doctor", cmd_doctor, "check everything a run needs, without loading the model")
    prompt_args(command("prompts", cmd_prompts, "read, check and split the prompt sets"))
    p = command("capture", cmd_capture, "record the direction prompts (needs the card)")
    p.add_argument("--capture-mode", choices=("auto", "default", "stream"), default="auto",
                   help="default pins every expert in RAM; stream reads them from disk per prompt (auto: by free RAM)")
    p = command("direction", cmd_direction, "compute the refusal direction from the recordings")
    p.add_argument("--measure-at", choices=("next_layer_input", "lane_output"), default="next_layer_input")
    p.add_argument("--token", choices=("last", "mean"), default="last")
    p.add_argument("--no-orthogonalize", action="store_true")
    fit_args(command("fit", cmd_fit, "search for the cut (needs the card; resumable)"))
    p = command("list", cmd_list, "show a study's front by (refusals, KL)")
    p.add_argument("--study", required=True)
    p.add_argument("-n", type=int, default=15)
    p = command("pin", cmd_pin, "keep one trial as a named candidate")
    p.add_argument("--study", required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--name", required=True)
    p = command("grade", cmd_grade, "out-of-sample refusals, over-refusals and KL: the stock model and candidates")
    p.add_argument("--harmful", required=True)
    p.add_argument("--benign")
    p.add_argument("--field")
    p.add_argument("--refusal-markers")
    p.add_argument("candidates", nargs="*", help="pinned names or sidecar.json paths")
    p = command("chat", cmd_chat, "talk to a candidate (or the stock model)")
    p.add_argument("candidate", nargs="?")
    p.add_argument("--context", type=int)
    p.add_argument("--no-thinking", action="store_true")
    p = command("export", cmd_export, "a shareable copy of a candidate: no local paths, what it was fitted under")
    p.add_argument("candidate")
    p.add_argument("--name", required=True)
    p.add_argument("--source", help="describe the prompt sets in plain words (default: from the run)")
    p = command("all", cmd_all, "doctor, prompts, capture, direction and fit in order")
    prompt_args(p)
    fit_args(p)
    p.add_argument("--capture-mode", choices=("auto", "default", "stream"), default="auto")
    p.add_argument("--measure-at", choices=("next_layer_input", "lane_output"), default="next_layer_input")
    p.add_argument("--token", choices=("last", "mean"), default="last")
    p.add_argument("--no-orthogonalize", action="store_true")
    # Each command parses with parse_intermixed_args, so names after options (grade RUN --harmful H mine theirs) land
    # in the positional list; plain parse_args binds an nargs='*' list, empty, together with RUN.
    argv = sys.argv[1:]
    a = parsers[argv[0]].parse_intermixed_args(argv[1:]) if argv and argv[0] in parsers else ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
