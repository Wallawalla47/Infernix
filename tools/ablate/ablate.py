#!/usr/bin/env python3
"""Fit a refusal-ablation sidecar (--ablate) for your model on your own setup, and evaluate it.

    python tools/ablate/ablate.py init RUN --production prod.txt      # your infernix-serve command line, once
    python tools/ablate/ablate.py all RUN --harmful H --harmless G --study s1
    python tools/ablate/ablate.py list RUN --study s1                  # the front by (refusals, KL), the pick marked
    python tools/ablate/ablate.py pin RUN --study s1 --trial 115 --name mine
    python tools/ablate/ablate.py grade RUN --harmful T --benign B mine
    python tools/ablate/ablate.py eval RUN mine                        # capability: the stock model vs the candidate
    python tools/ablate/ablate.py export RUN mine --name my-cut
    python tools/ablate/ablate.py datasets list                        # public prompt sets and their licences
    python tools/ablate/ablate.py datasets fetch advbench-alpaca --out prompts
    python tools/ablate/ablate.py auto RUN --harmful H --harmless G --validate-harmful VH --validate-benign VB \
        --test-harmful TH --test-benign TB --out sidecars --name my-cut
    python tools/ablate/ablate.py choose RUN --trial 42 --out sidecars --name my-cut-t42   # another trial of auto's study

`all` is doctor, prompts, capture, direction and fit in order; each also runs on its own. `auto` runs them, grades the
fit's best trials on the validation pair and picks the cut on those numbers, grades it on the test pair and exports
it in one resumable command (the launcher's Ablation page runs it); `choose` does the same with another trial of the
same study. docs/ablate-fitting.md is
the guide (what to install, where to get prompt sets, how long each step takes, how to read the results).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablatelib import platform_util as PU  # noqa: E402
from ablatelib import production as P  # noqa: E402
from ablatelib import serve_flags as SF  # noqa: E402
from ablatelib.run import Run, write_json  # noqa: E402


def cmd_init(a) -> int:
    launch = Path(a.launch_dir).resolve() if a.launch_dir else Path.cwd()
    given = (a.production, a.production_argv, a.production_config, a.from_running or None)
    if sum(x is not None and x is not False for x in given) != 1:
        raise SystemExit("give exactly one of --production FILE, --production-argv JSON, --production-config TOML, "
                         "--from-running")
    # The infernix-serve first: its --help says which of the command line's options take a value.
    named = None
    if a.production or a.production_argv:
        tokens = P.read_command_line(Path(a.production)) if a.production else P.read_argv(Path(a.production_argv))
        named = P.named_serve_binary(tokens, launch)
    elif a.from_running:
        tokens, launch, running = P.running_command_line()
        named = P.named_serve_binary(tokens, launch)
    serve = Path(a.serve_binary).resolve() if a.serve_binary else named
    if serve is None:
        raise SystemExit("which infernix-serve? pass --serve-binary (the production command line did not name one)")
    arity = SF.arities_of(serve)
    if a.production or a.production_argv:
        prod = P.parse_tokens(tokens, arity, launch, source=str(a.production or a.production_argv))
    elif a.production_config:
        prod = P.from_config_file(Path(a.production_config), arity, launch, extra=a.extra)
    else:
        prod = P.parse_tokens(tokens, arity, launch, source=running)
    refused = prod.refused()
    if refused:
        raise SystemExit("this setup cannot be fitted as it stands:\n  " + "\n  ".join(refused))
    run = Run(dir=Path(a.run).resolve(), production=prod, serve_binary=serve, context=a.context)
    if run.path.exists() and not a.force:
        raise SystemExit(f"{run.path} exists (pass --force to replace the setup; prompts and results are kept)")
    run.save()
    print(f"run {run.dir}\n  model {prod.artifact}\n  serve {serve}\n  from {prod.source}\nnext: ablate.py doctor {a.run}")
    return 0


def cmd_doctor(a) -> int:
    from ablatelib import doctor
    return 1 if doctor.check(Run.load(Path(a.run))) else 0


def need_free_card(run) -> None:
    busy = PU.card_busy(int(run.production.value("--device") or 0))
    if busy:
        raise SystemExit(f"the card is busy ({busy}): every step starts its own engine on the whole card, so stop "
                         "your production server first")


def cmd_prompts(a) -> int:
    from ablatelib import prompts as PR
    run = Run.load(Path(a.run))
    sets = PR.prepare(run, Path(a.harmful), Path(a.harmless), a.held_out, a.paired, a.field)
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
    C.run_capture(run, {"harmful": d["harmful"], "harmless": d["harmless"]})
    print(f"recordings: {run.dir / 'capture'}")
    return 0


def cmd_direction(a) -> int:
    from ablatelib import direction as D
    run = Run.load(Path(a.run))
    index = json.loads((run.dir / "capture" / "index.json").read_text(encoding="utf-8"))
    path, report = D.build(run, index, run.dir / "direction", orthogonalize=not a.no_orthogonalize)
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
    s = F.fit(run, a.study, a.trials, kl_floor=a.kl_floor, kl_scale=a.kl_scale, startup=a.startup,
              markers=markers_of(a))
    print_front(s["front"], s["control_refusals"])
    return 0 if s["status"] in ("complete", "interrupted") else 1


def print_front(rows, control=None, picked=None) -> None:
    print("front by (refusals, KL): no trial beats one of these on both" +
          (f"  (stock refuses {control})" if control is not None else "") +
          ("; * = the pick by the fit's numbers" if picked is not None else ""))
    for r in rows:
        mark = "*" if picked is not None and r["trial"] == picked else " "
        print(f" {mark}trial {r['trial']:>4}: refusals {r['refusals']}/{r['n']}  KL {r['kl']:.4f}  {r['what']}")


def cmd_list(a) -> int:
    from ablatelib import auto as AU
    from ablatelib import fit as F
    from ablatelib import validate as V
    run = Run.load(Path(a.run))
    rows = F.records(F.open_study(run, a.study))
    control = AU.last_control(run, a.study)
    best = F.pick(rows, control, a.kl_max, a.kl_scale)[0] if control is not None else None
    print_front(F.front(F.open_study(run, a.study))[:a.n], control, best["trial"] if best else None)
    # The rows graded on a validation pair (auto, choose), each pair's record on its own.
    by_trial = {r["trial"]: r for r in rows}
    for record in sorted((run.dir / "fit" / a.study / "validation").glob("*.json")):
        kept = json.loads(record.read_text(encoding="utf-8"))
        validated = [by_trial[int(t)] for t in kept["trials"] if int(t) in by_trial]
        picked = V.pick(validated, kept, a.kl_scale)[0]
        stock = kept["stock"]
        print(f"\nvalidated on {Path(kept['harmful']).name} + {Path(kept['benign']).name} (stock refuses "
              f"{stock['refusals']}/{stock['n']} harmful, {stock['benign_refusals']}/{stock['benign_n']} benign); "
              "v = the pick on these numbers")
        for r in sorted(validated, key=lambda r: V.score(kept["trials"][str(r["trial"])], stock, a.kl_scale)):
            v = kept["trials"][str(r["trial"])]
            kl = v["kl_answered"] if v["kl_answered"] is not None else v["kl"]
            mark = "v" if picked is not None and r["trial"] == picked["trial"] else " "
            print(f" {mark}trial {r['trial']:>4}: refusals {v['refusals']}/{v['n']}  benign refused "
                  f"{v['benign_refusals']}/{v['benign_n']}  KL where stock answered {kl:.4f}  "
                  f"(fit: {r['refusals']}/{r['n']}, KL {r['kl']:.4f})")
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
    print(f"exported {CD.export(Run.load(Path(a.run)), a.candidate, a.name, a.source, Path(a.out) if a.out else None)}")
    return 0


def cmd_datasets(a) -> int:
    from ablatelib import datasets as DS
    if a.action == "list":
        if a.json:
            print(json.dumps(DS.catalogue(), indent=1))
            return 0
        for preset in DS.catalogue():
            print(f"{preset['name']}: {preset['title']}\n  {preset['description']}")
            for source in preset["sources"]:
                print(f"    {source['title']}: {source['licence']} ({source['licence_note']}) {source['home']}")
        return 0
    if not a.preset or not a.out:
        raise SystemExit("datasets fetch PRESET --out DIR")
    DS.fetch_preset(a.preset, Path(a.out))
    return 0


def validation_pair(a) -> tuple[Path | None, Path | None]:
    if bool(a.validate_harmful) != bool(a.validate_benign):
        raise SystemExit("--validate-harmful and --validate-benign go together (refusals and KL)")
    if a.validate_rows < 1:
        raise SystemExit("--validate-rows must be at least 1")
    return (Path(a.validate_harmful), Path(a.validate_benign)) if a.validate_harmful else (None, None)


def cmd_auto(a) -> int:
    from ablatelib import auto as AU
    run = Run.load(Path(a.run))
    if a.test_benign and not a.test_harmful:
        raise SystemExit("--test-benign goes with --test-harmful")
    validate_harmful, validate_benign = validation_pair(a)
    rc = AU.run_auto(run, harmful=Path(a.harmful), harmless=Path(a.harmless), paired=a.paired, field=a.field,
                     held_out=a.held_out, study=a.study, trials=a.trials, kl_floor=a.kl_floor, kl_scale=a.kl_scale,
                     kl_max=a.kl_max, startup=a.startup, markers=markers_of(a),
                     validate_harmful=validate_harmful, validate_benign=validate_benign, validate_rows=a.validate_rows,
                     test_harmful=Path(a.test_harmful) if a.test_harmful else None,
                     test_benign=Path(a.test_benign) if a.test_benign else None, out=Path(a.out), name=a.name,
                     source=a.source)
    return rc


def cmd_choose(a) -> int:
    from ablatelib import auto as AU
    run = Run.load(Path(a.run))
    if a.test_benign and not a.test_harmful:
        raise SystemExit("--test-benign goes with --test-harmful")
    validate_harmful, validate_benign = validation_pair(a)
    if a.test_harmful or validate_harmful:
        need_free_card(run)
    return AU.run_choose(run, study=a.study, trial=a.trial,
                         test_harmful=Path(a.test_harmful) if a.test_harmful else None,
                         test_benign=Path(a.test_benign) if a.test_benign else None, field=a.field,
                         markers=markers_of(a), out=Path(a.out), name=a.name, source=a.source, kl_max=a.kl_max,
                         kl_scale=a.kl_scale, validate_harmful=validate_harmful, validate_benign=validate_benign,
                         validate_rows=a.validate_rows)


def cmd_eval(a) -> int:
    from ablatelib import evaluate as E
    run = Run.load(Path(a.run))
    need_free_card(run)
    out = E.evaluate(run, a.candidate, tasks=a.tasks, context=a.context, limit=a.limit, kl_prompts=a.kl_prompts,
                     field=a.field, unsandboxed=a.unsandboxed_code)
    print(f"summary: {out / 'summary.md'}")
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

    def objective_args(p):
        p.add_argument("--kl-floor", type=float, default=0.01,
                       help="first-token KL below which the KL objective follows the refusals instead, so the search "
                            "skips cuts that change nothing (heretic's 0.01)")
        p.add_argument("--kl-scale", type=float, default=1.0,
                       help="the KL that weighs as much as all of the stock model's refusals (heretic's 1.0): the "
                            "objectives' and the pick's balance")
        p.add_argument("--startup", type=int, default=60, help="random trials before TPE starts modelling (60)")
        p.add_argument("--refusal-markers", help="extra refusal phrases, one per line")

    def pick_args(p):
        p.add_argument("--kl-max", type=float, default=0.4,
                       help="the pick takes no trial at or above this first-token KL (0.4)")

    def fit_args(p):
        p.add_argument("--study", required=True, help="the study name; rerun with the same name to continue it")
        p.add_argument("--trials", type=int, default=300,
                       help="completed trials to add to the study (300; a failed trial is replaced)")
        objective_args(p)

    p = command("init", cmd_init, "record your production setup in a new run directory")
    p.add_argument("--production", help="a text file holding your infernix-serve command line")
    p.add_argument("--production-argv", help="a JSON file holding it as a list of words (what the launcher writes)")
    p.add_argument("--production-config", help="your --config TOML (options after `--` are added on top)")
    p.add_argument("--from-running", action="store_true", help="Linux: read the running infernix-serve's command line")
    p.add_argument("--launch-dir", help="the directory production is started from (relative paths; default: here)")
    p.add_argument("--serve-binary", help="the infernix-serve to run (default: the one the command line names)")
    p.add_argument("--context", type=int, default=16384, help="context of the tool's engines (16384)")
    p.add_argument("--force", action="store_true")
    p.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    command("doctor", cmd_doctor, "check everything a run needs, without loading the model")
    prompt_args(command("prompts", cmd_prompts, "read, check and split the prompt sets"))
    command("capture", cmd_capture, "record the direction prompts' hidden states (needs the card)")
    p = command("direction", cmd_direction, "compute the refusal direction from the recordings")
    p.add_argument("--no-orthogonalize", action="store_true")
    fit_args(command("fit", cmd_fit, "search for the cut (needs the card; resumable)"))
    p = command("list", cmd_list, "show a study's front by (refusals, KL), the pick marked")
    p.add_argument("--study", required=True)
    p.add_argument("-n", type=int, default=40)
    pick_args(p)
    p.add_argument("--kl-scale", type=float, default=1.0, help="the pick's balance (as auto's --kl-scale)")
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
    p = command("eval", cmd_eval, "capability: the stock model against a candidate (needs requirements-eval.txt)")
    p.add_argument("candidate")
    p.add_argument("--tasks", default="kl,gsm8k,ifeval,truthfulqa,humaneval",
                   help="comma list of kl, gsm8k, ifeval, truthfulqa, humaneval")
    p.add_argument("--context", type=int, help="context of the evaluation engines (default: the run's, 16384)")
    p.add_argument("--limit", type=int, help="first N items of each task (a quick look; not comparable to a full run)")
    p.add_argument("--kl-prompts", help="harmless prompts for the KL test (default: the run's held-out harmless)")
    p.add_argument("--field")
    p.add_argument("--unsandboxed-code", action="store_true",
                   help="humaneval without bubblewrap: the model's code runs as you, with only a 90 s timeout")
    p = command("export", cmd_export, "a shareable copy of a candidate: no local paths, what it was fitted under")
    p.add_argument("candidate")
    p.add_argument("--name", required=True)
    p.add_argument("--source", help="describe the prompt sets in plain words (default: from the run)")
    p.add_argument("--out", help="the folder to write it in (default: the run's export/ folder)")
    p = command("auto", cmd_auto, "prompts to an exported sidecar in one resumable command (progress in RUN/progress.json)")
    prompt_args(p)
    p.add_argument("--study", default="pareto", help="the study name (pareto)")
    p.add_argument("--trials", type=int, default=300, help="trials the study holds when the fit ends (300)")
    objective_args(p)
    pick_args(p)

    def keep_args(p):
        p.add_argument("--validate-harmful", help="harmful prompts the fit never scored, to grade the fit's best trials "
                                                  "on and pick by (with --validate-benign)")
        p.add_argument("--validate-benign", help="benign prompts the fit never scored (over-refusals and KL)")
        p.add_argument("--validate-rows", type=int, default=6,
                       help="trials graded on the validation pair, best by the fit's rule first, on the front or "
                            "not (6)")
        p.add_argument("--test-harmful", help="harmful prompts neither the fit nor the validation saw, to grade the "
                                              "kept trial once")
        p.add_argument("--test-benign", help="benign prompts neither saw (over-refusals and KL)")
        p.add_argument("--out", required=True, help="the folder the sidecar is exported to")
        p.add_argument("--name", required=True, help="the exported sidecar's folder name")
        p.add_argument("--source", help="describe the prompt sets in plain words (default: from the run)")

    keep_args(p)
    p = command("choose", cmd_choose, "keep another trial of a study: grade it on the test pair and export it, as auto "
                                      "does with its pick (progress in RUN/progress.json)")
    p.add_argument("--study", default="pareto", help="the study (auto's: pareto)")
    p.add_argument("--trial", type=int, required=True, help="a completed trial of the study (see `list`)")
    p.add_argument("--field")
    p.add_argument("--refusal-markers", help="extra refusal phrases, one per line")
    p.add_argument("--kl-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    pick_args(p)
    keep_args(p)
    p = sub.add_parser("datasets", help="public prompt sets: list them, or download and convert one",
                       description="public prompt sets: list them (with their licences), or download and convert one")
    p.add_argument("action", choices=("list", "fetch"))
    p.add_argument("preset", nargs="?", help="the preset to fetch (see `datasets list`)")
    p.add_argument("--out", help="fetch: the folder the preset's folder is written in")
    p.add_argument("--json", action="store_true", help="list: as JSON")
    p.set_defaults(fn=cmd_datasets)
    parsers["datasets"] = p
    p = command("all", cmd_all, "doctor, prompts, capture, direction and fit in order")
    prompt_args(p)
    fit_args(p)
    p.add_argument("--no-orthogonalize", action="store_true")
    # A stop request (the launcher's Ctrl+Break on Windows, SIGTERM elsewhere) unwinds like Ctrl+C, so every engine the
    # tool started is stopped on the way out instead of keeping the GPU (Python ends on SIGBREAK without unwinding).
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)
    # Each command parses with parse_intermixed_args, so names after options (grade RUN --harmful H mine theirs) land
    # in the positional list; plain parse_args binds an nargs='*' list, empty, together with RUN.
    argv = sys.argv[1:]
    a = parsers[argv[0]].parse_intermixed_args(argv[1:]) if argv and argv[0] in parsers else ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
