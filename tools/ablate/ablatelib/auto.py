"""auto: from two prompt lists to an exported sidecar in one resumable command (what the launcher's Ablation page runs),
and choose: another trial of the same study, graded and exported the same way.

auto runs doctor, prompts, capture, direction, the fit until the study holds --trials trials (fit.py: two objectives,
refusals and KL), validate (given a validation pair), the pick, grade (given a test pair) and export. Rerunning the
same command continues: recordings are kept, the study only gains the trials it lacks, rows already validated are not
graded again, and a pick already graded or exported is not graded or exported again.

The pick: with a validation pair (validate.py), the --validate-rows best trials by the fit's rule are graded on it and
the rule is applied to those numbers, the lowest validation refusal rate over the stock model's + the KL
where the stock model answered / --kl-scale; without one (fit.pick), among trials whose first-token KL is below
--kl-max and that refuse fewer held-out prompts than the stock model, the lowest refusal score + KL / --kl-scale. The
held-out prompts are scored on every trial, so the search fits them: validate on prompts it never scored. Nothing is
exported when the stock model refuses none of the prompts or no trial qualifies: the run ends with state "no_sidecar"
and says why. choose keeps the trial it is given instead (any completed one), so a person can take another point of
the front.

RUN/progress.json says where it is, for a program watching it:
  state        running | done | no_sidecar | stopped | failed
  command      auto | choose
  phase        the current or last phase (PHASES); phases_done: those finished
  message      a line for a person
  trials       {"done": completed trials in the study, "target": --trials, "seconds": the mean of the last (up to)
               20 trials' times in this invocation, once one finished}
  control      {"refusals", "n"}: the stock model on the held-out harmful prompts
  rule         {"kl_max", "kl_scale"}: the pick's rule
  front        the study's Pareto front, fewest refusals first: [{"trial", "refusals", "n", "kl", "what"}], a validated
               row with "validation": {"refusals", "n", "benign_refusals", "benign_n", "kl", "kl_answered"}; once
               validated, each validated trial off the front too, with "off_front": true
  validation   once validated: {"harmful", "benign" (the lists' file names), "stock" (the stock model's numbers, as
               a row's), "trials" (the validated trials)}; null without a validation pair
  pick         the pick so far by the rule: {"trial", "refusals", "n", "kl", "what"} (and "validation" once picked
               on it)
  picked_on    "fit" (the held-out numbers, until validation or without a validation pair) or "validation"
  best         the trial being kept: the pick (auto), or the chosen trial (choose)
  grade        the test pair's results for the stock model and the kept trial (when graded)
  export       the exported sidecar.json
  started, updated   ISO times
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .run import write_json

PHASES = ("doctor", "prompts", "capture", "direction", "fit", "validate", "pick", "grade", "export")


class Progress:
    def __init__(self, run, target: int, command: str = "auto", kl_max: float = 0.4, kl_scale: float = 1.0):
        self.path = run.dir / "progress.json"
        now = _now()
        self.data: dict = {"state": "running", "command": command, "phase": None, "phases_done": [],
                           "message": "starting", "trials": {"done": 0, "target": target}, "control": None,
                           "rule": {"kl_max": kl_max, "kl_scale": kl_scale}, "front": [], "validation": None,
                           "pick": None, "picked_on": "fit", "best": None, "grade": None, "export": None,
                           "started": now, "updated": now}
        self.write()

    def write(self) -> None:
        self.data["updated"] = _now()
        write_json(self.path, self.data)

    def phase(self, name: str, message: str) -> None:
        if self.data["phase"] and self.data["phase"] not in self.data["phases_done"]:
            self.data["phases_done"].append(self.data["phase"])
        self.data.update(phase=name, message=message)
        self.write()

    def update(self, **fields) -> None:
        self.data.update(fields)
        self.write()

    def finish(self, state: str, message: str, **fields) -> None:
        if state == "done" and self.data["phase"] not in self.data["phases_done"]:
            self.data["phases_done"].append(self.data["phase"])
        self.data.update(state=state, message=message, **fields)
        self.write()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _row(r: dict, validated: dict | None = None) -> dict:
    """A trial as progress.json shows it, with its validation numbers when it has them."""
    row = {"trial": r["trial"], "refusals": r["refusals"], "n": r["n"], "kl": round(r["kl"], 4), "what": r["what"]}
    arm = (validated or {}).get(str(r["trial"]))
    if arm:
        from . import validate as V
        row["validation"] = V.shown(arm)
    return row


def _table(rows: list[dict], validated: dict | None = None) -> list[dict]:
    """progress.json's trade-off table: the front, and each validated trial off it (validation takes the best trials the
    rule allows, on the front or not), fewest refusals first."""
    from . import fit as F
    front = F.pareto_front(rows)
    on = {r["trial"] for r in front}
    off = [{**_row(r, validated), "off_front": True} for r in rows
           if str(r["trial"]) in (validated or {}) and r["trial"] not in on]
    return sorted([_row(r, validated) for r in front] + off, key=lambda r: (r["refusals"], r["kl"], r["trial"]))


def _validation(kept: dict | None) -> dict | None:
    if not kept or not kept.get("stock"):
        return None
    from . import validate as V
    return {"harmful": Path(kept["harmful"]).name, "benign": Path(kept["benign"]).name,
            "stock": V.shown(kept["stock"]), "trials": sorted(int(t) for t in kept["trials"])}


def last_control(run, study: str) -> int | None:
    """The stock model's held-out refusals from the study's newest control measurement, if any."""
    controls = sorted((run.dir / "fit" / study).glob("run-*/control.json"))
    if not controls:
        return None
    return json.loads(controls[-1].read_text(encoding="utf-8"))["refusals"]["refusals"]


def run_auto(run, *, harmful: Path, harmless: Path, paired: bool, field: str | None, held_out: int, study: str,
             trials: int, kl_floor: float, kl_scale: float, kl_max: float, startup: int, markers: tuple[str, ...],
             validate_harmful: Path | None, validate_benign: Path | None, validate_rows: int,
             test_harmful: Path | None, test_benign: Path | None, out: Path, name: str, source: str | None,
             log=print) -> int:
    from . import capture as C
    from . import direction as D
    from . import doctor
    from . import fit as F
    from . import prompts as PR
    from . import validate as V

    p = Progress(run, trials, "auto", kl_max, kl_scale)
    try:
        p.phase("doctor", "checking the setup")
        if doctor.check(run, log=log):
            p.finish("failed", "the setup check found problems (see the log)")
            return 1

        p.phase("prompts", "reading and splitting the prompt lists")
        sets = PR.prepare(run, harmful, harmless, held_out, paired, field)

        p.phase("capture", f"recording {len(sets['direction']['harmful']) + len(sets['direction']['harmless'])} "
                           "prompts' hidden states")
        index = C.run_capture(run, {"harmful": sets["direction"]["harmful"],
                                    "harmless": sets["direction"]["harmless"]}, log=log)

        p.phase("direction", "computing the refusal direction")
        _, report = D.build(run, index, run.dir / "direction")
        write_json(run.dir / "direction" / "report.json", report)

        p.phase("fit", "measuring the stock model")
        study_path = run.dir / "fit" / study / "study.sqlite"
        # Completed trials count toward the target: one a stop interrupted (or an engine failure ended) is recorded
        # as failed, and the study gets another in its place.
        rows = F.records(F.open_study(run, study)) if study_path.is_file() else []
        existing = len(rows)
        control = last_control(run, study)
        paces: list[float] = []   # the last trials' seconds: trials.seconds, for a time estimate

        def standing() -> dict:
            best = F.pick(rows, control, kl_max, kl_scale)[0] if control is not None else None
            shown = _row(best) if best else None
            return {"front": [_row(r) for r in F.pareto_front(rows)], "pick": shown, "best": shown}

        p.update(trials={"done": existing, "target": trials}, **standing())

        def on_progress(event: dict) -> None:
            nonlocal control
            if "trial" not in event:   # the control (a trial's record names the control's refusals too)
                control = event["control_refusals"]
                p.update(control={"refusals": control, "n": event["n_harmful"]},
                         message=f"stock model: {control}/{event['n_harmful']} held-out harmful prompts refused",
                         **standing())
                return
            rows.append({"trial": event["trial"], "refusals": event["refusals"], "n": event["n_harmful"],
                         "kl": float(event["kl"]), "refusal_score": float(event["refusal_score"]),
                         "what": event["what"], "sidecar": event["sidecar"]})
            paces.append(float(event.get("seconds") or 0.0))
            done = p.data["trials"]["done"] + 1
            p.update(trials={"done": done, "target": trials, "seconds": round(sum(paces[-20:]) / len(paces[-20:]), 1)},
                     message=f"trial {done} of {trials}: {event['refusals']}/{event['n_harmful']} refused, "
                             f"KL {event['kl']:.3f}", **standing())

        if existing < trials:
            summary = F.fit(run, study, trials - existing, kl_floor=kl_floor, kl_scale=kl_scale, startup=startup,
                            markers=markers, log=log, progress=on_progress)
            if summary["status"] == "interrupted":
                raise KeyboardInterrupt
            if summary["status"] == "stopped":
                p.finish("failed", "the fit stopped: three trials in a row failed (see the trials' engine logs)")
                return 1
        if control is None:
            raise SystemExit("the study has no control measurement: run the fit again")
        p.update(control={"refusals": control, "n": len(sets["held_out"]["harmful"])})

        rows = F.records(F.open_study(run, study))
        if validate_harmful is not None:
            graded = V.candidates(rows, control, kl_max, kl_scale, validate_rows)
            if not graded:
                why = F.pick(rows, control, kl_max, kl_scale)[1]
                p.finish("no_sidecar", why)
                log(f"no sidecar: {why}")
                return 0
            p.phase("validate", f"grading the fit's {len(graded)} best trials on the validation prompts")
            kept = V.validate(run, study, graded, validate_harmful, validate_benign, field, markers, log=log)
            p.update(validation=_validation(kept), front=_table(rows, kept["trials"]))
            p.phase("pick", "picking the cut on the validation numbers")
            best, why = V.pick(graded, kept, kl_scale)
            if best is not None:
                shown = _row(best, kept["trials"])
                p.update(pick=shown, best=shown, picked_on="validation")
        else:
            p.phase("pick", "picking the cut")
            best, why = F.pick(rows, control, kl_max, kl_scale)
        if best is None:
            p.finish("no_sidecar", why)
            log(f"no sidecar: {why}")
            return 0
        return keep(run, p, study, best["trial"], why, test_harmful=test_harmful, test_benign=test_benign,
                    field=field, markers=markers, out=out, name=name, source=source, log=log)
    except KeyboardInterrupt:
        p.finish("stopped", "stopped; run it again to continue (finished work is kept)")
        return 130
    except SystemExit as e:
        p.finish("failed", str(e))
        raise
    except Exception as e:
        p.finish("failed", f"{type(e).__name__}: {e}")
        raise


def run_choose(run, *, study: str, trial: int, test_harmful: Path | None, test_benign: Path | None,
               field: str | None, markers: tuple[str, ...], out: Path, name: str, source: str | None,
               kl_max: float = 0.4, kl_scale: float = 1.0, validate_harmful: Path | None = None,
               validate_benign: Path | None = None, validate_rows: int = 6, log=print) -> int:
    """Keep trial `trial` of `study`: pin it, validate it (given a validation pair), grade it on the test pair and
    export it, as auto does with its pick."""
    from . import fit as F
    from . import validate as V

    path = run.dir / "fit" / study / "study.sqlite"
    if not path.is_file():
        raise SystemExit(f"no study {study!r} in {run.dir / 'fit'}")
    rows = F.records(F.open_study(run, study))
    chosen = next((r for r in rows if r["trial"] == trial), None)
    if chosen is None:
        raise SystemExit(f"trial {trial} of study {study!r} did not complete (completed: "
                         f"{', '.join(str(r['trial']) for r in rows) or 'none'})")
    control = last_control(run, study)
    if control is None:
        raise SystemExit("the study has no control measurement")
    p = Progress(run, len(rows), "choose", kl_max, kl_scale)
    try:
        picked, why, kept, picked_on = F.pick(rows, control, kl_max, kl_scale)[0], F.describe(chosen, control), None, "fit"
        p.update(phases_done=["doctor", "prompts", "capture", "direction", "fit"],
                 trials={"done": len(rows), "target": len(rows)},
                 control={"refusals": control, "n": chosen["n"]},
                 front=[_row(r) for r in F.pareto_front(rows)], pick=_row(picked) if picked else None,
                 best=_row(chosen))
        if validate_harmful is not None:
            # The chosen trial beside the rows auto validated (kept: graded once): the pick on show is auto's.
            p.phase("validate", f"grading trial {trial} on the validation prompts")
            graded = V.candidates(rows, control, kl_max, kl_scale, validate_rows)
            kept = V.validate(run, study, [chosen], validate_harmful, validate_benign, field, markers, log=log)
            validated = V.pick(graded, kept, kl_scale)[0] if all(str(r["trial"]) in kept["trials"] for r in graded) \
                else None
            if validated is not None:
                picked, picked_on = validated, "validation"
            why = V.describe(chosen, kept["trials"][str(trial)], kept["stock"])
            p.update(validation=_validation(kept), front=_table(rows, kept["trials"]),
                     pick=_row(picked, kept["trials"]) if picked else None, picked_on=picked_on,
                     best=_row(chosen, kept["trials"]))
        p.phase("pick", f"keeping trial {trial}")
        return keep(run, p, study, trial, why, test_harmful=test_harmful,
                    test_benign=test_benign, field=field, markers=markers, out=out, name=name, source=source, log=log)
    except KeyboardInterrupt:
        p.finish("stopped", "stopped; run it again to continue (finished work is kept)")
        return 130
    except SystemExit as e:
        p.finish("failed", str(e))
        raise
    except Exception as e:
        p.finish("failed", f"{type(e).__name__}: {e}")
        raise


def keep(run, p: Progress, study: str, trial: int, why: str, *, test_harmful: Path | None, test_benign: Path | None,
         field: str | None, markers: tuple[str, ...], out: Path, name: str, source: str | None, log=print) -> int:
    """Pin the trial (as STUDY-tTRIAL), grade it on the test pair once, export it as out/name, finish the progress."""
    from . import candidates as CD

    candidate = f"{study}-t{trial}"
    if not (run.dir / "candidates" / candidate).is_dir():
        CD.pin(run, study, trial, candidate)
    p.update(message=why)

    if test_harmful is not None:
        from . import validate as V
        p.phase("grade", f"grading trial {trial} on the test prompts")
        # Kept with the candidate under the test lists' key, so a rerun does not grade the same trial on the same
        # lists again, and other lists are graded.
        kept = run.dir / "candidates" / candidate / f"grade-{V.lists_key(run, test_harmful, test_benign, markers)}.json"
        if kept.is_file():
            grade = json.loads(kept.read_text(encoding="utf-8"))
        else:
            graded = CD.grade(run, test_harmful, test_benign, [candidate], field, markers, log=log)
            summary = json.loads((graded / "summary.json").read_text(encoding="utf-8"))
            grade = {arm["arm"]: {"harmful": arm["harmful"], "benign": arm["benign"], "kl": arm.get("kl"),
                                  "kl_stock_answered": arm.get("kl_stock_answered")} for arm in summary["arms"]}
            write_json(kept, grade)
        p.update(grade=grade)

    p.phase("export", "exporting the sidecar")
    dest = Path(out) / name
    if (dest / "sidecar.json").is_file():
        if not CD.same_cut(CD.candidate_sidecar(run, candidate), dest / "sidecar.json"):
            raise SystemExit(f"{dest} holds another cut: choose another name")
        log(f"already exported: {dest / 'sidecar.json'}")
    else:
        dest = CD.export(run, candidate, name, source, out)
        log(f"exported {dest / 'sidecar.json'}")
    p.finish("done", f"sidecar exported: {why}", export=str(dest / "sidecar.json"))
    return 0
