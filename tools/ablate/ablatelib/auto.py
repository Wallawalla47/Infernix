"""auto: from two prompt lists to an exported sidecar in one resumable command (what the launcher's Ablation page runs).

doctor, prompts, capture, direction, fit until the study holds --trials trials, the pick, grade (given a test pair)
and export. Rerunning the same command continues: recordings are kept, the study only gains the trials it lacks, and
an export already made is not made again.

The pick (fit.best_trial): the fewest held-out refusals among trials whose first-token KL is below the KL target, then
the lowest KL. Nothing is exported when the stock model refuses none of the held-out prompts, no trial is below the
target, or none refuses fewer than the stock model: the run ends with state "no_sidecar" and says why.

RUN/progress.json says where it is, for a program watching it:
  state        running | done | no_sidecar | stopped | failed
  phase        the current or last phase (PHASES); phases_done: those finished in this invocation
  message      a line for a person
  trials       {"done": completed trials in the study, "target": --trials, "seconds": the mean of the last (up to)
               20 trials' times in this invocation, once one finished}
  control      {"refusals", "n"}: the stock model on the held-out harmful prompts
  best         the best trial so far by the pick's rule: {"trial", "refusals", "kl", "what"}
  grade        the test pair's results for the stock model and the pick (when graded)
  export       the exported sidecar.json
  started, updated   ISO times
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .run import write_json

PHASES = ("doctor", "prompts", "capture", "direction", "fit", "pick", "grade", "export")


class Progress:
    def __init__(self, run, target: int):
        self.path = run.dir / "progress.json"
        now = _now()
        self.data: dict = {"state": "running", "phase": None, "phases_done": [], "message": "starting",
                           "trials": {"done": 0, "target": target}, "control": None, "best": None, "grade": None,
                           "export": None, "started": now, "updated": now}
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


def run_auto(run, *, harmful: Path, harmless: Path, paired: bool, field: str | None, held_out: int, study: str,
             trials: int, kl_target: float, startup: int, markers: tuple[str, ...], test_harmful: Path | None,
             test_benign: Path | None, out: Path, name: str, source: str | None, log=print) -> int:
    from . import capture as C
    from . import candidates as CD
    from . import direction as D
    from . import doctor
    from . import fit as F
    from . import prompts as PR

    p = Progress(run, trials)
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
        completed = ([t for t in F.open_study(run, study).trials if t.state.name == "COMPLETE"]
                     if study_path.is_file() else [])
        existing = len(completed)
        below = [t for t in completed if t.user_attrs.get("kl", 9.9) < kl_target]
        if below:
            b = min(below, key=lambda t: (t.user_attrs.get("refusals", 99), t.user_attrs.get("kl", 9.9), t.number))
            p.update(best={"trial": b.number, "refusals": b.user_attrs.get("refusals"),
                           "kl": round(float(b.user_attrs.get("kl")), 4), "what": b.user_attrs.get("what")})
        p.update(trials={"done": existing, "target": trials})
        control = None
        paces: list[float] = []   # the last trials' seconds: trials.seconds, for a time estimate

        def on_progress(event: dict) -> None:
            nonlocal control
            if "trial" not in event:   # the control (a trial's record names the control's refusals too)
                control = event["control_refusals"]
                p.update(control={"refusals": control, "n": event["n_harmful"]},
                         message=f"stock model: {control}/{event['n_harmful']} held-out harmful prompts refused")
                return
            done = p.data["trials"]["done"] + 1
            best = p.data["best"]
            if event["kl"] < kl_target and (best is None or (event["refusals"], event["kl"]) < (best["refusals"], best["kl"])):
                best = {"trial": event["trial"], "refusals": event["refusals"], "kl": round(event["kl"], 4),
                        "what": event["what"]}
            paces.append(float(event.get("seconds") or 0.0))
            p.update(trials={"done": done, "target": trials, "seconds": round(sum(paces[-20:]) / len(paces[-20:]), 1)},
                     best=best,
                     message=f"trial {done} of {trials}: {event['refusals']}/{event['n_harmful']} refused, "
                             f"KL {event['kl']:.3f}")

        if existing < trials:
            summary = F.fit(run, study, trials - existing, kl_target=kl_target, startup=startup, markers=markers,
                            log=log, progress=on_progress)
            if summary["status"] == "interrupted":
                raise KeyboardInterrupt
            if summary["status"] == "stopped":
                p.finish("failed", "the fit stopped: three trials in a row failed (see the trials' engine logs)")
                return 1
        else:
            # Resumed with every trial done: the stock model's refusals come from the last fit's control.
            controls = sorted((run.dir / "fit" / study).glob("run-*/control.json"))
            if controls:
                control = json.loads(controls[-1].read_text(encoding="utf-8"))["refusals"]["refusals"]
        if control is None:
            raise SystemExit("the study has no control measurement: run the fit again")

        p.phase("pick", "picking the cut")
        trial, why = F.best_trial(F.open_study(run, study), kl_target, control)
        if trial is None:
            p.finish("no_sidecar", why)
            log(f"no sidecar: {why}")
            return 0
        candidate = f"auto-t{trial}"
        if not (run.dir / "candidates" / candidate).is_dir():
            CD.pin(run, study, trial, candidate)
        p.update(message=why)

        if test_harmful is not None:
            p.phase("grade", "grading the pick on the test prompts")
            # Kept with the candidate, so a rerun does not grade the same pick again.
            kept = run.dir / "candidates" / candidate / "grade.json"
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
            log(f"already exported: {dest / 'sidecar.json'}")
        else:
            dest = CD.export(run, candidate, name, source, out)
            log(f"exported {dest / 'sidecar.json'}")
        p.finish("done", f"sidecar exported: {why}", export=str(dest / "sidecar.json"))
        return 0
    except KeyboardInterrupt:
        p.finish("stopped", "stopped; run it again to continue (finished work is kept)")
        return 130
    except SystemExit as e:
        p.finish("failed", str(e))
        raise
    except Exception as e:
        p.finish("failed", f"{type(e).__name__}: {e}")
        raise
