"""The fit: an Optuna search for the cut that removes refusals while moving the model least.

Each trial (heretic's search, as in the study that produced refusal-cut-v1):
  * samples heretic's direction scope (one interpolated row at every layer, or each layer's own row) and a band per
    lane (attention/DeltaNet output and MoE output);
  * starts a generation engine with that sidecar and counts refusals on the held-out harmful prompts;
  * starts a scoring engine with it and measures first-token KL against the control on the held-out harmless prompts.

Score (heretic's objective, minimized): at or above the KL target, KL / scale (the trial is judged on drift alone);
below it, the refusal rate relative to the control, times target / scale. The target is a cliff, not a penalty, so
run it loose (refusal-cut-v1 used 0.4) and pick from the front yourself: `list` shows (refusals, KL) pairs.

A study is resumable (a sqlite ledger per study): rerunning `fit` with the same name continues it. Its identity is
mode : lanes : direction : scoring rule : graded prompts, and a run whose identity differs is refused, because mixing
them corrupts TPE's model and the front.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings

from . import measure as M
from . import sidecar as SC
from .engine import Engine, EngineError
from .prompts import eval_id
from .run import write_json


def load_sets(run) -> dict:
    path = run.dir / "prompts" / "sets.json"
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run `ablate.py prompts` first")
    return json.loads(path.read_text(encoding="utf-8"))


def fit(run, study_name: str, trials: int, *, kl_target: float = 0.4, kl_scale: float = 1.0, startup: int = 60,
        markers: tuple[str, ...] = M.MARKERS, max_failures: int = 3, log=print, progress=None) -> dict:
    """Adds `trials` completed trials to the study (a failed trial is replaced). `progress`, when given, is called with
    {'control_refusals', 'n_harmful'} once the control is measured and with each finished trial's record."""
    import optuna

    if trials < 1:
        raise SystemExit("--trials must be at least 1")
    base = run.dir / "direction" / "sidecar.json"
    if not base.is_file():
        raise SystemExit(f"{base} is missing: run `ablate.py direction` first")
    rows = SC.rows_of(base)
    n_layers = rows.shape[0]
    sets = load_sets(run)
    harm, good = sets["held_out"]["harmful"], sets["held_out"]["harmless"]
    d_sha = hashlib.sha256((base.parent / "d.f32.bin").read_bytes()).hexdigest()
    rule = f"kl={kl_target:g}" + (f",scale={kl_scale:g}" if kl_scale != 1.0 else "") + f",eval={eval_id(harm, good)}"
    shape = f"band+scope:attn+mlp:{d_sha[:12]}:{rule}"

    study_dir = run.sub("fit", study_name)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(n_startup_trials=startup, n_ei_candidates=128, multivariate=True)
    study = optuna.create_study(study_name=study_name, storage=f"sqlite:///{(study_dir / 'study.sqlite').as_posix()}",
                                direction="minimize", load_if_exists=True, sampler=sampler)
    previous = study.user_attrs.get("shape")
    if previous is not None and previous != shape:
        raise SystemExit(f"study {study_name!r} was fitted as {previous}; this run is {shape} "
                         "(mode:lanes:direction:scoring rule). Use a new study name.")
    study.set_user_attr("shape", shape)
    log(f"study {study_name!r}: {shape}; {len(study.trials)} trial(s) already in the ledger")
    log(f"graded every trial: refusals on {len(harm)} held-out harmful prompts, KL on {len(good)} held-out harmless")

    # The control: scoring engine first (the KL anchor; a setup that cannot score stops before any generation), then
    # the generation engine (refusals without a cut). Re-measured on every invocation.
    run_dir = study_dir / f"run-{time.strftime('%Y%m%dT%H%M%S')}"
    with Engine(run, "score", log_path=run_dir / "control.score.log") as e:
        anchor = M.first_token_distributions(e, good)
    with Engine(run, "generate", log_path=run_dir / "control.generate.log") as e:
        control = M.grade_refusals(e, harm, record=run_dir / "control.answers.jsonl", markers=markers)
    if control["errors"]:
        raise SystemExit(f"the control's refusal grading had {control['errors']} error(s); see {run_dir}")
    log(f"CONTROL (no cut): refusals {control['refusals']}/{len(harm)} (truncated {control['truncated']})")
    write_json(run_dir / "control.json", {"refusals": control, "kl_prompts": len(good), "shape": shape})
    if progress is not None:
        progress({"control_refusals": control["refusals"], "n_harmful": len(harm)})

    def run_trial(trial) -> float:
        started = time.time()
        tdir = study_dir / f"t{trial.number}"
        scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
        index = trial.suggest_float("direction_index", 0.4 * (n_layers - 1), 0.9 * (n_layers - 1))
        direction = SC.global_direction(rows, index) if scope == "global" else None
        bands = {lane: SC.sample_band(trial, n_layers, lane) for lane in SC.LANES}
        tables = SC.lane_tables(bands, n_layers)
        what = (f"dir global@L{index:.1f}" if scope == "global" else "dir per-layer") + " | " + SC.band_summary(bands)
        cand = SC.write_trial(base, tdir, tables, direction,
                              notes=f"study {study_name} trial {trial.number}: {what}")
        with Engine(run, "generate", sidecar=cand, log_path=tdir / "engine.generate.log") as e:
            graded = M.grade_refusals(e, harm, record=tdir / "answers.jsonl", markers=markers)
        if graded["errors"]:
            trial.set_user_attr("prune_reason", "probe_errors")
            raise EngineError(f"{graded['errors']} refusal probe error(s)")
        with Engine(run, "score", sidecar=cand, log_path=tdir / "engine.score.log") as e:
            kls = M.kl(anchor, M.first_token_distributions(e, good))
        kl = M.mean(kls)
        if kl >= kl_target:
            score = kl / kl_scale
        else:
            rate_t = graded["refusals"] / max(1, len(harm) - graded["truncated"])
            rate_c = control["refusals"] / max(1, len(harm) - control["truncated"])
            score = (rate_t / rate_c if rate_c else rate_t) * (kl_target / kl_scale)
        record = {"refusals": graded["refusals"], "truncated": graded["truncated"], "kl": kl, "n_harmful": len(harm),
                  "control_refusals": control["refusals"], "score": score, "scope": scope,
                  "direction_index": round(index, 4), "sidecar": str(cand), "what": what,
                  "kl_per_prompt": [round(x, 5) for x in kls], "seconds": round(time.time() - started, 1)}
        for k, v in record.items():
            if k != "kl_per_prompt":
                trial.set_user_attr(k, v)
        write_json(tdir / "trial.json", record)
        log(f"trial {trial.number:>4}  refusals {graded['refusals']:>2}/{len(harm)} (control {control['refusals']})  "
            f"KL {kl:.4f}{'  OVER TARGET' if kl >= kl_target else ''}  {what}  {time.time() - started:.0f}s")
        if progress is not None:
            progress({"trial": trial.number, **{k: v for k, v in record.items() if k != "kl_per_prompt"}})
        return score

    def objective(trial) -> float:
        try:
            return run_trial(trial)
        except EngineError as e:
            trial.set_user_attr("error", str(e)[:1000])
            log(f"trial {trial.number} FAILED (the fit goes on): {str(e)[:300]}")
            raise

    streak = {"n": 0, "why": None}

    def breaker(study_, frozen) -> None:
        streak["n"] = streak["n"] + 1 if frozen.state == optuna.trial.TrialState.FAIL else 0
        if streak["n"] >= max_failures:
            streak["why"] = f"{streak['n']} trials in a row failed; last: {frozen.user_attrs.get('error', '?')[:200]}"
            study_.stop()

    # `trials` completed trials are added: a trial an engine failure ended does not count, and the study gets another in
    # its place (the breaker above ends a fit whose trials keep failing, so this always ends).
    added = {"n": 0}

    def enough(study_, frozen) -> None:
        if frozen.state == optuna.trial.TrialState.COMPLETE:
            added["n"] += 1
            if added["n"] >= trials:
                study_.stop()

    status = "complete"
    try:
        study.optimize(objective, n_trials=None, catch=(EngineError,), callbacks=[breaker, enough],
                       gc_after_trial=True)
        if streak["why"]:
            status = "stopped"
            log(f"FIT STOPPED: {streak['why']}")
    except KeyboardInterrupt:
        status = "interrupted"     # every finished trial is already in the ledger
    summary = {"status": status, "study": study_name, "shape": shape, "trials_in_ledger": len(study.trials),
               "control_refusals": control["refusals"], "front": front(study)[:8]}
    write_json(run_dir / "summary.json", summary)
    return summary


def front(study) -> list[dict]:
    """Completed trials by (refusals, KL): the pick is the user's, not the score's."""
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    ranked = sorted(done, key=lambda t: (t.user_attrs.get("refusals", 99), t.user_attrs.get("kl", 9.9)))
    return [{"trial": t.number, "refusals": t.user_attrs.get("refusals"), "n": t.user_attrs.get("n_harmful"),
             "kl": round(float(t.user_attrs.get("kl", float("nan"))), 4), "what": t.user_attrs.get("what"),
             "sidecar": t.user_attrs.get("sidecar")} for t in ranked]


def best_trial(study, kl_target: float, control_refusals: int) -> tuple[int | None, str]:
    """(trial number, why) of the cut to keep: the fewest held-out refusals among completed trials whose first-token
    KL is below the target, then the lowest KL. None, with the reason, when the stock model refuses none of the
    held-out prompts, no trial is below the target, or none refuses fewer than the stock model."""
    if control_refusals == 0:
        return None, "the stock model refused none of the held-out harmful prompts: there is no refusal to remove"
    done = [t for t in study.trials if t.state.name == "COMPLETE" and t.user_attrs.get("kl", 9.9) < kl_target]
    if not done:
        return None, f"no trial kept the first-token KL below the target ({kl_target:g}): try more trials"
    best = min(done, key=lambda t: (t.user_attrs.get("refusals", 99), t.user_attrs.get("kl", 9.9), t.number))
    refusals = best.user_attrs.get("refusals", 99)
    if refusals >= control_refusals:
        return None, (f"no trial below the KL target refused fewer held-out prompts than the stock model "
                      f"({control_refusals}): try more trials")
    return best.number, (f"trial {best.number}: {refusals}/{best.user_attrs.get('n_harmful')} held-out harmful prompts "
                         f"refused (stock {control_refusals}), first-token KL {best.user_attrs.get('kl', 0.0):.3f}")


def open_study(run, study_name: str):
    import optuna
    path = run.dir / "fit" / study_name / "study.sqlite"
    if not path.is_file():
        raise SystemExit(f"no study {study_name!r} in {run.dir / 'fit'}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{path.as_posix()}")
