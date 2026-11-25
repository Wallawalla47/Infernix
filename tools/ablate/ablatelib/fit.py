"""The fit: an Optuna search for the cut that removes refusals while moving the model least.

Each trial (heretic 1.4's search):
  * samples heretic's direction scope (one interpolated row at every layer, or each layer's own row) and a band per
    lane (attention/DeltaNet output and MoE output);
  * starts a generation engine with that sidecar and counts refusals on the held-out harmful prompts;
  * starts a scoring engine with it and measures first-token KL against the control on the held-out harmless prompts.

Objectives (heretic 1.4's, a two-objective study, both minimized):
  refusal score  the trial's refusal rate on the held-out harmful prompts over the control's
  KL score       first-token KL / kl_scale; below kl_floor (0.01) the refusal score x kl_floor / kl_scale instead, so
                 the sampler does not spend its trials on cuts that change nothing
TPE models both, so the search fills the front (for each number of refusals, the lowest KL it found) instead of
stopping at the first cut under a KL bound.

The pick (pick()): among completed trials whose KL is below kl_max (0.4) and that refuse fewer held-out prompts than
the stock model, the lowest refusal score + KL / kl_scale. That is heretic's balance: a KL of kl_scale (1.0) weighs
as much as every refusal the stock model makes. `list` shows the front; `choose` keeps another trial instead.

A study is resumable (a sqlite ledger per study): rerunning `fit` with the same name continues it. Its identity is
mode : lanes : direction : objectives : graded prompts, and a run whose identity differs is refused, because mixing
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


KL_FLOOR = 0.01   # heretic 1.4's kl_divergence_target
KL_SCALE = 1.0    # heretic 1.4's kl_divergence_scale
KL_MAX = 0.4      # the pick's ceiling


def refusal_rate(refusals: int, truncated: int, n: int) -> float:
    """Refusals over the answers that ended (an answer cut off at the token limit is not graded)."""
    return refusals / max(1, n - truncated)


def objectives(refusals: int, truncated: int, n: int, control: dict, kl: float, kl_floor: float = KL_FLOOR,
               kl_scale: float = KL_SCALE) -> tuple[float, float]:
    """(KL score, refusal score): heretic 1.4's two objectives (evaluator.get_score)."""
    rate_c = refusal_rate(control["refusals"], control["truncated"], n)
    rate_t = refusal_rate(refusals, truncated, n)
    refusal_score = rate_t / rate_c if rate_c else rate_t
    kl_score = kl / kl_scale if kl >= kl_floor else refusal_score * kl_floor / kl_scale
    return kl_score, refusal_score


def fit(run, study_name: str, trials: int, *, kl_floor: float = KL_FLOOR, kl_scale: float = KL_SCALE,
        startup: int = 60, markers: tuple[str, ...] = M.MARKERS, max_failures: int = 3, log=print,
        progress=None) -> dict:
    """Adds `trials` completed trials to the study (a failed trial is replaced). `progress`, when given, is called with
    {'control_refusals', 'n_harmful'} once the control is measured and with each finished trial's record."""
    import optuna

    if trials < 1:
        raise SystemExit("--trials must be at least 1")
    if not (kl_floor >= 0 and kl_scale > 0):
        raise SystemExit("--kl-floor must be at least 0 and --kl-scale above 0")
    base = run.dir / "direction" / "sidecar.json"
    if not base.is_file():
        raise SystemExit(f"{base} is missing: run `ablate.py direction` first")
    rows = SC.rows_of(base)
    n_layers = rows.shape[0]
    sets = load_sets(run)
    harm, good = sets["held_out"]["harmful"], sets["held_out"]["harmless"]
    d_sha = hashlib.sha256((base.parent / "d.f32.bin").read_bytes()).hexdigest()
    rule = f"pareto,floor={kl_floor:g},scale={kl_scale:g},eval={eval_id(harm, good)}"
    shape = f"band+scope:attn+mlp:{d_sha[:12]}:{rule}"
    space = SC.search_space(n_layers)

    study_dir = run.sub("fit", study_name)
    storage = f"sqlite:///{(study_dir / 'study.sqlite').as_posix()}"
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if (study_dir / "study.sqlite").is_file():
        # Read the ledger's identity before opening it as a two-objective study (an older one-objective study cannot
        # be opened that way, and must be refused by name rather than by Optuna's error).
        try:
            previous = optuna.load_study(study_name=study_name, storage=storage).user_attrs.get("shape")
        except KeyError:
            previous = None
        if previous is not None and previous != shape:
            raise SystemExit(f"study {study_name!r} was fitted as {previous}; this run is {shape} "
                             "(mode:lanes:direction:objectives). Use a new study name.")
    sampler = optuna.samplers.TPESampler(n_startup_trials=startup, n_ei_candidates=128, multivariate=True)
    study = optuna.create_study(study_name=study_name, storage=storage, directions=["minimize", "minimize"],
                                load_if_exists=True, sampler=sampler)
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

    def run_trial(trial) -> tuple[float, float]:
        started = time.time()
        tdir = study_dir / f"t{trial.number}"
        scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
        index = trial.suggest_float("direction_index", *space["direction_index"])
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
        kl_score, refusal_score = objectives(graded["refusals"], graded["truncated"], len(harm), control, kl,
                                             kl_floor, kl_scale)
        record = {"refusals": graded["refusals"], "truncated": graded["truncated"], "kl": kl, "n_harmful": len(harm),
                  "control_refusals": control["refusals"], "refusal_score": refusal_score, "kl_score": kl_score,
                  "scope": scope, "direction_index": round(index, 4), "sidecar": str(cand), "what": what,
                  "kl_per_prompt": [round(x, 5) for x in kls], "seconds": round(time.time() - started, 1)}
        for k, v in record.items():
            if k != "kl_per_prompt":
                trial.set_user_attr(k, v)
        write_json(tdir / "trial.json", record)
        log(f"trial {trial.number:>4}  refusals {graded['refusals']:>2}/{len(harm)} (control {control['refusals']})  "
            f"KL {kl:.4f}  {what}  {time.time() - started:.0f}s")
        if progress is not None:
            progress({"trial": trial.number, **{k: v for k, v in record.items() if k != "kl_per_prompt"}})
        return kl_score, refusal_score

    def objective(trial) -> tuple[float, float]:
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


def records(study) -> list[dict]:
    """The completed trials' results: {trial, refusals, n, kl, refusal_score, what, sidecar}."""
    out = []
    for t in study.trials:
        a = t.user_attrs
        if t.state.name != "COMPLETE" or "kl" not in a or "refusal_score" not in a:
            continue
        out.append({"trial": t.number, "refusals": a["refusals"], "n": a.get("n_harmful"), "kl": float(a["kl"]),
                    "refusal_score": float(a["refusal_score"]), "what": a.get("what"), "sidecar": a.get("sidecar")})
    return out


def pareto_front(rows: list[dict]) -> list[dict]:
    """The trials no other trial beats on both refusals and KL, fewest refusals first: each has a lower KL than every
    trial with as few refusals or fewer (heretic 1.4's selection list)."""
    front, best_kl = [], float("inf")
    for r in sorted(rows, key=lambda r: (r["refusals"], r["kl"], r["trial"])):
        if r["kl"] < best_kl:
            front.append(r)
            best_kl = r["kl"]
    return front


def pick_score(row: dict, kl_scale: float = KL_SCALE) -> float:
    """The pick's balance: the refusal score plus KL / kl_scale."""
    return row["refusal_score"] + row["kl"] / kl_scale


def pick(rows: list[dict], control_refusals: int, kl_max: float = KL_MAX,
         kl_scale: float = KL_SCALE) -> tuple[dict | None, str]:
    """(row, why) of the cut to keep: among trials whose KL is below kl_max and that refuse fewer held-out prompts
    than the stock model, the lowest pick_score (ties: fewer refusals, then lower KL). None, with the reason, when
    the stock model refuses none of the held-out prompts or no trial qualifies."""
    if control_refusals == 0:
        return None, "the stock model refused none of the held-out harmful prompts: there is no refusal to remove"
    below = [r for r in rows if r["kl"] < kl_max]
    if not below:
        return None, f"no trial kept the first-token KL below {kl_max:g}: try more trials"
    better = [r for r in below if r["refusals"] < control_refusals]
    if not better:
        return None, (f"no trial with KL below {kl_max:g} refused fewer held-out prompts than the stock model "
                      f"({control_refusals}): try more trials")
    best = min(better, key=lambda r: (pick_score(r, kl_scale), r["refusals"], r["kl"], r["trial"]))
    return best, describe(best, control_refusals)


def describe(row: dict, control_refusals: int) -> str:
    return (f"trial {row['trial']}: {row['refusals']}/{row['n']} held-out harmful prompts refused "
            f"(stock {control_refusals}), first-token KL {row['kl']:.3f}")


def front(study) -> list[dict]:
    """The study's Pareto front (see pareto_front), KL rounded for display."""
    return [{**r, "kl": round(r["kl"], 4)} for r in pareto_front(records(study))]


def open_study(run, study_name: str):
    import optuna
    path = run.dir / "fit" / study_name / "study.sqlite"
    if not path.is_file():
        raise SystemExit(f"no study {study_name!r} in {run.dir / 'fit'}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{path.as_posix()}")
