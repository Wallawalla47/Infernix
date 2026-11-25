"""validate: the fit's best trials graded on prompts the search never scored, and the pick made on those numbers.

The fit scores every trial on the same held-out prompts (40 by default), so the search fits them: rows whose fit scores
are a few hundredths apart can differ several times over on other prompts. On Inkling 2 Mini (2026-11-25, 300 trials)
the pick by the fit's numbers won by 0.443 against 0.449, at 2/40 against 5/40 held-out refusals; on JBB-Behaviors'
91 harmful prompts the two refused 3 and 23. A few held-out prompts most trials still refuse decide most of the fit's
refusal count, so the count cannot tell a cut that removes the refusal from one that removes the easy cases.

candidates() takes the `count` best trials by the fit's pick score among those the rule allows (KL below kl_max, fewer
held-out refusals than the stock model), on the front or not: a trial one held-out refusal behind a front row is no
worse a candidate, given how little those counts separate, and the front's far end is no candidate at all (a local
search around Inkling's pick: the front's rows 3-6 refused 6-21 of 40 held out, 11-42 of 87 validation prompts, while
four trials off the front refused 1-4 of 40). validate() grades them, with the stock model, on a validation
pair (harmful prompts for refusals, benign ones for over-refusals and KL) with candidates.grade; pick() applies the
rule to the validation numbers: the lowest refusal rate over the stock model's plus the KL where the stock model
answered / kl_scale, among rows that refuse fewer validation prompts than the stock model. The test pair is then
graded once, on the kept trial, so its numbers stay out of sample.

Results are kept in fit/<study>/validation/<key>.json, the key a hash of the validation lists, the refusal phrases
and the setup (capture.identity), so a rerun grades only the rows it lacks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import measure as M
from .run import write_json

ROWS = 6   # trials graded on the validation pair


def candidates(rows: list[dict], control_refusals: int, kl_max: float, kl_scale: float, count: int = ROWS) -> list[dict]:
    """The trials the rule allows, best fit pick score first, at most `count`."""
    from . import fit as F
    allowed = [r for r in rows if r["kl"] < kl_max and r["refusals"] < control_refusals]
    return sorted(allowed, key=lambda r: (F.pick_score(r, kl_scale), r["refusals"], r["kl"], r["trial"]))[:count]


def lists_key(run, harmful: Path, benign: Path | None, markers: tuple[str, ...]) -> str:
    """What decides a grade besides the cut: the lists' bytes, the refusal phrases and the setup (capture.identity).
    A grade kept under this key (validation here, the test pair's in auto.keep) is only reused for the same."""
    from . import capture as C
    h = hashlib.sha256()
    for part in (Path(harmful).read_bytes(), Path(benign).read_bytes() if benign else b"",
                 json.dumps(list(markers)).encode(), json.dumps(C.identity(run), sort_keys=True).encode()):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:16]


def _arm(summary_arm: dict) -> dict:
    h, b = summary_arm["harmful"], summary_arm["benign"]
    return {"refusals": h["refusals"], "n": h["n"] - h["errors"] - h["truncated"],
            "benign_refusals": b["refusals"], "benign_n": b["n"] - b["errors"] - b["truncated"],
            "kl": summary_arm.get("kl"), "kl_answered": summary_arm.get("kl_stock_answered")}


def validate(run, study: str, rows: list[dict], harmful: Path, benign: Path, field: str | None,
             markers: tuple[str, ...], log=print) -> dict:
    """{'key', 'harmful', 'benign', 'stock': arm, 'trials': {trial: arm}} for every row (graded now or kept from an
    earlier run). An arm: {'refusals', 'n' (graded), 'benign_refusals', 'benign_n', 'kl', 'kl_answered'}."""
    from . import candidates as CD
    folder = run.dir / "fit" / study / "validation"
    key = lists_key(run, harmful, benign, markers)
    path = folder / f"{key}.json"
    kept = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "key": key, "harmful": str(harmful), "benign": str(benign), "stock": None, "trials": {}}
    todo = [r for r in rows if str(r["trial"]) not in kept["trials"]]
    if todo:
        sidecars = [str(run.dir / "fit" / study / f"t{r['trial']}" / "sidecar.json") for r in todo]
        log(f"validating {len(todo)} trial(s) on {Path(harmful).name} + {Path(benign).name}: "
            f"trial(s) {', '.join(str(r['trial']) for r in todo)}")
        graded = CD.grade(run, Path(harmful), Path(benign), sidecars, field, markers, log=log)
        summary = json.loads((graded / "summary.json").read_text(encoding="utf-8"))
        arms = {a["arm"]: a for a in summary["arms"]}
        kept["stock"] = _arm(arms["stock"])
        for r in todo:
            kept["trials"][str(r["trial"])] = {**_arm(arms[f"t{r['trial']}"]), "grade": str(graded)}
        write_json(path, kept)
    return kept


def score(arm: dict, stock: dict, kl_scale: float) -> float:
    """The rule on validation numbers: refusal rate over the stock model's, plus KL where the stock model answered
    (all benign prompts when it answered none) / kl_scale."""
    rate = arm["refusals"] / max(1, arm["n"])
    rate_stock = stock["refusals"] / max(1, stock["n"])
    kl = arm["kl_answered"] if arm["kl_answered"] is not None else arm["kl"]
    return (rate / rate_stock if rate_stock else rate) + kl / kl_scale


def pick(rows: list[dict], kept: dict, kl_scale: float) -> tuple[dict | None, str]:
    """(row, why): among the validated rows that refuse fewer validation prompts than the stock model, the lowest
    score (ties: fewer validation refusals, then lower KL). None, with the reason, when none qualifies."""
    stock = kept["stock"]
    if not stock or stock["refusals"] == 0:
        return None, "the stock model refused none of the validation prompts: they cannot tell the cuts apart"
    graded = [(r, kept["trials"][str(r["trial"])]) for r in rows if str(r["trial"]) in kept["trials"]]
    better = [(r, a) for r, a in graded if a["refusals"] / max(1, a["n"]) < stock["refusals"] / max(1, stock["n"])]
    if not better:
        return None, (f"no validated row refused fewer validation prompts than the stock model "
                      f"({stock['refusals']}/{stock['n']})")
    row, arm = min(better, key=lambda ra: (score(ra[1], stock, kl_scale), ra[1]["refusals"], ra[1]["kl_answered"] or 0,
                                           ra[0]["trial"]))
    return row, describe(row, arm, stock)


def describe(row: dict, arm: dict, stock: dict) -> str:
    kl = arm["kl_answered"] if arm["kl_answered"] is not None else arm["kl"]
    return (f"trial {row['trial']}: {arm['refusals']}/{arm['n']} validation harmful prompts refused (stock "
            f"{stock['refusals']}/{stock['n']}), KL {kl:.3f} where the stock model answered; in the fit "
            f"{row['refusals']}/{row['n']} held out, KL {row['kl']:.3f}")


def shown(arm: dict) -> dict:
    """A validated row's numbers as progress.json shows them."""
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in arm.items() if k != "grade"}


__all__ = ["ROWS", "candidates", "validate", "score", "pick", "describe", "shown", "M"]
