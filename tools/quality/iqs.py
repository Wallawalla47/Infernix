"""IQS-2 decisions: the Infernix Quality Standard (docs/maintainer/model-quality.md, section 6).

Every shipped path is compared with its reference R(A, K): the artifact's declared design computed exactly
(the lossless route; the design reference where the artifact declares activation quantization or the KV
format is coded). infernix-certify writes one row per teacher-forced position (KL over the reference's top
256 tokens plus one bucket for the rest, top-1 flip, NLL change); this tool turns an arm's rows into
estimates, one-sided upper confidence bounds and verdicts.

Certification of an arm against R(A, K):
  C0              no non-finite value
  C1 mean KL      <= max(1.36e-3 nats, kappa N)   llama.cpp's Q8_0 (LLaMA-3-8B: 1.355e-3)
  C2 flip rate    <= max(2.33 %, kappa N)         the same Q8_0 run (top-1 agreement 97.674 %)
  C3 p99.9 KL     <= max(0.05 nats, kappa N)
  C4 clipped KL   <= kappa N                      the mean of min(KL, 0.1 nats): the typical position's error,
                                                  which the few near-tie positions that decide the mean cannot swing
N is the same statistic between R(A, K) and its noise twin (the exact route with stochastic rounding,
infernix-certify prefill --lossless --noise-seed): an ideal BF16 implementation's own deviation, a property of
the model; kappa = 2. Disclosed, not gated: perplexity (dNLL), the mean change in the written token's negative
log-likelihood (its exponential is the perplexity ratio), and every value per domain. Every arm is the shipped
default route; the reproducible route is no baseline (its bit-exactness and request invariance are tested bitwise).

Scored positions: a position counts when the model writes the token it predicts. A text window scores every
position; a chat window only its assistant turns (infernix-certify mask writes the artifact's list, the
`scored` field of a plan item). The system prompt, tool schemas, user turns and tool responses are text the
model is never trained to predict: its predictions there are arbitrary, extremely sharp and numerically
fragile, and (measured, first calibration run) held 99 % of every arm's KL while having no bearing on any
token the model emits. The estimate over every position is still reported beside the scored one.

Statistics: the unit is the window (a document); bounds are percentile bootstraps resampling windows
(paired: both runs saw the same positions), one-sided at 97.5 %. A set with fewer windows than min_units (the
32K long set: 4) resamples blocks of 4,096 positions of its windows instead. A criterion passes when its
bound is within budget. Budgets, kappa and the confidence are parameters (--set NAME=VALUE), so stored runs
are decided again under other thresholds without measuring.

Subcommands:
  decide ARM [--twin TWIN]          one arm (infernix-certify prefill/decode output prefix)
  disclose ARM --kv K               a KV format's (or a declared design's) design error against its class ceiling
  certificate PLAN --out DIR        every decision a run plan lists, plus its invariance and functional
                                    results: certificate.json and certificate.html

Exit status: 0 everything passes, 1 a criterion fails, 2 a measurement is unusable.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Policy:
    mean_kl: float = 1.36e-3
    flip_rate: float = 0.0233
    p999_kl: float = 0.05
    clip_kl: float = 0.1            # the clipped KL caps each position here (nats)
    kappa: float = 2.0
    confidence: float = 0.975
    resamples: int = 10000
    seed: int = 20261022
    min_units: int = 8
    block: int = 4096  # positions per resampling unit when a set has fewer windows than min_units


POLICY = Policy()
METRICS = ("mean_kl", "flip_rate", "p999_kl", "clipped_kl", "mean_dnll")  # every reported statistic
CERTIFIED = ("mean_kl", "flip_rate", "p999_kl", "clipped_kl")              # the gated ones
LABELS = {"mean_kl": "C1 mean KL", "flip_rate": "C2 flip rate", "p999_kl": "C3 p99.9 KL",
          "clipped_kl": "C4 clipped KL", "mean_dnll": "perplexity (dNLL)"}

# A KV format's design error ceiling (mean KL, flip rate): its class, from llama.cpp's LLaMA-3-8B weight
# quantizations (8-bit: Q8_0; mixed: Q6_K; 4-bit and VQ: between Q5_K_M and Q4_K_M).
KV_CLASSES = {
    "int8": ("8-bit", 1.36e-3, 0.0233),
    "fp8": ("8-bit", 1.36e-3, 0.0233),
    "k8v4": ("mixed", 5.45e-3, 0.0397),
    "nvfp4": ("4-bit and VQ", 1.5e-2, 0.04),
    "k4v2": ("4-bit and VQ", 1.5e-2, 0.04),
    "vq2": ("4-bit and VQ", 1.5e-2, 0.04),
}


class Unusable(RuntimeError):
    """The measurement cannot support a decision."""


def policy_with(overrides: list[str] | None) -> Policy:
    if not overrides:
        return POLICY
    changes = {}
    names = {f for f in Policy.__dataclass_fields__}
    for item in overrides:
        name, _, value = item.partition("=")
        if name not in names or not value:
            raise Unusable(f"--set {item}: expected NAME=VALUE with NAME one of {sorted(names)}")
        current = getattr(POLICY, name)
        changes[name] = type(current)(float(value)) if isinstance(current, (int, float)) else value
    return replace(POLICY, **changes)


@dataclass
class Arm:
    prefix: Path
    meta: dict
    window: np.ndarray
    kl: np.ndarray
    top_mass: np.ndarray
    flip: np.ndarray
    dnll: np.ndarray
    position: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    unit: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # the bootstrap's resampling unit
    domains: dict[int, str] = field(default_factory=dict)
    unscored: dict[str, float] | None = None  # the estimate over the positions the model does not write
    unscored_positions: int = 0

    @property
    def name(self) -> str:
        return self.prefix.name


def load_scored(path: Path) -> dict[int, np.ndarray]:
    """Per window index, a boolean array over positions (target token indices): true where the model writes the
    token (infernix-certify mask)."""
    if not path.is_file():
        raise Unusable(f"missing scored-position list {path}")
    out = {}
    for entry in json.loads(path.read_text(encoding="utf-8"))["windows"]:
        flags = np.zeros(int(entry["positions"]) + 1, dtype=bool)
        for first, last in entry["runs"]:
            flags[first:last + 1] = True
        out[int(entry["index"])] = flags
    return out


def resampling_units(window: np.ndarray, position: np.ndarray, policy: Policy) -> np.ndarray:
    """Windows are the resampling units; a set with too few of them resamples blocks of its windows."""
    if np.unique(window).size >= policy.min_units:
        return window
    return window * ((int(position.max()) - 1) // policy.block + 2) + (position - 1) // policy.block


def load_part(prefix: Path) -> tuple[dict, np.ndarray]:
    meta_path = Path(str(prefix) + ".json")
    rows_path = Path(str(prefix) + ".positions.csv")
    if not meta_path.is_file() or not rows_path.is_file():
        raise Unusable(f"missing arm {prefix} (.json and .positions.csv)")
    data = np.loadtxt(rows_path, delimiter=",", skiprows=1, ndmin=2)
    if data.size == 0:
        raise Unusable(f"{rows_path} holds no positions")
    return json.loads(meta_path.read_text(encoding="utf-8")), data


def load_arm(prefix: Path | Sequence[Path], scored: Path | None = None, policy: Policy = POLICY) -> Arm:
    """One arm, from one prefix or from parts (prefixes holding disjoint windows of the same run: a corpus layer
    scored later). The parts must agree on route and KV format."""
    parts = [Path(prefix)] if isinstance(prefix, (str, Path)) else [Path(p) for p in prefix]
    loaded = [load_part(p) for p in parts]
    meta = dict(loaded[0][0])
    for p, (m, _) in zip(parts[1:], loaded[1:]):
        if (m.get("route"), m.get("kv_dtype")) != (meta.get("route"), meta.get("kv_dtype")):
            raise Unusable(f"{p}: route {m.get('route')}/{m.get('kv_dtype')} differs from {parts[0]}")
    meta["nonfinite"] = sum(int(m.get("nonfinite", 0)) for m, _ in loaded)
    seen: set[int] = set()
    for p, (_, d) in zip(parts, loaded):
        windows = set(np.unique(d[:, 0]).astype(int).tolist())
        if seen & windows:
            raise Unusable(f"{p} repeats window(s) {sorted(seen & windows)} of an earlier part")
        seen |= windows
    data = np.concatenate([d for _, d in loaded])
    domains: dict[int, str] = {}
    for m, _ in loaded:
        references = m.get("reference", [])
        for reference in [references] if isinstance(references, str) else references:
            ref_meta = Path(str(reference) + ".json")
            if ref_meta.is_file():
                for entry in json.loads(ref_meta.read_text(encoding="utf-8")).get("windows", []):
                    domains[int(entry["index"])] = entry["domain"]
    prefix = parts[0]
    window, position = data[:, 0].astype(np.int64), data[:, 2].astype(np.int64)
    arm = Arm(prefix=prefix, meta=meta, window=window, kl=data[:, 3], top_mass=data[:, 4], flip=data[:, 5],
              dnll=data[:, 6], position=position, domains=domains)
    if scored is not None:
        flags = load_scored(scored)
        keep = np.zeros(len(window), dtype=bool)
        for w in np.unique(window):
            if int(w) not in flags:
                raise Unusable(f"{scored} lists no window {int(w)}")
            at = window == w
            table = flags[int(w)]
            inside = position[at] < len(table)
            keep[np.nonzero(at)[0][inside]] = table[position[at][inside]]
        if not keep.any():
            raise Unusable(f"{prefix}: no scored position")
        arm.unscored = point(arm.kl[~keep], arm.flip[~keep], arm.dnll[~keep]) if (~keep).any() else None
        arm.unscored_positions = int((~keep).sum())
        arm = replace(arm, window=window[keep], kl=arm.kl[keep], top_mass=arm.top_mass[keep], flip=arm.flip[keep],
                      dnll=arm.dnll[keep], position=position[keep])
    return replace(arm, unit=resampling_units(arm.window, arm.position, policy))


def point(kl: np.ndarray, flip: np.ndarray, dnll: np.ndarray, clip: float = POLICY.clip_kl) -> dict[str, float]:
    return {"mean_kl": float(np.mean(kl)), "flip_rate": float(np.mean(flip)),
            "p999_kl": float(np.quantile(kl, 0.999, method="higher")),
            "clipped_kl": float(np.mean(np.minimum(kl, clip))), "mean_dnll": float(np.mean(dnll))}


def resample(unit: np.ndarray, name: str, policy: Policy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bootstrap's draws: per resample, how often each unit (window) is drawn; and per position its unit."""
    units, inverse = np.unique(unit, return_inverse=True)
    n = len(units)
    if n < policy.min_units:
        raise Unusable(f"{name}: {n} resampling units; bounds need at least {policy.min_units}")
    rng = np.random.default_rng(policy.seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=policy.resamples).astype(np.float64)
    return inverse, weights, weights @ np.bincount(inverse, minlength=n).astype(np.float64)


def mean_bounds(values: dict[str, np.ndarray], inverse: np.ndarray, weights: np.ndarray, total: np.ndarray,
                policy: Policy) -> dict[str, float]:
    """One-sided upper bounds of position means: each resample weighs a unit's positions together."""
    n = weights.shape[1]
    return {m: float(np.quantile((weights @ np.bincount(inverse, weights=v, minlength=n)) / total, policy.confidence))
            for m, v in values.items()}


def bootstrap_bounds(arm: Arm, policy: Policy) -> dict[str, float]:
    """One-sided upper bounds: percentile bootstrap over windows (each resample draws windows with
    replacement; a window's positions keep their weight together)."""
    inverse, weights, total = resample(arm.unit, arm.name, policy)
    bounds = mean_bounds({"mean_kl": arm.kl, "flip_rate": arm.flip, "clipped_kl": np.minimum(arm.kl, policy.clip_kl),
                          "mean_dnll": arm.dnll}, inverse, weights, total, policy)
    bounds["p999_kl"] = float(np.quantile(weighted_upper_quantile(arm.kl, inverse, weights, total, 0.999),
                                          policy.confidence))
    return {m: bounds[m] for m in METRICS}


def weighted_upper_quantile(values: np.ndarray, window: np.ndarray, weights: np.ndarray, total: np.ndarray,
                            q: float, chunk: int = 1000) -> np.ndarray:
    """Per resample (a row of window weights), the smallest value whose cumulative weight in ascending order
    reaches q of the total: equivalently the value at the deepest descending rank j whose top-j weight is at most
    (1 - q) of the total. The top 1 % of values is enough unless a resample weights them nearly out; those
    resamples take the whole column."""
    order = np.argsort(-values, kind="stable")
    descending, owner = values[order], window[order]
    n_windows = weights.shape[1]
    m = min(len(values), max(64, math.ceil(0.01 * len(values))))
    head = np.zeros((m + 1, n_windows))
    head[np.arange(1, m + 1), owner[:m]] = 1.0
    head = np.cumsum(head, axis=0)  # head[j, w]: positions of window w among the top j
    out = np.empty(len(weights))
    limit = (1.0 - q) * total
    for start in range(0, len(weights), chunk):
        block = slice(start, start + chunk)
        tails = weights[block] @ head.T
        rank = (tails <= limit[block, None]).sum(axis=1) - 1
        out[block] = descending[np.minimum(rank, len(descending) - 1)]
        for b in np.nonzero(rank >= m)[0]:
            row = start + b
            full = np.concatenate(([0.0], np.cumsum(weights[row][owner])))
            deepest = np.searchsorted(full, limit[row], side="right") - 1
            out[row] = descending[min(deepest, len(descending) - 1)]
    return out


def budgets(policy: Policy, floor: dict[str, float] | None) -> dict[str, float]:
    """C1-C3: max(fixed, kappa N); C4 has no fixed budget, so without a floor it is not decided."""
    fixed = {m: getattr(policy, m) for m in ("mean_kl", "flip_rate", "p999_kl")}
    if floor is None:
        return fixed
    out = {m: max(fixed[m], policy.kappa * max(floor[m], 0.0)) for m in fixed}
    out["clipped_kl"] = policy.kappa * max(floor["clipped_kl"], 0.0)
    return out


def decide(arm: Arm, policy: Policy = POLICY, floor: dict[str, float] | None = None) -> dict:
    """Every criterion of one arm: estimate, bound, budget, verdict."""
    criteria = []
    counted   = int(np.sum(~(np.isfinite(arm.kl) & np.isfinite(arm.dnll))))
    nonfinite = max(int(arm.meta.get("nonfinite", 0)), counted)
    criteria.append({"name": "C0 finite", "estimate": nonfinite, "bound": nonfinite, "budget": 0, "passed": nonfinite == 0})
    finite = np.isfinite(arm.kl) & np.isfinite(arm.dnll)
    arm_f = replace(arm, window=arm.window[finite], kl=arm.kl[finite], top_mass=arm.top_mass[finite],
                    flip=arm.flip[finite], dnll=arm.dnll[finite], position=arm.position[finite],
                    unit=arm.unit[finite])
    estimate = point(arm_f.kl, arm_f.flip, arm_f.dnll, policy.clip_kl)
    bound = bootstrap_bounds(arm_f, policy)
    budget = budgets(policy, floor)
    for m in CERTIFIED:
        if m in budget:
            criteria.append({"name": LABELS[m], "estimate": estimate[m], "bound": bound[m], "budget": budget[m],
                             "passed": bound[m] <= budget[m]})
    # Disclosed: perplexity and the per-domain values.
    perplexity = {"mean_dnll": estimate["mean_dnll"], "bound": bound["mean_dnll"],
                  "ratio": math.exp(estimate["mean_dnll"]), "ratio_bound": math.exp(bound["mean_dnll"])}
    by_domain = {}
    if arm_f.domains:
        domain_of = np.array([arm_f.domains.get(int(w), "?") for w in arm_f.window])
        for domain in sorted(set(domain_of)):
            mask = domain_of == domain
            by_domain[domain] = point(arm_f.kl[mask], arm_f.flip[mask], arm_f.dnll[mask], policy.clip_kl)
    return {"arm": str(arm.prefix), "route": arm.meta.get("route"), "kv_dtype": arm.meta.get("kv_dtype"),
            "fast_paths": arm.meta.get("load", {}).get("fast_paths", []), "positions": int(arm_f.kl.size),
            "windows": int(np.unique(arm_f.window).size), "reference_top_mass": float(np.mean(arm_f.top_mass)),
            "noise_floor": floor, "estimate": estimate, "bound": bound, "budget": budget, "perplexity": perplexity,
            "domains": by_domain, "criteria": criteria, "passed": all(c["passed"] for c in criteria),
            "unscored": {"positions": arm.unscored_positions, "estimate": arm.unscored}}


def disclose(arm: Arm, kv: str | None, policy: Policy = POLICY) -> dict:
    """A design error against the lossless route: a KV format's exact codec against its class ceiling, or (kv None)
    an artifact's declared activation quantization, which has no ceiling and is reported only."""
    estimate = point(arm.kl, arm.flip, arm.dnll)
    bound = bootstrap_bounds(arm, policy)
    if kv is None:
        return {"arm": str(arm.prefix), "kv_dtype": None, "class": "declared activation quantization (reported)",
                "estimate": estimate, "bound": bound, "criteria": [], "passed": True,
                "unscored": {"positions": arm.unscored_positions, "estimate": arm.unscored}}
    if kv not in KV_CLASSES:
        raise Unusable(f"no KV class for {kv}")
    name, kl_ceiling, flip_ceiling = KV_CLASSES[kv]
    criteria = [{"name": "class mean KL", "estimate": estimate["mean_kl"], "bound": bound["mean_kl"],
                 "budget": kl_ceiling, "passed": bound["mean_kl"] <= kl_ceiling},
                {"name": "class flip rate", "estimate": estimate["flip_rate"], "bound": bound["flip_rate"],
                 "budget": flip_ceiling, "passed": bound["flip_rate"] <= flip_ceiling}]
    return {"arm": str(arm.prefix), "kv_dtype": kv, "class": name, "estimate": estimate, "bound": bound,
            "criteria": criteria, "passed": all(c["passed"] for c in criteria),
            "unscored": {"positions": arm.unscored_positions, "estimate": arm.unscored}}


def floor_of(twin: Path | Sequence[Path] | None, scored: Path | None = None,
             policy: Policy = POLICY) -> dict[str, float] | None:
    if twin is None:
        return None
    arm = load_arm(twin, scored)
    if arm.meta.get("route") != "noise_twin":
        raise Unusable(f"{twin} is not a noise-twin run")
    return point(arm.kl, arm.flip, arm.dnll, policy.clip_kl)


def print_decision(decision: dict) -> None:
    print(f"{decision['arm']}  route={decision.get('route')}  kv={decision.get('kv_dtype')}  "
          f"positions={decision.get('positions', '-')}  windows={decision.get('windows', '-')}")
    print(f"  {'criterion':<34}{'estimate':>12}{'bound':>12}{'budget':>12}  result")
    for c in decision["criteria"]:
        print(f"  {c['name']:<34}{c['estimate']:>12.4g}{c['bound']:>12.4g}{c['budget']:>12.4g}  "
              f"{'PASS' if c['passed'] else 'FAIL'}")
    print(f"  => {'PASS' if decision['passed'] else 'FAIL'}")


def render_html(certificate: dict) -> str:
    rows = []
    for d in certificate["decisions"]:
        worst = max((c for c in d["criteria"] if c["budget"]), key=lambda c: c["bound"] / c["budget"], default=None)
        rows.append(
            f"<tr><td>{html.escape(d.get('label', d['arm']))}</td><td>{html.escape(str(d.get('route')))}</td>"
            f"<td>{html.escape(str(d.get('kv_dtype')))}</td>"
            + "".join(f"<td class=n>{d['bound'][m]:.3g} / {d['budget'][m]:.3g}</td>" if m in d["budget"]
                      else "<td class=n>-</td>" for m in CERTIFIED)
            + f"<td class=n>{d['perplexity']['ratio']:.5f} (&le; {d['perplexity']['ratio_bound']:.5f})</td>"
            + f"<td>{html.escape(worst['name']) if worst else ''}</td>"
            f"<td class={'p' if d['passed'] else 'f'}>{'PASS' if d['passed'] else 'FAIL'}</td></tr>")
    disclosures = "".join(
        f"<tr><td>{html.escape(d.get('label', d['arm']))}</td><td>{html.escape(d['class'])}</td>"
        f"<td class=n>{d['bound']['mean_kl']:.3g}</td><td class=n>{d['bound']['flip_rate']:.3%}</td>"
        f"<td class={'p' if d['passed'] else 'f'}>{'PASS' if d['passed'] else 'FAIL'}</td></tr>"
        for d in certificate.get("disclosures", []))
    checks = "".join(
        f"<tr><td>{html.escape(c['name'])}</td><td>{html.escape(c.get('detail', ''))}</td>"
        f"<td class={'p' if c['passed'] else 'f'}>{'PASS' if c['passed'] else 'FAIL'}</td></tr>"
        for c in certificate.get("checks", []))
    head = "".join(f"<th>{LABELS[m]} bound / budget</th>" for m in CERTIFIED) + "<th>Perplexity ratio</th>"
    verdict = "PASS" if certificate["passed"] else "FAIL"
    return f"""<!doctype html><meta charset=utf-8><title>IQS-2 certificate</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1200px}}table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}.n{{text-align:right;font-family:monospace}}
.p{{color:#1e6b3a;font-weight:600}}.f{{color:#a3271f;font-weight:600}}</style>
<h1>IQS-2 certificate: {verdict}</h1>
<pre>{html.escape(json.dumps(certificate.get('provenance', {}), indent=2))}</pre>
<h2>Path fidelity</h2><table><tr><th>Path</th><th>Route</th><th>KV</th>{head}<th>Nearest budget</th><th>Verdict</th></tr>
{''.join(rows)}</table>
<h2>Design error disclosures</h2><table><tr><th>Format or design</th><th>Class</th><th>Mean KL bound</th><th>Flip bound</th><th>Verdict</th></tr>{disclosures}</table>
<h2>Exactness, invariance and end-to-end checks</h2><table><tr><th>Check</th><th>Detail</th><th>Verdict</th></tr>{checks}</table>
"""


def parts(base: Path, value: str | list[str]) -> list[Path]:
    """A plan entry's arm or twin: one prefix, or a list of parts (relative to the plan)."""
    return [base / v for v in ([value] if isinstance(value, str) else value)]


def run_certificate(plan_path: Path, out: Path, policy: Policy) -> int:
    """A run plan: {"provenance": {...}, "decisions": [{"label", "arm", "twin" (optional)}],
    "disclosures": [{"label", "arm", "kv" (absent: a declared activation quantization)}],
    "checks": [{"name", "result": path to a JSON with "passed" and optionally "detail"}]}; paths relative to
    the plan. An arm or twin is one prefix or a list of parts holding disjoint windows."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base = plan_path.parent
    certificate = {"provenance": plan.get("provenance", {}), "policy": policy.__dict__, "decisions": [],
                   "disclosures": [], "checks": [], "unusable": []}
    for item in plan.get("decisions", []):
        try:
            scored = base / item["scored"] if item.get("scored") else None
            # the noise twin always ran on the main set, whatever set the arm ran on
            twin_scored = base / item["twin_scored"] if item.get("twin_scored") else scored
            floor = floor_of(parts(base, item["twin"]), twin_scored, policy) if item.get("twin") else None
            decision = decide(load_arm(parts(base, item["arm"]), scored, policy), policy, floor)
            decision["label"] = item.get("label", item["arm"])
            certificate["decisions"].append(decision)
        except Unusable as error:
            certificate["unusable"].append({"label": item.get("label", item["arm"]), "error": str(error)})
    for item in plan.get("disclosures", []):
        try:
            scored = base / item["scored"] if item.get("scored") else None
            disclosure = disclose(load_arm(parts(base, item["arm"]), scored, policy), item.get("kv"), policy)
            disclosure["label"] = item.get("label", item["arm"])
            certificate["disclosures"].append(disclosure)
        except Unusable as error:
            certificate["unusable"].append({"label": item.get("label", item["arm"]), "error": str(error)})
    for item in plan.get("checks", []):
        path = base / item["result"]
        if not path.is_file():
            certificate["checks"].append({"name": item["name"], "detail": f"missing {path}", "passed": False})
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        certificate["checks"].append({"name": item["name"], "detail": str(result.get("detail", "")),
                                      "passed": bool(result.get("passed"))})
    certificate["passed"] = (not certificate["unusable"] and all(d["passed"] for d in certificate["decisions"])
                             and all(d["passed"] for d in certificate["disclosures"])
                             and all(c["passed"] for c in certificate["checks"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "certificate.json").write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    (out / "certificate.html").write_text(render_html(certificate), encoding="utf-8")
    for decision in certificate["decisions"]:
        print_decision(decision)
    for item in certificate["unusable"]:
        print(f"UNUSABLE {item['label']}: {item['error']}")
    print(f"certificate: {'PASS' if certificate['passed'] else 'FAIL'} ({out / 'certificate.html'})")
    if certificate["unusable"]:
        return 2
    return 0 if certificate["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("decide", help="decide one arm")
    one.add_argument("arm", type=Path)
    one.add_argument("--twin", type=Path, help="the artifact's noise-twin arm (its noise floor)")
    disc = commands.add_parser("disclose", help="a KV format's design error against its class")
    disc.add_argument("arm", type=Path)
    disc.add_argument("--kv", help="the KV format (absent: a declared activation quantization, reported only)")
    cert = commands.add_parser("certificate", help="every decision of a run plan")
    cert.add_argument("plan", type=Path)
    cert.add_argument("--out", type=Path, required=True)
    for sub in (one, disc):
        sub.add_argument("--scored", type=Path, help="the artifact's scored-position list (infernix-certify mask)")
    for sub in (one, disc, cert):
        sub.add_argument("--set", action="append", metavar="NAME=VALUE", help="a Policy value (budgets, kappa, ...)")
        sub.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        policy = policy_with(args.set)
        if args.command == "certificate":
            return run_certificate(args.plan, args.out, policy)
        if args.command == "decide":
            result = decide(load_arm(args.arm, args.scored, policy), policy, floor_of(args.twin, args.scored, policy))
            print_decision(result)
        else:
            result = disclose(load_arm(args.arm, args.scored, policy), args.kv, policy)
            print_decision(result)
        if args.json:
            args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0 if result["passed"] else 1
    except Unusable as error:
        print(f"UNUSABLE: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
