#!/usr/bin/env python3
"""Analyze an agentic A/B run directory and write report.md + summary.json into it.

Usage: python analyze.py <run_dir>
       python analyze.py --aggregate <out_dir> <run_dir> <run_dir> [...]

Each arm directory (control, treatment and optionally alt) holds the client's own record of
every request (client.jsonl, with the per-request seed and the workload class) and the serve's
request log (request_log.jsonl). They are joined on the request seed, which is identical in
every arm, so every metric compares the same logical requests. All timings and token counts
come from the serve's request log; every arm is compared with the control.
"""
import json
import os
import random
import re
import statistics
import sys

# Every arm is compared with the control; `alt` is AB_ALT_EXE (default the treatment build) with
# AB_ALT_EXTRA_FLAGS; AB_CONTROL_LABEL, AB_TREATMENT_LABEL and AB_ALT_LABEL rename the arms.
ARMS = ("control", "treatment", "alt")
LABEL = {"control": "NInfer + Windows port", "treatment": "Infernix", "alt": "Infernix, alternative"}
SHORT = {"control": "NInfer", "treatment": "Infernix", "alt": "Infernix alt"}


def use_labels(cfg):
    """Arm names the run set (AB_CONTROL_LABEL, AB_TREATMENT_LABEL, AB_ALT_LABEL)."""
    for arm in ARMS:
        if cfg.get(arm + "_label"):
            LABEL[arm] = cfg[arm + "_label"]
            SHORT[arm] = cfg[arm + "_label"]


CONTINUING = {"loop", "after_idle", "history_edit", "retry", "abort_retry", "subagent_loop"}
NEW_LONG = {"cold_resume", "compaction", "check"}
CLASS_DOC = [
    ("cold_resume", "a saved session resumed after a restart (whole prompt is new)"),
    ("loop", "an agent tool-loop turn continuing its own conversation"),
    ("subagent_first", "a subagent's first turn (shares its system+tools prefix with siblings)"),
    ("subagent_loop", "a subagent tool-loop turn"),
    ("compaction", "the whole conversation sent for summarisation (new prompt)"),
    ("restart", "the fresh context after compaction (system prompt + summary)"),
    ("check", "a loop-detection side call over the whole history (new prompt)"),
    ("after_idle", "a session resumed after sitting idle while other sessions ran"),
    ("history_edit", "the first turn after the client cleared old tool results"),
    ("retry", "the user retrying the previous turn with the identical prompt"),
    ("abort_retry", "the retry after the client aborted a long request"),
]


def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except ValueError:
                        pass
    return rows


def load_arm(run_dir, arm):
    d = os.path.join(run_dir, arm)
    client = load_jsonl(os.path.join(d, "client.jsonl"))
    server = load_jsonl(os.path.join(d, "request_log.jsonl"))
    meta = {}
    if os.path.exists(os.path.join(d, "arm.json")):
        with open(os.path.join(d, "arm.json"), encoding="utf-8") as f:
            meta = json.load(f)
    done, errs, start, thr = {}, {}, None, []
    for o in server:
        ev = o.get("event")
        seed = ((o.get("request") or {}).get("sampling") or {}).get("seed")
        if ev == "server_start":
            start = o
        elif ev == "request_done" and seed is not None:
            done[seed] = o
        elif ev in ("request_error", "request_rejected") and seed is not None:
            errs[seed] = o
        elif ev == "throughput":
            thr.append(o)
    reqs, events = [], []
    for c in client:
        if "event" in c:
            events.append(c)
            continue
        s = done.get(c["seed"])
        r = dict(c)
        r["server"] = s
        if s:
            res, t = s.get("result") or {}, s.get("timings_seconds") or {}
            spec = s.get("speculative") or {}
            r.update({
                "prompt": res.get("prompt_tokens") or 0,
                "hit": res.get("prefix_cache_hit_tokens") or 0,
                "computed": res.get("computed_prefill_tokens") or 0,
                "completion": res.get("completion_tokens") or 0,
                "thinking": res.get("model_thinking_tokens") or 0,
                "path": res.get("prefix_reuse_path") or "unknown",
                "finish_server": res.get("finish_reason"),
                # Every time is the client's: engines place their own timers differently (NInfer's
                # prefill timer stopped ~8 ms before Infernix's does), so only the client clock
                # measures every engine the same way. Token counts and cache hits are the server's.
                "ttft": (c["t_first"] - c["t_send"]) if c.get("t_first") and c.get("t_send") else None,
                "stream_s": (c["t_done"] - c["t_first"]) if c.get("t_first") and c.get("t_done") else None,
                "spec_accepted": spec.get("accepted_tokens") or 0,
                "spec_drafted": spec.get("drafted_tokens") or 0,
                "ngram_accepted": spec.get("ngram_accepted_tokens") or 0,
                "ngram_drafted": spec.get("ngram_drafted_tokens") or 0,
                "rounds": spec.get("rounds") or 0,
            })
        reqs.append(r)
    build = "?"
    serve_log = os.path.join(d, "serve.log")
    if os.path.exists(serve_log):
        with open(serve_log, encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = re.search(r"build ([0-9a-f]{7,40})", ln)
                if m:
                    build = m.group(1)
                    break
    return {"arm": arm, "requests": reqs, "events": events, "start": start,
            "throughput": thr, "meta": meta, "errors": errs, "build": build}


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def median(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def pct(v, p):
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_MIN_BLOCKS = 5   # fewer blocks than this give no meaningful interval


def in_flight(A):
    """Every request's (t_send, t_done) on the client clock, aborted ones included: they occupied the server."""
    return [(x["t_send"], x["t_done"], x["seed"]) for x in A["requests"] if x.get("t_send") and x.get("t_done")]


def alone_while(spans, seed, t0, t1):
    """No other request was in flight anywhere in [t0, t1]."""
    return not any(s != seed and a0 < t1 and a1 > t0 for a0, a1, s in spans)


def stream_rate(rows):
    """Output tok/s of client streams: tokens after the first / seconds from the first token to the last,
    a ratio of sums with a 95 % bootstrap interval over the requests; with it decode rounds/s (the server's
    round count, not a timer) and tokens per round."""
    if not rows:
        return {"n": 0, "seconds": 0.0, "tps": {"value": None, "lo": None, "hi": None},
                "rounds_per_s": {"value": None, "lo": None, "hi": None},
                "tokens_per_row_round": {"value": None, "lo": None, "hi": None}}
    parts = [((x["completion"] - 1), x["stream_s"], x["rounds"], x["completion"]) for x in rows]
    ratios = {"tps": (0, 1), "rounds_per_s": (2, 1), "tokens_per_row_round": (3, 2)}
    samples = {k: [] for k in ratios}
    rng = random.Random(0)
    for _ in range(BOOTSTRAP_RESAMPLES if len(parts) >= BOOTSTRAP_MIN_BLOCKS else 0):
        drawn = [parts[rng.randrange(len(parts))] for _ in parts]
        for k, (num, den) in ratios.items():
            den_sum = sum(p[den] for p in drawn)
            if den_sum:
                samples[k].append(sum(p[num] for p in drawn) / den_sum)
    out = {"n": len(parts), "seconds": sum(p[1] for p in parts)}
    for k, (num, den) in ratios.items():
        total = sum(p[den] for p in parts)
        out[k] = {"value": sum(p[num] for p in parts) / total if total else None,
                  "lo": pct(samples[k], 0.025), "hi": pct(samples[k], 0.975)}
    return out


def decode_metrics(A, rows):
    """Decode rates from the client clock.

    `one`: requests that streamed with no other request in flight from their first token to their last, so
    batching and other requests' prefill cannot move them. `all`: every streamed token over the time at
    least one request was streaming, at the batching the run produced."""
    spans = in_flight(A)
    streams = [x for x in rows if x.get("stream_s") and x["stream_s"] > 0 and x["completion"] > 1]
    one = [x for x in streams if alone_while(spans, x["seed"], x["t_first"], x["t_done"])]
    union, cur = 0.0, None
    for t0, t1 in sorted((x["t_first"], x["t_done"]) for x in streams):
        if cur and t0 <= cur[1]:
            cur[1] = max(cur[1], t1)
        else:
            if cur:
                union += cur[1] - cur[0]
            cur = [t0, t1]
    if cur:
        union += cur[1] - cur[0]
    tokens = sum(x["completion"] - 1 for x in streams)
    return {"one": stream_rate(one),
            "all": {"tps": {"value": tokens / union if union else None, "lo": None, "hi": None},
                    "seconds": union, "n": len(streams)}}


REQUEST_BLOCK = 10         # consecutive requests resampled together for request-metric intervals


def request_intervals(rows):
    """95 % block-bootstrap intervals for the TTFT and cache rows of one run.

    Requests are taken in send order and resampled in blocks of REQUEST_BLOCK: nearby requests
    share queueing and cache state, so a block keeps them together. The interval shows how much
    a metric moves with which stretches of the run it happened to contain; it cannot show how
    differently another run would interleave, which is what repeated seeds are for."""
    rows = sorted(rows, key=lambda r: r.get("t_send") or 0)
    blocks = [rows[i:i + REQUEST_BLOCK] for i in range(0, len(rows), REQUEST_BLOCK)]
    if len(blocks) < BOOTSTRAP_MIN_BLOCKS:
        return {}
    stats = {
        "ttft_mean": lambda rs: mean([r["ttft"] for r in rs]),
        "ttft_median": lambda rs: median([r["ttft"] for r in rs]),
        "ttft_p90": lambda rs: pct([r["ttft"] for r in rs], 0.9),
        "ttft_cont_mean": lambda rs: mean([r["ttft"] for r in rs if r["cls"] in CONTINUING]),
        "hit_rate": lambda rs: (sum(r["hit"] for r in rs) / sum(r["prompt"] for r in rs)
                                if sum(r["prompt"] for r in rs) else None),
        "computed_tokens": lambda rs: sum(r["computed"] for r in rs),
        "cont_full_prefill_main": lambda rs: sum(
            1 for r in rs if r["cls"] in CONTINUING and r["cls"] != "subagent_loop"
            and r["hit"] == 0),
    }
    samples = {k: [] for k in stats}
    rng = random.Random(1)
    for _ in range(BOOTSTRAP_RESAMPLES):
        drawn = [r for _ in blocks for r in blocks[rng.randrange(len(blocks))]]
        for k, f in stats.items():
            v = f(drawn)
            if v is not None:
                samples[k].append(v)
    return {k: (pct(v, 0.025), pct(v, 0.975)) for k, v in samples.items() if v}


def metrics(A, cold_seeds):
    rows = [r for r in A["requests"] if r.get("server") and r["cls"] != "aborted"]
    cont = [r for r in rows if r["cls"] in CONTINUING]
    newl = [r for r in rows if r["cls"] in NEW_LONG]
    cold = [r for r in rows if r["seed"] in cold_seeds]
    dec = [r for r in rows if r.get("stream_s") and r["stream_s"] > 0 and r["completion"] > 1]
    copy = [r for r in dec if r["copy"]]
    big = [r for r in cold if r["computed"] >= 32768]
    budget = ((A["start"] or {}).get("server") or {}).get("default_thinking_budget")
    tp = sum(r["prompt"] for r in rows)
    th = sum(r["hit"] for r in rows)
    m = {
        "n": len(rows),
        "n_failed": sum(1 for r in A["requests"] if r["cls"] != "aborted" and not r.get("server")),
        "ttft_mean": mean([r["ttft"] for r in rows]),
        "ttft_median": median([r["ttft"] for r in rows]),
        "ttft_p90": pct([r["ttft"] for r in rows], 0.9),
        "ttft_cont_mean": mean([r["ttft"] for r in cont]),
        "ttft_cont_median": median([r["ttft"] for r in cont]),
        "ttft_new_mean": mean([r["ttft"] for r in newl]),
        "n_cont": len(cont), "n_new": len(newl),
        "prompt_tokens": tp, "hit_tokens": th, "hit_rate": th / tp if tp else None,
        "computed_tokens": sum(r["computed"] for r in rows),
        "cont_full_prefill": sum(1 for r in cont if r["hit"] == 0),
        "cont_full_prefill_main": sum(1 for r in cont if r["hit"] == 0 and r["cls"] != "subagent_loop"),
        "cont_full_prefill_sub": sum(1 for r in cont if r["hit"] == 0 and r["cls"] == "subagent_loop"),
        "n_cont_main": sum(1 for r in cont if r["cls"] != "subagent_loop"),
        "n_cont_sub": sum(1 for r in cont if r["cls"] == "subagent_loop"),
        "cold_prefill_big": (sum(r["computed"] for r in big) /
                             sum(r["ttft"] for r in big)) if big else None,
        "n_cold_big": len(big),
        "decode": decode_metrics(A, rows),
        "thinking_budget_hits": sum(1 for r in rows if budget and r["thinking"] >= budget),
        "budget_hit_ngram": sum(r["ngram_accepted"] for r in rows
                                if budget and r["thinking"] >= budget),
        "budget_hit_tokens": sum(r["completion"] for r in rows
                                 if budget and r["thinking"] >= budget),
        "cont_hit_rate": (sum(r["hit"] for r in cont) / max(1, sum(r["prompt"] for r in cont))),
        "cold_prefill_median": median([r["computed"] / r["ttft"] for r in cold if r["ttft"]]),
        "cold_prefill_aggregate": (sum(r["computed"] for r in cold) /
                                   max(1e-9, sum(r["ttft"] for r in cold))) if cold else None,
        "n_cold": len(cold),
        "output_tps": (sum(r["completion"] - 1 for r in dec) / sum(r["stream_s"] for r in dec)) if dec else None,
        "output_tps_copy": (sum(r["completion"] - 1 for r in copy) / sum(r["stream_s"] for r in copy)) if copy else None,
        "n_copy": len(copy),
        "completion_tokens": sum(r["completion"] for r in rows),
        "thinking_tokens": sum(r["thinking"] for r in rows),
        "spec_accept": (sum(r["spec_accepted"] for r in rows) /
                        max(1, sum(r["spec_drafted"] for r in rows))),
        "ngram_accepted": sum(r["ngram_accepted"] for r in rows),
        "ngram_drafted": sum(r["ngram_drafted"] for r in rows),
        "wall_seconds": A["meta"].get("wall_seconds"),
        "context_guards": sum(1 for e in A["events"] if e.get("event") == "context_guard"),
        "intervals": request_intervals(rows),
    }
    paths = {}
    for r in rows:
        d = paths.setdefault(r["path"], [0, 0])
        d[0] += 1
        d[1] += r["hit"]
    m["paths"] = paths
    pressure = {}
    for o in A["throughput"]:
        cc = o.get("context_cache") or {}
        for k, v in (cc.get("pressure") or {}).items():
            if isinstance(v, (int, float)):
                pressure[k] = pressure.get(k, 0) + v
        for kind in ("main_kv_transfers",):
            for d, t in ((cc.get(kind) or {}).items()):
                pressure["%s_%s_bytes" % (kind, d)] = pressure.get("%s_%s_bytes" % (kind, d), 0) + \
                    (t.get("bytes") or 0)
    m["pressure"] = pressure
    return m


# ---------------------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------------------

def f1(v):
    return "n/a" if v is None else "%.1f" % v


def f2(v):
    return "n/a" if v is None else "%.2f" % v


def f0(v):
    return "n/a" if v is None else "%.0f" % v


def ntok(v):
    return "n/a" if v is None else format(int(round(v)), ",d")


def ppct(v):
    return "n/a" if v is None else "%.1f %%" % (v * 100)


def with_ci(fmt, seconds_key=None):
    """Formats a decode estimate with its 95 % interval, e.g. `187 (184-190)`."""
    def render(d):
        e, seconds = d
        if e["value"] is None:
            return "n/a"
        if e["lo"] is None:
            return "%s (%.0f s, too little for an interval)" % (fmt(e["value"]), seconds)
        return "%s (%s-%s)" % (fmt(e["value"]), fmt(e["lo"]), fmt(e["hi"]))
    return render


def chg(c, t):
    if c in (None, 0) or t is None:
        return "n/a"
    d = (t - c) / c * 100
    return "%s%.1f %%" % ("+" if d >= 0 else "−", abs(d))


def chg_pp(c, t):
    if c is None or t is None:
        return "n/a"
    d = (t - c) * 100
    return "%s%.1f points" % ("+" if d >= 0 else "−", abs(d))


def chg_abs(c, t):
    if c is None or t is None:
        return "n/a"
    d = t - c
    return "%s%d" % ("+" if d >= 0 else "−", abs(d))


def headline_specs(c):
    """The comparison rows: (label, metric path, format, change kind, divisor).

    A path ending in a decode estimate renders with its interval; a request metric renders with
    its interval from `intervals` when it has one. Change kinds: `rel` percent, `pp` points,
    `abs` count difference."""
    return [
        ("Average time to first token (s, lower is better)", ("ttft_mean",), f1, "rel", None),
        ("Median time to first token (s, lower is better)", ("ttft_median",), f2, "rel", None),
        ("90th-percentile time to first token (s, lower is better)", ("ttft_p90",), f1, "rel",
         None),
        ("Average TTFT, continuing-session turns (s)", ("ttft_cont_mean",), f2, "rel", None),
        ("Average TTFT, new long prompts (s)", ("ttft_new_mean",), f1, "rel", None),
        ("Prompt tokens served from cache", ("hit_rate",), ppct, "pp", None),
        ("Prompt tokens prefilled (lower is better)", ("computed_tokens",), ntok, "rel", None),
        ("Main-session turns that re-prefilled the whole prompt (of %d)" % c["n_cont_main"],
         ("cont_full_prefill_main",), f0, "abs", None),
        ("Subagent turns that re-prefilled the whole prompt (of %d)" % c["n_cont_sub"],
         ("cont_full_prefill_sub",), f0, "abs", None),
        ("Prefill tok/s, requests with no cache hit (all sizes)", ("cold_prefill_aggregate",),
         ntok, "rel", None),
        ("Prefill tok/s, requests with no cache hit, 32K+ tokens", ("cold_prefill_big",), ntok,
         "rel", None),
        ("Output tok/s, one request decoding", ("decode", "one", "tps"), ntok, "rel", None),
        ("Output tok/s, all decoding at the run's own batching", ("decode", "all", "tps"), ntok,
         "rel", None),
        ("Decode rounds/s, one request decoding (engine speed)",
         ("decode", "one", "rounds_per_s"), f1, "rel", None),
        ("Tokens per round, one request decoding (speculative acceptance)",
         ("decode", "one", "tokens_per_row_round"), f2, "rel", None),
        ("Workload wall time (min, lower is better)", ("wall_seconds",), f1, "rel", 60.0),
    ]


def resolve(m, path, divisor=None):
    """The scalar at `path` (a decode estimate's value), or None."""
    v = m
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    if isinstance(v, dict):
        v = v.get("value")
    return v / divisor if v is not None and divisor else v


def change_value(kind, c, t):
    if c is None or t is None:
        return None
    if kind == "rel":
        return (t - c) / c * 100 if c else None
    return (t - c) * 100 if kind == "pp" else t - c


def change_text(kind, d, spread=None):
    """`−12.3 %`, `+4.1 points` or `−7`, with an optional `(lo to hi)` range."""
    if d is None:
        return "n/a"
    unit = {"rel": " %", "pp": " points", "abs": ""}[kind]
    digits = "%.0f" if kind == "abs" and spread is None else "%.1f"

    def signed(x):
        return ("+" if x >= 0 else "−") + (digits % abs(x)) + unit

    if spread is None:
        return signed(d)
    return "%s (%s to %s)" % (signed(d), bare(signed(spread[0])), bare(signed(spread[1])))


def bare(text):
    """A formatted number without its unit, for the bounds of a range."""
    return text.replace(" points", "").replace(" %", "")


def cell(m, spec):
    """One arm's value in the single-run table, with its interval when it has one."""
    _, path, fmt, _, divisor = spec
    v = m
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    if isinstance(v, dict):   # decode estimate
        seconds = resolve(m, path[:-1] + ("seconds",)) or 0.0
        return with_ci(fmt)((v, seconds))
    value = resolve(m, path, divisor)
    ci = (m.get("intervals") or {}).get(path[-1]) if len(path) == 1 else None
    if value is None or not ci:
        return fmt(value)
    return "%s (%s-%s)" % (fmt(value), bare(fmt(ci[0])), bare(fmt(ci[1])))


def table_header(arms, change_label):
    others = arms[1:]
    return ["| Metric | %s | %s |" % (" | ".join(LABEL[a] for a in arms),
                                      " | ".join(change_label % (SHORT[a], SHORT[arms[0]])
                                                 for a in others)),
            "|---|" + "---|" * (len(arms) + len(others))]


def headline(arms, ms):
    """The comparison table: one column per arm, then each arm's change against the control."""
    L = table_header(arms, "%s vs %s")
    for spec in headline_specs(ms[arms[0]]):
        _, path, _, kind, divisor = spec
        c = resolve(ms[arms[0]], path, divisor)
        L.append("| %s | %s | %s |" % (
            spec[0], " | ".join(cell(ms[a], spec) for a in arms),
            " | ".join(change_text(kind, change_value(kind, c, resolve(ms[a], path, divisor)))
                       for a in arms[1:])))
    return "\n".join(L)


def each(arms, value, fmt=lambda v: v):
    """`ninfer 1.2, infernix 3.4, infernix alt 5.6`, for the notes."""
    return ", ".join("%s %s" % (SHORT[a].lower(), fmt(value(a))) for a in arms)


def cold_table(arms, A, cold_seeds):
    served = {a: {r["seed"]: r for r in A[a]["requests"] if r.get("server")} for a in arms}
    others = arms[1:]
    L = ["| Request | Class | Prompt tokens | %s | %s |"
         % (" | ".join("%s prefill tok/s" % SHORT[a] for a in arms),
            " | ".join("%s vs %s" % (SHORT[a], SHORT[arms[0]]) for a in others)),
         "|---|---|---|" + "---|" * (len(arms) + len(others))]
    base = served[arms[0]]
    for s in sorted(cold_seeds, key=lambda s: base[s]["computed"]):
        rates = [served[a][s]["computed"] / served[a][s]["ttft"]
                 if served[a][s]["ttft"] else None for a in arms]
        L.append("| %s | %s | %s | %s | %s |" % (
            base[s]["tag"], base[s]["cls"], ntok(base[s]["computed"]),
            " | ".join(ntok(r) for r in rates), " | ".join(chg(rates[0], r) for r in rates[1:])))
    return "\n".join(L)


def class_table(arms, A):
    L = ["| Class | n | %s | %s |" % (" | ".join("%s avg TTFT (s)" % SHORT[a] for a in arms),
                                     " | ".join("%s cached" % SHORT[a] for a in arms)),
         "|---|---|" + "---|" * (2 * len(arms))]
    for cls, _ in CLASS_DOC:
        rs = [[r for r in A[a]["requests"] if r["cls"] == cls and r.get("server")] for a in arms]
        if not any(rs):
            continue
        L.append("| %s | %d | %s | %s |" % (
            cls, max(len(r) for r in rs), " | ".join(f2(mean([x["ttft"] for x in r])) for r in rs),
            " | ".join(ppct(sum(x["hit"] for x in r) / max(1, sum(x["prompt"] for x in r)))
                       for r in rs)))
    return "\n".join(L)


def path_table(arms, ms):
    L = ["| Reuse path | %s |" % " | ".join("%s (requests / cached tokens)" % SHORT[a] for a in arms),
         "|---|" + "---|" * len(arms)]
    for p in sorted(set().union(*(ms[a]["paths"] for a in arms))):
        cells = [ms[a]["paths"].get(p, [0, 0]) for a in arms]
        L.append("| %s | %s |" % (p, " | ".join("%d / %s" % (n, ntok(t)) for n, t in cells)))
    return "\n".join(L)


def pressure_table(arms, ms):
    keys = ["private_owners_evicted", "private_owners_degraded", "shared_owners_evicted",
            "shared_owners_degraded", "checkpoints_dropped", "maximal_fallback_selections",
            "search_budget_exhaustions", "spill_pages", "main_kv_transfers_d2h_bytes",
            "main_kv_transfers_h2d_bytes"]
    L = ["| Engine counter (summed over the run) | %s |" % " | ".join(SHORT[a] for a in arms),
         "|---|" + "---|" * len(arms)]
    for k in keys:
        v = [ms[a]["pressure"].get(k) for a in arms]
        if all(x is None for x in v):
            continue
        fmt = (lambda v: "n/a" if v is None else "%.1f GiB" % (v / 2 ** 30)) if k.endswith("bytes") \
            else (lambda v: "n/a" if v is None else format(int(v), ",d"))
        L.append("| %s | %s |" % (k, " | ".join(fmt(x) for x in v)))
    return "\n".join(L)


def per_request(A):
    L = ["| Tag | Class | Prompt | Cached | Prefilled | Path | TTFT s | Out | Stream tok/s | Finish |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(A["requests"], key=lambda r: r.get("t_send") or 0):
        if not r.get("server"):
            L.append("| %s | %s | | | | | | | | %s |" % (r["tag"], r["cls"], r["status"]))
            continue
        L.append("| %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["tag"], r["cls"], " (copy)" if r["copy"] else "", ntok(r["prompt"]), ntok(r["hit"]),
            ntok(r["computed"]), r["path"], f2(r["ttft"]), ntok(r["completion"]),
            ntok((r["completion"] - 1) / r["stream_s"]) if r.get("stream_s") and r["completion"] > 1 else "n/a",
            r["finish_server"]))
    return "\n".join(L)


def server_row(A):
    s = A["start"] or {}
    e, mem = s.get("engine") or {}, s.get("memory") or {}
    cc = e.get("context_cache") or {}
    return ("KV capacity %s tokens; host state slots %s; host KV %.1f GiB; private "
            "continuations %s; long anchors per continuation %s; shared prefixes %s; build %s"
            % (ntok(e.get("kv_capacity")), cc.get("host_state_slots"),
               (mem.get("host_kv_capacity_bytes") or 0) / 2 ** 30, cc.get("max_private_continuations"),
               cc.get("max_long_anchors_per_continuation"), cc.get("max_shared_prefixes"),
               A["build"]))


def flag_str(flags):
    return " ".join(n + ("" if v is None else " " + (('"%s"' % v) if " " in v else v))
                    for n, v in flags)


def aggregate(out_dir, run_dirs):
    """One report over runs of the same arms with different workload seeds.

    Each cell is the mean over seeds with the min-max range; each change is the per-seed change
    against that seed's control, averaged, with its min-max range. A seed changes every
    observation and every sampled answer, so the ranges show how much a result depends on the
    particular session rather than on the build."""
    runs = []
    for d in run_dirs:
        with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
            runs.append((d, json.load(f)))
    use_labels(runs[0][1]["config"])
    arms = [a for a in ARMS if all(a in s for _, s in runs)]
    if arms[:1] != ["control"] or len(arms) < 2:
        raise SystemExit("aggregate needs a control arm and one other arm in every run")
    seeds = [s["config"]["seed"] for _, s in runs]
    L = ["# Agentic A/B over %d workload seeds: %s\n" % (len(runs), " vs ".join(LABEL[a] for a in arms)),
         "Runs: %s\n" % ", ".join("seed %d `%s`" % (seed, d) for seed, (d, _) in zip(seeds, runs))]
    first = runs[0][1]
    L.append("- Model: `%s`, `--max-context %s`; %d requests per arm per seed; every arm of a "
             "seed replays the same observations, and each seed replays different ones.\n"
             % (os.path.basename(first["config"]["model"]),
                ntok(first["config"].get("max_context")), first["control"]["n"]))
    problems = ["seed %d %s: %d failed request(s)" % (seed, a, s[a]["n_failed"])
                for seed, (_, s) in zip(seeds, runs) for a in arms if s[a]["n_failed"]]
    if problems:
        L.append("**Validity warnings:** " + "; ".join(problems) + "\n")
    L.append("## Headline (mean over seeds, min-max in brackets)\n")
    L += table_header(arms, "%s vs %s, per seed")
    for label, path, fmt, kind, divisor in headline_specs(first["control"]):
        fmt = f1 if kind == "abs" else fmt
        cells = []
        for a in arms:
            v = [resolve(s[a], path, divisor) for _, s in runs]
            v = [x for x in v if x is not None]
            cells.append("n/a" if not v else fmt(sum(v) / len(v)) if len(v) == 1 else
                         "%s (%s-%s)" % (fmt(sum(v) / len(v)), bare(fmt(min(v))),
                                         bare(fmt(max(v)))))
        changes = []
        for a in arms[1:]:
            d = [change_value(kind, resolve(s["control"], path, divisor),
                              resolve(s[a], path, divisor)) for _, s in runs]
            d = [x for x in d if x is not None]
            changes.append("n/a" if not d else
                           change_text(kind, sum(d) / len(d),
                                       (min(d), max(d)) if len(d) > 1 else None))
        L.append("| %s | %s | %s |" % (label, " | ".join(cells), " | ".join(changes)))
    L.append("")
    L.append("## Per seed\n")
    keys = [("Prompt tokens served from cache", ("hit_rate",), ppct, None),
            ("Average TTFT (s)", ("ttft_mean",), f1, None),
            ("Output tok/s, one request", ("decode", "one", "tps"), ntok, None),
            ("Decode rounds/s, one request", ("decode", "one", "rounds_per_s"), f1, None),
            ("Output tok/s, run's own batching", ("decode", "all", "tps"), ntok, None),
            ("Completion tokens", ("completion_tokens",), ntok, None),
            ("Wall time (min)", ("wall_seconds",), f1, 60.0)]
    L.append("| Seed | Arm | %s |" % " | ".join(k[0] for k in keys))
    L.append("|---|---|" + "---|" * len(keys))
    for seed, (_, s) in zip(seeds, runs):
        for a in arms:
            L.append("| %d | %s | %s |" % (seed, SHORT[a], " | ".join(
                fmt(resolve(s[a], path, divisor)) for _, path, fmt, divisor in keys)))
    L.append("")
    L.append("Each seed's own report (`report.md` in its run directory) has the intervals, notes "
             "and per-request detail for that seed.")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("report: %s" % os.path.join(out_dir, "report.md"))


def main(argv):
    if argv[:1] == ["--aggregate"]:
        aggregate(argv[1], argv[2:])
        return
    run_dir = argv[0]
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    use_labels(cfg)
    arms = [a for a in ARMS if os.path.exists(os.path.join(run_dir, a, "client.jsonl"))]
    if arms[:1] != ["control"] or len(arms) < 2:
        raise SystemExit("%s needs a control arm and at least one other arm" % run_dir)
    A = {a: load_arm(run_dir, a) for a in arms}
    # Matched cold set: requests with no cache hit and a real prefill in EVERY arm, sent while no other
    # request was in flight in every arm, so their client TTFT is prefill (no queueing behind others).
    served = [{r["seed"]: r for r in A[a]["requests"] if r.get("server")} for a in arms]
    spans = {a: in_flight(A[a]) for a in arms}
    cold = {s for s in served[0]
            if all(s in d and d[s]["hit"] == 0 and d[s]["computed"] >= 4096 and d[s]["ttft"]
                   and alone_while(spans[a], s, d[s]["t_send"], d[s]["t_send"] + 1e-6)
                   for a, d in zip(arms, served))}
    ms = {a: metrics(A[a], cold) for a in arms}
    summary = dict({"config": cfg, "cold_seeds": sorted(cold)}, **ms)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    c = ms["control"]
    ctx = cfg.get("max_context")
    gpu = next((A[a]["start"].get("environment", {}).get("gpu_name")
                for a in reversed(arms) if A[a]["start"]), None) or "GPU"
    L = []
    L.append("# Agentic A/B: %s\n" % " vs ".join(LABEL[a] for a in arms))
    L.append("Run directory: `%s`\n" % run_dir)
    L.append("- Model: `%s` on an %s, `--max-context %s` for every arm (launch bat: %s)."
             % (os.path.basename(cfg["model"]), gpu, ntok(ctx), ntok(cfg.get("bat_max_context"))))
    L.append("- Workload: %d requests per arm (seed %d, scale %.2f, corpus commit `%s`), "
             "`max_tokens: %d` on agent turns, thinking on."
             % (c["n"], cfg["seed"], cfg["scale"], cfg["corpus_commit"], cfg["agent_max_tokens"]))
    exes = {"control": cfg["control_exe"], "treatment": cfg["treatment_exe"],
            "alt": cfg.get("alt_exe") or cfg["treatment_exe"]}
    for a in arms:
        extra = (" with `%s`" % " ".join(cfg.get("alt_extra_flags") or [])) if a == "alt" else ""
        L.append("- %s: `%s`%s - %s" % (LABEL[a], exes[a], extra, server_row(A[a])))
    L.append("- Total benchmark time including model loads: %.1f min.\n"
             % (cfg.get("total_seconds", 0) / 60))
    problems = []
    for a in arms:
        if ms[a]["n_failed"]:
            problems.append("%s: %d workload request(s) have no request_done record"
                            % (a, ms[a]["n_failed"]))
        if ms[a]["context_guards"]:
            problems.append("%s: the client context guard fired %d time(s)"
                            % (a, ms[a]["context_guards"]))
    if problems:
        L.append("**Validity warnings:** " + "; ".join(problems) + "\n")

    def per_arm(value, fmt=str):
        return each(arms, lambda a: value(ms[a]), fmt)

    L.append("## Headline\n")
    L.append(headline(arms, ms) + "\n")
    L.append("How to read it:\n")
    L.append("- Every row compares the same logical requests: the client tags each one with a "
             "seed that is identical in every arm. Observations (tool results, user messages, "
             "summaries) are byte-identical; assistant turns are each arm's own output fed back, "
             "as an agent client does, so prompt totals differ slightly (tokens: %s)."
             % per_arm(lambda m: m["prompt_tokens"], ntok))
    L.append("- Every time is the client's clock (time.time() around the streaming request), for every "
             "engine alike: TTFT is the first streamed token less the send, and includes queueing behind "
             "other sessions. Engine-internal timers are not used, as engines place their boundaries "
             "differently.")
    L.append("- Continuing-session turns (%d) are tool-loop turns, retries, the post-idle and "
             "post-history-edit turns: cache retention decides their TTFT. New long prompts (%d) "
             "are resumed sessions, compaction and loop-check calls: prefill speed decides theirs."
             % (c["n_cont"], c["n_new"]))
    L.append("- \"No cache hit\" prefill rates are token-weighted (total prefilled tokens / total client "
             "TTFT) over the %d requests that had no cache hit, prefilled at least 4,096 tokens and were sent "
             "while no other request was in flight, in *every* arm (%d of them 32K+; per-request table "
             "below). Per-request median (tok/s): %s."
             % (c["n_cold"], c["n_cold_big"], per_arm(lambda m: m["cold_prefill_median"], ntok)))
    L.append("- Output tok/s is tokens after the first over the client's time from the first streamed token "
             "to the last. The one-request row uses the requests that streamed with no other request in "
             "flight (requests / seconds: %s); the *all* row is every streamed token over the time at least "
             "one request was streaming (seconds: %s). Brackets are 95 %% bootstrap intervals over the "
             "requests."
             % (per_arm(lambda m: "%d / %s" % (m["decode"]["one"]["n"], f1(m["decode"]["one"]["seconds"]))),
                per_arm(lambda m: f1(m["decode"]["all"]["seconds"]))))
    L.append("- Output tok/s = decode rounds/s x tokens per round. Rounds/s is the engine's own "
             "speed and barely moves with the text; tokens per round is speculative acceptance, "
             "which moves with what the model happened to write (file copies accept far more than "
             "fresh reasoning), so it carries most of the interval on output tok/s. Compare "
             "rounds/s for kernel and host speed, and tokens per round for drafting; each arm "
             "samples its own text, so acceptance differs between runs as well as builds. Tokens "
             "per request-round: the server's round counts. Over every streamed request (tok/s: %s; "
             "file-writing turns: %s), what one stream saw, including other requests' batching and prefill."
             % (per_arm(lambda m: m["output_tps"], ntok),
                per_arm(lambda m: m["output_tps_copy"], ntok)))
    L.append("- Output volume differs because the sampled text does: completion tokens %s "
             "(thinking: %s); turns that used the whole thinking budget: %s (ngram copies "
             "supplied %s of those turns' tokens; a high share signals repetitive thinking). "
             "Longer outputs hold more KV and add batching, so they also shift cache pressure "
             "and queueing; compare repeated runs before attributing a thinking-length "
             "difference to a build."
             % (per_arm(lambda m: m["completion_tokens"], ntok),
                per_arm(lambda m: m["thinking_tokens"], ntok),
                per_arm(lambda m: m["thinking_budget_hits"]),
                per_arm(lambda m: "%s of %s" % (ntok(m["budget_hit_ngram"]),
                                                ntok(m["budget_hit_tokens"])))))
    L.append("- Speculative acceptance of drafted tokens: %s. Ngram drafting accepted %s."
             % (per_arm(lambda m: m["spec_accept"], ppct),
                per_arm(lambda m: "%s of %s drafted tokens (%s of output)"
                        % (ntok(m["ngram_accepted"]), ntok(m["ngram_drafted"]),
                           ppct(m["ngram_accepted"] / max(1, m["completion_tokens"]))))))
    L.append("")
    L.append("## Prefill on requests with no cache hit\n")
    L.append(cold_table(arms, A, cold) + "\n")
    L.append("## By request class\n")
    L.append(class_table(arms, A) + "\n")
    L.append("Classes: " + "; ".join("`%s` %s" % c for c in CLASS_DOC) + ".\n")
    L.append("## Where cache hits came from\n")
    L.append(path_table(arms, ms) + "\n")
    L.append("## Cache pressure\n")
    L.append(pressure_table(arms, ms) + "\n")
    L.append("## Launch parameters\n")
    for a in arms:
        if a == "control":
            L.append("%s (Infernix-only flags dropped: %s):\n\n```text\n%s\n```\n"
                     % (LABEL[a], ", ".join("`%s`" % d for d in cfg.get("dropped_for_control", [])),
                        flag_str(cfg.get("control_flags", []))))
        else:
            L.append("%s:\n\n```text\n%s\n```\n" % (LABEL[a], flag_str(cfg.get(a + "_flags", []))))
    L.append("## Per-request detail\n")
    for a in arms:
        L.append("### %s\n\n%s\n" % (LABEL[a], per_request(A[a])))
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(headline(arms, ms))
    print("report: %s" % os.path.join(run_dir, "report.md"))


if __name__ == "__main__":
    main(sys.argv[1:])
