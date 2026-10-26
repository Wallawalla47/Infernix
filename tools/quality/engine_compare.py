"""Engine comparison against the exact model: llama.cpp's KL-divergence measurement on WikiText-2, for Infernix's
routes and fast paths and for other engines on the same weights.

  python -m tools.quality.engine_compare plan CONFIG.json
  python -m tools.quality.engine_compare run CONFIG.json [--only ID,...]
  python -m tools.quality.engine_compare report CONFIG.json

One CONFIG describes one weight set at one context: the artifact holding those exact weights, the text, and what
to measure. `run` executes the steps not yet done, in order (a step is done when its PREFIX.json exists), logging
each to WORK/logs; the fast-path steps appear once the default arms have reported which paths they run. `report`
writes WORK/report.md and WORK/report.json.

The reference is Infernix's lossless route (every operation correctly rounded, FP32 logits, BF16 KV) written as
llama.cpp's `_logits_` base file, which llama-perplexity --kl-divergence-base, Basalt's ppl_gate and every arm
here score against. Arms, on the base's positions (the second half of each chunk, llama.cpp's convention):

  P-*  the prompt path: each chunk in one pass (infernix-certify prefill; llama-perplexity -b n_ctx).
  D-*  the decode path: each chunk's first half prefilled, its scored half forced through decode steps in one
       request (infernix-certify decode; Basalt's ppl_gate), each step's whole logits row captured and scored
       as the P arms score theirs. The Qwen3.5 family captures rows on eager rounds (decode_cuda_graph false),
       and a pair of check arms (D-*-graphs, D-*-eager) shows the default route's arithmetic is the same with
       and without CUDA graphs.
  P-twin-sN  the noise twin: the lossless route with each rounding made stochastic (seed N), an ideal
       implementation off by at most one unit in the last place at every rounding: honest rounding's floor.
  *-reproducible-KV  fast paths off; *-default-KV  the product's defaults; *-path-NAME  fast paths off plus that
       one path (and the paths it needs to act): its own effect, against the arm without it.
  E-NAME  another engine (llama-perplexity's printout, or its logits rows streamed into infernix-certify
       llama-rows).

CONFIG (JSON):
  {"name": "Qwen3.8-27B NVIDIA NVFP4", "certify": ".../infernix-certify.exe", "artifact": ".../x.ninfer",
   "text": ".../wikitext-2-raw/wiki.test.raw", "context": 512, "chunks": 0, "work": ".../q27-nvfp4-c512",
   "kv": "int8", "twin_seeds": [1, 2], "concurrency": 8,
   "route_args": [], "spec": ["--spec", "dflash2", "--draft-tokens", "auto", "--ngram-draft-tokens", "auto"],
   "paths": "default", "budget": 0.25, "threads": 16,
   "engines": [{"name": "llama.cpp b10221", "kind": "llama-perplexity", "binary": ".../llama-perplexity.exe",
                "model": ".../x.gguf", "args": ["-ngl", "99", "-b", "512"], "protocol": "P"},
               {"name": "Strata 0.1.45", "kind": "rows-command", "argv": ["py", "-m", "...", "{base}"], "protocol": "D"},
               {"name": "Basalt", "kind": "positions", "prefix": ".../basalt-ppl-gate", "protocol": "D"}]}
  route_args are the artifact's companions every arm and the reference take (--ngram-volume FILE); spec holds
  the drafting options with their files (--draft-model FILE).
  chunks 0 takes every whole chunk of the text (WikiText-2 test: 580 of 512 tokens for Qwen's tokenizer);
  kv is the KV format the product recommends; spec, the recommended drafting (empty: none); paths, "default" (the
  paths the default arms run) or a list of names; budget, the decision rule's fraction of the floor.

Decision rule (proposed; the owner decides the budget): a fast path is rounding-level when the upper 95 %
bound of its increment is at most budget x the noise twin's value for mean KLD, KLD 99.9 % and top-1
disagreement, and its ln-perplexity increment at most budget x the twin's mean KLD. The report shows every
number; the rule only labels them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools.quality import llama_kld

# Where each fast path's kernels run in this measurement (include/infernix/types.h describes each path).
PROMPT_PATHS = {"prompt-pv8", "fp8-a8-pair", "fp8-a8-pair-block32", "q4-w4a8", "gdn-f16-accumulate",
                "expert-a4-residual", "expert-a4-residual-3", "ggml-bf16-prefill", "f16-accumulate-prefill",
                "bf16-prompt-split", "router-mma"}
DECODE_PATHS = {"decode-split64", "q8-sliced-k", "q5-ksplit-mma", "batched-decode", "batched-a8-decode",
                "batched-a4-decode", "batched-a4-residual-decode", "batched-a8-residual-decode",
                "batched-a8-subset-decode", "single-column-decode", "expert-a4-residual-decode", "batch-split-attention",
                "screened-lm-head"}
BOTH_PATHS = {"gdn-conv-fused", "qsa-prepare-fused", "fp16-gdn-state"}
# Paths that act only in speculative rounds: measured with the configuration's drafting.
SPEC_PATHS = {"adaptive-speculation", "adaptive-mtp", "adaptive-dflash", "dflash2-pairs", "gdn-tree-closed-form"}
# A path that acts only beside others: measured with them, against the arm with them alone.
PREREQUISITES = {"batched-a8-decode": ["batched-decode"], "batched-a4-decode": ["batched-decode"],
                 "batched-a4-residual-decode": ["batched-decode"], "batched-a8-residual-decode": ["batched-decode"],
                 "batched-a8-subset-decode": ["batched-decode"], "adaptive-mtp": ["adaptive-speculation"],
                 "adaptive-dflash": ["adaptive-speculation"]}
# Rows a decode path needs in a round to act (single-column-decode acts only on a lone row; dflash2-pairs on two).
DECODE_CONCURRENCY = {"single-column-decode": 1, "dflash2-pairs": 2, "adaptive-speculation": 1, "adaptive-mtp": 1,
                      "adaptive-dflash": 1, "gdn-tree-closed-form": 1, "screened-lm-head": 1}
NOT_EXERCISED = {
    "free-cache-resume": "acts on prefix-cache hits; the chunks share no prefix",
    "preemption": "acts under KV pressure; the arms reserve every chunk's KV",
    "idle-wide-prefill": "a no-op in scoring (no request waits to decode)",
    "sampled-mtp-drafts": "acts on sampled requests; the measurement decodes greedily",
    "block-verification": "acts on sampled requests; the measurement decodes greedily",
    "univer-trees": "acts on sampled requests; the measurement decodes greedily",
}
MINUTES = {"reference": 120, "twin": 120, "P": 45, "D": 240, "E": 180}


@dataclass
class Step:
    id: str
    role: str          # reference, twin, reproducible, default, path, invariance, engine
    protocol: str      # ref, P, D
    argv: list[str]
    minutes: int
    base: str | None = None      # the arm a path's increment is taken against
    path: str | None = None
    note: str = ""
    engine: dict | None = None
    meta: dict = field(default_factory=dict)


def _load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("name", "certify", "artifact", "text", "work"):
        if key not in config:
            raise ValueError(f"{path}: '{key}' is required")
    config.setdefault("context", 512)
    config.setdefault("chunks", 0)
    config.setdefault("kv", "bf16")
    config.setdefault("twin_seeds", [1, 2])
    config.setdefault("concurrency", 8)
    config.setdefault("route_args", [])
    config.setdefault("spec", [])
    config.setdefault("arms", "default")
    config.setdefault("paths", "default")
    config.setdefault("budget", 0.25)
    if config["arms"] not in ("default", "split"):
        raise ValueError(f"{path}: arms must be default or split")
    config.setdefault("engines", [])
    config.setdefault("decode", True)
    config.setdefault("logits", "bf16")
    config.setdefault("decode_cuda_graph", True)
    if config["logits"] not in ("bf16", "fp32"):
        raise ValueError(f"{path}: logits must be bf16 or fp32")
    return config


def _work(config: dict) -> Path:
    return Path(config["work"])


def _base(config: dict) -> Path:
    return _work(config) / "reference.kld"


def _done(config: dict, step: Step) -> bool:
    return (_work(config) / f"{step.id}.json").exists() or (step.role == "reference" and Path(str(_base(config)) + ".json").exists())


def _read(config: dict, arm_id: str) -> dict | None:
    path = _work(config) / f"{arm_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _arm(config: dict, step_id: str, protocol: str, args: list[str], role: str, concurrency: int | None = None,
         **extra) -> Step:
    certify = config["certify"]
    common = [config["artifact"], "--llama-base", str(_base(config)), "--out", str(_work(config) / step_id)]
    if "threads" in config:
        common += ["--threads", str(config["threads"])]
    bf16 = config["logits"] == "bf16"
    if protocol == "P":
        # The exact route's head rounds its exact sums to the precision; every other arm's FP32 logits are
        # rounded by the scorer, as another engine's are.
        precision = (["--logits-dtype", "bf16"] if role == "twin" else ["--round-logits", "bf16"]) if bf16 else []
        argv = [certify, "prefill", *common, *config["route_args"], *args, *precision]
    else:
        # The decode arm captures each step's whole row (the Qwen3.5 family on eager rounds: decode_cuda_graph
        # false) and the scorer rounds it; its --observed-only form reads the head's own log-probabilities.
        observed = "--observed-only" in args
        precision = (["--logits-dtype", "bf16"] if observed else ["--round-logits", "bf16"]) if bf16 else []
        graphs = [] if config["decode_cuda_graph"] or "--cuda-graph" in args else ["--no-cuda-graph"]
        args = [a for a in args if a != "--cuda-graph"]
        argv = [certify, "decode", *common, *config["route_args"], "--concurrency", str(concurrency or 1), *args,
                *graphs, *precision]
    return Step(step_id, role, protocol, argv, MINUTES[protocol], **extra)


def _paths(config: dict) -> tuple[list[str], list[str]]:
    """The paths measured alone: the prompt arms' and the decode arms' default selections (or the CONFIG's list),
    and whether they are known yet."""
    if config["paths"] != "default":
        return list(config["paths"]), []
    prompt, decode = _read(config, f"P-default-{config['kv']}"), _read(config, f"D-default-{config['kv']}-c1")
    spec = _read(config, f"D-default-{config['kv']}-spec-c1") if config["spec"] else None
    missing = [name for name, meta in (("P-default", prompt), ("D-default-c1", decode)) if meta is None]
    if config["spec"] and spec is None:
        missing.append("D-default-spec-c1")
    names: list[str] = []
    for meta in (prompt, decode, spec):
        for path in (meta or {}).get("load", {}).get("fast_paths", []):
            if path not in names:
                names.append(path)
    return names, missing


def plan(config: dict) -> tuple[list[Step], list[str]]:
    """Every step in run order, and notes on what is not planned (yet)."""
    kv, conc, notes = config["kv"], config["concurrency"], []
    reproducible = ["--reproducible", "--kv-dtype", kv]
    steps = [Step("reference", "reference", "ref",
                  [config["certify"], "reference", config["artifact"], "--format", "llama-kld", "--text", config["text"],
                   "--context", str(config["context"]), "--chunks", str(config["chunks"]), "--out", str(_base(config)),
                   "--logits-dtype", config["logits"], *config["route_args"]],
                  MINUTES["reference"])]
    for seed in config["twin_seeds"]:
        steps.append(_arm(config, f"P-twin-s{seed}", "P", ["--lossless", "--noise-seed", str(seed)], "twin"))
        steps[-1].minutes = MINUTES["twin"]
    if config["arms"] == "default":
        return _default_plan(config, steps), notes
    steps.append(_arm(config, "P-reproducible-bf16", "P", ["--reproducible", "--kv-dtype", "bf16"], "reproducible"))
    if kv != "bf16":
        steps.append(_arm(config, f"P-reproducible-{kv}", "P", reproducible, "reproducible"))
    steps.append(_arm(config, f"P-default-{kv}", "P", ["--kv-dtype", kv], "default"))
    if config["decode"]:
        steps.append(_arm(config, f"D-reproducible-{kv}-c{conc}", "D", reproducible, "reproducible", conc))
        # The reproducible route's request invariance, on the first chunks: one row at a time must give the bits
        # of eight together.
        check = 40 if config["chunks"] == 0 else min(40, config["chunks"])
        steps.append(_arm(config, f"D-reproducible-{kv}-c1-check", "D", reproducible + ["--chunks", str(check)],
                          "invariance", 1, base=f"D-reproducible-{kv}-c{conc}"))
        steps.append(_arm(config, f"D-default-{kv}-c1", "D", ["--kv-dtype", kv], "default", 1))
        if not config["decode_cuda_graph"]:
            # The arms capture rows on eager rounds: the observed form with and without CUDA graphs must agree bit
            # for bit, so the eager arms measure the graphs' arithmetic.
            check = ["--kv-dtype", kv, "--observed-only", "--chunks", str(40 if config["chunks"] == 0 else min(40, config["chunks"]))]
            steps.append(_arm(config, f"D-default-{kv}-c1-graphs", "D", check + ["--cuda-graph"], "graphs", 1,
                              base=f"D-default-{kv}-c1-eager"))
            steps.append(_arm(config, f"D-default-{kv}-c1-eager", "D", check, "graphs", 1))
        steps.append(_arm(config, f"D-default-{kv}-c{conc}", "D", ["--kv-dtype", kv], "default", conc))
        if config["spec"]:
            steps.append(_arm(config, f"D-reproducible-{kv}-spec-c1", "D", reproducible + config["spec"], "reproducible", 1))
            steps.append(_arm(config, f"D-default-{kv}-spec-c1", "D", ["--kv-dtype", kv, *config["spec"]], "default", 1))
    names, missing = _paths(config)
    if missing:
        notes.append("fast-path steps follow once these arms have run: " + ", ".join(missing))
    for name in names:
        if name in NOT_EXERCISED:
            notes.append(f"{name}: not measured ({NOT_EXERCISED[name]})")
            continue
        known = name in PROMPT_PATHS | DECODE_PATHS | BOTH_PATHS | SPEC_PATHS
        candidates = [*PREREQUISITES.get(name, []), name]
        flags = [flag for path in candidates for flag in ("--fast-path", path)]
        if name in PROMPT_PATHS | BOTH_PATHS or not known:
            base = f"P-path-{PREREQUISITES[name][-1]}" if name in PREREQUISITES else f"P-reproducible-{kv}"
            steps.append(_arm(config, f"P-path-{name}", "P", reproducible + flags, "path", base=base, path=name))
        if not config["decode"] or (known and name in PROMPT_PATHS):
            continue
        if name in SPEC_PATHS and not config["spec"]:
            notes.append(f"{name}: not measured (no drafting configured)")
            continue
        c = DECODE_CONCURRENCY.get(name, conc)
        spec = config["spec"] if name in SPEC_PATHS else []
        if name in PREREQUISITES:
            base = f"D-path-{PREREQUISITES[name][-1]}"
        elif spec:
            base = f"D-reproducible-{kv}-spec-c1"
        else:
            base = f"D-reproducible-{kv}-c{conc}"  # request-invariant: its bits at any concurrency
        steps.append(_arm(config, f"D-path-{name}", "D", reproducible + flags + spec, "path", c, base=base, path=name))
    for engine in config["engines"]:
        steps.append(_engine_step(config, engine))
    return steps, notes


def _default_plan(config: dict, steps: list[Step]) -> list[Step]:
    """arms "default": the configuration as shipped, every default fast path on, against the floor: the prompt path,
    and decoding with the configuration's drafting (no one decodes without it) at one request and at `concurrency`
    (the batched paths act only with several in flight). No reproducible control and no path-by-path arms."""
    kv, conc = config["kv"], config["concurrency"]
    steps.append(_arm(config, f"P-default-{kv}", "P", ["--kv-dtype", kv], "default"))
    if config["decode"] and config["spec"]:
        spec = ["--kv-dtype", kv, *config["spec"]]
        steps.append(_arm(config, f"D-default-{kv}-spec-c1", "D", spec, "default", 1))
        if not config["decode_cuda_graph"]:
            # The arms capture rows on eager rounds: the observed form with and without CUDA graphs must agree bit
            # for bit, so the eager arms measure the graphs' arithmetic.
            check = spec + ["--observed-only", "--chunks", str(40 if config["chunks"] == 0 else min(40, config["chunks"]))]
            steps.append(_arm(config, f"D-default-{kv}-spec-c1-graphs", "D", check + ["--cuda-graph"], "graphs", 1,
                              base=f"D-default-{kv}-spec-c1-eager"))
            steps.append(_arm(config, f"D-default-{kv}-spec-c1-eager", "D", check, "graphs", 1))
        steps.append(_arm(config, f"D-default-{kv}-spec-c{conc}", "D", spec, "default", conc))
    for engine in config["engines"]:
        steps.append(_engine_step(config, engine))
    return steps


def _engine_step(config: dict, engine: dict) -> Step:
    step_id = "E-" + "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in engine["name"])
    prefix = _work(config) / step_id
    protocol = engine.get("protocol", "P")
    kind = engine["kind"]
    if kind == "llama-perplexity":
        # Its KL mode scores every chunk of its base: base_chunks N gives it a base of the first N.
        base = _base(config) if "base_chunks" not in engine else _work(config) / f"reference.c{engine['base_chunks']}.kld"
        argv = [engine["binary"], "-m", engine["model"], "-c", str(config["context"]),
                "--kl-divergence-base", str(base), "--kl-divergence", *engine.get("args", [])]
    elif kind == "rows-command":
        # {base} the reference's path, {base_wsl} the same as WSL sees it (an engine that runs in WSL).
        from tools.quality.engines.basalt import wsl_path

        argv = [part.format(base=_base(config), base_wsl=wsl_path(_base(config)), context=config["context"],
                            chunks=config["chunks"]) for part in engine["argv"]]
    elif kind == "positions":
        argv = []
    else:
        raise ValueError(f"engine {engine['name']}: unknown kind {kind}")
    return Step(step_id, "engine", protocol, argv, engine.get("minutes", MINUTES["E"]), engine=engine,
                meta={"prefix": str(prefix)})


# ------------------------------------------------------------------------------------------------ running

def _log(config: dict, record: dict) -> None:
    record = {"time": dt.datetime.now().isoformat(timespec="seconds"), **record}
    with (_work(config) / "status.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _run_engine(config: dict, step: Step, log) -> int:
    engine, prefix = step.engine, Path(step.meta["prefix"])
    # Rows are rounded by the scorer; a tool that scores inside itself (llama-perplexity, Basalt's ppl_gate)
    # keeps its own logits precision unless its build rounds them (the engine entry's "logits").
    logits = config["logits"] if engine["kind"] == "rows-command" else engine.get("logits", "fp32")
    meta = {"arm": "engine", "engine": engine["name"], "kind": engine["kind"], "protocol": step.protocol,
            "settings": engine.get("settings", ""), "logits": logits}
    # The engine's own environment (a patched build's switches, e.g. LLAMA_KLD_ROUND_BF16).
    env = dict(os.environ, **{key: str(value) for key, value in engine.get("env", {}).items()})
    started = time.monotonic()
    if engine["kind"] == "llama-perplexity":
        if "base_chunks" in engine:
            base = Path(step.argv[step.argv.index("--kl-divergence-base") + 1])
            if not base.exists():
                llama_kld.head(_base(config), int(engine["base_chunks"]), base)
            meta["chunks"] = int(engine["base_chunks"])
        result = subprocess.run(step.argv, stdout=log, stderr=subprocess.STDOUT, timeout=step.minutes * 60, env=env)
        if result.returncode != 0:
            return result.returncode
        text = Path(log.name).read_text(encoding="utf-8", errors="replace")
        start = text.find("====== Perplexity statistics")
        if start < 0:
            log.write(b"no llama-perplexity summary in the output\n")
            return 3
        Path(str(prefix) + ".summary.txt").write_text(text[start:], encoding="utf-8")
        meta["printout"] = str(prefix) + ".summary.txt"
    elif engine["kind"] == "rows-command":
        scorer = [config["certify"], "llama-rows", "--llama-base", str(_base(config)), "--rows", "-",
                  "--out", str(prefix) + ".rows", "--label", engine["name"], "--chunks", str(config["chunks"]),
                  "--round-logits", "bf16" if config["logits"] == "bf16" else "none"]
        producer = subprocess.Popen(step.argv, stdout=subprocess.PIPE, stderr=log, env=env)
        consumer = subprocess.Popen(scorer, stdin=producer.stdout, stdout=log, stderr=log)
        producer.stdout.close()
        try:
            code = consumer.wait(timeout=step.minutes * 60)
            if code != 0:
                producer.kill()
                return code
            code = producer.wait(timeout=600)
        except subprocess.TimeoutExpired:
            producer.kill()
            consumer.kill()
            return 124
        if code != 0:
            return code
        meta["positions"] = str(prefix) + ".rows"
    else:  # positions an engine's own tool wrote (an adapter's conversion)
        meta["positions"] = engine["prefix"]
        if not Path(engine["prefix"] + ".positions.csv").exists():
            return 2
    meta["seconds"] = time.monotonic() - started
    Path(str(prefix) + ".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


def run(config: dict, dry_run: bool = False, only: set[str] | None = None) -> int:
    work = _work(config)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    failed: set[str] = set()
    while True:
        steps, notes = plan(config)
        pending = [s for s in steps if not _done(config, s) and s.id not in failed and (only is None or s.id in only)]
        if dry_run:
            for note in notes:
                print(f"# {note}")
            for s in steps:
                state = "done" if _done(config, s) else "pending"
                print(f"{state:8s} {s.id:40s} {subprocess.list2cmdline(s.argv)}")
            return 0
        if not pending:
            for note in notes:
                print(f"# {note}")
            return 1 if failed else 0
        step = pending[0]
        if step.id != "reference" and not _done(config, steps[0]):
            print("the reference is missing: nothing else can run", file=sys.stderr)
            return 1
        print(f"[{dt.datetime.now():%H:%M:%S}] {step.id}", flush=True)
        _log(config, {"step": step.id, "event": "start", "argv": step.argv})
        started = time.monotonic()
        with (work / "logs" / f"{step.id}.log").open("wb") as log:
            try:
                if step.role == "engine":
                    code = _run_engine(config, step, log)
                else:
                    code = subprocess.run(step.argv, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=step.minutes * 60).returncode
            except subprocess.TimeoutExpired:
                code = 124
        _log(config, {"step": step.id, "event": "end", "code": code, "seconds": round(time.monotonic() - started, 1)})
        if code != 0:
            print(f"  {step.id} failed ({code}); see {work / 'logs' / (step.id + '.log')}", file=sys.stderr)
            failed.add(step.id)
            if step.role == "reference":
                return 1


# ------------------------------------------------------------------------------------------------ report

LABELS = {"ppl": "PPL", "ln_ppl_ratio": "ln(PPL/PPL ref)", "mean_kld": "Mean KLD", "p999_kld": "KLD 99.9%",
          "same_top": "Top-1 agreement"}


def _arm_stats(config: dict, arm_id: str, resamples: int) -> tuple[llama_kld.Arm | None, dict | None]:
    meta = _read(config, arm_id)
    if meta is None:
        return None, None
    if meta.get("arm") == "engine":
        if "printout" in meta:
            return None, llama_kld.printout_point(Path(meta["printout"]).read_text(encoding="utf-8"))
        arm = llama_kld.load(Path(meta["positions"]))
        return arm, llama_kld.table(arm, resamples)
    arm = llama_kld.load(_work(config) / arm_id)
    return arm, llama_kld.table(arm, resamples)


def _floor(twins: list[dict]) -> dict[str, float]:
    """The noise twin's values: mean KLD, KLD 99.9 %, and top-1 disagreement, averaged over seeds."""
    return {"mean_kld": float(np.mean([t["mean_kld"]["value"] for t in twins])),
            "p999_kld": float(np.mean([t["p999_kld"]["value"] for t in twins])),
            "disagreement": float(np.mean([1 - t["same_top"]["value"] for t in twins]))}


def verdict(increment: dict, floor: dict[str, float], budget: float, identical: bool) -> dict:
    """The decision rule on a path's paired increment (llama_kld.paired of the path's arm against its base)."""
    if identical:
        return {"label": "no effect", "ratios": {}}
    upper = {"mean_kld": increment["mean_kld"]["high"], "p999_kld": increment["p999_kld"]["high"],
             # Disagreement rises when agreement falls: its upper bound is the agreement's lower bound, negated.
             "disagreement": -increment["same_top"]["low"], "ln_ppl_ratio": increment["ln_ppl_ratio"]["high"]}
    limits = {"mean_kld": budget * floor["mean_kld"], "p999_kld": budget * floor["p999_kld"],
              "disagreement": budget * floor["disagreement"], "ln_ppl_ratio": budget * floor["mean_kld"]}
    ratios = {m: (upper[m] / (limits[m] / budget) if limits[m] > 0 else float("inf")) for m in upper}
    within = all(upper[m] <= limits[m] for m in upper)
    return {"label": "rounding-level" if within else "changes outputs", "upper": upper, "limits": limits, "ratios": ratios}


def _cell(metric: str, row: dict | None) -> str:
    if row is None or metric not in row:
        return "-"
    r = row[metric]
    scale, digits = (100.0, 3) if metric == "same_top" else (1.0, 6)
    value = f"{r['value'] * scale:.{digits}f}" + (" %" if metric == "same_top" else "")
    if np.isnan(r["low"]):
        return value
    return f"{value} [{r['low'] * scale:.{digits}f}, {r['high'] * scale:.{digits}f}]"


def _delta(metric: str, row: dict) -> str:
    r = row[metric]
    scale, digits = (100.0, 3) if metric == "same_top" else (1.0, 6)
    return f"{r['value'] * scale:+.{digits}f} [{r['low'] * scale:+.{digits}f}, {r['high'] * scale:+.{digits}f}]"


ROW_NAMES = {"twin": "Rounding floor (noise twin, seed {seed})", "reproducible": "Infernix, fast paths off ({kv} KV{extra})",
             "default": "Infernix default ({kv} KV{extra})"}


def _describe(step: Step, config: dict) -> str:
    if step.role == "twin":
        return ROW_NAMES["twin"].format(seed=step.id.rsplit("-s", 1)[1])
    if step.role == "engine":
        settings = step.engine.get("settings")
        return step.engine["name"] + (f" ({settings})" if settings else "")
    kv = step.argv[step.argv.index("--kv-dtype") + 1] if "--kv-dtype" in step.argv else "bf16"
    extra = ""
    if "--concurrency" in step.argv:
        c = step.argv[step.argv.index("--concurrency") + 1]
        extra += f", {c} at once" if c != "1" else ", one request"
    if "--spec" in step.argv:
        extra += ", " + step.argv[step.argv.index("--spec") + 1] + " drafting"
    return ROW_NAMES[step.role].format(kv=kv, extra=extra)


def report(config: dict, resamples: int = llama_kld.RESAMPLES) -> dict:
    steps, notes = plan(config)
    ref = _read(config, "reference.kld") or {}
    stats: dict[str, dict] = {}
    arms: dict[str, llama_kld.Arm] = {}
    for step in steps:
        if step.role == "reference":
            continue
        arm, table = _arm_stats(config, step.id, resamples)
        if table is not None:
            stats[step.id] = table
        if arm is not None:
            arms[step.id] = arm
    twins = [stats[s.id] for s in steps if s.role == "twin" and s.id in stats]
    floor = _floor(twins) if twins else None
    out = {"name": config["name"], "artifact": config["artifact"], "context": config["context"],
           "reference": ref, "floor": floor, "budget": config["budget"], "arms": {}, "paths": [], "checks": [], "notes": notes}
    lines = [f"# {config['name']}: engines against the exact model", "",
             f"WikiText-2 test, {ref.get('chunks', '?')} chunks of {config['context']} tokens, the second half of each "
             f"scored ({next(iter(arms.values())).kld.size if arms else '?'} positions); the reference is Infernix's "
             f"lossless route (every operation correctly rounded, {config['logits'].upper()} logits, BF16 KV) on "
             f"`{config['artifact']}`, "
             "as a llama.cpp `--kl-divergence-base` file. Statistics are llama-perplexity's; brackets are 95 % "
             "intervals (a bootstrap over chunks, or llama.cpp's own ±).", ""]
    for protocol, title in (("P", "Prompt path (llama-perplexity -b n_ctx)"),
                            ("D", "Decode path (first half prefilled, the rest decoded token by token)")):
        rows = [s for s in steps if s.protocol == protocol and s.role in ("twin", "reproducible", "default", "engine")
                and s.id in stats]
        if not rows:
            continue
        lines += [f"## {title}", "", "| Engine / route | " + " | ".join(LABELS[m] for m in llama_kld.METRICS) + " |",
                  "|---|" + "---|" * len(llama_kld.METRICS)]
        for step in rows:
            out["arms"][step.id] = {"label": _describe(step, config), "protocol": protocol, "stats": stats[step.id],
                                    "seconds": (_read(config, step.id) or {}).get("seconds")}
            lines.append(f"| {_describe(step, config)} | " + " | ".join(_cell(m, stats[step.id]) for m in llama_kld.METRICS) + " |")
        lines.append("")
    if floor is not None:
        defaults = [s for s in steps if s.role == "default" and s.id in stats]
        if defaults:
            lines += ["## Infernix defaults against the floor", "",
                      f"Each default arm's statistic divided by the noise twin's (mean of {len(twins)} seeds: mean KLD "
                      f"{floor['mean_kld']:.6f}, KLD 99.9% {floor['p999_kld']:.6f}, top-1 disagreement "
                      f"{100 * floor['disagreement']:.3f} %); brackets divide the arm's 95 % interval by the floor.", "",
                      "| Route | Mean KLD / floor | KLD 99.9% / floor | Top-1 disagreement / floor |", "|---|---|---|---|"]
            for step in defaults:
                s = stats[step.id]
                ratios = {"mean_kld": tuple(s["mean_kld"][k] / floor["mean_kld"] for k in ("value", "low", "high")),
                          "p999_kld": tuple(s["p999_kld"][k] / floor["p999_kld"] for k in ("value", "low", "high")),
                          # Disagreement's interval: 1 - agreement's, so its low end is the agreement's high end.
                          "disagreement": tuple((1 - s["same_top"][k]) / floor["disagreement"]
                                                for k in ("value", "high", "low"))}
                out.setdefault("default_ratios", {})[step.id] = ratios
                lines.append(f"| {_describe(step, config)} | " + " | ".join(
                    f"{v:.2f}" if np.isnan(lo) else f"{v:.2f} [{lo:.2f}, {hi:.2f}]" for v, lo, hi in ratios.values()) + " |")
            lines.append("")
        path_steps = [s for s in steps if s.role == "path" and s.id in arms and s.base in arms]
        if path_steps:
            lines += ["## Fast paths, one at a time", "",
                      f"Each path's increment over the same route without it (paired on the same positions). Floor: the "
                      f"noise twin (mean KLD {floor['mean_kld']:.6f}, KLD 99.9% {floor['p999_kld']:.6f}, top-1 "
                      f"disagreement {100 * floor['disagreement']:.3f} %); rounding-level: every upper bound at most "
                      f"{config['budget']} x the floor.", "",
                      "| Path | Path of | Δ Mean KLD | Δ KLD 99.9% | Δ Top-1 agreement (pt) | Δ ln PPL | Worst ratio to floor | Verdict |",
                      "|---|---|---|---|---|---|---|---|"]
            for step in path_steps:
                same = llama_kld.identical(arms[step.id], arms[step.base])
                inc = llama_kld.paired(arms[step.id], arms[step.base], resamples)
                v = verdict(inc, floor, config["budget"], same)
                worst = max(v["ratios"].values()) if v["ratios"] else 0.0
                out["paths"].append({"path": step.path, "protocol": step.protocol, "arm": step.id, "base": step.base,
                                     "increment": inc, "verdict": v})
                lines.append(f"| {step.path} | {'prompt' if step.protocol == 'P' else 'decode'} | {_delta('mean_kld', inc)} | "
                             f"{_delta('p999_kld', inc)} | {_delta('same_top', inc)} | {_delta('ln_ppl_ratio', inc)} | "
                             f"{worst:.2f} | {v['label']} |")
            lines.append("")
    lines += ["## Checks", ""]
    for step in steps:
        if step.role == "invariance" and step.id in arms and step.base in arms:
            a, b = arms[step.id], arms[step.base]
            n = a.kld.size
            same = bool(np.array_equal(a.kld, b.kld[:n]) and np.array_equal(a.nll, b.nll[:n]) and
                        np.array_equal(a.same_top, b.same_top[:n]))
            out["checks"].append({"check": "request invariance", "arms": [step.id, step.base], "passed": same})
            lines.append(f"- Request invariance (fast paths off, one request against {step.base.rsplit('-c', 1)[1]} at "
                         f"once, first {np.unique(a.chunk).size} chunks): {'bit-identical' if same else 'DIFFERENT'}.")
        if step.role == "graphs" and step.base is not None and step.id in arms and step.base in arms:
            same = llama_kld.identical(arms[step.id], arms[step.base])
            out["checks"].append({"check": "cuda graphs", "arms": [step.id, step.base], "passed": same})
            lines.append(f"- CUDA graphs (the decode arms capture rows on eager rounds): the default route's observed "
                         f"form with graphs and without, first {np.unique(arms[step.id].chunk).size} chunks: "
                         f"{'bit-identical' if same else 'DIFFERENT'}.")
    for step in steps:
        meta = _read(config, step.id) if step.role == "engine" else None
        if meta is not None and meta.get("logits") != config["logits"]:
            lines.append(f"- {step.engine['name']}: scored with its own {meta.get('logits', '?').upper()} logits, not "
                         f"the comparison's {config['logits'].upper()} (its tool scores inside itself).")
    for note in notes:
        lines.append(f"- {note}")
    out["markdown"] = "\n".join(lines) + "\n"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        if name == "run":
            p.add_argument("--only", help="comma-separated step ids")
        if name == "report":
            p.add_argument("--resamples", type=int, default=llama_kld.RESAMPLES)
    args = parser.parse_args(argv)
    config = _load_config(args.config)
    if args.command == "plan":
        return run(config, dry_run=True)
    if args.command == "run":
        return run(config, only=set(args.only.split(",")) if args.only else None)
    result = report(config, args.resamples)
    work = _work(config)
    (work / "report.md").write_text(result["markdown"], encoding="utf-8")
    (work / "report.json").write_text(json.dumps({k: v for k, v in result.items() if k != "markdown"}, indent=2) + "\n",
                                      encoding="utf-8")
    print(result["markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
