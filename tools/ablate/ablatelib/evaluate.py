"""eval: does the candidate still do what the stock model does? The stock model's results are cached per setup.

Tasks (each the same protocol on both arms; the engines are the run's, at --context, default 16384):
  kl          first-answer-token KL(stock || candidate) on harmless prompts (thinking off, top-256): mean, median,
              max, and how often the top token changes. Use prompts the fit never saw (--kl-prompts).
  gsm8k       lm-eval gsm8k, 0-shot, thinking on, up to 7168 new tokens
  ifeval      lm-eval leaderboard_ifeval, thinking on, up to 4096 new tokens
  truthfulqa  lm-eval truthfulqa (mc1/mc2 by log-likelihood through /v1/score with thinking off; gen with thinking on)
  humaneval   HumanEval+ (EvalPlus v0.1.10, downloaded once and checksum-pinned), thinking on; the model's code
              runs in a bubblewrap sandbox (no network, read-only system, its own process namespace, 90 s); where
              bubblewrap is unavailable (Windows, or Linux without it) it refuses unless --unsandboxed-code
How lm-eval meets Infernix: there is no /v1/completions, so lm-eval runs with the chat template and the engine
renders it; log-likelihoods come from /v1/score (a multiple-choice question whose choices are one token each is read
from one top-256 call); generation is greedy chat completions, stop strings cut client-side.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from . import measure as M
from .engine import MODEL_ID, Engine, EngineError
from .prompts import read_prompts
from .run import sha256_file, write_json

LM_TASKS = {"gsm8k": {"tasks": ["gsm8k"], "num_fewshot": 0, "max_gen_toks": 7168},
            "ifeval": {"tasks": ["leaderboard_ifeval"], "num_fewshot": None, "max_gen_toks": 4096},
            "truthfulqa": {"tasks": ["truthfulqa"], "num_fewshot": None, "max_gen_toks": 4096}}
HUMANEVAL_URL = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"
HUMANEVAL_SHA = "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101"
ASK = ("Complete the following Python function. Reply with the complete implementation (the function, plus any "
       "imports or helpers it needs) in a single ```python code block.\n\n```python\n{prompt}```")


class Arm:
    """One engine at a time for one arm, switching purpose (generate | score) on demand."""

    def __init__(self, run, sidecar: Path | None, out: Path, context: int):
        self.run, self.sidecar, self.out, self.context = run, sidecar, out, context
        self.out.mkdir(parents=True, exist_ok=True)   # the adapters write here before the first engine starts
        self.engine: Engine | None = None

    def get(self, purpose: str) -> Engine:
        if self.engine is None or self.engine.purpose != purpose:
            self.close()
            self.engine = Engine(self.run, purpose, sidecar=self.sidecar, context=self.context,
                                 log_path=self.out / f"engine.{purpose}.log")
            self.engine.start()
        return self.engine

    def close(self) -> None:
        if self.engine is not None:
            self.engine.stop()
            self.engine = None


def make_lm(arm: Arm, gen_log: Path):
    from lm_eval.api.model import LM
    from lm_eval.models.api_models import JsonChatStr

    class InfernixLM(LM):
        def __init__(self):
            super().__init__()
            self.tok: dict[str, list[int]] = {}

        @property
        def tokenizer_name(self) -> str:
            return "infernix"

        def chat_template(self, chat_template=False):
            return "infernix"

        def apply_chat_template(self, chat_history, add_generation_prompt=True):  # type: ignore[override]
            if not add_generation_prompt:
                raise NotImplementedError("assistant prefill has no Infernix route that means the same thing")
            return JsonChatStr(json.dumps(chat_history, ensure_ascii=False))

        def _score(self, ctx, continuation: str, top_k: int) -> list[dict]:
            if not isinstance(ctx, JsonChatStr):
                raise NotImplementedError("raw-text scoring: run lm-eval with the chat template")
            out = arm.get("score").post("/v1/score", {"model": MODEL_ID, "apply_chat_template": True,
                                                      "messages": json.loads(ctx.prompt), "enable_thinking": False,
                                                      "continuation": continuation, "top_k": top_k})
            positions = out.get("positions") or []
            if out.get("chat_template_applied") is not True:
                raise EngineError("/v1/score did not apply the chat template")
            if out.get("enable_thinking") is not False or not positions or \
                    any(len(p.get("top_logprobs") or []) != top_k for p in positions):
                raise EngineError(f"/v1/score answered an unexpected shape for {continuation!r}")
            self.tok[continuation] = [p["token_id"] for p in positions]
            return positions

        @staticmethod
        def _ll(positions):
            return (math.fsum(p["logprob"] for p in positions),
                    all(p["token_id"] == p["top_logprobs"][0]["token_id"] for p in positions))

        def loglikelihood(self, requests, disable_tqdm=False):
            out: list = [None] * len(requests)
            groups: dict = {}
            for i, r in enumerate(requests):
                ctx, cont = r.args
                groups.setdefault(ctx.prompt if isinstance(ctx, JsonChatStr) else ctx, []).append((i, ctx, cont))
            for items in groups.values():
                fast = len(items) > 1
                first_i, ctx, cont = items[0]
                pos = self._score(ctx, cont, M.TOP_K if fast else 1)
                out[first_i] = self._ll(pos)
                fast = fast and len(pos) == 1
                top0 = {t["token_id"]: t["logprob"] for t in pos[0]["top_logprobs"]} if fast else {}
                best = pos[0]["top_logprobs"][0]["token_id"] if fast else None
                for i, ctx_i, cont_i in items[1:]:
                    ids = self.tok.get(cont_i)
                    if fast and ids is not None and len(ids) == 1 and ids[0] in top0:
                        out[i] = (top0[ids[0]], ids[0] == best)
                    else:
                        out[i] = self._ll(self._score(ctx_i, cont_i, 1))
            return out

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError("not needed by these tasks")

        def generate_until(self, requests, disable_tqdm=False):
            results = []
            with gen_log.open("a", encoding="utf-8") as logf:
                for r in requests:
                    ctx, kw = r.args
                    if not isinstance(ctx, JsonChatStr):
                        raise NotImplementedError("raw-text generation: Infernix has no /v1/completions")
                    until = kw.get("until") or []
                    until = [until] if isinstance(until, str) else list(until)
                    max_tokens = min(int(kw.get("max_gen_toks", 4096)), arm.context - 1024)
                    reply = arm.get("generate").post("/v1/chat/completions", {
                        "model": MODEL_ID, "messages": json.loads(ctx.prompt), "max_tokens": max_tokens,
                        "temperature": 0.0, "enable_thinking": True}, timeout=3600)
                    choice = reply["choices"][0]
                    content = (choice.get("message") or {}).get("content") or ""
                    for stop in until:
                        if stop and stop in content:
                            content = content[:content.index(stop)]
                    logf.write(json.dumps({"task": getattr(r, "task_name", None), "doc_id": getattr(r, "doc_id", None),
                                           "finish_reason": choice.get("finish_reason"), "max_tokens": max_tokens,
                                           "content": content}, ensure_ascii=False) + "\n")
                    results.append(content)
            return results

    return InfernixLM()


def run_lm_task(arm: Arm, task: str, limit: int | None) -> dict:
    import lm_eval
    spec = LM_TASKS[task]
    res = lm_eval.simple_evaluate(  # type: ignore[call-arg]  (lm-eval 0.4.13's signature)
        model=make_lm(arm, arm.out / "generations.jsonl"), tasks=spec["tasks"],
                                  num_fewshot=spec["num_fewshot"], limit=limit, apply_chat_template=True,
                                  fewshot_as_multiturn=True, gen_kwargs={"max_gen_toks": spec["max_gen_toks"]},
                                  bootstrap_iters=1000)
    if not res:
        raise EngineError(f"lm-eval returned nothing for {task}")
    metrics = {}
    for name, values in (res.get("results") or {}).items():
        for key, value in values.items():
            if "," in key and not key.startswith("alias") and isinstance(value, (int, float)):
                metric, _, filt = key.partition(",")
                if metric.endswith("_stderr"):
                    continue
                stderr = values.get(f"{metric}_stderr,{filt}")
                metrics[f"{name}/{metric}/{filt}"] = {"value": value, "stderr": stderr
                                                      if isinstance(stderr, (int, float)) else None}
    truncated = sum(1 for line in (arm.out / "generations.jsonl").read_text(encoding="utf-8").splitlines()
                    if json.loads(line).get("finish_reason") == "length") if (arm.out / "generations.jsonl").is_file() else 0
    return {"metrics": metrics, "truncated": truncated, "lm_eval": lm_eval.__version__, "limit": limit}


def humaneval_problems(run) -> list[dict]:
    path = run.sub("eval", "data") / "HumanEvalPlus.jsonl.gz"
    if not path.is_file():
        with urllib.request.urlopen(HUMANEVAL_URL, timeout=300) as reply:
            path.write_bytes(reply.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != HUMANEVAL_SHA:
        raise SystemExit(f"{path}: sha256 {digest}, expected {HUMANEVAL_SHA} (EvalPlus v0.1.10); delete it to retry")
    with gzip.open(path, "rt", encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


RUNNER = r'''
import json, math, sys, copy
sys.setrecursionlimit(10000)
P = json.load(open("problem.json"))
def close(a, b, atol):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=atol) or (math.isnan(a) and math.isnan(b))
        except TypeError:
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(close(x, y, atol) for x, y in zip(a, b))
    return a == b
ns_c, ns_r = {}, {}
exec(P["prompt"], ns_c)
exec(P["candidate"], ns_c)
exec(P["prompt"] + P["canonical"], ns_r)
cand, ref = ns_c[P["entry"]], ns_r[P["entry"]]
ns_t = dict(ns_c)
exec(P["test"], ns_t)
try:
    ns_t["check"](cand); base = True
except BaseException:
    base = False
plus = base
if base:
    atol = P["atol"] or 1e-6
    for inp in P["plus_input"]:
        try:
            want = ref(*copy.deepcopy(inp))
        except BaseException:
            continue
        try:
            got = cand(*copy.deepcopy(inp))
        except BaseException:
            plus = False; break
        if not close(got, want, atol):
            plus = False; break
print("RESULT", json.dumps({"base": base, "plus": plus}))
'''


def sandbox_argv(workdir: str, unsandboxed: bool) -> list[str]:
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        return ["bwrap", "--tmpfs", "/", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc",
                "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                "--tmpfs", "/tmp", "--bind", workdir, "/work", "--dev", "/dev", "--proc", "/proc", "--unshare-net",
                "--unshare-pid", "--die-with-parent", "--chdir", "/work", "python3", "runner.py"]
    if unsandboxed:
        return [sys.executable, "runner.py"]
    raise SystemExit("humaneval runs the model's code, and bubblewrap (bwrap) is not available here to contain it; "
                     "install it (Linux), or pass --unsandboxed-code to run it as you with only a timeout")


def run_humaneval(arm: Arm, limit: int | None, unsandboxed: bool) -> dict:
    problems = humaneval_problems(arm.run)[:limit] if limit else humaneval_problems(arm.run)
    sandbox_argv(".", unsandboxed)       # refuse before any generation if code cannot run here
    max_tokens = arm.context - 1024
    rows = []
    with (arm.out / "humaneval.jsonl").open("w", encoding="utf-8") as out:
        for p in problems:
            choice = arm.get("generate").chat(ASK.format(prompt=p["prompt"]), max_tokens=max_tokens,
                                              enable_thinking=True, timeout=3600)
            text = (choice.get("message") or {}).get("content") or ""
            blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
            imports = "\n".join(line for line in p["prompt"].splitlines() if line.startswith(("import ", "from ")))
            code = imports + "\n\n" + (blocks[-1] if blocks else text)
            with tempfile.TemporaryDirectory(prefix="he-") as d:
                Path(d, "problem.json").write_text(json.dumps({
                    "candidate": code, "prompt": p["prompt"], "canonical": p["canonical_solution"],
                    "entry": p["entry_point"], "test": p["test"], "atol": p.get("atol") or 0,
                    "plus_input": p["plus_input"]}), encoding="utf-8")
                Path(d, "runner.py").write_text(RUNNER, encoding="utf-8")
                try:
                    r = subprocess.run(sandbox_argv(d, unsandboxed), cwd=d, capture_output=True, text=True, timeout=90)
                    m = re.search(r"RESULT (\{.*\})", r.stdout)
                    res = json.loads(m.group(1)) if m else {"base": False, "plus": False}
                except subprocess.TimeoutExpired:
                    res = {"base": False, "plus": False}
            row = {"task_id": p["task_id"], **res, "finish_reason": choice.get("finish_reason")}
            rows.append(row)
            out.write(json.dumps(row) + "\n")
    n = len(rows)
    return {"n": n, "base": sum(r["base"] for r in rows), "plus": sum(r["plus"] for r in rows),
            "cap_hits": sum(r["finish_reason"] == "length" for r in rows), "max_tokens": max_tokens,
            "per_problem": {r["task_id"]: r["plus"] for r in rows}}


def run_kl(arm_stock: Arm, arm_cand: Arm, prompts: list[str]) -> dict:
    stock = M.first_token_distributions(arm_stock.get("score"), prompts)
    arm_stock.close()
    cand = M.first_token_distributions(arm_cand.get("score"), prompts)
    kls = M.kl(stock, cand)
    flips = sum(max(s, key=lambda t: s[t]) != max(c, key=lambda t: c[t]) for s, c in zip(stock, cand))
    return {"n": len(prompts), "mean": M.mean(kls), "median": statistics.median(kls), "max": max(kls),
            "top_token_changed": flips}


def mcnemar(a: dict[str, bool], b: dict[str, bool]) -> dict:
    only_a = sum(1 for k in a if a[k] and not b.get(k))
    only_b = sum(1 for k in a if b.get(k) and not a[k])
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n)
    return {"stock_only": only_a, "candidate_only": only_b, "p_two_sided": p}


def evaluate(run, candidate: str, tasks: str, context: int | None, limit: int | None, kl_prompts: str | None,
             field: str | None, unsandboxed: bool = False, log=print) -> Path:
    from .candidates import candidate_sidecar
    sc = candidate_sidecar(run, candidate)
    context = int(context or run.context)
    wanted = [t.strip() for t in tasks.split(",") if t.strip()]
    unknown = sorted(set(wanted) - {"kl", "humaneval", *LM_TASKS})
    if unknown:
        raise SystemExit(f"unknown task(s) {unknown}")
    label = Path(candidate).parent.name if candidate.endswith(".json") else candidate
    out = run.sub("eval", f"{label}-{time.strftime('%Y%m%dT%H%M%S')}")
    # The stock results are reused while the engine's command line and binary are the same (an Infernix update is
    # another stock model).
    setup = hashlib.sha256(json.dumps([Engine(run, "generate", context=context).argv(port=1), limit,
                                       sha256_file(run.serve_binary)]).encode()).hexdigest()[:16]
    stock_dir = run.sub("eval", "stock", setup)
    results: dict = {"candidate": str(sc), "context": context, "limit": limit, "stock_cache": str(stock_dir), "tasks": {}}
    for task in wanted:
        log(f"== {task}")
        t0 = time.time()
        if task == "kl":
            prompts = read_prompts(Path(kl_prompts), field)[0] if kl_prompts else \
                json.loads((run.dir / "prompts" / "sets.json").read_text(encoding="utf-8"))["held_out"]["harmless"]
            a, b = Arm(run, None, out / "kl-stock", context), Arm(run, sc, out / "kl-candidate", context)
            try:
                results["tasks"]["kl"] = run_kl(a, b, prompts)
            finally:
                a.close(); b.close()
        else:
            arms = {}
            for label, sidecar_path, base in (("stock", None, stock_dir), ("candidate", sc, out)):
                cached = base / f"{task}.json"
                if label == "stock" and cached.is_file():
                    arms[label] = json.loads(cached.read_text(encoding="utf-8"))
                    log(f"  stock: cached ({cached})")
                    continue
                arm = Arm(run, sidecar_path, base / task, context)
                try:
                    arms[label] = run_humaneval(arm, limit, unsandboxed) if task == "humaneval" else \
                        run_lm_task(arm, task, limit)
                finally:
                    arm.close()
                write_json(cached, arms[label])
            results["tasks"][task] = compare(task, arms["stock"], arms["candidate"], limited=bool(limit))
        log(f"  {task}: {time.time() - t0:.0f} s")
        write_json(out / "results.json", results)
    (out / "summary.md").write_text(summary_md(results), encoding="utf-8")
    return out


def compare(task: str, stock: dict, cand: dict, limited: bool = False) -> dict:
    if task == "humaneval":
        return {"stock": {k: stock[k] for k in ("n", "base", "plus", "cap_hits")},
                "candidate": {k: cand[k] for k in ("n", "base", "plus", "cap_hits")},
                "mcnemar_plus": mcnemar(stock["per_problem"], cand["per_problem"])}
    rows = {}
    for key, s in stock["metrics"].items():
        c = cand["metrics"].get(key)
        if c is None:
            continue
        band = (1.96 * math.sqrt(s["stderr"] ** 2 + c["stderr"] ** 2)
                if s.get("stderr") is not None and c.get("stderr") is not None else None)
        diff = c["value"] - s["value"]
        # A --limit sample is too small for its band to mean anything (3 right answers give a band of 0).
        rows[key] = {"stock": s["value"], "candidate": c["value"], "difference": diff,
                     "band_95": None if limited else band,
                     "reading": None if band is None or limited else ("within the noise band" if abs(diff) <= band
                                                                      else "outside the noise band")}
    return {"metrics": rows, "truncated": {"stock": stock.get("truncated"), "candidate": cand.get("truncated")}}


def summary_md(results: dict) -> str:
    lines = [f"# eval: {results['candidate']}", "", f"context {results['context']}"
             + (f", first {results['limit']} items per task (not comparable to full runs)" if results["limit"] else ""), ""]
    t = results["tasks"]
    if "kl" in t:
        k = t["kl"]
        lines += [f"**KL test** ({k['n']} prompts, first answer token): mean {k['mean']:.4f}, median {k['median']:.4f}, "
                  f"max {k['max']:.3f}; top token changed on {k['top_token_changed']}/{k['n']}", ""]
    if "humaneval" in t:
        h = t["humaneval"]
        lines += [f"**HumanEval+**: plus {h['stock']['plus']}/{h['stock']['n']} (stock) vs {h['candidate']['plus']}/"
                  f"{h['candidate']['n']}; base {h['stock']['base']} vs {h['candidate']['base']}; McNemar p = "
                  f"{h['mcnemar_plus']['p_two_sided']:.3g}; cap hits {h['stock']['cap_hits']} / {h['candidate']['cap_hits']}", ""]
    for task in LM_TASKS:
        if task in t:
            lines += [f"**{task}**", "", "| metric | stock | candidate | difference | 95% band | reading |", "|---|---|---|---|---|---|"]
            for key, r in t[task]["metrics"].items():
                band = "-" if r["band_95"] is None else f"{r['band_95']:.4f}"
                lines.append(f"| {key} | {r['stock']:.4f} | {r['candidate']:.4f} | {r['difference']:+.4f} | {band} | "
                             f"{r['reading'] or '-'} |")
            lines += ["", f"answers cut at the token limit: stock {t[task]['truncated']['stock']}, candidate "
                           f"{t[task]['truncated']['candidate']}", ""]
    return "\n".join(lines) + "\n"
