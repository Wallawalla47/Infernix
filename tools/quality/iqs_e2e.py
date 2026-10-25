"""IQS-2 end-to-end checks through infernix-serve (docs/maintainer/model-quality.md, section 6, block F).

Each check starts the server on one artifact, drives it through the OpenAI Chat Completions API like any client,
and writes a JSON result {"passed": bool, "detail": str, ...} that tools/quality/iqs.py's certificate collects.

  identity   G4/G5: on --reproducible, every response under concurrency and a warm prefix cache equals the same
             request served alone (greedy and seeded sampling); with --golden FILE the responses served alone are
             compared with the hashes FILE stores from an earlier build (recorded when FILE is absent): a
             difference is reported, not failed, since a build may change the route's bits by a numerics change
             its release declares (model-quality.md 6.1), which then records FILE again; with
             --negative-control the default route's differences under concurrency are counted and reported (an
             artifact whose default route happens to be request-invariant shows none: that says nothing against
             its reproducible route)
  sensitivity  the comparison above can see a difference: at least one artifact's negative control saw one
  longgen    G5: free-running thinking outputs on the default route against --reproducible: mean output length
             within 90-110 %, no more looping outputs, no fewer natural stops (MLPerf TEST06/TEST09 style)
  specdist   G5: with speculation the default route's sampled continuations follow the same distribution as
             without it (chi-square homogeneity, Bonferroni over prompts, alpha 0.001)
  tasks      a coarse check: GSM8K and MMLU accuracy of the default route against --reproducible; fails when
             the one-sided 97.5 % bound on the paired accuracy loss exceeds 5 points (GSM8K) or 3 (MMLU)

Usage: python -m tools.quality.iqs_e2e CHECK --serve EXE --artifact PATH --out RESULT.json [server options]
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import math
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "eval" / "corpora" / "iqs-1" / "tasks"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def server(exe: str, artifact: str, extra: list[str], log: Path, concurrency: int = 4, startup_s: int = 1800):
    port = free_port()
    args = [exe, artifact, "--port", str(port), "--max-concurrency", str(concurrency), "--log-level", "warning"] + extra
    with log.open("ab") as out:
        proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.time() + startup_s
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"infernix-serve exited {proc.returncode} during startup (see {log})")
                try:
                    with urllib.request.urlopen(base + "/v1/models", timeout=5) as response:
                        model = json.load(response)["data"][0]["id"]
                        break
                except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError):
                    if time.time() > deadline:
                        raise RuntimeError("infernix-serve did not become ready")
                    time.sleep(2)
            yield Client(base, model)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()


class Client:
    def __init__(self, base: str, model: str):
        self.base, self.model = base, model

    def chat(self, messages: list[dict], max_tokens: int, temperature: float = 0.0, seed: int | None = None,
             thinking: bool = False) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                "enable_thinking": thinking}
        if seed is not None:
            body["seed"] = seed
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3600) as response:
            data = json.load(response)
        choice = data["choices"][0]
        return {"content": choice["message"].get("content") or "",
                "reasoning": choice["message"].get("reasoning_content") or "",
                "finish_reason": choice.get("finish_reason"),
                "tokens": data.get("usage", {}).get("completion_tokens", 0)}

    def many(self, requests: list[dict], workers: int) -> list[dict]:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda r: self.chat(**r), requests))


def gsm8k(count: int, first: int = 0) -> list[dict]:
    lines = (TASKS / "gsm8k-250.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[first:first + count]]


def mmlu(count: int) -> list[dict]:
    lines = (TASKS / "mmlu-500.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines[:count]]


def identity_requests() -> tuple[list[dict], list[dict]]:
    """Two request sets: plain prompts, and the same behind one long shared system prompt (prefix-cache hits)."""
    items = gsm8k(8)
    shared = ("You are a careful assistant. " + " ".join(f"Rule {i}: state each step briefly and check the arithmetic."
                                                         for i in range(120)))
    plain, prefixed = [], []
    for i, item in enumerate(items):
        user = {"role": "user", "content": item["question"]}
        for target, messages in ((plain, [user]), (prefixed, [{"role": "system", "content": shared}, user])):
            target.append({"messages": messages, "max_tokens": 96})
            target.append({"messages": messages, "max_tokens": 96, "temperature": 0.8, "seed": 1000 + i})
    return plain, prefixed


def same(a: dict, b: dict) -> bool:
    return a["content"] == b["content"] and a["reasoning"] == b["reasoning"]


def check_identity(args) -> dict:
    plain, prefixed = identity_requests()
    differences = []
    hashes = []
    with server(args.serve, args.artifact, args.server_args + ["--reproducible"], args.log) as client:
        for name, requests in (("plain", plain), ("shared prefix", prefixed)):
            alone = [client.chat(**r) for r in requests]
            together = client.many(requests, workers=len(requests))
            again = client.many(requests[::-1], workers=4)[::-1]
            for i, (a, b, c) in enumerate(zip(alone, together, again)):
                if not same(a, b) or not same(a, c):
                    differences.append(f"{name} request {i}")
            hashes += [hashlib.sha256((o["reasoning"] + "\u0000" + o["content"]).encode()).hexdigest() for o in alone]
    result = {"passed": not differences, "requests": 2 * len(plain),
              "detail": "every response identical alone, together and reversed" if not differences
              else f"{len(differences)} differ: {', '.join(differences[:6])}", "differences": differences}
    if args.golden:
        result.update(golden(Path(args.golden), args.artifact, hashes))
        result["detail"] += "; " + result["golden"]["detail"]
    if args.negative_control:
        with server(args.serve, args.artifact, args.server_args, args.log) as client:
            alone = [client.chat(**r) for r in plain]
            together = client.many(plain, workers=len(plain))
            seen = sum(1 for a, b in zip(alone, together) if not same(a, b))
        result["negative_control"] = {"route": "default", "differences": seen}
        result["detail"] += f"; negative control (default route) saw {seen} differences"
    return result


def check_sensitivity(args) -> dict:
    """The identity comparison is shown able to see a difference when some artifact's default route, compared
    the same way, showed one: the comparison code is the same for every artifact."""
    seen = {}
    for path in args.results:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        control = result.get("negative_control")
        if control is not None:
            seen[Path(path).stem] = int(control.get("differences", 0))
    passed = any(count > 0 for count in seen.values())
    detail = ", ".join(f"{name}: {count}" for name, count in sorted(seen.items())) or "no negative control ran"
    return {"passed": passed, "controls": seen,
            "detail": ("the default route showed differences under concurrency (" + detail + ")") if passed
            else "no artifact's default route showed a difference, so the comparison is unproven (" + detail + ")"}


def golden(store: Path, artifact: str, hashes: list[str]) -> dict:
    """The reproducible route's responses against the hashes an earlier build stored (recorded when absent).
    Disclosure, not a criterion: a changed hash is a numerics change for the release to declare, which then
    records the new hashes this result carries."""
    if not store.is_file():
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps({"artifact": artifact, "hashes": hashes}, indent=1) + "\n", encoding="utf-8")
        return {"golden": {"equal": True, "recorded": True, "detail": f"recorded {len(hashes)} golden hashes"}}
    stored = json.loads(store.read_text(encoding="utf-8"))["hashes"]
    changed = [i for i, (a, b) in enumerate(zip(stored, hashes)) if a != b]
    equal = not changed and len(stored) == len(hashes)
    return {"golden": {"equal": equal, "changed": changed, "hashes": hashes,
                       "detail": f"golden hashes equal ({len(hashes)})" if equal
                       else f"{len(changed)} of {len(hashes)} reproducible responses changed since {store}: a "
                            "numerics change for the release to declare (model-quality.md 6.1)"}}


def looping(text: str, n: int = 24, repeats: int = 6) -> bool:
    words = text.split()
    counts: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        key = tuple(words[i:i + n])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= repeats:
            return True
    return False


def generate_long(client: Client, count: int, max_tokens: int) -> list[dict]:
    requests = [{"messages": [{"role": "user", "content": it["question"]}], "max_tokens": max_tokens, "thinking": True}
                for it in gsm8k(count, first=16)]
    return client.many(requests, workers=4)


def summarize(outputs: list[dict]) -> dict:
    return {"mean_tokens": sum(o["tokens"] for o in outputs) / len(outputs),
            "looping": sum(1 for o in outputs if looping(o["reasoning"] + " " + o["content"])),
            "stops": sum(1 for o in outputs if o["finish_reason"] == "stop")}


def check_longgen(args) -> dict:
    with server(args.serve, args.artifact, args.server_args + ["--reproducible"], args.log) as client:
        base = summarize(generate_long(client, args.count, args.max_tokens))
    with server(args.serve, args.artifact, args.server_args, args.log) as client:
        fast = summarize(generate_long(client, args.count, args.max_tokens))
    ratio = fast["mean_tokens"] / max(base["mean_tokens"], 1.0)
    passed = 0.9 <= ratio <= 1.1 and fast["looping"] <= base["looping"] and fast["stops"] >= base["stops"] - max(1, args.count // 10)
    return {"passed": passed, "reproducible": base, "default": fast, "length_ratio": ratio,
            "detail": f"length ratio {ratio:.3f}, looping {fast['looping']} vs {base['looping']}, "
                      f"stops {fast['stops']} vs {base['stops']} of {args.count}"}


def chi2_sf(x: float, k: int) -> float:
    """Upper tail of the chi-square distribution: the regularized gamma Q(k/2, x/2)."""
    if k <= 0 or x <= 0:
        return 1.0
    a, y = k / 2.0, x / 2.0
    if y < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(10000):
            n += 1
            term *= y / n
            total += term
            if term < total * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-y + a * math.log(y) - math.lgamma(a)))
    b, c, d = y + 1 - a, 1e300, 1 / (y + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(-y + a * math.log(y) - math.lgamma(a)) * h)


def homogeneity(a: list[str], b: list[str]) -> tuple[float, int]:
    """Chi-square homogeneity of two samples of categories (categories expected below 5 merged)."""
    na, nb = len(a), len(b)
    counts: dict[str, list[int]] = {}
    for value in a:
        counts.setdefault(value, [0, 0])[0] += 1
    for value in b:
        counts.setdefault(value, [0, 0])[1] += 1
    merged: list[list[int]] = []
    rest = [0, 0]
    for pair in counts.values():
        if (pair[0] + pair[1]) * min(na, nb) / (na + nb) < 5:
            rest[0] += pair[0]
            rest[1] += pair[1]
        else:
            merged.append(pair)
    if sum(rest):
        merged.append(rest)
    if len(merged) < 2:
        return 1.0, 0
    total = na + nb
    stat = 0.0
    for pair in merged:
        row = pair[0] + pair[1]
        for column, n in ((0, na), (1, nb)):
            expected = row * n / total
            stat += (pair[column] - expected) ** 2 / expected
    dof = len(merged) - 1
    return chi2_sf(stat, dof), dof


def sample(client: Client, prompts: list[str], samples: int) -> list[list[str]]:
    out = []
    for p, prompt in enumerate(prompts):
        requests = [{"messages": [{"role": "user", "content": prompt}], "max_tokens": 4, "temperature": 1.0,
                     "seed": 100000 * (p + 1) + s} for s in range(samples)]
        out.append([r["content"] for r in client.many(requests, workers=8)])
    return out


def check_specdist(args) -> dict:
    prompts = ["Write one word that describes the sea.", "Name a colour.",
               "Continue the story: The door opened and", "Give a random number between 1 and 50."]
    if not args.spec_args:
        raise ValueError("specdist needs the speculation arguments (--spec-arg)")
    # serve loads no speculative backend without --spec, so the plain arm is the server arguments alone
    with server(args.serve, args.artifact, args.server_args + args.spec_args, args.log, concurrency=8) as client:
        with_spec = sample(client, prompts, args.samples)
    with server(args.serve, args.artifact, args.server_args, args.log, concurrency=8) as client:
        without = sample(client, prompts, args.samples)
    tests = [homogeneity(a, b) for a, b in zip(with_spec, without)]
    alpha = 1e-3 / len(prompts)
    passed = all(p >= alpha for p, _ in tests)
    return {"passed": passed, "p_values": [p for p, _ in tests], "dof": [d for _, d in tests],
            "detail": "speculative sampling matches plain sampling: p " + ", ".join(f"{p:.3g}" for p, _ in tests)}


def gsm_answer(text: str) -> str | None:
    found = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return found[-1].replace(",", "").rstrip(".") if found else None


def run_tasks(client: Client, gsm_count: int, mmlu_count: int) -> tuple[list[bool], list[bool]]:
    gsm = gsm8k(gsm_count)
    gsm_out = client.many([{"messages": [{"role": "user", "content": it["question"] + "\nEnd with 'Answer: <number>'."}],
                            "max_tokens": 400} for it in gsm], workers=8)
    gsm_ok = [gsm_answer(o["content"]) == it["answer"].replace(",", "") for o, it in zip(gsm_out, gsm)]
    questions = mmlu(mmlu_count)
    prompts = []
    for q in questions:
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", q["choices"]))
        prompts.append({"messages": [{"role": "user", "content": f"{q['question']}\n{options}\nAnswer with the letter only."}],
                        "max_tokens": 8})
    mmlu_out = client.many(prompts, workers=8)
    mmlu_ok = []
    for o, q in zip(mmlu_out, questions):
        found = re.search(r"\b([ABCD])\b", o["content"])
        mmlu_ok.append(bool(found) and found.group(1) == q["answer"])
    return gsm_ok, mmlu_ok


def check_tasks(args) -> dict:
    with server(args.serve, args.artifact, args.server_args + ["--reproducible"], args.log, concurrency=8) as client:
        base = run_tasks(client, args.gsm8k, args.mmlu)
    with server(args.serve, args.artifact, args.server_args, args.log, concurrency=8) as client:
        fast = run_tasks(client, args.gsm8k, args.mmlu)
    result = {"passed": True, "tasks": {}}
    for name, a, b in (("gsm8k", base[0], fast[0]), ("mmlu", base[1], fast[1])):
        result["tasks"][name] = paired_loss(a, b, TASK_MARGINS[name])
        result["passed"] = result["passed"] and result["tasks"][name]["passed"]
    result["detail"] = "; ".join(f"{k} {v['default']:.3f} vs {v['reproducible']:.3f} ({v['flips']} flips of {v['items']}, "
                                 f"loss bound {v['loss_bound']:+.3f} of {v['margin']:.2f})" for k, v in result["tasks"].items())
    return result


# The accuracy a default route may lose on a task (one-sided 97.5 % bound): a coarse check for gross breakage.
TASK_MARGINS = {"gsm8k": 0.05, "mmlu": 0.03}


def paired_loss(base: list[bool], fast: list[bool], margin: float) -> dict:
    """Paired non-inferiority on the items the two routes answer differently: the default route's accuracy loss
    (b - c) / n, b the items only the reproducible route answers, c those only the default route answers, with
    the normal bound of a paired difference of proportions."""
    n = len(base)
    only_base = sum(1 for x, y in zip(base, fast) if x and not y)
    only_fast = sum(1 for x, y in zip(base, fast) if y and not x)
    loss = (only_base - only_fast) / n
    se = math.sqrt(max(only_base + only_fast - n * loss * loss, 0.0)) / n
    bound = loss + 1.96 * se
    return {"reproducible": sum(base) / n, "default": sum(fast) / n, "flips": only_base + only_fast, "items": n,
            "only_reproducible": only_base, "only_default": only_fast, "loss": loss, "loss_bound": bound,
            "margin": margin, "passed": bound <= margin}


CHECKS = {"identity": check_identity, "longgen": check_longgen, "specdist": check_specdist, "tasks": check_tasks,
          "sensitivity": check_sensitivity}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=sorted(CHECKS))
    parser.add_argument("--serve")
    parser.add_argument("--artifact", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results", nargs="*", default=[], help="sensitivity: the identity results to read")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--server-arg", dest="server_args", action="append", default=[],
                        help="an infernix-serve argument (repeat; e.g. --server-arg=--kv-dtype --server-arg=bf16)")
    parser.add_argument("--spec-arg", dest="spec_args", action="append", default=[],
                        help="specdist: the speculation arguments the spec arm adds")
    parser.add_argument("--negative-control", action="store_true")
    parser.add_argument("--golden", help="identity: the release's stored hash file of the reproducible responses")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--gsm8k", type=int, default=250)
    parser.add_argument("--mmlu", type=int, default=500)
    args = parser.parse_args(argv)
    if args.check != "sensitivity" and not (args.serve and args.artifact):
        parser.error(f"{args.check} needs --serve and --artifact")
    args.log = args.log or args.out.with_suffix(".serve.log")
    started = time.time()
    try:
        result = CHECKS[args.check](args)
    except Exception as error:  # a failed run is a failed check, recorded with its cause
        result = {"passed": False, "detail": f"{type(error).__name__}: {error}"}
    result["check"] = args.check
    result["artifact"] = args.artifact
    result["seconds"] = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"{args.check}: {'PASS' if result['passed'] else 'FAIL'} - {result.get('detail', '')}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
