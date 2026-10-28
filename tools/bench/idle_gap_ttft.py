#!/usr/bin/env python3
"""TTFT of a short continuing turn after an idle gap (candidates NN-09: the 2026-10-26 sizing found 6-9 ms of
excess at 12-60 s gaps, so the GPU keep-warm was not built into the Engine).

Against a running `infernix-serve` (prefix cache on, one request at a time), each pair is
  request 1   a long document (about --doc-tokens tokens) as the first user turn; prefilled and cached
  <gap>       the client sleeps --gaps seconds (the server and the GPU are idle)
  request 2   the same document, a short assistant turn and a new question of about --question-tokens
              tokens: the document is a cache hit, so the request prefills only the question
and the time from sending request 2 to its first streamed token is the measurement. A GPU that has
dropped to a low-power clock state pays the climb back inside it. Gaps are shuffled within every repeat.
Just before request 2 the client reads the GPU's performance state and SM clock from nvidia-smi.

  idle_gap_ttft.py run --url http://127.0.0.1:8095 [--model NAME] --corpus docs.txt --out arm.jsonl
                       [--gaps 0,3,12,30,60] [--repeats 6] [--doc-tokens 20000] [--question-tokens 200]
  idle_gap_ttft.py summary arm_a.jsonl [arm_b.jsonl ...]    per gap: n, median and p90 TTFT, P-state counts
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

CHARS_PER_TOKEN = 3.6  # English prose and markdown; the server's own counts are in its request log


def post_stream(url: str, body: dict, timeout: float = 900.0) -> tuple[float | None, float]:
    """Seconds to the first streamed token and to the end of the response."""
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            for choice in json.loads(payload).get("choices", ()):
                delta = choice.get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                    first = time.perf_counter() - start
    return first, time.perf_counter() - start


def gpu_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=pstate,clocks.sm,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().split(",")
        return {"pstate": out[0].strip(), "sm_mhz": float(out[1]), "watts": float(out[2])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def wait_healthy(url: str, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    raise SystemExit("the server did not become healthy")


def run(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    text = Path(args.corpus).read_text(encoding="utf-8", errors="replace")
    doc_chars, question_chars = int(args.doc_tokens * CHARS_PER_TOKEN), int(args.question_tokens * CHARS_PER_TOKEN)
    if len(text) < doc_chars + question_chars:
        raise SystemExit("the corpus is shorter than one pair")
    gaps = [float(g) for g in args.gaps.split(",")]
    wait_healthy(args.url, args.startup_seconds)
    model = args.model
    if not model:
        with urllib.request.urlopen(args.url + "/v1/models", timeout=30) as response:
            model = json.loads(response.read())["data"][0]["id"]

    def window(chars: int) -> str:
        begin = rng.randrange(0, len(text) - chars)
        return text[begin : begin + chars]

    def body(messages: list[dict]) -> dict:
        return {
            "model": model, "messages": messages, "max_tokens": 1, "stream": True, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as sink:
        # One discarded pair first: the first request after a server start is slow.
        for repeat in range(-1, args.repeats):
            order = gaps[:]
            rng.shuffle(order)
            for gap in order:
                document = window(doc_chars)
                question = "Answer in one word: " + window(question_chars)
                first1, total1 = post_stream(args.url, body([{"role": "user", "content": document}]))
                if gap > 0.5:
                    time.sleep(gap - 0.3)
                state = gpu_state()
                if gap > 0.5:
                    time.sleep(0.3)
                first2, total2 = post_stream(args.url, body([
                    {"role": "user", "content": document},
                    {"role": "assistant", "content": "Understood."},
                    {"role": "user", "content": question},
                ]))
                if repeat >= 0:
                    sink.write(json.dumps({"repeat": repeat, "gap": gap, "ttft1": first1, "ttft2": first2,
                                           "total2": total2, **state}) + "\n")
                    sink.flush()
    return 0


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


def summary(paths: list[str]) -> int:
    print(f"{'file':32} {'gap s':>6} {'n':>3} {'median ms':>10} {'p90 ms':>8} {'excess ms':>10}  P-states before request 2")
    for path in paths:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        by_gap: dict[float, list[dict]] = {}
        for row in rows:
            if row.get("ttft2") is not None:
                by_gap.setdefault(row["gap"], []).append(row)
        base = statistics.median([r["ttft2"] for r in by_gap[min(by_gap)]]) if by_gap else 0.0
        for gap in sorted(by_gap):
            ttfts = [r["ttft2"] for r in by_gap[gap]]
            states = Counter(r.get("pstate", "?") for r in by_gap[gap])
            median = statistics.median(ttfts)
            print(f"{Path(path).name[:32]:32} {gap:>6g} {len(ttfts):>3} {1000 * median:>10.1f} {1000 * p90(ttfts):>8.1f} "
                  f"{1000 * (median - base):>10.1f}  {dict(states)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--url", required=True)
    runner.add_argument("--model", default="", help="the served alias (default: the server's first model)")
    runner.add_argument("--corpus", required=True)
    runner.add_argument("--out", required=True)
    runner.add_argument("--gaps", default="0,3,12,30,60")
    runner.add_argument("--repeats", type=int, default=6)
    runner.add_argument("--doc-tokens", type=int, default=20000)
    runner.add_argument("--question-tokens", type=int, default=200)
    runner.add_argument("--seed", type=int, default=20261026)
    runner.add_argument("--startup-seconds", type=float, default=600.0)
    reporter = sub.add_parser("summary")
    reporter.add_argument("files", nargs="+")
    args = parser.parse_args()
    return run(args) if args.command == "run" else summary(args.files)


if __name__ == "__main__":
    sys.exit(main())
