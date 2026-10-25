#!/usr/bin/env python3
"""Basalt parity runs (candidates B0a and BX-02; .local/review-2026-10-24): Infernix on Basalt's own
workload and on giveen's cross-engine chart protocol. Every measurement is one fresh infernix_bench
process with no warm-up, as `basalt-bench --community` runs (cold first runs); one discarded process
goes first (the first session after a build decodes slow).

Suite `community` (B0a), Basalt's data/bench fixtures read as data (chat-templated Qwen3.8 token ids):
  conv    up to 1,200 tokens, T 0.6, top_p 0.95, top_k 20
  struct  up to 8,192 tokens, greedy (Basalt's run ends at EOS after 7,230)
  prose   up to 1,200 tokens, T 0.6, top_p 0.95, top_k 20
  prefill 16,384 and 65,536 tokens: prefill.ids's head, 4 random ids, its body cut or repeated to the
          depth, its tail (basalt-bench's construction), one output token
Every decode request ends at EOS (--stop-at-eos), as Basalt's do. `decode_tps` is the generated tokens of
conv, struct and prose over their summed decode time, Basalt's token-weighted figure.

Suite `chart` (BX-02, giveen's ninfer-ext chart): cold unique prompts of the same construction at 16K /
64K / 128K / 256K, 262,144 max context, 256 output tokens, greedy and T 1.0 (top_p 0.95, top_k 20).

Arms (--arms): plain; mtp (MTP auto); mtp-copies (MTP auto, n-gram copies at min match 4 with
INFERNIX_TMP_NGRAM_AGREE=1); dflash2 (DFlash2 auto, needs --draft-model). Each sampled request runs
--seeds times; seed s is the same for every arm. --route-trace-arm writes one route trace per decode
fixture (seed 0), for the D12 / PUB-22 cache-policy replays.

Usage:
  basalt_parity.py --bench BENCH --model ARTIFACT --fixtures BASALT/data/bench --out DIR
                   [--suite community|chart|both] [--arms plain,mtp,mtp-copies] [--seeds 3]
                   [--ngram-volume PATH] [--draft-model PATH] [--route-trace-arm mtp]
                   [--timeout 2400] [--extra-arg=ARG ...]
Writes DIR/runs/*.{json,log,ids}, DIR/summary.json and DIR/report.md (Basalt's published figures beside
ours, with the review's comparability caveats).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The decode fixtures of basalt-bench (src/program/bench_main.cpp kDecode): name, file, tokens, temperature.
DECODE_FIXTURES = (("conv", "conv.ids", 1200, 0.6), ("struct", "struct.ids", 8192, 0.0),
                   ("prose", "prose.ids", 1200, 0.6))
STRUCT_EXPECTED_TOKENS = 7230
COMMUNITY_PREFILL_DEPTHS = (16384, 65536)
CHART_OUTPUT_TOKENS = 256
CHART_MAX_CONTEXT = 262144
# Total prompt tokens; the "256K" point is the longest prompt the context leaves room for.
CHART_DEPTHS = (16384, 65536, 131072, CHART_MAX_CONTEXT - CHART_OUTPUT_TOKENS - 256)
FILLER_RANDOM_IDS = 4
# Ordinary vocabulary ids for the inserted random ids (below the Qwen3.8 special tokens).
RANDOM_ID_RANGE = (1000, 150000)

# Basalt's published figures (giveen's community row, Basalt PR #35): one RTX 5090 on PCIe x16, Linux,
# 247 GB RAM, 475 W, the NVFP4 pack (lossier than Dense8 outside the experts), int8 KV.
BASALT_REFERENCE = {
    "source": "giveen, Basalt community-1 row (PR #35): RTX 5090, PCIe x16, Linux, 475 W, NVFP4 pack",
    "conv_tok_s": 208.0, "struct_tok_s": 367.0, "prose_tok_s": 212.0, "decode_tps": 311.8,
    "prefill_16384_tok_s": 9556.0, "prefill_65536_tok_s": 10542.0,
}

ARMS: dict[str, tuple[list[str], dict[str, str]]] = {
    "plain": ([], {}),
    "mtp": (["--spec", "mtp", "--draft-tokens", "auto"], {}),
    "mtp-copies": (["--spec", "mtp", "--draft-tokens", "auto", "--ngram-draft-tokens", "7",
                    "--ngram-min-match", "4"], {"INFERNIX_TMP_NGRAM_AGREE": "1"}),
    "dflash2": (["--spec", "dflash2", "--draft-tokens", "auto"], {}),
}


@dataclass
class Run:
    suite: str
    arm: str
    name: str            # conv, struct, prose, prefill16384, chart131072-t1.0, ...
    seed: int
    report: dict[str, Any] | None
    returncode: int
    seconds: float
    problems: list[str] = field(default_factory=list)


def read_ids(path: Path) -> list[list[int]]:
    """Each non-empty line of comma-separated ids."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            lines.append([int(word) for word in line.replace(",", " ").split()])
    return lines


def filler_prompt(parts: list[list[int]], depth: int, rng: random.Random) -> list[int]:
    """basalt-bench's prefill prompt: head, 4 random ids, the body cut or repeated to `depth`, tail."""
    if len(parts) != 3 or not parts[1]:
        raise ValueError("prefill.ids needs three lines (head, body, tail) with a non-empty body")
    head, body, tail = parts
    repeated = (body * (depth // len(body) + 1))[:depth]
    noise = [rng.randrange(*RANDOM_ID_RANGE) for _ in range(FILLER_RANDOM_IDS)]
    return [*head, *noise, *repeated, *tail]


def fitted_context(prompt_tokens: int, output_tokens: int) -> int:
    """--max-ctx for one request: prompt and output with 64 tokens to spare, in whole 256-token steps."""
    need = prompt_tokens + output_tokens + 64
    return (need + 255) // 256 * 256


def test_of(report: dict[str, Any] | None) -> dict[str, Any] | None:
    tests = (report or {}).get("tests", [])
    return tests[0] if tests else None


def figures(run: Run) -> dict[str, Any]:
    """The run's figures: decode/prefill tok/s, generated tokens, decode seconds, tokens per round."""
    test = test_of(run.report)
    if test is None:
        return {}
    rep = (test.get("reps") or [{}])[0]
    speculative = test.get("speculative") or {}
    return {
        "decode_tok_s": test.get("decode_output_tok_s_mean"),
        "prefill_tok_s": test.get("prefill_tok_s_mean"),
        "generated_tokens": rep.get("generated_output_tokens"),
        "decode_seconds": test.get("decode_seconds_mean"),
        "tokens_per_round": speculative.get("acceptance_length"),
    }


def median(values: list[float]) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def summarize(runs: list[Run]) -> dict[str, Any]:
    """Per suite and arm: medians over seeds per measurement, and community decode_tps per seed."""
    out: dict[str, Any] = {}
    for run in runs:
        bucket = out.setdefault(run.suite, {}).setdefault(run.arm, {}).setdefault(run.name, [])
        bucket.append({"seed": run.seed, "returncode": run.returncode, "problems": run.problems, **figures(run)})
    for suite, arms in out.items():
        for arm, names in arms.items():
            table: dict[str, Any] = {}
            for name, rows in names.items():
                table[name] = {
                    "runs": rows,
                    "decode_tok_s": median([r.get("decode_tok_s") for r in rows]),
                    "prefill_tok_s": median([r.get("prefill_tok_s") for r in rows]),
                    "tokens_per_round": median([r.get("tokens_per_round") for r in rows]),
                    "generated_tokens": median([r.get("generated_tokens") for r in rows]),
                }
            if suite == "community":
                weighted = []
                for seed in sorted({r["seed"] for rows in names.values() for r in rows}):
                    tokens = seconds = 0.0
                    complete = True
                    for fixture, _, _, temperature in DECODE_FIXTURES:
                        # A greedy fixture runs once and serves every seed.
                        want = 0 if temperature == 0.0 else seed
                        row = next((r for r in names.get(fixture, []) if r["seed"] == want), None)
                        if row is None or not row.get("decode_seconds") or row.get("generated_tokens") is None:
                            complete = False
                            break
                        tokens += max(0, row["generated_tokens"] - 1)
                        seconds += row["decode_seconds"]
                    if complete and seconds > 0:
                        weighted.append(tokens / seconds)
                table["decode_tps"] = {"per_seed": weighted, "median": median(weighted)}
            arms[arm] = table
    return out


def run_bench(args: argparse.Namespace, suite: str, arm: str, name: str, seed: int, prompt: list[int],
              output_tokens: int, temperature: float, max_context: int, route_trace: bool) -> Run:
    runs_dir = args.out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{suite}-{arm}-{name}-s{seed}"
    prompt_path = runs_dir / f"{tag}.ids"
    prompt_path.write_text(",".join(map(str, prompt)) + "\n", encoding="utf-8")
    report_path, log_path = runs_dir / f"{tag}.json", runs_dir / f"{tag}.log"
    arm_args, arm_env = ARMS[arm]
    command = [str(args.bench), "--weights", str(args.model), "--kv-dtype", "int8", "--ngram-residency", "volume",
               "--prompt-ids", str(prompt_path), "-n", str(output_tokens), "-r", "1", "--warmup", "0",
               "--max-ctx", str(max_context), "--stop-at-eos", "--seed", str(seed)]
    if args.ngram_volume:
        command += ["--ngram-volume", str(args.ngram_volume)]
    if temperature > 0.0:
        command += ["--temperature", f"{temperature:g}", "--top-p", "0.95", "--top-k", "20"]
    command += arm_args
    if arm == "dflash2":
        command += ["--draft-model", str(args.draft_model)]
    if route_trace:
        command += ["--route-trace", str(runs_dir / f"{tag}.routes")]
    command += [*args.extra_arg, "-o", "json", "--output-file", str(report_path)]
    env = dict(os.environ)
    env.update(arm_env)
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        try:
            code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                                  check=False).returncode
        except subprocess.TimeoutExpired:
            code = -9
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None
    run = Run(suite, arm, name, seed, report, code, time.time() - start)
    if code != 0:
        run.problems.append(f"exit {code} (see {log_path.name})")
    if name == "struct" and (figures(run).get("generated_tokens") or 0) != STRUCT_EXPECTED_TOKENS:
        run.problems.append(f"struct generated {figures(run).get('generated_tokens')} tokens, Basalt "
                            f"{STRUCT_EXPECTED_TOKENS}: the texts differ")
    print(f"[basalt_parity] {tag}: exit {code}, {figures(run)}", flush=True)
    return run


def plan(args: argparse.Namespace) -> list[tuple]:
    """(suite, arm, name, seed, prompt, output tokens, temperature, max context, route trace) per run."""
    fixtures = Path(args.fixtures)
    filler = read_ids(fixtures / "prefill.ids")
    rng = random.Random(20261024)
    jobs = []
    for arm in args.arms:
        if args.suite in ("community", "both"):
            for seed in range(args.seeds):
                for name, file, tokens, temperature in DECODE_FIXTURES:
                    if temperature == 0.0 and seed > 0:
                        continue  # greedy: one run
                    prompt = read_ids(fixtures / file)[0]
                    trace = arm == args.route_trace_arm and seed == 0
                    jobs.append(("community", arm, name, seed, prompt, tokens, temperature,
                                 fitted_context(len(prompt), tokens), trace))
            for depth in COMMUNITY_PREFILL_DEPTHS:
                prompt = filler_prompt(filler, depth, rng)
                jobs.append(("community", arm, f"prefill{depth}", 0, prompt, 1, 0.0, fitted_context(len(prompt), 1),
                             False))
        if args.suite in ("chart", "both"):
            for depth in CHART_DEPTHS:
                for temperature in (0.0, 1.0):
                    prompt = filler_prompt(filler, depth - len(filler[0]) - len(filler[2]) - FILLER_RANDOM_IDS, rng)
                    jobs.append(("chart", arm, f"chart{depth}-t{temperature:g}", 0, prompt, CHART_OUTPUT_TOKENS,
                                 temperature, CHART_MAX_CONTEXT, False))
    return jobs


def write_report(out: Path, summary: dict[str, Any]) -> None:
    lines = ["# Basalt parity runs", "",
             f"Basalt reference: {BASALT_REFERENCE['source']}.",
             "Not like for like: x8 Windows here, other quantization (Basalt's dense layers are 6-bit), the same",
             "prompts and sampling. Medians over seeds; tokens per round from the bench's acceptance length.", ""]
    community = summary.get("community", {})
    if community:
        lines += ["## community-1 (B0a)", "",
                  "| arm | conv tok/s | struct tok/s | prose tok/s | decode_tps | prefill 16K | prefill 64K | "
                  "tokens/round conv / struct / prose |", "|---|---|---|---|---|---|---|---|"]
        ref = BASALT_REFERENCE
        lines.append(f"| Basalt (reference) | {ref['conv_tok_s']:.0f} | {ref['struct_tok_s']:.0f} | "
                     f"{ref['prose_tok_s']:.0f} | {ref['decode_tps']:.1f} | {ref['prefill_16384_tok_s']:.0f} | "
                     f"{ref['prefill_65536_tok_s']:.0f} | 2.63 / 7.49 / 2.19 |")

        def cell(value: Any, digits: int = 0) -> str:
            return "-" if value is None else f"{value:.{digits}f}"

        for arm, table in community.items():
            get = lambda name, key: (table.get(name) or {}).get(key)  # noqa: E731
            lines.append(
                f"| {arm} | {cell(get('conv', 'decode_tok_s'))} | {cell(get('struct', 'decode_tok_s'))} | "
                f"{cell(get('prose', 'decode_tok_s'))} | {cell((table.get('decode_tps') or {}).get('median'), 1)} | "
                f"{cell(get('prefill16384', 'prefill_tok_s'))} | {cell(get('prefill65536', 'prefill_tok_s'))} | "
                f"{cell(get('conv', 'tokens_per_round'), 2)} / {cell(get('struct', 'tokens_per_round'), 2)} / "
                f"{cell(get('prose', 'tokens_per_round'), 2)} |")
        lines.append("")
    chart = summary.get("chart", {})
    if chart:
        lines += ["## giveen's chart protocol (BX-02)", "", "| arm | run | prefill tok/s | decode tok/s |",
                  "|---|---|---|---|"]
        for arm, table in chart.items():
            for name, row in table.items():
                lines.append(f"| {arm} | {name} | {row.get('prefill_tok_s') or '-'} | {row.get('decode_tok_s') or '-'} |")
        lines.append("")
    problems = [f"{suite}/{arm}/{name} seed {r['seed']}: {p}" for suite, arms in summary.items()
                for arm, table in arms.items() for name, row in table.items() if isinstance(row, dict)
                for r in row.get("runs", []) for p in r.get("problems", [])]
    lines += ["## Problems", ""] + ([f"- {p}" for p in problems] or ["- none"])
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True, help="Basalt's data/bench directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=("community", "chart", "both"), default="community")
    parser.add_argument("--arms", type=lambda s: s.split(","), default=["plain", "mtp", "mtp-copies"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ngram-volume", type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--route-trace-arm", default="mtp")
    parser.add_argument("--timeout", type=float, default=2400.0, help="seconds per process (watchdog)")
    parser.add_argument("--extra-arg", action="append", default=[], help="passed to every bench (use --extra-arg=X)")
    parser.add_argument("--no-discard", action="store_true", help="skip the discarded first process")
    args = parser.parse_args(argv)
    unknown = [arm for arm in args.arms if arm not in ARMS]
    if unknown:
        parser.error(f"unknown arms {unknown}; known: {sorted(ARMS)}")
    if "dflash2" in args.arms and args.draft_model is None:
        parser.error("the dflash2 arm needs --draft-model")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = plan(args)
    if not args.no_discard and jobs:
        suite, arm, name, seed, prompt, tokens, temperature, context, _ = jobs[0]
        run_bench(args, "discard", arm, name, seed, prompt, tokens, temperature, context, False)
    runs = [run_bench(args, *job) for job in jobs]
    summary = summarize(runs)
    (args.out / "summary.json").write_text(json.dumps({"basalt_reference": BASALT_REFERENCE, "results": summary},
                                                      indent=2), encoding="utf-8")
    write_report(args.out, summary)
    print(f"[basalt_parity] wrote {args.out / 'report.md'}")
    return 0 if all(run.returncode == 0 for run in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
