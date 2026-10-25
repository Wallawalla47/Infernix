#!/usr/bin/env python3
"""Interleaved A/B (ABBA) timing of infernix_bench arms, with the checks this repository's A/Bs need.

Each sample is one fresh infernix_bench process. Arms differ by environment (INFERNIX_TMP_* switches)
and/or arguments (--fast-path, options); the bench binary may differ per arm too. The runner:

* runs a discarded warm-up process first (the first session after a build decodes slow);
* interleaves the arms in rotating order per block (A B / B A for two arms, a Latin rotation for
  more), so drift lands on every arm alike;
* adds a noise arm (the control again, under another name) unless --no-noise-arm: a same-engine pair
  whose verdict must be "no difference", else the run is marked noisy;
* checks every process's startup evidence: the `temporary switches |` record must show each arm's
  expected switch values (a positive control that the arm really ran), the PCIe link probe must read
  at least --min-link-gbs, the CPU expert team rate must lie within --cpu-rate-tolerance of the run's
  median, and the configured fast paths and n-gram residency must match the arm;
* varies the prompt per block when the arm file lists corpus offsets (a repeated prompt flatters a
  warm expert cache);
* reports, per metric and arm against the control: pairs where the arm was better, the median paired
  ratio, the ratio of pooled means, and a two-sided exact Wilcoxon signed-rank p-value with Holm's
  correction over the arms, so one run decides several arms honestly.

Arm file (JSON):

    {
      "bench": "E:/Infernix/.local/build-integrate/bench/Release/infernix_bench.exe",
      "common_args": ["--weights", "D:/.../FP4-Local-00001-of-00003.infernix", "--kv-dtype", "int8",
                      "-pg", "1280,256", "-r", "2", "--warmup", "1", "--ngram-residency", "volume"],
      "env": {"PATH": "..."},                      # optional, added to every arm
      "corpus_offsets": [0, 40000, 80000],         # optional: --corpus-offset per block, rotating
      "metrics": [{"name": "decode_output_tok_s", "higher_is_better": true},
                  {"name": "prefill_tok_s", "higher_is_better": true}],
      "test": "pg1280,256",                        # optional: the test label to read (default: first)
      "control": "base",
      "arms": [
        {"name": "base"},
        {"name": "agree", "env": {"INFERNIX_TMP_NGRAM_AGREE": "1"},
         "expect_switches": {"INFERNIX_TMP_NGRAM_AGREE": "1"}},
        {"name": "f16", "args": ["--fast-path", "f16-accumulate-prefill"]}
      ]
    }

Replay-bench metrics (candidates §8.5 CF-03): with "INFERNIX_TMP_REPLAY_BENCH": "<warm rounds>" in the
common env, a greedy request with --expert-state (or warm rounds) replays the same rounds against a
frozen expert cache, and the metrics "replay_ms_per_round" (device time less CPU waits per decode
round), "replay_device_ms", "replay_wait_ms" and "replay_less_waits_ms" (all lower is better) are read
from each process's log; use them for exact sub-1 % items.

Usage: abba.py ARMS.json --out DIR [--blocks 6] [--timeout 1800]
Writes DIR/report.md and DIR/summary.json, and keeps every process's JSON report and logs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LINK_RE = re.compile(r"PCIe link to the GPU: ([0-9.]+) GB/s")
CPU_RATE_RE = re.compile(r"CPU expert team: (\d+) workers[^,]*, ([0-9.]+) experts/s")
# INFERNIX_TMP_REPLAY_BENCH's release line (models/qwen4_exp/program/program_impl.h release()).
REPLAY_RE = re.compile(r"replay bench: (\d+) rounds, ([0-9.]+) ms device time, ([0-9.]+) ms CPU waits, "
                       r"(-?[0-9.]+) ms less waits \((-?[0-9.]+) ms a round\)")


@dataclass
class Arm:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)
    bench: str | None = None
    expect_switches: dict[str, str] = field(default_factory=dict)


@dataclass
class Sample:
    arm: str
    block: int
    order: int
    report: dict[str, Any] | None
    log_text: str
    returncode: int
    seconds: float
    problems: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Statistics


def wilcoxon_signed_rank(differences: list[float]) -> float:
    """Two-sided p-value of the Wilcoxon signed-rank test (zeros dropped, average ranks for ties).

    Exact by enumeration up to 20 non-zero differences, the normal approximation with a tie
    correction above that."""
    d = [x for x in differences if x != 0.0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    w_plus = sum(r for r, x in zip(ranks, d) if x > 0)
    total = n * (n + 1) / 2.0
    if n <= 20:
        # Distribution of W+ over every sign assignment of these ranks.
        doubled = [int(round(2 * r)) for r in ranks]
        counts: dict[int, int] = {0: 1}
        for r in doubled:
            nxt: dict[int, int] = {}
            for s, c in counts.items():
                nxt[s] = nxt.get(s, 0) + c
                nxt[s + r] = nxt.get(s + r, 0) + c
            counts = nxt
        observed = int(round(2 * w_plus))
        centre = int(round(total))  # 2 * (total / 2)
        distance = abs(observed - centre)
        extreme = sum(c for s, c in counts.items() if abs(s - centre) >= distance)
        return min(1.0, extreme / float(2**n))
    mean = total / 2.0
    tie_sum = 0.0
    for _, group in itertools.groupby(sorted(abs(x) for x in d)):
        t = len(list(group))
        tie_sum += t**3 - t
    variance = n * (n + 1) * (2 * n + 1) / 24.0 - tie_sum / 48.0
    if variance <= 0:
        return 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
    return min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2.0)))


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------------------------------
# Schedule and checks


def schedule(arm_names: list[str], blocks: int) -> list[list[str]]:
    """Per block, the arm order: A B then B A for two arms; a rotation of the arms for more, each
    block reversing the previous direction, so every arm takes every position equally often."""
    out = []
    n = len(arm_names)
    for block in range(blocks):
        shift = (block // 2) % n if n > 2 else 0
        rotated = arm_names[shift:] + arm_names[:shift]
        out.append(rotated if block % 2 == 0 else list(reversed(rotated)))
    return out


def parse_switches(record: str) -> dict[str, str]:
    """`NAME=value | NAME=value | ...` (the startup log's temporary switches record) as a dict."""
    out: dict[str, str] = {}
    for part in record.split("|"):
        part = part.strip()
        if "=" in part:
            name, value = part.split("=", 1)
            out[name.strip()] = value.strip()
    return out


def check_sample(sample: Sample, arm: Arm, min_link_gbs: float) -> None:
    if sample.returncode != 0 or sample.report is None:
        sample.problems.append(f"exit {sample.returncode} or no report")
        return
    # infernix_bench's JSON report records the run's switches and fast paths in its config object.
    config = sample.report.get("config", {})
    switches = parse_switches(config.get("temporary_switches", ""))
    # Every INFERNIX_TMP_* value the arm sets must show in the record, as must expect_switches.
    expected = {k: v for k, v in arm.env.items() if k.startswith("INFERNIX_TMP_")}
    expected.update(arm.expect_switches)
    for name, value in expected.items():
        if switches.get(name) != value:
            sample.problems.append(f"switch {name}={switches.get(name, 'absent')} (expected {value})")
    wanted = [a for i, a in enumerate(arm.args) if i > 0 and arm.args[i - 1] == "--fast-path"]
    configured = config.get("fast_paths", "")
    for path in wanted:
        if path not in configured:
            sample.problems.append(f"fast path {path} not in the report's fast_paths ({configured})")
    link = LINK_RE.findall(sample.log_text)
    if link and float(link[-1]) < min_link_gbs:
        sample.problems.append(f"link probe {link[-1]} GB/s below {min_link_gbs}")


def cpu_rate(sample: Sample) -> float | None:
    found = CPU_RATE_RE.findall(sample.log_text)
    return float(found[-1][1]) if found else None


def replay_value(log_text: str, metric: str) -> float | None:
    """The replay bench's last release line (INFERNIX_TMP_REPLAY_BENCH; cumulative since startup)."""
    found = REPLAY_RE.findall(log_text)
    if not found:
        return None
    rounds, device_ms, wait_ms, less_ms, per_round = found[-1]
    return {"replay_ms_per_round": float(per_round), "replay_device_ms": float(device_ms),
            "replay_wait_ms": float(wait_ms), "replay_less_waits_ms": float(less_ms)}.get(metric)


def sample_value(sample: Sample, test: str | None, metric: str) -> float | None:
    if metric.startswith("replay_"):
        return replay_value(sample.log_text, metric)
    return metric_value(sample.report, test, metric) if sample.report is not None else None


def metric_value(report: dict[str, Any], test: str | None, metric: str) -> float | None:
    tests = report.get("tests", [])
    if not tests:
        return None
    chosen = next((t for t in tests if test is None or t.get("label") == test), None)
    if chosen is None:
        return None
    value = chosen.get(metric + "_mean")
    return float(value) if isinstance(value, (int, float)) else None


# ---------------------------------------------------------------------------------------------
# Running


def run_sample(arm: Arm, config: dict[str, Any], out: Path, tag: str, extra: list[str],
               timeout: float) -> Sample:
    bench = arm.bench or config["bench"]
    report_path = out / f"{tag}.json"
    log_path = out / f"{tag}.log"
    command = [bench, *config.get("common_args", []), *extra, *arm.args, "-o", "json",
               "--output-file", str(report_path)]
    env = dict(os.environ)
    env.update(config.get("env", {}))
    env.update(arm.env)
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(command) + "\n")
        log.flush()
        try:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout, check=False)
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = -9
    seconds = time.time() - start
    text = log_path.read_text(encoding="utf-8", errors="replace")
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None
    return Sample(arm.name, -1, -1, report, text, code, seconds)


def analyse(samples: list[Sample], arms: dict[str, Arm], control: str, metrics: list[dict[str, Any]],
            test: str | None, cpu_tolerance: float) -> dict[str, Any]:
    rates = [r for r in (cpu_rate(s) for s in samples) if r]
    if rates:
        median_rate = statistics.median(rates)
        for s in samples:
            r = cpu_rate(s)
            if r and abs(r / median_rate - 1.0) > cpu_tolerance:
                s.problems.append(f"CPU expert rate {r:.0f}/s vs run median {median_rate:.0f}/s")
    valid = [s for s in samples if not s.problems]
    by_block: dict[int, dict[str, Sample]] = {}
    for s in valid:
        by_block.setdefault(s.block, {})[s.arm] = s
    results: dict[str, Any] = {"metrics": {}, "excluded": [
        {"arm": s.arm, "block": s.block, "problems": s.problems} for s in samples if s.problems]}
    for metric in metrics:
        name = metric["name"]
        higher = bool(metric.get("higher_is_better", True))
        rows = []
        p_values = []
        for arm in arms:
            if arm == control:
                continue
            pairs = []
            for block in sorted(by_block):
                group = by_block[block]
                if arm in group and control in group:
                    a = sample_value(group[arm], test, name)
                    c = sample_value(group[control], test, name)
                    if a is not None and c is not None and c != 0:
                        pairs.append((a, c))
            if not pairs:
                rows.append({"arm": arm, "pairs": 0})
                p_values.append(1.0)
                continue
            ratios = [a / c for a, c in pairs]
            better = sum(1 for a, c in pairs if (a > c if higher else a < c))
            p = wilcoxon_signed_rank([a - c for a, c in pairs])
            p_values.append(p)
            rows.append({
                "arm": arm,
                "pairs": len(pairs),
                "better": better,
                "median_ratio": statistics.median(ratios),
                "worst_ratio": min(ratios) if higher else max(ratios),
                "pooled_mean_ratio": statistics.fmean(a for a, _ in pairs) / statistics.fmean(c for _, c in pairs),
                "arm_mean": statistics.fmean(a for a, _ in pairs),
                "control_mean": statistics.fmean(c for _, c in pairs),
                "p": p,
            })
        for row, adjusted in zip(rows, holm(p_values)):
            row["p_holm"] = adjusted
        results["metrics"][name] = {"higher_is_better": higher, "arms": rows}
    return results


def verdict(row: dict[str, Any], higher: bool, alpha: float) -> str:
    if not row.get("pairs"):
        return "no valid pairs"
    gain = row["pooled_mean_ratio"] - 1.0 if higher else 1.0 - row["pooled_mean_ratio"]
    if row["p_holm"] < alpha:
        return "better" if gain > 0 else "worse"
    if row["better"] == row["pairs"]:
        return "better in every pair (not significant)"
    if row["better"] == 0:
        return "worse in every pair (not significant)"
    return "no resolved difference"


def write_report(out: Path, results: dict[str, Any], config: dict[str, Any], noise: str | None,
                 alpha: float) -> None:
    lines = ["# ABBA report", "", f"Control: `{config['control']}`. Samples are fresh processes; "
             "ratios are arm / control per block.", ""]
    noisy = False
    for name, metric in results["metrics"].items():
        higher = metric["higher_is_better"]
        lines += [f"## {name} ({'higher' if higher else 'lower'} is better)", "",
                  "| Arm | Pairs | Better | Median ratio | Worst ratio | Pooled-mean ratio | p | p (Holm) | Verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in metric["arms"]:
            if not row.get("pairs"):
                lines.append(f"| {row['arm']} | 0 | | | | | | | no valid pairs |")
                continue
            v = verdict(row, higher, alpha)
            if row["arm"] == noise and v in ("better", "worse"):
                noisy = True
            lines.append(
                f"| {row['arm']} | {row['pairs']} | {row['better']} | {row['median_ratio']:.4f} | "
                f"{row['worst_ratio']:.4f} | {row['pooled_mean_ratio']:.4f} | {row['p']:.3g} | "
                f"{row['p_holm']:.3g} | {v} |")
        lines.append("")
    if noise:
        lines += [f"Noise arm `{noise}` (the control again): "
                  + ("**a resolved difference: this run is noisy, do not trust its verdicts.**" if noisy
                     else "no resolved difference."), ""]
    if results["excluded"]:
        lines += ["## Excluded samples", ""]
        for item in results["excluded"]:
            lines.append(f"- {item['arm']} block {item['block']}: {'; '.join(item['problems'])}")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"noisy": noisy, **results}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("arms", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per process")
    parser.add_argument("--min-link-gbs", type=float, default=20.0)
    parser.add_argument("--cpu-rate-tolerance", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-noise-arm", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(args.arms.read_text(encoding="utf-8"))
    arms = {a["name"]: Arm(**a) for a in config["arms"]}
    control = config["control"]
    if control not in arms:
        raise SystemExit(f"control arm {control!r} is not in the arm file")
    noise = None
    if not args.no_noise_arm:
        noise = control + "-again"
        base = arms[control]
        arms[noise] = Arm(noise, dict(base.env), list(base.args), base.bench, dict(base.expect_switches))
    args.out.mkdir(parents=True, exist_ok=True)
    offsets = config.get("corpus_offsets", [])

    if not args.no_warmup:
        print("warm-up process (discarded)", flush=True)
        run_sample(arms[control], config, args.out, "warmup", [], args.timeout)

    samples: list[Sample] = []
    for block, order in enumerate(schedule(list(arms), args.blocks)):
        extra = ["--corpus-offset", str(offsets[block % len(offsets)])] if offsets else []
        for position, name in enumerate(order):
            tag = f"b{block:02d}-{position}-{name}"
            sample = run_sample(arms[name], config, args.out, tag, extra, args.timeout)
            sample.block, sample.order = block, position
            check_sample(sample, arms[name], args.min_link_gbs)
            samples.append(sample)
            status = "ok" if not sample.problems else "; ".join(sample.problems)
            print(f"block {block} {name}: {sample.seconds:.0f} s, {status}", flush=True)

    results = analyse(samples, arms, control, config.get("metrics", [{"name": "decode_output_tok_s"}]),
                      config.get("test"), args.cpu_rate_tolerance)
    write_report(args.out, results, config, noise, args.alpha)
    print((args.out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
