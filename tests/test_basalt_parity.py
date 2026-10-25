"""tools/bench/basalt_parity.py: the fixture prompts, the run plan and the community summary."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))

import basalt_parity as parity  # noqa: E402


def write_fixtures(directory: Path) -> None:
    (directory / "conv.ids").write_text("1,2,3\n", encoding="utf-8")
    (directory / "struct.ids").write_text("4,5\n", encoding="utf-8")
    (directory / "prose.ids").write_text("6\n", encoding="utf-8")
    (directory / "prefill.ids").write_text("10,11\n20,21,22\n30\n", encoding="utf-8")


def test_filler_prompt_cuts_and_repeats_the_body() -> None:
    parts = [[10, 11], [20, 21, 22], [30]]
    prompt = parity.filler_prompt(parts, 7, random.Random(1))
    assert prompt[:2] == [10, 11] and prompt[-1] == 30
    assert prompt[2 + parity.FILLER_RANDOM_IDS:-1] == [20, 21, 22, 20, 21, 22, 20]
    assert len(prompt) == 2 + parity.FILLER_RANDOM_IDS + 7 + 1
    assert all(parity.RANDOM_ID_RANGE[0] <= t < parity.RANDOM_ID_RANGE[1] for t in prompt[2:6])


def test_fitted_context_rounds_up() -> None:
    assert parity.fitted_context(100, 1200) == 1536
    assert parity.fitted_context(1, 1) == 256


def test_plan_runs_greedy_struct_once_and_fits_the_chart_context(tmp_path: Path) -> None:
    write_fixtures(tmp_path)
    args = argparse.Namespace(fixtures=tmp_path, arms=["plain", "mtp"], suite="both", seeds=3, route_trace_arm="mtp")
    jobs = parity.plan(args)
    community = [j for j in jobs if j[0] == "community" and j[1] == "plain"]
    names = [j[2] for j in community]
    assert names.count("struct") == 1 and names.count("conv") == 3 and names.count("prose") == 3
    assert {"prefill16384", "prefill65536"} <= set(names)
    traced = [j for j in jobs if j[8]]
    assert {j[1] for j in traced} == {"mtp"} and {j[3] for j in traced} == {0} and len(traced) == 3
    for job in jobs:
        if job[0] == "chart":
            assert len(job[4]) + job[5] <= parity.CHART_MAX_CONTEXT


def test_community_decode_tps_is_token_weighted_per_seed() -> None:
    def run(name: str, seed: int, tokens: int, seconds: float) -> parity.Run:
        report = {"tests": [{"decode_output_tok_s_mean": (tokens - 1) / seconds, "decode_seconds_mean": seconds,
                             "reps": [{"generated_output_tokens": tokens}], "speculative": {}}]}
        return parity.Run("community", "mtp", name, seed, report, 0, 1.0)

    runs = [run("struct", 0, 1001, 2.0)]
    for seed in (0, 1):
        runs += [run("conv", seed, 101, 1.0), run("prose", seed, 201, 1.0 + seed)]
    summary = parity.summarize(runs)["community"]["mtp"]
    per_seed = summary["decode_tps"]["per_seed"]
    # Seed 0: (100 + 1000 + 200) / (1 + 2 + 1); seed 1 reuses the greedy struct run.
    assert per_seed == [1300 / 4.0, 1300 / 5.0]
    assert summary["conv"]["decode_tok_s"] == 100.0
