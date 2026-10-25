"""The engine comparison's plan and report (tools/quality/engine_compare.py): which arms a weight set takes, how
the fast-path arms follow the default arms' selection, and the decision rule's labels on synthetic arms."""

from __future__ import annotations

import json

import numpy as np

from tools.quality import engine_compare, llama_kld


def _config(tmp_path, **extra):
    config = {"name": "test", "certify": "certify.exe", "artifact": "model.infernix", "text": "wiki.test.raw",
              "work": str(tmp_path), "kv": "int8", "route_args": ["--ngram-volume", "v.ngram"],
              "spec": ["--spec", "mtp", "--draft-tokens", "auto"], **extra}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return engine_compare._load_config(path)


def _meta(tmp_path, arm_id, **meta):
    (tmp_path / f"{arm_id}.json").write_text(json.dumps(meta), encoding="utf-8")


def _ids(steps):
    return [s.id for s in steps]


def test_the_static_arms_and_the_reference(tmp_path):
    config = _config(tmp_path)
    steps, notes = engine_compare.plan(config)
    assert _ids(steps) == ["reference", "P-twin-s1", "P-twin-s2", "P-reproducible-bf16", "P-reproducible-int8",
                           "P-default-int8", "D-reproducible-int8-c8", "D-reproducible-int8-c1-check", "D-default-int8-c1",
                           "D-default-int8-c8", "D-reproducible-int8-spec-c1", "D-default-int8-spec-c1"]
    reference = steps[0].argv
    assert reference[1:3] == ["reference", "model.infernix"] and "--ngram-volume" in reference
    assert reference[reference.index("--chunks") + 1] == "0"
    twin = steps[1].argv
    assert twin[1] == "prefill" and twin[-3:] == ["--lossless", "--noise-seed", "1"]
    assert any("follow once" in note for note in notes)


def test_paths_follow_the_default_arms(tmp_path):
    config = _config(tmp_path)
    _meta(tmp_path, "P-default-int8", load={"fast_paths": ["prompt-pv8", "fp16-gdn-state"]})
    _meta(tmp_path, "D-default-int8-c1", load={"fast_paths": ["prompt-pv8", "batched-decode", "batched-a4-decode",
                                                              "preemption", "fp16-gdn-state"]})
    _meta(tmp_path, "D-default-int8-spec-c1", load={"fast_paths": ["batched-decode", "adaptive-speculation", "adaptive-mtp"]})
    steps, notes = engine_compare.plan(config)
    by_id = {s.id: s for s in steps}
    assert "P-path-prompt-pv8" in by_id and "D-path-prompt-pv8" not in by_id  # prompt only
    assert by_id["P-path-prompt-pv8"].base == "P-reproducible-int8"
    assert {"P-path-fp16-gdn-state", "D-path-fp16-gdn-state"} <= set(by_id)
    assert by_id["D-path-batched-decode"].base == "D-reproducible-int8-c8"
    a4 = by_id["D-path-batched-a4-decode"]
    assert a4.base == "D-path-batched-decode"
    flags = [a4.argv[i + 1] for i, part in enumerate(a4.argv) if part == "--fast-path"]
    assert flags == ["batched-decode", "batched-a4-decode"] and "--reproducible" in a4.argv
    mtp = by_id["D-path-adaptive-mtp"]
    assert mtp.base == "D-path-adaptive-speculation" and "--spec" in mtp.argv
    assert mtp.argv[mtp.argv.index("--concurrency") + 1] == "1"
    assert by_id["D-path-adaptive-speculation"].base == "D-reproducible-int8-spec-c1"
    assert any(note.startswith("preemption: not measured") for note in notes)
    assert not any("follow once" in note for note in notes)


def _arm(tmp_path, arm_id, kld, same_top, chunks=40, nll_shift=0.0, **meta):
    n = len(kld)
    chunk = np.repeat(np.arange(chunks), n // chunks)
    llama_kld.write_positions(tmp_path / arm_id, chunk, np.arange(n), kld, np.zeros(n), same_top, 2.0 + nll_shift + kld,
                              np.full(n, 2.0), 0.95 * kld)
    _meta(tmp_path, arm_id, arm="llama_prefill", seconds=1.0, **meta)


def test_the_report_labels_each_path(tmp_path):
    config = _config(tmp_path, decode=False, spec=[], paths=["prompt-pv8", "fp16-gdn-state", "router-mma"])
    rng = np.random.default_rng(5)
    n = 8000
    _meta(tmp_path, "reference.kld", chunks=40)
    floor = rng.exponential(2e-3, n)
    for seed in (1, 2):
        _arm(tmp_path, f"P-twin-s{seed}", rng.permutation(floor), rng.random(n) > 0.01)
    base = rng.exponential(1e-3, n)
    same = rng.random(n) > 0.005
    for arm_id in ("P-reproducible-bf16", "P-reproducible-int8", "P-default-int8"):
        _arm(tmp_path, arm_id, base, same)
    _arm(tmp_path, "P-path-prompt-pv8", base, same)                       # bit-identical: no effect
    _arm(tmp_path, "P-path-fp16-gdn-state", base * 1.01, same)            # +1 % of 1e-3: within 0.25 x 2e-3
    flipped = same.copy()
    flipped[: n // 20] = False
    _arm(tmp_path, "P-path-router-mma", base * 3.0, flipped)              # far beyond
    result = engine_compare.report(config, resamples=400)
    labels = {p["path"]: p["verdict"]["label"] for p in result["paths"]}
    assert labels == {"prompt-pv8": "no effect", "fp16-gdn-state": "rounding-level", "router-mma": "changes outputs"}
    assert result["floor"]["mean_kld"] > 1.5e-3
    md = result["markdown"]
    assert "Rounding floor (noise twin, seed 1)" in md and "Infernix default (int8 KV)" in md
    assert "Fast paths, one at a time" in md
    check = [c for c in result["checks"] if c["check"] == "decode KLD form"]
    assert check and abs(check[0]["ratio_low"] - 0.95) < 1e-6


def test_an_engine_printout_joins_the_table(tmp_path):
    printout = (tmp_path / "llama.txt")
    config = _config(tmp_path, decode=False, spec=[], paths=[],
                     engines=[{"name": "llama.cpp b10221", "kind": "llama-perplexity", "binary": "lp.exe", "model": "m.gguf",
                               "args": ["-ngl", "99"], "settings": "-b 512"}])
    steps, _ = engine_compare.plan(config)
    engine = steps[-1]
    assert engine.id == "E-llama.cpp-b10221" and "--kl-divergence" in engine.argv and "-ngl" in engine.argv
    printout.write_text("====== Perplexity statistics ======\n"
                        "Mean PPL(Q)                   :   6.123400 ±   0.300000\n"
                        "Mean ln(PPL(Q)/PPL(base))     :   0.001230 ±   0.000450\n"
                        "====== KL divergence statistics ======\n"
                        "Mean    KLD:   0.001000 ±   0.000100\n"
                        "99.9%   KLD:   0.300000\n"
                        "====== Token probability statistics ======\n"
                        "Same top p: 99.000 ± 0.500 %\n", encoding="utf-8")
    _meta(tmp_path, engine.id, arm="engine", engine="llama.cpp b10221", printout=str(printout))
    _meta(tmp_path, "reference.kld", chunks=40)
    result = engine_compare.report(config, resamples=50)
    assert "llama.cpp b10221 (-b 512)" in result["markdown"]
    assert result["arms"][engine.id]["stats"]["mean_kld"]["value"] == 0.001
