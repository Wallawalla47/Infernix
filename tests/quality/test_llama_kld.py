"""llama.cpp-format KLD statistics (tools/quality/llama_kld.py): bootstrap intervals over chunks, the weighted median,
paired differences, and llama-perplexity's printout."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.quality import llama_kld

PRINTOUT = """llama_perplexity: some log line
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1       6.1234 ±    0.3000       0.00123 ±    0.00045       0.00100 ±    0.00010    0.900 ±  0.100 %    99.000 ±  0.500 %

====== Perplexity statistics ======
Mean PPL(Q)                   :   6.123400 ±   0.300000
Mean PPL(base)                :   6.115900 ±   0.299000
Cor(ln(PPL(Q)), ln(PPL(base))):  99.95%
Mean ln(PPL(Q)/PPL(base))     :   0.001230 ±   0.000450

====== KL divergence statistics ======
Mean    KLD:   0.001000 ±   0.000100
Maximum KLD:   1.250000
99.9%   KLD:   0.300000
Median  KLD:   0.000210
 0.1%   KLD:  -0.000001

====== Token probability statistics ======
Mean    Δp: -0.012 ± 0.004 %
99.0%   Δp:  1.234%
RMS Δp    :  0.900 ± 0.100 %
Same top p: 99.000 ± 0.500 %
"""


def _arm(tmp_path, name, kld, same_top, chunks, dnll=0.0):
    prefix = tmp_path / name
    n = len(kld)
    chunk = np.repeat(np.arange(chunks), n // chunks)
    rows = np.column_stack([chunk, np.arange(n), kld, np.zeros(n), same_top.astype(int), np.full(n, 2.0 + dnll),
                            np.full(n, 2.0)])
    np.savetxt(str(prefix) + ".positions.csv", rows, delimiter=",", header="chunk,position,kld,p_diff,same_top,nll,nll_base",
               comments="")
    (tmp_path / f"{name}.json").write_text(json.dumps({"arm": "llama_prefill", "route": "default"}), encoding="utf-8")
    return llama_kld.load(prefix)


def test_point_statistics_follow_their_definitions(tmp_path):
    rng = np.random.default_rng(1)
    kld = rng.exponential(1e-3, 2000)
    kld[:5] = 0.5  # beyond the clip
    same = rng.random(2000) > 0.02
    arm = _arm(tmp_path, "a", kld, same, 20, dnll=1e-3)
    p = llama_kld.point(arm)
    assert p["median_kld"] == pytest.approx(np.median(kld))
    assert p["clipped_kld"] == pytest.approx(np.minimum(kld, 0.1).mean())
    assert p["flip_rate"] == pytest.approx(1 - same.mean())
    assert p["mean_dnll"] == pytest.approx(1e-3)


def test_weighted_median_matches_the_resampled_sample():
    rng = np.random.default_rng(2)
    values = rng.random(60)
    inverse = np.repeat(np.arange(6), 10)
    weights = rng.multinomial(6, np.full(6, 1 / 6), size=50).astype(float)
    got = llama_kld._weighted_quantiles(values, inverse, weights, 0.5)
    for row, w in zip(got, weights):
        sample = np.concatenate([np.repeat(values[inverse == u], int(c)) for u, c in enumerate(w)])
        ordered = np.sort(sample)
        # The lower middle value of an even sample.
        assert row == ordered[(len(ordered) - 1) // 2]


def test_intervals_cover_and_paired_differences_cancel(tmp_path):
    rng = np.random.default_rng(3)
    kld = rng.exponential(1e-3, 4000)
    same = rng.random(4000) > 0.02
    a = _arm(tmp_path, "a", kld, same, 40)
    t = llama_kld.table(a, resamples=2000)
    for m in llama_kld.GATED:
        assert t[m]["low"] <= t[m]["value"] <= t[m]["high"]
    b = _arm(tmp_path, "b", kld * 1.5, same, 40)
    d = llama_kld.paired(b, a, resamples=2000)
    assert d["median_kld"]["low"] > 0 and d["flip_rate"]["value"] == 0
    same_arm = llama_kld.paired(a, a, resamples=500)
    assert all(v["low"] == v["high"] == 0 for v in same_arm.values())
    with pytest.raises(ValueError):
        llama_kld.paired(a, _arm(tmp_path, "c", kld[:2000], same[:2000], 20))


def test_printout_is_read_line_by_line():
    parsed = llama_kld.parse_printout(PRINTOUT)
    assert parsed["Mean KLD"] == (0.001, 0.0001)
    assert parsed["Median KLD"] == (0.00021, None)
    assert parsed["99.9% KLD"] == (0.3, None)
    assert parsed["0.1% KLD"] == (-0.000001, None)
    assert parsed["Cor(ln(PPL(Q)), ln(PPL(base)))"] == (99.95, None)
    assert parsed["Mean ln(PPL(Q)/PPL(base))"] == (0.00123, 0.00045)
    assert parsed["Same top p"] == (99.0, 0.5)
    assert parsed["RMS Δp"] == (0.9, 0.1)
    assert "chunk" not in " ".join(parsed)  # the per-chunk table is not the summary
    rows = llama_kld.printout_diff(PRINTOUT, PRINTOUT.replace("0.000210", "0.000211"))
    assert [label for label, *_, same in rows if not same] == ["Median KLD"]
