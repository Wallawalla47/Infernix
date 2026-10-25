"""llama.cpp-format KLD statistics (tools/quality/llama_kld.py): the published statistics, bootstrap intervals over
chunks, the weighted quantile, paired differences, and llama-perplexity's printout."""

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


def _arm(tmp_path, name, kld, same_top, chunks, dnll=0.0, kld_top=None):
    prefix = tmp_path / name
    n = len(kld)
    chunk = np.repeat(np.arange(chunks), n // chunks)
    llama_kld.write_positions(prefix, chunk, np.arange(n), kld, np.zeros(n), same_top, np.full(n, 2.0 + dnll),
                              np.full(n, 2.0), kld if kld_top is None else kld_top)
    (tmp_path / f"{name}.json").write_text(json.dumps({"arm": "llama_prefill", "route": "default"}), encoding="utf-8")
    return llama_kld.load(prefix)


def test_point_statistics_follow_their_definitions(tmp_path):
    rng = np.random.default_rng(1)
    kld = rng.exponential(1e-3, 2000)
    same = rng.random(2000) > 0.02
    arm = _arm(tmp_path, "a", kld, same, 20, dnll=1e-3, kld_top=kld * 0.9)
    p = llama_kld.point(arm)
    assert p["mean_kld"] == pytest.approx(kld.mean())
    assert p["p999_kld"] == pytest.approx(np.quantile(kld, 0.999))
    assert p["same_top"] == pytest.approx(same.mean())
    assert p["ln_ppl_ratio"] == pytest.approx(1e-3)
    assert p["ppl"] == pytest.approx(np.exp(2.001))
    assert p["mean_kld_top"] == pytest.approx(0.9 * kld.mean())


def test_an_arm_without_the_observed_form_reports_none(tmp_path):
    kld = np.full(400, 1e-3)
    arm = _arm(tmp_path, "full", kld, np.ones(400, bool), 4, kld_top=np.full(400, np.nan))
    assert "mean_kld_top" not in llama_kld.point(arm)
    assert "mean_kld_top" not in llama_kld.table(arm, resamples=50)


def test_columns_are_checked(tmp_path):
    prefix = tmp_path / "old"
    (tmp_path / "old.positions.csv").write_text("chunk,position,kld,p_diff,same_top,nll,nll_base\n0,1,0,0,1,1,1\n")
    (tmp_path / "old.json").write_text("{}")
    with pytest.raises(ValueError):
        llama_kld.load(prefix)


def test_weighted_quantile_matches_the_resampled_sample():
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
    for m in llama_kld.METRICS:
        assert t[m]["low"] <= t[m]["value"] <= t[m]["high"]
    b = _arm(tmp_path, "b", kld * 1.5, same, 40)
    d = llama_kld.paired(b, a, resamples=2000)
    assert d["mean_kld"]["low"] > 0 and d["same_top"]["value"] == 0
    same_arm = llama_kld.paired(a, a, resamples=500)
    assert all(v["low"] == v["high"] == 0 for v in same_arm.values())
    assert llama_kld.identical(a, _arm(tmp_path, "a2", kld, same, 40)) and not llama_kld.identical(a, b)
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


def test_head_keeps_the_first_chunks(tmp_path):
    context, vocab, chunks = 8, 5, 3
    words = 2 * ((vocab + 1) // 2) + 4
    rows = context - 1 - context // 2
    tokens = np.arange(context * chunks, dtype="<i4")
    body = np.arange(chunks * rows * words, dtype="<u2")
    base = tmp_path / "base.kld"
    base.write_bytes(b"_logits_" + np.array([context, vocab, chunks], "<i4").tobytes() + tokens.tobytes() + body.tobytes())
    out = tmp_path / "head.kld"
    llama_kld.head(base, 2, out)
    data = out.read_bytes()
    assert data[:8] == b"_logits_" and np.frombuffer(data[8:20], "<i4").tolist() == [context, vocab, 2]
    assert np.array_equal(np.frombuffer(data[20:20 + 4 * 2 * context], "<i4"), tokens[: 2 * context])
    assert np.array_equal(np.frombuffer(data[20 + 4 * 2 * context:], "<u2"), body[: 2 * rows * words])
    with pytest.raises(ValueError):
        llama_kld.head(base, 4, tmp_path / "too-many.kld")


def test_a_printout_stands_for_an_engine_without_positions():
    p = llama_kld.printout_point(PRINTOUT)
    assert p["mean_kld"]["value"] == 0.001 and p["mean_kld"]["high"] == pytest.approx(0.001 + 1.96e-4)
    assert p["same_top"]["value"] == pytest.approx(0.99) and np.isnan(p["p999_kld"]["low"])
    assert p["ln_ppl_ratio"]["value"] == pytest.approx(0.00123)
