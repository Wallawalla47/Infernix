"""The VRAM expert-cache policy replay (tools/expert_cache_replay/replay.py): Belady MIN with and
without bypass on traces small enough to solve by hand, the geometry of FreeToken and engine route
traces, and the headroom report's arms, layer groups and decision rule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tools.expert_cache_replay import host_tier as ht
from tools.expert_cache_replay import replay as rp
from tools.expert_cache_replay import route_predictor as rpred


def _belady(cap, groups, bypass):
    groups = [np.asarray(g, np.int64) for g in groups]
    sim = rp.Belady(cap, rp.next_uses(groups), bypass)
    sim.run(groups, 1)
    return sim


def test_belady_bypass_keeps_the_residents_a_far_miss_would_evict():
    # a b c a b with two slots. Demand fetch must admit c and evicts b (next use 4, farther than a's
    # 3): a hits, b misses. With bypass c's next use (never) is farther than every resident's, so it
    # is served without admission and both a and b hit.
    a, b, c = 0, 1, 2
    demand = _belady(2, [[a], [b], [c], [a], [b]], bypass=False)
    assert (demand.hits, demand.miss, demand.promo) == (1, 4, 4)
    served = _belady(2, [[a], [b], [c], [a], [b]], bypass=True)
    assert (served.hits, served.miss, served.promo) == (2, 3, 2)


def test_belady_never_evicts_the_current_group():
    # Two slots, groups of two: [a b] [c a] [b c]. In the second group a is current, so only b can
    # make room for c although a's next use (never) is the farthest; the third group then hits c
    # and misses b.
    a, b, c = 0, 1, 2
    for bypass in (False, True):
        sim = _belady(2, [[a, b], [c, a], [b, c]], bypass)
        assert (sim.hits, sim.miss) == (2, 4)
        assert set(sim.res) == {b, c}


def test_belady_bounds_the_engine_lfru_on_a_synthetic_trace():
    trace = ht.synthetic_trace(layers=4, experts=32, top_k=4, requests=4, prompt=64, decode=160, chunk=32, seed=3)
    geo = rp.Geometry(4, 32, 4)
    groups, scored, tokens, warm = rp.decode_groups([trace], geo, 0)
    nxt = rp.next_uses(groups)
    cap = 40
    bypass = rp.run_decode_arm(rp.Belady(cap, nxt, True), groups, scored, geo)
    demand = rp.run_decode_arm(rp.Belady(cap, nxt, False), groups, scored, geo)
    lfru = rp.run_decode_arm(rp.LFRU(cap), groups, scored, geo)
    engine = rp.run_engine_arm([trace], geo, cap, 0)
    assert bypass.rate() >= demand.rate() >= lfru.rate()
    assert bypass.rate() > engine.rate()
    assert bypass.keys.sum() == demand.keys.sum() == engine.keys.sum() == sum(len(g) for g in groups)


def test_route_trace_geometry_comes_from_the_header(tmp_path):
    # 3 layers x 16 experts, top-2, verification rounds of 3 columns: groups are the distinct experts
    # each decode-class round computes per layer (rejected columns included), keys layer * 16 + id.
    trace = ht.synthetic_trace(layers=3, experts=16, top_k=2, requests=2, prompt=20, decode=12, chunk=8,
                               verify_width=3, seed=5)
    path = tmp_path / 'trace.bin'
    ht.write_trace(path, trace.header, trace.rounds)
    assert rp.is_route_trace(str(path)) and not rp.is_route_trace(str(tmp_path))
    geo, traces = rp.read_route_traces([str(path)])
    assert geo == rp.Geometry(3, 16, 2)
    decode = [r for r in traces[0].rounds if r.kind in (ht.DECODE, ht.VERIFY)]
    groups, scored, tokens, warm = rp.decode_groups(traces, geo, 0)
    assert len(groups) == 3 * len(decode) and all(scored) and warm == 0
    assert tokens == sum(r.tokens for r in decode)
    for i, g in enumerate(groups):
        r, layer = decode[i // 3], i % 3
        assert g.tolist() == sorted(set((r.routes[layer].ravel() + layer * 16).tolist()))
    # Warm-up: rounds are scored once the decode tokens before them reach the warm-up.
    groups, scored, tokens, warm = rp.decode_groups(traces, geo, 5)
    first = scored.index(True) // 3
    assert sum(r.tokens for r in decode[:first]) >= 5 > sum(r.tokens for r in decode[:first - 1])
    assert warm + tokens == sum(r.tokens for r in decode)


def test_traces_of_different_geometry_are_refused(tmp_path):
    a = ht.synthetic_trace(layers=2, experts=16, top_k=2, requests=1, prompt=8, decode=4, chunk=8)
    b = ht.synthetic_trace(layers=2, experts=32, top_k=2, requests=1, prompt=8, decode=4, chunk=8)
    ht.write_trace(tmp_path / 'a.bin', a.header, a.rounds)
    ht.write_trace(tmp_path / 'b.bin', b.header, b.rounds)
    with pytest.raises(ValueError):
        rp.read_route_traces([str(tmp_path / 'a.bin'), str(tmp_path / 'b.bin')])


def test_freetoken_traces_take_their_layers_from_the_arrays(tmp_path):
    rng = np.random.default_rng(0)
    for name, steps in (('warmup', 6), ('a', 5), ('b', 4)):
        ids = np.stack([np.stack([rng.permutation(24)[:5] for _ in range(3)]) for _ in range(steps)]).astype(np.int32)
        np.savez(tmp_path / f'{name}-trace.npz', ids=ids)
    geo, seqs = rp.load(str(tmp_path), experts=24, top_k=4)
    assert geo == rp.Geometry(3, 24, 4)
    assert [s.shape for s in seqs] == [(6, 3, 4), (5, 3, 4), (4, 3, 4)]
    assert all(((s // 24) == np.arange(3)[None, :, None]).all() for s in seqs)
    with pytest.raises(ValueError):
        rp.load(str(tmp_path), experts=16, top_k=4)  # ids up to 23 do not fit 16 experts


def test_engine_arm_counts_the_hits_the_tier_model_reads():
    # The report scores the engine policy from outside TierSim: its per-layer decode hits must be the
    # model's own decode and verification VRAM hits.
    trace = ht.synthetic_trace(layers=3, experts=24, top_k=3, requests=3, prompt=48, decode=40, chunk=16,
                               verify_width=3, seed=8)
    geo = rp.Geometry(3, 24, 3)
    arm = rp.run_engine_arm([trace], geo, 30, 0, prefill_promotions=0, promotion_interval=2)
    sim = ht.TierSim(trace.header, vram=30, ram_slots=None, prior='none', prompt_seed=(ht.PROMPT_SEED_CREDIT, 0, 0),
                     prefill_promotions=0, promotion_interval=2)
    sim.run(trace.rounds)
    kinds = sim.metrics[0].by_kind, sim.metrics[1].by_kind
    assert arm.hits.sum() == sum(k[x]['vram_hits'] for k in kinds for x in (ht.DECODE, ht.VERIFY))
    assert arm.keys.sum() == sum(k[x]['computed'] for k in kinds for x in (ht.DECODE, ht.VERIFY))


def test_engine_rule_is_read_from_the_recorded_budgets():
    def round_(kind, width, budget, tokens=1):
        return ht.Round(kind, 1, width, tokens, budget, 0, np.zeros(1, np.int32), np.ones(width, bool),
                        np.zeros((1, width, 1), np.int32))
    # Qwen3.5-122B-A10B: no prefill-chunk promotions; one promotion every third decode token once full.
    rounds = [round_(ht.PREFILL, 512, 0)] + [round_(ht.DECODE, 1, 1)] * 10
    rounds += [round_(ht.DECODE, 1, 1 if i % 3 == 2 else 0) for i in range(30)]
    assert rp.engine_rule(rounds) == (0, 3)
    # Flash-Next: 16 per chunk; a short chunk (served like decode) does not count.
    rounds = [round_(ht.PREFILL, 4, 4), round_(ht.PREFILL, 1024, 16)] + [round_(ht.DECODE, 1, i % 4 == 3) for i in range(16)]
    assert rp.engine_rule(rounds) == (16, 4)


def test_layer_groups_and_the_decision_rule():
    assert rp.parse_layer_groups('', 48) == [('0-11', 0, 11), ('12-23', 12, 23), ('24-35', 24, 35), ('36-47', 36, 47)]
    assert rp.parse_layer_groups('0-19,20-47', 48) == [('0-19', 0, 19), ('20-47', 20, 47)]
    with pytest.raises(ValueError):
        rp.parse_layer_groups('40-48', 48)
    assert 'build' in rp.verdict(3.0)
    assert 'no cache-policy work' in rp.verdict(2.99)
    assert 'closes' in rp.verdict(0.99)


def test_headroom_report_runs_on_a_route_trace(tmp_path, capsys):
    trace = ht.synthetic_trace(layers=4, experts=32, top_k=4, requests=3, prompt=64, decode=96, chunk=32, seed=1)
    trace.header['frames'] = 48
    path = tmp_path / 'trace.bin'
    ht.write_trace(path, trace.header, trace.rounds)
    out = tmp_path / 'headroom.json'
    assert rp.main([str(path), '--headroom', '--capacities', '32,48', '--layer-groups', '0-1,2-3',
                    '--warmup-tokens', '16', '--json', str(out)]) == 0
    text = capsys.readouterr().out
    assert '4 layers x 32 experts, top-4' in text and 'decision at 48 frames' in text
    report = json.loads(out.read_text())
    assert report['geometry'] == {'layers': 4, 'experts': 32, 'top_k': 4} and report['decision_frames'] == 48
    assert [r['frames'] for r in report['results']] == [32, 48]
    at48 = report['results'][1]
    names = [a['arm'] for a in at48['arms']]
    assert names[0] == 'engine LFRU (rule budgets)' and 'engine LFRU (recorded budgets)' in names
    assert any('bypass' in n for n in names) and any('demand-fetch' in n for n in names)
    engine, bypass = at48['arms'][0], next(a for a in at48['arms'] if 'bypass' in a['arm'])
    assert report['decision_gap_points'] == pytest.approx(100 * (bypass['hit']['all'] - engine['hit']['all']))
    # The groups' hit rates are the per-layer rates' key-weighted means; every rate is a fraction.
    assert all(0.0 <= v <= 1.0 for a in at48['arms'] for v in a['hit'].values())


def _decode_round(routes):
    """A one-row decode round computing `routes` ([layers, top_k] expert ids)."""
    routes = np.asarray(routes, np.int32)[:, None, :]
    return ht.Round(ht.DECODE, 1, 1, 1, 1, ht.UNKNOWN_POSITION, np.zeros(1, np.int32), np.ones(1, bool), routes)


def test_guarded_residents_are_evicted_last():
    # Two frames hold experts 0 and 1 of one layer; 0 is the least frequent, so an admission evicts it
    # unless it is guarded. With both residents guarded, protection gives way to the LFRU order.
    header = {'layers': 1, 'experts': 8, 'top_k': 1}
    for guard, evicted in ((None, 0), ({0}, 1), ({0, 1}, 0)):
        sim = ht.TierSim(header, vram=2, ram_slots=None, prior='none', prompt_seed=None, promotion_interval=1)
        sim.guard = (lambda guarded: (lambda i, r: np.array(sorted(guarded), np.int64)) if guarded else None)(guard)
        for e in (0, 1, 1, 2, 2, 2):
            sim.step(_decode_round([[e]]))
        resident = set(int(k) for k in sim.v_keys[:sim.v_n])
        assert resident == {0, 1, 2} - {evicted}, (guard, resident)
        assert not sim.guard_mask.any()


def test_next_rounds_oracle_names_the_next_decode_rounds_keys():
    trace = ht.synthetic_trace(layers=2, experts=16, top_k=2, requests=2, prompt=16, decode=6, chunk=8, seed=4)
    rounds = trace.rounds
    oracle = rpred.NextRoundsOracle(rounds, 2, 16, 2)
    decode = [i for i, r in enumerate(rounds) if r.kind == ht.DECODE]
    base = np.array([[0], [16]])
    for i in range(len(rounds)):
        ahead = [j for j in decode if j > i][:2]
        want = sorted(set(int(k) for j in ahead for k in (rounds[j].routes.reshape(2, -1) + base).ravel()))
        got = oracle(i, rounds[i])
        assert (got is None and not want) or got.tolist() == want


def test_transition_predictor_learns_a_lane_sequence():
    # One layer cycling 0 -> 3 -> 5 -> 0 (top-1): after a few cycles each token predicts its successor,
    # and a second lane's history does not leak into the first lane's transitions.
    predictor = rpred.TransitionPredictor(1, 8, per_layer=1)
    cycle = [0, 3, 5]
    for t in range(12):
        r = _decode_round([[cycle[t % 3]]])
        guard = predictor(t, r)
        if t >= 3:
            assert guard.tolist() == [cycle[(t + 1) % 3]]
    other = ht.Round(ht.DECODE, 1, 1, 1, 1, ht.UNKNOWN_POSITION, np.ones(1, np.int32), np.ones(1, bool),
                     np.array([[[7]]], np.int32))
    predictor.observe(other)
    assert predictor.predict([0]).tolist() == [cycle[12 % 3]]
    assert predictor.predict().size == 0  # the last round's lane (1) has no successor counts yet
    assert predictor.counts[0, 5, 7] == 0  # lane 1's first token follows nothing


def test_transition_predictor_rescales_and_caps_a_rounds_tokens():
    # A half-life of one token doubles the increment every token: the counts are rescaled once it
    # passes 1e20, and the ratios the prediction reads survive (ties go to the lower expert).
    predictor = rpred.TransitionPredictor(1, 4, per_layer=2, half_life=1.0)
    for t in range(80):
        predictor.observe(_decode_round([[t % 2]]))
    assert predictor.weight <= rpred.RESCALE_ABOVE and np.isfinite(predictor.counts).all()
    assert predictor.predict().tolist() == [0]  # after 1 always comes 0; 1 itself never follows 1
    # A prompt chunk contributes its last MAX_ROUND_TOKENS tokens: 299 earlier pairs are not counted.
    chunk = 300 + rpred.MAX_ROUND_TOKENS
    routes = np.tile(np.array([[[2], [3]]], np.int32), (1, chunk // 2, 1))
    prompt = ht.Round(ht.PREFILL, 1, chunk, chunk, 0, 0, np.full(1, 5, np.int32), np.ones(chunk, bool), routes)
    fresh = rpred.TransitionPredictor(1, 4, per_layer=1)
    fresh.observe(prompt)
    assert fresh.out[0].sum() == pytest.approx(sum(fresh.growth ** i for i in range(rpred.MAX_ROUND_TOKENS - 1)), rel=1e-5)


def test_protection_report_runs_on_a_route_trace(tmp_path, capsys):
    trace = ht.synthetic_trace(layers=4, experts=32, top_k=4, requests=3, prompt=64, decode=96, chunk=32,
                               verify_width=3, seed=2)
    trace.header['frames'] = 48
    path = tmp_path / 'trace.bin'
    ht.write_trace(path, trace.header, trace.rounds)
    out = tmp_path / 'protection.json'
    assert rp.main([str(path), '--headroom', '--capacities', '48', '--protection', '--protect-horizons', '1,4',
                    '--json', str(out)]) == 0
    text = capsys.readouterr().out
    assert 'protection at 48 frames' in text
    report = json.loads(out.read_text())
    names = [a['arm'] for a in report['results'][0]['arms']]
    assert any('next 1 round (oracle)' in n for n in names) and any('next 4 rounds (oracle)' in n for n in names)
    assert 'engine LFRU + transition predictor, 8 per layer' in names
    gains = report['results'][0]['protection_gain_points']
    assert report['protection_ceiling_points'] == max(v for k, v in gains.items() if 'oracle' in k)
    assert report['protection_verdict'] == rp.protection_verdict(report['protection_gain_points'],
                                                                 report['protection_ceiling_points'])
    assert 'A/B' in rp.protection_verdict(3.0, 5.0) and 'closes' in rp.protection_verdict(1.0, 2.0)
    assert 'no A/B' in rp.protection_verdict(1.0, 4.0)
