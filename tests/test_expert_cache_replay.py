"""The VRAM expert-cache policy replay (tools/expert_cache_replay/replay.py): Belady MIN with and
without bypass on traces small enough to solve by hand, the geometry of FreeToken and engine route
traces, the headroom report's arms, layer groups and decision rule, and the engine arm's residency
rules (ExpertResidency: count halving, per-layer protection, fill-phase landing, warm start, frame
changes, pools) with the engine inputs it reads (saved state, calibrated order, engine log)."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from tools.expert_cache_replay import engine_inputs as ei
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
    # No prefill-chunk promotions (both engines today); one promotion every third decode token once full.
    rounds = [round_(ht.PREFILL, 512, 0)] + [round_(ht.DECODE, 1, 1)] * 10
    rounds += [round_(ht.DECODE, 1, 1 if i % 3 == 2 else 0) for i in range(30)]
    assert rp.engine_rule(rounds) == (0, 3)
    # A recorded chunk budget is read whatever it is; a short chunk (served like decode) does not count.
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
    # At the trace's own frames the engine arm takes the recorded budgets (the rule beside it); elsewhere the rule.
    assert report['results'][0]['arms'][0]['arm'] == 'engine LFRU (rule budgets)'
    at48 = report['results'][1]
    names = [a['arm'] for a in at48['arms']]
    assert names[0] == 'engine LFRU (recorded budgets)' and 'engine LFRU (rule budgets)' in names
    assert 'Belady MIN with bypass, engine budgets' in names
    assert any('CPU-serve' in n for n in names) and any('demand-fetch' in n for n in names)
    engine = at48['arms'][0]
    bypass = next(a for a in at48['arms'] if 'CPU-serve' in a['arm'])
    budgeted = next(a for a in at48['arms'] if a['arm'] == 'Belady MIN with bypass, engine budgets')
    assert report['decision_gap_points'] == pytest.approx(100 * (bypass['hit']['all'] - engine['hit']['all']))
    assert report['decision_budgeted_gap_points'] == pytest.approx(100 * (budgeted['hit']['all'] - engine['hit']['all']))
    # The groups' hit rates are the per-layer rates' key-weighted means; every rate is a fraction.
    assert all(0.0 <= v <= 1.0 for a in at48['arms'] for v in a['hit'].values())


def _decode_round(routes, frames=0):
    """A one-row decode round computing `routes` ([layers, top_k] expert ids), run with `frames` frames."""
    routes = np.asarray(routes, np.int32)[:, None, :]
    return ht.Round(ht.DECODE, 1, 1, 1, 1, ht.UNKNOWN_POSITION, np.zeros(1, np.int32), np.ones(1, bool), routes,
                    frames)


def _resident(sim):
    return set(int(k) for k in sim.v_keys[:sim.v_n])


def test_guarded_residents_are_evicted_last():
    # Two frames hold experts 0 and 1 of one layer; 0 is the least frequent, so an admission evicts it
    # unless it is guarded. With both residents guarded, protection gives way to the LFRU order. In one
    # layer a round-wide guard and a layer guard protect alike.
    header = {'layers': 1, 'experts': 8, 'top_k': 1}
    for guard, evicted in ((None, 0), ({0}, 1), ({0, 1}, 0)):
        for layer_scope in (False, True):
            sim = ht.TierSim(header, vram=2, ram_slots=None, prior='none', prompt_seed=None, promotion_interval=1)
            keys = np.array(sorted(guard or ()), np.int64)
            if guard is not None:
                sim.guard = (lambda i, r: (lambda layer: keys)) if layer_scope else (lambda i, r: keys)
            for e in (0, 1, 1, 2, 2, 2):
                sim.step(_decode_round([[e]]))
            assert _resident(sim) == {0, 1, 2} - {evicted}, (guard, layer_scope, _resident(sim))
            assert not sim.guard_mask.any()


def test_a_layer_guard_protects_only_its_layers_predictions():
    # Two layers of four experts (keys 0-3 and 4-7) share three frames. Layer 1's admission of key 5 must
    # evict, and key 1 (layer 0, used once long ago) is the lowest score. The engine's guard is
    # predict(layer) during that layer's step only, so layer 0's prediction of key 1 does not spare it
    # from layer 1's step; the oracle's round-wide guard spares it at every step, and key 4 goes instead.
    header = {'layers': 2, 'experts': 4, 'top_k': 1}
    rounds = [[[1], [0]], [[0], [0]], [[0], [1]]]
    layer_guard = lambda i, r: (lambda layer: np.array([1] if layer == 0 else [], np.int64))
    round_guard = lambda i, r: np.array([1], np.int64)
    for guard, resident in ((None, {0, 4, 5}), (layer_guard, {0, 4, 5}), (round_guard, {0, 1, 5})):
        sim = ht.TierSim(header, vram=3, ram_slots=None, prior='none', prompt_seed=None, promotion_interval=1)
        sim.guard = guard
        for routes in rounds:
            sim.step(_decode_round(routes))
        assert _resident(sim) == resident
        assert not sim.guard_mask.any()


def test_count_halving_changes_an_admission_exactly_at_its_period():
    # Two frames hold experts 0 and 1 of one layer (one tick a round), used at ticks 1-3; at tick 4
    # experts 2 and 3 miss with a budget of one. Unhalved, 2's score 1 does not beat the victim's 3 / 2;
    # halved at that very tick (period 4), the victim's 1.5 / 2 does not hold it. Period 5 halves one
    # tick later, too late.
    header = {'layers': 1, 'experts': 8, 'top_k': 2}
    for halving, resident in ((0, {0, 1}), (4, {1, 2}), (5, {0, 1})):
        sim = ht.TierSim(header, vram=2, ram_slots=None, prior='none', prompt_seed=None, promotion_interval=1,
                         halving=halving)
        for routes in ([[0, 1]], [[0, 1]], [[0, 1]], [[2, 3]]):
            sim.step(_decode_round(routes))
        assert _resident(sim) == resident, halving
    # The engine's period is 8,192 ticks scaled by the pool's share of the layers.
    sim = ht.TierSim({'layers': 48, 'experts': 4, 'top_k': 1}, vram=8, ram_slots=None, prior='none',
                     pools=[range(0, 12), range(12, 48)])
    assert sim.pool_period == [2048, 6144]


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
            assert guard(0).tolist() == [cycle[(t + 1) % 3]]
    other = ht.Round(ht.DECODE, 1, 1, 1, 1, ht.UNKNOWN_POSITION, np.ones(1, np.int32), np.ones(1, bool),
                     np.array([[[7]]], np.int32))
    predictor.observe(other)
    assert predictor.predict(0, [0]).tolist() == [cycle[12 % 3]]
    assert predictor.predict(0).size == 0  # the last round's lane (1) has no successor counts yet
    assert predictor.counts[0, 5, 7] == 0  # lane 1's first token follows nothing


def test_transition_predictor_guards_each_layer_with_its_own_predictions():
    # Layer 0 alternates experts 0 and 1, layer 1 experts 2 and 3: after a token routed to 1 and 3 the
    # guard names key 0 for layer 0's step and key 4 + 2 for layer 1's, never the other layer's.
    predictor = rpred.TransitionPredictor(2, 4, per_layer=1)
    for t in range(6):
        guard = predictor(t, _decode_round([[t % 2], [2 + t % 2]]))
    assert guard(0).tolist() == [0] and guard(1).tolist() == [4 + 2]


def test_transition_predictor_rescales_and_caps_a_rounds_tokens():
    # A half-life of one token doubles the increment every token: the counts are rescaled once it
    # passes 1e20, and the ratios the prediction reads survive (ties go to the lower expert).
    predictor = rpred.TransitionPredictor(1, 4, per_layer=2, half_life=1.0)
    for t in range(80):
        predictor.observe(_decode_round([[t % 2]]))
    assert predictor.weight <= rpred.RESCALE_ABOVE and np.isfinite(predictor.counts).all()
    assert predictor.predict(0).tolist() == [0]  # after 1 always comes 0; 1 itself never follows 1
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
    assert 'engine LFRU + next 1 round (oracle, round-wide guard)' in names
    assert 'engine LFRU + next 4 rounds (oracle, round-wide guard)' in names
    assert 'engine LFRU + transition predictor, 8 per layer (layer guard)' in names
    gains = report['results'][0]['protection_gain_points']
    assert report['protection_ceiling_points'] == max(v for k, v in gains.items() if 'oracle' in k)
    assert report['protection_verdict'] == rp.protection_verdict(report['protection_gain_points'],
                                                                 report['protection_ceiling_points'])
    assert 'A/B' in rp.protection_verdict(3.0, 5.0) and 'closes' in rp.protection_verdict(1.0, 2.0)
    assert 'no A/B' in rp.protection_verdict(1.0, 4.0)


def test_landings_use_up_the_layer_budget():
    # The fill phase (16 frames free, a budget of one): a decode round's three misses land in reserved
    # frames and are adopted, which uses up the layer's promotion. Without landing one is promoted. With
    # 1/3 of the misses kept on the link the CPU serves the other two (the fewest-column ones, then in job
    # order) and only the staged one lands; a cap of one CPU job leaves two staged.
    header = {'layers': 1, 'experts': 16, 'top_k': 3}
    cases = (({}, {2, 5, 9}, 0, 3), ({'landing': False}, {2}, 1, 0), ({'link_divisor': 3}, {9}, 0, 1),
             ({'link_divisor': 3, 'cpu_jobs': 1}, {5, 9}, 0, 2))
    for options, resident, promotions, landed in cases:
        sim = ht.TierSim(header, vram=16, ram_slots=None, prior='none', prompt_seed=None, **options)
        sim.admission_log = []
        sim.step(_decode_round([[5, 2, 9]]))
        assert (_resident(sim), sim.promotions, sim.landed) == (resident, promotions, landed), options
        assert sim.admission_log == [(0, 0, promotions, landed, 1)]
        assert sim.vram_visible(np.array(sorted(resident))).all()  # read from their frames in the next round


def test_the_warm_start_fills_at_most_half_the_frames_with_capped_counts():
    header = {'layers': 2, 'experts': 8, 'top_k': 1}
    counts = np.zeros(16, np.uint32)
    counts[[3, 9, 12]] = [40, 7, 100]
    ranked = np.array([12, 3, 9, 1, 0, 8, 2, 10, 11], np.uint32)
    sim = ht.TierSim(header, vram=10, ram_slots=None, prior='none', warm_start=(counts, ranked))
    # One pool: int(0.5 x 10) = 5 experts, best first, published before the first round; counts capped at 16.
    assert sim.seeded_keys == [12, 3, 9, 1, 0] and _resident(sim) == {12, 3, 9, 1, 0}
    assert sim.vram_visible(np.array(sim.seeded_keys)).all()
    assert sim.v_count[[3, 9, 12]].tolist() == [16.0, 7.0, 16.0] and sim.promotions == 0
    # Seeds do not end the fill phase: the first decode round still promotes one per layer and token.
    sim.step(_decode_round([[5], [6]]))
    assert sim.applied_budgets == [1]
    # Two pools of one layer each: five frames each, each loading its share 5 x 5 / 10 = 2 of its own keys.
    sim = ht.TierSim(header, vram=10, ram_slots=None, prior='none', warm_start=(counts, ranked), pools=[[0], [1]])
    assert sim.pool_V == [5, 5] and sim.seeded_keys == [3, 1, 12, 9]


def test_a_recorded_frame_shrink_evicts_the_lowest_scores_and_growth_adds_room():
    # Rounds carry the frames they ran with. Four experts land (scores at tick 4: 1/4, 1/3, 1/2, 1); a
    # round at 2 frames first evicts the two lowest, and back at 4 a free frame takes the next landing.
    header = {'layers': 1, 'experts': 8, 'top_k': 1, 'frames': 4}
    sim = ht.TierSim(header, vram=4, ram_slots=None, prior='none', prompt_seed=None, follow_frames=True)
    for e in range(4):
        sim.step(_decode_round([[e]], frames=4))
    assert _resident(sim) == {0, 1, 2, 3}
    sim.step(_decode_round([[3]], frames=2))
    assert sim.V == 2 and _resident(sim) == {2, 3}
    sim.step(_decode_round([[5]], frames=4))
    assert sim.V == 4 and _resident(sim) == {2, 3, 5}
    # Another capacity moves by the trace's change from its startup count.
    sim = ht.TierSim(header, vram=6, ram_slots=None, prior='none', prompt_seed=None, follow_frames=True)
    sim.step(_decode_round([[0]], frames=2))
    assert sim.V == 4


def test_budget_matched_belady_keeps_the_engine_budgets_and_bounds_the_engine():
    trace = ht.synthetic_trace(layers=4, experts=32, top_k=4, requests=4, prompt=64, decode=160, chunk=32,
                               verify_width=3, seed=3)
    geo = rp.Geometry(4, 32, 4)
    engine = rp.run_engine_arm([trace], geo, 40, 0)
    budgets = engine.sim.applied_budgets
    assert len(budgets) == len(trace.rounds)
    sim = ht.TierSim(trace.header, vram=40, ram_slots=None, prior='none', prompt_seed=(ht.PROMPT_SEED_CREDIT, 0, 0),
                     vram_policy='belady', next_use=ht.NextUse(trace.rounds, 4, 32), budgets=budgets)
    sim.admission_log = []
    sim.run(trace.rounds)
    assert len(sim.admission_log) == 4 * len(trace.rounds)
    assert all(budget == budgets[i] for i, _, _, _, budget in sim.admission_log)
    assert all(admitted <= max(0, budget - adopted) for _, _, admitted, adopted, budget in sim.admission_log)
    belady = rp.run_engine_arm([trace], geo, 40, 0, budgets=budgets, vram_policy='belady',
                               next_use=ht.NextUse(trace.rounds, 4, 32))
    assert belady.keys.sum() == engine.keys.sum()
    assert belady.rate() >= engine.rate()


def _write_state(path, identity, counts, ranked):
    """An expert state file as save_expert_state writes it."""
    data = b'NINFQ4ES' + struct.pack('<II', 1, len(identity)) + identity.encode()
    data += struct.pack('<I', len(counts)) + np.asarray(counts, '<u4').tobytes()
    data += struct.pack('<I', len(ranked)) + np.asarray(ranked, '<u4').tobytes() + b'NINFQ4ES'
    path.write_bytes(data)


def test_saved_states_and_the_calibrated_order_read_as_the_engine_reads_them(tmp_path):
    path = tmp_path / 'experts.state'
    counts = np.arange(4 * 32, dtype=np.uint32) % 23
    _write_state(path, 'artifact-id', counts, [7, 100, 3])
    state = ei.read_expert_state(path)
    assert state.identity == 'artifact-id' and state.counts.tolist() == counts.tolist()
    assert state.ranked.tolist() == [7, 100, 3]
    path.write_bytes(path.read_bytes()[:-8])  # no footer
    with pytest.raises(ValueError):
        ei.read_expert_state(path)
    # The calibrated order: each layer's counts are its ranks reversed (a permutation of 1..512), and the
    # ranking takes every layer's first expert before any layer's second.
    counts, ranked = ei.calibrated_expert_order()
    layers, experts, _ = ei.CALIBRATED_SHAPE
    assert counts.size == ranked.size == layers * experts
    assert (np.sort(counts.reshape(layers, experts), axis=1) == np.arange(1, experts + 1)).all()
    assert (ranked[:layers] // experts).tolist() == list(range(layers)) and (counts[ranked[:layers]] == experts).all()
    assert sorted(ranked.tolist()) == list(range(layers * experts))


def test_pools_are_read_from_any_of_their_forms(tmp_path):
    forms = ({'pools': [[0, 2], [1, 3]]},
             {'pools': [{'stride': 900, 'layers': [1, 3]}, {'stride': 400, 'layers': [2, 0]}]},
             {'pool_of_layer': [0, 1, 0, 1]})
    path = tmp_path / 'pools.json'
    for spec in forms:
        path.write_text(json.dumps(spec))
        assert ei.read_pools(path, 4) == [[0, 2], [1, 3]]
    path.write_text(json.dumps({'pools': [[0, 1], [1, 2, 3]]}))
    with pytest.raises(ValueError):
        ei.read_pools(path, 4)


def test_engine_log_parser_reads_the_expert_cache_lines(tmp_path):
    log = tmp_path / 'engine.log'
    log.write_text('\n'.join([
        'expert cache: no saved state yet, so it starts empty and fills as requests arrive',
        'expert cache: 9000 frames; request 10.0% of 100 routed experts hit, 1 promotions, 0 landed, '
        '2 misses CPU-served; since start 10.0%',  # an earlier session's
        '[info] CPU expert team: 6 workers, 16021 experts/s and 51200 expert columns/s (measured; other processes '
        'used 3 % of the CPU); decode misses kept on the link: 1/3',
        '[info] expert cache: no saved state yet, so it starts with the 4721 highest-ranked experts of the '
        'calibrated order (12.2 GiB in 1.48 s)',
        '[debug] expert cache: 9443 frames; request 62.3% of 118272 routed experts hit, 1873 promotions, 412 '
        'landed, 20817 misses CPU-served; since start 62.3%; VRAM free 1024 MiB; RAM: process 3.1 GiB private, '
        '2.9 GiB resident, 40.2 of 95.8 GiB free',
        '[debug] expert cache: 9443 frames; request 70.1% of 50000 routed experts hit, 300 promotions, 0 landed, '
        '9000 misses CPU-served; since start 64.6%',
    ]))
    parsed = ei.read_engine_log(log)
    assert (parsed.link_divisor, parsed.link_share, parsed.seed, parsed.seeded) == (3, 0, 'calibrated', 4721)
    assert [q.routed for q in parsed.requests] == [118272, 50000]
    first = parsed.requests[0]
    assert (first.frames, first.hit_percent, first.promotions, first.landed, first.cpu_served,
            first.since_start_percent) == (9443, 62.3, 1873, 412, 20817, 62.3)
    log.write_text('CPU expert team: 6 workers; decode misses kept on the link: 85/256 (INFERNIX_TMP_LINK_SHARE; '
                   'the divisor rule\'s 1/3)\n')
    assert (ei.read_engine_log(log).link_share, ei.read_engine_log(log).link_divisor) == (85, 0)


def test_validate_compares_the_engine_arm_with_the_engine_log(tmp_path, capsys):
    trace = ht.synthetic_trace(layers=4, experts=32, top_k=4, requests=3, prompt=64, decode=48, chunk=32,
                               verify_width=3, seed=6)
    trace.header['frames'] = 40
    for i, r in enumerate(trace.rounds):
        r.frames = 40 if i < len(trace.rounds) // 2 else 32
    path = tmp_path / 'trace.bin'
    ht.write_trace(path, trace.header, trace.rounds)
    # What the engine counts as routed: every round's distinct live experts per layer, prefill included.
    routed = [sum(np.unique(r.routes[layer][r.live]).size for layer in range(4)) for r in trace.rounds]
    per_request, previous = [], None
    for r, n in zip(trace.rounds, routed):
        if r.kind == ht.PREFILL and previous != ht.PREFILL:
            per_request.append(0)
        per_request[-1] += n
        previous = r.kind
    log = tmp_path / 'engine.log'
    log.write_text('decode misses kept on the link: 1/2\n'
                   'expert cache: no saved state yet, so it starts empty and fills as requests arrive\n'
                   + ''.join(f'expert cache: 32 frames; request 50.0% of {n} routed experts hit, 10 promotions, '
                             f'5 landed, 7 misses CPU-served; since start 50.0%\n' for n in per_request))
    out = tmp_path / 'validate.json'
    assert rp.main([str(path), '--validate', '--engine-log', str(log), '--json', str(out)]) == 0
    text = capsys.readouterr().out
    assert 'since start: replay - engine' in text and 'routed experts differ in 0' in text
    report = json.loads(out.read_text())
    assert len(per_request) == 3 and report['replay']['routed'] == sum(routed) == report['engine']['routed']
    assert report['replay']['frames'] == 32  # the last rounds ran with 32 frames
    assert report['engine_setup']['link'].startswith('1/2') and report['engine']['seed'] == 'none'
    assert report['since_start_points'] == pytest.approx(100 * report['replay']['since_start'] - 50.0)


def test_headroom_takes_the_warm_start_and_the_pools(tmp_path):
    trace = ht.synthetic_trace(layers=4, experts=32, top_k=4, requests=2, prompt=64, decode=64, chunk=32, seed=9)
    trace.header['frames'] = 48
    path = tmp_path / 'trace.bin'
    ht.write_trace(path, trace.header, trace.rounds)
    counts = ht.use_counts([trace], 4 * 32, 32)
    ranked = np.argsort(-counts, kind='stable')
    state = tmp_path / 'experts.state'
    _write_state(state, 'synthetic', counts, ranked)
    pools = tmp_path / 'pools.json'
    pools.write_text(json.dumps({'pools': [[0, 1], [2, 3]]}))
    out = tmp_path / 'headroom.json'
    assert rp.main([str(path), '--headroom', '--capacities', '48', '--seed', str(state), '--pools', str(pools),
                    '--json', str(out)]) == 0
    report = json.loads(out.read_text())
    assert report['engine_setup']['pools'].startswith('2 from') and 'synthetic' in report['engine_setup']['seed']
    # 48 frames in two pools of 24: half of them seeded, 12 per pool, each from its own layers' best keys.
    engine = rp.run_engine_arm([trace], rp.Geometry(4, 32, 4), 48, 0,
                               setup={'warm_start': (counts, ranked), 'pools': [[0, 1], [2, 3]]})
    seeded = engine.sim.seeded_keys
    assert len(seeded) == 24 and sum(1 for k in seeded if k < 2 * 32) == 12
    assert seeded[:12] == [int(k) for k in ranked if k < 2 * 32][:12]
