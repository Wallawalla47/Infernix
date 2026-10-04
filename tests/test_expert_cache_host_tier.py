"""The two-level expert-tier replay (tools/expert_cache_replay/host_tier.py, memory track R0):
trace format, the decayed-LFU key and prefill clock, and the tier semantics on synthetic traces."""

from __future__ import annotations

import math
import struct

import numpy as np

from tools.expert_cache_replay import host_tier as ht


def _record(kind, rows, width, tokens, budget, position, lanes, live, routes):
    columns = rows * width
    out = struct.pack('<8I', kind, columns, rows, width, tokens, budget, position, 0)
    out += struct.pack(f'<{rows}i', *lanes)
    out += bytes(live) + bytes((-columns) % 4)
    out += np.asarray(routes, np.int32).tobytes()
    return out


def test_parser_reads_the_engine_record_layout(tmp_path):
    # route_trace.h: 64-byte header, rounds with lanes, a padded live mask and [layers][top_k * columns]
    # routes, and payload-free graph records.
    header = struct.pack('<8s14I', b'NRTRACE1', 1, 2, 8, 2, 5, 4, 2, 2, 1, 0, 3, 0, 0, 0)
    prefill = [[[0, 1], [2, 3], [4, 5]], [[6, 7], [0, 2], [1, 3]]]
    verify = [[[7, 6], [5, 4], [3, 2], [1, 0]], [[0, 7], [1, 6], [2, 5], [3, 4]]]
    data = header
    data += _record(0, 1, 3, 3, 16, 0, [1], [1, 1, 1], prefill)
    data += struct.pack('<8I', 4, 1, 2, 2, 0, 5, 0x40000000, 4)  # graph: verify, batch 2, width 2
    data += _record(3, 2, 2, 2, 1, ht.UNKNOWN_POSITION, [1, 0], [1, 1, 1, 0], verify)
    path = tmp_path / 'trace.bin'
    path.write_bytes(data)

    trace = ht.read_trace(path)
    assert trace.header['frames'] == 5 and trace.header['lanes'] == 2 and trace.header['prefill_chunk'] == 3
    assert len(trace.rounds) == 2 and len(trace.graphs) == 1
    first, second = trace.rounds
    assert first.kind == ht.PREFILL and first.position == 0 and list(first.lanes) == [1]
    assert first.routes.shape == (2, 3, 2) and first.routes[1, 2].tolist() == [1, 3]
    assert second.kind == ht.VERIFY and second.rows == 2 and second.width == 2 and second.tokens == 2
    assert second.live.tolist() == [True, True, True, False]
    assert second.routes[0, 3].tolist() == [1, 0]
    g = trace.graphs[0]
    assert (g['family'], g['batch'], g['width']) == (1, 2, 2)
    assert g['free_before'] == 5 << 32 and g['free_after'] == 0x40000000 | 4 << 32


def test_writer_round_trips(tmp_path):
    trace = ht.synthetic_trace(layers=3, experts=16, top_k=2, requests=2, prompt=20, decode=6, chunk=8,
                               verify_width=3, seed=4)
    path = tmp_path / 'synthetic.bin'
    ht.write_trace(path, trace.header, trace.rounds)
    again = ht.read_trace(path)
    assert len(again.rounds) == len(trace.rounds)
    for a, b in zip(trace.rounds, again.rounds):
        assert (a.kind, a.rows, a.width, a.tokens, a.budget, a.position) == \
               (b.kind, b.rows, b.width, b.tokens, b.budget, b.position)
        assert np.array_equal(a.live, b.live) and np.array_equal(a.routes, b.routes)


def test_decayed_lfu_key_ranks_like_the_naive_score():
    rng = np.random.default_rng(7)
    header = {'layers': 1, 'experts': 12, 'top_k': 2}
    h = 16.0
    sim = ht.TierSim(header, vram=0, ram_slots=400, policy='lfu:16', prior='none')
    uses = []  # (time, key, weight)
    for _ in range(300):
        sim.t += float(rng.integers(0, 4))
        keys = np.unique(rng.integers(0, 12, size=3))
        weights = rng.choice([1.0, 0.25, 16 / 1024], size=keys.size)
        sim._ram_use(keys, weights)
        uses.extend((sim.t, int(k), float(w)) for k, w in zip(keys, weights))
        naive = np.zeros(12)
        for t_use, k, w in uses:
            naive[k] += w * 2.0 ** (-(sim.t - t_use) / h)
        used = naive > 0
        assert np.allclose(np.exp2(sim.rank[used] - sim.t / h), naive[used], rtol=1e-9)
        order_key = np.lexsort((np.arange(12)[used], sim.rank[used]))
        order_naive = np.lexsort((np.arange(12)[used], naive[used]))
        assert np.array_equal(order_key, order_naive)


def test_a_prefill_chunk_advances_the_clock_by_at_most_sixteen_tokens():
    header = {'layers': 1, 'experts': 8, 'top_k': 1}
    sim = ht.TierSim(header, vram=0, ram_slots=400, policy='lfu:64', prior='none', reserve=False)
    routes = np.zeros((1, 1024, 1), np.int32)  # expert 0 in every column of a 1,024-token chunk
    sim.step(ht.Round(ht.PREFILL, 1, 1024, 1024, 16, 0, np.zeros(1, np.int32), np.ones(1024, bool), routes))
    assert sim.t == 16.0
    # The chunk weighs as 16 decode uses at the chunk's time.
    assert math.isclose(2.0 ** (sim.rank[0] - sim.t / 64.0), 16.0, rel_tol=1e-12)


def test_full_mode_never_reads_the_ssd():
    trace = ht.synthetic_trace(seed=1)
    sim = ht.TierSim(trace.header, vram=100, ram_slots=None, prior='none')
    sim.run(trace.rounds)
    total = sim.summary()['total']
    assert total['ssd_reads'] == 0 and total['promotions'] > 0 and total['vram_hit_live'] > 0.2


def _covering(header, rounds, allowance):
    n = header['layers'] * header['experts']
    vram = 120
    sim = ht.TierSim(header, vram=vram, ram_slots=n - vram + ht.RAM_EXTRA_SLOTS, policy='lfu:32',
                     prior='spread', demotion_allowance=allowance)
    sim.run(rounds)
    return sim.summary()['total']


def test_decode_in_tiers_that_hold_every_expert_never_reads_the_ssd():
    # V + H_res = N (plan section 4.1, "tier, no steady-state SSD"): each decode promotion swaps an
    # expert between RAM and VRAM through the demotion list, so none leaves both tiers; without
    # demotion allowance the admissions wait instead of dropping a victim (plan section 4.8, T6).
    trace = ht.synthetic_trace(requests=4, decode=160, seed=2)
    decode = [r for r in trace.rounds if r.kind == ht.DECODE]
    with_demotions = _covering(trace.header, decode, ht.DECODE_DEMOTION_ALLOWANCE)
    assert with_demotions['ssd_reads'] == 0 and with_demotions['vram_drops'] == 0
    assert with_demotions['demotions'] > 0 and with_demotions['promotions'] > 0
    without = _covering(trace.header, decode, 0)
    assert without['ssd_reads'] == 0 and without['vram_drops'] == 0 and without['demotions'] == 0
    assert without['deferred_admissions'] > 0


def test_exclusive_tiers_beat_an_inclusive_ram_tier_when_ram_is_short():
    trace = ht.synthetic_trace(layers=6, experts=96, top_k=4, requests=6, prompt=96, decode=160, chunk=48,
                               topics=3, seed=3)
    n = 6 * 96
    reads = {}
    for mode in ('exclusive', 'inclusive'):
        sim = ht.TierSim(trace.header, vram=150, ram_slots=150 + ht.RAM_EXTRA_SLOTS, policy='lfu:32', mode=mode,
                         prior='spread')
        sim.run(trace.rounds)
        reads[mode] = sim.summary()['total']['ssd_reads']
    assert n > 300 and reads['exclusive'] < reads['inclusive']


def test_the_prompt_landing_reserve_keeps_a_chunks_cold_experts_for_the_next():
    # Two chunks of one prompt route the same cold experts. Without the reserve only the last 128
    # ring landings survive to the boundary; the reserve keeps min(1024, H_res / 8) more.
    layers, experts = 4, 512
    rng = np.random.default_rng(5)
    routes = np.stack([np.stack([rng.permutation(experts)[:8] for _ in range(256)]) for _ in range(layers)])
    rounds = [ht.Round(ht.PREFILL, 1, 256, 256, 16, pos, np.zeros(1, np.int32), np.ones(256, bool),
                       routes.astype(np.int32)) for pos in (0, 256)]
    header = {'layers': layers, 'experts': experts, 'top_k': 8}
    second = {}
    for reserve in (False, True):
        sim = ht.TierSim(header, vram=64, ram_slots=1600 + ht.RAM_EXTRA_SLOTS, policy='lfu:64', prior='none',
                         reserve=reserve)
        sim.step(rounds[0])
        before = sim.metrics[0].by_kind[ht.PREFILL]['ssd_reads']
        sim.step(rounds[1])
        second[reserve] = sim.metrics[0].by_kind[ht.PREFILL]['ssd_reads'] - before
    assert second[True] + 150 < second[False]


def test_ram_slots_follow_the_record_size():
    assert ht.ram_slots_for_mib(43000) == 16308
    assert ht.ram_slots_for_mib(29000) == 10998
    assert ht.ram_slots_for_mib(13000) == 4930


def test_policy_choice_keeps_decayed_lfu_unless_another_reads_five_percent_less():
    keep = ht.policy_verdict({'lfu:64': 100, 'lfu:8': 98, 'lru': 94, 'clock': 99})
    assert keep['policy'] == 'lfu:8' and not keep['deviation']
    switch = ht.policy_verdict({'lfu:64': 100, 'lru': 95, 'clock': 97})
    assert switch['policy'] == 'lru' and switch['deviation']
    assert not ht.policy_verdict({'lfu:64': 0, 'lru': 0})['deviation']
