"""Two-level replay of the Qwen3.8-Flash-Next routed-expert tiers: VRAM frames, pinned host (RAM)
slots, and the artifact on SSD read in place (memory track, docs/maintainer/qwen3_8-flash-next-design.md
section 19.3.7, measurement RM0f). Python 3.11+ with numpy.

Input: route traces written by the engine's internal route trace (ProgramOptions::route_trace; the
format is documented in src/models/qwen4_exp/program/route_trace.h), captured with
infernix_qwen4_exp_route_trace_real_test, replayed in the order given as one session.

Model of one round (one trace record), following the plan's boundary protocol:

1. Round start: promotions whose copies completed are published (a RAM copy becomes a shadow in
   exclusive mode, is freed in strict mode); the demotion list is replenished (RAM back to H_res).
   Before the first decode round after a prompt, the optional prompt-informed prefetch runs.
2. A prefill chunk first takes its landing reserve (free slots, then the lowest residents).
3. Every expert the round computes (all columns, rejected drafts included) is a VRAM hit when it
   is resident in a published frame, a RAM hit when it has a host slot, else an SSD read.
4. Boundary: the policy clock advances (decode +1 per token, verification + the longest row's
   tokens, a prefill chunk of T tokens + min(T, 16)) and live uses update the RAM ranking (decode
   weight 1 per use, prefill min(T, 16) / T). Reserve landings become residents; of the other
   landings only the last 128 (the ring) are still there, and each is admitted if a slot is free, a
   shadow can go, or it outranks the lowest resident. Then the VRAM policy runs per layer: the
   engine's LFRU (f / (age + 1) per layer call), budgeted admission (16 per layer per prefill chunk;
   decode one per layer per token while frames are free, then one every fourth token), admitting
   only experts with a host copy, and the eviction gate: a victim with a host copy is dropped from
   VRAM, one that is not demotion-worthy is dropped to SSD, a demotion-worthy one is demoted (D2H;
   its frame is held, so the newcomer is visible one round later) while the boundary's allowance
   lasts (32 per decode boundary, unlimited at prefill boundaries), and otherwise the layer's
   admissions wait. Demotion-worthy: a free or shadow slot, or a slot of the 32-slot demotion list
   (the promoted expert's own RAM slot refills it when its copy is published, so an exclusive swap
   needs no other slot), or a score >= 1.25 x the lowest resident's.

RAM policies: decayed LFU with half-life h tokens (score sum w * 2^-(t - t_use)/h, kept as the
time-invariant key log2(s) + t/h), LRU, and CLOCK (second chance; admission and demotion
comparisons use recency). Victims are shadows first, then residents; pinned slots (landings of
the boundary, promotion copies in flight, demotions) are never victims. Modes: exclusive (shadows),
strict (no shadows: a promoted expert's RAM slot is freed), inclusive (independent RAM tier, no
demotions). Full mode (--ram-slots all) holds every expert in RAM: the VRAM part is then the
engine's current policy, and --validate replays it with the trace's recorded budgets to compare
the VRAM hit rate with the engine's own report.

Approximations: a promotion is visible from the next round (two with a held frame); copy, read
and prefetch timing are not modelled; landings are read in layer order, ascending expert ids.

Usage (from the repository root):

    py -3.13 tools/expert_cache_replay/host_tier.py TRACE [TRACE ...] [--vram 9394,8684,3000]
        [--ram-mib 43000,29000,13000] [--policies lfu:8,...,lru,clock] [--modes exclusive]
        [--priors spread] [--jobs 8] [--json OUT]
    py -3.13 tools/expert_cache_replay/host_tier.py TRACE --validate
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import struct
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

RECORD_BYTES = 2_764_800
RAM_EXTRA_SLOTS = 176  # 128 ring + 16 prefetch + 32 demotion slots outside the resident pool
RING_SLOTS = 128
DEMOTION_LIST = 32
PREFILL_PROMOTIONS = 16
DECODE_PROMOTIONS = 1
PROMOTION_INTERVAL = 4
DECODE_DEMOTION_ALLOWANCE = 32
RESERVE_MAX = 1024
PREFILL_CLOCK_MAX = 16
DEMOTION_FACTOR = 1.25
PREFETCH_FACTOR = 1.25
PREFETCH_MAX = 512
UNKNOWN_POSITION = 0xFFFFFFFF

PREFILL, FORCED, DECODE, VERIFY, GRAPH = 0, 1, 2, 3, 4
KIND_NAMES = {PREFILL: 'prefill', FORCED: 'forced', DECODE: 'decode', VERIFY: 'verify'}
RESIDENT, SHADOW = 1, 2

HEADER_FIELDS = ('version', 'layers', 'experts', 'top_k', 'frames', 'max_columns', 'lanes', 'max_width',
                 'mtp_draft_tokens', 'ngram_draft_tokens', 'prefill_chunk')
DEFAULT_POLICIES = 'lfu:8,lfu:16,lfu:32,lfu:64,lfu:128,lfu:256,lru,clock'


# ---------------------------------------------------------------------------------------------- trace

@dataclass
class Round:
    kind: int
    rows: int
    width: int
    tokens: int
    budget: int
    position: int
    lanes: np.ndarray   # int32 [rows]
    live: np.ndarray    # bool [columns]
    routes: np.ndarray  # int32 [layers, columns, top_k]


@dataclass
class Trace:
    header: dict
    rounds: list = field(default_factory=list)
    graphs: list = field(default_factory=list)


def read_trace(path):
    """Parses a route trace (route_trace.h). Graph records go to Trace.graphs."""
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < 64:
        raise ValueError(f'{path}: too short for a route trace header')
    magic, *words = struct.unpack_from('<8s14I', data, 0)
    if magic != b'NRTRACE1' or words[0] != 1:
        raise ValueError(f'{path}: not a version 1 route trace')
    header = dict(zip(HEADER_FIELDS, words[:len(HEADER_FIELDS)]))
    layers, top_k = header['layers'], header['top_k']
    trace = Trace(header)
    off = 64
    while off < len(data):
        if off + 32 > len(data):
            raise ValueError(f'{path}: truncated record at byte {off}')
        head = struct.unpack_from('<8I', data, off)
        off += 32
        kind = head[0]
        if kind == GRAPH:
            trace.graphs.append({'family': head[1], 'batch': head[2], 'width': head[3],
                                 'free_before': head[4] | head[5] << 32, 'free_after': head[6] | head[7] << 32})
            continue
        if kind > VERIFY:
            raise ValueError(f'{path}: unknown record kind {kind}')
        columns, rows, width, tokens, budget, position = head[1:7]
        if columns == 0 or rows * width != columns:
            raise ValueError(f'{path}: malformed round at byte {off - 32}')
        need = 4 * rows + (columns + 3) // 4 * 4 + 4 * layers * top_k * columns
        if off + need > len(data):
            raise ValueError(f'{path}: truncated round at byte {off - 32}')
        lanes = np.frombuffer(data, np.int32, rows, off).copy()
        off += 4 * rows
        live = np.frombuffer(data, np.uint8, columns, off) != 0
        off += (columns + 3) // 4 * 4
        routes = np.frombuffer(data, np.int32, layers * top_k * columns, off).reshape(layers, columns, top_k)
        off += 4 * layers * top_k * columns
        trace.rounds.append(Round(kind, rows, width, tokens, budget, position, lanes, live, routes))
    return trace


def write_trace(path, header, rounds):
    """Writes rounds in the engine's route-trace format (for synthetic traces and tests)."""
    words = [header.get(name, 0) for name in HEADER_FIELDS]
    words[0] = 1
    with open(path, 'wb') as f:
        f.write(struct.pack('<8s14I', b'NRTRACE1', *words, 0, 0, 0))
        for r in rounds:
            columns = r.rows * r.width
            f.write(struct.pack('<8I', r.kind, columns, r.rows, r.width, r.tokens, r.budget, r.position, 0))
            f.write(np.asarray(r.lanes, np.int32).tobytes())
            live = np.zeros((columns + 3) // 4 * 4, np.uint8)
            live[:columns] = np.asarray(r.live, bool)
            f.write(live.tobytes())
            f.write(np.ascontiguousarray(r.routes, np.int32).tobytes())


def synthetic_trace(layers=8, experts=64, top_k=4, requests=6, prompt=160, decode=96, chunk=64, topics=3,
                    verify_width=0, seed=0):
    """A routing trace with per-layer Zipf popularity that shifts between topics: `requests` prompts of
    `prompt` tokens in `chunk`-token prefill chunks, each followed by `decode` tokens (plain decode, or
    verification rounds of `verify_width` columns accepting a random prefix)."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / np.arange(1, experts + 1) ** 1.1
    perms = [[rng.permutation(experts) for _ in range(layers)] for _ in range(topics)]

    def route(topic, columns):
        logp = np.empty((layers, experts))
        for layer in range(layers):
            logp[layer, perms[topic][layer]] = np.log(weights)
        noisy = logp[:, None, :] + rng.gumbel(size=(layers, columns, experts))
        return np.argsort(-noisy, axis=2)[:, :, :top_k].astype(np.int32)

    rounds = []
    for q in range(requests):
        topic = q % topics
        for pos in range(0, prompt, chunk):
            width = min(chunk, prompt - pos)
            rounds.append(Round(PREFILL, 1, width, width, PREFILL_PROMOTIONS, pos, np.zeros(1, np.int32),
                                np.ones(width, bool), route(topic, width)))
        made = 1
        while made < decode:
            if verify_width > 1:
                accepted = int(rng.integers(0, verify_width))
                live = np.arange(verify_width) <= accepted
                rounds.append(Round(VERIFY, 1, verify_width, accepted + 1, 1, UNKNOWN_POSITION,
                                    np.zeros(1, np.int32), live, route(topic, verify_width)))
                made += accepted + 1
            else:
                rounds.append(Round(DECODE, 1, 1, 1, 1, UNKNOWN_POSITION, np.zeros(1, np.int32),
                                    np.ones(1, bool), route(topic, 1)))
                made += 1
    header = {'layers': layers, 'experts': experts, 'top_k': top_k, 'frames': 0, 'lanes': 1,
              'max_columns': max(chunk, verify_width, 1), 'max_width': max(verify_width, 1), 'prefill_chunk': chunk}
    return Trace(header, rounds)


def use_counts(traces, num_keys, experts):
    """Live routed uses per key over the traces (the oracle and profile priors' ranking)."""
    counts = np.zeros(num_keys, np.int64)
    for trace in traces:
        for r in trace.rounds:
            routes = r.routes[:, r.live, :]
            keys = routes.reshape(routes.shape[0], -1) + (np.arange(routes.shape[0]) * experts)[:, None]
            np.add.at(counts, keys.ravel(), 1)
    return counts


# ---------------------------------------------------------------------------------------------- model

class Metrics:
    COUNTERS = ('rounds', 'tokens', 'prompt_tokens', 'computed', 'vram_hits', 'ram_hits', 'ssd_reads',
                'layers_with_reads', 'live_keys', 'live_vram_hits')

    def __init__(self):
        self.by_kind = {kind: dict.fromkeys(self.COUNTERS, 0) for kind in KIND_NAMES}
        self.round_reads = []  # SSD reads of each decode or verification round
        self.events = dict.fromkeys(('promotions', 'demotions', 'overflow_demotions', 'deferred_admissions',
                                     'vram_drops', 'landings_admitted', 'landings_rejected', 'landings_overwritten',
                                     'ram_evictions',
                                     'reserve_slots', 'reserve_landings', 'prefetch_reads'), 0)

    def summary(self):
        k = self.by_kind
        gen_tokens = sum(k[x]['tokens'] for x in (DECODE, VERIFY, FORCED))
        gen_reads = sum(k[x]['ssd_reads'] for x in (DECODE, VERIFY, FORCED))
        prompt = k[PREFILL]['prompt_tokens']
        live = sum(v['live_keys'] for v in k.values())
        live_hits = sum(v['live_vram_hits'] for v in k.values())
        computed = sum(v['computed'] for v in k.values())
        reads = sorted(self.round_reads)
        out = {
            'generated_tokens': gen_tokens,
            'prompt_tokens': prompt,
            'ssd_reads': sum(v['ssd_reads'] for v in k.values()),
            'decode_ssd_reads': gen_reads,
            'prefill_ssd_reads': k[PREFILL]['ssd_reads'],
            'reads_per_token': gen_reads / gen_tokens if gen_tokens else 0.0,
            'layers_with_reads_per_token': (sum(k[x]['layers_with_reads'] for x in (DECODE, VERIFY, FORCED))
                                            / gen_tokens if gen_tokens else 0.0),
            'prefill_reads_per_1k_tokens': 1000.0 * k[PREFILL]['ssd_reads'] / prompt if prompt else 0.0,
            'round_reads_p50': reads[len(reads) // 2] if reads else 0,
            'round_reads_p90': reads[min(len(reads) - 1, int(0.9 * len(reads)))] if reads else 0,
            'round_reads_max': reads[-1] if reads else 0,
            'vram_hit_live': live_hits / live if live else 0.0,
            'vram_share': sum(v['vram_hits'] for v in k.values()) / computed if computed else 0.0,
            'ram_share': sum(v['ram_hits'] for v in k.values()) / computed if computed else 0.0,
        }
        for name, value in self.events.items():
            out[name] = value
        out['demotions_per_token'] = self.events['demotions'] / gen_tokens if gen_tokens else 0.0
        return out


class TierSim:
    """One configuration of the VRAM / RAM / SSD hierarchy over a key space of layers x experts."""

    def __init__(self, header, *, vram, ram_slots, policy='lfu:64', mode='exclusive', prior='spread',
                 prior_counts=None, demotion_allowance=DECODE_DEMOTION_ALLOWANCE, reserve=True,
                 prefetch_after_prefill=False, vram_budget='rule', seed=1):
        self.L, self.E, self.K = header['layers'], header['experts'], header['top_k']
        self.N = self.L * self.E
        self.V = max(0, min(int(vram), self.N - 1))
        self.full = ram_slots is None
        self.H = 0 if self.full else int(ram_slots)
        self.H_res = 0 if self.full else max(0, self.H - RAM_EXTRA_SLOTS)
        if mode not in ('exclusive', 'strict', 'inclusive'):
            raise ValueError(f'unknown mode {mode}')
        self.mode = mode
        if policy.startswith('lfu:'):
            self.policy, self.half_life = 'lfu', float(policy[4:])
        elif policy in ('lru', 'clock'):
            self.policy, self.half_life = policy, 0.0
        else:
            raise ValueError(f'unknown policy {policy}')
        if vram_budget not in ('rule', 'recorded'):
            raise ValueError(f'unknown VRAM budget {vram_budget}')
        self.vram_budget = vram_budget
        self.allowance_per_decode = demotion_allowance
        self.reserve_on = reserve and not self.full
        self.prefetch_on = prefetch_after_prefill and not self.full
        self.rng = np.random.default_rng(seed)

        n = self.N
        # VRAM: the engine's LfruPolicy, a dense resident list and a per-key position.
        self.v_count = np.zeros(n, np.int64)
        self.v_last = np.zeros(n, np.int64)
        self.v_now = 0
        self.v_pos = np.full(n, -1, np.int64)
        self.v_keys = np.zeros(max(self.V, 1), np.int64)
        self.v_n = 0
        self.visible = np.zeros(n, np.int64)  # first round a resident key is read from its frame
        self.mark = np.zeros(n, bool)
        self.promotions = 0
        self.budget_tokens = 0
        # RAM: state, ranking key, lazy heaps per class (or CLOCK queues), pins.
        self.r_state = np.zeros(n, np.int8)
        self.rank = np.full(n, -np.inf)
        self.stamp = np.zeros(n, np.int64)
        self.heaps = {RESIDENT: [], SHADOW: []}
        self.clock = {RESIDENT: OrderedDict(), SHADOW: OrderedDict()}
        self.ref = np.zeros(n, bool)
        self.pin = np.zeros(n, np.int64)  # a RAM copy is pinned while the current round < pin
        self.protect = set()              # pinned at this boundary (landings, demotions)
        self.n_ram = 0
        self.pending = {}                 # promoted key -> round its copy is published
        self.allowance = 0
        self.reserve_slots = 0
        self.reserve_landed = []
        self.chunk_reads = {}             # lane -> SSD reads of its last prefill chunk
        self.prefetch_due = set()         # lanes whose prompt just finished
        self.t = 0.0
        self.round = 0
        self.lane_request = {}
        self.requests = 0
        self.metrics = [Metrics(), Metrics()]
        self.current_half = 0
        self.half_of_request = lambda q: 0
        if self.full:
            self.r_state[:] = RESIDENT
        self._apply_prior(prior, prior_counts)

    # ------------------------------------------------------------------ priors
    def _apply_prior(self, prior, counts):
        if prior == 'none':
            return
        if prior == 'spread':
            order = (np.arange(self.E)[:, None] + np.arange(self.L)[None, :] * self.E).ravel()
        elif prior == 'random':
            order = self.rng.permutation(self.N)
        elif prior in ('oracle', 'profile'):
            if counts is None:
                raise ValueError(f'prior {prior} needs counts')
            order = np.lexsort((np.arange(self.N), -np.asarray(counts)))
        else:
            raise ValueError(f'unknown prior {prior}')
        order = [int(k) for k in order]
        for k in order[:self.V]:
            self._v_insert(k)
        if self.full:
            return
        ram = order[:self.H_res] if self.mode == 'inclusive' else order[self.V:self.V + self.H_res]
        # Prior ranks sit far below any real use, best first: LFU -100 - i * 1e-6 (a use is >= -6 at
        # t = 0), recency -1 - i * 1e-6.
        base = -100.0 if self.policy == 'lfu' else -1.0
        for i, k in enumerate(reversed(ram)):
            self.rank[k] = base - (len(ram) - 1 - i) * 1e-6
            self._ram_add(k, RESIDENT)

    # ------------------------------------------------------------------ VRAM (engine LFRU)
    def _v_insert(self, k):
        self.v_pos[k] = self.v_n
        self.v_keys[self.v_n] = k
        self.v_n += 1

    def _v_erase(self, k):
        at = self.v_pos[k]
        tail = self.v_keys[self.v_n - 1]
        self.v_keys[at] = tail
        self.v_pos[tail] = at
        self.v_n -= 1
        self.v_pos[k] = -1

    def _vscore(self, k):
        return self.v_count[k] / (self.v_now - self.v_last[k] + 1)

    def _vram_victims(self, need):
        keys = self.v_keys[:self.v_n]
        cand = keys[~self.mark[keys]]
        if cand.size == 0:
            return []
        score = self.v_count[cand] / (self.v_now - self.v_last[cand] + 1)
        if need == 1:
            low = score.min()
            return [int(cand[score == low].min())]
        return [int(x) for x in cand[np.lexsort((cand, score))[:need]]]

    def _budget(self, r):
        if self.vram_budget == 'recorded':
            return r.budget
        if r.kind == PREFILL:
            return PREFILL_PROMOTIONS
        if r.kind == FORCED:
            return DECODE_PROMOTIONS
        filling = self.promotions < self.V if self.full else self.v_n < self.V
        if filling:
            return DECODE_PROMOTIONS * r.tokens
        self.budget_tokens += r.tokens
        due = self.budget_tokens // PROMOTION_INTERVAL
        self.budget_tokens %= PROMOTION_INTERVAL
        return due * DECODE_PROMOTIONS

    def _vram_step(self, group, budget):
        self.v_now += 1
        if group.size == 0:
            return
        self.v_count[group] += 1
        self.v_last[group] = self.v_now
        misses = group[self.v_pos[group] < 0]
        if misses.size == 0 or self.V == 0:
            return
        self.mark[group] = True
        try:
            if self.full and budget >= misses.size:
                # The engine's unbudgeted path: every miss is admitted.
                total = self.v_n + misses.size
                for v in (self._vram_victims(total - self.V) if total > self.V else []):
                    self._vram_evict(v)
                for k in misses:
                    if self.v_n < self.V:
                        self._vram_admit(int(k), held=False)
                return
            if not self.full:
                misses = misses[self.r_state[misses] != 0]  # only experts with a host copy
            if misses.size == 0 or budget == 0:
                return
            score = self.v_count[misses] / (self.v_now - self.v_last[misses] + 1)
            admitted = 0
            for i in np.lexsort((misses, -score)):
                if admitted >= budget:
                    break
                k = int(misses[i])
                held = False
                if self.v_n >= self.V:
                    victims = self._vram_victims(1)
                    if not victims or not score[i] > self._vscore(victims[0]):
                        break
                    v = victims[0]
                    action = 'keep' if self.full else self._gate(v)
                    if action is None:
                        self._m().events['deferred_admissions'] += 1
                        break
                    self._vram_evict(v)
                    if action == 'demote':
                        self._demote(v)
                        held = True
                    elif action == 'drop':
                        self._m().events['vram_drops'] += 1
                self._vram_admit(k, held)
                admitted += 1
        finally:
            self.mark[group] = False

    def _gate(self, v):
        """The eviction gate (plan section 4.8): 'keep' (host copy), 'drop', 'demote' or None (wait)."""
        if self.mode == 'inclusive':
            return 'keep' if self.r_state[v] != 0 else 'drop'
        if self.r_state[v] != 0:
            return 'keep'
        if not self._demotion_worthy(v):
            return 'drop'
        if self.allowance <= 0:
            return None
        self.allowance -= 1
        return 'demote'

    def _vram_evict(self, v):
        self._v_erase(v)
        if self.full:
            return
        if v in self.pending:  # promoted, copy not yet published: the RAM copy stays a resident
            del self.pending[v]
        if self.mode == 'exclusive' and self.r_state[v] == SHADOW:
            self._ram_set_class(v, RESIDENT)

    def _vram_admit(self, k, held):
        self._v_insert(k)
        self.promotions += 1
        self._m().events['promotions'] += 1
        publish = self.round + (2 if held else 1)
        self.visible[k] = publish
        if self.full:
            return
        self.pin[k] = max(self.pin[k], publish)  # the copy reads the RAM slot until it lands
        if self.mode != 'inclusive':
            self.pending[k] = publish

    def _publish(self):
        if self.full:
            return
        for k in [k for k, at in self.pending.items() if at <= self.round]:
            del self.pending[k]
            if self.v_pos[k] < 0 or self.r_state[k] == 0:
                continue
            if self.mode == 'exclusive':
                self._ram_set_class(k, SHADOW)
            else:
                self._ram_drop(k, count=False)
        # Replenish the demotion list; slots still held by unpublished promotions come back when
        # those publish.
        while self.n_ram > self.H_res + len(self.pending):
            victim = self._ram_victim()
            if victim is None:
                break
            self._ram_drop(victim)

    # ------------------------------------------------------------------ RAM
    def _protected(self, k):
        return k in self.protect or self.pin[k] > self.round

    def _ram_add(self, k, cls):
        if self.r_state[k] == 0:
            self.n_ram += 1
        else:
            self._clock_remove(k)
        self.r_state[k] = cls
        self.stamp[k] += 1
        self._enqueue(k, cls)

    def _ram_set_class(self, k, cls):
        self._clock_remove(k)
        self.r_state[k] = cls
        self.stamp[k] += 1
        self._enqueue(k, cls)

    def _ram_drop(self, k, count=True):
        self._clock_remove(k)
        self.r_state[k] = 0
        self.stamp[k] += 1
        self.n_ram -= 1
        if count:
            self._m().events['ram_evictions'] += 1

    def _enqueue(self, k, cls):
        if self.policy == 'clock':
            self.clock[cls][k] = None
        else:
            heapq.heappush(self.heaps[cls], (float(self.rank[k]), k, int(self.stamp[k])))

    def _clock_remove(self, k):
        if self.policy == 'clock' and self.r_state[k] != 0:
            self.clock[int(self.r_state[k])].pop(k, None)

    def _peek(self, cls):
        if self.policy == 'clock':
            queue = self.clock[cls]
            for _ in range(2 * len(queue) + 1):
                if not queue:
                    return None
                k = next(iter(queue))
                if self._protected(k):
                    queue.move_to_end(k)
                    continue
                if self.ref[k]:
                    self.ref[k] = False
                    queue.move_to_end(k)
                    continue
                return k
            return None
        heap = self.heaps[cls]
        held, found = [], None
        while heap:
            value, k, stamp = heap[0]
            if stamp != self.stamp[k] or self.r_state[k] != cls:
                heapq.heappop(heap)
                continue
            current = float(self.rank[k])
            if value != current:
                heapq.heapreplace(heap, (current, k, stamp))
                continue
            if self._protected(k):
                held.append(heapq.heappop(heap))
                continue
            found = k
            break
        for entry in held:
            heapq.heappush(heap, entry)
        return found

    def _ram_victim(self):
        victim = self._peek(SHADOW)
        return victim if victim is not None else self._peek(RESIDENT)

    def _beats(self, a, b, factor):
        if self.policy == 'lfu' and factor > 1.0:
            return self.rank[a] >= self.rank[b] + math.log2(factor)
        return self.rank[a] > self.rank[b]

    def _free(self):
        return self.n_ram + self.reserve_slots < self.H_res

    def _overflow_slot(self):
        """A demotion-list slot for the victim of a swap: the promoted expert's RAM slot, which turns
        into a shadow (or is freed) when its copy is published, refills the list at the next round
        start. Only exclusive and strict modes swap."""
        return self.mode != 'inclusive' and self.n_ram - self.H_res < DEMOTION_LIST

    def _demotion_worthy(self, v):
        if self._free() or self._overflow_slot() or self._peek(SHADOW) is not None:
            return True
        low = self._peek(RESIDENT)
        return low is not None and self._beats(v, low, DEMOTION_FACTOR)

    def _demote(self, v):
        events = self._m().events
        events['demotions'] += 1
        self.protect.add(v)
        self.pin[v] = max(self.pin[v], self.round + 1)
        if not self._free():
            shadow = self._peek(SHADOW)
            if shadow is not None:
                self._ram_drop(shadow)
            elif self._overflow_slot():
                events['overflow_demotions'] += 1
            else:
                low = self._peek(RESIDENT)
                if low is not None:
                    self._ram_drop(low)
        self._ram_add(v, RESIDENT)

    def _admit_landing(self, k, factor=1.0):
        if self.r_state[k] != 0:
            return True
        if not self._free():
            shadow = self._peek(SHADOW)
            if shadow is not None:
                self._ram_drop(shadow)
            else:
                low = self._peek(RESIDENT)
                if low is None or not self._beats(k, low, factor):
                    return False
                self._ram_drop(low)
        self._ram_add(k, RESIDENT)
        self.protect.add(k)
        return True

    def _ram_use(self, keys, weights):
        if self.policy == 'lfu':
            base = self.t / self.half_life
            self.rank[keys] = base + np.log2(np.exp2(self.rank[keys] - base) + weights)
        else:
            self.rank[keys] = self.t
            if self.policy == 'clock':
                self.ref[keys] = True

    # ------------------------------------------------------------------ prefill reserve and prefetch
    def _take_reserve(self, r):
        lane = int(r.lanes[0])
        if r.position == 0:
            ssd_only = int(np.count_nonzero((self.v_pos < 0) & (self.r_state == 0)))
            estimate = min(RESERVE_MAX, ssd_only)
        else:
            estimate = self.chunk_reads.get(lane, 0)
        want = min(RESERVE_MAX, self.H_res // 8, estimate)
        taken = min(want, max(0, self.H_res - self.n_ram))
        while taken < want:
            low = self._peek(RESIDENT)
            if low is None:
                break
            self._ram_drop(low)
            taken += 1
        self.reserve_slots = taken
        self.reserve_landed = []
        self._m().events['reserve_slots'] += taken

    def _prefetch(self):
        cap = min(PREFETCH_MAX, self.H // 16)
        if cap <= 0:
            return
        ssd_only = np.flatnonzero((self.v_pos < 0) & (self.r_state == 0) & np.isfinite(self.rank))
        if ssd_only.size == 0:
            return
        best = ssd_only[np.lexsort((ssd_only, -self.rank[ssd_only]))][:cap]
        for k in best:
            if not self._admit_landing(int(k), PREFETCH_FACTOR):
                break
            self._m().events['prefetch_reads'] += 1

    # ------------------------------------------------------------------ rounds
    def _m(self):
        return self.metrics[self.current_half]

    def run(self, rounds):
        starts = sum(1 for r in rounds if r.kind == PREFILL and r.position == 0)
        self.half_of_request = lambda q: 0 if q < (starts + 1) // 2 else 1
        for r in rounds:
            self.step(r)

    def step(self, r):
        if r.kind == PREFILL and r.position == 0:
            self.lane_request[int(r.lanes[0])] = self.requests
            self.requests += 1
        self.current_half = self.half_of_request(self.lane_request.get(int(r.lanes[0]), 0))
        m = self._m().by_kind[r.kind]
        self.allowance = 1 << 60 if r.kind == PREFILL else self.allowance_per_decode
        self._publish()
        if self.prefetch_on and r.kind in (DECODE, VERIFY):
            for lane in r.lanes:
                if int(lane) in self.prefetch_due:
                    self.prefetch_due.discard(int(lane))
                    self._prefetch()
        if self.reserve_on and r.kind == PREFILL:
            self._take_reserve(r)

        # Reads: every expert the round computes, live or not.
        layer_base = (np.arange(self.L) * self.E)[:, None]
        computed = r.routes.reshape(self.L, -1) + layer_base
        landings = []
        reads = 0
        layers_with_reads = 0
        for layer in range(self.L):
            keys = np.unique(computed[layer])
            in_vram = (self.v_pos[keys] >= 0) & (self.visible[keys] <= self.round)
            m['computed'] += keys.size
            m['vram_hits'] += int(np.count_nonzero(in_vram))
            if self.full:
                m['ram_hits'] += int(keys.size - np.count_nonzero(in_vram))
                continue
            in_ram = ~in_vram & (self.r_state[keys] != 0)
            m['ram_hits'] += int(np.count_nonzero(in_ram))
            ssd = keys[~in_vram & ~in_ram]
            if ssd.size:
                layers_with_reads += 1
                reads += ssd.size
                landings.extend(int(k) for k in ssd)
        m['ssd_reads'] += reads
        m['layers_with_reads'] += layers_with_reads
        m['rounds'] += 1
        if r.kind in (DECODE, VERIFY):
            self._m().round_reads.append(reads)

        # Boundary: clock and live uses.
        live_routes = r.routes[:, r.live, :].reshape(self.L, -1) + layer_base
        if r.kind in (PREFILL, FORCED):
            columns = r.rows * r.width
            delta = min(columns, PREFILL_CLOCK_MAX)
            self.t += delta
            weight = delta / columns
            if r.kind == PREFILL:
                m['prompt_tokens'] += columns
            else:
                m['tokens'] += columns
        else:
            self.t += r.tokens
            weight = 1.0
            m['tokens'] += int(np.count_nonzero(r.live))
        groups = []
        for layer in range(self.L):
            keys, counts = np.unique(live_routes[layer], return_counts=True)
            groups.append(keys)
            m['live_keys'] += keys.size
            m['live_vram_hits'] += int(np.count_nonzero((self.v_pos[keys] >= 0) & (self.visible[keys] <= self.round)))
            if not self.full:
                self._ram_use(keys, counts * weight)

        if not self.full:
            events = self._m().events
            if r.kind == PREFILL and self.reserve_on:
                held = landings[:self.reserve_slots]
                landings = landings[self.reserve_slots:]
                self.reserve_slots = 0
                for k in held:
                    self._ram_add(k, RESIDENT)
                    self.protect.add(k)
                events['reserve_landings'] += len(held)
                self.chunk_reads[int(r.lanes[0])] = reads
            # The landing ring is reused within the round once its slots are consumed: only its last
            # RING_SLOTS landings are still there to be admitted at the boundary.
            events['landings_overwritten'] += max(0, len(landings) - RING_SLOTS)
            for k in landings[-RING_SLOTS:]:
                if self._admit_landing(k):
                    events['landings_admitted'] += 1
                else:
                    events['landings_rejected'] += 1
            if r.kind == PREFILL:
                self.prefetch_due.add(int(r.lanes[0]))

        budget = self._budget(r)
        for layer in range(self.L):
            self._vram_step(groups[layer], budget)
        self.protect.clear()
        self.round += 1

    def summary(self):
        both = Metrics()
        for half in self.metrics:
            for kind, counters in half.by_kind.items():
                for name, value in counters.items():
                    both.by_kind[kind][name] += value
            for name, value in half.events.items():
                both.events[name] += value
            both.round_reads.extend(half.round_reads)
        return {'total': both.summary(), 'first_half': self.metrics[0].summary(),
                'second_half': self.metrics[1].summary()}


# ---------------------------------------------------------------------------------------------- driver

def ram_slots_for_mib(mib):
    return int(mib * 2**20 // RECORD_BYTES)


def policy_verdict(reads):
    """Plan Q13: keep decayed LFU (its best half-life) unless LRU or CLOCK reads >= 5 % fewer."""
    lfu = {name: value for name, value in reads.items() if name.startswith('lfu:')}
    if not lfu:
        return None
    best = min(lfu, key=lambda name: (lfu[name], float(name[4:])))
    others = {name: value for name, value in reads.items() if name in ('lru', 'clock')}
    if others:
        rival = min(others, key=others.get)
        if others[rival] < lfu[best] and others[rival] <= 0.95 * lfu[best]:
            return {'policy': rival, 'deviation': True, 'reads': others[rival], 'best_lfu': best,
                    'best_lfu_reads': lfu[best]}
    return {'policy': best, 'deviation': False, 'reads': lfu[best], 'best_lfu': best, 'best_lfu_reads': lfu[best]}


_TRACES = None


def _load(paths):
    global _TRACES
    if _TRACES is None or _TRACES[0] != paths:
        _TRACES = (paths, [read_trace(p) for p in paths])
    return _TRACES[1]


def run_config(paths, config, counts=None):
    traces = _load(tuple(paths))
    header = traces[0].header
    rounds = [r for t in traces for r in t.rounds]
    sim = TierSim(header, prior_counts=counts, **config)
    sim.run(rounds)
    return {'config': config, **sim.summary()}


def _job(item):
    paths, config, counts = item
    return run_config(paths, config, counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--vram', default='9394,8684,3000', help='frame counts V')
    ap.add_argument('--ram-mib', default='43000,29000,13000', help='expert RAM caps (MiB) giving H slots')
    ap.add_argument('--ram-slots', default=None, help='H slots directly (comma list; "all" = full mode)')
    ap.add_argument('--policies', default=DEFAULT_POLICIES)
    ap.add_argument('--modes', default='exclusive', help='exclusive, strict, inclusive')
    ap.add_argument('--priors', default='spread', help='none, spread, random, oracle, profile:TRACE')
    ap.add_argument('--demotion-allowance', type=int, default=DECODE_DEMOTION_ALLOWANCE)
    ap.add_argument('--no-reserve', action='store_true', help='no prompt landing reserve')
    ap.add_argument('--prefetch-after-prefill', action='store_true')
    ap.add_argument('--vram-budget', default='rule', choices=('rule', 'recorded'))
    ap.add_argument('--t0-ms', type=float, default=11.2, help='base decode time per token (plain)')
    ap.add_argument('--cost-ms', type=float, default=0.65, help='exposed time per decode SSD read')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--json', default=None)
    ap.add_argument('--validate', action='store_true',
                    help='full mode at the trace frames with recorded budgets: VRAM hit rate vs the engine')
    args = ap.parse_args(argv)

    paths = tuple(args.traces)
    traces = _load(paths)
    header = traces[0].header
    for t in traces[1:]:
        if (t.header['layers'], t.header['experts'], t.header['top_k']) != (header['layers'], header['experts'],
                                                                            header['top_k']):
            raise SystemExit('traces disagree on layers, experts or top_k')
    rounds = sum(len(t.rounds) for t in traces)
    print(f'traces: {len(traces)}, {rounds} rounds; {header["layers"]} layers x {header["experts"]} experts, '
          f'top-{header["top_k"]}; captured with {header["frames"]} frames, {header["lanes"]} lanes, max width '
          f'{header["max_width"]}, MTP {header["mtp_draft_tokens"]}, n-gram {header["ngram_draft_tokens"]}')

    if args.validate:
        result = run_config(paths, {'vram': header['frames'], 'ram_slots': None, 'vram_budget': 'recorded',
                                    'prior': 'none'})
        s = result['total']
        print(f'full mode, {header["frames"]} frames, recorded budgets: VRAM hits {100 * s["vram_hit_live"]:.2f} % '
              f'of live routed experts (engine report: "since start" hit rate), {s["promotions"]} promotions')
        return 0

    n = header['layers'] * header['experts']
    jobs = []
    if args.ram_slots is not None:
        slots = [None if x == 'all' else int(x) for x in args.ram_slots.split(',')]
        ram = [(x, None if x is None else round(x * RECORD_BYTES / 2**20)) for x in slots]
    else:
        ram = [(ram_slots_for_mib(int(x)), int(x)) for x in args.ram_mib.split(',')]
    for prior in args.priors.split(','):
        counts = None
        if prior == 'oracle':
            counts = use_counts(traces, n, header['experts'])
        elif prior.startswith('profile:'):
            counts = use_counts([read_trace(prior[8:])], n, header['experts'])
            prior = 'profile'
        for mode in args.modes.split(','):
            for vram in (int(x) for x in args.vram.split(',')):
                for slots, mib in ram:
                    for policy in args.policies.split(','):
                        config = {'vram': vram, 'ram_slots': slots, 'policy': policy, 'mode': mode, 'prior': prior,
                                  'demotion_allowance': args.demotion_allowance, 'reserve': not args.no_reserve,
                                  'prefetch_after_prefill': args.prefetch_after_prefill,
                                  'vram_budget': args.vram_budget}
                        jobs.append((paths, config, counts, mib))
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as pool:
            results = list(pool.map(_job, [(p, c, k) for p, c, k, _ in jobs]))
    else:
        results = [run_config(p, c, k) for p, c, k, _ in jobs]
    for result, job in zip(results, jobs):
        result['ram_mib'] = job[3]

    cells = {}
    for result in results:
        c = result['config']
        cells.setdefault((c['prior'], c['mode'], c['vram'], c['ram_slots'], result['ram_mib']), []).append(result)
    verdicts = []
    for (prior, mode, vram, slots, mib), group in cells.items():
        h_res = None if slots is None else max(0, slots - RAM_EXTRA_SLOTS)
        print(f'\n=== V {vram}, RAM {"all" if slots is None else f"{mib} MiB = {slots} slots (H_res {h_res})"}, '
              f'fast capacity {"all" if slots is None else vram + h_res}, {mode}, prior {prior}')
        print(f'{"policy":9s} {"reads/tok":>9s} {"2nd half":>8s} {"1st half":>8s} {"prefill/1K":>10s} '
              f'{"SSD total":>9s} {"VRAM%":>6s} {"RAM%":>6s} {"demote/tok":>10s} {"deferred":>8s} {"drops":>6s} '
              f'{"tok/s 2nd":>9s}')
        reads = {}
        for result in group:
            s, s2, s1 = result['total'], result['second_half'], result['first_half']
            speed = 1000.0 / (args.t0_ms + s2['reads_per_token'] * args.cost_ms)
            print(f'{result["config"]["policy"]:9s} {s["reads_per_token"]:9.2f} {s2["reads_per_token"]:8.2f} '
                  f'{s1["reads_per_token"]:8.2f} {s["prefill_reads_per_1k_tokens"]:10.1f} {s["ssd_reads"]:9d} '
                  f'{100 * s["vram_share"]:6.1f} {100 * s["ram_share"]:6.1f} {s["demotions_per_token"]:10.2f} '
                  f'{s["deferred_admissions"]:8d} {s["vram_drops"]:6d} {speed:9.1f}')
            reads[result['config']['policy']] = s['ssd_reads']
        verdict = policy_verdict(reads)
        if verdict:
            verdicts.append({'prior': prior, 'mode': mode, 'vram': vram, 'ram_slots': slots, **verdict})
            print(f'Q13: {verdict["policy"]}' + (f' (deviation: {verdict["reads"]} vs {verdict["best_lfu"]} '
                                                f'{verdict["best_lfu_reads"]} SSD reads)' if verdict['deviation']
                                                else f' ({verdict["reads"]} SSD reads)'))
    # Q13 over every V and RAM size of one prior and mode: total SSD reads per policy.
    overall = []
    for prior, mode in dict.fromkeys((r['config']['prior'], r['config']['mode']) for r in results):
        totals = {}
        for result in results:
            c = result['config']
            if (c['prior'], c['mode']) == (prior, mode):
                totals[c['policy']] = totals.get(c['policy'], 0) + result['total']['ssd_reads']
        verdict = policy_verdict(totals)
        if verdict:
            overall.append({'prior': prior, 'mode': mode, 'totals': totals, **verdict})
            print(f'\nQ13 over every V and RAM size, {mode}, prior {prior} (SSD reads {totals}): {verdict["policy"]}'
                  + (' (deviation from decayed LFU)' if verdict['deviation'] else ''))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'header': header, 'results': results, 'verdicts': verdicts, 'overall': overall}, f, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
