"""Two-level replay of the Qwen3.8-Flash-Next routed-expert tiers: VRAM frames, pinned host (RAM)
slots, and the artifact on SSD read in place (memory track, docs/maintainer/qwen3_8-flash-next-design.md
section 19.3.7, measurement RM0f). Python 3.11+ with numpy.

Input: route traces written by the engine's internal route trace (ProgramOptions::route_trace; the
format is documented in src/runtime/expert_offload/route_trace.h), captured with
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
   engine's LFRU (f / (age + 1) per layer call), budgeted admission (16 per layer per prefill chunk
   wider than 8 columns by default, although both engines now record 0; narrower calls and decode
   one per layer per token while frames are free, in full mode until the session's promotions and
   landings reach the frames, then one every fourth token), admitting
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
engine's current policy, and --validate replays it with the trace's recorded budgets and frame
counts to compare the VRAM hit rate with the engine's own report (replay.py --validate also applies
a warm start, pools and the engine log's CPU split, and compares with the log's lines).

The VRAM LFRU follows ExpertResidency in both modes: one pool per record stride (one by default;
`pools` gives a GGUF import's), each with its own clock (one tick per call of its layers), its share
of the frames (split_frames) and its count halving (every 8,192 ticks scaled by its share of the
layers; halved counts below 2^-10 become 0). In full mode also: the fill-phase landing (S4: while
frames are free, a decode or verification round reserves min(16, free / layers) frames per layer,
the call's first staged misses in job order land there, the CPU taking its share of the misses
first, and each layer adopts them before its promotions, which they use up); the warm start
(`warm_start`: counts capped at 16, at most half the frames, each pool its share, best first);
`follow_frames` (each round's recorded frame count: a shrink evicts the lowest scores); and the
budget-matched Belady policy ('belady', with a NextUse oracle and another run's budgets).

Approximations: a promotion is visible from the next round (two with a held frame); copy, read
and prefetch timing are not modelled; landings are read in layer order, ascending expert ids.
Frame landing assumes job order is ascending expert ids and the CPU split of a call's misses by the
whole divisor or share (not a GGUF import's one-column divisor by context); the SSD tier's landing
and warm start, and frame lending, are not modelled.

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
PROMPT_SEED_CREDIT = 256.0  # ExpertOffload's kPromptSeedCredit: the engine's prompt seeding is (256, 0, 0)
EARLY_TOKENS = 256  # design §17 H8: the hit rate of the first decode tokens after a prompt
UNKNOWN_POSITION = 0xFFFFFFFF
LFRU_HALVING_PERIOD = 8192  # ExpertResidency::kLfruHalvingPeriod: a pool's period is its share of the layers
COUNT_FLOOR = 1.0 / 1024.0  # LfruPolicy::kCountFloor: halved counts below it become 0
LANDING_SLOTS = 16          # ExpertResidency::kLandingSlots: fill-phase landing frames per layer and round
SERVED_CALL_COLUMNS = 8     # the Program's kServedCallColumns: a prefill call this narrow promotes at the decode rate
CPU_JOBS = 16               # ProgramOptions::cpu_expert_jobs: CPU-served misses per decode or verification call
SEED_COUNT_CAP = 16         # the warm start caps every saved count (kSeedCountCap)
SEED_SHARE = 0.5            # and loads at most this share of the frames (kSeedShare)

PREFILL, FORCED, DECODE, VERIFY, GRAPH = 0, 1, 2, 3, 4
KIND_NAMES = {PREFILL: 'prefill', FORCED: 'forced', DECODE: 'decode', VERIFY: 'verify'}
RESIDENT, SHADOW = 1, 2

HEADER_FIELDS = ('version', 'layers', 'experts', 'top_k', 'frames', 'max_columns', 'lanes', 'max_width',
                 'mtp_draft_tokens', 'ngram_draft_tokens', 'prefill_chunk')
# The engine's route-trace version (route_trace.h). Version 2 stamps each round's eighth header word
# with the round's expert-pool frame count; version 1 left it reserved (0).
TRACE_VERSION = 2
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
    frames: int = 0     # the expert pool's frame count this round ran with (0: not recorded)


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
    if magic != b'NRTRACE1' or words[0] != TRACE_VERSION:
        raise ValueError(f'{path}: not a version {TRACE_VERSION} route trace')
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
        columns, rows, width, tokens, budget, position, frames = head[1:8]
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
        trace.rounds.append(Round(kind, rows, width, tokens, budget, position, lanes, live, routes, frames))
    return trace


def write_trace(path, header, rounds):
    """Writes rounds in the engine's route-trace format (for synthetic traces and tests)."""
    words = [header.get(name, 0) for name in HEADER_FIELDS]
    words[0] = TRACE_VERSION
    with open(path, 'wb') as f:
        f.write(struct.pack('<8s14I', b'NRTRACE1', *words, 0, 0, 0))
        for r in rounds:
            columns = r.rows * r.width
            f.write(struct.pack('<8I', r.kind, columns, r.rows, r.width, r.tokens, r.budget, r.position,
                                getattr(r, 'frames', 0)))
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
    # early_*: live keys of one-row decode and verification rounds within EARLY_TOKENS of their prompt.
    COUNTERS = ('rounds', 'tokens', 'prompt_tokens', 'computed', 'vram_hits', 'ram_hits', 'ssd_reads',
                'layers_with_reads', 'live_keys', 'live_vram_hits', 'early_live_keys', 'early_live_vram_hits')

    def __init__(self):
        self.by_kind = {kind: dict.fromkeys(self.COUNTERS, 0) for kind in KIND_NAMES}
        self.round_reads = []  # SSD reads of each decode or verification round
        self.events = dict.fromkeys(('promotions', 'demotions', 'overflow_demotions', 'deferred_admissions',
                                     'vram_drops', 'landings_admitted', 'landings_rejected', 'landings_overwritten',
                                     'ram_evictions',
                                     'reserve_slots', 'reserve_landings', 'prefetch_reads',
                                     'prompt_seeds', 'prompt_seed_promotions', 'adaptive_admissions',
                                     'adaptive_swaps', 'frame_landings', 'warm_start_seeds', 'resizes'), 0)

    def summary(self):
        k = self.by_kind
        gen_tokens = sum(k[x]['tokens'] for x in (DECODE, VERIFY, FORCED))
        gen_reads = sum(k[x]['ssd_reads'] for x in (DECODE, VERIFY, FORCED))
        prompt = k[PREFILL]['prompt_tokens']
        live = sum(v['live_keys'] for v in k.values())
        live_hits = sum(v['live_vram_hits'] for v in k.values())
        decode_live = sum(k[x]['live_keys'] for x in (DECODE, VERIFY))
        decode_hits = sum(k[x]['live_vram_hits'] for x in (DECODE, VERIFY))
        early_live = sum(k[x]['early_live_keys'] for x in (DECODE, VERIFY))
        early_hits = sum(k[x]['early_live_vram_hits'] for x in (DECODE, VERIFY))
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
            'decode_vram_hit_live': decode_hits / decode_live if decode_live else 0.0,
            'early_vram_hit_live': early_hits / early_live if early_live else 0.0,
            'early_live_keys': early_live,
            'vram_share': sum(v['vram_hits'] for v in k.values()) / computed if computed else 0.0,
            'ram_share': sum(v['ram_hits'] for v in k.values()) / computed if computed else 0.0,
        }
        for name, value in self.events.items():
            out[name] = value
        out['demotions_per_token'] = self.events['demotions'] / gen_tokens if gen_tokens else 0.0
        # The link's share (candidates D12 + PUB-22, review 2026-10-24 §7.2): every VRAM admission is one
        # record over the link, a decode miss is served by the CPU or the link.
        out['decode_vram_misses_per_token'] = (decode_live - decode_hits) / gen_tokens if gen_tokens else 0.0
        out['promotions_per_token'] = self.events['promotions'] / gen_tokens if gen_tokens else 0.0
        return out


def link_charged_speed(s, *, t0_ms, cost_ms, miss_ms, link_gbs, record_bytes=None):
    """Tokens per second with SSD reads, decode VRAM misses and promotion copies all charged: t0, plus
    cost_ms per SSD read, miss_ms per decode VRAM miss, and each promotion's record over a link of
    link_gbs GB/s (a policy that buys hits with copies pays for them here)."""
    record_bytes = RECORD_BYTES if record_bytes is None else record_bytes
    promotion_ms = record_bytes / (link_gbs * 1e9) * 1e3
    ms = (t0_ms + s['reads_per_token'] * cost_ms + s['decode_vram_misses_per_token'] * miss_ms
          + s['promotions_per_token'] * promotion_ms)
    return 1000.0 / ms


class NextUse:
    """Belady's oracle over one round sequence (TierSim's 'belady' VRAM policy): for each key, the next
    decode or verification round that computes it (any column, as the round reads it); NEVER when none
    does. keys[i] and after[i]: the keys round i computes and their next use after it (None for other
    rounds); first: every key's first use."""
    NEVER = np.iinfo(np.int32).max

    def __init__(self, rounds, layers, experts):
        base = (np.arange(layers) * experts)[:, None]
        upcoming = np.full(layers * experts, self.NEVER, np.int32)
        self.keys = [None] * len(rounds)
        self.after = [None] * len(rounds)
        for i in range(len(rounds) - 1, -1, -1):
            r = rounds[i]
            if r.kind not in (DECODE, VERIFY):
                continue
            ids = r.routes.reshape(layers, -1)
            keys = np.unique((ids + base)[(ids >= 0) & (ids < experts)])
            self.keys[i] = keys
            self.after[i] = upcoming[keys]
            upcoming[keys] = i
        self.first = upcoming


class TierSim:
    """One configuration of the VRAM / RAM / SSD hierarchy over a key space of layers x experts."""

    def __init__(self, header, *, vram, ram_slots, policy='lfu:64', mode='exclusive', prior='spread',
                 prior_counts=None, demotion_allowance=DECODE_DEMOTION_ALLOWANCE, reserve=True,
                 prefetch_after_prefill=False, vram_budget='rule', prompt_seed=None, seed=1,
                 prefill_promotions=PREFILL_PROMOTIONS, promotion_interval=PROMOTION_INTERVAL,
                 vram_policy='lfru', halving=LFRU_HALVING_PERIOD, pools=None, landing=True, link_divisor=1,
                 link_share=0, cpu_jobs=CPU_JOBS, warm_start=None, follow_frames=False, budgets=None,
                 next_use=None, adaptive_groups=None):
        self.L, self.E, self.K = header['layers'], header['experts'], header['top_k']
        self.N = self.L * self.E
        self.full = ram_slots is None
        # Pools (ExpertResidency, one per record stride): each holds its layers' experts in its own frames
        # with its own LFRU, whose clock ticks once per call of its layers and whose counts halve every
        # `halving` x its share of the layers ticks (0: never).
        layer_lists = [list(range(self.L))] if pools is None else [sorted(int(x) for x in p) for p in pools]
        if any(not p for p in layer_lists) or sorted(x for p in layer_lists for x in p) != list(range(self.L)):
            raise ValueError('the pools must hold each layer exactly once')
        self.P = len(layer_lists)
        self.pool_layers = layer_lists
        self.pool_of_layer = np.zeros(self.L, np.int64)
        for p, layers in enumerate(layer_lists):
            self.pool_of_layer[layers] = p
        self.pool_of_key = np.repeat(self.pool_of_layer, self.E)
        self.pool_keys = [np.concatenate([np.arange(x * self.E, (x + 1) * self.E) for x in layers])
                          for layers in layer_lists]
        self.pool_period = [max(1, int(halving) * len(layers) // self.L) if halving else 0 for layers in layer_lists]
        self.pool_now = [0] * self.P
        self.pool_n = [0] * self.P
        self.requested = max(0, min(int(vram), self.N - 1))
        self.pool_V = self._split(self.requested)
        self.V = sum(self.pool_V)
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
        # `budgets`: one promotion budget per round, applied as given (another run's applied_budgets,
        # so a Belady arm admits under the engine arm's budgets); else the rule or the recorded ones.
        self.budgets = None if budgets is None else list(budgets)
        self.applied_budgets = []  # the budget each round's boundary applied (LFRU and Belady)
        # The VRAM policy: the engine's LFRU with its budgets ('lfru'); Belady MIN with bypass under the
        # same budgets ('belady', with a NextUse oracle: victim choice with perfect knowledge, the
        # admission throttle unchanged); or Basalt's adaptive tier ('adaptive[:DECAY,FLOOR,MARGIN,SWAPS]',
        # candidates §7.6 D12): per-key counts decayed by DECAY each round, a non-resident key a candidate
        # at FLOOR or more, swapped for its group's least-counted resident when it leads by MARGIN, at most
        # SWAPS a round by gain (4 x SWAPS after a round that ranked 2 x SWAPS candidates, back to SWAPS at
        # every request's start).
        self.belady = vram_policy == 'belady'
        self.adaptive = None if self.belady else parse_vram_policy(vram_policy)
        if self.belady and (next_use is None or not self.full):
            raise ValueError('the Belady VRAM policy needs full mode and a NextUse oracle')
        if self.adaptive and self.P > 1:
            raise ValueError("Basalt's adaptive tier has a single pool")
        self.next_use = next_use
        self.nu = next_use.first.copy() if self.belady else None  # every key's next use after this round
        self.a_count = np.zeros(self.L * self.E, np.float64)
        self.adaptive_cap = self.adaptive[3] if self.adaptive else 0
        # The adaptive tier's groups of layers whose frames are interchangeable (the engine's per-stride
        # pools); None: all layers one group.
        self.adaptive_groups = adaptive_groups
        # The 'rule' budgets: promotions per layer of a prefill chunk wider than SERVED_CALL_COLUMNS (both
        # models' engines now record 0; narrower calls take the decode rule) and decode tokens per
        # promotion once the frames are full (ExpertOffload::promotion_interval).
        self.prefill_promotions = int(prefill_promotions)
        self.promotion_interval = max(1, int(promotion_interval))
        # Fill-phase landing (full mode, the LFRU and Belady policies): a call's misses the CPU does not
        # serve are staged, the CPU taking min(cpu_jobs, M - kept) of a call's M misses, kept M / d
        # (link_divisor d) or (M x link_share) >> 8; the first staged ones land in the round's reserved
        # frames.
        self.landing = landing and self.full and not self.adaptive
        self.link_divisor, self.link_share, self.cpu_jobs = int(link_divisor), int(link_share), int(cpu_jobs)
        self.landed = 0  # adopted landings (ExpertResidency::Stats::landed)
        # The frame count each round ran with (follow_frames): this replay's frames move by the trace's
        # change from its startup count (the header's, else the first recorded one).
        self.follow_frames = follow_frames
        self.base_frames = int(header.get('frames', 0))
        self.target = self.requested
        self.allowance_per_decode = demotion_allowance
        self.reserve_on = reserve and not self.full
        self.prefetch_on = prefetch_after_prefill and not self.full
        # Prompt seeding (ExpertResidency::seed_prompt; the engine's is (256, 0, 0)): (credit, per-layer
        # budget, window).
        self.prompt_seed = prompt_seed
        self.prompt_rounds = {}  # lane -> the prefill rounds of its prompt not yet seeded
        self.since_prompt = {}   # lane -> tokens decoded since its prompt ended
        self.rng = np.random.default_rng(seed)

        n = self.N
        # VRAM: the engine's LfruPolicy per pool, one dense resident list over the pools and a per-key
        # position. Counts are binary64 as in the engine (prompt credits and halvings are fractions);
        # v_last is in the key's pool's ticks.
        self.v_count = np.zeros(n, np.float64)
        self.v_last = np.zeros(n, np.int64)
        self.v_pos = np.full(n, -1, np.int64)
        self.v_keys = np.zeros(n, np.int64)
        self.v_n = 0
        self.visible = np.zeros(n, np.int64)  # first round a resident key is read from its frame
        self.mark = np.zeros(n, bool)
        # Eviction protection (candidates §13 PUB-22, replay only): `guard(round index, round)`, when set,
        # names the keys a predictor expects next, evicted at that round's boundary only when no
        # unprotected resident is left. It returns None; keys (protected at every layer's step: the
        # oracle's round-wide guard); or a function of the layer giving that layer's keys, evaluated when
        # the layer's step first needs a victim and protecting them in that step only (LfruPolicy::step's
        # guard, RoutePredictor::predict(layer): the engine's).
        self.guard = None
        self.guard_mask = np.zeros(n, bool)
        self.layer_guard = None  # the current step's layer guard, its layer and its keys once evaluated
        self.guard_layer = 0
        self.guard_keys = None
        self.promotions = 0
        self.budget_tokens = 0
        self.seeded_keys = []      # the warm start's experts, in load order
        self.round_log = None      # a list: per round (kind, lane of row 0, live keys, live hits, promotions, landed)
        self.admission_log = None  # a list: per round and layer (round, layer, admitted, adopted landings, budget)
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
        if warm_start is not None:
            if not self.full or prior != 'none':
                raise ValueError('the warm start applies in full mode with no prior')
            self._warm_start(*warm_start)

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
        # The best keys while their pools have frames (one pool: the first V).
        in_vram = set()
        for k in order:
            if self.v_n >= self.V:
                break
            p = self.pool_of_key[k]
            if self.pool_n[p] < self.pool_V[p]:
                self._v_insert(k)
                in_vram.add(k)
        if self.full:
            return
        ram = order[:self.H_res] if self.mode == 'inclusive' else [k for k in order if k not in in_vram][:self.H_res]
        # Prior ranks sit far below any real use, best first: LFU -100 - i * 1e-6 (a use is >= -6 at
        # t = 0), recency -1 - i * 1e-6.
        base = -100.0 if self.policy == 'lfu' else -1.0
        for i, k in enumerate(reversed(ram)):
            self.rank[k] = base - (len(ram) - 1 - i) * 1e-6
            self._ram_add(k, RESIDENT)

    # ------------------------------------------------------------------ VRAM (engine LFRU)
    def _split(self, frames):
        """ExpertResidency::split_frames: one pool takes every frame; several take frames / layers per layer."""
        if self.P == 1:
            return [min(frames, self.N - 1)]
        per_layer = frames // self.L
        return [min(per_layer * len(layers), len(layers) * self.E - 1) for layers in self.pool_layers]

    def _v_insert(self, k):
        self.v_pos[k] = self.v_n
        self.v_keys[self.v_n] = k
        self.v_n += 1
        self.pool_n[self.pool_of_key[k]] += 1

    def _v_erase(self, k):
        at = self.v_pos[k]
        tail = self.v_keys[self.v_n - 1]
        self.v_keys[at] = tail
        self.v_pos[tail] = at
        self.v_n -= 1
        self.v_pos[k] = -1
        self.pool_n[self.pool_of_key[k]] -= 1

    def vram_visible(self, keys):
        """Which keys this round reads from a published VRAM frame (resident, copy landed)."""
        return (self.v_pos[keys] >= 0) & (self.visible[keys] <= self.round)

    def _vscore(self, k):
        return self.v_count[k] / (self.pool_now[self.pool_of_key[k]] - self.v_last[k] + 1)

    def _vram_victims(self, need, p):
        """Pool p's next `need` victims outside the current group (mark): the lowest (score, key), guarded
        keys last (LfruPolicy::select_victims); the Belady policy's farthest next use, then the lower key."""
        keys = self.v_keys[:self.v_n]
        if self.P > 1:
            keys = keys[self.pool_of_key[keys] == p]
        cand = keys[~self.mark[keys]]
        if cand.size == 0:
            return []
        if self.belady:
            far = self.nu[cand].astype(np.int64)
            if need == 1:
                return [int(cand[far == far.max()].min())]
            return [int(x) for x in cand[np.lexsort((cand, -far))[:need]]]
        if self.layer_guard is not None and self.guard_keys is None:
            self.guard_keys = np.asarray(self.layer_guard(self.guard_layer), np.int64)
            self.guard_mask[self.guard_keys] = True
        score = self.v_count[cand] / (self.pool_now[p] - self.v_last[cand] + 1)
        guarded = self.guard_mask[cand]
        if need == 1:
            eligible = ~guarded if not guarded.all() else np.ones_like(guarded)
            low = score[eligible].min()
            return [int(cand[eligible & (score == low)].min())]
        return [int(x) for x in cand[np.lexsort((cand, score, guarded))[:need]]]

    def _budget(self, r):
        if self.budgets is not None:
            return int(self.budgets[self.round])
        if self.vram_budget == 'recorded':
            return r.budget
        if r.kind == FORCED:
            return DECODE_PROMOTIONS
        if r.kind == PREFILL and r.rows * r.width > SERVED_CALL_COLUMNS:
            return self.prefill_promotions
        # Decode, verification and CPU-served prefill calls (the Program's decode_budget): one per layer
        # and token while this session's promotions and landings are fewer than the frames (warm-start
        # seeds do not count), then one every promotion_interval tokens.
        filling = self.promotions + self.landed < self.V if self.full else self.v_n < self.V
        if filling:
            return DECODE_PROMOTIONS * r.tokens
        self.budget_tokens += r.tokens
        due = self.budget_tokens // self.promotion_interval
        self.budget_tokens %= self.promotion_interval
        return due * DECODE_PROMOTIONS

    def _tick(self, p):
        """One call of pool p's layers (LfruPolicy::step): the clock advances, and every count of the pool
        halves on its period's ticks."""
        now = self.pool_now[p] + 1
        self.pool_now[p] = now
        period = self.pool_period[p]
        if period and now % period == 0:
            keys = self.pool_keys[p]
            half = self.v_count[keys] * 0.5
            half[half < COUNT_FLOOR] = 0.0
            self.v_count[keys] = half
        return now

    def _vram_step(self, group, budget, layer):
        """One layer's routed group at the boundary (CacheController::on_route); returns the admissions."""
        p = self.pool_of_layer[layer]
        now = self._tick(p)
        if group.size == 0:
            return 0
        self.v_count[group] += 1
        self.v_last[group] = now
        capacity = self.pool_V[p]
        misses = group[self.v_pos[group] < 0]
        if misses.size == 0 or capacity == 0:
            return 0
        self.mark[group] = True
        try:
            if self.belady:
                return self._belady_admit(misses, budget, p)
            if self.full and budget >= misses.size:
                # The engine's unbudgeted path: every miss is admitted.
                total = self.pool_n[p] + misses.size
                for v in (self._vram_victims(total - capacity, p) if total > capacity else []):
                    self._vram_evict(v)
                admitted = 0
                for k in misses:
                    if self.pool_n[p] < capacity:
                        self._vram_admit(int(k), held=False)
                        admitted += 1
                return admitted
            if not self.full:
                misses = misses[self.r_state[misses] != 0]  # only experts with a host copy
            if misses.size == 0 or budget == 0:
                return 0
            score = self.v_count[misses] / (now - self.v_last[misses] + 1)
            admitted = 0
            for i in np.lexsort((misses, -score)):
                if admitted >= budget:
                    break
                k = int(misses[i])
                held = False
                if self.pool_n[p] >= capacity:
                    victims = self._vram_victims(1, p)
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
            return admitted
        finally:
            self.mark[group] = False

    def _belady_admit(self, misses, budget, p):
        """Belady MIN with bypass within the step's budget: the misses with the nearest next use first, each
        into a free frame, or over the resident whose next use is the farthest while its own is nearer; a
        miss never used again is served without admission."""
        nu = self.nu[misses].astype(np.int64)
        admitted = 0
        for i in np.lexsort((misses, nu)):
            if admitted >= budget or nu[i] >= NextUse.NEVER:
                break
            if self.pool_n[p] >= self.pool_V[p]:
                victims = self._vram_victims(1, p)
                if not victims or not nu[i] < self.nu[victims[0]]:
                    break
                self._vram_evict(victims[0])
            self._vram_admit(int(misses[i]), held=False)
            admitted += 1
        return admitted

    def _landing_slots(self, r):
        """ExpertResidency::before_round(landing): a decode or verification round reserves, for every layer,
        min(LANDING_SLOTS, free frames / layers) frames of its pool (the scarcest pool's count for all)."""
        if not self.landing or r.kind not in (DECODE, VERIFY) or self.V == 0:
            return 0
        slots = LANDING_SLOTS
        for p in range(self.P):
            slots = min(slots, (self.pool_V[p] - self.pool_n[p]) // len(self.pool_layers[p]))
        return max(0, slots)

    def _staged(self, missed, columns):
        """The misses of one call the GPU stages, in job order (ascending experts): the CPU's plan takes
        min(cpu_jobs, M - kept) of the M misses, the fewest-column ones first and in job order within a
        width, and keeps the rest on the link (cpu_plan_body)."""
        m = missed.size
        if self.link_share > 0:
            kept = (m * self.link_share) >> 8
        else:
            kept = m // self.link_divisor if self.link_divisor > 0 else 0
        cpu = min(self.cpu_jobs, m - kept)
        if cpu <= 0:
            return missed
        staged = np.ones(m, bool)
        staged[np.lexsort((missed, columns))[:cpu]] = False
        return missed[staged]

    def _adopt(self, layer, landed, group):
        """ExpertResidency::after_round's landing: each expert that landed in one of the layer's reserved
        frames becomes resident there (CacheController::adopt: published next round; a full pool first
        evicts its lowest-score resident outside the layer's group), unless it is already resident or
        loading. Returns how many were adopted; they use up the layer's promotion budget."""
        if landed is None or landed.size == 0:
            return 0
        p = self.pool_of_layer[layer]
        adopted = 0
        self.mark[group] = True
        try:
            for k in (int(x) for x in landed):
                if self.v_pos[k] >= 0:
                    continue
                if self.pool_n[p] >= self.pool_V[p]:
                    victims = self._vram_victims(1, p)
                    if not victims:
                        continue
                    self._vram_evict(victims[0])
                self._v_insert(k)
                self.visible[k] = self.round + 1
                adopted += 1
        finally:
            self.mark[group] = False
        self.landed += adopted
        self._m().events['frame_landings'] += adopted
        return adopted

    def _belady_advance(self):
        """After a round's reads: the next use of each key it computed moves past it."""
        keys = self.next_use.keys[self.round]
        if keys is not None:
            self.nu[keys] = self.next_use.after[self.round]

    def _warm_start(self, counts, ranked):
        """ExpertResidency::warm_start as the Program calls it: every key's count from the state, capped at
        SEED_COUNT_CAP (last use 0, every clock 0), then at most SEED_SHARE of the frames loaded, each pool
        its share of them, its keys in the state's ranking (CacheController::seed). Published before the
        first round, counted as neither promotions nor landings."""
        counts = np.asarray(counts, np.int64)
        if counts.size != self.N:
            raise ValueError(f'the warm start has {counts.size} counts for {self.N} experts')
        self.v_count[:] = np.minimum(counts, SEED_COUNT_CAP)
        if self.V == 0:
            return
        max_keys = int(SEED_SHARE * self.V)
        ranked = np.asarray(ranked, np.int64)
        ranked = ranked[(ranked >= 0) & (ranked < self.N)]
        for p in range(self.P):
            share = max_keys * self.pool_V[p] // self.V
            for k in (int(x) for x in ranked[self.pool_of_key[ranked] == p][:share]):
                if self.v_pos[k] >= 0 or self.pool_n[p] >= self.pool_V[p]:
                    continue
                self._v_insert(k)
                self.seeded_keys.append(k)
        self._m().events['warm_start_seeds'] += len(self.seeded_keys)

    def _follow(self, r):
        """The frame count the round ran with: the pool resized between rounds (apply_vram_target)."""
        if not r.frames:
            return
        if not self.base_frames:
            self.base_frames = r.frames
        target = max(0, min(self.requested + r.frames - self.base_frames, self.N - 1))
        if target != self.target:
            self.target = target
            self._resize(target)

    def _resize(self, frames):
        """ExpertResidency::resize: each pool takes its split of `frames`; a shrink evicts the pool's
        lowest-score residents wherever their frames are (LfruPolicy::set_capacity: no group or guard is
        spared; Belady: the farthest next uses), and a grow only adds room."""
        targets = self._split(frames)
        for p, target in enumerate(targets):
            excess = self.pool_n[p] - target
            if excess > 0:
                for v in self._vram_victims(excess, p):
                    if not self.full and self.r_state[v] == 0:
                        self._m().events['vram_drops'] += 1  # resizing drops, never demotes
                    self._vram_evict(v)
            self.pool_V[p] = target
        self.V = sum(targets)
        self._m().events['resizes'] += 1

    def _adaptive_step(self, live_counts, r):
        """Basalt's adaptive tier at a round boundary (generate.cpp rank_swaps / xq_round, b80e026,
        bde2328): decay, count the round's live routes, then free frames to the best candidates and
        swaps by gain, under the round's cap. Admission and eviction take the same paths as LFRU's
        (the eviction gate, demotions, publication a round later)."""
        decay, floor, margin, base = self.adaptive
        if r.kind == PREFILL and r.position == 0:
            self.adaptive_cap = base
        self.a_count *= decay
        for keys, counts in live_counts:
            self.a_count[keys] += counts
        if self.V == 0:
            return
        events = self._m().events
        candidates, pairs = [], []
        # Pairs across the layers of each group, as the engine pairs across each frame pool's layers (Basalt
        # 557d442, candidates BG-01: a layer that routes more takes frames from one that routes fewer).
        groups = self.adaptive_groups or [list(range(self.L))]
        key_sets = [np.concatenate([np.arange(l * self.E, (l + 1) * self.E) for l in group]) for group in groups]
        for keys in key_sets:
            counts = self.a_count[keys]
            resident = self.v_pos[keys] >= 0
            mask = ~resident & (counts >= floor)
            if not self.full:
                mask &= self.r_state[keys] != 0  # only experts with a host copy
            cand = keys[mask]
            if cand.size == 0:
                continue
            cand = cand[np.lexsort((cand, -self.a_count[cand]))]
            candidates.extend(int(k) for k in cand)
            victims = keys[resident]
            victims = victims[np.lexsort((victims, self.a_count[victims]))]
            for k, v in zip(cand, victims):
                gain = self.a_count[k] - self.a_count[v]
                if gain < margin:
                    break
                pairs.append((-gain, int(k), int(v)))
        cap = self.adaptive_cap
        self.adaptive_cap = 4 * base if len(candidates) >= 2 * base else base
        done = set()
        # Free frames first, the best candidates of any layer.
        free = self.V - self.v_n
        if free > 0:
            for k in sorted(candidates, key=lambda key: (-self.a_count[key], key))[:min(free, cap)]:
                self._vram_admit(k, held=False)
                done.add(k)
                events['adaptive_admissions'] += 1
        for _, k, v in sorted(pairs):
            if len(done) >= cap:
                break
            if k in done or self.v_pos[v] < 0 or self.v_pos[k] >= 0:
                continue
            action = 'keep' if self.full else self._gate(v)
            if action is None:
                events['deferred_admissions'] += 1
                break
            self._vram_evict(v)
            held = False
            if action == 'demote':
                self._demote(v)
                held = True
            elif action == 'drop':
                events['vram_drops'] += 1
            self._vram_admit(k, held)
            done.add(k)
            events['adaptive_swaps'] += 1

    def _seed_prompt(self, rounds):
        """ExpertResidency::seed_prompt at a prompt's end: each key gains credit x (its counted tokens /
        the counted tokens), used now (no clock tick). The engine seeds credit 256 over every prompt
        token with no admissions (design §19.3.27); `budget` (full mode only) and `window` are this
        replay's alternatives: up to `budget` of each layer's credited misses admitted where they
        outrank the lowest resident outside the layer's credited keys, and only the prompt's last
        `window` tokens counted."""
        credit, budget, window = self.prompt_seed
        known = [r for r in rounds if r.position != UNKNOWN_POSITION]
        end = max((r.position + r.rows * r.width for r in known), default=None)
        uses = np.zeros(self.N, np.int64)
        counted = 0
        layer_base = (np.arange(self.L) * self.E)[:, None]
        for r in rounds:
            columns = r.rows * r.width
            start = 0
            if window and end is not None and r.position != UNKNOWN_POSITION:
                start = min(max(end - window - r.position, 0), columns)
            if start >= columns:
                continue
            keys = (r.routes[:, start:, :].reshape(self.L, -1) + layer_base).ravel()
            uses += np.bincount(keys, minlength=self.N)
            counted += columns - start
        if counted == 0:
            return
        events = self._m().events
        events['prompt_seeds'] += 1
        credited = np.nonzero(uses)[0]
        scale = credit / counted  # the engine's per-token credit, then scale x uses (the same roundings)
        self.v_count[credited] += scale * uses[credited]
        self.v_last[credited] = np.asarray(self.pool_now, np.int64)[self.pool_of_key[credited]]
        if not self.full or budget == 0 or self.V == 0:
            return
        for layer in range(self.L):
            p = self.pool_of_layer[layer]
            group = credited[(credited >= layer * self.E) & (credited < (layer + 1) * self.E)]
            misses = group[self.v_pos[group] < 0]
            if misses.size == 0:
                continue
            self.mark[group] = True
            try:
                score = self.v_count[misses] / (self.pool_now[p] - self.v_last[misses] + 1)
                admitted = 0
                for i in np.lexsort((misses, -score)):
                    if admitted >= budget:
                        break
                    if self.pool_n[p] >= self.pool_V[p]:
                        victims = self._vram_victims(1, p)
                        if not victims or not score[i] > self._vscore(victims[0]):
                            break
                        self._vram_evict(victims[0])
                    self._vram_admit(int(misses[i]), held=False)
                    events['prompt_seed_promotions'] += 1
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
        promotions_before, landed_before = self.promotions, self.landed
        self.allowance = 1 << 60 if r.kind == PREFILL else self.allowance_per_decode
        self._publish()
        # A lane's prefill rounds form its prompt until its first decode or verification round, before
        # which the prompt seeds (the engine seeds right after the prompt's last chunk, before that
        # boundary's resize).
        if r.kind == PREFILL:
            self.prompt_rounds.setdefault(int(r.lanes[0]), []).append(r)
        elif r.kind in (DECODE, VERIFY):
            for lane in (int(x) for x in r.lanes):
                rounds = self.prompt_rounds.pop(lane, None)
                if rounds is not None:
                    self.since_prompt[lane] = 0
                    if self.prompt_seed is not None:
                        self._seed_prompt(rounds)
        if self.follow_frames:
            self._follow(r)
        early = (r.kind in (DECODE, VERIFY) and r.rows == 1
                 and self.since_prompt.get(int(r.lanes[0]), EARLY_TOKENS) < EARLY_TOKENS)
        if self.prefetch_on and r.kind in (DECODE, VERIFY):
            for lane in r.lanes:
                if int(lane) in self.prefetch_due:
                    self.prefetch_due.discard(int(lane))
                    self._prefetch()
        if self.reserve_on and r.kind == PREFILL:
            self._take_reserve(r)
        # Fill-phase landing frames per layer (full mode): each layer's call lands its first staged misses.
        slots = self._landing_slots(r)
        frame_landed = [None] * self.L

        # Reads: every expert the round computes, live or not.
        layer_base = (np.arange(self.L) * self.E)[:, None]
        computed = r.routes.reshape(self.L, -1) + layer_base
        landings = []
        reads = 0
        layers_with_reads = 0
        for layer in range(self.L):
            if slots:
                keys, widths = np.unique(computed[layer], return_counts=True)  # widths: the columns per expert
            else:
                keys = np.unique(computed[layer])
            in_vram = self.vram_visible(keys)
            if slots:
                frame_landed[layer] = self._staged(keys[~in_vram], widths[~in_vram])[:slots]
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
        live_counts = []
        round_keys = round_hits = 0
        for layer in range(self.L):
            keys, counts = np.unique(live_routes[layer], return_counts=True)
            groups.append(keys)
            live_counts.append((keys, counts * weight))
            hits = int(np.count_nonzero(self.vram_visible(keys)))
            round_keys += keys.size
            round_hits += hits
            m['live_keys'] += keys.size
            m['live_vram_hits'] += hits
            if early:
                m['early_live_keys'] += keys.size
                m['early_live_vram_hits'] += hits
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

        if self.adaptive:
            self._adaptive_step(live_counts, r)
        else:
            budget = self._budget(r)
            self.applied_budgets.append(budget)
            if self.belady:
                self._belady_advance()
            guarded = self.guard(self.round, r) if self.guard is not None else None
            layer_guard = guarded if callable(guarded) else None
            round_guard = None if guarded is None or layer_guard is not None else np.asarray(guarded, np.int64)
            if round_guard is not None:
                self.guard_mask[round_guard] = True
            try:
                for layer in range(self.L):
                    # The layer's landed experts first: they use up its promotion budget
                    # (ExpertResidency::after_round).
                    adopted = self._adopt(layer, frame_landed[layer], groups[layer])
                    self.layer_guard, self.guard_layer = layer_guard, layer
                    admitted = self._vram_step(groups[layer], max(0, budget - adopted), layer)
                    if self.guard_keys is not None:
                        self.guard_mask[self.guard_keys] = False
                        self.guard_keys = None
                    self.layer_guard = None
                    if self.admission_log is not None:
                        self.admission_log.append((self.round, layer, admitted, adopted, budget))
            finally:
                if round_guard is not None:
                    self.guard_mask[round_guard] = False
        self.protect.clear()
        if r.kind in (DECODE, VERIFY):
            for lane in (int(x) for x in r.lanes):
                if lane in self.since_prompt:
                    self.since_prompt[lane] += r.tokens
        if self.round_log is not None:
            self.round_log.append((r.kind, int(r.lanes[0]), round_keys, round_hits,
                                   self.promotions - promotions_before, self.landed - landed_before))
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

def parse_prompt_seed(text):
    """"none" or CREDIT[,BUDGET[,WINDOW]] -> None or (credit, budget, window)."""
    text = text.strip()
    if text in ('', 'none'):
        return None
    parts = [float(x) for x in text.split(',')] + [0, 0]
    if not parts[0] > 0:
        raise ValueError(f'prompt seed {text!r}: CREDIT must be > 0')
    return (parts[0], int(parts[1]), int(parts[2]))


def format_prompt_seed(seed):
    return 'none' if seed is None else f'credit {seed[0]:g}, budget {seed[1]}, window {seed[2]}'


# The engine's adaptive tier (ExpertResidency::Adaptive): Basalt's rules (b80e026, bde2328) of decay 0.85 per round,
# margin 1.5 and 96 swaps a round, with the candidate floor 1.0 where Basalt's is 2.0 (2026-10-29).
ADAPTIVE_DEFAULTS = (0.85, 1.0, 1.5, 96)


def parse_vram_policy(text):
    """"lfru" -> None (the engine's LFRU), or "adaptive[:DECAY,FLOOR,MARGIN,SWAPS]" -> the four rules."""
    text = text.strip()
    if text == 'lfru':
        return None
    if text == 'adaptive' or text.startswith('adaptive:'):
        values = list(ADAPTIVE_DEFAULTS)
        if text.startswith('adaptive:'):
            given = [x for x in text[9:].split(',') if x]
            if len(given) > 4:
                raise ValueError(f'VRAM policy {text!r}: at most DECAY,FLOOR,MARGIN,SWAPS')
            for i, x in enumerate(given):
                values[i] = float(x)
        decay, floor, margin, swaps = values
        if not (0.0 < decay <= 1.0 and floor >= 0.0 and margin >= 0.0 and swaps >= 1):
            raise ValueError(f'VRAM policy {text!r}: DECAY in (0, 1], FLOOR and MARGIN >= 0, SWAPS >= 1')
        return (decay, floor, margin, int(swaps))
    raise ValueError(f'unknown VRAM policy {text!r} (lfru, adaptive[:DECAY,FLOOR,MARGIN,SWAPS])')


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


def parse_layer_groups(text):
    """"0-8,12;9-11,13-47" -> [[0..8, 12], [9..11, 13..47]]; "" -> None (all layers one group)."""
    groups = []
    for group in (g.strip() for g in text.split(';')):
        if not group:
            continue
        layers = []
        for part in (p.strip() for p in group.split(',') if p.strip()):
            first, _, last = part.partition('-')
            layers.extend(range(int(first), int(last or first) + 1))
        groups.append(layers)
    return groups or None


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
    ap.add_argument('--vram-policies', default='lfru',
                    help='";"-separated VRAM policies: lfru (the engine) and adaptive[:DECAY,FLOOR,MARGIN,SWAPS] '
                         f'(the engine\'s, default {",".join(f"{x:g}" for x in ADAPTIVE_DEFAULTS)}; Basalt\'s floor is 2; '
                         'candidates D12; swaps pair across the layers of each --adaptive-groups group, BG-01)')
    ap.add_argument('--adaptive-groups', default='',
                    help='adaptive: ";"-separated groups of layers sharing frames (the engine\'s per-stride '
                         'pools), each a comma list of layers or FIRST-LAST ranges; empty: all layers one group')
    ap.add_argument('--t0-ms', type=float, default=11.2, help='base decode time per token (plain)')
    ap.add_argument('--cost-ms', type=float, default=0.65, help='exposed time per decode SSD read')
    ap.add_argument('--miss-ms', type=float, default=0.035,
                    help='exposed time per decode VRAM miss (CPU or link service), for the link-charged speed')
    ap.add_argument('--link-gbs', type=float, default=27.5,
                    help='host-to-device link rate charged to each promotion (PCIe 5.0 x8 here)')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--json', default=None)
    ap.add_argument('--validate', action='store_true',
                    help='full mode at the trace frames with recorded budgets: VRAM hit rate vs the engine')
    ap.add_argument('--prompt-seeds', default='none',
                    help='";"-separated prompt seedings, each "none" or CREDIT[,BUDGET[,WINDOW]] '
                         '(the engine seeds 256 with no budget or window; a trace captured from it '
                         'already carries that seeding\'s effect on its routes)')
    args = ap.parse_args(argv)
    seeds = [parse_prompt_seed(x) for x in args.prompt_seeds.split(';')]
    vram_policies = [x.strip() for x in args.vram_policies.split(';') if x.strip()]
    for vram_policy in vram_policies:
        parse_vram_policy(vram_policy)
    adaptive_groups = parse_layer_groups(args.adaptive_groups)

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
        for seed in seeds:
            result = run_config(paths, {'vram': header['frames'], 'ram_slots': None, 'vram_budget': 'recorded',
                                        'prior': 'none', 'prompt_seed': seed, 'follow_frames': True})
            s = result['total']
            print(f'full mode, {header["frames"]} frames, recorded budgets, prompt seed {format_prompt_seed(seed)}: '
                  f'VRAM hits {100 * s["vram_hit_live"]:.2f} % of live routed experts (engine report: "since start" '
                  f'hit rate), decode {100 * s["decode_vram_hit_live"]:.2f} %, first {EARLY_TOKENS} decode tokens '
                  f'{100 * s["early_vram_hit_live"]:.2f} % ({s["early_live_keys"]} keys), {s["promotions"]} promotions '
                  f'({s["prompt_seed_promotions"]} by {s["prompt_seeds"]} seeds)')
        recorded = [r for t in traces for r in t.rounds if r.frames]
        if recorded:
            weights = [max(r.tokens, 1) for r in recorded]
            print(f'per-round pool: {min(r.frames for r in recorded)}..{max(r.frames for r in recorded)} frames '
                  f'over {len(recorded)} rounds; mean {sum(r.frames for r in recorded) / len(recorded):.0f}; '
                  f'token-weighted {sum(r.frames * w for r, w in zip(recorded, weights)) / sum(weights):.0f}')
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
                        for seed in seeds:
                            for vram_policy in vram_policies:
                                config = {'vram': vram, 'ram_slots': slots, 'policy': policy, 'mode': mode,
                                          'prior': prior, 'demotion_allowance': args.demotion_allowance,
                                          'reserve': not args.no_reserve,
                                          'prefetch_after_prefill': args.prefetch_after_prefill,
                                          'vram_budget': args.vram_budget, 'prompt_seed': seed,
                                          'vram_policy': vram_policy, 'adaptive_groups': adaptive_groups}
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
        cells.setdefault((c['prior'], c['mode'], c['vram'], c['ram_slots'], result['ram_mib'],
                          format_prompt_seed(c['prompt_seed']), c['vram_policy']), []).append(result)
    verdicts = []
    for (prior, mode, vram, slots, mib, seed, vram_policy), group in cells.items():
        h_res = None if slots is None else max(0, slots - RAM_EXTRA_SLOTS)
        print(f'\n=== V {vram}, RAM {"all" if slots is None else f"{mib} MiB = {slots} slots (H_res {h_res})"}, '
              f'fast capacity {"all" if slots is None else vram + h_res}, {mode}, prior {prior}, prompt seed {seed}, '
              f'VRAM policy {vram_policy}')
        print(f'{"policy":9s} {"reads/tok":>9s} {"2nd half":>8s} {"1st half":>8s} {"prefill/1K":>10s} '
              f'{"SSD total":>9s} {"VRAM%":>6s} {"RAM%":>6s} {"demote/tok":>10s} {"deferred":>8s} {"drops":>6s} '
              f'{"tok/s 2nd":>9s} {"miss/tok":>8s} {"promo/tok":>9s} {"tok/s link":>10s}')
        reads = {}
        for result in group:
            s, s2, s1 = result['total'], result['second_half'], result['first_half']
            speed = 1000.0 / (args.t0_ms + s2['reads_per_token'] * args.cost_ms)
            linked = link_charged_speed(s2, t0_ms=args.t0_ms, cost_ms=args.cost_ms, miss_ms=args.miss_ms,
                                        link_gbs=args.link_gbs)
            result['link_charged_tok_s'] = linked
            print(f'{result["config"]["policy"]:9s} {s["reads_per_token"]:9.2f} {s2["reads_per_token"]:8.2f} '
                  f'{s1["reads_per_token"]:8.2f} {s["prefill_reads_per_1k_tokens"]:10.1f} {s["ssd_reads"]:9d} '
                  f'{100 * s["vram_share"]:6.1f} {100 * s["ram_share"]:6.1f} {s["demotions_per_token"]:10.2f} '
                  f'{s["deferred_admissions"]:8d} {s["vram_drops"]:6d} {speed:9.1f} '
                  f'{s2["decode_vram_misses_per_token"]:8.1f} {s2["promotions_per_token"]:9.2f} {linked:10.1f}')
            reads[result['config']['policy']] = s['ssd_reads']
        verdict = policy_verdict(reads)
        if verdict:
            verdicts.append({'prior': prior, 'mode': mode, 'vram': vram, 'ram_slots': slots, 'prompt_seed': seed,
                             **verdict})
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
