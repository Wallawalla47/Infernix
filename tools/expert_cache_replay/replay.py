"""Trace-driven replay of VRAM expert-cache policies for decode routing.

Supporting evidence for docs/maintainer/qwen3_8-flash-next-design.md (section 9.4); the starting
point of milestone M1. Python 3.11 + numpy.

Input, either of:

- A directory with warmup-trace.npz, a-trace.npz and b-trace.npz (Qwen3.8-Flash-Next), each holding
  `ids` [steps, layers, >= top_k] int32 with the layer's top-k expert ids in the first top_k columns,
  as published in https://github.com/hz1ulqu01gmnZH4/qwen38-freetoken
  (experiments/2026-09-22_tg-expert-cache-trace-replay/data). The layers come from the arrays;
  --experts and --top-k (default 512 and 10, Flash-Next's) give the rest.
- Route traces written by the engine (EngineOptions::route_trace_path, `infernix-serve --route-trace`,
  `infernix_decode_quality_gen ... route-trace=FILE`; format in src/runtime/expert_offload/
  route_trace.h), from any model with offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B):
  layers, experts and top-k come from the header. The decode and verification rounds are replayed in
  the order given as one session; prefill chunks and forced tokens are served by the prefill stream
  and are not part of the decode policies' accesses. A group is one (round, layer): the distinct
  experts the round computes in that layer, rejected draft columns included. The first
  --warmup-tokens decode tokens warm every policy and are not scored.

With the FreeToken traces, `warmup` warms every policy and A and B are scored. Accesses are
event-atomic: the experts of the current group must be resident together, so the current group is
never evicted. With --window W (FreeToken traces only), a group is the per-layer union of W
consecutive tokens, approximating a fully accepted W-token speculative verify window.

Policies admit every miss ("promotion") unless marked CPU-served, in which case misses are computed
by the CPU and the policy promotes lazily. Output: hit rate, misses and promotions per token.

Headroom report (--headroom, route traces only): the engine's own policy (the LFRU with budgeted
admission, replayed by host_tier.TierSim in full mode: every miss is served from host memory)
against Belady MIN with bypass (the optimum when misses are served from host memory, as the engine
serves them) and Belady MIN without bypass (the demand-fetch optimum), per layer group, at each
--capacities frame count. Hit rates count the distinct experts each scored decode or verification
round computes per layer. The decision (--decision-frames, default the trace's frames) follows the
candidate's rule: build cache-policy work only on a gap of >= 3 points between Belady with bypass and
the engine's policy; a gap under 1 point closes the topic.

Protection arms (--headroom --protection; candidates §13 PUB-22, SeqMoE): the engine's LFRU whose
evictions spare the experts a routing-sequence predictor expects next (route_predictor.py): the next
rounds' actual keys (the oracle ceiling, per --protect-horizons) and the engine's causal transition
predictor (INFERNIX_TMP_ROUTE_PROTECT, on the host, so at the same frames). A/B the switch only when
the predictor gains >= 3 points over the engine's LFRU; an oracle ceiling under 3 points closes the
topic.

    py -3.11 -m tools.expert_cache_replay.replay DIR [--capacities 4008,10200] [--window 1]
    py -3.11 -m tools.expert_cache_replay.replay TRACE [TRACE ...] [--warmup-tokens N]
    py -3.11 -m tools.expert_cache_replay.replay TRACE [TRACE ...] --headroom [--capacities 3600,4815]
        [--decision-frames 4815] [--layer-groups 0-19,20-47] [--warmup-tokens N] [--json OUT]
        [--protection [--protect-horizons 1,8] [--protect-per-layer 20]]
"""
import argparse
import heapq
import json
import os
from collections import OrderedDict, defaultdict, namedtuple

import numpy as np

from . import host_tier as ht
from . import route_predictor as rpred

Geometry = namedtuple('Geometry', 'layers experts top_k')
FLASH_NEXT = Geometry(48, 512, 10)

BUILD_GAP_POINTS = 3.0   # the candidate's rule: policy work only on a gap of 3 points or more
CLOSE_GAP_POINTS = 1.0   # under 1 point the topic closes


def load(directory, experts=FLASH_NEXT.experts, top_k=FLASH_NEXT.top_k):
    """Returns the geometry and [steps, L, K] global expert keys (layer * E + expert) for warmup, a, b."""
    seqs = []
    layers = None
    for n in ('warmup', 'a', 'b'):
        ids = np.load(f'{directory}/{n}-trace.npz')['ids']
        if layers is None:
            layers = ids.shape[1]
        if ids.ndim != 3 or ids.shape[1] != layers or ids.shape[2] < top_k:
            raise ValueError(f'{directory}/{n}-trace.npz: ids {ids.shape} is not [steps, {layers}, >= {top_k}]')
        ids = ids[:, :, :top_k].astype(np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= experts):
            raise ValueError(f'{directory}/{n}-trace.npz: expert ids outside 0..{experts - 1}')
        keys = ids + (np.arange(layers)[None, :, None] * experts)
        seqs.append(keys)
    return Geometry(layers, experts, top_k), seqs


class Sim:
    """Base: groups of keys per (step, layer); the current group is never evicted."""
    demand = True   # admit every miss

    def __init__(self, cap):
        self.cap = cap
        self.hits = self.miss = self.promo = 0

    def run(self, groups, score):
        for g in groups:
            self.step_group(g, score)


# ---------------------------------------------------------------- recency family
class LRU(Sim):
    name = 'LRU'

    def __init__(self, cap):
        super().__init__(cap)
        self.c = OrderedDict()

    def step_group(self, g, score):
        c = self.c
        missing = []
        for k in g:
            if k in c:
                c.move_to_end(k)
                if score: self.hits += 1
            else:
                missing.append(k)
        if score: self.miss += len(missing)
        gs = set(g)
        for k in missing:
            while len(c) >= self.cap:
                v = next(iter(c))
                if v in gs:   # protected current group: rotate it to MRU
                    c.move_to_end(v); continue
                c.popitem(last=False)
            c[k] = True
            if score: self.promo += 1


class LRUPerLayer(Sim):
    name = 'LRU, equal per-layer partition'

    def __init__(self, cap, geo):
        super().__init__(cap)
        self.E = geo.experts
        self.c = [OrderedDict() for _ in range(geo.layers)]
        self.q = cap // geo.layers

    def step_group(self, g, score):
        c = self.c[int(g[0]) // self.E]
        miss = []
        for k in g:
            if k in c:
                c.move_to_end(k); self.hits += score
            else:
                miss.append(k)
        self.miss += score * len(miss)
        gs = set(g)
        for k in miss:
            while len(c) >= self.q:
                v = next(iter(c))
                if v in gs: c.move_to_end(v); continue
                c.popitem(last=False)
            c[k] = True; self.promo += score


class SLRU(Sim):
    name = 'SLRU (20% probation / 80% protected)'

    def __init__(self, cap, prot=0.8):
        super().__init__(cap)
        self.p = OrderedDict(); self.m = OrderedDict(); self.pcap = int(cap * prot)

    def step_group(self, g, score):
        gs = set(g); miss = []
        for k in g:
            if k in self.m:
                self.m.move_to_end(k); self.hits += score
            elif k in self.p:
                del self.p[k]; self.m[k] = True; self.hits += score
                while len(self.m) > self.pcap:
                    v = next(iter(self.m))
                    if v in gs: self.m.move_to_end(v); continue
                    self.m.popitem(last=False); self.p[v] = True
            else:
                miss.append(k)
        self.miss += score * len(miss)
        for k in miss:
            while len(self.p) + len(self.m) >= self.cap:
                src = self.p if self.p and any(x not in gs for x in self.p) else self.m
                v = next(iter(src))
                if v in gs: src.move_to_end(v); continue
                src.popitem(last=False)
            self.p[k] = True; self.promo += score


class ARC(Sim):
    name = 'ARC'

    def __init__(self, cap):
        super().__init__(cap)
        self.t1 = OrderedDict(); self.t2 = OrderedDict(); self.b1 = OrderedDict(); self.b2 = OrderedDict(); self.pp = 0

    def _replace(self, k, gs):
        c = self.cap
        for _ in range(4 * c):
            use_t1 = self.t1 and (len(self.t1) > self.pp or (k in self.b2 and len(self.t1) == self.pp))
            src, ghost = (self.t1, self.b1) if use_t1 or not self.t2 else (self.t2, self.b2)
            if not src:
                src, ghost = (self.t2, self.b2) if src is self.t1 else (self.t1, self.b1)
            v = next(iter(src))
            if v in gs:
                src.move_to_end(v); continue
            src.popitem(last=False); ghost[v] = True
            return

    def step_group(self, g, score):
        gs = set(g); c = self.cap
        for k in g:
            if k in self.t1:
                del self.t1[k]; self.t2[k] = True; self.hits += score
            elif k in self.t2:
                self.t2.move_to_end(k); self.hits += score
            else:
                self.miss += score; self.promo += score
                if k in self.b1:
                    self.pp = min(c, self.pp + max(1, len(self.b2) // max(1, len(self.b1))))
                    self._replace(k, gs); del self.b1[k]; self.t2[k] = True
                elif k in self.b2:
                    self.pp = max(0, self.pp - max(1, len(self.b1) // max(1, len(self.b2))))
                    self._replace(k, gs); del self.b2[k]; self.t2[k] = True
                else:
                    if len(self.t1) + len(self.b1) >= c:
                        if len(self.t1) < c:
                            self.b1.popitem(last=False); self._replace(k, gs)
                        else:
                            self._replace(k, gs)   # evicts from t1 into b1
                            if len(self.b1) > c: self.b1.popitem(last=False)
                    elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= c:
                        if len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= 2 * c and self.b2:
                            self.b2.popitem(last=False)
                        if len(self.t1) + len(self.t2) >= c:
                            self._replace(k, gs)
                    self.t1[k] = True


class LRU2(Sim):
    name = 'LRU-2'

    def __init__(self, cap):
        super().__init__(cap)
        self.res = set(); self.hist = {}; self.t = 0; self.h = []; self.hq = heapq

    def step_group(self, g, score):
        self.t += 1; gs = set(g); miss = []
        for k in g:
            last, pen = self.hist.get(k, (-1, -1))
            self.hist[k] = (self.t, last)
            if k in self.res:
                self.hits += score; self.hq.heappush(self.h, (last, self.t, k))
            else: miss.append(k)
        self.miss += score * len(miss)
        for k in miss:
            if len(self.res) >= self.cap:
                held = []
                while True:
                    pen, last, v = self.hq.heappop(self.h)
                    if v not in self.res or self.hist[v] != (last, pen): continue
                    if v in gs: held.append((pen, last, v)); continue
                    break
                for x in held: self.hq.heappush(self.h, x)
                self.res.discard(v)
            self.res.add(k); self.promo += score
            last, pen = self.hist[k]; self.hq.heappush(self.h, (pen, last, k))


class SIEVE(Sim):
    name = 'SIEVE'

    def __init__(self, cap):
        super().__init__(cap)
        self.prev = {}; self.next = {}; self.vis = {}; self.head = None; self.tail = None; self.hand = None

    def _remove(self, v):
        p, n = self.prev.pop(v), self.next.pop(v); del self.vis[v]
        if p is not None: self.next[p] = n
        else: self.head = n
        if n is not None: self.prev[n] = p
        else: self.tail = p

    def step_group(self, g, score):
        gs = set(g); miss = []
        for k in g:
            if k in self.vis: self.vis[k] = True; self.hits += score
            else: miss.append(k)
        self.miss += score * len(miss)
        for k in miss:
            if len(self.vis) >= self.cap:
                o = self.hand if self.hand is not None else self.tail
                while self.vis[o] or o in gs:
                    self.vis[o] = False
                    o = self.prev[o] if self.prev[o] is not None else self.tail
                self.hand = self.prev[o]
                self._remove(o)
            self.prev[k] = None; self.next[k] = self.head; self.vis[k] = False
            if self.head is not None: self.prev[self.head] = k
            self.head = k
            if self.tail is None: self.tail = k
            self.promo += score


class S3FIFO(Sim):
    name = 'S3-FIFO'

    def __init__(self, cap):
        super().__init__(cap)
        self.s = OrderedDict(); self.m = OrderedDict(); self.g = OrderedDict(); self.f = {}
        self.scap = max(1, cap // 10)

    def _evict(self, gs):
        for _ in range(10 * self.cap):
            if len(self.s) >= self.scap or not self.m:
                v = next(iter(self.s)); self.s.popitem(last=False)
                if v in gs or self.f[v] > 0:
                    self.f[v] = 0 if v not in gs else self.f[v]; self.m[v] = True
                    continue
                del self.f[v]; self.g[v] = True
                if len(self.g) > self.cap: self.g.popitem(last=False)
                return
            v = next(iter(self.m)); self.m.popitem(last=False)
            if v in gs or self.f[v] > 0:
                self.f[v] = max(0, self.f[v] - 1); self.m[v] = True; continue
            del self.f[v]; return

    def step_group(self, g, score):
        gs = set(g)
        for k in g:
            if k in self.f:
                self.f[k] = min(3, self.f[k] + 1); self.hits += score; continue
            self.miss += score; self.promo += score
            while len(self.s) + len(self.m) >= self.cap: self._evict(gs)
            if k in self.g: del self.g[k]; self.m[k] = True
            else: self.s[k] = True
            self.f[k] = 0


# ---------------------------------------------------------------- frequency family
class DecayedLFU(Sim):
    name = 'decayed global LFU'

    def __init__(self, cap, geo, half=64):
        super().__init__(cap)
        self.E = geo.experts
        self.res = set(); self.cnt = defaultdict(float); self.last = {}; self.t = 0; self.half = half
        self.h = []; self.hq = heapq; self.step = 0
        self.name = f'decayed global LFU (half-life {half} tokens)'

    def step_group(self, g, score):
        self.t += 1
        if int(g[0]) // self.E == 0: self.step += 1
        inc = 2.0 ** (self.step / self.half)   # exponential decay by scaling increments
        gs = set(g); miss = []
        for k in g:
            self.cnt[k] += inc; self.last[k] = self.t
            if k in self.res:
                self.hits += score; self.hq.heappush(self.h, (self.cnt[k], self.t, k))
            else: miss.append(k)
        self.miss += score * len(miss)
        for k in miss:
            if len(self.res) >= self.cap:
                held = []
                while True:
                    c, t, v = self.hq.heappop(self.h)
                    if v not in self.res or self.cnt[v] != c or self.last[v] != t: continue
                    if v in gs: held.append((c, t, v)); continue
                    break
                for x in held: self.hq.heappush(self.h, x)
                self.res.discard(v)
            self.res.add(k); self.promo += score; self.hq.heappush(self.h, (self.cnt[k], self.t, k))


class WTinyLFU(Sim):
    """1% window LRU + SLRU main; candidates evicted from the window enter main only if their
    (aged) frequency beats the main victim's. Rejected candidates are dropped (they were already
    copied into the window, so promotions are still counted on window admission)."""
    name = 'W-TinyLFU (1% window)'

    def __init__(self, cap, geo, win=0.01, age=2048):
        super().__init__(cap)
        self.K = geo.top_k
        self.wcap = max(geo.top_k + 1, int(cap * win)); self.w = OrderedDict()
        self.main = SLRU(cap - self.wcap); self.f = defaultdict(int); self.n = 0; self.age = age * geo.layers

    def _in_main(self, k): return k in self.main.p or k in self.main.m

    def step_group(self, g, score):
        gs = set(g)
        for k in g:
            self.f[k] += 1; self.n += 1
        if self.n >= self.age * self.K:
            for k in list(self.f): self.f[k] //= 2
            self.n = 0
        for k in g:
            if k in self.w: self.w.move_to_end(k); self.hits += score
            elif self._in_main(k):
                self.hits += score
                if k in self.main.p:
                    del self.main.p[k]; self.main.m[k] = True
                    while len(self.main.m) > self.main.pcap:
                        v = next(iter(self.main.m))
                        if v in gs: self.main.m.move_to_end(v); continue
                        self.main.m.popitem(last=False); self.main.p[v] = True
                else:
                    self.main.m.move_to_end(k)
            else:
                self.miss += score; self.promo += score
                self.w[k] = True
                while len(self.w) > self.wcap:
                    c = next(iter(self.w))
                    if c in gs: self.w.move_to_end(c); continue
                    self.w.popitem(last=False)
                    mm = self.main
                    if len(mm.p) + len(mm.m) < mm.cap:
                        mm.p[c] = True; continue
                    src = mm.p if mm.p else mm.m
                    vict = next((x for x in src if x not in gs), None)
                    if vict is not None and self.f[c] > self.f[vict]:
                        del src[vict]; mm.p[c] = True


# ---------------------------------------------------------------- CPU-serve policies (no demand admission)
class StrataExchange(Sim):
    """Strata-style: misses are computed on the CPU (no admission). Every 4 steps, per layer, swap the
    hottest non-resident (decayed count >= 2) for the coldest resident if >= victim + 1.5 (additive, as generate.cpp adapt()); <= 96 swaps,
    then decay counts by 0.7. Seeded from a static profile (warmup frequencies)."""
    name = 'Strata-style decayed-LFU exchange (CPU-served misses)'
    demand = False

    def __init__(self, cap, geo, seed_counts):
        super().__init__(cap)
        self.L, self.E = geo.layers, geo.experts
        order = sorted(range(self.L * self.E), key=lambda k: -seed_counts.get(k, 0))
        self.res = set(order[:cap]); self.cnt = defaultdict(float); self.step = 0

    def step_group(self, g, score):
        L, E = self.L, self.E
        for k in g:
            self.cnt[k] += 1
            if k in self.res: self.hits += score
            else: self.miss += score
        if int(g[0]) // E == L - 1:
            self.step += 1
            if self.step % 4 == 0:
                cands = []
                for layer in range(L):
                    lo = layer * E
                    out = [(self.cnt[k], k) for k in range(lo, lo + E) if k not in self.res and self.cnt[k] >= 2]
                    if not out: continue
                    ins = sorted((self.cnt[k], k) for k in range(lo, lo + E) if k in self.res)
                    out.sort(reverse=True)
                    for (co, ko), (ci, ki) in zip(out, ins):
                        if co >= ci + 1.5: cands.append((co - ci, ko, ki))
                        else: break
                cands.sort(reverse=True)
                for _, ko, ki in cands[:96]:
                    self.res.discard(ki); self.res.add(ko); self.promo += score
                for k in list(self.cnt): self.cnt[k] *= 0.7


class StaticHindsight(Sim):
    name = 'static top-N by hindsight frequency of scored traces (not deployable)'
    demand = False

    def __init__(self, cap, counts):
        super().__init__(cap)
        self.res = set(sorted(counts, key=lambda k: -counts[k])[:cap])

    def step_group(self, g, score):
        for k in g:
            if k in self.res: self.hits += score
            else: self.miss += score


class Belady(Sim):
    """Farthest-next-use with group protection. bypass=True: a miss is not admitted when its next use
    is farther than every evictable resident's (optimal for the CPU-serve model's hit count)."""

    def __init__(self, cap, nxt, bypass=False):
        super().__init__(cap)
        self.nxt = nxt; self.bypass = bypass; self.i = 0
        self.res = {}  # key -> next use index
        self.h = []; self.hq = heapq
        self.name = 'Belady MIN' + (' with bypass (CPU-serve optimum)' if bypass else ' (demand-fetch optimum)')
        self.demand = not bypass

    def step_group(self, g, score):
        hq = self.hq; gs = set(g); miss = []
        for j, k in enumerate(g):
            nu = self.nxt[self.i][j]
            if k in self.res:
                self.res[k] = nu; hq.heappush(self.h, (-nu, k)); self.hits += score
            else:
                miss.append((k, nu))
        self.miss += score * len(miss)
        for k, nu in miss:
            if len(self.res) >= self.cap:
                held = []
                while True:
                    negnu, v = hq.heappop(self.h)
                    if v not in self.res or self.res[v] != -negnu: continue
                    if v in gs: held.append((negnu, v)); continue
                    break
                for x in held: hq.heappush(self.h, x)
                if self.bypass and -negnu < nu:
                    hq.heappush(self.h, (negnu, v)); continue
                del self.res[v]
            self.res[k] = nu; hq.heappush(self.h, (-nu, k)); self.promo += score
        self.i += 1


class LazyLRFU(Sim):
    """CPU-served misses; once per decode round, promote the highest-score non-resident experts over the
    lowest-score residents while score(candidate) >= score(victim) + margin, at most `budget` per round.
    Score: exponentially decayed reference count (half-life in rounds), i.e. LRFU."""
    demand = False

    def __init__(self, cap, geo, seed_counts, half=128, margin=1.0, budget=16):
        super().__init__(cap)
        self.L, self.E = geo.layers, geo.experts
        order = sorted(range(self.L * self.E), key=lambda k: -seed_counts.get(k, 0))
        self.res = set(order[:cap]); self.s = defaultdict(float); self.step = 0
        self.half, self.margin, self.budget = half, margin, budget
        self.touched = set(); self.name = f'lazy LRFU hl{half} margin{margin} budget{budget}/round'

    def step_group(self, g, score):
        inc = 2.0 ** (self.step / self.half)
        for k in g:
            self.s[k] += inc; self.touched.add(int(k))
            if k in self.res: self.hits += score
            else: self.miss += score
        if int(g[0]) // self.E == self.L - 1:
            self.step += 1
            unit = 2.0 ** (self.step / self.half)
            cands = sorted((k for k in self.touched if k not in self.res), key=lambda k: -self.s[k])[:self.budget]
            if cands:
                vict = heapq.nsmallest(len(cands), self.res, key=lambda k: self.s[k])
                for c, v in zip(cands, vict):
                    if self.s[c] >= self.s[v] + self.margin * unit:
                        self.res.discard(v); self.res.add(c); self.promo += score
                    else:
                        break
            self.touched = set()


COUNT_FLOOR = 1.0 / 1024.0   # halved counts below this become 0 (LfruPolicy::kCountFloor)
ENGINE_HALVING = 8192        # the engine's period (ExpertResidency::kLfruHalvingPeriod)


class LFRU(Sim):
    """Zhang (arXiv 2608.07911): score = f / (clock - last + 1), f a global count never reset on eviction;
    evict the lowest score among non-current residents; admit every miss. Clock in layer-call ticks. With
    `halving` > 0 every count halves exactly each that many ticks (the engine's LfruPolicy, operation for
    operation in binary64)."""
    name = 'LFRU f/(age+1)'

    def __init__(self, cap, halving=0):
        super().__init__(cap)
        self.res = set(); self.f = defaultdict(float); self.last = {}; self.t = 0; self.halving = halving
        if halving:
            self.name = f'LFRU f/(age+1), halving {halving}'
        self.trace = None   # set to a list to record (group, victims) for conformance fixtures

    def step_group(self, g, score):
        self.t += 1; gs = set(g); miss = []
        if self.halving and self.t % self.halving == 0:
            for k, c in self.f.items():
                self.f[k] = 0.0 if c * 0.5 < COUNT_FLOOR else c * 0.5
        for k in g:
            self.f[k] += 1.0; self.last[k] = self.t
            if k in self.res: self.hits += score
            else: miss.append(k)
        self.miss += score * len(miss)
        if miss and len(self.res) + len(miss) > self.cap:
            t = self.t
            need = len(self.res) + len(miss) - self.cap
            # IEEE binary64 score, ties to the lower key: the engine's policy reproduces this exactly.
            vict = heapq.nsmallest(need, (x for x in self.res if x not in gs),
                                   key=lambda x: (self.f[x] / (t - self.last[x] + 1), x))
            for v in vict: self.res.discard(v)
        else:
            vict = []
        if self.trace is not None:
            self.trace.append(([int(k) for k in g], [int(v) for v in vict]))
        for k in miss:
            self.res.add(k); self.promo += score


def next_uses(groups):
    last = {}
    out = [None] * len(groups)
    for i in range(len(groups) - 1, -1, -1):
        out[i] = [last.get(int(k), 10 ** 12) for k in groups[i]]
        for k in groups[i]:
            last[int(k)] = i
    return out


def to_groups(seq, window):
    groups = []
    for s in range(0, len(seq) - window + 1, window):
        for layer in range(seq.shape[1]):
            groups.append(np.array(list(dict.fromkeys(int(x) for x in seq[s:s + window, layer].reshape(-1))),
                                   dtype=np.int64))
    return groups


# ---------------------------------------------------------------- engine route traces
def is_route_trace(path):
    if not os.path.isfile(path):
        return False
    with open(path, 'rb') as f:
        return f.read(8) == b'NRTRACE1'


def read_route_traces(paths):
    """Parses engine route traces that agree on layers, experts and top-k -> (geometry, traces)."""
    traces = [ht.read_trace(p) for p in paths]
    h = traces[0].header
    geo = Geometry(h['layers'], h['experts'], h['top_k'])
    for p, t in zip(paths, traces):
        if (t.header['layers'], t.header['experts'], t.header['top_k']) != tuple(geo):
            raise ValueError(f'{p}: layers, experts or top_k differ from {paths[0]}')
    if min(geo) == 0:
        raise ValueError(f'{paths[0]}: the header has no layers, experts or top_k')
    return geo, traces


def round_keys(r, geo, layer):
    """The distinct global keys (layer * E + expert) round `r` computes in `layer`, ascending."""
    ids = r.routes[layer].ravel()
    ids = np.unique(ids[(ids >= 0) & (ids < geo.experts)]).astype(np.int64)
    if ids.size == 0:
        raise ValueError(f'a round routes no valid expert in layer {layer}')
    return ids + layer * geo.experts


def decode_groups(traces, geo, warmup_tokens):
    """The decode policies' accesses: one group per (decode or verification round, layer), in order.
    Returns (groups, scored flags, scored tokens, warm-up tokens). A round is scored once the decode
    tokens before it reach `warmup_tokens`."""
    groups, scored = [], []
    seen = tokens = 0
    for t in traces:
        for r in t.rounds:
            if r.kind not in (ht.DECODE, ht.VERIFY):
                continue
            score = seen >= warmup_tokens
            seen += r.tokens
            tokens += r.tokens if score else 0
            for layer in range(geo.layers):
                groups.append(round_keys(r, geo, layer))
                scored.append(score)
    return groups, scored, tokens, seen - tokens


def engine_rule(rounds):
    """The engine's admission budgets, read from the recorded ones: (promotions per layer of a
    prefill chunk wider than ExpertOffload::kServedCallColumns, decode tokens per promotion once the
    frames are full). Qwen3.8-Flash-Next records 16 per chunk; Qwen3.5-122B-A10B records 0 (its
    prompt passes stream every missing expert). The decode interval scales with the measured link
    (ExpertOffload::promotion_interval), so it comes from the last half of the decode rounds, past
    the fill phase."""
    chunks = [r.budget for r in rounds if r.kind == ht.PREFILL and r.rows * r.width > 8]
    prefill = max(chunks) if chunks else ht.PREFILL_PROMOTIONS
    decode = [r for r in rounds if r.kind in (ht.DECODE, ht.VERIFY)]
    tail = decode[len(decode) // 2:]
    budget, tokens = sum(r.budget for r in tail), sum(r.tokens for r in tail)
    if not tail:
        interval = ht.PROMOTION_INTERVAL
    elif budget == 0:
        interval = 8
    else:
        interval = int(min(8, max(1, round(tokens / budget))))
    return prefill, interval


def parse_layer_groups(text, layers):
    """'0-19,20-47' -> [(name, first, last)]; empty: quarters of the layers."""
    if not text:
        q = max(1, layers // 4)
        bounds = [(i, min(layers, i + q) - 1) for i in range(0, layers, q)]
    else:
        bounds = []
        for part in text.split(','):
            a, _, b = part.partition('-')
            bounds.append((int(a), int(b or a)))
    for a, b in bounds:
        if not 0 <= a <= b < layers:
            raise ValueError(f'layer group {a}-{b} is outside 0-{layers - 1}')
    return [(f'{a}-{b}', a, b) for a, b in bounds]


class ArmResult:
    """Scored hits and misses per layer of one arm."""

    def __init__(self, name, layers):
        self.name = name
        self.hits = np.zeros(layers, np.int64)
        self.keys = np.zeros(layers, np.int64)

    def rate(self, first=0, last=None):
        last = len(self.keys) - 1 if last is None else last
        keys = int(self.keys[first:last + 1].sum())
        return int(self.hits[first:last + 1].sum()) / keys if keys else 0.0

    def misses(self):
        return int((self.keys - self.hits).sum())


def run_decode_arm(sim, groups, scored, geo, name=None):
    """Steps a replay policy over the decode groups, attributing each scored group's hits to its layer."""
    out = ArmResult(name or sim.name, geo.layers)
    for i, g in enumerate(groups):
        before = sim.hits
        sim.step_group(g, 1 if scored[i] else 0)
        if scored[i]:
            layer = int(g[0]) // geo.experts
            out.hits[layer] += sim.hits - before
            out.keys[layer] += len(g)
    return out


def run_engine_arm(traces, geo, frames, warmup_tokens, *, vram_budget='rule', prefill_promotions=None,
                   promotion_interval=None, name=None, guard=None):
    """The engine's policy over every round of the traces (host_tier.TierSim, full mode: no SSD tier,
    the engine's prompt seeding, an empty cache at the start), scored like the decode arms: a scored
    decode or verification round's computed key is a hit when it is resident in a published frame.
    `guard`: eviction protection (TierSim.guard; route_predictor)."""
    header = {'layers': geo.layers, 'experts': geo.experts, 'top_k': geo.top_k}
    rule = {}
    if prefill_promotions is not None:
        rule['prefill_promotions'] = prefill_promotions
    if promotion_interval is not None:
        rule['promotion_interval'] = promotion_interval
    sim = ht.TierSim(header, vram=frames, ram_slots=None, prior='none', vram_budget=vram_budget,
                     prompt_seed=(ht.PROMPT_SEED_CREDIT, 0, 0), **rule)
    sim.guard = guard
    out = ArmResult(name or f'engine LFRU ({vram_budget} budgets)', geo.layers)
    seen = 0
    for t in traces:
        for r in t.rounds:
            if r.kind in (ht.DECODE, ht.VERIFY):
                if seen >= warmup_tokens:
                    for layer in range(geo.layers):
                        keys = round_keys(r, geo, layer)
                        out.hits[layer] += int(np.count_nonzero(sim.vram_visible(keys)))
                        out.keys[layer] += keys.size
                seen += r.tokens
            sim.step(r)
    return out


def verdict(gap_points):
    if gap_points >= BUILD_GAP_POINTS:
        return f'gap >= {BUILD_GAP_POINTS:g} points: build cache-policy work'
    if gap_points < CLOSE_GAP_POINTS:
        return f'gap < {CLOSE_GAP_POINTS:g} point: the cache-policy topic closes'
    return f'gap below {BUILD_GAP_POINTS:g} points: no cache-policy work'


def protection_verdict(gain_points, ceiling_points):
    """PUB-22's rule: A/B the engine's predictor (INFERNIX_TMP_ROUTE_PROTECT, on the host: no frames)
    only on a gain of >= 3 points over the engine's LFRU. `ceiling_points`: the oracle's best gain."""
    if gain_points >= BUILD_GAP_POINTS:
        return f'predictor gain >= {BUILD_GAP_POINTS:g} points: A/B INFERNIX_TMP_ROUTE_PROTECT'
    if ceiling_points < BUILD_GAP_POINTS:
        return (f'oracle protection gains under {BUILD_GAP_POINTS:g} points: no protection predictor can, '
                f'the topic closes')
    return (f'predictor gain below {BUILD_GAP_POINTS:g} points: no A/B (a better sequence model might reach '
            f'the oracle ceiling)')


def protection_arms(traces, rounds, geo, cap, args, prefill, interval):
    """PUB-22's arms at `cap` frames: the engine's LFRU protected by the next rounds' actual keys (one
    arm per --protect-horizons entry), and by the engine's transition predictor."""
    arms = []
    for horizon in (int(x) for x in args.protect_horizons.split(',')):
        oracle = rpred.NextRoundsOracle(rounds, geo.layers, geo.experts, horizon)
        arms.append(run_engine_arm(traces, geo, cap, args.warmup_tokens, prefill_promotions=prefill,
                                   promotion_interval=interval, guard=oracle, name=f'engine LFRU + {oracle.name}'))
    predictor = rpred.TransitionPredictor(geo.layers, geo.experts, args.protect_per_layer or 2 * geo.top_k)
    arms.append(run_engine_arm(traces, geo, cap, args.warmup_tokens, prefill_promotions=prefill,
                               promotion_interval=interval, guard=predictor, name=f'engine LFRU + {predictor.name}'))
    return arms


def headroom(paths, args):
    geo, traces = read_route_traces(paths)
    h = traces[0].header
    rounds = [r for t in traces for r in t.rounds]
    n = geo.layers * geo.experts
    recorded = [r.frames for r in rounds if r.frames]
    trace_frames = h['frames'] or (recorded[0] if recorded else 0)
    decision = args.decision_frames or trace_frames
    if not decision:
        raise SystemExit('the trace records no frame count: pass --decision-frames')
    if args.capacities:
        caps = [int(x) for x in args.capacities.split(',')]
    else:
        caps = sorted({int(round(decision * f)) for f in (0.5, 0.75, 1.0, 1.25, 1.5)})
    if decision not in caps:
        caps = sorted(caps + [decision])
    caps = [c for c in caps if 0 < c < n]
    layer_groups = parse_layer_groups(args.layer_groups, geo.layers)
    prefill, interval = engine_rule(rounds)
    if args.promotion_interval:
        interval = args.promotion_interval
    groups, scored, tokens, warm = decode_groups(traces, geo, args.warmup_tokens)
    if tokens == 0:
        raise SystemExit('no scored decode tokens (a decode trace, or fewer --warmup-tokens)')
    nxt = next_uses(groups)
    kinds = defaultdict(int)
    for r in rounds:
        kinds[ht.KIND_NAMES[r.kind]] += 1
    print(f'route traces: {len(traces)}, {len(rounds)} rounds ({", ".join(f"{v} {k}" for k, v in kinds.items())}); '
          f'{geo.layers} layers x {geo.experts} experts, top-{geo.top_k}; captured with {trace_frames} frames'
          + (f' (per round {min(recorded)}..{max(recorded)})' if recorded else '')
          + f', {h["lanes"]} lanes, max width {h["max_width"]}, MTP {h["mtp_draft_tokens"]}, '
          f'n-gram {h["ngram_draft_tokens"]}')
    print(f'engine rule: {prefill} promotions per layer per prefill chunk; once full, decode promotes one '
          f'expert per layer every {interval} tokens' + (' (set)' if args.promotion_interval else ' (from the trace)'))
    print(f'scored: {tokens} decode tokens after {warm} warm-up tokens; mean distinct experts per layer per '
          f'round {np.mean([len(g) for g, s in zip(groups, scored) if s]):.2f}')
    names = ['all'] + [g[0] for g in layer_groups]
    results = []
    for cap in caps:
        engine = run_engine_arm(traces, geo, cap, args.warmup_tokens, prefill_promotions=prefill,
                                promotion_interval=interval)
        bypass = run_decode_arm(Belady(cap, nxt, True), groups, scored, geo)
        arms = [engine]
        if cap == trace_frames:
            # The budgets the engine applied at its own frame count: ties the replay to the run.
            arms.append(run_engine_arm(traces, geo, cap, args.warmup_tokens, vram_budget='recorded'))
        arms += [bypass, run_decode_arm(Belady(cap, nxt, False), groups, scored, geo)]
        protected = protection_arms(traces, rounds, geo, cap, args, prefill, interval) if args.protection else []
        arms += protected
        print(f'\n=== {cap} frames ({cap / n:.1%} of {n} experts)' + (' (decision)' if cap == decision else ''))
        print(f'{"arm":44s} ' + ' '.join(f'{x:>7s}' for x in names) + f' {"miss/tok":>9s}')
        rows = []
        for arm in arms:
            rates = [arm.rate()] + [arm.rate(a, b) for _, a, b in layer_groups]
            print(f'{arm.name:44s} ' + ' '.join(f'{100 * x:7.2f}' for x in rates) + f' {arm.misses() / tokens:9.2f}')
            rows.append({'arm': arm.name, 'hit': dict(zip(names, rates)), 'misses_per_token': arm.misses() / tokens,
                         'per_layer_hit': [float(x) for x in arm.hits / np.maximum(arm.keys, 1)]})
        gaps = [100 * (bypass.rate() - engine.rate())] + [100 * (bypass.rate(a, b) - engine.rate(a, b))
                                                          for _, a, b in layer_groups]
        print(f'{"gap, Belady with bypass - engine (points)":44s} ' + ' '.join(f'{x:7.2f}' for x in gaps))
        result = {'frames': cap, 'arms': rows, 'gap_points': dict(zip(names, gaps))}
        if protected:
            gains = {arm.name: 100 * (arm.rate() - engine.rate()) for arm in protected}
            for name, gain in gains.items():
                print(f'{"gain, " + name:44s} {gain:7.2f}')
            result['protection_gain_points'] = gains
        results.append(result)
    final = next(r for r in results if r['frames'] == decision)
    gap = final['gap_points']['all']
    print(f'\ndecision at {decision} frames: {gap:.2f} points -> {verdict(gap)}')
    report = {'header': h, 'geometry': geo._asdict(), 'engine_rule': {'prefill_promotions': prefill,
              'promotion_interval': interval}, 'scored_tokens': tokens, 'warmup_tokens': warm,
              'decision_frames': decision, 'decision_gap_points': gap, 'verdict': verdict(gap), 'results': results}
    if args.protection:
        gains = final['protection_gain_points']
        predictor = next(v for k, v in gains.items() if 'transition' in k)
        ceiling = max(v for k, v in gains.items() if 'oracle' in k)
        print(f'protection at {decision} frames: predictor {predictor:.2f} points, oracle ceiling {ceiling:.2f} '
              f'-> {protection_verdict(predictor, ceiling)}')
        report.update({'protection_gain_points': predictor, 'protection_ceiling_points': ceiling,
                       'protection_verdict': protection_verdict(predictor, ceiling)})
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(report, f, indent=1)
    return 0


def policy_table(geo, warm, scored, tokens, capacities):
    groups = warm + scored
    nxt = next_uses(groups)
    warm_counts = defaultdict(int)
    for g in warm:
        for k in g:
            warm_counts[int(k)] += 1
    scored_counts = defaultdict(int)
    for g in scored:
        for k in g:
            scored_counts[int(k)] += 1
    L, E = geo.layers, geo.experts
    for cap in capacities:
        print(f'\n=== capacity {cap} slots ({cap / (L * E):.1%} of {L * E} experts)')
        print(f'{"policy":66s} {"hit":>6s} {"miss/tok":>9s} {"promo/tok":>9s}')
        policies = [LRU(cap), LFRU(cap), LFRU(cap, ENGINE_HALVING), LRUPerLayer(cap, geo), SLRU(cap), ARC(cap), LRU2(cap),
                    S3FIFO(cap), SIEVE(cap), DecayedLFU(cap, geo), WTinyLFU(cap, geo), StrataExchange(cap, geo, warm_counts),
                    LazyLRFU(cap, geo, warm_counts, 128, 0.5, 32), StaticHindsight(cap, scored_counts),
                    Belady(cap, nxt, False), Belady(cap, nxt, True)]
        for p in policies:
            for i, g in enumerate(groups):
                p.step_group(g, 1 if i >= len(warm) else 0)
            print(f'{p.name:66s} {p.hits / (p.hits + p.miss):6.3f} {p.miss / tokens:9.1f} {p.promo / tokens:9.1f}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('traces', nargs='+', help='a FreeToken trace directory, or engine route-trace files')
    ap.add_argument('--capacities', default=None,
                    help='frame counts (default: 4008,6347,8000,10200,12000 for FreeToken traces; '
                         'the trace frames x 0.5..1.5 for route traces)')
    ap.add_argument('--window', type=int, default=1, help='FreeToken traces: tokens per group')
    ap.add_argument('--experts', type=int, default=FLASH_NEXT.experts, help='FreeToken traces: experts per layer')
    ap.add_argument('--top-k', type=int, default=FLASH_NEXT.top_k, help='FreeToken traces: routed experts per token')
    ap.add_argument('--warmup-tokens', type=int, default=0, help='route traces: decode tokens not scored')
    ap.add_argument('--headroom', action='store_true',
                    help='route traces: the engine policy against Belady MIN with and without bypass')
    ap.add_argument('--decision-frames', type=int, default=0, help='headroom: frames of the decision (default: the trace\'s)')
    ap.add_argument('--layer-groups', default='', help='headroom: layer ranges, e.g. 0-19,20-47 (default: quarters)')
    ap.add_argument('--promotion-interval', type=int, default=0,
                    help='headroom: decode tokens per promotion once full (default: from the recorded budgets)')
    ap.add_argument('--json', default=None, help='headroom: write the results here')
    ap.add_argument('--protection', action='store_true',
                    help='headroom: add the eviction-protection arms (candidates §13 PUB-22)')
    ap.add_argument('--protect-horizons', default='1,8', help='protection: the oracle arms\' horizons in rounds')
    ap.add_argument('--protect-per-layer', type=int, default=0,
                    help='protection: experts the predictor protects per layer and lane (default: 2 x top-k; '
                         'INFERNIX_TMP_ROUTE_PROTECT\'s value)')
    args = ap.parse_args(argv)

    if all(is_route_trace(p) for p in args.traces):
        if args.window != 1:
            raise SystemExit('--window applies to FreeToken traces; route traces record their verify rounds')
        if args.headroom:
            return headroom(args.traces, args)
        geo, traces = read_route_traces(args.traces)
        groups, flags, tokens, warm_tokens = decode_groups(traces, geo, args.warmup_tokens)
        first = flags.index(True) if True in flags else len(groups)
        warm, scored = groups[:first], groups[first:]
        if not scored:
            raise SystemExit('no scored decode tokens')
        frames = traces[0].header['frames']
        caps = ([int(x) for x in args.capacities.split(',')] if args.capacities
                else sorted({int(round(frames * f)) for f in (0.5, 0.75, 1.0, 1.25, 1.5)}))
        print(f'{geo.layers} layers x {geo.experts} experts, top-{geo.top_k}; scored decode tokens {tokens} '
              f'after {warm_tokens} warm-up tokens, mean distinct experts per layer per round '
              f'{np.mean([len(g) for g in scored]):.1f}')
        policy_table(geo, warm, scored, tokens, caps)
        return 0
    if len(args.traces) != 1 or not os.path.isdir(args.traces[0]):
        raise SystemExit('pass one FreeToken trace directory or engine route-trace files')
    if args.headroom:
        raise SystemExit('--headroom needs engine route traces (the engine policy replays their prefill rounds)')
    geo, seqs = load(args.traces[0], args.experts, args.top_k)
    warm = to_groups(seqs[0], args.window)
    scored = to_groups(seqs[1], args.window) + to_groups(seqs[2], args.window)
    tokens = (len(seqs[1]) // args.window + len(seqs[2]) // args.window) * args.window
    print(f'scored tokens {tokens}, window {args.window}, '
          f'mean distinct experts per layer per window {np.mean([len(g) for g in scored]):.1f}')
    policy_table(geo, warm, scored, tokens,
                 [int(x) for x in (args.capacities or '4008,6347,8000,10200,12000').split(',')])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
