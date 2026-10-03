"""Trace-driven replay of VRAM expert-cache policies for Qwen3.8-Flash-Next decode routing.

Supporting evidence for docs/maintainer/qwen3_8-flash-next-design.md (section 9.4); the starting
point of milestone M1. Python 3.11 + numpy.

Input: a directory with warmup-trace.npz, a-trace.npz and b-trace.npz, each holding `ids`
[steps, 48, >=10] int32 with the layer's top-10 expert ids in the first 10 columns, as published in
https://github.com/hz1ulqu01gmnZH4/qwen38-freetoken
(experiments/2026-09-22_tg-expert-cache-trace-replay/data).

The traces are replayed as one continuous session: `warmup` warms every policy, A and B are scored.
Accesses are event-atomic: the experts of the current (round, layer) group must be resident
together, so the current group is never evicted. With --window W, a group is the per-layer union of
W consecutive tokens, approximating a fully accepted W-token speculative verify window.

Policies admit every miss ("promotion") unless marked CPU-served, in which case misses are computed
by the CPU and the policy promotes lazily. Output: hit rate, misses and promotions per token.

    python3 -m tools.expert_cache_replay.replay DIR [--capacities 4008,10200] [--window 1]
"""
import argparse
import heapq
from collections import OrderedDict, defaultdict

import numpy as np

L, E, K = 48, 512, 10


def load(directory):
    """Returns [steps, L, K] global expert keys (layer * E + expert) for warmup, a, b."""
    seqs = []
    for n in ('warmup', 'a', 'b'):
        ids = np.load(f'{directory}/{n}-trace.npz')['ids'][:, :, :K].astype(np.int64)
        keys = ids + (np.arange(L)[None, :, None] * E)
        seqs.append(keys)
    return seqs


class Sim:
    """Base: groups of K keys per (step, layer); the current group is never evicted."""
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

    def __init__(self, cap):
        super().__init__(cap)
        self.c = [OrderedDict() for _ in range(L)]
        self.q = cap // L

    def step_group(self, g, score):
        c = self.c[int(g[0]) // E]
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

    def __init__(self, cap, half=64):
        super().__init__(cap)
        self.res = set(); self.cnt = defaultdict(float); self.last = {}; self.t = 0; self.half = half
        self.h = []; self.hq = heapq; self.step = 0
        self.name = f'decayed global LFU (half-life {half} tokens)'

    def step_group(self, g, score):
        self.t += 1
        if int(g[0]) // E == 0: self.step += 1
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

    def __init__(self, cap, win=0.01, age=2048):
        super().__init__(cap)
        self.wcap = max(K + 1, int(cap * win)); self.w = OrderedDict()
        self.main = SLRU(cap - self.wcap); self.f = defaultdict(int); self.n = 0; self.age = age * L

    def _in_main(self, k): return k in self.main.p or k in self.main.m

    def step_group(self, g, score):
        gs = set(g)
        for k in g:
            self.f[k] += 1; self.n += 1
        if self.n >= self.age * K:
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

    def __init__(self, cap, seed_counts):
        super().__init__(cap)
        order = sorted(range(L * E), key=lambda k: -seed_counts.get(k, 0))
        self.res = set(order[:cap]); self.cnt = defaultdict(float); self.step = 0

    def step_group(self, g, score):
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

    def __init__(self, cap, seed_counts, half=128, margin=1.0, budget=16):
        super().__init__(cap)
        order = sorted(range(L * E), key=lambda k: -seed_counts.get(k, 0))
        self.res = set(order[:cap]); self.s = defaultdict(float); self.step = 0
        self.half, self.margin, self.budget = half, margin, budget
        self.touched = set(); self.name = f'lazy LRFU hl{half} margin{margin} budget{budget}/round'

    def step_group(self, g, score):
        inc = 2.0 ** (self.step / self.half)
        for k in g:
            self.s[k] += inc; self.touched.add(int(k))
            if k in self.res: self.hits += score
            else: self.miss += score
        if int(g[0]) // E == L - 1:
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


class LFRU(Sim):
    """Zhang (arXiv 2608.07911): score = f / (clock - last + 1), f a global count never reset on eviction;
    evict the lowest score among non-current residents; admit every miss. Clock in layer-call ticks."""
    name = 'LFRU f/(age+1)'

    def __init__(self, cap):
        super().__init__(cap)
        self.res = set(); self.f = defaultdict(int); self.last = {}; self.t = 0
        self.trace = None   # set to a list to record (group, victims) for conformance fixtures

    def step_group(self, g, score):
        self.t += 1; gs = set(g); miss = []
        for k in g:
            self.f[k] += 1; self.last[k] = self.t
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
        for layer in range(L):
            groups.append(np.array(list(dict.fromkeys(int(x) for x in seq[s:s + window, layer].reshape(-1))),
                                   dtype=np.int64))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('traces')
    ap.add_argument('--capacities', default='4008,6347,8000,10200,12000')
    ap.add_argument('--window', type=int, default=1)
    args = ap.parse_args()
    seqs = load(args.traces)
    warm = to_groups(seqs[0], args.window)
    scored = to_groups(seqs[1], args.window) + to_groups(seqs[2], args.window)
    groups = warm + scored
    nxt = next_uses(groups)
    tokens = (len(seqs[1]) // args.window + len(seqs[2]) // args.window) * args.window
    warm_counts = defaultdict(int)
    for g in warm:
        for k in g:
            warm_counts[int(k)] += 1
    scored_counts = defaultdict(int)
    for g in scored:
        for k in g:
            scored_counts[int(k)] += 1
    print(f'scored tokens {tokens}, window {args.window}, '
          f'mean distinct experts per layer per window {np.mean([len(g) for g in scored]):.1f}')
    for cap in (int(x) for x in args.capacities.split(',')):
        print(f'\n=== capacity {cap} slots ({cap / (L * E):.1%} of {L * E} experts)')
        print(f'{"policy":66s} {"hit":>6s} {"miss/tok":>9s} {"promo/tok":>9s}')
        policies = [LRU(cap), LFRU(cap), LRUPerLayer(cap), SLRU(cap), ARC(cap), LRU2(cap), S3FIFO(cap),
                    SIEVE(cap), DecayedLFU(cap), WTinyLFU(cap), StrataExchange(cap, warm_counts),
                    LazyLRFU(cap, warm_counts, 128, 0.5, 32), StaticHindsight(cap, scored_counts),
                    Belady(cap, nxt, False), Belady(cap, nxt, True)]
        for p in policies:
            for i, g in enumerate(groups):
                p.step_group(g, 1 if i >= len(warm) else 0)
            print(f'{p.name:66s} {p.hits / (p.hits + p.miss):6.3f} {p.miss / tokens:9.1f} {p.promo / tokens:9.1f}')


if __name__ == '__main__':
    main()
