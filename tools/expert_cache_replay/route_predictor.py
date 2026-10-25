"""Routing-sequence predictors for eviction protection over the engine's LFRU (candidates §13 PUB-22;
SeqMoE, arXiv 2609.12978). Python 3.11 + numpy.

A predictor names, at each round's boundary, the experts it expects the next rounds to compute;
host_tier.TierSim then evicts them only when no unprotected resident is left (protection only: the
prefetch half of SeqMoE is the rejected prefetch class). Both are guards for TierSim.guard:

- NextRoundsOracle: the keys the next `horizon` decode or verification rounds actually compute. It
  is the ceiling of any protection predictor at that horizon: when protection with perfect
  knowledge gains too little, no learned predictor can gain more.
- TransitionPredictor: the engine's causal routing-sequence model (INFERNIX_TMP_ROUTE_PROTECT,
  src/runtime/expert_offload/route_predictor.h), operation for operation in FP32. Per layer,
  exponentially decayed counts of expert b at a lane's live token t + 1 after expert a at its token t
  (prompt tokens included; each row of a round contributes its last MAX_ROUND_TOKENS live tokens);
  at a boundary each lane of the round predicts, per layer, the `per_layer` experts b with the
  largest sum over its last token's experts a of counts[a, b] / out[a] (ties: the lower expert),
  positive scores only. It runs on the host beside the policy, so it costs no frames.
"""
import numpy as np

from . import host_tier as ht

MAX_ROUND_TOKENS = 256    # RoutePredictor::kMaxRoundTokens
RESCALE_ABOVE = 1e20      # the increment past which every count is rescaled (FP32 headroom)


def lane_columns(r):
    """(lane, live column indices) of each row of round `r`, in column order."""
    for row in range(r.rows):
        cols = np.arange(row * r.width, (row + 1) * r.width)
        yield int(r.lanes[row]), cols[r.live[cols]]


class NextRoundsOracle:
    """The keys the next `horizon` decode or verification rounds compute (every column, as read)."""

    def __init__(self, rounds, layers, experts, horizon):
        base = (np.arange(layers) * experts)[:, None]
        decode = [i for i, r in enumerate(rounds) if r.kind in (ht.DECODE, ht.VERIFY)]

        def computed(r):
            ids = r.routes.reshape(layers, -1)
            return np.unique((ids + base)[(ids >= 0) & (ids < experts)])

        keys = {i: computed(rounds[i]) for i in decode}
        self.guards = [None] * len(rounds)
        at = 0
        for i in range(len(rounds)):
            while at < len(decode) and decode[at] <= i:
                at += 1
            ahead = decode[at:at + horizon]
            if ahead:
                self.guards[i] = np.unique(np.concatenate([keys[j] for j in ahead]))
        self.name = f'next {horizon} round{"s" if horizon > 1 else ""} (oracle)'

    def __call__(self, index, r):
        return self.guards[index]


class TransitionPredictor:
    """Decayed per-layer expert transition counts between a lane's consecutive live tokens."""

    def __init__(self, layers, experts, per_layer, half_life=4096.0):
        self.L, self.E, self.P = layers, experts, min(int(per_layer), experts)
        self.counts = np.zeros((layers, experts, experts), np.float32)
        self.out = np.zeros((layers, experts), np.float32)  # row sums of counts
        # Increments grow by 2^(1 / half_life) per token instead of decaying every count.
        self.growth = 2.0 ** (1.0 / half_life)
        self.weight = 1.0
        self.last = {}  # lane -> [layers, top_k] experts of its last live token
        self.lanes = []  # the lanes of the last observed round
        self.name = f'transition predictor, {self.P} per layer'

    def _add(self, prev, cur, weights):
        """Adds one weight per token pair (prev[:, i] -> cur[:, i]) in order, as the engine does."""
        k = cur.shape[2]
        a = np.repeat(prev, k, axis=2)        # [L, pairs, k * k]
        b = np.tile(cur, (1, 1, k))           # [L, pairs, k * k]
        w = np.broadcast_to(np.asarray(weights, np.float32)[None, :, None], a.shape)
        for layer in range(self.L):
            al, bl, wl = a[layer].ravel(), b[layer].ravel(), w[layer].ravel()
            valid = (al >= 0) & (al < self.E) & (bl >= 0) & (bl < self.E)
            if not valid.all():
                al, bl, wl = al[valid], bl[valid], wl[valid]
            np.add.at(self.counts[layer], (al, bl), wl)
            np.add.at(self.out[layer], al, wl)

    def observe(self, r):
        self.lanes = [int(x) for x in r.lanes[:r.rows]]
        for lane, cols in lane_columns(r):
            cols = cols[-MAX_ROUND_TOKENS:]
            if cols.size == 0:
                continue
            tokens = r.routes[:, cols, :]  # [layers, tokens, top_k]
            prev = self.last.get(lane)
            seq = tokens if prev is None else np.concatenate([prev[:, None, :], tokens], axis=1)
            start = 0
            weights = []
            for i in range(seq.shape[1] - 1):
                weights.append(np.float32(self.weight))
                self.weight *= self.growth
                if self.weight > RESCALE_ABOVE:
                    self._add(seq[:, start:i + 1], seq[:, start + 1:i + 2], weights)
                    scale = np.float32(1.0 / self.weight)
                    self.counts *= scale
                    self.out *= scale
                    self.weight = 1.0
                    start, weights = i + 1, []
            if weights:
                self._add(seq[:, start:-1], seq[:, start + 1:], weights)
            self.last[lane] = tokens[:, -1, :]

    def predict(self, lanes=None):
        out = []
        layer_base = np.arange(self.L) * self.E
        experts = np.arange(self.E)
        for lane in (self.lanes if lanes is None else lanes):
            prev = self.last.get(int(lane))
            if prev is None:
                continue
            for layer in range(self.L):
                score = np.zeros(self.E, np.float32)
                for a in prev[layer]:
                    if 0 <= a < self.E and self.out[layer, a] > 0:
                        score += self.counts[layer, a] / self.out[layer, a]
                top = np.lexsort((experts, -score))[:self.P]
                top = top[score[top] > 0]
                out.append(top + layer_base[layer])
        return np.unique(np.concatenate(out)) if out else None

    def __call__(self, index, r):
        self.observe(r)
        return self.predict()
