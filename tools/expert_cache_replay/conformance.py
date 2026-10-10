"""Write the LFRU conformance fixture that the engine's expert-cache policy test replays.

    python3 -m tools.expert_cache_replay.conformance tests/fixtures/expert_cache/lfru_conformance.txt

The trace is synthetic (Zipf popularity with bursts of locality, so scores tie and drift) and
fully determined by the integer generator below. Each line is ``G <keys...> | <victims...>``:
one routed group and the residents the replay tool's LFRU evicts for it, in eviction order. The
LFRU halves its counts every HALVING ticks (12 halvings over the trace), as the engine's does.
"""

from __future__ import annotations

import sys

from .replay import LFRU

KEYS, CAPACITY, GROUPS, GROUP, HALVING = 6 * 64, 96, 1200, 10, 100


def _rng(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield state >> 33


def groups():
    r = _rng(2026)
    hot = [next(r) % KEYS for _ in range(40)]
    for i in range(GROUPS):
        if i % 150 == 0:  # a topic switch moves the hot set
            hot = [next(r) % KEYS for _ in range(40)]
        layer = i % 6
        g = []
        while len(g) < GROUP:
            x = next(r)
            key = hot[x % len(hot)] if x % 4 else (x >> 3) % KEYS
            key = layer * 64 + key % 64
            if key not in g:
                g.append(key)
        yield g


def main(path):
    sim = LFRU(CAPACITY, HALVING)
    sim.trace = []
    sim.run(list(groups()), True)
    with open(path, "w") as out:
        out.write(f"# keys {KEYS} capacity {CAPACITY} halving {HALVING} hits {sim.hits} misses {sim.miss}\n")
        for g, v in sim.trace:
            out.write("G " + " ".join(map(str, g)) + " | " + " ".join(map(str, v)) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
