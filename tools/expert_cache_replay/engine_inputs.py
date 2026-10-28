"""What the expert-cache replays read from the engine besides its route trace (replay.py, host_tier.py):
a saved expert state, Qwen3.8-Flash-Next's calibrated expert order, a GGUF import's pool layout, and
the expert-cache lines of an engine log. Python 3.11 + numpy.

- Saved expert state (--expert-state FILE; src/runtime/expert_offload/expert_state.cpp, version 1):
  little-endian magic "NINFQ4ES", version, identity length and bytes, key count, every key's LFRU
  count (rounded), the ranked count and the resident keys highest score first, the magic again.
- Calibrated order (src/models/qwen4_exp/program/calibrated_order.cpp): a C++ table compiled into the
  engine, read here from the literal in its source text, and turned into a saved state as
  calibrated_expert_order() does there.
- Pools (ExpertResidency: one pool of frames, one LFRU and one halving period per record stride). An
  NVFP4 or block-FP8 artifact has one; a GGUF import's layers have up to seven strides. The trace
  header does not record them, so a JSON file gives them: {"pools": [[layers...], ...]},
  {"pools": [{"stride": BYTES, "layers": [...]}, ...]} or {"pool_of_layer": [pool of layer 0, ...]}
  (layers with the same record stride share a pool).
- Engine log (a server or bench log with Debug diagnostics): each finished request's "expert cache:"
  line (Program::report_cache, ExpertOffload::request_report), the CPU team's "decode misses kept on
  the link" split and the warm start's source.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]

STATE_MAGIC = b'NINFQ4ES'
STATE_VERSION = 1


@dataclass
class ExpertState:
    identity: str        # the artifact the state belongs to (the engine ignores a state of another)
    counts: np.ndarray   # uint32 [layers * experts]: every key's LFRU count, rounded
    ranked: np.ndarray   # uint32: the resident keys, highest score first


def read_expert_state(path):
    """Reads a saved expert state (load_expert_state's checks, without the identity: the replay does not
    know the artifact's)."""
    data = Path(path).read_bytes()
    offset = 0

    def fail(why):
        raise ValueError(f'{path}: {why}')

    def word():
        nonlocal offset
        if offset + 4 > len(data):
            fail('truncated expert state')
        (value,) = struct.unpack_from('<I', data, offset)
        offset += 4
        return value

    if data[:8] != STATE_MAGIC:
        fail('not an expert state file')
    offset = 8
    if (version := word()) != STATE_VERSION:
        fail(f'expert state version {version}, not {STATE_VERSION}')
    identity_bytes = word()
    if identity_bytes > 1 << 16 or offset + identity_bytes > len(data):
        fail('damaged expert state identity')
    identity = data[offset:offset + identity_bytes].decode('utf-8', 'replace')
    offset += identity_bytes
    keys = word()
    if offset + 4 * keys > len(data):
        fail('truncated expert state counts')
    counts = np.frombuffer(data, '<u4', keys, offset).astype(np.uint32)
    offset += 4 * keys
    ranked_count = word()
    if ranked_count > keys or offset + 4 * ranked_count + len(STATE_MAGIC) > len(data):
        fail('damaged expert state ranking')
    ranked = np.frombuffer(data, '<u4', ranked_count, offset).astype(np.uint32)
    offset += 4 * ranked_count
    if data[offset:offset + len(STATE_MAGIC)] != STATE_MAGIC:
        fail('truncated expert state (no footer)')
    if ranked.size and int(ranked.max()) >= keys:
        fail('an expert state key is out of range')
    return ExpertState(identity, counts, ranked)


CALIBRATED_ORDER_SOURCE = REPOSITORY / 'src' / 'models' / 'qwen4_exp' / 'program' / 'calibrated_order.cpp'
CALIBRATED_SHAPE = (48, 512, 10)  # kCalibratedLayers, kCalibratedExperts and Flash-Next's top-k (has_calibrated_order)


def calibrated_expert_order(path=CALIBRATED_ORDER_SOURCE):
    """Flash-Next's calibrated expert order as a saved state (calibrated_expert_order): a key's count is its
    rank reversed (512 for a layer's first expert), and the ranking takes every layer's first expert, then
    every layer's second, and so on. Returns (counts, ranked); refuses a table that is not 48 rows of
    permutations of the 512 experts."""
    text = Path(path).read_text(encoding='utf-8')
    opening = 'kCalibratedOrder{{'
    start = text.find(opening)
    end = text.find('}};', start)
    if start < 0 or end < 0:
        raise ValueError(f'{path}: no kCalibratedOrder table')
    body = re.sub(r'//[^\n]*', '', text[start + len(opening):end])
    values = np.array([int(x) for x in re.findall(r'\d+', body)], np.int64)
    layers, experts, _ = CALIBRATED_SHAPE
    if values.size != layers * experts:
        raise ValueError(f'{path}: the calibrated order has {values.size} entries, not {layers} x {experts}')
    order = values.reshape(layers, experts)
    if not (np.sort(order, axis=1) == np.arange(experts)).all():
        raise ValueError(f'{path}: a calibrated order row is not a permutation of the experts')
    keys = order + (np.arange(layers) * experts)[:, None]
    counts = np.zeros(layers * experts, np.uint32)
    counts[keys.ravel()] = np.tile(np.arange(experts, 0, -1), layers)
    return counts, keys.T.ravel().astype(np.uint32)


def read_pools(path, layers):
    """The expert pools of `layers` layers from a JSON file (module docstring): each pool's layers,
    ascending, in increasing record stride when strides are given (the engine's order)."""
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'pool_of_layer' in spec:
        of = spec['pool_of_layer']
        items = enumerate(of) if isinstance(of, list) else ((int(k), v) for k, v in of.items())
        by_pool = {}
        for layer, pool in items:
            by_pool.setdefault(pool, []).append(int(layer))
        pools = [sorted(by_pool[p]) for p in sorted(by_pool)]
    elif 'pools' in spec:
        entries = spec['pools']
        if all(isinstance(e, dict) for e in entries):
            entries = sorted(entries, key=lambda e: e.get('stride', 0))
            pools = [sorted(int(x) for x in e['layers']) for e in entries]
        else:
            pools = [sorted(int(x) for x in e) for e in entries]
    else:
        raise ValueError(f'{path}: expected "pools" or "pool_of_layer"')
    if any(not p for p in pools) or sorted(x for p in pools for x in p) != list(range(layers)):
        raise ValueError(f'{path}: the pools must hold each of the {layers} layers exactly once')
    return pools


# Program::report_cache and ExpertOffload::request_report, one per finished request (Debug).
CACHE_LINE = re.compile(r'expert cache: (\d+) frames; request ([0-9.]+)% of (\d+) routed experts hit, (\d+) promotions, '
                        r'(\d+) landed, (\d+) misses CPU-served; since start ([0-9.]+)%')
# The CPU team's rates line: decode misses kept on the link, M / d or (M x share) >> 8.
LINK_LINE = re.compile(r'decode misses kept on the link: (?:(\d+)/256 \(INFERNIX_TMP_LINK_SHARE|1/(\d+))')
# The warm start (warm_start_frames): a saved state, the calibrated order, or none.
SEED_LINES = ((re.compile(r'expert cache: restored the (\d+) most-used experts'), 'saved'),
              (re.compile(r'expert cache: no saved state yet, so it starts with the (\d+) highest-ranked'), 'calibrated'),
              (re.compile(r'expert cache: (?:--expert-state off|no saved state yet), so it starts empty'), 'none'))


@dataclass
class CacheReport:
    """One finished request's expert-cache line: the residency's counters between its admission and its
    finish (routed: every round's distinct live experts per layer; hits: those in a published frame when
    the round started) and the hit rate since the engine started."""
    frames: int
    hit_percent: float
    routed: int
    promotions: int
    landed: int
    cpu_served: int
    since_start_percent: float

    @property
    def hits(self):
        """The request's hits, from its rate as printed (0.1 % resolution)."""
        return self.hit_percent / 100.0 * self.routed


@dataclass
class EngineLog:
    requests: list = field(default_factory=list)  # CacheReport per finished request, in log order
    link_divisor: int = 0  # decode misses kept on the link: M / d (0: not logged)
    link_share: int = 0    # or (M x share) >> 8 (INFERNIX_TMP_LINK_SHARE; 0: not set)
    seed: str | None = None  # the warm start: 'saved', 'calibrated', 'none' (None: not logged)
    seeded: int = 0          # experts it loaded


def read_engine_log(path):
    """The expert-cache lines of an engine log. Every session logs its warm start once at startup, so the
    requests of a log holding several sessions are the last session's."""
    log = EngineLog()
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if m := CACHE_LINE.search(line):
                frames, rate, routed, promotions, landed, cpu, since = m.groups()
                log.requests.append(CacheReport(int(frames), float(rate), int(routed), int(promotions), int(landed),
                                                int(cpu), float(since)))
                continue
            if m := LINK_LINE.search(line):
                share, divisor = m.groups()
                log.link_share, log.link_divisor = (int(share), 0) if share else (0, int(divisor))
                continue
            for pattern, source in SEED_LINES:
                if m := pattern.search(line):
                    log.seed, log.seeded = source, int(m.group(1)) if m.groups() else 0
                    log.requests = []  # a new session
                    break
    return log
