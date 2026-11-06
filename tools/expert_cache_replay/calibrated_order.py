"""A calibrated expert order from route traces: each layer's routed experts ranked by how often a mixed task set
routes to them, the expert cache's seed when no saved state exists (candidates §23 NU-43; the Qwen3.5-122B-A10B's
table, src/models/qwen3_5/program/calibrated_order.inc, read by calibrated_order.h). Python 3.11 + numpy.

The method is the one behind Qwen3.8-Flash-Next's built-in order (NInfer c996baa8 / f8a774f73, design §19.3.24):

- Input: route traces (src/runtime/expert_offload/route_trace.h; infernix-serve --route-trace) of a mixed task set,
  one or more per speculative arm: plain decoding (--plain), MTP (--mtp) and DFlash2 (--dflash). collect_traces.py
  writes them.
- Requests: a lane's rounds from a prefill chunk at position 0, or the first prefill chunk after its decoding, up to
  its next request. Within a request, each layer's routed experts in the rounds of at most --max-columns columns
  (decode and verification rounds, forced tokens and narrow prefill chunks: the calls the expert cache serves; every
  physical column counts, rejected drafts included, since each loads its experts) are counted and normalized to a
  frequency (uses / the request's uses in that layer), so a long request weighs as much as a short one.
- An arm's frequency is the mean over its requests; the arms are mixed by --weights (plain / MTP / DFlash2,
  0.50 / 0.25 / 0.25 by default, renormalized over the arms given). Each layer ranks its experts by that score,
  ties to the lower expert id.

Output: the C++ table (--out; its layer, expert and top-k counts, and one row per layer, highest priority first) with
the provenance in its header comment, and optionally (--json) the ranks and scores. read_table() reads a table back
(replay.py --seed calibrated uses it for the 122B's geometry).

Usage (from the repository root):

    py -3.11 -m tools.expert_cache_replay.calibrated_order --plain plain.bin --mtp mtp.bin --dflash dflash2.bin \\
        --out src/models/qwen3_5/program/calibrated_order.inc [--max-columns 64] [--weights 0.5,0.25,0.25] \\
        [--expect 48,256,8] [--json order.json]
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import host_tier as ht

ARMS = ('plain', 'mtp', 'dflash')
DEFAULT_WEIGHTS = (0.50, 0.25, 0.25)  # NInfer c996baa8: ordinary, MTP3 and DFlash7 at 50 / 25 / 25 %
DEFAULT_MAX_COLUMNS = 64              # NInfer's "main physical columns <= 64"
REPOSITORY = Path(__file__).resolve().parents[2]
TABLE_122B = REPOSITORY / 'src' / 'models' / 'qwen3_5' / 'program' / 'calibrated_order.inc'


@dataclass
class ArmCounts:
    """One arm's per-request frequencies, summed (divide by `requests` for the mean)."""
    frequency: np.ndarray        # float64 [layers, experts]: the sum over requests of each request's frequencies
    requests: int = 0            # requests that routed at least one counted column
    skipped: int = 0             # requests with no counted column (none of their rounds was narrow enough)
    rounds: int = 0              # counted rounds
    columns: int = 0             # counted columns
    traces: list = field(default_factory=list)


def read_header(path):
    """A route trace's header fields (host_tier.HEADER_FIELDS)."""
    with open(path, 'rb') as f:
        data = f.read(64)
    if len(data) < 64:
        raise ValueError(f'{path}: too short for a route trace header')
    magic, *words = struct.unpack_from('<8s14I', data, 0)
    if magic != b'NRTRACE1' or words[0] != ht.TRACE_VERSION:
        raise ValueError(f'{path}: not a version {ht.TRACE_VERSION} route trace')
    return dict(zip(ht.HEADER_FIELDS, words[:len(ht.HEADER_FIELDS)]))


def iter_rounds(path, max_columns):
    """The rounds of a route trace in order, as (kind, rows, width, position, lanes, routes): routes int32
    [layers, columns, top_k] for a round of at most `max_columns` columns, None for a wider one (its routes are
    skipped unread, so a trace of long prompts is not held in memory). Graph records are skipped."""
    header = read_header(path)
    layers, top_k = header['layers'], header['top_k']
    with open(path, 'rb') as f:
        f.seek(64)
        offset = 64
        while True:
            head = f.read(32)
            if not head:
                return
            if len(head) < 32:
                raise ValueError(f'{path}: truncated record at byte {offset}')
            kind, columns, rows, width, _tokens, _budget, position, _frames = struct.unpack('<8I', head)
            offset += 32
            if kind == ht.GRAPH:
                continue
            if kind > ht.VERIFY or columns == 0 or rows * width != columns:
                raise ValueError(f'{path}: malformed round at byte {offset - 32}')
            lanes = np.frombuffer(f.read(4 * rows), np.int32)
            f.seek((columns + 3) // 4 * 4, 1)  # the live mask: every physical column counts here
            routes_bytes = 4 * layers * top_k * columns
            if columns > max_columns:
                f.seek(routes_bytes, 1)
                routes = None
            else:
                raw = f.read(routes_bytes)
                if len(raw) < routes_bytes or lanes.size < rows:
                    raise ValueError(f'{path}: truncated round at byte {offset - 32}')
                routes = np.frombuffer(raw, np.int32).reshape(layers, columns, top_k)
            offset += 4 * rows + (columns + 3) // 4 * 4 + routes_bytes
            yield kind, rows, width, position, [int(x) for x in lanes], routes


def request_counts(path, max_columns):
    """Splits a trace into requests and counts each one's routed experts per layer in its rounds of at most
    `max_columns` columns. A request is a lane's rounds from a prefill chunk at position 0, or the first prefill
    chunk after the lane decoded, to its next request. Returns (list of int64 [layers, experts] arrays, counted
    rounds, counted columns)."""
    header = read_header(path)
    layers, experts = header['layers'], header['experts']
    offsets = (np.arange(layers, dtype=np.int64) * experts)[:, None]
    current = {}      # lane -> index into `keys`
    decoded = set()   # lanes whose current request has had a round other than a prefill chunk
    keys = []         # per request: the counted (layer, expert) keys of each counted row
    rounds = columns = 0
    for kind, rows, width, position, lanes, routes in iter_rounds(path, max_columns):
        if kind == ht.PREFILL:
            lane = lanes[0]
            if lane not in current or lane in decoded or position == 0:
                current[lane] = len(keys)
                keys.append([])
                decoded.discard(lane)
        else:
            decoded.update(lanes)
        if routes is None:
            continue
        rounds += 1
        for row, lane in enumerate(lanes):
            if lane not in current:  # a round on a lane whose prompt the trace does not hold
                current[lane] = len(keys)
                keys.append([])
            ids = routes[:, row * width:(row + 1) * width, :].reshape(layers, -1).astype(np.int64)
            if (ids < 0).any() or (ids >= experts).any():
                raise ValueError(f'{path}: a round routes an expert outside 0..{experts - 1}')
            keys[current[lane]].append((ids + offsets).ravel())
            columns += width
    counts = [np.bincount(np.concatenate(k), minlength=layers * experts).reshape(layers, experts) if k
              else np.zeros((layers, experts), np.int64) for k in keys]
    return counts, rounds, columns


def arm_counts(paths, max_columns, shape=None):
    """Reads an arm's traces and sums its requests' per-layer frequencies. `shape` (layers, experts, top_k) is
    checked when given and otherwise taken from the first trace."""
    arm = None
    for path in paths:
        h = read_header(path)
        here = (h['layers'], h['experts'], h['top_k'])
        if shape is None:
            shape = here
        elif here != tuple(shape):
            raise ValueError(f'{path}: {here[0]} layers x {here[1]} experts, top-{here[2]}; the other traces have '
                             f'{shape[0]} x {shape[1]}, top-{shape[2]}')
        if arm is None:
            arm = ArmCounts(np.zeros((shape[0], shape[1]), np.float64))
        counts, rounds, columns = request_counts(path, max_columns)
        arm.rounds += rounds
        arm.columns += columns
        for c in counts:
            totals = c.sum(axis=1, keepdims=True)
            if totals.sum() == 0:
                arm.skipped += 1
                continue
            arm.frequency += c / np.maximum(totals, 1)
            arm.requests += 1
        arm.traces.append(str(path))
    return arm, shape


def rank(score):
    """Each layer's experts by descending score, ties to the lower expert id."""
    layers, experts = score.shape
    ids = np.arange(experts)
    return np.stack([np.lexsort((ids, -score[layer])) for layer in range(layers)]).astype(np.int64)


def calibrated_order(arms, weights, max_columns=DEFAULT_MAX_COLUMNS, expect=None):
    """`arms`: {name: [trace paths]} (empty arms dropped); `weights`: {name: weight}. Returns (order [layers,
    experts], score [layers, experts], shape (layers, experts, top_k), per-arm ArmCounts, the weights used)."""
    shape = tuple(expect) if expect else None
    counted, used = {}, {}
    for name, paths in arms.items():
        if not paths:
            continue
        arm, shape = arm_counts(paths, max_columns, shape)
        if arm.requests == 0:
            raise ValueError(f'the {name} traces hold no request with a round of at most {max_columns} columns')
        counted[name], used[name] = arm, float(weights[name])
    if not counted:
        raise ValueError('no traces given')
    total = sum(used.values())
    if total <= 0:
        raise ValueError('the weights of the arms given sum to zero')
    used = {name: w / total for name, w in used.items()}
    score = sum(used[name] * arm.frequency / arm.requests for name, arm in counted.items())
    return rank(score), score, shape, counted, used


def format_table(order, shape, counted, weights, max_columns, generated=None):
    """The calibrated_order.inc text: its provenance, the table's counts and one row per layer."""
    layers, experts, top_k = shape
    generated = generated or datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [
        '// Generated by tools/expert_cache_replay/calibrated_order.py: the calibrated expert order of',
        '// models/qwen3_5/program/calibrated_order.h (do not edit by hand; regenerate it from route traces).',
        f'// {generated}; rounds of at most {max_columns} columns; each request\'s per-layer frequencies, the mean',
        '// over an arm\'s requests, the arms mixed by weight; each layer\'s experts by that score, ties to the lower id.',
    ]
    for name, arm in counted.items():
        names = ', '.join(Path(t).name for t in arm.traces)
        lines.append(f'// {name} {weights[name]:.3f}: {arm.requests} requests ({arm.skipped} without a counted round), '
                     f'{arm.rounds} rounds, {arm.columns} columns; {names}')
    lines += [
        f'constexpr std::uint32_t kTableLayers  = {layers};',
        f'constexpr std::uint32_t kTableExperts = {experts};',
        f'constexpr std::uint32_t kTableTopK    = {top_k};',
        'constexpr std::array<std::uint16_t, static_cast<std::size_t>(kTableLayers) * kTableExperts> kTable{{',
    ]
    for layer in range(layers):
        lines.append(f'    // layer {layer}')
        row = [int(x) for x in order[layer]]
        for begin in range(0, experts, 16):
            lines.append('    ' + ' '.join(f'{x:3d},' for x in row[begin:begin + 16]))
    lines.append('}};')
    return '\n'.join(lines) + '\n'


def read_table(path=TABLE_122B):
    """Reads a calibrated_order.inc table. Returns ((layers, experts, top_k), order int64 [layers, experts]) or
    None when the table is empty (no order generated); refuses rows that are not permutations of the experts."""
    text = Path(path).read_text(encoding='utf-8')

    def constant(name):
        found = re.search(rf'{name}\s*=\s*(\d+)\s*;', text)
        if not found:
            raise ValueError(f'{path}: no {name}')
        return int(found.group(1))

    layers, experts, top_k = constant('kTableLayers'), constant('kTableExperts'), constant('kTableTopK')
    if layers == 0:
        return None
    start = text.find('kTable{{')
    end = text.find('}};', start)
    if start < 0 or end < 0:
        raise ValueError(f'{path}: no kTable{{{{ ... }}}} literal')
    body = re.sub(r'//[^\n]*', '', text[start + len('kTable{{'):end])
    values = np.array([int(x) for x in re.findall(r'\d+', body)], np.int64)
    if values.size != layers * experts:
        raise ValueError(f'{path}: {values.size} entries, not {layers} x {experts}')
    order = values.reshape(layers, experts)
    if not (np.sort(order, axis=1) == np.arange(experts)).all():
        raise ValueError(f'{path}: a row is not a permutation of the experts')
    return (layers, experts, top_k), order


def order_state(order):
    """An order [layers, experts] as the warm start's saved state (expert_cache::calibrated_order_state): counts
    are the ranks reversed, the ranking takes every layer's next expert in turn. Returns (counts, ranked)."""
    layers, experts = order.shape
    keys = order + (np.arange(layers) * experts)[:, None]
    counts = np.zeros(layers * experts, np.uint32)
    counts[keys.ravel()] = np.tile(np.arange(experts, 0, -1), layers)
    return counts, keys.T.ravel().astype(np.uint32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plain', nargs='+', default=[], help='route traces of plain decoding')
    ap.add_argument('--mtp', nargs='+', default=[], help='route traces with MTP drafts')
    ap.add_argument('--dflash', nargs='+', default=[], help='route traces with DFlash2 (or DFlash) drafts')
    ap.add_argument('--weights', default=','.join(str(w) for w in DEFAULT_WEIGHTS),
                    help='plain,mtp,dflash weights (default 0.5,0.25,0.25; renormalized over the arms given)')
    ap.add_argument('--max-columns', type=int, default=DEFAULT_MAX_COLUMNS,
                    help='count rounds of at most this many columns (default 64)')
    ap.add_argument('--expect', default=None,
                    help='layers,experts,top_k the traces must have (the 122B: 48,256,8)')
    ap.add_argument('--out', required=True, help='the table to write (src/models/qwen3_5/program/calibrated_order.inc)')
    ap.add_argument('--json', default=None, help='also write the order and scores as JSON')
    args = ap.parse_args(argv)
    weights = dict(zip(ARMS, (float(x) for x in args.weights.split(','))))
    if len(weights) != len(ARMS) or any(w < 0 for w in weights.values()):
        raise SystemExit('--weights takes three nonnegative numbers: plain,mtp,dflash')
    expect = tuple(int(x) for x in args.expect.split(',')) if args.expect else None
    if expect is not None and len(expect) != 3:
        raise SystemExit('--expect takes layers,experts,top_k')
    try:
        order, score, shape, counted, used = calibrated_order(
            {'plain': args.plain, 'mtp': args.mtp, 'dflash': args.dflash}, weights, args.max_columns, expect)
    except ValueError as error:
        raise SystemExit(str(error))
    Path(args.out).write_text(format_table(order, shape, counted, used, args.max_columns), encoding='utf-8',
                              newline='\n')
    for name, arm in counted.items():
        print(f'{name}: weight {used[name]:.3f}, {arm.requests} requests, {arm.rounds} rounds, {arm.columns} columns')
    # How concentrated the use is: the share of a layer's uses its top quarter of experts takes (mean over layers).
    top = max(1, shape[1] // 4)
    share = np.mean([np.sort(score[layer])[::-1][:top].sum() / max(score[layer].sum(), 1e-30)
                     for layer in range(shape[0])])
    print(f'{shape[0]} layers x {shape[1]} experts, top-{shape[2]}: the top quarter of each layer\'s experts takes '
          f'{100 * share:.1f} % of its uses on average; wrote {args.out}')
    if args.json:
        Path(args.json).write_text(json.dumps({
            'layers': shape[0], 'experts': shape[1], 'top_k': shape[2], 'max_columns': args.max_columns,
            'weights': used, 'arms': {name: {'requests': arm.requests, 'rounds': arm.rounds, 'columns': arm.columns,
                                             'traces': arm.traces} for name, arm in counted.items()},
            'order': order.tolist(), 'score': score.tolist()}), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
