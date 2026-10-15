"""Build a shipped expert routing profile (docs/maintainer/artifact-container.md §9.3)
from recorded session data, for ``python -m tools.convert --expert-profile``.

The converter's own calibration measures activations, not routing, so the profile's frequencies
come from route traces or from an engine-saved expert state:

- from route traces: each trace's live routed uses per expert (``host_tier.use_counts``, the same
  oracle the replay priors rank with) become the profile's counts; the ranking orders every expert
  by descending use, ties layer-interleaved exactly like the engine's own fill order
  (``program_impl.h::tier_fill_order``), so an all-zero tail degrades to today's spread.
- from a saved state (``--expert-state``): the runtime-learned counts and ranking of one artifact
  are re-shipped into the pack; the sidecar's artifact identity is dropped, since the pack binds
  its own profile.

Usage (from the repository root, over the bench/route_traces manifest's recorded sessions):

    python3.11 -m tools.expert_cache_replay.profile TRACE [TRACE ...] -o OUT
    python3.11 -m tools.expert_cache_replay.profile --expert-state ARTIFACT.expert-state -o OUT
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from tools.artifact.expert_profile import U32_MAX, encode
from tools.expert_cache_replay import host_tier

STATE_MAGIC = b"NINFQ4ES"  # the engine's saved-state file (expert_state.cpp)
STATE_VERSION = 1


def _ranking(counts: np.ndarray, layers: int, experts: int) -> list[int]:
    """Every key, hottest first; ties and zero-count tails keep the layer-interleaved spread."""
    keys = layers * experts
    uses = counts[:keys].astype(np.int64)  # negation must not wrap an unsigned dtype
    spread = (np.arange(keys) % experts) * layers + np.arange(keys) // experts
    order = np.lexsort((spread, -uses))
    return [int(key) for key in order]


def profile_from_traces(paths: list[str | Path]) -> bytes:
    """The profile over one continuous session made of the given route traces."""
    traces = [host_tier.read_trace(path) for path in paths]
    if not traces:
        raise ValueError("at least one route trace is required")
    header = traces[0].header
    shape = (header["layers"], header["experts"])
    for trace in traces[1:]:
        if (trace.header["layers"], trace.header["experts"]) != shape:
            raise ValueError("route traces describe different models")
    layers, experts = shape
    keys = layers * experts
    counts = host_tier.use_counts(traces, keys, experts)
    capped = np.minimum(counts, U32_MAX).astype(np.uint32)
    return encode(capped.tolist(), _ranking(capped, layers, experts))


def profile_from_expert_state(path: str | Path) -> bytes:
    """The runtime-learned counts and ranking of a saved-state sidecar, as a shipped profile."""
    data = Path(path).read_bytes()
    try:
        return _expert_state_body(data)
    except struct.error as error:
        raise ValueError(f"{path}: the expert state file is truncated ({error})") from error


def _expert_state_body(data: bytes) -> bytes:
    magic, version = struct.unpack_from("<8sI", data, 0)
    if magic != STATE_MAGIC or version != STATE_VERSION:
        raise ValueError("not a version 1 expert state file")
    (identity_bytes,) = struct.unpack_from("<I", data, 12)
    at = 16 + identity_bytes
    (keys,) = struct.unpack_from("<I", data, at)
    counts = list(struct.unpack_from(f"<{keys}I", data, at + 4))
    (ranked_count,) = struct.unpack_from("<I", data, at + 4 + 4 * keys)
    ranked = list(struct.unpack_from(f"<{ranked_count}I", data, at + 8 + 4 * keys))
    footer = data[at + 8 + 4 * keys + 4 * ranked_count :][:8]
    if footer != STATE_MAGIC or len(data) != at + 16 + 4 * keys + 4 * ranked_count:
        raise ValueError("the expert state file is damaged")
    return encode(counts, ranked)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="*", metavar="TRACE", help="engine route traces (NRTRACE)")
    parser.add_argument("--expert-state", metavar="FILE",
                        help="re-ship an engine saved-state file instead of measuring traces")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if (args.expert_state is None) == (not args.traces):
        parser.error("give route traces or --expert-state, one of them")
    profile = (profile_from_expert_state(args.expert_state) if args.expert_state is not None
               else profile_from_traces(args.traces))
    args.out.write_bytes(profile)
    print(f"wrote {args.out}: {len(profile)} bytes")


if __name__ == "__main__":
    main()
