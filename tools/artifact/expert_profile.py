"""The shipped expert routing profile (docs/maintainer/artifact-container.md §9.3).

One resource object inside the pack, holding a converter- or calibration-produced frequency ranking
that the loader uses to seed the expert cache's warm start on a cold first run. The bytes mirror the
engine's saved-state body (src/models/qwen4_exp/program/expert_cache/expert_state.cpp) without the
identity: the profile rides inside the artifact, which is its own binding.

Layout, little-endian::

    magic   8 bytes  b"NINFQ4EP"
    version u32      1
    keys    u32      layers * experts
    counts  u32[keys]     recorded live routing uses per key (key = layer * experts + expert)
    ranked  u32           number of ranked keys (0..keys)
    ranked  u32[ranked]   the keys, hottest first

A ranking prefix is enough: the warm start appends the keys it does not rank in its own layer-
interleaved order. The writer validates every bound so a pack never carries a profile the loader
would refuse.
"""

from __future__ import annotations

import struct
from typing import Sequence

MAGIC = b"NINFQ4EP"
VERSION = 1
HEADER = struct.Struct("<8sII")  # magic, version, keys
U32 = struct.Struct("<I")
U32_MAX = 0xFFFFFFFF

ROLE = "expert_profile"  # the text component's resource role
OBJECT_ID = f"resource/text/{ROLE}"


def _check_ranking(keys: int, ranked: list[int]) -> None:
    if len(ranked) > keys:
        raise ValueError(f"a {keys}-key profile ranks {len(ranked)} experts")
    seen: set[int] = set()
    for key in ranked:
        if key >= keys:
            raise ValueError(f"ranked key {key} is outside the {keys}-key profile")
        if key in seen:
            raise ValueError(f"ranked key {key} repeats")
        seen.add(key)


def encode(counts: Sequence[int], ranked: Sequence[int]) -> bytes:
    """The profile bytes for per-key use counts and a hottest-first ranking (every entry < keys, distinct)."""

    keys = len(counts)
    if keys == 0:
        raise ValueError("an expert profile needs at least one key")
    for name, values in (("counts", counts), ("ranked", ranked)):
        for value in values:
            if type(value) is not int or not 0 <= value <= U32_MAX:
                raise ValueError(f"{name}: entries must be u32 integers, got {value!r}")
    _check_ranking(keys, list(ranked))
    return HEADER.pack(MAGIC, VERSION, keys) + b"".join(
        U32.pack(value) for value in (*counts, len(ranked), *ranked)
    )


def parse(data: bytes, *, layers: int, experts: int) -> tuple[list[int], list[int]]:
    """Decode profile bytes, checked against the model's geometry; raises ValueError otherwise."""

    expected_keys = layers * experts
    if len(data) < HEADER.size:
        raise ValueError("an expert profile is at least 16 bytes")
    magic, version, keys = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError("not an expert profile")
    if version != VERSION:
        raise ValueError(f"unsupported expert profile version {version}")
    if keys != expected_keys:
        raise ValueError(
            f"the profile ranks {keys} experts, {layers} layers x {experts} experts is {expected_keys}"
        )
    body = HEADER.size + 4 * keys
    if len(data) < body + U32.size:
        raise ValueError("the expert profile is truncated")
    (ranked_count,) = U32.unpack_from(data, body)
    if len(data) != body + U32.size + 4 * ranked_count:
        raise ValueError("the expert profile's length differs from its declared contents")
    words = struct.unpack_from(f"<{keys + 1 + ranked_count}I", data, HEADER.size)
    counts = list(words[:keys])
    ranked = list(words[keys + 1 :])
    _check_ranking(keys, ranked)
    return counts, ranked
