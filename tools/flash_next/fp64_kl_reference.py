"""Write the FP64 reference's distributions as an infernix-perplexity --kl-reference file.

    python -m tools.flash_next.fp64_kl_reference --model CHECKPOINT --tokens windows.json \\
        --out fp64.top [--reference-points] [--a16-experts]

``windows.json`` is the --save-tokens output of an infernix-perplexity run (its windows' input
tokens). For every window the independent FP64 forward of reference.py runs over the input tokens,
and the 64 most probable tokens of every target position, their log-probabilities and the target's
own log-probability are written in the tool's top-token reference format (version 1, the same
header, windows and fingerprints). Scoring any route with ``--kl-reference fp64.top`` then reports
its KL divergence from the FP64 evaluation and its top-1 agreement exactly as the gates measure
them (docs/maintainer/model-quality.md §6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from tools.flash_next.reference import Reference

_MAGIC = b"NINFTOPK"
_VERSION = 1
_END = 0xFFFFFFFF
_TOP = 64


def fingerprint(tokens: list[int]) -> int:
    """FNV-1a over each token's four little-endian bytes (apps/perplexity token_fingerprint)."""
    h = 14695981039346656037
    for token in tokens:
        value = token & 0xFFFFFFFF
        for _ in range(4):
            h ^= value & 0xFF
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            value >>= 8
    return h


def string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def window_record(window: dict, logits: np.ndarray) -> bytes:
    """One window: targets [target_begin, target_end) are tokens[first_target:], each predicted by
    the logits of the position before it."""
    tokens, first = window["tokens"], window["first_target"]
    count = window["target_end"] - window["target_begin"]
    if first < 1 or first + count != len(tokens):
        raise ValueError("a window's targets must be its input's last tokens after the first")
    rows = logits[first - 1 : first - 1 + count].astype(np.float64)
    shifted = rows - rows.max(axis=1, keepdims=True)
    logprobs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    targets = np.array(tokens[first:], dtype=np.int64)
    target_lp = logprobs[np.arange(count), targets].astype(np.float32)
    # Descending log-probability; ties to the lower id.
    order = np.lexsort((np.broadcast_to(np.arange(rows.shape[1]), rows.shape), -logprobs), axis=1)[:, :_TOP]
    top_lp = np.take_along_axis(logprobs, order, axis=1).astype(np.float32)
    return (struct.pack("<IQQQ", window["stream_index"], window["target_begin"], window["target_end"],
                        window["input_hash"])
            + target_lp.tobytes() + order.astype(np.int32).tobytes() + top_lp.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True, help="infernix-perplexity --save-tokens output")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-points", action="store_true")
    parser.add_argument("--a16-experts", action="store_true")
    args = parser.parse_args()
    saved = json.loads(args.tokens.read_text(encoding="utf-8"))
    ref = Reference(args.model, reference_points=args.reference_points, a16_experts=args.a16_experts)
    out = bytearray(_MAGIC + struct.pack("<IIII", _VERSION, _TOP, saved["context"], saved["stride"]))
    out += string(saved["corpus_id"]) + string(saved["mode"])
    for index, window in enumerate(saved["windows"]):
        if fingerprint(window["tokens"]) != window["input_hash"]:
            raise ValueError(f"window {index}: the tokens do not match their fingerprint")
        record = ref.forward(window["tokens"], log=lambda line: print(f"window {index}: {line}", flush=True))
        out += window_record(window, record["logits"])
    out += struct.pack("<I", _END)
    args.out.write_bytes(bytes(out))
    print(f"wrote {len(saved['windows'])} windows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
