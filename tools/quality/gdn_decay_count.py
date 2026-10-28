"""How slowly do a Qwen3.5-family model's Gated DeltaNet heads forget (candidates LE-05 (a)).

`fp16-gdn-state` rounds the recurrent state to FP16 after every decoded token. A head whose per-step
forgetting 1 - decay is far below FP16's relative step (2^-11) lets rounding error accumulate across a
long generation instead of decaying away. The decay of a head is exp(g), g = -exp(A_log) * softplus(a +
dt_bias), a the input-dependent projection output. This script reads A_log and dt_bias of every Linear
Attention layer straight from a Hugging Face checkpoint (the safetensors headers and the few hundred
bytes of each tensor, by HTTP range, no shard downloaded) and counts, per layer and overall, the heads
with 1 - decay below 2^-12 at a = 0 and at a = +-2 (about the projection's spread).

  py -3.11 -I tools/quality/gdn_decay_count.py [--repo Qwen/Qwen3.8-27B] [--revision main]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import urllib.request

THRESHOLD = 2.0**-12
NAME = re.compile(r"^(?P<prefix>.*layers\.(?P<layer>\d+)\.linear_attn)\.(?P<field>A_log|dt_bias)$")


def fetch(url: str, start: int | None = None, length: int | None = None) -> bytes:
    request = urllib.request.Request(url)
    if start is not None:
        request.add_header("Range", f"bytes={start}-{start + length - 1}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def tensor(url_base: str, shard: str, name: str, headers: dict[str, tuple[int, dict]]) -> list[float]:
    if shard not in headers:
        url = f"{url_base}/{shard}"
        size = struct.unpack("<Q", fetch(url, 0, 8))[0]
        headers[shard] = (8 + size, json.loads(fetch(url, 8, size)))
    base, header = headers[shard]
    entry = header[name]
    begin, end = entry["data_offsets"]
    raw = fetch(f"{url_base}/{shard}", base + begin, end - begin)
    dtype = entry["dtype"]
    if dtype == "F32":
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    if dtype == "BF16":
        return [struct.unpack("<f", b"\x00\x00" + raw[i : i + 2])[0] for i in range(0, len(raw), 2)]
    if dtype == "F16":
        return list(struct.unpack(f"<{len(raw) // 2}e", raw))
    raise SystemExit(f"{name}: unsupported dtype {dtype}")


def softplus(x: float) -> float:
    return x if x > 20.0 else math.log1p(math.exp(x))


def forgetting(a_log: float, dt_bias: float, a: float) -> float:
    """1 - decay for one head at projection output a."""
    return -math.expm1(-math.exp(a_log) * softplus(a + dt_bias))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}"
    index = json.loads(fetch(f"{base}/model.safetensors.index.json"))["weight_map"]

    layers: dict[int, dict[str, str]] = {}
    for name in index:
        match = NAME.match(name)
        if match:
            layers.setdefault(int(match["layer"]), {})[match["field"]] = name
    if not layers:
        raise SystemExit("no linear_attn.A_log / dt_bias tensors in the index")

    headers: dict[str, tuple[int, dict]] = {}
    totals = {a: 0 for a in (-2.0, 0.0, 2.0)}
    heads = 0
    slowest = (1.0, -1, -1)
    print(f"{args.repo}: {len(layers)} Linear Attention layers; heads with 1 - decay < 2^-12 at a = -2 / 0 / +2")
    for layer in sorted(layers):
        fields = layers[layer]
        a_log = tensor(base, index[fields["A_log"]], fields["A_log"], headers)
        dt_bias = tensor(base, index[fields["dt_bias"]], fields["dt_bias"], headers)
        counts = {a: sum(forgetting(x, b, a) < THRESHOLD for x, b in zip(a_log, dt_bias)) for a in totals}
        for a in totals:
            totals[a] += counts[a]
        heads += len(a_log)
        for head, (x, b) in enumerate(zip(a_log, dt_bias)):
            slowest = min(slowest, (forgetting(x, b, 0.0), layer, head))
        print(f"layer {layer:3d}: {len(a_log)} heads  {counts[-2.0]:3d} {counts[0.0]:3d} {counts[2.0]:3d}")
    print(f"total: {heads} heads  {totals[-2.0]} {totals[0.0]} {totals[2.0]}")
    print(f"slowest head at a = 0: layer {slowest[1]} head {slowest[2]}, 1 - decay = {slowest[0]:.3e} (2^-12 = {THRESHOLD:.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
