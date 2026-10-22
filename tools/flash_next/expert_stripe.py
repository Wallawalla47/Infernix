"""Write or verify a Qwen3.8-Flash-Next expert stripe: the tail of every routed expert record, for
a second drive.

    python -m tools.flash_next.expert_stripe ARTIFACT D:/stripes/MODEL.stripe [--tail-fraction 0.5]
    python -m tools.flash_next.expert_stripe ARTIFACT D:/stripes/MODEL.stripe --verify

With `--expert-stripe` the SSD expert tier reads each record's head from the artifact and its tail
from the stripe at the same time, so a record that misses RAM crosses two drives in parallel. The
artifact keeps every record whole; the stripe holds a copy of the tails only. The format (v1) is
defined in src/runtime/expert_offload/expert_store.h; the stripe names the artifact's id, so it serves only
that artifact. A stripe is written to `<out>.partial` and renamed once complete.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import sys
import time

from tools.artifact.layouts import (
    EXPERT_RECORD_ALIGNMENT,
    FP8_EXPERT_B128_V1,
    GGML_EXPERT_DOWN_FORMATS,
    NVFP4_EXPERT_RG16_V1,
    align_up,
    expert_bank_geometry,
    fp8_expert_geometry,
    ggml_expert_geometry,
)
from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, TensorObject

MAGIC = b"INFXSTR1"
VERSION = 1
FIXED = struct.Struct("<8sIIII16s")  # magic, version, layers, experts, header bytes, artifact id
LAYER = struct.Struct("<QQQ")  # record stride, head bytes, file offset of the first tail
BANK_ID = re.compile(r"text/layers/(\d+)/moe/experts")
ALIGN = EXPERT_RECORD_ALIGNMENT


@dataclass(frozen=True)
class Bank:
    offset: int  # the bank object's offset in the artifact payload
    experts: int
    stride: int
    head: int
    tail_offset: int  # in the stripe file

    @property
    def tail(self) -> int:
        return self.stride - self.head


def record_stride(obj: TensorObject) -> int:
    if obj.layout == NVFP4_EXPERT_RG16_V1.name:
        return expert_bank_geometry(obj.format, obj.shape).record_stride
    if obj.layout == FP8_EXPERT_B128_V1.name:
        return fp8_expert_geometry(obj.format, obj.shape).record_stride
    if obj.layout in GGML_EXPERT_DOWN_FORMATS:
        return ggml_expert_geometry(obj.layout, obj.format, obj.shape).record_stride
    raise ArtifactError(f"{obj.id}: {obj.layout} is not a routed expert bank layout")


def plan(artifact: Artifact, tail_fraction: float) -> list[Bank]:
    """The routed expert banks of `artifact`, layer by layer, with their split and stripe offsets."""
    if not 0.0 < tail_fraction < 1.0:
        raise ArtifactError("the tail fraction must lie strictly between 0 and 1")
    # Each layer's bank is the object its parameter binds (object ids are the writer's own names).
    objects = {obj.id: obj for obj in artifact.directory.objects}
    found: dict[int, TensorObject] = {}
    for name, binding in artifact.directory.bindings.items():
        match = BANK_ID.fullmatch(name)
        obj = objects.get(binding.get("object")) if match and isinstance(binding, dict) else None
        if isinstance(obj, TensorObject):
            found[int(match.group(1))] = obj
    if not found or sorted(found) != list(range(len(found))):
        raise ArtifactError("the artifact has no routed expert banks text/layers/0..L-1/moe/experts")
    header = align_up(FIXED.size + LAYER.size * len(found), ALIGN)
    banks: list[Bank] = []
    offset = header
    for layer in range(len(found)):
        obj = found[layer]
        stride = record_stride(obj)
        head = (int(stride * (1.0 - tail_fraction)) // ALIGN) * ALIGN
        if not 0 <= head < stride:
            raise ArtifactError(f"{obj.id}: no tail at fraction {tail_fraction}")
        banks.append(Bank(obj.offset, obj.shape[0], stride, head, offset))
        offset += (stride - head) * obj.shape[0]
    if len({bank.experts for bank in banks}) != 1:
        raise ArtifactError("the expert banks differ in expert count")
    return banks


def header_bytes(artifact_id: bytes, banks: list[Bank]) -> bytes:
    size = banks[0].tail_offset
    data = FIXED.pack(MAGIC, VERSION, len(banks), banks[0].experts, size, artifact_id)
    data += b"".join(LAYER.pack(b.stride, b.head, b.tail_offset) for b in banks)
    return data + bytes(size - len(data))


def read_header(path: Path) -> tuple[bytes, list[Bank]]:
    with open(path, "rb") as file:
        fixed = file.read(FIXED.size)
        if len(fixed) != FIXED.size:
            raise ArtifactError(f"{path}: shorter than its header")
        magic, version, layers, experts, size, artifact_id = FIXED.unpack(fixed)
        if magic != MAGIC or version != VERSION:
            raise ArtifactError(f"{path}: not an expert stripe v1 (or not completely written)")
        rows = file.read(LAYER.size * layers)
    banks = []
    for layer in range(layers):
        stride, head, offset = LAYER.unpack_from(rows, LAYER.size * layer)
        banks.append(Bank(-1, experts, stride, head, offset))
    if banks and banks[0].tail_offset != size:
        raise ArtifactError(f"{path}: the first tail does not follow the header")
    return artifact_id, banks


def write(artifact_path: Path, out: Path, tail_fraction: float) -> dict:
    started = time.monotonic()
    partial = out.with_name(out.name + ".partial")
    with Artifact(artifact_path) as artifact:
        banks = plan(artifact, tail_fraction)
        written = 0
        with open(partial, "wb") as file:
            # The header goes last, so a file cut short has no magic.
            file.write(bytes(banks[0].tail_offset))
            for layer, bank in enumerate(banks):
                for expert in range(bank.experts):
                    record = bank.offset + expert * bank.stride
                    file.write(artifact.read_range(record + bank.head, bank.tail))
                written += bank.tail * bank.experts
                print(f"layer {layer}: {bank.experts} tails of {bank.tail} bytes", file=sys.stderr, flush=True)
            file.flush()
            os.fsync(file.fileno())
            file.seek(0)
            file.write(header_bytes(artifact.artifact_id, banks))
            file.flush()
            os.fsync(file.fileno())
        artifact_id = artifact.artifact_id.hex()
    os.replace(partial, out)
    return {
        "stripe": str(out),
        "artifact_id": artifact_id,
        "layers": len(banks),
        "tail_bytes": written,
        "head_fraction_of_records": round(sum(b.head for b in banks) / sum(b.stride for b in banks), 4),
        "seconds": round(time.monotonic() - started, 1),
    }


def verify(artifact_path: Path, stripe: Path) -> dict:
    started = time.monotonic()
    artifact_id, banks = read_header(stripe)
    mismatches = 0
    with Artifact(artifact_path) as artifact:
        if artifact_id != artifact.artifact_id:
            raise ArtifactError(f"{stripe}: belongs to artifact {artifact_id.hex()}, not {artifact.artifact_id.hex()}")
        expected = plan(artifact, 0.5)  # offsets and strides; the split comes from the stripe
        if len(expected) != len(banks):
            raise ArtifactError(f"{stripe}: {len(banks)} layers, the artifact has {len(expected)}")
        with open(stripe, "rb") as file:
            for layer, (bank, actual) in enumerate(zip(banks, expected)):
                if bank.stride != actual.stride or bank.experts != actual.experts:
                    raise ArtifactError(f"{stripe}: layer {layer} has another geometry than the artifact")
                file.seek(bank.tail_offset)
                for expert in range(bank.experts):
                    record = actual.offset + expert * bank.stride
                    if file.read(bank.tail) != artifact.read_range(record + bank.head, bank.tail):
                        mismatches += 1
                print(f"layer {layer}: verified", file=sys.stderr, flush=True)
    return {"stripe": str(stripe), "records": sum(b.experts for b in banks), "mismatches": mismatches,
            "seconds": round(time.monotonic() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("artifact", type=Path, help="the artifact's entry file")
    parser.add_argument("stripe", type=Path, help="the stripe file (on another drive than the artifact)")
    parser.add_argument("--tail-fraction", type=float, default=0.5,
                        help="share of each record kept in the stripe, rounded to 4 KiB (default 0.5)")
    parser.add_argument("--verify", action="store_true", help="compare an existing stripe with the artifact")
    args = parser.parse_args()

    result = verify(args.artifact, args.stripe) if args.verify else write(args.artifact, args.stripe, args.tail_fraction)
    print(json.dumps(result, indent=2))
    if args.verify and result["mismatches"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
