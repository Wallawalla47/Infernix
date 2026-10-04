"""Recipe A qualification of a Qwen3.8-Flash-Next artifact (design §16.3): word equality.

    python -m tools.flash_next.verify_artifact --model CHECKPOINT --artifact OUT.ninfer \
        [--ngram OUT.ninfer.ngram] [--ngram-rows all|N]

Every logical parameter's stored words must equal the checkpoint's:

- each routed expert's E2M1 codes, E4M3 block scales and FP32 ``weight_scale_2`` (decoded from the
  ``nvfp4_expert_rg16_v1`` record), and its gate/up/down ``input_scale`` words;
- every BF16 tensor bit for bit, and every FP32 widening of a BF16 tensor exactly;
- the n-gram volume's header, geometry and id, and its FP8 rows (all of them, or a sample).

Source names and axes come from the converter's logical model, so this proves the writer, the
recipe and the bindings; the model's semantics are proven by tools/flash_next/reference.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import struct
import time

import torch

from tools.artifact.codecs.nvfp4_expert import decode_nvfp4_expert_bank_words
from tools.artifact.reader import Artifact
from tools.convert import qwen4_exp
from tools.convert.sources.modelopt import nvfp4_matrix_words
from tools.convert.sources.safetensors import SafetensorsSource

_WORD = {"bf16": (torch.bfloat16, 2), "fp32": (torch.float32, 4), "int32": (torch.int32, 4)}


def _binding_values(artifact: Artifact, binding: dict) -> tuple[str, torch.Tensor]:
    parts = binding["parts"] if "parts" in binding else [{"object": binding["object"]}]
    formats, chunks = set(), []
    for part in parts:
        obj = artifact.object(part["object"])
        formats.add(obj.format)
        dtype, width = _WORD[obj.format]
        begin, end = part.get("range", (0, obj.bytes // width))
        raw = artifact.read_range(obj.offset + begin * width, (end - begin) * width)
        chunks.append(torch.frombuffer(bytearray(raw), dtype=dtype))
    if len(formats) != 1:
        raise ValueError("a direct binding mixes formats")
    return formats.pop(), torch.cat(chunks)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


def verify_bank(artifact, store, binding, prefix, shape) -> int:
    obj = artifact.object(binding["object"])
    words = decode_nvfp4_expert_bank_words(artifact.read_object(obj.id), shape)
    e, h, i = shape
    bad = 0
    for x in range(e):
        for role, rows, cols in (("gate", i, h), ("up", i, h), ("down", h, i)):
            src = nvfp4_matrix_words(store, f"{prefix}{x}.{role}_proj", (rows, cols))
            col = {"gate": 0, "up": 1, "down": 2}[role]
            ok = (
                torch.equal(words[f"{role}_codes"][x], src.codes)
                and torch.equal(words[f"{role}_scales"][x], src.scales)
                and torch.equal(words["multipliers"][x, col].view(torch.int32), src.weight_scale_2.view(torch.int32))
            )
            bad += not ok
    return bad


def verify_ngram(store, shards, path: Path, table: dict, rows: str) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(qwen4_exp.NGRAM_HEADER.size)
        magic, version, header_bytes, nrows, row_bytes, per_block, block_bytes, blocks, vid = (
            qwen4_exp.NGRAM_HEADER.unpack(header)
        )
        expected = (qwen4_exp.NGRAM_MAGIC, qwen4_exp.NGRAM_VERSION, table["header_bytes"], table["rows"],
                    table["row_bytes"], table["rows_per_block"], table["block_bytes"], table["blocks"])
        if (magic, version, header_bytes, nrows, row_bytes, per_block, block_bytes, blocks) != expected:
            raise ValueError("n-gram volume header differs from the artifact's table geometry")
        if vid.hex() != table["volume_id"]:
            raise ValueError("n-gram volume id differs from the artifact's")
        if path.stat().st_size != table["file_bytes"]:
            raise ValueError("n-gram volume size differs from its geometry")
        starts, total = [], 0
        for name, count in shards:
            starts.append(total)
            total += count
        if rows == "all":
            checked = bad = 0
            for (name, count), first in zip(shards, starts):
                step = 1 << 20
                for begin in range(0, count, step):
                    end = min(count, begin + step)
                    src = store.read_flat(name, begin * row_bytes, end * row_bytes).view(torch.uint8)
                    src = src.reshape(end - begin, row_bytes)
                    r0, r1 = first + begin, first + end
                    b0, b1 = r0 // per_block, (r1 - 1) // per_block
                    stream.seek(header_bytes + b0 * block_bytes)
                    raw = torch.frombuffer(bytearray(stream.read((b1 - b0 + 1) * block_bytes)), dtype=torch.uint8)
                    vol = raw.reshape(-1, block_bytes)[:, : per_block * row_bytes].reshape(-1, row_bytes)
                    vol = vol[r0 - b0 * per_block : r1 - b0 * per_block]
                    bad += int((vol != src).any(dim=1).sum())
                    checked += end - begin
            return checked, bad
        rng = random.Random(7)
        sample = sorted(rng.randrange(nrows) for _ in range(int(rows)))
        bad = 0
        for r in sample:
            index = max(i for i, s in enumerate(starts) if s <= r)
            name, _ = shards[index]
            local = r - starts[index]
            src = store.read_flat(name, local * row_bytes, (local + 1) * row_bytes).view(torch.uint8)
            stream.seek(header_bytes + (r // per_block) * block_bytes + (r % per_block) * row_bytes)
            vol = torch.frombuffer(bytearray(stream.read(row_bytes)), dtype=torch.uint8)
            bad += not torch.equal(src, vol)
        return len(sample), bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ngram", type=Path)
    parser.add_argument("--ngram-rows", default="200000", help="'all' or a sample size")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    start = time.time()
    report = {"experts": 0, "expert_mismatches": 0, "tensors": 0, "tensor_mismatches": [],
              "input_scale_mismatches": 0}
    with SafetensorsSource(args.model) as store, Artifact(args.artifact) as artifact:
        text = artifact.directory.components["text"]["config"]
        components = tuple(artifact.directory.components)
        model = qwen4_exp.build_model(store, components=components)
        bindings = artifact.directory.bindings
        if set(bindings) != set(model.parameters):
            missing = sorted(set(model.parameters) - set(bindings))[:5]
            extra = sorted(set(bindings) - set(model.parameters))[:5]
            raise ValueError(f"artifact bindings differ from the logical model: {missing} {extra}")
        for name, parameter in model.parameters.items():
            binding = bindings[name]
            if name.endswith("/moe/experts") and name.startswith("text/"):
                source = parameter.source
                report["expert_mismatches"] += verify_bank(artifact, store, binding, source.prefix, source.shape)
                report["experts"] += source.shape[0]
                print(f"{name}: {source.shape[0]} experts verified ({time.time() - start:.0f}s)", flush=True)
                continue
            fmt, stored = _binding_values(artifact, binding)
            expected = parameter.source.values()
            if name.endswith("/moe/expert_input_scales"):
                report["input_scale_mismatches"] += int((_bits(stored) != _bits(expected.to(torch.float32))).sum())
                continue
            report["tensors"] += 1
            # BF16 sources: kept bit for bit, or widened exactly to FP32.
            if expected.dtype != torch.bfloat16 or not torch.equal(
                _bits(stored), _bits(expected.to(_WORD[fmt][0]))
            ):
                report["tensor_mismatches"].append(name)
        if args.ngram is not None:
            table = text["ngram_table"]
            shards = qwen4_exp.ngram_volume_shards(store, text)
            checked, bad = verify_ngram(store, shards, args.ngram, table, args.ngram_rows)
            report["ngram_rows_checked"], report["ngram_row_mismatches"] = checked, bad
    report["seconds"] = time.time() - start
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    ok = (not report["expert_mismatches"] and not report["tensor_mismatches"]
          and not report["input_scale_mismatches"] and not report.get("ngram_row_mismatches"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
