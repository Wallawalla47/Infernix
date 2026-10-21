"""Word-equality qualification of a Qwen3.8-Flash-Next artifact against its checkpoint (design §6.1, §16.3).

    python -m tools.flash_next.verify_artifact --model CHECKPOINT --artifact OUT.infernix \
        [--ngram OUT.ngram] [--ngram-rows all|N]

Every logical parameter the recipe stores as the checkpoint has it must equal the checkpoint's words:

- each routed expert's E2M1 codes, E4M3 block scales and FP32 multiplier (decoded from the
  ``nvfp4_expert_rg16_v1`` record): ModelOpt's ``weight_scale_2`` with its gate/up/down
  ``input_scale`` words, or a weight-only compressed-tensors export's reciprocal global scale; or,
  for Qwen's official FP8 checkpoint, each ``fp8_expert_b128_v1`` record byte for byte (its E4M3FN
  codes and BF16 tile scales);
- every FP8 row-scaled projection's E4M3FN codes and BF16 row scales, and every MXFP8 projection's
  E4M3FN codes and UE8M0 group scales (``fp8_e4m3fn_g32_e8m0``);
- every BF16 and FP32 tensor bit for bit, and every FP32 widening of a BF16 tensor exactly;
- every tensor stored as the BF16 rounding of exact source values (FP4-Local's small MXFP8 classes
  and vision tower) as that one rounding; the values it does not hold exactly (the weight-only NVFP4
  vision matrices) are counted per tensor (``rounded``), not as mismatches;
- the n-gram volume's header, geometry and id, and its rows (all of them, or a sample): FP8 rows
  equal the checkpoint's FP8 words, or its BF16 table's per-tensor quantization; the BF16 rows of an
  artifact bound to a BF16 volume equal its BF16 table, or quantize to its FP8 table's words
  (qwen4_exp.NgramTable); NVFP4 rows equal the checkpoint's codes and block scales, and the header
  its format tag and FP32 multiplier.

Parameters the recipe re-quantizes (``q8_g32_fp16``/``q4_g64_fp16``/``q2_g64_fp16``: every recipe's MTP drafter,
recipes B and C's 8-bit classes) are listed per component, not compared. Recipe A passes with only
the drafter re-quantized.

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

import numpy as np

from tools.artifact.codecs.fp8_b128 import decode_expert as decode_fp8_expert
from tools.artifact.codecs.fp8_g32 import decode_fp8_group_scaled_words
from tools.artifact.codecs.fp8_row import decode_fp8_row_scaled_words
from tools.artifact.codecs.nvfp4_expert import decode_nvfp4_expert_bank_words
from tools.artifact.layouts import fp8_expert_geometry
from tools.artifact.reader import Artifact
from tools.convert import qwen4_exp
from tools.convert.methods import bf16_from_float64
from tools.convert.sources.modelopt import fp8_block_matrix_raw, nvfp4_matrix_words, nvfp4_packed_matrix_words
from tools.convert.sources.safetensors import SafetensorsSource

_FP8_ROWS = "fp8_e4m3fn_row_bf16"
_FP8_GROUPS = "fp8_e4m3fn_g32_e8m0"

_WORD = {"bf16": (torch.bfloat16, 2), "fp32": (torch.float32, 4), "int32": (torch.int32, 4)}
_REQUANTIZED = {"q8_g32_fp16", "q4_g64_fp16", "q2_g64_fp16"}


def _binding_formats(artifact: Artifact, binding: dict) -> set[str]:
    parts = binding["parts"] if "parts" in binding else [{"object": binding["object"]}]
    return {artifact.object(part["object"]).format for part in parts}


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


def verify_fp8_rows(artifact, store, binding, parameter) -> bool:
    """The bound rows' E4M3FN codes and BF16 row scales equal the source's encoded rows."""

    n, k = parameter.shape
    expected = parameter.source_factory(store, _FP8_ROWS).read_encoded(0, n)
    parts = binding["parts"] if "parts" in binding else [{"object": binding["object"]}]
    codes, scales = [], []
    for part in parts:
        obj = artifact.object(part["object"])
        c, s = decode_fp8_row_scaled_words(artifact.read_object(obj.id), obj.shape)
        begin, end = part.get("range", (0, obj.shape[0] * obj.shape[1]))
        codes.append(c[begin // k : end // k])
        scales.append(s[begin // k : end // k])
    return torch.equal(torch.cat(codes), expected.codes.reshape(n, k)) and torch.equal(
        torch.cat(scales).view(torch.int16), expected.scales.to(torch.bfloat16).view(torch.int16))


def verify_fp8_groups(artifact, store, binding, parameter) -> bool:
    """The bound rows' E4M3FN codes and UE8M0 group scales equal the source's encoded rows."""

    n, k = parameter.shape
    expected = parameter.source_factory(store, _FP8_GROUPS).read_encoded(0, n)
    parts = binding["parts"] if "parts" in binding else [{"object": binding["object"]}]
    codes, scales = [], []
    for part in parts:
        obj = artifact.object(part["object"])
        c, s = decode_fp8_group_scaled_words(artifact.read_object(obj.id), obj.shape)
        begin, end = part.get("range", (0, obj.shape[0] * obj.shape[1]))
        codes.append(c[begin // k : end // k])
        scales.append(s[begin // k : end // k])
    return torch.equal(torch.cat(codes), expected.codes) and torch.equal(torch.cat(scales), expected.scales)


def verify_bank(artifact, store, binding, prefix, shape, packed) -> int:
    obj = artifact.object(binding["object"])
    words = decode_nvfp4_expert_bank_words(artifact.read_object(obj.id), shape)
    read = nvfp4_packed_matrix_words if packed else nvfp4_matrix_words
    e, h, i = shape
    bad = 0
    for x in range(e):
        for role, rows, cols in (("gate", i, h), ("up", i, h), ("down", h, i)):
            src = read(store, f"{prefix}{x}.{role}_proj", (rows, cols))
            col = {"gate": 0, "up": 1, "down": 2}[role]
            ok = (
                torch.equal(words[f"{role}_codes"][x], src.codes)
                and torch.equal(words[f"{role}_scales"][x], src.scales)
                and torch.equal(words["multipliers"][x, col].view(torch.int32), src.weight_scale_2.view(torch.int32))
            )
            bad += not ok
    return bad


def verify_fp8_bank(artifact, binding, source: qwen4_exp.Fp8ExpertBankSource) -> int:
    """Each ``fp8_expert_b128_v1`` record equals the checkpoint's codes and BF16 tile scales of its
    expert, then zero padding; returns the mismatching experts."""

    obj = artifact.object(binding["object"])
    g = fp8_expert_geometry(obj.format, obj.shape)
    if tuple(obj.shape) != source.shape or obj.layout != "fp8_expert_b128_v1":
        raise ValueError(f"{source.label}: stored bank differs from the checkpoint's shape or layout")
    _, hidden, inter = source.shape
    pad = bytes(g.record_stride - g.record_bytes)
    bad = 0
    for e in range(source.shape[0]):
        record = artifact.read_range(obj.offset + e * g.record_stride, g.record_stride)
        leaf = f"{source.prefix}{e}."
        expected = (fp8_block_matrix_raw(source.store, leaf + "gate_proj", (inter, hidden)),
                    fp8_block_matrix_raw(source.store, leaf + "up_proj", (inter, hidden)),
                    fp8_block_matrix_raw(source.store, leaf + "down_proj", (hidden, inter)))
        same = record[g.record_bytes :] == pad and all(
            np.array_equal(codes, got_codes) and np.array_equal(words, got_words)
            for (codes, words), (got_codes, got_words) in zip(expected, decode_fp8_expert(record, hidden, inter)))
        bad += not same
    return bad


def verify_ngram(source: qwen4_exp.NgramTable, path: Path, table: dict, rows: str) -> tuple[int, int]:
    shards = source.shards
    bf16 = table["format"] == "bf16"

    def mismatches(name: str, begin: int, end: int, vol: torch.Tensor) -> int:
        # FP8 rows equal the checkpoint's words; BF16 rows are its BF16 table (NgramTable.bf16_rows_match).
        if bf16:
            return int((~source.bf16_rows_match(name, begin, end, vol)).sum())
        return int((vol != source.codes(name, begin, end)).any(dim=1).sum())

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
        if table["format"] == "nvfp4":
            stream.seek(0)
            fields = stream.read(qwen4_exp.NGRAM_NVFP4_SCALE_OFFSET + qwen4_exp.NGRAM_NVFP4_SCALE.size)
            tag = qwen4_exp.NGRAM_FORMAT.unpack_from(fields, qwen4_exp.NGRAM_FORMAT_OFFSET)[0]
            bits = qwen4_exp.NGRAM_NVFP4_SCALE.unpack_from(fields, qwen4_exp.NGRAM_NVFP4_SCALE_OFFSET)[0]
            if tag != qwen4_exp.NGRAM_FORMAT_CODES["nvfp4"] or bits != (
                source.nvfp4_scale().view(torch.int32).item() & 0xFFFFFFFF
            ):
                raise ValueError("NVFP4 n-gram volume header differs from the table's format or multiplier")
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
                    r0, r1 = first + begin, first + end
                    b0, b1 = r0 // per_block, (r1 - 1) // per_block
                    stream.seek(header_bytes + b0 * block_bytes)
                    raw = torch.frombuffer(bytearray(stream.read((b1 - b0 + 1) * block_bytes)), dtype=torch.uint8)
                    vol = raw.reshape(-1, block_bytes)[:, : per_block * row_bytes].reshape(-1, row_bytes)
                    vol = vol[r0 - b0 * per_block : r1 - b0 * per_block]
                    bad += mismatches(name, begin, end, vol)
                    checked += end - begin
            return checked, bad
        rng = random.Random(7)
        sample = sorted(rng.randrange(nrows) for _ in range(int(rows)))
        bad = 0
        for r in sample:
            index = max(i for i, s in enumerate(starts) if s <= r)
            name, _ = shards[index]
            local = r - starts[index]
            stream.seek(header_bytes + (r // per_block) * block_bytes + (r % per_block) * row_bytes)
            vol = torch.frombuffer(bytearray(stream.read(row_bytes)), dtype=torch.uint8).reshape(1, row_bytes)
            bad += mismatches(name, local, local + 1, vol)
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
              "input_scale_mismatches": 0, "requantized": {}, "rounded": {}}
    with SafetensorsSource(args.model) as store, Artifact(args.artifact) as artifact:
        text = artifact.directory.components["text"]["config"]
        components = tuple(artifact.directory.components)
        model = qwen4_exp.build_model(store, components=components, ngram_format=text["ngram_table"]["format"])
        # The proposal head (--lm-head-draft) is built from a frequency ranking, outside the logical model.
        bindings = {k: v for k, v in artifact.directory.bindings.items() if not k.startswith("proposal/")}
        report["proposal_bindings"] = len(artifact.directory.bindings) - len(bindings)
        if set(model.parameters) != set(bindings):
            missing = sorted(set(model.parameters) - set(bindings))[:5]
            extra = sorted(set(bindings) - set(model.parameters))[:5]
            raise ValueError(f"artifact bindings differ from the logical model: {missing} {extra}")
        for name, parameter in model.parameters.items():
            binding = bindings[name]
            if name.endswith("/moe/experts") and name.startswith("text/"):
                source = parameter.source
                if isinstance(source, qwen4_exp.Fp8ExpertBankSource):
                    report["expert_mismatches"] += verify_fp8_bank(artifact, binding, source)
                else:
                    report["expert_mismatches"] += verify_bank(artifact, store, binding, source.prefix, source.shape,
                                                               source.packed)
                report["experts"] += source.shape[0]
                print(f"{name}: {source.shape[0]} experts verified ({time.time() - start:.0f}s)", flush=True)
                continue
            formats = _binding_formats(artifact, binding)
            if formats & _REQUANTIZED:
                component = name.split("/", 1)[0]
                report["requantized"][component] = report["requantized"].get(component, 0) + 1
                continue
            if formats == {_FP8_ROWS}:
                report["tensors"] += 1
                if not verify_fp8_rows(artifact, store, binding, parameter):
                    report["tensor_mismatches"].append(name)
                continue
            if formats == {_FP8_GROUPS}:
                report["tensors"] += 1
                if not verify_fp8_groups(artifact, store, binding, parameter):
                    report["tensor_mismatches"].append(name)
                continue
            fmt, stored = _binding_values(artifact, binding)
            expected = parameter.source.values()
            if name.endswith("/moe/expert_input_scales"):
                report["input_scale_mismatches"] += int((_bits(stored) != _bits(expected.to(torch.float32))).sum())
                continue
            report["tensors"] += 1
            if expected.dtype == torch.float64 and fmt == "bf16":
                # Exact source values (MXFP8 or weight-only NVFP4) stored as their one BF16 rounding.
                if not torch.equal(_bits(stored), _bits(bf16_from_float64(expected))):
                    report["tensor_mismatches"].append(name)
                inexact = int((stored.to(torch.float64) != expected).sum())
                if inexact:
                    report["rounded"][name] = inexact
                continue
            # BF16 and FP32 sources: kept bit for bit, or BF16 widened exactly to FP32.
            if expected.dtype not in (torch.bfloat16, torch.float32) or not torch.equal(
                _bits(stored), _bits(expected.to(_WORD[fmt][0]))
            ) or (expected.dtype == torch.float32 and fmt != "fp32"):
                report["tensor_mismatches"].append(name)
        if args.ngram is not None:
            table = text["ngram_table"]
            checked, bad = verify_ngram(qwen4_exp.ngram_table(store, text), args.ngram, table, args.ngram_rows)
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
