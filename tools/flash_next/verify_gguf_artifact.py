"""Byte-equality qualification of a Qwen3.8-Flash-Next artifact imported from a GGUF
(docs/maintainer/qwen3_8-flash-next-design.md §19.3.25).

    python -m tools.flash_next.verify_gguf_artifact --model QWEN_DIR --gguf SHARD [--gguf SHARD ...] \
        --artifact OUT.infernix [--ngram OUT.ngram] [--ngram-rows all|N]

Every logical parameter the recipe imports must equal the GGUF:

- each routed expert's record: its gate, up and down rows, the GGUF's stored blocks byte for byte,
  then zero padding to the record stride;
- every ggml_rows_v1 projection's rows, the stored blocks byte for byte (whole-row selections and
  the GDN value-head permutation included); a ggml_tiles16_v1 matrix's rows after untiling, and its
  last tile's padding rows zero blocks;
- the Q8_0 tensor's codes and scales, as the identical q8_g32_fp16 words;
- every BF16/FP32 tensor value for value (the converter's checked exact transforms applied);
- the n-gram volume's header and its IQ4_NL rows (all of them, or a sample);
- with the vision component (``--mmproj``), every vision tensor value for value against the mmproj,
  and with ``--vision-reference`` bit for bit against another artifact's vision bindings (one
  converted from Qwen's checkpoint: the mmproj then holds Qwen's tower).

Parameters the recipe re-quantizes (the MTP drafter from Qwen's checkpoint) are counted, not
compared.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.artifact.codecs.row_split import decode_row_split_codes
from tools.artifact.layouts import ggml_expert_geometry, ggml_rows_geometry, ggml_tiles16_geometry
from tools.artifact.reader import Artifact
from tools.convert import qwen4_exp_gguf
from tools.convert.sources.gguf_raw import GgufStore
from tools.convert.sources.safetensors import SafetensorsSource
from tools.flash_next.verify_artifact import _WORD, _binding_values, _bits, verify_ngram

_REQUANTIZED_COMPONENTS = ("mtp/", "proposal/")


def _parts(binding: dict) -> list[dict]:
    return binding["parts"] if "parts" in binding else [{"object": binding["object"]}]


def verify_bank(artifact: Artifact, binding: dict, bank: qwen4_exp_gguf.GgmlExpertBank) -> int:
    obj = artifact.object(binding["object"])
    g = ggml_expert_geometry(obj.layout, obj.format, obj.shape)
    if (g.gate_up_format, g.down_format) != (bank.gate_up_format, bank.down_format):
        raise ValueError(f"{bank.label}: stored formats differ from the GGUF's")
    experts, hidden, inter = bank.shape
    pad = bytes(g.record_stride - g.record_bytes)
    bad = 0
    for e in range(experts):
        record = artifact.read_range(obj.offset + e * g.record_stride, g.record_stride)
        expected = (bank.store.raw(bank.gate.name, e * inter, (e + 1) * inter).tobytes()
                    + bank.store.raw(bank.up.name, e * inter, (e + 1) * inter).tobytes()
                    + bank.store.raw(bank.down.name, e * hidden, (e + 1) * hidden).tobytes() + pad)
        bad += record != expected
    return bad


def tiles16_rows(raw: bytes, n: int, row_bytes: int, block_bytes: int) -> np.ndarray:
    """The rows of a whole ggml_tiles16_v1 payload ([tiles, block, row, block bytes] -> [rows, row
    bytes]); raises unless the last tile's padding rows are zero blocks."""
    tiles, blocks = (n + 15) // 16, row_bytes // block_bytes
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(tiles, blocks, 16, block_bytes).transpose(0, 2, 1, 3)
    rows = rows.reshape(tiles * 16, row_bytes)
    if rows[n:].any():
        raise ValueError("a ggml_tiles16_v1 matrix's padding rows are not zero blocks")
    return rows[:n]


def verify_ggml_rows(artifact: Artifact, binding: dict, parameter) -> bool:
    n, k = parameter.shape
    expected = parameter.source.read_encoded(0, n).codes.numpy()
    stored = []
    for part in _parts(binding):
        obj = artifact.object(part["object"])
        tiled = obj.layout == "ggml_tiles16_v1"
        g = ggml_tiles16_geometry(obj.format, obj.shape) if tiled else ggml_rows_geometry(obj.format, obj.shape)
        begin, end = part.get("range", (0, g.n * g.k))
        if begin % k or end % k:
            raise ValueError("a ggml binding must cover whole rows")
        if tiled:  # the whole matrix untiled (its padding checked), then the part's rows
            rows = tiles16_rows(artifact.read_range(obj.offset, g.payload_bytes), g.n, g.row_bytes, g.row_bytes // g.blocks_per_row)
            stored.append(rows[begin // k : end // k])
            continue
        raw = artifact.read_range(obj.offset + begin // k * g.row_bytes, (end - begin) // k * g.row_bytes)
        stored.append(np.frombuffer(raw, dtype=np.uint8).reshape(-1, g.row_bytes))
    return np.array_equal(np.concatenate(stored), expected)


def verify_q8(artifact: Artifact, binding: dict, parameter) -> bool:
    n, k = parameter.shape
    expected = parameter.source.read_encoded(0, n)
    codes, scales = [], []
    for part in _parts(binding):
        obj = artifact.object(part["object"])
        s, c = decode_row_split_codes(artifact.read_object(obj.id), obj.format, obj.shape)
        begin, end = part.get("range", (0, obj.shape[0] * obj.shape[1]))
        codes.append(c[begin // k : end // k, : k // 32])
        scales.append(s[begin // k : end // k, : k // 32])
    return torch.equal(torch.cat(codes).reshape(n, k // 32, 32), expected.codes) and torch.equal(
        torch.cat(scales).view(torch.int16), expected.scales.view(torch.int16))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="the Qwen checkpoint directory the import used")
    parser.add_argument("--gguf", type=Path, action="append", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--ngram", type=Path)
    parser.add_argument("--ngram-rows", default="200000", help="'all' or a sample size")
    parser.add_argument("--mmproj", type=Path, help="the mmproj the vision component was imported from")
    parser.add_argument("--vision-reference", type=Path,
                        help="an artifact whose vision bindings must equal this one's bit for bit")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    start = time.time()
    report = {"experts": 0, "expert_mismatches": 0, "ggml_tensors": 0, "q8_tensors": 0, "tensors": 0,
              "tensor_mismatches": [], "requantized": {}}
    with ExitStack() as stack:
        base = stack.enter_context(SafetensorsSource(args.model))
        store = stack.enter_context(GgufStore(args.gguf))
        artifact = stack.enter_context(Artifact(args.artifact))
        mmproj = None
        if args.mmproj is not None:
            from tools.convert.sources.gguf import GGUFSource  # the gguf package, for the mmproj only

            mmproj = stack.enter_context(GGUFSource(args.mmproj))
        text = artifact.directory.components["text"]["config"]
        model = qwen4_exp_gguf.build_model(base, store, components=tuple(artifact.directory.components), mmproj=mmproj)
        bindings = {k: v for k, v in artifact.directory.bindings.items() if not k.startswith("proposal/")}
        report["proposal_bindings"] = len(artifact.directory.bindings) - len(bindings)
        if set(bindings) != set(model.parameters):
            missing = sorted(set(model.parameters) - set(bindings))[:5]
            extra = sorted(set(bindings) - set(model.parameters))[:5]
            raise ValueError(f"artifact bindings differ from the logical model: {missing} {extra}")
        for name, parameter in model.parameters.items():
            binding = bindings[name]
            if name.startswith(_REQUANTIZED_COMPONENTS):
                component = name.split("/", 1)[0]
                report["requantized"][component] = report["requantized"].get(component, 0) + 1
                continue
            if isinstance(parameter.source, qwen4_exp_gguf.GgmlExpertBank):
                report["expert_mismatches"] += verify_bank(artifact, binding, parameter.source)
                report["experts"] += parameter.shape[0]
                print(f"{name}: {parameter.shape[0]} experts verified ({time.time() - start:.0f}s)", flush=True)
                continue
            formats = {artifact.object(part["object"]).format for part in _parts(binding)}
            if any(f.startswith("ggml_") for f in formats):
                report["ggml_tensors"] += 1
                ok = verify_ggml_rows(artifact, binding, parameter)
            elif formats == {"q8_g32_fp16"}:
                report["q8_tensors"] += 1
                ok = verify_q8(artifact, binding, parameter)
            else:
                report["tensors"] += 1
                fmt, stored = _binding_values(artifact, binding)
                expected = parameter.source.values()
                ok = torch.equal(_bits(stored), _bits(expected.to(_WORD[fmt][0])))
            if not ok:
                report["tensor_mismatches"].append(name)
        if args.ngram is not None:
            table = text["ngram_table"]
            checked, bad = verify_ngram(qwen4_exp_gguf.Iq4NlTable(store, table["rows"]), args.ngram, table,
                                        args.ngram_rows)
            report["ngram_rows_checked"], report["ngram_row_mismatches"] = checked, bad
        if args.vision_reference is not None:
            reference = stack.enter_context(Artifact(args.vision_reference))
            names = sorted(name for name in bindings if name.startswith("vision/"))
            if not names or set(names) != {n for n in reference.directory.bindings if n.startswith("vision/")}:
                raise ValueError("the vision bindings differ from the reference artifact's")
            mismatches = []
            for name in names:
                fmt, stored = _binding_values(artifact, bindings[name])
                ref_fmt, ref_stored = _binding_values(reference, reference.directory.bindings[name])
                if fmt != ref_fmt or not torch.equal(_bits(stored), _bits(ref_stored)):
                    mismatches.append(name)
            report["vision_reference_tensors"], report["vision_reference_mismatches"] = len(names), mismatches
    report["seconds"] = time.time() - start
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    ok = (not report["expert_mismatches"] and not report["tensor_mismatches"] and not report.get("ngram_row_mismatches")
          and not report.get("vision_reference_mismatches"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
