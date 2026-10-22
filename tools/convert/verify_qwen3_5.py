"""Word-equality qualification of a Qwen3.5 MoE artifact converted from a ModelOpt export
(``qwen3_5_moe_nvfp4_modelopt``; design docs/maintainer/qwen3_5-122b-offload-design.md §6).

    python -m tools.convert.verify_qwen3_5 --model CHECKPOINT --artifact OUT.infernix \
        [--dflash2 DRAFTER] [--report FILE]

Every logical parameter the recipe stores as the checkpoint has it must equal the checkpoint's words:

- each routed and shared expert's E2M1 codes, E4M3 block scales and FP32 ``weight_scale_2``
  multipliers, decoded from its ``nvfp4_expert_rg16_v1`` record;
- the output head's codes, block scales and multiplier, decoded from ``nvfp4_rg16_v1``;
- every ``fp8_e4m3fn_row_f32`` projection's E4M3FN codes and FP32 row multipliers (its matrix's
  ``weight_scale`` in each row), and each of its Uses' ``activation_input_scale`` (the matrix's
  ``input_scale`` word);
- every BF16 and FP32 tensor bit for bit, and every FP32 widening of a BF16 tensor exactly.

Parameters the recipe re-quantizes (the MTP drafter) are counted per component, not compared.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import struct
import time

import torch

from tools.artifact.codecs.fp8_row import decode_fp8_row_scaled_words
from tools.artifact.codecs.nvfp4_expert import decode_nvfp4_expert_bank_words, decode_nvfp4_rg16_words
from tools.artifact.reader import Artifact
from tools.convert.expert_banks import ExpertBankSource
from tools.convert.qwen3_5 import build_model
from tools.convert.sources.modelopt import nvfp4_matrix_words
from tools.convert.sources.safetensors import SafetensorsSource

_FP8_F32 = "fp8_e4m3fn_row_f32"
_WORD = {"bf16": (torch.bfloat16, 2), "fp32": (torch.float32, 4), "int32": (torch.int32, 4)}
_REQUANTIZED = {"q8_g32_fp16", "q4_g64_fp16", "q2_g64_fp16"}


def _parts(binding: dict) -> list[dict]:
    return binding["parts"] if "parts" in binding else [{"object": binding["object"]}]


def _binding_formats(artifact: Artifact, binding: dict) -> set[str]:
    return {artifact.object(part["object"]).format for part in _parts(binding)}


def _binding_values(artifact: Artifact, binding: dict) -> tuple[str, torch.Tensor]:
    formats, chunks = set(), []
    for part in _parts(binding):
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


def verify_bank(artifact: Artifact, binding: dict, source: ExpertBankSource) -> int:
    """Mismatching matrices of one bank (every expert's gate, up and down)."""

    obj = artifact.object(binding["object"])
    if tuple(obj.shape) != source.shape or obj.layout != "nvfp4_expert_rg16_v1":
        raise ValueError(f"{source.label}: stored bank differs from the checkpoint's shape or layout")
    words = decode_nvfp4_expert_bank_words(artifact.read_object(obj.id), source.shape)
    e, h, i = source.shape
    bad = 0
    for x in range(e):
        for column, (role, rows, cols) in enumerate((("gate", i, h), ("up", i, h), ("down", h, i))):
            src = nvfp4_matrix_words(source.store, source.leaf(x, role), (rows, cols))
            ok = (
                torch.equal(words[f"{role}_codes"][x], src.codes)
                and torch.equal(words[f"{role}_scales"][x], src.scales)
                and torch.equal(words["multipliers"][x, column].view(torch.int32), src.weight_scale_2.view(torch.int32))
            )
            bad += not ok
    return bad


def verify_head(artifact: Artifact, binding: dict, parameter, store) -> bool:
    obj = artifact.object(binding["object"])
    if obj.layout != "nvfp4_rg16_v1" or tuple(obj.shape) != parameter.shape:
        return False
    words = decode_nvfp4_rg16_words(artifact.read_object(obj.id), obj.shape)
    expected = parameter.source_factory(store, "nvfp4_mul").read_encoded(0, parameter.shape[0])
    return (
        torch.equal(words["codes"], expected.codes.reshape(words["codes"].shape))
        and torch.equal(words["scales"], expected.scales)
        and struct.pack("<f", float(words["multiplier"])) == expected.weight_multiplier
    )


def verify_fp8_rows(artifact: Artifact, store, binding: dict, parameter) -> bool:
    """The bound rows' E4M3FN codes and FP32 row multipliers equal the source's encoded rows."""

    n, k = parameter.shape
    expected = parameter.source_factory(store, _FP8_F32).read_encoded(0, n)
    codes, scales = [], []
    for part in _parts(binding):
        obj = artifact.object(part["object"])
        c, s = decode_fp8_row_scaled_words(artifact.read_object(obj.id), obj.shape, _FP8_F32)
        begin, end = part.get("range", (0, obj.shape[0] * obj.shape[1]))
        codes.append(c[begin // k : end // k])
        scales.append(s[begin // k : end // k])
    return torch.equal(torch.cat(codes), expected.codes.reshape(n, k)) and torch.equal(
        torch.cat(scales).view(torch.int32), expected.scales.to(torch.float32).view(torch.int32))


def verify_input_scales(artifact: Artifact, store, model, uses: list[dict]) -> tuple[int, int]:
    """Each static A8 Use's ``activation_input_scale`` equals its matrix's ``input_scale`` word."""

    checked = bad = 0
    for use in uses:
        binding = use.get("auxiliaries", {}).get("activation_input_scale")
        if binding is None:
            continue
        parameter = model.parameters[use["parameter"]]
        stored = artifact.read_object(binding["object"])
        expected = parameter.source_factory(store, _FP8_F32).input_scale()
        checked += 1
        bad += stored != expected
    return checked, bad


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dflash2", type=Path, help="the DFlash2 drafter checkpoint, when the artifact has one")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    start = time.time()
    report = {"experts": 0, "expert_mismatches": 0, "tensors": 0, "tensor_mismatches": [],
              "input_scales": 0, "input_scale_mismatches": 0, "requantized": {}}
    with ExitStack() as stack:
        store = stack.enter_context(SafetensorsSource(args.model))
        artifact = stack.enter_context(Artifact(args.artifact))
        components = tuple(artifact.directory.components)
        companions = {}
        if "dflash2" in components:
            if args.dflash2 is None:
                raise ValueError("the artifact holds a DFlash2 drafter: give its checkpoint with --dflash2")
            companions["dflash2"] = stack.enter_context(SafetensorsSource(args.dflash2))
        model = build_model(store, components=components, companions=companions)
        bindings = {k: v for k, v in artifact.directory.bindings.items() if not k.startswith("proposal/")}
        report["proposal_bindings"] = len(artifact.directory.bindings) - len(bindings)
        if set(model.parameters) != set(bindings):
            missing = sorted(set(model.parameters) - set(bindings))[:5]
            extra = sorted(set(bindings) - set(model.parameters))[:5]
            raise ValueError(f"artifact bindings differ from the logical model: {missing} {extra}")
        for name, parameter in model.parameters.items():
            binding = bindings[name]
            if isinstance(parameter.source, ExpertBankSource):
                report["expert_mismatches"] += verify_bank(artifact, binding, parameter.source)
                report["experts"] += parameter.source.shape[0]
                print(f"{name}: {parameter.source.shape[0]} experts verified ({time.time() - start:.0f}s)", flush=True)
                continue
            formats = _binding_formats(artifact, binding)
            if formats & _REQUANTIZED:
                component = name.split("/", 1)[0]
                report["requantized"][component] = report["requantized"].get(component, 0) + 1
                continue
            report["tensors"] += 1
            if formats == {"nvfp4_mul"}:
                if not verify_head(artifact, binding, parameter, store):
                    report["tensor_mismatches"].append(name)
                continue
            if formats == {_FP8_F32}:
                if not verify_fp8_rows(artifact, store, binding, parameter):
                    report["tensor_mismatches"].append(name)
                continue
            fmt, stored = _binding_values(artifact, binding)
            expected = parameter.source.values()
            # BF16 and FP32 sources: kept bit for bit, or BF16 widened exactly to FP32.
            if expected.dtype not in (torch.bfloat16, torch.float32) or not torch.equal(
                _bits(stored), _bits(expected.to(_WORD[fmt][0]))
            ) or (expected.dtype == torch.float32 and fmt != "fp32"):
                report["tensor_mismatches"].append(name)
        report["input_scales"], report["input_scale_mismatches"] = verify_input_scales(
            artifact, store, model, list(artifact.directory.uses)
        )
    report["seconds"] = time.time() - start
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    ok = not report["expert_mismatches"] and not report["tensor_mismatches"] and not report["input_scale_mismatches"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
