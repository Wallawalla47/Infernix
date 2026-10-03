"""M0 tools of the Flash-Next design: the source-fact inspector and the A4 reference quantizer."""

from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from tools.flash_next.a4_reference import a4_dequantize, a4_quantize, modelopt_rule, synthetic_activations
from tools.flash_next.inspect_checkpoint import inspect, tensor_class


def _checkpoint(tmp_path):
    t = {}
    prefix = "model.language_model.layers.0.mlp.experts."
    for e in range(3):
        for role in ("gate", "up", "down"):
            leaf = f"{prefix}{e}.{role}_proj"
            t[leaf + ".weight"] = torch.zeros(32, 32, dtype=torch.uint8)
            t[leaf + ".weight_scale"] = torch.zeros(32, 4, dtype=torch.uint8).view(torch.float8_e4m3fn)
            t[leaf + ".weight_scale_2"] = torch.tensor([0.25 if e == 0 else 1 / 3], dtype=torch.float32)
            t[leaf + ".input_scale"] = torch.tensor([0.02 if (e == 2 and role == "up") else 0.01], dtype=torch.float32)
    for e in range(2):
        for role in ("gate", "up", "down"):
            leaf = f"mtp.layers.0.mlp.experts.{e}.{role}_proj"
            t[leaf + ".weight"] = torch.zeros(128, 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
            t[leaf + ".weight_scale_inv"] = torch.ones(1, 1)
    for s in range(2):
        t[f"model.language_model.ple.ngram_embedding.shard_{s}.weight"] = torch.zeros(8, 16, dtype=torch.uint8).view(torch.float8_e4m3fn)
        t[f"model.language_model.ple.ngram_embedding.shard_{s}.weight_scale"] = torch.ones((), dtype=torch.bfloat16)
    t["model.language_model.embed_tokens.weight"] = torch.zeros(10, 64, dtype=torch.bfloat16)
    save_file(t, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(
        {"architectures": ["Qwen4ExpForCausalLM"], "text_config": {"num_experts": 3, "hidden_size": 64}}))
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(
        {"quantization": {"quant_algo": "NVFP4", "group_size": 16, "exclude_modules": ["lm_head"]}}))


def test_inspector_records_the_m0_source_facts(tmp_path):
    _checkpoint(tmp_path)
    facts = inspect(tmp_path / "model.safetensors")
    assert facts["model"]["num_experts"] == 3 and facts["model"]["architectures"] == ["Qwen4ExpForCausalLM"]
    assert facts["quantization"]["quantization"]["quant_algo"] == "NVFP4"
    bank = facts["expert_banks"]["model.language_model.layers.0.mlp.experts"]
    assert bank["experts"] == 3 and bank["expert_ids_contiguous"]
    assert bank["weight_scale_2"]["scalar_fp32_matrices"] == 9
    assert bank["weight_scale_2"]["powers_of_two"] == 3
    assert bank["gate_up_input_scale"]["equal_words"] == 2
    assert bank["gate_up_input_scale"]["different"] == 1
    assert bank["gate_up_input_scale"]["examples"][0]["expert"] == 2
    mtp = facts["expert_banks"]["mtp.layers.0.mlp.experts"]
    assert mtp["dtypes"] == {"weight": ["F8_E4M3"], "weight_scale_inv": ["F32"]}
    assert "input_scale" not in mtp  # weight-only FP8: the drafter's activations stay BF16
    ngram = facts["ngram"]["model.language_model.ple.ngram_embedding.shard_*.weight"]
    assert ngram["count"] == 2 and ngram["dtypes"] == ["F8_E4M3"]


def test_tensor_class_hides_indices():
    assert tensor_class("model.layers.12.mlp.experts.300.up_proj.weight") == "model.layers.*.mlp.experts.*.up_proj.weight"
    assert tensor_class("a.ngram.shard_17.weight") == "a.ngram.shard_*.weight"


def _bf16(values):
    return torch.tensor(values, dtype=torch.float32).to(torch.bfloat16)


def test_a4_rounds_ties_to_even_and_saturates():
    # g = 1/2688 and amax = 6 give s = e4m3(6 / (6 g)) = 448, d = 448 g = 1/6.
    g = torch.tensor(1 / 2688, dtype=torch.float32).item()
    c2, s = a4_quantize(_bf16([6.0] + [0.0] * 15), g)
    assert int(s) == 0x7E and c2[0] == 12
    # With g = 2^-10 and amax = 6 * 224 g, s = 224 and d = 224 g = 7/32: every tie value t * d is
    # exact in BF16, so each element divides to exactly the tie.
    g = 2.0**-10
    d = 224 * g
    ties = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    expect = [0, 2, 2, 4, 4, 8, 8]  # doubled E2M1, ties to even
    v = [6 * d] + [t * d for t in ties] + [-t * d for t in ties] + [0.0]
    c2, s = a4_quantize(_bf16(v), g)
    assert float(s.view(torch.float8_e4m3fn).float()) * g == d
    assert c2[0] == 12
    assert c2[1:8].tolist() == expect and c2[8:15].tolist() == [-e for e in expect]


def test_a4_zero_scale_block_is_zero():
    c2, s = a4_quantize(_bf16([1e-30] * 16), 0.05)
    assert int(s) == 0 and not c2.any()


def test_a4_agrees_with_modelopt_outside_its_scale_guard():
    for _, v, g in synthetic_activations(seed=5):
        v = v[:32]
        c2, s = a4_quantize(v, g)
        ours = a4_dequantize(c2, s, g).reshape(*v.shape[:-1], -1, 16)
        theirs, guard = modelopt_rule(v, g)
        theirs = theirs.reshape(ours.shape)
        same = ((ours == theirs) | ((ours == 0) & (theirs == 0))).all(dim=-1)
        assert bool((same | guard).all())
