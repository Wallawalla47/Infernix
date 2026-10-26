"""The ModelOpt MIXED_PRECISION Qwen3.5 MoE import (``qwen3_5_moe_nvfp4_modelopt``; design
docs/maintainer/qwen3_5-122b-offload-design.md §6): the exact ``fp8_e4m3fn_row_f32`` and
``nvfp4_rg16_v1`` encodings, and a whole synthetic checkpoint converted and verified word for word."""

from __future__ import annotations

import json
import struct

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.codecs.fp8_row import (
    decode_fp8_row_scaled_words,
    dequantize_fp8_row_scaled,
    encode_fp8_row_scaled,
)
from tools.artifact.codecs.nvfp4_expert import (
    decode_nvfp4_rg16_words,
    dequantize_nvfp4_mul,
    encode_nvfp4_rg16,
    encode_nvfp4_rg16_rows,
)
from tools.artifact.layouts import rg16_matrix_geometry, row_scale_geometry
from tools.artifact.reader import Artifact
from tools.convert import verify_qwen3_5
from tools.convert.expert_banks import ExpertBankSource
from tools.convert.official_recipes import qwen3_5_moe_nvfp4_modelopt
from tools.convert.pipeline import convert
from tools.convert.qwen3_5 import build_model
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource

F32 = "fp8_e4m3fn_row_f32"


def _fp8_codes(generator, shape):
    codes = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x30  # no NaN codes
    return codes


def _e4m3_scales(generator, shape):
    return torch.randint(0, 0x7F, shape, generator=generator, dtype=torch.uint8)


def test_fp32_row_multipliers_round_trip_exactly():
    generator = torch.Generator().manual_seed(1)
    codes = _fp8_codes(generator, (5, 32))
    scales = torch.tensor([3.0517578e-05, 1.0, 7.0117770e-05, 0.000388, 2.0e-30], dtype=torch.float32)
    payload = encode_fp8_row_scaled(codes, scales, (5, 32), F32)
    geometry = row_scale_geometry(F32, (5, 32))
    assert len(payload) == geometry.payload_bytes == 256 + 5 * 4
    got_codes, got_scales = decode_fp8_row_scaled_words(payload, (5, 32), F32)
    assert torch.equal(got_codes, codes) and torch.equal(got_scales.view(torch.int32), scales.view(torch.int32))
    values = dequantize_fp8_row_scaled(payload, (5, 32), torch.float64, F32)
    assert torch.equal(values, codes.view(torch.float8_e4m3fn).double() * scales.double()[:, None])
    with pytest.raises(ValueError, match="nonnegative finite FP32"):
        encode_fp8_row_scaled(codes, -scales, (5, 32), F32)
    with pytest.raises(TypeError, match="FP32"):
        encode_fp8_row_scaled(codes, scales.to(torch.bfloat16), (5, 32), F32)


def test_dense_rg16_matrix_round_trips_and_row_groups_are_contiguous():
    generator = torch.Generator().manual_seed(2)
    n, k = 64, 48
    codes = torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.uint8)
    scales = _e4m3_scales(generator, (n, k // 16))
    multiplier = torch.tensor(7.011777e-05, dtype=torch.float32)
    payload = encode_nvfp4_rg16(codes, scales, multiplier, (n, k))
    geometry = rg16_matrix_geometry("nvfp4_mul", (n, k))
    assert len(payload) == geometry.payload_bytes
    words = decode_nvfp4_rg16_words(payload, (n, k))
    assert torch.equal(words["codes"], codes) and torch.equal(words["scales"], scales)
    assert words["multiplier"].view(torch.int32) == multiplier.view(torch.int32)
    # A block of whole row groups is the byte run at its first group.
    run = encode_nvfp4_rg16_rows(codes[16:48], scales[16:48])
    begin = 16 // 16 * geometry.blocks * 144
    assert payload[begin : begin + len(run)] == run
    assert torch.equal(
        dequantize_nvfp4_mul(words["codes"], words["scales"], float(words["multiplier"])),
        dequantize_nvfp4_mul(codes, scales, float(multiplier)),
    )


# ---------------------------------------------------------------------------- synthetic checkpoint

H, E, I, SHARED, VOCAB = 32, 2, 16, 16, 32
HEADS, KV, D = 2, 1, 16
KH, KD, VH, VD, TAPS = 1, 16, 2, 16, 4


def _config(*, mtp: bool, per_channel: bool = False) -> dict:
    text = {
        "hidden_size": H,
        "vocab_size": VOCAB,
        "num_hidden_layers": 2,
        "max_position_embeddings": 128,
        "layer_types": ["linear_attention", "full_attention"],
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "head_dim": D,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        "linear_num_key_heads": KH,
        "linear_key_head_dim": KD,
        "linear_num_value_heads": VH,
        "linear_value_head_dim": VD,
        "linear_conv_kernel_dim": TAPS,
        "num_experts": E,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": I,
        "shared_expert_intermediate_size": SHARED,
        "mamba_ssm_dtype": "float32",
    }
    if mtp:
        text["mtp_num_hidden_layers"] = 1
    # dealignai's Ornith-1.5-122B-UNCENSORED-NVFP4: per-channel FP8 projections, shared experts and
    # head, block-FP8 MTP experts.
    layers = ({"lm_head": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}, "e": {"quant_algo": "W4A16_NVFP4"},
               "m": {"quant_algo": "FP8_PB_WO"}} if per_channel else
              {"lm_head": {"quant_algo": "W4A16_NVFP4"}, "a": {"quant_algo": "FP8"}})
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": text,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                "quantized_layers": layers},
    }


class _Tensors:
    def __init__(self, seed: int):
        self.generator = torch.Generator().manual_seed(seed)
        self.tensors: dict[str, torch.Tensor] = {}

    def bf16(self, name, shape):
        self.tensors[name] = (torch.randn(shape, generator=self.generator) * 0.1).to(torch.bfloat16)

    def nvfp4(self, leaf, rows, cols, *, input_scale=False):
        self.tensors[leaf + ".weight"] = torch.randint(0, 256, (rows, cols // 2), generator=self.generator,
                                                       dtype=torch.uint8)
        self.tensors[leaf + ".weight_scale"] = _e4m3_scales(self.generator, (rows, cols // 16)).view(torch.float8_e4m3fn)
        self.tensors[leaf + ".weight_scale_2"] = torch.rand((), generator=self.generator) * 1e-3 + 1e-6
        if input_scale:
            self.tensors[leaf + ".input_scale"] = torch.tensor(0.02, dtype=torch.float32)

    def fp8(self, leaf, rows, cols, input_scale):
        self.tensors[leaf + ".weight"] = _fp8_codes(self.generator, (rows, cols)).view(torch.float8_e4m3fn)
        if input_scale is None:
            # FP8_PER_CHANNEL_PER_TOKEN: an FP32 scale per row, activations quantized at run time.
            self.tensors[leaf + ".weight_scale"] = torch.rand((rows,), generator=self.generator) * 1e-3 + 1e-5
            return
        self.tensors[leaf + ".weight_scale"] = torch.rand((), generator=self.generator) * 1e-3 + 1e-5
        self.tensors[leaf + ".input_scale"] = torch.tensor(input_scale, dtype=torch.float32)

    def fp8_block(self, leaf, rows, cols):
        """FP8_PB_WO: E4M3FN codes and an FP32 multiplier per 128 x 128 tile."""
        self.tensors[leaf + ".weight"] = _fp8_codes(self.generator, (rows, cols)).view(torch.float8_e4m3fn)
        self.tensors[leaf + ".weight_scale_inv"] = (
            torch.rand((-(-rows // 128), -(-cols // 128)), generator=self.generator) * 1e-3 + 1e-5)


def _tensors(*, mtp: bool, w4a4: bool = False, z_scale: float = 0.05,
             per_channel: bool = False) -> dict[str, torch.Tensor]:
    t = _Tensors(7)
    t.bf16("model.language_model.embed_tokens.weight", (VOCAB, H))
    t.bf16("model.language_model.norm.weight", (H,))
    if per_channel:
        t.fp8("lm_head", VOCAB, H, None)
    else:
        t.nvfp4("lm_head", VOCAB, H)
    scale = (lambda value: None) if per_channel else (lambda value: value)
    for layer, kind in enumerate(("linear_attention", "full_attention")):
        p = f"model.language_model.layers.{layer}."
        t.bf16(p + "input_layernorm.weight", (H,))
        t.bf16(p + "post_attention_layernorm.weight", (H,))
        if kind == "linear_attention":
            channels = 2 * KH * KD + VH * VD
            for name, shape in (("A_log", (VH,)), ("dt_bias", (VH,)), ("conv1d.weight", (channels, 1, TAPS)),
                                ("in_proj_a.weight", (VH, H)), ("in_proj_b.weight", (VH, H)),
                                ("norm.weight", (VD,))):
                t.bf16(p + "linear_attn." + name, shape)
            t.fp8(p + "linear_attn.in_proj_qkv", channels, H, scale(0.05))
            t.fp8(p + "linear_attn.in_proj_z", VH * VD, H, scale(z_scale))
            t.fp8(p + "linear_attn.out_proj", H, VH * VD, scale(0.007))
        else:
            t.fp8(p + "self_attn.q_proj", 2 * HEADS * D, H, scale(0.14))
            t.fp8(p + "self_attn.k_proj", KV * D, H, scale(0.14))
            t.fp8(p + "self_attn.v_proj", KV * D, H, scale(0.14))
            t.fp8(p + "self_attn.o_proj", H, HEADS * D, scale(0.006))
            t.bf16(p + "self_attn.q_norm.weight", (D,))
            t.bf16(p + "self_attn.k_norm.weight", (D,))
        t.bf16(p + "mlp.gate.weight", (E, H))
        t.bf16(p + "mlp.shared_expert_gate.weight", (1, H))
        for e in range(E):
            for role, rows, cols in (("gate", I, H), ("up", I, H), ("down", H, I)):
                t.nvfp4(p + f"mlp.experts.{e}.{role}_proj", rows, cols, input_scale=w4a4)
        for role, rows, cols in (("gate", SHARED, H), ("up", SHARED, H), ("down", H, SHARED)):
            if per_channel:
                t.fp8(p + f"mlp.shared_expert.{role}_proj", rows, cols, None)
            else:
                t.nvfp4(p + f"mlp.shared_expert.{role}_proj", rows, cols)
    if mtp and per_channel:
        # The uncensored export's MTP layer: per-channel FP8 dense projections, block-FP8 experts.
        m = "mtp.layers.0."
        t.bf16("mtp.fc.weight", (H, 2 * H))
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            t.bf16(f"mtp.{name}.weight", (H,))
        for name, shape in (("input_layernorm.weight", (H,)), ("post_attention_layernorm.weight", (H,)),
                            ("self_attn.q_norm.weight", (D,)), ("self_attn.k_norm.weight", (D,)),
                            ("mlp.gate.weight", (E, H)), ("mlp.shared_expert_gate.weight", (1, H))):
            t.bf16(m + name, shape)
        for name, rows, cols in (("self_attn.q_proj", 2 * HEADS * D, H), ("self_attn.k_proj", KV * D, H),
                                 ("self_attn.v_proj", KV * D, H), ("self_attn.o_proj", H, HEADS * D),
                                 ("mlp.shared_expert.gate_proj", SHARED, H), ("mlp.shared_expert.up_proj", SHARED, H),
                                 ("mlp.shared_expert.down_proj", H, SHARED)):
            t.fp8(m + name, rows, cols, None)
        for e in range(E):
            for role, rows, cols in (("gate", I, H), ("up", I, H), ("down", H, I)):
                t.fp8_block(m + f"mlp.experts.{e}.{role}_proj", rows, cols)
    elif mtp:
        m = "mtp.layers.0."
        t.bf16("mtp.fc.weight", (H, 2 * H))
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            t.bf16(f"mtp.{name}.weight", (H,))
        for name, shape in (("input_layernorm.weight", (H,)), ("post_attention_layernorm.weight", (H,)),
                            ("self_attn.q_proj.weight", (2 * HEADS * D, H)), ("self_attn.k_proj.weight", (KV * D, H)),
                            ("self_attn.v_proj.weight", (KV * D, H)), ("self_attn.o_proj.weight", (H, HEADS * D)),
                            ("self_attn.q_norm.weight", (D,)), ("self_attn.k_norm.weight", (D,)),
                            ("mlp.gate.weight", (E, H)), ("mlp.shared_expert_gate.weight", (1, H)),
                            ("mlp.shared_expert.gate_proj.weight", (SHARED, H)),
                            ("mlp.shared_expert.up_proj.weight", (SHARED, H)),
                            ("mlp.shared_expert.down_proj.weight", (H, SHARED))):
            t.bf16(m + name, shape)
        for e in range(E):
            for role, shape in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I))):
                t.bf16(m + f"mlp.experts.{e}.{role}_proj.weight", shape)
    return t.tensors


def _checkpoint(path, *, mtp=False, per_channel=False, **options):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config(mtp=mtp, per_channel=per_channel)))
    save_file(_tensors(mtp=mtp, per_channel=per_channel, **options), str(path / "model.safetensors"))
    for role, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return path


def _dflash2(path):
    """A tiny DFlash2 drafter (Ornith-1.5-122B-DFlash2's structure)."""
    path.mkdir()
    taps, groups, rank, inter, heads, kv, d = 2, H // 16, 8, 48, 2, 1, 16
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "hidden_act": "silu",
        "attention_bias": False, "is_causal": False, "hidden_size": H, "vocab_size": VOCAB,
        "num_target_layers": 2, "intermediate_size": inter, "num_attention_heads": heads,
        "num_key_value_heads": kv, "head_dim": d, "num_hidden_layers": 1, "max_position_embeddings": 128,
        "layer_types": ["sliding_attention"], "sliding_window": 8, "rope_parameters": {"rope_theta": 1e7},
        "dflash_config": {"target_layer_ids": [0, 1], "mask_token_id": 3, "conv_kernel_size": taps,
                          "conv_group_size": 16, "selector_rank": rank, "selector_top_k": 4, "block_size": 16},
    }
    (path / "config.json").write_text(json.dumps(config))
    t = _Tensors(11)
    t.bf16("fc.weight", (H, 2 * H))
    t.bf16("hidden_norm.weight", (H,))
    t.bf16("norm.weight", (H,))
    p = "layers.0."
    for name, shape in (("input_layernorm.weight", (H,)), ("post_attention_layernorm.weight", (H,)),
                        ("self_attn.q_proj.weight", (heads * d, H)), ("self_attn.k_proj.weight", (kv * d, H)),
                        ("self_attn.v_proj.weight", (kv * d, H)), ("self_attn.o_proj.weight", (H, heads * d)),
                        ("self_attn.q_norm.weight", (d,)), ("self_attn.k_norm.weight", (d,)),
                        ("mlp.gate_proj.weight", (inter, H)), ("mlp.up_proj.weight", (inter, H)),
                        ("mlp.down_proj.weight", (H, inter))):
        t.bf16(p + name, shape)
    for branch in ("attention_conv", "mlp_conv"):
        t.bf16(p + branch + ".base_kernel", (2, taps, H))
        t.bf16(p + branch + ".kernel_projection.weight", (2 * taps * groups, H))
    t.bf16("candidate_selector.hidden_projection.weight", (rank, H))
    t.bf16("candidate_selector.predecessor_codebook", (VOCAB, rank))
    t.bf16("candidate_selector.successor_codebook", (VOCAB, rank))
    save_file(t.tensors, str(path / "model.safetensors"))
    return path


def test_modelopt_moe_checkpoint_converts_bit_exactly(tmp_path):
    source = _checkpoint(tmp_path / "source", mtp=True)
    drafter = _dflash2(tmp_path / "dflash2")
    out = tmp_path / "out.infernix"
    with SafetensorsSource(source) as store, SafetensorsSource(drafter) as draft:
        model = build_model(store, components=("text", "mtp", "dflash2"), companions={"dflash2": draft})
        bank = model.parameters["text/layers/0/moe/experts"]
        assert isinstance(bank.source, ExpertBankSource) and bank.shape == (E, H, I) and bank.source.weight_only
        assert model.parameters["text/layers/1/moe/shared_expert"].shape == (1, H, SHARED)
        assert "text/layers/1/moe/shared/gate" not in model.parameters
        # Ornith's MTP layer keeps per-expert BF16 tensors: per-expert logical matrices.
        assert "mtp/layers/0/moe/experts/1/down" in model.parameters
        recipe = Recipe(model)
        qwen3_5_moe_nvfp4_modelopt(model, recipe, {"base": store})
        convert(model, recipe, out, device="cpu", rows_per_chunk=16)
    assert verify_qwen3_5.main(["--model", str(source), "--artifact", str(out), "--dflash2", str(drafter)]) == 0
    with Artifact(out) as artifact:
        directory = artifact.directory
        head = artifact.object(directory.bindings["text/output_head"]["object"])
        assert (head.format, head.layout) == ("nvfp4_mul", "nvfp4_rg16_v1")
        uses = {(u["parameter"], u["input"]): u for u in directory.uses}
        query = uses[("text/layers/1/attention/query", "text/layers/1/mixer_input")]
        assert query["activation_policy"] == "AllowA8" and "activation_input_scale" in query["auxiliaries"]
        stored = artifact.read_object(query["auxiliaries"]["activation_input_scale"]["object"])
        assert struct.unpack("<f", stored)[0] == pytest.approx(0.14)
        expert_use = uses[("text/layers/0/moe/experts", "text/layers/0/ffn_input")]
        assert expert_use["activation_policy"] == "A16Only" and "auxiliaries" not in expert_use
        # One fused parent per mixer: GDN q/k/v/z and attention q/k/gate/v.
        gdn = directory.bindings["text/layers/0/gdn/z"]["parts"][0]["object"]
        assert artifact.object(gdn).shape == (2 * KH * KD + 2 * VH * VD, H)
        attention = directory.bindings["text/layers/1/attention/value"]["parts"][0]["object"]
        assert artifact.object(attention).shape == (2 * HEADS * D + 2 * KV * D, H)
        mtp_experts = artifact.object(directory.bindings["mtp/layers/0/moe/experts/0/gate"]["parts"][0]["object"])
        assert mtp_experts.format == "q4_g64_fp16"
        # DFlash2: Q8 projections, BF16 dynamic-convolution and selector projections and codebooks.
        def stored_format(name):
            binding = directory.bindings[name]
            return artifact.object(binding["parts"][0]["object"] if "parts" in binding else binding["object"]).format
        assert stored_format("dflash2/layers/0/attention/query") == "q8_g32_fp16"
        assert stored_format("dflash2/feature_projection") == "q8_g32_fp16"
        assert stored_format("dflash2/layers/0/attention_conv/kernel_projection") == "bf16"
        assert stored_format("dflash2/candidate_selector/predecessor_codebook") == "bf16"
        assert directory.bindings["dflash2/layers/0/attention/context_key"] == \
            directory.bindings["dflash2/layers/0/attention/key"]


def test_per_channel_fp8_export_converts_bit_exactly(tmp_path):
    """dealignai's Ornith-1.5-122B-UNCENSORED-NVFP4 layout: per-row FP32-scaled FP8 projections without
    activation scales (A16Only), dense FP8 shared experts beside the expert bank, an FP8 head, and an MTP
    layer re-quantized from its per-row FP8 projections and block-FP8 experts' exact values."""
    source = _checkpoint(tmp_path / "source", mtp=True, per_channel=True)
    out = tmp_path / "out.infernix"
    with SafetensorsSource(source) as store:
        model = build_model(store, components=("text", "mtp"))
        assert "text/layers/1/moe/shared_expert" not in model.parameters
        assert model.parameters["text/layers/1/moe/shared/gate"].shape == (SHARED, H)
        # The drafter's sources read the checkpoint's exact values: code x FP32 row scale, code x tile
        # multiplier (no BF16 rounding of the scales).
        tensors = _tensors(mtp=True, per_channel=True)
        leaf = "mtp.layers.0.mlp.experts.1.down_proj"
        expected = tensors[leaf + ".weight"].double() * float(tensors[leaf + ".weight_scale_inv"][0, 0])
        assert torch.equal(model.parameters["mtp/layers/0/moe/experts/1/down"].source.values().reshape(H, I),
                           expected)
        leaf = "mtp.layers.0.self_attn.o_proj"
        expected = tensors[leaf + ".weight"].double() * tensors[leaf + ".weight_scale"].double()[:, None]
        assert torch.equal(model.parameters["mtp/layers/0/attention/output"].source.values().reshape(H, HEADS * D),
                           expected)
        recipe = Recipe(model)
        qwen3_5_moe_nvfp4_modelopt(model, recipe, {"base": store})
        convert(model, recipe, out, device="cpu", rows_per_chunk=16)
    assert verify_qwen3_5.main(["--model", str(source), "--artifact", str(out)]) == 0
    with Artifact(out) as artifact:
        directory = artifact.directory

        def stored(name):
            binding = directory.bindings[name]
            return artifact.object(binding["parts"][0]["object"] if "parts" in binding else binding["object"])

        assert (stored("text/output_head").format, stored("text/output_head").shape) == (F32, (VOCAB, H))
        # The shared expert's gate and up form one FP8 parent; down its own.
        assert (stored("text/layers/0/moe/shared/up").format, stored("text/layers/0/moe/shared/up").shape) == \
            (F32, (2 * SHARED, H))
        assert stored("text/layers/0/moe/shared/down").format == F32
        uses = {(u["parameter"], u["input"]): u for u in directory.uses}
        # The mixers' projections permit 8-bit activations with no static scale (one per column at run
        # time); the shared expert and head keep 16-bit activations.
        for key, policy in ((("text/layers/1/attention/query", "text/layers/1/mixer_input"), "AllowA8"),
                            (("text/layers/0/gdn/z", "text/layers/0/mixer_input"), "AllowA8"),
                            (("text/layers/1/attention/output", "text/layers/1/attention/gated_output"), "AllowA8"),
                            (("text/layers/0/moe/shared/gate", "text/layers/0/ffn_input"), "A16Only"),
                            (("text/output_head", "text/final_hidden"), "A16Only")):
            assert uses[key]["activation_policy"] == policy and "auxiliaries" not in uses[key]
        assert stored("mtp/layers/0/moe/experts/0/gate").format == "q4_g64_fp16"
        assert stored("mtp/layers/0/attention/output").format == "q8_g32_fp16"
        assert stored("mtp/layers/0/moe/shared/down").format == "q8_g32_fp16"


def test_recipe_refuses_activation_scaled_experts(tmp_path):
    source = _checkpoint(tmp_path / "source", w4a4=True)
    with SafetensorsSource(source) as store:
        model = build_model(store)
        with pytest.raises(ValueError, match="weight-only"):
            qwen3_5_moe_nvfp4_modelopt(model, Recipe(model), {"base": store})


def test_fused_fp8_parents_need_one_static_activation_scale(tmp_path):
    source = _checkpoint(tmp_path / "source", z_scale=0.06)
    with SafetensorsSource(source) as store:
        model = build_model(store)
        recipe = Recipe(model)
        qwen3_5_moe_nvfp4_modelopt(model, recipe, {"base": store})
        with pytest.raises(ValueError, match="one static activation scale"):
            convert(model, recipe, tmp_path / "out.infernix", device="cpu", rows_per_chunk=16)
