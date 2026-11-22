"""Inkling2Loop conversion (tools/convert/inkling2_loop.py, recipe ``inkling2_mini``) on a synthetic checkpoint of the
upstream tensor names at small shapes: every stored word comes back unchanged, the value-preserving transforms
(lag-major convs, regrouped cross-pass queries, dropped padding rows, the derived Fold) hold, and the tokenizer
resources are kept byte for byte."""

from __future__ import annotations

import json
from math import prod

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.mxfp8_expert import decode_mxfp8_expert_bank_words
from tools.artifact.codecs.nvfp4_expert import decode_nvfp4_expert_bank_words
from tools.artifact.reader import Artifact
from tools.convert.__main__ import main

H, LAYERS, HEADS, KV, D, REL, EXTENT, K = 64, 6, 4, 2, 16, 4, 8, 4
E, TOP, I, RANK = 4, 2, 32, 8
VOCAB, PADDED = 12, 16
LOCAL = [0, 1, 2, 4]
CORE = (2, 4)
NVFP4_LAYERS = range(4)  # layers 4 and 5 are MXFP8
DRAFT_LAYERS, DRAFT_INTER, MARKOV = 2, 96, 8
# A two-part Split pattern: the Qwen normalizer would rewrite it; Inkling must keep it.
SPLIT = r"[^\r\n\p{L}\p{N}]?\p{L}+|\s+"


def _config() -> dict:
    return {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "eos_token_id": 11,
        "text_config": {
            "model_type": "inkling2_loop_text", "model_max_length": 4096, "hidden_size": H,
            "num_hidden_layers": LAYERS, "vocab_size": PADDED, "unpadded_vocab_size": VOCAB,
            "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": D, "d_rel": REL,
            "rel_extent": EXTENT, "q_bias": False, "o_bias": False, "log_scaling_n_floor": 8192,
            "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6, "use_value_norm_no_gamma": True,
            "qk_l2_clip_threshold": 22.627416997969522, "rel_l2_clip_threshold": 8.0, "local_layer_ids": LOCAL,
            "sconv_kernel_size": K, "final_logit_softcapping": 30.0, "swa_head_dim": D,
            "swa_num_attention_heads": HEADS, "swa_num_key_value_heads": KV, "sliding_window_size": 8,
            "n_layers_per_attn_res_block": 2, "attn_res_normalize_values": True, "n_routed_experts": E,
            "num_experts_per_tok": TOP, "n_shared_experts": 0, "intermediate_size": I, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": True,
            "loop_config": {"core_start": CORE[0], "core_end": CORE[1], "num_passes": 2, "gate_rank": RANK,
                            "is_varloop": False},
        },
        "mtp_config": {
            "hidden_size": H, "num_hidden_layers": DRAFT_LAYERS, "num_attention_heads": HEADS,
            "num_key_value_heads": KV, "head_dim": D, "intermediate_size": DRAFT_INTER, "rms_norm_eps": 1e-6,
            "rope_theta": 1e6, "local_layer_ids": [0], "sliding_window_size": 8, "block_size": 4,
            "markov_rank": MARKOV, "markov_head_type": "rnn_hidden", "q_bias": False, "o_bias": False,
            # visits = (core_end - core_start) * (passes - 1) + layers = 8
            "target_layer_ids": [1, 3, 5, 8],
        },
    }


def _tokenizer() -> dict:
    return {
        "version": "1.0",
        "added_tokens": [
            {"id": 10, "content": "<|endoftext|>", "special": True},
            {"id": 11, "content": "<|content_model_end_sampling|>", "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ]},
        "model": {"type": "BPE", "vocab": {c: i for i, c in enumerate("abcdefghij")}, "merges": [],
                  "ignore_merges": True},
    }


def _bf16(*shape, generator) -> torch.Tensor:
    return torch.randn(*shape, generator=generator).to(torch.bfloat16)


def _bytes(*shape, high, generator) -> torch.Tensor:
    return torch.randint(0, high, shape, generator=generator, dtype=torch.uint8)


def _checkpoint(tmp_path, *, zero_amax: bool = False):
    g = torch.Generator().manual_seed(7)
    t, p = {}, "model.llm."
    t[p + "embed.weight"] = _bf16(PADDED, H, generator=g)
    t[p + "unembed.weight"] = _bf16(PADDED, H, generator=g)
    for name in ("embed_norm.weight", "norm.weight"):
        t[p + name] = _bf16(H, generator=g)
    t[p + "attn_res.output_query"] = torch.randn(H, generator=g)
    for i in range(LAYERS):
        full, sp, sa = i not in LOCAL, f"{p}layers.{i}.", f"{p}layers.{i}.attn."
        if i > 0:
            t[f"{p}attn_res.attn_queries.{i - 1}"] = torch.randn(*((3, H) if full else (H,)), generator=g)
        t[f"{p}attn_res.mlp_queries.{i}"] = torch.randn(H, generator=g)
        for leaf in (("q_input_norm", "k_input_norm", "v_input_norm") if full else ("attn_norm",)) + ("mlp_norm",):
            t[f"{sp}{leaf}.weight"] = _bf16(H, generator=g)
        for leaf, rows in (("wq_du", HEADS * D), ("wr_du", HEADS * REL), ("gate_attn", HEADS * D),
                           ("wk_dv", KV * D), ("wv_dv", KV * D), ("attn_sink_proj", HEADS)):
            t[f"{sa}{leaf}.weight"] = _bf16(rows, H, generator=g)
        t[sa + "wo_ud.weight"] = _bf16(H, HEADS * D, generator=g)
        t[sa + "q_l2_clip.scale"] = _bf16(HEADS, D, generator=g)
        t[sa + "rel_l2_clip.scale"] = _bf16(HEADS, REL, generator=g)
        t[sa + "rel_logits_proj.proj"] = _bf16(REL, EXTENT, generator=g)
        for leaf in ("k_sconv", "v_sconv"):
            if full:
                t[f"{sa}{leaf}.static_w"] = _bf16(K, KV * D, generator=g)
                t[f"{sa}{leaf}.weight_proj"] = _bf16(KV * K, H, generator=g)
            else:
                t[f"{sa}{leaf}.weight"] = _bf16(KV * D, 1, K, generator=g)
        t[sp + "attn_sconv.weight"] = _bf16(H, 1, K, generator=g)
        t[sp + "mlp_sconv.weight"] = _bf16(H, 1, K, generator=g)
        if CORE[0] <= i < CORE[1]:
            for kind in ("attn", "k", "v", "mlp"):
                c = f"{sp}controllers.{kind}."
                t[c + "down_weight"] = _bf16(RANK, 2 * H, generator=g)
                t[c + "forget_weight"] = _bf16(H, RANK, generator=g)
                t[c + "gain_weight"] = _bf16(H, RANK, generator=g)
        m = sp + "mlp."
        t[m + "gate.weight"] = _bf16(E, H, generator=g)
        t[m + "gate.bias"] = torch.randn(E, generator=g)
        x = m + "experts."
        if i in NVFP4_LAYERS:
            t[x + "w13_weight"] = _bytes(E, 2 * I, H // 2, high=256, generator=g)
            t[x + "w13_weight_scale"] = _bytes(E, 2 * I, H // 16, high=0x7F, generator=g).view(torch.float8_e4m3fn)
            t[x + "w2_weight"] = _bytes(E, H, I // 2, high=256, generator=g)
            t[x + "w2_weight_scale"] = _bytes(E, H, I // 16, high=0x7F, generator=g).view(torch.float8_e4m3fn)
            t[x + "w13_weight_amax"] = torch.rand(E, generator=g) + 0.05
            t[x + "w2_weight_amax"] = torch.rand(E, generator=g) + 0.05
            t[x + "w13_weight_outer_scale_denominator"] = torch.tensor([2688.0])
            t[x + "w2_weight_outer_scale_denominator"] = torch.tensor([2688.0])
            if zero_amax and i == 1:
                t[x + "w2_weight_amax"][2] = 0.0
        else:
            t[x + "w13_weight"] = _bytes(E, 2 * I, H, high=0x7F, generator=g).view(torch.float8_e4m3fn)
            t[x + "w13_weight_scale"] = _bytes(E, 2 * I, H // 32, high=240, generator=g) + 3
            t[x + "w2_weight"] = _bytes(E, H, I, high=0x7F, generator=g).view(torch.float8_e4m3fn)
            t[x + "w2_weight_scale"] = _bytes(E, H, I // 32, high=240, generator=g) + 3
    loop_layers = CORE[1] - CORE[0]
    t[p + "middle_cycle_cross_pass_queries"] = torch.randn(2 * loop_layers, H, generator=g)
    t[p + "middle_cycle_cross_pass_kv_queries"] = torch.randn(2 * loop_layers, H, generator=g)
    save_file(t, str(tmp_path / "model.safetensors"))

    d = {"fuse_weight": _bf16(4, 4, generator=g), "mask_embed": _bf16(H, generator=g), "norm.weight": _bf16(H, generator=g)}
    for i in range(DRAFT_LAYERS):
        sp, sa = f"layers.{i}.", f"layers.{i}.self_attn."
        for leaf in ("dflare_norm", "input_layernorm", "post_attention_layernorm"):
            d[f"{sp}{leaf}.weight"] = _bf16(H, generator=g)
        for leaf, rows in (("q_proj", HEADS * D), ("k_proj", KV * D), ("v_proj", KV * D), ("k_proj_ctx", KV * D),
                           ("v_proj_ctx", KV * D)):
            d[f"{sa}{leaf}.weight"] = _bf16(rows, H, generator=g)
        for leaf in ("q_norm", "k_norm", "k_norm_ctx"):
            d[f"{sa}{leaf}.weight"] = _bf16(D, generator=g)
        d[sa + "o_proj.weight"] = _bf16(H, HEADS * D, generator=g)
        d[sp + "mlp.gate_proj.weight"] = _bf16(DRAFT_INTER, H, generator=g)
        d[sp + "mlp.up_proj.weight"] = _bf16(DRAFT_INTER, H, generator=g)
        d[sp + "mlp.down_proj.weight"] = _bf16(H, DRAFT_INTER, generator=g)
    mk = "markov_head."
    d[mk + "dflare_norm.weight"] = _bf16(H, generator=g)
    d[mk + "hidden_out.weight"] = _bf16(H, MARKOV, generator=g)
    d[mk + "hidden_out.bias"] = _bf16(H, generator=g)
    d[mk + "markov_w1.weight"] = _bf16(PADDED, MARKOV, generator=g)
    d[mk + "rnn_proj.weight"] = _bf16(3 * MARKOV, 2 * MARKOV + 2 * H, generator=g)
    d[mk + "rnn_proj.bias"] = _bf16(3 * MARKOV, generator=g)
    save_file(d, str(tmp_path / "mtp.safetensors"))

    (tmp_path / "config.json").write_text(json.dumps(_config()), encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text(json.dumps(_tokenizer()), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|content_model_end_sampling|>",
                                                                "pad_token": "<|endoftext|>"}), encoding="utf-8")
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 11}), encoding="utf-8")
    return t, d


def _convert(tmp_path, **options):
    source = tmp_path / "source"
    source.mkdir()
    tensors, drafter = _checkpoint(source, **options)
    out = tmp_path / "model.infernix"
    main(["--model", str(source), "--recipe", "inkling2_mini", "--components", "text,dspark", "--out", str(out),
          "--device", "cpu"])
    return source, tensors, drafter, out


def _bound(artifact: Artifact, name: str) -> bytes:
    binding = artifact.directory.bindings[name]
    if "object" in binding:
        return artifact.read_object(binding["object"])
    parts = []
    for part in binding["parts"]:
        obj = artifact.object(part["object"])
        width = len(artifact.read_object(obj.id)) // prod(obj.shape)
        first, last = part["range"]
        parts.append(artifact.read_object(obj.id)[first * width:last * width])
    return b"".join(parts)


def _words(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def test_every_word_converts_exactly(tmp_path):
    source, t, d, out = _convert(tmp_path)
    p = "model.llm."
    with Artifact(out) as artifact:
        config = artifact.directory.components["text"]["config"]
        assert config["architectures"] == ["Inkling2LoopForCausalLM"] and config["vocab_size"] == VOCAB
        assert config["expert_formats"] == ["nvfp4_four_over_six"] * 4 + ["mxfp8"] * 2
        assert artifact.directory.components["dspark"]["config"]["target_layer_ids"] == [1, 3, 5, 8]

        # Rows past the real vocabulary are dropped; everything else word for word.
        assert _bound(artifact, "text/token_embedding") == _words(t[p + "embed.weight"][:VOCAB])
        assert _bound(artifact, "text/output_head") == _words(t[p + "unembed.weight"][:VOCAB])
        assert _bound(artifact, "dspark/markov/embedding") == _words(d["markov_head.markov_w1.weight"][:VOCAB])
        assert _bound(artifact, "text/output_query") == _words(t[p + "attn_res.output_query"])
        assert _bound(artifact, "text/layers/3/attention_mix_query") == _words(t[p + "attn_res.attn_queries.2"])
        assert _bound(artifact, "text/layers/2/attention/key") == _words(t[p + "layers.2.attn.wk_dv.weight"])
        assert _bound(artifact, "text/layers/3/attention/value_taps") == _words(t[p + "layers.3.attn.v_sconv.weight_proj"])
        assert _bound(artifact, "text/layers/3/messages/key/down") == _words(t[p + "layers.3.controllers.k.down_weight"])

        # Static convs lag-major (row 0 = the current token = the checkpoint's last tap); dynamic static taps as stored.
        conv = t[p + "layers.0.attn_sconv.weight"].reshape(H, K)
        assert _bound(artifact, "text/layers/0/attention_conv") == _words(conv.flip(1).t())
        assert _bound(artifact, "text/layers/3/attention/key_conv") == _words(t[p + "layers.3.attn.k_sconv.static_w"])

        # Cross-pass queries regrouped per loop layer in (query, key, value, mlp) order.
        q, kv = t[p + "middle_cycle_cross_pass_queries"], t[p + "middle_cycle_cross_pass_kv_queries"]
        expected = torch.stack((q[2], kv[2], kv[3], q[3]))
        assert _bound(artifact, "text/layers/3/message_queries") == _words(expected)

        # NVFP4 banks: the checkpoint's codes and scales, gate rows 0::2 and up rows 1::2; the exact amax and
        # denominators; the multiplier plane fl32(amax / denominator).
        x = p + "layers.1.mlp.experts."
        words = decode_nvfp4_expert_bank_words(_bound(artifact, "text/layers/1/moe/experts"), (E, H, I))
        assert torch.equal(words["gate_codes"], t[x + "w13_weight"][:, 0::2])
        assert torch.equal(words["up_codes"], t[x + "w13_weight"][:, 1::2])
        assert torch.equal(words["up_scales"], t[x + "w13_weight_scale"].view(torch.uint8)[:, 1::2])
        assert torch.equal(words["down_codes"], t[x + "w2_weight"])
        assert torch.equal(words["down_scales"], t[x + "w2_weight_scale"].view(torch.uint8))
        den = torch.tensor(2688.0)
        assert torch.equal(words["multipliers"][:, 0], t[x + "w13_weight_amax"] / den)
        assert torch.equal(words["multipliers"][:, 2], t[x + "w2_weight_amax"] / den)
        amax = torch.stack((t[x + "w13_weight_amax"], t[x + "w2_weight_amax"]), dim=1)
        assert _bound(artifact, "text/layers/1/moe/expert_amax") == _words(amax)
        assert _bound(artifact, "text/layers/1/moe/expert_denominators") == _words(torch.tensor([2688.0, 2688.0]))

        # MXFP8 banks: codes and UE8M0 scales unchanged.
        x = p + "layers.5.mlp.experts."
        words = decode_mxfp8_expert_bank_words(_bound(artifact, "text/layers/5/moe/experts"), (E, H, I))
        assert torch.equal(words["gate_codes"], t[x + "w13_weight"].view(torch.uint8)[:, 0::2])
        assert torch.equal(words["up_scales"], t[x + "w13_weight_scale"][:, 1::2])
        assert torch.equal(words["down_codes"], t[x + "w2_weight"].view(torch.uint8))
        assert torch.equal(words["down_scales"], t[x + "w2_weight_scale"])
        assert "text/layers/5/moe/expert_amax" not in artifact.directory.bindings

        # The activation policies permit the trained W4A4 / W8A8 arithmetic.
        policies = {(u["parameter"], u["input"]): u.get("activation_policy") for u in artifact.directory.uses}
        assert policies[("text/layers/1/moe/experts", "text/layers/1/moe/input")] == "AllowA4"
        assert policies[("text/layers/5/moe/experts", "text/layers/5/moe/input")] == "AllowA8"

        # The Fold as refresh_fold derives it: FP64 head rows times hidden_out, copied into BF16.
        head = t[p + "unembed.weight"][:VOCAB].double()
        fold = torch.empty(VOCAB, MARKOV, dtype=torch.bfloat16)
        fold.copy_(head @ d["markov_head.hidden_out.weight"].double())
        fold_bias = torch.empty(VOCAB, dtype=torch.bfloat16)
        fold_bias.copy_(head @ d["markov_head.hidden_out.bias"].double())
        assert _bound(artifact, "dspark/markov/fold") == _words(fold)
        assert _bound(artifact, "dspark/markov/fold_bias") == _words(fold_bias)

        # Tokenizer resources byte for byte (no Qwen normalization).
        for role in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
            refs = artifact.directory.components["text"]["resources"]
            assert artifact.read_object(refs[role]) == (source / role).read_bytes()


def test_the_independent_verifier_accepts_the_artifact(tmp_path):
    from tools.convert.verify_inkling2 import verify

    source, _, _, out = _convert(tmp_path)
    result = verify(source, out)
    assert (result["nvfp4_banks"], result["mxfp8_banks"], result["derived"]) == (4, 2, 2)


def test_a_nonpositive_amax_is_refused(tmp_path):
    with pytest.raises(ValueError, match="finite and positive"):
        _convert(tmp_path, zero_amax=True)


VH, VHEADS, VLAYERS, VPATCH, VMERGE, VGRID, VPROJ = 16, 2, 2, 2, 2, 4, 24


def _vision_checkpoint(source):
    """Add a small tower (model.visual.*) and processor_config.json to a synthetic checkpoint."""
    g = torch.Generator().manual_seed(11)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["vision_config"] = {
        "model_type": "inkling2_vision", "text_hidden_size": H, "temporal_patch_size": 2,
        "native_patch_size": VPATCH, "adapter_merge_size": VMERGE, "projection_hidden_dim": VPROJ,
        "projector_bias": False, "hidden_size": VH, "num_hidden_layers": VLAYERS, "num_attention_heads": VHEADS,
        "mlp_ratio": 2.0, "posemb_grid_size": VGRID, "layer_norm_eps": 1e-5,
    }
    (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
    side = VPATCH * VMERGE
    (source / "processor_config.json").write_text(json.dumps({
        "image_processor": {"do_convert_rgb": True, "do_normalize": True, "do_rescale": True,
                            "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Inkling2ImageProcessor",
                            "image_std": [0.5, 0.5, 0.5], "rescale_factor": 0.00392156862745098,
                            "size": {"height": side, "width": side}},
        "image_token": "<|unused_200054|>", "processor_class": "Inkling2Processor"}), encoding="utf-8")
    from safetensors.torch import load_file
    t = load_file(str(source / "model.safetensors"))
    v, e = "model.visual.", "model.visual.backbone.embeddings."
    t[e + "patch_embedding.weight"] = _bf16(VH, 3 * VPATCH * VPATCH, generator=g)
    t[e + "position_embedding"] = _bf16(VGRID * VGRID, VH, generator=g)
    t[e + "class_embedding"] = _bf16(VH, generator=g)
    t[e + "cls_position_embedding"] = _bf16(VH, generator=g)
    for leaf in ("weight", "bias"):
        t[f"{v}backbone.ln_pre.{leaf}"] = _bf16(VH, generator=g)
    for i in range(VLAYERS):
        sp = f"{v}backbone.layers.{i}."
        for leaf in ("ln_1", "ln_2"):
            t[f"{sp}{leaf}.weight"] = _bf16(VH, generator=g)
            t[f"{sp}{leaf}.bias"] = _bf16(VH, generator=g)
        for leaf in ("q_proj", "k_proj", "v_proj", "out_proj"):
            t[f"{sp}self_attn.{leaf}.weight"] = _bf16(VH, VH, generator=g)
            t[f"{sp}self_attn.{leaf}.bias"] = _bf16(VH, generator=g)
        t[sp + "ls_1.gamma"] = _bf16(VH, generator=g)
        t[sp + "ls_2.gamma"] = _bf16(VH, generator=g)
        t[sp + "mlp.c_fc.weight"] = _bf16(2 * VH, VH, generator=g)
        t[sp + "mlp.c_fc.bias"] = _bf16(2 * VH, generator=g)
        t[sp + "mlp.c_proj.weight"] = _bf16(VH, 2 * VH, generator=g)
        t[sp + "mlp.c_proj.bias"] = _bf16(VH, generator=g)
    t[v + "proj_in.weight"] = _bf16(VPROJ, VMERGE * VMERGE * VH, generator=g)
    t[v + "proj_out.weight"] = _bf16(H, VPROJ, generator=g)
    save_file(t, str(source / "model.safetensors"))
    return t


def test_the_vision_tower_converts_word_for_word(tmp_path):
    from tools.convert.verify_inkling2 import verify

    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    t = _vision_checkpoint(source)
    out = tmp_path / "model.infernix"
    main(["--model", str(source), "--recipe", "inkling2_mini", "--components", "text,dspark,vision", "--out",
          str(out), "--device", "cpu"])
    v, e = "model.visual.", "model.visual.backbone.embeddings."
    with Artifact(out) as artifact:
        config = artifact.directory.components["vision"]["config"]
        assert (config["hidden_size"], config["num_hidden_layers"], config["intermediate_size"]) == (VH, VLAYERS, 2 * VH)
        assert (config["patch_size"], config["spatial_merge_size"], config["position_grid"]) == (VPATCH, VMERGE, VGRID)
        assert (config["output_size"], config["rope_theta"], config["image_token"]) == (H, 10000.0, "<|unused_200054|>")
        assert _bound(artifact, "vision/patch_embedding") == _words(t[e + "patch_embedding.weight"])
        assert _bound(artifact, "vision/position_embedding") == _words(t[e + "position_embedding"])
        assert _bound(artifact, "vision/class_row") == _words(t[e + "class_embedding"] + t[e + "cls_position_embedding"])
        assert _bound(artifact, "vision/layers/1/attention/key") == _words(t[v + "backbone.layers.1.self_attn.k_proj.weight"])
        assert _bound(artifact, "vision/layers/1/attention/value_bias") == _words(t[v + "backbone.layers.1.self_attn.v_proj.bias"])
        assert _bound(artifact, "vision/layers/0/mlp_scale") == _words(t[v + "backbone.layers.0.ls_2.gamma"])
        assert _bound(artifact, "vision/projector/out") == _words(t[v + "proj_out.weight"])
        # q/k/v (and their biases) share one parent, in that order: the fused projection is one GEMM.
        parents = [artifact.directory.bindings[f"vision/layers/0/attention/{r}"] for r in ("query", "key", "value")]
        assert all("parts" in b and len(b["parts"]) == 1 for b in parents)
        assert len({b["parts"][0]["object"] for b in parents}) == 1
        assert [b["parts"][0]["range"][0] for b in parents] == [0, VH * VH, 2 * VH * VH]
        refs = artifact.directory.components["vision"]["resources"]
        assert artifact.read_object(refs["processor_config.json"]) == (source / "processor_config.json").read_bytes()
    result = verify(source, out)
    assert result["derived"] == 3 and "unconverted" not in result


def test_a_different_image_processor_is_refused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source)
    _vision_checkpoint(source)
    processor = json.loads((source / "processor_config.json").read_text(encoding="utf-8"))
    processor["image_processor"]["image_mean"] = [0.48145466, 0.4578275, 0.40821073]
    (source / "processor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    with pytest.raises(ValueError, match="image_mean"):
        main(["--model", str(source), "--recipe", "inkling2_mini", "--components", "text,vision", "--out",
              str(tmp_path / "model.infernix"), "--device", "cpu"])
