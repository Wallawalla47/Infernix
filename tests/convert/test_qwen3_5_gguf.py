"""Qwen3.5 dense import from a llama.cpp ``qwen35`` GGUF (tools/convert/qwen3_5_gguf.py).

A small BF16 checkpoint is exported as llama.cpp's converter exports it (written here
independently: +1 on the zero-centred norms, value heads in ggml's tiled order, ``-exp(A_log)``,
the squeezed convolution, Q8_0 matrices by gguf-py's quantizer), imported back, and compared with
the checkpoint's own logical model: the same parameters, inputs and packing groups, every plain
value exact, every matrix the Q8_0 quantization of the checkpoint's matrix in Hugging Face order.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

gguf = pytest.importorskip("gguf")

from tools.convert import qwen3_5, qwen3_5_gguf
from tools.convert.methods import cast_direct, import_encoded
from tools.convert.official_recipes import qwen3_5_gguf_q8_0
from tools.convert.recipe import Recipe
from tools.convert.sources.gguf_logical import value_head_columns
from tools.convert.sources.gguf_raw import GgufStore
from tools.convert.sources.logical import array_source
from tools.convert.sources.safetensors import SafetensorsSource

H, V, I = 64, 8, 96
HEADS, KV, D = 2, 1, 32
NK, DK, NV, DV, TAPS = 2, 32, 4, 32, 4
KG, VG = NK * DK, NV * DV
CHANNELS = 2 * KG + VG
Q8 = "q8_g32_fp16"


def _config() -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "hidden_size": H, "vocab_size": V, "num_hidden_layers": 2, "max_position_embeddings": 128,
            "layer_types": ["linear_attention", "full_attention"], "full_attention_interval": 2,
            "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": D, "rms_norm_eps": 1e-6,
            "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.25,
                                "rope_theta": 10_000_000, "rope_type": "default"},
            "linear_num_key_heads": NK, "linear_key_head_dim": DK, "linear_num_value_heads": NV,
            "linear_value_head_dim": DV, "linear_conv_kernel_dim": TAPS, "mamba_ssm_dtype": "float32",
            "intermediate_size": I, "tie_word_embeddings": False,
        },
    }


def _directory(path, tensors=None):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config()))
    if tensors is not None:
        save_file(tensors, path / "model.safetensors")
    for role, value in {"tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
                        "tokenizer_config.json": {}, "generation_config.json": {}}.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return SafetensorsSource(path)


def _bf16(rng, *shape, scale=1.0, low=None):
    values = torch.from_numpy(rng.standard_normal(shape).astype(np.float32) * scale)
    if low is not None:  # keep |g| away from 0 so g + 1 is exact in binary32, as for real gammas
        values = torch.where(values.abs() < low, torch.full_like(values, low), values)
    return values.to(torch.bfloat16)


def _checkpoint(rng) -> dict[str, torch.Tensor]:
    t = {"model.language_model.embed_tokens.weight": _bf16(rng, V, H), "lm_head.weight": _bf16(rng, V, H),
         "model.language_model.norm.weight": _bf16(rng, H, scale=0.1, low=2**-10)}
    for i in range(2):
        p = f"model.language_model.layers.{i}."
        t[p + "input_layernorm.weight"] = _bf16(rng, H, scale=0.1, low=2**-10)
        t[p + "post_attention_layernorm.weight"] = _bf16(rng, H, scale=0.1, low=2**-10)
        t[p + "mlp.gate_proj.weight"] = _bf16(rng, I, H)
        t[p + "mlp.up_proj.weight"] = _bf16(rng, I, H)
        t[p + "mlp.down_proj.weight"] = _bf16(rng, H, I)
    a = "model.language_model.layers.0.linear_attn."
    t[a + "A_log"] = torch.from_numpy(rng.uniform(0.1, 2.7, NV).astype(np.float32)).to(torch.bfloat16)
    t[a + "dt_bias"] = _bf16(rng, NV)
    t[a + "conv1d.weight"] = _bf16(rng, CHANNELS, 1, TAPS)
    t[a + "in_proj_a.weight"] = _bf16(rng, NV, H)
    t[a + "in_proj_b.weight"] = _bf16(rng, NV, H)
    t[a + "in_proj_qkv.weight"] = _bf16(rng, CHANNELS, H)
    t[a + "in_proj_z.weight"] = _bf16(rng, VG, H)
    t[a + "norm.weight"] = (1 + _bf16(rng, DV, scale=0.1).float()).to(torch.bfloat16)
    t[a + "out_proj.weight"] = _bf16(rng, H, VG)
    s = "model.language_model.layers.1.self_attn."
    t[s + "q_proj.weight"] = _bf16(rng, 2 * HEADS * D, H)
    t[s + "k_proj.weight"] = _bf16(rng, KV * D, H)
    t[s + "v_proj.weight"] = _bf16(rng, KV * D, H)
    t[s + "q_norm.weight"] = _bf16(rng, D, scale=0.1, low=2**-10)
    t[s + "k_norm.weight"] = _bf16(rng, D, scale=0.1, low=2**-10)
    t[s + "o_proj.weight"] = _bf16(rng, H, HEADS * D)
    return t


def _tiled(x: np.ndarray, axis: int) -> np.ndarray:
    """llama.cpp's value-head reorder along ``axis`` (heads x width there): grouped [NK, R] to tiled [R, NK]."""
    x = np.moveaxis(x, axis, 0)
    shape = x.shape
    x = x.reshape(NK, NV // NK, -1, *shape[1:]).swapaxes(0, 1).reshape(shape)
    return np.moveaxis(x, 0, axis)


def _export(t: dict[str, torch.Tensor]) -> dict[str, tuple[np.ndarray, bool]]:
    """GGUF tensors as convert_hf_to_gguf.py's Qwen3_5TextModel writes them: (binary32 values, Q8_0?)."""
    f = {name: value.float().numpy() for name, value in t.items()}
    out = {"token_embd.weight": (f["model.language_model.embed_tokens.weight"], True),
           "output.weight": (f["lm_head.weight"], True),
           "output_norm.weight": (f["model.language_model.norm.weight"] + np.float32(1), False)}
    for i in range(2):
        p, b = f"model.language_model.layers.{i}.", f"blk.{i}."
        out[b + "attn_norm.weight"] = (f[p + "input_layernorm.weight"] + np.float32(1), False)
        out[b + "post_attention_norm.weight"] = (f[p + "post_attention_layernorm.weight"] + np.float32(1), False)
        for role in ("gate", "up", "down"):
            out[b + f"ffn_{role}.weight"] = (f[p + f"mlp.{role}_proj.weight"], True)
    a, b = "model.language_model.layers.0.linear_attn.", "blk.0."
    out[b + "ssm_a"] = (_tiled(-torch.exp(t[a + "A_log"].float()).numpy(), 0), False)
    out[b + "ssm_dt.bias"] = (_tiled(f[a + "dt_bias"], 0), False)
    conv = f[a + "conv1d.weight"][:, 0, :]
    out[b + "ssm_conv1d.weight"] = (np.concatenate([conv[: 2 * KG], _tiled(conv[2 * KG :], 0)]), False)
    out[b + "ssm_alpha.weight"] = (_tiled(f[a + "in_proj_a.weight"], 0), False)
    out[b + "ssm_beta.weight"] = (_tiled(f[a + "in_proj_b.weight"], 0), False)
    qkv = f[a + "in_proj_qkv.weight"]
    out[b + "attn_qkv.weight"] = (np.concatenate([qkv[: 2 * KG], _tiled(qkv[2 * KG :], 0)]), True)
    out[b + "attn_gate.weight"] = (_tiled(f[a + "in_proj_z.weight"], 0), True)
    out[b + "ssm_norm.weight"] = (f[a + "norm.weight"], False)
    out[b + "ssm_out.weight"] = (_tiled(f[a + "out_proj.weight"], 1), True)
    s, b = "model.language_model.layers.1.self_attn.", "blk.1."
    out[b + "attn_q.weight"] = (f[s + "q_proj.weight"], True)
    out[b + "attn_k.weight"] = (f[s + "k_proj.weight"], True)
    out[b + "attn_v.weight"] = (f[s + "v_proj.weight"], True)
    out[b + "attn_q_norm.weight"] = (f[s + "q_norm.weight"] + np.float32(1), False)
    out[b + "attn_k_norm.weight"] = (f[s + "k_norm.weight"] + np.float32(1), False)
    out[b + "attn_output.weight"] = (f[s + "o_proj.weight"], True)
    return out


def _write_gguf(path, tensors: dict[str, tuple[np.ndarray, bool]], **metadata) -> None:
    writer = gguf.GGUFWriter(str(path), "qwen35")
    values = {"block_count": 2, "embedding_length": H, "feed_forward_length": I, "attention.head_count": HEADS,
              "attention.head_count_kv": KV, "attention.key_length": D, "attention.value_length": D,
              "rope.dimension_count": 8, "ssm.conv_kernel": TAPS, "ssm.state_size": DK, "ssm.group_count": NK,
              "ssm.time_step_rank": NV, "ssm.inner_size": VG, "full_attention_interval": 2}
    values.update(metadata)
    for key, value in values.items():
        writer.add_uint32("qwen35." + key, value)
    writer.add_float32("qwen35.rope.freq_base", 10_000_000.0)
    writer.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_array("qwen35.rope.dimension_sections", [2, 1, 1, 0])
    for name, (value, quantized) in tensors.items():
        value = np.ascontiguousarray(value, dtype=np.float32)
        if quantized:
            writer.add_tensor(name, gguf.quants.quantize(value, gguf.GGMLQuantizationType.Q8_0),
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _q8_0(values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """gguf-py's Q8_0 blocks of a matrix, decoded by the block layout: (codes [N, K/32, 32], binary16 scales)."""
    n, k = values.shape
    blocks = gguf.quants.quantize(values.float().numpy(), gguf.GGMLQuantizationType.Q8_0).reshape(n, k // 32, 34)
    return blocks[..., 2:].copy().view(np.int8), blocks[..., :2].copy().view("<f2")[..., 0]


@pytest.fixture
def imported(tmp_path):
    rng = np.random.default_rng(35)
    checkpoint = _checkpoint(rng)
    _write_gguf(tmp_path / "model.gguf", _export(checkpoint))
    with _directory(tmp_path / "hf", checkpoint) as hf, _directory(tmp_path / "meta") as meta, \
            GgufStore([tmp_path / "model.gguf"]) as store:
        yield qwen3_5.build_model(hf), qwen3_5_gguf.build_model(meta, store)


def test_import_inverts_the_llama_cpp_export(imported):
    reference, model = imported
    assert model.config == reference.config
    assert sorted(model.parameters) == sorted(reference.parameters)
    assert set(model.packing_groups) == set(reference.packing_groups)
    matrices = 0
    for name, parameter in model.parameters.items():
        expected = reference.parameters[name]
        assert (parameter.shape, parameter.inputs, parameter.direct_format) == (
            expected.shape, expected.inputs, expected.direct_format), name
        truth = expected.source.values().reshape(expected.shape)
        if parameter.source.read_encoded is None:
            assert torch.equal(parameter.source.values().reshape(parameter.shape), truth.to(torch.float64)), name
            continue
        matrices += 1
        codes, scales = _q8_0(truth)
        rows = parameter.source.read_encoded(0, parameter.shape[0])
        assert rows.format == Q8 and np.array_equal(rows.codes.numpy(), codes), name
        assert np.array_equal(rows.scales.numpy().view(np.uint16), scales.view(np.uint16)), name
        dequantized = torch.from_numpy(codes.astype(np.float64) * scales.astype(np.float64)[..., None])
        assert torch.equal(parameter.source.values().reshape(parameter.shape), dequantized.reshape(parameter.shape)), name
        # Row windows that straddle the encoded rows' boundaries read the same values.
        k = parameter.shape[1]
        assert torch.equal(parameter.source.values(k - 3, 2 * k + 5), dequantized.reshape(-1)[k - 3 : 2 * k + 5]), name
    assert matrices == 18  # embedding, head; GDN q/k/v/z/output; QSA query/gate/key/value/output; 2 x MLP gate/up/down


def test_recipe_stores_the_q8_0_words_and_exact_values(imported):
    reference, model = imported
    recipe = Recipe(model)
    qwen3_5_gguf_q8_0(model, recipe, {})
    for name, parameter in model.parameters.items():
        (selection,) = recipe.selections[name]
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            assert name in recipe.separate_parameters and selection.method is cast_direct
        elif parameter.source.read_encoded is not None:
            assert (selection.format, selection.method) == (Q8, import_encoded), name
        else:
            assert (selection.format, selection.method) == (parameter.direct_format, cast_direct), name
    with pytest.raises(ValueError, match="Q8_0 matrices of a GGUF import"):
        qwen3_5_gguf_q8_0(reference, Recipe(reference), {})


def test_inexact_or_inconsistent_ggufs_are_refused(tmp_path):
    rng = np.random.default_rng(36)
    checkpoint = _checkpoint(rng)
    tensors = _export(checkpoint)

    def build(name, tensors, **metadata):
        path = tmp_path / f"{name}.gguf"
        _write_gguf(path, tensors, **metadata)
        store = GgufStore([path])
        meta = _directory(tmp_path / f"{name}-meta")
        return qwen3_5_gguf.build_model(meta, store)

    with pytest.raises(ValueError, match="ssm.group_count"):
        build("groups", tensors, **{"ssm.group_count": 1})
    with pytest.raises(ValueError, match="full_attention_interval"):
        build("interval", tensors, full_attention_interval=4)

    norm = dict(tensors)
    gamma = norm["blk.0.attn_norm.weight"][0].copy()
    gamma[3] = np.float32(1 + 2.0**-10 + 2.0**-20)  # g = 2^-10 + 2^-20 is not a BF16 value
    norm["blk.0.attn_norm.weight"] = (gamma, False)
    model = build("norm", norm)
    with pytest.raises(ValueError, match="not exact in BF16 after undoing"):
        model.parameters["text/layers/0/input_norm"].source.values()

    a = dict(tensors)
    stored = a["blk.0.ssm_a"][0].copy()
    stored[1] = np.float32(stored[1] * (1 + 2.0**-12))  # no BF16 A_log within two units in the last place
    a["blk.0.ssm_a"] = (stored, False)
    model = build("a", a)
    with pytest.raises(ValueError, match="cannot be recovered exactly"):
        model.parameters["text/layers/0/gdn/a_log"].source.values()


def test_value_head_columns_move_whole_blocks_only():
    matrix = array_source(torch.arange(2 * 4 * 32, dtype=torch.float64).reshape(2, 128), "m")
    moved = value_head_columns(matrix, 4, 32, 2).values().reshape(2, 4, 32)
    original = matrix.values().reshape(2, 4, 32)
    assert [int(moved[0, h, 0]) // 32 for h in range(4)] == [0, 2, 1, 3]  # tiled slot of HF head h
    assert torch.equal(moved[:, 1], original[:, 2]) and torch.equal(moved[:, 2], original[:, 1])
    with pytest.raises(ValueError, match="whole 32-column blocks"):
        value_head_columns(array_source(torch.zeros(2, 64, dtype=torch.float64), "m"), 4, 16, 2)
