"""The format maps of the Qwen3.8-Flash-Next recipes (design §6.1): recipe A keeps every main-model
tensor as NVIDIA stores it and re-quantizes only the MTP drafter; recipe B (Dense8) adds q8_g32_fp16 for
the dense projection classes that dominate per-token weight reads and leaves the rest as recipe A; recipe
C (weight-only exports) is selected by its W4A16 banks."""
from __future__ import annotations

import pytest
import torch

from tools.convert.methods import cast_direct, grouped_absmax, grouped_mse, import_encoded
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import (
    qwen3_8_flash_next_dflash2,
    qwen3_8_flash_next_fp8,
    qwen3_8_flash_next_nvfp4,
    qwen3_8_flash_next_nvfp4_dense8,
    qwen3_8_flash_next_nvfp4_orcarouter,
)
from tools.convert.expert_banks import ExpertBankSource, import_expert_bank
from tools.convert.qwen4_exp import Fp8ExpertBankSource, import_fp8_expert_bank
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

Q4 = "q4_g64_fp16"
Q8 = "q8_g32_fp16"
LAYER = "text/layers/0"
MTP = "mtp/layers/0"

DENSE8 = (
    f"{LAYER}/attn_hc/down", f"{LAYER}/attn_hc/inject", f"{LAYER}/attn_hc/up",
    f"{LAYER}/mlp_hc/down", f"{LAYER}/mlp_hc/inject", f"{LAYER}/mlp_hc/up",
    f"{LAYER}/gdn/query", f"{LAYER}/gdn/key", f"{LAYER}/gdn/value", f"{LAYER}/gdn/z", f"{LAYER}/gdn/output",
    f"{LAYER}/attention/output",
    f"{LAYER}/moe/shared/gate", f"{LAYER}/moe/shared/up", f"{LAYER}/moe/shared/down",
    f"{LAYER}/ple/key_projection", f"{LAYER}/ple/value_projection",
    "text/final_mixer/down", "text/final_mixer/up", "text/output_head",
)
# Kept as NVIDIA stores them in both recipes.
KEPT = (
    "text/token_embedding",
    f"{LAYER}/gdn/a_projection", f"{LAYER}/gdn/b_projection",
    f"{LAYER}/attention/query", f"{LAYER}/attention/key", f"{LAYER}/attention/value",
    f"{LAYER}/attention/index_query",
    f"{LAYER}/moe/router", f"{LAYER}/moe/shared_score",
)
DRAFTER_Q8 = (f"{MTP}/attention/query", f"{MTP}/attention/output", f"{MTP}/moe/shared/up")
DRAFTER_KEPT = (f"{MTP}/moe/router", f"{MTP}/moe/shared_score")
DRAFTER_EXPERTS = (f"{MTP}/moe/experts/0/gate", f"{MTP}/moe/experts/0/down")
BANK = f"{LAYER}/moe/experts"


def _model() -> Model:
    model = Model({"text": {"config": {"architectures": ["Qwen4ExpForCausalLM"]}}})
    for name in (*DENSE8, *KEPT, *DRAFTER_Q8, *DRAFTER_KEPT, *DRAFTER_EXPERTS):
        inputs = () if name == "text/token_embedding" else ("input",)
        source = array_source(torch.ones((4, 64), dtype=torch.bfloat16), name)
        model.add(Parameter(name, (4, 64), source, inputs=inputs))
    bank = array_source(torch.ones((2, 4, 64), dtype=torch.bfloat16), BANK)
    model.add(Parameter(BANK, (2, 4, 64), bank, inputs=("input",)))
    return model


def _selections(recipe_function) -> dict:
    model = _model()
    recipe = Recipe(model)
    recipe_function(model, recipe, {})
    return recipe.selections


def _only(selections, name):
    chosen = selections.get(name, [])
    assert len(chosen) == 1, f"{name}: {len(chosen)} selections"
    return chosen[0]


def _check_common(selections) -> None:
    bank = _only(selections, BANK)
    assert (bank.format, bank.layout, bank.method) == ("nvfp4_mul", "nvfp4_expert_rg16_v1", import_expert_bank)
    for name in DRAFTER_EXPERTS:
        chosen = _only(selections, name)
        assert (chosen.format, chosen.method) == (Q4, grouped_mse), name
    for name in DRAFTER_Q8:
        chosen = _only(selections, name)
        assert (chosen.format, chosen.method) == (Q8, grouped_absmax), name
    for name in (*KEPT, *DRAFTER_KEPT):
        chosen = _only(selections, name)
        assert (chosen.format, chosen.method) == ("bf16", cast_direct), f"{name} must keep NVIDIA's representation"


def test_recipe_a_keeps_the_main_model_and_requantizes_only_the_drafter() -> None:
    selections = _selections(qwen3_8_flash_next_nvfp4)
    _check_common(selections)
    for name in DENSE8:
        chosen = _only(selections, name)
        assert (chosen.format, chosen.method) == ("bf16", cast_direct), f"recipe A must keep {name} as stored"


def test_recipe_b_adds_q8_for_the_dense_classes_only() -> None:
    selections = _selections(qwen3_8_flash_next_nvfp4_dense8)
    _check_common(selections)
    for name in DENSE8:
        chosen = _only(selections, name)
        assert (chosen.format, chosen.method) == (Q8, grouped_absmax), name


def _weight_only_model() -> Model:
    model = _model()
    del model.parameters[BANK]
    bank = ExpertBankSource((2, 4, 64), BANK, None, "experts.", packed=True)
    model.add(Parameter(BANK, (2, 4, 64), bank, inputs=("input",)))
    return model


def test_weight_only_experts_take_recipe_c_and_modelopt_ones_recipes_a_and_b() -> None:
    # A weight-only bank has no activation scales: recipes A and B (W4A4) refuse it, recipe C (W4A16)
    # refuses a ModelOpt bank.
    for recipe_function in (qwen3_8_flash_next_nvfp4, qwen3_8_flash_next_nvfp4_dense8):
        model = _weight_only_model()
        with pytest.raises(ValueError, match="recipe C"):
            recipe_function(model, Recipe(model), {})
    model = _model()
    with pytest.raises(ValueError, match="weight-only"):
        qwen3_8_flash_next_nvfp4_orcarouter(model, Recipe(model), {"base": None})


def _fp8_model() -> Model:
    model = _model()
    del model.parameters[BANK]
    bank = Fp8ExpertBankSource((2, 128, 128), BANK, None, "experts.")
    model.add(Parameter(BANK, (2, 128, 128), bank, inputs=("input",)))
    return model


def test_fp8_recipe_imports_the_block_fp8_banks_verbatim() -> None:
    """F-A keeps every dense tensor as stored and the experts as fp8_e4m3fn_b128 records; only the
    drafter is re-quantized, its routed experts q2_g64_fp16 (MSE scales). The NVFP4 recipes refuse an
    FP8 bank and the FP8 recipe an NVFP4 one."""

    model = _fp8_model()
    recipe = Recipe(model)
    qwen3_8_flash_next_fp8(model, recipe, {})
    bank = _only(recipe.selections, BANK)
    assert (bank.format, bank.layout, bank.method) == ("fp8_e4m3fn_b128", "fp8_expert_b128_v1", import_fp8_expert_bank)
    for name in (*DENSE8, *KEPT, *DRAFTER_KEPT):
        chosen = _only(recipe.selections, name)
        assert (chosen.format, chosen.method) == ("bf16", cast_direct), name
    for name in DRAFTER_Q8:
        chosen = _only(recipe.selections, name)
        assert (chosen.format, chosen.method) == (Q8, grouped_absmax), name
    for name in DRAFTER_EXPERTS:
        chosen = _only(recipe.selections, name)
        assert (chosen.format, chosen.method) == ("q2_g64_fp16", grouped_mse), name
    for recipe_function in (qwen3_8_flash_next_nvfp4, qwen3_8_flash_next_nvfp4_dense8):
        model = _fp8_model()
        with pytest.raises(ValueError, match="FP8 recipe"):
            recipe_function(model, Recipe(model), {})
    model = _model()
    with pytest.raises(ValueError, match="block-FP8"):
        qwen3_8_flash_next_fp8(model, Recipe(model), {})


def test_fp4_local_recipe_keeps_the_mxfp8_words_and_defaults_the_experts_to_a16() -> None:
    """FP4-Local (design §6.1.2): the ModelOpt bank becomes an nvfp4_mul bank whose Use permits 4-bit
    activations, with its FP32 activation scales, and the text config makes 16-bit expert
    activations the default; the MXFP8 GDN, QSA (the indexer included) and shared-expert projections
    keep their words, every other main tensor its stored dtype (GDN a/b their exact BF16 values), the
    QSA projections and indexer form one parent, and the drafter is recipe A's. The weight-only NVFP4
    vision MLPs and merger keep their words as dense nvfp4_mul matrices in nvfp4_rg16_v1 (16-bit
    activations, each its own parent); the rest of the vision tower keeps its BF16 values. Other banks
    and other n-gram tables are refused."""

    from tools.convert.official_recipes import qwen3_8_flash_next_fp4_local

    mxfp8 = tuple(n for n in DENSE8 if any(n.endswith(s) for s in ("/gdn/query", "/gdn/key", "/gdn/value", "/gdn/z",
                                                                      "/gdn/output", "/attention/output",
                                                                      "/moe/shared/gate", "/moe/shared/up",
                                                                      "/moe/shared/down")))
    mxfp8 += (f"{LAYER}/attention/query", f"{LAYER}/attention/key", f"{LAYER}/attention/value",
              f"{LAYER}/attention/index_query")

    vision_nvfp4 = ("vision/layers/0/mlp/fc1", "vision/layers/0/mlp/fc2", "vision/merger/fc1", "vision/merger/fc2")
    vision_bf16 = ("vision/layers/0/attention/query", "vision/layers/0/attention/output", "vision/patch_embedding")
    vision_direct = ("vision/layers/0/mlp/fc1_bias", "vision/merger/norm_weight")
    encoded = {}  # each vision NVFP4 parameter's nvfp4_mul source, as the builder's factory returns it

    def model(bank_source, table="nvfp4") -> Model:
        m = Model({"text": {"config": {"architectures": ["Qwen4ExpForCausalLM"], "ngram_table": {"format": table},
                                       "layer_types": ["linear_attention"]}}})
        for name in (*DENSE8, *KEPT, *DRAFTER_Q8, *DRAFTER_KEPT, *DRAFTER_EXPERTS, *vision_bf16):
            inputs = () if name == "text/token_embedding" else ("input",)
            source = array_source(torch.ones((4, 64), dtype=torch.bfloat16), name)
            m.add(Parameter(name, (4, 64), source, lambda store, format=None, s=source: s, inputs))
        for name in vision_nvfp4:
            values = array_source(torch.ones((16, 64), dtype=torch.float64), name)
            encoded[name] = array_source(torch.ones((16, 64), dtype=torch.float64), name + " (nvfp4_mul)")

            def factory(store, format=None, values=values, words=encoded[name]):
                assert format in (None, "nvfp4_mul")
                return words if format == "nvfp4_mul" else values

            m.add(Parameter(name, (16, 64), values, factory, (name + "_input",)))
        for name in vision_direct:
            m.add(Parameter(name, (64,), array_source(torch.ones(64, dtype=torch.bfloat16), name)))
        m.add(Parameter(BANK, (2, 4, 64), bank_source, inputs=("input",)))
        scales = f"{LAYER}/moe/expert_input_scales"
        m.add(Parameter(scales, (2, 3), array_source(torch.ones((2, 3)), scales), direct_format="fp32"))
        return m

    m = model(ExpertBankSource((2, 4, 64), BANK, None, "experts."))
    recipe = Recipe(m)
    qwen3_8_flash_next_fp4_local(m, recipe, {"base": None})
    selections = recipe.selections
    for name in vision_nvfp4:
        chosen = _only(selections, name)
        assert (chosen.format, chosen.layout, chosen.method) == ("nvfp4_mul", "nvfp4_rg16_v1", import_encoded), name
        assert chosen.source is encoded[name], name
        assert recipe.policies[(name, name + "_input")] == "A16Only"
    for name in (*vision_bf16, *vision_direct):
        chosen = _only(selections, name)
        assert (chosen.format, chosen.method) == ("bf16", cast_direct), name
    assert not any(any(n.startswith("vision/") for n in names) for names, _ in recipe.explicit_groups)
    bank = _only(selections, BANK)
    assert (bank.format, bank.layout, bank.method) == ("nvfp4_mul", "nvfp4_expert_rg16_v1", import_expert_bank)
    for name in DRAFTER_EXPERTS:
        assert (_only(selections, name).format, _only(selections, name).method) == (Q4, grouped_mse), name
    for name in DRAFTER_Q8:
        assert (_only(selections, name).format, _only(selections, name).method) == (Q8, grouped_absmax), name
    for name in (*KEPT, *DRAFTER_KEPT):
        if name not in mxfp8:  # the embedding, GDN a/b, router and shared-expert gate
            chosen = _only(selections, name)
            assert (chosen.format, chosen.method) == ("bf16", cast_direct), name
    scales = _only(selections, f"{LAYER}/moe/expert_input_scales")
    assert (scales.format, scales.method) == ("fp32", cast_direct)
    assert all(recipe.policies[(BANK, use)] == "AllowA4" for use in m.parameters[BANK].inputs)
    assert m.components["text"]["config"]["expert_a4_activations"] == "optional"
    # No explicit attention groups: the builder's one QSA parent (query, gate, key, value, indexer) stands.
    assert not any(any("/attention/" in n for n in names) for names, _ in recipe.explicit_groups)
    for name in mxfp8:
        chosen = _only(selections, name)
        assert (chosen.format, chosen.method) == ("fp8_e4m3fn_g32_e8m0", import_encoded), name
    for name in DENSE8:
        if name not in mxfp8:  # HC mixers, PLE projections, the final mixer and lm_head: BF16 as stored
            chosen = _only(selections, name)
            assert (chosen.format, chosen.method) == ("bf16", cast_direct), name
    with pytest.raises(ValueError, match="ModelOpt NVFP4 experts"):
        weight_only = model(ExpertBankSource((2, 4, 64), BANK, None, "experts.", packed=True))
        qwen3_8_flash_next_fp4_local(weight_only, Recipe(weight_only), {"base": None})
    with pytest.raises(ValueError, match="NVFP4 n-gram table"):
        fp8_table = model(ExpertBankSource((2, 4, 64), BANK, None, "experts."), table="fp8_e4m3fn")
        qwen3_8_flash_next_fp4_local(fp8_table, Recipe(fp8_table), {"base": None})


def test_iq3s_recipe_tiles_the_tiled_formats_and_takes_a_q2_drafter() -> None:
    """On a GGUF-built model the IQ3_S recipe stores the Q4_K, Q5_K, Q6_K and IQ4_XS dense matrices in
    ggml_tiles16_v1 and every other GGUF format, and the token embedding (a row gather), in
    ggml_rows_v1, each by the verbatim import; the drafter's routed experts take q2_g64_fp16."""

    from types import SimpleNamespace

    from tools.convert.methods import import_ggml_rows
    from tools.convert.official_recipes import qwen3_8_flash_next_gsq_rco_iq3s

    encoded = {f"{LAYER}/gdn/query": "ggml_q6_k", f"{LAYER}/gdn/z": "ggml_iq4_xs", f"{LAYER}/moe/shared/gate": "ggml_q4_k",
               f"{LAYER}/attention/query": "ggml_q5_k", f"{LAYER}/attention/output": "ggml_iq4_nl",
               "text/token_embedding": "ggml_q6_k"}
    tiled = {"ggml_q4_k", "ggml_q5_k", "ggml_q6_k", "ggml_iq4_xs"}
    m = Model({"text": {"config": {"architectures": ["Qwen4ExpForCausalLM"], "gdn_a": "direct"}}})
    for name, fmt in encoded.items():
        source = SimpleNamespace(shape=(32, 256), read_encoded=lambda start, count, fmt=fmt: SimpleNamespace(format=fmt))
        m.add(Parameter(name, (32, 256), source, inputs=() if name == "text/token_embedding" else ("input",)))
    for name in (*DRAFTER_Q8, *DRAFTER_KEPT, *DRAFTER_EXPERTS):
        m.add(Parameter(name, (4, 64), array_source(torch.ones((4, 64), dtype=torch.bfloat16), name), inputs=("input",)))
    recipe = Recipe(m)
    qwen3_8_flash_next_gsq_rco_iq3s(m, recipe, {})
    for name, fmt in encoded.items():
        chosen = _only(recipe.selections, name)
        layout = "ggml_tiles16_v1" if fmt in tiled and name != "text/token_embedding" else "ggml_rows_v1"
        assert (chosen.format, chosen.layout or "ggml_rows_v1", chosen.method) == (fmt, layout, import_ggml_rows), name
    for name in DRAFTER_EXPERTS:
        chosen = _only(recipe.selections, name)
        assert (chosen.format, chosen.method) == ("q2_g64_fp16", grouped_mse), name


def test_fp8_bank_import_writes_each_expert_verbatim(tmp_path) -> None:
    """Each record is the expert's gate, up and down codes, then their BF16 tile scales, padded to
    4 KiB, read back by the codec's decoder; a non-BF16 scale plane is refused."""

    import numpy as np
    from safetensors.torch import save_file

    from tools.artifact.codecs.fp8_b128 import decode_expert
    from tools.artifact.layouts import fp8_expert_geometry
    from tools.artifact.schema import TensorSpec
    from tools.convert.methods import MethodInput, PrepareRequest
    from tools.convert.sources.safetensors import SafetensorsSource

    rng = np.random.default_rng(3)
    experts, hidden, inter = 2, 256, 128
    tensors, expected = {}, []
    for e in range(experts):
        parts = []
        for role, shape in (("gate", (inter, hidden)), ("up", (inter, hidden)), ("down", (hidden, inter))):
            codes = rng.integers(0, 256, size=shape, dtype=np.uint8)
            codes[codes & 0x7F == 0x7F] = 0x3C  # no NaN codes
            scales = torch.from_numpy(rng.uniform(2.0**-12, 2.0**-6, size=(shape[0] // 128, shape[1] // 128)))
            scales = scales.to(torch.bfloat16)
            tensors[f"experts.{e}.{role}_proj.weight"] = torch.from_numpy(codes).view(torch.float8_e4m3fn)
            tensors[f"experts.{e}.{role}_proj.weight_scale_inv"] = scales
            parts.append((codes, scales.view(torch.int16).numpy().view(np.uint16)))
        expected.append(parts)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    class Output:
        def __init__(self, size):
            self.data = bytearray(size)

        def write_bytes(self, offset, data):
            self.data[offset : offset + len(data)] = data

    with SafetensorsSource(tmp_path / "model.safetensors") as store:
        bank = Fp8ExpertBankSource((experts, hidden, inter), BANK, store, "experts.")
        g = fp8_expert_geometry("fp8_e4m3fn_b128", bank.shape)
        assert g.record_stride % 4096 == 0 and g.record_bytes == 3 * hidden * inter + 3 * 2 * 2
        spec = TensorSpec("weight/0", bank.shape, "fp8_e4m3fn_b128", "fp8_expert_b128_v1")
        out = Output(g.payload_bytes)
        import_fp8_expert_bank(PrepareRequest(spec, (MethodInput("experts", bank, ()),), {}, {})).produce(out)
    for e in range(experts):
        record = bytes(out.data[e * g.record_stride : (e + 1) * g.record_stride])
        for (codes, words), (got_codes, got_words) in zip(expected[e], decode_expert(record, hidden, inter)):
            assert np.array_equal(codes, got_codes) and np.array_equal(words, got_words)
        assert record[g.record_bytes :] == bytes(g.record_stride - g.record_bytes)

    tensors["experts.1.down_proj.weight_scale_inv"] = tensors["experts.1.down_proj.weight_scale_inv"].float()
    save_file(tensors, str(tmp_path / "f32.safetensors"))
    with SafetensorsSource(tmp_path / "f32.safetensors") as store:
        bank = Fp8ExpertBankSource((experts, hidden, inter), BANK, store, "experts.")
        out = Output(g.payload_bytes)
        with pytest.raises(ValueError, match="BF16"):
            import_fp8_expert_bank(PrepareRequest(spec, (MethodInput("experts", bank, ()),), {}, {})).produce(out)


# ---------------------------------------------------------------- DFlash2 companion (--draft-model)

D2 = "dflash2/layers/0"
D2_Q8 = (
    "dflash2/feature_projection",
    f"{D2}/attention/query", f"{D2}/attention/key", f"{D2}/attention/value", f"{D2}/attention/output",
    f"{D2}/mlp/gate", f"{D2}/mlp/up", f"{D2}/mlp/down",
)
D2_BF16_PROJECTIONS = (
    f"{D2}/attention_conv/kernel_projection", f"{D2}/mlp_conv/kernel_projection",
    "dflash2/candidate_selector/hidden_projection",
)
D2_DIRECT = ("dflash2/final_norm", f"{D2}/input_norm", "dflash2/candidate_selector/predecessor_codebook")
TARGET_HEADS = (("text/output_head", "dflash2/final_hidden"), ("proposal/head", "dflash2/final_hidden"))


def _companion(external_uses=TARGET_HEADS) -> Model:
    model = Model(
        {"dflash2": {"config": {"num_hidden_layers": 1}, "target": "text"}},
        target={"architectures": ["Qwen4ExpForCausalLM"]},
        external_uses=external_uses,
    )
    for name in (*D2_Q8, *D2_BF16_PROJECTIONS, f"{D2}/attention/context_key", f"{D2}/attention/context_value"):
        model.add(Parameter(name, (4, 64), array_source(torch.ones((4, 64), dtype=torch.bfloat16), name), inputs=("x",)))
    for name in D2_DIRECT:
        model.add(Parameter(name, (64,), array_source(torch.ones((64,), dtype=torch.bfloat16), name)))
    return model


def test_dflash2_companion_takes_the_official_27b_drafter_quantization() -> None:
    model = _companion()
    recipe = Recipe(model)
    qwen3_8_flash_next_dflash2(model, recipe, {})
    for name in D2_Q8:
        chosen = _only(recipe.selections, name)
        assert (chosen.format, chosen.method) == (Q8, grouped_absmax), name
    for name in (*D2_BF16_PROJECTIONS, *D2_DIRECT):
        assert _only(recipe.selections, name).format == "bf16", name
    # The context K/V projections share the block's K/V words.
    assert recipe.aliases[f"{D2}/attention/context_key"] == f"{D2}/attention/key"
    assert recipe.aliases[f"{D2}/attention/context_value"] == f"{D2}/attention/value"
    # Its Uses of the target's heads are declared (16-bit activations), with no binding of their own.
    for key in TARGET_HEADS:
        assert recipe.policies[key] == "A16Only"
        assert key[0] not in recipe.selections


def test_dflash2_companion_recipe_refuses_other_models() -> None:
    model = _model()
    with pytest.raises(ValueError, match="companion"):
        qwen3_8_flash_next_dflash2(model, Recipe(model), {})
    other = _companion()
    other.target = {"architectures": ["Qwen3_5ForCausalLM"]}
    with pytest.raises(ValueError, match="companion"):
        qwen3_8_flash_next_dflash2(other, Recipe(other), {})
    # An external Use names a parameter of the target, never one of the companion's own.
    with pytest.raises(ValueError, match="external Use"):
        Recipe(_companion(external_uses=(("dflash2/feature_projection", "dflash2/final_hidden"),)))
