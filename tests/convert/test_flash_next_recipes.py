"""The format maps of the Qwen3.8-Flash-Next recipes (design §6.1): recipe A keeps every main-model
tensor as NVIDIA stores it and re-quantizes only the MTP drafter; recipe B (Dense8) adds q8_g32_fp16 for
the dense projection classes that dominate per-token weight reads and leaves the rest as recipe A; recipe
C (weight-only exports) is selected by its W4A16 banks."""
from __future__ import annotations

import pytest
import torch

from tools.convert.methods import cast_direct, grouped_absmax, grouped_mse
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import (
    qwen3_8_flash_next_nvfp4,
    qwen3_8_flash_next_nvfp4_dense8,
    qwen3_8_flash_next_nvfp4_orcarouter,
)
from tools.convert.qwen4_exp import ExpertBankSource, import_expert_bank
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
    f"{LAYER}/attention/indexer_query",
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
