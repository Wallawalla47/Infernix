from __future__ import annotations

import pytest
import torch

from tools.convert.methods import cast_direct, grouped_absmax, import_encoded, nvfp4_mse
from tools.convert.model import Model, Parameter
from tools.convert.official_recipes import (
    RECIPES,
    qwen3_8_27b,
    qwen3_8_27b_nvfp4_nvidia,
    qwen3_8_27b_q6,
)
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import EncodedRows, LogicalSource, array_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"

LAYER = "text/layers/0"


def _dense_model() -> Model:
    model = Model({"text": {"config": {}}})
    names = (
        "text/token_embedding",
        "text/output_head",
        f"{LAYER}/gdn/query",
        f"{LAYER}/gdn/output",
        f"{LAYER}/attention/query",
        f"{LAYER}/attention/key",
        f"{LAYER}/attention/output",
        f"{LAYER}/mlp/gate",
        f"{LAYER}/mlp/up",
        f"{LAYER}/mlp/down",
    )
    for name in names:
        inputs = () if name in ("text/token_embedding", "text/output_head") else ("input",)
        source = array_source(torch.ones((4, 8), dtype=torch.bfloat16), name)
        model.add(Parameter(name, (4, 8), source, inputs=inputs))
    return model


def _formats(recipe_function) -> dict[str, set[str]]:
    model = _dense_model()
    recipe = Recipe(model)
    recipe_function(model, recipe, {})
    return {
        name: {selection.format for selection in selections}
        for name, selections in recipe.selections.items()
    }


def _single(formats: dict[str, set[str]], name: str) -> str:
    values = formats[name]
    assert len(values) == 1, f"{name} is split across formats {values}"
    return next(iter(values))


def test_q6_recipe_is_registered() -> None:
    assert RECIPES["qwen3_8_27b_q6"] is qwen3_8_27b_q6


def test_registered_recipe_gives_the_mlp_pair_q4() -> None:
    formats = _formats(qwen3_8_27b)
    assert _single(formats, f"{LAYER}/mlp/gate") == Q4
    assert _single(formats, f"{LAYER}/mlp/up") == Q4
    assert _single(formats, f"{LAYER}/mlp/down") == Q5
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q8


def test_q6_recipe_moves_only_the_mlp_pair() -> None:
    registered = _formats(qwen3_8_27b)
    q6 = _formats(qwen3_8_27b_q6)

    assert set(registered) == set(q6)
    moved = {name for name in q6 if q6[name] != registered[name]}
    assert moved == {f"{LAYER}/mlp/gate", f"{LAYER}/mlp/up"}

    for name in moved:
        assert registered[name] == {Q4}
        assert q6[name] == {Q6}


def test_q6_recipe_keeps_the_vocabulary_at_q8() -> None:
    """Q8 is 8.5 bits per weight, so the vocabulary endpoints already outrank Q6."""
    formats = _formats(qwen3_8_27b_q6)
    assert _single(formats, "text/token_embedding") == Q8
    assert _single(formats, "text/output_head") == Q8


def test_q6_recipe_leaves_the_other_projections_alone() -> None:
    formats = _formats(qwen3_8_27b_q6)
    assert _single(formats, f"{LAYER}/attention/query") == Q4
    assert _single(formats, f"{LAYER}/attention/key") == Q4
    assert _single(formats, f"{LAYER}/gdn/query") == Q4
    assert _single(formats, f"{LAYER}/attention/output") == Q5
    assert _single(formats, f"{LAYER}/gdn/output") == Q5
    assert _single(formats, f"{LAYER}/mlp/down") == Q5


FP8_ROW = "fp8_e4m3fn_row_bf16"
GDN_LAYER = "text/layers/1"


def _stored(name: str, stored: str | None) -> LogicalSource:
    """A checkpoint matrix whose first row reports its stored encoding (None: unquantized values)."""
    values = torch.ones((4, 8), dtype=torch.bfloat16)
    encoded = None
    if stored is not None:
        encoded = lambda begin, end: EncodedRows(stored, torch.zeros(1), torch.zeros(1))  # noqa: E731
    return LogicalSource((4, 8), name, lambda begin, end: values.reshape(-1)[begin:end], encoded)


# nvidia/Qwen3.8-27B-NVFP4's layout: NVFP4 MLP, per-tensor FP8 attention and GDN.
_AUTOQUANT = {
    "/gdn/query": FP8_ROW,
    "/gdn/output": FP8_ROW,
    "/attention/query": FP8_ROW,
    "/attention/output": FP8_ROW,
    "/mlp/gate": "nvfp4",
    "/mlp/up": "nvfp4",
    "/mlp/down": "nvfp4",
}


def _nvidia_model(stored: dict[str, str | None] | None = None, extra: tuple[str, ...] = ()) -> Model:
    """``stored`` maps a projection's role (or a full name) to its checkpoint encoding."""
    stored = {**_AUTOQUANT, **(stored or {})}
    model = Model({"text": {"config": {}}})
    names = (
        "text/token_embedding",
        "text/output_head",
        f"{LAYER}/gdn/a_projection",
        f"{LAYER}/gdn/b_projection",
        f"{LAYER}/gdn/query",
        f"{LAYER}/gdn/output",
        f"{LAYER}/attention/query",
        f"{LAYER}/attention/output",
        f"{LAYER}/mlp/gate",
        f"{LAYER}/mlp/up",
        f"{LAYER}/mlp/down",
    ) + extra
    for name in names:
        inputs = () if name == "text/token_embedding" else ("input",)
        role = "/" + name.split("/", 3)[3] if name.startswith("text/layers/") else name
        source = _stored(name, stored.get(name, stored.get(role)))
        model.add(
            Parameter(
                name,
                (4, 8),
                source,
                source_factory=lambda store, fmt, _source=source: _source,
                inputs=inputs,
            )
        )
    return model


def test_nvidia_recipe_stores_the_head_as_q8() -> None:
    """NInfer registers the vocabulary projection only for Q8, Q6 and FP8, so the
    source's NVFP4 head is dequantised and re-quantised to Q8 (16-bit activations)."""
    model = _nvidia_model()
    recipe = Recipe(model)
    qwen3_8_27b_nvfp4_nvidia(model, recipe, {"quantized": object()})

    head = recipe.selections["text/output_head"]
    assert len(head) == 1
    assert head[0].format == Q8
    assert head[0].method is grouped_absmax
    assert all(policy == "A16Only" for (parameter, _), policy in recipe.policies.items() if parameter == "text/output_head")

    assert _single(_selection_formats(recipe), f"{LAYER}/mlp/gate") == "nvfp4"
    assert _single(_selection_formats(recipe), f"{LAYER}/mlp/down") == "nvfp4"
    mlp_gate = recipe.selections[f"{LAYER}/mlp/gate"][0]
    assert mlp_gate.method is import_encoded
    assert _single(_selection_formats(recipe), f"{LAYER}/attention/query") == (
        "fp8_e4m3fn_row_bf16"
    )
    assert _single(_selection_formats(recipe), f"{LAYER}/gdn/query") == (
        "fp8_e4m3fn_row_bf16"
    )


def _selection_formats(recipe: Recipe) -> dict[str, set[str]]:
    return {
        name: {selection.format for selection in selections}
        for name, selections in recipe.selections.items()
    }


def test_nvidia_recipe_imports_each_projection_in_its_stored_encoding() -> None:
    """nvidia/Qwen3.8-27B-NVFP4-QAD keeps some GDN layers FP8 and makes the others NVFP4: each matrix is
    imported as stored, NVFP4 with 4-bit activations permitted and FP8 with 8-bit ones."""
    gdn = (f"{GDN_LAYER}/gdn/query", f"{GDN_LAYER}/gdn/output")
    model = _nvidia_model({name: "nvfp4" for name in gdn}, extra=gdn)
    recipe = Recipe(model)
    qwen3_8_27b_nvfp4_nvidia(model, recipe, {"quantized": object()})

    for name, format, policy in (
        (f"{GDN_LAYER}/gdn/query", "nvfp4", "AllowA4"),
        (f"{GDN_LAYER}/gdn/output", "nvfp4", "AllowA4"),
        (f"{LAYER}/gdn/query", FP8_ROW, "AllowA8"),
        (f"{LAYER}/gdn/output", FP8_ROW, "AllowA8"),
        (f"{LAYER}/mlp/down", "nvfp4", "AllowA4"),
    ):
        (selection,) = recipe.selections[name]
        assert (selection.format, selection.method) == (format, import_encoded), name
        assert recipe.policies[(name, "input")] == policy, name


def test_nvidia_recipe_refuses_an_unquantized_projection() -> None:
    model = _nvidia_model({"/attention/output": None})
    with pytest.raises(ValueError, match="attention/output: expected a ModelOpt NVFP4 or per-tensor FP8"):
        qwen3_8_27b_nvfp4_nvidia(model, Recipe(model), {"quantized": object()})


_STEM = ("mtp/input_projection", "mtp/embedding_norm", "mtp/hidden_norm", "mtp/final_norm")


class _Store:
    def __init__(self, names=()):
        self.names = set(names)

    def has(self, name):
        return name in self.names


def _mtp_model(stem_store) -> Model:
    """The NVIDIA model with an MTP stem whose sources come from ``stem_store`` when selected from it."""
    model = _nvidia_model()
    model.components["mtp"] = {"config": {}, "target": "text"}
    for name in _STEM:
        shape = (4, 8) if name == "mtp/input_projection" else (8,)
        base = array_source(torch.ones(shape, dtype=torch.bfloat16), "base:" + name)
        other = array_source(torch.full(shape, 2.0, dtype=torch.bfloat16), "mtp:" + name)
        model.add(
            Parameter(
                name,
                shape,
                base,
                source_factory=lambda store, fmt, _b=base, _o=other: _o if store is stem_store else _b,
                inputs=("mtp/stem_input",) if name == "mtp/input_projection" else (),
            )
        )
    return model


def test_nvidia_recipe_takes_a_missing_mtp_stem_from_the_mtp_source() -> None:
    """The QAD export has the MTP decoder layer but not its stem (fc and three norms)."""
    stem = _Store()
    model = _mtp_model(stem)
    recipe = Recipe(model)
    qwen3_8_27b_nvfp4_nvidia(model, recipe, {"quantized": object(), "base": _Store(), "mtp": stem})
    for name in _STEM:
        (selection,) = recipe.selections[name]
        assert selection.source.label == "mtp:" + name
    assert recipe.selections["mtp/input_projection"][0].format == Q8


def test_nvidia_recipe_keeps_a_stored_mtp_stem() -> None:
    model = _mtp_model(_Store())
    recipe = Recipe(model)
    sources = {"quantized": object(), "base": _Store({"mtp.fc.weight"})}  # no mtp source needed
    qwen3_8_27b_nvfp4_nvidia(model, recipe, sources)
    for name in _STEM:
        (selection,) = recipe.selections[name]
        assert selection.source is None or selection.source.label == "base:" + name


def test_exact_nvidia_recipe_keeps_every_stored_word() -> None:
    """The bit-exact recipe: NVFP4 as nvfp4_mul (W4A4 permitted, the head A16), per-tensor FP8 as
    fp8_e4m3fn_row_f32 (W8A8 permitted), and every BF16 tensor (embedding, GDN a/b) as stored."""
    gdn = (f"{GDN_LAYER}/gdn/query", f"{GDN_LAYER}/gdn/output")
    model = _nvidia_model({**{name: "nvfp4" for name in gdn}, "text/output_head": "nvfp4"}, extra=gdn)
    recipe = Recipe(model)
    RECIPES["qwen3_8_27b_nvfp4_nvidia_exact"](model, recipe, {"quantized": object()})

    for name, format, layout, policy in (
        (f"{GDN_LAYER}/gdn/query", "nvfp4_mul", "block_scale_k16_m128x4_v1", "AllowA4"),
        (f"{LAYER}/mlp/gate", "nvfp4_mul", "block_scale_k16_m128x4_v1", "AllowA4"),
        ("text/output_head", "nvfp4_mul", "block_scale_k16_m128x4_v1", "A16Only"),
        (f"{LAYER}/gdn/query", "fp8_e4m3fn_row_f32", None, "AllowA8"),
        (f"{LAYER}/attention/output", "fp8_e4m3fn_row_f32", None, "AllowA8"),
    ):
        (selection,) = recipe.selections[name]
        assert (selection.format, selection.layout, selection.method) == (format, layout, import_encoded), name
        assert recipe.policies[(name, "input")] == policy, name
    for name in ("text/token_embedding", f"{LAYER}/gdn/a_projection"):
        (selection,) = recipe.selections[name]
        assert (selection.format, selection.method) == ("bf16", cast_direct), name  # the stored BF16 values


def test_exact_nvidia_recipe_refuses_a_head_it_cannot_run() -> None:
    model = _nvidia_model({"text/output_head": FP8_ROW})
    with pytest.raises(ValueError, match="NVFP4 head"):
        RECIPES["qwen3_8_27b_nvfp4_nvidia_exact"](model, Recipe(model), {"quantized": object()})


def test_nvidia_recipe_leaves_a_grafted_dflash2_alone() -> None:
    """A drafter copied from another artifact has a record but no parameters to assign or share."""
    model = _nvidia_model()
    model.components["dflash2"] = {"config": {"num_hidden_layers": 1}}
    model.grafted = ("dflash2",)
    recipe = Recipe(model)
    qwen3_8_27b_nvfp4_nvidia(model, recipe, {"quantized": object()})
    assert not any(name.startswith("dflash2/") for name in recipe.selections)


def test_nvidia_recipe_stores_the_dflash2_gate_up_as_nvfp4() -> None:
    model = _nvidia_model()
    model.components["dflash2"] = {"config": {"num_hidden_layers": 1}}
    for name in (
        "dflash2/layers/0/attention/key",
        "dflash2/layers/0/attention/value",
        "dflash2/layers/0/attention/context_key",
        "dflash2/layers/0/attention/context_value",
        "dflash2/layers/0/mlp/gate",
        "dflash2/layers/0/mlp/up",
        "dflash2/layers/0/mlp/down",
    ):
        source = array_source(torch.ones((128, 64), dtype=torch.bfloat16), name)
        model.add(
            Parameter(
                name,
                (128, 64),
                source,
                source_factory=lambda store, fmt, _source=source: _source,
                inputs=("input",),
            )
        )
    recipe = Recipe(model)
    qwen3_8_27b_nvfp4_nvidia(model, recipe, {"quantized": object()})

    for role in ("gate", "up"):
        selection = recipe.selections[f"dflash2/layers/0/mlp/{role}"][0]
        assert selection.format == "nvfp4"
        assert selection.method is nvfp4_mse
        assert recipe.policies[(f"dflash2/layers/0/mlp/{role}", "input")] == "A16Only"
    down = recipe.selections["dflash2/layers/0/mlp/down"][0]
    assert down.format == Q8
    assert down.method is grouped_absmax
