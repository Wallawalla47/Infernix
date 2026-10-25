"""The heads' 2-bit screens (tools/convert/screen.py, --screen-heads): two extra q2_g64_fp16 parameters,
the LM head's rows and the proposal head's, fitted by grouped_mse_norm from the LM head's source values
and emitted after every other object, so the recipe's own objects keep their ids, order and bytes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.convert.methods import grouped_mse_norm
from tools.convert.model import Model, Parameter
from tools.convert.proposal import add_proposal
from tools.convert.recipe import Recipe
from tools.convert.resources import token_domain
from tools.convert.screen import OUTPUT_HEAD, PROPOSAL_HEAD, add_screen_heads
from tools.convert.sources.logical import array_source

VOCAB, HIDDEN = 8, 128


def _model(tmp_path, *, proposal: bool) -> tuple[Model, Recipe, torch.Tensor]:
    rng = np.random.default_rng(3)
    head = torch.from_numpy((rng.standard_normal((VOCAB, HIDDEN)) * 0.02).astype(np.float32)).to(torch.bfloat16)
    count, special = token_domain(
        {"model": {"vocab": {str(i): i for i in range(VOCAB)}}, "added_tokens": []},
        {"added_tokens_decoder": {}},
        vocab_size=VOCAB,
    )
    model = Model(
        {
            "text": {"config": {"architectures": ["Qwen4ExpForCausalLM"], "vocab_size": VOCAB}},
            "mtp": {"config": {}, "target": "text"},
        },
        token_count=count,
        special_token_ids=special,
    )
    model.add(Parameter("text/output_head", (VOCAB, HIDDEN), array_source(head, "head"), inputs=("text/final_hidden",)))
    norm = torch.ones(HIDDEN, dtype=torch.bfloat16)
    model.add(Parameter("text/final_norm", (HIDDEN,), array_source(norm, "norm")))
    recipe = Recipe(model)
    recipe.assign("text/output_head", format="q8_g32_fp16", method="grouped_absmax")
    if proposal:
        ranking = tmp_path / "ranking.i64"
        np.array([[9, 1, 8, 2, 7, 3, 6, 4]], dtype="<i8").tofile(ranking)
        add_proposal(recipe, ranking=ranking, rows=4)
    return model, recipe, head


def test_screens_follow_every_other_parameter_and_leave_its_objects(tmp_path) -> None:
    model, recipe, head = _model(tmp_path, proposal=True)
    before = recipe.prepare(device="cpu")
    add_screen_heads(recipe)
    after = recipe.prepare(device="cpu")
    # The recipe's own objects: the same ids, shapes, formats, layouts and bindings, in the same order.
    assert [job.spec for job in after.weights[: len(before.weights)]] == [job.spec for job in before.weights]
    assert {k: v for k, v in after.bindings.items() if not k.startswith("screen/")} == before.bindings
    screens = after.weights[len(before.weights):]
    assert [job.parameters for job in screens] == [(OUTPUT_HEAD,), (PROPOSAL_HEAD,)]
    for job in screens:
        assert job.spec.format == "q2_g64_fp16" and job.spec.layout == "row_split_k128_v1"
        assert job.method_name == grouped_mse_norm.__name__
    assert screens[0].spec.shape == (VOCAB, HIDDEN) and screens[1].spec.shape == (4, HIDDEN)
    # No Use: a screen is read by the screened heads, not computed as a projection.
    assert all(use["parameter"] not in (OUTPUT_HEAD, PROPOSAL_HEAD) for use in after.uses)
    # The proposal screen holds the proposal's rows, in its order, from the LM head's source values.
    ids = model.parameters["proposal/token_ids"].source.values().tolist()
    rows = model.parameters[PROPOSAL_HEAD].source.values().reshape(4, HIDDEN)
    assert torch.equal(rows, head[ids])
    assert torch.equal(model.parameters[OUTPUT_HEAD].source.values().reshape(VOCAB, HIDDEN), head)


def test_screens_without_a_proposal_screen_the_lm_head_only(tmp_path) -> None:
    model, recipe, _ = _model(tmp_path, proposal=False)
    add_screen_heads(recipe)
    assert OUTPUT_HEAD in model.parameters and PROPOSAL_HEAD not in model.parameters
    with pytest.raises(ValueError, match="already configured"):
        add_screen_heads(recipe)


def test_screens_need_a_qwen4exp_model(tmp_path) -> None:
    model, recipe, _ = _model(tmp_path, proposal=False)
    model.components["text"]["config"]["architectures"] = ["Qwen3_5ForCausalLM"]
    with pytest.raises(ValueError, match="Qwen4Exp"):
        add_screen_heads(recipe)
