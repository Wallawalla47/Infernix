"""The shipped expert routing profile (docs/maintainer/artifact-container.md §9.3):
the converter embeds a provided profile as the text component's `expert_profile` resource, the
generic reader round-trips the bytes, and tools.expert_cache_replay.profile produces profiles from
recorded route traces and from an engine saved-state sidecar."""

from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from tools.artifact.expert_profile import MAGIC, OBJECT_ID, ROLE, U32_MAX, encode, parse
from tools.artifact.reader import Artifact
from tools.convert.model import Model
from tools.convert.pipeline import convert
from tools.convert.qwen4_exp import attach_expert_profile
from tools.convert.recipe import Recipe
from tools.expert_cache_replay import host_tier
from tools.expert_cache_replay.profile import profile_from_expert_state, profile_from_traces

LAYERS, EXPERTS = 2, 4
KEYS = LAYERS * EXPERTS
COUNTS = [8, 0, 3, 0, 7, 0, 1, 0]
RANKED = [4, 0, 2]


def test_encode_parse_round_trip():
    data = encode(COUNTS, RANKED)
    assert data[:8] == MAGIC
    assert parse(data, layers=LAYERS, experts=EXPERTS) == (COUNTS, RANKED)
    assert parse(encode(COUNTS, []), layers=LAYERS, experts=EXPERTS) == (COUNTS, [])


def test_parse_refuses_damaged_blobs():
    data = encode(COUNTS, RANKED)
    with pytest.raises(ValueError, match="expert profile"):
        parse(b"X" + data[1:], layers=LAYERS, experts=EXPERTS)
    with pytest.raises(ValueError, match="version"):
        parse(data[:8] + struct.pack("<I", 2) + data[12:], layers=LAYERS, experts=EXPERTS)
    with pytest.raises(ValueError, match="length differs"):
        parse(data + b"\0\0\0\0", layers=LAYERS, experts=EXPERTS)
    with pytest.raises(ValueError, match="at least"):
        parse(data[:12], layers=LAYERS, experts=EXPERTS)


def test_parse_refuses_another_model_shape():
    with pytest.raises(ValueError, match="ranks"):
        parse(encode(COUNTS, RANKED), layers=LAYERS + 1, experts=EXPERTS)


def test_encode_validates_the_ranking():
    with pytest.raises(ValueError, match="outside"):
        encode(COUNTS, [KEYS])
    with pytest.raises(ValueError, match="repeats"):
        encode(COUNTS, [0, 0])
    with pytest.raises(ValueError, match="ranks"):
        encode(COUNTS, list(range(KEYS + 1)))
    with pytest.raises(ValueError, match="u32"):
        encode([U32_MAX + 1] * KEYS, [])
    with pytest.raises(ValueError, match="at least one key"):
        encode([], [])


def _model(tmp_path: Path) -> Model:
    model = Model(
        {
            "text": {
                "config": {
                    "architectures": ["Qwen4ExpForCausalLM"],
                    "model_type": "qwen4_exp_text",
                    "num_hidden_layers": LAYERS,
                    "num_experts": EXPERTS,
                }
            }
        }
    )
    (tmp_path / "profile.bin").write_bytes(encode(COUNTS, RANKED))
    return model


def test_converter_embeds_the_profile_section(tmp_path):
    model = _model(tmp_path)
    attach_expert_profile(model, tmp_path / "profile.bin")
    assert model.components["text"]["resources"][ROLE] == OBJECT_ID
    assert model.resources[OBJECT_ID] == encode(COUNTS, RANKED)
    out = tmp_path / "pack.infernix"
    report = convert(
        model,
        Recipe(model),
        out,
        device="cpu",
        provenance={"expert_profile": str(tmp_path / "profile.bin")},
    )
    with Artifact.open(out) as artifact:
        assert artifact.directory.components["text"]["resources"][ROLE] == OBJECT_ID
        obj = artifact.object(OBJECT_ID)
        assert obj.kind == "resource" and obj.encoding == "raw_bytes_v1"
        assert parse(artifact.read_object(OBJECT_ID), layers=LAYERS, experts=EXPERTS) == (COUNTS, RANKED)
    written = json.loads(Path(str(out) + ".conversion.json").read_text())
    assert written["provenance"]["expert_profile"].endswith("profile.bin")
    assert report["objects"] == 1  # just the profile resource


def test_attach_refuses_a_mismatched_profile(tmp_path):
    model = _model(tmp_path)
    (tmp_path / "wide.bin").write_bytes(encode([1] * (KEYS + EXPERTS), []))
    with pytest.raises(ValueError, match="layers x"):
        attach_expert_profile(model, tmp_path / "wide.bin")


def _round(live, routes):
    columns = len(live)
    return host_tier.Round(
        kind=host_tier.DECODE, rows=1, width=columns, tokens=1, budget=0, position=0,
        lanes=np.array([7], np.int32), live=np.array(live, bool),
        routes=np.asarray(routes, np.int32),
    )


def test_profile_from_traces_ranks_live_uses(tmp_path):
    # Two decode rounds over one session; the dead columns' routes must not count.
    round_a = _round([True, True, False],
                     [[[1, 0], [2, 2], [3, 3]], [[0, 1], [1, 1], [2, 2]]])
    round_b = _round([True, False, True],
                     [[[1, 0], [3, 3], [0, 1]], [[2, 3], [2, 2], [3, 3]]])
    trace = tmp_path / "trace.bin"
    host_tier.write_trace(trace, {"layers": LAYERS, "experts": EXPERTS, "top_k": 2}, [round_a, round_b])
    counts, ranked = parse(profile_from_traces([trace]), layers=LAYERS, experts=EXPERTS)
    # layer 0: k0 1+2, k1 1+2, k2 2; layer 1: k4 1, k5 3, k6 1, k7 3; k3 unused.
    assert counts == [3, 3, 2, 0, 1, 3, 1, 3]
    # Hottest first; ties and the unused tail keep the engine's layer-interleaved spread order.
    assert ranked == [0, 1, 5, 7, 2, 4, 6, 3]


def test_profile_from_traces_requires_matching_models(tmp_path):
    small = host_tier.Round(
        kind=host_tier.DECODE, rows=1, width=1, tokens=1, budget=0, position=0,
        lanes=np.array([0], np.int32), live=np.array([True]),
        routes=np.array([[[0, 0]]], np.int32),
    )
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    host_tier.write_trace(a, {"layers": LAYERS, "experts": EXPERTS, "top_k": 2},
                          [_round([True], [[[0, 0]], [[1, 1]]])])
    host_tier.write_trace(b, {"layers": 1, "experts": EXPERTS, "top_k": 2}, [small])
    with pytest.raises(ValueError, match="different models"):
        profile_from_traces([a, b])


def test_profile_from_expert_state_strips_identity(tmp_path):
    counts, ranked = [3, 1, 0, 9, 2, 0, 0, 0], [3, 0, 4]
    identity = b"artifact=/x/pack.infernix;size=1;mtime=2"
    sidecar = (b"NINFQ4ES" + struct.pack("<II", 1, len(identity)) + identity +
               struct.pack("<I", KEYS) + struct.pack(f"<{KEYS}I", *counts) +
               struct.pack("<I", len(ranked)) + struct.pack(f"<{len(ranked)}I", *ranked) + b"NINFQ4ES")
    path = tmp_path / "pack.expert-state"
    path.write_bytes(sidecar)
    assert parse(profile_from_expert_state(path), layers=LAYERS, experts=EXPERTS) == (counts, ranked)


def test_converter_cli_refuses_the_profile_for_other_models(tmp_path):
    # The Qwen4Exp gate in tools/convert: a non-Qwen4Exp source must refuse --expert-profile
    # before any conversion work happens.
    from tools.convert import __main__ as convert_main

    source = tmp_path / "qwen35"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3_5ForCausalLM"], "hidden_size": 128, "vocab_size": 8,
        "num_hidden_layers": 1, "max_position_embeddings": 128, "layer_types": ["full_attention"],
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8, "intermediate_size": 24,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    }))
    for name, payload in {
        "tokenizer.json": {"model": {"vocab": {}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (source / name).write_text(json.dumps(payload))
    profile = tmp_path / "profile.bin"
    profile.write_bytes(encode([0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7]))
    out = tmp_path / "pack.ninfer"
    recipe = tmp_path / "recipe.py"
    recipe.write_text("")  # never reached: the refusal fires while picking the model branch
    with pytest.raises(ValueError, match="--expert-profile applies only to Qwen4Exp sources"):
        convert_main.main(["--model", str(source), "--out", str(out), "--recipe", str(recipe),
                           "--expert-profile", str(profile)])
    assert not out.exists()
