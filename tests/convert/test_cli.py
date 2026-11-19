from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.__main__ import _recipe_parts


def test_recipe_parts_keeps_path_colons():
    # A Windows drive colon is not the FILE[:function] separator.
    assert _recipe_parts(r"C:\Users\Example User\temp\recipe.py") == (
        r"C:\Users\Example User\temp\recipe.py",
        "configure",
    )
    assert _recipe_parts(r"C:\Users\Example User\temp\recipe.py:custom") == (
        r"C:\Users\Example User\temp\recipe.py",
        "custom",
    )
    assert _recipe_parts("recipe.py") == ("recipe.py", "configure")
    assert _recipe_parts("recipe.py:custom") == ("recipe.py", "custom")
    # A colon inside a POSIX path component stays in the path too.
    assert _recipe_parts("/tmp/col:oned/recipe.py") == (
        "/tmp/col:oned/recipe.py",
        "configure",
    )


def _tiny_qwen3_5_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 128,
        "vocab_size": 8,
        "num_hidden_layers": 1,
        "max_position_embeddings": 128,
        "layer_types": ["full_attention"],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 24,
        "rope_parameters": {"partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]},
    }
    (source / "config.json").write_text(json.dumps(config))
    for name, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(6)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (source / name).write_text(json.dumps(value))
    return source


def _tiny_dflash2_source(tmp_path: Path) -> Path:
    """A DFlash2 drafter config for _tiny_qwen3_5_source's target (its values come from the recipe)."""
    draft = tmp_path / "dflash2"
    draft.mkdir()
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": 128,
        "vocab_size": 8,
        "num_target_layers": 1,
        "num_hidden_layers": 1,
        "intermediate_size": 24,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "layer_types": ["sliding_attention"],
        "sliding_window": 32,
        "dflash_config": {
            "target_layer_ids": [0],
            "mask_token_id": 7,
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 2,
        },
    }
    (draft / "config.json").write_text(json.dumps(config))
    return draft


# Distinct values per parameter, so a copied object is told apart from a converted one.
_VALUES_RECIPE = """import torch
import zlib
from tools.convert.sources.logical import array_source

def configure(model, recipe, sources):
    for name, parameter in model.parameters.items():
        count = 1
        for size in parameter.shape:
            count *= size
        seed = zlib.crc32(name.encode()) % 97
        value = ((torch.arange(count) + seed) % 13).float().div(8).reshape(parameter.shape).to(torch.bfloat16)
        recipe.assign(name, source=array_source(value, name))
"""


def _drafter(artifact: Artifact, component: str = "dflash2"):
    """Its record, Uses and, per parameter, the bytes and element ranges of what it binds."""
    directory = artifact.directory
    words = {}
    for name, binding in directory.bindings.items():
        if name.startswith(component + "/"):
            parts = binding_parts(binding, artifact.by_id)
            words[name] = [(artifact.read_object(obj), begin, end) for obj, begin, end in parts]
    uses = sorted(
        (u["parameter"], u["input"], u["activation_policy"])
        for u in directory.uses
        if u["parameter"].startswith(component + "/")
    )
    return directory.components[component], uses, words


def test_cli_graft_copies_a_drafter_byte_for_byte(tmp_path):
    source = _tiny_qwen3_5_source(tmp_path)
    (source / "chat_template.jinja").write_text("{{ messages }}")
    draft = _tiny_dflash2_source(tmp_path)
    recipe = tmp_path / "recipe.py"
    recipe.write_text(_VALUES_RECIPE, encoding="utf-8")
    reference = tmp_path / "reference.infernix"
    common = ["--model", str(source), "--recipe", str(recipe), "--device", "cpu"]
    result = _convert(
        [*common, "--components", "text,dflash2", "--source", f"dflash2={draft}", "--out", str(reference)]
    )
    assert result.returncode == 0, result.stderr

    grafted = tmp_path / "grafted.infernix"
    result = _convert([*common, "--graft", f"dflash2={reference}", "--out", str(grafted)])
    assert result.returncode == 0, result.stderr
    with Artifact(reference) as before, Artifact(grafted) as after:
        assert _drafter(after) == _drafter(before)
        assert set(after.directory.components) == {"text", "dflash2"}
        # The new artifact's own head reads the drafter's final hidden state, as a converted drafter's does.
        heads = {(u["parameter"], u["input"]) for u in after.directory.uses}
        assert ("text/output_head", "dflash2/final_hidden") in heads
        provenance = after.directory.provenance["grafts"]["dflash2"]
        assert provenance["artifact_id"] == before.artifact_id.hex()
    report = json.loads(Path(str(grafted) + ".conversion.json").read_text(encoding="utf-8"))
    (copied,) = report["grafts"]
    assert copied["component"] == "dflash2" and copied["objects"]

    # A drafter of another target's geometry is refused before anything is written.
    narrow = tmp_path / "narrow"
    narrow.mkdir()
    config = json.loads((source / "config.json").read_text())
    (narrow / "config.json").write_text(json.dumps({**config, "hidden_size": 64}))
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        (narrow / name).write_bytes((source / name).read_bytes())
    refused = tmp_path / "refused.infernix"
    result = _convert(
        ["--model", str(narrow), "--recipe", str(recipe), "--device", "cpu", "--graft", f"dflash2={reference}",
         "--out", str(refused)]
    )
    assert result.returncode != 0
    assert "grafted dflash2/" in result.stderr and "elements" in result.stderr, result.stderr
    assert not refused.exists()

    # A drafter is grafted or converted, not both.
    result = _convert(
        [*common, "--components", "text,dflash2", "--source", f"dflash2={draft}", "--graft",
         f"dflash2={reference}", "--out", str(tmp_path / "both.infernix")]
    )
    assert result.returncode != 0
    assert "a grafted drafter is not also converted" in result.stderr, result.stderr


def _convert(arguments: list[str]) -> subprocess.CompletedProcess:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, "-B", "-m", "tools.convert", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_cli_ngram_reuse_refused_for_qwen3_5(tmp_path):
    # --ngram-reuse binds a Qwen4Exp n-gram volume; a Qwen3.5 source refuses it, as it refuses --ngram-out,
    # instead of converting without it.
    source = _tiny_qwen3_5_source(tmp_path)
    volume = tmp_path / "table.ngram"
    volume.write_bytes(b"")
    recipe = tmp_path / "recipe.py"
    recipe.write_text(
        "import torch\n"
        "from tools.convert.sources.logical import array_source\n\n"
        "def configure(model, recipe, sources):\n"
        "    for name, parameter in model.parameters.items():\n"
        "        recipe.assign(name, source=array_source(torch.ones(parameter.shape, dtype=torch.bfloat16), 'ones'))\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.infernix"
    result = _convert(
        ["--model", str(source), "--recipe", str(recipe), "--out", str(output), "--device", "cpu",
         "--ngram-reuse", str(volume)]
    )
    assert result.returncode != 0
    assert "--ngram-reuse applies only to Qwen4Exp sources" in result.stderr, result.stderr
    assert not output.exists()


def test_cli_custom_sources_method_template_and_shards(tmp_path):
    source = _tiny_qwen3_5_source(tmp_path)
    template = tmp_path / "template.jinja"
    template.write_text("custom {{ messages }}")
    recipe = tmp_path / "recipe.py"
    recipe.write_text("""import torch
from tools.convert.sources.logical import array_source

def custom(request):
    return request.job(produce=lambda output: output.write_values(0, request.source.values().to(torch.bfloat16)))

def configure(model, recipe, sources):
    for name, parameter in model.parameters.items():
        value = torch.ones(parameter.shape, dtype=torch.bfloat16)
        recipe.assign(name, source=array_source(value, "my-importer"))
    recipe.assign(("text/token_embedding", "text/layers/0/mlp/gate", "text/layers/0/mlp/up"), method=custom)
    for role, format in zip(("query", "key", "gate", "value"), ("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16")):
        recipe.assign("text/layers/0/attention/" + role, format=format, method="grouped_absmax")
""")
    output = tmp_path / "custom.ninfer"
    arguments = [
        "--model",
        str(source),
        "--recipe",
        str(recipe),
        "--out",
        str(output),
        "--name",
        "my-training-run",
        "--resource",
        f"chat_template.jinja={template}",
        "--source",
        f"unused={tmp_path / 'missing'}",
        "--device",
        "cpu",
        "--max-file-bytes",
        "16384",
        "--rows-per-chunk",
        "3",
    ]
    result = _convert(arguments)
    assert result.returncode == 0, result.stderr
    with Artifact(output) as artifact:
        assert set(artifact.directory.components) == {"text"}
        assert artifact.directory.metadata == {"name": "my-training-run"}
        assert len(artifact.directory.files) > 1
        resource = artifact.directory.components["text"]["resources"][
            "chat_template.jinja"
        ]
        assert artifact.read_object(resource) == template.read_bytes()
        embedding = binding_parts(
            artifact.directory.bindings["text/token_embedding"], artifact.by_id
        )[0][0]
        assert artifact.read_object(embedding) == b"\x80\x3f" * (8 * 128)
        parents = []
        for role in ("query", "key", "gate", "value"):
            name = "text/layers/0/attention/" + role
            parents.append(
                binding_parts(artifact.directory.bindings[name], artifact.by_id)[0][0]
            )
        assert len(set(parents)) == 4
        assert [artifact.object(parent).format for parent in parents] == [
            "q4_g64_fp16",
            "q5_g64_fp16",
            "q6_g64_fp16",
            "q8_g32_fp16",
        ]
        gate = binding_parts(
            artifact.directory.bindings["text/layers/0/mlp/gate"], artifact.by_id
        )[0][0]
        up = binding_parts(
            artifact.directory.bindings["text/layers/0/mlp/up"], artifact.by_id
        )[0][0]
        assert gate != up
