"""Exact reading of NVIDIA's Qwen3.8-Flash-Next-FP4-Local tensor encodings (design §6.1.2): the
NVFP4-CSF compressed expert scales, the MXFP8 projections (``fp8_e4m3fn_g32_e8m0``) and the weight-only
NVFP4 vision and MTP matrices, and the conversion of each into the artifact (the vision MLPs and merger
word for word as dense ``nvfp4_mul`` matrices in ``nvfp4_rg16_v1``)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.fp8_g32 import (
    decode_fp8_group_scaled_words,
    encode_fp8_group_scaled,
    fp8_group_values,
    rows16_scale_plane,
)
from tools.artifact.formats import decode_e4m3fn_word
from tools.artifact.layouts import group_scale_geometry
from tools.artifact.reader import Artifact
from tools.convert.methods import bf16_from_float64, cast_direct, import_encoded
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.compressed_tensors import matrix_source
from tools.convert.sources.logical import select_rows
from tools.convert.sources.modelopt import csf_scale_words, nvfp4_matrix_words, nvfp4_mul_matrix_source
from tools.convert.sources.safetensors import SafetensorsSource

MXFP8 = "fp8_e4m3fn_g32_e8m0"


# ---------------------------------------------------------------------------- NVFP4-CSF


def _csf_encode(scales: np.ndarray, bases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """b12x's byte-window encoding (codec 0) of E4M3 scale words with the given row bases: offsets
    in [0, 15] over the base, every other byte an exception record."""
    rows, columns = scales.shape
    difference = scales.astype(np.int16) - bases[:, None]
    outside = (difference < 0) | (difference > 15)
    difference[outside] = 0
    difference = difference.astype(np.uint8)
    packed = difference[:, ::2] | (difference[:, 1::2] << 4)
    stream = np.concatenate((bases.reshape(-1, 16), packed.reshape(rows // 16, -1)), axis=1)
    positions = np.flatnonzero(outside).astype(np.uint32)
    words = positions | (scales.ravel()[positions].astype(np.uint32) << 24)
    return stream, words


def _csf_checkpoint(tmp_path, rows: int, k: int, *, fixed_2d: bool, seed: int = 5):
    rng = np.random.default_rng(seed)
    columns = k // 16
    scales = rng.integers(0x30, 0x7E, size=(rows, columns), dtype=np.uint8)
    scales[0, 0], scales[-1, -1] = 0x00, 0x7E  # exceptions at the first and last positions
    bases = np.clip(np.median(scales, axis=1).astype(np.int64) - 7, 0, 240).astype(np.uint8)
    bases[0] = 240  # the largest admitted base
    fixed, exceptions = _csf_encode(scales, bases)
    leaf = "model.language_model.layers.0.mlp.experts.0.gate_proj"
    codes = torch.from_numpy(rng.integers(0, 256, size=(rows, k // 2), dtype=np.uint8))
    tensors = {
        leaf + ".weight": codes,
        leaf + ".weight_scale.nvfp4_csf_fixed": torch.from_numpy(fixed if fixed_2d else fixed.reshape(-1)),
        leaf + ".weight_scale.nvfp4_csf_exceptions": torch.from_numpy(exceptions.astype(np.uint32)),
        leaf + ".weight_scale_2": torch.tensor(1.0 / 3.0, dtype=torch.float32),
        leaf + ".input_scale": torch.tensor(0.01, dtype=torch.float32),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return leaf, codes, torch.from_numpy(scales), int(len(exceptions))


@pytest.mark.parametrize("fixed_2d", [False, True])
def test_csf_scales_restore_every_scale_word(tmp_path, fixed_2d):
    leaf, codes, scales, exceptions = _csf_checkpoint(tmp_path, 64, 512, fixed_2d=fixed_2d)
    assert exceptions > 2  # the fixture exercises the exception stream
    with SafetensorsSource(tmp_path) as store:
        assert torch.equal(csf_scale_words(store, leaf, 64, 32), scales)
        words = nvfp4_matrix_words(store, leaf, (64, 512))
    assert torch.equal(words.codes, codes) and torch.equal(words.scales, scales)
    assert words.weight_scale_2.view(torch.int32) == torch.tensor(1.0 / 3.0, dtype=torch.float32).view(torch.int32)
    assert words.input_scale is not None


def test_csf_refuses_unordered_exceptions_and_oversized_bases(tmp_path):
    leaf, _, _, _ = _csf_checkpoint(tmp_path, 32, 256, fixed_2d=True)
    path = tmp_path / "model.safetensors"
    with SafetensorsSource(tmp_path) as store:
        fixed = store.read_flat(leaf + ".weight_scale.nvfp4_csf_fixed").clone()
        exceptions = store.read_flat(leaf + ".weight_scale.nvfp4_csf_exceptions").clone()
        others = {name: store.read_flat(name).clone().reshape(store.describe(name).shape)
                  for name in (leaf + ".weight", leaf + ".weight_scale_2", leaf + ".input_scale")}
    for broken_fixed, broken_exceptions, message in (
        (fixed, torch.from_numpy(exceptions.numpy()[::-1].copy()), "increasing"),
        (torch.where(torch.arange(fixed.numel()) == 1, torch.tensor(241, dtype=torch.uint8), fixed), exceptions,
         "exceeds 240"),
    ):
        path.unlink()
        save_file({**others, leaf + ".weight_scale.nvfp4_csf_fixed": broken_fixed,
                   leaf + ".weight_scale.nvfp4_csf_exceptions": broken_exceptions}, str(path))
        with SafetensorsSource(tmp_path) as store, pytest.raises(ValueError, match=message):
            csf_scale_words(store, leaf, 32, 16)


def test_weight_only_nvfp4_values_use_the_multiplier_exactly(tmp_path):
    rng = np.random.default_rng(9)
    n, k = 32, 64
    codes = torch.from_numpy(rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8))
    scales = torch.from_numpy(rng.integers(0x20, 0x7E, size=(n, k // 16), dtype=np.uint8))
    leaf = "model.visual.blocks.0.mlp.linear_fc1"
    save_file({leaf + ".weight": codes, leaf + ".weight_scale": scales.view(torch.float8_e4m3fn),
               leaf + ".weight_scale_2": torch.tensor(1.0 / 7.0, dtype=torch.float32)},
              str(tmp_path / "model.safetensors"))
    multiplier = Fraction(float(torch.tensor(1.0 / 7.0, dtype=torch.float32)))
    e2m1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    with SafetensorsSource(tmp_path) as store:
        source = nvfp4_mul_matrix_source(store, leaf, (n, k))
        values = source.values()
        assert values.dtype == torch.float64
        assert source.values(5 * k + 3, 9 * k + 1).equal(values[5 * k + 3 : 9 * k + 1])
        # The stored words, unchanged: codes, block scales and the multiplier's FP32 word.
        rows = source.read_encoded(5, 9)
        assert rows.format == "nvfp4_mul" and rows.weight_divisor is None
        assert torch.equal(rows.codes, codes[5:9]) and torch.equal(rows.scales, scales[5:9])
        assert rows.weight_multiplier == torch.tensor(1.0 / 7.0, dtype=torch.float32).reshape(1).view(torch.uint8).numpy().tobytes()
    for row, col in ((0, 0), (3, 17), (31, 63)):
        nibble = (int(codes[row, col // 2]) >> (4 * (col % 2))) & 15
        code = Fraction(e2m1[nibble & 7]) * (-1 if nibble & 8 else 1)
        exact = code * Fraction(decode_e4m3fn_word(int(scales[row, col // 16]))) * multiplier
        assert Fraction(float(values[row * k + col])) == exact  # 30 significant bits: exact in binary64


# ---------------------------------------------------------------------------- MXFP8


def _mxfp8_checkpoint(tmp_path, matrices: dict[str, tuple[int, int]], seed: int = 3):
    rng = np.random.default_rng(seed)
    tensors, words = {}, {}
    for prefix, (n, k) in matrices.items():
        codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
        codes[(codes & 0x7F) == 0x7F] = 0x7E  # finite E4M3FN words
        scales = rng.integers(110, 130, size=(n, k // 32), dtype=np.uint8)
        tensors[prefix + ".weight"] = torch.from_numpy(codes).view(torch.float8_e4m3fn)
        tensors[prefix + ".weight_scale"] = torch.from_numpy(scales)
        words[prefix] = (torch.from_numpy(codes), torch.from_numpy(scales))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    return words


def test_mxfp8_matrices_resolve_to_their_codes_and_ue8m0_scales(tmp_path):
    prefix = "model.language_model.layers.0.linear_attn.in_proj_z"
    words = _mxfp8_checkpoint(tmp_path, {prefix: (48, 128)})[prefix]
    with SafetensorsSource(tmp_path) as store:
        source = matrix_source(store, prefix + ".weight", (48, 128))
        rows = source.read_encoded(16, 48)
        assert rows.format == MXFP8
        assert torch.equal(rows.codes, words[0][16:48]) and torch.equal(rows.scales, words[1][16:48])
        values = source.values()
    code, scale = int(words[0][7, 77]), int(words[1][7, 77 // 32])
    assert Fraction(float(values[7 * 128 + 77])) == Fraction(decode_e4m3fn_word(code)) * Fraction(2) ** (scale - 127)
    assert torch.equal(values.reshape(48, 128), fp8_group_values(*words))


def test_group_scaled_codec_stores_scales_in_16_row_tiles():
    rng = np.random.default_rng(1)
    codes = torch.from_numpy(rng.integers(0, 0x7E, size=(32, 96), dtype=np.uint8))
    scales = torch.from_numpy(rng.integers(3, 247, size=(32, 3), dtype=np.uint8))
    scales[0, 0], scales[31, 2] = 3, 246  # the admitted extremes
    payload = encode_fp8_group_scaled(codes, scales, (32, 96))
    geometry = group_scale_geometry(MXFP8, (32, 96))
    assert len(payload) == geometry.payload_bytes == 32 * 96 + 32 * 3  # 3072 is 256-aligned
    plane = payload[geometry.scale_plane_offset :]
    for row, group in ((0, 0), (5, 2), (16, 1), (31, 2)):
        assert plane[(row // 16) * 3 * 16 + group * 16 + row % 16] == int(scales[row, group])
    decoded = decode_fp8_group_scaled_words(payload, (32, 96))
    assert torch.equal(decoded[0], codes) and torch.equal(decoded[1], scales)
    for word in (0xFF, 247, 2):
        with pytest.raises(ValueError):
            encode_fp8_group_scaled(codes, torch.full((32, 3), word, dtype=torch.uint8), (32, 96))
    assert torch.equal(rows16_scale_plane(scales), torch.frombuffer(bytearray(plane), dtype=torch.uint8))


def _object_bytes(path):
    with Artifact(path) as artifact:
        return {obj.id: (obj, b"".join(artifact.iter_range(obj.offset, obj.bytes))) for obj in artifact.objects}


def test_grouped_mxfp8_rows_and_bf16_values_convert_exactly(tmp_path):
    # Two row-split members of one parent (as the GDN q/k/v/z and QSA query/gate/key/value), and a
    # small MXFP8 matrix cast to its exact BF16 values (as GDN a/b and the indexer).
    q, z, a = "layers.0.in_proj_qkv", "layers.0.in_proj_z", "layers.0.in_proj_a"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    words = _mxfp8_checkpoint(source_dir, {q: (64, 64), z: (48, 64), a: (16, 64)})
    with SafetensorsSource(source_dir) as store:
        model = Model({"text": {"config": {}}})
        for name, (leaf, shape, rows) in {
            "text/query": (q, (32, 64), ((0, 16), (48, 64))),  # rows of two spans
            "text/z": (z, (48, 64), None),
            "text/a": (a, (16, 64), None),
        }.items():
            reference = matrix_source(store, leaf + ".weight", (64, 64) if leaf == q else shape)
            source = select_rows(reference, rows) if rows else reference

            def factory(selected, format=None, leaf=leaf, shape=shape, rows=rows):
                found = matrix_source(selected, leaf + ".weight", (64, 64) if leaf == q else shape, format)
                return select_rows(found, rows) if rows else found

            model.add(Parameter(name, shape, source, factory, ("input",), "bf16"))
        recipe = Recipe(model)
        for name in ("text/query", "text/z"):
            recipe.assign(name, format=MXFP8, method=import_encoded, source=model.source(name, store, MXFP8),
                          activation_policy="A16Only")
        recipe.group(["text/query", "text/z"])
        convert(model, recipe, tmp_path / "out.infernix", device="cpu", rows_per_chunk=16)
    objects = _object_bytes(tmp_path / "out.infernix")
    parents = [(obj, data) for obj, data in objects.values() if obj.format == MXFP8]
    assert len(parents) == 1
    obj, data = parents[0]
    codes, scales = decode_fp8_group_scaled_words(data, obj.shape)
    q_codes, q_scales = words[q]
    expect_codes = torch.cat((q_codes[0:16], q_codes[48:64], words[z][0]))
    expect_scales = torch.cat((q_scales[0:16], q_scales[48:64], words[z][1]))
    assert torch.equal(codes, expect_codes) and torch.equal(scales, expect_scales)
    cast = [data for obj, data in objects.values() if obj.format == "bf16" and tuple(obj.shape) == (16, 64)]
    assert len(cast) == 1
    stored = torch.frombuffer(bytearray(cast[0]), dtype=torch.bfloat16).reshape(16, 64)
    assert torch.equal(stored.to(torch.float64), fp8_group_values(*words[a]))  # MXFP8 values are BF16 values


def test_vision_nvfp4_matrices_convert_word_for_word(tmp_path):
    """The recipe's vision rule (official_recipes._fp4_local_vision) through the converter: each
    weight-only NVFP4 vision matrix becomes a dense nvfp4_mul matrix in nvfp4_rg16_v1 whose E2M1
    codes, E4M3 block scales (plain, or restored from NVFP4-CSF) and FP32 multiplier word are the
    checkpoint's, at a K of an odd block count (as fc2's 4304) and over several 16-row chunks; its
    represented weights are the checkpoint's exact values; its Use runs 16-bit activations; a BF16
    neighbour keeps its bytes; verify_artifact's check accepts it."""

    from tools.artifact.codecs.nvfp4_expert import decode_nvfp4_rg16_words, dequantize_nvfp4_mul
    from tools.convert.official_recipes import _fp4_local_vision
    from tools.convert.qwen4_exp import _Qwen4ExpBuilder
    from tools.flash_next.verify_artifact import verify_rg16

    rng = np.random.default_rng(11)
    fc1, fc2 = "model.visual.blocks.0.mlp.linear_fc1", "model.visual.merger.linear_fc2"
    matrices = {fc1: (48, 80), fc2: (32, 64)}  # K 80: five blocks, a partial 128-element stage
    names = {fc1: "vision/layers/0/mlp/fc1", fc2: "vision/merger/fc2"}
    tensors, words = {}, {}
    for leaf, (n, k) in matrices.items():
        codes = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
        scales = rng.integers(0x20, 0x7E, size=(n, k // 16), dtype=np.uint8)
        scales[0, 0], scales[-1, -1] = 0x00, 0x7E  # a zero scale and the largest finite one
        multiplier = torch.tensor(np.float32(rng.uniform(1e-4, 1e-2)))  # not a power of two
        tensors[leaf + ".weight"] = torch.from_numpy(codes)
        tensors[leaf + ".weight_scale_2"] = multiplier
        if leaf == fc2:  # the scales in NVFP4-CSF, as the checkpoint stores its experts'
            bases = np.clip(np.median(scales, axis=1).astype(np.int64) - 7, 0, 240).astype(np.uint8)
            fixed, exceptions = _csf_encode(scales, bases)
            tensors[leaf + ".weight_scale.nvfp4_csf_fixed"] = torch.from_numpy(fixed.reshape(-1))
            tensors[leaf + ".weight_scale.nvfp4_csf_exceptions"] = torch.from_numpy(exceptions.astype(np.uint32))
        else:
            tensors[leaf + ".weight_scale"] = torch.from_numpy(scales).view(torch.float8_e4m3fn)
        words[leaf] = (torch.from_numpy(codes), torch.from_numpy(scales), multiplier)
    bias = torch.from_numpy(rng.standard_normal(48).astype(np.float32)).to(torch.bfloat16)
    tensors[fc1 + ".bias"] = bias
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    save_file(tensors, str(source_dir / "model.safetensors"))
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                        dtype=torch.float64)
    with SafetensorsSource(source_dir) as store:
        model = Model({"text": {"config": {}}, "vision": {"config": {}, "target": "text"}})
        builder = _Qwen4ExpBuilder(model)
        for leaf, shape in matrices.items():
            builder.add(names[leaf], store, leaf + ".weight", shape, inputs=(names[leaf] + "_input",))
        builder.add("vision/layers/0/mlp/fc1_bias", store, fc1 + ".bias", (48,))
        recipe = Recipe(model)
        _fp4_local_vision(model, recipe, store)
        convert(model, recipe, tmp_path / "out.infernix", device="cpu", rows_per_chunk=16)
        with Artifact(tmp_path / "out.infernix") as artifact:
            bindings = artifact.directory.bindings
            policies = {(u["parameter"], u["input"]): u["activation_policy"] for u in artifact.directory.uses}
            for leaf, (n, k) in matrices.items():
                name = names[leaf]
                obj = artifact.object(bindings[name]["object"])
                assert (obj.format, obj.layout, tuple(obj.shape)) == ("nvfp4_mul", "nvfp4_rg16_v1", (n, k))
                stored = decode_nvfp4_rg16_words(artifact.read_object(obj.id), obj.shape)
                codes, scales, multiplier = words[leaf]
                assert torch.equal(stored["codes"], codes) and torch.equal(stored["scales"], scales)
                assert stored["multiplier"].view(torch.int32) == multiplier.view(torch.int32)
                nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(n, k).long()
                exact = (e2m1[nibbles] * scales.view(torch.float8_e4m3fn).double().repeat_interleave(16, dim=1)
                         * float(multiplier))
                assert torch.equal(dequantize_nvfp4_mul(stored["codes"], stored["scales"], float(stored["multiplier"])),
                                   exact)
                assert policies[(name, name + "_input")] == "A16Only"
                assert verify_rg16(artifact, store, bindings[name], model.parameters[name])
            kept = artifact.object(bindings["vision/layers/0/mlp/fc1_bias"]["object"])
            assert kept.format == "bf16" and artifact.read_object(kept.id) == bias.view(torch.int16).numpy().tobytes()


# ---------------------------------------------------------------------------- conversion helpers


def test_bf16_rounds_binary64_once_to_nearest_even():
    ties = torch.tensor([1 + 2.0**-8 + 2.0**-30, 1 + 2.0**-8, 1 + 3 * 2.0**-8, -(1 + 2.0**-8 + 2.0**-40)],
                        dtype=torch.float64)
    assert bf16_from_float64(ties).tolist() == [1.0078125, 1.0, 1.015625, -1.0078125]
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(200_000, generator=generator, dtype=torch.float64) * torch.exp2(
        torch.randint(-40, 40, (200_000,), generator=generator).double())
    rounded = bf16_from_float64(values)
    words = rounded.view(torch.int16)
    below = (words - 1).view(torch.bfloat16).double()
    above = (words + 1).view(torch.bfloat16).double()
    error = (rounded.double() - values).abs()
    assert bool(((error <= (below - values).abs()) & (error <= (above - values).abs())).all())
    with pytest.raises(ValueError):
        bf16_from_float64(torch.tensor([1e-300], dtype=torch.float64))
