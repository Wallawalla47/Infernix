"""Exact GGUF block import (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25).

The codec's decoder is checked against gguf-py's reference dequantizer on random blocks (every byte
pattern is a valid encoding once the binary16 scales are finite), the raw reader against a written
GGUF, and the expert-bank import against the stored bytes, record by record.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from tools.artifact.codecs import ggml
from tools.artifact.layouts import encoded_size, ggml_expert_geometry, ggml_rows_geometry, ggml_tiles16_geometry
from tools.convert.sources.gguf_raw import GgufStore


def _random_blocks(fmt: ggml.GgmlType, blocks: int, rng: np.random.Generator) -> np.ndarray:
    data = rng.integers(0, 256, size=(blocks, fmt.block_bytes), dtype=np.uint8)
    # Finite binary16 scales of moderate magnitude.
    words = (rng.integers(0, 1 << 10, size=(blocks, 2)) | (rng.integers(5, 21, size=(blocks, 2)) << 10)
             | (rng.integers(0, 2, size=(blocks, 2)) << 15)).astype("<u2")
    offsets = [208] if fmt is ggml.Q6_K else [0, 2] if fmt.has_min else [0]
    for i, offset in enumerate(offsets):
        data[:, offset : offset + 2] = words[:, i : i + 1].view(np.uint8).reshape(blocks, 2)
    return data


@pytest.mark.parametrize("name", sorted(ggml.GGML_TYPES))
def test_decoder_matches_the_reference_dequantizer(name):
    fmt = ggml.GGML_TYPES[name]
    rng = np.random.default_rng(fmt.ggml_id)
    k = fmt.block_elems * 4
    raw = _random_blocks(fmt, 3 * 4, rng)
    ours = ggml.decode_values(fmt, raw, 3, k)
    if fmt is ggml.Q2_0:
        # Not in gguf-py: the formula itself, (c - 1) * d over 2-bit codes.
        d = raw[:, 0:2].copy().view("<f2").astype(np.float64)[:, 0]
        codes = np.stack([(raw[:, 2 + j // 4] >> (2 * (j % 4))) & 3 for j in range(64)], axis=1).astype(np.float64) - 1
        assert np.array_equal(ours.reshape(-1, 64), codes * d[:, None])
        return
    ref = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType(fmt.ggml_id)).astype(np.float64).reshape(3, k)
    # gguf-py rounds each product to binary32; the exact values round to the same binary32 values.
    assert np.array_equal(ours.astype(np.float32), ref.astype(np.float32))


@pytest.mark.parametrize("name", sorted(ggml.GGML_TYPES))
def test_non_finite_scales_are_rejected(name):
    fmt = ggml.GGML_TYPES[name]
    rng = np.random.default_rng(11)
    k = fmt.block_elems * 2
    raw = _random_blocks(fmt, 3 * 2, rng)
    ggml.require_finite_scales(fmt, raw, 3, k, "finite")
    offsets = [208] if fmt is ggml.Q6_K else [0, 2] if fmt.has_min else [0]
    for offset in offsets:
        for word in (0x7C00, 0xFE00):  # +inf, a NaN
            bad = raw.copy()
            bad[5, offset : offset + 2] = np.array([word], "<u2").view(np.uint8)
            with pytest.raises(ValueError, match="row 2"):
                ggml.require_finite_scales(fmt, bad, 3, k, "bad")


def test_terms_reconstruct_the_values():
    rng = np.random.default_rng(7)
    for fmt in ggml.GGML_TYPES.values():
        k = fmt.block_elems * 2
        raw = _random_blocks(fmt, 2 * 2, rng)
        t = ggml.decode_terms(fmt, raw, 2, k)
        assert t.code.dtype == np.int32 and np.abs(t.code).max() <= 127
        value = t.d * t.sub * t.code - (0 if t.min_sub is None else t.dmin * t.min_sub)
        assert np.array_equal(value * fmt.kappa, ggml.decode_values(fmt, raw, 2, k))


def test_store_reads_stored_rows(tmp_path):
    rng = np.random.default_rng(3)
    fmt = ggml.IQ4_NL
    raw = _random_blocks(fmt, 6 * 2, rng)  # 6 rows of 64 elements
    path = tmp_path / "rows.gguf"
    writer = gguf.GGUFWriter(str(path), "test")
    writer.add_tensor("w", raw.reshape(6, 2 * fmt.block_bytes), raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)
    writer.add_tensor("f", np.arange(10, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with GgufStore([path]) as store:
        t = store.tensor("w")
        assert t.shape == (6, 64) and t.ggml_type == fmt.ggml_id and t.row_bytes == 36
        assert np.array_equal(np.asarray(store.raw("w", 2, 5)), raw.reshape(6, 36)[2:5])
        assert np.array_equal(store.plain("f"), np.arange(10, dtype=np.float64))


def test_layout_geometry():
    g = ggml_rows_geometry("ggml_q6_k", (10240, 2560))
    assert g.row_bytes == 10 * 210 and g.payload_bytes == 10240 * 2100
    e = ggml_expert_geometry("ggml_expert_iq4_nl_v1", "ggml_iq2_s", (512, 2560, 640))
    assert e.gate_row_bytes == 10 * 82 and e.down_row_bytes == 20 * 18
    assert e.record_bytes == 1_971_200 and e.record_stride == 1_974_272  # IQ2_S is 2.5625 bits per weight
    q = ggml_expert_geometry("ggml_expert_q2_0_v1", "ggml_iq3_s", (512, 2560, 640))
    assert q.record_bytes == 1_868_800 and q.record_stride == 1_871_872 and q.record_stride % 4096 == 0
    assert encoded_size("ggml_expert_q2_0_v1", "ggml_iq3_s", (512, 2560, 640)) == 512 * 1_871_872
    with pytest.raises(ValueError):
        ggml_rows_geometry("ggml_q4_k", (8, 100))


def test_tiles16_layout_is_a_block_permutation():
    """ggml_tiles16_v1 (ggml_linear.h): tile i at 16 i row_bytes, block b of the tile's row r at
    (16 b + r) block_bytes, the last tile padded with zero blocks; untiling returns every row's bytes
    and the converter writes it chunk by chunk at the rows layout's offsets."""
    from tools.convert.methods import ggml_tiles16_blocks, ggml_tiles16_rows
    from tools.flash_next.verify_gguf_artifact import tiles16_rows

    rng = np.random.default_rng(16)
    fmt = ggml.Q6_K
    g = ggml_tiles16_geometry("ggml_q6_k", (37, 768))
    assert g.row_bytes == 3 * 210 and g.payload_bytes == 48 * 3 * 210
    rows = _random_blocks(fmt, 37 * 3, rng).reshape(37, g.row_bytes)
    tiled = ggml_tiles16_blocks(torch.from_numpy(rows), fmt.block_bytes).numpy()
    assert tiled.size == g.payload_bytes
    for row in (0, 15, 16, 36):
        for b in range(3):
            at = row // 16 * 16 * g.row_bytes + (16 * b + row % 16) * fmt.block_bytes
            assert np.array_equal(tiled[at : at + fmt.block_bytes], rows[row, b * fmt.block_bytes : (b + 1) * fmt.block_bytes])
    assert not tiled.reshape(3, 3, 16, fmt.block_bytes)[2, :, 5:].any()  # rows 37..47: zero blocks
    assert np.array_equal(ggml_tiles16_rows(torch.from_numpy(tiled), 37, g.row_bytes, fmt.block_bytes).numpy(), rows)
    assert np.array_equal(tiles16_rows(tiled.tobytes(), 37, g.row_bytes, fmt.block_bytes), rows)
    # Chunks of whole tiles land where the whole matrix's tiles do.
    chunked = np.concatenate([ggml_tiles16_blocks(torch.from_numpy(rows[i : i + 16]), fmt.block_bytes).numpy()
                              for i in range(0, 37, 16)])
    assert np.array_equal(chunked, tiled)
    bad = tiled.copy()
    bad[-1] = 1
    with pytest.raises(ValueError):
        tiles16_rows(bad.tobytes(), 37, g.row_bytes, fmt.block_bytes)
    with pytest.raises(ValueError):
        ggml_tiles16_geometry("ggml_iq4_nl", (16, 256))


_VISION = {"depth": 1, "hidden_size": 8, "intermediate_size": 12, "num_heads": 2, "patch_size": 2,
           "temporal_patch_size": 2, "spatial_merge_size": 2, "num_position_embeddings": 4}


def _bf16_values(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(torch.bfloat16).float().numpy()


def _write_mmproj(path, tensors: dict, *, blocks: int = 1, projection: int = 16) -> None:
    writer = gguf.GGUFWriter(str(path), "clip")
    writer.add_string("clip.projector_type", "qwen3vl_merger")
    writer.add_uint32("clip.vision.block_count", blocks)
    writer.add_uint32("clip.vision.embedding_length", _VISION["hidden_size"])
    writer.add_uint32("clip.vision.feed_forward_length", _VISION["intermediate_size"])
    writer.add_uint32("clip.vision.attention.head_count", _VISION["num_heads"])
    writer.add_uint32("clip.vision.patch_size", _VISION["patch_size"])
    writer.add_uint32("clip.vision.spatial_merge_size", _VISION["spatial_merge_size"])
    writer.add_uint32("clip.vision.projection_dim", projection)
    writer.add_uint32("clip.vision.image_size", 4)
    writer.add_array("clip.vision.is_deepstack_layers", [False] * blocks)
    for name, values in tensors.items():
        if name.endswith("attn_qkv.weight"):  # BF16, as the mmproj stores matrices
            words = torch.from_numpy(values).to(torch.bfloat16).view(torch.int16).numpy()
            writer.add_tensor(name, words.view(np.uint8), raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(name, values)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _mmproj_tensors(rng: np.random.Generator) -> dict:
    h, i, ps = _VISION["hidden_size"], _VISION["intermediate_size"], _VISION["patch_size"]
    merged = 4 * h
    t = {"v.patch_embd.weight": _bf16_values(rng, h, 3, ps, ps), "v.patch_embd.weight.1": _bf16_values(rng, h, 3, ps, ps),
         "v.patch_embd.bias": _bf16_values(rng, h), "v.position_embd.weight": _bf16_values(rng, 4, h),
         "v.post_ln.weight": _bf16_values(rng, h), "v.post_ln.bias": _bf16_values(rng, h),
         "mm.0.weight": _bf16_values(rng, merged, merged), "mm.0.bias": _bf16_values(rng, merged),
         "mm.2.weight": _bf16_values(rng, 16, merged), "mm.2.bias": _bf16_values(rng, 16)}
    for role, shape in (("ln1.weight", (h,)), ("ln1.bias", (h,)), ("ln2.weight", (h,)), ("ln2.bias", (h,)),
                        ("attn_qkv.weight", (3 * h, h)), ("attn_qkv.bias", (3 * h,)), ("attn_out.weight", (h, h)),
                        ("attn_out.bias", (h,)), ("ffn_up.weight", (i, h)), ("ffn_up.bias", (i,)),
                        ("ffn_down.weight", (h, i)), ("ffn_down.bias", (h,))):
        t["v.blk.0." + role] = _bf16_values(rng, *shape)
    return t


def test_mmproj_vision_tower_is_exact(tmp_path):
    """The mmproj's tensors under the checkpoint's names: the Conv3d patch kernel reassembled from
    its two temporal slices, BF16 and F32 values exact, the metadata checked against vision_config."""

    from tools.convert.qwen4_exp_gguf import _mmproj_checks, vision_source
    from tools.convert.sources.gguf import GGUFSource

    rng = np.random.default_rng(5)
    tensors = _mmproj_tensors(rng)
    path = tmp_path / "mmproj.gguf"
    _write_mmproj(path, tensors)
    with GGUFSource(path) as mmproj:
        _mmproj_checks(mmproj, _VISION, {"hidden_size": 16})
        source = vision_source(mmproj, _VISION)
        qkv = source.read_flat("model.visual.blocks.0.attn.qkv.weight")
        assert torch.equal(qkv, torch.from_numpy(tensors["v.blk.0.attn_qkv.weight"]).reshape(-1))
        assert source.describe("model.visual.blocks.0.attn.qkv.weight").shape == (24, 8)
        kernel = source.read_flat("model.visual.patch_embed.proj.weight").reshape(8, 3, 2, 2, 2)
        assert torch.equal(kernel[:, :, 0], torch.from_numpy(tensors["v.patch_embd.weight"]))
        assert torch.equal(kernel[:, :, 1], torch.from_numpy(tensors["v.patch_embd.weight.1"]))
        assert torch.equal(source.read_flat("model.visual.merger.norm.bias"), torch.from_numpy(tensors["v.post_ln.bias"]))
        with pytest.raises(ValueError, match="projection_dim"):
            _mmproj_checks(mmproj, _VISION, {"hidden_size": 32})
        with pytest.raises(ValueError, match="block_count"):
            _mmproj_checks(mmproj, {**_VISION, "depth": 2}, {"hidden_size": 16})

    tensors["v.blk.0.ln1.bias"][3] = np.float32(1.0 + 2.0 ** -12)  # not a BF16 value
    inexact = tmp_path / "inexact.gguf"
    _write_mmproj(inexact, tensors)
    with GGUFSource(inexact) as mmproj:
        with pytest.raises(ValueError, match="not exact in BF16"):
            vision_source(mmproj, _VISION).read_flat("model.visual.blocks.0.norm1.bias")


def test_expert_bank_import_is_verbatim(tmp_path):
    """Each record is the expert's gate rows, up rows and down rows, as stored, padded to 4 KiB."""

    from tools.convert.qwen4_exp_gguf import GgmlExpertBank, import_ggml_expert_bank
    from tools.convert.methods import MethodInput, PrepareRequest
    from tools.artifact.schema import TensorSpec

    rng = np.random.default_rng(11)
    experts, hidden, inter = 3, 256, 64
    gate = _random_blocks(ggml.IQ3_S, experts * inter * 1, rng)
    up = _random_blocks(ggml.IQ3_S, experts * inter * 1, rng)
    down = _random_blocks(ggml.IQ4_NL, experts * hidden * 2, rng)
    path = tmp_path / "experts.gguf"
    writer = gguf.GGUFWriter(str(path), "test")
    writer.add_tensor("blk.0.ffn_gate_exps.weight", gate.reshape(experts, inter, 110),
                      raw_dtype=gguf.GGMLQuantizationType.IQ3_S)
    writer.add_tensor("blk.0.ffn_up_exps.weight", up.reshape(experts, inter, 110),
                      raw_dtype=gguf.GGMLQuantizationType.IQ3_S)
    writer.add_tensor("blk.0.ffn_down_exps.weight", down.reshape(experts, hidden, 36),
                      raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    class Output:
        def __init__(self, size):
            self.data = bytearray(size)

        def write_bytes(self, offset, data):
            self.data[offset : offset + len(data)] = data

    with GgufStore([path]) as store:
        bank = GgmlExpertBank(store, 0, (experts, hidden, inter))
        assert bank.layout == "ggml_expert_iq4_nl_v1" and bank.gate_up_format == "ggml_iq3_s"
        g = ggml_expert_geometry(bank.layout, bank.gate_up_format, bank.shape)
        spec = TensorSpec("weight/0", bank.shape, bank.gate_up_format, bank.layout)
        request = PrepareRequest(spec, (MethodInput("experts", bank, ()),), {}, {})
        out = Output(g.payload_bytes)
        import_ggml_expert_bank(request).produce(out)
    for e in range(experts):
        record = bytes(out.data[e * g.record_stride : (e + 1) * g.record_stride])
        expected = (gate.reshape(experts, -1)[e].tobytes() + up.reshape(experts, -1)[e].tobytes()
                    + down.reshape(experts, -1)[e].tobytes())
        assert record[: g.record_bytes] == expected
        assert record[g.record_bytes :] == bytes(g.record_stride - g.record_bytes)
