from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from tools.convert.qwen4_exp import NgramTable, read_ngram_volume_id, write_ngram_volume
from tools.convert.sources.safetensors import SafetensorsSource

ROW_BYTES = 160  # 25 rows per 4 KiB block, as the real table
PREFIX = "model.layers.0."
NAME = PREFIX + "ple.ple_embedding.ngram_embedding."
CONFIG = {"ple_embed_dim": 2560, "ngram_size": 3, "heads_per_ngram": 8, "split_ngram_parts": 2}


def _save(tmp_path, label: str, tensors: dict) -> SafetensorsSource:
    directory = tmp_path / label
    directory.mkdir()
    save_file(tensors, directory / "model.safetensors")
    return SafetensorsSource(directory)


def _fp8_table(tmp_path, seed: int, rows: tuple[int, int]) -> NgramTable:
    """A two-shard FP8 n-gram table of `rows` rows per shard."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for index, count in enumerate(rows):
        words = torch.randint(0, 0x7F, (count, ROW_BYTES), dtype=torch.uint8, generator=generator)
        tensors[f"{NAME}shard_{index}.weight"] = words.view(torch.float8_e4m3fn)
    tensors[NAME + "weight_scale"] = torch.tensor([2.0**-10], dtype=torch.float32)  # exact in BF16, as checkpoints
    return NgramTable(_save(tmp_path, f"fp8_{seed}", tensors), PREFIX, CONFIG, sum(rows))


def _bf16_values(seed: int, rows: tuple[int, int]) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [(torch.randn(count, ROW_BYTES, generator=generator) * 0.01).to(torch.bfloat16) for count in rows]


def test_reuse_adopts_the_volume_of_the_same_table(tmp_path):
    table  = _fp8_table(tmp_path, 1, (1000, 337))
    volume = tmp_path / "table.ngram"
    write_ngram_volume(table, volume, bytes(range(16)))
    assert read_ngram_volume_id(table, volume) == bytes(range(16))


def test_reuse_refuses_another_checkpoints_table_of_the_same_geometry(tmp_path):
    table = _fp8_table(tmp_path, 1, (1000, 337))
    other = _fp8_table(tmp_path, 2, (1000, 337))
    volume = tmp_path / "other.ngram"
    write_ngram_volume(other, volume, bytes(16))
    with pytest.raises(ValueError, match="differs from this checkpoint"):
        read_ngram_volume_id(table, volume)


def test_reuse_refuses_a_volume_with_one_changed_sampled_row(tmp_path):
    table  = _fp8_table(tmp_path, 3, (1000, 337))
    volume = tmp_path / "table.ngram"
    write_ngram_volume(table, volume, bytes(16))
    # The table's last row (always sampled) lies in the second shard: block 53, slot 11 of 25.
    last = 1336
    offset = 4096 + (last // 25) * 4096 + (last % 25) * ROW_BYTES
    data = bytearray(volume.read_bytes())
    data[offset] ^= 0x01
    volume.write_bytes(bytes(data))
    with pytest.raises(ValueError, match=f"row {last} differs"):
        read_ngram_volume_id(table, volume)


def test_bf16_table_is_quantized_per_tensor_in_bf16(tmp_path):
    shards = _bf16_values(4, (500, 251))
    store  = _save(tmp_path, "bf16", {f"{NAME}shard_{i}.weight": v for i, v in enumerate(shards)})
    table  = NgramTable(store, PREFIX, CONFIG, 751)
    assert table.bf16
    amax = torch.cat(shards).abs().max()
    scale = table.scale()
    assert scale.dtype == torch.bfloat16 and torch.equal(scale, (amax / torch.tensor(448.0, dtype=torch.bfloat16)).reshape(1))
    # code = e4m3_rn_satfinite(bf16(v / s)), the division rounded to BF16 first.
    expected = (shards[1] / scale).float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(table.codes(f"{NAME}shard_1.weight", 0, 251), expected)
    # The rounding differs from an FP32 division on some values; the BF16 rule is the one NVIDIA's
    # FP8 tables of the same BF16 table follow.
    fp32 = (shards[1].float() / scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert not torch.equal(expected, fp32)


def test_bf16_export_reuses_the_volume_of_its_fp8_quantization(tmp_path):
    shards = _bf16_values(5, (600, 400))
    bf16   = NgramTable(_save(tmp_path, "bf16", {f"{NAME}shard_{i}.weight": v for i, v in enumerate(shards)}),
                        PREFIX, CONFIG, 1000)
    # The FP8 table another exporter writes from the same BF16 table with the same rule and scale.
    fp8_tensors = {f"{NAME}shard_{i}.weight": bf16.codes(f"{NAME}shard_{i}.weight", 0, v.shape[0]).view(torch.float8_e4m3fn)
                   for i, v in enumerate(shards)}
    fp8_tensors[NAME + "weight_scale"] = bf16.scale().float()
    fp8 = NgramTable(_save(tmp_path, "fp8", fp8_tensors), PREFIX, CONFIG, 1000)
    assert torch.equal(fp8.scale(), bf16.scale())
    volume = tmp_path / "shared.ngram"
    write_ngram_volume(fp8, volume, bytes(range(16, 32)))
    assert read_ngram_volume_id(bf16, volume) == bytes(range(16, 32))


def _recorded_scale(volume) -> int:
    from tools.convert.qwen4_exp import NGRAM_SCALE, NGRAM_SCALE_OFFSET

    data = volume.read_bytes()[NGRAM_SCALE_OFFSET : NGRAM_SCALE_OFFSET + NGRAM_SCALE.size]
    return NGRAM_SCALE.unpack(data)[0]


def test_an_fp8_volume_records_its_scale_and_others_none(tmp_path):
    # The scale lets the FP8 volume serve any artifact (the engine reads it from the header); BF16
    # rows need none.
    shards, bf16, fp8 = _bf16_and_fp8_tables(tmp_path, 11, (600, 400))
    volume = tmp_path / "fp8.ngram"
    write_ngram_volume(fp8, volume, bytes(16))
    bits = int(fp8.scale().view(torch.int16)[0]) & 0xFFFF
    assert _recorded_scale(volume) == bits and bits != 0
    quantized = tmp_path / "quantized.ngram"  # a BF16 checkpoint's table, quantized with its derived scale
    write_ngram_volume(bf16, quantized, bytes(16))
    assert _recorded_scale(quantized) == bits
    words = _bf16_volume(tmp_path, "words", shards, bytes(16))
    assert _recorded_scale(words) == 0


def test_bf16_volume_keeps_the_words_twelve_rows_per_block(tmp_path):
    # The BF16 volume (tools/flash_next/ngram_bf16.py): rows of 2 x 160 bytes, the table's BF16 words
    # unchanged, 12 per 4 KiB block (the engine reads the format from this row size).
    shards = _bf16_values(6, (30, 7))
    store  = _save(tmp_path, "bf16", {f"{NAME}shard_{i}.weight": v for i, v in enumerate(shards)})
    table  = NgramTable(store, PREFIX, CONFIG, 37, keep_bf16=True)
    assert table.row_bytes == 2 * ROW_BYTES
    volume = tmp_path / "bf16.ngram"
    geometry = write_ngram_volume(table, volume, bytes(range(16)))
    assert geometry["rows_per_block"] == 12 and geometry["blocks"] == 4
    data = volume.read_bytes()
    assert len(data) == geometry["file_bytes"]
    rows = torch.cat(shards).view(torch.uint8).reshape(37, 2 * ROW_BYTES)
    for r in (0, 11, 12, 30, 36):  # block edges and the second shard
        at = 4096 + (r // 12) * 4096 + (r % 12) * 2 * ROW_BYTES
        assert data[at : at + 2 * ROW_BYTES] == rows[r].numpy().tobytes()


def _bf16_and_fp8_tables(tmp_path, seed: int, rows: tuple[int, int]) -> tuple[list[torch.Tensor], NgramTable, NgramTable]:
    """A BF16 table and the FP8 checkpoint of it (its per-tensor quantization and scale, as ModelOpt)."""
    shards = _bf16_values(seed, rows)
    bf16 = NgramTable(_save(tmp_path, f"bf16_{seed}", {f"{NAME}shard_{i}.weight": v for i, v in enumerate(shards)}),
                      PREFIX, CONFIG, sum(rows))
    fp8_tensors = {f"{NAME}shard_{i}.weight": bf16.codes(f"{NAME}shard_{i}.weight", 0, v.shape[0]).view(torch.float8_e4m3fn)
                   for i, v in enumerate(shards)}
    fp8_tensors[NAME + "weight_scale"] = bf16.scale().float()
    return shards, bf16, NgramTable(_save(tmp_path, f"fp8_{seed}", fp8_tensors), PREFIX, CONFIG, sum(rows))


def _bf16_volume(tmp_path, label: str, shards: list[torch.Tensor], volume_id: bytes):
    store = _save(tmp_path, label, {f"{NAME}shard_{i}.weight": v for i, v in enumerate(shards)})
    volume = tmp_path / f"{label}.ngram"
    write_ngram_volume(NgramTable(store, PREFIX, CONFIG, sum(v.shape[0] for v in shards), keep_bf16=True), volume, volume_id)
    return volume


def test_fp8_checkpoint_binds_the_bf16_volume_it_was_quantized_from(tmp_path):
    # Recipe F-A with the BF16 table: the FP8 checkpoint's sampled rows are the BF16 rows' quantization.
    shards, bf16, fp8 = _bf16_and_fp8_tables(tmp_path, 8, (600, 400))
    volume = _bf16_volume(tmp_path, "full", shards, bytes(range(32, 48)))
    assert read_ngram_volume_id(fp8, volume, bf16=True) == bytes(range(32, 48))
    assert read_ngram_volume_id(bf16, volume, bf16=True) == bytes(range(32, 48))  # a BF16 checkpoint: its words
    # Every row, as verify_artifact checks it.
    rows = volume.read_bytes()
    first = torch.frombuffer(bytearray(rows[4096 : 4096 + 12 * 2 * ROW_BYTES]), dtype=torch.uint8).reshape(12, -1)
    assert bool(fp8.bf16_rows_match(f"{NAME}shard_0.weight", 0, 12, first).all())


def test_fp8_checkpoint_refuses_a_bf16_volume_of_another_table(tmp_path):
    shards, _, fp8 = _bf16_and_fp8_tables(tmp_path, 9, (600, 400))
    # The last row (always sampled) moved by far more than one FP8 step.
    changed = [shards[0], shards[1].clone()]
    changed[1][-1] = changed[1][-1] * 3
    volume = _bf16_volume(tmp_path, "changed", changed, bytes(16))
    with pytest.raises(ValueError, match="row 999 differs"):
        read_ngram_volume_id(fp8, volume, bf16=True)
    # A volume of the FP8 rows is no BF16 volume of this table.
    fp8_volume = tmp_path / "fp8.ngram"
    write_ngram_volume(fp8, fp8_volume, bytes(16))
    with pytest.raises(ValueError, match="not an n-gram volume of this checkpoint's table"):
        read_ngram_volume_id(fp8, fp8_volume, bf16=True)


def test_ngram_bf16_verify_compares_every_bf16_row(tmp_path):
    # ngram_bf16 --verify: a BF16 volume against the checkpoint's BF16 table, and a --dequantize
    # volume against the FP8 table's dequantized rows (verify_artifact.verify_ngram, format bf16).
    from tools.convert.qwen4_exp import ngram_geometry
    from tools.flash_next.ngram_bf16 import DequantizedTable
    from tools.flash_next.verify_artifact import verify_ngram

    shards, _, fp8 = _bf16_and_fp8_tables(tmp_path, 10, (600, 400))
    store = _save(tmp_path, "keep", {f"{NAME}shard_{i}.weight": v for i, v in enumerate(shards)})
    for label, table in (("keep", NgramTable(store, PREFIX, CONFIG, 1000, keep_bf16=True)),
                         ("dequantized", DequantizedTable(fp8))):
        volume = tmp_path / f"{label}.ngram"
        write_ngram_volume(table, volume, bytes(range(16)))
        geometry = {**ngram_geometry(table.rows, table.row_bytes), "format": "bf16",
                    "volume_id": bytes(range(16)).hex()}
        assert verify_ngram(table, volume, geometry, "all") == (1000, 0)
        assert verify_ngram(table, volume, geometry, "50") == (50, 0)
        data = bytearray(volume.read_bytes())
        data[4096] ^= 0x01  # row 0
        volume.write_bytes(bytes(data))
        assert verify_ngram(table, volume, geometry, "all") == (1000, 1)


def test_bf16_volume_needs_a_bf16_table(tmp_path):
    with pytest.raises(ValueError, match="needs the checkpoint's BF16 table"):
        generator = torch.Generator().manual_seed(7)
        tensors = {f"{NAME}shard_{i}.weight": torch.randint(0, 0x7F, (10, ROW_BYTES), dtype=torch.uint8,
                                                             generator=generator).view(torch.float8_e4m3fn)
                   for i in range(2)}
        tensors[NAME + "weight_scale"] = torch.tensor([0.001], dtype=torch.float32)
        NgramTable(_save(tmp_path, "fp8", tensors), PREFIX, CONFIG, 20, keep_bf16=True)


def _nvfp4_table(tmp_path, label: str, seed: int, rows: tuple[int, int], multiplier: float = 1.0 / 30_000) -> NgramTable:
    """A two-shard NVFP4 table as FP4-Local stores it: E2M1 codes [rows, 80], E4M3 block scales
    [rows, 10] per shard, one FP32 weight_scale_2 for the table."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for index, count in enumerate(rows):
        tensors[f"{NAME}shard_{index}.weight"] = torch.randint(0, 256, (count, 80), dtype=torch.uint8, generator=generator)
        tensors[f"{NAME}shard_{index}.weight_scale"] = torch.randint(
            0, 0x7F, (count, 10), dtype=torch.uint8, generator=generator).view(torch.float8_e4m3fn)
    tensors[NAME + "weight_scale_2"] = torch.tensor([multiplier], dtype=torch.float32)
    return NgramTable(_save(tmp_path, label, tensors), PREFIX, CONFIG, sum(rows))


def test_nvfp4_volume_keeps_codes_then_scales_and_records_its_format_and_multiplier(tmp_path):
    from tools.convert.qwen4_exp import (
        NGRAM_FORMAT, NGRAM_FORMAT_OFFSET, NGRAM_NVFP4_SCALE, NGRAM_NVFP4_SCALE_OFFSET, read_ngram_volume_format)

    table = _nvfp4_table(tmp_path, "nvfp4", 12, (50, 13))
    assert (table.row_bytes, table.volume_format) == (90, "nvfp4")
    volume = tmp_path / "nvfp4.ngram"
    geometry = write_ngram_volume(table, volume, bytes(range(16)))
    assert geometry["rows_per_block"] == 45 and geometry["blocks"] == 2  # IQ4_NL's row size
    data = volume.read_bytes()
    assert NGRAM_FORMAT.unpack_from(data, NGRAM_FORMAT_OFFSET)[0] == 4
    multiplier = torch.tensor([1.0 / 30_000], dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF
    assert NGRAM_NVFP4_SCALE.unpack_from(data, NGRAM_NVFP4_SCALE_OFFSET)[0] == multiplier
    assert _recorded_scale(volume) == 0  # no FP8 scale
    assert read_ngram_volume_format(volume, ROW_BYTES) == "nvfp4"
    with SafetensorsSource(tmp_path / "nvfp4") as store:
        for r, (shard, local) in ((0, (0, 0)), (44, (0, 44)), (45, (0, 45)), (50, (1, 0)), (62, (1, 12))):
            codes = store.read_flat(f"{NAME}shard_{shard}.weight").reshape(-1, 80)[local]
            scales = store.read_flat(f"{NAME}shard_{shard}.weight_scale").view(torch.uint8).reshape(-1, 10)[local]
            at = 4096 + (r // 45) * 4096 + (r % 45) * 90
            assert data[at : at + 90] == codes.numpy().tobytes() + scales.numpy().tobytes()
    # Another conversion of the same checkpoint adopts the volume; a table of another multiplier does not.
    assert read_ngram_volume_id(table, volume) == bytes(range(16))
    other = _nvfp4_table(tmp_path, "other", 12, (50, 13), multiplier=1.0 / 29_000)
    with pytest.raises(ValueError, match="multiplier"):
        read_ngram_volume_id(other, volume)


def test_every_new_volume_records_its_format_and_old_ones_are_read_by_row_size(tmp_path):
    from tools.convert.qwen4_exp import NGRAM_FORMAT, NGRAM_FORMAT_OFFSET, read_ngram_volume_format

    shards, bf16, fp8 = _bf16_and_fp8_tables(tmp_path, 13, (30, 7))
    fp8_volume = tmp_path / "fp8.ngram"
    write_ngram_volume(fp8, fp8_volume, bytes(16))
    words = _bf16_volume(tmp_path, "words", shards, bytes(16))
    for volume, format, tag in ((fp8_volume, "fp8_e4m3fn", 1), (words, "bf16", 3)):
        data = bytearray(volume.read_bytes())
        assert NGRAM_FORMAT.unpack_from(data, NGRAM_FORMAT_OFFSET)[0] == tag
        assert read_ngram_volume_format(volume, ROW_BYTES) == format
        data[NGRAM_FORMAT_OFFSET : NGRAM_FORMAT_OFFSET + 2] = bytes(2)  # as published before the tag
        volume.write_bytes(bytes(data))
        assert read_ngram_volume_format(volume, ROW_BYTES) == format


def test_an_nvfp4_table_refuses_fp8_scales_and_bf16_volumes(tmp_path):
    table = _nvfp4_table(tmp_path, "nvfp4", 14, (20, 5))
    with pytest.raises(ValueError, match="not an FP8 scale"):
        table.scale()
    with pytest.raises(ValueError, match="needs the checkpoint's BF16 table"):
        NgramTable(SafetensorsSource(tmp_path / "nvfp4"), PREFIX, CONFIG, 25, keep_bf16=True)
