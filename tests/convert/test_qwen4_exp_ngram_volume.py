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
    tensors[NAME + "weight_scale"] = torch.tensor([0.001], dtype=torch.float32)
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
