from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from tools.convert.qwen4_exp import read_ngram_volume_id, write_ngram_volume
from tools.convert.sources.safetensors import SafetensorsSource

ROW_BYTES = 160  # 25 rows per 4 KiB block, as the real table


def _table(tmp_path, seed: int, rows: tuple[int, int]):
    """A two-shard FP8 n-gram table of `rows` rows per shard."""
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for index, count in enumerate(rows):
        words = torch.randint(0, 0x7F, (count, ROW_BYTES), dtype=torch.uint8, generator=generator)
        tensors[f"ngram.{index}.weight"] = words.view(torch.float8_e4m3fn)
    directory = tmp_path / f"checkpoint{seed}"
    directory.mkdir()
    save_file(tensors, directory / "model.safetensors")
    return SafetensorsSource(directory), [(f"ngram.{i}.weight", count) for i, count in enumerate(rows)]


def test_reuse_adopts_the_volume_of_the_same_table(tmp_path):
    store, shards = _table(tmp_path, 1, (1000, 337))
    volume = tmp_path / "table.ngram"
    write_ngram_volume(store, shards, volume, bytes(range(16)))
    assert read_ngram_volume_id(store, shards, volume) == bytes(range(16))


def test_reuse_refuses_another_checkpoints_table_of_the_same_geometry(tmp_path):
    store, shards = _table(tmp_path, 1, (1000, 337))
    other, other_shards = _table(tmp_path, 2, (1000, 337))
    volume = tmp_path / "other.ngram"
    write_ngram_volume(other, other_shards, volume, bytes(16))
    with pytest.raises(ValueError, match="differs from this checkpoint"):
        read_ngram_volume_id(store, shards, volume)


def test_reuse_refuses_a_volume_with_one_changed_sampled_row(tmp_path):
    store, shards = _table(tmp_path, 3, (1000, 337))
    volume = tmp_path / "table.ngram"
    write_ngram_volume(store, shards, volume, bytes(16))
    # The table's last row (always sampled) lies in the second shard: block 53, slot 11 of 25.
    last = 1336
    offset = 4096 + (last // 25) * 4096 + (last % 25) * ROW_BYTES
    data = bytearray(volume.read_bytes())
    data[offset] ^= 0x01
    volume.write_bytes(bytes(data))
    with pytest.raises(ValueError, match=f"row {last} differs"):
        read_ngram_volume_id(store, shards, volume)
