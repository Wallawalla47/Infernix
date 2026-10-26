"""A GGUF's PLE n-gram table as an Infernix volume (tools/convert/qwen4_exp_gguf.GgufNgramTable): IQ4_NL and Q8_0
rows stored verbatim, never straddling a 4 KiB block, the row format recorded in the header."""

from __future__ import annotations

import gguf
import numpy as np
import pytest

from tools.convert.qwen4_exp import NGRAM_FORMAT, NGRAM_FORMAT_OFFSET, read_ngram_volume_format, write_ngram_volume
from tools.convert.qwen4_exp_gguf import NGRAM_TABLE, GgufNgramTable
from tools.convert.sources.gguf_raw import GgufStore

WIDTH = 160  # Qwen3.8-Flash-Next's n-gram head width


def _table_gguf(path, kind: gguf.GGMLQuantizationType, rows: int, seed: int) -> np.ndarray:
    """A table of random block bytes (a volume copies rows without reading them)."""
    elems, size = gguf.GGML_QUANT_SIZES[kind]
    blocks = np.random.default_rng(seed).integers(0, 256, (rows, WIDTH // elems * size), dtype=np.uint8)
    writer = gguf.GGUFWriter(str(path), "qwen4exp")
    if kind == gguf.GGMLQuantizationType.F16:
        writer.add_tensor(NGRAM_TABLE, np.zeros((rows, WIDTH), dtype=np.float16))
    else:
        writer.add_tensor(NGRAM_TABLE, blocks, raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return blocks


@pytest.mark.parametrize("kind, name, tag, row_bytes, per_block", [
    (gguf.GGMLQuantizationType.Q8_0, "q8_0", 5, 170, 24),
    (gguf.GGMLQuantizationType.IQ4_NL, "iq4_nl", 2, 90, 45),
])
def test_a_gguf_table_is_stored_verbatim(tmp_path, kind, name, tag, row_bytes, per_block):
    rows = 2 * per_block + 7
    stored = _table_gguf(tmp_path / f"{name}.gguf", kind, rows, seed=tag)
    with GgufStore([tmp_path / f"{name}.gguf"]) as store:
        table = GgufNgramTable(store, rows)
        assert (table.volume_format, table.row_bytes) == (name, row_bytes)
        volume = tmp_path / f"{name}.ngram"
        geometry = write_ngram_volume(table, volume, bytes(range(16)))
    assert geometry["rows_per_block"] == per_block and geometry["blocks"] == 3
    data = volume.read_bytes()
    assert NGRAM_FORMAT.unpack_from(data, NGRAM_FORMAT_OFFSET)[0] == tag
    assert read_ngram_volume_format(volume, WIDTH) == name
    for r in (0, per_block - 1, per_block, rows - 1):
        at = 4096 + (r // per_block) * 4096 + (r % per_block) * row_bytes
        assert data[at:at + row_bytes] == stored[r].tobytes()
    # Each block's tail past its last whole row is zero.
    assert data[4096 + per_block * row_bytes:8192] == bytes(4096 - per_block * row_bytes)


def test_another_table_type_is_refused(tmp_path):
    _table_gguf(tmp_path / "f16.gguf", gguf.GGMLQuantizationType.F16, 32, seed=9)
    with GgufStore([tmp_path / "f16.gguf"]) as store, pytest.raises(ValueError, match="not IQ4_NL or Q8_0"):
        GgufNgramTable(store, 32)
