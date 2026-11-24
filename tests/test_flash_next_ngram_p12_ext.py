"""tools/flash_next/ngram_p12_ext.py: the BF16 P12 blockdir packer against the BF16 words it packs
(the oracle: the format is lossless), its base-table and escape gates, the block directory, and the
side stream a block with more than five escapes uses."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tools.flash_next import ngram_p12 as p12
from tools.flash_next import ngram_p12_ext as ext

WIDTH = 160


def _word(sign: int, exponent: int, mantissa: int) -> int:
    return (sign << 15) | (exponent << 7) | mantissa


def _rows(count: int, seed: int) -> np.ndarray:
    """Rows over four bases, with zeros, subnormals and escapes that stay inside the window."""
    rng = np.random.default_rng(seed)
    bases = [0, 100, 120, 200]
    words = np.empty((count, WIDTH), np.uint16)
    for row in range(count):
        base = bases[row % len(bases)]
        words[row] = ((rng.integers(0, 2, WIDTH) << 15) |
                      ((base + rng.integers(0, 15, WIDTH)) << 7) |
                      rng.integers(0, 128, WIDTH))
    words[0] = 0                          # all zeros: base 0
    words[1] = 0x8000                     # negative zero
    words[2] = [_word(j & 1, 0, 1 + j % 127) for j in range(WIDTH)]  # subnormals
    return words.astype(np.uint16)


def _block_of(count: int, base: int, escape_exponent: int, escape_rows: int) -> np.ndarray:
    """`escape_rows` rows of 159 values at `base` and one at `escape_exponent` (one escape a row)."""
    words = np.empty((count, WIDTH), np.uint16)
    words[:] = (base << 7) | 0x55
    for row in range(escape_rows):
        words[row, 0] = _word(0, escape_exponent, 0x11)
    return words


def _write_bf16_volume(path, words: np.ndarray) -> None:
    rows, width = words.shape
    row_bytes = 2 * width
    per_block = p12.BLOCK_BYTES // row_bytes
    blocks = -(-rows // per_block)
    header = bytearray(p12.BLOCK_BYTES)
    p12.NGRAM_HEADER.pack_into(header, 0, p12.NGRAM_MAGIC, p12.NGRAM_VERSION, p12.BLOCK_BYTES, rows,
                               row_bytes, per_block, p12.BLOCK_BYTES, blocks, bytes(range(16)))
    struct.pack_into("<H", header, p12.NGRAM_FORMAT_OFFSET, p12.TAG_BF16)
    body = np.zeros((blocks, p12.BLOCK_BYTES), np.uint8)
    padded = np.zeros((blocks * per_block, row_bytes), np.uint8)
    padded[:rows] = words.astype("<u2").view(np.uint8).reshape(rows, row_bytes)
    body[:, : per_block * row_bytes] = padded.reshape(blocks, per_block * row_bytes)
    path.write_bytes(bytes(header) + body.tobytes())


def _pack(tmp_path, words: np.ndarray):
    source_path = tmp_path / "bf16.ngram"
    _write_bf16_volume(source_path, words)
    source = p12.Volume(source_path)
    out = tmp_path / "blockdir.ngram"
    geometry = ext.write_volume(source, out, ext.packed_volume_id(source.volume_id))
    return source, ext.PackedVolume(out), geometry


def test_round_trip_and_side_stream(tmp_path):
    words = _rows(ext.ROWS_PER_BLOCK * 8 + 3, 5)  # a partial last block
    source, volume, geometry = _pack(tmp_path, words)
    assert geometry == {"rows": len(words), "row_bytes": 256, "rows_per_block": 17, "block_bytes": 4096,
                        "header_bytes": 4096, "blocks": 9, "side_bytes": 0, "side_blocks": 0,
                        "file_bytes": 4096 + 9 * 4096}
    assert ext.verify(source, volume, "all")["row_mismatches"] == 0
    assert np.array_equal(volume.words(0, len(words)), words)


def test_five_escapes_stay_in_the_directory(tmp_path):
    # Block 0: five rows with one escape each, each delta +20 (inside the 6-bit range): five slots.
    words = _block_of(ext.ROWS_PER_BLOCK, 100, 115, 5)
    source, volume, geometry = _pack(tmp_path, words)
    assert geometry["side_blocks"] == 0 and geometry["side_bytes"] == 0
    directory = int.from_bytes(volume.read_block(0)[ext.ROWS_PER_BLOCK * ext.PLANE_BYTES:], "little")
    assert (directory >> (2 * ext.ROWS_PER_BLOCK)) & 7 == 5
    assert ext.verify(source, volume, "all")["row_mismatches"] == 0


def test_more_than_five_escapes_use_the_side_stream(tmp_path):
    # Six rows with one escape each: more than the five slots, so all six go to the side stream.
    words = _block_of(ext.ROWS_PER_BLOCK, 100, 115, 6)
    source, volume, geometry = _pack(tmp_path, words)
    assert geometry["side_blocks"] == 1 and geometry["side_bytes"] == 4096
    assert len(volume.side[0]) == 6
    directory = int.from_bytes(volume.read_block(0)[ext.ROWS_PER_BLOCK * ext.PLANE_BYTES:], "little")
    assert (directory >> (2 * ext.ROWS_PER_BLOCK)) & 7 == ext.SIDE_SENTINEL
    assert ext.verify(source, volume, "all")["row_mismatches"] == 0


def test_a_delta_outside_the_slot_range_uses_the_side_stream(tmp_path):
    # One escape a row at exponent 50 under base 100: delta -50 does not fit six signed bits.
    words = _block_of(ext.ROWS_PER_BLOCK, 100, 50, 1)
    source, volume, geometry = _pack(tmp_path, words)
    assert geometry["side_blocks"] == 1
    assert ext.verify(source, volume, "all")["row_mismatches"] == 0


def test_more_than_four_bases_are_refused(tmp_path):
    words = _rows(ext.ROWS_PER_BLOCK, 6)
    for row in range(5):  # five distinct bases, one a row
        words[row] = ([0, 10, 20, 30, 40][row] << 7) | 0x01
    source_path = tmp_path / "bf16.ngram"
    _write_bf16_volume(source_path, words)
    with pytest.raises(ValueError, match="more than four distinct row-base"):
        ext.write_volume(p12.Volume(source_path), tmp_path / "blockdir.ngram", bytes(16))


def test_a_row_over_the_escape_budget_is_refused(tmp_path):
    words = _block_of(ext.ROWS_PER_BLOCK, 100, 120, 1)
    words[0, 0:5] = [_word(0, 0, j + 1) for j in range(5)]  # five escapes in one row
    source_path = tmp_path / "bf16.ngram"
    _write_bf16_volume(source_path, words)
    with pytest.raises(ValueError, match="needs 5 escapes"):
        ext.write_volume(p12.Volume(source_path), tmp_path / "blockdir.ngram", bytes(16))


def test_census_reports_the_base_and_escape_shape(tmp_path):
    words = _rows(ext.ROWS_PER_BLOCK * 3, 7)
    source_path = tmp_path / "bf16.ngram"
    _write_bf16_volume(source_path, words)
    result = ext.census(p12.Volume(source_path))
    assert result["base"]["global_distinct"] <= 4 and result["base"]["fits_4_entry_table"]
    assert result["escapes"]["max_per_row"] <= ext.MAX_ESCAPES_PER_ROW
    assert result["escapes"]["max_per_block"] <= ext.MAX_BLOCK_ESCAPES
    assert result["escapes"]["blocks_over_5"] == 0


def test_volume_id_follows_the_source_and_a_second_pack_is_identical(tmp_path):
    words = _rows(ext.ROWS_PER_BLOCK * 2, 8)
    source_path = tmp_path / "bf16.ngram"
    _write_bf16_volume(source_path, words)
    source = p12.Volume(source_path)
    volume_id = ext.packed_volume_id(source.volume_id)
    first = tmp_path / "first.ngram"
    ext.write_volume(source, first, volume_id)
    second = tmp_path / "second.ngram"
    ext.write_volume(source, second, volume_id)
    assert first.read_bytes() == second.read_bytes()
    assert ext.PackedVolume(first).volume_id == volume_id
    with pytest.raises(FileExistsError):
        ext.write_volume(source, first, volume_id)
