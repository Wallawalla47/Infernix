"""tools/flash_next/ngram_p12.py: the BF16 P12 packer against the BF16 words it packs (the oracle: the
format is lossless), its escape budget, and the volume container the engine reads."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tools.flash_next import ngram_p12 as p12

WIDTH = 160  # Qwen3.8-Flash-Next's row: 160 values, 320 BF16 bytes, 256 P12 bytes


def _bf16(sign: int, exponent: int, mantissa: int) -> int:
    return (sign << 15) | (exponent << 7) | mantissa


def _words(rows: int, seed: int) -> np.ndarray:
    """Rows of BF16 words: random rows in a window, zeros, -0, subnormals, and rows with escapes."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 241, size=(rows, 1))
    exponent = base + rng.integers(0, 15, size=(rows, WIDTH))
    words = (rng.integers(0, 2, size=(rows, WIDTH)) << 15) | (exponent << 7) | rng.integers(0, 128, size=(rows, WIDTH))
    words = words.astype(np.uint16)
    words[0] = 0  # all zeros
    words[1] = 0x8000  # negative zero
    words[2] = [_bf16(j & 1, 0, 1 + j % 127) for j in range(WIDTH)]  # subnormals, in window at base 0
    words[3] = [_bf16(0, 120 + j % 4, j % 128) for j in range(WIDTH)]  # a narrow window
    words[3, [0, 7, 80, 159]] = [_bf16(1, 0, 0), _bf16(0, 255, 64), _bf16(0, 1, 5), _bf16(1, 250, 1)]  # 4 escapes
    words[4] = [_bf16(0, 60, 3) for _ in range(WIDTH)]
    words[4, 100] = _bf16(0, 255, 0)  # +inf, an escape
    return words


def _write_bf16_volume(path, words: np.ndarray, *, tag: int, volume_id: bytes) -> None:
    rows, width = words.shape
    row_bytes = 2 * width
    per_block = p12.BLOCK_BYTES // row_bytes
    blocks = -(-rows // per_block)
    header = bytearray(p12.BLOCK_BYTES)
    p12.NGRAM_HEADER.pack_into(header, 0, p12.NGRAM_MAGIC, p12.NGRAM_VERSION, p12.BLOCK_BYTES, rows, row_bytes,
                               per_block, p12.BLOCK_BYTES, blocks, volume_id)
    struct.pack_into("<H", header, p12.NGRAM_FORMAT_OFFSET, tag)
    body = np.zeros((blocks, p12.BLOCK_BYTES), np.uint8)
    padded = np.zeros((blocks * per_block, row_bytes), np.uint8)
    padded[:rows] = words.astype("<u2").view(np.uint8).reshape(rows, row_bytes)
    body[:, : per_block * row_bytes] = padded.reshape(blocks, per_block * row_bytes)
    path.write_bytes(bytes(header) + body.tobytes())


def test_rows_decode_to_the_packed_words():
    words = _words(500, 1)
    packed = p12.pack_rows(words, WIDTH)
    assert packed.shape == (500, WIDTH + WIDTH // 2 + 16)
    assert np.array_equal(p12.decode_rows(packed, WIDTH), words)
    assert packed[3, WIDTH + WIDTH // 2 + 1] == 4  # the fourth row uses the whole escape budget


def test_a_row_past_the_escape_budget_is_refused():
    words = _words(5, 2)
    words[3, [0, 7, 80, 159, 33]] = [_bf16(0, 0, 1), _bf16(0, 255, 64), _bf16(0, 1, 5), _bf16(1, 250, 1),
                                     _bf16(0, 200, 0)]
    with pytest.raises(ValueError, match="needs 5 escapes"):
        p12.pack_rows(words, WIDTH, row_offset=10)


@pytest.mark.parametrize("tag", [3, 0])  # tagged, and untagged like the published BF16 volume
def test_volume_container_verify_and_id(tmp_path, tag):
    words = _words(101, 3)  # 101 rows: a partial last block in both layouts
    source_path = tmp_path / "bf16.ngram"
    source_id = bytes(range(16))
    _write_bf16_volume(source_path, words, tag=tag, volume_id=source_id)
    source = p12.Volume(source_path)
    out = tmp_path / "p12.ngram"
    volume_id = p12.packed_volume_id(source.volume_id)
    geometry = p12.write_volume(source, WIDTH, out, volume_id)
    assert geometry == {"rows": 101, "row_bytes": 256, "rows_per_block": 16, "block_bytes": 4096,
                        "header_bytes": 4096, "blocks": 7, "file_bytes": 4096 * 8}
    volume = p12.Volume(out)
    assert volume.tag == p12.TAG_BF16_P12 and volume.volume_id == volume_id and volume.rows == 101
    assert p12.verify(source, volume, "all") == {"source": str(source_path), "volume": str(out),
                                                 "rows_checked": 101, "row_mismatches": 0}
    assert np.array_equal(p12.decode_rows(volume.read_rows(0, 101), WIDTH), words)
    # The id follows from the source's: a second pack of the same source is the same file.
    assert volume_id == p12.packed_volume_id(source_id) and volume_id != source_id
    again = tmp_path / "again.ngram"
    p12.write_volume(source, WIDTH, again, p12.packed_volume_id(p12.Volume(source_path).volume_id))
    assert again.read_bytes() == out.read_bytes()
    with pytest.raises(FileExistsError):
        p12.write_volume(source, WIDTH, out, volume_id)
