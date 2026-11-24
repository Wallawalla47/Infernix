"""Pack a BF16 n-gram table volume into a lossless BF16 P12 follow-on volume.

    python3 -m tools.flash_next.ngram_p12_ext --source BF16.ngram --format blockdir --census
    python3 -m tools.flash_next.ngram_p12_ext --source BF16.ngram --format blockdir --out OUT.ngram
    python3 -m tools.flash_next.ngram_p12_ext --source BF16.ngram --format blockdir --out OUT.ngram --verify --rows all
    python3 -m tools.flash_next.ngram_p12_ext --source BF16.ngram --format blockdir --out OUT.ngram --stats

SOURCE is an Infernix BF16 n-gram volume (the engine's own layout: a 4 KiB header, then 4 KiB
blocks of whole rows; Qwen3.8-Flash-Next's is row_bytes=320, 12 rows per block, tag 3 = BF16). An
untagged volume (tag 0) whose rows are an even number of bytes is accepted as BF16, as the engine
resolves it by row size.

The P12 codec (tools/flash_next/ngram_p12.py) shrinks each BF16 row of W = 160 values to 240 data
bytes (160 sign-and-mantissa bytes, 80 q-nibble bytes), a base byte and up to four escapes. This
tool replaces the 17 per-row trailers of a 4 KiB block with one 16-byte block directory:

    17 x 240 data bytes              (the P12 planes verbatim)
    16-byte directory               17 x 2-bit base codes into the header's 4-entry base table,
                                    3-bit escape count (7 = all escapes in the side stream),
                                    5 x 18-bit escape slots (12-bit position in the block,
                                    6-bit signed exponent delta from the row's base)

17 x (240 + 16 / 17) = 4096 bytes: the same data as 16 P12 rows, in 17 rows a block. A block whose
escapes do not fit five slots (more than five, or a delta outside -32..31) puts all its escapes in
a file-level side stream, addressed by block id. A row may hold at most four escapes (the P12 row's
trailer bound); the encoder refuses a table with a row past it or with more than four distinct
row-base values table-wide. The decoder here is independent of the engine's.

Only Python 3 and numpy are used (no torch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

from tools.flash_next import ngram_p12 as p12

BLOCK_BYTES = p12.BLOCK_BYTES
HEADER_BYTES = BLOCK_BYTES
MAGIC = p12.NGRAM_MAGIC
VERSION = p12.NGRAM_VERSION
HEADER = p12.NGRAM_HEADER
FORMAT_OFFSET = p12.NGRAM_FORMAT_OFFSET
TAG_BF16_P12_BLOCKDIR = 7

# Fields after the 68-byte v1 fixed header (all little-endian, zero when a format does not use one).
EXT_WIDTH_OFFSET = 68          # u16: W, the values in a row
EXT_SLOT_BYTES_OFFSET = 70     # u16: bytes of the format's on-disk row (240)
EXT_BASES_OFFSET = 72          # 4 x u8: the file's base table
EXT_CODE_LENGTHS_OFFSET = 76   # 16 x u8: a Huffman code-length table (qhuf)
EXT_SIDE_OFFSET = 92           # u64: absolute file offset of the side stream (0 when empty)
EXT_SIDE_BYTES = 100           # u64: its length in bytes (multiple of 4096)
EXT_SIDE_COUNT = 108           # u64: records (blocks for blockdir, rows for qhuf)
EXT_HEADER_BYTES = 116         # bytes of the header this tool reads and writes

WIDTH = 160                    # a blockdir row of 160 values: 17 x 240 + 16 = 4096
ROWS_PER_BLOCK = 17
PLANE_BYTES = WIDTH + WIDTH // 2
MAX_ESCAPES_PER_ROW = p12.MAX_ESCAPES
MAX_BLOCK_ESCAPES = 5
DELTA_MIN, DELTA_MAX = -32, 31
SIDE_SENTINEL = 7


def align_block(value: int) -> int:
    return (value + BLOCK_BYTES - 1) // BLOCK_BYTES * BLOCK_BYTES


class PackedVolume:
    """An n-gram volume this tool wrote: header fields plus blocks and an optional side stream."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            fields = stream.read(EXT_HEADER_BYTES)
        if len(fields) < EXT_HEADER_BYTES:
            raise ValueError(f"{self.path}: short n-gram volume header")
        (magic, version, header_bytes, rows, row_bytes, rows_per_block, block_bytes, blocks,
         volume_id) = HEADER.unpack_from(fields)
        if magic != MAGIC or version != VERSION:
            raise ValueError(f"{self.path}: not an n-gram volume (v1)")
        if header_bytes != HEADER_BYTES or block_bytes != BLOCK_BYTES:
            raise ValueError(f"{self.path}: n-gram blocks and header must be whole 4 KiB pages")
        if rows_per_block != ROWS_PER_BLOCK:
            raise ValueError(f"{self.path}: a blockdir volume holds {ROWS_PER_BLOCK} rows a block, "
                             f"this one {rows_per_block}")
        if blocks != -(-rows // rows_per_block):
            raise ValueError(f"{self.path}: n-gram block count differs from its rows")
        (self.tag,) = struct.unpack_from("<H", fields, FORMAT_OFFSET)
        if self.tag != TAG_BF16_P12_BLOCKDIR:
            raise ValueError(f"{self.path}: not a BF16 P12 blockdir volume (tag {self.tag})")
        (self.width,) = struct.unpack_from("<H", fields, EXT_WIDTH_OFFSET)
        (self.slot_bytes,) = struct.unpack_from("<H", fields, EXT_SLOT_BYTES_OFFSET)
        self.bases = list(fields[EXT_BASES_OFFSET:EXT_BASES_OFFSET + 4])
        (self.side_offset,) = struct.unpack_from("<Q", fields, EXT_SIDE_OFFSET)
        (self.side_bytes,) = struct.unpack_from("<Q", fields, EXT_SIDE_BYTES)
        (self.side_count,) = struct.unpack_from("<Q", fields, EXT_SIDE_COUNT)
        if self.width != WIDTH or self.slot_bytes != PLANE_BYTES or row_bytes != p12_row_bytes():
            raise ValueError(f"{self.path}: a blockdir volume of {WIDTH} values has "
                             f"{p12_row_bytes()}-byte decoded rows and {PLANE_BYTES}-byte planes")
        if any(base > 240 for base in self.bases):
            raise ValueError(f"{self.path}: a base table entry is outside [0, 240]")
        self.header_bytes = header_bytes
        self.rows = rows
        self.rows_per_block = rows_per_block
        self.block_bytes = block_bytes
        self.blocks = blocks
        self.volume_id = volume_id
        expected = header_bytes + blocks * block_bytes + self.side_bytes
        if self.path.stat().st_size != expected:
            raise ValueError(f"{self.path}: n-gram volume size differs from its geometry")
        if self.side_bytes and self.side_offset != header_bytes + blocks * block_bytes:
            raise ValueError(f"{self.path}: the side stream does not follow the blocks")
        self.side: dict[int, list[tuple[int, int]]] = {}
        if self.side_bytes:
            with self.path.open("rb") as stream:
                stream.seek(self.side_offset)
                data = stream.read(self.side_bytes)
            if len(data) != self.side_bytes:
                raise ValueError(f"{self.path}: short side stream")
            at = 0
            for _ in range(self.side_count):
                block, count = struct.unpack_from("<IH", data, at)
                at += 6
                entries = []
                for _ in range(count):
                    position, bits = struct.unpack_from("<HH", data, at)
                    at += 4
                    entries.append((position, bits))
                if block in self.side:
                    raise ValueError(f"{self.path}: block {block} appears twice in the side stream")
                self.side[block] = entries

    def read_block(self, block: int) -> bytes:
        with self.path.open("rb") as stream:
            stream.seek(self.header_bytes + block * self.block_bytes)
            data = stream.read(self.block_bytes)
        if len(data) != self.block_bytes:
            raise ValueError(f"{self.path}: short n-gram volume read")
        return data

    def decode_block(self, block: int) -> np.ndarray:
        """The block's rows as canonical P12 rows uint8 [rows_per_block, 256]."""
        data = self.read_block(block)
        directory = int.from_bytes(data[ROWS_PER_BLOCK * PLANE_BYTES:], "little")

        def bits(at: int, count: int) -> int:
            return (directory >> at) & ((1 << count) - 1)

        count = bits(2 * ROWS_PER_BLOCK, 3)
        if count == 6:
            raise ValueError(f"{self.path}: block {block} records an invalid escape count 6")
        if count == SIDE_SENTINEL and block not in self.side:
            raise ValueError(f"{self.path}: block {block} names the side stream without a record")
        rows = np.frombuffer(data[:ROWS_PER_BLOCK * PLANE_BYTES], np.uint8).reshape(ROWS_PER_BLOCK,
                                                                                   PLANE_BYTES).copy()
        out = np.zeros((ROWS_PER_BLOCK, p12_row_bytes()), np.uint8)
        out[:, :PLANE_BYTES] = rows
        escapes: list[list[tuple[int, int]]] = [[] for _ in range(ROWS_PER_BLOCK)]
        for e in range(count if count != SIDE_SENTINEL else 0):
            slot = bits(2 * ROWS_PER_BLOCK + 3 + 18 * e, 18)
            position, delta = slot & 0xFFF, slot >> 12
            if delta & 32:
                delta -= 64
            row, column = divmod(position, WIDTH)
            exponent = self.bases[bits(2 * row, 2)] + delta
            if not 0 <= exponent <= 255:
                raise ValueError(f"{self.path}: block {block} has an escape outside the BF16 exponent")
            plane = int(out[row, column])
            escapes[row].append((column, ((plane & 0x80) << 8) | (exponent << 7) | (plane & 0x7F)))
        for position, bits_value in self.side.get(block, []):
            row, column = divmod(position, WIDTH)
            escapes[row].append((column, bits_value))
        for row, entries in enumerate(escapes):
            if len(entries) > MAX_ESCAPES_PER_ROW:
                raise ValueError(f"{self.path}: block {block} row {row} holds {len(entries)} escapes; "
                                 f"a P12 row holds {MAX_ESCAPES_PER_ROW}")
            out[row, PLANE_BYTES] = self.bases[bits(2 * row, 2)]
            out[row, PLANE_BYTES + 1] = len(entries)
            for e, (column, value) in enumerate(entries):
                entry = PLANE_BYTES + 2 + 3 * e
                out[row, entry] = column
                out[row, entry + 1] = value & 0xFF
                out[row, entry + 2] = value >> 8
        return out

    def words(self, begin: int, end: int) -> np.ndarray:
        """Rows [begin, end) as little-endian BF16 words uint16 [end - begin, W]."""
        if begin < 0 or end > self.rows or begin > end:
            raise ValueError(f"{self.path}: rows [{begin}, {end}) are outside the table")
        return self.words_from(np.arange(begin, end))

    def words_from(self, indices: np.ndarray) -> np.ndarray:
        """The given rows as little-endian BF16 words uint16 [len(indices), W]."""
        out = np.empty((len(indices), WIDTH), np.uint16)
        cached_block, cached_rows = None, None
        for i, row in enumerate(indices):
            block = int(row) // ROWS_PER_BLOCK
            if block != cached_block:
                cached_rows = self.decode_block(block)
                cached_block = block
            slot = int(row) % ROWS_PER_BLOCK
            out[i] = p12.decode_rows(cached_rows[slot:slot + 1], WIDTH)[0]
        return out


def p12_row_bytes() -> int:
    return WIDTH + WIDTH // 2 + 16


def analyze_block(words: np.ndarray, row_offset: int) -> dict:
    """The bases, escapes and their placement of one output block (rows may be fewer than 17)."""
    rows = words.shape[0]
    counts = p12.window_counts(words)
    base, in_window = p12.best_base(counts, 0)
    exponent = ((words >> 7) & 0xFF).astype(np.int64)
    offset = exponent - base[:, None].astype(np.int64)
    escaped = (offset < 0) | (offset > 14)
    escapes = int(escaped.sum())
    row_escapes = escaped.sum(axis=1)
    if row_escapes.size and int(row_escapes.max()) > MAX_ESCAPES_PER_ROW:
        bad = row_offset + int(row_escapes.argmax())
        raise ValueError(f"row {bad} needs {int(row_escapes.max())} escapes; a BF16 P12 row holds "
                         f"{MAX_ESCAPES_PER_ROW}")
    distinct = sorted({int(value) for value in base})
    entries = []
    deltas_ok = True
    if escapes <= MAX_BLOCK_ESCAPES:
        for row in range(rows):
            for column in np.nonzero(escaped[row])[0]:
                delta = int(offset[row, column])
                if delta < DELTA_MIN or delta > DELTA_MAX:
                    deltas_ok = False
                entries.append((int(row) * WIDTH + int(column), delta))
    side_entries = []
    if escapes > MAX_BLOCK_ESCAPES or not deltas_ok:
        for row in range(rows):
            for column in np.nonzero(escaped[row])[0]:
                bits_value = int(words[row, column])
                side_entries.append((int(row) * WIDTH + int(column), bits_value))
    return {"words": words, "base": base, "offset": offset, "escaped": escaped, "escapes": escapes,
            "row_escapes": row_escapes, "distinct": distinct,
            "slots": [] if side_entries else entries, "side": side_entries}


def pack_block(analysis: dict, bases: list[int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """One 4 KiB block and its side-stream entries (empty unless the slots overflow)."""
    words, base, offset, escaped = analysis["words"], analysis["base"], analysis["offset"], analysis["escaped"]
    rows = words.shape[0]
    nibble = np.where(escaped, 15, offset).astype(np.uint8)
    codes = (((words >> 8) & 0x80) | (words & 0x7F)).astype(np.uint8)
    plane = np.zeros((ROWS_PER_BLOCK, PLANE_BYTES), np.uint8)
    plane[:rows, :WIDTH] = codes
    plane[:rows, WIDTH:PLANE_BYTES] = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).astype(np.uint8)
    directory = 0
    for row in range(ROWS_PER_BLOCK):
        value = bases.index(int(base[row])) if row < rows else 0
        directory |= value << (2 * row)
    side = analysis["side"]
    count = len(side) if side else len(analysis["slots"])
    directory |= (SIDE_SENTINEL if side else count) << (2 * ROWS_PER_BLOCK)
    if not side:
        for e, (position, delta) in enumerate(analysis["slots"]):
            if delta < 0:
                delta += 64
            directory |= (position | (delta << 12)) << (2 * ROWS_PER_BLOCK + 3 + 18 * e)
    block = np.zeros(BLOCK_BYTES, np.uint8)
    block[:ROWS_PER_BLOCK * PLANE_BYTES] = plane.reshape(-1)
    block[ROWS_PER_BLOCK * PLANE_BYTES:] = np.frombuffer(directory.to_bytes(16, "little"), np.uint8)
    return block, side


def census(source: p12.Volume, chunk_blocks: int = 256) -> dict:
    if source.row_bytes // 2 != WIDTH:
        raise ValueError(f"{source.path}: a blockdir row holds {WIDTH} values; this source's rows hold "
                         f"{source.row_bytes // 2}")
    total_blocks = -(-source.rows // ROWS_PER_BLOCK)
    distinct_histogram: dict[int, int] = {}
    escape_histogram: dict[int, int] = {}
    global_bases: set[int] = set()
    row_escape_max = 0
    side_blocks = 0
    delta_blocks = 0
    blocks_over_four = 0
    escapes_total = 0
    rows = source.rows
    begin = 0
    while begin < rows:
        end = min(rows, begin + chunk_blocks * ROWS_PER_BLOCK)
        words = source.words(begin, end)
        for first in range(0, end - begin, ROWS_PER_BLOCK):
            block_rows = words[first:first + ROWS_PER_BLOCK]
            analysis = analyze_block(block_rows, begin + first)
            global_bases.update(analysis["distinct"])
            distinct = len(analysis["distinct"])
            distinct_histogram[distinct] = distinct_histogram.get(distinct, 0) + 1
            if distinct > 4:
                blocks_over_four += 1
            escape_histogram[analysis["escapes"]] = escape_histogram.get(analysis["escapes"], 0) + 1
            escapes_total += analysis["escapes"]
            if analysis["row_escapes"].size:
                row_escape_max = max(row_escape_max, int(analysis["row_escapes"].max()))
            if analysis["side"]:
                side_blocks += 1
                if analysis["escapes"] <= MAX_BLOCK_ESCAPES:
                    delta_blocks += 1
        begin = end
    return {
        "source": str(source.path),
        "rows": rows,
        "width": WIDTH,
        "blocks": total_blocks,
        "base": {
            "global_distinct": len(global_bases),
            "values": sorted(global_bases),
            "per_block_distinct_histogram": {str(k): v for k, v in sorted(distinct_histogram.items())},
            "blocks_over_4": blocks_over_four,
            "fits_4_entry_table": len(global_bases) <= 4 and blocks_over_four == 0,
        },
        "escapes": {
            "per_block_histogram": {str(k): v for k, v in sorted(escape_histogram.items())},
            "mean_per_block": escapes_total / total_blocks if total_blocks else 0.0,
            "max_per_block": max(escape_histogram) if escape_histogram else 0,
            "max_per_row": row_escape_max,
            "blocks_over_5": sum(v for k, v in escape_histogram.items() if k > MAX_BLOCK_ESCAPES),
            "p_over_5": sum(v for k, v in escape_histogram.items() if k > MAX_BLOCK_ESCAPES) / total_blocks
            if total_blocks else 0.0,
        },
        "side_stream": {
            "blocks": side_blocks,
            "delta_out_of_range_blocks": delta_blocks,
            "p_side": side_blocks / total_blocks if total_blocks else 0.0,
        },
    }


def build_bases(source: p12.Volume, chunk_blocks: int = 256) -> list[int]:
    """The file's 4-entry base table, or a refusal: a blockdir volume needs all of a table's bases."""
    bases: set[int] = set()
    rows = source.rows
    begin = 0
    while begin < rows:
        end = min(rows, begin + chunk_blocks * ROWS_PER_BLOCK)
        words = source.words(begin, end)
        for first in range(0, end - begin, ROWS_PER_BLOCK):
            analysis = analyze_block(words[first:first + ROWS_PER_BLOCK], begin + first)
            bases.update(analysis["distinct"])
            if len(bases) > 4:
                raise ValueError(f"{source.path}: more than four distinct row-base values; a blockdir "
                                 f"volume's base table holds four")
        begin = end
    return sorted(bases) + [0] * (4 - len(bases))


def write_volume(source: p12.Volume, out: Path, volume_id: bytes, progress=None) -> dict:
    if len(volume_id) != 16:
        raise ValueError("the n-gram volume id is 16 bytes")
    if source.row_bytes // 2 != WIDTH:
        raise ValueError(f"{source.path}: a blockdir row holds {WIDTH} values")
    rows = source.rows
    blocks = -(-rows // ROWS_PER_BLOCK)
    bases = build_bases(source)
    side_records: list[tuple[int, list[tuple[int, int]]]] = []
    # The side stream's size is needed in the header: count its records first.
    begin = 0
    while begin < rows:
        end = min(rows, begin + 256 * ROWS_PER_BLOCK)
        words = source.words(begin, end)
        for first in range(0, end - begin, ROWS_PER_BLOCK):
            analysis = analyze_block(words[first:first + ROWS_PER_BLOCK], begin + first)
            if analysis["side"]:
                side_records.append((begin // ROWS_PER_BLOCK + first // ROWS_PER_BLOCK, analysis["side"]))
        begin = end
    side_length = align_block(sum(6 + 4 * len(entries) for _, entries in side_records))
    side_offset = HEADER_BYTES + blocks * BLOCK_BYTES
    header = bytearray(HEADER_BYTES)
    HEADER.pack_into(header, 0, MAGIC, VERSION, HEADER_BYTES, rows, p12_row_bytes(), ROWS_PER_BLOCK,
                     BLOCK_BYTES, blocks, volume_id)
    struct.pack_into("<H", header, p12.NGRAM_SCALE_OFFSET, 0)
    struct.pack_into("<H", header, FORMAT_OFFSET, TAG_BF16_P12_BLOCKDIR)
    struct.pack_into("<I", header, p12.NGRAM_NVFP4_SCALE_OFFSET, 0)
    struct.pack_into("<H", header, EXT_WIDTH_OFFSET, WIDTH)
    struct.pack_into("<H", header, EXT_SLOT_BYTES_OFFSET, PLANE_BYTES)
    header[EXT_BASES_OFFSET:EXT_BASES_OFFSET + 4] = bytes(bases)
    struct.pack_into("<Q", header, EXT_SIDE_OFFSET, side_offset if side_length else 0)
    struct.pack_into("<Q", header, EXT_SIDE_BYTES, side_length)
    struct.pack_into("<Q", header, EXT_SIDE_COUNT, len(side_records))
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"n-gram volume already exists: {out}")
    temporary = out.with_name(out.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(header)
            begin = 0
            while begin < rows:
                end = min(rows, begin + 256 * ROWS_PER_BLOCK)
                words = source.words(begin, end)
                for first in range(0, end - begin, ROWS_PER_BLOCK):
                    analysis = analyze_block(words[first:first + ROWS_PER_BLOCK], begin + first)
                    block, _ = pack_block(analysis, bases)
                    stream.write(block.tobytes())
                begin = end
                if progress is not None:
                    progress(end, rows)
            written = 0
            for block, entries in side_records:
                record = struct.pack("<IH", block, len(entries))
                record += b"".join(struct.pack("<HH", position, bits_value) for position, bits_value in entries)
                stream.write(record)
                written += len(record)
            stream.write(bytes(side_length - written))
        if temporary.stat().st_size != side_offset + side_length:
            raise ValueError("n-gram volume size differs from its geometry")
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": rows, "row_bytes": p12_row_bytes(), "rows_per_block": ROWS_PER_BLOCK,
            "block_bytes": BLOCK_BYTES, "header_bytes": HEADER_BYTES, "blocks": blocks,
            "side_bytes": side_length, "side_blocks": len(side_records),
            "file_bytes": side_offset + side_length}


def verify(source: p12.Volume, volume: PackedVolume, rows: str, chunk_rows: int = 1 << 15) -> dict:
    if source.row_bytes // 2 != WIDTH:
        raise ValueError(f"{source.path}: a blockdir row holds {WIDTH} values")
    if volume.rows != source.rows:
        raise ValueError(f"{volume.path}: {volume.rows} rows differ from the source's {source.rows}")
    checked = 0
    bad = 0
    if rows == "all":
        for begin in range(0, source.rows, chunk_rows):
            end = min(source.rows, begin + chunk_rows)
            want = source.words(begin, end)
            got = volume.words(begin, end)
            bad += int((got != want).any(axis=1).sum())
            checked += end - begin
    else:
        count = min(int(rows), source.rows)
        rng = np.random.default_rng(7)
        indices = np.sort(rng.choice(source.rows, size=count, replace=False))
        want = np.ascontiguousarray(source.read_many(indices)).view("<u2").reshape(len(indices), WIDTH)
        got = volume.words_from(indices)
        bad += int((got != want).any(axis=1).sum())
        checked += len(indices)
    return {"source": str(source.path), "volume": str(volume.path), "rows_checked": checked,
            "row_mismatches": bad}


def stats(source: p12.Volume, volume: PackedVolume) -> dict:
    histogram: dict[int, int] = {}
    distinct_histogram: dict[int, int] = {}
    for block in range(volume.blocks):
        data = volume.read_block(block)
        directory = int.from_bytes(data[ROWS_PER_BLOCK * PLANE_BYTES:], "little")
        count = (directory >> (2 * ROWS_PER_BLOCK)) & 7
        count = len(volume.side.get(block, [])) if count == SIDE_SENTINEL else count
        histogram[count] = histogram.get(count, 0) + 1
        codes = {(directory >> (2 * row)) & 3 for row in range(ROWS_PER_BLOCK)}
        distinct_histogram[len(codes)] = distinct_histogram.get(len(codes), 0) + 1
    return {
        "volume": str(volume.path),
        "format": "bf16_p12_blockdir",
        "row_bytes": p12_row_bytes(),
        "width": WIDTH,
        "rows": volume.rows,
        "file_bytes": volume.path.stat().st_size,
        "volume_id": volume.volume_id.hex(),
        "bases": volume.bases,
        "side_blocks": volume.side_count,
        "side_bytes": volume.side_bytes,
        "escape_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "per_block_distinct_histogram": {str(k): v for k, v in sorted(distinct_histogram.items())},
        "source_rows": source.rows,
        "source_row_bytes": source.row_bytes,
    }


def packed_volume_id(source_id: bytes) -> bytes:
    return hashlib.sha256(b"infernix bf16_p12_blockdir v1" + bytes(source_id)).digest()[:16]


def require_bf16_source(source: p12.Volume, action: str) -> None:
    p12.require_bf16_source(source, action)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="the BF16 n-gram volume to read")
    parser.add_argument("--format", choices=["blockdir"], required=True, help="the row format to pack")
    parser.add_argument("--out", type=Path,
                        help="the packed volume to write (--out) or to read (--verify, --stats)")
    parser.add_argument("--census", action="store_true", help="per-block base and escape statistics")
    parser.add_argument("--verify", action="store_true", help="compare the packed volume with the source row by row")
    parser.add_argument("--stats", action="store_true", help="the packed volume's directories and geometry")
    parser.add_argument("--rows", default="200000", help="rows --verify checks: 'all' or a sample size")
    args = parser.parse_args()
    writing = args.out is not None and not (args.verify or args.stats)
    if sum((args.census, args.verify, args.stats, writing)) != 1:
        parser.error("give exactly one of --census, --out, --verify or --stats")
    if (args.verify or args.stats) and args.out is None:
        parser.error("--verify and --stats read the packed volume given by --out")

    source = p12.Volume(args.source)
    if args.census:
        require_bf16_source(source, "--census")
        result = census(source)
        print(json.dumps(result, indent=2))
        if not result["base"]["fits_4_entry_table"]:
            sys.exit(1)
        return
    if writing:
        require_bf16_source(source, "--out")
        volume_id = packed_volume_id(source.volume_id)
        printed = [0]

        def progress(done: int, total: int) -> None:
            if done == total or done - printed[0] >= max(1, total // 100):
                printed[0] = done
                print(f"{done}/{total} rows", file=sys.stderr, flush=True)

        geometry = write_volume(source, args.out, volume_id, progress=progress)
        print(json.dumps({"volume": str(args.out), "format": "bf16_p12_blockdir",
                          "volume_id": volume_id.hex(), **geometry}, indent=2))
        return
    volume = PackedVolume(args.out)
    if args.verify:
        result = verify(source, volume, args.rows)
        print(json.dumps(result, indent=2))
        if result["row_mismatches"]:
            sys.exit(1)
        return
    print(json.dumps(stats(source, volume), indent=2))


if __name__ == "__main__":
    main()
