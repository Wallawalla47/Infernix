"""Pack a BF16 n-gram table volume into a lossless 12-bit (BF16 P12) volume.

    python3 -m tools.flash_next.ngram_p12 --source BF16.ngram --census
    python3 -m tools.flash_next.ngram_p12 --source BF16.ngram --out P12.ngram
    python3 -m tools.flash_next.ngram_p12 --source BF16.ngram --out P12.ngram --verify --rows all
    python3 -m tools.flash_next.ngram_p12 --source BF16.ngram --out P12.ngram --stats

SOURCE is an Infernix BF16 n-gram volume (the engine's own layout: a 4 KiB header, then 4 KiB
blocks of whole rows; Qwen3.8-Flash-Next's is row_bytes=320, 12 rows per block, tag 3 = BF16). An
untagged volume (tag 0) whose rows are an even number of bytes is accepted as BF16, as the engine
resolves it by row size; the published Qwen3.8-Flash-Next BF16 volume is untagged. The header's rows
and row_bytes are read, never assumed; a BF16 source must hold 12 rows per block.

Each BF16 row of W values becomes W + W / 2 + 16 bytes:

    0     .. W-1        W code bytes: sign << 7 | mantissa7 of value j at byte j
    W     .. W+W/2-1    W/2 nibbles: value 2j in the low nibble of nibble j/2, 2j+1 in the high
    W+W/2               the base b (the exponent window starts at b)
    W+W/2+1             the escape count K (0..4)
    W+W/2+2 .. +13      K 3-byte escapes: column u8, BF16 bits u16 little-endian
    W+W/2+14 .. +15     zero

Decode of value j: q = exponent - b; q < 15 reconstructs ((code & 0x80) << 8) | ((b + q) << 7) |
(code & 0x7F); q == 15 reads the escape whose column is j. The base is the b in [0, 240] whose
15-exponent window [b, b+14] holds the most of the row's values, ties to the lowest b, so a zero
(exponent 0) is in window exactly at b == 0. A row with more than 4 escapes cannot be packed.

The packer writes the same v1 container the engine reads (header, tag 6, 4 KiB blocks, zero tail),
and --verify decodes each row with an independent reader and compares it bit for bit with the source
volume. Only Python 3 and numpy are used (no torch, no checkpoint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

NGRAM_MAGIC = b"NINFERNG"
NGRAM_VERSION = 1
BLOCK_BYTES = 4096
NGRAM_HEADER = struct.Struct("<8sIIQIIIQ16s")  # magic, version, header bytes, rows, row bytes,
#                                               rows per block, block bytes, blocks, volume id
NGRAM_SCALE_OFFSET = NGRAM_HEADER.size          # per-tensor FP8 scale (BF16 bits); 0 here
NGRAM_FORMAT_OFFSET = NGRAM_SCALE_OFFSET + 2    # row format tag
NGRAM_NVFP4_SCALE_OFFSET = NGRAM_FORMAT_OFFSET + 2  # an NVFP4 table's FP32 multiplier; 0 here
TAG_BF16 = 3
TAG_BF16_P12 = 6
TAG_NAMES = {0: "untagged", 1: "fp8_e4m3fn", 2: "iq4_nl", 3: "bf16", 4: "nvfp4", 5: "q8_0",
             6: "bf16_p12"}
MAX_ESCAPES = 4


class Volume:
    """An Infernix n-gram volume's header and its rows (never the whole file in memory)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            fields = stream.read(NGRAM_NVFP4_SCALE_OFFSET + 4)
        if len(fields) < NGRAM_NVFP4_SCALE_OFFSET + 4:
            raise ValueError(f"{self.path}: short n-gram volume header")
        (magic, version, header_bytes, rows, row_bytes, rows_per_block, block_bytes, blocks,
         volume_id) = NGRAM_HEADER.unpack_from(fields)
        if magic != NGRAM_MAGIC or version != NGRAM_VERSION:
            raise ValueError(f"{self.path}: not an n-gram volume (v1)")
        if header_bytes != BLOCK_BYTES or block_bytes != BLOCK_BYTES:
            raise ValueError(f"{self.path}: n-gram blocks and header must be whole 4 KiB pages")
        if row_bytes == 0 or rows_per_block != block_bytes // row_bytes:
            raise ValueError(f"{self.path}: n-gram rows of {row_bytes} bytes do not fill "
                             f"{rows_per_block} per block")
        if blocks != -(-rows // rows_per_block):
            raise ValueError(f"{self.path}: n-gram block count differs from its rows")
        if self.path.stat().st_size != header_bytes + blocks * block_bytes:
            raise ValueError(f"{self.path}: n-gram volume size differs from its geometry")
        self.header_bytes = header_bytes
        self.rows = rows
        self.row_bytes = row_bytes
        self.rows_per_block = rows_per_block
        self.block_bytes = block_bytes
        self.blocks = blocks
        self.volume_id = volume_id
        (self.fp8_scale,) = struct.unpack_from("<H", fields, NGRAM_SCALE_OFFSET)
        (self.tag,) = struct.unpack_from("<H", fields, NGRAM_FORMAT_OFFSET)
        (self.nvfp4_scale,) = struct.unpack_from("<I", fields, NGRAM_NVFP4_SCALE_OFFSET)

    @property
    def format(self) -> str:
        return TAG_NAMES.get(self.tag, f"tag {self.tag}")

    def read_rows(self, begin: int, end: int) -> np.ndarray:
        """Rows [begin, end) as uint8 [end - begin, row_bytes]; a block's zero tail is skipped."""
        if begin < 0 or end > self.rows or begin > end:
            raise ValueError(f"{self.path}: rows [{begin}, {end}) are outside the table")
        if begin == end:
            return np.empty((0, self.row_bytes), np.uint8)
        first = begin // self.rows_per_block
        last = (end - 1) // self.rows_per_block
        length = (last - first + 1) * self.block_bytes
        with self.path.open("rb") as stream:
            stream.seek(self.header_bytes + first * self.block_bytes)
            data = stream.read(length)
        if len(data) != length:
            raise ValueError(f"{self.path}: short n-gram volume read")
        blocks = np.frombuffer(data, np.uint8).reshape(-1, self.block_bytes)
        rows = blocks[:, : self.rows_per_block * self.row_bytes].reshape(-1, self.row_bytes)
        lo = begin - first * self.rows_per_block
        return np.array(rows[lo:lo + (end - begin)], np.uint8)

    def read_many(self, indices: np.ndarray) -> np.ndarray:
        """The given rows, one seek each, with one open file."""
        out = np.empty((len(indices), self.row_bytes), np.uint8)
        with self.path.open("rb") as stream:
            for i, row in enumerate(indices):
                block, offset = divmod(int(row), self.rows_per_block)
                stream.seek(self.header_bytes + block * self.block_bytes + offset * self.row_bytes)
                data = stream.read(self.row_bytes)
                if len(data) != self.row_bytes:
                    raise ValueError(f"{self.path}: short n-gram volume read")
                out[i] = np.frombuffer(data, np.uint8)
        return out

    def words(self, begin: int, end: int) -> np.ndarray:
        """Rows [begin, end) as little-endian BF16 words uint16 [end - begin, row_bytes / 2]."""
        if self.row_bytes % 2:
            raise ValueError(f"{self.path}: BF16 rows must be an even number of bytes")
        rows = np.ascontiguousarray(self.read_rows(begin, end))
        return rows.view("<u2").reshape(end - begin, self.row_bytes // 2)


def window_counts(words: np.ndarray) -> np.ndarray:
    """Per row, the count of exponents in [b, b + 14] for every b in [0, 240]: int64 [n, 241]."""
    rows = words.shape[0]
    exponent = ((words >> 7) & 0xFF).astype(np.int64)
    index = (np.arange(rows, dtype=np.int64)[:, None] * 256 + exponent).ravel()
    histogram = np.bincount(index, minlength=rows * 256).reshape(rows, 256)
    cumulative = np.concatenate((np.zeros((rows, 1), np.int64), np.cumsum(histogram, axis=1)), axis=1)
    return cumulative[:, 15:256] - cumulative[:, 0:241]


def best_base(counts: np.ndarray, first: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The base in [first, 240] whose window holds the most values, ties to the lowest base."""
    candidates = counts[:, first:]
    best = candidates.max(axis=1)
    base = first + (candidates == best[:, None]).argmax(axis=1)  # argmax: the first (lowest) maximum
    return base.astype(np.uint8), best.astype(np.int64)


def pack_rows(words: np.ndarray, width: int, row_offset: int = 0) -> np.ndarray:
    """Pack BF16 words uint16 [n, width] into volume rows uint8 [n, width + width / 2 + 16]."""
    if words.shape[1] != width:
        raise ValueError(f"expected {width} values per row, got {words.shape[1]}")
    rows = words.shape[0]
    row_bytes = width + width // 2 + 16
    base, in_window = best_base(window_counts(words), 0)
    escapes = width - in_window
    if escapes.size and int(escapes.max()) > MAX_ESCAPES:
        bad = row_offset + int(escapes.argmax())
        raise ValueError(f"row {bad} needs {int(escapes.max())} escapes; a BF16 P12 row holds {MAX_ESCAPES}")
    exponent = ((words >> 7) & 0xFF).astype(np.int32)
    offset = exponent - base[:, None].astype(np.int32)
    escaped = (offset < 0) | (offset > 14)
    nibble = np.where(escaped, 15, offset).astype(np.uint8)
    codes = (((words >> 8) & 0x80) | (words & 0x7F)).astype(np.uint8)
    out = np.zeros((rows, row_bytes), np.uint8)
    out[:, :width] = codes
    out[:, width:width + width // 2] = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).astype(np.uint8)
    base_at = width + width // 2
    out[:, base_at] = base
    out[:, base_at + 1] = escapes.astype(np.uint8)
    # Each escaped value's (column, BF16 bits) in column order: a stable sort puts the escaped
    # columns first while keeping their ascending order.
    order = np.argsort(-escaped.astype(np.int8), axis=1, kind="stable")[:, :MAX_ESCAPES]
    valid = np.arange(MAX_ESCAPES)[None, :] < escapes[:, None]
    bits = np.take_along_axis(words, order, axis=1)
    for slot in range(MAX_ESCAPES):
        present = valid[:, slot]
        entry = base_at + 2 + 3 * slot
        out[present, entry] = order[present, slot].astype(np.uint8)
        out[present, entry + 1] = (bits[present, slot] & 0xFF).astype(np.uint8)
        out[present, entry + 2] = ((bits[present, slot] >> 8) & 0xFF).astype(np.uint8)
    return out


def decode_rows(rows: np.ndarray, width: int) -> np.ndarray:
    """An independent reader: volume rows uint8 [n, row_bytes] -> BF16 bits uint16 [n, width]."""
    count, row_bytes = rows.shape
    if row_bytes != width + width // 2 + 16:
        raise ValueError(f"a BF16 P12 row of {width} values is {width + width // 2 + 16} bytes")
    codes = rows[:, :width].astype(np.uint16)
    nibbles = rows[:, width:width + width // 2]
    base = rows[:, width + width // 2].astype(np.uint16)
    escapes = rows[:, width + width // 2 + 1].astype(np.int64)
    if np.any(escapes > MAX_ESCAPES):
        raise ValueError("a BF16 P12 row records more than 4 escapes")
    if np.any(base > 240):
        raise ValueError("a BF16 P12 base is outside [0, 240]")
    entries = rows[:, width + width // 2 + 2:width + width // 2 + 2 + 3 * MAX_ESCAPES].reshape(
        count, MAX_ESCAPES, 3)
    j = np.arange(width)
    q = (nibbles[:, j // 2].astype(np.uint16) >> (4 * (j & 1)).astype(np.uint16)) & 15
    in_window = q < 15
    bits = (((codes & 0x80) << 8) | ((base[:, None] + q) << 7) | (codes & 0x7F)).astype(np.uint16)
    out = np.zeros((count, width), np.uint16)
    out[in_window] = bits[in_window]
    row_index = np.arange(count)
    hit = np.zeros((count, width), bool)
    for slot in range(MAX_ESCAPES):
        present = escapes > slot
        column = entries[:, slot, 0].astype(np.int64)
        value = entries[:, slot, 1].astype(np.uint16) | (entries[:, slot, 2].astype(np.uint16) << 8)
        out[row_index[present], column[present]] = value[present]
        hit[row_index[present], column[present]] = True
    escaped = ~in_window
    if np.any(escaped.sum(axis=1) != escapes):
        raise ValueError("a BF16 P12 row's escape count differs from its escaped columns")
    if np.any(hit & ~escaped):
        raise ValueError("a BF16 P12 escape names a column that is in the exponent window")
    if not np.all(hit[escaped]):
        raise ValueError("a BF16 P12 row has an escaped column with no escape entry")
    return out


def packed_volume_id(source_id: bytes) -> bytes:
    """The packed volume's id, which follows from its source's: every repack of one BF16 volume is the
    same file byte for byte (a local repack can be checked against a published one by its hash)."""
    return hashlib.sha256(b"infernix bf16_p12 v1" + bytes(source_id)).digest()[:16]


def write_volume(source: Volume, width: int, out: Path, volume_id: bytes, progress=None) -> dict:
    """Write the packed rows in the engine's v1 container (tag 6), the layout Volume reads."""
    if len(volume_id) != 16:
        raise ValueError("the n-gram volume id is 16 bytes")
    row_bytes = width + width // 2 + 16
    rows_per_block = BLOCK_BYTES // row_bytes
    if row_bytes * rows_per_block != BLOCK_BYTES:
        raise ValueError(f"BF16 P12 rows of {row_bytes} bytes do not fill a 4 KiB block")
    rows = source.rows
    blocks = -(-rows // rows_per_block)
    header = bytearray(BLOCK_BYTES)
    NGRAM_HEADER.pack_into(header, 0, NGRAM_MAGIC, NGRAM_VERSION, BLOCK_BYTES, rows, row_bytes,
                           rows_per_block, BLOCK_BYTES, blocks, volume_id)
    struct.pack_into("<H", header, NGRAM_SCALE_OFFSET, 0)
    struct.pack_into("<H", header, NGRAM_FORMAT_OFFSET, TAG_BF16_P12)
    struct.pack_into("<I", header, NGRAM_NVFP4_SCALE_OFFSET, 0)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"n-gram volume already exists: {out}")
    temporary = out.with_name(out.name + ".tmp")
    batch_rows = 1 << 16
    try:
        with temporary.open("xb") as stream:
            stream.write(header)
            pending = np.empty((0, row_bytes), np.uint8)
            for begin in range(0, rows, batch_rows):
                end = min(rows, begin + batch_rows)
                pending = np.concatenate((pending, pack_rows(source.words(begin, end), width, begin)), axis=0)
                whole = pending.shape[0] // rows_per_block
                if whole:
                    block_array = np.zeros((whole, BLOCK_BYTES), np.uint8)
                    block_array[:, : rows_per_block * row_bytes] = pending[: whole * rows_per_block].reshape(
                        whole, rows_per_block * row_bytes)
                    stream.write(block_array.tobytes())
                    pending = pending[whole * rows_per_block:]
                if progress is not None:
                    progress(end, rows)
            if pending.shape[0]:
                filled = np.zeros((rows_per_block, row_bytes), np.uint8)
                filled[: pending.shape[0]] = pending
                block_array = np.zeros((1, BLOCK_BYTES), np.uint8)
                block_array[0, : rows_per_block * row_bytes] = filled.reshape(-1)
                stream.write(block_array.tobytes())
        if temporary.stat().st_size != BLOCK_BYTES * (1 + blocks):
            raise ValueError("n-gram volume size differs from its geometry")
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return {"rows": rows, "row_bytes": row_bytes, "rows_per_block": rows_per_block,
            "block_bytes": BLOCK_BYTES, "header_bytes": BLOCK_BYTES, "blocks": blocks,
            "file_bytes": BLOCK_BYTES * (1 + blocks)}


def percentile(histogram: np.ndarray, total: int, fraction: float) -> int:
    if total == 0:
        return 0
    target = max(1, math.ceil(fraction * total))
    return int(np.searchsorted(np.cumsum(histogram), target))


def histogram_dict(histogram: np.ndarray) -> dict:
    return {str(value): int(count) for value, count in enumerate(histogram) if count}


def require_bf16_source(source: Volume, action: str) -> None:
    # The engine resolves an untagged volume (tag 0) by its row size, where BF16 rows are the first
    # inference; accept an even-sized untagged row as BF16 the same way.
    if not (source.tag == TAG_BF16 or (source.tag == 0 and source.row_bytes % 2 == 0)):
        raise ValueError(f"{source.path}: {action} needs a BF16 volume (tag 3, or untagged even-sized rows), "
                         f"this one is {source.format} with {source.row_bytes}-byte rows")
    if source.rows_per_block != 12:
        raise ValueError(f"{source.path}: a BF16 source must hold 12 rows per block, this one holds "
                         f"{source.rows_per_block}")


def census(source: Volume, chunk_rows: int = 1 << 15) -> dict:
    require_bf16_source(source, "--census")
    width = source.row_bytes // 2
    codec_histogram = np.zeros(width + 1, np.int64)  # the codec's rule, b in [0, 240]
    contrast_histogram = np.zeros(width + 1, np.int64)  # b in [1, 240]
    all_zero = 0
    for begin in range(0, source.rows, chunk_rows):
        end = min(source.rows, begin + chunk_rows)
        words = source.words(begin, end)
        counts = window_counts(words)
        _, codec_window = best_base(counts, 0)
        _, contrast_window = best_base(counts, 1)
        codec_histogram += np.bincount(width - codec_window, minlength=width + 1)
        contrast_histogram += np.bincount(width - contrast_window, minlength=width + 1)
        all_zero += int(np.all((words & 0x7FFF) == 0, axis=1).sum())
    total = int(codec_histogram.sum())
    return {
        "source": str(source.path),
        "rows": source.rows,
        "row_bytes": source.row_bytes,
        "width": width,
        "codec": {
            "rule": "base in [0, 240] (exponent 0 is in window at base 0)",
            "escape_histogram": histogram_dict(codec_histogram),
            "p99_9_escapes": percentile(codec_histogram, total, 0.999),
            "max_escapes": int(np.max(np.nonzero(codec_histogram)[0], initial=0)),
            "rows_over_budget": int(codec_histogram[MAX_ESCAPES + 1:].sum()),
            "all_zero_rows": all_zero,
        },
        "contrast": {
            "rule": "base in [1, 240] (exponent 0 always escapes)",
            "escape_histogram": histogram_dict(contrast_histogram),
            "p99_9_escapes": percentile(contrast_histogram, total, 0.999),
            "max_escapes": int(np.max(np.nonzero(contrast_histogram)[0], initial=0)),
            "rows_over_budget": int(contrast_histogram[MAX_ESCAPES + 1:].sum()),
        },
    }


def verify(source: Volume, volume: Volume, rows: str, chunk_rows: int = 1 << 15) -> dict:
    require_bf16_source(source, "--verify")
    if volume.tag != TAG_BF16_P12:
        raise ValueError(f"{volume.path}: --verify needs a BF16 P12 volume (tag 6), this one is {volume.format}")
    width = source.row_bytes // 2
    if volume.row_bytes != width + width // 2 + 16:
        raise ValueError(f"{volume.path}: BF16 P12 rows of {volume.row_bytes} bytes do not match a "
                         f"{width}-value row")
    if volume.rows != source.rows:
        raise ValueError(f"{volume.path}: {volume.rows} rows differ from the source's {source.rows}")
    checked = 0
    bad = 0
    if rows == "all":
        for begin in range(0, source.rows, chunk_rows):
            end = min(source.rows, begin + chunk_rows)
            want = source.words(begin, end)
            got = decode_rows(volume.read_rows(begin, end), width)
            bad += int((got != want).any(axis=1).sum())
            checked += end - begin
    else:
        count = min(int(rows), source.rows)
        rng = np.random.default_rng(7)
        indices = np.sort(rng.choice(source.rows, size=count, replace=False))
        for begin in range(0, count, chunk_rows):
            chunk = indices[begin:begin + chunk_rows]
            want = np.ascontiguousarray(source.read_many(chunk)).view("<u2").reshape(len(chunk), width)
            got = decode_rows(volume.read_many(chunk), width)
            bad += int((got != want).any(axis=1).sum())
            checked += len(chunk)
    return {"source": str(source.path), "volume": str(volume.path), "rows_checked": checked,
            "row_mismatches": bad}


def stats(source: Volume, volume: Volume) -> dict:
    if volume.tag != TAG_BF16_P12:
        raise ValueError(f"{volume.path}: --stats needs a BF16 P12 volume (tag 6), this one is {volume.format}")
    width = (volume.row_bytes - 16) * 2 // 3
    if width <= 0 or width + width // 2 + 16 != volume.row_bytes:
        raise ValueError(f"{volume.path}: {volume.row_bytes}-byte rows are not a BF16 P12 row")
    histogram = np.zeros(width + 1, np.int64)
    base_max = 0
    nibble_escapes = 0
    chunk_rows = 1 << 16
    for begin in range(0, volume.rows, chunk_rows):
        end = min(volume.rows, begin + chunk_rows)
        rows = volume.read_rows(begin, end)
        base = rows[:, width + width // 2]
        escapes = rows[:, width + width // 2 + 1].astype(np.int64)
        base_max = max(base_max, int(base.max(initial=0)))
        histogram += np.bincount(escapes, minlength=width + 1)
        nibbles = rows[:, width:width + width // 2]
        nibble_escapes += int(((nibbles & 0x0F) == 15).sum()) + int(((nibbles >> 4) == 15).sum())
    recorded = int(sum(value * count for value, count in enumerate(histogram)))
    return {
        "volume": str(volume.path),
        "format": volume.format,
        "row_bytes": volume.row_bytes,
        "width": width,
        "rows": volume.rows,
        "file_bytes": volume.path.stat().st_size,
        "volume_id": volume.volume_id.hex(),
        "escape_histogram": histogram_dict(histogram),
        "rows_over_budget": int(histogram[MAX_ESCAPES + 1:].sum()),
        "max_base": base_max,
        "recorded_escapes": recorded,
        "nibble_escapes": nibble_escapes,
        "source_rows": source.rows,
        "source_row_bytes": source.row_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="the BF16 n-gram volume to read")
    parser.add_argument("--out", type=Path,
                        help="the BF16 P12 volume to write (--out) or to read (--verify, --stats)")
    parser.add_argument("--census", action="store_true", help="per-row base and escape statistics of the source")
    parser.add_argument("--verify", action="store_true", help="compare the P12 volume with the source row by row")
    parser.add_argument("--stats", action="store_true", help="the P12 volume's escape histogram and geometry")
    parser.add_argument("--rows", default="200000", help="rows --verify checks: 'all' or a sample size")
    args = parser.parse_args()
    writing = args.out is not None and not (args.verify or args.stats)
    if sum((args.census, args.verify, args.stats, writing)) != 1:
        parser.error("give exactly one of --census, --out, --verify or --stats")
    if (args.verify or args.stats) and args.out is None:
        parser.error("--verify and --stats read the P12 volume given by --out")

    source = Volume(args.source)
    if args.census:
        result = census(source)
        print(json.dumps(result, indent=2))
        if result["codec"]["rows_over_budget"]:
            sys.exit(1)
        return
    if writing:
        require_bf16_source(source, "--out")
        width = source.row_bytes // 2
        if width > 255 or width % 2:
            parser.error(f"{args.source}: a BF16 P12 row needs an even width up to 255, this one is {width}")
        volume_id = packed_volume_id(source.volume_id)
        printed = [0]

        def progress(done: int, total: int) -> None:
            if done == total or done - printed[0] >= max(1, total // 100):
                printed[0] = done
                print(f"{done}/{total} rows", file=sys.stderr, flush=True)

        geometry = write_volume(source, width, args.out, volume_id, progress=progress)
        print(json.dumps({"volume": str(args.out), "format": "bf16_p12", "volume_id": volume_id.hex(),
                          **geometry}, indent=2))
        return
    volume = Volume(args.out)
    if args.verify:
        result = verify(source, volume, args.rows)
        print(json.dumps(result, indent=2))
        if result["row_mismatches"]:
            sys.exit(1)
        return
    print(json.dumps(stats(source, volume), indent=2))


if __name__ == "__main__":
    main()
