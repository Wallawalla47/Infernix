"""Counts the KV blocks a 12-bit lossless page form would hold (candidates §23.2 PB-01, step 1).

ZipFusion (arXiv 2610.33719) stores a block of 512 BF16 values whose exponents fit a 16-binade
window as each value's low byte (its seven fraction bits and lowest exponent bit r) plus a 4-bit
code: the sign and d = (e - b) // 2, the offset from the block base b, the block's smallest
exponent rounded down to even. Attention rebuilds e = b + 2d + r in registers: 12 bits a value. A
block whose exponents do not fit [b, b + 15] stays raw (16 bits). A zero has exponent field 0, so a
block holding one qualifies only if every value is below 2^-111.

This reads the dumps INFERNIX_TMP_KV_DUMP writes (one JSON per prompt; per attention layer one K
and one V file of the prompt's pages, each [kv head][slot][dim]) and reports, per layer and for K
and V, the share of complete 512-value blocks that qualify, in the page's storage order (a block is
512 / head_dim consecutive slots of one KV head), under

  block   the paper's rule: one base per block (1 byte of metadata per block);
  channel one base per head-dim channel per page and KV head, the smallest exponent of that channel
          over the page's slots rounded down to even (1 byte per channel and page: 0.125 bits a
          value), every value of the block within 15 of its channel's base;

with the average bits per value each gives (12 for a qualifying block, 16 for a raw one, plus the
metadata above and a 4-byte offset per page and KV head), and

  row     UZ-40's bf16_p12 form chosen at append time, with no calibration: one base per head row (a slot's
          head_dim values of one KV head), the lowest of the 15-binade windows holding most of the row's
          nonzero exponents; each value stores its sign and seven fraction bits plus a 4-bit exponent code,
          and the values outside the window (zeros among them) are the row's escapes, kept in E inline slots
          of 3 bytes (column and BF16 bits). A row with more than E escapes stays raw. Reported for E = 2, 4,
          6 and 8: the share of rows that fit and the bits a value of a fixed-size row (12 + (8 + 24 E) /
          head_dim), and of the whole plane with the raw rows in a pool of their own;

  bf16p   the shipped --kv-dtype bf16p (KvCacheStorage::Bf16Packed): the row rule with a 14-binade window of
          normal exponents, zeros and subnormals coded without escapes, and E = 6 escapes in an 8-byte side
          record (every D256 row 384 + 8 bytes); a row with more escapes has no packed form, and the request
          that appends it fails: the share of rows that pack, for E = 2, 4, 6 and 8;

The page size is 64 slots.

V is stored as FP16 on the default route (--kv-dtype bf16: the append converts V's BF16 source with
round-to-nearest). The 12-bit form keeps BF16's low byte, so it would store that BF16 source and
convert in registers, giving the same FP16 bits; the counts use the source, recovered exactly from
the FP16 value wherever the conversion was exact (2^-17 <= |v| < 2^16). The report gives the share of
V values outside that range, whose source the dump cannot recover; such values are tiny and break
their block under either rule anyway. A dump taken with --lossless stores V as BF16 and needs no
recovery.

  D:\\Infernix-fp4local-work\\venv311\\Scripts\\python.exe -I tools/kv_lossless_count/count.py DUMP_DIR_OR_JSON... [--csv FILE]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BLOCK = 512
WINDOW = 15  # e - base <= 15: sixteen binades
ROW_WINDOW = 15  # the row rule's codes 0-14 (15 escapes)
ROW_SLOTS = (2, 4, 6, 8)
PAGE = 64
TARGET = 0.80  # PB-01 builds only if at least 80 % of K and V bytes qualify


@dataclass
class Count:
    blocks: int = 0
    block_ok: int = 0
    channel_ok: int = 0
    zero_blocks: int = 0
    values: int = 0
    uncertain: int = 0  # V values whose BF16 source the FP16 dump cannot recover
    spans: list = field(default_factory=list)  # per-plane span histograms (max e - min e), 0..255
    block_meta_bits: float = 0.0
    channel_meta_bits: float = 0.0
    rows: int = 0
    row_ok: dict = field(default_factory=lambda: {e: 0 for e in ROW_SLOTS})
    row_escapes: int = 0  # escapes over every live row
    # The shipped bf16p rule: one of the 16 codes marks exponent 0 (zeros and subnormals: the code byte is then the
    # value), so they cost no escape, and the window of normal exponents is 14 binades.
    row_ok_z: dict = field(default_factory=lambda: {e: 0 for e in ROW_SLOTS})
    row_escapes_z: int = 0
    dim: int = 0

    def add(self, other: "Count") -> None:
        self.rows += other.rows
        for e in ROW_SLOTS:
            self.row_ok[e] += other.row_ok[e]
            self.row_ok_z[e] += other.row_ok_z[e]
        self.row_escapes += other.row_escapes
        self.row_escapes_z += other.row_escapes_z
        self.dim = other.dim or self.dim
        self.blocks += other.blocks
        self.block_ok += other.block_ok
        self.channel_ok += other.channel_ok
        self.zero_blocks += other.zero_blocks
        self.values += other.values
        self.uncertain += other.uncertain
        self.spans.extend(other.spans)
        self.block_meta_bits += other.block_meta_bits
        self.channel_meta_bits += other.channel_meta_bits

    def share(self, ok: int) -> float:
        return ok / self.blocks if self.blocks else float("nan")

    def bits(self, ok: int, meta_bits: float) -> float:
        if not self.blocks:
            return float("nan")
        payload = 12.0 * ok + 16.0 * (self.blocks - ok)
        return payload / self.blocks + meta_bits / (self.blocks * BLOCK)

    def row_bits(self, slots: int) -> tuple[float, float]:
        """Bits a value of a fixed-size row, and of the plane with raw rows pooled apart."""
        if not self.rows:
            return (float("nan"), float("nan"))
        fixed = 12.0 + (8.0 + 24.0 * slots) / self.dim
        share = self.row_ok[slots] / self.rows
        return fixed, share * fixed + (1.0 - share) * (16.0 + 32.0 / self.dim)

    def bf16p_share(self, slots: int) -> float:
        """The share of rows with a packed form under the shipped rule with `slots` escapes."""
        return self.row_ok_z[slots] / self.rows if self.rows else float("nan")

    def span_percentiles(self) -> tuple[int, int, int]:
        if not self.spans:
            return (0, 0, 0)
        histogram = np.sum(self.spans, axis=0)
        cumulative = np.cumsum(histogram)
        total = cumulative[-1]
        return tuple(int(np.searchsorted(cumulative, q * total)) for q in (0.5, 0.9, 0.99))


def bf16_bits(raw: np.ndarray, dtype: str) -> tuple[np.ndarray, int]:
    """The values as BF16 bits, and how many of them could not be recovered exactly."""
    if dtype == "bf16":
        return raw, 0
    if dtype != "fp16":
        raise SystemExit(f"unsupported plane dtype {dtype}")
    values = raw.view(np.float16).astype(np.float32)
    bits = values.view(np.uint32)
    magnitude = np.abs(values)
    uncertain = int(np.count_nonzero(~np.isfinite(values) | (magnitude < 2.0**-17)))
    return (bits >> 16).astype(np.uint16), uncertain


def count_plane(path: Path, dtype: str, pages: int, heads: int, dim: int, tokens: int) -> Count:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != pages * heads * PAGE * dim:
        raise SystemExit(f"{path}: {raw.size} values, expected {pages * heads * PAGE * dim}")
    bits, uncertain = bf16_bits(raw, dtype)
    exponent = ((bits >> 7) & 0xFF).astype(np.uint8).reshape(pages, heads, PAGE, dim)
    if BLOCK % dim != 0:
        raise SystemExit(f"head_dim {dim} does not divide a {BLOCK}-value block")
    slots_per_block = BLOCK // dim
    blocks_per_page = PAGE // slots_per_block

    # Slots holding the prompt (the last page may be partial) and the blocks wholly inside them.
    live_slots = np.minimum(PAGE, tokens - PAGE * np.arange(pages))
    slot_live = np.arange(PAGE)[None, :] < live_slots[:, None]  # [pages, PAGE]
    block_live = (np.arange(blocks_per_page)[None, :] + 1) * slots_per_block <= live_slots[:, None]
    block_live = np.broadcast_to(block_live[:, None, :], (pages, heads, blocks_per_page))

    blocks = exponent.reshape(pages, heads, blocks_per_page, BLOCK)
    lowest = blocks.min(axis=-1)
    highest = blocks.max(axis=-1)
    base = lowest & np.uint8(0xFE)
    block_ok = (highest.astype(np.int16) - base.astype(np.int16)) <= WINDOW

    # Per-channel bases over each page's live slots.
    masked = np.where(slot_live[:, None, :, None], exponent, np.uint8(0xFF))
    channel_base = masked.min(axis=2) & np.uint8(0xFE)  # [pages, heads, dim]
    within = (exponent.astype(np.int16) - channel_base[:, :, None, :].astype(np.int16)) <= WINDOW
    channel_ok = within.reshape(pages, heads, blocks_per_page, BLOCK).all(axis=-1)

    # The row rule: per (page, head, slot) row, the window [b, b + 14] (b in 1..240) holding most nonzero exponents.
    row_live = np.broadcast_to(slot_live[:, None, :], (pages, heads, PAGE))
    flat = exponent.reshape(-1, dim)[row_live.reshape(-1)]
    histogram = np.zeros((flat.shape[0], 256), dtype=np.int32)
    np.add.at(histogram, (np.repeat(np.arange(flat.shape[0]), dim), flat.reshape(-1).astype(np.int64)), 1)
    histogram[:, 0] = 0  # zeros (and subnormals) always escape
    histogram[:, 255] = 0
    cumulative = np.concatenate([np.zeros((flat.shape[0], 1), dtype=np.int32), np.cumsum(histogram, axis=1)], axis=1)
    windows = cumulative[:, 1 + ROW_WINDOW : 241 + ROW_WINDOW] - cumulative[:, 1:241]  # bases 1..240
    escapes = dim - windows.max(axis=1)
    zeros = np.count_nonzero(flat == 0, axis=1)
    windows_z = cumulative[:, ROW_WINDOW : 240 + ROW_WINDOW] - cumulative[:, 1:241]  # [b, b + 13], b in 1..240
    escapes_z = dim - zeros - windows_z.max(axis=1)

    span = (highest.astype(np.int16) - lowest.astype(np.int16))[block_live]
    live_blocks = int(np.count_nonzero(block_live))
    groups = pages * heads  # one 4-byte offset per page and KV head
    return Count(
        blocks=live_blocks,
        block_ok=int(np.count_nonzero(block_ok & block_live)),
        channel_ok=int(np.count_nonzero(channel_ok & block_live)),
        zero_blocks=int(np.count_nonzero((lowest == 0) & block_live)),
        values=int(raw.size),
        uncertain=uncertain,
        spans=[np.bincount(span, minlength=256)[:256]],
        block_meta_bits=live_blocks * 8.0 + groups * 32.0,
        channel_meta_bits=groups * (dim * 8.0 + 32.0) + live_blocks * 1.0,
        rows=int(flat.shape[0]),
        row_ok={e: int(np.count_nonzero(escapes <= e)) for e in ROW_SLOTS},
        row_escapes=int(escapes.sum()),
        row_ok_z={e: int(np.count_nonzero(escapes_z <= e)) for e in ROW_SLOTS},
        row_escapes_z=int(escapes_z.sum()),
        dim=dim,
    )


HEADER = (
    f"{'layer':>6} {'model':>5} {'plane':>5} {'blocks':>9} {'zero%':>6} {'span p50/p90/p99':>17}"
    f" {'block ok%':>9} {'bits':>6} {'channel ok%':>11} {'bits':>6}"
    f" {'esc/row':>7} " + " ".join(f"{f'row E{e} ok%':>11} {'fixed':>6} {'pooled':>6}" for e in ROW_SLOTS)
    + f" {'bf16p esc/row':>13} " + " ".join(f"{f'bf16p E{e} ok%':>13}" for e in ROW_SLOTS)
)


def line(layer: str, model_layer: str, plane: str, count: Count) -> str:
    p50, p90, p99 = count.span_percentiles()
    zero = 100.0 * count.zero_blocks / count.blocks if count.blocks else float("nan")
    text = (
        f"{layer:>6} {model_layer:>5} {plane:>5} {count.blocks:>9} {zero:>6.2f} {f'{p50}/{p90}/{p99}':>17}"
        f" {100.0 * count.share(count.block_ok):>9.2f} {count.bits(count.block_ok, count.block_meta_bits):>6.2f}"
        f" {100.0 * count.share(count.channel_ok):>11.2f} {count.bits(count.channel_ok, count.channel_meta_bits):>6.2f}"
        f" {count.row_escapes / count.rows if count.rows else float('nan'):>7.3f} "
        + " ".join(
            f"{100.0 * count.row_ok[e] / count.rows if count.rows else float('nan'):>11.2f}"
            f" {count.row_bits(e)[0]:>6.2f} {count.row_bits(e)[1]:>6.2f}"
            for e in ROW_SLOTS
        )
        + f" {count.row_escapes_z / count.rows if count.rows else float('nan'):>13.3f} "
        + " ".join(f"{100.0 * count.bf16p_share(e):>13.4f}" for e in ROW_SLOTS)
    )
    if count.uncertain:
        text += f"  source unrecoverable: {100.0 * count.uncertain / count.values:.4f}% of values"
    return text


def verdict(k: Count, v: Count) -> str:
    parts = []
    for name, ok_k, ok_v in (("block", k.block_ok, v.block_ok), ("channel", k.channel_ok, v.channel_ok)):
        share_k, share_v = k.share(ok_k), v.share(ok_v)
        met = share_k >= TARGET and share_v >= TARGET
        parts.append(f"{name} base: K {100 * share_k:.1f}%, V {100 * share_v:.1f}% -> {'met' if met else 'not met'}")
    return f"PB-01 (>= {100 * TARGET:.0f}% of K and of V bytes): " + "; ".join(parts)


def dumps(paths: list[Path]) -> list[Path]:
    found = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.glob("kv-*.json")))
        else:
            found.append(path)
    if not found:
        raise SystemExit("no kv-*.json dumps found")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dumps", nargs="+", type=Path, help="dump directories or kv-*.json files")
    parser.add_argument("--csv", type=Path, help="also write one row per dump, layer and plane")
    args = parser.parse_args()

    rows = []
    by_model: dict[str, dict[str, Count]] = {}
    for meta_path in dumps(args.dumps):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != "infernix-kv-dump-1":
            raise SystemExit(f"{meta_path}: not an INFERNIX_TMP_KV_DUMP file")
        if meta["page_size"] != PAGE:
            raise SystemExit(f"{meta_path}: page size {meta['page_size']}, expected {PAGE}")
        tokens, pages, heads, dim = meta["tokens"], meta["pages"], meta["kv_heads"], meta["head_dim"]
        print(
            f"{meta_path.stem}: model={meta['model']} tokens={tokens} pages={pages} kv_heads={heads}"
            f" head_dim={dim} rotary_dim={meta['rotary_dim']} K={meta['key_dtype']} V={meta['value_dtype']}"
        )
        if meta["value_dtype"] == "fp16":
            print("  V stored as FP16: counted on its BF16 source (recovered from the FP16 values)")
        print(HEADER)
        totals = {"K": Count(), "V": Count()}
        for layer in meta["layers"]:
            for plane, key, dtype in (("K", "key", meta["key_dtype"]), ("V", "value", meta["value_dtype"])):
                count = count_plane(meta_path.parent / layer[key], dtype, pages, heads, dim, tokens)
                totals[plane].add(count)
                print(line(f"L{layer['ordinal']:02d}", str(layer["model_layer"]), plane, count))
                rows.append(
                    {
                        "dump": meta_path.stem,
                        "model": meta["model"],
                        "tokens": tokens,
                        "layer": layer["ordinal"],
                        "model_layer": layer["model_layer"],
                        "plane": plane,
                        "blocks": count.blocks,
                        "zero_blocks": count.zero_blocks,
                        "block_ok": count.block_ok,
                        "block_bits": count.bits(count.block_ok, count.block_meta_bits),
                        "channel_ok": count.channel_ok,
                        "channel_bits": count.bits(count.channel_ok, count.channel_meta_bits),
                        "rows": count.rows,
                        "row_escapes": count.row_escapes,
                        **{f"row_ok_e{e}": count.row_ok[e] for e in ROW_SLOTS},
                        "bf16p_escapes": count.row_escapes_z,
                        **{f"bf16p_ok_e{e}": count.row_ok_z[e] for e in ROW_SLOTS},
                        "unrecoverable_values": count.uncertain,
                    }
                )
        both = Count()
        both.add(totals["K"])
        both.add(totals["V"])
        for name, count in (("K", totals["K"]), ("V", totals["V"]), ("K+V", both)):
            print(line("all", "", name, count))
        print("  " + verdict(totals["K"], totals["V"]))
        model = by_model.setdefault(meta["model"], {"K": Count(), "V": Count()})
        model["K"].add(totals["K"])
        model["V"].add(totals["V"])
        print()

    if len(rows) and len({row["dump"] for row in rows}) > 1:
        for model, totals in by_model.items():
            print(f"{model}: every dump")
            print(HEADER)
            both = Count()
            both.add(totals["K"])
            both.add(totals["V"])
            for name, count in (("K", totals["K"]), ("V", totals["V"]), ("K+V", both)):
                print(line("all", "", name, count))
            print("  " + verdict(totals["K"], totals["V"]))
            print()

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
