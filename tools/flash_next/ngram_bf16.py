"""Write or verify a BF16 n-gram volume: Qwen3.8-Flash-Next's PLE n-gram table unquantized.

    python -m tools.flash_next.ngram_bf16 --model CHECKPOINT --out VOLUME.ngram
    python -m tools.flash_next.ngram_bf16 --model CHECKPOINT --out VOLUME.ngram --verify [--rows all|N]

The artifacts store the table quantized (NVIDIA's FP8 E4M3 rows with one per-tensor scale, or a GGUF
import's IQ4_NL rows). A BF16 volume holds the checkpoint's BF16 rows unchanged (2 x 160 bytes each,
12 per 4 KiB block: ~109 GB), and any Qwen3.8-Flash-Next artifact runs with it in place of its own
table: pass it with --ngram-volume (the engine reads the row format from the volume's header; the
PLE then reads the BF16 rows as its embedding, with no dequantization). Rows are read as from any
volume (one 4 KiB block per row), so it reads no more often than the FP8 volume, only takes twice the
space; a drive with fast random reads (Optane, then NVMe) suits it.

CHECKPOINT is a BF16 Qwen3.8-Flash-Next checkpoint directory, or only its n-gram table and
config.json, e.g. from Qwen/Qwen3.8-Flash-Next (~102 GB of table):

    python -m tools.convert.fetch_tensors --repo Qwen/Qwen3.8-Flash-Next --revision SHA --out DIR \\
        --pattern "model.language_model.layers.*.ple.ple_embedding.*"

The volume is written to VOLUME.tmp and renamed once complete; --verify compares its rows (a sample,
or all) with the checkpoint's.

--dequantize instead writes an FP8 table's values as BF16, bf16(e4m3(code) x scale), which is exactly
what the PLE computes from the FP8 rows: runs with that volume must give the FP8 volume's outputs bit
for bit. It qualifies the engine's BF16 route; it adds no precision.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

from tools.convert import qwen4_exp
from tools.convert.sources.safetensors import SafetensorsSource


class DequantizedTable:
    """An FP8 table's rows as the PLE expands them, bf16(e4m3(code) x scale), in BF16 words (the
    product of a 4-bit and an 8-bit significand is exact in FP32; one rounding to BF16)."""

    def __init__(self, fp8: qwen4_exp.NgramTable):
        if fp8.bf16:
            raise ValueError("--dequantize needs a checkpoint whose n-gram table is FP8")
        self.fp8, self.shards, self.rows = fp8, fp8.shards, fp8.rows
        self.row_bytes = 2 * fp8.width
        self._scale = fp8.scale().float()

    def codes(self, name: str, begin: int, end: int) -> torch.Tensor:
        values = self.fp8.codes(name, begin, end).view(torch.float8_e4m3fn).float() * self._scale
        return values.to(torch.bfloat16).contiguous().view(torch.uint8).reshape(end - begin, self.row_bytes)

    def bf16_rows_match(self, name: str, begin: int, end: int, rows: torch.Tensor) -> torch.Tensor:
        """Per row, whether BF16 volume rows are these dequantized rows, word for word."""
        return (rows == self.codes(name, begin, end)).all(dim=1)


def bf16_table(store: SafetensorsSource, dequantize: bool = False):
    """The checkpoint's n-gram table as BF16 rows: its BF16 words, or with `dequantize` its FP8 rows'
    values."""

    config = qwen4_exp.text_config(store.config, mtp=False)
    text_prefix = "model.language_model." if "text_config" in store.config else "model."
    (ple_layer,) = config["ple_layer_ids"]
    prefix = f"{text_prefix}layers.{ple_layer - 1}."
    rows = qwen4_exp._check_ngram_buffers(store, prefix, config, 0)
    if dequantize:
        return DequantizedTable(qwen4_exp.NgramTable(store, prefix, config, rows))
    return qwen4_exp.NgramTable(store, prefix, config, rows, keep_bf16=True)


def write(model: Path, out: Path, dequantize: bool = False) -> dict:
    start = time.time()
    with SafetensorsSource(model) as store:
        table = bf16_table(store, dequantize)

        def progress(index: int, total: int) -> None:
            print(f"shard {index + 1}/{total} ({time.time() - start:.0f}s)", file=sys.stderr, flush=True)

        volume_id = os.urandom(16)
        geometry = qwen4_exp.write_ngram_volume(table, out, volume_id, progress=progress)
    return {"volume": str(out), "format": "bf16", "dequantized": dequantize, "volume_id": volume_id.hex(),
            **geometry, "seconds": round(time.time() - start, 1)}


def verify(model: Path, volume: Path, rows: str, dequantize: bool = False) -> dict:
    from tools.flash_next.verify_artifact import verify_ngram

    start = time.time()
    with SafetensorsSource(model) as store, volume.open("rb") as stream:
        table = bf16_table(store, dequantize)
        fields = qwen4_exp.NGRAM_HEADER.unpack(stream.read(qwen4_exp.NGRAM_HEADER.size))
        geometry = qwen4_exp.ngram_geometry(table.rows, table.row_bytes)
        checked, bad = verify_ngram(table, volume,
                                    {**geometry, "format": "bf16", "volume_id": fields[-1].hex()}, rows)
    return {"volume": str(volume), "rows_checked": checked, "row_mismatches": bad,
            "seconds": round(time.time() - start, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="BF16 checkpoint (or its n-gram table)")
    parser.add_argument("--out", type=Path, required=True, help="the BF16 n-gram volume")
    parser.add_argument("--verify", action="store_true", help="compare an existing volume with the checkpoint")
    parser.add_argument("--rows", default="200000", help="rows --verify compares: 'all' or a sample size")
    parser.add_argument("--dequantize", action="store_true",
                        help="an FP8 table's values as BF16 (qualifies the BF16 route; no added precision)")
    args = parser.parse_args()
    result = (verify(args.model, args.out, args.rows, args.dequantize) if args.verify
              else write(args.model, args.out, args.dequantize))
    print(json.dumps(result, indent=2))
    if args.verify and result["row_mismatches"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
