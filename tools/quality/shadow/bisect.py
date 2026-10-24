"""Where the lossless route first leaves the publisher's code (IQS-2 G1): a layer-by-layer comparison.

    python -m tools.quality.shadow.bisect --artifact ART --capture CAPTURE --dump DIR [--tokens N] [--sequence I]

Run `infernix-certify capture ... --windows I --tokens 128` with INFERNIX_LOSSLESS_DUMP_DIR=DIR first: the engine
then writes the BF16 residual stream after the embedding, after every layer's mixer and after every layer
(src/models/qwen3_5/execution/text.cpp). This tool runs the publisher's code in exact arithmetic on the first N
tokens of the same sequence (N at most the capture's) and compares each checkpoint bit for bit, reporting for each
the number of differing values, the largest difference and the first differing (token, channel): the first
checkpoint that differs names the operation to examine. Qwen3.5 and Qwen3.6 artifacts (transformers 5.5.0 first on
PYTHONPATH, as for the conformance run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .weights import Weights


def bf16_values(raw: np.ndarray) -> np.ndarray:
    """BF16 bit patterns as float32."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def load_dump(directory: Path, chunk: int, name: str, tokens: int, hidden: int) -> np.ndarray | None:
    path = directory / f"chunk{chunk}_{name}.bin"
    if not path.is_file():
        return None
    raw = np.fromfile(path, dtype="<u2")
    return bf16_values(raw.reshape(-1, hidden)[:tokens])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--capture", type=Path, required=True, help="infernix-certify capture's prefix")
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--sequence", type=int, default=0, help="index of the captured sequence")
    parser.add_argument("--chunk", type=int, default=0, help="which lossless prefill call of the engine run")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--detail-layers", default="",
                        help="comma-separated delta-net layers whose intermediates are compared too")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    meta = json.loads(Path(str(args.capture) + ".json").read_text(encoding="utf-8"))
    sequence = [s for s in meta["sequences"] if s["index"] == args.sequence][0]
    weights = Weights(args.artifact)
    hidden_size = int(weights.config["hidden_size"])
    embed = args.dump / f"chunk{args.chunk}_embed.bin"
    # the engine feeds all but the window's last token (that one is only a target), so it dumps one row fewer
    rows = embed.stat().st_size // (2 * hidden_size) if embed.is_file() else args.tokens
    tokens = sequence["tokens"][: min(args.tokens, rows)]
    model_type = weights.config.get("model_type", "")
    if model_type not in ("qwen3_5_text", "qwen3_5_moe_text"):
        raise SystemExit(f"bisect covers Qwen3.5/3.6 artifacts, not {model_type!r}")
    from . import qwen3_5
    hidden = int(weights.config["hidden_size"])
    started = time.time()
    record: dict = {}
    detail = frozenset(int(x) for x in args.detail_layers.split(",") if x)
    qwen3_5.run(weights, [tokens], [[len(tokens) - 1]], lambda m: print(f"[{time.time() - started:7.0f} s] {m}", flush=True),
                record=record, stop_after_layers=True, detail=detail)
    layers = int(weights.config["num_hidden_layers"])
    order = ["embed"] + [(i, tag) for i in range(layers) for tag in ("mixer", "layer")]
    report, first = [], None
    for key in order:
        engine = load_dump(args.dump, args.chunk, key if isinstance(key, str) else f"layer{key[0]:02d}_{key[1]}",
                           len(tokens), hidden)
        if engine is None:
            report.append({"checkpoint": str(key), "missing": True})
            continue
        ours = record[key][0].to(torch.float32).numpy()
        differ = engine.view(np.uint32) != ours.view(np.uint32)
        entry = {"checkpoint": str(key), "values": int(engine.size), "different": int(differ.sum()),
                 "tokens_with_a_difference": int(differ.any(axis=1).sum())}
        if differ.any():
            t, ch = np.argwhere(differ)[0]
            entry.update({"first": {"token": int(t), "channel": int(ch), "engine": float(engine[t, ch]),
                                    "shadow": float(ours[t, ch])},
                          "max_abs_difference": float(np.abs(engine.astype(np.float64) - ours.astype(np.float64)).max())})
            first = first or entry
        report.append(entry)
        print(f"{str(key):>18}  differ {entry['different']:>8} of {entry['values']}"
              + (f"  max |d| {entry['max_abs_difference']:.4g}  first {entry['first']}" if differ.any() else "  identical"),
              flush=True)
    for layer in sorted(detail):
        for name in ("mixed", "z", "a", "b", "g", "beta", "conv", "o", "on", "y"):
            key = (layer, "gdn", name)
            path = args.dump / f"chunk{args.chunk}_layer{layer:02d}_gdn_{name}{'_f32' if name == 'g' else ''}.bin"
            if key not in record or not path.is_file():
                print(f"{str(key):>24}  missing")
                continue
            ours = record[key][0].to(torch.float32).numpy()
            channels = ours.shape[-1]
            if name == "g":
                engine = np.fromfile(path, dtype="<f4").reshape(-1, channels)[: len(tokens)]
            else:
                engine = bf16_values(np.fromfile(path, dtype="<u2").reshape(-1, channels)[: len(tokens)])
            differ = engine.view(np.uint32) != ours.view(np.uint32)
            entry = {"checkpoint": str(key), "values": int(engine.size), "different": int(differ.sum())}
            if differ.any():
                t, ch = np.argwhere(differ)[0]
                entry["first"] = {"token": int(t), "channel": int(ch), "engine": float(engine[t, ch]),
                                  "shadow": float(ours[t, ch])}
                entry["max_abs_difference"] = float(np.abs(engine.astype(np.float64) - ours.astype(np.float64)).max())
                entry["tokens"] = sorted({int(x) for x in np.argwhere(differ)[:, 0]})
            report.append(entry)
            print(f"{str(key):>24}  differ {entry['different']:>7} of {entry['values']}"
                  + (f"  max |d| {entry['max_abs_difference']:.4g}  first {entry['first']}  tokens {entry['tokens']}"
                     if differ.any() else "  identical"), flush=True)
    result = {"artifact": args.artifact, "tokens": len(tokens), "first_difference": first, "checkpoints": report}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print("first difference:", first["checkpoint"] if first else "none: every checkpoint is bit-identical")
    return 0 if first is None else 1


if __name__ == "__main__":
    sys.exit(main())
