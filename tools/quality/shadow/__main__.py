"""IQS-2 G1, reference conformance: the publisher's transformers code in exact arithmetic against --lossless.

    python -m tools.quality.shadow CAPTURE --artifact ART --out RESULT.json [--sequences N] [--ngram-volume V]

CAPTURE is `infernix-certify capture`'s prefix: CAPTURE.json (the sequences' tokens and the positions whose
logits it kept) and CAPTURE.f32 (those rows, the lossless route's FP32 logits). The shadow (tools/quality/
shadow) evaluates the same tokens with the artifact's weights decoded exactly, every operation of the
publisher's model code correctly rounded to its dtype, and every logit must equal the engine's bit for bit.

The publisher's code is pinned per architecture (docs/maintainer/lossless-route.md): Qwen3.5/3.6 on
transformers 5.5.0 (run with that release first on PYTHONPATH), Qwen3.8-Flash-Next on main 3bb8b727.
RESULT.json: {"passed", "detail", "rows", "elements", "mismatched_rows", "mismatched_elements", ...}, the form
tools/quality/iqs.py's certificate reads. Exit status 0 when every logit is equal, 1 otherwise.
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


def compare(engine: np.ndarray, shadow: torch.Tensor) -> dict:
    """Bitwise comparison of one sequence's rows ([positions, vocab] FP32 against binary64 holding FP32 values)."""
    ours = shadow.to(torch.float32).numpy()
    differ = engine.view(np.uint32) != ours.view(np.uint32)
    rows = differ.any(axis=1)
    out = {"rows": int(engine.shape[0]), "elements": int(engine.size), "mismatched_rows": int(rows.sum()),
           "mismatched_elements": int(differ.sum()),
           "top1_equal": int((engine.argmax(axis=1) == ours.argmax(axis=1)).sum())}
    if differ.any():
        r, v = np.argwhere(differ)[0]
        out["first"] = {"row": int(r), "token": int(v), "engine": float(engine[r, v]), "shadow": float(ours[r, v])}
        out["max_abs_difference"] = float(np.abs(engine.astype(np.float64) - ours.astype(np.float64)).max())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequences", type=int, help="compare only the first N sequences")
    parser.add_argument("--ngram-volume", help="Qwen3.8-Flash-Next: the artifact's n-gram volume")
    parser.add_argument("--threads", type=int)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    meta = json.loads(Path(str(args.capture) + ".json").read_text(encoding="utf-8"))
    vocab = int(meta["vocab"])
    sequences = meta["sequences"][: args.sequences] if args.sequences else meta["sequences"]
    rows = np.fromfile(Path(str(args.capture) + ".f32"), dtype="<f4")
    started = time.time()
    weights = Weights(args.artifact)
    model_type = weights.config.get("model_type", "")
    tokens = [s["tokens"] for s in sequences]
    positions = [s["positions"] for s in sequences]
    log = lambda message: print(f"[{time.time() - started:7.0f} s] {message}", flush=True)
    if model_type in ("qwen3_5_text", "qwen3_5_moe_text"):
        from . import qwen3_5
        logits = qwen3_5.run(weights, tokens, positions, log)
    elif model_type.startswith("qwen4_exp"):
        from . import qwen4_exp
        logits = qwen4_exp.run(weights, tokens, positions, log, ngram_volume=args.ngram_volume)
    else:
        raise SystemExit(f"no shadow for {model_type!r}")
    results, offset = [], 0
    for s, ours in zip(sequences, logits):
        count = len(s["positions"])
        engine = rows[offset * vocab:(offset + count) * vocab].reshape(count, vocab)
        offset += count
        results.append({"index": s["index"], "id": s["id"], **compare(engine, ours)})
    total = {k: sum(r[k] for r in results) for k in ("rows", "elements", "mismatched_rows", "mismatched_elements",
                                                     "top1_equal")}
    passed = total["mismatched_elements"] == 0
    import transformers
    result = {"passed": passed, **total, "sequences": results, "transformers": transformers.__version__,
              "artifact": args.artifact, "seconds": time.time() - started,
              "detail": (f"{total['elements']} logits of {total['rows']} positions bit-identical to the publisher's "
                         f"code in exact arithmetic (transformers {transformers.__version__})" if passed else
                         f"{total['mismatched_elements']} of {total['elements']} logits differ in "
                         f"{total['mismatched_rows']} of {total['rows']} positions")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(result["detail"])
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
