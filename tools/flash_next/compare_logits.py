"""Compare Infernix logits (FP32 .bin of one position) with the FP64 reference (reference.py .npz).

    python -m tools.flash_next.compare_logits --reference ref.npz --position -1 --infernix logits.bin         [--residuals residuals.bin --routes routes.bin]

The optional per-block taps of infernix_qwen4_exp_forward_real_test (BF16 [blocks, T, S*H] residuals
and I32 [blocks, T, top k] routed expert ids) are compared block by block: the relative residual
error per position and the routed-set overlap show where any divergence starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref_lp, cand_lp = log_softmax(reference), log_softmax(candidate)
    p = np.exp(ref_lp)
    top_ref = np.argsort(-reference)[:10]
    top_cand = np.argsort(-candidate)[:10]
    return {
        "kl_ref_to_infernix": float((p * (ref_lp - cand_lp)).sum()),
        "max_abs_logit_diff": float(np.abs(reference - candidate).max()),
        "rms_logit_diff": float(np.sqrt(((reference - candidate) ** 2).mean())),
        "top1_agree": bool(top_ref[0] == top_cand[0]),
        "top10_overlap": int(len(set(top_ref) & set(top_cand))),
        "reference_top5": top_ref[:5].tolist(),
        "infernix_top5": top_cand[:5].tolist(),
        "reference_top1_prob": float(p[top_ref[0]]),
    }


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def compare_blocks(reference: np.lib.npyio.NpzFile, residuals: Path | None, routes: Path | None) -> list[dict]:
    rows = []
    blocks, tokens = reference["routed"].shape[:2]
    if residuals is not None:
        ref = reference["layer_residuals"].astype(np.float64)
        got = bf16_to_f32(np.fromfile(residuals, dtype=np.uint16)).reshape(ref.shape).astype(np.float64)
        error = np.linalg.norm(got - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
    if routes is not None:
        ref_routes = reference["routed"]
        got_routes = np.fromfile(routes, dtype=np.int32).reshape(ref_routes.shape)
    for b in range(blocks):
        row = {"block": b}
        if residuals is not None:
            row["residual_rel_error"] = [round(float(v), 5) for v in error[b]]
        if routes is not None:
            row["routed_overlap"] = [len(set(ref_routes[b, t].tolist()) & set(got_routes[b, t].tolist()))
                                     for t in range(tokens)]
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--infernix", type=Path, required=True)
    parser.add_argument("--residuals", type=Path)
    parser.add_argument("--routes", type=Path)
    args = parser.parse_args()
    reference = np.load(args.reference)["logits"][args.position]
    candidate = np.fromfile(args.ninfer, dtype=np.float32)
    if candidate.shape != reference.shape:
        raise ValueError(f"shape {candidate.shape} differs from the reference {reference.shape}")
    print(json.dumps(compare(reference, candidate), indent=2))
    if args.residuals is not None or args.routes is not None:
        for row in compare_blocks(np.load(args.reference), args.residuals, args.routes):
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
