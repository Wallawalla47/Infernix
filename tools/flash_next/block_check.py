"""Check each Infernix Qwen4Exp block's op chains against the FP64 reference on identical inputs.

    python -m tools.flash_next.block_check --model CHECKPOINT --tokens 1,2,3 --blocks blocks.bin \\
        --residuals residuals.bin [--layers 0,1,3] [--a16-experts]

``blocks.bin`` and ``residuals.bin`` are the ``--blocks`` and ``--residuals`` taps of
infernix_qwen4_exp_forward_real_test: BF16 ``[layers][mixer in, mixer out, MoE in, MoE out][T][H]``
and BF16 ``[layers][T][S*H]`` (the residual after each block). For every selected layer the
reference recomputes, from Infernix's own inputs:

- ``attn_mix``: the PLE injection (its layer only) and the attention-side hyper-connection mixer,
  from the residual entering the block;
- ``mixer``: GDN or QSA from Infernix's mixer input;
- ``mlp_mix``: the injection of Infernix's mixer output and the MLP-side mixer;
- ``moe``: the MoE from Infernix's MoE input;
- ``residual``: the injection of Infernix's MoE output,

so each reported error belongs to that op chain alone rather than to accumulated drift.

``--reference-points`` and ``--a16-experts`` evaluate as reference.py does with those options (the
composed kernels' BF16 points; 16-bit expert activations). ``--skip-ple`` omits the PLE injection's
layer from ``attn_mix`` and ``residual`` (for an artifact whose n-gram rows are not the checkpoint's,
such as the FP8 import bound to the BF16 volume); its mixer and MoE are still checked. ``--summary``
prints, per op chain, the mean and largest relative error over every checked layer and position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.flash_next.reference import F64, Reference


def bf16_file(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(raw.view(np.float32).reshape(shape)).to(F64)


def relative(got: torch.Tensor, ref: torch.Tensor) -> list[float]:
    error = (got - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)
    return [round(float(v), 5) for v in error]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--layers", help="comma-separated layer indices (default: all)")
    parser.add_argument("--reference-points", action="store_true")
    parser.add_argument("--a16-experts", action="store_true")
    parser.add_argument("--skip-ple", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    ids = [int(v) for v in args.tokens.split(",") if v]
    ref = Reference(args.model, reference_points=args.reference_points, a16_experts=args.a16_experts)
    layers_total, hidden, T = ref.c["num_hidden_layers"], ref.H, len(ids)
    taps = bf16_file(args.blocks, (layers_total, 4, T, hidden))
    residuals = bf16_file(args.residuals, (layers_total, T, ref.hc * hidden))
    name = ref.w.prefix + "embed_tokens.weight"
    embedded = torch.stack([ref.w.store.read_flat(name, i * hidden, (i + 1) * hidden) for i in ids]).to(F64)
    layers = [int(v) for v in args.layers.split(",")] if args.layers else range(layers_total)
    positions = torch.arange(T)
    chains: dict[str, list[float]] = {}
    for layer in layers:
        x, y, xm, ym = taps[layer]
        R = embedded.repeat(1, ref.hc) if layer == 0 else residuals[layer - 1]
        ple = layer + 1 in ref.ple_layers
        if ple and not args.skip_ple:
            out, _ = ref.ple(layer, ref.ple_layers.index(layer + 1), R, ids)
            R = ref.round(R + out)
        x_ref, inject = ref.hc_mix(f"layers.{layer}.attn_hyper_connection.", R)
        if ref.layer_types[layer] == "linear":
            mixer = ref.gdn(layer, x)
        else:
            mixer = ref.qsa(layer, x, positions)
        R = ref.inject(R, y, inject)
        xm_ref, inject = ref.hc_mix(f"layers.{layer}.mlp_hyper_connection.", R)
        moe, _ = ref.moe(layer, xm)
        R = ref.inject(R, ym, inject)
        row = {"layer": layer, "mixer": ref.layer_types[layer], "mixer_rel_error": relative(y, mixer),
               "mlp_mix": relative(xm, xm_ref), "moe_rel_error": relative(ym, moe)}
        if not (ple and args.skip_ple):
            row["attn_mix"] = relative(x, x_ref)
            row["residual"] = relative(residuals[layer], R)
        for key, values in row.items():
            if isinstance(values, list):
                chains.setdefault(key if key != "mixer_rel_error" else f"{ref.layer_types[layer]}_rel_error", []).extend(values)
        if not args.summary:
            print(json.dumps(row))
    if args.summary:
        print(json.dumps({key: {"mean": round(sum(v) / len(v), 6), "max": round(max(v), 6), "count": len(v)}
                          for key, v in chains.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
