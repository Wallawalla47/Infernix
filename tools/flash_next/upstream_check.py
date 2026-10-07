"""Compare Infernix's Qwen4Exp op-chain error with upstream Transformers' on identical inputs.

    python -m tools.flash_next.upstream_check --model CHECKPOINT --blocks blocks.bin \\
        --residuals residuals.bin [--layers 0,3,7]

Needs a Transformers build that has ``qwen4_exp`` (main as of 2026-10). For each selected layer it
builds the upstream modules (both hyper-connection mixers, GatedDeltaNet or QSA attention) in
BF16 with the checkpoint's weights, runs them on Infernix's own tapped inputs (see block_check.py)
and reports, per position, the relative error against the FP64 reference of

- ``upstream``: Transformers in BF16, the precision of the major implementations;
- ``infernix``: Infernix's tapped output.

The PLE layer's attention-side mixer is skipped: its input includes the PLE injection, which needs
the n-gram table. Routed experts are NVFP4 and have no upstream BF16 module; block_check.py covers
the MoE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen4_exp import modeling_qwen4_exp as upstream
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from tools.flash_next.block_check import bf16_file, relative
from tools.flash_next.reference import F64, Reference


class Checkpoint:
    def __init__(self, root: Path):
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.root, self.index = root, index

    def load(self, module: torch.nn.Module, prefix: str) -> torch.nn.Module:
        state = {}
        for key in module.state_dict():
            name = prefix + key
            with safe_open(self.root / self.index[name], framework="pt") as f:
                state[key] = f.get_tensor(name).to(torch.bfloat16)
        module.load_state_dict(state)
        return module.to(torch.bfloat16).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, required=True)
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True, help="number of positions in the taps")
    parser.add_argument("--layers", help="comma-separated layer indices (default: all)")
    args = parser.parse_args()
    ref = Reference(args.model)
    raw = json.loads((args.model / "config.json").read_text())["text_config"]
    config = Qwen4ExpTextConfig(**raw)
    config._attn_implementation = "eager"
    ckpt = Checkpoint(args.model)
    L, H, T = ref.c["num_hidden_layers"], ref.H, args.tokens
    taps = bf16_file(args.blocks, (L, 4, T, H))
    residuals = bf16_file(args.residuals, (L, T, ref.hc * H))
    positions = torch.arange(T)
    rotary = upstream.Qwen4ExpTextRotaryEmbedding(config)
    mask = torch.full((1, 1, T, T), float("-inf")).triu(1).to(torch.bfloat16)
    prefix = "model.language_model.layers."
    layers = [int(v) for v in args.layers.split(",")] if args.layers else range(L)

    def bf(x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16)[None]

    with torch.no_grad():
        for layer in layers:
            x, y, xm, _ = taps[layer]
            row = {"layer": layer, "mixer": ref.layer_types[layer]}
            # Attention-side mixer, from the residual entering the block.
            if layer + 1 not in ref.ple_layers:
                R = residuals[layer - 1] if layer else None
                if R is not None:
                    hc = ckpt.load(upstream.Qwen4ExpTextGatedResidual(config), f"{prefix}{layer}.attn_hyper_connection.")
                    up, _, _ = hc(bf(R))
                    want, _ = ref.hc_mix(f"layers.{layer}.attn_hyper_connection.", R)
                    row["attn_mix"] = {"upstream": relative(up[0].to(F64), want), "infernix": relative(x, want)}
            # The mixer.
            if ref.layer_types[layer] == "linear":
                module = ckpt.load(upstream.Qwen4ExpTextGatedDeltaNet(config, layer), f"{prefix}{layer}.linear_attn.")
                up = module(bf(x))
                want = ref.gdn(layer, x)
            else:
                module = ckpt.load(upstream.Qwen4ExpTextAttention(config, layer), f"{prefix}{layer}.self_attn.")
                cos, sin = rotary(bf(x), positions[None, None].expand(3, 1, T))
                up, _ = module(bf(x), (cos.to(torch.bfloat16), sin.to(torch.bfloat16)), mask)
                want = ref.qsa(layer, x, positions)
            row["mixer_error"] = {"upstream": relative(up[0].to(F64), want), "infernix": relative(y, want)}
            # MLP-side mixer: inject Infernix's mixer output into the reference's view of the residual.
            if layer + 1 not in ref.ple_layers and layer:
                R = residuals[layer - 1]
                _, inject = ref.hc_mix(f"layers.{layer}.attn_hyper_connection.", R)
                Rm = ref.inject(R, y, inject)
                hc = ckpt.load(upstream.Qwen4ExpTextGatedResidual(config), f"{prefix}{layer}.mlp_hyper_connection.")
                up, _, _ = hc(bf(Rm))
                want, _ = ref.hc_mix(f"layers.{layer}.mlp_hyper_connection.", Rm)
                row["mlp_mix"] = {"upstream": relative(up[0].to(F64), want), "infernix": relative(xm, want)}
            print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
