"""The A4 activation quantizer of the Flash-Next design (§16.2), checked against ModelOpt's.

    python3 -m tools.flash_next.a4_reference [--activations acts.pt ...] [--json report.json]

``a4_quantize`` is an exact port of ``canon::quantize_a4_block`` (src/ops/common/canonical_math.h).
``modelopt_rule`` ports ModelOpt's FP4 fake quantizer, the one that calibrated NVIDIA's
checkpoint (``fp4_fake_quant_kernel`` with ``fp8_quantize_scale`` and ``fp4_round_magnitude``).
Both are evaluated in IEEE FP32 with round-to-nearest-even.

The two rules agree except for one guard in ModelOpt: a dequantized block scale below 1e-5 is
replaced by 1.0, so a block whose amax is below about 6e-5 quantizes to zeros. TensorRT-LLM's
deployed FP4 quantizer has no such guard, and neither does §16.2; the report counts the blocks it
affects separately from every other mismatch.

With CUDA and ``nvidia-modelopt`` installed, the CLI also launches ModelOpt's own Triton kernel
and compares it with both ports. That is the M0 check of design §19. The Triton interpreter is not
a substitute: its FP32-to-FP8 cast does not round to nearest even.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

E4M3_MAX = 448.0
MODELOPT_SCALE_FLOOR = 1e-5


def _e4m3_rn_satfinite(x: torch.Tensor) -> torch.Tensor:
    """FP32 (nonnegative, finite) to E4M3FN words, round to nearest even, saturating at 448."""

    return torch.clamp(x, max=E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def _e4m3_value(words: torch.Tensor) -> torch.Tensor:
    return words.view(torch.float8_e4m3fn).to(torch.float32)


def e2m1_round_magnitude(a: torch.Tensor) -> torch.Tensor:
    """|x| / d to the E2M1 magnitude grid, ties to even, saturating at 6 (as FP32 values)."""

    out = torch.full_like(a, 6.0)
    for bound, value, inclusive in (
        (5.0, 4.0, True), (3.5, 3.0, False), (2.5, 2.0, True), (1.75, 1.5, False),
        (1.25, 1.0, True), (0.75, 0.5, False), (0.25, 0.0, True),
    ):
        out = torch.where(a <= bound if inclusive else a < bound, torch.tensor(value), out)
    return out


def _blocks(v: torch.Tensor) -> torch.Tensor:
    if v.shape[-1] % 16:
        raise ValueError("A4 vectors must have a multiple of 16 elements")
    return v.to(torch.float32).reshape(*v.shape[:-1], v.shape[-1] // 16, 16)


def a4_quantize(v: torch.Tensor, input_scale: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Design §16.2: returns doubled E2M1 values ``c2`` (int8, [..., K]) and E4M3FN scale words."""

    if v.dtype != torch.bfloat16:
        raise ValueError("A4 quantizes BF16 vectors")
    g = torch.as_tensor(input_scale, dtype=torch.float32)
    x = _blocks(v)
    amax = x.abs().amax(dim=-1)
    s = _e4m3_rn_satfinite(amax / (g * 6.0))
    d = (_e4m3_value(s) * g).unsqueeze(-1)
    zero = (s == 0).unsqueeze(-1)
    q = e2m1_round_magnitude((x / torch.where(zero, torch.ones_like(d), d)).abs())
    c2 = torch.where(x < 0, -2.0 * q, 2.0 * q)
    c2 = torch.where(zero, torch.zeros_like(c2), c2).to(torch.int8)
    return c2.reshape(v.shape), s


def a4_dequantize(c2: torch.Tensor, scale_words: torch.Tensor, input_scale: float | torch.Tensor) -> torch.Tensor:
    """FP32 value of each A4 element: (c2 / 2) * fl(e4m3(s) * g), as ModelOpt's fake quant forms it."""

    g = torch.as_tensor(input_scale, dtype=torch.float32)
    d = (_e4m3_value(scale_words) * g).unsqueeze(-1)
    q = c2.to(torch.float32).reshape(*c2.shape[:-1], c2.shape[-1] // 16, 16) * 0.5
    return (q * d).reshape(c2.shape)


def modelopt_rule(v: torch.Tensor, input_scale: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ModelOpt's FP4 fake quantization of ``v`` with global scale ``input_scale``.

    Returns the FP32 fake-quantized values and a per-block mask of blocks where the 1e-5 guard fired.
    """

    g = torch.as_tensor(input_scale, dtype=torch.float32)
    x = _blocks(v)
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = _e4m3_value(_e4m3_rn_satfinite(amax / (g * 6.0))) * g
    guard = ~(scale >= MODELOPT_SCALE_FLOOR)
    scale = torch.where(guard, torch.ones_like(scale), scale)
    q = e2m1_round_magnitude(x.abs() / scale) * scale
    y = torch.where(x >= 0, q, -q)
    return y.reshape(v.shape), guard.squeeze(-1)


def modelopt_kernel(v: torch.Tensor, input_scale: float) -> torch.Tensor:
    """Launch ModelOpt's own Triton FP4 fake-quant kernel on CUDA with global scale ``input_scale``."""

    from modelopt.torch.kernels.quantization.gemm import fp4_kernel_hopper as k  # noqa: PLC0415

    x = v.to(torch.float32).reshape(-1, v.shape[-1]).contiguous().cuda()
    y = torch.empty_like(x)
    m, n = x.shape
    tile_m, tile_n = 16, 64
    g = torch.tensor([input_scale], dtype=torch.float32, device="cuda")
    grid = ((m + tile_m - 1) // tile_m, (n + tile_n - 1) // tile_n)
    k.fp4_fake_quant_kernel[grid](
        x, y, m, n, g, x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        BLOCK_SIZE=16, TILE_M=tile_m, TILE_N=tile_n, NUM_FP4_BLOCKS=tile_n // 16,
        OUT_DTYPE=k.tl.float32,
    )
    return y.cpu().reshape(v.shape)


# ---------------------------------------------------------------------------- test vectors


def synthetic_activations(seed: int = 0) -> list[tuple[str, torch.Tensor, float]]:
    """Named BF16 sets with input scales: typical, heavy-tailed, tiny, and rounding-boundary values."""

    gen = torch.Generator().manual_seed(seed)
    sets = []
    for g in (2.0**-8, 3.1e-3, 0.0117, 0.05):
        amax = g * 6 * 448
        normal = torch.randn(512, 2560, generator=gen) * (amax / 8)
        sets.append((f"normal g={g:.4g}", normal.to(torch.bfloat16), g))
        heavy = torch.randn(512, 2560, generator=gen) ** 3 * (amax / 40)
        sets.append((f"heavy-tailed g={g:.4g}", heavy.to(torch.bfloat16), g))
        tiny = torch.randn(64, 2560, generator=gen) * 2e-5
        sets.append((f"tiny g={g:.4g}", tiny.to(torch.bfloat16), g))
        # Blocks whose elements sit on E2M1 rounding boundaries of the block's own scale.
        boundary = torch.randn(256, 2560, generator=gen).abs() * (amax / 4) + amax / 64
        x = _blocks(boundary.to(torch.bfloat16))
        _, s = a4_quantize(boundary.to(torch.bfloat16), g)
        d = (_e4m3_value(s) * g).unsqueeze(-1)
        ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        pick = ties[torch.randint(0, 7, x.shape, generator=gen)]
        sign = torch.where(torch.rand(x.shape, generator=gen) < 0.5, -1.0, 1.0)
        on_tie = (pick * d * sign).to(torch.bfloat16).to(torch.float32)
        keep_amax = x.abs() == x.abs().amax(dim=-1, keepdim=True)
        sets.append((f"boundary g={g:.4g}", torch.where(keep_amax, x, on_tie).reshape(boundary.shape).to(torch.bfloat16), g))
    return sets


def compare(v: torch.Tensor, g: float, reference: torch.Tensor | None = None) -> dict:
    """Mismatch counts of §16.2 against ModelOpt (its kernel output if given, else its rule)."""

    c2, s = a4_quantize(v, g)
    ours = a4_dequantize(c2, s, g)
    rule, guard = modelopt_rule(v, g)
    theirs = rule if reference is None else reference
    differ = (ours.view(torch.int32) != theirs.to(torch.float32).view(torch.int32)) & ~(
        (ours == 0) & (theirs == 0)
    )
    blocks = differ.reshape(*differ.shape[:-1], -1, 16).any(dim=-1)
    report = {
        "elements": v.numel(),
        "blocks": blocks.numel(),
        "guard_blocks": int(guard.sum()),
        "mismatching_elements": int(differ.sum()),
        "mismatching_blocks": int(blocks.sum()),
        "mismatching_blocks_outside_guard": int((blocks & ~guard).sum()),
    }
    if reference is not None:
        rule_differ = (rule.view(torch.int32) != reference.to(torch.float32).view(torch.int32)) & ~(
            (rule == 0) & (reference == 0)
        )
        report["modelopt_rule_vs_kernel_mismatching_elements"] = int(rule_differ.sum())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--activations", type=Path, nargs="*", default=[],
                        help="torch.save files of {'x': BF16 [..., K], 'input_scale': float} recorded from real prompts")
    parser.add_argument("--no-kernel", action="store_true", help="compare with the ported rule even if CUDA is present")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    use_kernel = not args.no_kernel and torch.cuda.is_available()
    if use_kernel:
        try:
            import modelopt  # noqa: F401, PLC0415
        except ImportError:
            print("nvidia-modelopt is not installed; comparing with the ported rule only", file=sys.stderr)
            use_kernel = False
    sets = synthetic_activations()
    for path in args.activations:
        record = torch.load(path)
        sets.append((str(path), record["x"].to(torch.bfloat16), float(record["input_scale"])))
    results = []
    for name, v, g in sets:
        reference = modelopt_kernel(v, g) if use_kernel else None
        r = {"set": name, "input_scale": g, "reference": "modelopt kernel" if use_kernel else "modelopt rule port",
             **compare(v, g, reference)}
        results.append(r)
        print(f"{name:32} blocks {r['blocks']:>8}  guard {r['guard_blocks']:>6}  mismatching blocks "
              f"{r['mismatching_blocks']:>6} (outside guard {r['mismatching_blocks_outside_guard']})"
              + (f"  rule-vs-kernel elements {r['modelopt_rule_vs_kernel_mismatching_elements']}" if use_kernel else ""))
    if args.json:
        args.json.write_text(json.dumps(results, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
