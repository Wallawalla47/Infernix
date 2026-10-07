"""Exact words of NVIDIA ModelOpt checkpoints, with NVIDIA's scale semantics kept as stored.

ModelOpt NVFP4 matrices store ``weight`` (packed E2M1, even element in the low nibble),
``weight_scale`` (E4M3FN per 16), ``weight_scale_2`` (an FP32 *multiplier*) and ``input_scale``
(the FP32 activation global scale). Block-FP8 matrices (Qwen's FP8 checkpoints, copied by NVIDIA for
the MTP experts) store ``weight`` (E4M3FN) and ``weight_scale_inv`` (an FP32 multiplier per 128 x 128
tile, despite its name). Nothing here inverts, rounds or merges a scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from tools.artifact.codecs.nvfp4_expert import encode_nvfp4_expert_bank
from tools.artifact.formats import valid_positive_fp32_word
from .safetensors import SafetensorsSource


@dataclass(frozen=True)
class Nvfp4MatrixWords:
    codes: torch.Tensor          # uint8 [N, K/2]
    scales: torch.Tensor         # uint8 [N, K/16], E4M3FN words
    weight_scale_2: torch.Tensor  # float32 scalar, the multiplier
    input_scale: torch.Tensor    # float32 scalar, the activation global scale


def _signature(store: SafetensorsSource, name: str, shape: tuple[int, ...], dtype: str) -> None:
    info = store.describe(name)
    if tuple(info.shape) != shape or info.dtype != dtype:
        raise ValueError(f"{name}: expected {dtype}{shape}, got {info.dtype}{tuple(info.shape)}")


def _fp32_scalar(store: SafetensorsSource, name: str) -> torch.Tensor:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected one FP32 scalar")
    value = store.read_flat(name).reshape(()).clone()
    word = int(value.view(torch.int32).item()) & 0xFFFFFFFF
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: must be finite and positive")
    return value


def nvfp4_matrix_words(store: SafetensorsSource, leaf: str, shape: tuple[int, int]) -> Nvfp4MatrixWords:
    """Read one ModelOpt NVFP4 matrix ``leaf`` (for example ``...experts.7.gate_proj``) exactly."""

    n, k = shape
    if k % 16:
        raise ValueError(f"{leaf}: NVFP4 K must be divisible by 16")
    _signature(store, leaf + ".weight", (n, k // 2), "U8")
    _signature(store, leaf + ".weight_scale", (n, k // 16), "F8_E4M3")
    codes = store.read_flat(leaf + ".weight").reshape(n, k // 2).clone()
    scales = store.read_flat(leaf + ".weight_scale").view(torch.uint8).reshape(n, k // 16).clone()
    if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
        raise ValueError(f"{leaf}.weight_scale: expected nonnegative finite E4M3FN words")
    return Nvfp4MatrixWords(
        codes=codes,
        scales=scales,
        weight_scale_2=_fp32_scalar(store, leaf + ".weight_scale_2"),
        input_scale=_fp32_scalar(store, leaf + ".input_scale"),
    )


def fp8_block_matrix_words(
    store: SafetensorsSource, leaf: str, shape: tuple[int, int], block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one block-FP8 matrix: E4M3FN codes ``[N, K]`` and FP32 tile multipliers.

    Qwen's FP8 checkpoints store the multipliers in FP32 and NVIDIA's MTP experts in BF16; a BF16
    multiplier widens to FP32 exactly.
    """

    n, k = shape
    rows, cols = -(-n // block), -(-k // block)
    _signature(store, leaf + ".weight", (n, k), "F8_E4M3")
    scale = leaf + ".weight_scale_inv"
    info = store.describe(scale)
    if tuple(info.shape) != (rows, cols) or info.dtype not in ("F32", "BF16"):
        raise ValueError(f"{scale}: expected F32 or BF16{(rows, cols)}, got {info.dtype}{tuple(info.shape)}")
    codes = store.read_flat(leaf + ".weight").view(torch.uint8).reshape(n, k).clone()
    scales = store.read_flat(scale).reshape(rows, cols).to(torch.float32).clone()
    return codes, scales


@dataclass(frozen=True)
class ExpertBank:
    payload: bytes               # nvfp4_expert_rg16_v1 bytes for [E, H, I]
    input_scales: torch.Tensor   # float32 [E, 3]: gate, up, down activation global scales
    shared_gate_up_input: bool   # every expert's gate and up input scales are equal


def nvfp4_expert_bank(
    store: SafetensorsSource, experts_prefix: str, experts: int, hidden: int, intermediate: int
) -> ExpertBank:
    """Pack ``{experts_prefix}{e}.{gate,up,down}_proj`` of one layer into one bank, exactly."""

    gc, gs, uc, us, dc, ds, mult, inp = [], [], [], [], [], [], [], []
    for e in range(experts):
        gate = nvfp4_matrix_words(store, f"{experts_prefix}{e}.gate_proj", (intermediate, hidden))
        up = nvfp4_matrix_words(store, f"{experts_prefix}{e}.up_proj", (intermediate, hidden))
        down = nvfp4_matrix_words(store, f"{experts_prefix}{e}.down_proj", (hidden, intermediate))
        gc.append(gate.codes)
        gs.append(gate.scales)
        uc.append(up.codes)
        us.append(up.scales)
        dc.append(down.codes)
        ds.append(down.scales)
        mult.append(torch.stack((gate.weight_scale_2, up.weight_scale_2, down.weight_scale_2)))
        inp.append(torch.stack((gate.input_scale, up.input_scale, down.input_scale)))
    multipliers = torch.stack(mult)
    input_scales = torch.stack(inp)
    payload = encode_nvfp4_expert_bank(
        torch.stack(gc), torch.stack(gs), torch.stack(uc), torch.stack(us),
        torch.stack(dc), torch.stack(ds), multipliers, (experts, hidden, intermediate),
    )
    shared = bool(torch.equal(input_scales[:, 0].view(torch.int32), input_scales[:, 1].view(torch.int32)))
    return ExpertBank(payload=payload, input_scales=input_scales, shared_gate_up_input=shared)
