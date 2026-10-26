"""Exact words of NVIDIA ModelOpt checkpoints, with NVIDIA's scale semantics kept as stored.

ModelOpt NVFP4 matrices store ``weight`` (packed E2M1, even element in the low nibble),
``weight_scale`` (E4M3FN per 16), ``weight_scale_2`` (an FP32 *multiplier*) and ``input_scale``
(the FP32 activation global scale). Block-FP8 matrices (Qwen's FP8 checkpoints, copied by NVIDIA for
the MTP experts) store ``weight`` (E4M3FN) and ``weight_scale_inv`` (an FP32 multiplier per 128 x 128
tile, despite its name). MXFP8 matrices store ``weight`` (E4M3FN) and ``weight_scale`` (UE8M0 bytes,
one per 32 K elements). Nothing here inverts, rounds or merges a scale.

NVIDIA's Qwen3.8-Flash-Next-FP4-Local stores its routed experts' block scales in Local Inference
Lab's lossless NVFP4-CSF encoding (``weight_scale.nvfp4_csf_fixed`` and
``weight_scale.nvfp4_csf_exceptions``); :func:`csf_scale_words` restores the E4M3FN words.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

from tools.artifact.codecs.fp8_g32 import fp8_group_values, validate_fp8_group_words
from tools.artifact.codecs.nvfp4_expert import encode_nvfp4_expert_bank
from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource

CSF_FIXED_SUFFIX = ".weight_scale.nvfp4_csf_fixed"
CSF_EXCEPTIONS_SUFFIX = ".weight_scale.nvfp4_csf_exceptions"
MXFP8_FORMAT = "fp8_e4m3fn_g32_e8m0"
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                     dtype=torch.float64)


@dataclass(frozen=True)
class Nvfp4MatrixWords:
    codes: torch.Tensor          # uint8 [N, K/2]
    scales: torch.Tensor         # uint8 [N, K/16], E4M3FN words
    weight_scale_2: torch.Tensor  # float32 scalar, the multiplier
    input_scale: torch.Tensor | None  # float32 scalar, the activation global scale (None: weight-only)


def csf_scale_words(store: SafetensorsSource, leaf: str, rows: int, columns: int) -> torch.Tensor:
    """The E4M3FN block-scale words ``uint8 [rows, columns]`` of ``leaf`` stored in NVFP4-CSF (Local
    Inference Lab's lossless scale encoding, the byte-window codec of b12x's ``Nvfp4CsfBatch``).

    ``{leaf}.weight_scale.nvfp4_csf_fixed`` (U8, ``rows / 16`` slabs of ``16 * (1 + columns / 2)``
    bytes, stored 1-D or 2-D) holds per 16-row slab the rows' base bytes (each at most 240), then
    every row's ``columns / 2`` bytes of 4-bit offsets, the even column in the low nibble: a scale
    is its row's base plus its offset. ``{leaf}.weight_scale.nvfp4_csf_exceptions`` (U32) replaces
    the scales the window cannot hold: word ``position | byte << 24`` with ``position = row *
    columns + column``, strictly increasing. Every restored byte is either stored verbatim or a base
    plus an offset, so the decoding is exact; the result is validated like any E4M3FN scale plane.
    """

    if rows % 16 or columns % 2:
        raise ValueError(f"{leaf}: NVFP4-CSF needs 16-row slabs and an even scale count")
    fixed_name, exceptions_name = leaf + CSF_FIXED_SUFFIX, leaf + CSF_EXCEPTIONS_SUFFIX
    slab = 16 * (1 + columns // 2)
    fixed_info = store.describe(fixed_name)
    if fixed_info.dtype != "U8" or prod(fixed_info.shape) != rows // 16 * slab:
        raise ValueError(f"{fixed_name}: expected U8 with {rows // 16} slabs of {slab} bytes, "
                         f"got {fixed_info.dtype}{tuple(fixed_info.shape)}")
    exceptions_info = store.describe(exceptions_name)
    if exceptions_info.dtype != "U32" or len(exceptions_info.shape) != 1:
        raise ValueError(f"{exceptions_name}: expected one U32 vector, got "
                         f"{exceptions_info.dtype}{tuple(exceptions_info.shape)}")
    stream = store.read_flat(fixed_name).reshape(rows // 16, slab)
    bases = stream[:, :16].reshape(rows)
    if bool((bases > 240).any()):
        raise ValueError(f"{fixed_name}: a row base exceeds 240")
    packed = stream[:, 16:].reshape(rows, columns // 2)
    offsets = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, columns)
    scales = (bases.unsqueeze(1) + offsets).contiguous()  # at most 255: no wrap
    if exceptions_info.shape[0]:
        words = torch.from_numpy(store.read_flat(exceptions_name).numpy().astype("int64"))
        positions = words & 0xFFFFFF
        if int(positions[-1]) >= rows * columns or bool((positions[1:] <= positions[:-1]).any()):
            raise ValueError(f"{exceptions_name}: positions must be unique, increasing and inside the matrix")
        scales.view(-1)[positions] = (words >> 24).to(torch.uint8)
    return scales


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
    """Read one ModelOpt NVFP4 matrix ``leaf`` (for example ``...experts.7.gate_proj``) exactly. Its
    block scales are ``weight_scale`` or, in an NVFP4-CSF checkpoint, the CSF planes
    (:func:`csf_scale_words`); a weight-only (W4A16) matrix has no ``input_scale`` (None)."""

    n, k = shape
    if k % 16:
        raise ValueError(f"{leaf}: NVFP4 K must be divisible by 16")
    _signature(store, leaf + ".weight", (n, k // 2), "U8")
    codes = store.read_flat(leaf + ".weight").reshape(n, k // 2).clone()
    if store.has(leaf + CSF_FIXED_SUFFIX):
        scales = csf_scale_words(store, leaf, n, k // 16)
    else:
        _signature(store, leaf + ".weight_scale", (n, k // 16), "F8_E4M3")
        scales = store.read_flat(leaf + ".weight_scale").view(torch.uint8).reshape(n, k // 16).clone()
    if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
        raise ValueError(f"{leaf}.weight_scale: expected nonnegative finite E4M3FN words")
    return Nvfp4MatrixWords(
        codes=codes,
        scales=scales,
        weight_scale_2=_fp32_scalar(store, leaf + ".weight_scale_2"),
        input_scale=_fp32_scalar(store, leaf + ".input_scale") if store.has(leaf + ".input_scale") else None,
    )


def nvfp4_values(words: Nvfp4MatrixWords) -> torch.Tensor:
    """The exact represented weights of ModelOpt NVFP4 words in binary64:
    ``e2m1(code) * e4m3fn(scale) * weight_scale_2`` (at most 6 + 24 significant bits)."""

    n = words.codes.shape[0]
    codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(n, -1)
    scales = words.scales.view(torch.float8_e4m3fn).to(torch.float64).repeat_interleave(16, dim=1)
    return _E2M1[codes.long()] * scales * float(words.weight_scale_2)


NVFP4_MUL_FORMAT = "nvfp4_mul"


def nvfp4_mul_matrix_source(store: SafetensorsSource, leaf: str, shape: tuple[int, int]) -> LogicalSource:
    """One ModelOpt NVFP4 matrix ``leaf`` with NVIDIA's multiplier semantics (a weight-only W4A16
    tensor: the vision MLPs and merger and the MTP experts of Qwen3.8-Flash-Next-FP4-Local): its exact
    values (binary64), for a recipe that re-quantizes it, and its stored words as ``nvfp4_mul``
    encoded rows (packed E2M1 codes ``uint8 [rows, K / 2]``, E4M3FN block scales ``uint8
    [rows, K / 16]``, plain or restored from NVFP4-CSF, and the FP32 ``weight_scale_2`` word as the
    multiplier), for an exact import. The generic NVFP4 reader divides by the multiplier's rounded
    reciprocal instead."""

    n, k = shape

    def encoded(begin: int, end: int) -> EncodedRows:
        # Each read decodes the whole matrix's words and keeps the rows it covers (no per-matrix
        # cache: a conversion reads thousands of these matrices).
        if not 0 <= begin < end <= n:
            raise ValueError(f"{leaf}: invalid encoded rows [{begin},{end})")
        words = nvfp4_matrix_words(store, leaf, shape)
        multiplier = words.weight_scale_2.reshape(1).view(torch.uint8).numpy().tobytes()
        return EncodedRows(NVFP4_MUL_FORMAT, words.codes[begin:end].contiguous(), words.scales[begin:end].contiguous(),
                           None, multiplier)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float64)
        first, last = begin // k, -(-end // k)
        words = nvfp4_matrix_words(store, leaf, shape)
        rows = Nvfp4MatrixWords(words.codes[first:last], words.scales[first:last], words.weight_scale_2, None)
        return nvfp4_values(rows).reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{leaf} ({NVFP4_MUL_FORMAT})", read, encoded)


def is_mxfp8_matrix(store: SafetensorsSource, prefix: str, shape: tuple[int, int]) -> bool:
    """Whether ``prefix`` is a ModelOpt MXFP8 matrix: E4M3FN ``weight`` and one UE8M0 byte of
    ``weight_scale`` per 32 K elements."""

    n, k = shape
    if k % 32 or not store.has(prefix + ".weight") or not store.has(prefix + ".weight_scale"):
        return False
    weight, scale = store.describe(prefix + ".weight"), store.describe(prefix + ".weight_scale")
    return weight.dtype == "F8_E4M3" and scale.dtype == "U8" and tuple(scale.shape) == (n, k // 32)


def mxfp8_matrix_source(store: SafetensorsSource, prefix: str, shape: tuple[int, int]) -> LogicalSource:
    """One ModelOpt MXFP8 matrix: its exact values (binary64) and its stored words as
    ``fp8_e4m3fn_g32_e8m0`` encoded rows (codes ``uint8 [rows, K]``, UE8M0 ``uint8 [rows, K / 32]``)."""

    n, k = shape
    groups = k // 32
    if not is_mxfp8_matrix(store, prefix, shape):
        raise ValueError(f"{prefix}: expected an MXFP8 matrix of shape {shape}")

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        codes = store.read_flat(prefix + ".weight", begin * k, end * k).view(torch.uint8).reshape(end - begin, k)
        scales = store.read_flat(prefix + ".weight_scale", begin * groups, end * groups).reshape(end - begin, groups)
        validate_fp8_group_words(codes, scales)
        return EncodedRows(MXFP8_FORMAT, codes, scales)

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float64)
        first, last = begin // k, -(-end // k)
        words = encoded(first, last)
        return fp8_group_values(words.codes, words.scales).reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{prefix} ({MXFP8_FORMAT})", read, encoded)


def nvfp4_packed_matrix_words(store: SafetensorsSource, leaf: str, shape: tuple[int, int]) -> Nvfp4MatrixWords:
    """Read one weight-only compressed-tensors NVFP4 matrix ``leaf`` exactly: ``weight_packed`` codes
    (two per byte, the first element in the low nibble, as ModelOpt packs them) and ``weight_scale``
    E4M3FN block scales, unchanged. The format's ``weight_global_scale`` divides (value = code x
    block scale / global scale); the expert bank stores a multiplier, so weight_scale_2 is its
    reciprocal rounded once to FP32. There is no activation scale (design §16.2.1)."""

    n, k = shape
    if k % 16:
        raise ValueError(f"{leaf}: NVFP4 K must be divisible by 16")
    _signature(store, leaf + ".weight_packed", (n, k // 2), "U8")
    _signature(store, leaf + ".weight_scale", (n, k // 16), "F8_E4M3")
    if store.has(leaf + ".input_global_scale"):
        raise ValueError(f"{leaf}: an activation scale belongs to a W4A4 export, not a weight-only one")
    codes = store.read_flat(leaf + ".weight_packed").reshape(n, k // 2).clone()
    scales = store.read_flat(leaf + ".weight_scale").view(torch.uint8).reshape(n, k // 16).clone()
    if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
        raise ValueError(f"{leaf}.weight_scale: expected nonnegative finite E4M3FN words")
    divisor = _fp32_scalar(store, leaf + ".weight_global_scale")
    multiplier = torch.ones((), dtype=torch.float32) / divisor  # one IEEE FP32 division
    if not bool(torch.isfinite(multiplier)) or float(multiplier) <= 0:
        raise ValueError(f"{leaf}.weight_global_scale: its reciprocal is not a positive finite FP32")
    return Nvfp4MatrixWords(codes=codes, scales=scales, weight_scale_2=multiplier, input_scale=None)


def fp8_block_matrix_words(
    store: SafetensorsSource, leaf: str, shape: tuple[int, int], block: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one block-FP8 matrix: E4M3FN codes ``[N, K]`` and FP32 tile multipliers.

    Qwen/Qwen3.8-Flash-Next-FP8 and NVIDIA's MTP experts store the multipliers in BF16 (other
    checkpoints may use FP32); a BF16 multiplier widens to FP32 exactly.
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


def fp8_block_matrix_raw(store: SafetensorsSource, leaf: str, shape: tuple[int, int], block: int = 128):
    """One block-FP8 matrix as stored: E4M3FN code bytes ``[N, K]`` (numpy uint8) and its BF16 tile
    scale words ``[N/block, K/block]`` (numpy uint16), for a verbatim import (``fp8_e4m3fn_b128``)."""

    n, k = shape
    if n % block or k % block:
        raise ValueError(f"{leaf}: {shape} is not whole {block} x {block} tiles")
    _signature(store, leaf + ".weight", (n, k), "F8_E4M3")
    scale = leaf + ".weight_scale_inv"
    info = store.describe(scale)
    if tuple(info.shape) != (n // block, k // block) or info.dtype != "BF16":
        raise ValueError(f"{scale}: fp8_e4m3fn_b128 stores BF16{(n // block, k // block)} scales verbatim, "
                         f"got {info.dtype}{tuple(info.shape)}")
    codes = store.read_flat(leaf + ".weight").view(torch.uint8).reshape(n, k).numpy().copy()
    words = store.read_flat(scale).reshape(n // block, k // block).view(torch.int16).numpy().view("<u2").copy()
    return codes, words


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


def dequantize_fp8_block128(codes: torch.Tensor, scales: torch.Tensor, block: int = 128) -> torch.Tensor:
    """Exact represented weights in binary64 (4 + 24 significant bits)."""

    n, k = codes.shape[-2:]
    expanded = scales.to(torch.float64).repeat_interleave(block, dim=-2)[..., :n, :]
    expanded = expanded.repeat_interleave(block, dim=-1)[..., :k]
    return codes.view(torch.float8_e4m3fn).to(torch.float64) * expanded


def is_fp8_block_matrix(store: SafetensorsSource, prefix: str) -> bool:
    """``prefix.weight`` is E4M3FN with a ``weight_scale_inv`` tile multiplier per 128 x 128."""

    return (
        store.has(prefix + ".weight_scale_inv")
        and store.has(prefix + ".weight")
        and store.describe(prefix + ".weight").dtype == "F8_E4M3"
    )


def fp8_block_values(store: SafetensorsSource, leaf: str, shape: tuple[int, int]) -> LogicalSource:
    """The exact values of one block-FP8 matrix ``leaf`` (code x FP32 tile multiplier, binary64).

    NVIDIA stores Qwen3.8-Flash-Next's MTP routed experts this way, per expert, and so does ModelOpt's
    ``FP8_PB_WO`` (dealignai's Ornith-1.5-122B-UNCENSORED MTP experts). Each read decodes only the rows
    it covers, so a recipe's row-chunked quantizer never holds more than one matrix's codes.
    """

    n, k = shape

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float64)
        first, last = begin // k, -(-end // k)
        codes, scales = fp8_block_matrix_words(store, leaf, shape)
        lo, hi = first // 128 * 128, -(-last // 128) * 128
        rows = dequantize_fp8_block128(codes[lo:min(hi, n)], scales[lo // 128 : -(-min(hi, n) // 128)])
        flat = rows.reshape(-1)
        start = begin - lo * k
        return flat[start : start + end - begin]

    return LogicalSource(shape, f"{store.path}:{leaf}", read)
