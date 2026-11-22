"""Inkling2Loop (Thinking Machines Inkling 2 Mini) architecture adapter: mathematical config, logical parameters,
packing groups and the exact expert banks.

The mathematics is the upstream ``inkling2_loop`` model (docs/maintainer/inkling2_mini-design.md §2): 24 layers whose
core 6-17 runs three passes with loop message gates, sliding-window and global attention with per-token relative bias,
sinks, L2-clipped queries and keys and short convolutions, Block Attention Residuals, and a 128-expert MoE whose
experts are NVFP4 with four-over-six activations (layers 0-19) or MXFP8 (layers 20-23). The DSpark drafter comes from
``mtp.safetensors``. Every stored tensor keeps its words; the only transforms are value-preserving: rows past the real
vocabulary are dropped, the static convolutions become lag-major ``[k, C]`` (row 0 the current token), the cross-pass
message queries are regrouped per loop layer in (query, key, value, mlp) order, the drafter's ``Fold`` and
``fold_bias`` are derived as the references derive them at load, and the vision tower's class row is the one BF16
addition ``class_embedding + cls_position_embedding`` both references make. Recipes live in official_recipes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Mapping

import torch

from tools.artifact.codecs.mxfp8_expert import encode_mxfp8_expert_records
from tools.artifact.codecs.nvfp4_expert import encode_nvfp4_expert_bank
from tools.artifact.layouts import expert_bank_geometry, mxfp8_expert_geometry

from .methods import PrepareRequest, PreparedMethod
from .model import Model, Parameter
from .resources import load_resources
from .sources.logical import LogicalSource, select_rows
from .sources.safetensors import SafetensorsSource, tensor_source

ARCHITECTURE = "Inkling2LoopForConditionalGeneration"
TEXT_ARCHITECTURE = "Inkling2LoopForCausalLM"
DSPARK_ARCHITECTURE = "Inkling2DSpark"
VISION_ARCHITECTURE = "Inkling2Vision"
PREFIX = "model.llm."
VISUAL = "model.visual."
IMAGE_TOKEN = "<|unused_200054|>"         # one per 40-pixel cell (Inkling2Processor.image_token)
IMAGE_START_TOKEN = "<|content_image|>"  # the chat template's image part marker
PROCESSOR = "processor_config.json"
MESSAGE_KINDS = (("query", "attn"), ("key", "k"), ("value", "v"), ("mlp", "mlp"))
NVFP4_EXPERTS = "nvfp4_four_over_six"  # W4A4: per-token four-over-six NVFP4 activations
MXFP8_EXPERTS = "mxfp8"                # W8A8: MXFP8 activations
_EXPERT_CHUNK = 16                     # experts encoded per step (~0.2 GB of NVFP4 or ~0.25 GB of MXFP8 words)


def is_inkling2_loop(config: Mapping) -> bool:
    return config.get("architectures") == [ARCHITECTURE]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Inkling2Loop config: {message}")


def _positive_int(raw: Mapping, key: str) -> int:
    value = raw.get(key)
    _require(type(value) is int and value > 0, f"{key} must be a positive integer")
    return value


def _number(raw: Mapping, key: str) -> float:
    value = raw.get(key)
    _require(type(value) in (int, float) and value == value and abs(value) != float("inf"), f"{key} must be finite")
    return float(value)


def text_config(source: Mapping, expert_formats: list[str]) -> dict:
    """The mathematical text config. ``expert_formats`` gives each layer's expert arithmetic (NVFP4_EXPERTS or
    MXFP8_EXPERTS), read from the checkpoint's tensors: the activation rule is part of the model, and the loader
    refuses an artifact whose banks disagree with it."""

    _require(is_inkling2_loop(source), f"architectures must be [{ARCHITECTURE!r}]")
    raw = source.get("text_config")
    _require(isinstance(raw, Mapping) and raw.get("model_type") == "inkling2_loop_text", "text_config is missing")
    for key, expected in (
        ("q_bias", False), ("o_bias", False), ("n_shared_experts", 0), ("use_gate_bias", True),
        ("use_value_norm_no_gamma", True), ("attn_res_normalize_values", True),
    ):
        _require(raw.get(key) == expected, f"{key} must be {expected!r}")
    h = _positive_int(raw, "hidden_size")
    layers = _positive_int(raw, "num_hidden_layers")
    heads, kv_heads, head_dim = (_positive_int(raw, key) for key in ("num_attention_heads", "num_key_value_heads",
                                                                     "head_dim"))
    _require(heads % kv_heads == 0, "num_attention_heads must be a multiple of num_key_value_heads")
    for key, value in (("swa_head_dim", head_dim), ("swa_num_attention_heads", heads),
                       ("swa_num_key_value_heads", kv_heads)):
        _require(raw.get(key) == value, f"{key} must equal the global attention's")
    local = raw.get("local_layer_ids")
    _require(isinstance(local, list) and all(type(i) is int and 0 <= i < layers for i in local)
             and local == sorted(set(local)), "local_layer_ids must be sorted layer indices")
    loop = raw.get("loop_config")
    _require(isinstance(loop, Mapping) and loop.get("is_varloop") is False, "loop_config must be a fixed loop")
    start, end, passes = (_positive_int(loop, key) for key in ("core_start", "core_end", "num_passes"))
    block = _positive_int(raw, "n_layers_per_attn_res_block")
    _require(0 < start < end <= layers and start % block == 0 and end % block == 0,
             "the loop core must be whole attention-residual blocks")
    vocab = _positive_int(raw, "unpadded_vocab_size")
    _require(vocab <= _positive_int(raw, "vocab_size"), "unpadded_vocab_size exceeds vocab_size")
    _require(len(expert_formats) == layers and set(expert_formats) <= {NVFP4_EXPERTS, MXFP8_EXPERTS},
             "every layer needs an expert format")
    window = _positive_int(raw, "sliding_window_size")
    return {
        "architectures": [TEXT_ARCHITECTURE],
        "model_type": "inkling2_loop_text",
        "hidden_size": h,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "layer_types": ["sliding_attention" if i in local else "full_attention" for i in range(layers)],
        "sliding_window": window,
        "relative_dim": _positive_int(raw, "d_rel"),
        "relative_extent": _positive_int(raw, "rel_extent"),
        "qk_clip_threshold": _number(raw, "qk_l2_clip_threshold"),
        "relative_clip_threshold": _number(raw, "rel_l2_clip_threshold"),
        "log_scaling_alpha": _number(raw, "log_scaling_alpha"),
        "log_scaling_floor": _positive_int(raw, "log_scaling_n_floor"),
        "conv_kernel_size": _positive_int(raw, "sconv_kernel_size"),
        "attention_residual_block": block,
        "loop": {"core_start": start, "core_end": end, "passes": passes,
                 "message_rank": _positive_int(loop, "gate_rank")},
        "num_experts": _positive_int(raw, "n_routed_experts"),
        "num_experts_per_tok": _positive_int(raw, "num_experts_per_tok"),
        "moe_intermediate_size": _positive_int(raw, "intermediate_size"),
        "route_scale": _number(raw, "route_scale"),
        "gate_softcap": _number(raw, "mlp_gate_softcap"),
        "up_softcap": _number(raw, "mlp_up_softcap"),
        "expert_formats": list(expert_formats),
        "final_logit_softcapping": _number(raw, "final_logit_softcapping"),
        "rms_norm_eps": _number(raw, "rms_norm_eps"),
        "max_position_embeddings": _positive_int(raw, "model_max_length"),
        "eos_token_id": source.get("eos_token_id"),
    }


def dspark_config(source: Mapping, text: Mapping) -> dict:
    raw = source.get("mtp_config")
    _require(isinstance(raw, Mapping), "mtp_config is missing")
    for key, expected in (("q_bias", False), ("o_bias", False), ("markov_head_type", "rnn_hidden"),
                          ("hidden_size", text["hidden_size"])):
        _require(raw.get(key) == expected, f"mtp_config.{key} must be {expected!r}")
    layers = _positive_int(raw, "num_hidden_layers")
    local = raw.get("local_layer_ids")
    _require(isinstance(local, list) and all(type(i) is int and 0 <= i < layers for i in local),
             "mtp_config.local_layer_ids must be layer indices")
    visits = (text["loop"]["core_end"] - text["loop"]["core_start"]) * (text["loop"]["passes"] - 1) + \
        text["num_hidden_layers"]
    targets = raw.get("target_layer_ids")
    _require(isinstance(targets, list) and len(targets) >= 2 and targets[-1] == visits
             and all(type(v) is int and 0 <= v < visits for v in targets[:-1]),
             "mtp_config.target_layer_ids must be visit inputs and then the final state")
    return {
        "architectures": [DSPARK_ARCHITECTURE],
        "hidden_size": raw["hidden_size"],
        "num_hidden_layers": layers,
        "num_attention_heads": _positive_int(raw, "num_attention_heads"),
        "num_key_value_heads": _positive_int(raw, "num_key_value_heads"),
        "head_dim": _positive_int(raw, "head_dim"),
        "intermediate_size": _positive_int(raw, "intermediate_size"),
        "layer_types": ["sliding_attention" if i in local else "full_attention" for i in range(layers)],
        "sliding_window": _positive_int(raw, "sliding_window_size"),
        "rope_theta": _number(raw, "rope_theta"),
        "rms_norm_eps": _number(raw, "rms_norm_eps"),
        "block_size": _positive_int(raw, "block_size"),
        "markov_rank": _positive_int(raw, "markov_rank"),
        # Inputs of these visits (0-based over the 48 visits of a forward), then the final state.
        "target_layer_ids": list(targets),
    }


def vision_config(source: Mapping, text: Mapping) -> dict:
    """The vision tower's mathematics (design §15): the ``inkling2_vision`` ViT, its 2 x 2 merge and the projector into
    the text width. Its fixed semantics (CLS token, axial rotary on interleaved pairs, exact GELU, layer scale) are the
    model's; the config gives the sizes."""

    raw = source.get("vision_config")
    _require(isinstance(raw, Mapping), "vision_config is missing")
    for key, expected in (("model_type", "inkling2_vision"), ("projector_bias", False),
                          ("text_hidden_size", text["hidden_size"])):
        _require(raw.get(key) == expected, f"vision_config.{key} must be {expected!r}")
    _require(raw.get("hidden_act", "gelu") == "gelu", "vision_config.hidden_act must be gelu (exact)")
    _require(raw.get("num_channels", 3) == 3, "vision_config.num_channels must be 3")
    theta = 10000.0  # Transformers' default_theta; SGLang's rotary hardcodes it
    rope = raw.get("rope_parameters")
    if rope is not None:
        _require(isinstance(rope, Mapping) and rope.get("rope_type", "axial") == "axial", "vision rope must be axial")
        if "rope_theta" in rope:
            theta = _number(rope, "rope_theta")
    hidden, heads = _positive_int(raw, "hidden_size"), _positive_int(raw, "num_attention_heads")
    intermediate = hidden * _number(raw, "mlp_ratio")
    _require(intermediate == int(intermediate) and intermediate > 0, "vision MLP width must be a positive integer")
    _require(hidden % heads == 0 and (hidden // heads) % 4 == 0, "vision heads must split the width into 4k features")
    return {
        "architectures": [VISION_ARCHITECTURE],
        "model_type": "inkling2_vision",
        "hidden_size": hidden,
        "num_hidden_layers": _positive_int(raw, "num_hidden_layers"),
        "num_attention_heads": heads,
        "intermediate_size": int(intermediate),
        "patch_size": _positive_int(raw, "native_patch_size"),
        "spatial_merge_size": _positive_int(raw, "adapter_merge_size"),
        "position_grid": _positive_int(raw, "posemb_grid_size"),
        "projection_hidden_size": _positive_int(raw, "projection_hidden_dim"),
        "output_size": text["hidden_size"],
        "layer_norm_eps": _number(raw, "layer_norm_eps"),
        "rope_theta": theta,
        "image_token": IMAGE_TOKEN,
        "image_start_token": IMAGE_START_TOKEN,
        # Inkling2ImageProcessor's pixel mapping, validated against processor_config.json: v * rescale, then
        # (x - mean) / std; one image token per (patch * merge)-pixel cell.
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 1.0 / 255.0,
    }


def processor_resource(root: Path, vision: Mapping) -> bytes:
    """``processor_config.json`` byte for byte, after checking it describes the processor the frontend implements."""

    path = root / PROCESSOR
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    image = value.get("image_processor") if isinstance(value, dict) else None
    _require(isinstance(image, dict), f"{path}: image_processor is missing")
    side = vision["patch_size"] * vision["spatial_merge_size"]
    for key, expected in (("image_processor_type", "Inkling2ImageProcessor"), ("do_convert_rgb", True),
                          ("do_rescale", True), ("do_normalize", True), ("image_mean", vision["image_mean"]),
                          ("image_std", vision["image_std"]), ("rescale_factor", vision["rescale_factor"]),
                          ("size", {"height": side, "width": side})):
        _require(image.get(key) == expected, f"{path}: image_processor.{key} must be {expected!r}")
    _require(value.get("image_token", IMAGE_TOKEN) == IMAGE_TOKEN, f"{path}: image_token must be {IMAGE_TOKEN}")
    return data


# ---------------------------------------------------------------------------- expert banks


@dataclass(frozen=True, slots=True)
class StackedNvfp4Bank:
    """One layer's NVFP4 experts as the checkpoint stacks them (``w13_weight`` [E, 2I, H/2] with rows 2i gate_i and
    2i+1 up_i, ``w2_weight`` [E, H, I/2], E4M3 block scales, per-expert FP32 ``amax`` and one FP32
    ``outer_scale_denominator`` per matrix); only ``import_nvfp4_bank`` can read it."""

    shape: tuple[int, int, int]
    label: str
    store: SafetensorsSource
    prefix: str  # "...mlp.experts."
    read_encoded: object = None
    weight_divisor: object = None
    input_divisor: object = None
    input_scale: object = None

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


@dataclass(frozen=True, slots=True)
class StackedMxfp8Bank:
    """One layer's MXFP8 experts (``w13_weight`` [E, 2I, H] E4M3FN with rows 2i gate_i and 2i+1 up_i, ``w2_weight``
    [E, H, I], UE8M0 scales per 32 elements); only ``import_mxfp8_bank`` can read it."""

    shape: tuple[int, int, int]
    label: str
    store: SafetensorsSource
    prefix: str
    read_encoded: object = None
    weight_divisor: object = None
    input_divisor: object = None
    input_scale: object = None

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


def _words(store: SafetensorsSource, name: str, shape: tuple[int, ...], first: int, count: int) -> torch.Tensor:
    """Experts [first, first + count) of a stacked tensor ``[E, ...]`` as its raw bytes, one byte a word."""

    info = store.describe(name)
    if tuple(info.shape) != shape or info.dtype not in ("U8", "F8_E4M3"):
        raise ValueError(f"{name}: expected one-byte words of shape {shape}, got {info.dtype} {info.shape}")
    per = prod(shape[1:])
    raw = store.read_flat(name, first * per, (first + count) * per)
    return raw.view(torch.uint8).reshape(count, *shape[1:])


def _fp32_vector(store: SafetensorsSource, name: str, count: int) -> torch.Tensor:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != count:
        raise ValueError(f"{name}: expected {count} FP32 values, got {info.dtype} {info.shape}")
    values = store.read_flat(name).reshape(count)
    if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
        raise ValueError(f"{name}: every value must be finite and positive")
    return values


def nvfp4_multipliers(amax13: torch.Tensor, den13: torch.Tensor, amax2: torch.Tensor,
                      den2: torch.Tensor) -> torch.Tensor:
    """The layout's [E, 3] FP32 (gate, up, down) multiplier plane: ``fl32(amax / denominator)``, one IEEE FP32
    division. The model's arithmetic reads the exact ``amax`` and denominators (``moe/expert_amax``,
    ``moe/expert_denominators``); this plane serves generic decoders."""

    w13 = amax13.to(torch.float32) / den13.to(torch.float32)
    return torch.stack((w13, w13, amax2.to(torch.float32) / den2.to(torch.float32)), dim=1).contiguous()


def import_nvfp4_bank(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's stacked NVFP4 experts bit-exactly in ``nvfp4_expert_rg16_v1``."""

    if request.target.format != "nvfp4_mul" or request.target.layout != "nvfp4_expert_rg16_v1":
        raise ValueError("import_nvfp4_bank requires nvfp4_mul in nvfp4_expert_rg16_v1")
    if request.parameters:
        raise ValueError("import_nvfp4_bank accepts no numerical parameters")
    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, StackedNvfp4Bank):
        raise ValueError("import_nvfp4_bank requires exactly one stacked NVFP4 bank")
    source = request.inputs[0].source
    if tuple(request.target.shape) != source.shape:
        raise ValueError("expert bank target shape differs from its source")
    e, h, i = source.shape
    geometry = expert_bank_geometry("nvfp4_mul", source.shape)
    p, store = source.prefix, source.store

    def produce(output):
        multipliers = nvfp4_multipliers(
            _fp32_vector(store, p + "w13_weight_amax", e), _fp32_vector(store, p + "w13_weight_outer_scale_denominator", 1),
            _fp32_vector(store, p + "w2_weight_amax", e), _fp32_vector(store, p + "w2_weight_outer_scale_denominator", 1))
        for first in range(0, e, _EXPERT_CHUNK):
            count = min(_EXPERT_CHUNK, e - first)
            w13 = _words(store, p + "w13_weight", (e, 2 * i, h // 2), first, count)
            s13 = _words(store, p + "w13_weight_scale", (e, 2 * i, h // 16), first, count)
            w2 = _words(store, p + "w2_weight", (e, h, i // 2), first, count)
            s2 = _words(store, p + "w2_weight_scale", (e, h, i // 16), first, count)
            part = encode_nvfp4_expert_bank(
                w13[:, 0::2].contiguous(), s13[:, 0::2].contiguous(), w13[:, 1::2].contiguous(),
                s13[:, 1::2].contiguous(), w2.contiguous(), s2.contiguous(), multipliers[first:first + count],
                (count, h, i))
            output.write_bytes(first * geometry.record_stride, memoryview(part)[: count * geometry.record_stride])
        output.write_bytes(geometry.multiplier_offset, multipliers.numpy().tobytes())

    return request.job(produce=produce)


def import_mxfp8_bank(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's stacked MXFP8 experts bit-exactly in ``mxfp8_expert_rg16_v1``."""

    if request.target.format != "fp8_e4m3fn_g32_e8m0" or request.target.layout != "mxfp8_expert_rg16_v1":
        raise ValueError("import_mxfp8_bank requires fp8_e4m3fn_g32_e8m0 in mxfp8_expert_rg16_v1")
    if request.parameters:
        raise ValueError("import_mxfp8_bank accepts no numerical parameters")
    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, StackedMxfp8Bank):
        raise ValueError("import_mxfp8_bank requires exactly one stacked MXFP8 bank")
    source = request.inputs[0].source
    if tuple(request.target.shape) != source.shape:
        raise ValueError("expert bank target shape differs from its source")
    e, h, i = source.shape
    geometry = mxfp8_expert_geometry("fp8_e4m3fn_g32_e8m0", source.shape)
    p, store = source.prefix, source.store

    def produce(output):
        for first in range(0, e, _EXPERT_CHUNK):
            count = min(_EXPERT_CHUNK, e - first)
            w13 = _words(store, p + "w13_weight", (e, 2 * i, h), first, count)
            s13 = _words(store, p + "w13_weight_scale", (e, 2 * i, h // 32), first, count)
            w2 = _words(store, p + "w2_weight", (e, h, i), first, count)
            s2 = _words(store, p + "w2_weight_scale", (e, h, i // 32), first, count)
            part = encode_mxfp8_expert_records(
                w13[:, 0::2].contiguous(), s13[:, 0::2].contiguous(), w13[:, 1::2].contiguous(),
                s13[:, 1::2].contiguous(), w2.contiguous(), s2.contiguous(), (count, h, i))
            output.write_bytes(first * geometry.record_stride, part)

    return request.job(produce=produce)


# ---------------------------------------------------------------------------- derived sources


def lag_major_conv(store: SafetensorsSource, name: str, channels: int, kernel: int) -> LogicalSource:
    """A static depthwise causal conv ``[C, 1, k]`` (``F.conv1d`` with left padding: tap k-1 weights the current
    token) as ``[k, C]`` with row l weighting the token l steps back."""

    def read(begin: int, end: int) -> torch.Tensor:
        info = store.describe(name)
        if tuple(info.shape) != (channels, 1, kernel):
            raise ValueError(f"{name}: expected [{channels}, 1, {kernel}], got {info.shape}")
        weights = store.read_flat(name).reshape(channels, kernel)
        return weights.flip(1).t().contiguous().reshape(-1)[begin:end]

    return LogicalSource((kernel, channels), f"{store.path}:{name} lag-major", read)


def message_queries(store: SafetensorsSource, loop_index: int, hidden: int, loop_layers: int) -> LogicalSource:
    """Loop layer ``loop_index``'s four cross-pass queries ``[4, H]`` in (query, key, value, mlp) order: the
    checkpoint stores (query, mlp) row pairs in ``middle_cycle_cross_pass_queries`` and (key, value) pairs in
    ``middle_cycle_cross_pass_kv_queries`` (Transformers' Inkling2LoopMergeMessageQueries)."""

    names = (PREFIX + "middle_cycle_cross_pass_queries", PREFIX + "middle_cycle_cross_pass_kv_queries")

    def read(begin: int, end: int) -> torch.Tensor:
        pairs = []
        for name in names:
            info = store.describe(name)
            if info.dtype != "F32" or tuple(info.shape) != (2 * loop_layers, hidden):
                raise ValueError(f"{name}: expected FP32 [{2 * loop_layers}, {hidden}], got {info.dtype} {info.shape}")
            pairs.append(store.read_flat(name, 2 * loop_index * hidden, (2 * loop_index + 2) * hidden).reshape(2, hidden))
        (query, mlp), (key, value) = pairs
        return torch.stack((query, key, value, mlp)).reshape(-1)[begin:end]

    return LogicalSource((4, hidden), f"{store.path}:cross-pass queries of loop layer {loop_index}", read)


def stacked_fp32(store: SafetensorsSource, names: tuple[str, ...], count: int, label: str, *,
                 flat: bool = False) -> LogicalSource:
    """``[count, len(names)]`` FP32 from per-name vectors of ``count`` finite positive values (the experts' amax
    pair); ``flat`` with ``count`` 1: ``[len(names)]`` (the two denominators)."""

    def read(begin: int, end: int) -> torch.Tensor:
        return torch.stack([_fp32_vector(store, name, count) for name in names], dim=1).reshape(-1)[begin:end]

    if flat and count != 1:
        raise ValueError("a flat stack holds one value a name")
    return LogicalSource((len(names),) if flat else (count, len(names)), label, read)


def class_row(store: SafetensorsSource, hidden: int) -> LogicalSource:
    """The class token's row ``[H]``: ``class_embedding + cls_position_embedding``, one BF16 addition as both references
    form it before the tower (``(class + cls_pos).to(bf16)`` of two BF16 vectors)."""

    names = (VISUAL + "backbone.embeddings.class_embedding", VISUAL + "backbone.embeddings.cls_position_embedding")

    def read(begin: int, end: int) -> torch.Tensor:
        parts = []
        for name in names:
            info = store.describe(name)
            if info.dtype != "BF16" or tuple(info.shape) != (hidden,):
                raise ValueError(f"{name}: expected BF16 [{hidden}], got {info.dtype} {info.shape}")
            parts.append(store.read_flat(name).reshape(hidden))
        return (parts[0] + parts[1])[begin:end]

    return LogicalSource((hidden,), f"{store.path}:class_embedding + cls_position_embedding", read)


def fold_source(head: SafetensorsSource, head_name: str, rows: int, drafter: SafetensorsSource,
                *, bias: bool) -> LogicalSource:
    """DSpark's Fold ``[rows, rank]`` (``bias``: fold_bias ``[rows]``) as TokenSpeed's and SGLang's
    ``refresh_fold`` derive it: the head rows times ``hidden_out`` (weight, or bias) in FP64, copied into a BF16
    tensor (torch's double-to-BF16 conversion)."""

    weight_name, bias_name = "markov_head.hidden_out.weight", "markov_head.hidden_out.bias"
    h = head.describe(head_name).shape[1]
    rank = drafter.describe(weight_name).shape[1]
    cache: list[torch.Tensor] = []

    def compute() -> torch.Tensor:
        if not cache:
            if bias:
                projection = drafter.read_flat(bias_name).reshape(h).to(torch.float64)
                out = torch.empty(rows, dtype=torch.bfloat16)
            else:
                projection = drafter.read_flat(weight_name).reshape(h, rank).to(torch.float64)
                out = torch.empty(rows, rank, dtype=torch.bfloat16)
            for first in range(0, rows, 4096):
                last = min(rows, first + 4096)
                head_rows = head.read_flat(head_name, first * h, last * h).reshape(last - first, h).to(torch.float64)
                out[first:last].copy_(head_rows @ projection)
            cache.append(out.reshape(-1))
        return cache[0]

    shape = (rows,) if bias else (rows, rank)
    label = f"fold{'_bias' if bias else ''}({head.path}:{head_name}, {drafter.path})"
    return LogicalSource(shape, label, lambda begin, end: compute()[begin:end])


# ---------------------------------------------------------------------------- builder


class _Builder:
    def __init__(self, model: Model):
        self.model = model

    def add(self, name: str, store: SafetensorsSource, source_name: str | None, shape, *, inputs=(),
            direct: str = "bf16", rows: int | None = None, source: LogicalSource | None = None,
            source_shape=None) -> None:
        shape = tuple(shape)
        if source is None:
            info = store.describe(source_name)
            expected = tuple(source_shape) if source_shape is not None else shape
            if tuple(info.shape) != expected:
                raise ValueError(f"{source_name}: expected shape {expected}, got {tuple(info.shape)}")
            wanted = {"bf16": "BF16", "fp32": "F32"}[direct]
            if info.dtype != wanted:
                raise ValueError(f"{source_name}: expected {wanted}, got {info.dtype}")
            source = tensor_source(store, source_name, expected)
            if rows is not None:
                source = select_rows(source, ((0, rows),))
        if source.shape != shape:
            raise ValueError(f"{name}: source shape {source.shape} differs from {shape}")
        self.model.add(Parameter(name, shape, source, None, tuple(inputs), direct, residency=name.split("/", 1)[0]))

    def group(self, *names: str) -> None:
        self.model.packing_groups.append(tuple(names))

    def layer(self, store: SafetensorsSource, config: dict, index: int) -> None:
        h = config["hidden_size"]
        heads, kv, d = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        rel, k = config["relative_dim"], config["conv_kernel_size"]
        full = config["layer_types"][index] == "full_attention"
        p, sp = f"text/layers/{index}/", f"{PREFIX}layers.{index}."
        sa = sp + "attn."
        xq, xk, xv = p + "attention/query_input", p + "attention/key_input", p + "attention/value_input"
        start, end = config["loop"]["core_start"], config["loop"]["core_end"]

        # Block Attention Residuals: the layer's mix queries (none at layer 0, whose input is the embedding alone).
        if index > 0:
            query_shape = (3, h) if full else (h,)
            self.add(p + "attention_mix_query", store, f"{PREFIX}attn_res.attn_queries.{index - 1}", query_shape,
                     direct="fp32")
        self.add(p + "mlp_mix_query", store, f"{PREFIX}attn_res.mlp_queries.{index}", (h,), direct="fp32")
        if full:
            for role, leaf in (("query_norm", "q_input_norm"), ("key_norm", "k_input_norm"),
                               ("value_norm", "v_input_norm")):
                self.add(p + role, store, f"{sp}{leaf}.weight", (h,))
        else:
            self.add(p + "attention_norm", store, sp + "attn_norm.weight", (h,))
        self.add(p + "mlp_norm", store, sp + "mlp_norm.weight", (h,))

        # Attention: the K2 parent (query, relative, gate, key, value: 8,576 rows, a multiple of 16) and a small
        # parent of the few-row projections (sink; the full layers' conv taps).
        a = p + "attention/"
        for role, leaf, rows, x in (
            ("query", "wq_du", heads * d, xq), ("relative", "wr_du", heads * rel, xq),
            ("gate", "gate_attn", heads * d, xq), ("key", "wk_dv", kv * d, xk), ("value", "wv_dv", kv * d, xv),
        ):
            self.add(a + role, store, f"{sa}{leaf}.weight", (rows, h), inputs=(x,))
        self.group(*(a + role for role in ("query", "relative", "gate", "key", "value")))
        self.add(a + "sink", store, sa + "attn_sink_proj.weight", (heads, h), inputs=(xq,))
        small = [a + "sink"]
        if full:
            self.add(a + "key_taps", store, sa + "k_sconv.weight_proj", (kv * k, h), inputs=(xk,))
            self.add(a + "value_taps", store, sa + "v_sconv.weight_proj", (kv * k, h), inputs=(xv,))
            small += [a + "key_taps", a + "value_taps"]
            self.group(*small)
        self.add(a + "output", store, sa + "wo_ud.weight", (h, heads * d), inputs=(a + "gated",))
        self.add(a + "query_scale", store, sa + "q_l2_clip.scale", (heads, d))
        self.add(a + "relative_scale", store, sa + "rel_l2_clip.scale", (heads, rel))
        self.add(a + "relative_table", store, sa + "rel_logits_proj.proj", (rel, config["relative_extent"]))
        for role, leaf in (("key_conv", "k_sconv"), ("value_conv", "v_sconv")):
            if full:  # the dynamic conv's static taps, stored lag-major ([k, C], row 0 the current token)
                self.add(a + role, store, f"{sa}{leaf}.static_w", (k, kv * d))
            else:
                self.add(a + role, store, None, (k, kv * d),
                         source=lag_major_conv(store, f"{sa}{leaf}.weight", kv * d, k))
        self.add(p + "attention_conv", store, None, (k, h), source=lag_major_conv(store, sp + "attn_sconv.weight", h, k))
        self.add(p + "mlp_conv", store, None, (k, h), source=lag_major_conv(store, sp + "mlp_sconv.weight", h, k))

        # Loop message gates (core layers): the four kinds' down projections form one parent; each kind's forget
        # and gain projections another.
        if start <= index < end:
            rank = config["loop"]["message_rank"]
            self.add(p + "message_queries", store, None, (4, h), direct="fp32",
                     source=message_queries(store, index - start, h, end - start))
            downs = []
            for kind, leaf in MESSAGE_KINDS:
                g, sg = f"{p}messages/{kind}/", f"{sp}controllers.{leaf}."
                self.add(g + "down", store, sg + "down_weight", (rank, 2 * h), inputs=(g + "pair",))
                self.add(g + "forget", store, sg + "forget_weight", (h, rank), inputs=(g + "latent",))
                self.add(g + "gain", store, sg + "gain_weight", (h, rank), inputs=(g + "latent",))
                self.group(g + "forget", g + "gain")
                downs.append(g + "down")
            self.group(*downs)

        # MoE: router, its FP32 selection bias, and the exact expert bank.
        m, sm = p + "moe/", sp + "mlp."
        e, ir = config["num_experts"], config["moe_intermediate_size"]
        self.add(m + "router", store, sm + "gate.weight", (e, h), inputs=(m + "input",))
        self.add(m + "router_bias", store, sm + "gate.bias", (e,), direct="fp32")
        prefix = sm + "experts."
        if config["expert_formats"][index] == NVFP4_EXPERTS:
            bank = StackedNvfp4Bank((e, h, ir), f"{store.path}:{prefix}*", store, prefix)
            self.model.add(Parameter(m + "experts", (e, h, ir), bank, None, (m + "input",), "nvfp4_mul",
                                     residency="text"))
            self.add(m + "expert_amax", store, None, (e, 2), direct="fp32", source=stacked_fp32(
                store, (prefix + "w13_weight_amax", prefix + "w2_weight_amax"), e, f"{store.path}:{prefix}*amax"))
            self.add(m + "expert_denominators", store, None, (2,), direct="fp32", source=stacked_fp32(
                store, (prefix + "w13_weight_outer_scale_denominator", prefix + "w2_weight_outer_scale_denominator"),
                1, f"{store.path}:{prefix}*denominator", flat=True))
        else:
            bank = StackedMxfp8Bank((e, h, ir), f"{store.path}:{prefix}*", store, prefix)
            self.model.add(Parameter(m + "experts", (e, h, ir), bank, None, (m + "input",), "fp8_e4m3fn_g32_e8m0",
                                     residency="text"))

    def vision(self, store: SafetensorsSource, config: dict) -> None:
        """The tower as the checkpoint stores it (BF16; the patch embedding's columns in (ky, kx, channel) order, the
        position table [grid * grid, H] row-major by (row, column)), q/k/v and their biases each one parent (one
        GEMM, one bias pass), and the class row derived."""

        h, inter = config["hidden_size"], config["intermediate_size"]
        patch, merge, grid = config["patch_size"], config["spatial_merge_size"], config["position_grid"]
        e = VISUAL + "backbone.embeddings."
        self.add("vision/patch_embedding", store, e + "patch_embedding.weight", (h, 3 * patch * patch),
                 inputs=("vision/patch_input",))
        self.add("vision/position_embedding", store, e + "position_embedding", (grid * grid, h))
        self.add("vision/class_row", store, None, (h,), source=class_row(store, h))
        self.add("vision/pre_norm", store, VISUAL + "backbone.ln_pre.weight", (h,))
        self.add("vision/pre_norm_bias", store, VISUAL + "backbone.ln_pre.bias", (h,))
        for index in range(config["num_hidden_layers"]):
            p, sp = f"vision/layers/{index}/", f"{VISUAL}backbone.layers.{index}."
            a = p + "attention/"
            self.add(p + "attention_norm", store, sp + "ln_1.weight", (h,))
            self.add(p + "attention_norm_bias", store, sp + "ln_1.bias", (h,))
            for role, leaf in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
                self.add(a + role, store, f"{sp}self_attn.{leaf}.weight", (h, h), inputs=(a + "input",))
                self.add(a + role + "_bias", store, f"{sp}self_attn.{leaf}.bias", (h,))
            self.group(a + "query", a + "key", a + "value")
            self.group(a + "query_bias", a + "key_bias", a + "value_bias")
            self.add(a + "output", store, sp + "self_attn.out_proj.weight", (h, h), inputs=(a + "attended",))
            self.add(a + "output_bias", store, sp + "self_attn.out_proj.bias", (h,))
            self.add(p + "attention_scale", store, sp + "ls_1.gamma", (h,))
            self.add(p + "mlp_norm", store, sp + "ln_2.weight", (h,))
            self.add(p + "mlp_norm_bias", store, sp + "ln_2.bias", (h,))
            self.add(p + "mlp/up", store, sp + "mlp.c_fc.weight", (inter, h), inputs=(p + "mlp/input",))
            self.add(p + "mlp/up_bias", store, sp + "mlp.c_fc.bias", (inter,))
            self.add(p + "mlp/down", store, sp + "mlp.c_proj.weight", (h, inter), inputs=(p + "mlp/hidden",))
            self.add(p + "mlp/down_bias", store, sp + "mlp.c_proj.bias", (h,))
            self.add(p + "mlp_scale", store, sp + "ls_2.gamma", (h,))
        width = config["projection_hidden_size"]
        self.add("vision/projector/in", store, VISUAL + "proj_in.weight", (width, merge * merge * h),
                 inputs=("vision/projector/merged",))
        self.add("vision/projector/out", store, VISUAL + "proj_out.weight", (config["output_size"], width),
                 inputs=("vision/projector/hidden",))

    def dspark(self, store: SafetensorsSource, base: SafetensorsSource, config: dict, text: dict) -> None:
        h, vocab = config["hidden_size"], text["vocab_size"]
        heads, kv, d = config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        inter, rank = config["intermediate_size"], config["markov_rank"]
        features = len(config["target_layer_ids"])
        self.add("dspark/feature_weights", store, "fuse_weight", (features, features))
        self.add("dspark/mask_embedding", store, "mask_embed", (h,))
        self.add("dspark/final_norm", store, "norm.weight", (h,))
        for index in range(config["num_hidden_layers"]):
            p, sp = f"dspark/layers/{index}/", f"layers.{index}."
            x, c = p + "hidden_input", p + "context_input"
            self.add(p + "context_norm", store, sp + "dflare_norm.weight", (h,))
            self.add(p + "input_norm", store, sp + "input_layernorm.weight", (h,))
            self.add(p + "post_attention_norm", store, sp + "post_attention_layernorm.weight", (h,))
            a, sa = p + "attention/", sp + "self_attn."
            self.add(a + "query", store, sa + "q_proj.weight", (heads * d, h), inputs=(x,))
            self.add(a + "key", store, sa + "k_proj.weight", (kv * d, h), inputs=(x,))
            self.add(a + "value", store, sa + "v_proj.weight", (kv * d, h), inputs=(x,))
            self.group(a + "query", a + "key", a + "value")
            self.add(a + "context_key", store, sa + "k_proj_ctx.weight", (kv * d, h), inputs=(c,))
            self.add(a + "context_value", store, sa + "v_proj_ctx.weight", (kv * d, h), inputs=(c,))
            self.group(a + "context_key", a + "context_value")
            self.add(a + "query_norm", store, sa + "q_norm.weight", (d,))
            self.add(a + "key_norm", store, sa + "k_norm.weight", (d,))
            self.add(a + "context_key_norm", store, sa + "k_norm_ctx.weight", (d,))
            self.add(a + "output", store, sa + "o_proj.weight", (h, heads * d), inputs=(a + "mixed",))
            f, sf = p + "mlp/", sp + "mlp."
            self.add(f + "gate", store, sf + "gate_proj.weight", (inter, h), inputs=(f + "input",))
            self.add(f + "up", store, sf + "up_proj.weight", (inter, h), inputs=(f + "input",))
            self.group(f + "gate", f + "up")
            self.add(f + "down", store, sf + "down_proj.weight", (h, inter), inputs=(f + "product",))
        mk, smk = "dspark/markov/", "markov_head."
        markov_rows = store.describe(smk + "markov_w1.weight").shape[0]
        self.add(mk + "context_norm", store, smk + "dflare_norm.weight", (h,))
        self.add(mk + "embedding", store, smk + "markov_w1.weight", (vocab, rank), rows=vocab,
                 source_shape=(markov_rows, rank))
        width = 2 * rank + 2 * h
        self.add(mk + "projection", store, smk + "rnn_proj.weight", (3 * rank, width), inputs=(mk + "input",))
        self.add(mk + "projection_bias", store, smk + "rnn_proj.bias", (3 * rank,))
        head = PREFIX + "unembed.weight"
        self.add(mk + "fold", store, None, (vocab, rank), inputs=(mk + "latent",),
                 source=fold_source(base, head, vocab, store, bias=False))
        self.add(mk + "fold_bias", store, None, (vocab,), source=fold_source(base, head, vocab, store, bias=True))


def _expert_formats(store: SafetensorsSource, layers: int) -> list[str]:
    formats = []
    for index in range(layers):
        prefix = f"{PREFIX}layers.{index}.mlp.experts."
        if store.has(prefix + "w13_weight_amax"):
            if store.describe(prefix + "w13_weight").dtype != "U8":
                raise ValueError(f"{prefix}w13_weight: NVFP4 experts store packed U8 codes")
            formats.append(NVFP4_EXPERTS)
        elif store.describe(prefix + "w13_weight").dtype == "F8_E4M3" and \
                store.describe(prefix + "w13_weight_scale").dtype == "U8":
            formats.append(MXFP8_EXPERTS)
        else:
            raise ValueError(f"{prefix}: unrecognized expert encoding")
    return formats


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    drafter: SafetensorsSource | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    """The logical model: ``text`` and optionally ``dspark`` (``drafter``: ``mtp.safetensors``) and ``vision`` (the
    tower and projector, with ``processor_config.json``)."""

    selected = set(components)
    if "text" not in selected or selected - {"text", "dspark", "vision"}:
        raise ValueError("Inkling2Loop: select text and optionally dspark and vision")
    layers = base.config.get("text_config", {}).get("num_hidden_layers")
    if type(layers) is not int or layers <= 0:
        raise ValueError("Inkling2Loop config: text_config.num_hidden_layers is missing")
    config = text_config(base.config, _expert_formats(base, layers))
    records = {"text": {"config": config}}
    if "dspark" in selected:
        if drafter is None:
            raise ValueError("the dspark component reads the drafter (mtp.safetensors)")
        records["dspark"] = {"config": dspark_config(base.config, config), "target": "text"}
    if "vision" in selected:
        records["vision"] = {"config": vision_config(base.config, config), "target": "text"}
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], overrides=resource_overrides, qwen_normalizers=False
    )
    if count != config["vocab_size"]:
        raise ValueError(f"the tokenizer's {count} public tokens differ from the model's {config['vocab_size']} rows")
    if "vision" in selected:
        object_id = f"resource/vision/{PROCESSOR}"
        refs["vision"] = {PROCESSOR: object_id}
        resources[object_id] = processor_resource(Path(base.root), records["vision"]["config"])
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Builder(model)
    h, vocab = config["hidden_size"], config["vocab_size"]
    padded = base.describe(PREFIX + "embed.weight").shape[0]
    builder.add("text/token_embedding", base, PREFIX + "embed.weight", (vocab, h), rows=vocab,
                source_shape=(padded, h))
    builder.add("text/embedding_norm", base, PREFIX + "embed_norm.weight", (h,))
    heads = ("text/final_hidden",) + (("dspark/final_hidden",) if "dspark" in selected else ())
    builder.add("text/output_head", base, PREFIX + "unembed.weight", (vocab, h), rows=vocab, inputs=heads,
                source_shape=(padded, h))
    builder.add("text/final_norm", base, PREFIX + "norm.weight", (h,))
    builder.add("text/output_query", base, PREFIX + "attn_res.output_query", (h,), direct="fp32")
    for index in range(layers):
        builder.layer(base, config, index)
    # The [3, H] mix queries sit exactly at the full layers' indices minus one (the layer - 1 offset).
    full = [i for i, kind in enumerate(config["layer_types"]) if kind == "full_attention"]
    triples = [i for i in range(layers - 1) if len(base.describe(f"{PREFIX}attn_res.attn_queries.{i}").shape) == 2]
    if triples != [i - 1 for i in full if i > 0]:
        raise ValueError("the [3, H] attention-residual queries do not sit at the full layers' indices minus one")
    if "dspark" in selected:
        builder.dspark(drafter, base, records["dspark"]["config"], config)
    if "vision" in selected:
        builder.vision(base, records["vision"]["config"])
    return model
