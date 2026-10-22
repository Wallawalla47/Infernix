"""Qwen4Exp (Qwen3.8-Flash-Next) architecture adapter: mathematical config, source axes, packing
groups, the exact routed-expert banks and the PLE n-gram volume.

The mathematics is the upstream ``qwen4_exp`` model (docs/maintainer/qwen3_8-flash-next-design.md
§2): Qwen3.5 GDN and gated attention blocks inside four hyper-connection residual streams, QSA
sparse attention with a pooled-key indexer on every full-attention layer, a 512-expert MoE with a
gated shared expert, and one PLE n-gram injection. Recipes and quantization assignments live in
official_recipes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from pathlib import Path
import struct
from typing import Mapping

import torch

from tools.artifact.codecs.fp8_b128 import encode_expert as encode_fp8_expert
from tools.artifact.layouts import fp8_expert_geometry
from tools.flash_next.ngram import NgramConfig, head_tables, layer_multipliers

from .expert_banks import ExpertBankSource
from .methods import PrepareRequest, PreparedMethod
from .model import Model, Parameter
from .qwen3_5 import _Builder, _f32, _fixed, _positive, _rope_source, vision_config
from .resources import load_resources
from .sources.logical import LogicalSource
from .sources.modelopt import (
    dequantize_fp8_block128,
    fp8_block_matrix_raw,
    fp8_block_matrix_words,
    nvfp4_values_source,
)
from .sources.safetensors import SafetensorsSource, tensor_source

ARCHITECTURES = ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration")
_LAYER_KINDS = {
    "full_attention": "full_attention",
    "qwen_sparse_attention": "full_attention",
    "linear_attention": "linear_attention",
}


def is_qwen4_exp(config: Mapping) -> bool:
    architectures = config.get("architectures")
    return isinstance(architectures, list) and any(a in ARCHITECTURES for a in architectures)


def _eos(raw: dict) -> int:
    value = raw.get("eos_token_id")
    if isinstance(value, list):
        value = value[0] if value else None
    if type(value) is not int or value < 0:
        raise ValueError("text.eos_token_id must be a nonnegative integer")
    return value


def text_config(source: dict, *, mtp: bool) -> dict:
    architectures = source.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1 or architectures[0] not in ARCHITECTURES:
        raise ValueError(f"unsupported Qwen4Exp architecture {architectures!r}")
    raw = source.get("text_config", source)
    _fixed(raw, "hidden_act", "silu", "text")
    _fixed(raw, "attention_bias", False, "text")
    _fixed(raw, "mamba_ssm_dtype", "float32", "text")
    _fixed(raw, "norm_topk_prob", True, "text")
    _fixed(raw, "tie_word_embeddings", False, "text")
    result = {"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text"}
    for key in ("hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings"):
        result[key] = _positive(raw.get(key), "text." + key)
    result["tie_word_embeddings"] = False
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps")
    layers = raw.get("layer_types")
    if not isinstance(layers, list) or len(layers) != result["num_hidden_layers"]:
        raise ValueError("text.layer_types must describe every block")
    if any(kind not in _LAYER_KINDS for kind in layers):
        raise ValueError("text.layer_types contains an unsupported block")
    result["layer_types"] = [_LAYER_KINDS[kind] for kind in layers]

    for key in ("num_attention_heads", "num_key_value_heads", "head_dim"):
        result[key] = _positive(raw.get(key), "text." + key)
    if result["num_attention_heads"] % result["num_key_value_heads"]:
        raise ValueError("text attention heads must be divisible by KV heads")
    rope = _rope_source(raw, "text")
    _fixed(rope, "mrope_interleaved", True, "text.rope_parameters")
    factor = _f32(rope.get("partial_rotary_factor", raw.get("partial_rotary_factor", 0.25)), "partial_rotary_factor")
    theta = _f32(rope.get("rope_theta", raw.get("rope_theta", 10_000_000)), "rope_theta")
    sections = rope.get("mrope_section")
    rotary = int(result["head_dim"] * factor)
    if (
        not isinstance(sections, list)
        or len(sections) != 3
        or any(type(v) is not int or v < 0 for v in sections)
        or rotary <= 0
        or rotary % 2
        or sum(sections) != rotary // 2
    ):
        raise ValueError("MRoPE sections and rotary width disagree")
    result["rope_parameters"] = {
        "rope_theta": theta,
        "partial_rotary_factor": factor,
        "mrope_section": list(sections),
    }

    for key in (
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    ):
        result[key] = _positive(raw.get(key), "text." + key)
    if result["linear_num_value_heads"] % result["linear_num_key_heads"]:
        raise ValueError("linear value heads must be divisible by key heads")
    gate = raw.get("output_gate_type") or raw.get("hidden_act")
    if gate not in ("sigmoid", "silu"):
        raise ValueError(f"text.output_gate_type {gate!r} is not sigmoid or silu")
    result["output_gate_type"] = gate

    for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size"):
        result[key] = _positive(raw.get(key), "text." + key)
    if result["num_experts_per_tok"] > result["num_experts"]:
        raise ValueError("selected experts exceed expert count")
    result["norm_topk_prob"] = True

    result["hc_count"] = _positive(raw.get("hc_count", 4), "text.hc_count")
    if result["hc_count"] < 2:
        raise ValueError("text.hc_count must exceed 1")
    result["hc_lowrank"] = _positive(raw.get("hc_lowrank", 320), "text.hc_lowrank")

    for key in ("indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio"):
        result[key] = _positive(raw.get(key), "text." + key)
    if result["indexer_kv_heads"] != 1:
        raise ValueError("QSA requires one indexer key head")
    if result["indexer_budget"] % result["indexer_compress_ratio"]:
        raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
    if rotary > result["indexer_head_dim"]:
        raise ValueError("the attention RoPE width must fit the QSA index head")

    ple = raw.get("ple_layer_ids") or []
    if not isinstance(ple, list) or any(type(v) is not int for v in ple) or len(set(ple)) != len(ple):
        raise ValueError("text.ple_layer_ids must be distinct integers")
    if len(ple) != 1:
        raise ValueError("this adapter implements exactly one PLE layer")
    ple = sorted(ple)
    for layer_id in ple:
        if not 1 <= layer_id <= result["num_hidden_layers"] or result["layer_types"][layer_id - 1] != "linear_attention":
            raise ValueError("PLE layers must be one-indexed linear-attention layers")
    result["ple_layer_ids"] = ple
    result["ple_embed_dim"] = _positive(raw.get("ple_embed_dim", result["hidden_size"]), "text.ple_embed_dim")
    result["ple_conv_kernel_size"] = _positive(raw.get("ple_conv_kernel_size", 4), "text.ple_conv_kernel_size")
    result["ngram_size"] = _positive(raw.get("ngram_size", 3), "text.ngram_size")
    result["heads_per_ngram"] = _positive(raw.get("heads_per_ngram", 8), "text.heads_per_ngram")
    if result["ngram_size"] < 2:
        raise ValueError("text.ngram_size must be at least 2")
    heads = (result["ngram_size"] - 1) * result["heads_per_ngram"]
    if result["ple_embed_dim"] % heads:
        raise ValueError("ple_embed_dim must be divisible by the n-gram head count")
    result["ngram_vocab_size_base"] = _positive(raw.get("ngram_vocab_size_base", 20_000_000), "text.ngram_vocab_size_base")
    result["make_ngram_vocab_size_divisible_by"] = _positive(
        raw.get("make_ngram_vocab_size_divisible_by", 128), "text.make_ngram_vocab_size_divisible_by"
    )
    seed = raw.get("seed", 1234)
    if type(seed) is not int or seed < 0:
        raise ValueError("text.seed must be a nonnegative integer")
    result["seed"] = seed
    result["split_ngram_parts"] = _positive(raw.get("split_ngram_parts", 512), "text.split_ngram_parts")
    result["eos_token_id"] = _eos(raw)
    # NVIDIA's checkpoints label the table FP8 or NVFP4 (FP4-Local); an unlabelled one is BF16.
    if raw.get("ple_embedding_dtype", "float8_e4m3fn") not in ("float8_e4m3fn", "nvfp4"):
        raise ValueError("this adapter imports an FP8, NVFP4 or BF16 n-gram table")

    if mtp:
        _fixed(raw, "mtp_num_hidden_layers", 1, "text")
        _fixed(raw, "mtp_use_dedicated_embeddings", False, "text")
        mtp_raw = raw.get("mtp", {})
        if isinstance(mtp_raw, dict) and mtp_raw.get("layer_types", ["full_attention"]) != ["full_attention"]:
            raise ValueError("the MTP block must be one full-attention (QSA) layer")
    return result


def ngram_config(config: dict) -> NgramConfig:
    return NgramConfig(
        vocab_size=config["vocab_size"],
        eos_token_id=config["eos_token_id"],
        ngram_size=config["ngram_size"],
        heads_per_ngram=config["heads_per_ngram"],
        ngram_vocab_size_base=config["ngram_vocab_size_base"],
        make_ngram_vocab_size_divisible_by=config["make_ngram_vocab_size_divisible_by"],
        seed=config["seed"],
    )


# ---------------------------------------------------------------------------- routed expert banks


_BANK_CHUNK = 32  # FP8 experts encoded per step; bounds conversion memory


@dataclass(frozen=True, slots=True)
class Fp8ExpertBankSource:
    """One layer's block-FP8 experts (Qwen/Qwen3.8-Flash-Next-FP8: E4M3FN codes, one BF16 scale per
    128 x 128 tile, per expert); only ``import_fp8_expert_bank`` can read it."""

    shape: tuple[int, int, int]
    label: str
    store: SafetensorsSource
    prefix: str  # "...mlp.experts." so expert e's gate is f"{prefix}{e}.gate_proj"
    read_encoded: object = None
    weight_divisor: object = None
    input_divisor: object = None

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


def fp8_expert_records(source: Fp8ExpertBankSource, first: int, count: int) -> bytes:
    """Experts [first, first + count) of a block-FP8 bank as ``fp8_expert_b128_v1`` records."""

    _, hidden, intermediate = source.shape
    parts = []
    for e in range(first, first + count):
        leaf = f"{source.prefix}{e}."
        parts.append(encode_fp8_expert(
            fp8_block_matrix_raw(source.store, leaf + "gate_proj", (intermediate, hidden)),
            fp8_block_matrix_raw(source.store, leaf + "up_proj", (intermediate, hidden)),
            fp8_block_matrix_raw(source.store, leaf + "down_proj", (hidden, intermediate)),
            label=f"{source.label}.{e}"))
    return b"".join(parts)


def import_fp8_expert_bank(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's block-FP8 experts verbatim in ``fp8_expert_b128_v1`` (storage layouts §9)."""

    if request.target.format != "fp8_e4m3fn_b128" or request.target.layout != "fp8_expert_b128_v1":
        raise ValueError("import_fp8_expert_bank requires fp8_e4m3fn_b128 in fp8_expert_b128_v1")
    if request.parameters:
        raise ValueError("import_fp8_expert_bank accepts no numerical parameters")
    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, Fp8ExpertBankSource):
        raise ValueError("import_fp8_expert_bank requires exactly one block-FP8 expert-bank source")
    source = request.inputs[0].source
    if tuple(request.target.shape) != source.shape:
        raise ValueError("expert bank target shape differs from its source")
    geometry = fp8_expert_geometry("fp8_e4m3fn_b128", source.shape)

    def produce(output):
        for first in range(0, source.shape[0], _BANK_CHUNK):
            count = min(_BANK_CHUNK, source.shape[0] - first)
            output.write_bytes(first * geometry.record_stride, fp8_expert_records(source, first, count))

    return request.job(produce=produce)


def _expert_input_scales(store: SafetensorsSource, prefix: str, experts: int) -> LogicalSource:
    """FP32 ``[experts, 3]`` activation global scales in (gate, up, down) order, as stored."""

    cache: list[torch.Tensor] = []

    def read(begin: int, end: int) -> torch.Tensor:
        if not cache:
            words = []
            for e in range(experts):
                for role in ("gate", "up", "down"):
                    name = f"{prefix}{e}.{role}_proj.input_scale"
                    info = store.describe(name)
                    if info.dtype != "F32" or prod(info.shape) != 1:
                        raise ValueError(f"{name}: expected one FP32 scalar")
                    words.append(store.read_flat(name).reshape(()))
            cache.append(torch.stack(words))
        return cache[0][begin:end]

    return LogicalSource((experts, 3), f"{store.path}:{prefix}*.input_scale", read)


def fp8_block_values(store: SafetensorsSource, leaf: str, shape: tuple[int, int]) -> LogicalSource:
    """The exact values of one block-FP8 matrix ``leaf`` (code x FP32 tile multiplier, binary64).

    NVIDIA stores the MTP routed experts this way, per expert. Each read decodes only the rows it
    covers, so a recipe's row-chunked quantizer never holds more than one matrix's codes.
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


# ---------------------------------------------------------------------------- builder


class _Qwen4ExpBuilder(_Builder):
    """Reuses the Qwen3.5 GDN and Vision source axes; every other block is Qwen4Exp's own."""

    def validate_source(self, selected, original, name):
        if selected is not original:
            raise ValueError(f"{name}: Qwen4Exp converts from its single NVIDIA source")

    def add_fp8_block(self, name, store, leaf, shape, *, inputs=()):
        """A parameter read through its exact block-FP8 values (recipes quantize or cast them)."""

        def factory(selected, format=None):
            self.validate_source(selected, store, name)
            if format is not None:
                raise ValueError(f"{name}: block-scaled FP8 is converted through its values")
            return fp8_block_values(selected, leaf, shape)

        self.model.add(Parameter(name, tuple(shape), factory(store), factory, tuple(inputs), "bf16",
                                 residency=name.split("/", 1)[0]))

    def add_nvfp4_values(self, name, store, leaf, shape, *, inputs=()):
        """A weight-only ModelOpt NVFP4 matrix read through its exact values (multiplier semantics;
        recipes cast or quantize them): the FP4-Local vision MLPs and MTP experts."""

        def factory(selected, format=None):
            self.validate_source(selected, store, name)
            if format is not None:
                raise ValueError(f"{name}: a weight-only NVFP4 matrix is converted through its values")
            return nvfp4_values_source(selected, leaf, tuple(shape))

        self.model.add(Parameter(name, tuple(shape), factory(store), factory, tuple(inputs), "bf16",
                                 residency=name.split("/", 1)[0]))

    def add(self, name, store, source_name, shape, **options):
        # A vision tower stored as ModelOpt NVFP4 (FP4-Local's MLPs and merger: weight-only, a FP32
        # multiplier) reads its exact values; every other tensor takes the shared source rules (MXFP8
        # matrices, FP4-Local's vision attention included, are recognized there).
        prefix = source_name.removesuffix(".weight")
        if (
            name.startswith("vision/")
            and len(shape) == 2
            and source_name.endswith(".weight")
            and not any(options.get(key) for key in ("offset", "rows", "transpose", "source_shape"))
            and store.has(prefix + ".weight_scale_2")
            and store.describe(source_name).dtype == "U8"
        ):
            self.add_nvfp4_values(name, store, prefix, shape, inputs=options.get("inputs", ()))
            return
        super().add(name, store, source_name, shape, **options)

    def norm(self, name, store, source_name, shape):
        """A norm weight kept in its stored precision: BF16, or FP32 where the checkpoint trained it
        so (NVIDIA FP4-Local's hyper-connection and PLE norms); the engine reads either."""

        self.add(name, store, source_name, shape, direct="fp32" if store.describe(source_name).dtype == "F32" else "bf16")

    def hyper_connection(self, prefix, source_prefix, store, config, *, combine=True):
        width = config["hc_count"] * config["hidden_size"]
        rank = config["hc_lowrank"]
        self.norm(prefix + "norm", store, source_prefix + "hc_norm.weight", (width,))
        self.add(prefix + "down", store, source_prefix + "input_mix_weight_down.weight", (rank, width),
                 inputs=(prefix + "input",))
        self.add(prefix + "up", store, source_prefix + "input_mix_weight_up.weight", (width, rank),
                 inputs=(prefix + "mix_activation",))
        if combine:
            self.add(prefix + "inject", store, source_prefix + "block_inject_weight.weight",
                     (config["hc_count"], width), inputs=(prefix + "input",))
            self.group(prefix + "down", prefix + "inject")

    def qsa_attention(self, prefix, source_prefix, store, config):
        h, d = config["hidden_size"], config["head_dim"]
        heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
        q, k = heads * d, kv * d
        mixer_input = prefix + "mixer_input"
        for role, gate in (("query", False), ("gate", True)):
            ranges = tuple(
                (head * 2 * d + int(gate) * d, head * 2 * d + (int(gate) + 1) * d) for head in range(heads)
            )
            self.add(prefix + "attention/" + role, store, source_prefix + "self_attn.q_proj.weight", (q, h),
                     source_shape=(2 * q, h), rows=ranges, inputs=(mixer_input,))
        for role, field in (("key", "k_proj"), ("value", "v_proj")):
            self.add(prefix + "attention/" + role, store, source_prefix + "self_attn." + field + ".weight", (k, h),
                     inputs=(mixer_input,))
        iq = config["indexer_n_heads"] * config["indexer_head_dim"]
        ik = config["indexer_kv_heads"] * config["indexer_head_dim"]
        index = source_prefix + "self_attn.indexer.index_qk_proj.weight"
        self.add(prefix + "attention/index_query", store, index, (iq, h), source_shape=(iq + ik, h),
                 rows=((0, iq),), inputs=(mixer_input,))
        self.add(prefix + "attention/index_key", store, index, (ik, h), source_shape=(iq + ik, h),
                 rows=((iq, iq + ik),), inputs=(mixer_input,))
        for role, field, width in (
            ("query_norm", "self_attn.q_norm.weight", d),
            ("key_norm", "self_attn.k_norm.weight", d),
            ("index_query_norm", "self_attn.indexer.q_layernorm.weight", config["indexer_head_dim"]),
            ("index_key_norm", "self_attn.indexer.k_layernorm.weight", config["indexer_head_dim"]),
        ):
            self.add(prefix + "attention/" + role, store, source_prefix + field, (width,))
        self.add(prefix + "attention/output", store, source_prefix + "self_attn.o_proj.weight", (h, q),
                 inputs=(prefix + "attention/gated_output",))
        self.group(*(prefix + "attention/" + r for r in ("query", "gate", "key", "value", "index_query", "index_key")))

    def moe(self, prefix, source_prefix, store, config, *, mtp=False):
        h, e = config["hidden_size"], config["num_experts"]
        ir, shared = config["moe_intermediate_size"], config["shared_expert_intermediate_size"]
        sp, p = source_prefix + "mlp.", prefix + "moe/"
        ffn_input = prefix + "ffn_input"
        self.add(p + "router", store, sp + "gate.weight", (e, h), inputs=(ffn_input,))
        self.add(p + "shared_score", store, sp + "shared_expert_gate.weight", (1, h), inputs=(ffn_input,))
        self.group(p + "router", p + "shared_score")
        for role in ("gate", "up"):
            self.add(p + "shared/" + role, store, sp + "shared_expert." + role + "_proj.weight", (shared, h),
                     inputs=(ffn_input,))
        self.add(p + "shared/down", store, sp + "shared_expert.down_proj.weight", (h, shared),
                 inputs=(p + "shared/product",))
        self.group(p + "shared/gate", p + "shared/up")
        if mtp:
            # Each expert's matrices become row ranges of two parents (gate and up interleaved per
            # expert, then every down), so the drafter's resident-expert kernel selects them by id.
            # NVIDIA stores the MTP experts per expert in Qwen's block-scaled FP8 (E4M3FN codes, one
            # FP32 multiplier per 128 x 128 tile; their values are the exact products), or in
            # weight-only ModelOpt NVFP4 (FP4-Local); a BF16 export keeps them fused, gate_up_proj
            # [E, 2I, H] (the gate rows first) and down_proj [E, H, I].
            fused = store.has(sp + "experts.gate_up_proj")
            nvfp4 = not fused and store.describe(sp + "experts.0.gate_proj.weight").dtype == "U8"
            per_expert = self.add_nvfp4_values if nvfp4 else self.add_fp8_block
            gate_up, downs = [], []
            for expert in range(e):
                ep, leaf = p + f"experts/{expert}/", sp + f"experts.{expert}."
                for half, role in enumerate(("gate", "up")):
                    if fused:
                        self.add(ep + role, store, sp + "experts.gate_up_proj", (ir, h), source_shape=(e, 2 * ir, h),
                                 offset=(2 * expert + half) * ir * h, inputs=(ffn_input,))
                    else:
                        per_expert(ep + role, store, leaf + role + "_proj", (ir, h), inputs=(ffn_input,))
                    gate_up.append(ep + role)
                if fused:
                    self.add(ep + "down", store, sp + "experts.down_proj", (h, ir), source_shape=(e, h, ir),
                             offset=expert * h * ir, inputs=(ep + "product",))
                else:
                    per_expert(ep + "down", store, leaf + "down_proj", (h, ir), inputs=(ep + "product",))
                downs.append(ep + "down")
            self.group(*gate_up)
            self.group(*downs)
            return
        # Qwen's official FP8 export: block-FP8 experts, imported verbatim and run W8A16 (design §19.3.25).
        if not store.has(sp + "experts.0.gate_proj.weight_packed") and \
                store.describe(sp + "experts.0.gate_proj.weight").dtype == "F8_E4M3":
            bank = Fp8ExpertBankSource((e, h, ir), f"{store.path}:{sp}experts", store, sp + "experts.")
            self.model.add(Parameter(p + "experts", (e, h, ir), bank, None, (ffn_input,), "fp8_e4m3fn_b128",
                                     residency="text"))
            return
        # ModelOpt W4A4 banks carry each matrix's activation scale; a weight-only compressed-tensors
        # bank (weight_packed) has none and is W4A16 (design §16.2.1).
        packed = store.has(sp + "experts.0.gate_proj.weight_packed")
        bank = ExpertBankSource((e, h, ir), f"{store.path}:{sp}experts", store, sp + "experts.", packed=packed)
        self.model.add(Parameter(p + "experts", (e, h, ir), bank, None, (ffn_input,), "nvfp4_mul", residency="text"))
        if not packed:
            self.model.add(Parameter(p + "expert_input_scales", (e, 3), _expert_input_scales(store, sp + "experts.", e),
                                     None, (), "fp32", residency="text"))

    def ple(self, prefix, source_prefix, store, config):
        h, width, dim = config["hidden_size"], config["hc_count"] * config["hidden_size"], config["ple_embed_dim"]
        taps = config["ple_conv_kernel_size"]
        sp = source_prefix + "ple."
        embeddings = prefix + "ple/embeddings"
        self.add(prefix + "ple/key_projection", store, sp + "key_proj.weight", (width, dim), inputs=(embeddings,))
        self.add(prefix + "ple/value_projection", store, sp + "value_proj.weight", (h, dim), inputs=(embeddings,))
        self.group(prefix + "ple/key_projection", prefix + "ple/value_projection")
        for role, field in (("key_norm", "norm_key"), ("query_norm", "norm_query"), ("conv_norm", "norm_conv")):
            self.norm(prefix + "ple/" + role, store, sp + field + ".weight", (width,))
        self.add(prefix + "ple/convolution", store, sp + "conv1d.weight", (taps, width),
                 source_shape=(width, 1, taps), transpose=(2, 0, 1))
        # An FP8 volume's per-tensor scale (the checkpoint's own, or a BF16 table's derived one, BF16)
        # or an NVFP4 volume's FP32 multiplier. A BF16 volume's rows are the values themselves.
        table = self.ngram
        format = config["ngram_table"]["format"]
        if format == "fp8_e4m3fn":
            scale = LogicalSource((1,), f"{store.path}:{table.prefix}scale", lambda begin, end: table.scale()[begin:end])
            self.model.add(Parameter(prefix + "ple/ngram_scale", (1,), scale, None, (), "bf16", residency="text"))
        elif format == "nvfp4":
            scale = LogicalSource((1,), f"{store.path}:{table.prefix}weight_scale_2",
                                  lambda begin, end: table.nvfp4_scale()[begin:end])
            self.model.add(Parameter(prefix + "ple/ngram_scale", (1,), scale, None, (), "fp32", residency="text"))

    def block(self, prefix, source_prefix, store, config, mixer, *, mtp=False):
        self.hyper_connection(prefix + "attn_hc/", source_prefix + "attn_hyper_connection.", store, config)
        if mixer == "full_attention":
            self.qsa_attention(prefix, source_prefix, store, config)
        else:
            self.gdn(prefix, source_prefix, store, config)
        self.hyper_connection(prefix + "mlp_hc/", source_prefix + "mlp_hyper_connection.", store, config)
        self.moe(prefix, source_prefix, store, config, mtp=mtp)


def _check_ngram_buffers(store: SafetensorsSource, prefix: str, config: dict, ple_index: int) -> int:
    """The checkpoint's hash buffers must equal the config's derivation; returns the table rows."""

    spec = ngram_config(config)
    sizes, offsets, padded = head_tables(spec, ple_index)
    expected = {
        "layer_multipliers": layer_multipliers(spec, ple_index),
        "ngram_heads_vocab_sizes": sizes,
        "ngram_heads_offsets": offsets,
    }
    for name, values in expected.items():
        source = prefix + "ple.ple_embedding." + name
        if store.has(source):
            stored = store.read_flat(source).tolist()
            if stored != values:
                raise ValueError(f"{source}: checkpoint buffer differs from the config's derivation")
    return padded


class NgramTable:
    """The PLE n-gram table as the volume stores it: E4M3FN rows with one per-tensor scale, or an
    NVFP4 table's rows verbatim.

    FP8 shards (NVIDIA) are the checkpoint's words and its ``weight_scale``. BF16 shards (a BF16
    export) are quantized by the per-tensor rule ModelOpt applies in the table's own dtype:
    s = bf16(amax / 448) and code = e4m3_rn_satfinite(bf16(v / s)). From the same BF16 table this
    reproduces NVIDIA's FP8 table word for word, so such a conversion can reuse its n-gram volume.

    With ``keep_bf16`` a BF16 table's words are kept as they are (rows of 2 x width bytes): the BF16
    volume any Qwen3.8-Flash-Next artifact accepts in place of its quantized table
    (tools/flash_next/ngram_bf16.py).

    NVFP4 shards (NVIDIA's FP4-Local: ``shard_i.weight`` U8 ``[rows, width / 2]``, E2M1 with the even
    element in the low nibble, ``shard_i.weight_scale`` E4M3FN ``[rows, width / 16]``, and one FP32
    ``weight_scale_2`` multiplier for the table) keep their words: a volume row is the row's code
    bytes, then its scale bytes (width / 2 + width / 16 = 90 bytes), and the multiplier is the
    table's ``nvfp4_scale``. Value = e2m1(code) * e4m3fn(scale) * multiplier.
    """

    def __init__(self, store: SafetensorsSource, prefix: str, config: dict, rows: int, *, keep_bf16: bool = False):
        self.store = store
        self.prefix = prefix + "ple.ple_embedding.ngram_embedding."
        width = config["ple_embed_dim"] // ((config["ngram_size"] - 1) * config["heads_per_ngram"])
        if width % 16:
            raise ValueError("an n-gram row must be whole 16-element blocks")
        self.shards: list[tuple[str, int]] = []
        dtypes = set()
        for index in range(config["split_ngram_parts"]):
            name = f"{self.prefix}shard_{index}.weight"
            info = store.describe(name)
            if info.dtype == "U8":
                scale = store.describe(f"{self.prefix}shard_{index}.weight_scale")
                if (len(info.shape) != 2 or info.shape[1] != width // 2 or scale.dtype != "F8_E4M3"
                        or tuple(scale.shape) != (info.shape[0], width // 16)):
                    raise ValueError(f"{name}: expected NVFP4 U8 [rows, {width // 2}] with E4M3 scales [rows, {width // 16}]")
            elif info.dtype not in ("F8_E4M3", "BF16") or len(info.shape) != 2 or info.shape[1] != width:
                raise ValueError(f"{name}: expected F8_E4M3 or BF16 [rows, {width}], or NVFP4 U8 [rows, {width // 2}]")
            dtypes.add(info.dtype)
            self.shards.append((name, info.shape[0]))
        if len(dtypes) != 1:
            raise ValueError("n-gram shards mix dtypes")
        self.bf16 = dtypes == {"BF16"}
        self.nvfp4 = dtypes == {"U8"}
        if keep_bf16 and not self.bf16:
            raise ValueError("a BF16 n-gram volume needs the checkpoint's BF16 table (this one is quantized)")
        self.keep_bf16 = keep_bf16
        self.rows = sum(count for _, count in self.shards)
        self.width = width
        # BF16 words, one E4M3FN byte per element, or NVFP4 codes then block scales.
        if keep_bf16:
            self.row_bytes = 2 * width
        elif self.nvfp4:
            self.row_bytes = width // 2 + width // 16
        else:
            self.row_bytes = width
        if self.rows != rows:
            raise ValueError(f"n-gram shards hold {self.rows} rows, the hash domain needs {rows}")
        self._scale: torch.Tensor | None = None
        self._nvfp4_scale: torch.Tensor | None = None

    @property
    def volume_format(self) -> str:
        """The artifact's ``ngram_table.format`` of the rows :meth:`codes` returns."""

        return "bf16" if self.keep_bf16 else "nvfp4" if self.nvfp4 else "fp8_e4m3fn"

    def nvfp4_scale(self) -> torch.Tensor:
        """An NVFP4 table's FP32 multiplier ``[1]`` (``weight_scale_2``, positive and finite)."""

        if not self.nvfp4:
            raise ValueError("only an NVFP4 n-gram table has an FP32 multiplier")
        if self._nvfp4_scale is None:
            name = self.prefix + "weight_scale_2"
            info = self.store.describe(name)
            if info.dtype != "F32" or prod(info.shape) != 1:
                raise ValueError(f"{name}: expected one FP32 multiplier")
            value = self.store.read_flat(name).reshape(1).clone()
            if not bool(torch.isfinite(value).all()) or float(value) <= 0:
                raise ValueError(f"{name}: the multiplier must be positive and finite")
            self._nvfp4_scale = value
        return self._nvfp4_scale

    def scale(self) -> torch.Tensor:
        """The per-tensor scale, BF16 ``[1]`` (exact: NVIDIA's FP32 word is a BF16 value)."""

        if self.nvfp4:
            raise ValueError("an NVFP4 n-gram table has block scales and an FP32 multiplier, not an FP8 scale")
        if self._scale is None:
            if self.bf16:
                amax = torch.zeros((), dtype=torch.bfloat16)
                for name, count in self.shards:
                    step = 1 << 20
                    for begin in range(0, count, step):
                        end = min(count, begin + step)
                        chunk = self.store.read_flat(name, begin * self.width, end * self.width)
                        amax = torch.maximum(amax, chunk.abs().max())
                if not bool(torch.isfinite(amax)) or float(amax) <= 0:
                    raise ValueError("the BF16 n-gram table must have a positive finite absolute maximum")
                self._scale = (amax / torch.tensor(448.0, dtype=torch.bfloat16)).reshape(1)
            else:
                name = self.prefix + "weight_scale"
                info = self.store.describe(name)
                if info.dtype not in ("F32", "BF16") or prod(info.shape) != 1:
                    raise ValueError(f"{name}: expected one scalar scale")
                value = self.store.read_flat(name).reshape(1)
                scale = value.to(torch.bfloat16)
                if not torch.equal(scale.to(value.dtype), value):
                    raise ValueError(f"{name}: the volume's scale must be exact in BF16")
                self._scale = scale
        return self._scale

    def codes(self, name: str, begin: int, end: int) -> torch.Tensor:
        """The volume's bytes ``uint8 [end - begin, row_bytes]`` of rows [begin, end) of shard ``name``:
        E4M3FN words, with ``keep_bf16`` the BF16 words unchanged, or an NVFP4 row's code bytes then
        its E4M3FN block-scale bytes."""

        if self.nvfp4:
            half, blocks = self.width // 2, self.width // 16
            codes = self.store.read_flat(name, begin * half, end * half).reshape(end - begin, half)
            scales = (
                self.store.read_flat(name.removesuffix(".weight") + ".weight_scale", begin * blocks, end * blocks)
                .view(torch.uint8)
                .reshape(end - begin, blocks)
            )
            if bool((((scales & 0x80) != 0) | (scales == 0x7F)).any()):
                raise ValueError(f"{name}: n-gram block scales must be nonnegative finite E4M3FN words")
            return torch.cat((codes, scales), dim=1)
        values = self.store.read_flat(name, begin * self.width, end * self.width)
        if self.keep_bf16:
            return values.contiguous().view(torch.uint8).reshape(end - begin, self.row_bytes)
        if not self.bf16:
            return values.view(torch.uint8).reshape(end - begin, self.row_bytes)
        return self.quantize(values).reshape(end - begin, self.row_bytes)

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        """E4M3FN words (``uint8``) of BF16 ``values`` by the per-tensor rule and this table's scale."""

        quotient = values / self.scale()  # BF16 division: one rounding to BF16, as ModelOpt
        codes = quotient.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)  # BF16 -> FP32 is exact
        return codes.view(torch.uint8)

    def bf16_rows_match(self, name: str, begin: int, end: int, rows: torch.Tensor) -> torch.Tensor:
        """Per row, whether BF16 volume rows (``uint8 [end - begin, 2 x width]``) are rows [begin, end) of
        shard ``name`` of this table: a BF16 table's words, or values whose per-tensor quantization is
        an FP8 table's words (the FP8 checkpoint of the same BF16 table: the rule ModelOpt applies)."""

        if self.nvfp4:
            raise ValueError("an NVFP4 table binds its own rows; give a BF16 volume at run time with --ngram-volume")
        values = rows.contiguous().view(torch.bfloat16).reshape(end - begin, self.width)
        stored = self.store.read_flat(name, begin * self.width, end * self.width).reshape(end - begin, self.width)
        if self.bf16:
            return (values.view(torch.int16) == stored.contiguous().view(torch.int16)).all(dim=1)
        return (self.quantize(values) == stored.view(torch.uint8)).all(dim=1)


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str | Path] | None = None,
    ngram_format: str | None = None,
) -> Model:
    """The logical model. The n-gram table binds as the checkpoint's own rows: FP8 rows with their
    per-tensor scale (an FP8 or BF16 table, ``ngram_format`` "fp8_e4m3fn"), or an NVFP4 table's rows
    with its FP32 multiplier ("nvfp4"); "bf16" binds a BF16 table's unquantized rows instead (a BF16
    volume, ``--ngram-reuse``). None selects the checkpoint's own format."""

    selected = set(components)
    if "text" not in selected or selected - {"text", "vision", "mtp"}:
        raise ValueError("select text and optional vision and mtp components")
    config = text_config(base.config, mtp="mtp" in selected)
    text_prefix = "model.language_model." if "text_config" in base.config else "model."
    (ple_layer,) = config["ple_layer_ids"]
    ple_prefix = f"{text_prefix}layers.{ple_layer - 1}."
    rows = _check_ngram_buffers(base, ple_prefix, config, 0)
    table = NgramTable(base, ple_prefix, config, rows)
    own = table.volume_format
    if ngram_format is None:
        ngram_format = own
    allowed = ("nvfp4",) if table.nvfp4 else ("fp8_e4m3fn", "bf16")
    if ngram_format not in allowed:
        raise ValueError(f"n-gram table format {ngram_format!r}: this checkpoint's {own} table binds as {allowed}")
    config["ngram_table"] = {
        "format": ngram_format,
        **ngram_geometry(rows, 2 * table.width if ngram_format == "bf16" else table.row_bytes),
    }
    records = {"text": {"config": config}}
    if "vision" in selected:
        vision = vision_config(base.config, config)
        vision["model_type"] = "qwen4_exp_vision"
        records["vision"] = {"config": vision, "target": "text"}
    if "mtp" in selected:
        records["mtp"] = {"config": {"architectures": ["Qwen4ExpMTP"]}, "target": "text"}
    refs, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        vision_config=records["vision"]["config"] if "vision" in selected else None,
        overrides=resource_overrides,
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Qwen4ExpBuilder(model)
    builder.ngram = table
    h, r = config["hidden_size"], config["vocab_size"]
    builder.add("text/token_embedding", base, text_prefix + "embed_tokens.weight", (r, h))
    heads = ("text/final_hidden",) + (("mtp/final_hidden",) if "mtp" in selected else ())
    builder.add("text/output_head", base, "lm_head.weight", (r, h), inputs=heads)
    builder.hyper_connection("text/final_mixer/", text_prefix + "hyper_connection_mixer.", base, config,
                             combine=False)
    for i, kind in enumerate(config["layer_types"]):
        prefix, source_prefix = f"text/layers/{i}/", f"{text_prefix}layers.{i}."
        if i + 1 == ple_layer:
            builder.ple(prefix, source_prefix, base, config)
        builder.block(prefix, source_prefix, base, config, kind)
    if "mtp" in selected:
        width = config["hc_count"] * h
        builder.add("mtp/embedding_norm", base, "mtp.pre_fc_norm_embedding.weight", (h,))
        builder.add("mtp/hidden_norm", base, "mtp.pre_fc_norm_hidden.weight", (width,))
        builder.add("mtp/embedding_projection", base, "mtp.fc_embedding.weight", (h, h),
                    inputs=("mtp/embedding_input",))
        builder.add("mtp/hidden_projection", base, "mtp.fc_hidden.weight", (h, h), inputs=("mtp/hidden_input",))
        builder.hyper_connection("mtp/final_mixer/", "mtp.hyper_connection_mixer.", base, config, combine=False)
        builder.block("mtp/layers/0/", "mtp.layers.0.", base, config, "full_attention", mtp=True)
    if "vision" in selected:
        builder.vision(base, records["vision"]["config"], h)
    return model


DFLASH2_TARGET_FIELDS = ("architectures", "hidden_size", "vocab_size", "num_hidden_layers")


def build_dflash2_companion(base: SafetensorsSource, draft: SafetensorsSource) -> Model:
    """A companion artifact holding only a DFlash2 drafter of a Qwen4Exp model, loaded beside any
    artifact of that model (``--draft-model``). ``base`` supplies the target's config and tokenizer
    (no weights are read from it); the drafter reuses the target's token embedding and output or
    proposal head, so the companion declares its Uses of those target parameters. Its config
    records the target identity the loader checks (``target_config``)."""

    from .qwen3_5 import draft_config

    target = text_config(base.config, mtp=False)
    config = draft_config(draft.config, target, "dflash2")
    config["target_config"] = {key: target[key] for key in DFLASH2_TARGET_FIELDS}
    _, _, count, special = load_resources(base.root, vocab_size=target["vocab_size"])
    if config["dflash_config"]["selector_top_k"] > count:
        raise ValueError("DFlash2 selector_top_k exceeds public token domain")
    model = Model(
        {"dflash2": {"config": config, "target": "text"}},
        token_count=count,
        special_token_ids=special,
        target=target,
        external_uses=(
            ("text/output_head", "dflash2/final_hidden"),
            ("proposal/head", "dflash2/final_hidden"),
        ),
    )
    _Qwen4ExpBuilder(model).draft(draft, config, target, "dflash2")
    return model


def ngram_table(base: SafetensorsSource, config: dict) -> NgramTable:
    """The n-gram table of a model built by :func:`build_model`."""

    text_prefix = "model.language_model." if "text_config" in base.config else "model."
    (ple_layer,) = config["ple_layer_ids"]
    return NgramTable(base, f"{text_prefix}layers.{ple_layer - 1}.", config, config["ngram_table"]["rows"])


# ---------------------------------------------------------------------------- n-gram volume

NGRAM_MAGIC = b"NINFERNG"
NGRAM_VERSION = 1
NGRAM_BLOCK_BYTES = 4096
NGRAM_HEADER = struct.Struct("<8sIIQIIIQ16s")  # magic, version, header bytes, rows, row bytes,
#                                               rows per block, block bytes, blocks, volume id
# An FP8 volume's per-tensor scale (BF16 bits) right after the fixed fields; 0 in volumes written
# before 2026-10-18. It lets the volume serve any artifact of the model (the artifact's own FP8 table
# keeps its ngram_scale; the engine knows the scale of NVIDIA's published volume by its id).
NGRAM_SCALE = struct.Struct("<H")
NGRAM_SCALE_OFFSET = NGRAM_HEADER.size
# The row format, explicit (volumes written since 2026-10-19; 0 before, when the row size alone told
# FP8, IQ4_NL and BF16 apart; NVFP4 rows have IQ4_NL's 90 bytes), and an NVFP4 volume's FP32
# multiplier (bits; 0 for other formats). Volumes published before stay unchanged.
NGRAM_FORMAT = struct.Struct("<H")
NGRAM_FORMAT_OFFSET = NGRAM_SCALE_OFFSET + NGRAM_SCALE.size
NGRAM_FORMAT_CODES = {"fp8_e4m3fn": 1, "iq4_nl": 2, "bf16": 3, "nvfp4": 4}
NGRAM_NVFP4_SCALE = struct.Struct("<I")
NGRAM_NVFP4_SCALE_OFFSET = NGRAM_FORMAT_OFFSET + NGRAM_FORMAT.size


def bf16_bits(value: torch.Tensor) -> int:
    """The BF16 word of a one-element BF16 tensor."""

    return int(value.reshape(1).view(torch.int16)[0]) & 0xFFFF


def ngram_geometry(rows: int, row_bytes: int) -> dict:
    rows_per_block = NGRAM_BLOCK_BYTES // row_bytes
    if rows_per_block == 0:
        raise ValueError("an n-gram row must fit one 4 KiB block")
    blocks = -(-rows // rows_per_block)
    return {
        "rows": rows,
        "row_bytes": row_bytes,
        "rows_per_block": rows_per_block,
        "block_bytes": NGRAM_BLOCK_BYTES,
        "header_bytes": NGRAM_BLOCK_BYTES,
        "blocks": blocks,
        "file_bytes": NGRAM_BLOCK_BYTES * (1 + blocks),
    }


NGRAM_REUSE_SAMPLES = 512


def read_ngram_volume_format(path: Path, width: int) -> str:
    """A volume's row format: its header's tag, or for a volume written before the tag, the one its
    row size implies (FP8 ``width``, BF16 ``2 x width``, IQ4_NL ``width / 32 x 18`` bytes)."""

    with Path(path).open("rb") as stream:
        header = stream.read(NGRAM_NVFP4_SCALE_OFFSET + NGRAM_NVFP4_SCALE.size)
    row_bytes = NGRAM_HEADER.unpack_from(header)[4]
    (tag,) = NGRAM_FORMAT.unpack_from(header, NGRAM_FORMAT_OFFSET)
    if tag:
        names = {code: name for name, code in NGRAM_FORMAT_CODES.items()}
        if tag not in names:
            raise ValueError(f"{path}: unknown n-gram row format tag {tag}")
        return names[tag]
    if row_bytes == 2 * width:
        return "bf16"
    if row_bytes == width:
        return "fp8_e4m3fn"
    if width % 32 == 0 and row_bytes == width // 32 * 18:
        return "iq4_nl"
    raise ValueError(f"{path}: n-gram rows of {row_bytes} bytes fit no table of this model")


def ngram_volume_format(base: SafetensorsSource, path: Path) -> str:
    """The row format of an existing volume of this checkpoint's table (read_ngram_volume_id checks
    the rest)."""

    config = text_config(base.config, mtp=False)
    width = config["ple_embed_dim"] // ((config["ngram_size"] - 1) * config["heads_per_ngram"])
    return read_ngram_volume_format(path, width)


def read_ngram_volume_id(table: NgramTable, path: Path, *, bf16: bool = False) -> bytes:
    """The volume id of an existing n-gram volume written from this table's words.

    Another recipe of the same checkpoint (recipe B) shares recipe A's volume, and so does a BF16
    export of the same table (recipe C; NgramTable quantizes it to the same words). The header
    geometry and the file size must match this table, and NGRAM_REUSE_SAMPLES rows spread over the
    table (the first and last included) must equal this table's rows byte for byte, so a volume
    written from another table with the same geometry is refused instead of adopted (the runtime
    only compares the id the artifact stores).

    With ``bf16`` the volume holds the unquantized table's BF16 rows (tools/flash_next/ngram_bf16.py):
    each sampled row must be a BF16 table's words, or quantize to an FP8 table's words
    (NgramTable.bf16_rows_match), so an FP8 checkpoint binds the BF16 table it was quantized from.
    """

    row_bytes = 2 * table.width if bf16 else table.row_bytes
    rows, shards = table.rows, table.shards
    geometry = ngram_geometry(rows, row_bytes)
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(NGRAM_HEADER.size)
    magic, version, header_bytes, stored_rows, stored_row_bytes, per_block, block_bytes, blocks, volume_id = (
        NGRAM_HEADER.unpack(header)
    )
    if (
        magic != NGRAM_MAGIC
        or version != NGRAM_VERSION
        or header_bytes != NGRAM_BLOCK_BYTES
        or stored_rows != rows
        or stored_row_bytes != row_bytes
        or per_block != geometry["rows_per_block"]
        or block_bytes != NGRAM_BLOCK_BYTES
        or blocks != geometry["blocks"]
        or path.stat().st_size != geometry["file_bytes"]
    ):
        raise ValueError(f"{path}: not an n-gram volume of this checkpoint's table")
    samples = sorted({rows - 1, *(k * rows // NGRAM_REUSE_SAMPLES for k in range(NGRAM_REUSE_SAMPLES))})
    starts = []
    first = 0
    for name, count in shards:
        starts.append((first, name, count))
        first += count
    with path.open("rb") as stream:
        for row in samples:
            shard_first, name, _ = next(s for s in reversed(starts) if s[0] <= row)
            local = row - shard_first
            per_block = geometry["rows_per_block"]
            stream.seek(NGRAM_BLOCK_BYTES + (row // per_block) * NGRAM_BLOCK_BYTES + (row % per_block) * row_bytes)
            stored = stream.read(row_bytes)
            if bf16:
                words = torch.frombuffer(bytearray(stored), dtype=torch.uint8).reshape(1, row_bytes)
                same = bool(table.bf16_rows_match(name, local, local + 1, words)[0])
            else:
                same = stored == table.codes(name, local, local + 1).numpy().tobytes()
            if not same:
                raise ValueError(f"{path}: row {row} differs from this checkpoint's n-gram table "
                                 "(a volume written from another checkpoint)")
    if getattr(table, "nvfp4", False) and not bf16:
        # An NVFP4 volume's rows are only the table with its multiplier, which its header records.
        expected = table.nvfp4_scale().view(torch.int32).item() & 0xFFFFFFFF
        with path.open("rb") as stream:
            fields = stream.read(NGRAM_NVFP4_SCALE_OFFSET + NGRAM_NVFP4_SCALE.size)
        tag = NGRAM_FORMAT.unpack_from(fields, NGRAM_FORMAT_OFFSET)[0]
        bits = NGRAM_NVFP4_SCALE.unpack_from(fields, NGRAM_NVFP4_SCALE_OFFSET)[0]
        if tag != NGRAM_FORMAT_CODES["nvfp4"] or bits != expected:
            raise ValueError(f"{path}: not this checkpoint's NVFP4 volume (format tag or multiplier differs)")
    return volume_id


def write_ngram_volume(table: NgramTable, path: Path, volume_id: bytes, *, progress=None) -> dict:
    """Write the FP8 n-gram rows, 25 per 4 KiB block and never straddling one (design §12.2).

    Row ``r`` lives at ``header_bytes + (r // rows_per_block) * 4096 + (r % rows_per_block) * row_bytes``;
    each block's tail is zero. The codes are the table's words (NgramTable: the checkpoint's FP8 words
    unchanged, or a BF16 table's per-tensor quantization).
    """

    if len(volume_id) != 16:
        raise ValueError("the n-gram volume id is 16 bytes")
    row_bytes, rows, shards = table.row_bytes, table.rows, table.shards
    geometry = ngram_geometry(rows, row_bytes)
    per_block = geometry["rows_per_block"]
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"n-gram volume already exists: {path}")
    temporary = path.with_name(path.name + ".tmp")
    header = NGRAM_HEADER.pack(
        NGRAM_MAGIC, NGRAM_VERSION, NGRAM_BLOCK_BYTES, rows, row_bytes, per_block, NGRAM_BLOCK_BYTES,
        geometry["blocks"], volume_id,
    )
    # The explicit row format; FP8 rows record their scale and NVFP4 rows their multiplier (a BF16 or
    # IQ4_NL volume's rows are self-contained).
    format = table.volume_format
    header += NGRAM_SCALE.pack(bf16_bits(table.scale()) if format == "fp8_e4m3fn" else 0)
    header += NGRAM_FORMAT.pack(NGRAM_FORMAT_CODES[format])
    header += NGRAM_NVFP4_SCALE.pack(
        table.nvfp4_scale().view(torch.int32).item() & 0xFFFFFFFF if format == "nvfp4" else 0
    )
    batch_blocks = 4096  # 16 MiB of output per write
    try:
        with temporary.open("xb") as stream:
            stream.write(header + bytes(NGRAM_BLOCK_BYTES - len(header)))
            pending = torch.empty(0, row_bytes, dtype=torch.uint8)
            written_rows = 0

            def flush(rows_tensor: torch.Tensor, final: bool) -> torch.Tensor:
                nonlocal written_rows
                whole = rows_tensor.shape[0] // per_block
                if final and rows_tensor.shape[0] % per_block:
                    whole += 1
                    pad = whole * per_block - rows_tensor.shape[0]
                    rows_tensor = torch.cat((rows_tensor, torch.zeros(pad, row_bytes, dtype=torch.uint8)))
                if whole == 0:
                    return rows_tensor
                used = rows_tensor[: whole * per_block].reshape(whole, per_block * row_bytes)
                blocks = torch.zeros(whole, NGRAM_BLOCK_BYTES, dtype=torch.uint8)
                blocks[:, : per_block * row_bytes] = used
                stream.write(blocks.numpy().tobytes())
                written_rows += whole * per_block
                return rows_tensor[whole * per_block :]

            for index, (name, count) in enumerate(shards):
                step = batch_blocks * per_block
                for begin in range(0, count, step):
                    end = min(count, begin + step)
                    pending = flush(torch.cat((pending, table.codes(name, begin, end))), False)
                if progress is not None:
                    progress(index, len(shards))
            flush(pending, True)
        if temporary.stat().st_size != geometry["file_bytes"]:
            raise ValueError("n-gram volume size differs from its geometry")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return geometry
