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

from tools.artifact.codecs.nvfp4_expert import encode_nvfp4_expert_bank
from tools.artifact.layouts import expert_bank_geometry
from tools.flash_next.ngram import NgramConfig, head_tables, layer_multipliers

from .methods import PrepareRequest, PreparedMethod
from .model import Model, Parameter
from .qwen3_5 import _Builder, _f32, _fixed, _positive, _rope_source, vision_config
from .resources import load_resources
from .sources.logical import LogicalSource
from .sources.modelopt import dequantize_fp8_block128, fp8_block_matrix_words, nvfp4_matrix_words
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
    if raw.get("ple_embedding_dtype", "float8_e4m3fn") != "float8_e4m3fn":
        raise ValueError("this adapter imports the FP8 n-gram table")

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


@dataclass(frozen=True, slots=True)
class ExpertBankSource:
    """One layer's ModelOpt NVFP4 experts; only ``import_expert_bank`` can read it."""

    shape: tuple[int, int, int]
    label: str
    store: SafetensorsSource
    prefix: str  # "...mlp.experts." so expert e's gate is f"{prefix}{e}.gate_proj"
    read_encoded: object = None
    weight_divisor: object = None
    input_divisor: object = None

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


_BANK_CHUNK = 32  # experts encoded per step; bounds conversion memory at ~0.4 GB


def import_expert_bank(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's experts bit-exactly in ``nvfp4_expert_rg16_v1`` (design §6.2)."""

    if request.target.format != "nvfp4_mul" or request.target.layout != "nvfp4_expert_rg16_v1":
        raise ValueError("import_expert_bank requires nvfp4_mul in nvfp4_expert_rg16_v1")
    if request.parameters:
        raise ValueError("import_expert_bank accepts no numerical parameters")
    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, ExpertBankSource):
        raise ValueError("import_expert_bank requires exactly one expert-bank source")
    source = request.inputs[0].source
    if tuple(request.target.shape) != source.shape:
        raise ValueError("expert bank target shape differs from its source")
    experts, hidden, intermediate = source.shape
    geometry = expert_bank_geometry("nvfp4_mul", source.shape)

    def produce(output):
        multipliers = []
        for first in range(0, experts, _BANK_CHUNK):
            count = min(_BANK_CHUNK, experts - first)
            gc, gs, uc, us, dc, ds, mult = [], [], [], [], [], [], []
            for e in range(first, first + count):
                gate = nvfp4_matrix_words(source.store, f"{source.prefix}{e}.gate_proj", (intermediate, hidden))
                up = nvfp4_matrix_words(source.store, f"{source.prefix}{e}.up_proj", (intermediate, hidden))
                down = nvfp4_matrix_words(source.store, f"{source.prefix}{e}.down_proj", (hidden, intermediate))
                gc.append(gate.codes)
                gs.append(gate.scales)
                uc.append(up.codes)
                us.append(up.scales)
                dc.append(down.codes)
                ds.append(down.scales)
                mult.append(torch.stack((gate.weight_scale_2, up.weight_scale_2, down.weight_scale_2)))
            part = encode_nvfp4_expert_bank(
                torch.stack(gc), torch.stack(gs), torch.stack(uc), torch.stack(us),
                torch.stack(dc), torch.stack(ds), torch.stack(mult), (count, hidden, intermediate),
            )
            output.write_bytes(first * geometry.record_stride, memoryview(part)[: count * geometry.record_stride])
            multipliers.append(torch.stack(mult))
        plane = torch.cat(multipliers).contiguous()
        output.write_bytes(geometry.multiplier_offset, plane.numpy().tobytes())

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

    def hyper_connection(self, prefix, source_prefix, store, config, *, combine=True):
        width = config["hc_count"] * config["hidden_size"]
        rank = config["hc_lowrank"]
        self.add(prefix + "norm", store, source_prefix + "hc_norm.weight", (width,))
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
            # NVIDIA stores the MTP experts per expert in Qwen's block-scaled FP8 (E4M3FN codes, one
            # FP32 multiplier per 128 x 128 tile). Each expert's matrices become row ranges of two
            # parents (gate and up interleaved per expert, then every down), so the drafter's
            # resident-expert kernel selects them by id; their values are the exact products.
            gate_up, downs = [], []
            for expert in range(e):
                ep, leaf = p + f"experts/{expert}/", sp + f"experts.{expert}."
                for role in ("gate", "up"):
                    self.add_fp8_block(ep + role, store, leaf + role + "_proj", (ir, h), inputs=(ffn_input,))
                    gate_up.append(ep + role)
                self.add_fp8_block(ep + "down", store, leaf + "down_proj", (h, ir), inputs=(ep + "product",))
                downs.append(ep + "down")
            self.group(*gate_up)
            self.group(*downs)
            return
        bank = ExpertBankSource((e, h, ir), f"{store.path}:{sp}experts", store, sp + "experts.")
        self.model.add(Parameter(p + "experts", (e, h, ir), bank, None, (ffn_input,), "nvfp4_mul", residency="text"))
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
            self.add(prefix + "ple/" + role, store, sp + field + ".weight", (width,))
        self.add(prefix + "ple/convolution", store, sp + "conv1d.weight", (taps, width),
                 source_shape=(width, 1, taps), transpose=(2, 0, 1))
        self.add(prefix + "ple/ngram_scale", store, sp + "ple_embedding.ngram_embedding.weight_scale", (1,))

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


def ngram_shards(store: SafetensorsSource, prefix: str, config: dict, rows: int) -> list[tuple[str, int]]:
    """``(tensor, rows)`` for every table shard in row order; shards concatenate along rows."""

    parts = config["split_ngram_parts"]
    width = config["ple_embed_dim"] // ((config["ngram_size"] - 1) * config["heads_per_ngram"])
    out, total = [], 0
    for index in range(parts):
        name = f"{prefix}ple.ple_embedding.ngram_embedding.shard_{index}.weight"
        info = store.describe(name)
        if info.dtype != "F8_E4M3" or len(info.shape) != 2 or info.shape[1] != width:
            raise ValueError(f"{name}: expected F8_E4M3 [rows, {width}]")
        out.append((name, info.shape[0]))
        total += info.shape[0]
    if total != rows:
        raise ValueError(f"n-gram shards hold {total} rows, the hash domain needs {rows}")
    return out


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = set(components)
    if "text" not in selected or selected - {"text", "vision", "mtp"}:
        raise ValueError("select text and optional vision and mtp components")
    config = text_config(base.config, mtp="mtp" in selected)
    text_prefix = "model.language_model." if "text_config" in base.config else "model."
    (ple_layer,) = config["ple_layer_ids"]
    ple_prefix = f"{text_prefix}layers.{ple_layer - 1}."
    rows = _check_ngram_buffers(base, ple_prefix, config, 0)
    shards = ngram_shards(base, ple_prefix, config, rows)
    config["ngram_table"] = {
        "format": "fp8_e4m3fn",
        **ngram_geometry(rows, base.describe(shards[0][0]).shape[1]),
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


def ngram_volume_shards(base: SafetensorsSource, config: dict) -> list[tuple[str, int]]:
    """The table shards of a model built by :func:`build_model`, in row order."""

    text_prefix = "model.language_model." if "text_config" in base.config else "model."
    (ple_layer,) = config["ple_layer_ids"]
    return ngram_shards(base, f"{text_prefix}layers.{ple_layer - 1}.", config, config["ngram_table"]["rows"])


# ---------------------------------------------------------------------------- n-gram volume

NGRAM_MAGIC = b"NINFERNG"
NGRAM_VERSION = 1
NGRAM_BLOCK_BYTES = 4096
NGRAM_HEADER = struct.Struct("<8sIIQIIIQ16s")  # magic, version, header bytes, rows, row bytes,
#                                               rows per block, block bytes, blocks, volume id


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


def read_ngram_volume_id(store: SafetensorsSource, shards, path: Path) -> bytes:
    """The volume id of an existing n-gram volume written from this checkpoint's table.

    Another recipe of the same checkpoint (recipe B) shares recipe A's volume: the rows are the
    checkpoint's words in both. The header geometry and the file size must match this table.
    """

    row_bytes = store.describe(shards[0][0]).shape[1]
    rows = sum(count for _, count in shards)
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
    return volume_id


def write_ngram_volume(store: SafetensorsSource, shards, path: Path, volume_id: bytes, *, progress=None) -> dict:
    """Write the FP8 n-gram rows, 25 per 4 KiB block and never straddling one (design §12.2).

    Row ``r`` lives at ``header_bytes + (r // rows_per_block) * 4096 + (r % rows_per_block) * row_bytes``;
    each block's tail is zero. The codes are the checkpoint's words unchanged.
    """

    if len(volume_id) != 16:
        raise ValueError("the n-gram volume id is 16 bytes")
    row_bytes = store.describe(shards[0][0]).shape[1]
    rows = sum(count for _, count in shards)
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
                    chunk = store.read_flat(name, begin * row_bytes, end * row_bytes).view(torch.uint8)
                    pending = flush(torch.cat((pending, chunk.reshape(end - begin, row_bytes))), False)
                if progress is not None:
                    progress(index, len(shards))
            flush(pending, True)
        if temporary.stat().st_size != geometry["file_bytes"]:
            raise ValueError("n-gram volume size differs from its geometry")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return geometry
