"""Qwen4Exp (Qwen3.8-Flash-Next) from a llama.cpp GGUF, imported bit-exactly.

The logical model is the one ``qwen4_exp.build_model`` describes (the same parameter names and
shapes), with every source read from the GGUF instead of the Hugging Face checkpoint: quantized
tensors expose their stored block rows (``read_encoded``) so recipes copy them verbatim, and plain
tensors (BF16, F32, F16) their exact values. llama.cpp's export transforms are undone exactly and
checked (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25):

* ``+1`` on the zero-centred RMSNorm gammas: g - 1 must be a BF16 value (checked per tensor);
* the GDN value heads in ggml's tiled order: rows are permuted back (whole rows, exact); the
  output projection's *columns* cannot be (its K-blocks span two heads), so ``ssm_out`` stays in
  GGUF order and the text config records ``gdn_output_order: ggml_tiled`` for the runtime;
* ``A_log`` stored as ``-exp(A_log)``: kept as ``gdn/a`` (FP32 A), used directly by the runtime;
* the indexer's fused q/k projection split in two: concatenated again as the index parent.

The GGUF holds no MTP head; with the ``mtp`` component its tensors come from the ``--model``
checkpoint (Qwen's), built exactly as recipe A builds them. The config, tokenizer and frontend
resources also come from that checkpoint; the GGUF metadata must agree with it.

With the ``vision`` component the vision tower comes from the GGUF repository's BF16 ``mmproj``
(llama.cpp's ``clip`` export of the same tower): ``sources.gguf.qwen35_vision_name_map`` names its
tensors (the Conv3d patch kernel reassembled from its two temporal slices), its metadata must agree
with the checkpoint's ``vision_config``, and every value, F32 companions included, must be a BF16
value, so the BF16 bindings hold exactly what the mmproj holds.
"""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from tools.artifact.codecs.ggml import (BY_GGML_ID, GGML_TYPES, GGUF_BF16, GGUF_F16, GGUF_F32, GGUF_Q8_0, decode_q8_0,
                                        decode_values, require_finite_scales)
from tools.artifact.layouts import ggml_expert_geometry
from tools.flash_next.ngram import head_tables, layer_multipliers

from . import qwen4_exp
from .methods import PrepareRequest, PreparedMethod
from .model import Model, Parameter
from .qwen3_5 import vision_config
from .resources import load_resources
from .sources.gguf_raw import GgufStore
from .sources.logical import EncodedRows, LogicalSource, select_rows, transpose_source
from .sources.safetensors import SafetensorsSource

NGRAM_TABLE = "per_layer_token_embd.weight"
IQ4_NL_ROW_FORMAT = "iq4_nl"


# ---------------------------------------------------------------------------- sources


def ggml_matrix(store: GgufStore, name: str, shape: tuple[int, int]) -> LogicalSource:
    """A quantized 2-D GGUF tensor: exact values, and its stored rows as EncodedRows."""

    t = store.tensor(name)
    if t.shape != tuple(shape):
        raise ValueError(f"{name}: GGUF shape {t.shape} differs from {shape}")
    k = shape[1]

    if t.ggml_type == GGUF_Q8_0:
        def encoded(begin: int, end: int) -> EncodedRows:
            codes, scales = decode_q8_0(np.ascontiguousarray(store.raw(name, begin, end)), end - begin, k)
            if (codes == -128).any():
                raise ValueError(f"{name}: Q8_0 code -128 has no q8_g32_fp16 encoding")
            return EncodedRows("q8_g32_fp16", torch.from_numpy(codes.copy()).reshape(end - begin, k // 32, 32),
                               torch.from_numpy(scales.copy()).view(torch.float16))

        def values(begin: int, end: int) -> torch.Tensor:
            rows = encoded(begin // k, -(-end // k))
            v = rows.codes.to(torch.float64) * rows.scales.to(torch.float64)[..., None]
            flat = v.reshape(-1)
            start = begin - begin // k * k
            return flat[start : start + end - begin]

        return LogicalSource(tuple(shape), f"{store.path}:{name}", values, encoded)

    if t.ggml_type not in BY_GGML_ID:
        raise ValueError(f"{name}: GGUF type {t.ggml_type} is not a ggml block format this importer stores")
    fmt = BY_GGML_ID[t.ggml_type]

    def encoded(begin: int, end: int) -> EncodedRows:
        rows = np.array(store.raw(name, begin, end))  # a private copy of the stored bytes
        return EncodedRows(fmt.name, torch.from_numpy(rows), torch.empty(0, dtype=torch.uint8))

    def values(begin: int, end: int) -> torch.Tensor:
        first, last = begin // k, -(-end // k)
        rows = np.ascontiguousarray(store.raw(name, first, last))
        flat = torch.from_numpy(decode_values(fmt, rows, last - first, k)).reshape(-1)
        start = begin - first * k
        return flat[start : start + end - begin]

    return LogicalSource(tuple(shape), f"{store.path}:{name}", values, encoded)


def plain(store: GgufStore, name: str, shape: tuple[int, ...], *, minus_one: bool = False,
          bf16_exact: bool = True) -> LogicalSource:
    """An F32/F16/BF16 GGUF tensor's exact values (binary64), optionally g - 1; when ``bf16_exact``
    every value must be a BF16 value, so a BF16 binding stores exactly what the GGUF holds."""

    t = store.tensor(name)
    if t.ggml_type not in (GGUF_F32, GGUF_F16, GGUF_BF16):
        raise ValueError(f"{name}: expected an F32, F16 or BF16 tensor")
    if prod(t.shape) != prod(shape):
        raise ValueError(f"{name}: GGUF shape {t.shape} differs from {shape}")
    cache: list[torch.Tensor] = []

    def read(begin: int, end: int) -> torch.Tensor:
        if not cache:
            v = torch.from_numpy(store.plain(name).reshape(-1).astype(np.float64))
            if minus_one:
                v = v - 1.0  # exact in binary64
            if bf16_exact and not torch.equal(v.to(torch.bfloat16).to(torch.float64), v):
                raise ValueError(f"{name}: a value is not exact in BF16" + (" after undoing +1" if minus_one else ""))
            cache.append(v)
        return cache[0][begin:end]

    return LogicalSource(tuple(shape), f"{store.path}:{name}" + (" - 1" if minus_one else ""), read)


def head_rows(source: LogicalSource, heads: int, rows_per_head: int, key_heads: int, row_offset: int = 0) -> LogicalSource:
    """Rows of ``heads`` value heads in Hugging Face order from ggml's tiled order.

    Hugging Face head h = kk * R + r (key head kk, R value heads per key head) is ggml slot
    g = r * key_heads + kk (llama.cpp ``_LinearAttentionVReorderBase``; its graph pairs value slot g
    with key head g % key_heads)."""

    per_key = heads // key_heads
    ranges = []
    for h in range(heads):
        g = (h % per_key) * key_heads + h // per_key
        first = row_offset + g * rows_per_head
        ranges.append((first, first + rows_per_head))
    return select_rows(source, tuple(ranges))


def permute_vector(source: LogicalSource, heads: int, key_heads: int) -> LogicalSource:
    """A per-value-head vector (one element per head) in Hugging Face order."""

    per_key = heads // key_heads
    order = [(h % per_key) * key_heads + h // per_key for h in range(heads)]

    def read(begin: int, end: int) -> torch.Tensor:
        return source.values()[order][begin:end]

    return LogicalSource(source.shape, f"heads({source.label})", read)


# ---------------------------------------------------------------------------- expert banks


class GgmlExpertBank:
    """One layer's routed experts in the GGUF; only ``import_ggml_expert_bank`` reads it."""

    def __init__(self, store: GgufStore, layer: int, shape: tuple[int, int, int]):
        self.store = store
        self.shape = shape
        self.label = f"{store.path}:blk.{layer}.ffn_*_exps"
        experts, hidden, inter = shape
        self.gate = store.tensor(f"blk.{layer}.ffn_gate_exps.weight")
        self.up = store.tensor(f"blk.{layer}.ffn_up_exps.weight")
        self.down = store.tensor(f"blk.{layer}.ffn_down_exps.weight")
        if self.gate.dims != (hidden, inter, experts) or self.up.dims != self.gate.dims:
            raise ValueError(f"{self.label}: gate/up dims {self.gate.dims} differ from [{hidden}, {inter}, {experts}]")
        if self.down.dims != (inter, hidden, experts):
            raise ValueError(f"{self.label}: down dims {self.down.dims} differ from [{inter}, {hidden}, {experts}]")
        if self.gate.ggml_type != self.up.ggml_type or self.gate.ggml_type not in BY_GGML_ID or self.down.ggml_type not in BY_GGML_ID:
            raise ValueError(f"{self.label}: gate and up must share one ggml format")
        self.gate_up_format = BY_GGML_ID[self.gate.ggml_type].name
        self.down_format = BY_GGML_ID[self.down.ggml_type].name

    @property
    def layout(self) -> str:
        return "ggml_expert_" + self.down_format.removeprefix("ggml_") + "_v1"

    def values(self, begin: int = 0, end: int | None = None) -> torch.Tensor:
        raise ValueError(f"{self.label}: an expert bank is imported exactly, never as values")


_BANK_CHUNK = 32


def import_ggml_expert_bank(request: PrepareRequest) -> PreparedMethod:
    """Write one layer's experts as ``ggml_expert_*_v1`` records: per expert its gate rows, up rows
    and down rows, each the GGUF's stored blocks, the record padded to 4 KiB."""

    if len(request.inputs) != 1 or not isinstance(request.inputs[0].source, GgmlExpertBank):
        raise ValueError("import_ggml_expert_bank requires one GGUF expert bank")
    bank = request.inputs[0].source
    if request.parameters:
        raise ValueError("import_ggml_expert_bank accepts no numerical parameters")
    if request.target.format != bank.gate_up_format or request.target.layout != bank.layout:
        raise ValueError(f"{bank.label}: target {request.target.format}/{request.target.layout} differs from the bank's "
                         f"{bank.gate_up_format}/{bank.layout}")
    if tuple(request.target.shape) != bank.shape:
        raise ValueError("expert bank target shape differs from its source")
    g = ggml_expert_geometry(bank.layout, bank.gate_up_format, bank.shape)
    experts, hidden, inter = bank.shape
    store = bank.store
    gate_up, down = GGML_TYPES[bank.gate_up_format], GGML_TYPES[bank.down_format]

    def produce(output):
        pad = bytes(g.record_stride - g.record_bytes)
        for first in range(0, experts, _BANK_CHUNK):
            count = min(_BANK_CHUNK, experts - first)
            part = bytearray()
            for e in range(first, first + count):
                for name, fmt, rows, k in ((bank.gate.name, gate_up, inter, hidden), (bank.up.name, gate_up, inter, hidden),
                                           (bank.down.name, down, hidden, inter)):
                    raw = store.raw(name, e * rows, (e + 1) * rows)
                    require_finite_scales(fmt, raw, rows, k, f"{name} expert {e}")
                    part += raw.tobytes()
                part += pad
            output.write_bytes(first * g.record_stride, part)

    return request.job(produce=produce)


# ---------------------------------------------------------------------------- builder


def _metadata_checks(store: GgufStore, config: dict, rows: int) -> None:
    m = store.metadata
    if m.get("general.architecture") != "qwen4exp":
        raise ValueError(f"{store.path}: GGUF architecture {m.get('general.architecture')!r} is not qwen4exp")
    expected = {
        "qwen4exp.block_count": config["num_hidden_layers"],
        "qwen4exp.embedding_length": config["hidden_size"],
        "qwen4exp.attention.head_count": config["num_attention_heads"],
        "qwen4exp.attention.head_count_kv": config["num_key_value_heads"],
        "qwen4exp.attention.key_length": config["head_dim"],
        "qwen4exp.expert_count": config["num_experts"],
        "qwen4exp.expert_used_count": config["num_experts_per_tok"],
        "qwen4exp.expert_feed_forward_length": config["moe_intermediate_size"],
        "qwen4exp.expert_shared_feed_forward_length": config["shared_expert_intermediate_size"],
        "qwen4exp.ssm.group_count": config["linear_num_key_heads"],
        "qwen4exp.ssm.time_step_rank": config["linear_num_value_heads"],
        "qwen4exp.ssm.state_size": config["linear_key_head_dim"],
        "qwen4exp.hyper_connection.count": config["hc_count"],
        "qwen4exp.hyper_connection.low_rank": config["hc_lowrank"],
        "qwen4exp.attention.indexer.head_count": config["indexer_n_heads"],
        "qwen4exp.attention.indexer.key_length": config["indexer_head_dim"],
        "qwen4exp.attention.indexer.top_k": config["indexer_budget"],
        "qwen4exp.ple.ngram_size": config["ngram_size"],
        "qwen4exp.ple.heads_per_ngram": config["heads_per_ngram"],
        "qwen4exp.ple.conv_kernel": config["ple_conv_kernel_size"],
    }
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f"{store.path}: {key} = {m.get(key)!r}, the checkpoint config says {value!r}")
    if m.get("qwen4exp.ple.layers") != [config["ple_layer_ids"][0] - 1]:
        raise ValueError(f"{store.path}: PLE layers {m.get('qwen4exp.ple.layers')} differ from the config")
    spec = qwen4_exp.ngram_config(config)
    sizes, offsets, padded = head_tables(spec, 0)
    if (m.get("qwen4exp.ple.layer_multipliers") != layer_multipliers(spec, 0)
            or m.get("qwen4exp.ple.head_vocab_sizes") != sizes or m.get("qwen4exp.ple.head_offsets") != offsets):
        raise ValueError(f"{store.path}: the PLE hash constants differ from the config's derivation")
    if padded != rows:
        raise ValueError(f"{store.path}: the n-gram table has {rows} rows, the hash domain needs {padded}")


class _Builder:
    def __init__(self, model: Model, store: GgufStore, config: dict):
        self.model = model
        self.store = store
        self.config = config

    def add(self, name, source, *, inputs=(), direct="bf16"):
        self.model.add(Parameter(name, tuple(source.shape), source, None, tuple(inputs), direct,
                                 residency=name.split("/", 1)[0]))

    def matrix(self, name, tensor, shape, *, inputs=()):
        self.add(name, ggml_or_plain(self.store, tensor, shape), inputs=inputs)

    def hyper_connection(self, prefix, gguf_prefix, *, combine=True):
        c = self.config
        width, rank = c["hc_count"] * c["hidden_size"], c["hc_lowrank"]
        self.add(prefix + "norm", plain(self.store, gguf_prefix + "norm.weight", (width,), minus_one=True))
        self.matrix(prefix + "down", gguf_prefix + "down.weight", (rank, width), inputs=(prefix + "input",))
        self.matrix(prefix + "up", gguf_prefix + "up.weight", (width, rank), inputs=(prefix + "mix_activation",))
        if combine:
            self.matrix(prefix + "inject", gguf_prefix + "inject.weight", (c["hc_count"], width), inputs=(prefix + "input",))
            self.model.packing_groups.append((prefix + "down", prefix + "inject"))

    def gdn(self, prefix, blk):
        c = self.config
        h = c["hidden_size"]
        nk, dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
        nv, dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
        kg, vg, taps = nk * dk, nv * dv, c["linear_conv_kernel_dim"]
        channels = 2 * kg + vg
        p, inputs = prefix + "gdn/", (prefix + "mixer_input",)
        qkv = ggml_or_plain(self.store, blk + "attn_qkv.weight", (channels, h))
        self.add(p + "query", select_rows(qkv, ((0, kg),)), inputs=inputs)
        self.add(p + "key", select_rows(qkv, ((kg, 2 * kg),)), inputs=inputs)
        self.add(p + "value", head_rows(qkv, nv, dv, nk, row_offset=2 * kg), inputs=inputs)
        self.add(p + "z", head_rows(ggml_or_plain(self.store, blk + "attn_gate.weight", (vg, h)), nv, dv, nk), inputs=inputs)
        for role, tensor in (("a_projection", "ssm_alpha.weight"), ("b_projection", "ssm_beta.weight")):
            self.add(p + role, head_rows(ggml_or_plain(self.store, blk + tensor, (nv, h)), nv, 1, nk), inputs=inputs)
        self.add(p + "a", permute_vector(plain(self.store, blk + "ssm_a", (nv,), bf16_exact=False), nv, nk), direct="fp32")
        self.add(p + "dt_bias", permute_vector(plain(self.store, blk + "ssm_dt.bias", (nv,), bf16_exact=False), nv, nk),
                 direct="fp32")
        conv = plain(self.store, blk + "ssm_conv1d.weight", (channels, taps))
        conv = select_rows(conv, ((0, 2 * kg),) + tuple(
            (2 * kg + ((h_ % (nv // nk)) * nk + h_ // (nv // nk)) * dv,
             2 * kg + ((h_ % (nv // nk)) * nk + h_ // (nv // nk)) * dv + dv) for h_ in range(nv)))
        self.add(p + "convolution", transpose_source(conv, (1, 0), (taps, channels)))
        self.add(p + "norm", plain(self.store, blk + "ssm_norm.weight", (dv,)))
        # Columns stay in ggml's tiled value-head order (gdn_output_order); see the module docstring.
        self.matrix(p + "output", blk + "ssm_out.weight", (h, vg), inputs=(p + "gated_output",))
        roles = [p + role for role in ("query", "key", "value", "z")]
        self.model.packing_groups.extend((tuple(roles), tuple(roles[:2]), tuple(roles[2:]),
                                          (p + "a_projection", p + "b_projection")))

    def qsa(self, prefix, blk):
        c = self.config
        h, d = c["hidden_size"], c["head_dim"]
        heads, kv = c["num_attention_heads"], c["num_key_value_heads"]
        q, k = heads * d, kv * d
        p, inputs = prefix + "attention/", (prefix + "mixer_input",)
        fused = ggml_or_plain(self.store, blk + "attn_q.weight", (2 * q, h))
        for role, gate in (("query", False), ("gate", True)):
            ranges = tuple((head * 2 * d + int(gate) * d, head * 2 * d + (int(gate) + 1) * d) for head in range(heads))
            self.add(p + role, select_rows(fused, ranges), inputs=inputs)
        self.matrix(p + "key", blk + "attn_k.weight", (k, h), inputs=inputs)
        self.matrix(p + "value", blk + "attn_v.weight", (k, h), inputs=inputs)
        iq = c["indexer_n_heads"] * c["indexer_head_dim"]
        self.matrix(p + "index_query", blk + "indexer.q_proj.weight", (iq, h), inputs=inputs)
        self.matrix(p + "index_key", blk + "indexer.k_proj.weight", (c["indexer_head_dim"], h), inputs=inputs)
        for role, tensor, width in (
            ("query_norm", "attn_q_norm.weight", d),
            ("key_norm", "attn_k_norm.weight", d),
            ("index_query_norm", "indexer.q_norm.weight", c["indexer_head_dim"]),
            ("index_key_norm", "indexer.k_norm.weight", c["indexer_head_dim"]),
        ):
            self.add(p + role, plain(self.store, blk + tensor, (width,), minus_one=True))
        self.matrix(p + "output", blk + "attn_output.weight", (h, q), inputs=(p + "gated_output",))
        # One parent when every input projection shares a format; else the GGUF projections (one parent
        # where their formats agree) and the BF16 indexer pair, which the runtime runs as two projections.
        roles = ("query", "gate", "key", "value", "index_query", "index_key")
        self.model.packing_groups.extend((tuple(p + r for r in roles), tuple(p + r for r in roles[:4]),
                                          (p + "query", p + "gate"), tuple(p + r for r in roles[4:])))

    def moe(self, prefix, blk, layer):
        c = self.config
        h, e, ir, shared = c["hidden_size"], c["num_experts"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
        p, inputs = prefix + "moe/", (prefix + "ffn_input",)
        self.matrix(p + "router", blk + "ffn_gate_inp.weight", (e, h), inputs=inputs)
        self.add(p + "shared_score", plain(self.store, blk + "ffn_gate_inp_shexp.weight", (1, h)), inputs=inputs)
        self.model.packing_groups.append((p + "router", p + "shared_score"))
        self.matrix(p + "shared/gate", blk + "ffn_gate_shexp.weight", (shared, h), inputs=inputs)
        self.matrix(p + "shared/up", blk + "ffn_up_shexp.weight", (shared, h), inputs=inputs)
        self.matrix(p + "shared/down", blk + "ffn_down_shexp.weight", (h, shared), inputs=(p + "shared/product",))
        self.model.packing_groups.append((p + "shared/gate", p + "shared/up"))
        bank = GgmlExpertBank(self.store, layer, (e, h, ir))
        self.model.add(Parameter(p + "experts", (e, h, ir), bank, None, inputs, bank.gate_up_format, residency="text"))

    def ple(self, prefix, blk):
        c = self.config
        h, width, dim = c["hidden_size"], c["hc_count"] * c["hidden_size"], c["ple_embed_dim"]
        p, embeddings = prefix + "ple/", prefix + "ple/embeddings"
        self.matrix(p + "key_projection", blk + "ple_key.weight", (width, dim), inputs=(embeddings,))
        self.matrix(p + "value_projection", blk + "ple_value.weight", (h, dim), inputs=(embeddings,))
        self.model.packing_groups.append((p + "key_projection", p + "value_projection"))
        for role, tensor in (("key_norm", "ple_norm_key"), ("query_norm", "ple_norm_query"), ("conv_norm", "ple_norm_conv")):
            self.add(p + role, plain(self.store, blk + tensor + ".weight", (width,), minus_one=True))
        taps = c["ple_conv_kernel_size"]
        conv = plain(self.store, blk + "ple_conv1d.weight", (width, taps))
        self.add(p + "convolution", transpose_source(conv, (1, 0), (taps, width)))


def ggml_or_plain(store: GgufStore, name: str, shape: tuple[int, int]) -> LogicalSource:
    """A 2-D GGUF tensor: stored rows for ggml formats and Q8_0, exact values for BF16/F32/F16 (a
    BF16 binding: every value must be a BF16 value, so an F32 matrix is never rounded silently)."""

    t = store.tensor(name)
    if t.ggml_type in (GGUF_F32, GGUF_F16, GGUF_BF16):
        return plain(store, name, shape)
    return ggml_matrix(store, name, shape)


class Bf16ExactSource:
    """A source whose every value must be a BF16 value (a BF16 binding then stores it exactly)."""

    def __init__(self, source):
        self._source = source
        self.path = source.path
        self.config: dict = {}

    def has(self, name: str) -> bool:
        return self._source.has(name)

    def describe(self, name: str):
        return self._source.describe(name)

    def read_flat(self, name: str, begin: int = 0, end: int | None = None) -> torch.Tensor:
        values = self._source.read_flat(name, begin, end)
        if not torch.equal(values.to(torch.bfloat16).to(values.dtype), values):
            raise ValueError(f"{self.path}: {name} holds a value that is not exact in BF16")
        return values


def _mmproj_checks(mmproj, vision: dict, text: dict) -> None:
    """The mmproj's metadata must describe the checkpoint's vision tower (``vision_config``)."""

    meta = mmproj.metadata()
    expected = {
        "general.architecture": "clip",
        "clip.projector_type": "qwen3vl_merger",
        "clip.vision.block_count": vision["depth"],
        "clip.vision.embedding_length": vision["hidden_size"],
        "clip.vision.feed_forward_length": vision["intermediate_size"],
        "clip.vision.attention.head_count": vision["num_heads"],
        "clip.vision.patch_size": vision["patch_size"],
        "clip.vision.spatial_merge_size": vision["spatial_merge_size"],
        "clip.vision.projection_dim": text["hidden_size"],
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{mmproj.path}: {key} is {meta.get(key)!r}, the checkpoint's vision tower has {value!r}")
    if any(meta.get("clip.vision.is_deepstack_layers") or []):
        raise ValueError(f"{mmproj.path}: deepstack layers are not part of this vision tower")
    side = meta.get("clip.vision.image_size", 0) // vision["patch_size"]
    if side * side != vision["num_position_embeddings"]:
        raise ValueError(f"{mmproj.path}: image size gives {side * side} positions, not {vision['num_position_embeddings']}")


def vision_source(mmproj, vision: dict) -> Bf16ExactSource:
    """The mmproj's tensors under the checkpoint's ``model.visual.*`` names, exactly."""

    from .sources.gguf import HFAliasSource, qwen35_vision_name_map

    name_map, transforms, shapes, composites = qwen35_vision_name_map(
        vision["depth"], hidden_size=vision["hidden_size"], temporal_patch_size=vision["temporal_patch_size"],
        patch_size=vision["patch_size"])
    return Bf16ExactSource(HFAliasSource(mmproj, name_map, transforms=transforms, shapes=shapes, composites=composites))


def build_model(
    base: SafetensorsSource,
    store: GgufStore,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str | Path] | None = None,
    mmproj=None,
) -> Model:
    """The Qwen4Exp logical model of a GGUF (``store``), configured by Qwen's checkpoint (``base``);
    the vision component's tower comes from ``mmproj`` (a ``sources.gguf.GGUFSource``)."""

    selected = set(components)
    if "text" not in selected or selected - {"text", "vision", "mtp"}:
        raise ValueError("select text and optional vision and mtp components")
    raw = dict(base.config)
    text = dict(raw.get("text_config", raw))
    text.pop("ple_embedding_dtype", None)  # the GGUF's table is IQ4_NL, checked below
    if "text_config" in raw:
        raw["text_config"] = text
    else:
        raw = text
    config = qwen4_exp.text_config(raw, mtp="mtp" in selected)
    table = store.tensor(NGRAM_TABLE)
    if table.ggml_type != 20 or table.row_elems != config["ple_embed_dim"] // (
            (config["ngram_size"] - 1) * config["heads_per_ngram"]):
        raise ValueError(f"{store.path}: {NGRAM_TABLE} must be IQ4_NL rows of one n-gram head")
    _metadata_checks(store, config, table.rows)
    config["gdn_output_order"] = "ggml_tiled"
    config["gdn_a"] = "direct"
    config["ngram_table"] = {"format": IQ4_NL_ROW_FORMAT, **qwen4_exp.ngram_geometry(table.rows, table.row_bytes)}
    records = {"text": {"config": config}}
    if "vision" in selected:
        if mmproj is None:
            raise ValueError("the vision component of a GGUF import needs the repository's mmproj (--mmproj)")
        vision = vision_config(base.config, config)
        vision["model_type"] = "qwen4_exp_vision"
        _mmproj_checks(mmproj, vision, config)
        records["vision"] = {"config": vision, "target": "text"}
    elif mmproj is not None:
        raise ValueError("an mmproj needs the vision component")
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
    b = _Builder(model, store, config)
    h, v = config["hidden_size"], config["vocab_size"]
    b.matrix("text/token_embedding", "token_embd.weight", (v, h))
    heads = ("text/final_hidden",) + (("mtp/final_hidden",) if "mtp" in selected else ())
    b.matrix("text/output_head", "output.weight", (v, h), inputs=heads)
    b.hyper_connection("text/final_mixer/", "output_hc_", combine=False)
    (ple_layer,) = config["ple_layer_ids"]
    for i, kind in enumerate(config["layer_types"]):
        prefix, blk = f"text/layers/{i}/", f"blk.{i}."
        if i + 1 == ple_layer:
            b.ple(prefix, blk)
        b.hyper_connection(prefix + "attn_hc/", blk + "hc_attn_")
        if kind == "full_attention":
            b.qsa(prefix, blk)
        else:
            b.gdn(prefix, blk)
        b.hyper_connection(prefix + "mlp_hc/", blk + "hc_ffn_")
        b.moe(prefix, blk, i)
    if "mtp" in selected:
        # Qwen's MTP head, exactly as recipe A builds it (qwen4_exp.build_model).
        mtp = qwen4_exp._Qwen4ExpBuilder(model)
        width = config["hc_count"] * h
        mtp.add("mtp/embedding_norm", base, "mtp.pre_fc_norm_embedding.weight", (h,))
        mtp.add("mtp/hidden_norm", base, "mtp.pre_fc_norm_hidden.weight", (width,))
        mtp.add("mtp/embedding_projection", base, "mtp.fc_embedding.weight", (h, h), inputs=("mtp/embedding_input",))
        mtp.add("mtp/hidden_projection", base, "mtp.fc_hidden.weight", (h, h), inputs=("mtp/hidden_input",))
        mtp.hyper_connection("mtp/final_mixer/", "mtp.hyper_connection_mixer.", base, config, combine=False)
        mtp.block("mtp/layers/0/", "mtp.layers.0.", base, config, "full_attention", mtp=True)
    if "vision" in selected:
        vision = records["vision"]["config"]
        qwen4_exp._Qwen4ExpBuilder(model).vision(vision_source(mmproj, vision), vision, h)
    return model


# ---------------------------------------------------------------------------- n-gram volume


class Iq4NlTable:
    """The PLE n-gram table as the volume stores it: the GGUF's IQ4_NL rows (90 B), verbatim."""

    volume_format = "iq4_nl"

    def __init__(self, store: GgufStore, rows: int):
        self.store = store
        t = store.tensor(NGRAM_TABLE)
        self.row_bytes = t.row_bytes
        self.rows = t.rows
        if self.rows != rows:
            raise ValueError(f"{store.path}: n-gram table rows {self.rows} differ from {rows}")
        self.shards = [(NGRAM_TABLE, self.rows)]

    def codes(self, name: str, begin: int, end: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.store.raw(name, begin, end)))
