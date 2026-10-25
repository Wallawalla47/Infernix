"""Qwen3.5 dense text (``Qwen3_5ForCausalLM``) from a llama.cpp ``qwen35`` GGUF, imported bit-exactly.

The logical model is the one ``qwen3_5.build_model`` describes (the same parameter names, shapes,
inputs and packing groups), with every text source read from the GGUF instead of the Hugging Face
checkpoint: Q8_0 matrices expose their stored blocks (``read_encoded``), the ``q8_g32_fp16`` words,
and plain tensors (F32, F16, BF16) their exact values. llama.cpp's export transforms
(``convert_hf_to_gguf.py`` ``Qwen3_5TextModel``) are undone exactly and checked:

* ``+1`` on the zero-centred RMSNorm gammas (every norm but the gated delta-net's): g - 1 must be a
  BF16 value. Where llama.cpp's binary32 1 + g dropped low bits of a tiny gamma (|g| < 2^-16: two
  elements of Qwen3.8-27B's), the import holds the GGUF's gamma, so both engines run the same
  weights;
* the gated delta-net's value heads in ggml's tiled order: rows are permuted back (whole rows), and
  the output projection's columns as whole Q8_0 blocks (a value head is whole blocks: 128 columns
  are four), so its stored blocks are exactly the blocks of the matrix in Hugging Face order and the
  runtime needs no ggml order;
* ``A_log`` stored as ``-exp(A_log)`` in F32: recovered as the one BF16 value (the checkpoint's
  dtype) whose -exp is within two F32 units in the last place of the stored value, for the
  converter's expf; any other count is refused;
* the causal convolution squeezed to [channels, taps]: transposed to the runtime's [taps, channels].

The GGUF's MTP block (``nextn``) and any vision tower are not imported. The config, tokenizer and
frontend resources come from ``--model`` (the checkpoint's config and tokenizer files suffice; no
weights are read), and the GGUF metadata must agree with that config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from . import qwen3_5
from .model import Model, Parameter
from .resources import load_resources
from .sources.gguf_logical import (ggml_or_plain, head_rows, permute_vector, plain, value_head_columns,
                                   value_head_slots)
from .sources.gguf_raw import GgufStore
from .sources.logical import LogicalSource, select_rows, transpose_source
from .sources.safetensors import SafetensorsSource


def _finite_bf16() -> np.ndarray:
    words = np.arange(1 << 16, dtype=np.uint32) << 16
    words = words[(words >> 23) & 0xFF != 0xFF]  # infinities and NaNs have the all-ones exponent
    return words.view(np.float32).astype(np.float64)


def a_log_from_a(source: LogicalSource) -> LogicalSource:
    """``A_log`` from llama.cpp's stored ``A = -exp(A_log)`` (F32): per head the one BF16 value
    whose -exp lies within two F32 units in the last place of A (rounding plus the converter's expf
    error). A BF16 grid this coarse leaves at most one candidate there unless ``|A_log|`` is tiny;
    an ambiguous or missing candidate is refused rather than guessed."""

    cache: list[torch.Tensor] = []

    def read(begin: int, end: int) -> torch.Tensor:
        if not cache:
            a = source.values().numpy().astype(np.float64)
            candidates = _finite_bf16()
            with np.errstate(over="ignore"):
                exact = -np.exp(candidates)
            tolerance = 2 * np.spacing(np.abs(a).astype(np.float32)).astype(np.float64)
            hits = np.abs(exact[None, :] - a[:, None]) <= tolerance[:, None]
            counts = hits.sum(axis=1)
            if (counts != 1).any():
                head = int(np.flatnonzero(counts != 1)[0])
                raise ValueError(f"{source.label}: A = {a[head]!r} (head {head}) has {int(counts[head])} BF16 "
                                 "A_log candidates; it cannot be recovered exactly")
            cache.append(torch.from_numpy(candidates[hits.argmax(axis=1)]))
        return cache[0][begin:end]

    return LogicalSource(source.shape, f"a_log({source.label})", read)


def _metadata_checks(store: GgufStore, config: dict) -> None:
    m = store.metadata
    if m.get("general.architecture") != "qwen35":
        raise ValueError(f"{store.path}: GGUF architecture {m.get('general.architecture')!r} is not qwen35")
    rope = config["rope_parameters"]
    nv, dv = config["linear_num_value_heads"], config["linear_value_head_dim"]
    interval = m.get("qwen35.full_attention_interval")
    expected = {
        "qwen35.block_count": config["num_hidden_layers"] + m.get("qwen35.nextn_predict_layers", 0),
        "qwen35.embedding_length": config["hidden_size"],
        "qwen35.feed_forward_length": config["intermediate_size"],
        "qwen35.attention.head_count": config["num_attention_heads"],
        "qwen35.attention.head_count_kv": config["num_key_value_heads"],
        "qwen35.attention.key_length": config["head_dim"],
        "qwen35.attention.value_length": config["head_dim"],
        "qwen35.attention.layer_norm_rms_epsilon": config["rms_norm_eps"],
        "qwen35.rope.freq_base": rope["rope_theta"],
        "qwen35.rope.dimension_count": int(config["head_dim"] * rope["partial_rotary_factor"]),
        "qwen35.rope.dimension_sections": rope["mrope_section"] + [0],
        "qwen35.ssm.conv_kernel": config["linear_conv_kernel_dim"],
        "qwen35.ssm.state_size": config["linear_key_head_dim"],
        "qwen35.ssm.group_count": config["linear_num_key_heads"],
        "qwen35.ssm.time_step_rank": nv,
        "qwen35.ssm.inner_size": nv * dv,
    }
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f"{store.path}: {key} = {m.get(key)!r}, the checkpoint config says {value!r}")
    layers = ["full_attention" if type(interval) is int and (i + 1) % interval == 0 else "linear_attention"
              for i in range(config["num_hidden_layers"])]
    if layers != config["layer_types"]:
        raise ValueError(f"{store.path}: full_attention_interval {interval!r} differs from the config's layer types")


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

    def norm(self, name, tensor, width):
        self.add(name, plain(self.store, tensor, (width,), minus_one=True))

    def group(self, *names):
        self.model.packing_groups.append(tuple(names))

    def attention(self, prefix, blk):
        c = self.config
        h, d, heads = c["hidden_size"], c["head_dim"], c["num_attention_heads"]
        q, k = heads * d, c["num_key_value_heads"] * d
        p, inputs = prefix + "attention/", (prefix + "mixer_input",)
        fused = ggml_or_plain(self.store, blk + "attn_q.weight", (2 * q, h))
        for role, gate in (("query", False), ("gate", True)):
            ranges = tuple((head * 2 * d + int(gate) * d, head * 2 * d + (int(gate) + 1) * d) for head in range(heads))
            self.add(p + role, select_rows(fused, ranges), inputs=inputs)
        self.matrix(p + "key", blk + "attn_k.weight", (k, h), inputs=inputs)
        self.matrix(p + "value", blk + "attn_v.weight", (k, h), inputs=inputs)
        self.norm(p + "query_norm", blk + "attn_q_norm.weight", d)
        self.norm(p + "key_norm", blk + "attn_k_norm.weight", d)
        self.matrix(p + "output", blk + "attn_output.weight", (h, q), inputs=(p + "gated_output",))
        roles = [p + role for role in ("query", "key", "gate", "value")]
        self.group(*roles)
        self.group(*roles[:2])
        self.group(*roles[2:])

    def gdn(self, prefix, blk):
        c = self.config
        h = c["hidden_size"]
        nk, dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
        nv, dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
        kg, vg, taps = nk * dk, nv * dv, c["linear_conv_kernel_dim"]
        channels = 2 * kg + vg
        p, inputs = prefix + "gdn/", (prefix + "mixer_input",)
        a = permute_vector(plain(self.store, blk + "ssm_a", (nv,), bf16_exact=False), nv, nk)
        self.add(p + "a_log", a_log_from_a(a), direct="fp32")
        self.add(p + "dt_bias", permute_vector(plain(self.store, blk + "ssm_dt.bias", (nv,)), nv, nk), direct="fp32")
        conv = plain(self.store, blk + "ssm_conv1d.weight", (channels, taps))
        conv = select_rows(conv, ((0, 2 * kg),) + tuple(
            (2 * kg + g * dv, 2 * kg + (g + 1) * dv) for g in value_head_slots(nv, nk)))
        self.add(p + "convolution", transpose_source(conv, (1, 0), (taps, channels)))
        for role, tensor in (("a_projection", "ssm_alpha.weight"), ("b_projection", "ssm_beta.weight")):
            self.add(p + role, head_rows(plain(self.store, blk + tensor, (nv, h)), nv, 1, nk), inputs=inputs)
        qkv = ggml_or_plain(self.store, blk + "attn_qkv.weight", (channels, h))
        self.add(p + "query", select_rows(qkv, ((0, kg),)), inputs=inputs)
        self.add(p + "key", select_rows(qkv, ((kg, 2 * kg),)), inputs=inputs)
        self.add(p + "value", head_rows(qkv, nv, dv, nk, row_offset=2 * kg), inputs=inputs)
        self.add(p + "z", head_rows(ggml_or_plain(self.store, blk + "attn_gate.weight", (vg, h)), nv, dv, nk),
                 inputs=inputs)
        self.add(p + "norm", plain(self.store, blk + "ssm_norm.weight", (dv,)))
        output = ggml_or_plain(self.store, blk + "ssm_out.weight", (h, vg))
        self.add(p + "output", value_head_columns(output, nv, dv, nk), inputs=(p + "gated_output",))
        roles = [p + role for role in ("query", "key", "value", "z")]
        self.group(*roles)
        self.group(*roles[:2])
        self.group(*roles[2:])
        self.group(p + "a_projection", p + "b_projection")

    def dense(self, prefix, blk):
        c = self.config
        h, intermediate = c["hidden_size"], c["intermediate_size"]
        p = prefix + "mlp/"
        self.matrix(p + "gate", blk + "ffn_gate.weight", (intermediate, h), inputs=(prefix + "ffn_input",))
        self.matrix(p + "up", blk + "ffn_up.weight", (intermediate, h), inputs=(prefix + "ffn_input",))
        self.matrix(p + "down", blk + "ffn_down.weight", (h, intermediate), inputs=(p + "product",))
        self.group(p + "gate", p + "up")

    def block(self, prefix, blk, mixer):
        h = self.config["hidden_size"]
        self.norm(prefix + "input_norm", blk + "attn_norm.weight", h)
        if mixer == "full_attention":
            self.attention(prefix, blk)
        else:
            self.gdn(prefix, blk)
        self.norm(prefix + "post_attention_norm", blk + "post_attention_norm.weight", h)
        self.dense(prefix, blk)


def build_model(
    base: SafetensorsSource,
    store: GgufStore,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    """The Qwen3.5 dense text model of a ``qwen35`` GGUF (``store``), configured by Qwen's
    checkpoint (``base``: its config and tokenizer files)."""

    if tuple(components) != ("text",):
        raise ValueError("a qwen35 GGUF import builds the text component only")
    config = qwen3_5.text_config(base.config, mtp=False)
    if "num_experts" in config:
        raise ValueError("a qwen35 GGUF import requires Qwen3.5 dense mathematics")
    _metadata_checks(store, config)
    records = {"text": {"config": config}}
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], overrides=resource_overrides)
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    b = _Builder(model, store, config)
    h, v = config["hidden_size"], config["vocab_size"]
    b.matrix("text/token_embedding", "token_embd.weight", (v, h))
    head = "token_embd.weight" if config["tie_word_embeddings"] else "output.weight"
    b.matrix("text/output_head", head, (v, h), inputs=("text/final_hidden",))
    b.norm("text/final_norm", "output_norm.weight", h)
    for i, kind in enumerate(config["layer_types"]):
        b.block(f"text/layers/{i}/", f"blk.{i}.", kind)
    return model
