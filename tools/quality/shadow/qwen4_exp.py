"""Qwen3.8-Flash-Next (qwen4_exp_text) in exact arithmetic on transformers main 3bb8b727's code.

The authority (docs/maintainer/lossless-route.md §2a) is modeling_qwen4_exp.py at transformers main 3bb8b727
(installed as 5.21.0.dev0). Its modules run unchanged on Shadow tensors, layer by layer, with three
substitutions that change no arithmetic: the routed experts read each hit expert's weights from the artifact
when its turn comes (the same operations as the eager experts loop; holding all 512 experts in binary64
would take tens of gigabytes per layer), the n-gram table's rows come from the artifact's .ngram volume (the
rows the hashed ids name), and the masks are built directly (causal for QSA attention, none for the
recurrent mixers). The fused kernels are evaluated as single operations as for Qwen3.5 (overrides.py):
attention over QSA's selected keys, the GatedDeltaNet causal convolution and the gated delta rule (this code
calls them as module functions, so the module's names are rebound). PLE's dilated convolution is F.conv1d
followed by F.silu, two operations, as the code writes it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from . import exact
from .overrides import exact_attention, exact_causal_conv, fla_gated_delta_rule, set_parameter
from .tensor import F64, Shadow, shadow, value
from .weights import Weights, qwen4_exp_layer

AUTHORITY = "5.21.0.dev0"  # transformers main 3bb8b727
ORIGINALS: dict = {}  # the module functions the shadow rebinds


def _transformers():
    import transformers
    if transformers.__version__ != AUTHORITY:
        raise SystemExit(f"Qwen3.8-Flash-Next's authority is transformers main 3bb8b727 ({AUTHORITY}); this "
                         f"interpreter has {transformers.__version__}")
    from transformers import AttentionInterface
    from transformers.models.qwen4_exp import configuration_qwen4_exp, modeling_qwen4_exp
    AttentionInterface.register("iqs_exact", exact_attention)
    ORIGINALS.setdefault("causal_conv1d_fn", modeling_qwen4_exp.causal_conv1d_fn)
    ORIGINALS.setdefault("torch_chunk_gated_delta_rule", modeling_qwen4_exp.torch_chunk_gated_delta_rule)
    modeling_qwen4_exp.causal_conv1d_fn = exact_causal_conv
    modeling_qwen4_exp.torch_chunk_gated_delta_rule = fla_gated_delta_rule
    return configuration_qwen4_exp, modeling_qwen4_exp


def source_digest() -> str:
    from transformers.models.qwen4_exp import modeling_qwen4_exp
    return hashlib.sha256(Path(modeling_qwen4_exp.__file__).read_bytes()).hexdigest()


class VolumeEmbedding(torch.nn.Module):
    """The n-gram table's rows, read from the .ngram volume (BF16 rows: their exact values)."""

    def __init__(self, path: str, table: dict, dim: int):
        super().__init__()
        if table["format"] != "bf16":
            raise NotImplementedError(f"shadow n-gram rows: a {table['format']} volume")
        self.path, self.table, self.dim = path, table, dim
        self.weight = torch.empty(0)  # the module reads weight.device

    def forward(self, ids: torch.Tensor) -> Shadow:
        t = self.table
        flat = ids.reshape(-1)
        rows, inverse = torch.unique(flat, return_inverse=True)
        values = np.empty((len(rows), self.dim), dtype=np.uint16)
        with open(self.path, "rb") as stream:
            for n, r in enumerate(rows.tolist()):
                stream.seek(t["header_bytes"] + (r // t["rows_per_block"]) * t["block_bytes"]
                            + (r % t["rows_per_block"]) * t["row_bytes"])
                values[n] = np.frombuffer(stream.read(self.dim * 2), dtype="<u2")
        words = torch.from_numpy(values.astype(np.int32) << 16).view(torch.float32).to(F64)
        return Shadow(words[inverse].reshape(*ids.shape, self.dim), torch.bfloat16)


def _experts_forward(w: Weights, bank: str):
    """Qwen4ExpTextExperts.forward (eager) with each hit expert's weights read from the artifact."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            gate_w, up_w, down_w = w.expert_bank(bank, int(expert_idx))
            gate_up_proj = Shadow(torch.cat([gate_w, up_w]), torch.bfloat16)
            down_proj = Shadow(down_w, torch.bfloat16)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = torch.nn.functional.linear(current_state, gate_up_proj).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = torch.nn.functional.linear(current_hidden_states, down_proj)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    return forward


def run(weights: Weights | str, sequences: list[list[int]], positions: list[list[int]], log=print,
        ngram_volume: str | None = None) -> list[torch.Tensor]:
    """FP32 logits rows ([len(positions[i]), vocab] binary64 tensors) of each sequence at its positions."""
    config_module, model = _transformers()
    w = Weights(weights) if isinstance(weights, str) else weights
    c = dict(w.config)
    table = c.pop("ngram_table", None)
    if c.get("ple_layer_ids") and not ngram_volume:
        raise SystemExit("Qwen3.8-Flash-Next's PLE needs the artifact's n-gram volume (--ngram-volume)")
    c["rope_parameters"] = {"rope_type": "default", **c["rope_parameters"]}
    config = config_module.Qwen4ExpTextConfig(**{k: v for k, v in c.items() if k not in ("architectures", "model_type")})
    config._attn_implementation = "iqs_exact"
    config._experts_implementation = "eager"
    h, vocab, hc = c["hidden_size"], c["vocab_size"], c["hc_count"]
    tokens = torch.tensor(sequences)
    batch, length = tokens.shape

    distinct = sorted(set(tokens.reshape(-1).tolist()))
    rows = w.rows("text/token_embedding", distinct)
    index = {t: i for i, t in enumerate(distinct)}
    hidden = shadow(rows[torch.tensor([[index[t] for t in row] for row in sequences])], torch.bfloat16)
    del rows
    rotary = model.Qwen4ExpTextRotaryEmbedding(config)
    dim = int(c["head_dim"] * c["rope_parameters"].get("partial_rotary_factor", 1.0))
    steps = shadow(torch.arange(0, dim, 2, dtype=torch.int64).to(F64), torch.float32)
    rotary._buffers["inv_freq"] = 1.0 / (c["rope_parameters"]["rope_theta"] ** (steps / dim))
    position_ids = torch.arange(length)[None].expand(batch, length)
    cos, sin = rotary(hidden, position_ids)
    hidden = hidden.repeat(1, 1, hc)
    visible = ~torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    mask = visible[None, None].expand(batch, 1, length, length)

    for i in range(c["num_hidden_layers"]):
        started = time.time()
        with torch.device("meta"):
            layer = model.Qwen4ExpTextDecoderLayer(config, i)
        for name, (values, dtype) in qwen4_exp_layer(w, i).items():
            set_parameter(layer, name, Shadow(values, dtype))
        experts = layer.mlp.experts
        experts.forward = types.MethodType(_experts_forward(w, f"text/layers/{i}/moe/experts"), experts)
        if layer.ple is not None:
            emb = layer.ple.ple_embedding
            emb._buffers["layer_multipliers"] = model._build_layer_multipliers(
                emb.unigram_vocab_size, emb.ngram_size, emb.ple_layer_index, emb.seed)
            emb._buffers["ngram_heads_vocab_sizes"] = torch.tensor(emb.head_vocab_sizes, dtype=torch.long)
            emb._buffers["ngram_heads_offsets"] = torch.tensor(emb.head_offsets, dtype=torch.long)
            emb.ngram_embedding = VolumeEmbedding(ngram_volume, table, c["ple_embed_dim"] // emb.ngram_heads)
        left = [n for n, p in layer.named_parameters()
                if not isinstance(p, Shadow) and not n.startswith("mlp.experts.")]
        if left:
            raise RuntimeError(f"layer {i}: no artifact value for {left}")
        with torch.no_grad():
            hidden = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask, conv_mask=None,
                           past_key_values=None, ple_input_ids=tokens)
        w.release()
        log(f"layer {i + 1}/{c['num_hidden_layers']} ({c['layer_types'][i]}) {time.time() - started:.0f} s")
        del layer

    with torch.device("meta"):
        mixer = model.Qwen4ExpTextGatedResidual(config, use_combine=False)
    width, rank = hc * h, c["hc_lowrank"]
    for hf, logical, shape in (("hc_norm.weight", "norm", (width,)), ("input_mix_weight_down.weight", "down", (rank, width)),
                               ("input_mix_weight_up.weight", "up", (width, rank))):
        set_parameter(mixer, hf, Shadow(w.logical("text/final_mixer/" + logical, shape),
                                        w.nominal("text/final_mixer/" + logical)))
    with torch.no_grad():
        final = value(mixer(hidden))
    outputs = [torch.empty(len(rows), vocab, dtype=F64) for rows in positions]
    block = 16384
    for first in range(0, vocab, block):
        head = w.rows("text/output_head", slice(first, min(vocab, first + block)))
        for b, rows in enumerate(positions):
            outputs[b][:, first:first + head.shape[0]] = exact.matmul(final[b, rows], head.t(), torch.float32)
    return outputs
