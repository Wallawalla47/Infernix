"""Qwen3.5 / Qwen3.6 (qwen3_5_text, qwen3_5_moe_text) in exact arithmetic on transformers 5.5.0's code.

The authority (docs/maintainer/lossless-route.md §1-2) is transformers 5.5's modeling_qwen3_5.py and
modeling_qwen3_5_moe.py. Their modules run unchanged on Shadow tensors, layer by layer with the artifact's
weights; four operations are the fused kernels the model was trained with, each evaluated as one operation:

- attention (flash attention): s_j = (q . k_j) d^-1/2 exact, p_j = fp32(exp(fp32(s_j - max s))), and
  o = bf16(sum p_j v_j / sum p_j) with both sums exact;
- the causal convolution (causal_conv1d): bf16(silu(sum over taps of w x)), the sum exact;
- the gated delta rule (FLA's chunk kernel): q and k L2-normalized in FP32, q scaled by fp32(d_k^-1/2) (a
  multiplication, as FLA does; transformers' torch fallback divides), then the recurrence operation by
  operation as torch_recurrent_gated_delta_rule writes it;
- the head: logits rounded once to FP32 (the route's declared deviation from transformers' BF16 logits).

transformers 5.5 differs from later releases where it matters: its MoE router keeps the routing weights in
FP32 (later releases cast them to BF16).
"""

from __future__ import annotations

import time

import torch

from . import exact
from .overrides import exact_attention, exact_causal_conv, fla_gated_delta_rule, set_parameter
from .tensor import F64, Shadow, shadow, value
from .weights import Weights, qwen3_5_layer

AUTHORITY = "5.5.0"


def _transformers():
    import transformers
    if transformers.__version__ != AUTHORITY:
        raise SystemExit(f"Qwen3.5's authority is transformers {AUTHORITY}; this interpreter has "
                         f"{transformers.__version__} (install it in its own environment)")
    from transformers import AttentionInterface
    from transformers.models.qwen3_5 import configuration_qwen3_5, modeling_qwen3_5
    from transformers.models.qwen3_5_moe import configuration_qwen3_5_moe, modeling_qwen3_5_moe
    AttentionInterface.register("iqs_exact", exact_attention)
    return (configuration_qwen3_5, modeling_qwen3_5), (configuration_qwen3_5_moe, modeling_qwen3_5_moe)


# ------------------------------------------------------------------------------------------------- forward

def _record_gdn(gdn, record: dict, index: int, batch: int, length: int) -> None:
    """Hooks that record one delta-net layer's intermediates (see run)."""
    for name, module in (("mixed", gdn.in_proj_qkv), ("z", gdn.in_proj_z), ("a", gdn.in_proj_a),
                         ("b", gdn.in_proj_b), ("y", gdn.out_proj)):
        module.register_forward_hook(
            lambda m, args, output, name=name: record.__setitem__((index, "gdn", name), value(output)))
    gdn.norm.register_forward_hook(
        lambda m, args, output: record.__setitem__((index, "gdn", "on"), value(output).reshape(batch, length, -1)))
    conv_fn = gdn.causal_conv1d_fn

    def conv_wrapped(*fn_args, **fn_kwargs):
        out = conv_fn(*fn_args, **fn_kwargs)
        record[(index, "gdn", "conv")] = value(out).transpose(1, 2)  # [batch, channels, tokens] to [batch, tokens, channels]
        return out

    gdn.causal_conv1d_fn = conv_wrapped
    rule_fn = gdn.chunk_gated_delta_rule

    def rule_wrapped(*fn_args, **fn_kwargs):
        out = rule_fn(*fn_args, **fn_kwargs)
        record[(index, "gdn", "g")] = value(fn_kwargs["g"])
        record[(index, "gdn", "beta")] = value(fn_kwargs["beta"])
        record[(index, "gdn", "o")] = value(out[0]).reshape(batch, length, -1)
        return out

    gdn.chunk_gated_delta_rule = rule_wrapped


def run(weights: Weights | str, sequences: list[list[int]], positions: list[list[int]], log=print,
        record: dict | None = None, stop_after_layers: bool = False,
        detail: frozenset = frozenset()) -> list[torch.Tensor]:
    """FP32 logits rows ([len(positions[i]), vocab] binary64 tensors) of each sequence at its positions.

    record (a dict) receives the residual stream, binary64 values of the BF16 tensors ([batch, tokens, hidden]):
    "embed", then (layer, "mixer") after each layer's attention or delta-net block and (layer, "layer") after the
    layer, for tools.quality.shadow.bisect; stop_after_layers returns once the layers have run. For the
    delta-net layers in `detail` it also records (layer, "gdn", name), name in mixed, z, a, b, g, beta, conv, o, on,
    y: the projections, the gates, the convolution, the delta rule, the gated norm and the output projection
    ([batch, tokens, channels])."""
    (dense_config, dense), (moe_config, moe) = _transformers()
    w = Weights(weights) if isinstance(weights, str) else weights
    c = dict(w.config)
    is_moe = "num_experts" in c
    config_module, model = (moe_config, moe) if is_moe else (dense_config, dense)
    config_class = config_module.Qwen3_5MoeTextConfig if is_moe else config_module.Qwen3_5TextConfig
    c["rope_parameters"] = {"rope_type": "default", **c["rope_parameters"]}
    config = config_class(**{k: v for k, v in c.items() if k not in ("architectures", "model_type")})
    config._attn_implementation = "iqs_exact"
    if is_moe:
        config._experts_implementation = "eager"
    h, vocab = c["hidden_size"], c["vocab_size"]
    tokens = torch.tensor(sequences)
    batch, length = tokens.shape

    names = {"rotary": "Qwen3_5MoeTextRotaryEmbedding" if is_moe else "Qwen3_5TextRotaryEmbedding",
             "norm": "Qwen3_5MoeRMSNorm" if is_moe else "Qwen3_5RMSNorm",
             "layer": "Qwen3_5MoeDecoderLayer" if is_moe else "Qwen3_5DecoderLayer"}
    distinct = sorted(set(tokens.reshape(-1).tolist()))
    table = w.rows("text/token_embedding", distinct)
    index = {t: i for i, t in enumerate(distinct)}
    hidden = shadow(table[torch.tensor([[index[t] for t in row] for row in sequences])], torch.bfloat16)
    del table
    rotary = getattr(model, names["rotary"])(config)
    dim = int(c["head_dim"] * c["rope_parameters"].get("partial_rotary_factor", 1.0))
    steps = shadow(torch.arange(0, dim, 2, dtype=torch.int64).to(F64), torch.float32)
    rotary._buffers["inv_freq"] = 1.0 / (c["rope_parameters"]["rope_theta"] ** (steps / dim))
    position_ids = torch.arange(length)[None].expand(batch, length)
    cos, sin = rotary(hidden, position_ids)
    if record is not None:
        record["embed"] = value(hidden)

    layer_class = getattr(model, names["layer"])
    for i in range(c["num_hidden_layers"]):
        started = time.time()
        with torch.device("meta"):
            layer = layer_class(config, i)
        for name, (values, dtype) in qwen3_5_layer(w, i).items():
            set_parameter(layer, name, Shadow(values, dtype))
        w.release()
        left = [n for n, p in layer.named_parameters() if not isinstance(p, Shadow)]
        if left:
            raise RuntimeError(f"layer {i}: no artifact value for {left}")
        if c["layer_types"][i] == "linear_attention":
            layer.linear_attn.causal_conv1d_fn = exact_causal_conv
            layer.linear_attn.chunk_gated_delta_rule = fla_gated_delta_rule
        if record is not None:
            seen = {}
            layer.register_forward_pre_hook(
                lambda module, args, kwargs, seen=seen: seen.__setitem__("input", kwargs.get("hidden_states", args[0] if args else None)),
                with_kwargs=True)
            mixer = layer.linear_attn if c["layer_types"][i] == "linear_attention" else layer.self_attn

            def after_mixer(module, args, kwargs, output, seen=seen, index=i):
                out = output[0] if isinstance(output, tuple) else output
                record[(index, "mixer")] = value(seen["input"] + out)  # the decoder layer's residual add

            mixer.register_forward_hook(after_mixer, with_kwargs=True)
            if i in detail and c["layer_types"][i] == "linear_attention":
                _record_gdn(layer.linear_attn, record, i, batch, length)
        with torch.no_grad():
            hidden = layer(hidden, position_embeddings=(cos, sin), attention_mask=None, position_ids=position_ids)
        if record is not None:
            record[(i, "layer")] = value(hidden)
        log(f"layer {i + 1}/{c['num_hidden_layers']} ({c['layer_types'][i]}) {time.time() - started:.0f} s")
        del layer
    if stop_after_layers:
        return []

    norm = getattr(model, names["norm"])(h, eps=c["rms_norm_eps"])
    norm._parameters["weight"] = Shadow(w.logical("text/final_norm", (h,)), w.nominal("text/final_norm"))
    with torch.no_grad():
        final = value(norm(hidden))
    outputs = [torch.empty(len(rows), vocab, dtype=F64) for rows in positions]
    block = 16384
    for first in range(0, vocab, block):
        head = w.rows("text/output_head", slice(first, min(vocab, first + block)))
        for b, rows in enumerate(positions):
            outputs[b][:, first:first + head.shape[0]] = exact.matmul(final[b, rows], head.t(), torch.float32)
    return outputs
