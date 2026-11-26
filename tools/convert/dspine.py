"""DSpine drafters (``DSpineDraftModel``: Tencent's Qwen3.8-27B-DSpine-b16) as standalone drafter artifacts.

The artifact's only component is ``dspine`` (docs/maintainer/dspine.md, "The artifact"), named with ``--draft-model``
beside a Qwen3.5-family model: its layers bind as a DFlash drafter's do, and its code bridge as below. The two
derived tables are computed here from the checkpoint's own tensors:

- ``dspine/bridge/projection`` [3R, C]: the bridge's gate, predecessor-gate and value (``prev``) weights, one below
  the other, as stored.
- ``dspine/bridge/sender_table`` [V, 2R]: every token's predecessor-gate and value rows,
  ``bf16(codes(codebook[v])) @ [predecessor_gate; prev]^T``, where ``codes`` is the Op ``dspine_bridge_codes``'s
  normalization (sqrt(C) x / (||x|| + eps), in FP32, rounded to BF16) and the product is formed in FP64 and rounded
  to BF16 once (order-independent; NInfer's FP32 product differs from it in about 0.01 % of the words, by its own
  summation order).
- ``dspine/bridge/head_write`` [R, H]: the last layer's write, transposed.
"""

from __future__ import annotations

import torch

from .model import Model, Parameter
from .sources.logical import array_source

COMPONENT = "dspine"


def is_dspine(config: dict) -> bool:
    return config.get("architectures") == ["DSpineDraftModel"]


def component_config(raw: dict) -> dict:
    """The stored config the engine's DSpine config reads (models/qwen3_5/config.cpp)."""
    draft = raw["dflash_config"]
    bridge = draft["selected_prefix_code_bridge"]
    second = raw.get("code_bridge_second_pass", {})
    rope = raw.get("rope_parameters") or {"rope_theta": raw["rope_theta"]}
    out = {key: raw[key] for key in (
        "architectures", "model_type", "hidden_size", "intermediate_size", "num_attention_heads",
        "num_key_value_heads", "head_dim", "num_hidden_layers", "max_position_embeddings", "vocab_size",
        "num_target_layers", "block_size", "rms_norm_eps", "layer_types", "sliding_window")}
    out["rope_parameters"] = {"rope_theta": float(rope["rope_theta"])}
    out["dspine_config"] = {
        "target_layer_ids": list(draft["target_layer_ids"]),
        "mask_token_id": draft["mask_token_id"],
        "code_size": bridge["code_size"],
        "rank": bridge["rank"],
        "bridge_eps": bridge["eps"],
        "head_top_k": second.get("head_topk", draft.get("selected_prefix_last_write_refine", {}).get("head_topk")),
    }
    return out


def _codes(x: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    return (x.shape[1] ** 0.5) * x / (x.norm(dim=1, keepdim=True) + eps)


def sender_table(codebook: torch.Tensor, predecessor_gate: torch.Tensor, value: torch.Tensor, eps: float,
                 rows_per_chunk: int = 16384) -> torch.Tensor:
    """[V, 2R] BF16: each token's predecessor-gate and value rows from its code (module docstring)."""
    weight = torch.cat([predecessor_gate, value], 0).double().T  # [C, 2R]
    out = torch.empty((codebook.shape[0], weight.shape[1]), dtype=torch.bfloat16)
    for begin in range(0, codebook.shape[0], rows_per_chunk):
        end = min(begin + rows_per_chunk, codebook.shape[0])
        codes = _codes(codebook[begin:end], eps).to(torch.bfloat16).double()
        out[begin:end] = (codes @ weight).to(torch.bfloat16)
    return out


def _direct(model: Model, name: str, values: torch.Tensor, label: str, inputs: tuple[str, ...] = ()) -> None:
    source = array_source(values.to(torch.bfloat16), label)

    def factory(selected, format=None):
        if format is not None:
            raise ValueError(f"{name}: a derived table takes no encoded source")
        return source

    model.add(Parameter(name, tuple(values.shape), source, factory, inputs, "bf16", residency=COMPONENT))


def build_model(store) -> Model:
    """A standalone DSpine drafter artifact's model from its checkpoint (``store``: its safetensors)."""
    from .qwen3_5 import _Builder

    raw = store.config
    if not is_dspine(raw):
        raise ValueError("not a DSpineDraftModel checkpoint")
    config = component_config(raw)
    h, vocab = config["hidden_size"], config["vocab_size"]
    bridge = config["dspine_config"]
    rank, code = bridge["rank"], bridge["code_size"]
    layers = config["num_hidden_layers"]
    # The builder's text config (the drafter binds the target's embedding and head at load) and its drafter view.
    target = {"hidden_size": h, "vocab_size": vocab}
    shim = dict(config, dflash_config={"target_layer_ids": bridge["target_layer_ids"]})
    model = Model({COMPONENT: {"config": config}}, target=target)
    builder = _Builder(model)
    builder.draft(store, shim, target, COMPONENT)

    prefix = "selected_prefix_bridge."

    def tensor(name: str) -> torch.Tensor:
        return store.read_flat(name).reshape(store.describe(name).shape)

    gate, predecessor, value = (tensor(prefix + n + ".weight") for n in ("gate", "predecessor_gate", "prev"))
    _direct(model, f"{COMPONENT}/bridge/projection", torch.cat([gate, predecessor, value], 0),
            "DSpine bridge projection: gate, predecessor_gate, prev", (f"{COMPONENT}/bridge/code_input",))
    builder.add(f"{COMPONENT}/bridge/gate_bias", store, prefix + "gate.bias", (rank,))
    early = tensor(prefix + "early_read_weights")
    writes = tensor(prefix + "feature_write_weights")
    if tuple(early.shape) != (layers - 1, code, h) or tuple(writes.shape) != (layers, h, rank):
        raise ValueError("DSpine bridge: the read and write stacks do not match the layers")
    for layer in range(layers):
        p = f"{COMPONENT}/bridge/layers/{layer}/"
        if layer + 1 < layers:
            builder.add(p + "read", store, prefix + "early_read_weights", (code, h), source_shape=tuple(early.shape),
                        offset=layer * code * h, inputs=(p + "read_input",))
        else:
            builder.add(p + "read", store, prefix + "read_weight", (code, h), inputs=(p + "read_input",))
        builder.add(p + "write", store, prefix + "feature_write_weights", (h, rank), source_shape=tuple(writes.shape),
                    offset=layer * h * rank, inputs=(p + "message",))
    _direct(model, f"{COMPONENT}/bridge/head_write", writes[layers - 1].T.contiguous(),
            "DSpine head write: the last layer's write, transposed")
    _direct(model, f"{COMPONENT}/bridge/sender_table",
            sender_table(tensor(prefix + "codebook"), predecessor, value, float(bridge["bridge_eps"])),
            "DSpine sender table: bf16(codes(codebook)) @ [predecessor_gate; prev]^T (FP64)")
    return model
