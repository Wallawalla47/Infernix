"""A tiny random Qwen3.5 (two layers: GatedDeltaNet and attention; and an MoE variant) through the shadow, on
transformers 5.5.0: the forward runs end to end, is deterministic, and stays close to the same modules
evaluated in plain binary64 (which round nowhere). Run by test_shadow_qwen3_5.py with 5.5 on PYTHONPATH."""

from __future__ import annotations

import sys

import torch

from tools.quality.shadow import exact, qwen3_5
from tools.quality.shadow.overrides import set_parameter

F64 = torch.float64


class FakeWeights:
    def __init__(self, config: dict, seed: int):
        self.config = config
        self.values: dict[str, torch.Tensor] = {}
        self.g = torch.Generator().manual_seed(seed)

    def logical(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if name not in self.values:
            scale = 1.0 if name.endswith(("norm", "_norm")) else 0.2
            v = torch.randn(shape, generator=self.g, dtype=F64) * scale
            if name.endswith(("a_log", "dt_bias")):
                v = exact.round_to(v.abs() + 0.5, torch.float32)
            else:
                v = exact.round_to(v, torch.bfloat16)
            self.values[name] = v
        return self.values[name].reshape(shape)

    def nominal(self, name: str) -> torch.dtype:
        return torch.float32 if name.endswith(("a_log", "dt_bias")) else torch.bfloat16

    def rows(self, name: str, rows) -> torch.Tensor:
        full = self.logical(name, (self.config["vocab_size"], self.config["hidden_size"]))
        return full[rows] if not isinstance(rows, list) else full[torch.tensor(rows)]

    def release(self) -> None:
        pass


def config(moe: bool) -> dict:
    c = {"model_type": "qwen3_5_moe_text" if moe else "qwen3_5_text", "hidden_size": 64, "vocab_size": 97,
         "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"], "max_position_embeddings": 64,
         "tie_word_embeddings": False, "rms_norm_eps": 1e-6, "num_attention_heads": 4, "num_key_value_heads": 2,
         "head_dim": 16, "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                             "mrope_section": [2, 1, 1]},
         "linear_num_key_heads": 2, "linear_key_head_dim": 16, "linear_num_value_heads": 4,
         "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4}
    if moe:
        c.update(num_experts=6, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)
    else:
        c["intermediate_size"] = 96
    return c


def plain(w: FakeWeights, sequences, positions) -> list[torch.Tensor]:
    """The same transformers modules in binary64 throughout (eager attention, the torch fallbacks)."""
    from transformers.models.qwen3_5 import configuration_qwen3_5 as dc, modeling_qwen3_5 as dm
    from transformers.models.qwen3_5_moe import configuration_qwen3_5_moe as mc, modeling_qwen3_5_moe as mm
    from tools.quality.shadow.weights import qwen3_5_layer
    c = dict(w.config)
    moe = "num_experts" in c
    c["rope_parameters"] = {"rope_type": "default", **c["rope_parameters"]}
    cfg = (mc.Qwen3_5MoeTextConfig if moe else dc.Qwen3_5TextConfig)(
        **{k: v for k, v in c.items() if k not in ("architectures", "model_type")})
    cfg._attn_implementation = "eager"
    if moe:
        cfg._experts_implementation = "eager"
    m = mm if moe else dm
    prefix = "Qwen3_5Moe" if moe else "Qwen3_5"
    tokens = torch.tensor(sequences)
    hidden = w.logical("text/token_embedding", (c["vocab_size"], c["hidden_size"]))[tokens]
    rotary = getattr(m, prefix + "TextRotaryEmbedding")(cfg).to(F64)
    position_ids = torch.arange(tokens.shape[1])[None].expand(*tokens.shape)
    cos, sin = rotary(hidden, position_ids)
    length = tokens.shape[1]
    causal = torch.full((length, length), -torch.inf, dtype=F64).triu(1)[None, None]
    for i in range(c["num_hidden_layers"]):
        layer = getattr(m, prefix + "DecoderLayer")(cfg, i).to(F64)
        for name, (values, _) in qwen3_5_layer(w, i).items():
            set_parameter(layer, name, torch.nn.Parameter(values, requires_grad=False))
        mask = causal if c["layer_types"][i] == "full_attention" else None
        with torch.no_grad():
            hidden = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask, position_ids=position_ids)
    norm = getattr(m, prefix + "RMSNorm")(c["hidden_size"], eps=c["rms_norm_eps"]).to(F64)
    norm.weight.data = w.logical("text/final_norm", (c["hidden_size"],))
    with torch.no_grad():
        final = norm(hidden)
    head = w.logical("text/output_head", (c["vocab_size"], c["hidden_size"]))
    return [final[b, rows] @ head.t() for b, rows in enumerate(positions)]


def main() -> int:
    import transformers
    assert transformers.__version__ == qwen3_5.AUTHORITY, transformers.__version__
    sequences = [[(7 * i + 3 * s) % 97 for i in range(24)] for s in range(2)]
    positions = [[5, 23], [0, 11, 23]]
    for moe in (False, True):
        w = FakeWeights(config(moe), 1)
        a = qwen3_5.run(w, sequences, positions, log=lambda m: None)
        b = qwen3_5.run(w, sequences, positions, log=lambda m: None)
        assert all(torch.equal(x, y) for x, y in zip(a, b)), "the shadow is not deterministic"
        for rows in a:
            assert torch.equal(exact.round_to(rows, torch.float32), rows), "logits are not FP32 values"
            assert torch.isfinite(rows).all()
        # the bisect's record: the embedding, then each layer's mixer and layer, BF16 values, and recording
        # changes nothing
        record: dict = {}
        c = config(moe)
        c_layers = c["num_hidden_layers"]
        recorded = qwen3_5.run(w, sequences, positions, log=lambda m: None, record=record)
        assert all(torch.equal(x, y) for x, y in zip(a, recorded)), "recording changed the logits"
        assert set(record) == {"embed", *[(i, t) for i in range(c_layers) for t in ("mixer", "layer")]}, sorted(
            map(str, record))
        for key, stream in record.items():
            assert tuple(stream.shape) == (2, 24, c["hidden_size"]), (key, stream.shape)
            assert torch.equal(exact.round_to(stream, torch.bfloat16), stream), f"{key} is not BF16 values"
        assert not torch.equal(record[(0, "mixer")], record["embed"]), "a mixer changed nothing"
        assert qwen3_5.run(w, sequences, positions, log=lambda m: None, record={}, stop_after_layers=True) == []
        linear = [i for i, kind in enumerate(c["layer_types"]) if kind == "linear_attention"]
        detailed: dict = {}
        qwen3_5.run(w, sequences, positions, log=lambda m: None, record=detailed, stop_after_layers=True,
                    detail=frozenset(linear[:1]))
        names = {"mixed", "z", "a", "b", "g", "beta", "conv", "o", "on", "y"}
        assert {k[2] for k in detailed if isinstance(k, tuple) and len(k) == 3} == names, sorted(map(str, detailed))
        for name in names:
            stream = detailed[(linear[0], "gdn", name)]
            assert stream.shape[0] == 2 and stream.shape[1] == 24, (name, stream.shape)
        assert torch.equal(detailed[(linear[0], "layer")], record[(linear[0], "layer")]), "detail changed the stream"
        reference = plain(w, sequences, positions)
        scale = max(float(x.abs().max()) for x in reference)
        worst = max(float((x - y).abs().max()) for x, y in zip(a, reference))
        agree = sum(int((x.argmax(-1) == y.argmax(-1)).sum()) for x, y in zip(a, reference))
        print(f"{'moe' if moe else 'dense'}: max |logit| {scale:.3f}, max |shadow - binary64| {worst:.4f}, "
              f"top-1 equal {agree}/5")
        assert worst < 0.05 * scale, "the shadow is far from the binary64 evaluation"
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
