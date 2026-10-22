"""A tiny random Qwen3.8-Flash-Next through the shadow (tools/quality/shadow/qwen4_exp.py) on its authority,
transformers main 3bb8b727: GatedDeltaNet with PLE (n-gram rows from a real volume file), QSA attention whose
indexer selects a subset of the keys, hyper-connections and lazily read experts. The forward runs end to
end, is deterministic, and stays close to the same modules in plain binary64 (eager attention, the torch
fallbacks, the whole n-gram table and expert banks as module weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
from tools.quality.shadow import exact, qwen4_exp  # noqa: E402
from tools.quality.shadow.overrides import set_parameter  # noqa: E402
from tools.quality.shadow.weights import qwen4_exp_layer  # noqa: E402

F64 = torch.float64
pytestmark = pytest.mark.skipif(transformers.__version__ != qwen4_exp.AUTHORITY,
                                reason="Flash-Next's authority is transformers main 3bb8b727")

CONFIG = {"model_type": "qwen4_exp_text", "hidden_size": 32, "vocab_size": 61, "num_hidden_layers": 2,
          "layer_types": ["linear_attention", "full_attention"], "max_position_embeddings": 64,
          "tie_word_embeddings": False, "rms_norm_eps": 1e-6, "num_attention_heads": 2, "num_key_value_heads": 1,
          "head_dim": 16, "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                              "mrope_section": [2, 1, 1]},
          "linear_num_key_heads": 2, "linear_key_head_dim": 8, "linear_num_value_heads": 4,
          "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4, "output_gate_type": "sigmoid",
          "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
          "shared_expert_intermediate_size": 16, "norm_topk_prob": True, "hc_count": 2, "hc_lowrank": 8,
          "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 16, "indexer_budget": 8,
          "indexer_compress_ratio": 2, "ple_layer_ids": [1], "ple_embed_dim": 32, "ple_conv_kernel_size": 3,
          "ngram_size": 3, "heads_per_ngram": 2, "ngram_vocab_size_base": 101,
          "make_ngram_vocab_size_divisible_by": 8, "seed": 1234, "split_ngram_parts": 1, "eos_token_id": 60}


class FakeWeights:
    def __init__(self, config: dict, table_rows: int, seed: int):
        self.config = dict(config, ngram_table={"format": "bf16", "rows": table_rows, "row_bytes": 16,
                                                "rows_per_block": 256, "block_bytes": 4096, "header_bytes": 4096})
        self.values: dict[str, torch.Tensor] = {}
        self.g = torch.Generator().manual_seed(seed)

    def logical(self, name, shape):
        if name not in self.values:
            v = torch.randn(shape, generator=self.g, dtype=F64) * (1.0 if "norm" in name else 0.25)
            self.values[name] = exact.round_to(v.abs() + 0.5, torch.float32) if name.endswith(("a_log", "dt_bias")) \
                else exact.round_to(v, torch.bfloat16)
        return self.values[name].reshape(shape)

    def nominal(self, name):
        return torch.float32 if name.endswith(("a_log", "dt_bias")) else torch.bfloat16

    def rows(self, name, rows):
        full = self.logical(name, (self.config["vocab_size"], self.config["hidden_size"]))
        return full[torch.tensor(rows)] if isinstance(rows, list) else full[rows]

    def expert_bank(self, name, e):
        c = self.config
        i, h = c["moe_intermediate_size"], c["hidden_size"]
        return (self.logical(f"{name}/{e}/gate", (i, h)), self.logical(f"{name}/{e}/up", (i, h)),
                self.logical(f"{name}/{e}/down", (h, i)))

    def release(self):
        pass


def write_volume(path: Path, table: torch.Tensor) -> None:
    words = (table.to(torch.float32).view(torch.int32) >> 16).to(torch.int32).numpy().astype("<u2")
    rows, dim = words.shape
    blob = bytearray(4096 + ((rows + 255) // 256) * 4096)
    for r in range(rows):
        offset = 4096 + (r // 256) * 4096 + (r % 256) * dim * 2
        blob[offset:offset + dim * 2] = words[r].tobytes()
    path.write_bytes(bytes(blob))


def plain(w: FakeWeights, table: torch.Tensor, sequences, positions):
    config_module, m = qwen4_exp._transformers()
    saved = {name: getattr(m, name) for name in qwen4_exp.ORIGINALS}
    for name, original in qwen4_exp.ORIGINALS.items():
        setattr(m, name, original)
    try:
        c = {k: v for k, v in w.config.items() if k not in ("ngram_table", "model_type")}
        c["rope_parameters"] = {"rope_type": "default", **c["rope_parameters"]}
        cfg = config_module.Qwen4ExpTextConfig(**c)
        cfg._attn_implementation = "eager"
        cfg._experts_implementation = "eager"
        tokens = torch.tensor(sequences)
        length = tokens.shape[1]
        hidden = w.logical("text/token_embedding", (c["vocab_size"], c["hidden_size"]))[tokens]
        rotary = m.Qwen4ExpTextRotaryEmbedding(cfg).to(F64)
        position_ids = torch.arange(length)[None].expand(*tokens.shape)
        cos, sin = rotary(hidden, position_ids)
        hidden = hidden.repeat(1, 1, c["hc_count"])
        mask = torch.full((length, length), torch.finfo(F64).min, dtype=F64).triu(1)[None, None].expand(
            tokens.shape[0], 1, length, length)
        for i in range(c["num_hidden_layers"]):
            layer = m.Qwen4ExpTextDecoderLayer(cfg, i).to(F64)
            for name, (values, _) in qwen4_exp_layer(w, i).items():
                set_parameter(layer, name, torch.nn.Parameter(values, requires_grad=False))
            banks = [w.expert_bank(f"text/layers/{i}/moe/experts", e) for e in range(c["num_experts"])]
            layer.mlp.experts.gate_up_proj.data = torch.stack([torch.cat([g, u]) for g, u, _ in banks])
            layer.mlp.experts.down_proj.data = torch.stack([d for _, _, d in banks])
            if layer.ple is not None:
                emb = layer.ple.ple_embedding.ngram_embedding
                emb.weight.data = table[: emb.weight.shape[0]].to(F64)
            with torch.no_grad():
                hidden = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask, conv_mask=None,
                               past_key_values=None, ple_input_ids=tokens)
        mixer = m.Qwen4ExpTextGatedResidual(cfg, use_combine=False).to(F64)
        width, rank = c["hc_count"] * c["hidden_size"], c["hc_lowrank"]
        mixer.hc_norm.weight.data = w.logical("text/final_mixer/norm", (width,))
        mixer.input_mix_weight_down.weight.data = w.logical("text/final_mixer/down", (rank, width))
        mixer.input_mix_weight_up.weight.data = w.logical("text/final_mixer/up", (width, rank))
        with torch.no_grad():
            final = mixer(hidden)
        head = w.logical("text/output_head", (c["vocab_size"], c["hidden_size"]))
        return [final[b, rows] @ head.t() for b, rows in enumerate(positions)]
    finally:
        for name, function in saved.items():
            setattr(m, name, function)


def test_tiny_flash_next_through_the_shadow(tmp_path: Path):
    g = torch.Generator().manual_seed(9)
    table = exact.round_to(torch.randn(512, 8, generator=g, dtype=F64) * 0.5, torch.bfloat16)
    volume = tmp_path / "tiny.ngram"
    write_volume(volume, table)
    sequences = [[(5 * i + 11 * s) % 61 for i in range(24)] for s in range(2)]
    positions = [[3, 23], [0, 15, 23]]
    w = FakeWeights(CONFIG, len(table), 1)
    a = qwen4_exp.run(w, sequences, positions, log=lambda m: None, ngram_volume=str(volume))
    b = qwen4_exp.run(w, sequences, positions, log=lambda m: None, ngram_volume=str(volume))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    for rows in a:
        assert torch.isfinite(rows).all() and torch.equal(exact.round_to(rows, torch.float32), rows)
    reference = plain(w, table, sequences, positions)
    scale = max(float(x.abs().max()) for x in reference)
    worst = max(float((x - y).abs().max()) for x, y in zip(a, reference))
    assert worst < 0.05 * scale, (worst, scale)
