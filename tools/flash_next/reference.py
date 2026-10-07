"""Independent FP64 reference forward of Qwen3.8-Flash-Next (Qwen4Exp) on the NVIDIA checkpoint.

    python -m tools.flash_next.reference --model PATH --tokens 248045,9707,11 --out ref.npz

This is the model-level oracle for Infernix's Qwen4Exp implementation (AGENTS.md "Verification"). It
is written from the upstream ``qwen4_exp`` definitions (transformers ``modular_qwen4_exp.py`` and
the Qwen3.5 / Qwen3-Next blocks it reuses) and reads the checkpoint directly, never Infernix's
artifact or converter:

- every dense and BF16 tensor is widened exactly and every product is formed in binary64;
- routed experts use NVIDIA's NVFP4 words, decoded exactly, with activations quantized to A4 by the
  ModelOpt rule (design §16.2) on BF16 inputs with each matrix's own ``input_scale``;
- the PLE n-gram rows are the checkpoint's FP8 codes times its BF16 scale, rounded to BF16.

``--bf16-boundaries`` (default on) rounds the activations that deployed BF16 runtimes store in
BF16 -- the residual streams, every block input and output -- so differences against Infernix are
kernel arithmetic, not storage precision. QSA selection is evaluated exactly; for prompts up to
``indexer_budget + compress_ratio - 1`` tokens every token is selected and QSA equals causal
attention, which the script checks.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from tools.convert.sources.modelopt import nvfp4_matrix_words
from tools.convert.sources.safetensors import SafetensorsSource
from tools.flash_next.ngram import NgramConfig, head_tables, layer_multipliers, row_ids

F64 = torch.float64
_E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=F64)


def bf16(x: torch.Tensor) -> torch.Tensor:
    """Round to BF16 (nearest even) and widen back to binary64."""
    return x.to(torch.float32).to(torch.bfloat16).to(F64)


class Weights:
    def __init__(self, root: Path):
        self.store = SafetensorsSource(root)
        self.config = self.store.config["text_config"]
        self.prefix = "model.language_model."

    def get(self, name: str) -> torch.Tensor:
        info = self.store.describe(name)
        return self.store.read_flat(name).reshape(info.shape).to(F64)

    def text(self, name: str) -> torch.Tensor:
        return self.get(self.prefix + name)


# ---------------------------------------------------------------------------- primitives


def rmsnorm_1p(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int | None = None) -> torch.Tensor:
    """Qwen3.5 RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w), optionally per group of ``group``."""
    shape = x.shape
    if group is not None:
        x = x.reshape(*shape[:-1], -1, group)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y.reshape(shape) * (1.0 + weight)


def silu(x):
    return x * torch.sigmoid(x)


def softplus(x):
    return torch.nn.functional.softplus(x)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float, rotary: int) -> torch.Tensor:
    """Text-token MRoPE: every section uses the same position, so it is plain half-split RoPE on
    the first ``rotary`` dimensions. x: [T, H, D]."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary, 2, dtype=F64) / rotary))
    freqs = positions.to(F64)[:, None] * inv[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
    xr, xp = x[..., :rotary], x[..., rotary:]
    half = rotary // 2
    rotated = torch.cat((-xr[..., half:], xr[..., :half]), dim=-1)
    return torch.cat((xr * cos + rotated * sin, xp), dim=-1)


# ---------------------------------------------------------------------------- A4 and experts


def _e4m3_rn_satfinite(v: torch.Tensor) -> torch.Tensor:
    """FP32 -> E4M3FN value (as FP32), round-to-nearest-even with saturation to 448."""
    return v.to(torch.float32).clamp(max=448.0).to(torch.float8_e4m3fn).to(torch.float32)


def _e2m1_rn(v: torch.Tensor) -> torch.Tensor:
    """FP32 -> E2M1 value, round-to-nearest-even with saturation to 6 (ties to the even code)."""
    a = v.abs()
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    # Boundaries between grid values with the tie rule of design §16.2 (<= keeps the even code).
    code = torch.zeros_like(a, dtype=torch.int64)
    code = torch.where(a > 0.25, 1, code)
    code = torch.where(a >= 0.75, 2, code)
    code = torch.where(a > 1.25, 3, code)
    code = torch.where(a >= 1.75, 4, code)
    code = torch.where(a > 2.5, 5, code)
    code = torch.where(a >= 3.5, 6, code)
    code = torch.where(a > 5.0, 7, code)
    return torch.copysign(grid[code], v)


def a4_dequantized(x_bf16: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """ModelOpt NVFP4 activation rule in exact FP32 (design §16.2), returned dequantized in FP64:
    s = e4m3(amax / fl(6 g)); d = fl(e4m3(s) * g); code = e2m1(v / d); value = code * d."""
    g = input_scale.to(torch.float32)
    v = x_bf16.to(torch.float32).reshape(*x_bf16.shape[:-1], -1, 16)
    amax = v.abs().amax(-1, keepdim=True)
    s = _e4m3_rn_satfinite(amax / (torch.tensor(6.0, dtype=torch.float32) * g))
    d = s * g
    codes = torch.where(s > 0, _e2m1_rn(v / torch.where(s > 0, d, torch.ones_like(d))), torch.zeros_like(v))
    return (codes.to(F64) * d.to(F64)).reshape(x_bf16.shape)


class Experts:
    """Decodes each routed expert when it is used; one decoded expert is ~40 MB in binary64."""

    def __init__(self, weights: Weights):
        self.w = weights

    def _matrix(self, leaf: str, shape):
        words = nvfp4_matrix_words(self.w.store, leaf, shape)
        n, k = shape
        codes = torch.stack((words.codes & 0xF, words.codes >> 4), dim=-1).reshape(n, k).long()
        scales = words.scales.view(torch.float8_e4m3fn).to(F64).repeat_interleave(16, dim=-1)
        weight = _E2M1[codes] * scales * float(words.weight_scale_2)
        return weight, words.input_scale

    def get(self, layer: int, expert: int):
        leaf = f"{self.w.prefix}layers.{layer}.mlp.experts.{expert}."
        h, i = self.w.config["hidden_size"], self.w.config["moe_intermediate_size"]
        return (
            self._matrix(leaf + "gate_proj", (i, h)),
            self._matrix(leaf + "up_proj", (i, h)),
            self._matrix(leaf + "down_proj", (h, i)),
        )

    def forward(self, layer: int, expert: int, x_bf16: torch.Tensor) -> torch.Tensor:
        """One routed expert on rows x [n, H] (BF16 values): W4A4 with BF16 at its outputs."""
        (wg, sg), (wu, su), (wd, sd) = self.get(layer, expert)
        g = bf16(a4_dequantized(x_bf16, sg) @ wg.T)
        u = bf16(a4_dequantized(x_bf16, su) @ wu.T)
        h = bf16(silu(g) * u)
        return bf16(a4_dequantized(h, sd) @ wd.T)


# ---------------------------------------------------------------------------- model


class Reference:
    def __init__(self, root: Path, *, bf16_boundaries: bool = True):
        self.w = Weights(root)
        self.c = self.w.config
        self.eps = float(self.c["rms_norm_eps"])
        self.hc = int(self.c["hc_count"])
        self.H = int(self.c["hidden_size"])
        self.experts = Experts(self.w)
        self.round = bf16 if bf16_boundaries else (lambda x: x)
        rope = self.c["rope_parameters"]
        self.theta = float(rope["rope_theta"])
        self.rotary = int(self.c["head_dim"] * rope["partial_rotary_factor"])
        self.layer_types = [
            "full" if kind in ("full_attention", "qwen_sparse_attention") else "linear" for kind in self.c["layer_types"]
        ]
        eos = self.c["eos_token_id"]
        self.eos = eos[0] if isinstance(eos, list) else eos
        self.ple_layers = list(self.c.get("ple_layer_ids") or [])

    # -- hyper-connections
    def hc_mix(self, prefix: str, R: torch.Tensor, combine: bool = True):
        w = self.w
        Rn = rmsnorm_1p(R, w.text(prefix + "hc_norm.weight"), self.eps, group=self.H)
        Rn = self.round(Rn)
        z = Rn @ w.text(prefix + "input_mix_weight_down.weight").T
        m = silu(z / self.hc)
        u = torch.sigmoid(m @ w.text(prefix + "input_mix_weight_up.weight").T)
        T = R.shape[0]
        x = (u.reshape(T, self.hc, self.H) * Rn.reshape(T, self.hc, self.H)).mean(dim=1)
        if not combine:
            return self.round(x)
        inj = 2.0 * torch.sigmoid((Rn @ w.text(prefix + "block_inject_weight.weight").T) / self.hc)
        return self.round(x), inj

    def inject(self, R: torch.Tensor, y: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
        T = R.shape[0]
        out = R.reshape(T, self.hc, self.H) + y[:, None, :] * inj[:, :, None]
        return self.round(out.reshape(T, self.hc * self.H))

    # -- GDN
    def gdn(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        c, w = self.c, self.w
        p = f"layers.{layer}.linear_attn."
        nk, dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
        nv, dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
        T = x.shape[0]
        qkv = x @ w.text(p + "in_proj_qkv.weight").T
        z = x @ w.text(p + "in_proj_z.weight").T
        b = x @ w.text(p + "in_proj_b.weight").T
        a = x @ w.text(p + "in_proj_a.weight").T
        conv = w.text(p + "conv1d.weight")[:, 0, :]  # [C, taps]
        taps = conv.shape[1]
        padded = torch.cat((torch.zeros(taps - 1, qkv.shape[1], dtype=F64), qkv), dim=0)
        mixed = sum(conv[:, j][None, :] * padded[j : j + T] for j in range(taps))
        mixed = silu(mixed)
        q, k, v = mixed.split([nk * dk, nk * dk, nv * dv], dim=-1)
        q = q.reshape(T, nk, dk).repeat_interleave(nv // nk, dim=1)
        k = k.reshape(T, nk, dk).repeat_interleave(nv // nk, dim=1)
        v = v.reshape(T, nv, dv)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(b)
        g = -w.text(p + "A_log").exp() * softplus(a + w.text(p + "dt_bias"))
        S = torch.zeros(nv, dk, dv, dtype=F64)
        out = torch.zeros(T, nv, dv, dtype=F64)
        for t in range(T):
            S = S * g[t].exp()[:, None, None]
            kv = (S * k[t][:, :, None]).sum(dim=1)
            delta = (v[t] - kv) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            out[t] = (S * q[t][:, :, None]).sum(dim=1)
        on = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        gate = torch.sigmoid if (c.get("output_gate_type") or c["hidden_act"]) == "sigmoid" else silu
        on = w.text(p + "norm.weight") * on * gate(z.reshape(T, nv, dv))
        return self.round(on.reshape(T, nv * dv) @ w.text(p + "out_proj.weight").T)

    # -- QSA
    def qsa(self, layer: int, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        c, w = self.c, self.w
        p = f"layers.{layer}.self_attn."
        nh, nkv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        T = x.shape[0]
        qg = (x @ w.text(p + "q_proj.weight").T).reshape(T, nh, 2 * d)
        q, gate = qg[..., :d], qg[..., d:].reshape(T, nh * d)
        k = (x @ w.text(p + "k_proj.weight").T).reshape(T, nkv, d)
        v = (x @ w.text(p + "v_proj.weight").T).reshape(T, nkv, d)
        q = rope(rmsnorm_1p(q, w.text(p + "q_norm.weight"), self.eps), positions, self.theta, self.rotary)
        k = rope(rmsnorm_1p(k, w.text(p + "k_norm.weight"), self.eps), positions, self.theta, self.rotary)
        selected = self.qsa_select(layer, x, positions)
        rep = nh // nkv
        out = torch.zeros(T, nh, d, dtype=F64)
        for t in range(T):
            idx = selected[t]
            kk = k[idx].repeat_interleave(rep, dim=1)  # [n, nh, d]
            vv = v[idx].repeat_interleave(rep, dim=1)
            scores = torch.einsum("hd,nhd->hn", q[t], kk) / math.sqrt(d)
            prob = torch.softmax(scores, dim=-1)
            out[t] = torch.einsum("hn,nhd->hd", prob, vv)
        o = out.reshape(T, nh * d) * torch.sigmoid(gate)
        return self.round(o @ w.text(p + "o_proj.weight").T)

    def qsa_select(self, layer: int, x: torch.Tensor, positions: torch.Tensor) -> list[torch.Tensor]:
        """Upstream Qwen4Exp indexer: selected token positions per query (prefix of length T)."""
        c, w = self.c, self.w
        p = f"layers.{layer}.self_attn.indexer."
        nh, d, ratio = c["indexer_n_heads"], c["indexer_head_dim"], c["indexer_compress_ratio"]
        budget = c["indexer_budget"] // ratio
        T = x.shape[0]
        qk = x @ w.text(p + "index_qk_proj.weight").T
        q = qk[:, : nh * d].reshape(T, nh, d)
        raw = self.round(qk[:, nh * d :])  # [T, d], cached in BF16 upstream
        q = rope(rmsnorm_1p(q, w.text(p + "q_layernorm.weight"), self.eps), positions, self.theta, self.rotary)
        out = []
        for t in range(T):
            visible = t + 1
            blocks = visible // ratio
            chosen = []
            if blocks > 0:
                pooled = self.round(raw[: blocks * ratio].reshape(blocks, ratio, d).mean(1))
                pooled = rmsnorm_1p(pooled, w.text(p + "k_layernorm.weight"), self.eps)
                starts = positions[torch.arange(blocks) * ratio]
                pooled = rope(pooled[:, None, :], starts, self.theta, self.rotary)[:, 0]
                scores = torch.relu(q[t] @ pooled.T).sum(0) / math.sqrt(d)
                count = min(budget, blocks)
                # Ties are resolved toward the lower block id (the order Infernix implements).
                order = sorted(range(blocks), key=lambda b: (-float(scores[b]), b))[:count]
                for b in sorted(order):
                    chosen.extend(range(b * ratio, b * ratio + ratio))
            chosen.extend(range(blocks * ratio, visible))
            out.append(torch.tensor(chosen, dtype=torch.long))
        return out

    # -- MoE
    def moe(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        c, w = self.c, self.w
        p = f"layers.{layer}.mlp."
        # Router logits are not rounded: Infernix keeps them FP32 (as its Qwen3.5 sparse MoE and
        # llama.cpp-derived engines do), so near-boundary experts do not collapse into BF16 ties.
        # Exact ties go to the lower expert id.
        logits = x @ w.text(p + "gate.weight").T
        probs = torch.softmax(logits, dim=-1)
        order = torch.sort(probs, dim=-1, descending=True, stable=True).indices
        idx = order[:, : c["num_experts_per_tok"]]
        top = torch.gather(probs, -1, idx)
        top = top / top.sum(-1, keepdim=True)
        T = x.shape[0]
        routed = torch.zeros(T, self.H, dtype=F64)
        for e in torch.unique(idx).tolist():
            rows, slots = torch.where(idx == e)
            y = self.experts.forward(layer, e, bf16(x[rows]))
            routed.index_add_(0, rows, y * top[rows, slots][:, None])
        g = x @ w.text(p + "shared_expert.gate_proj.weight").T
        u = x @ w.text(p + "shared_expert.up_proj.weight").T
        shared = (silu(g) * u) @ w.text(p + "shared_expert.down_proj.weight").T
        shared = shared * torch.sigmoid(x @ w.text(p + "shared_expert_gate.weight").T)
        return self.round(routed + shared), idx

    # -- PLE
    def ple(self, layer: int, ple_index: int, R: torch.Tensor, ids: list[int]) -> torch.Tensor:
        c, w = self.c, self.w
        p = f"layers.{layer}.ple."
        spec = NgramConfig(
            vocab_size=c["vocab_size"], eos_token_id=self.eos, ngram_size=c["ngram_size"],
            heads_per_ngram=c["heads_per_ngram"], ngram_vocab_size_base=c["ngram_vocab_size_base"],
            make_ngram_vocab_size_divisible_by=c["make_ngram_vocab_size_divisible_by"], seed=c.get("seed", 1234),
        )
        history = [self.eos] * (spec.ngram_size - 1) + list(ids)
        rows = row_ids(spec, ple_index, history, len(ids))
        _, _, padded = head_tables(spec, ple_index)
        parts = c["split_ngram_parts"]
        per = padded // parts
        scale = w.text(p + "ple_embedding.ngram_embedding.weight_scale")
        T = R.shape[0]
        emb = []
        for t in range(T):
            vals = []
            for r in rows[t]:
                shard, local = divmod(r, per)
                name = f"{w.prefix}{p}ple_embedding.ngram_embedding.shard_{shard}.weight"
                width = w.store.describe(name).shape[1]
                raw = w.store.read_flat(name, local * width, (local + 1) * width).view(torch.float8_e4m3fn)
                vals.append(bf16(raw.to(F64) * scale))
            emb.append(torch.cat(vals))
        e = torch.stack(emb)  # [T, 2560]
        key = rmsnorm_1p(e @ w.text(p + "key_proj.weight").T, w.text(p + "norm_key.weight"), self.eps, group=self.H)
        value = e @ w.text(p + "value_proj.weight").T
        query = rmsnorm_1p(R, w.text(p + "norm_query.weight"), self.eps, group=self.H)
        gate = (key.reshape(T, self.hc, self.H) * query.reshape(T, self.hc, self.H)).sum(-1) / math.sqrt(self.H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate)[:, :, None] * value[:, None, :]).reshape(T, self.hc * self.H)
        normed = rmsnorm_1p(gated, w.text(p + "norm_conv.weight"), self.eps, group=self.H)
        kernel = w.text(p + "conv1d.weight")[:, 0, :]  # [C, taps]
        taps, dilation = kernel.shape[1], c["ngram_size"]
        span = (taps - 1) * dilation
        padded_in = torch.cat((torch.zeros(span, normed.shape[1], dtype=F64), normed), dim=0)
        conv = sum(kernel[:, j][None, :] * padded_in[j * dilation : j * dilation + T] for j in range(taps))
        return self.round(gated + silu(conv)), rows

    # -- full forward
    def forward(self, ids: list[int], *, layers: int | None = None, log=print) -> dict:
        c, w = self.c, self.w
        T = len(ids)
        positions = torch.arange(T)
        name = w.prefix + "embed_tokens.weight"
        x = torch.stack([w.store.read_flat(name, i * self.H, (i + 1) * self.H) for i in ids]).to(F64)
        R = x.repeat(1, self.hc)
        count = c["num_hidden_layers"] if layers is None else layers
        record = {"layer_residual_rms": [], "layer_residuals": [], "routed": [], "ngram_rows": None}
        start = time.time()
        for layer in range(count):
            if layer + 1 in self.ple_layers:
                out, rows = self.ple(layer, self.ple_layers.index(layer + 1), R, ids)
                R = self.round(R + out)
                record["ngram_rows"] = np.array(rows, dtype=np.int64)
            xa, inj = self.hc_mix(f"layers.{layer}.attn_hyper_connection.", R)
            if self.layer_types[layer] == "linear":
                y = self.gdn(layer, xa)
            else:
                y = self.qsa(layer, xa, positions)
            R = self.inject(R, y, inj)
            xm, inj = self.hc_mix(f"layers.{layer}.mlp_hyper_connection.", R)
            y, idx = self.moe(layer, xm)
            R = self.inject(R, y, inj)
            record["layer_residual_rms"].append(float(R.pow(2).mean().sqrt()))
            record["routed"].append(idx.numpy())
            record["layer_residuals"].append(R.to(torch.float32).numpy())
            log(f"layer {layer:2d} {self.layer_types[layer]:6s} rms {record['layer_residual_rms'][-1]:.5f} "
                f"({time.time() - start:.0f}s)")
        record["final_residual"] = R.to(torch.float32).numpy()
        if layers is None or layers == c["num_hidden_layers"]:
            xf = self.hc_mix("hyper_connection_mixer.", R, combine=False)
            # Row chunks keep the binary64 widening of the [V, H] head to ~0.3 GB at a time.
            name, rows = "lm_head.weight", w.store.describe("lm_head.weight").shape[0]
            chunks = []
            for begin in range(0, rows, 16384):
                end = min(rows, begin + 16384)
                head = w.store.read_flat(name, begin * self.H, end * self.H).reshape(end - begin, self.H)
                chunks.append(xf @ head.to(F64).T)
            record["logits"] = torch.cat(chunks, dim=-1).to(torch.float32).numpy()
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--layers", type=int, help="stop after this many layers (no logits)")
    parser.add_argument("--no-bf16-boundaries", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    ids = [int(v) for v in args.tokens.split(",") if v]
    ref = Reference(args.model, bf16_boundaries=not args.no_bf16_boundaries)
    record = ref.forward(ids, layers=args.layers)
    arrays = {"tokens": np.array(ids, dtype=np.int64), "final_residual": record["final_residual"]}
    if "logits" in record:
        arrays["logits"] = record["logits"]
        top = np.argsort(-record["logits"], axis=-1)[:, :5]
        print("top-5 next tokens per position:", top.tolist())
    if record["ngram_rows"] is not None:
        arrays["ngram_rows"] = record["ngram_rows"]
    arrays["routed"] = np.stack(record["routed"])
    arrays["layer_residuals"] = np.stack(record["layer_residuals"])
    np.savez(args.out, **arrays)
    Path(str(args.out) + ".json").write_text(json.dumps({"layer_residual_rms": record["layer_residual_rms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
