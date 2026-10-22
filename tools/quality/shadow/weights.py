"""An artifact's parameters decoded exactly and placed where the publisher's model holds them.

Every stored format decodes to its exact real values (tools/artifact/codecs): a row-split code times its
FP16 group scale, row FP8 an E4M3 code times its BF16 row scale, block FP8 times its BF16 tile scale,
NVFP4 an E2M1 code times its E4M3 block scale and FP32 multiplier; BF16 and FP32 words as they are. The
values stay binary64; a parameter's nominal dtype is FP32 where the artifact stores FP32 words and BF16
otherwise (the dtype the checkpoint loads in), so quantized weights keep their exact values, as on the
lossless route, rather than a BF16 rounding of them.

The placement inverts tools/convert/qwen3_5.py's logical model (the converter's source names, row ranges
and transposes) for the text decoder.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from tools.artifact.codecs.direct import decode_direct
from tools.artifact.codecs.fp8_b128 import decode_expert as decode_fp8_expert, dequantize as dequantize_fp8_tiles
from tools.artifact.codecs.fp8_row import dequantize_fp8_row_scaled
from tools.artifact.codecs.nvfp4_expert import decode_nvfp4_expert_bank_words, dequantize_nvfp4_mul
from tools.artifact.codecs.row_split import dequantize_row_split, gather_row_planes, split_row_planes
from tools.artifact.layouts import fp8_expert_geometry, row_split_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts

F64 = torch.float64
_ROW_SPLIT = {"q2_g64_fp16", "q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16", "q8_g32_fp16"}


class Weights:
    def __init__(self, path: str, cache_objects: int = 8):
        self.artifact = Artifact.open(path)
        directory = self.artifact.directory
        self.bindings = directory.bindings
        self.objects = {o.id: o for o in directory.objects}
        self.config = directory.components["text"]["config"]
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._limit = cache_objects
        self._payloads: dict[str, bytes] = {}

    def decode(self, object_id: str) -> torch.Tensor:
        """A whole object's exact values ([rows, cols] for matrices), cached for the bindings that share it."""
        if object_id in self._cache:
            self._cache.move_to_end(object_id)
            return self._cache[object_id]
        obj = self.objects[object_id]
        payload = self.artifact.read_object(object_id)
        if obj.format in ("bf16", "fp32", "int32"):
            values = decode_direct(payload, obj.format, obj.shape).to(F64)
        elif obj.format in _ROW_SPLIT:
            values = dequantize_row_split(payload, obj.format, obj.shape, dtype=torch.float32).to(F64)
        elif obj.format == "fp8_e4m3fn_row_bf16":
            values = dequantize_fp8_row_scaled(payload, obj.shape, dtype=F64)
        else:
            raise NotImplementedError(f"{object_id}: {obj.format} is decoded per expert (expert_bank)")
        self._cache[object_id] = values
        while len(self._cache) > self._limit:
            self._cache.popitem(last=False)
        return values

    def rows(self, name: str, rows: slice | list[int]) -> torch.Tensor:
        """A row range or row list of a 2-D parameter stored as one whole object (the embedding, the head),
        decoded without the rest of the matrix."""
        (oid, begin, end), = binding_parts(self.bindings[name], self.objects, name)
        obj = self.objects[oid]
        n, k = obj.shape
        if begin != 0 or end != n * k:
            raise NotImplementedError(f"{name}: rows of a part of an object")
        if oid not in self._payloads:
            self._payloads[oid] = self.artifact.read_object(oid)
        payload = self._payloads[oid]
        if obj.format in _ROW_SPLIT:
            geometry = row_split_geometry(obj.format, obj.shape)
            if isinstance(rows, slice):
                count = rows.stop - rows.start
                planes = split_row_planes(payload, geometry, rows.start, count)
            else:
                count = len(rows)
                planes = gather_row_planes(payload, geometry, rows)
            return dequantize_row_split(planes, obj.format, (count, k), dtype=torch.float32).to(F64)
        if obj.format in ("bf16", "fp32"):
            return decode_direct(payload, obj.format, obj.shape)[rows].to(F64)
        if obj.format == "fp8_e4m3fn_row_bf16":
            return self.decode(oid)[rows]
        raise NotImplementedError(f"{name}: rows of {obj.format}")

    def nominal(self, name: str) -> torch.dtype:
        formats = {self.objects[oid].format for oid, _, _ in binding_parts(self.bindings[name], self.objects, name)}
        return torch.float32 if formats == {"fp32"} else torch.bfloat16

    def logical(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        """One logical parameter's exact values in its logical shape (row-major element ranges of its objects)."""
        chunks = [self.decode(oid).reshape(-1)[begin:end]
                  for oid, begin, end in binding_parts(self.bindings[name], self.objects, name)]
        values = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        return values.reshape(shape)

    def expert_bank(self, name: str, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(gate [I, H], up [I, H], down [H, I]) of one routed expert of a bank object, exact."""
        (oid, _, _), = binding_parts(self.bindings[name], self.objects, name)
        obj = self.objects[oid]
        experts, hidden, intermediate = obj.shape
        if obj.format == "fp8_e4m3fn_b128":
            geometry = fp8_expert_geometry(obj.format, obj.shape)
            stride = geometry.record_stride
            raw = self.artifact.read_range(obj.offset + expert * stride, geometry.record_bytes)
            (gc, gs), (uc, us), (dc, ds) = decode_fp8_expert(raw, hidden, intermediate)
            return tuple(torch.from_numpy(dequantize_fp8_tiles(c, s)).to(F64) for c, s in ((gc, gs), (uc, us), (dc, ds)))
        if obj.format == "nvfp4_mul":
            words = self._bank_words(oid)
            m = words["multipliers"][expert].tolist()
            return (dequantize_nvfp4_mul(words["gate_codes"][expert], words["gate_scales"][expert], m[0]),
                    dequantize_nvfp4_mul(words["up_codes"][expert], words["up_scales"][expert], m[1]),
                    dequantize_nvfp4_mul(words["down_codes"][expert], words["down_scales"][expert], m[2]))
        raise NotImplementedError(f"{name}: expert bank format {obj.format}")

    def _bank_words(self, oid: str) -> dict:
        key = "bank:" + oid
        if key not in self._cache:
            obj = self.objects[oid]
            self._cache[key] = decode_nvfp4_expert_bank_words(self.artifact.read_object(oid), obj.shape)
            while len(self._cache) > self._limit:
                self._cache.popitem(last=False)
        return self._cache[key]

    def release(self) -> None:
        self._cache.clear()


def _attention(w: Weights, p: str, put, out: dict) -> None:
    """Gated attention's projections and norms (Qwen3.5 and Qwen3.8-Flash-Next share them)."""
    c, h = w.config, w.config["hidden_size"]
    d, nq, nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    q = w.logical(p + "attention/query", (nq, d, h))
    gate = w.logical(p + "attention/gate", (nq, d, h))
    # q_proj's rows per head: the head's query rows, then its gate rows.
    out["self_attn.q_proj.weight"] = (torch.stack([q, gate], dim=1).reshape(2 * nq * d, h),
                                      w.nominal(p + "attention/query"))
    put("self_attn.k_proj.weight", p + "attention/key", (nkv * d, h))
    put("self_attn.v_proj.weight", p + "attention/value", (nkv * d, h))
    put("self_attn.o_proj.weight", p + "attention/output", (h, nq * d))
    put("self_attn.q_norm.weight", p + "attention/query_norm", (d,))
    put("self_attn.k_norm.weight", p + "attention/key_norm", (d,))


def _gdn(w: Weights, p: str, put, out: dict) -> None:
    c, h = w.config, w.config["hidden_size"]
    kg = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    nv, dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
    vg, taps = nv * dv, c["linear_conv_kernel_dim"]
    g = p + "gdn/"
    out["linear_attn.in_proj_qkv.weight"] = (torch.cat([w.logical(g + "query", (kg, h)), w.logical(g + "key", (kg, h)),
                                                       w.logical(g + "value", (vg, h))]), w.nominal(g + "query"))
    put("linear_attn.in_proj_z.weight", g + "z", (vg, h))
    put("linear_attn.in_proj_b.weight", g + "b_projection", (nv, h))
    put("linear_attn.in_proj_a.weight", g + "a_projection", (nv, h))
    conv = w.logical(g + "convolution", (taps, 2 * kg + vg))
    out["linear_attn.conv1d.weight"] = (conv.t().unsqueeze(1).contiguous(), w.nominal(g + "convolution"))
    put("linear_attn.A_log", g + "a_log", (nv,))
    put("linear_attn.dt_bias", g + "dt_bias", (nv,))
    put("linear_attn.norm.weight", g + "norm", (dv,))
    put("linear_attn.out_proj.weight", g + "output", (h, vg))


def _putter(w: Weights, out: dict):
    def put(hf: str, logical: str, shape: tuple[int, ...]) -> None:
        out[hf] = (w.logical(logical, shape), w.nominal(logical))
    return put


def qwen4_exp_layer(w: Weights, i: int) -> dict[str, tuple[torch.Tensor, torch.dtype]]:
    """A Qwen3.8-Flash-Next decoder layer's parameters by their transformers names (Qwen4ExpTextDecoderLayer),
    except the routed experts (read per expert: Weights.expert_bank) and the n-gram table (the volume)."""
    c = w.config
    p, h = f"text/layers/{i}/", c["hidden_size"]
    width, rank = c["hc_count"] * h, c["hc_lowrank"]
    out: dict[str, tuple[torch.Tensor, torch.dtype]] = {}
    put = _putter(w, out)
    for hf, logical in (("attn_hyper_connection.", "attn_hc/"), ("mlp_hyper_connection.", "mlp_hc/")):
        put(hf + "hc_norm.weight", p + logical + "norm", (width,))
        put(hf + "input_mix_weight_down.weight", p + logical + "down", (rank, width))
        put(hf + "input_mix_weight_up.weight", p + logical + "up", (width, rank))
        put(hf + "block_inject_weight.weight", p + logical + "inject", (c["hc_count"], width))
    if c["layer_types"][i] == "full_attention":
        _attention(w, p, put, out)
        iq, ik = c["indexer_n_heads"] * c["indexer_head_dim"], c["indexer_kv_heads"] * c["indexer_head_dim"]
        out["self_attn.indexer.index_qk_proj.weight"] = (
            torch.cat([w.logical(p + "attention/index_query", (iq, h)), w.logical(p + "attention/index_key", (ik, h))]),
            w.nominal(p + "attention/index_query"))
        put("self_attn.indexer.q_layernorm.weight", p + "attention/index_query_norm", (c["indexer_head_dim"],))
        put("self_attn.indexer.k_layernorm.weight", p + "attention/index_key_norm", (c["indexer_head_dim"],))
    else:
        _gdn(w, p, put, out)
    m, shared = p + "moe/", c["shared_expert_intermediate_size"]
    put("mlp.gate.weight", m + "router", (c["num_experts"], h))
    put("mlp.shared_expert_gate.weight", m + "shared_score", (1, h))
    for role in ("gate", "up", "down"):
        put(f"mlp.shared_expert.{role}_proj.weight", m + f"shared/{role}", (h, shared) if role == "down" else (shared, h))
    if i + 1 in c.get("ple_layer_ids", []):
        e, taps = c["ple_embed_dim"], c["ple_conv_kernel_size"]
        put("ple.key_proj.weight", p + "ple/key_projection", (width, e))
        put("ple.value_proj.weight", p + "ple/value_projection", (h, e))
        put("ple.norm_key.weight", p + "ple/key_norm", (width,))
        put("ple.norm_query.weight", p + "ple/query_norm", (width,))
        put("ple.norm_conv.weight", p + "ple/conv_norm", (width,))
        conv = w.logical(p + "ple/convolution", (taps, width))
        out["ple.conv1d.weight"] = (conv.t().unsqueeze(1).contiguous(), w.nominal(p + "ple/convolution"))
    return out


def qwen3_5_layer(w: Weights, i: int) -> dict[str, tuple[torch.Tensor, torch.dtype]]:
    """A Qwen3.5 decoder layer's parameters by their transformers names (Qwen3_5DecoderLayer and the MoE one)."""
    c = w.config
    p, h = f"text/layers/{i}/", c["hidden_size"]
    out: dict[str, tuple[torch.Tensor, torch.dtype]] = {}
    put = _putter(w, out)
    put("input_layernorm.weight", p + "input_norm", (h,))
    put("post_attention_layernorm.weight", p + "post_attention_norm", (h,))
    if c["layer_types"][i] == "full_attention":
        _attention(w, p, put, out)
    else:
        _gdn(w, p, put, out)
    if "num_experts" in c:
        e, ir, shared = c["num_experts"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
        m = p + "moe/"
        put("mlp.gate.weight", m + "router", (e, h))
        put("mlp.shared_expert_gate.weight", m + "shared_score", (1, h))
        gate_up = torch.empty(e, 2 * ir, h, dtype=F64)
        down = torch.empty(e, h, ir, dtype=F64)
        for x in range(e):
            gate_up[x, :ir] = w.logical(m + f"experts/{x}/gate", (ir, h))
            gate_up[x, ir:] = w.logical(m + f"experts/{x}/up", (ir, h))
            down[x] = w.logical(m + f"experts/{x}/down", (h, ir))
        out["mlp.experts.gate_up_proj"] = (gate_up, torch.bfloat16)
        out["mlp.experts.down_proj"] = (down, torch.bfloat16)
        for role in ("gate", "up", "down"):
            shape = (h, shared) if role == "down" else (shared, h)
            put(f"mlp.shared_expert.{role}_proj.weight", m + f"shared/{role}", shape)
    else:
        ii = c["intermediate_size"]
        put("mlp.gate_proj.weight", p + "mlp/gate", (ii, h))
        put("mlp.up_proj.weight", p + "mlp/up", (ii, h))
        put("mlp.down_proj.weight", p + "mlp/down", (h, ii))
    return out
