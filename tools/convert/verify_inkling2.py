"""Word-equality qualification of an Inkling 2 Mini artifact (recipe ``inkling2_mini``; design
docs/maintainer/inkling2_mini-design.md §18), independent of the converter: its own safetensors reader and its own
inverse of every mapping.

    python -m tools.convert.verify_inkling2 --model CHECKPOINT --artifact OUT.infernix [--report FILE] [--token-rank F]

Every bound parameter must equal its checkpoint source:

- BF16 and FP32 tensors bit for bit; the embedding, output head and Markov embedding as the first
  ``unpadded_vocab_size`` rows; the static convs as ``weight[:, 0, ::-1].T`` (lag-major); the cross-pass queries as
  (query, key, value, mlp) rows of loop layer l = (queries[2l], kv[2l], kv[2l+1], queries[2l+1]);
- NVFP4 banks: gate codes and scales = ``w13`` rows 0::2, up = rows 1::2, down = ``w2``, decoded from
  ``nvfp4_expert_rg16_v1``; the multiplier plane = ``fl32(amax / denominator)`` (one FP32 division); ``expert_amax``
  and ``expert_denominators`` the checkpoint's words, every amax and denominator finite and positive;
- MXFP8 banks: codes and UE8M0 scales decoded from ``mxfp8_expert_rg16_v1``;
- the DSpark Fold and fold_bias recomputed as ``refresh_fold`` (FP64 head rows times ``hidden_out``, copied into BF16);
- the vision tower's tensors bit for bit, its class row as the BF16 sum ``class_embedding + cls_position_embedding``,
  and ``processor_config.json`` byte for byte.

Every checkpoint tensor must be consumed, except ``hidden_out`` (folded) and, in an artifact without the vision
component, the vision tower.
"""

from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path
import re
import struct
import time

import numpy as np
import torch

from tools.artifact.codecs.mxfp8_expert import decode_mxfp8_expert_bank_words
from tools.artifact.codecs.nvfp4_expert import decode_nvfp4_expert_bank_words
from tools.artifact.reader import Artifact

_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "U8": torch.uint8, "F8_E4M3": torch.uint8}
_WIDTH = {"bf16": 2, "fp32": 4}


class _Checkpoint:
    """A safetensors file read through memory mapping; ``get`` returns raw words (F8_E4M3 as bytes)."""

    def __init__(self, path: Path):
        with open(path, "rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            self.header = json.loads(handle.read(length))
        self.header.pop("__metadata__", None)
        self.base = 8 + length
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.used: set[str] = set()

    def get(self, name: str) -> torch.Tensor:
        info = self.header[name]
        begin, end = info["data_offsets"]
        self.used.add(name)
        raw = torch.from_numpy(np.array(self.data[self.base + begin:self.base + end]))
        return raw.view(_DTYPES[info["dtype"]]).reshape(info["shape"])


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def _bound(artifact: Artifact, name: str) -> bytes:
    binding = artifact.directory.bindings[name]
    if "object" in binding:
        return artifact.read_object(binding["object"])
    chunks = []
    for part in binding["parts"]:
        obj = artifact.object(part["object"])
        width = _WIDTH[obj.format]
        first, last = part["range"]
        chunks.append(artifact.read_range(obj.offset + first * width, (last - first) * width))
    return b"".join(chunks)


def _object_format(artifact: Artifact, name: str) -> str:
    binding = artifact.directory.bindings[name]
    return artifact.object(binding["object"] if "object" in binding else binding["parts"][0]["object"]).format


def _lag_major(weight: torch.Tensor) -> torch.Tensor:
    return weight[:, 0, :].flip(1).t()


def _expected(name: str, ck: _Checkpoint, mtp: _Checkpoint, config: dict) -> torch.Tensor | None:
    """The checkpoint words a direct parameter must equal (None: not a direct parameter)."""

    vocab, p = config["vocab_size"], "model.llm."
    fixed = {
        "text/token_embedding": lambda: ck.get(p + "embed.weight")[:vocab],
        "text/output_head": lambda: ck.get(p + "unembed.weight")[:vocab],
        "text/embedding_norm": lambda: ck.get(p + "embed_norm.weight"),
        "text/final_norm": lambda: ck.get(p + "norm.weight"),
        "text/output_query": lambda: ck.get(p + "attn_res.output_query"),
        "dspark/feature_weights": lambda: mtp.get("fuse_weight"),
        "dspark/mask_embedding": lambda: mtp.get("mask_embed"),
        "dspark/final_norm": lambda: mtp.get("norm.weight"),
        "dspark/markov/context_norm": lambda: mtp.get("markov_head.dflare_norm.weight"),
        "dspark/markov/embedding": lambda: mtp.get("markov_head.markov_w1.weight")[:vocab],
        "dspark/markov/projection": lambda: mtp.get("markov_head.rnn_proj.weight"),
        "dspark/markov/projection_bias": lambda: mtp.get("markov_head.rnn_proj.bias"),
    }
    if name in fixed:
        return fixed[name]()
    match = re.fullmatch(r"text/layers/(\d+)/(.+)", name)
    if match:
        index, role = int(match.group(1)), match.group(2)
        sp, sa = f"{p}layers.{index}.", f"{p}layers.{index}.attn."
        full = config["layer_types"][index] == "full_attention"
        simple = {
            "attention_mix_query": f"{p}attn_res.attn_queries.{index - 1}", "mlp_mix_query": f"{p}attn_res.mlp_queries.{index}",
            "attention_norm": sp + "attn_norm.weight", "query_norm": sp + "q_input_norm.weight",
            "key_norm": sp + "k_input_norm.weight", "value_norm": sp + "v_input_norm.weight",
            "mlp_norm": sp + "mlp_norm.weight", "attention/query": sa + "wq_du.weight",
            "attention/relative": sa + "wr_du.weight", "attention/gate": sa + "gate_attn.weight",
            "attention/key": sa + "wk_dv.weight", "attention/value": sa + "wv_dv.weight",
            "attention/sink": sa + "attn_sink_proj.weight", "attention/key_taps": sa + "k_sconv.weight_proj",
            "attention/value_taps": sa + "v_sconv.weight_proj", "attention/output": sa + "wo_ud.weight",
            "attention/query_scale": sa + "q_l2_clip.scale", "attention/relative_scale": sa + "rel_l2_clip.scale",
            "attention/relative_table": sa + "rel_logits_proj.proj", "moe/router": sp + "mlp.gate.weight",
            "moe/router_bias": sp + "mlp.gate.bias",
        }
        if role in simple:
            return ck.get(simple[role])
        if role in ("attention/key_conv", "attention/value_conv"):
            leaf = sa + ("k_sconv" if role.endswith("key_conv") else "v_sconv")
            return ck.get(leaf + ".static_w") if full else _lag_major(ck.get(leaf + ".weight"))
        if role == "attention_conv":
            return _lag_major(ck.get(sp + "attn_sconv.weight"))
        if role == "mlp_conv":
            return _lag_major(ck.get(sp + "mlp_sconv.weight"))
        if role == "message_queries":
            loop = index - config["loop"]["core_start"]
            q = ck.get(p + "middle_cycle_cross_pass_queries")
            kv = ck.get(p + "middle_cycle_cross_pass_kv_queries")
            return torch.stack((q[2 * loop], kv[2 * loop], kv[2 * loop + 1], q[2 * loop + 1]))
        message = re.fullmatch(r"messages/(query|key|value|mlp)/(down|forget|gain)", role)
        if message:
            kind = {"query": "attn", "key": "k", "value": "v", "mlp": "mlp"}[message.group(1)]
            return ck.get(f"{sp}controllers.{kind}.{message.group(2)}_weight")
        x = sp + "mlp.experts."
        if role == "moe/expert_amax":
            return torch.stack((ck.get(x + "w13_weight_amax"), ck.get(x + "w2_weight_amax")), dim=1)
        if role == "moe/expert_denominators":
            return torch.cat((ck.get(x + "w13_weight_outer_scale_denominator"),
                              ck.get(x + "w2_weight_outer_scale_denominator")))
        return None
    v = "model.visual."
    vision = {
        "vision/patch_embedding": v + "backbone.embeddings.patch_embedding.weight",
        "vision/position_embedding": v + "backbone.embeddings.position_embedding",
        "vision/pre_norm": v + "backbone.ln_pre.weight", "vision/pre_norm_bias": v + "backbone.ln_pre.bias",
        "vision/projector/in": v + "proj_in.weight", "vision/projector/out": v + "proj_out.weight",
    }
    if name in vision:
        return ck.get(vision[name])
    match = re.fullmatch(r"vision/layers/(\d+)/(.+)", name)
    if match:
        sp = f"{v}backbone.layers.{match.group(1)}."
        leaf = {
            "attention_norm": "ln_1.weight", "attention_norm_bias": "ln_1.bias",
            "attention/query": "self_attn.q_proj.weight", "attention/query_bias": "self_attn.q_proj.bias",
            "attention/key": "self_attn.k_proj.weight", "attention/key_bias": "self_attn.k_proj.bias",
            "attention/value": "self_attn.v_proj.weight", "attention/value_bias": "self_attn.v_proj.bias",
            "attention/output": "self_attn.out_proj.weight", "attention/output_bias": "self_attn.out_proj.bias",
            "attention_scale": "ls_1.gamma", "mlp_norm": "ln_2.weight", "mlp_norm_bias": "ln_2.bias",
            "mlp/up": "mlp.c_fc.weight", "mlp/up_bias": "mlp.c_fc.bias", "mlp/down": "mlp.c_proj.weight",
            "mlp/down_bias": "mlp.c_proj.bias", "mlp_scale": "ls_2.gamma",
        }.get(match.group(2))
        return None if leaf is None else ck.get(sp + leaf)
    match = re.fullmatch(r"dspark/layers/(\d+)/(.+)", name)
    if match:
        sp, role = f"layers.{match.group(1)}.", match.group(2)
        leaf = {
            "context_norm": "dflare_norm.weight", "input_norm": "input_layernorm.weight",
            "post_attention_norm": "post_attention_layernorm.weight", "attention/query": "self_attn.q_proj.weight",
            "attention/key": "self_attn.k_proj.weight", "attention/value": "self_attn.v_proj.weight",
            "attention/context_key": "self_attn.k_proj_ctx.weight", "attention/context_value": "self_attn.v_proj_ctx.weight",
            "attention/query_norm": "self_attn.q_norm.weight", "attention/key_norm": "self_attn.k_norm.weight",
            "attention/context_key_norm": "self_attn.k_norm_ctx.weight", "attention/output": "self_attn.o_proj.weight",
            "mlp/gate": "mlp.gate_proj.weight", "mlp/up": "mlp.up_proj.weight", "mlp/down": "mlp.down_proj.weight",
        }.get(role)
        return None if leaf is None else mtp.get(sp + leaf)
    return None


def _fold(ck: _Checkpoint, mtp: _Checkpoint, vocab: int, *, bias: bool) -> torch.Tensor:
    projection = mtp.get("markov_head.hidden_out.bias" if bias else "markov_head.hidden_out.weight").double()
    head = ck.get("model.llm.unembed.weight")
    out = torch.empty((vocab,) if bias else (vocab, projection.shape[1]), dtype=torch.bfloat16)
    for first in range(0, vocab, 4096):
        last = min(vocab, first + 4096)
        out[first:last].copy_(head[first:last].double() @ projection)
    return out


def _verify_nvfp4(artifact: Artifact, name: str, ck: _Checkpoint, x: str, shape) -> None:
    words = decode_nvfp4_expert_bank_words(_bound(artifact, name), shape)
    w13, s13 = ck.get(x + "w13_weight"), ck.get(x + "w13_weight_scale")
    pairs = (("gate_codes", w13[:, 0::2]), ("gate_scales", s13[:, 0::2]), ("up_codes", w13[:, 1::2]),
             ("up_scales", s13[:, 1::2]), ("down_codes", ck.get(x + "w2_weight")),
             ("down_scales", ck.get(x + "w2_weight_scale")))
    for key, expected in pairs:
        if not torch.equal(words[key], expected):
            raise AssertionError(f"{name}: {key} differ from the checkpoint")
    amax13, amax2 = ck.get(x + "w13_weight_amax"), ck.get(x + "w2_weight_amax")
    den13 = ck.get(x + "w13_weight_outer_scale_denominator").reshape(())
    den2 = ck.get(x + "w2_weight_outer_scale_denominator").reshape(())
    for label, values in (("amax", torch.cat((amax13, amax2))), ("denominators", torch.stack((den13, den2)))):
        if not bool(torch.isfinite(values).all() and (values > 0).all()):
            raise AssertionError(f"{name}: a nonpositive or nonfinite {label}")
    expected = torch.stack((amax13 / den13, amax13 / den13, amax2 / den2), dim=1)
    if not torch.equal(words["multipliers"], expected):
        raise AssertionError(f"{name}: the multiplier plane is not fl32(amax / denominator)")


def _verify_mxfp8(artifact: Artifact, name: str, ck: _Checkpoint, x: str, shape) -> None:
    words = decode_mxfp8_expert_bank_words(_bound(artifact, name), shape)
    w13, s13 = ck.get(x + "w13_weight"), ck.get(x + "w13_weight_scale")
    pairs = (("gate_codes", w13[:, 0::2]), ("gate_scales", s13[:, 0::2]), ("up_codes", w13[:, 1::2]),
             ("up_scales", s13[:, 1::2]), ("down_codes", ck.get(x + "w2_weight")),
             ("down_scales", ck.get(x + "w2_weight_scale")))
    for key, expected in pairs:
        if not torch.equal(words[key], expected):
            raise AssertionError(f"{name}: {key} differ from the checkpoint")


def _verify_rank(stored: bytes, model: Path, vocab: int, token_rank: Path | None) -> None:
    """The stored rank order: a permutation of the vocabulary, the tokenizer's special tokens first, and (given the
    file it was made from) that file's words."""
    if len(stored) != 4 * vocab:
        raise AssertionError("proposal/token_rank: not one int32 per token")
    order = [int.from_bytes(stored[i : i + 4], "little", signed=True) for i in range(0, len(stored), 4)]
    if sorted(order) != list(range(vocab)):
        raise AssertionError("proposal/token_rank: not a permutation of the vocabulary")
    added = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"]
    special = {int(token["id"]) for token in added if token.get("special")}
    if set(order[: len(special)]) != special:
        raise AssertionError("proposal/token_rank: the special tokens do not come first")
    if token_rank is not None and token_rank.read_bytes() != stored:
        raise AssertionError("proposal/token_rank: differs from the rank file")


def verify(model: Path, artifact_path: Path, token_rank: Path | None = None) -> dict:
    began = time.time()
    index = model / "model.safetensors.index.json"
    if index.exists():
        files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if len(files) != 1:
            raise ValueError("the verifier reads a single-file Inkling checkpoint")
        ck = _Checkpoint(model / files.pop())
    else:
        ck = _Checkpoint(model / "model.safetensors")
    mtp = _Checkpoint(model / "mtp.safetensors")
    counts = {"direct": 0, "nvfp4_banks": 0, "mxfp8_banks": 0, "derived": 0}
    with Artifact(artifact_path) as artifact:
        config = artifact.directory.components["text"]["config"]
        vocab = config["vocab_size"]
        for name in sorted(artifact.directory.bindings):
            match = re.fullmatch(r"text/layers/(\d+)/moe/experts", name)
            if match:
                index = int(match.group(1))
                x = f"model.llm.layers.{index}.mlp.experts."
                obj_format = _object_format(artifact, name)
                shape = (config["num_experts"], config["hidden_size"], config["moe_intermediate_size"])
                if config["expert_formats"][index] == "nvfp4_four_over_six":
                    if obj_format != "nvfp4_mul":
                        raise AssertionError(f"{name}: stored as {obj_format}, its config says NVFP4")
                    _verify_nvfp4(artifact, name, ck, x, shape)
                    counts["nvfp4_banks"] += 1
                else:
                    if obj_format != "fp8_e4m3fn_g32_e8m0":
                        raise AssertionError(f"{name}: stored as {obj_format}, its config says MXFP8")
                    _verify_mxfp8(artifact, name, ck, x, shape)
                    counts["mxfp8_banks"] += 1
                continue
            if name == "proposal/token_rank":
                _verify_rank(_bound(artifact, name), model, vocab, token_rank)
                counts["token_rank"] = vocab
                continue
            if name in ("dspark/markov/fold", "dspark/markov/fold_bias"):
                expected = _fold(ck, mtp, vocab, bias=name.endswith("bias"))
                counts["derived"] += 1
            elif name == "vision/class_row":
                e = "model.visual.backbone.embeddings."
                expected = ck.get(e + "class_embedding") + ck.get(e + "cls_position_embedding")
                counts["derived"] += 1
            else:
                expected = _expected(name, ck, mtp, config)
                if expected is None:
                    raise AssertionError(f"{name}: no checkpoint source is known")
                counts["direct"] += 1
            if _bound(artifact, name) != _bytes(expected):
                raise AssertionError(f"{name}: the artifact's words differ from the checkpoint's")
        with_vision = "vision" in artifact.directory.components
        if with_vision:
            refs = artifact.directory.components["vision"]["resources"]
            if artifact.read_object(refs["processor_config.json"]) != (model / "processor_config.json").read_bytes():
                raise AssertionError("processor_config.json differs from the checkpoint's")
        with_dspark = "dspark" in artifact.directory.components
    mtp.used.add("markov_head.hidden_out.weight")
    mtp.used.add("markov_head.hidden_out.bias")
    unused = sorted(n for n in ck.header
                    if n not in ck.used and (with_vision or not n.startswith("model.visual.")))
    # The drafter's file is consumed only by an artifact with the dspark component (--components text has none).
    if with_dspark:
        unused += sorted("mtp:" + n for n in mtp.header if n not in mtp.used)
    if unused:
        raise AssertionError(f"checkpoint tensors not in the artifact: {unused[:8]} ({len(unused)})")
    unconverted = {} if with_vision else {"unconverted": "model.visual.* (no vision component)"}
    if not with_dspark:
        unconverted["unconverted_drafter"] = "mtp.safetensors (no dspark component)"
    return {**counts, **unconverted, "seconds": round(time.time() - began, 1)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--token-rank", type=Path, help="the rank file proposal/token_rank was made from")
    args = parser.parse_args(argv)
    result = verify(args.model, args.artifact, args.token_rank)
    print(json.dumps(result, indent=1))
    if args.report:
        args.report.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
