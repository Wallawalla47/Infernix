"""Exact reading of ModelOpt NVFP4 experts and block-FP8 matrices into Flash-Next banks."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.nvfp4_expert import decode_nvfp4_expert_bank_words
from tools.convert.sources.modelopt import fp8_block_matrix_words, nvfp4_expert_bank
from tools.convert.sources.safetensors import SafetensorsSource


def _modelopt_experts(tmp_path, experts, hidden, inter, *, split_input=False):
    g = torch.Generator().manual_seed(3)
    tensors, expect = {}, []
    prefix = "model.language_model.layers.0.mlp.experts."
    for e in range(experts):
        entry = {}
        for role, (n, k) in (("gate", (inter, hidden)), ("up", (inter, hidden)), ("down", (hidden, inter))):
            leaf = f"{prefix}{e}.{role}_proj"
            codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
            scales = torch.randint(0, 0x7F, (n, k // 16), generator=g, dtype=torch.uint8)
            ws2 = torch.tensor([1.0 / (3 + e)], dtype=torch.float32)  # not a power of two
            inp = torch.tensor([0.01 * (1 + e) * (2 if split_input and role == "up" else 1)], dtype=torch.float32)
            tensors[leaf + ".weight"] = codes
            tensors[leaf + ".weight_scale"] = scales.view(torch.float8_e4m3fn)
            tensors[leaf + ".weight_scale_2"] = ws2
            tensors[leaf + ".input_scale"] = inp
            entry[role] = (codes, scales, ws2, inp)
        expect.append(entry)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return prefix, expect


@pytest.mark.parametrize("split_input", [False, True])
def test_expert_bank_keeps_every_modelopt_word(tmp_path, split_input):
    e, h, i = 3, 64, 32
    prefix, expect = _modelopt_experts(tmp_path, e, h, i, split_input=split_input)
    with SafetensorsSource(tmp_path) as store:
        bank = nvfp4_expert_bank(store, prefix, e, h, i)
    words = decode_nvfp4_expert_bank_words(bank.payload, (e, h, i))
    assert bank.shared_gate_up_input is (not split_input)
    for x in range(e):
        for index, role in enumerate(("gate", "up", "down")):
            codes, scales, ws2, inp = expect[x][role]
            assert torch.equal(words[role + "_codes"][x], codes)
            assert torch.equal(words[role + "_scales"][x], scales)
            # The multiplier and the input scale are stored as the checkpoint's own FP32 words.
            assert words["multipliers"][x, index].view(torch.int32) == ws2.view(torch.int32)[0]
            assert bank.input_scales[x, index].view(torch.int32) == inp.view(torch.int32)[0]


def test_expert_bank_rejects_a_missing_or_reshaped_tensor(tmp_path):
    prefix, _ = _modelopt_experts(tmp_path, 1, 64, 32)
    with SafetensorsSource(tmp_path) as store:
        with pytest.raises(ValueError):
            nvfp4_expert_bank(store, prefix, 1, 64, 48)


def test_fp8_block_matrix_words(tmp_path):
    codes = torch.randint(0, 0x7E, (200, 130), dtype=torch.uint8)
    scales = torch.rand((2, 2), dtype=torch.float32)
    save_file({"mtp.layers.0.mlp.experts.0.gate_proj.weight": codes.view(torch.float8_e4m3fn),
               "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": scales},
              str(tmp_path / "model.safetensors"))
    with SafetensorsSource(tmp_path) as store:
        c, s = fp8_block_matrix_words(store, "mtp.layers.0.mlp.experts.0.gate_proj", (200, 130))
    assert torch.equal(c, codes) and torch.equal(s, scales)
