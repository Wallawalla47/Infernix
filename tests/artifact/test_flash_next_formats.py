"""nvfp4_mul expert banks (nvfp4_expert_rg16_v1)."""

from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.codecs.nvfp4_expert import (
    decode_nvfp4_expert_bank_words,
    dequantize_nvfp4_mul,
    encode_nvfp4_expert_bank,
    expert_bank_record_offset,
)
from tools.artifact.formats import decode_e2m1_word, decode_e4m3fn_word
from tools.artifact.layouts import encoded_size, expert_bank_geometry


def _random_bank(e: int, h: int, i: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    def codes(rows, k):
        return torch.randint(0, 256, (e, rows, k // 2), generator=g, dtype=torch.uint8)

    def scales(rows, k):
        return torch.randint(0, 0x7F, (e, rows, k // 16), generator=g, dtype=torch.uint8)

    mult = torch.rand((e, 3), generator=g, dtype=torch.float32) * 1e-3 + 1e-5
    return dict(
        gate_codes=codes(i, h), gate_scales=scales(i, h),
        up_codes=codes(i, h), up_scales=scales(i, h),
        down_codes=codes(h, i), down_scales=scales(h, i),
        multipliers=mult,
    )


def test_geometry_of_the_flash_next_bank():
    g = expert_bank_geometry("nvfp4_mul", (512, 2560, 640))
    assert g.record_bytes == g.record_stride == 2_764_800 == 675 * 4096
    assert g.gate_up_bytes == 1_843_200
    assert g.multiplier_offset == 512 * 2_764_800
    assert encoded_size("nvfp4_expert_rg16_v1", "nvfp4_mul", (512, 2560, 640)) == g.payload_bytes
    with pytest.raises(ValueError):
        encoded_size("nvfp4_expert_rg16_v1", "nvfp4", (1, 2560, 640))
    with pytest.raises(ValueError):
        expert_bank_geometry("nvfp4_mul", (1, 40, 640))


def _code(packed: torch.Tensor, row: int, k: int) -> int:
    byte = int(packed[row, k // 2])
    return byte & 0xF if k % 2 == 0 else byte >> 4


def test_bank_layout_places_every_nibble_and_scale_as_documented():
    e, h, i = 2, 32, 16
    words = _random_bank(e, h, i)
    payload = encode_nvfp4_expert_bank(**words, shape=(e, h, i))
    geo = expert_bank_geometry("nvfp4_mul", (e, h, i))
    assert geo.record_bytes == 864 and geo.record_stride == 4096
    for expert in range(e):
        base = expert_bank_record_offset(geo, expert)
        assert payload[base + geo.record_bytes : base + geo.record_stride] == bytes(geo.record_stride - geo.record_bytes)

        def check(matrix_offset, rows, k, code_of, scale_of):
            blocks = k // 16
            for r in range(rows):
                rg, ri = divmod(r, 16)
                for kk in range(k):
                    b, kb = divmod(kk, 16)
                    unit = base + matrix_offset + (rg * blocks + b) * 144
                    byte = payload[unit + 32 * (kb // 4) + 4 * (ri % 8) + kb % 4]
                    got = byte & 0xF if ri < 8 else byte >> 4
                    assert got == code_of(r, kk)
                for b in range(blocks):
                    assert payload[base + matrix_offset + (rg * blocks + b) * 144 + 128 + ri] == scale_of(r, b)

        # Gate/up rows interleave: row 2j is gate_j, row 2j+1 is up_j.
        def gu_code(r, kk):
            src = words["gate_codes"] if r % 2 == 0 else words["up_codes"]
            return _code(src[expert], r // 2, kk)

        def gu_scale(r, b):
            src = words["gate_scales"] if r % 2 == 0 else words["up_scales"]
            return int(src[expert, r // 2, b])

        check(0, 2 * i, h, gu_code, gu_scale)
        check(geo.gate_up_bytes, h, i, lambda r, kk: _code(words["down_codes"][expert], r, kk),
              lambda r, b: int(words["down_scales"][expert, r, b]))
    mult = payload[geo.multiplier_offset :]
    assert mult == words["multipliers"].numpy().tobytes()


def test_bank_round_trip_is_exact_and_multipliers_are_not_inverted():
    e, h, i = 3, 64, 32
    words = _random_bank(e, h, i, seed=4)
    # A multiplier whose reciprocal is not representable: storing 1/m would change it.
    words["multipliers"][0, 0] = 1.0 / 3.0
    payload = encode_nvfp4_expert_bank(**words, shape=(e, h, i))
    decoded = decode_nvfp4_expert_bank_words(payload, (e, h, i))
    for key, value in words.items():
        assert torch.equal(decoded[key].view(torch.uint8) if key == "multipliers" else decoded[key],
                           value.view(torch.uint8) if key == "multipliers" else value), key
    word = struct.unpack("<I", payload[expert_bank_geometry("nvfp4_mul", (e, h, i)).multiplier_offset:][:4])[0]
    assert word == struct.unpack("<I", struct.pack("<f", 1.0 / 3.0))[0]


def test_nvfp4_mul_dequantization_matches_the_scalar_definition():
    words = _random_bank(1, 32, 16, seed=8)
    m = float(words["multipliers"][0, 2])
    w = dequantize_nvfp4_mul(words["down_codes"][0], words["down_scales"][0], m)
    for r in range(32):
        for k in range(16):
            expect = decode_e2m1_word(_code(words["down_codes"][0], r, k)) * decode_e4m3fn_word(
                int(words["down_scales"][0, r, k // 16])) * m
            assert float(w[r, k]) == expect


def test_bank_rejects_invalid_words():
    words = _random_bank(1, 32, 16)
    bad = dict(words)
    bad["gate_scales"] = words["gate_scales"].clone()
    bad["gate_scales"][0, 0, 0] = 0x7F
    with pytest.raises(ValueError):
        encode_nvfp4_expert_bank(**bad, shape=(1, 32, 16))
    bad = dict(words)
    bad["multipliers"] = words["multipliers"].clone()
    bad["multipliers"][0, 1] = 0.0
    with pytest.raises(ValueError):
        encode_nvfp4_expert_bank(**bad, shape=(1, 32, 16))
    with pytest.raises(ValueError):
        decode_nvfp4_expert_bank_words(b"\0" * 7, (1, 32, 16))
