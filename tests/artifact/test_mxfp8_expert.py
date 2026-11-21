"""``mxfp8_expert_rg16_v1`` (tools/artifact/codecs/mxfp8_expert.py): exact round trip, the m16n8k32 A-fragment order
of a unit, the record geometry the C++ reader shares (tests/artifact/test_reader.cpp) and the refused NaN words."""

from __future__ import annotations

import pytest
import torch

from tools.artifact.codecs.mxfp8_expert import decode_mxfp8_expert_bank_words, encode_mxfp8_expert_records
from tools.artifact.layouts import encoded_size, mxfp8_expert_geometry

E, H, I = 3, 64, 96


def _words(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def codes(*shape):
        return torch.randint(0, 0x7F, shape, generator=g, dtype=torch.uint8) | \
            (torch.randint(0, 2, shape, generator=g, dtype=torch.uint8) << 7)

    def scales(*shape):
        return torch.randint(3, 247, shape, generator=g, dtype=torch.uint8)

    return {
        "gate_codes": codes(E, I, H), "gate_scales": scales(E, I, H // 32),
        "up_codes": codes(E, I, H), "up_scales": scales(E, I, H // 32),
        "down_codes": codes(E, H, I), "down_scales": scales(E, H, I // 32),
    }


def test_round_trip_is_exact():
    words = _words()
    payload = encode_mxfp8_expert_records(*words.values(), (E, H, I))
    assert len(payload) == encoded_size("mxfp8_expert_rg16_v1", "fp8_e4m3fn_g32_e8m0", (E, H, I))
    decoded = decode_mxfp8_expert_bank_words(payload, (E, H, I))
    assert all(torch.equal(words[key], decoded[key]) for key in words)


def test_unit_is_the_k32_a_fragment():
    words = _words(1)
    payload = torch.frombuffer(bytearray(encode_mxfp8_expert_records(*words.values(), (E, H, I))), dtype=torch.uint8)
    g = mxfp8_expert_geometry("fp8_e4m3fn_g32_e8m0", (E, H, I))
    rows = torch.stack((words["gate_codes"], words["up_codes"]), dim=2).reshape(E, 2 * I, H)  # 2i gate, 2i+1 up
    row_scales = torch.stack((words["gate_scales"], words["up_scales"]), dim=2).reshape(E, 2 * I, H // 32)
    expert, group, block = 2, 3, 1
    base = expert * g.record_stride + (group * g.gate_up_blocks + block) * 528
    tile = rows[expert, 16 * group:16 * group + 16, 32 * block:32 * block + 32]
    for lane in range(32):
        r, t = lane // 4, lane % 4
        expected = torch.cat((tile[r, 4 * t:4 * t + 4], tile[r + 8, 4 * t:4 * t + 4],
                              tile[r, 16 + 4 * t:20 + 4 * t], tile[r + 8, 16 + 4 * t:20 + 4 * t]))
        assert torch.equal(payload[base + 16 * lane:base + 16 * lane + 16], expected)
    assert torch.equal(payload[base + 512:base + 528], row_scales[expert, 16 * group:16 * group + 16, block])
    # The down matrix starts at gate_up_bytes; the record's padding is zero.
    down = payload[expert * g.record_stride + g.gate_up_bytes:][:16]
    assert torch.equal(down[:4], words["down_codes"][expert, 0, :4])
    assert not payload[expert * g.record_stride + g.record_bytes:(expert + 1) * g.record_stride].any()


def test_inkling_geometry_matches_the_reader():
    g = mxfp8_expert_geometry("fp8_e4m3fn_g32_e8m0", (128, 2304, 1152))
    assert (g.record_bytes, g.record_stride, g.gate_up_bytes, g.payload_bytes) == \
        (8211456, 8212480, 5474304, 128 * 8212480)


@pytest.mark.parametrize("key,value,message", [("gate_codes", 0x7F, "finite"), ("down_codes", 0xFF, "finite"),
                                               ("up_scales", 0xFF, "UE8M0"), ("down_scales", 2, "UE8M0")])
def test_words_outside_the_format_are_refused(key, value, message):
    words = _words(2)
    words[key].view(-1)[5] = value
    with pytest.raises(ValueError, match=message):
        encode_mxfp8_expert_records(*words.values(), (E, H, I))
