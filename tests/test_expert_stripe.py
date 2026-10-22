"""Expert stripes (tools/flash_next/expert_stripe.py): the tails of every routed expert record,
written for a second drive. Against the artifact's own bytes: the header the C++ reader parses
(src/runtime/expert_offload/expert_store.h), every tail, records that straddle part files, the split
fraction rounded to 4 KiB, verification of a damaged stripe, and refusal of another artifact."""

from __future__ import annotations

import random
import struct

import pytest

from tools.artifact.layouts import fp8_expert_geometry
from tools.artifact.schema import ArtifactError, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.flash_next.expert_stripe import read_header, verify, write

SHAPE = (3, 256, 128)  # experts, hidden, intermediate
FORMAT, LAYOUT = "fp8_e4m3fn_b128", "fp8_expert_b128_v1"


def _artifact(path, layers=2, seed=0):
    geometry = fp8_expert_geometry(FORMAT, SHAPE)
    rng = random.Random(seed)
    # As the converter writes them: objects with the writer's own ids, bound to the bank parameters
    # (bound in reverse order, so the stripe follows the bindings, not the object order).
    payloads = {f"weight/{l:06d}": rng.randbytes(geometry.payload_bytes) for l in range(layers)}
    specs = [TensorSpec(name, SHAPE, FORMAT, LAYOUT) for name in payloads]
    bindings = {f"text/layers/{l}/moe/experts": {"object": f"weight/{layers - 1 - l:06d}"} for l in range(layers)}
    with ArtifactWriter(
        path, specs, components={"text": {"config": {}, "resources": {}}}, bindings=bindings,
        max_file_bytes=5 * geometry.record_stride // 2,  # records straddle part files
    ) as writer:
        for name, data in payloads.items():
            writer.write_object(name, data)
        identity = writer.artifact_id
    by_layer = [payloads[bindings[f"text/layers/{l}/moe/experts"]["object"]] for l in range(layers)]
    return geometry.record_stride, by_layer, identity


def _records(stride, by_layer):
    return [data[e * stride:(e + 1) * stride] for data in by_layer for e in range(SHAPE[0])]


@pytest.mark.parametrize("fraction", [0.5, 0.3])
def test_stripe_holds_every_tail_after_the_header(tmp_path, fraction):
    stride, payloads, identity = _artifact(tmp_path / "m.infernix")
    out = tmp_path / "m.stripe"
    result = write(tmp_path / "m.infernix", out, fraction)
    assert not (tmp_path / "m.stripe.partial").exists()
    artifact_id, banks = read_header(out)
    head = (int(stride * (1 - fraction)) // 4096) * 4096
    assert artifact_id == identity and len(banks) == 2
    assert all(b.stride == stride and b.head == head and b.tail_offset % 4096 == 0 for b in banks)
    raw = out.read_bytes()
    magic, version, layers, experts, size, _ = struct.unpack_from("<8sIIII16s", raw)
    assert (magic, version, layers, experts, size) == (b"INFXSTR1", 1, 2, SHAPE[0], banks[0].tail_offset)
    tails = b"".join(record[head:] for record in _records(stride, payloads))
    assert raw[size:] == tails and result["tail_bytes"] == len(tails)
    assert verify(tmp_path / "m.infernix", out)["mismatches"] == 0


def test_verify_counts_a_damaged_tail_and_refuses_another_artifact(tmp_path):
    _artifact(tmp_path / "a.infernix", seed=1)
    _artifact(tmp_path / "b.infernix", seed=2)
    out = tmp_path / "a.stripe"
    write(tmp_path / "a.infernix", out, 0.5)
    data = bytearray(out.read_bytes())
    data[-1] ^= 0xFF
    out.write_bytes(bytes(data))
    assert verify(tmp_path / "a.infernix", out)["mismatches"] == 1
    with pytest.raises(ArtifactError, match="belongs to artifact"):
        verify(tmp_path / "b.infernix", out)


def test_fraction_outside_the_record_is_refused(tmp_path):
    _artifact(tmp_path / "m.infernix")
    with pytest.raises(ArtifactError):
        write(tmp_path / "m.infernix", tmp_path / "m.stripe", 1.0)
