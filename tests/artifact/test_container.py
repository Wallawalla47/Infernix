from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from tools.artifact.reader import Artifact
from tools.artifact.framing import (
    EXPERT_STREAM_ALIGNMENT,
    HEADER,
    MAGIC,
    PART_MAGIC,
    PAYLOAD_ALIGNMENT,
)
from tools.artifact.layouts import align_up, expert_bank_geometry
from tools.artifact.schema import (
    ArtifactError,
    ResourceSpec,
    TensorSpec,
    encode_directory,
    parse_directory,
)
from tools.artifact.writer import ArtifactWriter, layout_directory


def _small(path: Path) -> dict:
    specs = [
        TensorSpec("w", (2, 2), "bf16", "contiguous_le_v1"),
        TensorSpec("scale", (), "fp32", "contiguous_le_v1"),
        ResourceSpec("template", 5),
    ]
    components = {
        "text": {
            "config": {"architectures": ["Qwen3_5ForCausalLM"], "hidden_size": 2},
            "resources": {"chat_template.jinja": "template"},
        }
    }
    bindings = {
        "text/weight": {"object": "w"},
        "text/reordered": {
            "parts": [
                {"object": "w", "range": [2, 4]},
                {"object": "w", "range": [0, 2]},
            ]
        },
    }
    uses = [
        {
            "parameter": "text/weight",
            "input": "text/input",
            "activation_policy": "AllowA4",
            "auxiliaries": {"activation_input_divisor": {"object": "scale"}},
        }
    ]
    with ArtifactWriter(
        path, specs, components=components, bindings=bindings, uses=uses
    ) as writer:
        writer.write_object("w", struct.pack("<4H", 0, 0x8000, 0x3F80, 0x7FC1))
        writer.write_object("scale", struct.pack("<f", 2.0))
        writer.write_object("template", b"hello")
        return writer.directory.to_json()


def _rewrite(path: Path, directory: dict) -> None:
    with path.open("r+b") as stream:
        magic, reserve, identity = HEADER.unpack(stream.read(HEADER.size))
        data = encode_directory(directory)
        assert len(data) <= reserve
        stream.seek(HEADER.size)
        stream.write(data + b" " * (reserve - len(data)))


def test_single_file_known_header_payload_and_refs(tmp_path):
    path = tmp_path / "test.ninfer"
    directory = _small(path)
    raw = path.read_bytes()
    magic, json_bytes, identity = HEADER.unpack(raw[:32])
    assert magic == b"NINFER\0\3" and len(identity) == 16
    payload_offset = (32 + json_bytes + 4095) // 4096 * 4096
    assert payload_offset == 4096
    assert raw[payload_offset : payload_offset + 8] == struct.pack(
        "<4H", 0, 0x8000, 0x3F80, 0x7FC1
    )
    assert raw[payload_offset + 8 : payload_offset + 256] == bytes(248)
    assert raw[payload_offset + 256 : payload_offset + 260] == struct.pack("<f", 2)
    assert raw[-5:] == b"hello"
    with Artifact(path) as reader:
        assert reader.directory.to_json() == directory
        assert reader.read_object("template") == b"hello"
        assert reader.read_range(reader.payload_bytes, 0) == b""
        with pytest.raises(ArtifactError, match="exceeds payload"):
            reader.read_range(reader.payload_bytes, 1)


def test_shards_use_recorded_names_and_open_only_when_needed(tmp_path):
    path = tmp_path / "large.ninfer"
    payload = bytes(range(251)) * 110
    specs = [ResourceSpec("data", len(payload))]
    with ArtifactWriter(
        path,
        specs,
        components={"text": {"config": {}, "resources": {"data": "data"}}},
        bindings={},
        max_file_bytes=12288,
    ) as writer:
        writer.write_object(
            "data", (payload[:5000], payload[5000:9000], payload[9000:])
        )
        directory = writer.directory.to_json()
        identity = writer.artifact_id
    assert [f["path"] for f in directory["files"]] == [
        None,
        "large.ninfer.part-0001",
        "large.ninfer.part-0002",
        "large.ninfer.part-0003",
    ]
    for index in range(1, 4):
        part = tmp_path / directory["files"][index]["path"]
        assert HEADER.unpack(part.read_bytes()[:32]) == (PART_MAGIC, index, identity)
        assert part.stat().st_size <= 12288
    renamed = tmp_path / "user-chosen-part"
    (tmp_path / directory["files"][1]["path"]).rename(renamed)
    directory["files"][1]["path"] = renamed.name
    _rewrite(path, directory)
    with Artifact(path) as reader:
        assert reader.read_object("data") == payload
        assert (
            b"".join(reader.iter_range(8120, 17300, chunk_bytes=503))
            == payload[8120:25420]
        )

    renamed.unlink()
    with Artifact(path) as reader:
        assert reader.read_range(0, 100) == payload[:100]
        with pytest.raises(FileNotFoundError):
            reader.read_range(8200, 10)


@pytest.mark.parametrize("failure", ["missing", "producer"])
def test_incomplete_or_failed_output_is_not_published(tmp_path, failure):
    path = tmp_path / "failed.ninfer"
    with pytest.raises((ArtifactError, RuntimeError)):
        with ArtifactWriter(
            path,
            [ResourceSpec("r", 20000)],
            components={"text": {"config": {}}},
            bindings={},
            max_file_bytes=12288,
        ) as writer:
            writer.write_region("r", 0, b"abc")
            if failure == "producer":
                raise RuntimeError("source failed")
    assert list(tmp_path.iterdir()) == []


def test_existing_output_is_preserved(tmp_path):
    path = tmp_path / "existing.ninfer"
    path.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        _small(path)
    assert path.read_bytes() == b"old"


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda d: d["objects"][1].update(offset=1), id="overlapping-objects"
        ),
        pytest.param(
            lambda d: d["bindings"]["text/reordered"]["parts"][0].update(range=[0, 5]),
            id="binding-outside-parent",
        ),
    ],
)
def test_directory_rejects_invalid_structure(tmp_path, change):
    directory = _small(tmp_path / "base.ninfer")
    change(directory)
    with pytest.raises(ArtifactError):
        parse_directory(directory)


def test_unknown_codec_is_deferred_until_object_is_consumed(tmp_path):
    path = tmp_path / "future.ninfer"
    directory = _small(path)
    directory["objects"][0]["format"] = "future_codec"
    _rewrite(path, directory)
    with Artifact(path) as reader:
        assert reader.read_object("template") == b"hello"
        with pytest.raises(ArtifactError, match="future_codec"):
            reader.read_object("w")


def test_default_file_limit_accounts_for_framing():
    size = 32_000_000_000
    description = {
        "components": {"text": {"config": {}}},
        "bindings": {},
        "uses": [],
        "objects": [
            {
                "id": "r",
                "kind": "resource",
                "encoding": "raw_bytes_v1",
                "offset": 0,
                "bytes": size,
            }
        ],
    }
    directory, _, start = layout_directory("large.ninfer", description, size)
    assert len(directory.files) == 2
    assert start + directory.files[0].payload_bytes <= size
    assert 4096 + directory.files[1].payload_bytes <= size
    description["objects"][0]["bytes"] = size - start
    directory, _, same_start = layout_directory(
        "small.ninfer", description, size - start
    )
    assert same_start == start and len(directory.files) == 1
    with pytest.raises(ArtifactError):
        layout_directory("small.ninfer", description, 1 << 64)


def _expert_specs():
    return [
        TensorSpec("bias", (130,), "bf16", "contiguous_le_v1"),
        ResourceSpec("template", 5),
        TensorSpec("experts", (16, 16, 16), "nvfp4_mul", "nvfp4_expert_rg16_v1"),
        TensorSpec("experts2", (16, 16, 16), "nvfp4_mul", "nvfp4_expert_rg16_v1"),
        TensorSpec("tail", (2,), "fp32", "contiguous_le_v1"),
    ]


def _write_expert_artifact(path: Path, *, max_file_bytes=None):
    assert EXPERT_STREAM_ALIGNMENT == 2 * 1024 * 1024
    bank = bytes(
        (offset * 131 + 7) % 253 for offset in range(EXPERT_STREAM_ALIGNMENT // 4)
    )
    geometry = expert_bank_geometry("nvfp4_mul", (16, 16, 16))
    bank = (bank * (geometry.payload_bytes // len(bank) + 1))[
        : geometry.payload_bytes
    ]
    banks = {"experts": bank, "experts2": bytes(b ^ 0x5A for b in bank)}
    options = {} if max_file_bytes is None else {"max_file_bytes": max_file_bytes}
    with ArtifactWriter(
        path,
        _expert_specs(),
        components={
            "text": {
                "config": {},
                "resources": {"chat_template.jinja": "template"},
            }
        },
        bindings={"bias": {"object": "bias"}},
        **options,
    ) as writer:
        writer.write_object("bias", bytes(260))
        writer.write_object("template", b"hello")
        writer.write_object("experts", banks["experts"])
        writer.write_object("experts2", banks["experts2"])
        writer.write_object("tail", struct.pack("<2f", 1.0, 2.0))
        return writer.directory.to_json(), writer.payload_offset, banks


def test_expert_stream_object_is_aligned_in_file_space(tmp_path):
    path = tmp_path / "expert.ninfer"
    directory, payload_offset, banks = _write_expert_artifact(path)
    assert payload_offset == EXPERT_STREAM_ALIGNMENT
    raw = path.read_bytes()
    magic, json_bytes, _ = HEADER.unpack(raw[: HEADER.size])
    assert magic == MAGIC
    # The padded directory region reaches exactly the aligned payload start.
    assert HEADER.size + json_bytes == payload_offset
    objects = {obj["id"]: obj for obj in directory["objects"]}
    assert objects["bias"]["offset"] == 0
    assert objects["template"]["offset"] == 260
    assert objects["experts"]["offset"] == 2 * 1024 * 1024
    assert objects["experts2"]["offset"] == 4 * 1024 * 1024
    # Objects after the streams keep their layout minimum: 256-byte, not 2 MiB, alignment.
    assert objects["tail"]["offset"] == 4 * 1024 * 1024 + align_up(
        len(banks["experts2"]), 256
    )
    for name in ("experts", "experts2"):
        start = payload_offset + objects[name]["offset"]
        assert start % EXPERT_STREAM_ALIGNMENT == 0
        assert raw[start : start + len(banks[name])] == banks[name]
    with Artifact(path) as reader:
        assert reader.read_object("experts") == banks["experts"]
        assert reader.read_object("experts2") == banks["experts2"]
        assert reader.read_object("template") == b"hello"


def test_expert_stream_alignment_holds_across_parts(tmp_path):
    path = tmp_path / "expert.ninfer"
    directory, payload_offset, banks = _write_expert_artifact(
        path, max_file_bytes=3 * 1024 * 1024
    )
    files = directory["files"]
    assert [file["path"] for file in files] == [
        None,
        "expert.ninfer.part-0001",
        "expert.ninfer.part-0002",
        "expert.ninfer.part-0003",
    ]
    # Segment capacities keep every following volume start 4 KiB past a 2 MiB boundary.
    assert files[0]["payload_bytes"] % EXPERT_STREAM_ALIGNMENT == PAYLOAD_ALIGNMENT
    assert files[1]["payload_bytes"] == EXPERT_STREAM_ALIGNMENT
    assert files[2]["payload_bytes"] == EXPERT_STREAM_ALIGNMENT
    objects = {obj["id"]: obj for obj in directory["objects"]}

    # First bank: starts at the first byte of part-0001's payload region, straddles the
    # volume end and resumes at part-0002's 4 KiB-aligned payload start.
    experts = objects["experts"]
    local = experts["offset"] - files[0]["payload_bytes"]
    segment_start = PAYLOAD_ALIGNMENT + local
    assert segment_start == EXPERT_STREAM_ALIGNMENT
    part = (tmp_path / files[1]["path"]).read_bytes()
    visible = files[1]["payload_bytes"] - local
    assert part[segment_start : segment_start + visible] == banks["experts"][:visible]
    part2 = (tmp_path / files[2]["path"]).read_bytes()
    assert part2[PAYLOAD_ALIGNMENT : PAYLOAD_ALIGNMENT + len(banks["experts"]) - visible] == (
        banks["experts"][visible:]
    )

    # Strong property across two full volumes: the second bank starts inside part-0002;
    # its 2 MiB-aligned logical offset maps to an exactly 2 MiB-aligned file offset there.
    experts2 = objects["experts2"]
    part2_begin = files[0]["payload_bytes"] + files[1]["payload_bytes"]
    local2 = experts2["offset"] - part2_begin
    segment_start2 = PAYLOAD_ALIGNMENT + local2
    assert segment_start2 % EXPERT_STREAM_ALIGNMENT == 0
    visible2 = files[2]["payload_bytes"] - local2
    assert part2[segment_start2 : segment_start2 + visible2] == banks["experts2"][:visible2]

    # The straddle resumes at part-0003's payload start (4 KiB aligned, as documented).
    remainder = (tmp_path / files[3]["path"]).read_bytes()
    assert remainder[PAYLOAD_ALIGNMENT : PAYLOAD_ALIGNMENT + len(banks["experts2"]) - visible2] == (
        banks["experts2"][visible2:]
    )
    assert remainder[-8:] == struct.pack("<2f", 1.0, 2.0)
    with Artifact(path) as reader:
        assert reader.read_object("experts") == banks["experts"]
        assert reader.read_object("experts2") == banks["experts2"]


def test_artifacts_without_expert_streams_keep_the_4kib_default(tmp_path):
    path = tmp_path / "plain.ninfer"
    with ArtifactWriter(
        path,
        [TensorSpec("w", (2, 2), "bf16", "contiguous_le_v1")],
        components={"text": {"config": {}}},
        bindings={"w": {"object": "w"}},
    ) as writer:
        writer.write_object("w", bytes(8))
        assert writer.segment_alignment == PAYLOAD_ALIGNMENT
        assert writer.payload_offset == PAYLOAD_ALIGNMENT
    raw = path.read_bytes()
    _, json_bytes, _ = HEADER.unpack(raw[: HEADER.size])
    assert HEADER.size + json_bytes == PAYLOAD_ALIGNMENT
    assert len(raw) == PAYLOAD_ALIGNMENT + 8
