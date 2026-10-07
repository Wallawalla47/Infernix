from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.artifact.framing import HEADER, PART_MAGIC
from tools.artifact.reader import Artifact
from tools.artifact.rename import rename
from tools.artifact.schema import ArtifactError, ResourceSpec
from tools.artifact.writer import ArtifactWriter

PAYLOAD = bytes(range(251)) * 110


def _sharded(path: Path) -> bytes:
    with ArtifactWriter(
        path,
        [ResourceSpec("data", len(PAYLOAD))],
        components={"text": {"config": {}, "resources": {"data": "data"}}},
        bindings={},
        max_file_bytes=12288,
    ) as writer:
        writer.write_object("data", PAYLOAD)
        identity = writer.artifact_id
    report = {
        "output": str(path),
        "files": [path.name] + [f"{path.name}.part-{i:04d}" for i in (1, 2, 3)],
    }
    path.with_name(path.name + ".conversion.json").write_text(json.dumps(report))
    return identity


def test_rename_moves_volumes_and_keeps_payload_in_place(tmp_path):
    old = tmp_path / "model.ninfer"
    identity = _sharded(old)
    old_header = HEADER.unpack(old.read_bytes()[: HEADER.size])
    (tmp_path / "model.ninfer.ngram").write_bytes(b"volume")

    result = rename(old, "model.infernix")

    new = tmp_path / "model.infernix"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "model.infernix",
        "model.infernix.conversion.json",
        "model.infernix.part-0001",
        "model.infernix.part-0002",
        "model.infernix.part-0003",
        "model.ninfer.ngram",
    ]
    assert result["not_renamed"] == [str(tmp_path / "model.ninfer.ngram")]
    # Same directory length and identity: the payload did not move.
    assert HEADER.unpack(new.read_bytes()[: HEADER.size]) == old_header
    for index in (1, 2, 3):
        part = tmp_path / f"model.infernix.part-{index:04d}"
        assert HEADER.unpack(part.read_bytes()[: HEADER.size]) == (PART_MAGIC, index, identity)
    with Artifact(new) as artifact:
        assert [f.path for f in artifact.directory.files] == [
            None,
            "model.infernix.part-0001",
            "model.infernix.part-0002",
            "model.infernix.part-0003",
        ]
        assert artifact.read_object("data") == PAYLOAD
    report = json.loads((tmp_path / "model.infernix.conversion.json").read_text())
    assert report["output"] == str(new)
    assert report["files"] == [
        "model.infernix",
        "model.infernix.part-0001",
        "model.infernix.part-0002",
        "model.infernix.part-0003",
    ]


@pytest.mark.parametrize("blocker", ["model.infernix", "model.infernix.part-0002"])
def test_existing_target_leaves_the_set_unchanged(tmp_path, blocker):
    old = tmp_path / "model.ninfer"
    _sharded(old)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    (tmp_path / blocker).write_bytes(b"keep")

    with pytest.raises(ArtifactError, match="already exists"):
        rename(old, "model.infernix")

    after = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.name != blocker}
    assert after == before
    assert (tmp_path / blocker).read_bytes() == b"keep"


def test_rejects_a_path_as_the_new_name(tmp_path):
    old = tmp_path / "model.ninfer"
    _sharded(old)
    with pytest.raises(ArtifactError, match="file name"):
        rename(old, "sub/model.infernix")
