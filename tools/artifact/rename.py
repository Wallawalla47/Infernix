"""Rename an Infernix v3 file set in place, for example a `.ninfer` download to `.infernix`.

    python -m tools.artifact.rename DIR/model.ninfer model.infernix

The continuation volumes take the writer's names for the new entry (`<entry>.part-NNNN`), and the
entry's `files` table is rewritten inside the entry's existing directory space (the JSON's trailing
reserve and the alignment area before the payload), so no payload byte moves and the artifact_id is
kept. The converter's `<entry>.conversion.json` report follows, with the file names in it updated.
Other `<entry>.*` sidecars (an n-gram volume, a saved expert state) are listed, not renamed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from .framing import HEADER, PAYLOAD_ALIGNMENT
from .layouts import align_up
from .reader import Artifact
from .schema import ArtifactError, encode_directory, parse_directory

REPORT_SUFFIX = ".conversion.json"


def part_name(entry_name: str, index: int) -> str:
    return f"{entry_name}.part-{index:04d}"


def _plan(entry: Path, new_name: str) -> tuple[list[tuple[Path, Path]], bytes, bytes]:
    """Validate the set and return the volume renames, the old and the new directory region."""
    if not new_name or Path(new_name).name != new_name or new_name in (".", ".."):
        raise ArtifactError(f"{new_name!r}: give a file name, not a path")
    if new_name == entry.name:
        raise ArtifactError("the new name is the current name")
    with Artifact(entry) as artifact:
        files = artifact.directory.files
        offset = 0
        for file in files[:-1]:
            offset += file.payload_bytes
            artifact.read_range(offset, 1)  # opens and checks the next continuation volume
        payload_offset = artifact.payload_offset
    with entry.open("rb") as stream:
        old_region = stream.read(payload_offset)
    _, json_bytes, _ = HEADER.unpack(old_region[: HEADER.size])
    value = json.loads(old_region[HEADER.size : HEADER.size + json_bytes].decode("utf-8"))
    renames = []
    for index in range(1, len(value["files"])):
        old = entry.parent / value["files"][index]["path"]
        new = entry.parent / part_name(new_name, index)
        value["files"][index]["path"] = new.name
        if old != new:
            renames.append((old, new))
    data = encode_directory(value)
    capacity = payload_offset - HEADER.size
    if len(data) > capacity:
        raise ArtifactError(
            f"the new file names need {len(data) - capacity} more bytes of directory space than the "
            "entry has; convert again with the new --out name instead"
        )
    new_json_bytes = max(json_bytes, len(data))
    assert align_up(HEADER.size + new_json_bytes, PAYLOAD_ALIGNMENT) == payload_offset
    parse_directory(json.loads(data.decode("utf-8")), entry_name=new_name)
    new_region = (
        old_region[:8]
        + new_json_bytes.to_bytes(8, "little")
        + old_region[16 : HEADER.size]
        + data
        + b" " * (new_json_bytes - len(data))
        + bytes(payload_offset - HEADER.size - new_json_bytes)
    )
    targets = [new for _, new in renames] + [entry.parent / new_name]
    for target in targets:
        if target.exists():
            raise ArtifactError(f"{target} already exists")
    return renames, old_region, new_region


def _rename_report(entry: Path, new_entry: Path, names: dict[str, str]) -> Path | None:
    report = entry.with_name(entry.name + REPORT_SUFFIX)
    if not report.exists():
        return None
    target = new_entry.with_name(new_entry.name + REPORT_SUFFIX)
    if target.exists():
        raise ArtifactError(f"{target} already exists")
    pattern = re.compile("|".join(re.escape(old) for old in sorted(names, key=len, reverse=True)))
    text = pattern.sub(lambda match: names[match.group(0)], report.read_text(encoding="utf-8"))
    json.loads(text)
    target.write_text(text, encoding="utf-8")
    report.unlink()
    return target


def rename(entry: str | Path, new_name: str) -> dict:
    entry = Path(entry)
    renames, old_region, new_region = _plan(entry, new_name)
    new_entry = entry.with_name(new_name)
    done: list[tuple[Path, Path]] = []
    written = False
    try:
        for old, new in renames:
            os.rename(old, new)
            done.append((old, new))
        with entry.open("r+b") as stream:
            written = True
            stream.write(new_region)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(entry, new_entry)
    except BaseException:
        if written:
            with entry.open("r+b") as stream:
                stream.write(old_region)
        for old, new in reversed(done):
            os.rename(new, old)
        raise
    names = {entry.name: new_entry.name}
    names.update({old.name: new.name for old, new in renames})
    report = _rename_report(entry, new_entry, names)
    with Artifact(new_entry) as artifact:
        offset = 0
        for file in artifact.directory.files[:-1]:
            offset += file.payload_bytes
            artifact.read_range(offset, 1)
        artifact_id = artifact.artifact_id.hex()
    others = sorted(
        str(path) for path in entry.parent.glob(entry.name + ".*") if path.is_file()
    )
    return {
        "entry": str(new_entry),
        "artifact_id": artifact_id,
        "renamed": {old: new for old, new in names.items()},
        "report": str(report) if report else None,
        "not_renamed": others,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("entry", help="the entry file of the set")
    parser.add_argument("new_name", help="the new entry file name, in the same directory")
    args = parser.parse_args()
    print(json.dumps(rename(args.entry, args.new_name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
