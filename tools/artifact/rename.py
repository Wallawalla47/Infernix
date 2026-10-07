"""Rename an Infernix v3 file set in place, for example a `.ninfer` download to `.infernix`.

    python -m tools.artifact.rename DIR/model.ninfer model.infernix
    python -m tools.artifact.rename DIR/model.infernix model.infernix --numbered

By default the continuation volumes take the writer's names for the new entry
(`<entry>.part-NNNN`). `--numbered` names every file of the set as a numbered shard instead,
`model-00001-of-00003.infernix` (the entry, which is the file to open) to
`model-00003-of-00003.infernix`. Either way the entry's `files` table is rewritten inside the entry's
existing directory space (the JSON's trailing reserve and the alignment area before the payload), so
no payload byte moves and the artifact_id is kept. The converter's `<entry>.conversion.json` report
follows, with the file names in it updated. Other `<entry>.*` sidecars (an n-gram volume, a saved
expert state) are listed, not renamed.
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


def file_names(new_name: str, count: int, numbered: bool) -> list[str]:
    """The set's new file names, the entry first."""
    if not numbered:
        return [new_name] + [part_name(new_name, index) for index in range(1, count)]
    stem, dot, extension = new_name.rpartition(".")
    if not dot or not stem:
        raise ArtifactError(f"{new_name!r}: a numbered name needs an extension, e.g. model.infernix")
    return [f"{stem}-{index:05d}-of-{count:05d}.{extension}" for index in range(1, count + 1)]


def _plan(
    entry: Path, new_name: str, numbered: bool
) -> tuple[Path, list[tuple[Path, Path]], bytes, bytes]:
    """Validate the set; return the new entry, the volume renames and the old and new directory region."""
    if not new_name or Path(new_name).name != new_name or new_name in (".", ".."):
        raise ArtifactError(f"{new_name!r}: give a file name, not a path")
    with Artifact(entry) as artifact:
        files = artifact.directory.files
        offset = 0
        for file in files[:-1]:
            offset += file.payload_bytes
            artifact.read_range(offset, 1)  # opens and checks the next continuation volume
        payload_offset = artifact.payload_offset
    names = file_names(new_name, len(files), numbered)
    if names[0] == entry.name:
        raise ArtifactError("the new name is the current name")
    with entry.open("rb") as stream:
        old_region = stream.read(payload_offset)
    _, json_bytes, _ = HEADER.unpack(old_region[: HEADER.size])
    value = json.loads(old_region[HEADER.size : HEADER.size + json_bytes].decode("utf-8"))
    renames = []
    for index in range(1, len(value["files"])):
        old = entry.parent / value["files"][index]["path"]
        new = entry.parent / names[index]
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
    parse_directory(json.loads(data.decode("utf-8")), entry_name=names[0])
    new_region = (
        old_region[:8]
        + new_json_bytes.to_bytes(8, "little")
        + old_region[16 : HEADER.size]
        + data
        + b" " * (new_json_bytes - len(data))
        + bytes(payload_offset - HEADER.size - new_json_bytes)
    )
    new_entry = entry.parent / names[0]
    for target in [new for _, new in renames] + [new_entry]:
        if target.exists():
            raise ArtifactError(f"{target} already exists")
    return new_entry, renames, old_region, new_region


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


def rename(entry: str | Path, new_name: str, *, numbered: bool = False) -> dict:
    entry = Path(entry)
    new_entry, renames, old_region, new_region = _plan(entry, new_name, numbered)
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
    parser.add_argument(
        "--numbered",
        action="store_true",
        help="name every file NAME-0000i-of-0000n.EXT (the first is the entry) instead of "
        "NAME.EXT plus NAME.EXT.part-NNNN",
    )
    args = parser.parse_args()
    result = rename(args.entry, args.new_name, numbered=args.numbered)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
