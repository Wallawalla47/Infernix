"""Inspect Infernix v3 configurations, objects, bindings and files without numerical libraries.

``--compare OTHER`` lists what differs between two artifacts of one model, for a reconversion that
must keep every other object: the components, the resources and every binding's bound objects
(record and payload, byte for byte) and Use, matched by name rather than by position, so objects
that only moved compare equal."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .reader import Artifact
from .schema import TensorObject

_COMPARE_CHUNK_BYTES = 64 << 20


def artifact_summary(artifact: Artifact) -> dict:
    tensors = [obj for obj in artifact.objects if isinstance(obj, TensorObject)]
    return {
        "path": str(artifact.path),
        "version": 3,
        "artifact_id": artifact.artifact_id.hex(),
        "name": artifact.directory.metadata.get("name"),
        "components": artifact.directory.components,
        "file_bytes": artifact.file_bytes,
        "payload_bytes": artifact.payload_bytes,
        "files": len(artifact.directory.files),
        "objects": len(artifact.objects),
        "tensors": len(tensors),
        "resources": len(artifact.objects) - len(tensors),
        "bindings": len(artifact.directory.bindings),
        "uses": len(artifact.directory.uses),
        "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
        "layouts": dict(sorted(Counter(obj.layout for obj in tensors).items())),
    }


def _record(obj) -> tuple:
    if isinstance(obj, TensorObject):
        return ("tensor", obj.format, obj.layout, tuple(obj.shape), obj.bytes)
    return ("resource", obj.encoding, obj.bytes)


def _parts(reference: dict) -> list[tuple[str, tuple | None]]:
    if "parts" in reference:
        return [(part["object"], tuple(part["range"]) if "range" in part else None) for part in reference["parts"]]
    return [(reference["object"], None)]


class _Comparison:
    def __init__(self, first: Artifact, second: Artifact):
        self.first, self.second = first, second
        self._same: dict[tuple[str, str], bool] = {}

    def same_object(self, first_id: str, second_id: str) -> bool:
        key = (first_id, second_id)
        if key not in self._same:
            a, b = self.first.object(first_id), self.second.object(second_id)
            same = _record(a) == _record(b)
            for at in range(0, a.bytes if same else 0, _COMPARE_CHUNK_BYTES):
                count = min(_COMPARE_CHUNK_BYTES, a.bytes - at)
                if self.first.read_range(a.offset + at, count) != self.second.read_range(b.offset + at, count):
                    same = False
                    break
            self._same[key] = same
        return self._same[key]

    def same_reference(self, first: dict, second: dict) -> bool:
        a, b = _parts(first), _parts(second)
        return len(a) == len(b) and all(
            ra == rb and self.same_object(ia, ib) for (ia, ra), (ib, rb) in zip(a, b)
        )


def compare_artifacts(first: Artifact, second: Artifact) -> dict:
    """What differs between two artifacts: bindings whose bound objects differ in record or bytes,
    bindings and Uses present in only one, Uses whose policy or auxiliary bytes differ, resources
    whose bytes differ, and whether the components and metadata are equal."""

    comparison = _Comparison(first, second)
    a, b = first.directory, second.directory
    out = {"components_equal": a.components == b.components, "metadata_equal": a.metadata == b.metadata}
    names = sorted(set(a.bindings) | set(b.bindings))
    out["only_in_first"] = [n for n in names if n not in b.bindings]
    out["only_in_second"] = [n for n in names if n not in a.bindings]
    shared = [n for n in names if n in a.bindings and n in b.bindings]
    out["differing_bindings"] = [n for n in shared if not comparison.same_reference(a.bindings[n], b.bindings[n])]
    out["identical_bindings"] = len(shared) - len(out["differing_bindings"])

    def uses(directory):
        return {(use["parameter"], use["input"]): use for use in directory.uses}

    ua, ub = uses(a), uses(b)
    differing_uses = sorted(set(ua) ^ set(ub))
    for key in sorted(set(ua) & set(ub)):
        x, y = ua[key], ub[key]
        xa, ya = x.get("auxiliaries", {}), y.get("auxiliaries", {})
        if x["activation_policy"] != y["activation_policy"] or set(xa) != set(ya) or not all(
            comparison.same_reference(xa[role], ya[role]) for role in xa
        ):
            differing_uses.append(key)
    out["differing_uses"] = [f"{parameter}@{name}" for parameter, name in differing_uses]
    resources = sorted(
        obj.id for obj in first.objects if not isinstance(obj, TensorObject) and obj.id in second.by_id
    )
    out["differing_resources"] = [r for r in resources if not comparison.same_object(r, r)]
    out["resources_only_in_one"] = sorted(
        {obj.id for obj in first.objects if not isinstance(obj, TensorObject)}
        ^ {obj.id for obj in second.objects if not isinstance(obj, TensorObject)}
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--objects", action="store_true", help="include physical object records"
    )
    parser.add_argument(
        "--bindings", action="store_true", help="include logical bindings and Uses"
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    parser.add_argument(
        "--compare", type=Path, metavar="OTHER",
        help="list what differs from another artifact of the same model (reads both payloads)",
    )
    args = parser.parse_args()

    if args.compare is not None:
        with Artifact(args.artifact) as first, Artifact(args.compare) as second:
            print(json.dumps(compare_artifacts(first, second), ensure_ascii=False, indent=2))
        return
    with Artifact(args.artifact) as artifact:
        summary = artifact_summary(artifact)
        if args.json:
            if args.objects:
                summary["object_records"] = [obj.to_json() for obj in artifact.objects]
            if args.bindings:
                summary["binding_records"] = artifact.directory.bindings
                summary["use_records"] = list(artifact.directory.uses)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        for key, value in summary.items():
            print(f"{key}: {value}")
        if args.objects:
            for obj in artifact.objects:
                storage = (
                    f"{obj.format}/{obj.layout} {list(obj.shape)}"
                    if isinstance(obj, TensorObject)
                    else obj.encoding
                )
                print(
                    f"{obj.offset:>14} {obj.bytes:>14} {obj.kind:<8} {storage:<42} {obj.id}"
                )
        if args.bindings:
            print(
                json.dumps(
                    {
                        "bindings": artifact.directory.bindings,
                        "uses": list(artifact.directory.uses),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
