"""Fetch named tensors and the frontend files of a Hugging Face checkpoint without its shards.

A GGUF import (tools.convert --gguf) takes its config, tokenizer and, for the MTP drafter, the MTP
head from the original checkpoint, but needs only a few GB of a 180 GB repository. This tool reads
the index, then each needed shard's safetensors header, and HTTP range-reads exactly the selected
tensors into one local ``model.safetensors`` (bytes unchanged, SHA256 per tensor recorded), plus
the small frontend files. The revision is pinned: every request names the commit.

    python -m tools.convert.fetch_tensors --repo Qwen/Qwen3.8-Flash-Next-FP8 --revision SHA \
        --out DIR --pattern "mtp.*"
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

FRONTEND_FILES = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "vocab.json", "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json",
)


def _url(repo: str, revision: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}"


def _get(url: str, first: int | None = None, last: int | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "infernix-fetch"})
    if first is not None:
        request.add_header("Range", f"bytes={first}-{last}")
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if first is not None and len(data) != last - first + 1:
        raise IOError(f"{url}: range {first}-{last} returned {len(data)} bytes")
    return data


def fetch(repo: str, revision: str, out: Path, patterns: list[str], files: tuple[str, ...] = FRONTEND_FILES) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for name in files:
        try:
            data = _get(_url(repo, revision, name))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        (out / name).write_bytes(data)
    index = json.loads(_get(_url(repo, revision, "model.safetensors.index.json")))
    wanted = sorted(name for name in index["weight_map"] if any(fnmatchcase(name, p) for p in patterns))
    if not wanted:
        raise ValueError(f"no tensor of {repo} matches {patterns}")
    by_shard: dict[str, list[str]] = {}
    for name in wanted:
        by_shard.setdefault(index["weight_map"][name], []).append(name)
    # Every tensor's place in the local file follows from the shards' headers, so the header is written
    # first and each tensor's bytes stream into place: memory stays bounded by one request (256 MiB),
    # whatever the selection's size (a 102 GB n-gram table).
    header: dict = {}
    plan: list[tuple[str, int, list[list[tuple[int, int, str]]]]] = []  # shard url, data base, groups
    offset = 0
    for shard, names in sorted(by_shard.items()):
        url = _url(repo, revision, shard)
        length = struct.unpack("<Q", _get(url, 0, 7))[0]
        info = json.loads(_get(url, 8, 8 + length - 1))
        # Adjacent ranges are merged into one request.
        spans = sorted((info[n]["data_offsets"][0], info[n]["data_offsets"][1], n) for n in names)
        groups: list[list[tuple[int, int, str]]] = []
        for span in spans:
            if groups and span[0] - groups[-1][-1][1] <= 1 << 20 and span[1] - groups[-1][0][0] <= 256 << 20:
                groups[-1].append(span)
            else:
                groups.append([span])
        for begin, finish, name in spans:
            header[name] = {"dtype": info[name]["dtype"], "shape": info[name]["shape"],
                            "data_offsets": [offset, offset + finish - begin]}
            offset += finish - begin
        plan.append((shard, 8 + length, groups))
    header["__metadata__"] = {"format": "pt", "source": f"{repo}@{revision}"}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    manifest = {"repo": repo, "revision": revision, "tensors": {}}
    temporary = out / "model.safetensors.tmp"
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for shard, base, groups in plan:
            url = _url(repo, revision, shard)
            for group in groups:
                start, end = group[0][0], max(s[1] for s in group)
                data = _get(url, base + start, base + end - 1)
                for begin, finish, name in group:  # in local order: spans are sorted within a shard
                    blob = data[begin - start : finish - start]
                    manifest["tensors"][name] = {"shard": shard, "sha256": hashlib.sha256(blob).hexdigest()}
                    stream.write(blob)
            print(f"{shard}: {sum(len(g) for g in groups)} tensors", flush=True)
    temporary.replace(out / "model.safetensors")
    (out / "fetch_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True, help="a commit SHA (pinned)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pattern", action="append", default=[], help="fnmatch pattern of tensor names")
    args = parser.parse_args(argv)
    manifest = fetch(args.repo, args.revision, args.out, args.pattern or ["mtp.*"])
    print(f"wrote {args.out / 'model.safetensors'}: {len(manifest['tensors'])} tensors", flush=True)


if __name__ == "__main__":
    main()
