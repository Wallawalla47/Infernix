"""Proves a pure refactor left the GPU code unchanged: compares every kernel's SASS between two
builds of a binary, so a change that should alter no instruction (a header split, a rename of a
host symbol, moved launch code) is shown to be one without a GPU slot (TensorFold's SASS identity,
review TX-15).

`cuobjdump -sass BINARY` lists each kernel's instructions per architecture. Each kernel is keyed by
its stable name (check_cuda_stack.py's: nvcc's per-file ids replaced by the file's name) and its
architecture; its instructions are compared with addresses and encodings removed, which leaves
the instruction text and its operands (branch targets are offsets inside the function, so they
compare too). Exit status 0 when every kernel is identical, 1 when any differs, appears or goes.

    python tools/cuda_stack/sass_identity.py BEFORE AFTER [--cuobjdump EXE] [--show N]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cuda_stack import stable_name  # noqa: E402

ADDRESS = "/*"


def kernels(binary: Path, cuobjdump: str) -> dict[tuple[str, str], list[str]]:
    """(architecture, stable kernel name) -> its normalized instruction lines."""
    text = subprocess.run([cuobjdump, "-sass", str(binary)], capture_output=True, text=True, check=True).stdout
    out: dict[tuple[str, str], list[str]] = {}
    arch, current = "", None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("code for "):
            arch, current = line[len("code for "):], None
        elif line.startswith("Function :"):
            key = (arch, stable_name(line[len("Function :"):].strip()))
            current = out.setdefault(key, [])
        elif current is not None and line.startswith(ADDRESS):
            # `/*0040*/   IMAD R1, ... ;   /* 0x... */`: the instruction between the two comments.
            body = line.split("*/", 1)[1]
            body = body.split("/*", 1)[0].strip()
            if body:
                current.append(body)
    return out


def digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    parser.add_argument("--show", type=int, default=3, help="diff lines shown for up to N differing kernels")
    args = parser.parse_args()

    before, after = kernels(args.before, args.cuobjdump), kernels(args.after, args.cuobjdump)
    removed = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(k for k in set(before) & set(after) if digest(before[k]) != digest(after[k]))
    same = len(set(before) & set(after)) - len(changed)
    print(f"{same} kernels identical, {len(changed)} changed, {len(added)} added, {len(removed)} removed")
    for arch, name in removed:
        print(f"  removed  {arch}  {name}")
    for arch, name in added:
        print(f"  added    {arch}  {name}")
    for index, (arch, name) in enumerate(changed):
        print(f"  changed  {arch}  {name}")
        if index < args.show:
            diff = difflib.unified_diff(before[(arch, name)], after[(arch, name)], lineterm="", n=1)
            for line in list(diff)[2:42]:
                print(f"           {line}")
    return 0 if not (removed or added or changed) else 1


if __name__ == "__main__":
    sys.exit(main())
