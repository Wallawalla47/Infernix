"""Fails when a CUDA kernel's per-thread stack or local memory grows: register spills and local
arrays land there, and a spill that appears in a hot kernel is a silent slowdown.

`cuobjdump --dump-resource-usage BINARY` lists every kernel's REG / STACK / LOCAL use. `check`
compares each kernel's STACK and LOCAL with the allow-list (a kernel missing from it must use
none); `update` rewrites the allow-list from the binary, after a reviewed change. Kernels are keyed
by their mangled names, so a renamed or re-templated kernel is a new entry. Names of internal
linkage (anonymous namespaces, file statics) carry a per-file id that changes with the file and the
headers it includes; keys replace it with the file's name.

    python tools/cuda_stack/check_cuda_stack.py check BINARY [--allowlist FILE] [--cuobjdump EXE]
    python tools/cuda_stack/check_cuda_stack.py update BINARY [--allowlist FILE] [--cuobjdump EXE]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ALLOWLIST = Path(__file__).with_name("cuda_stack_allowlist.json")
FUNCTION = re.compile(r"^\s*Function\s+(\S+):\s*$")
USAGE = re.compile(r"\b(STACK|LOCAL):(\d+)\b")
# nvcc's internal-linkage names: an anonymous namespace `<len>_GLOBAL__N__<hash>_<n>_<file>_<hash>_<id>`
# (its identifier length first) and a file-static prefix `__nv_static_<n>__<hash>_<n>_<file>_<hash>_<id>__`.
ANONYMOUS = re.compile(r"(\d+)(_GLOBAL__N__[0-9a-f]+_(\d+)_)")
FILE_STATIC = re.compile(r"^__nv_static_\d+__[0-9a-f]+_(\d+)_")


def stable_name(name: str) -> str:
    """The kernel's key: its mangled name with nvcc's per-file ids replaced by the file's name."""
    match = FILE_STATIC.match(name)
    if match:
        file_end = match.end() + int(match.group(1))
        # The id after the file's hash is absent for some kernels (`qsa_cu_940f13ef__ZN...`).
        tail = re.match(r"_[0-9a-f]+(?:_\d+)?__", name[file_end:])
        if tail:
            name = f"__nv_static__{name[match.end():file_end]}__{name[file_end + tail.end():]}"
    out, pos = [], 0
    while (match := ANONYMOUS.search(name, pos)):
        start, file_end = match.start(2), match.end() + int(match.group(3))
        suffix = re.match(r"_[0-9a-f]+_", name[file_end:])
        digits = match.group(1)
        end = None
        if suffix:
            id_start = file_end + suffix.end()
            # The identifier's length prefix may follow other digits of the mangled name: take the
            # suffix of the digit run whose extent ends inside the id's digits.
            for k in range(len(digits)):
                if digits[k] == "0":  # a length has no leading zero
                    continue
                candidate = start + int(digits[k:])
                if id_start < candidate <= len(name) and name[id_start:candidate].isdigit():
                    end = candidate
                    out.append(name[pos:match.start(1) + k])
                    break
        if end is None:
            # A namespace with no id after the file's hash (`..._6_qsa_cu_940f13ef11pool_kernel`):
            # only the per-build hash before the file's name is dropped.
            out.append(name[pos:start])
            out.append(f"_GLOBAL__N__{name[match.end():file_end]}")
            pos = file_end
            continue
        out.append(f"_GLOBAL__N__{name[match.end():file_end]}")
        pos = end
    out.append(name[pos:])
    return "".join(out)


def resource_usage(binary: Path, cuobjdump: str) -> dict[str, dict[str, int]]:
    """Kernel name -> {"stack": bytes, "local": bytes}, the largest over the binary's cubins."""
    text = subprocess.run([cuobjdump, "--dump-resource-usage", str(binary)], check=True,
                          capture_output=True, text=True, errors="replace").stdout
    kernels: dict[str, dict[str, int]] = {}
    name = None
    for line in text.splitlines():
        match = FUNCTION.match(line)
        if match:
            name = stable_name(match.group(1))
            continue
        if name is None:
            continue
        values = {key.lower(): int(value) for key, value in USAGE.findall(line)}
        if values:
            entry = kernels.setdefault(name, {"stack": 0, "local": 0})
            for key in ("stack", "local"):
                entry[key] = max(entry[key], values.get(key, 0))
            name = None
    if not kernels:
        raise SystemExit(f"cuobjdump reported no kernels in {binary}")
    return kernels


def format_allowlist(listed: dict[str, dict[str, int]]) -> str:
    """One kernel per line, so a reviewed change shows as a one-line diff."""
    lines = [f"  {json.dumps(name)}: {{\"stack\": {used['stack']}, \"local\": {used['local']}}}"
             for name, used in sorted(listed.items())]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def check(kernels: dict[str, dict[str, int]], allowed: dict[str, dict[str, int]]) -> list[str]:
    failures = []
    for name, used in sorted(kernels.items()):
        limit = allowed.get(name, {"stack": 0, "local": 0})
        for key in ("stack", "local"):
            if used[key] > limit.get(key, 0):
                failures.append(f"{name}: {key.upper()} {used[key]} bytes > allowed {limit.get(key, 0)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "update"))
    parser.add_argument("binary", type=Path, help="an executable or library holding the kernels")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    args = parser.parse_args()
    kernels = resource_usage(args.binary, args.cuobjdump)
    if args.command == "update":
        listed = {name: used for name, used in sorted(kernels.items()) if used["stack"] or used["local"]}
        args.allowlist.write_text(format_allowlist(listed), encoding="utf-8")
        print(f"{len(listed)} of {len(kernels)} kernels use stack or local memory; wrote {args.allowlist}")
        return 0
    allowed = json.loads(args.allowlist.read_text(encoding="utf-8"))
    failures = check(kernels, allowed)
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"{len(kernels)} kernels checked, {len(failures)} over their allowed stack or local memory")
    if failures:
        print("A spill or local array grew: fix it, or after review run "
              "`python tools/cuda_stack/check_cuda_stack.py update BINARY`.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
