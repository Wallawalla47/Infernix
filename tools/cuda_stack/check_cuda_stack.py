"""Fails when a CUDA kernel's per-thread stack or local memory grows: register spills and local
arrays land there, and a spill that appears in a hot kernel is a silent slowdown.

`cuobjdump --dump-resource-usage BINARY` lists every kernel's REG / STACK / LOCAL use, and its code
holds its local-memory stores and loads (SASS STL, LDL). Resource usage alone misses spills: it
reported STACK:0 LOCAL:0 for a bf16 grouped-attention kernel whose SASS held 34 STL and 32 LDL, 30
of each inside its key-tile loop (candidates IP-03).
`check` compares each kernel's STACK and LOCAL bytes and its STL and LDL counts with the allow-list
(a kernel missing from it must have none); `update` rewrites the allow-list from the binary, after a
reviewed change. Kernels are keyed by their mangled names, so a renamed or re-templated kernel is a
new entry. Names of internal linkage (anonymous namespaces, file statics) carry a per-file id that
changes with the file and the headers it includes; keys replace it with the file's name.

The server's SASS listing is some 7 GB of text, nearly all of it one 2 GB device-linked cubin:
`cuobjdump -sass` takes 19 minutes over it here, and `nvdisasm` holds the whole cubin in private
memory per process. The stores and loads are counted in the code instead: the cubins are extracted
(`cuobjdump -xelf all`) and each function's executable section is read on its own, its 16-byte
sm_120 instructions matched on their opcode field (LOCAL_OPCODES).

    python tools/cuda_stack/check_cuda_stack.py check BINARY [--allowlist FILE] [--cuobjdump EXE]
    python tools/cuda_stack/check_cuda_stack.py update BINARY [--allowlist FILE] [--cuobjdump EXE]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_ALLOWLIST = Path(__file__).with_name("cuda_stack_allowlist.json")
FUNCTION = re.compile(r"^\s*Function\s+(\S+):\s*$")
USAGE = re.compile(r"\b(STACK|LOCAL):(\d+)\b")
# An sm_120 instruction is 16 little-endian bytes whose low 12 bits are its opcode (the guard predicate follows in bits
# 12-15): STL is 0x387 and LDL 0x983, whatever their width, cache and address forms (`STL.64 [R1+0x8], R4` encodes as
# 0x...7387, `@P0 LDL.LU R2, [R1+0x8]` as 0x...0983). Checked against `cuobjdump -sass` over every kernel of the
# server when this replaced the listing (candidates IP-03).
INSTRUCTION_BYTES = 16
LOCAL_OPCODES = {0x387: "stl", 0x983: "ldl"}
# What each kernel is held to: stack and local bytes (resource usage), local stores and loads (SASS).
KEYS = ("stack", "local", "stl", "ldl")
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


def opcode_counts(code: bytes) -> dict[str, int]:
    """STL and LDL counts of one function's code: instructions whose low 12 bits are LOCAL_OPCODES'."""
    if len(code) % INSTRUCTION_BYTES:
        raise SystemExit(f"a code section of {len(code)} bytes is not whole {INSTRUCTION_BYTES}-byte instructions")
    low, high = code[0::INSTRUCTION_BYTES], code[1::INSTRUCTION_BYTES]
    counts = dict.fromkeys(LOCAL_OPCODES.values(), 0)
    for opcode, key in LOCAL_OPCODES.items():
        at = low.find(opcode & 0xFF)
        while at >= 0:
            counts[key] += (high[at] & 0x0F) == opcode >> 8
            at = low.find(opcode & 0xFF, at + 1)
    return counts


def cubin_accesses(cubin: Path) -> dict[str, dict[str, int]]:
    """Function name -> its STL and LDL counts, over the executable `.text.<name>` sections of an ELF64 cubin, each
    read on its own."""
    with cubin.open("rb") as f:
        header = f.read(64)
        if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
            raise SystemExit(f"{cubin}: not a little-endian ELF64 cubin")
        (shoff,) = struct.unpack_from("<Q", header, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", header, 0x3A)
        # (sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize)
        entry = "<IIQQQQIIQQ"
        f.seek(shoff)
        first = struct.unpack(entry, f.read(shentsize)[:struct.calcsize(entry)])
        # Extended numbering (the server's device-linked cubin has more than 65,279 sections): the count and the
        # names' index are in section 0.
        shnum = shnum or first[5]
        shstrndx = first[6] if shstrndx == 0xFFFF else shstrndx
        f.seek(shoff)
        table = f.read(shentsize * shnum)
        sections = [struct.unpack_from(entry, table, i * shentsize) for i in range(shnum)]
        f.seek(sections[shstrndx][4])
        names = f.read(sections[shstrndx][5])
        functions: dict[str, dict[str, int]] = {}
        for name_offset, kind, flags, _, offset, size, *_ in sections:
            name = names[name_offset:names.index(b"\0", name_offset)].decode("utf-8", "replace")
            # SHT_PROGBITS with SHF_EXECINSTR: a function's code.
            if kind != 1 or not flags & 0x4 or not name.startswith(".text."):
                continue
            f.seek(offset)
            functions[stable_name(name[len(".text."):])] = opcode_counts(f.read(size))
    return functions


def local_accesses(listings: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    """Kernel name -> {"stl": stores, "ldl": loads}, the largest over its listings (a template kernel may be compiled
    in several cubins)."""
    kernels: dict[str, dict[str, int]] = {}
    for listing in listings:
        for name, counts in listing.items():
            entry = kernels.setdefault(name, {"stl": 0, "ldl": 0})
            for key in ("stl", "ldl"):
                entry[key] = max(entry[key], counts[key])
    return kernels


def code_accesses(binary: Path, cuobjdump: str) -> dict[str, dict[str, int]]:
    """Kernel name -> its STL and LDL counts over every cubin in `binary`."""
    with tempfile.TemporaryDirectory(prefix="cuda_stack_") as scratch:
        subprocess.run([cuobjdump, "-xelf", "all", str(binary.resolve())], cwd=scratch, check=True,
                       capture_output=True)
        cubins = sorted(Path(scratch).glob("*.cubin"))
        if not cubins:
            raise SystemExit(f"cuobjdump extracted no cubins from {binary}")
        return local_accesses([cubin_accesses(cubin) for cubin in cubins])


def kernel_usage(binary: Path, cuobjdump: str) -> dict[str, dict[str, int]]:
    """Kernel name -> its stack and local bytes and its local store and load counts."""
    kernels = resource_usage(binary, cuobjdump)
    accesses = code_accesses(binary, cuobjdump)
    if not accesses:
        raise SystemExit(f"no function code found in {binary}")
    for name in set(kernels) | set(accesses):
        entry = kernels.setdefault(name, {"stack": 0, "local": 0})
        entry.update(accesses.get(name, {"stl": 0, "ldl": 0}))
    return kernels


def format_allowlist(listed: dict[str, dict[str, int]]) -> str:
    """One kernel per line, so a reviewed change shows as a one-line diff."""
    lines = [f"  {json.dumps(name)}: {{" + ", ".join(f"\"{key}\": {used.get(key, 0)}" for key in KEYS) + "}"
             for name, used in sorted(listed.items())]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def check(kernels: dict[str, dict[str, int]], allowed: dict[str, dict[str, int]]) -> list[str]:
    failures = []
    for name, used in sorted(kernels.items()):
        limit = allowed.get(name, {})
        for key in KEYS:
            if used.get(key, 0) > limit.get(key, 0):
                unit = "bytes" if key in ("stack", "local") else "instructions"
                failures.append(f"{name}: {key.upper()} {used[key]} {unit} > allowed {limit.get(key, 0)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "update"))
    parser.add_argument("binary", type=Path, help="an executable or library holding the kernels")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    args = parser.parse_args()
    kernels = kernel_usage(args.binary, args.cuobjdump)
    if args.command == "update":
        listed = {name: used for name, used in sorted(kernels.items()) if any(used.get(key, 0) for key in KEYS)}
        args.allowlist.write_text(format_allowlist(listed), encoding="utf-8")
        print(f"{len(listed)} of {len(kernels)} kernels use stack or local memory; wrote {args.allowlist}")
        return 0
    allowed = json.loads(args.allowlist.read_text(encoding="utf-8"))
    failures = check(kernels, allowed)
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"{len(kernels)} kernels checked, {len(failures)} over their allowed stack, local memory or local "
          "stores and loads")
    if failures:
        print("A spill or local array grew: fix it, or after review run "
              "`python tools/cuda_stack/check_cuda_stack.py update BINARY`.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
