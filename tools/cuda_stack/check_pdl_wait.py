"""Fails when a programmatic dependent kernel may read its producer's output before its PDL wait.

A kernel launched with programmatic dependent launch (PDL) can start while the kernel before it still
runs; `griddepcontrol.wait` (SASS ACQBULK) is where it waits for that kernel's writes. A global load
before the wait may read only memory no running producer writes: weights, scales, descriptors, or a
documented serialized producer's output. Source order does not decide this: the compiler issues a
`const __restrict__` load as a non-coherent LDG.E.CONSTANT and may move it above the wait. `0e1c221d`
found the NVFP4 A16 GEMV's activation loads hoisted so, a race the PDL skew build cannot see (its spin
after the wait keeps the compiler from hoisting), and `0a2e3de5` took the qualifier off every
producer-written input then found.

For every kernel with a wait in the given CUDA objects, `check` lists the global loads before its first
ACQBULK (LDG, LD, LDGSTS, UTMALDG, UBLKCP) and the kernel parameter each load's address traces back to:
`p+0x18` is the 64-bit parameter at byte 0x18 of the parameter block, `*p+0x8` an address read from
memory reached from parameter 0x8, `?` an address the trace could not follow. It compares them with the
allow-list of reviewed origins per kernel; a kernel missing from it may have no load before its wait. A
new origin fails until it is reviewed: an immutable operand is added with `update`; a producer-written
one is a race, fixed by loading it after the wait (or dropping its `__restrict__`).

Kernels are keyed by check_cuda_stack.py's stable names (nvcc's per-file ids replaced by the file's
name). The objects' SASS listings are some 6 GB of text: each is streamed, and only the instructions
before each function's first wait are kept.

    python tools/cuda_stack/check_pdl_wait.py check (OBJECT... | --objects-file FILE) [--allowlist FILE]
    python tools/cuda_stack/check_pdl_wait.py update (OBJECT... | --objects-file FILE) [--allowlist FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_cuda_stack import stable_name  # noqa: E402

DEFAULT_ALLOWLIST = Path(__file__).with_name("pdl_wait_allowlist.json")
# sm_120's kernel parameters start at this offset of constant bank 0; below it are the driver's values.
PARAMETER_BASE = 0x380
WAIT = "ACQBULK"
# Global loads, by opcode (the text before the first '.'); LDGDEPBAR and LDGSTS's commit are barriers.
LOADS = {"LDG", "LD", "LDGSTS", "UTMALDG", "UBLKCP"}
# Instructions whose first operand is not a register they write.
NO_DESTINATION = {"LDGSTS", "UTMALDG", "UBLKCP", "UTMASTG", "UTMAPF", "ST", "STG", "STS", "STL", "RED", "REDG",
                  "REDUX", "BRA", "BRX", "CALL", "RET", "EXIT", "BAR", "BSSY", "BSYNC", "WARPSYNC", "NOP",
                  "MEMBAR", "FENCE", "ERRBAR", "DEPBAR", "LDGDEPBAR", "ACQBULK", "PREEXIT", "CCTL", "SYNCS"}
INSTRUCTION = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(.*?)\s*;")
REGISTER = re.compile(r"(?<![\w.])(U?R)(\d+)\b")
CONSTANT = re.compile(r"c\[0x0\]\[(0x[0-9a-f]+)\]")
BRACKET = re.compile(r"\[([^\[\]]*)\]")


class Instruction:
    __slots__ = ("opcode", "base", "operands", "predicated")

    def __init__(self, text: str):
        self.predicated = text.startswith("@")
        if self.predicated:
            text = text.split(None, 1)[1]
        self.opcode, _, rest = text.partition(" ")
        self.base = self.opcode.split(".", 1)[0]
        self.operands = [op.strip() for op in rest.split(",")] if rest.strip() else []

    def width(self) -> int:
        parts = self.opcode.split(".")
        return 4 if "128" in parts else 2 if ("64" in parts or "WIDE" in parts) else 1

    def written(self) -> set[str]:
        """The registers this instruction writes (its first operand, widened by .64 / .128 / .WIDE)."""
        if self.base in NO_DESTINATION or not self.operands:
            return set()
        match = REGISTER.fullmatch(self.operands[0].split(".", 1)[0].lstrip("-~!|"))
        if not match:
            return set()
        kind, first = match.group(1), int(match.group(2))
        return {f"{kind}{first + k}" for k in range(self.width())}

    def address_registers(self) -> set[str]:
        """A load's address registers: the last bracket of each source operand (`desc[UR36][R4.64+0x10]` gives R4;
        LDGSTS and the bulk copies' first operand is their shared-memory destination)."""
        found: set[str] = set()
        for operand in self.operands[1:]:
            brackets = BRACKET.findall(operand)
            if brackets:
                found |= {f"{k}{n}" for k, n in REGISTER.findall(brackets[-1])}
        return found

    def pointer_sources(self) -> tuple[list[str], list[int]]:
        """The source registers and constant-bank offsets a 64-bit address follows through this instruction:
        IMAD.WIDE's addend, LEA's base, every source of other arithmetic and moves."""
        operands = [op for op in self.operands[1:] if not re.fullmatch(r"!?U?P(T|\d+)", op)]
        if self.base in ("IMAD", "UIMAD") and "WIDE" in self.opcode.split("."):
            operands = operands[-1:]
        elif self.base in ("LEA", "ULEA") and "HI" not in self.opcode.split("."):
            operands = operands[1:2]
        registers = [f"{k}{n}" for op in operands for k, n in REGISTER.findall(op)]
        constants = [int(offset, 16) for op in operands for offset in CONSTANT.findall(op)]
        return registers, constants


def origins(prologue: list[Instruction], at: int, registers: set[str], depth: int = 0) -> set[str]:
    """Where the address in `registers`, as instruction `at` of a kernel's pre-wait instructions uses it, comes from:
    the 64-bit kernel parameters it traces back to (`p+0x18`), an address read from memory (`*` before that load's
    own origins), or `?`."""
    found: set[str] = set()
    pending = [(at, register) for register in registers if register not in ("RZ", "URZ")]
    seen: set[tuple[int, str]] = set()
    while pending:
        index, register = pending.pop()
        if (index, register) in seen:
            continue
        seen.add((index, register))
        writer = index - 1
        while writer >= 0 and register not in prologue[writer].written():
            writer -= 1
        if writer < 0:
            continue  # a special register or a value the kernel did not compute from memory
        instruction = prologue[writer]
        if instruction.predicated:
            pending.append((writer, register))  # an earlier definition may be the one in effect
        if instruction.base in LOADS:
            if depth < 3:
                found |= {"*" + origin for origin in
                          origins(prologue, writer, instruction.address_registers(), depth + 1) or {"?"}}
            else:
                found.add("?")
            continue
        if instruction.base in ("LDC", "LDCU", "ULDC"):
            # A 64-bit parameter is a pointer; a 32-bit constant feeds an index or one half of a split pointer,
            # which the trace does not follow.
            if instruction.width() == 2:
                found |= {f"p+{int(offset, 16) - PARAMETER_BASE:#x}"
                          for offset in CONSTANT.findall(" ".join(instruction.operands[1:]))
                          if int(offset, 16) >= PARAMETER_BASE}
            continue
        registers, constants = instruction.pointer_sources()
        # A 64-bit constant operand (IMAD.WIDE R2, R3, 0x2, c[0x0][0x388]) is the parameter itself.
        found |= {f"p+{offset - PARAMETER_BASE:#x}" for offset in constants if offset >= PARAMETER_BASE}
        pending.extend((writer, source) for source in registers)
    return found


def prewait_loads(prologue: list[Instruction]) -> set[str]:
    """The origins of every global load in a kernel's instructions before its first wait."""
    found: set[str] = set()
    for index, instruction in enumerate(prologue):
        if instruction.base in LOADS:
            found |= origins(prologue, index, instruction.address_registers()) or {"?"}
    return found


def parse_listing(lines) -> dict[str, set[str]]:
    """Kernel name -> its pre-wait load origins, for every function of a `cuobjdump -sass` listing with a wait."""
    kernels: dict[str, set[str]] = {}
    name, prologue = None, []
    for line in lines:
        if "Function :" in line:
            name, prologue = stable_name(line.split("Function :", 1)[1].strip()), []
            continue
        if name is None:
            continue
        if WAIT not in line:
            prologue.append(line)  # parsed only if the function waits
            continue
        match = INSTRUCTION.match(line)
        if not match or match.group(1).split()[-1] != WAIT:
            prologue.append(line)
            continue
        instructions = [Instruction(m.group(1)) for m in map(INSTRUCTION.match, prologue) if m]
        kernels.setdefault(name, set()).update(prewait_loads(instructions))
        name, prologue = None, []  # the rest of the function is after its wait
    return kernels


def object_loads(job: tuple[str, str]) -> dict[str, list[str]]:
    cuobjdump, path = job
    with subprocess.Popen([cuobjdump, "-sass", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, errors="replace") as process:
        kernels = parse_listing(process.stdout)
    if process.returncode != 0:
        raise SystemExit(f"cuobjdump -sass failed on {path}")
    return {name: sorted(found) for name, found in kernels.items()}


def kernel_loads(objects: list[Path], cuobjdump: str, jobs: int) -> dict[str, list[str]]:
    """Kernel name -> its pre-wait load origins over every object (a header kernel may be compiled in several)."""
    merged: dict[str, set[str]] = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for kernels in pool.map(object_loads, [(cuobjdump, str(path)) for path in objects]):
            for name, found in kernels.items():
                merged.setdefault(name, set()).update(found)
    return {name: sorted(found) for name, found in sorted(merged.items())}


def format_allowlist(listed: dict[str, list[str]]) -> str:
    """One kernel per line, so a reviewed change shows as a one-line diff."""
    lines = [f"  {json.dumps(name)}: {json.dumps(found)}" for name, found in sorted(listed.items())]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def check(kernels: dict[str, list[str]], allowed: dict[str, list[str]]) -> list[str]:
    failures = []
    for name, found in sorted(kernels.items()):
        new = sorted(set(found) - set(allowed.get(name, [])))
        if new:
            failures.append(f"{name}: loads before its PDL wait from {', '.join(new)} (allowed: "
                            f"{', '.join(allowed.get(name, [])) or 'none'})")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "update"))
    parser.add_argument("objects", type=Path, nargs="*", help="CUDA object files")
    parser.add_argument("--objects-file", type=Path, help="a file listing the object files, one per line")
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "cuobjdump")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    objects = list(args.objects)
    if args.objects_file:
        objects += [Path(line.strip()) for line in args.objects_file.read_text(encoding="utf-8").splitlines()
                    if line.strip().endswith((".obj", ".o"))]
    objects = [path for path in objects if path.suffix in (".obj", ".o") and ".cu." in path.name]
    if not objects:
        raise SystemExit("no CUDA object files given")
    kernels = kernel_loads(objects, args.cuobjdump, args.jobs)
    if not kernels:
        # The engine's ops use PDL throughout: finding no wait means the wait instruction went unrecognised
        # (another toolchain's SASS), and the check would pass having checked nothing.
        raise SystemExit(f"no kernel in {len(objects)} objects waits on a PDL dependency: the wait was not "
                         "recognised, so nothing was checked")
    listed = {name: found for name, found in kernels.items() if found}
    if args.command == "update":
        args.allowlist.write_text(format_allowlist(listed), encoding="utf-8")
        print(f"{len(listed)} of {len(kernels)} waiting kernels load before their wait; wrote {args.allowlist}")
        return 0
    allowed = json.loads(args.allowlist.read_text(encoding="utf-8"))
    failures = check(kernels, allowed)
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"{len(kernels)} waiting kernels in {len(objects)} objects checked, {len(listed)} load before their wait, "
          f"{len(failures)} from an origin not reviewed")
    if failures:
        print("A load before a PDL wait reads a new parameter: if it is a producer's output, load it after the wait "
              "(and drop its __restrict__); if it is immutable, after review run "
              "`python tools/cuda_stack/check_pdl_wait.py update ...`.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
