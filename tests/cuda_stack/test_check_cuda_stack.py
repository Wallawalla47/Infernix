"""check_cuda_stack: growth past the allow-list fails, shrinkage and listed use pass, an unlisted
kernel may use no stack or local memory and have no local stores or loads in its code, and the
written allow-list reads back unchanged."""

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "cuda_stack"))

import check_cuda_stack as tool  # noqa: E402

SECTION = "<IIQQQQIIQQ"


def cubin(functions: dict[str, bytes], extended: bool) -> bytes:
    """An ELF64 file holding each function's code in its `.text.<name>` section (SHT_PROGBITS, alloc + exec), its
    section count and names' index in the header or, extended, in section 0."""
    names = b"\0.shstrtab\0" + b"".join(b".text." + name.encode() + b"\0" for name in functions)
    body = bytearray(64)
    offsets = []
    for code in functions.values():
        offsets.append(len(body))
        body += code
    names_offset = len(body)
    body += names
    count = 2 + len(functions)
    table = [struct.pack(SECTION, 0, 0, 0, 0, 0, count if extended else 0, 1 if extended else 0, 0, 0, 0),
             struct.pack(SECTION, 1, 3, 0, 0, names_offset, len(names), 0, 0, 1, 0)]
    name_at = len(b"\0.shstrtab\0")
    for (name, code), offset in zip(functions.items(), offsets):
        table.append(struct.pack(SECTION, name_at, 1, 0x6, 0, offset, len(code), 0, 0, 128, 0))
        name_at += len(".text.") + len(name) + 1
    shoff = len(body)
    body += b"".join(table)
    body[:64] = (b"\x7fELF" + bytes([2, 1, 1]) + bytes(9) +
                 struct.pack("<HHIQQQIHHHHHH", 2, 190, 1, 0, 0, shoff, 0, 64, 0, 0, struct.calcsize(SECTION),
                             0 if extended else count, 0xFFFF if extended else 1))
    return bytes(body)


def main() -> int:
    allowed = {"k_spill": {"stack": 32, "local": 0}, "k_local": {"stack": 0, "local": 64}}
    clean = {"k_spill": {"stack": 24, "local": 0}, "k_local": {"stack": 0, "local": 64},
             "k_none": {"stack": 0, "local": 0}}
    assert tool.check(clean, allowed) == [], "within the allow-list must pass"
    grown = dict(clean, k_spill={"stack": 40, "local": 0})
    assert len(tool.check(grown, allowed)) == 1, "a stack that grew must fail"
    unlisted = dict(clean, k_new={"stack": 8, "local": 0})
    assert len(tool.check(unlisted, allowed)) == 1, "an unlisted kernel with a stack must fail"
    local = dict(clean, k_local={"stack": 0, "local": 128})
    assert len(tool.check(local, allowed)) == 1, "local memory that grew must fail"
    full = {name: dict(used, stl=0, ldl=0) for name, used in allowed.items()}
    assert json.loads(tool.format_allowlist(full)) == full, "the allow-list must round-trip"
    # Local stores and loads in the code: a spill that resource usage does not report (STACK:0 LOCAL:0) still fails,
    # its count is the largest over the kernel's cubins, predicated forms count, and neither an opcode sharing STL's
    # low byte nor STL's bytes inside an instruction does. The encodings are cuobjdump's for sm_120a; the second cubin
    # numbers its sections the extended way, as the server's device-linked cubin does.
    def instruction(low: int, high: int = 0x000FE80000100A00) -> bytes:
        return struct.pack("<QQ", low, high)

    k_spill = b"".join([
        instruction(0x00000A00FF017624),  # IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]
        instruction(0x0000080401007387),  # STL.64 [R1+0x8], R4
        instruction(0x0000080001020983),  # @P0 LDL.LU R2, [R1+0x8]
        instruction(0x00000C0001037983),  # LDL R3, [R1+0xc]
        instruction(0x000000000000794D, 0x7387000003800000),  # EXIT, with STL's two bytes ending its high word
    ])
    k_none = instruction(0x0000000400007187) + instruction(0x000000000000794D)  # opcode 0x187, EXIT
    with tempfile.TemporaryDirectory() as scratch:
        first, second = Path(scratch, "a.cubin"), Path(scratch, "b.cubin")
        first.write_bytes(cubin({"k_spill": k_spill, "k_none": k_none}, extended=False))
        second.write_bytes(cubin({"k_spill": instruction(0x0000000401007387)}, extended=True))
        accesses = tool.local_accesses([tool.cubin_accesses(first), tool.cubin_accesses(second)])
    assert accesses == {"k_spill": {"stl": 1, "ldl": 2}, "k_none": {"stl": 0, "ldl": 0}}, accesses
    spilled = dict(clean, k_none={"stack": 0, "local": 0, "stl": 2, "ldl": 2})
    assert len(tool.check(spilled, full)) == 2, "local stores and loads in an unlisted kernel must fail"
    assert tool.check(dict(clean, k_spill=dict(full["k_spill"])), full) == [], "listed counts must pass"
    # nvcc's per-file ids (they change with the file and its headers) leave the key; the length
    # prefix of an anonymous namespace may follow template digits, and the next identifier's length
    # may follow the id's digits.
    first = ("__nv_static_41__36334fb8_14_qsa_prepare_cu_eb9e1a4a_9616__ZN8infernix3ops52_GLOBAL__N__"
             "36334fb8_14_qsa_prepare_cu_eb9e1a4a_961618qsa_prepare_kernelILi2EEEvNS1_11PrepareArgsE")
    later = ("__nv_static_42__36334fb8_14_qsa_prepare_cu_eb9e1a4a_23356__ZN8infernix3ops53_GLOBAL__N__"
             "36334fb8_14_qsa_prepare_cu_eb9e1a4a_2335618qsa_prepare_kernelILi2EEEvNS1_11PrepareArgsE")
    assert tool.stable_name(first) == tool.stable_name(later), "per-file ids must not change the key"
    assert tool.stable_name(later) == ("__nv_static__qsa_prepare_cu__ZN8infernix3ops_GLOBAL__N__qsa_prepare_cu"
                                       "18qsa_prepare_kernelILi2EEEvNS1_11PrepareArgsE"), tool.stable_name(later)
    nested = "_ZN8infernix3ops6detail16kernelILi2044_GLOBAL__N__2cce7e46_6_abc_cu_05678a3e_239882EEEvv"
    assert tool.stable_name(nested) == "_ZN8infernix3ops6detail16kernelILi20_GLOBAL__N__abc_cu2EEEvv", \
        tool.stable_name(nested)
    # A file with no id after its hash: the per-build path hash before the file's name leaves the key
    # (two checkouts of the same tree), the file hash after it stays.
    here = ("__nv_static_27__fac0c9fe_6_qsa_cu_940f13ef__ZN8infernix3ops38_GLOBAL__N__fac0c9fe_6_qsa_cu_"
            "940f13ef11pool_kernelEv")
    there = ("__nv_static_27__2f914340_6_qsa_cu_940f13ef__ZN8infernix3ops38_GLOBAL__N__2f914340_6_qsa_cu_"
             "940f13ef11pool_kernelEv")
    assert tool.stable_name(here) == tool.stable_name(there), "the checkout path must not change the key"
    assert tool.stable_name(there) == ("__nv_static__qsa_cu__ZN8infernix3ops38_GLOBAL__N__qsa_cu_940f13ef"
                                       "11pool_kernelEv"), tool.stable_name(there)
    plain = "_Z6kernelPf"
    assert tool.stable_name(plain) == plain, "names without internal linkage stay as they are"
    print("check_cuda_stack checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
