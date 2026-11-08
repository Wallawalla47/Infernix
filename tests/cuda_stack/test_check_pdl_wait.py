"""check_pdl_wait: a load before a kernel's PDL wait is traced to the parameter its address comes from (through
index arithmetic, LEA, a constant-bank addend and an address read from memory), a kernel without a wait is not
listed, an origin missing from the allow-list fails, and the written allow-list reads back unchanged."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "cuda_stack"))

import check_pdl_wait as tool  # noqa: E402


def listing(functions: dict[str, list[str]]) -> list[str]:
    """A `cuobjdump -sass` listing of the functions, each instruction followed by its encoding's second line."""
    lines = ["code for sm_120a"]
    for name, instructions in functions.items():
        lines.append(f"\t\tFunction : {name}")
        for index, text in enumerate(instructions):
            lines.append(f"        /*{index * 16:04x}*/                   {text} ;   /* 0x000000000000782e */")
            lines.append("                                                                   /* 0x000fe20000000800 */")
    return lines


def main() -> int:
    kernels = tool.parse_listing(listing({
        # The hoist 0e1c221d fixed: a parameter's non-coherent load above the wait (parameters start at 0x380).
        "hoisted": ["LDC.64 R4, c[0x0][0x398]", "LDG.E.CONSTANT R18, desc[UR40][R4.64]", "ACQBULK",
                    "LDG.E R2, desc[UR40][R6.64]"],
        # Index arithmetic: IMAD.WIDE's addend is the pointer; the index's own constant is not an origin.
        "indexed": ["S2R R0, SR_TID.X", "LDC R7, c[0x0][0x3b0]", "IMAD R0, R0, R7, RZ", "LDC.64 R2, c[0x0][0x380]",
                    "IMAD.WIDE R2, R0, 0x4, R2", "LDG.E R5, desc[UR4][R2.64+0x10]", "ACQBULK"],
        # LEA's base, through a cp.async whose first operand is its shared-memory destination.
        "copied": ["LDC.64 R10, c[0x0][0x3a0]", "S2R R0, SR_TID.X", "LEA R8, P0, R0, R10, 0x2",
                   "LEA.HI.X R9, R0, R11, RZ, 0x2, P0", "LDGSTS.E.BYPASS.128 [R3], desc[UR4][R8.64]", "LDGDEPBAR",
                   "ACQBULK"],
        # A constant-bank addend and an address read from memory (a record table).
        "addend": ["S2R R0, SR_CTAID.X", "IMAD.WIDE R2, R0, 0x8, c[0x0][0x390]", "LDG.E.64 R4, desc[UR4][R2.64]",
                   "@P0 LDG.E R6, desc[UR4][R4.64]", "ACQBULK"],
        # An address the trace cannot follow, and a uniform-datapath parameter.
        "untraced": ["S2R R2, SR_TID.X", "MOV R3, RZ", "LDG.E R4, desc[UR4][R2.64]", "LDCU.64 UR6, c[0x0][0x388]",
                     "LDG.E R5, desc[UR4][UR6.64]", "ACQBULK"],
        "no_loads": ["LDC.64 R2, c[0x0][0x380]", "ACQBULK", "LDG.E R4, desc[UR4][R2.64]"],
        "no_wait": ["LDC.64 R2, c[0x0][0x380]", "LDG.E R4, desc[UR4][R2.64]", "EXIT"],
    }))
    expected = {"hoisted": {"p+0x18"}, "indexed": {"p+0x0"}, "copied": {"p+0x20"}, "addend": {"p+0x10", "*p+0x10"},
                "untraced": {"?", "p+0x8"}, "no_loads": set()}
    assert kernels == expected, f"traced origins: {kernels}"

    found = {name: sorted(origins) for name, origins in kernels.items()}
    allowed = {"hoisted": ["p+0x18"], "indexed": ["p+0x0"], "copied": ["p+0x20"], "addend": ["*p+0x10", "p+0x10"],
               "untraced": ["?", "p+0x8"]}
    assert tool.check(found, allowed) == [], "reviewed origins must pass"
    assert len(tool.check(dict(found, hoisted=["p+0x10", "p+0x18"]), allowed)) == 1, "a new origin must fail"
    assert len(tool.check(dict(found, fresh=["p+0x0"]), allowed)) == 1, "an unlisted kernel's load must fail"
    assert tool.check(dict(found, fresh=[]), allowed) == [], "an unlisted kernel without loads before its wait passes"
    assert json.loads(tool.format_allowlist(allowed)) == allowed, "the allow-list must round-trip"
    # Kernels are keyed by stable names: nvcc's per-file id in an internal-linkage name does not reach the key.
    static = "__nv_static_46__6fa03c6b_24_ggml_encode_producers_cu_9cd89126__ZN8infernix3ops4kernEv"
    assert list(tool.parse_listing(listing({static: ["ACQBULK"]}))) == [tool.stable_name(static)]
    print("check_pdl_wait tool checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
