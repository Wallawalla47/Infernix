"""check_cuda_stack: growth past the allow-list fails, shrinkage and listed use pass, an unlisted
kernel may use no stack or local memory, and the written allow-list reads back unchanged."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "cuda_stack"))

import check_cuda_stack as tool  # noqa: E402


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
    assert json.loads(tool.format_allowlist(allowed)) == allowed, "the allow-list must round-trip"
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
