#pragma once

// The Qwen3.5 real tests' artifacts, one variable per kind of artifact, so that one CTest run covers
// every real test:
//
//   INFERNIX_TEST_ARTIFACT            a dense artifact holding MTP and DFlash2 (the NVFP4 27B)
//   INFERNIX_TEST_MOE_ARTIFACT        a MoE artifact holding DFlash (the 35B-A3B): the DFlash and MoE tests
//   INFERNIX_TEST_OFFLOADED_ARTIFACT  a MoE artifact whose experts are offloaded (the 122B-A10B): the MoE
//                                     test's offloaded entry
//
// A test skips (exit 77) when its variable is unset or empty.

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace infernix::test {

// The Host tier a generating Engine of a test that does not size it starts with: 1 GiB, page-locked at the start. The
// automatic tier (the Engine's default) page-locks most of the free RAM at every start and releases it at the end,
// about 6 s of each 27B Engine (a dflash2 tree entry took 13.1-14.2 s with it and 7.3-7.4 s without, 2026-11-19);
// a tier of this size keeps the snapshot spills and restores the tests rely on. Tests of the tier's own sizing, and
// the offloaded MoE's RAM layouts, keep their own settings.
inline constexpr std::size_t kTestHostTierBytes = std::size_t{1} << 30;

inline constexpr const char* kDenseArtifactVariable     = "INFERNIX_TEST_ARTIFACT";
inline constexpr const char* kMoeArtifactVariable       = "INFERNIX_TEST_MOE_ARTIFACT";
inline constexpr const char* kOffloadedArtifactVariable = "INFERNIX_TEST_OFFLOADED_ARTIFACT";

// The artifact `variable` names, or nullptr (after printing the skip reason) when it names none.
inline const char* real_test_artifact(const char* variable) {
    const char* value = std::getenv(variable);
    if (value == nullptr || *value == '\0') {
        std::printf("skip: %s is not set\n", variable);
        return nullptr;
    }
    return value;
}

} // namespace infernix::test
