// Recovery from a unit out of device memory on the Qwen3.5 dense artifact (fix sweep FS-01): see
// tests/memory_exhausted_recovery.h for the steps. Exact route, one lane, the artifact's own drafter.
// Skips (77) unless INFERNIX_TEST_ARTIFACT names the artifact (real_test_artifact.h).
//
//   INFERNIX_TEST_ARTIFACT=out.infernix infernix_qwen3_5_memory_exhausted_real_test

#include "memory_exhausted_recovery.h"
#include "real_test_artifact.h"

#include <cstdio>
#include <exception>

int main() {
    const char* artifact = infernix::test::real_test_artifact(infernix::test::kDenseArtifactVariable);
    if (artifact == nullptr) { return 77; }
    int failures = 0;
    try {
        infernix::EngineOptions options;
        options.artifact_path        = artifact;
        options.max_context          = 4096;
        options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(4608);
        options.prefill_chunk        = 512;
        options.max_concurrency      = 1;
        options.max_pending_requests = 2;
        options.fast_paths           = false;
        infernix::Engine engine(options);
        failures = infernix::test::memory_exhausted::run_all(engine, 700, 24);
    } catch (const std::exception& error) {
        infernix::runtime::testing::request_memory_fault(0);
        std::printf("  FAIL exception: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
