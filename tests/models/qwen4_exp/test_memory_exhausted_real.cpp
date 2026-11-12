// Recovery from a unit out of device memory on a Qwen3.8-Flash-Next artifact (fix sweep FS-01): see
// tests/memory_exhausted_recovery.h for the steps. Exact route, one lane. Skips (77) unless
// INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its volume).
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer infernix_qwen4_exp_memory_exhausted_real_test

#include "engine_host.h"
#include "memory_exhausted_recovery.h"

#include <cstdio>
#include <cstdlib>
#include <exception>

int main() {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    int failures = 0;
    try {
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr && *ngram != '\0') { options.ngram_volume_path = ngram; }
        options.max_context          = 4096;
        options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(4608);
        options.prefill_chunk        = 512;
        options.kv_cache             = infernix::KvCacheStorage::Int8Group64;
        options.max_concurrency      = 1;
        options.max_pending_requests = 2;
        options.fast_paths           = false;
        infernix::Engine engine(options, infernix::test::engine_host());
        failures = infernix::test::memory_exhausted::run_all(engine, 700, 24);
    } catch (const std::exception& error) {
        infernix::runtime::testing::request_memory_fault(0);
        std::printf("  FAIL exception: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "all checks passed\n" : "%d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
