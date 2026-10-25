// A continuation resumed from the prefix cache after speculative decoding against a cold run, on Qwen3.8-Flash-Next
// (candidates §15 PR6-02; tests/models/resume_cold_probe.h). INFERNIX_QWEN4_ARTIFACT names the artifact,
// INFERNIX_QWEN4_NGRAM its n-gram volume, INFERNIX_QWEN4_DFLASH2 the DFlash2 companion (dflash2 skips without it),
// INFERNIX_QWEN4_KV the KV storage (default int8). Per drafter (arguments mtp, dflash2):
// - the reproducible route (fast paths off): the resumed continuation equals the cold one, tokens and
//   teacher-forced log-probability bits;
// - the default route resumes from the endpoint the speculative request's decode left (the terminal frontier;
//   2026-10-25 on Dense8: 947 of 949 prompt tokens): the teacher-forced log-probabilities of the 16 tokens past it
//   stay within max(0.1, 4 x) of a resume from a prefill-made state in the same Engine.
//
//   infernix_qwen4_exp_resume_cold_real_test mtp|dflash2...

#include "kv_cache_storage.h"
#include "models/resume_cold_probe.h"
#include "infernix/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Setup {
    const char* artifact = nullptr;
    const char* ngram    = nullptr;
    const char* drafter  = nullptr;
};

infernix::EngineOptions engine_options(const Setup& setup, std::string_view drafter, bool fast_paths) {
    infernix::EngineOptions options;
    options.artifact_path = setup.artifact;
    if (setup.ngram != nullptr) { options.ngram_volume_path = setup.ngram; }
    options.max_context                       = 4096;
    options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(4096);
    const char* kv                            = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache                          = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                              : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency                   = 1;
    options.max_pending_requests              = 1;
    options.prefill_chunk                     = 512;
    options.context_cache.host_capacity_bytes = 1ULL << 30;
    options.fast_paths                        = fast_paths;
    if (drafter == "mtp") {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    } else if (drafter == "dflash2") {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        options.speculative.draft_model  = setup.drafter;
    } else {
        throw std::invalid_argument("the drafter must be mtp or dflash2");
    }
    return options;
}

int run(const Setup& setup, std::string_view drafter, bool fast_paths) {
    infernix::Engine engine(engine_options(setup, drafter, fast_paths));
    const std::string route = fast_paths ? "default" : "reproducible";
    const auto separator    = engine.tokenize_text("\n\n");
    if (separator.empty()) { throw std::runtime_error("the tokenizer encoded no separator"); }
    const auto start       = engine.tokenize_text(infernix::test::resume_probe_text("lighthouses", 24));
    const auto calibration = engine.tokenize_text(infernix::test::resume_probe_text("river deltas", 24));
    const auto probe       = infernix::test::resume_probe(engine, start, 256, separator.front());
    const auto prefill     = infernix::test::resume_probe(engine, calibration, 0, separator.front());
    std::printf("%s %s: start %zu, continuation %zu, reused %zu; mean |dlogprob| %.6f (prefill-made state: %.6f, "
                "reused %zu of %zu)\n",
                std::string(drafter).c_str(), route.c_str(), probe.start, probe.continuation, probe.reused,
                probe.mean_abs_delta(), prefill.mean_abs_delta(), prefill.reused, prefill.continuation);
    int failures     = 0;
    const auto check = [&](bool condition, const std::string& what) {
        if (!condition) {
            std::fprintf(stderr, "FAIL %s %s: %s\n", std::string(drafter).c_str(), route.c_str(), what.c_str());
            ++failures;
        }
    };
    check(probe.reused > 0, "the continuation reused nothing");
    if (!fast_paths) {
        check(probe.bit_identical(), "the resumed continuation differs from the cold one");
        check(prefill.bit_identical(), "the continuation resumed from a prefill-made state differs from the cold one");
    } else {
        check(probe.reused > probe.start, "the continuation did not resume from the decode-computed endpoint (reused " +
                                              std::to_string(probe.reused) + " of " + std::to_string(probe.start) +
                                              " prompt tokens before the decode)");
        const double bound = std::max(0.1, 4.0 * prefill.mean_abs_delta());
        check(probe.mean_abs_delta() <= bound,
              "teacher-forced log-probabilities past the decoded text moved by " +
                  std::to_string(probe.mean_abs_delta()) + " on average (bound " + std::to_string(bound) + ")");
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Setup setup;
    setup.artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    setup.ngram    = std::getenv("INFERNIX_QWEN4_NGRAM");
    setup.drafter  = std::getenv("INFERNIX_QWEN4_DFLASH2");
    if (setup.artifact == nullptr || *setup.artifact == '\0') {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    try {
        int failures = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view drafter = argv[i];
            if (drafter == "dflash2" && (setup.drafter == nullptr || *setup.drafter == '\0')) {
                std::printf("skip dflash2: INFERNIX_QWEN4_DFLASH2 is not set\n");
                continue;
            }
            for (const bool fast_paths : {false, true}) { failures += run(setup, drafter, fast_paths); }
        }
        std::printf(failures == 0 ? "resume against cold passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
