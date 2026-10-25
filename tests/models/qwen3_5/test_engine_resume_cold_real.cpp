#include "infernix/engine.h"
#include "models/resume_cold_probe.h"

#include <algorithm>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// A continuation resumed from the prefix cache against a cold run, after speculative decoding (candidates §15
// PR6-02; tests/models/resume_cold_probe.h). INFERNIX_TEST_ARTIFACT names a Qwen3.5 artifact holding the drafters
// the arguments name (mtp, dflash2). Per drafter:
// - the reproducible route (fast paths off) resumes on the prefill chunk grid, and its output does not depend on
//   the prefix cache: the resumed continuation equals the cold one, tokens and teacher-forced log-probability bits;
// - the default route (free-cache-resume) resumes from the decode-computed endpoint the speculative request left:
//   the teacher-forced log-probabilities of the 16 tokens past it stay within max(0.1, 4 x) of a resume from a
//   prefill-made state in the same Engine (the stale-state failure measured 3.99).
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

infernix::EngineOptions engine_options(const std::filesystem::path& artifact, std::string_view drafter,
                                       bool fast_paths) {
    infernix::EngineOptions options;
    options.artifact_path   = artifact;
    options.fast_paths      = fast_paths;
    options.max_context     = 4096;
    options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency = 1;
    options.prefill_chunk   = 512;
    options.enable_vision   = false;
    if (drafter == "mtp") {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    } else if (drafter == "dflash2") {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
    } else {
        throw std::invalid_argument("the drafter must be mtp or dflash2");
    }
    return options;
}

int run(const std::filesystem::path& artifact, std::string_view drafter, bool fast_paths) {
    infernix::Engine engine(engine_options(artifact, drafter, fast_paths));
    const std::string route = fast_paths ? "default" : "reproducible";
    const auto separator    = engine.tokenize_text("\n\n");
    require(!separator.empty(), "the tokenizer encoded no separator");
    // About 700 prompt tokens: the reproducible route's grid point at 512 lies inside it.
    const auto start       = engine.tokenize_text(infernix::test::resume_probe_text("lighthouses", 24));
    const auto calibration = engine.tokenize_text(infernix::test::resume_probe_text("river deltas", 24));
    const auto probe       = infernix::test::resume_probe(engine, start, 256, separator.front());
    const auto prefill     = infernix::test::resume_probe(engine, calibration, 0, separator.front());
    std::printf("%s %s: start %zu, continuation %zu, reused %zu; mean |dlogprob| %.6f (prefill-made state: %.6f, "
                "reused %zu of %zu)\n",
                std::string(drafter).c_str(), route.c_str(), probe.start, probe.continuation, probe.reused,
                probe.mean_abs_delta(), prefill.mean_abs_delta(), prefill.reused, prefill.continuation);
    int failures = 0;
    const auto check = [&](bool condition, const std::string& what) {
        if (!condition) {
            std::cerr << "FAIL " << drafter << ' ' << route << ": " << what << '\n';
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
              "teacher-forced log-probabilities past the decode-computed endpoint moved by " +
                  std::to_string(probe.mean_abs_delta()) + " on average (bound " + std::to_string(bound) + ")");
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: set INFERNIX_TEST_ARTIFACT to an explicit Qwen3.5 artifact\n";
        return 77;
    }
    try {
        require(argc >= 2, "usage: infernix_qwen3_5_resume_cold_real_test mtp|dflash2...");
        int failures = 0;
        for (int i = 1; i < argc; ++i) {
            for (const bool fast_paths : {false, true}) { failures += run(artifact, argv[i], fast_paths); }
        }
        if (failures != 0) {
            std::cerr << failures << " resume check(s) failed\n";
            return 1;
        }
        std::cout << "resume against cold passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
