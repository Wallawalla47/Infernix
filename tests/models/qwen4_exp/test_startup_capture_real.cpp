// The startup capture of Qwen3.8-Flash-Next's CUDA graph set (candidates §23 NU-04; design §19.3.37) against
// capture at first use, through the public Engine on the real artifact.
// Per drafter (arguments plain, mtp, dflash2), two Engines run the same two prompts, one capturing each graph at
// its first use and one capturing the set at startup (models::qwen4_exp::testing::set_startup_capture):
// - the reproducible route (fast paths off), the prompts one at a time and then both together: the generated ids
//   must be equal, since a startup-captured graph holds the same kernels over the same addresses;
// - the default route, the same runs (two rows together take pair rounds there): every graph the rounds launched
//   must come from the startup set (no "captured at its first use" warning). Its ids are reported, not required
//   equal: the drafter pool's swaps and pair rounds make its drafts, and so its verified widths, depend on timing.
// Each Engine's construction time, the startup line and free VRAM after construction are printed (the set's cost).
// Skips (77) without INFERNIX_QWEN4_ARTIFACT; dflash2 skips without INFERNIX_QWEN4_DFLASH2.
//
//   INFERNIX_QWEN4_ARTIFACT=out.infernix [INFERNIX_QWEN4_NGRAM=out.ngram] [INFERNIX_QWEN4_DFLASH2=drafter.infernix]
//   [INFERNIX_QWEN4_KV=int8] infernix_qwen4_exp_startup_capture_real_test plain mtp dflash2

#include "kv_cache_storage.h"
#include "models/qwen4_exp/program/graph_set.h"
#include "infernix/engine.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Setup {
    const char* artifact = nullptr;
    const char* ngram    = nullptr;
    const char* drafter  = nullptr;
};

// A code prompt whose answer repeats its own structure (n-gram copies reach the widest verifications) and a prose
// prompt (short drafts).
const char* const kPrompts[] = {
    "Write a C++ function `int classify(int code)` that returns 1 for each of the codes 100, 101, 102, 103, 104, 105, "
    "106, 107, 108 and 109 and 0 otherwise, as a switch statement with one case line per code. Then write the same "
    "function again for the codes 200 to 209.",
    "In two short paragraphs, explain why lighthouses were built on headlands and how their lights were kept burning "
    "before electricity.",
};
constexpr std::uint32_t kMaxTokens = 160;

infernix::EngineOptions engine_options(const Setup& setup, std::string_view drafter, bool fast_paths) {
    infernix::EngineOptions options;
    options.artifact_path = setup.artifact;
    if (setup.ngram != nullptr) { options.ngram_volume_path = setup.ngram; }
    options.max_context           = 4096;
    options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(4096);
    const char* kv                = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache              = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                  : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency       = 2;
    options.max_pending_requests  = 2;
    options.prefill_chunk         = 512;
    options.context_cache.enabled = false; // every run prefills its whole prompt
    options.fast_paths            = fast_paths;
    if (drafter == "mtp") {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    } else if (drafter == "dflash2") {
        options.speculative.backend      = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        options.speculative.draft_model  = setup.drafter;
    } else if (drafter != "plain") {
        throw std::invalid_argument("the drafter must be plain, mtp or dflash2");
    }
    return options;
}

infernix::PromptInput prompt_input(const char* text) {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text, .text = text});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions request_options() {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = kMaxTokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

std::size_t free_vram() {
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) { return 0; }
    return free_bytes;
}

struct Arm {
    // The prompts one at a time, then both submitted together.
    std::vector<std::vector<infernix::TokenId>> alone, together;
    std::uint32_t late_captures = 0;
    std::string startup_line;
};

Arm run_arm(const Setup& setup, std::string_view drafter, bool fast_paths, bool startup_capture) {
    infernix::models::qwen4_exp::testing::set_startup_capture(startup_capture);
    Arm arm;
    std::mutex mutex;
    auto options = engine_options(setup, drafter, fast_paths);
    options.diagnostic_observer.callback = [&](const infernix::Diagnostic& d) {
        const std::lock_guard<std::mutex> lock(mutex);
        if (d.message.find("captured at its first use") != std::string::npos) {
            ++arm.late_captures;
            std::printf("  %s\n", d.message.c_str());
        }
        if (d.message.rfind("CUDA graphs: captured", 0) == 0) { arm.startup_line = d.message; }
    };
    const char* label = startup_capture ? "startup capture" : "first-use capture";
    const auto began  = std::chrono::steady_clock::now();
    {
        infernix::Engine engine(options);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        std::printf("%s %s, %s: Engine constructed in %.2f s, %zu MiB VRAM free%s%s\n", std::string(drafter).c_str(),
                    fast_paths ? "default route" : "reproducible route", label, seconds, free_vram() >> 20,
                    arm.startup_line.empty() ? "" : "; ", arm.startup_line.c_str());
        for (const char* text : kPrompts) {
            const auto t0 = std::chrono::steady_clock::now();
            arm.alone.push_back(engine.generate(engine.prepare(prompt_input(text)), request_options()).generated_token_ids);
            std::printf("  alone: %zu tokens in %.2f s\n", arm.alone.back().size(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        std::vector<infernix::GenerationHandle> handles;
        for (const char* text : kPrompts) { handles.push_back(engine.submit(engine.prepare(prompt_input(text)), request_options())); }
        for (auto& handle : handles) { arm.together.push_back(handle.wait().generated_token_ids); }
    }
    infernix::models::qwen4_exp::testing::set_startup_capture(std::nullopt);
    std::fflush(stdout);
    return arm;
}

std::size_t first_difference(const std::vector<infernix::TokenId>& a, const std::vector<infernix::TokenId>& b) {
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) { ++i; }
    return i;
}

int run(const Setup& setup, std::string_view drafter, bool fast_paths) {
    const std::string name = std::string(drafter) + (fast_paths ? " default route" : " reproducible route");
    const Arm first   = run_arm(setup, drafter, fast_paths, false);
    const Arm startup = run_arm(setup, drafter, fast_paths, true);
    int failures      = 0;
    const auto check  = [&](bool condition, const std::string& what) {
        if (!condition) {
            std::fprintf(stderr, "FAIL %s: %s\n", name.c_str(), what.c_str());
            ++failures;
        }
    };
    check(!startup.startup_line.empty(), "the startup-capture Engine logged no startup capture");
    check(first.startup_line.empty(), "the first-use Engine captured at startup");
    check(startup.late_captures == 0,
          std::to_string(startup.late_captures) + " graphs were captured at first use after the startup capture");
    for (const bool together : {false, true}) {
        const auto& a = together ? first.together : first.alone;
        const auto& b = together ? startup.together : startup.alone;
        for (std::size_t p = 0; p < a.size() && p < b.size(); ++p) {
            const bool equal = a[p] == b[p];
            const std::string what = std::string(together ? "together" : "alone") + " prompt " + std::to_string(p) +
                                     ": " + std::to_string(a[p].size()) + " / " + std::to_string(b[p].size()) +
                                     " tokens, first difference at " + std::to_string(first_difference(a[p], b[p]));
            if (fast_paths) {
                std::printf("  %s ids %s (%s; not required on the default route)\n", name.c_str(),
                            equal ? "equal" : "differ", what.c_str());
            } else {
                check(equal, "ids differ, " + what);
            }
        }
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
        std::printf(failures == 0 ? "startup capture against first-use capture passed\n" : "FAIL: %d checks failed\n",
                    failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
