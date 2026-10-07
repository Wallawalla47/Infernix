// Engine-level VRAM pressure on the real artifact (memory track RT13, design §19.3.7). A fake VRAM
// source reports the real card less a "taken" amount, which the test raises by 2 GiB (another
// program) and lowers again:
//   1. during a greedy generation: the expert cache shrinks at the next round boundary and grows
//      back after the (shortened) grow delay, and the token ids equal an undisturbed run;
//   2. while the engine is idle: the monitor wakes the engine, which shrinks and grows the cache
//      outside any round (maintain()).
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer [INFERNIX_QWEN4_NGRAM=out.ninfer.ngram] infernix_qwen4_exp_vram_pressure_real_test

#include "core/vram_budget.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "infernix/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint64_t kGiB = 1ULL << 30;

struct Diagnostics {
    std::mutex mutex;
    std::vector<std::string> lines;

    void add(const std::string& text) {
        const std::lock_guard<std::mutex> lock(mutex);
        lines.push_back(text);
        std::printf("  [engine] %s\n", text.c_str());
    }
    // (from, to) frame counts of every resize line since `first`.
    std::vector<std::pair<int, int>> resizes(std::size_t first) {
        const std::lock_guard<std::mutex> lock(mutex);
        static const std::regex pattern(R"(expert cache (\d+) -> (\d+) frames)");
        std::vector<std::pair<int, int>> out;
        for (std::size_t i = first; i < lines.size(); ++i) {
            std::smatch m;
            if (std::regex_search(lines[i], m, pattern)) { out.emplace_back(std::stoi(m[1]), std::stoi(m[2])); }
        }
        return out;
    }
    std::size_t size() {
        const std::lock_guard<std::mutex> lock(mutex);
        return lines.size();
    }
};

infernix::PromptInput story() {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back({.kind = infernix::MessagePartKind::Text,
                             .text = "Write a long story about a lighthouse keeper who befriends a seagull."});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    try {
        // The real source first, then the fake that every engine source becomes.
        auto real = infernix::open_vram_budget_source(0);
        std::atomic<std::uint64_t> taken{0};
        infernix::testing::set_vram_budget_source([&] {
            infernix::VramSnapshot s = real->query();
            const std::uint64_t t  = taken.load();
            s.device_free          = s.device_free > t ? s.device_free - t : 0;
            if (s.has_budget) { s.local_budget = s.local_budget > t ? s.local_budget - t : 0; }
            return s;
        });
        infernix::models::qwen4_exp::testing::set_vram_grow_delay(1.0);

        Diagnostics diagnostics;
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        options.context_cache.enabled = false; // no prefix cache (Qwen4Exp has no Legacy cache)
        if (ngram != nullptr) { options.ngram_volume_path = ngram; }
        options.max_context = 4096;
        options.kv_capacity = infernix::KvCapacityPolicy::explicit_capacity(4096);
        options.kv_cache    = infernix::KvCacheStorage::Int8Group64;
        options.diagnostic_observer.callback = [&](const infernix::Diagnostic& d) { diagnostics.add(d.message); };
        infernix::Engine engine(options);

        constexpr std::uint32_t kTokens = 700;
        const auto baseline = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
        check(baseline.size() == kTokens, "baseline generated " + std::to_string(baseline.size()) + " tokens");

        // 1. Pressure during a generation: another program takes 2 GiB at 1.5 s and frees it at 4 s.
        const std::size_t mark = diagnostics.size();
        std::thread squeeze([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            taken = 2 * kGiB;
            std::this_thread::sleep_for(std::chrono::milliseconds(2500));
            taken = 0;
        });
        const auto pressured = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
        squeeze.join();
        check(pressured == baseline, "ids under pressure equal the undisturbed run");
        const auto during = diagnostics.resizes(mark);
        bool shrank = false, regrew = false;
        for (const auto& [from, to] : during) {
            shrank |= to < from;
            regrew |= shrank && to > from;
        }
        check(shrank, "the cache shrank during the generation (" + std::to_string(during.size()) + " resizes)");
        check(regrew, "the cache grew back after the memory was freed");

        // 2. Pressure while idle: the monitor wakes the engine, which resizes outside any round.
        std::this_thread::sleep_for(std::chrono::milliseconds(3000)); // finish any regrowth
        const std::size_t idle_mark = diagnostics.size();
        taken = 2 * kGiB;
        std::this_thread::sleep_for(std::chrono::milliseconds(3000));
        const auto idle_shrink = diagnostics.resizes(idle_mark);
        check(!idle_shrink.empty() && idle_shrink.front().second < idle_shrink.front().first,
              "an idle engine shrinks the cache");
        const std::size_t grow_mark = diagnostics.size();
        taken = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(4000));
        const auto idle_grow = diagnostics.resizes(grow_mark);
        check(!idle_grow.empty() && idle_grow.back().second > idle_grow.back().first,
              "an idle engine grows the cache back");
        const auto after = engine.generate(engine.prepare(story()), greedy(kTokens)).generated_token_ids;
        check(after == baseline, "ids after the idle resizes equal the undisturbed run");
        infernix::testing::set_vram_budget_source({});
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "qwen4_exp VRAM pressure checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
