// Qwen3.8-Flash-Next's reproducible route resumed from taps placed off the chunk grid, against a cold prefill
// (exact-route request invariance, docs/maintainer/model-quality.md 6.1; the 2026-10-27 resume-equals-cold
// regression). Several prefill routes change their arithmetic with the call's width, so the route taps and resumes
// only on the prefill-chunk grid, where a resumed prefill runs the cold prefill's calls; every case must then give the
// cold bits. On the 16-token grid used before, every r = 1040 case differed (2026-10-27).
//
// Each case uses its own prompt (no case reuses another's blocks). Request A prefills the first r + 1 tokens with an
// exact tap forced at r (models::qwen4_exp::testing::set_forced_tap), so the cache holds prefill-made state at r;
// request B prefills the first L tokens with prefix reuse and resumes at r; request C prefills them cold. B and C read
// out the log-probability of the prompt's next token; their bits must be equal. The taps sit where a 16-token grid
// would resume; with 512-token calls such a resume would give:
// - r = 1040, L = r + t: B the last t tokens in a t-token call, C a (16 + t)-token call [1024, L);
// - r = 1008, L = 1024 + t: C's last call t tokens, B's (16 + t);
// - r = 1024 (a chunk boundary): both the same calls (control);
// - r = 1552, L = 2000: wide calls on both sides.
// Skips (77) without INFERNIX_QWEN4_ARTIFACT. Arguments: none (all cases) or "r:L" pairs.
//
//   INFERNIX_QWEN4_ARTIFACT=out.infernix [INFERNIX_QWEN4_NGRAM=out.ngram] infernix_qwen4_exp_resume_width_real_test

#include "kv_cache_storage.h"
#include "models/qwen4_exp/program/prefix/test_taps.h"
#include "models/resume_cold_probe.h"
#include "infernix/engine.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

infernix::EngineOptions engine_options(const char* artifact, const char* ngram) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr && *ngram != '\0') { options.ngram_volume_path = ngram; }
    options.max_context                       = 4096;
    options.kv_capacity                       = infernix::KvCapacityPolicy::explicit_capacity(4096);
    const char* kv                            = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache                          = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                              : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency                   = 1;
    options.max_pending_requests              = 1;
    options.prefill_chunk                     = 512;
    options.context_cache.host_capacity_bytes = 1ULL << 30;
    options.fast_paths                        = false; // the reproducible route
    return options;
}

struct Readout {
    float logprob    = 0.0F;
    std::size_t reused = 0;
};

Readout read_next(infernix::Engine& engine, const std::vector<infernix::TokenId>& tokens, std::size_t length, bool reuse) {
    const std::vector<infernix::TokenId> prompt(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(length));
    infernix::RequestOptions options        = infernix::test::resume_probe_request(1, reuse);
    options.execution.readout_tokens        = {tokens[length]};
    const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), options);
    if (!result.readout.has_value() || result.readout->logprobs.size() != 1) {
        throw std::runtime_error("a readout returned no log-probability");
    }
    return {result.readout->logprobs.front(), result.reused_prompt_tokens};
}

const char* const kThemes[] = {
    "lighthouses", "river deltas", "glaciers", "coral reefs", "tide pools", "volcanic islands", "salt marshes",
    "mountain passes", "desert oases", "kelp forests", "river canyons", "alpine meadows", "mangrove coasts",
    "prairie storms", "fjord villages", "limestone caves", "sand dunes", "boreal forests", "arctic tundra",
    "monsoon rains"};

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> cases; // (r, L)
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto colon      = arg.find(':');
            if (colon == std::string::npos) { throw std::invalid_argument("cases are r:L"); }
            cases.emplace_back(static_cast<std::uint32_t>(std::stoul(arg.substr(0, colon))),
                               static_cast<std::uint32_t>(std::stoul(arg.substr(colon + 1))));
        }
    } else {
        for (const std::uint32_t t : {1U, 5U, 8U, 9U, 12U, 15U, 16U, 17U, 23U, 32U, 100U}) { cases.emplace_back(1040U, 1040U + t); }
        for (const std::uint32_t t : {1U, 5U, 8U, 12U}) { cases.emplace_back(1008U, 1024U + t); }
        cases.emplace_back(1024U, 1029U);
        cases.emplace_back(1024U, 1047U);
        cases.emplace_back(1552U, 2000U);
    }
    if (cases.size() > std::size(kThemes)) { throw std::invalid_argument("more cases than prompts"); }
    // The 1 GiB Host tier fills after about eight cases; GDSF then keeps the earlier cases' hit snapshots over a new
    // case's (hybrid-prefix-cache-spec.md 9.3), so a fresh Engine every six cases.
    constexpr std::size_t kCasesPerEngine = 6;
    try {
        std::optional<infernix::Engine> holder;
        int failures = 0, tested = 0;
        for (std::size_t c = 0; c < cases.size(); ++c) {
            if (c % kCasesPerEngine == 0) {
                holder.reset();
                holder.emplace(engine_options(artifact, std::getenv("INFERNIX_QWEN4_NGRAM")));
            }
            infernix::Engine& engine = *holder;
            const auto [r, L] = cases[c];
            const auto tokens = engine.tokenize_text(infernix::test::resume_probe_text(kThemes[c], 90));
            if (tokens.size() <= L + 1 || r >= L) { throw std::runtime_error("a case's prompt is too short or its resume past its end"); }
            infernix::models::qwen4_exp::testing::set_forced_tap(r);
            {
                const std::vector<infernix::TokenId> head(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(r + 1));
                (void)engine.generate(engine.prepare_tokens(head), infernix::test::resume_probe_request(1, true));
            }
            infernix::models::qwen4_exp::testing::set_forced_tap(std::nullopt);
            const Readout resumed = read_next(engine, tokens, L, true);
            const Readout cold    = read_next(engine, tokens, L, false);
            const bool equal      = std::bit_cast<std::uint32_t>(resumed.logprob) == std::bit_cast<std::uint32_t>(cold.logprob);
            const std::uint32_t cold_last = L - (L % 512 == 0 ? 512 : L % 512);
            std::printf("r %4u L %4u: resumed call [%zu, %u) (%u tokens), cold call [%u, %u) (%u tokens): %s, |dlogprob| %.6g\n",
                        r, L, resumed.reused, L, L - static_cast<std::uint32_t>(resumed.reused), cold_last, L, L - cold_last,
                        equal ? "bit-identical" : "DIFFERS",
                        std::fabs(static_cast<double>(resumed.logprob) - static_cast<double>(cold.logprob)));
            // The reproducible route resumes on the prefill-chunk grid (the forced tap moves to the grid at or
            // before it).
            if (resumed.reused == 0) {
                std::printf("  (nothing reused: not counted)\n");
                continue;
            }
            ++tested;
            if (!equal) { ++failures; }
        }
        if (tested == 0) { throw std::runtime_error("no case resumed at its forced tap"); }
        std::printf(failures == 0 ? "resumes off the chunk grid equal cold prefills (%d cases)\n"
                                  : "FAIL: %d of %d resumes differ from their cold prefill\n",
                    failures == 0 ? tested : failures, tested);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
