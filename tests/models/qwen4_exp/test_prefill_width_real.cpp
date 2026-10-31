// Qwen3.8-Flash-Next's reproducible route: whether a prompt's prefill keeps its bits when its wide calls change
// width over the same call grid. That is the condition for fitting the route's prefill chunk to the expert cache at
// each call by merging whole grid calls (docs/maintainer/qwen3_8-flash-next-design.md, the prefill chunk fit): its
// fixed chunk is the one the cache lends with every context full, and merging would keep each request's last call.
//
// One Engine per chunk (prefix reuse off, one lane); prompts of m * G + r tokens, G the widest chunk and r below the
// narrowest, so every chunk dividing G ends the prompt with the same call [L - r, L) and splits the tokens before it
// into calls of whole chunks (r = 0: the last call is one chunk of each width). Each prompt reads out the
// log-probabilities of 256 next tokens (the prompt's own next token and 255 others) and the top token; every
// chunk's readouts must equal the first chunk's bit for bit. A chunk the Engine cut at startup (its warning) fails
// the run: the cases would not compare the planned grids.
// Skips (77) without INFERNIX_QWEN4_ARTIFACT. INFERNIX_QWEN4_KV selects the KV format (default int8); arguments are
// the chunks, the widest a multiple of the others (default 1024 3072 6144).
//
//   INFERNIX_QWEN4_ARTIFACT=out.infernix [INFERNIX_QWEN4_NGRAM=out.ngram] infernix_qwen4_exp_prefill_width_real_test

#include "kv_cache_storage.h"
#include "models/resume_cold_probe.h"
#include "infernix/engine.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext = 26624;
constexpr std::uint32_t kReadouts   = infernix::kMaximumReadoutTokens;
const std::uint32_t kRemainders[]   = {0, 1, 100, 700};
const std::uint32_t kMultiples[]    = {1, 2, 4};

struct Readout {
    std::vector<float> logprobs;
    infernix::TokenId top = 0;
    float top_logprob     = 0.0F;
};

infernix::EngineOptions engine_options(const char* artifact, const char* ngram, std::uint32_t chunk,
                                       std::vector<std::string>& cuts, std::mutex& mutex) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr && *ngram != '\0') { options.ngram_volume_path = ngram; }
    options.max_context          = kMaxContext;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(kMaxContext);
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv) : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    options.prefill_chunk        = chunk;
    options.fast_paths           = false; // the reproducible route
    options.diagnostic_observer.callback = [&cuts, &mutex](const infernix::Diagnostic& d) {
        if (d.level == infernix::DiagnosticLevel::Warning && d.message.find("prefill chunk") != std::string::npos) {
            const std::lock_guard<std::mutex> lock(mutex);
            cuts.push_back(d.message);
        }
    };
    return options;
}

// The readout tokens of a prompt of `length` tokens: its next token, then distinct tokens of the prompt, then ids
// counting up, `kReadouts` in all.
std::vector<infernix::TokenId> readout_tokens(const std::vector<infernix::TokenId>& tokens, std::size_t length) {
    std::vector<infernix::TokenId> out{tokens[length]};
    for (std::size_t i = 0; i < length && out.size() < kReadouts / 2; ++i) {
        if (std::find(out.begin(), out.end(), tokens[i]) == out.end()) { out.push_back(tokens[i]); }
    }
    for (infernix::TokenId id = 0; out.size() < kReadouts; ++id) {
        if (std::find(out.begin(), out.end(), id) == out.end()) { out.push_back(id); }
    }
    return out;
}

Readout read(infernix::Engine& engine, const std::vector<infernix::TokenId>& tokens, std::size_t length) {
    const std::vector<infernix::TokenId> prompt(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(length));
    infernix::RequestOptions options        = infernix::test::resume_probe_request(1, false);
    options.execution.readout_tokens        = readout_tokens(tokens, length);
    const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), options);
    if (!result.readout.has_value() || result.readout->logprobs.size() != kReadouts) {
        throw std::runtime_error("a readout returned no log-probabilities");
    }
    return {result.readout->logprobs, result.readout->top_token, result.readout->top_logprob};
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    std::vector<std::uint32_t> chunks;
    for (int i = 1; i < argc; ++i) { chunks.push_back(static_cast<std::uint32_t>(std::stoul(argv[i]))); }
    if (chunks.empty()) { chunks = {1024, 3072, 6144}; }
    const std::uint32_t grid = *std::max_element(chunks.begin(), chunks.end());
    for (const std::uint32_t chunk : chunks) {
        if (chunk % 128 != 0 || grid % chunk != 0 || chunk <= kRemainders[std::size(kRemainders) - 1]) {
            std::fprintf(stderr, "FAIL: chunk %u must be a multiple of 128 dividing %u and above every remainder\n", chunk, grid);
            return 1;
        }
    }
    try {
        std::vector<std::size_t> lengths;
        for (const std::uint32_t m : kMultiples) {
            for (const std::uint32_t r : kRemainders) {
                if (static_cast<std::size_t>(m) * grid + r + 1 < kMaxContext) { lengths.push_back(static_cast<std::size_t>(m) * grid + r); }
            }
        }
        std::vector<infernix::TokenId> tokens;
        std::vector<std::vector<Readout>> readouts; // [chunk][length]
        int failures = 0;
        for (const std::uint32_t chunk : chunks) {
            std::vector<std::string> cuts;
            std::mutex mutex;
            infernix::Engine engine(engine_options(artifact, std::getenv("INFERNIX_QWEN4_NGRAM"), chunk, cuts, mutex));
            if (!cuts.empty()) {
                std::fprintf(stderr, "FAIL: chunk %u was cut at startup: %s\n", chunk, cuts.front().c_str());
                return 1;
            }
            if (tokens.empty()) {
                tokens = engine.tokenize_text(infernix::test::resume_probe_text("lighthouses and the people who kept them", 1600));
                if (tokens.size() <= lengths.back() + 1) { throw std::runtime_error("the prompt text is too short"); }
            }
            std::vector<Readout> row;
            for (const std::size_t length : lengths) { row.push_back(read(engine, tokens, length)); }
            readouts.push_back(std::move(row));
            const std::size_t c = readouts.size() - 1;
            for (std::size_t l = 0; c > 0 && l < lengths.size(); ++l) {
                const Readout& base = readouts.front()[l];
                const Readout& now  = readouts[c][l];
                std::size_t differ  = 0;
                double worst        = 0.0;
                for (std::size_t i = 0; i < kReadouts; ++i) {
                    if (std::bit_cast<std::uint32_t>(base.logprobs[i]) != std::bit_cast<std::uint32_t>(now.logprobs[i])) {
                        ++differ;
                        worst = std::max(worst, std::fabs(static_cast<double>(base.logprobs[i]) - static_cast<double>(now.logprobs[i])));
                    }
                }
                const bool top_equal = base.top == now.top &&
                                       std::bit_cast<std::uint32_t>(base.top_logprob) == std::bit_cast<std::uint32_t>(now.top_logprob);
                const std::size_t r = lengths[l] % grid;
                std::printf("L %6zu (last call %5zu tokens at %u, %5zu at %u): chunk %5u against %5u: %s",
                            lengths[l], r == 0 ? static_cast<std::size_t>(chunks.front()) : r, chunks.front(),
                            r == 0 ? static_cast<std::size_t>(chunk) : r, chunk, chunk, chunks.front(),
                            differ == 0 && top_equal ? "bit-identical\n" : "DIFFERS");
                if (differ != 0 || !top_equal) {
                    std::printf(" (%zu of %u log-probabilities, max |d| %.3g; top token %s)\n", differ, kReadouts, worst,
                                top_equal ? "equal" : "differs");
                    ++failures;
                }
            }
        }
        const std::size_t compared = (chunks.size() - 1) * lengths.size();
        if (failures == 0) {
            std::printf("prefill keeps its bits across call widths on one grid (%zu comparisons)\n", compared);
            return 0;
        }
        std::printf("FAIL: %d of %zu comparisons differ\n", failures, compared);
        return 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
