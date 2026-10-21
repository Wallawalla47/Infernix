// Lossless generation on a real Qwen3.8-Flash-Next artifact (INFERNIX_QWEN4_ARTIFACT, optionally its n-gram
// volume in INFERNIX_QWEN4_NGRAM; docs/maintainer/lossless-route.md): every token is a function of the request
// and the artifact alone. The long prompt (about 2,300 tokens) passes the QSA budget, so decode selects blocks.
//  - Greedy generation picks the lossless scoring route's top token at every position: prefill in chunks,
//    the KV pages, pooled keys and tails, the PLE history, the GDN state and decode compute what one scoring
//    pass computes.
//  - Sampled requests (temperature 1, fixed seeds) give the same tokens alone, concurrently in another
//    order, with another prefill chunk and resumed from the prefix cache.
// Skips (77) without the artifact.

#include "infernix/engine.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kContext = 3072;
constexpr std::uint32_t kOutput  = 32;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

infernix::EngineOptions base_options(const char* artifact) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    options.lossless      = true;
    options.fast_paths    = false;
    options.kv_cache      = infernix::KvCacheStorage::BFloat16;
    options.max_context   = kContext;
    options.enable_vision = false;
    if (const char* volume = std::getenv("INFERNIX_QWEN4_NGRAM"); volume != nullptr && *volume != '\0') {
        options.ngram_volume_path = volume;
    }
    return options;
}

infernix::EngineOptions generation_options(const char* artifact, std::uint32_t chunk, std::uint32_t lanes) {
    infernix::EngineOptions options        = base_options(artifact);
    options.max_concurrency                = lanes;
    options.kv_capacity                    = infernix::KvCapacityPolicy::explicit_capacity(lanes * kContext);
    options.max_pending_requests           = 2 * lanes;
    options.prefill_chunk                  = chunk;
    options.context_cache.enabled          = true;
    options.context_cache.host_capacity_bytes = std::size_t{2} << 30U;
    return options;
}

infernix::RequestOptions request(bool greedy, std::uint64_t seed, bool prefix_reuse = false) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = kOutput;
    options.execution.sampling.temperature    = greedy ? 0.0F : 1.0F;
    options.execution.sampling.top_k          = 0;
    options.execution.sampling.top_p          = 1.0F;
    options.execution.sampling.seed           = seed;
    if (greedy) {
        // Greedy picks the scoring pass's top token only without penalties, whatever the model's presets.
        options.execution.sampling.presence_penalty  = 0.0F;
        options.execution.sampling.frequency_penalty = 0.0F;
    }
    options.execution.allow_prefix_reuse      = prefix_reuse;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

std::string long_notes() {
    std::string notes = "Field notes from the river survey, read them and continue the log.\n";
    for (int entry = 0; entry < 70; ++entry) {
        notes += "Entry " + std::to_string(entry) + ": the team logged " + std::to_string(3 * entry + 7) +
                 " birds near marker " + std::to_string(entry % 17) + ", wind from the " +
                 (entry % 2 ? "north" : "south-west") + ", water level " + std::to_string(40 + entry % 9) +
                 " centimetres.\n";
    }
    return notes;
}

void require_same(const std::string& label, const std::vector<infernix::TokenId>& a,
                  const std::vector<infernix::TokenId>& b) {
    require(!a.empty(), label + ": the reference generated nothing");
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) { ++i; }
    if (a != b) {
        throw std::runtime_error(label + ": tokens differ at " + std::to_string(i) + " of " +
                                 std::to_string(a.size()) + "/" + std::to_string(b.size()));
    }
    std::cout << label << ": " << a.size() << " tokens identical\n";
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n";
        return 77;
    }
    try {
        std::vector<std::vector<infernix::TokenId>> tokens;
        std::vector<infernix::TokenId> greedy;
        std::vector<std::vector<infernix::TokenId>> solo;
        {
            infernix::Engine engine(generation_options(artifact, 1024, 2));
            for (const std::string& text :
                 {long_notes(), std::string("Write a short story about a lighthouse keeper who finds a message in a bottle.")}) {
                tokens.push_back(engine.tokenize_text(text));
            }
            require(tokens[0].size() > 2100 && tokens[0].size() + kOutput < kContext,
                    "the long prompt must pass the QSA budget and fit the context");
            greedy = engine.generate(engine.prepare_tokens(tokens[0]), request(true, 0)).generated_token_ids;
            require(greedy.size() == kOutput, "the greedy request stopped early");
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                solo.push_back(
                    engine.generate(engine.prepare_tokens(tokens[i]), request(false, 200 + i)).generated_token_ids);
            }
            std::vector<infernix::GenerationHandle> handles;
            for (std::size_t i = tokens.size(); i-- > 0;) {
                handles.push_back(engine.submit(engine.prepare_tokens(tokens[i]), request(false, 200 + i)));
            }
            for (std::size_t j = 0; j < handles.size(); ++j) {
                const std::size_t i = tokens.size() - 1 - j;
                require_same("concurrent request " + std::to_string(i), solo[i], handles[j].wait().generated_token_ids);
            }
            std::vector<infernix::TokenId> other = tokens[0];
            const auto tail = engine.tokenize_text("\nHow many birds were logged in all?");
            other.insert(other.end(), tail.begin(), tail.end());
            // Only its prompt is needed: it fills the prefix cache.
            infernix::RequestOptions warming = request(false, 7, true);
            warming.execution.requested_output_tokens = 1;
            (void)engine.generate(engine.prepare_tokens(other), warming);
            const auto before = engine.runtime_stats();
            const auto warm   = engine.generate(engine.prepare_tokens(tokens[0]), request(false, 200, true));
            require(engine.runtime_stats().reused_prompt_tokens > before.reused_prompt_tokens,
                    "the warm request reused no prefix");
            require_same("warm prefix", solo[0], warm.generated_token_ids);
        }
        {
            infernix::Engine engine(generation_options(artifact, 2048, 1));
            // Sampled (temperature 1) shows any arithmetic difference greedy decoding would.
            require_same("prefill chunk 2048 (sampled)", solo[0],
                         engine.generate(engine.prepare_tokens(tokens[0]), request(false, 200)).generated_token_ids);
        }
        {
            infernix::EngineOptions options = base_options(artifact);
            options.purpose                 = infernix::EnginePurpose::CausalScoring;
            options.prefill_chunk           = kContext;
            infernix::Engine engine(options);
            std::vector<infernix::TokenId> sequence = tokens[0];
            sequence.insert(sequence.end(), greedy.begin(), greedy.end());
            const auto first = static_cast<std::uint32_t>(tokens[0].size());
            const infernix::ScoreResult score = engine.score_tokens(sequence, first, {.top_k = 1});
            require(score.top_ids.size() == greedy.size(), "the score has an unexpected shape");
            for (std::size_t k = 0; k < greedy.size(); ++k) {
                if (score.top_ids[k] != greedy[k]) {
                    throw std::runtime_error("generated token " + std::to_string(k) + " is " + std::to_string(greedy[k]) +
                                             ", the scoring route's top token " + std::to_string(score.top_ids[k]));
                }
            }
            std::cout << "greedy generation: " << greedy.size() << " tokens equal the scoring route's top tokens\n";
        }
        std::cout << "OK Qwen3.8-Flash-Next lossless generation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "lossless generation: " << error.what() << '\n';
        return 1;
    }
}
