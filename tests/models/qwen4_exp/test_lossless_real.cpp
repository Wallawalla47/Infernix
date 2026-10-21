// The lossless route on a real Qwen3.8-Flash-Next artifact (INFERNIX_QWEN4_ARTIFACT, optionally its
// n-gram volume in INFERNIX_QWEN4_NGRAM): every scored value is a function of the window's tokens alone.
// A repeated score is bit-identical, and shorter windows score their positions bit-identically to a
// longer one, whose queries past 2,051 tokens select their QSA blocks (so neither the call's width, the
// expert staging passes nor the selection's presence changes an earlier position); log-probabilities are
// valid and a repeated text is predicted well. Skips (77) without the artifact.

#include "infernix/engine.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n";
        return 77;
    }
    constexpr std::uint32_t kContext = 2688;
    constexpr std::size_t kTokens    = 2600;
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = infernix::EnginePurpose::CausalScoring;
    options.max_context   = kContext;
    options.prefill_chunk = kContext;
    options.kv_cache      = infernix::KvCacheStorage::BFloat16;
    options.fast_paths    = false;
    options.lossless      = true;
    if (const char* volume = std::getenv("INFERNIX_QWEN4_NGRAM"); volume != nullptr && *volume != '\0') {
        options.ngram_volume_path = volume;
    }
    infernix::Engine engine(options);

    const std::string paragraph =
        "The lossless route computes each operation of the model exactly and rounds it once, so a "
        "token's log-probability depends on nothing but the tokens before it. Rivers carve canyons "
        "over millions of years; the Colorado cut the Grand Canyon through layers of sandstone, "
        "limestone and shale, exposing nearly two billion years of geological history.\n";
    std::string text;
    std::vector<infernix::TokenId> tokens;
    while (tokens.size() < kTokens) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(kTokens);

    constexpr std::uint32_t kTop = 8;
    const infernix::ScoreResult full   = engine.score_tokens(tokens, 1, {.top_k = kTop});
    const infernix::ScoreResult repeat = engine.score_tokens(tokens, 1, {.top_k = kTop});
    if (full.logprobs.size() != tokens.size() - 1) {
        std::cerr << "FAIL lossless scoring returned an invalid shape\n";
        return 1;
    }
    if (full.logprobs != repeat.logprobs || full.top_ids != repeat.top_ids || full.top_logprobs != repeat.top_logprobs) {
        std::cerr << "FAIL a repeated lossless score differs\n";
        return 1;
    }
    for (const float v : full.logprobs) {
        if (!std::isfinite(v) || v > 0.0F) {
            std::cerr << "FAIL lossless scoring returned an invalid log-probability\n";
            return 1;
        }
    }
    // Shorter windows over the same prefix (one within the QSA budget, one just past it): the same bits
    // at every shared position.
    for (const std::size_t length : {std::size_t{700}, std::size_t{2100}}) {
        std::vector<infernix::TokenId> prefix(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(length));
        const infernix::ScoreResult part = engine.score_tokens(prefix, 1, {.top_k = kTop});
        for (std::size_t i = 0; i < part.logprobs.size(); ++i) {
            if (part.logprobs[i] != full.logprobs[i]) {
                std::cerr << "FAIL position " << i << " of a " << length << "-token window scores "
                          << part.logprobs[i] << ", the " << kTokens << "-token window " << full.logprobs[i] << "\n";
                return 1;
            }
        }
        for (std::size_t i = 0; i < part.top_ids.size(); ++i) {
            if (part.top_ids[i] != full.top_ids[i] || part.top_logprobs[i] != full.top_logprobs[i]) {
                std::cerr << "FAIL top tokens differ between window widths at " << i / kTop << "\n";
                return 1;
            }
        }
    }
    // The paragraph repeats: from its third copy on the model predicts it closely.
    double sum = 0.0;
    const std::size_t from = tokens.size() / 2;
    for (std::size_t i = from; i < full.logprobs.size(); ++i) { sum += full.logprobs[i]; }
    const double mean = sum / static_cast<double>(full.logprobs.size() - from);
    std::cout << "lossless: mean log-probability over the repeated half " << mean << "\n";
    if (mean < -0.5) {
        std::cerr << "FAIL the repeated text is predicted poorly (mean log-probability " << mean << ")\n";
        return 1;
    }
    std::cout << "lossless: deterministic and width-invariant over " << tokens.size() << " tokens\n";
    return 0;
}
