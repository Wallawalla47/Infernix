// Causal scoring through the public Engine (EnginePurpose::CausalScoring) on the real artifact, against
// an independent teacher-forced reference: the logits infernix_qwen4_exp_forward_real_test --dump-logits
// writes from the Forward alone (every expert in RAM, no Program, tier, layer walk or logit tiles). Per
// position: the target's log-probability against the reference's FP64 log-softmax, top-1 agreement,
// and the top-k ids where the reference separates them. The Engine's prefill chunks may round
// attention differently from the dump's, so the criteria are statistical, not bit equality. --chunk
// must equal the dump's (both default to 256): different chunkings change a call's rounding, which
// routing near-ties amplify beyond these criteria (2026-10-20, 512 tokens, dump 256 against Engine
// 1,024: FP4-Local mean |dlogp| 0.024, Dense8 0.087; equal chunks: FP4-Local 3e-5 and 4e-5).
// Skips (77) unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   INFERNIX_QWEN4_ARTIFACT=out.infernix [INFERNIX_QWEN4_NGRAM=v.ngram] [INFERNIX_EXPERT_RAM_MIB=N]
//   infernix_qwen4_exp_score_real_test @TOKENS REFERENCE.bin [--chunk N] [--top-k K]
//
// TOKENS holds ids separated by commas or whitespace; REFERENCE is the dump of the same ids (int32
// vocabulary, int32 rows, then one FP32 row per position). INFERNIX_EXPERT_RAM_MIB selects the SSD tier.

#include "infernix/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<infernix::TokenId> read_ids(const std::string& arg) {
    std::string text = arg;
    if (!text.empty() && text.front() == '@') {
        std::ifstream file(text.substr(1));
        text.assign(std::istreambuf_iterator<char>(file), {});
    }
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream in(text);
    std::vector<infernix::TokenId> ids;
    for (long long id = 0; in >> id;) { ids.push_back(static_cast<infernix::TokenId>(id)); }
    return ids;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || argc < 3) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT and pass @TOKENS REFERENCE.bin\n");
        return 77;
    }
    std::uint32_t chunk = 256, top_k = 8;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string option = argv[i];
        if (option == "--chunk") { chunk = static_cast<std::uint32_t>(std::stoul(argv[i + 1])); }
        if (option == "--top-k") { top_k = static_cast<std::uint32_t>(std::stoul(argv[i + 1])); }
    }
    try {
        const auto ids = read_ids(argv[1]);
        std::ifstream reference(argv[2], std::ios::binary);
        std::int32_t header[2] = {0, 0};
        reference.read(reinterpret_cast<char*>(header), sizeof(header));
        const std::int32_t vocab = header[0], rows = header[1];
        if (!reference || ids.size() < 2 || rows < static_cast<std::int32_t>(ids.size()) - 1) {
            std::printf("FAIL: the reference does not hold a row per scored position (%d rows, %zu ids)\n", rows,
                        ids.size());
            return 1;
        }

        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM")) { options.ngram_volume_path = ngram; }
        if (const char* mib = std::getenv("INFERNIX_EXPERT_RAM_MIB")) { options.expert_ram_bytes = std::stoull(mib) << 20; }
        options.purpose       = infernix::EnginePurpose::CausalScoring;
        options.max_context   = static_cast<std::uint32_t>(ids.size());
        options.prefill_chunk = chunk;
        options.kv_cache      = infernix::KvCacheStorage::Int8Group64;
        infernix::Engine engine(options);
        infernix::ScoreOptions score;
        score.top_k = top_k;
        const infernix::ScoreResult result = engine.score_tokens(ids, 1, score);

        const std::size_t positions = ids.size() - 1;
        std::vector<float> row(static_cast<std::size_t>(vocab));
        double sum_abs = 0, max_abs = 0, nll = 0, reference_nll = 0;
        std::size_t top1 = 0, topk_checked = 0, topk_equal = 0;
        for (std::size_t p = 0; p < positions; ++p) {
            reference.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
            if (!reference) { throw std::runtime_error("the reference ends early"); }
            const double maximum = *std::max_element(row.begin(), row.end());
            double sum           = 0;
            for (const float v : row) { sum += std::exp(static_cast<double>(v) - maximum); }
            const double lse    = maximum + std::log(sum);
            const double target = static_cast<double>(row[static_cast<std::size_t>(ids[p + 1])]) - lse;
            const double diff   = std::abs(static_cast<double>(result.logprobs[p]) - target);
            sum_abs += diff;
            max_abs = std::max(max_abs, diff);
            nll -= result.logprobs[p];
            reference_nll -= target;
            std::vector<std::int32_t> order(row.size());
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + top_k + 1, order.end(),
                              [&](std::int32_t a, std::int32_t b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
            top1 += result.top_ids[p * top_k] == order[0] ? 1U : 0U;
            // Ranks the reference separates by more than a rounding (0.05 in logit) must agree.
            bool separated = true;
            for (std::uint32_t k = 0; k < top_k; ++k) { separated = separated && row[order[k]] - row[order[k + 1]] > 0.05F; }
            if (separated) {
                ++topk_checked;
                topk_equal += std::equal(order.begin(), order.begin() + top_k, result.top_ids.begin() + static_cast<std::ptrdiff_t>(p * top_k))
                                  ? 1U
                                  : 0U;
            }
        }
        const double mean_abs = sum_abs / static_cast<double>(positions);
        std::printf("scored %zu positions (chunk %u): perplexity %.4f (reference %.4f); |logprob - reference| mean %.5f "
                    "max %.4f; top-1 agreement %.2f %%; top-%u ids equal at %zu of %zu separated positions\n",
                    positions, chunk, std::exp(nll / static_cast<double>(positions)),
                    std::exp(reference_nll / static_cast<double>(positions)), mean_abs, max_abs,
                    100.0 * static_cast<double>(top1) / static_cast<double>(positions), top_k, topk_equal, topk_checked);
        const bool ok = result.logprobs.size() == positions && result.top_ids.size() == positions * top_k &&
                        mean_abs < 0.02 && 100 * top1 >= 98 * positions && topk_equal == topk_checked;
        std::printf("%s\n", ok ? "OK" : "FAIL: the Engine's scores differ from the Forward's");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
}
