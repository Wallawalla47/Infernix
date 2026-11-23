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
// The dump runs none of the default fast paths: --no-fast-paths runs the Engine on the same kernels. On the default
// route NVIDIA NVFP4 QAD Dense8's fast paths move the scores beyond these criteria (2026-11-23, 300 tokens at chunk
// 256: mean |dlogp| 0.067, top-1 96 %; with --no-fast-paths 0.00028, 100 %, and the hidden states bit-identical).
//
// With --blocks BLOCKS.bin (the forward test's --blocks of the same ids and --chunk: BF16 [position][layer][mixer in,
// mixer out, MoE in, MoE out][H]), the prompt's hidden states (ScoreOptions::prompt_hidden_states, the window's last
// token as the continuation) are checked too: against the Forward's mixer inputs of layers 1.. and the last layer's
// MoE input at the last prompt position (per layer relative L2 within 1e-2: the Program's kernels round differently,
// a wrong column or layer is off by order one); the scores identical with and without them; and the same hidden
// states, within that bound, when the continuation token changes (the capture reads the prompt's column).

#include "infernix/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

float bf16_value(std::uint16_t bits) {
    const std::uint32_t word = std::uint32_t{bits} << 16U;
    float value              = 0;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

// The worst layer's relative L2 difference of two [layers][H] hidden-state blocks.
double worst_relative(const std::vector<float>& a, const std::vector<float>& b, std::size_t layers) {
    if (a.size() != b.size() || layers == 0 || a.size() % layers != 0) { return INFINITY; }
    const std::size_t width = a.size() / layers;
    double worst            = 0;
    for (std::size_t layer = 0; layer < layers; ++layer) {
        double diff = 0, norm = 0;
        for (std::size_t i = layer * width; i < (layer + 1) * width; ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            diff += d * d;
            norm += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        worst = std::max(worst, std::sqrt(diff / std::max(norm, 1e-30)));
    }
    return worst;
}

// --blocks: the prompt's hidden states (all ids but the last; the last is the continuation) against the Forward's taps.
bool check_hidden_states(infernix::Engine& engine, const std::vector<infernix::TokenId>& ids, const std::string& path) {
    const auto first_target = static_cast<std::uint32_t>(ids.size() - 1);
    const infernix::ScoreResult plain  = engine.score_tokens(ids, first_target);
    const infernix::ScoreResult hidden = engine.score_tokens(ids, first_target, {.prompt_hidden_states = true});
    const std::size_t layers           = hidden.hidden_state_layers;
    if (layers == 0 || hidden.hidden_states.empty() || hidden.hidden_states.size() % layers != 0) {
        std::printf("FAIL: no hidden states\n");
        return false;
    }
    const std::size_t H = hidden.hidden_states.size() / layers, T = ids.size(), position = T - 2;
    std::ifstream file(path, std::ios::binary);
    std::vector<std::uint16_t> taps(T * layers * 4 * H);
    file.read(reinterpret_cast<char*>(taps.data()), static_cast<std::streamsize>(taps.size() * 2));
    if (!file || file.peek() != std::char_traits<char>::eof()) {
        std::printf("FAIL: %s is not BF16 [%zu positions][%zu layers][4][%zu]\n", path.c_str(), T, layers, H);
        return false;
    }
    // Row L: layer L + 1's mixer input (tap 0), and the last layer's MoE input (tap 2). Position-major, as the forward
    // test writes; the layer-major reading [layer][tap][position][H] is printed beside it for comparison.
    std::vector<float> forward(layers * H), layer_major(layers * H);
    for (std::size_t row = 0; row < layers; ++row) {
        const std::size_t layer = row + 1 < layers ? row + 1 : row, tap = row + 1 < layers ? 0 : 2;
        for (std::size_t h = 0; h < H; ++h) {
            forward[row * H + h]     = bf16_value(taps[((position * layers + layer) * 4 + tap) * H + h]);
            layer_major[row * H + h] = bf16_value(taps[((layer * 4 + tap) * T + position) * H + h]);
        }
    }
    const double against_forward = worst_relative(hidden.hidden_states, forward, layers);
    const double against_misread = worst_relative(hidden.hidden_states, layer_major, layers);
    std::printf("per layer (relative L2, cosine) against the Forward:");
    for (std::size_t row = 0; row < layers; ++row) {
        double diff = 0, norm = 0, dot = 0, own = 0;
        for (std::size_t h = 0; h < H; ++h) {
            const double a = hidden.hidden_states[row * H + h], b = forward[row * H + h];
            diff += (a - b) * (a - b);
            norm += b * b;
            dot += a * b;
            own += a * a;
        }
        std::printf(" %zu:%.3g/%.4f", row, std::sqrt(diff / std::max(norm, 1e-30)), dot / std::sqrt(std::max(norm * own, 1e-30)));
    }
    std::printf("\n");
    std::vector<infernix::TokenId> other = ids;
    other.back()                         = ids.back() == ids[0] ? ids[1] : ids[0];
    const infernix::ScoreResult changed  = engine.score_tokens(other, first_target, {.prompt_hidden_states = true});
    const double against_changed         = worst_relative(changed.hidden_states, hidden.hidden_states, layers);
    const bool same_scores               = plain.logprobs == hidden.logprobs && plain.hidden_states.empty();
    std::printf("hidden states (%zu layers x %zu) at position %zu: worst layer relative L2 against the Forward %.3g "
                "(the layer-major reading of the same file: %.3g); with another continuation token %.3g (%s); scores "
                "%s with and without them\n",
                layers, H, position, against_forward, against_misread, against_changed,
                changed.hidden_states == hidden.hidden_states ? "bit-identical" : "not bit-identical",
                same_scores ? "identical" : "DIFFERENT");
    const bool ok = against_forward <= 1e-2 && against_changed <= 1e-2 && same_scores;
    std::printf("%s\n", ok ? "OK hidden states" : "FAIL: the hidden states differ from the Forward's");
    return ok;
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
    std::string blocks_path;
    bool fast_paths = true;
    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--no-fast-paths") { // the Forward's kernels: the dump runs none of the default fast paths
            fast_paths = false;
            continue;
        }
        if (i + 1 >= argc) { break; }
        if (option == "--chunk") { chunk = static_cast<std::uint32_t>(std::stoul(argv[i + 1])); }
        if (option == "--top-k") { top_k = static_cast<std::uint32_t>(std::stoul(argv[i + 1])); }
        if (option == "--blocks") { blocks_path = argv[i + 1]; }
        ++i;
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
        options.fast_paths    = fast_paths;
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
        bool hidden_ok = true;
        if (!blocks_path.empty()) { hidden_ok = check_hidden_states(engine, ids, blocks_path); }
        return ok && hidden_ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
}
