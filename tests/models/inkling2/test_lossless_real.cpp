// Inkling 2's --lossless route on a real artifact (docs/maintainer/lossless-route.md §3a): a position's logits are a
// function of the request and the artifact alone.
// 1. The cache form against the window form, its exact oracle: a 1,226-token sequence (pseudo-random ids) scored as one
//    window, then through LosslessForward's cache form from a zero lane state (a permuted block table) in calls of
//    mixed widths: a first call wider than a sliding visit's ring (1,088 positions) then single columns (decode calls),
//    and calls of 100, 333 and 586 across the ring's wrap and the 1,024-key window with single columns between; and a
//    state saved after the first call, restored, and continued as one call. Every call's logits rows equal the
//    window's byte for byte.
// 2. Generation through the Engine (--lossless, --kv-dtype bf16, chunks of 100, two lanes, the prefix cache): the
//    greedy tokens of a 1,200-token prompt are the lossless scoring Engine's top tokens; the prompt readouts (the
//    greedy request's, and the prompt continued by 1, 12 and 23 of its tokens, resumed from the prefix cache) equal
//    the scoring Engine's log-probabilities byte for byte; sampled requests (temperature 1: Gumbel-max) give the same
//    tokens alone, two at once in reverse order, with chunks of 512 and resumed from the prefix cache.
// 3. The prefix cache's lower tiers on this route: a snapshot pushed out to the SSD tier and read back, and one saved
//    by --prefix-cache-file and restored by a new Engine, each resume giving the same sampled tokens.
//
//   INFERNIX_INKLING2_ARTIFACT  an artifact converted with the inkling2_mini recipe (skips without it)

#include "artifact/reader.h"
#include "core/device.h"
#include "infernix/engine.h"
#include "infernix/ops/inkling2_attention.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/lossless.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/load.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
namespace ik = ops::inkling2;
namespace m  = models::inkling2;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}
void cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

// ---------------------------------------------------------------------------- 1. the cache form

constexpr std::int32_t kN = 1226;

std::uint64_t row_hash(const float* row, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(row);
    for (std::size_t i = 0; i < n * 4; ++i) { h = (h ^ bytes[i]) * 1099511628211ULL; }
    return h;
}

// Logits rows [first, first + count) of a LosslessForward's last run, hashed, 64 at a time.
std::vector<std::uint64_t> head_hashes(m::LosslessForward& f, std::int32_t first, std::int32_t count, std::int32_t V) {
    std::vector<std::uint64_t> out;
    std::vector<float> rows;
    for (std::int32_t c = 0; c < count; c += m::LosslessForward::kHeadRows) {
        const std::int32_t n = std::min(m::LosslessForward::kHeadRows, count - c);
        f.head(first + c, n, nullptr);
        rows.resize(static_cast<std::size_t>(n) * V);
        cuda(cudaMemcpy(rows.data(), f.logits(), rows.size() * 4, cudaMemcpyDeviceToHost), "logits");
        for (std::int32_t i = 0; i < n; ++i) { out.push_back(row_hash(rows.data() + static_cast<std::size_t>(i) * V, V)); }
    }
    return out;
}

// One lane's state: its image and the full visits' pools, pages named through a permuted block-table row.
struct LaneState {
    m::StateLayout layout;
    std::uint16_t* image = nullptr;
    std::vector<std::uint16_t*> pool_k, pool_v;
    std::int32_t* table = nullptr;
    std::size_t image_bytes = 0, pool_bytes = 0;
    std::vector<std::vector<std::uint8_t>> saved;
    static constexpr int kPoolPages = 23; // prime: page b of the sequence at (5 b + 3) % 23
    explicit LaneState(const m::TextConfig& c) : layout(m::state_layout(c)) {
        image_bytes = static_cast<std::size_t>(layout.lane_words) * 2;
        cuda(cudaMalloc(&image, image_bytes), "image");
        pool_bytes = static_cast<std::size_t>(kPoolPages) * ik::kKvHeads * ik::kKeyUnit * ik::kHeadDim * 2;
        for (int s = 0; s < layout.full_visits; ++s) {
            for (auto* plane : {&pool_k, &pool_v}) {
                std::uint16_t* p = nullptr;
                cuda(cudaMalloc(&p, pool_bytes), "pool");
                plane->push_back(p);
            }
        }
        const int pages = (kN + ik::kKeyUnit - 1) / ik::kKeyUnit;
        require(pages <= kPoolPages, "the pool holds the sequence");
        std::vector<std::int32_t> row(static_cast<std::size_t>(pages));
        for (int b = 0; b < pages; ++b) { row[static_cast<std::size_t>(b)] = (5 * b + 3) % kPoolPages; }
        cuda(cudaMalloc(&table, row.size() * 4), "table");
        cuda(cudaMemcpy(table, row.data(), row.size() * 4, cudaMemcpyHostToDevice), "table");
        reset();
    }
    ~LaneState() {
        cudaFree(image);
        for (auto* p : pool_k) { cudaFree(p); }
        for (auto* p : pool_v) { cudaFree(p); }
        cudaFree(table);
    }
    LaneState(const LaneState&)            = delete;
    LaneState& operator=(const LaneState&) = delete;
    void reset() {
        cuda(cudaMemset(image, 0, image_bytes), "reset");
        for (auto* p : pool_k) { cuda(cudaMemset(p, 0, pool_bytes), "reset"); }
        for (auto* p : pool_v) { cuda(cudaMemset(p, 0, pool_bytes), "reset"); }
    }
    std::vector<std::uint16_t*> planes() const {
        std::vector<std::uint16_t*> out = pool_k;
        out.insert(out.end(), pool_v.begin(), pool_v.end());
        return out;
    }
    void save() {
        saved.clear();
        saved.emplace_back(image_bytes);
        cuda(cudaMemcpy(saved.back().data(), image, image_bytes, cudaMemcpyDeviceToHost), "save");
        for (auto* p : planes()) {
            saved.emplace_back(pool_bytes);
            cuda(cudaMemcpy(saved.back().data(), p, pool_bytes, cudaMemcpyDeviceToHost), "save");
        }
    }
    void restore() {
        cuda(cudaMemcpy(image, saved[0].data(), image_bytes, cudaMemcpyHostToDevice), "restore");
        const auto all = planes();
        for (std::size_t i = 0; i < all.size(); ++i) {
            cuda(cudaMemcpy(all[i], saved[i + 1].data(), pool_bytes, cudaMemcpyHostToDevice), "restore");
        }
    }
    m::LosslessLaneState at(std::int32_t first) const {
        m::LosslessLaneState s;
        s.image  = image;
        s.layout = &layout;
        s.pool_k = pool_k;
        s.pool_v = pool_v;
        s.table  = table;
        s.first  = first;
        return s;
    }
};

// Runs `widths` (summing to the rest of the sequence from `first`) through the cache form; each call's rows must equal
// the window's.
void cache_calls(const std::string& label, m::LosslessForward& cache, LaneState& lane, const std::vector<std::int32_t>& tokens,
                 const std::vector<std::uint64_t>& expected, std::int32_t first, const std::vector<std::int32_t>& widths,
                 std::int32_t V, bool save_after_first = false) {
    std::int32_t at = first;
    for (std::size_t i = 0; i < widths.size(); ++i) {
        const std::int32_t w = widths[i];
        cache.run(std::span<const std::int32_t>(tokens.data() + at, static_cast<std::size_t>(w)), lane.at(at), nullptr);
        const std::vector<std::uint64_t> got = head_hashes(cache, 0, w, V);
        for (std::int32_t c = 0; c < w; ++c) {
            if (got[static_cast<std::size_t>(c)] != expected[static_cast<std::size_t>(at + c)]) {
                throw std::runtime_error(label + ": position " + std::to_string(at + c) + " (call " + std::to_string(i) +
                                         ", " + std::to_string(w) + " columns from " + std::to_string(at) +
                                         ") differs from the window form's logits");
            }
        }
        if (save_after_first && i == 0) { lane.save(); }
        at += w;
    }
    require(at == kN, label + ": the calls cover the sequence");
    std::cout << label << ": " << widths.size() << " calls, " << (kN - first) << " positions equal the window form's\n";
}

void cache_against_window(const m::Parameters& p, const m::TextConfig& c) {
    const auto V = static_cast<std::int32_t>(c.vocab_size);
    std::vector<std::int32_t> tokens(static_cast<std::size_t>(kN));
    std::uint64_t x = 0x9E3779B97F4A7C15ULL;
    for (auto& t : tokens) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        t = 1000 + static_cast<std::int32_t>((x >> 33) % 60000U);
    }
    std::vector<std::uint64_t> expected;
    {
        m::LosslessForward window(p, kN);
        window.run(tokens, nullptr);
        expected = head_hashes(window, 0, kN, V);
    }
    std::cout << "window form: " << kN << " positions scored\n";
    m::LosslessForward cache(p, 1150, false, kN);
    LaneState lane(c);
    cache_calls("cache form, a wide call then decode calls", cache, lane, tokens, expected, 0, {1150, 1, 1, 1, 70, 1, 1, 1},
                V, true);
    lane.restore();
    cache_calls("cache form, restored after the wide call", cache, lane, tokens, expected, 1150, {76}, V);
    lane.reset();
    cache_calls("cache form, calls across the ring's wrap and the window", cache, lane, tokens, expected, 0,
                {100, 100, 100, 1, 1, 1, 1, 333, 1, 586, 1, 1}, V);
}

// ---------------------------------------------------------------------------- 2. the Engine

constexpr std::uint32_t kContext = 2048;
constexpr std::uint32_t kOutput  = 24;

EngineOptions engine_options(const char* artifact) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.lossless      = true;
    options.fast_paths    = false;
    options.kv_cache      = KvCacheStorage::BFloat16;
    options.max_context   = kContext;
    options.speculative.backend = SpeculativeBackend::None;
    options.diagnostic_observer.callback = [](const Diagnostic& d) {
        if (d.level >= DiagnosticLevel::Info) { std::cerr << "[engine] " << d.message << '\n'; }
    };
    return options;
}

EngineOptions generation_options(const char* artifact, std::uint32_t chunk, std::uint32_t lanes) {
    EngineOptions options                     = engine_options(artifact);
    options.max_concurrency                   = lanes;
    options.max_pending_requests              = 2 * lanes;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(2 * lanes * kContext);
    options.prefill_chunk                     = chunk;
    options.context_cache.enabled             = true;
    options.context_cache.host_capacity_bytes = std::size_t{3} << 30U;
    return options;
}

RequestOptions request(bool greedy, std::uint64_t seed, bool reuse, std::uint32_t outputs = kOutput) {
    RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = greedy ? 0.0F : 1.0F;
    options.execution.sampling.top_k          = 0;
    options.execution.sampling.top_p          = 1.0F;
    options.execution.sampling.min_p          = 0.0F;
    options.execution.sampling.seed           = seed;
    options.execution.sampling.presence_penalty  = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

std::string long_notes() {
    std::string notes = "Field notes from the river survey, read them and continue the log.\n";
    for (int entry = 0; entry < 64; ++entry) {
        notes += "Entry " + std::to_string(entry) + ": the team logged " + std::to_string(3 * entry + 7) +
                 " birds near marker " + std::to_string(entry % 17) + ", wind from the " +
                 (entry % 2 ? "north" : "south-west") + ", water level " + std::to_string(40 + entry % 9) +
                 " centimetres.\n";
    }
    return notes;
}

void require_same(const std::string& label, const std::vector<TokenId>& a, const std::vector<TokenId>& b) {
    require(!a.empty(), label + ": the reference generated nothing");
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) { ++i; }
    if (a != b) {
        throw std::runtime_error(label + ": tokens differ at " + std::to_string(i) + " of " + std::to_string(a.size()) +
                                 "/" + std::to_string(b.size()));
    }
    std::cout << label << ": " << a.size() << " tokens identical\n";
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

// 3. `solo`: the prompt's sampled tokens (seed 200) from a cold run.
void lower_tiers(const char* artifact, const std::vector<TokenId>& prompt, const std::vector<TokenId>& solo) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix_inkling2_lossless_tiers";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::create_directories(dir);
    {
        // A Host tier of one snapshot (a 190 MiB image and the prompt's 18 blocks): another prompt pushes it to the SSD.
        EngineOptions options                          = generation_options(artifact, 512, 1);
        options.context_cache.host_capacity_bytes      = std::size_t{260} << 20U;
        options.context_cache.ssd_capacity_bytes       = std::uint64_t{2} << 30U;
        options.context_cache.ssd_write_bytes_per_hour = ContextCacheOptions::kSsdWritesUnlimited;
        options.context_cache.ssd_directory            = dir;
        Engine engine(options);
        (void)engine.generate(engine.prepare_tokens(prompt), request(false, 7, true, 1));
        const std::vector<TokenId> other(prompt.rbegin(), prompt.rend());
        (void)engine.generate(engine.prepare_tokens(other), request(false, 7, true, 1));
        const RuntimeStats before = engine.runtime_stats();
        const GenerationResult warm = engine.generate(engine.prepare_tokens(prompt), request(false, 200, true));
        const RuntimeStats after  = engine.runtime_stats();
        require(after.hybrid_ssd_promotions > before.hybrid_ssd_promotions, "ssd tier: the source was not read back");
        require(after.hybrid_ssd_failures == 0, "ssd tier: an SSD transfer failed");
        require(warm.reused_prompt_tokens >= 1152, "ssd tier: the resume reused too little");
        require_same("ssd tier (" + std::to_string(warm.reused_prompt_tokens) + " tokens reused, " +
                         std::to_string(after.hybrid_ssd_read_bytes / 1000000) + " MB read)",
                     solo, warm.generated_token_ids);
    }
    {
        EngineOptions options                            = generation_options(artifact, 512, 1);
        options.context_cache.hybrid.persistent_file     = dir / "lossless.cache";
        options.context_cache.hybrid.persistent_identity = "infernix_inkling2_lossless_real_test";
        {
            Engine engine(options);
            (void)engine.generate(engine.prepare_tokens(prompt), request(false, 7, true, 1));
            engine.stop(); // the orderly stop saves the Host tier
        }
        require(std::filesystem::exists(options.context_cache.hybrid.persistent_file), "file: nothing was saved at shutdown");
        Engine engine(options);
        const GenerationResult warm = engine.generate(engine.prepare_tokens(prompt), request(false, 200, true));
        require(warm.reused_prompt_tokens >= 1152, "file: the restarted Engine reused too little");
        require_same("prefix cache file (" + std::to_string(warm.reused_prompt_tokens) + " tokens reused)", solo,
                     warm.generated_token_ids);
    }
    std::filesystem::remove_all(dir, ignored);
}

void generation(const char* artifact) {
    std::vector<std::vector<TokenId>> prompts;
    std::vector<TokenId> greedy;
    std::vector<std::vector<TokenId>> solo;
    PromptReadout greedy_readout;
    struct Forced {
        std::uint32_t k = 0;
        float logprob   = 0.0F;
        std::uint32_t reused = 0;
    };
    std::vector<Forced> forced;
    {
        Engine engine(generation_options(artifact, 100, 2));
        prompts.push_back(engine.tokenize_text(long_notes()));
        prompts.push_back(engine.tokenize_text("Write a short story about a lighthouse keeper who finds a message in a bottle."));
        require(prompts[0].size() >= 1200, "the long prompt passes a sliding visit's ring");
        prompts[0].resize(1200);
        RequestOptions first = request(true, 0, true);
        first.execution.readout_tokens = {prompts[0][0], prompts[0][1]};
        const GenerationResult g = engine.generate(engine.prepare_tokens(prompts[0]), first);
        greedy                   = g.generated_token_ids;
        require(greedy.size() == kOutput && g.readout.has_value(), "the greedy request stopped early or read nothing");
        greedy_readout = *g.readout;
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            solo.push_back(engine.generate(engine.prepare_tokens(prompts[i]), request(false, 200 + i, false)).generated_token_ids);
        }
        std::vector<GenerationHandle> handles;
        for (std::size_t i = prompts.size(); i-- > 0;) {
            handles.push_back(engine.submit(engine.prepare_tokens(prompts[i]), request(false, 200 + i, false)));
        }
        for (std::size_t j = 0; j < handles.size(); ++j) {
            const std::size_t i = prompts.size() - 1 - j;
            require_same("concurrent request " + std::to_string(i), solo[i], handles[j].wait().generated_token_ids);
        }
        // The prompt continued by greedy tokens, resumed where the cache serves it: its readout of the next one.
        for (const std::uint32_t k : {1U, 12U, 23U}) {
            std::vector<TokenId> continued = prompts[0];
            continued.insert(continued.end(), greedy.begin(), greedy.begin() + k);
            RequestOptions r            = request(true, 0, true, 1);
            r.execution.readout_tokens  = {greedy[k]};
            const GenerationResult out  = engine.generate(engine.prepare_tokens(continued), r);
            require(out.readout.has_value() && out.readout->logprobs.size() == 1, "a continued prompt read nothing");
            forced.push_back({k, out.readout->logprobs[0], out.reused_prompt_tokens});
        }
        const auto warm = engine.generate(engine.prepare_tokens(prompts[0]), request(false, 200, true));
        require(warm.reused_prompt_tokens != 0, "the warm request reused no prefix");
        require_same("warm prefix (" + std::to_string(warm.reused_prompt_tokens) + " tokens reused)", solo[0],
                     warm.generated_token_ids);
    }
    {
        Engine engine(generation_options(artifact, 512, 1));
        require_same("prefill chunk 512 (sampled)", solo[0],
                     engine.generate(engine.prepare_tokens(prompts[0]), request(false, 200, false)).generated_token_ids);
    }
    {
        EngineOptions options = engine_options(artifact);
        options.purpose       = EnginePurpose::CausalScoring;
        options.prefill_chunk = kContext;
        Engine engine(options);
        std::vector<TokenId> sequence = prompts[0];
        sequence.insert(sequence.end(), greedy.begin(), greedy.end());
        const auto first = static_cast<std::uint32_t>(prompts[0].size());
        ScoreOptions score_options;
        score_options.top_k                   = 1;
        score_options.candidates_per_position = 1;
        score_options.candidates.assign(sequence.begin() + first, sequence.end());
        const ScoreResult score = engine.score_tokens(sequence, first, score_options);
        require(score.top_ids.size() == greedy.size() && score.candidate_logprobs.size() == greedy.size(),
                "the score has an unexpected shape");
        for (std::size_t k = 0; k < greedy.size(); ++k) {
            if (score.top_ids[k] != greedy[k]) {
                throw std::runtime_error("generated token " + std::to_string(k) + " is " + std::to_string(greedy[k]) +
                                         ", the scoring route's top token " + std::to_string(score.top_ids[k]));
            }
        }
        std::cout << "greedy generation: " << greedy.size() << " tokens are the scoring route's top tokens\n";
        // The greedy request's readout: its top token and log-probability are position first - 1's.
        require(greedy_readout.top_token == greedy[0] && same_bits(greedy_readout.top_logprob, score.top_logprobs[0]) &&
                    same_bits(greedy_readout.top_logprob, score.candidate_logprobs[0]),
                "the greedy request's readout differs from the scoring route's log-probability");
        for (const Forced& f : forced) {
            if (!same_bits(f.logprob, score.candidate_logprobs[f.k])) {
                throw std::runtime_error("the prompt continued by " + std::to_string(f.k) + " tokens reads " +
                                         std::to_string(f.logprob) + ", the scoring route " +
                                         std::to_string(score.candidate_logprobs[f.k]));
            }
            std::cout << "continued by " << f.k << " (" << f.reused << " tokens reused): log p " << f.logprob
                      << " equals the scoring route's bit for bit\n";
        }
    }
    lower_tiers(artifact, prompts[0], solo[0]);
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: INFERNIX_INKLING2_ARTIFACT is not set\n";
        return 77;
    }
    try {
        {
            DeviceContext device(0);
            artifact::Reader reader(artifact);
            auto model = m::materialize_model(m::plan_load(reader, models::LoadOptions{}), device);
            const m::Parameters parameters(*model);
            cache_against_window(parameters, model->config().text);
        }
        generation(artifact);
        std::cout << "OK Inkling 2 lossless\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
