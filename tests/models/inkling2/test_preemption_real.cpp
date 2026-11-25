// Inkling 2's shared KV pool through the public Engine on a real artifact: requests that together outgrow the pool
// (--kv-capacity of one request's context) pause and resume, and every request's tokens equal its tokens alone in an
// Engine whose pool holds every lane (no pause). A pause publishes the lane's state at its frontier (a decode-route key
// past the prompt's full blocks), so a resume restores it or replays the rest exactly: the prompt route below the
// prompt's end, decode-route calls past it.
// 1. Greedy, plain decoding, C = 2 (two long prompts, long replies): preemptions happen, the tokens equal solo.
// 2. The same under DSpark (--spec auto with the bundled drafter) and under --kv-dtype int8.
// 3. Sampled (temperature 1, a seed each) at C = 3, and with a presence penalty (the resume rebuilds the counts).
// 4. Without the prefix cache, a paused request replays its whole ledger (replayed tokens): plain BF16, DSpark with
//    INT8 KV, and --a8-activations.
// 5. A 2,600-token prompt prefilled in 512-token chunks beside a reply: it binds with its first chunk and its later chunks
//    outgrow the pool (a pause beside or inside a prompt; a fresh request that cannot bind waits instead of pausing).
// 6. At scale: a 32,768-token pool, two 12,000-token prompts with 6,000-token replies under DSpark and INT8 KV.
// 7. The owner's launcher configuration (2026-11-25: 240,000 tokens, four lanes, INT8 KV, vision offloaded, embedding on
//    the host, DSpark with --lm-head-draft) plans and starts, and serves a request.
// Reported: preemptions, snapshot restores, replays, replayed tokens and the time paused per case.
//
//   INFERNIX_INKLING2_ARTIFACT   an artifact converted with the inkling2_mini recipe (with its DSpark drafter and the
//                                vision tower)
//   INFERNIX_INKLING2_NO_OWNER_CONFIG / INFERNIX_INKLING2_NO_SCALE_CASE  skip 7 / 6

#include "infernix/engine.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// One configuration of the Engine pair (the reference, whose pool holds every lane, and the pressured one).
struct Setup {
    std::uint32_t context = 4096;
    bool dspark           = false;
    infernix::KvCacheStorage kv = infernix::KvCacheStorage::BFloat16;
    bool cache            = true;
    std::uint32_t chunk   = 0; // the default
    bool a8               = false;
};

infernix::EngineOptions options_for(const std::filesystem::path& artifact, std::uint32_t lanes, std::uint32_t pool_tokens,
                                    const Setup& setup) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = setup.context;
    options.max_concurrency      = lanes;
    options.max_pending_requests = 16;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(pool_tokens);
    options.kv_cache             = setup.kv;
    options.prefill_chunk        = setup.chunk;
    options.a8_activations       = setup.a8;
    options.context_cache.enabled             = setup.cache;
    options.context_cache.host_capacity_bytes = setup.cache ? std::size_t{8} << 30U : 0U;
    options.speculative.backend = setup.dspark ? infernix::SpeculativeBackend::DSpark : infernix::SpeculativeBackend::None;
    options.speculative.draft_tokens      = 7;
    options.speculative.draft_tokens_auto = true;
    options.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        if (d.level >= infernix::DiagnosticLevel::Warning) { std::cerr << "[engine " << static_cast<int>(d.level) << "] " << d.message << '\n'; }
    };
    return options;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

infernix::RequestOptions sampled(std::uint32_t tokens, unsigned long long seed, float presence = 0.0F) {
    infernix::RequestOptions options = greedy(tokens);
    options.execution.sampling.temperature      = 1.0F;
    options.execution.sampling.top_k            = 20;
    options.execution.sampling.top_p            = 1.0F;
    options.execution.sampling.seed             = seed;
    options.execution.sampling.presence_penalty = presence;
    return options;
}

// A prompt of `count` raw-text ids: numbered sentences whose wording depends on `variant`.
Tokens prompt(const infernix::Engine& engine, std::size_t count, int variant) {
    static const char* const subjects[] = {"lighthouse", "orchard", "compiler", "river", "archive", "observatory"};
    static const char* const verbs[]    = {"records", "guards", "measures", "translates", "repairs", "counts"};
    std::string text = "Read the notes below, then write a long, careful summary of every item in order.\n";
    for (int i = 0; text.size() < count * 12; ++i) {
        text += "Note " + std::to_string(i) + ": the " + subjects[(i + variant) % 6] + " " + verbs[(i * 7 + variant) % 6] +
                " item " + std::to_string((i * 31 + variant * 17) % 97) + " before " + std::to_string(i % 24) + ":00.\n";
    }
    Tokens ids = engine.tokenize_text(text);
    require(ids.size() >= count, "the prompt's text encodes to at least " + std::to_string(count) + " ids");
    ids.resize(count);
    return ids;
}

struct Job {
    Tokens prompt;
    infernix::RequestOptions options;
};

struct Totals {
    std::uint64_t preemptions = 0, snapshot_restores = 0, replay_restores = 0, replayed_tokens = 0, paused_ns = 0;
};

std::vector<Tokens> solo(infernix::Engine& engine, const std::vector<Job>& jobs) {
    std::vector<Tokens> out;
    for (const Job& job : jobs) {
        const auto result = engine.generate(engine.prepare_tokens(job.prompt), job.options);
        require(result.scheduling.preemptions == 0, "a solo request was paused");
        out.push_back(result.generated_token_ids);
    }
    return out;
}

Totals together(infernix::Engine& engine, const std::vector<Job>& jobs, const std::vector<Tokens>& reference,
                const std::string& label) {
    std::vector<infernix::GenerationHandle> handles;
    for (const Job& job : jobs) { handles.push_back(engine.submit(engine.prepare_tokens(job.prompt), job.options)); }
    Totals totals;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        const auto result = handles[i].wait();
        const Tokens& got = result.generated_token_ids;
        std::size_t at    = 0;
        while (at < got.size() && at < reference[i].size() && got[at] == reference[i][at]) { ++at; }
        require(!got.empty() && got == reference[i], label + ", request " + std::to_string(i) + ": tokens differ at " +
                                                         std::to_string(at) + " of " + std::to_string(reference[i].size()));
        require(result.prompt.prompt_tokens == jobs[i].prompt.size(), label + ": the prompt's usage changed");
        totals.preemptions += result.scheduling.preemptions;
        totals.snapshot_restores += result.scheduling.snapshot_restores;
        totals.replay_restores += result.scheduling.replay_restores;
        totals.replayed_tokens += result.scheduling.replayed_tokens;
        totals.paused_ns += result.scheduling.paused_ns;
    }
    std::cout << label << ": " << jobs.size() << " requests' tokens equal solo; preemptions " << totals.preemptions
              << ", snapshot restores " << totals.snapshot_restores << ", replays " << totals.replay_restores
              << ", replayed tokens " << totals.replayed_tokens << ", paused " << totals.paused_ns / 1000000 << " ms\n";
    return totals;
}

// Runs `jobs` alone in an Engine whose pool holds every lane, then together in one whose pool holds one context.
Totals pressure(const std::filesystem::path& artifact, const std::vector<Job>& jobs, const std::string& label,
                const Setup& setup) {
    const auto lanes = static_cast<std::uint32_t>(jobs.size());
    std::vector<Tokens> reference;
    {
        infernix::Engine engine(options_for(artifact, lanes, lanes * setup.context, setup));
        reference = solo(engine, jobs);
    }
    infernix::Engine engine(options_for(artifact, lanes, setup.context, setup));
    const Totals totals = together(engine, jobs, reference, label);
    require(totals.preemptions != 0, label + ": the requests never outgrew the pool (no preemption)");
    return totals;
}

void owner_configuration(const std::filesystem::path& artifact) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 240000;
    options.max_concurrency      = 4;
    options.max_pending_requests = 16;
    options.kv_cache             = infernix::KvCacheStorage::Int8Group64;
    options.kv_capacity          = infernix::KvCapacityPolicy::automatic();
    options.vram_headroom_bytes  = std::uint64_t{128} << 20U;
    options.vram_past_budget     = true;
    options.embedding_host       = true;
    options.enable_vision        = true;
    options.vision_offload       = infernix::VisionOffload::On;
    options.speculative.backend_auto      = true;
    options.speculative.draft_tokens_auto = true;
    options.speculative.proposal_head     = infernix::ProposalHead::Optimized;
    const infernix::EnginePlan plan = infernix::Engine::plan(options);
    std::cout << "owner configuration: plan " << (plan.fits ? "fits" : "refused: " + plan.error) << ", KV pool "
              << plan.vram.kv_tokens << " tokens, " << (plan.vram.left_bytes >> 20U) << " MiB left\n";
    require(plan.fits && plan.vram.kv_tokens >= 240000, "owner configuration: the plan does not fit one 240,000-token context");
    infernix::Engine engine(options);
    const auto result = engine.generate(engine.prepare_tokens(prompt(engine, 200, 9)), greedy(32));
    require(result.generated_token_ids.size() == 32, "owner configuration: the request did not complete");
    std::cout << "owner configuration: started and served a request\n";
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    if (artifact == nullptr) {
        std::cout << "skip: set INFERNIX_INKLING2_ARTIFACT\n";
        return 77;
    }
    try {
        std::vector<Job> two, three, penalised, prompts, long_two;
        {
            Setup probe;
            probe.cache = false;
            infernix::Engine engine(options_for(artifact, 1, probe.context, probe));
            // 1,500-token prompts and 900-token replies: two lanes hold ~4,800 positions of a 4,096-token pool.
            two   = {{prompt(engine, 1500, 0), greedy(900)}, {prompt(engine, 1500, 1), greedy(900)}};
            three = {{prompt(engine, 1100, 2), sampled(700, 7001)},
                     {prompt(engine, 1100, 3), sampled(700, 7002)},
                     {prompt(engine, 1100, 4), sampled(700, 7003)}};
            penalised = {{prompt(engine, 1500, 5), sampled(900, 7101, 0.5F)}, {prompt(engine, 1500, 6), sampled(900, 7102, 0.5F)}};
            // A reply beside a prompt prefilled in 512-token chunks: the second binds with its first chunk, and its later
            // chunks outgrow the pool beside the first request's reply (a fresh request that cannot bind waits instead).
            prompts  = {{prompt(engine, 1500, 7), greedy(900)}, {prompt(engine, 2600, 8), greedy(100)}};
            long_two = {{prompt(engine, 12000, 10), greedy(6000)}, {prompt(engine, 12000, 11), greedy(6000)}};
        }
        using KV = infernix::KvCacheStorage;
        pressure(artifact, two, "greedy, plain, C = 2", {});
        pressure(artifact, two, "greedy, DSpark, C = 2", {.dspark = true});
        pressure(artifact, two, "greedy, plain, --kv-dtype int8", {.kv = KV::Int8Group64});
        pressure(artifact, three, "sampled, DSpark, C = 3", {.dspark = true});
        pressure(artifact, penalised, "sampled with a presence penalty, C = 2", {});
        for (const auto& [label, setup] : std::vector<std::pair<std::string, Setup>>{
                 {"greedy, plain, no prefix cache", {.cache = false}},
                 {"greedy, DSpark, --kv-dtype int8, no prefix cache", {.dspark = true, .kv = KV::Int8Group64, .cache = false}},
                 {"greedy, plain, --a8-activations, no prefix cache", {.cache = false, .a8 = true}}}) {
            const Totals uncached = pressure(artifact, two, label, setup);
            require(uncached.replayed_tokens != 0, label + ": a resume without the prefix cache replays its ledger");
        }
        pressure(artifact, prompts, "a prompt prefilling in 512-token chunks beside a reply", {.chunk = 512});
        if (std::getenv("INFERNIX_INKLING2_NO_SCALE_CASE") == nullptr) {
            pressure(artifact, long_two, "at scale: 32,768-token pool, DSpark, --kv-dtype int8",
                     {.context = 32768, .dspark = true, .kv = KV::Int8Group64});
        }
        if (std::getenv("INFERNIX_INKLING2_NO_OWNER_CONFIG") == nullptr) { owner_configuration(artifact); }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
