// An EngineHost's reused SSD expert tier on a real Qwen3.8-Flash-Next artifact (INFERNIX_QWEN4_ARTIFACT, its n-gram
// volume in INFERNIX_QWEN4_NGRAM; include/infernix/engine.h, docs/maintainer/engine-architecture.md): Engines started
// one after another on one host keep the expert records the last left in the tier's page-locked slots, and give the
// tokens Engines without a host give. The reproducible route (fast paths off), whose output depends only on the request,
// the artifact and the options, whatever the caches hold:
//   A  without a host: a 1,500-token prompt's greedy continuation and a seeded sampled one (the reference).
//   B  on a new host (nothing kept: every record read): the same tokens; then a 4,000-token prompt moves the RAM tier.
//   C  on the host with other options (two lanes, MTP 3): it keeps most of B's records and serves a request.
//   D  on the host with A's options: it keeps most records, reads the rest, and gives A's tokens.
// Between Engines the host keeps their slots (EngineHost::kept_bytes), none while an Engine runs, and release() frees
// them. Each start's time and the records it read and kept are printed. Skips (77) without the artifact.

#include "infernix/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    if (!ok) { ++failures; }
}

std::string first_difference(const Tokens& a, const Tokens& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { return "first difference at " + std::to_string(i); }
    }
    return a.size() == b.size() ? "identical" : "lengths differ";
}

Tokens synthetic_tokens(std::size_t count, std::uint32_t seed) {
    Tokens tokens;
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<infernix::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

infernix::EngineOptions base_options(const char* artifact) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (const char* volume = std::getenv("INFERNIX_QWEN4_NGRAM"); volume != nullptr && *volume != '\0') {
        options.ngram_volume_path = volume;
    }
    options.max_context           = 6144;
    options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(6144); // B's 4,000-token prompt
    options.kv_cache              = infernix::KvCacheStorage::Int8Group64;
    options.fast_paths            = false;
    options.context_cache.enabled = false;
    return options;
}

infernix::RequestOptions request(float temperature, std::uint64_t seed, std::uint32_t tokens) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = temperature;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.top_p          = 0.95F;
    options.execution.sampling.seed           = seed;
    options.stop.include_model_defaults       = false;
    return options;
}

struct Outputs {
    Tokens greedy, sampled;
};

Outputs run(infernix::Engine& engine, const Tokens& prompt) {
    Outputs out;
    out.greedy  = engine.generate(engine.prepare_tokens(prompt), request(0.0F, 0, 48)).generated_token_ids;
    out.sampled = engine.generate(engine.prepare_tokens(prompt), request(1.0F, 7, 48)).generated_token_ids;
    return out;
}

struct Start {
    std::uint64_t read = 0, kept = 0;
};

Start report(const char* name, const infernix::Engine& engine) {
    const infernix::LoadSummary load = engine.load_summary();
    std::printf("  %s: started in %.1f s; expert records read %llu, kept %llu\n", name, load.load_seconds,
                static_cast<unsigned long long>(load.expert_ram_records_read),
                static_cast<unsigned long long>(load.expert_ram_records_kept));
    return {load.expert_ram_records_read, load.expert_ram_records_kept};
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::printf("SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n");
        return 77;
    }
    try {
        const Tokens prompt = synthetic_tokens(1500, 21);
        Outputs reference;
        Start fresh;
        {
            infernix::Engine engine(base_options(artifact));
            fresh     = report("A (no host)", engine);
            reference = run(engine, prompt);
        }
        check(fresh.read > 0 && fresh.kept == 0, "A reads its records and keeps none");
        infernix::EngineHost host;
        Start b;
        {
            infernix::Engine engine(base_options(artifact), host);
            b = report("B (new host)", engine);
            check(host.kept_bytes() == 0, "the host keeps nothing while its Engine runs");
            const Outputs got = run(engine, prompt);
            check(got.greedy == reference.greedy, "B's greedy tokens equal A's (" + first_difference(got.greedy, reference.greedy) + ")");
            check(got.sampled == reference.sampled,
                  "B's sampled tokens equal A's (" + first_difference(got.sampled, reference.sampled) + ")");
            (void)engine.generate(engine.prepare_tokens(synthetic_tokens(4000, 22)), request(0.0F, 0, 8));
        }
        check(b.read > 0 && b.kept == 0, "B on a new host reads every record");
        check(host.kept_bytes() > 0, "the host keeps B's slots");
        {
            infernix::EngineOptions options        = base_options(artifact);
            options.max_concurrency                = 2;
            options.speculative.backend            = infernix::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens       = 3;
            infernix::Engine engine(std::move(options), host);
            const Start c = report("C (two lanes, MTP)", engine);
            check(2 * c.kept > b.read, "C keeps most of B's records (" + std::to_string(c.kept) + " of " + std::to_string(b.read) + ")");
            const Tokens out = engine.generate(engine.prepare_tokens(prompt), request(0.0F, 0, 16)).generated_token_ids;
            check(out.size() == 16, "C serves a request");
        }
        {
            infernix::Engine engine(base_options(artifact), host);
            const Start d = report("D (A's options)", engine);
            check(2 * d.kept > b.read && d.read < d.kept,
                  "D keeps most records and reads fewer (" + std::to_string(d.kept) + " kept, " + std::to_string(d.read) + " read)");
            const Outputs got = run(engine, prompt);
            check(got.greedy == reference.greedy, "D's greedy tokens equal A's (" + first_difference(got.greedy, reference.greedy) + ")");
            check(got.sampled == reference.sampled,
                  "D's sampled tokens equal A's (" + first_difference(got.sampled, reference.sampled) + ")");
        }
        check(host.kept_bytes() > 0, "the host keeps D's slots");
        host.release();
        check(host.kept_bytes() == 0, "release() frees them");
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "engine host checks passed\n" : "%d engine host check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
