// Real-artifact scenarios for the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md
// §13.3). Requires INFERNIX_TEST_ARTIFACT; INFERNIX_HYBRID_REAL_SCENARIO selects one scenario.

#include "test_temp_path.h"
#include "kv_cache_storage.h"
#include "runtime/contract/execution.h"
#include "models/qwen3_5/program/prefix/test_seams.h"
#include "infernix/engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace qwen_testing = infernix::models::qwen3_5::testing;

constexpr std::uint32_t kPrefillChunk = 512;
// Tokens per cached KV block.
constexpr std::uint32_t kBlock = 64;

// INFERNIX_HYBRID_KV_DTYPE selects the KV storage every scenario runs with (default bf16), so the
// Host tier's page records and restores are exercised for each profile.
infernix::KvCacheStorage kv_storage = infernix::KvCacheStorage::BFloat16;

// Deterministic ordinary-vocabulary tokens; the content only has to be reproducible.
std::vector<infernix::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<infernix::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t index = 0; index < count; ++index) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<infernix::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

infernix::EngineOptions hybrid_options(const char* artifact, infernix::SpeculativeBackend backend,
                                     std::uint32_t kv_tokens, std::size_t host_bytes,
                                     std::uint32_t device_slots) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(kv_tokens);
    options.prefill_chunk        = kPrefillChunk;
    options.kv_cache             = kv_storage;
    options.speculative.backend  = backend;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    if (backend == infernix::SpeculativeBackend::Mtp) {
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = infernix::ProposalHead::Optimized;
    } else if (backend == infernix::SpeculativeBackend::DFlash2) {
        options.speculative.draft_tokens = 7;
    }
    options.context_cache.host_capacity_bytes          = host_bytes;
    options.context_cache.hybrid.device_snapshot_slots = device_slots;
    // The cache resumes off the prefill chunk grid, from taps and endpoints, only under the
    // free-cache-resume fast path; these cases test those resumes.
    options.fast_path_candidates = {infernix::FastPath::FreeCacheResume};
    return options;
}

infernix::RequestOptions greedy(std::uint32_t outputs) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    return options;
}

template <typename Call>
bool fails_unavailable(Call&& call) {
    try {
        std::forward<Call>(call)();
    } catch (const infernix::RequestError& error) {
        return error.kind() == infernix::RequestErrorKind::Unavailable;
    }
    return false;
}

struct Observed {
    std::vector<infernix::TokenId> tokens;
    std::uint32_t reused = 0;
    infernix::RuntimeStats stats;
};

// Restore-exact's two prompts (the SSD tier's too): the second shares the first's first 1400 tokens.
std::pair<std::vector<infernix::TokenId>, std::vector<infernix::TokenId>> restore_prompts() {
    std::vector<infernix::TokenId> first = synthetic_tokens(1500, 1);
    std::vector<infernix::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<infernix::TokenId> suffix = synthetic_tokens(300, 2);
    second.insert(second.end(), suffix.begin(), suffix.end());
    return {std::move(first), std::move(second)};
}

// The Device-resident engine on restore_prompts() (16384 KV tokens, a 1 GiB Host tier, eight Device snapshot slots):
// the first prompt, then the second, which resumes from its snapshot. Restore-exact compares its Host restore with it
// and the SSD tier its SSD restore, so one run per drafter serves both, made by whichever runs first (a fresh Engine of
// the same options gives the same greedy tokens, which restore-exact's own comparison already rests on).
Observed device_restore_reference(const char* artifact, infernix::SpeculativeBackend backend) {
    static std::vector<std::pair<infernix::SpeculativeBackend, Observed>> runs;
    for (const auto& [ran, observed] : runs) {
        if (ran == backend) { return observed; }
    }
    const auto prompts = restore_prompts();
    const std::vector<infernix::TokenId>& first  = prompts.first;
    const std::vector<infernix::TokenId>& second = prompts.second;
    infernix::Engine engine(hybrid_options(artifact, backend, 16384, 1ULL << 30, 8));
    (void)engine.generate(engine.prepare_tokens(first), greedy(8));
    const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(second), greedy(24));
    runs.emplace_back(backend, Observed{result.generated_token_ids, result.reused_prompt_tokens, engine.runtime_stats()});
    return runs.back().second;
}

// GitHub #29: after a ladder of long prompts with repeats on a small Host tier, the cache stopped
// reusing for the rest of the session (identical repeats and continued turns prefilled from zero,
// endpoints_created frozen). A tap or endpoint superseded the lineage's previous snapshot before it
// had a Device slot and never undid that when refused, and long snapshots that lost their Host copy
// held every Device slot. The serve scenario that reproduced it (automatic KV capacity, a 512 MiB
// Host tier, two Device snapshot slots), at the Engine: the ladder, then a short prompt and its
// repeat, then a two-turn conversation. A repeat resumes from the snapshot at its last chunk's
// start, and a repeat's endpoint supersedes that snapshot (spec §9.2), so of two repeats one may
// prefill again; the defect is reuse that never returns: every prompt must reuse in at least one
// repeat, both later requests must reuse, and the ladder must keep publishing endpoints. The
// prompts are raw tokens without the chat template's structural taps, so the short one spans two
// chunks.
int exercise_issue29(const char* artifact) {
    infernix::EngineOptions options = hybrid_options(artifact, infernix::SpeculativeBackend::None, 65536,
                                                     512ULL << 20, 2);
    options.kv_capacity   = infernix::KvCapacityPolicy::automatic();
    options.max_context   = 65536;
    options.prefill_chunk = 4096;
    infernix::Engine engine(std::move(options));
    int failures = 0;
    const std::array<std::pair<std::size_t, std::uint32_t>, 3> ladder{{{37000, 11}, {15000, 12}, {25000, 13}}};
    for (const auto& [tokens, seed] : ladder) {
        const std::vector<infernix::TokenId> prompt = synthetic_tokens(tokens, seed);
        const std::uint64_t endpoints_before        = engine.runtime_stats().hybrid_endpoints_created;
        std::uint32_t reused = 0;
        for (int attempt = 0; attempt < 3; ++attempt) {
            const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(prompt), greedy(16));
            std::cout << "issue29: " << tokens << "-token prompt, run " << attempt << ": reused "
                      << result.reused_prompt_tokens << '\n';
            if (attempt > 0) { reused = std::max(reused, result.reused_prompt_tokens); }
        }
        if (reused == 0) {
            std::cerr << "issue29: neither repeat of the " << tokens << "-token prompt reused anything\n";
            ++failures;
        }
        if (engine.runtime_stats().hybrid_endpoints_created == endpoints_before) {
            std::cerr << "issue29: the " << tokens << "-token prompt published no endpoint\n";
            ++failures;
        }
    }
    const std::vector<infernix::TokenId> short_prompt = synthetic_tokens(6000, 14);
    const infernix::RuntimeStats before_short = engine.runtime_stats();
    (void)engine.generate(engine.prepare_tokens(short_prompt), greedy(16));
    const infernix::RuntimeStats after_short = engine.runtime_stats();
    // As in the ladder, one of two repeats may find its tail snapshot superseded by the previous run's
    // endpoint (raw tokens carry no structural tap before the answer, as a chat prompt does).
    std::uint32_t short_reuse = 0;
    for (int attempt = 1; attempt < 3; ++attempt) {
        const std::uint32_t reused = engine.generate(engine.prepare_tokens(short_prompt), greedy(16)).reused_prompt_tokens;
        std::cout << "issue29: short prompt, run " << attempt << ": reused " << reused << '\n';
        short_reuse = std::max(short_reuse, reused);
    }
    // What the short prompt's first run published, for a repeat that reuses nothing.
    const auto delta = [&](std::uint64_t infernix::RuntimeStats::* field) { return after_short.*field - before_short.*field; };
    std::cout << "issue29: short prompt run: taps created " << delta(&infernix::RuntimeStats::hybrid_taps_created)
              << ", skipped " << delta(&infernix::RuntimeStats::hybrid_taps_skipped) << " (no slot "
              << delta(&infernix::RuntimeStats::hybrid_taps_skipped_no_slot) << "), endpoints "
              << delta(&infernix::RuntimeStats::hybrid_endpoints_created) << ", Host image refusals "
              << delta(&infernix::RuntimeStats::hybrid_host_image_write_refusals) << ", block refusals "
              << delta(&infernix::RuntimeStats::hybrid_host_block_write_refusals) << ", slot evictions "
              << delta(&infernix::RuntimeStats::hybrid_device_slot_evictions) << ", evicted blocks "
              << delta(&infernix::RuntimeStats::hybrid_evicted_blocks) << ", snapshots now " << after_short.hybrid_snapshots
              << ", free slots " << after_short.hybrid_free_device_snapshot_slots << '\n';
    std::vector<infernix::TokenId> turn = synthetic_tokens(2500, 15);
    const infernix::GenerationResult first_turn = engine.generate(engine.prepare_tokens(turn), greedy(32));
    turn.insert(turn.end(), first_turn.generated_token_ids.begin(), first_turn.generated_token_ids.end());
    const std::vector<infernix::TokenId> follow = synthetic_tokens(200, 16);
    turn.insert(turn.end(), follow.begin(), follow.end());
    const std::uint32_t turn_reuse = engine.generate(engine.prepare_tokens(turn), greedy(16)).reused_prompt_tokens;
    const infernix::RuntimeStats stats = engine.runtime_stats();
    if (short_reuse == 0 || turn_reuse == 0) {
        std::cerr << "issue29: after the ladder the short prompt's repeats reused at most " << short_reuse
                  << " tokens and the second turn " << turn_reuse << " (both must reuse)\n";
        ++failures;
    }
    std::cout << "issue29: short prompt reused " << short_reuse << ", second turn " << turn_reuse
              << "; endpoints " << stats.hybrid_endpoints_created << ", skipped "
              << stats.hybrid_endpoints_skipped << ", taps skipped for a slot " << stats.hybrid_taps_skipped_no_slot
              << ", slot refusals " << stats.hybrid_device_slot_refusals << '\n';
    return failures;
}

// A later request restores a flexible prompt-tail snapshot the first request captured at its last
// prefill chunk boundary. The Device-resident engine restores it with Device copies; the
// constrained engine is first forced to evict the snapshot image and most of its blocks, so the
// same snapshot comes back through the Host slabs. Both restores carry identical bytes and resume
// at the same frontier with the same chunking, so greedy generation must match token for token.
int exercise_restore_exact(const char* artifact, infernix::SpeculativeBackend backend) {
    const auto prompts = restore_prompts();
    const std::vector<infernix::TokenId>& first  = prompts.first;
    const std::vector<infernix::TokenId>& second = prompts.second;
    // 3900 prompt tokens need 61 of the constrained engine's 64 pages, evicting all but the
    // first few of the first request's cached blocks.
    const std::vector<infernix::TokenId> pressure = synthetic_tokens(3900, 3);

    const Observed device = device_restore_reference(artifact, backend);
    const Observed host   = [&] {
        infernix::Engine engine(hybrid_options(artifact, backend, 4096, 1ULL << 30, 1));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        (void)engine.generate(engine.prepare_tokens(pressure), greedy(4));
        const infernix::GenerationResult result = engine.generate(engine.prepare_tokens(second), greedy(24));
        return Observed{result.generated_token_ids, result.reused_prompt_tokens, engine.runtime_stats()};
    }();

    // The first prompt's chunks end at 512 and 1024; the final chunk [1024, 1500) holds the
    // flexible prompt-tail tap, realized at its start.
    constexpr std::uint32_t kExpectedReuse = 2 * kPrefillChunk;
    int failures                           = 0;
    if (device.reused != kExpectedReuse || host.reused != kExpectedReuse) {
        std::cerr << "restore-exact: reused device=" << device.reused << " host=" << host.reused
                  << ", expected the prompt-tail snapshot at " << kExpectedReuse << '\n';
        ++failures;
    }
    if (device.stats.hybrid_host_image_restores != 0 ||
        device.stats.hybrid_host_block_restores != 0) {
        std::cerr << "restore-exact: the Device-resident engine restored from Host\n";
        ++failures;
    }
    if (host.stats.hybrid_host_image_restores == 0 || host.stats.hybrid_host_block_restores == 0) {
        std::cerr << "restore-exact: the constrained engine did not restore through Host slabs"
                  << " (images=" << host.stats.hybrid_host_image_restores
                  << " blocks=" << host.stats.hybrid_host_block_restores << ")\n";
        ++failures;
    }
    // Copy calls accompany the bytes: at least one per restore and per write, never with no bytes.
    if (host.stats.hybrid_host_restore_copies < host.stats.hybrid_host_image_restores ||
        host.stats.hybrid_host_write_copies == 0 || device.stats.hybrid_host_restore_copies != 0) {
        std::cerr << "restore-exact: copy counts disagree with the transfers (restore copies="
                  << host.stats.hybrid_host_restore_copies << " write copies=" << host.stats.hybrid_host_write_copies
                  << " device-resident restore copies=" << device.stats.hybrid_host_restore_copies << ")\n";
        ++failures;
    }
    if (device.tokens.size() != 24 || device.tokens != host.tokens) {
        std::cerr << "restore-exact: Host and Device restores generated different tokens\n";
        ++failures;
    }
    return failures;
}

// The SSD tier (spec §5.7): restore-exact's prompts on a Host tier of about two snapshot images with
// an SSD tier, and three pressure prompts in between, push the first prompt's prompt-tail snapshot
// (1024) and its blocks out of the Host tier. The second prompt reads them back from the SSD tier and
// generates exactly what the Device-resident engine does (byte copies at every tier).
int exercise_ssd_restore(const char* artifact, infernix::SpeculativeBackend backend) {
    const auto prompts = restore_prompts();
    const std::vector<infernix::TokenId>& first  = prompts.first;
    const std::vector<infernix::TokenId>& second = prompts.second;
    const std::vector<infernix::TokenId> device_tokens = device_restore_reference(artifact, backend).tokens;
    std::filesystem::path directory;
    // A directory the scenario made goes when it ends: declared before its Engines, after their stops.
    struct RemoveOnExit {
        std::filesystem::path path;
        ~RemoveOnExit() {
            std::error_code ignored;
            if (!path.empty()) { std::filesystem::remove_all(path, ignored); }
        }
    } temporary;
    if (const char* dir = std::getenv("INFERNIX_TEST_SSD_DIR"); dir != nullptr && *dir != '\0') {
        directory = dir;
    } else {
        directory = infernix::test::unique_temp_path("hybrid_ssd_tier");
        std::filesystem::create_directories(directory);
        temporary.path = directory;
    }
    const auto ssd_options = [&](const std::filesystem::path& persistent) {
        infernix::EngineOptions options = hybrid_options(artifact, backend, 4096, 480ULL << 20, 1);
        options.context_cache.ssd_capacity_bytes = 8ULL << 30;
        // The write budget's level persists in the test directory: exactness and timing run without it.
        options.context_cache.ssd_write_bytes_per_hour = infernix::ContextCacheOptions::kSsdWritesUnlimited;
        options.context_cache.ssd_directory      = directory;
        if (!persistent.empty()) {
            options.context_cache.hybrid.persistent_file     = persistent;
            options.context_cache.hybrid.persistent_identity = "ssd-kept-test";
        }
        return options;
    };
    const auto pressure = [&](infernix::Engine& engine) {
        for (std::uint32_t seed = 3; seed < 6; ++seed) {
            (void)engine.generate(engine.prepare_tokens(synthetic_tokens(3900, seed)), greedy(4));
        }
    };
    int failures = 0;
    {
        // A restart keeps the SSD tier with --prefix-cache-file (spec §5.5): the first prompt's snapshot, pushed
        // to the SSD tier before the stop, comes back from the kept SSD file without being read at startup and
        // serves the second prompt exactly.
        const std::filesystem::path persistent = infernix::test::unique_temp_path("infernix-hybrid-ssd-kept.bin");
        std::error_code ignored;
        std::filesystem::remove(persistent, ignored);
        {
            infernix::Engine engine(ssd_options(persistent));
            (void)engine.generate(engine.prepare_tokens(first), greedy(8));
            pressure(engine);
        }
        // The restore refills the Host tier with entries the SSD does not hold yet; the promotion waits for their
        // write-back to land (SsdTier::Promotion::Pending) rather than falling back to a cold prefill.
        std::optional<infernix::Engine> holder;
        holder.emplace(ssd_options(persistent));
        infernix::Engine& restarted = *holder;
        const infernix::LoadSummary::PrefixCacheRestore restore = restarted.load_summary().prefix_cache;
        std::cout << "ssd-restore (restart): restored " << restore.blocks << " blocks and " << restore.snapshots
                  << " snapshots, " << restore.ssd_blocks << " and " << restore.ssd_snapshots
                  << " of them on the SSD tier (" << restore.message << restore.ssd_message << ")\n";
        if (!restore.restored || restore.ssd_snapshots == 0 || !restore.ssd_message.empty()) {
            std::cerr << "ssd-restore (restart): the restart did not take back the SSD tier's snapshots\n";
            ++failures;
        }
        const infernix::RuntimeStats before   = restarted.runtime_stats();
        const infernix::GenerationResult warm = restarted.generate(restarted.prepare_tokens(second), greedy(24));
        const infernix::RuntimeStats after    = restarted.runtime_stats();
        std::cout << "ssd-restore (restart): reused " << warm.reused_prompt_tokens << " (blocks held "
                  << warm.cached_block_tokens << "), " << after.hybrid_ssd_promotions - before.hybrid_ssd_promotions
                  << " SSD promotions, " << after.hybrid_ssd_failures << " SSD failures\n";
        if (after.hybrid_ssd_promotions <= before.hybrid_ssd_promotions ||
            warm.reused_prompt_tokens != 2 * kPrefillChunk) {
            std::cerr << "ssd-restore (restart): the second prompt did not resume from the kept SSD tier (reused "
                      << warm.reused_prompt_tokens << ")\n";
            ++failures;
        }
        if (warm.generated_token_ids != device_tokens || after.hybrid_ssd_failures != 0) {
            std::cerr << "ssd-restore (restart): the kept SSD restore generated other tokens or failed a transfer\n";
            ++failures;
        }
        // The stop saves the file again, so it goes after the Engine.
        holder.reset();
        std::filesystem::remove(persistent, ignored);
        if (std::filesystem::exists(persistent)) {
            std::cerr << "ssd-restore (restart): the kept SSD file was left in the temp directory\n";
            ++failures;
        }
    }
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().filename().string().starts_with("infernix-context-kept-")) {
            std::error_code ignored;
            std::filesystem::remove(entry.path(), ignored);
        }
    }
    infernix::Engine engine(ssd_options({}));
    (void)engine.generate(engine.prepare_tokens(first), greedy(8));
    pressure(engine);
    const infernix::RuntimeStats before   = engine.runtime_stats();
    const infernix::GenerationResult warm = engine.generate(engine.prepare_tokens(second), greedy(24));
    const infernix::RuntimeStats after    = engine.runtime_stats();
    std::cout << "ssd-restore: " << after.hybrid_ssd_write_bytes / 1000000 << " MB written ("
              << after.hybrid_ssd_image_writes << " images, " << after.hybrid_ssd_block_writes << " blocks), "
              << after.hybrid_ssd_read_bytes / 1000000 << " MB read in " << after.hybrid_ssd_promotions
              << " promotions, " << after.hybrid_ssd_host_image_demotions << " image demotions, Host snapshot evictions "
              << after.hybrid_host_snapshot_evictions << ", reused " << warm.reused_prompt_tokens << '\n';
    if (after.hybrid_ssd_promotions <= before.hybrid_ssd_promotions) {
        std::cerr << "ssd-restore: the second prompt did not read its source from the SSD tier\n";
        ++failures;
    }
    if (warm.reused_prompt_tokens != 2 * kPrefillChunk) {
        std::cerr << "ssd-restore: reused " << warm.reused_prompt_tokens << ", expected the prompt-tail snapshot at "
                  << 2 * kPrefillChunk << '\n';
        ++failures;
    }
    if (warm.generated_token_ids.size() != 24 || warm.generated_token_ids != device_tokens) {
        std::cerr << "ssd-restore: the SSD restore generated other tokens than the Device-resident engine\n";
        ++failures;
    }
    if (after.hybrid_ssd_failures != 0) {
        std::cerr << "ssd-restore: " << after.hybrid_ssd_failures << " SSD transfers failed\n";
        ++failures;
    }
    return failures;
}

// A restarted Engine resumes from the Host tier its predecessor saved when stopped mid-generation:
// the restored snapshot is the same bytes, so greedy generation matches an Engine that never
// restarted. A file written for a different build identity is ignored, a Host tier one slab
// smaller than the file restores some but not all of its snapshots, and an abandoned save keeps
// the previous file and leaves no temporary one.
int exercise_persist(const char* artifact) {
    const std::filesystem::path file =
        infernix::test::unique_temp_path("infernix-hybrid-persist-real-test.bin");
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const std::vector<infernix::TokenId> first = synthetic_tokens(1500, 4);
    std::vector<infernix::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<infernix::TokenId> suffix = synthetic_tokens(300, 5);
    second.insert(second.end(), suffix.begin(), suffix.end());
    const auto options = [&](const char* identity) {
        infernix::EngineOptions result =
            hybrid_options(artifact, infernix::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
        if (identity != nullptr) {
            result.context_cache.hybrid.persistent_file     = file;
            result.context_cache.hybrid.persistent_identity = identity;
        }
        return result;
    };

    std::vector<infernix::TokenId> reference;
    {
        infernix::Engine engine(options(nullptr));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        reference = engine.generate(engine.prepare_tokens(second), greedy(24)).generated_token_ids;
    }
    int failures                      = 0;
    std::uint32_t retry_reused_tokens = 0;
    {
        // The Engine stops the way infernix-serve does: stop() while a generation is still far from
        // its output limit. It fails as Unavailable instead of running on, new work is refused,
        // and the Host tier is still saved.
        infernix::Engine saver(options("persist-test"));
        (void)saver.generate(saver.prepare_tokens(first), greedy(8));
        // A retry resumes from the tap the first request captured, and its endpoint supersedes
        // that tap: the file must still keep it (issue #59), or `second` below and a retry after
        // the restart prefill from an earlier tap.
        retry_reused_tokens =
            saver.generate(saver.prepare_tokens(first), greedy(8)).reused_prompt_tokens;
        if (retry_reused_tokens == 0) {
            std::cerr << "persist: a retry did not resume from the first request's tap\n";
            ++failures;
        }
        infernix::GenerationHandle running =
            saver.submit(saver.prepare_tokens(synthetic_tokens(200, 6)), greedy(3500));
        std::this_thread::sleep_for(std::chrono::seconds(2));
        saver.stop();
        if (!fails_unavailable([&] { (void)running.wait(); })) {
            std::cerr << "persist: a running generation was not ended as Unavailable by stop()\n";
            ++failures;
        }
        // The answer comes before the Host tier is saved (seconds for a large one), so the file
        // is not in place yet.
        if (std::filesystem::exists(file)) {
            std::cerr << "persist: stop() answered the running generation only after the save\n";
            ++failures;
        }
        if (!fails_unavailable(
                [&] { (void)saver.submit(saver.prepare_tokens(second), greedy(4)); })) {
            std::cerr << "persist: a stopped Engine accepted a request\n";
            ++failures;
        }
        // A request for no output tokens is answered without the queue, but not by a stopped Engine.
        if (!fails_unavailable(
                [&] { (void)saver.submit(saver.prepare_tokens(second), greedy(0)).wait(); })) {
            std::cerr << "persist: a stopped Engine answered a zero-token request\n";
            ++failures;
        }
    }
    if (!std::filesystem::exists(file)) {
        std::cerr << "persist: the Engine did not save its Host tier\n";
        return 1;
    }
    // The saver's file, for the smaller Host tier below (the loader saves over `file`).
    const std::filesystem::path partial =
        infernix::test::unique_temp_path("infernix-hybrid-persist-real-test-partial.bin");
    std::filesystem::copy_file(file, partial, std::filesystem::copy_options::overwrite_existing);
    // The file read is published as one startup phase: the whole file, never a failure (a failed
    // phase would be logged as a failed startup).
    std::vector<infernix::StartupEvent> load_events;
    const auto observe = [&](infernix::EngineOptions result) {
        load_events.clear();
        result.startup_observer.callback = [&](const infernix::StartupEvent& event) {
            if (event.phase == infernix::StartupPhase::PrefixCacheLoad) {
                load_events.push_back(event);
            }
        };
        return result;
    };
    const auto observed  = [&](const char* identity) { return observe(options(identity)); };
    const auto one_phase = [&](std::uint64_t expected_bytes) {
        bool monotonic = true;
        for (std::size_t index = 1; index < load_events.size(); ++index) {
            monotonic = monotonic && load_events[index].current >= load_events[index - 1].current &&
                        load_events[index].current <= expected_bytes;
        }
        return load_events.size() >= 2 &&
               load_events.front().status == infernix::StartupStatus::Begin &&
               load_events.front().total == expected_bytes &&
               load_events.back().status == infernix::StartupStatus::Complete &&
               load_events.back().current == expected_bytes && monotonic &&
               std::all_of(load_events.begin() + 1, load_events.end() - 1,
                           [](const infernix::StartupEvent& event) {
                               return event.status == infernix::StartupStatus::Progress;
                           });
    };
    std::uint64_t required_host_bytes = 0;
    std::uint64_t saved_snapshots     = 0;
    {
        const std::uint64_t file_bytes = std::filesystem::file_size(file);
        infernix::Engine loader(observed("persist-test"));
        if (!one_phase(file_bytes)) {
            std::cerr << "persist: the file read was not published as one complete phase ("
                      << load_events.size() << " events)\n";
            ++failures;
        }
        const infernix::LoadSummary summary = loader.load_summary();
        const infernix::GenerationResult resumed =
            loader.generate(loader.prepare_tokens(second), greedy(24));
        const infernix::RuntimeStats stats                     = loader.runtime_stats();
        const infernix::LoadSummary::PrefixCacheRestore& cache = summary.prefix_cache;
        if (!cache.restored || cache.snapshots == 0 || cache.snapshots != cache.saved_snapshots ||
            cache.blocks != cache.saved_blocks || cache.required_host_bytes > cache.host_bytes) {
            std::cerr << "persist: the saved Host tier was not wholly restored (" << cache.message
                      << ")\n";
            ++failures;
        }
        required_host_bytes = cache.required_host_bytes;
        saved_snapshots     = cache.saved_snapshots;
        if (resumed.reused_prompt_tokens != 2 * kPrefillChunk ||
            stats.hybrid_host_block_restores == 0 || stats.hybrid_host_image_restores == 0) {
            std::cerr << "persist: the restarted Engine reused " << resumed.reused_prompt_tokens
                      << " tokens (blocks restored " << stats.hybrid_host_block_restores << ")\n";
            ++failures;
        }
        if (resumed.generated_token_ids != reference) {
            std::cerr << "persist: the restarted Engine generated different tokens\n";
            ++failures;
        }
        const std::uint32_t retried =
            loader.generate(loader.prepare_tokens(first), greedy(8)).reused_prompt_tokens;
        if (retried < retry_reused_tokens) {
            std::cerr << "persist: a retry after the restart reused " << retried << " tokens, before it "
                      << retry_reused_tokens << "\n";
            ++failures;
        }
    }
    {
        infernix::Engine foreign(observed("another-build"));
        const infernix::LoadSummary summary = foreign.load_summary();
        if (!load_events.empty()) {
            std::cerr << "persist: a file from another build published a load phase\n";
            ++failures;
        }
        if (summary.prefix_cache.restored || summary.prefix_cache.message.empty() ||
            foreign.generate(foreign.prepare_tokens(second), greedy(4)).reused_prompt_tokens != 0) {
            std::cerr << "persist: a file from another build was not ignored\n";
            ++failures;
        }
    }
    if (saved_snapshots < 2) {
        std::cerr << "persist: the saved tier holds " << saved_snapshots
                  << " snapshots; a smaller tier cannot be checked\n";
        ++failures;
    } else {
        // One slab short of the file: whole snapshots are dropped, never all of them, and only the
        // chosen entries are read.
        infernix::EngineOptions small                    = observe(hybrid_options(
            artifact, infernix::SpeculativeBackend::None, 16384, required_host_bytes - 1, 8));
        small.context_cache.hybrid.persistent_file     = partial;
        small.context_cache.hybrid.persistent_identity = "persist-test";
        const std::uint64_t file_bytes                 = std::filesystem::file_size(partial);
        infernix::Engine engine(std::move(small));
        const infernix::LoadSummary::PrefixCacheRestore cache = engine.load_summary().prefix_cache;
        if (!cache.restored || cache.snapshots == 0 || cache.snapshots >= cache.saved_snapshots ||
            cache.blocks > cache.saved_blocks || cache.required_host_bytes <= cache.host_bytes) {
            std::cerr << "persist: a smaller Host tier restored " << cache.snapshots << " of "
                      << cache.saved_snapshots << " snapshots (" << cache.message << ")\n";
            ++failures;
        }
        if (!one_phase(cache.bytes) || cache.bytes >= file_bytes) {
            std::cerr << "persist: a smaller Host tier did not read only what it restored ("
                      << cache.bytes << " of " << file_bytes << " bytes)\n";
            ++failures;
        }
        (void)engine.generate(engine.prepare_tokens(second), greedy(4));
    }
    // A Ctrl+C during infernix-serve's stop abandons the save: the unfinished file is deleted and
    // the previous file stays as it was, whether the save was writing or had not begun.
    std::filesystem::path temporary = file;
    temporary += ".tmp";
    const auto unchanged = [&, bytes = std::filesystem::file_size(file),
                            time = std::filesystem::last_write_time(file)] {
        std::error_code error;
        return !std::filesystem::exists(temporary, error) &&
               std::filesystem::file_size(file, error) == bytes &&
               std::filesystem::last_write_time(file, error) == time;
    };
    using Abandon = infernix::PrefixCacheSaveControl::Abandon;
    {
        infernix::EngineOptions writing                = options("persist-test");
        const infernix::PrefixCacheSaveControl control = writing.context_cache.hybrid.persistent_save;
        std::optional<Abandon> result;
        {
            infernix::Engine engine(std::move(writing));
            (void)engine.generate(engine.prepare_tokens(first), greedy(8));
            std::thread interrupt([&] {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                std::error_code error;
                while (!std::filesystem::exists(temporary, error) &&
                       std::chrono::steady_clock::now() < deadline) {}
                result = control.abandon(std::chrono::seconds(5));
            });
            engine.stop();
            interrupt.join();
        }
        if (result != Abandon::Unsaved || !unchanged()) {
            std::cerr << "persist: a save abandoned while writing left its file or replaced the "
                         "previous one\n";
            ++failures;
        }
    }
    {
        infernix::EngineOptions early                  = options("persist-test");
        const infernix::PrefixCacheSaveControl control = early.context_cache.hybrid.persistent_save;
        Abandon result                               = Abandon::StillWriting;
        {
            infernix::Engine engine(std::move(early));
            (void)engine.generate(engine.prepare_tokens(first), greedy(8));
            result = control.abandon(std::chrono::milliseconds(0));
        }
        if (result != Abandon::Unsaved || !unchanged()) {
            std::cerr << "persist: a save abandoned before it began still wrote a file\n";
            ++failures;
        }
    }
    std::filesystem::remove(file, ignored);
    std::filesystem::remove(partial, ignored);
    return failures;
}

// Two requests submitted together share a prefix nothing has cached yet. The second waits for
// the first's snapshot where the prompts diverge and resumes from it instead of prefilling the
// prefix again (spec §12.2).
// - On a chunk boundary both runs compute the same chunks, and the leader stops after its first
//   token so the follower never shares a decode round: the follower's output must equal an
//   uncached run's.
// - Inside a block, the snapshot goes on the block boundary below, where it publishes at once.
// - A leader with a long prompt past the shared prefix (`leader_suffix` of several chunks): the follower is
//   admitted when the leader's snapshot at the divergence publishes, not when the leader's whole prefill ends
//   (2026-11-15: admission was re-run only at the end of the leader's prompt, so the follower queued for all of it).
int exercise_coalesce(const char* artifact, std::uint32_t shared_tokens, bool compare_output,
                      std::uint32_t leader_suffix = 40) {
    const std::vector<infernix::TokenId> shared        = synthetic_tokens(shared_tokens, 6);
    std::vector<infernix::TokenId> first               = shared;
    std::vector<infernix::TokenId> second              = shared;
    const std::vector<infernix::TokenId> first_suffix  = synthetic_tokens(leader_suffix, 7);
    const std::vector<infernix::TokenId> second_suffix = synthetic_tokens(40, 8);
    first.insert(first.end(), first_suffix.begin(), first_suffix.end());
    second.insert(second.end(), second_suffix.begin(), second_suffix.end());
    const std::uint32_t expected = shared_tokens / kBlock * kBlock;

    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    options.max_concurrency = 2;
    if (shared_tokens + leader_suffix + 64U > options.max_context) { options.max_context = 8192; }
    std::vector<infernix::TokenId> reference;
    if (compare_output) {
        infernix::Engine engine(options);
        reference = engine.generate(engine.prepare_tokens(second), greedy(16)).generated_token_ids;
    }
    infernix::Engine engine(options);
    infernix::GenerationHandle leader    = engine.submit(engine.prepare_tokens(first), greedy(1));
    infernix::GenerationHandle follower  = engine.submit(engine.prepare_tokens(second), greedy(16));
    const infernix::GenerationResult led = leader.wait();
    const infernix::GenerationResult followed = follower.wait();
    int failures                            = 0;
    if (led.reused_prompt_tokens != 0 || followed.reused_prompt_tokens != expected) {
        std::cerr << "coalesce " << shared_tokens << ": the leader reused "
                  << led.reused_prompt_tokens << " tokens and the follower "
                  << followed.reused_prompt_tokens << " (expected 0 and " << expected << ")\n";
        ++failures;
    }
    if (compare_output && followed.generated_token_ids != reference) {
        std::cerr << "coalesce " << shared_tokens
                  << ": the follower generated different tokens than an uncached run\n";
        ++failures;
    }
    if (leader_suffix >= kPrefillChunk) {
        // The follower waits for the leader's prefill of the shared prefix only: under half of the leader's
        // prompt here, and later chunks cost more. Three quarters of the leader's prefill time bounds it with room.
        const double waited  = followed.engine_timing.queue_wait_seconds;
        const double prefill = led.timings.prefill_seconds;
        std::cout << "coalesce " << shared_tokens << "+" << leader_suffix << ": the follower queued " << waited
                  << " s of the leader's " << prefill << " s prefill\n";
        if (!(waited < 0.75 * prefill)) {
            std::cerr << "coalesce " << shared_tokens << "+" << leader_suffix
                      << ": the follower was admitted only once the leader's whole prefill had run\n";
            ++failures;
        }
    }
    return failures;
}

// A burst of identical prompts (NInfer #414 shipped exactly this failure: every request failed and
// the engine stayed unavailable): four copies of one prompt at C = 4 with MTP, at a length that is
// not a block multiple, submitted together, so later copies coalesce on the first one's snapshot or
// endpoint while it is still being published (candidates §13 NN-05). Four rounds (§15 NN-08: NInfer's
// failure came from an exact hit pinning a resident writer's partial tail page, which later rounds
// meet once the earlier rounds' entries exist). Every request completes, each generates what the
// prompt generates alone (greedy on the reproducible route, which no other request, batch or cache
// state changes), and the engine serves the prompt again afterwards.
int exercise_identical_burst(const char* artifact) {
    const std::vector<infernix::TokenId> prompt = synthetic_tokens(2 * kPrefillChunk + kBlock / 2 + 37, 13);
    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::Mtp, 16384, 1ULL << 30, 8);
    options.max_concurrency      = 4;
    options.max_pending_requests = 4;
    options.fast_paths           = false;
    options.fast_path_candidates.clear();
    std::vector<infernix::TokenId> alone;
    {
        infernix::Engine engine(options);
        alone = engine.generate(engine.prepare_tokens(prompt), greedy(32)).generated_token_ids;
    }
    infernix::Engine engine(options);
    int failures = 0;
    for (int round = 0; round < 4; ++round) {
        std::vector<infernix::GenerationHandle> burst;
        for (int i = 0; i < 4; ++i) { burst.push_back(engine.submit(engine.prepare_tokens(prompt), greedy(32))); }
        for (std::size_t i = 0; i < burst.size(); ++i) {
            const infernix::GenerationResult result = burst[i].wait();
            if (result.generated_token_ids != alone) {
                std::cerr << "identical burst: round " << round << " request " << i << " generated "
                          << result.generated_token_ids.size() << " tokens unlike the prompt alone (reused "
                          << result.reused_prompt_tokens << ")\n";
                ++failures;
            }
        }
    }
    const infernix::GenerationResult after = engine.generate(engine.prepare_tokens(prompt), greedy(32));
    if (after.generated_token_ids != alone) {
        std::cerr << "identical burst: the engine served the prompt differently after the burst\n";
        ++failures;
    }
    return failures;
}

// Two requests submitted together with unrelated prompts: the second is admitted between the
// first one's prefill chunks, so their chunks interleave on shared execution scalars. Each must
// generate what it generates alone. The second stops after its first token so the first decodes
// alone, as in its reference.
int exercise_interleaved(const char* artifact) {
    const std::vector<infernix::TokenId> first  = synthetic_tokens(3 * kPrefillChunk + 100, 9);
    const std::vector<infernix::TokenId> second = synthetic_tokens(3 * kPrefillChunk + 60, 10);
    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    options.max_concurrency = 2;
    std::vector<infernix::TokenId> first_alone;
    std::vector<infernix::TokenId> second_alone;
    {
        infernix::Engine engine(options);
        first_alone = engine.generate(engine.prepare_tokens(first), greedy(16)).generated_token_ids;
        second_alone =
            engine.generate(engine.prepare_tokens(second), greedy(1)).generated_token_ids;
    }
    infernix::Engine engine(options);
    infernix::GenerationHandle a = engine.submit(engine.prepare_tokens(first), greedy(16));
    infernix::GenerationHandle b = engine.submit(engine.prepare_tokens(second), greedy(1));
    const infernix::GenerationResult first_together  = a.wait();
    const infernix::GenerationResult second_together = b.wait();
    int failures                                   = 0;
    if (first_together.generated_token_ids != first_alone ||
        second_together.generated_token_ids != second_alone) {
        std::cerr
            << "interleaved: concurrent prefills generated different tokens than alone (first "
            << (first_together.generated_token_ids == first_alone ? "same" : "differs")
            << ", second "
            << (second_together.generated_token_ids == second_alone ? "same" : "differs") << ")\n";
        ++failures;
    }
    return failures;
}

// Requests cancellation once the first prefill chunk is reported. Non-final prefill steps return
// without waiting for the Device, so the cancellation usually reaches the Program with a later
// chunk still queued.
class CancelAfterFirstChunk final : public infernix::OutputSink {
public:
    void start(infernix::GenerationStart) override {}

    void progress(infernix::PromptProgress progress) override {
        if (progress.processed_prompt_tokens != 0) { requested_.store(true); }
    }

    void timing(infernix::GenerationTimingObservation) override {}

    void tool_call(infernix::ToolCallDelta) override {}

    void publish(infernix::OutputDelta) override {}

    [[nodiscard]] bool requested() const { return requested_.load(); }

private:
    std::atomic<bool> requested_{false};
};

// A request cancelled mid-prefill frees its lane and publishes its committed prefix. The lane's
// next request, on an unrelated prompt, must generate what it generates on an Engine that never
// saw the cancellation, and a request extending the cancelled prompt must resume from that prefix
// and generate what an uncached run generates.
int exercise_cancel_prefill(const char* artifact, infernix::SpeculativeBackend backend) {
    // Longer than idle-wide-prefill's widest first step (4,096 tokens), so the request is still
    // prefilling when the cancellation arrives after its first step.
    const std::vector<infernix::TokenId> prompt = synthetic_tokens(12 * kPrefillChunk + 100, 11);
    std::vector<infernix::TokenId> extended     = prompt;
    const std::vector<infernix::TokenId> suffix = synthetic_tokens(60, 12);
    extended.insert(extended.end(), suffix.begin(), suffix.end());
    const std::vector<infernix::TokenId> unrelated = synthetic_tokens(2 * kPrefillChunk + 30, 13);
    infernix::EngineOptions options = hybrid_options(artifact, backend, 16384, 1ULL << 30, 8);
    options.max_context             = 8192; // the prompt is longer than the other cases' 4,096

    std::vector<infernix::TokenId> extended_uncached;
    std::vector<infernix::TokenId> unrelated_alone;
    {
        infernix::Engine engine(options);
        extended_uncached =
            engine.generate(engine.prepare_tokens(extended), greedy(16)).generated_token_ids;
        unrelated_alone =
            engine.generate(engine.prepare_tokens(unrelated), greedy(16)).generated_token_ids;
    }
    infernix::Engine engine(options);
    CancelAfterFirstChunk sink;
    infernix::GenerationHandle handle =
        engine.submit(engine.prepare_tokens(prompt), greedy(16),
                      infernix::OutputConsumerMode::Streaming, {.prompt_progress = true});
    const infernix::GenerationResult cancelled =
        handle.wait(&sink, infernix::CancellationView([&sink] { return sink.requested(); }));
    const infernix::GenerationResult next =
        engine.generate(engine.prepare_tokens(unrelated), greedy(16));
    const infernix::GenerationResult resumed =
        engine.generate(engine.prepare_tokens(extended), greedy(16));

    const char* name = backend == infernix::SpeculativeBackend::DFlash2 ? "cancel-prefill-dflash2"
                                                                      : "cancel-prefill";
    int failures     = 0;
    if (cancelled.finish_reason != infernix::FinishReason::Cancelled ||
        !cancelled.generated_token_ids.empty()) {
        std::cerr << name << ": the request was not cancelled during prefill (reason="
                  << static_cast<int>(cancelled.finish_reason)
                  << " generated=" << cancelled.generated_token_ids.size() << ")\n";
        ++failures;
    }
    std::cout << name << ": the extended prompt resumed at " << resumed.reused_prompt_tokens
              << " of the cancelled request's " << prompt.size() << " prompt tokens\n";
    if (next.generated_token_ids != unrelated_alone) {
        std::cerr << name << ": the lane's next request generated different tokens than alone\n";
        ++failures;
    }
    if (resumed.reused_prompt_tokens == 0 || resumed.reused_prompt_tokens % kBlock != 0 ||
        resumed.reused_prompt_tokens > prompt.size()) {
        std::cerr << name << ": the extended prompt reused " << resumed.reused_prompt_tokens
                  << " tokens, expected the cancelled request's committed prefix\n";
        ++failures;
    }
    if (resumed.generated_token_ids.size() != 16 ||
        resumed.generated_token_ids != extended_uncached) {
        std::cerr << name << ": the extended prompt resumed from the cancelled prefix ("
                  << resumed.reused_prompt_tokens
                  << " tokens) generated different tokens than an uncached run\n";
        ++failures;
    }
    return failures;
}

infernix::ChatMessage text_message(infernix::ChatRole role, std::string text) {
    infernix::ChatMessage message;
    message.role = role;
    message.parts.push_back(infernix::MessagePart{
        .kind = infernix::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

// Protocol adapters shape the hints the way OpenAI default caching does: the Engine's automatic
// Legacy shared-prefix candidates are disabled and an automatic marker follows the last message.
// Hybrid taps must keep their semantic placement under those hints.
bool protocol_hints = false;

infernix::PromptInput conversation(std::vector<infernix::ChatMessage> messages) {
    infernix::PromptInput input;
    input.messages                = std::move(messages);
    input.options.enable_thinking = false;
    if (protocol_hints) {
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        input.context_cache.markers.push_back(infernix::PromptCacheMarker{
            .after_message_count = static_cast<std::uint32_t>(input.messages.size()),
            .kind                = infernix::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence            = infernix::SharedCandidateEvidence::DefaultAutomatic,
            .location            = infernix::PromptCacheMarkerLocation::MessageBoundary,
        });
    }
    return input;
}

// Exact semantic taps: the next turn of a conversation resumes at the previous turn's generation
// opener (not 64-token-floored), and a new conversation with the same leading system block resumes
// at that block's end.
int exercise_turns(const char* artifact, std::size_t host_bytes) {
    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::None, 8192, host_bytes, 0);
    options.context_cache.hybrid.device_snapshot_slots.reset();
    infernix::Engine engine(std::move(options));

    std::string system = "You are a careful assistant for a small engineering team.";
    for (int sentence = 0; sentence < 60; ++sentence) {
        system += " Rule " + std::to_string(sentence) +
                  ": answer precisely, cite the relevant component, and keep replies short.";
    }
    const infernix::ChatMessage leading = text_message(infernix::ChatRole::System, system);
    const infernix::ChatMessage first_user =
        text_message(infernix::ChatRole::User, "Name three prime numbers.");

    const infernix::GenerationResult first =
        engine.generate(engine.prepare(conversation({leading, first_user})), greedy(24));
    const infernix::GenerationResult second = engine.generate(
        engine.prepare(conversation({leading, first_user,
                                     text_message(infernix::ChatRole::Assistant, first.content),
                                     text_message(infernix::ChatRole::User, "Now name two more.")})),
        greedy(8));
    const infernix::GenerationResult other = engine.generate(
        engine.prepare(
            conversation({leading, text_message(infernix::ChatRole::User, "What is a hash table?")})),
        greedy(8));
    const std::uint32_t other_user_tokens =
        engine
            .prepare(conversation({text_message(infernix::ChatRole::User, "What is a hash table?")}))
            .summary()
            .prompt_tokens;

    int failures = 0;
    // The next turn resumes at the first turn's generation opener, a few tokens before its end, or
    // at the earliest boundary of the opener's cluster (the system-block end after a short user
    // turn), at most the 64-token tap separation earlier.
    if (second.reused_prompt_tokens + 64U + 16U < first.prompt.prompt_tokens ||
        second.reused_prompt_tokens >= second.prompt.prompt_tokens) {
        std::cerr << "turns: second turn reused " << second.reused_prompt_tokens
                  << " of the first turn's " << first.prompt.prompt_tokens
                  << " prompt tokens; expected the generation opener\n";
        ++failures;
    }
    // The shared system block ends where the new user turn begins.
    if (other.reused_prompt_tokens + other_user_tokens + 16U < other.prompt.prompt_tokens) {
        std::cerr << "turns: a new conversation reused " << other.reused_prompt_tokens << " of "
                  << other.prompt.prompt_tokens << " tokens; expected the system block end\n";
        ++failures;
    }
    const infernix::RuntimeStats stats = engine.runtime_stats();
    if (stats.hybrid_taps_created == 0 || stats.hybrid_snapshot_hits < 2) {
        std::cerr << "turns: taps=" << stats.hybrid_taps_created
                  << " hits=" << stats.hybrid_snapshot_hits << '\n';
        ++failures;
    }
    return failures;
}

// QK-02, the pair window: a chat turn whose generation opener tap falls in its prompt's last pass takes the tap
// inside that pass. With the cut (testing::set_pair_windows(false)) the turn's last pass begins at the opener; with
// the pair no pass does, the next turn still resumes at the opener, and it generates the tokens it generates
// resuming from the cut's tap: the tap's state (the GDN convolution and recurrence split at the opener, the
// continuation hidden, the drafter's ring) is the cut's, where the head's kernels take the same routes at the
// pair's width as at its own (heads of ~200 columns here; a resumed 77-column head paired to 84 changed its next
// turn's tokens in a served A/B). Two turns: a cold one, then one resumed from its tap that pairs its own opener.
// Each arm runs on a fresh Engine. A KV window (vq2, q4k-vq2) keeps the cut: both arms cut.
// The pair-window exercises' conversation: a cold turn, a short follow-up, and a long resumed turn with its reply.
struct PairConversation {
    infernix::ChatMessage system, first_user, reply, second_user, follow_user, follow_reply;
    std::string question;
};

PairConversation pair_conversation() {
    PairConversation p;
    p.system = text_message(infernix::ChatRole::System, "You are a concise assistant for a hardware shop. Answer in plain words.");
    // Longer than the 64-token cluster distance, so the opener keeps its own exact tap past the system block's.
    p.question = "A customer asks which of three drills to buy.";
    for (int i = 0; i < 6; ++i) {
        p.question += " Drill " + std::to_string(i) + " has a different battery, chuck size, weight and warranty, and the "
                      "customer mostly hangs shelves and assembles furniture at home.";
    }
    p.first_user = text_message(infernix::ChatRole::User, p.question);
    p.reply = text_message(infernix::ChatRole::Assistant, "Take the lightest one with a two-year warranty and a spare battery.");
    p.second_user = text_message(infernix::ChatRole::User, "And which bits should go with it?");
    // The resumed turn's question: long enough that its opener lies well past where it resumes (its own exact tap,
    // with a wide head).
    std::string follow_up = "Thanks. Now the same customer also asks about storage.";
    for (int i = 0; i < 6; ++i) {
        follow_up += " Case " + std::to_string(i) + " is a plastic box with foam inserts, a metal clasp, room for twenty "
                     "bits and a belt clip, sold at a slightly different price.";
    }
    p.follow_user  = text_message(infernix::ChatRole::User, follow_up);
    p.follow_reply = text_message(infernix::ChatRole::Assistant, "The second case: foam inserts keep the bits from rattling.");
    return p;
}

int exercise_pair_window(const char* artifact, infernix::SpeculativeBackend backend) {
    const std::string name = std::string("pair-window") +
                             (backend == infernix::SpeculativeBackend::DFlash2 ? " (dflash2)"
                              : backend == infernix::SpeculativeBackend::Mtp   ? " (mtp)"
                                                                               : "");
    const PairConversation conv = pair_conversation();
    const infernix::ChatMessage &system = conv.system, &first_user = conv.first_user, &reply = conv.reply,
                                &second_user = conv.second_user, &follow_user = conv.follow_user,
                                &follow_reply = conv.follow_reply;
    const std::string& question = conv.question;
    const bool kv_window = kv_storage == infernix::KvCacheStorage::Vq2 || kv_storage == infernix::KvCacheStorage::Q4KeyVq2Value;
    struct Arm {
        std::vector<std::uint32_t> bases, follow_bases; // where each text pass of the two turns began
        std::uint32_t first_tokens = 0, follow_tokens = 0, reused = 0, follow_reused = 0, last_reused = 0;
        std::uint64_t refusals = 0; // Device snapshot slots refused during the resumed turn
        std::vector<infernix::TokenId> second, last;
    };
    const auto run = [&](bool pair) {
        qwen_testing::set_pair_windows(pair);
        infernix::EngineOptions options = hybrid_options(artifact, backend, 8192, 1ULL << 30, 0);
        options.context_cache.hybrid.device_snapshot_slots.reset();
        infernix::Engine engine(std::move(options));
        Arm arm;
        const auto passes = [] {
            std::vector<std::uint32_t> bases;
            for (const qwen_testing::TextPass& pass : qwen_testing::text_passes()) { bases.push_back(pass.base); }
            qwen_testing::record_text_passes(false);
            return bases;
        };
        qwen_testing::record_text_passes(true);
        const infernix::GenerationResult first = engine.generate(engine.prepare(conversation({system, first_user})), greedy(8));
        arm.bases = passes();
        const infernix::GenerationResult second =
            engine.generate(engine.prepare(conversation({system, first_user, reply, second_user})), greedy(24));
        // The resumed turn (from the first turn's opener tap) whose own opener pairs, then the turn after it. A turn
        // that finds every Device snapshot slot still held (the previous reply's snapshots landing on the Host) is
        // cut instead (hybrid spec §7.10); the pause, as a client's next turn arrives, makes the pair the usual case
        // without depending on it: a cut is accepted where a slot refusal explains it.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        arm.refusals = engine.runtime_stats().hybrid_device_slot_refusals;
        qwen_testing::record_text_passes(true);
        const infernix::GenerationResult follow =
            engine.generate(engine.prepare(conversation({system, first_user, reply, follow_user})), greedy(8));
        arm.follow_bases = passes();
        arm.refusals     = engine.runtime_stats().hybrid_device_slot_refusals - arm.refusals;
        const infernix::GenerationResult last = engine.generate(
            engine.prepare(conversation({system, first_user, reply, follow_user, follow_reply, second_user})), greedy(24));
        arm.first_tokens  = first.prompt.prompt_tokens;
        arm.follow_tokens = follow.prompt.prompt_tokens;
        arm.reused        = second.reused_prompt_tokens;
        arm.follow_reused = follow.reused_prompt_tokens;
        arm.last_reused   = last.reused_prompt_tokens;
        arm.second        = second.generated_token_ids;
        arm.last          = last.generated_token_ids;
        return arm;
    };
    Arm cut, pair;
    try {
        cut  = run(false);
        pair = run(true);
    } catch (...) {
        qwen_testing::set_pair_windows(true);
        qwen_testing::record_text_passes(false);
        throw;
    }
    qwen_testing::set_pair_windows(true);
    int failures = 0;
    // The cut's last pass begins at the opener, 2-8 tokens before the prompt's end.
    if (cut.bases.size() < 2 || cut.bases.back() + 8U < cut.first_tokens || cut.bases.back() + 2U > cut.first_tokens) {
        std::cerr << name << ": the cut's first turn ran " << cut.bases.size() << " passes, the last from "
                  << (cut.bases.empty() ? 0U : cut.bases.back()) << " of " << cut.first_tokens
                  << " tokens; expected one from the generation opener\n";
        return 1;
    }
    const std::uint32_t opener = cut.bases.back();
    // The pair arm's passes: the cut's without the one from the opener (with a KV window, the cut's).
    const auto paired = [&](const std::vector<std::uint32_t>& bases) {
        return kv_window ? bases : std::vector<std::uint32_t>(bases.begin(), bases.end() - 1);
    };
    if (pair.bases != paired(cut.bases)) {
        std::cerr << name << ": the pair's first turn ran " << pair.bases.size() << " passes, the cut's "
                  << cut.bases.size() << (kv_window ? " (a KV window keeps the cut)" : "") << '\n';
        ++failures;
    }
    if (cut.reused != opener || pair.reused != opener) {
        std::cerr << name << ": the second turn reused " << cut.reused << " (cut) and " << pair.reused
                  << " (pair) tokens; expected the opener, " << opener << '\n';
        ++failures;
    }
    if (pair.second != cut.second) {
        std::cerr << name << ": the second turn generated other tokens after the pair's tap than after the cut's\n";
        ++failures;
    }
    // The resumed turn: it resumes at the first turn's opener, its own last pass begins at its opener in the cut
    // (2-8 tokens before its end) and not in the pair, and the turn after it resumes there with the cut's tokens.
    const std::uint32_t follow_opener = cut.follow_bases.empty() ? 0U : cut.follow_bases.back();
    const bool follow_paired = pair.follow_bases == paired(cut.follow_bases);
    const bool follow_cut_for_room = !follow_paired && pair.follow_bases == cut.follow_bases && pair.refusals > 0;
    if (cut.follow_reused != opener || pair.follow_reused != opener || follow_opener + 8U < cut.follow_tokens ||
        follow_opener + 2U > cut.follow_tokens || !(follow_paired || follow_cut_for_room)) {
        std::cerr << name << ": the resumed turn reused " << cut.follow_reused << " / " << pair.follow_reused << " of "
                  << cut.follow_tokens << " tokens and ran " << cut.follow_bases.size() << " passes cut, "
                  << pair.follow_bases.size() << " paired (last cut pass from " << follow_opener << "; "
                  << pair.refusals << " Device snapshot slots refused during it)\n";
        ++failures;
    }
    if (cut.last_reused != follow_opener || pair.last_reused != follow_opener || pair.last != cut.last) {
        std::cerr << name << ": the turn after the resumed one reused " << cut.last_reused << " (cut) and "
                  << pair.last_reused << " (pair), expected " << follow_opener << ", and generated "
                  << (pair.last == cut.last ? "equal" : "different") << " tokens\n";
        ++failures;
    }
    std::cout << name << ": a " << cut.first_tokens << "-token turn ran " << cut.bases.size() << " passes cut and "
              << pair.bases.size() << " paired (opener at " << opener << "); the next turn resumed at " << pair.reused
              << " (cut " << cut.reused << "), tokens " << (pair.second == cut.second ? "equal" : "different")
              << "; a resumed " << cut.follow_tokens << "-token turn ran " << cut.follow_bases.size() << " passes cut and "
              << pair.follow_bases.size() << (follow_cut_for_room ? " (cut: no slot)" : " paired") << " (opener at "
              << follow_opener << "); the turn after it resumed at "
              << pair.last_reused << ", tokens " << (pair.last == cut.last ? "equal" : "different") << '\n';
    if (backend != infernix::SpeculativeBackend::None || kv_window) { return failures; }

    // A device failure anywhere in a pair turn's work, its pass among them, gives back the room the pass's tap took
    // (its Device snapshot slot and state image). On an Engine with two Device snapshot slots each swept turn (a new
    // question, so each pairs its opener) fails with the out-of-memory error or completes, a clean turn after it
    // completes, and then a new conversation's next turn still resumes at its opener: two leaked slots would leave
    // its tap none, paired or cut (a pass without room for its tap is cut, and the capture after its head pass
    // tries again).
    const std::uint32_t tail = cut.first_tokens - opener;
    infernix::EngineOptions faulted = hybrid_options(artifact, backend, 8192, 1ULL << 30, 2);
    infernix::Engine engine(std::move(faulted));
    const auto turn = [&](const std::string& text) {
        return conversation({system, text_message(infernix::ChatRole::User, text)});
    };
    const std::vector<std::uint64_t> sweep{1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597};
    std::uint32_t failed = 0;
    for (std::size_t i = 0; i < sweep.size(); ++i) {
        infernix::runtime::testing::request_memory_fault(sweep[i]);
        std::string error;
        try {
            (void)engine.generate(engine.prepare(turn(question + " Case " + std::to_string(i) + ".")), greedy(4));
        } catch (const std::exception& e) { error = e.what(); }
        infernix::runtime::testing::request_memory_fault(0);
        if (!error.empty()) {
            ++failed;
            if (error.find("out of device memory") == std::string::npos) {
                std::cerr << name << ": a turn faulted at check " << sweep[i] << " failed otherwise: " << error << '\n';
                ++failures;
            }
        }
        try {
            (void)engine.generate(engine.prepare(turn("Say yes. Item " + std::to_string(i) + ".")), greedy(2));
        } catch (const std::exception& e) {
            std::cerr << name << ": a clean turn after the fault at check " << sweep[i] << " failed: " << e.what() << '\n';
            return failures + 1;
        }
    }
    const infernix::ChatMessage last_user = text_message(infernix::ChatRole::User, question + " After the faults.");
    qwen_testing::record_text_passes(true);
    const infernix::GenerationResult last = engine.generate(engine.prepare(conversation({system, last_user})), greedy(8));
    std::vector<std::uint32_t> last_bases;
    for (const qwen_testing::TextPass& pass : qwen_testing::text_passes()) { last_bases.push_back(pass.base); }
    qwen_testing::record_text_passes(false);
    const infernix::GenerationResult next =
        engine.generate(engine.prepare(conversation({system, last_user, reply, second_user})), greedy(4));
    const std::uint32_t last_opener = last.prompt.prompt_tokens - tail;
    const bool last_paired = std::find(last_bases.begin(), last_bases.end(), last_opener) == last_bases.end();
    if (next.reused_prompt_tokens != last_opener) {
        std::cerr << name << ": after " << failed << " faulted turns a new conversation's opener (" << last_opener
                  << ", " << (last_paired ? "paired" : "cut") << ") did not serve its next turn, which resumed at "
                  << next.reused_prompt_tokens << '\n';
        ++failures;
    }
    std::cout << name << ": " << failed << " of " << sweep.size() << " faulted pair turns failed; afterwards a new "
              << "conversation's opener was " << (last_paired ? "paired" : "cut") << " and its next turn resumed at "
              << next.reused_prompt_tokens << '\n';
    return failures;
}

// QK-02 beside a landing restore: the pair takes its tap's room before its pass, superseding the snapshot its turn
// resumed from, while that snapshot's Host restore is still landing. With one Device snapshot slot the first turn's
// opener tap gives its Device copy up to another conversation's snapshot, so the resumed turn restores it from the
// Host, held back by testing::set_restore_delay so the copies are in flight when the pair asks for room. In both arms
// the resumed turn restores from the Host and resumes at the first opener, and the turn after it resumes at the
// resumed turn's opener with the same tokens; the pair arm pairs, or is cut where a slot refusal explains it.
int exercise_pair_window_restore(const char* artifact) {
    const std::string name = "pair-window-restore";
    const PairConversation conv = pair_conversation();
    const infernix::ChatMessage other_system =
        text_message(infernix::ChatRole::System, "You answer questions about trains in one short sentence.");
    std::string other_question = "Which train leaves first?";
    for (int i = 0; i < 6; ++i) {
        other_question += " Train " + std::to_string(i) + " leaves platform " + std::to_string(i + 2) +
                          " a few minutes after the one before it and stops at every station on the coast line.";
    }
    const infernix::ChatMessage other_user = text_message(infernix::ChatRole::User, other_question);
    struct Arm {
        std::uint32_t first_tokens = 0, follow_reused = 0, last_reused = 0;
        std::vector<std::uint32_t> follow_bases;
        std::uint64_t restores = 0, refusals = 0;
        std::vector<infernix::TokenId> last;
    };
    const auto run = [&](bool pair) {
        qwen_testing::set_pair_windows(pair);
        infernix::Engine engine(hybrid_options(artifact, infernix::SpeculativeBackend::None, 8192, 1ULL << 30, 1));
        Arm arm;
        const infernix::GenerationResult first =
            engine.generate(engine.prepare(conversation({conv.system, conv.first_user})), greedy(8));
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // its snapshots land on the Host
        (void)engine.generate(engine.prepare(conversation({other_system, other_user})), greedy(8));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const infernix::RuntimeStats before = engine.runtime_stats();
        qwen_testing::set_restore_delay(20000);
        qwen_testing::record_text_passes(true);
        infernix::GenerationResult follow;
        try {
            follow = engine.generate(
                engine.prepare(conversation({conv.system, conv.first_user, conv.reply, conv.follow_user})), greedy(8));
        } catch (...) {
            qwen_testing::set_restore_delay(0);
            qwen_testing::record_text_passes(false);
            throw;
        }
        qwen_testing::set_restore_delay(0);
        for (const qwen_testing::TextPass& pass : qwen_testing::text_passes()) { arm.follow_bases.push_back(pass.base); }
        qwen_testing::record_text_passes(false);
        const infernix::RuntimeStats after = engine.runtime_stats();
        arm.restores = after.hybrid_host_image_restores - before.hybrid_host_image_restores;
        arm.refusals = after.hybrid_device_slot_refusals - before.hybrid_device_slot_refusals;
        const infernix::GenerationResult last = engine.generate(
            engine.prepare(conversation(
                {conv.system, conv.first_user, conv.reply, conv.follow_user, conv.follow_reply, conv.second_user})),
            greedy(24));
        arm.first_tokens  = first.prompt.prompt_tokens;
        arm.follow_reused = follow.reused_prompt_tokens;
        arm.last_reused   = last.reused_prompt_tokens;
        arm.last          = last.generated_token_ids;
        return arm;
    };
    Arm cut, pair;
    try {
        cut  = run(false);
        pair = run(true);
    } catch (...) {
        qwen_testing::set_pair_windows(true);
        throw;
    }
    qwen_testing::set_pair_windows(true);
    int failures = 0;
    const std::uint32_t follow_opener = cut.follow_bases.empty() ? 0U : cut.follow_bases.back();
    const bool paired = !cut.follow_bases.empty() &&
                        pair.follow_bases == std::vector<std::uint32_t>(cut.follow_bases.begin(), cut.follow_bases.end() - 1);
    const bool cut_for_room = !paired && pair.follow_bases == cut.follow_bases && pair.refusals > 0;
    if (cut.restores == 0 || pair.restores == 0 || cut.follow_reused != pair.follow_reused ||
        cut.follow_reused + 16U < cut.first_tokens || cut.follow_reused >= cut.first_tokens || !(paired || cut_for_room)) {
        std::cerr << name << ": the resumed turn restored " << cut.restores << " / " << pair.restores
                  << " images from the Host and reused " << cut.follow_reused << " / " << pair.follow_reused << " of "
                  << cut.first_tokens << " tokens; " << pair.follow_bases.size() << " passes paired against "
                  << cut.follow_bases.size() << " cut (" << pair.refusals << " slot refusals)\n";
        ++failures;
    }
    if (cut.last_reused != follow_opener || pair.last_reused != follow_opener || pair.last != cut.last) {
        std::cerr << name << ": the turn after the resumed one reused " << cut.last_reused << " (cut) and "
                  << pair.last_reused << " (pair), expected " << follow_opener << ", and generated "
                  << (pair.last == cut.last ? "equal" : "different") << " tokens\n";
        ++failures;
    }
    std::cout << name << ": the resumed turn restored " << pair.restores << " images from the Host (cut " << cut.restores
              << ") under a held restore and was " << (paired ? "paired" : cut_for_room ? "cut (no slot)" : "?")
              << "; the turn after it resumed at " << pair.last_reused << ", tokens "
              << (pair.last == cut.last ? "equal" : "different") << '\n';
    return failures;
}

std::vector<std::uint8_t> gradient_ppm(int size, int phase) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(size) + ' ' + std::to_string(size) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < size * size; ++index) {
        ppm.push_back(static_cast<std::uint8_t>((index + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>(((index / size) * 3 + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7 + phase * 5) & 0xff));
    }
    return ppm;
}

// Vision identity: the same image and text resume at the generation opener, while identical text
// tokens with a different image cannot reuse anything from the image onwards (its blocks carry the
// image's content key) and never resume inside the image's span.
int exercise_vision(const char* artifact) {
    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::None, 8192, 1ULL << 30, 0);
    options.context_cache.hybrid.device_snapshot_slots.reset();
    options.enable_vision = true;
    infernix::Engine engine(std::move(options));

    std::string system = "You describe images for an accessibility service.";
    for (int sentence = 0; sentence < 30; ++sentence) {
        system +=
            " Rule " + std::to_string(sentence) + ": name colours, shapes and layout plainly.";
    }
    const auto prompt = [&](const std::vector<std::uint8_t>& image, const char* question) {
        infernix::ChatMessage user;
        user.role = infernix::ChatRole::User;
        infernix::MessagePart media;
        media.kind              = infernix::MessagePartKind::Media;
        media.media.kind        = infernix::MediaKind::Image;
        media.media.bytes       = image;
        media.media.media_type  = "image/x-portable-pixmap";
        media.media.source_name = "image.ppm";
        user.parts.push_back(std::move(media));
        user.parts.push_back(infernix::MessagePart{
            .kind = infernix::MessagePartKind::Text, .text = question, .media = {}});
        return conversation({text_message(infernix::ChatRole::System, system), std::move(user)});
    };
    const std::vector<std::uint8_t> first_image  = gradient_ppm(512, 0);
    const std::vector<std::uint8_t> second_image = gradient_ppm(512, 97);

    const infernix::GenerationResult first =
        engine.generate(engine.prepare(prompt(first_image, "Describe the image.")), greedy(4));
    const infernix::GenerationResult same =
        engine.generate(engine.prepare(prompt(first_image, "Describe the image.")), greedy(4));
    const infernix::GenerationResult other =
        engine.generate(engine.prepare(prompt(second_image, "Describe the image.")), greedy(4));

    int failures = 0;
    if (same.reused_prompt_tokens + 64U < first.prompt.prompt_tokens) {
        std::cerr << "vision: the same image and text reused " << same.reused_prompt_tokens
                  << " of " << first.prompt.prompt_tokens << " tokens\n";
        ++failures;
    }
    // A 512x512 image is 256 merged tokens; a different image must lose at least those.
    if (other.prompt.prompt_tokens != first.prompt.prompt_tokens ||
        other.reused_prompt_tokens + 256U > first.prompt.prompt_tokens) {
        std::cerr << "vision: a different image reused " << other.reused_prompt_tokens << " of "
                  << other.prompt.prompt_tokens << " tokens\n";
        ++failures;
    }
    if (first.generated_token_ids.size() != 4 || other.generated_token_ids.size() != 4) {
        std::cerr << "vision: generation did not complete\n";
        ++failures;
    }
    return failures;
}

// A turn resumed past an image against the same prompt run cold, on the reproducible route (bit for bit). A
// multi-row image advances the rotary position by less than its token count, so every token after it sits at
// position + rope_delta. Turn 1 leaves a snapshot on the chunk grid past the image; turn 2 resumes there and
// prefills only text, through the text pass, which must carry the prompt's rope delta (2026-11-15: it ran at delta
// 0 since the pass's context gained a member, so a resumed image conversation computed its suffix at the wrong
// positions and published it, while a cold run of the same prompt, a multimodal pass throughout, did not).
int exercise_vision_resume(const char* artifact) {
    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::None, 8192, 1ULL << 30, 0);
    options.context_cache.hybrid.device_snapshot_slots.reset();
    options.enable_vision = true;
    options.fast_paths    = false;
    options.fast_path_candidates.clear();

    const std::string system = "You describe images for an accessibility service.";
    // Long enough that a chunk boundary past the image lies before the generation opener.
    std::string question = "Describe the image.";
    for (int sentence = 0; sentence < 150; ++sentence) {
        question += " Detail " + std::to_string(sentence) + ": name the colours, shapes and layout plainly.";
    }
    const std::vector<std::uint8_t> image = gradient_ppm(512, 23);
    const auto first_turn                 = [&] {
        infernix::ChatMessage user;
        user.role = infernix::ChatRole::User;
        infernix::MessagePart media;
        media.kind              = infernix::MessagePartKind::Media;
        media.media.kind        = infernix::MediaKind::Image;
        media.media.bytes       = image;
        media.media.media_type  = "image/x-portable-pixmap";
        media.media.source_name = "image.ppm";
        user.parts.push_back(std::move(media));
        user.parts.push_back(
            infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = question, .media = {}});
        return std::vector<infernix::ChatMessage>{text_message(infernix::ChatRole::System, system), std::move(user)};
    };
    const auto second_turn = [&] {
        std::vector<infernix::ChatMessage> messages = first_turn();
        messages.push_back(text_message(infernix::ChatRole::Assistant, "A diagonal colour gradient."));
        messages.push_back(text_message(infernix::ChatRole::User, "Which colour dominates the top left corner?"));
        return messages;
    };

    std::vector<infernix::TokenId> cold;
    std::uint32_t prompt_tokens = 0;
    {
        infernix::Engine engine(options);
        const infernix::GenerationResult result = engine.generate(engine.prepare(conversation(second_turn())), greedy(24));
        cold          = result.generated_token_ids;
        prompt_tokens = result.prompt.prompt_tokens;
        if (result.reused_prompt_tokens != 0) {
            std::cerr << "vision-resume: the cold run reused " << result.reused_prompt_tokens << " tokens\n";
            return 1;
        }
    }
    infernix::Engine engine(options);
    (void)engine.generate(engine.prepare(conversation(first_turn())), greedy(4));
    const infernix::GenerationResult resumed = engine.generate(engine.prepare(conversation(second_turn())), greedy(24));
    std::cout << "vision-resume: turn 2 reused " << resumed.reused_prompt_tokens << " of " << prompt_tokens
              << " prompt tokens\n";
    int failures = 0;
    // The system text is under a chunk and a 512x512 image is 256 merged tokens: a resume from the second chunk
    // boundary on lies past the image.
    if (resumed.reused_prompt_tokens < 2U * kPrefillChunk || resumed.reused_prompt_tokens % kPrefillChunk != 0) {
        std::cerr << "vision-resume: turn 2 resumed from " << resumed.reused_prompt_tokens
                  << ", not from a chunk boundary past the image\n";
        ++failures;
    }
    if (resumed.generated_token_ids != cold) {
        std::cerr << "vision-resume: the turn resumed past the image differs from the same prompt run cold\n";
        ++failures;
    }
    return failures;
}

// Two requests submitted together carry the same system text and image and ask different
// questions: the second waits for the first one's snapshot past the image and encodes nothing.
// A different image behind the same text shares only the text before it and is encoded.
int exercise_coalesce_vision(const char* artifact) {
    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    options.max_concurrency = 2;
    options.enable_vision   = true;
    infernix::Engine engine(std::move(options));

    std::string system = "You inspect screenshots for a test automation service.";
    for (int sentence = 0; sentence < 60; ++sentence) {
        system += " Rule " + std::to_string(sentence) +
                  ": report the colours, regions and any visible text exactly.";
    }
    const auto prompt = [&](const std::vector<std::uint8_t>& image, const char* question) {
        infernix::ChatMessage user;
        user.role = infernix::ChatRole::User;
        infernix::MessagePart media;
        media.kind              = infernix::MessagePartKind::Media;
        media.media.kind        = infernix::MediaKind::Image;
        media.media.bytes       = image;
        media.media.media_type  = "image/x-portable-pixmap";
        media.media.source_name = "image.ppm";
        user.parts.push_back(std::move(media));
        user.parts.push_back(infernix::MessagePart{
            .kind = infernix::MessagePartKind::Text, .text = question, .media = {}});
        return conversation({text_message(infernix::ChatRole::System, system), std::move(user)});
    };
    const std::vector<std::uint8_t> image = gradient_ppm(512, 11);
    const std::vector<std::uint8_t> other = gradient_ppm(512, 151);

    int failures = 0;
    const auto pair = [&](const std::vector<std::uint8_t>& second_image, bool same) {
        infernix::GenerationHandle leader =
            engine.submit(engine.prepare(prompt(image, "Which colour dominates?")), greedy(1));
        infernix::GenerationHandle follower = engine.submit(
            engine.prepare(prompt(second_image, "Is there any text in the image?")), greedy(4));
        const infernix::GenerationResult led      = leader.wait();
        const infernix::GenerationResult followed = follower.wait();
        // The image's 256 merged tokens end within a few tokens of where the questions diverge.
        const std::uint32_t question_tokens =
            followed.prompt.prompt_tokens >= 40U ? 40U : followed.prompt.prompt_tokens;
        const bool resumed_past_image =
            followed.reused_prompt_tokens + question_tokens >= followed.prompt.prompt_tokens;
        if (same && (!resumed_past_image || followed.timings.vision_seconds != 0.0)) {
            std::cerr << "coalesce-vision: the same image reused " << followed.reused_prompt_tokens
                      << " of " << followed.prompt.prompt_tokens << " tokens and encoded for "
                      << followed.timings.vision_seconds << " s (expected past the image, 0 s)\n";
            ++failures;
        }
        if (!same && (resumed_past_image || followed.timings.vision_seconds == 0.0)) {
            std::cerr << "coalesce-vision: a different image reused "
                      << followed.reused_prompt_tokens << " tokens and encoded for "
                      << followed.timings.vision_seconds << " s\n";
            ++failures;
        }
        if (led.generated_token_ids.size() != 1 || followed.generated_token_ids.size() != 4) {
            std::cerr << "coalesce-vision: generation did not complete\n";
            ++failures;
        }
    };
    pair(image, true);
    // The image is cached now; the different image starts from the shared system block.
    pair(other, false);
    return failures;
}

// Non-finite logits after a request published entries (spec §7.8, candidates SQ-01). A test hook
// (runtime::testing::set_nonfinite_fault) reports a row as non-finite, as a NaN reaching the sampler
// would. Engine 1 caches a clean prompt, its periodic save writes the cache file, then the failing
// prompt's first token fails after its prefill published blocks and a tap: the request fails alone and
// the tree holds what it held before. Engine 2 loads the file saved at Engine 1's stop: the clean prompt
// resumes from it, the failed one resumes from nothing and equals a cache-off run. A request failing in
// decode is withdrawn the same way. The exact route publishes on the chunk grid only, so neither failure
// publishes a block past the prompt's last grid position (the default route's decode-published blocks are
// not covered here: its resend would not equal the cache-off run).
int exercise_nonfinite_withdraw(const char* artifact) {
    const std::filesystem::path file = infernix::test::unique_temp_path("infernix-hybrid-nonfinite-real-test.bin");
    std::error_code ignored;
    std::filesystem::remove(file, ignored);
    const std::vector<infernix::TokenId> clean_prompt = synthetic_tokens(1500, 51);
    const std::vector<infernix::TokenId> failing      = synthetic_tokens(1500, 52);
    const std::vector<infernix::TokenId> decoding     = synthetic_tokens(1500, 53);
    const auto options = [&] {
        infernix::EngineOptions result =
            hybrid_options(artifact, infernix::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
        result.context_cache.hybrid.persistent_file          = file;
        result.context_cache.hybrid.persistent_identity      = "nonfinite-test";
        result.context_cache.hybrid.persistent_save_interval = std::chrono::seconds(1);
        // The exact route: a request that reuses nothing equals its cache-off run only where taps cut calls
        // on the chunk grid (on the default route its taps' cuts change the qualified paths' bits).
        result.fast_paths = false;
        result.fast_path_candidates.clear();
        return result;
    };
    infernix::RequestOptions cache_off         = greedy(24);
    cache_off.execution.allow_prefix_reuse     = false;
    const auto fails_nonfinite = [&](infernix::Engine& engine, const std::vector<infernix::TokenId>& prompt,
                                     std::uint32_t outputs, std::uint32_t rows) {
        infernix::runtime::testing::set_nonfinite_fault(rows);
        bool failed = false;
        try {
            (void)engine.generate(engine.prepare_tokens(prompt), greedy(outputs));
        } catch (const std::exception& error) {
            failed = std::string_view(error.what()).find("non-finite") != std::string_view::npos;
        }
        infernix::runtime::testing::set_nonfinite_fault(0);
        return failed;
    };
    // The Engine publishes its statistics after answering: waits until the tree is back to `before`
    // with `published` more blocks counted (at most 5 s).
    const auto withdrawn = [](infernix::Engine& engine, const infernix::RuntimeStats& before, std::uint64_t published) {
        infernix::RuntimeStats stats = engine.runtime_stats();
        for (int i = 0; i < 250; ++i) {
            if (stats.hybrid_blocks_inserted >= before.hybrid_blocks_inserted + published &&
                stats.hybrid_tree_blocks == before.hybrid_tree_blocks && stats.hybrid_snapshots == before.hybrid_snapshots) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            stats = engine.runtime_stats();
        }
        std::cerr << "nonfinite: the tree holds " << stats.hybrid_tree_blocks << " blocks and " << stats.hybrid_snapshots
                  << " snapshots (before the failure " << before.hybrid_tree_blocks << " and " << before.hybrid_snapshots
                  << "); " << stats.hybrid_blocks_inserted - before.hybrid_blocks_inserted << " blocks were published\n";
        return false;
    };
    // Without free-cache-resume a request publishes state only on the prefill-chunk grid and no decode-computed
    // block (spec §7.9): the 1500-token prompts publish their blocks up to grid position 1024, in both failures.
    const std::uint64_t grid_blocks = (1500 / kPrefillChunk) * kPrefillChunk / kBlock;
    int failures = 0;
    {
        infernix::Engine engine(options());
        (void)engine.generate(engine.prepare_tokens(clean_prompt), greedy(8));
        bool saved = false;
        for (int i = 0; i < 300 && !saved; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            saved = std::filesystem::exists(file);
        }
        if (!saved) {
            std::cerr << "nonfinite: the periodic save did not write the clean prompt's entries\n";
            ++failures;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const infernix::RuntimeStats before = engine.runtime_stats();
        if (!fails_nonfinite(engine, failing, 24, 1)) {
            std::cerr << "nonfinite: the failing prompt's first token did not fail with non-finite logits\n";
            ++failures;
        }
        // The prompt's grid blocks were published before its first token failed.
        if (!withdrawn(engine, before, grid_blocks)) { ++failures; }
        std::cout << "nonfinite: after the failure the cache file "
                  << (std::filesystem::exists(file) ? "exists" : "was deleted (a save ran during the request)") << '\n';
    } // the stop saves the Host tier
    if (!std::filesystem::exists(file)) {
        std::cerr << "nonfinite: Engine 1's stop did not save the Host tier\n";
        return failures + 1;
    }
    {
        infernix::Engine engine(options());
        const infernix::LoadSummary::PrefixCacheRestore restore = engine.load_summary().prefix_cache;
        if (!restore.restored || restore.snapshots == 0) {
            std::cerr << "nonfinite: Engine 2 restored nothing (" << restore.message << ")\n";
            ++failures;
        }
        const std::vector<infernix::TokenId> cold =
            engine.generate(engine.prepare_tokens(failing), cache_off).generated_token_ids;
        const infernix::GenerationResult after_restart = engine.generate(engine.prepare_tokens(failing), greedy(24));
        if (after_restart.reused_prompt_tokens != 0 || after_restart.generated_token_ids != cold) {
            std::cerr << "nonfinite: after the restart the failed prompt reused " << after_restart.reused_prompt_tokens
                      << " tokens (expected 0) or differs from its cache-off run\n";
            ++failures;
        }
        if (engine.generate(engine.prepare_tokens(clean_prompt), greedy(8)).reused_prompt_tokens == 0) {
            std::cerr << "nonfinite: the clean prompt did not resume from the restored file\n";
            ++failures;
        }
        // decode: the 60th row fails, 59 tokens past the 1500-token prompt, after the prompt's grid blocks were
        // published. The cache-off reference generates as many tokens as the run it is compared with.
        infernix::RequestOptions cache_off_decoding     = greedy(96);
        cache_off_decoding.execution.allow_prefix_reuse = false;
        const std::vector<infernix::TokenId> cold_decoding =
            engine.generate(engine.prepare_tokens(decoding), cache_off_decoding).generated_token_ids;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const infernix::RuntimeStats before = engine.runtime_stats();
        if (!fails_nonfinite(engine, decoding, 96, 60)) {
            std::cerr << "nonfinite: the decode row did not fail with non-finite logits\n";
            ++failures;
        }
        if (!withdrawn(engine, before, grid_blocks)) { ++failures; }
        const infernix::GenerationResult again = engine.generate(engine.prepare_tokens(decoding), greedy(96));
        if (again.reused_prompt_tokens != 0 || again.generated_token_ids != cold_decoding) {
            std::cerr << "nonfinite: after the decode failure the prompt reused " << again.reused_prompt_tokens
                      << " tokens (expected 0) or differs from its cache-off run\n";
            ++failures;
        }
        if (engine.generate(engine.prepare_tokens(decoding), greedy(8)).reused_prompt_tokens == 0) {
            std::cerr << "nonfinite: the cache no longer serves the prompt after the withdrawal\n";
            ++failures;
        }
    }
    std::filesystem::remove(file, ignored);
    if (failures == 0) { std::cout << "nonfinite: withdrawal checks passed\n"; }
    return failures;
}

} // namespace

// Waits until the Engine has ended the next-turn pass started after `before` (run to its end,
// yielded, skipped or dropped); false after a minute.
bool await_next_turn(infernix::Engine& engine, const infernix::RuntimeStats& before) {
    const auto ended = [&](const infernix::RuntimeStats& now) {
        return now.next_turn_prefills_completed + now.next_turn_prefills_yielded +
                   now.next_turn_prefills_skipped + now.next_turn_prefills_dropped >
               before.next_turn_prefills_completed + before.next_turn_prefills_yielded +
                   before.next_turn_prefills_skipped + before.next_turn_prefills_dropped;
    };
    for (int i = 0; i < 6000; ++i) {
        if (ended(engine.runtime_stats())) { return true; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

template <class Chat, class Request>
int exercise_next_turn_yield(infernix::Engine& engine, const std::string& name, bool reproducible,
                             const Chat& chat, const Request& request);

// Next-turn prefill (spec §7.9). With preserve_thinking false the template renders a reply without
// its thinking block once a user turn follows it, so the next request diverges at the reply's
// opener and, without the pass, prefills the re-rendered reply itself. Engine::prefill_next_turn
// prefills that history while the Engine is idle:
// - the next request resumes at the history's end (on the reproducible route at the chunk-grid
//   position at or before it);
// - its tokens equal an uncached run's on the reproducible route; on the default route, where a
//   resume is not bitwise an uncached prefill (model-quality §6.7), they equal those of an Engine in
//   which an ordinary request computed the same history (the pass is that request);
// - a request sharing a long new history, submitted once the pass has computed a chunk, makes the
//   pass yield at its next chunk boundary: it is admitted within about one chunk and resumes from the
//   frontier the pass reached.
int exercise_next_turn(const char* artifact, bool reproducible) {
    const std::string name = reproducible ? "next-turn (reproducible)" : "next-turn";
    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::None, 32768, 1ULL << 30, 0);
    options.context_cache.hybrid.device_snapshot_slots.reset();
    options.max_context                                   = 16384;
    options.context_cache.hybrid.next_turn_prefill_quiet = std::chrono::milliseconds{20};
    if (reproducible) {
        options.fast_paths = false;
        options.fast_path_candidates.clear();
    }
    const std::uint32_t grid = reproducible ? kPrefillChunk : 1U;
    const auto chat          = [](std::vector<infernix::ChatMessage> messages) {
        infernix::PromptInput input;
        input.messages                  = std::move(messages);
        input.options.enable_thinking   = false;
        input.options.preserve_thinking = false;
        return input;
    };
    const auto request = [](std::uint32_t outputs, bool reuse) {
        infernix::RequestOptions result         = greedy(outputs);
        result.execution.allow_prefix_reuse     = reuse;
        result.stop.include_model_defaults      = true;
        return result;
    };
    std::string system = "You explain systems topics to new engineers.";
    for (int sentence = 0; sentence < 40; ++sentence) {
        system += " Rule " + std::to_string(sentence) + ": keep answers concrete and name one example.";
    }
    const auto turn1_messages = [&] {
        return std::vector<infernix::ChatMessage>{
            text_message(infernix::ChatRole::System, system),
            text_message(infernix::ChatRole::User,
                         "Explain in detail, in at least 150 words, why caches make programs faster.")};
    };
    int failures = 0;

    infernix::GenerationResult first, second;
    std::vector<infernix::ChatMessage> turn1, turn2;
    infernix::PromptInput rendered;
    {
        infernix::Engine engine(options);
        // Turn 1's prompt ends 32-92 tokens before a chunk boundary, so on the reproducible route the
        // reply (at least 100 tokens) crosses one and the pass has a grid position to publish.
        for (int pad = 0; pad < 600; ++pad) {
            const std::uint32_t phase = engine.prepare(chat(turn1_messages())).summary().prompt_tokens % kPrefillChunk;
            if (phase >= kPrefillChunk - 92U && phase < kPrefillChunk - 32U) { break; }
            system += " Note.";
        }
        turn1 = turn1_messages();
        first = engine.generate(engine.prepare(chat(turn1)), request(160, true));
        std::vector<infernix::ChatMessage> history = turn1;
        history.push_back(text_message(infernix::ChatRole::Assistant, first.content));
        turn2 = history;
        turn2.push_back(text_message(infernix::ChatRole::User, "Now say it in one sentence."));
        // The pass's prompt as the Engine renders it; its last state sits one token before its end.
        rendered                          = chat(history);
        rendered.options.continuation     = infernix::PromptContinuationMode::NextTurnHistory;
        const std::uint32_t pass_tokens   = engine.prepare(rendered).summary().prompt_tokens;
        const std::uint32_t expected      = (pass_tokens - 1U) / grid * grid;
        const infernix::RuntimeStats idle = engine.runtime_stats();
        engine.prefill_next_turn(chat(history));
        if (!await_next_turn(engine, idle)) {
            std::cerr << name << ": the pass did not end within a minute\n";
            return failures + 1;
        }
        const infernix::RuntimeStats passed = engine.runtime_stats();
        second = engine.generate(engine.prepare(chat(turn2)), request(24, true));
        std::cout << name << ": reply " << first.generated_token_ids.size() << " tokens, history " << pass_tokens
                  << " tokens; turn 2 reused " << second.reused_prompt_tokens << " of "
                  << second.prompt.prompt_tokens << " (expected at least " << expected << "); pass computed "
                  << passed.next_turn_prefill_tokens - idle.next_turn_prefill_tokens << " tokens\n";
        if (first.generated_token_ids.size() < 100) {
            std::cerr << name << ": turn 1's reply is too short for the test (" << first.generated_token_ids.size()
                      << " tokens)\n";
            ++failures;
        }
        if (passed.next_turn_prefills_completed != idle.next_turn_prefills_completed + 1) {
            std::cerr << name << ": the pass did not run to its end (skipped "
                      << passed.next_turn_prefills_skipped - idle.next_turn_prefills_skipped << ", dropped "
                      << passed.next_turn_prefills_dropped - idle.next_turn_prefills_dropped << ")\n";
            ++failures;
        }
        if (second.reused_prompt_tokens < expected || second.reused_prompt_tokens >= second.prompt.prompt_tokens) {
            std::cerr << name << ": turn 2 did not resume at the pass's history end\n";
            ++failures;
        }
        if (reproducible) {
            const infernix::GenerationResult uncached =
                engine.generate(engine.prepare(chat(turn2)), request(24, false));
            if (second.generated_token_ids != uncached.generated_token_ids) {
                std::cerr << name << ": turn 2 after the pass generated different tokens than an uncached run\n";
                ++failures;
            }
        }
        failures += exercise_next_turn_yield(engine, name, reproducible, chat, request);
    }
    if (!reproducible) {
        // One Engine at a time: the control replaces the pass by the ordinary request it stands for.
        infernix::EngineOptions control_options                      = options;
        control_options.context_cache.hybrid.next_turn_prefill_quiet = std::chrono::milliseconds{0};
        infernix::Engine control(std::move(control_options));
        const infernix::GenerationResult control_first =
            control.generate(control.prepare(chat(turn1)), request(160, true));
        (void)control.generate(control.prepare(rendered), request(1, true));
        const infernix::GenerationResult control_second =
            control.generate(control.prepare(chat(turn2)), request(24, true));
        if (control_first.generated_token_ids != first.generated_token_ids ||
            control_second.reused_prompt_tokens != second.reused_prompt_tokens ||
            control_second.generated_token_ids != second.generated_token_ids) {
            std::cerr << name << ": turn 2 after the pass differs from turn 2 after an ordinary request "
                      << "computed the same history (reused " << control_second.reused_prompt_tokens << ")\n";
            ++failures;
        }
    }
    return failures;
}

// The yield half of exercise_next_turn: a new conversation with a long document; the request that
// shares its history arrives once the pass has computed its first chunk.
template <class Chat, class Request>
int exercise_next_turn_yield(infernix::Engine& engine, const std::string& name, bool reproducible,
                             const Chat& chat, const Request& request) {
    int failures = 0;
    std::string document;
    for (int paragraph = 0; paragraph < 400; ++paragraph) {
        document += "Section " + std::to_string(paragraph) + " records that the warehouse moved " +
                    std::to_string(paragraph * 37 % 101) + " crates of part " + std::to_string(paragraph % 17) +
                    " to bay " + std::to_string(paragraph % 9) + " during the night shift. ";
    }
    const std::vector<infernix::ChatMessage> long_history{
        text_message(infernix::ChatRole::System, "You summarise warehouse logs."),
        text_message(infernix::ChatRole::User, document + "Summarise the log."),
        text_message(infernix::ChatRole::Assistant, "Crates moved between bays every night.")};
    std::vector<infernix::ChatMessage> long_turn = long_history;
    long_turn.push_back(text_message(infernix::ChatRole::User, "Which bay received the most?"));
    const infernix::RuntimeStats before_long = engine.runtime_stats();
    const auto submitted = std::chrono::steady_clock::now();
    engine.prefill_next_turn(chat(long_history));
    bool computing = false;
    for (int i = 0; i < 6000 && !computing; ++i) {
        computing = engine.runtime_stats().next_turn_prefill_tokens > before_long.next_turn_prefill_tokens;
        if (!computing) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    }
    // The pass's first unit, timed from the call (its render and quiet period included, so at least one unit).
    const double first_unit = std::chrono::duration<double>(std::chrono::steady_clock::now() - submitted).count();
    const infernix::GenerationResult interrupted =
        engine.generate(engine.prepare(chat(long_turn)), request(8, true));
    const infernix::RuntimeStats after_long = engine.runtime_stats();
    const std::uint64_t reached = after_long.next_turn_prefill_tokens - before_long.next_turn_prefill_tokens;
    std::cout << name << ": the long pass computed " << reached << " tokens before yielding (first unit "
              << first_unit << " s); the request waited " << interrupted.engine_timing.queue_wait_seconds
              << " s for admission and reused " << interrupted.reused_prompt_tokens << " of "
              << interrupted.prompt.prompt_tokens << '\n';
    if (!computing || after_long.next_turn_prefills_yielded != before_long.next_turn_prefills_yielded + 1) {
        std::cerr << name << ": the long pass did not yield to the request\n";
        ++failures;
    }
    // A request waits at most one prefill unit (hybrid-prefix-cache-spec §7.9): the unit that began as the first
    // ended, then admission. On an idle Engine a unit is the idle prefill width (idle-wide-prefill: 3,584 tokens), so
    // the bound follows the pass's own first unit: twice it, as the unit waited for lies deeper in the prompt and
    // admission follows it (2026-11-09: groupwise 27B first unit 0.91 s, wait 1.21 s; official NVFP4 27B 0.25 s and
    // 0.31 s), and never below the second this check used to fix, which failed the groupwise artifact.
    const double wait_bound = std::max(1.0, 2.0 * first_unit);
    if (interrupted.engine_timing.queue_wait_seconds > wait_bound) {
        std::cerr << name << ": the request waited " << interrupted.engine_timing.queue_wait_seconds
                  << " s, more than one prefill unit (bound " << wait_bound << " s) for the pass to yield\n";
        ++failures;
    }
    // The yield kept the frontier the pass reached (a chunk boundary: the endpoint of its abort, or on
    // the reproducible route a tap on the grid).
    if (interrupted.reused_prompt_tokens < reached) {
        std::cerr << name << ": the request did not resume from the frontier the pass reached\n";
        ++failures;
    }
    if (reproducible) {
        const infernix::GenerationResult uncached =
            engine.generate(engine.prepare(chat(long_turn)), request(8, false));
        if (interrupted.generated_token_ids != uncached.generated_token_ids) {
            std::cerr << name << ": the request after the yield generated different tokens than an uncached run\n";
            ++failures;
        }
    }
    return failures;
}

int main() {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }
    if (const char* storage = std::getenv("INFERNIX_HYBRID_KV_DTYPE");
        storage != nullptr && *storage != '\0') {
        try {
            kv_storage = infernix::test::parse_kv_cache_storage(storage);
        } catch (const std::invalid_argument&) {
            std::cerr << "unknown INFERNIX_HYBRID_KV_DTYPE " << storage << '\n';
            return 1;
        }
    }
    const char* selected            = std::getenv("INFERNIX_HYBRID_REAL_SCENARIO");
    const std::string_view scenario = selected != nullptr && *selected != '\0' ? selected : "all";
    constexpr std::array<std::string_view, 27> kScenarios{"all",
                                                          "pair-window",
                                                          "pair-window-restore",
                                                          "pair-window-dflash2",
                                                          "pair-window-mtp",
                                                          "vision-resume",
                                                          "ssd-restore",
                                                          "ssd-restore-mtp",
                                                          "next-turn",
                                                          "next-turn-reproducible",
                                                          "issue29",
                                                          "restore-exact",
                                                          "restore-exact-mtp",
                                                          "restore-exact-dflash2",
                                                          "turns",
                                                          "turns-device-only",
                                                          "vision",
                                                          "persist",
                                                          "turns-protocol",
                                                          "coalesce",
                                                          "coalesce-vision",
                                                          "interleaved",
                                                          "identical-burst",
                                                          "cancel-prefill",
                                                          "cancel-prefill-dflash2",
                                                          "nonfinite-withdraw"};
    if (std::find(kScenarios.begin(), kScenarios.end(), scenario) == kScenarios.end()) {
        std::cerr << "unknown INFERNIX_HYBRID_REAL_SCENARIO " << scenario << '\n';
        return 1;
    }
    int failures = 0;
    try {
        const bool all = scenario == "all";
        if (all || scenario == "issue29") { failures += exercise_issue29(artifact); }
        if (all || scenario == "restore-exact") {
            failures += exercise_restore_exact(artifact, infernix::SpeculativeBackend::None);
        }
        if (all || scenario == "restore-exact-mtp") {
            failures += exercise_restore_exact(artifact, infernix::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "restore-exact-dflash2") {
            failures += exercise_restore_exact(artifact, infernix::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "ssd-restore") {
            failures += exercise_ssd_restore(artifact, infernix::SpeculativeBackend::None);
        }
        if (all || scenario == "ssd-restore-mtp") {
            failures += exercise_ssd_restore(artifact, infernix::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "turns") { failures += exercise_turns(artifact, 1ULL << 30); }
        if (all || scenario == "pair-window") {
            failures += exercise_pair_window(artifact, infernix::SpeculativeBackend::None);
        }
        if (all || scenario == "pair-window-dflash2") {
            failures += exercise_pair_window(artifact, infernix::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "pair-window-mtp") {
            failures += exercise_pair_window(artifact, infernix::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "pair-window-restore") { failures += exercise_pair_window_restore(artifact); }
        if (all || scenario == "turns-device-only") { failures += exercise_turns(artifact, 0); }
        if (all || scenario == "vision") { failures += exercise_vision(artifact); }
        if (all || scenario == "vision-resume") { failures += exercise_vision_resume(artifact); }
        if (all || scenario == "persist") { failures += exercise_persist(artifact); }
        if (all || scenario == "interleaved") { failures += exercise_interleaved(artifact); }
        if (all || scenario == "identical-burst") { failures += exercise_identical_burst(artifact); }
        if (all || scenario == "cancel-prefill") {
            failures += exercise_cancel_prefill(artifact, infernix::SpeculativeBackend::None);
        }
        if (all || scenario == "cancel-prefill-dflash2") {
            failures += exercise_cancel_prefill(artifact, infernix::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "coalesce") {
            failures += exercise_coalesce(artifact, 6 * kPrefillChunk, true);
            failures += exercise_coalesce(artifact, 3000, false);
            failures += exercise_coalesce(artifact, 6 * kPrefillChunk, false, 8 * kPrefillChunk);
        }
        if (all || scenario == "coalesce-vision") {
            failures += exercise_coalesce_vision(artifact);
        }
        if (all || scenario == "nonfinite-withdraw") { failures += exercise_nonfinite_withdraw(artifact); }
        if (all || scenario == "next-turn") { failures += exercise_next_turn(artifact, false); }
        if (all || scenario == "next-turn-reproducible") { failures += exercise_next_turn(artifact, true); }
        if (all || scenario == "turns-protocol") {
            protocol_hints = true;
            failures += exercise_turns(artifact, 1ULL << 30);
            protocol_hints = false;
        }
    } catch (const std::exception& error) {
        std::cerr << "hybrid prefix real test failed: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "hybrid prefix real tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
