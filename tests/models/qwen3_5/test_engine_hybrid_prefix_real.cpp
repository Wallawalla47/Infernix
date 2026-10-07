// Real-artifact scenarios for the hybrid prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md
// §13.3). Requires INFERNIX_TEST_ARTIFACT; INFERNIX_HYBRID_REAL_SCENARIO selects one scenario.

#include "kv_cache_storage.h"
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
    options.context_cache.mode                         = infernix::ContextCacheMode::Hybrid;
    options.context_cache.host_capacity_bytes          = host_bytes;
    options.context_cache.hybrid.device_snapshot_slots = device_slots;
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

// A later request restores a flexible prompt-tail snapshot the first request captured at its last
// prefill chunk boundary. The Device-resident engine restores it with Device copies; the
// constrained engine is first forced to evict the snapshot image and most of its blocks, so the
// same snapshot comes back through the Host slabs. Both restores carry identical bytes and resume
// at the same frontier with the same chunking, so greedy generation must match token for token.
int exercise_restore_exact(const char* artifact, infernix::SpeculativeBackend backend) {
    const std::vector<infernix::TokenId> first = synthetic_tokens(1500, 1);
    std::vector<infernix::TokenId> second(first.begin(), first.begin() + 1400);
    const std::vector<infernix::TokenId> suffix = synthetic_tokens(300, 2);
    second.insert(second.end(), suffix.begin(), suffix.end());
    // 3900 prompt tokens need 61 of the constrained engine's 64 pages, evicting all but the
    // first few of the first request's cached blocks.
    const std::vector<infernix::TokenId> pressure = synthetic_tokens(3900, 3);

    const auto run = [&](infernix::EngineOptions options, bool apply_pressure) {
        infernix::Engine engine(std::move(options));
        (void)engine.generate(engine.prepare_tokens(first), greedy(8));
        if (apply_pressure) { (void)engine.generate(engine.prepare_tokens(pressure), greedy(4)); }
        const infernix::GenerationResult result =
            engine.generate(engine.prepare_tokens(second), greedy(24));
        return Observed{result.generated_token_ids, result.reused_prompt_tokens,
                        engine.runtime_stats()};
    };
    const Observed device = run(hybrid_options(artifact, backend, 16384, 1ULL << 30, 8), false);
    const Observed host   = run(hybrid_options(artifact, backend, 4096, 1ULL << 30, 1), true);

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
    if (device.tokens.size() != 24 || device.tokens != host.tokens) {
        std::cerr << "restore-exact: Host and Device restores generated different tokens\n";
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
        std::filesystem::temp_directory_path() / "infernix-hybrid-persist-real-test.bin";
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
    int failures = 0;
    {
        // The Engine stops the way infernix-serve does: stop() while a generation is still far from
        // its output limit. It fails as Unavailable instead of running on, new work is refused,
        // and the Host tier is still saved.
        infernix::Engine saver(options("persist-test"));
        (void)saver.generate(saver.prepare_tokens(first), greedy(8));
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
    }
    if (!std::filesystem::exists(file)) {
        std::cerr << "persist: the Engine did not save its Host tier\n";
        return 1;
    }
    // The saver's file, for the smaller Host tier below (the loader saves over `file`).
    const std::filesystem::path partial =
        std::filesystem::temp_directory_path() / "infernix-hybrid-persist-real-test-partial.bin";
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
int exercise_coalesce(const char* artifact, std::uint32_t shared_tokens, bool compare_output) {
    const std::vector<infernix::TokenId> shared        = synthetic_tokens(shared_tokens, 6);
    std::vector<infernix::TokenId> first               = shared;
    std::vector<infernix::TokenId> second              = shared;
    const std::vector<infernix::TokenId> first_suffix  = synthetic_tokens(40, 7);
    const std::vector<infernix::TokenId> second_suffix = synthetic_tokens(40, 8);
    first.insert(first.end(), first_suffix.begin(), first_suffix.end());
    second.insert(second.end(), second_suffix.begin(), second_suffix.end());
    const std::uint32_t expected = shared_tokens / kBlock * kBlock;

    infernix::EngineOptions options =
        hybrid_options(artifact, infernix::SpeculativeBackend::None, 16384, 1ULL << 30, 8);
    options.max_concurrency = 2;
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
    const std::vector<infernix::TokenId> prompt = synthetic_tokens(6 * kPrefillChunk + 100, 11);
    std::vector<infernix::TokenId> extended     = prompt;
    const std::vector<infernix::TokenId> suffix = synthetic_tokens(60, 12);
    extended.insert(extended.end(), suffix.begin(), suffix.end());
    const std::vector<infernix::TokenId> unrelated = synthetic_tokens(2 * kPrefillChunk + 30, 13);
    const infernix::EngineOptions options = hybrid_options(artifact, backend, 16384, 1ULL << 30, 8);

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

} // namespace

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
    constexpr std::array<std::string_view, 15> kScenarios{"all",
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
                                                          "cancel-prefill",
                                                          "cancel-prefill-dflash2"};
    if (std::find(kScenarios.begin(), kScenarios.end(), scenario) == kScenarios.end()) {
        std::cerr << "unknown INFERNIX_HYBRID_REAL_SCENARIO " << scenario << '\n';
        return 1;
    }
    int failures = 0;
    try {
        const bool all = scenario == "all";
        if (all || scenario == "restore-exact") {
            failures += exercise_restore_exact(artifact, infernix::SpeculativeBackend::None);
        }
        if (all || scenario == "restore-exact-mtp") {
            failures += exercise_restore_exact(artifact, infernix::SpeculativeBackend::Mtp);
        }
        if (all || scenario == "restore-exact-dflash2") {
            failures += exercise_restore_exact(artifact, infernix::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "turns") { failures += exercise_turns(artifact, 1ULL << 30); }
        if (all || scenario == "turns-device-only") { failures += exercise_turns(artifact, 0); }
        if (all || scenario == "vision") { failures += exercise_vision(artifact); }
        if (all || scenario == "persist") { failures += exercise_persist(artifact); }
        if (all || scenario == "interleaved") { failures += exercise_interleaved(artifact); }
        if (all || scenario == "cancel-prefill") {
            failures += exercise_cancel_prefill(artifact, infernix::SpeculativeBackend::None);
        }
        if (all || scenario == "cancel-prefill-dflash2") {
            failures += exercise_cancel_prefill(artifact, infernix::SpeculativeBackend::DFlash2);
        }
        if (all || scenario == "coalesce") {
            failures += exercise_coalesce(artifact, 6 * kPrefillChunk, true);
            failures += exercise_coalesce(artifact, 3000, false);
        }
        if (all || scenario == "coalesce-vision") {
            failures += exercise_coalesce_vision(artifact);
        }
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
