#include <infernix/engine.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

infernix::PromptInput prompt() {
    infernix::PromptInput input;
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back(
        {.kind  = infernix::MessagePartKind::Text,
         .text  = "Output Python code assigning SQUARES a dictionary literal mapping every integer "
                  "key 0 through 31 to its square. Use one key/value pair per line. Include all 32 "
                  "entries explicitly, no comprehension or helper functions. Only code, no prose "
                  "or Markdown fences.",
         .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

void check_file(const infernix::GenerationResult& result) {
    auto code = result.content;
    std::erase_if(code, [](unsigned char c) { return std::isspace(c); });
    std::string expected = "SQUARES={";
    for (unsigned i = 0; i < 32; ++i) {
        expected += std::to_string(i) + ":" + std::to_string(i * i) + ",";
    }
    const auto trailing = expected + "}";
    expected.back()     = '}';
    require(code == expected || code == trailing, "incorrect regenerated dictionary");
}

// A second, distinct regenerated file so a concurrent lane cannot silently emit another lane's
// retained source.
infernix::PromptInput cubes_prompt() {
    infernix::PromptInput input;
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back(
        {.kind  = infernix::MessagePartKind::Text,
         .text  = "Output Python code assigning CUBES a dictionary literal mapping every integer "
                  "key 0 through 31 to its cube. Use one key/value pair per line. Include all 32 "
                  "entries explicitly, no comprehension or helper functions. Only code, no prose "
                  "or Markdown fences.",
         .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

void check_cubes(const infernix::GenerationResult& result) {
    auto code = result.content;
    std::erase_if(code, [](unsigned char c) { return std::isspace(c); });
    std::string expected = "CUBES={";
    for (unsigned i = 0; i < 32; ++i) {
        expected += std::to_string(i) + ":" + std::to_string(i * i * i) + ",";
    }
    const auto trailing = expected + "}";
    expected.back()     = '}';
    require(code == expected || code == trailing, "incorrect regenerated cube dictionary");
}

struct Sink : infernix::OutputSink {
    std::atomic<std::size_t> bytes{0};

    void start(infernix::GenerationStart) override {}

    void progress(infernix::PromptProgress) override {}

    void timing(infernix::GenerationTimingObservation) override {}

    void tool_call(infernix::ToolCallDelta) override {}

    void publish(infernix::OutputDelta delta) override { bytes += delta.text.size(); }
};

} // namespace

int main(int argc, char** argv) {
    const auto* artifact = std::getenv("INFERNIX_NGRAM_TEST_WEIGHTS");
    if (!artifact || !*artifact) { return 77; }
    try {
        const std::string backend = argc > 1 ? argv[1] : "mtp";
        require(argc <= 5,
                "expected backend, optional graph mode (0/1), concurrency and route (default/reproducible)");
        require(argc < 3 || std::string(argv[2]) == "0" || std::string(argv[2]) == "1",
                "graph mode must be 0 or 1");
        const unsigned concurrency = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3])) : 1U;
        require(concurrency >= 1 && concurrency <= 8, "concurrency must be in 1..8");
        require(argc < 5 || std::string(argv[4]) == "default" || std::string(argv[4]) == "reproducible",
                "route must be default or reproducible");
        const bool reproducible = argc > 4 && std::string(argv[4]) == "reproducible";
        infernix::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 4096;
        options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(4096);
        options.enable_vision   = false;
        options.max_concurrency = concurrency;
        options.prefill_chunk   = 1024;
        options.kv_cache        = infernix::KvCacheStorage::Nvfp4Group16;
        options.use_cuda_graph  = argc < 3 || std::stoi(argv[2]) != 0;
        if (concurrency > 1) {
            options.context_cache.hybrid.device_snapshot_slots        = 2;
            options.context_cache.host_capacity_bytes = 1ULL << 30;
        }
        if (backend == "mtp") {
            options.speculative.backend = infernix::SpeculativeBackend::Mtp;
        } else if (backend == "dflash") {
            options.speculative.backend = infernix::SpeculativeBackend::DFlash;
        } else if (backend == "dflash2") {
            options.speculative.backend = infernix::SpeculativeBackend::DFlash2;
        } else {
            throw std::invalid_argument("unsupported backend");
        }
        options.speculative.draft_tokens        = 5;
        options.speculative.ngram_draft_tokens  = 63;
        options.speculative.proposal_head       = infernix::ProposalHead::Optimized;
        options.speculative.ngram_archive_bytes = 16ULL << 20;
        options.speculative.ngram_session_bytes = 4ULL << 20;
        options.fast_paths                      = !reproducible;
        infernix::Engine engine(options);
        infernix::RequestOptions request;
        request.execution.requested_output_tokens    = 1024;
        request.execution.sampling.temperature       = 0.0f;
        request.execution.sampling.presence_penalty  = 0.0f;
        request.execution.sampling.frequency_penalty = 0.0f;
        request.execution.allow_prefix_reuse         = false;
        request.ngram_session.key                    = "parent";
        if (reproducible) { request.execution.sampling.seed = 7; }
        auto run                                     = [&] {
            // Every request contains only the formula, never the old file text.
            auto result = engine.generate(engine.prepare(prompt()), request);
            check_file(result);
            require(result.reused_prompt_tokens == 0 && result.ngram_archive.bound &&
                                                            result.ngram_archive.published && result.ngram_archive.sources > 0,
                                                        "archive publication or prefix-cache independence failed");
            return result;
        };
        auto previous = run();
        for (unsigned depth = 1; depth <= 3; ++depth) {
            auto result = run();
            require(result.speculative.ngram_archive_accepted_tokens > 0 &&
                        result.ngram_archive.generation > previous.ngram_archive.generation,
                    "retained draft or generation did not advance");
            // The default route separates each bound request's random domain; the reproducible route
            // keeps the caller's seed (docs/ngram.md).
            require(reproducible
                        ? result.ngram_archive.sampling_seed == std::uint64_t{7}
                        : result.ngram_archive.sampling_seed != previous.ngram_archive.sampling_seed,
                    reproducible ? "the reproducible route changed a bound request's seed"
                                 : "random domain did not advance");
            previous = std::move(result);
        }
        if (reproducible) {
            // A sampled request bound to a session replays exactly on the reproducible route: the
            // second and third runs draft from the first one's retained output and still give its
            // tokens.
            auto sampled                              = request;
            sampled.execution.sampling.temperature    = 1.0f;
            sampled.execution.sampling.seed           = 1234;
            sampled.ngram_session                     = {.key = "sampled"};
            sampled.execution.requested_output_tokens = 256;
            const auto first = engine.generate(engine.prepare(prompt()), sampled);
            for (unsigned replay = 0; replay < 2; ++replay) {
                const auto again = engine.generate(engine.prepare(prompt()), sampled);
                require(again.ngram_archive.bound && again.ngram_archive.sampling_seed == 1234U,
                        "a sampled bound request lost its seed on the reproducible route");
                require(again.generated_token_ids == first.generated_token_ids,
                        "a sampled bound request did not replay on the reproducible route");
            }
            std::cout << "reproducible route: sampled bound request replayed exactly ("
                      << first.generated_token_ids.size() << " tokens)\n";
        }
        request.ngram_session = {.key               = "child",
                                 .parent            = "parent",
                                 .parent_generation = previous.ngram_archive.generation};
        const auto child      = run();
        require(child.speculative.ngram_archive_accepted_tokens > 0, "fork lost retained source");
        request.ngram_session = {.key = "unrelated"};
        require(run().speculative.ngram_archive_accepted_tokens == 0, "cross-session proposal");
        request.ngram_session = {.key = "parent", .reset = true};
        const auto reset      = run();
        require(reset.speculative.ngram_archive_accepted_tokens == 0 &&
                    reset.ngram_archive.generation > previous.ngram_archive.generation,
                "reset retained sources or recycled a generation");
        request.ngram_session = {.key = "child"};
        Sink sink;
        const auto cancelled =
            engine.generate(engine.prepare(prompt()), request, &sink,
                            infernix::CancellationView([&] { return sink.bytes.load() >= 128; }));
        require(cancelled.finish_reason == infernix::FinishReason::Cancelled &&
                    cancelled.ngram_archive.bound && !cancelled.ngram_archive.published &&
                    cancelled.ngram_archive.generation == child.ngram_archive.generation,
                "cancelled request published or altered the completed generation");
        require(run().speculative.ngram_archive_accepted_tokens > 0,
                "cancelled request lost the completed archive");
        std::cout
            << "archive source-absent resumes, isolation, fork, reset and cancellation passed\n";

        if (concurrency > 1) {
            // Seed independent sessions sequentially, then exercise the archive with concurrent
            // requests. Each lane's immutable snapshot is taken at its own admission, so lanes
            // must never observe another session's retained sources.
            auto seed = [&](const std::string& key, infernix::PromptInput input,
                            const auto& check) {
                request.ngram_session = {.key = key};
                auto result           = engine.generate(engine.prepare(std::move(input)), request);
                check(result);
                require(result.ngram_archive.bound && result.ngram_archive.published &&
                            result.ngram_archive.sources > 0,
                        "concurrent seed did not publish retained sources");
                return result.ngram_archive.generation;
            };
            seed("ca", prompt(), check_file);
            seed("cx", cubes_prompt(), check_cubes);

            // Concurrent requests on distinct seeded sessions must each regenerate their own file
            // and publish; the archive attribution is reported because the request-local index can
            // legitimately win the same span.
            request.ngram_session = {.key = "ca"};
            auto handle_a         = engine.submit(engine.prepare(prompt()), request);
            request.ngram_session = {.key = "cx"};
            auto handle_c         = engine.submit(engine.prepare(cubes_prompt()), request);
            const auto result_a   = handle_a.wait();
            const auto result_c   = handle_c.wait();
            check_file(result_a);
            check_cubes(result_c);
            require(result_a.ngram_archive.bound && result_c.ngram_archive.bound &&
                        result_a.ngram_archive.published && result_c.ngram_archive.published,
                    "concurrent distinct sessions did not both bind and publish");
            std::cout << "concurrent distinct sessions: archive_a="
                      << result_a.speculative.ngram_archive_accepted_tokens << " archive_c="
                      << result_c.speculative.ngram_archive_accepted_tokens << "\n";

            // Concurrent *forks* of one source into two fresh sessions, using the parent's latest
            // published generation. A fork's request-local index holds only the formula, so
            // accepted copy tokens must come from the inherited archive source.
            const auto latest_a   = result_a.ngram_archive.generation;
            request.ngram_session = {.key = "fa", .parent = "ca", .parent_generation = latest_a};
            auto handle_fa        = engine.submit(engine.prepare(prompt()), request);
            request.ngram_session = {.key = "fb", .parent = "ca", .parent_generation = latest_a};
            auto handle_fb        = engine.submit(engine.prepare(prompt()), request);
            const auto result_fa  = handle_fa.wait();
            const auto result_fb  = handle_fb.wait();
            check_file(result_fa);
            check_file(result_fb);
            std::cout << "concurrent forks: bound_fa=" << result_fa.ngram_archive.bound
                      << " bound_fb=" << result_fb.ngram_archive.bound
                      << " archive_fa=" << result_fa.speculative.ngram_archive_accepted_tokens
                      << " archive_fb=" << result_fb.speculative.ngram_archive_accepted_tokens
                      << "\n";
            require(result_fa.speculative.ngram_archive_accepted_tokens > 0 &&
                        result_fb.speculative.ngram_archive_accepted_tokens > 0,
                    "concurrent forks did not copy from their inherited archive sources");

            // Same-key concurrency: a session admits one binding at a time. Both lanes must still
            // decode exactly; the unbound lane falls back to request-local drafting.
            seed("cd", prompt(), check_file);
            request.ngram_session = {.key = "cd"};
            auto handle_d1        = engine.submit(engine.prepare(prompt()), request);
            auto handle_d2        = engine.submit(engine.prepare(prompt()), request);
            const auto result_d1  = handle_d1.wait();
            const auto result_d2  = handle_d2.wait();
            check_file(result_d1);
            check_file(result_d2);
            require(!(result_d1.ngram_archive.bound && result_d2.ngram_archive.bound),
                    "same-key concurrency bound two requests to one session");
            std::cout << "same-key concurrency: bound=" << result_d1.ngram_archive.bound << "/"
                      << result_d2.ngram_archive.bound << " archive="
                      << result_d1.speculative.ngram_archive_accepted_tokens << "/"
                      << result_d2.speculative.ngram_archive_accepted_tokens << "\n";

            // Soak: repeated concurrent pairs must stay exact on every round. This is the drift
            // probe for the reported "descends into gibberish after a while" failure mode.
            for (unsigned round = 0; round < 4; ++round) {
                request.ngram_session = {.key = "ca"};
                auto handle_1         = engine.submit(engine.prepare(prompt()), request);
                request.ngram_session = {.key = "cx"};
                auto handle_2         = engine.submit(engine.prepare(cubes_prompt()), request);
                check_file(handle_1.wait());
                check_cubes(handle_2.wait());
            }
            std::cout << "concurrent archive soak passed; rounds=4 concurrency=" << concurrency
                      << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
