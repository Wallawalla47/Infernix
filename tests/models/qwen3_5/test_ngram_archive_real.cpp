#include <ninfer/engine.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

ninfer::PromptInput prompt() {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(
        {.kind  = ninfer::MessagePartKind::Text,
         .text  = "Output Python code assigning SQUARES a dictionary literal mapping every integer "
                  "key 0 through 31 to its square. Use one key/value pair per line. Include all 32 "
                  "entries explicitly, no comprehension or helper functions. Only code, no prose "
                  "or Markdown fences.",
         .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

void check_file(const ninfer::GenerationResult& result) {
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

struct Sink : ninfer::OutputSink {
    std::atomic<std::size_t> bytes{0};

    void start(ninfer::GenerationStart) override {}

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta delta) override { bytes += delta.text.size(); }
};

} // namespace

int main(int argc, char** argv) {
    const auto* artifact = std::getenv("NINFER_NGRAM_TEST_WEIGHTS");
    if (!artifact || !*artifact) { return 77; }
    try {
        const std::string backend = argc > 1 ? argv[1] : "mtp";
        require(argc <= 3, "expected backend and optional graph mode (0/1)");
        require(argc < 3 || std::string(argv[2]) == "0" || std::string(argv[2]) == "1",
                "graph mode must be 0 or 1");
        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 4096;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.enable_vision   = false;
        options.max_concurrency = 1;
        options.prefill_chunk   = 1024;
        options.kv_cache        = ninfer::KvCacheStorage::Nvfp4Group16;
        options.use_cuda_graph  = argc < 3 || std::stoi(argv[2]) != 0;
        if (backend == "mtp") {
            options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        } else if (backend == "dflash") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
        } else if (backend == "dflash2") {
            options.speculative.backend = ninfer::SpeculativeBackend::DFlash2;
        } else {
            throw std::invalid_argument("unsupported backend");
        }
        options.speculative.draft_tokens        = 5;
        options.speculative.ngram_draft_tokens  = 63;
        options.speculative.proposal_head       = ninfer::ProposalHead::Optimized;
        options.speculative.ngram_archive_bytes = 16ULL << 20;
        options.speculative.ngram_session_bytes = 4ULL << 20;
        ninfer::Engine engine(options);
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens    = 1024;
        request.execution.sampling.temperature       = 0;
        request.execution.sampling.presence_penalty  = 0;
        request.execution.sampling.frequency_penalty = 0;
        request.execution.allow_prefix_reuse         = false;
        request.ngram_session.key                    = "parent";
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
                        result.ngram_archive.generation > previous.ngram_archive.generation &&
                        result.ngram_archive.sampling_seed != previous.ngram_archive.sampling_seed,
                    "retained draft, generation or random domain did not advance");
            previous = std::move(result);
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
                            ninfer::CancellationView([&] { return sink.bytes.load() >= 128; }));
        require(cancelled.finish_reason == ninfer::FinishReason::Cancelled &&
                    cancelled.ngram_archive.bound && !cancelled.ngram_archive.published &&
                    cancelled.ngram_archive.generation == child.ngram_archive.generation,
                "cancelled request published or altered the completed generation");
        require(run().speculative.ngram_archive_accepted_tokens > 0,
                "cancelled request lost the completed archive");
        std::cout
            << "archive source-absent resumes, isolation, fork, reset and cancellation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
