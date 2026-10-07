#include "infernix/engine.h"
#include "speculative_page_boundary.h"
#include "kv_cache_storage.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

infernix::RequestOptions request(std::uint32_t outputs, bool reuse = false) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

void valid(const infernix::GenerationResult& result, std::size_t outputs) {
    require(result.generated_token_ids.size() == outputs &&
                result.finish_reason == infernix::FinishReason::OutputLimit,
            "DFlash2 did not honor the requested output budget");
    require(
        result.speculative.backend == infernix::SpeculativeBackend::DFlash2 &&
            (outputs == 1 || result.speculative.rounds + result.speculative.fallback_steps != 0),
        "generation bypassed DFlash2");
}

infernix::PromptInput media_prompt(infernix::MediaKind kind) {
    const std::string header = "P6\n64 64\n255\n";
    infernix::MessagePart media;
    media.kind              = infernix::MessagePartKind::Media;
    media.media.kind        = kind;
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "pattern.ppm";
    media.media.bytes.assign(header.begin(), header.end());
    for (int i = 0; i < 64 * 64; ++i) {
        media.media.bytes.push_back(i & 255);
        media.media.bytes.push_back((i * 3) & 255);
        media.media.bytes.push_back((i * 7) & 255);
    }
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back(std::move(media));
    user.parts.push_back({.kind  = infernix::MessagePartKind::Text,
                          .text  = "Describe the pattern briefly.",
                          .media = {}});
    infernix::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}
} // namespace

// Optional K, Graph, optimized-head, B, KV codec, Vision, state-slot and tree-table arguments
// select representative integration routes without multiplying test binaries. The artifact
// supplies the actual weight representations. A tree table ("16" or "16,12,10", see
// --draft-tree-nodes) runs every round family through DFlash2 tree verification; "auto" lets the
// automatic widths choose.
int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const auto k         = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 15U;
        const bool graph     = argc > 2 ? std::stoi(argv[2]) != 0 : true;
        const bool optimized = argc > 3 ? std::stoi(argv[3]) != 0 : true;
        const auto batch     = argc > 4 ? static_cast<unsigned>(std::stoul(argv[4])) : 8U;
        infernix::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 2304;
        options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(2304 * batch);
        options.prefill_chunk   = 2304;
        options.max_concurrency = batch;
        options.context_cache.device_state_slots = argc > 7 ? std::stoul(argv[7]) : 3U;
        options.use_cuda_graph                   = graph;
        options.enable_vision                    = argc > 6 && std::stoi(argv[6]) != 0;
        options.kv_cache = infernix::test::parse_kv_cache_storage(argc > 5 ? argv[5] : "bf16");
        auto penalty     = request(24);
        penalty.execution.sampling.presence_penalty  = 0.5F;
        penalty.execution.sampling.frequency_penalty = 0.25F;
        options.speculative.backend                  = infernix::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens             = k;
        options.speculative.proposal_head =
            optimized ? infernix::ProposalHead::Optimized : infernix::ProposalHead::Full;
        const bool tree                     = argc > 8;
        const bool tree_auto                = tree && std::string(argv[8]) == "auto";
        options.speculative.draft_tree_auto = tree_auto;
        if (tree && !tree_auto) {
            std::string table = argv[8];
            std::size_t entry = 0;
            for (std::size_t begin = 0; begin <= table.size() && entry < 8; ++entry) {
                const std::size_t end = std::min(table.find(',', begin), table.size());
                options.speculative.draft_tree_nodes[entry] =
                    static_cast<std::uint32_t>(std::stoul(table.substr(begin, end - begin)));
                begin = end + 1;
            }
            for (; entry < 8; ++entry)
                options.speculative.draft_tree_nodes[entry] =
                    options.speculative.draft_tree_nodes[entry - 1];
        }
        infernix::Engine engine(options);
        require(engine.memory_summary().kv_cache == options.kv_cache,
                "Engine did not select the requested KV dtype");
        if (graph) {
            const infernix::MemorySummary memory = engine.memory_summary();
            require(memory.cuda_graph_measured_bytes != 0 &&
                        memory.cuda_graph_measured_bytes <= memory.cuda_graph_allowance_bytes,
                    ("CUDA Graph memory " + std::to_string(memory.cuda_graph_measured_bytes) +
                     " exceeds its allowance " + std::to_string(memory.cuda_graph_allowance_bytes))
                        .c_str());
        }
        const auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");
        infernix::test::speculative_page_boundary(engine);
        const auto first = engine.generate(engine.prepare_tokens(prompt), request(24));
        valid(first, 24);
        const auto& reference = first.generated_token_ids;
        require(first.speculative.accepted_tokens != 0, "real draft fixture accepted no proposal");
        // Automatic widths time the widest tree first, so they verify trees from the start too.
        require(!tree || (!tree_auto && options.speculative.draft_tree_nodes[0] == 0) ||
                    first.speculative.tree_rounds != 0,
                "a tree table did not verify tree rounds");
        // Every side round accepts at least the draft where its path leaves the main chain.
        require(first.speculative.tree_side_rounds <= first.speculative.tree_side_accepted_tokens &&
                    first.speculative.tree_side_accepted_tokens <=
                        first.speculative.accepted_tokens,
                "inconsistent tree side-branch statistics");
        const auto penalized = engine.generate(engine.prepare_tokens(prompt), penalty);
        valid(penalized, 24);

        infernix::PromptInput thinking_prompt;
        thinking_prompt.options.enable_thinking = true;
        thinking_prompt.messages.push_back({
            .role  = infernix::ChatRole::User,
            .parts = {{.kind = infernix::MessagePartKind::Text,
                       .text = "Explain why there are infinitely many prime numbers."}},
        });
        auto thinking_request                      = request(64);
        thinking_request.execution.thinking.budget = 1;
        const auto forced = engine.generate(engine.prepare(thinking_prompt), thinking_request);
        valid(forced, 64);
        require(forced.thinking.applied && forced.thinking.injected_tokens != 0,
                "DFlash2 did not commit the forced thinking-control suffix");

        // All rows share a known target prefix, while their budgets force P=0, partial and full W.
        std::vector<infernix::GenerationHandle> handles;
        for (unsigned row = 0; row < batch; ++row) {
            handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(2 + row * 3)));
        }
        for (unsigned row = 0; row < batch; ++row) {
            const auto result = handles[row].wait();
            valid(result, 2 + row * 3);
        }
        // Rejection RNG uses the same logical positions and seed for repeat requests.
        auto sampled                                 = request(16);
        sampled.execution.sampling.temperature       = 0.8F;
        sampled.execution.sampling.top_p             = 0.9F;
        sampled.execution.sampling.top_k             = 20;
        sampled.execution.sampling.presence_penalty  = 0.3F;
        sampled.execution.sampling.frequency_penalty = 0.2F;
        sampled.execution.sampling.seed              = 42;
        const auto sample1 = engine.generate(engine.prepare_tokens(prompt), sampled);
        const auto sample2 = engine.generate(engine.prepare_tokens(prompt), sampled);
        valid(sample1, 16);
        valid(sample2, 16);
        // Automatic widths follow measured round time, so the two requests may verify different
        // trees and sample different (equally distributed) text.
        require(tree_auto || sample1.generated_token_ids == sample2.generated_token_ids,
                "same DFlash2 seed and inputs did not reproduce the conditional path");

        // Terminal flush and fork must retain the consumed frontier for a later request.
        const auto retained = engine.generate(engine.prepare_tokens(prompt), request(12, true));
        auto continuation   = prompt;
        continuation.insert(continuation.end(), retained.generated_token_ids.begin(),
                            retained.generated_token_ids.end());
        continuation.push_back(198);
        const auto reused = engine.generate(engine.prepare_tokens(continuation), request(8, true));
        const auto fresh  = engine.generate(engine.prepare_tokens(continuation), request(8, false));
        valid(reused, 8);
        valid(fresh, 8);
        require(reused.reused_prompt_tokens != 0 && fresh.reused_prompt_tokens == 0,
                "DFlash2 prefix restore did not reuse its retained frontier");

        if (k >= 7) {
            bool checked_partial = false;
            for (std::size_t i = 1; i < std::min<std::size_t>(k, reference.size()); ++i) {
                if (std::find(reference.begin(), reference.begin() + i, reference[i]) !=
                    reference.begin() + i) {
                    continue;
                }
                auto stopped_options = request(24, false);
                stopped_options.stop.token_ids.push_back(reference[i]);
                const auto stopped =
                    engine.generate(engine.prepare_tokens(prompt), stopped_options);
                const auto licensed = 1 + stopped.speculative.rounds +
                                      stopped.speculative.accepted_tokens +
                                      stopped.speculative.fallback_steps;
                if (stopped.generated_token_ids.size() >= licensed) { continue; }
                require(stopped.finish_reason == infernix::FinishReason::StopToken &&
                            !stopped.generated_token_ids.empty() &&
                            stopped.generated_token_ids.back() == reference[i],
                        "partial terminal did not stop at its token");
                auto follow = prompt;
                follow.insert(follow.end(), stopped.generated_token_ids.begin(),
                              stopped.generated_token_ids.end());
                follow.push_back(198);
                const auto after_stop =
                    engine.generate(engine.prepare_tokens(follow), request(8, false));
                valid(after_stop, 8);
                checked_partial = true;
                break;
            }
            require(checked_partial, "fixture did not exercise a stop within a licensed block");
        }
        if (options.enable_vision) {
            for (const auto kind : {infernix::MediaKind::Image, infernix::MediaKind::Video}) {
                const auto image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                const auto reused_image =
                    engine.generate(engine.prepare(media_prompt(kind)), request(8, true));
                valid(image, 8);
                valid(reused_image, 8);
                require(image.prompt.has_media && image.timings.vision_seconds > 0 &&
                            reused_image.reused_prompt_tokens != 0 &&
                            reused_image.timings.vision_seconds == 0.0,
                        "Vision DFlash2 capture/restore recomputed retained media");
            }
        }
        const auto stats = engine.runtime_stats();
        require(stats.device_backend_kv_occupied_pages == 0 && stats.backend_kv_d2h_bytes == 0 &&
                    stats.backend_kv_h2d_bytes == 0,
                "DFlash2 allocated or transferred a full backend KV pool");
        if (k == 15) {
            // One oversized prefill replaces the ring, then decode appends across its wrap point.
            auto long_prompt = std::vector<infernix::TokenId>(2100, 198);
            long_prompt.insert(long_prompt.end(), prompt.begin(), prompt.end());
            const auto long_run =
                engine.generate(engine.prepare_tokens(long_prompt), request(12, true));
            valid(long_run, 12);
            long_prompt.insert(long_prompt.end(), long_run.generated_token_ids.begin(),
                               long_run.generated_token_ids.end());
            const auto long_reuse =
                engine.generate(engine.prepare_tokens(long_prompt), request(6, true));
            const auto long_fresh =
                engine.generate(engine.prepare_tokens(long_prompt), request(6, false));
            valid(long_reuse, 6);
            valid(long_fresh, 6);
            require(long_reuse.reused_prompt_tokens > 2048 && long_fresh.reused_prompt_tokens == 0,
                    "DFlash2 ring wrap lost its retained frontier");
        }
        if (k == 15) {
            auto tail_prompt   = std::vector<infernix::TokenId>(options.max_context - 4, 198);
            tail_prompt.back() = prompt.back();
            const auto tail    = engine.generate(engine.prepare_tokens(tail_prompt), request(9));
            require(tail.finish_reason == infernix::FinishReason::ContextCapacity &&
                        tail.generated_token_ids.size() == 5,
                    "full proposal window escaped the target context capacity tail");
        }
        std::cout << "ok K=" << k << " B=" << batch << " graph=" << graph
                  << " optimized=" << optimized << " accepted=" << first.speculative.accepted_tokens
                  << "/" << first.speculative.drafted_tokens
                  << " tree_rounds=" << first.speculative.tree_rounds
                  << " side_rounds=" << first.speculative.tree_side_rounds
                  << " state_d2h=" << stats.state_d2h_count
                  << " state_h2d=" << stats.state_h2d_count << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
