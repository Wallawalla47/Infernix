#include "infernix/engine.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

// Prepended to the test prompts on a Qwen4Exp artifact: its hybrid prefix cache keeps whole 64-token
// blocks, so the prompts must span several for the reuse checks to see a cached prefix.
std::string g_preamble;

infernix::PromptInput prompt(bool thinking = false) {
    infernix::PromptInput input;
    input.options.enable_thinking = thinking;
    input.messages.push_back({.role  = infernix::ChatRole::User,
                              .parts = {{.kind = infernix::MessagePartKind::Text,
                                         .text = g_preamble + "Return the requested answer."}}});
    return input;
}

infernix::RequestOptions literal(const std::string& answer, float temperature = 0.0f) {
    infernix::RequestOptions request;
    request.constraint =
        infernix::OutputConstraint::grammar("root ::= " + nlohmann::json(answer).dump());
    request.execution.requested_output_tokens = 160;
    request.execution.sampling.temperature    = temperature;
    request.execution.sampling.top_k          = 20;
    request.execution.sampling.seed           = 78123;
    return request;
}

struct Sink final : infernix::OutputSink {
    std::string content;

    void start(infernix::GenerationStart) override {}

    void progress(infernix::PromptProgress) override {}

    void timing(infernix::GenerationTimingObservation) override {}

    void publish(infernix::OutputDelta delta) override {
        if (delta.channel == infernix::OutputChannel::Content) { content += delta.text; }
    }
};

void choice_and_regex(infernix::Engine& engine, unsigned concurrency, bool speculative) {
    const std::vector<std::string> choices{"route_search", "route_calculate", "route_answer"};
    const std::regex pattern("(BUG|TASK)-[0-9]{4}");
    auto choice             = literal("unused", 0.8f);
    choice.constraint       = infernix::OutputConstraint::choice(choices);
    auto regex              = choice;
    regex.constraint        = infernix::OutputConstraint::regex("(BUG|TASK)-[0-9]{4}");
    const auto valid_choice = [&](const std::string& value) {
        return std::find(choices.begin(), choices.end(), value) != choices.end();
    };
    for (const auto& request : {choice, regex}) {
        Sink sink;
        const auto result = engine.generate(engine.prepare(prompt()), request, &sink);
        require(result.finish_reason == infernix::FinishReason::StopToken &&
                    sink.content == result.content,
                "choice/regex streaming or completion changed");
        require(request.constraint->kind == infernix::OutputConstraintKind::Choice
                    ? valid_choice(result.content)
                    : std::regex_match(result.content, pattern),
                "choice/regex produced content outside its language");
        std::cout << "choice/regex: \"" << result.content << "\" in " << result.generated_token_ids.size()
                  << " tokens, " << result.speculative.rounds << " speculative rounds\n";
        if (speculative)
            require(result.speculative.rounds > 0, "choice/regex bypassed speculation");
    }
    const std::string literal_bytes = " 你好 \"a|b\\c\"\n";
    auto exact                      = choice;
    exact.constraint                = infernix::OutputConstraint::choice({literal_bytes});
    require(engine.generate(engine.prepare(prompt()), exact).content == literal_bytes,
            "choice changed Unicode, whitespace or regex metacharacters");
    auto thinking                      = choice;
    thinking.execution.thinking.budget = 2;
    const auto thought                 = engine.generate(engine.prepare(prompt(true)), thinking);
    require(valid_choice(thought.content) &&
                thought.finish_reason == infernix::FinishReason::StopToken,
            "choice constrained the thinking channel or lost the content boundary");
    for (const auto& constraint : {infernix::OutputConstraint::choice({"TASK-12", "TASK-123"}),
                                   infernix::OutputConstraint::regex("(BUG|TASK)-[0-9]{4}")}) {
        auto input                 = prompt();
        input.options.continuation = infernix::PromptContinuationMode::ContinueFinalAssistant;
        input.messages.push_back(
            {.role  = infernix::ChatRole::Assistant,
             .parts = {{.kind = infernix::MessagePartKind::Text, .text = "TASK-"}}});
        auto request       = choice;
        request.constraint = constraint;
        const auto result  = engine.generate(engine.prepare(input), request);
        const auto full    = "TASK-" + result.content;
        require(result.finish_reason == infernix::FinishReason::StopToken &&
                    (constraint.kind == infernix::OutputConstraintKind::Choice
                         ? full == "TASK-12" || full == "TASK-123"
                         : std::regex_match(full, pattern)),
                "choice/regex continuation did not consume its existing prefix");
    }
    for (const auto& constraint :
         {infernix::OutputConstraint::choice({""}), infernix::OutputConstraint::regex("")}) {
        auto request       = choice;
        request.constraint = constraint;
        const auto result  = engine.generate(engine.prepare(prompt()), request);
        require(result.content.empty() && result.finish_reason == infernix::FinishReason::StopToken,
                "empty choice/regex did not finish with empty content");
    }
    auto limited                              = regex;
    limited.constraint                        = infernix::OutputConstraint::regex("[ab]{1000}");
    limited.execution.requested_output_tokens = 2;
    const auto partial                        = engine.generate(engine.prepare(prompt()), limited);
    require(partial.finish_reason == infernix::FinishReason::OutputLimit &&
                !partial.content.empty() && partial.content.size() < 1000 &&
                partial.content.find_first_not_of("ab") == std::string::npos,
            "regex truncation was reported as a completed match");
    infernix::RequestOptions free;
    free.execution.requested_output_tokens = 8;
    std::vector<infernix::GenerationHandle> handles;
    for (unsigned row = 0; row < concurrency; ++row)
        handles.push_back(engine.submit(engine.prepare(prompt()), row % 3 == 0   ? choice
                                                                  : row % 3 == 1 ? regex
                                                                                 : free));
    for (unsigned row = 0; row < concurrency; ++row) {
        const auto result = handles[row].wait();
        if (row % 3 != 2)
            require(result.finish_reason == infernix::FinishReason::StopToken &&
                        (row % 3 == 0 ? valid_choice(result.content)
                                      : std::regex_match(result.content, pattern)),
                    "mixed batch used another row's choice/regex");
    }
    std::cout << "choice/regex: literals, fullmatch, thinking, continuation, EOS, truncation and "
                 "mixed rows passed\n";
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const std::string backend = argc > 1 ? argv[1] : "none";
        infernix::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 1024;
        options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(2048);
        options.max_concurrency = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3])) : 2;
        options.enable_vision   = argc > 4 && std::string_view(argv[4]) == "vision";
        options.prefill_chunk   = 128;
        options.kv_cache        = infernix::KvCacheStorage::Fp8E4M3Row256;
        options.context_cache.hybrid.device_snapshot_slots = 8;
        options.context_cache.host_capacity_bytes          = 512ULL << 20;
        // The prefix cache keeps whole 64-token blocks and snapshots inside the prompt, so the
        // prompts carry a fixed preamble long enough for it to keep.
        for (int line = 1; line <= 12; ++line) {
            g_preamble += "Note " + std::to_string(line) +
                          ": this line belongs to a fixed preamble that only makes the prompt long "
                          "enough for the prefix cache to keep.\n";
        }
        // INFERNIX_TEST_NGRAM_VOLUME names a Qwen4Exp artifact's n-gram volume. That model keeps
        // its snapshots in its default Host tier and drafts adaptively: a one-row round verifies
        // only confident drafts, so a short sampled answer may take no round.
        const char* volume = std::getenv("INFERNIX_TEST_NGRAM_VOLUME");
        const bool qwen4   = volume != nullptr && *volume != '\0';
        if (qwen4) {
            options.ngram_volume_path                 = volume;
            options.context_cache.hybrid.device_snapshot_slots = {};
            options.context_cache.host_capacity_bytes          = {};
        }
        options.use_cuda_graph = argc < 3 || std::string_view(argv[2]) != "eager";
        if (backend == "mtp")
            options.speculative.backend = infernix::SpeculativeBackend::Mtp;
        else if (backend == "dflash")
            options.speculative.backend = infernix::SpeculativeBackend::DFlash;
        else if (backend == "dflash2")
            options.speculative.backend = infernix::SpeculativeBackend::DFlash2;
        else
            require(backend == "none", "unknown backend");
        if (backend != "none") {
            const auto draft_tokens           = std::getenv("INFERNIX_TEST_DRAFT_TOKENS");
            options.speculative.draft_tokens  = draft_tokens ? std::stoul(draft_tokens) : 3;
            options.speculative.proposal_head = infernix::ProposalHead::Optimized;
            // Copy rounds verify their own drafts against the grammar masks.
            if (const char* ngram = std::getenv("INFERNIX_TEST_NGRAM_DRAFT_TOKENS")) {
                options.speculative.ngram_draft_tokens = std::stoul(ngram);
                options.speculative.ngram_min_match    = 4;
            }
            // A DFlash2 tree width for every batch size: rounds with a constrained row still
            // verify the chain, unconstrained rows of the mixed batches verify trees.
            if (const char* tree = std::getenv("INFERNIX_TEST_DRAFT_TREE_NODES")) {
                options.speculative.draft_tree_nodes.fill(std::stoul(tree));
            }
        }
        infernix::Engine engine(options);
        choice_and_regex(engine, options.max_concurrency, backend != "none" && !qwen4);
        const std::string answer =
            "  {\"value\":\"你好\",\"literal\":\"<tool_call>x</tool_call>\"}";
        for (float temperature : {0.0f, 0.8f}) {
            Sink sink;
            auto request                                 = literal(answer, temperature);
            request.execution.sampling.presence_penalty  = 0.5f;
            request.execution.sampling.frequency_penalty = 0.125f;
            const auto result = engine.generate(engine.prepare(prompt()), request, &sink);
            require(result.content == answer && sink.content == answer && result.tool_calls.empty(),
                    "grammar content bytes or streaming publication changed");
            require(result.finish_reason == infernix::FinishReason::StopToken,
                    "grammar did not finish through EOS");
            if (backend != "none")
                require(result.speculative.rounds > 0, "constraint bypassed speculative backend");
            std::cout << "literal temp=" << temperature << " reuse=" << result.reused_prompt_tokens
                      << " prepare_ms=" << result.timings.prepare_seconds * 1000
                      << " decode_ms=" << result.timings.decode_seconds * 1000 << '\n';
        }

        auto thinking                      = literal(answer);
        thinking.execution.thinking.budget = 2;
        auto thought = engine.generate(engine.prepare(prompt(true)), thinking);
        require(thought.content == answer && thought.thinking.applied &&
                    thought.thinking.injected_tokens > 0,
                "thinking control broke constrained content");

        auto continued = prompt();
        continued.options.continuation = infernix::PromptContinuationMode::ContinueFinalAssistant;
        const std::string prefix       = "{\"value\":";
        const std::string whole        = prefix + "\"你好\"}";
        continued.messages.push_back(
            {.role  = infernix::ChatRole::Assistant,
             .parts = {{.kind = infernix::MessagePartKind::Text, .text = prefix}}});
        const auto suffix = engine.generate(engine.prepare(continued), literal(whole));
        require(prefix + suffix.content == whole,
                "continuation matcher did not start at rendered content prefix");

        infernix::RequestOptions free;
        free.execution.requested_output_tokens = 12;
        free.execution.sampling.temperature    = 0.0f;
        std::vector<infernix::GenerationHandle> mixed;
        for (unsigned row = 0; row < options.max_concurrency; ++row) {
            const std::string row_answer = answer + std::to_string(row);
            mixed.push_back(engine.submit(engine.prepare(prompt()),
                                          row % 2 ? free : literal(row_answer, 0.8f)));
        }
        for (unsigned row = 0; row < mixed.size(); ++row) {
            const auto result = mixed[row].wait();
            if (row % 2 == 0)
                require(result.content == answer + std::to_string(row),
                        "mixed batch used another row's grammar");
        }
        if (options.enable_vision) {
            auto image_prompt = prompt();
            infernix::MessagePart image;
            image.kind               = infernix::MessagePartKind::Media;
            image.media.kind         = infernix::MediaKind::Image;
            image.media.media_type   = "image/x-portable-pixmap";
            image.media.source_name  = "pattern.ppm";
            const std::string header = "P6\n64 64\n255\n";
            image.media.bytes.assign(header.begin(), header.end());
            image.media.bytes.resize(header.size() + 64 * 64 * 3, 42);
            image_prompt.messages[0].parts.insert(image_prompt.messages[0].parts.begin(),
                                                  std::move(image));
            require(engine.generate(engine.prepare(image_prompt), literal(answer)).content ==
                        answer,
                    "Vision prefill lost first-token grammar binding");
        }

        auto truncated       = literal(std::string(1024, 'a'));
        truncated.constraint = infernix::OutputConstraint::grammar("root ::= \"a\"{1024}");
        truncated.execution.requested_output_tokens = 2;
        auto partial = engine.generate(engine.prepare(prompt()), truncated);
        require(partial.finish_reason == infernix::FinishReason::OutputLimit &&
                    !partial.content.empty() &&
                    partial.content.find_first_not_of('a') == std::string::npos,
                "length-limited output is not a valid grammar prefix");
        const auto raw_prompt = engine.tokenize_text(g_preamble + "A raw prompt checkpoint. Answer: ");
        auto seed             = literal("yes");
        seed.execution.requested_output_tokens = 1;
        (void)engine.generate(engine.prepare_tokens(raw_prompt), seed);
        const auto hit = engine.generate(engine.prepare_tokens(raw_prompt), literal("no"));
        std::cout << "exact hit: content=\"" << hit.content << "\" reused=" << hit.reused_prompt_tokens << " of "
                  << raw_prompt.size() << '\n';
        // The prefix cache resumes from its last snapshot before the prompt's end and prefills the
        // rest; the first token must still follow this request's grammar.
        require(hit.content == "no" && hit.reused_prompt_tokens > 0 &&
                    hit.reused_prompt_tokens < raw_prompt.size(),
                "exact prefix hit reused grammar state or bypassed first-token mask");
        auto raw = engine.generate(engine.prepare_tokens(engine.tokenize_text("Answer: ")),
                                   literal(answer));
        require(raw.content == answer, "raw-token prompt did not constrain newly generated bytes");

        // Cross a context profile during generation, then return to short, differently masked
        // requests. This exercises repeated handoffs and profile changes on the same Program.
        std::vector<infernix::TokenId> long_prompt(480, 198);
        std::string long_answer;
        for (unsigned i = 0; i < 80; ++i) { long_answer += std::to_string(i) + ":a;"; }
        auto long_request                              = literal(long_answer);
        long_request.execution.allow_prefix_reuse      = false;
        long_request.execution.requested_output_tokens = 500;
        const auto long_result = engine.generate(engine.prepare_tokens(long_prompt), long_request);
        require(long_result.content == long_answer, "context profile change lost grammar position");
        for (const std::string answer_after : {"after-long", "different mask"}) {
            auto next                         = literal(answer_after);
            next.execution.allow_prefix_reuse = false;
            require(engine.generate(engine.prepare(prompt()), next).content == answer_after,
                    "short request reused a stale draft handoff or mask");
        }

        // JSON entry points use the same tokenizer, row mapping and transaction path as GBNF.
        const nlohmann::ordered_json schema = {
            {"type", "object"},
            {"properties",
             {{"description", {{"type", "string"}, {"enum", {"你好", "code"}}}},
              {"values",
               {{"type", "array"},
                {"prefixItems",
                 {{{"type", "number"}, {"minimum", 1e-8}, {"maximum", 2e-8}},
                  {{"type", "number"}, {"exclusiveMinimum", 0.1}, {"maximum", 0.2}}}},
                {"items", false},
                {"minItems", 2},
                {"maxItems", 2}}}}},
            {"required", {"description", "values"}},
            {"additionalProperties", false}};
        auto json_request       = literal("unused", 0.8f);
        json_request.constraint = infernix::OutputConstraint::json_schema(schema.dump());
        const auto record       = [&](const nlohmann::ordered_json& spec,
                                const infernix::GenerationResult& value) {
            require(value.finish_reason == infernix::FinishReason::StopToken,
                          "JSON generation was truncated");
            require(nlohmann::json::accept(value.content), "JSON output is not parseable");
            if (const char* report = std::getenv("INFERNIX_TEST_SCHEMA_REPORT")) {
                std::ofstream file(report, std::ios::app);
                file << nlohmann::ordered_json{{"schema", spec}, {"content", value.content}}.dump()
                     << '\n';
                require(bool(file), "could not record schema validation output");
            }
        };
        const auto structured = engine.generate(engine.prepare(prompt()), json_request);
        record(schema, structured);
        const auto parsed = nlohmann::json::parse(structured.content);
        require(parsed.size() == 2 && parsed.contains("description") &&
                    parsed["values"].size() == 2 && parsed["values"][0] >= 1e-8 &&
                    parsed["values"][0] <= 2e-8 && parsed["values"][1] > 0.1 &&
                    parsed["values"][1] <= 0.2,
                "schema fields missing");
        auto json_thinking                      = json_request;
        json_thinking.execution.thinking.budget = 2;
        record(schema, engine.generate(engine.prepare(prompt(true)), json_thinking));
        auto numeric_prefix = prompt();
        numeric_prefix.options.continuation =
            infernix::PromptContinuationMode::ContinueFinalAssistant;
        const std::string partial_number = "{\"description\":\"你好\",\"values\":[1.5e-";
        numeric_prefix.messages.push_back(
            {.role  = infernix::ChatRole::Assistant,
             .parts = {{.kind = infernix::MessagePartKind::Text, .text = partial_number}}});
        auto numeric_suffix    = engine.generate(engine.prepare(numeric_prefix), json_request);
        numeric_suffix.content = partial_number + numeric_suffix.content;
        record(schema, numeric_suffix);
        auto json_prompt                      = prompt();
        json_prompt.messages[0].parts[0].text = "Return exactly the JSON object {\"ok\":true}.";
        auto object_request                   = json_request;
        object_request.constraint             = infernix::OutputConstraint::json_object();
        object_request.execution.sampling.temperature = 0;
        record({{"type", "object"}}, engine.generate(engine.prepare(json_prompt), object_request));

        const auto fixed  = nlohmann::ordered_json{{"const", {{"text", "你好\n"}, {"n", 2}}}};
        auto continuation = prompt();
        continuation.options.continuation = infernix::PromptContinuationMode::ContinueFinalAssistant;
        const std::string json_prefix = "{\"text\":";
        continuation.messages.push_back(
            {.role  = infernix::ChatRole::Assistant,
             .parts = {{.kind = infernix::MessagePartKind::Text, .text = json_prefix}}});
        auto fixed_request       = json_request;
        fixed_request.constraint = infernix::OutputConstraint::json_schema(fixed.dump());
        auto continued_json      = engine.generate(engine.prepare(continuation), fixed_request);
        continued_json.content   = json_prefix + continued_json.content;
        record(fixed, continued_json);
        auto changed       = json_request;
        changed.constraint = infernix::OutputConstraint::json_schema(R"({"const":{"changed":true}})");
        const auto changed_result = engine.generate(engine.prepare(prompt()), changed);
        record({{"const", {{"changed", true}}}}, changed_result);
        require(changed_result.reused_prompt_tokens > 0,
                "schema switch did not exercise prefix reuse");

        std::vector<infernix::GenerationHandle> json_batch;
        for (unsigned row = 0; row < options.max_concurrency; ++row) {
            auto selected = row % 3 == 0 ? json_request : row % 3 == 1 ? object_request : free;
            json_batch.push_back(engine.submit(engine.prepare(json_prompt), selected));
        }
        for (unsigned row = 0; row < json_batch.size(); ++row) {
            const auto value = json_batch[row].wait();
            if (row % 3 == 0)
                record(schema, value);
            else if (row % 3 == 1)
                record({{"type", "object"}}, value);
        }
        auto limited       = json_request;
        limited.constraint = infernix::OutputConstraint::json_schema(
            R"({"type":"array","items":{"const":"word"},"minItems":128,"maxItems":128})");
        limited.execution.requested_output_tokens = 2;
        require(engine.generate(engine.prepare(prompt()), limited).finish_reason ==
                    infernix::FinishReason::OutputLimit,
                "JSON truncation was reported as normal completion");

        auto invalid       = literal("yes");
        invalid.constraint = infernix::OutputConstraint::grammar("root ::= missing");
        bool rejected      = false;
        try {
            (void)engine.submit(engine.prepare(prompt()), invalid);
        } catch (const infernix::RequestError& error) {
            rejected = error.kind() == infernix::RequestErrorKind::InvalidGrammar;
        }
        require(rejected && engine.is_available(),
                "invalid grammar was not isolated before admission");
        require(engine.generate(engine.prepare(prompt()), literal("yes")).content == "yes",
                "Engine did not remain usable after rejected request");
        std::cout << "constraints " << backend << (options.use_cuda_graph ? " graph" : " eager")
                  << ": content, sampling, thinking, continuation, mixed batch, truncation, raw "
                     "input, JSON/schema passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GBNF integration: " << error.what() << '\n';
        return 1;
    }
}
