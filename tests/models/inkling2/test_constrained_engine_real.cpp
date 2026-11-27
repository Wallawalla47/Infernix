// Inkling 2's constrained decoding through the public Engine on a real artifact (src/models/inkling2/frontend/
// constraints.h; the Program's masks). The eager Engine without a drafter, one request at a time, is the reference;
// every other route must give its tokens:
// 1. Graphs: the forward-only graph with the sample after the masks gives the reference's tokens.
// 2. DSpark (auto with graphs, 7 drafts eager): each verify column samples under its position's mask after the drafts
//    before it, so speculative output is plain decoding's; the acceptance is reported.
// 3. Beside unconstrained requests (C = 4, graphs): each constrained request gives its solo tokens, and each
//    unconstrained one its own.
// 4. --lm-head-draft (the drafter's shortlist) beside DSpark gives the reference's tokens.
// 4b. DSpark rounds of several rows (graphs, C = 4, auto, auto with a skewed deal and 7 drafts): constrained requests
//    drafting beside an unconstrained one, each row's verify columns under its own masks, give their solo tokens; under
//    the skewed deal rows verified none and all of their drafts.
// 5. Token constraints (TokenConstraint, /v1/decide's generated primitives) on a prompt that opens the answer: every
//    token in its step's set, forced steps exact, one draw a step; step 0's restricted probabilities, mass, observed
//    log-probabilities and captured logits against an unconstrained readout of the same position; graphs, C = 4
//    beside other requests and a DSpark Engine give the eager tokens and draws bit for bit.
// Every constrained request's output is in its language: the JSON answer parses with the schema's shape, the regex and
// choice texts match, a required choice calls a declared tool (the strict one with its schema's shape).
// Reported, not judged: a tools-declared free turn's raw text (the headers the model writes), and a turn that answers
// in text under ToolConstraintMode::Basic against the same turn free (the same tokens when the model keeps the
// framing the grammar allows).
//
//   INFERNIX_INKLING2_ARTIFACT  an artifact converted with the inkling2_mini recipe (its DSpark drafter included)

#include "infernix/engine.h"
#include "models/inkling2/drafter_choice.h"
#include "models/inkling2/program/test_taps.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;
using Json   = nlohmann::json;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void require_same(const std::string& label, const Tokens& a, const Tokens& b) {
    std::size_t at = 0;
    while (at < a.size() && at < b.size() && a[at] == b[at]) { ++at; }
    require(!a.empty() && a == b, label + ": tokens differ at " + std::to_string(at) + " of " + std::to_string(a.size()) +
                                      "/" + std::to_string(b.size()));
    std::cout << label << ": " << a.size() << " tokens identical\n";
}

infernix::EngineOptions engine_options(const std::filesystem::path& artifact, bool graphs, std::uint32_t concurrency,
                                       bool dspark = false, bool automatic = true) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 2048;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(concurrency * 2048);
    options.max_concurrency      = concurrency;
    options.max_pending_requests = 16;
    options.context_cache.host_capacity_bytes = std::size_t{2} << 30U;
    options.use_cuda_graph       = graphs;
    if (dspark) {
        infernix::test::inkling2::use_drafter(options.speculative);
        options.speculative.draft_tokens_auto = automatic;
        options.speculative.draft_tokens      = 7;
    }
    options.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        if (d.level >= infernix::DiagnosticLevel::Warning) { std::cerr << "[engine] " << d.message << '\n'; }
    };
    return options;
}

infernix::PromptInput chat(const std::string& text, const std::vector<std::string>& tools = {}) {
    infernix::PromptInput input;
    input.messages.push_back({.role = infernix::ChatRole::User, .parts = {{.text = text}}});
    input.options.reasoning_effort = infernix::ReasoningEffort::Low;
    input.options.tool_jsons       = tools;
    return input;
}

const std::string kWeather =
    R"({"type":"function","function":{"name":"get_weather","description":"Current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["c","f"]}},"required":["city"]}}})";
const std::string kSearch =
    R"({"type":"function","function":{"name":"search","description":"Search the web.","strict":true,"parameters":{"type":"object","properties":{"query":{"type":"string"},"results":{"type":"integer"}},"required":["query","results"],"additionalProperties":false}}})";
const std::string kPerson =
    R"({"type":"object","properties":{"name":{"type":"string"},"born":{"type":"integer"},"fields":{"type":"array","items":{"type":"string"}}},"required":["name","born","fields"],"additionalProperties":false})";

struct Case {
    std::string name;
    infernix::PromptInput prompt;
    infernix::RequestOptions options;
    std::function<void(const infernix::GenerationResult&)> check;
};

infernix::RequestOptions request(float temperature, std::uint64_t seed) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = 400;
    options.execution.sampling.temperature    = temperature;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.seed           = seed;
    options.execution.thinking.budget         = 48; // keeps the runs short: the answer follows the control
    return options;
}

std::vector<Case> cases() {
    std::vector<Case> out;
    {
        infernix::RequestOptions o = request(0.0F, 0);
        o.constraint               = infernix::OutputConstraint::json_schema(kPerson);
        out.push_back({"json schema, greedy", chat("Give Marie Curie's name, birth year and fields of science as JSON."), o,
                       [](const infernix::GenerationResult& r) {
                           const Json value = Json::parse(r.content);
                           require(value.is_object() && value.size() == 3 && value["name"].is_string() &&
                                       value["born"].is_number_integer() && value["fields"].is_array(),
                                   "json schema, greedy: the answer has the schema's shape: " + r.content);
                       }});
    }
    {
        infernix::RequestOptions o = request(1.0F, 7);
        o.constraint               = infernix::OutputConstraint::json_schema(kPerson);
        out.push_back({"json schema, sampled", chat("Describe Alan Turing as JSON: name, birth year, fields."), o,
                       [](const infernix::GenerationResult& r) {
                           const Json value = Json::parse(r.content);
                           require(value.is_object() && value.size() == 3, "json schema, sampled: the schema's shape: " + r.content);
                       }});
    }
    {
        infernix::RequestOptions o = request(0.0F, 0);
        o.constraint               = infernix::OutputConstraint::regex("[0-9]{4}-[0-9]{2}-[0-9]{2}");
        out.push_back({"regex, greedy", chat("On what date did Apollo 11 land on the Moon? Answer as YYYY-MM-DD."), o,
                       [](const infernix::GenerationResult& r) {
                           require(std::regex_match(r.content, std::regex("[0-9]{4}-[0-9]{2}-[0-9]{2}")),
                                   "regex, greedy: the answer matches: " + r.content);
                       }});
    }
    {
        infernix::RequestOptions o = request(0.0F, 0);
        o.constraint               = infernix::OutputConstraint::choice({"positive", "negative", "neutral"});
        out.push_back({"choice, greedy", chat("Sentiment of: 'The service was slow but the food was wonderful.'"), o,
                       [](const infernix::GenerationResult& r) {
                           require(r.content == "positive" || r.content == "negative" || r.content == "neutral",
                                   "choice, greedy: a choice: " + r.content);
                       }});
    }
    {
        infernix::RequestOptions o = request(1.0F, 3);
        o.tool_choice              = {.mode = infernix::ToolChoiceMode::Required};
        out.push_back({"tools required, sampled", chat("Find three recent papers on solid-state batteries.", {kWeather, kSearch}),
                       o, [](const infernix::GenerationResult& r) {
                           require(!r.tool_calls.empty(), "tools required: the turn calls");
                           for (const auto& call : r.tool_calls) {
                               const Json args = Json::parse(call.arguments_json);
                               require(call.name == "get_weather" || call.name == "search", "tools required: a declared tool");
                               require(args.is_object(), "tools required: object arguments");
                               if (call.name == "search") {
                                   require(args.size() == 2 && args["query"].is_string() && args["results"].is_number_integer(),
                                           "tools required: the strict call's shape: " + call.arguments_json);
                               }
                           }
                       }});
    }
    {
        infernix::RequestOptions o = request(0.0F, 0);
        o.tool_choice = {.mode = infernix::ToolChoiceMode::Auto, .constraints = infernix::ToolConstraintMode::Basic};
        out.push_back({"tools auto (basic), greedy", chat("What is the weather in Oslo right now?", {kWeather}), o,
                       [](const infernix::GenerationResult& r) {
                           for (const auto& call : r.tool_calls) {
                               require(call.name == "get_weather" && Json::parse(call.arguments_json).is_object(),
                                       "tools auto: a declared call with object arguments");
                           }
                       }});
    }
    return out;
}

infernix::RequestOptions unconstrained(std::size_t i) {
    infernix::RequestOptions options = request(1.0F, 100 + i);
    options.execution.requested_output_tokens = 96;
    options.execution.thinking.budget.reset();
    return options;
}

const std::vector<std::string> kTexts = {"Explain how a compiler turns source code into machine instructions.",
                                         "Describe the water cycle to a ten-year-old."};

// /v1/decide's shape: a prompt that opens the answer (thinking off), and a token constraint over its first steps.
infernix::PromptInput answer_prompt(const std::string& text) {
    infernix::PromptInput input = chat(text);
    input.options.reasoning_effort.reset();
    input.options.enable_thinking = false;
    input.options.open_answer     = true;
    return input;
}

struct TokenCase {
    infernix::PromptInput prompt;
    infernix::RequestOptions options;
    std::vector<std::vector<infernix::TokenId>> steps;
};

// Steps: a digit (sampled), a forced "0", a digit set of four, a forced "7"; observing the ten digits at step 0, the
// first step's logits captured. Sampled at temperature 1 (seed `seed`) so the draws are the distribution's.
TokenCase token_case(infernix::Engine& engine, const std::string& text, std::uint64_t seed) {
    std::vector<infernix::TokenId> digits;
    for (int d = 0; d < 10; ++d) {
        const auto ids = engine.tokenize_text(std::to_string(d));
        require(ids.size() == 1, "a digit is one token");
        digits.push_back(ids.front());
    }
    TokenCase out{.prompt = answer_prompt(text)};
    out.steps = {digits, {digits[0]}, {digits[1], digits[3], digits[5], digits[9]}, {digits[7]}};
    out.options.execution.requested_output_tokens = static_cast<std::uint32_t>(out.steps.size());
    out.options.execution.sampling.temperature    = 1.0F;
    out.options.execution.sampling.top_k          = 20;
    out.options.execution.sampling.seed           = seed;
    out.options.execution.constraint.steps        = out.steps;
    out.options.execution.constraint.observe      = {digits, {}, {}, {}}; // one list a step: the digits at step 0
    out.options.execution.constraint.capture_logits = true;
    out.options.stop.include_model_defaults       = false;
    return out;
}

// The draws against the steps, and step 0 against an unconstrained readout of the same position (the same logits
// row before the mask): the restricted probabilities, the mass, the observed log-probabilities and the captured row.
void check_token_case(const std::string& label, infernix::Engine& engine, const TokenCase& c,
                      const infernix::GenerationResult& r) {
    require(r.generated_token_ids.size() == c.steps.size(), label + ": one token a step");
    require(r.constrained_draws.size() == c.steps.size(), label + ": one draw a step");
    for (std::size_t i = 0; i < c.steps.size(); ++i) {
        const auto& set = c.steps[i];
        require(std::find(set.begin(), set.end(), r.generated_token_ids[i]) != set.end(), label + ": a step's token");
        require(r.constrained_draws[i].token == r.generated_token_ids[i], label + ": a draw names its token");
        require(r.constrained_draws[i].probabilities.size() == set.size(), label + ": a draw covers its step's tokens");
        double sum = 0;
        for (const float p : r.constrained_draws[i].probabilities) { sum += p; }
        require(std::abs(sum - 1.0) < 1e-4, label + ": a draw's probabilities sum to 1");
        require(r.constrained_draws[i].mass > 0.0F && r.constrained_draws[i].mass <= 1.0001F, label + ": a draw's mass");
    }
    require(r.generated_token_ids[1] == c.steps[1][0] && r.generated_token_ids[3] == c.steps[3][0],
            label + ": the forced steps' tokens");
    infernix::RequestOptions readout;
    readout.execution.requested_output_tokens = 1;
    readout.execution.sampling.temperature    = 0.0F;
    readout.execution.readout_tokens          = c.steps[0];
    readout.stop.include_model_defaults       = false;
    const auto reference = engine.generate(engine.prepare(c.prompt), readout);
    require(reference.readout.has_value(), label + ": the reference readout");
    const auto& lp = reference.readout->logprobs;
    double total   = 0;
    for (const float v : lp) { total += std::exp(static_cast<double>(v)); }
    const auto& draw = r.constrained_draws[0];
    double worst     = 0;
    for (std::size_t i = 0; i < lp.size(); ++i) {
        worst = std::max(worst, std::abs(std::exp(static_cast<double>(lp[i])) / total - draw.probabilities[i]));
        worst = std::max(worst, std::abs(static_cast<double>(lp[i]) - draw.observed_logprobs[i]));
    }
    worst = std::max(worst, std::abs(total - draw.mass));
    worst = std::max(worst, std::abs(static_cast<double>(reference.readout->top_logprob) - draw.top_logprob));
    // The captured row is the logits the readout normalized: its digits' log-softmax is the readout's.
    double max = -1e30, sum = 0;
    for (const float v : draw.logits) { max = std::max(max, static_cast<double>(v)); }
    for (const float v : draw.logits) { sum += std::exp(static_cast<double>(v) - max); }
    for (std::size_t i = 0; i < lp.size(); ++i) {
        const double logprob = static_cast<double>(draw.logits[static_cast<std::size_t>(c.steps[0][i])]) - max - std::log(sum);
        worst                = std::max(worst, std::abs(logprob - lp[i]));
    }
    std::cout << label << ": " << r.generated_token_ids.size() << " constrained tokens, step 0 mass " << draw.mass
              << ", worst difference from the readout " << worst << '\n';
    require(worst < 2e-4, label + ": step 0's records differ from the readout of the same logits");
}

void report(const std::string& label, const infernix::GenerationResult& r) {
    std::cout << label << ": " << r.generated_token_ids.size() << " tokens, content " << Json(r.content.substr(0, 200)).dump();
    for (const auto& call : r.tool_calls) { std::cout << ", call " << call.name << ' ' << call.arguments_json; }
    if (r.constraint) {
        std::cout << ", constraint complete " << r.constraint->complete << " masks " << r.constraint->mask_positions;
    }
    if (r.speculative.rounds != 0) {
        std::cout << ", dspark " << r.speculative.accepted_tokens << "/" << r.speculative.drafted_tokens << " in "
                  << r.speculative.rounds << " rounds";
    }
    std::cout << '\n';
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    if (artifact == nullptr) {
        std::cout << "skip: set INFERNIX_INKLING2_ARTIFACT\n";
        return 77;
    }
    try {
        const std::vector<Case> all = cases();
        std::vector<Tokens> reference, plain;
        {
            infernix::Engine engine(engine_options(artifact, false, 1));
            for (const Case& c : all) {
                const auto result = engine.generate(engine.prepare(c.prompt), c.options);
                report("eager " + c.name, result);
                c.check(result);
                require(result.constraint.has_value(), c.name + ": a constrained request reports its constraint");
                reference.push_back(result.generated_token_ids);
            }
            for (std::size_t i = 0; i < kTexts.size(); ++i) {
                plain.push_back(engine.generate(engine.prepare(chat(kTexts[i])), unconstrained(i)).generated_token_ids);
            }
            // Reported: the free turn's raw text (the headers the model writes) and a Basic turn against the same turn
            // free.
            infernix::RequestOptions raw = request(0.0F, 0);
            raw.output.raw               = true;
            raw.stop.include_model_defaults = true;
            raw.tool_choice = {.mode = infernix::ToolChoiceMode::Auto, .constraints = infernix::ToolConstraintMode::Automatic};
            const auto free_turn = engine.generate(engine.prepare(chat("What is the weather in Oslo right now?", {kWeather})), raw);
            std::cout << "free raw turn: " << Json(free_turn.content).dump() << '\n';
            infernix::RequestOptions text = request(0.0F, 0);
            text.tool_choice = {.mode = infernix::ToolChoiceMode::Auto, .constraints = infernix::ToolConstraintMode::Automatic};
            const auto free_text = engine.generate(engine.prepare(chat("Say good morning in French.", {kWeather})), text);
            text.tool_choice.constraints = infernix::ToolConstraintMode::Basic;
            const auto basic_text = engine.generate(engine.prepare(chat("Say good morning in French.", {kWeather})), text);
            std::cout << "a text turn with a tool declared: free " << free_text.generated_token_ids.size()
                      << " tokens, basic " << basic_text.generated_token_ids.size() << " tokens, "
                      << (free_text.generated_token_ids == basic_text.generated_token_ids ? "identical" : "different")
                      << "; basic content " << Json(basic_text.content).dump() << '\n';
        }
        {
            infernix::Engine engine(engine_options(artifact, true, 4));
            for (std::size_t i = 0; i < all.size(); ++i) {
                require_same("graphs: " + all[i].name, reference[i],
                             engine.generate(engine.prepare(all[i].prompt), all[i].options).generated_token_ids);
            }
            // C = 4: three constrained requests beside an unconstrained one, then the other unconstrained one beside
            // three more constrained.
            std::vector<infernix::GenerationHandle> handles;
            handles.push_back(engine.submit(engine.prepare(chat(kTexts[0])), unconstrained(0)));
            for (std::size_t i = 0; i < 3; ++i) { handles.push_back(engine.submit(engine.prepare(all[i].prompt), all[i].options)); }
            require_same("C = 4: unconstrained 0", plain[0], handles[0].wait().generated_token_ids);
            for (std::size_t i = 0; i < 3; ++i) {
                require_same("C = 4: " + all[i].name, reference[i], handles[i + 1].wait().generated_token_ids);
            }
            handles.clear();
            for (std::size_t i = 3; i < all.size(); ++i) { handles.push_back(engine.submit(engine.prepare(all[i].prompt), all[i].options)); }
            handles.push_back(engine.submit(engine.prepare(chat(kTexts[1])), unconstrained(1)));
            for (std::size_t i = 3; i < all.size(); ++i) {
                require_same("C = 4: " + all[i].name, reference[i], handles[i - 3].wait().generated_token_ids);
            }
            require_same("C = 4: unconstrained 1", plain[1], handles.back().wait().generated_token_ids);
        }
        for (const bool automatic : {true, false}) {
            infernix::Engine engine(engine_options(artifact, automatic, 1, true, automatic));
            const std::string label = automatic ? "dspark (auto, graphs)" : "dspark (7 drafts, eager)";
            for (std::size_t i = 0; i < all.size(); ++i) {
                const auto result = engine.generate(engine.prepare(all[i].prompt), all[i].options);
                report(label + ": " + all[i].name, result);
                require_same(label + ": " + all[i].name, reference[i], result.generated_token_ids);
            }
        }
        for (const int mode : {0, 1, 2}) {
            // Rounds of several rows draft together (graphs, C = 4): three constrained requests beside an unconstrained
            // one, then the rest beside the other; each row's verify columns sample under its own masks. Auto, auto
            // with the skewed deal (testing::set_skewed_dealing: rows, constrained ones too, verify none and all of
            // their drafts; it must have dealt both) and 7 drafts.
            const bool automatic = mode != 2;
            const bool skewed    = mode == 1;
            infernix::models::inkling2::testing::set_skewed_dealing(skewed);
            const infernix::models::inkling2::testing::Dealing dealt = infernix::models::inkling2::testing::dealing();
            infernix::Engine engine(engine_options(artifact, true, 4, true, automatic));
            const std::string label =
                skewed ? "dspark rows (auto, skewed deal)" : automatic ? "dspark rows (auto)" : "dspark rows (7 drafts)";
            std::vector<infernix::GenerationHandle> handles;
            handles.push_back(engine.submit(engine.prepare(chat(kTexts[0])), unconstrained(0)));
            for (std::size_t i = 0; i < 3; ++i) { handles.push_back(engine.submit(engine.prepare(all[i].prompt), all[i].options)); }
            require_same(label + ", C = 4: unconstrained 0", plain[0], handles[0].wait().generated_token_ids);
            for (std::size_t i = 0; i < 3; ++i) {
                const auto result = handles[i + 1].wait();
                report(label + ", C = 4: " + all[i].name, result);
                require_same(label + ", C = 4: " + all[i].name, reference[i], result.generated_token_ids);
            }
            handles.clear();
            for (std::size_t i = 3; i < all.size(); ++i) { handles.push_back(engine.submit(engine.prepare(all[i].prompt), all[i].options)); }
            handles.push_back(engine.submit(engine.prepare(chat(kTexts[1])), unconstrained(1)));
            for (std::size_t i = 3; i < all.size(); ++i) {
                const auto result = handles[i - 3].wait();
                report(label + ", C = 4: " + all[i].name, result);
                require_same(label + ", C = 4: " + all[i].name, reference[i], result.generated_token_ids);
            }
            require_same(label + ", C = 4: unconstrained 1", plain[1], handles.back().wait().generated_token_ids);
            const infernix::models::inkling2::testing::Dealing after = infernix::models::inkling2::testing::dealing();
            std::cout << label << ": " << after.unequal_rounds - dealt.unequal_rounds << " rounds of unequal widths, "
                      << after.empty_rows - dealt.empty_rows << " rows verified none of their drafts, "
                      << after.full_rows - dealt.full_rows << " all of them\n";
            if (skewed) {
                require(after.empty_rows > dealt.empty_rows && after.full_rows > dealt.full_rows,
                        label + ": no row verified none or all of its drafts");
            }
            infernix::models::inkling2::testing::set_skewed_dealing(false);
        }
        {
            // --lm-head-draft shortlists the drafter's proposals only: the masks read the target's whole head.
            infernix::EngineOptions options             = engine_options(artifact, true, 1, true, true);
            options.speculative.proposal_head           = infernix::ProposalHead::Optimized;
            infernix::Engine engine(options);
            for (std::size_t i = 0; i < all.size(); ++i) {
                require_same("dspark + --lm-head-draft: " + all[i].name, reference[i],
                             engine.generate(engine.prepare(all[i].prompt), all[i].options).generated_token_ids);
            }
        }
        // Token constraints (/v1/decide's generated primitives) on a prompt that opens the answer: the eager Engine's
        // tokens and draws are the reference; graphs, C = 4 beside other requests and a DSpark Engine (the
        // constrained row decodes plainly) give them.
        const std::vector<std::string> questions = {"How many legs does a spider have? Answer with a number.",
                                                    "Pick a number between 0 and 9."};
        std::vector<infernix::GenerationResult> token_reference;
        {
            infernix::Engine engine(engine_options(artifact, false, 1));
            for (std::size_t i = 0; i < questions.size(); ++i) {
                const TokenCase c = token_case(engine, questions[i], 40 + i);
                token_reference.push_back(engine.generate(engine.prepare(c.prompt), c.options));
                check_token_case("eager token constraint " + std::to_string(i), engine, c, token_reference.back());
            }
        }
        const auto same_draws = [&](const std::string& label, const infernix::GenerationResult& a,
                                    const infernix::GenerationResult& b) {
            require_same(label, a.generated_token_ids, b.generated_token_ids);
            for (std::size_t i = 0; i < a.constrained_draws.size(); ++i) {
                require(a.constrained_draws[i].probabilities == b.constrained_draws[i].probabilities &&
                            a.constrained_draws[i].mass == b.constrained_draws[i].mass,
                        label + ": the draws differ at step " + std::to_string(i));
            }
        };
        for (const bool dspark : {false, true}) {
            infernix::Engine engine(engine_options(artifact, true, 4, dspark));
            const std::string label = dspark ? "token constraint, dspark engine" : "token constraint, graphs";
            for (std::size_t i = 0; i < questions.size(); ++i) {
                const TokenCase c = token_case(engine, questions[i], 40 + i);
                same_draws(label + " " + std::to_string(i), token_reference[i],
                           engine.generate(engine.prepare(c.prompt), c.options));
            }
            if (!dspark) {
                std::vector<infernix::GenerationHandle> handles;
                handles.push_back(engine.submit(engine.prepare(chat(kTexts[0])), unconstrained(0)));
                for (std::size_t i = 0; i < questions.size(); ++i) {
                    const TokenCase c = token_case(engine, questions[i], 40 + i);
                    handles.push_back(engine.submit(engine.prepare(c.prompt), c.options));
                }
                handles.push_back(engine.submit(engine.prepare(all[0].prompt), all[0].options));
                require_same("token constraint C = 4: unconstrained 0", plain[0], handles[0].wait().generated_token_ids);
                for (std::size_t i = 0; i < questions.size(); ++i) {
                    same_draws("token constraint C = 4: " + std::to_string(i), token_reference[i], handles[i + 1].wait());
                }
                require_same("token constraint C = 4: " + all[0].name, reference[0], handles.back().wait().generated_token_ids);
            }
        }
        // Full-vocabulary sampling (top_k 0, candidates JM-40): the eager Engine's tokens are the reference; graphs at
        // C = 4 beside top-k 20 requests, and DSpark (auto, graphs; each verify column draws as plain decoding), give
        // them, and the top-k 20 requests keep theirs.
        {
            const auto whole = [](std::size_t i) {
                infernix::RequestOptions options        = unconstrained(i);
                options.execution.sampling.top_k        = 0;
                return options;
            };
            std::vector<Tokens> full;
            {
                infernix::Engine engine(engine_options(artifact, false, 1));
                for (std::size_t i = 0; i < kTexts.size(); ++i) {
                    full.push_back(engine.generate(engine.prepare(chat(kTexts[i])), whole(i)).generated_token_ids);
                }
            }
            std::size_t same = 0;
            while (same < full[0].size() && same < plain[0].size() && full[0][same] == plain[0][same]) { ++same; }
            std::cout << "full vocabulary against top-k 20 (same seed): the first " << same << " of " << full[0].size()
                      << " tokens agree\n";
            {
                infernix::Engine engine(engine_options(artifact, true, 4));
                std::vector<infernix::GenerationHandle> handles;
                for (std::size_t i = 0; i < kTexts.size(); ++i) {
                    handles.push_back(engine.submit(engine.prepare(chat(kTexts[i])), whole(i)));
                    handles.push_back(engine.submit(engine.prepare(chat(kTexts[i])), unconstrained(i)));
                }
                for (std::size_t i = 0; i < kTexts.size(); ++i) {
                    require_same("full vocabulary, graphs C = 4: " + std::to_string(i), full[i],
                                 handles[2 * i].wait().generated_token_ids);
                    require_same("top-k 20 beside full vocabulary, C = 4: " + std::to_string(i), plain[i],
                                 handles[2 * i + 1].wait().generated_token_ids);
                }
            }
            {
                infernix::Engine engine(engine_options(artifact, true, 1, true, true));
                for (std::size_t i = 0; i < kTexts.size(); ++i) {
                    const auto result = engine.generate(engine.prepare(chat(kTexts[i])), whole(i));
                    report("full vocabulary, dspark " + std::to_string(i), result);
                    require_same("full vocabulary, dspark: " + std::to_string(i), full[i], result.generated_token_ids);
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
