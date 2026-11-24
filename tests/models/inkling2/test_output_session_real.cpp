// Inkling 2's output session (src/models/inkling2/frontend/output_session.h) on the real tokenizer.
//
// 1. Against the checkpoint's response_template as transformers' ResponseParser applies it
//    (tests/fixtures/inkling2/output_session_golden.json, generate_output_golden.py): every golden turn, fed one token a
//    round, in rounds of 2, 3, 5, 8 and 16 tokens (each round previewed, discarded, previewed again and committed) and
//    whole, publishes the parser's reasoning and content and returns its tool calls (arguments compared as JSON
//    values). The goldens cover headers with and without a name, joined blocks, the three content openers, text
//    between blocks, non-ASCII text, calls (compact and spaced JSON, several), spoofed marker spellings in ordinary
//    tokens, an added token inside text, turns a limit cuts, and calls that are text (malformed, undeclared, no tools).
// 2. The session's own rules: stop strings per channel (split across tokens and rounds), a thinking budget's control
//    and the round cap before it, the reasoning ban while thinking is open, raw output, and a limit inside a call.
//
//   INFERNIX_TEST_INKLING_TOKENIZER_DIR  a directory holding tokenizer.json, tokenizer_config.json and
//                                        generation_config.json (generate_tokenizer_golden.py writes it)

#include "models/inkling2/frontend/output_session.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;
using infernix::FinishReason;
using infernix::OutputChannel;
using infernix::TokenId;
namespace ink = infernix::models::inkling2;

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

struct Turn {
    std::string reasoning, content;
    std::vector<infernix::GeneratedToolCall> calls;
    FinishReason finish = FinishReason::None;
    std::vector<std::uint32_t> rounds; // accepted tokens per committed round
};

class Harness {
public:
    explicit Harness(std::shared_ptr<const infernix::text::Tokenizer> tokenizer) : tokenizer_(std::move(tokenizer)) {
        const auto id = [&](const char* spelling) {
            const auto* token = tokenizer_->find_added_token(spelling);
            if (token == nullptr) { throw std::runtime_error(std::string("the tokenizer lacks ") + spelling); }
            return static_cast<TokenId>(token->id);
        };
        markers.message_model = id("<|message_model|>");
        markers.thinking      = id("<|content_thinking|>");
        markers.content       = {id("<|content_text|>"), id("<|content_tool_error|>"), id("<|content_invoke_tool_text|>")};
        markers.tool_call     = id("<|content_invoke_tool_json|>");
        markers.end_message   = id("<|end_message|>");
        markers.end_of_turn   = id("<|content_model_end_sampling|>");
    }

    [[nodiscard]] ink::OutputSession::Settings settings(const Json& tools) const {
        ink::OutputSession::Settings s;
        s.tokenizer               = tokenizer_;
        s.markers                 = markers;
        s.policy.token_ids        = {markers.end_of_turn};
        s.thinking_control        = std::make_shared<const std::vector<TokenId>>(std::vector{markers.end_message});
        if (tools.is_array()) {
            auto names = std::make_shared<ink::ToolNames>();
            for (const auto& name : tools) { names->declared.insert(name.get<std::string>()); }
            s.tools = std::move(names);
        }
        return s;
    }

    [[nodiscard]] std::vector<TokenId> encode(const std::string& text) const { return tokenizer_->encode(text); }
    [[nodiscard]] std::string decode(std::span<const TokenId> ids) const { return tokenizer_->decode(ids); }

    // Feeds `ids` in rounds of `round` tokens (0: whole), each capped by the session's model budget, previewing and
    // discarding every round once before committing it when `discard`.
    static Turn run(ink::OutputSession& session, const std::vector<TokenId>& ids, std::size_t round, bool discard,
                    std::uint32_t total = 1U << 20) {
        Turn turn;
        const auto take = [&](const ink::PublishedOutput& output) {
            for (const auto& delta : output) {
                (delta.channel == OutputChannel::Reasoning ? turn.reasoning : turn.content) += delta.text;
            }
        };
        std::size_t at = 0;
        while (at < ids.size()) {
            const auto control = session.pending_control_tokens();
            if (!control.empty()) {
                const std::vector<TokenId> span(control.begin(), control.end());
                const auto decision = session.preview_control(span, total);
                take(session.commit_preview());
                total -= decision.accepted_tokens;
                continue;
            }
            std::size_t width = round == 0 ? ids.size() - at : std::min(round, ids.size() - at);
            width = std::min<std::size_t>(width, session.model_token_budget_remaining(total));
            if (width == 0) { throw std::logic_error("the session allows no model token"); }
            const std::span<const TokenId> tokens(ids.data() + at, width);
            if (discard) {
                (void)session.preview_model(tokens, total, FinishReason::OutputLimit);
                session.discard_preview();
            }
            const auto decision = session.preview_model(tokens, total, FinishReason::OutputLimit);
            take(session.commit_preview());
            turn.rounds.push_back(decision.accepted_tokens);
            at += decision.accepted_tokens;
            total -= decision.accepted_tokens;
            if (decision.finished()) {
                turn.finish = decision.finish_reason;
                turn.calls  = session.take_tool_calls();
                return turn;
            }
        }
        (void)session.preview_terminal(FinishReason::OutputLimit);
        take(session.commit_preview());
        turn.finish = FinishReason::OutputLimit;
        turn.calls  = session.take_tool_calls();
        return turn;
    }

    ink::OutputMarkers markers;

private:
    std::shared_ptr<const infernix::text::Tokenizer> tokenizer_;
};

bool same_calls(const std::vector<infernix::GeneratedToolCall>& calls, const Json& expected) {
    if (calls.size() != expected.size()) { return false; }
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (calls[i].name != expected[i].at("name").get<std::string>()) { return false; }
        if (Json::parse(calls[i].arguments_json) != expected[i].at("arguments")) { return false; }
    }
    return true;
}

void golden_cases(const Harness& h, const Json& golden) {
    std::size_t runs = 0;
    for (const Json& c : golden.at("cases")) {
        const std::string name        = c.at("name").get<std::string>();
        const auto ids                = c.at("ids").get<std::vector<TokenId>>();
        const std::string reasoning   = c.at("reasoning").get<std::string>();
        const std::string content     = c.at("content").get<std::string>();
        const bool ends_with_stop     = !ids.empty() && ids.back() == h.markers.end_of_turn;
        for (const std::size_t round : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}, std::size_t{8},
                                        std::size_t{16}, std::size_t{0}}) {
            ink::OutputSession session(h.settings(c.at("tools")));
            const Turn turn = Harness::run(session, ids, round, round != 0 && round != 1);
            ++runs;
            const std::string where = name + " (rounds of " + std::to_string(round) + ")";
            check(turn.reasoning == reasoning, where + ": reasoning " + Json(turn.reasoning).dump() + ", expected " +
                                                   Json(reasoning).dump());
            check(turn.content == content,
                  where + ": content " + Json(turn.content).dump() + ", expected " + Json(content).dump());
            check(same_calls(turn.calls, c.at("tool_calls")), where + ": tool calls differ");
            check(turn.finish == (ends_with_stop ? FinishReason::StopToken : FinishReason::OutputLimit),
                  where + ": finish reason");
        }
    }
    std::cout << golden.at("cases").size() << " golden turns, " << runs << " splits\n";
}

std::vector<TokenId> cat(std::initializer_list<std::vector<TokenId>> parts) {
    std::vector<TokenId> out;
    for (const auto& part : parts) { out.insert(out.end(), part.begin(), part.end()); }
    return out;
}

void rule_cases(const Harness& h) {
    const auto& k = h.markers;
    const std::vector<TokenId> M{k.message_model}, CT{k.thinking}, CX{k.content[0]}, CJ{k.tool_call},
        E{k.end_message}, EOS{k.end_of_turn};

    { // A content stop string, split over tokens, in rounds of one: the content up to it, the matched string.
        auto s = h.settings(nullptr);
        s.policy.strings = {{.text = "is 4", .channel = OutputChannel::Content}};
        ink::OutputSession session(std::move(s));
        const auto ids  = cat({M, CT, h.encode("this 4 is 4 in thinking"), E, M, CX, h.encode("The answer is 4."), E, EOS});
        const Turn turn = Harness::run(session, ids, 1, false);
        check(turn.finish == FinishReason::StopString, "content stop: finishes on the string");
        check(turn.content == "The answer ", "content stop: content " + Json(turn.content).dump());
        check(turn.reasoning == "this 4 is 4 in thinking", "content stop: thinking is not matched");
        check(session.matched_stop_string() == std::optional<std::string>("is 4"), "content stop: matched string");
    }
    { // A reasoning stop string with include_in_output, in rounds of 3.
        auto s = h.settings(nullptr);
        s.policy.strings = {{.text = "STOP", .channel = OutputChannel::Reasoning, .include_in_output = true}};
        ink::OutputSession session(std::move(s));
        const auto ids  = cat({M, CT, h.encode("go on STOP not this"), E, EOS});
        const Turn turn = Harness::run(session, ids, 3, true);
        check(turn.finish == FinishReason::StopString && turn.reasoning == "go on STOP",
              "reasoning stop: " + Json(turn.reasoning).dump());
    }
    { // A thinking budget of 3: the control closes the thinking after three thinking tokens; rounds never pass it.
        auto s = h.settings(nullptr);
        s.thinking.budget = 3;
        ink::OutputSession session(std::move(s));
        const auto thinking = h.encode(" one two three four five six");
        const auto answer   = cat({M, CX, h.encode("done"), E, EOS});
        // The model writes the thinking past the budget; the test feeds what the session allows, then the answer.
        Turn turn;
        std::uint32_t total = 1000;
        check(session.model_token_budget_remaining(total) == 4, "budget: a round from between blocks holds the opener");
        auto d = session.preview_model(cat({M, CT}), total, FinishReason::OutputLimit);
        (void)session.commit_preview();
        total -= d.accepted_tokens;
        check(session.reasoning_ban().active() && session.reasoning_ban().close == k.end_message &&
                  session.reasoning_ban().ends[0] == k.end_of_turn,
              "budget: the ban is active while thinking is open");
        check(session.model_token_budget_remaining(total) == 3, "budget: three thinking tokens left");
        d = session.preview_model(std::span(thinking.data(), 3), total, FinishReason::OutputLimit);
        check(d.continuation == infernix::runtime::ContinuationAction::ApplyTargetControl,
              "budget: the third thinking token asks for the control");
        for (const auto& delta : session.commit_preview()) { turn.reasoning += delta.text; }
        total -= d.accepted_tokens;
        check(session.pending_control_tokens().size() == 1 && session.pending_control_tokens()[0] == k.end_message,
              "budget: the control is <|end_message|>");
        check(session.model_token_budget_remaining(total) == 0, "budget: no model token while the control is pending");
        const std::vector<TokenId> control{k.end_message};
        (void)session.preview_control(control, total);
        (void)session.commit_preview();
        // The budget holds for the turn: after the control every column refuses the thinking opener, and the ban never
        // lifts (its close is the opener itself).
        const infernix::runtime::ReasoningBan after = session.reasoning_ban();
        check(after.active() && after.close == k.thinking && after.ends[0] == k.thinking && after.ends[1] == -1 &&
                  after.withheld == -1,
              "budget: after the control the thinking opener is banned for the rest of the turn");
        const Turn rest = Harness::run(session, answer, 16, false, total - 1);
        check(turn.reasoning == h.decode(std::span(thinking.data(), 3)) && rest.content == "done",
              "budget: reasoning " + Json(turn.reasoning).dump() + ", content " + Json(rest.content).dump());
        const auto stats = session.thinking_stats();
        check(stats.applied && stats.model_thinking_tokens == 3 && stats.injected_tokens == 1, "budget: stats");
    }
    { // A thinking block that closes on its own at the budget spends it: later rounds are not capped, and every later
      // column refuses the thinking opener (a turn's budget, not a block's).
        auto s = h.settings(nullptr);
        s.thinking.budget = 3;
        ink::OutputSession session(std::move(s));
        const auto two = h.encode(" one two three");
        const std::vector<TokenId> first = cat({M, CT, {two[0], two[1]}, E});
        std::uint32_t total = 1000;
        check(session.model_token_budget_remaining(total) == 4, "spent: the first round holds the opener and three");
        const auto d = session.preview_model(std::span(first.data(), 4), total, FinishReason::OutputLimit);
        (void)session.commit_preview();
        total -= d.accepted_tokens;
        check(d.continuation == infernix::runtime::ContinuationAction::Decode,
              "spent: no control while the block is still open below the budget");
        const auto d2 = session.preview_model(std::span(first.data() + 4, 1), total, FinishReason::OutputLimit);
        (void)session.commit_preview();
        total -= d2.accepted_tokens;
        check(d2.continuation == infernix::runtime::ContinuationAction::Decode && session.pending_control_tokens().empty(),
              "spent: the close at the budget needs no control");
        check(session.model_token_budget_remaining(total) == total, "spent: rounds are no longer capped");
        const infernix::runtime::ReasoningBan between = session.reasoning_ban();
        check(between.active() && between.close == k.thinking && between.ends[0] == k.thinking,
              "spent: between blocks the thinking opener is banned");
        const auto d3 = session.preview_model(cat({M, CX}), total, FinishReason::OutputLimit);
        (void)session.commit_preview();
        total -= d3.accepted_tokens;
        check(session.reasoning_ban().active() && session.reasoning_ban().ends[0] == k.thinking,
              "spent: inside the answer the thinking opener stays banned");
        const Turn rest = Harness::run(session, cat({h.encode("ok"), E, EOS}), 16, false, total);
        check(rest.finish == FinishReason::StopToken && rest.content == "ok", "spent: the answer follows");
        check(!session.reasoning_ban().active(), "spent: a finished turn bans nothing");
        check(!session.thinking_stats().applied, "spent: no control was applied");
    }
    { // Raw output: every byte, markers included, is content.
        auto s = h.settings(nullptr);
        s.output.raw = true;
        ink::OutputSession session(std::move(s));
        const Turn turn = Harness::run(session, cat({M, CX, h.encode("hi"), E, EOS}), 2, false);
        check(turn.content == "<|message_model|><|content_text|>hi<|end_message|>",
              "raw: content " + Json(turn.content).dump());
    }
    { // A limit inside a call: its bytes are the turn's text.
        auto s = h.settings(Json::array({"f"}));
        ink::OutputSession session(std::move(s));
        const Turn turn = Harness::run(session, cat({M, CJ, h.encode(R"({"name":"f","args":{"a":)")}), 4, true);
        check(turn.finish == FinishReason::OutputLimit && turn.calls.empty() &&
                  turn.content == R"({"name":"f","args":{"a":)",
              "limit in a call: content " + Json(turn.content).dump());
        check(session.tool_call_parse_diagnostics().fallback_reason ==
                  infernix::ToolCallParseFallbackReason::MalformedStructure,
              "limit in a call: diagnostics");
    }
}

} // namespace

int main() {
    const char* directory = std::getenv("INFERNIX_TEST_INKLING_TOKENIZER_DIR");
    if (directory == nullptr || *directory == '\0') {
        std::cout << "skip: INFERNIX_TEST_INKLING_TOKENIZER_DIR is not set\n";
        return 77;
    }
    try {
        const std::filesystem::path root(directory);
        const std::string tokenizer_json = read_file(root / "tokenizer.json");
        const Json golden = Json::parse(read_file(std::filesystem::path(INFERNIX_SOURCE_DIR) / "tests" / "fixtures" /
                                                  "inkling2" / "output_session_golden.json"));
        if (golden.at("tokenizer_json_bytes").get<std::size_t>() != tokenizer_json.size()) {
            std::cout << "skip: the tokenizer.json is not the one the goldens were made with\n";
            return 77;
        }
        auto tokenizer = std::make_shared<const infernix::text::Tokenizer>(infernix::text::TokenizerResources{
            .tokenizer_json         = tokenizer_json,
            .tokenizer_config_json  = read_file(root / "tokenizer_config.json"),
            .generation_config_json = read_file(root / "generation_config.json"),
            .pipeline               = infernix::text::TokenizerPipeline::Inkling});
        const Harness harness(tokenizer);
        golden_cases(harness, golden);
        rule_cases(harness);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " checks failed\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
