#include "models/inkling2/frontend/output_session.h"

#include "models/output_text.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace infernix::models::inkling2 {
namespace {

// Where the turn is: between blocks (headers, stray text and the end-of-turn token publish nothing), or inside a
// thinking, content or tool-call block until its <|end_message|>.
enum class Phase : std::uint8_t { Between, Reasoning, Content, Call };

std::size_t channel_index(OutputChannel channel) noexcept { return channel == OutputChannel::Reasoning ? 0 : 1; }

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

// The earliest match wins: by token, then by byte, then by declaration (as Qwen's sessions order them).
bool precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut, std::uint32_t order, const StopMatch& current) {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) { return committed_tokens < current.committed_tokens; }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return order < current.declaration_order;
}

// Everything one round's preview copies: the decoder's position, the thinking budget's accounting and the calls the
// turn has completed.
struct State {
    Phase phase = Phase::Between;
    std::string utf8_pending;                // an incomplete code point of the open block
    std::array<std::string, 2> stop_pending; // per channel: bytes that may start a stop string
    std::uint64_t decoded_bytes = 0;         // bytes fed to the channels (stop-match order)
    std::string call_body;                   // the open call's bytes
    bool terminal                  = false;
    std::uint32_t reasoning_tokens = 0;
    std::optional<std::uint32_t> matched_stop_order;
    std::uint32_t model_thinking_tokens = 0;
    std::uint32_t injected_tokens       = 0;
    bool control_pending                = false;
    bool applied                        = false; // the control closed the thinking
    // A thinking block closed on its own with the budget used up: the budget is spent, as an applied control spends
    // it (later thinking is not capped, and answer rounds are not held to one token).
    bool spent = false;
    std::vector<GeneratedToolCall> calls;
    ToolCallParseDiagnostics diagnostics;
};

void feed_channel(State& state, OutputChannel channel, std::string_view text, const StopPolicy& policy,
                  PublishedOutput& emitted, std::uint32_t committed_tokens, StopMatch* best) {
    if (text.empty()) { return; }
    std::string& pending          = state.stop_pending[channel_index(channel)];
    std::string combined          = pending;
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;
    if (best != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel || stop.text.empty()) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const auto order = static_cast<std::uint32_t>(declaration);
            if (!precedes(committed_tokens, combined_start + found, order, *best)) { continue; }
            PublishedOutput candidate = emitted;
            candidate.append(channel, std::string_view(combined).substr(0, found));
            if (stop.include_in_output) { candidate.append(channel, stop.text); }
            *best = StopMatch{.found             = true,
                              .committed_tokens  = committed_tokens,
                              .byte_cut          = combined_start + found,
                              .declaration_order = order,
                              .output            = std::move(candidate)};
        }
    }
    const std::size_t hold = stop_hold_size(combined, channel, policy);
    emitted.append(channel, std::string_view(combined).substr(0, combined.size() - hold));
    pending = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

// Publishes what a channel held for a stop string's possible start: a block's close ends the text a stop string can
// span.
void flush_channel(State& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    emitted.append(channel, pending);
    pending.clear();
}

} // namespace

void PublishedOutput::append(OutputChannel channel, std::string_view text) {
    if (text.empty()) { return; }
    if (!values_.empty() && values_.back().channel == channel) {
        values_.back().text.append(text);
    } else {
        values_.push_back(OutputDelta{.channel = channel, .text = std::string(text)});
    }
}

class OutputSession::Impl {
public:
    explicit Impl(Settings settings) : s(std::move(settings)) {
        if (s.tokenizer == nullptr) { throw std::invalid_argument("inkling2 output session: no tokenizer"); }
        if (s.thinking.budget && *s.thinking.budget == 0) {
            throw std::invalid_argument("thinking budget must be positive");
        }
        if (s.thinking.loop != ReasoningLoopAction::Off) {
            throw std::invalid_argument("the reasoning-loop guard is not available for Inkling 2 yet");
        }
        if (s.thinking_control == nullptr || s.thinking_control->empty() ||
            s.thinking_control->back() != s.markers.end_message) {
            throw std::invalid_argument("inkling2 output session: the thinking control must end with <|end_message|>");
        }
        raw            = s.output.raw;
        preserve       = s.output.raw || s.output.preserve_special_tokens;
        parse_calls    = !raw && s.tools != nullptr;
    }

    [[nodiscard]] bool stop_token(TokenId token) const {
        return std::find(s.policy.token_ids.begin(), s.policy.token_ids.end(), token) != s.policy.token_ids.end();
    }

    [[nodiscard]] bool content_opener(TokenId token) const {
        return std::find(s.markers.content.begin(), s.markers.content.end(), token) != s.markers.content.end() ||
               (token == s.markers.tool_call && !parse_calls);
    }

    // Decodes the open block's bytes into its channel.
    void feed_text(State& state, OutputChannel channel, std::string_view bytes, PublishedOutput& emitted,
                   std::uint32_t committed, StopMatch* best) const {
        state.utf8_pending.append(bytes);
        const std::string text = consume_generated_utf8(state.utf8_pending);
        feed_channel(state, channel, text, s.policy, emitted, committed, best);
    }

    void close_text(State& state, OutputChannel channel, PublishedOutput& emitted, std::uint32_t committed,
                    StopMatch* best) const {
        if (!state.utf8_pending.empty()) { // the close cuts a code point
            state.utf8_pending.clear();
            feed_channel(state, channel, kUtf8Replacement, s.policy, emitted, committed, best);
        }
        flush_channel(state, channel, emitted);
    }

    // A call that is not one becomes the turn's text, unmatched (a call's bytes are the tool's data).
    void fall_back(State& state, std::string body, ToolCallParseFallbackReason reason, PublishedOutput& emitted) const {
        if (state.diagnostics.fallback_reason == ToolCallParseFallbackReason::None) {
            state.diagnostics.fallback_reason = reason;
        }
        static const StopPolicy kNoStops{};
        std::string text = consume_generated_utf8(body);
        if (!body.empty()) { text.append(kUtf8Replacement); }
        flush_channel(state, OutputChannel::Content, emitted);
        feed_channel(state, OutputChannel::Content, text, kNoStops, emitted, 0, nullptr);
    }

    // The response_template's tool_calls field: {"name": NAME, "args": {...}} becomes the call NAME(args).
    void close_call(State& state, PublishedOutput& emitted) const {
        std::string body = std::move(state.call_body);
        state.call_body.clear();
        nlohmann::ordered_json call;
        try {
            call = nlohmann::ordered_json::parse(body);
        } catch (const nlohmann::json::exception&) {
            fall_back(state, std::move(body), ToolCallParseFallbackReason::MalformedStructure, emitted);
            return;
        }
        const auto name = call.is_object() ? call.find("name") : call.end();
        const auto args = call.is_object() ? call.find("args") : call.end();
        if (!call.is_object() || args == call.end() || !args->is_object() || name == call.end()) {
            fall_back(state, std::move(body), ToolCallParseFallbackReason::MalformedStructure, emitted);
            return;
        }
        if (!name->is_string() || name->get_ref<const std::string&>().empty()) {
            fall_back(state, std::move(body), ToolCallParseFallbackReason::InvalidToolName, emitted);
            return;
        }
        const std::string& tool = name->get_ref<const std::string&>();
        if (!s.tools->open && !s.tools->declared.contains(tool)) {
            fall_back(state, std::move(body), ToolCallParseFallbackReason::UndeclaredTool, emitted);
            return;
        }
        state.calls.push_back(GeneratedToolCall{.name = tool, .arguments_json = args->dump()});
        ++state.diagnostics.structured_call_count;
    }

    // One generated (or injected) token.
    void feed_token(State& state, TokenId token, PublishedOutput& emitted, std::uint32_t committed,
                    StopMatch* best) const {
        const text::DecodedTokenView decoded = s.tokenizer->decoded_token(token);
        if (raw) {
            feed_text(state, OutputChannel::Content, decoded.bytes, emitted, committed, best);
            return;
        }
        switch (state.phase) {
        case Phase::Between:
            if (token == s.markers.thinking) {
                state.phase = Phase::Reasoning;
            } else if (content_opener(token)) {
                state.phase = Phase::Content;
            } else if (token == s.markers.tool_call) {
                state.phase = Phase::Call;
                state.diagnostics.marker_seen = true;
            }
            return; // headers, stray text and other tokens between blocks publish nothing
        case Phase::Reasoning:
        case Phase::Content: {
            const OutputChannel channel =
                state.phase == Phase::Reasoning ? OutputChannel::Reasoning : OutputChannel::Content;
            if (token == s.markers.end_message) {
                close_text(state, channel, emitted, committed, best);
                state.phase = Phase::Between;
                return;
            }
            if (decoded.special && !preserve) { return; }
            feed_text(state, channel, decoded.bytes, emitted, committed, best);
            return;
        }
        case Phase::Call:
            if (token == s.markers.end_message) {
                close_call(state, emitted);
                state.phase = Phase::Between;
                return;
            }
            if (decoded.special && !preserve) { return; }
            state.call_body.append(decoded.bytes);
            return;
        }
    }

    // The turn ends with a block open (a limit, a cancellation, a stop token): its text publishes; an open call is
    // text.
    void terminalize(State& state, PublishedOutput& emitted) const {
        if (state.phase == Phase::Reasoning || raw) {
            close_text(state, raw ? OutputChannel::Content : OutputChannel::Reasoning, emitted, 0, nullptr);
        } else if (state.phase == Phase::Content) {
            close_text(state, OutputChannel::Content, emitted, 0, nullptr);
        } else if (state.phase == Phase::Call) {
            std::string body = std::move(state.call_body);
            state.call_body.clear();
            fall_back(state, std::move(body), ToolCallParseFallbackReason::MalformedStructure, emitted);
        }
        flush_channel(state, OutputChannel::Reasoning, emitted);
        flush_channel(state, OutputChannel::Content, emitted);
        state.utf8_pending.clear();
        state.phase    = Phase::Between;
        state.terminal = true;
    }

    void begin_preview() {
        preview       = committed;
        preview_ready = false;
        preview_output.clear();
    }

    Settings s;
    bool raw         = false;
    bool preserve    = false;
    bool parse_calls = false;
    State committed;
    State preview;
    PublishedOutput preview_output;
    bool preview_ready                 = false;
    FinishReason preview_finish_reason = FinishReason::None;
    std::vector<GeneratedToolCall> tool_calls; // the terminal turn's calls
};

OutputSession::OutputSession() noexcept = default;
OutputSession::OutputSession(Settings settings) : impl_(std::make_unique<Impl>(std::move(settings))) {}
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
                                                     std::uint32_t total_budget_remaining, FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    Impl& m = *impl_;
    if (m.committed.terminal) { throw std::logic_error("output session is already terminal"); }
    if (m.preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (m.committed.control_pending) {
        throw std::logic_error("model output cannot advance while thinking control is pending");
    }
    if (tokens.empty()) { throw std::invalid_argument("cannot preview an empty generated-token round"); }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit && limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }
    m.begin_preview();
    State& state = m.preview;
    const std::optional<std::uint32_t> budget = m.s.thinking.budget;

    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              runtime::ContinuationAction continuation = runtime::ContinuationAction::Decode) {
        m.preview_finish_reason = reason;
        m.preview_ready         = true;
        return runtime::OutputDecision{.accepted_tokens = count, .finish_reason = reason, .continuation = continuation};
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto count    = static_cast<std::uint32_t>(index + 1);
        const TokenId token = tokens[index];
        const bool thinking = !m.raw && state.phase == Phase::Reasoning;
        if (thinking) {
            ++state.reasoning_tokens;
            ++state.model_thinking_tokens;
            if (budget && !state.applied && !state.spent && state.model_thinking_tokens > *budget) {
                throw std::logic_error("model output exceeded the licensed thinking budget");
            }
        }
        if (m.stop_token(token)) {
            if (m.s.policy.publish_stop_token) { m.feed_token(state, token, m.preview_output, count, nullptr); }
            m.terminalize(state, m.preview_output);
            return complete(count, FinishReason::StopToken);
        }
        StopMatch match;
        m.feed_token(state, token, m.preview_output, count, &match);
        if (thinking && budget && state.phase != Phase::Reasoning && state.model_thinking_tokens >= *budget) {
            state.spent = true;
        }
        if (match.found) {
            state.matched_stop_order = match.declaration_order;
            state.utf8_pending.clear();
            state.stop_pending = {};
            state.call_body.clear();
            state.phase    = Phase::Between;
            state.terminal = true;
            m.preview_output = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }
    }
    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == total_budget_remaining) {
        m.terminalize(state, m.preview_output);
        return complete(count, limit_reason);
    }
    if (budget && !state.applied && !state.spent && state.phase == Phase::Reasoning &&
        state.model_thinking_tokens == *budget) {
        // The control and one model token after it must fit the output budget; otherwise the turn ends at its limit.
        const std::uint64_t remaining = total_budget_remaining - count;
        if (remaining < m.s.thinking_control->size() + 1U) {
            m.terminalize(state, m.preview_output);
            return complete(count, FinishReason::OutputLimit);
        }
        state.control_pending = true;
        return complete(count, FinishReason::None, runtime::ContinuationAction::ApplyTargetControl);
    }
    return complete(count, FinishReason::None);
}

std::uint32_t OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    if (impl_ == nullptr) { return total_budget_remaining; }
    const State& state = impl_->committed;
    if (state.control_pending) { return 0; }
    const std::optional<std::uint32_t> budget = impl_->s.thinking.budget;
    if (!budget || state.applied || state.spent || impl_->raw) { return total_budget_remaining; }
    // A round may open a thinking block (its opener is not a thinking token) and then fill what is left of the
    // budget: the budget is reached at a round's last token at the latest.
    const std::uint32_t left = *budget - std::min(*budget, state.model_thinking_tokens);
    const std::uint32_t cap  = state.phase == Phase::Reasoning ? left : left + 1U;
    return std::min(total_budget_remaining, cap);
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    if (impl_ == nullptr || !impl_->committed.control_pending) { return {}; }
    return *impl_->s.thinking_control;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId> tokens,
                                                       std::uint32_t total_budget_remaining) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    Impl& m = *impl_;
    if (m.committed.terminal) { throw std::logic_error("output session is already terminal"); }
    if (m.preview_ready) { throw std::logic_error("output session already has a preview"); }
    const std::span<const TokenId> expected = pending_control_tokens();
    if (expected.empty() || !std::equal(tokens.begin(), tokens.end(), expected.begin(), expected.end())) {
        throw std::invalid_argument("thinking control preview requires the exact pending span");
    }
    if (tokens.size() > total_budget_remaining) {
        throw std::invalid_argument("thinking control span exceeds the remaining output budget");
    }
    m.begin_preview();
    State& state = m.preview;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (state.phase == Phase::Reasoning) { ++state.reasoning_tokens; }
        m.feed_token(state, tokens[index], m.preview_output, static_cast<std::uint32_t>(index + 1), nullptr);
    }
    if (state.phase != Phase::Between) {
        throw std::logic_error("the thinking control did not close the thinking block");
    }
    state.control_pending   = false;
    state.applied           = true;
    state.injected_tokens   = static_cast<std::uint32_t>(tokens.size());
    m.preview_finish_reason = FinishReason::None;
    m.preview_ready         = true;
    return runtime::OutputDecision{.accepted_tokens = static_cast<std::uint32_t>(tokens.size())};
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    const std::optional<std::uint32_t> budget = impl_->s.thinking.budget;
    if (!budget || impl_->raw || effective_output_tokens <= *budget) { return; }
    // The budget is reached after its opener and *budget thinking tokens; the control and one model token follow.
    const std::uint64_t required = static_cast<std::uint64_t>(*budget) + 1U + impl_->s.thinking_control->size() + 1U;
    if (effective_output_tokens < required) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the complete control suffix and one "
            "post-close model token");
    }
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    Impl& m = *impl_;
    if (m.committed.terminal) { throw std::logic_error("output session is already terminal"); }
    if (m.preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString || reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    m.begin_preview();
    m.preview.control_pending = false;
    m.terminalize(m.preview, m.preview_output);
    m.preview_finish_reason = reason;
    m.preview_ready         = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

PublishedOutput OutputSession::commit_preview() {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    Impl& m = *impl_;
    std::swap(m.committed, m.preview);
    m.preview_ready       = false;
    PublishedOutput output = std::move(m.preview_output);
    m.preview_output.clear();
    if (m.committed.terminal) { m.tool_calls = std::move(m.committed.calls); }
    return output;
}

void OutputSession::discard_preview() {
    if (impl_ == nullptr) { return; }
    impl_->preview_ready = false;
    impl_->preview_output.clear();
}

runtime::ReasoningBan OutputSession::reasoning_ban() const noexcept {
    if (impl_ == nullptr || impl_->raw || impl_->s.thinking.end_turn_in_thinking ||
        impl_->committed.phase != Phase::Reasoning || impl_->committed.terminal) {
        return {};
    }
    return {.close    = impl_->s.markers.end_message,
            .ends     = {impl_->s.markers.end_of_turn, -1},
            .withheld = impl_->parse_calls ? impl_->s.markers.tool_call : TokenId{-1}};
}

std::uint64_t OutputSession::grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>) {
    throw std::logic_error("mask requested for unconstrained output");
}

std::vector<GeneratedToolCall> OutputSession::take_tool_calls() noexcept {
    return impl_ != nullptr ? std::move(impl_->tool_calls) : std::vector<GeneratedToolCall>{};
}

bool OutputSession::has_tool_calls() const noexcept { return impl_ != nullptr && !impl_->tool_calls.empty(); }

ToolCallParseDiagnostics OutputSession::tool_call_parse_diagnostics() const noexcept {
    return impl_ != nullptr ? impl_->committed.diagnostics : ToolCallParseDiagnostics{};
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->committed.reasoning_tokens : 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return ThinkingBudgetStats{.configured_budget     = impl_->s.thinking.budget,
                               .model_thinking_tokens = impl_->committed.model_thinking_tokens,
                               .injected_tokens       = impl_->committed.injected_tokens,
                               .applied               = impl_->committed.applied};
}

std::optional<std::string> OutputSession::matched_stop_string() const {
    if (impl_ == nullptr || !impl_->committed.matched_stop_order) { return std::nullopt; }
    const std::size_t index = *impl_->committed.matched_stop_order;
    if (index >= impl_->s.policy.strings.size()) {
        throw std::logic_error("matched stop declaration is outside the stop policy");
    }
    return impl_->s.policy.strings[index].text;
}

} // namespace infernix::models::inkling2
