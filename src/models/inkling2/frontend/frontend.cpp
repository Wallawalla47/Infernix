#include "models/inkling2/frontend/frontend.h"

#include "models/inkling2/frontend/constraints.h"
#include "runtime/prefix_cache/block_hash.h"
#include "text/grammar.h"
#include "text/jinja.h"
#include "text/tokenizer.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::inkling2 {
namespace {

using Json  = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

std::string_view role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
    case ChatRole::Developer: // the template has no developer role: an instruction renders as a system turn
        return "system";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

Json parse_object(std::string_view json, std::string_view what) {
    if (json.empty()) { return Json::object(); }
    Json value;
    try {
        value = Json::parse(json);
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument(std::string(what) + " is not valid JSON: " + error.what());
    }
    if (!value.is_object()) { throw std::invalid_argument(std::string(what) + " must be a JSON object"); }
    return value;
}

// One message as the template reads it (its OpenAI-shaped fields); its images, in order, appended to `media` (null:
// images are refused, the Engine runs without --vision).
Json message_json(const ChatMessage& message, std::vector<const OwnedMedia*>* media) {
    Json out    = Json::object();
    out["role"] = role_name(message.role);
    bool images = false;
    for (const MessagePart& part : message.parts) {
        if (part.kind != MessagePartKind::Media) { continue; }
        if (media == nullptr) {
            throw RequestError(RequestErrorKind::InvalidMedia,
                               "Inkling 2 takes images with --vision (an artifact converted with the vision component)");
        }
        if (part.media.kind != MediaKind::Image) {
            throw RequestError(RequestErrorKind::InvalidMedia, "Inkling 2 takes images, not video");
        }
        images = true;
    }
    if (message.parts.size() == 1 && !images) {
        out["content"] = message.parts.front().text;
    } else if (message.parts.empty()) {
        out["content"] = "";
    } else {
        Json parts = Json::array();
        for (const MessagePart& part : message.parts) {
            if (part.kind == MessagePartKind::Media) {
                parts.push_back({{"type", "image"}});
                media->push_back(&part.media);
            } else {
                parts.push_back({{"type", "text"}, {"text", part.text}});
            }
        }
        out["content"] = std::move(parts);
    }
    if (!message.reasoning_content.empty()) { out["reasoning_content"] = message.reasoning_content; }
    if (!message.tool_calls.empty()) {
        Json calls = Json::array();
        for (const ToolCall& call : message.tool_calls) {
            Json entry = Json::object();
            if (!call.id.empty()) { entry["id"] = call.id; }
            entry["type"]     = "function";
            entry["function"] = {{"name", call.name},
                                 {"arguments", parse_object(call.arguments_json, "tool call arguments")}};
            calls.push_back(std::move(entry));
        }
        out["tool_calls"] = std::move(calls);
    }
    if (!message.tool_call_id.empty()) { out["tool_call_id"] = message.tool_call_id; }
    return out;
}

std::string tool_name(const Json& tool) {
    const Json& fn = tool.contains("function") ? tool.at("function") : tool;
    if (!fn.is_object() || !fn.contains("name") || !fn.at("name").is_string()) {
        throw std::invalid_argument("a tool declaration needs a string name");
    }
    return fn.at("name").get<std::string>();
}

// A declaration as the constrained route reads it: its name, its parameter schema and whether it is strict (a strict
// tool's calls must satisfy that schema; strict false or absent is the default).
DeclaredTool declared_tool(const Json& tool, std::size_t index) {
    const Json& fn = tool.contains("function") ? tool.at("function") : tool;
    DeclaredTool out{.name = tool_name(tool), .declaration_index = index};
    if (fn.contains("parameters") && !fn.at("parameters").is_null()) { out.parameters = fn.at("parameters").dump(); }
    out.strict = fn.contains("strict") && fn.at("strict").is_boolean() && fn.at("strict").get<bool>();
    return out;
}

constexpr std::uint64_t mix64(std::uint64_t state, std::uint64_t value) noexcept {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6U) + (state >> 2U);
    state *= 0xbf58476d1ce4e5b9ULL;
    return state ^ (state >> 31U);
}

// The prefix key `cumulative` extended by an image (the blocks from its first token's on carry it): its placement,
// its cells and every word of its processed patches, which are all the model reads of it.
std::uint64_t image_prefix_key(std::uint64_t cumulative, const PromptImage& image) {
    std::uint64_t key = mix64(cumulative == 0 ? 0x696e6b6c696e6732ULL : cumulative, image.first_token);
    key               = mix64(key, static_cast<std::uint32_t>(image.image.rows));
    key               = mix64(key, static_cast<std::uint32_t>(image.image.cols));
    const std::vector<std::uint16_t>& words = *image.image.patches;
    std::size_t i                           = 0;
    for (; i + 4 <= words.size(); i += 4) {
        key = mix64(key, std::uint64_t{words[i]} | std::uint64_t{words[i + 1]} << 16U |
                             std::uint64_t{words[i + 2]} << 32U | std::uint64_t{words[i + 3]} << 48U);
    }
    for (; i < words.size(); ++i) { key = mix64(key, words[i]); }
    return mix64(key, words.size());
}

} // namespace

struct Frontend::Impl {
    std::shared_ptr<const text::Tokenizer> tokenizer;
    text::JinjaTemplate tmpl;
    OutputMarkers markers;
    StopPolicy defaults;
    ModelSamplingDefaults sampling;
    std::shared_ptr<const std::vector<TokenId>> thinking_control;
    std::uint32_t max_context = 0;
    std::optional<VisionConfig> vision; // --vision
    std::uint32_t max_cells = 0;
    TokenId image_token     = -1;       // one per image cell (the template renders one an image part)
    TokenId message_system  = -1;
    std::array<TokenId, 4> message_starts{}; // <|message_{user,model,system,tool}|>: each message's first token

    // The prompt's tap hints (prepared_prompt's tap_hints): each message start; the leading system block's last
    // message (the template's thinking effort, which differs between requests that share the system prompt and the
    // tools before it) and the block's end; and the generation opener when the prompt ends inside the last model
    // message (the next turn's history departs there when its client re-renders the reply).
    [[nodiscard]] std::vector<runtime::prefix_cache::TapHint> tap_hints(std::span<const TokenId> ids) const {
        using runtime::prefix_cache::TapHint;
        using runtime::prefix_cache::TapHintKind;
        std::vector<TapHint> out;
        std::vector<std::uint32_t> starts;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (std::find(message_starts.begin(), message_starts.end(), ids[i]) != message_starts.end()) {
                starts.push_back(static_cast<std::uint32_t>(i));
            }
        }
        for (const std::uint32_t s : starts) {
            if (s != 0) { out.push_back(TapHint{s, TapHintKind::MessageBoundary}); }
        }
        std::size_t leading = 0; // the leading system messages
        while (leading < starts.size() && ids[starts[leading]] == message_system) { ++leading; }
        if (leading > 0 && leading < starts.size()) {
            if (starts[leading - 1] != 0) { out.push_back(TapHint{starts[leading - 1], TapHintKind::Structural}); }
            out.push_back(TapHint{starts[leading], TapHintKind::Structural});
        }
        if (!starts.empty() && ids[starts.back()] == markers.message_model && starts.back() != 0 &&
            std::find(ids.begin() + starts.back(), ids.end(), markers.end_message) == ids.end()) {
            out.push_back(TapHint{starts.back(), TapHintKind::GenerationOpener});
        }
        std::sort(out.begin(), out.end(), [](const TapHint& a, const TapHint& b) { return a.position < b.position; });
        return out;
    }

    Impl(const FrontendResources& resources, const FrontendOptions& options)
        : tokenizer(resources.tokenizer),
          tmpl(std::string(resources.chat_template_jinja), "chat_template.jinja"),
          max_context(options.max_context), vision(options.vision), max_cells(options.vision_max_cells),
          grammar_cache_bytes(options.grammar_cache_bytes) {
        if (tokenizer == nullptr) { throw std::invalid_argument("inkling2 frontend: no tokenizer"); }
        if (max_context == 0) { throw std::invalid_argument("frontend max_context must be nonzero"); }
        const auto id = [&](const char* spelling) {
            const auto* token = tokenizer->find_added_token(spelling);
            if (token == nullptr) { throw std::invalid_argument(std::string("inkling2 tokenizer lacks ") + spelling); }
            return static_cast<TokenId>(token->id);
        };
        markers.message_model = id("<|message_model|>");
        message_system        = id("<|message_system|>");
        message_starts        = {id("<|message_user|>"), markers.message_model, message_system, id("<|message_tool|>")};
        markers.thinking      = id("<|content_thinking|>");
        markers.content       = {id("<|content_text|>"), id("<|content_tool_error|>"), id("<|content_invoke_tool_text|>")};
        markers.tool_call     = id("<|content_invoke_tool_json|>");
        markers.end_message   = id("<|end_message|>");
        markers.end_of_turn   = id("<|content_model_end_sampling|>");
        for (const int token : tokenizer->default_stop_token_ids()) { defaults.token_ids.push_back(token); }
        if (std::find(defaults.token_ids.begin(), defaults.token_ids.end(), markers.end_of_turn) ==
            defaults.token_ids.end()) {
            throw std::invalid_argument("inkling2 generation_config.json must stop at <|content_model_end_sampling|>");
        }
        // generation_config.json sets temperature 1 alone; the sampler takes at most 20 candidates.
        const SamplingPreset preset{.temperature = 1.0F, .top_k = 20, .top_p = 1.0F};
        sampling = {.thinking = preset, .non_thinking = preset};
        std::vector<TokenId> control;
        if (!options.thinking_budget_message.empty()) {
            for (const int token : tokenizer->encode(options.thinking_budget_message, {.parse_added_tokens = false})) {
                control.push_back(token);
            }
        }
        control.push_back(markers.end_message);
        thinking_control = std::make_shared<const std::vector<TokenId>>(std::move(control));
        if (vision) {
            if (max_cells == 0) { throw std::invalid_argument("inkling2 frontend: the image-token budget must be positive"); }
            image_token = id(vision->image_token.c_str());
            (void)id(vision->image_start_token.c_str());
        }
    }

    // The template's image placeholders expanded to each image's cells: the ids, and each image's first token.
    struct Expanded {
        std::vector<TokenId> ids;
        std::vector<std::uint32_t> first;
    };
    [[nodiscard]] Expanded expand(std::span<const TokenId> ids, std::span<const std::int32_t> cells) const {
        Expanded out;
        std::size_t image = 0;
        for (const TokenId token : ids) {
            if (token != image_token || image_token < 0) {
                out.ids.push_back(token);
                continue;
            }
            if (image >= cells.size()) { throw std::logic_error("inkling2 frontend: more image placeholders than images"); }
            out.first.push_back(static_cast<std::uint32_t>(out.ids.size()));
            out.ids.insert(out.ids.end(), static_cast<std::size_t>(cells[image++]), token);
        }
        if (image != cells.size()) { throw std::logic_error("inkling2 frontend: fewer image placeholders than images"); }
        return out;
    }

    // The tokens a render takes once its images are expanded (the context check), from the images' geometry.
    [[nodiscard]] std::vector<std::int32_t> geometry_cells(std::span<const OwnedMedia* const> media) const {
        std::vector<std::int32_t> out;
        for (const OwnedMedia* item : media) {
            const MediaGeometry g = image_geometry(*item);
            const ImageCells c    = image_cells(g.width, g.height, *vision, max_cells);
            if (c.resized && item->image_resize_policy == ImageResizePolicy::RejectOversized) {
                throw RequestError(RequestErrorKind::MediaBudgetExceeded,
                                   "Inkling 2 image: " + std::to_string(g.width) + " x " + std::to_string(g.height) +
                                       " pixels take more than the " + std::to_string(max_cells) +
                                       " image tokens --vision-max-merged allows, and the request refuses downsizing");
            }
            out.push_back(c.rows * c.cols);
        }
        return out;
    }

    // The template's context: messages, the declared tools and the thinking effort; the images, in order, into `media`.
    [[nodiscard]] Json context(const PromptInput& input, std::vector<const OwnedMedia*>& media) const {
        const PromptOptions& options = input.options;
        if (options.continuation != PromptContinuationMode::NewAssistantTurn) {
            throw std::invalid_argument("Inkling 2 renders a new assistant turn only (no continuation modes yet)");
        }
        Json out = parse_object(options.chat_template_kwargs_json, "chat_template_kwargs");
        if (options.reasoning_effort) {
            out["reasoning_effort"] = reasoning_effort_name(*options.reasoning_effort);
        } else if (options.enable_thinking == false) {
            out["reasoning_effort"] = "none"; // the template's lowest effort (it does not turn thinking off)
        }
        Json messages = Json::array();
        for (const ChatMessage& message : input.messages) {
            messages.push_back(message_json(message, vision ? &media : nullptr));
        }
        out["messages"] = std::move(messages);
        if (!options.tool_jsons.empty()) {
            Json tools = Json::array();
            for (const std::string& tool : options.tool_jsons) { tools.push_back(parse_object(tool, "a tool declaration")); }
            out["tools"] = std::move(tools);
        } else {
            out.erase("tools");
        }
        return out;
    }

    // The render before image expansion (one placeholder an image), its images in `media`.
    [[nodiscard]] Rendered render(const PromptInput& input, std::size_t max_tokens,
                                  std::vector<const OwnedMedia*>& media) const {
        const Json ctx = context(input, media);
        text::TemplateOutput rendered;
        try {
            rendered = tmpl.render(ctx);
        } catch (const RequestError&) {
            throw;
        } catch (const std::exception& error) {
            throw std::invalid_argument(std::string("chat template: ") + error.what());
        }
        Rendered out;
        out.ids = tokenizer
                      ->encode_with_boundaries(rendered.text, {}, {.parse_added_tokens = true, .max_tokens = max_tokens},
                                               rendered.literal_spans)
                      .input_ids;
        out.text = std::move(rendered.text);
        // PromptOptions::open_answer: the reply's answer opener ends the prompt (the model writes no thinking block
        // first), so the next token is the answer's first.
        if (input.options.open_answer) {
            out.ids.push_back(markers.message_model);
            out.ids.push_back(markers.content[0]);
            out.text += "<|message_model|><|content_text|>";
        }
        return out;
    }

    // The calls a turn may make: the declared and deferred tools, less the withdrawn ones; any name when the client
    // keeps its own.
    [[nodiscard]] std::shared_ptr<const ToolNames> tool_names(const PromptOptions& options) const {
        auto names  = std::make_shared<ToolNames>();
        names->open = options.open_tool_names;
        std::size_t index = 0;
        for (const auto* list : {&options.tool_jsons, &options.deferred_tool_jsons}) {
            for (const std::string& tool : *list) {
                DeclaredTool declared = declared_tool(parse_object(tool, "a tool declaration"), index++);
                names->declared.insert(declared.name);
                names->tools.push_back(std::move(declared));
            }
        }
        for (const std::string& withdrawn : options.withdrawn_tool_names) {
            names->declared.erase(withdrawn);
            std::erase_if(names->tools, [&](const DeclaredTool& tool) { return tool.name == withdrawn; });
        }
        if (names->declared.empty() && !names->open) { return nullptr; }
        return names;
    }

    // The constraint compiler, built at its first use (warm_constraints builds it beside the Engine's startup).
    [[nodiscard]] text::GrammarCompiler& grammars() const {
        std::call_once(grammar_once, [&] {
            grammar_compiler = make_grammar_compiler(*tokenizer, markers, defaults.token_ids, grammar_cache_bytes);
        });
        return *grammar_compiler;
    }
    std::size_t grammar_cache_bytes = 0;
    mutable std::once_flag grammar_once;
    mutable std::unique_ptr<text::GrammarCompiler> grammar_compiler;

    [[nodiscard]] StopPolicy merge(const StopPolicy& caller) const {
        StopPolicy result;
        result.publish_stop_token = caller.publish_stop_token;
        const auto append_token   = [&](TokenId token) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " + std::to_string(token));
            }
            if (std::find(result.token_ids.begin(), result.token_ids.end(), token) == result.token_ids.end()) {
                result.token_ids.push_back(token);
            }
        };
        if (caller.include_model_defaults) {
            for (const TokenId token : defaults.token_ids) { append_token(token); }
        }
        for (const TokenId token : caller.token_ids) { append_token(token); }
        for (const StopString& stop : caller.strings) {
            if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
            (void)text::unicode_internal::utf8_codepoints(stop.text, "stop string");
            const auto duplicate = std::find_if(result.strings.begin(), result.strings.end(), [&](const StopString& s) {
                return s.text == stop.text && s.channel == stop.channel && s.include_in_output == stop.include_in_output;
            });
            if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
        }
        return result;
    }
};

PromptSummary PreparedPrompt::summary() const {
    if (!data_) { throw std::logic_error("prepared prompt is empty"); }
    // The model opens its thinking blocks in its output, at every effort (none included), unless the prompt opened
    // the answer (PromptOptions::open_answer).
    PromptSummary out{.starts_in_reasoning = false,
                      .opens_reasoning     = !data_->answer_opened,
                      .prompt_tokens       = static_cast<std::uint32_t>(data_->token_ids.size())};
    out.has_media = !data_->images.empty();
    for (const PromptImage& image : data_->images) { out.media.push_back(image.image.source); }
    return out;
}

PromptPreparationStats PreparedPrompt::preparation_stats() const noexcept {
    return data_ ? data_->stats : PromptPreparationStats{};
}

std::span<const TokenId> PreparedPrompt::token_ids() const noexcept {
    return data_ ? std::span<const TokenId>(data_->token_ids) : std::span<const TokenId>{};
}

Frontend::Frontend(const FrontendResources& resources, FrontendOptions options)
    : impl_(std::make_shared<const Impl>(resources, options)) {}

Frontend::Rendered Frontend::render(const PromptInput& input, std::size_t max_tokens) const {
    std::vector<const OwnedMedia*> media;
    Rendered rendered = impl_->render(input, max_tokens, media);
    if (!media.empty()) { rendered.ids = impl_->expand(rendered.ids, impl_->geometry_cells(media)).ids; }
    return rendered;
}

namespace {

// A cancelled or expired request stops between images and between an image's cell rows.
void check_control(const PreparationControl& control) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled, "inference request was cancelled during image preparation");
    }
    if (control.deadline != Clock::time_point{} && Clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout, "inference request expired during image preparation");
    }
}

// The prompt's length past the context limit, as the Engine reports it (`limit` = max_context + 1 bounds the
// tokenization: a render that reached it is "more than").
[[noreturn]] void context_exceeded(std::size_t tokens, std::uint32_t max_context, bool exact) {
    throw RequestError(ContextLengthDetails{.prompt_tokens = tokens, .max_context = max_context, .exact = exact});
}

} // namespace

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    const auto start  = Clock::now();
    const auto limit  = static_cast<std::size_t>(impl_->max_context) + 1U;
    std::vector<const OwnedMedia*> media;
    Rendered rendered = impl_->render(input, limit, media);
    const auto encoded = Clock::now();
    const bool exact   = rendered.ids.size() < limit;
    if (rendered.ids.size() > impl_->max_context) { context_exceeded(rendered.ids.size(), impl_->max_context, exact); }
    auto data = std::make_unique<PreparedPromptData>();
    if (!media.empty()) {
        // The expanded length from the images' geometry first (an overlong prompt fails before any pixel decode).
        const std::vector<std::int32_t> cells = impl_->geometry_cells(media);
        std::size_t total                      = rendered.ids.size() - media.size();
        for (const std::int32_t c : cells) { total += static_cast<std::size_t>(c); }
        if (total > impl_->max_context) { context_exceeded(total, impl_->max_context, exact); }
        const auto media_start = Clock::now();
        std::vector<std::int32_t> processed_cells;
        for (const OwnedMedia* item : media) {
            check_control(control);
            PromptImage image;
            image.image = process_image(*item, *impl_->vision, impl_->max_cells,
                                        [&] { check_control(control); });
            processed_cells.push_back(image.image.cells());
            data->stats.media_bytes += item->bytes.size();
            data->stats.raw_patches += static_cast<std::uint64_t>(image.image.cells()) * impl_->vision->merge *
                                       impl_->vision->merge;
            data->stats.vision_tokens += static_cast<std::uint64_t>(image.image.cells());
            data->stats.patch_bytes += image.image.patches->size() * 2;
            data->images.push_back(std::move(image));
        }
        Impl::Expanded expanded = impl_->expand(rendered.ids, processed_cells);
        for (std::size_t i = 0; i < data->images.size(); ++i) { data->images[i].first_token = expanded.first[i]; }
        rendered.ids                       = std::move(expanded.ids);
        data->stats.media_items            = media.size();
        data->stats.media_preprocess_seconds = std::chrono::duration<double>(Clock::now() - media_start).count();
        data->stats.media_preprocess_work_seconds = data->stats.media_preprocess_seconds;
    }
    data->token_ids     = std::move(rendered.ids);
    data->tools         = impl_->tool_names(input.options);
    data->answer_opened = input.options.open_answer;
    // The prefix cache's keys: each image's content key accumulated into every block at or past its first token's.
    {
        namespace pc         = runtime::prefix_cache;
        const std::size_t blocks = data->token_ids.size() / pc::kBlockTokens;
        if (!data->images.empty()) {
            data->block_extras.resize(blocks);
            std::uint64_t cumulative = 0;
            std::size_t next         = 0;
            for (std::size_t block = 0; block < blocks; ++block) {
                const std::uint64_t end = static_cast<std::uint64_t>(block + 1) * pc::kBlockTokens;
                while (next < data->images.size() && data->images[next].first_token < end) {
                    check_control(control);
                    cumulative = image_prefix_key(cumulative, data->images[next]);
                    ++next;
                }
                data->block_extras[block] = cumulative;
            }
        }
        data->block_hashes = pc::block_lookup_hashes(data->token_ids, data->block_extras);
        data->tap_hints    = impl_->tap_hints(data->token_ids);
    }
    data->stats.seconds          = std::chrono::duration<double>(Clock::now() - start).count();
    data->stats.tokenize_seconds = std::chrono::duration<double>(encoded - start).count();
    return PreparedPrompt(std::move(data));
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl&) const {
    const auto limit = static_cast<std::size_t>(impl_->max_context) + 1U;
    std::vector<const OwnedMedia*> media;
    const Rendered rendered = impl_->render(input, limit, media);
    const bool exact        = rendered.ids.size() < limit;
    std::size_t total       = rendered.ids.size();
    if (!media.empty() && total <= impl_->max_context) {
        total -= media.size();
        for (const std::int32_t c : impl_->geometry_cells(media)) { total += static_cast<std::size_t>(c); }
    }
    if (total > impl_->max_context) { context_exceeded(total, impl_->max_context, exact); }
    return static_cast<std::uint32_t>(total);
}

VisionImageBudget Frontend::vision_image_budget() const noexcept {
    if (!impl_->vision) { return {}; }
    const std::uint64_t cell = impl_->vision->cell();
    return {.prompt_tokens = impl_->max_context,
            .token_edge    = static_cast<std::uint32_t>(cell),
            .min_pixels    = cell * cell,
            .max_pixels    = static_cast<std::uint64_t>(impl_->max_cells) * cell * cell};
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids, bool allow_prefix_identity) const {
    const auto start = Clock::now();
    if (token_ids.size() > impl_->max_context) {
        throw RequestError(ContextLengthDetails{.prompt_tokens = token_ids.size(), .max_context = impl_->max_context});
    }
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the checkpoint vocabulary: " + std::to_string(token));
        }
    }
    auto data       = std::make_unique<PreparedPromptData>();
    data->token_ids = std::move(token_ids);
    data->reusable  = allow_prefix_identity;
    if (allow_prefix_identity) {
        data->block_hashes = runtime::prefix_cache::block_lookup_hashes(data->token_ids, {});
        data->tap_hints    = impl_->tap_hints(data->token_ids);
    }
    data->stats.seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(data));
}

std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    const std::vector<int> ids = impl_->tokenizer->encode(text, {.parse_added_tokens = false});
    return {ids.begin(), ids.end()};
}

std::string Frontend::detokenize(std::span<const TokenId> token_ids) const { return impl_->tokenizer->decode(token_ids); }

std::string Frontend::detokenize_bytes(std::span<const TokenId> token_ids) const {
    return impl_->tokenizer->decode_bytes(token_ids);
}

ChatTokens Frontend::tokenize_chat(PromptInput input) const {
    Rendered rendered = render(input, std::numeric_limits<std::size_t>::max());
    ChatTokens out;
    out.generated.resize(rendered.ids.size(), 0);
    bool inside = false;
    for (std::size_t i = 0; i < rendered.ids.size(); ++i) {
        const TokenId id = rendered.ids[i];
        if (id == impl_->markers.message_model) { inside = true; }
        out.generated[i] = inside ? 1 : 0;
        if (id == impl_->markers.end_of_turn) { inside = false; }
    }
    out.ids = std::move(rendered.ids);
    return out;
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller_stop,
                                            const OutputOptions& output, const ThinkingControlOptions& thinking,
                                            const std::optional<OutputConstraint>& constraint,
                                            const ToolChoice& tool_choice) const {
    if (!prompt) { throw std::invalid_argument("output session needs a prepared prompt"); }
    // The constrained route (constraints.h): a content constraint, a tool policy, or both.
    const std::optional<ToolPolicy> tools = select_tool_policy(prompt.data().tools.get(), tool_choice);
    if (prompt.data().answer_opened && (constraint || tools)) {
        // The grammars frame a whole turn; a prompt that opened the answer (/v1/decide) takes token constraints only.
        throw std::invalid_argument("a prompt that opens the answer takes no output constraint or tool policy");
    }
    std::shared_ptr<text::GrammarSession> matcher;
    if (constraint || tools) {
        const RequestErrorKind kind =
            tools ? RequestErrorKind::InvalidToolConstraint : text::constraint_error_kind(constraint->kind);
        const RequestErrorSource source = tools ? RequestErrorSource::Tools : RequestErrorSource::OutputConstraint;
        // The grammar's end is the model's stop token, and its language the decoded text.
        if (!caller_stop.token_ids.empty() || !caller_stop.strings.empty() || !caller_stop.include_model_defaults ||
            caller_stop.publish_stop_token || output.raw || output.preserve_special_tokens) {
            throw RequestError(kind, "constraints require default EOS, text output, no custom stops, and one output language",
                               {}, source);
        }
        try {
            // A new grammar's token-mask compile runs beside the prefill (GrammarCompiler::Compile::Deferred).
            matcher = compile_turn_grammar(impl_->grammars(), impl_->markers, constraint, tools ? &*tools : nullptr,
                                           text::GrammarCompiler::Compile::Deferred);
        } catch (const RequestError&) {
            throw;
        } catch (const std::invalid_argument& error) {
            throw RequestError(kind, error.what(), {}, source);
        }
    }
    OutputSession::Settings settings;
    settings.tokenizer        = impl_->tokenizer;
    settings.markers          = impl_->markers;
    settings.policy           = impl_->merge(caller_stop);
    settings.output           = output;
    settings.thinking         = thinking;
    settings.thinking_control = impl_->thinking_control;
    // tool_choice none: the turn makes no call (its grammar has none, or, free, a call is text).
    settings.tools         = tool_choice.mode == ToolChoiceMode::None ? nullptr : prompt.data().tools;
    settings.grammar       = std::move(matcher);
    settings.answer_opened = prompt.data().answer_opened;
    return OutputSession(std::move(settings));
}

void Frontend::warm_constraints() const { (void)impl_->grammars(); }

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept { return impl_->sampling; }

} // namespace infernix::models::inkling2
