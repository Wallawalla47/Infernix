#include "serve/decide.h"

#include "serve/openai_chat.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::serve {
namespace {

using Json = RequestJson;

// Prompt JSON is the caller's value in the caller's key order; invalid UTF-8 is replaced rather
// than refused, as the chat adapters' template rendering does.
std::string compact(const Json& value) {
    return value.dump(-1, ' ', false, Json::error_handler_t::replace);
}

std::string question_param(const std::string& id) { return "questions." + id; }

[[noreturn]] void refuse(const std::string& id, std::string message, std::string code) {
    bad_request("question \"" + ascii_preview(id) + "\": " + std::move(message),
                question_param(id), std::move(code));
}

bool blank(std::string_view text) {
    return std::all_of(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c); });
}

bool instructions_empty(const Json& value) {
    if (value.is_string()) { return blank(value.get_ref<const std::string&>()); }
    if (value.is_object() || value.is_array()) { return value.empty(); }
    return true;
}

// Refuses anything that would make the readout report a reasoning trace's first token.
void refuse_thinking(const Json& body) {
    const auto enabled = [](const Json& object, const char* key) {
        return object.contains(key) && object.at(key).is_boolean() && object.at(key).get<bool>();
    };
    bool thinking = enabled(body, "enable_thinking");
    if (body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").is_object()) {
        thinking = thinking || enabled(body.at("chat_template_kwargs"), "enable_thinking");
    }
    if (body.contains("reasoning_effort") && body.at("reasoning_effort").is_string()) {
        const std::string effort = body.at("reasoning_effort").get<std::string>();
        thinking = thinking || (effort != "none" && effort != "default" && effort != "auto");
    }
    if (thinking) {
        bad_request("a decision is read at the position its prompt ends, where a thinking prompt "
                    "opens a reasoning block; /v1/decide runs with thinking off",
                    "enable_thinking", "thinking_unsupported");
    }
}

// An array is content parts only when every element is an object with a string type, which
// evidence like [{"id": 1, "title": "..."}] does not have.
bool content_parts_shaped(const Json& state) {
    if (!state.is_array() || state.empty()) { return false; }
    return std::all_of(state.begin(), state.end(), [](const Json& item) {
        return item.is_object() && item.contains("type") && item.at("type").is_string();
    });
}

const Json* field(const Json& object, const char* name, const char* alias) {
    if (object.contains(name) && !object.at(name).is_null()) { return &object.at(name); }
    if (alias != nullptr && object.contains(alias) && !object.at(alias).is_null()) {
        return &object.at(alias);
    }
    return nullptr;
}

std::vector<DecideOption> noul_options(const std::string& id, const Json* criteria) {
    if (criteria != nullptr && !criteria->is_object()) {
        refuse(id, "a noul's criteria must be an object of true and false", "malformed_criteria");
    }
    std::vector<DecideOption> options;
    for (const auto& [name, fallback] : {std::pair<const char*, const char*>{"true", "Yes"},
                                         std::pair<const char*, const char*>{"false", "No"}}) {
        std::string description = fallback;
        if (criteria != nullptr && criteria->contains(name) && !criteria->at(name).is_null()) {
            const Json& stated = criteria->at(name);
            if (!stated.is_string() || blank(stated.get_ref<const std::string&>())) {
                refuse(id, std::string("criteria.") + name + " must be a non-empty string or null",
                       "malformed_criteria");
            }
            description = stated.get<std::string>();
        }
        options.push_back(DecideOption{.name = name, .description = std::move(description)});
    }
    return options;
}

std::vector<DecideOption> choice_options(const std::string& id, const Json* criteria) {
    if (criteria == nullptr) {
        refuse(id, "a choice requires criteria", "missing_criteria");
    }
    if (!criteria->is_object()) {
        refuse(id, "a choice's criteria must be an object of option to description",
               "malformed_criteria");
    }
    if (criteria->empty()) { refuse(id, "a choice must declare at least one option", "no_options"); }
    std::vector<DecideOption> options;
    for (const auto& [name, description] : criteria->items()) {
        if (blank(name)) { refuse(id, "an option has a blank name", "unclean_option"); }
        if (description.is_null()) {
            // Jev's rule: an option that needs no detail is its own description.
            options.push_back(DecideOption{.name = name, .description = name});
        } else if (description.is_string() &&
                   !blank(description.get_ref<const std::string&>())) {
            options.push_back(
                DecideOption{.name = name, .description = description.get<std::string>()});
        } else {
            refuse(id, "option \"" + ascii_preview(name) +
                           "\" must describe itself with a non-empty string or null",
                   "malformed_criteria");
        }
    }
    return options;
}

std::vector<DecideOption> score_options(const std::string& id, const Json* criteria) {
    if (criteria == nullptr || !criteria->is_array()) {
        refuse(id, "a score's criteria must be an ordered array of levels", "malformed_criteria");
    }
    if (criteria->size() < 2) {
        refuse(id, "a score must declare at least two levels", "too_few_levels");
    }
    std::vector<DecideOption> options;
    for (std::size_t index = 0; index < criteria->size(); ++index) {
        const Json& level = criteria->at(index);
        if (!level.is_string() || blank(level.get_ref<const std::string&>())) {
            refuse(id, "score level " + std::to_string(index) + " must be a non-empty string",
                   "malformed_criteria");
        }
        options.push_back(
            DecideOption{.name = std::to_string(index), .description = level.get<std::string>()});
    }
    return options;
}

DecideQuestion parse_question(const std::string& id, const Json& item,
                              std::size_t alphabet_size) {
    if (!item.is_object()) { refuse(id, "a question must be an object", "malformed_question"); }
    static const std::set<std::string, std::less<>> known = {"type",     "instructions",
                                                             "question", "criteria",
                                                             "options",  "digits"};
    for (const auto& [key, value] : item.items()) {
        // A field a caller wrote is never silently dropped: ignis's generated and attention
        // primitives (digits, method, within, ...) are not served here.
        if (!known.contains(key)) {
            refuse(id, "field \"" + ascii_preview(key) + "\" is not supported",
                   "unsupported_field");
        }
    }
    if (!item.contains("type") || !item.at("type").is_string()) {
        refuse(id, "type must be one of noul, choice, score, number, scalar, point or box",
               "malformed_question");
    }
    const std::string type = item.at("type").get<std::string>();
    DecideQuestion question;
    question.id = id;
    if (type == "noul" || type == "boolean") {
        question.kind = DecideKind::Noul;
    } else if (type == "choice") {
        question.kind = DecideKind::Choice;
    } else if (type == "score") {
        question.kind = DecideKind::Score;
    } else if (type == "number") {
        question.kind = DecideKind::Number;
    } else if (type == "scalar") {
        question.kind = DecideKind::Scalar;
    } else if (type == "point") {
        question.kind = DecideKind::Point;
    } else if (type == "box") {
        question.kind = DecideKind::Box;
    } else {
        refuse(id,
               "type \"" + ascii_preview(type) +
                   "\" is not served; this endpoint serves noul, choice, score, number, scalar, "
                   "point and box",
               "question_type_unsupported");
    }
    const Json* instructions = field(item, "instructions", "question");
    if (instructions == nullptr || instructions_empty(*instructions)) {
        refuse(id, "instructions must be a non-empty string, object or array",
               "empty_instructions");
    }
    question.instructions = *instructions;
    const Json* criteria  = field(item, "criteria", "options");
    const Json* digits    = field(item, "digits", nullptr);
    if (decide_generates(question.kind)) {
        // A field a caller wrote is never ignored: a generated answer has no options.
        if (criteria != nullptr) {
            refuse(id, "a " + type + " generates its answer and declares no criteria",
                   "criteria_unsupported");
        }
        const bool scalar          = question.kind == DecideKind::Scalar;
        const std::uint32_t most   = scalar ? kDecideScalarDigitsMax : kDecideFieldDigitsMax;
        question.digits            = scalar ? kDecideScalarDigitsDefault
                                            : kDecideFieldDigitsDefault;
        if (digits != nullptr) {
            if (!digits->is_number_integer() || digits->get<std::int64_t>() < 1 ||
                digits->get<std::int64_t>() > static_cast<std::int64_t>(most)) {
                refuse(id, "digits must be an integer from 1 to " + std::to_string(most),
                       "digits_out_of_range");
            }
            question.digits = static_cast<std::uint32_t>(digits->get<std::int64_t>());
        }
        return question;
    }
    if (digits != nullptr) {
        refuse(id, "a " + type + " reads one position, so digits cannot be honoured",
               "digits_unsupported");
    }
    switch (question.kind) {
    case DecideKind::Noul:
        question.options = noul_options(id, criteria);
        break;
    case DecideKind::Choice:
        question.options = choice_options(id, criteria);
        break;
    case DecideKind::Score:
        question.options = score_options(id, criteria);
        break;
    default:
        break;
    }
    const std::size_t ceiling = std::min(kDecideMaximumOptions, alphabet_size);
    if (question.options.size() > ceiling) {
        refuse(id,
               "declares " + std::to_string(question.options.size()) + " options; " +
                   std::to_string(ceiling) + " is the most this model's labels can name",
               "too_many_options");
    }
    return question;
}

double log_sum_exp(std::span<const float> values) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (const float value : values) { maximum = std::max(maximum, static_cast<double>(value)); }
    if (!std::isfinite(maximum)) { return maximum; }
    double sum = 0.0;
    for (const float value : values) { sum += std::exp(static_cast<double>(value) - maximum); }
    return maximum + std::log(sum);
}

std::uint64_t field_scale(std::uint32_t digits) {
    std::uint64_t scale = 1;
    for (std::uint32_t i = 0; i < digits; ++i) { scale *= 10; }
    return scale - 1;
}

std::string spelled(std::uint32_t digits) {
    static constexpr const char* kNames[] = {"one", "two", "three", "four", "five", "six"};
    return digits >= 1 && digits <= 6 ? kNames[digits - 1] : std::to_string(digits);
}

// ignis's system texts (crates/server/src/numbers.rs, scalar.rs), byte for byte: point at three
// digits is the text its pointing accuracy was measured with.
std::string system_text(const DecideQuestion& question) {
    const std::string max   = std::to_string(field_scale(question.digits));
    const std::string field = std::string(question.digits, 'N');
    switch (question.kind) {
    case DecideKind::Number:
        return "Apply the supplied instruction to the supplied evidence and answer with a single "
               "whole number from 0 to " +
               max + ". Reply with only a JSON object of the form {\"value\":" + field + "}, " +
               spelled(question.digits) + " digits, right-aligned and padded on the left with zeros.";
    case DecideKind::Scalar:
        return "Apply the supplied instruction to the supplied evidence and answer with a single "
               "number, which may have a decimal part and may be negative. Reply with only a JSON "
               "object of the form {\"value\":N}, using at most " +
               std::to_string(question.digits) +
               " digits, and close the object as soon as the number is complete.";
    case DecideKind::Point:
        return "You are given a screenshot and an instruction. Answer with the position on the "
               "screen the instruction refers to. Use a 0-" +
               max + " scale on each axis, where x=0 is the left edge, x=" + max +
               " the right edge, y=0 the top edge and y=" + max +
               " the bottom edge. Reply with only a JSON object of the form {\"x\":" + field +
               ",\"y\":" + field + "}, " + spelled(question.digits) + " digits each.";
    case DecideKind::Box:
        return "You are given a screenshot and an instruction. Answer with the bounding box on the "
               "screen the instruction refers to. Use a 0-" +
               max + " scale on each axis, where x=0 is the left edge, x=" + max +
               " the right edge, y=0 the top edge and y=" + max +
               " the bottom edge. Reply with only a JSON object of the form {\"x0\":" + field +
               ",\"y0\":" + field + ",\"x1\":" + field + ",\"y1\":" + field +
               "}, where (x0,y0) is the top-left corner and (x1,y1) the bottom-right, " +
               spelled(question.digits) + " digits each.";
    default:
        return std::string(kDecideSystemInstruction);
    }
}

std::vector<double> restricted_probabilities(std::span<const float> logprobs) {
    const double total = log_sum_exp(logprobs);
    std::vector<double> probabilities(logprobs.size(), 0.0);
    if (!std::isfinite(total)) { return probabilities; }
    for (std::size_t i = 0; i < logprobs.size(); ++i) {
        probabilities[i] = std::exp(static_cast<double>(logprobs[i]) - total);
    }
    return probabilities;
}

} // namespace

std::vector<DecideLabel>
build_decide_alphabet(const std::function<std::vector<infernix::TokenId>(std::string_view)>& encode) {
    std::vector<std::string> candidates;
    for (char c = 'A'; c <= 'Z'; ++c) { candidates.emplace_back(1, c); }
    for (char c = 'a'; c <= 'z'; ++c) { candidates.emplace_back(1, c); }
    for (char c = '0'; c <= '9'; ++c) { candidates.emplace_back(1, c); }
    for (char a = 'A'; a <= 'Z'; ++a) {
        for (char b = 'A'; b <= 'Z'; ++b) { candidates.push_back(std::string{a, b}); }
    }
    std::vector<DecideLabel> labels;
    std::set<infernix::TokenId> seen;
    for (std::string& label : candidates) {
        const std::vector<infernix::TokenId> ids = encode(label);
        if (ids.size() != 1 || !seen.insert(ids.front()).second) { continue; }
        labels.push_back(DecideLabel{.text = std::move(label), .token = ids.front()});
    }
    return labels;
}

DecideRequest parse_decide_request(const RequestJson& body, std::size_t alphabet_size) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    refuse_thinking(body);
    DecideRequest request;
    if (body.contains("model") && !body.at("model").is_null()) {
        if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
        request.model = body.at("model").get<std::string>();
    }
    if (!body.contains("state") || body.at("state").is_null()) {
        bad_request("missing required field: state", "state");
    }
    const Json& state = body.at("state");
    if (content_parts_shaped(state)) {
        ChatTurn turn;
        turn.role = ChatRole::User;
        parse_openai_message_content(state, turn, 0);
        request.parts = std::move(turn.content);
    } else {
        request.state = state;
    }
    if (!body.contains("questions") || !body.at("questions").is_object() ||
        body.at("questions").empty()) {
        bad_request("questions must be a non-empty object of question id to question",
                    "questions", "no_questions");
    }
    for (const auto& [id, item] : body.at("questions").items()) {
        request.questions.push_back(parse_question(id, item, alphabet_size));
    }
    const auto images = static_cast<std::size_t>(
        std::count_if(request.parts.begin(), request.parts.end(),
                      [](const ContentPart& part) { return part.kind == ContentKind::Image; }));
    for (const DecideQuestion& question : request.questions) {
        const bool spatial =
            question.kind == DecideKind::Point || question.kind == DecideKind::Box;
        if (spatial && images != 1) {
            // The answer is in the submitted image's pixels, so there must be exactly one.
            refuse(question.id, "a point or box is answered over a state carrying exactly one image",
                   "spatial_needs_one_image");
        }
    }
    return request;
}

std::vector<ChatTurn> decide_messages(const DecideRequest& request,
                                      const DecideQuestion& question,
                                      std::span<const DecideLabel> labels) {
    if (labels.size() < question.options.size()) {
        throw std::logic_error("decide question has more options than labels");
    }
    std::string ask;
    if (decide_generates(question.kind)) {
        // A generated answer's shape lives in the system text; the user turn is the instruction.
        ask = "{\"instruction\":" + compact(question.instructions) + "}";
    } else {
        // `description` before `letter`: the key order every published measurement used.
        Json options = Json::array();
        for (std::size_t i = 0; i < question.options.size(); ++i) {
            Json option;
            option["description"] = question.options[i].description;
            option["letter"]      = labels[i].text;
            options.push_back(std::move(option));
        }
        ask = "{\"criterion\":" + compact(question.instructions) +
              ",\"options\":" + compact(options) + "}";
    }
    const std::string instruction = system_text(question);
    const auto text_part = [](std::string text) {
        return ContentPart{.kind = ContentKind::Text, .text = std::move(text), .type_raw = "text"};
    };
    std::vector<ChatTurn> messages(2);
    messages[0].role = ChatRole::System;
    messages[1].role = ChatRole::User;
    if (request.has_parts()) {
        messages[0].content.push_back(text_part(instruction));
        messages[1].content = request.parts;
        // An explicit boundary after the evidence makes it an exact prefix-cache resume point:
        // every later question over the same parts resumes past their images instead of
        // encoding and prefilling them again. (A JSON state ends the system block, which is a
        // structural boundary already.)
        messages[1].content.back().cache_boundary_after = CacheBoundary{};
        messages[1].content.push_back(text_part(ask));
    } else {
        // One blank line, then the evidence: the instruction is the same bytes for every
        // question over every state, so it leads the shared prefix.
        messages[0].content.push_back(
            text_part(instruction + "\n\n{\"evidence\":" + compact(request.state) + "}"));
        messages[1].content.push_back(text_part(ask));
    }
    return messages;
}

RequestJson decide_answer(const DecideQuestion& question, std::span<const float> logprobs) {
    if (logprobs.size() != question.options.size()) {
        throw std::logic_error("decide readout does not match the question's options");
    }
    const std::vector<double> probabilities = restricted_probabilities(logprobs);
    Json answer;
    const auto named = [&] {
        Json values = Json::object();
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            values[question.options[i].name] = probabilities[i];
        }
        return values;
    };
    switch (question.kind) {
    case DecideKind::Noul:
        answer["type"] = "noul";
        // The true option is slot 0 by construction.
        answer["noul"] = probabilities.front();
        break;
    case DecideKind::Choice: {
        const auto winner = static_cast<std::size_t>(
            std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
        answer["type"]          = "choice";
        answer["choice"]        = question.options[winner].name;
        answer["probabilities"] = named();
        // The top probability: the abstention signal ignis measured (every answer above 0.9
        // correct on its authored set).
        answer["confidence"] = probabilities[winner];
        break;
    }
    case DecideKind::Score: {
        double mean = 0.0;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            mean += static_cast<double>(i) * probabilities[i];
        }
        double variance = 0.0;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            const double offset = static_cast<double>(i) - mean;
            variance += probabilities[i] * offset * offset;
        }
        // Spread, not height: an even split between adjacent levels knows where the answer is.
        const double half_range = static_cast<double>(probabilities.size() - 1) / 2.0;
        Json legend             = Json::object();
        for (const DecideOption& option : question.options) {
            legend[option.name] = option.description;
        }
        answer["type"]          = "score";
        answer["score"]         = mean;
        answer["legend"]        = std::move(legend);
        answer["probabilities"] = named();
        answer["confidence"]    = std::clamp(1.0 - std::sqrt(variance) / half_range, 0.0, 1.0);
        break;
    }
    }
    answer["answer_mass"] = decide_answer_mass(logprobs);
    return answer;
}

double decide_answer_mass(std::span<const float> logprobs) {
    const double total = log_sum_exp(logprobs);
    if (!std::isfinite(total)) { return 0.0; }
    return std::clamp(std::exp(total), 0.0, 1.0);
}

namespace {

infernix::TokenId single_token(const DecideEncoder& encode, std::string_view text) {
    const std::vector<infernix::TokenId> ids = encode(text);
    if (ids.size() != 1) {
        throw ApiException(ApiError{
            .status  = 500,
            .type    = "internal_error",
            .message = "this model's tokenizer spells \"" + std::string(text) + "\" as " +
                       std::to_string(ids.size()) +
                       " tokens, and a constrained step names single tokens",
            .code    = "tokenizer_unsupported",
        });
    }
    return ids.front();
}

RequestJson run_error(std::string code, std::string message) {
    Json answer;
    answer["type"]    = "error";
    answer["code"]    = std::move(code);
    answer["message"] = std::move(message);
    return answer;
}

struct AxisReading {
    std::uint64_t value = 0;
    double sigma        = 0.0;
    Json digits         = Json::array();
};

int digit_of(const DecideSchedule& schedule, infernix::TokenId token) {
    const auto found = std::find(schedule.digit_tokens.begin(), schedule.digit_tokens.end(), token);
    return found == schedule.digit_tokens.end()
               ? -1
               : static_cast<int>(found - schedule.digit_tokens.begin());
}

Json digit_draw(int digit, double probability) {
    Json draw;
    draw["digit"]       = digit;
    draw["probability"] = probability;
    return draw;
}

// The digit's own probability: digit steps permit the ten digits in digit order (and a scalar's
// steps append its structural tokens after them), so the digit's slot is its value.
double draw_probability(const infernix::ConstrainedDraw& draw, int digit) {
    return digit >= 0 && static_cast<std::size_t>(digit) < draw.probabilities.size()
               ? static_cast<double>(draw.probabilities[static_cast<std::size_t>(digit)])
               : 0.0;
}

// sigma = sum((1 - p_k) * 10^place): ignis's self-declared uncertainty in units of the value. A
// left-padded field (number) skips its padding zeros, which the model fills rather than chooses.
std::optional<AxisReading> read_axis(const DecideSchedule& schedule, const DecideAxis& axis,
                                     std::span<const infernix::ConstrainedDraw> run,
                                     bool left_padded) {
    AxisReading reading;
    const std::size_t width = axis.end - axis.begin;
    std::size_t significant = 0;
    if (left_padded) {
        significant = width - 1;
        for (std::size_t place = 0; place < width; ++place) {
            if (digit_of(schedule, run[axis.begin + place].token) != 0) {
                significant = place;
                break;
            }
        }
    }
    for (std::size_t place = 0; place < width; ++place) {
        const infernix::ConstrainedDraw& draw = run[axis.begin + place];
        const int digit                     = digit_of(schedule, draw.token);
        if (digit < 0) { return std::nullopt; }
        const double probability = draw_probability(draw, digit);
        reading.value            = reading.value * 10 + static_cast<std::uint64_t>(digit);
        if (place >= significant) {
            reading.sigma += (1.0 - probability) *
                             std::pow(10.0, static_cast<double>(width - 1 - place));
        }
        reading.digits.push_back(digit_draw(digit, probability));
    }
    return reading;
}

// The accepted spelling of a scalar: an optional sign, digits, and at most one point followed by
// digits. Enumerated rather than parsed, because a parser accepts "3." as 3.
bool well_formed_scalar(std::string_view text) {
    if (!text.empty() && text.front() == '-') { text.remove_prefix(1); }
    const std::size_t point = text.find('.');
    const auto plain        = [](std::string_view part) {
        return !part.empty() &&
               std::all_of(part.begin(), part.end(), [](char c) { return c >= '0' && c <= '9'; });
    };
    if (point == std::string_view::npos) { return plain(text); }
    return plain(text.substr(0, point)) && plain(text.substr(point + 1));
}

} // namespace

DecideSchedule decide_schedule(const DecideQuestion& question, const DecideEncoder& encode) {
    if (!decide_generates(question.kind)) {
        throw std::logic_error("only generated decide questions have a schedule");
    }
    DecideSchedule schedule;
    for (int digit = 0; digit < 10; ++digit) {
        schedule.digit_tokens[static_cast<std::size_t>(digit)] =
            single_token(encode, std::to_string(digit));
    }
    const std::vector<infernix::TokenId> digits(schedule.digit_tokens.begin(),
                                              schedule.digit_tokens.end());
    std::vector<std::vector<infernix::TokenId>>& steps = schedule.constraint.steps;
    // Literals are encoded standalone, as ignis appends them, and forced one token per step.
    const auto force = [&](std::string_view literal) {
        const std::vector<infernix::TokenId> ids = encode(literal);
        if (ids.empty()) {
            throw ApiException(ApiError{.status  = 500,
                                        .type    = "internal_error",
                                        .message = "this model's tokenizer cannot encode a literal",
                                        .code    = "tokenizer_unsupported"});
        }
        for (const infernix::TokenId id : ids) { steps.push_back({id}); }
    };
    const auto axis = [&](std::string name, std::string_view literal) {
        force(literal);
        const std::size_t begin = steps.size();
        for (std::uint32_t place = 0; place < question.digits; ++place) { steps.push_back(digits); }
        schedule.axes.push_back(DecideAxis{.name = std::move(name), .begin = begin,
                                           .end = steps.size()});
    };
    switch (question.kind) {
    case DecideKind::Number:
        axis("value", "{\"value\":");
        break;
    case DecideKind::Point:
        axis("x", "{\"x\":");
        axis("y", ",\"y\":");
        break;
    case DecideKind::Box:
        axis("x0", "{\"x0\":");
        axis("y0", ",\"y0\":");
        axis("x1", ",\"x1\":");
        axis("y1", ",\"y1\":");
        break;
    case DecideKind::Scalar: {
        // The tokenizer writes `{"value":-7.25}` as `{"`, `value`, `":-`, `7`, ...: the sign
        // merges into the colon's token. ignis forces `":` and then offers a separate `-`, a pair
        // the model almost never writes, so a negative answer held about 1 % of the mass. The
        // colon is a choice instead: `":` opens a positive value, `":-` a negative one.
        force("{\"value");
        schedule.point         = single_token(encode, ".");
        schedule.terminator    = single_token(encode, "}");
        schedule.colon         = single_token(encode, "\":");
        schedule.colon_negated = single_token(encode, "\":-");
        std::vector<infernix::TokenId> rest = digits;
        rest.push_back(schedule.point);
        rest.push_back(schedule.terminator);
        const std::size_t begin = steps.size();
        steps.push_back({schedule.colon, schedule.colon_negated});
        steps.push_back(digits);
        // Room for every digit plus the point and the brace, so a full-width decimal can still
        // close itself.
        for (std::uint32_t step = 1; step < question.digits + 2; ++step) { steps.push_back(rest); }
        schedule.axes.push_back(DecideAxis{.name = "value", .begin = begin, .end = steps.size()});
        break;
    }
    default:
        break;
    }
    if (steps.size() > infernix::kMaximumConstraintSteps) {
        throw ApiException(ApiError{.status  = 400,
                                    .message = "question " + question.id +
                                               " needs more generated steps than are served",
                                    .code    = "too_many_steps"});
    }
    return schedule;
}

RequestJson decide_generated_answer(const DecideQuestion& question, const DecideSchedule& schedule,
                                    std::span<const infernix::ConstrainedDraw> run,
                                    std::span<const infernix::MediaGeometry> media) {
    const std::size_t steps = schedule.constraint.steps.size();
    if (run.size() > steps) {
        return run_error("off_schedule", "the run is longer than its schedule");
    }
    // The least of the chosen steps' masses: how much of the distribution the schedule held at
    // its weakest step.
    double mass = 1.0;
    for (std::size_t index = 0; index < run.size(); ++index) {
        if (schedule.constraint.steps[index].size() > 1) {
            mass = std::min(mass, static_cast<double>(run[index].mass));
        }
    }

    if (question.kind == DecideKind::Scalar) {
        const DecideAxis& axis = schedule.axes.front();
        std::string text;
        Json digits = Json::array();
        std::vector<double> probabilities;
        bool terminated = false;
        for (std::size_t index = axis.begin; index < run.size(); ++index) {
            const infernix::ConstrainedDraw& draw = run[index];
            if (draw.token == schedule.terminator) {
                terminated = true;
                break;
            }
            if (draw.token == schedule.colon) { continue; }
            if (draw.token == schedule.point) {
                text.push_back('.');
            } else if (draw.token == schedule.colon_negated) {
                text.push_back('-');
            } else {
                const int digit = digit_of(schedule, draw.token);
                if (digit < 0) {
                    return run_error("off_schedule", "a step committed a token it does not permit");
                }
                text.push_back(static_cast<char>('0' + digit));
                probabilities.push_back(draw_probability(draw, digit));
                digits.push_back(digit_draw(digit, probabilities.back()));
            }
        }
        if (!terminated && run.size() < steps) {
            return run_error("run_cut_short",
                             "the engine stopped this run before it closed its object or reached "
                             "its cap, so its digits are the front of a number and not a number");
        }
        if (probabilities.size() > question.digits) {
            return run_error("too_many_digits", "the run spells " +
                                                    std::to_string(probabilities.size()) +
                                                    " digits where the question allowed " +
                                                    std::to_string(question.digits));
        }
        if (!well_formed_scalar(text)) {
            return run_error("malformed_scalar", "the run spells \"" + text +
                                                     "\", which is not a number");
        }
        // A digit's place is known only once the point has been seen.
        const std::string_view unsigned_text =
            text.front() == '-' ? std::string_view(text).substr(1) : std::string_view(text);
        const std::size_t integer_digits = std::min(unsigned_text.find('.'), unsigned_text.size());
        double uncertainty = 0.0;
        for (std::size_t index = 0; index < probabilities.size(); ++index) {
            uncertainty += (1.0 - probabilities[index]) *
                           std::pow(10.0, static_cast<double>(integer_digits) - 1.0 -
                                              static_cast<double>(index));
        }
        Json answer;
        answer["type"]        = "scalar";
        answer["value"]       = std::stod(text);
        answer["text"]        = text;
        answer["uncertainty"] = uncertainty;
        answer["digits"]      = std::move(digits);
        answer["answer_mass"] = mass;
        return answer;
    }

    if (run.size() != steps) {
        return run_error("run_cut_short",
                         "the engine stopped this run before its last digit, so its digits are "
                         "the front of a number and not a number");
    }
    const bool left_padded = question.kind == DecideKind::Number;
    std::vector<AxisReading> readings;
    for (const DecideAxis& axis : schedule.axes) {
        std::optional<AxisReading> reading = read_axis(schedule, axis, run, left_padded);
        if (!reading) {
            return run_error("off_schedule", "a digit step committed a token that is not a digit");
        }
        readings.push_back(std::move(*reading));
    }
    Json answer;
    if (question.kind == DecideKind::Number) {
        answer["type"]        = "number";
        answer["number"]      = readings.front().value;
        answer["uncertainty"] = readings.front().sigma;
        answer["digits"]      = std::move(readings.front().digits);
        answer["answer_mass"] = mass;
        return answer;
    }
    if (media.empty() || media.front().width <= 0 || media.front().height <= 0) {
        return run_error("image_size_unknown", "the submitted image's size is unknown");
    }
    // The model answers 0-scale on both axes; each axis is rescaled by its own side, the top of
    // the range mapping to the side length (ignis's arithmetic).
    const double scale = static_cast<double>(field_scale(question.digits));
    Json pixels        = Json::object();
    Json normalized    = Json::object();
    Json uncertainty   = Json::object();
    Json digits        = Json::object();
    for (std::size_t index = 0; index < schedule.axes.size(); ++index) {
        const std::string& name = schedule.axes[index].name;
        const double side       = static_cast<double>(name.front() == 'x' ? media.front().width
                                                                         : media.front().height);
        const double per_unit   = side / scale;
        pixels[name]            = static_cast<std::int64_t>(
            std::llround(static_cast<double>(readings[index].value) * per_unit));
        normalized[name]  = readings[index].value;
        uncertainty[name] = readings[index].sigma * per_unit;
        digits[name]      = std::move(readings[index].digits);
    }
    answer["type"]        = question.kind == DecideKind::Point ? "point" : "box";
    answer["method"]      = "chain";
    answer["pixels"]      = std::move(pixels);
    answer["normalized"]  = std::move(normalized);
    answer["uncertainty"] = std::move(uncertainty);
    answer["digits"]      = std::move(digits);
    answer["answer_mass"] = mass;
    return answer;
}

} // namespace infernix::serve
