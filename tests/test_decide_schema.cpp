// /v1/decide protocol: label alphabet, all-or-nothing validation, the measured prompt layout and
// the answer arithmetic. No Engine: the readout itself is covered by the real-model test.
#include "serve/decide.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace infernix::serve;
using Json = RequestJson;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

bool near(double a, double b, double tolerance = 1e-9) { return std::fabs(a - b) <= tolerance; }

// A fixture tokenizer: every printable ASCII byte is one token (its code), except that the given
// labels split into one token per byte and the given aliases share one id with another label.
struct FixtureTokenizer {
    std::vector<std::string> split;
    std::map<std::string, infernix::TokenId> aliases;

    std::vector<infernix::TokenId> operator()(std::string_view text) const {
        const std::string key(text);
        if (const auto alias = aliases.find(key); alias != aliases.end()) {
            return {alias->second};
        }
        if (text.size() == 1) { return {static_cast<infernix::TokenId>(text[0])}; }
        for (const std::string& label : split) {
            if (label == text) {
                return {static_cast<infernix::TokenId>(text[0]),
                        static_cast<infernix::TokenId>(text[1])};
            }
        }
        // Bigrams are single tokens above the byte range.
        return {static_cast<infernix::TokenId>(1000 + (text[0] - 'A') * 26 + (text[1] - 'A'))};
    }
};

std::string api_code(const Json& body, std::size_t alphabet = 300) {
    try {
        (void)parse_decide_request(body, alphabet);
    } catch (const ApiException& exception) {
        return exception.error().code.empty() ? std::string("<no code>")
                                              : exception.error().code;
    }
    return {};
}

Json choice_body(std::size_t options) {
    Json criteria = Json::object();
    for (std::size_t i = 0; i < options; ++i) { criteria["option" + std::to_string(i)] = nullptr; }
    Json body;
    body["state"]                          = "evidence";
    body["questions"]["q"]["type"]         = "choice";
    body["questions"]["q"]["instructions"] = "Pick one";
    body["questions"]["q"]["criteria"]     = criteria;
    return body;
}

void test_alphabet() {
    const FixtureTokenizer tokenizer{.split = {"BQ"}, .aliases = {{"AC", 1000}}};
    const std::vector<DecideLabel> labels = build_decide_alphabet(std::cref(tokenizer));
    check(labels.size() == 26 + 26 + 10 + 676 - 2, "alphabet drops exactly the two bad bigrams");
    check(labels[0].text == "A" && labels[25].text == "Z" && labels[26].text == "a" &&
              labels[52].text == "0" && labels[62].text == "AA",
          "alphabet keeps pool order A-Z, a-z, 0-9, AA..");
    bool bq = false;
    bool ac = false;
    for (const DecideLabel& label : labels) {
        bq = bq || label.text == "BQ";
        ac = ac || label.text == "AC";
    }
    check(!bq, "a label that encodes to two tokens is not admitted");
    check(!ac, "a label whose token an earlier label already names is not admitted");
}

void test_validation() {
    check(api_code(choice_body(256)).empty(), "256 options are served");
    check(api_code(choice_body(257)) == "too_many_options", "257 options are refused");
    check(api_code(choice_body(10), 9) == "too_many_options",
          "a question wider than the tokenizer's alphabet is refused");

    Json thinking        = choice_body(2);
    thinking["enable_thinking"] = true;
    check(api_code(thinking) == "thinking_unsupported", "enable_thinking=true is refused");
    Json kwargs = choice_body(2);
    kwargs["chat_template_kwargs"]["enable_thinking"] = true;
    check(api_code(kwargs) == "thinking_unsupported", "kwargs enable_thinking=true is refused");
    Json effort              = choice_body(2);
    effort["reasoning_effort"] = "high";
    check(api_code(effort) == "thinking_unsupported", "a reasoning effort is refused");
    Json off              = choice_body(2);
    off["enable_thinking"] = false;
    off["reasoning_effort"] = "none";
    check(api_code(off).empty(), "thinking explicitly off is accepted");

    Json locate                      = choice_body(2);
    locate["questions"]["q"]["type"] = "locate";
    check(api_code(locate) == "question_type_unsupported", "attention-read primitives are refused");
    Json within                        = choice_body(2);
    within["questions"]["q"]["within"] = "/log";
    check(api_code(within) == "unsupported_field", "an unknown question field is refused");
    Json empty                              = choice_body(2);
    empty["questions"]["q"]["instructions"] = "  ";
    check(api_code(empty) == "empty_instructions", "blank instructions are refused");

    // One malformed question refuses the whole request.
    Json mixed                                 = choice_body(2);
    mixed["questions"]["bad"]["type"]          = "score";
    mixed["questions"]["bad"]["instructions"] = "Rate it";
    mixed["questions"]["bad"]["criteria"]     = Json::array({"only one level"});
    check(api_code(mixed) == "too_few_levels", "one malformed question refuses every question");

    Json none;
    none["state"]     = "x";
    none["questions"] = Json::object();
    check(api_code(none) == "no_questions", "an empty questions object is refused");
}

void test_options_and_order() {
    Json body;
    body["state"]                             = "s";
    body["questions"]["first"]["type"]        = "choice";
    body["questions"]["first"]["question"]    = "Which?";
    body["questions"]["first"]["options"]["zeta"]  = "Last letter";
    body["questions"]["first"]["options"]["alpha"] = nullptr;
    body["questions"]["second"]["type"]         = "noul";
    body["questions"]["second"]["instructions"] = "Is it?";
    body["questions"]["second"]["criteria"]["false"] = "Definitely not";
    const DecideRequest request = parse_decide_request(body, 300);
    check(request.questions.size() == 2 && request.questions[0].id == "first" &&
              request.questions[1].id == "second",
          "questions keep the order they were written");
    const DecideQuestion& choice = request.questions[0];
    check(choice.options.size() == 2 && choice.options[0].name == "zeta" &&
              choice.options[1].name == "alpha" && choice.options[1].description == "alpha",
          "choice options keep written order and a null description is the option's name");
    const DecideQuestion& noul = request.questions[1];
    check(noul.options[0].name == "true" && noul.options[0].description == "Yes" &&
              noul.options[1].description == "Definitely not",
          "noul is true then false, with defaults for the omitted description");
}

void test_messages() {
    Json body;
    body["state"]["order"]             = 149.0;
    body["state"]["note"]              = "gift";
    body["questions"]["q"]["type"]     = "choice";
    body["questions"]["q"]["instructions"] = "Which queue?";
    body["questions"]["q"]["criteria"]["payouts"] = "Money leaving";
    body["questions"]["q"]["criteria"]["billing"] = "Money coming in";
    const DecideRequest request = parse_decide_request(body, 300);
    const std::vector<DecideLabel> labels{{"A", 10}, {"B", 11}};
    const std::vector<ChatTurn> messages = decide_messages(request, request.questions[0], labels);
    check(messages.size() == 2 && messages[0].role == infernix::ChatRole::System &&
              messages[1].role == infernix::ChatRole::User,
          "a JSON state is asked as a system and a user turn");
    check(messages[0].content.size() == 1 &&
              messages[0].content[0].text ==
                  std::string(kDecideSystemInstruction) +
                      "\n\n{\"evidence\":{\"order\":149.0,\"note\":\"gift\"}}",
          "the evidence follows the instruction in the caller's key order");
    check(messages[1].content.size() == 1 &&
              messages[1].content[0].text ==
                  "{\"criterion\":\"Which queue?\",\"options\":[{\"description\":\"Money "
                  "leaving\",\"letter\":\"A\"},{\"description\":\"Money coming "
                  "in\",\"letter\":\"B\"}]}",
          "the user turn is the measured criterion and options payload");

    // Reordering the options is a different prompt.
    Json swapped = body;
    swapped["questions"]["q"]["criteria"] = Json::object();
    swapped["questions"]["q"]["criteria"]["billing"] = "Money coming in";
    swapped["questions"]["q"]["criteria"]["payouts"] = "Money leaving";
    const DecideRequest reordered = parse_decide_request(swapped, 300);
    check(decide_messages(reordered, reordered.questions[0], labels)[1].content[0].text !=
              messages[1].content[0].text,
          "option order is part of the prompt");

    Json image;
    image["state"] = Json::array(
        {Json{{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AAAA"}}}},
         Json{{"type", "text"}, {"text", "Screenshot"}}});
    image["questions"]["q"] = body["questions"]["q"];
    const DecideRequest parts = parse_decide_request(image, 300);
    check(parts.has_parts() && parts.parts.size() == 2 &&
              parts.parts[0].kind == ContentKind::Image,
          "content-part evidence is read as parts");
    const std::vector<ChatTurn> visual = decide_messages(parts, parts.questions[0], labels);
    check(visual[0].content[0].text == kDecideSystemInstruction && visual[1].content.size() == 3 &&
              visual[1].content[0].kind == ContentKind::Image,
          "content-part evidence leads the user turn, the system turn is the instruction alone");
    check(visual[1].content[1].cache_boundary_after.has_value() &&
              !visual[1].content[2].cache_boundary_after.has_value(),
          "the evidence parts end at a prefix-cache boundary, the question does not");

    Json records;
    records["state"] = Json::array({Json{{"id", 1}, {"title", "x"}}});
    records["questions"]["q"] = body["questions"]["q"];
    check(!parse_decide_request(records, 300).has_parts(),
          "an array of objects without a type is evidence, not content parts");
}

void test_answers() {
    DecideQuestion score;
    score.kind    = DecideKind::Score;
    score.options = {{"0", "calm"}, {"1", "annoyed"}, {"2", "angry"}};
    // Probabilities {0.05, 0.3, 0.65} with half of the vocabulary's mass on these labels.
    const std::vector<float> logprobs{static_cast<float>(std::log(0.025)),
                                      static_cast<float>(std::log(0.15)),
                                      static_cast<float>(std::log(0.325))};
    const Json answer = decide_answer(score, logprobs);
    check(near(answer.at("score").get<double>(), 1.6, 1e-6), "score is the expected level (1.6)");
    check(near(answer.at("answer_mass").get<double>(), 0.5, 1e-6),
          "answer mass is the labels' share of the whole distribution");
    check(near(answer.at("probabilities").at("2").get<double>(), 0.65, 1e-6),
          "probabilities are renormalized over the declared options");
    check(answer.at("legend").at("1").get<std::string>() == "annoyed", "legend maps levels");

    const std::vector<float> adjacent{static_cast<float>(std::log(0.5)),
                                      static_cast<float>(std::log(0.5)), -INFINITY};
    const std::vector<float> extremes{static_cast<float>(std::log(0.5)), -INFINITY,
                                      static_cast<float>(std::log(0.5))};
    check(decide_answer(score, adjacent).at("confidence").get<double>() >
              decide_answer(score, extremes).at("confidence").get<double>(),
          "a split between adjacent levels is more confident than one between the extremes");
    check(near(decide_answer(score, extremes).at("confidence").get<double>(), 0.0, 1e-6),
          "an even split between the extremes has no confidence");

    DecideQuestion choice;
    choice.kind    = DecideKind::Choice;
    choice.options = {{"payouts", "a"}, {"billing", "b"}};
    const Json picked = decide_answer(choice, std::vector<float>{-2.0F, -0.5F});
    check(picked.at("choice").get<std::string>() == "billing", "choice is the most probable");
    check(near(picked.at("confidence").get<double>(),
               std::exp(-0.5) / (std::exp(-2.0) + std::exp(-0.5)), 1e-6),
          "choice confidence is the top restricted probability");

    DecideQuestion noul;
    noul.kind    = DecideKind::Noul;
    noul.options = {{"true", "Yes"}, {"false", "No"}};
    check(near(decide_answer(noul, std::vector<float>{static_cast<float>(std::log(0.3)),
                                                      static_cast<float>(std::log(0.1))})
                   .at("noul")
                   .get<double>(),
               0.75, 1e-6),
          "noul is the true option's restricted probability");
    check(decide_answer_mass(std::vector<float>{-INFINITY, -INFINITY}) == 0.0,
          "labels with no probability have no mass");
}

infernix::TokenId token(char c) { return static_cast<infernix::TokenId>(c) + 1000; }

// The Qwen tokenizer's merges that the scalar schedule depends on: `":` and `":-` are one token
// each (a sign after a colon merges into it).
constexpr infernix::TokenId kColon        = 5000;
constexpr infernix::TokenId kColonNegated = 5001;

// One token per character, offset so a digit's id is never its value, except the merged colons.
std::vector<infernix::TokenId> qwen_like(std::string_view text) {
    std::vector<infernix::TokenId> ids;
    for (std::size_t i = 0; i < text.size();) {
        if (text.substr(i, 3) == "\":-") {
            ids.push_back(kColonNegated);
            i += 3;
        } else if (text.substr(i, 2) == "\":") {
            ids.push_back(kColon);
            i += 2;
        } else {
            ids.push_back(token(text[i]));
            ++i;
        }
    }
    return ids;
}

// A run whose step i drew the i-th token of `text` with the given probability of the drawn token.
std::vector<infernix::ConstrainedDraw> run_of(const DecideSchedule& schedule, std::string_view text,
                                            float probability = 0.9F) {
    const std::vector<infernix::TokenId> tokens = qwen_like(text);
    std::vector<infernix::ConstrainedDraw> run;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto& step = schedule.constraint.steps[i];
        infernix::ConstrainedDraw draw{.token = tokens[i], .probabilities = {}, .mass = 0.99F};
        draw.probabilities.assign(step.size(), 0.0F);
        const auto slot = std::find(step.begin(), step.end(), draw.token) - step.begin();
        if (slot < static_cast<std::ptrdiff_t>(step.size())) {
            draw.probabilities[static_cast<std::size_t>(slot)] = step.size() == 1 ? 1.0F
                                                                                   : probability;
        }
        run.push_back(std::move(draw));
    }
    return run;
}

DecideQuestion generated(DecideKind kind, std::uint32_t digits) {
    DecideQuestion question;
    question.id     = "q";
    question.kind   = kind;
    question.digits = digits;
    return question;
}

void test_generated_validation() {
    Json number;
    number["state"]                          = "s";
    number["questions"]["q"]["type"]         = "number";
    number["questions"]["q"]["instructions"] = "How many?";
    const DecideRequest parsed               = parse_decide_request(number, 300);
    check(parsed.questions[0].kind == DecideKind::Number && parsed.questions[0].digits == 3,
          "a number defaults to the measured three-digit field");
    Json scalar                      = number;
    scalar["questions"]["q"]["type"] = "scalar";
    check(parse_decide_request(scalar, 300).questions[0].digits == 8,
          "a scalar defaults to an eight-digit ceiling");
    Json wide                        = number;
    wide["questions"]["q"]["digits"] = 7;
    check(api_code(wide) == "digits_out_of_range", "a number field wider than six is refused");
    scalar["questions"]["q"]["digits"] = 15;
    check(api_code(scalar).empty(), "a scalar ceiling of fifteen is served");
    Json criteria                        = number;
    criteria["questions"]["q"]["criteria"] = Json::object({{"a", nullptr}});
    check(api_code(criteria) == "criteria_unsupported", "criteria on a generated type are refused");
    Json digits_on_choice                      = choice_body(2);
    digits_on_choice["questions"]["q"]["digits"] = 3;
    check(api_code(digits_on_choice) == "digits_unsupported", "digits on a readout are refused");
    Json point                      = number;
    point["questions"]["q"]["type"] = "point";
    check(api_code(point) == "spatial_needs_one_image", "a point over text is refused");
    point["state"] = Json::array(
        {Json{{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AAAA"}}}}});
    check(api_code(point).empty(), "a point over one image is served");
}

void test_generated_prompts() {
    // ignis's measured pointing prompt, byte for byte at three digits.
    Json body;
    body["state"] = Json::array(
        {Json{{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AAAA"}}}}});
    body["questions"]["q"]["type"]         = "point";
    body["questions"]["q"]["instructions"] = "the Save button";
    const DecideRequest request            = parse_decide_request(body, 300);
    const std::vector<ChatTurn> messages   = decide_messages(request, request.questions[0], {});
    check(messages[0].content[0].text ==
              "You are given a screenshot and an instruction. Answer with the position on the "
              "screen the instruction refers to. Use a 0-999 scale on each axis, where x=0 is the "
              "left edge, x=999 the right edge, y=0 the top edge and y=999 the bottom edge. Reply "
              "with only a JSON object of the form {\"x\":NNN,\"y\":NNN}, three digits each.",
          "the point prompt is ignis's measured text");
    check(messages[1].content.back().text == "{\"instruction\":\"the Save button\"}",
          "a generated question's user turn is its instruction");
}

void test_schedules_and_answers() {
    const DecideEncoder encode = qwen_like;
    const DecideSchedule number = decide_schedule(generated(DecideKind::Number, 3), encode);
    // {"value": is eight forced tokens ({, ", v, a, l, u, e, ":), then three digit steps.
    check(number.constraint.steps.size() == 11 && number.constraint.steps[0].size() == 1 &&
              number.constraint.steps[8].size() == 10 && number.axes.size() == 1 &&
              number.axes[0].begin == 8 && number.axes[0].end == 11,
          "a number forces its opening literal and then reads its field");
    const Json padded = decide_generated_answer(generated(DecideKind::Number, 3), number,
                                                run_of(number, "{\"value\":047", 0.9F), {});
    check(padded.at("number").get<std::uint64_t>() == 47, "the number reads its digits");
    check(near(padded.at("uncertainty").get<double>(), 0.1 * 10 + 0.1 * 1, 1e-5),
          "a padding zero carries no uncertainty");
    check(decide_generated_answer(generated(DecideKind::Number, 3), number,
                                  run_of(number, "{\"value\":04"), {})
                  .at("code") == "run_cut_short",
          "a run the engine cut short is an error, not a number");

    const DecideSchedule point = decide_schedule(generated(DecideKind::Point, 3), encode);
    const std::vector<infernix::MediaGeometry> media{{1600, 900}};
    const Json located = decide_generated_answer(generated(DecideKind::Point, 3), point,
                                                 run_of(point, "{\"x\":500,\"y\":999"), media);
    check(located.at("type") == "point" && located.at("normalized").at("x") == 500 &&
              located.at("pixels").at("x") == 801 && located.at("pixels").at("y") == 900,
          "a point is rescaled per axis onto the image's own pixels");

    const DecideSchedule scalar = decide_schedule(generated(DecideKind::Scalar, 4), encode);
    // {"value is seven forced tokens; then the colon (positive or merged with the sign), a first
    // digit, and room for the other three digits, the point and the brace, plus one.
    const auto& steps = scalar.constraint.steps;
    const auto permits = [](const std::vector<infernix::TokenId>& step, infernix::TokenId id) {
        return std::find(step.begin(), step.end(), id) != step.end();
    };
    check(scalar.terminator == token('}') && steps.size() == 7 + 1 + 1 + 5 &&
              steps[7] == std::vector<infernix::TokenId>{kColon, kColonNegated} &&
              steps[8].size() == 10 && !permits(steps[8], token('-')) &&
              !permits(steps[8], token('}')) && permits(steps[9], token('}')) &&
              permits(steps[9], token('.')),
          "a scalar chooses its colon with or without the merged sign, opens with a digit and "
          "leaves room to close itself");
    check(std::none_of(steps.begin(), steps.end(),
                       [&](const auto& step) { return permits(step, token('-')); }),
          "a bare sign token, which the tokenizer never writes after a colon, is never offered");
    const auto scalar_answer = [&](std::string_view text) {
        return decide_generated_answer(generated(DecideKind::Scalar, 4), scalar,
                                       run_of(scalar, text, 0.9F), {});
    };
    const Json half = scalar_answer("{\"value\":3.5}");
    check(half.at("value").get<double>() == 3.5 && half.at("text") == "3.5" &&
              near(half.at("uncertainty").get<double>(), 0.1 * 1 + 0.1 * 0.1, 1e-5),
          "a scalar weights each digit by its place once the point is seen");
    check(scalar_answer("{\"value\":-0.25}").at("value").get<double>() == -0.25,
          "a negative decimal round-trips");
    check(scalar_answer("{\"value\":3.}").at("code") == "malformed_scalar",
          "a trailing point is refused, never read as 3");
    check(scalar_answer("{\"value\":12345}").at("code") == "too_many_digits",
          "more digits than the ceiling is its own error");
    check(scalar_answer("{\"value\":12").at("code") == "run_cut_short",
          "an unclosed run that did not spend its schedule is cut short");
}

} // namespace

int main() {
    test_generated_validation();
    test_generated_prompts();
    test_schedules_and_answers();
    test_alphabet();
    test_validation();
    test_options_and_order();
    test_messages();
    test_answers();
    if (failures != 0) {
        std::cerr << failures << " decide schema check(s) failed\n";
        return 1;
    }
    std::cout << "decide schema tests passed\n";
    return 0;
}
