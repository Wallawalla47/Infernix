#pragma once

// POST /v1/decide: typed decisions read from the model's next-token distribution after one prompt
// prefill, with nothing generated for the answer. The wire shape is TypeSafe Jev's
// POST /v1/systemone as served by gpillon/ignis (ADR 0034); the prompt is SemIf's direct prompt,
// byte for byte, because every published accuracy figure was measured with it.
//
// Each question names its options by single-token labels; the answer is the label tokens'
// log-probabilities at the prompt's last position, renormalized over the declared options. Their
// total probability under the whole vocabulary is the answer mass, the evidence that the model
// answered inside the declared set at all.

#include "serve/request.h"
#include "serve/request_json.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

inline constexpr std::string_view kDecideSystemInstruction =
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning.";

// Options one question may declare: the width ignis measured without decay (answer mass p50
// >= 0.996 from 8 to 256 options), not a property of the model.
inline constexpr std::size_t kDecideMaximumOptions = 256;

struct DecideLabel {
    std::string text;
    ninfer::TokenId token = 0;
};

// Candidate labels A-Z, a-z, 0-9, then the uppercase bigrams AA-ZZ, in that order. A label is
// admitted only when it encodes to exactly one token that no earlier label encodes to: a label
// that splits (BQ is B then Q) would read another option's logit.
[[nodiscard]] std::vector<DecideLabel>
build_decide_alphabet(const std::function<std::vector<ninfer::TokenId>(std::string_view)>& encode);

enum class DecideKind : std::uint8_t {
    // Read at one position (nothing generated):
    // Yes/no: the probability of the true option.
    Noul,
    // One option of a declared set.
    Choice,
    // The expected level index over ordered levels.
    Score,
    // Generated under a token constraint, one restricted step per token:
    // A whole number in a field of `digits` digits, right-aligned.
    Number,
    // A number of at most `digits` digits that may be negative or have a decimal part and closes
    // its own object.
    Scalar,
    // x and y on a 0-(10^digits - 1) scale over the submitted image, answered in its pixels.
    Point,
    // x0, y0, x1, y1 likewise.
    Box,
};

[[nodiscard]] constexpr bool decide_generates(DecideKind kind) noexcept {
    return kind == DecideKind::Number || kind == DecideKind::Scalar ||
           kind == DecideKind::Point || kind == DecideKind::Box;
}

// Digit widths the generated primitives accept: a field for number/point/box (whose default is
// the width ignis measured), a ceiling for scalar (whose default covers everyday quantities and
// keeps the schedule short).
inline constexpr std::uint32_t kDecideFieldDigitsMax   = 6;
inline constexpr std::uint32_t kDecideFieldDigitsDefault = 3;
inline constexpr std::uint32_t kDecideScalarDigitsMax     = 15;
inline constexpr std::uint32_t kDecideScalarDigitsDefault = 8;

struct DecideOption {
    // What the answer names it by: true/false, the caller's key, or the level index.
    std::string name;
    std::string description;
};

struct DecideQuestion {
    std::string id;
    DecideKind kind = DecideKind::Choice;
    // A string, object or array, serialized into the prompt in the order it was written.
    RequestJson instructions;
    std::vector<DecideOption> options;
    // Generated primitives: the field width (number/point/box) or ceiling (scalar).
    std::uint32_t digits = 0;
};

struct DecideRequest {
    std::string model;
    // JSON evidence (string, object or array), used when `parts` is empty.
    RequestJson state;
    // OpenAI content parts: evidence that may include images.
    std::vector<ContentPart> parts;
    std::vector<DecideQuestion> questions;

    [[nodiscard]] bool has_parts() const noexcept { return !parts.empty(); }
};

// Validates the whole body or throws ApiException (400): no question is asked unless every one is
// well formed. `alphabet_size` is the loaded tokenizer's label count; a question with more options
// than it can name is refused rather than served with two options sharing a token.
[[nodiscard]] DecideRequest parse_decide_request(const RequestJson& body,
                                                 std::size_t alphabet_size);

// The chat turns one question is asked as. JSON evidence rides in the system message after the
// instruction, so every question over one state shares it as a prompt prefix; content parts stay
// in the user turn, ahead of the question.
[[nodiscard]] std::vector<ChatTurn> decide_messages(const DecideRequest& request,
                                                    const DecideQuestion& question,
                                                    std::span<const DecideLabel> labels);

// The answer object for one question from its options' log-probabilities, in option order.
[[nodiscard]] RequestJson decide_answer(const DecideQuestion& question,
                                        std::span<const float> logprobs);

// exp(logsumexp(logprobs)): the declared options' share of the whole next-token distribution.
[[nodiscard]] double decide_answer_mass(std::span<const float> logprobs);

using DecideEncoder = std::function<std::vector<ninfer::TokenId>(std::string_view)>;

// One axis of a generated primitive: its name and the output steps holding its digits.
struct DecideAxis {
    std::string name;
    std::size_t begin = 0;
    std::size_t end   = 0;
};

// The token schedule a generated question runs under and how its output is read. The opening
// literal ({"x": and the like) is forced output, so the model sees the text ignis prefilled.
struct DecideSchedule {
    ninfer::TokenConstraint constraint;
    std::vector<DecideAxis> axes;
    std::array<ninfer::TokenId, 10> digit_tokens{};
    // Scalar only: the decimal point, the closing brace that ends the run, and the two openers
    // of the value: `":` before a positive number and `":-`, the one token the tokenizer writes
    // for a colon followed by a sign.
    ninfer::TokenId point         = -1;
    ninfer::TokenId terminator    = -1;
    ninfer::TokenId colon         = -1;
    ninfer::TokenId colon_negated = -1;
};

// The schedule for a generated question, from the loaded tokenizer. Every digit and structural
// token must be one token; otherwise throws ApiException naming what the tokenizer cannot do.
[[nodiscard]] DecideSchedule decide_schedule(const DecideQuestion& question,
                                             const DecideEncoder& encode);

// The answer of a generated question from its run (one draw per generated token, in order).
// `media` is the submitted media's display sizes, for point and box pixels.
[[nodiscard]] RequestJson decide_generated_answer(const DecideQuestion& question,
                                                  const DecideSchedule& schedule,
                                                  std::span<const ninfer::ConstrainedDraw> run,
                                                  std::span<const ninfer::MediaGeometry> media);

} // namespace ninfer::serve
