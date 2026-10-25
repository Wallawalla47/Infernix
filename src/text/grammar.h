#pragma once

#include "infernix/types.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xgrammar {
class Grammar;
}

namespace infernix::text {

[[nodiscard]] RequestErrorKind constraint_error_kind(OutputConstraintKind kind);

// The model's reasoning ahead of a constrained output: any tokens but the close token, then that
// token and the fixed separator bytes the output starts after. Only the close token's identity
// ends the reasoning, never its spelling in ordinary tokens. Without a close token the output
// starts at once.
struct ReasoningFraming {
    std::optional<std::int32_t> close_token;
    std::string_view separator;
};

// A request's companion to its grammar for a rule the context-free grammar cannot express (Qwen
// tool calls: a parameter name used once per call). It reads the tokens the matcher accepts and
// refuses tokens from its own state; the session keeps both in one transaction.
class GrammarStateConstraint {
public:
    virtual ~GrammarStateConstraint() = default;
    // Clears the bits of the tokens refused in the current state.
    virtual void filter(std::span<std::uint32_t> mask) = 0;
    // Advances by one token; false (state unchanged) when the token is refused.
    [[nodiscard]] virtual bool accept(std::int32_t token) = 0;
    // The current state becomes the committed one; discard() returns to it.
    virtual void confirm() noexcept = 0;
    virtual void discard() noexcept = 0;
};

// One immutable vocabulary/compiler per model; one transactional matcher per request.
class GrammarSession {
public:
    ~GrammarSession();
    GrammarSession(GrammarSession&&) noexcept;
    GrammarSession& operator=(GrammarSession&&) noexcept;
    // Returns one dead-end bit per position; at most 63 drafts plus the bonus position.
    [[nodiscard]] std::uint64_t masks(std::span<const std::int32_t> drafts,
                                      std::span<std::uint32_t> words);
    void accept(std::int32_t token);
    void accept(std::span<const std::int32_t> tokens);
    void confirm() noexcept;
    void discard();
    [[nodiscard]] std::size_t mask_words() const noexcept;
    void observe(bool timings, double prepare_seconds = 0.0) noexcept;
    void uploaded(std::size_t bytes) noexcept;
    [[nodiscard]] ConstraintObservation observation() const;

private:
    class Impl;
    explicit GrammarSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class GrammarCompiler;
};

class GrammarCompiler {
public:
    // Empty entries explicitly identify forbidden/special IDs. EOS is supplied separately.
    GrammarCompiler(std::vector<std::string> vocabulary, std::vector<std::int32_t> eos,
                    std::size_t cache_bytes);
    ~GrammarCompiler();
    // `continuation` is an assistant prefix the matcher accepts as bytes.
    [[nodiscard]] std::unique_ptr<GrammarSession> compile(const OutputConstraint& constraint,
                                                          const ReasoningFraming& reasoning,
                                                          std::string_view continuation);
    // Model-owned composition, built only on a miss in the same bounded compiler cache. A model
    // grammar may hold token-level terminals, so its continuation is accepted as tokens, by the
    // companion (if any) as well.
    [[nodiscard]] std::unique_ptr<GrammarSession>
    compile_model(std::string_view identity, const std::function<xgrammar::Grammar()>& build,
                  const ReasoningFraming& reasoning, std::span<const std::int32_t> continuation,
                  std::unique_ptr<GrammarStateConstraint> companion = {});
    // Each token's bytes (empty for special and invalid ids), shared with companions.
    [[nodiscard]] const std::shared_ptr<const std::vector<std::string>>& vocabulary() const noexcept;
    // The tokens whose bytes contain '>' (ascending): the only ones that can end a tag.
    [[nodiscard]] const std::shared_ptr<const std::vector<std::int32_t>>& tag_end_tokens() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infernix::text
