#include "text/grammar.h"
#include "text/json_schema.h"
#include "text/unicode.h"

#include <xgrammar/xgrammar.h>
#include "grammar_functor.h"
#include "grammar_impl.h"
#include "regex_converter.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace infernix::text {
namespace {
class WorkTimer {
public:
    WorkTimer(bool enabled, double& seconds) : seconds_(enabled ? &seconds : nullptr) {
        if (seconds_) started_ = std::chrono::steady_clock::now();
    }

    ~WorkTimer() {
        if (seconds_)
            *seconds_ +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
    }
private:
    double* seconds_;
    std::chrono::steady_clock::time_point started_;
};

ConstraintCacheAccess cache_access(xgrammar::CompilationCacheAccess access) {
    switch (access) {
    case xgrammar::CompilationCacheAccess::kHit:
        return ConstraintCacheAccess::Hit;
    case xgrammar::CompilationCacheAccess::kBuilt:
        return ConstraintCacheAccess::Built;
    case xgrammar::CompilationCacheAccess::kWaited:
        return ConstraintCacheAccess::Waited;
    }
    throw std::logic_error("invalid compilation cache observation");
}

void validate_utf8(std::string_view value) {
    for (std::size_t offset = 0; offset < value.size();) {
        offset += unicode_internal::utf8_codepoint_at(value, offset, "output constraint").length;
    }
}

// Text publication consumes Unicode scalar values, including in negated regex classes.
class ScalarRegex final : public xgrammar::GrammarMutator {
    int32_t VisitCharacterClass(const GrammarExpr& expression) final {
        using Range = xgrammar::GrammarBuilder::CharacterClassElement;
        std::vector<Range> ranges;
        const bool negative = expression[0] != 0;
        for (int i = 1; i < expression.size(); i += 2) {
            const int lo = expression[i], hi = expression[i + 1];
            if (negative) {
                ranges.push_back({lo, hi});
            } else {
                if (lo <= 0xd7ff) ranges.push_back({lo, std::min(hi, 0xd7ff)});
                if (hi >= 0xe000) ranges.push_back({std::max(lo, 0xe000), hi});
            }
        }
        if (negative) ranges.push_back({0xd800, 0xdfff});
        std::sort(ranges.begin(), ranges.end(),
                  [](const auto& a, const auto& b) { return a.lower < b.lower; });
        std::vector<Range> merged;
        for (const auto& range : ranges) {
            if (!merged.empty() && range.lower <= merged.back().upper + 1)
                merged.back().upper = std::max(merged.back().upper, range.upper);
            else
                merged.push_back(range);
        }
        return builder_->AddCharacterClass(merged, negative);
    }

    int32_t VisitCharacterClassStar(const GrammarExpr& expression) final {
        return builder_->AddRepeatFromExpr("scalar", VisitCharacterClass(expression), 0, -1);
    }
};

xgrammar::Grammar choice_grammar(const std::vector<std::string>& choices) {
    if (choices.empty()) throw std::invalid_argument("choice requires at least one string");
    xgrammar::GrammarBuilder builder;
    std::vector<int32_t> alternatives;
    alternatives.reserve(choices.size());
    for (const auto& value : choices) {
        validate_utf8(value);
        alternatives.push_back(builder.AddByteString(value));
    }
    return xgrammar::GrammarNormalizer::Apply(
        builder.Get(builder.AddRuleWithHint("root", builder.AddChoices(alternatives))));
}

std::string quoted(unsigned char c) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result   = "\"\\x00\"";
    result[3]            = hex[c >> 4];
    result[4]            = hex[c & 15];
    return result;
}

// Free reasoning is any token but the close token (and the withheld token, if any); the close token
// then requires the separator. Its spelling in ordinary tokens stays reasoning, as in the output
// decoder. Concatenation keeps the user's rule names in their own namespace.
xgrammar::Grammar reasoning_prefix(const ReasoningFraming& framing) {
    const std::string close = std::to_string(*framing.close_token);
    std::string excluded    = close;
    if (framing.withheld_token) { excluded += ", " + std::to_string(*framing.withheld_token); }
    std::string source = "root ::= reasoning\nreasoning ::= ExcludeToken(" + excluded +
                         ") reasoning | Token(" + close + ")";
    for (unsigned char c : framing.separator) { source += " " + quoted(c); }
    return xgrammar::Grammar::FromEBNF(source + "\n");
}

// The compile-cache identity of a reasoning framing.
std::string framing_key(const ReasoningFraming& framing) {
    if (!framing.close_token) { return "-:"; }
    return std::to_string(*framing.close_token) + ":" + std::to_string(framing.separator.size()) +
           ":" + std::string(framing.separator) + ":" +
           (framing.withheld_token ? std::to_string(*framing.withheld_token) : "-");
}

void validate(const xgrammar::Grammar& grammar, const std::vector<std::string>& vocab,
              const std::vector<std::int32_t>& eos) {
    for (int i = 0; i < grammar->NumRules(); ++i) {
        const auto& rule = grammar->GetRule(i);
        if (rule.max_tokens >= 0 || rule.max_chars >= 0 || !rule.capture_name.empty() ||
            rule.is_lazy || rule.temperature || grammar->GetSuffixStopInfo(i)) {
            throw std::invalid_argument(
                "GBNF rules cannot set budgets, captures, lazy matching or temperature");
        }
    }
    using Type = xgrammar::Grammar::Impl::GrammarExprType;
    for (int i = 0; i < grammar->NumGrammarExprs(); ++i) {
        const auto expression = grammar->GetGrammarExpr(i);
        if (expression.type == Type::kTagDispatch || expression.type == Type::kTokenTagDispatch ||
            expression.type == Type::kRegex || expression.type == Type::kSubstring) {
            throw std::invalid_argument("grammar uses an extension outside the GBNF contract");
        }
        if (expression.type == Type::kToken || expression.type == Type::kExcludeToken) {
            for (int id : expression) {
                if (id < 0 || static_cast<std::size_t>(id) >= vocab.size() || vocab[id].empty() ||
                    std::find(eos.begin(), eos.end(), id) != eos.end()) {
                    throw std::invalid_argument(
                        "GBNF token references must name ordinary vocabulary tokens");
                }
            }
        }
    }
}
} // namespace

RequestErrorKind constraint_error_kind(OutputConstraintKind kind) {
    switch (kind) {
    case OutputConstraintKind::Grammar:
        return RequestErrorKind::InvalidGrammar;
    case OutputConstraintKind::Choice:
        return RequestErrorKind::InvalidChoice;
    case OutputConstraintKind::Regex:
        return RequestErrorKind::InvalidRegex;
    case OutputConstraintKind::JsonObject:
    case OutputConstraintKind::JsonSchema:
        return RequestErrorKind::InvalidJsonSchema;
    }
    throw std::invalid_argument("unknown output constraint kind");
}

namespace {
std::shared_ptr<const std::vector<std::int32_t>> tokens_containing(const std::vector<std::string>& vocab,
                                                                  char byte) {
    auto ids = std::make_shared<std::vector<std::int32_t>>();
    for (std::size_t id = 0; id < vocab.size(); ++id) {
        if (vocab[id].find(byte) != std::string::npos) { ids->push_back(static_cast<std::int32_t>(id)); }
    }
    return ids;
}
} // namespace

// A deferred compile's result, made on a compiler worker.
struct DeferredGrammar {
    xgrammar::CompiledGrammar grammar;
    double seconds;
    ConstraintCacheAccess cache;
};

class GrammarCompiler::Impl {
public:
    // The workers deferred compiles run on: the vendored compiler admits two cold compiles at once.
    static constexpr std::size_t kWorkers = 2;

    Impl(std::vector<std::string> vocab_, std::vector<std::int32_t> eos_, std::size_t bytes)
        : vocab(std::make_shared<const std::vector<std::string>>(std::move(vocab_))),
          eos(std::move(eos_)), tag_ends(tokens_containing(*vocab, '>')),
          tokenizer(*vocab, xgrammar::VocabType::RAW, static_cast<int>(vocab->size()), eos),
          compiler(tokenizer, 1, true, static_cast<std::int64_t>(bytes)) {}

    ~Impl() {
        {
            std::lock_guard lock(jobs_mutex);
            stopping = true;
        }
        jobs_ready.notify_all();
        for (auto& worker : workers) { worker.join(); }
        // Compiles still queued are destroyed unrun: their sessions see a broken promise.
    }

    // Compiles `grammar` (already built and validated) under `key` on a worker. A compile error
    // the library reports becomes a RequestError of `error_kind` when one is given, as compile()'s
    // own path reports it.
    std::shared_future<DeferredGrammar> defer(std::string key, xgrammar::Grammar grammar,
                                              std::optional<RequestErrorKind> error_kind) {
        std::packaged_task<DeferredGrammar()> job(
            [this, key = std::move(key), grammar = std::move(grammar), error_kind]() {
                const auto started = std::chrono::steady_clock::now();
                xgrammar::CompilationCacheAccess access;
                auto compiled = [&] {
                    try {
                        return compiler.CompileCachedGrammar(key, [&] { return grammar; }, &access);
                    } catch (const xgrammar::LogFatalError& error) {
                        if (!error_kind) { throw; }
                        throw RequestError(*error_kind, error.what());
                    }
                }();
                return DeferredGrammar{
                    std::move(compiled),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                    cache_access(access)};
            });
        auto result = job.get_future().share();
        {
            std::lock_guard lock(jobs_mutex);
            if (stopping) { throw std::logic_error("grammar compiler is stopping"); }
            jobs.push_back(std::move(job));
            if (workers.size() < kWorkers && jobs.size() > idle) {
                workers.emplace_back([this] { work(); });
            }
        }
        jobs_ready.notify_one();
        return result;
    }

    std::shared_ptr<const std::vector<std::string>> vocab;
    std::vector<std::int32_t> eos;
    std::shared_ptr<const std::vector<std::int32_t>> tag_ends;
    xgrammar::TokenizerInfo tokenizer;
    xgrammar::GrammarCompiler compiler;

private:
    void work() {
        std::unique_lock lock(jobs_mutex);
        for (;;) {
            ++idle;
            jobs_ready.wait(lock, [&] { return stopping || !jobs.empty(); });
            --idle;
            if (stopping) { return; }
            auto job = std::move(jobs.front());
            jobs.pop_front();
            lock.unlock();
            job(); // a compile error is stored in the job's future
            lock.lock();
        }
    }

    std::mutex jobs_mutex;
    std::condition_variable jobs_ready;
    std::deque<std::packaged_task<DeferredGrammar()>> jobs;
    std::size_t idle = 0;
    bool stopping    = false;
    std::vector<std::thread> workers;
};

class GrammarSession::Impl {
public:
    Impl(const xgrammar::CompiledGrammar& grammar, int vocab)
        : matcher(std::in_place, grammar), vocabulary(vocab), words((vocab + 31) / 32) {}

    Impl(std::shared_future<DeferredGrammar> compile, int vocab)
        : deferred(std::move(compile)), vocabulary(vocab), words((vocab + 31) / 32) {}

    std::optional<xgrammar::GrammarMatcher> matcher;
    // A deferred compile until the matcher exists (or the compile failed).
    std::shared_future<DeferredGrammar> deferred;
    std::exception_ptr failure;
    int vocabulary;
    std::size_t words;
    int tentative = 0; // tokens the matcher accepted since the last confirm (its rollback)
    std::unique_ptr<GrammarStateConstraint> companion;
    ConstraintObservation observed;

    [[nodiscard]] bool pending() const noexcept {
        return !matcher && !failure && deferred.valid() &&
               deferred.wait_for(std::chrono::seconds(0)) != std::future_status::ready;
    }

    // The matcher, made from the deferred compile at first use (waiting for it); null if that
    // compile failed.
    xgrammar::GrammarMatcher* ready() {
        if (matcher) { return &*matcher; }
        if (failure || !deferred.valid()) { return nullptr; }
        const auto started = std::chrono::steady_clock::now();
        const bool waits   = pending();
        try {
            const DeferredGrammar& done = deferred.get();
            matcher.emplace(done.grammar);
            observed.cache = done.cache;
            if (observed.timings_collected) { observed.compile_seconds = done.seconds; }
        } catch (...) { failure = std::current_exception(); }
        if (waits && observed.timings_collected) {
            observed.wait_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        }
        deferred = {};
        return matcher ? &*matcher : nullptr;
    }

    // Advances by one token: the matcher, unless the companion takes the token as text, then the
    // companion. Refused (nothing advanced): nullopt; else whether the matcher advanced.
    std::optional<bool> advance(std::int32_t token) {
        const bool literal = companion && companion->literal(token);
        if (!literal && (matcher->IsTerminated() || !matcher->AcceptToken(token))) { return std::nullopt; }
        if (companion && !companion->accept(token)) {
            if (!literal) { matcher->Rollback(1); }
            return std::nullopt;
        }
        return !literal;
    }

    void fill(std::span<std::uint32_t> target) {
        if (matcher->IsTerminated()) {
            std::fill(target.begin(), target.end(), 0);
            return;
        }
        std::int64_t shape[] = {1, static_cast<std::int64_t>(words)};
        DLTensor tensor{target.data(), {kDLCPU, 0}, 2, {kDLInt, 32, 1}, shape, nullptr, 0};
        matcher->FillNextTokenBitmask(&tensor);
        if (vocabulary % 32) { target.back() &= (1u << (vocabulary % 32)) - 1u; }
        if (companion) { companion->filter(target); }
    }
};

GrammarCompiler::GrammarCompiler(std::vector<std::string> vocab, std::vector<std::int32_t> eos,
                                 std::size_t bytes)
    : impl_(std::make_unique<Impl>(std::move(vocab), std::move(eos), bytes)) {}

GrammarCompiler::~GrammarCompiler() = default;

const std::shared_ptr<const std::vector<std::string>>& GrammarCompiler::vocabulary() const noexcept {
    return impl_->vocab;
}

const std::shared_ptr<const std::vector<std::int32_t>>& GrammarCompiler::tag_end_tokens() const noexcept {
    return impl_->tag_ends;
}

std::unique_ptr<GrammarSession>
GrammarCompiler::compile_model(std::string_view identity,
                               const std::function<xgrammar::Grammar()>& build,
                               const ReasoningFraming& reasoning,
                               std::span<const std::int32_t> continuation,
                               std::unique_ptr<GrammarStateConstraint> companion, Compile mode) {
    const std::string key = "model:" + framing_key(reasoning) + std::string(identity);
    const auto framed     = [&] {
        auto grammar = build();
        return reasoning.close_token
                   ? xgrammar::Grammar::Concat({reasoning_prefix(reasoning), grammar})
                   : grammar;
    };
    const int vocabulary = static_cast<int>(impl_->vocab->size());
    std::unique_ptr<GrammarSession::Impl> session;
    if (mode == Compile::Deferred && continuation.empty() && !impl_->compiler.HasCachedGrammar(key)) {
        session = std::make_unique<GrammarSession::Impl>(impl_->defer(key, framed(), std::nullopt),
                                                         vocabulary);
        session->observed.cache = ConstraintCacheAccess::Built;
    } else {
        xgrammar::CompilationCacheAccess access;
        auto compiled = impl_->compiler.CompileCachedGrammar(key, framed, &access);
        session       = std::make_unique<GrammarSession::Impl>(compiled, vocabulary);
        session->observed.cache = cache_access(access);
    }
    session->companion = std::move(companion);
    for (const std::int32_t token : continuation) {
        if (!session->advance(token))
            throw RequestError(RequestErrorKind::InvalidToolConstraint,
                               "assistant continuation is not a prefix of the tool grammar");
    }
    if (session->companion) { session->companion->confirm(); }
    return std::unique_ptr<GrammarSession>(new GrammarSession(std::move(session)));
}

std::unique_ptr<GrammarSession> GrammarCompiler::compile(const OutputConstraint& constraint,
                                                         const ReasoningFraming& reasoning,
                                                         std::string_view continuation,
                                                         Compile mode) {
    const auto error_kind = constraint_error_kind(constraint.kind);
    try {
        auto choices = constraint.choices;
        if (constraint.kind == OutputConstraintKind::Choice) {
            if (!constraint.source.empty())
                throw std::invalid_argument("choice uses literal alternatives, not source text");
            std::sort(choices.begin(), choices.end());
            choices.erase(std::unique(choices.begin(), choices.end()), choices.end());
        } else if (!choices.empty()) {
            throw std::invalid_argument("literal alternatives require a choice constraint");
        }
        std::string key = std::to_string(static_cast<int>(constraint.kind)) + ":" +
                          framing_key(reasoning) + constraint.source;
        for (const auto& value : choices) key += std::to_string(value.size()) + ":" + value;
        const auto build = [&] {
            auto grammar = [&]() -> xgrammar::Grammar {
                if (constraint.kind == OutputConstraintKind::Grammar) {
                    auto parsed = xgrammar::Grammar::FromEBNF(constraint.source, "root");
                    validate(parsed, *impl_->vocab, impl_->eos);
                    return parsed;
                } else if (constraint.kind == OutputConstraintKind::Choice) {
                    return choice_grammar(choices);
                } else if (constraint.kind == OutputConstraintKind::Regex) {
                    validate_utf8(constraint.source);
                    return xgrammar::GrammarNormalizer::Apply(
                        ScalarRegex().Apply(xgrammar::Grammar::FromRegex(
                            xgrammar::NormalizeRegexPattern(constraint.source, true))));
                } else {
                    return build_json_grammar(constraint);
                }
            }();
            if (reasoning.close_token)
                grammar = xgrammar::Grammar::Concat({reasoning_prefix(reasoning), grammar});
            return grammar;
        };
        const int vocabulary = static_cast<int>(impl_->vocab->size());
        if (mode == Compile::Deferred && continuation.empty() &&
            !impl_->compiler.HasCachedGrammar(key)) {
            auto session = std::make_unique<GrammarSession::Impl>(
                impl_->defer(key, build(), error_kind), vocabulary);
            session->observed.cache = ConstraintCacheAccess::Built;
            return std::unique_ptr<GrammarSession>(new GrammarSession(std::move(session)));
        }
        xgrammar::CompilationCacheAccess access;
        auto compiled = impl_->compiler.CompileCachedGrammar(key, build, &access);
        auto session  = std::make_unique<GrammarSession::Impl>(compiled, vocabulary);
        session->observed.cache = cache_access(access);
        if (!continuation.empty() && !session->matcher->AcceptString(std::string(continuation))) {
            throw std::invalid_argument("assistant continuation is not a prefix of the grammar");
        }
        return std::unique_ptr<GrammarSession>(new GrammarSession(std::move(session)));
    } catch (const xgrammar::JSONSchemaCompileError& error) {
        RequestErrorKind kind = RequestErrorKind::InvalidJsonSchema;
        if (error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema)
            kind = RequestErrorKind::UnsupportedJsonSchema;
        else if (error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema)
            kind = RequestErrorKind::UnsatisfiableJsonSchema;
        throw RequestError(kind,
                           std::string(error.what()) + " at " +
                               (error.pointer.empty() ? "/" : error.pointer),
                           error.pointer);
    } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(error_kind, error.what());
    } catch (const RequestError&) { throw; } catch (const std::invalid_argument& error) {
        throw RequestError(error_kind, error.what());
    }
}

GrammarSession::GrammarSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

GrammarSession::~GrammarSession()                                    = default;
GrammarSession::GrammarSession(GrammarSession&&) noexcept            = default;
GrammarSession& GrammarSession::operator=(GrammarSession&&) noexcept = default;

std::size_t GrammarSession::mask_words() const noexcept { return impl_->words; }

std::uint64_t GrammarSession::masks(std::span<const std::int32_t> drafts,
                                    std::span<std::uint32_t> words) {
    if (drafts.size() >= 64 || impl_->tentative ||
        words.size() != (drafts.size() + 1) * impl_->words) {
        throw std::logic_error("invalid grammar lookahead transaction");
    }
    // A deferred compile is waited for here, outside the mask work's time.
    xgrammar::GrammarMatcher* const matcher = impl_->ready();
    WorkTimer timer(impl_->observed.timings_collected, impl_->observed.mask_seconds);
    if (matcher == nullptr) {
        // The deferred compile failed: no position has a legal token (safe nonempty device masks).
        for (std::size_t position = 0; position <= drafts.size(); ++position) {
            auto mask = words.subspan(position * impl_->words, impl_->words);
            std::fill(mask.begin(), mask.end(), 0);
            mask[0] = 1;
        }
        return 1;
    }
    int advanced       = 0;
    bool reachable     = true;
    std::uint64_t dead = 0;
    try {
        for (std::size_t position = 0; position <= drafts.size(); ++position) {
            auto mask = words.subspan(position * impl_->words, impl_->words);
            if (reachable && !matcher->IsTerminated()) {
                impl_->fill(mask);
                ++impl_->observed.mask_positions;
                if (std::all_of(mask.begin(), mask.end(), [](auto word) { return word == 0; })) {
                    dead |= std::uint64_t{1} << position;
                    reachable = false;
                    mask[0]   = 1;
                }
            } else {
                std::fill(mask.begin(), mask.end(), 0);
                mask[0] = 1;
            }
            if (position == drafts.size()) { break; }
            // Rejected proposals are expected; advance only through the allowed token set.
            const auto draft = drafts[position];
            std::optional<bool> moved;
            if (reachable && !matcher->IsTerminated() && draft >= 0 && draft < impl_->vocabulary &&
                (mask[draft / 32] & (1u << (draft % 32))) != 0) {
                moved = impl_->advance(draft);
            }
            if (!moved) {
                reachable = false;
            } else if (*moved) {
                ++advanced;
            }
        }
    } catch (...) {
        if (advanced) { matcher->Rollback(advanced); }
        if (impl_->companion) { impl_->companion->discard(); }
        throw;
    }
    if (advanced) { matcher->Rollback(advanced); }
    // No token was tentative on entry, so the committed state is the one the lookahead began at.
    if (impl_->companion) { impl_->companion->discard(); }
    return dead;
}

void GrammarSession::accept(std::int32_t token) { accept(std::span(&token, 1)); }

void GrammarSession::accept(std::span<const std::int32_t> tokens) {
    if (impl_->ready() == nullptr) {
        throw std::logic_error("a token was selected for a grammar whose compile failed");
    }
    WorkTimer timer(impl_->observed.timings_collected, impl_->observed.matcher_seconds);
    for (const auto token : tokens) {
        const std::optional<bool> moved = impl_->advance(token);
        if (!moved) throw std::logic_error("selected token violates the request grammar");
        if (*moved) ++impl_->tentative;
    }
}

void GrammarSession::observe(bool timings, double prepare_seconds) noexcept {
    impl_->observed.timings_collected = timings;
    impl_->observed.prepare_seconds   = timings ? prepare_seconds : 0.0;
}

void GrammarSession::uploaded(std::size_t bytes) noexcept {
    impl_->observed.mask_upload_bytes += bytes;
}

ConstraintObservation GrammarSession::observation() const {
    if (impl_->tentative) throw std::logic_error("cannot publish tentative grammar state");
    // A finished deferred compile is taken up without waiting; one still running (a request that
    // ended before its first token) reports nothing generated.
    const xgrammar::GrammarMatcher* const matcher = impl_->pending() ? nullptr : impl_->ready();
    auto result       = impl_->observed;
    result.terminated = matcher != nullptr && matcher->IsTerminated();
    result.complete   = result.terminated || (matcher != nullptr && matcher->IsCompleted());
    return result;
}

bool GrammarSession::pending() const noexcept { return impl_->pending(); }

std::exception_ptr GrammarSession::failure() const noexcept { return impl_->failure; }

void GrammarSession::confirm() noexcept {
    impl_->tentative = 0;
    if (impl_->companion) { impl_->companion->confirm(); }
}

void GrammarSession::discard() {
    WorkTimer timer(impl_->observed.timings_collected, impl_->observed.matcher_seconds);
    // Tokens are tentative only on a matcher that exists.
    if (impl_->tentative) { impl_->matcher->Rollback(impl_->tentative); }
    impl_->tentative = 0;
    if (impl_->companion) { impl_->companion->discard(); }
}
} // namespace infernix::text
