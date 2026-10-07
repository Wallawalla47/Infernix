// Constrained generation (TokenConstraint) on the real artifact, with and without DFlash2
// speculation. Every committed token belongs to its step's set, a one-token step forces its
// token, a stop token among the permitted tokens ends the run, an unconstrained request sharing
// the rounds generates what it generates alone, and each step's reported distribution matches the
// CausalScoring oracle (score_tokens over prompt + output with the step's tokens as candidates).
// Requires INFERNIX_TEST_ARTIFACT (an artifact with a DFlash2 drafter).

#include "infernix/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kMaxContext = 2048;
// BF16 logits from two routes over the same weights; restricted probabilities are compared.
constexpr double kProbabilityTolerance = 0.03;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::vector<infernix::TokenId> single(infernix::Engine& engine, const std::string& text) {
    const std::vector<infernix::TokenId> ids = engine.tokenize_text(text);
    if (ids.size() != 1) { throw std::runtime_error("fixture text is not one token: " + text); }
    return ids;
}

infernix::RequestOptions constrained(infernix::TokenConstraint constraint, float temperature,
                                   std::uint64_t seed) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens =
        static_cast<std::uint32_t>(constraint.steps.size());
    options.execution.sampling.temperature = temperature;
    options.execution.sampling.seed        = seed;
    options.execution.constraint           = std::move(constraint);
    options.stop.include_model_defaults    = false;
    return options;
}

infernix::EngineOptions engine_options(const char* artifact, infernix::SpeculativeBackend backend) {
    infernix::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = kMaxContext;
    options.kv_capacity     = infernix::KvCapacityPolicy::explicit_capacity(4 * kMaxContext);
    options.kv_cache        = infernix::KvCacheStorage::BFloat16;
    options.max_concurrency = 2;
    // /v1/decide runs on infernix-serve's default cache, whose capacity may exceed the lanes' sum.
    options.context_cache.mode  = infernix::ContextCacheMode::Hybrid;
    options.speculative.backend = backend;
    if (backend == infernix::SpeculativeBackend::DFlash2) { options.speculative.draft_tokens = 7; }
    return options;
}

std::vector<infernix::TokenId> prompt_tokens(infernix::Engine& engine) {
    return engine.tokenize_text(
        "The warehouse shipped 4417 parcels on Monday, 3920 on Tuesday and 5102 on Wednesday. "
        "Answer as JSON. The number of parcels shipped on Tuesday was {\"value\":");
}

struct Run {
    std::vector<infernix::TokenId> prompt;
    infernix::TokenConstraint constraint;
    infernix::GenerationResult result;
};

void check_run(const std::string& label, const Run& run) {
    const auto& tokens = run.result.generated_token_ids;
    check(!tokens.empty() && tokens.size() <= run.constraint.steps.size(),
          label + ": the run is empty or longer than its schedule");
    check(run.result.constrained_draws.size() == tokens.size(),
          label + ": one draw per generated token");
    for (std::size_t i = 0; i < tokens.size() && i < run.result.constrained_draws.size(); ++i) {
        const auto& step = run.constraint.steps[i];
        const auto& draw = run.result.constrained_draws[i];
        check(std::find(step.begin(), step.end(), tokens[i]) != step.end(),
              label + ": step " + std::to_string(i) + " committed a token it does not permit");
        check(draw.token == tokens[i] && draw.probabilities.size() == step.size(),
              label + ": draw " + std::to_string(i) + " does not describe its token and step");
        double sum = 0.0;
        for (const float p : draw.probabilities) { sum += p; }
        check(std::fabs(sum - 1.0) < 1e-3 && draw.mass >= 0.0F && draw.mass <= 1.0F,
              label + ": draw " + std::to_string(i) + " is not a distribution");
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: INFERNIX_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        std::vector<Run> greedy_runs;
        for (const auto backend :
             {infernix::SpeculativeBackend::None, infernix::SpeculativeBackend::DFlash2}) {
            const std::string name =
                backend == infernix::SpeculativeBackend::None ? "plain" : "dflash2";
            infernix::Engine engine(engine_options(artifact, backend));
            std::vector<infernix::TokenId> digits;
            for (char c = '0'; c <= '9'; ++c) { digits.push_back(single(engine, std::string(1, c)).front()); }
            const infernix::TokenId brace = single(engine, "}").front();
            const std::vector<infernix::TokenId> prompt = prompt_tokens(engine);

            // Four digit steps, then a forced brace.
            infernix::TokenConstraint field;
            field.steps.assign(4, digits);
            field.steps.push_back({brace});
            Run greedy{prompt, field,
                       engine.generate(engine.prepare_tokens(prompt), constrained(field, 0.0F, 1))};
            check_run(name + " greedy", greedy);
            check(greedy.result.generated_token_ids.size() == 5 &&
                      greedy.result.generated_token_ids.back() == brace,
                  name + ": the forced brace closes the field");
            for (std::size_t i = 0; i + 1 < greedy.result.constrained_draws.size(); ++i) {
                const auto& draw = greedy.result.constrained_draws[i];
                const auto best =
                    std::max_element(draw.probabilities.begin(), draw.probabilities.end()) -
                    draw.probabilities.begin();
                check(digits[static_cast<std::size_t>(best)] == draw.token,
                      name + ": greedy step " + std::to_string(i) +
                          " did not take its most probable permitted token");
            }
            greedy_runs.push_back(greedy);

            for (std::uint64_t seed = 1; seed <= 3; ++seed) {
                Run sampled{prompt, field,
                            engine.generate(engine.prepare_tokens(prompt),
                                            constrained(field, 1.0F, seed))};
                check_run(name + " sampled seed " + std::to_string(seed), sampled);
            }

            // A stop token among the permitted tokens ends the run where it is drawn.
            std::vector<infernix::TokenId> closing = digits;
            closing.push_back(brace);
            infernix::TokenConstraint open;
            open.steps.assign(8, closing);
            infernix::RequestOptions stop = constrained(open, 0.0F, 1);
            stop.stop.token_ids         = {brace};
            const infernix::GenerationResult stopped =
                engine.generate(engine.prepare_tokens(prompt), std::move(stop));
            check(stopped.finish_reason == infernix::FinishReason::StopToken &&
                      !stopped.generated_token_ids.empty() &&
                      stopped.generated_token_ids.back() == brace &&
                      stopped.constrained_draws.size() == stopped.generated_token_ids.size(),
                  name + ": a permitted stop token ends the run and keeps its draw");

            // An unconstrained request sharing the rounds is unaffected by its neighbour's masks.
            infernix::RequestOptions free;
            free.execution.requested_output_tokens = 24;
            free.execution.sampling.temperature    = 0.0F;
            free.stop.include_model_defaults       = false;
            // (Batched rounds may round differently from solo ones, so the checks are about the
            // masks: the neighbour keeps writing non-digits, the constrained run keeps its steps.)
            const std::vector<infernix::TokenId> other =
                engine.tokenize_text("List the first ten prime numbers separated by commas:");
            infernix::GenerationHandle masked =
                engine.submit(engine.prepare_tokens(prompt), constrained(field, 0.0F, 1));
            infernix::GenerationHandle beside = engine.submit(engine.prepare_tokens(other), free);
            Run masked_run{prompt, field, masked.wait()};
            const infernix::GenerationResult beside_result = beside.wait();
            check_run(name + " beside a neighbour", masked_run);
            check(std::any_of(beside_result.generated_token_ids.begin(),
                              beside_result.generated_token_ids.end(),
                              [&](infernix::TokenId id) {
                                  return std::find(digits.begin(), digits.end(), id) ==
                                         digits.end();
                              }) &&
                      beside_result.constrained_draws.empty(),
                  name + ": an unconstrained neighbour was constrained");
        }

        // The oracle: CausalScoring scores each output position with the step's tokens.
        infernix::EngineOptions scoring;
        scoring.artifact_path = artifact;
        scoring.purpose       = infernix::EnginePurpose::CausalScoring;
        scoring.max_context   = kMaxContext;
        scoring.kv_cache      = infernix::KvCacheStorage::BFloat16;
        infernix::Engine oracle(scoring);
        double worst = 0.0;
        for (const Run& run : greedy_runs) {
            std::vector<infernix::TokenId> tokens = run.prompt;
            const auto digit_steps              = run.result.generated_token_ids.size() - 1;
            tokens.insert(tokens.end(), run.result.generated_token_ids.begin(),
                          run.result.generated_token_ids.begin() +
                              static_cast<std::ptrdiff_t>(digit_steps));
            infernix::ScoreOptions options{.candidates_per_position = 10};
            for (std::size_t i = 0; i < digit_steps; ++i) {
                options.candidates.insert(options.candidates.end(), run.constraint.steps[i].begin(),
                                          run.constraint.steps[i].end());
            }
            const infernix::ScoreResult expected = oracle.score_tokens(
                tokens, static_cast<std::uint32_t>(run.prompt.size()), std::move(options));
            for (std::size_t i = 0; i < digit_steps; ++i) {
                double total = 0.0;
                for (std::size_t c = 0; c < 10; ++c) {
                    total += std::exp(static_cast<double>(expected.candidate_logprobs[i * 10 + c]));
                }
                for (std::size_t c = 0; c < 10; ++c) {
                    const double restricted =
                        std::exp(static_cast<double>(expected.candidate_logprobs[i * 10 + c])) /
                        total;
                    worst = std::max(worst,
                                     std::fabs(restricted - static_cast<double>(
                                                                run.result.constrained_draws[i]
                                                                    .probabilities[c])));
                }
            }
        }
        std::cout << "constraint draws vs CausalScoring: worst |d p| = " << worst << '\n';
        check(worst <= kProbabilityTolerance, "the reported distributions disagree with the oracle");
    } catch (const std::exception& error) {
        std::cerr << "constraint real test failed: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) { return 1; }
    std::cout << "OK constraint_real\n";
    return 0;
}
