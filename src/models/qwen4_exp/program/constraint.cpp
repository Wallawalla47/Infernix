// /v1/decide for Qwen4Exp: constrained generation (TokenConstraint) and prompt readouts. A round's
// constrained rows mask their logits with ops::constrain_logits before the sampler or the draft
// acceptance reads them; the probability records of the tokens a round produced become the request's
// ConstrainedDraws as its tokens commit. A readout reads the prompt's next-token distribution from the
// logits its first token is sampled from, before any mask.

#include "models/qwen4_exp/program/program_impl.h"

#include "ninfer/ops/target_logprobs.h"
#include "ninfer/ops/top_logprobs.h"

#include <algorithm>
#include <iterator>

namespace ninfer::models::qwen4_exp::detail {

namespace {

constexpr std::size_t kSets    = kMaximumConstraintSets;
constexpr std::size_t kChoices = ops::kTokenConstraintChoices;
constexpr std::size_t kRecord  = ops::kTokenConstraintRecord;

} // namespace

// The constraint_ buffer: choices [kChoices, lanes x kSets], counts [lanes x kSets], descriptors
// [columns], records [kRecord, columns] (all 4-byte words).
std::shared_ptr<const ConstraintPlan> ProgramImpl::compile_constraint(const TokenConstraint& constraint) {
    if (constraint.empty()) { return nullptr; }
    auto plan = std::make_shared<ConstraintPlan>();
    for (const std::vector<TokenId>& step : constraint.steps) {
        if (step.empty() || step.size() > kMaximumConstraintChoices) {
            throw std::invalid_argument("constraint step has an invalid token count");
        }
        for (const TokenId id : step) {
            if (id < 0 || id >= token_domain_) {
                throw std::invalid_argument("constraint token is outside the public token domain");
            }
        }
        if (step.size() == 1) {
            plan->steps.push_back(-2 - step.front());
            continue;
        }
        auto set = std::find(plan->sets.begin(), plan->sets.end(), step);
        if (set == plan->sets.end()) {
            if (plan->sets.size() == kSets) { throw std::invalid_argument("constraint uses too many distinct token sets"); }
            set = plan->sets.insert(plan->sets.end(), step);
        }
        plan->steps.push_back(static_cast<std::int32_t>(set - plan->sets.begin()));
    }
    plan->serial = next_constraint_serial_++;
    return plan;
}

bool ProgramImpl::stage_constraints(std::span<const std::uint32_t> lanes, std::int32_t width) {
    constraint_columns_ = 0;
    if (std::none_of(lanes.begin(), lanes.end(), [&](std::uint32_t l) { return lanes_[l].constraint != nullptr; })) {
        return false;
    }
    const cudaStream_t s = device_.stream;
    const auto columns   = lanes.size() * static_cast<std::size_t>(width);
    if (columns > constraint_host_.size()) { throw std::logic_error("Qwen4Exp: a constrained round is wider than its table"); }
    const std::size_t sets_total = static_cast<std::size_t>(options_.max_concurrency) * kSets;
    auto* const words            = static_cast<std::int32_t*>(constraint_.p);
    std::fill(constraint_host_.begin(), constraint_host_.begin() + static_cast<std::ptrdiff_t>(columns), -1);
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t index = lanes[row];
        const Lane& lane          = lanes_[index];
        if (!lane.constraint) { continue; }
        const ConstraintPlan& plan = *lane.constraint;
        if (constraint_serials_[index] != plan.serial) {
            // The lane's set columns: written once per request.
            std::vector<std::int32_t> choices(kSets * kChoices, 0), counts(kSets, 0);
            for (std::size_t set = 0; set < plan.sets.size(); ++set) {
                std::copy(plan.sets[set].begin(), plan.sets[set].end(), choices.begin() + static_cast<std::ptrdiff_t>(set * kChoices));
                counts[set] = static_cast<std::int32_t>(plan.sets[set].size());
            }
            CUDA_CHECK(cudaMemcpyAsync(words + index * kSets * kChoices, choices.data(), 4ULL * choices.size(),
                                       cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemcpyAsync(words + sets_total * kChoices + index * kSets, counts.data(), 4ULL * counts.size(),
                                       cudaMemcpyHostToDevice, s));
            constraint_serials_[index] = plan.serial;
        }
        // Column j of the row samples output step (generated tokens) + j.
        const std::size_t first = lane.history.size() - lane.prompt_tokens;
        for (std::size_t j = 0; j < static_cast<std::size_t>(width) && first + j < plan.steps.size(); ++j) {
            const std::int32_t d = plan.steps[first + j];
            constraint_host_[row * static_cast<std::size_t>(width) + j] =
                d >= 0 ? static_cast<std::int32_t>(index * kSets) + d : d;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(words + sets_total * (kChoices + 1), constraint_host_.data(), 4ULL * columns,
                               cudaMemcpyHostToDevice, s));
    constraint_columns_ = static_cast<std::int32_t>(columns);
    return true;
}

void ProgramImpl::constrain(Tensor& logits, Tensor* argmax) {
    const std::int32_t columns = constraint_columns_;
    if (columns == 0 || logits.ne[1] != columns) { throw std::logic_error("Qwen4Exp: constrained logits do not match the staged round"); }
    const auto sets_total = static_cast<std::int32_t>(options_.max_concurrency * kSets);
    auto* const words     = static_cast<std::int32_t*>(constraint_.p);
    const Tensor choices(words, DType::I32, {static_cast<std::int32_t>(kChoices), sets_total});
    const Tensor counts(words + sets_total * kChoices, DType::I32, {sets_total});
    const Tensor descriptors(words + sets_total * (kChoices + 1), DType::I32, {columns});
    Tensor records(words + sets_total * (kChoices + 1) + constraint_host_.size(), DType::FP32,
                   {static_cast<std::int32_t>(kRecord), columns});
    ops::constrain_logits(logits, argmax, descriptors, choices, counts, token_domain_, records, device_.stream);
}

void ProgramImpl::take_constraint_draws(Lane& lane, std::size_t first_column, std::span<const std::int32_t> tokens) {
    if (!lane.constraint) { return; }
    const ConstraintPlan& plan = *lane.constraint;
    if (first_column + tokens.size() > static_cast<std::size_t>(constraint_columns_)) {
        throw std::logic_error("Qwen4Exp: constraint records are outside the round");
    }
    const std::size_t sets_total = static_cast<std::size_t>(options_.max_concurrency) * kSets;
    const auto* records = reinterpret_cast<const float*>(static_cast<const std::int32_t*>(constraint_.p) +
                                                         sets_total * (kChoices + 1) + constraint_host_.size()) +
                          first_column * kRecord;
    std::vector<float> host(tokens.size() * kRecord);
    CUDA_CHECK(cudaMemcpy(host.data(), records, 4ULL * host.size(), cudaMemcpyDeviceToHost));
    const std::size_t first = lane.history.size() - lane.prompt_tokens;
    lane.constraint_round.clear();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (first + i >= plan.steps.size()) {
            throw std::logic_error("Qwen4Exp: a constrained request produced a token past its last step");
        }
        const std::int32_t d     = plan.steps[first + i];
        const std::size_t count  = d >= 0 ? plan.sets[static_cast<std::size_t>(d)].size() : 1U;
        const float* record      = host.data() + i * kRecord;
        lane.constraint_round.push_back(ConstrainedDraw{.token         = tokens[i],
                                                        .probabilities = std::vector<float>(record, record + count),
                                                        .mass          = record[kChoices]});
    }
}

void ProgramImpl::commit_constraint_draws(Lane& lane, std::size_t accepted) {
    if (lane.constraint_round.empty()) { return; }
    if (accepted > lane.constraint_round.size()) {
        throw std::logic_error("Qwen4Exp: a constrained row commits more tokens than it drew");
    }
    lane.constraint_trace.insert(lane.constraint_trace.end(), std::make_move_iterator(lane.constraint_round.begin()),
                                 std::make_move_iterator(lane.constraint_round.begin() + static_cast<std::ptrdiff_t>(accepted)));
    lane.constraint_round.clear();
}

PromptReadout ProgramImpl::read_prompt_frontier(const Lane& lane) {
    const std::size_t n = lane.readout.size();
    if (n == 0 || n > kMaximumReadoutTokens) { throw std::invalid_argument("prompt readout names an invalid token count"); }
    const cudaStream_t s = device_.stream;
    // The sampler read these logits rounded to BF16 (and masked them for a constrained request):
    // round the unmasked FP32 column again.
    Tensor wide(logits32_.p, DType::FP32, {vocab_, 1});
    Tensor narrow(logits16_.p, DType::BF16, {vocab_, 1});
    ops::cast_fp32_to_bf16(wide, narrow, s);
    auto scope          = work_->scope();
    const auto count    = static_cast<std::int32_t>(n);
    Tensor ids          = work_->alloc(DType::I32, {count, 1});
    Tensor logprobs     = work_->alloc(DType::FP32, {count, 1});
    Tensor top_id       = work_->alloc(DType::I32, {1, 1});
    Tensor top_logprob  = work_->alloc(DType::FP32, {1, 1});
    CUDA_CHECK(cudaMemcpyAsync(ids.data, lane.readout.data(), 4ULL * n, cudaMemcpyHostToDevice, s));
    ops::target_logprobs(narrow, ids, token_domain_, logprobs, s);
    ops::top_logprobs(narrow, token_domain_, top_id, top_logprob, s);
    PromptReadout readout;
    readout.logprobs.resize(n);
    CUDA_CHECK(cudaMemcpyAsync(readout.logprobs.data(), logprobs.data, 4ULL * n, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(&readout.top_token, top_id.data, sizeof(TokenId), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(&readout.top_logprob, top_logprob.data, sizeof(float), cudaMemcpyDeviceToHost, s));
    device_.synchronize();
    return readout;
}

} // namespace ninfer::models::qwen4_exp::detail
