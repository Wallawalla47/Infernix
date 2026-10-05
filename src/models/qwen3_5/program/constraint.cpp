// Constrained generation (TokenConstraint): staging each round's per-column descriptors for
// ops::constrain_logits and reading back the probability records of the tokens it produced.
#include "models/qwen3_5/program/program_impl.h"

#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

void ProgramImpl::bind_constraint_sets(std::uint32_t lane, const RequestConstraintPlan* plan) {
    if (!io.constraint || lane >= max_concurrency) {
        throw std::logic_error("constraint controls are unavailable for this lane");
    }
    const std::uint64_t serial = plan != nullptr ? plan->serial : 0U;
    if (constraint_bound_serials_[lane] == serial) { return; }
    constexpr std::size_t kSets    = kMaximumConstraintSets;
    constexpr std::size_t kChoices = ops::kTokenConstraintChoices;
    std::vector<std::int32_t> choices(kSets * kChoices, 0);
    std::vector<std::int32_t> counts(kSets, 0);
    if (plan != nullptr) {
        for (std::size_t set = 0; set < plan->sets.size(); ++set) {
            std::copy(plan->sets[set].begin(), plan->sets[set].end(),
                      choices.begin() + static_cast<std::ptrdiff_t>(set * kChoices));
            counts[set] = static_cast<std::int32_t>(plan->sets[set].size());
        }
    }
    auto* const device_choices = static_cast<std::int32_t*>(io.constraint->choices.data);
    auto* const device_counts  = static_cast<std::int32_t*>(io.constraint->counts.data);
    CUDA_CHECK(cudaMemcpyAsync(device_choices + lane * kSets * kChoices, choices.data(),
                               choices.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               device.stream));
    CUDA_CHECK(cudaMemcpyAsync(device_counts + lane * kSets, counts.data(),
                               counts.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               device.stream));
    constraint_bound_serials_[lane] = serial;
}

void ProgramImpl::stage_constraint_columns(std::span<const ConstraintRow> rows,
                                           std::uint32_t width) {
    if (!io.constraint) { return; }
    const std::size_t columns = rows.size() * width;
    if (columns > static_cast<std::size_t>(kTokenConstraintColumns)) {
        throw std::logic_error("constrained round is wider than its descriptor table");
    }
    bool constrained = false;
    for (const ConstraintRow& row : rows) { constrained = constrained || row.plan != nullptr; }
    if (!constrained && constraint_device_columns_ == 0) { return; }

    std::vector<std::int32_t>& host = constraint_descriptors_host_;
    std::fill(host.begin(), host.begin() + static_cast<std::ptrdiff_t>(columns), -1);
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const ConstraintRow& row = rows[index];
        if (row.plan == nullptr) { continue; }
        bind_constraint_sets(row.lane, row.plan);
        const auto& steps = row.plan->step_descriptors;
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::size_t step = static_cast<std::size_t>(row.first_step) + column;
            if (step >= steps.size()) { break; }
            const std::int32_t descriptor = steps[step];
            host[index * width + column] =
                descriptor >= 0
                    ? static_cast<std::int32_t>(row.lane * kMaximumConstraintSets) + descriptor
                    : descriptor;
        }
    }
    // Columns a previous round constrained beyond this round's extent return to -1 as well.
    const std::size_t extent = std::max(columns, constraint_device_columns_);
    std::fill(host.begin() + static_cast<std::ptrdiff_t>(columns),
              host.begin() + static_cast<std::ptrdiff_t>(extent), -1);
    CUDA_CHECK(cudaMemcpyAsync(io.constraint->descriptors.data, host.data(),
                               extent * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               device.stream));
    constraint_device_columns_ = constrained ? columns : 0;
}

std::vector<ConstrainedDraw>
ProgramImpl::read_constraint_draws(const RequestConstraintPlan& plan, std::uint32_t first_step,
                                   std::size_t first_column, std::span<const TokenId> tokens) {
    if (!io.constraint || first_column + tokens.size() > kTokenConstraintColumns) {
        throw std::logic_error("constraint records are outside the round");
    }
    constexpr std::size_t kRecord = ops::kTokenConstraintRecord;
    std::vector<float> records(tokens.size() * kRecord);
    CUDA_CHECK(cudaMemcpy(records.data(),
                          static_cast<const float*>(io.constraint->records.data) +
                              first_column * kRecord,
                          records.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<ConstrainedDraw> draws;
    draws.reserve(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::size_t step = static_cast<std::size_t>(first_step) + index;
        if (step >= plan.step_descriptors.size()) {
            throw std::logic_error("constrained request produced a token past its last step");
        }
        const std::int32_t descriptor = plan.step_descriptors[step];
        const std::size_t permitted =
            descriptor >= 0 ? plan.sets[static_cast<std::size_t>(descriptor)].size() : 1U;
        const float* record = records.data() + index * kRecord;
        draws.push_back(ConstrainedDraw{
            .token         = tokens[index],
            .probabilities = std::vector<float>(record, record + permitted),
            .mass          = record[ops::kTokenConstraintChoices],
        });
    }
    return draws;
}

const RequestConstraintPlan* ProgramImpl::constraint_plan(std::uint32_t lane) const noexcept {
    const RequestControl& request = requests[lane];
    return request.base && !request.base->constraint.empty() ? &request.base->constraint : nullptr;
}

std::uint32_t ProgramImpl::constraint_first_step(std::uint32_t lane) const {
    const SequenceState& sequence = sequences[lane];
    const std::uint32_t prompt    = requests[lane].base->summary.prompt_tokens;
    if (sequence.ledger.size() < prompt) {
        throw std::logic_error("constrained sequence ledger is shorter than its prompt");
    }
    return static_cast<std::uint32_t>(sequence.ledger.size() - prompt);
}

} // namespace ninfer::models::qwen3_5::detail
