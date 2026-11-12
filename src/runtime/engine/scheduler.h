#pragma once

#include "infernix/types.h"
#include "runtime/contract/execution.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace infernix::runtime {

// Decode share between prefill units (temporary switches INFERNIX_TMP_DECODE_SHARE and
// INFERNIX_TMP_DECODE_SHARE_MIN_TOKENS). Each cycle runs one Control/Decode batch before its
// prefill turn. After an ordinary prefill unit of wall time P that leaves at least
// `min_remaining_tokens` of its prompt (counted at unit boundaries) while another lane waits to
// decode, prefill turns yield until Control/Decode work has had share * P of wall time (the next
// cycle's batch included), so decode gets share / (1 + share) of the wall time while a long prompt
// prefills; share 0 is today's one batch per unit. A prompt's last unit earns no share. Only whole
// units interleave, so every prompt keeps its chunk grid and every output bit is unchanged.
// Replay and recovering requests never yield.
struct DecodeShare {
    double share                       = 0.0; // in [0, 1]
    std::uint32_t min_remaining_tokens = 0;
};

// Membership and fairness only. Native stores own every physical reservation.
template <class Request>
class Scheduler {
public:
    using RequestPtr     = std::shared_ptr<Request>;
    using SequenceHandle = typename Request::SequenceHandle;

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ControlMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::vector<TokenId> tokens;
        std::uint32_t row_stride = 0;
        std::size_t size         = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }
    };

    template <class Slots>
    [[nodiscard]] RoundMembership build_round_membership(const Slots& slots,
                                                         std::uint32_t max_concurrency) const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_decode_ready()) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            if (!request->sequence) {
                throw std::logic_error("decode-ready request has no sequence handle");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.budgets[membership.size]   = RoundBudget{
                  .generated_tokens_remaining = request->recovery_pending
                                                    ? 1U
                                                    : request->output.model_token_budget_remaining(
                                                        request->budget->remaining()),
            };
            if (membership.budgets[membership.size].generated_tokens_remaining == 0) {
                throw std::logic_error("decode-ready request has no licensed model tokens");
            }
            ++membership.size;
        }
        return membership;
    }

    template <class Slots>
    [[nodiscard]] ControlMembership build_control_membership(const Slots& slots,
                                                             std::uint32_t max_concurrency) const {
        ControlMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_control_ready()) { continue; }
            if (!request->budget || !request->sequence) {
                throw std::logic_error("control-ready request has no generation state");
            }
            const std::span<const TokenId> control = request->output.pending_control_tokens();
            if (control.empty() || control.size() > request->budget->remaining()) {
                throw std::logic_error("control-ready request has no admissible control span");
            }
            if (membership.row_stride == 0) {
                membership.row_stride = static_cast<std::uint32_t>(control.size());
                membership.tokens.reserve(static_cast<std::size_t>(membership.row_stride) *
                                          max_concurrency);
            } else if (control.size() != membership.row_stride) {
                throw std::logic_error("compact control membership has ragged target spans");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.tokens.insert(membership.tokens.end(), control.begin(), control.end());
            ++membership.size;
        }
        return membership;
    }

    // The lane whose prefill or Replay unit runs next: the least remaining prompt first, aged
    // (engine-architecture §"Prefill order"). A lane's key is the prompt tokens it still prefills (a
    // Replay counts none: it resumes a request that already had its turn) minus half the tokens other
    // lanes prefilled while it waited. The lane closest to its first token therefore keeps the turn
    // (its key falls by every token it prefills, a waiting lane's by half of each), so prompts that arrive
    // together reach their first tokens one after another instead of all together, and a short prompt
    // beside a long one goes first; a waiting lane outranks any newcomer once others have prefilled twice
    // its remaining prompt, which bounds its wait. Ties go to the rotation order after the last lane
    // served. Each unit keeps the request's own chunk grid: only the order of units changes.
    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    next_prefill(const Slots& slots, std::uint32_t concurrency) const noexcept {
        std::optional<std::uint32_t> best;
        std::int64_t best_key = 0;
        for (std::uint32_t offset = 0; offset < concurrency; ++offset) {
            const std::uint32_t lane = (prefill_cursor_ + offset) % concurrency;
            const auto& request      = slots[lane];
            if (!request || !(request->is_prefilling() || request->is_replaying())) { continue; }
            const std::int64_t key = prefill_key(*request, lane);
            if (!best || key < best_key) {
                best     = lane;
                best_key = key;
            }
        }
        return best;
    }

    // After lane `lane`'s unit prefilled or replayed `tokens`: every other lane still prefilling or
    // replaying waited that long.
    template <class Slots>
    void prefill_executed(std::uint32_t lane, std::uint32_t concurrency, const Slots& slots,
                          std::uint32_t tokens) noexcept {
        prefill_cursor_ = (lane + 1U) % concurrency;
        for (std::uint32_t other = 0; other < concurrency; ++other) {
            const auto& request = slots[other];
            if (other == lane || !request || !(request->is_prefilling() || request->is_replaying())) {
                continue;
            }
            if (prefill_waiter_[other] != request->id) {
                prefill_waiter_[other] = request->id;
                prefill_waited_[other] = 0;
            }
            prefill_waited_[other] += tokens;
        }
    }

    // Decode share between prefill units (DecodeShare).
    void set_decode_share(DecodeShare policy) {
        if (!(policy.share >= 0.0 && policy.share <= 1.0)) {
            throw std::invalid_argument("decode share must lie in [0, 1]");
        }
        decode_share_ = policy;
        decode_owed_ns_ = 0;
    }

    [[nodiscard]] const DecodeShare& decode_share() const noexcept { return decode_share_; }

    // `remaining_tokens`: the prompt tokens still to prefill after the unit; `decode_waiting`:
    // whether another lane is ready for Control/Decode work.
    void prefill_unit_finished(std::uint64_t wall_ns, bool complete, std::uint32_t remaining_tokens,
                               bool decode_waiting) noexcept {
        if (decode_share_.share <= 0.0 || complete || !decode_waiting ||
            remaining_tokens < decode_share_.min_remaining_tokens) {
            decode_owed_ns_ = 0;
            return;
        }
        // A batch's overshoot past the previous unit's share (at most one batch) is carried, so the
        // long-run split follows the share.
        decode_owed_ns_ = std::min<std::int64_t>(decode_owed_ns_, 0) +
                          static_cast<std::int64_t>(decode_share_.share * static_cast<double>(wall_ns));
    }

    void decode_work_finished(std::uint64_t wall_ns) noexcept {
        if (decode_owed_ns_ > 0) { decode_owed_ns_ -= static_cast<std::int64_t>(wall_ns); }
    }

    // A cycle without Control/Decode work ends the share: nothing it could buy is waiting.
    void decode_work_absent() noexcept { decode_owed_ns_ = 0; }

    // Whether this cycle's (unprotected) prefill turn gives way to further Control/Decode batches.
    [[nodiscard]] bool prefill_yields() noexcept {
        if (decode_owed_ns_ <= 0) { return false; }
        ++decode_share_yields_;
        return true;
    }

    [[nodiscard]] std::uint64_t decode_share_yields() const noexcept { return decode_share_yields_; }

    // Queue/capacity changes start a bounded pass. Ordinary decode cycles only continue
    // an unfinished pass; they do not restart failed admission checks.
    void begin_admission_scan() noexcept { admission_scan_requested_ = true; }

    [[nodiscard]] bool admission_scan_pending() const noexcept {
        return admission_scan_requested_ || !admission_scan_remaining_.empty();
    }

    void admission_candidate_checked(std::uint64_t request_id) noexcept {
        std::erase(admission_scan_remaining_, request_id);
        if (request_id != admission_scan_head_) { scan_after_ = request_id; }
    }

    // Only a successful younger admission consumes the older request's bypass allowance.
    [[nodiscard]] std::vector<RequestPtr> fresh_candidates(const std::deque<RequestPtr>& fresh,
                                                           std::uint32_t concurrency) {
        std::vector<RequestPtr> candidates;
        if (fresh.empty()) {
            admission_scan_requested_ = false;
            admission_scan_remaining_.clear();
            return candidates;
        }
        admission_scan_head_ = fresh.front()->id;
        std::size_t limit    = fresh.size();
        for (std::size_t i = 0; i < fresh.size(); ++i) {
            if (fresh[i]->admission_bypasses >= concurrency) {
                limit = i + 1;
                break;
            }
        }
        if (admission_scan_requested_) {
            admission_scan_remaining_.clear();
            for (std::size_t i = 0; i < limit; ++i) {
                admission_scan_remaining_.push_back(fresh[i]->id);
            }
            admission_scan_requested_ = false;
        } else {
            // Cancellation and a newly exhausted bypass allowance can remove eligibility
            // while the pass is in progress. Keep tickets, not queue positions or iterators.
            std::erase_if(admission_scan_remaining_, [&](std::uint64_t id) {
                return std::none_of(fresh.begin(), fresh.begin() + limit,
                                    [id](const auto& request) { return request->id == id; });
            });
        }
        if (admission_scan_remaining_.empty()) { return candidates; }
        candidates.push_back(fresh.front());
        std::vector<RequestPtr> younger;
        for (std::size_t i = 1; i < limit; ++i) {
            if (std::find(admission_scan_remaining_.begin(), admission_scan_remaining_.end(),
                          fresh[i]->id) != admission_scan_remaining_.end()) {
                younger.push_back(fresh[i]);
            }
        }
        std::sort(younger.begin(), younger.end(),
                  [](const auto& a, const auto& b) { return a->id < b->id; });
        const auto next = std::find_if(younger.begin(), younger.end(), [&](const auto& request) {
            return request->id > scan_after_;
        });
        std::rotate(younger.begin(), next, younger.end());
        const auto count = std::min<std::size_t>(concurrency, younger.size());
        candidates.insert(candidates.end(), younger.begin(), younger.begin() + count);
        return candidates;
    }

    void admitted(const std::deque<RequestPtr>& fresh, const Request& admitted,
                  std::uint32_t concurrency) {
        for (const auto& skipped : fresh) {
            if (skipped->id == admitted.id) {
                service_order_.push_back(admitted.id);
                return;
            }
            if (skipped->admission_bypasses >= concurrency) {
                throw std::logic_error("fresh admission exceeded the FIFO bypass allowance");
            }
            ++skipped->admission_bypasses;
        }
        throw std::logic_error("admitted request is absent from the fresh queue");
    }

    // Bounded service under KV pressure (resource-scheduling §5.1): while admitted requests
    // contend for capacity, one of them owns each turn. The progress owner keeps the next
    // capacity hand-over across reclamation, pause, binding and Replay (no other resident takes
    // capacity released for it); turns rotate through the admitted requests in admission order.
    [[nodiscard]] std::optional<std::uint64_t> progress_owner() const noexcept {
        if (!pressure_ || service_order_.empty()) { return std::nullopt; }
        return service_order_.front();
    }

    // `request` could not obtain its unit after reclamation. Without a turn in progress it takes
    // the first; during one it only records contention (the turn then rotates instead of ending).
    void require_progress(std::uint64_t request) {
        contended_ = true;
        if (pressure_) { return; }
        const auto owner = std::find(service_order_.begin(), service_order_.end(), request);
        if (owner == service_order_.end()) {
            throw std::logic_error("capacity waiter is absent from admitted membership");
        }
        std::rotate(service_order_.begin(), owner, service_order_.end());
        pressure_ = true;
        reset_service();
    }

    // An executed unit with committed new progress for `request` (Replay and waiting for a grant
    // are not service).
    void served(std::uint64_t request, std::uint64_t elapsed_ns) noexcept {
        if (progress_owner() != request) { return; }
        served_ = true;
        service_ns_ += std::min(elapsed_ns, maximum_service_ns - service_ns_);
    }

    // Replay toward the owner's paused frontier: lengthens its turn (up to maximum_service_ns),
    // never counts as service.
    void recovery_work(std::uint64_t request, std::uint64_t elapsed_ns) noexcept {
        if (progress_owner() == request) {
            recovery_ns_ += std::min(elapsed_ns, maximum_service_ns - recovery_ns_);
        }
    }

    // At a cycle boundary with no context transaction: ends the turn once its owner has served
    // max(service_quantum_ns, its recovery cost), never while a recovery is pending. Pressure
    // ends when nothing is paused and no other request contended during the turn; otherwise the
    // next admitted request takes the turn. Returns whether the turn ended.
    bool advance_service(bool have_paused, bool recovering) noexcept {
        if (!pressure_ || recovering || !served_) { return false; }
        if (service_ns_ < std::max(service_quantum_ns, recovery_ns_) && service_order_.size() > 1) {
            return false;
        }
        if (!have_paused && (!contended_ || service_order_.size() <= 1)) {
            pressure_ = false;
            begin_admission_scan();
        } else {
            std::rotate(service_order_.begin(), service_order_.begin() + 1, service_order_.end());
        }
        reset_service();
        contended_ = false;
        return true;
    }

    // A terminal request leaves the service order; an owner's turn passes to the next request.
    void finished(std::uint64_t request) noexcept {
        const bool owner = progress_owner() == request;
        std::erase(service_order_, request);
        if (owner) { reset_service(); }
        if (service_order_.empty()) { pressure_ = false; }
        begin_admission_scan();
    }

    // The resident a pressure turn pauses for its owner: the youngest one that is not the owner,
    // not binding, not finished and not protecting a Native recovery, outside `excluded` (lanes
    // that could not pause). None without a turn.
    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    preemption_victim(const Slots& slots, std::uint32_t concurrency,
                      std::span<const std::uint32_t> excluded = {}) const noexcept {
        const auto owner = progress_owner();
        if (!owner) { return std::nullopt; }
        std::optional<std::uint32_t> youngest;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            const auto& request = slots[lane];
            if (!request || request->id == *owner || request->is_materializing() ||
                request->is_model_finished() || request->recovery_pending ||
                std::find(excluded.begin(), excluded.end(), lane) != excluded.end()) {
                continue;
            }
            if (!youngest || request->id > slots[*youngest]->id) { youngest = lane; }
        }
        return youngest;
    }

    // A turn grants at least one second of useful execution; an expensive restore earns up to
    // four seconds to amortize its Replay or transfer, a finite bound either way.
    static constexpr std::uint64_t service_quantum_ns = 1'000'000'000;
    static constexpr std::uint64_t maximum_service_ns = 4 * service_quantum_ns;

private:
    template <class R>
    [[nodiscard]] std::int64_t prefill_key(const R& request, std::uint32_t lane) const noexcept {
        const std::int64_t remaining =
            request.is_replaying() ? 0 : static_cast<std::int64_t>(request.prefill_remaining_tokens());
        const std::uint64_t waited = prefill_waiter_[lane] == request.id ? prefill_waited_[lane] : 0;
        return remaining - static_cast<std::int64_t>(waited / 2U);
    }

    std::uint32_t prefill_cursor_      = 0;
    // Per lane: the request that has waited for prefill units there, and the tokens other lanes
    // prefilled meanwhile (next_prefill's ageing).
    std::array<std::uint64_t, kMaximumConcurrency> prefill_waiter_{};
    std::array<std::uint64_t, kMaximumConcurrency> prefill_waited_{};
    DecodeShare decode_share_;
    std::int64_t decode_owed_ns_       = 0;
    std::uint64_t decode_share_yields_ = 0;
    std::uint64_t scan_after_          = 0;
    std::uint64_t admission_scan_head_ = 0;
    std::vector<std::uint64_t> admission_scan_remaining_;
    bool admission_scan_requested_ = false;
    std::deque<std::uint64_t> service_order_;
    bool pressure_             = false;
    bool contended_            = false;
    bool served_               = false;
    std::uint64_t service_ns_  = 0;
    std::uint64_t recovery_ns_ = 0;

    void reset_service() noexcept {
        served_      = false;
        service_ns_  = 0;
        recovery_ns_ = 0;
    }
};

} // namespace infernix::runtime
