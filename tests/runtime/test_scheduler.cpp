#include "runtime/engine/scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

struct Request {
    using SequenceHandle = std::uint64_t;
    enum class Phase { Waiting, Materializing, Prefill, Replay, Decode, Control, Finished };

    struct Budget {
        std::uint32_t tokens = 8;

        [[nodiscard]] std::uint32_t remaining() const noexcept { return tokens; }
    };

    struct Output {
        std::uint32_t model_tokens = 8;
        std::vector<infernix::TokenId> control;

        [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t total) const {
            return std::min(total, model_tokens);
        }

        [[nodiscard]] std::span<const infernix::TokenId> pending_control_tokens() const {
            return control;
        }
    };

    [[nodiscard]] bool is_decode_ready() const { return phase == Phase::Decode; }

    [[nodiscard]] bool is_control_ready() const { return phase == Phase::Control; }

    [[nodiscard]] bool is_prefilling() const { return phase == Phase::Prefill; }

    [[nodiscard]] bool is_replaying() const { return phase == Phase::Replay; }

    [[nodiscard]] std::uint32_t prefill_remaining_tokens() const { return remaining; }

    [[nodiscard]] bool is_materializing() const { return phase == Phase::Materializing; }

    [[nodiscard]] bool is_model_finished() const { return phase == Phase::Finished; }

    [[nodiscard]] bool is_decision() const { return decision; }

    std::uint64_t id                 = 0;
    std::uint32_t remaining          = 0; // prompt tokens still to prefill
    bool decision                    = false;
    std::uint32_t admission_bypasses = 0;
    Phase phase                      = Phase::Waiting;
    bool recovery_pending            = false;
    std::optional<Budget> budget     = Budget{};
    std::optional<SequenceHandle> sequence;
    Output output;
};

using Scheduler  = infernix::runtime::Scheduler<Request>;
using RequestPtr = std::shared_ptr<Request>;
using Slots      = std::array<RequestPtr, infernix::kMaximumConcurrency>;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

RequestPtr request(std::uint64_t id, Request::Phase phase = Request::Phase::Waiting) {
    auto value      = std::make_shared<Request>();
    value->id       = id;
    value->phase    = phase;
    value->sequence = id;
    return value;
}

RequestPtr decision(std::uint64_t id, Request::Phase phase = Request::Phase::Waiting) {
    auto value      = request(id, phase);
    value->decision = true;
    return value;
}

void test_fifo_bypass_survives_pressure() {
    Scheduler scheduler;
    const auto head = request(10);
    std::deque<RequestPtr> fresh{head, request(11), request(12), request(13)};

    // Merely inspecting requests that cannot obtain resources never spends bypass credit.
    for (int attempt = 0; attempt < 6; ++attempt) {
        scheduler.begin_admission_scan();
        const auto candidates = scheduler.fresh_candidates(fresh, 2);
        for (const auto& candidate : candidates) {
            scheduler.admission_candidate_checked(candidate->id);
        }
        require(!candidates.empty() && candidates.front() == head && candidates.size() <= 3,
                "fresh scan did not preserve the head and bounded work");
    }
    require(head->admission_bypasses == 0 && fresh[1]->admission_bypasses == 0,
            "failed admission attempts consumed FIFO bypass credit");

    scheduler.admitted(fresh, *fresh[1], 2);
    fresh.erase(fresh.begin() + 1);
    require(head->admission_bypasses == 1, "successful younger admission was not charged");

    scheduler.require_progress(11);
    scheduler.served(11, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(false, false) && !scheduler.progress_owner(),
            "resolved pressure retained the service owner");
    require(!scheduler.fresh_candidates(fresh, 2).empty(),
            "resolved pressure did not reopen fresh candidate scanning");
    require(head->admission_bypasses == 1, "pressure recovery reset a waiting request's credit");

    scheduler.begin_admission_scan();
    const auto candidates = scheduler.fresh_candidates(fresh, 2);
    const auto second     = std::find(candidates.begin(), candidates.end(), fresh[1]);
    require(second != candidates.end(), "second permitted younger request was not considered");
    scheduler.admitted(fresh, **second, 2);
    fresh.erase(fresh.begin() + 1);
    for (std::uint64_t id = 14; id < 20; ++id) {
        fresh.push_back(request(id));
        scheduler.begin_admission_scan();
        const auto blocked = scheduler.fresh_candidates(fresh, 2);
        require(blocked.size() == 1 && blocked.front() == head,
                "new arrivals or completion events bypassed an exhausted FIFO head");
    }

    // Cancelling the blocked head exposes the next request with its own existing credit.
    fresh.pop_front();
    scheduler.begin_admission_scan();
    const auto after_cancel = scheduler.fresh_candidates(fresh, 2);
    require(after_cancel.size() > 1 && after_cancel.size() <= 3 && after_cancel.front()->id == 13 &&
                after_cancel.front()->admission_bypasses == 0,
            "head removal transferred its exhausted credit to the next request");
}

void test_scan_progress_and_waiter_barrier() {
    Scheduler scheduler;
    std::deque<RequestPtr> fresh{request(1), request(2), request(3),
                                 request(4), request(5), request(6)};
    std::vector<std::uint64_t> seen;
    const auto observe = [&](const std::vector<RequestPtr>& candidates) {
        require(!candidates.empty() && candidates.front() == fresh.front() &&
                    candidates.size() <= 3,
                "candidate scan exceeded the head plus bounded younger requests");
        for (const auto& candidate : candidates) {
            require(std::find(fresh.begin(), fresh.end(), candidate) != fresh.end(),
                    "removed request remained a candidate");
            scheduler.admission_candidate_checked(candidate->id);
            if (std::find(seen.begin(), seen.end(), candidate->id) == seen.end()) {
                seen.push_back(candidate->id);
            }
        }
    };
    scheduler.begin_admission_scan();
    const auto first = scheduler.fresh_candidates(fresh, 2);
    observe(first);
    require(scheduler.admission_scan_pending(),
            "failed bounded batch abandoned later requests that could fit");
    const auto next = scheduler.fresh_candidates(fresh, 2);
    require(
        next.size() == 3 && next[1]->id == 4 && next[2]->id == 5,
        "a later fitting request required another arrival or resident completion to be checked");
    observe(next);

    // Remove the last examined ticket and add an arrival. The next event must preserve
    // cursor progress rather than restart indefinitely at the first failed requests.
    std::erase(fresh, next.back());
    fresh.push_back(request(7));
    scheduler.begin_admission_scan();
    const auto after_change = scheduler.fresh_candidates(fresh, 2);
    require(after_change.size() == 3 && after_change[1]->id == 6 && after_change[2]->id == 7,
            "queue changes reset the stable-ticket admission cursor");
    observe(after_change);
    for (std::size_t attempt = 0; scheduler.admission_scan_pending() && attempt < fresh.size();
         ++attempt) {
        observe(scheduler.fresh_candidates(fresh, 2));
    }
    require(!scheduler.admission_scan_pending(), "a complete admission pass kept retrying");
    require(scheduler.fresh_candidates(fresh, 2).empty(),
            "ordinary decode restarted a failed admission pass");
    for (const auto& pending : fresh) {
        require(std::find(seen.begin(), seen.end(), pending->id) != seen.end(),
                "failed leading candidates starved a later admission attempt");
        require(pending->admission_bypasses == 0, "candidate scanning spent admission credit");
    }

    // A skipped waiter retains its own limit even while an older request is still ahead of it.
    fresh[1]->admission_bypasses = 2;
    for (int attempt = 0; attempt < 3; ++attempt) {
        scheduler.begin_admission_scan();
        const auto candidates = scheduler.fresh_candidates(fresh, 2);
        require(candidates.size() == 2 && candidates[1] == fresh[1],
                "rotating scan crossed an exhausted non-head waiter");
        observe(candidates);
        require(!scheduler.admission_scan_pending(),
                "blocked younger requests kept the scan awake behind a FIFO barrier");
    }
}

void test_scan_keeps_unexamined_candidates() {
    Scheduler scheduler;
    std::deque<RequestPtr> fresh{request(1), request(2), request(3), request(4)};
    scheduler.begin_admission_scan();
    const auto first = scheduler.fresh_candidates(fresh, 2);
    require(first.size() == 3, "admission did not expose a bounded candidate batch");
    scheduler.admission_candidate_checked(first[0]->id);
    scheduler.admission_candidate_checked(first[1]->id);
    scheduler.admitted(fresh, *first[1], 2);
    std::erase(fresh, first[1]);
    // Admission returns early, so the last returned candidate has not actually been inspected.
    const auto continued = scheduler.fresh_candidates(fresh, 2);
    require(continued.size() == 3 && continued[1]->id == 3 && continued[2]->id == 4,
            "successful admission discarded candidates that had never been checked");
    for (const auto& candidate : continued) {
        scheduler.admission_candidate_checked(candidate->id);
    }
    require(!scheduler.admission_scan_pending(),
            "completed candidate checks did not retire the pass");
    scheduler.begin_admission_scan();
    require(!scheduler.fresh_candidates(fresh, 2).empty(),
            "a later capacity event did not retry a previously failed request");
}

void admit(Scheduler& scheduler, const RequestPtr& value) {
    scheduler.admitted(std::deque<RequestPtr>{value}, *value, 4);
}

// Bounded service under KV pressure (resource-scheduling §5.2), after NInfer a754fcf3's tests.
void test_pressure_handoff_and_fair_service() {
    Scheduler scheduler;
    Slots slots{};
    auto a = slots[0] = request(10, Request::Phase::Decode);
    auto b = slots[1] = request(20, Request::Phase::Decode);
    auto c = slots[2] = request(30, Request::Phase::Decode);
    for (const auto& value : {a, b, c}) { admit(scheduler, value); }
    require(!scheduler.progress_owner() && !scheduler.preemption_victim(slots, 4),
            "a turn existed or a resident was preempted without pressure");

    scheduler.require_progress(a->id);
    require(scheduler.preemption_victim(slots, 4) == 2, "capacity handoff did not exclude its beneficiary");
    const std::array excluded{2U};
    require(scheduler.preemption_victim(slots, 4, excluded) == 1,
            "a resident that could not pause prevented another release candidate");
    slots[2].reset(); // c paused; the same capacity is still owed to a.
    scheduler.require_progress(c->id);
    require(scheduler.progress_owner() == a->id && !scheduler.advance_service(true, false),
            "a paused victim stole an unfinished capacity handoff");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(true, false), "another row's work expired the owner's turn");
    scheduler.served(a->id, Scheduler::service_quantum_ns - 1);
    require(!scheduler.advance_service(true, false), "an incomplete service slice expired");
    scheduler.served(a->id, 1);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == b->id,
            "a waiting request did not receive the next service turn");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == c->id,
            "a paused request was skipped by service rotation");
    require(scheduler.preemption_victim(slots, 4) == 1,
            "a paused owner could not preempt an older served resident");
    b->recovery_pending = true;
    require(scheduler.preemption_victim(slots, 4) == 0, "a protected Native recovery was chosen as a victim");
    b->recovery_pending = false;
    slots[3] = request(40, Request::Phase::Materializing);
    require(scheduler.preemption_victim(slots, 4) == 1, "a binding destination was chosen as a victim");
    slots[3].reset();
    // A pause that leaves the lane's state cached at its frontier is preferred over the youngest one
    // whose pause would recompute (a Qwen3.5 prefill without a whole DFlash2 ring); equals stay youngest.
    slots[3] = request(40, Request::Phase::Prefill);
    const std::array<std::uint32_t, 4> replays{0U, 0U, 0U, 4096U};
    const auto cost = [&](std::uint32_t lane) { return replays[lane]; };
    require(scheduler.preemption_victim(slots, 4, {}, cost) == 1,
            "a pause that replays was chosen over one that resumes from its own snapshot");
    const std::array<std::uint32_t, 4> uneven{512U, 1024U, 0U, 1024U};
    require(scheduler.preemption_victim(slots, 4, {}, [&](std::uint32_t lane) { return uneven[lane]; }) == 0,
            "the cheaper of two replaying pauses was not chosen");
    const std::array<std::uint32_t, 4> equal{7U, 7U, 7U, 7U};
    require(scheduler.preemption_victim(slots, 4, {}, [&](std::uint32_t lane) { return equal[lane]; }) == 3,
            "equal pause costs did not keep the youngest victim");
    slots[3].reset();
    // Arbitrarily many resource polls and Replay boundaries cannot consume c's turn.
    for (int i = 0; i < 100; ++i) {
        require(!scheduler.advance_service(true, true) && scheduler.progress_owner() == c->id,
                "Replay exhausted a turn without new progress");
    }
    scheduler.served(c->id, Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(true, true), "Native recovery protection was bypassed");
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == a->id,
            "fair rotation failed to return to the earlier request");
    scheduler.finished(a->id);
    require(scheduler.progress_owner() == b->id, "a terminal owner retained the handoff");
    scheduler.finished(b->id);
    require(scheduler.progress_owner() == c->id, "a terminal owner blocked the remaining work");
    scheduler.served(c->id, 1);
    require(scheduler.advance_service(false, false) && !scheduler.progress_owner(),
            "resolved pressure did not reopen fresh admission");
    scheduler.finished(c->id);
    require(!scheduler.progress_owner(), "an empty service order kept a turn");
}

// A turn is 30 seconds of service (candidates QN-01): Replay never counts as service, and an expensive one no
// longer lengthens the turn past it.
void test_recovery_cost_does_not_lengthen_a_turn() {
    Scheduler scheduler;
    auto a = request(1), b = request(2);
    admit(scheduler, a);
    admit(scheduler, b);
    scheduler.require_progress(a->id);
    scheduler.recovery_work(a->id, 10 * Scheduler::maximum_service_ns);
    require(!scheduler.advance_service(true, false), "Replay was counted as new progress");
    scheduler.served(a->id, Scheduler::service_quantum_ns - 1);
    require(!scheduler.advance_service(true, false), "a turn ended before its quantum of service");
    scheduler.served(a->id, 1);
    require(scheduler.advance_service(true, false) && scheduler.progress_owner() == b->id,
            "an expensive Replay lengthened the turn past its quantum");
    scheduler.served(b->id, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(true, false), "recovery cost leaked into another turn");
}

void test_bounded_service_under_continuous_contention() {
    Scheduler scheduler;
    std::array<RequestPtr, 4> requests{request(1), request(2), request(3), request(4)};
    for (const auto& value : requests) { admit(scheduler, value); }
    scheduler.require_progress(1);
    std::array<unsigned, 4> grants{};
    for (unsigned turn = 0; turn < 40; ++turn) {
        const auto owner = scheduler.progress_owner();
        require(owner && *owner == turn % 4 + 1, "an admitted waiter was bypassed");
        ++grants[*owner - 1];
        for (const auto& value : requests) { scheduler.require_progress(value->id); }
        scheduler.served(*owner, Scheduler::service_quantum_ns);
        require(scheduler.advance_service(true, false), "contended service failed to rotate");
    }
    require(std::all_of(grants.begin(), grants.end(), [](auto n) { return n == 10; }),
            "continuous contention starved a request");
}

// Runs one prefill unit of up to `unit` tokens on the lane next_prefill picks; returns that lane.
std::uint32_t run_unit(Scheduler& scheduler, Slots& slots, std::uint32_t concurrency, std::uint32_t unit) {
    const auto lane = scheduler.next_prefill(slots, concurrency);
    require(lane.has_value(), "no prefill unit was scheduled");
    Request& r                = *slots[*lane];
    const std::uint32_t done  = r.is_replaying() ? unit : std::min(unit, r.remaining);
    if (!r.is_replaying()) {
        r.remaining -= done;
        if (r.remaining == 0) { r.phase = Request::Phase::Decode; }
    }
    scheduler.prefill_executed(*lane, concurrency, slots, done);
    return *lane;
}

void test_prefill_order() {
    // Prompts that arrive together reach their first tokens one after another: the lane picked first
    // keeps the turn to the end of its prompt (no lockstep rotation).
    {
        Scheduler scheduler;
        Slots slots{};
        for (std::uint32_t lane = 0; lane < 3; ++lane) {
            slots[lane]            = request(lane + 1, Request::Phase::Prefill);
            slots[lane]->remaining = 4096 * 4;
        }
        std::vector<std::uint32_t> order;
        while (scheduler.next_prefill(slots, 3)) { order.push_back(run_unit(scheduler, slots, 3, 4096)); }
        const std::vector<std::uint32_t> expected{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
        require(order == expected, "equal prompts arriving together rotated instead of finishing in turn");
    }
    // A short prompt beside a long one goes first, even when the long one started.
    {
        Scheduler scheduler;
        Slots slots{};
        slots[0]            = request(1, Request::Phase::Prefill);
        slots[0]->remaining = 60000;
        require(run_unit(scheduler, slots, 2, 4096) == 0, "the only prefilling lane was not served");
        slots[1]            = request(2, Request::Phase::Prefill);
        slots[1]->remaining = 2000;
        require(run_unit(scheduler, slots, 2, 4096) == 1, "a short prompt waited behind a long one");
        require(slots[1]->is_decode_ready() && run_unit(scheduler, slots, 2, 4096) == 0,
                "the long prompt did not continue after the short one");
    }
    // A waiting lane's wait is bounded: shorter newcomers arriving on another lane overtake it only
    // until others have prefilled twice its remaining prompt.
    {
        Scheduler scheduler;
        Slots slots{};
        slots[0]            = request(1, Request::Phase::Prefill);
        slots[0]->remaining = 8192;
        std::uint64_t next_id = 2;
        std::uint32_t others  = 0;
        bool served           = false;
        for (int step = 0; step < 64 && !served; ++step) {
            if (!slots[1] || !slots[1]->is_prefilling()) {
                slots[1]            = request(next_id++, Request::Phase::Prefill);
                slots[1]->remaining = 4096;
            }
            const std::uint32_t lane = run_unit(scheduler, slots, 2, 4096);
            if (lane == 0) {
                served = true;
            } else {
                others += 4096;
            }
        }
        require(served && others <= 2 * 8192 + 4096, "a long prompt waited without bound behind newcomers");
    }
    // A Replay resumes a request that already had its turn: it precedes fresh prompts; unready lanes are
    // never picked.
    {
        Scheduler scheduler;
        Slots slots{};
        slots[0]            = request(1, Request::Phase::Decode);
        slots[1]            = request(2, Request::Phase::Prefill);
        slots[1]->remaining = 100;
        slots[2]            = request(3, Request::Phase::Replay);
        require(scheduler.next_prefill(slots, 4) == 2, "a Replay waited behind a fresh prompt");
        slots[2]->phase = Request::Phase::Decode;
        require(scheduler.next_prefill(slots, 4) == 1, "a prefilling lane lost its unit to unready members");
        slots[1]->phase = Request::Phase::Decode;
        require(!scheduler.next_prefill(slots, 4), "an unready member was scheduled");
    }
    // A new request on a lane starts its own wait: the previous request's ageing is not inherited.
    {
        Scheduler scheduler;
        Slots slots{};
        slots[0]            = request(1, Request::Phase::Prefill);
        slots[0]->remaining = 4096 * 3;
        slots[1]            = request(2, Request::Phase::Prefill);
        slots[1]->remaining = 4096 * 3;
        require(run_unit(scheduler, slots, 2, 4096) == 0, "the tie did not go to the rotation order");
        slots[1]            = request(3, Request::Phase::Prefill);
        slots[1]->remaining = 4096 * 2 + 1;
        require(run_unit(scheduler, slots, 2, 4096) == 0,
                "a replacement request inherited its lane's earlier wait");
    }
}

// The cycle's prefill permit under a service turn (resource-scheduling §5.2). The turn's owner takes it while it
// can run. An owner the Program blocks (on Qwen3.8-Flash-Next a prefilling lane behind another lane's layer walk) is
// absent from the runnable view: the permit then goes to the lane it waits behind, which would otherwise never
// advance, and neither would the owner; its turn is not spent meanwhile.
void test_blocked_owner_leaves_the_prefill_permit() {
    Scheduler scheduler;
    Slots slots{};
    auto decoder = slots[0] = request(10, Request::Phase::Decode);
    auto owner = slots[1] = request(20, Request::Phase::Prefill);
    auto walker = slots[2] = request(30, Request::Phase::Prefill);
    owner->remaining  = 50000;
    walker->remaining = 60000;
    for (const auto& value : {decoder, owner, walker}) { admit(scheduler, value); }
    scheduler.require_progress(decoder->id);
    scheduler.require_progress(walker->id); // contention: the turn rotates instead of ending
    scheduler.served(decoder->id, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(false, false) && scheduler.progress_owner() == owner->id,
            "the turn did not pass to the next admitted request");

    require(scheduler.prefill_turn(slots, 3, 1U) == 1, "a runnable owner lost the prefill permit");
    Slots runnable = slots;
    runnable[1].reset(); // the walker's layer walk blocks the owner's context
    require(scheduler.prefill_turn(runnable, 3, 1U) == 2,
            "a blocked owner kept the prefill permit from the lane it waits behind");
    // The walker's units are not the owner's service: the turn stays with it until it has run its own slice.
    scheduler.served(walker->id, 10 * Scheduler::service_quantum_ns);
    require(!scheduler.advance_service(false, false) && scheduler.progress_owner() == owner->id,
            "the blocked owner's turn was spent by the lane it waited behind");
    require(scheduler.prefill_turn(slots, 3, 1U) == 1, "the owner did not take the permit once it could run");
    scheduler.served(owner->id, Scheduler::service_quantum_ns);
    require(scheduler.advance_service(false, false), "the owner's turn did not end after its own slice");

    // An owner that decodes holds no prefill permit, and without a turn the order is next_prefill's.
    require(scheduler.prefill_turn(slots, 3, 0U) == 1 && scheduler.prefill_turn(slots, 3, std::nullopt) == 1,
            "a decoding owner or no owner changed the prefill order");
    // A blocked owner with nothing else to prefill: no permit, rather than one it cannot use.
    runnable[2].reset();
    require(!scheduler.prefill_turn(runnable, 3, 1U), "a permit went to a lane that cannot run");
}

// A unit short of capacity waits for a resident that has finished (its lane and pages come back at the next cycle
// boundary) instead of pausing another resident or failing with a capacity error: no victim is not no releaser.
void test_finished_resident_is_a_pending_release() {
    Scheduler scheduler;
    Slots slots{};
    auto owner = slots[0] = request(10, Request::Phase::Prefill);
    auto finishing = slots[1] = request(20, Request::Phase::Decode);
    for (const auto& value : {owner, finishing}) { admit(scheduler, value); }
    require(!scheduler.release_pending(slots, 2), "a decoding resident was taken for a pending release");
    scheduler.require_progress(owner->id);
    finishing->phase = Request::Phase::Finished; // its last round ended it; it is settled at the next boundary
    require(!scheduler.preemption_victim(slots, 2), "a finished resident was offered as a pause victim");
    require(scheduler.release_pending(slots, 2), "a finished resident's release was not seen: the owner would fail");
    slots[1].reset();
    scheduler.finished(finishing->id);
    require(!scheduler.release_pending(slots, 2) && scheduler.progress_owner() == owner->id,
            "the settled resident left a pending release or took the turn");
}

void test_decode_and_control_membership() {
    Scheduler scheduler;
    Slots slots{};
    slots[0]                      = request(1, Request::Phase::Decode);
    slots[0]->budget->tokens      = 11;
    slots[0]->output.model_tokens = 3;
    slots[2]                      = request(2, Request::Phase::Decode);
    slots[2]->recovery_pending    = true;
    slots[3]                      = request(3, Request::Phase::Prefill);
    const auto round              = scheduler.build_round_membership(slots, 4);
    require(round.size == 2 && round.lanes[0] == 0 && round.lanes[1] == 2 &&
                round.sequences[0] == 1 && round.sequences[1] == 2 &&
                round.budgets[0].generated_tokens_remaining == 3 &&
                round.budgets[1].generated_tokens_remaining == 1,
            "compact decode lost output limits or first-unit limits, or took a prefill lane");

    slots[0]->phase          = Request::Phase::Control;
    slots[0]->output.control = {7, 8, 9};
    slots[2]->phase          = Request::Phase::Control;
    slots[2]->output.control = {10, 11, 12};
    const auto control       = scheduler.build_control_membership(slots, 4);
    require(control.size == 2 && control.row_stride == 3 &&
                control.tokens == std::vector<infernix::TokenId>({7, 8, 9, 10, 11, 12}) &&
                control.sequences[0] == 1 && control.sequences[1] == 2,
            "compact control membership changed committed control spans or request order");
}

void test_decisions_queue_first() {
    std::deque<RequestPtr> fresh;
    for (const auto& item : {request(1), decision(2), request(3), decision(4)}) {
        Scheduler::enqueue(fresh, item, 2);
    }
    const auto ids = [&] {
        std::vector<std::uint64_t> order;
        for (const auto& item : fresh) { order.push_back(item->id); }
        return order;
    };
    require(ids() == std::vector<std::uint64_t>({2, 4, 1, 3}),
            "decisions were not queued ahead of other requests in arrival order");
    // Each request a decision passed was charged one bypass of its FIFO allowance.
    require(fresh[2]->admission_bypasses == 2 && fresh[3]->admission_bypasses == 1,
            "passing a waiting request did not charge its bypass allowance");

    // Admitting the decision at the head charges nobody again.
    Scheduler scheduler;
    scheduler.begin_admission_scan();
    const auto candidates = scheduler.fresh_candidates(fresh, 2);
    require(!candidates.empty() && candidates.front()->id == 2,
            "the queued decision was not the first admission candidate");
    scheduler.admitted(fresh, *fresh.front(), 2);
    require(fresh[2]->admission_bypasses == 2 && fresh[3]->admission_bypasses == 1,
            "admitting a queued decision counted as a FIFO bypass");
}

void test_decisions_keep_the_fifo_bound() {
    // A stream of decisions passes a waiting request at most `concurrency` times: later ones queue
    // behind it, in arrival order.
    std::deque<RequestPtr> fresh;
    Scheduler::enqueue(fresh, request(1), 2);
    for (std::uint64_t id = 2; id <= 6; ++id) { Scheduler::enqueue(fresh, decision(id), 2); }
    std::vector<std::uint64_t> order;
    for (const auto& item : fresh) { order.push_back(item->id); }
    require(order == std::vector<std::uint64_t>({2, 3, 1, 4, 5, 6}),
            "a stream of decisions passed a waiting request beyond its FIFO allowance");

    // A request that arrives after the stream is passed only by the decisions after it.
    Scheduler::enqueue(fresh, request(7), 2);
    Scheduler::enqueue(fresh, decision(8), 2);
    require(fresh[5]->id == 6 && fresh[6]->id == 8 && fresh[7]->id == 7,
            "a decision did not queue after the decisions ahead of it");

    // The admission scan stops at the request whose allowance is spent: no decision behind it is a
    // candidate, so the exhausted request is admitted before them.
    Scheduler scheduler;
    scheduler.begin_admission_scan();
    fresh.pop_front();
    fresh.pop_front();
    const auto candidates = scheduler.fresh_candidates(fresh, 2);
    require(candidates.size() == 1 && candidates.front()->id == 1,
            "a decision was an admission candidate ahead of a request whose allowance was spent");
}

void test_decisions_keep_their_queue_order_in_the_scan() {
    // Behind a head that cannot be admitted, the decisions are tried first and in queue order (their
    // ids need not follow it), then the other requests in the rotation.
    std::deque<RequestPtr> fresh{request(1), decision(5), decision(3), request(2), request(4)};
    Scheduler scheduler;
    scheduler.begin_admission_scan();
    const auto candidates = scheduler.fresh_candidates(fresh, 4);
    std::vector<std::uint64_t> order;
    for (const auto& item : candidates) { order.push_back(item->id); }
    require(order == std::vector<std::uint64_t>({1, 5, 3, 2, 4}),
            "the admission scan did not try the decisions in queue order ahead of the others");
}

void test_decision_lanes() {
    std::array<bool, 4> taken{};
    const auto is_free = [&](std::uint32_t lane) { return !taken[lane]; };
    // Four lanes, the last one reserved: a decision takes it first, others never do.
    require(Scheduler::free_lane(4, 1, true, is_free) == 3u,
            "a decision did not take the reserved lane first");
    require(Scheduler::free_lane(4, 1, false, is_free) == 0u,
            "an ordinary request did not take the first general lane");
    taken = {true, true, true, false};
    require(!Scheduler::free_lane(4, 1, false, is_free),
            "an ordinary request took the decision lane");
    require(Scheduler::free_lane(4, 1, true, is_free) == 3u,
            "a decision could not take its free lane while the general lanes were busy");
    taken = {true, false, true, true};
    require(Scheduler::free_lane(4, 1, true, is_free) == 1u,
            "a decision did not fall back to a general lane when the decision lane was busy");
    taken = {false, false, false, false};
    require(Scheduler::free_lane(4, 0, false, is_free) == 0u &&
                Scheduler::free_lane(4, 0, true, is_free) == 0u,
            "without decision lanes, lane choice changed");
}

void test_decision_prefill_turn() {
    Scheduler scheduler;
    Slots slots{};
    slots[0]            = request(1, Request::Phase::Prefill);
    slots[1]            = request(2, Request::Phase::Prefill);
    slots[0]->remaining = slots[1]->remaining = 1000;
    slots[2]            = decision(5, Request::Phase::Prefill);
    slots[3]            = decision(4, Request::Phase::Replay);
    require(scheduler.next_prefill(slots, 4) == 3u,
            "the oldest prefilling or replaying decision did not take the next prefill turn");
    scheduler.prefill_executed(3, 4, slots, 16, true);
    slots[3]->phase = Request::Phase::Finished;
    require(scheduler.next_prefill(slots, 4) == 2u, "the second decision lost its turn");
    scheduler.prefill_executed(2, 4, slots, 16, true);
    slots[2]->phase = Request::Phase::Finished;
    // Decisions took their turns out of order: the rotation resumes where it was (equal prompts, so
    // the aged keys tie and the rotation decides).
    for (const std::uint32_t expected : {0U, 1U, 0U}) {
        const auto lane = scheduler.next_prefill(slots, 4);
        require(lane == expected, "decision turns disturbed the prefill rotation");
        scheduler.prefill_executed(*lane, 4, slots, 16);
    }
}

void test_decisions_keep_the_prefill_bound() {
    // A stream of prefilling decisions keeps the turn only until a waiting request's aged key falls
    // below zero (others prefilled more than twice its remaining prompt): then the aged order serves it.
    Scheduler scheduler;
    Slots slots{};
    slots[0]            = request(1, Request::Phase::Prefill);
    slots[0]->remaining = 40;
    std::uint64_t next  = 100;
    std::uint32_t units = 0;
    for (;;) {
        if (!slots[1]) {
            slots[1]            = decision(next++, Request::Phase::Prefill);
            slots[1]->remaining = 16;
        }
        const auto lane = scheduler.next_prefill(slots, 2);
        require(lane.has_value(), "no prefill unit was scheduled");
        if (*lane == 0) { break; }
        scheduler.prefill_executed(1, 2, slots, 16, true);
        slots[1].reset();
        require(++units <= 6, "a stream of decisions held the prefill turn past the ageing bound");
    }
    require(units == 6, "a waiting request took the turn from a decision before its bound");

    // A Replay that has waited at all is overdue too.
    Scheduler replay_scheduler;
    slots[0]            = request(2, Request::Phase::Replay);
    slots[1]            = decision(next++, Request::Phase::Prefill);
    slots[1]->remaining = 16;
    require(replay_scheduler.next_prefill(slots, 2) == 1u, "a fresh Replay outranked a decision");
    replay_scheduler.prefill_executed(1, 2, slots, 16, true);
    require(replay_scheduler.next_prefill(slots, 2) == 0u,
            "a Replay that waited was not served before the next decision unit");
}

void test_decision_is_last_victim() {
    Scheduler scheduler;
    Slots slots{};
    auto a = slots[0] = request(10, Request::Phase::Decode);
    auto b = slots[1] = request(20, Request::Phase::Decode);
    auto c = slots[2] = decision(30, Request::Phase::Prefill);
    auto d = slots[3] = decision(40, Request::Phase::Prefill);
    for (const auto& value : {a, b, c, d}) { admit(scheduler, value); }
    // a owns the turn: the younger decisions stay while an ordinary resident can be paused.
    scheduler.require_progress(a->id);
    require(scheduler.preemption_victim(slots, 4) == 1u,
            "a younger decision was preempted while an ordinary resident could be");
    const std::array excluded{1U};
    require(scheduler.preemption_victim(slots, 4, excluded) == 3u,
            "the youngest decision was not the victim with no other candidate");
    slots[1].reset();
    require(scheduler.preemption_victim(slots, 4) == 3u,
            "a decision was never a victim, even with no other candidate");
    slots[3].reset();
    slots[2].reset();
    require(!scheduler.preemption_victim(slots, 4), "the turn's owner was preempted");
}

} // namespace

int main() {
    try {
        test_fifo_bypass_survives_pressure();
        test_scan_progress_and_waiter_barrier();
        test_scan_keeps_unexamined_candidates();
        test_pressure_handoff_and_fair_service();
        test_recovery_cost_does_not_lengthen_a_turn();
        test_bounded_service_under_continuous_contention();
        test_prefill_order();
        test_blocked_owner_leaves_the_prefill_permit();
        test_finished_resident_is_a_pending_release();
        test_decode_and_control_membership();
        test_decisions_queue_first();
        test_decisions_keep_the_fifo_bound();
        test_decisions_keep_their_queue_order_in_the_scan();
        test_decision_lanes();
        test_decision_prefill_turn();
        test_decisions_keep_the_prefill_bound();
        test_decision_is_last_victim();
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
