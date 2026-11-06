#pragma once

// The drafter pool (LoadOptions::mtp_expert_pool, candidates §22 VM-01 variant B): an MTP drafter's routed experts in
// a pool of device slots, shared by Qwen3.8-Flash-Next and the Qwen3.5-122B. Their banks stay in pinned host memory
// (LoadOptions::mtp_expert_slots) and each draft step routes among the experts the pool holds
// (ops::moe_route_pooled), so drafts may change while outputs do not (verification decides) and the frames the
// bank took go to the expert cache.
//
// The pool follows the drafter's demand. After each decode or verification round (outside any capture: Flash-Next's
// settle_round, the 122B's round end) step() drains the demand counts the draft steps added (into pinned memory, read once the
// drain has run), keeps a decayed count per expert, and swaps the least demanded slot's expert for the most
// demanded expert without a slot when that one leads by a margin, at most kMaxInFlight at a time.
//
// A swap never races a reader: the departing expert's table entry is cleared on the compute stream; the copy
// stream waits for that point (every draft step that could read the slot was enqueued before it) and copies the
// new expert's rows; the first step() that finds the copy landed publishes its entry on the compute stream. A
// swap's slot is out of the route meanwhile, so at least top_k + kMaxInFlight slots keep top_k routable.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::runtime::expert_offload {

class DrafterExpertPool {
public:
    static constexpr std::int32_t kMaxInFlight = 2;

    // Device bytes of `slots` slots of `expert_bytes` (one expert's gate/up and down rows, codes and scales) over
    // `experts` experts: the four bank regions, the slot table and the demand counts, each 256-byte aligned.
    [[nodiscard]] static std::size_t device_bytes(std::uint64_t expert_bytes, std::int32_t slots, std::int32_t experts);

    // The bytes of one expert of row-split banks shaped as gate_up [E * 2I, H] and down [E * H, I] (Q2/Q4/Q8 with
    // FP16 group scales, as ops::resident_moe_experts reads them).
    [[nodiscard]] static std::uint64_t expert_bytes(const Weight& gate_up, const Weight& down, std::int32_t intermediate,
                                                    std::int32_t hidden);

    // gate_up / down: the drafter's banks in pinned host memory; `memory` holds device_bytes(...) bytes. Fills the
    // slots with experts 0..slots-1 on `stream` (complete once the stream is).
    DrafterExpertPool(const Weight& gate_up, const Weight& down, std::int32_t experts, std::int32_t intermediate,
                      std::int32_t hidden, std::int32_t slots, void* memory, std::size_t bytes, cudaStream_t stream);
    ~DrafterExpertPool();
    DrafterExpertPool(const DrafterExpertPool&)            = delete;
    DrafterExpertPool& operator=(const DrafterExpertPool&) = delete;

    // The pool's banks ([slots * 2I, H] and [slots * H, I]), its slot table (device I32 [E]: slot or -1) and its
    // demand counts (device I32 [E + 1], ops::moe_route_pooled).
    [[nodiscard]] const Weight& gate_up() const noexcept { return pool_gate_up_; }
    [[nodiscard]] const Weight& down() const noexcept { return pool_down_; }
    [[nodiscard]] Tensor slot_of() const noexcept { return Tensor(table_, DType::I32, {experts_}); }
    [[nodiscard]] std::int32_t* demand() const noexcept { return demand_; }
    [[nodiscard]] std::int32_t slots() const noexcept { return slots_; }

    // Between rounds, on the compute stream (no capture): publishes landed swaps, folds in drained demand,
    // starts swaps and the next drain.
    void step(cudaStream_t stream);

    struct Stats {
        std::uint64_t picks = 0;  // drained unrestricted top-k picks
        std::uint64_t misses = 0; // of them, experts without a slot
        std::uint64_t swaps = 0;  // swaps started
    };
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    struct Segment {
        const std::uint8_t* source = nullptr; // pinned bank base
        std::uint8_t* pool         = nullptr; // device slot 0
        std::size_t bytes          = 0;       // one expert's
    };
    struct Swap {
        bool active        = false;
        std::int32_t slot  = -1;
        std::int32_t expert = -1;
        cudaEvent_t landed = nullptr;
    };
    static constexpr std::int32_t kInFlight = -2; // expert_slot_: its copy has not been published yet

    void copy_in(std::int32_t expert, std::int32_t slot, cudaStream_t stream);

    std::int32_t experts_ = 0, slots_ = 0;
    std::array<Segment, 4> segments_{}; // gate/up codes and scales, down codes and scales
    Weight pool_gate_up_, pool_down_;
    std::int32_t* table_  = nullptr;
    std::int32_t* demand_ = nullptr;
    // Host view: each expert's published slot (-1 none, kInFlight copying), each slot's expert.
    std::vector<std::int32_t> expert_slot_, slot_expert_;
    std::vector<float> score_;
    PinnedHostBuffer drained_;
    cudaEvent_t drain_done_ = nullptr;
    bool drain_pending_     = false;
    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t cleared_      = nullptr;
    std::array<Swap, kMaxInFlight> swaps_{};
    Stats stats_;
};

} // namespace infernix::runtime::expert_offload
