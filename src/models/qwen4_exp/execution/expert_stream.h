#pragma once

// Prefill expert streaming (docs/maintainer/qwen3_8-flash-next-design.md §19.3.8 F2): the records of a
// prefill chunk's non-resident experts reach the device by copy-engine DMA, two layers ahead, into a
// ring of borrowed device memory (F3: frames lent by the expert cache), instead of through the MoE's
// staging kernel, which reads pinned memory with the SMs at about two thirds of the link rate and
// cannot start before the layer's router.
//
// A wide chunk routes nearly every expert of every layer, so a layer's copies do not wait for its
// routing: they take the layer's non-resident experts in ascending id, up to one half of the ring.
// Layer l uses half l % 2. Its copies wait until layer l - 2's experts finished reading that half,
// and layer l's MoE waits for them (one event per layer each way). Experts beyond the half's
// capacity, and any the chunk does not route, stay with the MoE (staged as before, or unused). With
// the SSD tier an expert without a host copy is not streamed (the MoE's fetch channel serves it).
// Results are unchanged: a streamed record is read exactly as a resident or staged one
// (ops::MoeExpertSource::prefetched).
//
// The ring's first bytes hold the chunk's slot tables (I32 [layers][experts]: slot or -1), so the
// stream owns no device memory of its own.
//
// Gated chunks (design §19.3.12, a short prompt's CPU share) copy nothing ahead: once a layer's
// routing is known, plan_layer copies only the experts the call routes that are neither resident
// nor left to the CPU, and uploads the layer's slot table with them.

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <functional>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

class ExpertStream {
public:
    // A non-resident expert's pinned host record, or null when it has none (SSD-only). Stable for
    // the chunk: begin() is called after the round's boundary.
    using RecordOf = std::function<const std::uint8_t*(std::uint32_t layer, std::uint32_t expert)>;

    // `one_allocation`: a layer's records are one pinned allocation (the banks), so adjacent
    // records copy as one run; otherwise (the SSD tier's slots, pinned in chunks) one copy each.
    ExpertStream(std::uint32_t layers, std::uint32_t experts, std::uint64_t record_stride, RecordOf record_of,
                 bool one_allocation);
    ~ExpertStream();
    ExpertStream(const ExpertStream&)            = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    // The stream that writes the ring (a writer for the lender to order after its promotions).
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    // Records per ring of `bytes`: what is left after the slot tables, an even number.
    [[nodiscard]] std::uint32_t slots_in(std::size_t bytes) const noexcept;

    // Starts a chunk over `ring` with the frame tables `residency` (host I32 [layers][experts]:
    // frame or -1, as the chunk's kernels see them). On `compute`: uploads the slot tables; on the
    // copy stream, behind `compute`'s work so far, the copies of layers 0 and 1.
    // `gated`: no copies now; plan_layer enqueues each layer's.
    void begin(DeviceSpan ring, const std::int32_t* residency, cudaStream_t compute, bool gated = false);
    // A gated chunk's layer `layer`, in layer order: copies (and the slot table of) every expert with
    // columns[e] > 0 that is not resident and has cpu[e] == 0, up to the layer's half, behind the
    // release of layer - 2's half. Returns how many it copies.
    std::uint32_t plan_layer(std::uint32_t layer, std::span<const std::int32_t> columns,
                             std::span<const std::uint8_t> cpu);
    [[nodiscard]] bool gated() const noexcept { return gated_; }
    // Records per half of the active ring.
    [[nodiscard]] std::uint32_t half() const noexcept { return half_; }
    [[nodiscard]] bool active() const noexcept { return active_; }
    // Layer l's slot table (device I32 [experts]) and the records' base.
    [[nodiscard]] const std::int32_t* slots(std::uint32_t layer) const noexcept;
    [[nodiscard]] const std::uint8_t* base() const noexcept { return records_; }
    // Before layer `layer`'s experts: `compute` waits for the layer's copies.
    void before_experts(std::uint32_t layer, cudaStream_t compute);
    // After the layer's experts are enqueued on `compute`: the half is free once they finish; the
    // copies of layer + 2 follow.
    void after_experts(std::uint32_t layer, cudaStream_t compute);
    // After the chunk's last layer: `compute` is ordered after every copy (each was waited for).
    void end() noexcept { active_ = false; }

    struct Stats {
        std::uint64_t streamed = 0; // expert records copied
        std::uint64_t copies   = 0; // DMA calls (runs of consecutive experts)
    };
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    void enqueue(std::uint32_t layer);

    std::uint32_t experts_ = 0, layers_ = 0;
    std::uint64_t stride_  = 0;
    RecordOf record_of_;
    bool one_allocation_ = true;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t uploaded_ = nullptr;
    std::vector<cudaEvent_t> landed_, consumed_;
    PinnedHostBuffer tables_host_{1};
    std::int32_t* tables_     = nullptr; // device, at the ring's start
    std::uint8_t* records_    = nullptr; // device, after the tables
    std::uint32_t half_       = 0;       // slots per half
    std::vector<std::uint8_t> streams_;  // per layer: whether it has copies
    bool active_    = false;
    bool uploading_ = false;
    bool gated_     = false;
    const std::int32_t* residency_ = nullptr; // host [layers][experts] of the active chunk
    Stats stats_;
};

} // namespace ninfer::models::qwen4_exp::execution
