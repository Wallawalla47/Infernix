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
// the SSD tier an expert without a host copy is streamed too when the tier has stream slots
// (SsdReads): its record is read from the artifact into a stream slot kReadAhead layers before its
// copy, up to the slots' share per layer, and the slot is given back once the copy has landed; a
// layer walk's span thus reads each SSD-only expert once instead of once per call. Others without
// a host copy stay with the MoE's fetch channel. Results are unchanged: a streamed record is read
// exactly as a resident or staged one (ops::MoeExpertSource::prefetched).
//
// The ring's first bytes hold the chunk's slot tables (I32 [layers][experts]: slot or -1), so the
// stream owns no device memory of its own.
//
// Gated chunks (design §19.3.12, a short prompt's CPU share) copy nothing ahead: once a layer's
// routing is known, plan_layer copies only the experts the call routes that are neither resident
// nor left to the CPU, and uploads the layer's slot table with them.
//
// Staging (set_staging; the SSD tier's unpinned records, which the copy engine cannot read without
// the driver holding the host thread): up to per_part of a layer's streamed records may be unpinned.
// When the layer's copies are enqueued, the host copies them into pinned bounce slots (part l % 4,
// whose previous DMA, layer l - 4's, finished long before) and the DMA reads the bounce, still two
// layers ahead. Four parts keep the host from waiting on layer l - 2's DMA, which would hold it to
// the device's pace while it copies.

#include "core/arena.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <functional>
#include <utility>
#include <vector>

namespace infernix::models::qwen4_exp::execution {

class ExpertStream {
public:
    // A non-resident expert's pinned host record, or null when it has none (SSD-only). Stable for
    // the chunk: begin() is called after the round's boundary.
    using RecordOf = std::function<const std::uint8_t*(std::uint32_t layer, std::uint32_t expert)>;
    // Called for every record a chunk plans to copy, when it is planned: the SSD tier pins it, so the
    // record stays in place until the copies of the chunk (or a layer walk's span) have landed.
    using Hold = std::function<void(std::uint32_t layer, std::uint32_t expert)>;
    // The SSD tier's stream slots (HostExpertTier::stream_read, stream_wait, stream_release).
    struct SsdReads {
        std::function<std::int32_t(std::uint32_t layer, std::uint32_t expert)> read; // a slot, or -1: none free
        std::function<const std::uint8_t*(std::int32_t slot)> wait;                  // the landed record, or null
        std::function<void(std::int32_t slot)> release;
        std::function<std::uint32_t()> slots; // the stream slots now (a span's begin plans with them)
    };
    // Layers between a stream read's request and its copy.
    static constexpr std::uint32_t kReadAhead = 2;

    // `one_allocation`: a layer's records are one pinned allocation (the banks), so adjacent
    // records copy as one run; otherwise (the SSD tier's slots, pinned in chunks) one copy each.
    // `record_stride` is the ring's slot pitch (the largest record); `host_strides`, when given, are
    // the banks' own record strides per layer (at most the pitch: a GGUF import's records differ).
    ExpertStream(std::uint32_t layers, std::uint32_t experts, std::uint64_t record_stride, RecordOf record_of,
                 bool one_allocation, Hold hold = {}, std::vector<std::uint64_t> host_strides = {}, SsdReads ssd = {});
    ~ExpertStream();
    ExpertStream(const ExpertStream&)            = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    // Host-only records streamed through pinned bounce slots (see the file comment).
    struct Staging {
        RecordOf unpinned;                                // a non-resident expert's unpinned record, or null
        std::function<std::uint8_t*(std::uint32_t)> slot; // bounce slot i of kBounceParts * per_part (pinned)
        // Copies whole records (destination, source) and returns once they have landed.
        std::function<void(std::span<const std::pair<std::uint8_t*, const std::uint8_t*>>)> copy;
        std::uint32_t per_part = 0;
    };
    static constexpr std::uint32_t kBounceParts = 4;
    void set_staging(Staging staging);

    // The stream that writes the ring (a writer for the lender to order after its promotions).
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    // Records per ring of `bytes`: what is left after the slot tables, an even number.
    [[nodiscard]] std::uint32_t slots_in(std::size_t bytes) const noexcept;

    // Starts a chunk over `ring` with the frame tables `residency` (host I32 [layers][experts]:
    // frame or -1, as the chunk's kernels see them). On `compute`: uploads the slot tables; on the
    // copy stream, behind `compute`'s work so far, the copies of layers 0 and 1.
    // `gated`: no copies now; plan_layer enqueues each layer's. `ssd`: SSD-only experts are streamed
    // too (a layer walk's span, which reads each once for all its calls; a single call reads them
    // through the fetch channel, which overlaps the drive with the calls' work).
    void begin(DeviceSpan ring, const std::int32_t* residency, cudaStream_t compute, bool gated = false,
               bool ssd = false);
    // A gated chunk's layer `layer`, in layer order: copies (and the slot table of) every expert with
    // columns[e] > 0 that is not resident and has cpu[e] == 0, up to the layer's half, behind the
    // release of layer - 2's half. Returns how many it copies.
    std::uint32_t plan_layer(std::uint32_t layer, std::span<const std::int32_t> columns,
                             std::span<const std::uint8_t> cpu);
    [[nodiscard]] bool gated() const noexcept { return gated_; }
    // Whether the active (ungated) chunk copies every expert of `layer` that was not resident when it
    // began: its MoE then has nothing to stage.
    [[nodiscard]] bool complete(std::uint32_t layer) const noexcept { return active_ && !gated_ && complete_[layer] != 0; }
    // Records per half of the active ring.
    [[nodiscard]] std::uint32_t half() const noexcept { return half_; }
    [[nodiscard]] bool active() const noexcept { return active_; }
    // Layer l's slot table (device I32 [experts]) and the records' base.
    [[nodiscard]] const std::int32_t* slots(std::uint32_t layer) const noexcept;
    [[nodiscard]] const std::uint8_t* base() const noexcept { return records_; }
    // The slot pitch (MoeExpertSource::prefetch_stride).
    [[nodiscard]] std::uint64_t pitch() const noexcept { return stride_; }
    // Before layer `layer`'s experts: `compute` waits for the layer's copies.
    void before_experts(std::uint32_t layer, cudaStream_t compute);
    // After the layer's experts are enqueued on `compute`: the half is free once they finish; the
    // copies of layer + 2 follow.
    void after_experts(std::uint32_t layer, cudaStream_t compute);
    // After the chunk's last layer: `compute` is ordered after every copy (each was waited for). Gives
    // every stream slot back once its read and copy are done.
    void end() noexcept;

    struct Stats {
        std::uint64_t streamed = 0; // expert records copied
        std::uint64_t bytes    = 0; // bytes copied
        std::uint64_t copies   = 0; // DMA calls (runs of consecutive experts)
        std::uint64_t ssd      = 0; // of the records, those read into stream slots
        std::uint64_t staged   = 0; // of the streamed records, unpinned ones bounced
    };
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    void enqueue(std::uint32_t layer);
    // Requests the stream reads of `layer` (once per chunk).
    void request_reads(std::uint32_t layer);
    // Gives back the stream slots of layers whose copies have landed (all of the oldest such layer
    // when `wait`, waiting for its copies); whether any came back.
    bool release_landed(bool wait);
    // Whether expert e of layer l is streamed: not resident, room in the half, and a pinned record
    // (or an unpinned one while the layer's staging has room, counted in `staged`).
    bool takes(std::uint32_t layer, std::uint32_t e, std::uint32_t taken, std::uint32_t& staged) const;

    std::uint32_t experts_ = 0, layers_ = 0;
    std::uint64_t stride_  = 0;
    std::vector<std::uint64_t> host_strides_; // per layer, or empty: stride_
    RecordOf record_of_;
    Hold hold_;
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
    std::vector<std::uint8_t> complete_; // per layer: every non-resident expert copied (complete())
    bool active_    = false;
    bool uploading_ = false;
    bool gated_     = false;
    const std::int32_t* residency_ = nullptr; // host [layers][experts] of the active chunk
    // Stream reads: per layer the SSD-only experts planned (expert, ring slot), their stream slots
    // (-1: not requested yet), whether requested; the layers whose copies hold slots, oldest first.
    SsdReads ssd_;
    std::vector<std::vector<std::pair<std::uint32_t, std::int32_t>>> ssd_planned_;
    std::vector<std::vector<std::int32_t>> ssd_slots_;
    std::vector<std::uint8_t> ssd_requested_;
    std::vector<std::uint32_t> ssd_held_;
    Staging staging_;
    cudaEvent_t bounced_[kBounceParts] = {}; // each bounce part's last DMA
    bool bounce_used_[kBounceParts]    = {};
    std::vector<std::pair<std::uint8_t*, const std::uint8_t*>> staged_; // (ring slot, unpinned record)
    std::vector<std::pair<std::uint8_t*, const std::uint8_t*>> fill_;
    Stats stats_;
};

} // namespace infernix::models::qwen4_exp::execution
