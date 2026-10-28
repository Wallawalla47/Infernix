#include "runtime/expert_offload/drafter_pool.h"

#include "core/device.h"
#include "infernix/ops/offloaded_sparse_moe.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::runtime::expert_offload {
namespace {

// An expert's demand decays by kDecay per drain (one per round or two: a time constant of about 20 drains, a few
// hundred draft picks at one lane). A candidate takes a slot when its demand exceeds the slot expert's by kLead
// times plus kLeadPicks picks, so near-equal experts do not trade places on noise.
constexpr float kDecay     = 0.95F;
constexpr float kLead      = 1.25F;
constexpr float kLeadPicks = 1.0F;

constexpr std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// A row-split bank's row bytes, as ops::resident_moe_experts reads them: 16-byte chunks of codes, then FP16
// group scales.
std::size_t code_row_bytes(const Weight& w) {
    const auto k = static_cast<std::size_t>(w.padded_shape[1]);
    switch (w.qtype) {
    case QType::Q2_G64_FP16:
        return k / 4;
    case QType::Q4_G64_FP16:
        return k / 2;
    case QType::Q8_G32_FP16:
        return k;
    default:
        throw std::invalid_argument(
            "drafter expert pool: the banks must be row-split Q2_G64_FP16, Q4_G64_FP16 or Q8_G32_FP16");
    }
}

std::size_t scale_row_bytes(const Weight& w) {
    const auto k = static_cast<std::size_t>(w.padded_shape[1]);
    return (w.qtype == QType::Q8_G32_FP16 ? k / 32 : k / 64) * 2;
}

void require_bank(const Weight& w, std::int64_t rows, const char* name) {
    if (w.layout != QuantLayout::RowSplit || w.qdata == nullptr || w.scales == nullptr || w.qhigh != nullptr ||
        w.n != rows) {
        throw std::invalid_argument(std::string("drafter expert pool: the ") + name + " bank must be row-split with " +
                                    std::to_string(rows) + " rows");
    }
    (void)code_row_bytes(w);
}

} // namespace

std::size_t DrafterExpertPool::device_bytes(std::uint64_t expert_bytes, std::int32_t slots, std::int32_t experts) {
    // The four bank regions are each 256-byte aligned (at most 4 x 255 bytes beyond the experts' bytes).
    return align256(static_cast<std::size_t>(expert_bytes) * static_cast<std::size_t>(slots)) + 4 * 256 +
           align256(sizeof(std::int32_t) * static_cast<std::size_t>(experts)) +
           align256(sizeof(std::int32_t) * (static_cast<std::size_t>(experts) + 1));
}

std::uint64_t DrafterExpertPool::expert_bytes(const Weight& gate_up, const Weight& down, std::int32_t intermediate,
                                              std::int32_t hidden) {
    return 2ULL * static_cast<std::uint64_t>(intermediate) * (code_row_bytes(gate_up) + scale_row_bytes(gate_up)) +
           static_cast<std::uint64_t>(hidden) * (code_row_bytes(down) + scale_row_bytes(down));
}

DrafterExpertPool::DrafterExpertPool(const Weight& gate_up, const Weight& down, std::int32_t experts,
                                     std::int32_t intermediate, std::int32_t hidden, std::int32_t slots, void* memory,
                                     std::size_t bytes, cudaStream_t stream)
    : experts_(experts), slots_(slots), expert_slot_(static_cast<std::size_t>(experts), -1),
      slot_expert_(static_cast<std::size_t>(slots), -1), score_(static_cast<std::size_t>(experts), 0.0F),
      drained_(sizeof(std::int32_t) * (static_cast<std::size_t>(experts) + 1)) {
    require_bank(gate_up, 2LL * experts * intermediate, "gate/up");
    require_bank(down, static_cast<std::int64_t>(experts) * hidden, "down");
    if (gate_up.qtype != down.qtype || slots <= kMaxInFlight || slots >= experts) {
        throw std::invalid_argument("drafter expert pool: one codec for both banks and fewer slots than experts");
    }
    const std::size_t gate_rows = 2ULL * static_cast<std::size_t>(intermediate);
    const std::size_t down_rows = static_cast<std::size_t>(hidden);
    const std::size_t sizes[4]  = {gate_rows * code_row_bytes(gate_up), gate_rows * scale_row_bytes(gate_up),
                                   down_rows * code_row_bytes(down), down_rows * scale_row_bytes(down)};
    const void* sources[4]      = {gate_up.qdata, gate_up.scales, down.qdata, down.scales};
    auto* at        = static_cast<std::uint8_t*>(memory);
    auto* const end = at + bytes;
    for (std::size_t i = 0; i < segments_.size(); ++i) {
        segments_[i] = {static_cast<const std::uint8_t*>(sources[i]), at, sizes[i]};
        at += align256(sizes[i] * static_cast<std::size_t>(slots));
    }
    table_ = reinterpret_cast<std::int32_t*>(at);
    at += align256(sizeof(std::int32_t) * static_cast<std::size_t>(experts));
    demand_ = reinterpret_cast<std::int32_t*>(at);
    at += align256(sizeof(std::int32_t) * (static_cast<std::size_t>(experts) + 1));
    if (at > end) { throw std::logic_error("drafter expert pool: its device plan is smaller than its layout"); }

    const auto pooled = [&](const Weight& bank, const Segment& codes, const Segment& scales, std::size_t rows) {
        Weight w            = bank;
        w.qdata             = codes.pool;
        w.scales            = scales.pool;
        w.payload           = codes.pool;
        w.payload_bytes     = (codes.bytes + scales.bytes) * static_cast<std::size_t>(slots);
        w.n                 = static_cast<std::int32_t>(rows * static_cast<std::size_t>(slots));
        w.shape[0]          = w.n;
        w.padded_shape[0]   = w.n;
        return w;
    };
    pool_gate_up_ = pooled(gate_up, segments_[0], segments_[1], gate_rows);
    pool_down_    = pooled(down, segments_[2], segments_[3], down_rows);

    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&cleared_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&drain_done_, cudaEventDisableTiming));
    for (auto& swap : swaps_) { CUDA_CHECK(cudaEventCreateWithFlags(&swap.landed, cudaEventDisableTiming)); }

    // Experts 0..slots-1 to start; the first rounds' demand moves the pool to the drafter's hot set.
    std::vector<std::int32_t> table(static_cast<std::size_t>(experts), -1);
    for (std::int32_t s = 0; s < slots; ++s) {
        copy_in(s, s, stream);
        table[static_cast<std::size_t>(s)]        = s;
        expert_slot_[static_cast<std::size_t>(s)] = s;
        slot_expert_[static_cast<std::size_t>(s)] = s;
    }
    CUDA_CHECK(cudaMemcpyAsync(table_, table.data(), table.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(demand_, 0, sizeof(std::int32_t) * (static_cast<std::size_t>(experts) + 1), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // `table` is pageable host memory
}

DrafterExpertPool::~DrafterExpertPool() {
    if (copy_stream_ != nullptr) { cudaStreamSynchronize(copy_stream_); }
    for (auto& swap : swaps_) {
        if (swap.landed != nullptr) { cudaEventDestroy(swap.landed); }
    }
    if (drain_done_ != nullptr) { cudaEventSynchronize(drain_done_); cudaEventDestroy(drain_done_); }
    if (cleared_ != nullptr) { cudaEventDestroy(cleared_); }
    if (copy_stream_ != nullptr) { cudaStreamDestroy(copy_stream_); }
}

void DrafterExpertPool::copy_in(std::int32_t expert, std::int32_t slot, cudaStream_t stream) {
    for (const Segment& s : segments_) {
        CUDA_CHECK(cudaMemcpyAsync(s.pool + static_cast<std::size_t>(slot) * s.bytes,
                                   s.source + static_cast<std::size_t>(expert) * s.bytes, s.bytes, cudaMemcpyHostToDevice,
                                   stream));
    }
}

void DrafterExpertPool::step(cudaStream_t stream) {
    ops::MoePoolWrites writes;
    const auto write = [&](std::int32_t expert, std::int32_t value) {
        writes.expert[writes.count] = expert;
        writes.value[writes.count]  = value;
        ++writes.count;
    };
    // Copies that landed: their experts join the route.
    for (Swap& swap : swaps_) {
        if (!swap.active) { continue; }
        const cudaError_t state = cudaEventQuery(swap.landed);
        if (state == cudaErrorNotReady) { continue; }
        CUDA_CHECK(state);
        swap.active                                     = false;
        expert_slot_[static_cast<std::size_t>(swap.expert)] = swap.slot;
        write(swap.expert, swap.slot);
    }
    // The last drain's counts, once it has run.
    if (drain_pending_) {
        const cudaError_t state = cudaEventQuery(drain_done_);
        if (state != cudaErrorNotReady) {
            CUDA_CHECK(state);
            drain_pending_     = false;
            const auto* counts = static_cast<const std::int32_t*>(drained_.data());
            std::uint64_t picks = 0;
            for (std::int32_t e = 0; e < experts_; ++e) {
                const std::int32_t c              = counts[e];
                score_[static_cast<std::size_t>(e)] = score_[static_cast<std::size_t>(e)] * kDecay + static_cast<float>(c);
                picks += static_cast<std::uint64_t>(c);
            }
            stats_.picks += picks;
            stats_.misses += static_cast<std::uint64_t>(counts[experts_]);
        }
    }
    // New swaps into the free swap records: the most demanded expert without a slot against the least demanded
    // published slot.
    std::array<Swap*, kMaxInFlight> started{};
    std::size_t starts = 0;
    for (Swap& swap : swaps_) {
        if (swap.active) { continue; }
        std::int32_t candidate = -1;
        float best             = 0.0F;
        for (std::int32_t e = 0; e < experts_; ++e) {
            const float s = score_[static_cast<std::size_t>(e)];
            if (expert_slot_[static_cast<std::size_t>(e)] == -1 && s > best) {
                best      = s;
                candidate = e;
            }
        }
        std::int32_t victim = -1;
        float low           = std::numeric_limits<float>::infinity();
        for (std::int32_t s = 0; s < slots_; ++s) {
            const std::int32_t e = slot_expert_[static_cast<std::size_t>(s)];
            if (e >= 0 && expert_slot_[static_cast<std::size_t>(e)] == s && score_[static_cast<std::size_t>(e)] < low) {
                low    = score_[static_cast<std::size_t>(e)];
                victim = s;
            }
        }
        if (candidate < 0 || victim < 0 || !(best > kLead * low + kLeadPicks)) { break; }
        const std::int32_t leaving = slot_expert_[static_cast<std::size_t>(victim)];
        write(leaving, -1);
        expert_slot_[static_cast<std::size_t>(leaving)]   = -1;
        expert_slot_[static_cast<std::size_t>(candidate)] = kInFlight;
        slot_expert_[static_cast<std::size_t>(victim)]    = candidate;
        swap.active = true;
        swap.slot   = victim;
        swap.expert = candidate;
        started[starts++] = &swap;
        ++stats_.swaps;
    }
    // The clears go first on the compute stream; each copy waits for them (and so for every draft step that
    // could still read its slot), and publishing waits for a later step that finds the copy landed.
    ops::moe_pool_write(table_, writes, stream);
    if (starts > 0) {
        CUDA_CHECK(cudaEventRecord(cleared_, stream));
        CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, cleared_, 0));
        for (std::size_t i = 0; i < starts; ++i) {
            copy_in(started[i]->expert, started[i]->slot, copy_stream_);
            CUDA_CHECK(cudaEventRecord(started[i]->landed, copy_stream_));
        }
    }
    if (!drain_pending_) {
        ops::moe_pool_drain(demand_, static_cast<std::int32_t*>(drained_.data()), experts_ + 1, stream);
        CUDA_CHECK(cudaEventRecord(drain_done_, stream));
        drain_pending_ = true;
    }
}

} // namespace infernix::runtime::expert_offload
