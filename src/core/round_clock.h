#pragma once

// TEMPORARY (INFERNIX_TMP_REPLAY_BENCH, candidates §8.5 CF-03): the replay bench's GPU round clock.
// Each round records an event pair on its stream, at its submission and after its commit's work; a
// pair's elapsed time is the device time from the round's first work (or the previous round's last)
// to its own last, so host gaps between rounds are not counted, while a round's own waits (CPU
// expert results, host reads inside the round) are. Pairs are read once both events completed.

#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace infernix {

class RoundClock {
public:
    static constexpr std::size_t kSlots = 8; // rounds in flight before begin() waits for the oldest

    RoundClock() {
        for (auto& pair : pairs_) {
            CUDA_CHECK(cudaEventCreate(&pair.begin));
            CUDA_CHECK(cudaEventCreate(&pair.end));
        }
    }
    ~RoundClock() {
        for (auto& pair : pairs_) {
            (void)cudaEventDestroy(pair.begin);
            (void)cudaEventDestroy(pair.end);
        }
    }
    RoundClock(const RoundClock&)            = delete;
    RoundClock& operator=(const RoundClock&) = delete;

    // A round starts on `stream`. A round begun and never ended (discarded) is dropped.
    void begin(cudaStream_t stream) {
        harvest(false);
        Pair& pair = pairs_[next_ % kSlots];
        if (pair.state == State::Recorded) { read(pair, true); }
        CUDA_CHECK(cudaEventRecord(pair.begin, stream));
        pair.state = State::Open;
        open_      = next_ % kSlots;
        ++next_;
    }

    // The open round's last work is on `stream`.
    void end(cudaStream_t stream) {
        if (open_ == kNone) { return; }
        Pair& pair = pairs_[open_];
        CUDA_CHECK(cudaEventRecord(pair.end, stream));
        pair.state = State::Recorded;
        open_      = kNone;
    }

    // Every recorded round read (waits for them).
    void drain() { harvest(true); }

    [[nodiscard]] std::uint64_t rounds() const noexcept { return rounds_; }
    [[nodiscard]] double milliseconds() const noexcept { return milliseconds_; }

private:
    enum class State : std::uint8_t { Free, Open, Recorded };
    struct Pair {
        cudaEvent_t begin = nullptr, end = nullptr;
        State state       = State::Free;
    };
    static constexpr std::size_t kNone = kSlots;

    void read(Pair& pair, bool wait) {
        if (wait) {
            CUDA_CHECK(cudaEventSynchronize(pair.end));
        } else {
            const cudaError_t status = cudaEventQuery(pair.end);
            if (status == cudaErrorNotReady) { return; }
            CUDA_CHECK(status);
        }
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, pair.begin, pair.end));
        milliseconds_ += static_cast<double>(ms);
        ++rounds_;
        pair.state = State::Free;
    }

    void harvest(bool wait) {
        for (std::size_t i = 0; i < kSlots; ++i) {
            if (pairs_[i].state == State::Recorded) { read(pairs_[i], wait); }
            // An open pair is dropped when its slot comes round again (begin overwrites it).
        }
    }

    std::array<Pair, kSlots> pairs_{};
    std::size_t next_ = 0;
    std::size_t open_ = kNone;
    std::uint64_t rounds_ = 0;
    double milliseconds_  = 0.0;
};

} // namespace infernix
