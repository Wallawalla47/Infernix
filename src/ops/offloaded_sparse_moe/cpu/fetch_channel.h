#pragma once

// Host side of the SSD expert tier's fetch channel (design §19.3.7; protocol in fetch_request.h):
// the mapped request, response, consumed and heartbeat words and the device sequence counter, and
// the responder's half of the protocol. One responder thread (the tier's agent) calls every
// method except channel(); the device publishes at most one request at a time.

#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_request.h"

#include <cstdint>
#include <span>
#include <vector>

namespace infernix::ops::offloaded_moe {

class FetchChannel {
public:
    FetchChannel();
    ~FetchChannel();
    FetchChannel(const FetchChannel&)            = delete;
    FetchChannel& operator=(const FetchChannel&) = delete;

    [[nodiscard]] MoeFetchChannel channel(int layer) const;

    struct Request {
        std::uint32_t sequence = 0;
        int layer              = 0;
        std::span<const std::int32_t> experts; // in job order; valid until the next poll
    };
    // A request published since the last one: acknowledges it (landed 0, status 0, then the
    // sequence) and returns true. Records then land through land() or the request fails.
    bool poll(Request& out);
    // Record `index` of the current request is at `record` (device-readable host memory). Landings
    // may come in any order; the device sees each prefix of landed records as soon as it is complete.
    void land(std::uint32_t index, const std::uint8_t* record);
    // Fails the current request with an errno value: the device copies nothing more of it.
    void fail(std::uint32_t status);
    // Records [0, consumed()) of the current request have been copied by the device; their slots
    // may be reused.
    [[nodiscard]] std::uint32_t consumed() const noexcept;
    [[nodiscard]] std::uint32_t landed() const noexcept { return landed_; }
    [[nodiscard]] std::uint32_t count() const noexcept { return count_; }
    // Keeps the device's waits alive: call at least every few milliseconds while a request is open.
    void beat() noexcept;

private:
    FetchRequest* request_    = nullptr; // mapped
    FetchResponse* response_  = nullptr; // mapped
    std::uint64_t* consumed_  = nullptr; // mapped
    std::uint32_t* heartbeat_ = nullptr; // mapped
    std::uint32_t* sequence_  = nullptr; // device
    std::uint32_t seen_       = 0;       // the last sequence acknowledged
    std::uint32_t count_      = 0;
    std::uint32_t landed_     = 0;
    std::uint32_t beat_       = 0;
    std::vector<std::int32_t> experts_;
    std::vector<std::uint8_t> arrived_;
};

} // namespace infernix::ops::offloaded_moe
