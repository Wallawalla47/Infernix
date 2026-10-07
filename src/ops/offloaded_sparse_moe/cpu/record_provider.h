#pragma once

// Records the CPU miss service computes but no host bank holds (design §19.3.7, the SSD expert
// tier): the service asks for each such expert as soon as it reads a request, computes the others,
// then waits for these and computes them from the landed bytes. Implemented by the Program's tier.

#include <cstdint>

namespace infernix::ops::offloaded_moe {

class RecordProvider {
public:
    virtual ~RecordProvider() = default;

    // Starts reading expert `expert` of layer `layer`; returns a ticket for wait and done. Called by
    // the service thread only, between rounds' boundaries.
    virtual std::uint32_t demand(int layer, int expert) noexcept = 0;
    // Whether wait() would return at once (the record landed or its read failed).
    [[nodiscard]] virtual bool landed(std::uint32_t ticket) const noexcept = 0;
    // Blocks until the ticket's record has landed and returns it, or returns null with a nonzero
    // `status` (an errno value) when it cannot be read.
    virtual const std::uint8_t* wait(std::uint32_t ticket, std::uint32_t& status) noexcept = 0;
    // The service has finished reading the record.
    virtual void done(std::uint32_t ticket) noexcept = 0;
};

} // namespace infernix::ops::offloaded_moe
