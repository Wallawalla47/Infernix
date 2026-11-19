#pragma once

// The CPU miss service answers a request, then counts it (cpu/miss_service.cpp publishes the done word before
// served_experts() moves), so right after the device finishes a call the count may still trail by a moment: a thread
// preempted between the two stores (a loaded machine) left a check made at once without the request it served.

#include "ops/offloaded_sparse_moe/cpu/miss_service.h"

#include <chrono>
#include <cstdint>
#include <thread>

namespace infernix::test {

// served_experts() once it exceeds `before`, waiting up to 30 s for the count to catch up with the device; the count
// at the deadline otherwise (a service that served nothing).
inline std::uint64_t served_experts_past(const ops::offloaded_moe::CpuMissService& service, std::uint64_t before) {
    const auto until  = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    std::uint64_t now = service.served_experts();
    while (now <= before && std::chrono::steady_clock::now() < until) {
        std::this_thread::yield();
        now = service.served_experts();
    }
    return now;
}

} // namespace infernix::test
