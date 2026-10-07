#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace infernix::runtime::prefix_cache {

// Machine model used only to rank admission sources and to value snapshots for eviction. It never
// decides feasibility. The Program fills it from the calibrated prefill coefficients and the
// transfer bandwidth measured at startup.
struct CacheCostModel {
    double chunk_seconds          = 0.0;
    std::uint32_t chunk_tokens    = 2048;
    double token_seconds          = 1.0 / 4000.0;
    double attention_pair_seconds = 0.0;
    double h2d_bytes_per_second   = 50.0e9;
    double transfer_batch_seconds = 20.0e-6;
    // Saturation of the per-call cost with the call's width, for models whose fixed call cost is
    // weight traffic that grows with the distinct routes a call touches (offloaded experts): each
    // token adds a fraction rho of the remaining route cost. 0 charges every call chunk_seconds.
    double call_route_fraction = 0.0;

    // Fixed cost of one prefill call of `tokens` tokens: chunk_seconds * (1 - (1 - rho)^tokens).
    [[nodiscard]] double call_seconds(std::uint32_t tokens) const noexcept {
        if (tokens == 0) { return 0.0; }
        if (call_route_fraction <= 0.0) { return chunk_seconds; }
        const double rho = std::min(call_route_fraction, 1.0);
        if (rho >= 1.0) { return chunk_seconds; }
        return -chunk_seconds * std::expm1(static_cast<double>(tokens) * std::log1p(-rho));
    }

    // Prefill of `tokens` tokens appended after `base` reused tokens, in calls of chunk_tokens.
    [[nodiscard]] double prefill_seconds(std::uint32_t base, std::uint32_t tokens) const noexcept {
        if (tokens == 0) { return 0.0; }
        const double s            = static_cast<double>(tokens);
        const double pairs        = static_cast<double>(base) * s + s * (s + 1.0) / 2.0;
        const std::uint32_t chunk = std::max<std::uint32_t>(chunk_tokens, 1U);
        double calls_seconds      = 0.0;
        if (call_route_fraction <= 0.0) {
            calls_seconds = static_cast<double>((tokens + chunk - 1U) / chunk) * chunk_seconds;
        } else {
            calls_seconds = static_cast<double>(tokens / chunk) * call_seconds(chunk) +
                            call_seconds(tokens % chunk);
        }
        return calls_seconds + s * token_seconds + pairs * attention_pair_seconds;
    }

    [[nodiscard]] double restore_seconds(std::uint64_t bytes) const noexcept {
        if (bytes == 0) { return 0.0; }
        return transfer_batch_seconds + static_cast<double>(bytes) / h2d_bytes_per_second;
    }
};

} // namespace infernix::runtime::prefix_cache
