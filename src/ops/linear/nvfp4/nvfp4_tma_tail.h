#pragma once

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// ZK-09: the remainder past an NVFP4 A4 TMA linear_add's last whole 256-token tile runs on the MMA tiles instead of a
// partly filled TMA tile when that tile would add a wave, up to this many columns (the measured range). The MMA and
// TMA routes give a column the same bits (one K order per output). RTX 5090, nvfp4_linear_add_bench (a4, N = 5120,
// 2026-11-12), against the padded tile: K = 17408 1,025-1,088 columns -12.7 .. -14.6 %, 2,049-2,112 -6.3 .. -9.1 %;
// K = 6144 -9.4 .. -11.1 % and -3.4 .. -6.4 %; equal output hashes at every width. The 27B's prefill (NVFP4, two
// A B B A blocks): 1,025-1,088 tokens -3.0 .. -3.9 %, 2,049-2,112 -0.2 .. -0.5 %.
inline constexpr std::int32_t kNvfp4TmaTailMaxColumns = 64;

// The device's SM count: the 256-token TMA linear_add's CTA (its 256 x 136 BF16 output tile alone is 68 KB of shared
// memory) owns its SM, so a launch of n CTAs runs in ceil(n / SMs) waves.
inline std::int64_t nvfp4_tma_sms() {
    static const std::int64_t sms = [] {
        int device = 0, count = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return static_cast<std::int64_t>(count);
    }();
    return sms;
}

// The remainder of a `tokens`-column linear_add of `rows` output rows whose TMA route reads 128-row x 256-token
// tiles, when the whole tiles before it (more than `min_wide` columns) still take that route and the remainder's
// partly filled tile would add a wave to them; else 0. A tile that fits in the whole tiles' last wave costs nothing
// there (5,120 rows: 769-832 columns as fast as 768 on four tiles, 1,281-1,344 as 1,280 on six), and a separate MMA
// call behind it would only add its own time (op +7 .. +25 %, the 27B's prefill +0.6 .. +1.8 %).
inline std::int32_t nvfp4_tma_tail(std::int32_t tokens, std::int32_t rows, std::int32_t min_wide) {
    const std::int32_t tail = tokens % 256;
    if (tail == 0 || tail > kNvfp4TmaTailMaxColumns || tokens - tail <= min_wide) { return 0; }
    const std::int64_t row_tiles = rows / 128, whole = tokens / 256, sms = nvfp4_tma_sms();
    const auto waves = [&](std::int64_t tiles) { return (row_tiles * tiles + sms - 1) / sms; };
    return waves(whole + 1) > waves(whole) ? tail : 0;
}

// The remainder's one-term plane, after the `wide` whole-tile columns: their 256-token tiled scales end where
// row-major ones for the same columns would, so the remainder's row-major scales follow them and its codes follow
// theirs, and the two calls share no bytes.
inline Nvfp4A4Workspace nvfp4_a4_tail_plane(Nvfp4A4Workspace planes, std::int32_t input_rows, std::int32_t wide) {
    const std::size_t code_bytes  = static_cast<std::size_t>(wide) * (static_cast<std::size_t>(input_rows) / 2);
    const std::size_t scale_bytes = static_cast<std::size_t>(wide) * (static_cast<std::size_t>(input_rows) / 16);
    planes.codes += code_bytes;
    planes.scales += scale_bytes;
    planes.scale_bytes -= scale_bytes;
    return planes;
}

} // namespace infernix::ops::detail
