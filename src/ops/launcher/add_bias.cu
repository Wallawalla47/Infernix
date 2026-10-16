#include "ops/launcher/add_bias.h"

#include "ops/common/math.h"
#include "ops/kernel/add_bias.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace infernix::ops::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    constexpr int block     = 256;
    const std::int64_t n    = x.numel();
    const std::int64_t rows = n / x.ne[0];
    const auto addresses =
        reinterpret_cast<std::uintptr_t>(bias.data) | reinterpret_cast<std::uintptr_t>(x.data);
    // One 16-byte pack per thread over the whole tensor (RTX 5090, 2026-10-16, against the earlier
    // per-row packs and the large-domain BF16x2 stream: [4304,4096] 19.3 -> 15.1 us, [4304,65536]
    // 722.9 -> 649.2 us, [2048,16384] -13 %, [3456,65536] -2.6 %; launch-bound shapes unchanged).
    if ((x.ne[0] % 8) == 0 && (addresses & (alignof(Bf16x8Pack) - 1)) == 0 &&
        n / 8 <= std::numeric_limits<std::uint32_t>::max()) {
        const auto total = static_cast<std::uint32_t>(n / 8);
        add_bias_bf16x8_stream_kernel<block>
            <<<static_cast<unsigned>(div_up(static_cast<std::int64_t>(total), static_cast<std::int64_t>(block))),
               block, 0, stream>>>(static_cast<const Bf16x8Pack*>(bias.data), static_cast<Bf16x8Pack*>(x.data),
                                   static_cast<std::uint32_t>(x.ne[0] / 8), total);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool paired = (x.ne[0] % 2) == 0 && (addresses & 0x3u) == 0;
    if (paired && rows <= std::numeric_limits<std::int32_t>::max()) {
        const std::int32_t pairs = x.ne[0] / 2;
        const unsigned grid_x =
            static_cast<unsigned>(div_up(pairs, block * kAddBiasPairsPerThread));
        const unsigned grid_y = static_cast<unsigned>(
            std::min<std::int64_t>(rows, std::numeric_limits<unsigned short>::max()));
        add_bias_bf16x2_kernel<block><<<dim3(grid_x, grid_y, 1u), block, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(bias.data), static_cast<__nv_bfloat162*>(x.data),
            pairs, static_cast<std::int32_t>(rows));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    add_bias_kernel<<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(bias.data),
                                                static_cast<__nv_bfloat16*>(x.data), x.ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
