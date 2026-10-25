#include "ops/hyper_connection/hyper_connection_up_collapse.h"

#include "core/temporary_switches.h"

#include "ops/linear/bf16/bf16_schedule.cuh"
#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kStreams = 4;

// The collapse's own expressions (hyper_connection.cu collapse_kernel), so the fused result is
// bit-identical to the composed route's.
__device__ __forceinline__ float hc_sigmoid(float x) { return 1.0F / (1.0F + expf(-x)); }

// A CTA's BlockRows rows are the 4 streams of Features consecutive features: local row
// s * Features + j is W_up row s * hidden + d0 + j, d0 = blockIdx.x * Features.
template <int Features>
struct HcStreamRows {
    static constexpr int kOutputRowsPerCta = Features;
    int hidden;

    __device__ __forceinline__ int weight_row(int row0, int local_row, int) const {
        return (local_row / Features) * hidden + row0 + local_row % Features;
    }
};

// Never stored to: the epilogue writes x itself.
struct HcNoOutput {};

struct HcCollapseEpilogue {
    const bf16* normalized; // [S*H, T]
    bf16* x;                // [H, T]
    int hidden;

    // The tile of BF16-rounded u values, token-major with a padded pitch: [BlockTokens][BlockRows + 1].
    template <class Cfg>
    static constexpr int kSharedBytes = Cfg::kBlockTokens * (Cfg::kBlockRows + 1) * 4;

    template <class Cfg, bool Full, class Tile, class Acc>
    __device__ __forceinline__ void finish_tile(const Tile&, unsigned char* scratch, const Acc& acc, int m0, int n0,
                                                int, int n) const {
        constexpr int BM = Cfg::kBlockRows, BN = Cfg::kBlockTokens, Pitch = BM + 1;
        constexpr int Features = BM / kStreams;
        auto* u = reinterpret_cast<float*>(scratch);
        const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
        const int wm = warp / Cfg::kWarpGridTokens, wn = warp % Cfg::kWarpGridTokens;
        const int gid = lane >> 2, lid = lane & 3;
        __syncthreads(); // the operand tiles are no longer read
#pragma unroll
        for (int mi = 0; mi < Cfg::kMmaRows; ++mi) {
            const int r0 = wm * Cfg::kWarpRows + mi * 16 + gid;
#pragma unroll
            for (int ni = 0; ni < Cfg::kMmaTokens; ++ni) {
                const int c0 = wn * Cfg::kWarpTokens + ni * 8 + 2 * lid;
                // u as the composed route stores it: one BF16 rounding of the FP32 accumulator.
                u[c0 * Pitch + r0]           = __bfloat162float(__float2bfloat16_rn(acc[mi][ni][0]));
                u[(c0 + 1) * Pitch + r0]     = __bfloat162float(__float2bfloat16_rn(acc[mi][ni][1]));
                u[c0 * Pitch + r0 + 8]       = __bfloat162float(__float2bfloat16_rn(acc[mi][ni][2]));
                u[(c0 + 1) * Pitch + r0 + 8] = __bfloat162float(__float2bfloat16_rn(acc[mi][ni][3]));
            }
        }
        __syncthreads();
        const std::size_t width = static_cast<std::size_t>(kStreams) * hidden;
        for (int item = tid; item < Features * BN; item += Cfg::kThreads) {
            const int j = item % Features, c = item / Features;
            const int t = n0 + c, d = m0 + j;
            if (!Full && t >= n) { continue; }
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const std::size_t i = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
                sum += hc_sigmoid(u[c * Pitch + s * Features + j]) * __bfloat162float(normalized[i]);
            }
            x[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum / static_cast<float>(kStreams));
        }
    }
};

// BF16 W_up: the BF16 MMA tile with the CTA's rows mapped to the 4 streams of Features consecutive
// features (bf16_a16_mma_kernel's weight_row hook) and the same collapse in its epilogue. Each u value
// accumulates over K in k16 order in FP32, as every BF16 MMA tile of the composed linear does.
template <int Features>
struct HcBf16CollapseEpilogue {
    const bf16* normalized; // [S*H, T]
    bf16* x;                // [H, T]
    int hidden;

    template <class Schedule>
    static constexpr int kSharedBytes = Schedule::kBlockTokens * (Schedule::kBlockRows + 1) * 4;

    __device__ __forceinline__ std::int64_t weight_row(int m0, int row) const {
        return static_cast<std::int64_t>(row / Features) * hidden + m0 / kStreams + row % Features;
    }

    template <class Schedule, bool Full, class Destination, class Acc>
    __device__ __forceinline__ void finish_tile(const Destination&, unsigned char* scratch, const Acc& acc, int m0,
                                                int n0, int, int n) const {
        constexpr int BM = Schedule::kBlockRows, BN = Schedule::kBlockTokens, Pitch = BM + 1;
        static_assert(BM == kStreams * Features);
        auto* u = reinterpret_cast<float*>(scratch);
        const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
        const int wm = warp / Schedule::kWarpsTokens, wn = warp % Schedule::kWarpsTokens;
        const int gid = lane >> 2, lid = lane & 3;
#pragma unroll
        for (int mi = 0; mi < Schedule::kMmaRows; ++mi) {
            const int r0 = wm * Schedule::kWarpRows + mi * 16 + gid;
#pragma unroll
            for (int ni = 0; ni < Schedule::kMmaTokens; ++ni) {
                const int c0 = wn * Schedule::kWarpTokens + ni * 8 + 2 * lid;
                u[c0 * Pitch + r0]           = __bfloat162float(__float2bfloat16_rn(acc[mi][ni][0]));
                u[(c0 + 1) * Pitch + r0]     = __bfloat162float(__float2bfloat16_rn(acc[mi][ni][1]));
                u[c0 * Pitch + r0 + 8]       = __bfloat162float(__float2bfloat16_rn(acc[mi][ni][2]));
                u[(c0 + 1) * Pitch + r0 + 8] = __bfloat162float(__float2bfloat16_rn(acc[mi][ni][3]));
            }
        }
        __syncthreads();
        const std::size_t width = static_cast<std::size_t>(kStreams) * hidden;
        const int d0            = m0 / kStreams;
        for (int item = tid; item < Features * BN; item += Schedule::kThreads) {
            const int j = item % Features, c = item / Features;
            const int t = n0 + c, d = d0 + j;
            if (!Full && t >= n) { continue; }
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const std::size_t i = static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d;
                sum += hc_sigmoid(u[c * Pitch + s * Features + j]) * __bfloat162float(normalized[i]);
            }
            x[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum / static_cast<float>(kStreams));
        }
    }
};

// The BF16 tiles' output is the epilogue's own.
struct HcBf16NoOutput {};

template <class Schedule>
void launch_bf16(const Tensor& m, const Weight& up, const Tensor& normalized, Tensor& x, cudaStream_t stream) {
    constexpr int Features = Schedule::kBlockRows / kStreams;
    const int hidden       = x.ne[0];
    launch_bf16_a16_mma<Schedule>(
        bf16_a16_operands(m, up), HcBf16NoOutput{},
        HcBf16CollapseEpilogue<Features>{static_cast<const bf16*>(normalized.data), static_cast<bf16*>(x.data),
                                         hidden},
        stream);
}

template <class Schedule>
void launch(const Tensor& m, const Weight& up, const Tensor& normalized, Tensor& x, cudaStream_t stream) {
    constexpr int Features = Schedule::kBlockRows / kStreams;
    const int hidden       = x.ne[0];
    launch_q8_a16_mma<Schedule, HcStreamRows<Features>>(
        q8_linear_operands(m, up), HcNoOutput{},
        HcCollapseEpilogue{static_cast<const bf16*>(normalized.data), static_cast<bf16*>(x.data), hidden}, stream,
        HcStreamRows<Features>{hidden});
}

} // namespace

bool hc_up_collapse_supported(const Weight& up, std::int32_t hidden, std::int32_t streams,
                              std::int32_t columns) noexcept {
    // Up to 96 columns the separate linear and collapse measured as fast or faster (+0.5-2 %, Q8; the
    // BF16 tiles take the same admission, untimed).
    if (streams != kStreams || hidden <= 0 || hidden % 16 != 0 || columns <= 96 || up.qdata == nullptr ||
        up.n != streams * hidden) {
        return false;
    }
    if (up.qtype == QType::BF16) { return up.k % 64 == 0 && reinterpret_cast<std::uintptr_t>(up.qdata) % 16 == 0; }
    return up.qtype == QType::Q8_G32_FP16 && up.layout == QuantLayout::RowSplit && up.scales != nullptr &&
           up.padded_shape[1] % 128 == 0 && up.padded_shape[1] >= up.k && up.k % 8 == 0;
}

bool hc_bf16_up_collapse_enabled() noexcept {
    // TEMPORARY (INFERNIX_TMP_HC_BF16_UP_COLLAPSE) until the op test shows the composed route's bits
    // and the prefill A/B a gain.
    static const bool enabled = temporary_switches::flag("INFERNIX_TMP_HC_BF16_UP_COLLAPSE");
    return enabled;
}

void hc_up_collapse(const Tensor& m, const Weight& up, const Tensor& normalized, std::int32_t streams, Tensor& x,
                    cudaStream_t stream) {
    const std::int32_t T = m.ne[1];
    if (!hc_up_collapse_supported(up, x.ne[0], streams, T) || m.ne[0] != up.k || normalized.ne[1] != T ||
        normalized.ne[0] != up.n || x.ne[1] != T) {
        throw std::invalid_argument("hyper_connection: fused up + collapse shapes disagree");
    }
    if (up.qtype == QType::BF16) {
        // 16 features (64 stream rows) per CTA; the token tile by width. Untimed against the composed
        // route's TMA tiles.
        using Narrow = Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg,
                                          Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
        using Wide   = Bf16A16MmaSchedule<64, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg,
                                          Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
        if (T <= 128) return launch_bf16<Narrow>(m, up, normalized, x, stream);
        return launch_bf16<Wide>(m, up, normalized, x, stream);
    }
    // Tiles by width from the sweep of 2026-10-14 (whole mixer, 8 Q8 layers, RTX 5090); every tile
    // gives the same bits.
    if (T <= 128) return launch<q8_instances::MmaR32T64>(m, up, normalized, x, stream);
    return launch<q8_instances::MmaR64T128>(m, up, normalized, x, stream);
}

} // namespace infernix::ops::detail
