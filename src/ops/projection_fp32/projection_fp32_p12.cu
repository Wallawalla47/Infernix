// projection_fp32 over a bf16_p12 weight (a BF16 vocabulary head packed at load): the barrier-free register-decode
// MMA (ops/linear/bf16_p12/bf16_p12_direct.cuh) with four K slices, one schedule a width class in 64-column
// passes. Every schedule with four K slices sums an output in the same order, so a column's bits do not depend on
// the call's width at any width. RTX 5090, [248320, 2560], cold L2 (bench/ops/bf16_p12_sweep.cu, 2026-11-01): 562-577
// us at 1-16 columns against the BF16 head's 701-708 us, 589-614 us at 24-32 against 2,325-3,084, 853 us at 64 against
// 3,048.

#include "ops/projection_fp32/projection_fp32_p12.h"

#include "infernix/ops/bf16_p12.h"
#include "ops/linear/bf16_p12/bf16_p12_direct.cuh"

#include <algorithm>
#include <stdexcept>

namespace infernix::ops {

bool bf16_p12_projection_fp32_supported(std::int32_t n, std::int32_t k) noexcept {
    return n > 0 && n % 16 == 0 && k % 128 == 0 && k / 128 >= 4 && k < 65536;
}

namespace detail {

void projection_fp32_bf16_p12(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.qtype != QType::BF16_P12 || w.layout != QuantLayout::P12Rows || !bf16_p12_projection_fp32_supported(w.n, w.k) ||
        x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1] || out.dtype != DType::FP32) {
        throw std::invalid_argument("projection_fp32: a bf16_p12 weight [N, K] with N a multiple of 16 and K of 512");
    }
    constexpr std::int32_t kPass = 64;
    for (std::int32_t first = 0; first < x.ne[1]; first += kPass) {
        const std::int32_t count = std::min(kPass, x.ne[1] - first);
        const Tensor xs          = x.slice(1, first, count);
        Tensor os                = out.slice(1, first, count);
        const LinearFp32Output output{static_cast<float*>(os.data), w.n};
        if (count <= 8) {
            launch_bf16_p12_direct_output<P12DirectSchedule<1, 1, 2, 4, 3>>(xs, w, output, stream);
        } else if (count <= 16) {
            launch_bf16_p12_direct_output<P12DirectSchedule<1, 2, 2, 4, 3>>(xs, w, output, stream);
        } else if (count <= 32) {
            launch_bf16_p12_direct_output<P12DirectSchedule<1, 4, 2, 4, 2>>(xs, w, output, stream);
        } else {
            launch_bf16_p12_direct_output<P12DirectSchedule<1, 8, 2, 4, 2>>(xs, w, output, stream);
        }
    }
}

} // namespace detail
} // namespace infernix::ops
