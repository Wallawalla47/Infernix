// infernix::ops - gdn_gating launcher: grid/block/stream configuration + kernel launch.
#include "ops/launcher/gdn_gating.h"

#include "ops/common/math.h"
#include "ops/kernel/gdn_gating.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace infernix::ops::detail {

void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                       Tensor& g, Tensor& beta, bool a_direct, cudaStream_t stream) {
    const std::int64_t n = g.numel();
    constexpr int kBlock = 256;
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));

    // A single column has no stride of its own; its heads are contiguous either way.
    const auto column_stride = [](const Tensor& t) {
        return t.ne[1] == 1 ? std::int64_t{48} : t.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
    };
    CUDA_CHECK(pdl::launch_consumer({dim3(grid), dim3(kBlock), 0, stream},
                                    a_direct ? gdn_gating_kernel<true> : gdn_gating_kernel<false>,
                                    static_cast<const __nv_bfloat16*>(a.data),
                                    static_cast<const __nv_bfloat16*>(b.data), column_stride(a),
                                    column_stride(b), static_cast<const float*>(A_log.data),
                                    static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
                                    static_cast<float*>(beta.data), n));
}

} // namespace infernix::ops::detail
