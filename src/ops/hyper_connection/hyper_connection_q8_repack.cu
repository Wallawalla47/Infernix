// hyper_connection_repack_q8 (include/infernix/ops/hyper_connection.h): a BF16 Contiguous hyper-connection
// matrix rewritten in its own bytes as Q8_G32_FP16 RowSplit for `--a8-activations` (LOSSY, opt-in), the
// layout hyper_connection_mix's Q8 routes read (hyper_connection_mix_fused.cu's Q8Codec: codes at
// row * padded_k + k, the FP16 scale of group g at row * (padded_k / 32) * 2 + 2 g past the codes).

#include "infernix/ops/hyper_connection.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr int kThreads = 256; // eight warps a CTA, one 32-element group a warp at a time

[[nodiscard]] int padded_columns(int k) { return (k + 127) / 128 * 128; }

[[nodiscard]] bool repackable(const Weight& w) {
    return w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.qdata != nullptr && w.n > 0 && w.k > 0 &&
           w.k % 32 == 0 && (w.padded_shape[1] == w.k || w.padded_shape[1] <= 1) &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0;
}

// CTA r takes row r; warp j its groups j, j + 8, ...; lane l element 32 g + l.
__global__ void __launch_bounds__(kThreads)
    repack_kernel(const __nv_bfloat16* __restrict__ src, int k, int padded_k, std::int8_t* __restrict__ codes,
                  __half* __restrict__ scales) {
    const int row = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int groups = padded_k / 32;
    for (int g = warp; g < groups; g += kThreads / 32) {
        const int col = 32 * g + lane;
        const float v = col < k ? __bfloat162float(src[static_cast<std::size_t>(row) * k + col]) : 0.0F;
        float m = fabsf(v);
        for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
        const __half s16 = __float2half_rn(m / 127.0F);
        const float s    = __half2float(s16);
        int q            = 0;
        if (s > 0.0F) {
            q = __float2int_rn(v / s);
            q = q > 127 ? 127 : (q < -127 ? -127 : q);
        }
        codes[static_cast<std::size_t>(row) * padded_k + col] = static_cast<std::int8_t>(q);
        if (lane == 0) { scales[static_cast<std::size_t>(row) * groups + g] = s > 0.0F ? s16 : __float2half(0.0F); }
    }
}

} // namespace

std::size_t hyper_connection_repack_q8_scratch_bytes(const Weight& w) {
    if (!repackable(w)) { return 0; }
    const std::size_t padded = static_cast<std::size_t>(padded_columns(w.k));
    const std::size_t q8     = static_cast<std::size_t>(w.n) * padded + static_cast<std::size_t>(w.n) * (padded / 32) * 2;
    const std::size_t bf16   = static_cast<std::size_t>(w.n) * static_cast<std::size_t>(w.k) * 2;
    return q8 <= bf16 ? bf16 : 0; // the Q8 matrix must fit in the BF16 matrix's bytes
}

void hyper_connection_repack_q8(Weight& w, void* scratch, cudaStream_t stream) {
    if (hyper_connection_repack_q8_scratch_bytes(w) == 0 || scratch == nullptr) {
        throw std::invalid_argument("hyper_connection_repack_q8: needs a 16-byte aligned BF16 Contiguous [n, k] weight "
                                    "(k a multiple of 32) whose Q8 form fits in its bytes, and scratch");
    }
    const int n = w.n, k = w.k, padded_k = padded_columns(k);
    auto* base = static_cast<std::uint8_t*>(const_cast<void*>(w.qdata));
    const std::size_t bf16_bytes = static_cast<std::size_t>(n) * k * 2;
    if (cudaMemcpyAsync(scratch, base, bf16_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("hyper_connection_repack_q8: copy to scratch failed");
    }
    auto* codes  = reinterpret_cast<std::int8_t*>(base);
    auto* scales = reinterpret_cast<__half*>(base + static_cast<std::size_t>(n) * padded_k);
    repack_kernel<<<n, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(scratch), k, padded_k, codes, scales);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection_repack_q8: ") + cudaGetErrorString(e));
    }
    const std::size_t scale_row = static_cast<std::size_t>(padded_k / 32) * 2;
    w.qtype            = QType::Q8_G32_FP16;
    w.layout           = QuantLayout::RowSplit;
    w.payload          = base;
    w.payload_bytes    = static_cast<std::size_t>(n) * padded_k + static_cast<std::size_t>(n) * scale_row;
    w.high_plane_bytes = 0;
    w.qdata            = base;
    w.qhigh            = nullptr;
    w.scales           = scales;
    w.padded_shape[1]  = padded_k;
    w.group_size       = 32;
    w.group            = 32;
    w.scale_dtype      = DType::FP16;
    w.scale_ne[0]      = padded_k / 32;
    w.scale_ne[1]      = n;
    w.scale_nb[0]      = 2;
    w.scale_nb[1]      = static_cast<std::int64_t>(scale_row);
    w.scale_nb[2] = w.scale_nb[3] = w.scale_nb[1] * n;
}

} // namespace infernix::ops
