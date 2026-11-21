// quantize_bf16_q8_g32 (include/infernix/ops/q8_g32.h): a BF16 Contiguous matrix's 8-bit form, Q8_G32_FP16 RowSplit
// (row_split_k128_v1: codes at row * padded_k + k, the FP16 scale of group g at row * (padded_k / 32) + g of the
// scale plane).

#include "infernix/ops/q8_g32.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr int kThreads = 256; // eight warps a CTA, one 32-element group a warp at a time

[[nodiscard]] int padded_columns(int k) { return (k + 127) / 128 * 128; }

// CTA r takes row r; warp j its groups j, j + 8, ...; lane l element 32 g + l.
__global__ void __launch_bounds__(kThreads)
    quantize_kernel(const __nv_bfloat16* __restrict__ src, int k, int padded_k, std::int8_t* __restrict__ codes,
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

bool q8_g32_quantizable(const Weight& w) noexcept {
    return w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.qdata != nullptr && w.n > 0 && w.k > 0 &&
           w.k % 32 == 0 && (w.padded_shape[1] == w.k || w.padded_shape[1] <= 1) &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0;
}

std::int32_t q8_g32_padded_columns(std::int32_t k) noexcept { return padded_columns(k); }

std::size_t q8_g32_bytes(std::int32_t n, std::int32_t k) noexcept {
    const std::size_t padded = static_cast<std::size_t>(padded_columns(k));
    return static_cast<std::size_t>(n) * padded + static_cast<std::size_t>(n) * (padded / 32) * 2;
}

Weight quantize_bf16_q8_g32(const Weight& src, void* codes, void* scales, cudaStream_t stream) {
    if (!q8_g32_quantizable(src) || codes == nullptr || scales == nullptr ||
        reinterpret_cast<std::uintptr_t>(codes) % 16 != 0 || reinterpret_cast<std::uintptr_t>(scales) % 2 != 0) {
        throw std::invalid_argument("quantize_bf16_q8_g32: needs a 16-byte aligned BF16 Contiguous [n, k] weight (k a "
                                    "multiple of 32), 16-byte aligned codes and 2-byte aligned scales");
    }
    const int n = src.n, k = src.k, padded_k = padded_columns(k);
    quantize_kernel<<<n, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(src.qdata), k, padded_k,
                                                static_cast<std::int8_t*>(codes), static_cast<__half*>(scales));
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        throw std::runtime_error(std::string("quantize_bf16_q8_g32: ") + cudaGetErrorString(e));
    }
    const std::size_t scale_row = static_cast<std::size_t>(padded_k / 32) * 2;
    Weight w           = src;
    w.qtype            = QType::Q8_G32_FP16;
    w.layout           = QuantLayout::RowSplit;
    w.payload          = codes;
    w.payload_bytes    = q8_g32_bytes(n, k);
    w.high_plane_bytes = 0;
    w.qdata            = codes;
    w.qhigh            = nullptr;
    w.scales           = scales;
    w.escapes          = nullptr;
    w.padded_shape[1]  = padded_k;
    w.group_size       = 32;
    w.group            = 32;
    w.scale_dtype      = DType::FP16;
    w.scale_ne[0]      = padded_k / 32;
    w.scale_ne[1]      = n;
    w.scale_nb[0]      = 2;
    w.scale_nb[1]      = static_cast<std::int64_t>(scale_row);
    w.scale_nb[2] = w.scale_nb[3] = w.scale_nb[1] * n;
    return w;
}

} // namespace infernix::ops
