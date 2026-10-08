#include "ops/linear/bf16/bf16_general.cuh"
#include "core/device.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/bf16/bf16_skinny.cuh"

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr std::int32_t kSkinnyMaxTokens = 8;

constexpr std::int32_t kSkinnySmallNRows    = 256;
constexpr std::int32_t kSkinnySmallNPreload = 16; // chunks per lane: K <= 4096

template <int Tokens>
void bf16_skinny_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n <= kSkinnySmallNRows && weight.k <= 8 * 32 * kSkinnySmallNPreload) {
        launch_bf16_skinny_gemv<Tokens, 2, 1, kSkinnySmallNPreload>(x, weight, out, stream);
    } else {
        launch_bf16_skinny_gemv<Tokens, 8, 2, 0>(x, weight, out, stream);
    }
}

void bf16_skinny_dispatch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1: return bf16_skinny_gemv<1>(x, weight, out, stream);
    case 2: return bf16_skinny_gemv<2>(x, weight, out, stream);
    case 3: return bf16_skinny_gemv<3>(x, weight, out, stream);
    case 4: return bf16_skinny_gemv<4>(x, weight, out, stream);
    case 5: return bf16_skinny_gemv<5>(x, weight, out, stream);
    case 6: return bf16_skinny_gemv<6>(x, weight, out, stream);
    case 7: return bf16_skinny_gemv<7>(x, weight, out, stream);
    default: return bf16_skinny_gemv<8>(x, weight, out, stream);
    }
}

// Runtime-shape bf16 GEMM used for every linear projection the specialised bf16 kernels do not
// tile. x is [tokens, k], weight is [n, k] row-major, and out is [tokens, n] token-major, matching
// the contiguous layout the specialised kernels produce.
void bf16_general_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t N = weight.n;
    const std::int32_t K = weight.k;
    const std::int32_t T = x.ne[1];
    const dim3 grid(static_cast<unsigned>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned>((T + kTileT - 1) / kTileT));
    bf16_general_gemm_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), N, K, T);
    CUDA_CHECK(cudaGetLastError());
}

// Decode-sized calls stream each weight row once when the rows can be read in 16-byte chunks.
void bf16_general_small(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool vector = weight.k % 8 == 0 && reinterpret_cast<std::uintptr_t>(weight.qdata) % 16 == 0 &&
                        reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0;
    if (vector) {
        bf16_skinny_dispatch(x, weight, out, stream);
    } else {
        bf16_general_gemm(x, weight, out, stream);
    }
}

} // namespace

[[nodiscard]] Bf16Launch select_bf16_general_launch(std::int32_t tokens) {
    return tokens <= kSkinnyMaxTokens ? &bf16_general_small : &bf16_general_gemm;
}

} // namespace infernix::ops::detail
