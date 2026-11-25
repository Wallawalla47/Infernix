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
// K <= 4608 (Inkling 2's message-gate downs [64, 4608]): the default instance would give them 4 CTAs. A separate
// instance, so the shapes above keep theirs.
constexpr std::int32_t kSkinnySmallNPreloadWide = 18;

template <int Tokens>
void bf16_skinny_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n <= kSkinnySmallNRows && weight.k <= 8 * 32 * kSkinnySmallNPreload) {
        launch_bf16_skinny_gemv<Tokens, 2, 1, kSkinnySmallNPreload>(x, weight, out, stream);
    } else if (weight.n <= kSkinnySmallNRows && weight.k <= 8 * 32 * kSkinnySmallNPreloadWide) {
        launch_bf16_skinny_gemv<Tokens, 2, 1, kSkinnySmallNPreloadWide>(x, weight, out, stream);
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
    const auto* xs = static_cast<const __nv_bfloat16*>(x.data);
    const auto* ws = static_cast<const __nv_bfloat16*>(weight.qdata);
    if (out.dtype == DType::FP32) {
        bf16_general_gemm_kernel<<<grid, kThreads, 0, stream>>>(xs, ws, static_cast<float*>(out.data), N, K, T);
    } else {
        bf16_general_gemm_kernel<<<grid, kThreads, 0, stream>>>(xs, ws, static_cast<__nv_bfloat16*>(out.data), N,
                                                                K, T);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Calls of 9 to 64 columns: the chain kernel (the tile's bits) where it applies, else the tile.
void bf16_general_chain(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t T = x.ne[1];
    const bool chain = T <= kChainMaxTokens && out.dtype == DType::BF16 && weight.n <= kChainMaxRows && weight.k % 8 == 0 &&
                       reinterpret_cast<std::uintptr_t>(weight.qdata) % 16 == 0 &&
                       reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0;
    if (!chain) {
        bf16_general_gemm(x, weight, out, stream);
        return;
    }
    const std::int32_t shift = T <= 16 ? 4 : (T <= 32 ? 5 : 6);
    const std::int32_t rows  = kChainThreads >> shift;
    const dim3 grid(static_cast<unsigned>((weight.n + rows - 1) / rows));
    bf16_general_chain_kernel<<<grid, kChainThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<__nv_bfloat16*>(out.data), weight.n, weight.k, T, shift);
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
    if (tokens <= kSkinnyMaxTokens) { return &bf16_general_small; }
    return tokens <= kChainMaxTokens ? &bf16_general_chain : &bf16_general_gemm;
}

} // namespace infernix::ops::detail
