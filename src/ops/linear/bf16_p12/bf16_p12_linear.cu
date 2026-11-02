// infernix::ops - Linear over bf16_p12 weights: the registered routes of the packed shapes (bf16_p12_routes.h), the
// packed skinny GEMV elsewhere (bf16_p12_dispatch.h).
#include "ops/linear/bf16_p12/bf16_p12_dispatch.h"

#include "core/device.h" // CUDA_CHECK
#include "infernix/ops/bf16_p12.h"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"
#include "ops/linear/bf16_p12/bf16_p12_routes.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

// The skinny GEMV's instances: a small matrix (at most 256 rows, K at most 4096) spreads over N / 2 CTAs of one
// row a warp; the others take 8 warps of 2 rows.
void skinny(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (w.n <= 256 && w.k <= 4096) {
        bf16_p12_skinny_launch<2, 1, 2>(x.ne[1])(x, w, out, stream);
    } else {
        bf16_p12_skinny_launch<8, 2, 1>(x.ne[1])(x, w, out, stream);
    }
}

// One thread a chunk of 8 weights.
__global__ void bf16_p12_unpack_kernel(const uint2* __restrict__ codes, const unsigned* __restrict__ nibbles,
                                       const unsigned* __restrict__ words, const unsigned* __restrict__ escapes,
                                       std::int64_t chunks, std::int32_t chunks_per_row, uint4* __restrict__ out) {
    const std::int64_t c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= chunks) { return; }
    const std::int64_t row = c / chunks_per_row;
    out[c] = p12_decode_chunk(codes[c], nibbles[c], p12_row(words, row), static_cast<std::int32_t>(c - row * chunks_per_row),
                              escapes);
}

void check_weight(const Weight& w) {
    if (w.qtype != QType::BF16_P12 || w.layout != QuantLayout::P12Rows || w.qdata == nullptr || w.qhigh == nullptr ||
        w.scales == nullptr || w.escapes == nullptr || w.k % 8 != 0) {
        throw std::invalid_argument("bf16_p12 linear: a packed [n, k] weight with k a multiple of 8");
    }
}

} // namespace

P12Launch select_bf16_p12_launch(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    if (k == 2560 && n == 16384) { return select_bf16_p12_n16384_k2560(tokens); }
    if (k == 2560 && n == 13952) { return select_bf16_p12_n13952_k2560(tokens); }
    if (k == 6144 && n == 2560) { return select_bf16_p12_n2560_k6144(tokens); }
    return nullptr;
}

void bf16_p12_unpack(const Weight& w, void* dst, cudaStream_t stream) {
    check_weight(w);
    const std::int64_t chunks = static_cast<std::int64_t>(w.n) * (w.k / 8);
    bf16_p12_unpack_kernel<<<static_cast<unsigned>((chunks + 255) / 256), 256, 0, stream>>>(
        static_cast<const uint2*>(w.qdata), static_cast<const unsigned*>(w.qhigh), static_cast<const unsigned*>(w.scales),
        static_cast<const unsigned*>(w.escapes), chunks, w.k / 8, static_cast<uint4*>(dst));
    CUDA_CHECK(cudaGetLastError());
}

void bf16_p12_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy, cudaStream_t stream) {
    check_weight(w);
    if (!valid_linear_policy(policy)) { throw std::invalid_argument("bf16_p12 linear: unsupported policy"); }
    if (!column_invariant(policy)) {
        if (const P12Launch launch = select_bf16_p12_launch(w.n, w.k, x.ne[1])) {
            launch(x, w, out, stream);
            return;
        }
    }
    for (std::int32_t first = 0; first < x.ne[1]; first += kBf16P12SkinnyMaxTokens) {
        const std::int32_t count = std::min(kBf16P12SkinnyMaxTokens, x.ne[1] - first);
        const Tensor xs          = x.slice(1, first, count);
        Tensor os                = out.slice(1, first, count);
        skinny(xs, w, os, stream);
    }
}

} // namespace infernix::ops::detail

namespace infernix::ops {

bool bf16_p12_linear_tuned(std::int32_t n, std::int32_t k) noexcept {
    return detail::select_bf16_p12_launch(n, k, 1) != nullptr;
}

} // namespace infernix::ops
