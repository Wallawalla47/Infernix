// The Qwen3.8-27B Gated DeltaNet control with a LoRA term (gdn_gating_proj's adapted form, include/infernix/ops/
// gdn_gating_proj.h): a = W_a h and b = W_b h over the normalized input, each plus its canonical term
// (include/infernix/ops/lora.h), then the gates.
//
// One schedule at every width, so a column's g and beta are functions of that column alone: a CTA takes one 16-row
// tile of the 96 rows (a's 48, then b's) over kColumns columns; its eight warps each chain one eighth of K (40 BF16
// m16n8k16 k-steps from +0) and the eighths are added in order through shared memory. A warp then adds its fragment's
// term (the chain over the rank, B's rows as the A operand and U's column as the B operand, as every projection kernel
// runs it) and applies the gates: g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b).

#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/lora/lora_device.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kHeads    = 48;
constexpr int kHidden   = 5120;
constexpr int kRows     = 2 * kHeads;
constexpr int kWarps    = 8;
constexpr int kSlice    = kHidden / kWarps; // 640 values a warp
constexpr int kSteps    = kSlice / 16;      // 40 k-steps
static_assert(kHeads % 16 == 0 && kHidden % (kWarps * 16) == 0);

template <int Columns>
__global__ __launch_bounds__(kWarps * 32) void bf16_gdn_gating_lora_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, const LoraDeviceAddend lora, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta, int tokens) {
    constexpr int kTiles = Columns / 8; // n8 column tiles
    __shared__ float part[kWarps][16][Columns];
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    const int gi = lane >> 2, q = lane & 3;
    const int n0 = static_cast<int>(blockIdx.y) * 16; // in the 96 logical rows
    const int t0 = static_cast<int>(blockIdx.x) * Columns;
    const bool is_b = n0 >= kHeads;
    const __nv_bfloat16* w = (is_b ? b_weight + static_cast<std::int64_t>(n0 - kHeads) * kHidden
                                   : a_weight + static_cast<std::int64_t>(n0) * kHidden) +
                             warp * kSlice + 2 * q;
    const __nv_bfloat16* w_lo = w + static_cast<std::int64_t>(gi) * kHidden;
    const __nv_bfloat16* w_hi = w_lo + static_cast<std::int64_t>(8) * kHidden;
    pdl::enter();

    float acc[kTiles][4] = {};
#pragma unroll 8
    for (int step = 0; step < kSteps; ++step) {
        const unsigned a0 = lora_load_pair(w_lo + step * 16), a1 = lora_load_pair(w_hi + step * 16);
        const unsigned a2 = lora_load_pair(w_lo + step * 16 + 8), a3 = lora_load_pair(w_hi + step * 16 + 8);
#pragma unroll
        for (int nt = 0; nt < kTiles; ++nt) {
            const int column = t0 + nt * 8 + gi;
            const bool live  = column < tokens;
            const __nv_bfloat16* xc =
                x + static_cast<std::int64_t>(live ? column : 0) * kHidden + warp * kSlice + step * 16 + 2 * q;
            const unsigned b0 = live ? lora_load_pair(xc) : 0U, b1 = live ? lora_load_pair(xc + 8) : 0U;
            mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a0, a1, a2, a3, b0, b1);
        }
    }
#pragma unroll
    for (int nt = 0; nt < kTiles; ++nt) {
        const int c = nt * 8 + 2 * q;
        part[warp][gi][c]         = acc[nt][0];
        part[warp][gi][c + 1]     = acc[nt][1];
        part[warp][gi + 8][c]     = acc[nt][2];
        part[warp][gi + 8][c + 1] = acc[nt][3];
    }
    __syncthreads();
    // The eighths of K in order, into part[0].
    for (int e = static_cast<int>(threadIdx.x); e < 16 * Columns; e += kWarps * 32) {
        const int r = e / Columns, c = e % Columns;
        float v     = part[0][r][c];
#pragma unroll
        for (int s = 1; s < kWarps; ++s) { v = __fadd_rn(v, part[s][r][c]); }
        part[0][r][c] = v;
    }
    __syncthreads();
    if (warp >= kTiles) { return; }
    // Warp nt: its n8 tile's term chain, then the gates.
    const int segment = lora.segment_of(n0), u_offset = lora.offset_of(segment);
    const float scale = lora.scale_of(segment);
    float term[4]     = {0.0F, 0.0F, 0.0F, 0.0F};
    if (u_offset >= 0) {
        const int column           = t0 + warp * 8 + gi;
        const bool live            = column < tokens;
        const __nv_bfloat16* b_lo  = lora.b + static_cast<std::int64_t>(n0 + gi) * lora.rank + 2 * q;
        const __nv_bfloat16* b_hi  = b_lo + static_cast<std::int64_t>(8) * lora.rank;
        const __nv_bfloat16* u_col = lora.u + static_cast<std::int64_t>(live ? column : 0) * lora.u_rows + u_offset + 2 * q;
        for (int ks = 0; ks < lora.rank / 16; ++ks) {
            const unsigned a0 = lora_load_pair(b_lo + ks * 16), a1 = lora_load_pair(b_hi + ks * 16);
            const unsigned a2 = lora_load_pair(b_lo + ks * 16 + 8), a3 = lora_load_pair(b_hi + ks * 16 + 8);
            const unsigned b0 = live ? lora_load_pair(u_col + ks * 16) : 0U;
            const unsigned b1 = live ? lora_load_pair(u_col + ks * 16 + 8) : 0U;
            mma_bf16(term[0], term[1], term[2], term[3], a0, a1, a2, a3, b0, b1);
        }
    }
#pragma unroll
    for (int v = 0; v < 4; ++v) {
        const int r = gi + (v >= 2 ? 8 : 0), c = warp * 8 + 2 * q + (v & 1);
        const int column = t0 + c;
        if (column >= tokens) { continue; }
        float value = part[0][r][c];
        if (u_offset >= 0) { value = __fadd_rn(value, __fmul_rn(term[v], scale)); }
        const int row                = n0 + r;
        const std::int64_t out_index = static_cast<std::int64_t>(column) * kHeads + (is_b ? row - kHeads : row);
        if (is_b) {
            beta[out_index] = sigmoid(value);
        } else {
            g[out_index] = -expf(A_log[row]) * softplus(value + dt_bias[row]);
        }
    }
}

template <int Columns>
void launch_as(const Tensor& x, const Weight& a_weight, const Weight& b_weight, const LoraAddend& lora,
               const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta, cudaStream_t stream) {
    const int tokens = x.ne[1];
    const dim3 grid(static_cast<unsigned>((tokens + Columns - 1) / Columns), static_cast<unsigned>(kRows / 16));
    CUDA_CHECK(pdl::launch_consumer({grid, dim3(kWarps * 32), 0, stream}, bf16_gdn_gating_lora_kernel<Columns>,
                                    static_cast<const __nv_bfloat16*>(x.data),
                                    static_cast<const __nv_bfloat16*>(a_weight.qdata),
                                    static_cast<const __nv_bfloat16*>(b_weight.qdata),
                                    lora_device_addend<LoraDeviceAddend>(lora), static_cast<const float*>(A_log.data),
                                    static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
                                    static_cast<float*>(beta.data), tokens));
}

} // namespace

void bf16_gdn_gating_lora_launch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                                 const LoraAddend& lora, const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                 Tensor& beta, cudaStream_t stream) {
    // Eight columns a CTA for decode and short verification (more CTAs), 64 for prompts (W read once per 64).
    if (x.ne[1] <= 32) {
        launch_as<8>(x, a_weight, b_weight, lora, A_log, dt_bias, g, beta, stream);
    } else {
        launch_as<64>(x, a_weight, b_weight, lora, A_log, dt_bias, g, beta, stream);
    }
}

} // namespace infernix::ops::detail
