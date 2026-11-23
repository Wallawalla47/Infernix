// The Qwen3.8-27B Gated DeltaNet control with a LoRA term (gdn_gating_proj's adapted form, include/infernix/ops/
// gdn_gating_proj.h): a = W_a h and b = W_b h over the normalized input, each plus its canonical term
// (include/infernix/ops/lora.h), then the gates.
//
// One arithmetic at every width, so a column's g and beta are functions of that column alone: each 16-row tile of the
// 96 rows (a's 48, then b's) is chained in eight eighths of K (40 BF16 m16n8k16 k-steps from +0 each) and the eighths
// are added in order. A warp then adds its fragment's term (the chain over the rank, B's rows as the A operand and U's
// column as the B operand, as every projection kernel runs it) and applies the gates: g = -exp(A_log) softplus(a +
// dt_bias), beta = sigmoid(b). Prompts take a CTA a tile over 64 columns, its eight warps the eighths (shared memory);
// decode and short verification a CTA an eighth, a tile's eighths one cluster (distributed shared memory).

#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/lora/lora_device.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

namespace cg = cooperative_groups;

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

// Decode and short verification: the tile kernel's 48 chains (six row tiles, eight eighths of K) one a CTA, so they
// run on 48 SMs instead of six (a 16-row tile's 160 KB of W had one SM to stream them). A CTA (one warp) chains its
// eighth of K for one 16-row tile over eight columns, as a warp of bf16_gdn_gating_lora_kernel<8> does; the tile's
// eight eighths are one cluster, whose rank 0 adds them in order (its own, then ranks 1..7 through distributed shared
// memory) and runs the term chain and the gates. Every value's operations are the tile kernel's: the same bits.
// The CTA first copies its W slice (immutable: before the PDL wait), its x columns and, on rank 0, the term's B rows
// and U columns into shared memory with every copy in flight at once, then takes the fragments by ldmatrix (the
// register-loaded form waited on 40 dependent rounds of global loads: 6 us a call at decode).
constexpr int kSliceStride = kSlice + 8;          // BF16 a staged row: 1,296 bytes (eight rows, eight distinct banks)
constexpr int kTermStride  = kLoraMaxRank + 8;    // B's rows / U's columns of the term, up to the wide rank
__global__ __launch_bounds__(32) void bf16_gdn_gating_lora_split_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, const LoraDeviceAddend lora, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta, int tokens) {
    constexpr int Columns = 8;
    __shared__ float part[16][Columns];
    __shared__ __align__(16) __nv_bfloat16 w_s[16][kSliceStride];
    __shared__ __align__(16) __nv_bfloat16 x_s[Columns][kSliceStride];
    __shared__ __align__(16) __nv_bfloat16 b_s[16][kTermStride];
    __shared__ __align__(16) __nv_bfloat16 u_s[Columns][kTermStride];
    cg::cluster_group cluster = cg::this_cluster();
    const int eighth = static_cast<int>(cluster.block_rank());
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int gi = lane >> 2, q = lane & 3;
    const int n0 = static_cast<int>(blockIdx.y) * 16; // in the 96 logical rows
    const int t0 = static_cast<int>(blockIdx.x) / kWarps * Columns;
    const bool is_b = n0 >= kHeads;
    const __nv_bfloat16* w = (is_b ? b_weight + static_cast<std::int64_t>(n0 - kHeads) * kHidden
                                   : a_weight + static_cast<std::int64_t>(n0) * kHidden) +
                             eighth * kSlice;
    constexpr int kSlicePieces = kSlice / 8; // 16-byte pieces of a row's slice
    for (int i = lane; i < 16 * kSlicePieces; i += 32) {
        const int r = i / kSlicePieces, piece = i % kSlicePieces;
        cp_async<16>(&w_s[r][piece * 8], w + static_cast<std::int64_t>(r) * kHidden + piece * 8);
    }
    const int segment = lora.segment_of(n0), u_offset = lora.offset_of(segment);
    const int term_pieces = lora.rank / 8;
    if (eighth == 0 && u_offset >= 0) {
        for (int i = lane; i < 16 * term_pieces; i += 32) {
            const int r = i / term_pieces, piece = i % term_pieces;
            cp_async<16>(&b_s[r][piece * 8], lora.b + static_cast<std::int64_t>(n0 + r) * lora.rank + piece * 8);
        }
    }
    pdl::enter();
    for (int i = lane; i < Columns * kSlicePieces; i += 32) {
        const int c = i / kSlicePieces, piece = i % kSlicePieces;
        const bool live = t0 + c < tokens;
        cp_async_zfill<16>(&x_s[c][piece * 8],
                           x + static_cast<std::int64_t>(live ? t0 + c : 0) * kHidden + eighth * kSlice + piece * 8,
                           live ? 16 : 0);
    }
    if (eighth == 0 && u_offset >= 0) {
        for (int i = lane; i < Columns * term_pieces; i += 32) {
            const int c = i / term_pieces, piece = i % term_pieces;
            const bool live = t0 + c < tokens;
            cp_async_zfill<16>(&u_s[c][piece * 8],
                               lora.u + static_cast<std::int64_t>(live ? t0 + c : 0) * lora.u_rows + u_offset + piece * 8,
                               live ? 16 : 0);
        }
    }
    cp_commit();
    cp_wait<0>();
    __syncwarp();

    // ldmatrix lanes: the A operand's four 8 x 8 matrices (rows 0-7 / 8-15 at columns +0, then +8), the B operand's two
    // (eight columns at +0, then +8): the fragments the register-loaded form read.
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
    const int b_row = lane & 7, b_col = ((lane >> 3) & 1) * 8;
    float acc[4] = {};
#pragma unroll 8
    for (int step = 0; step < kSteps; ++step) {
        unsigned a0, a1, a2, a3, b0, b1;
        ldmatrix_x4(a0, a1, a2, a3, smem_addr(&w_s[a_row][step * 16 + a_col]));
        ldmatrix_x2(b0, b1, smem_addr(&x_s[b_row][step * 16 + b_col]));
        mma_bf16(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
    }
    part[gi][2 * q]         = acc[0];
    part[gi][2 * q + 1]     = acc[1];
    part[gi + 8][2 * q]     = acc[2];
    part[gi + 8][2 * q + 1] = acc[3];
    cluster.sync(); // every eighth's partial sums
    if (eighth == 0) {
        // The eighths of K in order, into rank 0's own part.
        for (int e = lane; e < 16 * Columns; e += 32) {
            const int r = e / Columns, c = e % Columns;
            float v     = part[r][c];
#pragma unroll
            for (int s = 1; s < kWarps; ++s) { v = __fadd_rn(v, *cluster.map_shared_rank(&part[r][c], s)); }
            part[r][c] = v;
        }
        __syncwarp();
        const float scale = lora.scale_of(segment);
        float term[4]     = {0.0F, 0.0F, 0.0F, 0.0F};
        if (u_offset >= 0) {
            // The tile kernel's chain on the staged B rows and U columns (a column past the call is zero).
            for (int ks = 0; ks < lora.rank / 16; ++ks) {
                unsigned a0, a1, a2, a3, b0, b1;
                ldmatrix_x4(a0, a1, a2, a3, smem_addr(&b_s[a_row][ks * 16 + a_col]));
                ldmatrix_x2(b0, b1, smem_addr(&u_s[b_row][ks * 16 + b_col]));
                mma_bf16(term[0], term[1], term[2], term[3], a0, a1, a2, a3, b0, b1);
            }
        }
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int r = gi + (v >= 2 ? 8 : 0), c = 2 * q + (v & 1);
            const int column = t0 + c;
            if (column >= tokens) { continue; }
            float value = part[r][c];
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
    cluster.sync(); // no rank leaves while rank 0 reads its partial sums
}

void launch_split(const Tensor& x, const Weight& a_weight, const Weight& b_weight, const LoraAddend& lora,
                  const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta, cudaStream_t stream) {
    constexpr auto kernel = bf16_gdn_gating_lora_split_kernel;
    const int tokens      = x.ne[1];
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id               = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = static_cast<unsigned>(kWarps);
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    unsigned count                 = 1;
    if (pdl::graph_dependent(stream, kernel)) {
        attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
        count                                                    = 2;
    }
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(static_cast<unsigned>((tokens + 7) / 8 * kWarps), static_cast<unsigned>(kRows / 16));
    config.blockDim = dim3(32);
    config.stream   = stream;
    config.attrs    = attributes;
    config.numAttrs = count;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, static_cast<const __nv_bfloat16*>(x.data),
                                  static_cast<const __nv_bfloat16*>(a_weight.qdata),
                                  static_cast<const __nv_bfloat16*>(b_weight.qdata),
                                  lora_device_addend<LoraDeviceAddend>(lora), static_cast<const float*>(A_log.data),
                                  static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
                                  static_cast<float*>(beta.data), tokens));
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
    // A cluster a row tile's eighths of K for decode and short verification (48 CTAs a column tile), 64 columns a CTA
    // for prompts (W read once per 64): the same arithmetic.
    if (x.ne[1] <= 32) {
        launch_split(x, a_weight, b_weight, lora, A_log, dt_bias, g, beta, stream);
    } else {
        launch_as<64>(x, a_weight, b_weight, lora, A_log, dt_bias, g, beta, stream);
    }
}

} // namespace infernix::ops::detail
