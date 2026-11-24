// Inkling 2's LoRA term (include/infernix/ops/inkling2_lora.h). A CTA owns kWarps row groups of 16 of one target and a
// column tile (MN fragments of 8 columns); a warp computes its 16 rows' term with the canonical MMA chain of
// include/infernix/ops/lora.h (the k-steps over j ascending, from +0) on top of the y it loaded and writes y back.
// B is immutable, so a lane loads its operands before the PDL wait and only U and y follow it. Separate instances per
// k-step count (a compile-time loop, no branch a step), column tile and output type; nothing here is shared with the
// base projections.

#include "infernix/ops/inkling2_lora.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace infernix::ops::inkling2 {
namespace {

constexpr int kWarps    = 8;
constexpr int kTileRows = kWarps * 16;
constexpr int kNarrowT  = 8; // calls of at most this many columns: one fragment a warp
constexpr int kWideMN   = 4; // wider calls: four fragments a warp, the rows' B held for all of them

struct TermParams {
    LoraTarget targets[kMaxLoraTargets];
    std::int32_t cta_begin[kMaxLoraTargets]; // the target's first CTA (blockIdx.x)
    std::int32_t count;
    std::int32_t columns;
};

template <bool Fp32>
__device__ __forceinline__ float load_y(const void* y, std::int32_t stride, int t, int r) {
    const std::int64_t at = static_cast<std::int64_t>(t) * stride + r;
    if constexpr (Fp32) {
        return static_cast<const float*>(y)[at];
    } else {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(y)[at]);
    }
}

template <bool Fp32>
__device__ __forceinline__ void store_y(void* y, std::int32_t stride, int t, int r, float value) {
    const std::int64_t at = static_cast<std::int64_t>(t) * stride + r;
    if constexpr (Fp32) {
        static_cast<float*>(y)[at] = value;
    } else {
        static_cast<__nv_bfloat16*>(y)[at] = __float2bfloat16_rn(value);
    }
}

__device__ __forceinline__ unsigned load_pair(const std::uint16_t* p) { return *reinterpret_cast<const unsigned*>(p); }

template <int Steps, int MN, bool Fp32>
__device__ __forceinline__ void term_body(const TermParams& a) {
    constexpr int kRank = Steps * 16;
    const int lane = static_cast<int>(threadIdx.x & 31U), warp = static_cast<int>(threadIdx.x >> 5);
    const int g = lane >> 2, q = lane & 3;
    const int cta = static_cast<int>(blockIdx.x);
    // The CTA's target by selection over compile-time indices: a run-time index into the by-value parameters would
    // copy them to local memory.
    LoraTarget t = a.targets[0];
    int begin    = 0;
#pragma unroll
    for (int i = 1; i < kMaxLoraTargets; ++i) {
        if (i < a.count && cta >= a.cta_begin[i]) {
            t     = a.targets[i];
            begin = a.cta_begin[i];
        }
    }
    const int row0 = (cta - begin) * kTileRows + warp * 16;
    const bool live = row0 < t.rows;
    // This lane's rows of B, immutable: issued before the wait. A fragment's rows g and g + 8, k = 16 ks + 2q (+ 8).
    unsigned b[Steps][4] = {};
    const int r_lo = row0 + g, r_hi = row0 + g + 8;
    if (live) {
        const std::uint16_t* b_lo = t.b + static_cast<std::int64_t>(r_lo) * kRank;
        const std::uint16_t* b_hi = t.b + static_cast<std::int64_t>(r_hi) * kRank;
#pragma unroll
        for (int ks = 0; ks < Steps; ++ks) {
            const int k = ks * 16 + 2 * q;
            b[ks][0]    = __ldg(reinterpret_cast<const unsigned*>(b_lo + k));
            b[ks][1]    = __ldg(reinterpret_cast<const unsigned*>(b_hi + k));
            b[ks][2]    = __ldg(reinterpret_cast<const unsigned*>(b_lo + k + 8));
            b[ks][3]    = __ldg(reinterpret_cast<const unsigned*>(b_hi + k + 8));
        }
    }
    pdl::enter();
    if (!live) { return; }
    const int t0 = static_cast<int>(blockIdx.y) * MN * 8;
    const bool has_lo = r_lo < t.rows, has_hi = r_hi < t.rows;
#pragma unroll
    for (int mn = 0; mn < MN; ++mn) {
        // This lane's C fragment elements: rows g and g + 8, columns 2q and 2q + 1 of the 8-column fragment; its B
        // operand: U's column g of the fragment.
        const int c0 = t0 + mn * 8 + 2 * q, c1 = c0 + 1;
        const bool live0 = c0 < a.columns, live1 = c1 < a.columns;
        float acc[4];
        acc[0] = has_lo && live0 ? load_y<Fp32>(t.y, t.y_stride, c0, r_lo) : 0.0F;
        acc[1] = has_lo && live1 ? load_y<Fp32>(t.y, t.y_stride, c1, r_lo) : 0.0F;
        acc[2] = has_hi && live0 ? load_y<Fp32>(t.y, t.y_stride, c0, r_hi) : 0.0F;
        acc[3] = has_hi && live1 ? load_y<Fp32>(t.y, t.y_stride, c1, r_hi) : 0.0F;
        const int token  = t0 + mn * 8 + g;
        const bool has_u = token < a.columns;
        const std::uint16_t* u_col = t.u + static_cast<std::int64_t>(has_u ? token : 0) * t.u_stride;
        float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
        for (int ks = 0; ks < Steps; ++ks) {
            const int k       = ks * 16 + 2 * q;
            const unsigned u0 = has_u ? load_pair(u_col + k) : 0U;
            const unsigned u1 = has_u ? load_pair(u_col + k + 8) : 0U;
            mma_bf16(c[0], c[1], c[2], c[3], b[ks][0], b[ks][1], b[ks][2], b[ks][3], u0, u1);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) { acc[e] = __fadd_rn(acc[e], __fmul_rn(c[e], t.scale)); }
        if (has_lo && live0) { store_y<Fp32>(t.y, t.y_stride, c0, r_lo, acc[0]); }
        if (has_lo && live1) { store_y<Fp32>(t.y, t.y_stride, c1, r_lo, acc[1]); }
        if (has_hi && live0) { store_y<Fp32>(t.y, t.y_stride, c0, r_hi, acc[2]); }
        if (has_hi && live1) { store_y<Fp32>(t.y, t.y_stride, c1, r_hi, acc[3]); }
    }
}

// Calls of at most kNarrowT columns: one CTA per 128 rows, fewer CTAs than the GPU has SMs.
template <int Steps, bool Fp32>
__global__ void __launch_bounds__(kWarps * 32) lora_term_kernel(const TermParams a) {
    term_body<Steps, 1, Fp32>(a);
}

// The register budget ptxas compiles the wide instances for, as CTAs an SM (.minnctapersm). Read off the SASS with CUDA
// 13.4 for sm_120a: no instance gets worse in instructions, registers or the loads issued ahead of its first MMA, and
// the 7-step instances stop duplicating their addressing (1,046 instructions without it, 565 at 8 steps). Bounds of 4
// and 5 fit more CTAs at 5, 6 and 8 steps but issue fewer loads ahead of the MMAs: a trade only a GPU run can settle.
constexpr int kWideMinCtas = 3;

template <int Steps, bool Fp32>
__global__ void __launch_bounds__(kWarps * 32, kWideMinCtas) lora_term_wide_kernel(const TermParams a) {
    term_body<Steps, kWideMN, Fp32>(a);
}

template <int Steps, bool Fp32>
void launch(const TermParams& p, unsigned rows_grid, int columns, cudaStream_t stream) {
    if (columns <= kNarrowT) {
        CUDA_CHECK(pdl::launch_consumer({dim3(rows_grid, 1), dim3(kWarps * 32), 0, stream},
                                        lora_term_kernel<Steps, Fp32>, p));
    } else {
        const unsigned tiles = static_cast<unsigned>((columns + kWideMN * 8 - 1) / (kWideMN * 8));
        CUDA_CHECK(pdl::launch_consumer({dim3(rows_grid, tiles), dim3(kWarps * 32), 0, stream},
                                        lora_term_wide_kernel<Steps, Fp32>, p));
    }
}

template <bool Fp32>
void launch_steps(const TermParams& p, int steps, unsigned rows_grid, int columns, cudaStream_t stream) {
    switch (steps) {
    case 1: launch<1, Fp32>(p, rows_grid, columns, stream); break;
    case 2: launch<2, Fp32>(p, rows_grid, columns, stream); break;
    case 3: launch<3, Fp32>(p, rows_grid, columns, stream); break;
    case 4: launch<4, Fp32>(p, rows_grid, columns, stream); break;
    case 5: launch<5, Fp32>(p, rows_grid, columns, stream); break;
    case 6: launch<6, Fp32>(p, rows_grid, columns, stream); break;
    case 7: launch<7, Fp32>(p, rows_grid, columns, stream); break;
    default: launch<8, Fp32>(p, rows_grid, columns, stream); break;
    }
}

} // namespace

void lora_term_add(const LoraTermArgs& a, cudaStream_t stream) {
    if (a.columns <= 0 || a.count <= 0 || a.count > kMaxLoraTargets || a.rank <= 0 || a.rank % 16 != 0 ||
        a.rank > kLoraMaxRank) {
        throw std::invalid_argument("inkling2 lora_term_add: 1-8 targets, positive columns, a padded rank of at most 128");
    }
    TermParams p{};
    p.count   = a.count;
    p.columns = a.columns;
    int ctas  = 0;
    for (int i = 0; i < a.count; ++i) {
        const LoraTarget& t = a.targets[i];
        if (t.b == nullptr || t.u == nullptr || t.y == nullptr || t.rows <= 0 || t.y_stride < t.rows ||
            (t.u_stride != 0 && t.u_stride < a.rank) || reinterpret_cast<std::uintptr_t>(t.b) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(t.u) % 4 != 0) {
            throw std::invalid_argument("inkling2 lora_term_add: a target's B (16-byte aligned), U, y, rows or strides");
        }
        p.targets[i]          = t;
        p.targets[i].u_stride = t.u_stride != 0 ? t.u_stride : a.rank;
        p.cta_begin[i]        = ctas;
        ctas += (t.rows + kTileRows - 1) / kTileRows;
    }
    const auto grid = static_cast<unsigned>(ctas);
    if (a.fp32_out) {
        launch_steps<true>(p, a.rank / 16, grid, a.columns, stream);
    } else {
        launch_steps<false>(p, a.rank / 16, grid, a.columns, stream);
    }
}

} // namespace infernix::ops::inkling2
