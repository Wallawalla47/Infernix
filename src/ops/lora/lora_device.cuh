#pragma once

// The canonical LoRA addend (include/infernix/ops/lora.h) for the projection kernels: one device form of
// ops::LoraAddend, and warp-level helpers that compute the term with the one fixed MMA sequence in either operand
// orientation, so every kernel adds bit-identical terms.

#include "infernix/ops/lora.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <type_traits>

namespace infernix::ops::detail {

// Two classes of adapter instance: an adapter of padded rank up to kLoraNarrowRank runs the narrow instances (the
// term's k-step operands sized for 4 steps), a larger one, up to kLoraMaxRank, the wide instances (8 steps). They are
// separate instances of every kernel, so the narrow ones do not depend on the wide class: a wide adapter costs a narrow
// one nothing.
inline constexpr int kLoraNarrowRank = 64;
static_assert(kLoraMaxRank == 2 * kLoraNarrowRank, "the wide class holds twice the narrow class's k-steps");

// Kernel argument form of LoraAddend (passed by value): a narrow adapter (padded rank <= kLoraNarrowRank).
struct LoraDeviceAddend {
    static constexpr int kMaxRank = kLoraNarrowRank;

    const __nv_bfloat16* b = nullptr; // [rows][rank]
    const __nv_bfloat16* u = nullptr; // [T][u_rows]
    std::int32_t rank      = 0;       // multiple of 16
    std::int32_t u_rows    = 0;
    std::int32_t segments  = 0;
    std::int32_t row_end[4]{};
    std::int32_t u_offset[4]{};
    float scale[4]{};

    // The segment index of parent row `row` (segments are few: compare in order).
    __device__ __forceinline__ int segment_of(int row) const {
        int s = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i) { s += (i + 1 < segments && row >= row_end[i]) ? 1 : 0; }
        return s;
    }
    // Segment s's U offset and scale by selection, not a computed index: the addend is a by-value kernel parameter, and
    // indexing it with a run-time segment would copy it to local memory (STL/LDL) in every adapter instance.
    __device__ __forceinline__ int offset_of(int s) const {
        return s == 0 ? u_offset[0] : s == 1 ? u_offset[1] : s == 2 ? u_offset[2] : u_offset[3];
    }
    __device__ __forceinline__ float scale_of(int s) const {
        return s == 0 ? scale[0] : s == 1 ? scale[1] : s == 2 ? scale[2] : scale[3];
    }
};

// The same for a wide adapter (kLoraNarrowRank < padded rank <= kLoraMaxRank): the narrow form's fields and lookups
// (a narrow adapter's kernels see the narrow struct exactly as before the wide class existed). The helpers whose
// operands are sized by the rank take the class as a template parameter (P::kMaxRank); the rank-agnostic ones take
// the narrow form by reference, which a wide one binds to.
struct LoraDeviceAddendWide : LoraDeviceAddend {
    static constexpr int kMaxRank = kLoraMaxRank;
};

// The k-steps (16 values of the rank each) an instance's term operands hold.
template <class P>
inline constexpr int kLoraSteps = P::kMaxRank / 16;

// body(StaticCount<steps>{}) for the term's k-steps (p.rank / 16), an instance per count of the addend class's
// range: a narrow class's 1 .. 4, a wide class's 5 .. 8 (its padded rank is above the narrow class's). A loop over
// the k-steps then unrolls without a per-step branch.
template <int Min, int Max, class Body>
__device__ __forceinline__ void with_lora_step_count(int steps, Body&& body) {
    if constexpr (Min == Max) {
        body(StaticCount<Min>{});
    } else {
        if (steps >= Max) {
            body(StaticCount<Max>{});
        } else {
            with_lora_step_count<Min, Max - 1>(steps, body);
        }
    }
}
template <class P, class Body>
__device__ __forceinline__ void with_lora_steps(int steps, Body&& body) {
    constexpr int kMax = kLoraSteps<P>;
    constexpr int kMin = kMax > kLoraNarrowRank / 16 ? kLoraNarrowRank / 16 + 1 : 1;
    with_lora_step_count<kMin, kMax>(steps, body);
}

template <class P = LoraDeviceAddend>
[[nodiscard]] inline P lora_device_addend(const LoraAddend& addend) {
    P d;
    d.b        = static_cast<const __nv_bfloat16*>(addend.b);
    d.u        = static_cast<const __nv_bfloat16*>(addend.u);
    d.rank     = addend.rank;
    d.u_rows   = addend.u_rows;
    d.segments = addend.segments;
    for (int i = 0; i < 4; ++i) {
        d.row_end[i]  = addend.segment[i].row_end;
        d.u_offset[i] = addend.segment[i].u_offset;
        d.scale[i]    = addend.segment[i].scale;
    }
    return d;
}

// Marks a projection kernel instantiated with the addend; kernels without it keep their code unchanged.
struct LoraNone {};

// Kernels that take their term as a parameter of their own (no epilogue type) take LoraNone, LoraDeviceAddend or
// LoraDeviceAddendWide.
template <class L>
inline constexpr bool kLoraTerm = std::is_same_v<L, LoraDeviceAddend> || std::is_same_v<L, LoraDeviceAddendWide>;

// Whether a term of this padded rank runs the narrow instances.
[[nodiscard]] inline bool lora_narrow(const LoraAddend& addend) { return addend.rank <= kLoraNarrowRank; }

// Calls launch(term) with LoraNone, or with the device form of `lora` (from column `first_column`: a token slice's)
// of its rank's class.
template <class Launch>
void with_lora_term(const LoraAddend* lora, std::int32_t first_column, Launch&& launch) {
    if (lora == nullptr || !lora->present()) {
        launch(LoraNone{});
    } else if (lora_narrow(*lora)) {
        launch(lora_device_addend<LoraDeviceAddend>(lora_columns_from(*lora, first_column)));
    } else {
        launch(lora_device_addend<LoraDeviceAddendWide>(lora_columns_from(*lora, first_column)));
    }
}

__device__ __forceinline__ unsigned lora_load_pair(const __nv_bfloat16* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

// v + term where the row takes a term (`term` = fl32(c * scale)); v unchanged otherwise.
__device__ __forceinline__ float lora_add(float value, float term, bool has) {
    return has ? __fadd_rn(value, term) : value;
}

// A single-value epilogue (identity, residual add) in the adapter instance: the inner epilogue on the kernel's own
// value expression, so its arithmetic (a product fused with the residual's add included) is the no-adapter
// kernel's, then the term added to its FP32 result. With B = 0 the output is the no-adapter kernel's bits.
template <class Epilogue>
__device__ __forceinline__ float lora_epilogue_apply(const Epilogue& epilogue, int row, int token, float value,
                                                     float term, bool has) {
    return lora_add(epilogue.apply(row, token, value), term, has);
}

// The addend for a warp whose rows are `rows[0..Rows)` (Rows <= 16, < 0: none) over the columns token0 ..
// token0 + Tokens - 1 (Tokens <= 8; columns >= live get none), delivered to every lane: add[r][t] = fl32(c * scale)
// for rows[r] and column token0 + t, and has[r] whether rows[r] takes a term at all (its segment is adapted). The
// caller adds a term as lora_add(value, add, has). Rows of different segments take one chain per segment, the
// MMA m16n8k16 chain over the rank from +0 with B's rows as the A operand (fragment rows g and g + 8) and U's column
// as the B operand (column g): chain(s, local_lo, local_hi, row_lo, row_hi, token_g, c) runs segment s's for this
// lane's fragment rows (local_lo = g, local_hi = g + 8 among the warp's rows, or -1; row_lo / row_hi their parent
// rows) and column token_g (< 0: zero), c as the MMA's C fragment. Every lane must call it with the same arguments
// (the MMA is warp-wide).
template <int Rows, int Tokens, class Chain>
__device__ __forceinline__ void lora_warp_rows_with(const LoraDeviceAddend& p, const int (&rows)[Rows], int token0,
                                                    int live, float (&add)[Rows][Tokens], bool (&has)[Rows],
                                                    Chain chain) {
    static_assert(Rows >= 1 && Rows <= 16 && Tokens >= 1 && Tokens <= 8);
    const int lane = threadIdx.x & 31, g = lane >> 2;
    int seg[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        seg[r] = rows[r] < 0 ? -1 : p.segment_of(rows[r]);
        has[r] = seg[r] >= 0 && p.offset_of(seg[r]) >= 0;
#pragma unroll
        for (int t = 0; t < Tokens; ++t) { add[r][t] = 0.0F; }
    }
    const int token_g = g < Tokens && token0 + g < live ? token0 + g : -1;
    for (int s = 0; s < p.segments; ++s) {
        if (p.offset_of(s) < 0) { continue; }
        bool any = false;
#pragma unroll
        for (int r = 0; r < Rows; ++r) { any |= seg[r] == s; }
        if (!any) { continue; }
        // This lane's fragment rows g and g + 8 (the warp's rows in order; others zero).
        int row_lo = -1, row_hi = -1, local_lo = -1, local_hi = -1;
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r == g && seg[r] == s) {
                row_lo   = rows[r];
                local_lo = r;
            }
            if (r == g + 8 && seg[r] == s) {
                row_hi   = rows[r];
                local_hi = r;
            }
        }
        float c[4];
        chain(s, local_lo, local_hi, row_lo, row_hi, token_g, c);
        const float scale = p.scale_of(s);
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
            for (int t = 0; t < Tokens; ++t) {
                // Element (r, t) lives in lane 4 * (r % 8) + t / 2, component (r >= 8 ? 2 : 0) + t % 2.
                const float v0 = __shfl_sync(0xffffffffU, (t & 1) ? c[1] : c[0], 4 * (r & 7) + (t >> 1));
                const float v1 = __shfl_sync(0xffffffffU, (t & 1) ? c[3] : c[2], 4 * (r & 7) + (t >> 1));
                const float chain = r < 8 ? v0 : v1;
                if (seg[r] == s && token0 + t < live) { add[r][t] = __fmul_rn(chain, scale); }
            }
        }
    }
}

// A warp's term operands in shared memory, for a kernel whose warps own rows and stream their weights (a GEMV): at the
// end of the main loop B's rows are no longer in L2 (the weight stream evicts what a prefetch brought), so loading
// them there waits a DRAM round trip, in every wave. The warp copies its Rows rows of B before the PDL wait
// (B is immutable) and, after it, U's columns token0 .. token0 + Tokens - 1 of each adapted segment its rows touch
// (contiguous or paired rows touch at most min(Rows, 4)); lora_warp_rows_staged reads both from here. Rows of
// rank * 2 + 16 bytes (an odd number of 16-byte pieces), sized for the instance's class (MaxRank: the addend's
// kMaxRank, kLoraStageRank of the kernel's epilogue or term type).
template <int Rows, int Tokens, int MaxRank = kLoraNarrowRank>
struct LoraWarpStage {
    static constexpr int kStride = 2 * MaxRank + 16;
    static constexpr int kSlots  = Rows < 4 ? Rows : 4;
    alignas(16) unsigned char b[Rows][kStride];
    alignas(16) unsigned char u[kSlots][Tokens][kStride];
};

// The adapted segments rows[0..Rows) touch, as a mask (slot of segment s = the set bits below s).
template <int Rows>
__device__ __forceinline__ unsigned lora_touched_segments(const LoraDeviceAddend& p, const int (&rows)[Rows]) {
    unsigned touched = 0;
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        const int s = rows[r] < 0 ? 0 : p.segment_of(rows[r]);
        touched |= rows[r] >= 0 && p.offset_of(s) >= 0 ? 1U << s : 0U;
    }
    return touched;
}

// B's rows of the warp (before the PDL wait: immutable). Warp-wide; the copies complete at lora_warp_rows_staged.
template <int Rows, int Tokens, class P, int MaxRank>
__device__ __forceinline__ void lora_stage_warp_b(const P& p, const int (&rows)[Rows],
                                                  LoraWarpStage<Rows, Tokens, MaxRank>& stage) {
    static_assert(MaxRank == P::kMaxRank, "the stage is sized for the addend's class");
    const int lane = threadIdx.x & 31, pieces = p.rank >> 3, stride = 2 * p.rank + 16;
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        if (lane < pieces && rows[r] >= 0) {
            cp_async<16>(&stage.b[0][0] + r * stride + lane * 16,
                         p.b + static_cast<std::int64_t>(rows[r]) * p.rank + lane * 8);
        }
    }
    cp_commit();
}

// U's columns token0 .. token0 + Tokens - 1 (zero from live) of the warp's adapted segments (after the PDL wait: the
// shrink's output). Warp-wide.
template <int Rows, int Tokens, class P, int MaxRank>
__device__ __forceinline__ void lora_stage_warp_u(const P& p, const int (&rows)[Rows], int token0, int live,
                                                  LoraWarpStage<Rows, Tokens, MaxRank>& stage) {
    static_assert(MaxRank == P::kMaxRank, "the stage is sized for the addend's class");
    const int lane = threadIdx.x & 31, pieces = p.rank >> 3, stride = 2 * p.rank + 16;
    const unsigned touched = lora_touched_segments(p, rows);
    int slot               = 0;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        if (((touched >> s) & 1U) == 0U || slot >= LoraWarpStage<Rows, Tokens, MaxRank>::kSlots) { continue; }
        const int uo = p.offset_of(s);
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            if (lane < pieces) {
                const bool valid = token0 + t < live;
                cp_async_zfill<16>(&stage.u[0][0][0] + (slot * Tokens + t) * stride + lane * 16,
                                   p.u + static_cast<std::int64_t>(valid ? token0 + t : 0) * p.u_rows + uo + lane * 8,
                                   valid ? 16 : 0);
            }
        }
        ++slot;
    }
    cp_commit();
}

// The warp's addend (as lora_warp_rows_with describes) on the operands lora_stage_warp_b / lora_stage_warp_u copied
// (the same rows and live; U staged from column stage_token0, these Tokens columns within it). Waits for the warp's
// copies. Warp-wide.
template <int Rows, int Tokens, int StageTokens, class P, int MaxRank>
__device__ __forceinline__ void lora_warp_rows_staged(const P& p, const int (&rows)[Rows], int token0, int live,
                                                      float (&add)[Rows][Tokens], bool (&has)[Rows],
                                                      const LoraWarpStage<Rows, StageTokens, MaxRank>& stage,
                                                      int stage_token0) {
    static_assert(Tokens <= StageTokens);
    static_assert(MaxRank == P::kMaxRank, "the stage is sized for the addend's class");
    cp_wait<0>();
    __syncwarp();
    const int q = threadIdx.x & 3, stride = 2 * p.rank + 16;
    const unsigned touched = lora_touched_segments(p, rows);
    lora_warp_rows_with<Rows, Tokens>(
        p, rows, token0, live, add, has,
        [&](int s, int local_lo, int local_hi, int, int, int token_g, float (&c)[4]) {
            const int slot              = __popc(touched & ((1U << s) - 1U));
            const unsigned char* b_lo   = &stage.b[0][0] + (local_lo < 0 ? 0 : local_lo) * stride;
            const unsigned char* b_hi   = &stage.b[0][0] + (local_hi < 0 ? 0 : local_hi) * stride;
            const unsigned char* u_col  = &stage.u[0][0][0] +
                                         (slot * StageTokens + (token_g < 0 ? 0 : token_g - stage_token0)) * stride;
            c[0] = c[1] = c[2] = c[3] = 0.0F;
            for (int k = 0; k < p.rank; k += 16) {
                const int at      = (k + 2 * q) * 2;
                const unsigned a0 = local_lo < 0 ? 0U : *reinterpret_cast<const unsigned*>(b_lo + at);
                const unsigned a1 = local_hi < 0 ? 0U : *reinterpret_cast<const unsigned*>(b_hi + at);
                const unsigned a2 = local_lo < 0 ? 0U : *reinterpret_cast<const unsigned*>(b_lo + at + 16);
                const unsigned a3 = local_hi < 0 ? 0U : *reinterpret_cast<const unsigned*>(b_hi + at + 16);
                const unsigned b0 = token_g < 0 ? 0U : *reinterpret_cast<const unsigned*>(u_col + at);
                const unsigned b1 = token_g < 0 ? 0U : *reinterpret_cast<const unsigned*>(u_col + at + 16);
                mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b0, b1);
            }
        });
}

template <int Rows, int Tokens, class P, int MaxRank>
__device__ __forceinline__ void lora_warp_rows_staged(const P& p, const int (&rows)[Rows], int token0, int live,
                                                      float (&add)[Rows][Tokens], bool (&has)[Rows],
                                                      const LoraWarpStage<Rows, Tokens, MaxRank>& stage) {
    lora_warp_rows_staged<Rows, Tokens, Tokens>(p, rows, token0, live, add, has, stage, token0);
}

// One staged row's term (lora_stage_warp_b / lora_stage_warp_u with Tokens = T from token0) over the columns
// token0 .. token0 + T - 1 (any T; columns >= live get none): every lane receiving all T (row_all), or lane t
// column t's and other lanes 0 (row_tokens). Warp-wide.
template <int T, class P, int MaxRank>
__device__ __forceinline__ void lora_warp_row_all_staged(const P& p, int row, int token0, int live, float (&add)[T],
                                                         bool& has, const LoraWarpStage<1, T, MaxRank>& stage) {
    constexpr int kChunk = T < 8 ? T : 8;
    const int rows[1]    = {row};
    has                  = false;
#pragma unroll
    for (int c = 0; c < T; c += kChunk) {
        float chunk[1][kChunk];
        bool chunk_has[1];
        lora_warp_rows_staged<1, kChunk, T>(p, rows, token0 + c, live, chunk, chunk_has, stage, token0);
        has = chunk_has[0];
#pragma unroll
        for (int t = 0; t < kChunk; ++t) {
            if (c + t < T) { add[c + t] = chunk[0][t]; }
        }
    }
}

template <int T, class P, int MaxRank>
__device__ __forceinline__ float lora_warp_row_tokens_staged(const P& p, int row, int token0, int live, bool& has,
                                                             const LoraWarpStage<1, T, MaxRank>& stage) {
    constexpr int kChunk = T < 8 ? T : 8;
    const int lane       = threadIdx.x & 31;
    const int rows[1]    = {row};
    float mine           = 0.0F;
    has                  = false;
#pragma unroll
    for (int c = 0; c < T; c += kChunk) {
        float add[1][kChunk];
        bool chunk_has[1];
        lora_warp_rows_staged<1, kChunk, T>(p, rows, token0 + c, live, add, chunk_has, stage, token0);
        has = chunk_has[0];
#pragma unroll
        for (int t = 0; t < kChunk; ++t) {
            if (c + t == lane) { mine = add[0][t]; }
        }
    }
    return mine;
}

// Row orientation tile: acc[MM][MN][4] are MMA C fragments whose rows are weight rows row_of(mm, i) (i in 0..15) and
// whose columns are tokens token_of(mn, j) (j in 0..7; < 0 past the call). Each element must already hold the
// kernel's scaled value; the canonical term is added to the elements of adapted rows. Rows 0-7 and 8-15 of a fragment
// may lie in different segments (paired gate/up rows); each 8-row half lies in one. Warp-wide.
template <int MM, int MN, class P, class RowOf, class TokenOf>
__device__ __forceinline__ void lora_add_rows_tile(const P& p, float (&acc)[MM][MN][4], RowOf row_of,
                                                   TokenOf token_of) {
    constexpr int kLoraMaxSteps = kLoraSteps<P>;
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int steps = p.rank >> 4;
#pragma unroll
    for (int mm = 0; mm < MM; ++mm) {
        const int seg_lo = p.segment_of(row_of(mm, 0)), seg_hi = p.segment_of(row_of(mm, 8));
        const int row_lo = row_of(mm, g), row_hi = row_of(mm, g + 8);
        // One chain covers both halves when they share a segment; otherwise one chain a half, the other half's rows
        // zero (an element's chain does not depend on the fragment's other rows).
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const bool shared = seg_lo == seg_hi;
            if (half == 1 && shared) { break; }
            const int seg = half == 0 ? seg_lo : seg_hi;
            const int uo  = p.offset_of(seg);
            if (uo < 0) { continue; }
            const bool use_lo = half == 0, use_hi = half == 1 || shared;
            const __nv_bfloat16* b_lo = p.b + static_cast<std::int64_t>(row_lo) * p.rank;
            const __nv_bfloat16* b_hi = p.b + static_cast<std::int64_t>(row_hi) * p.rank;
            unsigned a[kLoraMaxSteps][4];
#pragma unroll
            for (int ks = 0; ks < kLoraMaxSteps; ++ks) {
                if (ks < steps) {
                    const int k = ks * 16 + 2 * q;
                    a[ks][0]    = use_lo ? lora_load_pair(b_lo + k) : 0U;
                    a[ks][1]    = use_hi ? lora_load_pair(b_hi + k) : 0U;
                    a[ks][2]    = use_lo ? lora_load_pair(b_lo + k + 8) : 0U;
                    a[ks][3]    = use_hi ? lora_load_pair(b_hi + k + 8) : 0U;
                }
            }
            const float scale = p.scale_of(seg);
#pragma unroll
            for (int mn = 0; mn < MN; ++mn) {
                const int token = token_of(mn, g);
                const __nv_bfloat16* u_col =
                    p.u + static_cast<std::int64_t>(token < 0 ? 0 : token) * p.u_rows + uo;
                float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int ks = 0; ks < kLoraMaxSteps; ++ks) {
                    if (ks < steps) {
                        const int k       = ks * 16 + 2 * q;
                        const unsigned b0 = token < 0 ? 0U : lora_load_pair(u_col + k);
                        const unsigned b1 = token < 0 ? 0U : lora_load_pair(u_col + k + 8);
                        mma_bf16(c[0], c[1], c[2], c[3], a[ks][0], a[ks][1], a[ks][2], a[ks][3], b0, b1);
                    }
                }
                if (use_lo) {
                    acc[mm][mn][0] = __fadd_rn(acc[mm][mn][0], __fmul_rn(c[0], scale));
                    acc[mm][mn][1] = __fadd_rn(acc[mm][mn][1], __fmul_rn(c[1], scale));
                }
                if (use_hi) {
                    acc[mm][mn][2] = __fadd_rn(acc[mm][mn][2], __fmul_rn(c[2], scale));
                    acc[mm][mn][3] = __fadd_rn(acc[mm][mn][3], __fmul_rn(c[3], scale));
                }
            }
        }
    }
}

// Token orientation, one fragment column: the 8 weight rows of an MMA C fragment's columns (one segment), their
// segment's U offset (< 0: unadapted, nothing else loaded) and scale, and this lane's B operand (row_g = the
// fragment's column g's parent row, lane g = lane / 4), for an addend of class P.
template <class P>
struct LoraTokensColumn {
    int uo;
    float scale;
    unsigned b[kLoraSteps<P>][2];
};

template <class P>
__device__ __forceinline__ LoraTokensColumn<P> lora_tokens_column(const P& p, int row0, int row_g) {
    constexpr int kLoraMaxSteps = kLoraSteps<P>;
    const int q = threadIdx.x & 3, steps = p.rank >> 4;
    LoraTokensColumn<P> column;
    const int seg = p.segment_of(row0);
    column.uo     = p.offset_of(seg);
    if (column.uo < 0) { return column; }
    column.scale             = p.scale_of(seg);
    const __nv_bfloat16* b_g = p.b + static_cast<std::int64_t>(row_g) * p.rank;
#pragma unroll
    for (int ks = 0; ks < kLoraMaxSteps; ++ks) {
        if (ks < steps) {
            column.b[ks][0] = lora_load_pair(b_g + ks * 16 + 2 * q);
            column.b[ks][1] = lora_load_pair(b_g + ks * 16 + 8 + 2 * q);
        }
    }
    return column;
}

// The chain c of one token-orientation fragment of an adapted column: fragment rows = tokens t_lo (row g) and t_hi
// (row g + 8), < 0 past the call (zero operands), the m16n8k16 MMA chain over the rank from +0 with U's columns as
// the A operand. Warp-wide.
template <class P>
__device__ __forceinline__ void lora_tokens_fragment(const P& p, const LoraTokensColumn<P>& column, int t_lo, int t_hi,
                                                     float (&c)[4]) {
    constexpr int kLoraMaxSteps = kLoraSteps<P>;
    const int q = threadIdx.x & 3, steps = p.rank >> 4;
    const __nv_bfloat16* u_lo = p.u + static_cast<std::int64_t>(t_lo < 0 ? 0 : t_lo) * p.u_rows + column.uo;
    const __nv_bfloat16* u_hi = p.u + static_cast<std::int64_t>(t_hi < 0 ? 0 : t_hi) * p.u_rows + column.uo;
    c[0] = c[1] = c[2] = c[3] = 0.0F;
#pragma unroll
    for (int ks = 0; ks < kLoraMaxSteps; ++ks) {
        if (ks < steps) {
            const int k       = ks * 16 + 2 * q;
            const unsigned a0 = t_lo < 0 ? 0U : lora_load_pair(u_lo + k);
            const unsigned a1 = t_hi < 0 ? 0U : lora_load_pair(u_hi + k);
            const unsigned a2 = t_lo < 0 ? 0U : lora_load_pair(u_lo + k + 8);
            const unsigned a3 = t_hi < 0 ? 0U : lora_load_pair(u_hi + k + 8);
            mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, column.b[ks][0], column.b[ks][1]);
        }
    }
}

// Token orientation tile: acc[MT][MR][4] are MMA C fragments whose rows are tokens token_of(mt, i) (i in 0..15; < 0
// past the call) and whose columns are weight rows row_of(mr, j) (j in 0..7, one segment a fragment). As above.
// (The chain is lora_tokens_fragment's, written out: the tile kernels' register allocation was measured on this form.)
template <int MT, int MR, class P, class TokenOf, class RowOf>
__device__ __forceinline__ void lora_add_tokens_tile(const P& p, float (&acc)[MT][MR][4], TokenOf token_of,
                                                     RowOf row_of) {
    constexpr int kLoraMaxSteps = kLoraSteps<P>;
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
    const int steps = p.rank >> 4;
#pragma unroll
    for (int mr = 0; mr < MR; ++mr) {
        const int seg = p.segment_of(row_of(mr, 0));
        const int uo  = p.offset_of(seg);
        if (uo < 0) { continue; }
        const float scale         = p.scale_of(seg);
        const __nv_bfloat16* b_g = p.b + static_cast<std::int64_t>(row_of(mr, g)) * p.rank;
        unsigned b[kLoraMaxSteps][2];
#pragma unroll
        for (int ks = 0; ks < kLoraMaxSteps; ++ks) {
            if (ks < steps) {
                b[ks][0] = lora_load_pair(b_g + ks * 16 + 2 * q);
                b[ks][1] = lora_load_pair(b_g + ks * 16 + 8 + 2 * q);
            }
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int t_lo = token_of(mt, g), t_hi = token_of(mt, g + 8);
            const __nv_bfloat16* u_lo = p.u + static_cast<std::int64_t>(t_lo < 0 ? 0 : t_lo) * p.u_rows + uo;
            const __nv_bfloat16* u_hi = p.u + static_cast<std::int64_t>(t_hi < 0 ? 0 : t_hi) * p.u_rows + uo;
            float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
            for (int ks = 0; ks < kLoraMaxSteps; ++ks) {
                if (ks < steps) {
                    const int k       = ks * 16 + 2 * q;
                    const unsigned a0 = t_lo < 0 ? 0U : lora_load_pair(u_lo + k);
                    const unsigned a1 = t_hi < 0 ? 0U : lora_load_pair(u_hi + k);
                    const unsigned a2 = t_lo < 0 ? 0U : lora_load_pair(u_lo + k + 8);
                    const unsigned a3 = t_hi < 0 ? 0U : lora_load_pair(u_hi + k + 8);
                    mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b[ks][0], b[ks][1]);
                }
            }
#pragma unroll
            for (int v = 0; v < 4; ++v) { acc[mt][mr][v] = __fadd_rn(acc[mt][mr][v], __fmul_rn(c[v], scale)); }
        }
    }
}

// lora_add_rows_tile for a CTA tile kernel whose consumer threads have finished with `smem` (SmemBytes, free) and
// hold the tile's fragments, every consumer thread calling it; a 16-row fragment lies in one segment (its rows a
// 16-aligned run, as the ring tiles' are). The CTA copies B's rows of the tile once (cta_row(i): parent row of tile
// row i) and, one adapted segment a pass (as many as fit), U's columns of the tile's BlockTokens tokens from
// token_begin (zero from token_count); each warp then runs its fragments of the pass's segments: B's rows by ldmatrix
// into registers (the A operand, the whole rank), U's columns by ldmatrix (the B operand). Every warp has fragments in
// every segment and pass, so none idles. local_row(mm, i): fragment row i's tile row; local_token(mn, j): fragment
// column j's tile token. The operands, chain and order are lora_add_rows_tile's: the same bits.
template <int BlockRows, int BlockTokens, int SmemBytes, int MM, int MN, class P, class LocalRow, class LocalToken,
          class CtaRow, class Barrier>
__device__ __forceinline__ void lora_add_rows_tile_staged(const P& p, float (&acc)[MM][MN][4], LocalRow local_row,
                                                          LocalToken local_token, CtaRow cta_row, int token_begin,
                                                          int token_count, unsigned char* smem, int thread, int threads,
                                                          Barrier barrier) {
    constexpr int kMaxSteps  = kLoraSteps<P>;
    constexpr int kPieces    = P::kMaxRank / 8;
    constexpr int kPieceBits = kPieces == 16 ? 4 : 3;
    static_assert(kPieces == 1 << kPieceBits);
    constexpr int kMaxStride = 2 * P::kMaxRank + 16;
    static_assert(BlockRows % 16 == 0 && (BlockRows + BlockTokens) * kMaxStride <= SmemBytes,
                  "B's rows and one segment's U columns over the tile must fit");
    const int lane = threadIdx.x & 31;
    const int steps = p.rank >> 4, pieces = p.rank >> 3, stride = 2 * p.rank + 16;
    unsigned touched = 0;
#pragma unroll
    for (int f = 0; f < BlockRows / 16; ++f) {
        const int s = p.segment_of(cta_row(16 * f));
        touched |= p.offset_of(s) >= 0 ? 1U << s : 0U;
    }
    if (touched == 0) { return; } // tile-uniform; shared memory untouched
    cp_wait<0>();
    barrier(); // every consumer is done with the main loop's shared memory
    unsigned char* b_s = smem;
    unsigned char* u_s = smem + BlockRows * stride;
    const int u_bytes  = BlockTokens * stride;
    const int fit      = (SmemBytes - BlockRows * stride) / u_bytes;
    const int live     = token_count - token_begin;
    for (int i = thread; i < BlockRows * kPieces; i += threads) {
        const int row = i >> kPieceBits, piece = i & (kPieces - 1);
        if (piece < pieces) {
            cp_async<16>(b_s + row * stride + piece * 16, p.b + static_cast<std::int64_t>(cta_row(row)) * p.rank + piece * 8);
        }
    }
    // ldmatrix lane roles: B's rows (the A operand) x4 = rows 0-7 / 8-15 at piece 2ks, then at 2ks + 1; U's tokens (the
    // B operand) x2 = 8 tokens at piece 2ks, then 2ks + 1.
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_piece = lane >> 4;
    const int u_row = lane & 7, u_piece = (lane >> 3) & 1;
    bool first = true;
    while (touched != 0) {
        unsigned pass = 0;
        int slots     = 0;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (((touched >> s) & 1U) != 0U && slots < fit) {
                pass |= 1U << s;
                ++slots;
            }
        }
        touched &= ~pass;
        if (!first) { barrier(); } // the previous pass's reads finish before its U columns are overwritten
        first    = false;
        int slot = 0;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (((pass >> s) & 1U) == 0U) { continue; }
            const int uo        = p.offset_of(s);
            unsigned char* dest = u_s + slot * u_bytes;
            for (int i = thread; i < BlockTokens * kPieces; i += threads) {
                const int t = i >> kPieceBits, piece = i & (kPieces - 1);
                if (piece < pieces) {
                    const bool valid = t < live;
                    cp_async_zfill<16>(dest + t * stride + piece * 16,
                                       p.u + static_cast<std::int64_t>(token_begin + (valid ? t : 0)) * p.u_rows + uo +
                                           piece * 8,
                                       valid ? 16 : 0);
                }
            }
            ++slot;
        }
        cp_commit();
        cp_wait<0>();
        barrier();
#pragma unroll
        for (int mm = 0; mm < MM; ++mm) {
            const int seg = p.segment_of(cta_row(local_row(mm, 0)));
            if (((pass >> seg) & 1U) == 0U) { continue; }
            const unsigned char* u_slot = u_s + __popc(pass & ((1U << seg) - 1U)) * u_bytes;
            const float scale           = p.scale_of(seg);
            const unsigned char* b_base = b_s + local_row(mm, a_row) * stride;
            unsigned a[kMaxSteps][4];
#pragma unroll
            for (int ks = 0; ks < kMaxSteps; ++ks) {
                if (ks < steps) {
                    ldmatrix_x4(a[ks][0], a[ks][1], a[ks][2], a[ks][3], smem_addr(b_base + (2 * ks + a_piece) * 16));
                }
            }
#pragma unroll
            for (int mn = 0; mn < MN; ++mn) {
                const unsigned char* u_base = u_slot + local_token(mn, u_row) * stride;
                float c[4]                  = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int ks = 0; ks < kMaxSteps; ++ks) {
                    if (ks < steps) {
                        unsigned b0, b1;
                        ldmatrix_x2(b0, b1, smem_addr(u_base + (2 * ks + u_piece) * 16));
                        mma_bf16(c[0], c[1], c[2], c[3], a[ks][0], a[ks][1], a[ks][2], a[ks][3], b0, b1);
                    }
                }
#pragma unroll
                for (int v = 0; v < 4; ++v) { acc[mm][mn][v] = __fadd_rn(acc[mm][mn][v], __fmul_rn(c[v], scale)); }
            }
        }
    }
    barrier(); // the last pass's reads finish before the caller reuses shared memory
}

// lora_add_tokens_tile for a CTA tile kernel whose consumer warps (WarpsTokens x WarpsRows: warp (warp_m, warp_n)
// holds the fragments of token group warp_m and row group warp_n) have finished with `smem` (SmemBytes, free); every
// consumer thread calls it. The CTA copies B's rows of the tile once (cta_row(i): parent row of tile row i). Each token
// group's warps then walk their work items, one 16-token fragment row (mt) and one adapted segment at a time: the
// group's two slots receive the items' U columns two items ahead (the group's warps share the copies), each warp takes
// the whole rank of U into registers by ldmatrix, the item after next starts into the slot just read, and the warp runs
// the segment's row fragments with B by ldmatrix from the shared rows under those copies. So every warp works through
// the whole term, U is read once a tile and each copy has two items' MMAs to land. A group synchronizes on its own named
// barrier (2 + warp_m, WarpsRows warps). The operands, chain and order are lora_add_tokens_tile's: the same bits.
// LowRegisters (a kernel whose accumulators leave few registers, the FP8 tiles): instead of the whole rank of U, each
// warp holds two row fragments' chains and one k-step of U at a time, reading U again from the buffer for each pair (the
// same chains in the same order: the same bits).
template <int BlockRows, int SmemBytes, int WarpsTokens, int WarpsRows, bool LowRegisters = false, int MT, int MR,
          class P, class LocalToken, class LocalRow, class CtaRow, class Barrier>
__device__ __forceinline__ void lora_add_tokens_tile_group_staged(const P& p, float (&acc)[MT][MR][4],
                                                                  LocalToken local_token, LocalRow local_row,
                                                                  CtaRow cta_row, int token_begin, int token_count,
                                                                  unsigned char* smem, int thread, int warp_m,
                                                                  Barrier barrier) {
    constexpr int kMaxSteps   = kLoraSteps<P>;
    constexpr int kPieces     = P::kMaxRank / 8;
    constexpr int kPieceBits  = kPieces == 16 ? 4 : 3;
    static_assert(kPieces == 1 << kPieceBits);
    constexpr int kMaxStride  = 2 * P::kMaxRank + 16;
    constexpr int kThreads    = WarpsTokens * WarpsRows * 32;
    constexpr int kGroupLanes = WarpsRows * 32;
    constexpr int kSlots      = 2; // two U slots a token group: the next items' copies land under this item's MMAs
    static_assert(BlockRows % 8 == 0 && WarpsTokens + 2 <= 16, "named barriers 2 .. 1 + WarpsTokens");
    static_assert((BlockRows + WarpsTokens * kSlots * 16) * kMaxStride <= SmemBytes,
                  "B's rows and the 16-token U buffers of the token groups must fit");
    const int lane = threadIdx.x & 31;
    const int steps = p.rank >> 4, pieces = p.rank >> 3, stride = 2 * p.rank + 16;
    unsigned touched = 0;
#pragma unroll
    for (int f = 0; f < BlockRows / 8; ++f) {
        const int s = p.segment_of(cta_row(8 * f));
        touched |= p.offset_of(s) >= 0 ? 1U << s : 0U;
    }
    if (touched == 0) { return; } // tile-uniform; shared memory untouched
    // Every consumer is done with the main loop's shared memory, and no copy of this thread's is in flight.
    cp_wait<0>();
    barrier();
    unsigned char* b_s     = smem;
    unsigned char* u_group = smem + BlockRows * stride + warp_m * kSlots * 16 * stride;
    for (int i = thread; i < BlockRows * kPieces; i += kThreads) {
        const int row = i >> kPieceBits, piece = i & (kPieces - 1);
        if (piece < pieces) {
            cp_async<16>(b_s + row * stride + piece * 16, p.b + static_cast<std::int64_t>(cta_row(row)) * p.rank + piece * 8);
        }
    }
    // The segments the group's row fragments touch (every warp of a token group has the same row-fragment segments:
    // the tile's rows are split among the row warps alike, so this is the union over the tile's segments).
    const unsigned mine = touched;
    const int live      = token_count - token_begin;
    const int group_lane = thread % kGroupLanes;
    // Item (mt, s)'s U columns into the group's buffer `slot`: the group's lanes share the 16 x pieces copies.
    const auto copy_u = [&](int mt, int s, int slot) {
        const int uo     = p.offset_of(s);
        unsigned char* d = u_group + slot * 16 * stride;
        for (int i = group_lane; i < 16 * kPieces; i += kGroupLanes) {
            const int t = i >> kPieceBits, piece = i & (kPieces - 1);
            if (piece < pieces) {
                const int token  = local_token(mt, t);
                const bool valid = token < live;
                cp_async_zfill<16>(d + t * stride + piece * 16,
                                   p.u + static_cast<std::int64_t>(token_begin + (valid ? token : 0)) * p.u_rows + uo +
                                       piece * 8,
                                   valid ? 16 : 0);
            }
        }
    };
    const int first_segment = __ffs(static_cast<int>(mine)) - 1;
    copy_u(0, first_segment, 0);
    cp_commit();
    cp_wait<0>();
    barrier(); // B's rows and every group's first item are in shared memory
    const int group_barrier = 2 + warp_m;
    const auto group_sync   = [&] { asm volatile("bar.sync %0, %1;" ::"r"(group_barrier), "r"(kGroupLanes) : "memory"); };
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_piece = lane >> 4;
    const int b_row = lane & 7, b_piece = lane >> 3;
    const auto next_item = [&](int mt, int s) {
        const unsigned later = mine & ~((2U << s) - 1U);
        return later != 0U ? int2{mt, __ffs(static_cast<int>(later)) - 1} : int2{mt + 1, first_segment};
    };
    if constexpr (LowRegisters) {
        int item = 0;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                if (((mine >> s) & 1U) == 0U) { continue; }
                if (item > 0) {
                    cp_wait<0>();
                    group_sync(); // this item's copy is visible; the other slot's last reads (the item before) are done
                }
                {
                    const int2 next = next_item(mt, s);
                    const int next_mt = next.x, next_s = next.y;
                    if (next_mt < MT) { copy_u(next_mt, next_s, (item + 1) & 1); }
                    cp_commit();
                }
                const unsigned char* u_base = u_group + (item & 1) * 16 * stride + a_row * stride;
                const float scale           = p.scale_of(s);
                // Two row fragments' chains at a time; one k-step of U and of each fragment's B in registers.
#pragma unroll
                for (int m0 = 0; m0 < MR; m0 += 2) {
                    const bool take0 = p.segment_of(cta_row(local_row(m0, 0))) == s;
                    const bool take1 = m0 + 1 < MR && p.segment_of(cta_row(local_row(m0 + 1 < MR ? m0 + 1 : m0, 0))) == s;
                    if (!take0 && !take1) { continue; }
                    const unsigned char* b0_base = b_s + local_row(m0, b_row) * stride;
                    const unsigned char* b1_base = b_s + local_row(m0 + 1 < MR ? m0 + 1 : m0, b_row) * stride;
                    float c0[4] = {0.0F, 0.0F, 0.0F, 0.0F}, c1[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                    for (int ks = 0; ks < kMaxSteps; ++ks) {
                        if (ks < steps) {
                            unsigned a0, a1, a2, a3, b00, b01, b10, b11;
                            ldmatrix_x4(a0, a1, a2, a3, smem_addr(u_base + (2 * ks + a_piece) * 16));
                            ldmatrix_x4(b00, b01, b10, b11, smem_addr((b_piece < 2 ? b0_base : b1_base) +
                                                                      (2 * ks + (b_piece & 1)) * 16));
                            if (take0) { mma_bf16(c0[0], c0[1], c0[2], c0[3], a0, a1, a2, a3, b00, b01); }
                            if (take1) { mma_bf16(c1[0], c1[1], c1[2], c1[3], a0, a1, a2, a3, b10, b11); }
                        }
                    }
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        if (take0) { acc[mt][m0][v] = __fadd_rn(acc[mt][m0][v], __fmul_rn(c0[v], scale)); }
                        if (m0 + 1 < MR && take1) {
                            acc[mt][m0 + 1][v] = __fadd_rn(acc[mt][m0 + 1][v], __fmul_rn(c1[v], scale));
                        }
                    }
                }
                ++item;
            }
        }
        cp_wait<0>();
        barrier(); // every group's reads finish before the caller reuses shared memory
        return;
    }
    // Two items' copies in flight: U goes into registers right after its wait, so a slot takes the item after next as
    // soon as both warps of the group have read it, and that copy has two items' MMAs to land.
    {
        const int2 second = next_item(0, first_segment);
        if (second.x < MT) { copy_u(second.x, second.y, 1); }
        cp_commit();
    }
    int item = 0;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (((mine >> s) & 1U) == 0U) { continue; }
            if (item > 0) {
                cp_wait<1>(); // this item's copy (the one after it may still be in flight)
                group_sync(); // the group's copies of this item are visible
            }
            const unsigned char* u_base = u_group + (item & 1) * 16 * stride + a_row * stride;
            unsigned a[kMaxSteps][4];
#pragma unroll
            for (int ks = 0; ks < kMaxSteps; ++ks) {
                if (ks < steps) {
                    ldmatrix_x4(a[ks][0], a[ks][1], a[ks][2], a[ks][3], smem_addr(u_base + (2 * ks + a_piece) * 16));
                }
            }
            group_sync(); // every warp of the group holds this item's U: its slot takes the item after next
            {
                const int2 next  = next_item(mt, s);
                const int2 after = next.x < MT ? next_item(next.x, next.y) : int2{MT, 0};
                if (after.x < MT) { copy_u(after.x, after.y, item & 1); }
                cp_commit();
            }
            const float scale = p.scale_of(s);
#pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                if (p.segment_of(cta_row(local_row(mr, 0))) != s) { continue; }
                const unsigned char* b_base = b_s + local_row(mr, b_row) * stride;
                unsigned b[kMaxSteps][2];
#pragma unroll
                for (int ks = 0; ks < kMaxSteps; ks += 2) {
                    if (ks + 1 < steps) {
                        ldmatrix_x4(b[ks][0], b[ks][1], b[ks + 1][0], b[ks + 1][1],
                                    smem_addr(b_base + (2 * ks + b_piece) * 16));
                    } else if (ks < steps) {
                        ldmatrix_x2(b[ks][0], b[ks][1], smem_addr(b_base + (2 * ks + (b_piece & 1)) * 16));
                    }
                }
                float c[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int ks = 0; ks < kMaxSteps; ++ks) {
                    if (ks < steps) { mma_bf16(c[0], c[1], c[2], c[3], a[ks][0], a[ks][1], a[ks][2], a[ks][3], b[ks][0], b[ks][1]); }
                }
#pragma unroll
                for (int v = 0; v < 4; ++v) { acc[mt][mr][v] = __fadd_rn(acc[mt][mr][v], __fmul_rn(c[v], scale)); }
            }
            ++item;
        }
    }
    cp_wait<0>();
    barrier(); // every group's reads finish before the caller reuses shared memory
}

// The term on a warp's staged fragments, for a wide instance (lora_add_tokens_tile_staged's operands in shared
// memory: B's rows at b_s, the pass's segments' U columns at u_s, u_bytes a segment, rows of `stride` bytes, U's
// first staged token at tile token `token_offset`; in_slice(mt) whether fragment mt's tokens are staged). Row
// fragments in pairs: a pair in one segment (gate's, up's or down's) shares each U fragment between its two chains
// (half the U loads, two independent MMAs a load). Each chain is lora_add_tokens_tile's on the same operands in the
// same order: the same bits.
// Swizzled: the staged rows hold exactly the rank's 16-byte pieces, piece i of row r at position i ^ (r & 7) (eight
// consecutive rows' same piece in distinct banks without a padding piece); otherwise rows of `stride` bytes, piece i at
// position i. Paired false: every row fragment alone (half the B operands live at once, for a register-capped tile).
template <bool Swizzled = false, bool Paired = true, class P, int MT, int MR, class LocalToken, class LocalRow,
          class CtaRow, class InSlice>
__device__ __forceinline__ void lora_staged_pairs(const P& p, float (&acc)[MT][MR][4], unsigned pass,
                                                  const unsigned char* b_s, const unsigned char* u_s, int u_bytes,
                                                  int stride, LocalToken local_token, LocalRow local_row,
                                                  CtaRow cta_row, InSlice in_slice, int token_offset) {
    static_assert(MR % 2 == 0);
    const int lane  = threadIdx.x & 31;
    const int steps = p.rank >> 4;
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_piece = lane >> 4;
    const int b_row = lane & 7, b_piece = lane >> 3;
    const auto piece_at = [](int row, int piece) { return (Swizzled ? piece ^ (row & 7) : piece) * 16; };
    // The k-steps as a compile-time count (with_lora_steps): no per-step branch around the warp-wide loads.
    with_lora_steps<P>(steps, [&](auto count) {
        constexpr int kSteps = decltype(count){};
        // B's fragments of row fragment mr, every k-step.
        const auto load_b = [&](int mr, unsigned (&b)[kSteps][2]) {
            const int row               = local_row(mr, b_row);
            const unsigned char* b_base = b_s + row * stride;
#pragma unroll
            for (int ks = 0; ks < kSteps; ks += 2) {
                if (ks + 1 < kSteps) {
                    ldmatrix_x4(b[ks][0], b[ks][1], b[ks + 1][0], b[ks + 1][1],
                                smem_addr(b_base + piece_at(row, 2 * ks + b_piece)));
                } else {
                    ldmatrix_x2(b[ks][0], b[ks][1], smem_addr(b_base + piece_at(row, 2 * ks + (b_piece & 1))));
                }
            }
        };
#pragma unroll
        for (int mr0 = 0; mr0 < MR; mr0 += 2) {
            const int seg0 = p.segment_of(cta_row(local_row(mr0, 0)));
            const int seg1 = p.segment_of(cta_row(local_row(mr0 + 1, 0)));
            if (Paired && seg0 == seg1) {
                if (((pass >> seg0) & 1U) == 0U) { continue; }
                const unsigned char* u_slot = u_s + __popc(pass & ((1U << seg0) - 1U)) * u_bytes;
                const float scale           = p.scale_of(seg0);
                unsigned b0[kSteps][2], b1[kSteps][2];
                load_b(mr0, b0);
                load_b(mr0 + 1, b1);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (!in_slice(mt)) { continue; }
                    const int u_row             = local_token(mt, a_row) - token_offset;
                    const unsigned char* u_base = u_slot + u_row * stride;
                    float c0[4] = {0.0F, 0.0F, 0.0F, 0.0F}, c1[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                    for (int ks = 0; ks < kSteps; ++ks) {
                        unsigned a0, a1, a2, a3;
                        ldmatrix_x4(a0, a1, a2, a3, smem_addr(u_base + piece_at(u_row, 2 * ks + a_piece)));
                        mma_bf16(c0[0], c0[1], c0[2], c0[3], a0, a1, a2, a3, b0[ks][0], b0[ks][1]);
                        mma_bf16(c1[0], c1[1], c1[2], c1[3], a0, a1, a2, a3, b1[ks][0], b1[ks][1]);
                    }
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        acc[mt][mr0][v]     = __fadd_rn(acc[mt][mr0][v], __fmul_rn(c0[v], scale));
                        acc[mt][mr0 + 1][v] = __fadd_rn(acc[mt][mr0 + 1][v], __fmul_rn(c1[v], scale));
                    }
                }
                continue;
            }
#pragma unroll
            for (int mr = mr0; mr < mr0 + 2; ++mr) {
                const int seg = mr == mr0 ? seg0 : seg1;
                if (((pass >> seg) & 1U) == 0U) { continue; }
                const unsigned char* u_slot = u_s + __popc(pass & ((1U << seg) - 1U)) * u_bytes;
                const float scale           = p.scale_of(seg);
                unsigned b[kSteps][2];
                load_b(mr, b);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (!in_slice(mt)) { continue; }
                    const int u_row             = local_token(mt, a_row) - token_offset;
                    const unsigned char* u_base = u_slot + u_row * stride;
                    float c[4]                  = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                    for (int ks = 0; ks < kSteps; ++ks) {
                        unsigned a0, a1, a2, a3;
                        ldmatrix_x4(a0, a1, a2, a3, smem_addr(u_base + piece_at(u_row, 2 * ks + a_piece)));
                        mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b[ks][0], b[ks][1]);
                    }
#pragma unroll
                    for (int v = 0; v < 4; ++v) { acc[mt][mr][v] = __fadd_rn(acc[mt][mr][v], __fmul_rn(c[v], scale)); }
                }
            }
        }
    });
}

// lora_add_tokens_tile_staged (below) for a wide instance whose B rows and one segment's U columns over the whole tile
// do not fit the free shared memory (the NVFP4 A4 tile of 256 tokens): the same staging with U's columns copied for
// half the tile's tokens at a time. A warp's fragments lie in one half (its tokens are a contiguous kWarpTokens-run that
// divides half the tile), so each fragment's chain is lora_add_tokens_tile_staged's on the same operands in the same
// order: the same bits. B's rows stay staged for both halves.
template <int BlockTokens, int BlockRows, int SmemBytes, int MT, int MR, class P, class LocalToken, class LocalRow,
          class CtaRow, class Barrier>
__device__ __forceinline__ void lora_add_tokens_tile_staged_halves(const P& p, float (&acc)[MT][MR][4],
                                                                   LocalToken local_token, LocalRow local_row,
                                                                   CtaRow cta_row, int token_begin, int token_count,
                                                                   unsigned char* smem, int thread, int threads,
                                                                   Barrier barrier) {
    constexpr int kPieces       = P::kMaxRank / 8;
    constexpr int kPieceBits    = kPieces == 16 ? 4 : 3;
    static_assert(kPieces == 1 << kPieceBits);
    constexpr int kMaxStride = 2 * P::kMaxRank + 16;
    constexpr int kHalf      = BlockTokens / 2;
    static_assert(BlockRows % 8 == 0 && kHalf % 16 == 0 && (BlockRows + kHalf) * kMaxStride <= SmemBytes,
                  "the tile's B rows and one segment's U columns over half the tile must fit");
    const int pieces = p.rank >> 3, stride = 2 * p.rank + 16;
    unsigned touched = 0;
#pragma unroll
    for (int f = 0; f < BlockRows / 8; ++f) {
        const int s = p.segment_of(cta_row(8 * f));
        touched |= p.offset_of(s) >= 0 ? 1U << s : 0U;
    }
    if (touched == 0) { return; } // tile-uniform; shared memory untouched
    unsigned char* b_s = smem;
    unsigned char* u_s = smem + BlockRows * stride;
    const int u_bytes  = kHalf * stride;
    const int fit      = (SmemBytes - BlockRows * stride) / u_bytes;
    const int live     = token_count - token_begin;
    for (int i = thread; i < BlockRows * kPieces; i += threads) {
        const int row = i >> kPieceBits, piece = i & (kPieces - 1);
        if (piece < pieces) {
            cp_async<16>(b_s + row * stride + piece * 16,
                         p.b + static_cast<std::int64_t>(cta_row(row)) * p.rank + piece * 8);
        }
    }
    bool first = true;
    for (int half = 0; half < 2; ++half) {
        const int half_begin = half * kHalf;
        unsigned remaining   = touched;
        while (remaining != 0) {
            unsigned pass = 0;
            int slots     = 0;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                if (((remaining >> s) & 1U) != 0U && slots < fit) {
                    pass |= 1U << s;
                    ++slots;
                }
            }
            remaining &= ~pass;
            if (!first) { barrier(); } // the previous pass's reads finish before its U rows are overwritten
            first    = false;
            int slot = 0;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                if (((pass >> s) & 1U) == 0U) { continue; }
                const int uo        = p.offset_of(s);
                unsigned char* dest = u_s + slot * u_bytes;
                for (int i = thread; i < kHalf * kPieces; i += threads) {
                    const int t = i >> kPieceBits, piece = i & (kPieces - 1);
                    if (piece < pieces) {
                        const bool valid = half_begin + t < live;
                        cp_async_zfill<16>(dest + t * stride + piece * 16,
                                           p.u + static_cast<std::int64_t>(token_begin + (valid ? half_begin + t : 0)) *
                                                     p.u_rows +
                                               uo + piece * 8,
                                           valid ? 16 : 0);
                    }
                }
                ++slot;
            }
            cp_commit();
            cp_wait<0>();
            barrier();
            lora_staged_pairs(p, acc, pass, b_s, u_s, u_bytes, stride, local_token, local_row, cta_row,
                              [&](int mt) { return local_token(mt, 0) / kHalf == half; }, half_begin);
        }
    }
    barrier(); // the last pass's reads finish before the caller reuses shared memory
}

// lora_add_tokens_tile_staged (below) for a wide instance whose B rows and one segment's U columns over the whole tile
// fit `smem` only in rows of exactly 2 * kMaxRank bytes (the NVFP4 A4 tile of 256 tokens, whose wide adapter instances
// size their scratch for it): the pieces XOR-swizzled by the row (lora_staged_pairs<true>), one adapted segment's U
// columns over every token a pass. Every warp runs its fragments of the pass's segment each pass (the halves form
// above stages half the tokens a pass, so half the warps wait at each barrier: 6 % of the A4 gate/up tile's samples at
// rank 128). Each chain is lora_add_tokens_tile_staged's on the same operands in the same order: the same bits.
// LowRegisters: the row fragments one at a time (lora_staged_pairs<true, false>), for a tile whose accumulators leave
// few registers (the FP8 tiles).
template <int BlockTokens, int BlockRows, int SmemBytes, bool LowRegisters = false, int MT, int MR, class P,
          class LocalToken, class LocalRow, class CtaRow, class Barrier>
__device__ __forceinline__ void lora_add_tokens_tile_staged_swizzled(const P& p, float (&acc)[MT][MR][4],
                                                                     LocalToken local_token, LocalRow local_row,
                                                                     CtaRow cta_row, int token_begin, int token_count,
                                                                     unsigned char* smem, int thread, int threads,
                                                                     Barrier barrier) {
    constexpr int kPieces    = P::kMaxRank / 8;
    constexpr int kPieceBits = kPieces == 16 ? 4 : 3;
    static_assert(kPieces == 1 << kPieceBits);
    constexpr int kRowBytes = 2 * P::kMaxRank;
    static_assert(BlockRows % 8 == 0 && (BlockRows + BlockTokens) * kRowBytes <= SmemBytes,
                  "the tile's B rows and one segment's U columns over the tile must fit");
    const int pieces = p.rank >> 3;
    unsigned touched = 0;
#pragma unroll
    for (int f = 0; f < BlockRows / 8; ++f) {
        const int s = p.segment_of(cta_row(8 * f));
        touched |= p.offset_of(s) >= 0 ? 1U << s : 0U;
    }
    if (touched == 0) { return; } // tile-uniform; shared memory untouched
    unsigned char* b_s = smem;
    unsigned char* u_s = smem + BlockRows * kRowBytes;
    const int live     = token_count - token_begin;
    for (int i = thread; i < BlockRows * kPieces; i += threads) {
        const int row = i >> kPieceBits, piece = i & (kPieces - 1);
        if (piece < pieces) {
            cp_async<16>(b_s + row * kRowBytes + (piece ^ (row & 7)) * 16,
                         p.b + static_cast<std::int64_t>(cta_row(row)) * p.rank + piece * 8);
        }
    }
    bool first = true;
#pragma unroll 1
    for (int s = 0; s < 4; ++s) {
        if (((touched >> s) & 1U) == 0U) { continue; }
        if (!first) { barrier(); } // the previous pass's reads finish before its U rows are overwritten
        first        = false;
        const int uo = p.offset_of(s);
        for (int i = thread; i < BlockTokens * kPieces; i += threads) {
            const int t = i >> kPieceBits, piece = i & (kPieces - 1);
            if (piece < pieces) {
                const bool valid = t < live;
                cp_async_zfill<16>(u_s + t * kRowBytes + (piece ^ (t & 7)) * 16,
                                   p.u + static_cast<std::int64_t>(token_begin + (valid ? t : 0)) * p.u_rows + uo +
                                       piece * 8,
                                   valid ? 16 : 0);
            }
        }
        cp_commit();
        cp_wait<0>();
        barrier();
        lora_staged_pairs<true, !LowRegisters>(p, acc, 1U << s, b_s, u_s, 0, kRowBytes, local_token, local_row, cta_row,
                                [](int) { return true; }, 0);
    }
    barrier(); // the last pass's reads finish before the caller reuses shared memory
}

// lora_add_tokens_tile for a CTA tile kernel whose consumer threads have finished with `smem` (SmemBytes, free) and
// hold the tile's fragments; every consumer thread calls it. Per warp, the plain form reloads U from global memory
// for each of its MR row fragments (576 scalar loads a thread on the NVFP4 A4 TMA tile); here the CTA copies its
// operands once, in 16-byte pieces: B's rows of the tile's BlockRows rows (cta_row(i): parent row of tile row i) and,
// per adapted segment the rows touch, U's columns of the BlockTokens tokens from token_begin (zero from token_count),
// into rows of rank * 2 + 16 bytes (an odd number of 16-byte pieces: ldmatrix's eight rows hit distinct banks).
// Each warp then takes its fragments by ldmatrix and runs lora_add_tokens_tile's MMA chain on the same operands in
// the same order: the same bits. local_token(mt, i) is fragment row i's tile token (0 .. BlockTokens - 1),
// local_row(mr, j) fragment column j's tile row; a fragment's 8 columns lie in one segment. The touched segments are
// staged together when they fit, else as many a pass as fit. barrier() synchronizes the consumer threads; the call
// ends with one, so the caller may reuse `smem` at once. A wide instance whose padded rows (P::kMaxRank) do not fit
// the free shared memory over the whole tile stages them unpadded and swizzled (lora_add_tokens_tile_staged_swizzled)
// when that fits, else U half the tile at a time (lora_add_tokens_tile_staged_halves), or, when not even that fits,
// reads its operands from L2 (lora_add_tokens_tile): the same chains, the same bits. LowRegisters: the swizzled form
// takes the row fragments one at a time (a register-capped tile).
template <int BlockTokens, int BlockRows, int SmemBytes, bool LowRegisters = false, int MT, int MR, class P,
          class LocalToken, class LocalRow, class CtaRow, class Barrier>
__device__ __forceinline__ void lora_add_tokens_tile_staged(const P& p, float (&acc)[MT][MR][4],
                                                            LocalToken local_token, LocalRow local_row,
                                                            CtaRow cta_row, int token_begin, int token_count,
                                                            unsigned char* smem, int thread, int threads,
                                                            Barrier barrier) {
    [[maybe_unused]] constexpr int kLoraMaxSteps = kLoraSteps<P>;
    constexpr int kPieces       = P::kMaxRank / 8; // 16-byte pieces of a row: 8 or 16
    constexpr int kPieceBits    = kPieces == 16 ? 4 : 3;
    static_assert(kPieces == 1 << kPieceBits);
    constexpr int kMaxStride    = 2 * P::kMaxRank + 16;
    static_assert(BlockRows % 8 == 0);
    if constexpr (P::kMaxRank > kLoraNarrowRank && (BlockRows + BlockTokens) * kMaxStride > SmemBytes &&
                  (BlockRows + BlockTokens) * 2 * P::kMaxRank <= SmemBytes) {
        lora_add_tokens_tile_staged_swizzled<BlockTokens, BlockRows, SmemBytes, LowRegisters>(
            p, acc, local_token, local_row, cta_row, token_begin, token_count, smem, thread, threads, barrier);
        return;
    } else if constexpr ((BlockRows + BlockTokens) * kMaxStride > SmemBytes &&
                         (BlockRows + BlockTokens / 2) * kMaxStride <= SmemBytes && BlockTokens % 32 == 0) {
        static_assert(P::kMaxRank > kLoraNarrowRank, "a narrow instance's B rows and one segment's U columns fit");
        lora_add_tokens_tile_staged_halves<BlockTokens, BlockRows, SmemBytes>(p, acc, local_token, local_row, cta_row,
                                                                              token_begin, token_count, smem, thread,
                                                                              threads, barrier);
        return;
    } else if constexpr ((BlockRows + BlockTokens) * kMaxStride > SmemBytes) {
        static_assert(P::kMaxRank > kLoraNarrowRank, "a narrow instance's B rows and one segment's U columns fit");
        lora_add_tokens_tile<MT, MR>(
            p, acc,
            [&](int mt, int i) {
                const int token = token_begin + local_token(mt, i);
                return token < token_count ? token : -1;
            },
            [&](int mr, int j) { return cta_row(local_row(mr, j)); });
        barrier();
        return;
    }
    [[maybe_unused]] const int lane = threadIdx.x & 31;
    [[maybe_unused]] const int steps = p.rank >> 4, pieces = p.rank >> 3, stride = 2 * p.rank + 16;
    unsigned touched = 0;
#pragma unroll
    for (int f = 0; f < BlockRows / 8; ++f) {
        const int s = p.segment_of(cta_row(8 * f));
        touched |= p.offset_of(s) >= 0 ? 1U << s : 0U;
    }
    if (touched == 0) { return; } // tile-uniform; shared memory untouched
    unsigned char* b_s = smem;
    unsigned char* u_s = smem + BlockRows * stride;
    const int u_bytes  = BlockTokens * stride;
    const int fit      = (SmemBytes - BlockRows * stride) / u_bytes;
    const int live     = token_count - token_begin;
    for (int i = thread; i < BlockRows * kPieces; i += threads) {
        const int row = i >> kPieceBits, piece = i & (kPieces - 1);
        if (piece < pieces) {
            cp_async<16>(b_s + row * stride + piece * 16,
                         p.b + static_cast<std::int64_t>(cta_row(row)) * p.rank + piece * 8);
        }
    }
    // ldmatrix lane roles: U (the A operand) x4 = rows 0-7 / 8-15 at piece 2ks, then at 2ks + 1; B x4 = two k-steps
    // of the fragment's 8 rows (pieces 2ks .. 2ks + 3).
    [[maybe_unused]] const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_piece = lane >> 4;
    [[maybe_unused]] const int b_row = lane & 7, b_piece = lane >> 3;
    bool first = true;
    while (touched != 0) {
        unsigned pass = 0;
        int slots     = 0;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (((touched >> s) & 1U) != 0U && slots < fit) {
                pass |= 1U << s;
                ++slots;
            }
        }
        touched &= ~pass;
        if (!first) { barrier(); } // the previous pass's reads finish before its U rows are overwritten
        first    = false;
        int slot = 0;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            if (((pass >> s) & 1U) == 0U) { continue; }
            const int uo        = p.offset_of(s);
            unsigned char* dest = u_s + slot * u_bytes;
            for (int i = thread; i < BlockTokens * kPieces; i += threads) {
                const int t = i >> kPieceBits, piece = i & (kPieces - 1);
                if (piece < pieces) {
                    const bool valid = t < live;
                    cp_async_zfill<16>(dest + t * stride + piece * 16,
                                       p.u + static_cast<std::int64_t>(token_begin + (valid ? t : 0)) * p.u_rows + uo +
                                           piece * 8,
                                       valid ? 16 : 0);
                }
            }
            ++slot;
        }
        cp_commit();
        cp_wait<0>();
        barrier();
        if constexpr (P::kMaxRank > kLoraNarrowRank && MR % 2 == 0) {
            lora_staged_pairs<false, !LowRegisters>(p, acc, pass, b_s, u_s, u_bytes, stride, local_token, local_row,
                                                     cta_row, [](int) { return true; }, 0);
        } else {
            // The rank's k-steps as a compile-time count (an instance per count): the unrolled loads and MMAs carry no
            // per-step branch (each one, around ldmatrix's warp-wide form, cost a reconvergence and a WARPSYNC), so a
            // fragment's chain and its neighbours' interleave. The chains are the run-time form's.
            with_lora_steps<P>(steps, [&](auto count) {
                constexpr int kSteps = decltype(count){};
#pragma unroll
                for (int mr = 0; mr < MR; ++mr) {
                    const int seg = p.segment_of(cta_row(local_row(mr, 0)));
                    if (((pass >> seg) & 1U) == 0U) { continue; }
                    const unsigned char* u_slot = u_s + __popc(pass & ((1U << seg) - 1U)) * u_bytes;
                    const float scale           = p.scale_of(seg);
                    const unsigned char* b_base = b_s + local_row(mr, b_row) * stride;
                    unsigned b[kSteps][2];
#pragma unroll
                    for (int ks = 0; ks < kSteps; ks += 2) {
                        if (ks + 1 < kSteps) {
                            ldmatrix_x4(b[ks][0], b[ks][1], b[ks + 1][0], b[ks + 1][1],
                                        smem_addr(b_base + (2 * ks + b_piece) * 16));
                        } else {
                            ldmatrix_x2(b[ks][0], b[ks][1], smem_addr(b_base + (2 * ks + (b_piece & 1)) * 16));
                        }
                    }
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        const unsigned char* u_base = u_slot + local_token(mt, a_row) * stride;
                        float c[4]                  = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                        for (int ks = 0; ks < kSteps; ++ks) {
                            unsigned a0, a1, a2, a3;
                            ldmatrix_x4(a0, a1, a2, a3, smem_addr(u_base + (2 * ks + a_piece) * 16));
                            mma_bf16(c[0], c[1], c[2], c[3], a0, a1, a2, a3, b[ks][0], b[ks][1]);
                        }
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            acc[mt][mr][v] = __fadd_rn(acc[mt][mr][v], __fmul_rn(c[v], scale));
                        }
                    }
                }
            });
        }
    }
    barrier(); // the last pass's reads finish before the caller reuses shared memory
}

// The adapter form of a row-orientation tile's single-value epilogue (identity or residual add) for a kernel that
// hands its epilogue a product, `value(mm, e, v)` (the no-adapter kernel's own expression for element e of fragment
// row mm, e.g. v * alpha), which the residual's add may fuse into one FMA: in place, each element of a live column
// (token_of >= 0) becomes the inner epilogue's FP32 result of that expression, the no-adapter kernel's arithmetic,
// and the term is added to it; lora_store_rows_tile then stores the tile. With B = 0 the output is the no-adapter
// kernel's bits. Warp-wide (the term's MMA chains).
template <int MM, int MN, class Epilogue, class RowOf, class TokenOf, class Value>
__device__ __forceinline__ void lora_finish_rows_tile(const Epilogue& epilogue, float (&acc)[MM][MN][4], RowOf row_of,
                                                      TokenOf token_of, Value value) {
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int mm = 0; mm < MM; ++mm) {
        const int row_lo = row_of(mm, g), row_hi = row_of(mm, g + 8);
#pragma unroll
        for (int mn = 0; mn < MN; ++mn) {
            const int t0 = token_of(mn, 2 * q), t1 = token_of(mn, 2 * q + 1);
            if (t0 >= 0) {
                acc[mm][mn][0] = epilogue.apply(row_lo, t0, value(mm, 0, acc[mm][mn][0]));
                acc[mm][mn][2] = epilogue.apply(row_hi, t0, value(mm, 2, acc[mm][mn][2]));
            }
            if (t1 >= 0) {
                acc[mm][mn][1] = epilogue.apply(row_lo, t1, value(mm, 1, acc[mm][mn][1]));
                acc[mm][mn][3] = epilogue.apply(row_hi, t1, value(mm, 3, acc[mm][mn][3]));
            }
        }
    }
    lora_add_rows_tile<MM, MN>(epilogue.lora, acc, row_of, token_of);
}

// Stores a tile lora_finish_rows_tile finished: each element of a live column at its row and token.
template <int MM, int MN, class Destination, class RowOf, class TokenOf>
__device__ __forceinline__ void lora_store_rows_tile(const Destination& destination, const float (&acc)[MM][MN][4],
                                                     RowOf row_of, TokenOf token_of) {
    const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
#pragma unroll
    for (int mm = 0; mm < MM; ++mm) {
        const int row_lo = row_of(mm, g), row_hi = row_of(mm, g + 8);
#pragma unroll
        for (int mn = 0; mn < MN; ++mn) {
            const int t0 = token_of(mn, 2 * q), t1 = token_of(mn, 2 * q + 1);
            if (t0 >= 0) {
                destination.store(row_lo, t0, acc[mm][mn][0]);
                destination.store(row_hi, t0, acc[mm][mn][2]);
            }
            if (t1 >= 0) {
                destination.store(row_lo, t1, acc[mm][mn][1]);
                destination.store(row_hi, t1, acc[mm][mn][3]);
            }
        }
    }
}

// lora_finish_rows_tile and lora_store_rows_tile one 16-row block at a time (the epilogue, the term, the store), so
// a block's registers retire before the next block's term: the same operations on each element.
template <int MM, int MN, class Epilogue, class Destination, class RowOf, class TokenOf, class Value>
__device__ __forceinline__ void lora_finish_store_rows_tile(const Epilogue& epilogue, const Destination& destination,
                                                            float (&acc)[MM][MN][4], RowOf row_of, TokenOf token_of,
                                                            Value value) {
#pragma unroll
    for (int mm = 0; mm < MM; ++mm) {
        auto& block         = reinterpret_cast<float(&)[1][MN][4]>(acc[mm]);
        const auto rows     = [&](int, int i) { return row_of(mm, i); };
        const auto value_of = [&](int, int e, float v) { return value(mm, e, v); };
        lora_finish_rows_tile<1, MN>(epilogue, block, rows, token_of, value_of);
        lora_store_rows_tile<1, MN>(destination, block, rows, token_of);
    }
}

// L2 prefetch of a warp's B rows rows[0..Rows) (lane i takes row i) and of U's columns [token0, token0 + tokens):
// issued before a streaming main loop, so the term's loads after it hit L2.
template <int Rows>
__device__ __forceinline__ void lora_prefetch_warp(const LoraDeviceAddend& p, const int (&rows)[Rows], int token0,
                                                   int tokens) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        if (lane == r % 32 && rows[r] >= 0) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p.b + static_cast<std::int64_t>(rows[r]) * p.rank));
        }
    }
    const std::int64_t bytes = static_cast<std::int64_t>(tokens) * p.u_rows * 2;
    const char* u            = reinterpret_cast<const char*>(p.u + static_cast<std::int64_t>(token0) * p.u_rows);
    for (std::int64_t at = static_cast<std::int64_t>(lane) * 128; at < bytes; at += 32 * 128) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(u + at));
    }
}

// L2 prefetch, by `threads` threads from `thread`, of the B rows row_of(i) for i < count and of U's columns
// [token0, token0 + tokens): a tile kernel's term, issued before its main loop.
template <class RowOf>
__device__ __forceinline__ void lora_prefetch_tile(const LoraDeviceAddend& p, int count, RowOf row_of, int token0,
                                                   int tokens, int thread, int threads) {
    for (int i = thread; i < count; i += threads) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p.b + static_cast<std::int64_t>(row_of(i)) * p.rank));
    }
    if (tokens <= 0) { return; }
    const std::int64_t bytes = static_cast<std::int64_t>(tokens) * p.u_rows * 2;
    const char* u            = reinterpret_cast<const char*>(p.u + static_cast<std::int64_t>(token0) * p.u_rows);
    for (std::int64_t at = static_cast<std::int64_t>(thread) * 128; at < bytes;
         at += static_cast<std::int64_t>(threads) * 128) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(u + at));
    }
}

// The adapter form of a projection kernel's epilogue: the inner epilogue unchanged (inherited, with every nested
// member), plus the term. Kernels test kLoraEpilogue<Epilogue> and add the term in FP32 before the BF16 rounding:
// to gate and up before a SwiGLU; for a single value, to the value the kernel hands its epilogue where that is an
// exactly rounded value (an accumulator, an explicitly rounded product), or to the epilogue's result where the
// kernel hands it a product the residual's add may fuse (lora_finish_rows_tile, lora_epilogue_apply), so that the
// adapter instance's arithmetic is the kernel's own. Every other instantiation is the kernel as it was. A narrow
// adapter's (LoraEpilogue) and a wide one's (LoraEpilogueWide) are separate instances.
template <class Inner>
struct LoraEpilogue : Inner {
    LoraDeviceAddend lora;
};
template <class Inner>
struct LoraEpilogueWide : Inner {
    LoraDeviceAddendWide lora;
};

template <class E>
inline constexpr bool kLoraEpilogue = false;
template <class Inner>
inline constexpr bool kLoraEpilogue<LoraEpilogue<Inner>> = true;
template <class Inner>
inline constexpr bool kLoraEpilogue<LoraEpilogueWide<Inner>> = true;

// The rank an instance's staged term operands are sized for (LoraWarpStage): its addend class's, for an epilogue or a
// term type (kLoraNarrowRank for every other instance, whose stage is never used).
template <class T>
inline constexpr int kLoraStageRank = kLoraNarrowRank;
template <class Inner>
inline constexpr int kLoraStageRank<LoraEpilogueWide<Inner>> = kLoraMaxRank;
template <>
inline constexpr int kLoraStageRank<LoraDeviceAddendWide> = kLoraMaxRank;

// The blocks-per-SM bound of a register-capped kernel's adapter instance: the largest bound at most `blocks` whose
// register cap (65,536 registers an SM, allocated by 8) is at least `extra` above `blocks`' own (the term's operands
// and addresses: 8 for a narrow instance, 16 for a wide one, whose k-step operands are twice as many), so the instance
// keeps the no-adapter kernel's registers instead of spilling them.
constexpr int lora_min_blocks(int threads, int blocks, int extra = 8) {
    const auto cap = [threads](int n) { return 65536 / (n * threads) / 8 * 8; };
    int n = blocks;
    while (n > 1 && cap(n) < cap(blocks) + extra) { --n; }
    return n;
}
template <class Epilogue>
constexpr int lora_launch_min_blocks(int threads, int blocks) {
    return !kLoraEpilogue<Epilogue>                       ? blocks
           : kLoraStageRank<Epilogue> > kLoraNarrowRank ? lora_min_blocks(threads, blocks, 16)
                                                          : lora_min_blocks(threads, blocks);
}

// Calls launch(epilogue) with `inner`, or with its adapter form of the rank's class when `lora` is given.
template <class Inner, class Launch>
void with_lora_epilogue(const LoraAddend* lora, const Inner& inner, Launch&& launch) {
    if (lora == nullptr || !lora->present()) {
        launch(inner);
    } else if (lora_narrow(*lora)) {
        launch(LoraEpilogue<Inner>{inner, lora_device_addend<LoraDeviceAddend>(*lora)});
    } else {
        launch(LoraEpilogueWide<Inner>{inner, lora_device_addend<LoraDeviceAddendWide>(*lora)});
    }
}

} // namespace infernix::ops::detail
