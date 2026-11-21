#pragma once

// What Inkling 2's two attention routes share (include/infernix/ops/inkling2_attention.h): constants, the KV row of a
// position, and the prologue's launch (attention.cu), which both routes run before their attention kernel.

#include "infernix/ops/inkling2_attention.h"
#include "ops/common/canonical_math.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::inkling2::detail {

inline constexpr int kD          = kHeadDim;
inline constexpr int kKvWidth    = kKvHeads * kD;    // 1024
inline constexpr int kQWidth     = kHeads * kD;      // 3072
inline constexpr int kRWidth     = kHeads * kRelDim; // 384
inline constexpr float kInvSqrtD = static_cast<float>(0.08838834764831845);
inline constexpr float kLog2e    = static_cast<float>(1.4426950408889634);

__device__ __forceinline__ float neg_inf() { return canon::f32_from_bits(0xFF800000U); }

__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// The element offset of a position's row of 128 in a visit's K or V.
__device__ __forceinline__ std::size_t kv_row(const KvView& kv, int lane, int head, int pos) {
    if (kv.table == nullptr) {
        // The state's ring at its constant page count (decode), a staging ring at its own.
        const int pages = kv.ring_pages == kWindowRingPages ? kWindowRingPages : kv.ring_pages;
        const int page  = kv.ring_pages == kWindowRingPages ? (pos / kKeyUnit) % kWindowRingPages
                                                            : (pos / kKeyUnit) % kv.ring_pages;
        return lane * kv.lane_stride + ((static_cast<std::size_t>(head) * pages + page) * kKeyUnit + (pos % kKeyUnit)) * kD;
    }
    const std::int64_t page = kv.table[static_cast<std::int64_t>(lane) * kv.table_stride + pos / kKeyUnit];
    return ((static_cast<std::size_t>(head) * kv.pool_pages + page) * kKeyUnit + (pos % kKeyUnit)) * kD;
}

// The argument checks both routes share, and the prologue over every column of the call (attention.cu).
void check_attention_args(const AttentionArgs& a);
void launch_attention_prologue(const AttentionArgs& a, cudaStream_t stream);

} // namespace infernix::ops::inkling2::detail
