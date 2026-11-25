#pragma once

// What Inkling 2's two attention routes share (include/infernix/ops/inkling2_attention.h): constants, the KV row of a
// position, and the prologue's launch (attention.cu), which both routes run before their attention kernel.

#include "infernix/ops/inkling2_attention.h"
#include "ops/common/canonical_math.h"
#include "ops/inkling2/kv_codec.cuh"

#include <cuda_fp16.h>
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
    // A page-major pool: one page's rows are contiguous in each plane (the prefix cache copies a page as one run).
    const std::int64_t page = kv.table[static_cast<std::int64_t>(lane) * kv.table_stride + pos / kKeyUnit];
    return ((static_cast<std::size_t>(page) * kKvHeads + head) * kKeyUnit + (pos % kKeyUnit)) * kD;
}

// ---------------------------------------------------------------------------- quantized pools (kv_codec.cuh)

// Whether a call's visit runs the kernels' quantized instances (its pool, or a sliding visit's window ring, is not BF16
// words): the BF16 instances carry no codec code.
inline bool quantized_pool(const AttentionArgs& a) { return a.kv.codec != KvCodec::Bf16; }

// Whether a visit's pool holds rotated codes (vq2, k4v2): its scores and P x V run in the rotated domain (the rows'
// q' = kv_rotate_query(q), the cached S c), and its output is o = 2^-4 H128 o'.
__host__ __device__ __forceinline__ bool rotated_pool(const AttentionArgs& a) {
    return a.layer.full && rotated(a.kv.codec);
}

// The form of a visit's plane (which 0 K, 1 V): its pool's or its window ring's codec (KvView::codec).
__device__ __forceinline__ KvForm plane_form(const KvView& kv, int which) {
    return which == 0 ? key_form(kv.codec) : value_form(kv.codec);
}
// A quantized row's codes and side record in plane `which`, by its element offset in the BF16 layout. A quantized
// window ring's codes and side records are dense per plane in their own buffers (KvView), so the same offsets hold.
__device__ __forceinline__ std::uint8_t* row_codes(const KvView& kv, int which, KvForm form, std::size_t element) {
    return reinterpret_cast<std::uint8_t*>(which == 0 ? kv.k : kv.v) + element / kD * code_bytes(form);
}
__device__ __forceinline__ std::uint8_t* row_side(const KvView& kv, int which, KvForm form, std::size_t element) {
    return (which == 0 ? kv.k_side : kv.v_side) + element / kD * side_bytes(form);
}

// A unit's 64 rows of both planes (`page`: its first row's element offset) decoded into the tiles' represented words
// (row stride `stride` words; `at(row, d)` the word offset of value d of a row), by `threads` threads from `tid`.
// Rotated false: as kv_decode16's.
template <bool Rotated = true, class At>
__device__ __forceinline__ void decode_unit(const KvView& kv, std::size_t page, std::uint16_t* k_tile,
                                            std::uint16_t* v_tile, At at, int tid, int threads) {
    const KvForm forms[2] = {plane_form(kv, 0), plane_form(kv, 1)};
    for (int i = tid; i < 2 * kKeyUnit * (kD / 16); i += threads) {
        const int plane = i / (kKeyUnit * (kD / 16)), rest = i % (kKeyUnit * (kD / 16));
        const int row = rest / (kD / 16), chunk = rest % (kD / 16);
        const std::size_t element = page + static_cast<std::size_t>(row) * kD;
        uint4 lo, hi;
        kv_decode16<Rotated>(forms[plane], row_codes(kv, plane, forms[plane], element), row_side(kv, plane, forms[plane], element),
                    chunk, plane == 1, lo, hi);
        std::uint16_t* tile = plane == 0 ? k_tile : v_tile;
        *reinterpret_cast<uint4*>(tile + at(row, 16 * chunk))     = lo;
        *reinterpret_cast<uint4*>(tile + at(row, 16 * chunk + 8)) = hi;
    }
}

// decode_unit for an int8 G64 pool or window ring (both planes int8): a thread decodes 8-value pieces (kv_decode8_int8),
// consecutive threads consecutive 16-byte pieces of a row (unit stores without bank conflicts), each plane in its own
// loop (the word form fixed). The same words as decode_unit.
template <class At>
__device__ __forceinline__ void decode_unit_int8(const KvView& kv, std::size_t page, std::uint16_t* k_tile,
                                                 std::uint16_t* v_tile, At at, int tid, int threads) {
    constexpr int kPieces = kD / 8; // a row's 8-value pieces
    const auto plane = [&]<bool Value>(const std::uint8_t* codes, const std::uint8_t* side, std::uint16_t* tile) {
        for (int i = tid; i < kKeyUnit * kPieces; i += threads) {
            const int row = i / kPieces, piece = i % kPieces;
            const uint2 c = *reinterpret_cast<const uint2*>(codes + row * kD + 8 * piece);
            const std::uint16_t s = reinterpret_cast<const std::uint16_t*>(side + row * 4)[piece / 8];
            *reinterpret_cast<uint4*>(tile + at(row, 8 * piece)) = kv_decode8_int8<Value>(c, s);
        }
    };
    plane.template operator()<false>(row_codes(kv, 0, KvForm::Int8, page), row_side(kv, 0, KvForm::Int8, page), k_tile);
    plane.template operator()<true>(row_codes(kv, 1, KvForm::Int8, page), row_side(kv, 1, KvForm::Int8, page), v_tile);
}

// The argument checks both routes share, and the prologue over every column of the call (attention.cu).
void check_attention_args(const AttentionArgs& a);
void launch_attention_prologue(const AttentionArgs& a, cudaStream_t stream);

} // namespace infernix::ops::inkling2::detail
