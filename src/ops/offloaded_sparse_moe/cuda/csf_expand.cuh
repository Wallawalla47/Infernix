#pragma once

// GPU expansion of CSF host records (ops/offloaded_sparse_moe/csf_record.h). A CTA expands one chunk (Geometry's
// chunks: one gate/up row group, or down_per_chunk() down row groups) from the chunk's compressed regions in shared
// memory into the expanded record's units, every byte as csf::expand_record writes it. The stage kernel fills the
// shared buffer from a host record (misses, landings); expand_in_place_kernel from the record's own device copy
// (copy-engine landings: the prefill stream's ring, frame promotions).

#include "ops/offloaded_sparse_moe/csf_expand.h"
#include "ops/offloaded_sparse_moe/csf_record.h"

#include <cstdint>

namespace infernix::ops::offloaded_moe::csf {

// A chunk's compressed regions (from the chunk's first region's start).
struct ChunkBuffer {
    alignas(16) std::uint8_t data[kMaxChunkBytes];
};

// The CTA's scratch for one chunk's expansion.
struct ChunkScratch {
    alignas(16) std::uint32_t scales[kMaxChunkUnits][4]; // each compressed unit's 16 scale bytes
    std::int32_t escape_at[kMaxChunkUnits];              // exclusive prefix of the units' escape counts
    std::int32_t warp_total[kMaxChunkUnits / 32];
};

// The escaped nibbles (value kEscape) of a word of eight offsets.
__device__ __forceinline__ int escaped_nibbles(std::uint32_t word) {
    return __popc(word & (word >> 1) & (word >> 2) & (word >> 3) & 0x11111111U);
}

// 16-byte asynchronous copies of [src, src + bytes) into shared `dst` by the CTA (Threads threads), one commit group.
// Both 16-byte aligned, bytes a multiple of 16.
template <int Threads>
__device__ __forceinline__ void load_chunk_async(const std::uint8_t* src, std::uint32_t bytes, std::uint8_t* dst) {
    for (std::uint32_t i = threadIdx.x * 16U; i < bytes; i += Threads * 16U) {
        const auto s = static_cast<unsigned>(__cvta_generic_to_shared(dst + i));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src + i) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// [src, src + bytes) into shared `dst` by the CTA (Threads threads) through registers: every thread's 16-byte loads
// are issued before any store, so the whole chunk is in flight at once. For sources in mapped host memory (the stage
// kernel's misses): cp.async from host memory ran the stage ~14x slower than these loads (2026-11-09). Streaming
// loads; both 16-byte aligned, bytes a multiple of 16, at most kMaxChunkBytes.
template <int Threads>
__device__ __forceinline__ void load_chunk_registers(const std::uint8_t* src, std::uint32_t bytes, std::uint8_t* dst) {
    constexpr int kPerThread = (kMaxChunkBytes / 16 + Threads - 1) / Threads;
    const auto* s       = reinterpret_cast<const uint4*>(src);
    auto* d             = reinterpret_cast<uint4*>(dst);
    const int vectors   = static_cast<int>(bytes / 16);
    uint4 v[kPerThread];
#pragma unroll
    for (int u = 0; u < kPerThread; ++u) {
        const int i = static_cast<int>(threadIdx.x) + u * Threads;
        if (i < vectors) { v[u] = __ldcs(s + i); }
    }
#pragma unroll
    for (int u = 0; u < kPerThread; ++u) {
        const int i = static_cast<int>(threadIdx.x) + u * Threads;
        if (i < vectors) { d[i] = v[u]; }
    }
}

// CTA-wide (Threads threads): expands chunk c of a record of geometry g into `record` (the expanded record's base).
// `words` (shared or registers) holds the record's table entries table[r0 .. r0 + chunk_rgs(c)] of the chunk's
// row groups (r0 = chunk_first_rg(c)); `src` holds the chunk's compressed bytes [words[0], words[chunk_rgs(c)]) of
// the record. Begins after the caller's barrier that publishes `src` and `words`; ends with a barrier only when
// `final_barrier` (src and the scratch are free again after it).
template <int Threads>
__device__ void expand_chunk(const Geometry& g, int c, const std::uint32_t* words, const std::uint8_t* src,
                             std::uint8_t* record, ChunkScratch& s, bool final_barrier = true) {
    static_assert(Threads >= kMaxChunkUnits && Threads % 32 == 0, "a thread scans one unit's escapes");
    const int t      = static_cast<int>(threadIdx.x);
    const int r0     = g.chunk_first_rg(c);
    const int blocks = g.blocks(r0); // every row group of a chunk has as many units
    const int n      = g.chunk_rgs(c) * blocks;
    const std::uint32_t w0 = words[0];
    // This thread's unit: its offsets and escape count (raw groups have none).
    uint2 offsets{};
    int count = 0;
    bool compressed = false;
    int j = 0, b = 0;
    if (t < n) {
        j = t / blocks;
        b = t - j * blocks;
        const std::uint32_t begin = words[j] - w0, end = words[j + 1] - w0;
        compressed = end - begin != static_cast<std::uint32_t>(blocks) * kUnitBytes;
        if (compressed) {
            const std::uint8_t* body = src + end - static_cast<std::uint32_t>(blocks - b) * kBodyBytes;
            offsets = *reinterpret_cast<const uint2*>(body + kCodeBytes);
            count   = escaped_nibbles(offsets.x) + escaped_nibbles(offsets.y);
        }
    }
    // Exclusive scan of the counts over the chunk's units (a group's escapes start at its first unit's prefix).
    const int lane = t % 32, warp = t / 32;
    int inclusive = count;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const int v = __shfl_up_sync(0xFFFFFFFFU, inclusive, d);
        if (lane >= d) { inclusive += v; }
    }
    if (lane == 31 && warp < kMaxChunkUnits / 32) { s.warp_total[warp] = inclusive; }
    __syncthreads();
    if (t < n) {
        int before = inclusive - count;
        for (int w = 0; w < warp; ++w) { before += s.warp_total[w]; }
        s.escape_at[t] = before;
    }
    __syncthreads();
    if (compressed) {
        const std::uint32_t begin = words[j] - w0;
        const std::uint8_t* bases = src + begin;
        const std::uint8_t* e     = src + begin + kBaseBytes + (s.escape_at[t] - s.escape_at[j * blocks]);
        std::uint32_t out[4]      = {0, 0, 0, 0};
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            const std::uint32_t word = r < 8 ? offsets.x : offsets.y;
            const int o              = static_cast<int>((word >> (4 * (r % 8))) & 15U);
            const std::uint32_t v    = o == kEscape ? *e++ : (static_cast<std::uint32_t>(bases[r]) + o) & 0xFFU;
            out[r / 4] |= v << (8 * (r % 4));
        }
        *reinterpret_cast<uint4*>(s.scales[t]) = make_uint4(out[0], out[1], out[2], out[3]);
    }
    __syncthreads();
    // The chunk's units are contiguous in the expanded record: unit u's 16-byte vector v at 9 u + v.
    auto* out = reinterpret_cast<uint4*>(record + g.expanded_offset(r0));
    for (int i = t; i < 9 * n; i += Threads) {
        const int u  = i / 9, v = i - 9 * u;
        const int ju = u / blocks, bu = u - ju * blocks;
        const std::uint32_t begin = words[ju] - w0, end = words[ju + 1] - w0;
        uint4 value;
        if (end - begin == static_cast<std::uint32_t>(blocks) * kUnitBytes) {
            value = *reinterpret_cast<const uint4*>(src + begin + static_cast<std::uint32_t>(bu) * kUnitBytes + 16 * v);
        } else if (v < 8) {
            const auto* p = reinterpret_cast<const uint2*>(src + end - static_cast<std::uint32_t>(blocks - bu) * kBodyBytes + 16 * v);
            const uint2 lo = p[0], hi = p[1];
            value = make_uint4(lo.x, lo.y, hi.x, hi.y);
        } else {
            value = *reinterpret_cast<const uint4*>(s.scales[u]);
        }
        out[i] = value;
    }
    if (final_barrier) { __syncthreads(); }
}

} // namespace infernix::ops::offloaded_moe::csf
