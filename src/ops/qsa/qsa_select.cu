// QSA block selection: multi-CTA scoring and exact radix selection (qsa_select.h,
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.6 item 2).

#include "ops/qsa/qsa_select.h"

#include "ops/qsa/page_spaces.cuh"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/pdl.cuh"
#include "ops/common/score_id_order.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using bf16   = __nv_bfloat16;

constexpr int kIndexDim       = 128;
constexpr unsigned kFullMask  = 0xFFFFFFFFU;
constexpr int kLanePartials   = kIndexDim / 32; // products each lane accumulates per dot

// Scoring: a warp step reads kScoreStepBlocks pooled keys and, per column, reduce-scatters the
// kScoreStepBlocks x kScoreHeadSlots = 32 lane partials of its dots.
constexpr int kScoreThreads     = 256;
constexpr int kScoreWarps       = kScoreThreads / 32;
constexpr int kScoreCtasPerSm   = 2;
constexpr int kScoreStepBlocks  = 8;
constexpr int kScoreHeadSlots   = 4;  // index heads (1..4; unused slots hold zero queries)
constexpr int kScoreTileColumns = 16; // columns of one row scored from one key read
static_assert(kScoreStepBlocks * kScoreHeadSlots == 32);

// Selection.
constexpr int kSelectThreads     = 512;
constexpr int kSelectWarps       = kSelectThreads / 32;
constexpr int kScoreBins         = 2048; // ordered score bits [30:20]: exponent and 3 mantissa bits
constexpr int kBinShift          = 20;
constexpr int kCandidateCapacity = 4096;
constexpr std::size_t kCandidateBytes = sizeof(std::uint64_t) * kCandidateCapacity;
constexpr int kMaxSlices         = 8;    // selection CTAs per column
constexpr int kSliceBlocks       = 4096; // selection: blocks per CTA at the largest context
constexpr int kWalkBatch         = 4;    // 128-score chunks each warp keeps in flight
static_assert(kScoreBins == 4 * kSelectThreads, "find_score_bin gives each thread four bins");

struct ScoreTiles {
    std::int32_t count = 0;
    std::uint8_t first[kQsaSelectGroupColumns];   // first column, counted from the group's first
    std::uint8_t columns[kQsaSelectGroupColumns]; // 1..kScoreTileColumns, all of one row
};
static_assert(kQsaSelectGroupColumns <= 256, "ScoreTiles stores group-relative columns in bytes");

// ------------------------------------------------------------------------------------- scoring

// Same addressing as qsa.cu's pool and attention kernels: the pooled key of block b fills the R
// token slots of its block, slot_width = Di / R elements each.
// A page id's space is resolved (design §19.3.11): the pool's pooled plane or the spaces' array.
__device__ __forceinline__ const bf16* pooled_element(const bf16* pool, const QsaPageSpaces& spaces,
                                                      const std::int32_t* table, int token, int slot_width, int slot) {
    int page;
    const bf16* plane = qsa_pooled_plane(spaces, pool, table[token >> kPagedKVPageShift], page);
    return plane + static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
           static_cast<std::int64_t>(slot_width) * (token & kPagedKVPageMask) + slot;
}

// One stage of the reduce-scatter: lanes whose `Half` bit is clear keep values [0, Half), the
// others [Half, 2 Half); each kept value adds its partner lane's copy (own + partner, as
// warp_sum's `v += __shfl_xor_sync(v, Half)`).
template <int Half>
__device__ __forceinline__ void reduce_scatter_stage(float (&v)[32], int lane) {
    const bool upper = (lane & Half) != 0;
#pragma unroll
    for (int i = 0; i < Half; ++i) {
        const float keep = upper ? v[i + Half] : v[i];
        const float send = upper ? v[i] : v[i + Half];
        v[i]             = keep + __shfl_xor_sync(kFullMask, send, Half);
    }
}

// Lane l returns the 32-lane sum of its value l, bitwise equal to warp_sum's result for that value
// (31 shuffles for 32 sums instead of 160). After the stages at offsets 16..o, each value lane l
// keeps is the sum over the lanes l ^ m, m a combination of those offsets, formed by exactly the
// additions the xor butterfly performs at lane l; the butterfly ends with that sum in every lane.
__device__ __forceinline__ float reduce_scatter_32(float (&v)[32], int lane) {
    reduce_scatter_stage<16>(v, lane);
    reduce_scatter_stage<8>(v, lane);
    reduce_scatter_stage<4>(v, lane);
    reduce_scatter_stage<2>(v, lane);
    reduce_scatter_stage<1>(v, lane);
    return v[0];
}

// Grid (CTAs per tile, tiles). A tile is up to 16 consecutive columns of one row; its CTAs split
// the row's blocks in steps of 8, one step per warp at a time. Each pooled key is read once per
// tile and scored for every column of the tile that sees it. With `clear`, the CTAs first zero
// `clear_words` words of selection state (histograms and arrival counters).
__global__ void __launch_bounds__(kScoreThreads, kScoreCtasPerSm)
    qsa_score_kernel(const bf16* __restrict__ index_q, int index_heads, const bf16* __restrict__ pooled,
                     const std::int32_t* __restrict__ tables, int table_stride,
                     const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                     int width, int ratio, int top_blocks, int group_begin, const ScoreTiles tiles,
                     float* __restrict__ scores, int score_stride, std::uint32_t* __restrict__ clear,
                     int clear_words, QsaPageSpaces spaces) {
    pdl::enter_streaming();
    if (clear != nullptr) {
        const std::int64_t ctas = static_cast<std::int64_t>(gridDim.x) * gridDim.y;
        const std::int64_t cta  = static_cast<std::int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
        const int begin         = static_cast<int>(clear_words * cta / ctas);
        const int end           = static_cast<int>(clear_words * (cta + 1) / ctas);
        for (int i = begin + static_cast<int>(threadIdx.x); i < end; i += kScoreThreads) { clear[i] = 0; }
    }
    __shared__ float4 qs[kScoreTileColumns][kScoreHeadSlots][32]; // lane l: q[h][l + 32 j], j = 0..3
    __shared__ int column_blocks[kScoreTileColumns];
    const int tile = static_cast<int>(blockIdx.y);
    const int c0 = tiles.first[tile], nc = tiles.columns[tile];
    const int t0 = group_begin + c0;
    int tile_blocks = 0;
    for (int c = 0; c < nc; ++c) { tile_blocks = max(tile_blocks, (positions[t0 + c] + 1) / ratio); }
    const int steps      = (tile_blocks + kScoreStepBlocks - 1) / kScoreStepBlocks;
    const int first_step = static_cast<int>(blockIdx.x) * kScoreWarps;
    if (tile_blocks <= top_blocks || first_step >= steps) {
        pdl::trigger_dependents();
        return; // every column of the tile is dense, or this CTA has no step
    }
    if (static_cast<int>(threadIdx.x) < nc) {
        column_blocks[threadIdx.x] = (positions[t0 + static_cast<int>(threadIdx.x)] + 1) / ratio;
    }
    for (int i = static_cast<int>(threadIdx.x); i < nc * kScoreHeadSlots * kIndexDim; i += kScoreThreads) {
        const int c = i / (kScoreHeadSlots * kIndexDim), h = (i / kIndexDim) % kScoreHeadSlots, d = i % kIndexDim;
        const float value =
            h < index_heads
                ? __bfloat162float(index_q[(static_cast<std::size_t>(t0 + c) * index_heads + h) * kIndexDim + d])
                : 0.0F;
        reinterpret_cast<float*>(&qs[c][h][d % 32])[d / 32] = value;
    }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t0 / width]) * table_stride;
    const int slot_width      = kIndexDim / ratio;
    int token_offset[kLanePartials], slot[kLanePartials];
#pragma unroll
    for (int j = 0; j < kLanePartials; ++j) {
        token_offset[j] = (lane + 32 * j) / slot_width;
        slot[j]         = (lane + 32 * j) % slot_width;
    }
    const bool block_in_page = ratio <= kPagedKVPageSize; // a block's tokens share one page
    const float scale        = rsqrtf(static_cast<float>(kIndexDim));
    for (int step = first_step + warp; step < steps; step += static_cast<int>(gridDim.x) * kScoreWarps) {
        const int b0 = step * kScoreStepBlocks;
        float key[kScoreStepBlocks][kLanePartials];
#pragma unroll
        for (int bb = 0; bb < kScoreStepBlocks; ++bb) {
            const int b = b0 + bb;
            if (b >= tile_blocks) {
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) { key[bb][j] = 0.0F; }
            } else if (block_in_page) {
                int local;
                const bf16* plane = qsa_pooled_plane(spaces, pooled, table[(ratio * b) >> kPagedKVPageShift], local);
                const std::int64_t page = static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * local;
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) {
                    const int token = ratio * b + token_offset[j];
                    key[bb][j] = __bfloat162float(plane[page + static_cast<std::int64_t>(slot_width) *
                                                                   (token & kPagedKVPageMask) + slot[j]]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < kLanePartials; ++j) {
                    key[bb][j] = __bfloat162float(
                        *pooled_element(pooled, spaces, table, ratio * b + token_offset[j], slot_width, slot[j]));
                }
            }
        }
        for (int c = 0; c < nc; ++c) {
            const int blocks = column_blocks[c];
            if (b0 >= blocks) { continue; } // warp-uniform: this column sees none of the step
            float4 q[kScoreHeadSlots];
#pragma unroll
            for (int h = 0; h < kScoreHeadSlots; ++h) { q[h] = qs[c][h][lane]; }
            // Value bb * 4 + h: this lane's partial of block b0 + bb, head h, accumulated in j order
            // from 0 like the one-warp-per-block kernel's `dot += q[h][lane + 32 j] * key[j]`.
            float v[32];
#pragma unroll
            for (int bb = 0; bb < kScoreStepBlocks; ++bb) {
#pragma unroll
                for (int h = 0; h < kScoreHeadSlots; ++h) {
                    float dot = 0.0F;
                    dot += q[h].x * key[bb][0];
                    dot += q[h].y * key[bb][1];
                    dot += q[h].z * key[bb][2];
                    dot += q[h].w * key[bb][3];
                    v[bb * kScoreHeadSlots + h] = dot;
                }
            }
            // Lane 4 bb + h now holds head h's dot of block b0 + bb; heads join in head order.
            const float relu  = fmaxf(reduce_scatter_32(v, lane), 0.0F);
            const float relu1 = __shfl_down_sync(kFullMask, relu, 1);
            const float relu2 = __shfl_down_sync(kFullMask, relu, 2);
            const float relu3 = __shfl_down_sync(kFullMask, relu, 3);
            const int b       = b0 + lane / kScoreHeadSlots;
            if (lane % kScoreHeadSlots == 0 && b < blocks) {
                float score = 0.0F;
                score += relu;
                if (index_heads > 1) { score += relu1; }
                if (index_heads > 2) { score += relu2; }
                if (index_heads > 3) { score += relu3; }
                scores[static_cast<std::size_t>(c0 + c) * score_stride + b] = score * scale;
            }
        }
    }
    pdl::trigger_dependents();
}

// ----------------------------------------------------------------------------------- selection

struct SelectShared {
    using Scan = cub::BlockScan<int, kSelectThreads>;
    typename Scan::TempStorage scan;
    unsigned radix[256];
    int pick_digit, pick_above, pick_count; // radix_threshold's digit
    int bin, bin_above, bin_count;          // the score bin holding the top-th score
    unsigned candidates;                    // keys appended to this CTA's candidate buffer
    unsigned long long and_key, or_key;     // over this CTA's share of the bin's keys
    int last;                               // this CTA arrived last at its column
};

// Scores are non-negative (a sum of ReLUs times a positive scale), so the ordered bits' top bit is
// set and bits [30:20] order the bins like the scores.
__device__ __forceinline__ int score_bin(float score) {
    return static_cast<int>((ordered_score_bits(score) >> kBinShift) & (kScoreBins - 1));
}

// First block of slice s of S over n blocks; multiples of 4 so float4 loads stay aligned.
__host__ __device__ __forceinline__ int slice_begin(int n, int s, int slices) {
    if (s >= slices) { return n; }
    return static_cast<int>((static_cast<std::int64_t>(n) * s / slices) & ~std::int64_t{3});
}

// Walks scores [lo, hi) of a column (lo a multiple of 4) in chunks of 128, each warp one chunk at
// a time with kWalkBatch chunks' loads in flight. Lane l sees scores base..base+3, base = lo +
// 128 chunk + 4 l; `valid` masks those < hi. Calls are warp-uniform.
template <class Visit>
__device__ __forceinline__ void walk_scores(const float* column, int lo, int hi, Visit&& visit) {
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int chunks = (hi - lo + 127) / 128;
    for (int first = warp; first < chunks; first += kSelectWarps * kWalkBatch) {
        float4 v[kWalkBatch];
#pragma unroll
        for (int u = 0; u < kWalkBatch; ++u) {
            const int base = lo + (first + u * kSelectWarps) * 128 + 4 * lane;
            v[u] = base < hi ? *reinterpret_cast<const float4*>(column + base) : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
#pragma unroll
        for (int u = 0; u < kWalkBatch; ++u) {
            const int chunk = first + u * kSelectWarps;
            if (chunk >= chunks) { break; }
            const int base     = lo + chunk * 128 + 4 * lane;
            const int left     = hi - base;
            const unsigned ok  = left >= 4 ? 0xFU : left > 0 ? (1U << left) - 1 : 0U;
            visit(chunk, base, v[u], ok);
        }
    }
}

// Lanes 8m..8m+7 hold four flags each of scores 32m..32m+31 of a chunk; lane 8m returns the word.
__device__ __forceinline__ unsigned chunk_word(unsigned nibble, int lane) {
    unsigned word = nibble << (4 * (lane & 7));
    word |= __shfl_xor_sync(kFullMask, word, 1);
    word |= __shfl_xor_sync(kFullMask, word, 2);
    word |= __shfl_xor_sync(kFullMask, word, 4);
    return word;
}

__device__ __forceinline__ void score_histogram(const float* column, int lo, int hi, unsigned* hist) {
    walk_scores(column, lo, hi, [&](int, int, float4 v, unsigned ok) {
        const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if ((ok >> k) & 1U) { atomicAdd(hist + score_bin(s[k]), 1U); }
        }
    });
}

// The bin holding the top-th highest score: sh.bin, the count above it and its own count.
__device__ void find_score_bin(const unsigned* hist, int top_blocks, SelectShared& sh) {
    const int tid = static_cast<int>(threadIdx.x);
    int count[4], local = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        count[k] = static_cast<int>(hist[kScoreBins - 1 - 4 * tid - k]);
        local += count[k];
    }
    int above = 0;
    SelectShared::Scan(sh.scan).ExclusiveSum(local, above);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (above < top_blocks && top_blocks <= above + count[k]) {
            sh.bin       = kScoreBins - 1 - 4 * tid - k;
            sh.bin_above = above;
            sh.bin_count = count[k];
        }
        above += count[k];
    }
    __syncthreads();
}

// One pass over scores [lo, hi): scores above the bin set their bit in `bitmap` (word w holds
// scores lo + 32 w ..), the bin's scores append their order keys to `candidates` (up to
// kCandidateCapacity; `count` counts them all) and fold into sh.and_key / sh.or_key.
__device__ void filter_scores(const float* column, int lo, int hi, int bin, unsigned* bitmap,
                              std::uint64_t* candidates, unsigned* count, SelectShared& sh) {
    const int lane              = static_cast<int>(threadIdx.x) % 32;
    unsigned long long and_key  = ~0ULL, or_key = 0;
    walk_scores(column, lo, hi, [&](int chunk, int base, float4 v, unsigned ok) {
        const float s[4] = {v.x, v.y, v.z, v.w};
        unsigned above = 0, inside = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            if ((ok >> k) & 1U) {
                const int b = score_bin(s[k]);
                above |= static_cast<unsigned>(b > bin) << k;
                inside |= static_cast<unsigned>(b == bin) << k;
            }
        }
        const unsigned word = chunk_word(above, lane);
        if ((lane & 7) == 0) { bitmap[chunk * 4 + lane / 8] = word; }
        const int mine = __popc(inside);
        if (__any_sync(kFullMask, mine != 0)) {
            int inclusive = mine;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int other = __shfl_up_sync(kFullMask, inclusive, o);
                if (lane >= o) { inclusive += other; }
            }
            unsigned start = 0;
            if (lane == 31) { start = atomicAdd(count, static_cast<unsigned>(inclusive)); }
            start    = __shfl_sync(kFullMask, start, 31);
            int slot = static_cast<int>(start) + inclusive - mine;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if ((inside >> k) & 1U) {
                    const std::uint64_t key = score_id_order_key(s[k], base + k);
                    if (slot < kCandidateCapacity) { candidates[slot] = key; }
                    ++slot;
                    and_key &= key;
                    or_key |= key;
                }
            }
        }
    });
    const unsigned and_hi = __reduce_and_sync(kFullMask, static_cast<unsigned>(and_key >> 32));
    const unsigned and_lo = __reduce_and_sync(kFullMask, static_cast<unsigned>(and_key));
    const unsigned or_hi  = __reduce_or_sync(kFullMask, static_cast<unsigned>(or_key >> 32));
    const unsigned or_lo  = __reduce_or_sync(kFullMask, static_cast<unsigned>(or_key));
    if (lane == 0) {
        atomicAnd(&sh.and_key, (static_cast<unsigned long long>(and_hi) << 32) | and_lo);
        atomicOr(&sh.or_key, (static_cast<unsigned long long>(or_hi) << 32) | or_lo);
    }
}

// Keys of the candidate buffer.
struct CandidateKeys {
    const std::uint64_t* keys;
    int count;
    template <class F>
    __device__ __forceinline__ void for_each(F&& f) const {
        for (int i = static_cast<int>(threadIdx.x); i < count; i += kSelectThreads) { f(keys[i]); }
    }
};

// Keys of the bin's scores, read from the column (when the bin exceeds the candidate buffer).
struct BinKeys {
    const float* column;
    int n;
    int bin;
    template <class F>
    __device__ __forceinline__ void for_each(F&& f) const {
        walk_scores(column, 0, n, [&](int, int base, float4 v, unsigned ok) {
            const float s[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (((ok >> k) & 1U) && score_bin(s[k]) == bin) { f(score_id_order_key(s[k], base + k)); }
            }
        });
    }
};

// Warp 0: the digit holding the `remaining`-th highest key among sh.radix's 256 bins.
__device__ __forceinline__ void choose_digit(SelectShared& sh, int remaining) {
    const int lane = static_cast<int>(threadIdx.x);
    int count[8], local = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        count[k] = static_cast<int>(sh.radix[255 - 8 * lane - k]);
        local += count[k];
    }
    int inclusive = local;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int other = __shfl_up_sync(kFullMask, inclusive, o);
        if (lane >= o) { inclusive += other; }
    }
    int above = inclusive - local;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (above < remaining && remaining <= above + count[k]) {
            sh.pick_digit = 255 - 8 * lane - k;
            sh.pick_above = above;
            sh.pick_count = count[k];
        }
        above += count[k];
    }
}

// The `needed`-th highest of a set of distinct 64-bit keys, as a threshold T: exactly `needed` keys
// of the set are >= T. Radix select, 8 bits a pass from the highest bit in which the keys differ
// (and_keys / or_keys over the set), stopping once a digit's keys are all taken. Block-wide.
template <class Keys>
__device__ std::uint64_t radix_threshold(const Keys& keys, int needed, std::uint64_t and_keys,
                                         std::uint64_t or_keys, SelectShared& sh) {
    const std::uint64_t differ = and_keys ^ or_keys;
    if (differ == 0) { return or_keys; } // a single key
    const int top        = 63 - __clzll(static_cast<long long>(differ));
    std::uint64_t mask   = top == 63 ? 0 : ~((std::uint64_t{2} << top) - 1); // bits every key shares
    std::uint64_t prefix = or_keys & mask;
    int remaining        = needed;
    int high             = top;
    for (;;) {
        const int low             = high >= 7 ? high - 7 : 0;
        const unsigned digit_mask = (2U << (high - low)) - 1;
        for (int i = static_cast<int>(threadIdx.x); i < 256; i += kSelectThreads) { sh.radix[i] = 0; }
        __syncthreads();
        keys.for_each([&](std::uint64_t key) {
            if ((key & mask) == prefix) {
                atomicAdd(&sh.radix[static_cast<unsigned>(key >> low) & digit_mask], 1U);
            }
        });
        __syncthreads();
        if (threadIdx.x < 32) { choose_digit(sh, remaining); }
        __syncthreads();
        remaining -= sh.pick_above;
        const bool done = sh.pick_count == remaining || low == 0;
        prefix |= static_cast<std::uint64_t>(sh.pick_digit) << low;
        mask |= static_cast<std::uint64_t>(digit_mask) << low;
        __syncthreads(); // every thread has read sh.pick_* before the next pass or caller reuses sh
        if (done) { return prefix; }
        high = low - 1;
    }
}

// Writes id_base + (bit positions of words[0..nwords)) in ascending order to out. Block-wide.
__device__ void compact_words(const unsigned* words, int nwords, int id_base, std::int32_t* out,
                              SelectShared& sh) {
    const int per   = (nwords + kSelectThreads - 1) / kSelectThreads;
    const int begin = min(nwords, static_cast<int>(threadIdx.x) * per);
    const int end   = min(nwords, begin + per);
    int local       = 0;
    for (int w = begin; w < end; ++w) { local += __popc(words[w]); }
    int offset = 0;
    SelectShared::Scan(sh.scan).ExclusiveSum(local, offset);
    for (int w = begin; w < end; ++w) {
        unsigned bits = words[w];
        while (bits != 0) {
            out[offset++] = id_base + 32 * w + (__ffs(static_cast<int>(bits)) - 1);
            bits &= bits - 1;
        }
    }
}

// Grid (slices, columns of the group). Each CTA adds the histogram of its slice
// of the column into the column's global histogram (zeroed by qsa_score_kernel); the last CTA to
// arrive finds the bin, filters the column (bitmap above the bin, candidates in it), selects the
// candidates by radix and writes the ids in ascending order.
__global__ void __launch_bounds__(kSelectThreads)
    qsa_select_global_kernel(const std::int32_t* __restrict__ positions, int ratio, int top_blocks,
                             int group_begin, const float* __restrict__ scores, int score_stride,
                             unsigned* __restrict__ histograms, unsigned* __restrict__ arrivals,
                             std::int32_t* __restrict__ selected, std::int32_t* __restrict__ counts) {
    pdl::enter();
    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    __shared__ SelectShared sh;
    const int c = static_cast<int>(blockIdx.y), t = group_begin + c;
    const int slice = static_cast<int>(blockIdx.x), slices = static_cast<int>(gridDim.x);
    const int n = (positions[t] + 1) / ratio;
    if (n <= top_blocks) {
        if (slice == 0 && threadIdx.x == 0) { counts[t] = -1; }
        return;
    }
    const float* column = scores + static_cast<std::size_t>(c) * score_stride;
    auto* hist          = reinterpret_cast<unsigned*>(dynamic_smem); // [kScoreBins], then candidates
    for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) { hist[i] = 0; }
    __syncthreads();
    score_histogram(column, slice_begin(n, slice, slices), slice_begin(n, slice + 1, slices), hist);
    __syncthreads();
    unsigned* column_hist = histograms + static_cast<std::size_t>(c) * kScoreBins;
    for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) {
        if (hist[i] != 0) { atomicAdd(column_hist + i, hist[i]); }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) { sh.last = atomicAdd(arrivals + c, 1U) + 1 == static_cast<unsigned>(slices); }
    __syncthreads();
    if (sh.last == 0) { return; }
    __threadfence();

    for (int i = static_cast<int>(threadIdx.x); i < kScoreBins; i += kSelectThreads) { hist[i] = __ldcg(column_hist + i); }
    if (threadIdx.x == 0) {
        sh.candidates = 0;
        sh.and_key    = ~0ULL;
        sh.or_key     = 0;
    }
    __syncthreads();
    find_score_bin(hist, top_blocks, sh);
    const int bin = sh.bin, needed = top_blocks - sh.bin_above, in_bin = sh.bin_count;
    auto* candidates = reinterpret_cast<std::uint64_t*>(dynamic_smem); // the histogram is read
    auto* bitmap     = reinterpret_cast<unsigned*>(dynamic_smem + kCandidateBytes);
    filter_scores(column, 0, n, bin, bitmap, candidates, &sh.candidates, sh);
    __syncthreads();
    const bool buffered = in_bin <= kCandidateCapacity;
    const std::uint64_t and_key = sh.and_key, or_key = sh.or_key;
    const std::uint64_t threshold = buffered ? radix_threshold(CandidateKeys{candidates, in_bin}, needed, and_key, or_key, sh)
                                             : radix_threshold(BinKeys{column, n, bin}, needed, and_key, or_key, sh);
    if (buffered) {
        for (int i = static_cast<int>(threadIdx.x); i < in_bin; i += kSelectThreads) {
            const std::uint64_t key = candidates[i];
            if (key >= threshold) {
                const int id = id_from_order_key(key);
                atomicOr(bitmap + id / 32, 1U << (id % 32));
            }
        }
    } else {
        const int lane = static_cast<int>(threadIdx.x) % 32;
        walk_scores(column, 0, n, [&](int chunk, int base, float4 v, unsigned ok) {
            const float s[4] = {v.x, v.y, v.z, v.w};
            unsigned chosen  = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (((ok >> k) & 1U) && score_bin(s[k]) == bin && score_id_order_key(s[k], base + k) >= threshold) {
                    chosen |= 1U << k;
                }
            }
            const unsigned word = chunk_word(chosen, lane);
            if ((lane & 7) == 0) { bitmap[chunk * 4 + lane / 8] |= word; }
        });
    }
    __syncthreads();
    compact_words(bitmap, (n + 31) / 32, 0, selected + static_cast<std::size_t>(t) * top_blocks, sh);
    if (threadIdx.x == 0) { counts[t] = top_blocks; }
}

// ------------------------------------------------------------------------------------------ host

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa select: ") + message); }
}

int ceil_div(std::int64_t a, std::int64_t b) { return static_cast<int>((a + b - 1) / b); }

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

int multiprocessors() {
    static const int count = [] {
        int device = 0, value = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    }();
    return count;
}

struct Layout {
    std::int32_t group  = 0; // columns per score pass
    std::int32_t stride = 0; // floats per column score row
    std::size_t scores = 0, state = 0, total = 0;
};

Layout layout(const QsaGeometry& g, std::int32_t columns, std::int32_t max_context) {
    Layout l;
    l.group  = std::min(columns, kQsaSelectGroupColumns);
    l.stride = (max_context / g.ratio + 1 + 31) / 32 * 32;
    l.scores = align256(sizeof(float) * static_cast<std::size_t>(l.group) * l.stride);
    // kScoreBins histogram words, then one arrival counter, per column.
    l.state = align256(sizeof(std::uint32_t) * static_cast<std::size_t>(l.group) * (kScoreBins + 1));
    l.total = l.scores + l.state;
    return l;
}

ScoreTiles score_tiles(std::int32_t begin, std::int32_t columns, std::int32_t width) {
    ScoreTiles tiles;
    for (std::int32_t c = 0; c < columns;) {
        const std::int32_t t       = begin + c;
        const std::int32_t row_end = (t / width + 1) * width;
        const std::int32_t n       = std::min({kScoreTileColumns, columns - c, row_end - t});
        tiles.first[tiles.count]   = static_cast<std::uint8_t>(c);
        tiles.columns[tiles.count] = static_cast<std::uint8_t>(n);
        ++tiles.count;
        c += n;
    }
    return tiles;
}

// Opts a kernel into `bytes` of dynamic shared memory beyond the default 48 KiB, once per size.
template <class Kernel>
void reserve_shared(Kernel kernel, std::size_t bytes, std::atomic<std::size_t>& reserved) {
    if (bytes <= 48 * 1024 || bytes <= reserved.load(std::memory_order_acquire)) { return; }
    int device = 0, limit = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    require(bytes <= static_cast<std::size_t>(limit), "max_context needs more shared memory than a CTA has");
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    std::size_t seen = reserved.load(std::memory_order_relaxed);
    while (seen < bytes && !reserved.compare_exchange_weak(seen, bytes, std::memory_order_acq_rel)) {}
}

std::atomic<std::size_t> g_global_reserved{0};

} // namespace

std::size_t qsa_select_scratch_bytes(const QsaGeometry& geometry, std::int32_t columns, std::int32_t max_context) {
    require(columns > 0 && max_context > 0 && geometry.ratio > 0, "columns and max_context must be positive");
    return layout(geometry, columns, max_context).total;
}

QsaSelectScores qsa_select_scores(const void* scratch, const QsaGeometry& geometry, std::int32_t max_context) {
    return {static_cast<const float*>(scratch), layout(geometry, 1, max_context).stride};
}

void qsa_select(const Tensor& index_q, const Tensor& pooled_pages, const QsaBatch& batch,
                const QsaGeometry& geometry, std::int32_t max_context, void* scratch,
                std::size_t scratch_bytes, const QsaSelectOutput& output, cudaStream_t stream,
                const QsaPageSpaces& spaces) {
    const std::int32_t columns = batch.batch * batch.width;
    require(geometry.index_head_dim == kIndexDim && geometry.index_heads >= 1 &&
                geometry.index_heads <= kScoreHeadSlots && geometry.ratio > 1 && kIndexDim % geometry.ratio == 0 &&
                geometry.budget > 0 && geometry.budget % geometry.ratio == 0,
            "unsupported geometry");
    require(columns > 0 && index_q.data != nullptr && index_q.dtype == DType::BF16 && index_q.is_contiguous() &&
                index_q.ne[0] == kIndexDim && index_q.ne[1] == geometry.index_heads && index_q.ne[2] == columns,
            "index queries disagree with the batch");
    require(pooled_pages.data != nullptr && pooled_pages.dtype == DType::BF16 && pooled_pages.is_contiguous() &&
                pooled_pages.ne[0] == kIndexDim / geometry.ratio,
            "pooled pages must be the layer's BF16 plane");
    require(output.selected != nullptr && output.counts != nullptr, "selection outputs are missing");
    const Layout l = layout(geometry, columns, max_context);
    require(scratch != nullptr && scratch_bytes >= l.total, "scratch is too small");

    const std::int32_t top_blocks = geometry.budget / geometry.ratio;
    const std::int32_t max_blocks = max_context / geometry.ratio;
    auto* base                    = static_cast<unsigned char*>(scratch);
    auto* scores                  = reinterpret_cast<float*>(base);
    auto* state                   = reinterpret_cast<std::uint32_t*>(base + l.scores);
    auto* histograms              = state;
    auto* arrivals                = state + static_cast<std::size_t>(l.group) * kScoreBins;

    // Dynamic shared memory: candidates (aliasing the histogram) and the column bitmap.
    const std::size_t smem = kCandidateBytes + sizeof(unsigned) * 4 * static_cast<std::size_t>(ceil_div(max_blocks + 1, 128));
    reserve_shared(qsa_select_global_kernel, smem, g_global_reserved);

    const auto* index      = static_cast<const bf16*>(index_q.data);
    const auto* pooled     = static_cast<const bf16*>(pooled_pages.data);
    const auto* tables     = static_cast<const std::int32_t*>(batch.block_tables.data);
    const int table_stride = static_cast<int>(batch.block_tables.ne[0]);
    const auto* rows       = static_cast<const std::int32_t*>(batch.table_rows.data);
    const auto* positions  = static_cast<const std::int32_t*>(batch.positions.data);
    const int sms          = multiprocessors();
    const int max_steps    = ceil_div(max_blocks + 1, kScoreStepBlocks);
    for (std::int32_t begin = 0; begin < columns; begin += l.group) {
        const std::int32_t n   = std::min(l.group, columns - begin);
        const ScoreTiles tiles = score_tiles(begin, n, batch.width);
        const int ctas = std::clamp(ceil_div(std::int64_t{kScoreCtasPerSm} * sms, tiles.count), 1,
                                    std::max(1, ceil_div(max_steps, kScoreWarps)));
        CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(ctas), static_cast<unsigned>(tiles.count)),
                                         dim3(kScoreThreads), 0, stream},
                                        qsa_score_kernel, index, geometry.index_heads, pooled, tables, table_stride,
                                        rows, positions, batch.width, geometry.ratio, top_blocks, begin, tiles,
                                        scores, l.stride, state, l.group * (kScoreBins + 1), spaces));
        const int slices = std::clamp(std::min(ceil_div(max_blocks, kSliceBlocks), ceil_div(std::int64_t{4} * sms, n)),
                                      1, kMaxSlices);
        CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(slices), static_cast<unsigned>(n)),
                                         dim3(kSelectThreads), smem, stream},
                                        qsa_select_global_kernel, positions, geometry.ratio, top_blocks, begin,
                                        static_cast<const float*>(scores), l.stride, histograms, arrivals,
                                        output.selected, output.counts));
    }
}

} // namespace ninfer::ops::detail
