// Screened heads (include/infernix/ops/screened_head.h): every row of a vocabulary head scored from its
// converter-written 2-bit copy (q2_g64_fp16), and the rows near each column's best rescored exactly
// with the exact head's own arithmetic (B-03; Basalt's two-stage head).
//
// Cost model (RTX 5090, K = 2560): the screen streams 720 bytes per row (2.25 bits a weight) against
// the exact head's 1,360 (q4_g64_fp16 proposal head, 131,072 rows: 94 MB against 178 MB) or 5,120 (a
// BF16 vocabulary head, 248,320 rows: 179 MB against 1.27 GB), at the DRAM roofline. Its arithmetic is
// int8 tensor-core products: each 2-bit code is expanded with two integer instructions per four codes
// (no runtime repacking: the row-split layout's four codes per byte line up with the MMA fragment once
// the K order inside each 16-weight word is permuted, which the activation copy absorbs). The rescore
// reads at most `rescored` rows per column.
//
// Kernels, in launch order:
//   screen     persistent CTAs (one per SM), four warps splitting K per 16-row item, a four-stage
//              cp.async ring of items (11.5 KB each), the column quantized to int8 per 64-group in
//              shared memory. Per column it keeps the CTA's best kListLength rows and (top form) the
//              online softmax partial of its rows; the logits form writes every screen score.
//   select     one CTA per column: the best screen score M, the rows within the margin of it (at most
//              `rescored`, the best first, at least the ranks), by radix selection on (score, row)
//              keys when the count is out of bounds.
//   rescore    the exact head's arithmetic for each selected row (the top form inside the selection
//              CTA; the logits form in its own launch: q8 rows by warps, BF16 rows by the tensor-core
//              tile over the listed rows).

#include "infernix/ops/screened_head.h"
#include "ops/screened_head/screened_head_launch.h"

#include "core/device.h"
#include "core/layout.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/score_id_order.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using namespace detail::screened_head;
using bf16 = __nv_bfloat16;

constexpr int kGroups        = kHidden / 64;     // 40 weight groups of 64 per row
constexpr int kScreenWarps   = 4;                // K split of an item: warp w takes groups [10w, 10w + 10)
constexpr int kWarpGroups    = kGroups / kScreenWarps;
constexpr int kScreenThreads = kScreenWarps * 32;
constexpr int kItemRows      = 16;               // one m16 row tile
constexpr int kStages        = 4;
constexpr int kCodeRowBytes  = kHidden / 4;      // 640: four 2-bit codes per byte
constexpr int kCodePitch     = kCodeRowBytes + 16; // rows g and g + 8 of a quad's word on distinct banks
constexpr int kScaleRowBytes = kGroups * 2;      // 80
constexpr int kStageCodeBytes = kItemRows * kCodePitch;
constexpr int kStageBytes     = kStageCodeBytes + kItemRows * kScaleRowBytes;
constexpr int kColumnPitch    = kHidden + 64;    // a column's int8 copy; 16 banks apart from the next
constexpr int kSelectThreads  = 512;
constexpr int kSelectWarps    = kSelectThreads / 32;
constexpr int kMaxRescored    = 256;
constexpr int kMaxEntries     = kMaxGrid * kListLength; // candidates of one column
constexpr int kEntriesPerThread = kMaxEntries / kSelectThreads;
static_assert(kGroups % kScreenWarps == 0 && kMaxEntries % kSelectThreads == 0);
static_assert(kScreenThreads == 128 && kStageBytes % 16 == 0 && kStageCodeBytes % 16 == 0);

__device__ __forceinline__ float negative_infinity() { return __int_as_float(0xff800000); }

template <int Tiles>
__host__ __device__ constexpr int screen_columns() {
    return Tiles * 8;
}

// Dynamic shared memory of the screen kernel: the item ring, the int8 columns, each (column, group)'s
// FP32 scale and twice its code sum, and the four warps' partial sums (warp 0's slot then holds the
// item's scores).
template <int Tiles>
constexpr int screen_shared_bytes() {
    constexpr int C = screen_columns<Tiles>();
    return kStages * kStageBytes + C * kColumnPitch + C * kGroups * 8 + kScreenWarps * kItemRows * C * 4;
}
static_assert(screen_shared_bytes<2>() <= 101376, "the screen kernel's ring and columns fit one CTA");

struct ScreenParams {
    const bf16* x;              // [K, T]
    const std::uint8_t* codes;  // [R, 640]
    const std::uint8_t* scales; // [R, 40] binary16
    std::int32_t rows;          // R, a multiple of 16
    std::int32_t valid_rows;
    std::int32_t columns;       // T
    float* logits;              // [R, T] (the logits form)
    float2* lists;              // [T][grid][kListLength]: (score, row bits)
    float2* partials;           // [T][grid]: (max, sum of exp(score - max)) (the top form)
};

// (v, r) ranks above (w, s): the higher score, on equal scores the lower row.
__device__ __forceinline__ bool ranks_above(float v, int r, float w, int s) {
    return v > w || (v == w && r < s);
}

// The column's quantized int8 copy in the MMA's K order. Inside each 16-weight word of a code row,
// field f of byte b is weight 4b + f (row-split q2 packing); the A fragment takes field f of the four
// bytes as one register (k slots 4t..4t+3, t the thread's index in its quad). So the activation chunk
// of 16 values is stored transposed: byte 4f + b holds x[4b + f]. Columns past T are zero.
template <int Tiles>
__device__ void quantize_columns(const ScreenParams& p, std::uint8_t* xq, float2* xs) {
    constexpr int C = screen_columns<Tiles>();
    // Eight threads per (column, group), eight values each; 128 threads cover 16 groups per pass and
    // every pass is uniform, so the eight-lane reductions never diverge.
    for (int item = static_cast<int>(threadIdx.x); item < C * kGroups * 8; item += kScreenThreads) {
        const int column = item / (kGroups * 8), rest = item % (kGroups * 8);
        const int group = rest / 8, octet = rest % 8;
        float v[8];
        if (column < p.columns) {
            const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(p.x + static_cast<std::size_t>(column) * kHidden +
                                                                    group * 64 + octet * 8));
            const bf16* values = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[j] = __bfloat162float(values[j]); }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[j] = 0.0f; }
        }
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { amax = fmaxf(amax, fabsf(v[j])); }
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset)); }
        const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
        int sum = 0;
        std::uint8_t* chunk = xq + column * kColumnPitch + group * 64 + (octet / 2) * 16;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int q = max(-127, min(127, __float2int_rn(v[j] * inverse)));
            sum += q;
            const int position = (octet & 1) * 8 + j; // the weight within its 16-value chunk
            chunk[4 * (position % 4) + position / 4] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        }
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, offset); }
        if (octet == 0) { xs[column * kGroups + group] = make_float2(amax / 127.0f, static_cast<float>(2 * sum)); }
    }
}

// One persistent CTA per SM over 16-row items i = blockIdx.x + k * gridDim.x. A column's screen
// scores, lists and partials depend on that column alone (its own int8 copy, a fixed grid).
template <int Tiles, bool Logits, bool Softmax>
__global__ void __launch_bounds__(kScreenThreads, 1) screen_kernel(ScreenParams p) {
    extern __shared__ __align__(16) std::uint8_t shared[];
    constexpr int C        = screen_columns<Tiles>();
    std::uint8_t* stages   = shared;
    std::uint8_t* xq       = shared + kStages * kStageBytes;
    float2* xs             = reinterpret_cast<float2*>(xq + C * kColumnPitch);
    float* partial         = reinterpret_cast<float*>(xs + C * kGroups);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int items = p.rows / kItemRows;
    const int first = static_cast<int>(blockIdx.x), stride = static_cast<int>(gridDim.x);
    const int mine  = first < items ? (items - first + stride - 1) / stride : 0;

    // Item k of this CTA into stage k % kStages: 16 contiguous code rows (into padded rows) and their
    // contiguous scales.
    const auto issue = [&](int k) {
        const std::size_t item = static_cast<std::size_t>(first) + static_cast<std::size_t>(k) * stride;
        std::uint8_t* stage    = stages + (k % kStages) * kStageBytes;
        const std::uint8_t* code_source  = p.codes + item * (kItemRows * kCodeRowBytes);
        const std::uint8_t* scale_source = p.scales + item * (kItemRows * kScaleRowBytes);
        for (int c = tid; c < kItemRows * kCodeRowBytes / 16; c += kScreenThreads) {
            const int row = c / (kCodeRowBytes / 16), part = c % (kCodeRowBytes / 16);
            cp_async<16, Cache::cg>(stage + row * kCodePitch + part * 16, code_source + c * 16);
        }
        for (int c = tid; c < kItemRows * kScaleRowBytes / 16; c += kScreenThreads) {
            cp_async<16, Cache::cg>(stage + kStageCodeBytes + c * 16, scale_source + c * 16);
        }
    };

    // The codes and scales are immutable: the first items stream in before the wait.
#pragma unroll
    for (int k = 0; k < kStages - 1; ++k) {
        if (k < mine) { issue(k); }
        cp_commit();
    }
    pdl::enter_streaming();
    quantize_columns<Tiles>(p, xq, xs);

    // Thread c < T keeps column c's best rows (descending) and its softmax partial.
    float best[kListLength];
    int best_row[kListLength];
#pragma unroll
    for (int j = 0; j < kListLength; ++j) {
        best[j]     = negative_infinity();
        best_row[j] = INT_MAX;
    }
    float column_max = negative_infinity(), column_sum = 0.0f;

    for (int k = 0; k < mine; ++k) {
        cp_wait<kStages - 2>();
        __syncthreads(); // item k landed; every thread is done with item k - 1's stage and scores
        if (k + kStages - 1 < mine) { issue(k + kStages - 1); }
        cp_commit();
        const std::uint8_t* stage = stages + (k % kStages) * kStageBytes;
        const auto* scales        = reinterpret_cast<const __half*>(stage + kStageCodeBytes);
        float acc[Tiles][4];
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
#pragma unroll 2
        for (int j = 0; j < kWarpGroups; ++j) {
            const int group = warp * kWarpGroups + j;
            // Word t of the group in rows g and g + 8, each 2-bit code moved to its level c + 2 in [0, 3].
            const std::uint32_t w0 =
                *reinterpret_cast<const std::uint32_t*>(stage + g * kCodePitch + group * 16 + t * 4) ^ 0xAAAAAAAAu;
            const std::uint32_t w1 =
                *reinterpret_cast<const std::uint32_t*>(stage + (g + 8) * kCodePitch + group * 16 + t * 4) ^
                0xAAAAAAAAu;
            const float s0 = __half2float(scales[g * kGroups + group]);
            const float s1 = __half2float(scales[(g + 8) * kGroups + group]);
            constexpr std::uint32_t kLevels = 0x03030303u;
#pragma unroll
            for (int nt = 0; nt < Tiles; ++nt) {
                const uint4 b = *reinterpret_cast<const uint4*>(xq + (nt * 8 + g) * kColumnPitch + group * 64 + t * 16);
                int c0, c1, c2, c3;
                mma_s8s8_k32_zero(c0, c1, c2, c3, w0 & kLevels, w1 & kLevels, (w0 >> 2) & kLevels, (w1 >> 2) & kLevels,
                                  b.x, b.y);
                mma_s8(c0, c1, c2, c3, (w0 >> 4) & kLevels, (w1 >> 4) & kLevels, (w0 >> 6) & kLevels,
                       (w1 >> 6) & kLevels, b.z, b.w);
                // sum c x = sum (c + 2) x - 2 sum x, all exact in int32 and float.
                const float2 x0 = xs[(nt * 8 + 2 * t) * kGroups + group];
                const float2 x1 = xs[(nt * 8 + 2 * t + 1) * kGroups + group];
                acc[nt][0] = fmaf(s0 * x0.x, static_cast<float>(c0) - x0.y, acc[nt][0]);
                acc[nt][1] = fmaf(s0 * x1.x, static_cast<float>(c1) - x1.y, acc[nt][1]);
                acc[nt][2] = fmaf(s1 * x0.x, static_cast<float>(c2) - x0.y, acc[nt][2]);
                acc[nt][3] = fmaf(s1 * x1.x, static_cast<float>(c3) - x1.y, acc[nt][3]);
            }
        }
        float* own = partial + warp * kItemRows * C;
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) {
            own[g * C + nt * 8 + 2 * t]           = acc[nt][0];
            own[g * C + nt * 8 + 2 * t + 1]       = acc[nt][1];
            own[(g + 8) * C + nt * 8 + 2 * t]     = acc[nt][2];
            own[(g + 8) * C + nt * 8 + 2 * t + 1] = acc[nt][3];
        }
        __syncthreads();
        const int row0 = (first + k * stride) * kItemRows;
        for (int e = tid; e < kItemRows * C; e += kScreenThreads) {
            const float v = ((partial[e] + partial[kItemRows * C + e]) + partial[2 * kItemRows * C + e]) +
                            partial[3 * kItemRows * C + e];
            partial[e] = v;
            if constexpr (Logits) {
                const int column = e % C;
                if (column < p.columns) {
                    p.logits[static_cast<std::size_t>(column) * p.rows + row0 + e / C] = v;
                }
            }
        }
        __syncthreads();
        if (tid < p.columns) {
            float item_max = negative_infinity();
#pragma unroll 4
            for (int r = 0; r < kItemRows; ++r) {
                const float v = partial[r * C + tid];
                const int row = row0 + r;
                if (row >= p.valid_rows) { continue; }
                item_max = fmaxf(item_max, v);
                if (!ranks_above(v, row, best[kListLength - 1], best_row[kListLength - 1])) { continue; }
                // Insert in order with constant indices (registers, no local memory).
                bool placed = false;
#pragma unroll
                for (int i = kListLength - 1; i > 0; --i) {
                    if (!placed) {
                        if (ranks_above(v, row, best[i - 1], best_row[i - 1])) {
                            best[i]     = best[i - 1];
                            best_row[i] = best_row[i - 1];
                        } else {
                            best[i]     = v;
                            best_row[i] = row;
                            placed      = true;
                        }
                    }
                }
                if (!placed) {
                    best[0]     = v;
                    best_row[0] = row;
                }
            }
            if constexpr (Softmax) {
                if (item_max > negative_infinity()) {
                    const float top = fmaxf(column_max, item_max);
                    float sum       = column_sum * __expf(column_max - top);
                    for (int r = 0; r < kItemRows && row0 + r < p.valid_rows; ++r) { sum += __expf(partial[r * C + tid] - top); }
                    column_max = top;
                    column_sum = sum;
                }
            }
        }
    }
    pdl::trigger_dependents();
    if (tid < p.columns) {
        float2* list = p.lists + (static_cast<std::size_t>(tid) * gridDim.x + blockIdx.x) * kListLength;
#pragma unroll
        for (int j = 0; j < kListLength; ++j) { list[j] = make_float2(best[j], __int_as_float(best_row[j])); }
        if constexpr (Softmax) {
            p.partials[static_cast<std::size_t>(tid) * gridDim.x + blockIdx.x] = make_float2(column_max, column_sum);
        }
    }
}

// ---- Selection (one CTA of kSelectThreads per column)

struct SelectShared {
    std::uint32_t histogram[256];
    std::uint64_t reduce64[kSelectWarps];
    int reduce32[kSelectWarps];
    std::uint64_t prefix;
    int need;
    int digit_above; // keys above the chosen digit, in this pass
    int digit;
    int rows[kMaxRescored];
    float screen[kMaxRescored];
};

__device__ __forceinline__ std::uint64_t block_max(std::uint64_t value, SelectShared& s) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const std::uint64_t other = __shfl_xor_sync(0xffffffffu, value, offset);
        value                     = other > value ? other : value;
    }
    if (lane == 0) { s.reduce64[warp] = value; }
    __syncthreads();
    std::uint64_t out = s.reduce64[0];
    for (int w = 1; w < kSelectWarps; ++w) { out = s.reduce64[w] > out ? s.reduce64[w] : out; }
    __syncthreads();
    return out;
}

__device__ __forceinline__ int block_sum(int value, SelectShared& s) {
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xffffffffu, value, offset); }
    if (lane == 0) { s.reduce32[warp] = value; }
    __syncthreads();
    int out = 0;
    for (int w = 0; w < kSelectWarps; ++w) { out += s.reduce32[w]; }
    __syncthreads();
    return out;
}

// The `target`-th largest of the block's distinct nonzero keys (the keys at or above it are exactly
// `target`), by eight passes of 8-bit radix selection from the top byte.
__device__ std::uint64_t select_key(const std::uint64_t (&keys)[kEntriesPerThread], int target, SelectShared& s) {
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    if (tid == 0) {
        s.prefix = 0;
        s.need   = target;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        if (tid < 256) { s.histogram[tid] = 0; }
        __syncthreads();
        const std::uint64_t prefix = s.prefix;
        const std::uint64_t mask   = shift == 56 ? 0 : ~((std::uint64_t{1} << (shift + 8)) - 1);
#pragma unroll
        for (int i = 0; i < kEntriesPerThread; ++i) {
            if (keys[i] != 0 && (keys[i] & mask) == prefix) {
                atomicAdd(&s.histogram[(keys[i] >> shift) & 0xFF], 1u);
            }
        }
        __syncthreads();
        if (warp == 0) {
            // Lane l owns digits 255 - 8l down to 248 - 8l; scan from the top.
            int counts[8];
            int own = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                counts[i] = static_cast<int>(s.histogram[255 - 8 * lane - i]);
                own += counts[i];
            }
            int above = own; // inclusive scan over lanes (lane 0 = the highest digits)
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int other = __shfl_up_sync(0xffffffffu, above, offset);
                if (lane >= offset) { above += other; }
            }
            const int before = above - own;
            const int need   = s.need;
            const unsigned holder = __ballot_sync(0xffffffffu, before < need && above >= need);
            if (lane == __ffs(holder) - 1) {
                int running = before;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (running + counts[i] >= need) {
                        s.digit       = 255 - 8 * lane - i;
                        s.digit_above = running;
                        break;
                    }
                    running += counts[i];
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            s.prefix |= static_cast<std::uint64_t>(s.digit) << shift;
            s.need -= s.digit_above;
        }
        __syncthreads();
    }
    return s.prefix;
}

// Selects column `column`'s rows to rescore from its lists ([grid][kListLength]) into s.rows / s.screen
// in list order; returns their count.
__device__ int select_rows(const float2* lists, int grid, float margin, int ranks, int cap, SelectShared& s) {
    const int tid     = static_cast<int>(threadIdx.x);
    const int entries = grid * kListLength;
    std::uint64_t keys[kEntriesPerThread];
    std::uint64_t top = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const int e = tid + i * kSelectThreads;
        keys[i]     = 0;
        if (e < entries) {
            const float2 entry = __ldcg(lists + e);
            const int row      = __float_as_int(entry.y);
            if (row != INT_MAX) { keys[i] = score_id_order_key(entry.x, row); }
        }
        top = keys[i] > top ? keys[i] : top;
    }
    top = block_max(top, s);
    const float threshold = score_from_order_key(top) - margin;
    int within = 0, valid = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        valid += keys[i] != 0;
        within += keys[i] != 0 && score_from_order_key(keys[i]) >= threshold;
    }
    within = block_sum(within, s);
    valid  = block_sum(valid, s);
    const int target = min(min(max(within, ranks), cap), valid);
    // Within the bounds every row within the margin is taken; otherwise the `target` largest keys (the
    // keys at or above the target-th, which is the cut).
    const bool by_margin    = target == within;
    const std::uint64_t cut = by_margin ? 0 : select_key(keys, target, s);
    // Compaction in entry order (entry e = tid + i * kSelectThreads, ranked by (i, tid)).
    int offset = 0;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const bool chosen =
            keys[i] != 0 && (by_margin ? score_from_order_key(keys[i]) >= threshold : keys[i] >= cut);
        const int lane = tid & 31, warp = tid >> 5;
        const unsigned ballot = __ballot_sync(0xffffffffu, chosen);
        if (lane == 0) { s.reduce32[warp] = __popc(ballot); }
        __syncthreads();
        int base = offset;
        for (int w = 0; w < warp; ++w) { base += s.reduce32[w]; }
        int total = 0;
        for (int w = 0; w < kSelectWarps; ++w) { total += s.reduce32[w]; }
        if (chosen) {
            const int position = base + __popc(ballot & ((1u << lane) - 1u));
            if (position < kMaxRescored) {
                s.rows[position]   = id_from_order_key(keys[i]);
                s.screen[position] = score_from_order_key(keys[i]);
            }
        }
        offset += total;
        __syncthreads();
    }
    return min(target, kMaxRescored);
}

// ---- Exact rows (the narrow mapping of projection_fp32: lane l sums chunks l, l + 32, ... of eight
// weights in order with fmaf, then the xor butterfly), for one (row, column).

__device__ __forceinline__ float warp_butterfly(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xffffffffu, v, offset); }
    return v;
}

// q4_g64_fp16: row r's nibbles at r * padded_k / 2 (low nibble first), its scales at r * padded_k / 64.
__device__ __forceinline__ float q4_row(const std::uint8_t* codes, const __half* scales, std::int64_t row, int padded_k,
                                        const bf16* x, int lane) {
    const auto* w      = reinterpret_cast<const std::uint32_t*>(codes + row * (padded_k / 2));
    const __half* srow = scales + row * (padded_k / 64);
    float sum          = 0.0f;
#pragma unroll
    for (int i = 0; i < kHidden / 8 / 32; ++i) {
        const int chunk            = lane + 32 * i;
        const std::uint32_t packed = __ldg(w + chunk);
        const float scale          = __half2float(srow[chunk / 8]);
        const uint4 xp             = *reinterpret_cast<const uint4*>(x + chunk * 8);
        const bf16* xv             = reinterpret_cast<const bf16*>(&xp);
        float wf[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int u = static_cast<int>((packed >> (4 * j)) & 0xFU);
            wf[j]       = static_cast<float>(u >= 8 ? u - 16 : u) * scale;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) { sum = fmaf(wf[j], __bfloat162float(xv[j]), sum); }
    }
    return warp_butterfly(sum);
}

// q8_g32_fp16: row r's codes at r * padded_k, its scales at r * padded_k / 32.
__device__ __forceinline__ float q8_row(const std::int8_t* codes, const __half* scales, std::int64_t row, int padded_k,
                                        const bf16* x, int lane) {
    const auto* w      = reinterpret_cast<const uint2*>(codes + row * padded_k);
    const __half* srow = scales + row * (padded_k / 32);
    float sum          = 0.0f;
#pragma unroll
    for (int i = 0; i < kHidden / 8 / 32; ++i) {
        const int chunk    = lane + 32 * i;
        const uint2 packed = __ldg(w + chunk);
        const float scale  = __half2float(srow[chunk / 4]);
        const uint4 xp     = __ldcg(reinterpret_cast<const uint4*>(x + chunk * 8));
        const bf16* xv     = reinterpret_cast<const bf16*>(&xp);
        const auto* wv     = reinterpret_cast<const std::int8_t*>(&packed);
        float wf[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { wf[j] = static_cast<float>(wv[j]) * scale; }
#pragma unroll
        for (int j = 0; j < 8; ++j) { sum = fmaf(wf[j], __bfloat162float(xv[j]), sum); }
    }
    return warp_butterfly(sum);
}

// ---- The top form's selection, rescore and ranks (one CTA per column)

struct TopParams {
    const bf16* x;
    const std::uint8_t* codes; // the q4 head
    const __half* scales;
    std::int32_t padded_k;
    const float2* lists;
    const float2* partials;
    std::int32_t grid;
    std::int32_t columns;
    float margin;
    std::int32_t cap;
    std::int32_t ranks;
    std::int32_t* rows_out;  // [ranks, T]
    float* logprobs_out;     // [ranks, T]
    const std::int32_t* row_ids;
    std::int32_t* tokens;
};

// The normalizer's tail rows, those at their screen scores, stand for their exact logits at s + v / 2:
// the converter's MSE screen leaves e = s + eps with eps about independent of s (the fit's residual is
// orthogonal to its copy), so E[exp(e) | s] = exp(s + v / 2) for a residual of variance v, and exp(s)
// alone undercounts the tail (by 0.3-0.6 nats on flat columns of the 131K-row proposal head). v is the
// mean square of e - s over the rescored rows outside the four best exact logits (the rows a column
// boosts carry residuals the screen attenuates), from at least kTailResidualRows rows, else 0.
constexpr int kTailResidualRows   = 8;
constexpr float kTailResidualMax  = 4.0f;
constexpr int kNormalizerTerms    = 5; // tail partials, rescored exact, rescored screen, residual squares, rows

struct TopShared {
    SelectShared select;
    __align__(16) bf16 x[kHidden];
    float rounded[kMaxRescored];
    float reduce[kNormalizerTerms][kSelectWarps];
    int taken[kMaxRescored];
    int picked[kScreenedHeadMaxRanks];
    float picked_value[kScreenedHeadMaxRanks];
};

__global__ void __launch_bounds__(kSelectThreads) screened_top_kernel(TopParams p) {
    __shared__ TopShared s;
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int column = static_cast<int>(blockIdx.x);
    pdl::enter();
    for (int i = tid; i < kHidden / 8; i += kSelectThreads) {
        reinterpret_cast<uint4*>(s.x)[i] =
            __ldcg(reinterpret_cast<const uint4*>(p.x + static_cast<std::size_t>(column) * kHidden) + i);
    }
    const int n = select_rows(p.lists + static_cast<std::size_t>(column) * p.grid * kListLength, p.grid, p.margin,
                              p.ranks, p.cap, s.select);
    __syncthreads();
    for (int e = warp; e < n; e += kSelectWarps) {
        const float v = q4_row(p.codes, p.scales, s.select.rows[e], p.padded_k, s.x, lane);
        if (lane == 0) {
            s.rounded[e] = __bfloat162float(__float2bfloat16_rn(v)); // the logit argmax and top_logprobs read
            s.taken[e]   = 0;
        }
    }
    __syncthreads();
    if (warp == 0) {
        // The ranks among the rescored rows: the larger BF16 logit, on ties the lower row. All four are
        // taken whatever p.ranks, so the normalizer's residual rows (and with it every log-probability)
        // do not depend on the ranks a call asks for.
        for (int j = 0; j < kScreenedHeadMaxRanks; ++j) {
            float v = negative_infinity();
            int row = INT_MAX, at = -1;
            for (int e = lane; e < n; e += 32) {
                if (!s.taken[e] && ranks_above(s.rounded[e], s.select.rows[e], v, row)) {
                    v   = s.rounded[e];
                    row = s.select.rows[e];
                    at  = e;
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, offset);
                const int orow = __shfl_xor_sync(0xffffffffu, row, offset);
                const int oat  = __shfl_xor_sync(0xffffffffu, at, offset);
                if (ranks_above(ov, orow, v, row)) {
                    v   = ov;
                    row = orow;
                    at  = oat;
                }
            }
            if (lane == 0) {
                s.picked[j]       = at >= 0 ? row : 0;
                s.picked_value[j] = v;
                if (at >= 0) { s.taken[at] = 1; }
            }
            __syncwarp();
        }
    }
    __syncthreads();
    // The log-normalizer: the CTAs' partials over every row at its screen score, with the rescored
    // rows' screen terms swapped for their exact BF16 logits and the rest raised by exp(v / 2).
    float top = s.picked_value[0];
    for (int c = tid; c < p.grid; c += kSelectThreads) {
        top = fmaxf(top, __ldcg(p.partials + static_cast<std::size_t>(column) * p.grid + c).x);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset)); }
    if (lane == 0) { s.reduce[0][warp] = top; }
    __syncthreads();
    top = s.reduce[0][0];
    for (int w = 1; w < kSelectWarps; ++w) { top = fmaxf(top, s.reduce[0][w]); }
    __syncthreads();
    float terms[kNormalizerTerms] = {};
    for (int c = tid; c < p.grid; c += kSelectThreads) {
        const float2 part = __ldcg(p.partials + static_cast<std::size_t>(column) * p.grid + c);
        if (part.y > 0.0f) { terms[0] += part.y * expf(part.x - top); }
    }
    for (int e = tid; e < n; e += kSelectThreads) {
        terms[1] += expf(s.rounded[e] - top);
        terms[2] += expf(s.select.screen[e] - top);
        if (!s.taken[e]) {
            const float residual = s.rounded[e] - s.select.screen[e];
            terms[3] += residual * residual;
            terms[4] += 1.0f;
        }
    }
#pragma unroll
    for (int i = 0; i < kNormalizerTerms; ++i) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) { terms[i] += __shfl_xor_sync(0xffffffffu, terms[i], offset); }
        if (lane == 0) { s.reduce[i][warp] = terms[i]; }
    }
    __syncthreads();
    if (tid == 0) {
        float sums[kNormalizerTerms] = {};
        for (int i = 0; i < kNormalizerTerms; ++i) {
            for (int w = 0; w < kSelectWarps; ++w) { sums[i] += s.reduce[i][w]; }
        }
        const float residual_rows = sums[4];
        const float v = residual_rows >= static_cast<float>(kTailResidualRows)
                            ? fminf(sums[3] / residual_rows, kTailResidualMax)
                            : 0.0f;
        // The tail: every row's screen term less the rescored rows' (sums[0] - sums[2] >= 0 up to rounding).
        const float tail = fmaxf(sums[0] - sums[2], 0.0f) * expf(0.5f * v);
        float total      = sums[1] + tail;
        total                = fmaxf(total, expf(s.picked_value[0] - top)); // at least the pick's own term
        const float normalizer = top + logf(total);
        for (int j = 0; j < p.ranks; ++j) {
            p.rows_out[static_cast<std::size_t>(column) * p.ranks + j]     = s.picked[j];
            p.logprobs_out[static_cast<std::size_t>(column) * p.ranks + j] = s.picked_value[j] - normalizer;
        }
        if (p.tokens != nullptr) { p.tokens[column] = p.row_ids[s.picked[0]]; }
    }
}

// ---- The logits form's selection (one CTA per column) and q8 rescore (a warp per listed row)

struct ListParams {
    const float2* lists;
    std::int32_t grid;
    float margin;
    std::int32_t cap;
    std::int32_t* entries; // [T][cap]
    std::int32_t* counts;  // [T]
};

__global__ void __launch_bounds__(kSelectThreads) screened_list_kernel(ListParams p) {
    __shared__ SelectShared s;
    const int tid = static_cast<int>(threadIdx.x), column = static_cast<int>(blockIdx.x);
    pdl::enter();
    const int n = select_rows(p.lists + static_cast<std::size_t>(column) * p.grid * kListLength, p.grid, p.margin, 1,
                              p.cap, s);
    __syncthreads();
    // Every slot holds a row (the tail repeats the first), so a gathered tile reads only valid rows.
    for (int i = tid; i < p.cap; i += kSelectThreads) {
        p.entries[static_cast<std::size_t>(column) * p.cap + i] = s.rows[i < n ? i : 0];
    }
    if (tid == 0) { p.counts[column] = n; }
}

struct Q8RescoreParams {
    const bf16* x;
    const std::int8_t* codes;
    const __half* scales;
    std::int32_t padded_k;
    std::int32_t vocab_rows;
    std::int32_t columns;
    std::int32_t cap;
    const std::int32_t* entries;
    const std::int32_t* counts;
    float* logits;
};

__global__ void __launch_bounds__(kSelectThreads) q8_rescore_kernel(Q8RescoreParams p) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int e    = static_cast<int>(blockIdx.x) * kSelectWarps + (static_cast<int>(threadIdx.x) >> 5);
    pdl::enter();
    const int column = e / p.cap, i = e % p.cap;
    if (column >= p.columns || i >= __ldcg(p.counts + column)) { return; }
    const int row = __ldcg(p.entries + e);
    const float v = q8_row(p.codes, p.scales, row, p.padded_k, p.x + static_cast<std::size_t>(column) * kHidden, lane);
    if (lane == 0) { p.logits[static_cast<std::size_t>(column) * p.vocab_rows + row] = v; }
}

// ---- Host side

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("screened_head: " + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

void validate_policy(const ScreenedHeadPolicy& policy) {
    require(std::isfinite(policy.margin) && policy.margin >= 0.0F, "the margin must be finite and non-negative");
    require(policy.rescored >= 32 && policy.rescored <= kMaxRescored && policy.rescored % 32 == 0,
            "rescored rows must be a multiple of 32 in [32, 256]");
}

void validate_x(const Tensor& x) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[0] == kHidden && x.ne[2] == 1 &&
                x.ne[3] == 1 && x.ne[1] >= 1 && x.ne[1] <= kScreenedHeadMaxColumns && aligned16(x.data),
            "x must be contiguous, aligned BF16 [2560, T] with T in [1, 16]");
}

void validate_screen(const Weight& screen) {
    require(screen.qtype == QType::Q2_G64_FP16 && screen.layout == QuantLayout::RowSplit && screen.k == kHidden &&
                screen.padded_shape[1] == kHidden && screen.n > 0 && screen.n % kItemRows == 0 &&
                screen.qdata != nullptr && screen.scales != nullptr && aligned16(screen.qdata) &&
                aligned16(screen.scales),
            "the screen must be a row-split q2_g64_fp16 [R, 2560] with R a multiple of 16");
}

// One CTA per SM, at most kMaxGrid: fixed per device, so a column's lists do not depend on the call.
int screen_grid(int items) {
    static std::array<std::atomic<int>, 64> grids{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    const int slot = device >= 0 && device < 64 ? device : 0;
    int grid       = grids[slot].load(std::memory_order_acquire);
    if (grid == 0) {
        int sms = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        grid = std::clamp(sms, 1, kMaxGrid);
        grids[slot].store(grid, std::memory_order_release);
    }
    return std::min(grid, items);
}

template <int Tiles, bool Logits, bool Softmax>
void launch_screen(const ScreenParams& p, int grid, cudaStream_t stream) {
    constexpr auto kernel = screen_kernel<Tiles, Logits, Softmax>;
    constexpr int bytes   = screen_shared_bytes<Tiles>();
    static const cudaError_t configured =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    CUDA_CHECK(configured);
    CUDA_CHECK(pdl::launch_consumer({dim3(grid), dim3(kScreenThreads), static_cast<std::size_t>(bytes), stream},
                                    kernel, p));
}

template <bool Logits, bool Softmax>
void run_screen(const ScreenParams& p, int grid, cudaStream_t stream) {
    if (p.columns <= 8) {
        launch_screen<1, Logits, Softmax>(p, grid, stream);
    } else {
        launch_screen<2, Logits, Softmax>(p, grid, stream);
    }
}

// The workspace recipe, shared by the query and the calls.
struct Scratch {
    Tensor lists, partials, entries, counts;
};

template <class Allocator>
Scratch allocate_scratch(Allocator& allocator, std::int32_t columns, std::int32_t cap) {
    Scratch s;
    s.lists    = allocator.alloc(DType::FP32, {2 * kListLength, kMaxGrid, columns}, 256);
    s.partials = allocator.alloc(DType::FP32, {2, kMaxGrid, columns}, 256);
    s.entries  = allocator.alloc(DType::I32, {cap, columns}, 256);
    s.counts   = allocator.alloc(DType::I32, {columns}, 256);
    return s;
}

ScreenParams screen_params(const Tensor& x, const Weight& screen, std::int32_t valid_rows, const Scratch& scratch) {
    return {.x          = static_cast<const bf16*>(x.data),
            .codes      = static_cast<const std::uint8_t*>(screen.qdata),
            .scales     = static_cast<const std::uint8_t*>(screen.scales),
            .rows       = screen.n,
            .valid_rows = valid_rows,
            .columns    = x.ne[1],
            .logits     = nullptr,
            .lists      = static_cast<float2*>(scratch.lists.data),
            .partials   = static_cast<float2*>(scratch.partials.data)};
}

} // namespace

std::size_t screened_head_workspace_bytes(std::int32_t head_rows, std::int32_t max_columns,
                                          const ScreenedHeadPolicy& policy) {
    validate_policy(policy);
    if (head_rows <= 0 || max_columns < 1 || max_columns > kScreenedHeadMaxColumns) {
        throw std::invalid_argument("screened_head workspace: invalid rows or column interval");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, max_columns, policy.rescored);
    return layout.peak_bytes();
}

void screened_head_top(const Tensor& x, const Weight& head, const Weight& screen, std::int32_t valid_rows,
                       const ScreenedHeadPolicy& policy, Tensor& rows, Tensor& logprobs, const Tensor& row_ids,
                       Tensor& tokens, WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);
    validate_x(x);
    validate_screen(screen);
    const std::int32_t T = x.ne[1], R = screen.n;
    require(head.qtype == QType::Q4_G64_FP16 && head.layout == QuantLayout::RowSplit && head.n == R &&
                head.k == kHidden && head.padded_shape[1] % 64 == 0 && head.padded_shape[1] >= kHidden &&
                head.qhigh == nullptr && head.qdata != nullptr && head.scales != nullptr && aligned16(head.qdata),
            "the head must be a row-split q4_g64_fp16 [R, 2560] matching the screen");
    require(valid_rows >= 1 && valid_rows <= R, "valid_rows must be in [1, R]");
    const std::int32_t ranks = rows.ne[0];
    require(ranks >= 1 && ranks <= kScreenedHeadMaxRanks && rows.data != nullptr && rows.dtype == DType::I32 &&
                rows.is_contiguous() && rows.ne[1] == T && logprobs.data != nullptr && logprobs.dtype == DType::FP32 &&
                logprobs.is_contiguous() && logprobs.ne[0] == ranks && logprobs.ne[1] == T,
            "rows and logprobs must be contiguous I32 / FP32 [ranks, T] with ranks in [1, 4]");
    require(ranks <= policy.rescored, "the policy must rescore at least the ranks");
    const bool mapped = row_ids.data != nullptr;
    require(mapped == (tokens.data != nullptr), "row_ids and tokens are both given or both empty");
    if (mapped) {
        require(row_ids.dtype == DType::I32 && row_ids.numel() == R && tokens.dtype == DType::I32 && tokens.numel() == T,
                "row_ids must be I32 [R] and tokens I32 [T]");
    }
    auto scope          = workspace.scope();
    const Scratch scratch = allocate_scratch(workspace, T, policy.rescored);
    const int grid      = screen_grid(R / kItemRows);
    run_screen<false, true>(screen_params(x, screen, valid_rows, scratch), grid, stream);
    const TopParams top{.x            = static_cast<const bf16*>(x.data),
                        .codes        = static_cast<const std::uint8_t*>(head.qdata),
                        .scales       = static_cast<const __half*>(head.scales),
                        .padded_k     = head.padded_shape[1],
                        .lists        = static_cast<const float2*>(scratch.lists.data),
                        .partials     = static_cast<const float2*>(scratch.partials.data),
                        .grid         = grid,
                        .columns      = T,
                        .margin       = policy.margin,
                        .cap          = policy.rescored,
                        .ranks        = ranks,
                        .rows_out     = static_cast<std::int32_t*>(rows.data),
                        .logprobs_out = static_cast<float*>(logprobs.data),
                        .row_ids      = mapped ? static_cast<const std::int32_t*>(row_ids.data) : nullptr,
                        .tokens       = mapped ? static_cast<std::int32_t*>(tokens.data) : nullptr};
    CUDA_CHECK(pdl::launch_consumer({dim3(T), dim3(kSelectThreads), 0, stream}, screened_top_kernel, top));
}

void screened_head_logits(const Tensor& x, const Tensor& bf16_head, const Weight& head, const Weight& screen,
                          std::int32_t valid_rows, const ScreenedHeadPolicy& policy, Tensor& logits,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);
    validate_x(x);
    validate_screen(screen);
    const std::int32_t T = x.ne[1], R = screen.n;
    const bool rows_bf16 = bf16_head.data != nullptr;
    if (rows_bf16) {
        require(head.qdata == nullptr && bf16_head.dtype == DType::BF16 && bf16_head.is_contiguous() &&
                    bf16_head.ne[0] == kHidden && bf16_head.ne[1] == R && bf16_head.ne[2] == 1 && aligned16(bf16_head.data),
                "a BF16 head must be contiguous, aligned rows [R, 2560] (a BF16 tensor [2560, R]) matching the screen");
    } else {
        require(head.qtype == QType::Q8_G32_FP16 && head.layout == QuantLayout::RowSplit && head.n == R &&
                    head.k == kHidden && head.padded_shape[1] % 32 == 0 && head.padded_shape[1] >= kHidden &&
                    head.qhigh == nullptr && head.qdata != nullptr && head.scales != nullptr && aligned16(head.qdata),
                "the head must be BF16 rows or a row-split q8_g32_fp16 [R, 2560] matching the screen");
    }
    require(valid_rows >= 1 && valid_rows <= R, "valid_rows must be in [1, R]");
    require(logits.data != nullptr && logits.dtype == DType::FP32 && logits.is_contiguous() && logits.ne[0] == R &&
                logits.ne[1] == T && logits.ne[2] == 1 && logits.ne[3] == 1,
            "logits must be contiguous FP32 [R, T]");
    auto scope            = workspace.scope();
    const Scratch scratch = allocate_scratch(workspace, T, policy.rescored);
    const int grid        = screen_grid(R / kItemRows);
    ScreenParams params   = screen_params(x, screen, valid_rows, scratch);
    params.logits         = static_cast<float*>(logits.data);
    params.partials       = nullptr;
    run_screen<true, false>(params, grid, stream);
    const ListParams list{.lists   = static_cast<const float2*>(scratch.lists.data),
                          .grid    = grid,
                          .margin  = policy.margin,
                          .cap     = policy.rescored,
                          .entries = static_cast<std::int32_t*>(scratch.entries.data),
                          .counts  = static_cast<std::int32_t*>(scratch.counts.data)};
    CUDA_CHECK(pdl::launch_consumer({dim3(T), dim3(kSelectThreads), 0, stream}, screened_list_kernel, list));
    if (rows_bf16) {
        detail::screened_head::bf16_rescore(x, bf16_head.data, R, T, list.entries, list.counts, policy.rescored,
                                            static_cast<float*>(logits.data), stream);
    } else {
        const Q8RescoreParams q8{.x          = static_cast<const bf16*>(x.data),
                                 .codes      = static_cast<const std::int8_t*>(head.qdata),
                                 .scales     = static_cast<const __half*>(head.scales),
                                 .padded_k   = head.padded_shape[1],
                                 .vocab_rows = R,
                                 .columns    = T,
                                 .cap        = policy.rescored,
                                 .entries    = list.entries,
                                 .counts     = list.counts,
                                 .logits     = static_cast<float*>(logits.data)};
        const int warps = T * policy.rescored;
        CUDA_CHECK(pdl::launch_consumer({dim3((warps + kSelectWarps - 1) / kSelectWarps), dim3(kSelectThreads), 0, stream},
                                        q8_rescore_kernel, q8));
    }
}

} // namespace infernix::ops
