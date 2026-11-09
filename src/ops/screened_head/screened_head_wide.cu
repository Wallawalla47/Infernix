// The 5,120-wide screened head (include/infernix/ops/screened_head.h, screened_head_topk and screened_head_fit):
// the top sixteen rows of a vocabulary head per column, every row scored from a 2-bit copy fitted at load and the
// rows near each column's best rescored exactly (candidates RV-41: the Qwen3.8-27B DFlash2 drafter's proposal head).
//
// Cost model (RTX 5090): the screen streams 1,440 bytes per row (2.25 bits a weight) against the FP8 row head's
// 5,122 (248,320 rows: 357.6 MB against 1.27 GB) at the DRAM roofline; the rescore reads at most `rescored` rows a
// column.
//
// Kernels, in launch order:
//   screen    persistent CTAs (one per SM), eight warps splitting K per 16-row item (warp w takes groups
//             [10w, 10w + 10)), a three-stage cp.async ring of items (23.3 KB each). Unlike the 2,560-wide screen,
//             whose columns sit in shared memory, each thread holds its K slice of the int8 columns in registers
//             (80 registers at 16 columns): 16 columns of 5,120 bytes would leave no room for the ring in 99 KB.
//             Per column it keeps the CTA's best kListLength rows.
//   topk      one CTA per column: the rows to rescore (select_rows), their exact scores (FP32 sums of the decoded
//             head row times the column, a warp per row), and the sixteen best by (score, lower row).
// The fit (screened_head_fit): one thread per 64-weight group of a decoded head row: the binary16 scale of least
// squared error over a sweep of the group's max-abs times [0.3, 1] in both signs (and the least-squares scale of the
// max-abs codes), its codes in {-2, -1, 0, 1}, then the scale stretched to keep the group's norm (the converter's
// grouped_mse_norm, tools/convert/quantization/groupwise.py, with a coarser candidate set).

#include "infernix/ops/screened_head.h"
#include "ops/screened_head/screened_head_launch.h"
#include "ops/screened_head/screened_head_select.cuh"

#include "core/device.h"
#include "core/layout.h"
#include "core/pdl.cuh"
#include "ops/common/bytes.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using namespace detail::screened_head;
using bf16 = __nv_bfloat16;

constexpr int kK               = kScreenedTopkHidden; // 5,120
constexpr int kKGroups         = kK / 64;             // 80 weight groups of 64 per row
constexpr int kWarps           = 8;                   // K split of an item
constexpr int kWarpGroups      = kKGroups / kWarps;   // 10
constexpr int kThreads         = kWarps * 32;
constexpr int kItemRows        = 16;                  // one m16 row tile
constexpr int kStages          = 3;
constexpr int kCodeRowBytes    = kK / 4;              // 1,280: four 2-bit codes per byte
constexpr int kCodePitch       = kCodeRowBytes + 16;  // rows g and g + 8 of a quad's word on distinct banks
constexpr int kScaleRowBytes   = kKGroups * 2;        // 160
constexpr int kStageCodeBytes  = kItemRows * kCodePitch;
constexpr int kStageBytes      = kStageCodeBytes + kItemRows * kScaleRowBytes;
constexpr int kScreenCodeBytes = kCodeRowBytes;       // a screen row's code bytes
static_assert(kKGroups % kWarps == 0 && kStageBytes % 16 == 0 && kStageCodeBytes % 16 == 0);

// Dynamic shared memory of the screen: the item ring, each (column, group)'s FP32 scale and twice its code sum, and
// the eight warps' partial sums (then the item's scores).
template <int Tiles>
constexpr int wide_screen_shared_bytes() {
    constexpr int C = Tiles * 8;
    return kStages * kStageBytes + C * kKGroups * 8 + kWarps * kItemRows * C * 4;
}
static_assert(wide_screen_shared_bytes<2>() <= 101376, "the screen's ring, scales and partials fit one CTA");

struct WideScreenParams {
    const bf16* x;              // [K, T]
    const std::uint8_t* codes;  // [R, 1,280]
    const std::uint8_t* scales; // [R, 80] binary16
    std::int32_t rows;          // R, a multiple of 16
    std::int32_t valid_rows;
    std::int32_t columns;       // T
    float2* lists;              // [T][grid][kListLength]: (score, row bits)
};

// Thread (warp w, quad g, lane t) holds, for its tile nt's column c = 8 nt + g and each of its groups
// j (group 10 w + j), chunk t of the group's int8 copy in the MMA's K order: byte 4f + b holds the value 4b + f of
// the chunk (the 2,560-wide screen's quantize_columns layout). The quad's four chunks are the whole group, so its
// max-abs and code sum reduce over the quad; lane t = 0 publishes the group's (scale, twice the code sum).
template <int Tiles>
__device__ __forceinline__ void quantize_registers(const WideScreenParams& p, int warp, int g, int t,
                                                   uint4 (&xr)[kWarpGroups][Tiles], float2* xs) {
#pragma unroll
    for (int nt = 0; nt < Tiles; ++nt) {
        const int column = nt * 8 + g;
#pragma unroll
        for (int j = 0; j < kWarpGroups; ++j) {
            const int group = warp * kWarpGroups + j;
            float v[16];
            if (column < p.columns) {
                const auto* source = reinterpret_cast<const uint4*>(p.x + static_cast<std::size_t>(column) * kK +
                                                                    group * 64 + t * 16);
                const uint4 a = __ldcg(source), b = __ldcg(source + 1);
                const bf16* lo = reinterpret_cast<const bf16*>(&a);
                const bf16* hi = reinterpret_cast<const bf16*>(&b);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v[i]     = __bfloat162float(lo[i]);
                    v[8 + i] = __bfloat162float(hi[i]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) { v[i] = 0.0f; }
            }
            float amax = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; ++i) { amax = fmaxf(amax, fabsf(v[i])); }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            const float inverse = amax > 0.0f ? 127.0f / amax : 0.0f;
            int q[16];
            int sum = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                q[i] = max(-127, min(127, __float2int_rn(v[i] * inverse)));
                sum += q[i];
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            const auto word = [&](int f) {
                return (static_cast<std::uint32_t>(q[f]) & 0xFFu) | ((static_cast<std::uint32_t>(q[4 + f]) & 0xFFu) << 8) |
                       ((static_cast<std::uint32_t>(q[8 + f]) & 0xFFu) << 16) |
                       ((static_cast<std::uint32_t>(q[12 + f]) & 0xFFu) << 24);
            };
            xr[j][nt] = make_uint4(word(0), word(1), word(2), word(3));
            if (t == 0) { xs[column * kKGroups + group] = make_float2(amax / 127.0f, static_cast<float>(2 * sum)); }
        }
    }
}

// One persistent CTA per SM over 16-row items i = blockIdx.x + k * gridDim.x. A column's lists depend on that column
// alone (its own int8 copy, a fixed grid).
template <int Tiles>
__global__ void __launch_bounds__(kThreads, 1) wide_screen_kernel(WideScreenParams p) {
    extern __shared__ __align__(16) std::uint8_t shared[];
    constexpr int C      = Tiles * 8;
    std::uint8_t* stages = shared;
    float2* xs           = reinterpret_cast<float2*>(shared + kStages * kStageBytes);
    float* partial       = reinterpret_cast<float*>(xs + C * kKGroups);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int items = p.rows / kItemRows;
    const int first = static_cast<int>(blockIdx.x), stride = static_cast<int>(gridDim.x);
    const int mine  = first < items ? (items - first + stride - 1) / stride : 0;

    // Item k of this CTA into stage k % kStages: 16 contiguous code rows (into padded rows) and their scales.
    const auto issue = [&](int k) {
        const std::size_t item = static_cast<std::size_t>(first) + static_cast<std::size_t>(k) * stride;
        std::uint8_t* stage    = stages + (k % kStages) * kStageBytes;
        const std::uint8_t* code_source  = p.codes + item * (kItemRows * kCodeRowBytes);
        const std::uint8_t* scale_source = p.scales + item * (kItemRows * kScaleRowBytes);
        for (int c = tid; c < kItemRows * kCodeRowBytes / 16; c += kThreads) {
            const int row = c / (kCodeRowBytes / 16), part = c % (kCodeRowBytes / 16);
            cp_async<16, Cache::cg>(stage + row * kCodePitch + part * 16, code_source + c * 16);
        }
        for (int c = tid; c < kItemRows * kScaleRowBytes / 16; c += kThreads) {
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
    uint4 xr[kWarpGroups][Tiles];
    quantize_registers<Tiles>(p, warp, g, t, xr, xs);

    // Thread c < T keeps column c's best rows (descending).
    float best[kListLength];
    int best_row[kListLength];
#pragma unroll
    for (int j = 0; j < kListLength; ++j) {
        best[j]     = negative_infinity();
        best_row[j] = INT_MAX;
    }

    for (int k = 0; k < mine; ++k) {
        cp_wait<kStages - 2>();
        __syncthreads(); // item k landed (and, at k = 0, every column scale); item k - 1's stage and scores are done
        if (k + kStages - 1 < mine) { issue(k + kStages - 1); }
        cp_commit();
        const std::uint8_t* stage = stages + (k % kStages) * kStageBytes;
        const auto* scales        = reinterpret_cast<const __half*>(stage + kStageCodeBytes);
        float acc[Tiles][4];
#pragma unroll
        for (int nt = 0; nt < Tiles; ++nt) { acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f; }
#pragma unroll
        for (int j = 0; j < kWarpGroups; ++j) {
            const int group = warp * kWarpGroups + j;
            // Word t of the group in rows g and g + 8, each 2-bit code moved to its level c + 2 in [0, 3].
            const std::uint32_t w0 =
                *reinterpret_cast<const std::uint32_t*>(stage + g * kCodePitch + group * 16 + t * 4) ^ 0xAAAAAAAAu;
            const std::uint32_t w1 =
                *reinterpret_cast<const std::uint32_t*>(stage + (g + 8) * kCodePitch + group * 16 + t * 4) ^
                0xAAAAAAAAu;
            const float s0 = __half2float(scales[g * kKGroups + group]);
            const float s1 = __half2float(scales[(g + 8) * kKGroups + group]);
            constexpr std::uint32_t kLevels = 0x03030303u;
#pragma unroll
            for (int nt = 0; nt < Tiles; ++nt) {
                const uint4 b = xr[j][nt];
                int c0, c1, c2, c3;
                mma_s8s8_k32_zero(c0, c1, c2, c3, w0 & kLevels, w1 & kLevels, (w0 >> 2) & kLevels, (w1 >> 2) & kLevels,
                                  b.x, b.y);
                mma_s8(c0, c1, c2, c3, (w0 >> 4) & kLevels, (w1 >> 4) & kLevels, (w0 >> 6) & kLevels,
                       (w1 >> 6) & kLevels, b.z, b.w);
                // sum c x = sum (c + 2) x - 2 sum x, all exact in int32 and float.
                const float2 x0 = xs[(nt * 8 + 2 * t) * kKGroups + group];
                const float2 x1 = xs[(nt * 8 + 2 * t + 1) * kKGroups + group];
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
        for (int e = tid; e < kItemRows * C; e += kThreads) {
            float v = partial[e];
#pragma unroll
            for (int w = 1; w < kWarps; ++w) { v += partial[w * kItemRows * C + e]; }
            partial[e] = v;
        }
        __syncthreads();
        if (tid < p.columns) {
#pragma unroll 4
            for (int r = 0; r < kItemRows; ++r) {
                const int row = row0 + r;
                if (row < p.valid_rows) { list_insert(best, best_row, partial[r * C + tid], row); }
            }
        }
    }
    pdl::trigger_dependents();
    if (tid < p.columns) {
        float2* list = p.lists + (static_cast<std::size_t>(tid) * gridDim.x + blockIdx.x) * kListLength;
#pragma unroll
        for (int j = 0; j < kListLength; ++j) { list[j] = make_float2(best[j], __int_as_float(best_row[j])); }
    }
}

// ---- The head's rows, decoded

enum class HeadRows : std::int32_t { Fp8Bf16, Fp8F32, Q8 };

struct HeadView {
    const std::uint8_t* codes;
    const void* scales;
    HeadRows kind;
    std::int32_t pitch; // a row's code bytes (q8: its padded K)
};

__device__ __forceinline__ float fp8_value(std::uint8_t code) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), __NV_E4M3)));
}

__device__ __forceinline__ float row_scale(const HeadView& h, std::int64_t row) {
    return h.kind == HeadRows::Fp8Bf16 ? __bfloat162float(static_cast<const bf16*>(h.scales)[row])
                                       : static_cast<const float*>(h.scales)[row];
}

// Eight consecutive weights of `row` from element k (a multiple of 8): FP8 codes times the row scale, or q8 codes
// times their group scale.
__device__ __forceinline__ void decode8(const HeadView& h, std::int64_t row, int k, float (&w)[8]) {
    const uint2 packed = __ldg(reinterpret_cast<const uint2*>(h.codes + row * h.pitch + k));
    if (h.kind == HeadRows::Q8) {
        const float scale = __half2float(static_cast<const __half*>(h.scales)[row * (h.pitch / 32) + k / 32]);
#pragma unroll
        for (int j = 0; j < 8; ++j) { w[j] = static_cast<float>(signed_byte_of(packed, j)) * scale; }
    } else {
        const float scale = row_scale(h, row);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            w[j] = fp8_value(static_cast<std::uint8_t>((j < 4 ? packed.x : packed.y) >> (8 * (j & 3)))) * scale;
        }
    }
}

// The exact score of `row` for the column in `x` (shared memory): lane l sums chunks l, l + 32, ... of eight weights
// in order with fmaf, then the xor butterfly.
__device__ __forceinline__ float exact_row(const HeadView& h, std::int64_t row, const bf16* x, int lane) {
    float sum = 0.0f;
#pragma unroll 4
    for (int i = 0; i < kK / 8 / 32; ++i) {
        const int chunk = lane + 32 * i;
        float w[8];
        decode8(h, row, chunk * 8, w);
        const uint4 xp = *reinterpret_cast<const uint4*>(x + chunk * 8);
        const bf16* xv = reinterpret_cast<const bf16*>(&xp);
#pragma unroll
        for (int j = 0; j < 8; ++j) { sum = fmaf(w[j], __bfloat162float(xv[j]), sum); }
    }
    return warp_butterfly(sum);
}

// ---- Selection, rescore and the sixteen (one CTA per column)

struct TopkParams {
    const bf16* x;
    HeadView head;
    const float2* lists;
    std::int32_t grid;
    float margin;
    std::int32_t cap;
    std::int32_t* ids;   // [16, T]
    float* scores;       // [16, T]
};

struct TopkShared {
    SelectShared select;
    __align__(16) bf16 x[kK];
    float exact[kMaxRescored];
    int taken[kMaxRescored];
};

__global__ void __launch_bounds__(kSelectThreads) wide_topk_kernel(TopkParams p) {
    __shared__ TopkShared s;
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, warp = tid >> 5;
    const int column = static_cast<int>(blockIdx.x);
    pdl::enter();
    for (int i = tid; i < kK / 8; i += kSelectThreads) {
        reinterpret_cast<uint4*>(s.x)[i] =
            __ldcg(reinterpret_cast<const uint4*>(p.x + static_cast<std::size_t>(column) * kK) + i);
    }
    const int n = select_rows(p.lists + static_cast<std::size_t>(column) * p.grid * kListLength, p.grid, p.margin,
                              kScreenedTopkCandidates, p.cap, s.select);
    __syncthreads();
    for (int e = warp; e < n; e += kSelectWarps) {
        const float v = exact_row(p.head, s.select.rows[e], s.x, lane);
        if (lane == 0) {
            s.exact[e] = v;
            s.taken[e] = 0;
        }
    }
    __syncthreads();
    if (warp == 0) {
        // The sixteen among the rescored rows: the larger score, on ties the lower row; a column with fewer rescored
        // rows (fewer valid rows than sixteen) repeats its last with a score of minus infinity.
        int last = s.select.rows[0];
        for (int j = 0; j < kScreenedTopkCandidates; ++j) {
            float v = negative_infinity();
            int row = INT_MAX, at = -1;
            for (int e = lane; e < n; e += 32) {
                if (!s.taken[e] && ranks_above(s.exact[e], s.select.rows[e], v, row)) {
                    v   = s.exact[e];
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
            if (at >= 0) { last = row; }
            if (lane == 0) {
                p.ids[static_cast<std::size_t>(column) * kScreenedTopkCandidates + j]    = at >= 0 ? row : last;
                p.scores[static_cast<std::size_t>(column) * kScreenedTopkCandidates + j] = v;
                if (at >= 0) { s.taken[at] = 1; }
            }
            __syncwarp();
        }
    }
}

// ---- The fit

struct FitParams {
    HeadView head;
    std::int32_t rows;
    std::uint8_t* codes; // [R, 1,280]
    __half* scales;      // [R, 80]
};

constexpr int kSweep = 29; // the group's max-abs times 0.300, 0.325, ..., 1.000

// Every step of the fit is an explicitly rounded FP32 operation (no contraction), so its bytes are reproducible: the
// op test replays it on the host.
//
// Codes clamp(rint(w * fl32(1 / s)), -2, 1), as the converter encodes them.
__device__ __forceinline__ int q2_code(float w, float reciprocal) {
    return max(-2, min(1, __float2int_rn(__fmul_rn(w, reciprocal))));
}

__device__ __forceinline__ float q2_reciprocal(float scale) { return scale != 0.0f ? __fdiv_rn(1.0f, scale) : 0.0f; }

// sum over the group of (w - c s)^2 in index order.
__device__ __forceinline__ float q2_error(const float (&w)[64], float scale) {
    const float reciprocal = q2_reciprocal(scale);
    float sse              = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        const float e = __fmaf_rn(-static_cast<float>(q2_code(w[i], reciprocal)), scale, w[i]);
        sse           = __fmaf_rn(e, e, sse);
    }
    return sse;
}

__global__ void __launch_bounds__(128) wide_fit_kernel(FitParams p) {
    const std::int64_t task = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (task >= static_cast<std::int64_t>(p.rows) * kKGroups) { return; }
    const std::int64_t row = task / kKGroups;
    const int group        = static_cast<int>(task % kKGroups);
    float w[64];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        float v[8];
        decode8(p.head, row, group * 64 + c * 8, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { w[c * 8 + j] = v[j]; }
    }
    float amax = 0.0f, energy = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
        amax   = fmaxf(amax, fabsf(w[i]));
        energy = __fmaf_rn(w[i], w[i], energy);
    }
    float scale = 0.0f;
    if (amax > 0.0f) {
        // The sweep in both signs (a negative scale puts the group's positive side on -2), then the least-squares
        // scale of the max-abs codes; the first least error wins.
        float best_error    = INFINITY;
        const auto consider = [&](float candidate) {
            const float rounded = __half2float(__float2half_rn(candidate));
            if (rounded == 0.0f || !isfinite(rounded)) { return; }
            const float error = q2_error(w, rounded);
            if (error < best_error) {
                best_error = error;
                scale      = rounded;
            }
        };
        for (int sign = 1; sign >= -1; sign -= 2) {
            for (int i = 0; i < kSweep; ++i) {
                const float factor = __fadd_rn(0.300f, __fmul_rn(0.025f, static_cast<float>(i)));
                consider(__fmul_rn(static_cast<float>(sign) * amax, factor));
            }
        }
        const float canonical = __half2float(__float2half_rn(amax));
        if (canonical > 0.0f && isfinite(canonical)) {
            const float base = __fdiv_rn(1.0f, canonical);
            float overlap = 0.0f, norm = 0.0f;
#pragma unroll
            for (int i = 0; i < 64; ++i) {
                const float c = static_cast<float>(q2_code(w[i], base));
                overlap       = __fmaf_rn(w[i], c, overlap);
                norm          = __fmaf_rn(c, c, norm);
            }
            if (norm > 0.0f) { consider(__fdiv_rn(overlap, norm)); }
        }
    }
    // The codes at the chosen scale, and the scale stretched by sqrt(|w|^2 / <w, w_hat>) (w_hat = codes * scale).
    const float reciprocal = q2_reciprocal(scale);
    std::uint8_t* out = p.codes + row * kScreenCodeBytes + group * 16;
    float overlap     = 0.0f;
#pragma unroll
    for (int b = 0; b < 16; ++b) {
        std::uint32_t byte = 0;
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            const int c = q2_code(w[4 * b + f], reciprocal);
            overlap     = __fmaf_rn(w[4 * b + f], __fmul_rn(static_cast<float>(c), scale), overlap);
            byte |= (static_cast<std::uint32_t>(c) & 3u) << (2 * f);
        }
        out[b] = static_cast<std::uint8_t>(byte);
    }
    float stretched = scale;
    if (overlap > 0.0f) {
        const float candidate = __fmul_rn(scale, __fsqrt_rn(__fdiv_rn(energy, overlap)));
        if (isfinite(__half2float(__float2half_rn(candidate)))) { stretched = candidate; }
    }
    p.scales[row * kKGroups + group] = __float2half_rn(stretched);
}

// ---- Host side

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("screened_head (5,120-wide): " + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

HeadView head_view(const Weight& head) {
    if ((head.qtype == QType::FP8_E4M3FN_ROW_BF16 || head.qtype == QType::FP8_E4M3FN_ROW_F32) &&
        head.layout == QuantLayout::RowScale && head.k == kK && head.padded_shape[1] == kK && head.qdata != nullptr &&
        head.scales != nullptr && aligned16(head.qdata)) {
        return {static_cast<const std::uint8_t*>(head.qdata), head.scales,
                head.qtype == QType::FP8_E4M3FN_ROW_BF16 ? HeadRows::Fp8Bf16 : HeadRows::Fp8F32, kK};
    }
    require(head.qtype == QType::Q8_G32_FP16 && head.layout == QuantLayout::RowSplit && head.k == kK &&
                head.padded_shape[1] % 32 == 0 && head.padded_shape[1] >= kK && head.qhigh == nullptr &&
                head.qdata != nullptr && head.scales != nullptr && aligned16(head.qdata),
            "the head must be an FP8 row-scaled or row-split q8_g32_fp16 [R, 5120]");
    return {static_cast<const std::uint8_t*>(head.qdata), head.scales, HeadRows::Q8, head.padded_shape[1]};
}

void validate_wide_screen(const Weight& screen, std::int32_t rows) {
    require(screen.qtype == QType::Q2_G64_FP16 && screen.layout == QuantLayout::RowSplit && screen.k == kK &&
                screen.padded_shape[1] == kK && screen.n == rows && rows > 0 && rows % kItemRows == 0 &&
                screen.qdata != nullptr && screen.scales != nullptr && aligned16(screen.qdata) &&
                aligned16(screen.scales),
            "the screen must be a row-split q2_g64_fp16 [R, 5120] of the head's rows, R a multiple of 16");
}

// One CTA per SM, at most kMaxGrid: fixed per device, so a column's lists do not depend on the call.
int wide_screen_grid(int items) {
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

template <int Tiles>
void launch_wide_screen(const WideScreenParams& p, int grid, cudaStream_t stream) {
    constexpr auto kernel = wide_screen_kernel<Tiles>;
    constexpr int bytes   = wide_screen_shared_bytes<Tiles>();
    static const cudaError_t configured =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    CUDA_CHECK(configured);
    CUDA_CHECK(pdl::launch_consumer({dim3(grid), dim3(kThreads), static_cast<std::size_t>(bytes), stream}, kernel, p));
}

template <class Allocator>
Tensor allocate_lists(Allocator& allocator, std::int32_t columns) {
    return allocator.alloc(DType::FP32, {2 * kListLength, kMaxGrid, columns}, 256);
}

void validate_topk_policy(const ScreenedHeadPolicy& policy) {
    require(std::isfinite(policy.margin) && policy.margin >= 0.0F, "the margin must be finite and non-negative");
    require(policy.rescored >= 32 && policy.rescored <= kMaxRescored && policy.rescored % 32 == 0,
            "rescored rows must be a multiple of 32 in [32, 256]");
}

} // namespace

std::size_t screened_head_wide_screen_bytes(std::int32_t rows) {
    if (rows <= 0 || rows % kItemRows != 0) {
        throw std::invalid_argument("screened_head_wide_screen_bytes: rows must be a positive multiple of 16");
    }
    return align256(static_cast<std::size_t>(rows) * kScreenCodeBytes) +
           align256(static_cast<std::size_t>(rows) * kScaleRowBytes);
}

Weight screened_head_wide_screen(void* storage, std::int32_t rows) {
    const std::size_t code_bytes = align256(static_cast<std::size_t>(rows) * kScreenCodeBytes);
    if (storage == nullptr || !aligned16(storage) || rows <= 0 || rows % kItemRows != 0) {
        throw std::invalid_argument("screened_head_wide_screen: aligned storage and rows a positive multiple of 16");
    }
    Weight w;
    w.payload         = storage;
    w.payload_bytes   = screened_head_wide_screen_bytes(rows);
    w.qtype           = QType::Q2_G64_FP16;
    w.group_size      = 64;
    w.ndim            = 2;
    w.shape[0]        = rows;
    w.shape[1]        = kK;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = kK;
    w.qdata           = storage;
    w.scales          = static_cast<const std::uint8_t*>(storage) + code_bytes;
    w.n               = rows;
    w.k               = kK;
    w.group           = 64;
    w.layout          = QuantLayout::RowSplit;
    w.scale_dtype     = DType::FP16;
    return w;
}

void screened_head_fit(const Weight& head, const Weight& screen, cudaStream_t stream) {
    const HeadView view = head_view(head);
    validate_wide_screen(screen, head.n);
    const FitParams p{.head   = view,
                      .rows   = head.n,
                      .codes  = static_cast<std::uint8_t*>(const_cast<void*>(screen.qdata)),
                      .scales = static_cast<__half*>(const_cast<void*>(screen.scales))};
    const std::int64_t tasks = static_cast<std::int64_t>(head.n) * kKGroups;
    wide_fit_kernel<<<static_cast<unsigned>((tasks + 127) / 128), 128, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

std::size_t screened_head_topk_workspace_bytes(std::int32_t max_columns, const ScreenedHeadPolicy& policy) {
    validate_topk_policy(policy);
    if (max_columns < 1 || max_columns > kScreenedHeadMaxColumns) {
        throw std::invalid_argument("screened_head_topk workspace: columns must be in [1, 16]");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_lists(layout, max_columns);
    return layout.peak_bytes();
}

void screened_head_topk(const Tensor& x, const Weight& head, const Weight& screen, std::int32_t valid_rows,
                        const ScreenedHeadPolicy& policy, Tensor& candidate_ids, Tensor& candidate_scores,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    validate_topk_policy(policy);
    const HeadView view = head_view(head);
    const std::int32_t R = head.n;
    validate_wide_screen(screen, R);
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[0] == kK && x.ne[2] == 1 &&
                x.ne[3] == 1 && x.ne[1] >= 1 && x.ne[1] <= kScreenedHeadMaxColumns && aligned16(x.data),
            "x must be contiguous, aligned BF16 [5120, T] with T in [1, 16]");
    const std::int32_t T = x.ne[1];
    require(valid_rows >= kScreenedTopkCandidates && valid_rows <= R, "valid_rows must be in [16, R]");
    require(candidate_ids.data != nullptr && candidate_ids.dtype == DType::I32 && candidate_ids.is_contiguous() &&
                candidate_ids.ne[0] == kScreenedTopkCandidates && candidate_ids.numel() == 16LL * T &&
                candidate_scores.data != nullptr && candidate_scores.dtype == DType::FP32 &&
                candidate_scores.is_contiguous() && candidate_scores.ne[0] == kScreenedTopkCandidates &&
                candidate_scores.numel() == 16LL * T,
            "candidate_ids and candidate_scores must be contiguous I32 / FP32 [16, T]");
    require(kScreenedTopkCandidates <= policy.rescored, "the policy must rescore at least sixteen rows");
    auto scope        = workspace.scope();
    const Tensor lists = allocate_lists(workspace, T);
    const int grid     = wide_screen_grid(R / kItemRows);
    const WideScreenParams params{.x          = static_cast<const bf16*>(x.data),
                                  .codes      = static_cast<const std::uint8_t*>(screen.qdata),
                                  .scales     = static_cast<const std::uint8_t*>(screen.scales),
                                  .rows       = R,
                                  .valid_rows = valid_rows,
                                  .columns    = T,
                                  .lists      = static_cast<float2*>(lists.data)};
    if (T <= 8) {
        launch_wide_screen<1>(params, grid, stream);
    } else {
        launch_wide_screen<2>(params, grid, stream);
    }
    const TopkParams top{.x      = static_cast<const bf16*>(x.data),
                         .head   = view,
                         .lists  = static_cast<const float2*>(lists.data),
                         .grid   = grid,
                         .margin = policy.margin,
                         .cap    = policy.rescored,
                         .ids    = static_cast<std::int32_t*>(candidate_ids.data),
                         .scores = static_cast<float*>(candidate_scores.data)};
    CUDA_CHECK(pdl::launch_consumer({dim3(T), dim3(kSelectThreads), 0, stream}, wide_topk_kernel, top));
}

} // namespace infernix::ops
