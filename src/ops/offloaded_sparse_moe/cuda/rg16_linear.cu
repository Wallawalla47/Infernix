// The canonical W4A16 dense projection of an nvfp4_rg16_v1 matrix (include/infernix/ops/rg16_linear.h):
// Qwen3.5-122B-A10B's output head [248320, 3072], 429 MB read once per call.
//
// Through eight columns a dp4a GEMV: x is encoded once per call (canon::a16 encoding, per column),
// every CTA keeps the encoded columns in shared memory, and each warp owns one 16-row group at a
// time, streaming its 192 units (evict-first loads) with the narrow expert route's three dp4a per
// block and an exact int64 row sum. Beyond eight columns the exact wide expert route's grouped
// integer tensor-core kernel (wide_expert_a16.cuh) runs on the matrix as one job, its tile columns
// the call's columns. Both form the same exact S, so outputs depend on neither the route nor T.

#include "infernix/ops/rg16_linear.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

namespace moe  = offloaded_moe;
namespace wide = offloaded_moe::wide;

// The registered matrix: Qwen3.5-122B-A10B's head.
using Geometry          = moe::Qwen3_5_122BExperts; // K = H = 3072
constexpr int kRows     = 248320;
constexpr int kK        = Geometry::kHidden;
constexpr int kBlocks   = kK / 16; // 192 units per row group
constexpr int kGroups   = kRows / 16;
constexpr int kNarrow   = moe::kMaxColumns; // the GEMV's widest call
constexpr int kWarps    = 16;
constexpr int kThreads  = kWarps * 32;
static_assert(kRows % 128 == 0, "the wide route's row blocks are 128 rows");

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("rg16_linear: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("rg16_linear ") + what + ": " + cudaGetErrorString(error));
    }
}

// An encoded A16 block as the dp4a loop reads it: elements 4q..4q+3's bytes in word q (the narrow
// expert route's A16Act).
struct Act {
    std::uint32_t lo[4];
    std::uint32_t mid[4];
    std::int32_t hi[4];
};

// Signed bytes of a times unsigned bytes of b, plus c (PTX dp4a.s32.u32).
__device__ __forceinline__ int dp4a_su(int a, std::uint32_t b, int c) {
    int d;
    asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// One CTA per column, one 16-element block per thread: the column exponent, then every block's
// encoding into acts[column][block].
__global__ void __launch_bounds__(kBlocks) encode_kernel(const std::uint16_t* __restrict__ x, Act* __restrict__ acts,
                                                          std::int32_t* __restrict__ emax) {
    __shared__ unsigned keys[kBlocks / 32];
    const int t = static_cast<int>(blockIdx.x), b = static_cast<int>(threadIdx.x);
    const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * kK + 16 * b);
    const uint4 v0 = source[0], v1 = source[1];
    const unsigned words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    unsigned key = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) { key = max(key, max(words[w] & 0x7FFFU, (words[w] >> 16) & 0x7FFFU)); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (b % 32 == 0) { keys[b / 32] = key; }
    __syncthreads();
    unsigned all = 0;
#pragma unroll
    for (int w = 0; w < kBlocks / 32; ++w) { all = max(all, keys[w]); }
    const int e = all >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(all >> 7), 1);
    Act out{};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const auto h  = static_cast<std::uint16_t>(j % 2 == 0 ? words[j / 2] & 0xFFFFU : words[j / 2] >> 16);
        const auto xv = static_cast<std::uint32_t>(e == canon::kA16NonFinite ? 0 : canon::a16_value(h, e));
        const int q = j / 4, s = 8 * (j % 4);
        out.lo[q] |= (xv & 0xFFU) << s;
        out.mid[q] |= ((xv >> 8) & 0xFFU) << s;
        out.hi[q] = static_cast<std::int32_t>(static_cast<std::uint32_t>(out.hi[q]) | (((xv >> 16) & 0xFFU) << s));
    }
    acts[static_cast<std::size_t>(t) * kBlocks + b] = out;
    if (b == 0) { emax[t] = e; }
}

// The GEMV of Columns-wide calls (n <= Columns columns). Lane l = 16 half + r holds row r of the
// warp's row group at k = 8 half .. 8 half + 7 of each block; the halves' products are joined
// before the scale, as the narrow expert route does.
template <int Columns, bool F32>
__global__ void __launch_bounds__(kThreads) narrow_kernel(const Act* __restrict__ acts,
                                                          const std::int32_t* __restrict__ emax,
                                                          const std::uint8_t* __restrict__ matrix, float multiplier,
                                                          int n, void* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* shared = reinterpret_cast<Act*>(smem_raw);
    for (int i = static_cast<int>(threadIdx.x); i < n * kBlocks; i += kThreads) { shared[i] = acts[i]; }
    __syncthreads();
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int r = lane & 15, half = lane >> 4;
    const int shift = r < 8 ? 0 : 4;
    const int code_offset  = 64 * half + 4 * (r & 7);
    const int scale_offset = 128 + 4 * (r >> 2), scale_shift = 8 * (r & 3);
    int e[Columns];
#pragma unroll
    for (int c = 0; c < Columns; ++c) { e[c] = c < n ? emax[c] : 0; }
    for (int g = static_cast<int>(blockIdx.x) * kWarps + warp; g < kGroups; g += static_cast<int>(gridDim.x) * kWarps) {
        const std::uint8_t* units = matrix + static_cast<std::size_t>(g) * kBlocks * moe::kUnitBytes;
        long long s[Columns];
#pragma unroll
        for (int c = 0; c < Columns; ++c) { s[c] = 0; }
#pragma unroll 4
        for (int b = 0; b < kBlocks; ++b) {
            const std::uint8_t* unit = units + b * static_cast<int>(moe::kUnitBytes);
            const unsigned c0 = __ldcs(reinterpret_cast<const unsigned*>(unit + code_offset));
            const unsigned c1 = __ldcs(reinterpret_cast<const unsigned*>(unit + code_offset + 32));
            const unsigned sc = __ldcs(reinterpret_cast<const unsigned*>(unit + scale_offset));
            const int w0 = static_cast<int>(canon::e2m1_x2_quad((c0 >> shift) & 0x0F0F0F0FU));
            const int w1 = static_cast<int>(canon::e2m1_x2_quad((c1 >> shift) & 0x0F0F0F0FU));
            const int sw = canon::e4m3_scaled((sc >> scale_shift) & 0xFFU);
#pragma unroll
            for (int c = 0; c < Columns; ++c) {
                if (c < n) { // uniform across the warp
                    const Act& a  = shared[c * kBlocks + b];
                    const int lo  = dp4a_su(w1, a.lo[2 * half + 1], dp4a_su(w0, a.lo[2 * half], 0));
                    const int mid = dp4a_su(w1, a.mid[2 * half + 1], dp4a_su(w0, a.mid[2 * half], 0));
                    const int hi  = __dp4a(w1, a.hi[2 * half + 1], __dp4a(w0, a.hi[2 * half], 0));
                    unsigned p = static_cast<unsigned>(lo) + (static_cast<unsigned>(mid) << 8) + (static_cast<unsigned>(hi) << 16);
                    p += static_cast<unsigned>(__shfl_xor_sync(0xFFFFFFFFU, static_cast<int>(p), 16));
                    s[c] += static_cast<long long>(static_cast<int>(p)) * sw;
                }
            }
        }
        if (lane < 16) {
            const std::size_t row = static_cast<std::size_t>(g) * 16 + r;
#pragma unroll
            for (int c = 0; c < Columns; ++c) {
                if (c >= n) { continue; }
                const std::size_t at = static_cast<std::size_t>(c) * kRows + row;
                if constexpr (F32) {
                    static_cast<float*>(out)[at] = canon::a16_row_output_f32(s[c], e[c], multiplier);
                } else {
                    static_cast<std::uint16_t*>(out)[at] = canon::a16_row_output(s[c], e[c], multiplier);
                }
            }
        }
    }
}

// The wide route's matrix: the head as one job's gate/up-like matrix, its rows by token.
struct HeadMatrix {
    static constexpr int kRowGroups      = kGroups;
    static constexpr int kBlocks         = kK / 16;
    static constexpr std::size_t kOffset = 0;
    static constexpr int kRowBytes       = wide::A16Planes<Geometry>::kXRowBytes;
    static constexpr bool kTokenRows     = true;
};

// The canonical output of each tile column into its logits column (rows 2i, 2i + 1 in one thread).
template <bool F32>
struct HeadEpilogue {
    const std::int32_t* x_emax;
    void* out;
    float multiplier;

    __device__ __forceinline__ void operator()(wide::SumsA16& sums, const wide::TileContext& ctx) const {
        using S = wide::ScheduleA16;
#pragma unroll
        for (int m = 0; m < S::kMmaTokens; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * S::kWarpTokens + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                const int t    = ctx.first + column; // entries are the identity
                const int emax = x_emax[t];
                const std::size_t base = static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(ctx.row_block) * S::kBlockRows;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::size_t row = base + wide::a16_tile_row(ctx.warp_n, r, ctx.tig);
                    if constexpr (F32) {
                        *reinterpret_cast<float2*>(static_cast<float*>(out) + row) =
                            make_float2(canon::a16_row_output_f32(sums[m][r][2 * half], emax, multiplier),
                                        canon::a16_row_output_f32(sums[m][r][2 * half + 1], emax, multiplier));
                    } else {
                        const std::uint32_t y0 = canon::a16_row_output(sums[m][r][2 * half], emax, multiplier);
                        const std::uint32_t y1 = canon::a16_row_output(sums[m][r][2 * half + 1], emax, multiplier);
                        *reinterpret_cast<std::uint32_t*>(static_cast<std::uint16_t*>(out) + row) = y0 | (y1 << 16);
                    }
                }
            }
        }
    }
};

// The wide route's bookkeeping in the workspace: the matrix as the one job of a dispatch whose
// entries are the call's columns, and its 64-column tiles in one pass.
struct WidePlan {
    std::int32_t* jobs;      // [1]
    std::int32_t* job_count; // [1]
    std::int32_t* offsets;   // [2]
    std::int32_t* entries;   // [T]
    std::int32_t* tiles;     // [2 tiles]
    std::int32_t* pass_tiles; // [2]
    const std::uint8_t** job_records; // [1]
    std::uint8_t* x_limbs;   // [T][kRowBytes]
    std::int32_t* x_emax;    // [T]
};

__global__ void plan_kernel(WidePlan plan, const std::uint8_t* matrix, int columns, int tiles) {
    for (int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); i < columns;
         i += static_cast<int>(gridDim.x * blockDim.x)) {
        plan.entries[i] = i;
        if (i < tiles) {
            plan.tiles[2 * i]     = 0;
            plan.tiles[2 * i + 1] = i * wide::kTileColumns;
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        plan.jobs[0]        = 0;
        *plan.job_count     = 1;
        plan.offsets[0]     = 0;
        plan.offsets[1]     = columns;
        plan.pass_tiles[0]  = 0;
        plan.pass_tiles[1]  = tiles;
        plan.job_records[0] = matrix;
    }
}

std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

struct NarrowPlan {
    Act* acts;
    std::int32_t* emax;
};

template <class Arena>
NarrowPlan carve_narrow(Arena& arena, std::int32_t columns) {
    return {static_cast<Act*>(arena.alloc_bytes(round256(sizeof(Act) * kBlocks * columns), 256).data),
            static_cast<std::int32_t*>(arena.alloc_bytes(round256(sizeof(std::int32_t) * columns), 256).data)};
}

template <class Arena>
WidePlan carve_wide(Arena& arena, std::int32_t columns) {
    const auto t     = static_cast<std::size_t>(columns);
    const auto tiles = static_cast<std::size_t>((columns + wide::kTileColumns - 1) / wide::kTileColumns);
    WidePlan plan{};
    const auto i32 = [&](std::size_t count) {
        return static_cast<std::int32_t*>(arena.alloc_bytes(round256(count * sizeof(std::int32_t)), 256).data);
    };
    plan.jobs        = i32(1);
    plan.job_count   = i32(1);
    plan.offsets     = i32(2);
    plan.entries     = i32(t);
    plan.tiles       = i32(2 * tiles);
    plan.pass_tiles  = i32(2);
    plan.job_records = static_cast<const std::uint8_t**>(arena.alloc_bytes(256, 256).data);
    plan.x_limbs     = static_cast<std::uint8_t*>(arena.alloc_bytes(round256(t * HeadMatrix::kRowBytes), 256).data);
    plan.x_emax      = i32(t);
    return plan;
}

void require_operands(const Tensor& x, const Weight& weight, const Tensor& out) {
    require(weight.qtype == QType::NVFP4_MUL && weight.layout == QuantLayout::Rg16 && weight.n == kRows &&
                weight.k == kK && weight.qdata != nullptr && reinterpret_cast<std::uintptr_t>(weight.qdata) % 16 == 0 &&
                std::isfinite(weight.weight_multiplier),
            "weight must be an nvfp4_rg16_v1 [248320, 3072] matrix with a finite multiplier");
    require(x.dtype == DType::BF16 && x.is_contiguous() && x.ne[0] == kK && x.ne[1] > 0 && x.ne[2] == 1 &&
                x.ne[3] == 1 && reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0,
            "x must be contiguous, 16-byte aligned BF16 [3072, T]");
    require((out.dtype == DType::FP32 || out.dtype == DType::BF16) && out.is_contiguous() && out.ne[0] == kRows &&
                out.ne[1] == x.ne[1] && out.ne[2] == 1 && out.ne[3] == 1 &&
                reinterpret_cast<std::uintptr_t>(out.data) % 16 == 0,
            "out must be contiguous FP32 or BF16 [248320, T]");
}

template <int Columns, bool F32>
void launch_narrow(const NarrowPlan& plan, const Weight& weight, int n, void* out, cudaStream_t stream) {
    constexpr int bytes = static_cast<int>(sizeof(Act)) * kBlocks * Columns;
    static_assert(bytes <= 99 * 1024);
    static const int ctas = [] {
        CUDA_CHECK(cudaFuncSetAttribute(narrow_kernel<Columns, F32>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int device = 0, sms = 0, per_sm = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, narrow_kernel<Columns, F32>, kThreads, bytes));
        // A persistent grid: every resident CTA copies the encoded columns once.
        return std::min((kGroups + kWarps - 1) / kWarps, sms * std::max(per_sm, 1));
    }();
    narrow_kernel<Columns, F32><<<ctas, kThreads, bytes, stream>>>(plan.acts, plan.emax,
                                                                  static_cast<const std::uint8_t*>(weight.qdata),
                                                                  weight.weight_multiplier, n, out);
    check_launch("narrow");
}

template <bool F32>
void dispatch_narrow(const NarrowPlan& plan, const Weight& weight, int n, void* out, cudaStream_t stream) {
    if (n == 1) return launch_narrow<1, F32>(plan, weight, n, out, stream);
    if (n == 2) return launch_narrow<2, F32>(plan, weight, n, out, stream);
    if (n <= 4) return launch_narrow<4, F32>(plan, weight, n, out, stream);
    launch_narrow<kNarrow, F32>(plan, weight, n, out, stream);
}

} // namespace

std::size_t rg16_linear_workspace_bytes(std::int32_t rows, std::int32_t input_rows, std::int32_t columns) {
    require(rows == kRows && input_rows == kK && columns > 0, "unregistered shape or empty call");
    WorkspaceLayoutBuilder layout;
    if (columns <= kNarrow) {
        (void)carve_narrow(layout, columns);
    } else {
        (void)carve_wide(layout, columns);
    }
    return layout.peak_bytes(1);
}

void rg16_linear(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    require_operands(x, weight, out);
    const int columns = x.ne[1];
    const bool f32    = out.dtype == DType::FP32;
    auto scope        = workspace.scope();
    if (columns <= kNarrow) {
        const NarrowPlan plan = carve_narrow(workspace, columns);
        encode_kernel<<<columns, kBlocks, 0, stream>>>(static_cast<const std::uint16_t*>(x.data), plan.acts, plan.emax);
        check_launch("encode");
        if (f32) {
            dispatch_narrow<true>(plan, weight, columns, out.data, stream);
        } else {
            dispatch_narrow<false>(plan, weight, columns, out.data, stream);
        }
        return;
    }
    const WidePlan plan = carve_wide(workspace, columns);
    const int tiles     = (columns + wide::kTileColumns - 1) / wide::kTileColumns;
    plan_kernel<<<(columns + 255) / 256, 256, 0, stream>>>(plan, static_cast<const std::uint8_t*>(weight.qdata), columns,
                                                         tiles);
    check_launch("plan");
    wide::encode_x_limbs(moe::kShapeOf<Geometry>, x, plan.x_limbs, plan.x_emax, stream);
    wide::Call call;
    call.dispatch.jobs      = plan.jobs;
    call.dispatch.job_count = plan.job_count;
    call.dispatch.offsets   = plan.offsets;
    call.dispatch.entries   = plan.entries;
    call.shape              = moe::kShapeOf<Geometry>;
    call.job_records        = plan.job_records;
    call.tiles              = plan.tiles;
    call.pass_tiles         = plan.pass_tiles;
    call.passes             = 1;
    call.a16                = true;
    call.x_limbs            = plan.x_limbs;
    call.x_emax             = plan.x_emax;
    call.top_k              = 1;
    if (f32) {
        wide::launch_a16<HeadMatrix>(call, 0, HeadEpilogue<true>{plan.x_emax, out.data, weight.weight_multiplier}, stream);
    } else {
        wide::launch_a16<HeadMatrix>(call, 0, HeadEpilogue<false>{plan.x_emax, out.data, weight.weight_multiplier}, stream);
    }
}

} // namespace infernix::ops
