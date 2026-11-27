// Inkling 2's routed experts on the prompt route (include/infernix/ops/w4a4_sparse_moe.h moe_experts_prompt): the NVFP4
// layers as offloaded_sparse_moe's grouped block-scaled GEMM (ops/offloaded_sparse_moe/cuda/wide_expert.cuh, at
// Inkling's expert geometry) with this Op's own epilogues; the MXFP8 layers as this file's grouped block-scaled FP8
// GEMM (below).
//
//   plan       one CTA: each job's 64-entry tiles and its record
//   gather x   every dispatched entry's A4(x) row (the MoE mixer's four-over-six codes and scales of its column) in the
//              GEMM's plane layout, in dispatch order
//   gate/up    grouped GEMM; epilogue g, u = bf16(acc * rs), rs the narrow route's row scale, h = the softcapped SwiGLU
//              times gamma into the entry's BF16 h row
//   h          a warp per entry: the row's amax, its four-over-six blocks (canon::four_over_six_block, the narrow
//              route's bits) into the h plane, the amax for the down epilogue
//   down       grouped GEMM; epilogue y = bf16(acc * rs_h) into the entry's output row
//
// Every output depends on its own entry's rows and the expert's weights alone (an MMA's output element reads only its
// row and column, in one fixed K order), so a prompt column's outputs do not depend on the call's other columns.

#include "infernix/ops/w4a4_sparse_moe.h"

#include "core/device.h"
#include "core/tensor.h"
#include "core/tma_descriptor_staging.cuh"
#include "ops/ggml/ggml_linear_a8.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.cuh"
#include "ops/offloaded_sparse_moe/cuda/wide_expert_a8.cuh"
#include "ops/offloaded_sparse_moe/nvfp4_a8.h"
#include "ops/w4a4_sparse_moe/expert_math.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops::w4a4_moe {
namespace {

namespace wide = ::infernix::ops::offloaded_moe::wide;
using G        = ::infernix::ops::offloaded_moe::ExpertGeometry<2304, 1152>;

constexpr int kHidden       = 2304;
constexpr int kInter        = 1152;
constexpr int kXCodeBytes   = kHidden / 2;  // 1,152
constexpr int kXScaleBytes  = kHidden / 16; //   144
constexpr int kHCodeBytes   = kInter / 2;   //   576
constexpr int kHScaleMap    = 80;           // 72 scale bytes; the last 16-byte box reads 8 more inside the row
constexpr int kRowXBytes    = kHidden / 2 + kHidden / 16; // the MoE mixer's natural x row (codes, then scales)
constexpr int kTile         = wide::kTileColumns;         // 64
constexpr int kPlanThreads  = 1024;
constexpr int kGatherThreads = 256;
constexpr int kHWarps       = 8;
static_assert(kPromptXRowBytes == kXCodeBytes + kXScaleBytes && kPromptHRowBytes == kHCodeBytes + kHScaleMap);
static_assert(kPromptXRowBytes % 16 == 0 && kPromptHRowBytes % 16 == 0);

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("w4a4_sparse_moe prompt ") + what + ": " + cudaGetErrorString(error));
    }
}

// Each job's 64-entry tiles (job, first entry within the job) in job order, every job's record, and the one pass's
// tile range.
__global__ void __launch_bounds__(kPlanThreads)
    plan_kernel(Workspace ws, const std::uint8_t* records, std::uint64_t record_stride, int experts) {
    __shared__ int scan[kPlanThreads];
    const int tid  = static_cast<int>(threadIdx.x);
    const int jobs = *ws.job_count;
    int n          = 0;
    if (tid < jobs) {
        const int expert = ws.jobs[tid];
        n                = (ws.offsets[expert + 1] - ws.offsets[expert] + kTile - 1) / kTile;
        ws.job_records[tid] = records + static_cast<std::size_t>(expert) * record_stride;
    }
    scan[tid] = n;
    __syncthreads();
    for (int step = 1; step < kPlanThreads; step <<= 1) {
        const int add = tid >= step ? scan[tid - step] : 0;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    const int first = scan[tid] - n;
    for (int i = 0; i < n; ++i) {
        ws.tiles[2 * (first + i)]     = tid;
        ws.tiles[2 * (first + i) + 1] = i * kTile;
    }
    if (tid == kPlanThreads - 1) {
        ws.pass_tiles[0] = 0;
        ws.pass_tiles[1] = scan[tid];
    }
    (void)experts;
}

// The 16 E2M1 codes of a block (element order) as the plane's 8 bytes: byte j (j < 4) holds elements j and 4 + j,
// byte 4 + j elements 8 + j and 12 + j (wide_expert.cu quantize_kernel's packing).
__device__ __forceinline__ uint2 plane_codes(const std::uint8_t (&c)[16]) {
    unsigned low = 0, high = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        low |= (static_cast<unsigned>(c[j]) | (static_cast<unsigned>(c[4 + j]) << 4)) << (8 * j);
        high |= (static_cast<unsigned>(c[8 + j]) | (static_cast<unsigned>(c[12 + j]) << 4)) << (8 * j);
    }
    return make_uint2(low, high);
}

// Entry p's A4(x) row from its column's natural row (byte j of a block: elements 2j low, 2j + 1 high).
__global__ void __launch_bounds__(kGatherThreads)
    gather_x_kernel(const std::uint8_t* __restrict__ rows, Workspace ws, int entries) {
    constexpr int kBlocks = kHidden / 16;
    for (int task = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); task < entries * kBlocks;
         task += static_cast<int>(gridDim.x * blockDim.x)) {
        const int p = task / kBlocks, b = task % kBlocks;
        const int column = ws.entries[p] / kTopK;
        const std::uint8_t* src = rows + static_cast<std::size_t>(column) * kRowXBytes;
        const uint2 natural = *reinterpret_cast<const uint2*>(src + 8 * b);
        const unsigned words[2] = {natural.x, natural.y};
        std::uint8_t c[16];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned byte = (words[j / 4] >> (8 * (j % 4))) & 0xFFU;
            c[2 * j]     = static_cast<std::uint8_t>(byte & 0xFU);
            c[2 * j + 1] = static_cast<std::uint8_t>(byte >> 4);
        }
        std::uint8_t* dst = ws.x_plane + static_cast<std::size_t>(p) * kPromptXRowBytes;
        const int position = wide::plane_block(b);
        *reinterpret_cast<uint2*>(dst + 8 * position) = plane_codes(c);
        dst[kXCodeBytes + position]                    = src[kXCodeBytes + b];
    }
}

// g, u = bf16(acc * rs) (rs = (amax_x * amax_w) * rcp), h = swiglu(g, u) * gamma into the entry's BF16 h row.
struct GateUpEpilogue {
    const std::int32_t* entries;
    const float* x_amax;
    const float* gamma;
    const float* w_amax;
    const float* denominators;
    float gate_cap, up_cap;
    std::uint16_t* h;

    template <int T>
    __device__ __forceinline__ void operator()(float (&acc)[T][wide::Schedule::kMmaRows][4],
                                               const wide::TileContext& ctx) const {
        constexpr int kRows = wide::Schedule::kMmaRows;
        const float wa      = w_amax[2 * ctx.expert];
        const float rcp     = detail::nvfp4_rcp(denominators[0]);
#pragma unroll
        for (int m = 0; m < T; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * 16 * T + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                const int entry = entries[ctx.first + column];
                const float rs  = detail::nvfp4_row_scale(x_amax[entry / kTopK], wa, rcp);
                const float g   = gamma[entry];
                std::uint16_t* row = h + static_cast<std::size_t>(entry) * kInter;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    const int i = (ctx.row_block * wide::Schedule::kBlockRows + wide::tile_row(ctx.warp_n, r, ctx.tig)) / 2;
                    const std::uint16_t gate = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half], rs));
                    const std::uint16_t up   = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half + 1], rs));
                    row[i] = detail::swiglu(gate, up, g, gate_cap, up_cap);
                }
            }
        }
    }
};

// y = bf16(acc * rs_h) (rs_h = (amax_h * amax_w) * rcp) into the entry's output row.
struct DownEpilogue {
    const std::int32_t* entries;
    const std::uint32_t* h_amax;
    const float* w_amax;
    const float* denominators;
    std::uint16_t* outputs;

    template <int T>
    __device__ __forceinline__ void operator()(float (&acc)[T][wide::Schedule::kMmaRows][4],
                                               const wide::TileContext& ctx) const {
        constexpr int kRows = wide::Schedule::kMmaRows;
        const float wa      = w_amax[2 * ctx.expert + 1];
        const float rcp     = detail::nvfp4_rcp(denominators[1]);
#pragma unroll
        for (int m = 0; m < T; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = ctx.warp_m * 16 * T + m * 16 + ctx.group + 8 * half;
                if (column >= ctx.columns) { continue; }
                const int entry = entries[ctx.first + column];
                const float rs  = detail::nvfp4_row_scale(canon::f32_from_bits(h_amax[entry]), wa, rcp);
                std::uint16_t* out = outputs + static_cast<std::size_t>(entry) * kHidden +
                                     ctx.row_block * wide::Schedule::kBlockRows;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    const std::uint32_t y0 = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half], rs));
                    const std::uint32_t y1 = canon::f32_to_bf16_rn(canon::mul_rn(acc[m][r][2 * half + 1], rs));
                    *reinterpret_cast<std::uint32_t*>(out + wide::tile_row(ctx.warp_n, r, ctx.tig)) = y0 | (y1 << 16);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------- A8 (--a8-activations), NVFP4 layers

// wide_expert_a8.cuh's grouped 8-bit GEMM over Inkling's matrices (the header's A8 order, so a column's bits are the
// narrow route's): x as the columns' A8 rows (Activations::rows, by token), h as A8 rows in dispatch order (the
// workspace's h_a8, encoded from the BF16 h plane in dispatch order).
template <wide::Matrix M>
struct A8Shape {
    static constexpr int kRowGroups      = M == wide::Matrix::GateUp ? G::kGateUpRowGroups : G::kDownRowGroups;
    static constexpr int kBlocks         = M == wide::Matrix::GateUp ? G::kGateUpBlocks : G::kDownBlocks;
    static constexpr std::size_t kOffset = M == wide::Matrix::GateUp ? 0 : G::kGateUpBytes;
    static constexpr int kK              = M == wide::Matrix::GateUp ? kHidden : kInter;
    static constexpr int kRowBytes       = M == wide::Matrix::GateUp ? kA8XRowBytes : kA8HRowBytes;
    static constexpr bool kTokenRows     = M == wide::Matrix::GateUp;
    static constexpr bool kSharedJobs    = false;
    static_assert(!kSharedJobs, "Inkling's routed calls carry no shared-expert jobs");
    static_assert(kRowGroups % wide::ScheduleA8Tall::kRowGroups == 0 && kBlocks % wide::ScheduleA8Tall::kUnitsPerKTile == 0,
                  "whole 256-row blocks and 128-element stages");
    static_assert(kRowBytes == kK + kK / 8, "a row holds its codes, then its slice scales");
};

// g, u = nvfp4_a8::row_output(acc, m) with m = fl32(amax_w / denominator), h = the softcapped SwiGLU times gamma into
// the BF16 h plane row of the tile column's dispatch position.
struct GateUpA8Epilogue {
    const std::int32_t* entries;
    const float* gamma;
    const float* w_amax;
    const float* denominators;
    float gate_cap, up_cap;
    std::uint16_t* h;

    __device__ __forceinline__ void operator()(wide::AccumulatorsA8& acc, const wide::TileContext& ctx) const {
        using S       = wide::ScheduleA8Tall;
        const float m = canon::div_rn(w_amax[2 * ctx.expert], denominators[0]);
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = wide::a8_tile_column(ctx, mt, half);
                if (column >= ctx.columns) { continue; }
                const int p      = ctx.first + column;
                const float gm   = gamma[entries[p]];
                std::uint16_t* hp = h + static_cast<std::size_t>(p) * kInter;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint16_t gate = offloaded_moe::nvfp4_a8::row_output(acc[mt][r][2 * half], m);
                    const std::uint16_t up   = offloaded_moe::nvfp4_a8::row_output(acc[mt][r][2 * half + 1], m);
                    hp[(ctx.row_block * S::kBlockRows + 16 * ctx.warp_n + 8 * r + 2 * ctx.tig) / 2] =
                        detail::swiglu(gate, up, gm, gate_cap, up_cap);
                }
            }
        }
    }
};

// y = nvfp4_a8::row_output(acc, m_down) into the entry's output row.
struct DownA8Epilogue {
    const std::int32_t* entries;
    const float* w_amax;
    const float* denominators;
    std::uint16_t* outputs;

    __device__ __forceinline__ void operator()(wide::AccumulatorsA8& acc, const wide::TileContext& ctx) const {
        using S       = wide::ScheduleA8Tall;
        const float m = canon::div_rn(w_amax[2 * ctx.expert + 1], denominators[1]);
#pragma unroll
        for (int mt = 0; mt < S::kMmaTokens; ++mt) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int column = wide::a8_tile_column(ctx, mt, half);
                if (column >= ctx.columns) { continue; }
                std::uint16_t* out = outputs + static_cast<std::size_t>(entries[ctx.first + column]) * kHidden +
                                     ctx.row_block * S::kBlockRows + 16 * ctx.warp_n;
#pragma unroll
                for (int r = 0; r < S::kMmaRows; ++r) {
                    const std::uint32_t y0 = offloaded_moe::nvfp4_a8::row_output(acc[mt][r][2 * half], m);
                    const std::uint32_t y1 = offloaded_moe::nvfp4_a8::row_output(acc[mt][r][2 * half + 1], m);
                    *reinterpret_cast<std::uint32_t*>(out + 8 * r + 2 * ctx.tig) = y0 | (y1 << 16);
                }
            }
        }
    }
};

// A warp per dispatched entry: its BF16 h row's amax and four-over-six blocks into the h plane (blocks l, l + 32,
// l + 64 on lane l), the amax's bits into h_amax for the down epilogue.
__global__ void __launch_bounds__(kHWarps * 32) quantize_h_kernel(Workspace ws, int entries) {
    constexpr int kBlocks = kInter / 16; // 72
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int p    = static_cast<int>(blockIdx.x) * kHWarps + static_cast<int>(threadIdx.x) / 32;
    if (p >= entries) { return; }
    const int entry = ws.entries[p];
    const std::uint16_t* row = ws.h + static_cast<std::size_t>(entry) * kInter;
    float x[3][16];
    std::uint32_t mag = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int b = lane + 32 * i;
        if (b >= kBlocks) { continue; }
        const uint4 q0 = *reinterpret_cast<const uint4*>(row + 16 * b);
        const uint4 q1 = *(reinterpret_cast<const uint4*>(row + 16 * b) + 1);
        const std::uint32_t w[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[i][2 * j]     = canon::f32_from_bits(w[j] << 16);
            x[i][2 * j + 1] = canon::f32_from_bits(w[j] & 0xFFFF0000U);
        }
#pragma unroll
        for (int j = 0; j < 16; ++j) { mag = max(mag, canon::f32_bits(x[i][j]) & 0x7FFFFFFFU); }
    }
#pragma unroll
    for (int o = 16; o >= 1; o /= 2) { mag = max(mag, __shfl_xor_sync(0xFFFFFFFFU, mag, o)); }
    const canon::FourOverSixRow r = canon::four_over_six_row(canon::f32_from_bits(mag));
    std::uint8_t* dst = ws.h_plane + static_cast<std::size_t>(p) * kPromptHRowBytes;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int b = lane + 32 * i;
        if (b >= kBlocks) { continue; }
        const canon::FourOverSixBlock q = canon::four_over_six_block(x[i], r);
        const int position = wide::plane_block(b);
        *reinterpret_cast<uint2*>(dst + 8 * position) = plane_codes(q.code);
        dst[kHCodeBytes + position]                    = q.scale_word;
    }
    if (lane == 0) { ws.h_amax[entry] = mag; }
}

// ---------------------------------------------------------------------------- MXFP8 (layers 20-23)
//
// A grouped GEMM over the same tiles on block-scaled FP8 tensor cores (mma m16n8k32 kind::mxf8f6f4, UE8M0 per 32):
// a CTA of 8 warps takes 128 rows (one 16-row group a warp, the record's units as A fragments as the narrow kernels
// read them) by a tile's 64 entries (B from shared memory: the entries' E4M3 codes and UE8M0 words, 4 blocks a stage),
// accumulating each output in FP32 over K's 32-blocks in ascending order. Gate/up: g, u = bf16(acc), h = the softcapped
// SwiGLU times gamma into the entry's BF16 h row; then each h row's MXFP8 groups (canon::mxfp8_group, a warp an entry)
// into the x plane's row of its position (free in an MXFP8 layer); down: y = bf16(acc) into the entry's output row.

constexpr int kA8Unit       = 528; // 512 code bytes in A-fragment order, 16 UE8M0 row words
constexpr int kA8Warps      = 8;
constexpr int kA8Threads    = kA8Warps * 32;
constexpr int kA8Blocks     = 4;   // K blocks of 32 a stage
constexpr int kA8RowStride  = 32 * kA8Blocks + 16; // a staged entry's codes, padded (conflict-free B reads)
constexpr int kA8XRowBytes  = kHidden + kHidden / 32; // the mixer's MXFP8 x row: codes then UE8M0 words (2,376)
constexpr int kA8HScales    = kInter;                // an h plane row: 1,152 codes then 36 words
constexpr int kA8RowBlocks  = 2304 / 128;            // 18 for gate/up (2 I rows) and for down (H rows)
constexpr std::uint64_t kA8GateUpBytes = std::uint64_t{2 * kInter / 16} * (kHidden / 32) * kA8Unit;

struct A8Smem {
    std::uint8_t codes[2][kTile][kA8RowStride];
    std::uint8_t scales[2][kTile][kA8Blocks];
};

struct A8Call {
    Workspace ws;
    const std::uint8_t* records;
    std::uint64_t record_stride;
    const std::uint8_t* x_rows; // gate/up: the columns' MXFP8 rows
    const float* gamma;
    float gate_cap, up_cap;
    std::uint16_t* outputs;
};

template <bool Down>
__global__ void __launch_bounds__(kA8Threads, 2) a8_grouped_kernel(const A8Call c) {
    __shared__ __align__(16) A8Smem sm;
    constexpr int kBlocks   = Down ? kInter / 32 : kHidden / 32; // K blocks: 36 or 72
    constexpr int kStages   = kBlocks / kA8Blocks;              // 9 or 18
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int work = c.ws.pass_tiles[1] * kA8RowBlocks;
    for (int w = static_cast<int>(blockIdx.x); w < work; w += static_cast<int>(gridDim.x)) {
        const int tile = w / kA8RowBlocks, rb = w % kA8RowBlocks;
        const int job = c.ws.tiles[2 * tile], column = c.ws.tiles[2 * tile + 1];
        const int expert = c.ws.jobs[job];
        const int first  = c.ws.offsets[expert] + column;
        const int count  = min(kTile, c.ws.offsets[expert + 1] - first);
        const int rg     = rb * kA8Warps + warp; // this warp's row group
        const std::uint8_t* units = c.records + static_cast<std::size_t>(expert) * c.record_stride +
                                    (Down ? kA8GateUpBytes : 0) + static_cast<std::size_t>(rg) * kBlocks * kA8Unit;
        // The stage's B: each entry's codes and words of K blocks 4s .. 4s + 3 (past the tile's entries: its last).
        const auto stage_b = [&](int buf, int s) {
            for (int i = tid; i < kTile * (kA8Blocks * 32 / 8); i += kA8Threads) {
                const int e = i / (kA8Blocks * 4), part = i % (kA8Blocks * 4);
                const int p = first + min(e, count - 1);
                const std::uint8_t* row = Down ? c.ws.x_plane + static_cast<std::size_t>(p) * kPromptXRowBytes
                                               : c.x_rows + static_cast<std::size_t>(c.ws.entries[p] / kTopK) * kA8XRowBytes;
                ops::cp_async<8>(&sm.codes[buf][e][8 * part], row + 128 * s + 8 * part);
            }
            for (int e = tid; e < kTile; e += kA8Threads) {
                const int p = first + min(e, count - 1);
                const std::uint8_t* row = Down ? c.ws.x_plane + static_cast<std::size_t>(p) * kPromptXRowBytes
                                               : c.x_rows + static_cast<std::size_t>(c.ws.entries[p] / kTopK) * kA8XRowBytes;
                ops::cp_async<4>(&sm.scales[buf][e][0], row + (Down ? kA8HScales : kHidden) + kA8Blocks * s);
            }
            ops::cp_commit();
        };
        float acc[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) { acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0F; }
        __syncthreads(); // the previous work item's last stage is read
        stage_b(0, 0);
        for (int s = 0; s < kStages; ++s) {
            const int buf = s & 1;
            // This stage's weights: the warp's unit of each of its 4 blocks, as the narrow kernels read a unit.
            uint4 a[kA8Blocks];
            unsigned sfa[kA8Blocks];
#pragma unroll
            for (int b = 0; b < kA8Blocks; ++b) {
                const std::uint8_t* unit = units + static_cast<std::size_t>(kA8Blocks * s + b) * kA8Unit;
                a[b]   = __ldcs(reinterpret_cast<const uint4*>(unit) + lane);
                sfa[b] = unit[512 + ((t & 1) != 0 ? g + 8 : g)];
            }
            if (s + 1 < kStages) {
                stage_b(buf ^ 1, s + 1);
                ops::cp_wait<1>();
            } else {
                ops::cp_wait<0>();
            }
            __syncthreads();
#pragma unroll
            for (int b = 0; b < kA8Blocks; ++b) {
#pragma unroll
                for (int n = 0; n < 8; ++n) {
                    const std::uint8_t* col = sm.codes[buf][8 * n + g] + 32 * b;
                    const unsigned b0 = *reinterpret_cast<const unsigned*>(col + 4 * t);
                    const unsigned b1 = *reinterpret_cast<const unsigned*>(col + 16 + 4 * t);
                    mma_fp8_e4m3_block_scaled<0, 0>(acc[n][0], acc[n][1], acc[n][2], acc[n][3], a[b].x, a[b].y, a[b].z,
                                                     a[b].w, b0, b1, sfa[b], sm.scales[buf][8 * n + g][b]);
                }
            }
            __syncthreads(); // the stage is refilled two steps on
        }
        // Accumulator (n, i): row 16 rg + g (+ 8 for i >= 2), entry 8 n + 2 t + (i & 1) of the tile.
        if constexpr (Down) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int e = 8 * n + 2 * t + (i & 1);
                    if (e >= count) { continue; }
                    const int entry = c.ws.entries[first + e];
                    c.outputs[static_cast<std::size_t>(entry) * kHidden + 16 * rg + g + (i < 2 ? 0 : 8)] =
                        canon::f32_to_bf16_rn(acc[n][i]);
                }
            }
        } else {
            // Row 2j is intermediate j's gate, 2j + 1 its up: even-g lanes hold gates, lane + 4 the matching ups.
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                std::uint16_t y[4], up[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) { y[i] = canon::f32_to_bf16_rn(acc[n][i]); }
#pragma unroll
                for (int i = 0; i < 4; ++i) { up[i] = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y[i], 4)); }
                if ((g & 1) != 0) { continue; }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int e = 8 * n + 2 * t + (i & 1);
                    if (e >= count) { continue; }
                    const int entry = c.ws.entries[first + e];
                    const int j     = (16 * rg + g + (i < 2 ? 0 : 8)) / 2;
                    c.ws.h[static_cast<std::size_t>(entry) * kInter + j] =
                        detail::swiglu(y[i], up[i], c.gamma[entry], c.gate_cap, c.up_cap);
                }
            }
        }
    }
}

// A warp per dispatched entry: its BF16 h row's MXFP8 groups (lane l: groups l and l + 32) into the x plane's row.
__global__ void __launch_bounds__(kHWarps * 32) quantize_h_a8_kernel(Workspace ws, int entries) {
    constexpr int kGroups = kInter / 32; // 36
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int p    = static_cast<int>(blockIdx.x) * kHWarps + static_cast<int>(threadIdx.x) / 32;
    if (p >= entries) { return; }
    const std::uint16_t* row = ws.h + static_cast<std::size_t>(ws.entries[p]) * kInter;
    std::uint8_t* dst        = ws.x_plane + static_cast<std::size_t>(p) * kPromptXRowBytes;
    for (int grp = lane; grp < kGroups; grp += 32) {
        float v[32];
#pragma unroll
        for (int j = 0; j < 32; ++j) { v[j] = canon::bf16_to_f32(row[32 * grp + j]); }
        const canon::Mxfp8Group q = canon::mxfp8_group(v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            reinterpret_cast<std::uint32_t*>(dst + 32 * grp)[j] =
                q.code[4 * j] | (static_cast<std::uint32_t>(q.code[4 * j + 1]) << 8) |
                (static_cast<std::uint32_t>(q.code[4 * j + 2]) << 16) | (static_cast<std::uint32_t>(q.code[4 * j + 3]) << 24);
        }
        dst[kA8HScales + grp] = q.exponent;
    }
}

int a8_ctas() {
    static const int ctas = [] {
        int device = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return 2 * sms;
    }();
    return ctas;
}

#ifdef _WIN32
TmaDescriptorStaging<wide::Descriptors>& descriptor_staging() {
    static TmaDescriptorStaging<wide::Descriptors> staging;
    return staging;
}
#else
wide::Descriptors& host_descriptors() {
    static wide::Descriptors descriptors;
    return descriptors;
}
#endif

wide::Descriptors make_descriptors(const Workspace& ws) {
    const auto rows = static_cast<std::uint64_t>(ws.capacity) * kTopK;
    using S         = wide::Schedule;
    wide::Descriptors d{};
    d.x_codes  = ::infernix::ops::detail::nvfp4_make_tma_2d(ws.x_plane, CU_TENSOR_MAP_DATA_TYPE_UINT8, kXCodeBytes, rows,
                                           kPromptXRowBytes, S::kCodeRowBytes, S::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_64B,
                                           "inkling prompt x codes TMA");
    d.x_scales = ::infernix::ops::detail::nvfp4_make_tma_2d(ws.x_plane + kXCodeBytes, CU_TENSOR_MAP_DATA_TYPE_UINT8, kXScaleBytes, rows,
                                           kPromptXRowBytes, 16, S::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_NONE,
                                           "inkling prompt x scales TMA");
    d.h_codes  = ::infernix::ops::detail::nvfp4_make_tma_2d(ws.h_plane, CU_TENSOR_MAP_DATA_TYPE_UINT8, kHCodeBytes, rows,
                                           kPromptHRowBytes, S::kCodeRowBytes, S::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_64B,
                                           "inkling prompt h codes TMA");
    d.h_scales = ::infernix::ops::detail::nvfp4_make_tma_2d(ws.h_plane + kHCodeBytes, CU_TENSOR_MAP_DATA_TYPE_UINT8, kHScaleMap, rows,
                                           kPromptHRowBytes, 16, S::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_NONE,
                                           "inkling prompt h scales TMA");
    return d;
}

} // namespace

void moe_experts_prompt(const Activations& x, const Routing& routing, const Bank& bank, const Workspace& ws,
                        std::uint16_t* outputs, cudaStream_t stream) {
    if (bank.hidden != kHidden || bank.intermediate != kInter || bank.experts <= 0 || bank.experts > 256 ||
        x.columns <= 0 || x.columns > ws.capacity) {
        throw std::invalid_argument("w4a4_sparse_moe prompt: Inkling 2's [E, 2304, 1152] banks, 1..capacity columns");
    }
    const bool a8 = x.activation == ExpertActivation::A8;
    if (x.activation == ExpertActivation::A16 ||
        (a8 && (bank.format != ExpertFormat::Nvfp4FourOverSix || x.rows == nullptr || ws.h_a8 == nullptr))) {
        throw std::invalid_argument("w4a4_sparse_moe prompt: the checkpoint's activations, or A8 on an NVFP4 layer with the "
                                    "columns' A8 rows and a workspace carved for A8 (A16 takes moe_experts)");
    }
    const int entries = x.columns * kTopK;
    plan_kernel<<<1, kPlanThreads, 0, stream>>>(ws, bank.records, bank.record_stride, bank.experts);
    check_launch("plan");
    if (a8) {
        wide::Call call{};
        call.dispatch.counts    = ws.counts;
        call.dispatch.offsets   = ws.offsets;
        call.dispatch.entries   = ws.entries;
        call.dispatch.jobs      = ws.jobs;
        call.dispatch.job_count = ws.job_count;
        call.job_records        = ws.job_records;
        call.tiles              = ws.tiles;
        call.pass_tiles         = ws.pass_tiles;
        call.passes             = 1;
        call.top_k              = kTopK;
        call.a8                 = true;
        call.x_limbs            = x.rows;
        call.h_rows             = ws.h_a8;
        const GateUpA8Epilogue gate_up{ws.entries,       routing.gamma,  bank.amax, bank.denominators,
                                       bank.gate_softcap, bank.up_softcap, ws.h};
        wide::launch_a8<A8Shape<wide::Matrix::GateUp>, GateUpA8Epilogue, wide::ScheduleA8Tall>(call, 0, gate_up, stream);
        check_launch("A8 gate/up");
        // h's BF16 plane (dispatch order) as A8 rows, by the codec the x rows took.
        const Tensor h(ws.h, DType::BF16, {kInter, entries});
        Tensor h8(ws.h_a8, DType::I32, {kA8HRowBytes / 4, entries});
        ops::a8_encode_columns(h, h8, stream);
        const DownA8Epilogue down{ws.entries, bank.amax, bank.denominators, outputs};
        wide::launch_a8<A8Shape<wide::Matrix::Down>, DownA8Epilogue, wide::ScheduleA8Tall>(call, 0, down, stream);
        check_launch("A8 down");
        return;
    }
    if (bank.format == ExpertFormat::Mxfp8) {
        const A8Call call{ws, bank.records, bank.record_stride, x.rows, routing.gamma, bank.gate_softcap, bank.up_softcap,
                          outputs};
        a8_grouped_kernel<false><<<a8_ctas(), kA8Threads, 0, stream>>>(call);
        check_launch("MXFP8 gate/up");
        quantize_h_a8_kernel<<<(entries + kHWarps - 1) / kHWarps, kHWarps * 32, 0, stream>>>(ws, entries);
        check_launch("MXFP8 quantize h");
        a8_grouped_kernel<true><<<a8_ctas(), kA8Threads, 0, stream>>>(call);
        check_launch("MXFP8 down");
        return;
    }
    const int gather_ctas = std::min((entries * (kHidden / 16) + kGatherThreads - 1) / kGatherThreads, 4096);
    gather_x_kernel<<<gather_ctas, kGatherThreads, 0, stream>>>(x.rows, ws, entries);
    check_launch("gather x");

    const wide::Descriptors descriptors = make_descriptors(ws);
    wide::Call call{};
    call.dispatch.counts    = ws.counts;
    call.dispatch.offsets   = ws.offsets;
    call.dispatch.entries   = ws.entries;
    call.dispatch.jobs      = ws.jobs;
    call.dispatch.job_count = ws.job_count;
    call.job_records        = ws.job_records;
    call.tiles              = ws.tiles;
    call.pass_tiles         = ws.pass_tiles;
    call.passes             = 1;
#ifdef _WIN32
    call.descriptors = descriptor_staging().stage(descriptors, stream);
#else
    host_descriptors() = descriptors;
    call.descriptors   = &host_descriptors();
#endif
    const GateUpEpilogue gate_up{ws.entries,   x.amax,          routing.gamma, bank.amax, bank.denominators,
                                 bank.gate_softcap, bank.up_softcap, ws.h};
    wide::launch<wide::Matrix::GateUp, wide::Schedule, G>(call, 0, gate_up, stream);
    check_launch("gate/up");
    quantize_h_kernel<<<(entries + kHWarps - 1) / kHWarps, kHWarps * 32, 0, stream>>>(ws, entries);
    check_launch("quantize h");
    const DownEpilogue down{ws.entries, ws.h_amax, bank.amax, bank.denominators, outputs};
    wide::launch<wide::Matrix::Down, wide::Schedule, G>(call, 0, down, stream);
    check_launch("down");
}

} // namespace infernix::ops::w4a4_moe
