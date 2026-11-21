// Inkling 2's routed experts on the prompt route (include/infernix/ops/w4a4_sparse_moe.h moe_experts_prompt): the NVFP4
// layers as offloaded_sparse_moe's grouped block-scaled GEMM (ops/offloaded_sparse_moe/cuda/wide_expert.cuh, at
// Inkling's expert geometry) with this Op's own epilogues; the MXFP8 layers on the narrow kernels.
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
#include "core/tma_descriptor_staging.cuh"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.cuh"
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
    if (bank.format == ExpertFormat::Mxfp8) {
        moe_experts(x, routing, bank, ws, outputs, stream);
        return;
    }
    if (bank.hidden != kHidden || bank.intermediate != kInter || bank.experts <= 0 || bank.experts > 256 ||
        x.columns <= 0 || x.columns > ws.capacity) {
        throw std::invalid_argument("w4a4_sparse_moe prompt: Inkling 2's [E, 2304, 1152] NVFP4 banks, 1..capacity columns");
    }
    const int entries = x.columns * kTopK;
    plan_kernel<<<1, kPlanThreads, 0, stream>>>(ws, bank.records, bank.record_stride, bank.experts);
    check_launch("plan");
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
