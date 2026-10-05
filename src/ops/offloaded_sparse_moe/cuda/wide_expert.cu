// The wide route of offloaded_sparse_moe: work planning, A4 quantization of x for the wide
// entries, TMA descriptors and the per-pass GEMM launches (wide_expert.h, wide_expert.cuh).

#include "ops/offloaded_sparse_moe/cuda/wide_expert.cuh"

#include "core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::ops::offloaded_moe::wide {
namespace {

constexpr int kPlanThreads     = 1024;
constexpr int kQuantizeThreads = 256;

std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("offloaded_sparse_moe wide ") + what + ": " + cudaGetErrorString(error));
    }
}

// One CTA: the call's wide work tiles, (job, first column) for each 64-column slice of every wide
// job in job order, and for each pass of pass_jobs jobs the index of its first tile.
__global__ void __launch_bounds__(kPlanThreads)
    plan_kernel(MoeDispatch dispatch, const ExpertScales* __restrict__ scales, int max_jobs, int pass_jobs, int passes,
                std::int32_t* __restrict__ tiles, std::int32_t* __restrict__ pass_tiles) {
    __shared__ int scan[kPlanThreads];
    __shared__ int base;
    const int tid  = static_cast<int>(threadIdx.x);
    const int jobs = min(*dispatch.job_count, max_jobs);
    if (tid == 0) { base = 0; }
    __syncthreads();
    for (int chunk = 0; chunk < passes * pass_jobs; chunk += kPlanThreads) {
        const int job = chunk + tid;
        int n         = 0;
        if (job < jobs) {
            const int expert  = dispatch.jobs[job];
            const int columns = dispatch.offsets[expert + 1] - dispatch.offsets[expert];
            if (wide_route(columns, scales[expert])) { n = (columns + kTileColumns - 1) / kTileColumns; }
        }
        scan[tid] = n;
        __syncthreads();
        for (int step = 1; step < kPlanThreads; step <<= 1) {
            const int add = tid >= step ? scan[tid - step] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const int first = base + scan[tid] - n;
        for (int i = 0; i < n; ++i) {
            tiles[2 * (first + i)]     = job;
            tiles[2 * (first + i) + 1] = i * kTileColumns;
        }
        if (job % pass_jobs == 0 && job / pass_jobs < passes) { pass_tiles[job / pass_jobs] = first; }
        __syncthreads();
        if (tid == kPlanThreads - 1) { base = first + n; }
        __syncthreads();
    }
    if (tid == 0) { pass_tiles[passes] = base; }
}

// A4(x) of every wide entry with its expert's gate/up input scale, in the plane layout of
// wide_expert.h: row p (the entry's dispatch index) holds 1,280 code bytes then 160 scale bytes.
__global__ void __launch_bounds__(kQuantizeThreads)
    quantize_kernel(const std::uint16_t* __restrict__ x, int top_k, MoeDispatch dispatch,
                    const ExpertScales* __restrict__ scales, const std::int32_t* __restrict__ tiles,
                    const std::int32_t* __restrict__ pass_tiles, int passes, std::uint8_t* __restrict__ plane) {
    const int total = pass_tiles[passes];
    for (int tile = static_cast<int>(blockIdx.x); tile < total; tile += static_cast<int>(gridDim.x)) {
        const int job = tiles[2 * tile], column = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first  = dispatch.offsets[expert] + column;
        const int count  = min(kTileColumns, dispatch.offsets[expert + 1] - first);
        const float scale = scales[expert].input_gate;
        for (int task = static_cast<int>(threadIdx.x); task < count * kGateUpBlocks; task += kQuantizeThreads) {
            const int i = task / kGateUpBlocks, b = task % kGateUpBlocks;
            const int p = first + i;
            const int t = dispatch.entries[p] / top_k;
            const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * kHidden + 16 * b);
            const uint4 v0 = source[0], v1 = source[1];
            std::uint16_t v[16];
            const unsigned words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                v[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFU);
                v[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
            }
            const canon::A4Codes q = canon::quantize_a4_codes(v, scale);
            unsigned low = 0, high = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                low |= (static_cast<unsigned>(q.code[j]) | (static_cast<unsigned>(q.code[4 + j]) << 4)) << (8 * j);
                high |= (static_cast<unsigned>(q.code[8 + j]) | (static_cast<unsigned>(q.code[12 + j]) << 4)) << (8 * j);
            }
            std::uint8_t* row  = plane + static_cast<std::size_t>(p) * kXRowBytes;
            const int position = plane_block(b);
            *reinterpret_cast<uint2*>(row + 8 * position) = make_uint2(low, high);
            row[kXCodeBytes + position]                   = q.scale_word;
        }
    }
}

Descriptors make_descriptors(const std::uint8_t* x_plane, const std::uint8_t* h_plane, int entries) {
    const auto rows = static_cast<std::uint64_t>(entries);
    auto* x = const_cast<std::uint8_t*>(x_plane);
    auto* h = const_cast<std::uint8_t*>(h_plane);
    Descriptors d{};
    d.x_codes  = detail::nvfp4_make_tma_2d(x, CU_TENSOR_MAP_DATA_TYPE_UINT8, kXCodeBytes, rows, kXRowBytes,
                                           Schedule::kCodeRowBytes, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_64B,
                                           "encode wide x codes TMA");
    d.x_scales = detail::nvfp4_make_tma_2d(x + kXCodeBytes, CU_TENSOR_MAP_DATA_TYPE_UINT8, kXScaleBytes, rows,
                                           kXRowBytes, 16, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_NONE,
                                           "encode wide x scales TMA");
    d.h_codes  = detail::nvfp4_make_tma_2d(h, CU_TENSOR_MAP_DATA_TYPE_UINT8, kHCodeBytes, rows, kHRowStride,
                                           Schedule::kCodeRowBytes, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_64B,
                                           "encode wide h codes TMA");
    // 40 scale bytes per row: the last K tile's 16-byte box also reads the 8 bytes after them
    // (inside the row), which no MMA step uses.
    d.h_scales = detail::nvfp4_make_tma_2d(h + kHCodeBytes, CU_TENSOR_MAP_DATA_TYPE_UINT8, kHScaleMapBytes, rows,
                                           kHRowStride, 16, Schedule::kBlockTokens, CU_TENSOR_MAP_SWIZZLE_NONE,
                                           "encode wide h scales TMA");
    return d;
}

#ifdef _WIN32
// The single staging of the wide route's descriptors (core/tma_descriptor_staging.cuh); every wide
// launch runs on the compute stream that stages them.
TmaDescriptorStaging<Descriptors>& descriptor_staging() {
    static TmaDescriptorStaging<Descriptors> staging;
    return staging;
}
#else
// Kernel parameters are copied at launch, so one host copy serves every call in turn.
Descriptors& host_descriptors() {
    static Descriptors descriptors;
    return descriptors;
}
#endif

} // namespace

std::int32_t max_tiles(std::int32_t max_jobs, std::int32_t entries) {
    return max_jobs + (entries + kTileColumns - 1) / kTileColumns;
}

std::size_t experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries) {
    const auto e = static_cast<std::size_t>(entries), j = static_cast<std::size_t>(max_jobs);
    return round256(e * kHBlocks * sizeof(canon::A4Block)) + round256(j * sizeof(void*)) +
           round256(j * sizeof(std::int32_t)) + kCpuCallBytes + round256(e * kXRowBytes) +
           round256(static_cast<std::size_t>(max_tiles(max_jobs, entries)) * 2 * sizeof(std::int32_t)) +
           round256((j + 2) * sizeof(std::int32_t));
}

ExpertsWorkspace carve_experts_workspace(void* base, std::int32_t max_jobs, std::int32_t entries) {
    const auto e = static_cast<std::size_t>(entries), j = static_cast<std::size_t>(max_jobs);
    auto* p = static_cast<std::byte*>(base);
    ExpertsWorkspace w{};
    w.h_blocks = reinterpret_cast<canon::A4Block*>(p);
    p += round256(e * kHBlocks * sizeof(canon::A4Block));
    w.job_records = reinterpret_cast<const std::uint8_t**>(p);
    p += round256(j * sizeof(void*));
    w.cpu_flags = reinterpret_cast<std::int32_t*>(p);
    p += round256(j * sizeof(std::int32_t));
    w.cpu_call = p;
    p += kCpuCallBytes;
    w.x_plane = reinterpret_cast<std::uint8_t*>(p);
    p += round256(e * kXRowBytes);
    w.tiles = reinterpret_cast<std::int32_t*>(p);
    p += round256(static_cast<std::size_t>(max_tiles(max_jobs, entries)) * 2 * sizeof(std::int32_t));
    w.pass_tiles = reinterpret_cast<std::int32_t*>(p);
    return w;
}

int persistent_ctas() {
    static const int ctas = [] {
        int device = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return sms * Schedule::kMinBlocksPerSm;
    }();
    return ctas;
}

Call prepare(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source, std::int32_t top_k,
             std::int32_t max_jobs, std::int32_t pass_jobs, const ExpertsWorkspace& workspace, Tensor& outputs,
             cudaStream_t stream) {
    if (pass_jobs <= 0 || max_jobs <= 0) { throw std::invalid_argument("offloaded_sparse_moe wide: empty passes"); }
    const int entries = outputs.ne[1];
    Call call;
    call.dispatch    = dispatch;
    call.scales      = source.scales;
    call.job_records = workspace.job_records;
    call.tiles       = workspace.tiles;
    call.pass_tiles  = workspace.pass_tiles;
    call.h_plane     = reinterpret_cast<std::uint8_t*>(workspace.h_blocks);
    call.outputs     = static_cast<std::uint16_t*>(outputs.data);
    call.passes      = (max_jobs + pass_jobs - 1) / pass_jobs;
    plan_kernel<<<1, kPlanThreads, 0, stream>>>(dispatch, source.scales, max_jobs, pass_jobs, call.passes,
                                                workspace.tiles, workspace.pass_tiles);
    check_launch("plan");
    quantize_kernel<<<persistent_ctas() * 2, kQuantizeThreads, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), top_k, dispatch, source.scales, workspace.tiles,
        workspace.pass_tiles, call.passes, workspace.x_plane);
    check_launch("quantize");
    const Descriptors descriptors = make_descriptors(workspace.x_plane, call.h_plane, entries);
#ifdef _WIN32
    call.descriptors = descriptor_staging().stage(descriptors, stream);
#else
    host_descriptors() = descriptors;
    call.descriptors   = &host_descriptors();
#endif
    return call;
}

void run_pass(const Call& call, std::int32_t pass, cudaStream_t stream) {
    if (pass < 0 || pass >= call.passes) { throw std::out_of_range("offloaded_sparse_moe wide: pass out of range"); }
    launch<Matrix::GateUp>(call, pass, GateUpEpilogue{call.scales, call.h_plane}, stream);
    launch<Matrix::Down>(call, pass, DownEpilogue{call.scales, call.dispatch.entries, call.outputs}, stream);
}

} // namespace ninfer::ops::offloaded_moe::wide
