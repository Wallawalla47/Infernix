// The wide route of offloaded_sparse_moe: work planning, A4 quantization of x for the wide
// entries, TMA descriptors and the per-pass GEMM launches (wide_expert.h, wide_expert.cuh), and the
// W4A16 route's limb encodings and exact integer GEMMs (wide_expert_a16.cuh).

#include "ops/offloaded_sparse_moe/cuda/wide_expert_a16.cuh"

#include "core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infernix::ops::offloaded_moe::wide {
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

namespace {

// Byte offsets of the workspace's regions; `end` is its size.
struct Offsets {
    std::size_t job_records, cpu_flags, cpu_call, tiles, pass_tiles;
    std::size_t h_blocks, x_plane;                      // W4A4
    std::size_t x_limbs, h_rows, x_emax, h_emax;        // W4A16
    std::size_t end;
};

Offsets offsets(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns, ExpertActivation activation) {
    if (max_jobs <= 0 || entries <= 0 || columns <= 0) {
        throw std::invalid_argument("offloaded_sparse_moe experts workspace: empty call");
    }
    const auto e = static_cast<std::size_t>(entries), j = static_cast<std::size_t>(max_jobs);
    const auto t = static_cast<std::size_t>(columns);
    Offsets o{};
    std::size_t p = 0;
    const auto take = [&p](std::size_t bytes) {
        const std::size_t at = p;
        p += round256(bytes);
        return at;
    };
    o.job_records = take(j * sizeof(void*));
    o.cpu_flags   = take(j * sizeof(std::int32_t));
    o.cpu_call    = take(kCpuCallBytes);
    o.tiles       = take(static_cast<std::size_t>(max_tiles(max_jobs, entries)) * 2 * sizeof(std::int32_t));
    o.pass_tiles  = take((j + 2) * sizeof(std::int32_t));
    if (activation == ExpertActivation::kA16) {
        o.x_limbs = take(t * kA16XRowBytes);
        o.h_rows  = take(e * kA16HRowBytes);
        o.x_emax  = take(t * sizeof(std::int32_t));
        o.h_emax  = take(e * sizeof(std::int32_t));
    } else {
        o.h_blocks = take(e * kHBlocks * sizeof(canon::A4Block));
        o.x_plane  = take(e * kXRowBytes);
    }
    o.end = p;
    return o;
}

// One 16-element block b of a row into its limb-plane row (wide_expert_a16.cuh): the canonical X
// (canon::a16_value; zero in a non-finite column) as H1 = X >> 14, L1 = (X >> 3) & 255 and
// H0 | L0 << 4 with H0 = (X >> 11) & 7, L0 = X & 7, in K tile b / 4 at block offset 16 (b % 4).
template <class Byte>
__device__ __forceinline__ uint4 pack16(const Byte (&b)[16]) {
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i])) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 1])) << 8) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 2])) << 16) |
               (static_cast<unsigned>(static_cast<std::uint8_t>(b[4 * i + 3])) << 24);
    }
    return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ void store_a16_block(std::uint8_t* row, int b, const std::uint16_t (&v)[16], int emax) {
    std::uint8_t high[16], low[16], small[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const std::int32_t x = emax == canon::kA16NonFinite ? 0 : canon::a16_value(v[j], emax);
        high[j]  = static_cast<std::uint8_t>(x >> 14);
        low[j]   = static_cast<std::uint8_t>((x >> 3) & 255);
        small[j] = static_cast<std::uint8_t>(((x >> 11) & 7) | ((x & 7) << 4));
    }
    std::uint8_t* at                              = row + (b / 4) * kA16LimbTileBytes + 16 * (b % 4);
    *reinterpret_cast<uint4*>(at)                 = pack16(high);
    *reinterpret_cast<uint4*>(at + kA16KTile)     = pack16(low);
    *reinterpret_cast<uint4*>(at + 2 * kA16KTile) = pack16(small);
}

__device__ __forceinline__ void unpack16(const uint4 v0, const uint4 v1, std::uint16_t (&v)[16]) {
    const unsigned words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        v[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFU);
        v[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
    }
}

// The largest magnitude's bits of 16 BF16 values (their column-exponent key, canon::a16_column_exponent).
__device__ __forceinline__ unsigned a16_key(const std::uint16_t (&v)[16]) {
    unsigned key = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) { key = max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    return key;
}

__device__ __forceinline__ int a16_emax(unsigned key) {
    return key >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(key >> 7), 1);
}

constexpr int kEncodeXThreads = kHidden / 16; // one 16-element block per thread
constexpr int kEncodeWarps    = 8;

// Every token's x limbs and column exponent (W4A16 wide calls): one CTA per token.
__global__ void __launch_bounds__(kEncodeXThreads)
    encode_x_kernel(const std::uint16_t* __restrict__ x, std::uint8_t* __restrict__ limbs, std::int32_t* __restrict__ emax) {
    __shared__ unsigned keys[kEncodeXThreads / 32];
    const int t = static_cast<int>(blockIdx.x), b = static_cast<int>(threadIdx.x);
    const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * kHidden + 16 * b);
    std::uint16_t v[16];
    unpack16(source[0], source[1], v);
    const unsigned key = __reduce_max_sync(0xFFFFFFFFU, a16_key(v));
    if (b % 32 == 0) { keys[b / 32] = key; }
    __syncthreads();
    unsigned all = 0;
#pragma unroll
    for (int w = 0; w < kEncodeXThreads / 32; ++w) { all = max(all, keys[w]); }
    const int e = a16_emax(all);
    store_a16_block(limbs + static_cast<std::size_t>(t) * kA16XRowBytes, b, v, e);
    if (b == 0) { emax[t] = e; }
}

// The pass's wide entries' h rows rewritten in place from BF16 to limbs, with their exponents: one
// warp per entry, which holds the whole BF16 row in registers before it writes.
__global__ void __launch_bounds__(kEncodeWarps * 32)
    encode_h_kernel(MoeDispatch dispatch, const std::int32_t* __restrict__ tiles, const std::int32_t* __restrict__ pass_tiles,
                    int pass, std::uint8_t* __restrict__ h_rows, std::int32_t* __restrict__ h_emax) {
    constexpr int kBlocks = kIntermediate / 16; // 40: lane l holds blocks l and l + 32
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(blockIdx.x) * kEncodeWarps + static_cast<int>(threadIdx.x) / 32;
    const int warps = static_cast<int>(gridDim.x) * kEncodeWarps;
    const int first_tile = pass_tiles[pass], tile_count = pass_tiles[pass + 1] - first_tile;
    for (int task = warp; task < tile_count * kTileColumns; task += warps) {
        const int tile = first_tile + task / kTileColumns, c = task % kTileColumns;
        const int job = tiles[2 * tile], column = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int entry  = dispatch.offsets[expert] + column + c;
        if (entry >= dispatch.offsets[expert + 1]) { continue; } // warp-uniform
        std::uint8_t* row = h_rows + static_cast<std::size_t>(entry) * kA16HRowBytes;
        const auto* source = reinterpret_cast<const uint4*>(row);
        std::uint16_t v0[16] = {}, v1[16] = {};
        unpack16(source[2 * lane], source[2 * lane + 1], v0);
        const bool second = lane + 32 < kBlocks;
        if (second) { unpack16(source[2 * (lane + 32)], source[2 * (lane + 32) + 1], v1); }
        const int e = a16_emax(__reduce_max_sync(0xFFFFFFFFU, max(a16_key(v0), a16_key(v1))));
        __syncwarp();
        store_a16_block(row, lane, v0, e);
        if (second) { store_a16_block(row, lane + 32, v1, e); }
        if (lane == 0) { h_emax[entry] = e; }
    }
}

} // namespace

std::size_t experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns,
                                    ExpertActivation activation) {
    return offsets(max_jobs, entries, columns, activation).end;
}

ExpertsWorkspace carve_experts_workspace(void* base, std::int32_t max_jobs, std::int32_t entries, std::int32_t columns,
                                         ExpertActivation activation) {
    const Offsets o = offsets(max_jobs, entries, columns, activation);
    auto* p         = static_cast<std::byte*>(base);
    ExpertsWorkspace w{};
    w.job_records = reinterpret_cast<const std::uint8_t**>(p + o.job_records);
    w.cpu_flags   = reinterpret_cast<std::int32_t*>(p + o.cpu_flags);
    w.cpu_call    = p + o.cpu_call;
    w.tiles       = reinterpret_cast<std::int32_t*>(p + o.tiles);
    w.pass_tiles  = reinterpret_cast<std::int32_t*>(p + o.pass_tiles);
    if (activation == ExpertActivation::kA16) {
        w.x_limbs = reinterpret_cast<std::uint8_t*>(p + o.x_limbs);
        w.h_rows  = reinterpret_cast<std::uint8_t*>(p + o.h_rows);
        w.x_emax  = reinterpret_cast<std::int32_t*>(p + o.x_emax);
        w.h_emax  = reinterpret_cast<std::int32_t*>(p + o.h_emax);
    } else {
        w.h_blocks = reinterpret_cast<canon::A4Block*>(p + o.h_blocks);
        w.x_plane  = reinterpret_cast<std::uint8_t*>(p + o.x_plane);
    }
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
    if (source.activation == ExpertActivation::kA16) {
        // W4A16: every token's x limbs once (a token feeds up to top_k experts); h is encoded per pass.
        call.a16     = true;
        call.x_limbs = workspace.x_limbs;
        call.h_rows  = workspace.h_rows;
        call.x_emax  = workspace.x_emax;
        call.h_emax  = workspace.h_emax;
        call.top_k   = top_k;
        encode_x_kernel<<<x.ne[1], kEncodeXThreads, 0, stream>>>(static_cast<const std::uint16_t*>(x.data),
                                                                 workspace.x_limbs, workspace.x_emax);
        check_launch("encode x");
        return call;
    }
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
    if (call.a16) {
        launch_a16<Matrix::GateUp>(
            call, pass, GateUpEpilogueA16{call.scales, call.dispatch.entries, call.x_emax, call.h_rows, call.top_k},
            stream);
        encode_h_kernel<<<persistent_ctas(), kEncodeWarps * 32, 0, stream>>>(call.dispatch, call.tiles, call.pass_tiles,
                                                                             pass, call.h_rows, call.h_emax);
        check_launch("encode h");
        launch_a16<Matrix::Down>(call, pass, DownEpilogueA16{call.scales, call.dispatch.entries, call.h_emax, call.outputs},
                                 stream);
        return;
    }
    launch<Matrix::GateUp>(call, pass, GateUpEpilogue{call.scales, call.h_plane}, stream);
    launch<Matrix::Down>(call, pass, DownEpilogue{call.scales, call.dispatch.entries, call.outputs}, stream);
}

} // namespace infernix::ops::offloaded_moe::wide
