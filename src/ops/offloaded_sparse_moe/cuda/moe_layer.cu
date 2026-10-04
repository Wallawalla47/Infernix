// offloaded_sparse_moe on the GPU: routing, device-side dispatch, the exact W4A4 expert kernels
// and the combine (docs/maintainer/qwen3_8-flash-next-design.md §8.4-8.5, §16.2).
//
// Expert kernels read each `nvfp4_expert_rg16_v1` record straight from wherever it lives: a device
// frame, or the pinned host bank over PCIe (zero-copy). A gate/up CTA owns two row groups of one
// expert (16 intermediates: exactly one A4 block of h) and a down CTA owns four output row groups,
// so every weight unit is read once per pass of eight columns. Products are dp4a over doubled
// E2M1 codes and the block sums are exact int64, so the outputs equal the CPU engine's bits.

#include "ninfer/ops/offloaded_sparse_moe.h"

#include "ops/common/canonical_math.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

namespace moe = offloaded_moe;
using bf16    = __nv_bfloat16;

constexpr int kThreads      = 256;
constexpr int kWarps        = kThreads / 32;
constexpr int kPassColumns  = 8;
constexpr int kGateUpGroups = 2; // row groups per gate/up CTA: one 16-wide block of h
constexpr int kDownGroups   = 4; // row groups per down CTA: 64 output rows
constexpr int kGateUpCtas   = moe::kGateUpRowGroups / kGateUpGroups; // 40
constexpr int kDownCtas     = moe::kDownRowGroups / kDownGroups;     // 40
constexpr int kHBlocks      = moe::kHBlocks;                         // 40

static_assert(kGateUpCtas == kHBlocks, "a gate/up CTA produces one A4 block of h");

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("offloaded_sparse_moe: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("offloaded_sparse_moe ") + what + ": " +
                                 cudaGetErrorString(error));
    }
}

bool contiguous(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous();
}

// A quantized 16-element activation block in the layout the dp4a loop reads.
struct ActBlock {
    std::int32_t quads[4]; // doubled E2M1 codes, four per word
    std::int32_t scale;    // e4m3_scaled of the block scale
};

__device__ __forceinline__ ActBlock to_act(const canon::A4Block& a) {
    ActBlock out;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        int packed = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) { packed |= (static_cast<int>(a.c2[4 * q + j]) & 0xFF) << (8 * j); }
        out.quads[q] = packed;
    }
    out.scale = a.scale_scaled;
    return out;
}

// Doubled E2M1 codes of row r (0..15) at k = 4q..4q+3 of a unit, as four signed bytes.
__device__ __forceinline__ int row_quad(const std::uint8_t* unit, int r, int q) {
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * q + 4 * (r & 7));
    const int shift          = r < 8 ? 0 : 4;
    int packed               = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        packed |= (canon::e2m1_x2((word >> (8 * j + shift)) & 15U) & 0xFF) << (8 * j);
    }
    return packed;
}

// A CTA's weight slice is contiguous in the record. Every thread issues 16-byte asynchronous copies
// of it into shared memory, so the whole slice is in flight at once instead of one 144-byte unit
// per warp iteration; this matters most for records read zero-copy over PCIe.
__device__ __forceinline__ void stage_async(const std::uint8_t* src, std::uint8_t* dst, int bytes) {
    for (int i = static_cast<int>(threadIdx.x) * 16; i < bytes; i += static_cast<int>(blockDim.x) * 16) {
        const auto s = static_cast<unsigned>(__cvta_generic_to_shared(dst + i));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src + i) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void stage_wait() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ const std::uint8_t* record_of(const MoeExpertSource& source, int expert) {
    const int frame = source.frames[expert];
    return frame >= 0 ? source.frame_base + static_cast<std::uint64_t>(frame) * source.record_stride
                      : source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride;
}

// -------------------------------------------------------------------------------------- routing

// One warp per column: exact top-k by repeated arg-max with lower-id ties, then the weights.
__global__ void route_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                             std::int32_t* __restrict__ ids, float* __restrict__ weights,
                             float* __restrict__ shared_gate) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= columns) { return; }
    const float* column = logits + static_cast<std::size_t>(warp) * (experts + 1);
    // 512 experts: 16 per lane.
    constexpr int kPerLane = 16;
    float values[kPerLane];
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        values[i]   = e < experts ? column[e] : -INFINITY;
    }
    float selected[16];
    int chosen[16];
    for (int k = 0; k < top_k; ++k) {
        float best   = -INFINITY;
        int best_id  = 0x7FFFFFFF;
        for (int i = 0; i < kPerLane; ++i) {
            const int e = lane + 32 * i;
            if (e < experts && (values[i] > best || (values[i] == best && e < best_id))) {
                best    = values[i];
                best_id = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other    = __shfl_xor_sync(0xFFFFFFFFU, best, offset);
            const int other_id   = __shfl_xor_sync(0xFFFFFFFFU, best_id, offset);
            if (other > best || (other == best && other_id < best_id)) {
                best    = other;
                best_id = other_id;
            }
        }
        selected[k] = best;
        chosen[k]   = best_id;
        if (best_id % 32 == lane) { values[best_id / 32] = -INFINITY; }
    }
    if (lane == 0) {
        // softmax over all experts then renormalized over the selected ones equals the softmax of
        // the selected logits.
        const float top = selected[0];
        float sum       = 0.0F;
        for (int k = 0; k < top_k; ++k) { sum += expf(selected[k] - top); }
        for (int k = 0; k < top_k; ++k) {
            ids[static_cast<std::size_t>(warp) * top_k + k]     = chosen[k];
            weights[static_cast<std::size_t>(warp) * top_k + k] = expf(selected[k] - top) / sum;
        }
        const float s     = column[experts];
        shared_gate[warp] = 1.0F / (1.0F + expf(-s));
    }
}

// ------------------------------------------------------------------------------------- dispatch

__global__ void count_kernel(const std::int32_t* __restrict__ ids, int entries,
                             std::int32_t* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries) { atomicAdd(&counts[ids[i]], 1); }
}

// One CTA: exclusive scan of the counts, the job list, and the scatter cursors.
__global__ void scan_kernel(const std::int32_t* __restrict__ counts, int experts,
                            std::int32_t* __restrict__ offsets, std::int32_t* __restrict__ cursor,
                            std::int32_t* __restrict__ jobs, std::int32_t* __restrict__ job_count) {
    __shared__ std::int32_t scan[1024];
    __shared__ std::int32_t used[1024];
    const int e = threadIdx.x;
    const int c = e < experts ? counts[e] : 0;
    scan[e]     = c;
    used[e]     = c > 0 ? 1 : 0;
    __syncthreads();
    for (int step = 1; step < 1024; step <<= 1) {
        const int a = e >= step ? scan[e - step] : 0;
        const int b = e >= step ? used[e - step] : 0;
        __syncthreads();
        scan[e] += a;
        used[e] += b;
        __syncthreads();
    }
    if (e < experts) {
        const int start = scan[e] - c;
        offsets[e]      = start;
        cursor[e]       = start;
        if (c > 0) { jobs[used[e] - 1] = e; }
    }
    if (e == experts - 1) {
        offsets[experts] = scan[e];
        *job_count       = used[e];
    }
}

__global__ void scatter_kernel(const std::int32_t* __restrict__ ids, int entries,
                               std::int32_t* __restrict__ cursor, std::int32_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries) { out[atomicAdd(&cursor[ids[i]], 1)] = i; }
}

// --------------------------------------------------------------------------------------- experts

// Quantizes column x[:, t] to A4 blocks [0, blocks) with `scale` into smem.
__device__ void quantize_columns(const bf16* __restrict__ x, int hidden, const std::int32_t* entries,
                                 int first, int count, int top_k, float scale, ActBlock* acts,
                                 int blocks) {
    for (int i = threadIdx.x; i < count * blocks; i += blockDim.x) {
        const int c = i / blocks, b = i % blocks;
        const int t = entries[first + c] / top_k;
        const auto* v = reinterpret_cast<const std::uint16_t*>(x + static_cast<std::size_t>(t) * hidden + 16 * b);
        acts[c * blocks + b] = to_act(canon::quantize_a4_block(v, scale));
    }
}

// Exact sums S[g][c] of row (lane & 15) of `groups` consecutive row groups starting at `rg0`,
// over this warp's share of the units, for `n` columns. Lanes 0..15 hold the results.
template <int Groups>
__device__ void unit_sums(const std::uint8_t* matrix, int blocks, int rg0, const ActBlock* acts, int n,
                          std::int64_t (&s)[Groups][kPassColumns]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = lane & 15, half = lane >> 4;
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) { s[g][c] = 0; }
    }
    for (int u = warp; u < Groups * blocks; u += kWarps) {
        const int g = u / blocks, b = u % blocks;
        const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg0 + g) * blocks + b) * moe::kUnitBytes;
        const int w0 = row_quad(unit, r, 2 * half);
        const int w1 = row_quad(unit, r, 2 * half + 1);
        const int sw = canon::e4m3_scaled(unit[128 + r]);
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            if (c < n) { // n is uniform across the warp, so the shuffle below is converged
                const ActBlock& a = acts[c * blocks + b];
                int p             = __dp4a(w0, a.quads[2 * half], 0);
                p                 = __dp4a(w1, a.quads[2 * half + 1], p);
                p += __shfl_xor_sync(0xFFFFFFFFU, p, 16);
                // |P| <= 2304 and Sw < 2^18: P * Sw fits int32 (design §16.2).
#pragma unroll
                for (int gg = 0; gg < Groups; ++gg) {
                    if (gg == g) { s[gg][c] += static_cast<std::int64_t>(p * sw) * a.scale; }
                }
            }
        }
    }
}

constexpr int kGateUpSliceBytes = kGateUpGroups * moe::kGateUpBlocks * static_cast<int>(moe::kUnitBytes); // 46,080
constexpr int kDownSliceBytes   = kDownGroups * moe::kDownBlocks * static_cast<int>(moe::kUnitBytes);     // 23,040
static_assert(kGateUpSliceBytes % 16 == 0 && kDownSliceBytes % 16 == 0);

struct GateUpShared {
    alignas(16) std::uint8_t stage[kGateUpSliceBytes];
    ActBlock acts[kPassColumns * moe::kGateUpBlocks];          // 25,600 B
    std::int64_t partial[kWarps][16 * kGateUpGroups][kPassColumns]; // 16,384 B
    std::uint16_t y[kPassColumns][16 * kGateUpGroups];
    std::uint16_t h[kPassColumns][16];
};

__global__ void __launch_bounds__(kThreads)
    gate_up_kernel(const bf16* __restrict__ x, int hidden, MoeDispatch dispatch,
                   MoeExpertSource source, int top_k, canon::A4Block* __restrict__ h_blocks) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm       = *reinterpret_cast<GateUpShared*>(smem_raw);
    const int job  = blockIdx.y;
    if (job >= *dispatch.job_count) { return; }
    const int expert = dispatch.jobs[job];
    const int slice  = blockIdx.x; // row groups 2*slice, 2*slice+1; h block `slice`
    const std::uint8_t* record = record_of(source, expert);
    const moe::ExpertScales scales = source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const bool split_input = scales.input_up != scales.input_gate;
    stage_async(record + static_cast<std::size_t>(kGateUpGroups * slice) * moe::kGateUpBlocks * moe::kUnitBytes,
                sm.stage, kGateUpSliceBytes);

    for (int pass = 0; pass < count; pass += kPassColumns) {
        const int n = min(kPassColumns, count - pass);
        // Gate rows are even, up rows odd; with distinct input scales the up rows use their own A4.
        for (int parity = 0; parity < (split_input ? 2 : 1); ++parity) {
            __syncthreads();
            quantize_columns(x, hidden, dispatch.entries, first + pass, n, top_k,
                             parity == 0 ? scales.input_gate : scales.input_up, sm.acts,
                             moe::kGateUpBlocks);
            stage_wait();
            __syncthreads();
            std::int64_t s[kGateUpGroups][kPassColumns];
            unit_sums<kGateUpGroups>(sm.stage, moe::kGateUpBlocks, 0, sm.acts, n, s);
            if (lane < 16) {
#pragma unroll
                for (int g = 0; g < kGateUpGroups; ++g) {
#pragma unroll
                    for (int c = 0; c < kPassColumns; ++c) { sm.partial[warp][g * 16 + lane][c] = s[g][c]; }
                }
            }
            __syncthreads();
            for (int i = threadIdx.x; i < 16 * kGateUpGroups * n; i += blockDim.x) {
                const int row = i % (16 * kGateUpGroups), c = i / (16 * kGateUpGroups);
                const bool gate = (row & 1) == 0;
                if (split_input && gate != (parity == 0)) { continue; }
                std::int64_t total = 0;
                for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row][c]; }
                sm.y[c][row] = canon::a4_row_output(total, gate ? scales.alpha_gate : scales.alpha_up);
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 16 * n; i += blockDim.x) {
            const int k = i % 16, c = i / 16;
            sm.h[c][k] = canon::swiglu_bf16(sm.y[c][2 * k], sm.y[c][2 * k + 1]);
        }
        __syncthreads();
        if (threadIdx.x < n) {
            const int c = threadIdx.x;
            h_blocks[static_cast<std::size_t>(first + pass + c) * kHBlocks + slice] =
                canon::quantize_a4_block(sm.h[c], scales.input_down);
        }
    }
}

struct DownShared {
    alignas(16) std::uint8_t stage[kDownSliceBytes];
    ActBlock acts[kPassColumns * kHBlocks];                       // 6,400 B
    std::int64_t partial[kWarps][16 * kDownGroups][kPassColumns]; // 32,768 B
};

__global__ void __launch_bounds__(kThreads)
    down_kernel(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                const canon::A4Block* __restrict__ h_blocks, int columns_out,
                bf16* __restrict__ outputs) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<DownShared*>(smem_raw);
    const int job = blockIdx.y;
    if (job >= *dispatch.job_count) { return; }
    const int expert = dispatch.jobs[job];
    const int tile   = blockIdx.x; // down row groups 4*tile .. 4*tile+3
    const std::uint8_t* down = record_of(source, expert) + moe::kGateUpBytes;
    const moe::ExpertScales scales = source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    stage_async(down + static_cast<std::size_t>(kDownGroups * tile) * moe::kDownBlocks * moe::kUnitBytes, sm.stage,
                kDownSliceBytes);
    for (int pass = 0; pass < count; pass += kPassColumns) {
        const int n = min(kPassColumns, count - pass);
        __syncthreads();
        for (int i = threadIdx.x; i < n * kHBlocks; i += blockDim.x) {
            sm.acts[i] = to_act(h_blocks[static_cast<std::size_t>(first + pass) * kHBlocks + i]);
        }
        stage_wait();
        __syncthreads();
        std::int64_t s[kDownGroups][kPassColumns];
        unit_sums<kDownGroups>(sm.stage, moe::kDownBlocks, 0, sm.acts, n, s);
        if (lane < 16) {
#pragma unroll
            for (int g = 0; g < kDownGroups; ++g) {
#pragma unroll
                for (int c = 0; c < kPassColumns; ++c) { sm.partial[warp][g * 16 + lane][c] = s[g][c]; }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 16 * kDownGroups * n; i += blockDim.x) {
            const int row = i % (16 * kDownGroups), c = i / (16 * kDownGroups);
            std::int64_t total = 0;
            for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row][c]; }
            const int column = dispatch.entries[first + pass + c];
            const std::uint16_t y = canon::a4_row_output(total, scales.alpha_down);
            reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * hidden +
                                                      16 * kDownGroups * tile + row] = y;
        }
    }
    (void)columns_out;
}

// --------------------------------------------------------------------------------------- combine

__global__ void combine_kernel(const bf16* __restrict__ outputs, const float* __restrict__ weights,
                               const float* __restrict__ shared_gate, const bf16* __restrict__ shared,
                               int hidden, int top_k, int columns, bf16* __restrict__ y) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    float sum = 0.0F;
    for (int k = 0; k < top_k; ++k) { // fixed rank order: placement-invariant
        sum += weights[static_cast<std::size_t>(t) * top_k + k] *
               __bfloat162float(outputs[(static_cast<std::size_t>(t) * top_k + k) * hidden + d]);
    }
    sum += shared_gate[t] * __bfloat162float(shared[static_cast<std::size_t>(t) * hidden + d]);
    y[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum);
}

} // namespace

void moe_route(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, cudaStream_t stream) {
    require(contiguous(logits, DType::FP32) && contiguous(routing.ids, DType::I32) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32),
            "route requires contiguous FP32 logits and I32/FP32 outputs");
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    require(experts > 0 && experts <= 512 && top_k > 0 && top_k <= 16 && top_k <= experts && columns > 0,
            "route supports up to 512 experts and top-16");
    require(routing.ids.ne[0] == top_k && routing.ids.ne[1] == columns && routing.weights.ne[0] == top_k &&
                routing.weights.ne[1] == columns && routing.shared_gate.numel() == columns,
            "route output shapes disagree");
    const int warps_per_block = kThreads / 32;
    route_kernel<<<(columns + warps_per_block - 1) / warps_per_block, kThreads, 0, stream>>>(
        static_cast<const float*>(logits.data), experts, columns, top_k,
        static_cast<std::int32_t*>(routing.ids.data), static_cast<float*>(routing.weights.data),
        static_cast<float*>(routing.shared_gate.data));
    check_launch("route");
}

std::size_t moe_dispatch_bytes(std::int32_t experts, std::int32_t entries) {
    const std::size_t words = static_cast<std::size_t>(experts) * 4 + 1 + 1 + static_cast<std::size_t>(entries);
    return (words * sizeof(std::int32_t) + 255) / 256 * 256;
}

MoeDispatch carve_moe_dispatch(void* base, std::int32_t experts, std::int32_t entries) {
    auto* p = static_cast<std::int32_t*>(base);
    MoeDispatch out;
    out.counts    = p;
    out.offsets   = out.counts + experts;
    out.cursor    = out.offsets + experts + 1;
    out.jobs      = out.cursor + experts;
    out.job_count = out.jobs + experts;
    out.entries   = out.job_count + 1;
    (void)entries;
    return out;
}

void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch,
                  cudaStream_t stream) {
    require(experts > 0 && experts <= 1024, "dispatch supports up to 1024 experts");
    const int entries = static_cast<int>(routing.ids.numel());
    require(cudaMemsetAsync(dispatch.counts, 0, sizeof(std::int32_t) * experts, stream) == cudaSuccess,
            "dispatch could not clear its counts");
    const auto* ids = static_cast<const std::int32_t*>(routing.ids.data);
    count_kernel<<<(entries + kThreads - 1) / kThreads, kThreads, 0, stream>>>(ids, entries, dispatch.counts);
    check_launch("count");
    scan_kernel<<<1, 1024, 0, stream>>>(dispatch.counts, experts, dispatch.offsets, dispatch.cursor,
                                        dispatch.jobs, dispatch.job_count);
    check_launch("scan");
    scatter_kernel<<<(entries + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        ids, entries, dispatch.cursor, dispatch.entries);
    check_launch("scatter");
}

std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries) {
    (void)max_jobs;
    return (static_cast<std::size_t>(entries) * kHBlocks * sizeof(canon::A4Block) + 255) / 256 * 256;
}

void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream) {
    require(contiguous(x, DType::BF16) && contiguous(outputs, DType::BF16), "experts need BF16 x and outputs");
    require(x.ne[0] == moe::kHidden && outputs.ne[0] == moe::kHidden && outputs.ne[1] == x.ne[1] * top_k,
            "experts geometry differs from the nvfp4_expert_rg16_v1 record");
    require(max_jobs > 0 && max_jobs <= 65535 && source.scales != nullptr && source.frames != nullptr &&
                source.host_records != nullptr && source.record_stride >= moe::kRecordBytes,
            "experts source is incomplete");
    static bool configured = [] {
        cudaFuncSetAttribute(gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(GateUpShared)));
        cudaFuncSetAttribute(down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(DownShared)));
        return true;
    }();
    (void)configured;
    auto* h_blocks = static_cast<canon::A4Block*>(workspace);
    gate_up_kernel<<<dim3(kGateUpCtas, max_jobs), kThreads, sizeof(GateUpShared), stream>>>(
        static_cast<const bf16*>(x.data), moe::kHidden, dispatch, source, top_k, h_blocks);
    check_launch("gate/up");
    down_kernel<<<dim3(kDownCtas, max_jobs), kThreads, sizeof(DownShared), stream>>>(
        dispatch, source, moe::kHidden, h_blocks, outputs.ne[1], static_cast<bf16*>(outputs.data));
    check_launch("down");
}

void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream) {
    require(contiguous(outputs, DType::BF16) && contiguous(shared, DType::BF16) && contiguous(y, DType::BF16),
            "combine requires contiguous BF16 tensors");
    const int hidden = y.ne[0], columns = y.ne[1], top_k = routing.weights.ne[0];
    require(outputs.ne[0] == hidden && outputs.ne[1] == columns * top_k && shared.ne[0] == hidden &&
                shared.ne[1] == columns && routing.weights.ne[1] == columns,
            "combine shapes disagree");
    combine_kernel<<<dim3((hidden + kThreads - 1) / kThreads, columns), kThreads, 0, stream>>>(
        static_cast<const bf16*>(outputs.data), static_cast<const float*>(routing.weights.data),
        static_cast<const float*>(routing.shared_gate.data), static_cast<const bf16*>(shared.data),
        hidden, top_k, columns, static_cast<bf16*>(y.data));
    check_launch("combine");
}

} // namespace ninfer::ops
