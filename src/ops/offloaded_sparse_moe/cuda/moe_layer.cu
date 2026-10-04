// offloaded_sparse_moe on the GPU: routing, device-side dispatch, the exact W4A4 expert kernels
// and the combine (docs/maintainer/qwen3_8-flash-next-design.md §8.4-8.5, §16.2).
//
// Expert kernels read each `nvfp4_expert_rg16_v1` record straight from wherever it lives: a device
// frame, or the pinned host bank over PCIe (zero-copy). A gate/up CTA owns two row groups of one
// expert (16 intermediates: exactly one A4 block of h) and a down CTA owns four output row groups,
// so every weight unit is read once per pass of eight columns. Products are dp4a over doubled
// E2M1 codes and the block sums are exact int64, so the outputs equal the CPU engine's bits.

#include "ninfer/ops/offloaded_sparse_moe.h"

#include "core/device.h"

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
    return static_cast<int>(canon::e2m1_x2_quad((word >> shift) & 0x0F0F0F0FU));
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

// A call takes the fork route when it has a fork stream and fits one staging pass; moe_experts and
// moe_experts_cpu_wait decide it identically.
bool forked(const MoeExpertSource& source, std::int32_t max_jobs) {
    return source.fork_stream != nullptr && source.staging_slots > 0 &&
           max_jobs <= std::min(source.staging_slots, 512);
}

// Which jobs a pass launch computes: all of them, only those resident in frames (their record is
// resolved directly, so they need not wait for the stage kernel), or only the others.
enum class PassPhase : int { All = 0, Resident = 1, Staged = 2 };

__device__ __forceinline__ const std::uint8_t* pass_record(PassPhase phase, const MoeExpertSource& source,
                                                           int expert, const std::uint8_t* const* job_records,
                                                           int job) {
    if (phase == PassPhase::All) { return job_records[job]; }
    const bool resident = source.frames[expert] >= 0;
    if (resident != (phase == PassPhase::Resident)) { return nullptr; }
    return resident ? source.frame_base + static_cast<std::uint64_t>(source.frames[expert]) * source.record_stride
                    : job_records[job];
}

// ------------------------------------------------------------------------------------- staging

constexpr int kStageCtas  = 16;    // with 16 KiB chunks, a 256 KiB window of host reads in flight
constexpr int kNarrowPassColumns = 4; // calls of at most this many columns use the one-column kernels
constexpr int kStageChunk = 16384;
constexpr int kMaxPassJobs = 512;

// Resolves the record of every job in [job_base, job_base + pass_jobs) and copies the pass's
// non-resident records to the staging slots (pass_jobs <= slots, so all fit). Chunk c of the
// concatenated miss records is copied by CTA c % gridDim.x, keeping the CTAs' reads adjacent.
__global__ void __launch_bounds__(kThreads)
    stage_kernel(MoeDispatch dispatch, MoeExpertSource source, int job_base, int pass_jobs,
                 const std::int32_t* __restrict__ cpu_flags, const std::uint8_t** __restrict__ job_records) {
    __shared__ int miss_jobs[kMaxPassJobs];
    __shared__ int misses;
    const int jobs = min(*dispatch.job_count - job_base, pass_jobs);
    if (jobs <= 0) { return; }
    if (threadIdx.x == 0) {
        int n = 0;
        for (int j = 0; j < jobs; ++j) {
            if (cpu_flags != nullptr && cpu_flags[job_base + j] != 0) { continue; } // served by the CPU
            const int expert = dispatch.jobs[job_base + j];
            const int frame  = source.frames[expert];
            const std::uint8_t* record;
            if (frame >= 0) {
                record = source.frame_base + static_cast<std::uint64_t>(frame) * source.record_stride;
            } else if (source.staging_slots > 0) {
                record       = source.staging_base + static_cast<std::uint64_t>(n) * source.record_stride;
                miss_jobs[n] = job_base + j;
                ++n;
            } else {
                record = source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride;
            }
            if (blockIdx.x == 0) { job_records[job_base + j] = record; }
        }
        misses = n;
    }
    __syncthreads();
    constexpr int kChunks = (static_cast<int>(moe::kRecordBytes) + kStageChunk - 1) / kStageChunk;
    constexpr int kVec    = kStageChunk / 16;
    const int total       = misses * kChunks;
    for (int c = blockIdx.x; c < total; c += gridDim.x) {
        const int m = c / kChunks, offset = (c % kChunks) * kStageChunk;
        const int bytes = min(kStageChunk, static_cast<int>(moe::kRecordBytes) - offset);
        const int expert = dispatch.jobs[miss_jobs[m]];
        const auto* src = reinterpret_cast<const uint4*>(
            source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride + offset);
        auto* dst = reinterpret_cast<uint4*>(source.staging_base + static_cast<std::uint64_t>(m) * source.record_stride +
                                             offset);
        const int vectors = bytes / 16;
        uint4 v[kVec / kThreads];
#pragma unroll
        for (int u = 0; u < kVec / kThreads; ++u) {
            const int i = threadIdx.x + u * kThreads;
            if (i < vectors) { v[u] = __ldcs(src + i); }
        }
#pragma unroll
        for (int u = 0; u < kVec / kThreads; ++u) {
            const int i = threadIdx.x + u * kThreads;
            if (i < vectors) { dst[i] = v[u]; }
        }
    }
}

// -------------------------------------------------------------------------------------- routing

// One warp per column: exact top-k by repeated arg-max with lower-id ties, then the weights.
// Every array index is a compile-time constant (unrolled loops guarded by top_k), so the values,
// the selection and the weights stay in registers; the selection and softmax order are fixed.
__global__ void route_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                             std::int32_t* __restrict__ ids, float* __restrict__ weights,
                             float* __restrict__ shared_gate) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= columns) { return; }
    const float* column = logits + static_cast<std::size_t>(warp) * (experts + 1);
    // 512 experts: 16 per lane.
    constexpr int kPerLane = 16, kMaxTopK = 16;
    float values[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        values[i]   = e < experts ? column[e] : -INFINITY;
    }
    const float score = lane == 0 ? column[experts] : 0.0F;
    float selected[kMaxTopK];
    int chosen[kMaxTopK];
#pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) { break; }
        float best   = -INFINITY;
        int best_id  = 0x7FFFFFFF;
#pragma unroll
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
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            if (lane + 32 * i == best_id) { values[i] = -INFINITY; }
        }
    }
    if (lane == 0) {
        // softmax over all experts then renormalized over the selected ones equals the softmax of
        // the selected logits.
        const float top = selected[0];
        float sum       = 0.0F;
#pragma unroll
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k < top_k) { sum += expf(selected[k] - top); }
        }
#pragma unroll
        for (int k = 0; k < kMaxTopK; ++k) {
            if (k < top_k) {
                ids[static_cast<std::size_t>(warp) * top_k + k]     = chosen[k];
                weights[static_cast<std::size_t>(warp) * top_k + k] = expf(selected[k] - top) / sum;
            }
        }
        shared_gate[warp] = 1.0F / (1.0F + expf(-score));
    }
}

// ------------------------------------------------------------------------------------- dispatch

// Calls of at most this many entries are dispatched by one CTA; the dispatch serves at most this
// many experts (one scan thread each).
constexpr int kDispatchThreads = 1024;

// The dispatch scan, shared by both routes. Thread e of a 1024-thread CTA holds count c of expert e
// (0 for e >= experts). It writes offsets (exclusive scan, offsets[experts] = total), the scatter
// cursors (in global or shared memory), the job list in ascending expert id and the job count.
// Inclusive scans of the counts and of the used flags run within each warp by shuffles, then
// across the warp totals (in the caller's shared `warp_totals`); the results are integers, so any
// scan order gives the same values.
__device__ __forceinline__ void dispatch_scan(int c, int experts, std::int32_t* __restrict__ offsets,
                                              std::int32_t* cursor, std::int32_t* __restrict__ jobs,
                                              std::int32_t* __restrict__ job_count,
                                              std::int32_t (&warp_totals)[2][kDispatchThreads / 32]) {
    const int e = threadIdx.x, lane = e % 32, warp = e / 32;
    int sum = c, used = c > 0 ? 1 : 0;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int s = __shfl_up_sync(0xFFFFFFFFU, sum, offset);
        const int u = __shfl_up_sync(0xFFFFFFFFU, used, offset);
        if (lane >= offset) {
            sum += s;
            used += u;
        }
    }
    if (lane == 31) {
        warp_totals[0][warp] = sum;
        warp_totals[1][warp] = used;
    }
    __syncthreads();
    if (warp == 0) {
        int s = warp_totals[0][lane], u = warp_totals[1][lane];
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int ps = __shfl_up_sync(0xFFFFFFFFU, s, offset);
            const int pu = __shfl_up_sync(0xFFFFFFFFU, u, offset);
            if (lane >= offset) {
                s += ps;
                u += pu;
            }
        }
        warp_totals[0][lane] = s;
        warp_totals[1][lane] = u;
    }
    __syncthreads();
    if (warp > 0) {
        sum += warp_totals[0][warp - 1];
        used += warp_totals[1][warp - 1];
    }
    if (e < experts) {
        offsets[e] = sum - c;
        cursor[e]  = sum - c;
        if (c > 0) { jobs[used - 1] = e; }
    }
    if (e == experts - 1) {
        offsets[experts] = sum;
        *job_count       = used;
    }
}

// The whole dispatch of a call of at most kDispatchThreads entries in one CTA: the count with
// shared atomics (so no array needs clearing between calls and no memset node precedes it), the
// scan, the scatter, and the optional route-log copy of the ids. Same results as the three-kernel
// route below.
__global__ void __launch_bounds__(kDispatchThreads)
    dispatch_small_kernel(const std::int32_t* __restrict__ ids, int entries, int experts, MoeDispatch dispatch,
                          std::int32_t* __restrict__ route_log) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    const int i = threadIdx.x;
    count[i]    = 0;
    __syncthreads();
    const int id = i < entries ? ids[i] : 0;
    if (i < entries) {
        atomicAdd(&count[id], 1);
        if (route_log != nullptr) { route_log[i] = id; }
    }
    __syncthreads();
    const int c = count[i];
    if (i < experts) { dispatch.counts[i] = c; }
    dispatch_scan(c, experts, dispatch.offsets, cursor, dispatch.jobs, dispatch.job_count, warp_totals);
    __syncthreads();
    if (i < entries) { dispatch.entries[atomicAdd(&cursor[id], 1)] = i; }
}

__global__ void count_kernel(const std::int32_t* __restrict__ ids, int entries,
                             std::int32_t* __restrict__ counts, std::int32_t* __restrict__ route_log) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries) {
        const int id = ids[i];
        atomicAdd(&counts[id], 1);
        if (route_log != nullptr) { route_log[i] = id; }
    }
}

// One CTA: the dispatch scan of the counts into offsets, job list and scatter cursors.
__global__ void __launch_bounds__(kDispatchThreads)
    scan_kernel(const std::int32_t* __restrict__ counts, int experts, std::int32_t* __restrict__ offsets,
                std::int32_t* __restrict__ cursor, std::int32_t* __restrict__ jobs,
                std::int32_t* __restrict__ job_count) {
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    const int e = threadIdx.x;
    dispatch_scan(e < experts ? counts[e] : 0, experts, offsets, cursor, jobs, job_count, warp_totals);
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
template <int Groups, int Columns>
__device__ void unit_sums(const std::uint8_t* matrix, int blocks, int rg0, const ActBlock* acts, int n,
                          std::int64_t (&s)[Groups][Columns]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int r = lane & 15, half = lane >> 4;
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int c = 0; c < Columns; ++c) { s[g][c] = 0; }
    }
    for (int u = warp; u < Groups * blocks; u += kWarps) {
        const int g = u / blocks, b = u % blocks;
        const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg0 + g) * blocks + b) * moe::kUnitBytes;
        const int w0 = row_quad(unit, r, 2 * half);
        const int w1 = row_quad(unit, r, 2 * half + 1);
        const int sw = canon::e4m3_scaled(unit[128 + r]);
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
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

template <int Columns>
struct GateUpShared {
    alignas(16) std::uint8_t stage[kGateUpSliceBytes];
    // The activations are read only by unit_sums and the partial sums are written only after the
    // barrier that follows it, so they share storage. That keeps the one-column CTA (49,376 B)
    // under half of an SM's 100 KB, so two of them fit per SM.
    union {
        ActBlock acts[Columns * moe::kGateUpBlocks];
        std::int64_t partial[kWarps][16 * kGateUpGroups][Columns];
    };
    std::uint16_t y[Columns][16 * kGateUpGroups];
    std::uint16_t h[Columns][16];
};

static_assert(sizeof(GateUpShared<1>) + 1024 <= 102400 / 2, "two one-column gate/up CTAs must fit an SM");

template <int Columns>
__global__ void __launch_bounds__(kThreads, Columns == 1 ? 2 : 1)
    gate_up_kernel(const bf16* __restrict__ x, int hidden, MoeDispatch dispatch,
                   MoeExpertSource source, int top_k, const std::uint8_t* const* __restrict__ job_records,
                   const std::int32_t* __restrict__ cpu_flags, int job_base, canon::A4Block* __restrict__ h_blocks,
                   PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm       = *reinterpret_cast<GateUpShared<Columns>*>(smem_raw);
    const int job  = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const int slice  = blockIdx.x; // row groups 2*slice, 2*slice+1; h block `slice`
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const moe::ExpertScales scales = source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const bool split_input = scales.input_up != scales.input_gate;
    stage_async(record + static_cast<std::size_t>(kGateUpGroups * slice) * moe::kGateUpBlocks * moe::kUnitBytes,
                sm.stage, kGateUpSliceBytes);

    for (int pass = 0; pass < count; pass += Columns) {
        const int n = min(Columns, count - pass);
        // Gate rows are even, up rows odd; with distinct input scales the up rows use their own A4.
        for (int parity = 0; parity < (split_input ? 2 : 1); ++parity) {
            __syncthreads();
            quantize_columns(x, hidden, dispatch.entries, first + pass, n, top_k,
                             parity == 0 ? scales.input_gate : scales.input_up, sm.acts,
                             moe::kGateUpBlocks);
            stage_wait();
            __syncthreads();
            std::int64_t s[kGateUpGroups][Columns];
            unit_sums<kGateUpGroups, Columns>(sm.stage, moe::kGateUpBlocks, 0, sm.acts, n, s);
            __syncthreads(); // every warp has read sm.acts before sm.partial overwrites it
            if (lane < 16) {
#pragma unroll
                for (int g = 0; g < kGateUpGroups; ++g) {
#pragma unroll
                    for (int c = 0; c < Columns; ++c) { sm.partial[warp][g * 16 + lane][c] = s[g][c]; }
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

template <int Columns>
struct DownShared {
    alignas(16) std::uint8_t stage[kDownSliceBytes];
    ActBlock acts[Columns * kHBlocks];
    std::int64_t partial[kWarps][16 * kDownGroups][Columns];
};

template <int Columns>
__global__ void __launch_bounds__(kThreads)
    down_kernel(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                int job_base, const canon::A4Block* __restrict__ h_blocks, int columns_out,
                bf16* __restrict__ outputs, PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<DownShared<Columns>*>(smem_raw);
    const int job = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const int tile   = blockIdx.x; // down row groups 4*tile .. 4*tile+3
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const std::uint8_t* down = record + moe::kGateUpBytes;
    const moe::ExpertScales scales = source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    stage_async(down + static_cast<std::size_t>(kDownGroups * tile) * moe::kDownBlocks * moe::kUnitBytes, sm.stage,
                kDownSliceBytes);
    for (int pass = 0; pass < count; pass += Columns) {
        const int n = min(Columns, count - pass);
        __syncthreads();
        for (int i = threadIdx.x; i < n * kHBlocks; i += blockDim.x) {
            sm.acts[i] = to_act(h_blocks[static_cast<std::size_t>(first + pass) * kHBlocks + i]);
        }
        stage_wait();
        __syncthreads();
        std::int64_t s[kDownGroups][Columns];
        unit_sums<kDownGroups, Columns>(sm.stage, moe::kDownBlocks, 0, sm.acts, n, s);
        if (lane < 16) {
#pragma unroll
            for (int g = 0; g < kDownGroups; ++g) {
#pragma unroll
                for (int c = 0; c < Columns; ++c) { sm.partial[warp][g * 16 + lane][c] = s[g][c]; }
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

template <int Columns>
void launch_expert_pass(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                        const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                        canon::A4Block* h_blocks, int columns_out, bf16* outputs, cudaStream_t stream,
                        PassPhase phase = PassPhase::All) {
    static const bool configured = [] {
        cudaFuncSetAttribute(gate_up_kernel<Columns>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(GateUpShared<Columns>)));
        cudaFuncSetAttribute(down_kernel<Columns>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(DownShared<Columns>)));
        cudaFuncSetAttribute(gate_up_kernel<Columns>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(down_kernel<Columns>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        return true;
    }();
    (void)configured;
    gate_up_kernel<Columns><<<dim3(kGateUpCtas, jobs), kThreads, sizeof(GateUpShared<Columns>), stream>>>(
        x, moe::kHidden, dispatch, source, top_k, job_records, flags, base, h_blocks, phase);
    check_launch("gate/up");
    down_kernel<Columns><<<dim3(kDownCtas, jobs), kThreads, sizeof(DownShared<Columns>), stream>>>(
        dispatch, source, moe::kHidden, job_records, flags, base, h_blocks, columns_out, outputs, phase);
    check_launch("down");
}

// ------------------------------------------------------------------------------ CPU-served misses

// Device bookkeeping of a call's CPU jobs, in the caller's workspace.
struct CpuCall {
    std::int32_t pending; // the published sequence, 0 when nothing was published
    std::int32_t jobs;
    std::int32_t job[offloaded_moe::kMaxCpuJobs];
};

// One CTA: picks the CPU jobs (about two thirds of the misses, fewest columns first, each with at
// most kMaxCpuColumns columns), marks them in cpu_flags, publishes x and the request, then the
// sequence after a system-scope fence.
__global__ void __launch_bounds__(kThreads)
    cpu_plan_kernel(MoeDispatch dispatch, MoeExpertSource source, const bf16* __restrict__ x, int columns,
                    int top_k, int max_jobs, std::int32_t* __restrict__ cpu_flags, CpuCall* __restrict__ call) {
    __shared__ int chosen;
    const int jobs = min(*dispatch.job_count, max_jobs);
    for (int j = threadIdx.x; j < max_jobs; j += blockDim.x) { cpu_flags[j] = 0; }
    __syncthreads();
    if (threadIdx.x == 0) {
        const auto& channel = source.cpu;
        int misses = 0;
        for (int j = 0; j < jobs; ++j) { misses += source.frames[dispatch.jobs[j]] < 0 ? 1 : 0; }
        const int want = min(channel.max_jobs, channel.pcie_divisor > 0 ? misses - misses / channel.pcie_divisor : misses);
        int n = 0;
        for (int width = 1; width <= offloaded_moe::kMaxCpuColumns && n < want; ++width) {
            for (int j = 0; j < jobs && n < want; ++j) {
                const int expert = dispatch.jobs[j];
                if (source.frames[expert] >= 0) { continue; }
                const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
                if (count != width) { continue; }
                cpu_flags[j] = 1;
                call->job[n]                = j;
                channel.request->expert[n] = expert;
                channel.request->ncols[n]  = count;
                for (int c = 0; c < count; ++c) { channel.request->column[n][c] = dispatch.entries[first + c] / top_k; }
                ++n;
            }
        }
        call->jobs = n;
        chosen     = n;
        if (n > 0) {
            channel.request->layer = channel.layer;
            channel.request->jobs  = n;
        }
    }
    __syncthreads();
    if (chosen == 0) {
        if (threadIdx.x == 0) { call->pending = 0; }
        return;
    }
    const auto* src = reinterpret_cast<const std::uint32_t*>(x);
    auto* dst       = reinterpret_cast<std::uint32_t*>(source.cpu.x);
    for (int i = threadIdx.x; i < columns * moe::kHidden / 2; i += blockDim.x) { dst[i] = src[i]; }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const std::uint32_t sequence = atomicAdd(source.cpu.sequence, 1U) + 1U;
        *reinterpret_cast<volatile std::uint32_t*>(&source.cpu.request->sequence) = sequence;
        __threadfence_system();
        call->pending = static_cast<std::int32_t>(sequence);
    }
}

// One CTA: waits for the host's answer to this call's request and places the CPU-served outputs.
__global__ void __launch_bounds__(kThreads)
    cpu_wait_kernel(MoeDispatch dispatch, MoeExpertSource source, const CpuCall* __restrict__ call, int hidden,
                    bf16* __restrict__ outputs) {
    __shared__ int ready;
    if (threadIdx.x == 0) {
        const auto sequence = static_cast<std::uint32_t>(call->pending);
        ready = 0;
        if (sequence != 0) {
            std::uint64_t start;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            const auto* done = reinterpret_cast<const volatile std::uint32_t*>(source.cpu.done);
            while (*done != sequence) {
                std::uint64_t now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                if (now - start > 2000000000ULL) { asm volatile("trap;"); }
                __nanosleep(200);
            }
            __threadfence_system();
            ready = 1;
        }
    }
    // One lane per job finds its columns' output entries; slot i * kMaxCpuColumns + c of y holds
    // column c of job i (entry -1: unused).
    constexpr int kSlots = offloaded_moe::kMaxCpuJobs * offloaded_moe::kMaxCpuColumns;
    __shared__ int entries[kSlots];
    for (int s = threadIdx.x; s < kSlots; s += blockDim.x) { entries[s] = -1; }
    __syncthreads();
    if (!ready) { return; }
    const int n = call->jobs;
    if (threadIdx.x < n) {
        const int i      = threadIdx.x;
        const int expert = dispatch.jobs[call->job[i]];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        for (int c = 0; c < count; ++c) { entries[i * offloaded_moe::kMaxCpuColumns + c] = dispatch.entries[first + c]; }
    }
    __syncthreads();
    // The copy issues each thread's loads from mapped memory in batches before storing them, so a
    // batch costs one PCIe round trip rather than one per load. .cv loads: y is rewritten by every
    // layer's request.
    constexpr int kBatch = 8;
    const int vectors    = hidden * 2 / 16; // 16-byte vectors per column
    const int items      = n * offloaded_moe::kMaxCpuColumns * vectors;
    const auto* y        = reinterpret_cast<const int4*>(source.cpu.y);
    auto* out            = reinterpret_cast<int4*>(outputs);
    for (int base = threadIdx.x; base < items; base += kBatch * blockDim.x) {
        int4 loaded[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const int item = base + b * blockDim.x;
            if (item < items && entries[item / vectors] >= 0) { loaded[b] = __ldcv(y + item); }
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const int item = base + b * blockDim.x;
            if (item >= items) { continue; }
            const int entry = entries[item / vectors];
            if (entry >= 0) { out[static_cast<std::size_t>(entry) * vectors + item % vectors] = loaded[b]; }
        }
    }
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
                  std::int32_t* route_log, cudaStream_t stream) {
    require(experts > 0 && experts <= kDispatchThreads, "dispatch supports up to 1024 experts");
    const int entries = static_cast<int>(routing.ids.numel());
    const auto* ids   = static_cast<const std::int32_t*>(routing.ids.data);
    if (entries <= kDispatchThreads) {
        // Decode and verification widths (k * T <= 1024): one kernel and no memset node.
        dispatch_small_kernel<<<1, kDispatchThreads, 0, stream>>>(ids, entries, experts, dispatch, route_log);
        check_launch("dispatch");
        return;
    }
    require(cudaMemsetAsync(dispatch.counts, 0, sizeof(std::int32_t) * experts, stream) == cudaSuccess,
            "dispatch could not clear its counts");
    count_kernel<<<(entries + kThreads - 1) / kThreads, kThreads, 0, stream>>>(ids, entries, dispatch.counts,
                                                                                route_log);
    check_launch("count");
    scan_kernel<<<1, kDispatchThreads, 0, stream>>>(dispatch.counts, experts, dispatch.offsets, dispatch.cursor,
                                                    dispatch.jobs, dispatch.job_count);
    check_launch("scan");
    scatter_kernel<<<(entries + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        ids, entries, dispatch.cursor, dispatch.entries);
    check_launch("scatter");
}

std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries) {
    return (static_cast<std::size_t>(entries) * kHBlocks * sizeof(canon::A4Block) + 255) / 256 * 256 +
           (static_cast<std::size_t>(max_jobs) * sizeof(void*) + 255) / 256 * 256 +
           (static_cast<std::size_t>(max_jobs) * sizeof(std::int32_t) + 255) / 256 * 256 + 256;
}

namespace {

// The CPU bookkeeping of a call, after the expert outputs and job records in its workspace.
CpuCall* cpu_call_of(void* workspace, std::int32_t max_jobs, std::int32_t entries, std::int32_t** flags) {
    const std::size_t h_bytes =
        (static_cast<std::size_t>(entries) * kHBlocks * sizeof(canon::A4Block) + 255) / 256 * 256;
    const std::size_t records_bytes = (static_cast<std::size_t>(max_jobs) * sizeof(void*) + 255) / 256 * 256;
    auto* cpu_flags = reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(workspace) + h_bytes + records_bytes);
    if (flags != nullptr) { *flags = cpu_flags; }
    return reinterpret_cast<CpuCall*>(reinterpret_cast<std::byte*>(cpu_flags) +
                                      (static_cast<std::size_t>(max_jobs) * sizeof(std::int32_t) + 255) / 256 * 256);
}

bool cpu_served(const Tensor& x, const MoeExpertSource& source) {
    return source.cpu.max_jobs > 0 && x.ne[1] <= source.cpu.max_columns;
}

} // namespace

void moe_experts_cpu_wait(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                          std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream) {
    if (forked(source, max_jobs)) { CUDA_CHECK(cudaStreamWaitEvent(stream, source.fork_events[1], 0)); }
    if (!cpu_served(x, source)) { return; }
    CpuCall* call = cpu_call_of(workspace, max_jobs, outputs.ne[1], nullptr);
    cpu_wait_kernel<<<1, kThreads, 0, stream>>>(dispatch, source, call, moe::kHidden,
                                                static_cast<bf16*>(outputs.data));
    check_launch("cpu wait");
}

void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream, bool wait_for_cpu) {
    require(contiguous(x, DType::BF16) && contiguous(outputs, DType::BF16), "experts need BF16 x and outputs");
    require(x.ne[0] == moe::kHidden && outputs.ne[0] == moe::kHidden && outputs.ne[1] == x.ne[1] * top_k,
            "experts geometry differs from the nvfp4_expert_rg16_v1 record");
    require(max_jobs > 0 && max_jobs <= 65535 && source.scales != nullptr && source.frames != nullptr &&
                source.host_records != nullptr && source.record_stride >= moe::kRecordBytes &&
                source.record_stride % 16 == 0 && source.staging_slots >= 0 &&
                (source.staging_slots == 0 || source.staging_base != nullptr),
            "experts source is incomplete");
    auto* h_blocks = static_cast<canon::A4Block*>(workspace);
    const std::size_t h_bytes =
        (static_cast<std::size_t>(outputs.ne[1]) * kHBlocks * sizeof(canon::A4Block) + 255) / 256 * 256;
    auto** job_records = reinterpret_cast<const std::uint8_t**>(static_cast<std::byte*>(workspace) + h_bytes);
    std::int32_t* cpu_flags = nullptr;
    CpuCall* cpu_call       = cpu_call_of(workspace, max_jobs, outputs.ne[1], &cpu_flags);
    const bool cpu          = cpu_served(x, source);
    if (cpu) {
        require(source.cpu.request != nullptr && source.cpu.x != nullptr && source.cpu.y != nullptr &&
                    source.cpu.done != nullptr && source.cpu.sequence != nullptr &&
                    source.cpu.max_jobs <= offloaded_moe::kMaxCpuJobs,
                "CPU channel is incomplete");
        cpu_plan_kernel<<<1, kThreads, 0, stream>>>(dispatch, source, static_cast<const bf16*>(x.data), x.ne[1], top_k,
                                                    max_jobs, cpu_flags, cpu_call);
        check_launch("cpu plan");
    }
    const std::int32_t* flags = cpu ? cpu_flags : nullptr;
    // Passes of at most staging_slots jobs (one pass covering every job without staging).
    const int pass_jobs = source.staging_slots > 0 ? std::min(source.staging_slots, kMaxPassJobs) : max_jobs;
    const int half      = source.staging_slots / 2;
    if (source.overlap_stream != nullptr && half > 0 && max_jobs > half) {
        // Double-buffered passes: stage pass p+1 on the side stream while pass p computes.
        const auto* events = source.overlap_events;
        CUDA_CHECK(cudaEventRecord(events[0], stream));
        CUDA_CHECK(cudaStreamWaitEvent(source.overlap_stream, events[0], 0));
        int pass = 0;
        for (int base = 0; base < max_jobs; base += half, ++pass) {
            const int jobs = std::min(half, max_jobs - base);
            const int b    = pass % 2;
            MoeExpertSource buffer = source;
            buffer.staging_base    = source.staging_base + static_cast<std::uint64_t>(b) * half * source.record_stride;
            buffer.staging_slots   = half;
            if (pass >= 2) { CUDA_CHECK(cudaStreamWaitEvent(source.overlap_stream, events[3 + b], 0)); }
            stage_kernel<<<kStageCtas, kThreads, 0, source.overlap_stream>>>(dispatch, buffer, base, jobs, flags,
                                                                            job_records);
            check_launch("stage");
            CUDA_CHECK(cudaEventRecord(events[1 + b], source.overlap_stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream, events[1 + b], 0));
            launch_expert_pass<kPassColumns>(static_cast<const bf16*>(x.data), dispatch, buffer, top_k, job_records,
                                             flags, base, jobs, h_blocks, outputs.ne[1],
                                             static_cast<bf16*>(outputs.data), stream);
            CUDA_CHECK(cudaEventRecord(events[3 + b], stream));
        }
        if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
        return;
    }
    if (forked(source, max_jobs)) {
        // One pass: the misses are staged and then computed on the fork stream while this stream
        // computes the resident experts; both write disjoint outputs and h blocks, and every
        // expert's arithmetic is the same as in a serial pass.
        CUDA_CHECK(cudaEventRecord(source.fork_events[0], stream));
        CUDA_CHECK(cudaStreamWaitEvent(source.fork_stream, source.fork_events[0], 0));
        stage_kernel<<<kStageCtas, kThreads, 0, source.fork_stream>>>(dispatch, source, 0, max_jobs, flags,
                                                                      job_records);
        check_launch("stage");
        const auto* xs = static_cast<const bf16*>(x.data);
        auto* out      = static_cast<bf16*>(outputs.data);
        if (x.ne[1] <= kNarrowPassColumns) {
            launch_expert_pass<1>(xs, dispatch, source, top_k, job_records, flags, 0, max_jobs, h_blocks,
                                  outputs.ne[1], out, source.fork_stream, PassPhase::Staged);
            launch_expert_pass<1>(xs, dispatch, source, top_k, job_records, flags, 0, max_jobs, h_blocks,
                                  outputs.ne[1], out, stream, PassPhase::Resident);
        } else {
            launch_expert_pass<kPassColumns>(xs, dispatch, source, top_k, job_records, flags, 0, max_jobs, h_blocks,
                                             outputs.ne[1], out, source.fork_stream, PassPhase::Staged);
            launch_expert_pass<kPassColumns>(xs, dispatch, source, top_k, job_records, flags, 0, max_jobs, h_blocks,
                                             outputs.ne[1], out, stream, PassPhase::Resident);
        }
        CUDA_CHECK(cudaEventRecord(source.fork_events[1], source.fork_stream));
        // Joined here, or in moe_experts_cpu_wait so the caller's unrelated work (the shared
        // expert) also overlaps the staging.
        if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
        return;
    }
    for (int base = 0; base < max_jobs; base += pass_jobs) {
        const int jobs = std::min(pass_jobs, max_jobs - base);
        stage_kernel<<<source.staging_slots > 0 ? kStageCtas : 1, kThreads, 0, stream>>>(dispatch, source, base, jobs,
                                                                                       flags, job_records);
        check_launch("stage");
        // Calls of a few columns (decode, MTP verification) use the one-column kernels, which fit
        // two gate/up CTAs per SM; a job with several columns takes one pass per column over its
        // staged weights.
        if (x.ne[1] <= kNarrowPassColumns) {
            launch_expert_pass<1>(static_cast<const bf16*>(x.data), dispatch, source, top_k, job_records, flags, base,
                                  jobs, h_blocks, outputs.ne[1], static_cast<bf16*>(outputs.data), stream);
        } else {
            launch_expert_pass<kPassColumns>(static_cast<const bf16*>(x.data), dispatch, source, top_k, job_records,
                                             flags, base, jobs, h_blocks, outputs.ne[1],
                                             static_cast<bf16*>(outputs.data), stream);
        }
    }
    if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
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
