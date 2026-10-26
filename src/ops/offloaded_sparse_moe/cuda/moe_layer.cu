// offloaded_sparse_moe on the GPU: routing, device-side dispatch, the exact W4A4 expert kernels
// and the combine (docs/maintainer/qwen3_8-flash-next-design.md §8.4-8.5, §16.2).
//
// Expert kernels read each `nvfp4_expert_rg16_v1` record straight from wherever it lives: a device
// frame, or the pinned host bank over PCIe (zero-copy). A gate/up CTA owns two row groups of one
// expert (16 intermediates: exactly one A4 block of h) and a down CTA owns four output row groups,
// so every weight unit is read once per pass of eight columns. Products are dp4a over doubled
// E2M1 codes and the block sums are exact int64, so the outputs equal the CPU engine's bits.
// Experts with more than eight columns in a call take the tensor-core wide route instead
// (wide_expert.h); these kernels skip them.

#include "infernix/ops/offloaded_sparse_moe.h"

#include "core/device.h"
#include "core/temporary_switches.h"
#include "core/pdl.cuh"

#include "ops/common/canonical_math.h"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_exact.h"
#include "ops/ggml/ggml_bf16_tile.cuh"
#include "ops/ggml/ggml_tile.cuh"
#include "ops/fp8_block/fp8_exact.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"
#include "infernix/ops/lossless_linear.h"
#include "infernix/ops/lossless_quantize.h"
#include "ops/lossless/cr_math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::ops {
namespace {

namespace moe = offloaded_moe;
using bf16    = __nv_bfloat16;

constexpr int kThreads      = 256;
constexpr int kWarps        = kThreads / 32;
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

// The pitch of frames, staging slots and landings (MoeExpertSource::frame_stride; prefetch slots
// take prefetch_stride when it is set).
__host__ __device__ __forceinline__ std::uint64_t frame_pitch(const MoeExpertSource& source) {
    return source.frame_stride != 0 ? source.frame_stride : source.record_stride;
}

// Bytes a stage copies: a ggml bank's record, else the NVFP4 record of the bank's geometry.
__device__ __forceinline__ int record_copy_bytes(const MoeExpertSource& source) {
    return source.ggml.record_bytes != 0 ? static_cast<int>(source.ggml.record_bytes)
                                         : static_cast<int>(moe::wide::nvfp4_record_bytes(source.shape));
}

__host__ __device__ __forceinline__ bool ggml_bank(const MoeExpertSource& source) {
    return source.ggml.gate_up_format != QType::NVFP4_MUL;
}

// A non-resident expert's host record (MoeExpertSource::host_table, else the pinned bank).
__device__ __forceinline__ const std::uint8_t* host_record(const MoeExpertSource& source, int expert) {
    return source.host_table != nullptr ? source.host_table[expert]
                                        : source.host_records + static_cast<std::uint64_t>(expert) * source.record_stride;
}

// A call takes the fork route when it has a fork stream, fits one staging pass and is narrow (no
// expert can have more than eight columns); moe_experts and moe_experts_cpu_wait decide it
// identically.
bool forked(const MoeExpertSource& source, std::int32_t max_jobs, std::int32_t columns) {
    return source.fork_stream != nullptr && source.staging_slots > 0 &&
           max_jobs <= std::min(source.staging_slots, 512) && columns <= moe::kMaxColumns;
}

// Which jobs a pass launch computes: all of them, only those resident in frames (their record is
// resolved directly, so they need not wait for the stage kernel), or only the others.
enum class PassPhase : int { All = 0, Resident = 1, Staged = 2 };

// A job's record when no copy serves it: the layer's shared expert (MoeExpertSource::shared_record) or
// a frame-resident expert; null otherwise.
__device__ __forceinline__ const std::uint8_t* resident_record(const MoeExpertSource& source, int expert) {
    if (expert == source.shared_expert) { return source.shared_record; }
    const int frame = source.frames[expert];
    return frame >= 0 ? source.frame_base + static_cast<std::uint64_t>(frame) * frame_pitch(source) : nullptr;
}

__device__ __forceinline__ const std::uint8_t* pass_record(PassPhase phase, const MoeExpertSource& source,
                                                           int expert, const std::uint8_t* const* job_records,
                                                           int job) {
    if (phase == PassPhase::All) { return job_records[job]; }
    const std::uint8_t* resident = resident_record(source, expert);
    if ((resident != nullptr) != (phase == PassPhase::Resident)) { return nullptr; }
    return resident != nullptr ? resident : job_records[job];
}

// ------------------------------------------------------------ call bookkeeping, fetch channel

// Device bookkeeping of a call's CPU jobs, in the caller's workspace.
struct CpuCall {
    std::int32_t pending; // the published sequence, 0 when nothing was published
    std::int32_t jobs;
    std::int32_t job[offloaded_moe::kMaxCpuJobs];
};

// Device bookkeeping of a call's fetch request (MoeFetchChannel), in the caller's workspace after
// its CpuCall. Written by fetch_plan_kernel; the stage kernels of the call's passes share it.
struct FetchCall {
    std::uint32_t pending; // the published sequence, 0 when nothing was published
    std::int32_t first;    // job index of the first fetch-served job (they are the call's last jobs)
    std::int32_t count;
    std::uint32_t landed;  // device mirror of the host's landed count for `pending`
    std::uint32_t failed;  // nonzero once the request failed (host status or silent host)
    std::uint32_t poller;  // 1 while a CTA polls the mapped response
    std::uint32_t ctas;    // stage CTAs of the current pass that finished (wraps per pass)
    std::uint32_t beat;    // the host heartbeat last seen, and when (%globaltimer)
    std::uint64_t beat_ns;
};

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// The call's error word: layer << 16 | code (MoeExpertSource::error).
__device__ __forceinline__ void report_error(const MoeExpertSource& source, std::uint32_t code) {
    if (source.error == nullptr) { return; }
    const int layer = source.fetch.request != nullptr ? source.fetch.layer : source.cpu.layer;
    *reinterpret_cast<volatile std::uint32_t*>(source.error) = (static_cast<std::uint32_t>(layer) << 16U) | (code & 0xFFFFU);
}

// The fetch index of `job` in the call's request, or -1 when the job is not fetch-served.
__device__ __forceinline__ int fetch_index(const FetchCall* call, int job) {
    if (call == nullptr || call->pending == 0 || job < call->first || job >= call->first + call->count) { return -1; }
    return job - call->first;
}

// One poll of the host's answer by the CTA holding the poller token: mirrors `landed` into the
// device word the other CTAs read in L2 (keeping PCIe reads to one poller), and fails the request
// on a host status or when the heartbeat has not changed for source.heartbeat_timeout_ns.
__device__ void poll_fetch(const MoeExpertSource& source, FetchCall* call) {
    volatile FetchCall* v = call;
    const auto* response  = reinterpret_cast<const volatile offloaded_moe::FetchResponse*>(source.fetch.response);
    if (response->sequence == v->pending) {
        __threadfence_system();
        const std::uint32_t status = response->status;
        const std::uint32_t landed = response->landed;
        if (status != 0) {
            report_error(source, status);
            v->failed = 1;
            return;
        }
        if (landed > v->landed) {
            __threadfence_system();
            v->landed = landed;
        }
    }
    const std::uint32_t beat = *reinterpret_cast<const volatile std::uint32_t*>(source.fetch.heartbeat);
    const std::uint64_t now  = global_ns();
    if (beat != v->beat) {
        v->beat    = beat;
        v->beat_ns = now;
    } else if (now - v->beat_ns > source.heartbeat_timeout_ns) {
        report_error(source, offloaded_moe::kErrorFetchSilent);
        v->failed = 1;
    }
}

// Thread 0 of a stage CTA: waits until record f of the call's request has landed and returns its
// host address, or null once the request has failed.
__device__ const std::uint8_t* fetch_wait(const MoeExpertSource& source, FetchCall* call, int f) {
    volatile FetchCall* v = call;
    for (;;) {
        if (v->failed != 0) { return nullptr; }
        if (static_cast<int>(v->landed) > f) { break; }
        if (atomicCAS(&call->poller, 0U, 1U) == 0U) {
            poll_fetch(source, call);
            __threadfence();
            atomicExch(&call->poller, 0U);
        }
        __nanosleep(256);
    }
    __threadfence_system();
    const auto* record = reinterpret_cast<const volatile std::uint64_t*>(&source.fetch.response->record[f]);
    return reinterpret_cast<const std::uint8_t*>(*record);
}

// One CTA, after the CPU plan: the call's fetch-served jobs (non-resident, no host record, not a
// CPU job, not streamed) move to the end of dispatch.jobs in a stable partition (cpu_flags and the
// CPU call's job indices move with them), and their experts are published as one request in job
// order: the payload, a system-scope fence, then the sequence (never 0).
__global__ void __launch_bounds__(kThreads)
    fetch_plan_kernel(MoeDispatch dispatch, MoeExpertSource source, int max_jobs, std::int32_t* __restrict__ cpu_flags,
                      CpuCall* __restrict__ cpu_call, FetchCall* __restrict__ call) {
    __shared__ int expert_of[offloaded_moe::kMaxFetch];
    __shared__ int flag_of[offloaded_moe::kMaxFetch];
    __shared__ short rank_of[offloaded_moe::kMaxFetch]; // fetch jobs before this one
    __shared__ unsigned char fetch_of[offloaded_moe::kMaxFetch];
    __shared__ int warp_fetch[kWarps];
    __shared__ int carry;
    const int jobs = min(*dispatch.job_count, max_jobs);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (threadIdx.x == 0) {
        call->pending = 0;
        call->count   = 0;
        carry         = 0;
    }
    __syncthreads();
    for (int base = 0; base < jobs; base += blockDim.x) {
        const int j = base + threadIdx.x;
        bool fetch  = false;
        if (j < jobs) {
            const int expert = dispatch.jobs[j];
            const int flag   = cpu_flags != nullptr ? cpu_flags[j] : 0;
            expert_of[j]     = expert;
            flag_of[j]       = flag;
            fetch = resident_record(source, expert) == nullptr && source.host_table[expert] == nullptr && flag == 0 &&
                    (source.prefetched == nullptr || source.prefetched[expert] < 0);
            fetch_of[j] = fetch ? 1 : 0;
        }
        const unsigned ballot = __ballot_sync(0xFFFFFFFFU, fetch);
        if (lane == 0) { warp_fetch[warp] = __popc(ballot); }
        __syncthreads();
        if (j < jobs) {
            int rank = carry + __popc(ballot & ((1U << lane) - 1U));
            for (int w = 0; w < warp; ++w) { rank += warp_fetch[w]; }
            rank_of[j] = static_cast<short>(rank);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 0; w < kWarps; ++w) { carry += warp_fetch[w]; }
        }
        __syncthreads();
    }
    const int count = carry;
    if (count == 0) { return; }
    const int first = jobs - count;
    // New index: fetch jobs after every other job, each group in its old order.
    for (int j = threadIdx.x; j < jobs; j += blockDim.x) {
        const int to = fetch_of[j] != 0 ? first + rank_of[j] : j - rank_of[j];
        dispatch.jobs[to] = expert_of[j];
        if (cpu_flags != nullptr) { cpu_flags[to] = flag_of[j]; }
        if (fetch_of[j] != 0) { source.fetch.request->expert[rank_of[j]] = expert_of[j]; }
    }
    if (cpu_call != nullptr) {
        for (int s = threadIdx.x; s < cpu_call->jobs; s += blockDim.x) {
            const int j     = cpu_call->job[s]; // never a fetch job: it moves down by the fetch jobs before it
            cpu_call->job[s] = j - rank_of[j];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        source.fetch.request->layer = source.fetch.layer;
        source.fetch.request->count = count;
        __threadfence_system();
        std::uint32_t sequence = atomicAdd(source.fetch.sequence, 1U) + 1U;
        if (sequence == 0) { sequence = atomicAdd(source.fetch.sequence, 1U) + 1U; }
        *reinterpret_cast<volatile std::uint32_t*>(&source.fetch.request->sequence) = sequence;
        __threadfence_system();
        call->first   = first;
        call->count   = count;
        call->landed  = 0;
        call->failed  = 0;
        call->poller  = 0;
        call->ctas    = 0;
        call->beat    = *reinterpret_cast<const volatile std::uint32_t*>(source.fetch.heartbeat);
        call->beat_ns = global_ns();
        call->pending = sequence;
    }
}

// ------------------------------------------------------------------------------------- staging

constexpr int kStageCtas  = 16;    // with 16 KiB chunks, a 256 KiB window of host reads in flight
constexpr int kStageChunk = 16384;
constexpr int kMaxPassJobs = 512;

// Resolves the record of every job in [job_base, job_base + pass_jobs) and copies the pass's
// non-resident records that are not CPU-served (its misses) to their destinations: the n-th miss in
// job order lands in frame landing[n] when n < landing_slots and landing[n] >= 0 (block 0 logs
// landed[n] = its expert), else it takes the next staging slot (pass_jobs <= slots, so all fit).
// Every CTA ranks the misses with warp ballots; chunk c of the concatenated miss records is copied
// by CTA c % gridDim.x, keeping the CTAs' reads adjacent. A fetch-served miss (`fetch`, the call's
// request) is copied from the host's answer once its record has landed; after the pass's copies the
// last CTA to finish stores `consumed` for the request's records up to the pass's last one.
__global__ void __launch_bounds__(kThreads)
    stage_kernel(MoeDispatch dispatch, MoeExpertSource source, int job_base, int pass_jobs,
                 const std::int32_t* __restrict__ cpu_flags, const std::uint8_t** __restrict__ job_records,
                 FetchCall* __restrict__ fetch) {
    __shared__ int miss_jobs[kMaxPassJobs];
    __shared__ std::uint8_t* miss_dst[kMaxPassJobs];
    __shared__ int warp_misses[kWarps];
    __shared__ int landed_before[offloaded_moe::kMaxLandingSlots + 1]; // landing frames among slots [0, n)
    __shared__ int misses;
    pdl::enter(); // a programmatic dependent of the plan on the call's own stream
    const int jobs = min(*dispatch.job_count - job_base, pass_jobs);
    if (jobs <= 0) { return; }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int landing_slots = source.landing != nullptr ? source.landing_slots : 0;
    if (threadIdx.x == 0) {
        misses           = 0;
        landed_before[0] = 0;
        for (int n = 0; n < landing_slots; ++n) { landed_before[n + 1] = landed_before[n] + (source.landing[n] >= 0 ? 1 : 0); }
    }
    __syncthreads();
    for (int base = 0; base < jobs; base += blockDim.x) {
        const int j = base + threadIdx.x;
        bool miss = false, resolved = false;
        int expert = 0;
        const std::uint8_t* record = nullptr;
        if (j < jobs && (cpu_flags == nullptr || cpu_flags[job_base + j] == 0)) { // else served by the CPU
            resolved        = true;
            expert                       = dispatch.jobs[job_base + j];
            const std::uint8_t* resident = resident_record(source, expert);
            const int slot = resident == nullptr && source.prefetched != nullptr ? source.prefetched[expert] : -1;
            if (resident != nullptr) {
                record = resident;
            } else if (slot >= 0) { // streamed by the caller (F2)
                record = source.prefetch_base + static_cast<std::uint64_t>(slot) *
                                                    (source.prefetch_stride != 0 ? source.prefetch_stride : frame_pitch(source));
            } else if (source.staging_slots > 0) {
                miss = true;
            } else {
                record = host_record(source, expert);
                // SSD-only and not a CPU job (a fetch channel needs staging): the passes skip it.
                if (record == nullptr) { report_error(source, offloaded_moe::kErrorUnservedRecord); }
            }
        }
        const unsigned ballot = __ballot_sync(0xFFFFFFFFU, miss);
        if (lane == 0) { warp_misses[warp] = __popc(ballot); }
        __syncthreads();
        if (miss) {
            int n = misses + __popc(ballot & ((1U << lane) - 1U));
            for (int v = 0; v < warp; ++v) { n += warp_misses[v]; }
            std::uint8_t* destination;
            if (n < landing_slots && source.landing[n] >= 0) {
                // The frames are the caller's writable pool; frame_base is const for the read routes.
                destination = const_cast<std::uint8_t*>(source.frame_base) +
                              static_cast<std::uint64_t>(source.landing[n]) * frame_pitch(source);
                if (blockIdx.x == 0) { source.landed[n] = expert; }
            } else {
                const int slot = n - landed_before[min(n, landing_slots)];
                destination    = source.staging_base + static_cast<std::uint64_t>(slot) * frame_pitch(source);
            }
            miss_jobs[n] = job_base + j;
            miss_dst[n]  = destination;
            record       = destination;
        }
        if (resolved && blockIdx.x == 0) { job_records[job_base + j] = record; }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int v = 0; v < kWarps; ++v) { misses += warp_misses[v]; }
        }
        __syncthreads();
    }
    const int record_bytes = record_copy_bytes(source);
    const int kChunks      = (record_bytes + kStageChunk - 1) / kStageChunk;
    constexpr int kVec    = kStageChunk / 16;
    const int total       = misses * kChunks;
    // Tier records live in host slots the host rewrites (RAM slots, the fetch ring): .cv loads
    // discard any L2 line of an earlier occupant. The full-mode bank never changes.
    const bool mutable_host = source.host_table != nullptr;
    __shared__ const std::uint8_t* fetched;
    for (int c = blockIdx.x; c < total; c += gridDim.x) {
        const int m = c / kChunks, offset = (c % kChunks) * kStageChunk;
        const int bytes = min(kStageChunk, record_bytes - offset);
        const int expert = dispatch.jobs[miss_jobs[m]];
        const int f      = fetch_index(fetch, miss_jobs[m]);
        const std::uint8_t* record;
        if (f >= 0) { // uniform in the CTA: one chunk per iteration
            if (threadIdx.x == 0) { fetched = fetch_wait(source, fetch, f); }
            __syncthreads();
            record = fetched;
            __syncthreads();
        } else {
            record = host_record(source, expert);
            if (record == nullptr && threadIdx.x == 0) { report_error(source, offloaded_moe::kErrorUnservedRecord); }
        }
        if (record == nullptr) { continue; } // the round fails (error word); the passes compute garbage
        const auto* src = reinterpret_cast<const uint4*>(record + offset);
        auto* dst = reinterpret_cast<uint4*>(miss_dst[m] + offset);
        const int vectors = bytes / 16;
        uint4 v[kVec / kThreads];
#pragma unroll
        for (int u = 0; u < kVec / kThreads; ++u) {
            const int i = threadIdx.x + u * kThreads;
            if (i < vectors) { v[u] = mutable_host ? __ldcv(src + i) : __ldcs(src + i); }
        }
#pragma unroll
        for (int u = 0; u < kVec / kThreads; ++u) {
            const int i = threadIdx.x + u * kThreads;
            if (i < vectors) { dst[i] = v[u]; }
        }
    }
    // The pass's fetch-served records [0, b) of the request are copied once every CTA is here.
    const int b = fetch != nullptr && fetch->pending != 0 ? min(job_base + jobs, fetch->first + fetch->count) - fetch->first : 0;
    if (b > 0) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            if (atomicInc(&fetch->ctas, gridDim.x - 1) == gridDim.x - 1) {
                const std::uint64_t word = static_cast<std::uint64_t>(fetch->pending) << 32U | static_cast<std::uint32_t>(b);
                *reinterpret_cast<volatile std::uint64_t*>(source.fetch.consumed) = word;
                __threadfence_system();
            }
        }
    }
}

// -------------------------------------------------------------------------------------- routing

// A router logit as the route reads it (RouterLogits): as stored, or rounded once to BF16.
__device__ __forceinline__ float router_logit(float value, bool bf16_logits) {
    return bf16_logits ? __bfloat162float(__float2bfloat16_rn(value)) : value;
}

// One warp routes column t: exact top-k by repeated arg-max with lower-id ties, then the weights.
// Every array index is a compile-time constant (unrolled loops), so the values stay in registers;
// the selection and softmax order are fixed. Lane k < top_k returns the k-th selected id.
__device__ __forceinline__ int route_column(const float* __restrict__ logits, int experts, int top_k, int t,
                                            bool bf16_logits, std::int32_t* __restrict__ ids,
                                            float* __restrict__ weights, float* __restrict__ shared_gate) {
    const int lane      = threadIdx.x % 32;
    const float* column = logits + static_cast<std::size_t>(t) * (experts + 1);
    // 512 experts: 16 per lane.
    constexpr int kPerLane = 16;
    float values[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        values[i]   = e < experts ? router_logit(column[e], bf16_logits) : -INFINITY;
    }
    const float score = lane == 0 ? router_logit(column[experts], bf16_logits) : 0.0F;
    // Lane k keeps the k-th selection.
    float selected = -INFINITY;
    int chosen     = 0;
    for (int k = 0; k < top_k; ++k) {
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
        if (lane == k) {
            selected = best;
            chosen   = best_id;
        }
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            if (lane + 32 * i == best_id) { values[i] = -INFINITY; }
        }
    }
    // softmax over all experts then renormalized over the selected ones equals the softmax of the
    // selected logits. Every lane sums them in selection order.
    const float top = __shfl_sync(0xFFFFFFFFU, selected, 0);
    float sum       = 0.0F;
    for (int k = 0; k < top_k; ++k) { sum += expf(__shfl_sync(0xFFFFFFFFU, selected, k) - top); }
    if (lane < top_k) {
        ids[static_cast<std::size_t>(t) * top_k + lane]     = chosen;
        weights[static_cast<std::size_t>(t) * top_k + lane] = expf(selected - top) / sum;
    }
    if (lane == 0) { shared_gate[t] = 1.0F / (1.0F + expf(-score)); }
    return chosen;
}

__global__ void route_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                             bool bf16_logits, std::int32_t* __restrict__ ids, float* __restrict__ weights,
                             float* __restrict__ shared_gate) {
    pdl::enter(); // launched as a programmatic dependent of the router projection
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp < columns) { route_column(logits, experts, top_k, warp, bf16_logits, ids, weights, shared_gate); }
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

// Whether column t is live under the call's mask (MoeColumnMask).
__device__ __forceinline__ bool column_live(const MoeColumnMask& mask, int t) {
    return t % mask.width < mask.valid[t / mask.width];
}

// Whether entry i (routed entries t * top_k + slot, then a shared job's routed + t) belongs to a
// live column; every entry is live without a mask (kMasked false).
template <bool kMasked>
__device__ __forceinline__ bool entry_live(const MoeColumnMask& mask, int i, int routed, int top_k) {
    if constexpr (!kMasked) {
        return true;
    } else {
        return column_live(mask, i < routed ? i / top_k : i - routed);
    }
}

// The whole dispatch of a call of at most kDispatchThreads entries in one CTA: the count with
// shared atomics (so no array needs clearing between calls and no memset node precedes it), the
// scan, the scatter, and the optional route-log copy of the ids. Same results as the three-kernel
// route below.
// The caller has cleared `count` and synchronized; thread i < entries holds entry i's expert `id`
// and whether it is `live` (a masked column's entry is logged but not dispatched). `routed` of the
// entries are routed (the route log takes them); a shared job's follow.
__device__ __forceinline__ void dispatch_small(int id, bool live, int entries, int routed, int experts,
                                               const MoeDispatch& dispatch, std::int32_t* __restrict__ route_log,
                                               std::int32_t (&count)[kDispatchThreads],
                                               std::int32_t (&cursor)[kDispatchThreads],
                                               std::int32_t (&warp_totals)[2][kDispatchThreads / 32]) {
    const int i = threadIdx.x;
    if (i < entries) {
        if (live) { atomicAdd(&count[id], 1); }
        if (route_log != nullptr && i < routed) { route_log[i] = id; }
    }
    __syncthreads();
    const int c = count[i];
    if (i < experts) { dispatch.counts[i] = c; }
    dispatch_scan(c, experts, dispatch.offsets, cursor, dispatch.jobs, dispatch.job_count, warp_totals);
    __syncthreads();
    if (i < entries && live) { dispatch.entries[atomicAdd(&cursor[id], 1)] = i; }
}

// The dispatch's entry i: a routed id, or (i >= routed) the shared job's expert.
__device__ __forceinline__ int entry_expert(const std::int32_t* __restrict__ ids, int i, int routed, int shared_expert) {
    return i < routed ? ids[i] : shared_expert;
}

template <bool kMasked>
__global__ void __launch_bounds__(kDispatchThreads)
    dispatch_small_kernel(const std::int32_t* __restrict__ ids, int entries, int routed, int shared_expert,
                          int experts, MoeDispatch dispatch, std::int32_t* __restrict__ route_log,
                          MoeColumnMask mask, int top_k) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    pdl::enter(); // launched as a programmatic dependent of the route
    const int i = threadIdx.x;
    count[i]    = 0;
    __syncthreads();
    dispatch_small(i < entries ? entry_expert(ids, i, routed, shared_expert) : 0,
                   !kMasked || (i < entries && entry_live<kMasked>(mask, i, routed, top_k)), entries, routed, experts,
                   dispatch, route_log, count, cursor, warp_totals);
}

// Calls of at most this many columns (and kDispatchThreads entries) route and dispatch in one CTA,
// one warp per column: the dispatch reads the ids from shared memory, with no kernel boundary
// between the two. Same results as route_kernel followed by dispatch_small_kernel.
constexpr int kRouteDispatchColumns = kDispatchThreads / 32;

template <bool kMasked>
__global__ void __launch_bounds__(kDispatchThreads)
    route_dispatch_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                          bool bf16_logits, bool shared_job, std::int32_t* __restrict__ ids,
                          float* __restrict__ weights, float* __restrict__ shared_gate, MoeDispatch dispatch,
                          std::int32_t* __restrict__ route_log, MoeColumnMask mask) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t routed[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    pdl::enter(); // launched as a programmatic dependent of the router projection
    const int i = threadIdx.x, warp = i / 32, lane = i % 32;
    count[i]    = 0;
    if (warp < columns) {
        const int id = route_column(logits, experts, top_k, warp, bf16_logits, ids, weights, shared_gate);
        if (lane < top_k) { routed[warp * top_k + lane] = id; }
    }
    __syncthreads();
    const int routed_entries = columns * top_k, entries = routed_entries + (shared_job ? columns : 0);
    dispatch_small(i < entries ? (i < routed_entries ? routed[i] : experts) : 0,
                   !kMasked || (i < entries && entry_live<kMasked>(mask, i, routed_entries, top_k)), entries,
                   routed_entries, experts + (shared_job ? 1 : 0), dispatch, route_log, count, cursor, warp_totals);
}

template <bool kMasked>
__global__ void count_kernel(const std::int32_t* __restrict__ ids, int entries, int routed, int shared_expert,
                             std::int32_t* __restrict__ counts, std::int32_t* __restrict__ route_log,
                             MoeColumnMask mask, int top_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries) {
        const int id = entry_expert(ids, i, routed, shared_expert);
        if (entry_live<kMasked>(mask, i, routed, top_k)) { atomicAdd(&counts[id], 1); }
        if (route_log != nullptr && i < routed) { route_log[i] = id; }
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

template <bool kMasked>
__global__ void scatter_kernel(const std::int32_t* __restrict__ ids, int entries, int routed, int shared_expert,
                               std::int32_t* __restrict__ cursor, std::int32_t* __restrict__ out,
                               MoeColumnMask mask, int top_k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < entries && entry_live<kMasked>(mask, i, routed, top_k)) {
        out[atomicAdd(&cursor[entry_expert(ids, i, routed, shared_expert)], 1)] = i;
    }
}

// --------------------------------------------------------------------------------------- experts
//
// The W4A4 narrow kernels compute one column of a job at a time (a job of n columns makes n passes
// over its staged weights): at op level (frames, cold L2, uniform routing) they beat an eight-column
// form 1.26-2.3x at every width from one to 64 columns per expert, which was removed 2026-10-11.
// Under the temporary INFERNIX_TMP_NARROW_PAIRS a job of two or more columns runs them two at a time
// ("column pairs" below).
//
// Half2 unit products. A unit's code byte 32q + 4i + j holds rows i (low nibble) and i + 8 (high nibble) at k = 4q + j.
// cvt.rn.f16x2.e2m1x2 turns it into the two E2M1 values as a half2 (low nibble in .x), so lane
// l = 8q + i of a warp takes code word q, i of a unit and accumulates rows i and i + 8 over its four
// k with three HFMA2 against the activation codes, stored with each code in both nibbles so the
// same conversion broadcasts them. Every value is a multiple of 1/2 at most 6 in magnitude, so the
// products are multiples of 1/4 at most 36 and a lane's four-term sum, at most 144, is exact in
// FP16 (ulp 1/8 below 256). Four times it is the lane's share of the doubled-code product sum
// P_b of design §16.2, an integer of at most 576: (P_q * Sw) * Sa summed over the four lanes of a
// row equals (P_b * Sw) * Sa, so the int64 row sums are exactly the CPU engine's.

// One column's A4 block: in byte j of dup[q] the E2M1 code of element
// 4q + j in both nibbles, and the block's e4m3_scaled scale.
struct ActCodes {
    std::uint32_t dup[4];
    std::int32_t scale;
};

__device__ __forceinline__ std::uint32_t dup_nibbles(std::uint32_t codes) { return codes | (codes << 4); }

// The E2M1 code of a doubled value c2 (e2m1_x2's inverse; zero maps to code 0).
__device__ __forceinline__ std::uint32_t e2m1_code_of_x2(int c2) {
    const int m = c2 < 0 ? -c2 : c2; // 0, 1, 2, 3, 4, 6, 8, 12
    const std::uint32_t idx = m <= 4 ? static_cast<std::uint32_t>(m) : m == 6 ? 5U : m == 8 ? 6U : 7U;
    return idx | (c2 < 0 ? 8U : 0U);
}

__device__ __forceinline__ ActCodes act_codes_of(const canon::A4Block& a) {
    ActCodes out;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        std::uint32_t w = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) { w |= e2m1_code_of_x2(a.c2[4 * q + j]) << (8 * j); }
        out.dup[q] = dup_nibbles(w);
    }
    out.scale = a.scale_scaled;
    return out;
}

// The four bytes of a word converted from E2M1 pairs to half2 (low nibble in .x).
__device__ __forceinline__ void e2m1x2_word_to_half2(std::uint32_t word, std::uint32_t (&h)[4]) {
    asm("{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
        "mov.b32 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n\t"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n\t"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n\t"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n\t}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(word));
}

// Lane l = 8q + i's integer shares (P_q of rows i and i + 8) of one unit against one activation word.
__device__ __forceinline__ void unit_shares(std::uint32_t code_word, std::uint32_t act_word, int& lo, int& hi) {
    std::uint32_t w[4], a[4];
    e2m1x2_word_to_half2(code_word, w);
    e2m1x2_word_to_half2(act_word, a);
    __half2 acc = __hmul2(*reinterpret_cast<const __half2*>(&w[0]), *reinterpret_cast<const __half2*>(&a[0]));
#pragma unroll
    for (int j = 1; j < 4; ++j) {
        acc = __hfma2(*reinterpret_cast<const __half2*>(&w[j]), *reinterpret_cast<const __half2*>(&a[j]), acc);
    }
    const __half2 four = __hmul2(acc, __float2half2_rn(4.0F));
    lo = __half2int_rn(__low2half(four));
    hi = __half2int_rn(__high2half(four));
}

// Compact activations (the column pairs below and the two-term kernels): a 16-element block's E2M1
// codes with element j at nibble j % 8 of word j / 8.

// Lane q's activation word of a block (ActCodes::dup's layout): byte j holds the code of element
// 4q + j in both nibbles. One byte permute spreads the lane's two code bytes, two masks split their
// nibbles and the shifts duplicate them.
__device__ __forceinline__ std::uint32_t compact_act_word(const std::uint32_t (&codes)[2], int q) {
    const std::uint32_t v  = __byte_perm(codes[q >> 1], 0U, (q & 1) != 0 ? 0x3322U : 0x1100U);
    const std::uint32_t lo = v & 0x000F000FU, hi = v & 0xF000F000U;
    return lo | (lo << 4) | hi | (hi >> 4);
}

// Four times the half2 values of an activation word: exact (at most 24), and a lane's four-term sums
// of products with the weights' E2M1 values are then integers of at most 576, exact in FP16, equal to
// unit_shares's 4 * acc.
__device__ __forceinline__ void act_half2_x4(std::uint32_t word, __half2 (&a)[4]) {
    std::uint32_t h[4];
    e2m1x2_word_to_half2(word, h);
    const __half2 four = __float2half2_rn(4.0F);
#pragma unroll
    for (int j = 0; j < 4; ++j) { a[j] = __hmul2(*reinterpret_cast<const __half2*>(&h[j]), four); }
}

// A lane's integer shares (rows i and i + 8) of one converted weight word against one activation
// word converted by act_half2_x4: unit_shares's integers.
__device__ __forceinline__ void term_shares(const std::uint32_t (&w)[4], const __half2 (&a)[4], int& lo, int& hi) {
    __half2 acc = __hmul2(*reinterpret_cast<const __half2*>(&w[0]), a[0]);
#pragma unroll
    for (int j = 1; j < 4; ++j) { acc = __hfma2(*reinterpret_cast<const __half2*>(&w[j]), a[j], acc); }
    lo = __half2int_rn(__low2half(acc));
    hi = __half2int_rn(__high2half(acc));
}

// Lanes 0..15 of each row group's int64 sums: rows i (lo) and i + 8 (hi) per lane, the four lanes
// of a row reduced, as unit_sums leaves them.
template <int Groups, int Columns>
__device__ __forceinline__ void reduce_row_sums(std::int64_t (&lo)[Groups][Columns], std::int64_t (&hi)[Groups][Columns],
                                                std::int64_t (&s)[Groups][Columns]) {
    const int lane = static_cast<int>(threadIdx.x) % 32;
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
#pragma unroll
            for (int offset = 8; offset <= 16; offset <<= 1) {
                lo[g][c] += __shfl_xor_sync(0xFFFFFFFFU, lo[g][c], offset);
                hi[g][c] += __shfl_xor_sync(0xFFFFFFFFU, hi[g][c], offset);
            }
            s[g][c] = (lane & 8) != 0 ? hi[g][c] : lo[g][c];
        }
    }
}

// Exact sums of one column against `Groups` row groups: this warp's units over blocks b = warp,
// warp + kWarps, ... of every group (the groups of a block share its activation word), the int64
// sums of rows i and i + 8 per lane, then the four lanes of a row reduced so that lanes 0..15 hold
// row (lane & 15) of each group.
template <int Groups>
__device__ void unit_sums(const std::uint8_t* matrix, int blocks, const ActCodes* acts, std::int64_t (&s)[Groups]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int q = lane >> 3, i = lane & 7;
    std::int64_t lo[Groups], hi[Groups];
#pragma unroll
    for (int g = 0; g < Groups; ++g) { lo[g] = hi[g] = 0; }
    for (int b = warp; b < blocks; b += kWarps) {
        const ActCodes& a = acts[b];
        const std::uint32_t act_word = a.dup[q];
        const std::int32_t sa        = a.scale;
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(g) * blocks + b) * moe::kUnitBytes;
            const std::uint32_t code_word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * q + 4 * i);
            int p_lo, p_hi;
            unit_shares(code_word, act_word, p_lo, p_hi);
            // |P_q| <= 576 and Sw < 2^18: P_q * Sw fits int32.
            lo[g] += static_cast<std::int64_t>(p_lo * canon::e4m3_scaled(unit[128 + i])) * sa;
            hi[g] += static_cast<std::int64_t>(p_hi * canon::e4m3_scaled(unit[136 + i])) * sa;
        }
    }
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int offset = 8; offset <= 16; offset <<= 1) {
            lo[g] += __shfl_xor_sync(0xFFFFFFFFU, lo[g], offset);
            hi[g] += __shfl_xor_sync(0xFFFFFFFFU, hi[g], offset);
        }
        s[g] = (lane & 8) != 0 ? hi[g] : lo[g];
    }
}

// Quantizes column x[:, t] of entry `first` to A4 codes of blocks [0, blocks) with `scale` into smem.
__device__ void quantize_column_codes(const bf16* __restrict__ x, int hidden, const std::int32_t* entries, int first,
                                      int top_k, float scale, ActCodes* acts, int blocks) {
    const int t = entries[first] / top_k;
    for (int b = threadIdx.x; b < blocks; b += blockDim.x) {
        const auto* v = reinterpret_cast<const std::uint16_t*>(x + static_cast<std::size_t>(t) * hidden + 16 * b);
        const canon::A4Codes c = canon::quantize_a4_codes(v, scale);
        ActCodes out;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const std::uint32_t w = static_cast<std::uint32_t>(c.code[4 * q]) |
                                    (static_cast<std::uint32_t>(c.code[4 * q + 1]) << 8) |
                                    (static_cast<std::uint32_t>(c.code[4 * q + 2]) << 16) |
                                    (static_cast<std::uint32_t>(c.code[4 * q + 3]) << 24);
            out.dup[q] = dup_nibbles(w);
        }
        out.scale = canon::e4m3_scaled(c.scale_word);
        acts[b]   = out;
    }
}

// ------------------------------------------- column pairs (temporary INFERNIX_TMP_NARROW_PAIRS)
//
// A job of two or more columns runs them two at a time: each unit's weight word and scales are
// converted once for both columns, and a pair shares one quantize step, one walk of the staged
// weights and its barriers (candidates §0j-1; the eight-column form removed on 2026-10-11 lost on
// dp4a products, one CTA per SM and eight columns' registers at any width). The pair's activations
// are compact (ActCompact), so a gate/up CTA holding a pair still fits two per SM, and its products
// are term_shares's (activations times four): unit_shares's integers, so every output keeps its bits.
// An odd last column runs alone in the same form.

// One column's A4 block, compact: the E2M1 code of element j at nibble j % 8 of codes[j / 8], and the
// block's e4m3_scaled scale.
struct ActCompact {
    std::uint32_t codes[2];
    std::int32_t scale;
};

// Element j's E2M1 code is code(j).
template <class Code>
__device__ __forceinline__ ActCompact compact_block(Code code, std::int32_t scale) {
    ActCompact out;
    std::uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        lo |= static_cast<std::uint32_t>(code(j)) << (4 * j);
        hi |= static_cast<std::uint32_t>(code(8 + j)) << (4 * j);
    }
    out.codes[0] = lo;
    out.codes[1] = hi;
    out.scale    = scale;
    return out;
}

// Quantizes the n <= 2 columns of entries first, first + 1 to compact A4 blocks [0, blocks) with
// `scale`: column c's block b into acts[c * blocks + b].
__device__ void quantize_pair_codes(const bf16* __restrict__ x, int hidden, const std::int32_t* entries, int first,
                                    int n, int top_k, float scale, ActCompact* acts, int blocks) {
    for (int i = static_cast<int>(threadIdx.x); i < n * blocks; i += blockDim.x) {
        const int c = i / blocks, b = i % blocks;
        const int t = entries[first + c] / top_k;
        const auto* v = reinterpret_cast<const std::uint16_t*>(x + static_cast<std::size_t>(t) * hidden + 16 * b);
        const canon::A4Codes codes = canon::quantize_a4_codes(v, scale);
        acts[i] = compact_block([&](int j) { return codes.code[j]; }, canon::e4m3_scaled(codes.scale_word));
    }
}

// unit_sums for the n <= 2 columns of a pair (column c's blocks at acts[c * blocks]): s[g][c] as
// unit_sums leaves s[g].
template <int Groups>
__device__ void unit_sums_pair(const std::uint8_t* matrix, int blocks, const ActCompact* acts, int n,
                               std::int64_t (&s)[Groups][2]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int q = lane >> 3, i = lane & 7;
    std::int64_t lo[Groups][2], hi[Groups][2];
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int c = 0; c < 2; ++c) { lo[g][c] = hi[g][c] = 0; }
    }
    for (int b = warp; b < blocks; b += kWarps) {
        __half2 a[2][4];
        std::int32_t sa[2] = {0, 0};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (c < n) { // n is uniform across the CTA
                const ActCompact& x = acts[c * blocks + b];
                act_half2_x4(compact_act_word(x.codes, q), a[c]);
                sa[c] = x.scale;
            }
        }
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(g) * blocks + b) * moe::kUnitBytes;
            std::uint32_t w[4];
            e2m1x2_word_to_half2(*reinterpret_cast<const std::uint32_t*>(unit + 32 * q + 4 * i), w);
            const std::int32_t sw_lo = canon::e4m3_scaled(unit[128 + i]), sw_hi = canon::e4m3_scaled(unit[136 + i]);
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                if (c < n) {
                    int p_lo, p_hi;
                    term_shares(w, a[c], p_lo, p_hi);
                    // |P_q| <= 576 and Sw < 2^18: P_q * Sw fits int32 (unit_sums's terms).
                    lo[g][c] += static_cast<std::int64_t>(p_lo * sw_lo) * sa[c];
                    hi[g][c] += static_cast<std::int64_t>(p_hi * sw_hi) * sa[c];
                }
            }
        }
    }
    reduce_row_sums<Groups, 2>(lo, hi, s);
}

// Whether jobs of two or more columns take the column pairs (TEMPORARY, INFERNIX_TMP_NARROW_PAIRS,
// until the verify-round A/B). The kernels take it as a template parameter, so without it they are
// the one-column kernels exactly (registers and occupancy included).
bool narrow_pairs() {
    static const bool enabled = temporary_switches::flag("INFERNIX_TMP_NARROW_PAIRS");
    return enabled;
}

constexpr int kGateUpSliceBytes = kGateUpGroups * moe::kGateUpBlocks * static_cast<int>(moe::kUnitBytes); // 46,080
constexpr int kDownSliceBytes   = kDownGroups * moe::kDownBlocks * static_cast<int>(moe::kUnitBytes);     // 23,040
static_assert(kGateUpSliceBytes % 16 == 0 && kDownSliceBytes % 16 == 0);

struct GateUpShared {
    alignas(16) std::uint8_t stage[kGateUpSliceBytes];
    // The activations are read only by unit_sums (or unit_sums_pair) and the partial sums are written
    // only after the barrier that follows it, so they share storage. That keeps the CTA (50,112 B)
    // under half of an SM's 100 KB, so two of them fit per SM.
    union {
        ActCodes acts[moe::kGateUpBlocks];
        ActCompact pair[2 * moe::kGateUpBlocks];
        std::int64_t partial[kWarps][16 * kGateUpGroups];
    };
    std::uint16_t y[2][16 * kGateUpGroups]; // per column of a pair; one column uses y[0]
    std::uint16_t h[2][16];
};

static_assert(sizeof(GateUpShared) + 1024 <= 102400 / 2, "two gate/up CTAs must fit an SM");

// The column pairs of a gate/up CTA's job (count >= 2 columns), as the one-column loop below computes
// each column.
__device__ void gate_up_pairs(const bf16* __restrict__ x, int hidden, const MoeDispatch& dispatch, int top_k,
                              const moe::ExpertScales& scales, int first, int count, int slice,
                              canon::A4Block* __restrict__ h_blocks, GateUpShared& sm) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const bool split_input = scales.input_up != scales.input_gate;
    for (int c0 = 0; c0 < count; c0 += 2) {
        const int n = min(2, count - c0);
        for (int parity = 0; parity < (split_input ? 2 : 1); ++parity) {
            __syncthreads();
            const float input_scale = parity == 0 ? scales.input_gate : scales.input_up;
            quantize_pair_codes(x, hidden, dispatch.entries, first + c0, n, top_k, input_scale, sm.pair,
                                moe::kGateUpBlocks);
            stage_wait();
            pdl::trigger_dependents();
            __syncthreads();
            std::int64_t s[kGateUpGroups][2];
            unit_sums_pair<kGateUpGroups>(sm.stage, moe::kGateUpBlocks, sm.pair, n, s);
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                if (c < n) {
                    __syncthreads(); // every warp has read sm.pair, or the previous column's partial sums
                    if (lane < 16) {
#pragma unroll
                        for (int g = 0; g < kGateUpGroups; ++g) { sm.partial[warp][g * 16 + lane] = s[g][c]; }
                    }
                    __syncthreads();
                    for (int row = threadIdx.x; row < 16 * kGateUpGroups; row += blockDim.x) {
                        const bool gate = (row & 1) == 0;
                        if (split_input && gate != (parity == 0)) { continue; }
                        std::int64_t total = 0;
                        for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row]; }
                        sm.y[c][row] = canon::a4_row_output(total, gate ? scales.alpha_gate : scales.alpha_up);
                    }
                }
            }
        }
        __syncthreads();
        for (int k = threadIdx.x; k < 16 * n; k += blockDim.x) {
            sm.h[k / 16][k % 16] = canon::swiglu_bf16(sm.y[k / 16][2 * (k % 16)], sm.y[k / 16][2 * (k % 16) + 1]);
        }
        __syncthreads();
        if (static_cast<int>(threadIdx.x) < n) {
            const int c = static_cast<int>(threadIdx.x);
            h_blocks[static_cast<std::size_t>(first + c0 + c) * kHBlocks + slice] =
                canon::quantize_a4_block(sm.h[c], scales.input_down);
        }
    }
}

template <bool Pairs>
__global__ void __launch_bounds__(kThreads, 2)
    gate_up_kernel(const bf16* __restrict__ x, int hidden, MoeDispatch dispatch,
                   MoeExpertSource source, int top_k, const std::uint8_t* const* __restrict__ job_records,
                   const std::int32_t* __restrict__ cpu_flags, int job_base, canon::A4Block* __restrict__ h_blocks,
                   PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm       = *reinterpret_cast<GateUpShared*>(smem_raw);
    const int job  = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const int slice  = blockIdx.x; // row groups 2*slice, 2*slice+1; h block `slice`
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const moe::ExpertScales scales = source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (moe::wide_route(count, scales)) { return; }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const bool split_input = scales.input_up != scales.input_gate;
    stage_async(record + static_cast<std::size_t>(kGateUpGroups * slice) * moe::kGateUpBlocks * moe::kUnitBytes,
                sm.stage, kGateUpSliceBytes);
    if constexpr (Pairs) {
        if (count > 1) {
            gate_up_pairs(x, hidden, dispatch, top_k, scales, first, count, slice, h_blocks, sm);
            return;
        }
    }

    for (int c = 0; c < count; ++c) { // one pass per column of the job
        // Gate rows are even, up rows odd; with distinct input scales the up rows use their own A4.
        for (int parity = 0; parity < (split_input ? 2 : 1); ++parity) {
            __syncthreads();
            const float input_scale = parity == 0 ? scales.input_gate : scales.input_up;
            quantize_column_codes(x, hidden, dispatch.entries, first + c, top_k, input_scale, sm.acts,
                                  moe::kGateUpBlocks);
            stage_wait();
            // The down pass (a programmatic dependent in captured graphs) may launch once every CTA has its
            // slice: its CTAs then fetch their weights while this pass computes and finishes.
            pdl::trigger_dependents();
            __syncthreads();
            std::int64_t s[kGateUpGroups];
            unit_sums<kGateUpGroups>(sm.stage, moe::kGateUpBlocks, sm.acts, s);
            __syncthreads(); // every warp has read sm.acts before sm.partial overwrites it
            if (lane < 16) {
#pragma unroll
                for (int g = 0; g < kGateUpGroups; ++g) { sm.partial[warp][g * 16 + lane] = s[g]; }
            }
            __syncthreads();
            for (int row = threadIdx.x; row < 16 * kGateUpGroups; row += blockDim.x) {
                const bool gate = (row & 1) == 0;
                if (split_input && gate != (parity == 0)) { continue; }
                std::int64_t total = 0;
                for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row]; }
                sm.y[0][row] = canon::a4_row_output(total, gate ? scales.alpha_gate : scales.alpha_up);
            }
        }
        __syncthreads();
        for (int k = threadIdx.x; k < 16; k += blockDim.x) {
            sm.h[0][k] = canon::swiglu_bf16(sm.y[0][2 * k], sm.y[0][2 * k + 1]);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            h_blocks[static_cast<std::size_t>(first + c) * kHBlocks + slice] =
                canon::quantize_a4_block(sm.h[0], scales.input_down);
        }
    }
}

struct DownShared {
    alignas(16) std::uint8_t stage[kDownSliceBytes];
    union {
        ActCodes acts[kHBlocks];
        ActCompact pair[2 * kHBlocks];
    };
    std::int64_t partial[kWarps][16 * kDownGroups];
};

// The column pairs of a down CTA's job (count >= 2 columns), as the one-column loop below computes each
// column.
__device__ void down_pairs(const MoeDispatch& dispatch, int hidden, const moe::ExpertScales& scales, int first,
                           int count, int tile, const canon::A4Block* __restrict__ h_blocks, bf16* __restrict__ outputs,
                           DownShared& sm) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int c0 = 0; c0 < count; c0 += 2) {
        const int n = min(2, count - c0);
        __syncthreads();
        for (int i = threadIdx.x; i < n * kHBlocks; i += blockDim.x) {
            const canon::A4Block& a = h_blocks[static_cast<std::size_t>(first + c0) * kHBlocks + i];
            sm.pair[i] = compact_block([&](int j) { return e2m1_code_of_x2(a.c2[j]); }, a.scale_scaled);
        }
        stage_wait();
        __syncthreads();
        std::int64_t s[kDownGroups][2];
        unit_sums_pair<kDownGroups>(sm.stage, moe::kDownBlocks, sm.pair, n, s);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (c < n) {
                if (c > 0) { __syncthreads(); } // every thread has read the previous column's partial sums
                if (lane < 16) {
#pragma unroll
                    for (int g = 0; g < kDownGroups; ++g) { sm.partial[warp][g * 16 + lane] = s[g][c]; }
                }
                __syncthreads();
                const int column = dispatch.entries[first + c0 + c];
                for (int row = threadIdx.x; row < 16 * kDownGroups; row += blockDim.x) {
                    std::int64_t total = 0;
                    for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row]; }
                    reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * hidden +
                                                              16 * kDownGroups * tile + row] =
                        canon::a4_row_output(total, scales.alpha_down);
                }
            }
        }
    }
}

template <bool Pairs>
__global__ void __launch_bounds__(kThreads)
    down_kernel(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                int job_base, const canon::A4Block* __restrict__ h_blocks, int columns_out,
                bf16* __restrict__ outputs, PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<DownShared*>(smem_raw);
    const int job = job_base + static_cast<int>(blockIdx.y);
    const int tile = blockIdx.x; // down row groups 4*tile .. 4*tile+3
    // Launched as a programmatic dependent of the gate/up pass: everything read before the PDL wait (the
    // dispatch, the CPU flags, the job records, the frames and the weights) was written before that pass
    // started; only h_blocks, its output, is read after the wait. Every CTA waits, so this grid's
    // completion implies the gate/up pass's.
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    if (job < *dispatch.job_count && (cpu_flags == nullptr || cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(phase, source, expert, job_records, job);
    }
    moe::ExpertScales scales{};
    if (record != nullptr) {
        scales = source.scales[expert];
        first  = dispatch.offsets[expert];
        count  = dispatch.offsets[expert + 1] - first;
        if (moe::wide_route(count, scales)) { record = nullptr; }
    }
    if (record != nullptr) {
        stage_async(record + moe::kGateUpBytes +
                        static_cast<std::size_t>(kDownGroups * tile) * moe::kDownBlocks * moe::kUnitBytes,
                    sm.stage, kDownSliceBytes);
    }
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    if constexpr (Pairs) {
        if (count > 1) {
            down_pairs(dispatch, hidden, scales, first, count, tile, h_blocks, outputs, sm);
            return;
        }
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int c = 0; c < count; ++c) { // one pass per column of the job
        __syncthreads();
        for (int i = threadIdx.x; i < kHBlocks; i += blockDim.x) {
            sm.acts[i] = act_codes_of(h_blocks[static_cast<std::size_t>(first + c) * kHBlocks + i]);
        }
        stage_wait();
        __syncthreads();
        std::int64_t s[kDownGroups];
        unit_sums<kDownGroups>(sm.stage, moe::kDownBlocks, sm.acts, s);
        if (lane < 16) {
#pragma unroll
            for (int g = 0; g < kDownGroups; ++g) { sm.partial[warp][g * 16 + lane] = s[g]; }
        }
        __syncthreads();
        const int column = dispatch.entries[first + c];
        for (int row = threadIdx.x; row < 16 * kDownGroups; row += blockDim.x) {
            std::int64_t total = 0;
            for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row]; }
            const std::uint16_t y = canon::a4_row_output(total, scales.alpha_down);
            reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * hidden +
                                                      16 * kDownGroups * tile + row] = y;
        }
    }
    (void)columns_out;
}

template <bool Pairs>
void launch_expert_pass_as(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                           const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                           canon::A4Block* h_blocks, int columns_out, bf16* outputs, cudaStream_t stream,
                           PassPhase phase) {
    static const bool configured = [] {
        cudaFuncSetAttribute(gate_up_kernel<Pairs>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(GateUpShared)));
        cudaFuncSetAttribute(down_kernel<Pairs>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(DownShared)));
        cudaFuncSetAttribute(gate_up_kernel<Pairs>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(down_kernel<Pairs>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        return true;
    }();
    (void)configured;
    gate_up_kernel<Pairs><<<dim3(kGateUpCtas, jobs), kThreads, sizeof(GateUpShared), stream>>>(
        x, moe::kHidden, dispatch, source, top_k, job_records, flags, base, h_blocks, phase);
    check_launch("gate/up");
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_consumer({dim3(kDownCtas, jobs), dim3(kThreads), sizeof(DownShared), stream},
                                    down_kernel<Pairs>, dispatch, source, static_cast<int>(moe::kHidden), job_records,
                                    flags, base, static_cast<const canon::A4Block*>(h_blocks), columns_out, outputs,
                                    phase));
}

void launch_expert_pass(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                        const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                        canon::A4Block* h_blocks, int columns_out, bf16* outputs, cudaStream_t stream,
                        PassPhase phase = PassPhase::All) {
    if (narrow_pairs()) {
        launch_expert_pass_as<true>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_blocks, columns_out,
                                    outputs, stream, phase);
    } else {
        launch_expert_pass_as<false>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_blocks, columns_out,
                                     outputs, stream, phase);
    }
}

// ------------------------------------------------------------------------ A16 narrow route (§16.2.1)
//
// Experts stored without activation scales: x and h are encoded per column (canon::a16_value) and
// every product is three dp4a (signed codes times the unsigned lo and mid bytes and the signed hi
// bytes of X), so P_b and the int64 row sums are the CPU engine's bit for bit. Gate/up CTAs write
// h in BF16 to the entry's row of the workspace's h rows (wide_expert.h); down CTAs encode it under
// its own column exponent.

constexpr int kA16PassColumns = 4; // columns per pass of the multi-column A16 kernels
// Calls of at most this many columns take the one-column A16 kernels. They are faster only for an
// expert of one column (op level, frames, cold L2, uniform routing: 18-22 % at one column, 8-34 %
// slower at 2-8), and a call wider than four columns mostly holds experts of several (an MTP row's
// drafts route alike): C = 2 MTP decode measured 0.8 % faster at 4 than at 8 (C = 8 plain decode,
// independent rows, 1.7 % slower; C = 1 equal), 2026-10-11.
constexpr int kA16OneColumnCalls = 4;

// An encoded A16 block in the layout the dp4a loop reads: four elements' bytes per word.
struct A16Act {
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

// A 16-byte load, through L2 when the data was written by other CTAs of the running kernel.
template <bool kL2>
__device__ __forceinline__ uint4 load16(const uint4* p) {
    if constexpr (kL2) { return __ldcg(p); }
    return *p;
}

// The thread group a cooperative routine runs in: every thread of a kThreads-thread CTA.
struct CtaGroup {
    static constexpr int kThreads = ::infernix::ops::kThreads;
    __device__ static int thread() { return static_cast<int>(threadIdx.x); }
    __device__ static void sync() { __syncthreads(); }
};

// Encodes `n` <= Columns BF16 columns of `length` elements, column c at base(c), into acts
// [c][length / 16] with their exponents in emax[c]. Every thread of `Group` (kWarps warps) calls it;
// it ends with the group's barrier. kL2 reads the columns through L2 (written by other CTAs of this
// kernel).
template <int Columns, bool kL2 = false, class Group = CtaGroup, class Base>
__device__ void encode_a16_columns(Base base, int n, int length, A16Act* acts, int* emax,
                                   unsigned (&key)[kWarps][Columns]) {
    static_assert(Group::kThreads == kThreads, "the encode reduces over kWarps warps");
    const int tid  = Group::thread();
    const int lane = tid % 32, warp = tid / 32;
    const int vectors = length / 8; // uint4 loads of 8 elements
    unsigned m[Columns];
#pragma unroll
    for (int c = 0; c < Columns; ++c) { m[c] = 0; }
    for (int i = tid; i < n * vectors; i += Group::kThreads) {
        const int c   = i / vectors;
        const uint4 v = load16<kL2>(reinterpret_cast<const uint4*>(base(c)) + i % vectors);
        const unsigned words[4] = {v.x, v.y, v.z, v.w};
        unsigned mm = 0;
#pragma unroll
        for (int w = 0; w < 4; ++w) { mm = max(mm, max(words[w] & 0x7FFFU, (words[w] >> 16) & 0x7FFFU)); }
#pragma unroll
        for (int cc = 0; cc < Columns; ++cc) {
            if (cc == c) { m[cc] = max(m[cc], mm); }
        }
    }
#pragma unroll
    for (int c = 0; c < Columns; ++c) {
        m[c] = __reduce_max_sync(0xFFFFFFFFU, m[c]);
        if (lane == 0) { key[warp][c] = m[c]; }
    }
    Group::sync();
    if (tid < n) {
        unsigned k = 0;
        for (int w = 0; w < kWarps; ++w) { k = max(k, key[w][tid]); }
        emax[tid] = k >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(k >> 7), 1);
    }
    Group::sync();
    const int blocks = length / 16;
    for (int i = tid; i < n * blocks; i += Group::kThreads) {
        const int c = i / blocks, b = i % blocks;
        const int e = emax[c];
        const auto* src = reinterpret_cast<const uint4*>(base(c) + 16 * b);
        const uint4 v0 = load16<kL2>(src), v1 = load16<kL2>(src + 1);
        const unsigned words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        A16Act out{};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const auto h  = static_cast<std::uint16_t>(j % 2 == 0 ? words[j / 2] & 0xFFFFU : words[j / 2] >> 16);
            const auto xv = static_cast<std::uint32_t>(e == canon::kA16NonFinite ? 0 : canon::a16_value(h, e));
            const int q = j / 4, s = 8 * (j % 4);
            out.lo[q] |= (xv & 0xFFU) << s;
            out.mid[q] |= ((xv >> 8) & 0xFFU) << s;
            out.hi[q] = static_cast<std::int32_t>(static_cast<std::uint32_t>(out.hi[q]) | (((xv >> 16) & 0xFFU) << s));
        }
        acts[c * blocks + b] = out;
    }
    Group::sync();
}

// Exact sums S[g][c] of row (lane & 15) of `Groups` row groups, A16 activations (§16.2.1). P is
// assembled in wrapping arithmetic: its true value fits int32.
template <int Groups, int Columns>
__device__ void unit_sums_a16(const std::uint8_t* matrix, int blocks, const A16Act* acts, int n,
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
        const std::uint8_t* unit = matrix + (static_cast<std::size_t>(g) * blocks + b) * moe::kUnitBytes;
        const int w0 = row_quad(unit, r, 2 * half);
        const int w1 = row_quad(unit, r, 2 * half + 1);
        const int sw = canon::e4m3_scaled(unit[128 + r]);
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (c < n) { // n is uniform across the warp
                const A16Act& a = acts[c * blocks + b];
                const int lo  = dp4a_su(w1, a.lo[2 * half + 1], dp4a_su(w0, a.lo[2 * half], 0));
                const int mid = dp4a_su(w1, a.mid[2 * half + 1], dp4a_su(w0, a.mid[2 * half], 0));
                const int hi  = __dp4a(w1, a.hi[2 * half + 1], __dp4a(w0, a.hi[2 * half], 0));
                unsigned p = static_cast<unsigned>(lo) + (static_cast<unsigned>(mid) << 8) + (static_cast<unsigned>(hi) << 16);
                p += static_cast<unsigned>(__shfl_xor_sync(0xFFFFFFFFU, static_cast<int>(p), 16));
                const std::int64_t term = static_cast<std::int64_t>(static_cast<int>(p)) * sw;
#pragma unroll
                for (int gg = 0; gg < Groups; ++gg) {
                    if (gg == g) { s[gg][c] += term; }
                }
            }
        }
    }
}

// A gate/up CTA's slice (kGateUpGroups row groups) and a down CTA's (kDownGroups) of geometry G.
template <class G>
inline constexpr int kGateUpSliceBytesOf = kGateUpGroups * G::kGateUpBlocks * static_cast<int>(moe::kUnitBytes);
template <class G>
inline constexpr int kDownSliceBytesOf = kDownGroups * G::kDownBlocks * static_cast<int>(moe::kUnitBytes);

template <class G, int Columns>
struct GateUpShared16 {
    alignas(16) std::uint8_t stage[kGateUpSliceBytesOf<G>];
    union { // acts are read only before the barrier that precedes the partial sums
        A16Act acts[Columns * G::kGateUpBlocks];
        std::int64_t partial[kWarps][16 * kGateUpGroups][Columns];
    };
    std::uint16_t y[Columns][16 * kGateUpGroups];
    int emax[Columns];
    unsigned key[kWarps][Columns];
};

template <class G, int Columns>
__global__ void __launch_bounds__(kThreads)
    gate_up_kernel_a16(const bf16* __restrict__ x, int hidden, MoeDispatch dispatch, MoeExpertSource source,
                       int top_k, const std::uint8_t* const* __restrict__ job_records,
                       const std::int32_t* __restrict__ cpu_flags, int job_base, std::uint16_t* __restrict__ h_plane,
                       PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<GateUpShared16<G, Columns>*>(smem_raw);
    const int job = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const int slice  = blockIdx.x; // row groups 2*slice, 2*slice+1; h block `slice`
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const moe::ExpertScales scales = source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (moe::wide_route(count, scales)) { return; }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    stage_async(record + static_cast<std::size_t>(kGateUpGroups * slice) * G::kGateUpBlocks * moe::kUnitBytes,
                sm.stage, kGateUpSliceBytesOf<G>);
    for (int pass = 0; pass < count; pass += Columns) {
        const int n = min(Columns, count - pass);
        __syncthreads();
        const auto column = [&](int c) {
            return reinterpret_cast<const std::uint16_t*>(x) +
                   static_cast<std::size_t>(moe::wide::entry_column(dispatch.entries[first + pass + c], top_k,
                                                                    dispatch.shared_first)) *
                       hidden;
        };
        encode_a16_columns<Columns>(column, n, G::kHidden, sm.acts, sm.emax, sm.key);
        stage_wait();
        // The down pass (a programmatic dependent in captured graphs) may launch once every CTA has
        // its slice.
        pdl::trigger_dependents();
        __syncthreads();
        std::int64_t s[kGateUpGroups][Columns];
        unit_sums_a16<kGateUpGroups, Columns>(sm.stage, G::kGateUpBlocks, sm.acts, n, s);
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
            std::int64_t total = 0;
            for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row][c]; }
            sm.y[c][row] = canon::a16_row_output(total, sm.emax[c], gate ? scales.alpha_gate : scales.alpha_up);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 16 * n; i += blockDim.x) {
            const int k = i % 16, c = i / 16;
            h_plane[static_cast<std::size_t>(first + pass + c) * moe::wide::A16Planes<G>::kHRowElements + 16 * slice + k] =
                canon::swiglu_bf16(sm.y[c][2 * k], sm.y[c][2 * k + 1]);
        }
    }
}

template <class G, int Columns>
struct DownShared16 {
    alignas(16) std::uint8_t stage[kDownSliceBytesOf<G>];
    A16Act acts[Columns * G::kHBlocks];
    std::int64_t partial[kWarps][16 * kDownGroups][Columns];
    int emax[Columns];
    unsigned key[kWarps][Columns];
};

template <class G, int Columns>
__global__ void __launch_bounds__(kThreads)
    down_kernel_a16(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                    const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                    int job_base, const std::uint16_t* __restrict__ h_plane, bf16* __restrict__ outputs,
                    PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<DownShared16<G, Columns>*>(smem_raw);
    const int job = job_base + static_cast<int>(blockIdx.y);
    const int tile = blockIdx.x; // down row groups 4*tile .. 4*tile+3
    // A programmatic dependent of the gate/up pass, under down_kernel's contract: only h_plane is read
    // after the PDL wait, and every CTA waits.
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    if (job < *dispatch.job_count && (cpu_flags == nullptr || cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(phase, source, expert, job_records, job);
    }
    moe::ExpertScales scales{};
    if (record != nullptr) {
        scales = source.scales[expert];
        first  = dispatch.offsets[expert];
        count  = dispatch.offsets[expert + 1] - first;
        if (moe::wide_route(count, scales)) { record = nullptr; }
    }
    if (record != nullptr) {
        stage_async(record + G::kGateUpBytes +
                        static_cast<std::size_t>(kDownGroups * tile) * G::kDownBlocks * moe::kUnitBytes,
                    sm.stage, kDownSliceBytesOf<G>);
    }
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int pass = 0; pass < count; pass += Columns) {
        const int n = min(Columns, count - pass);
        __syncthreads();
        const auto column = [&](int c) {
            return h_plane + static_cast<std::size_t>(first + pass + c) * moe::wide::A16Planes<G>::kHRowElements;
        };
        encode_a16_columns<Columns>(column, n, G::kIntermediate, sm.acts, sm.emax, sm.key);
        stage_wait();
        __syncthreads();
        std::int64_t s[kDownGroups][Columns];
        unit_sums_a16<kDownGroups, Columns>(sm.stage, G::kDownBlocks, sm.acts, n, s);
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
            reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * hidden + 16 * kDownGroups * tile + row] =
                canon::a16_row_output(total, sm.emax[c], scales.alpha_down);
        }
    }
}

// One pass of the A16 narrow kernels at geometry G: a gate/up CTA per 16 intermediates (one h block)
// and a down CTA per 64 output rows of each job. At Qwen3.5-122B-A10B's geometry the four-column
// gate/up CTA holds its 55,296-byte slice and 36,864 bytes of encoded x (one CTA per SM, as the
// Flash-Next four-column CTA's 77 KB).
template <class G, int Columns>
void launch_expert_pass_a16(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                            const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                            std::uint16_t* h_plane, bf16* outputs, cudaStream_t stream,
                            PassPhase phase = PassPhase::All) {
    constexpr int kGateUpBytes = static_cast<int>(sizeof(GateUpShared16<G, Columns>));
    constexpr int kDownBytes   = static_cast<int>(sizeof(DownShared16<G, Columns>));
    static_assert(kGateUpBytes <= 99 * 1024 && kDownBytes <= 99 * 1024, "an A16 CTA must fit an SM");
    static_assert(G::kGateUpRowGroups / kGateUpGroups == G::kHBlocks, "a gate/up CTA produces one h block");
    static const bool configured = [] {
        cudaFuncSetAttribute(gate_up_kernel_a16<G, Columns>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kGateUpBytes);
        cudaFuncSetAttribute(down_kernel_a16<G, Columns>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDownBytes);
        cudaFuncSetAttribute(gate_up_kernel_a16<G, Columns>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(down_kernel_a16<G, Columns>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        return true;
    }();
    (void)configured;
    gate_up_kernel_a16<G, Columns>
        <<<dim3(G::kGateUpRowGroups / kGateUpGroups, jobs), kThreads, kGateUpBytes, stream>>>(
            x, G::kHidden, dispatch, source, top_k, job_records, flags, base, h_plane, phase);
    check_launch("A16 gate/up");
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(G::kDownRowGroups / kDownGroups, jobs), dim3(kThreads), static_cast<std::size_t>(kDownBytes), stream},
        down_kernel_a16<G, Columns>, dispatch, source, static_cast<int>(G::kHidden), job_records, flags, base,
        static_cast<const std::uint16_t*>(h_plane), outputs, phase));
}

// ------------------------------------------- two-term A4 narrow route (expert-a4-residual-decode)
//
// NVFP4 W4A16 experts under NarrowA16::kA4Residual (canonical_math.h, a4r_*; design §19.3.23): a
// gate/up CTA encodes its column's x as two NVFP4 terms per block under the column's row scale, and
// writes h in BF16 to the entry's row of the workspace's h rows as the A16 route does; a down CTA
// encodes that h row the same way (its row scale spans all 640 elements, so no gate/up CTA can). The
// products are the W4A4 kernels' half2 unit products, both terms against the same converted weight
// word, and every block sum is the exact (P_1 Sa_1 + P_2 Sa_2) Sw in int64: the CPU engine's S
// (a4r_expert.h) bit for bit, so a CPU-served miss changes no output. As the W4A4 kernels, one pass
// per column over the staged weights (or per pair of columns under INFERNIX_TMP_NARROW_PAIRS), two
// gate/up CTAs per SM.
//
// Per 64-element unit run a lane does one weight-word conversion and two term products (eight HFMA2,
// four conversions to int, two int32 and two int64 multiply-adds) against the W4A4 kernels' one;
// the activations' conversions are made once per block and shared by its row groups.

constexpr int kA4rTerms = canon::kA4ResidualTerms;
static_assert(kA4rTerms == 2, "the unit products are written for two terms");

// One 16-element block of a row as its terms: term t's E2M1 codes (element j at nibble j % 8 of word
// j / 8) and its e4m3_scaled scale.
struct A4rAct {
    std::uint32_t codes[kA4rTerms][2];
    std::int32_t scale[kA4rTerms];
};
static_assert(sizeof(A4rAct) == 24);

// The 16 BF16 values of a block at v (16-byte aligned) into e; returns their largest magnitude key.
__device__ __forceinline__ unsigned load_a4r_block(const std::uint16_t* __restrict__ v, std::uint16_t (&e)[16]) {
    const auto* src = reinterpret_cast<const uint4*>(v);
    const uint4 v0 = src[0], v1 = src[1];
    const unsigned words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    unsigned key = 0;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        e[2 * w]     = static_cast<std::uint16_t>(words[w] & 0xFFFFU);
        e[2 * w + 1] = static_cast<std::uint16_t>(words[w] >> 16);
        key          = max(key, max(words[w] & 0x7FFFU, (words[w] >> 16) & 0x7FFFU));
    }
    return key;
}

// A block's two terms under row scale rs (canon::a4r_block_terms).
__device__ __forceinline__ A4rAct encode_a4r_block(const std::uint16_t (&e)[16], float rs) {
    canon::A4Codes terms[kA4rTerms];
    canon::a4r_block_terms<kA4rTerms>(e, canon::a4r_multiplier(rs), terms);
    A4rAct out;
#pragma unroll
    for (int t = 0; t < kA4rTerms; ++t) {
        std::uint32_t lo = 0, hi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            lo |= static_cast<std::uint32_t>(terms[t].code[j]) << (4 * j);
            hi |= static_cast<std::uint32_t>(terms[t].code[8 + j]) << (4 * j);
        }
        out.codes[t][0] = lo;
        out.codes[t][1] = hi;
        out.scale[t]    = canon::e4m3_scaled(terms[t].scale_word);
    }
    return out;
}

// Encodes the BF16 row v of Blocks 16-element blocks into acts and its row scale: thread b loads and
// encodes block b (canon::a4r_block_terms), after a CTA-wide maximum of the row's magnitude keys.
// Every thread of the CTA calls it; it ends with a barrier.
template <int Blocks>
__device__ void encode_a4r_row(const std::uint16_t* __restrict__ v, A4rAct* acts, unsigned* keys, float* row_scale) {
    static_assert(Blocks <= kThreads);
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    std::uint16_t e[16] = {};
    unsigned key = 0;
    if (tid < Blocks) { key = load_a4r_block(v + 16 * tid, e); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (lane == 0) { keys[warp] = key; }
    __syncthreads();
    unsigned all = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) { all = max(all, keys[w]); }
    const float rs = canon::a4r_row_scale(all);
    if (tid == 0) { *row_scale = rs; }
    if (tid < Blocks) { acts[tid] = encode_a4r_block(e, rs); }
    __syncthreads();
}

// unit_sums for two-term activations: lanes 0..15 return row (lane & 15) of each of the Groups row
// groups, S = sum_b (P_1 Sa_1 + P_2 Sa_2) Sw over this warp's blocks, reduced as unit_sums does.
template <int Groups>
__device__ void unit_sums_a4r(const std::uint8_t* matrix, int blocks, const A4rAct* acts, std::int64_t (&s)[Groups]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int q = lane >> 3, i = lane & 7;
    std::int64_t lo[Groups], hi[Groups];
#pragma unroll
    for (int g = 0; g < Groups; ++g) { lo[g] = hi[g] = 0; }
    for (int b = warp; b < blocks; b += kWarps) {
        const A4rAct& a = acts[b];
        __half2 x0[4], x1[4];
        act_half2_x4(compact_act_word(a.codes[0], q), x0);
        act_half2_x4(compact_act_word(a.codes[1], q), x1);
        const std::int32_t sa0 = a.scale[0], sa1 = a.scale[1];
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(g) * blocks + b) * moe::kUnitBytes;
            std::uint32_t w[4];
            e2m1x2_word_to_half2(*reinterpret_cast<const std::uint32_t*>(unit + 32 * q + 4 * i), w);
            int p0_lo, p0_hi, p1_lo, p1_hi;
            term_shares(w, x0, p0_lo, p0_hi);
            term_shares(w, x1, p1_lo, p1_hi);
            // |P| <= 576 and Sa < 2^18: P_1 Sa_1 + P_2 Sa_2 < 2^28 fits int32.
            lo[g] += static_cast<std::int64_t>(p0_lo * sa0 + p1_lo * sa1) * canon::e4m3_scaled(unit[128 + i]);
            hi[g] += static_cast<std::int64_t>(p0_hi * sa0 + p1_hi * sa1) * canon::e4m3_scaled(unit[136 + i]);
        }
    }
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int offset = 8; offset <= 16; offset <<= 1) {
            lo[g] += __shfl_xor_sync(0xFFFFFFFFU, lo[g], offset);
            hi[g] += __shfl_xor_sync(0xFFFFFFFFU, hi[g], offset);
        }
        s[g] = (lane & 8) != 0 ? hi[g] : lo[g];
    }
}

// Column pairs of the two-term kernels (INFERNIX_TMP_NARROW_PAIRS; the W4A4 kernels' "column pairs"
// above). A pair's row scales come first, over every block of both rows; its blocks are then encoded
// a range at a time: a gate/up CTA stages each half of K for both columns (2 x 80 blocks) in the
// storage of one column's 160, so two CTAs still fit an SM, and a down CTA both columns' 40 blocks.

// The row scales of the n <= 2 rows row(c) of Blocks blocks each into row_scale[c], as encode_a4r_row
// computes one. Every thread of the CTA calls it; it ends with a barrier.
template <int Blocks, class Row>
__device__ void a4r_pair_row_scales(Row row, int n, unsigned (&keys)[kWarps][2], float (&row_scale)[2]) {
    const int tid = static_cast<int>(threadIdx.x), lane = tid % 32, warp = tid / 32;
    unsigned key[2] = {0, 0};
    for (int i = tid; i < n * Blocks; i += kThreads) {
        const int c = i / Blocks;
        std::uint16_t e[16];
        const unsigned k = load_a4r_block(row(c) + 16 * (i % Blocks), e);
#pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
            if (cc == c) { key[cc] = max(key[cc], k); }
        }
    }
#pragma unroll
    for (int c = 0; c < 2; ++c) {
        key[c] = __reduce_max_sync(0xFFFFFFFFU, key[c]);
        if (lane == 0) { keys[warp][c] = key[c]; }
    }
    __syncthreads();
    if (tid < n) {
        unsigned all = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { all = max(all, keys[w][tid]); }
        row_scale[tid] = canon::a4r_row_scale(all);
    }
    __syncthreads();
}

// Encodes blocks [b0, b0 + len) of the n <= 2 rows row(c) under row_scale[c]: column c's block b into
// acts[c * len + b - b0]. No barrier.
template <class Row>
__device__ void encode_a4r_pair_blocks(Row row, int n, int b0, int len, const float (&row_scale)[2], A4rAct* acts) {
    for (int i = static_cast<int>(threadIdx.x); i < n * len; i += kThreads) {
        const int c = i / len;
        std::uint16_t e[16];
        load_a4r_block(row(c) + 16 * (b0 + i % len), e);
        acts[i] = encode_a4r_block(e, row_scale[c]);
    }
}

// unit_sums_a4r for the n <= 2 columns of a pair over blocks [b0, b0 + len) (column c's block b at
// acts[c * len + b - b0]), added to lo and hi (rows i and i + 8 of each lane) for reduce_row_sums.
template <int Groups>
__device__ void unit_sums_a4r_pair(const std::uint8_t* matrix, int blocks, int b0, int len, const A4rAct* acts, int n,
                                   std::int64_t (&lo)[Groups][2], std::int64_t (&hi)[Groups][2]) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int q = lane >> 3, i = lane & 7;
    for (int k = warp; k < len; k += kWarps) {
        const int b = b0 + k;
        __half2 a[2][kA4rTerms][4];
        std::int32_t sa[2][kA4rTerms] = {};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (c < n) { // n is uniform across the CTA
                const A4rAct& x = acts[c * len + k];
#pragma unroll
                for (int t = 0; t < kA4rTerms; ++t) {
                    act_half2_x4(compact_act_word(x.codes[t], q), a[c][t]);
                    sa[c][t] = x.scale[t];
                }
            }
        }
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(g) * blocks + b) * moe::kUnitBytes;
            std::uint32_t w[4];
            e2m1x2_word_to_half2(*reinterpret_cast<const std::uint32_t*>(unit + 32 * q + 4 * i), w);
            const std::int32_t sw_lo = canon::e4m3_scaled(unit[128 + i]), sw_hi = canon::e4m3_scaled(unit[136 + i]);
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                if (c < n) {
                    int p0_lo, p0_hi, p1_lo, p1_hi;
                    term_shares(w, a[c][0], p0_lo, p0_hi);
                    term_shares(w, a[c][1], p1_lo, p1_hi);
                    // As unit_sums_a4r: P_1 Sa_1 + P_2 Sa_2 < 2^28 fits int32.
                    lo[g][c] += static_cast<std::int64_t>(p0_lo * sa[c][0] + p1_lo * sa[c][1]) * sw_lo;
                    hi[g][c] += static_cast<std::int64_t>(p0_hi * sa[c][0] + p1_hi * sa[c][1]) * sw_hi;
                }
            }
        }
    }
}

struct GateUpSharedA4r {
    alignas(16) std::uint8_t stage[kGateUpSliceBytes];
    // As in GateUpShared: the activations (one column's blocks, or half of K of a pair's) are read only
    // by the unit sums and the partial sums written only after the barrier that follows them.
    union {
        A4rAct acts[moe::kGateUpBlocks];
        std::int64_t partial[kWarps][16 * kGateUpGroups];
    };
    std::uint16_t y[2][16 * kGateUpGroups]; // per column of a pair; one column uses y[0]
    unsigned keys[kWarps][2];
    float row_scale[2];
};

static_assert(sizeof(GateUpSharedA4r) + 1024 <= 102400 / 2, "two two-term gate/up CTAs must fit an SM");

// The column pairs of a two-term gate/up CTA's job (count >= 2 columns), as the one-column loop below
// computes each column.
__device__ void gate_up_pairs_a4r(const bf16* __restrict__ x, const MoeDispatch& dispatch, int top_k,
                                  const moe::ExpertScales& scales, int first, int count, int slice,
                                  std::uint16_t* __restrict__ h_plane, GateUpSharedA4r& sm) {
    constexpr int kHalf = moe::kGateUpBlocks / 2;
    static_assert(moe::kGateUpBlocks == 2 * kHalf, "a pair stages K in two halves");
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int c0 = 0; c0 < count; c0 += 2) {
        const int n   = min(2, count - c0);
        const auto row = [&](int c) {
            return reinterpret_cast<const std::uint16_t*>(x) +
                   static_cast<std::size_t>(moe::wide::entry_column(dispatch.entries[first + c0 + c], top_k,
                                                                    dispatch.shared_first)) *
                       moe::kHidden;
        };
        __syncthreads(); // every thread is past the previous pair's reads of sm.y and sm.row_scale
        a4r_pair_row_scales<moe::kGateUpBlocks>(row, n, sm.keys, sm.row_scale);
        std::int64_t lo[kGateUpGroups][2] = {}, hi[kGateUpGroups][2] = {};
        for (int half = 0; half < 2; ++half) {
            if (half > 0) { __syncthreads(); } // every warp has read the first half's activations
            encode_a4r_pair_blocks(row, n, half * kHalf, kHalf, sm.row_scale, sm.acts);
            if (half == 0) {
                stage_wait();
                // The down pass (a programmatic dependent in captured graphs) may launch once every CTA
                // has its slice.
                pdl::trigger_dependents();
            }
            __syncthreads();
            unit_sums_a4r_pair<kGateUpGroups>(sm.stage, moe::kGateUpBlocks, half * kHalf, kHalf, sm.acts, n, lo, hi);
        }
        std::int64_t s[kGateUpGroups][2];
        reduce_row_sums<kGateUpGroups, 2>(lo, hi, s);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (c < n) {
                __syncthreads(); // every warp has read sm.acts, or the previous column's partial sums
                if (lane < 16) {
#pragma unroll
                    for (int g = 0; g < kGateUpGroups; ++g) { sm.partial[warp][g * 16 + lane] = s[g][c]; }
                }
                __syncthreads();
                for (int r = threadIdx.x; r < 16 * kGateUpGroups; r += blockDim.x) {
                    const bool gate    = (r & 1) == 0;
                    std::int64_t total = 0;
                    for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][r]; }
                    sm.y[c][r] = canon::a4r_row_output(total, sm.row_scale[c], gate ? scales.alpha_gate : scales.alpha_up);
                }
            }
        }
        __syncthreads();
        for (int k = threadIdx.x; k < 16 * n; k += blockDim.x) {
            const int c = k / 16, j = k % 16;
            h_plane[static_cast<std::size_t>(first + c0 + c) * moe::wide::kA16HRowElements + 16 * slice + j] =
                canon::swiglu_bf16(sm.y[c][2 * j], sm.y[c][2 * j + 1]);
        }
    }
}

template <bool Pairs>
__global__ void __launch_bounds__(kThreads, 2)
    gate_up_kernel_a4r(const bf16* __restrict__ x, MoeDispatch dispatch, MoeExpertSource source, int top_k,
                       const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                       int job_base, std::uint16_t* __restrict__ h_plane, PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<GateUpSharedA4r*>(smem_raw);
    const int job = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const int slice  = blockIdx.x; // row groups 2*slice, 2*slice+1; h elements 16*slice..16*slice+15
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const moe::ExpertScales scales = source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (moe::wide_route(count, scales)) { return; }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    stage_async(record + static_cast<std::size_t>(kGateUpGroups * slice) * moe::kGateUpBlocks * moe::kUnitBytes,
                sm.stage, kGateUpSliceBytes);
    if constexpr (Pairs) {
        if (count > 1) {
            gate_up_pairs_a4r(x, dispatch, top_k, scales, first, count, slice, h_plane, sm);
            return;
        }
    }
    for (int c = 0; c < count; ++c) { // one pass per column of the job
        __syncthreads();
        const auto* column = reinterpret_cast<const std::uint16_t*>(x) +
                             static_cast<std::size_t>(moe::wide::entry_column(dispatch.entries[first + c], top_k,
                                                                              dispatch.shared_first)) *
                                 moe::kHidden;
        encode_a4r_row<moe::kGateUpBlocks>(column, sm.acts, &sm.keys[0][0], &sm.row_scale[0]);
        stage_wait();
        // The down pass (a programmatic dependent in captured graphs) may launch once every CTA has
        // its slice.
        pdl::trigger_dependents();
        __syncthreads();
        std::int64_t s[kGateUpGroups];
        unit_sums_a4r<kGateUpGroups>(sm.stage, moe::kGateUpBlocks, sm.acts, s);
        __syncthreads(); // every warp has read sm.acts before sm.partial overwrites it
        if (lane < 16) {
#pragma unroll
            for (int g = 0; g < kGateUpGroups; ++g) { sm.partial[warp][g * 16 + lane] = s[g]; }
        }
        __syncthreads();
        for (int row = threadIdx.x; row < 16 * kGateUpGroups; row += blockDim.x) {
            const bool gate    = (row & 1) == 0;
            std::int64_t total = 0;
            for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row]; }
            sm.y[0][row] = canon::a4r_row_output(total, sm.row_scale[0], gate ? scales.alpha_gate : scales.alpha_up);
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            const int k = static_cast<int>(threadIdx.x);
            h_plane[static_cast<std::size_t>(first + c) * moe::wide::kA16HRowElements + 16 * slice + k] =
                canon::swiglu_bf16(sm.y[0][2 * k], sm.y[0][2 * k + 1]);
        }
    }
}

struct DownSharedA4r {
    alignas(16) std::uint8_t stage[kDownSliceBytes];
    A4rAct acts[2 * kHBlocks]; // one column's blocks, or both of a pair's
    std::int64_t partial[kWarps][16 * kDownGroups];
    unsigned keys[kWarps][2];
    float row_scale[2];
};

// The column pairs of a two-term down CTA's job (count >= 2 columns), as the one-column loop below
// computes each column.
__device__ void down_pairs_a4r(const MoeDispatch& dispatch, const moe::ExpertScales& scales, int first, int count,
                               int tile, const std::uint16_t* __restrict__ h_plane, bf16* __restrict__ outputs,
                               DownSharedA4r& sm) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int c0 = 0; c0 < count; c0 += 2) {
        const int n   = min(2, count - c0);
        const auto row = [&](int c) {
            return h_plane + static_cast<std::size_t>(first + c0 + c) * moe::wide::kA16HRowElements;
        };
        __syncthreads(); // every thread is past the previous pair's reads of sm.partial and sm.row_scale
        a4r_pair_row_scales<kHBlocks>(row, n, sm.keys, sm.row_scale);
        encode_a4r_pair_blocks(row, n, 0, kHBlocks, sm.row_scale, sm.acts);
        stage_wait();
        __syncthreads();
        std::int64_t lo[kDownGroups][2] = {}, hi[kDownGroups][2] = {};
        unit_sums_a4r_pair<kDownGroups>(sm.stage, moe::kDownBlocks, 0, kHBlocks, sm.acts, n, lo, hi);
        std::int64_t s[kDownGroups][2];
        reduce_row_sums<kDownGroups, 2>(lo, hi, s);
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            if (c < n) {
                if (c > 0) { __syncthreads(); } // every thread has read the previous column's partial sums
                if (lane < 16) {
#pragma unroll
                    for (int g = 0; g < kDownGroups; ++g) { sm.partial[warp][g * 16 + lane] = s[g][c]; }
                }
                __syncthreads();
                const int column = dispatch.entries[first + c0 + c];
                for (int r = threadIdx.x; r < 16 * kDownGroups; r += blockDim.x) {
                    std::int64_t total = 0;
                    for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][r]; }
                    reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * moe::kHidden +
                                                              16 * kDownGroups * tile + r] =
                        canon::a4r_row_output(total, sm.row_scale[c], scales.alpha_down);
                }
            }
        }
    }
}

template <bool Pairs>
__global__ void __launch_bounds__(kThreads)
    down_kernel_a4r(MoeDispatch dispatch, MoeExpertSource source, const std::uint8_t* const* __restrict__ job_records,
                    const std::int32_t* __restrict__ cpu_flags, int job_base, const std::uint16_t* __restrict__ h_plane,
                    bf16* __restrict__ outputs, PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm       = *reinterpret_cast<DownSharedA4r*>(smem_raw);
    const int job  = job_base + static_cast<int>(blockIdx.y);
    const int tile = blockIdx.x; // down row groups 4*tile .. 4*tile+3
    // A programmatic dependent of the gate/up pass, under down_kernel's contract: only h_plane is read
    // after the PDL wait, and every CTA waits.
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    if (job < *dispatch.job_count && (cpu_flags == nullptr || cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(phase, source, expert, job_records, job);
    }
    moe::ExpertScales scales{};
    if (record != nullptr) {
        scales = source.scales[expert];
        first  = dispatch.offsets[expert];
        count  = dispatch.offsets[expert + 1] - first;
        if (moe::wide_route(count, scales)) { record = nullptr; }
    }
    if (record != nullptr) {
        stage_async(record + moe::kGateUpBytes +
                        static_cast<std::size_t>(kDownGroups * tile) * moe::kDownBlocks * moe::kUnitBytes,
                    sm.stage, kDownSliceBytes);
    }
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    if constexpr (Pairs) {
        if (count > 1) {
            down_pairs_a4r(dispatch, scales, first, count, tile, h_plane, outputs, sm);
            return;
        }
    }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int c = 0; c < count; ++c) { // one pass per column of the job
        __syncthreads();
        encode_a4r_row<kHBlocks>(h_plane + static_cast<std::size_t>(first + c) * moe::wide::kA16HRowElements, sm.acts,
                                 &sm.keys[0][0], &sm.row_scale[0]);
        stage_wait();
        __syncthreads();
        std::int64_t s[kDownGroups];
        unit_sums_a4r<kDownGroups>(sm.stage, moe::kDownBlocks, sm.acts, s);
        if (lane < 16) {
#pragma unroll
            for (int g = 0; g < kDownGroups; ++g) { sm.partial[warp][g * 16 + lane] = s[g]; }
        }
        __syncthreads();
        const int column = dispatch.entries[first + c];
        for (int row = threadIdx.x; row < 16 * kDownGroups; row += blockDim.x) {
            std::int64_t total = 0;
            for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row]; }
            reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * moe::kHidden +
                                                      16 * kDownGroups * tile + row] =
                canon::a4r_row_output(total, sm.row_scale[0], scales.alpha_down);
        }
    }
}

template <bool Pairs>
void launch_expert_pass_a4r_as(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                               const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                               std::uint16_t* h_plane, bf16* outputs, cudaStream_t stream, PassPhase phase) {
    static const bool configured = [] {
        cudaFuncSetAttribute(gate_up_kernel_a4r<Pairs>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(GateUpSharedA4r)));
        cudaFuncSetAttribute(down_kernel_a4r<Pairs>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(DownSharedA4r)));
        cudaFuncSetAttribute(gate_up_kernel_a4r<Pairs>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(down_kernel_a4r<Pairs>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        return true;
    }();
    (void)configured;
    gate_up_kernel_a4r<Pairs><<<dim3(kGateUpCtas, jobs), kThreads, sizeof(GateUpSharedA4r), stream>>>(
        x, dispatch, source, top_k, job_records, flags, base, h_plane, phase);
    check_launch("A4 residual gate/up");
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_consumer({dim3(kDownCtas, jobs), dim3(kThreads), sizeof(DownSharedA4r), stream},
                                    down_kernel_a4r<Pairs>, dispatch, source, job_records, flags, base,
                                    static_cast<const std::uint16_t*>(h_plane), outputs, phase));
}

// One pass of the two-term narrow kernels: a gate/up CTA per 16 intermediates and a down CTA per 64
// output rows of each job; h_plane is the workspace's h rows (BF16, wide::kA16HRowElements apart).
void launch_expert_pass_a4r(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                            const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                            std::uint16_t* h_plane, bf16* outputs, cudaStream_t stream, PassPhase phase) {
    if (narrow_pairs()) {
        launch_expert_pass_a4r_as<true>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_plane, outputs,
                                        stream, phase);
    } else {
        launch_expert_pass_a4r_as<false>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_plane, outputs,
                                         stream, phase);
    }
}

#include "ops/offloaded_sparse_moe/cuda/ggml_expert.cuh"
#include "ops/offloaded_sparse_moe/cuda/fp8_expert.cuh"
#include "ops/offloaded_sparse_moe/cuda/lossless_expert.cuh"
#include "ops/offloaded_sparse_moe/cuda/narrow_ring_a16.cuh"

// One pass of the narrow kernels in the source's arithmetic, for a call of `columns` columns. W4A4
// jobs of any width take the one-column kernels (one pass per column over the staged weights, or per
// column pair under INFERNIX_TMP_NARROW_PAIRS); W4A16 jobs under NarrowA16::kA4Residual the two-term
// one-column kernels (likewise); other W4A16
// jobs take them in calls of at most kA16OneColumnCalls columns, else the multi-column kernels, and
// ring calls (moe::wide::ring_call: the workspace has their x_acts) the narrow ring. A GGUF import's
// or block-FP8 bank runs its own exact kernels.
void launch_pass(int columns, const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                 const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                 const moe::wide::ExpertsWorkspace& layout, int columns_out, bf16* outputs, cudaStream_t stream,
                 PassPhase phase = PassPhase::All) {
    // Wide calls (more than kMaxColumns columns) were planned by moe_experts.
    const bool wide = columns > moe::kMaxColumns;
    if (source.ggml.gate_up_format == QType::FP8_E4M3FN_B128) {
        launch_expert_pass_fp8(dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs, stream, phase,
                               wide ? columns_out : 0);
        return;
    }
    if (ggml_bank(source)) {
        launch_pass_ggml(x, dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs, stream, phase,
                         wide ? columns_out : 0, columns);
        return;
    }
    if (layout.x_acts != nullptr) {
        launch_ring_pass(columns, dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs, stream,
                         phase);
        return;
    }
    if (source.activation == moe::ExpertActivation::kA16) {
        auto* h_plane = reinterpret_cast<std::uint16_t*>(layout.h_rows);
        if (source.narrow_a16 == moe::NarrowA16::kA4Residual) { // Flash-Next's geometry (moe_experts checks)
            launch_expert_pass_a4r(x, dispatch, source, top_k, job_records, flags, base, jobs, h_plane, outputs, stream,
                                   phase);
            return;
        }
        moe::with_expert_geometry(source.shape, [&]<class G>() {
            if (columns <= kA16OneColumnCalls) {
                launch_expert_pass_a16<G, 1>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_plane,
                                             outputs, stream, phase);
            } else {
                launch_expert_pass_a16<G, kA16PassColumns>(x, dispatch, source, top_k, job_records, flags, base, jobs,
                                                           h_plane, outputs, stream, phase);
            }
        });
        return;
    }
    launch_expert_pass(x, dispatch, source, top_k, job_records, flags, base, jobs, layout.h_blocks, columns_out, outputs,
                       stream, phase);
}

// ------------------------------------------------------------------------------ CPU-served misses

// CTAs of cpu_wait_kernel: CTA c places the outputs of jobs c, c + kWaitCtas, ..., so the mapped
// reads of several jobs are in flight at once.
constexpr int kWaitCtas = offloaded_moe::kMaxCpuJobs < 8 ? offloaded_moe::kMaxCpuJobs : 8;

// One CTA picks the CPU jobs: of the call's M misses, want = min(cap, M - M / d) of
// those with at most max_job_columns columns, the fewest-column ones first and, within a width, in
// job order; cap is max_jobs, or wide_jobs for calls of at least wide_from columns (prefill
// assist); d is pcie_divisor, or one_column_divisor (when set) in a one-column call (none for d = 0);
// with pcie_share > 0 the stage keeps (M * pcie_share) >> 8 instead of M / pcie_divisor.
// Each thread ranks its jobs among their width with warp ballots, so no thread walks the
// job list. The chosen jobs are marked in cpu_flags and listed in that order in call->job and the
// request. The x columns they read are published compacted, in column order (the request names
// them by compact index), then the request's sequence after a system-scope fence.
// The CPU plan of one call by the threads [0, kThreads) of a CTA: the plan kernel's whole CTA, or
// (Named) the first kThreads threads of the fused route kernel, which synchronize on a named barrier.
template <bool Named>
__device__ __forceinline__ void plan_barrier() {
    if constexpr (Named) {
        asm volatile("bar.sync 1, %0;" ::"n"(kThreads) : "memory");
    } else {
        __syncthreads();
    }
}

// The shared state of one CPU plan (one allocation, so the fused route kernel's static shared memory
// holds a single plan).
struct CpuPlanShared {
    static constexpr int kWidths    = offloaded_moe::kMaxCpuColumns;
    // A miss's category: its width w for an SSD-only expert (no host record, MoeExpertSource::
    // host_table), kWidths + w for one in host memory. SSD-only experts are chosen first (only the
    // CPU can serve them), then host-memory misses under the PCIe divisor rule; without a host table
    // every miss is in host memory and the choice is the width-ordered one alone.
    static constexpr int kCats      = 2 * kWidths;
    static constexpr int kMaskWords = offloaded_moe::kMaxCpuCallColumns / 32;
    // cpu_plan_body ranks at most kChunks chunks of kThreads jobs: the dispatch's 1,024 experts
    // (moe_dispatch requires at most kDispatchThreads), so every call fits.
    static constexpr int kChunks    = kDispatchThreads / kThreads;
    int take[kCats + 1], first_slot[kCats + 1];
    unsigned used[kMaskWords];                          // x columns the chosen jobs read
    short compact[offloaded_moe::kMaxCpuCallColumns];   // column -> compact index
    short published[offloaded_moe::kMaxCpuXColumns];    // compact index -> column
    int slot_first[offloaded_moe::kMaxCpuJobs];         // a chosen job's first dispatch entry
    int chosen, publish_count;
    // Each (chunk, warp)'s misses of each category, then their exclusive prefix in (chunk, warp)
    // order; each (chunk, warp)'s host-memory misses; a chosen job's width.
    int chunk_misses[kChunks][kWarps][kCats + 1];
    int chunk_host[kChunks][kWarps];
    std::int8_t slot_width[offloaded_moe::kMaxCpuJobs];
};

// The plan (every request field, x column, cpu_flags entry and CpuCall word; byte for byte the
// earlier serial body's, which it replaced on 2026-10-24 after Dense8 decode measured +0.8 %, 6 of 6
// pairs; candidates OR-04):
// - each thread classifies its jobs (job q * kThreads + threadIdx.x of chunk q) once and ranks them
//   within its warp by ballots, with no shared atomics and so no clearing barrier;
// - warp 0 turns the (chunk, warp) counts into exclusive prefixes and the takes and first slots
//   into closed forms of the serial walk over widths (its running count is min(cap, prefix)), so a
//   chosen job's slot is first_slot + prefix + lane rank and the chunks need no carry barriers;
// - a chosen job's width stays in shared memory (not read back from the mapped request: a PCIe read
//   the host answers behind the link's staging traffic);
// - a call of at most 32 columns (every fused-route call) compacts x from the one mask word in
//   each thread (compact index = used columns below), so no thread walks the columns; wider calls
//   compact in one warp scan;
// - one system-scope fence, by thread 0 after the barrier that follows every thread's request and
//   x stores, then the sequence. Fences are cumulative (PTX memory model; the cooperative-groups grid
//   barrier relies on the same pattern): the stores thread 0 has synchronized with at that barrier
//   are ordered before its sequence store at system scope, and the host reads every field only
//   after it sees the sequence and an acquire fence (MissRequest, CpuMissService::serve). No fence
//   follows the sequence store: it orders nothing the host or a later kernel reads (call->pending
//   is read by the CPU wait, a later kernel in stream order), and a strong system-scope store needs
//   no fence to become visible.
// Five barriers at most (four for calls of at most 32 columns), against the serial body's 5 + 3 per
// chunk.
template <bool Named>
__device__ __forceinline__ void cpu_plan_body(const MoeDispatch& dispatch, const MoeExpertSource& source,
                                              const bf16* __restrict__ x, int columns, int top_k, int max_jobs,
                                              std::int32_t* __restrict__ cpu_flags, CpuCall* __restrict__ call,
                                              CpuPlanShared& sm) {
    constexpr int kWidths = CpuPlanShared::kWidths;
    constexpr int kCats   = CpuPlanShared::kCats;
    constexpr int kChunks = CpuPlanShared::kChunks;
    const auto& channel   = source.cpu;
    const int jobs        = min(*dispatch.job_count, max_jobs);
    const int chunks      = (jobs + kThreads - 1) / kThreads; // at most kChunks (cpu_plan)
    const int max_width   = channel.max_job_columns;
    const int cap = channel.wide_from > 0 && columns >= channel.wide_from ? channel.wide_jobs : channel.max_jobs;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int j = threadIdx.x; j < max_jobs; j += kThreads) { cpu_flags[j] = 0; }
    if (threadIdx.x < CpuPlanShared::kMaskWords) { sm.used[threadIdx.x] = 0; }
    // A job's category (0: not a CPU candidate) | its rank among its warp's jobs of that category << 8.
    int packed[kChunks];
#pragma unroll
    for (int q = 0; q < kChunks; ++q) {
        packed[q] = 0;
        if (q >= chunks) { continue; }
        const int j    = q * kThreads + static_cast<int>(threadIdx.x);
        int cat        = 0;
        bool host_miss = false; // M counts every host-memory miss; only those of at most max_width columns are candidates
        if (j < jobs) {
            const int expert = dispatch.jobs[j];
            // A streamed record counts as resident: neither staged nor CPU-served.
            if (resident_record(source, expert) == nullptr &&
                (source.prefetched == nullptr || source.prefetched[expert] < 0)) {
                const int first = dispatch.offsets[expert];
                const int count = dispatch.offsets[expert + 1] - first;
                // Tail skip: flagged 2, not a miss.
                bool tail = source.tail_skip_rank > 0 && count <= moe::kMaxColumns;
                for (int c = 0; c < count && tail; ++c) {
                    const int entry = dispatch.entries[first + c];
                    tail = entry < dispatch.shared_first && entry % top_k >= source.tail_skip_rank;
                }
                if (tail) {
                    cpu_flags[j] = 2;
                } else {
                    const bool ssd = source.host_table != nullptr && source.host_table[expert] == nullptr;
                    host_miss      = !ssd;
                    cat            = count <= max_width ? (ssd ? 0 : kWidths) + count : 0;
                }
            }
        }
        const unsigned host_ballot = __ballot_sync(0xFFFFFFFFU, host_miss);
        int rank = 0;
#pragma unroll
        for (int c = 1; c <= kCats; ++c) {
            const unsigned ballot = __ballot_sync(0xFFFFFFFFU, cat == c);
            if (lane == 0) { sm.chunk_misses[q][warp][c] = __popc(ballot); }
            if (cat == c) { rank = __popc(ballot & ((1U << lane) - 1U)); }
        }
        if (lane == 0) { sm.chunk_host[q][warp] = __popc(host_ballot); }
        packed[q] = cat | rank << 8;
    }
    plan_barrier<Named>();
    if (warp == 0) {
        // Lane c in [1, kCats]: category c's counts become exclusive prefixes in (chunk, warp) order
        // (the rank's earlier chunks and warps) and its total.
        int total = 0;
        if (lane >= 1 && lane <= kCats) {
            for (int q = 0; q < chunks; ++q) {
#pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    const int n                 = sm.chunk_misses[q][w][lane];
                    sm.chunk_misses[q][w][lane] = total;
                    total += n;
                }
            }
        }
        int misses = 0;
        for (int i = lane; i < chunks * kWarps; i += 32) { misses += sm.chunk_host[i / kWarps][i % kWarps]; }
        misses = __reduce_add_sync(0xFFFFFFFFU, misses);
        int incl = total; // inclusive prefix of the totals over categories (lane 0 holds none)
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int v = __shfl_up_sync(0xFFFFFFFFU, incl, offset);
            if (lane >= offset) { incl += v; }
        }
        const int excl    = incl - total;
        const int ssd     = __shfl_sync(0xFFFFFFFFU, incl, kWidths);
        const int host    = __shfl_sync(0xFFFFFFFFU, incl, kCats) - ssd;
        const int n       = min(cap, ssd); // SSD-only, up to the cap
        const int one     = channel.one_column_divisor_device != nullptr && columns == 1 ? *channel.one_column_divisor_device
                                                                                         : channel.one_column_divisor;
        // The link keeps M / one_column_divisor in a one-column call (when set), else (M * pcie_share) >> 8
        // with a share, else M / pcie_divisor (none for 0).
        const bool by_one = columns == 1 && one > 0;
        const int kept    = by_one                     ? misses / one
                            : channel.pcie_share > 0   ? (misses * channel.pcie_share) >> 8
                            : channel.pcie_divisor > 0 ? misses / channel.pcie_divisor
                                                       : 0;
        const int want    = min(cap - n, misses - kept);
        if (lane >= 1 && lane <= kWidths) {
            sm.first_slot[lane] = min(cap, excl);
            sm.take[lane]       = min(cap, incl) - min(cap, excl);
        } else if (lane > kWidths && lane <= kCats) {
            const int before    = excl - ssd, through = incl - ssd;
            sm.first_slot[lane] = n + min(want, before);
            sm.take[lane]       = min(want, through) - min(want, before);
        }
        if (lane == 0) {
            sm.chosen  = n + min(want, host);
            call->jobs = sm.chosen;
        }
    }
    plan_barrier<Named>();
    const int chosen = sm.chosen;
    if (chosen == 0) {
        if (threadIdx.x == 0) { call->pending = 0; }
        return;
    }
#pragma unroll
    for (int q = 0; q < kChunks; ++q) {
        const int cat = packed[q] & 0xFF;
        if (q >= chunks || cat == 0) { continue; }
        const int rank = (packed[q] >> 8) + sm.chunk_misses[q][warp][cat];
        if (rank >= sm.take[cat]) { continue; }
        const int j      = q * kThreads + static_cast<int>(threadIdx.x);
        const int expert = dispatch.jobs[j];
        const int first  = dispatch.offsets[expert];
        const int width  = (cat - 1) % kWidths + 1;
        const int slot   = sm.first_slot[cat] + rank;
        cpu_flags[j]                  = 1;
        call->job[slot]               = j;
        channel.request->expert[slot] = expert;
        channel.request->record[slot] =
            source.host_table != nullptr ? reinterpret_cast<std::uint64_t>(source.host_table[expert]) : 0;
        channel.request->ncols[slot]  = width;
        sm.slot_first[slot]           = first;
        sm.slot_width[slot]           = static_cast<std::int8_t>(width);
        for (int c = 0; c < width; ++c) {
            const int column = moe::wide::entry_column(dispatch.entries[first + c], top_k, dispatch.shared_first);
            atomicOr(&sm.used[column / 32], 1U << (column % 32));
        }
    }
    plan_barrier<Named>();
    if (threadIdx.x == 0) {
        channel.request->layer  = channel.layer;
        channel.request->jobs   = chosen;
        channel.request->tiered = source.host_table != nullptr ? 1 : 0;
    }
    const int kVectors = source.shape.hidden * 2 / 16;
    const auto* src    = reinterpret_cast<const int4*>(x);
    auto* dst          = reinterpret_cast<int4*>(channel.x);
    if (columns <= 32) {
        // One mask word: column t's compact index is the number of used columns below t.
        const unsigned used = sm.used[0];
        for (int i = threadIdx.x; i < chosen * offloaded_moe::kMaxCpuColumns; i += kThreads) {
            const int slot = i / offloaded_moe::kMaxCpuColumns, c = i % offloaded_moe::kMaxCpuColumns;
            if (c < sm.slot_width[slot]) {
                const int t = moe::wide::entry_column(dispatch.entries[sm.slot_first[slot] + c], top_k, dispatch.shared_first);
                channel.request->column[slot][c] = __popc(used & ((1U << t) - 1U));
            }
        }
        // The k-th compact column is the lowest set bit of `rest` once k_at = k lower ones are cleared;
        // a thread's k never decreases.
        const int publish_count = __popc(used);
        unsigned rest           = used;
        int k_at                = 0;
        for (int i = threadIdx.x; i < publish_count * kVectors; i += kThreads) {
            const int k = i / kVectors, v = i % kVectors;
            for (; k_at < k; ++k_at) { rest &= rest - 1U; }
            const int t = __ffs(static_cast<int>(rest)) - 1;
            dst[static_cast<std::size_t>(k) * kVectors + v] = src[static_cast<std::size_t>(t) * kVectors + v];
        }
    } else {
        if (warp == 0) {
            // Lane l compacts mask words 4l .. 4l + 3 after the used columns of the lower lanes.
            static_assert(CpuPlanShared::kMaskWords == 4 * 32, "warp 0 compacts four mask words a lane");
            const int words = (columns + 31) / 32;
            unsigned word[4];
            int count = 0;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int w = 4 * lane + u;
                word[u]     = w < words ? sm.used[w] : 0U;
                count += __popc(word[u]);
            }
            int incl = count;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int v = __shfl_up_sync(0xFFFFFFFFU, incl, offset);
                if (lane >= offset) { incl += v; }
            }
            int k = incl - count;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                for (unsigned m = word[u]; m != 0U; m &= m - 1U) {
                    const int t     = (4 * lane + u) * 32 + __ffs(static_cast<int>(m)) - 1;
                    sm.compact[t]   = static_cast<short>(k);
                    sm.published[k] = static_cast<short>(t);
                    ++k;
                }
            }
            if (lane == 31) { sm.publish_count = incl; }
        }
        plan_barrier<Named>();
        for (int i = threadIdx.x; i < chosen * offloaded_moe::kMaxCpuColumns; i += kThreads) {
            const int slot = i / offloaded_moe::kMaxCpuColumns, c = i % offloaded_moe::kMaxCpuColumns;
            if (c < sm.slot_width[slot]) {
                channel.request->column[slot][c] = sm.compact[moe::wide::entry_column(
                    dispatch.entries[sm.slot_first[slot] + c], top_k, dispatch.shared_first)];
            }
        }
        const int publish_count = sm.publish_count;
        for (int i = threadIdx.x; i < publish_count * kVectors; i += kThreads) {
            const int k = i / kVectors, v = i % kVectors;
            dst[static_cast<std::size_t>(k) * kVectors + v] =
                src[static_cast<std::size_t>(sm.published[k]) * kVectors + v];
        }
    }
    plan_barrier<Named>();
    if (threadIdx.x == 0) {
        __threadfence_system(); // cumulative: every thread's request and x stores before the sequence
        // 0 means "nothing published" to the wait (CpuCall::pending): skip it when the counter wraps.
        std::uint32_t sequence = atomicAdd(channel.sequence, 1U) + 1U;
        if (sequence == 0) { sequence = atomicAdd(channel.sequence, 1U) + 1U; }
        *reinterpret_cast<volatile std::uint32_t*>(&channel.request->sequence) = sequence;
        call->pending = static_cast<std::int32_t>(sequence);
    }
}

// The CPU plan of one call.
template <bool Named>
__device__ __forceinline__ void cpu_plan(const MoeDispatch& dispatch, const MoeExpertSource& source,
                                         const bf16* __restrict__ x, int columns, int top_k, int max_jobs,
                                         std::int32_t* __restrict__ cpu_flags, CpuCall* __restrict__ call) {
    static_assert(CpuPlanShared::kChunks * kThreads >= kDispatchThreads, "a call's jobs fit the plan's chunks");
    __shared__ CpuPlanShared sm;
    cpu_plan_body<Named>(dispatch, source, x, columns, top_k, max_jobs, cpu_flags, call, sm);
}

__global__ void __launch_bounds__(kThreads)
    cpu_plan_kernel(MoeDispatch dispatch, MoeExpertSource source, const bf16* __restrict__ x, int columns,
                    int top_k, int max_jobs, std::int32_t* __restrict__ cpu_flags, CpuCall* __restrict__ call) {
    pdl::enter(); // launched as a programmatic dependent of the dispatch
    cpu_plan<false>(dispatch, source, x, columns, top_k, max_jobs, cpu_flags, call);
}

// route_dispatch_kernel followed by the CPU plan in the same CTA: after a barrier of the whole CTA
// makes the dispatch visible, its first kThreads threads plan as cpu_plan_kernel does (the same
// results) and the rest exit, so the plan, and the host's start, follow the dispatch with no kernel
// boundary.
template <bool kMasked>
__global__ void __launch_bounds__(kDispatchThreads)
    route_dispatch_plan_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                               bool bf16_logits, bool shared_job, std::int32_t* __restrict__ ids,
                               float* __restrict__ weights,
                               float* __restrict__ shared_gate, MoeDispatch dispatch,
                               std::int32_t* __restrict__ route_log, MoeExpertSource source,
                               const bf16* __restrict__ x, int max_jobs, std::int32_t* __restrict__ cpu_flags,
                               CpuCall* __restrict__ call, MoeColumnMask mask) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t routed[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    pdl::enter(); // launched as a programmatic dependent of the router projection
    const int i = threadIdx.x, warp = i / 32, lane = i % 32;
    count[i]    = 0;
    if (warp < columns) {
        const int id = route_column(logits, experts, top_k, warp, bf16_logits, ids, weights, shared_gate);
        if (lane < top_k) { routed[warp * top_k + lane] = id; }
    }
    __syncthreads();
    const int routed_entries = columns * top_k, entries = routed_entries + (shared_job ? columns : 0);
    dispatch_small(i < entries ? (i < routed_entries ? routed[i] : experts) : 0,
                   !kMasked || (i < entries && entry_live<kMasked>(mask, i, routed_entries, top_k)), entries,
                   routed_entries, experts + (shared_job ? 1 : 0), dispatch, route_log, count, cursor, warp_totals);
    __syncthreads();
    if (i >= kThreads) { return; }
    cpu_plan<true>(dispatch, source, x, columns, top_k, max_jobs, cpu_flags, call);
}

// CTAs that warm the source's L2 spans after the waiting CTAs (design §19.3.3 Phase 2).
constexpr int kWarmCtas  = 8;
constexpr int kWarmPiece = 16384; // bytes per prefetch instruction

// CTA w of kWarmCtas: prefetches its 1/kWarmCtas share of the concatenated spans into L2, one piece
// per thread per step. Prefetches are hints: no result depends on them.
__device__ void warm_l2(const MoeL2Warm& warm, int w) {
    std::size_t total = 0;
#pragma unroll
    for (int i = 0; i < MoeL2Warm::kSpans; ++i) { total += warm.bytes[i] / 16 * 16; }
    const std::size_t share = (total / kWarmCtas + 15) / 16 * 16;
    const std::size_t begin = share * static_cast<std::size_t>(w);
    const std::size_t end   = begin + share < total ? begin + share : total;
    for (std::size_t at = begin + static_cast<std::size_t>(threadIdx.x) * kWarmPiece; at < end;
         at += static_cast<std::size_t>(blockDim.x) * kWarmPiece) {
        // Locate the span holding byte `at` of the concatenation; a piece never crosses a span.
        std::size_t offset = at;
        int span           = 0;
        while (span < MoeL2Warm::kSpans && offset >= warm.bytes[span] / 16 * 16) {
            offset -= warm.bytes[span] / 16 * 16;
            ++span;
        }
        if (span == MoeL2Warm::kSpans) { break; }
        const std::size_t left  = warm.bytes[span] / 16 * 16 - offset;
        std::size_t piece = end - at < static_cast<std::size_t>(kWarmPiece) ? end - at : static_cast<std::size_t>(kWarmPiece);
        piece             = piece < left ? piece : left;
        const auto* address = static_cast<const std::uint8_t*>(warm.ptr[span]) + offset;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(address), "r"(static_cast<unsigned>(piece))
                     : "memory");
    }
}

// Tail skip (MoeExpertSource::tail_skip_rank): no route writes a skipped expert's outputs, so they are
// zeroed for the combine. CTA b clears the entries of jobs b, b + gridDim.x, ... flagged 2.
__global__ void __launch_bounds__(kThreads)
    tail_zero_kernel(MoeDispatch dispatch, int max_jobs, const std::int32_t* __restrict__ cpu_flags, int hidden,
                     bf16* __restrict__ outputs) {
    const int jobs    = min(*dispatch.job_count, max_jobs);
    const int vectors = hidden * 2 / 16; // 16-byte vectors per column
    auto* out         = reinterpret_cast<int4*>(outputs);
    for (int j = static_cast<int>(blockIdx.x); j < jobs; j += static_cast<int>(gridDim.x)) {
        if (cpu_flags[j] != 2) { continue; }
        const int expert = dispatch.jobs[j];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        for (int i = threadIdx.x; i < count * vectors; i += blockDim.x) {
            out[static_cast<std::size_t>(dispatch.entries[first + i / vectors]) * vectors + i % vectors] =
                make_int4(0, 0, 0, 0);
        }
    }
}

// kWaitCtas (+ kWarmCtas) CTAs: each waiting CTA with jobs to place waits for the host's answer to
// this call's request, then places the outputs of jobs blockIdx.x, blockIdx.x + kWaitCtas, ...; the
// warming CTAs prefetch the source's L2 spans while the host works.
__global__ void __launch_bounds__(kThreads)
    cpu_wait_kernel(MoeDispatch dispatch, MoeExpertSource source, const CpuCall* __restrict__ call, int hidden,
                    bf16* __restrict__ outputs) {
    constexpr int kJobsPerCta = (offloaded_moe::kMaxCpuJobs + kWaitCtas - 1) / kWaitCtas;
    constexpr int kSlots      = kJobsPerCta * offloaded_moe::kMaxCpuColumns;
    __shared__ int entries[kSlots];
    const auto sequence = static_cast<std::uint32_t>(call->pending);
    const int n         = call->jobs;
    if (static_cast<int>(blockIdx.x) >= kWaitCtas) {
        if (sequence != 0) { warm_l2(source.l2_warm, static_cast<int>(blockIdx.x) - kWaitCtas); }
        return;
    }
    if (sequence == 0 || static_cast<int>(blockIdx.x) >= n) { return; }
    __shared__ bool silent;
    if (threadIdx.x == 0) {
        // The wait lasts while the host's heartbeat changes (an SSD-served job may take seconds);
        // without a heartbeat it is bounded at 2 s. A silent host fails the call through its error
        // word, or traps when it has none.
        const auto* done  = reinterpret_cast<const volatile std::uint32_t*>(source.cpu.done);
        const auto* beat  = reinterpret_cast<const volatile std::uint32_t*>(source.cpu.heartbeat);
        std::uint32_t seen = beat != nullptr ? *beat : 0;
        std::uint64_t since = global_ns();
        const std::uint64_t entered = since;
        const std::uint64_t limit = beat != nullptr ? source.heartbeat_timeout_ns : 2000000000ULL;
        silent = false;
        while (*done != sequence) {
            const std::uint64_t now = global_ns();
            if (beat != nullptr && *beat != seen) {
                seen  = *beat;
                since = now;
            } else if (now - since > limit) {
                if (beat == nullptr || source.error == nullptr) { asm volatile("trap;"); }
                report_error(source, offloaded_moe::kErrorHostSilent);
                silent = true;
                break;
            }
            __nanosleep(200);
        }
        if (source.wait_ns != nullptr && blockIdx.x == 0) {
            atomicAdd(source.wait_ns, static_cast<unsigned long long>(global_ns() - entered));
        }
        __threadfence_system();
        if (!silent && blockIdx.x == 0 && source.cpu.status != nullptr) {
            const std::uint32_t status = *reinterpret_cast<const volatile std::uint32_t*>(source.cpu.status);
            if (status != 0) { report_error(source, status); }
        }
    }
    __syncthreads();
    if (silent) { return; }
    // Local job k is job blockIdx.x + k * gridDim.x; local slot k * kMaxCpuColumns + c holds the
    // output entry of its column c (-1: unused), as slot i * kMaxCpuColumns + c of y does for job i.
    for (int s = threadIdx.x; s < kSlots; s += blockDim.x) { entries[s] = -1; }
    __syncthreads();
    const int mine = (n - 1 - static_cast<int>(blockIdx.x)) / kWaitCtas + 1;
    if (threadIdx.x < mine) {
        const int k      = threadIdx.x;
        const int expert = dispatch.jobs[call->job[blockIdx.x + k * kWaitCtas]];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        for (int c = 0; c < count; ++c) { entries[k * offloaded_moe::kMaxCpuColumns + c] = dispatch.entries[first + c]; }
    }
    __syncthreads();
    // The copy issues each thread's loads from mapped memory in batches before storing them, so a
    // batch costs one PCIe round trip rather than one per load. .cv loads: y is rewritten by every
    // layer's request.
    constexpr int kBatch = 8;
    const int vectors    = hidden * 2 / 16; // 16-byte vectors per column
    const int items      = mine * offloaded_moe::kMaxCpuColumns * vectors;
    const auto* y        = reinterpret_cast<const int4*>(source.cpu.y);
    auto* out            = reinterpret_cast<int4*>(outputs);
    const auto y_item    = [&](int item) {
        const int slot = item / vectors, k = slot / offloaded_moe::kMaxCpuColumns;
        const int job  = static_cast<int>(blockIdx.x) + k * kWaitCtas;
        return (static_cast<std::size_t>(job) * offloaded_moe::kMaxCpuColumns + slot % offloaded_moe::kMaxCpuColumns) *
                   vectors + item % vectors;
    };
    for (int base = threadIdx.x; base < items; base += kBatch * blockDim.x) {
        int4 loaded[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const int item = base + b * blockDim.x;
            if (item < items && entries[item / vectors] >= 0) { loaded[b] = __ldcv(y + y_item(item)); }
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

template <bool kMasked>
__global__ void combine_kernel(const bf16* __restrict__ outputs, const float* __restrict__ weights,
                               const float* __restrict__ shared_gate, const bf16* __restrict__ shared,
                               int hidden, int top_k, int columns, bf16* __restrict__ y, MoeColumnMask mask) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    if constexpr (kMasked) {
        // A masked column has no outputs (no dispatch entry): its y is 0.
        if (!column_live(mask, t)) {
            y[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(0.0F);
            return;
        }
    }
    float sum = 0.0F;
    for (int k = 0; k < top_k; ++k) { // fixed rank order: placement-invariant
        sum += weights[static_cast<std::size_t>(t) * top_k + k] *
               __bfloat162float(outputs[(static_cast<std::size_t>(t) * top_k + k) * hidden + d]);
    }
    sum += shared_gate[t] * __bfloat162float(shared[static_cast<std::size_t>(t) * hidden + d]);
    y[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum);
}

// combine_kernel's y, kept in a register, injected into the residual streams as
// hyper_connection_inject does: residual[s*H+d,t] = bf16(residual + inject[s,t] * y[d,t]).
__global__ void combine_inject_kernel(const bf16* __restrict__ outputs, const float* __restrict__ weights,
                                      const float* __restrict__ shared_gate, const bf16* __restrict__ shared,
                                      const float* __restrict__ inject, int hidden, int top_k, int streams,
                                      int columns, bf16* __restrict__ residual) {
    // A programmatic dependent; its small grid lets the next mixer launch and fetch its weights at once.
    pdl::enter();
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (d >= hidden || t >= columns) { return; }
    float sum = 0.0F;
    for (int k = 0; k < top_k; ++k) { // fixed rank order: placement-invariant
        sum += weights[static_cast<std::size_t>(t) * top_k + k] *
               __bfloat162float(outputs[(static_cast<std::size_t>(t) * top_k + k) * hidden + d]);
    }
    sum += shared_gate[t] * __bfloat162float(shared[static_cast<std::size_t>(t) * hidden + d]);
    const float v           = __bfloat162float(__float2bfloat16_rn(sum));
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int s = 0; s < streams; ++s) {
        bf16& r = residual[static_cast<std::size_t>(t) * width + static_cast<std::size_t>(s) * hidden + d];
        r       = __float2bfloat16_rn(__bfloat162float(r) + inject[static_cast<std::size_t>(t) * streams + s] * v);
    }
}

} // namespace

void moe_route(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, cudaStream_t stream,
               RouterLogits precision) {
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
    CUDA_CHECK(pdl::launch_consumer({dim3((columns + warps_per_block - 1) / warps_per_block), dim3(kThreads), 0, stream},
                                    route_kernel, static_cast<const float*>(logits.data), experts, columns,
                                    static_cast<int>(top_k), precision == RouterLogits::kBf16,
                                    static_cast<std::int32_t*>(routing.ids.data),
                                    static_cast<float*>(routing.weights.data),
                                    static_cast<float*>(routing.shared_gate.data)));
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

namespace {

// A column mask covers whole rows of the call's columns (MoeColumnMask); with a shared job, the
// shared entries of masked columns are dropped too, and without one (a dense shared expert beside the
// call) only routed entries exist.
bool masked_call(const MoeColumnMask& mask, int columns, bool shared_job) {
    (void)shared_job;
    if (mask.valid == nullptr) { return false; }
    require(mask.width > 0 && columns % mask.width == 0, "a column mask needs whole rows of the call's columns");
    return true;
}

} // namespace

void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch,
                  std::int32_t* route_log, cudaStream_t stream, bool shared_job) {
    const int routed = static_cast<int>(routing.ids.numel()), columns = routing.ids.ne[1];
    const int top_k  = routing.ids.ne[0];
    // A shared job is expert E with one entry per column after the routed ones.
    const int jobs_experts = experts + (shared_job ? 1 : 0), entries = routed + (shared_job ? columns : 0);
    const int shared       = shared_job ? experts : -1;
    require(jobs_experts > 0 && jobs_experts <= kDispatchThreads, "dispatch supports up to 1024 experts");
    const bool masked     = masked_call(routing.mask, columns, shared_job);
    dispatch.shared_first = shared_job ? routed : 0x7FFFFFFF;
    const auto* ids       = static_cast<const std::int32_t*>(routing.ids.data);
    if (entries <= kDispatchThreads) {
        // Decode and verification widths (k * T <= 1024): one kernel and no memset node.
        CUDA_CHECK(pdl::launch_consumer({dim3(1), dim3(kDispatchThreads), 0, stream},
                                        masked ? dispatch_small_kernel<true> : dispatch_small_kernel<false>, ids,
                                        entries, routed, shared, jobs_experts, dispatch, route_log, routing.mask,
                                        top_k));
        return;
    }
    require(cudaMemsetAsync(dispatch.counts, 0, sizeof(std::int32_t) * jobs_experts, stream) == cudaSuccess,
            "dispatch could not clear its counts");
    const int grid = (entries + kThreads - 1) / kThreads;
    if (masked) {
        count_kernel<true><<<grid, kThreads, 0, stream>>>(ids, entries, routed, shared, dispatch.counts, route_log,
                                                          routing.mask, top_k);
    } else {
        count_kernel<false><<<grid, kThreads, 0, stream>>>(ids, entries, routed, shared, dispatch.counts, route_log,
                                                           routing.mask, top_k);
    }
    check_launch("count");
    scan_kernel<<<1, kDispatchThreads, 0, stream>>>(dispatch.counts, jobs_experts, dispatch.offsets, dispatch.cursor,
                                                    dispatch.jobs, dispatch.job_count);
    check_launch("scan");
    if (masked) {
        scatter_kernel<true><<<grid, kThreads, 0, stream>>>(ids, entries, routed, shared, dispatch.cursor,
                                                            dispatch.entries, routing.mask, top_k);
    } else {
        scatter_kernel<false><<<grid, kThreads, 0, stream>>>(ids, entries, routed, shared, dispatch.cursor,
                                                             dispatch.entries, routing.mask, top_k);
    }
    check_launch("scatter");
}

void moe_route_dispatch(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, MoeDispatch& dispatch,
                        std::int32_t* route_log, cudaStream_t stream, RouterLogits precision, bool shared_job) {
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    const std::int64_t entries = static_cast<std::int64_t>(columns) * (top_k + (shared_job ? 1 : 0));
    if (columns > kRouteDispatchColumns || entries > kDispatchThreads) {
        moe_route(logits, top_k, routing, stream, precision);
        moe_dispatch(routing, experts, dispatch, route_log, stream, shared_job);
        return;
    }
    require(experts + (shared_job ? 1 : 0) <= kDispatchThreads, "dispatch supports up to 1024 experts");
    dispatch.shared_first = shared_job ? columns * top_k : 0x7FFFFFFF;
    require(contiguous(logits, DType::FP32) && contiguous(routing.ids, DType::I32) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32),
            "route requires contiguous FP32 logits and I32/FP32 outputs");
    require(experts > 0 && experts <= 512 && top_k > 0 && top_k <= 16 && top_k <= experts && columns > 0,
            "route supports up to 512 experts and top-16");
    require(routing.ids.ne[0] == top_k && routing.ids.ne[1] == columns && routing.weights.ne[0] == top_k &&
                routing.weights.ne[1] == columns && routing.shared_gate.numel() == columns,
            "route output shapes disagree");
    const bool masked = masked_call(routing.mask, columns, shared_job);
    CUDA_CHECK(pdl::launch_consumer({dim3(1), dim3(kDispatchThreads), 0, stream},
                                    masked ? route_dispatch_kernel<true> : route_dispatch_kernel<false>,
                                    static_cast<const float*>(logits.data), experts, columns,
                                    static_cast<int>(top_k), precision == RouterLogits::kBf16, shared_job,
                                    static_cast<std::int32_t*>(routing.ids.data),
                                    static_cast<float*>(routing.weights.data),
                                    static_cast<float*>(routing.shared_gate.data), dispatch, route_log, routing.mask));
}

std::size_t moe_experts_workspace_bytes(const moe::ExpertShape& shape, std::int32_t max_jobs, std::int32_t entries,
                                        std::int32_t columns, moe::ExpertActivation activation,
                                        moe::WideA16 wide_a16) {
    return moe::wide::experts_workspace_bytes(shape, max_jobs, entries, columns, activation, wide_a16);
}

namespace {

constexpr std::size_t kFetchCallOffset = (sizeof(CpuCall) + 63) / 64 * 64;
static_assert(kFetchCallOffset + sizeof(FetchCall) <= moe::wide::kCpuCallBytes,
              "the workspace reserves kCpuCallBytes for a call's CPU and fetch bookkeeping");

// The CPU bookkeeping of a call in its workspace.
CpuCall* cpu_call_of(void* workspace, const Tensor& x, const MoeExpertSource& source, std::int32_t max_jobs,
                     std::int32_t entries, std::int32_t** flags) {
    const auto layout = moe::wide::carve_experts_workspace(workspace, source.shape, max_jobs, entries, x.ne[1],
                                                           source.activation, source.wide_a16);
    if (flags != nullptr) { *flags = layout.cpu_flags; }
    return static_cast<CpuCall*>(layout.cpu_call);
}

// The fetch bookkeeping of a call in its workspace, after the CPU's.
FetchCall* fetch_call_of(void* workspace, const Tensor& x, const MoeExpertSource& source, std::int32_t max_jobs,
                         std::int32_t entries) {
    const auto layout = moe::wide::carve_experts_workspace(workspace, source.shape, max_jobs, entries, x.ne[1],
                                                           source.activation, source.wide_a16);
    return reinterpret_cast<FetchCall*>(static_cast<std::byte*>(layout.cpu_call) + kFetchCallOffset);
}

bool cpu_served(const Tensor& x, const MoeExpertSource& source) {
    return source.cpu.max_jobs > 0 && x.ne[1] <= source.cpu.max_columns;
}

void require_cpu_channel(const Tensor& x, const MoeExpertSource& source) {
    require(source.cpu.request != nullptr && source.cpu.x != nullptr && source.cpu.y != nullptr &&
                source.cpu.done != nullptr && source.cpu.sequence != nullptr &&
                source.cpu.max_jobs <= offloaded_moe::kMaxCpuJobs && source.cpu.wide_jobs >= 0 &&
                source.cpu.wide_jobs <= offloaded_moe::kMaxCpuJobs &&
                source.cpu.max_columns <= offloaded_moe::kMaxCpuCallColumns && source.cpu.max_job_columns >= 1 &&
                source.cpu.max_job_columns <= offloaded_moe::kMaxCpuColumns &&
                reinterpret_cast<std::uintptr_t>(x.data) % 16 == 0 &&
                reinterpret_cast<std::uintptr_t>(source.cpu.x) % 16 == 0,
            "CPU channel is incomplete");
    // A CPU-served W4A16 miss must take the GPU narrow route's arithmetic, or its output would depend
    // on placement.
    require(source.activation != offloaded_moe::ExpertActivation::kA16 || source.cpu.narrow_a16 == source.narrow_a16,
            "CPU channel computes another W4A16 narrow arithmetic than the call");
}

bool fetch_served(const MoeExpertSource& source) { return source.fetch.request != nullptr; }

} // namespace

bool moe_route_dispatch_plan(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, MoeDispatch& dispatch,
                             std::int32_t* route_log, const Tensor& x, const MoeExpertSource& source,
                             std::int32_t max_jobs, void* workspace, const Tensor& outputs, cudaStream_t stream,
                             RouterLogits precision, bool shared_job) {
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    if (!cpu_served(x, source) || columns > kRouteDispatchColumns ||
        static_cast<std::int64_t>(columns) * (top_k + (shared_job ? 1 : 0)) > kDispatchThreads) {
        moe_route_dispatch(logits, top_k, routing, dispatch, route_log, stream, precision, shared_job);
        return false;
    }
    require(experts + (shared_job ? 1 : 0) <= kDispatchThreads, "dispatch supports up to 1024 experts");
    dispatch.shared_first = shared_job ? columns * top_k : 0x7FFFFFFF;
    require(contiguous(logits, DType::FP32) && contiguous(routing.ids, DType::I32) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32),
            "route requires contiguous FP32 logits and I32/FP32 outputs");
    require(experts > 0 && experts <= 512 && top_k > 0 && top_k <= 16 && top_k <= experts && columns > 0,
            "route supports up to 512 experts and top-16");
    require(routing.ids.ne[0] == top_k && routing.ids.ne[1] == columns && routing.weights.ne[0] == top_k &&
                routing.weights.ne[1] == columns && routing.shared_gate.numel() == columns,
            "route output shapes disagree");
    require(contiguous(x, DType::BF16) && x.ne[1] == columns && max_jobs > 0 && max_jobs <= 65535,
            "route plan needs the experts' BF16 x and a job capacity");
    require_cpu_channel(x, source);
    const bool masked       = masked_call(routing.mask, columns, shared_job);
    std::int32_t* cpu_flags = nullptr;
    CpuCall* cpu_call       = cpu_call_of(workspace, x, source, max_jobs, outputs.ne[1], &cpu_flags);
    CUDA_CHECK(pdl::launch_consumer({dim3(1), dim3(kDispatchThreads), 0, stream},
                                    masked ? route_dispatch_plan_kernel<true> : route_dispatch_plan_kernel<false>,
                                    static_cast<const float*>(logits.data), experts, columns, static_cast<int>(top_k),
                                    precision == RouterLogits::kBf16, shared_job,
                                    static_cast<std::int32_t*>(routing.ids.data),
                                    static_cast<float*>(routing.weights.data),
                                    static_cast<float*>(routing.shared_gate.data), dispatch, route_log,
                                    source, static_cast<const bf16*>(x.data),
                                    static_cast<int>(max_jobs), cpu_flags, cpu_call, routing.mask));
    return true;
}

void moe_experts_cpu_wait(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                          std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream) {
    if (forked(source, max_jobs, x.ne[1])) { CUDA_CHECK(cudaStreamWaitEvent(stream, source.fork_events[1], 0)); }
    if (!cpu_served(x, source)) { return; }
    std::int32_t* cpu_flags = nullptr;
    CpuCall* call    = cpu_call_of(workspace, x, source, max_jobs, outputs.ne[1], &cpu_flags);
    if (source.tail_skip_rank > 0) {
        tail_zero_kernel<<<kWaitCtas, kThreads, 0, stream>>>(dispatch, static_cast<int>(max_jobs), cpu_flags,
                                                             source.shape.hidden, static_cast<bf16*>(outputs.data));
        check_launch("tail zero");
    }
    bool warm        = false;
    for (int i = 0; i < MoeL2Warm::kSpans; ++i) {
        require(source.l2_warm.bytes[i] == 0 || reinterpret_cast<std::uintptr_t>(source.l2_warm.ptr[i]) % 16 == 0,
                "experts L2 warm spans must be 16-byte aligned");
        warm = warm || source.l2_warm.bytes[i] >= 16;
    }
    const int ctas = kWaitCtas + (warm ? kWarmCtas : 0);
    cpu_wait_kernel<<<ctas, kThreads, 0, stream>>>(dispatch, source, call, source.shape.hidden,
                                                   static_cast<bf16*>(outputs.data));
    check_launch("cpu wait");
}

void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream, bool wait_for_cpu, bool cpu_planned) {
    require(contiguous(x, DType::BF16) && contiguous(outputs, DType::BF16), "experts need BF16 x and outputs");
    const bool ggml = ggml_bank(source);
    require(moe::wide::shape_supported(source.shape, source.activation, source.wide_a16,
                                       ggml || source.ggml.gate_up_format == QType::FP8_E4M3FN_B128),
            "experts: this expert geometry is registered for NVFP4 W4A16 banks on the exact, A4 residual and "
            "reference routes");
    // A shared job (design §8.6) adds one entry per column after the k*T routed ones.
    const bool shared_job = source.shared_record != nullptr;
    require(shared_job == (source.shared_expert >= 0) &&
                (!shared_job || (dispatch.shared_first == x.ne[1] * top_k && !ggml &&
                                 source.activation == moe::ExpertActivation::kA16 &&
                                 source.ggml.gate_up_format == QType::NVFP4_MUL)) &&
                (shared_job || dispatch.shared_first == 0x7FFFFFFF),
            "experts: a shared job needs an NVFP4 W4A16 bank, its record and a dispatch made with shared_job");
    require(x.ne[0] == source.shape.hidden && outputs.ne[0] == source.shape.hidden &&
                outputs.ne[1] == x.ne[1] * top_k + (shared_job ? x.ne[1] : 0),
            "experts geometry differs from the bank's record");
    const std::uint64_t record_bytes = ggml ? source.ggml.record_bytes : moe::wide::nvfp4_record_bytes(source.shape);
    require(max_jobs > 0 && max_jobs <= 65535 && (ggml || source.scales != nullptr) && source.frames != nullptr &&
                (source.host_records != nullptr || source.host_table != nullptr) &&
                source.record_stride >= record_bytes && frame_pitch(source) >= record_bytes &&
                source.record_stride % 16 == 0 && frame_pitch(source) % 16 == 0 && source.staging_slots >= 0 &&
                (source.staging_slots == 0 || source.staging_base != nullptr),
            "experts source is incomplete");
    require(source.narrow_a16 == moe::NarrowA16::kExact ||
                (source.activation == moe::ExpertActivation::kA16 && !ggml && source.shape == moe::kFlashNextShape &&
                 source.wide_a16 != moe::WideA16::kLossless),
            "experts: the two-term A4 narrow route needs an NVFP4 W4A16 bank of Flash-Next's geometry");
    require(!ggml || (source.activation == moe::ExpertActivation::kA16 && source.ggml.record_bytes % 16 == 0 &&
                      source.ggml.gate_row_bytes > 0 && source.ggml.down_row_bytes > 0),
            "a ggml bank takes the A16 workspace");
    require(source.landing_slots >= 0 && source.landing_slots <= offloaded_moe::kMaxLandingSlots &&
                (source.landing_slots == 0 || (source.landing != nullptr && source.landed != nullptr)),
            "experts landing is incomplete");
    require(source.prefetched == nullptr || source.prefetch_base != nullptr, "experts prefetch table without its base");
    const auto layout  = moe::wide::carve_experts_workspace(workspace, source.shape, max_jobs, outputs.ne[1], x.ne[1],
                                                            source.activation, source.wide_a16);
    auto** job_records = layout.job_records;
    std::int32_t* cpu_flags = nullptr;
    CpuCall* cpu_call       = cpu_call_of(workspace, x, source, max_jobs, outputs.ne[1], &cpu_flags);
    const bool cpu          = cpu_served(x, source);
    if (cpu) {
        require_cpu_channel(x, source);
        if (!cpu_planned) { // else moe_route_dispatch_plan planned the call
            CUDA_CHECK(pdl::launch_consumer({dim3(1), dim3(kThreads), 0, stream}, cpu_plan_kernel, dispatch,
                                            source, static_cast<const bf16*>(x.data),
                                            static_cast<int>(x.ne[1]), top_k, static_cast<int>(max_jobs), cpu_flags,
                                            cpu_call));
        }
    }
    FetchCall* fetch = nullptr;
    if (fetch_served(source)) {
        require(source.host_table != nullptr && source.fetch.response != nullptr && source.fetch.consumed != nullptr &&
                    source.fetch.sequence != nullptr && source.fetch.heartbeat != nullptr &&
                    source.staging_slots > 0 && max_jobs <= offloaded_moe::kMaxFetch,
                "experts fetch channel is incomplete (it needs a host table, staging and at most kMaxFetch jobs)");
        fetch = fetch_call_of(workspace, x, source, max_jobs, outputs.ne[1]);
        fetch_plan_kernel<<<1, kThreads, 0, stream>>>(dispatch, source, max_jobs, cpu ? cpu_flags : nullptr,
                                                      cpu ? cpu_call : nullptr, fetch);
        check_launch("fetch plan");
    }
    const std::int32_t* flags = cpu ? cpu_flags : nullptr;
    if (source.wide_a16 == moe::WideA16::kLossless) {
        require(!cpu && source.landing_slots == 0, "the lossless route computes every expert on the GPU, in serial passes");
        lossless_experts(x, dispatch, source, top_k, max_jobs, layout, fetch, outputs, stream);
        return;
    }
    // The narrow ring's x, encoded once here, before any fork (its job words and counters cleared).
    if (layout.x_acts != nullptr) { launch_ring_prepare(x, layout, static_cast<int>(max_jobs), stream); }
    // A ggml bank's gate/up kernels read every token's x limbs, encoded once here, before any fork.
    if (ggml) {
        launch_encode_x_ggml(x, layout, stream);
        if (x.ne[1] > moe::kMaxColumns) { launch_plan_ggml(dispatch, max_jobs, layout, stream); }
    }
    // Passes of at most staging_slots jobs (one pass covering every job without staging).
    const int pass_jobs = source.staging_slots > 0 ? std::min(source.staging_slots, kMaxPassJobs) : max_jobs;
    const int half      = source.staging_slots / 2;
    // In a call of more than eight columns, experts with more than eight take the wide route, pass
    // by pass once each pass's records are resolved; it is planned and x quantized before the passes.
    const bool use_wide = !ggml && x.ne[1] > moe::kMaxColumns && !forked(source, max_jobs, x.ne[1]);
    // The staging overlap is not combined with the wide route: in the Program, overlapped calls with
    // wide experts gave run-to-run differences (prefix-cache exactness failures, varying greedy ids;
    // design §19.3.8, "F1 nondeterminism") that serial passes do not; the cause is not identified.
    const bool overlap  = source.overlap_stream != nullptr && half > 0 && max_jobs > half && !use_wide;
    std::optional<moe::wide::Call> wide;
    if (overlap) {
        // Double-buffered passes: stage pass p+1 on the side stream while pass p computes.
        const auto* events = source.overlap_events;
        CUDA_CHECK(cudaEventRecord(events[0], stream));
        CUDA_CHECK(cudaStreamWaitEvent(source.overlap_stream, events[0], 0));
        // Planned after the fork, so the first pass's staging overlaps the plan and quantization.
        if (use_wide) { wide = moe::wide::prepare(x, dispatch, source, top_k, max_jobs, half, layout, outputs, stream); }
        int pass = 0;
        for (int base = 0; base < max_jobs; base += half, ++pass) {
            const int jobs = std::min(half, max_jobs - base);
            const int b    = pass % 2;
            MoeExpertSource buffer = source;
            buffer.staging_base    = source.staging_base + static_cast<std::uint64_t>(b) * half * frame_pitch(source);
            buffer.staging_slots   = half;
            buffer.landing         = nullptr; // landing is for forked calls only
            buffer.landing_slots   = 0;
            if (pass >= 2) { CUDA_CHECK(cudaStreamWaitEvent(source.overlap_stream, events[3 + b], 0)); }
            stage_kernel<<<kStageCtas, kThreads, 0, source.overlap_stream>>>(dispatch, buffer, base, jobs, flags,
                                                                            job_records, fetch);
            check_launch("stage");
            CUDA_CHECK(cudaEventRecord(events[1 + b], source.overlap_stream));
            CUDA_CHECK(cudaStreamWaitEvent(stream, events[1 + b], 0));
            launch_pass(x.ne[1], static_cast<const bf16*>(x.data), dispatch, buffer, top_k, job_records, flags, base, jobs,
                        layout, outputs.ne[1], static_cast<bf16*>(outputs.data), stream);
            if (wide) { moe::wide::run_pass(*wide, pass, stream); }
            CUDA_CHECK(cudaEventRecord(events[3 + b], stream));
        }
        if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
        return;
    }
    if (forked(source, max_jobs, x.ne[1])) {
        // One pass: the misses are staged and then computed on the fork stream while this stream
        // computes the resident experts; both write disjoint outputs and h blocks, and every
        // expert's arithmetic is the same as in a serial pass.
        CUDA_CHECK(cudaEventRecord(source.fork_events[0], stream));
        CUDA_CHECK(cudaStreamWaitEvent(source.fork_stream, source.fork_events[0], 0));
        stage_kernel<<<kStageCtas, kThreads, 0, source.fork_stream>>>(dispatch, source, 0, max_jobs, flags,
                                                                      job_records, fetch);
        check_launch("stage");
        const auto* xs = static_cast<const bf16*>(x.data);
        auto* out      = static_cast<bf16*>(outputs.data);
        launch_pass(x.ne[1], xs, dispatch, source, top_k, job_records, flags, 0, max_jobs, layout, outputs.ne[1], out,
                    source.fork_stream, PassPhase::Staged);
        launch_pass(x.ne[1], xs, dispatch, source, top_k, job_records, flags, 0, max_jobs, layout, outputs.ne[1], out,
                    stream, PassPhase::Resident);
        CUDA_CHECK(cudaEventRecord(source.fork_events[1], source.fork_stream));
        // Joined here, or in moe_experts_cpu_wait so the caller's unrelated work (the shared
        // expert) also overlaps the staging.
        if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
        return;
    }
    if (use_wide) { wide = moe::wide::prepare(x, dispatch, source, top_k, max_jobs, pass_jobs, layout, outputs, stream); }
    MoeExpertSource unlanded = source; // landing is for forked calls only
    unlanded.landing         = nullptr;
    unlanded.landing_slots   = 0;
    for (int base = 0; base < max_jobs; base += pass_jobs) {
        const int jobs = std::min(pass_jobs, max_jobs - base);
        CUDA_CHECK(pdl::launch_consumer({dim3(source.staging_slots > 0 ? kStageCtas : 1), dim3(kThreads), 0, stream},
                                        stage_kernel, dispatch, unlanded, base, jobs, flags, job_records, fetch));
        launch_pass(x.ne[1], static_cast<const bf16*>(x.data), dispatch, source, top_k, job_records, flags, base, jobs,
                    layout, outputs.ne[1], static_cast<bf16*>(outputs.data), stream);
        if (wide) { moe::wide::run_pass(*wide, base / pass_jobs, stream); }
    }
    if (wait_for_cpu) { moe_experts_cpu_wait(x, dispatch, source, max_jobs, workspace, outputs, stream); }
}

namespace offloaded_moe::wide {

std::size_t lossless_experts_bytes(const ExpertShape& shape, std::int32_t max_jobs, std::int32_t entries) {
    return carve_lossless(nullptr, shape, max_jobs, entries).bytes;
}

} // namespace offloaded_moe::wide

void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream) {
    require(contiguous(outputs, DType::BF16) && contiguous(shared, DType::BF16) && contiguous(y, DType::BF16),
            "combine requires contiguous BF16 tensors");
    const int hidden = y.ne[0], columns = y.ne[1], top_k = routing.weights.ne[0];
    require(outputs.ne[0] == hidden && outputs.ne[1] == columns * top_k && shared.ne[0] == hidden &&
                shared.ne[1] == columns && routing.weights.ne[1] == columns,
            "combine shapes disagree");
    // The dispatch checked the mask's rows and shared job; the combine needs only its rows.
    const bool masked = routing.mask.valid != nullptr;
    require(!masked || (routing.mask.width > 0 && columns % routing.mask.width == 0),
            "a column mask needs whole rows of the call's columns");
    const dim3 grid((hidden + kThreads - 1) / kThreads, columns);
    const auto* routed_out = static_cast<const bf16*>(outputs.data);
    const auto* weights    = static_cast<const float*>(routing.weights.data);
    const auto* gate       = static_cast<const float*>(routing.shared_gate.data);
    const auto* shared_out = static_cast<const bf16*>(shared.data);
    if (masked) {
        combine_kernel<true><<<grid, kThreads, 0, stream>>>(routed_out, weights, gate, shared_out, hidden, top_k, columns,
                                                            static_cast<bf16*>(y.data), routing.mask);
    } else {
        combine_kernel<false><<<grid, kThreads, 0, stream>>>(routed_out, weights, gate, shared_out, hidden, top_k,
                                                             columns, static_cast<bf16*>(y.data), routing.mask);
    }
    check_launch("combine");
}

void moe_combine_inject(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared,
                        const Tensor& inject, Tensor& residual, cudaStream_t stream) {
    require(contiguous(outputs, DType::BF16) && contiguous(shared, DType::BF16) &&
                contiguous(residual, DType::BF16) && contiguous(inject, DType::FP32),
            "combine inject requires contiguous BF16 tensors and FP32 inject");
    require(routing.mask.valid == nullptr, "combine inject takes no column mask");
    const int hidden = shared.ne[0], columns = shared.ne[1], top_k = routing.weights.ne[0];
    const int streams = inject.ne[0];
    require(outputs.ne[0] == hidden && outputs.ne[1] == columns * top_k && routing.weights.ne[1] == columns &&
                inject.ne[1] == columns && streams > 0 && residual.ne[0] == streams * hidden &&
                residual.ne[1] == columns,
            "combine inject shapes disagree");
    CUDA_CHECK(pdl::launch_consumer({dim3((hidden + kThreads - 1) / kThreads, columns), dim3(kThreads), 0, stream},
                                    combine_inject_kernel, static_cast<const bf16*>(outputs.data),
                                    static_cast<const float*>(routing.weights.data),
                                    static_cast<const float*>(routing.shared_gate.data),
                                    static_cast<const bf16*>(shared.data), static_cast<const float*>(inject.data),
                                    hidden, top_k, streams, columns, static_cast<bf16*>(residual.data)));
}

const MoeGgmlTiles& MoeGgmlTiles::from_environment() {
    static const MoeGgmlTiles tiles = [] {
        // A non-negative column count, else the default (unset).
        const auto columns = [](const char* name, std::int32_t fallback) {
            return static_cast<std::int32_t>(
                temporary_switches::integer(name, fallback, 0, std::numeric_limits<std::int32_t>::max()));
        };
        MoeGgmlTiles t;
        t.stage_decode_min_columns = columns("INFERNIX_TMP_IQ_STAGE_DECODE_MIN_COLUMNS", t.stage_decode_min_columns);
        t.stage_decode_max_columns = columns("INFERNIX_TMP_IQ_STAGE_DECODE_MAX_COLUMNS", t.stage_decode_max_columns);
        t.plane_min_columns        = columns("INFERNIX_TMP_IQ_PLANE_MIN_COLUMNS", t.plane_min_columns);
        t.shared_codebooks         = temporary_switches::flag("INFERNIX_TMP_GGML_SHARED_CODEBOOKS");
        return t;
    }();
    return tiles;
}

} // namespace infernix::ops
