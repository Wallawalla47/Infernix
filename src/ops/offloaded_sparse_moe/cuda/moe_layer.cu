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
#include "core/pdl.cuh"

#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_exact.h"
#include "ops/ggml/ggml_bf16_tile.cuh"
#include "ops/ggml/ggml_tile.cuh"
#include "ops/fp8_block/fp8_exact.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

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

// Bytes a stage copies: a ggml bank's record, else the NVFP4 record.
__device__ __forceinline__ int record_copy_bytes(const MoeExpertSource& source) {
    return source.ggml.record_bytes != 0 ? static_cast<int>(source.ggml.record_bytes) : static_cast<int>(moe::kRecordBytes);
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

__device__ __forceinline__ const std::uint8_t* pass_record(PassPhase phase, const MoeExpertSource& source,
                                                           int expert, const std::uint8_t* const* job_records,
                                                           int job) {
    if (phase == PassPhase::All) { return job_records[job]; }
    const bool resident = source.frames[expert] >= 0;
    if (resident != (phase == PassPhase::Resident)) { return nullptr; }
    return resident ? source.frame_base + static_cast<std::uint64_t>(source.frames[expert]) * frame_pitch(source)
                    : job_records[job];
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
            fetch = source.frames[expert] < 0 && source.host_table[expert] == nullptr && flag == 0 &&
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
            expert          = dispatch.jobs[job_base + j];
            const int frame = source.frames[expert];
            const int slot  = source.prefetched != nullptr ? source.prefetched[expert] : -1;
            if (frame >= 0) {
                record = source.frame_base + static_cast<std::uint64_t>(frame) * frame_pitch(source);
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

// One warp routes column t: exact top-k by repeated arg-max with lower-id ties, then the weights.
// Every array index is a compile-time constant (unrolled loops), so the values stay in registers;
// the selection and softmax order are fixed. Lane k < top_k returns the k-th selected id.
__device__ __forceinline__ int route_column(const float* __restrict__ logits, int experts, int top_k, int t,
                                            std::int32_t* __restrict__ ids, float* __restrict__ weights,
                                            float* __restrict__ shared_gate) {
    const int lane      = threadIdx.x % 32;
    const float* column = logits + static_cast<std::size_t>(t) * (experts + 1);
    // 512 experts: 16 per lane.
    constexpr int kPerLane = 16;
    float values[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        values[i]   = e < experts ? column[e] : -INFINITY;
    }
    const float score = lane == 0 ? column[experts] : 0.0F;
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
                             std::int32_t* __restrict__ ids, float* __restrict__ weights,
                             float* __restrict__ shared_gate) {
    pdl::enter(); // launched as a programmatic dependent of the router projection
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (warp < columns) { route_column(logits, experts, top_k, warp, ids, weights, shared_gate); }
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
// The caller has cleared `count` and synchronized; thread i < entries holds entry i's expert `id`.
__device__ __forceinline__ void dispatch_small(int id, int entries, int experts, const MoeDispatch& dispatch,
                                               std::int32_t* __restrict__ route_log,
                                               std::int32_t (&count)[kDispatchThreads],
                                               std::int32_t (&cursor)[kDispatchThreads],
                                               std::int32_t (&warp_totals)[2][kDispatchThreads / 32]) {
    const int i = threadIdx.x;
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

__global__ void __launch_bounds__(kDispatchThreads)
    dispatch_small_kernel(const std::int32_t* __restrict__ ids, int entries, int experts, MoeDispatch dispatch,
                          std::int32_t* __restrict__ route_log) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    pdl::enter(); // launched as a programmatic dependent of the route
    const int i = threadIdx.x;
    count[i]    = 0;
    __syncthreads();
    dispatch_small(i < entries ? ids[i] : 0, entries, experts, dispatch, route_log, count, cursor, warp_totals);
}

// Calls of at most this many columns (and kDispatchThreads entries) route and dispatch in one CTA,
// one warp per column: the dispatch reads the ids from shared memory, with no kernel boundary
// between the two. Same results as route_kernel followed by dispatch_small_kernel.
constexpr int kRouteDispatchColumns = kDispatchThreads / 32;

__global__ void __launch_bounds__(kDispatchThreads)
    route_dispatch_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                          std::int32_t* __restrict__ ids, float* __restrict__ weights,
                          float* __restrict__ shared_gate, MoeDispatch dispatch,
                          std::int32_t* __restrict__ route_log) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t routed[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    pdl::enter(); // launched as a programmatic dependent of the router projection
    const int i = threadIdx.x, warp = i / 32, lane = i % 32;
    count[i]    = 0;
    if (warp < columns) {
        const int id = route_column(logits, experts, top_k, warp, ids, weights, shared_gate);
        if (lane < top_k) { routed[warp * top_k + lane] = id; }
    }
    __syncthreads();
    const int entries = columns * top_k;
    dispatch_small(i < entries ? routed[i] : 0, entries, experts, dispatch, route_log, count, cursor, warp_totals);
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
//
// The W4A4 narrow kernels compute one column of a job at a time (a job of n columns makes n passes
// over its staged weights): at op level (frames, cold L2, uniform routing) they beat an eight-column
// form 1.26-2.3x at every width from one to 64 columns per expert, which was removed 2026-10-11.
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

constexpr int kGateUpSliceBytes = kGateUpGroups * moe::kGateUpBlocks * static_cast<int>(moe::kUnitBytes); // 46,080
constexpr int kDownSliceBytes   = kDownGroups * moe::kDownBlocks * static_cast<int>(moe::kUnitBytes);     // 23,040
static_assert(kGateUpSliceBytes % 16 == 0 && kDownSliceBytes % 16 == 0);

struct GateUpShared {
    alignas(16) std::uint8_t stage[kGateUpSliceBytes];
    // The activations are read only by unit_sums and the partial sums are written only after the
    // barrier that follows it, so they share storage. That keeps the CTA (49,376 B) under half of an
    // SM's 100 KB, so two of them fit per SM.
    union {
        ActCodes acts[moe::kGateUpBlocks];
        std::int64_t partial[kWarps][16 * kGateUpGroups];
    };
    std::uint16_t y[16 * kGateUpGroups];
    std::uint16_t h[16];
};

static_assert(sizeof(GateUpShared) + 1024 <= 102400 / 2, "two gate/up CTAs must fit an SM");

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
                sm.y[row] = canon::a4_row_output(total, gate ? scales.alpha_gate : scales.alpha_up);
            }
        }
        __syncthreads();
        for (int k = threadIdx.x; k < 16; k += blockDim.x) { sm.h[k] = canon::swiglu_bf16(sm.y[2 * k], sm.y[2 * k + 1]); }
        __syncthreads();
        if (threadIdx.x == 0) {
            h_blocks[static_cast<std::size_t>(first + c) * kHBlocks + slice] = canon::quantize_a4_block(sm.h, scales.input_down);
        }
    }
}

struct DownShared {
    alignas(16) std::uint8_t stage[kDownSliceBytes];
    ActCodes acts[kHBlocks];
    std::int64_t partial[kWarps][16 * kDownGroups];
};

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

void launch_expert_pass(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                        const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                        canon::A4Block* h_blocks, int columns_out, bf16* outputs, cudaStream_t stream,
                        PassPhase phase = PassPhase::All) {
    static const bool configured = [] {
        cudaFuncSetAttribute(gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(GateUpShared)));
        cudaFuncSetAttribute(down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sizeof(DownShared)));
        cudaFuncSetAttribute(gate_up_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(down_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        return true;
    }();
    (void)configured;
    gate_up_kernel<<<dim3(kGateUpCtas, jobs), kThreads, sizeof(GateUpShared), stream>>>(
        x, moe::kHidden, dispatch, source, top_k, job_records, flags, base, h_blocks, phase);
    check_launch("gate/up");
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_consumer({dim3(kDownCtas, jobs), dim3(kThreads), sizeof(DownShared), stream}, down_kernel,
                                    dispatch, source, static_cast<int>(moe::kHidden), job_records, flags, base,
                                    static_cast<const canon::A4Block*>(h_blocks), columns_out, outputs, phase));
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

// Encodes `n` <= Columns BF16 columns of `length` elements, column c at base(c), into acts
// [c][length / 16] with their exponents in emax[c]. Every thread of the CTA calls it; it ends with
// a barrier.
template <int Columns, class Base>
__device__ void encode_a16_columns(Base base, int n, int length, A16Act* acts, int* emax,
                                   unsigned (&key)[kWarps][Columns]) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int vectors = length / 8; // uint4 loads of 8 elements
    unsigned m[Columns];
#pragma unroll
    for (int c = 0; c < Columns; ++c) { m[c] = 0; }
    for (int i = threadIdx.x; i < n * vectors; i += blockDim.x) {
        const int c   = i / vectors;
        const uint4 v = reinterpret_cast<const uint4*>(base(c))[i % vectors];
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
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < n) {
        unsigned k = 0;
        for (int w = 0; w < kWarps; ++w) { k = max(k, key[w][threadIdx.x]); }
        emax[threadIdx.x] = k >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(k >> 7), 1);
    }
    __syncthreads();
    const int blocks = length / 16;
    for (int i = threadIdx.x; i < n * blocks; i += blockDim.x) {
        const int c = i / blocks, b = i % blocks;
        const int e = emax[c];
        const auto* src = reinterpret_cast<const uint4*>(base(c) + 16 * b);
        const uint4 v0 = src[0], v1 = src[1];
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
    __syncthreads();
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

template <int Columns>
struct GateUpShared16 {
    alignas(16) std::uint8_t stage[kGateUpSliceBytes];
    union { // acts are read only before the barrier that precedes the partial sums
        A16Act acts[Columns * moe::kGateUpBlocks];
        std::int64_t partial[kWarps][16 * kGateUpGroups][Columns];
    };
    std::uint16_t y[Columns][16 * kGateUpGroups];
    int emax[Columns];
    unsigned key[kWarps][Columns];
};

template <int Columns>
__global__ void __launch_bounds__(kThreads)
    gate_up_kernel_a16(const bf16* __restrict__ x, int hidden, MoeDispatch dispatch, MoeExpertSource source,
                       int top_k, const std::uint8_t* const* __restrict__ job_records,
                       const std::int32_t* __restrict__ cpu_flags, int job_base, std::uint16_t* __restrict__ h_plane,
                       PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<GateUpShared16<Columns>*>(smem_raw);
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
    stage_async(record + static_cast<std::size_t>(kGateUpGroups * slice) * moe::kGateUpBlocks * moe::kUnitBytes,
                sm.stage, kGateUpSliceBytes);
    for (int pass = 0; pass < count; pass += Columns) {
        const int n = min(Columns, count - pass);
        __syncthreads();
        const auto column = [&](int c) {
            return reinterpret_cast<const std::uint16_t*>(x) +
                   static_cast<std::size_t>(dispatch.entries[first + pass + c] / top_k) * hidden;
        };
        encode_a16_columns<Columns>(column, n, moe::kHidden, sm.acts, sm.emax, sm.key);
        stage_wait();
        // The down pass (a programmatic dependent in captured graphs) may launch once every CTA has
        // its slice.
        pdl::trigger_dependents();
        __syncthreads();
        std::int64_t s[kGateUpGroups][Columns];
        unit_sums_a16<kGateUpGroups, Columns>(sm.stage, moe::kGateUpBlocks, sm.acts, n, s);
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
            h_plane[static_cast<std::size_t>(first + pass + c) * moe::wide::kA16HRowElements + 16 * slice + k] =
                canon::swiglu_bf16(sm.y[c][2 * k], sm.y[c][2 * k + 1]);
        }
    }
}

template <int Columns>
struct DownShared16 {
    alignas(16) std::uint8_t stage[kDownSliceBytes];
    A16Act acts[Columns * kHBlocks];
    std::int64_t partial[kWarps][16 * kDownGroups][Columns];
    int emax[Columns];
    unsigned key[kWarps][Columns];
};

template <int Columns>
__global__ void __launch_bounds__(kThreads)
    down_kernel_a16(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                    const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                    int job_base, const std::uint16_t* __restrict__ h_plane, bf16* __restrict__ outputs,
                    PassPhase phase) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm      = *reinterpret_cast<DownShared16<Columns>*>(smem_raw);
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
        stage_async(record + moe::kGateUpBytes +
                        static_cast<std::size_t>(kDownGroups * tile) * moe::kDownBlocks * moe::kUnitBytes,
                    sm.stage, kDownSliceBytes);
    }
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int pass = 0; pass < count; pass += Columns) {
        const int n = min(Columns, count - pass);
        __syncthreads();
        const auto column = [&](int c) {
            return h_plane + static_cast<std::size_t>(first + pass + c) * moe::wide::kA16HRowElements;
        };
        encode_a16_columns<Columns>(column, n, moe::kIntermediate, sm.acts, sm.emax, sm.key);
        stage_wait();
        __syncthreads();
        std::int64_t s[kDownGroups][Columns];
        unit_sums_a16<kDownGroups, Columns>(sm.stage, moe::kDownBlocks, sm.acts, n, s);
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

template <int Columns>
void launch_expert_pass_a16(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                            const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                            std::uint16_t* h_plane, bf16* outputs, cudaStream_t stream,
                            PassPhase phase = PassPhase::All) {
    static const bool configured = [] {
        cudaFuncSetAttribute(gate_up_kernel_a16<Columns>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(GateUpShared16<Columns>)));
        cudaFuncSetAttribute(down_kernel_a16<Columns>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(sizeof(DownShared16<Columns>)));
        cudaFuncSetAttribute(gate_up_kernel_a16<Columns>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(down_kernel_a16<Columns>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        return true;
    }();
    (void)configured;
    gate_up_kernel_a16<Columns><<<dim3(kGateUpCtas, jobs), kThreads, sizeof(GateUpShared16<Columns>), stream>>>(
        x, moe::kHidden, dispatch, source, top_k, job_records, flags, base, h_plane, phase);
    check_launch("A16 gate/up");
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_consumer({dim3(kDownCtas, jobs), dim3(kThreads), sizeof(DownShared16<Columns>), stream},
                                    down_kernel_a16<Columns>, dispatch, source, static_cast<int>(moe::kHidden),
                                    job_records, flags, base, static_cast<const std::uint16_t*>(h_plane), outputs,
                                    phase));
}

#include "ops/offloaded_sparse_moe/cuda/ggml_expert.cuh"
#include "ops/offloaded_sparse_moe/cuda/fp8_expert.cuh"

// One pass of the narrow kernels in the source's arithmetic, for a call of `columns` columns. W4A4
// jobs of any width take the one-column kernels (one pass per column over the staged weights); W4A16
// jobs take them in calls of at most kA16OneColumnCalls columns, else the multi-column kernels. A GGUF
// import's or block-FP8 bank runs its own exact kernels.
void launch_pass(int columns, const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                 const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                 const moe::wide::ExpertsWorkspace& layout, int columns_out, bf16* outputs, cudaStream_t stream,
                 PassPhase phase = PassPhase::All) {
    // Wide calls (more than kMaxColumns columns) were planned by moe_experts.
    const bool wide = columns_out / top_k > moe::kMaxColumns;
    if (source.ggml.gate_up_format == QType::FP8_E4M3FN_B128) {
        launch_expert_pass_fp8(dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs, stream, phase,
                               wide ? columns_out : 0);
        return;
    }
    if (ggml_bank(source)) {
        launch_pass_ggml(x, dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs, stream, phase,
                         wide ? columns_out : 0);
        return;
    }
    if (source.activation == moe::ExpertActivation::kA16) {
        auto* h_plane = reinterpret_cast<std::uint16_t*>(layout.h_rows);
        if (columns <= kA16OneColumnCalls) {
            launch_expert_pass_a16<1>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_plane, outputs,
                                      stream, phase);
        } else {
            launch_expert_pass_a16<kA16PassColumns>(x, dispatch, source, top_k, job_records, flags, base, jobs, h_plane,
                                                    outputs, stream, phase);
        }
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
// assist); d is pcie_divisor, or one_column_divisor (when set) in a one-column call (none for d = 0).
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

template <bool Named>
__device__ __forceinline__ void cpu_plan_body(const MoeDispatch& dispatch, const MoeExpertSource& source,
                                              const bf16* __restrict__ x, int columns, int top_k, int max_jobs,
                                              std::int32_t* __restrict__ cpu_flags, CpuCall* __restrict__ call) {
    constexpr int kWidths    = offloaded_moe::kMaxCpuColumns;
    // A miss's category: its width w for an SSD-only expert (no host record, MoeExpertSource::
    // host_table), kWidths + w for one in host memory. SSD-only experts are chosen first (only the
    // CPU can serve them), then host-memory misses under the PCIe divisor rule; without a host table
    // every miss is in host memory and the choice is the width-ordered one alone.
    constexpr int kCats      = 2 * kWidths;
    constexpr int kMaskWords = offloaded_moe::kMaxCpuCallColumns / 32;
    __shared__ int width_misses[kCats + 1]; // eligible misses of each category
    __shared__ int warp_misses[kWarps][kCats + 1];
    __shared__ int take[kCats + 1], first_slot[kCats + 1], carry[kCats + 1];
    __shared__ unsigned used[kMaskWords];                                // x columns the chosen jobs read
    __shared__ short compact[offloaded_moe::kMaxCpuCallColumns];         // column -> compact index
    __shared__ short published[offloaded_moe::kMaxCpuXColumns];         // compact index -> column
    __shared__ int slot_first[offloaded_moe::kMaxCpuJobs];               // a chosen job's first dispatch entry
    __shared__ int misses, chosen, publish_count;
    const auto& channel = source.cpu;
    const int jobs      = min(*dispatch.job_count, max_jobs);
    const int max_width = channel.max_job_columns;
    const int cap = channel.wide_from > 0 && columns >= channel.wide_from ? channel.wide_jobs : channel.max_jobs;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int j = threadIdx.x; j < max_jobs; j += kThreads) { cpu_flags[j] = 0; }
    if (threadIdx.x <= kCats) {
        width_misses[threadIdx.x] = 0;
        carry[threadIdx.x]        = 0;
    }
    if (threadIdx.x < kMaskWords) { used[threadIdx.x] = 0; }
    if (threadIdx.x == 0) { misses = 0; }
    plan_barrier<Named>();
    // M counts every host-memory miss; only misses of at most max_width columns can be chosen.
    int own = 0;
    for (int j = threadIdx.x; j < jobs; j += kThreads) {
        const int expert = dispatch.jobs[j];
        // A streamed record counts as resident: neither staged nor CPU-served.
        if (source.frames[expert] >= 0 || (source.prefetched != nullptr && source.prefetched[expert] >= 0)) { continue; }
        const int count = dispatch.offsets[expert + 1] - dispatch.offsets[expert];
        // Tail skip (MoeExpertSource::tail_skip_rank): a miss every column routes at slot >= R only is
        // flagged 2, which every route passes over like a CPU job, and is not counted as a miss. Experts
        // of more than kMaxColumns columns are kept (the wide route does not read the flags).
        if (source.tail_skip_rank > 0 && count <= moe::kMaxColumns) {
            const int first = dispatch.offsets[expert];
            bool tail       = true;
            for (int c = 0; c < count && tail; ++c) { tail = dispatch.entries[first + c] % top_k >= source.tail_skip_rank; }
            if (tail) {
                cpu_flags[j] = 2;
                continue;
            }
        }
        const bool ssd  = source.host_table != nullptr && source.host_table[expert] == nullptr;
        own += ssd ? 0 : 1;
        if (count <= max_width) { atomicAdd(&width_misses[(ssd ? 0 : kWidths) + count], 1); }
    }
    own = __reduce_add_sync(0xFFFFFFFFU, own);
    if (lane == 0) { atomicAdd(&misses, own); }
    plan_barrier<Named>();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int w = 1; w <= kWidths; ++w) { // SSD-only, up to the cap
            first_slot[w] = n;
            take[w]       = min(width_misses[w], cap - n);
            n += take[w];
        }
        const int one     = channel.one_column_divisor_device != nullptr && columns == 1 ? *channel.one_column_divisor_device
                                                                                         : channel.one_column_divisor;
        const int divisor = columns == 1 && one > 0 ? one : channel.pcie_divisor;
        const int want    = min(cap - n, divisor > 0 ? misses - misses / divisor : misses);
        int m = 0;
        for (int w = 1; w <= kWidths; ++w) {
            first_slot[kWidths + w] = n + m;
            take[kWidths + w]       = min(width_misses[kWidths + w], want - m);
            m += take[kWidths + w];
        }
        chosen     = n + m;
        call->jobs = n + m;
    }
    plan_barrier<Named>();
    if (chosen == 0) {
        if (threadIdx.x == 0) { call->pending = 0; }
        return;
    }
    // Rank r of a width-w miss: earlier width-w misses in job order (earlier chunks, earlier warps,
    // lower lanes); it is chosen when r < take[w] and goes to slot first_slot[w] + r.
    for (int base = 0; base < jobs; base += kThreads) {
        const int j = base + threadIdx.x;
        int width = 0, cat = 0, expert = -1, first = 0;
        if (j < jobs) {
            expert = dispatch.jobs[j];
            if (source.frames[expert] < 0 && (source.prefetched == nullptr || source.prefetched[expert] < 0) &&
                cpu_flags[j] != 2) { // a skipped tail expert is not a miss to serve
                first           = dispatch.offsets[expert];
                const int count = dispatch.offsets[expert + 1] - first;
                width           = count <= max_width ? count : 0;
                const bool ssd  = source.host_table != nullptr && source.host_table[expert] == nullptr;
                cat             = width > 0 ? (ssd ? 0 : kWidths) + width : 0;
            }
        }
        int rank = 0;
#pragma unroll
        for (int c = 1; c <= kCats; ++c) {
            const unsigned ballot = __ballot_sync(0xFFFFFFFFU, cat == c);
            if (lane == 0) { warp_misses[warp][c] = __popc(ballot); }
            if (cat == c) { rank = __popc(ballot & ((1U << lane) - 1U)); }
        }
        plan_barrier<Named>();
        if (cat > 0) {
            rank += carry[cat];
            for (int v = 0; v < warp; ++v) { rank += warp_misses[v][cat]; }
            if (rank < take[cat]) {
                const int slot                = first_slot[cat] + rank;
                cpu_flags[j]                  = 1;
                call->job[slot]               = j;
                channel.request->expert[slot] = expert;
                channel.request->record[slot] =
                    source.host_table != nullptr ? reinterpret_cast<std::uint64_t>(source.host_table[expert]) : 0;
                channel.request->ncols[slot]  = width;
                slot_first[slot]              = first;
                for (int c = 0; c < width; ++c) {
                    const int column = dispatch.entries[first + c] / top_k;
                    atomicOr(&used[column / 32], 1U << (column % 32));
                }
            }
        }
        plan_barrier<Named>();
        if (threadIdx.x >= 1 && threadIdx.x <= kCats) {
            for (int v = 0; v < kWarps; ++v) { carry[threadIdx.x] += warp_misses[v][threadIdx.x]; }
        }
        plan_barrier<Named>();
    }
    if (threadIdx.x == 0) {
        int k = 0;
        for (int t = 0; t < columns; ++t) {
            if ((used[t / 32] >> (t % 32)) & 1U) {
                compact[t]     = static_cast<short>(k);
                published[k++] = static_cast<short>(t);
            }
        }
        publish_count          = k;
        channel.request->layer  = channel.layer;
        channel.request->jobs   = chosen;
        channel.request->tiered = source.host_table != nullptr ? 1 : 0;
    }
    plan_barrier<Named>();
    // Each chosen job's columns by compact index.
    for (int i = threadIdx.x; i < chosen * offloaded_moe::kMaxCpuColumns; i += kThreads) {
        const int slot = i / offloaded_moe::kMaxCpuColumns, c = i % offloaded_moe::kMaxCpuColumns;
        if (c < channel.request->ncols[slot]) {
            channel.request->column[slot][c] = compact[dispatch.entries[slot_first[slot] + c] / top_k];
        }
    }
    // The chosen jobs' columns, compacted, 16 bytes per store.
    constexpr int kVectors = moe::kHidden * 2 / 16;
    const auto* src        = reinterpret_cast<const int4*>(x);
    auto* dst              = reinterpret_cast<int4*>(channel.x);
    for (int i = threadIdx.x; i < publish_count * kVectors; i += kThreads) {
        const int k = i / kVectors, v = i % kVectors;
        dst[static_cast<std::size_t>(k) * kVectors + v] = src[static_cast<std::size_t>(published[k]) * kVectors + v];
    }
    __threadfence_system();
    plan_barrier<Named>();
    if (threadIdx.x == 0) {
        // 0 means "nothing published" to the wait (CpuCall::pending): skip it when the counter wraps.
        std::uint32_t sequence = atomicAdd(channel.sequence, 1U) + 1U;
        if (sequence == 0) { sequence = atomicAdd(channel.sequence, 1U) + 1U; }
        *reinterpret_cast<volatile std::uint32_t*>(&channel.request->sequence) = sequence;
        __threadfence_system();
        call->pending = static_cast<std::int32_t>(sequence);
    }
}

__global__ void __launch_bounds__(kThreads)
    cpu_plan_kernel(MoeDispatch dispatch, MoeExpertSource source, const bf16* __restrict__ x, int columns,
                    int top_k, int max_jobs, std::int32_t* __restrict__ cpu_flags, CpuCall* __restrict__ call) {
    pdl::enter(); // launched as a programmatic dependent of the dispatch
    cpu_plan_body<false>(dispatch, source, x, columns, top_k, max_jobs, cpu_flags, call);
}

// route_dispatch_kernel followed by the CPU plan in the same CTA: after a barrier of the whole CTA
// makes the dispatch visible, its first kThreads threads plan as cpu_plan_kernel does (the same
// results) and the rest exit, so the plan, and the host's start, follow the dispatch with no kernel
// boundary.
__global__ void __launch_bounds__(kDispatchThreads)
    route_dispatch_plan_kernel(const float* __restrict__ logits, int experts, int columns, int top_k,
                               std::int32_t* __restrict__ ids, float* __restrict__ weights,
                               float* __restrict__ shared_gate, MoeDispatch dispatch,
                               std::int32_t* __restrict__ route_log, MoeExpertSource source,
                               const bf16* __restrict__ x, int max_jobs, std::int32_t* __restrict__ cpu_flags,
                               CpuCall* __restrict__ call) {
    __shared__ std::int32_t count[kDispatchThreads];
    __shared__ std::int32_t cursor[kDispatchThreads];
    __shared__ std::int32_t routed[kDispatchThreads];
    __shared__ std::int32_t warp_totals[2][kDispatchThreads / 32];
    pdl::enter(); // launched as a programmatic dependent of the router projection
    const int i = threadIdx.x, warp = i / 32, lane = i % 32;
    count[i]    = 0;
    if (warp < columns) {
        const int id = route_column(logits, experts, top_k, warp, ids, weights, shared_gate);
        if (lane < top_k) { routed[warp * top_k + lane] = id; }
    }
    __syncthreads();
    const int entries = columns * top_k;
    dispatch_small(i < entries ? routed[i] : 0, entries, experts, dispatch, route_log, count, cursor, warp_totals);
    __syncthreads();
    if (i >= kThreads) { return; }
    cpu_plan_body<true>(dispatch, source, x, columns, top_k, max_jobs, cpu_flags, call);
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
    CUDA_CHECK(pdl::launch_consumer({dim3((columns + warps_per_block - 1) / warps_per_block), dim3(kThreads), 0, stream},
                                    route_kernel, static_cast<const float*>(logits.data), experts, columns,
                                    static_cast<int>(top_k), static_cast<std::int32_t*>(routing.ids.data),
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

void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch,
                  std::int32_t* route_log, cudaStream_t stream) {
    require(experts > 0 && experts <= kDispatchThreads, "dispatch supports up to 1024 experts");
    const int entries = static_cast<int>(routing.ids.numel());
    const auto* ids   = static_cast<const std::int32_t*>(routing.ids.data);
    if (entries <= kDispatchThreads) {
        // Decode and verification widths (k * T <= 1024): one kernel and no memset node.
        CUDA_CHECK(pdl::launch_consumer({dim3(1), dim3(kDispatchThreads), 0, stream}, dispatch_small_kernel, ids,
                                        entries, experts, dispatch, route_log));
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

void moe_route_dispatch(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, MoeDispatch& dispatch,
                        std::int32_t* route_log, cudaStream_t stream) {
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    if (columns > kRouteDispatchColumns || static_cast<std::int64_t>(columns) * top_k > kDispatchThreads) {
        moe_route(logits, top_k, routing, stream);
        moe_dispatch(routing, experts, dispatch, route_log, stream);
        return;
    }
    require(contiguous(logits, DType::FP32) && contiguous(routing.ids, DType::I32) &&
                contiguous(routing.weights, DType::FP32) && contiguous(routing.shared_gate, DType::FP32),
            "route requires contiguous FP32 logits and I32/FP32 outputs");
    require(experts > 0 && experts <= 512 && top_k > 0 && top_k <= 16 && top_k <= experts && columns > 0,
            "route supports up to 512 experts and top-16");
    require(routing.ids.ne[0] == top_k && routing.ids.ne[1] == columns && routing.weights.ne[0] == top_k &&
                routing.weights.ne[1] == columns && routing.shared_gate.numel() == columns,
            "route output shapes disagree");
    CUDA_CHECK(pdl::launch_consumer({dim3(1), dim3(kDispatchThreads), 0, stream}, route_dispatch_kernel,
                                    static_cast<const float*>(logits.data), experts, columns,
                                    static_cast<int>(top_k), static_cast<std::int32_t*>(routing.ids.data),
                                    static_cast<float*>(routing.weights.data),
                                    static_cast<float*>(routing.shared_gate.data), dispatch, route_log));
}

std::size_t moe_experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns,
                                        moe::ExpertActivation activation, moe::WideA16 wide_a16) {
    return moe::wide::experts_workspace_bytes(max_jobs, entries, columns, activation, wide_a16);
}

namespace {

constexpr std::size_t kFetchCallOffset = (sizeof(CpuCall) + 63) / 64 * 64;
static_assert(kFetchCallOffset + sizeof(FetchCall) <= moe::wide::kCpuCallBytes,
              "the workspace reserves kCpuCallBytes for a call's CPU and fetch bookkeeping");

// The CPU bookkeeping of a call in its workspace.
CpuCall* cpu_call_of(void* workspace, const Tensor& x, const MoeExpertSource& source, std::int32_t max_jobs,
                     std::int32_t entries, std::int32_t** flags) {
    const auto layout = moe::wide::carve_experts_workspace(workspace, max_jobs, entries, x.ne[1], source.activation, source.wide_a16);
    if (flags != nullptr) { *flags = layout.cpu_flags; }
    return static_cast<CpuCall*>(layout.cpu_call);
}

// The fetch bookkeeping of a call in its workspace, after the CPU's.
FetchCall* fetch_call_of(void* workspace, const Tensor& x, const MoeExpertSource& source, std::int32_t max_jobs,
                         std::int32_t entries) {
    const auto layout = moe::wide::carve_experts_workspace(workspace, max_jobs, entries, x.ne[1], source.activation, source.wide_a16);
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
}

bool fetch_served(const MoeExpertSource& source) { return source.fetch.request != nullptr; }

} // namespace

bool moe_route_dispatch_plan(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, MoeDispatch& dispatch,
                             std::int32_t* route_log, const Tensor& x, const MoeExpertSource& source,
                             std::int32_t max_jobs, void* workspace, const Tensor& outputs, cudaStream_t stream) {
    const int experts = logits.ne[0] - 1, columns = logits.ne[1];
    if (!cpu_served(x, source) || columns > kRouteDispatchColumns ||
        static_cast<std::int64_t>(columns) * top_k > kDispatchThreads) {
        moe_route_dispatch(logits, top_k, routing, dispatch, route_log, stream);
        return false;
    }
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
    std::int32_t* cpu_flags = nullptr;
    CpuCall* cpu_call       = cpu_call_of(workspace, x, source, max_jobs, outputs.ne[1], &cpu_flags);
    CUDA_CHECK(pdl::launch_consumer({dim3(1), dim3(kDispatchThreads), 0, stream}, route_dispatch_plan_kernel,
                                    static_cast<const float*>(logits.data), experts, columns, static_cast<int>(top_k),
                                    static_cast<std::int32_t*>(routing.ids.data),
                                    static_cast<float*>(routing.weights.data),
                                    static_cast<float*>(routing.shared_gate.data), dispatch, route_log, source,
                                    static_cast<const bf16*>(x.data), static_cast<int>(max_jobs), cpu_flags,
                                    cpu_call));
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
                                                             moe::kHidden, static_cast<bf16*>(outputs.data));
        check_launch("tail zero");
    }
    bool warm        = false;
    for (int i = 0; i < MoeL2Warm::kSpans; ++i) {
        require(source.l2_warm.bytes[i] == 0 || reinterpret_cast<std::uintptr_t>(source.l2_warm.ptr[i]) % 16 == 0,
                "experts L2 warm spans must be 16-byte aligned");
        warm = warm || source.l2_warm.bytes[i] >= 16;
    }
    const int ctas = kWaitCtas + (warm ? kWarmCtas : 0);
    cpu_wait_kernel<<<ctas, kThreads, 0, stream>>>(dispatch, source, call, moe::kHidden,
                                                   static_cast<bf16*>(outputs.data));
    check_launch("cpu wait");
}

void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream, bool wait_for_cpu, bool cpu_planned) {
    require(contiguous(x, DType::BF16) && contiguous(outputs, DType::BF16), "experts need BF16 x and outputs");
    require(x.ne[0] == moe::kHidden && outputs.ne[0] == moe::kHidden && outputs.ne[1] == x.ne[1] * top_k,
            "experts geometry differs from the nvfp4_expert_rg16_v1 record");
    const bool ggml = ggml_bank(source);
    const std::uint64_t record_bytes = ggml ? source.ggml.record_bytes : moe::kRecordBytes;
    require(max_jobs > 0 && max_jobs <= 65535 && (ggml || source.scales != nullptr) && source.frames != nullptr &&
                (source.host_records != nullptr || source.host_table != nullptr) &&
                source.record_stride >= record_bytes && frame_pitch(source) >= record_bytes &&
                source.record_stride % 16 == 0 && frame_pitch(source) % 16 == 0 && source.staging_slots >= 0 &&
                (source.staging_slots == 0 || source.staging_base != nullptr),
            "experts source is incomplete");
    require(!ggml || (source.activation == moe::ExpertActivation::kA16 && source.ggml.record_bytes % 16 == 0 &&
                      source.ggml.gate_row_bytes > 0 && source.ggml.down_row_bytes > 0),
            "a ggml bank takes the A16 workspace");
    require(source.landing_slots >= 0 && source.landing_slots <= offloaded_moe::kMaxLandingSlots &&
                (source.landing_slots == 0 || (source.landing != nullptr && source.landed != nullptr)),
            "experts landing is incomplete");
    require(source.prefetched == nullptr || source.prefetch_base != nullptr, "experts prefetch table without its base");
    const auto layout  = moe::wide::carve_experts_workspace(workspace, max_jobs, outputs.ne[1], x.ne[1], source.activation, source.wide_a16);
    auto** job_records = layout.job_records;
    std::int32_t* cpu_flags = nullptr;
    CpuCall* cpu_call       = cpu_call_of(workspace, x, source, max_jobs, outputs.ne[1], &cpu_flags);
    const bool cpu          = cpu_served(x, source);
    if (cpu) {
        require_cpu_channel(x, source);
        if (!cpu_planned) { // else moe_route_dispatch_plan planned the call
            CUDA_CHECK(pdl::launch_consumer({dim3(1), dim3(kThreads), 0, stream}, cpu_plan_kernel, dispatch, source,
                                            static_cast<const bf16*>(x.data), static_cast<int>(x.ne[1]), top_k,
                                            static_cast<int>(max_jobs), cpu_flags, cpu_call));
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

void moe_combine_inject(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared,
                        const Tensor& inject, Tensor& residual, cudaStream_t stream) {
    require(contiguous(outputs, DType::BF16) && contiguous(shared, DType::BF16) &&
                contiguous(residual, DType::BF16) && contiguous(inject, DType::FP32),
            "combine inject requires contiguous BF16 tensors and FP32 inject");
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

} // namespace infernix::ops
