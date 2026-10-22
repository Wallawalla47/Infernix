// The A16 narrow ring (docs/maintainer/qwen3_5-122b-offload-design.md §8.6): the narrow experts of a call
// of at most moe::wide::kRingColumns columns at Qwen3.5-122B-A10B's geometry (moe::wide::ring_call).
// Included by moe_layer.cu after the one-pass A16 kernels, whose arithmetic it shares.
//
// The one-pass kernels give each CTA a whole 16-intermediate slice: 55,296 bytes at this geometry, so
// one CTA per SM, and the slice is staged before its first product. Here one persistent pass serves
// every job of a launch. A producer warp pulls tasks from the pass's counter, first every job's
// gate/up row groups (16 interleaved gate/up rows of 192 units, job-major), then every job's down row
// groups (16 output rows of 64 units), and streams each task's 144-byte units through a kRingChunks-deep
// ring of kRingChunkUnits-unit chunks with cp.async, one chunk after another across tasks, so the next
// tasks' units are in flight while the consumer warps compute. x is encoded once per call column
// (ring_prepare_kernel, which also clears the jobs' words); the consumers read the encoded x from L2.
// A job's last gate/up task encodes the job's h, and its down tasks wait for that job only, not for the
// whole gate/up pass. Every product is the one-pass kernels' dp4a and every row sum the same exact
// int64, so the outputs are those kernels' bit for bit. (moe_layer.cu includes ops/common/mbarrier.cuh and
// ops/common/memory.cuh at file scope.)

constexpr int kRingChunkUnits   = 2 * kWarps; // two units per consumer warp per chunk
constexpr int kRingChunkBytes    = kRingChunkUnits * static_cast<int>(moe::kUnitBytes); // 2,304
constexpr int kRingChunks        = 8;          // ring depth: up to seven chunks in flight
constexpr int kRingQueue         = 4;          // tasks a CTA holds (see the producer)
constexpr int kRingThreads       = 32 + kThreads; // one producer warp, kWarps consumer warps
constexpr int kRingMinBlocksPerSm = 3;
static_assert(moe::wide::kA16ActBytes == sizeof(A16Act));
static_assert(moe::Qwen3_5_122BExperts::kGateUpBlocks % kRingChunkUnits == 0 &&
              moe::Qwen3_5_122BExperts::kDownBlocks % kRingChunkUnits == 0);

// The consumer warps of a ring CTA (warps 1..kWarps): named barrier 1.
struct RingConsumers {
    static constexpr int kThreads = ::infernix::ops::kThreads;
    __device__ static int thread() { return static_cast<int>(threadIdx.x) - 32; }
    __device__ static void sync() { asm volatile("bar.sync 1, %0;" ::"r"(kThreads) : "memory"); }
};

struct RingArgs {
    MoeDispatch dispatch;
    MoeExpertSource source;
    const std::uint8_t* const* job_records;
    const std::int32_t* flags; // CPU-served jobs (null: none)
    int base;                  // the launch's jobs: [base, base + jobs) of the call's
    int jobs;
    int top_k;
    PassPhase phase;
    const A16Act* x_acts;       // [T][kGateUpBlocks]
    const std::int32_t* x_emax; // [T]
    std::uint16_t* h_rows;      // [entries][kHRowElements]: BF16 h
    A16Act* h_acts;             // [entries][kHBlocks]
    std::int32_t* h_emax;       // [entries]
    std::uint32_t* job_done;    // [max_jobs]: gate/up tasks finished
    std::uint32_t* job_ready;   // [max_jobs]: 1 once the job's h is encoded
    std::uint32_t* counter;     // [2]: the pass's next task, CTAs finished
    std::uint16_t* outputs;     // BF16 [H, entries]
};

// A task: one row group of one job's gate/up or down matrix.
struct RingTask {
    const std::uint8_t* units; // the row group's first unit
    moe::ExpertScales scales;
    int job, rg, first, count, chunks, down;
    int column[moe::kMaxColumns]; // gate/up: each entry's x column
    int entry[moe::kMaxColumns];  // down: each entry's output column
};

template <int Columns>
struct RingShared {
    alignas(16) std::uint8_t ring[kRingChunks][kRingChunkBytes];
    std::uint64_t full[kRingChunks];  // the producer lanes' copies and lane 0's arrive
    std::uint64_t empty[kRingChunks]; // each consumer warp's release
    RingTask queue[kRingQueue];
    int slot_task[kRingChunks]; // the queue slot of the chunk in a ring slot; -1 ends the pass
    int slot_chunk[kRingChunks];
    std::int64_t partial[kWarps][16][Columns];
    std::uint16_t y[Columns][16];
    unsigned key[kWarps][Columns];
    int last;
};

__device__ __forceinline__ void ring_copies_arrive(std::uint64_t* barrier) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_addr(barrier)) : "memory");
}

__device__ __forceinline__ std::uint32_t ring_load_acquire(const std::uint32_t* p) {
    std::uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void ring_store_release(std::uint32_t* p, std::uint32_t v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// The producer's lane 0: task i of the launch (gate/up tasks first, job-major), or false when its job
// takes no task in this launch (a CPU-served job, another phase's, or a wide expert). `jobs` is the
// launch's live job count.
template <class G>
__device__ bool resolve_ring_task(const RingArgs& a, std::uint32_t i, int jobs, RingTask& t) {
    constexpr int kGateUp = G::kGateUpRowGroups, kDown = G::kDownRowGroups;
    const bool down  = i >= static_cast<std::uint32_t>(jobs * kGateUp);
    const int local  = static_cast<int>(down ? i - static_cast<std::uint32_t>(jobs * kGateUp) : i);
    const int per    = down ? kDown : kGateUp;
    const int job    = a.base + local / per;
    if (a.flags != nullptr && a.flags[job] != 0) { return false; }
    const int expert = a.dispatch.jobs[job];
    const std::uint8_t* record = pass_record(a.phase, a.source, expert, a.job_records, job);
    if (record == nullptr) { return false; }
    t.scales = a.source.scales[expert];
    t.first  = a.dispatch.offsets[expert];
    t.count  = a.dispatch.offsets[expert + 1] - t.first;
    if (moe::wide_route(t.count, t.scales)) { return false; }
    t.job    = job;
    t.rg     = local % per;
    t.down   = down ? 1 : 0;
    t.chunks = (down ? G::kDownBlocks : G::kGateUpBlocks) / kRingChunkUnits;
    t.units  = record + (down ? G::kGateUpBytes + static_cast<std::size_t>(t.rg) * G::kDownBlocks * moe::kUnitBytes
                              : static_cast<std::size_t>(t.rg) * G::kGateUpBlocks * moe::kUnitBytes);
    for (int c = 0; c < t.count; ++c) {
        const int entry = a.dispatch.entries[t.first + c];
        t.column[c]     = moe::wide::entry_column(entry, a.top_k, a.dispatch.shared_first);
        t.entry[c]      = entry;
    }
    return true;
}

// The end of the launch's index range of task i's job (its gate/up or down tasks): an invalid job's
// remaining tasks are skipped at once.
template <class G>
__device__ std::uint32_t ring_job_end(const RingArgs& a, std::uint32_t i, int jobs) {
    constexpr int kGateUp = G::kGateUpRowGroups, kDown = G::kDownRowGroups;
    const auto gate_up = static_cast<std::uint32_t>(jobs * kGateUp);
    if (i < gate_up) { return (i / kGateUp + 1) * kGateUp; }
    return gate_up + ((i - gate_up) / kDown + 1) * kDown;
}

template <class G, int Columns>
__global__ void __launch_bounds__(kRingThreads, kRingMinBlocksPerSm) narrow_ring_kernel_a16(RingArgs a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sm       = *reinterpret_cast<RingShared<Columns>*>(smem_raw);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    if (threadIdx.x == 0) {
        for (int s = 0; s < kRingChunks; ++s) {
            cta_mbarrier_init(&sm.full[s], 33);       // 32 copy arrivals and lane 0's
            cta_mbarrier_init(&sm.empty[s], kWarps);  // one per consumer warp
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (warp == 0) {
        // ------------------------------------------------------------------ producer
        // A queue slot is reused kRingQueue allocations later. The ring holds at most kRingChunks chunks
        // and a task has at least four (a down row group), so the tasks with chunks in it, and the one
        // the consumers may still be finishing, are at most four.
        int jobs = 0;
        std::uint32_t total = 0, next = 0;
        if (lane == 0) {
            jobs  = max(0, min(a.jobs, *a.dispatch.job_count - a.base));
            total = static_cast<std::uint32_t>(jobs * (G::kGateUpRowGroups + G::kDownRowGroups));
            next  = atomicAdd(&a.counter[0], 1U);
        }
        int task = -1, chunk = 0, chunks = 0, alloc = 0;
        for (int k = 0;; ++k) {
            const int s = k % kRingChunks;
            cta_mbarrier_wait(&sm.empty[s], 1U ^ ((static_cast<unsigned>(k) / kRingChunks) & 1U));
            int slot = -1, at = 0;
            if (lane == 0) {
                if (task >= 0 && chunk + 1 < chunks) {
                    slot = task;
                    at   = ++chunk;
                } else {
                    task = -1;
                    while (next < total) {
                        RingTask& t = sm.queue[alloc % kRingQueue];
                        const std::uint32_t i = next;
                        if (resolve_ring_task<G>(a, i, jobs, t)) {
                            task   = alloc++ % kRingQueue;
                            chunk  = 0;
                            chunks = t.chunks;
                            next   = atomicAdd(&a.counter[0], 1U); // used at the next allocation
                            break;
                        }
                        atomicMax(&a.counter[0], ring_job_end<G>(a, i, jobs));
                        next = atomicAdd(&a.counter[0], 1U);
                    }
                    slot = task;
                    at   = 0;
                }
                sm.slot_task[s]  = slot;
                sm.slot_chunk[s] = at;
            }
            __syncwarp();
            slot = __shfl_sync(0xFFFFFFFFU, slot, 0);
            at   = __shfl_sync(0xFFFFFFFFU, at, 0);
            if (slot >= 0) {
                const std::uint8_t* src = sm.queue[slot].units + static_cast<std::size_t>(at) * kRingChunkBytes;
                for (int i = lane; i < kRingChunkBytes / 16; i += 32) {
                    cp_async<16, Cache::cg>(sm.ring[s] + 16 * i, src + 16 * i);
                }
            }
            ring_copies_arrive(&sm.full[s]);
            if (lane == 0) { cta_mbarrier_arrive(&sm.full[s]); } // releases slot_task, slot_chunk and the queue
            if (slot < 0) { break; }
        }
        if (lane == 0) {
            // Dependents may launch once every CTA has issued its last copy; the last CTA out resets the
            // pass's counter for the next launch on this slot.
            pdl::trigger_dependents();
            if (atomicAdd(&a.counter[1], 1U) == gridDim.x - 1) {
                atomicExch(&a.counter[0], 0U);
                atomicExch(&a.counter[1], 0U);
            }
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }

    // -------------------------------------------------------------------- consumers
    const int cw = warp - 1, ctid = RingConsumers::thread();
    const int r = lane & 15, half = lane >> 4;
    std::int64_t s[Columns];
#pragma unroll
    for (int c = 0; c < Columns; ++c) { s[c] = 0; }
    for (int k = 0;; ++k) {
        const int st = k % kRingChunks;
        cta_mbarrier_wait(&sm.full[st], (static_cast<unsigned>(k) / kRingChunks) & 1U);
        const int slot = sm.slot_task[st];
        if (slot < 0) { break; }
        const RingTask& t = sm.queue[slot];
        const int chunk   = sm.slot_chunk[st];
        const int n       = t.count;
        if (chunk == 0) {
#pragma unroll
            for (int c = 0; c < Columns; ++c) { s[c] = 0; }
            if (t.down != 0) {
                // The job's h is encoded once all its gate/up tasks are done (they were taken before any
                // down task, so no CTA waits on work queued behind its own).
                if (ctid == 0) {
                    while (ring_load_acquire(&a.job_ready[t.job]) == 0U) { __nanosleep(128); }
                }
                RingConsumers::sync();
            }
        }
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int j              = 2 * cw + v;
            const int b              = chunk * kRingChunkUnits + j;
            const std::uint8_t* unit = sm.ring[st] + j * moe::kUnitBytes;
            const int w0 = row_quad(unit, r, 2 * half);
            const int w1 = row_quad(unit, r, 2 * half + 1);
            const int sw = canon::e4m3_scaled(unit[128 + r]);
#pragma unroll
            for (int c = 0; c < Columns; ++c) {
                if (c < n) { // uniform across the CTA
                    uint2 lo, mid, hi;
                    if (t.down != 0) {
                        // Written by another CTA of this kernel: through L2.
                        const auto* act = reinterpret_cast<const std::uint8_t*>(
                            a.h_acts + static_cast<std::size_t>(t.first + c) * G::kHBlocks + b);
                        lo  = __ldcg(reinterpret_cast<const uint2*>(act + 8 * half));
                        mid = __ldcg(reinterpret_cast<const uint2*>(act + 16 + 8 * half));
                        hi  = __ldcg(reinterpret_cast<const uint2*>(act + 32 + 8 * half));
                    } else {
                        const auto* act = reinterpret_cast<const std::uint8_t*>(
                            a.x_acts + static_cast<std::size_t>(t.column[c]) * G::kGateUpBlocks + b);
                        lo  = __ldg(reinterpret_cast<const uint2*>(act + 8 * half));
                        mid = __ldg(reinterpret_cast<const uint2*>(act + 16 + 8 * half));
                        hi  = __ldg(reinterpret_cast<const uint2*>(act + 32 + 8 * half));
                    }
                    const int plo  = dp4a_su(w1, lo.y, dp4a_su(w0, lo.x, 0));
                    const int pmid = dp4a_su(w1, mid.y, dp4a_su(w0, mid.x, 0));
                    const int phi  = __dp4a(w1, static_cast<int>(hi.y), __dp4a(w0, static_cast<int>(hi.x), 0));
                    unsigned p = static_cast<unsigned>(plo) + (static_cast<unsigned>(pmid) << 8) +
                                 (static_cast<unsigned>(phi) << 16);
                    p += static_cast<unsigned>(__shfl_xor_sync(0xFFFFFFFFU, static_cast<int>(p), 16));
                    s[c] += static_cast<std::int64_t>(static_cast<int>(p)) * sw;
                }
            }
        }
        __syncwarp();
        if (lane == 0) { cta_mbarrier_arrive(&sm.empty[st]); }
        if (chunk + 1 != t.chunks) { continue; }

        // ------------------------------------------------------------ the task's rows
        RingConsumers::sync(); // every consumer is past the previous task's reads of partial and y
        if (lane < 16) {
#pragma unroll
            for (int c = 0; c < Columns; ++c) { sm.partial[cw][r][c] = s[c]; }
        }
        RingConsumers::sync();
        if (t.down == 0) {
            for (int i = ctid; i < 16 * n; i += RingConsumers::kThreads) {
                const int row = i % 16, c = i / 16;
                std::int64_t total = 0;
                for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row][c]; }
                sm.y[c][row] = canon::a16_row_output(total, a.x_emax[t.column[c]],
                                                     (row & 1) == 0 ? t.scales.alpha_gate : t.scales.alpha_up);
            }
            RingConsumers::sync();
            for (int i = ctid; i < 8 * n; i += RingConsumers::kThreads) {
                const int kk = i % 8, c = i / 8;
                a.h_rows[static_cast<std::size_t>(t.first + c) * moe::wide::A16Planes<G>::kHRowElements + 8 * t.rg + kk] =
                    canon::swiglu_bf16(sm.y[c][2 * kk], sm.y[c][2 * kk + 1]);
            }
            __threadfence();
            RingConsumers::sync();
            if (ctid == 0) { sm.last = atomicAdd(&a.job_done[t.job], 1U) == G::kGateUpRowGroups - 1 ? 1 : 0; }
            RingConsumers::sync();
            if (sm.last != 0) {
                // The job's last gate/up task: every entry's h is written; encode it for the down tasks.
                __threadfence();
                const int first = t.first;
                const auto base = [&](int c) {
                    return a.h_rows + static_cast<std::size_t>(first + c) * moe::wide::A16Planes<G>::kHRowElements;
                };
                encode_a16_columns<Columns, true, RingConsumers>(base, n, G::kIntermediate,
                                                                 a.h_acts + static_cast<std::size_t>(first) * G::kHBlocks,
                                                                 a.h_emax + first, sm.key);
                __threadfence();
                RingConsumers::sync();
                if (ctid == 0) { ring_store_release(&a.job_ready[t.job], 1U); }
            }
        } else {
            for (int i = ctid; i < 16 * n; i += RingConsumers::kThreads) {
                const int row = i % 16, c = i / 16;
                std::int64_t total = 0;
                for (int w = 0; w < kWarps; ++w) { total += sm.partial[w][row][c]; }
                const int emax = __ldcg(a.h_emax + t.first + c);
                a.outputs[static_cast<std::size_t>(t.entry[c]) * G::kHidden + 16 * t.rg + row] =
                    canon::a16_row_output(total, emax, t.scales.alpha_down);
            }
        }
    }
}

// Every column's x in the dp4a layout with its exponent (CTA c < columns), and the jobs' words
// (`job_words`: ring_done then ring_ready) and the counters cleared (the last CTA). Before any pass of
// the call.
template <class G>
__global__ void __launch_bounds__(kThreads)
    ring_prepare_kernel(const bf16* __restrict__ x, int columns, A16Act* __restrict__ x_acts,
                        std::int32_t* __restrict__ x_emax, std::uint32_t* __restrict__ job_words, int max_jobs,
                        std::uint32_t* __restrict__ counters) {
    __shared__ unsigned key[kWarps][1];
    const int c = static_cast<int>(blockIdx.x);
    if (c < columns) {
        const auto base = [&](int) {
            return reinterpret_cast<const std::uint16_t*>(x) + static_cast<std::size_t>(c) * G::kHidden;
        };
        encode_a16_columns<1>(base, 1, G::kHidden, x_acts + static_cast<std::size_t>(c) * G::kGateUpBlocks,
                              x_emax + c, key);
        return;
    }
    for (int i = static_cast<int>(threadIdx.x); i < 2 * max_jobs; i += kThreads) { job_words[i] = 0U; }
    if (static_cast<int>(threadIdx.x) < 2 * moe::wide::kRingCounterSlots) { counters[threadIdx.x] = 0U; }
}

void launch_ring_prepare(const Tensor& x, const moe::wide::ExpertsWorkspace& layout, int max_jobs,
                         cudaStream_t stream) {
    const int columns = x.ne[1];
    ring_prepare_kernel<moe::Qwen3_5_122BExperts><<<columns + 1, kThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data), columns, reinterpret_cast<A16Act*>(layout.x_acts), layout.x_emax,
        layout.ring_done, max_jobs, layout.ring_counters);
    check_launch("A16 ring prepare");
}

template <class G, int Columns>
void launch_ring_pass_a16(const RingArgs& args, cudaStream_t stream) {
    constexpr int kBytes = static_cast<int>(sizeof(RingShared<Columns>));
    static const int ctas = [] {
        CUDA_CHECK(cudaFuncSetAttribute(narrow_ring_kernel_a16<G, Columns>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kBytes));
        CUDA_CHECK(cudaFuncSetAttribute(narrow_ring_kernel_a16<G, Columns>,
                                        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        int device = 0, sms = 0, per_sm = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, narrow_ring_kernel_a16<G, Columns>,
                                                                 kRingThreads, kBytes));
        return std::max(1, per_sm) * sms;
    }();
    const int tasks = args.jobs * (G::kGateUpRowGroups + G::kDownRowGroups);
    narrow_ring_kernel_a16<G, Columns><<<std::min(ctas, tasks), kRingThreads, kBytes, stream>>>(args);
    check_launch("A16 ring");
}

// The narrow experts of jobs [base, base + jobs) of a ring call (moe::wide::ring_call) with `columns`
// columns: one persistent launch on the phase's counter.
void launch_ring_pass(int columns, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                      const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                      const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream, PassPhase phase) {
    using G = moe::Qwen3_5_122BExperts;
    RingArgs args{};
    args.dispatch    = dispatch;
    args.source      = source;
    args.job_records = job_records;
    args.flags       = flags;
    args.base        = base;
    args.jobs        = jobs;
    args.top_k       = top_k;
    args.phase       = phase;
    args.x_acts      = reinterpret_cast<const A16Act*>(layout.x_acts);
    args.x_emax      = layout.x_emax;
    args.h_rows      = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    args.h_acts      = reinterpret_cast<A16Act*>(layout.h_acts);
    args.h_emax      = layout.h_emax;
    args.job_done    = layout.ring_done;
    args.job_ready   = layout.ring_ready;
    args.counter     = layout.ring_counters + 2 * static_cast<int>(phase);
    args.outputs     = reinterpret_cast<std::uint16_t*>(outputs);
    // An expert has at most min(columns, kMaxColumns) entries.
    const int most = std::min(columns, moe::kMaxColumns);
    if (most <= 1) {
        launch_ring_pass_a16<G, 1>(args, stream);
    } else if (most <= 2) {
        launch_ring_pass_a16<G, 2>(args, stream);
    } else if (most <= 4) {
        launch_ring_pass_a16<G, 4>(args, stream);
    } else {
        launch_ring_pass_a16<G, 8>(args, stream);
    }
}
