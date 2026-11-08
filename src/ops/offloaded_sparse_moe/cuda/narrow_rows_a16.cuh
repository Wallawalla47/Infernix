// The register-resident row-pair narrow route of NVFP4 W4A16 experts at Qwen3.8-Flash-Next's geometry (design
// §19.3.52; candidates §30.3#1 / TM-02, after ninferpro f4ffe290a7, 0bfc3f747d, 1b4d12c352 and f48f1c051a, an
// Apache-2.0 derivative of these kernels), included by moe_layer.cu after the tensor-core narrow kernels: the jobs of
// decode and verification calls of at most kA16RowsMaxColumns columns (a16_rows_call; wider ones keep the tensor-core
// kernels). Every row sum S = sum_b e4m3_scaled(s_b) P_b is the same int64 as the tensor-core and one-pass kernels'
// and the CPU engine's, so the outputs are theirs bit for bit wherever an expert runs.
//
// - The call's x is encoded once (a16_rows_prepare_kernel) into the workspace's x rows as A16Act blocks (X = lo +
//   256 mid + 65536 hi, the one-pass kernels' dp4a layout) with its column exponents, and the jobs' h counts are
//   cleared.
// - Gate/up: one CTA per (row group, job), eight warps. Warp w takes the row group's blocks 20 w .. 20 w + 19; lane
//   l = 8 s + i holds rows i and i + 8 of its units s, s + 4, ... Their code words and scales are loaded into
//   registers at the CTA's start, all at once, and stay there for every column of the job, so a job of n columns
//   reads its weights once and each warp keeps 2,880 contiguous bytes in flight. Three CTAs an SM (80 registers). A
//   row's block product P_b (doubled E2M1 codes times the 16 X values, |P_b| < 2^29) is three dp4a chains wrapping in
//   32 bits to the exact P_b, then P_b times the row's scale (a shared-memory table of e4m3_scaled) is the exact
//   int64 term. A column's x blocks come to shared memory, and each h element is counted (red.release) for the job's
//   down CTAs.
// - Down: one CTA per four output row groups of a job, warps 2 r and 2 r + 1 taking row group 4 tile + r's units
//   0..19 and 20..39, loaded before the wait. A CTA waits for its own job's h count (every element counted) instead of
//   the gate/up grid's PDL wait; it then encodes each column's h under its own exponent in shared memory.

constexpr int kRowsWarpUnits  = 20; // units a warp holds: a gate/up row group's 160 blocks or half of a down group's 40
constexpr int kRowsWarpSteps  = kRowsWarpUnits / 4;
constexpr int kRowsDownGroups = 4;  // output row groups of a down CTA
constexpr int kRowsMinCtas    = 3;  // CTAs an SM (80 registers; four spill)
// Calls of at most this many columns take these kernels. Op bench, the call as the engine replays it (a CUDA graph,
// cold L2, every record in frames), against the tensor-core kernels, same hashes: T = 1 28.1 -> 21.6 us, 2 44.5 ->
// 36.9, 3 52.9 -> 50.8; level from 4 (64.9 / 65.0, 5 74.8 / 75.1, 6 90.3 / 91.6, 8 108.3 / 107.9) where both stream
// at the bytes' floor (1.59 TB/s here). Design §19.3.52.
constexpr int kA16RowsMaxColumns = 3;

template <class G>
struct RowsShape {
    static_assert(G::kGateUpBlocks == kWarps * kRowsWarpUnits && G::kDownBlocks == 2 * kRowsWarpUnits &&
                      kWarps == 2 * kRowsDownGroups,
                  "a gate/up row group is eight warps' units, a down row group two warps'");
    static_assert(G::kGateUpBlocks * static_cast<int>(sizeof(A16Act)) <= moe::wide::A16Planes<G>::kXRowBytes,
                  "an x row of A16Act blocks fits the workspace's x row");
    static_assert(G::kDownRowGroups % kRowsDownGroups == 0 && G::kIntermediate % 16 == 0 &&
                      G::kIntermediate / 8 <= kThreads,
                  "whole down CTAs; a thread holds 8 h elements");
    static constexpr int kXRowBytes = moe::wide::A16Planes<G>::kXRowBytes;
    static constexpr int kDownCtas  = G::kDownRowGroups / kRowsDownGroups;
};

// Whether a call's W4A16 jobs take these kernels (moe_experts' prepare and launch_pass decide it identically).
inline bool a16_rows_call(const MoeExpertSource& source, int columns, const moe::wide::ExpertsWorkspace& layout) {
    return columns <= kA16RowsMaxColumns && a16_nmma_call(source, columns, layout);
}

// A lane's code words (q, i) of one unit, q = 0..3, and its two rows' scale words (byte 0: row i, byte 1: row i + 8).
struct UnitRows {
    std::uint32_t codes[4];
    std::uint32_t scales;
};

__device__ __forceinline__ UnitRows rows_load_unit(const std::uint8_t* unit, int i) {
    UnitRows w;
#pragma unroll
    for (int q = 0; q < 4; ++q) { w.codes[q] = __ldg(reinterpret_cast<const unsigned*>(unit + 32 * q + 4 * i)); }
    w.scales = static_cast<std::uint32_t>(__ldg(unit + 128 + i)) | (static_cast<std::uint32_t>(__ldg(unit + 136 + i)) << 8);
    return w;
}

// The lane's units j = 4 step + s of the warp (block base + j of `matrix`), all in flight at once.
__device__ __forceinline__ void rows_load_warp(const std::uint8_t* matrix, int base, UnitRows (&w)[kRowsWarpSteps]) {
    const int lane = static_cast<int>(threadIdx.x) % 32, s = lane >> 3, i = lane & 7;
#pragma unroll
    for (int step = 0; step < kRowsWarpSteps; ++step) {
        w[step] = rows_load_unit(matrix + static_cast<std::size_t>(base + 4 * step + s) * moe::kUnitBytes, i);
    }
}

// P_b times the scale of rows i (lo) and i + 8 (hi) of one unit against its column block `a` (`scaled`: the
// e4m3_scaled table).
__device__ __forceinline__ void rows_unit_terms(const UnitRows& w, const A16Act& a, const std::int32_t* scaled, std::int64_t& lo_row,
                                                std::int64_t& hi_row) {
    int l0 = 0, m0 = 0, h0 = 0, l1 = 0, m1 = 0, h1 = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int c0 = static_cast<int>(canon::e2m1_x2_quad(w.codes[q] & 0x0F0F0F0FU));
        const int c1 = static_cast<int>(canon::e2m1_x2_quad((w.codes[q] >> 4) & 0x0F0F0F0FU));
        l0 = dp4a_su(c0, a.lo[q], l0);
        m0 = dp4a_su(c0, a.mid[q], m0);
        h0 = __dp4a(c0, a.hi[q], h0);
        l1 = dp4a_su(c1, a.lo[q], l1);
        m1 = dp4a_su(c1, a.mid[q], m1);
        h1 = __dp4a(c1, a.hi[q], h1);
    }
    // Wrapping assembly: the true block sums fit int32.
    const unsigned p0 = static_cast<unsigned>(l0) + (static_cast<unsigned>(m0) << 8) + (static_cast<unsigned>(h0) << 16);
    const unsigned p1 = static_cast<unsigned>(l1) + (static_cast<unsigned>(m1) << 8) + (static_cast<unsigned>(h1) << 16);
    lo_row += static_cast<std::int64_t>(static_cast<int>(p0)) * scaled[w.scales & 0xFFU];
    hi_row += static_cast<std::int64_t>(static_cast<int>(p1)) * scaled[(w.scales >> 8) & 0xFFU];
}

// The warp's exact sums of rows i (lo) and i + 8 (hi) over its units (blocks base .. base + 19) for one column's
// blocks `acts`; lanes 0..7 hold them on return.
__device__ __forceinline__ void rows_sums(const UnitRows (&w)[kRowsWarpSteps], int base, const A16Act* acts, const std::int32_t* scaled,
                                          std::int64_t& lo, std::int64_t& hi) {
    const int s = (static_cast<int>(threadIdx.x) % 32) >> 3;
    lo = hi = 0;
#pragma unroll
    for (int step = 0; step < kRowsWarpSteps; ++step) { rows_unit_terms(w[step], acts[base + 4 * step + s], scaled, lo, hi); }
#pragma unroll
    for (int offset = 8; offset <= 16; offset <<= 1) {
        lo += __shfl_xor_sync(0xFFFFFFFFU, lo, offset);
        hi += __shfl_xor_sync(0xFFFFFFFFU, hi, offset);
    }
}

__device__ __forceinline__ void rows_fill_scaled(std::int32_t* scaled) {
    scaled[threadIdx.x] = canon::e4m3_scaled(static_cast<unsigned>(threadIdx.x)); // kThreads == 256 entries
}
static_assert(kThreads == 256, "one scale table entry per thread");

// Every token's x blocks and exponent (CTA t < columns), and the jobs' h counts cleared (CTA `columns`).
template <class G>
__global__ void __launch_bounds__(kThreads)
    a16_rows_prepare_kernel(const bf16* __restrict__ x, int columns, A16Act* __restrict__ acts,
                            std::int32_t* __restrict__ emax, std::uint32_t* __restrict__ job_counts, int max_jobs) {
    __shared__ unsigned key[kWarps][1];
    const int t = static_cast<int>(blockIdx.x);
    if (t == columns) {
        for (int j = static_cast<int>(threadIdx.x); j < max_jobs; j += kThreads) { job_counts[j] = 0U; }
        return;
    }
    const auto base = [&](int) { return reinterpret_cast<const std::uint16_t*>(x) + static_cast<std::size_t>(t) * G::kHidden; };
    encode_a16_columns<1>(base, 1, G::kHidden,
                          reinterpret_cast<A16Act*>(reinterpret_cast<std::uint8_t*>(acts) + static_cast<std::size_t>(t) * RowsShape<G>::kXRowBytes),
                          emax + t, key);
}

struct RowsArgs {
    MoeDispatch dispatch;
    MoeExpertSource source;
    int top_k;
    const std::uint8_t* const* job_records;
    const std::int32_t* cpu_flags;
    int job_base;
    const std::uint8_t* x_rows;  // [T][kXRowBytes]: A16Act blocks
    const std::int32_t* x_emax;  // [T]
    std::uint16_t* h_rows;       // [entries][kHRowElements]: BF16 h
    std::uint32_t* job_counts;   // [max_jobs]: h elements stored
    bf16* outputs;               // [entries][hidden]
    PassPhase phase;
    int job_wait;                // 1: down CTAs wait for their job's h count; 0: the PDL wait
};

template <class G>
struct RowsGateUpShared {
    A16Act acts[G::kGateUpBlocks];
    std::int64_t partial[kWarps][16];
    std::int32_t scaled[256];
    std::uint16_t y[16];
};

// One CTA per (gate/up row group, job): gate rows even, up rows odd, so it gives h elements 8 g .. 8 g + 7 of each of
// the job's columns.
template <class G>
__global__ void __launch_bounds__(kThreads, kRowsMinCtas) gate_up_rows_kernel(RowsArgs a) {
    __shared__ RowsGateUpShared<G> sm;
    // The down pass (a programmatic dependent in captured graphs) may launch and load its weights now: it reads h only
    // after its job's count below is complete. The trigger stays first on every path: a dependent grid launches only
    // once every CTA of this grid has triggered or exited, so every CTA a spinning down CTA waits for is resident.
    pdl::trigger_dependents();
    const MoeDispatch& dispatch = a.dispatch;
    const int job = a.job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (a.cpu_flags != nullptr && a.cpu_flags[job] != 0)) { return; }
    const int expert           = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(a.phase, a.source, expert, a.job_records, job);
    if (record == nullptr) { return; }
    const moe::ExpertScales scales = a.source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (moe::wide_route(count, scales)) { return; }
    const int g    = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    UnitRows w[kRowsWarpSteps];
    rows_load_warp(record + static_cast<std::size_t>(g) * G::kGateUpBlocks * moe::kUnitBytes, warp * kRowsWarpUnits, w);
    rows_fill_scaled(sm.scaled);
    for (int c = 0; c < count; ++c) {
        if (c > 0) { __syncthreads(); } // every warp has read the previous column's acts and y
        const int token = moe::wide::entry_column(dispatch.entries[first + c], a.top_k, dispatch.shared_first);
        stage_async(a.x_rows + static_cast<std::size_t>(token) * RowsShape<G>::kXRowBytes,
                    reinterpret_cast<std::uint8_t*>(sm.acts), static_cast<int>(sizeof(sm.acts)));
        stage_wait();
        __syncthreads();
        std::int64_t lo, hi;
        rows_sums(w, warp * kRowsWarpUnits, sm.acts, sm.scaled, lo, hi);
        if (lane < 8) {
            sm.partial[warp][lane]     = lo;
            sm.partial[warp][lane + 8] = hi;
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            const int row      = static_cast<int>(threadIdx.x);
            std::int64_t total = 0;
#pragma unroll
            for (int v = 0; v < kWarps; ++v) { total += sm.partial[v][row]; }
            sm.y[row] = canon::a16_row_output(total, a.x_emax[token], (row & 1) == 0 ? scales.alpha_gate : scales.alpha_up);
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            const int k = static_cast<int>(threadIdx.x);
            a.h_rows[static_cast<std::size_t>(first + c) * moe::wide::A16Planes<G>::kHRowElements + 8 * g + k] =
                canon::swiglu_bf16(sm.y[2 * k], sm.y[2 * k + 1]);
            // Counted for the job's down CTAs: fire-and-forget, the release orders this thread's h store before it.
            // Eight per column and row group, one per element, so every store is covered without a barrier.
            asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(a.job_counts + job) : "memory");
        }
    }
}

// A down CTA's wait for its job's h: the gate/up CTAs count every h element with a release, so acquire loads of the
// count reaching `target` see every h store. Bounded (offloaded_moe::kDownReadyTimeoutNs): a lost count fails the call
// through its error word and returns false (the CTA then reads no h), never hangs the stream.
__device__ __forceinline__ bool rows_wait_job(const MoeExpertSource& source, const std::uint32_t* count, std::uint32_t target) {
    if (load_acquire_gpu(count) >= target) { return true; }
    const std::uint64_t entered = global_ns();
    do {
        if (global_ns() - entered > offloaded_moe::kDownReadyTimeoutNs) {
            report_error(source, offloaded_moe::kErrorDownStalled);
            return false;
        }
        __nanosleep(64);
    } while (load_acquire_gpu(count) < target);
    return true;
}

// One h row of G::kIntermediate BF16 elements encoded under its own exponent into `acts`: thread t < kIntermediate / 8
// holds elements 8 t .. 8 t + 7 for both the exponent and its two quads of block t / 2. Ends with a barrier.
template <class G>
__device__ __forceinline__ void rows_encode_h(const std::uint16_t* column, A16Act* acts, int* emax, unsigned (&key)[kWarps]) {
    constexpr int kVectors = G::kIntermediate / 8;
    const int t = static_cast<int>(threadIdx.x), lane = t % 32, warp = t / 32;
    unsigned words[4] = {};
    if (t < kVectors) {
        const uint4 v = __ldcg(reinterpret_cast<const uint4*>(column) + t); // written by the gate/up grid: through L2
        words[0]      = v.x;
        words[1]      = v.y;
        words[2]      = v.z;
        words[3]      = v.w;
    }
    unsigned m = 0;
#pragma unroll
    for (int w = 0; w < 4; ++w) { m = max(m, max(words[w] & 0x7FFFU, (words[w] >> 16) & 0x7FFFU)); }
    m = __reduce_max_sync(0xFFFFFFFFU, m);
    if (lane == 0) { key[warp] = m; }
    __syncthreads();
    if (t == 0) {
        unsigned k = 0;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { k = max(k, key[w]); }
        *emax = k >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(k >> 7), 1);
    }
    __syncthreads();
    if (t < kVectors) {
        const int e = *emax;
        A16Act& out = acts[t / 2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int q = 2 * (t % 2) + half;
            std::uint32_t lo = 0, mid = 0, hi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned word = words[2 * half + j / 2];
                const auto h        = static_cast<std::uint16_t>(j % 2 == 0 ? word & 0xFFFFU : word >> 16);
                const auto xv       = static_cast<std::uint32_t>(e == canon::kA16NonFinite ? 0 : canon::a16_value(h, e));
                lo |= (xv & 0xFFU) << (8 * j);
                mid |= ((xv >> 8) & 0xFFU) << (8 * j);
                hi |= ((xv >> 16) & 0xFFU) << (8 * j);
            }
            out.lo[q]  = lo;
            out.mid[q] = mid;
            out.hi[q]  = static_cast<std::int32_t>(hi);
        }
    }
    __syncthreads();
}

template <class G>
struct RowsDownShared {
    A16Act acts[G::kHBlocks];
    std::int64_t partial[kWarps][16];
    std::int32_t scaled[256];
    int emax;
    unsigned key[kWarps];
    int ready;
};

// One CTA per (kRowsDownGroups down row groups, job): warps 2 r and 2 r + 1 take row group kRowsDownGroups tile + r,
// blocks 0..19 and 20..39. A programmatic dependent of the gate/up pass: the dispatch, the job records, the frames and
// the weights are read before the wait, h only after it.
template <class G>
__global__ void __launch_bounds__(kThreads, kRowsMinCtas) down_rows_kernel(RowsArgs a) {
    __shared__ RowsDownShared<G> sm;
    const MoeDispatch& dispatch = a.dispatch;
    const int job  = a.job_base + static_cast<int>(blockIdx.y);
    const int tile = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    if (job < *dispatch.job_count && (a.cpu_flags == nullptr || a.cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(a.phase, a.source, expert, a.job_records, job);
    }
    moe::ExpertScales scales{};
    if (record != nullptr) {
        scales = a.source.scales[expert];
        first  = dispatch.offsets[expert];
        count  = dispatch.offsets[expert + 1] - first;
        if (moe::wide_route(count, scales)) { record = nullptr; }
    }
    const int group = kRowsDownGroups * tile + warp / 2, base = (warp % 2) * kRowsWarpUnits;
    UnitRows w[kRowsWarpSteps];
    if (record != nullptr) {
        rows_load_warp(record + G::kGateUpBytes + static_cast<std::size_t>(group) * G::kDownBlocks * moe::kUnitBytes, base, w);
    }
    rows_fill_scaled(sm.scaled); // read after the barriers below
    if (a.job_wait != 0) {
        // The gate/up grid is a serialized launch, so every earlier kernel has finished, and this grid launched only
        // once every gate/up CTA had started (each triggers at entry): the job's CTAs are running.
        if (record == nullptr) { return; }
        if (threadIdx.x == 0) {
            sm.ready = rows_wait_job(a.source, a.job_counts + job, 8U * G::kGateUpRowGroups * static_cast<std::uint32_t>(count)) ? 1 : 0;
        }
        __syncthreads();
        if (sm.ready == 0) { return; } // the wait's bound passed: the call's error word fails the round
    } else {
        pdl::wait_for_dependencies();
        if (record == nullptr) { return; }
    }
    auto* out = reinterpret_cast<std::uint16_t*>(a.outputs);
    for (int c = 0; c < count; ++c) {
        if (c > 0) { __syncthreads(); } // every warp has read the previous column's acts and partials
        rows_encode_h<G>(a.h_rows + static_cast<std::size_t>(first + c) * moe::wide::A16Planes<G>::kHRowElements, sm.acts,
                         &sm.emax, sm.key);
        std::int64_t lo, hi;
        rows_sums(w, base, sm.acts, sm.scaled, lo, hi);
        if (lane < 8) {
            sm.partial[warp][lane]     = lo;
            sm.partial[warp][lane + 8] = hi;
        }
        __syncthreads();
        if (threadIdx.x < 16 * kRowsDownGroups) {
            const int row = static_cast<int>(threadIdx.x), r = row / 16, k = row % 16;
            const std::int64_t total = sm.partial[2 * r][k] + sm.partial[2 * r + 1][k];
            out[static_cast<std::size_t>(dispatch.entries[first + c]) * G::kHidden + 16 * kRowsDownGroups * tile + row] =
                canon::a16_row_output(total, sm.emax, scales.alpha_down);
        }
    }
}

// Encodes the call's x blocks and clears its jobs' h counts (once per call, before its passes).
void launch_prepare_a16_rows(const Tensor& x, int max_jobs, const moe::wide::ExpertsWorkspace& layout, cudaStream_t stream) {
    using G = moe::FlashNextExperts;
    const int columns = static_cast<int>(x.ne[1]);
    a16_rows_prepare_kernel<G><<<static_cast<unsigned>(columns + 1), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data), columns, reinterpret_cast<A16Act*>(layout.x_limbs), layout.x_emax, layout.job_ctas,
        max_jobs);
    check_launch("A16 rows prepare");
}

// One pass of the row-pair kernels: a gate/up CTA per row group and a down CTA per kRowsDownGroups output row groups
// of each job.
void launch_expert_pass_a16_rows(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                                 const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                                 const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                                 PassPhase phase, PassPart part) {
    using G = moe::FlashNextExperts;
    RowsArgs args{dispatch, source, top_k, job_records, flags, base, layout.x_limbs, layout.x_emax,
                  reinterpret_cast<std::uint16_t*>(layout.h_rows), layout.job_ctas, outputs, phase};
    // A split call's down launch is serialized behind its gate/up pass (an event wait), so it needs no wait.
    args.job_wait = part != PassPart::Down ? 1 : 0;
    if (part != PassPart::Down) {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized, {dim3(G::kGateUpRowGroups, jobs), dim3(kThreads), 0, stream},
                                    gate_up_rows_kernel<G>, args));
        check_launch("A16 rows gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    CUDA_CHECK(pdl::launch_with(part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic,
                                {dim3(RowsShape<G>::kDownCtas, jobs), dim3(kThreads), 0, stream}, down_rows_kernel<G>, args));
    check_launch("A16 rows down");
}
