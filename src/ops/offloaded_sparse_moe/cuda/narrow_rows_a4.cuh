// The register-resident row-pair narrow route of NVFP4 W4A4 experts at Qwen3.8-Flash-Next's geometry (candidates
// TM-02c, 2026-11-09; narrow_rows_a16.cuh's kernels on A4 activations), included by moe_layer.cu after the A4
// tensor-core narrow kernels: the jobs of decode and verification calls of at most moe::wide::kA4RowsMaxColumns columns
// of a bank whose experts share their gate and up input scales (a4_rows_call; wider calls keep the tensor-core
// kernels). Every row sum S = sum_b (P_b e4m3_scaled(w_b)) e4m3_scaled(a_b), with P_b the block's exact sum of doubled
// E2M1 code products, is the same int64 as the tensor-core and one-column kernels' and the CPU engine's, and h is
// quantized from the same BF16 values by the same rule, so the outputs are theirs bit for bit wherever an expert runs.
//
// - The call's jobs' h counts are cleared once per call (prepare_x, before its passes); each pass's jobs' columns'
//   A4(x) are encoded once per (job, column) by encode_nmma4_kernel (the tensor-core route's encoder and layout).
// - Gate/up: one CTA per (row group, job), eight warps; warp w takes the row group's blocks 20 w .. 20 w + 19 and lane
//   8 s + i rows i and i + 8 of its units s, s + 4, ..., whose code words and scales are loaded into registers at the
//   CTA's start and kept for every column of the job (the job's weights read once, 2,880 contiguous bytes in flight a
//   warp). The encoded x row of a column comes to shared memory as doubled-code quads and scaled block scales; a row's
//   P_b is four dp4a products. Each h element is stored as BF16 and counted (red.release) for the job's down CTAs.
// - Down: one CTA per four output row groups of a job, warps 2 r and 2 r + 1 taking row group 4 tile + r's units
//   0..19 and 20..39, loaded before the wait. A CTA waits for its own job's h count, then quantizes each column's h
//   (the down input scale, quantize_a4_block) into shared memory.

// Calls of at most this many columns take these kernels (moe::wide::kA4RowsMaxColumns; the workspace sizes their h
// rows and counts for them).
constexpr int kA4RowsMaxColumns = moe::wide::kA4RowsMaxColumns;

// Whether a call's W4A4 jobs take these kernels (prepare_x and launch_pass decide it identically).
inline bool a4_rows_call(const MoeExpertSource& source, int columns, const moe::wide::ExpertsWorkspace& layout) {
    return columns <= kA4RowsMaxColumns && a4_nmma_call(source, columns, layout) && layout.x_plane != nullptr &&
           layout.h_rows != nullptr && layout.job_ctas != nullptr;
}

// One column's A4 activation block in shared memory: elements 4 q .. 4 q + 3's doubled codes (s8 x 4) and the block's
// scaled scale.
struct A4RowsAct {
    int quad[4];
    int scale;
};

// P_b times the weight scale of rows i (lo) and i + 8 (hi) of one unit, times the column block's scale: the
// tensor-core kernels' terms (nmma4_unit), |P_b| <= 2,304 and Sw < 2^18 so P_b * Sw fits int32.
__device__ __forceinline__ void rows4_unit_terms(const UnitRows& w, const A4RowsAct& a, const std::int32_t* scaled,
                                                 std::int64_t& lo_row, std::int64_t& hi_row) {
    int p0 = 0, p1 = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int c0 = static_cast<int>(canon::e2m1_x2_quad(w.codes[q] & 0x0F0F0F0FU));        // row i
        const int c1 = static_cast<int>(canon::e2m1_x2_quad((w.codes[q] >> 4) & 0x0F0F0F0FU)); // row i + 8
        p0           = __dp4a(c0, a.quad[q], p0);
        p1           = __dp4a(c1, a.quad[q], p1);
    }
    lo_row += static_cast<std::int64_t>(p0 * scaled[w.scales & 0xFFU]) * a.scale;
    hi_row += static_cast<std::int64_t>(p1 * scaled[(w.scales >> 8) & 0xFFU]) * a.scale;
}

// The warp's exact sums of rows i (lo) and i + 8 (hi) over its units (blocks base .. base + 19) for one column's
// blocks `acts`; lanes 0..7 hold them on return.
__device__ __forceinline__ void rows4_sums(const UnitRows (&w)[kRowsWarpSteps], int base, const A4RowsAct* acts,
                                           const std::int32_t* scaled, std::int64_t& lo, std::int64_t& hi) {
    const int s = (static_cast<int>(threadIdx.x) % 32) >> 3;
    lo = hi = 0;
#pragma unroll
    for (int step = 0; step < kRowsWarpSteps; ++step) { rows4_unit_terms(w[step], acts[base + 4 * step + s], scaled, lo, hi); }
#pragma unroll
    for (int offset = 8; offset <= 16; offset <<= 1) {
        lo += __shfl_xor_sync(0xFFFFFFFFU, lo, offset);
        hi += __shfl_xor_sync(0xFFFFFFFFU, hi, offset);
    }
}

// Clears the call's jobs' h counts (prepare_x: once per call, before its passes).
__global__ void __launch_bounds__(kThreads) a4_rows_clear_kernel(std::uint32_t* __restrict__ job_counts, int max_jobs) {
    for (int j = static_cast<int>(threadIdx.x); j < max_jobs; j += kThreads) { job_counts[j] = 0U; }
}

template <class G>
struct Rows4GateUpShared {
    A4RowsAct acts[G::kGateUpBlocks];
    std::int64_t partial[kWarps][16];
    std::int32_t scaled[256];
    std::uint16_t y[16];
};

struct Rows4Args {
    MoeDispatch dispatch;
    MoeExpertSource source;
    int top_k;
    const std::uint8_t* const* job_records;
    const std::int32_t* cpu_flags;
    int job_base;
    const std::uint8_t* x_rows; // [entries][Nmma4Shape::kXRowBytes]: each narrow entry's A4(x) (encode_nmma4_kernel)
    std::uint16_t* h_rows;      // [entries][kHRowElements]: BF16 h
    std::uint32_t* job_counts;  // [max_jobs]: h elements stored
    bf16* outputs;              // [entries][hidden]
    PassPhase phase;
    int job_wait;               // 1: down CTAs wait for their job's h count; 0: the PDL wait
};

// One CTA per (gate/up row group, job): gate rows even, up rows odd, so it gives h elements 8 g .. 8 g + 7 of each of
// the job's columns. A programmatic dependent of encode_nmma4_kernel: it loads its weights before the PDL wait and
// reads the encoded x rows after it (every CTA waits, an idle one too, so this grid's completion implies the
// encode's). Quantizing x in each gate/up CTA instead (80 CTAs a job, the TC route's per-CTA encode below four
// columns) measured slower: T = 1 34.8, 2 53.3, 3 69.7 us against 30.8, 45.1-47.1 and 61.5 with the encode kernel
// (2026-11-09).
template <class G>
__global__ void __launch_bounds__(kThreads, kRowsMinCtas) gate_up_rows4_kernel(Rows4Args a) {
    using S = Nmma4Shape<G>;
    __shared__ Rows4GateUpShared<G> sm;
    // The down pass may launch and load its weights now: it reads h only after its job's count below is complete. The
    // trigger stays first on every path (narrow_rows_a16.cuh's contract).
    pdl::trigger_dependents();
    const MoeDispatch& dispatch = a.dispatch;
    const int job = a.job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (a.cpu_flags != nullptr && a.cpu_flags[job] != 0)) { return pdl::wait_for_dependencies(); }
    const int expert           = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(a.phase, a.source, expert, a.job_records, job);
    if (record == nullptr) { return pdl::wait_for_dependencies(); }
    const moe::ExpertScales scales = a.source.scales[expert];
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (moe::wide_route(count, scales)) { return pdl::wait_for_dependencies(); }
    const int g    = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    UnitRows w[kRowsWarpSteps];
    rows_load_warp(record + static_cast<std::size_t>(g) * G::kGateUpBlocks * moe::kUnitBytes, warp * kRowsWarpUnits, w);
    rows_fill_scaled(sm.scaled);
    pdl::wait_for_dependencies(); // the encoded x rows
    __syncthreads();              // the scale table
    for (int c = 0; c < count; ++c) {
        if (c > 0) { __syncthreads(); } // every warp has read the previous column's acts and y
        // Column c's A4(x) row as doubled-code quads and scaled scales: block b a thread.
        const std::uint8_t* row = a.x_rows + static_cast<std::size_t>(first + c) * S::kXRowBytes;
        for (int b = static_cast<int>(threadIdx.x); b < G::kGateUpBlocks; b += kThreads) {
            const uint2 codes = __ldcg(reinterpret_cast<const uint2*>(row + 8 * b));
            A4RowsAct act;
            act.quad[0] = nmma4_x_quad(codes.x & 0xFFFFU);
            act.quad[1] = nmma4_x_quad(codes.x >> 16);
            act.quad[2] = nmma4_x_quad(codes.y & 0xFFFFU);
            act.quad[3] = nmma4_x_quad(codes.y >> 16);
            act.scale   = sm.scaled[__ldcg(row + S::kXCodeBytes + b)];
            sm.acts[b]  = act;
        }
        __syncthreads();
        std::int64_t lo, hi;
        rows4_sums(w, warp * kRowsWarpUnits, sm.acts, sm.scaled, lo, hi);
        if (lane < 8) {
            sm.partial[warp][lane]     = lo;
            sm.partial[warp][lane + 8] = hi;
        }
        __syncthreads();
        if (threadIdx.x < 16) {
            const int r        = static_cast<int>(threadIdx.x);
            std::int64_t total = 0;
#pragma unroll
            for (int v = 0; v < kWarps; ++v) { total += sm.partial[v][r]; }
            sm.y[r] = canon::a4_row_output(total, (r & 1) == 0 ? scales.alpha_gate : scales.alpha_up);
        }
        __syncthreads();
        if (threadIdx.x < 8) {
            const int k = static_cast<int>(threadIdx.x);
            a.h_rows[static_cast<std::size_t>(first + c) * moe::wide::A16Planes<G>::kHRowElements + 8 * g + k] =
                canon::swiglu_bf16(sm.y[2 * k], sm.y[2 * k + 1]);
            // Counted for the job's down CTAs (narrow_rows_a16.cuh's protocol: one release per stored element).
            asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(a.job_counts + job) : "memory");
        }
    }
}

template <class G>
struct Rows4DownShared {
    A4RowsAct acts[G::kHBlocks];
    std::int64_t partial[kWarps][16];
    std::int32_t scaled[256];
    int ready;
};

// One CTA per (kRowsDownGroups down row groups, job): warps 2 r and 2 r + 1 take row group kRowsDownGroups tile + r,
// blocks 0..19 and 20..39. A programmatic dependent of the gate/up pass: the dispatch, the job records, the frames and
// the weights are read before the wait, h only after it.
template <class G>
__global__ void __launch_bounds__(kThreads, kRowsMinCtas) down_rows4_kernel(Rows4Args a) {
    static_assert(G::kHBlocks <= kThreads, "a thread quantizes one h block");
    __shared__ Rows4DownShared<G> sm;
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
        if (threadIdx.x < G::kHBlocks) {
            // h block t with the down input scale (the tensor-core and one-column kernels' quantize_a4_block), read
            // through L2 (the gate/up grid wrote it).
            const int t = static_cast<int>(threadIdx.x);
            const auto* src = reinterpret_cast<const uint4*>(
                a.h_rows + static_cast<std::size_t>(first + c) * moe::wide::A16Planes<G>::kHRowElements + 16 * t);
            const uint4 v0 = __ldcg(src), v1 = __ldcg(src + 1);
            const std::uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            std::uint16_t h[16];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                h[2 * k]     = static_cast<std::uint16_t>(words[k] & 0xFFFFU);
                h[2 * k + 1] = static_cast<std::uint16_t>(words[k] >> 16);
            }
            const canon::A4Block block = canon::quantize_a4_block(h, scales.input_down);
            A4RowsAct act;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                act.quad[q] = static_cast<int>(static_cast<std::uint32_t>(static_cast<std::uint8_t>(block.c2[4 * q])) |
                                               (static_cast<std::uint32_t>(static_cast<std::uint8_t>(block.c2[4 * q + 1])) << 8) |
                                               (static_cast<std::uint32_t>(static_cast<std::uint8_t>(block.c2[4 * q + 2])) << 16) |
                                               (static_cast<std::uint32_t>(static_cast<std::uint8_t>(block.c2[4 * q + 3])) << 24));
            }
            act.scale  = block.scale_scaled;
            sm.acts[t] = act;
        }
        __syncthreads();
        std::int64_t lo, hi;
        rows4_sums(w, base, sm.acts, sm.scaled, lo, hi);
        if (lane < 8) {
            sm.partial[warp][lane]     = lo;
            sm.partial[warp][lane + 8] = hi;
        }
        __syncthreads();
        if (threadIdx.x < 16 * kRowsDownGroups) {
            const int row = static_cast<int>(threadIdx.x), r = row / 16, k = row % 16;
            const std::int64_t total = sm.partial[2 * r][k] + sm.partial[2 * r + 1][k];
            out[static_cast<std::size_t>(dispatch.entries[first + c]) * G::kHidden + 16 * kRowsDownGroups * tile + row] =
                canon::a4_row_output(total, scales.alpha_down);
        }
    }
}

// Clears the call's jobs' h counts (once per call, before its passes).
void launch_prepare_a4_rows(int max_jobs, const moe::wide::ExpertsWorkspace& layout, cudaStream_t stream) {
    a4_rows_clear_kernel<<<1, kThreads, 0, stream>>>(layout.job_ctas, max_jobs);
    check_launch("A4 rows prepare");
}

// One pass of the row-pair kernels: the jobs' x encode, a gate/up CTA per row group and a down CTA per kRowsDownGroups
// output row groups of each job.
void launch_expert_pass_a4_rows(const bf16* x, int columns, const MoeDispatch& dispatch, const MoeExpertSource& source,
                                int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags, int base,
                                int jobs, const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                                PassPhase phase, PassPart part) {
    using G = moe::FlashNextExperts;
    static_assert(Nmma4Shape<G>::kXRowBytes == moe::wide::kXRowBytes, "the encoded rows are the workspace's x plane rows");
    const int call_columns = std::min(columns, kA4RowsMaxColumns);
    const Nmma4Args encode{reinterpret_cast<const std::uint16_t*>(x), dispatch, source, top_k, job_records, flags, base,
                           layout.h_blocks, outputs, phase, layout.x_plane};
    Rows4Args args{dispatch, source, top_k, job_records, flags, base, layout.x_plane,
                   reinterpret_cast<std::uint16_t*>(layout.h_rows), layout.job_ctas, outputs, phase};
    // A split call's down launch is serialized behind its gate/up pass (an event wait), so it needs no count wait.
    args.job_wait = part != PassPart::Down ? 1 : 0;
    if (part != PassPart::Down) {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized, {dim3(jobs, call_columns), dim3(G::kHidden / 16), 0, stream},
                                    encode_nmma4_kernel<G>, encode));
        check_launch("A4 rows x encode");
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Programmatic, {dim3(G::kGateUpRowGroups, jobs), dim3(kThreads), 0, stream},
                                    gate_up_rows4_kernel<G>, args));
        check_launch("A4 rows gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    CUDA_CHECK(pdl::launch_with(part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic,
                                {dim3(RowsShape<G>::kDownCtas, jobs), dim3(kThreads), 0, stream}, down_rows4_kernel<G>, args));
    check_launch("A4 rows down");
}
