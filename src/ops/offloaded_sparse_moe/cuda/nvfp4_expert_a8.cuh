// NVFP4 W4A16 expert banks on the 8-bit activation route (`--a8-activations`, NarrowA16::kA8; LOSSY, opt-in;
// ops/offloaded_sparse_moe/nvfp4_a8.h, which defines the arithmetic), included by moe_layer.cu after
// ggml_expert_a8.cuh, whose slice encoder (a8_quantize_slice: ops/ggml/ggml_a8.h's codec) and pass width it shares.
// x and h are quantized per 32-element slice to int8 codes with an FP32 scale; a 16-element block's product with
// the stored doubled E2M1 codes is an exact int32 (four dp4a, one per four weights, against three per weight on
// the exact A16 route), and each block's term (nvfp4_a8::add_block_term) adds into its row's FP32 partials in
// nvfp4_a8.h's order, which the CPU route follows, so a CPU-served miss returns these kernels' bits.
//
// As on the GGUF route, a call of at most kA8MaxCallColumns columns (decode, verification, up to eight lanes) runs
// every job here, whatever its column count, in passes of kA8Columns columns over the CTA's staged weights: no
// wide tile and no narrow ring take part, so an expert's bits depend on neither its column count nor its
// placement. Wider calls (prefill) keep the W4A16 routes, or with WideA16::kA8 (nvfp4_a8_wide_call) run their jobs
// of at most moe::kMaxColumns columns here and their wider jobs on the wide route's 8-bit grouped GEMM
// (wide_expert_a8.cuh: the same codec, int8 tensor cores, another rounding and order of the FP32 sums). Both
// registered geometries G (Flash-Next's H 2560 / I
// 640, Qwen3.5-122B-A10B's 3072 / 1024), and the 122B's shared expert as the call's last job.
// - nvfp4_a8_encode_x_kernel quantizes every token's x once per call into the workspace's A8 x rows
//   (wide::a8_x_rows, wide::a8_x_row_bytes apart).
// - A gate/up CTA stages one row group of the gate/up matrix (16 rows: eight intermediates' gate and up rows) with
//   cp.async, as the A16 narrow kernels stage theirs, and reads it back without bank conflicts: lane i + 8 m of
//   warp w owns rows i and i + 8 (one code word holds both) and partial p = 4 w + m, slices p, p + 16, ... The
//   butterfly's first two steps (o = 8, 4) cross warps through shared memory, its last two are shuffles; lanes
//   0-7 of warp 0 then round the rows, pair gate and up for SwiGLU and write BF16 h to the entry's h row.
// - A down CTA stages four row groups, one per warp (lane i + 8 m owns rows i and i + 8 and partial m, slices m,
//   m + 4, ...; the butterfly is two shuffles); per pass its threads quantize the pass's columns' h into shared
//   memory, one slice each.
// Every unit's codes are decoded once per pass and serve its eight columns; the partials stay in registers.

constexpr int kNvfp4A8Threads   = 128; // either kernel's CTA
constexpr int kNvfp4A8Warps     = kNvfp4A8Threads / 32;
constexpr int kNvfp4A8MinBlocks = 3; // registers for the three CTAs per SM their shared memory allows (122B down: two)
static_assert(kA8MaxCallColumns == moe::nvfp4_a8::kMaxCallColumns, "the route's widest call is nvfp4_a8.h's");
static_assert(moe::nvfp4_a8::kGateUpPartials == 4 * kNvfp4A8Warps && moe::nvfp4_a8::kDownPartials == 4,
              "a gate/up row's partials are four per warp, a down row's four lane groups of one warp");

// The route's constants at geometry G.
template <class G>
struct Nvfp4A8Shape {
    static constexpr int kXSlices = G::kHidden / 32;       // 80 or 96
    static constexpr int kHSlices = G::kIntermediate / 32; // 20 or 32
    static constexpr int kGateUpSteps = kXSlices / moe::nvfp4_a8::kGateUpPartials; // slices per gate/up partial
    static constexpr int kDownSteps   = kHSlices / moe::nvfp4_a8::kDownPartials;   // slices per down partial
    static constexpr int kGateUpStageBytes = G::kGateUpBlocks * static_cast<int>(moe::kUnitBytes); // one row group
    static constexpr int kDownStageBytes   = kNvfp4A8Warps * G::kDownBlocks * static_cast<int>(moe::kUnitBytes);
    static constexpr int kXRowBytes        = moe::wide::a8_x_row_bytes(moe::kShapeOf<G>);
    static_assert(kXSlices % moe::nvfp4_a8::kGateUpPartials == 0 && kHSlices % moe::nvfp4_a8::kDownPartials == 0,
                  "every partial takes the same number of slices");
    static_assert(G::kGateUpRowGroups * 8 == G::kIntermediate && G::kDownRowGroups % kNvfp4A8Warps == 0,
                  "a gate/up CTA owns eight intermediates, a down CTA four whole row groups");
    static_assert(kXRowBytes == static_cast<int>(a8::column_words(G::kHidden, false)) * 4 && kXRowBytes % 16 == 0,
                  "an A8 x row holds a column's codes then its slice scales");
    static_assert(kGateUpStageBytes % 16 == 0 && kDownStageBytes % 16 == 0, "the stage copies 16-byte pieces");
};

// Whether an NVFP4 W4A16 bank's call takes these kernels for every job (moe_experts, the CPU plan and launch_pass
// decide it identically).
__host__ __device__ __forceinline__ bool nvfp4_a8_call(const MoeExpertSource& source, int columns) {
    return source.narrow_a16 == moe::NarrowA16::kA8 && source.activation == moe::ExpertActivation::kA16 &&
           !ggml_bank(source) && columns <= kA8MaxCallColumns;
}

// Whether a wider call stays on 8-bit activations (WideA16::kA8): its jobs of at most moe::kMaxColumns columns on
// these kernels, its wider jobs on the wide route's 8-bit grouped GEMM (wide_expert_a8.cuh).
__host__ __device__ __forceinline__ bool nvfp4_a8_wide_call(const MoeExpertSource& source, int columns) {
    return source.narrow_a16 == moe::NarrowA16::kA8 && source.wide_a16 == moe::WideA16::kA8 &&
           source.activation == moe::ExpertActivation::kA16 && !ggml_bank(source) && columns > kA8MaxCallColumns;
}

// Whether a call takes the 8-bit activation route on either bank kind (MissRequest::a8): its CPU-served misses
// (at most kMaxCpuColumns columns each, so narrow jobs) then take the CPU route's 8-bit functions.
__host__ __device__ __forceinline__ bool a8_route_call(const MoeExpertSource& source, int columns) {
    return a8_call(source, columns) || nvfp4_a8_call(source, columns) || nvfp4_a8_wide_call(source, columns);
}

// Rows i and i + 8 of one 144-byte unit: their doubled E2M1 codes as signed bytes (word q: k = 4q .. 4q + 3, the
// low and high nibbles of the unit's code words 32q + 4i) and their E4M3FN scale words.
struct Nvfp4A8Rows {
    int lo[4], hi[4];
    unsigned scale_lo, scale_hi;
};

__device__ __forceinline__ Nvfp4A8Rows nvfp4_a8_rows(const std::uint8_t* __restrict__ unit, int i) {
    Nvfp4A8Rows u;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * q + 4 * i);
        u.lo[q] = static_cast<int>(canon::e2m1_x2_quad(word & 0x0F0F0F0FU));
        u.hi[q] = static_cast<int>(canon::e2m1_x2_quad((word >> 4) & 0x0F0F0F0FU));
    }
    u.scale_lo = unit[128 + i];
    u.scale_hi = unit[136 + i];
    return u;
}

// P_b of one row against a block's codes (word q: elements 4q .. 4q + 3), exact: |P_b| <= 12 * 16 * 127.
__device__ __forceinline__ int nvfp4_a8_block_dot(const int (&w)[4], const int4& x) {
    return __dp4a(w[3], x.w, __dp4a(w[2], x.z, __dp4a(w[1], x.y, __dp4a(w[0], x.x, 0))));
}

// One slice's terms for rows lo and hi into their partials: block 2s (b0, codes x0), then block 2s + 1 (b1, x1),
// both under the slice's scale d.
__device__ __forceinline__ void nvfp4_a8_slice(const Nvfp4A8Rows& b0, const Nvfp4A8Rows& b1, const int4& x0,
                                               const int4& x1, float d, float& lo, float& hi) {
    namespace spec = moe::nvfp4_a8;
    lo = spec::add_block_term(spec::add_block_term(lo, nvfp4_a8_block_dot(b0.lo, x0), d, b0.scale_lo),
                              nvfp4_a8_block_dot(b1.lo, x1), d, b1.scale_lo);
    hi = spec::add_block_term(spec::add_block_term(hi, nvfp4_a8_block_dot(b0.hi, x0), d, b0.scale_hi),
                              nvfp4_a8_block_dot(b1.hi, x1), d, b1.scale_hi);
}

// One CTA per column; thread s quantizes slice s.
constexpr int kNvfp4A8EncodeThreads = 96;

template <class G>
__global__ void __launch_bounds__(kNvfp4A8EncodeThreads)
    nvfp4_a8_encode_x_kernel(const bf16* __restrict__ x, std::uint8_t* __restrict__ x_rows) {
    using S = Nvfp4A8Shape<G>;
    static_assert(S::kXSlices <= kNvfp4A8EncodeThreads, "a thread per slice");
    const int t = static_cast<int>(blockIdx.x), s = static_cast<int>(threadIdx.x);
    if (s >= S::kXSlices) { return; }
    const auto* v      = reinterpret_cast<const std::uint16_t*>(x) + static_cast<std::size_t>(t) * G::kHidden + 32 * s;
    const A8SliceOut o = a8_quantize_slice(v);
    auto* row          = reinterpret_cast<std::int32_t*>(x_rows + static_cast<std::size_t>(t) * S::kXRowBytes);
    auto* codes        = reinterpret_cast<int4*>(row + s * a8::kSliceWords);
    codes[0]           = make_int4(o.q[0], o.q[1], o.q[2], o.q[3]);
    codes[1]           = make_int4(o.q[4], o.q[5], o.q[6], o.q[7]);
    reinterpret_cast<float*>(row + a8::scales_offset(G::kHidden))[s] = o.d;
}

template <class G>
__global__ void __launch_bounds__(kNvfp4A8Threads, kNvfp4A8MinBlocks)
    gate_up_kernel_nvfp4_a8(const std::uint8_t* __restrict__ x_rows, MoeDispatch dispatch, MoeExpertSource source,
                            int top_k, const std::uint8_t* const* __restrict__ job_records,
                            const std::int32_t* __restrict__ cpu_flags, int job_base,
                            std::uint16_t* __restrict__ h_plane, PassPhase phase, int narrow_max) {
    using S         = Nvfp4A8Shape<G>;
    constexpr int C = kA8Columns;
    __shared__ __align__(16) std::uint8_t stage[S::kGateUpStageBytes];
    // The butterfly's steps across warps: warps 2 and 3's partials (slot w - 2), then warp 1's pair sums (slot 1,
    // over the values it read); value h * C + c (row half h, column c), lane-minor.
    __shared__ float exchange[2][2 * C][32];
    const int job = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert           = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (count > narrow_max) { return; } // a wide job (nvfp4_a8_wide_call): the wide route computes it
    const int group = static_cast<int>(blockIdx.x); // rows 16 group .. 16 group + 15: intermediates 8 group .. + 7
    stage_async(record + static_cast<std::size_t>(group) * S::kGateUpStageBytes, stage, S::kGateUpStageBytes);
    const moe::ExpertScales scales = source.scales[expert];
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int i = lane & 7, part = 4 * warp + (lane >> 3);
    stage_wait();
    // The down pass (a programmatic dependent in captured graphs) may launch once every CTA has its slice.
    pdl::trigger_dependents();
    __syncthreads();
    for (int pass = 0; pass < count; pass += C) {
        const int n = min(C, count - pass);
        const std::int32_t* xs[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const int column =
                c < n ? moe::wide::entry_column(dispatch.entries[first + pass + c], top_k, dispatch.shared_first) : 0;
            xs[c] = reinterpret_cast<const std::int32_t*>(x_rows + static_cast<std::size_t>(column) * S::kXRowBytes);
        }
        float lo[C], hi[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { lo[c] = hi[c] = 0.0F; }
        for (int k = 0; k < S::kGateUpSteps; ++k) { // the partial's slices in ascending order
            const int s              = part + moe::nvfp4_a8::kGateUpPartials * k;
            const std::uint8_t* unit = stage + static_cast<std::size_t>(2 * s) * moe::kUnitBytes;
            const Nvfp4A8Rows b0 = nvfp4_a8_rows(unit, i), b1 = nvfp4_a8_rows(unit + moe::kUnitBytes, i);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) { // uniform across the CTA
                    const int4* q = reinterpret_cast<const int4*>(xs[c] + s * a8::kSliceWords);
                    const float d = __ldg(reinterpret_cast<const float*>(xs[c] + a8::scales_offset(G::kHidden)) + s);
                    nvfp4_a8_slice(b0, b1, __ldg(q), __ldg(q + 1), d, lo[c], hi[c]);
                }
            }
        }
        // The butterfly: o = 8 (warps 0, 1 take warps 2, 3's partials), o = 4 (warp 0 takes warp 1's), then o = 2
        // and 1 within warp 0 (lane groups m and m + 2, then m and m + 1).
        __syncthreads(); // warp 0 has read the previous pass's slot 1
        if (warp >= 2) {
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) {
                    exchange[warp - 2][c][lane]     = lo[c];
                    exchange[warp - 2][C + c][lane] = hi[c];
                }
            }
        }
        __syncthreads();
        if (warp < 2) {
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) {
                    lo[c] = canon::add_rn(lo[c], exchange[warp][c][lane]);
                    hi[c] = canon::add_rn(hi[c], exchange[warp][C + c][lane]);
                    if (warp == 1) { // the same words this lane read
                        exchange[1][c][lane]     = lo[c];
                        exchange[1][C + c][lane] = hi[c];
                    }
                }
            }
        }
        __syncthreads();
        if (warp == 0) {
            const float m = (i & 1) == 0 ? scales.alpha_gate : scales.alpha_up; // rows i and i + 8: even rows gate
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) {
                    float a = canon::add_rn(lo[c], exchange[1][c][lane]);
                    float b = canon::add_rn(hi[c], exchange[1][C + c][lane]);
                    a       = canon::add_rn(a, __shfl_xor_sync(0xFFFFFFFFU, a, 16));
                    b       = canon::add_rn(b, __shfl_xor_sync(0xFFFFFFFFU, b, 16));
                    a       = canon::add_rn(a, __shfl_xor_sync(0xFFFFFFFFU, a, 8));
                    b       = canon::add_rn(b, __shfl_xor_sync(0xFFFFFFFFU, b, 8));
                    const std::uint16_t ya = moe::nvfp4_a8::row_output(a, m);
                    const std::uint16_t yb = moe::nvfp4_a8::row_output(b, m);
                    // Gate row 2j and up row 2j + 1 are lanes 2j and 2j + 1.
                    const auto ua = static_cast<std::uint16_t>(__shfl_xor_sync(0xFFFFFFFFU, static_cast<unsigned>(ya), 1));
                    const auto ub = static_cast<std::uint16_t>(__shfl_xor_sync(0xFFFFFFFFU, static_cast<unsigned>(yb), 1));
                    if (lane < 8 && (lane & 1) == 0) {
                        std::uint16_t* h = h_plane +
                                           static_cast<std::size_t>(first + pass + c) * moe::wide::A16Planes<G>::kHRowElements +
                                           8 * group + i / 2;
                        h[0] = canon::swiglu_bf16(ya, ua); // rows i, i + 1: intermediate 8 group + i / 2
                        h[4] = canon::swiglu_bf16(yb, ub); // rows i + 8, i + 9: four intermediates on
                    }
                }
            }
        }
    }
}

template <class G>
__global__ void __launch_bounds__(kNvfp4A8Threads, kNvfp4A8MinBlocks)
    down_kernel_nvfp4_a8(MoeDispatch dispatch, MoeExpertSource source, const std::uint8_t* const* __restrict__ job_records,
                         const std::int32_t* __restrict__ cpu_flags, int job_base,
                         const std::uint16_t* __restrict__ h_plane, bf16* __restrict__ outputs, PassPhase phase,
                         int narrow_max) {
    using S         = Nvfp4A8Shape<G>;
    constexpr int C = kA8Columns;
    __shared__ __align__(16) std::uint8_t stage[S::kDownStageBytes];
    __shared__ __align__(16) std::int32_t codes[C][S::kHSlices * a8::kSliceWords];
    __shared__ float slice_scales[C][S::kHSlices];
    const int job  = job_base + static_cast<int>(blockIdx.y);
    const int tile = static_cast<int>(blockIdx.x); // row groups kNvfp4A8Warps tile .. + kNvfp4A8Warps - 1
    // A programmatic dependent of the gate/up pass, under down_kernel's contract: only h_plane is read after the
    // PDL wait, and every CTA waits.
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    if (job < *dispatch.job_count && (cpu_flags == nullptr || cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(phase, source, expert, job_records, job);
    }
    float m = 0.0F;
    if (record != nullptr) {
        first = dispatch.offsets[expert];
        count = dispatch.offsets[expert + 1] - first;
        m     = source.scales[expert].alpha_down;
        if (count <= narrow_max) { // a wide job's CTAs copy nothing and exit after the wait
            stage_async(record + G::kGateUpBytes + static_cast<std::size_t>(tile) * S::kDownStageBytes, stage,
                        S::kDownStageBytes);
        }
    }
    pdl::wait_for_dependencies();
    if (record == nullptr || count > narrow_max) { return; }
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int i = lane & 7, part = lane >> 3;
    const int group          = kNvfp4A8Warps * tile + warp; // output rows 16 group .. 16 group + 15
    const std::uint8_t* rows = stage + static_cast<std::size_t>(warp) * G::kDownBlocks * moe::kUnitBytes;
    for (int pass = 0; pass < count; pass += C) {
        const int n = min(C, count - pass);
        for (int item = static_cast<int>(threadIdx.x); item < n * S::kHSlices; item += kNvfp4A8Threads) {
            const int c = item / S::kHSlices, s = item % S::kHSlices;
            const std::uint16_t* h = h_plane +
                                     static_cast<std::size_t>(first + pass + c) * moe::wide::A16Planes<G>::kHRowElements +
                                     32 * s;
            const A8SliceOut o = a8_quantize_slice(h);
            auto* dst          = reinterpret_cast<int4*>(&codes[c][s * a8::kSliceWords]);
            dst[0]             = make_int4(o.q[0], o.q[1], o.q[2], o.q[3]);
            dst[1]             = make_int4(o.q[4], o.q[5], o.q[6], o.q[7]);
            slice_scales[c][s] = o.d;
        }
        stage_wait();
        __syncthreads();
        float lo[C], hi[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { lo[c] = hi[c] = 0.0F; }
        for (int k = 0; k < S::kDownSteps; ++k) { // the partial's slices in ascending order
            const int s              = part + moe::nvfp4_a8::kDownPartials * k;
            const std::uint8_t* unit = rows + static_cast<std::size_t>(2 * s) * moe::kUnitBytes;
            const Nvfp4A8Rows b0 = nvfp4_a8_rows(unit, i), b1 = nvfp4_a8_rows(unit + moe::kUnitBytes, i);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) {
                    const int4* q = reinterpret_cast<const int4*>(&codes[c][s * a8::kSliceWords]);
                    nvfp4_a8_slice(b0, b1, q[0], q[1], slice_scales[c][s], lo[c], hi[c]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < n) {
                // The butterfly: o = 2 (lane groups m and m + 2), then o = 1 (m and m + 1).
                float a = canon::add_rn(lo[c], __shfl_xor_sync(0xFFFFFFFFU, lo[c], 16));
                float b = canon::add_rn(hi[c], __shfl_xor_sync(0xFFFFFFFFU, hi[c], 16));
                a       = canon::add_rn(a, __shfl_xor_sync(0xFFFFFFFFU, a, 8));
                b       = canon::add_rn(b, __shfl_xor_sync(0xFFFFFFFFU, b, 8));
                if (part == 0) {
                    auto* y = reinterpret_cast<std::uint16_t*>(outputs) +
                              static_cast<std::size_t>(dispatch.entries[first + pass + c]) * G::kHidden + 16 * group + i;
                    y[0] = moe::nvfp4_a8::row_output(a, m);
                    y[8] = moe::nvfp4_a8::row_output(b, m);
                }
            }
        }
        __syncthreads(); // every thread has read the codes before the next pass rewrites them
    }
}

// Encodes the call's x for the 8-bit gate/up kernels (once per moe_experts call, before its passes).
void launch_encode_x_nvfp4_a8(const Tensor& x, const MoeExpertSource& source, const moe::wide::ExpertsWorkspace& layout,
                              cudaStream_t stream) {
    std::uint8_t* x_rows = moe::wide::a8_x_rows(layout);
    require(x_rows != nullptr, "the 8-bit activation route needs the x region of a W4A16 wide route's workspace");
    moe::with_expert_geometry(source.shape, [&]<class G>() {
        nvfp4_a8_encode_x_kernel<G><<<static_cast<unsigned>(x.ne[1]), kNvfp4A8EncodeThreads, 0, stream>>>(
            static_cast<const bf16*>(x.data), x_rows);
    });
    check_launch("nvfp4 a8 encode x");
}

// One pass of the 8-bit kernels at geometry G: a gate/up CTA per row group (eight intermediates) and a down CTA
// per four row groups (64 output rows) of each job of at most narrow_max columns; `part` launches only one matrix
// (a split forked call).
template <class G>
void launch_expert_pass_nvfp4_a8(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                                 const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                                 const std::uint8_t* x_rows, std::uint16_t* h_plane, bf16* outputs, cudaStream_t stream,
                                 PassPhase phase, PassPart part, int narrow_max) {
    // The largest carveout, as the A16 narrow kernels take: three CTAs of either kernel per SM (two of the 122B's
    // down CTAs), and the passes' CTAs may share SMs with the other NVFP4 narrow kernels.
    static const bool configured = [] {
        cudaFuncSetAttribute(gate_up_kernel_nvfp4_a8<G>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(down_kernel_nvfp4_a8<G>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        return true;
    }();
    (void)configured;
    if (part != PassPart::Down) {
        gate_up_kernel_nvfp4_a8<G><<<dim3(G::kGateUpRowGroups, jobs), kNvfp4A8Threads, 0, stream>>>(
            x_rows, dispatch, source, top_k, job_records, flags, base, h_plane, phase, narrow_max);
        check_launch("nvfp4 a8 gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_with(part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic,
                                {dim3(G::kDownRowGroups / kNvfp4A8Warps, jobs), dim3(kNvfp4A8Threads), 0, stream},
                                down_kernel_nvfp4_a8<G>, dispatch, source, job_records, flags, base,
                                static_cast<const std::uint16_t*>(h_plane), outputs, phase, narrow_max));
}

// launch_pass's NVFP4 W4A16 jobs on the 8-bit route: every job of an nvfp4_a8_call, the jobs of at most
// moe::kMaxColumns columns of an nvfp4_a8_wide_call (`wide`; the wide route computes the others).
void launch_pass_nvfp4_a8(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                          const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                          const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream, PassPhase phase,
                          PassPart part, bool wide = false) {
    const std::uint8_t* x_rows = moe::wide::a8_x_rows(layout);
    auto* h_plane              = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    const int narrow_max       = wide ? moe::kMaxColumns : std::numeric_limits<int>::max();
    moe::with_expert_geometry(source.shape, [&]<class G>() {
        launch_expert_pass_nvfp4_a8<G>(dispatch, source, top_k, job_records, flags, base, jobs, x_rows, h_plane, outputs,
                                       stream, phase, part, narrow_max);
    });
}
