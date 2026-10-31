// The tensor-core narrow route of NVFP4 W4A4 experts (design §16.2's arithmetic), included by moe_layer.cu after the
// A16 tensor-core route: the jobs of a call of at most moe::kMaxColumns columns (decode and verification at one
// lane) at Qwen3.8-Flash-Next's geometry, for banks whose experts all share their gate and up input scales
// (a4_nmma_call). Every row sum S = sum_b P_b e4m3_scaled(w_b) e4m3_scaled(a_b), with P_b the block's exact sum of
// doubled E2M1 code products, is the same int64 as the one-column kernels' (unit_sums) and the CPU engine's, so the
// outputs are theirs bit for bit wherever an expert runs.
//
// - Per 16-element block, rows g and g + 8 of a 16-row group form one m16n8k16 s8 A fragment (the record word
//   32 t + 4 g's low and high nibbles as doubled codes), and column g's doubled codes of elements 4 t .. 4 t + 3 the
//   B fragment: the MMA gives P_b exactly (|P_b| <= 16 * 144) for up to eight columns at once. Thread (g, t) adds
//   P_b times the row's and the column's scales into its four int64 sums: rows g and g + 8, columns 2 t and 2 t + 1.
// - Gate/up: a CTA owns four row groups (32 intermediates' gate and up rows, two A4 blocks of h) with two warps a
//   group, each streaming half of K through a private cp.async ring; the halves' sums add exactly. The CTA quantizes
//   its job's columns of x to A4 with the job's input scale by the canonical rule (quantize_a4_codes) into shared
//   memory while its rings fill, and quantizes its h with the down input scale (quantize_a4_block) into the h blocks
//   the down kernels read. (A first form, a once-per-call encode kernel before the passes and one warp a group over
//   its whole K, put a launch on the critical path and halved the warps in flight: 45.6 against the one-column
//   kernels' 41.6 us at one column, 2026-10-31; this form measures 41.6. Design §19.3.44.)
// - Down: a warp owns one output row group, its units copied before the PDL wait; the job's h blocks then come to
//   shared memory.

constexpr int kNmma4Warps      = 4;  // row groups of a gate/up CTA and of a down CTA
constexpr int kNmma4Threads    = 32 * kNmma4Warps;      // a down CTA
constexpr int kNmma4KSplit     = 2;  // warps a gate/up row group (each half of K)
constexpr int kNmma4GuWarps    = kNmma4Warps * kNmma4KSplit;
constexpr int kNmma4GuThreads  = 32 * kNmma4GuWarps;   // a gate/up CTA
constexpr int kNmma4StageUnits = 8;  // units per stage of a gate/up warp's ring (1,152 bytes)
constexpr int kNmma4Depth      = 3;  // stages in a gate/up warp's ring

template <class G>
struct Nmma4Shape {
    static constexpr int kStageBytes = kNmma4StageUnits * static_cast<int>(moe::kUnitBytes);
    static constexpr int kStages     = G::kGateUpBlocks / kNmma4StageUnits / kNmma4KSplit; // a warp's
    static constexpr int kGateUpCtas = G::kGateUpRowGroups / kNmma4Warps; // per job
    static constexpr int kDownCtas   = G::kDownRowGroups / kNmma4Warps;   // per job
    static constexpr int kDownBytes  = G::kDownBlocks * static_cast<int>(moe::kUnitBytes); // a row group's
    static constexpr int kXCodeBytes = G::kHidden / 2;
    static constexpr int kXRowBytes  = kXCodeBytes + G::kHidden / 16; // codes, then one scale word a block
    static constexpr int kCtaIntermediates = 8 * kNmma4Warps;         // two A4 blocks of h
    static_assert(G::kGateUpBlocks % (kNmma4StageUnits * kNmma4KSplit) == 0 && kStages >= kNmma4Depth,
                  "whole stages in each half, a full ring");
    static_assert(G::kGateUpRowGroups % kNmma4Warps == 0 && G::kDownRowGroups % kNmma4Warps == 0,
                  "every CTA owns whole row groups");
    static_assert(kCtaIntermediates % 16 == 0, "a CTA's intermediates are whole A4 blocks of h");
    static_assert(kStageBytes % 16 == 0 && kDownBytes % 16 == 0 && kXRowBytes % 8 == 0, "16-byte copies, 8-byte stores");
    static_assert(moe::kMaxColumns <= 8, "one MMA's n covers a job's columns");
};

// Whether a call's W4A4 jobs take these kernels (launch_pass decides it): calls of at most kMaxColumns columns, which
// have no wide jobs.
inline bool a4_nmma_call(const MoeExpertSource& source, int columns, const moe::wide::ExpertsWorkspace& layout) {
    return source.activation == moe::ExpertActivation::kA4 && source.a4_shared_input && !ggml_bank(source) &&
           source.wide_a16 != moe::WideA16::kLossless && source.shape == moe::kFlashNextShape &&
           columns <= moe::kMaxColumns && layout.h_blocks != nullptr && layout.x_acts == nullptr;
}

// D = A B on int8 tensor cores, k16: rows g and g + 8 of A (a0, a1) times column g of B (b0), K = 4 t .. 4 t + 3.
__device__ __forceinline__ void mma_k16_ss(int (&d)[4], int a0, int a1, int b0) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// The doubled codes of a block's elements 4 t .. 4 t + 3 from its packed bytes 2 t and 2 t + 1 (element 2 j in byte j's
// low nibble, 2 j + 1 in its high one).
__device__ __forceinline__ int nmma4_x_quad(std::uint32_t pair) {
    const std::uint32_t spread = (pair & 0xFU) | ((pair & 0xF0U) << 4) | ((pair & 0xF00U) << 8) | ((pair & 0xF000U) << 12);
    return static_cast<int>(canon::e2m1_x2_quad(spread));
}

// One unit's contribution to thread (g, t)'s sums: rows g and g + 8 of the unit against column g's codes `b`, the
// rows' weight scales through `scaled`, the columns 2 t and 2 t + 1's activation scales sa0 and sa1.
__device__ __forceinline__ void nmma4_unit(const std::uint8_t* unit, int b, const std::int32_t* scaled, int sa0, int sa1,
                                           int g, int t, long long (&s)[4]) {
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * t + 4 * g);
    const int a0 = static_cast<int>(canon::e2m1_x2_quad(word & 0x0F0F0F0FU));        // row g
    const int a1 = static_cast<int>(canon::e2m1_x2_quad((word >> 4) & 0x0F0F0F0FU)); // row g + 8
    int d[4];
    mma_k16_ss(d, a0, a1, b);
    const int w_lo = scaled[unit[128 + g]], w_hi = scaled[unit[136 + g]];
    // |P_b| <= 2,304 and Sw < 2^18: P_b * Sw fits int32 (unit_sums's fold, per block instead of per quarter).
    s[0] += static_cast<long long>(d[0] * w_lo) * sa0;
    s[1] += static_cast<long long>(d[1] * w_lo) * sa1;
    s[2] += static_cast<long long>(d[2] * w_hi) * sa0;
    s[3] += static_cast<long long>(d[3] * w_hi) * sa1;
}

__device__ __forceinline__ void nmma4_fill_scaled(std::int32_t* scaled, int threads) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += threads) { scaled[i] = canon::e4m3_scaled(static_cast<unsigned>(i)); }
}

struct Nmma4Args {
    const std::uint16_t* x;      // [T][hidden] BF16
    MoeDispatch dispatch;
    MoeExpertSource source;
    int top_k;
    const std::uint8_t* const* job_records;
    const std::int32_t* cpu_flags;
    int job_base;
    canon::A4Block* h_blocks;    // [entries][kHBlocks]
    bf16* outputs;               // [entries][hidden]
    PassPhase phase;
};

// A gate/up CTA's shared memory, sized at launch by the call's columns T (gate_up_nmma4_shared_bytes): the scale table,
// the warps' rings (the second K half's sums reuse them once the K loop is done), then T rows of A4(x) (codes, then
// scale words), whose memory the T rows of BF16 h reuse after the K loop. At one to three columns three CTAs fit an SM
// (100 KB): two columns' 400 CTAs then take one wave of 510 slots instead of 1.2 waves of 340.
template <class G>
struct Nmma4GateUpLayout {
    static constexpr int kScaled = 0;
    static constexpr int kRing   = 256 * 4;
    static constexpr int kRingBytes = kNmma4GuWarps * kNmma4Depth * Nmma4Shape<G>::kStageBytes;
    static constexpr int kX      = kRing + kRingBytes;
    static constexpr int kH      = kX; // after the K loop
    static_assert(kRing % 16 == 0 && kRingBytes % 16 == 0 && Nmma4Shape<G>::kXRowBytes % 16 == 0, "16-byte rows");
    static_assert(kNmma4Warps * 32 * 4 * 8 <= kRingBytes, "the second half's sums fit the rings");
    static_assert(Nmma4Shape<G>::kCtaIntermediates * 2 <= Nmma4Shape<G>::kXRowBytes, "a column's h fits its x row");
    __host__ __device__ static constexpr int bytes(int columns) { return kX + columns * Nmma4Shape<G>::kXRowBytes; }
};
template <class G>
std::size_t gate_up_nmma4_shared_bytes(int columns) { return static_cast<std::size_t>(Nmma4GateUpLayout<G>::bytes(columns)); }

template <class G>
__global__ void __launch_bounds__(kNmma4GuThreads) gate_up_nmma4_kernel(Nmma4Args a, int columns) {
    using S = Nmma4Shape<G>;
    using L = Nmma4GateUpLayout<G>;
    extern __shared__ __align__(16) unsigned char nmma4_gu_shared[];
    struct View {
        std::int32_t* scaled;
        std::uint8_t (*ring)[kNmma4Depth][S::kStageBytes];
        long long (*half)[32][4];
        std::uint8_t (*x)[S::kXRowBytes];
        std::uint16_t (*h)[S::kCtaIntermediates];
    } sm{reinterpret_cast<std::int32_t*>(nmma4_gu_shared + L::kScaled),
         reinterpret_cast<std::uint8_t (*)[kNmma4Depth][S::kStageBytes]>(nmma4_gu_shared + L::kRing),
         reinterpret_cast<long long (*)[32][4]>(nmma4_gu_shared + L::kRing),
         reinterpret_cast<std::uint8_t (*)[S::kXRowBytes]>(nmma4_gu_shared + L::kX),
         reinterpret_cast<std::uint16_t (*)[S::kCtaIntermediates]>(nmma4_gu_shared + L::kH)};
    (void)columns; // the launch's buffer rows (gate_up_nmma4_shared_bytes)
    // The down pass (a programmatic dependent in captured graphs) may launch and start its weight copies now: it
    // reads h only after its PDL wait.
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
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int g = lane / 4, t = lane % 4;
    const int rg = warp % kNmma4Warps, k_half = warp / kNmma4Warps; // this warp's row group in the CTA and K half
    const int group = static_cast<int>(blockIdx.x) * kNmma4Warps + rg;
    const std::uint8_t* units = record + (static_cast<std::size_t>(group) * G::kGateUpBlocks +
                                          static_cast<std::size_t>(k_half) * S::kStages * kNmma4StageUnits) * moe::kUnitBytes;
    auto& ring = sm.ring[warp];
    const auto issue = [&](int stage) {
        const std::uint8_t* src = units + static_cast<std::size_t>(stage) * S::kStageBytes;
        for (int i = lane; i < S::kStageBytes / 16; i += 32) { ggml::tile::cp_async16(ring[stage % kNmma4Depth] + 16 * i, src + 16 * i, 16); }
    };
#pragma unroll
    for (int k = 0; k < kNmma4Depth - 1; ++k) {
        issue(k);
        ggml::tile::cp_async_commit();
    }
    // The job's columns' A4(x) while the rings fill: one block of one column a thread.
    for (int i = static_cast<int>(threadIdx.x); i < count * (G::kHidden / 16); i += kNmma4GuThreads) {
        const int c = i / (G::kHidden / 16), b = i % (G::kHidden / 16);
        const int token = dispatch.entries[first + c] / a.top_k;
        const canon::A4Codes codes = canon::quantize_a4_codes(a.x + static_cast<std::size_t>(token) * G::kHidden + 16 * b,
                                                              scales.input_gate);
        std::uint32_t w[2] = {0, 0};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t byte = static_cast<std::uint32_t>(codes.code[2 * j]) | (static_cast<std::uint32_t>(codes.code[2 * j + 1]) << 4);
            w[j / 4] |= byte << (8 * (j % 4));
        }
        *reinterpret_cast<uint2*>(sm.x[c] + 8 * b) = make_uint2(w[0], w[1]);
        sm.x[c][S::kXCodeBytes + b] = codes.scale_word;
    }
    nmma4_fill_scaled(sm.scaled, kNmma4GuThreads);
    __syncthreads(); // x and the scale table
    // Column g's codes (a thread past the job's columns reads its last one; its sums are not used) and columns 2 t and
    // 2 t + 1's scale words, from this half's first block.
    const int b0 = k_half * S::kStages * kNmma4StageUnits;
    const std::uint8_t* xg  = sm.x[min(g, count - 1)] + 8 * b0 + 2 * t;
    const std::uint8_t* xs0 = sm.x[min(2 * t, count - 1)] + S::kXCodeBytes + b0;
    const std::uint8_t* xs1 = sm.x[min(2 * t + 1, count - 1)] + S::kXCodeBytes + b0;
    long long s[4] = {0, 0, 0, 0};
    for (int st = 0; st < S::kStages; ++st) {
        ggml::tile::cp_async_wait<kNmma4Depth - 2>();
        __syncwarp();
        if (st + kNmma4Depth - 1 < S::kStages) { issue(st + kNmma4Depth - 1); }
        ggml::tile::cp_async_commit();
        const std::uint8_t* stage = ring[st % kNmma4Depth];
#pragma unroll
        for (int u = 0; u < kNmma4StageUnits; ++u) {
            const int b = st * kNmma4StageUnits + u; // within the half
            nmma4_unit(stage + u * moe::kUnitBytes, nmma4_x_quad(*reinterpret_cast<const unsigned short*>(xg + 8 * b)),
                       sm.scaled, sm.scaled[xs0[b]], sm.scaled[xs1[b]], g, t, s);
        }
        __syncwarp(); // every lane is done with this stage before a later issue refills it
    }
    // The second half's sums join the first's (exact int64 adds), through the rings' memory once every warp is done
    // with its ring.
    ggml::tile::cp_async_wait<0>();
    __syncthreads();
    if (k_half == 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { sm.half[rg][lane][i] = s[i]; }
    }
    __syncthreads();
    if (k_half == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { s[i] += sm.half[rg][lane][i]; }
        // Rows g and g + 8 are gate rows for even g (row 2 i is intermediate i's gate, 2 i + 1 its up): the up row's
        // outputs come from lane + 4.
        const bool gate = (g & 1) == 0;
        std::uint16_t y[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { y[i] = canon::a4_row_output(s[i], gate ? scales.alpha_gate : scales.alpha_up); }
        std::uint16_t up[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { up[i] = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y[i], 4)); }
        if (gate) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = 2 * t + (i & 1);
                const int k = rg * 8 + g / 2 + (i < 2 ? 0 : 4); // rows g, g + 8: intermediates g / 2, g / 2 + 4
                if (c < count) { sm.h[c][k] = canon::swiglu_bf16(y[i], up[i]); }
            }
        }
    }
    __syncthreads();
    // The CTA's two A4 blocks of h for each column, with the down input scale.
    constexpr int kBlocks = S::kCtaIntermediates / 16;
    for (int i = static_cast<int>(threadIdx.x); i < count * kBlocks; i += kNmma4GuThreads) {
        const int c = i / kBlocks, blk = i % kBlocks;
        a.h_blocks[static_cast<std::size_t>(first + c) * G::kHBlocks + static_cast<int>(blockIdx.x) * kBlocks + blk] =
            canon::quantize_a4_block(&sm.h[c][16 * blk], scales.input_down);
    }
}

// A down CTA's shared memory: the warps' row groups, then the job's h blocks (as many columns as the call's, sized at
// launch).
template <class G>
struct Nmma4DownShared {
    std::int32_t scaled[256];
    alignas(16) std::uint8_t units[kNmma4Warps][Nmma4Shape<G>::kDownBytes];
    canon::A4Block h[1][G::kHBlocks]; // [columns] at launch
};
template <class G>
std::size_t nmma4_down_shared_bytes(int columns) {
    return offsetof(Nmma4DownShared<G>, h) + static_cast<std::size_t>(columns) * G::kHBlocks * sizeof(canon::A4Block);
}

template <class G>
__global__ void __launch_bounds__(kNmma4Threads) down_nmma4_kernel(Nmma4Args a) {
    using S = Nmma4Shape<G>;
    extern __shared__ __align__(16) unsigned char nmma4_shared[];
    auto& sm = *reinterpret_cast<Nmma4DownShared<G>*>(nmma4_shared);
    const MoeDispatch& dispatch = a.dispatch;
    const int job = a.job_base + static_cast<int>(blockIdx.y);
    // Only h_blocks is read after the wait (the job's gate/up CTAs wrote it); the weights do not depend on it, so
    // their copies start first.
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    moe::ExpertScales scales{};
    if (job < *dispatch.job_count && (a.cpu_flags == nullptr || a.cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(a.phase, a.source, expert, a.job_records, job);
    }
    if (record != nullptr) {
        scales = a.source.scales[expert];
        first  = dispatch.offsets[expert];
        count  = dispatch.offsets[expert + 1] - first;
        if (moe::wide_route(count, scales)) { record = nullptr; }
    }
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int g = lane / 4, t = lane % 4;
    const int group = static_cast<int>(blockIdx.x) * kNmma4Warps + warp;
    if (record != nullptr) {
        const std::uint8_t* src = record + G::kGateUpBytes + static_cast<std::size_t>(group) * S::kDownBytes;
        for (int i = lane; i < S::kDownBytes / 16; i += 32) { ggml::tile::cp_async16(sm.units[warp] + 16 * i, src + 16 * i, 16); }
    }
    ggml::tile::cp_async_commit();
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    constexpr int kWords = static_cast<int>(sizeof(canon::A4Block)) / 4;
    static_assert(sizeof(canon::A4Block) % 4 == 0, "word copies of the h blocks");
    const auto* src = reinterpret_cast<const std::uint32_t*>(a.h_blocks + static_cast<std::size_t>(first) * G::kHBlocks);
    auto* dst       = reinterpret_cast<std::uint32_t*>(&sm.h[0][0]);
    for (int i = static_cast<int>(threadIdx.x); i < count * G::kHBlocks * kWords; i += kNmma4Threads) { dst[i] = __ldcg(src + i); }
    nmma4_fill_scaled(sm.scaled, kNmma4Threads);
    ggml::tile::cp_async_wait<0>();
    __syncthreads(); // the h blocks, the scale table and every lane's weight copies
    const canon::A4Block* hg  = sm.h[min(g, count - 1)];
    const canon::A4Block* hs0 = sm.h[min(2 * t, count - 1)];
    const canon::A4Block* hs1 = sm.h[min(2 * t + 1, count - 1)];
    const std::uint8_t* units = sm.units[warp];
    long long s[4] = {0, 0, 0, 0};
#pragma unroll 8
    for (int b = 0; b < G::kDownBlocks; ++b) {
        const int codes = *reinterpret_cast<const int*>(hg[b].c2 + 4 * t);
        nmma4_unit(units + b * moe::kUnitBytes, codes, sm.scaled, hs0[b].scale_scaled, hs1[b].scale_scaled, g, t, s);
    }
    auto* out = reinterpret_cast<std::uint16_t*>(a.outputs);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = 2 * t + (i & 1);
        if (c < count) {
            const int row = group * 16 + g + (i < 2 ? 0 : 8);
            out[static_cast<std::size_t>(dispatch.entries[first + c]) * G::kHidden + row] =
                canon::a4_row_output(s[i], scales.alpha_down);
        }
    }
}

// One pass of the tensor-core narrow kernels: a gate/up CTA per kNmma4Warps row groups and a down CTA per
// kNmma4Warps output row groups of each job.
void launch_expert_pass_a4_nmma(const bf16* x, int columns, const MoeDispatch& dispatch, const MoeExpertSource& source,
                                int top_k, const std::uint8_t* const* job_records, const std::int32_t* flags, int base,
                                int jobs, const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                                PassPhase phase, PassPart part) {
    using G = moe::FlashNextExperts;
    using S = Nmma4Shape<G>;
    const Nmma4Args args{reinterpret_cast<const std::uint16_t*>(x), dispatch, source, top_k, job_records, flags, base,
                         layout.h_blocks, outputs, phase};
    static const bool configured = [] {
        const auto set = [](auto kernel, std::size_t bytes) {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)) ==
                       cudaSuccess &&
                   cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                        cudaSharedmemCarveoutMaxShared) == cudaSuccess;
        };
        return set(gate_up_nmma4_kernel<G>, gate_up_nmma4_shared_bytes<G>(moe::kMaxColumns)) &&
               set(down_nmma4_kernel<G>, nmma4_down_shared_bytes<G>(moe::kMaxColumns));
    }();
    if (!configured) { throw std::runtime_error("offloaded_sparse_moe: the A4 tensor-core kernels' shared memory"); }
    // A job has at most min(columns, kMaxColumns) columns, so both kernels' column buffers are sized by the call.
    const int call_columns = std::min(columns, moe::kMaxColumns);
    if (part != PassPart::Down) {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                    {dim3(S::kGateUpCtas, jobs), dim3(kNmma4GuThreads), gate_up_nmma4_shared_bytes<G>(call_columns), stream},
                                    gate_up_nmma4_kernel<G>, args, call_columns));
        check_launch("A4 tensor-core gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    const std::size_t down_bytes = nmma4_down_shared_bytes<G>(call_columns);
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_with(part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic,
                                {dim3(S::kDownCtas, jobs), dim3(kNmma4Threads), down_bytes, stream},
                                down_nmma4_kernel<G>, args));
    check_launch("A4 tensor-core down");
}
