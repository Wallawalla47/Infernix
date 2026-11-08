// The tensor-core narrow route of NVFP4 W4A16 experts (design §16.2.1's arithmetic, §19.3.40), included by
// moe_layer.cu after the A16 narrow kernels: the jobs of at most moe::kMaxColumns columns of a call of at most
// moe::wide::kA16NmmaCallColumns columns (decode, verification and prefill up to 4,096 columns; wider jobs keep the wide
// route) on the exact narrow arithmetic at Qwen3.8-Flash-Next's geometry (a16_nmma_call), except calls of at most
// kA16RowsMaxColumns columns, which take the row-pair kernels (narrow_rows_a16.cuh: faster there, level from four).
// Every row sum S = sum_b e4m3_scaled(s_b) P_b is the same int64 as the dp4a kernels' and the CPU engine's, so the
// outputs are theirs bit for bit wherever an expert runs.
//
// - The call's x is encoded once (a16_nmma_prepare_kernel) into the workspace's x limb rows (a16_limbs.cuh) with its
//   column exponents, and the jobs' gate/up CTA counts are cleared.
// - Per 16-element block, rows g and g + 8 of a 16-row group form one m16n8k32 A fragment with K the block twice,
//   A = [8 c | c] (c the doubled E2M1 codes, |8 c| <= 96 a signed byte): one record word per thread gives both
//   rows (its low and high nibbles). Against B = [H1 | H0] (s8) the MMA gives c . (X >> 11), against
//   B = [L1 | L0] (u8) c . (X & 2047), so P_b = 2048 (c . (X >> 11)) + c . (X & 2047) exactly in int32, for up to
//   eight columns at once (the n of the MMA), as the wide route forms it (wide_expert_a16.cuh). Thread (g, t) then
//   adds P times the row's scale into its four int64 sums: rows g and g + 8, columns 2 t and 2 t + 1.
// - Gate/up: a warp owns one row group (eight intermediates' gate and up rows) and streams its units through a
//   private cp.async ring; a gate row's thread pairs with its up row's (lane + 4) for SwiGLU and writes BF16 h. A
//   job of at most kNmma16SharedX columns reads x from a shared-memory copy, wider ones from L2. The job's last CTA
//   to finish (job_ctas) rewrites every entry's h in place as its limbs, with h_emax, once for the down CTAs.
// - Down: a warp owns one row group, its units copied before the PDL wait; the job's h limbs then come to shared
//   memory.

constexpr int kNmma16Warps      = 4;  // row groups of a gate/up CTA and of a down CTA
constexpr int kNmma16Threads    = 32 * kNmma16Warps;
constexpr int kNmma16StageUnits = 8;  // units per stage of a gate/up warp's ring (1,152 bytes)
constexpr int kNmma16Depth      = 4;  // stages in a gate/up warp's ring
constexpr int kNmma16SharedX    = 2;  // columns whose x a gate/up CTA copies to shared memory
constexpr int kNmma16SplitColumns = 4; // calls from this width split a gate/up row group's K over two warps
// A limb row's stride in shared memory: 16 bytes past the row, so two columns' words t fall in different banks.
template <int RowBytes>
inline constexpr int kNmma16RowStride = RowBytes + 16;

template <class G>
struct Nmma16Shape {
    static constexpr int kXRowBytes   = moe::wide::A16Planes<G>::kXRowBytes;
    static constexpr int kHRowBytes   = moe::wide::A16Planes<G>::kHRowBytes;
    static constexpr int kStageBytes  = kNmma16StageUnits * static_cast<int>(moe::kUnitBytes);
    static constexpr int kStages      = G::kGateUpBlocks / kNmma16StageUnits;
    static constexpr int kGateUpCtas  = G::kGateUpRowGroups / kNmma16Warps; // per job
    static constexpr int kDownCtas    = G::kDownRowGroups / kNmma16Warps;   // per job
    static constexpr int kDownBytes   = G::kDownBlocks * static_cast<int>(moe::kUnitBytes); // a row group's
    static_assert(G::kGateUpBlocks % kNmma16StageUnits == 0 && kStages >= kNmma16Depth, "whole stages, a full ring");
    static_assert(G::kGateUpRowGroups % kNmma16Warps == 0 && G::kDownRowGroups % kNmma16Warps == 0,
                  "every CTA owns whole row groups");
    static_assert(kStageBytes % 16 == 0 && kDownBytes % 16 == 0 && kXRowBytes % 16 == 0 && kHRowBytes % 16 == 0,
                  "16-byte copies");
    static_assert(moe::kMaxColumns <= 8, "one MMA's n covers a job's columns");
};

// Whether a call's W4A16 jobs take these kernels (moe_experts' prepare and launch_pass decide it identically).
inline bool a16_nmma_call(const MoeExpertSource& source, int columns, const moe::wide::ExpertsWorkspace& layout) {
    return source.activation == moe::ExpertActivation::kA16 && !ggml_bank(source) &&
           source.narrow_a16 == moe::NarrowA16::kExact && source.wide_a16 != moe::WideA16::kLossless &&
           source.wide_a16 != moe::WideA16::kA8 && source.shape == moe::kFlashNextShape &&
           moe::wide::a16_nmma_columns(columns) && layout.x_limbs != nullptr && layout.job_ctas != nullptr &&
           layout.x_acts == nullptr;
}

// D = A B on int8 tensor cores with signed A and unsigned B (ggml::tile::mma_k32 is signed times signed).
__device__ __forceinline__ void mma_k32_su(int (&d)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

__device__ __forceinline__ std::uint32_t nmma16_lds32(const std::uint8_t* p) { return *reinterpret_cast<const std::uint32_t*>(p); }

// One unit's contribution to thread (g, t)'s sums: its A fragment from the unit's word 32 t + 4 g (rows g and
// g + 8, elements 4 t .. 4 t + 3), B from column g's limb words t of the unit's block (`limbs`: that column's
// limb row at the block's tile, offset by 16 (b % 4) + 4 t), the rows' scales through `scaled`.
template <bool kGlobal>
__device__ __forceinline__ void nmma16_unit(const std::uint8_t* unit, const std::uint8_t* limbs, const std::int32_t* scaled,
                                            int g, int t, long long (&s)[4]) {
    const std::uint32_t word = nmma16_lds32(unit + 32 * t + 4 * g);
    const std::uint32_t c_lo = canon::e2m1_x2_quad(word & 0x0F0F0F0FU);        // row g
    const std::uint32_t c_hi = canon::e2m1_x2_quad((word >> 4) & 0x0F0F0F0FU); // row g + 8
    const int a0 = static_cast<int>((c_lo << 3) & 0xF8F8F8F8U); // 8 c: each byte's own bits
    const int a1 = static_cast<int>((c_hi << 3) & 0xF8F8F8F8U);
    std::uint32_t h1, l1, small;
    if constexpr (kGlobal) {
        h1    = __ldg(reinterpret_cast<const unsigned*>(limbs));
        l1    = __ldg(reinterpret_cast<const unsigned*>(limbs + moe::wide::kA16KTile));
        small = __ldg(reinterpret_cast<const unsigned*>(limbs + 2 * moe::wide::kA16KTile));
    } else {
        h1    = nmma16_lds32(limbs);
        l1    = nmma16_lds32(limbs + moe::wide::kA16KTile);
        small = nmma16_lds32(limbs + 2 * moe::wide::kA16KTile);
    }
    int hi[4], lo[4];
    ggml::tile::mma_k32(hi, a0, a1, static_cast<int>(c_lo), static_cast<int>(c_hi), static_cast<int>(h1),
                  static_cast<int>(small & 0x07070707U));
    mma_k32_su(lo, a0, a1, static_cast<int>(c_lo), static_cast<int>(c_hi), static_cast<int>(l1),
               static_cast<int>((small >> 4) & 0x07070707U));
    const int w_lo = scaled[unit[128 + g]], w_hi = scaled[unit[136 + g]];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // P = 2048 hi + lo in wrapping arithmetic: its true value fits int32.
        const int p = static_cast<int>((static_cast<unsigned>(hi[i]) << 11) + static_cast<unsigned>(lo[i]));
        moe::wide::fold(s[i], p, i < 2 ? w_lo : w_hi);
    }
}

// The limb words of block b for thread t within a limb row.
__device__ __forceinline__ int nmma16_block_offset(int b, int t) { return (b / 4) * moe::wide::kA16LimbTileBytes + 16 * (b % 4) + 4 * t; }

__device__ __forceinline__ void nmma16_fill_scaled(std::int32_t* scaled, int threads) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += threads) { scaled[i] = canon::e4m3_scaled(static_cast<unsigned>(i)); }
}

// Every token's x limbs and exponent (CTA t < columns, one block per thread), and the jobs' gate/up CTA counts
// cleared (CTA `columns`).
template <class G>
__global__ void __launch_bounds__(G::kHidden / 16)
    a16_nmma_prepare_kernel(const std::uint16_t* __restrict__ x, int columns, std::uint8_t* __restrict__ limbs,
                            std::int32_t* __restrict__ emax, std::uint32_t* __restrict__ job_ctas, int max_jobs) {
    constexpr int kEncodeThreads = G::kHidden / 16;
    static_assert(kEncodeThreads % 32 == 0);
    __shared__ unsigned keys[kEncodeThreads / 32];
    const int t = static_cast<int>(blockIdx.x), b = static_cast<int>(threadIdx.x);
    if (t == columns) {
        for (int j = b; j < max_jobs; j += kEncodeThreads) { job_ctas[j] = 0U; }
        return;
    }
    const auto* source = reinterpret_cast<const uint4*>(x + static_cast<std::size_t>(t) * G::kHidden + 16 * b);
    std::uint16_t v[16];
    moe::wide::unpack16(source[0], source[1], v);
    const unsigned key = __reduce_max_sync(0xFFFFFFFFU, moe::wide::a16_key(v));
    if (b % 32 == 0) { keys[b / 32] = key; }
    __syncthreads();
    unsigned all = 0;
#pragma unroll
    for (int w = 0; w < kEncodeThreads / 32; ++w) { all = max(all, keys[w]); }
    const int e = moe::wide::a16_emax(all);
    moe::wide::store_a16_block(limbs + static_cast<std::size_t>(t) * Nmma16Shape<G>::kXRowBytes, b, v, e);
    if (b == 0) { emax[t] = e; }
}

// Entry `entry`'s h (BF16, written by the job's gate/up CTAs) rewritten in place as its limbs, and its exponent:
// one warp, holding the row's blocks in registers (lane l blocks l, l + 32, ...) before it writes. Reads through L2.
template <class G>
__device__ __forceinline__ void nmma16_encode_h(std::uint8_t* __restrict__ h_rows, std::int32_t* __restrict__ h_emax,
                                                int entry, int lane) {
    constexpr int kBlocks = G::kHBlocks;
    constexpr int kPerLane = (kBlocks + 31) / 32;
    std::uint8_t* row = h_rows + static_cast<std::size_t>(entry) * Nmma16Shape<G>::kHRowBytes;
    std::uint16_t v[kPerLane][16];
    unsigned key = 0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int b = lane + 32 * i;
        if (b < kBlocks) {
            const auto* p = reinterpret_cast<const uint4*>(row + 32 * b);
            moe::wide::unpack16(__ldcg(p), __ldcg(p + 1), v[i]);
            key = max(key, moe::wide::a16_key(v[i]));
        }
    }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    const int e = moe::wide::a16_emax(key);
    __syncwarp(); // every lane holds its blocks before any limb overwrites the row
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int b = lane + 32 * i;
        if (b < kBlocks) { moe::wide::store_a16_block(row, b, v[i], e); }
    }
    if (lane == 0) { h_emax[entry] = e; }
}

struct Nmma16Args {
    const std::uint8_t* x_limbs; // [T][kXRowBytes]
    const std::int32_t* x_emax;  // [T]
    MoeDispatch dispatch;
    MoeExpertSource source;
    int top_k;
    const std::uint8_t* const* job_records;
    const std::int32_t* cpu_flags;
    int job_base;
    std::uint8_t* h_rows;        // [entries][kHRowBytes]: BF16 h, then the job's last gate/up CTA's limbs
    std::int32_t* h_emax;        // [entries]
    std::uint32_t* job_ctas;     // [max_jobs]: gate/up CTAs finished per job, then one more once its h is encoded
    bf16* outputs;               // [entries][hidden]
    PassPhase phase;
    // 1: each down CTA waits for its own job's h (job_ctas reaching kGateUpCtas + 1) instead of the whole gate/up
    // grid's PDL wait (calls of at most moe::kMaxColumns columns; launch_expert_pass_a16_nmma).
    int job_wait = 0;
};

// A gate/up CTA's warps: KSplit warps a row group, each streaming 1 / KSplit of K through a Depth-stage ring.
template <int KSplit>
struct Nmma16GateUpWarps {
    static constexpr int kSplit   = KSplit;
    static constexpr int kDepth   = KSplit == 1 ? kNmma16Depth : 3; // 3 keeps two split CTAs an SM (48 KB static)
    static constexpr int kWarps   = kNmma16Warps * KSplit;
    static constexpr int kThreads = 32 * kWarps;
};

template <class G, int KSplit>
struct Nmma16GateUpShared {
    using W = Nmma16GateUpWarps<KSplit>;
    std::int32_t scaled[256];
    alignas(16) std::uint8_t x[kNmma16SharedX][kNmma16RowStride<Nmma16Shape<G>::kXRowBytes>];
    // The rings; after the K loop the later K parts' sums ([part - 1][row group][lane][4]) reuse them.
    alignas(16) std::uint8_t ring[W::kWarps][W::kDepth][Nmma16Shape<G>::kStageBytes];
    static_assert(KSplit == 1 || (KSplit - 1) * kNmma16Warps * 32 * 4 * sizeof(long long) <= sizeof(ring),
                  "the later K parts' sums fit the rings");
};

template <class G, int KSplit>
__global__ void __launch_bounds__(Nmma16GateUpWarps<KSplit>::kThreads) gate_up_nmma16_kernel(Nmma16Args a) {
    using S = Nmma16Shape<G>;
    using W = Nmma16GateUpWarps<KSplit>;
    constexpr int kPartStages = S::kStages / KSplit;
    static_assert(S::kStages % KSplit == 0 && kPartStages >= W::kDepth, "whole stages in each K part, a full ring");
    __shared__ Nmma16GateUpShared<G, KSplit> sm;
    __shared__ int last_cta;
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
    // This warp's row group in the CTA and its part of K (blocks [part, part + 1) * kPartStages * kNmma16StageUnits).
    const int rg = warp % kNmma16Warps, part = warp / kNmma16Warps;
    const int group = static_cast<int>(blockIdx.x) * kNmma16Warps + rg;
    const int b0    = part * kPartStages * kNmma16StageUnits;
    const std::uint8_t* units =
        record + (static_cast<std::size_t>(group) * G::kGateUpBlocks + static_cast<std::size_t>(b0)) * moe::kUnitBytes;
    const auto token = [&](int c) { return moe::wide::entry_column(dispatch.entries[first + c], a.top_k, dispatch.shared_first); };
    const bool shared_x = count <= kNmma16SharedX;
    if (shared_x) {
        constexpr int kChunks = S::kXRowBytes / 16;
        for (int i = static_cast<int>(threadIdx.x); i < count * kChunks; i += W::kThreads) {
            const int c = i / kChunks, k = i % kChunks;
            ggml::tile::cp_async16(sm.x[c] + 16 * k, a.x_limbs + static_cast<std::size_t>(token(c)) * S::kXRowBytes + 16 * k, 16);
        }
    }
    ggml::tile::cp_async_commit();
    auto& ring = sm.ring[warp];
    const auto issue = [&](int stage) {
        const std::uint8_t* src = units + static_cast<std::size_t>(stage) * S::kStageBytes;
        for (int i = lane; i < S::kStageBytes / 16; i += 32) { ggml::tile::cp_async16(ring[stage % W::kDepth] + 16 * i, src + 16 * i, 16); }
    };
#pragma unroll
    for (int k = 0; k < W::kDepth - 1; ++k) {
        issue(k);
        ggml::tile::cp_async_commit();
    }
    nmma16_fill_scaled(sm.scaled, W::kThreads);
    ggml::tile::cp_async_wait<W::kDepth - 1>(); // the x group
    __syncthreads();                            // x and the scale table
    // Column g's limb row (a thread past the job's columns reads its last one; its sums are not used).
    const int column = min(g, count - 1);
    long long s[4] = {0, 0, 0, 0};
    const auto k_loop = [&](auto global_x, const std::uint8_t* limbs) {
        constexpr bool kGlobal = decltype(global_x)::value;
        for (int st = 0; st < kPartStages; ++st) {
            ggml::tile::cp_async_wait<W::kDepth - 2>();
            __syncwarp();
            if (st + W::kDepth - 1 < kPartStages) { issue(st + W::kDepth - 1); }
            ggml::tile::cp_async_commit();
            const std::uint8_t* stage = ring[st % W::kDepth];
#pragma unroll
            for (int u = 0; u < kNmma16StageUnits; ++u) {
                const int b = b0 + st * kNmma16StageUnits + u;
                nmma16_unit<kGlobal>(stage + u * moe::kUnitBytes, limbs + nmma16_block_offset(b, t), sm.scaled, g, t, s);
            }
            __syncwarp(); // every lane is done with this stage before a later issue refills it
        }
    };
    if (shared_x) {
        k_loop(std::false_type{}, sm.x[column]);
    } else {
        k_loop(std::true_type{}, a.x_limbs + static_cast<std::size_t>(token(column)) * S::kXRowBytes);
    }
    if constexpr (KSplit > 1) {
        // The later parts' sums join part 0's (exact int64 adds, so the bits are the unsplit kernel's), through the
        // rings' memory once every warp is done with its ring.
        auto* parts = reinterpret_cast<long long (*)[kNmma16Warps][32][4]>(&sm.ring[0][0][0]);
        ggml::tile::cp_async_wait<0>();
        __syncthreads();
        if (part > 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { parts[part - 1][rg][lane][i] = s[i]; }
        }
        __syncthreads();
        if (part == 0) {
#pragma unroll
            for (int p = 1; p < KSplit; ++p) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { s[i] += parts[p - 1][rg][lane][i]; }
            }
        }
    }
    if (part == 0) {
        // Rows g and g + 8 are gate rows for even g (row 2 i is intermediate i's gate, 2 i + 1 its up): the up row's
        // outputs come from lane + 4.
        const bool gate = (g & 1) == 0;
        std::uint16_t y[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = 2 * t + (i & 1);
            y[i] = canon::a16_row_output(s[i], a.x_emax[token(min(c, count - 1))], gate ? scales.alpha_gate : scales.alpha_up);
        }
        std::uint16_t up[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { up[i] = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y[i], 4)); }
        if (gate) {
            auto* h = reinterpret_cast<std::uint16_t*>(a.h_rows);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int c = 2 * t + (i & 1);
                const int k = group * 8 + g / 2 + (i < 2 ? 0 : 4); // rows g, g + 8: intermediates g / 2, g / 2 + 4
                if (c < count) {
                    h[static_cast<std::size_t>(first + c) * moe::wide::A16Planes<G>::kHRowElements + k] = canon::swiglu_bf16(y[i], up[i]);
                }
            }
        }
    }
    // The job's last CTA encodes its h. The barrier orders every thread's h stores before thread 0's acq_rel count
    // (a cumulative release), and the last CTA's acquire reaches its other threads through the barrier.
    __syncthreads();
    if (threadIdx.x == 0) {
        std::uint32_t done;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(done) : "l"(a.job_ctas + job) : "memory");
        last_cta = done == S::kGateUpCtas - 1 ? 1 : 0;
    }
    __syncthreads();
    if (last_cta == 0) { return; } // (a job belongs to one pass; the call's prepare kernel clears the counts)
    for (int c = warp; c < count; c += W::kWarps) { nmma16_encode_h<G>(a.h_rows, a.h_emax, first + c, lane); }
    // The job's h is complete: one more count, a fire-and-forget release after the barrier (cumulative over every
    // warp's limbs and exponents), lets this job's down CTAs start without waiting for the rest of the grid.
    __syncthreads();
    if (threadIdx.x == 0) { asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(a.job_ctas + job) : "memory"); }
}

// A down CTA's shared memory: the warps' row groups, then the job's h limb rows (as many as its columns, sized at
// launch).
template <class G>
struct Nmma16DownShared {
    std::int32_t scaled[256];
    int emax[moe::kMaxColumns];
    alignas(16) std::uint8_t units[kNmma16Warps][Nmma16Shape<G>::kDownBytes];
    alignas(16) std::uint8_t h[1][kNmma16RowStride<Nmma16Shape<G>::kHRowBytes>]; // [columns] at launch
};
template <class G>
std::size_t nmma16_down_shared_bytes(int columns) {
    return offsetof(Nmma16DownShared<G>, h) +
           static_cast<std::size_t>(columns) * kNmma16RowStride<Nmma16Shape<G>::kHRowBytes>;
}

template <class G>
__global__ void __launch_bounds__(kNmma16Threads) down_nmma16_kernel(Nmma16Args a) {
    using S = Nmma16Shape<G>;
    extern __shared__ __align__(16) unsigned char nmma16_shared[];
    auto& sm = *reinterpret_cast<Nmma16DownShared<G>*>(nmma16_shared);
    const MoeDispatch& dispatch = a.dispatch;
    const int job = a.job_base + static_cast<int>(blockIdx.y);
    // Only h_rows and h_emax are read after the wait (the job's last gate/up CTA encoded its h); the weights do not
    // depend on it, so their copies start first.
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
    const int group = static_cast<int>(blockIdx.x) * kNmma16Warps + warp;
    if (record != nullptr) {
        const std::uint8_t* src = record + G::kGateUpBytes + static_cast<std::size_t>(group) * S::kDownBytes;
        for (int i = lane; i < S::kDownBytes / 16; i += 32) { ggml::tile::cp_async16(sm.units[warp] + 16 * i, src + 16 * i, 16); }
    }
    ggml::tile::cp_async_commit();
    if (a.job_wait != 0) {
        // Only this job's h is read (and outputs written): wait for the count its last gate/up CTA releases after
        // encoding h (candidates YF-01, after ninferpro f48f1c051a). The gate/up grid is a serialized launch, so every
        // earlier kernel has finished, and this grid launched only once every gate/up CTA had started (each triggers
        // at entry): the job's CTAs are running, and its other jobs' down work need not wait for them. Bounded
        // (offloaded_moe::kDownReadyTimeoutNs): a lost count fails the call through its error word, never hangs.
        if (record != nullptr && threadIdx.x == 0) {
            const std::uint32_t ready = static_cast<std::uint32_t>(S::kGateUpCtas) + 1U;
            if (load_acquire_gpu(a.job_ctas + job) != ready) {
                const std::uint64_t entered = global_ns();
                do {
                    if (global_ns() - entered > offloaded_moe::kDownReadyTimeoutNs) {
                        report_error(a.source, offloaded_moe::kErrorDownStalled);
                        break;
                    }
                    __nanosleep(64);
                } while (load_acquire_gpu(a.job_ctas + job) != ready);
            }
        }
        __syncthreads(); // thread 0's acquire reaches the other threads (record is uniform in the CTA)
    } else {
        pdl::wait_for_dependencies();
    }
    if (record == nullptr) { return; }
    constexpr int kChunks = S::kHRowBytes / 16;
    for (int i = static_cast<int>(threadIdx.x); i < count * kChunks; i += kNmma16Threads) {
        const int c = i / kChunks, k = i % kChunks;
        ggml::tile::cp_async16(sm.h[c] + 16 * k, a.h_rows + static_cast<std::size_t>(first + c) * S::kHRowBytes + 16 * k, 16);
    }
    ggml::tile::cp_async_commit();
    if (static_cast<int>(threadIdx.x) < count) { sm.emax[threadIdx.x] = __ldcg(a.h_emax + first + threadIdx.x); }
    nmma16_fill_scaled(sm.scaled, kNmma16Threads);
    ggml::tile::cp_async_wait<0>();
    __syncthreads(); // the h limbs and exponents, the scale table and every lane's weight copies
    const std::uint8_t* limbs = sm.h[min(g, count - 1)];
    const std::uint8_t* units = sm.units[warp];
    long long s[4] = {0, 0, 0, 0};
#pragma unroll 8
    for (int b = 0; b < G::kDownBlocks; ++b) {
        nmma16_unit<false>(units + b * moe::kUnitBytes, limbs + nmma16_block_offset(b, t), sm.scaled, g, t, s);
    }
    auto* out = reinterpret_cast<std::uint16_t*>(a.outputs);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = 2 * t + (i & 1);
        if (c < count) {
            const int row = group * 16 + g + (i < 2 ? 0 : 8);
            out[static_cast<std::size_t>(dispatch.entries[first + c]) * G::kHidden + row] =
                canon::a16_row_output(s[i], sm.emax[c], scales.alpha_down);
        }
    }
}

// Encodes the call's x and clears its jobs' gate/up CTA counts (once per call, before its passes).
void launch_prepare_a16_nmma(const Tensor& x, int max_jobs, const moe::wide::ExpertsWorkspace& layout, cudaStream_t stream) {
    using G = moe::FlashNextExperts;
    const int columns = static_cast<int>(x.ne[1]);
    a16_nmma_prepare_kernel<G><<<static_cast<unsigned>(columns + 1), G::kHidden / 16, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), columns, layout.x_limbs, layout.x_emax, layout.job_ctas, max_jobs);
    check_launch("A16 tensor-core prepare");
}

// One pass of the tensor-core narrow kernels: a gate/up CTA per kNmma16Warps row groups and a down CTA per
// kNmma16Warps output row groups of each job.
void launch_expert_pass_a16_nmma(int columns, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                                 const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                                 const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                                 PassPhase phase, PassPart part) {
    using G = moe::FlashNextExperts;
    using S = Nmma16Shape<G>;
    Nmma16Args args{layout.x_limbs, layout.x_emax, dispatch, source, top_k, job_records, flags, base,
                    layout.h_rows, layout.h_emax, layout.job_ctas, outputs, phase};
    // Decode and verify calls (at most moe::kMaxColumns columns): each down CTA waits for its own job's h. ninferpro
    // measured that wait level with the grid-wide one at 12 columns and 0.8 % slower at 18 on its dp4a kernels, so
    // wider calls keep the PDL wait. A split call's down launch is serialized: its gate/up pass has already ended.
    // Measured against the grid-wide wait (2026-11-08, FP4-Local, DFlash2 and n-gram auto, BF16 KV, three ABBA blocks
    // of fresh processes, the same tokens): conv 213.6 -> 215.4 tok/s (0.6-1.1 % in each block), struct 474.8 ->
    // 477.7 (0.4-0.8 %); the GGUF kernels' twin on IQ3_S, struct 420.5 -> 428.6 (1.5-2.1 %).
    args.job_wait = part != PassPart::Down && columns <= moe::kMaxColumns ? 1 : 0;
    if (part != PassPart::Down) {
        // Calls of kNmma16SplitColumns columns or more split each row group's K over two warps; narrower ones keep one
        // warp a group (op bench, FP4-Local's geometry, frames, cold L2, three interleaved runs, 2026-11-03: split
        // against unsplit 1 column 43.6 / 43.8 us, 2 66.1 / 66.1, 3 74.1 / 72.4 (+2.4 %), 4 90.8 / 92.7, 6 115.4 /
        // 117.5, 8 133.9 / 135.7, 16 232.3 / 236.4; same hashes; FP4-Local decode, BF16 KV, three ABBA blocks: DFlash2
        // after a 16K prompt 227.25 -> 227.70 tok/s, 3 of 3 ahead, the struct fixture 430.83 -> 431.27, plain decode
        // (one column, never split) 128.90 -> 128.80, the same text and rounds).
        if (columns < kNmma16SplitColumns) {
            CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                        {dim3(S::kGateUpCtas, jobs), dim3(Nmma16GateUpWarps<1>::kThreads), 0, stream},
                                        gate_up_nmma16_kernel<G, 1>, args));
        } else {
            CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                        {dim3(S::kGateUpCtas, jobs), dim3(Nmma16GateUpWarps<2>::kThreads), 0, stream},
                                        gate_up_nmma16_kernel<G, 2>, args));
        }
        check_launch("A16 tensor-core gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    // A job has at most min(columns, kMaxColumns) columns, so the down CTAs' h rows are sized by the call.
    const std::size_t down_bytes = nmma16_down_shared_bytes<G>(std::min(columns, moe::kMaxColumns));
    // A programmatic dependent of the gate/up pass in captured graphs (down_kernel's PDL contract).
    CUDA_CHECK(pdl::launch_with(part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic,
                                {dim3(S::kDownCtas, jobs), dim3(kNmma16Threads), down_bytes, stream},
                                down_nmma16_kernel<G>, args));
    check_launch("A16 tensor-core down");
}
