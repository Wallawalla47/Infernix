// GGUF (ggml) expert banks on the GPU (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25), included
// by moe_layer.cu after its pass helpers. A record holds the gate rows, the up rows (one ggml
// format) and the down rows (another), each row verbatim GGUF blocks.
//
// The arithmetic is ops/ggml/ggml_exact.h: x and h are encoded per column as A16 limbs (ggml_dp4a.cuh's
// layout) and every output row is an exact int128 rounded once, so an expert's outputs depend neither on
// where its record lives nor on how many columns share the call. Jobs of at most kMaxColumns columns run the
// narrow kernels on int8 tensor cores (ops/ggml/ggml_narrow_mma.cuh; below), wider jobs the exact tiles.
//
// - ggml_encode_x_kernel encodes every token's x once per call into the A16 workspace's x limb
//   rows (kHidden / 32 slices of kSliceWords words: exactly kA16XRowBytes; the expert formats
//   have no min term, so no slice sums) and x_emax.

// The block-FP8 bank's dp4a kernels (fp8_expert.cuh): a gate/up CTA owns kGgmlGateUpUnits intermediates, one
// per warp; a down CTA kGgmlDownRows output rows, four lanes a row.
constexpr int kGgmlThreads       = 128;
constexpr int kGgmlWarps         = kGgmlThreads / 32;
constexpr int kGgmlGateUpUnits   = kGgmlWarps;      // intermediates per gate/up CTA
constexpr int kGgmlDownLanes     = 4;               // lanes per down row
constexpr int kGgmlDownRows      = kGgmlThreads / kGgmlDownLanes; // 32 rows per down CTA
constexpr int kGgmlColumns       = 4;               // columns per pass of either kernel
constexpr int kGgmlXSlices       = moe::kHidden / 32;
constexpr int kGgmlHSlices       = moe::kIntermediate / 32;
static_assert(kGgmlXSlices * ggml::fast::kSliceWords * 4 == moe::wide::kA16XRowBytes,
              "a ggml call's x limbs fill the A16 x limb row exactly");
static_assert(kGgmlColumns == kGgmlWarps, "a down CTA encodes one column per warp");

// One CTA per column; a last CTA clears the jobs' gate/up CTA counts (the tensor-core route's).
__global__ void __launch_bounds__(kThreads)
    ggml_encode_x_kernel(const bf16* __restrict__ x, int columns, std::uint8_t* __restrict__ x_limbs,
                         std::int32_t* __restrict__ x_emax, std::uint32_t* __restrict__ job_ctas, int max_jobs) {
    __shared__ unsigned warp_key[kWarps];
    __shared__ int column_emax;
    const int t = blockIdx.x;
    if (t == columns) {
        for (int j = static_cast<int>(threadIdx.x); j < max_jobs; j += kThreads) { job_ctas[j] = 0U; }
        return;
    }
    const auto* v = reinterpret_cast<const std::uint16_t*>(x) + static_cast<std::size_t>(t) * moe::kHidden;
    unsigned key = 0;
    for (int j = threadIdx.x; j < moe::kHidden; j += blockDim.x) { key = max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (threadIdx.x % 32 == 0) { warp_key[threadIdx.x / 32] = key; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int w = 0; w < kWarps; ++w) { m = max(m, warp_key[w]); }
        column_emax = m >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(m >> 7), 1);
        x_emax[t]   = column_emax;
    }
    __syncthreads();
    const int e = column_emax;
    auto* column = reinterpret_cast<std::int32_t*>(x_limbs + static_cast<std::size_t>(t) * moe::wide::kA16XRowBytes);
    const auto value = [&](int j) { return e == canon::kA16NonFinite ? 0 : canon::a16_value(v[j], e); };
    for (int g = threadIdx.x; g < moe::kHidden / 4; g += blockDim.x) {
        (void)ggml::fast::store_group(column, g, value(4 * g), value(4 * g + 1), value(4 * g + 2), value(4 * g + 3));
    }
}

__device__ __forceinline__ void ggml_shfl_add(ggml::I128& r, int offset) {
    const auto lo = __shfl_xor_sync(0xFFFFFFFFU, static_cast<unsigned long long>(r.lo), offset);
    const auto hi = __shfl_xor_sync(0xFFFFFFFFU, static_cast<long long>(r.hi), offset);
    ggml::add(r, ggml::I128{static_cast<std::uint64_t>(lo), static_cast<std::int64_t>(hi)});
}

// ------------------------------------------------------------------- narrow tensor-core route
//
// The same outputs on int8 tensor cores (ops/ggml/ggml_narrow_mma.cuh): a warp owns 8 weight rows and
// streams them through a private cp.async ring; CTAs of kNmmaWarps warps.
// - Gate/up: a warp's rows are 4 intermediates' gate rows then their up rows, K in ten one-block stages
//   kNmmaDepth deep; thread (g, 0) of a gate row pairs it with its up row (lane + 16) for SwiGLU. A job of at
//   most kNmmaSharedX columns reads x from a shared-memory copy. The job's last CTA to finish (a per-job count
//   in the workspace) encodes every entry's h once for the down tiles: its limbs over the h row, and h_emax.
// - Down: a warp's rows are 8 output rows, K in two resident stages of ten slices copied before the PDL wait;
//   the job's encoded h is then copied to shared memory (its columns: the call's, at most kMaxColumns).
//   Row sums accumulate in int64 units of the row's smallest scale shift (int128 when a row's span is wider).
// Measured on the RTX 5090 (2026-10-29): in-model decode +5.9 % (struct), +3.5 % (16K) over the dp4a kernels.
// Rejected: x a stage ahead in registers (156 registers), gate/up int64 base accumulation (its scale-word
// loads stalled, six K-loop copies), a persistent gate/up tile stream (whole-tile slots held shared memory
// while computing: -6 %), persistent down tiles with h through L1 (-20 % on the down pass), 4-warp gate/up
// CTAs for one- and two-column calls (no gain).

constexpr int kNmmaWarps      = 8;
constexpr int kNmmaThreads    = 32 * kNmmaWarps;
constexpr int kNmmaUnits      = 4 * kNmmaWarps; // intermediates per gate/up CTA
constexpr int kNmmaDepth      = 3;              // gate/up stages in a warp's ring
constexpr int kNmmaSharedX    = 2;              // columns whose x a gate/up CTA copies to shared memory
constexpr int kNmmaDownRows   = 8 * kNmmaWarps; // output rows per down CTA
constexpr int kNmmaDownSlices = kGgmlHSlices / 2;
constexpr int kNmmaGateUpCtas = moe::kIntermediate / kNmmaUnits; // per job
static_assert(moe::kIntermediate % kNmmaUnits == 0 && moe::kHidden % kNmmaDownRows == 0);
static_assert(moe::kMaxColumns <= kNmmaWarps, "a job's last gate/up CTA encodes one entry per warp");

template <QType GU>
using NmmaGateUpRing = ggml::nmma::Ring<8, kGgmlXSlices / 8, kNmmaDepth, ggml::nmma::kBlockBytes<GU>>;
template <QType D>
using NmmaDownRing =
    ggml::nmma::Ring<kNmmaDownSlices, 2, 2, kNmmaDownSlices / ggml::nmma::kBlockSlices<D> * ggml::nmma::kBlockBytes<D>>;

template <QType GU>
struct NmmaGateUpShared {
    ggml::fast::SharedCodebook<GU> cb;
    alignas(16) std::int32_t x[kNmmaSharedX][kGgmlXSlices * ggml::fast::kSliceWords];
    alignas(16) std::uint8_t ring[kNmmaWarps][NmmaGateUpRing<GU>::kBytes];
};

// A down CTA's shared memory: the rows' rings, then the job's encoded h, kA16HRowBytes a column for as many
// columns as the call has (at most kMaxColumns): sized at launch, so one- and two-column calls fit more CTAs.
template <QType D>
struct NmmaDownShared {
    int emax[moe::kMaxColumns];
    alignas(16) std::uint8_t ring[kNmmaWarps][NmmaDownRing<D>::kBytes];
    alignas(16) std::int32_t limbs[1][kGgmlHSlices * ggml::fast::kSliceWords]; // [columns] at launch
};
template <QType D>
std::size_t nmma_down_shared_bytes(int columns) {
    return offsetof(NmmaDownShared<D>, limbs) + static_cast<std::size_t>(std::min(columns, moe::kMaxColumns)) * moe::wide::kA16HRowBytes;
}

// Entry `entry`'s h (BF16, written by the call's gate/up CTAs) rewritten in place as its limbs, and its exponent:
// one warp, holding the whole row in registers (the wide route's encode_h_ggml layout). Reads through L2.
__device__ __forceinline__ void nmma_encode_h(std::uint16_t* __restrict__ h_plane, std::int32_t* __restrict__ h_emax,
                                              int entry, int lane) {
    constexpr int kGroups = moe::kIntermediate / 4 / 32; // per lane
    std::uint16_t* h = h_plane + static_cast<std::size_t>(entry) * moe::wide::kA16HRowElements;
    std::uint16_t v[kGroups][4];
    unsigned key = 0;
#pragma unroll
    for (int i = 0; i < kGroups; ++i) {
        const uint2 w = __ldcg(reinterpret_cast<const uint2*>(h + 4 * (lane + 32 * i)));
        v[i][0] = static_cast<std::uint16_t>(w.x & 0xFFFFU);
        v[i][1] = static_cast<std::uint16_t>(w.x >> 16);
        v[i][2] = static_cast<std::uint16_t>(w.y & 0xFFFFU);
        v[i][3] = static_cast<std::uint16_t>(w.y >> 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) { key = max(key, static_cast<unsigned>(v[i][j] & 0x7FFFU)); }
    }
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    const int e = key >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(key >> 7), 1);
    __syncwarp();
    const auto value = [&](std::uint16_t b) { return e == canon::kA16NonFinite ? 0 : canon::a16_value(b, e); };
    auto* column_words = reinterpret_cast<std::int32_t*>(h);
#pragma unroll
    for (int i = 0; i < kGroups; ++i) {
        (void)ggml::fast::store_group(column_words, lane + 32 * i, value(v[i][0]), value(v[i][1]), value(v[i][2]),
                                      value(v[i][3]));
    }
    if (lane == 0) { h_emax[entry] = e; }
}

// The arguments of a pass of the narrow tensor-core kernels.
struct NmmaPassArgs {
    const std::uint8_t* x_limbs;
    const std::int32_t* x_emax;
    MoeDispatch dispatch;
    MoeExpertSource source;
    int top_k;
    const std::uint8_t* const* job_records;
    const std::int32_t* cpu_flags;
    int job_base, jobs;
    std::uint16_t* h_plane;
    std::int32_t* h_emax;
    std::uint32_t* job_ctas; // [max_jobs]: gate/up CTAs finished per job
    bf16* outputs;
    int hidden;
    PassPhase phase;
    int narrow_max;
};

// Gate/up CTA tile `bx` (intermediates 32 bx ..) of job `job`: every thread of the CTA calls it. The codebook is
// in sm.cb already, or (load_codebook) loaded here and published by the x barrier.
template <QType GU>
__device__ __forceinline__ void gate_up_nmma_task(NmmaGateUpShared<GU>& sm, const NmmaPassArgs& a, int job, int bx,
                                                  bool load_codebook) {
    namespace nm = ggml::nmma;
    constexpr int W        = kNmmaWarps;
    constexpr int kThreads = kNmmaThreads;
    __shared__ int last_cta;
    const auto& dispatch = a.dispatch;
    const auto& source   = a.source;
    const std::uint8_t* x_limbs = a.x_limbs;
    const std::int32_t* x_emax  = a.x_emax;
    std::uint16_t* h_plane      = a.h_plane;
    const int top_k             = a.top_k;
    if (job >= *dispatch.job_count || (a.cpu_flags != nullptr && a.cpu_flags[job] != 0)) { return; }
    const int expert           = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(a.phase, source, expert, a.job_records, job);
    if (record == nullptr) { return; }
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (count > a.narrow_max) { return; } // a wide job: its tiles compute it
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int i0   = bx * 4 * W + 4 * warp;
    const std::size_t row_bytes = source.ggml.gate_row_bytes;
    const auto row_of = [&](int r) {
        return record + static_cast<std::size_t>(r < 4 ? i0 + r : moe::kIntermediate + i0 + r - 4) * row_bytes;
    };
    std::uint8_t* ring = sm.ring[warp];
    const auto token   = [&](int c) { return dispatch.entries[first + c] / top_k; };
    if (load_codebook) { sm.cb.template load<kThreads>(); }
    // A job of at most kNmmaSharedX columns reads its x from shared memory (one copy per CTA, the first commit
    // group), wider ones from L2 with the next stage's lines prefetched into L1.
    const bool shared_x = count <= kNmmaSharedX;
    if (shared_x) {
        constexpr int kChunks = moe::wide::kA16XRowBytes / 16;
        for (int i = static_cast<int>(threadIdx.x); i < count * kChunks; i += kThreads) {
            const int c = i / kChunks, k = i % kChunks;
            ggml::tile::cp_async16(reinterpret_cast<std::uint8_t*>(sm.x[c]) + 16 * k,
                                   x_limbs + static_cast<std::size_t>(token(c)) * moe::wide::kA16XRowBytes + 16 * k, 16);
        }
    }
    ggml::tile::cp_async_commit();
    NmmaGateUpRing<GU>::prologue(ring, row_of, lane);
    ggml::tile::cp_async_wait<kNmmaDepth - 1>(); // the x group
    __syncthreads();                             // the codebook and x
    // Columns [pass, pass + n) of the job (n <= C).
    const auto run = [&](auto width, auto from_shared, int pass, int n) {
        constexpr int C       = decltype(width)::value;
        constexpr bool kShared = decltype(from_shared)::value;
        const std::int32_t* xs[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            xs[c] = kShared ? sm.x[min(c, n - 1)]
                            : reinterpret_cast<const std::int32_t*>(x_limbs + static_cast<std::size_t>(token(pass + min(c, n - 1))) *
                                                                                  moe::wide::kA16XRowBytes);
        }
        const auto xpair = [&](int c, int slice, int word) {
            const std::int32_t* p = xs[c] + slice * ggml::fast::kSliceWords + word;
            if constexpr (kShared) {
                if constexpr (nm::kWordStep<GU> == 1) { return *reinterpret_cast<const int2*>(p); }
                return make_int2(p[0], p[nm::kWordStep<GU>]);
            } else {
                if constexpr (nm::kWordStep<GU> == 1) { return __ldg(reinterpret_cast<const int2*>(p)); }
                return make_int2(__ldg(p), __ldg(p + nm::kWordStep<GU>));
            }
        };
        const auto xprefetch = [&](int c, int slice, int word) {
            if constexpr (!kShared) {
                asm volatile("prefetch.global.L1 [%0];" ::"l"(xs[c] + slice * ggml::fast::kSliceWords + word));
            }
        };
        nm::Rows<C> acc;
        if (pass > 0) { NmmaGateUpRing<GU>::prologue(ring, row_of, lane); }
        NmmaGateUpRing<GU>::template body<GU, C, false>(ring, row_of, sm.cb, xpair, xprefetch, lane, acc);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const ggml::I128 r     = nm::combine_limbs(acc.r[c]);
            const std::uint16_t y  = ggml::output_bf16(r, x_emax[token(pass + min(c, n - 1))], GU);
            const std::uint16_t up = static_cast<std::uint16_t>(__shfl_down_sync(0xFFFFFFFFU, y, 16));
            if (c < n && lane < 16 && lane % 4 == 0) {
                h_plane[static_cast<std::size_t>(first + pass + c) * moe::wide::kA16HRowElements + i0 + lane / 4] =
                    canon::swiglu_bf16(y, up);
            }
        }
    };
    if (count == 1) {
        run(std::integral_constant<int, 1>{}, std::true_type{}, 0, 1);
    } else if (count == 2) {
        run(std::integral_constant<int, 2>{}, std::true_type{}, 0, 2);
    } else {
        for (int pass = 0; pass < count; pass += 4) { run(std::integral_constant<int, 4>{}, std::false_type{}, pass, min(4, count - pass)); }
    }
    // The job's last CTA encodes its h. The barrier orders every thread's h stores before thread 0's acq_rel
    // count (a cumulative release; per-thread __threadfence before it cost every CTA a store drain at exit: T = 8
    // op +2.7 %, struct decode -3 %), and the last CTA's acquire reaches its other threads through the barrier.
    __syncthreads();
    if (threadIdx.x == 0) {
        std::uint32_t done;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(done) : "l"(a.job_ctas + job) : "memory");
        last_cta = done == kNmmaGateUpCtas - 1 ? 1 : 0;
    }
    __syncthreads();
    if (last_cta == 0) { return; } // (a job belongs to one pass; the call's x encoder clears the counts)
    if (warp < count) { nmma_encode_h(h_plane, a.h_emax, first + warp, lane); }
}

template <QType GU>
__global__ void __launch_bounds__(kNmmaThreads) gate_up_nmma_ggml(NmmaPassArgs a) {
    extern __shared__ __align__(16) unsigned char nmma_shared[];
    pdl::trigger_dependents();
    gate_up_nmma_task<GU>(*reinterpret_cast<NmmaGateUpShared<GU>*>(nmma_shared), a, a.job_base + static_cast<int>(blockIdx.y),
                          static_cast<int>(blockIdx.x), true);
}

// Down CTA tile `bx` (output rows 64 bx ..) of job `job`: every thread of the CTA calls it, a programmatic
// dependent of the gate/up pass (every CTA waits on PDL, after its weight copies are issued). (One persistent
// launch of a pass's gate/up and down tiles, down tasks waiting for their job's h, measured slower: 16K 264.8 ->
// 258.2, struct 463.1 -> 455.9, 2026-10-29: its CTAs held every SM through the pass, where the two launches'
// retiring CTAs let the prioritized shared-expert and staged kernels in.)
template <QType D>
__device__ __forceinline__ void down_nmma_task(NmmaDownShared<D>& sm, const NmmaPassArgs& a, int job, int bx) {
    const auto& dispatch = a.dispatch;
    const auto& source   = a.source;
    const std::uint16_t* h_plane = a.h_plane;
    const std::int32_t* h_emax   = a.h_emax;
    bf16* outputs                = a.outputs;
    const int hidden             = a.hidden;
    const std::int32_t* cpu_flags = a.cpu_flags;
    const auto job_records       = a.job_records;
    const PassPhase phase        = a.phase;
    const int narrow_max         = a.narrow_max;
    namespace nm = ggml::nmma;
    // Only h_plane and h_emax are read after the wait (the job's last gate/up CTA encoded its h); the weights do
    // not depend on it, so their copies start first.
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    if (job < *dispatch.job_count && (cpu_flags == nullptr || cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(phase, source, expert, job_records, job);
    }
    if (record != nullptr) {
        first = dispatch.offsets[expert];
        count = dispatch.offsets[expert + 1] - first;
        if (count > narrow_max) { record = nullptr; }
    }
    const int lane = static_cast<int>(threadIdx.x) % 32, warp = static_cast<int>(threadIdx.x) / 32;
    const int r0   = bx * kNmmaDownRows + 8 * warp;
    const std::uint8_t* down = record + 2 * static_cast<std::size_t>(moe::kIntermediate) * source.ggml.gate_row_bytes;
    const auto row_of = [&](int r) { return down + static_cast<std::size_t>(r0 + r) * source.ggml.down_row_bytes; };
    std::uint8_t* ring = sm.ring[warp];
    if (record != nullptr) { NmmaDownRing<D>::issue_all(ring, row_of, lane); }
    pdl::wait_for_dependencies();
    if (record == nullptr) { return; }
    // The job's encoded h (its last gate/up CTA wrote each entry's limbs over its h row) and exponents.
    for (int i = static_cast<int>(threadIdx.x); i < count * (moe::wide::kA16HRowBytes / 16); i += kNmmaThreads) {
        const int c = i / (moe::wide::kA16HRowBytes / 16), k = i % (moe::wide::kA16HRowBytes / 16);
        ggml::tile::cp_async16(reinterpret_cast<std::uint8_t*>(sm.limbs[0] + c * kGgmlHSlices * ggml::fast::kSliceWords) + 16 * k,
                               reinterpret_cast<const std::uint8_t*>(h_plane + static_cast<std::size_t>(first + c) *
                                                                                   moe::wide::kA16HRowElements) + 16 * k,
                               16);
    }
    ggml::tile::cp_async_commit();
    if (static_cast<int>(threadIdx.x) < count) { sm.emax[threadIdx.x] = __ldcg(h_emax + first + threadIdx.x); }
    ggml::tile::cp_async_wait<0>();
    __syncthreads(); // the limbs, and every lane's weight copies
    const ggml::fast::DeviceCodebook<D> no_codebook{}; // the down formats read no table
    int base         = 0;
    const bool by_base = NmmaDownRing<D>::template row_base<D>(ring, row_of, lane, base);
    for (int pass = 0; pass < count; pass += 4) {
        const int n    = min(4, count - pass);
        const auto run = [&](auto width) {
            constexpr int C = decltype(width)::value;
            const auto xpair = [&](int c, int slice, int word) {
                const std::int32_t* p = sm.limbs[0] + (pass + min(c, n - 1)) * kGgmlHSlices * ggml::fast::kSliceWords + slice * ggml::fast::kSliceWords + word;
                if constexpr (nm::kWordStep<D> == 1) { return *reinterpret_cast<const int2*>(p); }
                return make_int2(p[0], p[nm::kWordStep<D>]);
            };
            nm::Rows<C> acc;
            acc.base = base;
            if (by_base) {
                NmmaDownRing<D>::template resident<D, C, true>(ring, row_of, no_codebook, xpair, lane, acc);
            } else {
                NmmaDownRing<D>::template resident<D, C, false>(ring, row_of, no_codebook, xpair, lane, acc);
            }
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const ggml::I128 r    = nm::combine_limbs(acc.r[c]);
                const std::uint16_t y = ggml::output_bf16(r, sm.emax[pass + min(c, n - 1)], D);
                if (c < n && lane % 4 == 0) {
                    const int column = dispatch.entries[first + pass + c];
                    reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * hidden + r0 + lane / 4] = y;
                }
            }
        };
        if (n == 1) {
            run(std::integral_constant<int, 1>{});
        } else if (n == 2) {
            run(std::integral_constant<int, 2>{});
        } else {
            run(std::integral_constant<int, 4>{});
        }
    }
}

template <QType D>
__global__ void __launch_bounds__(kNmmaThreads) down_nmma_ggml(NmmaPassArgs a) {
    extern __shared__ __align__(16) unsigned char nmma_shared[];
    down_nmma_task<D>(*reinterpret_cast<NmmaDownShared<D>*>(nmma_shared), a, a.job_base + static_cast<int>(blockIdx.y),
                      static_cast<int>(blockIdx.x));
}

// ------------------------------------------------------------------------------- wide route
//
// In a call of more than kMaxColumns columns (prefill), jobs with more than kMaxColumns columns run
// the tensor-core tiles of ops/ggml/ggml_tile.cuh instead (the narrow kernels skip them): the plan
// lists every such job's 64-column tiles in job order (tiles: (job, first column)) and each job's
// first tile index (first_tile, max_jobs + 1 entries), so a pass over jobs [base, base + n) owns
// tiles [first_tile[base], first_tile[base + n]); the workspace's tile list, sized for 64-column
// tiles (moe::wide::max_tiles), holds them. Gate/up CTAs own 32 intermediates (their 32 gate rows
// and 32 up rows) and write BF16 h; the pass's wide h rows are then rewritten in place as limbs (20
// slices of kSliceWords words: kA16HRowBytes), and down CTAs own 64 output rows.

constexpr int kGgmlPlanThreads = 512;
constexpr int kGgmlTileColumns = ggml::tile::kTileColumns; // columns per planned tile (one CTA)
// Threads of a gate/up or down tile CTA: sixteen warps of 16 x 16 outputs (one CTA per SM within 128
// registers), against eight warps of 32 x 16 that needed 253 registers for one CTA per SM. Measured
// on the RTX 5090: see docs/maintainer/qwen3_8-flash-next-design.md §19.3.25.
constexpr int kGgmlTileThreads = 512;
constexpr int kGgmlTileUnits   = ggml::tile::kTileRows / 2; // intermediates per gate/up tile
static_assert(kGgmlTileColumns >= moe::wide::kTileColumns, "the workspace's tile list bounds the plan's tiles");
static_assert(kGgmlHSlices * ggml::fast::kSliceWords * 4 == moe::wide::kA16HRowBytes,
              "a wide entry's h limbs fill its A16 h row exactly");

__global__ void __launch_bounds__(kGgmlPlanThreads)
    ggml_plan_kernel(MoeDispatch dispatch, int max_jobs, std::int32_t* __restrict__ tiles,
                     std::int32_t* __restrict__ first_tile) {
    __shared__ int scan[kGgmlPlanThreads];
    __shared__ int base;
    const int tid  = static_cast<int>(threadIdx.x);
    const int jobs = min(*dispatch.job_count, max_jobs);
    if (tid == 0) { base = 0; }
    __syncthreads();
    for (int chunk = 0; chunk <= max_jobs; chunk += kGgmlPlanThreads) {
        const int job = chunk + tid;
        int n         = 0;
        if (job < jobs) {
            const int expert  = dispatch.jobs[job];
            const int columns = dispatch.offsets[expert + 1] - dispatch.offsets[expert];
            if (columns > moe::kMaxColumns) { n = (columns + kGgmlTileColumns - 1) / kGgmlTileColumns; }
        }
        scan[tid] = n;
        __syncthreads();
        for (int step = 1; step < kGgmlPlanThreads; step <<= 1) {
            const int add = tid >= step ? scan[tid - step] : 0;
            __syncthreads();
            scan[tid] += add;
            __syncthreads();
        }
        const int first = base + scan[tid] - n;
        for (int i = 0; i < n; ++i) {
            tiles[2 * (first + i)]     = job;
            tiles[2 * (first + i) + 1] = i * kGgmlTileColumns;
        }
        if (job <= max_jobs) { first_tile[job] = first; }
        __syncthreads();
        if (tid == kGgmlPlanThreads - 1) { base = first + n; }
        __syncthreads();
    }
}

// The CTA's tile of a pass over jobs [job_base, job_base + jobs): its job, first column and the
// job's expert entries; false when the CTA has no tile.
struct GgmlTile {
    int job, column0, first, count;
};

// Planned tile `tile`: its job, first column and the job's expert entries.
__device__ __forceinline__ GgmlTile ggml_tile_at(const MoeDispatch& dispatch, const std::int32_t* tiles, int tile) {
    GgmlTile out;
    out.job     = tiles[2 * tile];
    out.column0 = tiles[2 * tile + 1];
    const int expert = dispatch.jobs[out.job];
    out.first = dispatch.offsets[expert];
    out.count = dispatch.offsets[expert + 1] - out.first;
    return out;
}

__device__ __forceinline__ bool ggml_tile_of(const MoeDispatch& dispatch, const std::int32_t* tiles,
                                             const std::int32_t* first_tile, int job_base, int jobs, GgmlTile& out) {
    const int tile = first_tile[job_base] + static_cast<int>(blockIdx.y);
    if (tile >= first_tile[job_base + jobs]) { return false; }
    out = ggml_tile_at(dispatch, tiles, tile);
    return out.column0 < out.count;
}

template <QType GU>
__global__ void __launch_bounds__(kGgmlTileThreads, 1)
    gate_up_tile_ggml(const std::uint8_t* __restrict__ x_limbs, const std::int32_t* __restrict__ x_emax,
                      MoeDispatch dispatch, MoeExpertSource source, int top_k,
                      const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ tiles,
                      const std::int32_t* __restrict__ first_tile, int job_base, int jobs, std::uint16_t* __restrict__ h_plane) {
    extern __shared__ __align__(16) unsigned char tile_shared[];
    auto& sm = *reinterpret_cast<ggml::tile::Shared*>(tile_shared);
    __shared__ std::uint16_t up[kGgmlTileUnits][ggml::tile::kTileColumns];
    GgmlTile tl;
    if (!ggml_tile_of(dispatch, tiles, first_tile, job_base, jobs, tl)) { return; }
    const std::uint8_t* record = job_records[tl.job];
    if (record == nullptr) { return; }
    const int i0        = static_cast<int>(blockIdx.x) * kGgmlTileUnits;
    const int row_bytes = static_cast<int>(source.ggml.gate_row_bytes);
    const auto row_of   = [&](int r) {
        return record +
               static_cast<std::size_t>(r < kGgmlTileUnits ? i0 + r : moe::kIntermediate + i0 + r - kGgmlTileUnits) * row_bytes;
    };
    const auto token = [&](int c) { return dispatch.entries[tl.first + min(tl.column0 + c, tl.count - 1)] / top_k; };
    const auto limbs = [&](int c) {
        return reinterpret_cast<const std::int32_t*>(x_limbs + static_cast<std::size_t>(token(c)) * moe::wide::kA16XRowBytes);
    };
    ggml::tile::Accumulators acc;
    const int columns = min(ggml::tile::kTileColumns, tl.count - tl.column0);
    const ggml::tile::WarpTile wt = ggml::tile::warp_tile<kGgmlTileThreads>(columns);
    ggml::tile::compute<GU, kGgmlTileThreads>(sm, row_of, kGgmlXSlices, limbs, 0, columns, acc);
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int mi = 0; mi < ggml::tile::kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < ggml::tile::kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = ggml::tile::out_row(wt, lane, mi, q), c = ggml::tile::out_column(wt, lane, nt, q);
                if (mi < wt.mtiles && nt < wt.ntiles && r >= kGgmlTileUnits) {
                    up[r - kGgmlTileUnits][c] = ggml::output_bf16(acc.r[mi][nt][q], x_emax[token(c)], GU);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int mi = 0; mi < ggml::tile::kMTiles; ++mi) {
#pragma unroll
        for (int nt = 0; nt < ggml::tile::kNTiles; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = ggml::tile::out_row(wt, lane, mi, q), c = ggml::tile::out_column(wt, lane, nt, q);
                if (mi < wt.mtiles && nt < wt.ntiles && r < kGgmlTileUnits && tl.column0 + c < tl.count) {
                    const std::uint16_t y = ggml::output_bf16(acc.r[mi][nt][q], x_emax[token(c)], GU);
                    h_plane[static_cast<std::size_t>(tl.first + tl.column0 + c) * moe::wide::kA16HRowElements + i0 + r] =
                        canon::swiglu_bf16(y, up[r][c]);
                }
            }
        }
    }
}

// The pass's wide entries' h rows rewritten in place from BF16 to limbs: one warp per entry, which
// holds the whole row in registers (five 4-element groups per lane) before it writes.
__global__ void __launch_bounds__(kThreads)
    encode_h_ggml(MoeDispatch dispatch, const std::int32_t* __restrict__ tiles, const std::int32_t* __restrict__ first_tile,
                  int job_base, int jobs, std::uint16_t* __restrict__ h_plane, std::int32_t* __restrict__ h_emax) {
    constexpr int kGroups = moe::kIntermediate / 4 / 32; // per lane
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int warp = static_cast<int>(blockIdx.x) * kWarps + static_cast<int>(threadIdx.x) / 32;
    const int t0 = first_tile[job_base], tasks = (first_tile[job_base + jobs] - t0) * kGgmlTileColumns;
    for (int task = warp; task < tasks; task += static_cast<int>(gridDim.x) * kWarps) {
        const int tile = t0 + task / kGgmlTileColumns;
        const int job = tiles[2 * tile], column = tiles[2 * tile + 1] + task % kGgmlTileColumns;
        const int expert = dispatch.jobs[job];
        const int entry  = dispatch.offsets[expert] + column;
        if (entry >= dispatch.offsets[expert + 1]) { continue; } // warp-uniform
        std::uint16_t* h = h_plane + static_cast<std::size_t>(entry) * moe::wide::kA16HRowElements;
        std::uint16_t v[kGroups][4];
        unsigned key = 0;
#pragma unroll
        for (int i = 0; i < kGroups; ++i) {
            const uint2 w = *reinterpret_cast<const uint2*>(h + 4 * (lane + 32 * i));
            v[i][0] = static_cast<std::uint16_t>(w.x & 0xFFFFU);
            v[i][1] = static_cast<std::uint16_t>(w.x >> 16);
            v[i][2] = static_cast<std::uint16_t>(w.y & 0xFFFFU);
            v[i][3] = static_cast<std::uint16_t>(w.y >> 16);
#pragma unroll
            for (int j = 0; j < 4; ++j) { key = max(key, static_cast<unsigned>(v[i][j] & 0x7FFFU)); }
        }
        key = __reduce_max_sync(0xFFFFFFFFU, key);
        const int e = key >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(key >> 7), 1);
        __syncwarp();
        const auto value = [&](std::uint16_t b) { return e == canon::kA16NonFinite ? 0 : canon::a16_value(b, e); };
        auto* column_words = reinterpret_cast<std::int32_t*>(h);
#pragma unroll
        for (int i = 0; i < kGroups; ++i) {
            (void)ggml::fast::store_group(column_words, lane + 32 * i, value(v[i][0]), value(v[i][1]), value(v[i][2]),
                                          value(v[i][3]));
        }
        if (lane == 0) { h_emax[entry] = e; }
    }
}

// Weight-stationary down tiles: CTA (x, y) owns output rows 64x .. 64x + 63 of the pass's job y
// (jobs without wide tiles exit), decodes that row tile's K = 640 (ten stages) into shared memory
// once and computes every one of the job's column tiles against it, so the decode, the scale pass
// and the stage pipeline's start are paid once per job instead of once per 64 columns.
constexpr int kDownStages = kGgmlHSlices / ggml::tile::kStageSlices;
// The down formats' compact stages (ggml_tile.cuh ResidentStage).
template <QType D>
using GgmlDownShared = ggml::tile::ResidentShared<kDownStages, ggml::tile::ResidentStage<D>>;

template <QType D>
__global__ void __launch_bounds__(kGgmlTileThreads, 1)
    down_resident_ggml(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                       const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ tiles,
                       const std::int32_t* __restrict__ first_tile, int job_base,
                       const std::uint16_t* __restrict__ h_plane, const std::int32_t* __restrict__ h_emax,
                       bf16* __restrict__ outputs) {
    extern __shared__ __align__(16) unsigned char tile_shared[];
    auto& sm      = *reinterpret_cast<GgmlDownShared<D>*>(tile_shared);
    const int job = job_base + static_cast<int>(blockIdx.y);
    const int t0 = first_tile[job], t1 = first_tile[job + 1];
    if (t0 == t1) { return; } // no wide tiles: the narrow kernels computed it
    const std::uint8_t* record = job_records[job];
    if (record == nullptr) { return; }
    const int row0 = static_cast<int>(blockIdx.x) * ggml::tile::kTileRows;
    const std::uint8_t* down = record + 2 * static_cast<std::size_t>(moe::kIntermediate) * source.ggml.gate_row_bytes;
    const auto row_of = [&](int r) { return down + static_cast<std::size_t>(row0 + r) * source.ggml.down_row_bytes; };
    const ggml::tile::Accum mode = ggml::tile::decode_resident<D, kDownStages, kGgmlTileThreads>(sm, row_of);
    const int lane    = threadIdx.x % 32;
    for (int tile = t0; tile < t1; ++tile) {
        const GgmlTile tl = ggml_tile_at(dispatch, tiles, tile);
        const auto entry  = [&](int c) { return tl.first + min(tl.column0 + c, tl.count - 1); };
        const auto limbs  = [&](int c) {
            return reinterpret_cast<const std::int32_t*>(h_plane +
                                                         static_cast<std::size_t>(entry(c)) * moe::wide::kA16HRowElements);
        };
        ggml::tile::Accumulators acc;
        const int columns = min(ggml::tile::kTileColumns, tl.count - tl.column0);
        const ggml::tile::WarpTile wt = ggml::tile::warp_tile<kGgmlTileThreads>(columns);
        ggml::tile::compute_resident<D, kDownStages, kGgmlTileThreads>(sm, mode, limbs, 0, columns, acc);
#pragma unroll
        for (int mi = 0; mi < ggml::tile::kMTiles; ++mi) {
#pragma unroll
            for (int nt = 0; nt < ggml::tile::kNTiles; ++nt) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int c = ggml::tile::out_column(wt, lane, nt, q);
                    if (mi < wt.mtiles && nt < wt.ntiles && tl.column0 + c < tl.count) {
                        const int row = row0 + ggml::tile::out_row(wt, lane, mi, q);
                        reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(dispatch.entries[entry(c)]) * hidden +
                                                                  row] = ggml::output_bf16(acc.r[mi][nt][q], h_emax[entry(c)], D);
                    }
                }
            }
        }
    }
}

// CTAs of the persistent h encoder: four per SM.
int ggml_encode_h_ctas() {
    static const int ctas = [] {
        int device = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        return 4 * sms;
    }();
    return ctas;
}

// Plans the call's wide tiles (once per moe_experts call of more than kMaxColumns columns).
void launch_plan_ggml(const MoeDispatch& dispatch, int max_jobs, const moe::wide::ExpertsWorkspace& layout,
                      cudaStream_t stream) {
    ggml_plan_kernel<<<1, kGgmlPlanThreads, 0, stream>>>(dispatch, max_jobs, layout.tiles, layout.pass_tiles);
    check_launch("ggml plan");
}

// Encodes the call's x for the gate/up kernels and clears its jobs' gate/up CTA counts (once per moe_experts
// call, before its passes).
void launch_encode_x_ggml(const Tensor& x, int max_jobs, const moe::wide::ExpertsWorkspace& layout, cudaStream_t stream) {
    const int columns = static_cast<int>(x.ne[1]);
    ggml_encode_x_kernel<<<static_cast<unsigned>(columns + 1), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data), columns, layout.x_limbs, layout.x_emax, layout.job_ctas, max_jobs);
    check_launch("ggml encode x");
}

// ------------------------------------------------------------------------- ggml-bf16-prefill route
//
// With WideA16::kGgmlBf16, a call of kGgmlStageDecodeMinColumns or more columns runs its passes'
// wide jobs on ops/ggml/ggml_bf16_tile.cuh's stage-decode tiles instead of the exact ones, every wide
// job of a pass at once (the pass's 128-entry tiles planned by ggml_bf16_plan_kernel; a job's last
// tile takes 64 columns when at most 64 of its entries remain). Each CTA copies its rows' raw GGUF
// bytes and decodes them per K stage into shared memory (compute_decode), so a job's weights are
// decoded once per 128 of its entries and the route needs no workspace of its own. Gate/up tiles of
// 128 rows (64 intermediates' gate rows and the same intermediates' up rows) by 128 of the job's
// entries write each entry's BF16 h = swiglu_bf16(bf16(gate), bf16(up)); down tiles of 128 rows by
// 128 entries write its BF16 output. x and h are read as stored (no limbs, no h encoding); every
// product is exact and only FP32 sums round, so an output depends only on its expert, its entry's x
// and the K order, not on the call. Jobs of at most kMaxColumns columns keep the narrow kernels.

constexpr int kBf16PlanThreads = 512;
constexpr int kBf16GateUpPairs = moe::kIntermediate / ggml::tile::kTileRows; // row-tile pairs (gate, up)
constexpr int kBf16DownTiles   = moe::kHidden / ggml::bf16::kRows;
constexpr int kBf16XStages     = moe::kHidden / (ggml::tile::kStageSlices * 32);
constexpr int kBf16HStages     = moe::kIntermediate / (ggml::tile::kStageSlices * 32);
static_assert(moe::kHidden % ggml::bf16::kRows == 0 && moe::kIntermediate % ggml::tile::kTileRows == 0);

// Whether job `job` of the call takes the BF16 tiles: wide, not CPU-served, with a record.
__device__ __forceinline__ bool bf16_job(const MoeDispatch& dispatch, const std::int32_t* cpu_flags,
                                         const std::uint8_t* const* job_records, int job) {
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return false; }
    const int expert = dispatch.jobs[job];
    return dispatch.offsets[expert + 1] - dispatch.offsets[expert] > moe::kMaxColumns && job_records[job] != nullptr;
}

// The 128-entry tiles of the BF16 jobs among [base, base + jobs) (1 <= jobs <= kBf16PlanThreads):
// tiles[2 i] the tile's job, tiles[2 i + 1] its first entry within the job; first_tile[j] (j = 0 ..
// jobs) the first tile of job base + j.
__global__ void __launch_bounds__(kBf16PlanThreads)
    ggml_bf16_plan_kernel(MoeDispatch dispatch, int base, int jobs, const std::int32_t* __restrict__ cpu_flags,
                          const std::uint8_t* const* __restrict__ job_records, std::int32_t* __restrict__ tiles,
                          std::int32_t* __restrict__ first_tile) {
    __shared__ int scan[kBf16PlanThreads];
    const int tid = static_cast<int>(threadIdx.x);
    const int job = base + tid;
    int n         = 0;
    if (tid < jobs && bf16_job(dispatch, cpu_flags, job_records, job)) {
        const int expert  = dispatch.jobs[job];
        const int columns = dispatch.offsets[expert + 1] - dispatch.offsets[expert];
        n = (columns + ggml::bf16::kColumns - 1) / ggml::bf16::kColumns;
    }
    scan[tid] = n;
    __syncthreads();
    for (int step = 1; step < kBf16PlanThreads; step <<= 1) {
        const int add = tid >= step ? scan[tid - step] : 0;
        __syncthreads();
        scan[tid] += add;
        __syncthreads();
    }
    const int first = scan[tid] - n;
    for (int i = 0; i < n; ++i) {
        tiles[2 * (first + i)]     = job;
        tiles[2 * (first + i) + 1] = i * ggml::bf16::kColumns;
    }
    if (tid < jobs) { first_tile[tid] = first; }
    if (tid == jobs - 1) { first_tile[jobs] = scan[tid]; }
}

// B columns of an expert tile: entry first + col0 + c of the job (past `count`: none), read from row
// entries[e] / top_k of x (top_k > 0: the entry's token) or row e of h (top_k == 0), `stride`
// elements apart.
struct GatherColumns {
    const std::uint16_t* base;
    std::size_t stride;
    const std::int32_t* entries;
    int first, col0, count, top_k;

    __device__ __forceinline__ const std::uint16_t* column(int c) const {
        if (col0 + c >= count) { return nullptr; }
        const int e = first + col0 + c;
        return base + static_cast<std::size_t>(top_k > 0 ? entries[e] / top_k : e) * stride;
    }
    __device__ __forceinline__ int element(int e) const { return e; }
    __device__ __forceinline__ const float* slice_sum(int, int) const { return nullptr; }
};

__device__ __forceinline__ std::uint16_t bf16_bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// Gate/up tiles of the pass: CTA (p, y) takes row pair p (intermediates 64 p .. 64 p + 63: its gate
// rows, then its up rows) of the pass's tiles y, y + gridDim.y, ...
template <QType GU>
__global__ void __launch_bounds__(ggml::bf16::kThreads, 1)
    ggml_gate_up_decode(const std::uint16_t* __restrict__ x, MoeDispatch dispatch, MoeExpertSource source, int top_k,
                        const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ tiles,
                        const std::int32_t* __restrict__ first_tile, int jobs, std::uint16_t* __restrict__ h_plane,
                        int kappa_log2) {
    namespace b = ggml::bf16;
    constexpr b::Scaling S = b::kScaling<GU>;
    extern __shared__ __align__(16) unsigned char bf16_shared[];
    auto& sm       = *reinterpret_cast<b::DecodeShared<GU>*>(bf16_shared);
    auto* up       = reinterpret_cast<std::uint16_t*>(&sm.x[0][0][0][0]); // [64][kColumns] after the K loop
    const int pair = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const std::size_t row_bytes = source.ggml.gate_row_bytes;
    const int t1                = first_tile[jobs];
    if (static_cast<int>(blockIdx.y) >= t1) { return; }
    sm.codebook.template load<b::kThreads>(); // published by compute_decode's first barrier
    for (int tile = static_cast<int>(blockIdx.y); tile < t1; tile += static_cast<int>(gridDim.y)) {
        const int job = tiles[2 * tile], col0 = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        const std::uint8_t* record = job_records[job];
        // CTA row r: gate row 64 p + r (r < 64), else up row 64 p + r - 64.
        const auto row_of = [&](int r) {
            const int half = r / ggml::tile::kTileRows;
            return record + (static_cast<std::size_t>(half) * moe::kIntermediate + pair * ggml::tile::kTileRows +
                             r % ggml::tile::kTileRows) * row_bytes;
        };
        const GatherColumns cols{x, static_cast<std::size_t>(moe::kHidden), dispatch.entries, first, col0, count, top_k};
        const auto run = [&](auto width) {
            constexpr int NT = decltype(width)::value;
            b::Accumulators acc;
            b::compute_decode<S, GU, NT>(sm, row_of, kBf16XStages, cols, acc);
            // The up rows (warp row 1) through shared memory to the gate rows' threads (warp row 0).
            if (warp >= 4) {
#pragma unroll
                for (int mi = 0; mi < b::kMTiles; ++mi) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int r = b::out_row(warp, lane, mi, q) - ggml::tile::kTileRows;
                            const int c = b::decode_out_column<NT>(warp, lane, nt, q);
                            up[r * b::kColumns + c] = bf16_bits(ldexpf(acc.r[mi][nt][q], kappa_log2));
                        }
                    }
                }
            }
            __syncthreads();
            if (warp < 4) {
#pragma unroll
                for (int mi = 0; mi < b::kMTiles; ++mi) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int r = b::out_row(warp, lane, mi, q), c = b::decode_out_column<NT>(warp, lane, nt, q);
                            if (col0 + c < count) {
                                const std::uint16_t y = bf16_bits(ldexpf(acc.r[mi][nt][q], kappa_log2));
                                h_plane[static_cast<std::size_t>(first + col0 + c) * moe::wide::kA16HRowElements +
                                        pair * ggml::tile::kTileRows + r] = canon::swiglu_bf16(y, up[r * b::kColumns + c]);
                            }
                        }
                    }
                }
            }
            __syncthreads(); // `up` is read before the next tile's copies refill shared memory
        };
        if (count - col0 > b::kColumns / 2) {
            run(std::integral_constant<int, 4>{});
        } else {
            run(std::integral_constant<int, 2>{});
        }
    }
}

// Down tiles of the pass: CTA (q, y) takes output rows 128 q .. 128 q + 127 of the pass's tiles y,
// y + gridDim.y, ...; each entry's output column is outputs[entries[e]].
template <QType D>
__global__ void __launch_bounds__(ggml::bf16::kThreads, 1)
    ggml_down_decode(MoeDispatch dispatch, MoeExpertSource source, int hidden, const std::uint8_t* const* __restrict__ job_records,
                     const std::int32_t* __restrict__ tiles, const std::int32_t* __restrict__ first_tile, int jobs,
                     const std::uint16_t* __restrict__ h_plane, std::uint16_t* __restrict__ outputs, int kappa_log2) {
    namespace b = ggml::bf16;
    constexpr b::Scaling S = b::kScaling<D>;
    extern __shared__ __align__(16) unsigned char bf16_shared[];
    auto& sm    = *reinterpret_cast<b::DecodeShared<D>*>(bf16_shared);
    const int q = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const std::size_t row_bytes = source.ggml.down_row_bytes;
    const int t1                = first_tile[jobs];
    if (static_cast<int>(blockIdx.y) >= t1) { return; }
    sm.codebook.template load<b::kThreads>(); // published by compute_decode's first barrier
    for (int tile = static_cast<int>(blockIdx.y); tile < t1; tile += static_cast<int>(gridDim.y)) {
        const int job = tiles[2 * tile], col0 = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        const std::uint8_t* down = job_records[job] + 2 * static_cast<std::size_t>(moe::kIntermediate) * source.ggml.gate_row_bytes;
        const auto row_of = [&](int r) { return down + static_cast<std::size_t>(q * b::kRows + r) * row_bytes; };
        const GatherColumns cols{h_plane, static_cast<std::size_t>(moe::wide::kA16HRowElements), dispatch.entries, first, col0,
                                 count, 0};
        const auto run = [&](auto width) {
            constexpr int NT = decltype(width)::value;
            b::Accumulators acc;
            b::compute_decode<S, D, NT>(sm, row_of, kBf16HStages, cols, acc);
#pragma unroll
            for (int mi = 0; mi < b::kMTiles; ++mi) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int c = b::decode_out_column<NT>(warp, lane, nt, k);
                        if (col0 + c < count) {
                            const int row = q * b::kRows + b::out_row(warp, lane, mi, k);
                            outputs[static_cast<std::size_t>(dispatch.entries[first + col0 + c]) * hidden + row] =
                                bf16_bits(ldexpf(acc.r[mi][nt][k], kappa_log2));
                        }
                    }
                }
            }
        };
        if (count - col0 > b::kColumns / 2) {
            run(std::integral_constant<int, 4>{});
        } else {
            run(std::integral_constant<int, 2>{});
        }
    }
}

// SMs of the device: the BF16 expert tiles' grid-stride loops run 2 x SMs CTAs.
int bf16_expert_sms() {
    static const int sms = [] {
        int device = 0, count = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    return sms;
}

// The stage-decode route's wide jobs of a pass (after its narrow kernels): one plan, one gate/up
// launch and one down launch over every wide job of the pass.
template <QType GU, QType D>
void launch_decode_pass_ggml(const std::uint16_t* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                             const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                             const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream) {
    namespace b = ggml::bf16;
    if (jobs > kBf16PlanThreads) { throw std::invalid_argument("offloaded_sparse_moe: a BF16 pass plans at most 512 jobs"); }
    constexpr int kGateUpShared = static_cast<int>(sizeof(b::DecodeShared<GU>));
    constexpr int kDownShared   = static_cast<int>(sizeof(b::DecodeShared<D>));
    static const bool configured = [] {
        return cudaFuncSetAttribute(ggml_gate_up_decode<GU>, cudaFuncAttributeMaxDynamicSharedMemorySize, kGateUpShared) ==
                   cudaSuccess &&
               cudaFuncSetAttribute(ggml_down_decode<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDownShared) ==
                   cudaSuccess;
    }();
    if (!configured) { throw std::runtime_error("offloaded_sparse_moe: the stage-decode tiles' shared memory exceeds the device's"); }
    auto* h_plane = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    ggml_bf16_plan_kernel<<<1, kBf16PlanThreads, 0, stream>>>(dispatch, base, jobs, flags, job_records, layout.tiles,
                                                               layout.pass_tiles);
    check_launch("ggml bf16 plan");
    // Grid-stride loops over the pass's tiles in 2 x SMs CTAs (two of one CTA per SM).
    const int sms = bf16_expert_sms();
    ggml_gate_up_decode<GU><<<dim3(kBf16GateUpPairs, (2 * sms + kBf16GateUpPairs - 1) / kBf16GateUpPairs), b::kThreads,
                              kGateUpShared, stream>>>(x, dispatch, source, top_k, job_records, layout.tiles,
                                                       layout.pass_tiles, jobs, h_plane, ggml::kappa_log2(GU));
    check_launch("ggml stage-decode gate/up tiles");
    ggml_down_decode<D><<<dim3(kBf16DownTiles, (2 * sms + kBf16DownTiles - 1) / kBf16DownTiles), b::kThreads, kDownShared,
                          stream>>>(dispatch, source, static_cast<int>(moe::kHidden), job_records, layout.tiles,
                                    layout.pass_tiles, jobs, h_plane, reinterpret_cast<std::uint16_t*>(outputs),
                                    ggml::kappa_log2(D));
    check_launch("ggml stage-decode down tiles");
}

// `entries` is the call's entry count when the call is wide (planned by launch_plan_ggml), else 0;
// `columns` its column count (T).
template <QType GU, QType D>
void launch_expert_pass_ggml(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                             const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                             const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                             PassPhase phase, int entries, int columns, PassPart part) {
    auto* h_plane        = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    const int narrow_max = entries > 0 ? moe::kMaxColumns : std::numeric_limits<int>::max();
    static const bool configured_narrow = [] {
        const auto shared_carveout = [](auto kernel, std::size_t bytes) {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared) ==
                       cudaSuccess &&
                   cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)) ==
                       cudaSuccess;
        };
        return shared_carveout(gate_up_nmma_ggml<GU>, sizeof(NmmaGateUpShared<GU>)) &&
               shared_carveout(down_nmma_ggml<D>, nmma_down_shared_bytes<D>(moe::kMaxColumns));
    }();
    if (!configured_narrow) {
        throw std::runtime_error("offloaded_sparse_moe: the ggml narrow kernels' shared memory exceeds the device's");
    }
    NmmaPassArgs args{};
    args.x_limbs     = layout.x_limbs;
    args.x_emax      = layout.x_emax;
    args.dispatch    = dispatch;
    args.source      = source;
    args.top_k       = top_k;
    args.job_records = job_records;
    args.cpu_flags   = flags;
    args.job_base    = base;
    args.jobs        = jobs;
    args.h_plane     = h_plane;
    args.h_emax      = layout.h_emax;
    args.job_ctas    = layout.job_ctas;
    args.outputs     = outputs;
    args.hidden      = static_cast<int>(moe::kHidden);
    args.phase       = phase;
    args.narrow_max  = narrow_max;
    if (part != PassPart::Down) {
        CUDA_CHECK(pdl::launch_with(pdl::Dependency::Serialized,
                                    {dim3(kNmmaGateUpCtas, jobs), dim3(kNmmaThreads), sizeof(NmmaGateUpShared<GU>), stream},
                                    gate_up_nmma_ggml<GU>, args));
        check_launch("ggml gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    CUDA_CHECK(pdl::launch_with(part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic,
                                {dim3(moe::kHidden / kNmmaDownRows, jobs), dim3(kNmmaThreads), nmma_down_shared_bytes<D>(columns),
                                 stream},
                                down_nmma_ggml<D>, args));
    check_launch("ggml down");
    if (entries == 0) { return; }
    if (source.wide_a16 == moe::WideA16::kGgmlBf16 && columns >= moe::wide::kGgmlStageDecodeMinColumns) {
        launch_decode_pass_ggml<GU, D>(reinterpret_cast<const std::uint16_t*>(x), dispatch, source, top_k, job_records, flags,
                                       base, jobs, layout, outputs, stream);
        return;
    }
    // At most one partial tile per job plus one per 64 entries; CTAs past the pass's tiles exit.
    const unsigned tile_bound = static_cast<unsigned>(jobs + (entries + kGgmlTileColumns - 1) / kGgmlTileColumns);
    constexpr int kTileShared = static_cast<int>(sizeof(ggml::tile::Shared));
    constexpr int kDownShared = static_cast<int>(sizeof(GgmlDownShared<D>));
    static const bool configured = [] {
        return cudaFuncSetAttribute(gate_up_tile_ggml<GU>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTileShared) ==
                   cudaSuccess &&
               cudaFuncSetAttribute(down_resident_ggml<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDownShared) ==
                   cudaSuccess;
    }();
    if (!configured) { throw std::runtime_error("offloaded_sparse_moe: the ggml tiles' shared memory exceeds the device's"); }
    gate_up_tile_ggml<GU><<<dim3(moe::kIntermediate / kGgmlTileUnits, tile_bound), kGgmlTileThreads, kTileShared, stream>>>(
        layout.x_limbs, layout.x_emax, dispatch, source, top_k, job_records, layout.tiles, layout.pass_tiles, base, jobs,
        h_plane);
    check_launch("ggml gate/up tiles");
    encode_h_ggml<<<ggml_encode_h_ctas(), kThreads, 0, stream>>>(dispatch, layout.tiles, layout.pass_tiles, base,
                                                                         jobs, h_plane, layout.h_emax);
    check_launch("ggml encode h");
    down_resident_ggml<D><<<dim3(moe::kHidden / ggml::tile::kTileRows, jobs), kGgmlTileThreads, kDownShared, stream>>>(
        dispatch, source, static_cast<int>(moe::kHidden), job_records, layout.tiles, layout.pass_tiles, base, h_plane,
        layout.h_emax, outputs);
    check_launch("ggml down tiles");
}

// The (gate/up, down) pairs of the GSQ-RCO builds and their siblings.
void launch_pass_ggml(const bf16* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                      const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                      const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream, PassPhase phase,
                      int entries, int columns, PassPart part) {
    // The wide tiles run every job of the pass: a forked call (Staged/Resident phases, gate/up and down
    // parts) must not have wide entries, which forked()'s column bound guarantees today.
    if (entries > 0 && (phase != PassPhase::All || part != PassPart::Both)) {
        throw std::logic_error("offloaded_sparse_moe: the GGUF wide tiles do not split by pass phase");
    }
    const QType gu = source.ggml.gate_up_format, d = source.ggml.down_format;
#define INFERNIX_GGML_PAIR(G, DD)                                                                                       \
    if (gu == G && d == DD) {                                                                                          \
        launch_expert_pass_ggml<G, DD>(x, dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs,   \
                                       stream, phase, entries, columns, part);                                        \
        return;                                                                                                        \
    }
    INFERNIX_GGML_PAIR(QType::GGML_IQ2_S, QType::GGML_IQ4_NL)
    INFERNIX_GGML_PAIR(QType::GGML_IQ2_S, QType::GGML_Q2_0)
    INFERNIX_GGML_PAIR(QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL)
    INFERNIX_GGML_PAIR(QType::GGML_IQ3_XXS, QType::GGML_Q2_0)
    INFERNIX_GGML_PAIR(QType::GGML_IQ3_S, QType::GGML_IQ4_NL)
    INFERNIX_GGML_PAIR(QType::GGML_IQ3_S, QType::GGML_Q2_0)
    INFERNIX_GGML_PAIR(QType::GGML_IQ4_XS, QType::GGML_IQ4_NL)
    INFERNIX_GGML_PAIR(QType::GGML_IQ4_XS, QType::GGML_Q2_0)
#undef INFERNIX_GGML_PAIR
    throw std::invalid_argument("offloaded_sparse_moe: unsupported ggml expert format pair");
}
