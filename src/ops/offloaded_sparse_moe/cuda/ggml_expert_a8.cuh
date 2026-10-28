// GGUF (ggml) expert banks on the 8-bit activation route (`--a8-activations`, MoeGgmlBank::a8; LOSSY,
// opt-in; docs/maintainer/qwen3_8-flash-next-design.md §19.3.31), included by moe_layer.cu after
// ggml_expert.cuh. The arithmetic is ops/ggml/ggml_a8.h's: x and h are quantized per 32-element slice to
// int8 codes with an FP32 scale, each slice's dot with the weight codes is an exact int32 (dp4a), and a row's
// slice terms add into an FP32 sum in a8::row_order, which the CPU route (cpu/ggml_expert.h's a8 functions)
// follows, so a CPU-served miss returns these kernels' bits.
//
// A call of at most kA8MaxCallColumns columns (decode, verification, up to eight lanes) runs every job on
// these kernels, whatever its column count (passes of kA8Columns). A wider call (prefill) keeps the exact and
// BF16 routes of ggml_expert.cuh, or with WideA16::kA8 stays on 8-bit activations: its jobs of at most
// moe::kMaxColumns columns on these kernels (and the CPU route's 8-bit functions for its CPU-served misses),
// its wider jobs on the wide tiles below (ops/ggml/ggml_a8_tile.cuh's arithmetic: int8 tensor cores, a
// different rounding and order of the FP32 sums, so a wide job's bits are not these kernels').
// - ggml_a8_encode_x_kernel quantizes every token's x once per call into the x limb rows (a8::column_words of
//   kHidden: kA8XRowBytes of each kA16XRowBytes row).
// - A gate/up CTA owns kGgmlGateUpUnits intermediates, one per warp: lanes 0-15 its gate row and lanes 16-31
//   its up row, slices part, part + 16, ... (a8::kGateUpPartials partials); a butterfly over the 16 lanes
//   adds them, one shuffle pairs gate and up for SwiGLU, and lane 0 writes BF16 h.
// - A down CTA owns kGgmlDownRows output rows, four lanes per row (a8::kDownPartials); per pass its threads
//   quantize the pass's columns' h into shared memory, one slice each.

namespace a8  = ggml::a8;
namespace a8w = ggml::a8::wide;

constexpr int kA8Columns        = 8;   // columns per pass of either kernel
constexpr int kA8MaxCallColumns = 128; // the widest call the narrow kernels take whole (8 lanes of 16)
constexpr int kA8XRowBytes      = static_cast<int>(a8::column_words(moe::kHidden, false)) * 4; // 2,880
static_assert(kA8XRowBytes <= moe::wide::kA16XRowBytes && kA8XRowBytes % 16 == 0,
              "a call's x codes fit the x limb rows");
static_assert(kGgmlXSlices % a8::kGateUpPartials == 0 && kGgmlHSlices % a8::kDownPartials == 0,
              "every partial takes the same number of slices");
static_assert(kGgmlDownLanes == a8::kDownPartials, "a down row's lanes are its partials");

// Whether a call takes the 8-bit route (moe_experts, the CPU plan and launch_pass decide it identically): every
// call of at most kA8MaxCallColumns columns, and with WideA16::kA8 every wider one.
__host__ __device__ __forceinline__ bool a8_call(const MoeExpertSource& source, int columns) {
    return source.ggml.a8 && (columns <= kA8MaxCallColumns || source.wide_a16 == moe::WideA16::kA8);
}

// Whether an 8-bit call runs its wide jobs (more than moe::kMaxColumns columns) on the wide tiles.
__host__ __device__ __forceinline__ bool a8_wide_call(const MoeExpertSource& source, int columns) {
    return a8_call(source, columns) && columns > kA8MaxCallColumns;
}

// The 8-bit codes, scale and code sum of 32 BF16 values at `v` (16-byte aligned): codes into four words
// of q[0..8) per half (elements 4w + j in byte j of q[w]).
struct A8SliceOut {
    int q[8];
    float d;
    int sum;
};

__device__ __forceinline__ A8SliceOut a8_quantize_slice(const std::uint16_t* __restrict__ v) {
    uint4 raw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { raw[i] = reinterpret_cast<const uint4*>(v)[i]; }
    unsigned key = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned w[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
#pragma unroll
        for (int k = 0; k < 4; ++k) { key = max(key, max(w[k] & 0x7FFFU, (w[k] >> 16) & 0x7FFFU)); }
    }
    const a8::SliceScale sc = a8::slice_scale(key);
    A8SliceOut o;
    o.d   = sc.d;
    o.sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned w[4] = {raw[i].x, raw[i].y, raw[i].z, raw[i].w};
#pragma unroll
        for (int k = 0; k < 2; ++k) { // word 2i + k: elements 8i + 4k .. 8i + 4k + 3
            const int c0 = a8::code(static_cast<std::uint16_t>(w[2 * k] & 0xFFFFU), sc.inv);
            const int c1 = a8::code(static_cast<std::uint16_t>(w[2 * k] >> 16), sc.inv);
            const int c2 = a8::code(static_cast<std::uint16_t>(w[2 * k + 1] & 0xFFFFU), sc.inv);
            const int c3 = a8::code(static_cast<std::uint16_t>(w[2 * k + 1] >> 16), sc.inv);
            o.q[2 * i + k] = ggml::fast::pack4(c0, c1, c2, c3);
            o.sum += c0 + c1 + c2 + c3;
        }
    }
    return o;
}

// One CTA per column; thread s quantizes slice s.
constexpr int kA8EncodeThreads = (kGgmlXSlices + 31) / 32 * 32;
__global__ void __launch_bounds__(kA8EncodeThreads)
    ggml_a8_encode_x_kernel(const bf16* __restrict__ x, std::uint8_t* __restrict__ x_rows) {
    const int t = blockIdx.x, s = threadIdx.x;
    if (s >= kGgmlXSlices) { return; }
    const auto* v = reinterpret_cast<const std::uint16_t*>(x) + static_cast<std::size_t>(t) * moe::kHidden + 32 * s;
    const A8SliceOut o = a8_quantize_slice(v);
    auto* row          = reinterpret_cast<std::int32_t*>(x_rows + static_cast<std::size_t>(t) * moe::wide::kA16XRowBytes);
    auto* codes        = reinterpret_cast<int4*>(row + s * a8::kSliceWords);
    codes[0]           = make_int4(o.q[0], o.q[1], o.q[2], o.q[3]);
    codes[1]           = make_int4(o.q[4], o.q[5], o.q[6], o.q[7]);
    reinterpret_cast<float*>(row + a8::scales_offset(moe::kHidden))[s] = o.d;
}

// The dot of a decoded weight slice with one column's slice codes, sub-scales applied (exact int32).
__device__ __forceinline__ int a8_slice_dot(const ggml::fast::Slice& w, const int4& x0, const int4& x1) {
    const int p0 = __dp4a(w.q[3], x0.w, __dp4a(w.q[2], x0.z, __dp4a(w.q[1], x0.y, __dp4a(w.q[0], x0.x, 0))));
    const int p1 = __dp4a(w.q[7], x1.w, __dp4a(w.q[6], x1.z, __dp4a(w.q[5], x1.y, __dp4a(w.q[4], x1.x, 0))));
    return w.sub0 * p0 + w.sub1 * p1;
}

// Jobs of more than narrow_max columns are the wide tiles' (an 8-bit call of more than kA8MaxCallColumns
// columns), and their CTAs exit.
template <QType GU>
__global__ void __launch_bounds__(kGgmlThreads)
    gate_up_kernel_ggml_a8(const std::uint8_t* __restrict__ x_rows, MoeDispatch dispatch, MoeExpertSource source,
                           int top_k, const std::uint8_t* const* __restrict__ job_records,
                           const std::int32_t* __restrict__ cpu_flags, int job_base, std::uint16_t* __restrict__ h_plane,
                           PassPhase phase, int narrow_max) {
    constexpr int C = kA8Columns;
    const int job   = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (count > narrow_max) { return; } // a wide job: the wide tiles compute it
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int part = lane % a8::kGateUpPartials, up = lane / a8::kGateUpPartials;
    const int i    = blockIdx.x * kGgmlGateUpUnits + warp; // intermediate
    const int row_bytes = static_cast<int>(source.ggml.gate_row_bytes);
    const std::uint8_t* row = record + (static_cast<std::size_t>(up) * moe::kIntermediate + i) * row_bytes;
    pdl::trigger_dependents();
    for (int pass = 0; pass < count; pass += C) {
        const int n = min(C, count - pass);
        const std::int32_t* xs[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const int token = c < n ? dispatch.entries[first + pass + c] / top_k : 0;
            xs[c] = reinterpret_cast<const std::int32_t*>(x_rows + static_cast<std::size_t>(token) * moe::wide::kA16XRowBytes);
        }
        float acc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { acc[c] = 0.0F; }
        for (int s = part; s < kGgmlXSlices; s += a8::kGateUpPartials) {
            const ggml::fast::Slice w = ggml::fast::slice<GU>(row, s);
            const float wd            = a8::f16_to_f32(w.d);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) { // uniform across the warp
                    const int4* q  = reinterpret_cast<const int4*>(xs[c] + s * a8::kSliceWords);
                    const float xd = __ldg(reinterpret_cast<const float*>(xs[c] + a8::scales_offset(moe::kHidden)) + s);
                    acc[c]         = a8::add_term(acc[c], a8_slice_dot(w, __ldg(q), __ldg(q + 1)), xd, wd);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < n) {
#pragma unroll
                for (int offset = a8::kGateUpPartials / 2; offset >= 1; offset >>= 1) {
                    acc[c] = canon::add_rn(acc[c], __shfl_xor_sync(0xFFFFFFFFU, acc[c], offset));
                }
                const std::uint16_t y     = canon::f32_to_bf16_rn(a8::output_f32(acc[c], ggml::kappa_log2(GU)));
                const std::uint16_t other = static_cast<std::uint16_t>(__shfl_xor_sync(0xFFFFFFFFU, y, 16));
                if (lane == 0) {
                    h_plane[static_cast<std::size_t>(first + pass + c) * moe::wide::kA16HRowElements + i] =
                        canon::swiglu_bf16(y, other);
                }
            }
        }
    }
}

template <QType D>
__global__ void __launch_bounds__(kGgmlThreads)
    down_kernel_ggml_a8(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                        const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                        int job_base, const std::uint16_t* __restrict__ h_plane, bf16* __restrict__ outputs,
                        PassPhase phase, int narrow_max) {
    constexpr int C = kA8Columns;
    __shared__ __align__(16) std::int32_t codes[C][kGgmlHSlices * a8::kSliceWords];
    __shared__ float scales[C][kGgmlHSlices];
    const int job = job_base + static_cast<int>(blockIdx.y);
    // A programmatic dependent of the gate/up pass (down_kernel's PDL contract): only h_plane is read
    // after the wait, and every CTA waits.
    int expert = 0, first = 0, count = 0;
    const std::uint8_t* record = nullptr;
    if (job < *dispatch.job_count && (cpu_flags == nullptr || cpu_flags[job] == 0)) {
        expert = dispatch.jobs[job];
        record = pass_record(phase, source, expert, job_records, job);
    }
    if (record != nullptr) {
        first = dispatch.offsets[expert];
        count = dispatch.offsets[expert + 1] - first;
    }
    pdl::wait_for_dependencies();
    if (record == nullptr || count > narrow_max) { return; }
    const int part = threadIdx.x % kGgmlDownLanes;
    const int row  = blockIdx.x * kGgmlDownRows + threadIdx.x / kGgmlDownLanes;
    const int row_bytes = static_cast<int>(source.ggml.down_row_bytes);
    const std::uint8_t* w = record + 2 * static_cast<std::size_t>(moe::kIntermediate) * source.ggml.gate_row_bytes +
                            static_cast<std::size_t>(row) * row_bytes;
    for (int pass = 0; pass < count; pass += C) {
        const int n = min(C, count - pass);
        for (int item = threadIdx.x; item < n * kGgmlHSlices; item += kGgmlThreads) {
            const int c = item / kGgmlHSlices, s = item % kGgmlHSlices;
            const std::uint16_t* h =
                h_plane + static_cast<std::size_t>(first + pass + c) * moe::wide::kA16HRowElements + 32 * s;
            const A8SliceOut o = a8_quantize_slice(h);
            auto* dst          = reinterpret_cast<int4*>(&codes[c][s * a8::kSliceWords]);
            dst[0]             = make_int4(o.q[0], o.q[1], o.q[2], o.q[3]);
            dst[1]             = make_int4(o.q[4], o.q[5], o.q[6], o.q[7]);
            scales[c][s]       = o.d;
        }
        __syncthreads();
        float acc[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { acc[c] = 0.0F; }
        for (int s = part; s < kGgmlHSlices; s += a8::kDownPartials) {
            const ggml::fast::Slice ws = ggml::fast::slice<D>(w, s);
            const float wd             = a8::f16_to_f32(ws.d);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) {
                    const int4* q = reinterpret_cast<const int4*>(&codes[c][s * a8::kSliceWords]);
                    acc[c]        = a8::add_term(acc[c], a8_slice_dot(ws, q[0], q[1]), scales[c][s], wd);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < n) {
#pragma unroll
                for (int offset = a8::kDownPartials / 2; offset >= 1; offset >>= 1) {
                    acc[c] = canon::add_rn(acc[c], __shfl_xor_sync(0xFFFFFFFFU, acc[c], offset));
                }
                if (part == 0) {
                    const int column = dispatch.entries[first + pass + c];
                    reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * hidden + row] =
                        canon::f32_to_bf16_rn(a8::output_f32(acc[c], ggml::kappa_log2(D)));
                }
            }
        }
        __syncthreads(); // every thread has read the codes before the next pass rewrites them
    }
}

// ----------------------------------------------------------------------- wide jobs (WideA16::kA8)
//
// In an 8-bit call of more than kA8MaxCallColumns columns, the jobs of more than moe::kMaxColumns columns run
// ops/ggml/ggml_a8_tile.cuh's stage-decode tiles (compute_decode) over the pass's 128-entry tiles
// (ggml_bf16_plan_kernel's plan), the BF16 stage-decode route's grid and layout with int8 operands:
// - gate/up: CTA (p, y) takes intermediates 64 p .. 64 p + 63 (their gate rows, then their up rows: 128 rows)
//   of the tiles y, y + gridDim.y, ...; B is every entry's token's x codes (the call's encoded rows). After the
//   K loop the up rows' BF16 outputs go through shared memory to the gate rows' threads, which form
//   h = swiglu_bf16(gate, up) there, and then each thread quantizes one (entry, 32-intermediate slice) of h with
//   ggml_a8.h's codec straight into the entry's h row (codes at byte 32 s, slice scales from byte
//   moe::kIntermediate): h's codes are those the narrow down kernels would form from the same BF16 h.
// - down: CTA (q, y) takes output rows 128 q .. 128 q + 127 of the same tiles, B the entries' h codes.
// A wide entry's output depends only on its expert's record and its token's x (ggml_a8_tile.cuh): not on the
// job's column count, the other columns or the tile.

constexpr int kA8WidePairs     = moe::kIntermediate / 64;                 // gate/up CTAs per tile: 10
constexpr int kA8WideDownTiles = moe::kHidden / a8w::kDecodeRows;          // down CTAs per tile: 20
constexpr int kA8WideXStages   = moe::kHidden / (32 * a8w::kStageSlices);  // 40
constexpr int kA8WideHStages   = moe::kIntermediate / (32 * a8w::kStageSlices); // 10
static_assert(ggml::bf16::kColumns == a8w::kColumns, "the plan's 128-entry tiles are the 8-bit tiles' columns");
static_assert(a8w::kDecodeRows == 128 && moe::kIntermediate % 64 == 0 && moe::kHidden % a8w::kDecodeRows == 0);
static_assert(moe::kIntermediate + 4 * kGgmlHSlices <= moe::wide::kA16HRowBytes, "a wide entry's h codes fit its h row");

// A wide job's B columns: entry first + col0 + c of the job (none past `count`) reads row entries[e] / top_k of
// `base` (top_k > 0: x's codes by token) or row e (top_k == 0: h's codes by entry), rows `row_bytes` apart; a
// row holds its codes, then from byte `scales` its slice scales.
struct A8GatherColumns {
    const std::uint8_t* base;
    std::size_t row_bytes;
    const std::int32_t* entries;
    int first, col0, count, top_k, scales;

    __device__ __forceinline__ const std::uint8_t* column(int c) const {
        if (col0 + c >= count) { return nullptr; }
        const int e = first + col0 + c;
        return base + static_cast<std::size_t>(top_k > 0 ? entries[e] / top_k : e) * row_bytes;
    }
};

template <QType GU>
__global__ void __launch_bounds__(a8w::kThreads, 1)
    ggml_gate_up_a8_wide(const std::uint8_t* __restrict__ x_rows, MoeDispatch dispatch, MoeExpertSource source, int top_k,
                         const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ tiles,
                         const std::int32_t* __restrict__ first_tile, int jobs, std::uint8_t* __restrict__ h_rows,
                         int kappa_log2) {
    extern __shared__ __align__(16) unsigned char a8w_shared[];
    auto& sm   = *reinterpret_cast<a8w::DecodeShared<GU>*>(a8w_shared);
    auto* hbuf = reinterpret_cast<std::uint16_t*>(&sm.x[0][0][0][0]); // [64][kColumns] BF16 after the K loop
    static_assert(64 * a8w::kColumns * 2 <= sizeof(sm.x), "the up and h values fit the columns' ring");
    const int pair = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
    const std::size_t row_bytes = source.ggml.gate_row_bytes;
    const int t1                = first_tile[jobs];
    if (static_cast<int>(blockIdx.y) >= t1) { return; }
    sm.codebook.template load<a8w::kThreads>(); // published by compute_decode's first barrier
    for (int tile = static_cast<int>(blockIdx.y); tile < t1; tile += static_cast<int>(gridDim.y)) {
        const int job = tiles[2 * tile], col0 = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        const std::uint8_t* record = job_records[job];
        // CTA row r: gate row 64 p + r (r < 64), else up row 64 p + r - 64.
        const auto row_of = [&](int r) {
            const int half = r / 64;
            return record + (static_cast<std::size_t>(half) * moe::kIntermediate + pair * 64 + r % 64) * row_bytes;
        };
        const A8GatherColumns cols{x_rows, static_cast<std::size_t>(moe::wide::kA16XRowBytes), dispatch.entries, first, col0,
                                   count, top_k, static_cast<int>(a8::scales_offset(moe::kHidden)) * 4};
        const auto run = [&](auto width) {
            constexpr int NT    = decltype(width)::value;
            constexpr int kCols = (a8w::kColumns / a8w::kWarpColumns) * 8 * NT;
            a8w::Accumulators acc;
            a8w::compute_decode<GU, NT>(sm, row_of, kA8WideXStages, cols, acc);
            // The up rows (warp row 1) into hbuf, then h = swiglu_bf16(gate, up) by the gate rows' threads in place.
            if (warp >= 4) {
#pragma unroll
                for (int mi = 0; mi < a8w::kMTiles; ++mi) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int r = a8w::out_row(warp, lane, mi, q) - 64, c = a8w::decode_out_column<NT>(warp, lane, nt, q);
                            hbuf[r * a8w::kColumns + c] = canon::f32_to_bf16_rn(a8::output_f32(acc.r[mi][nt][q], kappa_log2));
                        }
                    }
                }
            }
            __syncthreads();
            if (warp < 4) {
#pragma unroll
                for (int mi = 0; mi < a8w::kMTiles; ++mi) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int r = a8w::out_row(warp, lane, mi, q), c = a8w::decode_out_column<NT>(warp, lane, nt, q);
                            const std::uint16_t gate = canon::f32_to_bf16_rn(a8::output_f32(acc.r[mi][nt][q], kappa_log2));
                            hbuf[r * a8w::kColumns + c] = canon::swiglu_bf16(gate, hbuf[r * a8w::kColumns + c]);
                        }
                    }
                }
            }
            __syncthreads();
            // Thread (s, c) quantizes slice s (intermediates 64 p + 32 s ..) of column c's h into its entry's h row.
            if (tid < 2 * kCols) {
                const int c = tid % kCols, s = tid / kCols;
                if (col0 + c < count) {
                    std::uint16_t v[32];
                    unsigned key = 0;
#pragma unroll
                    for (int e = 0; e < 32; ++e) {
                        v[e] = hbuf[(32 * s + e) * a8w::kColumns + c];
                        key  = max(key, static_cast<unsigned>(v[e] & 0x7FFFU));
                    }
                    const a8::SliceScale sc = a8::slice_scale(key);
                    int words[8];
#pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        words[w] = ggml::fast::pack4(a8::code(v[4 * w], sc.inv), a8::code(v[4 * w + 1], sc.inv),
                                                     a8::code(v[4 * w + 2], sc.inv), a8::code(v[4 * w + 3], sc.inv));
                    }
                    std::uint8_t* row = h_rows + static_cast<std::size_t>(first + col0 + c) * moe::wide::kA16HRowBytes;
                    auto* codes       = reinterpret_cast<int4*>(row + (2 * pair + s) * 32);
                    codes[0]          = make_int4(words[0], words[1], words[2], words[3]);
                    codes[1]          = make_int4(words[4], words[5], words[6], words[7]);
                    reinterpret_cast<float*>(row + moe::kIntermediate)[2 * pair + s] = sc.d;
                }
            }
            __syncthreads(); // hbuf is read before the next tile's copies refill shared memory
        };
        if (count - col0 > a8w::kColumns / 2) {
            run(std::integral_constant<int, 4>{});
        } else {
            run(std::integral_constant<int, 2>{});
        }
    }
}

template <QType D>
__global__ void __launch_bounds__(a8w::kThreads, 1)
    ggml_down_a8_wide(MoeDispatch dispatch, MoeExpertSource source, int hidden, const std::uint8_t* const* __restrict__ job_records,
                      const std::int32_t* __restrict__ tiles, const std::int32_t* __restrict__ first_tile, int jobs,
                      const std::uint8_t* __restrict__ h_rows, std::uint16_t* __restrict__ outputs, int kappa_log2) {
    extern __shared__ __align__(16) unsigned char a8w_shared[];
    auto& sm    = *reinterpret_cast<a8w::DecodeShared<D>*>(a8w_shared);
    const int q = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const std::size_t row_bytes = source.ggml.down_row_bytes;
    const int t1                = first_tile[jobs];
    if (static_cast<int>(blockIdx.y) >= t1) { return; }
    sm.codebook.template load<a8w::kThreads>(); // published by compute_decode's first barrier
    for (int tile = static_cast<int>(blockIdx.y); tile < t1; tile += static_cast<int>(gridDim.y)) {
        const int job = tiles[2 * tile], col0 = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        const std::uint8_t* down = job_records[job] + 2 * static_cast<std::size_t>(moe::kIntermediate) * source.ggml.gate_row_bytes;
        const auto row_of = [&](int r) { return down + static_cast<std::size_t>(q * a8w::kDecodeRows + r) * row_bytes; };
        const A8GatherColumns cols{h_rows, static_cast<std::size_t>(moe::wide::kA16HRowBytes), dispatch.entries, first, col0,
                                   count, 0, moe::kIntermediate};
        const auto run = [&](auto width) {
            constexpr int NT = decltype(width)::value;
            a8w::Accumulators acc;
            a8w::compute_decode<D, NT>(sm, row_of, kA8WideHStages, cols, acc);
#pragma unroll
            for (int mi = 0; mi < a8w::kMTiles; ++mi) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const int c = a8w::decode_out_column<NT>(warp, lane, nt, k);
                        if (col0 + c < count) {
                            const int row = q * a8w::kDecodeRows + a8w::out_row(warp, lane, mi, k);
                            outputs[static_cast<std::size_t>(dispatch.entries[first + col0 + c]) * hidden + row] =
                                canon::f32_to_bf16_rn(a8::output_f32(acc.r[mi][nt][k], kappa_log2));
                        }
                    }
                }
            }
        };
        if (count - col0 > a8w::kColumns / 2) {
            run(std::integral_constant<int, 4>{});
        } else {
            run(std::integral_constant<int, 2>{});
        }
    }
}

// The wide jobs of a pass (after its narrow kernels): one plan, one gate/up launch and one down launch over every
// wide job of the pass, in 2 x SMs CTAs each (grid-stride over the pass's tiles, as the BF16 stage-decode route).
template <QType GU, QType D>
void launch_wide_pass_ggml_a8(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                              const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                              const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream) {
    if (jobs > kBf16PlanThreads) { throw std::invalid_argument("offloaded_sparse_moe: an 8-bit wide pass plans at most 512 jobs"); }
    constexpr int kGateUpShared = static_cast<int>(sizeof(a8w::DecodeShared<GU>));
    constexpr int kDownShared   = static_cast<int>(sizeof(a8w::DecodeShared<D>));
    static const bool configured = [] {
        return cudaFuncSetAttribute(ggml_gate_up_a8_wide<GU>, cudaFuncAttributeMaxDynamicSharedMemorySize, kGateUpShared) ==
                   cudaSuccess &&
               cudaFuncSetAttribute(ggml_down_a8_wide<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDownShared) ==
                   cudaSuccess;
    }();
    if (!configured) { throw std::runtime_error("offloaded_sparse_moe: the 8-bit wide tiles' shared memory exceeds the device's"); }
    ggml_bf16_plan_kernel<<<1, kBf16PlanThreads, 0, stream>>>(dispatch, base, jobs, flags, job_records, layout.tiles,
                                                               layout.pass_tiles);
    check_launch("ggml a8 wide plan");
    const int sms = bf16_expert_sms();
    ggml_gate_up_a8_wide<GU><<<dim3(kA8WidePairs, (2 * sms + kA8WidePairs - 1) / kA8WidePairs), a8w::kThreads, kGateUpShared,
                               stream>>>(layout.x_limbs, dispatch, source, top_k, job_records, layout.tiles, layout.pass_tiles,
                                         jobs, layout.h_rows, ggml::kappa_log2(GU));
    check_launch("ggml a8 wide gate/up tiles");
    ggml_down_a8_wide<D><<<dim3(kA8WideDownTiles, (2 * sms + kA8WideDownTiles - 1) / kA8WideDownTiles), a8w::kThreads,
                           kDownShared, stream>>>(dispatch, source, static_cast<int>(moe::kHidden), job_records, layout.tiles,
                                                  layout.pass_tiles, jobs, layout.h_rows,
                                                  reinterpret_cast<std::uint16_t*>(outputs), ggml::kappa_log2(D));
    check_launch("ggml a8 wide down tiles");
}

// Encodes the call's x for the 8-bit gate/up kernels and tiles (once per moe_experts call, before its passes).
void launch_encode_x_ggml_a8(const Tensor& x, const moe::wide::ExpertsWorkspace& layout, cudaStream_t stream) {
    ggml_a8_encode_x_kernel<<<static_cast<unsigned>(x.ne[1]), kA8EncodeThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data), layout.x_limbs);
    check_launch("ggml a8 encode x");
}

// One pass of the 8-bit kernels; `wide` (an a8_wide_call) leaves the jobs of more than moe::kMaxColumns columns to
// the wide tiles, launched after the narrow kernels.
template <QType GU, QType D>
void launch_expert_pass_ggml_a8(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                                const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                                const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                                PassPhase phase, PassPart part, bool wide) {
    auto* h_plane        = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    const int narrow_max = wide ? moe::kMaxColumns : std::numeric_limits<int>::max();
    if (part != PassPart::Down) {
        gate_up_kernel_ggml_a8<GU><<<dim3(moe::kIntermediate / kGgmlGateUpUnits, jobs), kGgmlThreads, 0, stream>>>(
            layout.x_limbs, dispatch, source, top_k, job_records, flags, base, h_plane, phase, narrow_max);
        check_launch("ggml a8 gate/up");
    }
    if (part == PassPart::GateUp) { return; }
    CUDA_CHECK(pdl::launch_with(part == PassPart::Down ? pdl::Dependency::Serialized : pdl::Dependency::Programmatic,
                                {dim3(moe::kHidden / kGgmlDownRows, jobs), dim3(kGgmlThreads), 0, stream},
                                down_kernel_ggml_a8<D>, dispatch, source, static_cast<int>(moe::kHidden), job_records,
                                flags, base, static_cast<const std::uint16_t*>(h_plane), outputs, phase, narrow_max));
    if (wide) { launch_wide_pass_ggml_a8<GU, D>(dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs, stream); }
}

// launch_pass_ggml's pairs on the 8-bit route.
void launch_pass_ggml_a8(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                         const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                         const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream, PassPhase phase,
                         PassPart part, bool wide) {
    // The wide tiles run every job of the pass: a forked call (Staged/Resident phases, gate/up and down parts)
    // is narrow (forked()'s column bound), so it never has them.
    if (wide && (phase != PassPhase::All || part != PassPart::Both)) {
        throw std::logic_error("offloaded_sparse_moe: the 8-bit wide tiles do not split by pass phase");
    }
    const QType gu = source.ggml.gate_up_format, d = source.ggml.down_format;
#define INFERNIX_GGML_A8_PAIR(G, DD)                                                                                   \
    if (gu == G && d == DD) {                                                                                          \
        launch_expert_pass_ggml_a8<G, DD>(dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs,   \
                                          stream, phase, part, wide);                                                 \
        return;                                                                                                        \
    }
    INFERNIX_GGML_A8_PAIR(QType::GGML_IQ2_S, QType::GGML_IQ4_NL)
    INFERNIX_GGML_A8_PAIR(QType::GGML_IQ2_S, QType::GGML_Q2_0)
    INFERNIX_GGML_A8_PAIR(QType::GGML_IQ3_XXS, QType::GGML_IQ4_NL)
    INFERNIX_GGML_A8_PAIR(QType::GGML_IQ3_XXS, QType::GGML_Q2_0)
    INFERNIX_GGML_A8_PAIR(QType::GGML_IQ3_S, QType::GGML_IQ4_NL)
    INFERNIX_GGML_A8_PAIR(QType::GGML_IQ3_S, QType::GGML_Q2_0)
    INFERNIX_GGML_A8_PAIR(QType::GGML_IQ4_XS, QType::GGML_IQ4_NL)
    INFERNIX_GGML_A8_PAIR(QType::GGML_IQ4_XS, QType::GGML_Q2_0)
#undef INFERNIX_GGML_A8_PAIR
    throw std::invalid_argument("offloaded_sparse_moe: unsupported ggml expert format pair");
}
