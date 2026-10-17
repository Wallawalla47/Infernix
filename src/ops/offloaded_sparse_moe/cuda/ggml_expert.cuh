// GGUF (ggml) expert banks on the GPU (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25), included
// by moe_layer.cu after its pass helpers. A record holds the gate rows, the up rows (one ggml
// format) and the down rows (another), each row verbatim GGUF blocks.
//
// The arithmetic is ops/ggml/ggml_exact.h in the dp4a form of ops/ggml/ggml_dp4a.cuh: x and h are
// encoded per column as A16 limbs and every output row is an exact int128 rounded once, so an
// expert's outputs depend neither on where its record lives nor on how many columns share the
// call; the same kernels serve every column count (there is no wide route for ggml banks).
//
// - ggml_encode_x_kernel encodes every token's x once per call into the A16 workspace's x limb
//   rows (kHidden / 32 slices of kSliceWords words: exactly kA16XRowBytes; the expert formats
//   have no min term, so no slice sums) and x_emax.
// - A gate/up CTA owns kGgmlGateUpUnits intermediates, one per warp: lanes 0-15 its gate row and
//   lanes 16-31 its up row, five slices each; one shuffle pairs them for SwiGLU, and lane 0 writes
//   BF16 h to the entry's h row.
// - A down CTA owns kGgmlDownRows output rows, four lanes per row (K = 640: twenty slices); each
//   warp encodes one column's h into shared memory per pass.

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

// One CTA per column.
__global__ void __launch_bounds__(kThreads)
    ggml_encode_x_kernel(const bf16* __restrict__ x, std::uint8_t* __restrict__ x_limbs, std::int32_t* __restrict__ x_emax) {
    __shared__ unsigned warp_key[kWarps];
    __shared__ int column_emax;
    const int t = blockIdx.x;
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

template <QType GU>
__global__ void __launch_bounds__(kGgmlThreads)
    gate_up_kernel_ggml(const std::uint8_t* __restrict__ x_limbs, const std::int32_t* __restrict__ x_emax,
                        MoeDispatch dispatch, MoeExpertSource source, int top_k,
                        const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                        int job_base, std::uint16_t* __restrict__ h_plane, PassPhase phase, int narrow_max) {
    constexpr int C = kGgmlColumns;
    const int job   = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (count > narrow_max) { return; } // a wide job: its tiles compute it
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int part = lane % 16, up = lane / 16;
    const int i    = blockIdx.x * kGgmlGateUpUnits + warp; // intermediate
    const int row_bytes = static_cast<int>(source.ggml.gate_row_bytes);
    const std::uint8_t* row = record + (static_cast<std::size_t>(up) * moe::kIntermediate + i) * row_bytes;
    pdl::trigger_dependents();
    for (int pass = 0; pass < count; pass += C) {
        const int n = min(C, count - pass);
        const std::int32_t* xs[C];
        int tokens[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            tokens[c] = c < n ? dispatch.entries[first + pass + c] / top_k : 0;
            xs[c]     = reinterpret_cast<const std::int32_t*>(x_limbs + static_cast<std::size_t>(tokens[c]) *
                                                                       moe::wide::kA16XRowBytes);
        }
        ggml::I128 acc[C];
        for (int s = part; s < kGgmlXSlices; s += 16) {
            const ggml::fast::Slice w      = ggml::fast::slice<GU>(row, s);
            const ggml::fast::SliceScale sc = ggml::fast::slice_scale(w);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) { ggml::fast::add_slice(acc[c], w, sc, xs[c] + s * ggml::fast::kSliceWords, 0); }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < n) {
#pragma unroll
                for (int offset = 8; offset >= 1; offset >>= 1) { ggml_shfl_add(acc[c], offset); }
                const std::uint16_t y     = ggml::output_bf16(acc[c], x_emax[tokens[c]], GU);
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
    down_kernel_ggml(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                     const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                     int job_base, const std::uint16_t* __restrict__ h_plane, bf16* __restrict__ outputs, PassPhase phase,
                     int narrow_max) {
    constexpr int C = kGgmlColumns;
    __shared__ __align__(16) std::int32_t limbs[C][kGgmlHSlices * ggml::fast::kSliceWords];
    __shared__ int emax[C];
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
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int part = lane % kGgmlDownLanes;
    const int row  = blockIdx.x * kGgmlDownRows + threadIdx.x / kGgmlDownLanes;
    const int row_bytes = static_cast<int>(source.ggml.down_row_bytes);
    const std::uint8_t* w = record + 2 * static_cast<std::size_t>(moe::kIntermediate) * source.ggml.gate_row_bytes +
                            static_cast<std::size_t>(row) * row_bytes;
    for (int pass = 0; pass < count; pass += C) {
        const int n = min(C, count - pass);
        // Warp c encodes column c's h: its exponent (a warp maximum), then its limbs.
        if (warp < n) {
            const std::uint16_t* h = h_plane + static_cast<std::size_t>(first + pass + warp) * moe::wide::kA16HRowElements;
            unsigned key = 0;
            for (int j = lane; j < moe::kIntermediate; j += 32) { key = max(key, static_cast<unsigned>(h[j] & 0x7FFFU)); }
            key = __reduce_max_sync(0xFFFFFFFFU, key);
            const int e = key >= 0x7F80U ? canon::kA16NonFinite : max(static_cast<int>(key >> 7), 1);
            if (lane == 0) { emax[warp] = e; }
            const auto value = [&](int j) { return e == canon::kA16NonFinite ? 0 : canon::a16_value(h[j], e); };
            for (int g = lane; g < moe::kIntermediate / 4; g += 32) {
                (void)ggml::fast::store_group(limbs[warp], g, value(4 * g), value(4 * g + 1), value(4 * g + 2),
                                              value(4 * g + 3));
            }
        }
        __syncthreads();
        ggml::I128 acc[C];
        for (int s = part; s < kGgmlHSlices; s += kGgmlDownLanes) {
            const ggml::fast::Slice ws      = ggml::fast::slice<D>(w, s);
            const ggml::fast::SliceScale sc = ggml::fast::slice_scale(ws);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) { ggml::fast::add_slice(acc[c], ws, sc, limbs[c] + s * ggml::fast::kSliceWords, 0); }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < n) {
#pragma unroll
                for (int offset = kGgmlDownLanes / 2; offset >= 1; offset >>= 1) { ggml_shfl_add(acc[c], offset); }
                if (part == 0) {
                    const int column = dispatch.entries[first + pass + c];
                    reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(column) * hidden + row] =
                        ggml::output_bf16(acc[c], emax[c], D);
                }
            }
        }
        __syncthreads(); // every warp has read the limbs before the next pass re-encodes them
    }
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

// Encodes the call's x for the gate/up kernels (once per moe_experts call, before its passes).
void launch_encode_x_ggml(const Tensor& x, const moe::wide::ExpertsWorkspace& layout, cudaStream_t stream) {
    ggml_encode_x_kernel<<<static_cast<unsigned>(x.ne[1]), kThreads, 0, stream>>>(static_cast<const bf16*>(x.data),
                                                                                   layout.x_limbs, layout.x_emax);
    check_launch("ggml encode x");
}

// `entries` is the call's entry count when the call is wide (planned by launch_plan_ggml), else 0.
template <QType GU, QType D>
void launch_expert_pass_ggml(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                             const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                             const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                             PassPhase phase, int entries) {
    auto* h_plane        = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    const int narrow_max = entries > 0 ? moe::kMaxColumns : std::numeric_limits<int>::max();
    gate_up_kernel_ggml<GU><<<dim3(moe::kIntermediate / kGgmlGateUpUnits, jobs), kGgmlThreads, 0, stream>>>(
        layout.x_limbs, layout.x_emax, dispatch, source, top_k, job_records, flags, base, h_plane, phase, narrow_max);
    check_launch("ggml gate/up");
    CUDA_CHECK(pdl::launch_consumer({dim3(moe::kHidden / kGgmlDownRows, jobs), dim3(kGgmlThreads), 0, stream},
                                    down_kernel_ggml<D>, dispatch, source, static_cast<int>(moe::kHidden), job_records,
                                    flags, base, static_cast<const std::uint16_t*>(h_plane), outputs, phase, narrow_max));
    if (entries == 0) { return; }
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
void launch_pass_ggml(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                      const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                      const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream, PassPhase phase,
                      int entries) {
    const QType gu = source.ggml.gate_up_format, d = source.ggml.down_format;
#define INFERNIX_GGML_PAIR(G, DD)                                                                                       \
    if (gu == G && d == DD) {                                                                                          \
        launch_expert_pass_ggml<G, DD>(dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs,      \
                                       stream, phase, entries);                                                       \
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
