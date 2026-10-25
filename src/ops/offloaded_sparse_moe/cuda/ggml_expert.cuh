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

// SharedTables (INFERNIX_TMP_GGML_SHARED_CODEBOOKS): the decode reads GU's codebook from a shared
// copy after the tile's dynamic shared memory (the same values, so the same bits).
template <QType GU, bool SharedTables>
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
    if constexpr (SharedTables) {
        static_assert(sizeof(ggml::tile::Shared) % 16 == 0);
        auto& codebook = *reinterpret_cast<ggml::fast::SharedCodebook<GU>*>(tile_shared + sizeof(ggml::tile::Shared));
        codebook.template load<kGgmlTileThreads>(); // published by compute's row_bases barrier
        ggml::tile::compute<GU, kGgmlTileThreads>(sm, row_of, kGgmlXSlices, limbs, 0, columns, acc, codebook);
    } else {
        ggml::tile::compute<GU, kGgmlTileThreads>(sm, row_of, kGgmlXSlices, limbs, 0, columns, acc);
    }
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

// ------------------------------------------------------------------------- ggml-bf16-prefill route
//
// With WideA16::kGgmlBf16 and the workspace's planes (calls of kGgmlBf16MinColumns or more columns),
// a pass's wide jobs run ops/ggml/ggml_bf16_tile.cuh's BF16 tiles instead of the exact ones,
// kGgmlBf16Experts jobs at a time: each wide job's gate/up rows and down rows are decoded into the
// planes (ggml_expert_plane_kernel); gate/up tiles of 128 rows (64 intermediates' gate rows and the
// same intermediates' up rows) by 128 of the job's entries write each entry's BF16 h =
// swiglu_bf16(bf16(gate), bf16(up)); down tiles of 128 rows by 128 entries write its BF16 output.
// x and h are read as stored (no limbs, no h encoding); every product is exact and only FP32 sums
// round, so an output depends only on its expert, its entry's x and the K order, not on the call.
// Jobs of at most kMaxColumns columns keep the narrow kernels. Calls that MoeGgmlTiles::stage_decode
// selects (TEMPORARY A/B switch) run the stage-decode tiles below instead: the same outputs without
// the planes, at any width.

constexpr int kBf16PlanThreads = 512;
constexpr int kBf16GateUpPairs = moe::kIntermediate / ggml::tile::kTileRows; // row-tile pairs (gate, up)
constexpr int kBf16DownTiles   = moe::kHidden / ggml::bf16::kRows;
constexpr int kBf16XStages     = moe::kHidden / (ggml::tile::kStageSlices * 32);
constexpr int kBf16HStages     = moe::kIntermediate / (ggml::tile::kStageSlices * 32);
static_assert(moe::wide::kGgmlCodeStageBytes == sizeof(ggml::tile::CodeStage), "the planes hold CodeStage images");
static_assert(moe::wide::kGgmlBf16GateUpImages == static_cast<std::size_t>(2 * kBf16GateUpPairs) * kBf16XStages);
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

// Decodes job (sub + z)'s plane tile x, stage y into slot z of the planes: its gate/up plane (gate rows
// 0 .. kIntermediate - 1, then the up rows; K = kHidden) or, with `down`, its down plane (K =
// kIntermediate). Jobs that do not take the BF16 tiles are skipped. SharedTables
// (INFERNIX_TMP_GGML_SHARED_CODEBOOKS): CTA (x, y, z) decodes stages kPlaneRun y .. kPlaneRun y +
// kPlaneRun - 1 with F's codebook copied once into shared memory (one stage per CTA could not
// amortise the copy); the same images.
constexpr int kPlaneRun = 5; // stages per CTA with SharedTables
static_assert(kBf16XStages % kPlaneRun == 0 && kBf16HStages % kPlaneRun == 0);
template <QType F, bool SharedTables>
__global__ void __launch_bounds__(ggml::tile::kThreads)
    ggml_expert_plane_kernel(MoeDispatch dispatch, MoeExpertSource source, const std::uint8_t* const* __restrict__ job_records,
                             const std::int32_t* __restrict__ cpu_flags, int sub, int down, std::uint8_t* __restrict__ planes) {
    __shared__ ggml::tile::CodeStage cs;
    const int job = sub + static_cast<int>(blockIdx.z);
    if (!bf16_job(dispatch, cpu_flags, job_records, job)) { return; }
    constexpr int kRun = SharedTables ? kPlaneRun : 1;
    const int tile = static_cast<int>(blockIdx.x), stages = static_cast<int>(gridDim.y) * kRun;
    const std::size_t row_bytes = down != 0 ? source.ggml.down_row_bytes : source.ggml.gate_row_bytes;
    const std::uint8_t* rows    = job_records[job] + (down != 0 ? 2 * static_cast<std::size_t>(moe::kIntermediate) *
                                                                       source.ggml.gate_row_bytes
                                                                 : 0);
    const int count = down != 0 ? moe::kHidden : 2 * moe::kIntermediate;
    const auto row_of = [&](int r) {
        return rows + static_cast<std::size_t>(min(tile * ggml::tile::kTileRows + r, count - 1)) * row_bytes;
    };
    std::uint8_t* plane = planes + static_cast<std::size_t>(blockIdx.z) * moe::wide::kGgmlBf16ExpertBytes +
                          (down != 0 ? moe::wide::kGgmlBf16GateUpImages * sizeof(ggml::tile::CodeStage) : 0);
    const auto store = [&](int st) {
        __syncthreads(); // the stage is decoded
        auto* dst       = reinterpret_cast<int4*>(plane + (static_cast<std::size_t>(tile) * stages + st) * sizeof(cs));
        const auto* src = reinterpret_cast<const int4*>(&cs);
        for (int i = static_cast<int>(threadIdx.x); i < ggml::tile::kImageBytes<F> / 16; i += static_cast<int>(blockDim.x)) {
            dst[i] = src[i];
        }
    };
    if constexpr (SharedTables) {
        __shared__ ggml::fast::SharedCodebook<F> codebook;
        codebook.template load<ggml::tile::kThreads>();
        __syncthreads();
        for (int k = 0; k < kRun; ++k) {
            const int st = static_cast<int>(blockIdx.y) * kRun + k;
            ggml::tile::decode_stage<F>(cs, row_of, st, codebook);
            store(st);
            __syncthreads(); // every thread copied the image out before the next stage's decode
        }
    } else {
        const int st = static_cast<int>(blockIdx.y);
        ggml::tile::decode_stage<F>(cs, row_of, st);
        store(st);
    }
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

// Gate/up tiles of jobs [sub, sub + jobs) of the pass that begins at job `base`: CTA (p, y) takes row
// pair p (intermediates 64 p .. 64 p + 63) of the sub-batch's tiles y, y + gridDim.y, ...
template <ggml::bf16::Scaling S>
__global__ void __launch_bounds__(ggml::bf16::kThreads, 1)
    ggml_gate_up_bf16(const std::uint16_t* __restrict__ x, MoeDispatch dispatch, int top_k,
                      const std::int32_t* __restrict__ tiles, const std::int32_t* __restrict__ first_tile, int base, int sub,
                      int jobs, const std::uint8_t* __restrict__ planes, std::uint16_t* __restrict__ h_plane, int kappa_log2) {
    namespace b = ggml::bf16;
    extern __shared__ __align__(16) unsigned char bf16_shared[];
    auto& sm       = *reinterpret_cast<b::Shared*>(bf16_shared);
    auto* up       = reinterpret_cast<std::uint16_t*>(&sm.x[0][0][0][0]); // [64][kColumns] after the K loop
    const int pair = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    constexpr std::size_t kImages = static_cast<std::size_t>(kBf16XStages) * sizeof(ggml::tile::CodeStage);
    const int t1 = first_tile[sub - base + jobs];
    for (int tile = first_tile[sub - base] + static_cast<int>(blockIdx.y); tile < t1; tile += static_cast<int>(gridDim.y)) {
        const int job = tiles[2 * tile], col0 = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        const std::uint8_t* plane = planes + static_cast<std::size_t>(job - sub) * moe::wide::kGgmlBf16ExpertBytes;
        const std::uint8_t* const images[2] = {plane + pair * kImages, plane + (kBf16GateUpPairs + pair) * kImages};
        const GatherColumns cols{x, static_cast<std::size_t>(moe::kHidden), dispatch.entries, first, col0, count, top_k};
        b::Accumulators acc;
        b::compute<S>(sm, images, kBf16XStages, cols, acc);
        // The up rows (warp row 1) through shared memory to the gate rows' threads (warp row 0).
        if (warp >= 4) {
#pragma unroll
            for (int mi = 0; mi < b::kMTiles; ++mi) {
#pragma unroll
                for (int nt = 0; nt < b::kNTiles; ++nt) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = b::out_row(warp, lane, mi, q) - ggml::tile::kTileRows, c = b::out_column(warp, lane, nt, q);
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
                for (int nt = 0; nt < b::kNTiles; ++nt) {
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int r = b::out_row(warp, lane, mi, q), c = b::out_column(warp, lane, nt, q);
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
    }
}

// Down tiles of jobs [sub, sub + jobs): CTA (q, y) takes output rows 128 q .. 128 q + 127 of the
// sub-batch's tiles y, y + gridDim.y, ...; each entry's output column is outputs[entries[e]].
template <ggml::bf16::Scaling S>
__global__ void __launch_bounds__(ggml::bf16::kThreads, 1)
    ggml_down_bf16(MoeDispatch dispatch, int hidden, const std::int32_t* __restrict__ tiles,
                   const std::int32_t* __restrict__ first_tile, int base, int sub, int jobs,
                   const std::uint8_t* __restrict__ planes, const std::uint16_t* __restrict__ h_plane,
                   std::uint16_t* __restrict__ outputs, int kappa_log2) {
    namespace b = ggml::bf16;
    extern __shared__ __align__(16) unsigned char bf16_shared[];
    auto& sm    = *reinterpret_cast<b::Shared*>(bf16_shared);
    const int q = static_cast<int>(blockIdx.x);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    constexpr std::size_t kImages = static_cast<std::size_t>(kBf16HStages) * sizeof(ggml::tile::CodeStage);
    const int t1 = first_tile[sub - base + jobs];
    for (int tile = first_tile[sub - base] + static_cast<int>(blockIdx.y); tile < t1; tile += static_cast<int>(gridDim.y)) {
        const int job = tiles[2 * tile], col0 = tiles[2 * tile + 1];
        const int expert = dispatch.jobs[job];
        const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
        const std::uint8_t* plane = planes + static_cast<std::size_t>(job - sub) * moe::wide::kGgmlBf16ExpertBytes +
                                    moe::wide::kGgmlBf16GateUpImages * sizeof(ggml::tile::CodeStage);
        const std::uint8_t* const images[2] = {plane + 2 * q * kImages, plane + (2 * q + 1) * kImages};
        const GatherColumns cols{h_plane, static_cast<std::size_t>(moe::wide::kA16HRowElements), dispatch.entries, first, col0,
                                 count, 0};
        b::Accumulators acc;
        b::compute<S>(sm, images, kBf16HStages, cols, acc);
#pragma unroll
        for (int mi = 0; mi < b::kMTiles; ++mi) {
#pragma unroll
            for (int nt = 0; nt < b::kNTiles; ++nt) {
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int c = b::out_column(warp, lane, nt, k);
                    if (col0 + c < count) {
                        const int row = q * b::kRows + b::out_row(warp, lane, mi, k);
                        outputs[static_cast<std::size_t>(dispatch.entries[first + col0 + c]) * hidden + row] =
                            bf16_bits(ldexpf(acc.r[mi][nt][k], kappa_log2));
                    }
                }
            }
        }
    }
}

// ---- The stage-decode tiles (MoeGgmlTiles::stage_decode; ggml_bf16_tile.cuh compute_decode)
//
// The planes' tiles without the planes: every wide job of a pass at once (no sub-batches), the
// pass's 128-entry tiles planned by ggml_bf16_plan_kernel; a job's last tile takes 64 columns when
// at most 64 of its entries remain. Each CTA copies its rows' raw GGUF bytes and decodes them per
// K stage, so a job's weights are decoded once per 128 of its entries instead of written to and read
// back from a plane; every output is the planes' bits.

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

// CTAs per SM of the BF16 expert tiles' grid-stride loops (the planes of a sub-batch, its tiles).
int bf16_expert_sms() {
    static const int sms = [] {
        int device = 0, count = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
        return count;
    }();
    return sms;
}

// The BF16 route's wide jobs of a pass (after its narrow kernels).
template <QType GU, QType D>
void launch_bf16_pass_ggml(const std::uint16_t* x, const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                           const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                           const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream) {
    namespace b = ggml::bf16;
    if (jobs > kBf16PlanThreads) { throw std::invalid_argument("offloaded_sparse_moe: a BF16 pass plans at most 512 jobs"); }
    constexpr int kShared = static_cast<int>(sizeof(b::Shared));
    static const bool configured = [] {
        return cudaFuncSetAttribute(ggml_gate_up_bf16<b::kScaling<GU>>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kShared) == cudaSuccess &&
               cudaFuncSetAttribute(ggml_down_bf16<b::kScaling<D>>, cudaFuncAttributeMaxDynamicSharedMemorySize, kShared) ==
                   cudaSuccess;
    }();
    if (!configured) { throw std::runtime_error("offloaded_sparse_moe: the BF16 tiles' shared memory exceeds the device's"); }
    auto* h_plane = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    ggml_bf16_plan_kernel<<<1, kBf16PlanThreads, 0, stream>>>(dispatch, base, jobs, flags, job_records, layout.tiles,
                                                               layout.pass_tiles);
    check_launch("ggml bf16 plan");
    const int batch = moe::wide::kGgmlBf16Experts, sms = bf16_expert_sms();
    const bool shared_tables = source.ggml.tiles.shared_codebooks;
    const int run            = shared_tables ? kPlaneRun : 1;
    for (int sub = base; sub < base + jobs; sub += batch) {
        const int n = std::min(batch, base + jobs - sub);
        const dim3 gate_up_grid(2 * kBf16GateUpPairs, kBf16XStages / run, n);
        const dim3 down_grid(moe::kHidden / ggml::tile::kTileRows, kBf16HStages / run, n);
        if (shared_tables) {
            ggml_expert_plane_kernel<GU, true><<<gate_up_grid, ggml::tile::kThreads, 0, stream>>>(
                dispatch, source, job_records, flags, sub, 0, layout.ggml_planes);
            check_launch("ggml bf16 gate/up planes");
            ggml_expert_plane_kernel<D, true><<<down_grid, ggml::tile::kThreads, 0, stream>>>(
                dispatch, source, job_records, flags, sub, 1, layout.ggml_planes);
        } else {
            ggml_expert_plane_kernel<GU, false><<<gate_up_grid, ggml::tile::kThreads, 0, stream>>>(
                dispatch, source, job_records, flags, sub, 0, layout.ggml_planes);
            check_launch("ggml bf16 gate/up planes");
            ggml_expert_plane_kernel<D, false><<<down_grid, ggml::tile::kThreads, 0, stream>>>(
                dispatch, source, job_records, flags, sub, 1, layout.ggml_planes);
        }
        check_launch("ggml bf16 down planes");
        ggml_gate_up_bf16<b::kScaling<GU>><<<dim3(kBf16GateUpPairs, (2 * sms + kBf16GateUpPairs - 1) / kBf16GateUpPairs),
                                             b::kThreads, kShared, stream>>>(
            x, dispatch, top_k, layout.tiles, layout.pass_tiles, base, sub, n, layout.ggml_planes, h_plane,
            ggml::kappa_log2(GU));
        check_launch("ggml bf16 gate/up tiles");
        ggml_down_bf16<b::kScaling<D>><<<dim3(kBf16DownTiles, (2 * sms + kBf16DownTiles - 1) / kBf16DownTiles), b::kThreads,
                                         kShared, stream>>>(dispatch, static_cast<int>(moe::kHidden), layout.tiles,
                                                            layout.pass_tiles, base, sub, n, layout.ggml_planes, h_plane,
                                                            reinterpret_cast<std::uint16_t*>(outputs), ggml::kappa_log2(D));
        check_launch("ggml bf16 down tiles");
    }
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
    // Grid-stride loops over the pass's tiles in 2 x SMs CTAs (two of one CTA per SM), as the planes' tiles.
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
                             PassPhase phase, int entries, int columns) {
    auto* h_plane        = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    const int narrow_max = entries > 0 ? moe::kMaxColumns : std::numeric_limits<int>::max();
    gate_up_kernel_ggml<GU><<<dim3(moe::kIntermediate / kGgmlGateUpUnits, jobs), kGgmlThreads, 0, stream>>>(
        layout.x_limbs, layout.x_emax, dispatch, source, top_k, job_records, flags, base, h_plane, phase, narrow_max);
    check_launch("ggml gate/up");
    CUDA_CHECK(pdl::launch_consumer({dim3(moe::kHidden / kGgmlDownRows, jobs), dim3(kGgmlThreads), 0, stream},
                                    down_kernel_ggml<D>, dispatch, source, static_cast<int>(moe::kHidden), job_records,
                                    flags, base, static_cast<const std::uint16_t*>(h_plane), outputs, phase, narrow_max));
    if (entries == 0) { return; }
    if (source.wide_a16 == moe::WideA16::kGgmlBf16 && source.ggml.tiles.stage_decode(columns)) {
        launch_decode_pass_ggml<GU, D>(reinterpret_cast<const std::uint16_t*>(x), dispatch, source, top_k, job_records, flags,
                                       base, jobs, layout, outputs, stream);
        return;
    }
    if (source.wide_a16 == moe::WideA16::kGgmlBf16 && layout.ggml_planes != nullptr) {
        launch_bf16_pass_ggml<GU, D>(reinterpret_cast<const std::uint16_t*>(x), dispatch, source, top_k, job_records, flags,
                                     base, jobs, layout, outputs, stream);
        return;
    }
    // At most one partial tile per job plus one per 64 entries; CTAs past the pass's tiles exit.
    const unsigned tile_bound = static_cast<unsigned>(jobs + (entries + kGgmlTileColumns - 1) / kGgmlTileColumns);
    constexpr int kTileShared   = static_cast<int>(sizeof(ggml::tile::Shared));
    constexpr int kTablesShared = kTileShared + static_cast<int>(sizeof(ggml::fast::SharedCodebook<GU>));
    constexpr int kDownShared   = static_cast<int>(sizeof(GgmlDownShared<D>));
    static const bool configured = [] {
        return cudaFuncSetAttribute(gate_up_tile_ggml<GU, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTileShared) ==
                   cudaSuccess &&
               cudaFuncSetAttribute(gate_up_tile_ggml<GU, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTablesShared) ==
                   cudaSuccess &&
               cudaFuncSetAttribute(down_resident_ggml<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, kDownShared) ==
                   cudaSuccess;
    }();
    if (!configured) { throw std::runtime_error("offloaded_sparse_moe: the ggml tiles' shared memory exceeds the device's"); }
    const dim3 gate_up_grid(moe::kIntermediate / kGgmlTileUnits, tile_bound);
    if (source.ggml.tiles.shared_codebooks) {
        gate_up_tile_ggml<GU, true><<<gate_up_grid, kGgmlTileThreads, kTablesShared, stream>>>(
            layout.x_limbs, layout.x_emax, dispatch, source, top_k, job_records, layout.tiles, layout.pass_tiles, base, jobs,
            h_plane);
    } else {
        gate_up_tile_ggml<GU, false><<<gate_up_grid, kGgmlTileThreads, kTileShared, stream>>>(
            layout.x_limbs, layout.x_emax, dispatch, source, top_k, job_records, layout.tiles, layout.pass_tiles, base, jobs,
            h_plane);
    }
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
                      int entries, int columns) {
    // The wide tiles run every job of the pass: a forked call (Staged/Resident phases) must not have
    // wide entries, which forked()'s column bound guarantees today.
    if (entries > 0 && phase != PassPhase::All) {
        throw std::logic_error("offloaded_sparse_moe: the GGUF wide tiles do not split by pass phase");
    }
    const QType gu = source.ggml.gate_up_format, d = source.ggml.down_format;
#define INFERNIX_GGML_PAIR(G, DD)                                                                                       \
    if (gu == G && d == DD) {                                                                                          \
        launch_expert_pass_ggml<G, DD>(x, dispatch, source, top_k, job_records, flags, base, jobs, layout, outputs,   \
                                       stream, phase, entries, columns);                                              \
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
