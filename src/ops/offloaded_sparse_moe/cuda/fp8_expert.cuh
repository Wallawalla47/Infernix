// Block-FP8 expert banks (fp8_expert_b128_v1, Qwen's official FP8 experts) on the GPU
// (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25), included by moe_layer.cu after
// ggml_expert.cuh, whose x encoding, h plane and pass plumbing they share. A record holds the gate,
// up and down rows (one E4M3FN byte per weight), then the three matrices' BF16 128 x 128 tile scales.
//
// The arithmetic is ops/fp8_block/fp8_exact.h. A weight's integer w = e4m3fn(c) * 2^9 (|w| < 2^18)
// is split into three balanced base-64 digits w = w2 * 2^12 + w1 * 2^6 + w0 (w0, w1 in [-32, 31],
// |w2| <= 57), each an int8 limb, so a 32-element slice's exact sum against the column's three A16
// byte limbs is nine dp4a sums (each |S| < 2^18) combined in int64; each slice adds into the row's
// int128 with its tile's scale. Exact in any order: every column count uses these kernels. The
// digits come from a 256-entry table in shared memory, once per slice for all of a pass's columns.

constexpr int kFp8Tile = fp8b128::kTile;

// Every E4M3FN code's three digits w = w2 * 2^12 + w1 * 2^6 + w0 as int8 bytes 0-2 of one word
// (byte 3 zero), computed at compile time; each CTA copies the table to shared memory, so a code
// word's limbs are four shared loads and seven byte permutes instead of ~60 integer instructions.
struct Fp8DigitTable {
    std::uint32_t v[256];
    constexpr Fp8DigitTable() : v{} {
        for (int c = 0; c < 256; ++c) {
            const int e = (c >> 3) & 15, m = c & 7;
            const int magnitude = e == 0 ? m : (8 + m) << (e - 1);
            const int w  = (c & 0x80) != 0 ? -magnitude : magnitude; // fp8b128::fp8_int
            const int d0 = ((w + 32) & 63) - 32;
            const int w1 = (w - d0) / 64; // exact: w - d0 is a multiple of 64
            const int d1 = ((w1 + 32) & 63) - 32;
            const int d2 = (w1 - d1) / 64;
            v[c] = (static_cast<std::uint32_t>(d0) & 0xFFU) | ((static_cast<std::uint32_t>(d1) & 0xFFU) << 8) |
                   ((static_cast<std::uint32_t>(d2) & 0xFFU) << 16);
        }
    }
};
__device__ constexpr Fp8DigitTable kFp8Digits{};

// The CTA's shared copy of the digit table (every thread calls it; the caller syncs before use).
__device__ __forceinline__ void load_fp8_digits(std::uint32_t* table) {
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) { table[i] = kFp8Digits.v[i]; }
}

// The int8 limb words (w0, w1, w2) of four E4M3FN codes (one code word), from the shared table.
__device__ __forceinline__ void fp8_limbs(unsigned codes, const std::uint32_t* table, int& l0, int& l1, int& l2) {
    const unsigned t0 = table[codes & 0xFFU], t1 = table[(codes >> 8) & 0xFFU];
    const unsigned t2 = table[(codes >> 16) & 0xFFU], t3 = table[codes >> 24];
    const unsigned u01 = __byte_perm(t0, t1, 0x5140), u23 = __byte_perm(t2, t3, 0x5140); // digits 0, 1 interleaved
    const unsigned v01 = __byte_perm(t0, t1, 0x0062), v23 = __byte_perm(t2, t3, 0x6200); // digit 2
    l0 = static_cast<int>(__byte_perm(u01, u23, 0x5410));
    l1 = static_cast<int>(__byte_perm(u01, u23, 0x7632));
    l2 = static_cast<int>(__byte_perm(v01, v23, 0x7610));
}

// One 32-element slice's digit words: l[w][i] is digit i of code word w (elements 4w..4w+3).
struct Fp8Slice {
    int l[8][3];
};

// Decodes a slice once (`codes` 16-byte aligned, `table` the shared digits); every column then
// takes its dot products from it.
__device__ __forceinline__ Fp8Slice fp8_decode(const std::uint8_t* codes, const std::uint32_t* table) {
    const uint4 c0 = *reinterpret_cast<const uint4*>(codes);
    const uint4 c1 = *reinterpret_cast<const uint4*>(codes + 16);
    const unsigned words[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    Fp8Slice o;
#pragma unroll
    for (int w = 0; w < 8; ++w) { fp8_limbs(words[w], table, o.l[w][0], o.l[w][1], o.l[w][2]); }
    return o;
}

// The exact sum of w * X over a decoded slice: `limbs` the column slice's kSliceWords words (16-byte
// aligned; global or shared memory).
__device__ __forceinline__ std::int64_t fp8_slice(const Fp8Slice& d, const std::int32_t* limbs) {
    int s[3][3] = {}; // [weight digit][X limb a, b, c]
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const int a = limbs[w], b = limbs[8 + w], c = limbs[16 + w];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            s[i][0] = __dp4a(d.l[w][i], a, s[i][0]);
            s[i][1] = __dp4a(d.l[w][i], b, s[i][1]);
            s[i][2] = __dp4a(d.l[w][i], c, s[i][2]);
        }
    }
    std::int64_t p = 0;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        p += (static_cast<std::int64_t>(s[i][0]) << (6 * i + 16)) + (static_cast<std::int64_t>(s[i][1]) << (6 * i + 8)) +
             (static_cast<std::int64_t>(s[i][2]) << (6 * i));
    }
    return p;
}

// The record's tile scales: gate, up (each [I/128][H/128]) and down ([H/128][I/128]).
__device__ __forceinline__ const std::uint16_t* fp8_scales(const std::uint8_t* record) {
    return reinterpret_cast<const std::uint16_t*>(record + 3 * static_cast<std::size_t>(moe::kIntermediate) * moe::kHidden);
}
constexpr int kFp8GateTiles = (moe::kIntermediate / kFp8Tile) * (moe::kHidden / kFp8Tile);

__global__ void __launch_bounds__(kGgmlThreads)
    gate_up_kernel_fp8(const std::uint8_t* __restrict__ x_limbs, const std::int32_t* __restrict__ x_emax,
                       MoeDispatch dispatch, MoeExpertSource source, int top_k,
                       const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                       int job_base, std::uint16_t* __restrict__ h_plane, PassPhase phase, int narrow_max) {
    constexpr int C     = kGgmlColumns;
    constexpr int kTiles = moe::kHidden / kFp8Tile; // 20 per row
    const int job       = job_base + static_cast<int>(blockIdx.y);
    if (job >= *dispatch.job_count || (cpu_flags != nullptr && cpu_flags[job] != 0)) { return; }
    const int expert = dispatch.jobs[job];
    const std::uint8_t* record = pass_record(phase, source, expert, job_records, job);
    if (record == nullptr) { return; }
    const int first = dispatch.offsets[expert], count = dispatch.offsets[expert + 1] - first;
    if (count > narrow_max) { return; } // a wide job: its tiles compute it
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int part = lane % 16, up = lane / 16;
    const int i    = blockIdx.x * kGgmlGateUpUnits + warp; // intermediate
    const std::uint8_t* row = record + (static_cast<std::size_t>(up) * moe::kIntermediate + i) * moe::kHidden;
    const std::uint16_t* tiles = fp8_scales(record) + up * kFp8GateTiles + (i / kFp8Tile) * kTiles;
    const int e_min = fp8b128::min_exponent(tiles, kTiles);
    __shared__ std::uint32_t digits[256];
    load_fp8_digits(digits);
    __syncthreads();
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
            const std::uint16_t scale = tiles[s * 32 / kFp8Tile];
            const Fp8Slice ws         = fp8_decode(row + s * 32, digits);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) { fp8b128::add_tile(acc[c], fp8_slice(ws, xs[c] + s * ggml::fast::kSliceWords), scale, e_min); }
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c < n) {
#pragma unroll
                for (int offset = 8; offset >= 1; offset >>= 1) { ggml_shfl_add(acc[c], offset); }
                // Each half-warp's e_min is its own row's: gate and up round separately.
                const std::uint16_t y     = fp8b128::output_bf16(acc[c], e_min, x_emax[tokens[c]]);
                const std::uint16_t other = static_cast<std::uint16_t>(__shfl_xor_sync(0xFFFFFFFFU, y, 16));
                if (lane == 0) {
                    h_plane[static_cast<std::size_t>(first + pass + c) * moe::wide::kA16HRowElements + i] =
                        canon::swiglu_bf16(y, other);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(kGgmlThreads)
    down_kernel_fp8(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                    const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ cpu_flags,
                    int job_base, const std::uint16_t* __restrict__ h_plane, bf16* __restrict__ outputs, PassPhase phase,
                    int narrow_max) {
    constexpr int C      = kGgmlColumns;
    constexpr int kTiles = moe::kIntermediate / kFp8Tile; // 5 per row
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
    __shared__ std::uint32_t digits[256];
    load_fp8_digits(digits); // the first pass's barrier orders it before any read
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int part = lane % kGgmlDownLanes;
    const int row  = blockIdx.x * kGgmlDownRows + threadIdx.x / kGgmlDownLanes;
    const std::uint8_t* w = record + 2 * static_cast<std::size_t>(moe::kIntermediate) * moe::kHidden +
                            static_cast<std::size_t>(row) * moe::kIntermediate;
    const std::uint16_t* tiles = fp8_scales(record) + 2 * kFp8GateTiles + (row / kFp8Tile) * kTiles;
    const int e_min = fp8b128::min_exponent(tiles, kTiles);
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
            const std::uint16_t scale = tiles[s * 32 / kFp8Tile];
            const Fp8Slice ws         = fp8_decode(w + s * 32, digits);
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c < n) { fp8b128::add_tile(acc[c], fp8_slice(ws, limbs[c] + s * ggml::fast::kSliceWords), scale, e_min); }
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
                        fp8b128::output_bf16(acc[c], e_min, emax[c]);
                }
            }
        }
        __syncthreads(); // every warp has read the limbs before the next pass re-encodes them
    }
}

// ------------------------------------------------------------------------------- wide route
//
// In a wide call, jobs of more than kMaxColumns columns run int8 tensor-core tiles over the same
// integers: the plan's 64-column tiles (ggml_expert.cuh), each as two CTAs of 32 columns. A CTA
// computes 32 weight rows x 32 columns with four warps of 16 x 16. Each stage (64 elements of K)
// copies its rows' codes and its columns' limbs one stage ahead (cp.async), decodes the codes into
// the three digit planes in m16n8k32 A-fragment order (the shared digit table), and multiplies every
// digit plane by every limb plane: nine MMAs per n8 tile and k32 step, each accumulating in its own
// int32 over a 128-element scale tile (|S| < 4 * 2^18). At the end of each scale tile the nine sums
// form the tile's P = sum S_dl * 2^(6d + 8(2 - l)) (|P| < 2^52), added into the output's int128 with
// the tile's scale (fp8b128::add_tile): the integers of fp8_slice/add_tile summed per tile instead
// of per slice, so every route returns the same bits. Small CTAs (three per SM) overlap one
// another's decode and MMAs.
namespace fp8t {

constexpr int kThreads     = 128; // 2 x 2 warps of 16 rows x 16 columns
constexpr int kRows        = 32;
constexpr int kColumns     = 32;
constexpr int kStageSlices = 2;
constexpr int kBWords      = ggml::tile::kBWords;
constexpr int kLimbCopies  = kStageSlices * kColumns * 6 / kThreads;
constexpr int kCodeWords   = kRows * kStageSlices * 8 / kThreads; // code words decoded per thread and stage
static_assert(kLimbCopies * kThreads == kStageSlices * kColumns * 6 && kCodeWords * kThreads == kRows * kStageSlices * 8);
static_assert(kRows * kStageSlices * 32 == kThreads * 16, "one 16-byte code copy per thread and stage");
static_assert(kFp8Tile == 2 * kStageSlices * 32, "a scale tile is two stages");

struct Shared {
    // The stage's digit planes in A-fragment order (as ggml::tile::Shared::a, one per digit).
    alignas(16) int a[kStageSlices][3][kRows / 16][8][4][4];
    alignas(16) int b[2][kStageSlices][kColumns][kBWords]; // limbs a, b, c of the columns, two stages
    alignas(16) std::uint8_t raw[2][kRows][kStageSlices * 32]; // the rows' codes, two stages
    std::uint32_t digits[256];
    std::uint16_t up[16][kColumns]; // gate/up tiles: the up rows' BF16 outputs
};

// d += a * b over one 16x8x32 int8 tile (exact int32).
__device__ __forceinline__ void mma_k32_acc(int (&d)[4], int a0, int a1, int a2, int a3, int b0, int b1) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The CTA's tile: rows row_of(r) (K codes each, 16-byte aligned), columns limbs(c) (slice s at
// + s * kSliceWords, 16-byte aligned), `slices` = K / 32 (a multiple of 4); scale_of(kt) is the
// warp's rows' scale word of scale tile kt and e_min their smallest scale exponent. Thread
// (warp, lane) ends with rows 16 (warp / 2) + g (+ 8 for q >= 2) and columns 16 (warp % 2) + 8 nt +
// 2t + (q & 1) in acc[nt][q]. Every thread calls it.
template <class RowOf, class ColumnLimbs, class ScaleOf>
__device__ __forceinline__ void compute(Shared& sm, RowOf row_of, int slices, ColumnLimbs limbs, ScaleOf scale_of,
                                        int e_min, ggml::I128 (&acc)[2][4]) {
    const int tid = static_cast<int>(threadIdx.x), warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, t = lane % 4, wm = warp / 2, wn = warp % 2;
    // The thread's copies, the same every stage: limbs (three 16-byte pieces) and one 16-byte piece
    // of a row's codes.
    const std::int32_t* lsrc[kLimbCopies];
    int ldst[kLimbCopies];
#pragma unroll
    for (int k = 0; k < kLimbCopies; ++k) {
        const int i = tid + kThreads * k;
        const int j = i / (kColumns * 6), c = (i / 6) % kColumns, u = i % 6;
        lsrc[k] = limbs(c) + j * ggml::fast::kSliceWords + 4 * u;
        ldst[k] = (j * kColumns + c) * kBWords + 4 * u;
    }
    const std::uint8_t* csrc = row_of(tid / 4) + 16 * (tid % 4);
    const auto issue = [&](int buf, int s0) {
#pragma unroll
        for (int k = 0; k < kLimbCopies; ++k) {
            ggml::tile::cp_async16(&sm.b[buf][0][0][0] + ldst[k], lsrc[k] + s0 * ggml::fast::kSliceWords, 16);
        }
        ggml::tile::cp_async16(&sm.raw[buf][tid / 4][16 * (tid % 4)], csrc + s0 * 32, 16);
    };
    load_fp8_digits(sm.digits);
    int s[3][3][2][4]; // [digit][limb][n8 tile][output]
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            acc[nt][q] = ggml::I128{};
#pragma unroll
            for (int d = 0; d < 3; ++d) {
#pragma unroll
                for (int l = 0; l < 3; ++l) { s[d][l][nt][q] = 0; }
            }
        }
    }
    const int stages = slices / kStageSlices;
    issue(0, 0);
    ggml::tile::cp_async_commit();
    for (int st = 0; st < stages; ++st) {
        const int buf = st & 1;
        if (st + 1 < stages) { issue(buf ^ 1, (st + 1) * kStageSlices); }
        ggml::tile::cp_async_commit();
        ggml::tile::cp_async_wait<1>(); // everything but this stage's issues has landed
        __syncthreads();                // (the first time, also the digit table)
#pragma unroll
        for (int k = 0; k < kCodeWords; ++k) {
            const int cw = tid + kThreads * k;
            const int r = cw / 16, j = (cw / 8) % 2, w = cw % 8; // word w of slice j of row r
            int l[3];
            fp8_limbs(*reinterpret_cast<const unsigned*>(&sm.raw[buf][r][32 * j + 4 * w]), sm.digits, l[0], l[1], l[2]);
            const int at = 2 * (w / 4) + (r % 16) / 8;
#pragma unroll
            for (int d = 0; d < 3; ++d) { sm.a[j][d][r / 16][r % 8][w % 4][at] = l[d]; }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kStageSlices; ++j) {
            int b0[2][3], b1[2][3];
#pragma unroll
            for (int qq = 0; qq < 3; ++qq) {
                const int f = 4 * qq + lane / 8, fnt = f / 6, fl = (f % 6) / 2, fh = f % 2;
                int r[4];
                ggml::tile::ldmatrix_x4(r, &sm.b[buf][j][16 * wn + 8 * fnt + lane % 8][8 * fl + 4 * fh]);
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int fm = 4 * qq + m, nt = fm / 6, li = (fm % 6) / 2;
                    if (fm % 2 == 0) {
                        b0[nt][li] = r[m];
                    } else {
                        b1[nt][li] = r[m];
                    }
                }
            }
#pragma unroll
            for (int d = 0; d < 3; ++d) {
                const int4 a = *reinterpret_cast<const int4*>(sm.a[j][d][wm][g][t]);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int li = 0; li < 3; ++li) { mma_k32_acc(s[d][li][nt], a.x, a.y, a.z, a.w, b0[nt][li], b1[nt][li]); }
                }
            }
        }
        if (st % 2 == 1) { // a scale tile (two stages) ends
            const std::uint16_t scale = scale_of(st / 2);
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    std::int64_t p = 0;
#pragma unroll
                    for (int d = 0; d < 3; ++d) {
#pragma unroll
                        for (int li = 0; li < 3; ++li) {
                            p += static_cast<std::int64_t>(s[d][li][nt][q]) * (std::int64_t{1} << (6 * d + 8 * (2 - li)));
                            s[d][li][nt][q] = 0;
                        }
                    }
                    fp8b128::add_tile(acc[nt][q], p, scale, e_min);
                }
            }
        }
        __syncthreads(); // the next stage's decode and copies reuse these buffers
    }
}

} // namespace fp8t

__global__ void __launch_bounds__(fp8t::kThreads, 3)
    gate_up_tile_fp8(const std::uint8_t* __restrict__ x_limbs, const std::int32_t* __restrict__ x_emax,
                     MoeDispatch dispatch, MoeExpertSource source, int top_k,
                     const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ tiles,
                     const std::int32_t* __restrict__ first_tile, int job_base, int jobs, std::uint16_t* __restrict__ h_plane) {
    __shared__ fp8t::Shared sm;
    GgmlTile tl;
    if (!ggml_tile_of(dispatch, tiles, first_tile, job_base, jobs, tl)) { return; }
    tl.column0 += fp8t::kColumns * static_cast<int>(blockIdx.z); // the planned tile's half
    if (tl.column0 >= tl.count) { return; }
    const std::uint8_t* record = job_records[tl.job];
    if (record == nullptr) { return; }
    const int i0 = static_cast<int>(blockIdx.x) * 16; // the CTA's 16 intermediates: their gate rows, then up rows
    const auto row_of = [&](int r) {
        return record + (static_cast<std::size_t>(r < 16 ? 0 : moe::kIntermediate) + i0 + r % 16) * moe::kHidden;
    };
    const auto token = [&](int c) { return dispatch.entries[tl.first + min(tl.column0 + c, tl.count - 1)] / top_k; };
    const auto limbs = [&](int c) {
        return reinterpret_cast<const std::int32_t*>(x_limbs + static_cast<std::size_t>(token(c)) * moe::wide::kA16XRowBytes);
    };
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const bool up = warp >= 2; // warps 2-3 hold the up rows
    constexpr int kTiles = moe::kHidden / kFp8Tile;
    const std::uint16_t* scales = fp8_scales(record) + (up ? kFp8GateTiles : 0) + (i0 / kFp8Tile) * kTiles;
    const int e_min = fp8b128::min_exponent(scales, kTiles);
    ggml::I128 acc[2][4];
    fp8t::compute(sm, row_of, kGgmlXSlices, limbs, [&](int kt) { return scales[kt]; }, e_min, acc);
    const int g = lane / 4, t = lane % 4, wn = warp % 2;
    if (up) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int c = 16 * wn + 8 * nt + 2 * t + (q & 1);
                sm.up[g + 8 * (q >> 1)][c] = fp8b128::output_bf16(acc[nt][q], e_min, x_emax[token(c)]);
            }
        }
    }
    __syncthreads();
    if (!up) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int r = g + 8 * (q >> 1), c = 16 * wn + 8 * nt + 2 * t + (q & 1);
                if (tl.column0 + c < tl.count) {
                    const std::uint16_t y = fp8b128::output_bf16(acc[nt][q], e_min, x_emax[token(c)]);
                    h_plane[static_cast<std::size_t>(tl.first + tl.column0 + c) * moe::wide::kA16HRowElements + i0 + r] =
                        canon::swiglu_bf16(y, sm.up[r][c]);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(fp8t::kThreads, 3)
    down_tile_fp8(MoeDispatch dispatch, MoeExpertSource source, int hidden,
                  const std::uint8_t* const* __restrict__ job_records, const std::int32_t* __restrict__ tiles,
                  const std::int32_t* __restrict__ first_tile, int job_base, int jobs,
                  const std::uint16_t* __restrict__ h_plane, const std::int32_t* __restrict__ h_emax,
                  bf16* __restrict__ outputs) {
    __shared__ fp8t::Shared sm;
    GgmlTile tl;
    if (!ggml_tile_of(dispatch, tiles, first_tile, job_base, jobs, tl)) { return; }
    tl.column0 += fp8t::kColumns * static_cast<int>(blockIdx.z);
    if (tl.column0 >= tl.count) { return; }
    const std::uint8_t* record = job_records[tl.job];
    if (record == nullptr) { return; }
    const int row0 = static_cast<int>(blockIdx.x) * fp8t::kRows;
    const std::uint8_t* down = record + 2 * static_cast<std::size_t>(moe::kIntermediate) * moe::kHidden;
    const auto row_of = [&](int r) { return down + static_cast<std::size_t>(row0 + r) * moe::kIntermediate; };
    const auto entry  = [&](int c) { return tl.first + min(tl.column0 + c, tl.count - 1); };
    const auto limbs  = [&](int c) {
        return reinterpret_cast<const std::int32_t*>(h_plane + static_cast<std::size_t>(entry(c)) * moe::wide::kA16HRowElements);
    };
    constexpr int kTiles = moe::kIntermediate / kFp8Tile;
    const std::uint16_t* scales = fp8_scales(record) + 2 * kFp8GateTiles + (row0 / kFp8Tile) * kTiles;
    const int e_min = fp8b128::min_exponent(scales, kTiles);
    ggml::I128 acc[2][4];
    fp8t::compute(sm, row_of, kGgmlHSlices, limbs, [&](int kt) { return scales[kt]; }, e_min, acc);
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32;
    const int g = lane / 4, t = lane % 4, wm = warp / 2, wn = warp % 2;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int c = 16 * wn + 8 * nt + 2 * t + (q & 1);
            if (tl.column0 + c < tl.count) {
                const int row = row0 + 16 * wm + g + 8 * (q >> 1);
                reinterpret_cast<std::uint16_t*>(outputs)[static_cast<std::size_t>(dispatch.entries[entry(c)]) * hidden + row] =
                    fp8b128::output_bf16(acc[nt][q], e_min, h_emax[entry(c)]);
            }
        }
    }
}

// One pass of a block-FP8 bank's experts (the call's x was encoded by launch_encode_x_ggml).
// `entries` is the call's entry count when the call is wide (planned by launch_plan_ggml), else 0:
// then the narrow kernels skip the jobs of more than kMaxColumns columns, which the tiles compute.
void launch_expert_pass_fp8(const MoeDispatch& dispatch, const MoeExpertSource& source, int top_k,
                            const std::uint8_t* const* job_records, const std::int32_t* flags, int base, int jobs,
                            const moe::wide::ExpertsWorkspace& layout, bf16* outputs, cudaStream_t stream,
                            PassPhase phase, int entries) {
    // As launch_pass_ggml: the wide tiles run every job of the pass, so no phase split.
    if (entries > 0 && phase != PassPhase::All) {
        throw std::logic_error("offloaded_sparse_moe: the block-FP8 wide tiles do not split by pass phase");
    }
    auto* h_plane        = reinterpret_cast<std::uint16_t*>(layout.h_rows);
    const int narrow_max = entries > 0 ? moe::kMaxColumns : std::numeric_limits<int>::max();
    gate_up_kernel_fp8<<<dim3(moe::kIntermediate / kGgmlGateUpUnits, jobs), kGgmlThreads, 0, stream>>>(
        layout.x_limbs, layout.x_emax, dispatch, source, top_k, job_records, flags, base, h_plane, phase, narrow_max);
    check_launch("fp8 gate/up");
    CUDA_CHECK(pdl::launch_consumer({dim3(moe::kHidden / kGgmlDownRows, jobs), dim3(kGgmlThreads), 0, stream},
                                    down_kernel_fp8, dispatch, source, static_cast<int>(moe::kHidden), job_records,
                                    flags, base, static_cast<const std::uint16_t*>(h_plane), outputs, phase, narrow_max));
    if (entries == 0) { return; }
    const unsigned tile_bound = static_cast<unsigned>(jobs + (entries + kGgmlTileColumns - 1) / kGgmlTileColumns);
    gate_up_tile_fp8<<<dim3(moe::kIntermediate / 16, tile_bound, 2), fp8t::kThreads, 0, stream>>>(
        layout.x_limbs, layout.x_emax, dispatch, source, top_k, job_records, layout.tiles, layout.pass_tiles, base, jobs,
        h_plane);
    check_launch("fp8 gate/up tiles");
    encode_h_ggml<<<ggml_encode_h_ctas(), kThreads, 0, stream>>>(dispatch, layout.tiles, layout.pass_tiles, base,
                                                                         jobs, h_plane, layout.h_emax);
    check_launch("fp8 encode h");
    down_tile_fp8<<<dim3(moe::kHidden / fp8t::kRows, tile_bound, 2), fp8t::kThreads, 0, stream>>>(
        dispatch, source, static_cast<int>(moe::kHidden), job_records, layout.tiles, layout.pass_tiles, base, jobs,
        h_plane, layout.h_emax, outputs);
    check_launch("fp8 down tiles");
}
