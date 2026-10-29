#pragma once

// Narrow (decode) row tiles of the exact ggml arithmetic (ops/ggml/ggml_exact.h) on int8 tensor cores: the
// expert kernels' few columns per job (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25).
//
// The dp4a form (ggml_dp4a.cuh) spends most of a decode call's issue slots on per-column dot chains and
// per-slice int128 terms, and its warps wait on their own weight loads. Here a warp owns 8 weight rows
// and streams them through a private cp.async ring of raw stages, so the bytes in flight do not depend on
// how many warps are resident. Per m16n8k32 MMA, thread (g, t) (g = lane / 4, t = lane % 4) decodes
// chunk t (8 elements) of its row g for both of the MMA's row slots:
//
// - kPairs formats (one sub-scale per 32 elements): slot g holds slice 2p of row g and slot g + 8 slice
//   2p + 1 of the same row, and the B columns are n = 2 l + s, X's limb plane l (a, b, c) of slice 2p + s
//   (n = 6, 7 zero). C[g][2 l] is row g's slice 2p against limb l and C[g + 8][2 l + 1] its slice 2p + 1:
//   thread t < 3 holds both for l = t (c0 and c3; the other two products pair a slice with the other
//   slice's x and are not used).
// - kHalves formats (one sub-scale per 16 elements, IQ2_S): slot g holds the first half of slice p of
//   row g (threads t < 2; the others write zero) and slot g + 8 its second half (t >= 2), with B columns
//   n = 2 l (odd n zero): thread t < 3 holds both halves for l = t (c0 and c2).
//
// Each chunk's 8 elements fill k = 4 t .. 4 t + 3 and 16 + 4 t .. 16 + 4 t + 3 of the slot; B follows the
// same order (kNatural formats, whose 4-byte groups hold elements 4 t.. and 16 + 4 t.., read X words t and
// 4 + t; the others words 2 t and 2 t + 1). So thread t sums sub * C into an int32 block sum per column and
// adds mant(d) * sum * 2^(e(d) - 1) to an int128 per block: the integers of fast::add_slice, split by limb
// plane. At the end the planes combine as a * 2^16 + b * 2^8 + c (two shuffles), so every output is the
// same int128, and the same bits, as every other route's.

#include "core/ggml_format.h"
#include "ops/ggml/ggml_dp4a.cuh"
#include "ops/ggml/ggml_exact.h"
#include "ops/ggml/ggml_tile.cuh"

#include <cstddef>
#include <cstdint>

namespace infernix::ggml::nmma {

// Formats whose 4-byte code groups hold elements 4 t .. 4 t + 3 and 16 + 4 t .. 16 + 4 t + 3 (nibbles).
template <QType F>
inline constexpr bool kNatural = F == QType::GGML_IQ4_NL || F == QType::GGML_IQ4_XS;
// Formats whose sub-scale covers 16 elements.
template <QType F>
inline constexpr bool kHalves = F == QType::GGML_IQ2_S;
// Slices of one MMA step.
template <QType F>
inline constexpr int kStepSlices = kHalves<F> ? 1 : 2;
template <QType F>
inline constexpr int kBlockSlices = static_cast<int>(ggml_block_elems(F)) / 32;
template <QType F>
inline constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
template <QType F>
inline constexpr bool kSupported = F == QType::GGML_IQ2_S || F == QType::GGML_IQ3_XXS || F == QType::GGML_IQ3_S ||
                                   F == QType::GGML_IQ4_XS || F == QType::GGML_IQ4_NL || F == QType::GGML_Q2_0;

// A thread's A fragment of one step, with the sub-scales of its two slots.
struct Step {
    int a[4];
    int sub0, sub1;
};

__device__ __forceinline__ std::uint32_t lds16(const std::uint8_t* p) { return *reinterpret_cast<const std::uint16_t*>(p); }
__device__ __forceinline__ std::uint32_t lds32(const std::uint8_t* p) { return *reinterpret_cast<const std::uint32_t*>(p); }
// Four bytes at a 2-byte-aligned address.
__device__ __forceinline__ std::uint32_t lds32_a2(const std::uint8_t* p) { return lds16(p) | (lds16(p + 2) << 16); }

// Chunk t (elements 8 t .. 8 t + 7: words w0, w1) of slice sb of block b (shared memory, the block's
// global alignment kept), for the kPairs formats without kNatural.
template <QType F, class Codebook>
__device__ __forceinline__ void chunk_words(const std::uint8_t* b, int sb, int t, const Codebook& cb, int& w0, int& w1,
                                            int& sub) {
    if constexpr (F == QType::GGML_IQ3_S) {
        const int qh           = b[66 + sb];
        const std::uint32_t qs = lds16(b + 2 + 8 * sb + 2 * t);
        const int i1           = static_cast<int>(qs & 0xFFU) | ((qh << (8 - 2 * t)) & 256);
        const int i2           = static_cast<int>(qs >> 8) | ((qh << (7 - 2 * t)) & 256);
        const unsigned sign    = b[74 + 4 * sb + t];
        w0  = fast::apply_signs4(cb.grid32(i1), sign & 0xFU);
        w1  = fast::apply_signs4(cb.grid32(i2), sign >> 4);
        sub = 2 * ((b[106 + sb / 2] >> (4 * (sb % 2))) & 0xF) + 1;
    } else if constexpr (F == QType::GGML_IQ3_XXS) {
        const std::uint32_t aux = lds32_a2(b + 66 + 4 * sb);
        const std::uint32_t qs  = lds16(b + 2 + 8 * sb + 2 * t);
        const unsigned sign     = cb.sign((aux >> (7 * t)) & 127);
        w0  = fast::apply_signs4(cb.grid32(qs & 0xFFU), sign & 0xFU);
        w1  = fast::apply_signs4(cb.grid32(qs >> 8), sign >> 4);
        sub = 2 * static_cast<int>(aux >> 28) + 1;
    } else if constexpr (F == QType::GGML_IQ2_S) {
        // sub: the slice's scale byte (half 0 its low nibble, half 1 its high).
        const int index          = b[2 + 4 * sb + t] | ((b[66 + sb] << (8 - 2 * t)) & 0x300);
        const std::uint64_t grid = cb.grid64(index);
        const unsigned sign      = b[34 + 4 * sb + t];
        w0  = fast::apply_signs4(static_cast<std::uint32_t>(grid), sign & 0xFU);
        w1  = fast::apply_signs4(static_cast<std::uint32_t>(grid >> 32), sign >> 4);
        sub = b[74 + sb];
    } else if constexpr (F == QType::GGML_Q2_0) {
        // Element 4 i + k at bits 2 k of byte i of the slice's eight bytes, each code minus 1.
        const std::uint32_t v = lds16(b + 2 + 8 * sb + 2 * t);
        const auto codes      = [](std::uint32_t x) {
            return fast::sub_bytes4((x & 3U) | (((x >> 2) & 3U) << 8) | (((x >> 4) & 3U) << 16) | (((x >> 6) & 3U) << 24),
                                    0x01010101U);
        };
        w0  = codes(v & 0xFFU);
        w1  = codes(v >> 8);
        sub = 1;
    } else {
        static_assert(F == QType::GGML_IQ3_S, "chunk_words: not a chunk-order format");
    }
}

// Words t and 4 + t of slice sb of block b, for the kNatural formats.
template <QType F>
__device__ __forceinline__ void natural_words(const std::uint8_t* b, int sb, int t, int& lo, int& hi, int& sub) {
    if constexpr (F == QType::GGML_IQ4_NL) {
        const int2 v = fast::iq4nl_values(lds32_a2(b + 2 + 4 * t));
        lo  = v.x;
        hi  = v.y;
        sub = 1;
    } else if constexpr (F == QType::GGML_IQ4_XS) {
        const int2 v = fast::iq4nl_values(lds32(b + 8 + 16 * sb + 4 * t));
        lo  = v.x;
        hi  = v.y;
        sub = (((b[4 + sb / 2] >> (4 * (sb % 2))) & 0xF) | (((lds16(b + 2) >> (2 * sb)) & 3) << 4)) - 32;
    } else {
        static_assert(F == QType::GGML_IQ4_NL, "natural_words: not a nibble format");
    }
}

// The thread's step at slice `s` of a row piece `row` (s a multiple of kStepSlices; the piece starts at a
// block).
template <QType F, class Codebook>
__device__ __forceinline__ Step step(const std::uint8_t* row, int s, int t, const Codebook& cb) {
    Step o;
    if constexpr (kHalves<F>) {
        const std::uint8_t* b = row + (s / kBlockSlices<F>) * kBlockBytes<F>;
        int w0, w1, sc;
        chunk_words<F>(b, s % kBlockSlices<F>, t, cb, w0, w1, sc);
        const bool first = t < 2;
        o.a[0] = first ? w0 : 0;
        o.a[2] = first ? w1 : 0;
        o.a[1] = first ? 0 : w0;
        o.a[3] = first ? 0 : w1;
        o.sub0 = 2 * (sc & 0xF) + 1;
        o.sub1 = 2 * (sc >> 4) + 1;
    } else {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int sj          = s + j;
            const std::uint8_t* b = row + (sj / kBlockSlices<F>) * kBlockBytes<F>;
            int lo, hi, sub;
            if constexpr (kNatural<F>) {
                natural_words<F>(b, sj % kBlockSlices<F>, t, lo, hi, sub);
            } else {
                chunk_words<F>(b, sj % kBlockSlices<F>, t, cb, lo, hi, sub);
            }
            o.a[j]     = lo; // a0: slot g (slice s), a1: slot g + 8 (slice s + 1)
            o.a[2 + j] = hi;
            (j == 0 ? o.sub0 : o.sub1) = sub;
        }
    }
    return o;
}

// R += +-mant(d) * v * 2^(e(d) - 1).
__device__ __forceinline__ void add_term(I128& r, std::uint32_t d, long long v) {
    std::int64_t mant;
    int shift;
    bool negative;
    f16_parts(static_cast<std::uint16_t>(d), mant, shift, negative);
    const long long term = mant * v;
    add_shifted(r, negative ? -term : term, shift);
}

// X's words of a thread's B fragment: words (2 t, 2 t + 1), or (t, 4 + t) for kNatural: the first and the
// distance to the second (an `xpair(c, slice, word)` functor reads word and word + kWordStep).
template <QType F>
__device__ __forceinline__ int b_word0(int t) { return kNatural<F> ? t : 2 * t; }
template <QType F>
inline constexpr int kWordStep = kNatural<F> ? 4 : 1;

// The B row of a thread's lane: its limb plane l and slice offset (or -1: a zero column).
template <QType F>
__device__ __forceinline__ int b_limb(int g) { return kHalves<F> ? (g % 2 == 0 && g < 6 ? g / 2 : -1) : (g < 6 ? g / 2 : -1); }
template <QType F>
__device__ __forceinline__ int b_slice(int g) { return kHalves<F> ? 0 : g % 2; }

// The rows' accumulators of one call of the K loop: each column's sum of limb plane t (t < 3), as an int128
// (r), or (Base accumulation) an int64 in units of 2^base, base the row's smallest scale shift (a).
template <int C>
struct Rows {
    I128 r[C];
    long long a[C];
    int base;
};

// Base accumulation (the one-block-per-step formats IQ4_NL and Q2_0, whose int128 term per block would cost
// more than the step): the block term mant(d) * v * 2^(e(d) - 1) is M * v in units of 2^base with
// M = +-mant(d) * 2^(shift - base), an int32 while shift - base <= kBaseSpan, base the row's smallest shift;
// |v| < 2^19 for one slice of IQ4_NL (less for Q2_0), so a row's twenty terms are below 2^55.
inline constexpr int kBaseSpan = 20;

__device__ __forceinline__ int scale_shift(std::uint32_t d) { return max(static_cast<int>((d >> 10) & 0x1FU), 1) - 1; }

__device__ __forceinline__ int base_m(std::uint32_t d, int base) {
    const int field = static_cast<int>((d >> 10) & 0x1FU);
    const int mant  = field == 0 ? static_cast<int>(d & 0x3FFU) : static_cast<int>(0x400U | (d & 0x3FFU));
    const int m     = mant << (max(field, 1) - 1 - base);
    return (d & 0x8000U) != 0 ? -m : m;
}

// Column c's B fragment of the step at stage slice s (stage s0) through `xpair(c, slice, word)`: column c's
// limb words `word` and word + kWordStep<F> of absolute slice `slice`.
template <QType F, class XPair>
__device__ __forceinline__ int2 b_frag(XPair& xpair, int c, int s0, int s, int lane) {
    const int g = lane / 4, t = lane % 4;
    const int limb = b_limb<F>(g);
    if (limb < 0) { return make_int2(0, 0); }
    return xpair(c, s0 + s + b_slice<F>(g), 8 * limb + b_word0<F>(t));
}

// One stage of a warp's 8 rows: `row` this thread's row g in the stage (its first block at the start), Slices
// slices from slice s0 of K; `bfrag(step, c)` the step's B fragment of column c. Adds into `acc` (blk: the
// 8-slice formats' block sums, carried across steps); Base accumulation for the one-block-per-step formats.
template <QType F, int C, int Slices, bool Base, class Codebook, class BFrag>
__device__ __forceinline__ void stage_steps(const std::uint8_t* row, int lane, const Codebook& cb, BFrag bfrag, Rows<C>& acc,
                                            int (&blk)[C]) {
    const int t = lane % 4;
    static_assert(Slices % kStepSlices<F> == 0 && (kBlockSlices<F> <= kStepSlices<F> || Slices % kBlockSlices<F> == 0),
                  "a stage holds whole steps and whole blocks");
    static_assert(!Base || kBlockSlices<F> <= kStepSlices<F>, "base accumulation: one block or two per step");
#pragma unroll
    for (int s = 0; s < Slices; s += kStepSlices<F>) {
        int2 b[C];
#pragma unroll
        for (int c = 0; c < C; ++c) { b[c] = bfrag(s / kStepSlices<F>, c); }
        const Step st = step<F>(row, s, t, cb);
        int m0 = 0, m1 = 0; // Base: the step's block multipliers (the second slot's too for IQ4_NL)
        if constexpr (Base) {
            m0 = base_m(lds16(row + (s / kBlockSlices<F>) * kBlockBytes<F>), acc.base);
            if constexpr (kBlockSlices<F> == 1) { m1 = base_m(lds16(row + (s + 1) * kBlockBytes<F>), acc.base); }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            int d[4];
            tile::mma_k32(d, st.a[0], st.a[1], st.a[2], st.a[3], b[c].x, b[c].y);
            const int lo = d[0], hi = kHalves<F> ? d[2] : d[3];
            if constexpr (Base && kBlockSlices<F> == 1) {
                acc.a[c] += static_cast<long long>(m0) * lo + static_cast<long long>(m1) * hi;
            } else if constexpr (Base) {
                acc.a[c] += static_cast<long long>(m0) * (lo + hi);
            } else if constexpr (kBlockSlices<F> == 1) {
                // A block per slice (IQ4_NL): each slot's own scale.
                add_term(acc.r[c], lds16(row + s * kBlockBytes<F>), lo);
                add_term(acc.r[c], lds16(row + (s + 1) * kBlockBytes<F>), hi);
            } else if constexpr (kBlockSlices<F> == kStepSlices<F>) {
                // A block per step (Q2_0): unit sub-scales.
                add_term(acc.r[c], lds16(row + (s / kBlockSlices<F>) * kBlockBytes<F>), static_cast<long long>(lo) + hi);
            } else {
                blk[c] += st.sub0 * lo + st.sub1 * hi;
                if ((s + kStepSlices<F>) % kBlockSlices<F> == 0) {
                    add_term(acc.r[c], lds16(row + (s / kBlockSlices<F>) * kBlockBytes<F>), blk[c]);
                    blk[c] = 0;
                }
            }
        }
    }
}

// Thread (g, 0)'s full int128 of row g, column c: the three limb planes (threads t = 0, 1, 2) combined as
// a * 2^16 + b * 2^8 + c. Every lane calls it; only t == 0's result is the row's.
__device__ __forceinline__ I128 combine_limbs(const I128& r) {
    const auto lo1 = __shfl_down_sync(0xFFFFFFFFU, static_cast<unsigned long long>(r.lo), 1);
    const auto hi1 = __shfl_down_sync(0xFFFFFFFFU, static_cast<long long>(r.hi), 1);
    const auto lo2 = __shfl_down_sync(0xFFFFFFFFU, static_cast<unsigned long long>(r.lo), 2);
    const auto hi2 = __shfl_down_sync(0xFFFFFFFFU, static_cast<long long>(r.hi), 2);
    const auto shl = [](std::uint64_t lo, std::int64_t hi, int s) {
        I128 o;
        o.hi = static_cast<std::int64_t>((static_cast<std::uint64_t>(hi) << s) | (lo >> (64 - s)));
        o.lo = lo << s;
        return o;
    };
    I128 out = shl(r.lo, r.hi, 16);
    add(out, shl(lo1, hi1, 8));
    add(out, I128{lo2, hi2});
    return out;
}

// --------------------------------------------------------------------------- a warp's ring

// Bytes of one row's piece of a stage of `bytes` bytes: its 16-byte-aligned superset, an odd number of 16-byte
// chunks, so the warp's 8 rows read at one offset fall in 8 different banks (an even count of chunks put rows
// of 128 bytes in one bank: an 8-way conflict on every decode load).
__host__ __device__ constexpr int piece_bytes(int bytes) {
    const int chunks = (bytes + 15 + 15) / 16;
    return 16 * (chunks % 2 == 0 ? chunks + 1 : chunks);
}

// Issues the cp.async copies of one stage of the warp's 8 rows (`row_of(r)`: row r's global start;
// `offset`: the stage's first byte within a row, `bytes` its length) into `stage` (8 pieces of Piece bytes),
// each piece holding its row's bytes at their global address modulo 16.
template <int Piece, class RowOf>
__device__ __forceinline__ void issue_stage(std::uint8_t* stage, RowOf row_of, std::size_t offset, int bytes, int lane) {
    constexpr int kChunks = Piece / 16;
    for (int i = lane; i < 8 * kChunks; i += 32) {
        const int r = i / kChunks, k = i % kChunks;
        const std::uint8_t* src = row_of(r) + offset;
        const auto at           = reinterpret_cast<std::uintptr_t>(src);
        const auto* aligned     = reinterpret_cast<const std::uint8_t*>(at & ~std::uintptr_t{15});
        const int need          = static_cast<int>(at & 15U) + bytes - 16 * k;
        if (need > 0) { tile::cp_async16(stage + r * Piece + 16 * k, aligned + 16 * k, need < 16 ? need : 16); }
    }
}

// Row g's bytes in a stage issued by issue_stage.
template <int Piece, class RowOf>
__device__ __forceinline__ const std::uint8_t* stage_row(const std::uint8_t* stage, RowOf row_of, std::size_t offset, int g) {
    return stage + g * Piece + static_cast<int>(reinterpret_cast<std::uintptr_t>(row_of(g) + offset) & 15U);
}

// A warp's ring of Depth stages of 8 rows' pieces, each stage Slices slices (StageBytes bytes a row) of a
// K of Stages stages.
template <int Slices, int Stages, int Depth, int StageBytes>
struct Ring {
    static constexpr int kPiece = piece_bytes(StageBytes);
    static constexpr int kStage = 8 * kPiece;
    static constexpr int kBytes = Depth * kStage;

    // The first Depth - 1 stages' copies, one commit group each.
    template <class RowOf>
    __device__ __forceinline__ static void prologue(std::uint8_t* ring, RowOf row_of, int lane) {
#pragma unroll
        for (int k = 0; k < Depth - 1; ++k) {
            if (k < Stages) { issue_stage<kPiece>(ring + k * kStage, row_of, static_cast<std::size_t>(k) * StageBytes, StageBytes, lane); }
            tile::cp_async_commit();
        }
    }

    // The K loop after prologue(): each column's int128 of limb plane t into `acc`. `xprefetch(c, slice, word)`
    // prefetches the line of a B fragment into L1: each stage prefetches the next stage's, so the MMAs do not
    // wait on x from L2 (loading the fragments a stage ahead into registers cost a third of the occupancy).
    template <QType F, int C, bool Base, class Codebook, class RowOf, class XPair, class XPrefetch>
    __device__ __forceinline__ static void body(std::uint8_t* ring, RowOf row_of, const Codebook& cb, XPair xpair,
                                                XPrefetch xprefetch, int lane, Rows<C>& acc) {
        int blk[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            acc.r[c] = I128{};
            acc.a[c] = 0;
            blk[c]   = 0;
        }
        const int g = lane / 4, t = lane % 4;
        const int limb = b_limb<F>(g), bs = b_slice<F>(g);
        const auto prefetch_stage = [&](int s0) {
            if (limb < 0) { return; }
#pragma unroll
            for (int s = 0; s < Slices; s += kStepSlices<F>) {
#pragma unroll
                for (int c = 0; c < C; ++c) { xprefetch(c, s0 + s + bs, 8 * limb + b_word0<F>(t)); }
            }
        };
        prefetch_stage(0);
        for (int st = 0; st < Stages; ++st) {
            tile::cp_async_wait<Depth - 2>();
            __syncwarp();
            const int next = st + Depth - 1;
            if (next < Stages) {
                issue_stage<kPiece>(ring + (next % Depth) * kStage, row_of, static_cast<std::size_t>(next) * StageBytes,
                                    StageBytes, lane);
            }
            tile::cp_async_commit();
            if (st + 1 < Stages) { prefetch_stage((st + 1) * Slices); }
            const std::uint8_t* row =
                stage_row<kPiece>(ring + (st % Depth) * kStage, row_of, static_cast<std::size_t>(st) * StageBytes, g);
            const int s0 = st * Slices;
            stage_steps<F, C, Slices, Base>(
                row, lane, cb, [&](int i, int c) { return b_frag<F>(xpair, c, s0, i * kStepSlices<F>, lane); }, acc, blk);
            __syncwarp(); // every lane is done with this stage before a later issue refills it
        }
        if constexpr (Base) {
#pragma unroll
            for (int c = 0; c < C; ++c) { add_shifted(acc.r[c], acc.a[c], acc.base); }
        }
    }

    // A resident ring (Depth == Stages): every stage's copies in one commit group, kept for several column
    // passes. The caller waits for the group (cp_async_wait<0>, __syncwarp) before resident().
    template <class RowOf>
    __device__ __forceinline__ static void issue_all(std::uint8_t* ring, RowOf row_of, int lane) {
        static_assert(Depth == Stages, "a resident ring holds every stage");
#pragma unroll
        for (int k = 0; k < Stages; ++k) {
            issue_stage<kPiece>(ring + k * kStage, row_of, static_cast<std::size_t>(k) * StageBytes, StageBytes, lane);
        }
        tile::cp_async_commit();
    }

    // Row g's smallest block scale shift over the resident stages into `base`, and whether every block of the
    // warp's 8 rows is within kBaseSpan of its row's (a warp vote): the condition of Base accumulation.
    template <QType F, class RowOf>
    __device__ __forceinline__ static bool row_base(const std::uint8_t* ring, RowOf row_of, int lane, int& base) {
        constexpr int kStageBlocks = Slices / kBlockSlices<F>;
        const int g = lane / 4, t = lane % 4;
        int lo = 31, hi = 0;
#pragma unroll
        for (int st = 0; st < Stages; ++st) {
            const std::uint8_t* row = stage_row<kPiece>(ring + st * kStage, row_of, static_cast<std::size_t>(st) * StageBytes, g);
            for (int b = t; b < kStageBlocks; b += 4) {
                const int s = scale_shift(lds16(row + b * kBlockBytes<F>));
                lo = min(lo, s);
                hi = max(hi, s);
            }
        }
#pragma unroll
        for (int offset = 1; offset < 4; offset <<= 1) {
            lo = min(lo, __shfl_xor_sync(0xFFFFFFFFU, lo, offset));
            hi = max(hi, __shfl_xor_sync(0xFFFFFFFFU, hi, offset));
        }
        base = lo;
        return __all_sync(0xFFFFFFFFU, hi - lo <= kBaseSpan) != 0;
    }

    // The K loop over a resident ring: each column's int128 of limb plane t into `acc` (Base: accumulated in
    // int64 units of 2^acc.base, which the caller set from row_base, then converted).
    template <QType F, int C, bool Base, class Codebook, class RowOf, class XPair>
    __device__ __forceinline__ static void resident(const std::uint8_t* ring, RowOf row_of, const Codebook& cb, XPair xpair,
                                                    int lane, Rows<C>& acc) {
        int blk[C];
#pragma unroll
        for (int c = 0; c < C; ++c) {
            acc.r[c] = I128{};
            acc.a[c] = 0;
            blk[c]   = 0;
        }
        const int g = lane / 4;
#pragma unroll
        for (int st = 0; st < Stages; ++st) {
            const std::uint8_t* row = stage_row<kPiece>(ring + st * kStage, row_of, static_cast<std::size_t>(st) * StageBytes, g);
            const int s0 = st * Slices;
            stage_steps<F, C, Slices, Base>(
                row, lane, cb, [&](int i, int c) { return b_frag<F>(xpair, c, s0, i * kStepSlices<F>, lane); }, acc, blk);
        }
        if constexpr (Base) {
#pragma unroll
            for (int c = 0; c < C; ++c) { add_shifted(acc.r[c], acc.a[c], acc.base); }
        }
    }
};

} // namespace infernix::ggml::nmma
