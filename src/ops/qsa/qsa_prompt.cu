// QSA prompt attention on Tensor Cores (see qsa_prompt.h).

#include "ops/qsa/qsa_prompt.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/q4_lloyd_codec.cuh"
#include "ops/kv_cache/vq2_codec.cuh"
#include "ops/qsa/page_spaces.cuh"
#include "ops/qsa/vq_window.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kD      = 256; // head dimension
constexpr int kTile   = 16;  // tokens per warp tile
constexpr int kRows   = 16;  // MMA rows: one KV head's query heads, padded
constexpr int kQRow   = 2 * kD; // bytes of one FP16/BF16 query row in shared memory

// How a stored row enters the MMAs. Keys: BF16 rows on BF16 Tensor Cores; 8-bit codes (INT8, FP8
// E4M3) widened exactly to FP16 with their scales applied to the FP32 partial of each scale group;
// NVFP4 codes decoded exactly to FP16 values (E2M1 times E4M3 is exact in FP16). Values: FP16 rows;
// 8-bit codes widened and multiplied once by their key's scale in FP16 (even/odd dimension tiles);
// NVFP4 rows decoded exactly to an FP16 tile in shared memory.
enum class Code { Bf16, Fp16, Int8, Fp8, Nvfp4 };

template <Code C>
struct Row;
template <>
struct Row<Code::Bf16> {
    static constexpr int Bytes = 2 * kD, ScaleSlot = 0;
};
template <>
struct Row<Code::Fp16> {
    static constexpr int Bytes = 2 * kD, ScaleSlot = 0;
};
template <>
struct Row<Code::Int8> {
    static constexpr int Bytes = kD, ScaleSlot = 8, Groups = 4;
    static constexpr float MaxScale = 256.0f; // 127 x 256 stays finite in FP16
};
// One FP16 row scale, gathered as the aligned 4-byte word holding it; its half is the token's parity.
template <>
struct Row<Code::Fp8> {
    static constexpr int Bytes = kD, ScaleSlot = 4, Groups = 1;
    static constexpr float MaxScale = 64.0f; // 448 x 64 stays finite in FP16
};
template <>
struct Row<Code::Nvfp4> {
    static constexpr int Bytes = kD / 2, ScaleSlot = 16;
};

template <KvCacheStorage Storage>
struct PromptShape;

template <>
struct PromptShape<KvCacheStorage::Int8Group64> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 4;
    static constexpr Code Key = Code::Int8, Value = Code::Int8;
};
template <>
struct PromptShape<KvCacheStorage::BFloat16> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 2;
    static constexpr Code Key = Code::Bf16, Value = Code::Fp16;
};
template <>
struct PromptShape<KvCacheStorage::Fp8E4M3Row256> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 4;
    static constexpr Code Key = Code::Fp8, Value = Code::Fp8;
};
template <>
struct PromptShape<KvCacheStorage::Nvfp4Group16> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 4;
    static constexpr Code Key = Code::Nvfp4, Value = Code::Nvfp4;
};
// Vector-quantized storages: each tile's rows become INT8-G64 rows of the rotated vectors (the window's
// exact rows, or the codes decoded with the row scale as every group's), then the INT8 path. `Vq` is
// the key code's bytes; three warps keep two stages, the codes and the codebook within one CTA.
template <>
struct PromptShape<KvCacheStorage::Vq2> {
    static constexpr int Vq    = kKVCacheVq2CodeBytes;
    static constexpr int Warps = 3;
    static constexpr Code Key = Code::Int8, Value = Code::Int8;
};
template <>
struct PromptShape<KvCacheStorage::Q4KeyVq2Value> {
    static constexpr int Vq    = kKVCacheQ4CodeBytes;
    static constexpr int Warps = 3;
    static constexpr Code Key = Code::Int8, Value = Code::Int8;
};
template <>
struct PromptShape<KvCacheStorage::Fp8KeyNvfp4Value> {
    static constexpr int Vq    = 0;
    static constexpr int Warps = 4;
    static constexpr Code Key = Code::Fp8, Value = Code::Nvfp4;
};

// One warp stage: [K rows | V rows | K scales | V scales | parity word], 16 tokens each; rows keep
// their 16-byte chunks swizzled. The parity word's bit j is token j's parity (FP8 row scales).
template <KvCacheStorage Storage>
struct Stage {
    using Shape                     = PromptShape<Storage>;
    using K                         = Row<Shape::Key>;
    using V                         = Row<Shape::Value>;
    static constexpr bool Parity    = Shape::Key == Code::Fp8 || Shape::Value == Code::Fp8 || Shape::Vq > 0;
    static constexpr int Values     = kTile * K::Bytes;
    static constexpr int KeyScales  = Values + kTile * V::Bytes;
    static constexpr int ValueScales = KeyScales + kTile * K::ScaleSlot;
    static constexpr int ParityWord = ValueScales + kTile * V::ScaleSlot;
    // Vector-quantized: the paged code rows, each row's FP16 code scale (K, V) as its aligned word,
    // the window tags (K, V) and each row's token and read mode.
    static constexpr int VqKeyCodes   = ParityWord + (Parity ? 16 : 0);
    static constexpr int VqValueCodes = VqKeyCodes + kTile * Shape::Vq;
    static constexpr int VqCodeScales = VqValueCodes + (Shape::Vq > 0 ? kTile * kKVCacheVq2CodeBytes : 0);
    static constexpr int VqTags       = VqCodeScales + (Shape::Vq > 0 ? kTile * 8 : 0);
    static constexpr int VqMeta       = VqTags + (Shape::Vq > 0 ? kTile * 8 : 0);
    static constexpr int Bytes        = VqMeta + (Shape::Vq > 0 ? kTile * 8 : 0);
    // NVFP4 values decode into one FP16 [16][256] tile per warp.
    static constexpr int Scratch = Shape::Value == Code::Nvfp4 ? kTile * 2 * kD : 0;
    static constexpr bool RotatedKeys   = Shape::Key != Code::Bf16;
    static constexpr bool RotatedValues = Shape::Value == Code::Nvfp4 || Shape::Vq > 0;
    static constexpr bool EvenOddValues = Shape::Value == Code::Int8 || Shape::Value == Code::Fp8;
    static_assert(Values % 16 == 0 && KeyScales % 16 == 0 && ValueScales % 16 == 0 && ParityWord % 16 == 0);
};

template <KvCacheStorage Storage>
__host__ __device__ constexpr int prompt_smem_bytes() {
    using S              = Stage<Storage>;
    constexpr int Warps  = PromptShape<Storage>::Warps;
    constexpr int stages = Warps * (2 * S::Bytes + S::Scratch);
    constexpr int merge  = Warps * (kRows * kD + kRows * 2) * 4 + (S::RotatedValues ? kRows * kD * 4 : 0);
    constexpr int book   = PromptShape<Storage>::Vq > 0 ? static_cast<int>(sizeof(kKVCacheVq2Codebook)) : 0;
    return kQRow * kRows + (stages > merge ? stages : merge) + book;
}

// 16-byte chunk c of row r lives at chunk c ^ (r & 7): the eight rows an ldmatrix reads hit
// distinct banks.
__device__ __forceinline__ int swizzled(int row, int row_bytes, int chunk) {
    return row * row_bytes + ((chunk ^ (row & 7)) << 4);
}

// The attended token at index i of a column's list (the decode kernel's order: selected blocks in
// ascending order, then the incomplete block's positions; every token 0..p while dense).
__device__ __forceinline__ int attended_token(int i, int count, const std::int32_t* list, int ratio, int p) {
    if (count < 0) { return i; }
    const int block_tokens = count * ratio;
    if (i < block_tokens) { return list[i / ratio] * ratio + i % ratio; }
    return (p + 1) / ratio * ratio + (i - block_tokens);
}

// 8-bit key rows: the k positions {2t, 2t+1 | 2t+8, 2t+9} of a 16-wide MMA k step hold dimensions
// {4t, 4t+1 | 4t+2, 4t+3}, which is what one ldmatrix lane receives of a code row (four
// consecutive bytes). The query rows are stored in that order, so the dot product is unchanged.
__device__ __forceinline__ int int8_query_position(int d) {
    const int local = d & 15;
    const int p     = ((local & 2) != 0 ? 8 : 0) + 2 * (local >> 2) + (local & 1);
    return (d & ~15) + p;
}

// NVFP4 key rows: one ldmatrix lane receives eight dimensions 32c + 8t .. + 7 of a code row (four
// bytes); k step 2c takes the first four at positions {2t, 2t+1 | 2t+8, 2t+9} and k step 2c + 1
// the last four.
__device__ __forceinline__ int nvfp4_query_position(int d) {
    const int c = d >> 5, t = (d >> 3) & 3, w = d & 7;
    const int step = 2 * c + (w >> 2), u = w & 3;
    return 16 * step + 2 * t + ((u & 2) != 0 ? 8 : 0) + (u & 1);
}

template <Code C>
__device__ __forceinline__ int query_position(int d) {
    if constexpr (C == Code::Int8 || C == Code::Fp8) { return int8_query_position(d); }
    if constexpr (C == Code::Nvfp4) { return nvfp4_query_position(d); }
    return d;
}

// Four INT8 codes (dimensions d..d+3 of one key) as the two FP16 B registers of a k step:
// {c0, c1} and {c2, c3}, widened exactly.
__device__ __forceinline__ void widen_codes(unsigned codes, unsigned& lo, unsigned& hi) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte.
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned e      = __byte_perm(biased, 0x64646464u, 0x5140);
    const unsigned o      = __byte_perm(biased, 0x64646464u, 0x7362);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 l       = __hsub2(load_vec<__half2>(&e), offset);
    const __half2 h       = __hsub2(load_vec<__half2>(&o), offset);
    lo                    = load_vec<unsigned>(&l);
    hi                    = load_vec<unsigned>(&h);
}

// Two E4M3 codes (low byte first) as an FP16 pair, exactly.
__device__ __forceinline__ unsigned e4m3x2_to_f16x2(unsigned pair) {
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
    return static_cast<unsigned>(raw.x) | (static_cast<unsigned>(raw.y) << 16);
}

// Four E4M3 codes (dimensions d..d+3 of one key) as {c0, c1} and {c2, c3}, exactly.
__device__ __forceinline__ void widen_e4m3(unsigned codes, unsigned& lo, unsigned& hi) {
    lo = e4m3x2_to_f16x2(codes & 0xffffu);
    hi = e4m3x2_to_f16x2(codes >> 16);
}

// Two E2M1 codes (low nibble first) times an FP16 scale pair, exactly.
__device__ __forceinline__ unsigned e2m1x2_times(unsigned byte, __half2 scale) {
    __nv_fp4x2_e2m1 codes;
    codes.__x       = static_cast<__nv_fp4x2_storage_t>(byte);
    const __half2 v = __hmul2(static_cast<__half2>(codes), scale);
    return load_vec<unsigned>(&v);
}

__device__ __forceinline__ __half e4m3_to_half(unsigned char code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return static_cast<__half>(v);
}

// One ldmatrix.trans b16 lane of an 8-bit [key][d] tile holds {V[k][d], V[k][d+1], V[k+1][d],
// V[k+1][d+1]}: the FP16 B halves {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code
// widened exactly and multiplied once by its key's scale (`scales` = {s_k, s_k+1}).
template <Code C>
__device__ __forceinline__ void decode_v_pair(unsigned codes, __half2 scales, unsigned& even, unsigned& odd) {
    if constexpr (C == Code::Int8) {
        const unsigned biased = codes ^ 0x80808080u;
        const unsigned e      = __byte_perm(biased, 0x64646464u, 0x4240);
        const unsigned o      = __byte_perm(biased, 0x64646464u, 0x4341);
        const __half2 offset  = __float2half2_rn(1152.0f);
        const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), scales);
        const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), scales);
        even                  = load_vec<unsigned>(&ve);
        odd                   = load_vec<unsigned>(&vo);
    } else {
        const unsigned e  = e4m3x2_to_f16x2(__byte_perm(codes, 0, 0x0020));
        const unsigned o  = e4m3x2_to_f16x2(__byte_perm(codes, 0, 0x0031));
        const __half2 ve = __hmul2(load_vec<__half2>(&e), scales);
        const __half2 vo = __hmul2(load_vec<__half2>(&o), scales);
        even              = load_vec<unsigned>(&ve);
        odd               = load_vec<unsigned>(&vo);
    }
}

template <KvCacheStorage Storage>
__global__ void __launch_bounds__(PromptShape<Storage>::Warps * 32, 1)
    qsa_prompt_kernel(const bf16* __restrict__ q, int heads, int kv_heads, const void* __restrict__ k_pages,
                      const void* __restrict__ v_pages, const void* __restrict__ k_scales,
                      const void* __restrict__ v_scales, const std::int32_t* __restrict__ tables,
                      int table_stride, const std::int32_t* __restrict__ table_rows,
                      const std::int32_t* __restrict__ positions, int width, int ratio,
                      const std::int32_t* __restrict__ selected, const std::int32_t* __restrict__ counts,
                      int top_blocks, float scale, bf16* __restrict__ out, QsaPageSpaces spaces, QsaVqWindow vq) {
    using Shape                 = PromptShape<Storage>;
    using S                     = Stage<Storage>;
    using K                     = Row<Shape::Key>;
    using V                     = Row<Shape::Value>;
    constexpr int Warps         = Shape::Warps;
    constexpr int Threads       = Warps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    extern __shared__ __align__(16) unsigned char smem[];

    const int t = static_cast<int>(blockIdx.x), kv_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, lid = lane & 3;
    const int group = heads / kv_heads;
    const int p     = positions[t];
    const int count = counts[t];
    const int total = count < 0 ? p + 1 : count * ratio + (p + 1) % ratio;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[t / width]) * table_stride;
    const std::int32_t* list  = selected + static_cast<std::size_t>(t) * top_blocks;

    // Query rows (rotated into the keys' Hadamard domain where keys are rotated, then FP16 in the
    // key code's k-step order; BF16 keys: the stored values). Rows past the group are zero.
    unsigned char* q_s = smem;
    for (int r = warp; r < kRows; r += Warps) {
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = r < group ? __bfloat162float(
                                   q[(static_cast<std::size_t>(t) * heads + kv_head * group + r) * kD + lane + 32 * k])
                             : 0.0f;
        }
        if constexpr (S::RotatedKeys) { normalized_hadamard_d256_inplace(v, lane); }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int d = lane + 32 * k;
            const int e = query_position<Shape::Key>(d);
            unsigned char* at = q_s + swizzled(r, kQRow, e >> 3) + 2 * (e & 7);
            if constexpr (Shape::Key == Code::Bf16) {
                *reinterpret_cast<bf16*>(at) = __float2bfloat16_rn(v[k]);
            } else {
                *reinterpret_cast<__half*>(at) = __float2half_rn(v[k]);
            }
        }
    }
    __syncthreads();

    // VQ2 codebook (vector-quantized storages): behind the stages and the merge rows.
    constexpr int kCodebookAt = prompt_smem_bytes<Storage>() -
                                (Shape::Vq > 0 ? static_cast<int>(sizeof(kKVCacheVq2Codebook)) : 0);
    const auto* codebook = reinterpret_cast<const std::int8_t*>(smem + kCodebookAt);
    if constexpr (Shape::Vq > 0) {
        for (int i = tid; i < static_cast<int>(sizeof(kKVCacheVq2Codebook)) / 16; i += Threads) {
            reinterpret_cast<uint4*>(const_cast<std::int8_t*>(codebook))[i] =
                reinterpret_cast<const uint4*>(g_kv_cache_vq2_codebook)[i];
        }
        __syncthreads();
    }
    const int call_first = positions[(t / width) * width];
    const int window_row = vq.slots != nullptr ? vq.slots[t / width] : 0;
    unsigned char* stages  = smem + kQRow * kRows + warp * (2 * S::Bytes + S::Scratch);
    unsigned char* scratch = stages + 2 * S::Bytes;
    const int tiles        = (total + kTile - 1) / kTile;

    // Gathers tile i's keys, values and scales into stage s; tokens past the column's list are
    // zero-filled. Lane l locates row l % 16 once (its token, page space and row in the planes); the
    // lanes copying a row's chunks take its addresses by shuffle.
    const auto issue = [&](int i, int s) {
        unsigned char* k_s = stages + s * S::Bytes;
        unsigned char* v_s = k_s + S::Values;
        const int row      = lane & 15;
        const int index    = i * kTile + row;
        const bool valid   = index < total;
        const int token    = valid ? attended_token(index, count, list, ratio, p) : 0;
        const int id       = table[token >> kPagedKVPageShift];
        int page;
        const auto* k_plane =
            static_cast<const unsigned char*>(qsa_space_plane(spaces, k_pages, spaces.host_k, spaces.lent_k, id, page));
        const auto* v_plane =
            static_cast<const unsigned char*>(qsa_space_plane(spaces, v_pages, spaces.host_v, spaces.lent_v, id, page));
        const std::size_t row_at =
            static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize + (token & kPagedKVPageMask);
        // Copies every row's `Bytes` from its lane's `own_row` (zero-filled unless `own_ok`), with the
        // 16-byte chunks swizzled for ldmatrix or in order.
        const auto copy_rows = [&](unsigned char* dst_rows, const void* own_row, bool own_ok, auto bytes, auto swizzle) {
            constexpr int Bytes  = decltype(bytes)::value;
            constexpr int Chunks = Bytes / 16;
            const auto own       = reinterpret_cast<unsigned long long>(own_row);
#pragma unroll
            for (int j = 0; j < kTile * Chunks / 32; ++j) {
                const int chunk = lane + 32 * j;
                const int r = chunk / Chunks, c = chunk % Chunks;
                const auto* src = reinterpret_cast<const unsigned char*>(__shfl_sync(FullMask, own, r));
                const bool ok   = __shfl_sync(FullMask, static_cast<int>(own_ok), r) != 0;
                unsigned char* dst = decltype(swizzle)::value ? dst_rows + swizzled(r, Bytes, c) : dst_rows + r * Bytes + 16 * c;
                cp_async_zfill<16, Cache::cg>(dst, src + 16 * c, ok ? 16 : 0);
            }
        };
        const auto rows = [&](unsigned char* dst_rows, const unsigned char* own_row, auto bytes) {
            copy_rows(dst_rows, own_row, valid, bytes, std::true_type{});
        };
        if constexpr (Shape::Vq > 0) {
            // Paged code rows; the exact rows (window slot or staged call row) of the keys this column
            // reads exactly, with their group scales and window tags; the code scales; token and mode.
            copy_rows(k_s + S::VqKeyCodes, k_plane + row_at * Shape::Vq, valid,
                      std::integral_constant<int, Shape::Vq>{}, std::false_type{});
            copy_rows(k_s + S::VqValueCodes, v_plane + row_at * kKVCacheVq2CodeBytes, valid,
                      std::integral_constant<int, kKVCacheVq2CodeBytes>{}, std::false_type{});
            const int mode = valid ? vq_read_mode(vq, token, p, call_first) : 0;
            std::int64_t exact = 0;
            if (mode == 2) { exact = vq_staged_row(vq, kv_head, token - call_first); }
            if (mode == 1) { exact = vq_window_row(window_row, kv_head, kv_heads, kv_window_slot(token)); }
            const std::int8_t* ek = mode == 2 ? vq.staged_k_codes : vq.k_codes;
            const std::int8_t* ev = mode == 2 ? vq.staged_v_codes : vq.v_codes;
            copy_rows(k_s, mode != 0 ? static_cast<const void*>(ek + exact * kD) : k_plane, mode != 0,
                      std::integral_constant<int, kD>{}, std::true_type{});
            copy_rows(v_s, mode != 0 ? static_cast<const void*>(ev + exact * kD) : v_plane, mode != 0,
                      std::integral_constant<int, kD>{}, std::true_type{});
            const int role = lane >> 4;
            int scale_page;
            const auto* code_scales = static_cast<const unsigned char*>(
                role ? qsa_space_plane(spaces, v_scales, spaces.host_v_scale, spaces.lent_v_scale, id, scale_page)
                     : qsa_space_plane(spaces, k_scales, spaces.host_k_scale, spaces.lent_k_scale, id, scale_page));
            cp_async_zfill<4>(k_s + S::VqCodeScales + (2 * row + role) * 4, code_scales + (2 * row_at & ~std::size_t{3}),
                              valid ? 4 : 0);
            if (mode != 0) {
                const __half* groups = mode == 2 ? (role ? vq.staged_v_scales : vq.staged_k_scales)
                                                 : (role ? vq.v_scales : vq.k_scales);
                cp_async<8>(k_s + (role ? S::ValueScales : S::KeyScales) + row * 8, groups + exact * kKVWindowGroups);
            }
            if (mode == 1) { cp_async<4>(k_s + S::VqTags + (2 * row + role) * 4, vq.tags + exact * 2 + role); }
            if (role == 0) {
                reinterpret_cast<int*>(k_s + S::VqMeta)[2 * row]     = token;
                reinterpret_cast<int*>(k_s + S::VqMeta)[2 * row + 1] = mode;
            }
            const unsigned odd = __ballot_sync(FullMask, role == 0 && (token & 1) != 0);
            if (lane == 0) { *reinterpret_cast<unsigned*>(k_s + S::ParityWord) = odd; }
            cp_commit();
            return;
        }
        rows(k_s, k_plane + row_at * K::Bytes, std::integral_constant<int, K::Bytes>{});
        rows(v_s, v_plane + row_at * V::Bytes, std::integral_constant<int, V::Bytes>{});
        if constexpr (K::ScaleSlot > 0 || V::ScaleSlot > 0) {
            // Lanes 0-15 gather their row's key scales, lanes 16-31 its value scales.
            const bool key    = lane < 16;
            const auto scales = [&](const void* pool, const void* host, const void* lent, int slot_bytes,
                                    unsigned char* dst) {
                int scale_page;
                const auto* base = static_cast<const unsigned char*>(qsa_space_plane(spaces, pool, host, lent, id, scale_page));
                if (slot_bytes == 4) { // one FP16: its aligned word
                    cp_async_zfill<4>(dst, base + (2 * row_at & ~std::size_t{3}), valid ? 4 : 0);
                } else if (slot_bytes == 8) {
                    cp_async_zfill<8>(dst, base + 8 * row_at, valid ? 8 : 0);
                } else {
                    cp_async_zfill<16>(dst, base + 16 * row_at, valid ? 16 : 0);
                }
            };
            if (key) {
                if constexpr (K::ScaleSlot > 0) {
                    scales(k_scales, spaces.host_k_scale, spaces.lent_k_scale, K::ScaleSlot,
                           k_s + S::KeyScales + row * K::ScaleSlot);
                }
            } else {
                if constexpr (V::ScaleSlot > 0) {
                    scales(v_scales, spaces.host_v_scale, spaces.lent_v_scale, V::ScaleSlot,
                           k_s + S::ValueScales + row * V::ScaleSlot);
                }
            }
            if constexpr (S::Parity) {
                const unsigned odd = __ballot_sync(FullMask, key && (token & 1) != 0);
                if (lane == 0) { *reinterpret_cast<unsigned*>(k_s + S::ParityWord) = odd; }
            }
        }
        cp_commit();
    };

    // The A fragments of k step kb of the query rows.
    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = a_rin + ((a_mat & 1) << 3);
    const auto query_fragment = [&](int kb, unsigned (&a)[4]) {
        ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(q_s + swizzled(a_row, kQRow, 2 * kb + (a_mat >> 1))));
    };

    // Output rows g (registers 0/1) and g + 8 (2/3) of 32 n8 tiles: 8-bit value tiles [db][0] and
    // [db][1] hold the even and odd dimensions of block db; FP16 tiles [db][h] dimensions
    // 16 db + 8 h .. + 7.
    float o[16][2][4];
#pragma unroll
    for (int b = 0; b < 16; ++b) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { o[b][h][i] = 0.0f; }
        }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const float scale_l2 = scale * Log2E;

    const auto compute = [&](int i, int s) {
        const unsigned char* k_s = stages + s * S::Bytes;
        const unsigned char* v_s = k_s + S::Values;
        unsigned parity          = 0;
        if constexpr (S::Parity) { parity = *reinterpret_cast<const unsigned*>(k_s + S::ParityWord); }
        // FP16 row scale of tile token `key` from its gathered word.
        const auto row_scale = [&](int scales_at, int key) {
            return reinterpret_cast<const __half*>(k_s + scales_at)[2 * key + ((parity >> key) & 1)];
        };
        float score[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { score[nt][j] = 0.0f; }
        }
        if constexpr (Shape::Key == Code::Int8 || Shape::Key == Code::Fp8) {
            constexpr int Groups = K::Groups, Steps = 16 / Groups;
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                float acc[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { acc[nt][j] = 0.0f; }
                }
#pragma unroll
                for (int kk = 0; kk < Steps; ++kk) {
                    const int kb = g * Steps + kk;
                    unsigned a[4];
                    query_fragment(kb, a);
                    unsigned r0, r1;
                    ldmatrix_x2(r0, r1, smem_addr(k_s + swizzled(lane & 15, K::Bytes, kb)));
                    unsigned lo, hi;
                    if constexpr (Shape::Key == Code::Int8) {
                        widen_codes(r0, lo, hi);
                    } else {
                        widen_e4m3(r0, lo, hi);
                    }
                    mma_f16(acc[0][0], acc[0][1], acc[0][2], acc[0][3], a[0], a[1], a[2], a[3], lo, hi);
                    if constexpr (Shape::Key == Code::Int8) {
                        widen_codes(r1, lo, hi);
                    } else {
                        widen_e4m3(r1, lo, hi);
                    }
                    mma_f16(acc[1][0], acc[1][1], acc[1][2], acc[1][3], a[0], a[1], a[2], a[3], lo, hi);
                }
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int key = nt * 8 + 2 * lid;
                    float ks0, ks1;
                    if constexpr (Shape::Key == Code::Int8) {
                        const __half* ks = reinterpret_cast<const __half*>(k_s + S::KeyScales);
                        ks0              = __half2float(ks[key * Groups + g]);
                        ks1              = __half2float(ks[(key + 1) * Groups + g]);
                    } else {
                        ks0 = __half2float(row_scale(S::KeyScales, key));
                        ks1 = __half2float(row_scale(S::KeyScales, key + 1));
                    }
                    score[nt][0] = __fmaf_rn(ks0, acc[nt][0], score[nt][0]);
                    score[nt][1] = __fmaf_rn(ks1, acc[nt][1], score[nt][1]);
                    score[nt][2] = __fmaf_rn(ks0, acc[nt][2], score[nt][2]);
                    score[nt][3] = __fmaf_rn(ks1, acc[nt][3], score[nt][3]);
                }
            }
        } else if constexpr (Shape::Key == Code::Nvfp4) {
            // Lane (gid, lid) decodes dimensions 32c + 8 lid .. + 7 of keys gid and gid + 8 with their
            // group scale 2c + lid / 2: exact FP16 values, no per-group partials.
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                unsigned r[2];
                ldmatrix_x2(r[0], r[1], smem_addr(k_s + swizzled(lane & 15, K::Bytes, c)));
                unsigned b[2][4];
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int key   = nt * 8 + gid;
                    const __half sc = e4m3_to_half(k_s[S::KeyScales + key * K::ScaleSlot + 2 * c + (lid >> 1)]);
                    const __half2 s2 = __halves2half2(sc, sc);
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) { b[nt][byte] = e2m1x2_times((r[nt] >> (8 * byte)) & 0xffu, s2); }
                }
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    unsigned a[4];
                    query_fragment(2 * c + half, a);
                    mma_f16(score[0][0], score[0][1], score[0][2], score[0][3], a[0], a[1], a[2], a[3], b[0][2 * half],
                            b[0][2 * half + 1]);
                    mma_f16(score[1][0], score[1][1], score[1][2], score[1][3], a[0], a[1], a[2], a[3], b[1][2 * half],
                            b[1][2 * half + 1]);
                }
            }
        } else {
            const int b_mat = lane >> 3, b_rin = lane & 7;
            const int b_row = b_rin + ((b_mat >> 1) << 3);
#pragma unroll
            for (int kb = 0; kb < 16; ++kb) {
                unsigned a[4];
                query_fragment(kb, a);
                unsigned b[4];
                ldmatrix_x4(b[0], b[1], b[2], b[3],
                            smem_addr(k_s + swizzled(b_row, K::Bytes, 2 * kb + (b_mat & 1))));
                mma_bf16(score[0][0], score[0][1], score[0][2], score[0][3], a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_bf16(score[1][0], score[1][1], score[1][2], score[1][3], a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }

        // Online softmax over the tile's 16 tokens (columns past the list are masked).
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int index = i * kTile + nt * 8 + 2 * lid;
            if (index >= total) { score[nt][0] = score[nt][2] = -CUDART_INF_F; }
            if (index + 1 >= total) { score[nt][1] = score[nt][3] = -CUDART_INF_F; }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);
        const float nm0 = fmaxf(m0, bm0), nm1 = fmaxf(m1, bm1);
        const float alpha0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - nm0) * scale_l2);
        const float alpha1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - nm1) * scale_l2);
        m0 = nm0;
        m1 = nm1;
        const float base0 = nm0 * scale_l2, base1 = nm1 * scale_l2;
        float pr[2][4];
        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pr[nt][0] = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -base0));
            pr[nt][1] = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -base0));
            pr[nt][2] = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -base1));
            pr[nt][3] = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -base1));
            bl0 += pr[nt][0] + pr[nt][1];
            bl1 += pr[nt][2] + pr[nt][3];
        }
        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
#pragma unroll
        for (int b = 0; b < 16; ++b) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                o[b][h][0] *= alpha0;
                o[b][h][1] *= alpha0;
                o[b][h][2] *= alpha1;
                o[b][h][3] *= alpha1;
            }
        }

        if constexpr (S::EvenOddValues) {
            // Decoded values stay finite in FP16: a tile whose largest scale exceeds MaxScale decodes
            // with its scales divided by 2^shift and multiplies the probabilities by 2^shift
            // (both exact, so every product is unchanged).
            constexpr int Groups = V::Groups;
            float own;
            if constexpr (Shape::Value == Code::Int8) {
                const __half* vs = reinterpret_cast<const __half*>(k_s + S::ValueScales);
                const __half2 pair = load_vec<__half2>(&vs[2 * lane]);
                own                = fmaxf(fabsf(__low2float(pair)), fabsf(__high2float(pair)));
            } else {
                own = lane < 16 ? fabsf(__half2float(row_scale(S::ValueScales, lane))) : 0.0f;
            }
            const float vmax = warp_max(own, FullMask);
            int shift        = 0;
            while (shift < 14 && ldexpf(vmax, -shift) > V::MaxScale) { ++shift; }
            const float up = ldexpf(1.0f, shift);
            unsigned pa[4];
            pa[0] = pack_f16x2(pr[0][0] * up, pr[0][1] * up);
            pa[1] = pack_f16x2(pr[0][2] * up, pr[0][3] * up);
            pa[2] = pack_f16x2(pr[1][0] * up, pr[1][1] * up);
            pa[3] = pack_f16x2(pr[1][2] * up, pr[1][3] * up);
            const __half2 down = __float2half2_rn(ldexpf(1.0f, -shift));
            const int v_mat = lane >> 3, v_rin = lane & 7;
            const int v_row = v_rin + ((v_mat & 1) << 3);
            const auto value_scale = [&](int key, int g) {
                if constexpr (Shape::Value == Code::Int8) {
                    return reinterpret_cast<const __half*>(k_s + S::ValueScales)[key * Groups + g];
                } else {
                    return row_scale(S::ValueScales, key);
                }
            };
#pragma unroll
            for (int db = 0; db < 16; db += 2) {
                unsigned r[4];
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              smem_addr(v_s + swizzled(v_row, V::Bytes, db + (v_mat >> 1))));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int g       = (db + h) * Groups / 16;
                    const int key     = 2 * lid;
                    const __half2 s01 = __hmul2(__halves2half2(value_scale(key, g), value_scale(key + 1, g)), down);
                    const __half2 s89 = __hmul2(__halves2half2(value_scale(key + 8, g), value_scale(key + 9, g)), down);
                    unsigned e0, o0, e1, o1;
                    decode_v_pair<Shape::Value>(r[2 * h], s01, e0, o0);
                    decode_v_pair<Shape::Value>(r[2 * h + 1], s89, e1, o1);
                    float* even = o[db + h][0];
                    float* odd  = o[db + h][1];
                    mma_f16(even[0], even[1], even[2], even[3], pa[0], pa[1], pa[2], pa[3], e0, e1);
                    mma_f16(odd[0], odd[1], odd[2], odd[3], pa[0], pa[1], pa[2], pa[3], o0, o1);
                }
            }
        } else {
            const unsigned char* fp16_rows = v_s;
            if constexpr (Shape::Value == Code::Nvfp4) {
                // Lane l decodes key l / 2's dimensions 128 (l % 2) .. + 127 into the FP16 tile.
                const int key = lane >> 1, half = lane & 1;
                const unsigned char* vs = k_s + S::ValueScales + key * V::ScaleSlot;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int chunk = 4 * half + c; // 16 code bytes: dimensions 32 chunk .. + 31
                    const uint4 codes = load_vec<uint4>(v_s + swizzled(key, V::Bytes, chunk));
                    const unsigned words[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
                    for (int w = 0; w < 4; ++w) { // dimensions 32 chunk + 8 w .. + 7
                        const __half sc  = e4m3_to_half(vs[2 * chunk + (w >> 1)]);
                        const __half2 s2 = __halves2half2(sc, sc);
                        uint4 values;
                        values.x = e2m1x2_times(words[w] & 0xffu, s2);
                        values.y = e2m1x2_times((words[w] >> 8) & 0xffu, s2);
                        values.z = e2m1x2_times((words[w] >> 16) & 0xffu, s2);
                        values.w = e2m1x2_times(words[w] >> 24, s2);
                        store_vec(scratch + swizzled(key, 2 * kD, 4 * chunk + w), values);
                    }
                }
                __syncwarp();
                fp16_rows = scratch;
            }
            unsigned pa[4];
            pa[0] = pack_f16x2(pr[0][0], pr[0][1]);
            pa[1] = pack_f16x2(pr[0][2], pr[0][3]);
            pa[2] = pack_f16x2(pr[1][0], pr[1][1]);
            pa[3] = pack_f16x2(pr[1][2], pr[1][3]);
            const int v_mat = lane >> 3, v_rin = lane & 7;
            const int v_row = v_rin + ((v_mat & 1) << 3);
#pragma unroll
            for (int c = 0; c < 32; c += 2) {
                unsigned r[4];
                ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                              smem_addr(fp16_rows + swizzled(v_row, 2 * kD, c + (v_mat >> 1))));
                float* first  = o[c >> 1][0];
                float* second = o[c >> 1][1];
                mma_f16(first[0], first[1], first[2], first[3], pa[0], pa[1], pa[2], pa[3], r[0], r[1]);
                mma_f16(second[0], second[1], second[2], second[3], pa[0], pa[1], pa[2], pa[3], r[2], r[3]);
            }
        }
    };

    int s = 0;
    if (warp < tiles) { issue(warp, 0); }
    for (int i = warp; i < tiles; i += Warps) {
        if (i + Warps < tiles) {
            issue(i + Warps, s ^ 1);
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncwarp();
        if constexpr (Shape::Vq > 0) {
            // Lane l takes row l % 16 of role l / 16 (K, V): its exact row stays when the read is from
            // the call's staged row or the window slot's tag matches the stored codes; otherwise the
            // codes are decoded into the INT8 row and the code scale becomes every group's scale.
            unsigned char* k_s = stages + s * S::Bytes;
            const int row = lane & 15, role = lane >> 4;
            const int token = reinterpret_cast<const int*>(k_s + S::VqMeta)[2 * row];
            const int mode  = reinterpret_cast<const int*>(k_s + S::VqMeta)[2 * row + 1];
            const unsigned parity  = (*reinterpret_cast<const unsigned*>(k_s + S::ParityWord) >> row) & 1U;
            const __half code_scale =
                reinterpret_cast<const __half*>(k_s + S::VqCodeScales + (2 * row + role) * 4)[parity];
            const unsigned char* codes = role ? k_s + S::VqValueCodes + row * kKVCacheVq2CodeBytes
                                              : k_s + S::VqKeyCodes + row * Shape::Vq;
            bool keep = mode == 2;
            if (mode == 1) {
                const std::uint32_t bits = __half_as_ushort(code_scale);
                const std::uint32_t tag =
                    role == 0 && Shape::Vq == kKVCacheQ4CodeBytes
                        ? kv_window_tag<kKVCacheQ4CodeBytes / 4>(token, reinterpret_cast<const std::uint32_t*>(codes), bits)
                        : kv_window_tag<kKVCacheVq2CodeBytes / 4>(token, reinterpret_cast<const std::uint32_t*>(codes), bits);
                keep = static_cast<std::uint32_t>(reinterpret_cast<const std::int32_t*>(k_s + S::VqTags)[2 * row + role]) == tag;
            }
            if (!keep) {
                unsigned char* rows_base = role ? k_s + S::Values : k_s;
#pragma unroll 4
                for (int w = 0; w < 32; ++w) {
                    const uint2 values = role == 0 && Shape::Vq == kKVCacheQ4CodeBytes
                                             ? kv_cache_q4_decode_word(reinterpret_cast<const std::uint32_t*>(codes)[w])
                                             : kv_cache_vq2_decode_word(reinterpret_cast<const std::uint16_t*>(codes)[w], codebook);
                    *reinterpret_cast<uint2*>(rows_base + swizzled(row, kD, w >> 1) + 8 * (w & 1)) = values;
                }
                __half* groups = reinterpret_cast<__half*>(k_s + (role ? S::ValueScales : S::KeyScales) + row * 8);
#pragma unroll
                for (int g = 0; g < kKVWindowGroups; ++g) { groups[g] = code_scale; }
            }
            __syncwarp();
        }
        compute(i, s);
        __syncwarp();
        s ^= 1;
    }
    cp_wait<0>();

    // Merge the warps' rows: each publishes its unnormalized rows and (max, sum).
    l0 += __shfl_xor_sync(FullMask, l0, 1);
    l0 += __shfl_xor_sync(FullMask, l0, 2);
    l1 += __shfl_xor_sync(FullMask, l1, 1);
    l1 += __shfl_xor_sync(FullMask, l1, 2);
    __syncthreads();
    float* rows  = reinterpret_cast<float*>(smem + kQRow * kRows);
    float* stats = rows + Warps * kRows * kD;
    float* mine  = rows + warp * kRows * kD;
#pragma unroll
    for (int b = 0; b < 16; ++b) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = gid + 8 * (j >> 1);
                const int d   = S::EvenOddValues ? 16 * b + 4 * lid + 2 * (j & 1) + h : 16 * b + 8 * h + 2 * lid + (j & 1);
                mine[row * kD + d] = o[b][h][j];
            }
        }
    }
    if (lid == 0) {
        stats[(warp * kRows + gid) * 2]         = m0;
        stats[(warp * kRows + gid) * 2 + 1]     = l0;
        stats[(warp * kRows + gid + 8) * 2]     = m1;
        stats[(warp * kRows + gid + 8) * 2 + 1] = l1;
    }
    __syncthreads();
    float* merged = stats + Warps * kRows * 2; // rotated values: the merged rows, rotated back below
    for (int e = tid; e < group * kD; e += Threads) {
        const int row = e / kD, d = e % kD;
        float m = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < Warps; ++w) { m = fmaxf(m, stats[(w * kRows + row) * 2]); }
        float l = 0.0f, acc = 0.0f;
#pragma unroll
        for (int w = 0; w < Warps; ++w) {
            const float mw = stats[(w * kRows + row) * 2];
            if (mw == -CUDART_INF_F) { continue; }
            const float f = exp2_approx((mw - m) * scale_l2);
            l             = __fmaf_rn(f, stats[(w * kRows + row) * 2 + 1], l);
            acc           = __fmaf_rn(f, rows[(w * kRows + row) * kD + d], acc);
        }
        const float value = l > 0.0f ? acc / l : 0.0f;
        if constexpr (S::RotatedValues) {
            merged[row * kD + d] = value;
        } else {
            out[(static_cast<std::size_t>(t) * heads + kv_head * group + row) * kD + d] = __float2bfloat16_rn(value);
        }
    }
    if constexpr (S::RotatedValues) {
        // Values are stored in the normalized Hadamard domain (its own inverse): rotate back.
        __syncthreads();
        for (int row = warp; row < group; row += Warps) {
            float v[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) { v[k] = merged[row * kD + lane + 32 * k]; }
            normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                out[(static_cast<std::size_t>(t) * heads + kv_head * group + row) * kD + lane + 32 * k] =
                    __float2bfloat16_rn(v[k]);
            }
        }
    }
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("qsa prompt: ") + message); }
}

template <KvCacheStorage Storage>
void launch(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch, const QsaGeometry& g, float scale,
            const std::int32_t* selected, const std::int32_t* counts, Tensor& out, cudaStream_t stream,
            const QsaVqWindow& vq) {
    constexpr int bytes = prompt_smem_bytes<Storage>();
    static_assert(bytes <= 99 * 1024, "one CTA's shared memory on sm_120");
    static const cudaError_t configured =
        cudaFuncSetAttribute(qsa_prompt_kernel<Storage>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    require(configured == cudaSuccess, "cannot reserve the kernel's shared memory");
    const int columns = q.ne[2];
    const dim3 grid(static_cast<unsigned>(columns), static_cast<unsigned>(g.kv_heads));
    qsa_prompt_kernel<Storage><<<grid, PromptShape<Storage>::Warps * 32, bytes, stream>>>(
        static_cast<const bf16*>(q.data), g.heads, g.kv_heads, layer.kv.k_pages.data, layer.kv.v_pages.data,
        layer.kv.k_scale_pages.data, layer.kv.v_scale_pages.data,
        static_cast<const std::int32_t*>(batch.block_tables.data), batch.block_tables.ne[0],
        static_cast<const std::int32_t*>(batch.table_rows.data), static_cast<const std::int32_t*>(batch.positions.data),
        batch.width, g.ratio, selected, counts, g.budget / g.ratio, scale, static_cast<bf16*>(out.data), layer.spaces, vq);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa prompt attention: ") + cudaGetErrorString(error)); }
}

} // namespace

bool qsa_prompt_supported(const QsaKVLayer& layer, const QsaGeometry& geometry) {
    switch (layer.kv.storage) {
    case KvCacheStorage::BFloat16:
    case KvCacheStorage::Int8Group64:
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Nvfp4Group16:
    case KvCacheStorage::Fp8KeyNvfp4Value:
    case KvCacheStorage::Vq2:
    case KvCacheStorage::Q4KeyVq2Value:
        return geometry.head_dim == kD && geometry.kv_heads > 0 && geometry.heads / geometry.kv_heads <= kRows;
    default:
        return false;
    }
}

void qsa_prompt_attention(const Tensor& q, const QsaKVLayer& layer, const QsaBatch& batch,
                          const QsaGeometry& geometry, float scale, const std::int32_t* selected,
                          const std::int32_t* counts, Tensor& out, cudaStream_t stream, const QsaVqWindow& vq) {
    require(qsa_prompt_supported(layer, geometry), "the prompt route does not serve this layer");
    switch (layer.kv.storage) {
    case KvCacheStorage::Int8Group64:
        launch<KvCacheStorage::Int8Group64>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        launch<KvCacheStorage::Fp8E4M3Row256>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Nvfp4Group16:
        launch<KvCacheStorage::Nvfp4Group16>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        launch<KvCacheStorage::Fp8KeyNvfp4Value>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Vq2:
        launch<KvCacheStorage::Vq2>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    case KvCacheStorage::Q4KeyVq2Value:
        launch<KvCacheStorage::Q4KeyVq2Value>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    default:
        launch<KvCacheStorage::BFloat16>(q, layer, batch, geometry, scale, selected, counts, out, stream, vq);
        break;
    }
}

} // namespace ninfer::ops::detail
