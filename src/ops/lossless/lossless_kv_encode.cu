#include "infernix/ops/lossless_ops.h"

#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "infernix/ops/kv_cache_append.h"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

// The design reference's KV storage of one window (lossless_kv_encode, docs/maintainer/lossless-route.md §3b): the
// code planes written by the production kv_cache_append through an identity block table (page p holds keys
// 64 p .. 64 p + 63, so a plane's rows are in lossless_codec_attention's page order), and for the VQ formats every
// key's exact window rows as the append's window copy encodes them (src/ops/kv_cache/append/vq2_kernel.cuh): the
// row rotated in FP32 (normalized_hadamard_d256_inplace), then each 64-coordinate group's INT8-G64 codes.

namespace infernix::ops {
namespace {

constexpr int kDim  = 256;
constexpr int kPage = 64;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless_kv_encode: ") + message); }
}

std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

struct Layout {
    PagedKVStorageLayout storage;
    int pages;
    std::size_t rows;
    std::size_t k_codes, k_scales, v_codes, v_scales, table, positions, wk_codes, wk_scales, wv_codes, wv_scales, total;
};

Layout layout_of(KvCacheStorage storage, int kv_heads, int columns) {
    require(storage != KvCacheStorage::BFloat16, "bf16 has no codes (the design reference rounds V to FP16)");
    Layout l{};
    l.storage    = paged_kv_storage_layout(storage, kDim);
    l.pages      = (columns + kPage - 1) / kPage;
    l.rows       = static_cast<std::size_t>(l.pages) * kPage * kv_heads;
    auto plane   = [&](DType dtype, int leading) { return align256(l.rows * leading * dtype_size(dtype)); };
    std::size_t at = 0;
    auto take      = [&](std::size_t bytes) {
        const std::size_t offset = at;
        at += bytes;
        return offset;
    };
    l.k_codes   = take(plane(l.storage.key.data_dtype, l.storage.key.data_leading_extent));
    l.k_scales  = take(plane(l.storage.key.scale_dtype, l.storage.key.scale_leading_extent));
    l.v_codes   = take(plane(l.storage.value.data_dtype, l.storage.value.data_leading_extent));
    l.v_scales  = take(plane(l.storage.value.scale_dtype, l.storage.value.scale_leading_extent));
    l.table     = take(align256(static_cast<std::size_t>(l.pages) * sizeof(std::int32_t)));
    l.positions = take(align256(static_cast<std::size_t>(columns) * sizeof(std::int32_t)));
    if (kv_storage_has_exact_window(storage)) {
        l.wk_codes  = take(align256(l.rows * kDim));
        l.wk_scales = take(align256(l.rows * kKVWindowGroups * sizeof(__half)));
        l.wv_codes  = take(align256(l.rows * kDim));
        l.wv_scales = take(align256(l.rows * kKVWindowGroups * sizeof(__half)));
    }
    l.total = at;
    return l;
}

__global__ void iota_kernel(std::int32_t* out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) { out[i] = i; }
}

// One warp per (key j, KV head h, role): the window row of the rotated K (role 0) or V (role 1) row, as the
// append's window copy encodes it, at the plane row of (j, h).
__global__ void window_rows_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v, int kv_heads, int columns,
                                   std::int8_t* wk_codes, __half* wk_scales, std::int8_t* wv_codes, __half* wv_scales) {
    constexpr unsigned kFull = 0xFFFFFFFFU;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const long long items = static_cast<long long>(columns) * kv_heads * 2;
    const long long warps = static_cast<long long>(gridDim.x) * (blockDim.x / 32);
    for (long long it = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32; it < items; it += warps) {
        const int role        = static_cast<int>(it % 2);
        const long long rr    = it / 2;
        const int kvh         = static_cast<int>(rr % kv_heads), j = static_cast<int>(rr / kv_heads);
        const __nv_bfloat16* src = (role == 0 ? k : v) + (static_cast<std::size_t>(j) * kv_heads + kvh) * kDim;
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { values[r] = __bfloat162float(src[lane + 32 * r]); }
        normalized_hadamard_d256_inplace(values, lane);
        const std::size_t row = (static_cast<std::size_t>(j / kPage) * kv_heads + kvh) * kPage + j % kPage;
        std::int8_t* codes    = (role == 0 ? wk_codes : wv_codes) + row * kDim;
        __half* scales        = (role == 0 ? wk_scales : wv_scales) + row * kKVWindowGroups;
        // Group g holds coordinates [64 g, 64 g + 64): this lane's elements 64 g + lane and 64 g + 32 + lane.
#pragma unroll
        for (int g = 0; g < kKVWindowGroups; ++g) {
            const float x0   = values[2 * g];
            const float x1   = values[2 * g + 1];
            const float amax = warp_max(fmaxf(fabsf(x0), fabsf(x1)), kFull);
            const auto quant = kv_cache_int8_quant_params(amax);
            codes[64 * g + lane]      = kv_cache_int8_quant_code(x0, quant.inverse_scale);
            codes[64 * g + 32 + lane] = kv_cache_int8_quant_code(x1, quant.inverse_scale);
            if (lane == 0) { scales[g] = quant.scale; }
        }
    }
}

// The planes' codecs for a storage (kv_cache_append: K rotated in every format, V in NVFP4, K8V4 and the VQ
// formats).
LosslessKvPlane plane_of(KvCacheStorage storage, bool key) {
    switch (storage) {
    case KvCacheStorage::Int8Group64: return {LosslessKvCodec::Int8G64, key};
    case KvCacheStorage::Fp8E4M3Row256: return {LosslessKvCodec::Fp8Row, key};
    case KvCacheStorage::Nvfp4Group16: return {LosslessKvCodec::Nvfp4G16, true};
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return key ? LosslessKvPlane{LosslessKvCodec::Fp8Row, true} : LosslessKvPlane{LosslessKvCodec::Nvfp4G16, true};
    case KvCacheStorage::Vq2: return {LosslessKvCodec::Vq2, true};
    case KvCacheStorage::Q4KeyVq2Value:
        return key ? LosslessKvPlane{LosslessKvCodec::Q4Lloyd, true} : LosslessKvPlane{LosslessKvCodec::Vq2, true};
    case KvCacheStorage::BFloat16: break;
    }
    throw std::invalid_argument("lossless_kv_encode: no codec planes for this storage");
}

} // namespace

std::size_t lossless_kv_encode_bytes(KvCacheStorage storage, int kv_heads, int columns) {
    return layout_of(storage, kv_heads, columns).total;
}

LosslessKvCodes lossless_kv_encode(const Tensor& k, const Tensor& v, KvCacheStorage storage, void* buffer,
                                   cudaStream_t stream) {
    require(k.dtype == DType::BF16 && v.dtype == DType::BF16 && k.is_contiguous() && v.is_contiguous() &&
                k.ne[0] == kDim && v.ne[0] == kDim && k.ne[1] == v.ne[1] && k.ne[2] == v.ne[2],
            "k and v contiguous BF16 [256, Hkv, T]");
    const int kv_heads = k.ne[1], columns = k.ne[2];
    const Layout l     = layout_of(storage, kv_heads, columns);
    auto* base         = static_cast<std::uint8_t*>(buffer);
    LosslessKvCodes out{};
    out.k        = plane_of(storage, true);
    out.v        = plane_of(storage, false);
    out.k.codes  = base + l.k_codes;
    out.k.scales = base + l.k_scales;
    out.v.codes  = base + l.v_codes;
    out.v.scales = base + l.v_scales;
    if (columns == 0) { return out; }
    auto* table     = reinterpret_cast<std::int32_t*>(base + l.table);
    auto* positions = reinterpret_cast<std::int32_t*>(base + l.positions);
    iota_kernel<<<(l.pages + 255) / 256, 256, 0, stream>>>(table, l.pages);
    CUDA_CHECK(cudaGetLastError());
    iota_kernel<<<(columns + 255) / 256, 256, 0, stream>>>(positions, columns);
    CUDA_CHECK(cudaGetLastError());
    const PagedKVStorageLayout& s = l.storage;
    PagedKVLayerView view{};
    view.k_pages = Tensor(base + l.k_codes, s.key.data_dtype, {s.key.data_leading_extent, kPage, kv_heads, l.pages});
    view.v_pages = Tensor(base + l.v_codes, s.value.data_dtype, {s.value.data_leading_extent, kPage, kv_heads, l.pages});
    view.k_scale_pages = Tensor(base + l.k_scales, s.key.scale_dtype, {s.key.scale_leading_extent, kPage, kv_heads, l.pages});
    view.v_scale_pages =
        Tensor(base + l.v_scales, s.value.scale_dtype, {s.value.scale_leading_extent, kPage, kv_heads, l.pages});
    view.block_table  = Tensor(table, DType::I32, {l.pages});
    view.head_dim     = kDim;
    view.num_kv_heads = kv_heads;
    view.storage      = storage;
    kv_cache_append(k, v, Tensor(positions, DType::I32, {columns}), view, stream);
    if (kv_storage_has_exact_window(storage)) {
        out.window_k      = {LosslessKvCodec::Int8G64, true, base + l.wk_codes, base + l.wk_scales};
        out.window_v      = {LosslessKvCodec::Int8G64, true, base + l.wv_codes, base + l.wv_scales};
        out.window_sinks  = kKVWindowSinkTokens;
        out.window_recent = kKVWindowRecentTokens;
        const long long warps = static_cast<long long>(columns) * kv_heads * 2;
        window_rows_kernel<<<static_cast<int>(std::min<long long>((warps + 7) / 8, 4096)), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data), kv_heads, columns,
            reinterpret_cast<std::int8_t*>(base + l.wk_codes), reinterpret_cast<__half*>(base + l.wk_scales),
            reinterpret_cast<std::int8_t*>(base + l.wv_codes), reinterpret_cast<__half*>(base + l.wv_scales));
        CUDA_CHECK(cudaGetLastError());
    }
    return out;
}

} // namespace infernix::ops
