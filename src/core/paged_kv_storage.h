#pragma once

#include "core/dtype.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

// Exact recent-key window of the vector-quantized formats (Vq2, Q4KeyVq2Value). Every execution
// row of a layer owns kKVWindowSlots INT8-G64 copies of rotated K and V rows: slots
// [0, kKVWindowSinkTokens) hold positions below kKVWindowSinkTokens and the remaining
// kKVWindowRingTokens slots hold position p >= kKVWindowSinkTokens at
// kKVWindowSinkTokens + p % kKVWindowRingTokens. Each slot carries a tag that hashes the
// position and the persistent codes it copies, so a reader uses a slot only while the paged
// row still holds those codes.
inline constexpr std::int32_t kKVWindowSinkTokens = 64;
inline constexpr std::int32_t kKVWindowRingTokens = 1024;
inline constexpr std::int32_t kKVWindowSlots      = kKVWindowSinkTokens + kKVWindowRingTokens;
// Keys at most this far before a call's first query are read from the window.
inline constexpr std::int32_t kKVWindowRecentTokens = 768;
// Calls up to this width write their window slots before attention; wider calls read their own
// keys from per-call staging and commit the window afterwards.
inline constexpr std::int32_t kKVWindowInlineWidth = kKVWindowRingTokens - kKVWindowRecentTokens;
inline constexpr std::int32_t kKVWindowGroups      = kD256KVCacheHeadDim / 64;

static_assert(kKVWindowInlineWidth >= 64);

[[nodiscard]] constexpr bool kv_storage_has_exact_window(KvCacheStorage storage) noexcept {
    return storage == KvCacheStorage::Vq2 || storage == KvCacheStorage::Q4KeyVq2Value;
}

/** Physical data/scale planes for one K or V vector. */
struct PagedKVVectorLayout {
    DType data_dtype                  = DType::BF16;
    std::int32_t data_leading_extent  = 0;
    DType scale_dtype                 = DType::U8;
    std::int32_t scale_leading_extent = 0;

    [[nodiscard]] constexpr bool has_scale() const noexcept { return scale_leading_extent != 0; }

    [[nodiscard]] std::size_t physical_bytes() const {
        return static_cast<std::size_t>(data_leading_extent) * dtype_size(data_dtype) +
               static_cast<std::size_t>(scale_leading_extent) * dtype_size(scale_dtype);
    }

    friend bool operator==(const PagedKVVectorLayout&, const PagedKVVectorLayout&) = default;
};

/** Resolved physical plane schema for one paged K/V layer. */
struct PagedKVStorageLayout {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    std::int32_t head_dim  = 0;
    PagedKVVectorLayout key;
    PagedKVVectorLayout value;

    [[nodiscard]] constexpr std::size_t planes_per_layer() const noexcept {
        return 2ULL + static_cast<std::size_t>(key.has_scale()) +
               static_cast<std::size_t>(value.has_scale());
    }

    [[nodiscard]] constexpr std::size_t logical_vector_bytes() const noexcept {
        return static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    }

    [[nodiscard]] constexpr std::size_t logical_bytes_per_token_head() const noexcept {
        return 2ULL * logical_vector_bytes();
    }

    [[nodiscard]] std::size_t physical_bytes_per_token_head() const {
        return key.physical_bytes() + value.physical_bytes();
    }

    friend bool operator==(const PagedKVStorageLayout&, const PagedKVStorageLayout&) = default;
};

[[nodiscard]] inline PagedKVStorageLayout paged_kv_storage_layout(KvCacheStorage storage,
                                                                  std::int32_t head_dim) {
    if (head_dim <= 0) { throw std::invalid_argument("KV-cache head dimension must be positive"); }

    const auto symmetric = [=](PagedKVVectorLayout vector) {
        return PagedKVStorageLayout{storage, head_dim, vector, vector};
    };
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {storage,
                head_dim,
                {DType::BF16, head_dim, DType::U8, 0},
                {DType::FP16, head_dim, DType::U8, 0}};
    case KvCacheStorage::Int8Group64:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::I8, 256, DType::FP16, 4}); }
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        if (head_dim == kD256KVCacheHeadDim) {
            return symmetric({DType::FP8_E4M3FN, 256, DType::FP16, 1});
        }
        break;
    case KvCacheStorage::Nvfp4Group16:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::U8, 128, DType::U8, 16}); }
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::FP8_E4M3FN, 256, DType::FP16, 1},
                    {DType::U8, 128, DType::U8, 16}};
        }
        break;
    case KvCacheStorage::Vq2:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::U8, 64, DType::FP16, 1}); }
        break;
    case KvCacheStorage::Q4KeyVq2Value:
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::U8, 128, DType::FP16, 1},
                    {DType::U8, 64, DType::FP16, 1}};
        }
        break;
    }
    throw std::invalid_argument("unsupported paged KV-cache storage geometry");
}

} // namespace ninfer
