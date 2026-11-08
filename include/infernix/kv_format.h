#pragma once

#include <cstdint>

namespace infernix {

// The KV cache storage formats (EngineOptions::kv_cache). Its own header so that Ops and core code that
// only need the format do not depend on the Engine's types (infernix/types.h; candidates RE-05).
enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8KeyNvfp4Value,
    // Rotated 2-bit eight-value vector codes for K and V plus an exact INT8 recent-key window.
    Vq2,
    // Rotated 4-bit Lloyd-Max K codes, 2-bit vector V codes and the same exact recent-key window.
    Q4KeyVq2Value,
    // BFloat16's K and V values packed losslessly to 12 bits a value and an 8-byte side record a row
    // (ops/kv_cache/bf16p_codec.cuh): 23 % fewer bytes than BFloat16; a request appending a row with no
    // packed form fails.
    Bf16Packed,
};

[[nodiscard]] inline const char* kv_cache_cli_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    case KvCacheStorage::Vq2:
        return "vq2";
    case KvCacheStorage::Q4KeyVq2Value:
        return "k4v2";
    case KvCacheStorage::Bf16Packed:
        return "bf16p";
    }
    return "unknown";
}

} // namespace infernix
