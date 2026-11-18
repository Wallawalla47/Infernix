#pragma once
#include "core/device.h"
#include "ops/common/dynamic_shared.cuh"

namespace infernix::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* q8_shared_storage() {
    static_assert(Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char dynamic_shared[];
        return dynamic_shared;
    } else {
        __shared__ __align__(16) unsigned char storage[Bytes];
        return storage;
    }
}
} // namespace infernix::ops::detail
