#pragma once
#include "core/device.h"
#include "ops/common/dynamic_shared.cuh"

namespace infernix::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* nvfp4_shared_storage() {
    static_assert(Bytes > 0 && Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char shared_dynamic[];
        return shared_dynamic;
    } else {
        __shared__ __align__(16) unsigned char shared_static[Bytes];
        return shared_static;
    }
}

template <class Schedule, class Epilogue>
inline constexpr int nvfp4_mma_shared_bytes = [] {
    constexpr int extra = [] {
        if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
            return Epilogue::template kSharedBytes<Schedule>;
        else
            return 0;
    }();
    return Schedule::kSharedBytes > extra ? Schedule::kSharedBytes : extra;
}();
} // namespace infernix::ops::detail
