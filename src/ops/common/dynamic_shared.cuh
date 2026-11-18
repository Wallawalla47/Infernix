#pragma once

// A kernel's dynamic shared memory. The attribute that lets a launch take more than the 48 KiB every kernel may is
// raised once per kernel and size; the first call's result is kept and checked at every call, so a failure is
// reported at each launch that needs it rather than by an unrelated later error check.

#include "core/device.h" // CUDA_CHECK

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// Raises `Kernel`'s dynamic shared-memory limit to `Bytes` and returns `Bytes`, the launch's dynamic size.
template <auto Kernel, int Bytes>
int reserve_dynamic_shared() {
    static const cudaError_t status = cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
    CUDA_CHECK(status);
    return Bytes;
}

// The launch's dynamic shared-memory size for a kernel that needs `Bytes`: past 48 KiB the buffer is dynamic and the
// limit is raised (reserve_dynamic_shared); within it the size is `Bytes` for a dynamic buffer (`Dynamic`) and 0 for
// a static one.
template <int Bytes, auto Kernel, bool Dynamic = false>
int prepare_dynamic_shared() {
    static_assert(Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        return reserve_dynamic_shared<Kernel, Bytes>();
    } else {
        return Dynamic ? Bytes : 0;
    }
}

} // namespace infernix::ops::detail
