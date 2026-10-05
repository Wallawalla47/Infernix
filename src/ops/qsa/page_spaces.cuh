#pragma once

// Page-id translation of QsaPageSpaces (ninfer/ops/qsa.h; design §19.3.11) for QSA's kernels: a
// block-table page id names a pool page, a host page or a lent page, and each space has its own
// planes of the pool's geometry.

#include "ninfer/ops/qsa.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The plane of `page`'s space (`pool`, `host` or `lent`), with the page's index inside that space.
__device__ __forceinline__ const void* qsa_space_plane(const QsaPageSpaces& spaces, const void* pool, const void* host,
                                                       const void* lent, int page, int& local) {
    if (page < spaces.device_pages) {
        local = page;
        return pool;
    }
    const int q = page - spaces.device_pages;
    if (q < spaces.host_pages) {
        local = q;
        return host;
    }
    local = q - spaces.host_pages;
    return lent;
}

// The pooled-key plane of `page` (the pool's, or the spaces' device array) and the page's index in it.
template <class T>
__device__ __forceinline__ T* qsa_pooled_plane(const QsaPageSpaces& spaces, T* pool, int page, int& local) {
    if (page < spaces.device_pages) {
        local = page;
        return pool;
    }
    local = page - spaces.device_pages;
    return static_cast<T*>(spaces.pooled);
}

} // namespace ninfer::ops::detail
