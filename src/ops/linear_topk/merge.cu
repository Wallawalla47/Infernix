#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/score_id_order.cuh"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cstdint>

namespace infernix::ops::detail {
namespace {

// The selection: each CTA of 256 threads takes one column's slice of 4096 producer keys, 16 a thread,
// sorts each thread's keys in registers and keeps the top sixteen through warp and CTA merges; one
// warp per column then merges the slices' lists. Keys are unique (distinct ids), so the selected
// keys and their order are those of a full sort whatever the merge tree.
inline constexpr int kSelectThreads = 256;
inline constexpr int kSelectWarps   = kSelectThreads / 32;
static_assert(kLinearTopKSliceKeys == kSelectThreads * kLinearTopK, "one sixteen-key list a thread");

using KeyList = std::uint64_t[kLinearTopK];

// Sorts a thread's sixteen keys descending (bitonic network, unrolled into registers).
__device__ __forceinline__ void sort_descending(KeyList& keys) {
#pragma unroll
    for (int size = 2; size <= kLinearTopK; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int i = 0; i < kLinearTopK; ++i) {
                const int j = i ^ stride;
                if (j > i) {
                    const std::uint64_t a = keys[i], b = keys[j];
                    const bool swap       = (i & size) == 0 ? a < b : a > b;
                    keys[i]               = swap ? b : a;
                    keys[j]               = swap ? a : b;
                }
            }
        }
    }
}

// keys := the top sixteen of keys and other, descending (both descending): the elementwise maximum of
// keys and reversed other is bitonic and holds them, and a bitonic merge sorts it.
__device__ __forceinline__ void merge_top(KeyList& keys, const KeyList& other) {
#pragma unroll
    for (int i = 0; i < kLinearTopK; ++i) {
        const std::uint64_t b = other[kLinearTopK - 1 - i];
        keys[i]               = keys[i] > b ? keys[i] : b;
    }
#pragma unroll
    for (int stride = kLinearTopK / 2; stride > 0; stride >>= 1) {
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) {
            if ((i & stride) == 0) {
                const std::uint64_t a = keys[i], b = keys[i + stride];
                keys[i]               = a < b ? b : a;
                keys[i + stride]      = a < b ? a : b;
            }
        }
    }
}

// Butterfly over lane masks [first, 1]: every lane of each 2*first-lane group ends with the group's top sixteen.
__device__ __forceinline__ void butterfly_top(KeyList& keys, int first) {
#pragma unroll
    for (int mask = first; mask > 0; mask >>= 1) {
        KeyList other;
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) { other[i] = __shfl_xor_sync(0xffffffffu, keys[i], mask); }
        merge_top(keys, other);
    }
}

__device__ __forceinline__ void load_list(KeyList& keys, const std::uint64_t* __restrict__ source) {
#pragma unroll
    for (int i = 0; i < kLinearTopK; i += 2) {
        const ulonglong2 pair = __ldcg(reinterpret_cast<const ulonglong2*>(source + i));
        keys[i]               = pair.x;
        keys[i + 1]           = pair.y;
    }
}

__device__ __forceinline__ void clear_list(KeyList& keys) {
#pragma unroll
    for (int i = 0; i < kLinearTopK; ++i) { keys[i] = 0; }
}

// Grid (slices, columns): slice_keys[column][slice] := the slice's top sixteen keys, descending. Keys past the
// column's end are the sentinel 0, below every key.
__global__ __launch_bounds__(kSelectThreads) void linear_topk_select_kernel(
    const std::uint64_t* __restrict__ partial_keys, std::uint64_t* __restrict__ slice_keys,
    std::int32_t column_keys, std::int32_t slices) {
    pdl::enter();
    const int slice  = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane = tid & 31, warp = tid >> 5;
    const int first  = slice * kLinearTopKSliceKeys;
    const int live   = min(kLinearTopKSliceKeys, column_keys - first);
    const std::uint64_t* __restrict__ source = partial_keys + static_cast<std::int64_t>(column) * column_keys + first;

    // Key pairs p = i * 256 + tid (16-byte reads; a column holds whole sixteen-key groups, so pairs are aligned).
    KeyList keys;
#pragma unroll
    for (int i = 0; i < kLinearTopK / 2; ++i) {
        const int k = 2 * (i * kSelectThreads + tid);
        if (k + 1 < live) {
            const ulonglong2 pair = __ldcg(reinterpret_cast<const ulonglong2*>(source + k));
            keys[2 * i]           = pair.x;
            keys[2 * i + 1]       = pair.y;
        } else {
            keys[2 * i]     = k < live ? __ldcg(source + k) : 0;
            keys[2 * i + 1] = 0;
        }
    }
    sort_descending(keys);
    butterfly_top(keys, 16);

    __shared__ std::uint64_t warp_lists[kSelectWarps][kLinearTopK];
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) { warp_lists[warp][i] = keys[i]; }
    }
    __syncthreads();
    if (warp != 0) { return; }
    if (lane < kSelectWarps) {
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) { keys[i] = warp_lists[lane][i]; }
    } else {
        clear_list(keys);
    }
    butterfly_top(keys, kSelectWarps / 2);
    if (lane == 0) {
        auto* out = reinterpret_cast<ulonglong2*>(slice_keys + (static_cast<std::int64_t>(column) * slices + slice) *
                                                                   kLinearTopK);
#pragma unroll
        for (int i = 0; i < kLinearTopK; i += 2) { out[i / 2] = make_ulonglong2(keys[i], keys[i + 1]); }
    }
}

// One warp per column: the slices' lists merged into the column's sixteen candidates.
__global__ __launch_bounds__(32) void linear_topk_final_kernel(const std::uint64_t* __restrict__ slice_keys,
                                                               std::int32_t* __restrict__ candidate_ids,
                                                               float* __restrict__ candidate_scores,
                                                               std::int32_t slices) {
    pdl::enter();
    const int column = static_cast<int>(blockIdx.x);
    const int lane   = static_cast<int>(threadIdx.x);
    const std::uint64_t* __restrict__ lists = slice_keys + static_cast<std::int64_t>(column) * slices * kLinearTopK;
    KeyList keys;
    if (lane < slices) {
        load_list(keys, lists + lane * kLinearTopK);
    } else {
        clear_list(keys);
    }
    for (int slice = lane + 32; slice < slices; slice += 32) {
        KeyList other;
        load_list(other, lists + slice * kLinearTopK);
        merge_top(keys, other);
    }
    butterfly_top(keys, 16);
    if (lane == 0) {
        auto* ids    = reinterpret_cast<int4*>(candidate_ids + static_cast<std::int64_t>(column) * kLinearTopK);
        auto* scores = reinterpret_cast<float4*>(candidate_scores + static_cast<std::int64_t>(column) * kLinearTopK);
#pragma unroll
        for (int i = 0; i < kLinearTopK; i += 4) {
            ids[i / 4]    = make_int4(id_from_order_key(keys[i]), id_from_order_key(keys[i + 1]),
                                      id_from_order_key(keys[i + 2]), id_from_order_key(keys[i + 3]));
            scores[i / 4] = make_float4(score_from_order_key(keys[i]), score_from_order_key(keys[i + 1]),
                                        score_from_order_key(keys[i + 2]), score_from_order_key(keys[i + 3]));
        }
    }
}

} // namespace

void linear_topk_merge_launch(const LinearTopKWorkspace& workspace, Tensor& candidate_ids,
                              Tensor& candidate_scores, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(static_cast<unsigned>(workspace.slices), static_cast<unsigned>(workspace.columns)),
         dim3(kSelectThreads), 0, stream},
        linear_topk_select_kernel, static_cast<const std::uint64_t*>(workspace.partial_keys.data),
        static_cast<std::uint64_t*>(workspace.slice_keys.data), workspace.producer_groups * kLinearTopK,
        workspace.slices));
    CUDA_CHECK(pdl::launch_consumer({dim3(static_cast<unsigned>(workspace.columns)), dim3(32), 0, stream},
                                    linear_topk_final_kernel,
                                    static_cast<const std::uint64_t*>(workspace.slice_keys.data),
                                    static_cast<std::int32_t*>(candidate_ids.data),
                                    static_cast<float*>(candidate_scores.data), workspace.slices));
}

} // namespace infernix::ops::detail
