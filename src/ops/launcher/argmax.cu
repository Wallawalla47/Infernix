// Implements: include/infernix/ops/argmax.h
// Match: validated contiguous FP32 or BF16 logits and I32 output.
// Algorithm assumptions: one tile uses a direct reduction; larger domains use
// zero-initialized atomic winners across route-selected row tiles.
#include "ops/launcher/argmax.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/argmax.cuh"
#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"

#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr std::int32_t kFullPhysicalRows      = 248320;
constexpr std::int32_t kFullValidRows         = 248077;
constexpr std::int32_t kShortlistRows         = 131072;
constexpr std::int32_t kSmallAggregateColumns = 8;
constexpr int kFullAggregateBlock             = 128;
constexpr int kShortlistAggregateBlock        = 256;

int tiled_block_for(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t t_count) {
    if (t_count <= kSmallAggregateColumns) { return kArgmaxBlock; }
    if (physical_rows == kFullPhysicalRows && valid_rows == kFullValidRows) {
        return kFullAggregateBlock;
    }
    if (physical_rows == kShortlistRows && valid_rows == kShortlistRows) {
        return kShortlistAggregateBlock;
    }
    return kArgmaxBlock;
}

template <class Logit>
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream);

template <class Logit>
void argmax_typed(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    // One CTA per column up to one block of rows; the tiled kernel beyond (each CTA covers
    // kArgmaxItemsPerThread rows per thread).
    if (div_up(valid_rows, kArgmaxBlock) < 2) {
        argmax_kernel<<<static_cast<unsigned int>(t_count), kArgmaxBlock, 0, stream>>>(
            static_cast<const Logit*>(logits.data), static_cast<std::int32_t*>(out.data), valid_rows,
            physical_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    argmax_tiled_atomic_launch<Logit>(logits, out, valid_rows, tiled_block_for(physical_rows, valid_rows, t_count),
                                      stream);
}

} // namespace

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                   cudaStream_t stream) {
    if (logits.ne[1] == 0) { return; }
    if (logits.dtype == DType::FP32) {
        argmax_typed<float>(logits, out, valid_rows, stream);
    } else {
        argmax_typed<__nv_bfloat16>(logits, out, valid_rows, stream);
    }
}

namespace {

__global__ void argmax_map_kernel(const std::int32_t* row_ids, std::int32_t* out, std::int32_t count) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t < count) { out[t] = row_ids[out[t]]; }
}

} // namespace

std::int32_t argmax_probability_tiles(DType logits, std::int32_t physical_rows, std::int32_t valid_rows,
                                      std::int32_t columns) {
    // As argmax: FP32 logits and domains of one block take one CTA per column.
    if (logits == DType::FP32 || div_up(valid_rows, kArgmaxBlock) < 2) { return 0; }
    return div_up(valid_rows, tiled_block_for(physical_rows, valid_rows, columns) * kArgmaxItemsPerThread);
}

namespace {

constexpr std::size_t kProbabilityRegionAlign = 256;

std::size_t probability_region(std::size_t bytes) {
    return round_up(bytes, kProbabilityRegionAlign);
}

} // namespace

std::size_t argmax_probability_workspace_bytes(std::int32_t tiles, std::int32_t columns) {
    if (tiles == 0 || columns == 0) { return 0; }
    const auto partials = static_cast<std::size_t>(tiles) * static_cast<std::size_t>(columns) * sizeof(float);
    return 2 * probability_region(partials) +
           probability_region(static_cast<std::size_t>(columns) * sizeof(unsigned int));
}

void argmax_probability_launch(const Tensor& logits, Tensor& out, Tensor& probability, std::int32_t valid_rows,
                               DeviceSpan workspace, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    auto* rows                       = static_cast<std::int32_t*>(out.data);
    auto* confidence                 = static_cast<float*>(probability.data);
    const std::int32_t tiles = argmax_probability_tiles(logits.dtype, physical_rows, valid_rows, t_count);
    if (tiles == 0) {
        const dim3 grid(static_cast<unsigned int>(t_count));
        if (logits.dtype == DType::FP32) {
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kArgmaxBlock), 0, stream},
                                            argmax_probability_column_kernel<float>,
                                            static_cast<const float*>(logits.data), rows, confidence, valid_rows,
                                            physical_rows));
        } else {
            CUDA_CHECK(pdl::launch_consumer({grid, dim3(kArgmaxBlock), 0, stream},
                                            argmax_probability_column_kernel<__nv_bfloat16>,
                                            static_cast<const __nv_bfloat16*>(logits.data), rows, confidence,
                                            valid_rows, physical_rows));
        }
        return;
    }
    const auto partials = probability_region(static_cast<std::size_t>(tiles) * t_count * sizeof(float));
    auto* base          = static_cast<unsigned char*>(workspace.data);
    auto* partial_value = reinterpret_cast<float*>(base);
    auto* partial_sum   = reinterpret_cast<float*>(base + partials);
    auto* arrivals      = reinterpret_cast<unsigned int*>(base + 2 * partials);
    // The atomic winners start from row 0 and every column's arrivals from 0.
    CUDA_CHECK(cudaMemsetAsync(rows, 0, static_cast<std::size_t>(t_count) * sizeof(std::int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(arrivals, 0, static_cast<std::size_t>(t_count) * sizeof(unsigned int), stream));
    const int block = tiled_block_for(physical_rows, valid_rows, t_count);
    const dim3 grid(static_cast<unsigned int>(tiles), static_cast<unsigned int>(t_count));
    CUDA_CHECK(pdl::launch_consumer({grid, dim3(block), 0, stream}, argmax_probability_tiled_kernel,
                                    static_cast<const __nv_bfloat16*>(logits.data), rows, confidence,
                                    partial_value, partial_sum, arrivals, valid_rows, physical_rows));
}

void argmax_map_launch(const Tensor& row_ids, Tensor& out, cudaStream_t stream) {
    const std::int32_t count = out.ne[0];
    if (count == 0) { return; }
    argmax_map_kernel<<<static_cast<unsigned int>((count + 127) / 128), 128, 0, stream>>>(
        static_cast<const std::int32_t*>(row_ids.data), static_cast<std::int32_t*>(out.data), count);
    CUDA_CHECK(cudaGetLastError());
}

namespace {

template <class Logit>
void argmax_tiled_atomic_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                                int block, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t t_count       = logits.ne[1];
    const int tiled_blocks           = div_up(valid_rows, block * kArgmaxItemsPerThread);
    for_each_token_slice(t_count, 1, [&](int token_offset, int token_count) {
        const Tensor logits_slice = logits.slice(1, token_offset, token_count);
        Tensor out_slice          = out.slice(0, token_offset, token_count);
        CUDA_CHECK(cudaMemsetAsync(out_slice.data, 0,
                                   static_cast<std::size_t>(token_count) * sizeof(std::int32_t),
                                   stream));
        const dim3 grid(static_cast<unsigned int>(tiled_blocks),
                        static_cast<unsigned int>(token_count));
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(grid), dim3(block), 0, stream}, argmax_tiled_atomic_kernel<Logit>,
            static_cast<const Logit*>(logits_slice.data),
            static_cast<std::int32_t*>(out_slice.data), valid_rows, physical_rows));
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace
} // namespace infernix::ops::detail
