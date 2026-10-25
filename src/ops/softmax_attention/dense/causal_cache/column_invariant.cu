#include "ops/softmax_attention/dense/causal_cache/column_invariant.h"

#include "core/device.h"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

__global__ void column_valid_kernel(const std::int32_t* counts, std::int32_t* flags,
                                    std::int32_t width, std::int32_t batch) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= width * batch) return;
    const int column = index / batch;
    const int row    = index % batch;
    flags[index]     = column < counts[row] ? 1 : 0;
}

__global__ void column_table_rows_kernel(const std::int32_t* table_rows, std::int32_t* rows,
                                         std::int32_t width, std::int32_t batch) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= width * batch) return;
    rows[index] = table_rows == nullptr ? 0 : table_rows[index % batch];
}

struct PermutationBatch {
    CausalColumnPermutation tensors[4];
};

// blockIdx.y selects the tensor; each thread copies 4-byte words of destination rows.
__global__ void columns_permute_kernel(PermutationBatch batch_tensors, std::int32_t width,
                                       std::int32_t batch, bool to_columns) {
    const CausalColumnPermutation tensor = batch_tensors.tensors[blockIdx.y];
    const std::int64_t words_per_row     = tensor.row_bytes / 4;
    const std::int64_t words             = words_per_row * width * batch;
    const auto* source                   = static_cast<const std::uint32_t*>(tensor.source);
    auto* destination                    = static_cast<std::uint32_t*>(tensor.destination);
    for (std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < words; index += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t row  = index / words_per_row;
        const std::int64_t word = index - row * words_per_row;
        // Column-major row r is column r / batch of round row r % batch.
        const std::int64_t column = row / batch;
        const std::int64_t round  = row - column * batch;
        const std::int64_t other  = round * width + column;
        if (to_columns) {
            destination[index] = source[other * words_per_row + word];
        } else {
            destination[other * words_per_row + word] = source[index];
        }
    }
}

} // namespace

void causal_columns_permute_launch(const CausalColumnPermutation* tensors, std::int32_t count,
                                   std::int32_t width, std::int32_t batch, bool to_columns,
                                   cudaStream_t stream) {
    if (count <= 0 || count > 4 || width <= 0 || batch <= 0) {
        throw std::invalid_argument("causal column permutation: invalid round shape");
    }
    PermutationBatch batch_tensors{};
    for (std::int32_t index = 0; index < count; ++index) {
        if (tensors[index].row_bytes <= 0 || tensors[index].row_bytes % 4 != 0 ||
            tensors[index].source == nullptr || tensors[index].destination == nullptr) {
            throw std::invalid_argument("causal column permutation: invalid tensor");
        }
        batch_tensors.tensors[index] = tensors[index];
    }
    constexpr int kThreads = 256;
    columns_permute_kernel<<<dim3(64, static_cast<unsigned>(count)), kThreads, 0, stream>>>(
        batch_tensors, width, batch, to_columns);
    CUDA_CHECK(cudaGetLastError());
}

void causal_column_valid_launch(const Tensor& counts, Tensor& flags, std::int32_t width,
                                std::int32_t batch, cudaStream_t stream) {
    if (width <= 0 || batch <= 0 || width * batch > 1024) {
        throw std::invalid_argument("causal column validity: invalid round shape");
    }
    column_valid_kernel<<<1, 1024, 0, stream>>>(static_cast<const std::int32_t*>(counts.data),
                                                static_cast<std::int32_t*>(flags.data), width,
                                                batch);
    CUDA_CHECK(cudaGetLastError());
}

void causal_column_table_rows_launch(const Tensor& table_rows, Tensor& rows, std::int32_t width,
                                     std::int32_t batch, cudaStream_t stream) {
    if (width <= 0 || batch <= 0 || width * batch > 1024 || rows.data == nullptr) {
        throw std::invalid_argument("causal column table rows: invalid round shape");
    }
    column_table_rows_kernel<<<1, 1024, 0, stream>>>(
        static_cast<const std::int32_t*>(table_rows.data), static_cast<std::int32_t*>(rows.data),
        width, batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
