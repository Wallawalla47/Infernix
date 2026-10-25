#include "core/decode_graph.h"

#include "core/device.h"

namespace infernix {
namespace {

__global__ void graph_gate_kernel(const float* __restrict__ value, double threshold, cudaGraphConditionalHandle handle) {
    cudaGraphSetConditional(handle, static_cast<double>(*value) < threshold ? 0U : 1U);
}

__global__ void graph_gate_limit_kernel(const float* __restrict__ value, double threshold,
                                        const std::int32_t* __restrict__ limit, std::int32_t step,
                                        cudaGraphConditionalHandle handle) {
    const bool confident = !(static_cast<double>(*value) < threshold);
    cudaGraphSetConditional(handle, confident && step + 1 < *limit ? 1U : 0U);
}

__global__ void graph_gate_rows_kernel(const float* __restrict__ values, const double* __restrict__ threshold,
                                       const std::int32_t* __restrict__ limits, std::int32_t rows, std::int32_t draft,
                                       cudaGraphConditionalHandle handle) {
    bool run = false;
    for (std::int32_t row = 0; row < rows; ++row) {
        run |= draft < limits[row] && (values == nullptr || !(static_cast<double>(values[row]) < *threshold));
    }
    cudaGraphSetConditional(handle, run ? 1U : 0U);
}

__global__ void graph_gate_product_kernel(const float* __restrict__ values, std::int32_t stride, std::int32_t step,
                                          GraphGateBins bins, const float* __restrict__ threshold,
                                          cudaGraphConditionalHandle handle) {
    cudaGraphSetConditional(handle, graph_gate_product(values, stride, step, bins) < *threshold ? 0U : 1U);
}

} // namespace

void launch_graph_gate_product(const float* values, std::int32_t stride, std::int32_t step, const GraphGateBins& bins,
                               const float* threshold, cudaGraphConditionalHandle handle, cudaStream_t stream) {
    graph_gate_product_kernel<<<1, 1, 0, stream>>>(values, stride, step, bins, threshold, handle);
    CUDA_CHECK(cudaGetLastError());
}

void launch_graph_gate_rows(const float* values, const double* threshold, const std::int32_t* limits,
                            std::int32_t rows, std::int32_t draft, cudaGraphConditionalHandle handle,
                            cudaStream_t stream) {
    graph_gate_rows_kernel<<<1, 1, 0, stream>>>(values, threshold, limits, rows, draft, handle);
    CUDA_CHECK(cudaGetLastError());
}

void launch_graph_gate(const float* value, double threshold, cudaGraphConditionalHandle handle, cudaStream_t stream) {
    // An ordinary launch: the value is read only after the kernel before it has completed.
    graph_gate_kernel<<<1, 1, 0, stream>>>(value, threshold, handle);
    CUDA_CHECK(cudaGetLastError());
}

void launch_graph_gate(const float* value, double threshold, const std::int32_t* limit, std::int32_t step,
                       cudaGraphConditionalHandle handle, cudaStream_t stream) {
    graph_gate_limit_kernel<<<1, 1, 0, stream>>>(value, threshold, limit, step, handle);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix
