#include "core/decode_graph.h"

#include "core/device.h"

namespace infernix {
namespace {

__global__ void graph_gate_kernel(const float* __restrict__ value, double threshold, cudaGraphConditionalHandle handle) {
    cudaGraphSetConditional(handle, static_cast<double>(*value) < threshold ? 0U : 1U);
}

} // namespace

void launch_graph_gate(const float* value, double threshold, cudaGraphConditionalHandle handle, cudaStream_t stream) {
    // An ordinary launch: the value is read only after the kernel before it has completed.
    graph_gate_kernel<<<1, 1, 0, stream>>>(value, threshold, handle);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix
