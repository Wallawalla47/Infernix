#include "infernix/ops/rows.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr int kMaxOutputs = 8;
constexpr int kThreads    = 256;

struct SplitTargets {
    __nv_bfloat16* data[kMaxOutputs];
    int begin[kMaxOutputs + 1];
    int count;
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("rows: ") + message); }
}

bool contiguous_bf16(const Tensor& t) {
    return t.data != nullptr && t.dtype == DType::BF16 && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1;
}

__global__ void split_kernel(const __nv_bfloat16* __restrict__ in, int rows, SplitTargets targets) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (r >= rows) { return; }
    int i = 0;
    while (r >= targets.begin[i + 1]) { ++i; }
    const int width = targets.begin[i + 1] - targets.begin[i];
    targets.data[i][static_cast<std::size_t>(t) * width + (r - targets.begin[i])] =
        in[static_cast<std::size_t>(t) * rows + r];
}

// Eight rows (16 bytes) per thread: every output boundary, the row count and every pointer are
// multiples of eight elements, so a thread's rows lie in one output and both accesses are aligned.
__global__ void split_vector_kernel(const uint4* __restrict__ in, int rows8, SplitTargets targets) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (r >= rows8) { return; }
    const int row = r * 8;
    int i         = 0;
    while (row >= targets.begin[i + 1]) { ++i; }
    const int width = targets.begin[i + 1] - targets.begin[i];
    auto* out       = reinterpret_cast<uint4*>(targets.data[i]);
    out[(static_cast<std::size_t>(t) * width + (row - targets.begin[i])) / 8] = in[static_cast<std::size_t>(t) * rows8 + r];
}

__global__ void gather_kernel(const __nv_bfloat16* __restrict__ in, int rows,
                              const std::int32_t* __restrict__ columns, __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (r < rows) {
        out[static_cast<std::size_t>(i) * rows + r] = in[static_cast<std::size_t>(columns[i]) * rows + r];
    }
}

} // namespace

void split_rows(const Tensor& in, std::span<Tensor* const> outs, cudaStream_t stream) {
    require(contiguous_bf16(in) && !outs.empty() && outs.size() <= kMaxOutputs, "split needs BF16 [R, T] and 1..8 outputs");
    SplitTargets targets{};
    targets.count    = static_cast<int>(outs.size());
    targets.begin[0] = 0;
    for (std::size_t i = 0; i < outs.size(); ++i) {
        require(contiguous_bf16(*outs[i]) && outs[i]->ne[1] == in.ne[1], "split outputs must be BF16 [r_i, T]");
        targets.data[i]      = static_cast<__nv_bfloat16*>(outs[i]->data);
        targets.begin[i + 1] = targets.begin[i] + outs[i]->ne[0];
    }
    for (std::size_t i = outs.size(); i < kMaxOutputs; ++i) { targets.begin[i + 1] = targets.begin[outs.size()]; }
    require(targets.begin[outs.size()] == in.ne[0], "split outputs must cover the input rows");
    const auto aligned = [](const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; };
    bool vector = in.ne[0] % 8 == 0 && aligned(in.data);
    for (std::size_t i = 0; i < outs.size(); ++i) {
        vector = vector && targets.begin[i + 1] % 8 == 0 && aligned(targets.data[i]);
    }
    if (vector) {
        const int rows8 = in.ne[0] / 8;
        split_vector_kernel<<<dim3((rows8 + kThreads - 1) / kThreads, in.ne[1]), kThreads, 0, stream>>>(
            static_cast<const uint4*>(in.data), rows8, targets);
    } else {
        split_kernel<<<dim3((in.ne[0] + kThreads - 1) / kThreads, in.ne[1]), kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(in.data), in.ne[0], targets);
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("split_rows: ") + cudaGetErrorString(error)); }
}

void gather_columns(const Tensor& in, const Tensor& columns, Tensor& out, cudaStream_t stream) {
    require(contiguous_bf16(in) && contiguous_bf16(out) && columns.dtype == DType::I32 && columns.is_contiguous() &&
                out.ne[0] == in.ne[0] && columns.numel() == out.ne[1],
            "gather needs BF16 [R, T] in, BF16 [R, N] out and I32 [N] columns");
    gather_kernel<<<dim3((in.ne[0] + kThreads - 1) / kThreads, out.ne[1]), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(in.data), in.ne[0], static_cast<const std::int32_t*>(columns.data),
        static_cast<__nv_bfloat16*>(out.data));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("gather_columns: ") + cudaGetErrorString(error)); }
}

} // namespace infernix::ops
