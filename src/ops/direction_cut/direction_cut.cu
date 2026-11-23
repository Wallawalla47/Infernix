// Direction cut and inject (include/infernix/ops/direction_cut.h; --ablate, docs/ablate.md). Both kernels take one
// column per CTA, eight hidden elements per thread (16-byte loads), so a column's dot product with the direction is
// a block reduction in a fixed order: the thread's eight products in ascending h, a shuffle-down tree per warp, then
// the warp sums in ascending order. Every load of a column issues before its arithmetic (one L2 round trip, as
// moe_combine_inject). The plain inject kernels (hyper_connection.cu, moe_layer.cu) are not touched: a call with no
// cut runs them.

#include "infernix/ops/direction_cut.h"
#include "infernix/ops/hyper_connection.h"

#include "core/grid_rows.h"
#include "core/pdl.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kVector     = 8;  // hidden elements a thread owns: one 16-byte BF16 load
// 512 threads (H up to 4096; Flash-Next's 2560 takes 320) leave a thread 128 registers: the combine holds a column's
// top-10 expert vectors (four registers each) in flight before it sums them. A 1024-thread bound capped it at 64 and
// spilled them to local memory.
constexpr int kMaxThreads = 512;
constexpr int kMaxWarps   = kMaxThreads / 32;
constexpr int kMaxTopK    = 16;
constexpr int kBatch      = 8; // expert vectors a thread holds at once (Flash-Next's top-10: two batches)
constexpr int kMaxStreams = 4;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("direction_cut: ") + message); }
}

void check_launch(cudaError_t error, const char* what) {
    if (error == cudaSuccess) { error = cudaGetLastError(); }
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("direction_cut ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1;
}

// The BF16 tensors the kernels read and write in 16-byte vectors (H a multiple of 8 keeps every column aligned).
bool aligned16(const Tensor& t) { return reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0; }

struct Bf16x8 {
    float v[kVector];
};

// Element i (0..7, a compile-time index after unrolling) of eight packed BF16 values, widened exactly. Register
// selects and shifts only: taking a vector's address to index it could place it in local memory.
__device__ __forceinline__ float bf16_lane(const uint4& v, int i) {
    const unsigned word = i < 2 ? v.x : (i < 4 ? v.y : (i < 6 ? v.z : v.w));
    return __uint_as_float((i & 1) != 0 ? (word & 0xFFFF0000U) : (word << 16));
}

__device__ __forceinline__ Bf16x8 load_bf16x8(const bf16* p) {
    const uint4 raw = *reinterpret_cast<const uint4*>(p);
    Bf16x8 out;
#pragma unroll
    for (int i = 0; i < kVector; ++i) { out.v[i] = bf16_lane(raw, i); }
    return out;
}

__device__ __forceinline__ unsigned bf16_bits(float x) {
    return static_cast<unsigned>(__bfloat16_as_ushort(__float2bfloat16_rn(x)));
}

__device__ __forceinline__ uint4 pack_bf16x8(const float* v) {
    return make_uint4(bf16_bits(v[0]) | (bf16_bits(v[1]) << 16), bf16_bits(v[2]) | (bf16_bits(v[3]) << 16),
                      bf16_bits(v[4]) | (bf16_bits(v[5]) << 16), bf16_bits(v[6]) | (bf16_bits(v[7]) << 16));
}

// The column's c = sum_h dir[h] * y[h] in the fixed order above, the same value in every thread. `scratch` holds a
// float per warp; the caller's next column may reuse it (the trailing barrier).
__device__ __forceinline__ float column_dot(const float* y, const float* d, bool owns, float* scratch) {
    float partial = 0.0F;
    if (owns) {
#pragma unroll
        for (int i = 0; i < kVector; ++i) { partial = __fmaf_rn(d[i], y[i], partial); }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) { partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset); }
    if ((threadIdx.x & 31) == 0) { scratch[threadIdx.x >> 5] = partial; }
    __syncthreads();
    float c = scratch[0];
    for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w) { c += scratch[w]; }
    __syncthreads();
    return c;
}

// residual[s*H + h, t] = bf16(residual + inject[s, t] * (y - g * dir)) for the thread's eight elements.
__device__ __forceinline__ void cut_and_inject(const float* y, const float* d, float g, const float* inject,
                                               int streams, bf16* residual_column, int hidden, int h0) {
    float v[kVector];
#pragma unroll
    for (int i = 0; i < kVector; ++i) { v[i] = __fmaf_rn(-g, d[i], y[i]); }
    Bf16x8 r[kMaxStreams];
#pragma unroll
    for (int s = 0; s < kMaxStreams; ++s) {
        if (s < streams) { r[s] = load_bf16x8(residual_column + static_cast<std::size_t>(s) * hidden + h0); }
    }
#pragma unroll
    for (int s = 0; s < kMaxStreams; ++s) {
        if (s < streams) {
            float out[kVector];
#pragma unroll
            for (int i = 0; i < kVector; ++i) { out[i] = __fmaf_rn(inject[s], v[i], r[s].v[i]); }
            *reinterpret_cast<uint4*>(residual_column + static_cast<std::size_t>(s) * hidden + h0) =
                pack_bf16x8(out);
        }
    }
}

// Producer-written inputs carry no __restrict__: a non-coherent load could be issued above the PDL wait.
__global__ void __launch_bounds__(kMaxThreads)
    cut_inject_kernel(const bf16* y, const float* direction, float alpha, const float* inject, int hidden,
                      int streams, int columns, bf16* residual) {
    __shared__ float scratch[kMaxWarps];
    pdl::enter();
    const int h0    = static_cast<int>(threadIdx.x) * kVector;
    const bool owns = h0 < hidden;
    float d[kVector] = {};
    if (owns) {
        const float4* dv = reinterpret_cast<const float4*>(direction + h0);
        const float4 a = dv[0], b = dv[1];
        d[0] = a.x, d[1] = a.y, d[2] = a.z, d[3] = a.w, d[4] = b.x, d[5] = b.y, d[6] = b.z, d[7] = b.w;
    }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int t = blockIdx.y; t < columns; t += gridDim.y) {
        Bf16x8 yv{};
        if (owns) { yv = load_bf16x8(y + static_cast<std::size_t>(t) * hidden + h0); }
        float a[kMaxStreams];
#pragma unroll
        for (int s = 0; s < kMaxStreams; ++s) { a[s] = s < streams ? inject[static_cast<std::size_t>(t) * streams + s] : 0.0F; }
        const float c = column_dot(yv.v, d, owns, scratch);
        if (owns) {
            cut_and_inject(yv.v, d, alpha * c, a, streams, residual + static_cast<std::size_t>(t) * width, hidden, h0);
        }
    }
}

__global__ void __launch_bounds__(kMaxThreads)
    combine_cut_inject_kernel(const bf16* outputs, const float* weights, const float* shared_gate, const bf16* shared,
                              const float* direction, float alpha, const float* inject, int hidden, int top_k,
                              int streams, int columns, bf16* residual) {
    __shared__ float scratch[kMaxWarps];
    pdl::enter();
    const int h0    = static_cast<int>(threadIdx.x) * kVector;
    const bool owns = h0 < hidden;
    float d[kVector] = {};
    if (owns) {
        const float4* dv = reinterpret_cast<const float4*>(direction + h0);
        const float4 a = dv[0], b = dv[1];
        d[0] = a.x, d[1] = a.y, d[2] = a.z, d[3] = a.w, d[4] = b.x, d[5] = b.y, d[6] = b.z, d[7] = b.w;
    }
    const std::size_t width = static_cast<std::size_t>(streams) * hidden;
    for (int t = blockIdx.y; t < columns; t += gridDim.y) {
        // y = bf16(sum_k w_k o_k + gate * shared), k in rank order (combine_kernel's FMA chain, so its bits).
        float y[kVector] = {};
        if (owns) {
            const float gate = shared_gate[t];
            const Bf16x8 own = load_bf16x8(shared + static_cast<std::size_t>(t) * hidden + h0);
            // The experts' outputs in batches of kBatch vectors, packed (four registers each) until the sums read
            // them: a batch's loads all issue before its sums, and the sums keep rank order across batches.
            float sum[kVector] = {};
#pragma unroll
            for (int k0 = 0; k0 < kMaxTopK; k0 += kBatch) {
                if (k0 >= top_k) { break; }
                uint4 o[kBatch];
                float w[kBatch];
#pragma unroll
                for (int j = 0; j < kBatch; ++j) {
                    if (k0 + j < top_k) {
                        const std::size_t entry = static_cast<std::size_t>(t) * top_k + k0 + j;
                        w[j] = weights[entry];
                        o[j] = *reinterpret_cast<const uint4*>(outputs + entry * hidden + h0);
                    }
                }
#pragma unroll
                for (int j = 0; j < kBatch; ++j) {
                    if (k0 + j < top_k) {
#pragma unroll
                        for (int i = 0; i < kVector; ++i) { sum[i] = __fmaf_rn(w[j], bf16_lane(o[j], i), sum[i]); }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < kVector; ++i) {
                y[i] = __uint_as_float(bf16_bits(__fmaf_rn(gate, own.v[i], sum[i])) << 16);
            }
        }
        float a[kMaxStreams];
#pragma unroll
        for (int s = 0; s < kMaxStreams; ++s) { a[s] = s < streams ? inject[static_cast<std::size_t>(t) * streams + s] : 0.0F; }
        const float c = column_dot(y, d, owns, scratch);
        if (owns) {
            cut_and_inject(y, d, alpha * c, a, streams, residual + static_cast<std::size_t>(t) * width, hidden, h0);
        }
    }
}

// The single-stream residual (Qwen3.5), after the lane's own write: x[h, t] = bf16(x - alpha * c * dir) with c the
// direction's component of x - before. A plain stream-ordered launch; a thread holds a few registers only, so 1024
// threads (H up to 8192) cost no spill.
constexpr int kMaxSingleThreads = 1024;
__global__ void __launch_bounds__(kMaxSingleThreads)
    residual_delta_cut_kernel(const bf16* before, const float* direction, float alpha, int hidden, int columns, bf16* x) {
    __shared__ float scratch[kMaxSingleThreads / 32];
    const int h0    = static_cast<int>(threadIdx.x) * kVector;
    const bool owns = h0 < hidden;
    float d[kVector] = {};
    if (owns) {
        const float4* dv = reinterpret_cast<const float4*>(direction + h0);
        const float4 a = dv[0], b = dv[1];
        d[0] = a.x, d[1] = a.y, d[2] = a.z, d[3] = a.w, d[4] = b.x, d[5] = b.y, d[6] = b.z, d[7] = b.w;
    }
    for (int t = blockIdx.y; t < columns; t += gridDim.y) {
        Bf16x8 bv{}, xv{};
        if (owns) {
            bv = load_bf16x8(before + static_cast<std::size_t>(t) * hidden + h0);
            xv = load_bf16x8(x + static_cast<std::size_t>(t) * hidden + h0);
        }
        float delta[kVector];
#pragma unroll
        for (int i = 0; i < kVector; ++i) { delta[i] = xv.v[i] - bv.v[i]; } // exact within 2^16 of magnitude
        const float g = alpha * column_dot(delta, d, owns, scratch);
        if (owns) {
            float out[kVector];
#pragma unroll
            for (int i = 0; i < kVector; ++i) { out[i] = __fmaf_rn(-g, d[i], xv.v[i]); }
            *reinterpret_cast<uint4*>(x + static_cast<std::size_t>(t) * hidden + h0) = pack_bf16x8(out);
        }
    }
}

int threads_for(int hidden) { return (hidden / kVector + 31) / 32 * 32; }

void require_cut(const DirectionCut& cut, const Tensor& inject, const Tensor& residual, int hidden, int columns) {
    require(hidden > 0 && hidden % kVector == 0 && hidden / kVector <= kMaxThreads,
            "the hidden width must be a positive multiple of 8, at most 4096");
    require(cut.direction != nullptr && reinterpret_cast<std::uintptr_t>(cut.direction) % 16 == 0,
            "the direction must be a 16-byte aligned device vector");
    require(cut.alpha > 0.0F && cut.alpha <= 4.0F, "alpha must be in (0, 4]");
    require(contiguous_2d(inject, DType::FP32) && contiguous_2d(residual, DType::BF16) && aligned16(residual),
            "inject requires contiguous tensors and a 16-byte aligned residual");
    const int streams = inject.ne[0];
    require(streams > 0 && streams <= kMaxStreams && inject.ne[1] == columns &&
                residual.ne[0] == static_cast<std::int64_t>(streams) * hidden && residual.ne[1] == columns &&
                columns > 0,
            "inject shapes disagree (at most 4 streams)");
}

} // namespace

void hyper_connection_cut_inject(const Tensor& y, const DirectionCut& cut, const Tensor& inject, Tensor& residual,
                                 cudaStream_t stream) {
    if (!cut.active()) {
        hyper_connection_inject(y, inject, residual, stream);
        return;
    }
    require(contiguous_2d(y, DType::BF16) && aligned16(y), "y must be a contiguous, 16-byte aligned BF16 [H, T] tensor");
    const int hidden = y.ne[0], columns = y.ne[1];
    require_cut(cut, inject, residual, hidden, columns);
    check_launch(pdl::launch_consumer({dim3(1, grid_rows(columns)), dim3(threads_for(hidden)), 0, stream},
                                      cut_inject_kernel, static_cast<const bf16*>(y.data), cut.direction, cut.alpha,
                                      static_cast<const float*>(inject.data), hidden, static_cast<int>(inject.ne[0]),
                                      columns, static_cast<bf16*>(residual.data)),
                 "cut inject");
}

void moe_combine_cut_inject(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared,
                            const DirectionCut& cut, const Tensor& inject, Tensor& residual, cudaStream_t stream) {
    if (!cut.active()) {
        moe_combine_inject(outputs, routing, shared, inject, residual, stream);
        return;
    }
    require(contiguous_2d(outputs, DType::BF16) && contiguous_2d(shared, DType::BF16) && aligned16(outputs) &&
                aligned16(shared) && contiguous_2d(routing.weights, DType::FP32) &&
                contiguous_2d(routing.shared_gate, DType::FP32),
            "combine cut inject requires contiguous, 16-byte aligned BF16 outputs and shared, FP32 routing weights");
    require(routing.mask.valid == nullptr, "combine cut inject takes no column mask");
    const int hidden = shared.ne[0], columns = shared.ne[1], top_k = routing.weights.ne[0];
    require(top_k > 0 && top_k <= kMaxTopK && outputs.ne[0] == hidden &&
                outputs.ne[1] == static_cast<std::int64_t>(columns) * top_k && routing.weights.ne[1] == columns &&
                routing.shared_gate.numel() == columns,
            "combine cut inject shapes disagree (at most top-16)");
    require_cut(cut, inject, residual, hidden, columns);
    check_launch(pdl::launch_consumer({dim3(1, grid_rows(columns)), dim3(threads_for(hidden)), 0, stream},
                                      combine_cut_inject_kernel, static_cast<const bf16*>(outputs.data),
                                      static_cast<const float*>(routing.weights.data),
                                      static_cast<const float*>(routing.shared_gate.data),
                                      static_cast<const bf16*>(shared.data), cut.direction, cut.alpha,
                                      static_cast<const float*>(inject.data), hidden, top_k,
                                      static_cast<int>(inject.ne[0]), columns, static_cast<bf16*>(residual.data)),
                 "combine cut inject");
}

void residual_delta_cut(const Tensor& before, const DirectionCut& cut, Tensor& x, cudaStream_t stream) {
    if (!cut.active()) { return; }
    require(contiguous_2d(before, DType::BF16) && contiguous_2d(x, DType::BF16) && aligned16(before) && aligned16(x),
            "residual delta cut requires contiguous, 16-byte aligned BF16 before and x");
    const int hidden = x.ne[0], columns = x.ne[1];
    require(before.ne[0] == hidden && before.ne[1] == columns && columns > 0, "residual delta cut shapes disagree");
    require(hidden > 0 && hidden % kVector == 0 && hidden / kVector <= kMaxSingleThreads,
            "the hidden width must be a positive multiple of 8, at most 8192");
    require(cut.direction != nullptr && reinterpret_cast<std::uintptr_t>(cut.direction) % 16 == 0,
            "the direction must be a 16-byte aligned device vector");
    require(cut.alpha > 0.0F && cut.alpha <= 4.0F, "alpha must be in (0, 4]");
    residual_delta_cut_kernel<<<dim3(1, grid_rows(columns)), dim3(threads_for(hidden)), 0, stream>>>(
        static_cast<const bf16*>(before.data), cut.direction, cut.alpha, hidden, columns, static_cast<bf16*>(x.data));
    check_launch(cudaSuccess, "residual delta cut");
}

} // namespace infernix::ops
