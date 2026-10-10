// Resident routed experts (include/infernix/ops/resident_moe.h). One CTA per (row tile, entry):
// each warp streams whole weight rows in 16-byte chunks against the entry's activation staged in
// shared memory as FP32 (swizzled, so lanes on different chunks read different banks), applies each
// chunk's FP16 group scale once, and reduces by a fixed butterfly, so an output's bits depend only
// on the shapes, never on the batch or placement.
#include "infernix/ops/resident_moe.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads       = 256;
constexpr int kWarps         = kThreads / 32;
constexpr int kGateRowsPerCta = kWarps * 2; // intermediate rows (each a gate and an up row)
constexpr int kDownRowsPerCta = kWarps * 4;
constexpr int kMaxHidden      = 4096;
constexpr int kMaxIntermediate = 2048;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("resident_moe_experts: ") + message); }
}

// Row-split codecs: one 16-byte chunk holds Q4's 32 or Q8's 16 consecutive codes of one group.
//
// Lane l of a warp reads chunks l, l + 32, ... of a row, so with the activation stored in order the
// lanes' reads of one chunk element are a whole number of banks apart (a 32- or 16-way conflict).
// The staged activation is swizzled instead: chunk c's float4 q lives at float4 c * kQuads +
// ((q + c / kChunksPerRow) % kQuads), where a row of 32 banks holds kChunksPerRow chunks. A quarter
// warp's eight 16-byte reads then fall in eight distinct bank groups. The products and their order
// are unchanged, so the output bits are too.
struct Q4 {
    static constexpr int kCodes = 32, kGroup = 64, kQuads = kCodes / 4, kChunksPerRow = 1;
    __device__ static float chunk_dot(uint4 w, const float4* x, int c) {
        const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
        float dot = 0.0F;
#pragma unroll
        for (int q = 0; q < kQuads; ++q) {
            const float4 v    = x[(q + c / kChunksPerRow) % kQuads];
            const float xs[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int u = static_cast<int>((words[q / 2] >> (4 * (4 * (q % 2) + j))) & 0xFU);
                dot = fmaf(static_cast<float>(u >= 8 ? u - 16 : u), xs[j], dot);
            }
        }
        return dot;
    }
};

struct Q8 {
    static constexpr int kCodes = 16, kGroup = 32, kQuads = kCodes / 4, kChunksPerRow = 2;
    __device__ static float chunk_dot(uint4 w, const float4* x, int c) {
        const auto* codes = reinterpret_cast<const std::int8_t*>(&w);
        float dot         = 0.0F;
#pragma unroll
        for (int q = 0; q < kQuads; ++q) {
            const float4 v    = x[(q + c / kChunksPerRow) % kQuads];
            const float xs[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) { dot = fmaf(static_cast<float>(codes[4 * q + j]), xs[j], dot); }
        }
        return dot;
    }
};

// Element i of the staged activation, at its swizzled position (in floats).
template <class Codec>
__device__ int staged_index(int i) {
    const int c = i / Codec::kCodes, q = i % Codec::kCodes / 4;
    return c * Codec::kCodes + 4 * ((q + c / Codec::kChunksPerRow) % Codec::kQuads) + i % 4;
}

struct Bank {
    const std::uint8_t* codes;
    const __half* scales;
    int padded_k;
};

// Dot product of row `row` of a bank with the staged (swizzled) FP32 vector x (K = padded_k), reduced
// over the warp.
template <class Codec>
__device__ float row_dot(const Bank& bank, std::int64_t row, const float* x, int lane) {
    const int chunks       = bank.padded_k / Codec::kCodes;
    const int groups       = bank.padded_k / Codec::kGroup;
    const auto* chunks_row = reinterpret_cast<const uint4*>(bank.codes) + row * chunks;
    const __half* scale_row = bank.scales + row * groups;
    float sum = 0.0F;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 w     = __ldcs(chunks_row + c);
        const float scale = __half2float(__ldg(scale_row + c * Codec::kCodes / Codec::kGroup));
        sum = fmaf(Codec::chunk_dot(w, reinterpret_cast<const float4*>(x) + c * Codec::kQuads, c), scale, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset); }
    return sum;
}

template <class Codec>
__global__ void __launch_bounds__(kThreads)
    gate_up_kernel(const bf16* __restrict__ x, const std::int32_t* __restrict__ ids, int top_k, int hidden,
                   int intermediate, Bank bank, float* __restrict__ h) {
    __shared__ __align__(16) float xs[kMaxHidden];
    const int entry = static_cast<int>(blockIdx.y);
    const int t     = entry / top_k;
    const int e     = ids[entry];
    for (int i = static_cast<int>(threadIdx.x); i < bank.padded_k; i += kThreads) {
        xs[staged_index<Codec>(i)] = i < hidden ? __bfloat162float(x[static_cast<std::int64_t>(t) * hidden + i]) : 0.0F;
    }
    __syncthreads();
    const int lane = static_cast<int>(threadIdx.x % 32), warp = static_cast<int>(threadIdx.x / 32);
    for (int i = 0; i < 2; ++i) {
        const int r = static_cast<int>(blockIdx.x) * kGateRowsPerCta + warp * 2 + i;
        if (r >= intermediate) { break; }
        const std::int64_t base = static_cast<std::int64_t>(e) * 2 * intermediate;
        const float g = row_dot<Codec>(bank, base + r, xs, lane);
        const float u = row_dot<Codec>(bank, base + intermediate + r, xs, lane);
        if (lane == 0) { h[static_cast<std::int64_t>(entry) * intermediate + r] = g / (1.0F + __expf(-g)) * u; }
    }
}

template <class Codec>
__global__ void __launch_bounds__(kThreads)
    down_kernel(const float* __restrict__ h, const std::int32_t* __restrict__ ids, int hidden, int intermediate,
                Bank bank, bf16* __restrict__ out) {
    __shared__ __align__(16) float hs[kMaxIntermediate];
    const int entry = static_cast<int>(blockIdx.y);
    const int e     = ids[entry];
    for (int i = static_cast<int>(threadIdx.x); i < bank.padded_k; i += kThreads) {
        hs[staged_index<Codec>(i)] = i < intermediate ? h[static_cast<std::int64_t>(entry) * intermediate + i] : 0.0F;
    }
    __syncthreads();
    const int lane = static_cast<int>(threadIdx.x % 32), warp = static_cast<int>(threadIdx.x / 32);
    for (int i = 0; i < kDownRowsPerCta / kWarps; ++i) {
        const int r = static_cast<int>(blockIdx.x) * kDownRowsPerCta + warp * (kDownRowsPerCta / kWarps) + i;
        if (r >= hidden) { break; }
        const float y = row_dot<Codec>(bank, static_cast<std::int64_t>(e) * hidden + r, hs, lane);
        if (lane == 0) { out[static_cast<std::int64_t>(entry) * hidden + r] = __float2bfloat16_rn(y); }
    }
}

Bank bank_of(const Weight& w) {
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const __half*>(w.scales), w.padded_shape[1]};
}

template <class Codec>
void launch(const Tensor& x, const Tensor& ids, const Weight& gate_up, const Weight& down, int top_k, int hidden,
            int intermediate, float* h, Tensor& outputs, cudaStream_t stream) {
    const int entries = ids.ne[0] * ids.ne[1];
    gate_up_kernel<Codec><<<dim3((intermediate + kGateRowsPerCta - 1) / kGateRowsPerCta, entries), kThreads, 0,
                            stream>>>(static_cast<const bf16*>(x.data), static_cast<const std::int32_t*>(ids.data),
                                      top_k, hidden, intermediate, bank_of(gate_up), h);
    CUDA_CHECK(cudaGetLastError());
    down_kernel<Codec><<<dim3((hidden + kDownRowsPerCta - 1) / kDownRowsPerCta, entries), kThreads, 0, stream>>>(
        h, static_cast<const std::int32_t*>(ids.data), hidden, intermediate, bank_of(down),
        static_cast<bf16*>(outputs.data));
    CUDA_CHECK(cudaGetLastError());
}

bool row_split(const Weight& w, QType qtype) {
    return w.qtype == qtype && w.layout == QuantLayout::RowSplit && w.qdata != nullptr && w.scales != nullptr &&
           w.qhigh == nullptr;
}

} // namespace

std::size_t resident_moe_workspace_bytes(std::int32_t pairs, std::int32_t intermediate) {
    return (static_cast<std::size_t>(pairs) * intermediate * sizeof(float) + 255) / 256 * 256;
}

void resident_moe_experts(const Tensor& x, const Tensor& ids, const Weight& gate_up, const Weight& down,
                          std::int32_t experts, std::int32_t intermediate, void* workspace,
                          std::size_t workspace_bytes, Tensor& outputs, cudaStream_t stream) {
    const int hidden = x.ne[0], columns = x.ne[1], top_k = ids.ne[0];
    require(x.dtype == DType::BF16 && x.is_contiguous() && ids.dtype == DType::I32 && ids.is_contiguous() &&
                outputs.dtype == DType::BF16 && outputs.is_contiguous(),
            "x, ids and outputs must be contiguous BF16/I32/BF16");
    require(ids.ne[1] == columns && top_k > 0 && outputs.ne[0] == hidden && outputs.ne[1] == top_k * columns,
            "ids must be [k, T] and outputs [H, k*T]");
    require(hidden > 0 && hidden <= kMaxHidden && intermediate > 0 && intermediate <= kMaxIntermediate,
            "hidden or intermediate size is out of range");
    require(gate_up.qtype == down.qtype, "both banks must share one codec");
    require(gate_up.n == experts * 2 * intermediate && gate_up.k == hidden && down.n == experts * hidden &&
                down.k == intermediate,
            "bank shapes differ from [E*2I, H] and [E*H, I]");
    require(workspace_bytes >= resident_moe_workspace_bytes(top_k * columns, intermediate) && workspace != nullptr,
            "workspace is too small");
    require(gate_up.padded_shape[1] <= kMaxHidden && down.padded_shape[1] <= kMaxIntermediate &&
                gate_up.padded_shape[1] >= hidden && down.padded_shape[1] >= intermediate,
            "padded rows exceed the staged activation");
    auto* h = static_cast<float*>(workspace);
    if (row_split(gate_up, QType::Q4_G64_FP16) && row_split(down, QType::Q4_G64_FP16)) {
        require(gate_up.padded_shape[1] % 64 == 0 && down.padded_shape[1] % 64 == 0, "Q4 rows must hold whole groups");
        launch<Q4>(x, ids, gate_up, down, top_k, hidden, intermediate, h, outputs, stream);
    } else if (row_split(gate_up, QType::Q8_G32_FP16) && row_split(down, QType::Q8_G32_FP16)) {
        require(gate_up.padded_shape[1] % 32 == 0 && down.padded_shape[1] % 32 == 0, "Q8 rows must hold whole groups");
        launch<Q8>(x, ids, gate_up, down, top_k, hidden, intermediate, h, outputs, stream);
    } else {
        throw std::invalid_argument("resident_moe_experts: banks must be row-split Q4_G64_FP16 or Q8_G32_FP16");
    }
}

} // namespace infernix::ops
