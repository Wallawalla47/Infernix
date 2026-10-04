#include "ops/offloaded_sparse_moe/cuda/narrow_expert.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::offloaded_moe {
namespace {

using canon::A4Block;

constexpr int kThreads = 256;

struct Workspace {
    A4Block* x_gate; // [job][kMaxColumns][kGateUpBlocks]
    A4Block* x_up;   // [job][kMaxColumns][kGateUpBlocks]
    A4Block* h;      // [job][kMaxColumns][kHBlocks]
};

__host__ __device__ inline Workspace carve(void* base, int jobs) {
    auto* p           = static_cast<A4Block*>(base);
    const std::size_t x = static_cast<std::size_t>(jobs) * kMaxColumns * kGateUpBlocks;
    return {p, p + x, p + 2 * x};
}

// Four doubled E2M1 codes of row r (0..15) at k = 4q..4q+3 of a unit, packed as signed bytes.
__device__ __forceinline__ int row_quad(const std::uint8_t* unit, int r, int q) {
    const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(unit + 32 * q + 4 * (r & 7));
    const int shift          = r < 8 ? 0 : 4;
    return static_cast<int>(canon::e2m1_x2_quad((word >> shift) & 0x0F0F0F0FU));
}

__device__ __forceinline__ int act_quad(const A4Block& a, int q) {
    return *reinterpret_cast<const int*>(&a.c2[4 * q]);
}

// Exact S = sum_b P_b * Sw_b * Sa_b for row r of row group rg, one column (design §16.2).
__device__ std::int64_t row_sum(const std::uint8_t* matrix, int blocks, int rg, int r, const A4Block* acts) {
    std::int64_t s = 0;
    for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * kUnitBytes;
        const A4Block& a         = acts[b];
        int p                    = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) { p = __dp4a(row_quad(unit, r, q), act_quad(a, q), p); }
        const int pw = p * canon::e4m3_scaled(unit[128 + r]); // |P| <= 2304, Sw < 2^18: fits int32
        s += static_cast<std::int64_t>(pw) * a.scale_scaled;
    }
    return s;
}

__global__ void __launch_bounds__(kThreads) quantize_x_kernel(const NarrowJob* jobs, Workspace ws) {
    const NarrowJob job = jobs[blockIdx.x];
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * kMaxColumns * kGateUpBlocks;
    for (int i = threadIdx.x; i < job.ncols * kGateUpBlocks; i += blockDim.x) {
        const int c = i / kGateUpBlocks, b = i % kGateUpBlocks;
        const std::uint16_t* v = job.x + static_cast<std::size_t>(c) * kHidden + 16 * b;
        ws.x_gate[base + i] = canon::quantize_a4_block(v, job.scales.input_gate);
        ws.x_up[base + i]   = job.scales.input_up == job.scales.input_gate
                                  ? ws.x_gate[base + i]
                                  : canon::quantize_a4_block(v, job.scales.input_up);
    }
}

// One CTA per (16-intermediate unit u, job): row groups 2u and 2u+1, gate rows even, up rows odd.
__global__ void __launch_bounds__(kThreads) gate_up_kernel(const NarrowJob* jobs, Workspace ws) {
    const int u = blockIdx.x;
    const int j = blockIdx.y;
    const NarrowJob job = jobs[j];
    __shared__ std::uint16_t y[kMaxColumns][32];
    __shared__ std::uint16_t h[kMaxColumns][16];
    const std::size_t xbase = static_cast<std::size_t>(j) * kMaxColumns * kGateUpBlocks;
    for (int i = threadIdx.x; i < 32 * job.ncols; i += blockDim.x) {
        const int row = i % 32, c = i / 32;
        const bool gate = (row & 1) == 0;
        const A4Block* acts = (gate ? ws.x_gate : ws.x_up) + xbase + static_cast<std::size_t>(c) * kGateUpBlocks;
        const std::int64_t s = row_sum(job.record, kGateUpBlocks, 2 * u + row / 16, row % 16, acts);
        y[c][row] = canon::a4_row_output(s, gate ? job.scales.alpha_gate : job.scales.alpha_up);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 16 * job.ncols; i += blockDim.x) {
        const int k = i % 16, c = i / 16;
        h[c][k] = canon::swiglu_bf16(y[c][2 * k], y[c][2 * k + 1]);
    }
    __syncthreads();
    if (threadIdx.x < job.ncols) {
        const int c = threadIdx.x;
        ws.h[(static_cast<std::size_t>(j) * kMaxColumns + c) * kHBlocks + u] =
            canon::quantize_a4_block(h[c], job.scales.input_down);
    }
}

// One CTA per (4 down row groups, job): 64 output rows for every column.
__global__ void __launch_bounds__(kThreads) down_kernel(const NarrowJob* jobs, Workspace ws) {
    const int j = blockIdx.y;
    const NarrowJob job = jobs[j];
    const std::uint8_t* down = job.record + kGateUpBytes;
    for (int i = threadIdx.x; i < 64 * job.ncols; i += blockDim.x) {
        const int row = i % 64, c = i / 64;
        const int rg  = 4 * blockIdx.x + row / 16;
        const A4Block* acts = ws.h + (static_cast<std::size_t>(j) * kMaxColumns + c) * kHBlocks;
        const std::int64_t s = row_sum(down, kDownBlocks, rg, row % 16, acts);
        job.y[static_cast<std::size_t>(c) * kHidden + 16 * rg + row % 16] = canon::a4_row_output(s, job.scales.alpha_down);
    }
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string("offloaded_moe narrow route: ") + what + ": " + cudaGetErrorString(e)); }
}

} // namespace

std::size_t narrow_workspace_bytes(int jobs) {
    return static_cast<std::size_t>(jobs) * kMaxColumns * (2 * kGateUpBlocks + kHBlocks) * sizeof(canon::A4Block);
}

void launch_narrow_experts(const NarrowJob* jobs, int job_count, void* workspace, cudaStream_t stream) {
    if (job_count <= 0) { return; }
    if (job_count > 65535) { throw std::invalid_argument("offloaded_moe narrow route: too many jobs"); }
    const Workspace ws = carve(workspace, job_count);
    quantize_x_kernel<<<job_count, kThreads, 0, stream>>>(jobs, ws);
    check(cudaGetLastError(), "quantize");
    gate_up_kernel<<<dim3(kHBlocks, job_count), kThreads, 0, stream>>>(jobs, ws);
    check(cudaGetLastError(), "gate/up");
    down_kernel<<<dim3(kDownRowGroups / 4, job_count), kThreads, 0, stream>>>(jobs, ws);
    check(cudaGetLastError(), "down");
}

} // namespace ninfer::ops::offloaded_moe
