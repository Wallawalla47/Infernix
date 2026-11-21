// The LoRA canonical arithmetic's device assumption (include/infernix/ops/lora.h): a BF16 m16n8k16 MMA chain's
// element result depends only on that element's two k-vectors, not on its row or column in the tile, the tile's other
// elements, or which operand holds which vector. Kernels place a (row, token) element anywhere in their fragments and
// in either orientation (weight rows as M or tokens as M); their terms are bit-identical only if this holds.
//
// A pool of BF16 k-vectors with adversarial magnitudes (wide exponents, near-cancellation, LoRA-like scales) fills
// random tiles in both orientations; every (p, q) pair must always give the same FP32 bits, for chains of 1-4 steps.
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr int kK      = 64;
constexpr int kPool   = 96;
constexpr int kTrials = 40000;

__device__ __forceinline__ void probe_mma(float& c0, float& c1, float& c2, float& c3, unsigned a0, unsigned a1,
                                          unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned pack(const __nv_bfloat16* v, int i) {
    return static_cast<unsigned>(__bfloat16_as_ushort(v[i])) |
           (static_cast<unsigned>(__bfloat16_as_ushort(v[i + 1])) << 16);
}

__global__ void probe(const __nv_bfloat16* m_pool, const __nv_bfloat16* n_pool, const int* m_index,
                      const int* n_index, int steps, float* out) {
    const int trial = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (trial >= kTrials) { return; }
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const __nv_bfloat16* row0 = m_pool + m_index[trial * 16 + g] * kK;
    const __nv_bfloat16* row1 = m_pool + m_index[trial * 16 + g + 8] * kK;
    const __nv_bfloat16* col  = n_pool + n_index[trial * 8 + g] * kK;
    float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
    for (int s = 0; s < steps; ++s) {
        const int k = s * 16;
        probe_mma(c0, c1, c2, c3, pack(row0, k + 2 * t), pack(row1, k + 2 * t), pack(row0, k + 2 * t + 8),
                  pack(row1, k + 2 * t + 8), pack(col, k + 2 * t), pack(col, k + 2 * t + 8));
    }
    float* o                   = out + trial * 128;
    o[g * 8 + 2 * t]           = c0;
    o[g * 8 + 2 * t + 1]       = c1;
    o[(g + 8) * 8 + 2 * t]     = c2;
    o[(g + 8) * 8 + 2 * t + 1] = c3;
}

} // namespace

INFERNIX_OP_TEST(lora_mma_position) {
    using namespace infernix;
    if (test::cuda_unavailable()) { return 0; }
    std::mt19937_64 rng(20261120);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::uniform_int_distribution<int> exponent(-20, 20);
    std::vector<std::uint16_t> pool_a(kPool * kK), pool_b(kPool * kK);
    for (int p = 0; p < kPool; ++p) {
        for (int k = 0; k < kK; ++k) {
            float va, vb;
            if (p % 4 < 2) {
                va = std::ldexp(unit(rng), exponent(rng));
                vb = std::ldexp(unit(rng), exponent(rng));
            } else if (p % 4 == 2) {
                const float x = unit(rng);
                va = (k % 2 == 0) ? std::ldexp(x, 10) : std::ldexp(-x, 10) + std::ldexp(unit(rng), -4);
                vb = std::ldexp(1.0F + 0.01F * unit(rng), (k % 7) - 3);
            } else {
                va = 0.02F * unit(rng);
                vb = 3.0F * unit(rng);
            }
            pool_a[p * kK + k] = test::f32_to_bf16(va);
            pool_b[p * kK + k] = test::f32_to_bf16(vb);
        }
    }
    std::uniform_int_distribution<int> pick(0, kPool - 1);
    std::vector<int> m_idx(kTrials * 16), n_idx(kTrials * 8);
    for (auto& v : m_idx) { v = pick(rng); }
    for (auto& v : n_idx) { v = pick(rng); }
    const DeviceBuffer a_d = test::to_device(pool_a), b_d = test::to_device(pool_b);
    const DeviceBuffer mi_d = test::to_device(m_idx), ni_d = test::to_device(n_idx);
    DeviceBuffer out_d(static_cast<std::size_t>(kTrials) * 128 * 4);
    int failures = 0;
    for (int steps = 1; steps <= kK / 16; ++steps) {
        std::vector<std::uint32_t> table(kPool * kPool, 0xFFFFFFFFU);
        long long mismatched = 0;
        for (int orientation = 0; orientation < 2; ++orientation) {
            const auto* m_pool = static_cast<const __nv_bfloat16*>(orientation == 0 ? a_d.p : b_d.p);
            const auto* n_pool = static_cast<const __nv_bfloat16*>(orientation == 0 ? b_d.p : a_d.p);
            probe<<<kTrials / 8, 256>>>(m_pool, n_pool, static_cast<const int*>(mi_d.p), static_cast<const int*>(ni_d.p),
                                        steps, static_cast<float*>(out_d.p));
            test::cuda_check_last_launch("lora mma probe");
            test::cuda_synchronize();
            const std::vector<float> out = test::from_device<float>(out_d, static_cast<std::size_t>(kTrials) * 128);
            for (int trial = 0; trial < kTrials; ++trial) {
                for (int i = 0; i < 16; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        const int m = m_idx[trial * 16 + i], n = n_idx[trial * 8 + j];
                        const int p = orientation == 0 ? m : n, q = orientation == 0 ? n : m;
                        std::uint32_t bits;
                        std::memcpy(&bits, &out[static_cast<std::size_t>(trial) * 128 + i * 8 + j], 4);
                        std::uint32_t& seen = table[p * kPool + q];
                        if (seen == 0xFFFFFFFFU) {
                            seen = bits;
                        } else if (seen != bits) {
                            ++mismatched;
                        }
                    }
                }
            }
        }
        if (mismatched != 0) {
            std::cerr << "lora mma position: " << steps << "-step chains: " << mismatched
                      << " element results differ by position or operand role\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
