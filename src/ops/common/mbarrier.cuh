#pragma once

#include "ops/common/memory.cuh"

#include <cstdint>

namespace infernix::ops {

__device__ __forceinline__ void cta_mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_wait(std::uint64_t* barrier, std::uint32_t phase) {
    constexpr std::uint32_t kSuspendTicks = 0x989680;
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_loop_%=: \n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@done bra wait_done_%=;\n"
                 "bra wait_loop_%=;\n"
                 "wait_done_%=: \n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                 : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive(std::uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier)) : "memory");
}

__device__ __forceinline__ void cta_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                              std::uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
}

// Called by each thread that read a pipeline stage with generic-proxy loads (ld.shared, ldmatrix)
// before its warp releases the stage (the empty-barrier arrive) to a producer that refills it with
// async-proxy copies (cp.async.bulk.tensor or cp.async.bulk). The arrive's release orders the reads
// only against generic-proxy writes; without this fence the refill could overwrite the stage while the
// reads were still in flight (tile-ai/tilelang #3529). On the RTX 5090 such stages came out corrupted,
// a warp's MMA fragment at a time, while shared-memory-heavy kernels shared the SMs (2026-10-20).
__device__ __forceinline__ void fence_reads_before_async_refill() {
    asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
}

__device__ __forceinline__ void cta_mbarrier_fence_init() {
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
}

} // namespace infernix::ops
