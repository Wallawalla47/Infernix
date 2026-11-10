#include "ops/offloaded_sparse_moe/cuda/csf_expand.cuh"

#include "core/device.h"

#include <stdexcept>

namespace infernix::ops::offloaded_moe::csf {
namespace {

constexpr int kInPlaceThreads = 256;

// One CTA keeps its record's table words (kMaxTableWords: the 122B's 321).
struct InPlaceShared {
    ChunkBuffer buffer[2];
    ChunkScratch scratch;
    std::uint32_t words[kMaxTableWords];
};

// One CTA a record: its chunks in descending order, each loaded whole into shared memory before any of its units is
// written. Chunk k's expanded units lie at or after its compressed regions and before chunk k + 1's expanded units,
// so they cover only compressed bytes of chunks >= k, already read; chunk k - 1's compressed regions end at or before
// chunk k's expanded start, so the next chunk is loaded while this one is written.
__global__ void __launch_bounds__(kInPlaceThreads) expand_in_place_kernel(Geometry g, InPlaceJobs jobs) {
    extern __shared__ __align__(16) std::uint8_t dynamic[];
    auto& s = *reinterpret_cast<InPlaceShared*>(dynamic);
    if (static_cast<int>(blockIdx.x) >= jobs.count) { return; }
    const InPlaceJob job = jobs.job[blockIdx.x];
    const int words      = g.table_words();
    for (int i = static_cast<int>(threadIdx.x); i < words; i += kInPlaceThreads) { s.words[i] = __ldcv(job.table + i); }
    __syncthreads();
    const int chunks = g.chunks();
    const auto load  = [&](int c, int buffer) {
        const int r0 = g.chunk_first_rg(c);
        load_chunk_async<kInPlaceThreads>(job.record + s.words[r0], s.words[r0 + g.chunk_rgs(c)] - s.words[r0],
                                          s.buffer[buffer].data);
    };
    load(chunks - 1, 0);
    for (int c = chunks - 1, buffer = 0; c >= 0; --c, buffer ^= 1) {
        if (c > 0) {
            load(c - 1, buffer ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        }
        __syncthreads();
        expand_chunk<kInPlaceThreads>(g, c, s.words + g.chunk_first_rg(c), s.buffer[buffer].data, job.record, s.scratch);
    }
}

} // namespace

void expand_in_place(const ExpertShape& shape, const InPlaceJobs& jobs, cudaStream_t stream) {
    if (jobs.count <= 0) { return; }
    if (!registered_shape(shape) || jobs.count > kMaxInPlaceJobs) { throw std::invalid_argument("csf: bad in-place expansion"); }
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(expand_in_place_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(sizeof(InPlaceShared))));
        return true;
    }();
    (void)configured;
    expand_in_place_kernel<<<jobs.count, kInPlaceThreads, sizeof(InPlaceShared), stream>>>(Geometry::of(shape), jobs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::offloaded_moe::csf
