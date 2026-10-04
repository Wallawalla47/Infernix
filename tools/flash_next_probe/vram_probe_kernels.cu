// Device side of vram_probe.cpp: a spin kernel (a known amount of GPU work for launch-latency and
// stall measurements) and two read kernels (sequential streaming and a gather of whole expert-sized
// records) for the VMM-against-cudaMalloc read parity check.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void spin_kernel(long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {}
}

__global__ void stream_kernel(const uint4* data, std::size_t count, unsigned* sink) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint4 v = data[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9E3779B9U) { *sink = acc; }
}

// Block b reads slice (b % slices) of record records[b / slices]; a record is `record_bytes`
// (a multiple of 16) at base + record * stride.
__global__ void gather_kernel(const unsigned char* base, std::size_t stride, const unsigned* records,
                              std::size_t record_bytes, unsigned slices, unsigned* sink) {
    const unsigned record = records[blockIdx.x / slices];
    const unsigned slice  = blockIdx.x % slices;
    const std::size_t vectors = record_bytes / 16;
    const std::size_t per     = (vectors + slices - 1) / slices;
    const std::size_t begin   = per * slice;
    const std::size_t end     = begin + per < vectors ? begin + per : vectors;
    const auto* data          = reinterpret_cast<const uint4*>(base + static_cast<std::size_t>(record) * stride);
    unsigned acc              = 0;
    for (std::size_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const uint4 v = data[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9E3779B9U) { *sink = acc; }
}

} // namespace

cudaError_t probe_spin(long long cycles, cudaStream_t stream) {
    spin_kernel<<<1, 32, 0, stream>>>(cycles);
    return cudaGetLastError();
}

cudaError_t probe_stream_read(const void* data, std::size_t bytes, unsigned* sink, cudaStream_t stream) {
    stream_kernel<<<170 * 4, 512, 0, stream>>>(static_cast<const uint4*>(data), bytes / 16, sink);
    return cudaGetLastError();
}

cudaError_t probe_gather_read(const void* base, std::size_t stride, const unsigned* records, unsigned count,
                              std::size_t record_bytes, unsigned slices, unsigned* sink, cudaStream_t stream) {
    gather_kernel<<<count * slices, 256, 0, stream>>>(static_cast<const unsigned char*>(base), stride, records,
                                                      record_bytes, slices, sink);
    return cudaGetLastError();
}
