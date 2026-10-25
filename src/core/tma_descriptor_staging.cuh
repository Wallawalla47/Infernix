#pragma once

#include "core/device.h"
#include "core/temporary_switches.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <vector>

namespace infernix {

// Stages an over-aligned (alignas(128)) CUtensorMap descriptor struct into device global memory
// for a kernel launch on Windows, where MSVC cannot pass the struct by value as a
// __grid_constant__ parameter (C2719). The kernel reads the descriptors from the device pointer
// returned by stage() and makes each tensor map visible to the TMA (tensormap) proxy with a
// fence.proxy.tensormap acquire before its first cp.async.bulk.tensor.
//
// An eager launch: MSVC does pass the same bytes as an 8-byte-aligned word array, so stage()
// launches a one-block kernel that takes the words by value and stores them into one persistent
// device buffer. Kernel parameters are copied when the launch is issued, so the launch never
// reads host memory the host may since have rewritten, and the host never waits for the GPU. The
// staging kernel and the TMA kernel that reads the buffer are ordinary stream-ordered launches,
// so a launch's descriptors are stored only after the previous TMA kernel on the stream has
// finished reading them.
//
// A launch being captured into a CUDA Graph: its descriptors never change across replays, so
// they are copied once, while capturing, into a device copy of their own that is never
// rewritten, and the graph carries no staging kernel. (A staging kernel in a decode graph runs
// on every replay: on an RTX 5090 the FP8 TMA decode route's 72 launches per round cost about
// 65 us of each 18-23 ms DFlash2 round.) Copies are keyed by the descriptor bytes, so a graph
// captured again over the same workspace reuses them instead of adding more.
//
// Invariants, documented here because this is the single staging site:
//   - one stream: every eager staged launch runs on the engine's single compute stream, so the
//     persistent device buffer orders against itself in-stream;
//   - allocation and copies outside the graph: buffers are allocated, and captured descriptors
//     copied, in relaxed capture mode by synchronous calls, so a route first reached while a
//     decode graph is being captured still works and the copy has landed before any replay;
//   - nothing is freed: device buffers are reclaimed at process exit.
namespace tma_staging_detail {

template <std::size_t Words>
struct DescriptorWords {
    std::uint64_t words[Words];
};

template <std::size_t Words>
__global__ void store_descriptor_words(const DescriptorWords<Words> source,
                                       std::uint64_t* __restrict__ destination) {
    for (std::size_t word = threadIdx.x; word < Words; word += blockDim.x) {
        destination[word] = source.words[word];
    }
    // The consumer's tensor-map acquire pairs with this release.
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" : : : "memory");
}

} // namespace tma_staging_detail

// Makes the tensor map at `tensor_map` (128 bytes in global memory) visible to the TMA proxy. Call
// once per tensor map, in the thread that issues cp.async.bulk.tensor with it, before its first use.
// System scope: an eager launch's map was stored by the staging kernel (a .gpu release, which a .sys
// acquire also pairs with), but a captured launch's map was written by the host with cudaMemcpy, which
// the CUDA Programming Guide (Asynchronous Data Copies, tensor maps in global memory) pairs with a
// .sys acquire.
// A map read from the kernel's parameters (InPlace, TmaDescriptorArgument's words) is not in global
// memory and needs no acquire.
template <bool InPlace = false>
__device__ __forceinline__ void acquire_staged_tensor_map(const void* tensor_map) {
    if constexpr (!InPlace) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;"
                     :
                     : "l"(tensor_map)
                     : "memory");
    }
}

// TEMPORARY (INFERNIX_TMP_TMA_PARAM_WORDS, candidates §2.3 GX-03): the descriptor words as a Windows
// kernel parameter. MSVC passes the descriptor bytes as an 8-byte-aligned word array by value, which the
// kernel reads as its tensor maps in place (a __grid_constant__ parameter, as Linux passes the struct)
// when the parameter space puts the words on a 64-byte boundary, as the TMA requires: no staging kernel
// then runs before an eager launch. That alignment is not documented, so a probe launch checks it once
// (tma_param_words_aligned).
template <class Descriptor>
struct TmaDescriptorArgument {
    tma_staging_detail::DescriptorWords<sizeof(Descriptor) / sizeof(std::uint64_t)> words{};
    __device__ const Descriptor& operator*() const { return *reinterpret_cast<const Descriptor*>(words.words); }
};

// A Windows TMA kernel's descriptor parameter: the address of the device copy TmaDescriptorStaging made,
// or (InPlace) the words read in place. A template argument rather than a runtime select, so the default
// instantiation compiles exactly as the plain pointer parameter did.
template <class Descriptor, bool InPlace>
using TmaDescriptorParameter = std::conditional_t<InPlace, TmaDescriptorArgument<Descriptor>, const Descriptor*>;

namespace tma_staging_detail {

struct alignas(128) ProbeDescriptor {
    std::uint64_t words[16];
};

// A template, so the header defines it in every translation unit without a duplicate device symbol
// (relocatable device code links them all).
template <class Descriptor = ProbeDescriptor>
__global__ void probe_parameter_alignment(const __grid_constant__ TmaDescriptorArgument<Descriptor> argument,
                                          std::uint32_t* __restrict__ misalignment) {
    *misalignment = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(argument.words.words) % 64U);
}

} // namespace tma_staging_detail

// Whether descriptor words in the first kernel parameter land on a 64-byte boundary (the switch on and
// the probe launch confirming it), once per process.
inline bool tma_param_words_aligned() {
    static const bool aligned = [] {
        if (!temporary_switches::flag("INFERNIX_TMP_TMA_PARAM_WORDS")) { return false; }
        // On a stream of its own, in relaxed capture mode: the first launch may come while this thread
        // captures a decode graph on another stream.
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        cudaStream_t probe          = nullptr;
        std::uint32_t* misalignment = nullptr;
        std::uint32_t host          = 1;
        cudaError_t status          = cudaStreamCreateWithFlags(&probe, cudaStreamNonBlocking);
        if (status == cudaSuccess) { status = cudaMalloc(&misalignment, sizeof(std::uint32_t)); }
        if (status == cudaSuccess) {
            tma_staging_detail::probe_parameter_alignment<><<<1, 1, 0, probe>>>({}, misalignment);
            status = cudaGetLastError();
        }
        if (status == cudaSuccess) { status = cudaMemcpyAsync(&host, misalignment, sizeof(host), cudaMemcpyDeviceToHost, probe); }
        if (status == cudaSuccess) { status = cudaStreamSynchronize(probe); }
        if (misalignment != nullptr) { (void)cudaFree(misalignment); }
        if (probe != nullptr) { (void)cudaStreamDestroy(probe); }
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        CUDA_CHECK(status);
        return host == 0;
    }();
    return aligned;
}

template <class Descriptor>
class TmaDescriptorStaging {
  public:
    static_assert(sizeof(Descriptor) % sizeof(std::uint64_t) == 0);

    // Stages the descriptors for the next launch on `stream` and returns the device pointer
    // the kernel must read them from.
    Descriptor* stage(const Descriptor& descriptors, cudaStream_t stream) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture == cudaStreamCaptureStatusActive) return captured(descriptors);
        if (device_ == nullptr) device_ = allocate(1);
        tma_staging_detail::DescriptorWords<kWords> words;
        std::memcpy(words.words, &descriptors, sizeof(Descriptor));
        tma_staging_detail::store_descriptor_words<kWords>
            <<<1, 64, 0, stream>>>(words, reinterpret_cast<std::uint64_t*>(device_));
        CUDA_CHECK(cudaGetLastError());
        return device_;
    }

  private:
    static constexpr std::size_t kWords         = sizeof(Descriptor) / sizeof(std::uint64_t);
    static constexpr std::size_t kCapturedChunk = 256;
    using Key                                   = std::array<std::uint64_t, kWords>;

    // cudaMalloc is not a stream operation; relaxed mode lets it run while this thread captures,
    // so it never lands in (or aborts) a graph.
    static Descriptor* allocate(std::size_t count) {
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        void* buffer             = nullptr;
        const cudaError_t status = cudaMalloc(&buffer, count * sizeof(Descriptor));
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        CUDA_CHECK(status);
        return static_cast<Descriptor*>(buffer);
    }

    // The device copy a captured launch reads, written now by a synchronous copy (which the
    // relaxed capture mode admits) and never again.
    Descriptor* captured(const Descriptor& descriptors) {
        Key key;
        std::memcpy(key.data(), &descriptors, sizeof(Descriptor));
        const std::lock_guard<std::mutex> lock(mutex_);
        if (const auto found = captured_.find(key); found != captured_.end()) return found->second;
        if (chunks_.empty() || used_ == kCapturedChunk) {
            chunks_.push_back(allocate(kCapturedChunk));
            used_ = 0;
        }
        Descriptor* slot           = chunks_.back() + used_++;
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        const cudaError_t status =
            cudaMemcpy(slot, &descriptors, sizeof(Descriptor), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        CUDA_CHECK(status);
        captured_.emplace(key, slot);
        return slot;
    }

    Descriptor* device_ = nullptr;
    std::mutex mutex_;
    std::map<Key, Descriptor*> captured_;
    std::vector<Descriptor*> chunks_;
    std::size_t used_ = 0;
};

// Calls `launch(std::bool_constant<InPlace>{}, parameter)` once with the descriptor parameter of the
// kernel instantiation to run: the words read in place when tma_param_words_aligned(), else the copy
// staged for the next launch on `stream` (every launch `launch` makes reads it, in stream order).
template <class Descriptor, class Launch>
void with_tma_descriptor_parameter(TmaDescriptorStaging<Descriptor>& staging, const Descriptor& descriptors,
                                   cudaStream_t stream, Launch&& launch) {
    if (tma_param_words_aligned()) {
        TmaDescriptorArgument<Descriptor> words;
        std::memcpy(words.words.words, &descriptors, sizeof(Descriptor));
        launch(std::true_type{}, words);
    } else {
        launch(std::false_type{}, static_cast<const Descriptor*>(staging.stage(descriptors, stream)));
    }
}

} // namespace infernix
