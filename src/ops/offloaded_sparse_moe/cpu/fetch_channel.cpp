#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infernix::ops::offloaded_moe {
namespace {

void cuda_require(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("fetch channel ") + what + ": " + cudaGetErrorString(error));
    }
}

void* mapped(std::size_t bytes) {
    void* p = nullptr;
    cuda_require(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
    std::memset(p, 0, bytes);
    return p;
}

template <class T> void store(T* word, T value) {
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile T*>(word) = value;
}

} // namespace

FetchChannel::FetchChannel() {
    request_   = static_cast<FetchRequest*>(mapped(sizeof(FetchRequest)));
    response_  = static_cast<FetchResponse*>(mapped(sizeof(FetchResponse)));
    consumed_  = static_cast<std::uint64_t*>(mapped(64));
    heartbeat_ = reinterpret_cast<std::uint32_t*>(consumed_ + 2);
    cuda_require(cudaMalloc(&sequence_, sizeof(std::uint32_t)), "cudaMalloc");
    cuda_require(cudaMemset(sequence_, 0, sizeof(std::uint32_t)), "cudaMemset");
    experts_.reserve(kMaxFetch);
    arrived_.assign(kMaxFetch, 0);
}

FetchChannel::~FetchChannel() {
    cudaFree(sequence_);
    cudaFreeHost(consumed_);
    cudaFreeHost(response_);
    cudaFreeHost(request_);
}

MoeFetchChannel FetchChannel::channel(int layer) const {
    MoeFetchChannel out; // mapped memory is device-addressable at its host address (UVA)
    out.request   = request_;
    out.response  = response_;
    out.consumed  = consumed_;
    out.sequence  = sequence_;
    out.heartbeat = heartbeat_;
    out.layer     = layer;
    return out;
}

bool FetchChannel::poll(Request& out) {
    const std::uint32_t sequence = *reinterpret_cast<const volatile std::uint32_t*>(&request_->sequence);
    if (sequence == seen_) { return false; }
    std::atomic_thread_fence(std::memory_order_acquire);
    seen_  = sequence;
    count_ = static_cast<std::uint32_t>(std::clamp(request_->count, 0, kMaxFetch));
    landed_ = 0;
    experts_.assign(request_->expert, request_->expert + count_);
    std::fill(arrived_.begin(), arrived_.begin() + count_, 0);
    store(&response_->landed, 0U);
    store(&response_->status, 0U);
    store(&response_->sequence, sequence);
    out.sequence = sequence;
    out.layer    = request_->layer;
    out.experts  = experts_;
    return true;
}

void FetchChannel::land(std::uint32_t index, const std::uint8_t* record) {
    if (index >= count_ || arrived_[index] != 0) { throw std::logic_error("fetch channel: landing outside the request"); }
    response_->record[index] = reinterpret_cast<std::uint64_t>(record);
    arrived_[index]          = 1;
    std::uint32_t landed     = landed_;
    while (landed < count_ && arrived_[landed] != 0) { ++landed; }
    if (landed != landed_) {
        landed_ = landed;
        store(&response_->landed, landed);
    }
}

void FetchChannel::fail(std::uint32_t status) { store(&response_->status, status != 0 ? status : 1U); }

std::uint32_t FetchChannel::consumed() const noexcept {
    const std::uint64_t word = *reinterpret_cast<const volatile std::uint64_t*>(consumed_);
    return static_cast<std::uint32_t>(word >> 32U) == seen_ ? static_cast<std::uint32_t>(word) : 0U;
}

void FetchChannel::beat() noexcept { *reinterpret_cast<volatile std::uint32_t*>(heartbeat_) = ++beat_; }

} // namespace infernix::ops::offloaded_moe
