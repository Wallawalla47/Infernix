#include "core/vmm_arena.h"

#include "core/cuda_vmm.h"

#include <stdexcept>

namespace infernix {

VmmArena::VmmArena(int device, std::size_t reserve_bytes, std::size_t chunk_bytes)
    : device_(device), chunk_(chunk_bytes) {
    if (!vmm::supported(device)) { throw std::runtime_error("VMM arena: the device lacks virtual memory management"); }
    const std::size_t granularity = vmm::granularity(device);
    if (chunk_bytes == 0 || chunk_bytes % granularity != 0) {
        throw std::invalid_argument("VMM arena: the chunk is not a multiple of the allocation granularity");
    }
    reserved_ = (reserve_bytes + chunk_ - 1) / chunk_ * chunk_;
    if (reserved_ == 0) { throw std::invalid_argument("VMM arena: nothing to reserve"); }
    vmm::check(cuMemAddressReserve(&base_, reserved_, chunk_, 0, 0), "cuMemAddressReserve");
}

VmmArena::~VmmArena() {
    while (!chunks_.empty()) {
        const std::size_t offset = (chunks_.size() - 1) * chunk_;
        (void)cuMemUnmap(base_ + offset, chunk_);
        (void)cuMemRelease(chunks_.back());
        chunks_.pop_back();
    }
    if (base_ != 0) { (void)cuMemAddressFree(base_, reserved_); }
}

bool VmmArena::supported(int device) { return vmm::supported(device); }

bool VmmArena::map_chunk() {
    if (mapped_bytes() + chunk_ > reserved_) { return false; }
    const CUmemAllocationProp prop = vmm::device_prop(device_);
    CUmemGenericAllocationHandle handle{};
    const CUresult created = cuMemCreate(&handle, chunk_, &prop, 0);
    if (created == CUDA_ERROR_OUT_OF_MEMORY) { return false; }
    vmm::check(created, "cuMemCreate");
    const CUdeviceptr va = base_ + mapped_bytes();
    if (const CUresult mapped = cuMemMap(va, chunk_, 0, handle, 0); mapped != CUDA_SUCCESS) {
        (void)cuMemRelease(handle);
        vmm::check(mapped, "cuMemMap");
    }
    try {
        vmm::set_access(va, chunk_, device_);
    } catch (...) {
        (void)cuMemUnmap(va, chunk_);
        (void)cuMemRelease(handle);
        throw;
    }
    chunks_.push_back(handle);
    return true;
}

VmmRange::VmmRange(int device, std::size_t reserve_bytes, std::size_t chunk_bytes)
    : device_(device), chunk_(chunk_bytes) {
    if (!vmm::supported(device)) { throw std::runtime_error("VMM range: the device lacks virtual memory management"); }
    const std::size_t granularity = vmm::granularity(device);
    if (chunk_bytes == 0 || chunk_bytes % granularity != 0) {
        throw std::invalid_argument("VMM range: the chunk is not a multiple of the allocation granularity");
    }
    reserved_ = (reserve_bytes + chunk_ - 1) / chunk_ * chunk_;
    if (reserved_ == 0) { throw std::invalid_argument("VMM range: nothing to reserve"); }
    vmm::check(cuMemAddressReserve(&base_, reserved_, chunk_, 0, 0), "cuMemAddressReserve");
    handles_.assign(reserved_ / chunk_, 0);
}

VmmRange::~VmmRange() {
    for (std::size_t chunk = 0; chunk < handles_.size(); ++chunk) {
        if (handles_[chunk] == 0) { continue; }
        (void)cuMemUnmap(base_ + chunk * chunk_, chunk_);
        (void)cuMemRelease(handles_[chunk]);
    }
    if (base_ != 0) { (void)cuMemAddressFree(base_, reserved_); }
}

bool VmmRange::map(std::size_t chunk) {
    if (chunk >= handles_.size()) { throw std::out_of_range("VMM range: chunk outside the reservation"); }
    if (handles_[chunk] != 0) { return true; }
    const CUmemAllocationProp prop = vmm::device_prop(device_);
    CUmemGenericAllocationHandle handle{};
    const CUresult created = cuMemCreate(&handle, chunk_, &prop, 0);
    if (created == CUDA_ERROR_OUT_OF_MEMORY) { return false; }
    vmm::check(created, "cuMemCreate");
    const CUdeviceptr va = base_ + chunk * chunk_;
    if (const CUresult mapped = cuMemMap(va, chunk_, 0, handle, 0); mapped != CUDA_SUCCESS) {
        (void)cuMemRelease(handle);
        vmm::check(mapped, "cuMemMap");
    }
    try {
        vmm::set_access(va, chunk_, device_);
    } catch (...) {
        (void)cuMemUnmap(va, chunk_);
        (void)cuMemRelease(handle);
        throw;
    }
    handles_[chunk] = handle;
    ++mapped_;
    return true;
}

void VmmRange::unmap(std::size_t chunk) {
    if (chunk >= handles_.size() || handles_[chunk] == 0) { throw std::logic_error("VMM range: chunk not mapped"); }
    vmm::check(cuMemUnmap(base_ + chunk * chunk_, chunk_), "cuMemUnmap");
    vmm::check(cuMemRelease(handles_[chunk]), "cuMemRelease");
    handles_[chunk] = 0;
    --mapped_;
}

void VmmArena::unmap_chunk() {
    if (chunks_.empty()) { throw std::logic_error("VMM arena: no chunk to release"); }
    const CUdeviceptr va = base_ + (chunks_.size() - 1) * chunk_;
    vmm::check(cuMemUnmap(va, chunk_), "cuMemUnmap");
    vmm::check(cuMemRelease(chunks_.back()), "cuMemRelease");
    chunks_.pop_back();
}

} // namespace infernix
