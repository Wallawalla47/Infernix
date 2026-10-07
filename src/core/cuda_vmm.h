#pragma once

// Internal helpers over the CUDA driver's virtual memory management (cuMemCreate / cuMemMap):
// shared by EvictableWeightPool and VmmArena. Device-resident pinned allocations with
// read-write access for one device.

#include <cuda.h>

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>

namespace infernix::vmm {

inline void check(CUresult result, const char* expr) {
    if (result == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(result, &name);
    throw std::runtime_error(std::string(expr) + " failed: " + (name != nullptr ? name : "unknown CUresult"));
}

inline void ensure_driver_initialized() {
    static std::once_flag once;
    std::call_once(once, [] { check(cuInit(0), "cuInit(0)"); });
}

inline bool supported(int device) {
    ensure_driver_initialized();
    CUdevice handle = 0;
    if (cuDeviceGet(&handle, device) != CUDA_SUCCESS) { return false; }
    int value = 0;
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, handle) !=
        CUDA_SUCCESS) {
        return false;
    }
    return value != 0;
}

inline CUmemAllocationProp device_prop(int device) {
    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device;
    return prop;
}

inline std::size_t granularity(int device) {
    ensure_driver_initialized();
    const CUmemAllocationProp prop = device_prop(device);
    std::size_t bytes              = 0;
    check(cuMemGetAllocationGranularity(&bytes, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
          "cuMemGetAllocationGranularity");
    return bytes;
}

inline void set_access(CUdeviceptr va, std::size_t bytes, int device) {
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = device;
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check(cuMemSetAccess(va, bytes, &access, 1), "cuMemSetAccess");
}

} // namespace infernix::vmm
