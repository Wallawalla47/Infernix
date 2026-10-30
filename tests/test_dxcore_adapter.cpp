// The WSL2 adapter facts (core/dxcore_adapter.h) that the VRAM source reads at every offloaded-expert
// startup: under WSL2 with a CUDA device, the NVIDIA adapter opens, by CUDA's LUID and without one,
// reports its shared system memory and answers the usage query. A build whose optimizer turned the DXCore
// calls into __cxa_pure_virtual aborted here (GitHub issue #34). Skips outside WSL2 or without a device.

#include "core/dxcore_adapter.h"

#include <cuda_runtime.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

void check_adapter(const char* how, const unsigned char* luid) {
    const auto adapter = infernix::DxcoreAdapter::open(luid);
    check(adapter != nullptr, how);
    if (adapter == nullptr) { return; }
    std::uint64_t usage = 0;
    check(adapter->shared_system() > 0, "the adapter reports its shared system memory");
    check(adapter->nonlocal_usage(usage), "the adapter answers the non-local usage query");
    std::printf("%s: shared system memory %llu MiB, this process's use %llu MiB\n", how,
                static_cast<unsigned long long>(adapter->shared_system() >> 20),
                static_cast<unsigned long long>(usage >> 20));
}

} // namespace

int main() {
    int devices = 0;
    if (::access("/dev/dxg", F_OK) != 0 || cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: not WSL2 (/dev/dxg) or no CUDA device\n");
        return 77;
    }
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0) == cudaSuccess, "the device's properties");
    check_adapter("the NVIDIA adapter opens without a LUID", nullptr);
    check_adapter("the NVIDIA adapter opens by CUDA's LUID", reinterpret_cast<const unsigned char*>(props.luid));
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    return 0;
}
