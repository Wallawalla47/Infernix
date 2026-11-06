// The PDL skew build's launch selection (CMake INFERNIX_PDL_SKEW, tests only; candidates LE-12): which kernels a
// captured launch makes programmatic dependents, so a skew failure can be narrowed to the consumer that reads early.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace infernix::pdl {

bool skew_selected(const void* kernel); // core/pdl.cuh

// INFERNIX_PDL_SKEW_SELECT=k/n: only the kernels whose name's FNV-1a hash is k modulo n (unset: every kernel);
// INFERNIX_PDL_SKEW_LOG=1 prints each kernel's choice, hash and name once. The others launch serialized.
bool skew_selected(const void* kernel) {
    static std::mutex mutex;
    static std::unordered_map<const void*, bool> chosen;
    const std::lock_guard lock(mutex);
    if (const auto it = chosen.find(kernel); it != chosen.end()) { return it->second; }
    const char* name = nullptr;
    if (cudaFuncGetName(&name, kernel) != cudaSuccess || name == nullptr) {
        (void)cudaGetLastError();
        name = "?";
    }
    std::uint64_t hash = 1469598103934665603ULL;
    for (const char* p = name; *p != '\0'; ++p) { hash = (hash ^ static_cast<unsigned char>(*p)) * 1099511628211ULL; }
    bool selected = true;
    if (const char* select = std::getenv("INFERNIX_PDL_SKEW_SELECT"); select != nullptr) {
        char* end             = nullptr;
        const auto bucket     = std::strtoull(select, &end, 10);
        const auto modulus    = end != nullptr && *end == '/' ? std::strtoull(end + 1, nullptr, 10) : 1ULL;
        selected              = modulus == 0 || hash % modulus == bucket;
    }
    if (std::getenv("INFERNIX_PDL_SKEW_LOG") != nullptr) {
        std::fprintf(stderr, "pdl-skew %s %llu %s\n", selected ? "on" : "off", static_cast<unsigned long long>(hash),
                     name);
    }
    chosen.emplace(kernel, selected);
    return selected;
}

} // namespace infernix::pdl
