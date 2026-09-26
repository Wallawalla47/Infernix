// Engine option combinations the Engine rejects before it reads any artifact.
#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// The exception an Engine over a nonexistent artifact raises for these options.
std::string construction_error(ninfer::EngineOptions options, bool& invalid_argument) {
    options.artifact_path = "absent-artifact-for-option-validation.ninfer";
    try {
        ninfer::Engine engine(std::move(options));
    } catch (const std::invalid_argument& error) {
        invalid_argument = true;
        return error.what();
    } catch (const std::exception& error) {
        invalid_argument = false;
        return error.what();
    }
    invalid_argument = false;
    return {};
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    ninfer::EngineOptions options;
    // Without the opt-out every KV format passes this check; INT8 then takes the fast kernel.
    for (const auto storage :
         {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Int8Group64,
          ninfer::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures += check(what.find("INT8 KV") == std::string::npos,
                          "the default prefill kernel choice was rejected");
    }
    options.original_int8_prefill_kernel = true;
    for (const auto storage : {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Fp8E4M3Row256,
                               ninfer::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures +=
            check(invalid_argument && what.find("INT8 KV") != std::string::npos,
                  "the original INT8 prefill kernel was accepted without the INT8 KV cache");
    }
    // With INT8 KV the options pass, so construction proceeds to (and fails on) the artifact.
    options.kv_cache       = ninfer::KvCacheStorage::Int8Group64;
    bool invalid_argument  = false;
    const std::string what = construction_error(options, invalid_argument);
    failures += check(what.find("INT8 KV") == std::string::npos,
                      "the original INT8 prefill kernel was rejected with the INT8 KV cache");
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Engine option validation\n";
    return failures == 0 ? 0 : 1;
}
