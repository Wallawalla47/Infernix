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
    // Without the opt-outs every KV format passes this check; INT8 and NVFP4 then take the fast
    // kernels.
    for (const auto storage :
         {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Int8Group64,
          ninfer::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures += check(what.find("INT8 KV") == std::string::npos &&
                              what.find("NVFP4 KV") == std::string::npos,
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
    options.original_int8_prefill_kernel  = false;
    options.original_nvfp4_prefill_kernel = true;
    for (const auto storage : {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Int8Group64,
                               ninfer::KvCacheStorage::Fp8E4M3Row256}) {
        options.kv_cache       = storage;
        bool nvfp4_invalid     = false;
        const std::string text = construction_error(options, nvfp4_invalid);
        failures +=
            check(nvfp4_invalid && text.find("NVFP4 KV") != std::string::npos,
                  "the original NVFP4 prefill kernel was accepted without the NVFP4 KV cache");
    }
    options.kv_cache             = ninfer::KvCacheStorage::Nvfp4Group16;
    bool nvfp4_invalid           = false;
    const std::string nvfp4_what = construction_error(options, nvfp4_invalid);
    failures += check(nvfp4_what.find("NVFP4 KV") == std::string::npos,
                      "the original NVFP4 prefill kernel was rejected with the NVFP4 KV cache");
    // 8-bit prefill P*V is chosen per KV format by default and every KV cache and prompt kernel
    // accepts either form.
    options.original_nvfp4_prefill_kernel = false;
    failures += check(ninfer::EngineOptions{}.prefill_8bit_pv == ninfer::PrefillPv8::Auto,
                      "8-bit prefill P*V must default to auto");
    for (const auto pv8 : {ninfer::PrefillPv8::Auto, ninfer::PrefillPv8::On, ninfer::PrefillPv8::Off}) {
        options.prefill_8bit_pv = pv8;
        for (const auto storage :
             {ninfer::KvCacheStorage::BFloat16, ninfer::KvCacheStorage::Int8Group64,
              ninfer::KvCacheStorage::Fp8E4M3Row256, ninfer::KvCacheStorage::Nvfp4Group16,
              ninfer::KvCacheStorage::Fp8KeyNvfp4Value}) {
            for (const bool original : {false, true}) {
                options.kv_cache = storage;
                options.original_int8_prefill_kernel =
                    original && storage == ninfer::KvCacheStorage::Int8Group64;
                options.original_nvfp4_prefill_kernel =
                    original && storage == ninfer::KvCacheStorage::Nvfp4Group16;
                bool pv8_invalid       = false;
                const std::string text = construction_error(options, pv8_invalid);
                failures += check(text.find("P*V") == std::string::npos,
                                  "the 8-bit prefill P*V setting was rejected");
            }
        }
    }
    // The prompt split workspace is bounded and applies to every KV cache.
    options.original_int8_prefill_kernel  = false;
    options.original_nvfp4_prefill_kernel = false;
    options.prefill_8bit_pv               = ninfer::PrefillPv8::Off;
    options.kv_cache                      = ninfer::KvCacheStorage::BFloat16;
    for (const std::uint32_t mib : {0U, ninfer::kMaximumPrefillSplitWorkspaceMiB}) {
        options.prefill_split_workspace_mib = mib;
        bool split_invalid                  = false;
        const std::string text              = construction_error(options, split_invalid);
        failures += check(text.find("prefill_split_workspace_mib") == std::string::npos,
                          "a prompt split workspace within [0,16384] MiB was rejected");
    }
    options.prefill_split_workspace_mib = ninfer::kMaximumPrefillSplitWorkspaceMiB + 1;
    bool split_invalid                  = false;
    const std::string split_text        = construction_error(options, split_invalid);
    failures += check(split_invalid && split_text.find("prefill_split_workspace_mib") !=
                                           std::string::npos,
                      "a prompt split workspace above 16384 MiB was accepted");
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Engine option validation\n";
    return failures == 0 ? 0 : 1;
}
