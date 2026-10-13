// Engine option combinations the Engine rejects before it reads any artifact.
#include "infernix/engine.h"

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
std::string construction_error(infernix::EngineOptions options, bool& invalid_argument,
                               const char* artifact = "absent-artifact-for-option-validation.ninfer") {
    options.artifact_path = artifact;
    try {
        infernix::Engine engine(std::move(options));
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
    infernix::EngineOptions options;
    // Both artifact extensions name the same container; any other is rejected before reading.
    for (const char* artifact : {"absent-artifact.infernix", "absent-artifact.ninfer"}) {
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument, artifact);
        failures += check(what.find("accepts only") == std::string::npos,
                          "an .infernix or .ninfer artifact path was rejected by its extension");
    }
    {
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument, "absent-artifact.gguf");
        failures += check(invalid_argument &&
                              what.find("accepts only .infernix and .ninfer") != std::string::npos,
                          "an artifact path with another extension was accepted");
    }
    // Without the opt-outs every KV format passes this check; INT8 and NVFP4 then take the fast
    // kernels.
    for (const auto storage :
         {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
          infernix::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures += check(what.find("INT8 KV") == std::string::npos &&
                              what.find("NVFP4 KV") == std::string::npos,
                          "the default prefill kernel choice was rejected");
    }
    options.original_int8_prefill_kernel = true;
    for (const auto storage : {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Fp8E4M3Row256,
                               infernix::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures +=
            check(invalid_argument && what.find("INT8 KV") != std::string::npos,
                  "the original INT8 prefill kernel was accepted without the INT8 KV cache");
    }
    // With INT8 KV the options pass, so construction proceeds to (and fails on) the artifact.
    options.kv_cache       = infernix::KvCacheStorage::Int8Group64;
    bool invalid_argument  = false;
    const std::string what = construction_error(options, invalid_argument);
    failures += check(what.find("INT8 KV") == std::string::npos,
                      "the original INT8 prefill kernel was rejected with the INT8 KV cache");
    options.original_int8_prefill_kernel  = false;
    options.original_nvfp4_prefill_kernel = true;
    for (const auto storage : {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
                               infernix::KvCacheStorage::Fp8E4M3Row256}) {
        options.kv_cache       = storage;
        bool nvfp4_invalid     = false;
        const std::string text = construction_error(options, nvfp4_invalid);
        failures +=
            check(nvfp4_invalid && text.find("NVFP4 KV") != std::string::npos,
                  "the original NVFP4 prefill kernel was accepted without the NVFP4 KV cache");
    }
    options.kv_cache             = infernix::KvCacheStorage::Nvfp4Group16;
    bool nvfp4_invalid           = false;
    const std::string nvfp4_what = construction_error(options, nvfp4_invalid);
    failures += check(nvfp4_what.find("NVFP4 KV") == std::string::npos,
                      "the original NVFP4 prefill kernel was rejected with the NVFP4 KV cache");
    // The prompt split workspace is bounded and applies to every KV cache.
    options.original_int8_prefill_kernel  = false;
    options.original_nvfp4_prefill_kernel = false;
    options.kv_cache                      = infernix::KvCacheStorage::BFloat16;
    for (const std::uint32_t mib : {0U, infernix::kMaximumPrefillSplitWorkspaceMiB}) {
        options.prefill_split_workspace_mib = mib;
        bool split_invalid                  = false;
        const std::string text              = construction_error(options, split_invalid);
        failures += check(text.find("prefill_split_workspace_mib") == std::string::npos,
                          "a prompt split workspace within [0,16384] MiB was rejected");
    }
    options.prefill_split_workspace_mib = infernix::kMaximumPrefillSplitWorkspaceMiB + 1;
    bool split_invalid                  = false;
    const std::string split_text        = construction_error(options, split_invalid);
    failures += check(split_invalid && split_text.find("prefill_split_workspace_mib") !=
                                           std::string::npos,
                      "a prompt split workspace above 16384 MiB was accepted");
    // Fast paths are on by default and accepted with every KV cache, but the 16-bit-activation
    // quality reference runs the exact route: it refuses fast paths and qualification candidates.
    options.prefill_split_workspace_mib = infernix::kDefaultPrefillSplitWorkspaceMiB;
    failures += check(infernix::EngineOptions{}.fast_paths, "fast paths must default to on");
    options.fast_paths = true;
    for (const auto storage : {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
                               infernix::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool fast_invalid      = false;
        const std::string text = construction_error(options, fast_invalid);
        failures += check(text.find("fast paths") == std::string::npos,
                          "fast paths were rejected without 16-bit activations");
    }
    options.a16_activations          = true;
    bool reference_invalid           = false;
    const std::string reference_text = construction_error(options, reference_invalid);
    failures += check(reference_invalid && reference_text.find("fast paths") != std::string::npos,
                      "fast paths were accepted with the 16-bit-activation reference");
    options.fast_paths           = false;
    options.fast_path_candidates = {infernix::FastPath::DecodeSplit64};
    bool candidate_invalid       = false;
    const std::string candidate_text = construction_error(options, candidate_invalid);
    failures += check(candidate_invalid && candidate_text.find("fast paths") != std::string::npos,
                      "a fast-path candidate was accepted with the 16-bit-activation reference");
    options.fast_path_candidates.clear();
    bool exact_invalid     = false;
    const std::string exact_text = construction_error(options, exact_invalid);
    failures += check(exact_text.find("fast paths") == std::string::npos,
                      "the exact route was rejected with the 16-bit-activation reference");
    // Every fast path round-trips through its name.
    for (const infernix::FastPath path : infernix::kFastPaths) {
        failures += check(infernix::parse_fast_path(infernix::fast_path_name(path)) == path,
                          "a fast path name does not round-trip");
    }
    failures += check(!infernix::parse_fast_path("unknown").has_value(),
                      "an unknown fast path name parsed");
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Engine option validation\n";
    return failures == 0 ? 0 : 1;
}
