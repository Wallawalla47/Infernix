// Fast-path selection (models/qwen3_5/program/planning/fast_paths.h): where each fast path's kernels
// run, which configurations the registry qualifies it for, and how a configuration is described.
// The selection itself (select_fast_paths) needs a loaded model; the engine real tests and startup
// logs cover it.

#include "models/qwen3_5/program/planning/fast_paths.h"

#include <iostream>
#include <string>

namespace {

using infernix::FastPath;
using infernix::KvCacheStorage;
using infernix::models::qwen3_5::ActivationClass;
using infernix::models::qwen3_5::FastPathConfiguration;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    namespace q = infernix::models::qwen3_5;
    int failures = 0;
    const infernix::EngineOptions options;

    // decode-split64: INT8 KV with 24 query heads only.
    const FastPathConfiguration int8_24{24, ActivationClass::A4, KvCacheStorage::Int8Group64};
    failures += check(q::fast_path_applies(FastPath::DecodeSplit64, int8_24, options),
                      "decode-split64 must apply to 24 heads with INT8 KV");
    failures += check(!q::fast_path_applies(FastPath::DecodeSplit64,
                                            {16, ActivationClass::A4, KvCacheStorage::Int8Group64}, options),
                      "decode-split64 must not apply to 16 heads");
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(!q::fast_path_applies(FastPath::DecodeSplit64, {24, ActivationClass::A4, kv}, options),
                          "decode-split64 must apply to INT8 KV only");
    }

    // prompt-pv8: the KV formats whose prompt kernel has an 8-bit P*V form.
    for (const auto kv : {KvCacheStorage::Int8Group64, KvCacheStorage::Nvfp4Group16,
                          KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_applies(FastPath::PromptPv8, {24, ActivationClass::A4, kv}, options),
                          "prompt-pv8 must apply where an 8-bit P*V form exists");
    }
    for (const auto kv : {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4M3Row256}) {
        failures += check(!q::fast_path_applies(FastPath::PromptPv8, {24, ActivationClass::A4, kv}, options),
                          "prompt-pv8 must not apply to BF16 or FP8 KV");
    }
    infernix::EngineOptions original = options;
    original.original_int8_prefill_kernel = true;
    failures += check(!q::fast_path_applies(FastPath::PromptPv8, int8_24, original),
                      "prompt-pv8 needs the fast INT8 prompt kernel");
    original                                = options;
    original.original_nvfp4_prefill_kernel  = true;
    failures += check(!q::fast_path_applies(FastPath::PromptPv8,
                                            {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}, original),
                      "prompt-pv8 needs the fast NVFP4 prompt kernel");

    // Qualification is per configuration: the measured one, not its neighbours.
    failures += check(q::fast_path_qualified(FastPath::DecodeSplit64, int8_24),
                      "decode-split64 is qualified for 24 heads, 4-bit activations, INT8 KV");
    for (const auto activations : {ActivationClass::A8, ActivationClass::A16}) {
        failures += check(!q::fast_path_qualified(FastPath::DecodeSplit64,
                                                  {24, activations, KvCacheStorage::Int8Group64}),
                          "decode-split64 is not qualified at higher activation precision");
    }
    // INT8 KV runs decode-split64 too, and that pair failed the decode gate.
    failures += check(!q::fast_path_qualified(FastPath::PromptPv8, int8_24),
                      "prompt-pv8 is not qualified beside decode-split64 on INT8 KV");
    for (const auto kv : {KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                          KvCacheStorage::Q4KeyVq2Value}) {
        failures += check(q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A4, kv}),
                          "prompt-pv8 is qualified for 24 heads and 4-bit activations on these KV formats");
        failures += check(!q::fast_path_qualified(FastPath::PromptPv8, {24, ActivationClass::A16, kv}),
                          "prompt-pv8 is not qualified at 16-bit activations");
    }
    failures += check(q::describe(int8_24) == "24 heads, 4-bit activations, int8 KV",
                      "configuration description changed");

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " fast-path selection\n";
    return failures == 0 ? 0 : 1;
}
