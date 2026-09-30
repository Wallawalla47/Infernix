#include "kv_cache_storage.h"
#include <iostream>
#include <optional>
#include <string_view>

int run_softmax_attention_causal_cache_tests(std::optional<ninfer::KvCacheStorage> storage);
int run_softmax_attention_extended_tests(std::optional<ninfer::KvCacheStorage> storage);
int run_softmax_attention_wide_tests(std::optional<ninfer::KvCacheStorage> storage);
int run_softmax_attention_plain_and_packed_tests();
int run_softmax_attention_context_tests();

int main(int argc, char** argv) {
    bool causal_only = false;
    bool extended    = false;
    bool wide        = false;
    std::optional<ninfer::KvCacheStorage> storage;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--causal-only")
                causal_only = true;
            else if (argument == "--extended")
                extended = true;
            else if (argument == "--wide-only")
                wide = true;
            else if (argument == "--kv-dtype" && i + 1 < argc) {
                const std::string_view name(argv[++i]);
                storage     = name == "all" ? std::nullopt
                                            : std::optional(ninfer::test::parse_kv_cache_storage(name));
                causal_only = true;
            } else
                throw std::invalid_argument("invalid attention test option");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what()
                  << "\nusage: ninfer_softmax_attention_test [--causal-only] "
                     "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|all] [--extended] [--wide-only]\n";
        return 2;
    }
    // --extended runs only the reads beyond the native visible-key ceiling (--rope-yarn-factor).
    if (extended) return run_softmax_attention_extended_tests(storage);
    // --wide-only runs only the ngram verification widths 17-64, single-row and batched.
    if (wide) return run_softmax_attention_wide_tests(storage);
    const int causal = run_softmax_attention_causal_cache_tests(storage);
    if (causal == 77) return 77;
    if (causal_only) return causal;

    const int plain_and_packed = run_softmax_attention_plain_and_packed_tests();
    if (plain_and_packed == 77) return 77;

    const int context = run_softmax_attention_context_tests();
    if (context == 77) return 77;

    const int failures = causal + plain_and_packed + context;
    std::cout << (failures == 0 ? "softmax_attention: PASS\n" : "softmax_attention: FAIL\n");
    return failures == 0 ? 0 : 1;
}
