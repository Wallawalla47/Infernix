#include "ops/op_family.h"
#include "kv_cache_storage.h"
#include <charconv>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

void select_softmax_attention_cases(std::optional<std::string> geometry, std::int32_t min_columns,
                                    std::int32_t max_columns);
int run_softmax_attention_causal_cache_tests(std::optional<infernix::KvCacheStorage> storage);
int run_softmax_attention_extended_tests(std::optional<infernix::KvCacheStorage> storage);
int run_softmax_attention_wide_tests(std::optional<infernix::KvCacheStorage> storage);
int run_softmax_attention_column_invariant_tests(std::optional<infernix::KvCacheStorage> storage);
int run_softmax_attention_plain_and_packed_tests();
int run_softmax_attention_context_tests();

INFERNIX_OP_TEST(softmax_attention) {
    bool causal_only     = false;
    bool non_causal_only = false;
    bool extended        = false;
    bool wide            = false;
    bool column_rounds   = false;
    std::optional<infernix::KvCacheStorage> storage;
    std::optional<std::string> geometry;
    std::int32_t min_columns = 1, max_columns = INT32_MAX;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            if (argument == "--causal-only")
                causal_only = true;
            else if (argument == "--non-causal-only")
                non_causal_only = true;
            else if (argument == "--extended")
                extended = true;
            else if (argument == "--wide-only")
                wide = true;
            else if (argument == "--column-invariant-only")
                column_rounds = true;
            else if (argument == "--kv-dtype" && i + 1 < argc) {
                const std::string_view name(argv[++i]);
                storage     = name == "all" ? std::nullopt
                                            : std::optional(infernix::test::parse_kv_cache_storage(name));
                causal_only = true;
                if (storage == infernix::KvCacheStorage::Vq2 ||
                    storage == infernix::KvCacheStorage::Q4KeyVq2Value)
                    throw std::invalid_argument(
                        "vq2 and k4v2 attention is qualified by infernix_vq_attention_test");
            } else if (argument == "--geometry" && i + 1 < argc) {
                geometry = argv[++i];
            } else if (argument == "--columns" && i + 1 < argc) {
                // LO-HI: only calls of LO to HI columns per row.
                const std::string_view range(argv[++i]);
                const auto dash = range.find('-');
                const auto parse = [](std::string_view text, std::int32_t& out) {
                    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
                    return error == std::errc{} && end == text.data() + text.size();
                };
                if (dash == std::string_view::npos || !parse(range.substr(0, dash), min_columns) ||
                    !parse(range.substr(dash + 1), max_columns))
                    throw std::invalid_argument("--columns takes LO-HI");
            } else
                throw std::invalid_argument("invalid attention test option");
        }
        if (non_causal_only && (causal_only || extended || wide || column_rounds))
            throw std::invalid_argument(
                "--non-causal-only excludes --causal-only, --kv-dtype, --extended and --wide-only");
        if ((geometry || min_columns != 1 || max_columns != INT32_MAX) && non_causal_only)
            throw std::invalid_argument("--geometry and --columns select causal cases");
        select_softmax_attention_cases(geometry, min_columns, max_columns);
    } catch (const std::exception& error) {
        std::cerr << error.what()
                  << "\nusage: infernix_softmax_attention_test [--causal-only | --non-causal-only] "
                     "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4|all] [--extended] [--wide-only] "
                     "[--column-invariant-only] [--geometry d256-h24-kv4|d256-h16-kv2|d256-h32-kv2] "
                     "[--columns LO-HI]\n";
        return 2;
    }
    // --extended runs only the reads beyond the native visible-key ceiling (--rope-yarn-factor).
    if (extended) return run_softmax_attention_extended_tests(storage);
    // --wide-only runs only the ngram verification widths 17-64, single-row and batched.
    if (wide) return run_softmax_attention_wide_tests(storage);
    // --column-invariant-only runs only the column-invariant verification rounds.
    if (column_rounds) return run_softmax_attention_column_invariant_tests(storage);
    // The causal cases run per KV format (CTest registers one entry per format, so they can run
    // concurrently); --non-causal-only runs the plain, packed and context sections without them.
    int causal = 0;
    if (!non_causal_only) {
        causal = run_softmax_attention_causal_cache_tests(storage);
        if (causal == 77) return 77;
        if (causal_only) return causal;
    }

    const int plain_and_packed = run_softmax_attention_plain_and_packed_tests();
    if (plain_and_packed == 77) return 77;

    const int context = run_softmax_attention_context_tests();
    if (context == 77) return 77;

    const int failures = causal + plain_and_packed + context;
    std::cout << (failures == 0 ? "softmax_attention: PASS\n" : "softmax_attention: FAIL\n");
    return failures == 0 ? 0 : 1;
}
