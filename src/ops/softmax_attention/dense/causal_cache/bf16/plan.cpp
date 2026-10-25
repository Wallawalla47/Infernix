#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/operands.h"
#include "ops/softmax_attention/common/causal_partition.h"
#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;
} // namespace

Bf16KvCausalPlan make_bf16_kv_causal_plan(int heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope,
                                          int multiprocessor_count) {
    if (multiprocessor_count <= 0 || !causal_registered_heads(heads) || width < 1 || batch < 1 ||
        batch > 8 || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("BF16 attention: invalid plan inputs");
    const int kv_heads = causal_kv_heads(heads), group = heads / kv_heads;
    const auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    const int visible   = envelope.max_visible_keys;
    if (width > kGroupedPrefillMaxWidth) {
        // The two-KV-head geometries (16/2 measured; 32/2 by the same KV-traffic rule, swept in M3).
        const auto instance =
            kv_heads == 2 && width >= 1024 && width % 128 == 0 && visible - width >= 8192
                ? Bf16KvInstance::Tiled128
                : Bf16KvInstance::Tiled64;
        Bf16KvCausalPlan plan{instance, {}, heads, width, batch, envelope};
        if (envelope.bf16_prompt_split && batch == 1) {
            plan.prompt_splits = bf16_kv_prompt_splits(
                heads, width, bf16_kv_instance_description(instance).query_rows, envelope.max_visible_keys,
                envelope.prompt_split_workspace_bytes, multiprocessor_count);
        }
        return plan;
    }
    const int rows                  = width * group;
    const auto instance             = width == 1   ? Bf16KvInstance::GroupedDecode
                                      : rows <= 32 ? Bf16KvInstance::Grouped32
                                                   : Bf16KvInstance::Grouped64;
    const auto description          = bf16_kv_instance_description(instance);
    const int tiles                 = ceil_div(rows, description.query_rows);
    const int independent_tiles     = (envelope.row_invariant_partition ? 1 : batch) * kv_heads * tiles;
    const bool many_memory_tiles    = description.query_rows == 32 && independent_tiles >= 16;
    const bool multiple_query_tiles = tiles > 1;
    const std::int64_t sms          = multiprocessor_count;
    // Small queries permit up to six CTAs per SM, with a half-SM-count split budget per
    // independent query tile. Wider prefill targets two CTAs per SM to limit partial traffic.
    const auto long_ctas = width <= 128 / group && (many_memory_tiles || multiple_query_tiles)
                               ? std::clamp((sms / 2) * independent_tiles, 2 * sms, 6 * sms)
                               : 2 * sms;
    Bf16KvPartition partition{1, causal_partition_target(2 * sms, independent_tiles),
                              causal_partition_target(long_ctas, independent_tiles),
                              description.key_rows};
    // The envelope bounds the largest live row. Other batch rows may be shorter.
    const int low      = batch == 1 ? static_cast<int>(envelope.min_visible_keys) : 1;
    partition.capacity = std::max(
        partition.interval_capacity(low, std::min(visible, 32767), partition.normal_target),
        partition.interval_capacity(std::max(low, 32768), visible, partition.long_target));
    return {instance, partition, heads, width, batch, envelope};
}

int bf16_kv_prompt_splits(int query_heads, int width, int query_rows, std::uint32_t visible_keys,
                          std::size_t workspace_bytes, int multiprocessor_count) {
    constexpr int kMaximumSplits    = 4;
    constexpr std::uint32_t kMinKeys = 1024;
    constexpr double kGain          = 1.03;
    const std::int64_t tiles = static_cast<std::int64_t>((width + query_rows - 1) / query_rows) * query_heads;
    const std::int64_t sms   = multiprocessor_count;
    const auto fill          = [&](std::int64_t ctas) {
        const std::int64_t waves = (ctas + sms - 1) / sms;
        return static_cast<double>(ctas) / static_cast<double>(waves * sms);
    };
    int best          = 1;
    double best_fill  = fill(tiles);
    for (int splits = 2; splits <= kMaximumSplits; ++splits) {
        if (visible_keys / static_cast<std::uint32_t>(splits) < kMinKeys) { break; }
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, query_heads, width, splits, 1);
        if (layout.peak_bytes(1) > workspace_bytes) { break; }
        if (const double f = fill(tiles * splits); f > best_fill * kGain) {
            best      = splits;
            best_fill = f;
        }
    }
    return best;
}

std::size_t bf16_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                    CausalAttentionExecutionEnvelope envelope,
                                    int multiprocessor_count) {
    std::size_t maximum = 0;
    // bf16-prompt-split: the split partials of the widest-reaching prompt-route launch.
    if (envelope.bf16_prompt_split && batch == 1) {
        for (int width = std::max(min_width, kGroupedPrefillMaxWidth + 1); width <= max_width; ++width) {
            const auto plan = make_bf16_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
            if (plan.prompt_splits < 2) { continue; }
            WorkspaceLayoutBuilder layout;
            (void)allocate_causal_partials(layout, heads, width, plan.prompt_splits, 1);
            maximum = std::max(maximum, layout.peak_bytes(1));
        }
    }
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan =
            make_bf16_kv_causal_plan(heads, width, batch, envelope, multiprocessor_count);
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, plan.partition.capacity, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace infernix::ops::detail
