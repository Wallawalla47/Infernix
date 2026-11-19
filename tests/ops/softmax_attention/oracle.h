#pragma once

#include "infernix/ops/attention_geometry.h"
#include "ops/host_parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infernix::test {

namespace detail {

// out[j] = the FP64 dot product of query head j (queries_t holds [d][head], `stride` heads per d)
// with `key`, each summed in ascending d: N independent sums side by side, none reordered.
template <int N>
void grouped_dots(const double* queries_t, int stride, const double* key, int head_dim, double* out) {
    double sums[N] = {};
    for (int d = 0; d < head_dim; ++d) {
        const double* const q = queries_t + static_cast<std::size_t>(d) * stride;
        const double k        = key[d];
        for (int j = 0; j < N; ++j) sums[j] += q[j] * k;
    }
    for (int j = 0; j < N; ++j) out[j] = sums[j];
}

} // namespace detail

// Independent logical Softmax Attention oracle. Callbacks expose represented public values and
// the entry-specific visible set; no production staging cast, tile, cache address, or reduction
// tree is reproduced here. Every sum keeps its natural order in FP64: a score sums its head_dim
// products in ascending d, and a denominator or numerator sums its keys in ascending key order.
// The work is organized by query and KV head: query heads that share a KV head read each key and
// value row once and advance their independent sums side by side, which reorders no sum, so the
// result is the per-head evaluation's bit for bit. A unit of work is one query and a run of the
// heads sharing a KV head: the whole group, or fewer when few queries would leave workers idle.
// Parallel workers own separate units; value callbacks must be read-only, `visible` must depend
// only on its arguments, and store must write only the addressed output. The default remains
// serial for callers that already parallelize batches.
template <typename QueryValue, typename KeyValue, typename ValueValue, typename Visible,
          typename Store>
void naive_dense_softmax_attention(ops::AttentionHeadGeometry geometry, int query_tokens,
                                   int key_tokens, double scale, QueryValue query_value,
                                   KeyValue key_value, ValueValue value_value, Visible visible,
                                   Store store, int worker_limit = 1) {
    if (!ops::valid_attention_head_geometry(geometry) || query_tokens < 0 || key_tokens < 0 ||
        worker_limit < 1) {
        throw std::invalid_argument("invalid naive Softmax Attention geometry");
    }
    const int group    = geometry.query_heads / geometry.kv_heads;
    const int head_dim = geometry.head_dim;
    const std::int64_t pairs = static_cast<std::int64_t>(query_tokens) * geometry.kv_heads;
    if (pairs == 0) return;
    // Heads per unit: the group, halved while that leaves fewer than four units per worker.
    int chunk = group;
    while (chunk > 1 && pairs * ((group + chunk - 1) / chunk) < 4LL * worker_limit) chunk = (chunk + 1) / 2;
    const int chunks         = (group + chunk - 1) / chunk;
    const std::int64_t units = pairs * chunks;
    constexpr double kMasked = -std::numeric_limits<double>::infinity();
    const auto run_units = [&](std::int64_t first, std::int64_t last) {
        // scores[key * heads + j]: head j's score, then its unnormalized weight, for `key`.
        std::vector<double> scores(static_cast<std::size_t>(key_tokens) * chunk);
        std::vector<double> numerators(static_cast<std::size_t>(chunk) * head_dim);
        std::vector<double> queries_t(static_cast<std::size_t>(head_dim) * chunk);
        std::vector<double> key_row(static_cast<std::size_t>(head_dim));
        std::vector<double> value_row(static_cast<std::size_t>(head_dim));
        std::vector<double> maxima(static_cast<std::size_t>(chunk));
        std::vector<double> denominators(static_cast<std::size_t>(chunk));
        for (auto unit = first; unit < last; ++unit) {
            const std::int64_t pair = unit / chunks;
            const int query         = static_cast<int>(pair / geometry.kv_heads);
            const int kv_head       = static_cast<int>(pair % geometry.kv_heads);
            const int first_head    = kv_head * group + static_cast<int>(unit % chunks) * chunk;
            const int heads         = std::min(chunk, kv_head * group + group - first_head);
            for (int j = 0; j < heads; ++j) {
                for (int d = 0; d < head_dim; ++d) {
                    queries_t[static_cast<std::size_t>(d) * heads + j] = query_value(d, first_head + j, query);
                }
            }
            std::fill(maxima.begin(), maxima.end(), kMasked);
            for (int key = 0; key < key_tokens; ++key) {
                double* const row = scores.data() + static_cast<std::size_t>(key) * heads;
                if (!visible(query, key)) {
                    std::fill(row, row + heads, kMasked);
                    continue;
                }
                for (int d = 0; d < head_dim; ++d) key_row[static_cast<std::size_t>(d)] = key_value(d, kv_head, key);
                int j = 0;
                for (; j + 8 <= heads; j += 8) detail::grouped_dots<8>(queries_t.data() + j, heads, key_row.data(), head_dim, row + j);
                for (; j + 4 <= heads; j += 4) detail::grouped_dots<4>(queries_t.data() + j, heads, key_row.data(), head_dim, row + j);
                for (; j + 2 <= heads; j += 2) detail::grouped_dots<2>(queries_t.data() + j, heads, key_row.data(), head_dim, row + j);
                for (; j < heads; ++j) detail::grouped_dots<1>(queries_t.data() + j, heads, key_row.data(), head_dim, row + j);
                for (j = 0; j < heads; ++j) {
                    const double score                   = row[j] * scale;
                    row[j]                               = score;
                    maxima[static_cast<std::size_t>(j)] = std::max(maxima[static_cast<std::size_t>(j)], score);
                }
            }

            std::fill(denominators.begin(), denominators.end(), 0.0);
            for (int key = 0; key < key_tokens; ++key) {
                double* const row = scores.data() + static_cast<std::size_t>(key) * heads;
                for (int j = 0; j < heads; ++j) {
                    const double maximum = maxima[static_cast<std::size_t>(j)];
                    if (maximum == kMasked || row[j] == kMasked) continue;
                    row[j] = std::exp(row[j] - maximum);
                    denominators[static_cast<std::size_t>(j)] += row[j];
                }
            }
            // Each numerator still sums its keys in ascending order; one value row serves every
            // head of the unit.
            std::fill(numerators.begin(), numerators.end(), 0.0);
            for (int key = 0; key < key_tokens; ++key) {
                const double* const row = scores.data() + static_cast<std::size_t>(key) * heads;
                bool weighted = false;
                for (int j = 0; j < heads; ++j) weighted = weighted || row[j] != kMasked;
                if (!weighted) continue;
                for (int d = 0; d < head_dim; ++d) value_row[static_cast<std::size_t>(d)] = value_value(d, kv_head, key);
                for (int j = 0; j < heads; ++j) {
                    const double weight = row[j];
                    if (weight == kMasked) continue;
                    double* const numerator = numerators.data() + static_cast<std::size_t>(j) * head_dim;
                    for (int d = 0; d < head_dim; ++d) numerator[d] += weight * value_row[static_cast<std::size_t>(d)];
                }
            }
            for (int j = 0; j < heads; ++j) {
                const double denominator      = denominators[static_cast<std::size_t>(j)];
                const double* const numerator = numerators.data() + static_cast<std::size_t>(j) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    store(d, first_head + j, query, denominator > 0.0 ? numerator[d] / denominator : 0.0);
                }
            }
        }
    };
    parallel_ranges(units, worker_limit, run_units);
}

} // namespace infernix::test
