// causal_softmax_attention_reference against an FP64 oracle (the reference route's attention,
// docs/maintainer/model-quality.md 6.2): BF16 Q and K, V stored as FP16 from BF16, every column's causal
// keys from the paged cache (an earlier call's prefix included), output rounded once to BF16. Its FP32
// scores and sums leave it within one BF16 ulp of the oracle, nearly always correctly rounded. A
// column's bits depend only on its own query and keys: identical from run to run and whatever call
// width carries it.
#include "core/paged_kv_cache.h"
#include "infernix/ops/softmax_attention.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr float kScale          = 0.0625F;

float fp16_round(float x) { return __half2float(__float2half_rn(x)); }

double bf16_ulp(double x) {
    const double a = std::max(std::abs(x), 1.0e-30);
    return std::ldexp(1.0, std::ilogb(a) - 7);
}

struct Case {
    std::int32_t query_heads;
    std::int32_t kv_heads;
    std::int32_t prefix;  // tokens appended by a first call
    std::int32_t tokens;  // tokens of the call under test
};

int run_case(const Case& c, std::uint32_t seed) {
    const std::int32_t context  = c.prefix + c.tokens;
    const std::int32_t pages    = (context + kPagedKVPageSize - 1) / kPagedKVPageSize;
    const std::int32_t physical = pages + 3;
    // A shuffled block table, so the kernel follows the paging.
    std::vector<int> table(pages);
    std::vector<int> order(physical);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(seed));
    std::copy(order.begin(), order.begin() + pages, table.begin());

    const std::size_t plane = static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize * c.kv_heads * physical;
    DeviceBuffer k_pages(plane * 2), v_pages(plane * 2);
    cuda_check(cudaMemset(k_pages.p, 0xff, plane * 2), "memset");
    cuda_check(cudaMemset(v_pages.p, 0xff, plane * 2), "memset");
    DeviceBuffer d_table = to_device_i32(table);
    DeviceBuffer d_row   = to_device_i32(std::vector<int>{0});
    PagedKVBatchLayerView cache{
        .k_pages      = Tensor(k_pages.p, DType::BF16, {kHeadDim, kPagedKVPageSize, c.kv_heads, physical}),
        .v_pages      = Tensor(v_pages.p, DType::FP16, {kHeadDim, kPagedKVPageSize, c.kv_heads, physical}),
        .block_tables = Tensor(d_table.p, DType::I32, {pages, 1}),
        .head_dim     = kHeadDim,
        .num_kv_heads = c.kv_heads,
        .storage      = KvCacheStorage::BFloat16,
    };
    const AttentionHeadGeometry geometry{kHeadDim, c.query_heads, c.kv_heads};

    std::vector<float> q(static_cast<std::size_t>(kHeadDim) * c.query_heads * context);
    std::vector<float> k(static_cast<std::size_t>(kHeadDim) * c.kv_heads * context);
    std::vector<float> v(k.size());
    fill_uniform(q, seed + 1, -2.0F, 2.0F);
    fill_uniform(k, seed + 2, -2.0F, 2.0F);
    fill_uniform(v, seed + 3, -1.0F, 1.0F);
    round_to_bf16(q);
    round_to_bf16(k);
    round_to_bf16(v);

    // One call over positions [begin, begin + count), writing out (BF16 [256, Hq, count]).
    const auto call = [&](std::int32_t begin, std::int32_t count, DeviceBuffer& out) {
        const std::size_t qo = static_cast<std::size_t>(kHeadDim) * c.query_heads;
        const std::size_t kv = static_cast<std::size_t>(kHeadDim) * c.kv_heads;
        DeviceBuffer dq = to_device_bf16(std::vector<float>(q.begin() + begin * qo, q.begin() + (begin + count) * qo));
        DeviceBuffer dk = to_device_bf16(std::vector<float>(k.begin() + begin * kv, k.begin() + (begin + count) * kv));
        DeviceBuffer dv = to_device_bf16(std::vector<float>(v.begin() + begin * kv, v.begin() + (begin + count) * kv));
        std::vector<int> positions(count);
        std::iota(positions.begin(), positions.end(), begin);
        DeviceBuffer dp = to_device_i32(positions);
        Tensor tq(dq.p, DType::BF16, {kHeadDim, c.query_heads, count});
        Tensor tk(dk.p, DType::BF16, {kHeadDim, c.kv_heads, count});
        Tensor tv(dv.p, DType::BF16, {kHeadDim, c.kv_heads, count});
        Tensor tp(dp.p, DType::I32, {count});
        Tensor rows(d_row.p, DType::I32, {1});
        Tensor to(out.p, DType::BF16, {kHeadDim, c.query_heads, count});
        ops::causal_softmax_attention_reference(tq, tk, tv, tp, rows, geometry, kScale, cache, to, nullptr);
        cuda_synchronize();
    };
    const std::size_t out_elements = static_cast<std::size_t>(kHeadDim) * c.query_heads * c.tokens;
    if (c.prefix > 0) {
        DeviceBuffer prefix_out(static_cast<std::size_t>(kHeadDim) * c.query_heads * c.prefix * 2);
        call(0, c.prefix, prefix_out);
    }
    DeviceBuffer out(out_elements * 2), again(out_elements * 2), split(out_elements * 2);
    call(c.prefix, c.tokens, out);
    call(c.prefix, c.tokens, again); // the same keys appended again: unchanged cache
    // The same columns in three calls of different widths.
    const std::int32_t first = c.tokens / 3, second = c.tokens / 2;
    {
        DeviceBuffer a(static_cast<std::size_t>(kHeadDim) * c.query_heads * first * 2);
        DeviceBuffer b(static_cast<std::size_t>(kHeadDim) * c.query_heads * second * 2);
        DeviceBuffer r(static_cast<std::size_t>(kHeadDim) * c.query_heads * (c.tokens - first - second) * 2);
        call(c.prefix, first, a);
        call(c.prefix + first, second, b);
        call(c.prefix + first + second, c.tokens - first - second, r);
        const std::size_t col = static_cast<std::size_t>(kHeadDim) * c.query_heads * 2;
        cuda_check(cudaMemcpy(split.p, a.p, first * col, cudaMemcpyDeviceToDevice), "copy");
        cuda_check(cudaMemcpy(static_cast<char*>(split.p) + first * col, b.p, second * col,
                              cudaMemcpyDeviceToDevice), "copy");
        cuda_check(cudaMemcpy(static_cast<char*>(split.p) + (first + second) * col, r.p,
                              (c.tokens - first - second) * col, cudaMemcpyDeviceToDevice), "copy");
    }
    const auto got       = from_device<std::uint16_t>(out, out_elements);
    const auto got_again = from_device<std::uint16_t>(again, out_elements);
    const auto got_split = from_device<std::uint16_t>(split, out_elements);
    int failures = 0;
    if (got != got_again) {
        std::cerr << "reference attention: a repeated call changed bits\n";
        ++failures;
    }
    if (got != got_split) {
        std::cerr << "reference attention: the call width changed bits\n";
        ++failures;
    }

    // FP64 oracle.
    const std::int32_t group = c.query_heads / c.kv_heads;
    std::int64_t outputs = 0, rounded = 0;
    double worst_ulps = 0.0;
    std::vector<double> scores(context);
    for (std::int32_t t = 0; t < c.tokens; ++t) {
        const std::int32_t position = c.prefix + t;
        for (std::int32_t h = 0; h < c.query_heads; ++h) {
            const std::int32_t kvh = h / group;
            const float* qv        = &q[(static_cast<std::size_t>(position) * c.query_heads + h) * kHeadDim];
            double best            = -INFINITY;
            for (std::int32_t j = 0; j <= position; ++j) {
                const float* kv = &k[(static_cast<std::size_t>(j) * c.kv_heads + kvh) * kHeadDim];
                double dot      = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) { dot += static_cast<double>(qv[d]) * kv[d]; }
                scores[j] = dot * kScale;
                best      = std::max(best, scores[j]);
            }
            double sum = 0.0;
            for (std::int32_t j = 0; j <= position; ++j) {
                scores[j] = std::exp(scores[j] - best);
                sum += scores[j];
            }
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double value = 0.0;
                for (std::int32_t j = 0; j <= position; ++j) {
                    value += scores[j] * fp16_round(v[(static_cast<std::size_t>(j) * c.kv_heads + kvh) * kHeadDim + d]);
                }
                value /= sum;
                const double actual =
                    bf16_to_f32(got[(static_cast<std::size_t>(t) * c.query_heads + h) * kHeadDim + d]);
                const double ulps = std::abs(actual - value) / bf16_ulp(value);
                worst_ulps        = std::max(worst_ulps, ulps);
                rounded += ulps <= 0.5 ? 1 : 0;
                ++outputs;
            }
        }
    }
    const double rounded_share = static_cast<double>(rounded) / static_cast<double>(outputs);
    std::printf("reference attention Hq=%d Hkv=%d prefix=%d T=%d: %lld outputs, %.4f%% correctly rounded, worst "
                "%.3f BF16 ulps\n",
                c.query_heads, c.kv_heads, c.prefix, c.tokens, static_cast<long long>(outputs), 100.0 * rounded_share,
                worst_ulps);
    // FP32 scores and sums: within one ulp everywhere, correctly rounded except near ties.
    if (worst_ulps > 1.0 || rounded_share < 0.99) {
        std::cerr << "reference attention: outside one BF16 ulp of the FP64 oracle or under 99% correctly rounded\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    int failures = 0;
    failures += run_case({24, 4, 100, 300}, 11);
    failures += run_case({16, 2, 0, 257}, 23);
    failures += run_case({24, 4, 64, 33}, 37);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "OK causal_softmax_attention_reference\n";
    return 0;
}
