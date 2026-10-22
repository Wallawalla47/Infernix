// fp8_e4m3fn_row_f32 (FP32 row multipliers, one static activation scale s per parent): the
// Qwen3.5-122B-A10B projections (docs/maintainer/qwen3_5-122b-offload-design.md §6.2, §8.2).
//
//   - The static A8 encoding (D4) bit for bit: code = e4m3_rn_satfinite(fl32(x * fl32(1 / s))) with
//     the sign of x, every token's scale s, on columns with saturating outliers, E4M3 subnormals and
//     negative zeros, at K = 3072 and 8192.
//   - attn_input_proj [17408, 3072], gdn_input_proj [20480, 3072] and linear_add [3072, 8192] under
//     A16Only, AllowA8 and A16ColumnInvariant against naive FP64 sums of the exactly decoded weights
//     (code * multiplier) over sampled rows. The A8 oracle takes the activations the D4 rule
//     represents (code * s), so A8 is held to the A16 criterion: what remains is FP32 accumulation and
//     the output's BF16 rounding. Widths cover every schedule band and the split-K tails.
//   - Column invariance: under A16ColumnInvariant a column's outputs are the same bits at every width.

#include "core/weight.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear_add.h"
#include "ops/common/canonical_math.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include "core/device.h"
#include "ops/fp8_static_a8_reference.h"
#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::input_projection;

namespace {

// The FP8 A16 projections' criterion (test_attn_input_proj): the output's BF16 rounding.
constexpr ReductionCriterion kCriterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr int kSamples = 31;
using fp8_static::activations;
using fp8_static::kScale;
using fp8_static::represented;

// FP64 sums of sampled rows [offset, offset + rows) of the parent against `x` ([k][tokens]).
std::vector<double> oracle(const quantized_weight::PackedWeight& weight, int offset, int rows,
                           const std::vector<double>& x, int k, int tokens) {
    const std::vector<std::int32_t> selected = sampled_rows(rows, kSamples);
    std::vector<double> out(selected.size() * static_cast<std::size_t>(tokens));
    std::vector<double> decoded(static_cast<std::size_t>(k));
    for (std::size_t r = 0; r < selected.size(); ++r) {
        for (int c = 0; c < k; ++c) {
            decoded[static_cast<std::size_t>(c)] = quantized_weight::logical_weight_fp64(weight, offset + selected[r], c);
        }
        for (int t = 0; t < tokens; ++t) {
            double sum = 0.0;
            for (int c = 0; c < k; ++c) {
                sum += decoded[static_cast<std::size_t>(c)] * x[static_cast<std::size_t>(t) * k + c];
            }
            out[r * static_cast<std::size_t>(tokens) + static_cast<std::size_t>(t)] = sum;
        }
    }
    return out;
}

std::vector<double> as_double(const std::vector<float>& x) { return {x.begin(), x.end()}; }

bool a8(ops::LinearPolicy policy, int tokens, int first_a8 = 17) {
    return policy == ops::LinearPolicy::AllowA8 && tokens >= first_a8;
}

std::string name(ops::LinearPolicy policy) {
    return policy == ops::LinearPolicy::AllowA8           ? "allow-a8"
           : policy == ops::LinearPolicy::A16ColumnInvariant ? "column-invariant"
                                                              : "a16";
}

// --------------------------------------------------------------------------- the D4 encoding

int run_quantizer(int k, int tokens) {
    const auto x    = activations(k, tokens, 11U + static_cast<std::uint32_t>(k + tokens));
    const auto bits = bf16_bits(x);
    DeviceBuffer input = to_device(bits);
    const std::size_t bytes = ops::detail::fp8_a8_workspace_capacity_bytes(tokens, k);
    GuardedDeviceBuffer storage(bytes);
    DeviceArena arena(DeviceSpan{storage.data(), bytes});
    const auto workspace = ops::detail::allocate_fp8_a8_workspace(arena, tokens, k);
    Weight weight{};
    weight.qtype       = QType::FP8_E4M3FN_ROW_F32;
    weight.n           = 16;
    weight.k           = k;
    weight.input_scale = kScale;
    DeviceContext device;
    ops::detail::launch_fp8_a8_quantize(Tensor(input.p, DType::BF16, {k, tokens}), weight, workspace, device.stream);
    cuda_synchronize(device.stream);
    const auto codes  = from_device<std::uint8_t>(workspace.codes, static_cast<std::size_t>(k) * tokens);
    const auto scales = from_device<float>(workspace.scales, static_cast<std::size_t>(tokens));
    const float r     = 1.0F / kScale;
    std::size_t bad = 0, saturated = 0, subnormal = 0;
    for (std::size_t i = 0; i < codes.size(); ++i) {
        const std::uint8_t want = fp8_static::code(x[i], r);
        bad += codes[i] != want;
        saturated += (want & 0x7F) == 0x7E ? 1 : 0;
        subnormal += (want & 0x78) == 0 && (want & 7) != 0 ? 1 : 0;
    }
    std::size_t scale_bad = 0;
    for (const float s : scales) { scale_bad += s != kScale; }
    const std::string where = "static A8 encoding K=" + std::to_string(k) + " T=" + std::to_string(tokens);
    if (bad != 0 || scale_bad != 0 || saturated == 0 || subnormal == 0) {
        std::cerr << where << ": " << bad << " codes and " << scale_bad << " scales differ (" << saturated
                  << " saturated, " << subnormal << " subnormal)\n";
        return 1;
    }
    return storage.verify_guards(where);
}

// ------------------------------------------------------------------------------ projections

int run_attention(DevicePackedWeight& parent, int tokens, ops::LinearPolicy policy,
                  std::vector<std::uint16_t>* first_column = nullptr) {
    constexpr int kHidden = 3072, kQ = 8192, kKv = 512, kRows = 17408;
    const auto x       = activations(kHidden, tokens, 101U);
    const auto x_bits  = bf16_bits(x);
    DeviceBuffer input = to_device(x_bits);
    GuardedBf16Tensor query(kQ, tokens), gate(kQ, tokens), key(kKv, tokens), value(kKv, tokens);
    Tensor tx(input.p, DType::BF16, {kHidden, tokens}), q = query.tensor(), g = gate.tensor(), k = key.tensor(),
                                                         v = value.tensor();
    const auto capacity = ops::attn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_F32, kRows, kHidden,
                                                                        policy, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    DeviceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 1)});
    DeviceContext device;
    ops::attn_input_proj(tx, parent.view(), q, g, k, v, policy, workspace, device.stream);
    cuda_synchronize(device.stream);
    const auto reference = a8(policy, tokens) ? represented(x) : as_double(x);
    const std::string tag = " " + name(policy) + " T=" + std::to_string(tokens);
    int failures = 0;
    const struct {
        const char* label;
        GuardedBf16Tensor* out;
        int offset, rows;
    } segments[] = {{"query", &query, 0, kQ}, {"key", &key, kQ, kKv}, {"gate", &gate, kQ + kKv, kQ},
                    {"value", &value, 2 * kQ + kKv, kKv}};
    for (const auto& s : segments) {
        const std::string label = std::string("attn ") + s.label + tag;
        failures += s.out->verify_guards(label) + s.out->verify_fully_written(label);
        failures += compare(label, gather_rows(s.out->values(), s.rows, 0, s.rows, tokens, kSamples),
                            oracle(parent.host, s.offset, s.rows, reference, kHidden, tokens), kCriterion);
    }
    if (first_column != nullptr) {
        // The query rows of column 0 (the first call's are the reference).
        const auto bits = query.bits();
        const std::vector<std::uint16_t> column(bits.begin(), bits.begin() + kQ);
        if (first_column->empty()) {
            *first_column = column;
        } else if (column != *first_column) {
            std::cerr << "attn" << tag << ": column 0 changed with the call's width\n";
            ++failures;
        }
    }
    return failures + scratch.verify_guards("attn" + tag) + parent.verify_preserved("attn parent");
}

int run_gdn(DevicePackedWeight& parent, int tokens, ops::LinearPolicy policy,
            std::vector<std::uint16_t>* first_column = nullptr) {
    constexpr int kHidden = 3072, kQkv = 12288, kZ = 8192, kRows = 20480;
    const auto x       = activations(kHidden, tokens, 201U);
    const auto x_bits  = bf16_bits(x);
    DeviceBuffer input = to_device(x_bits);
    GuardedBf16Tensor qkv(kQkv, tokens), z(kZ, tokens);
    Tensor tx(input.p, DType::BF16, {kHidden, tokens}), tqkv = qkv.tensor(), tz = z.tensor();
    const auto capacity = ops::gdn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_F32, kRows, kHidden,
                                                                       policy, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    DeviceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 1)});
    DeviceContext device;
    ops::gdn_input_proj(tx, parent.view(), tqkv, tz, policy, workspace, device.stream);
    cuda_synchronize(device.stream);
    const auto reference = a8(policy, tokens) ? represented(x) : as_double(x);
    const std::string tag = " " + name(policy) + " T=" + std::to_string(tokens);
    int failures = qkv.verify_guards("gdn qkv" + tag) + z.verify_guards("gdn z" + tag);
    failures += qkv.verify_fully_written("gdn qkv" + tag) + z.verify_fully_written("gdn z" + tag);
    failures += compare("gdn qkv" + tag, gather_rows(qkv.values(), kQkv, 0, kQkv, tokens, kSamples),
                        oracle(parent.host, 0, kQkv, reference, kHidden, tokens), kCriterion);
    failures += compare("gdn z" + tag, gather_rows(z.values(), kZ, 0, kZ, tokens, kSamples),
                        oracle(parent.host, kQkv, kZ, reference, kHidden, tokens), kCriterion);
    if (first_column != nullptr) {
        const auto bits = qkv.bits();
        const std::vector<std::uint16_t> column(bits.begin(), bits.begin() + kQkv);
        if (first_column->empty()) {
            *first_column = column;
        } else if (column != *first_column) {
            std::cerr << "gdn" << tag << ": column 0 changed with the call's width\n";
            ++failures;
        }
    }
    return failures + scratch.verify_guards("gdn" + tag) + parent.verify_preserved("gdn parent");
}

int run_linear_add(DevicePackedWeight& parent, int tokens, ops::LinearPolicy policy,
                   std::vector<std::uint16_t>* first_column = nullptr) {
    constexpr int kRows = 3072, kK = 8192;
    const auto x       = activations(kK, tokens, 301U);
    const auto x_bits  = bf16_bits(x);
    DeviceBuffer input = to_device(x_bits);
    auto residual      = make_bf16_activation(kRows, tokens, 401U);
    for (auto& v : residual) { v *= 50.0F; }
    round_to_bf16(residual);
    GuardedBf16Tensor out(kRows, tokens);
    out.copy_from_bits(bf16_bits(residual));
    Tensor tx(input.p, DType::BF16, {kK, tokens}), to = out.tensor();
    const auto capacity =
        ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_F32, kRows, kK, policy, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    DeviceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 1)});
    DeviceContext device;
    ops::linear_add(tx, parent.view(), to, policy, workspace, device.stream);
    cuda_synchronize(device.stream);
    const auto reference = a8(policy, tokens) ? represented(x) : as_double(x);
    auto expected        = oracle(parent.host, 0, kRows, reference, kK, tokens);
    const auto selected  = sampled_rows(kRows, kSamples);
    for (std::size_t r = 0; r < selected.size(); ++r) {
        for (int t = 0; t < tokens; ++t) {
            expected[r * static_cast<std::size_t>(tokens) + static_cast<std::size_t>(t)] +=
                residual[static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(selected[r])];
        }
    }
    const std::string tag = " " + name(policy) + " T=" + std::to_string(tokens);
    int failures = out.verify_guards("linear_add" + tag) + out.verify_fully_written("linear_add" + tag);
    failures += compare("linear_add" + tag, gather_rows(out.values(), kRows, 0, kRows, tokens, kSamples), expected,
                        kCriterion);
    if (first_column != nullptr) {
        // Column 0's residual is the same in every call (the generator's first column), so its
        // result must be too.
        const auto bits = out.bits();
        const std::vector<std::uint16_t> column(bits.begin(), bits.begin() + kRows);
        if (first_column->empty()) {
            *first_column = column;
        } else if (column != *first_column) {
            std::cerr << "linear_add" << tag << ": column 0 changed with the call's width\n";
            ++failures;
        }
    }
    return failures + scratch.verify_guards("linear_add" + tag) + parent.verify_preserved("linear_add parent");
}

} // namespace

int main() {
    int failures = 0;
    for (const int k : {3072, 8192}) {
        for (const int tokens : {1, 17, 300}) { failures += run_quantizer(k, tokens); }
    }

    const quantized_weight::PatternedWeightOptions options{.activation_scale = kScale};
    using P = ops::LinearPolicy;
    {
        DevicePackedWeight parent(
            quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 17408, 3072, 701U, options));
        for (int t : {1, 2, 5, 16, 17, 24, 32, 33, 64, 65, 80, 96, 97, 129, 160, 161}) {
            failures += run_attention(parent, t, P::A16Only);
        }
        for (int t : {16, 17, 32, 33, 96, 97, 128, 129, 192, 193, 288, 289, 300, 384, 385, 512, 1025}) {
            failures += run_attention(parent, t, P::AllowA8);
        }
        std::vector<std::uint16_t> column;
        for (int t : {1, 8, 9, 16, 17, 32, 33, 64, 65, 129}) { failures += run_attention(parent, t, P::A16ColumnInvariant, &column); }
    }
    {
        DevicePackedWeight parent(
            quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 20480, 3072, 703U, options));
        for (int t : {1, 2, 4, 5, 16, 17, 24, 25, 32, 33, 64, 65, 96, 97, 129}) { failures += run_gdn(parent, t, P::A16Only); }
        for (int t : {16, 17, 32, 33, 64, 65, 128, 129, 192, 193, 256, 257, 384, 385, 512, 513, 1025}) {
            failures += run_gdn(parent, t, P::AllowA8);
        }
        std::vector<std::uint16_t> column;
        for (int t : {1, 8, 9, 16, 17, 32, 33, 64, 65, 129}) { failures += run_gdn(parent, t, P::A16ColumnInvariant, &column); }
    }
    {
        DevicePackedWeight parent(
            quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 3072, 8192, 707U, options));
        for (int t : {1, 2, 8, 9, 16, 17, 33, 64, 65, 128, 129, 192, 193, 257, 384, 385}) {
            failures += run_linear_add(parent, t, P::A16Only);
        }
        for (int t : {16, 17, 64, 65, 128, 129, 256, 257, 384, 385, 512, 513, 768, 769, 1025, 2048}) {
            failures += run_linear_add(parent, t, P::AllowA8);
        }
        std::vector<std::uint16_t> column;
        for (int t : {1, 8, 9, 16, 17, 64, 65, 129}) { failures += run_linear_add(parent, t, P::A16ColumnInvariant, &column); }
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "fp8_e4m3fn_row_f32 projections: all checks passed\n";
    return 0;
}
