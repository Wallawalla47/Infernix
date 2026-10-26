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
//   - Per-token parents (no static scale; dealignai's Ornith-1.5-122B-UNCENSORED-NVFP4, ModelOpt
//     FP8_PER_CHANNEL_PER_TOKEN): each column's encoding bit for bit (scale fl32(max |x| / 448), code
//     e4m3_rn_satfinite(fl32(x * fl32(1 / scale)))), and the three projections' A8 routes against FP64
//     sums over the values each column's codes represent.
//   - The per-row FP8 shared expert (linear_swiglu [2048, 3072], linear [3072, 1024]) and vocabulary head
//     (linear [248320, 3072], BF16 and FP32 logits): under A16ColumnInvariant a column's outputs are the
//     same bits at every width (their FP64 criteria are test_fp8_a16's and linear_swiglu_fp8's).

#include "ops/op_family.h"
#include "core/weight.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
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
#include <functional>
#include <initializer_list>
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

// A per-token column's scale: fl32(max |x| / 448), 0 for an all-zero column.
float per_token_scale(const std::vector<float>& x, int k, int token) {
    float maximum = 0.0F;
    for (int c = 0; c < k; ++c) { maximum = std::max(maximum, std::fabs(x[static_cast<std::size_t>(token) * k + c])); }
    return maximum > 0.0F ? maximum / 448.0F : 0.0F;
}

// The values a per-token parent's codes represent ([k][tokens]): code * the column's scale.
std::vector<double> represented_per_token(const std::vector<float>& x, int k, int tokens) {
    std::vector<double> out(x.size());
    for (int t = 0; t < tokens; ++t) {
        const float scale   = per_token_scale(x, k, t);
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        for (int c = 0; c < k; ++c) {
            const std::size_t i = static_cast<std::size_t>(t) * k + c;
            out[i]              = fp8_static::e4m3_value(fp8_static::code(x[i], inverse)) * static_cast<double>(scale);
        }
    }
    return out;
}

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

// A per-token parent's encoding (input_scale 0): every column's scale and codes bit for bit.
int run_per_token_quantizer(int k, int tokens) {
    const auto x    = activations(k, tokens, 13U + static_cast<std::uint32_t>(k + tokens));
    const auto bits = bf16_bits(x);
    DeviceBuffer input = to_device(bits);
    const std::size_t bytes = ops::detail::fp8_a8_workspace_capacity_bytes(tokens, k);
    GuardedDeviceBuffer storage(bytes);
    DeviceArena arena(DeviceSpan{storage.data(), bytes});
    const auto workspace = ops::detail::allocate_fp8_a8_workspace(arena, tokens, k);
    Weight weight{};
    weight.qtype = QType::FP8_E4M3FN_ROW_F32;
    weight.n     = 16;
    weight.k     = k;
    DeviceContext device;
    ops::detail::launch_fp8_a8_quantize(Tensor(input.p, DType::BF16, {k, tokens}), weight, workspace, device.stream);
    cuda_synchronize(device.stream);
    const auto codes  = from_device<std::uint8_t>(workspace.codes, static_cast<std::size_t>(k) * tokens);
    const auto scales = from_device<float>(workspace.scales, static_cast<std::size_t>(tokens));
    std::size_t bad = 0, scale_bad = 0, saturated = 0;
    for (int t = 0; t < tokens; ++t) {
        const float scale   = per_token_scale(x, k, t);
        const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
        scale_bad += std::memcmp(&scales[static_cast<std::size_t>(t)], &scale, 4) != 0;
        for (int c = 0; c < k; ++c) {
            const std::size_t i     = static_cast<std::size_t>(t) * k + c;
            const std::uint8_t want = fp8_static::code(x[i], inverse);
            bad += codes[i] != want;
            saturated += (want & 0x7F) == 0x7E ? 1 : 0;
        }
    }
    const std::string where = "per-token A8 encoding K=" + std::to_string(k) + " T=" + std::to_string(tokens);
    // Each column's largest magnitude takes the top code.
    if (bad != 0 || scale_bad != 0 || saturated < static_cast<std::size_t>(tokens)) {
        std::cerr << where << ": " << bad << " codes and " << scale_bad << " scales differ (" << saturated
                  << " at the top code)\n";
        return 1;
    }
    return storage.verify_guards(where);
}

// ------------------------------------------------------------------------------ projections

// The values an A8 call's oracle reads: the static rule's, or each column's own scale for a per-token
// parent, else the BF16 activations themselves.
std::vector<double> oracle_input(const std::vector<float>& x, int k, int tokens, ops::LinearPolicy policy,
                                 bool per_token, int first_a8 = 17) {
    if (!a8(policy, tokens, first_a8)) { return as_double(x); }
    return per_token ? represented_per_token(x, k, tokens) : represented(x);
}

int run_attention(DevicePackedWeight& parent, int tokens, ops::LinearPolicy policy,
                  std::vector<std::uint16_t>* first_column = nullptr, bool per_token = false) {
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
    const auto reference = oracle_input(x, kHidden, tokens, policy, per_token);
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
            std::vector<std::uint16_t>* first_column = nullptr, bool per_token = false) {
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
    const auto reference = oracle_input(x, kHidden, tokens, policy, per_token);
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
                   std::vector<std::uint16_t>* first_column = nullptr, bool per_token = false) {
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
    const auto reference = oracle_input(x, kK, tokens, policy, per_token);
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

// Under A16ColumnInvariant every column of a call of each width in `widths` equals, bit for bit, the
// same column computed alone: the per-row FP8 shared expert's gate/up (linear_swiglu) and down, and the
// head with BF16 and FP32 logits. `rows` is the output's row count, `word` its element bytes.
int run_invariance(const char* label, DevicePackedWeight& parent, int k, int rows, std::size_t word,
                   std::initializer_list<int> widths,
                   const std::function<void(const Tensor&, Tensor&, WorkspaceArena&, cudaStream_t)>& call) {
    const int widest   = std::max(widths);
    const auto x       = activations(k, widest, 503U + static_cast<std::uint32_t>(rows));
    DeviceBuffer input = to_device(bf16_bits(x));
    DeviceBuffer output(static_cast<std::size_t>(rows) * widest * word);
    WorkspaceArena workspace(std::size_t{1} << 20);
    DeviceContext device;
    const auto run = [&](int column, int tokens) {
        const Tensor tx(static_cast<std::uint16_t*>(input.p) + static_cast<std::size_t>(column) * k, DType::BF16,
                        {k, tokens});
        Tensor out(output.p, word == 4 ? DType::FP32 : DType::BF16, {rows, tokens});
        call(tx, out, workspace, device.stream);
        cuda_synchronize(device.stream);
        return from_device<std::uint8_t>(output.p, static_cast<std::size_t>(rows) * tokens * word);
    };
    std::vector<std::vector<std::uint8_t>> alone;
    for (int column = 0; column < widest; ++column) { alone.push_back(run(column, 1)); }
    int failures = 0;
    for (const int tokens : widths) {
        const auto together = run(0, tokens);
        std::size_t differing = 0;
        for (int column = 0; column < tokens; ++column) {
            differing += std::memcmp(together.data() + static_cast<std::size_t>(column) * rows * word,
                                     alone[static_cast<std::size_t>(column)].data(), rows * word) != 0;
        }
        if (differing != 0) {
            std::cerr << label << " T=" << tokens << ": " << differing << " columns differ from the columns alone\n";
            ++failures;
        }
    }
    return failures + parent.verify_preserved(label);
}

} // namespace

INFERNIX_OP_TEST(fp8_row_f32) {
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
    // Per-token parents (no static scale): their encoding, then the three projections' A8 routes.
    for (const int k : {3072, 8192}) {
        for (const int tokens : {1, 17, 300}) { failures += run_per_token_quantizer(k, tokens); }
    }
    {
        DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 17408, 3072, 709U));
        for (int t : {16, 17, 33, 129, 193, 385, 1025}) { failures += run_attention(parent, t, P::AllowA8, nullptr, true); }
    }
    {
        DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 20480, 3072, 711U));
        for (int t : {16, 17, 33, 129, 257, 513, 1025}) { failures += run_gdn(parent, t, P::AllowA8, nullptr, true); }
    }
    {
        DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 3072, 8192, 713U));
        for (int t : {16, 17, 65, 129, 257, 513, 1025}) { failures += run_linear_add(parent, t, P::AllowA8, nullptr, true); }
    }
    // The per-row FP8 shared expert and head under A16ColumnInvariant.
    const auto widths = {2, 5, 8, 9, 16, 17, 33, 64, 65};
    {
        DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 2048, 3072, 715U));
        failures += run_invariance("shared gate/up", parent, 3072, 1024, 2, widths,
                                   [&](const Tensor& x, Tensor& out, WorkspaceArena& ws, cudaStream_t s) {
                                       ops::linear_swiglu(x, parent.view(), out, P::A16ColumnInvariant, ws, s);
                                   });
    }
    {
        DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 3072, 1024, 717U));
        failures += run_invariance("shared down", parent, 1024, 3072, 2, widths,
                                   [&](const Tensor& x, Tensor& out, WorkspaceArena& ws, cudaStream_t s) {
                                       ops::linear(x, parent.view(), out, P::A16ColumnInvariant, ws, s);
                                   });
    }
    {
        DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, 248320, 3072, 719U));
        for (const std::size_t word : {std::size_t{2}, std::size_t{4}}) {
            failures += run_invariance(word == 4 ? "head (FP32 logits)" : "head (BF16 logits)", parent, 3072, 248320, word,
                                       {2, 8, 9, 33}, [&](const Tensor& x, Tensor& out, WorkspaceArena& ws, cudaStream_t s) {
                                           ops::linear(x, parent.view(), out, P::A16ColumnInvariant, ws, s);
                                       });
        }
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "fp8_e4m3fn_row_f32 projections: all checks passed\n";
    return 0;
}
