#include "core/decode_graph.h"
#include "core/device.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/rmsnorm.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/input_projection_test_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::input_projection;

namespace {

constexpr int kHidden     = 5120;
constexpr int kParentRows = 14336;
constexpr float kEps      = 1.0e-6F;

template <class T>
int compare_bytes(const std::string& label, const T* reference, const T* fused, std::size_t count) {
    const auto expected = from_device<std::uint8_t>(reference, count * sizeof(T));
    const auto actual   = from_device<std::uint8_t>(fused, count * sizeof(T));
    if (actual == expected) { return 0; }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << label << ": byte mismatch at " << i << " (reference "
                      << unsigned(expected[i]) << ", fused " << unsigned(actual[i]) << ")\n";
            break;
        }
    }
    return 1;
}

// The complete fused formula in FP64: the offset RMSNorm of the residual, then the naive dot
// product with each sampled output row's decoded NVFP4 weight. The A4 activation quantization
// the route applies is inside the A4 criterion, as for the unfused Op.
std::vector<double> fused_oracle(const quantized_weight::PackedWeight& weight,
                                 std::int32_t weight_row_offset, std::int32_t output_rows,
                                 const std::vector<double>& normalized, int tokens) {
    std::vector<double> expected;
    for (const std::int32_t local_row : sampled_rows(output_rows, kA4SampleRows)) {
        std::vector<double> decoded(kHidden);
        for (int column = 0; column < kHidden; ++column) {
            decoded[column] =
                quantized_weight::logical_weight_fp64(weight, weight_row_offset + local_row, column);
        }
        for (int token = 0; token < tokens; ++token) {
            const double* input = normalized.data() + static_cast<std::size_t>(token) * kHidden;
            double sum          = 0.0;
            for (int column = 0; column < kHidden; ++column) { sum += decoded[column] * input[column]; }
            expected.push_back(sum);
        }
    }
    return expected;
}

std::vector<double> offset_rmsnorm_fp64(const std::vector<float>& residual,
                                        const std::vector<float>& gain, int tokens) {
    std::vector<double> normalized(residual.size());
    for (int token = 0; token < tokens; ++token) {
        const float* x = residual.data() + static_cast<std::size_t>(token) * kHidden;
        double squares = 0.0;
        for (int d = 0; d < kHidden; ++d) { squares += double(x[d]) * double(x[d]); }
        const double inverse = 1.0 / std::sqrt(squares / kHidden + double(kEps));
        for (int d = 0; d < kHidden; ++d) {
            normalized[static_cast<std::size_t>(token) * kHidden + d] =
                double(x[d]) * inverse * (double(gain[d]) + 1.0);
        }
    }
    return normalized;
}

int run_case(const Weight& weight, const quantized_weight::PackedWeight& host, int tokens,
             bool measure) {
    const auto values            = make_bf16_activation(kHidden, tokens, 871U + tokens);
    auto gains                   = make_bf16_activation(kHidden, 1, 119U);
    const auto value_bits        = bf16_bits(values);
    const auto gain_bits         = bf16_bits(gains);
    DeviceBuffer residual_buffer = to_device(value_bits);
    DeviceBuffer gain_buffer     = to_device(gain_bits);
    DeviceBuffer hidden_buffer(static_cast<std::size_t>(kHidden) * tokens * 2);
    Tensor residual(residual_buffer.p, DType::BF16, {kHidden, tokens});
    Tensor gain(gain_buffer.p, DType::BF16, {kHidden});
    Tensor hidden(hidden_buffer.p, DType::BF16, {kHidden, tokens});
    ops::rmsnorm(residual, gain, kEps, true, hidden, nullptr);

    const std::size_t capacity = ops::attn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kParentRows, kHidden, ops::LinearPolicy::AllowA4, tokens, tokens);
    DeviceArena reference_workspace(capacity);
    DeviceArena fused_workspace(capacity);
    int failures = 0;
    {
        auto reference_scope = reference_workspace.scope();
        auto fused_scope     = fused_workspace.scope();
        const auto layout = ops::detail::nvfp4_attn_input_scale_layout(tokens);
        const auto reference_planes =
            ops::detail::allocate_nvfp4_a4_workspace(reference_workspace, tokens, kHidden);
        const auto fused_planes =
            ops::detail::allocate_nvfp4_a4_workspace(fused_workspace, tokens, kHidden);
        ops::detail::launch_nvfp4_a4_quantize(hidden, weight, reference_planes, layout, nullptr);
        ops::detail::launch_nvfp4_attn_input_fused_rmsnorm_quantize(
            residual, gain, kEps, weight.input_scale_divisor, fused_planes, layout, nullptr);
        cuda_synchronize();
        const std::string suffix = " T=" + std::to_string(tokens);
        failures += compare_bytes("codes" + suffix, reference_planes.codes, fused_planes.codes,
                                  static_cast<std::size_t>(tokens) * kHidden / 2);
        failures += compare_bytes("scales" + suffix, reference_planes.scales, fused_planes.scales,
                                  reference_planes.scale_bytes);
    }

    constexpr int kQRows  = 6144;
    constexpr int kKvRows = 1024;
    DeviceBuffer reference_q(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer reference_gate(static_cast<std::size_t>(kQRows) * tokens * 2);
    DeviceBuffer reference_k(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer reference_v(static_cast<std::size_t>(kKvRows) * tokens * 2);
    DeviceBuffer fused_q(reference_q.bytes), fused_gate(reference_gate.bytes);
    DeviceBuffer fused_k(reference_k.bytes), fused_v(reference_v.bytes);
    Tensor rq(reference_q.p, DType::BF16, {kQRows, tokens});
    Tensor rg(reference_gate.p, DType::BF16, {kQRows, tokens});
    Tensor rk(reference_k.p, DType::BF16, {kKvRows, tokens});
    Tensor rv(reference_v.p, DType::BF16, {kKvRows, tokens});
    Tensor fq(fused_q.p, DType::BF16, {kQRows, tokens});
    Tensor fg(fused_gate.p, DType::BF16, {kQRows, tokens});
    Tensor fk(fused_k.p, DType::BF16, {kKvRows, tokens});
    Tensor fv(fused_v.p, DType::BF16, {kKvRows, tokens});
    ops::attn_input_proj(hidden, weight, rq, rg, rk, rv, ops::LinearPolicy::AllowA4,
                         reference_workspace, nullptr);
    ops::attn_input_proj_fused_rmsnorm_nvfp4(residual, gain, kEps, weight, fq, fg, fk, fv,
                                             ops::LinearPolicy::AllowA4, fused_workspace, nullptr);
    cuda_synchronize();
    const std::string suffix = " T=" + std::to_string(tokens);
    failures += compare_bytes("q" + suffix, static_cast<const std::uint8_t*>(reference_q.p),
                              static_cast<const std::uint8_t*>(fused_q.p), reference_q.bytes);
    failures += compare_bytes("gate" + suffix, static_cast<const std::uint8_t*>(reference_gate.p),
                              static_cast<const std::uint8_t*>(fused_gate.p), reference_gate.bytes);
    failures += compare_bytes("k" + suffix, static_cast<const std::uint8_t*>(reference_k.p),
                              static_cast<const std::uint8_t*>(fused_k.p), reference_k.bytes);
    failures += compare_bytes("v" + suffix, static_cast<const std::uint8_t*>(reference_v.p),
                              static_cast<const std::uint8_t*>(fused_v.p), reference_v.bytes);
    {
        // The primary verdict: the fused route against the FP64 formula. Byte parity with the
        // unfused chain above is supplementary.
        const std::vector<double> normalized = offset_rmsnorm_fp64(values, gains, tokens);
        const auto verify = [&](const char* name, const DeviceBuffer& output, std::int32_t rows,
                                std::int32_t weight_row_offset) {
            const std::vector<double> actual = gather_rows(
                from_device_bf16(output, static_cast<std::size_t>(rows) * tokens), rows, 0, rows,
                tokens, kA4SampleRows);
            return compare(std::string("fused ") + name + " FP64 oracle" + suffix, actual,
                           fused_oracle(host, weight_row_offset, rows, normalized, tokens),
                           kAttnInputProjA4Tolerance);
        };
        failures += verify("q", fused_q, kQRows, 0);
        failures += verify("k", fused_k, kKvRows, kQRows);
        failures += verify("gate", fused_gate, kQRows, kQRows + kKvRows);
        failures += verify("v", fused_v, kKvRows, 2 * kQRows + kKvRows);
    }
    if (tokens == 1500 && failures == 0) {
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::attn_input_proj_fused_rmsnorm_nvfp4(residual, gain, kEps, weight, fq, fg, fk, fv,
                                                     ops::LinearPolicy::AllowA4, fused_workspace,
                                                     device.stream);
        });
        graph.instantiate(definition);
        graph.launch(device.stream);
        cuda_synchronize(device.stream);
        failures +=
            compare_bytes("graph q" + suffix, static_cast<const std::uint8_t*>(reference_q.p),
                          static_cast<const std::uint8_t*>(fused_q.p), reference_q.bytes);
    }
    if (reference_workspace.used() != 0 || fused_workspace.used() != 0) {
        std::cerr << "attention input workspace scope was not released\n";
        ++failures;
    }
    if (measure && tokens != 1500 && failures == 0) {
        const auto reference_stage = [&] {
            ops::rmsnorm(residual, gain, kEps, true, hidden, nullptr);
            ops::attn_input_proj(hidden, weight, rq, rg, rk, rv, ops::LinearPolicy::AllowA4,
                                 reference_workspace, nullptr);
        };
        const auto fused_stage = [&] {
            ops::attn_input_proj_fused_rmsnorm_nvfp4(residual, gain, kEps, weight, fq, fg, fk, fv,
                                                     ops::LinearPolicy::AllowA4, fused_workspace,
                                                     nullptr);
        };
        const auto median_wall_us = [](const auto& stage) {
            for (int i = 0; i < 5; ++i) {
                stage();
                cuda_synchronize();
            }
            std::vector<double> samples;
            for (int i = 0; i < 31; ++i) {
                const auto start = std::chrono::steady_clock::now();
                stage();
                cuda_synchronize();
                const auto stop = std::chrono::steady_clock::now();
                samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
            }
            std::sort(samples.begin(), samples.end());
            return samples[samples.size() / 2];
        };
        const double reference_us = median_wall_us(reference_stage);
        const double fused_us     = median_wall_us(fused_stage);
        std::cout << "ATTN_INPUT_STAGE T=" << tokens << " reference_us=" << reference_us
                  << " fused_us=" << fused_us << " delta_us=" << reference_us - fused_us << '\n';
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (cuda_unavailable()) { return 77; }
        quantized_weight::PatternedWeightOptions options;
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
        DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::NVFP4, kParentRows,
                                                                          kHidden, 331U, options));
        const Weight weight = parent.view();
        if (ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::A16Only,
                                                              1024) ||
            ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                              511) ||
            !ops::attn_input_proj_fused_rmsnorm_nvfp4_eligible(weight, ops::LinearPolicy::AllowA4,
                                                               512)) {
            std::cerr << "fused route eligibility mismatch\n";
            return 1;
        }
        int failures       = 0;
        const bool measure = std::getenv("INFERNIX_MEASURE_FUSED_STAGE") != nullptr;
        // 512 and 700 read the 128-token scale tiles (700 a ragged last tile); from 1024 the tiles
        // are 256 tokens. 3584 is the production prefill chunk under the fast prefill kernel.
        for (const int tokens : {512, 700, 1024, 1500, 2048, 3584, 4096}) {
            failures += run_case(weight, parent.host, tokens, measure);
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fused attention input test: " << error.what() << '\n';
        return 1;
    }
}
