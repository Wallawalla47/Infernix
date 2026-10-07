// The text q/k norm+rope dispatch must compute the model's own RoPE theta and RMSNorm epsilon at
// every width. The fused Op fixes theta 1e7 and epsilon 1e-6, so a model with other constants
// has to take the three-call route even at the fused Op's decode widths.
#include "ops/op_tester.h"
#include "models/qwen3_5/execution/attention.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rope.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

// The rotation Op sizes its launch from the device's SM count.
DeviceExecutionView default_stream_execution() {
    int device = 0;
    int count  = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        throw std::runtime_error("cannot query the device's multiprocessor count");
    }
    return {nullptr, static_cast<std::int32_t>(count)};
}

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKeyHeads = 4;

int run_case(float theta, float epsilon, std::int32_t tokens, std::uint32_t seed) {
    const auto q_count = static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens;
    const auto k_count = static_cast<std::size_t>(kHeadDim) * kKeyHeads * tokens;
    std::vector<float> q(q_count), k(k_count), q_weight(kHeadDim), k_weight(kHeadDim);
    fill_uniform(q, seed, -4.0F, 4.0F);
    fill_uniform(k, seed + 1, -4.0F, 4.0F);
    fill_uniform(q_weight, seed + 2, -0.5F, 0.5F);
    fill_uniform(k_weight, seed + 3, -0.5F, 0.5F);
    std::vector<int> positions(static_cast<std::size_t>(tokens));
    // Far positions make a wrong theta visible in every rotated pair.
    fill_iota_i32(positions, 70000);

    DeviceBuffer dq = to_device_bf16(q), dk = to_device_bf16(k);
    DeviceBuffer dqw = to_device_bf16(q_weight), dkw = to_device_bf16(k_weight);
    DeviceBuffer dpos = to_device_i32(positions);
    DeviceBuffer dq_out = to_device_bf16(std::vector<float>(q_count));
    DeviceBuffer dk_out = to_device_bf16(std::vector<float>(k_count));
    DeviceBuffer dq_ref = to_device_bf16(std::vector<float>(q_count));
    DeviceBuffer dk_ref = to_device_bf16(std::vector<float>(k_count));

    Tensor q_in(dq.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    Tensor k_in(dk.p, DType::BF16, {kHeadDim, kKeyHeads, tokens});
    Tensor qw(dqw.p, DType::BF16, {kHeadDim});
    Tensor kw(dkw.p, DType::BF16, {kHeadDim});
    Tensor pos(dpos.p, DType::I32, {tokens});
    Tensor q_out(dq_out.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    Tensor k_out(dk_out.p, DType::BF16, {kHeadDim, kKeyHeads, tokens});
    Tensor q_ref(dq_ref.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    Tensor k_ref(dk_ref.p, DType::BF16, {kHeadDim, kKeyHeads, tokens});

    models::qwen3_5::RopeConfig rope;
    rope.rope_theta = theta;
    rope.rotary_dim = 64;
    const models::qwen3_5::AttentionConfig attention{kQueryHeads, kKeyHeads, kHeadDim};
    const ops::PreparedRope prepared = ops::prepare_rope(64, theta);

    models::qwen3_5::execution::text_qk_norm_rope(pos, rope, attention, epsilon, qw, kw, prepared,
                                                  q_in, k_in, q_out, k_out,
                                                  default_stream_execution());
    ops::rmsnorm(q_in, qw, epsilon, true, q_ref, nullptr);
    ops::rmsnorm(k_in, kw, epsilon, true, k_ref, nullptr);
    ops::rope(pos, prepared, q_ref, k_ref, default_stream_execution());
    cuda_synchronize();

    const std::string label = "text_qk_norm_rope theta=" + std::to_string(theta) +
                              " eps=" + std::to_string(epsilon) +
                              " T=" + std::to_string(tokens);
    int failures = verify_exact((label + " q").c_str(), from_device_bf16(dq_out, q_count),
                                from_device_bf16(dq_ref, q_count));
    failures += verify_exact((label + " k").c_str(), from_device_bf16(dk_out, k_count),
                             from_device_bf16(dk_ref, k_count));
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const std::int32_t tokens : {1, 16, 256, 257, 3584}) {
        // The native constants take the fused route at every width, a prefill chunk included, and
        // must match the three calls bit for bit.
        failures += run_case(1.0e7F, 1.0e-6F, tokens, 11U + tokens);
        // Any other constant must reach the three calls at every width.
        failures += run_case(5.0e6F, 1.0e-6F, tokens, 23U + tokens);
        failures += run_case(1.0e7F, 1.0e-5F, tokens, 37U + tokens);
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " text q/k norm+rope dispatch\n";
    return failures == 0 ? 0 : 1;
}
