#include "ops/op_family.h"
#include "infernix/ops/causal_conv1d_silu.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/gdn_gating.h"

#include "core/device.h"

#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;
using namespace infernix::test;

// gated_delta_net_replay_record_conv against its definition: causal_conv1d_silu_from_states_split, gdn_gating
// (gdn_gating_direct) and gated_delta_net_replay_record, bit for bit in the output and all three records,
// eagerly and as a captured programmatic dependent of the kernel that writes its convolution input.

namespace {

constexpr std::int32_t kDim        = 128;
constexpr std::int32_t kQkHeads    = 16;
constexpr std::int32_t kValueHeads = 48;
constexpr std::int32_t kChannels   = kDim * (2 * kQkHeads + kValueHeads);
constexpr std::int32_t kSlots      = 6;

std::vector<std::uint16_t> bf16_bits(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

struct Case {
    std::int32_t width;
    std::int32_t batch;
    bool fp16_state;
    bool a_direct;
};

int run_case(const Case& c, std::uint32_t seed) {
    const std::int32_t W = c.width, B = c.batch, T = W * B;
    const std::string label = "replay_record_conv W=" + std::to_string(W) + " B=" + std::to_string(B) +
                              (c.fp16_state ? " fp16-state" : " fp32-state") + (c.a_direct ? " a-direct" : "");

    // The producer's input, from which a from_states convolution writes the op's qkv: x drawn wide enough that
    // silu sees both signs and saturation, and the control rows reach softplus's linear branch.
    const std::vector<std::uint16_t> source_bits = bf16_bits(static_cast<std::size_t>(kChannels) * T, seed, -3.0F, 3.0F);
    const std::vector<std::uint16_t> weight_bits = bf16_bits(static_cast<std::size_t>(kChannels) * 4, seed + 1, -0.9F, 0.9F);
    const std::vector<std::uint16_t> history_bits =
        bf16_bits(static_cast<std::size_t>(kChannels) * 3 * kSlots, seed + 2, -2.0F, 2.0F);
    std::vector<std::uint16_t> ab_bits = bf16_bits(static_cast<std::size_t>(2 * kValueHeads) * T, seed + 3, -4.0F, 4.0F);
    ab_bits[0] = f32_to_bf16(24.0F); // a + dt_bias above softplus's threshold
    ab_bits[1] = 0x8000U;
    std::vector<float> a_log(kValueHeads), dt_bias(kValueHeads);
    fill_uniform(a_log, seed + 4, c.a_direct ? -6.0F : -1.5F, c.a_direct ? -0.05F : 1.5F);
    fill_uniform(dt_bias, seed + 5, -1.0F, 1.0F);
    std::vector<float> state(static_cast<std::size_t>(kDim) * kDim * kValueHeads * kSlots);
    fill_uniform(state, seed + 6, -0.05F, 0.05F);
    std::vector<std::int32_t> initial(static_cast<std::size_t>(B));
    for (std::int32_t row = 0; row < B; ++row) { initial[static_cast<std::size_t>(row)] = (row * 5 + 3) % kSlots; }

    DeviceBuffer source  = to_device(source_bits);
    DeviceBuffer qkv(source.bytes);
    DeviceBuffer weight  = to_device(weight_bits);
    DeviceBuffer history = to_device(history_bits);
    DeviceBuffer ab      = to_device(ab_bits);
    DeviceBuffer a_log_d = to_device(a_log);
    DeviceBuffer dt_d    = to_device(dt_bias);
    DeviceBuffer states  = c.fp16_state ? to_device(fp16_saturated_bits(state)) : to_device(state);
    DeviceBuffer initial_d = to_device(initial);

    const std::size_t qk_elements    = static_cast<std::size_t>(kDim) * kQkHeads * T;
    const std::size_t value_elements = static_cast<std::size_t>(kDim) * kValueHeads * T;
    const std::size_t gate_elements  = static_cast<std::size_t>(kValueHeads) * T;
    DeviceBuffer q(qk_elements * 2), k(qk_elements * 2), v(value_elements * 2);
    DeviceBuffer g(gate_elements * 4), beta(gate_elements * 4);
    struct Outputs {
        DeviceBuffer out, key, value, gate;
    };
    Outputs ref{DeviceBuffer(value_elements * 2), DeviceBuffer(qk_elements * 2), DeviceBuffer(value_elements * 2),
                DeviceBuffer(gate_elements * 8)};
    Outputs got{DeviceBuffer(value_elements * 2), DeviceBuffer(qk_elements * 2), DeviceBuffer(value_elements * 2),
                DeviceBuffer(gate_elements * 8)};

    const Tensor t_source(source.p, DType::BF16, {kChannels, W, B});
    Tensor t_qkv(qkv.p, DType::BF16, {kChannels, W, B});
    const Tensor t_weight(weight.p, DType::BF16, {kChannels, 4});
    const Tensor t_history(history.p, DType::BF16, {kChannels, 3, kSlots});
    const Tensor t_ab(ab.p, DType::BF16, {2 * kValueHeads, T});
    const Tensor t_a = t_ab.slice(0, 0, kValueHeads), t_b = t_ab.slice(0, kValueHeads, kValueHeads);
    const Tensor t_a_log(a_log_d.p, DType::FP32, {kValueHeads});
    const Tensor t_dt(dt_d.p, DType::FP32, {kValueHeads});
    Tensor t_states(states.p, c.fp16_state ? DType::FP16 : DType::FP32, {kDim, kDim, kValueHeads, kSlots});
    const Tensor t_initial(initial_d.p, DType::I32, {B});
    Tensor t_q(q.p, DType::BF16, {kDim * kQkHeads, T}), t_k(k.p, DType::BF16, {kDim * kQkHeads, T});
    Tensor t_v(v.p, DType::BF16, {kDim * kValueHeads, T});
    Tensor t_g(g.p, DType::FP32, {kValueHeads, T}), t_beta(beta.p, DType::FP32, {kValueHeads, T});
    const auto tensors = [&](Outputs& o) {
        return std::vector<Tensor>{Tensor(o.out.p, DType::BF16, {kDim, kValueHeads, W, B}),
                                   Tensor(o.key.p, DType::BF16, {kDim, kQkHeads, W, B}),
                                   Tensor(o.value.p, DType::BF16, {kDim, kValueHeads, W, B}),
                                   Tensor(o.gate.p, DType::FP32, {2, kValueHeads, W, B})};
    };
    std::vector<Tensor> ref_t = tensors(ref), got_t = tensors(got);
    const float scale         = 1.0F / std::sqrt(static_cast<float>(kDim));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto produce = [&] { ops::causal_conv1d_silu_from_states(t_source, t_weight, t_history, t_initial, t_qkv, stream); };
    const auto fused   = [&] {
        ops::gated_delta_net_replay_record_conv(t_qkv, t_weight, t_history, t_a, t_b, t_a_log, t_dt, c.a_direct, scale,
                                                t_states, t_initial, got_t[1], got_t[2], got_t[3], got_t[0], stream);
    };
    const auto poison = [&](Outputs& o) {
        for (DeviceBuffer* buffer : {&o.out, &o.key, &o.value, &o.gate}) {
            CUDA_CHECK(cudaMemsetAsync(buffer->p, 0xff, buffer->bytes, stream));
        }
    };

    poison(ref);
    produce();
    ops::causal_conv1d_silu_from_states_split(t_qkv, t_weight, t_history, t_initial, t_q, t_k, t_v, stream);
    if (c.a_direct) {
        ops::gdn_gating_direct(t_a, t_b, t_a_log, t_dt, t_g, t_beta, stream);
    } else {
        ops::gdn_gating(t_a, t_b, t_a_log, t_dt, t_g, t_beta, stream);
    }
    ops::gated_delta_net_replay_record(t_q.view({kDim, kQkHeads, W, B}), t_k.view({kDim, kQkHeads, W, B}),
                                       t_v.view({kDim, kValueHeads, W, B}), t_g.view({kValueHeads, W, B}),
                                       t_beta.view({kValueHeads, W, B}), scale, t_states, Tensor{}, t_initial,
                                       ref_t[1], ref_t[2], ref_t[3], ref_t[0], stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::vector<std::uint16_t> qkv_bits = from_device<std::uint16_t>(qkv, source_bits.size());

    int failures      = 0;
    const auto verify = [&](std::string_view pass) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const char* names[] = {"output", "key record", "value record", "gate record"};
        DeviceBuffer* lhs[] = {&ref.out, &ref.key, &ref.value, &ref.gate};
        DeviceBuffer* rhs[] = {&got.out, &got.key, &got.value, &got.gate};
        for (int i = 0; i < 4; ++i) {
            if (from_device<std::uint8_t>(*lhs[i], lhs[i]->bytes) != from_device<std::uint8_t>(*rhs[i], rhs[i]->bytes)) {
                std::cerr << label << " (" << pass << "): " << names[i] << " bits differ from the composed kernels\n";
                ++failures;
            }
        }
    };

    poison(got);
    fused();
    verify("eager");

    // Captured behind its producer: the convolution input is poisoned before the replay, so a read ahead of
    // the producer's completion would show.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    produce();
    fused();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaMemsetAsync(qkv.p, 0xff, qkv.bytes, stream));
    poison(got);
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    verify("graph");
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (from_device<std::uint16_t>(history, history_bits.size()) != history_bits ||
        from_device<std::uint16_t>(ab, ab_bits.size()) != ab_bits ||
        from_device<std::uint16_t>(qkv, qkv_bits.size()) != qkv_bits) {
        std::cerr << label << ": inputs were modified\n";
        ++failures;
    }
    const std::vector<std::uint8_t> state_bytes = from_device<std::uint8_t>(states, states.bytes);
    DeviceBuffer original = c.fp16_state ? to_device(fp16_saturated_bits(state)) : to_device(state);
    if (state_bytes != from_device<std::uint8_t>(original, original.bytes)) {
        std::cerr << label << ": the recurrent state was modified\n";
        ++failures;
    }
    return failures;
}

int expect_rejected(std::int32_t value_heads, std::int32_t width) {
    Tensor key(nullptr, DType::BF16, {kDim, kQkHeads, width, 1});
    Tensor value(nullptr, DType::BF16, {kDim, value_heads, width, 1});
    Tensor empty;
    try {
        ops::gated_delta_net_replay_record_conv(empty, empty, empty, empty, empty, empty, empty, false,
                                                1.0F / std::sqrt(128.0F), empty, empty, key, value, empty, value,
                                                nullptr);
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()).find("unsupported geometry") != std::string_view::npos) { return 0; }
    }
    std::cerr << "replay_record_conv accepted Hv=" << value_heads << " W=" << width << "\n";
    return 1;
}

} // namespace

INFERNIX_OP_TEST(gated_delta_net_replay_record_conv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = expect_rejected(48, 17) + expect_rejected(48, 1) + expect_rejected(32, 8);
    std::uint32_t seed = 4100U;
    for (std::int32_t width = 2; width <= ops::kGdnReplayConvMaxWidth; ++width) {
        failures += run_case({width, 1, false, false}, seed += 16);
        failures += run_case({width, 1, false, true}, seed += 16);
    }
    for (const Case c : {Case{4, 8, false, true}, Case{16, 8, false, false}, Case{9, 3, true, true},
                         Case{8, 2, true, false}, Case{5, 5, false, true}}) {
        failures += run_case(c, seed += 16);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net_replay_record_conv\n";
    return failures == 0 ? 0 : 1;
}
