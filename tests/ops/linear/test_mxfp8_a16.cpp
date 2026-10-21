// MXFP8 (fp8_e4m3fn_g32_e8m0, g32_scale_rows16_v1) Linear with 16-bit activations: the Qwen3.8-Flash-Next
// shapes of NVIDIA FP4-Local against the FP64 oracle of the represented weights at every decode and
// verification width and at the prefill tile boundaries; column invariance at every width (a column
// computed alone equals the same column in calls of every route's widths); acceptance of the
// column-invariant policy and refusal of malformed weights.

#include "core/weight.h"
#include "ops/dirty_shared_memory.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/mxfp8/mxfp8_format.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test::linear;

constexpr std::pair<int, int> kShapes[] = {{16384, 2560}, {13952, 2560}, {2560, 6144}, {1280, 2560}, {2560, 640}};

int run_flash_next_mxfp8_a16() {
    int failures       = 0;
    std::uint32_t seed = 1901U;
    for (const auto& [n, k] : kShapes) {
        std::vector<Invocation> calls;
        for (int t = 1; t <= 33; ++t) calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
        for (int t : {48, 63, 64, 65, 96, 97, 127, 128, 129, 192, 257, 384, 512, 769, 1024, 1537, 2049, 4097})
            calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only});
        for (int t : {3, 9, 17, 33, 97}) calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
        for (int t : {1, 8, 16, 128}) calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
        for (int t : {1, 9, 17}) calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16ColumnInvariant});
        calls.push_back({1, CallForm::A16Convenience});
        failures += run_shape("MXFP8_A16 Flash-Next", ActivationCompute::A16, make_mxfp8_weight,
                              {n, k, seed++, Comparison::Sampled, true, calls});
    }
    return failures;
}

// Qwen3.8-Flash-Next's exact route relies on this (prefill calls, taps and resumes on the 16-token
// grid, any decode batch): a column is bit-identical whether computed alone or in a call of any
// width. The widths cover every route seam of every shape (the schedule changes at 7, 9, 12, 16, 32,
// 48, 64, 96, 128, 256, 384, 512, 768, 1,024, 1,536, 2,048, 3,072 and 4,096 columns) and partial token
// tiles on both sides of each.
int mxfp8_a16_column_invariance() {
    int failures = 0;
    for (const auto& [n, k, columns] : {std::tuple{16384, 2560, 4113}, std::tuple{13952, 2560, 1553},
                                        std::tuple{2560, 6144, 1553}, std::tuple{1280, 2560, 3089},
                                        std::tuple{2560, 640, 2065}}) {
        // Every finite E4M3 code (the fixture) and activations over 25 binades, so FP32 sums round and
        // any difference in reduction order changes bits.
        auto host           = make_mxfp8_weight(n, k, 1951U + static_cast<std::uint32_t>(n));
        std::uint32_t state = static_cast<std::uint32_t>(k) ^ static_cast<std::uint32_t>(n);
        const auto next     = [&state] { return state = state * 1664525U + 1013904223U; };
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * columns);
        for (auto& bits : activation) {
            const std::uint32_t r = next();
            bits = static_cast<std::uint16_t>(((r >> 31) << 15) | ((107U + (r >> 8) % 25U) << 7) | (r & 0x7FU));
        }
        const std::size_t out_bytes = static_cast<std::size_t>(n) * columns * 2;
        void *weight = nullptr, *x = nullptr, *y = nullptr;
        infernix::test::cuda_check(cudaMalloc(&weight, host.payload.size()), "weight");
        infernix::test::cuda_check(cudaMalloc(&x, activation.size() * 2), "activation");
        infernix::test::cuda_check(cudaMalloc(&y, out_bytes), "output");
        infernix::test::cuda_check(cudaMemcpy(weight, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice),
                                   "upload weight");
        infernix::test::cuda_check(cudaMemcpy(x, activation.data(), activation.size() * 2, cudaMemcpyHostToDevice),
                                   "upload activation");
        const Weight w = host.device_weight(weight);
        std::vector<std::uint16_t> alone(static_cast<std::size_t>(n) * columns);
        for (int column = 0; column < columns; ++column) {
            Tensor input(static_cast<std::uint16_t*>(x) + static_cast<std::size_t>(column) * k, DType::BF16, {k, 1});
            Tensor output(y, DType::BF16, {n, 1});
            ops::linear(input, w, output, nullptr);
            infernix::test::cuda_check(cudaMemcpy(alone.data() + static_cast<std::size_t>(column) * n, y,
                                                  static_cast<std::size_t>(n) * 2, cudaMemcpyDeviceToHost),
                                       "read single column");
        }
        for (const int t : {2,    5,    6,    7,    8,    9,    10,   12,   13,   15,   16,   17,   24,   31,   32,
                            33,   47,   48,   49,   63,   64,   65,   95,   96,   97,   127,  128,  129,  255,  256,
                            257,  383,  384,  385,  511,  512,  513,  767,  768,  769,  1023, 1024, 1025, 1041, 1535,
                            1536, 1537, 1553, 2047, 2048, 2049, 2065, 3071, 3072, 3073, 3089, 4095, 4096, 4097, 4113}) {
            if (t > columns) { continue; }
            Tensor input(x, DType::BF16, {k, t});
            Tensor output(y, DType::BF16, {n, t});
            ops::linear(input, w, output, nullptr);
            std::vector<std::uint16_t> together(static_cast<std::size_t>(n) * t);
            infernix::test::cuda_check(cudaMemcpy(together.data(), y, together.size() * 2, cudaMemcpyDeviceToHost),
                                       "read columns");
            std::size_t differing = 0;
            for (std::size_t i = 0; i < together.size(); ++i) differing += together[i] != alone[i];
            if (differing != 0) {
                std::cerr << "MXFP8_A16 [" << n << "," << k << "] T=" << t << ": " << differing
                          << " outputs differ from the columns computed alone\n";
                ++failures;
            }
        }
        cudaFree(weight);
        cudaFree(x);
        cudaFree(y);
    }
    return failures;
}

// Calls at decode widths with every SM's shared memory dirtied before each one: the route must not
// read shared memory its loads did not fill.
int mxfp8_a16_dirty_shared() {
    int failures = 0;
    for (const auto& [n, k] : kShapes) {
        auto host           = make_mxfp8_weight(n, k, 6007U + static_cast<std::uint32_t>(n));
        std::uint32_t state = static_cast<std::uint32_t>(n) * 17U + static_cast<std::uint32_t>(k);
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * 2);
        for (auto& bits : activation) {
            state = state * 1664525U + 1013904223U;
            bits  = static_cast<std::uint16_t>(((state >> 31) << 15) | ((107U + (state >> 8) % 25U) << 7) | (state & 0x7FU));
        }
        void *weight = nullptr, *x = nullptr, *y = nullptr;
        infernix::test::cuda_check(cudaMalloc(&weight, host.payload.size()), "weight");
        infernix::test::cuda_check(cudaMalloc(&x, activation.size() * 2), "activation");
        infernix::test::cuda_check(cudaMalloc(&y, static_cast<std::size_t>(n) * 2 * 2), "output");
        infernix::test::cuda_check(cudaMemcpy(weight, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice),
                                   "upload weight");
        infernix::test::cuda_check(cudaMemcpy(x, activation.data(), activation.size() * 2, cudaMemcpyHostToDevice),
                                   "upload activation");
        const Weight w = host.device_weight(weight);
        for (const int t : {1, 2}) {
            Tensor input(x, DType::BF16, {k, t});
            Tensor output(y, DType::BF16, {n, t});
            std::vector<std::uint16_t> first(static_cast<std::size_t>(n) * t), again(first.size());
            int differing = 0;
            for (int repeat = 0; repeat < 32; ++repeat) {
                infernix::test::dirty_shared_memory(0x9E3779B9U * static_cast<std::uint32_t>(repeat + 1), nullptr);
                ops::linear(input, w, output, nullptr);
                infernix::test::cuda_check(cudaDeviceSynchronize(), "linear");
                auto& target = repeat == 0 ? first : again;
                infernix::test::cuda_check(cudaMemcpy(target.data(), y, target.size() * 2, cudaMemcpyDeviceToHost),
                                           "read output");
                differing += repeat != 0 && again != first ? 1 : 0;
            }
            if (differing != 0) {
                std::cerr << "MXFP8_A16 [" << n << "," << k << "] T=" << t << ": " << differing
                          << " of 31 calls after dirtied shared memory differ from the first\n";
                ++failures;
            }
        }
        cudaFree(weight);
        cudaFree(x);
        cudaFree(y);
    }
    return failures;
}

// Back-to-back identical calls at decode widths, as a decode round issues them: 256 launches into
// their own outputs with no host synchronization between them, eagerly and as a replayed CUDA graph
// (whose captured launches are programmatic), while a device copy runs on another stream. Every
// output must equal the first.
int mxfp8_a16_back_to_back() {
    int failures = 0;
    cudaStream_t stream = nullptr, other = nullptr;
    infernix::test::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    infernix::test::cuda_check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "other stream");
    constexpr std::size_t kNoiseBytes = std::size_t{256} << 20;
    constexpr int kLaunches           = 256;
    void *noise_a = nullptr, *noise_b = nullptr;
    infernix::test::cuda_check(cudaMalloc(&noise_a, kNoiseBytes), "noise");
    infernix::test::cuda_check(cudaMalloc(&noise_b, kNoiseBytes), "noise");
    for (const auto& [n, k] : kShapes) {
        auto host           = make_mxfp8_weight(n, k, 5113U + static_cast<std::uint32_t>(n));
        std::uint32_t state = static_cast<std::uint32_t>(n) * 13U + static_cast<std::uint32_t>(k);
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * 2);
        for (auto& bits : activation) {
            state = state * 1664525U + 1013904223U;
            bits  = static_cast<std::uint16_t>(((state >> 31) << 15) | ((107U + (state >> 8) % 25U) << 7) | (state & 0x7FU));
        }
        void *weight = nullptr, *x = nullptr, *y = nullptr;
        const std::size_t out_elements = static_cast<std::size_t>(n) * 2;
        infernix::test::cuda_check(cudaMalloc(&weight, host.payload.size()), "weight");
        infernix::test::cuda_check(cudaMalloc(&x, activation.size() * 2), "activation");
        infernix::test::cuda_check(cudaMalloc(&y, out_elements * 2 * kLaunches), "outputs");
        infernix::test::cuda_check(cudaMemcpy(weight, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice),
                                   "upload weight");
        infernix::test::cuda_check(cudaMemcpy(x, activation.data(), activation.size() * 2, cudaMemcpyHostToDevice),
                                   "upload activation");
        const Weight w = host.device_weight(weight);
        for (const int t : {1, 2}) {
            Tensor input(x, DType::BF16, {k, t});
            const auto issue = [&] {
                for (int launch = 0; launch < kLaunches; ++launch) {
                    Tensor output(static_cast<std::uint16_t*>(y) + launch * out_elements, DType::BF16, {n, t});
                    ops::linear(input, w, output, stream);
                }
            };
            cudaGraph_t graph      = nullptr;
            cudaGraphExec_t replay = nullptr;
            infernix::test::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
            issue();
            infernix::test::cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
            infernix::test::cuda_check(cudaGraphInstantiate(&replay, graph, 0), "instantiate");
            for (int round = 0; round < 8; ++round) {
                const bool graphed = round % 2 == 1;
                infernix::test::cuda_check(cudaMemsetAsync(y, 0xFF, out_elements * 2 * kLaunches, stream), "poison");
                infernix::test::cuda_check(cudaMemcpyAsync(noise_b, noise_a, kNoiseBytes, cudaMemcpyDeviceToDevice, other),
                                           "noise copy");
                if (graphed) {
                    infernix::test::cuda_check(cudaGraphLaunch(replay, stream), "replay");
                } else {
                    issue();
                }
                infernix::test::cuda_check(cudaStreamSynchronize(stream), "launches");
                std::vector<std::uint16_t> outputs(out_elements * kLaunches);
                infernix::test::cuda_check(cudaMemcpy(outputs.data(), y, outputs.size() * 2, cudaMemcpyDeviceToHost),
                                           "read outputs");
                int differing = 0;
                for (int launch = 1; launch < kLaunches; ++launch) {
                    differing += std::equal(outputs.begin(), outputs.begin() + static_cast<std::ptrdiff_t>(t * n),
                                            outputs.begin() + static_cast<std::ptrdiff_t>(launch * out_elements))
                                     ? 0
                                     : 1;
                }
                if (differing != 0) {
                    std::cerr << "MXFP8_A16 [" << n << "," << k << "] T=" << t << (graphed ? " graph" : " eager")
                              << " round " << round << ": " << differing << " of " << kLaunches - 1
                              << " back-to-back calls differ from the first\n";
                    ++failures;
                }
            }
            cudaGraphExecDestroy(replay);
            cudaGraphDestroy(graph);
        }
        cudaFree(weight);
        cudaFree(x);
        cudaFree(y);
    }
    infernix::test::cuda_check(cudaDeviceSynchronize(), "noise");
    cudaFree(noise_a);
    cudaFree(noise_b);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(other);
    return failures;
}

// Repeated identical calls at decode and verification widths give identical bits, with a device copy
// running on another stream so the CTAs' timing varies between repeats.
int mxfp8_a16_repeatability() {
    int failures = 0;
    cudaStream_t stream = nullptr, other = nullptr;
    infernix::test::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    infernix::test::cuda_check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "other stream");
    constexpr std::size_t kNoiseBytes = std::size_t{256} << 20;
    void *noise_a = nullptr, *noise_b = nullptr;
    infernix::test::cuda_check(cudaMalloc(&noise_a, kNoiseBytes), "noise");
    infernix::test::cuda_check(cudaMalloc(&noise_b, kNoiseBytes), "noise");
    for (const auto& [n, k] : kShapes) {
        auto host           = make_mxfp8_weight(n, k, 4441U + static_cast<std::uint32_t>(n));
        std::uint32_t state = static_cast<std::uint32_t>(n) * 7U + static_cast<std::uint32_t>(k);
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * 16);
        for (auto& bits : activation) {
            state = state * 1664525U + 1013904223U;
            bits  = static_cast<std::uint16_t>(((state >> 31) << 15) | ((107U + (state >> 8) % 25U) << 7) | (state & 0x7FU));
        }
        void *weight = nullptr, *x = nullptr, *y = nullptr;
        infernix::test::cuda_check(cudaMalloc(&weight, host.payload.size()), "weight");
        infernix::test::cuda_check(cudaMalloc(&x, activation.size() * 2), "activation");
        infernix::test::cuda_check(cudaMalloc(&y, static_cast<std::size_t>(n) * 16 * 2), "output");
        infernix::test::cuda_check(cudaMemcpy(weight, host.payload.data(), host.payload.size(), cudaMemcpyHostToDevice),
                                   "upload weight");
        infernix::test::cuda_check(cudaMemcpy(x, activation.data(), activation.size() * 2, cudaMemcpyHostToDevice),
                                   "upload activation");
        const Weight w = host.device_weight(weight);
        for (const int t : {1, 2, 5, 16}) {
            Tensor input(x, DType::BF16, {k, t});
            Tensor output(y, DType::BF16, {n, t});
            std::vector<std::uint16_t> first(static_cast<std::size_t>(n) * t), again(first.size());
            int differing = 0;
            for (int repeat = 0; repeat < 64; ++repeat) {
                infernix::test::cuda_check(cudaMemsetAsync(y, 0xFF, first.size() * 2, stream), "poison output");
                infernix::test::cuda_check(cudaMemcpyAsync(noise_b, noise_a, kNoiseBytes, cudaMemcpyDeviceToDevice, other),
                                           "noise copy");
                ops::linear(input, w, output, stream);
                infernix::test::cuda_check(cudaStreamSynchronize(stream), "linear");
                auto& target = repeat == 0 ? first : again;
                infernix::test::cuda_check(cudaMemcpy(target.data(), y, target.size() * 2, cudaMemcpyDeviceToHost),
                                           "read output");
                differing += repeat != 0 && again != first ? 1 : 0;
            }
            if (differing != 0) {
                std::cerr << "MXFP8_A16 [" << n << "," << k << "] T=" << t << ": " << differing
                          << " of 63 repeats differ from the first call\n";
                ++failures;
            }
        }
        cudaFree(weight);
        cudaFree(x);
        cudaFree(y);
    }
    infernix::test::cuda_check(cudaDeviceSynchronize(), "noise");
    cudaFree(noise_a);
    cudaFree(noise_b);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(other);
    return failures;
}

int mxfp8_contract() {
    int failures      = 0;
    const auto packed = make_mxfp8_weight(1280, 2560, 7U);
    const auto expect_invalid = [&](const char* label, const Weight& weight) {
        try {
            (void)ops::detail::validate_mxfp8_weight(weight, "test");
            std::cerr << "MXFP8 weight validation accepted " << label << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    Weight invalid = packed.weight;
    invalid.layout = QuantLayout::RowScale;
    expect_invalid("another layout", invalid);
    invalid           = packed.weight;
    invalid.scale_nb[2] = 16;
    expect_invalid("a scale tile stride", invalid);
    invalid        = packed.weight;
    invalid.scales = static_cast<const std::uint8_t*>(invalid.scales) + 16;
    expect_invalid("a scale plane offset", invalid);
    invalid               = packed.weight;
    invalid.payload_bytes = invalid.payload_bytes - 1;
    expect_invalid("a payload bound", invalid);
    try {
        // Every route is column-invariant, so the policy needs no route of its own.
        (void)ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_G32_E8M0, 1280, 2560,
                                                   ops::LinearPolicy::A16ColumnInvariant, 1, 4096);
    } catch (const std::exception& error) {
        std::cerr << "MXFP8 refused the column-invariant policy: " << error.what() << '\n';
        ++failures;
    }
    try {
        (void)ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_G32_E8M0, 1280, 2048, ops::LinearPolicy::A16Only,
                                                   1, 8);
        std::cerr << "MXFP8 accepted an unregistered shape\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (const auto& [n, k] : kShapes) { failures += verify_workspace_envelopes(QType::FP8_E4M3FN_G32_E8M0, n, k); }
    return failures;
}

} // namespace

int main() {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = mxfp8_contract() + run_flash_next_mxfp8_a16() + mxfp8_a16_column_invariance() +
                             mxfp8_a16_repeatability() + mxfp8_a16_back_to_back() + mxfp8_a16_dirty_shared();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " MXFP8 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "MXFP8 A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
