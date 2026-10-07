#include "infernix/ops/rows.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

std::vector<std::uint16_t> bit_pattern(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> values(count);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state     = state * 1664525u + 1013904223u;
        values[i] = static_cast<std::uint16_t>((state >> 16) ^ static_cast<std::uint32_t>(i));
    }
    return values;
}

// split_rows of a [sum(widths), T] matrix into outputs of `widths` rows, each placed `shift`
// elements into its buffer (a nonzero shift misaligns the outputs: the element-wise route).
int split_case(const std::vector<std::int32_t>& widths, std::int32_t tokens, std::int32_t shift) {
    std::int32_t rows = 0;
    for (const auto w : widths) { rows += w; }
    const auto input = bit_pattern(static_cast<std::size_t>(rows) * tokens, 0x2468'1357u + rows + tokens);
    GuardedDeviceBuffer device_input(input.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
    std::vector<GuardedDeviceBuffer> device_outputs;
    device_outputs.reserve(widths.size());
    std::vector<Tensor> tensors;
    tensors.reserve(widths.size());
    for (const auto w : widths) {
        device_outputs.emplace_back((static_cast<std::size_t>(w) * tokens + shift) * sizeof(std::uint16_t));
        device_outputs.back().fill(0xcd);
        tensors.push_back(Tensor(static_cast<std::uint16_t*>(device_outputs.back().data()) + shift, DType::BF16,
                                 {w, tokens}));
    }
    std::vector<Tensor*> outputs;
    for (auto& t : tensors) { outputs.push_back(&t); }
    ops::split_rows(Tensor(device_input.data(), DType::BF16, {rows, tokens}), outputs, nullptr);
    cuda_synchronize();

    std::string label = "split_rows T=" + std::to_string(tokens) + " shift=" + std::to_string(shift) + " widths";
    for (const auto w : widths) { label += " " + std::to_string(w); }
    int failures    = 0;
    std::int32_t at = 0;
    for (std::size_t i = 0; i < widths.size(); ++i) {
        const std::int32_t w = widths[i];
        std::vector<std::uint16_t> expected(static_cast<std::size_t>(w) * tokens + shift, 0xcdcd);
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t r = 0; r < w; ++r) {
                expected[static_cast<std::size_t>(shift) + static_cast<std::size_t>(t) * w + r] =
                    input[static_cast<std::size_t>(t) * rows + at + r];
            }
        }
        at += w;
        const std::string part = label + " output " + std::to_string(i);
        failures += verify_exact(part.c_str(), from_device<std::uint16_t>(device_outputs[i].data(), expected.size()),
                                 expected);
        failures += device_outputs[i].verify_guards(part.c_str());
    }
    failures += verify_exact((label + " preserves input").c_str(),
                             from_device<std::uint16_t>(device_input.data(), input.size()), input);
    failures += device_input.verify_guards((label + " input").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // Qwen4Exp's QSA projection split (q, gate, k, v, index q, index k) and GDN qkv/z split, at a
    // decode width, a verification width and a prefill chunk; the 16-byte route.
    for (const std::int32_t tokens : {1, 5, 4096}) {
        failures += split_case({6144, 6144, 512, 512, 512, 128}, tokens, 0);
        failures += split_case({8192, 4096}, tokens, 0);
    }
    // The element-wise route: a boundary off the eight-row grid, an odd total, misaligned outputs.
    failures += split_case({12, 20, 8}, 33, 0);
    failures += split_case({7, 9}, 17, 0);
    failures += split_case({64, 64}, 9, 1);
    failures += split_case({5}, 3, 0);
    std::cout << (failures ? "FAIL" : "OK") << " split_rows\n";
    return failures ? 1 : 0;
}
