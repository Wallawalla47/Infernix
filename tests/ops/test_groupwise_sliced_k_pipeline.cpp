// The pipelined groupwise sliced-K kernel (src/ops/linear/common/sliced_k_pipelined.cuh) returns
// the release sliced-K kernels' bits: every route of the temporary A/B switches (pipelined, one-wave)
// and every forced CTA shape (16/32/64 block rows x 1-4 ring slots, where admissible) gives the
// release route's output, bit for bit, on the groupwise 27B's decode and verification projections
// at 1-48 columns, under the column-invariant policy (the reproducible route's decode) and A16Only
// (batched-decode). The q5-ksplit-mma fast path (always pipelined) gives the same bits at every
// shape and is column-invariant. The shape rule (choose_sliced_k_shape) is checked on fixed
// candidates without a GPU. Numerics against the FP64 oracle are the format tests' (linear_add Q5,
// linear_swiglu Q4, attn/gdn input projections); this test covers only bit identity.

#include "ops/op_family.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
#include "ops/linear/common/sliced_k_route.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;
using ops::detail::SlicedKCandidate;
using ops::detail::SlicedKRoute;
using ops::detail::SlicedKShape;

constexpr std::int32_t kWidest = 48;

std::vector<std::int32_t> all_widths() {
    std::vector<std::int32_t> out;
    for (std::int32_t width = 1; width <= kWidest; ++width) { out.push_back(width); }
    return out;
}

// The forced-shape sweep's widths: every capacity boundary of the release ladders.
constexpr std::array<std::int32_t, 12> kShapeWidths{1, 2, 4, 5, 8, 9, 12, 16, 17, 24, 33, 48};

std::vector<std::uint16_t> activation(std::int32_t k, std::int32_t columns, std::uint32_t seed) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(k) * columns);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::uint32_t value = seed ^ static_cast<std::uint32_t>(i) * 0x9e3779b9U;
        value ^= value >> 16;
        value *= 0x7feb352dU;
        value ^= value >> 15;
        out[i] = f32_to_bf16(static_cast<float>(static_cast<int>(value & 0x3ffU) - 512) / 256.0F);
    }
    return out;
}

struct DeviceWeight {
    quantized_weight::PackedWeight host;
    DeviceBuffer device;
    Weight weight;

    DeviceWeight(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed)
        : host(quantized_weight::make_patterned_weight(qtype, n, k, seed, {})),
          device(host.payload.size()) {
        device.copy_from_host(host.payload.data(), host.payload.size());
        weight = host.device_weight(device.p);
    }
};

struct Projection {
    std::string name;
    std::int32_t input_rows;
    std::vector<std::int32_t> output_rows;
    std::function<void(const Tensor& x, std::vector<Tensor>& outputs, WorkspaceArena& workspace)>
        launch;
    bool residual = false;
    // The Op's capacity over 1..kWidest columns (its workspace query).
    std::size_t workspace = 0;
};

// Every output plane's bits after one call of `columns` columns.
std::vector<std::vector<std::uint16_t>> run(const Projection& p, const std::vector<std::uint16_t>& x_host,
                                            std::int32_t columns) {
    DeviceBuffer x(static_cast<std::size_t>(p.input_rows) * columns * 2);
    x.copy_from_host(x_host.data(), x.bytes);
    std::vector<DeviceBuffer> planes;
    std::vector<Tensor> outputs;
    for (const std::int32_t rows : p.output_rows) {
        planes.emplace_back(static_cast<std::size_t>(rows) * columns * 2);
        if (p.residual) {
            const std::vector<std::uint16_t> residual = activation(rows, columns, 0x51U);
            planes.back().copy_from_host(residual.data(), planes.back().bytes);
        }
        outputs.push_back(Tensor(planes.back().p, DType::BF16, {rows, columns}));
    }
    DeviceBuffer scratch(std::max<std::size_t>(p.workspace, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
    p.launch(Tensor(x.p, DType::BF16, {p.input_rows, columns}), outputs, workspace);
    cuda_synchronize();
    std::vector<std::vector<std::uint16_t>> out;
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        out.emplace_back(static_cast<std::size_t>(p.output_rows[plane]) * columns);
        planes[plane].copy_to_host(out.back().data(), planes[plane].bytes);
    }
    return out;
}

std::string describe(std::optional<SlicedKRoute> route, SlicedKShape forced) {
    std::string out = !route ? "release" : *route == SlicedKRoute::Pipelined ? "pipelined" : "one-wave";
    if (forced.block_rows != 0) {
        out += " " + std::to_string(forced.block_rows) + "x" + std::to_string(forced.stages);
    }
    return out;
}

// The release route's bits against every other route at every width, then every forced shape at
// the sweep's widths.
int check(const Projection& p) {
    int failures = 0;
    const std::vector<std::uint16_t> x = activation(p.input_rows, kWidest, 0x1234U);
    const auto compare = [&](std::optional<SlicedKRoute> route, SlicedKShape forced,
                             const std::vector<std::int32_t>& widths) {
        for (const std::int32_t width : widths) {
            ops::detail::set_groupwise_sliced_k_route_for_testing(std::nullopt);
            const auto release = run(p, x, width);
            ops::detail::set_groupwise_sliced_k_route_for_testing(route, forced);
            const auto candidate = run(p, x, width);
            ops::detail::set_groupwise_sliced_k_route_for_testing(std::nullopt);
            if (candidate != release) {
                std::cerr << p.name << ": the " << describe(route, forced)
                          << " route differs from the release route at width " << width << '\n';
                ++failures;
            }
        }
    };
    compare(SlicedKRoute::Pipelined, {}, all_widths());
    compare(SlicedKRoute::OneWave, {}, all_widths());
    const std::vector<std::int32_t> shape_widths(kShapeWidths.begin(), kShapeWidths.end());
    for (const int rows : {16, 32, 64}) {
        for (int stages = 1; stages <= 4; ++stages) {
            compare(SlicedKRoute::Pipelined, {rows, stages}, shape_widths);
        }
    }
    return failures;
}

constexpr auto kInvariant = ops::LinearPolicy::A16ColumnInvariant;
constexpr auto kA16       = ops::LinearPolicy::A16Only;

int check_swiglu() {
    DeviceWeight w(QType::Q4_G64_FP16, 34816, 5120, 71U);
    int failures = 0;
    for (const auto policy : {kInvariant, kA16}) {
        failures += check({std::string("Q4 linear_swiglu [34816,5120] ") +
                               (policy == kInvariant ? "invariant" : "a16"),
                           5120,
                           {17408},
                           [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear_swiglu(x, w.weight, out[0], policy, ws, nullptr);
                           },
                           false,
                           ops::linear_swiglu_workspace_capacity_bytes(QType::Q4_G64_FP16, 34816, 5120, policy, 1,
                                                                       kWidest)});
    }
    return failures;
}

int check_linear_add() {
    int failures = 0;
    for (const std::int32_t k : {6144, 17408}) {
        DeviceWeight w(QType::Q5_G64_FP16, 5120, k, 73U);
        for (const auto policy : {kInvariant, kA16}) {
            failures += check({"Q5 linear_add [5120," + std::to_string(k) + "] " +
                                   (policy == kInvariant ? "invariant" : "a16"),
                               k,
                               {5120},
                               [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                                   ops::linear_add(x, w.weight, out[0], policy, ws, nullptr);
                               },
                               true,
                               ops::linear_add_workspace_capacity_bytes(QType::Q5_G64_FP16, 5120, k, policy, 1,
                                                                        kWidest)});
        }
    }
    return failures;
}

int check_inputs() {
    int failures = 0;
    DeviceWeight qk(QType::Q4_G64_FP16, 7168, 5120, 79U), gv(QType::Q5_G64_FP16, 7168, 5120, 81U);
    DeviceWeight gqk(QType::Q4_G64_FP16, 4096, 5120, 83U), vz(QType::Q5_G64_FP16, 12288, 5120, 85U);
    for (const auto policy : {kInvariant, kA16}) {
        const std::string suffix = policy == kInvariant ? " invariant" : " a16";
        failures += check({"Q4/Q5 attn_input_proj" + suffix, 5120, {6144, 6144, 1024, 1024},
                           [&qk, &gv, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena&) {
                               ops::attn_input_proj(x, qk.weight, gv.weight, out[0], out[1], out[2],
                                                    out[3], policy, nullptr);
                           }});
        failures += check({"Q4/Q5 gdn_input_proj" + suffix, 5120, {10240, 6144},
                           [&gqk, &vz, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena&) {
                               ops::gdn_input_proj(x, gqk.weight, vz.weight, out[0], out[1], policy,
                                                   nullptr);
                           }});
    }
    return failures;
}

// The proposal head (Q4 [131072,5120]) and the Q6 27B's gate/up (Q6 [34816,5120]) on linear.
int check_linear() {
    int failures = 0;
    DeviceWeight head(QType::Q4_G64_FP16, 131072, 5120, 87U);
    failures += check({"Q4 linear [131072,5120] invariant", 5120, {131072},
                       [&head](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                           ops::linear(x, head.weight, out[0], kInvariant, ws, nullptr);
                       },
                       false,
                       ops::linear_workspace_capacity_bytes(QType::Q4_G64_FP16, 131072, 5120, kInvariant, 1, kWidest)});
    DeviceWeight q6(QType::Q6_G64_FP16, 34816, 5120, 89U);
    for (const auto policy : {kInvariant, kA16}) {
        failures += check({std::string("Q6 linear [34816,5120] ") + (policy == kInvariant ? "invariant" : "a16"),
                           5120,
                           {34816},
                           [&q6, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear(x, q6.weight, out[0], policy, ws, nullptr);
                           },
                           false,
                           ops::linear_workspace_capacity_bytes(QType::Q6_G64_FP16, 34816, 5120, policy, 1, kWidest)});
    }
    return failures;
}

// q5-ksplit-mma: every forced shape gives the rule's bits, a column's bits do not depend on the
// call's width under the column-invariant policy, and at 2-16 columns A16Only equals it (one
// reduction).
int check_ksplit() {
    int failures = 0;
    for (const std::int32_t k : {6144, 17408}) {
        DeviceWeight w(QType::Q5_G64_FP16, 5120, k, 73U);
        const std::vector<std::uint16_t> x = activation(k, kWidest, 0x1234U);
        const auto ksplit = [&](ops::LinearPolicy policy) {
            return Projection{"Q5 linear_add [5120," + std::to_string(k) + "] q5-ksplit-mma", k, {5120},
                              [&w, policy](const Tensor& in, std::vector<Tensor>& out, WorkspaceArena& ws) {
                                  ops::linear_add(in, w.weight, out[0], policy, ws, nullptr,
                                                  ops::LinearRoutes::Q5KSplit);
                              },
                              true,
                              ops::linear_add_workspace_capacity_bytes(QType::Q5_G64_FP16, 5120, k, policy, 1, kWidest)};
        };
        const Projection invariant = ksplit(kInvariant);
        const auto widest          = run(invariant, x, kWidest);
        for (const std::int32_t width : all_widths()) {
            const auto narrow = run(invariant, x, width);
            // Column 0 holds the same residual at every width (element-indexed pattern).
            if (!std::equal(narrow[0].begin(), narrow[0].begin() + 5120, widest[0].begin())) {
                std::cerr << invariant.name << ": column 0 changes at width " << width << '\n';
                ++failures;
            }
            if (width >= 2 && width <= 16 && run(ksplit(kA16), x, width) != narrow) {
                std::cerr << invariant.name << ": A16Only differs from the invariant policy at width "
                          << width << '\n';
                ++failures;
            }
        }
        for (const int rows : {16, 32}) {
            for (int stages = 1; stages <= 4; ++stages) {
                for (const std::int32_t width : kShapeWidths) {
                    const auto rule = run(invariant, x, width);
                    ops::detail::set_groupwise_sliced_k_route_for_testing(SlicedKRoute::Pipelined,
                                                                          {rows, stages});
                    const auto forced = run(invariant, x, width);
                    ops::detail::set_groupwise_sliced_k_route_for_testing(std::nullopt);
                    if (forced != rule) {
                        std::cerr << invariant.name << ": shape " << rows << "x" << stages
                                  << " differs at width " << width << '\n';
                        ++failures;
                    }
                }
            }
        }
    }
    return failures;
}

// The shape rule on fixed candidates (170 SMs).
int check_rule() {
    int failures = 0;
    const auto expect = [&](const std::vector<SlicedKCandidate>& candidates, SlicedKShape shape,
                            const char* label) {
        const std::size_t chosen = ops::detail::choose_sliced_k_shape(candidates, 170);
        if (!(candidates[chosen].shape == shape)) {
            std::cerr << "shape rule: " << label << " chose " << candidates[chosen].shape.block_rows << "x"
                      << candidates[chosen].shape.stages << '\n';
            ++failures;
        }
    };
    // Q5 down at 8 columns: 320 CTAs; every depth fits one wave, the deepest carries the most bytes.
    expect({{{16, 1}, 320, 12, 2688}, {{16, 2}, 320, 7, 2688}, {{16, 3}, 320, 4, 2688}, {{16, 4}, 320, 3, 2688}},
           {16, 4}, "one wave, deepest ring");
    // A second wave loses to one wave even with more bytes in flight.
    expect({{{16, 4}, 340, 1, 2688}, {{16, 3}, 340, 2, 2688}}, {16, 3}, "fewest waves first");
    // Equal waves and bytes: fewer rows, then the shallower ring.
    expect({{{32, 2}, 320, 2, 2048}, {{16, 4}, 320, 2, 1024}, {{16, 2}, 320, 2, 2048}}, {16, 2},
           "ties: fewer rows, shallower ring");
    // A candidate with no resident CTA is never chosen.
    expect({{{16, 4}, 320, 0, 2688}, {{16, 1}, 320, 6, 2688}}, {16, 1}, "no resident CTA");
    return failures;
}

} // namespace

INFERNIX_OP_TEST(groupwise_sliced_k_pipeline) {
    try {
        int failures = check_rule();
        if (cuda_unavailable()) {
            std::cout << (failures == 0 ? "SKIP" : "FAIL") << ": no usable CUDA device (shape rule "
                      << (failures == 0 ? "passed" : "failed") << ")\n";
            return failures == 0 ? 77 : 1;
        }
        failures += check_swiglu();
        failures += check_linear_add();
        failures += check_inputs();
        failures += check_linear();
        failures += check_ksplit();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " groupwise sliced-K pipeline\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "groupwise sliced-K pipeline test failed: " << error.what() << '\n';
        return 1;
    }
}
