// LinearPolicy::A16ColumnInvariant: every registered projection gives a column the same output bits
// whatever the call's width and other columns, on its dedicated column-invariant route or, where it
// has none, its exact one-column route once per column. Each projection's oracle tests cover the
// route's numerical criterion; this test covers the invariance itself:
//
// - a column's output is bit-identical at every width from 1 to the widest tested call, through
//   tile boundaries and the slicing of wider calls;
// - it does not change when every other column changes;
// - every column of the widest call equals that column projected alone.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/gdn_gating_proj.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;

constexpr std::int32_t kWidest = 72;
constexpr std::int32_t kWidths[] = {1, 2, 3, 5, 8, 9, 16, 17, 32, 33, 40, 64, 65, kWidest};

// A deterministic BF16 activation matrix [k, columns]; `seed` varies every value.
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

// One projection: its input rows, its output planes' rows (each [rows, T]) and its launch.
struct Projection {
    std::string name;
    std::int32_t input_rows;
    std::vector<std::int32_t> output_rows;
    // Launches the projection on x [input_rows, T] into outputs (each [rows, T]). A residual
    // projection reads and updates outputs[0].
    std::function<void(const Tensor& x, std::vector<Tensor>& outputs, WorkspaceArena& workspace)>
        launch;
    bool residual = false;
};

// Runs the projection on `columns` columns of `x_host` and returns every output plane's bits.
std::vector<std::vector<std::uint16_t>> run(const Projection& p,
                                            const std::vector<std::uint16_t>& x_host,
                                            std::int32_t columns) {
    DeviceBuffer x(static_cast<std::size_t>(p.input_rows) * columns * 2);
    x.copy_from_host(x_host.data(), x.bytes);
    std::vector<DeviceBuffer> planes;
    std::vector<Tensor> outputs;
    for (const std::int32_t rows : p.output_rows) {
        planes.emplace_back(static_cast<std::size_t>(rows) * columns * 2);
        if (p.residual) {
            // The residual input: a fixed per-column pattern.
            const std::vector<std::uint16_t> residual = activation(rows, columns, 0x51U);
            planes.back().copy_from_host(residual.data(), planes.back().bytes);
        }
        outputs.push_back(Tensor(planes.back().p, DType::BF16, {rows, columns}));
    }
    DeviceBuffer scratch(1 << 20);
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

// Column `column` of every plane.
std::vector<std::uint16_t> column_bits(const Projection& p,
                                       const std::vector<std::vector<std::uint16_t>>& planes,
                                       std::int32_t column) {
    std::vector<std::uint16_t> out;
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        const auto rows = static_cast<std::size_t>(p.output_rows[plane]);
        const auto begin = planes[plane].begin() + static_cast<std::ptrdiff_t>(column * rows);
        out.insert(out.end(), begin, begin + static_cast<std::ptrdiff_t>(rows));
    }
    return out;
}

// The residual input pattern is indexed by element, so column 0 holds the same residual at every
// width; a residual projection's other columns are therefore compared only through column 0.
int check(const Projection& p) {
    int failures = 0;
    const std::vector<std::uint16_t> x = activation(p.input_rows, kWidest, 0x1234U);
    const auto widest = run(p, x, kWidest);
    // Column 0 at every width, with the wider calls' other columns present.
    const std::vector<std::uint16_t> first = column_bits(p, widest, 0);
    for (const std::int32_t width : kWidths) {
        if (column_bits(p, run(p, x, width), 0) != first) {
            std::cerr << p.name << ": column 0 changes at width " << width << '\n';
            ++failures;
        }
    }
    // Column 0 with every other column changed.
    std::vector<std::uint16_t> other = activation(p.input_rows, kWidest, 0xbeefU);
    std::copy(x.begin(), x.begin() + p.input_rows, other.begin());
    if (column_bits(p, run(p, other, kWidest), 0) != first) {
        std::cerr << p.name << ": column 0 changes with its neighbours\n";
        ++failures;
    }
    // Every column of the widest call, projected alone.
    if (!p.residual) {
        for (std::int32_t column = 1; column < kWidest; ++column) {
            const std::vector<std::uint16_t> alone(
                x.begin() + static_cast<std::ptrdiff_t>(column) * p.input_rows,
                x.begin() + static_cast<std::ptrdiff_t>(column + 1) * p.input_rows);
            if (column_bits(p, run(p, alone, 1), 0) != column_bits(p, widest, column)) {
                std::cerr << p.name << ": column " << column << " differs from itself alone\n";
                ++failures;
            }
        }
    }
    return failures;
}

struct DeviceWeight {
    // NVFP4 carries global weight and activation divisors.
    static quantized_weight::PatternedWeightOptions options(QType qtype) {
        quantized_weight::PatternedWeightOptions out;
        if (qtype == QType::NVFP4) {
            out.weight_scale_divisor = 0.125F;
            out.input_scale_divisor  = 3.5F;
        }
        return out;
    }

    quantized_weight::PackedWeight host;
    DeviceBuffer device;
    Weight weight;

    DeviceWeight(QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed)
        : host(quantized_weight::make_patterned_weight(qtype, n, k, seed, options(qtype))),
          device(host.payload.size()) {
        device.copy_from_host(host.payload.data(), host.payload.size());
        weight = host.device_weight(device.p);
    }
};

constexpr auto kInvariant = ops::LinearPolicy::A16ColumnInvariant;

int check_formats(QType qtype, const char* format) {
    int failures = 0;
    {
        DeviceWeight w(qtype, 34816, 5120, 71U);
        failures += check({std::string(format) + " linear_swiglu [34816,5120]", 5120, {17408},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear_swiglu(x, w.weight, out[0], kInvariant, ws, nullptr);
                           }});
    }
    for (const std::int32_t k : {6144, 17408}) {
        DeviceWeight w(qtype, 5120, k, 73U);
        failures += check({std::string(format) + " linear_add [5120," + std::to_string(k) + "]", k,
                           {5120},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear_add(x, w.weight, out[0], kInvariant, ws, nullptr);
                           },
                           true});
    }
    return failures;
}

int check_inputs(QType qtype, const char* format) {
    int failures = 0;
    {
        DeviceWeight w(qtype, 14336, 5120, 79U);
        failures += check({std::string(format) + " attn_input_proj [14336,5120]", 5120,
                           {6144, 6144, 1024, 1024},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::attn_input_proj(x, w.weight, out[0], out[1], out[2], out[3],
                                                    kInvariant, ws, nullptr);
                           }});
    }
    {
        DeviceWeight w(qtype, 16384, 5120, 83U);
        failures += check({std::string(format) + " gdn_input_proj [16384,5120]", 5120,
                           {10240, 6144},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::gdn_input_proj(x, w.weight, out[0], out[1], kInvariant, ws,
                                                   nullptr);
                           }});
    }
    return failures;
}

// The GDN input norm and control: h [hidden], g and beta [heads] FP32 each (compared as 2*heads
// BF16-sized words per column). The 27B: 48 heads over 5120; the 35B-A3B: 32 heads over 2048.
int check_gating(std::int32_t heads, std::int32_t hidden) {
    const std::vector<std::uint16_t> weight_bits = activation(hidden, 2 * heads, 97U);
    const std::vector<std::uint16_t> norm_bits   = activation(hidden, 1, 101U);
    DeviceBuffer weight_buffer(weight_bits.size() * 2), norm_buffer(norm_bits.size() * 2);
    weight_buffer.copy_from_host(weight_bits.data(), weight_buffer.bytes);
    norm_buffer.copy_from_host(norm_bits.data(), norm_buffer.bytes);
    const std::vector<float> a_log(heads, -0.5F), dt_bias(heads, 0.25F);
    DeviceBuffer a_log_buffer(heads * 4), dt_bias_buffer(heads * 4);
    a_log_buffer.copy_from_host(a_log.data(), a_log_buffer.bytes);
    dt_bias_buffer.copy_from_host(dt_bias.data(), dt_bias_buffer.bytes);
    Weight weight{};
    weight.qtype         = QType::BF16;
    weight.layout        = QuantLayout::Contiguous;
    weight.payload       = weight_buffer.p;
    weight.payload_bytes = weight_bits.size() * 2;
    weight.qdata         = weight_buffer.p;
    weight.ndim          = 2;
    weight.shape[0]      = 2 * heads;
    weight.shape[1]      = hidden;
    weight.padded_shape[0] = 2 * heads;
    weight.padded_shape[1] = hidden;
    weight.n             = 2 * heads;
    weight.k             = hidden;
    int device = 0, multiprocessors = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const DeviceExecutionView execution{nullptr, multiprocessors};
    const Tensor norm(norm_buffer.p, DType::BF16, {hidden});
    const Tensor a(a_log_buffer.p, DType::FP32, {heads}), dt(dt_bias_buffer.p, DType::FP32, {heads});
    return check({"BF16 gdn_norm_gating_proj [" + std::to_string(2 * heads) + "," + std::to_string(hidden) + "]",
                  hidden, {hidden, 2 * heads, 2 * heads},
                  [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                      Tensor g(out[1].data, DType::FP32, {heads, x.ne[1]});
                      Tensor beta(out[2].data, DType::FP32, {heads, x.ne[1]});
                      ops::gdn_norm_gating_proj(x, norm, 1.0e-6F, weight, a, dt, kInvariant, ws,
                                                out[0], g, beta,
                                                execution);
                  }});
}

// The groupwise 27B's MLP gate/up (Q4 [34816,5120]): the invariant route, and at the widths the
// exact route already serves with the same small-T family (2..32) the exact route's own bits.
int check_q4_swiglu() {
    DeviceWeight w(QType::Q4_G64_FP16, 34816, 5120, 71U);
    const auto swiglu = [&](ops::LinearPolicy policy) {
        return Projection{"Q4 linear_swiglu [34816,5120]", 5120, {17408},
                          [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                              ops::linear_swiglu(x, w.weight, out[0], policy, ws, nullptr);
                          }};
    };
    int failures = check(swiglu(kInvariant));
    const std::vector<std::uint16_t> x = activation(5120, 32, 0x2345U);
    for (const std::int32_t width : {2, 9, 17, 32}) {
        if (run(swiglu(kInvariant), x, width) != run(swiglu(ops::LinearPolicy::A16Only), x, width)) {
            std::cerr << "Q4 linear_swiglu: the invariant route differs from the exact route at width " << width
                      << '\n';
            ++failures;
        }
    }
    return failures;
}

// The groupwise 27B's output projections (Q5 linear_add [5120,6144] and [5120,17408]): the invariant
// route, and at the widths the exact route already serves with four K warps and the linear
// reduction (5..32, 49..64) the exact route's own bits.
int check_q5_linear_add() {
    int failures = 0;
    for (const std::int32_t k : {6144, 17408}) {
        DeviceWeight w(QType::Q5_G64_FP16, 5120, k, 73U);
        const std::string name = "Q5 linear_add [5120," + std::to_string(k) + "]";
        const auto linear_add = [&](ops::LinearPolicy policy) {
            return Projection{name, k, {5120},
                              [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                                  ops::linear_add(x, w.weight, out[0], policy, ws, nullptr);
                              },
                              true};
        };
        failures += check(linear_add(kInvariant));
        const std::vector<std::uint16_t> x = activation(k, 64, 0x2345U);
        for (const std::int32_t width : {5, 9, 17, 25, 32, 49, 64}) {
            if (run(linear_add(kInvariant), x, width) != run(linear_add(ops::LinearPolicy::A16Only), x, width)) {
                std::cerr << name << ": the invariant route differs from the exact route at width " << width << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

// The groupwise 27B's paired input projections (Q4 + Q5 parents): attention [7168,5120] each, GDN
// Q4 [4096,5120] and Q5 [12288,5120]. At 7..12 columns the exact route's Q4 parent already runs the
// invariant schedule, so q/k (attention) and the qk rows (GDN) keep the exact route's bits there.
int check_paired() {
    int failures = 0;
    {
        DeviceWeight qk(QType::Q4_G64_FP16, 7168, 5120, 79U), gv(QType::Q5_G64_FP16, 7168, 5120, 81U);
        const auto attention = [&](ops::LinearPolicy policy) {
            return Projection{"Q4/Q5 attn_input_proj", 5120, {6144, 6144, 1024, 1024},
                              [&qk, &gv, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena&) {
                                  ops::attn_input_proj(x, qk.weight, gv.weight, out[0], out[1], out[2], out[3],
                                                       policy, nullptr);
                              }};
        };
        failures += check(attention(kInvariant));
        const std::vector<std::uint16_t> x = activation(5120, 12, 0x2345U);
        for (const std::int32_t width : {7, 12}) {
            const auto invariant = run(attention(kInvariant), x, width);
            const auto exact     = run(attention(ops::LinearPolicy::A16Only), x, width);
            if (invariant[0] != exact[0] || invariant[2] != exact[2]) {
                std::cerr << "Q4/Q5 attn_input_proj: invariant q/k differ from the exact route at width " << width
                          << '\n';
                ++failures;
            }
        }
    }
    {
        DeviceWeight qk(QType::Q4_G64_FP16, 4096, 5120, 83U), vz(QType::Q5_G64_FP16, 12288, 5120, 85U);
        const auto gdn = [&](ops::LinearPolicy policy) {
            return Projection{"Q4/Q5 gdn_input_proj", 5120, {10240, 6144},
                              [&qk, &vz, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena&) {
                                  ops::gdn_input_proj(x, qk.weight, vz.weight, out[0], out[1], policy, nullptr);
                              }};
        };
        failures += check(gdn(kInvariant));
        const std::vector<std::uint16_t> x = activation(5120, 12, 0x2345U);
        for (const std::int32_t width : {7, 12}) {
            const auto invariant = run(gdn(kInvariant), x, width);
            const auto exact     = run(gdn(ops::LinearPolicy::A16Only), x, width);
            for (std::int32_t column = 0; column < width; ++column) {
                const auto begin = static_cast<std::ptrdiff_t>(column) * 10240;
                if (!std::equal(invariant[0].begin() + begin, invariant[0].begin() + begin + 4096,
                                exact[0].begin() + begin)) {
                    std::cerr << "Q4/Q5 gdn_input_proj: invariant qk rows differ from the exact route at width "
                              << width << '\n';
                    ++failures;
                    break;
                }
            }
        }
    }
    return failures;
}

// The groupwise 27B's GDN input projection with its convolution (gdn_input_proj_conv_snapshot):
// W one-column calls (decode), each continuing from the previous call's snapshot, give the
// outputs and snapshots of one W-column call (verification), bit for bit.
int check_gdn_conv() {
    constexpr std::int32_t kHidden = 5120, kQuery = 2048, kKey = 2048, kValue = 6144, kZ = 6144;
    constexpr std::int32_t kChannels = kQuery + kKey + kValue;
    DeviceWeight qk(QType::Q4_G64_FP16, 4096, kHidden, 83U), vz(QType::Q5_G64_FP16, 12288, kHidden, 85U);
    const std::vector<std::uint16_t> conv_bits = activation(kChannels, 4, 0x77U);
    DeviceBuffer conv_buffer(conv_bits.size() * 2);
    conv_buffer.copy_from_host(conv_bits.data(), conv_buffer.bytes);
    const Tensor conv(conv_buffer.p, DType::BF16, {kChannels, 4});
    int failures = 0;
    for (const std::int32_t width : {9, 33}) {
        const std::int32_t slots = width + 1;
        const std::vector<std::uint16_t> x_bits = activation(kHidden, width, 0x1234U);
        std::vector<std::uint16_t> initial_state(static_cast<std::size_t>(kChannels) * 3 * slots, 0);
        const std::vector<std::uint16_t> history = activation(kChannels * 3, 1, 0x99U);
        std::copy(history.begin(), history.end(), initial_state.begin());
        const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            kQuery, kKey, kValue, kInvariant, 1, 1, width);
        DeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 256));
        // One call of `columns` columns from x column `first`, reading history slot `initial` and
        // writing slots base..base+columns-1; returns q, k, v, z.
        const auto call = [&](DeviceBuffer& state, std::int32_t first, std::int32_t columns,
                              std::int32_t initial, std::int32_t base) {
            DeviceBuffer x(static_cast<std::size_t>(kHidden) * columns * 2);
            x.copy_from_host(x_bits.data() + static_cast<std::size_t>(first) * kHidden, x.bytes);
            DeviceBuffer selectors(8);
            const std::int32_t selector_values[2]{initial, base};
            selectors.copy_from_host(selector_values, 8);
            std::vector<DeviceBuffer> planes;
            for (const std::int32_t rows : {kQuery, kKey, kValue, kZ}) {
                planes.emplace_back(static_cast<std::size_t>(rows) * columns * 2);
            }
            Tensor q(planes[0].p, DType::BF16, {kQuery, columns}), k(planes[1].p, DType::BF16, {kKey, columns});
            Tensor v(planes[2].p, DType::BF16, {kValue, columns}), z(planes[3].p, DType::BF16, {kZ, columns});
            Tensor states(state.p, DType::BF16, {kChannels, 3, slots});
            const Tensor initial_slot(selectors.p, DType::I32, {1});
            const Tensor base_slot(static_cast<std::int32_t*>(selectors.p) + 1, DType::I32, {1});
            WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
            ops::gdn_input_proj_conv_snapshot(Tensor(x.p, DType::BF16, {kHidden, columns}), qk.weight, vz.weight,
                                              conv, states, Tensor{}, initial_slot, base_slot, q, k, v, z,
                                              kInvariant, workspace, nullptr);
            cuda_synchronize();
            std::vector<std::vector<std::uint16_t>> out;
            for (DeviceBuffer& plane : planes) {
                out.emplace_back(plane.bytes / 2);
                plane.copy_to_host(out.back().data(), plane.bytes);
            }
            return out;
        };
        DeviceBuffer wide_state(initial_state.size() * 2), step_state(initial_state.size() * 2);
        wide_state.copy_from_host(initial_state.data(), wide_state.bytes);
        step_state.copy_from_host(initial_state.data(), step_state.bytes);
        const auto wide = call(wide_state, 0, width, 0, 1);
        bool outputs_equal = true;
        for (std::int32_t column = 0; column < width && outputs_equal; ++column) {
            const auto step = call(step_state, column, 1, column, column + 1);
            for (std::size_t plane = 0; plane < step.size(); ++plane) {
                const std::size_t rows = step[plane].size();
                if (!std::equal(step[plane].begin(), step[plane].end(), wide[plane].begin() + column * rows)) {
                    std::cerr << "Q4/Q5 gdn_input_proj_conv_snapshot W=" << width << ": column " << column
                              << " of output " << plane << " differs from the one-column call\n";
                    outputs_equal = false;
                    ++failures;
                    break;
                }
            }
        }
        std::vector<std::uint16_t> wide_bits(initial_state.size()), step_bits(initial_state.size());
        wide_state.copy_to_host(wide_bits.data(), wide_state.bytes);
        step_state.copy_to_host(step_bits.data(), step_state.bytes);
        if (outputs_equal && wide_bits != step_bits) {
            std::cerr << "Q4/Q5 gdn_input_proj_conv_snapshot W=" << width
                      << ": snapshots differ from the one-column calls'\n";
            ++failures;
        }
    }
    return failures;
}

// The 35B-A3B's Q8 projections: attention input [9216,2048], GDN input [12288,2048] and the
// output linear_add [2048,4096]; at the widths the exact route already serves with the invariant
// schedule (2..48, 2..32 and 2..12) the exact route's own bits.
int check_q8_35b() {
    int failures = 0;
    const auto same_as_exact = [&](const auto& make, std::int32_t input_rows,
                                   std::initializer_list<std::int32_t> widths) {
        const std::vector<std::uint16_t> x = activation(input_rows, 48, 0x2345U);
        for (const std::int32_t width : widths) {
            if (run(make(kInvariant), x, width) != run(make(ops::LinearPolicy::A16Only), x, width)) {
                std::cerr << make(kInvariant).name << ": the invariant route differs from the exact route at width "
                          << width << '\n';
                ++failures;
            }
        }
    };
    {
        DeviceWeight w(QType::Q8_G32_FP16, 9216, 2048, 91U);
        const auto attention = [&](ops::LinearPolicy policy) {
            return Projection{"Q8 attn_input_proj [9216,2048]", 2048, {4096, 4096, 512, 512},
                              [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                                  ops::attn_input_proj(x, w.weight, out[0], out[1], out[2], out[3], policy, ws,
                                                       nullptr);
                              }};
        };
        failures += check(attention(kInvariant));
        same_as_exact(attention, 2048, {2, 9, 17, 48});
    }
    {
        DeviceWeight w(QType::Q8_G32_FP16, 12288, 2048, 93U);
        const auto gdn = [&](ops::LinearPolicy policy) {
            return Projection{"Q8 gdn_input_proj [12288,2048]", 2048, {8192, 4096},
                              [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                                  ops::gdn_input_proj(x, w.weight, out[0], out[1], policy, ws, nullptr);
                              }};
        };
        failures += check(gdn(kInvariant));
        same_as_exact(gdn, 2048, {2, 9, 17, 32});
    }
    {
        DeviceWeight w(QType::Q8_G32_FP16, 2048, 4096, 95U);
        const auto linear_add = [&](ops::LinearPolicy policy) {
            return Projection{"Q8 linear_add [2048,4096]", 4096, {2048},
                              [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                                  ops::linear_add(x, w.weight, out[0], policy, ws, nullptr);
                              },
                              true};
        };
        failures += check(linear_add(kInvariant));
        same_as_exact(linear_add, 4096, {2, 5, 12});
    }
    return failures;
}

// The 122B-A10B's Q8 MTP attention input [17408,3072]: invariant at every width, and at the widths
// the exact route serves with the invariant schedule (2..48) the exact route's own bits.
int check_q8_122b() {
    int failures = 0;
    DeviceWeight w(QType::Q8_G32_FP16, 17408, 3072, 97U);
    const auto attention = [&](ops::LinearPolicy policy) {
        return Projection{"Q8 attn_input_proj [17408,3072]", 3072, {8192, 8192, 512, 512},
                          [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                              ops::attn_input_proj(x, w.weight, out[0], out[1], out[2], out[3], policy, ws,
                                                   nullptr);
                          }};
    };
    failures += check(attention(kInvariant));
    const std::vector<std::uint16_t> x = activation(3072, 48, 0x3456U);
    for (const std::int32_t width : {2, 9, 17, 48}) {
        if (run(attention(kInvariant), x, width) != run(attention(ops::LinearPolicy::A16Only), x, width)) {
            std::cerr << "Q8 attn_input_proj [17408,3072]: the invariant route differs from the exact route at width "
                      << width << '\n';
            ++failures;
        }
    }
    return failures;
}

// A contiguous BF16 weight [n,k] of patterned values.
struct Bf16DeviceWeight {
    DeviceBuffer device;
    Weight weight{};

    Bf16DeviceWeight(std::int32_t n, std::int32_t k, std::uint32_t seed)
        : device(static_cast<std::size_t>(n) * k * 2) {
        const std::vector<std::uint16_t> bits = activation(k, n, seed);
        device.copy_from_host(bits.data(), device.bytes);
        weight.qtype           = QType::BF16;
        weight.layout          = QuantLayout::Contiguous;
        weight.payload         = device.p;
        weight.payload_bytes   = device.bytes;
        weight.qdata           = device.p;
        weight.ndim            = 2;
        weight.shape[0]        = n;
        weight.shape[1]        = k;
        weight.padded_shape[0] = n;
        weight.padded_shape[1] = k;
        weight.n               = n;
        weight.k               = k;
    }
};

// The official Qwen3.6-27B NVFP4 artifact's BF16 projections (its first 24 layers' attention input
// and the [5120,6144] output projection).
int check_bf16() {
    int failures = 0;
    {
        Bf16DeviceWeight w(14336, 5120, 101U);
        failures += check({"BF16 attn_input_proj [14336,5120]", 5120, {6144, 6144, 1024, 1024},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::attn_input_proj(x, w.weight, out[0], out[1], out[2], out[3],
                                                    kInvariant, ws, nullptr);
                           }});
    }
    {
        Bf16DeviceWeight w(5120, 6144, 103U);
        failures += check({"BF16 linear_add [5120,6144]", 6144, {5120},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear_add(x, w.weight, out[0], kInvariant, ws, nullptr);
                           },
                           true});
    }
    return failures;
}

// Problems with no dedicated column-invariant route run their exact one-column route once per
// column under the policy: a Q8 MLP gate/up and Q4 output projection (user recipes).
int check_column_by_column() {
    int failures = 0;
    if (ops::linear_swiglu_column_invariant_route(QType::Q8_G32_FP16, 34816, 5120) ||
        ops::linear_add_column_invariant_route(QType::Q4_G64_FP16, 5120, 6144)) {
        std::cerr << "column-by-column: a probe problem gained a dedicated route; pick another\n";
        ++failures;
    }
    {
        DeviceWeight w(QType::Q8_G32_FP16, 34816, 5120, 71U);
        failures += check({"Q8 linear_swiglu [34816,5120] column by column", 5120, {17408},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear_swiglu(x, w.weight, out[0], kInvariant, ws, nullptr);
                           }});
    }
    {
        DeviceWeight w(QType::Q4_G64_FP16, 5120, 6144, 73U);
        failures += check({"Q4 linear_add [5120,6144] column by column", 6144, {5120},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear_add(x, w.weight, out[0], kInvariant, ws, nullptr);
                           },
                           true});
    }
    return failures;
}

// Q8 linear without a tuned column-invariant family (every problem but the vocabulary head): the
// runtime-shape SIMT pair, on the official drafters' shapes (MTP input, output and attention input
// of the 35B-A3B and the 27B) and one no tuned entry covers.
int check_q8_generic() {
    int failures = 0;
    struct Shape {
        std::int32_t n, k;
    };
    for (const Shape shape : {Shape{2048, 4096}, Shape{5120, 10240}, Shape{14336, 5120}, Shape{1032, 1536}}) {
        if (!ops::linear_column_invariant_route(QType::Q8_G32_FP16, shape.n, shape.k)) {
            std::cerr << "Q8 linear [" << shape.n << "," << shape.k << "]: no column-invariant route\n";
            ++failures;
            continue;
        }
        DeviceWeight w(QType::Q8_G32_FP16, shape.n, shape.k, 97U);
        failures += check({"Q8 linear [" + std::to_string(shape.n) + "," + std::to_string(shape.k) + "]", shape.k,
                           {shape.n},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear(x, w.weight, out[0], kInvariant, ws, nullptr);
                           }});
    }
    return failures;
}

// The official 27B's, 122B-A10B's and 35B-A3B's Q4 proposal heads.
int check_q4_proposal() {
    int failures = 0;
    for (const std::int32_t k : {5120, 3072, 2048}) {
        DeviceWeight w(QType::Q4_G64_FP16, 131072, k, 83U);
        failures += check({"Q4 linear [131072," + std::to_string(k) + "]", k, {131072},
                           [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                               ops::linear(x, w.weight, out[0], kInvariant, ws, nullptr);
                           }});
    }
    return failures;
}

// The Q6 projections of decode width: the 27B and 35B heads and the Q6 27B's fused gate/up; at the
// widths the exact route already serves with four K warps (3..32 at K=5120, 2..32 at K=2048) the
// exact route's own bits.
int check_q6() {
    int failures = 0;
    struct Shape {
        std::int32_t n, k, first_shared;
    };
    for (const Shape shape : {Shape{248320, 5120, 3}, Shape{34816, 5120, 3}, Shape{248320, 2048, 2}}) {
        DeviceWeight w(QType::Q6_G64_FP16, shape.n, shape.k, 89U);
        const std::string name = "Q6 linear [" + std::to_string(shape.n) + "," + std::to_string(shape.k) + "]";
        const auto linear = [&](ops::LinearPolicy policy) {
            return Projection{name, shape.k, {shape.n},
                              [&w, policy](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                                  ops::linear(x, w.weight, out[0], policy, ws, nullptr);
                              }};
        };
        failures += check(linear(kInvariant));
        const std::vector<std::uint16_t> x = activation(shape.k, 32, 0x2345U);
        for (const std::int32_t width : {shape.first_shared, 9, 17, 32}) {
            if (run(linear(kInvariant), x, width) != run(linear(ops::LinearPolicy::A16Only), x, width)) {
                std::cerr << name << ": the invariant route differs from the exact route at width " << width << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

// The vocabulary head: FP8 and Q8.
int check_head(QType qtype, const char* format) {
    DeviceWeight w(qtype, 248320, 5120, 89U);
    return check({std::string(format) + " linear [248320,5120]", 5120, {248320},
                  [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                      ops::linear(x, w.weight, out[0], kInvariant, ws, nullptr);
                  }});
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = check_formats(QType::NVFP4, "NVFP4");
        failures += check_formats(QType::FP8_E4M3FN_ROW_BF16, "FP8");
        failures += check_inputs(QType::NVFP4, "NVFP4");
        failures += check_inputs(QType::FP8_E4M3FN_ROW_BF16, "FP8");
        failures += check_head(QType::FP8_E4M3FN_ROW_BF16, "FP8");
        failures += check_head(QType::Q8_G32_FP16, "Q8");
        failures += check_gating(48, 5120);
        failures += check_gating(32, 2048);
        failures += check_q4_swiglu();
        failures += check_q5_linear_add();
        failures += check_q6();
        failures += check_paired();
        failures += check_gdn_conv();
        failures += check_q8_35b();
        failures += check_q8_122b();
        failures += check_bf16();
        failures += check_q8_generic();
        failures += check_q4_proposal();
        failures += check_column_by_column();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " column-invariant projections\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "column-invariant projection test failed: " << error.what() << '\n';
        return 1;
    }
}
