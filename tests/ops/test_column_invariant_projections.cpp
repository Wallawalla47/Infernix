// LinearPolicy::A16ColumnInvariant: every registered projection's column-invariant route gives a
// column the same output bits whatever the call's width and other columns. Each projection's
// oracle tests cover the route's numerical criterion; this test covers the invariance itself:
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

#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
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

// The 27B GDN input norm and control: h [5120], g and beta [48] FP32 each (compared as 96 BF16-sized
// words per column).
int check_gating() {
    const std::vector<std::uint16_t> weight_bits = activation(5120, 96, 97U);
    const std::vector<std::uint16_t> norm_bits   = activation(5120, 1, 101U);
    DeviceBuffer weight_buffer(weight_bits.size() * 2), norm_buffer(norm_bits.size() * 2);
    weight_buffer.copy_from_host(weight_bits.data(), weight_buffer.bytes);
    norm_buffer.copy_from_host(norm_bits.data(), norm_buffer.bytes);
    const std::vector<float> a_log(48, -0.5F), dt_bias(48, 0.25F);
    DeviceBuffer a_log_buffer(48 * 4), dt_bias_buffer(48 * 4);
    a_log_buffer.copy_from_host(a_log.data(), a_log_buffer.bytes);
    dt_bias_buffer.copy_from_host(dt_bias.data(), dt_bias_buffer.bytes);
    Weight weight{};
    weight.qtype         = QType::BF16;
    weight.layout        = QuantLayout::Contiguous;
    weight.payload       = weight_buffer.p;
    weight.payload_bytes = weight_bits.size() * 2;
    weight.qdata         = weight_buffer.p;
    weight.ndim          = 2;
    weight.shape[0]      = 96;
    weight.shape[1]      = 5120;
    weight.padded_shape[0] = 96;
    weight.padded_shape[1] = 5120;
    weight.n             = 96;
    weight.k             = 5120;
    int device = 0, multiprocessors = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const DeviceExecutionView execution{nullptr, multiprocessors};
    const Tensor norm(norm_buffer.p, DType::BF16, {5120});
    const Tensor a(a_log_buffer.p, DType::FP32, {48}), dt(dt_bias_buffer.p, DType::FP32, {48});
    return check({"BF16 gdn_norm_gating_proj [96,5120]", 5120, {5120, 96, 96},
                  [&](const Tensor& x, std::vector<Tensor>& out, WorkspaceArena& ws) {
                      Tensor g(out[1].data, DType::FP32, {48, x.ne[1]});
                      Tensor beta(out[2].data, DType::FP32, {48, x.ne[1]});
                      ops::gdn_norm_gating_proj(x, norm, 1.0e-6F, weight, a, dt, kInvariant, ws,
                                                out[0], g, beta,
                                                execution);
                  }});
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
        failures += check_gating();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " column-invariant projections\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "column-invariant projection test failed: " << error.what() << '\n';
        return 1;
    }
}
