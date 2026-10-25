// The slice-major ring tile (ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh), which serves the
// column-invariant NVFP4 A16 projections from 17 columns (32 for the [5120,17408] down projection), at the
// NVFP4 27B's MLP shapes (gate/up linear_swiglu [34816,5120], down linear_add [5120,17408]) and the
// [5120,6144] output projection:
//
// - bit for bit the sliced-K tiles it replaces (the reproducible route's bits), at widths through
//   its tile edges, partial token tiles and multi-tile calls;
// - the same bits on a repeated call and after every SM's shared memory was filled with NaN
//   patterns (the activation rows past a call's tokens are never staged);
// - within the A16 criterion of an FP64 oracle over the independently decoded weights (sampled
//   output rows, every column).

#include "ops/op_family.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/dirty_shared_memory.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;
namespace qw = infernix::test::quantized_weight;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kA16Tolerance{kBf16UnitRoundoff, kBf16UnitRoundoff,
                                           2.0 * kBf16UnitRoundoff};
constexpr std::int32_t kWidths[] = {17, 24, 31, 32, 33, 48, 63, 64, 65, 96, 127, 128, 129, 200};
constexpr std::int32_t kWidest   = 200;

// BF16 activations [k, columns] with magnitudes and signs varying by row and column.
std::vector<std::uint16_t> activation(std::int32_t k, std::int32_t columns, std::uint32_t seed) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(k) * columns);
    for (std::size_t i = 0; i < out.size(); ++i) {
        std::uint32_t value = seed ^ static_cast<std::uint32_t>(i) * 0x9e3779b9U;
        value ^= value >> 16;
        value *= 0x7feb352dU;
        value ^= value >> 15;
        const float magnitude = std::ldexp(1.0F, static_cast<int>((value >> 10) & 7U) - 4);
        out[i] = f32_to_bf16(static_cast<float>(static_cast<int>(value & 0x3ffU) - 512) / 512.0F *
                             magnitude);
    }
    return out;
}

struct DeviceWeight {
    qw::PackedWeight host;
    DeviceBuffer device;
    Weight weight;

    DeviceWeight(std::int32_t n, std::int32_t k, std::uint32_t seed)
        : host(qw::make_patterned_weight(
              QType::NVFP4, n, k, seed,
              {.weight_scale_divisor = 0.125F, .input_scale_divisor = 3.5F})),
          device(host.payload.size()) {
        device.copy_from_host(host.payload.data(), host.payload.size());
        weight = host.device_weight(device.p);
    }
};

using Launch = std::function<void(const Tensor& x, Tensor& out)>;

// One call over the first `columns` columns of x_host; `out_init` (if not empty) is the output's
// initial contents (the residual of linear_add). Returns the output bits.
std::vector<std::uint16_t> run(const Launch& launch, const std::vector<std::uint16_t>& x_host,
                               std::int32_t k, std::int32_t rows, std::int32_t columns,
                               const std::vector<std::uint16_t>& out_init, bool dirty) {
    DeviceBuffer x(static_cast<std::size_t>(k) * columns * 2);
    x.copy_from_host(x_host.data(), x.bytes);
    DeviceBuffer y(static_cast<std::size_t>(rows) * columns * 2);
    if (!out_init.empty()) {
        y.copy_from_host(out_init.data(), y.bytes);
    } else {
        y.fill(0xff);
    }
    if (dirty) { dirty_shared_memory(0x7fc17fc1U, nullptr); }
    Tensor input(x.p, DType::BF16, {k, columns});
    Tensor output(y.p, DType::BF16, {rows, columns});
    launch(input, output);
    cuda_synchronize();
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * columns);
    y.copy_to_host(out.data(), y.bytes);
    return out;
}

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    return {0, 1, 7, 8, 63, 64, 127, 128, rows / 3, rows / 2 + 5, rows - 65, rows - 2, rows - 1};
}

// The ring tile against the sliced-K tiles (bits), itself (repeat, dirty shared memory) and the
// FP64 oracle `reference(row, column)` over sampled rows.
int check(const std::string& name, const Launch& sliced, const Launch& ring, std::int32_t k,
          std::int32_t rows, bool residual,
          const std::function<double(std::int32_t, const std::vector<std::uint16_t>&,
                                     const std::vector<std::uint16_t>&, std::int32_t,
                                     std::int32_t)>& reference) {
    int failures = 0;
    const std::vector<std::uint16_t> x = activation(k, kWidest, 0x1234U);
    for (const std::int32_t width : kWidths) {
        const std::vector<std::uint16_t> init =
            residual ? activation(rows, width, 0x51U) : std::vector<std::uint16_t>{};
        const auto expected = run(sliced, x, k, rows, width, init, false);
        const auto got      = run(ring, x, k, rows, width, init, false);
        if (got != expected) {
            std::size_t first = 0;
            while (first < got.size() && got[first] == expected[first]) { ++first; }
            std::cerr << name << " T=" << width << ": ring tile differs from the sliced-K tiles at row "
                      << first % rows << " column " << first / rows << " (" << got[first] << " vs "
                      << expected[first] << ")\n";
            ++failures;
        }
        if (run(ring, x, k, rows, width, init, true) != got) {
            std::cerr << name << " T=" << width << ": ring tile not repeatable over dirty shared memory\n";
            ++failures;
        }
        std::vector<double> actual, oracle;
        for (const std::int32_t row : sampled_rows(rows)) {
            for (std::int32_t column = 0; column < width; ++column) {
                actual.push_back(bf16_to_f32(got[static_cast<std::size_t>(column) * rows + row]));
                oracle.push_back(reference(row, x, init, column, width));
            }
        }
        failures += verify_reduction(name + " T=" + std::to_string(width) + " FP64 oracle", actual,
                                     oracle, kA16Tolerance);
    }
    return failures;
}

// Weight row `row` times column `column` of x [k, columns], in FP64.
double dot(const qw::PackedWeight& packed, std::int32_t row, const std::vector<std::uint16_t>& x,
           std::int32_t k, std::int32_t column) {
    double sum = 0.0;
    for (std::int32_t i = 0; i < k; ++i) {
        sum += qw::logical_weight_fp64(packed, row, i) *
               static_cast<double>(bf16_to_f32(x[static_cast<std::size_t>(column) * k + i]));
    }
    return sum;
}

int check_swiglu() {
    constexpr std::int32_t kRows = 34816, kK = 5120, kOut = kRows / 2;
    const DeviceWeight w(kRows, kK, 1801U);
    const Launch sliced = [&](const Tensor& x, Tensor& out) {
        ops::detail::nvfp4_linear_swiglu_invariant_sliced_launch(x, w.weight, out, nullptr);
    };
    const Launch ring = [&](const Tensor& x, Tensor& out) {
        ops::detail::nvfp4_linear_swiglu_invariant_ring_launch(x, w.weight, out, nullptr);
    };
    return check("NVFP4 linear_swiglu [34816,5120]", sliced, ring, kK, kOut, false,
                 [&](std::int32_t row, const std::vector<std::uint16_t>& x,
                     const std::vector<std::uint16_t>&, std::int32_t column, std::int32_t) {
                     const double gate = dot(w.host, row, x, kK, column);
                     const double up   = dot(w.host, row + kOut, x, kK, column);
                     return gate / (1.0 + std::exp(-gate)) * up;
                 });
}

int check_linear_add(std::int32_t k) {
    constexpr std::int32_t kRows = 5120;
    const DeviceWeight w(kRows, k, 707U + static_cast<std::uint32_t>(k));
    const Launch sliced = [&](const Tensor& x, Tensor& out) {
        ops::detail::nvfp4_linear_add_invariant_sliced_launch(x, w.weight, out, nullptr);
    };
    const Launch ring = [&](const Tensor& x, Tensor& out) {
        ops::detail::nvfp4_linear_add_invariant_ring_launch(x, w.weight, out, nullptr);
    };
    return check("NVFP4 linear_add [5120," + std::to_string(k) + "]", sliced, ring, k, kRows, true,
                 [&](std::int32_t row, const std::vector<std::uint16_t>& x,
                     const std::vector<std::uint16_t>& residual, std::int32_t column,
                     std::int32_t) {
                     return dot(w.host, row, x, k, column) +
                            static_cast<double>(bf16_to_f32(
                                residual[static_cast<std::size_t>(column) * kRows + row]));
                 });
}

} // namespace

INFERNIX_OP_TEST(nvfp4_a16_slice_ring) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = check_swiglu();
        failures += check_linear_add(17408);
        failures += check_linear_add(6144);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4 A16 slice ring tile\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4 A16 slice ring tile: " << error.what() << '\n';
        return 1;
    }
}
