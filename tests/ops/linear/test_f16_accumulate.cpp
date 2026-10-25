// f16-accumulate-prefill (LinearRoutes::F16Accumulate, ops/linear/f16acc/f16acc_plan.h): the FP8, MXFP8
// and Q8 problems that register the route, against the FP64 oracle of the represented weights and
// activations, with the exact (FP32-accumulation) route's error on the same inputs reported beside it.
// A Q8 weight is its int8 code times its FP16 group scale; the exact Q8 route rounds that product to
// BF16, the FP16 route folds the scale into the FP32 carry, and the same criterion holds (chunk partial
// sums stay below 32 * 128 * 2 = 8,192).
//
// Named criterion: the route meets the A16 criterion (relative L2 <= 2^-8, |error| <= 2^-8 + 2^-7
// max|reference|), and, as a guard on its own arithmetic, its relative L2 error stays within twice the
// exact route's plus 2^-12. Derivation (f16acc_plan.h): the codes and the encoded activations are exact
// FP16 operands except values more than 2^17 below their column maximum; each 32-value chunk's FP16 sum
// rounds at each k16 step to 11 significant bits, so the route adds about 2^-11 of the output's
// magnitude in RMS to the exact route's 2^-9 BF16 output rounding (RMS about 1.1e-3); chunk partial sums
// stay below 32 * 448 * 2 = 28,672, so no input overflows FP16.
//
// Also: the encoder against the exact oracle; the worst case for overflow (every code 448 and every
// activation at the top of its binade); repeat determinism; column invariance across widths and
// schedules; the admission seam and the flags the route ignores; the workspace contract; and the MXFP8
// GDN input projection's rows against linear's under the flag.

#include "ops/op_family.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/gdn_input_proj.h"
#include "ops/linear/f16acc/f16acc_plan.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test::linear;

struct Problem {
    QType qtype;
    std::int32_t n, k;
};

// Recipe C's row-FP8 Flash-Next problems, the Qwen3.8-27B-class [5120,6144], FP4-Local's MXFP8 ones and
// Dense8's Q8 ones (ops/linear/q8/q8_f16acc.h).
constexpr std::array<Problem, 18> kProblems{{{QType::FP8_E4M3FN_ROW_BF16, 16384, 2560},
                                             {QType::FP8_E4M3FN_ROW_BF16, 13312, 2560},
                                             {QType::FP8_E4M3FN_ROW_BF16, 2560, 6144},
                                             {QType::FP8_E4M3FN_ROW_BF16, 1280, 2560},
                                             {QType::FP8_E4M3FN_ROW_BF16, 2560, 640},
                                             {QType::FP8_E4M3FN_ROW_BF16, 5120, 6144},
                                             {QType::FP8_E4M3FN_G32_E8M0, 16384, 2560},
                                             {QType::FP8_E4M3FN_G32_E8M0, 13952, 2560},
                                             {QType::FP8_E4M3FN_G32_E8M0, 2560, 6144},
                                             {QType::FP8_E4M3FN_G32_E8M0, 1280, 2560},
                                             {QType::FP8_E4M3FN_G32_E8M0, 2560, 640},
                                             {QType::Q8_G32_FP16, 16384, 2560},
                                             {QType::Q8_G32_FP16, 13952, 2560},
                                             {QType::Q8_G32_FP16, 12800, 2560},
                                             {QType::Q8_G32_FP16, 2560, 2560},
                                             {QType::Q8_G32_FP16, 1280, 2560},
                                             {QType::Q8_G32_FP16, 2560, 6144},
                                             {QType::Q8_G32_FP16, 2560, 640}}};
// ops/linear/fp8/fp8_shapes.h kFp8F16MinTokens, ops/linear/mxfp8/mxfp8_shapes.h kMxfp8F16MinTokens.
constexpr std::int32_t kAdmission  = 256;
constexpr ops::LinearRoutes kF16    = ops::LinearRoutes::F16Accumulate;
constexpr ops::LinearRoutes kExact  = ops::LinearRoutes::Exact;
constexpr test::ReductionCriterion kA16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

const char* format_name(QType qtype) {
    return qtype == QType::FP8_E4M3FN_G32_E8M0 ? "MXFP8" : qtype == QType::Q8_G32_FP16 ? "Q8" : "FP8";
}

std::string label(const Problem& p, std::int32_t t, const char* what) {
    return std::string(what) + " " + format_name(p.qtype) + " [" + std::to_string(p.n) + "," + std::to_string(p.k) +
           "] T=" + std::to_string(t);
}

test::quantized_weight::PackedWeight make_weight(const Problem& p, std::uint32_t seed) {
    if (p.qtype == QType::Q8_G32_FP16) return make_q8_g32_fp16_weight(p.n, p.k, seed);
    return p.qtype == QType::FP8_E4M3FN_G32_E8M0 ? make_mxfp8_weight(p.n, p.k, seed) : make_fp8_weight(p.n, p.k, seed);
}

struct DeviceProblem {
    test::quantized_weight::PackedWeight host;
    DeviceBuffer weight;
    Weight view;

    DeviceProblem(test::quantized_weight::PackedWeight packed) : host(std::move(packed)), weight(host.payload.size()) {
        weight.copy_from_host(host.payload.data(), host.payload.size());
        view = host.device_weight(weight.p);
    }
};

DeviceBuffer upload(const std::vector<std::uint16_t>& bits) {
    DeviceBuffer buffer(bits.size() * sizeof(std::uint16_t));
    buffer.copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    return buffer;
}

// One public call at its own capacity query; returns the BF16 output bits.
std::vector<std::uint16_t> run_linear(const Weight& weight, const DeviceBuffer& x, std::int32_t t,
                                      ops::LinearPolicy policy, ops::LinearRoutes routes) {
    const std::size_t capacity = ops::linear_workspace_capacity_bytes(weight.qtype, weight.n, weight.k, policy, t, t);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    DeviceBuffer y(static_cast<std::size_t>(weight.n) * t * sizeof(std::uint16_t));
    Tensor input(x.p, DType::BF16, {weight.k, t});
    Tensor output(y.p, DType::BF16, {weight.n, t});
    ops::linear(input, weight, output, policy, workspace, nullptr, routes);
    test::cuda_check(cudaDeviceSynchronize(), "f16 accumulate linear");
    if (workspace.peak_used() > capacity || workspace.used() != 0) {
        throw std::runtime_error("f16 accumulate linear: workspace query/scope mismatch");
    }
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(weight.n) * t);
    y.copy_to_host(bits.data(), bits.size() * sizeof(std::uint16_t));
    return bits;
}

// Activations shaped like a transformer's: column magnitudes over nine binades, an outlier channel
// every 61st (2^8 larger), a massive-activation channel in every fourth column (2^14 larger, so most
// of that column lies 2^14-2^18 below its maximum, where the FP16 encoding starts to round), every
// 37th column zero, and one column whose every 32nd value lies 2^28 below the rest (flushed).
std::vector<std::uint16_t> transformer_activations(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    std::uint32_t state = seed * 2654435761U + static_cast<std::uint32_t>(t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t channel = 0; channel < k; ++channel) {
            state            = state * 1664525U + 1013904223U;
            const float unit = static_cast<float>(static_cast<std::int32_t>(state >> 9) - (1 << 22)) /
                               static_cast<float>(1 << 22);
            float value = std::ldexp(unit, token % 9 - 4);
            if (channel % 61 == 7) value *= 256.0F;
            if (token % 4 == 1 && channel == 11) value = std::ldexp(1.5F, 14 + token % 9 - 4);
            if (token % 37 == 5) value = 0.0F;
            if (token == 3 && channel % 32 == 9) value = std::ldexp(unit, -28);
            bits[static_cast<std::size_t>(token) * k + channel] = test::f32_to_bf16(value);
        }
    }
    return bits;
}

// The reference of sampled rows, and the routes' errors against it.
struct Accuracy {
    test::ReductionStats exact, fast;
    int failures = 0;
};

Accuracy accuracy(const Problem& p, const DeviceProblem& problem, const std::vector<std::uint16_t>& bits,
                  std::int32_t t, const char* what) {
    std::vector<std::int32_t> rows;
    for (std::int32_t i = 0; i < 64; ++i) rows.push_back(static_cast<std::int32_t>((static_cast<std::int64_t>(p.n) - 1) * i / 63));
    const std::vector<float> oracle_weight = test::quantized_weight::materialize_rows_fp32(problem.host, rows);
    std::vector<float> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) values[i] = test::bf16_to_f32(bits[i]);
    const auto row_count = static_cast<std::int32_t>(rows.size());
    std::vector<double> reference(static_cast<std::size_t>(row_count) * t);
    cpu_linear_gemm_fp64(oracle_weight.data(), values.data(), reference.data(), row_count, p.k, t);
    const DeviceBuffer x = upload(bits);
    const auto sampled   = [&](const std::vector<std::uint16_t>& output) {
        std::vector<double> actual(reference.size());
        for (std::int32_t token = 0; token < t; ++token)
            for (std::int32_t r = 0; r < row_count; ++r)
                actual[static_cast<std::size_t>(token) * row_count + r] = test::bf16_to_f32(
                    output[static_cast<std::size_t>(token) * p.n + rows[static_cast<std::size_t>(r)]]);
        return actual;
    };
    const auto exact = sampled(run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, kExact));
    const auto fast  = sampled(run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, kF16));
    Accuracy out;
    const auto count = static_cast<std::int64_t>(reference.size());
    out.exact        = test::compute_reduction_stats(exact.data(), reference.data(), count);
    out.fast         = test::compute_reduction_stats(fast.data(), reference.data(), count);
    const std::string name = label(p, t, what);
    std::cout << name << ": relative L2 exact " << out.exact.relative_l2 << " f16 " << out.fast.relative_l2
              << ", max |error| / max |reference| exact "
              << out.exact.maximum_absolute_error / std::max(out.exact.maximum_absolute_reference, 1e-300) << " f16 "
              << out.fast.maximum_absolute_error / std::max(out.fast.maximum_absolute_reference, 1e-300) << '\n';
    out.failures += test::verify_reduction(name + " f16", fast, reference, kA16Criterion);
    if (!(out.fast.relative_l2 <= 2.0 * out.exact.relative_l2 + std::ldexp(1.0, -12))) {
        std::cerr << name << ": the FP16 route's relative L2 " << out.fast.relative_l2
                  << " exceeds twice the exact route's " << out.exact.relative_l2 << " + 2^-12\n";
        ++out.failures;
    }
    return out;
}

// Both routes at the admission, a partial tile, a bulk width and across the 4,096-column tile seam; MXFP8 also
// past its row-fast raster's 8,192 columns (f16acc_mma.cuh kF16AccRowFastTokens) with a partial tile.
int run_accuracy() {
    int failures       = 0;
    std::uint32_t seed = 4201U;
    for (const Problem& p : kProblems) {
        const DeviceProblem problem(make_weight(p, seed++));
        for (const std::int32_t t : {kAdmission, 300, 1024, 4097, 8229}) {
            if (t == 8229 && p.qtype != QType::FP8_E4M3FN_G32_E8M0) { continue; }
            failures += accuracy(p, problem, transformer_activations(p.k, t, seed), t, "f16-accumulate").failures;
        }
    }
    return failures;
}

// The worst case for FP16 overflow: the first 16 rows' codes all 448 (0x7E) and every activation at
// the top of the binade the encoding maps to [1, 2) (BF16 1.9921875 * 2^e, all positive), so every
// chunk partial reaches 32 * 448 * 1.9921875 = 28,560 before the carry.
int run_worst_case() {
    int failures = 0;
    for (const QType qtype : {QType::FP8_E4M3FN_ROW_BF16, QType::FP8_E4M3FN_G32_E8M0, QType::Q8_G32_FP16}) {
        const Problem p{qtype, 2560, 640};
        auto packed = make_weight(p, 4301U);
        // The largest code: E4M3 448 (0x7E), int8 127 (0x7F), in the codes plane's first 16 rows.
        const std::uint8_t largest = qtype == QType::Q8_G32_FP16 ? std::uint8_t{0x7F} : std::uint8_t{0x7E};
        for (std::int32_t row = 0; row < 16; ++row)
            std::fill_n(packed.payload.begin() + static_cast<std::ptrdiff_t>(row) * p.k, p.k, largest);
        const DeviceProblem problem(std::move(packed));
        for (const std::int32_t t : {kAdmission, 513}) {
            std::vector<std::uint16_t> bits(static_cast<std::size_t>(p.k) * t);
            for (std::int32_t token = 0; token < t; ++token)
                for (std::int32_t channel = 0; channel < p.k; ++channel)
                    bits[static_cast<std::size_t>(token) * p.k + channel] =
                        test::f32_to_bf16(std::ldexp(1.9921875F, token % 7 - 3));
            failures += accuracy(p, problem, bits, t, "f16-accumulate worst case").failures;
        }
    }
    return failures;
}

double f16_value(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 0x1F;
    const int mantissa = bits & 0x3FF;
    const double magnitude =
        exponent == 0 ? std::ldexp(mantissa, -24) : std::ldexp(1024 + mantissa, exponent - 25);
    return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
}

// The encoder against the exact oracle: each inverse is a power of two putting the column maximum in
// [1, 2) (a zero column: 1); decoded with it, every value within 2^16 of its column maximum is exact
// and every other one within 2^-25 of the maximum. Each 16-value block is stored in the ldmatrix B order:
// value 4q + r at 2q + r (r < 2) or 8 + 2q + r - 2.
std::size_t encoded_position(std::int32_t channel) {
    const std::int32_t value = channel % 16, q = value / 4, r = value % 4;
    return static_cast<std::size_t>(channel - value + (r < 2 ? 2 * q + r : 8 + 2 * q + r - 2));
}

int check_encoder(std::int32_t k, std::int32_t t, const std::vector<std::uint16_t>& bits) {
    const DeviceBuffer x = upload(bits);
    DeviceArena arena(ops::detail::f16_accumulate_workspace_capacity_bytes(t, k));
    const auto workspace = ops::detail::allocate_f16_accumulate_workspace(arena, t, k);
    ops::detail::launch_f16_accumulate_encode(Tensor(x.p, DType::BF16, {k, t}), workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "f16 accumulate encode");
    std::vector<std::uint16_t> encoded(static_cast<std::size_t>(k) * t);
    std::vector<float> inverses(static_cast<std::size_t>(t));
    test::cuda_check(cudaMemcpy(encoded.data(), workspace.x, encoded.size() * 2, cudaMemcpyDeviceToHost), "encode copy");
    test::cuda_check(cudaMemcpy(inverses.data(), workspace.inverses, inverses.size() * 4, cudaMemcpyDeviceToHost),
                     "encode copy");
    int mismatches = 0;
    for (std::int32_t token = 0; token < t; ++token) {
        double maximum = 0.0;
        for (std::int32_t channel = 0; channel < k; ++channel)
            maximum = std::max(maximum, std::fabs(static_cast<double>(
                                            test::bf16_to_f32(bits[static_cast<std::size_t>(token) * k + channel]))));
        const double inverse = inverses[static_cast<std::size_t>(token)];
        int exponent         = 0;
        const bool power     = inverse > 0.0 && std::frexp(inverse, &exponent) == 0.5;
        const double scaled  = maximum / inverse;
        const bool normal    = maximum >= std::ldexp(1.0, -126) && maximum < std::ldexp(1.0, 126);
        if (!power || (maximum == 0.0 && inverse != 1.0) || (normal && maximum > 0.0 && !(scaled >= 1.0 && scaled < 2.0))) {
            if (mismatches++ < 3)
                std::cerr << "f16 accumulate encoder K=" << k << " column " << token << ": inverse " << inverse
                          << " for maximum " << maximum << '\n';
        }
        for (std::int32_t channel = 0; channel < k; ++channel) {
            const std::size_t at = static_cast<std::size_t>(token) * k + channel;
            const double value   = test::bf16_to_f32(bits[at]);
            const double decoded = f16_value(encoded[static_cast<std::size_t>(token) * k + encoded_position(channel)]) *
                                   inverse;
            const bool ok        = std::fabs(value) >= std::ldexp(maximum, -16)
                                       ? decoded == value
                                       : std::fabs(decoded - value) <= std::ldexp(maximum, -25);
            if (!ok && mismatches++ < 3)
                std::cerr << "f16 accumulate encoder K=" << k << " column " << token << " channel " << channel << ": "
                          << value << " decodes to " << decoded << '\n';
        }
    }
    return mismatches != 0 ? 1 : 0;
}

// Columns over nine binades with outliers, a zero column, columns at the BF16 extremes (2^+-120), one
// whose every 32nd value lies 2^20 below the rest, then the transformer activations.
int encoder_exact() {
    int failures = 0;
    for (const std::int32_t k : {640, 2560, 6144}) {
        constexpr std::int32_t tokens = 40;
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
        std::uint32_t state = 1013U * static_cast<std::uint32_t>(k);
        for (std::int32_t token = 0; token < tokens; ++token) {
            for (std::int32_t channel = 0; channel < k; ++channel) {
                state = state * 1664525U + 1013904223U;
                const float unit =
                    static_cast<float>(static_cast<std::int32_t>(state >> 9) - (1 << 22)) / static_cast<float>(1 << 22);
                float value = std::ldexp(unit, token % 9 - 4);
                if (channel % 61 == 7) value *= 256.0F;
                if (token == 5) value = 0.0F;
                if (token == 6) value = std::ldexp(unit, 120);
                if (token == 7) value = std::ldexp(unit, -120);
                if (token == 8 && channel % 32 == 3) value = std::ldexp(unit, -24);
                bits[static_cast<std::size_t>(token) * k + channel] = test::f32_to_bf16(value);
            }
        }
        failures += check_encoder(k, tokens, bits);
        failures += check_encoder(k, 300, transformer_activations(k, 300, 4401U));
    }
    return failures;
}

// Repeated calls on one input give the same bits, in two schedules of each problem.
int deterministic() {
    int failures       = 0;
    std::uint32_t seed = 4501U;
    for (const Problem& p : kProblems) {
        const DeviceProblem problem(make_weight(p, seed++));
        for (const std::int32_t t : {300, 4097}) {
            const DeviceBuffer x = upload(transformer_activations(p.k, t, seed));
            const auto first     = run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, kF16);
            for (int repeat = 0; repeat < 4; ++repeat) {
                if (run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, kF16) != first) {
                    std::cerr << label(p, t, "f16-accumulate") << ": outputs differ between calls\n";
                    ++failures;
                    break;
                }
            }
        }
    }
    return failures;
}

// A column's bits depend only on the column at every width the route serves: the first columns of
// calls of 256, 1,537 and 5,000 columns (different schedules) agree bit for bit.
int column_invariant() {
    int failures       = 0;
    std::uint32_t seed = 4601U;
    for (const Problem& p : kProblems) {
        const DeviceProblem problem(make_weight(p, seed++));
        constexpr std::int32_t kWidest = 5000;
        const DeviceBuffer x           = upload(transformer_activations(p.k, kWidest, seed));
        const auto widest              = run_linear(problem.view, x, kWidest, ops::LinearPolicy::A16Only, kF16);
        for (const std::int32_t t : {kAdmission, 1537}) {
            const auto narrow = run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, kF16);
            if (!std::equal(narrow.begin(), narrow.end(), widest.begin())) {
                std::cerr << label(p, t, "f16-accumulate") << ": columns differ from the same columns at T=" << kWidest
                          << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

// Below the admission and under A16ColumnInvariant the flag runs the exact route bit for bit; a problem
// that registers no FP16 route ignores it; the capacity query covers the route from its admission.
int seams_and_workspace() {
    int failures       = 0;
    std::uint32_t seed = 4701U;
    const auto compare = [&](const Problem& p, const DeviceProblem& problem, std::int32_t t, ops::LinearPolicy policy) {
        const DeviceBuffer x = upload(transformer_activations(p.k, t, seed));
        if (run_linear(problem.view, x, t, policy, kExact) != run_linear(problem.view, x, t, policy, kF16)) {
            std::cerr << label(p, t, "f16-accumulate") << ": the flag changed a call it does not serve\n";
            ++failures;
        }
    };
    for (const Problem& p : kProblems) {
        const DeviceProblem problem(make_weight(p, seed++));
        compare(p, problem, 1, ops::LinearPolicy::A16Only);
        compare(p, problem, kAdmission - 1, ops::LinearPolicy::A16Only);
        if (p.qtype == QType::FP8_E4M3FN_G32_E8M0) compare(p, problem, 300, ops::LinearPolicy::A16ColumnInvariant);
        const auto capacity = [&](std::int32_t first, std::int32_t last) {
            return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, ops::LinearPolicy::A16Only, first, last);
        };
        // Below the admission an MXFP8 interval may still reach the block32 pair route (from T 33 on the
        // widest shapes), so it needs no more than the admission width does.
        const std::size_t encoded = ops::linear_f16_accumulate_workspace_bytes(p.k, 4096);
        if (capacity(1, 4096) < encoded || capacity(kAdmission, kAdmission) == 0 ||
            capacity(1, kAdmission - 1) > capacity(kAdmission, kAdmission)) {
            std::cerr << label(p, 4096, "f16-accumulate") << ": workspace contract mismatch\n";
            ++failures;
        }
    }
    // Qwen3.8-27B's FP8 [14336,5120] registers no FP16 route.
    const Problem dense{QType::FP8_E4M3FN_ROW_BF16, 14336, 5120};
    const DeviceProblem problem(make_weight(dense, seed++));
    compare(dense, problem, 300, ops::LinearPolicy::A16Only);
    return failures;
}

// FP4-Local's GDN input projection under the flag: its q/k/v and z rows hold linear's FP16-route bits.
int gdn_input_rows() {
    int failures = 0;
    for (const QType qtype : {QType::FP8_E4M3FN_G32_E8M0, QType::Q8_G32_FP16}) {
    const Problem p{qtype, 16384, 2560};
    const DeviceProblem problem(make_weight(p, 4801U));
    for (const std::int32_t t : {kAdmission - 1, 300, 2049}) {
        const DeviceBuffer x = upload(transformer_activations(p.k, t, 4802U));
        const auto linear    = run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, kF16);
        const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(p.qtype, p.n, p.k,
                                                                                  ops::LinearPolicy::A16Only, t, t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        DeviceBuffer qkv(static_cast<std::size_t>(10240) * t * 2), z(static_cast<std::size_t>(6144) * t * 2);
        Tensor input(x.p, DType::BF16, {p.k, t});
        Tensor qkv_out(qkv.p, DType::BF16, {10240, t}), z_out(z.p, DType::BF16, {6144, t});
        ops::gdn_input_proj(input, problem.view, qkv_out, z_out, ops::LinearPolicy::A16Only, workspace, nullptr, kF16);
        test::cuda_check(cudaDeviceSynchronize(), "f16 accumulate gdn_input_proj");
        std::vector<std::uint16_t> q(static_cast<std::size_t>(10240) * t), zz(static_cast<std::size_t>(6144) * t);
        qkv.copy_to_host(q.data(), q.size() * 2);
        z.copy_to_host(zz.data(), zz.size() * 2);
        std::size_t differing = 0;
        for (std::int32_t token = 0; token < t; ++token) {
            for (std::int32_t row = 0; row < 16384; ++row) {
                const std::uint16_t got = row < 10240 ? q[static_cast<std::size_t>(token) * 10240 + row]
                                                      : zz[static_cast<std::size_t>(token) * 6144 + row - 10240];
                differing += got != linear[static_cast<std::size_t>(token) * 16384 + row];
            }
        }
        if (differing != 0 || workspace.peak_used() > capacity) {
            std::cerr << label(p, t, "f16-accumulate gdn_input_proj") << ": " << differing
                      << " rows differ from linear's (or the workspace query is short)\n";
            ++failures;
        }
    }
    }
    return failures;
}

// The QSA group projection (ops::linear_qsa_group, the TEMPORARY INFERNIX_TMP_QSA_GROUP_DIRECT route): its
// six parts hold linear's rows bit for bit, on the exact routes (decode and prefill widths) and the
// f16-accumulate route, for Q8 (Dense8) and MXFP8 (FP4-Local); A16ColumnInvariant has no Q8 segmented form.
int qsa_group_rows() {
    int failures = 0;
    constexpr std::int32_t kRows[ops::kQsaGroupSegments] = {6144, 6144, 512, 512, 512, 128};
    for (const QType qtype : {QType::Q8_G32_FP16, QType::FP8_E4M3FN_G32_E8M0}) {
        const Problem p{qtype, 13952, 2560};
        const DeviceProblem problem(make_weight(p, 4901U));
        for (const std::int32_t t : {1, 4, 8, 64, 97, 300}) {
            for (const ops::LinearRoutes routes : {kExact, kF16}) {
                const DeviceBuffer x = upload(transformer_activations(p.k, t, 4902U));
                const auto linear    = run_linear(problem.view, x, t, ops::LinearPolicy::A16Only, routes);
                std::vector<DeviceBuffer> buffers;
                std::vector<Tensor> tensors;
                buffers.reserve(ops::kQsaGroupSegments);
                tensors.reserve(ops::kQsaGroupSegments);
                for (const std::int32_t rows : kRows) {
                    buffers.emplace_back(static_cast<std::size_t>(rows) * t * 2);
                    tensors.emplace_back(buffers.back().p, DType::BF16, std::initializer_list<std::int32_t>{rows, t});
                }
                Tensor* parts[ops::kQsaGroupSegments];
                for (std::int32_t i = 0; i < ops::kQsaGroupSegments; ++i) { parts[i] = &tensors[i]; }
                if (!ops::linear_qsa_group_supported(problem.view, parts, ops::LinearPolicy::A16Only)) {
                    std::cerr << label(p, t, "qsa group") << ": the segmented route is not supported\n";
                    ++failures;
                    continue;
                }
                const std::size_t capacity = ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k,
                                                                                  ops::LinearPolicy::A16Only, t, t);
                DeviceArena workspace(std::max<std::size_t>(capacity, 256));
                Tensor input(x.p, DType::BF16, {p.k, t});
                ops::linear_qsa_group(input, problem.view, parts, ops::LinearPolicy::A16Only, workspace, nullptr, routes);
                test::cuda_check(cudaDeviceSynchronize(), "qsa group");
                std::size_t differing = 0;
                std::int32_t first    = 0;
                for (std::int32_t i = 0; i < ops::kQsaGroupSegments; ++i) {
                    std::vector<std::uint16_t> got(static_cast<std::size_t>(kRows[i]) * t);
                    buffers[static_cast<std::size_t>(i)].copy_to_host(got.data(), got.size() * 2);
                    for (std::int32_t token = 0; token < t; ++token) {
                        for (std::int32_t row = 0; row < kRows[i]; ++row) {
                            differing += got[static_cast<std::size_t>(token) * kRows[i] + row] !=
                                         linear[static_cast<std::size_t>(token) * p.n + first + row];
                        }
                    }
                    first += kRows[i];
                }
                if (differing != 0 || workspace.peak_used() > capacity) {
                    std::cerr << label(p, t, routes == kF16 ? "qsa group (f16)" : "qsa group") << ": " << differing
                              << " values differ from linear's rows (or the workspace query is short)\n";
                    ++failures;
                }
            }
        }
        // Q8 under A16ColumnInvariant: no segmented form (linear's own column-invariant route serves it).
        if (qtype == QType::Q8_G32_FP16) {
            std::vector<DeviceBuffer> buffers;
            std::vector<Tensor> tensors;
            buffers.reserve(ops::kQsaGroupSegments);
            tensors.reserve(ops::kQsaGroupSegments);
            for (const std::int32_t rows : kRows) {
                buffers.emplace_back(static_cast<std::size_t>(rows) * 2);
                tensors.emplace_back(buffers.back().p, DType::BF16, std::initializer_list<std::int32_t>{rows, 1});
            }
            const Tensor* parts[ops::kQsaGroupSegments];
            for (std::int32_t i = 0; i < ops::kQsaGroupSegments; ++i) { parts[i] = &tensors[i]; }
            if (ops::linear_qsa_group_supported(problem.view, parts, ops::LinearPolicy::A16ColumnInvariant)) {
                std::cerr << "qsa group: Q8 claims a column-invariant segmented route\n";
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_f16_accumulate) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = encoder_exact() + run_accuracy() + run_worst_case() + deterministic() +
                             column_invariant() + seams_and_workspace() + gdn_input_rows() + qsa_group_rows();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " f16-accumulate-prefill Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "f16-accumulate-prefill Linear: " << error.what() << '\n';
        return 1;
    }
}
