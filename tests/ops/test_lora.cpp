// LoRA (include/infernix/ops/lora.h).
//
// 1. lora_shrink against an FP64 oracle at the 27B's stacked shapes, and its canonical bits: a column's U is the same
//    whatever the call's width and the column's position (every schedule).
// 2. The term on every adapted Op, format and route at the 27B shapes, on zero base weights: the projection's output
//    is then its epilogue of the term alone, so every row, segment and column of the term is observable against an
//    FP64 oracle (sum of B U times the segment's scale), through the SwiGLU, the residual add and the four-way
//    attention split. An unadapted segment (u_offset < 0) must stay exactly zero.
// 3. With non-zero base weights under LinearPolicy::A16ColumnInvariant, a column's output bits do not depend on the
//    call's width (the term keeps the routes' request invariance).

#include "ops/host_parallel.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/weight.h"
#include "infernix/ops/attn_input_proj.h"
#include "infernix/ops/gdn_gating_proj.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/lora.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace infernix;
namespace qw = test::quantized_weight;

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<std::uint16_t> out(count);
    for (auto& v : out) { v = test::f32_to_bf16(dist(rng)); }
    return out;
}

double bf16(std::uint16_t bits) { return static_cast<double>(test::bf16_to_f32(bits)); }

// |got - ref| <= abs + rel scale (scale = |ref|, or the given per-element magnitude) for every element; prints the
// worst.
int compare(const std::string& label, const std::vector<double>& got, const std::vector<double>& ref, double rel,
            double abs, const std::vector<double>* magnitude = nullptr) {
    std::size_t worst = 0, failures = 0;
    double worst_excess = -1.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double limit  = abs + rel * (magnitude ? (*magnitude)[i] : std::fabs(ref[i]));
        const double excess = std::isfinite(got[i]) ? std::fabs(got[i] - ref[i]) - limit : 1e300;
        if (excess > 0) { ++failures; }
        if (excess > worst_excess) {
            worst_excess = excess;
            worst        = i;
        }
    }
    if (failures == 0) { return 0; }
    std::cerr << label << ": " << failures << " of " << got.size() << " outside tolerance; worst at " << worst
              << " got " << got[worst] << " reference " << ref[worst] << '\n';
    return 1;
}

// Every element's bits; prints how many differ and the first.
int compare_bits(const std::string& label, const std::uint16_t* got, const std::uint16_t* ref, std::size_t count) {
    std::size_t differ = 0, first = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (got[i] != ref[i] && differ++ == 0) { first = i; }
    }
    if (differ == 0) { return 0; }
    std::cerr << label << ": " << differ << " of " << count << " differ; first at " << first << " (" << got[first]
              << " vs " << ref[first] << ")\n";
    return 1;
}

// ---------------------------------------------------------------------------------------------------------------
int check_shrink() {
    int failures                 = 0;
    // Every schedule the calls take: split (below 64 columns and the narrow stacks' short prompts), stream and tile
    // (the modelled fastest); the widest call's bits are then each width's.
    const std::int32_t widths[] = {1, 2, 7, 16, 32, 33, 100, 129, 511, 512, 1024, 4096, 4100};
    constexpr std::int32_t kMaxT = 4100;
    struct Shape {
        std::int32_t rows, k;
    };
    for (const Shape shape : {Shape{96, 5120}, Shape{64, 5120}, Shape{32, 6144}, Shape{32, 17408}, Shape{192, 5120},
                              Shape{128, 17408}, Shape{160, 5120}, Shape{240, 5120}, Shape{256, 5120}, Shape{288, 5120},
                              Shape{384, 5120}, Shape{336, 5120}, Shape{80, 17408}, Shape{112, 6144},
                              Shape{224, 5120}, Shape{512, 5120}}) {
        const auto a = random_bf16(static_cast<std::size_t>(shape.rows) * shape.k, 11 + shape.rows, -0.03F, 0.03F);
        // Wide activations with cancellation: alternating large and small values.
        auto x = random_bf16(static_cast<std::size_t>(shape.k) * kMaxT, 7 + shape.k, -2.0F, 2.0F);
        for (std::size_t i = 0; i < x.size(); i += 97) { x[i] = test::f32_to_bf16(test::bf16_to_f32(x[i]) * 64.0F); }
        test::GuardedDeviceBuffer a_d(a.size() * 2), x_d(x.size() * 2),
            u_d(static_cast<std::size_t>(shape.rows) * kMaxT * 2);
        a_d.copy_from_host(a.data(), a.size() * 2);
        x_d.copy_from_host(x.data(), x.size() * 2);
        const ops::LoraShrinkWeight weight{a_d.data(), shape.rows, shape.k};
        std::vector<std::uint16_t> widest;
        for (const std::int32_t t : widths) {
            Tensor xt(x_d.data(), DType::BF16, {shape.k, t});
            Tensor ut(u_d.data(), DType::BF16, {shape.rows, t});
            ops::lora_shrink(xt, weight, ut, nullptr);
            test::cuda_synchronize();
            std::vector<std::uint16_t> u(static_cast<std::size_t>(shape.rows) * t);
            u_d.copy_to_host(u.data(), u.size() * 2);
            const std::string label =
                "lora_shrink [" + std::to_string(shape.rows) + "," + std::to_string(shape.k) + "] T=" + std::to_string(t);
            // The oracle on a sample of columns (every column at small T).
            const std::int32_t step    = t <= 64 ? 1 : 37;
            const std::int32_t sampled = (t + step - 1) / step;
            const std::size_t count    = static_cast<std::size_t>(sampled) * shape.rows;
            std::vector<double> got(count), ref(count), mass(count);
            test::parallel_ranges(sampled, test::host_thread_count(), [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t s = begin; s < end; ++s) {
                    const std::int64_t c = s * step;
                    for (std::int32_t j = 0; j < shape.rows; ++j) {
                        double sum = 0.0, abs_sum = 0.0;
                        for (std::int32_t k = 0; k < shape.k; ++k) {
                            const double p = bf16(a[static_cast<std::size_t>(j) * shape.k + k]) *
                                             bf16(x[static_cast<std::size_t>(c) * shape.k + k]);
                            sum += p;
                            abs_sum += std::fabs(p);
                        }
                        const std::size_t i = static_cast<std::size_t>(s) * shape.rows + j;
                        got[i]              = bf16(u[static_cast<std::size_t>(c) * shape.rows + j]);
                        ref[i]              = sum;
                        mass[i]             = abs_sum;
                    }
                }
            });
            // BF16 storage (2^-8 relative, both roundings) plus FP32 accumulation over the chain and the sums.
            double worst_mass = 0.0;
            for (const double m : mass) { worst_mass = std::max(worst_mass, m); }
            failures += compare(label, got, ref, 1.0 / 256.0, worst_mass * 4.0e-6);
            // Canonical bits: every column equals its bits in the widest call, whatever the width or schedule.
            if (t == widths[std::size(widths) - 1]) { widest = u; }
        }
        for (const std::int32_t t : widths) {
            Tensor xt(x_d.data(), DType::BF16, {shape.k, t});
            Tensor ut(u_d.data(), DType::BF16, {shape.rows, t});
            ops::lora_shrink(xt, weight, ut, nullptr);
            test::cuda_synchronize();
            std::vector<std::uint16_t> u(static_cast<std::size_t>(shape.rows) * t);
            u_d.copy_to_host(u.data(), u.size() * 2);
            failures += compare_bits("lora_shrink canonical bits [" + std::to_string(shape.rows) + "," +
                                         std::to_string(shape.k) + "] T=" + std::to_string(t),
                                     u.data(), widest.data(), u.size());
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------------------------
// One adapted parent: B [rows, rank] and U [u_rows, T] (BF16), the segments, and the FP64 term per element.
struct Term {
    std::int32_t rows = 0, rank = 32, u_rows = 0;
    std::vector<std::uint16_t> b, u;
    ops::LoraAddend addend;
    // The FP64 term (sum over j of B[row, j] U[j, token], times the segment's scale; 0 on an unadapted segment) of
    // every row and token, column-major: computed once, since every policy and width of a check reads the same values.
    std::vector<double> values;

    double value(std::int32_t row, std::int32_t token) const {
        return values[static_cast<std::size_t>(token) * rows + row];
    }
};

Term make_term(std::int32_t rows, std::int32_t u_rows, std::int32_t max_tokens, std::uint32_t seed,
               std::initializer_list<ops::LoraSegment> segments, test::GuardedDeviceBuffer& b_d,
               test::GuardedDeviceBuffer& u_d, std::int32_t rank = 32) {
    Term term;
    term.rows   = rows;
    term.rank   = rank;
    term.u_rows = u_rows;
    term.b      = random_bf16(static_cast<std::size_t>(rows) * term.rank, seed, -0.05F, 0.05F);
    term.u      = random_bf16(static_cast<std::size_t>(u_rows) * max_tokens, seed + 1, -3.0F, 3.0F);
    b_d.copy_from_host(term.b.data(), term.b.size() * 2);
    u_d.copy_from_host(term.u.data(), term.u.size() * 2);
    term.addend.b        = b_d.data();
    term.addend.rank     = term.rank;
    term.addend.u        = u_d.data();
    term.addend.u_rows   = u_rows;
    term.addend.segments = 0;
    for (const ops::LoraSegment& s : segments) { term.addend.segment[term.addend.segments++] = s; }
    std::vector<double> b_value(term.b.size()), u_value(term.u.size());
    for (std::size_t i = 0; i < b_value.size(); ++i) { b_value[i] = bf16(term.b[i]); }
    for (std::size_t i = 0; i < u_value.size(); ++i) { u_value[i] = bf16(term.u[i]); }
    term.values.assign(static_cast<std::size_t>(rows) * max_tokens, 0.0);
    test::parallel_ranges(max_tokens, test::host_thread_count(), [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t token = begin; token < end; ++token) {
            int s = 0;
            for (std::int32_t row = 0; row < rows; ++row) {
                while (s + 1 < term.addend.segments && row >= term.addend.segment[s].row_end) { ++s; }
                const ops::LoraSegment& seg = term.addend.segment[s];
                if (seg.u_offset < 0) { continue; }
                const double* b_row = &b_value[static_cast<std::size_t>(row) * rank];
                const double* u_col = &u_value[static_cast<std::size_t>(token) * u_rows + seg.u_offset];
                double sum          = 0.0;
                for (std::int32_t j = 0; j < rank; ++j) { sum += b_row[j] * u_col[j]; }
                term.values[static_cast<std::size_t>(token) * rows + row] = sum * seg.scale;
            }
        }
    });
    return term;
}

// gdn_gating_proj's LoRA form (the 27B's Gated DeltaNet control, 48 heads over 5120 inputs): a and b plus their terms
// before the gates, against FP64 (the projection over the represented h and W, the term over the represented B and U,
// then g = -exp(A_log) softplus(a + dt_bias) and beta = sigmoid(b)); and the column invariance its one schedule
// promises (every column's g and beta bits equal those of the widest call). a's rows take rank r over U rows [0, r) at
// scale 2, b's over [r, 2r) at 0.5; a second check leaves b unadapted (its gate is then the plain projection's).
int check_gating_terms(std::int32_t rank, bool adapt_b) {
    constexpr std::int32_t heads = 48, k = 5120, rows = 2 * heads;
    constexpr std::int32_t kMax  = 300;
    int failures                 = 0;
    const auto w = random_bf16(static_cast<std::size_t>(rows) * k, 71U + rank, -0.02F, 0.02F); // a rows, then b rows
    const auto h = random_bf16(static_cast<std::size_t>(k) * kMax, 72U, -2.0F, 2.0F);
    std::vector<float> a_log(heads), dt_bias(heads);
    for (std::int32_t i = 0; i < heads; ++i) {
        a_log[i]   = -1.0F + 0.05F * static_cast<float>(i);
        dt_bias[i] = 0.5F - 0.02F * static_cast<float>(i);
    }
    test::GuardedDeviceBuffer w_d(w.size() * 2), h_d(h.size() * 2), a_log_d(heads * 4), dt_d(heads * 4),
        g_d(static_cast<std::size_t>(heads) * kMax * 4), beta_d(static_cast<std::size_t>(heads) * kMax * 4),
        b_d(static_cast<std::size_t>(rows) * rank * 2), u_d(static_cast<std::size_t>(2 * rank) * kMax * 2);
    w_d.copy_from_host(w.data(), w.size() * 2);
    h_d.copy_from_host(h.data(), h.size() * 2);
    a_log_d.copy_from_host(a_log.data(), heads * 4);
    dt_d.copy_from_host(dt_bias.data(), heads * 4);
    const Term term = make_term(rows, 2 * rank, kMax, 73U + rank,
                                {{heads, 0, 2.0F}, {rows, adapt_b ? rank : -1, adapt_b ? 0.5F : 0.0F}}, b_d, u_d, rank);
    Weight ab{};
    ab.qtype           = QType::BF16;
    ab.layout          = QuantLayout::Contiguous;
    ab.payload         = w_d.data();
    ab.payload_bytes   = static_cast<std::uint64_t>(rows) * k * 2;
    ab.qdata           = w_d.data();
    ab.ndim            = 2;
    ab.shape[0]        = rows;
    ab.shape[1]        = k;
    ab.padded_shape[0] = rows;
    ab.padded_shape[1] = k;
    ab.n               = rows;
    ab.k               = k;
    // The FP64 projections and their mass (the FP32 accumulation's error scale).
    std::vector<double> projection(static_cast<std::size_t>(rows) * kMax), mass(projection.size());
    test::parallel_ranges(kMax, test::host_thread_count(), [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t t = begin; t < end; ++t) {
            for (std::int32_t r = 0; r < rows; ++r) {
                double sum = 0.0, abs_sum = 0.0;
                for (std::int32_t j = 0; j < k; ++j) {
                    const double p = bf16(w[static_cast<std::size_t>(r) * k + j]) * bf16(h[static_cast<std::size_t>(t) * k + j]);
                    sum += p;
                    abs_sum += std::fabs(p);
                }
                projection[static_cast<std::size_t>(t) * rows + r] = sum + term.value(r, static_cast<std::int32_t>(t));
                mass[static_cast<std::size_t>(t) * rows + r]       = abs_sum + std::fabs(term.value(r, static_cast<std::int32_t>(t)));
            }
        }
    });
    const auto softplus = [](double v) { return std::max(v, 0.0) + std::log1p(std::exp(-std::fabs(v))); };
    const auto sigmoid  = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    std::vector<float> widest_g, widest_beta;
    for (const std::int32_t t : {kMax, 1, 2, 7, 8, 9, 32, 33, 64, 100}) {
        Tensor ht(h_d.data(), DType::BF16, {k, t});
        Tensor a_log_t(a_log_d.data(), DType::FP32, {heads});
        Tensor dt_t(dt_d.data(), DType::FP32, {heads});
        Tensor gt(g_d.data(), DType::FP32, {heads, t}), bt(beta_d.data(), DType::FP32, {heads, t});
        ops::gdn_gating_proj(ht, ab, a_log_t, dt_t, term.addend, gt, bt, nullptr);
        test::cuda_synchronize();
        std::vector<float> g(static_cast<std::size_t>(heads) * t), beta(g.size());
        g_d.copy_to_host(g.data(), g.size() * 4);
        beta_d.copy_to_host(beta.data(), beta.size() * 4);
        const std::string label = "gdn gating lora r" + std::to_string(rank) + (adapt_b ? "" : " a only") + " T=" + std::to_string(t);
        if (t == kMax) {
            // The oracle (every column of the widest call): the gates' slopes bound the FP32 projection's error.
            std::vector<double> got_g(g.size()), ref_g(g.size()), got_b(g.size()), ref_b(g.size()), slack_g(g.size()),
                slack_b(g.size());
            for (std::int32_t c = 0; c < t; ++c) {
                for (std::int32_t i = 0; i < heads; ++i) {
                    const std::size_t o  = static_cast<std::size_t>(c) * heads + i;
                    const double a       = projection[static_cast<std::size_t>(c) * rows + i];
                    const double b       = projection[static_cast<std::size_t>(c) * rows + heads + i];
                    const double decay   = std::exp(static_cast<double>(a_log[i]));
                    got_g[o]             = g[o];
                    ref_g[o]             = -decay * softplus(a + dt_bias[i]);
                    got_b[o]             = beta[o];
                    ref_b[o]             = sigmoid(b);
                    slack_g[o] = decay * 4.0e-6 * mass[static_cast<std::size_t>(c) * rows + i] + 4.0e-6 * std::fabs(ref_g[o]);
                    slack_b[o] = 0.25 * 4.0e-6 * mass[static_cast<std::size_t>(c) * rows + heads + i] + 4.0e-6;
                }
            }
            // The criteria's absolute parts carry the per-element slack (compare takes one magnitude vector).
            for (std::size_t o = 0; o < got_g.size(); ++o) {
                if (!(std::fabs(got_g[o] - ref_g[o]) <= slack_g[o])) {
                    std::cerr << label << " g: element " << o << " got " << got_g[o] << " reference " << ref_g[o] << '\n';
                    ++failures;
                    break;
                }
            }
            for (std::size_t o = 0; o < got_b.size(); ++o) {
                if (!(std::fabs(got_b[o] - ref_b[o]) <= slack_b[o])) {
                    std::cerr << label << " beta: element " << o << " got " << got_b[o] << " reference " << ref_b[o] << '\n';
                    ++failures;
                    break;
                }
            }
            widest_g    = g;
            widest_beta = beta;
        } else if (std::memcmp(g.data(), widest_g.data(), g.size() * 4) != 0 ||
                   std::memcmp(beta.data(), widest_beta.data(), beta.size() * 4) != 0) {
            std::cerr << label << ": a column's bits differ from the widest call's\n";
            ++failures;
        }
    }
    return failures;
}

// The weight's codes zeroed (every decoded value 0: Q4/Q5 codes are the nibble's two's complement, with the Q5 high
// plane zeroed too); scales and divisors kept.
qw::PackedWeight zero_weight(QType qtype, std::int32_t n, std::int32_t k, qw::PatternedWeightOptions options) {
    auto packed = qw::make_patterned_weight(qtype, n, k, 4242U, options);
    std::fill(packed.payload.begin(), packed.payload.begin() + static_cast<std::ptrdiff_t>(packed.code_plane_bytes), 0);
    if (packed.high_plane_bytes != 0) {
        const auto high = packed.payload.begin() + static_cast<std::ptrdiff_t>(packed.high_plane_offset);
        std::fill(high, high + static_cast<std::ptrdiff_t>(packed.high_plane_bytes), 0);
    }
    return packed;
}

// Every route boundary of the adapted Ops' formats: NVFP4/FP8 decode, small-T and tile widths; the groupwise
// routes' (Q5 split2 1-4, sliced 5-160, MMA 65-96 and 161-256, ring 257+ with its narrow tail; Q4 gate/up GEMV 1,
// small-T 2-32, Materialized 33-95 / 169-192 / 224, ring 96+; the Q4/Q5 attention pair 1-12, mixed 13-64, ring and
// pair tiles above; 257 and 385 columns split the FP8 A8 gate/up and attention tiles' K in their last wave).
const std::int32_t kWidths[] = {1,  2,  3,  4,   5,   8,   9,   12,  13,  16,  17,  24,  32,  33,
                                48, 64, 65, 96,  97,  128, 129, 160, 177, 224, 256, 257, 385, 513, 1000};
constexpr std::int32_t kMaxWidth = 1000;

const char* policy_name(ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only: return "A16Only";
    case ops::LinearPolicy::AllowA8: return "AllowA8";
    case ops::LinearPolicy::AllowA4: return "AllowA4";
    case ops::LinearPolicy::A16ColumnInvariant: return "A16ColumnInvariant";
    case ops::LinearPolicy::AllowA4Residual: return "AllowA4Residual";
    case ops::LinearPolicy::AllowA8Residual: return "AllowA8Residual";
    case ops::LinearPolicy::A8Only: return "A8Only";
    }
    return "?";
}

int check_linear_add_terms(QType qtype, std::int32_t k, qw::PatternedWeightOptions options,
                           std::initializer_list<ops::LinearPolicy> policies, std::int32_t rank = 32) {
    constexpr std::int32_t n = 5120;
    int failures             = 0;
    const auto packed        = zero_weight(qtype, n, k, options);
    test::GuardedDeviceBuffer w_d(packed.payload.size());
    w_d.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight = packed.device_weight(w_d.data());
    test::GuardedDeviceBuffer b_d(static_cast<std::size_t>(n) * rank * 2),
        u_d(static_cast<std::size_t>(rank) * kMaxWidth * 2);
    const Term term = make_term(n, rank, kMaxWidth, 900U + k + rank, {{n, 0, 2.0F}}, b_d, u_d, rank);
    const auto residual0 = random_bf16(static_cast<std::size_t>(n) * kMaxWidth, 33U, -4.0F, 4.0F);
    const auto x         = random_bf16(static_cast<std::size_t>(k) * kMaxWidth, 34U, -1.0F, 1.0F);
    test::GuardedDeviceBuffer x_d(x.size() * 2), r_d(residual0.size() * 2);
    x_d.copy_from_host(x.data(), x.size() * 2);
    for (const ops::LinearPolicy policy : policies) {
        for (const ops::LinearRoutes routes :
             {ops::LinearRoutes::Exact, ops::LinearRoutes::WideDownTile, ops::LinearRoutes::Q5KSplit}) {
            if (routes == ops::LinearRoutes::WideDownTile && !(qtype == QType::NVFP4 && k == 17408)) { continue; }
            if (routes == ops::LinearRoutes::Q5KSplit && qtype != QType::Q5_G64_FP16) { continue; }
            WorkspaceArena workspace(
                std::max<std::size_t>(ops::linear_add_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth), 256));
            for (const std::int32_t t : kWidths) {
                r_d.copy_from_host(residual0.data(), static_cast<std::size_t>(n) * t * 2);
                Tensor xt(x_d.data(), DType::BF16, {k, t});
                Tensor rt(r_d.data(), DType::BF16, {n, t});
                ops::linear_add(xt, weight, rt, policy, workspace, nullptr, routes, &term.addend);
                test::cuda_synchronize();
                std::vector<std::uint16_t> out(static_cast<std::size_t>(n) * t);
                r_d.copy_to_host(out.data(), out.size() * 2);
                std::vector<double> got(out.size()), ref(out.size()), magnitude(out.size());
                for (std::int32_t c = 0; c < t; ++c) {
                    for (std::int32_t row = 0; row < n; ++row) {
                        const std::size_t i = static_cast<std::size_t>(c) * n + row;
                        const double value  = term.value(row, c);
                        got[i]              = bf16(out[i]);
                        ref[i]              = bf16(residual0[i]) + value;
                        magnitude[i] = std::max({std::fabs(bf16(residual0[i])), std::fabs(value), std::fabs(ref[i])});
                    }
                }
                // BF16 roundings relative to the larger operand: the sum's, and on the routes that store the
                // projection in BF16 before the residual's add (the Q5 ring), the projection's; plus the term's
                // FP32 / BF16-U arithmetic.
                failures += compare("linear_add term " + std::string(policy_name(policy)) +
                                        (routes == ops::LinearRoutes::Exact          ? ""
                                         : routes == ops::LinearRoutes::WideDownTile ? " wide"
                                                                                     : " ksplit") +
                                        " K=" +
                                        std::to_string(k) + " qtype=" + std::to_string(static_cast<int>(qtype)) +
                                        " r=" + std::to_string(rank) + " T=" + std::to_string(t),
                                    got, ref, 1.0 / 128.0, 1.0e-4, &magnitude);
            }
        }
    }
    return failures;
}

int check_linear_swiglu_terms(QType qtype, qw::PatternedWeightOptions options,
                              std::initializer_list<ops::LinearPolicy> policies,
                              ops::LinearRoutes routes = ops::LinearRoutes::Exact, std::int32_t rank = 32) {
    constexpr std::int32_t n = 34816, k = 5120, half = n / 2;
    int failures      = 0;
    const auto packed = zero_weight(qtype, n, k, options);
    test::GuardedDeviceBuffer w_d(packed.payload.size());
    w_d.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight = packed.device_weight(w_d.data());
    test::GuardedDeviceBuffer b_d(static_cast<std::size_t>(n) * rank * 2),
        u_d(static_cast<std::size_t>(2 * rank) * kMaxWidth * 2);
    // Gate rows take U rows 0 .. rank - 1 at scale 2, up rows the next rank at scale 0.5.
    const Term term = make_term(n, 2 * rank, kMaxWidth, 77U + rank, {{half, 0, 2.0F}, {n, rank, 0.5F}}, b_d, u_d, rank);
    const auto x = random_bf16(static_cast<std::size_t>(k) * kMaxWidth, 35U, -1.0F, 1.0F);
    test::GuardedDeviceBuffer x_d(x.size() * 2), y_d(static_cast<std::size_t>(half) * kMaxWidth * 2);
    x_d.copy_from_host(x.data(), x.size() * 2);
    for (const ops::LinearPolicy policy : policies) {
        WorkspaceArena workspace(std::max<std::size_t>(
            ops::linear_swiglu_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth, routes), 256));
        for (const std::int32_t t : kWidths) {
            Tensor xt(x_d.data(), DType::BF16, {k, t});
            Tensor yt(y_d.data(), DType::BF16, {half, t});
            ops::linear_swiglu(xt, weight, yt, policy, workspace, nullptr, routes, &term.addend);
            test::cuda_synchronize();
            std::vector<std::uint16_t> out(static_cast<std::size_t>(half) * t);
            y_d.copy_to_host(out.data(), out.size() * 2);
            // The Q4 release route's Materialized widths store gate and up in BF16 (2^-9 relative each) before the
            // SwiGLU, whose relative sensitivity to the gate is |1 + g (1 - sigmoid(g))| (above 1 for negative
            // gates): their bound is 2^-8 (that sensitivity + 2), twice the first-order sum of the gate's, up's and
            // output's roundings; every other route keeps 1/128 of |ref|.
            const bool materialized = qtype == QType::Q4_G64_FP16 && policy == ops::LinearPolicy::A16Only &&
                                      ((t >= 33 && t <= 95) || (t >= 169 && t <= 192) || t == 224);
            std::vector<double> got(out.size()), ref(out.size()), magnitude(out.size());
            test::parallel_ranges(t, test::host_thread_count(), [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t c = begin; c < end; ++c) {
                    for (std::int32_t row = 0; row < half; ++row) {
                        const std::size_t i = static_cast<std::size_t>(c) * half + row;
                        const double g      = term.value(row, static_cast<std::int32_t>(c));
                        const double u      = term.value(half + row, static_cast<std::int32_t>(c));
                        const double sigmoid     = 1.0 / (1.0 + std::exp(-g));
                        const double sensitivity = std::fabs(1.0 + g * (1.0 - sigmoid));
                        got[i]       = bf16(out[i]);
                        ref[i]       = g * sigmoid * u;
                        magnitude[i] =
                            std::fabs(ref[i]) * (materialized ? std::max(1.0, (sensitivity + 2.0) / 2.0) : 1.0);
                    }
                }
            });
            failures += compare("linear_swiglu term " + std::string(policy_name(policy)) +
                                    (routes == ops::LinearRoutes::Exact ? "" : " w4a8") + " qtype=" +
                                    std::to_string(static_cast<int>(qtype)) + " r=" + std::to_string(rank) +
                                    " T=" + std::to_string(t),
                                got, ref, 1.0 / 128.0, 1.0e-4, &magnitude);
        }
    }
    return failures;
}

int check_attention_terms(QType qtype, qw::PatternedWeightOptions options,
                          std::initializer_list<ops::LinearPolicy> policies, std::int32_t rank = 32) {
    constexpr std::int32_t n = 14336, k = 5120, qw_rows = 6144, kv = 1024;
    int failures      = 0;
    const auto packed = zero_weight(qtype, n, k, options);
    test::GuardedDeviceBuffer w_d(packed.payload.size());
    w_d.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight = packed.device_weight(w_d.data());
    test::GuardedDeviceBuffer b_d(static_cast<std::size_t>(n) * rank * 2),
        u_d(static_cast<std::size_t>(2 * rank) * kMaxWidth * 2);
    // Query and gate rows share q_proj's U rows 0 .. rank - 1 (scale 2); key is not adapted (must stay exactly 0); value
    // takes the next rank U rows (scale 1.5).
    const Term term = make_term(n, 2 * rank, kMaxWidth, 99U + rank,
                                {{qw_rows, 0, 2.0F}, {qw_rows + kv, -1, 0.0F}, {2 * qw_rows + kv, 0, 2.0F},
                                 {n, rank, 1.5F}},
                                b_d, u_d, rank);
    const auto x = random_bf16(static_cast<std::size_t>(k) * kMaxWidth, 36U, -1.0F, 1.0F);
    test::GuardedDeviceBuffer x_d(x.size() * 2), q_d(static_cast<std::size_t>(qw_rows) * kMaxWidth * 2),
        g_d(static_cast<std::size_t>(qw_rows) * kMaxWidth * 2), k_d(static_cast<std::size_t>(kv) * kMaxWidth * 2),
        v_d(static_cast<std::size_t>(kv) * kMaxWidth * 2);
    x_d.copy_from_host(x.data(), x.size() * 2);
    for (const ops::LinearPolicy policy : policies) {
        WorkspaceArena workspace(std::max<std::size_t>(
            ops::attn_input_proj_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth), 256));
        for (const std::int32_t t : kWidths) {
            Tensor xt(x_d.data(), DType::BF16, {k, t});
            Tensor q(q_d.data(), DType::BF16, {qw_rows, t}), g(g_d.data(), DType::BF16, {qw_rows, t});
            Tensor kk(k_d.data(), DType::BF16, {kv, t}), v(v_d.data(), DType::BF16, {kv, t});
            ops::attn_input_proj(xt, weight, q, g, kk, v, policy, workspace, nullptr, &term.addend);
            test::cuda_synchronize();
            const auto read = [&](test::GuardedDeviceBuffer& d, std::int32_t rows) {
                std::vector<std::uint16_t> h(static_cast<std::size_t>(rows) * t);
                d.copy_to_host(h.data(), h.size() * 2);
                return h;
            };
            const auto qh = read(q_d, qw_rows), gh = read(g_d, qw_rows), kh = read(k_d, kv), vh = read(v_d, kv);
            std::vector<double> got, ref;
            for (std::int32_t c = 0; c < t; ++c) {
                for (std::int32_t r = 0; r < qw_rows; ++r) {
                    got.push_back(bf16(qh[static_cast<std::size_t>(c) * qw_rows + r]));
                    ref.push_back(term.value(r, c));
                    got.push_back(bf16(gh[static_cast<std::size_t>(c) * qw_rows + r]));
                    ref.push_back(term.value(qw_rows + kv + r, c));
                }
                for (std::int32_t r = 0; r < kv; ++r) {
                    got.push_back(bf16(kh[static_cast<std::size_t>(c) * kv + r]));
                    ref.push_back(0.0);
                    got.push_back(bf16(vh[static_cast<std::size_t>(c) * kv + r]));
                    ref.push_back(term.value(2 * qw_rows + kv + r, c));
                }
            }
            failures += compare("attn_input_proj term " + std::string(policy_name(policy)) + " qtype=" +
                                    std::to_string(static_cast<int>(qtype)) + " r=" + std::to_string(rank) +
                                    " T=" + std::to_string(t),
                                got, ref, 1.0 / 128.0, 1.0e-4);
        }
    }
    return failures;
}

// The paired groupwise projection (Q4 query/key, Q5 gate/value): the term over the single parent's logical rows.
int check_paired_attention_terms(std::initializer_list<ops::LinearPolicy> policies, std::int32_t rank = 32) {
    constexpr std::int32_t n = 14336, k = 5120, qw_rows = 6144, kv = 1024;
    int failures             = 0;
    const auto first         = zero_weight(QType::Q4_G64_FP16, qw_rows + kv, k, {});
    const auto second        = zero_weight(QType::Q5_G64_FP16, qw_rows + kv, k, {});
    test::GuardedDeviceBuffer w0_d(first.payload.size()), w1_d(second.payload.size());
    w0_d.copy_from_host(first.payload.data(), first.payload.size());
    w1_d.copy_from_host(second.payload.data(), second.payload.size());
    const Weight w0 = first.device_weight(w0_d.data());
    const Weight w1 = second.device_weight(w1_d.data());
    test::GuardedDeviceBuffer b_d(static_cast<std::size_t>(n) * rank * 2),
        u_d(static_cast<std::size_t>(2 * rank) * kMaxWidth * 2);
    // As check_attention_terms: query and gate share U rows 0 .. rank - 1, key unadapted, value the next rank rows.
    const Term term = make_term(n, 2 * rank, kMaxWidth, 101U + rank,
                                {{qw_rows, 0, 2.0F}, {qw_rows + kv, -1, 0.0F}, {2 * qw_rows + kv, 0, 2.0F},
                                 {n, rank, 1.5F}},
                                b_d, u_d, rank);
    const auto x = random_bf16(static_cast<std::size_t>(k) * kMaxWidth, 37U, -1.0F, 1.0F);
    test::GuardedDeviceBuffer x_d(x.size() * 2), q_d(static_cast<std::size_t>(qw_rows) * kMaxWidth * 2),
        g_d(static_cast<std::size_t>(qw_rows) * kMaxWidth * 2), k_d(static_cast<std::size_t>(kv) * kMaxWidth * 2),
        v_d(static_cast<std::size_t>(kv) * kMaxWidth * 2);
    x_d.copy_from_host(x.data(), x.size() * 2);
    for (const ops::LinearPolicy policy : policies) {
        for (const std::int32_t t : kWidths) {
            Tensor xt(x_d.data(), DType::BF16, {k, t});
            Tensor q(q_d.data(), DType::BF16, {qw_rows, t}), g(g_d.data(), DType::BF16, {qw_rows, t});
            Tensor kk(k_d.data(), DType::BF16, {kv, t}), v(v_d.data(), DType::BF16, {kv, t});
            ops::attn_input_proj(xt, w0, w1, q, g, kk, v, policy, nullptr, &term.addend);
            test::cuda_synchronize();
            const auto read = [&](test::GuardedDeviceBuffer& d, std::int32_t rows) {
                std::vector<std::uint16_t> h(static_cast<std::size_t>(rows) * t);
                d.copy_to_host(h.data(), h.size() * 2);
                return h;
            };
            const auto qh = read(q_d, qw_rows), gh = read(g_d, qw_rows), kh = read(k_d, kv), vh = read(v_d, kv);
            std::vector<double> got, ref;
            for (std::int32_t c = 0; c < t; ++c) {
                for (std::int32_t r = 0; r < qw_rows; ++r) {
                    got.push_back(bf16(qh[static_cast<std::size_t>(c) * qw_rows + r]));
                    ref.push_back(term.value(r, c));
                    got.push_back(bf16(gh[static_cast<std::size_t>(c) * qw_rows + r]));
                    ref.push_back(term.value(qw_rows + kv + r, c));
                }
                for (std::int32_t r = 0; r < kv; ++r) {
                    got.push_back(bf16(kh[static_cast<std::size_t>(c) * kv + r]));
                    ref.push_back(0.0);
                    got.push_back(bf16(vh[static_cast<std::size_t>(c) * kv + r]));
                    ref.push_back(term.value(2 * qw_rows + kv + r, c));
                }
            }
            failures += compare("paired attn_input_proj term " + std::string(policy_name(policy)) +
                                    " r=" + std::to_string(rank) + " T=" + std::to_string(t),
                                got, ref, 1.0 / 128.0, 1.0e-4);
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------------------------
// A trained adapter's initial state (B = 0, PEFT's init) on non-zero weights: the adapter instance of every route
// must return the no-adapter call's bits (the term is +0 and the kernel's own arithmetic is unchanged). Each Op is
// called with and without the addend at every width; a difference names the route's width.
enum class TermOp { LinearAdd, SwiGlu, Attention };

int check_zero_term_identity(TermOp op, QType qtype, std::int32_t n, std::int32_t k, qw::PatternedWeightOptions options,
                             std::initializer_list<ops::LinearPolicy> policies, ops::LinearRoutes routes) {
    int failures       = 0;
    const auto packed  = qw::make_patterned_weight(qtype, n, k, 77U, options);
    test::GuardedDeviceBuffer w_d(packed.payload.size());
    w_d.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight = packed.device_weight(w_d.data());
    // Rows of every segment kind: the term's B is zero, U random.
    const std::int32_t u_rows = 64;
    test::GuardedDeviceBuffer b_d(static_cast<std::size_t>(n) * 32 * 2), u_d(static_cast<std::size_t>(u_rows) * kMaxWidth * 2);
    std::vector<std::uint16_t> zeros(static_cast<std::size_t>(n) * 32, 0);
    b_d.copy_from_host(zeros.data(), zeros.size() * 2);
    const auto u = random_bf16(static_cast<std::size_t>(u_rows) * kMaxWidth, 501U, -3.0F, 3.0F);
    u_d.copy_from_host(u.data(), u.size() * 2);
    ops::LoraAddend addend;
    addend.b      = b_d.data();
    addend.rank   = 32;
    addend.u      = u_d.data();
    addend.u_rows = u_rows;
    if (op == TermOp::Attention) {
        addend.segments = 4;
        addend.segment  = {ops::LoraSegment{6144, 0, 2.0F}, ops::LoraSegment{7168, 32, 2.0F},
                           ops::LoraSegment{13312, 0, 2.0F}, ops::LoraSegment{14336, 32, 2.0F}};
    } else if (op == TermOp::SwiGlu) {
        addend.segments = 2;
        addend.segment  = {ops::LoraSegment{n / 2, 0, 2.0F}, ops::LoraSegment{n, 32, 0.5F}, {}, {}};
    } else {
        addend.segments = 1;
        addend.segment  = {ops::LoraSegment{n, 0, 2.0F}, {}, {}, {}};
    }
    const auto x        = random_bf16(static_cast<std::size_t>(k) * kMaxWidth, 503U, -1.0F, 1.0F);
    const auto residual = random_bf16(static_cast<std::size_t>(5120) * kMaxWidth, 505U, -2.0F, 2.0F);
    const std::int32_t out_rows = op == TermOp::SwiGlu ? n / 2 : n;
    test::GuardedDeviceBuffer x_d(x.size() * 2), y_d(static_cast<std::size_t>(out_rows) * kMaxWidth * 2);
    x_d.copy_from_host(x.data(), x.size() * 2);
    // linear_add also runs on a cancelling residual, minus the projection's own BF16 output: the sum then keeps the
    // product's low bits, so an epilogue that rounds the product where the no-adapter kernel fuses it into the
    // residual's add (one FMA) shows in most elements instead of a few.
    std::vector<std::uint16_t> cancel;
    if (op == TermOp::LinearAdd) {
        const ops::LinearPolicy policy = *policies.begin();
        WorkspaceArena workspace(
            std::max<std::size_t>(ops::linear_add_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth), 256));
        cancel.assign(static_cast<std::size_t>(n) * kMaxWidth, 0);
        y_d.copy_from_host(cancel.data(), cancel.size() * 2);
        Tensor xt(x_d.data(), DType::BF16, {k, kMaxWidth});
        Tensor yt(y_d.data(), DType::BF16, {n, kMaxWidth});
        ops::linear_add(xt, weight, yt, policy, workspace, nullptr, routes, nullptr);
        test::cuda_synchronize();
        y_d.copy_to_host(cancel.data(), cancel.size() * 2);
        for (std::uint16_t& bits : cancel) { bits ^= 0x8000U; }
    }
    for (const ops::LinearPolicy policy : policies) {
        const std::size_t capacity =
            op == TermOp::LinearAdd ? ops::linear_add_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth)
            : op == TermOp::SwiGlu  ? ops::linear_swiglu_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth, routes)
                                    : ops::attn_input_proj_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        for (int pass = 0; pass < (op == TermOp::LinearAdd ? 2 : 1); ++pass)
        for (const std::int32_t t : kWidths) {
            std::vector<std::uint16_t> outputs[2];
            for (int with = 0; with < 2; ++with) {
                const ops::LoraAddend* term = with != 0 ? &addend : nullptr;
                Tensor xt(x_d.data(), DType::BF16, {k, t});
                if (op == TermOp::LinearAdd) {
                    y_d.copy_from_host(pass == 0 ? residual.data() : cancel.data(), static_cast<std::size_t>(n) * t * 2);
                    Tensor yt(y_d.data(), DType::BF16, {n, t});
                    ops::linear_add(xt, weight, yt, policy, workspace, nullptr, routes, term);
                } else if (op == TermOp::SwiGlu) {
                    Tensor yt(y_d.data(), DType::BF16, {out_rows, t});
                    ops::linear_swiglu(xt, weight, yt, policy, workspace, nullptr, routes, term);
                } else {
                    // q, gate, k, v in one buffer: [6144 | 6144 | 1024 | 1024] x t.
                    auto* base = static_cast<std::uint16_t*>(y_d.data());
                    Tensor q(base, DType::BF16, {6144, t}), g(base + 6144 * t, DType::BF16, {6144, t});
                    Tensor kk(base + 12288 * t, DType::BF16, {1024, t}), v(base + 13312 * t, DType::BF16, {1024, t});
                    ops::attn_input_proj(xt, weight, q, g, kk, v, policy, workspace, nullptr, term);
                }
                test::cuda_synchronize();
                outputs[with].resize(static_cast<std::size_t>(out_rows) * t);
                y_d.copy_to_host(outputs[with].data(), outputs[with].size() * 2);
            }
            failures += compare_bits(std::string(op == TermOp::LinearAdd ? "linear_add" : op == TermOp::SwiGlu
                                                                                ? "linear_swiglu"
                                                                                : "attn_input_proj") +
                                         " zero-B identity " + (pass == 1 ? "cancelling " : "") + policy_name(policy) +
                                         " qtype=" +
                                         std::to_string(static_cast<int>(qtype)) + " K=" + std::to_string(k) +
                                         " T=" + std::to_string(t),
                                     outputs[1].data(), outputs[0].data(), outputs[0].size());
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------------------------
// The wide class against the narrow one: a rank-32 term (random B and U) and the same term zero-padded to rank 128
// (B's columns 32-127 zero, U's rows 32-127 of each segment random) run the narrow and the wide instances of every
// route; the chain over the padded rank adds exact zeros, so both must give the same bits, on non-zero weights, at
// every width.
int check_padded_term_identity(TermOp op, QType qtype, std::int32_t n, std::int32_t k,
                               qw::PatternedWeightOptions options, std::initializer_list<ops::LinearPolicy> policies,
                               ops::LinearRoutes routes) {
    constexpr std::int32_t kNarrow = 32, kWide = 128;
    int failures      = 0;
    const auto packed = qw::make_patterned_weight(qtype, n, k, 79U, options);
    test::GuardedDeviceBuffer w_d(packed.payload.size());
    w_d.copy_from_host(packed.payload.data(), packed.payload.size());
    const Weight weight = packed.device_weight(w_d.data());
    // Segments in rank units: their U offsets are 0 or 1 rank (-1: unadapted).
    std::vector<std::pair<std::int32_t, std::int32_t>> segments; // (row_end, u_offset in ranks)
    float scales[4] = {2.0F, 2.0F, 2.0F, 1.5F};
    if (op == TermOp::Attention) {
        segments = {{6144, 0}, {7168, -1}, {13312, 0}, {14336, 1}};
    } else if (op == TermOp::SwiGlu) {
        segments = {{n / 2, 0}, {n, 1}};
        scales[1] = 0.5F;
    } else {
        segments = {{n, 0}};
    }
    const std::int32_t slots = op == TermOp::LinearAdd ? 1 : 2;
    const auto b32 = random_bf16(static_cast<std::size_t>(n) * kNarrow, 601U, -0.05F, 0.05F);
    const auto u32 = random_bf16(static_cast<std::size_t>(slots * kNarrow) * kMaxWidth, 603U, -3.0F, 3.0F);
    const auto pad = random_bf16(static_cast<std::size_t>(slots * kWide) * kMaxWidth, 605U, -3.0F, 3.0F);
    std::vector<std::uint16_t> b128(static_cast<std::size_t>(n) * kWide, 0), u128(pad);
    for (std::int32_t row = 0; row < n; ++row) {
        std::copy_n(&b32[static_cast<std::size_t>(row) * kNarrow], kNarrow, &b128[static_cast<std::size_t>(row) * kWide]);
    }
    for (std::int32_t token = 0; token < kMaxWidth; ++token) {
        for (std::int32_t s = 0; s < slots; ++s) {
            std::copy_n(&u32[static_cast<std::size_t>(token) * slots * kNarrow + s * kNarrow], kNarrow,
                        &u128[static_cast<std::size_t>(token) * slots * kWide + s * kWide]);
        }
    }
    test::GuardedDeviceBuffer b32_d(b32.size() * 2), u32_d(u32.size() * 2), b128_d(b128.size() * 2), u128_d(u128.size() * 2);
    b32_d.copy_from_host(b32.data(), b32.size() * 2);
    u32_d.copy_from_host(u32.data(), u32.size() * 2);
    b128_d.copy_from_host(b128.data(), b128.size() * 2);
    u128_d.copy_from_host(u128.data(), u128.size() * 2);
    const auto addend_of = [&](std::int32_t rank, test::GuardedDeviceBuffer& b_d, test::GuardedDeviceBuffer& u_d) {
        ops::LoraAddend addend;
        addend.b      = b_d.data();
        addend.rank   = rank;
        addend.u      = u_d.data();
        addend.u_rows = slots * rank;
        for (std::size_t s = 0; s < segments.size(); ++s) {
            const auto [row_end, slot] = segments[s];
            addend.segment[s] = ops::LoraSegment{row_end, slot < 0 ? -1 : slot * rank, slot < 0 ? 0.0F : scales[s]};
        }
        addend.segments = static_cast<std::int32_t>(segments.size());
        return addend;
    };
    const ops::LoraAddend narrow = addend_of(kNarrow, b32_d, u32_d), wide = addend_of(kWide, b128_d, u128_d);
    const auto x        = random_bf16(static_cast<std::size_t>(k) * kMaxWidth, 607U, -1.0F, 1.0F);
    const auto residual = random_bf16(static_cast<std::size_t>(5120) * kMaxWidth, 609U, -2.0F, 2.0F);
    const std::int32_t out_rows = op == TermOp::SwiGlu ? n / 2 : n;
    test::GuardedDeviceBuffer x_d(x.size() * 2), y_d(static_cast<std::size_t>(out_rows) * kMaxWidth * 2);
    x_d.copy_from_host(x.data(), x.size() * 2);
    for (const ops::LinearPolicy policy : policies) {
        const std::size_t capacity =
            op == TermOp::LinearAdd ? ops::linear_add_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth)
            : op == TermOp::SwiGlu  ? ops::linear_swiglu_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth, routes)
                                    : ops::attn_input_proj_workspace_capacity_bytes(qtype, n, k, policy, 1, kMaxWidth);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        for (const std::int32_t t : kWidths) {
            std::vector<std::uint16_t> outputs[2];
            for (int with = 0; with < 2; ++with) {
                const ops::LoraAddend* term = with != 0 ? &wide : &narrow;
                Tensor xt(x_d.data(), DType::BF16, {k, t});
                if (op == TermOp::LinearAdd) {
                    y_d.copy_from_host(residual.data(), static_cast<std::size_t>(n) * t * 2);
                    Tensor yt(y_d.data(), DType::BF16, {n, t});
                    ops::linear_add(xt, weight, yt, policy, workspace, nullptr, routes, term);
                } else if (op == TermOp::SwiGlu) {
                    Tensor yt(y_d.data(), DType::BF16, {out_rows, t});
                    ops::linear_swiglu(xt, weight, yt, policy, workspace, nullptr, routes, term);
                } else {
                    auto* base = static_cast<std::uint16_t*>(y_d.data());
                    Tensor q(base, DType::BF16, {6144, t}), g(base + 6144 * t, DType::BF16, {6144, t});
                    Tensor kk(base + 12288 * t, DType::BF16, {1024, t}), v(base + 13312 * t, DType::BF16, {1024, t});
                    ops::attn_input_proj(xt, weight, q, g, kk, v, policy, workspace, nullptr, term);
                }
                test::cuda_synchronize();
                outputs[with].resize(static_cast<std::size_t>(out_rows) * t);
                y_d.copy_to_host(outputs[with].data(), outputs[with].size() * 2);
            }
            failures += compare_bits(std::string(op == TermOp::LinearAdd ? "linear_add" : op == TermOp::SwiGlu
                                                                                ? "linear_swiglu"
                                                                                : "attn_input_proj") +
                                         " rank 32 padded to 128 (wide = narrow) " + policy_name(policy) + " qtype=" +
                                         std::to_string(static_cast<int>(qtype)) + " K=" + std::to_string(k) +
                                         " T=" + std::to_string(t),
                                     outputs[1].data(), outputs[0].data(), outputs[0].size());
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------------------------
// Non-zero weights under A16ColumnInvariant: every column's output bits are the same in every call width (a
// request's column sits at any batch position). The widest call runs first; each narrower call's columns are compared
// with the same columns of it. Each output is column-major (column c's rows at c * rows), so a call of t columns is
// the widest's first rows * t elements of each output.
int check_invariance() {
    int failures = 0;
    constexpr std::int32_t widths[] = {129, 96, 64, 33, 32, 31, 17, 16, 8, 5, 2, 1};
    constexpr std::int32_t kMax     = 129;
    // outputs: one host vector per output tensor, rows[i] rows each.
    const auto compare_widths = [&](const std::string& label, std::int32_t t, const std::vector<std::int32_t>& rows,
                                    const std::vector<std::vector<std::uint16_t>>& outputs,
                                    std::vector<std::vector<std::uint16_t>>& widest) {
        if (t == kMax) {
            widest = outputs;
            return;
        }
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            failures += compare_bits(label + " output " + std::to_string(i) + " T=" + std::to_string(t), outputs[i].data(),
                                     widest[i].data(), static_cast<std::size_t>(rows[i]) * t);
        }
    };
    const auto x                    = random_bf16(static_cast<std::size_t>(17408) * kMax, 55U, -1.0F, 1.0F);
    test::GuardedDeviceBuffer x_d(x.size() * 2);
    x_d.copy_from_host(x.data(), x.size() * 2);
    const auto policy = ops::LinearPolicy::A16ColumnInvariant;
    // linear_add, NVFP4, FP8 and groupwise Q5, K 6144 and 17408.
    for (const QType qtype : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16, QType::FP8_E4M3FN_ROW_F32, QType::Q5_G64_FP16}) {
        for (const std::int32_t k : {6144, 17408}) {
            // FP8 row-F32: the QAD recipe's attention output (K 6144) with its static activation scale.
            if (qtype == QType::FP8_E4M3FN_ROW_F32 && k != 6144) { continue; }
            qw::PatternedWeightOptions options;
            if (qtype == QType::NVFP4) { options = {.weight_scale_divisor = 0.125F, .input_scale_divisor = 3.5F}; }
            if (qtype == QType::FP8_E4M3FN_ROW_F32) { options = {.activation_scale = 1.0F / 256.0F}; }
            const auto packed = qw::make_patterned_weight(qtype, 5120, k, 5U, options);
            test::GuardedDeviceBuffer w_d(packed.payload.size());
            w_d.copy_from_host(packed.payload.data(), packed.payload.size());
            const Weight weight = packed.device_weight(w_d.data());
            test::GuardedDeviceBuffer b_d(5120 * 32 * 2), u_d(32 * kMax * 2), r_d(5120 * kMax * 2);
            const Term term = make_term(5120, 32, kMax, 3U, {{5120, 0, 2.0F}}, b_d, u_d);
            const auto residual = random_bf16(5120 * kMax, 8U, -2.0F, 2.0F);
            WorkspaceArena workspace(std::max<std::size_t>(
                ops::linear_add_workspace_capacity_bytes(qtype, 5120, k, policy, 1, kMax), 256));
            std::vector<std::vector<std::uint16_t>> widest;
            for (const std::int32_t t : widths) {
                r_d.copy_from_host(residual.data(), 5120 * static_cast<std::size_t>(t) * 2);
                Tensor xt(x_d.data(), DType::BF16, {k, t});
                Tensor rt(r_d.data(), DType::BF16, {5120, t});
                ops::linear_add(xt, weight, rt, policy, workspace, nullptr, ops::LinearRoutes::Exact, &term.addend);
                test::cuda_synchronize();
                std::vector<std::vector<std::uint16_t>> out(1, std::vector<std::uint16_t>(5120 * static_cast<std::size_t>(t)));
                r_d.copy_to_host(out[0].data(), out[0].size() * 2);
                compare_widths("linear_add invariance qtype=" + std::to_string(static_cast<int>(qtype)) + " K=" +
                                   std::to_string(k),
                               t, {5120}, out, widest);
            }
        }
    }
    // linear_swiglu NVFP4 and groupwise Q4.
    for (const QType qtype : {QType::NVFP4, QType::Q4_G64_FP16}) {
        qw::PatternedWeightOptions options;
        if (qtype == QType::NVFP4) { options = {.weight_scale_divisor = 1.0F, .input_scale_divisor = 6.0F}; }
        const auto packed = qw::make_patterned_weight(qtype, 34816, 5120, 6U, options);
        test::GuardedDeviceBuffer w_d(packed.payload.size());
        w_d.copy_from_host(packed.payload.data(), packed.payload.size());
        const Weight weight = packed.device_weight(w_d.data());
        test::GuardedDeviceBuffer b_d(34816 * 32 * 2), u_d(64 * kMax * 2), y_d(17408 * kMax * 2);
        const Term term = make_term(34816, 64, kMax, 4U, {{17408, 0, 2.0F}, {34816, 32, 0.5F}}, b_d, u_d);
        WorkspaceArena workspace(std::max<std::size_t>(
            ops::linear_swiglu_workspace_capacity_bytes(qtype, 34816, 5120, policy, 1, kMax), 256));
        std::vector<std::vector<std::uint16_t>> widest;
        for (const std::int32_t t : widths) {
            Tensor xt(x_d.data(), DType::BF16, {5120, t});
            Tensor yt(y_d.data(), DType::BF16, {17408, t});
            ops::linear_swiglu(xt, weight, yt, policy, workspace, nullptr, ops::LinearRoutes::Exact, &term.addend);
            test::cuda_synchronize();
            std::vector<std::vector<std::uint16_t>> out(1, std::vector<std::uint16_t>(17408 * static_cast<std::size_t>(t)));
            y_d.copy_to_host(out[0].data(), out[0].size() * 2);
            compare_widths("linear_swiglu invariance qtype=" + std::to_string(static_cast<int>(qtype)), t, {17408}, out,
                           widest);
        }
    }
    // attn_input_proj NVFP4 and FP8.
    for (const QType qtype : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16, QType::FP8_E4M3FN_ROW_F32}) {
        qw::PatternedWeightOptions options;
        if (qtype == QType::NVFP4) { options = {.weight_scale_divisor = 1.0F, .input_scale_divisor = 6.0F}; }
        const auto packed = qw::make_patterned_weight(qtype, 14336, 5120, 7U, options);
        test::GuardedDeviceBuffer w_d(packed.payload.size());
        w_d.copy_from_host(packed.payload.data(), packed.payload.size());
        const Weight weight = packed.device_weight(w_d.data());
        test::GuardedDeviceBuffer b_d(14336 * 32 * 2), u_d(96 * kMax * 2), q_d(6144 * kMax * 2), g_d(6144 * kMax * 2),
            k_d(1024 * kMax * 2), v_d(1024 * kMax * 2);
        const Term term =
            make_term(14336, 96, kMax, 5U, {{6144, 0, 2.0F}, {7168, 32, 2.0F}, {13312, 0, 2.0F}, {14336, 64, 2.0F}},
                      b_d, u_d);
        WorkspaceArena workspace(std::max<std::size_t>(
            ops::attn_input_proj_workspace_capacity_bytes(qtype, 14336, 5120, policy, 1, kMax), 256));
        std::vector<std::vector<std::uint16_t>> widest;
        for (const std::int32_t t : widths) {
            Tensor xt(x_d.data(), DType::BF16, {5120, t});
            Tensor q(q_d.data(), DType::BF16, {6144, t}), g(g_d.data(), DType::BF16, {6144, t});
            Tensor kk(k_d.data(), DType::BF16, {1024, t}), v(v_d.data(), DType::BF16, {1024, t});
            ops::attn_input_proj(xt, weight, q, g, kk, v, policy, workspace, nullptr, &term.addend);
            test::cuda_synchronize();
            std::vector<std::vector<std::uint16_t>> out{std::vector<std::uint16_t>(6144 * static_cast<std::size_t>(t)),
                                                       std::vector<std::uint16_t>(6144 * static_cast<std::size_t>(t)),
                                                       std::vector<std::uint16_t>(1024 * static_cast<std::size_t>(t)),
                                                       std::vector<std::uint16_t>(1024 * static_cast<std::size_t>(t))};
            q_d.copy_to_host(out[0].data(), out[0].size() * 2);
            g_d.copy_to_host(out[1].data(), out[1].size() * 2);
            k_d.copy_to_host(out[2].data(), out[2].size() * 2);
            v_d.copy_to_host(out[3].data(), out[3].size() * 2);
            compare_widths("attn_input_proj invariance qtype=" + std::to_string(static_cast<int>(qtype)), t,
                           {6144, 6144, 1024, 1024}, out, widest);
        }
    }
    // The paired groupwise projection.
    {
        const auto q4 = qw::make_patterned_weight(QType::Q4_G64_FP16, 7168, 5120, 8U);
        const auto q5 = qw::make_patterned_weight(QType::Q5_G64_FP16, 7168, 5120, 9U);
        test::GuardedDeviceBuffer w0_d(q4.payload.size()), w1_d(q5.payload.size());
        w0_d.copy_from_host(q4.payload.data(), q4.payload.size());
        w1_d.copy_from_host(q5.payload.data(), q5.payload.size());
        const Weight w0 = q4.device_weight(w0_d.data()), w1 = q5.device_weight(w1_d.data());
        test::GuardedDeviceBuffer b_d(14336 * 32 * 2), u_d(96 * kMax * 2), q_d(6144 * kMax * 2), g_d(6144 * kMax * 2),
            k_d(1024 * kMax * 2), v_d(1024 * kMax * 2);
        const Term term =
            make_term(14336, 96, kMax, 10U, {{6144, 0, 2.0F}, {7168, 32, 2.0F}, {13312, 0, 2.0F}, {14336, 64, 2.0F}},
                      b_d, u_d);
        std::vector<std::vector<std::uint16_t>> widest;
        for (const std::int32_t t : widths) {
            Tensor xt(x_d.data(), DType::BF16, {5120, t});
            Tensor q(q_d.data(), DType::BF16, {6144, t}), g(g_d.data(), DType::BF16, {6144, t});
            Tensor kk(k_d.data(), DType::BF16, {1024, t}), v(v_d.data(), DType::BF16, {1024, t});
            ops::attn_input_proj(xt, w0, w1, q, g, kk, v, policy, nullptr, &term.addend);
            test::cuda_synchronize();
            std::vector<std::vector<std::uint16_t>> out{std::vector<std::uint16_t>(6144 * static_cast<std::size_t>(t)),
                                                       std::vector<std::uint16_t>(6144 * static_cast<std::size_t>(t)),
                                                       std::vector<std::uint16_t>(1024 * static_cast<std::size_t>(t)),
                                                       std::vector<std::uint16_t>(1024 * static_cast<std::size_t>(t))};
            q_d.copy_to_host(out[0].data(), out[0].size() * 2);
            g_d.copy_to_host(out[1].data(), out[1].size() * 2);
            k_d.copy_to_host(out[2].data(), out[2].size() * 2);
            v_d.copy_to_host(out[3].data(), out[3].size() * 2);
            compare_widths("paired attn_input_proj invariance", t, {6144, 6144, 1024, 1024}, out, widest);
        }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(lora) {
    if (test::cuda_unavailable()) { return 0; }
    int failures = 0;
    using P      = ops::LinearPolicy;
    // Each check counts an exception as one failure and the run goes on (a refused profile names itself).
    const auto run = [&](const char* name, const auto& check) {
        try {
            failures += check();
        } catch (const std::exception& error) {
            std::cerr << "lora test " << name << ": " << error.what() << '\n';
            ++failures;
        }
    };
    // Non-unit divisors, so a term added after the projection's scale (instead of before) shows; and the QAD recipe's
    // multiplier form (ModelOpt's FP32 multiplier, run as NVFP4).
    const qw::PatternedWeightOptions nvfp4{.weight_scale_divisor = 0.125F, .input_scale_divisor = 3.5F};
    const qw::PatternedWeightOptions nvfp4_mul{.activation_scale = 0.0625F, .weight_multiplier = 3.0F};
    // FP8 row-F32 parents with a static activation scale, as the QAD recipe's (a per-token parent's A8 route serves
    // the 122B's K 3072 / 8192 only).
    const qw::PatternedWeightOptions fp8_static{.activation_scale = 1.0F / 256.0F};
    run("shrink", [&] { return check_shrink(); });
    // The Gated DeltaNet control's adapted form (narrow and wide ranks, b adapted or not).
    for (const std::int32_t rank : {32, 128}) {
        run("gdn gating", [&] { return check_gating_terms(rank, true); });
    }
    run("gdn gating a only", [&] { return check_gating_terms(64, false); });
    // linear_add: NVFP4 down/output (every route), FP8 row-scaled (BF16 and FP32 row words).
    for (const std::int32_t k : {6144, 17408}) {
        run("linear_add nvfp4", [&] {
            return check_linear_add_terms(QType::NVFP4, k, nvfp4,
                                          {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant});
        });
        run("linear_add fp8 bf16", [&] {
            return check_linear_add_terms(QType::FP8_E4M3FN_ROW_BF16, k, {},
                                          {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
        });
    }
    run("linear_add fp8 f32", [&] {
        return check_linear_add_terms(QType::FP8_E4M3FN_ROW_F32, 6144, fp8_static,
                                      {P::A16Only, P::AllowA8, P::A8Only, P::A16ColumnInvariant});
    });
    run("linear_add nvfp4 multiplier", [&] {
        return check_linear_add_terms(QType::NVFP4, 17408, nvfp4_mul,
                                      {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A16ColumnInvariant});
    });
    run("linear_swiglu nvfp4 multiplier", [&] {
        return check_linear_swiglu_terms(QType::NVFP4, nvfp4_mul,
                                         {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A16ColumnInvariant});
    });
    run("linear_swiglu nvfp4", [&] {
        return check_linear_swiglu_terms(QType::NVFP4, nvfp4,
                                         {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant});
    });
    run("linear_swiglu fp8", [&] {
        return check_linear_swiglu_terms(QType::FP8_E4M3FN_ROW_BF16, {},
                                         {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
    });
    // The staged term operands beyond rank 32: 48 (an odd last k-step: one ldmatrix of two matrices) and 64 (the
    // largest; paired gate/up rows' two segments then take two passes on the A4 tile), on the routes that stage them
    // (NVFP4 / FP8 / Q4 / Q5 decode and small-T rows, the A4 tile).
    for (const std::int32_t rank : {48, 64}) {
        run("linear_add nvfp4 rank", [&] {
            return check_linear_add_terms(QType::NVFP4, 17408, nvfp4, {P::A16Only, P::AllowA4}, rank);
        });
        run("linear_swiglu nvfp4 rank", [&] {
            return check_linear_swiglu_terms(QType::NVFP4, nvfp4, {P::A16Only, P::AllowA4}, ops::LinearRoutes::Exact,
                                             rank);
        });
        run("linear_add fp8 rank", [&] {
            return check_linear_add_terms(QType::FP8_E4M3FN_ROW_F32, 6144, fp8_static, {P::A16Only}, rank);
        });
        run("linear_add q5 rank", [&] { return check_linear_add_terms(QType::Q5_G64_FP16, 17408, {}, {P::A16Only}, rank); });
        run("linear_swiglu q4 rank", [&] {
            return check_linear_swiglu_terms(QType::Q4_G64_FP16, {}, {P::A16Only}, ops::LinearRoutes::Exact, rank);
        });
    }
    // The wide class (padded rank 80 to 128): every route at 128 (nonzero factors past the narrow class's 64), and
    // 80 (an odd last k-step) and 96 (the Harness Hub super-adapter's) on the routes that stage the operands.
    run("linear_add nvfp4 wide", [&] {
        return check_linear_add_terms(QType::NVFP4, 17408, nvfp4,
                                      {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant}, 128);
    });
    run("linear_add fp8 bf16 wide", [&] {
        return check_linear_add_terms(QType::FP8_E4M3FN_ROW_BF16, 6144, {},
                                      {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant}, 128);
    });
    run("linear_add fp8 f32 wide", [&] {
        return check_linear_add_terms(QType::FP8_E4M3FN_ROW_F32, 6144, fp8_static,
                                      {P::A16Only, P::AllowA8, P::A8Only, P::A16ColumnInvariant}, 128);
    });
    run("linear_add q5 wide", [&] {
        return check_linear_add_terms(QType::Q5_G64_FP16, 17408, {}, {P::A16Only, P::A16ColumnInvariant}, 128);
    });
    run("linear_swiglu nvfp4 wide", [&] {
        return check_linear_swiglu_terms(QType::NVFP4, nvfp4,
                                         {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant},
                                         ops::LinearRoutes::Exact, 128);
    });
    run("linear_swiglu fp8 wide", [&] {
        return check_linear_swiglu_terms(QType::FP8_E4M3FN_ROW_BF16, {},
                                         {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant},
                                         ops::LinearRoutes::Exact, 128);
    });
    run("linear_swiglu q4 wide", [&] {
        return check_linear_swiglu_terms(QType::Q4_G64_FP16, {}, {P::A16Only, P::A16ColumnInvariant},
                                         ops::LinearRoutes::Exact, 128);
    });
    run("linear_swiglu q4 w4a8 wide", [&] {
        return check_linear_swiglu_terms(QType::Q4_G64_FP16, {}, {P::A16Only}, ops::LinearRoutes::Q4W4A8, 128);
    });
    run("attn nvfp4 wide", [&] {
        return check_attention_terms(QType::NVFP4, nvfp4, {P::A16Only, P::AllowA4, P::A8Only, P::A16ColumnInvariant}, 128);
    });
    run("attn fp8 bf16 wide", [&] {
        return check_attention_terms(QType::FP8_E4M3FN_ROW_BF16, {},
                                     {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant}, 128);
    });
    run("attn fp8 f32 wide", [&] {
        return check_attention_terms(QType::FP8_E4M3FN_ROW_F32, fp8_static,
                                     {P::A16Only, P::AllowA8, P::A8Only, P::A16ColumnInvariant}, 128);
    });
    run("paired attn wide", [&] { return check_paired_attention_terms({P::A16Only, P::A16ColumnInvariant}, 128); });
    run("linear_add nvfp4 K6144 wide", [&] {
        return check_linear_add_terms(QType::NVFP4, 6144, nvfp4,
                                      {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant}, 128);
    });
    run("linear_add fp8 bf16 K17408 wide", [&] {
        return check_linear_add_terms(QType::FP8_E4M3FN_ROW_BF16, 17408, {},
                                      {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant}, 128);
    });
    run("linear_add q5 K6144 wide", [&] {
        return check_linear_add_terms(QType::Q5_G64_FP16, 6144, {}, {P::A16Only, P::A16ColumnInvariant}, 128);
    });
    for (const std::int32_t rank : {80, 96}) {
        run("linear_add nvfp4 wide rank", [&] {
            return check_linear_add_terms(QType::NVFP4, 17408, nvfp4, {P::A16Only, P::AllowA4}, rank);
        });
        run("linear_swiglu nvfp4 wide rank", [&] {
            return check_linear_swiglu_terms(QType::NVFP4, nvfp4, {P::A16Only, P::AllowA4}, ops::LinearRoutes::Exact,
                                             rank);
        });
        run("linear_add fp8 wide rank", [&] {
            return check_linear_add_terms(QType::FP8_E4M3FN_ROW_F32, 6144, fp8_static, {P::A16Only}, rank);
        });
        run("linear_add q5 wide rank", [&] {
            return check_linear_add_terms(QType::Q5_G64_FP16, 17408, {}, {P::A16Only}, rank);
        });
        run("linear_swiglu q4 wide rank", [&] {
            return check_linear_swiglu_terms(QType::Q4_G64_FP16, {}, {P::A16Only}, ops::LinearRoutes::Exact, rank);
        });
    }
    run("attn nvfp4", [&] {
        return check_attention_terms(QType::NVFP4, nvfp4, {P::A16Only, P::AllowA4, P::A8Only, P::A16ColumnInvariant});
    });
    run("attn fp8 bf16", [&] {
        return check_attention_terms(QType::FP8_E4M3FN_ROW_BF16, {},
                                     {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
    });
    run("attn fp8 f32", [&] {
        return check_attention_terms(QType::FP8_E4M3FN_ROW_F32, fp8_static,
                                     {P::A16Only, P::AllowA8, P::A8Only, P::A16ColumnInvariant});
    });
    // The groupwise recipe: Q5 output and down (the release routes, q5-ksplit-mma, column-invariant), Q4 gate/up
    // (release, column-invariant, the q4-w4a8 opt-in), the Q4/Q5 attention pair.
    for (const std::int32_t k : {6144, 17408}) {
        run("linear_add q5", [&] {
            return check_linear_add_terms(QType::Q5_G64_FP16, k, {}, {P::A16Only, P::A16ColumnInvariant});
        });
    }
    run("linear_swiglu q4", [&] {
        return check_linear_swiglu_terms(QType::Q4_G64_FP16, {}, {P::A16Only, P::A16ColumnInvariant});
    });
    run("linear_swiglu q4 w4a8", [&] {
        return check_linear_swiglu_terms(QType::Q4_G64_FP16, {}, {P::A16Only}, ops::LinearRoutes::Q4W4A8);
    });
    run("paired attn", [&] { return check_paired_attention_terms({P::A16Only, P::A16ColumnInvariant}); });
    run("invariance", [&] { return check_invariance(); });
    // Zero B on non-zero weights: the adapter instance returns the no-adapter bits (QAD's MLP and attention, the
    // other NVFP4 / FP8 recipes, the groupwise recipe).
    const auto zero = [&](const char* name, TermOp op, QType qtype, std::int32_t n, std::int32_t k,
                          qw::PatternedWeightOptions options, std::initializer_list<ops::LinearPolicy> policies,
                          ops::LinearRoutes routes = ops::LinearRoutes::Exact) {
        run(name, [&] { return check_zero_term_identity(op, qtype, n, k, options, policies, routes); });
    };
    zero("zero-B swiglu nvfp4 mul", TermOp::SwiGlu, QType::NVFP4, 34816, 5120, nvfp4_mul,
         {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A16ColumnInvariant});
    zero("zero-B down nvfp4 mul", TermOp::LinearAdd, QType::NVFP4, 5120, 17408, nvfp4_mul,
         {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A16ColumnInvariant});
    zero("zero-B down nvfp4 wide", TermOp::LinearAdd, QType::NVFP4, 5120, 17408, nvfp4, {P::AllowA4},
         ops::LinearRoutes::WideDownTile);
    zero("zero-B down nvfp4", TermOp::LinearAdd, QType::NVFP4, 5120, 17408, nvfp4,
         {P::A16Only, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B o_proj nvfp4", TermOp::LinearAdd, QType::NVFP4, 5120, 6144, nvfp4,
         {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B swiglu nvfp4", TermOp::SwiGlu, QType::NVFP4, 34816, 5120, nvfp4,
         {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B o_proj fp8 bf16", TermOp::LinearAdd, QType::FP8_E4M3FN_ROW_BF16, 5120, 6144, {},
         {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B down fp8 bf16", TermOp::LinearAdd, QType::FP8_E4M3FN_ROW_BF16, 5120, 17408, {},
         {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B attn fp8 f32", TermOp::Attention, QType::FP8_E4M3FN_ROW_F32, 14336, 5120, fp8_static,
         {P::A16Only, P::AllowA8, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B o_proj fp8 f32", TermOp::LinearAdd, QType::FP8_E4M3FN_ROW_F32, 5120, 6144, fp8_static,
         {P::A16Only, P::AllowA8, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B attn nvfp4", TermOp::Attention, QType::NVFP4, 14336, 5120, nvfp4,
         {P::A16Only, P::AllowA4, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B swiglu fp8", TermOp::SwiGlu, QType::FP8_E4M3FN_ROW_BF16, 34816, 5120, {},
         {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B attn fp8 bf16", TermOp::Attention, QType::FP8_E4M3FN_ROW_BF16, 14336, 5120, {},
         {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
    zero("zero-B swiglu q4", TermOp::SwiGlu, QType::Q4_G64_FP16, 34816, 5120, {}, {P::A16Only, P::A16ColumnInvariant});
    zero("zero-B down q5", TermOp::LinearAdd, QType::Q5_G64_FP16, 5120, 17408, {}, {P::A16Only, P::A16ColumnInvariant});
    zero("zero-B down q5 ksplit", TermOp::LinearAdd, QType::Q5_G64_FP16, 5120, 17408, {}, {P::A16Only},
         ops::LinearRoutes::Q5KSplit);
    // The wide instances on a rank-32 term padded to 128 give the narrow instances' bits (every Op, format and route).
    const auto padded = [&](const char* name, TermOp op, QType qtype, std::int32_t n, std::int32_t k,
                            qw::PatternedWeightOptions options, std::initializer_list<ops::LinearPolicy> policies,
                            ops::LinearRoutes routes = ops::LinearRoutes::Exact) {
        run(name, [&] { return check_padded_term_identity(op, qtype, n, k, options, policies, routes); });
    };
    padded("padded down nvfp4", TermOp::LinearAdd, QType::NVFP4, 5120, 17408, nvfp4,
           {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant});
    padded("padded down nvfp4 wide tile", TermOp::LinearAdd, QType::NVFP4, 5120, 17408, nvfp4, {P::AllowA4},
           ops::LinearRoutes::WideDownTile);
    padded("padded o_proj fp8 bf16", TermOp::LinearAdd, QType::FP8_E4M3FN_ROW_BF16, 5120, 6144, {},
           {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
    padded("padded o_proj fp8 f32", TermOp::LinearAdd, QType::FP8_E4M3FN_ROW_F32, 5120, 6144, fp8_static,
           {P::A16Only, P::AllowA8, P::A8Only, P::A16ColumnInvariant});
    padded("padded swiglu nvfp4", TermOp::SwiGlu, QType::NVFP4, 34816, 5120, nvfp4,
           {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A8Only, P::A16ColumnInvariant});
    padded("padded swiglu nvfp4 mul", TermOp::SwiGlu, QType::NVFP4, 34816, 5120, nvfp4_mul,
           {P::A16Only, P::AllowA4, P::AllowA4Residual, P::A16ColumnInvariant});
    padded("padded swiglu fp8", TermOp::SwiGlu, QType::FP8_E4M3FN_ROW_BF16, 34816, 5120, {},
           {P::A16Only, P::AllowA8, P::AllowA8Residual, P::A8Only, P::A16ColumnInvariant});
    padded("padded attn nvfp4", TermOp::Attention, QType::NVFP4, 14336, 5120, nvfp4,
           {P::A16Only, P::AllowA4, P::A8Only, P::A16ColumnInvariant});
    padded("padded attn fp8 f32", TermOp::Attention, QType::FP8_E4M3FN_ROW_F32, 14336, 5120, fp8_static,
           {P::A16Only, P::AllowA8, P::A8Only, P::A16ColumnInvariant});
    padded("padded swiglu q4", TermOp::SwiGlu, QType::Q4_G64_FP16, 34816, 5120, {}, {P::A16Only, P::A16ColumnInvariant});
    padded("padded swiglu q4 w4a8", TermOp::SwiGlu, QType::Q4_G64_FP16, 34816, 5120, {}, {P::A16Only},
           ops::LinearRoutes::Q4W4A8);
    padded("padded down q5", TermOp::LinearAdd, QType::Q5_G64_FP16, 5120, 17408, {}, {P::A16Only, P::A16ColumnInvariant});
    padded("padded down q5 ksplit", TermOp::LinearAdd, QType::Q5_G64_FP16, 5120, 17408, {}, {P::A16Only},
           ops::LinearRoutes::Q5KSplit);
    return failures == 0 ? 0 : 1;
}
