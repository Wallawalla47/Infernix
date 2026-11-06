// ggml_linear's 8-bit activation route (LinearRoutes::GgmlA8, `--a8-activations`: LOSSY, opt-in;
// docs/maintainer/qwen3_8-flash-next-design.md §19.3.31) against an FP64 oracle.
//
// The semantic boundary is the activation codec (ops/ggml/ggml_a8.h): the test quantizes every column on the
// host with it (codes q_j and slice scales d_s, input element j read from x[perm(j)]), decodes the weights
// independently (ggml_block_oracle.h, binary64) and sums ref = sum_s d_s sum_(j in s) w_j q_j in binary64.
// Everything else the route does is FP32 rounding: every slice term is rounded once before it is added
// (fl32(dot * d), and the K-quants' min term apart from the codes' term), each addition rounds once, and an
// output's partial sums combine in at most 7 further additions (a split of 8, or a butterfly over 32 lanes).
// With n a row's slice terms (K / 32, twice that for Q4_K / Q5_K) and
//
//   M = sum_s |d_s| sum_(j in s) (|w_j| + 2 |o_j|) |q_j|        (o_j: the K-quants' subtracted dmin * min)
//
// bounding the terms' magnitudes, |y - ref| <= (n + 10) 2^-24 M + 2^-36 M (the oracle's own binary64 sums),
// plus 2^-8 (|ref| + that) for a BF16 output's one rounding. A column holding Inf or NaN must give NaN rows.
//
// Coverage: every format on ggml_rows_v1 (the dp4a rows kernel at 1-2 columns, the mma rows kernel from 3)
// and the four ggml_tiles16_v1 formats (the rows kernel for one-column Q6_K / IQ4_XS, the mma tiles
// otherwise), every width class 1-31 (one, two and four 8-column tiles a warp, partial tiles), FP32 and
// BF16 outputs, one output and outputs split at bounds inside segments, two-format (mixed) launches, the
// input permutation (16-byte and scalar encoder loads), short last units (32- and 64-element blocks), tile
// counts that reach every CTA shape, zero and sub-threshold slices, Inf and NaN, and the IQ3_S import's
// dense shapes. Each width also runs twice (the same bits) and with every other column changed (column 0's
// bits unchanged: a column does not depend on the others at a fixed shape), with a workspace of exactly
// ggml_linear_workspace_bytes. Calls of 32 columns must keep the other routes' bits.

#include "ops/op_family.h"
#include "core/device.h"
#include "core/ggml_format.h"
#include "infernix/ops/ggml_linear.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ggml_block_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace infernix;

namespace {

namespace a8 = ggml::a8;
using test::ggml_oracle::block_offsets;
using test::ggml_oracle::decode_block;

constexpr int kColumns = 32; // x's columns: the route's 31 and one for the boundary check

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << "\n";
    }
}

std::uint16_t bf16_rne(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

double from_bits(std::uint32_t bits, bool f32) {
    const std::uint32_t u = f32 ? bits : bits << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool min_format(QType q) { return q == QType::GGML_Q4_K || q == QType::GGML_Q5_K; }

std::string format_name(QType q) {
    switch (q) {
    case QType::GGML_Q2_0: return "Q2_0";
    case QType::GGML_Q4_K: return "Q4_K";
    case QType::GGML_Q5_K: return "Q5_K";
    case QType::GGML_Q6_K: return "Q6_K";
    case QType::GGML_IQ2_S: return "IQ2_S";
    case QType::GGML_IQ3_XXS: return "IQ3_XXS";
    case QType::GGML_IQ3_S: return "IQ3_S";
    case QType::GGML_IQ4_XS: return "IQ4_XS";
    case QType::GGML_IQ4_NL: return "IQ4_NL";
    default: return "?";
    }
}

template <class Fn>
void parallel_for(int n, const Fn& fn) {
    const int workers = static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1U, 32U));
    std::vector<std::thread> pool;
    for (int w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
            for (int i = w; i < n; i += workers) { fn(i); }
        });
    }
    for (auto& thread : pool) { thread.join(); }
}

// Random block bytes (every byte pattern is a valid encoding once the binary16 scales are finite), the
// scales' exponent fields 12-13 (Narrow: the GSQ-RCO import's rows) or 5-20 (Wide).
enum class Scales { Narrow, Wide };

std::vector<std::uint8_t> random_blocks(QType q, int rows, int k, std::mt19937& rng, Scales scales) {
    const int blocks = k / static_cast<int>(ggml_block_elems(q)), size = static_cast<int>(ggml_block_bytes(q));
    std::vector<std::uint8_t> data(static_cast<std::size_t>(rows) * blocks * size);
    for (std::size_t i = 0; i < data.size(); i += 4) {
        const std::uint32_t v = rng();
        for (std::size_t j = 0; j < 4 && i + j < data.size(); ++j) { data[i + j] = static_cast<std::uint8_t>(v >> (8 * j)); }
    }
    const auto scale = [&](std::uint8_t* p) {
        const unsigned field = scales == Scales::Narrow ? 12 + rng() % 2 : 5 + rng() % 16;
        const auto w         = static_cast<std::uint16_t>((rng() & 0x8000U) | (rng() & 0x3FFU) | (field << 10));
        p[0]                 = static_cast<std::uint8_t>(w & 0xFF);
        p[1]                 = static_cast<std::uint8_t>(w >> 8);
    };
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = data.data() + static_cast<std::size_t>(i) * size;
        if (q == QType::GGML_Q6_K) {
            scale(b + 208);
        } else {
            scale(b);
            if (min_format(q)) { scale(b + 2); }
        }
    }
    return data;
}

// A ggml_tiles16_v1 matrix of `rows` ggml_rows_v1 rows: tile i at 16 i row_bytes, block b of its row r at
// (16 b + r) * block_bytes, missing rows zero blocks.
std::vector<std::uint8_t> tiles16(QType q, int rows, int k, const std::uint8_t* data) {
    const std::size_t bb = ggml_block_bytes(q), blocks = static_cast<std::size_t>(k) / ggml_block_elems(q), row_bytes = blocks * bb;
    const int tiles = (rows + 15) / 16;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(tiles) * 16 * row_bytes, 0);
    for (int row = 0; row < rows; ++row) {
        for (std::size_t b = 0; b < blocks; ++b) {
            std::memcpy(out.data() + static_cast<std::size_t>(row / 16) * 16 * row_bytes + (16 * b + row % 16) * bb,
                        data + static_cast<std::size_t>(row) * row_bytes + b * bb, bb);
        }
    }
    return out;
}

// BF16 columns [k, columns]: N(0, 1) with every 97th value 64 times larger; every fourth column over
// magnitudes 2^-40 .. 2^12 with zeros, both signs.
std::vector<std::uint16_t> random_columns(int k, int columns, std::mt19937& rng) {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> exponent(-40.0F, 12.0F);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const int column = static_cast<int>(i / static_cast<std::size_t>(k));
        if (column % 4 == 3) {
            const float m = std::ldexp(1.0F, static_cast<int>(exponent(rng))) * (1.0F + normal(rng) * 0.25F);
            x[i]          = i % 17 == 0 ? std::uint16_t{0} : bf16_rne(i % 2 == 0 ? m : -m);
        } else {
            x[i] = bf16_rne(normal(rng) * (i % 97 == 0 ? 64.0F : 1.0F));
        }
    }
    return x;
}

// Edge columns: 1 a zero slice, a slice below the quantized range (2^-125: zero codes) and one just inside it
// (1.5 * 2^-120); 2 a NaN; 3 a +Inf; 4 a -Inf (each makes its column NaN); 5 values near 2^40; 6 a slice of
// equal values (every code 127) and one alternating between +-its maximum.
void make_edges(std::vector<std::uint16_t>& x, int k, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    const auto at = [&](int column, int j) -> std::uint16_t& { return x[static_cast<std::size_t>(column) * k + j]; };
    for (int j = 0; j < 32; ++j) {
        at(1, j)      = j % 2 == 0 ? std::uint16_t{0} : std::uint16_t{0x8000};
        at(1, 32 + j) = bf16_rne((j % 2 == 0 ? 1.0F : -1.0F) * std::ldexp(1.0F + 0.03F * static_cast<float>(j), -125));
        at(1, 64 + j) = bf16_rne((j % 3 == 0 ? -1.5F : 1.5F) * std::ldexp(1.0F, -120));
    }
    at(2, std::min(37, k - 1)) = 0x7FC0;
    at(3, 0)                   = 0x7F80;
    at(4, k - 1)               = 0xFF80;
    for (int j = 0; j < k; ++j) { at(5, j) = bf16_rne(normal(rng) * std::ldexp(1.0F, 40)); }
    for (int j = 0; j < 32; ++j) {
        at(6, j)      = bf16_rne(-3.25F);
        at(6, 32 + j) = bf16_rne(j % 2 == 0 ? 7.5F : -7.5F);
    }
}

// --------------------------------------------------------------------------------------------- the problems

struct Part {
    QType q;
    int rows;
};

struct Problem {
    std::string what;
    int k = 0;
    std::vector<Part> parts; // consecutive output rows, each its own matrix
    ops::GgmlLayout layout = ops::GgmlLayout::Rows;
    int pb                 = 0; // input permutation block (0: none)
    Scales scales          = Scales::Narrow;
    bool edges             = false;
};

// The codec's codes and slice scales of x's first columns (input element j at x[perm(j)]).
struct Quantized {
    std::vector<double> q; // [column][k]
    std::vector<double> d; // [column][k / 32]: NaN for a slice holding Inf or NaN
};

Quantized quantize(const std::vector<std::uint16_t>& x, int k, int columns, const std::vector<std::int32_t>& perm, int pb) {
    Quantized out;
    const int slices = k / 32;
    out.q.resize(static_cast<std::size_t>(columns) * k);
    out.d.resize(static_cast<std::size_t>(columns) * slices);
    for (int t = 0; t < columns; ++t) {
        for (int s = 0; s < slices; ++s) {
            std::uint16_t v[32];
            unsigned key = 0;
            for (int j = 0; j < 32; ++j) {
                const int e = 32 * s + j;
                v[j]        = x[static_cast<std::size_t>(t) * k + (pb > 0 ? perm[static_cast<std::size_t>(e / pb)] * pb + e % pb : e)];
                key         = std::max(key, static_cast<unsigned>(v[j] & 0x7FFFU));
            }
            const a8::SliceScale sc = a8::slice_scale(key);
            out.d[static_cast<std::size_t>(t) * slices + s] = sc.d;
            for (int j = 0; j < 32; ++j) { out.q[static_cast<std::size_t>(t) * k + 32 * s + j] = a8::code(v[j], sc.inv); }
        }
    }
    return out;
}

struct Expected {
    std::vector<double> ref, mag; // [column][row]
    std::vector<char> nan;
};

struct Prepared {
    Problem p;
    std::vector<int> start; // each part's first output row
    int n = 0;              // output rows
    std::vector<std::vector<std::uint8_t>> host; // ggml_rows_v1 bytes of each part
    std::vector<DeviceBuffer> dev;               // as `layout` stores them
    std::vector<std::int32_t> perm;
    std::vector<std::uint16_t> x, x2; // x2: columns 1.. changed
    DeviceBuffer dx, dx2, dperm;
    Expected e;
    double worst = 0.0;     // largest |y - ref| / bound over the problem's checks
    double worst_rel = 0.0; // largest |y - ref| / M
};

Expected oracle(const Prepared& p, const Quantized& xq, int columns) {
    const int k = p.p.k, slices = k / 32;
    Expected e;
    e.ref.assign(static_cast<std::size_t>(columns) * p.n, 0.0);
    e.mag.assign(e.ref.size(), 0.0);
    e.nan.assign(e.ref.size(), 0);
    parallel_for(p.n, [&](int r) {
        std::size_t part = 0;
        while (part + 1 < p.start.size() && r >= p.start[part + 1]) { ++part; }
        const QType q          = p.p.parts[part].q;
        const std::size_t bb   = ggml_block_bytes(q), be = ggml_block_elems(q);
        const std::size_t rowb = static_cast<std::size_t>(k) / be * bb;
        const std::uint8_t* row = p.host[part].data() + static_cast<std::size_t>(r - p.start[part]) * rowb;
        std::vector<double> w(static_cast<std::size_t>(k)), off(static_cast<std::size_t>(k)), a(static_cast<std::size_t>(k));
        for (std::size_t b = 0; b < static_cast<std::size_t>(k) / be; ++b) {
            decode_block(q, row + b * bb, w.data() + b * be);
            block_offsets(q, row + b * bb, off.data() + b * be);
        }
        for (int j = 0; j < k; ++j) { a[static_cast<std::size_t>(j)] = std::fabs(w[static_cast<std::size_t>(j)]) + 2.0 * std::fabs(off[static_cast<std::size_t>(j)]); }
        for (int t = 0; t < columns; ++t) {
            double ref = 0.0, mag = 0.0;
            bool nan   = false;
            for (int s = 0; s < slices; ++s) {
                const double d = xq.d[static_cast<std::size_t>(t) * slices + s];
                if (std::isnan(d)) {
                    nan = true;
                    continue;
                }
                const double* qs = xq.q.data() + static_cast<std::size_t>(t) * k + 32 * s;
                double dot = 0.0, abs_dot = 0.0;
                for (int j = 0; j < 32; ++j) {
                    dot += w[static_cast<std::size_t>(32 * s + j)] * qs[j];
                    abs_dot += a[static_cast<std::size_t>(32 * s + j)] * std::fabs(qs[j]);
                }
                ref += dot * d;
                mag += abs_dot * std::fabs(d);
            }
            const std::size_t i = static_cast<std::size_t>(t) * p.n + r;
            e.ref[i]            = ref;
            e.mag[i]            = mag;
            e.nan[i]            = nan ? 1 : 0;
        }
    });
    return e;
}

Prepared prepare(const Problem& problem, std::mt19937& rng) {
    Prepared p;
    p.p = problem;
    const int k = problem.k;
    for (const Part& part : problem.parts) {
        p.start.push_back(p.n);
        p.n += part.rows;
        p.host.push_back(random_blocks(part.q, part.rows, k, rng, problem.scales));
        const auto bytes = problem.layout == ops::GgmlLayout::Tiles16 ? tiles16(part.q, part.rows, k, p.host.back().data()) : p.host.back();
        p.dev.emplace_back(bytes.size());
        p.dev.back().copy_from_host(bytes.data(), bytes.size(), 0);
    }
    p.x = random_columns(k, kColumns, rng);
    if (problem.edges) { make_edges(p.x, k, rng); }
    p.x2 = p.x;
    const auto other = random_columns(k, kColumns, rng);
    std::copy(other.begin() + k, other.end(), p.x2.begin() + k); // every column but 0
    if (problem.pb > 0) {
        p.perm.resize(static_cast<std::size_t>(k / problem.pb));
        for (std::size_t i = 0; i < p.perm.size(); ++i) { p.perm[i] = static_cast<std::int32_t>(i); }
        std::shuffle(p.perm.begin(), p.perm.end(), rng);
        p.dperm = DeviceBuffer(p.perm.size() * 4);
        p.dperm.copy_from_host(p.perm.data(), p.perm.size() * 4, 0);
    }
    p.dx  = DeviceBuffer(p.x.size() * 2);
    p.dx2 = DeviceBuffer(p.x2.size() * 2);
    p.dx.copy_from_host(p.x.data(), p.x.size() * 2, 0);
    p.dx2.copy_from_host(p.x2.data(), p.x2.size() * 2, 0);
    p.e = oracle(p, quantize(p.x, k, kColumns - 1, p.perm, problem.pb), kColumns - 1);
    return p;
}

// The call's segments into one output (out_offset in the problem's rows): a Rows part as two segments (its
// halves), a Tiles16 part as one.
std::vector<ops::GgmlSegment> base_segments(const Prepared& p) {
    std::vector<ops::GgmlSegment> segments;
    for (std::size_t i = 0; i < p.p.parts.size(); ++i) {
        const Part& part      = p.p.parts[i];
        const auto* data      = static_cast<const std::uint8_t*>(p.dev[i].p);
        if (p.p.layout == ops::GgmlLayout::Tiles16) {
            segments.push_back({part.q, data, part.rows, p.start[i], 0, ops::GgmlLayout::Tiles16});
            continue;
        }
        const int half = part.rows / 2;
        if (half > 0) { segments.push_back({part.q, data, half, p.start[i], 0, ops::GgmlLayout::Rows}); }
        segments.push_back({part.q, data + static_cast<std::size_t>(half) * ops::ggml_row_bytes(part.q, p.p.k), part.rows - half,
                            p.start[i] + half, 0, ops::GgmlLayout::Rows});
    }
    return segments;
}

// Output bounds: one output, one per part, or three cut inside segments (Rows: around the middle; Tiles16:
// at 16 and 32 rows of the first part, which must hold more than 32).
enum class Outputs { One, PerPart, Three };

std::vector<std::int32_t> output_bounds(const Prepared& p, Outputs outputs) {
    if (outputs == Outputs::One) { return {0, p.n}; }
    if (outputs == Outputs::PerPart) {
        std::vector<std::int32_t> bounds(p.start.begin(), p.start.end());
        bounds.push_back(p.n);
        return bounds;
    }
    if (p.p.layout == ops::GgmlLayout::Tiles16) {
        if (p.p.parts.front().rows <= 32) { throw std::logic_error("three tiled outputs need a first part of more than 32 rows"); }
        return {0, 16, 32, p.n};
    }
    return {0, p.n / 2 - 3, p.n / 2 + 5, p.n};
}

// The call's outputs [T][n] as raw bits (binary32, or BF16 in the low half), in the problem's row order.
std::vector<std::uint32_t> run(const Prepared& p, bool second_x, int columns, bool f32, Outputs outputs, ops::LinearRoutes routes) {
    const int k        = p.p.k;
    const auto bounds  = output_bounds(p, outputs);
    const auto base    = base_segments(p);
    const auto segments = bounds.size() > 2 ? ops::ggml_split_segments(base, k, bounds) : base;
    const std::size_t out_bytes = f32 ? 4 : 2;
    std::vector<DeviceBuffer> buffers;
    std::vector<Tensor> tensors;
    for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
        const int n = bounds[i + 1] - bounds[i];
        buffers.emplace_back(static_cast<std::size_t>(n) * columns * out_bytes);
        CUDA_CHECK(cudaMemset(buffers.back().p, 0xFF, buffers.back().bytes));
        tensors.push_back(Tensor(buffers.back().p, f32 ? DType::FP32 : DType::BF16, {n, columns}));
    }
    std::vector<Tensor*> outs;
    for (auto& tensor : tensors) { outs.push_back(&tensor); }
    Tensor x((second_x ? p.dx2 : p.dx).p, DType::BF16, {k, columns});
    // Exactly the advertised workspace: the route's encoding must fit it.
    WorkspaceArena work(ops::ggml_linear_workspace_bytes(k, columns));
    const auto* perm = p.p.pb > 0 ? static_cast<const std::int32_t*>(p.dperm.p) : nullptr;
    if (outs.size() == 1) {
        ops::ggml_linear(x, segments, *outs.front(), perm, p.p.pb, work, nullptr, routes);
    } else {
        ops::ggml_linear(x, segments, std::span<Tensor* const>(outs), perm, p.p.pb, work, nullptr, routes);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> bits(static_cast<std::size_t>(columns) * p.n);
    for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
        const int n = bounds[i + 1] - bounds[i];
        std::vector<std::uint8_t> raw(buffers[i].bytes);
        buffers[i].copy_to_host(raw.data(), raw.size(), 0);
        for (int t = 0; t < columns; ++t) {
            for (int r = 0; r < n; ++r) {
                const std::size_t at = (static_cast<std::size_t>(t) * n + r) * out_bytes;
                std::uint32_t v      = 0;
                std::memcpy(&v, raw.data() + at, out_bytes);
                bits[static_cast<std::size_t>(t) * p.n + bounds[i] + r] = v;
            }
        }
    }
    return bits;
}

// One width of a problem: the outputs against the oracle within the FP32-sum bound, a second run's bits,
// and column 0's bits with every other column changed.
void check_width(Prepared& p, int columns, bool f32, Outputs outputs) {
    const std::string label = "a8 " + p.p.what + " T=" + std::to_string(columns) + (f32 ? " f32" : "") +
                              (outputs == Outputs::Three ? " three outputs" : outputs == Outputs::PerPart ? " per-part outputs" : "");
    const auto bits = run(p, false, columns, f32, outputs, ops::LinearRoutes::GgmlA8);
    std::size_t bad = 0, missing_nan = 0;
    for (int t = 0; t < columns; ++t) {
        for (int r = 0; r < p.n; ++r) {
            std::size_t part = 0;
            while (part + 1 < p.start.size() && r >= p.start[part + 1]) { ++part; }
            const std::size_t i = static_cast<std::size_t>(t) * p.n + r;
            const double y      = from_bits(bits[i], f32);
            if (p.e.nan[i] != 0) {
                missing_nan += !std::isnan(y);
                continue;
            }
            const QType q       = p.p.parts[part].q;
            const int terms     = p.p.k / 32 * (min_format(q) ? 2 : 1);
            const double m      = p.e.mag[i];
            const double fp32   = (terms + 10) * std::ldexp(m, -24) + std::ldexp(m, -36);
            const double bound  = fp32 + (f32 ? 0.0 : std::ldexp(std::fabs(p.e.ref[i]) + fp32, -8)) + 1e-30;
            const double err    = std::fabs(y - p.e.ref[i]);
            bad += !(err <= bound); // NaN fails
            if (std::isfinite(err)) {
                p.worst     = std::max(p.worst, err / bound);
                p.worst_rel = std::max(p.worst_rel, m > 0 ? err / m : 0.0);
            }
        }
    }
    check(bad == 0, label + ": " + std::to_string(bad) + " of " + std::to_string(bits.size()) + " outputs outside the FP32-sum bound");
    check(missing_nan == 0, label + ": " + std::to_string(missing_nan) + " outputs of non-finite columns are not NaN");
    check(run(p, false, columns, f32, outputs, ops::LinearRoutes::GgmlA8) == bits, label + ": a second run's bits differ");
    if (columns > 1) {
        const auto other = run(p, true, columns, f32, outputs, ops::LinearRoutes::GgmlA8);
        check(std::equal(bits.begin(), bits.begin() + p.n, other.begin()),
              label + ": column 0's bits depend on the other columns' values");
    }
}

void report(const Prepared& p) {
    std::printf("info a8 %s: worst error / bound %.3f, error / M %.2e\n", p.p.what.c_str(), p.worst, p.worst_rel);
}

void run_problem(const Problem& problem, std::mt19937& rng, const std::vector<int>& widths, const std::vector<int>& f32_widths = {},
                 const std::vector<int>& three_widths = {}, Outputs outputs = Outputs::One) {
    Prepared p = prepare(problem, rng);
    for (const int columns : widths) { check_width(p, columns, false, outputs); }
    for (const int columns : f32_widths) { check_width(p, columns, true, outputs); }
    for (const int columns : three_widths) { check_width(p, columns, false, Outputs::Three); }
    report(p);
}

// A call of 32 columns with the flag keeps the route it would take without it: the exact route's bits, and
// with ggml-bf16-prefill the BF16 tiles'.
void check_boundary(const Problem& problem, std::mt19937& rng) {
    const Prepared p = prepare(problem, rng);
    using ops::LinearRoutes;
    check(run(p, false, kColumns, false, Outputs::One, LinearRoutes::GgmlA8) ==
              run(p, false, kColumns, false, Outputs::One, LinearRoutes::Exact),
          "a8 " + problem.what + " T=32: the flag changed the exact route's bits");
    check(run(p, false, kColumns, false, Outputs::One, LinearRoutes::GgmlA8 | LinearRoutes::GgmlBf16Wide) ==
              run(p, false, kColumns, false, Outputs::One, LinearRoutes::GgmlBf16Wide),
          "a8 " + problem.what + " T=32: the flag changed the BF16 route's bits");
}

} // namespace

INFERNIX_OP_TEST(ggml_linear_a8) {
    try {
        using ops::GgmlLayout;
        std::mt19937 rng(20261027);
        const std::vector<int> every = {1, 2, 3, 4, 5, 8, 9, 12, 16, 17, 24, 25, 31};
        const QType formats[] = {QType::GGML_Q2_0, QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ2_S,
                                 QType::GGML_IQ3_XXS, QType::GGML_IQ3_S, QType::GGML_IQ4_XS, QType::GGML_IQ4_NL};
        const QType tiled[] = {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ4_XS};

        // ggml_rows_v1, every format: two segments per matrix (an unaligned second half), partial row tiles,
        // the permutation by 16-byte (blocks of 128) and scalar (blocks of 4) encoder loads, a K of 24 units,
        // and short last units (K = 96 and 192: one unit of 3 and 6 slices; K = 640: 2.5 units).
        for (const QType q : formats) {
            const bool small = ggml_block_elems(q) < 256;
            const int k      = small ? 640 : 2560;
            const std::string name = format_name(q);
            run_problem({name + " rows k=" + std::to_string(k), k, {{q, 70}}, GgmlLayout::Rows, 0, Scales::Wide}, rng, every,
                        {1, 3, 13, 31}, {2, 8, 20});
            run_problem({name + " rows perm128", k, {{q, 37}}, GgmlLayout::Rows, 128}, rng, {1, 5, 17});
            run_problem({name + " rows perm4", k, {{q, 21}}, GgmlLayout::Rows, 4}, rng, {2, 9});
            run_problem({name + " rows k=" + std::to_string(small ? 2560 : 6144), small ? 2560 : 6144, {{q, 37}}, GgmlLayout::Rows}, rng,
                        {1, 5, 31}, {5});
            if (small) {
                const int short_k = q == QType::GGML_IQ4_NL ? 96 : 192;
                run_problem({name + " rows k=" + std::to_string(short_k), short_k, {{q, 21}}, GgmlLayout::Rows}, rng, {1, 2, 3, 9, 31});
            }
        }
        // ggml_tiles16_v1: two matrices (the second at an odd output row), partial tiles, three outputs cut at
        // whole tiles, tile counts of one-unit rows and of K = 2560 rows that reach every CTA shape of the
        // occupancy rule, the permutation at K = 6144, a one-tile call of three rows.
        for (const QType q : tiled) {
            const std::string name = format_name(q);
            run_problem({name + " tiled", 2560, {{q, 29}, {q, 16}}, GgmlLayout::Tiles16, 0, Scales::Wide}, rng, every, {1, 2, 9, 31});
            run_problem({name + " tiled three", 2560, {{q, 61}, {q, 20}}, GgmlLayout::Tiles16}, rng, {}, {}, {1, 3, 16, 31});
            for (const int tiles : {200, 800, 3000}) {
                run_problem({name + " tiled k=256 tiles=" + std::to_string(tiles), 256, {{q, 16 * tiles - 5}}, GgmlLayout::Tiles16}, rng,
                            {1, 2, 9, 24});
            }
            for (const int tiles : {3, 40, 300}) {
                run_problem({name + " tiled k=2560 tiles=" + std::to_string(tiles), 2560, {{q, 16 * tiles - 7}}, GgmlLayout::Tiles16}, rng,
                            {1, 5, 31});
            }
            run_problem({name + " tiled perm k=6144", 6144, {{q, 37}}, GgmlLayout::Tiles16, 128, Scales::Wide}, rng, {1, 5, 13, 31}, {1});
            run_problem({name + " tiled three rows", 2560, {{q, 3}}, GgmlLayout::Tiles16}, rng, {1, 5});
        }
        // Two formats (the mixed launch: Rows from 3 columns with BF16 outputs, Tiles16 at every width; per
        // format otherwise, FP32 outputs and three outputs of Rows matrices included).
        const std::pair<QType, QType> mixed[] = {{QType::GGML_Q6_K, QType::GGML_IQ4_XS}, {QType::GGML_IQ4_XS, QType::GGML_Q6_K},
                                                 {QType::GGML_Q4_K, QType::GGML_Q6_K},   {QType::GGML_Q5_K, QType::GGML_IQ4_XS},
                                                 {QType::GGML_Q4_K, QType::GGML_Q5_K},   {QType::GGML_IQ4_XS, QType::GGML_Q4_K}};
        for (const auto& [q0, q1] : mixed) {
            const std::string name = format_name(q0) + "+" + format_name(q1);
            run_problem({name + " rows", 2560, {{q0, 33}, {q1, 33}}, GgmlLayout::Rows}, rng, {1, 2, 3, 5, 8, 9, 16, 17, 31}, {5}, {5, 17});
            run_problem({name + " tiled", 2560, {{q0, 40}, {q1, 26}}, GgmlLayout::Tiles16}, rng, {1, 2, 3, 5, 8, 9, 16, 17, 31}, {5},
                        {5, 17});
        }
        // Zero and sub-threshold slices, Inf and NaN columns, large values.
        run_problem({"edges Q6_K tiled", 2560, {{QType::GGML_Q6_K, 45}}, GgmlLayout::Tiles16, 0, Scales::Narrow, true}, rng, {1, 7, 31});
        run_problem({"edges Q4_K rows", 2560, {{QType::GGML_Q4_K, 45}}, GgmlLayout::Rows, 0, Scales::Narrow, true}, rng, {1, 2, 7, 31});
        run_problem({"edges IQ4_NL rows", 640, {{QType::GGML_IQ4_NL, 45}}, GgmlLayout::Rows, 0, Scales::Narrow, true}, rng, {2, 7, 16});
        // The other routes keep their bits at 32 columns.
        check_boundary({"Q6_K rows boundary", 2560, {{QType::GGML_Q6_K, 70}}, GgmlLayout::Rows}, rng);
        check_boundary({"IQ4_XS+Q6_K tiled boundary", 2560, {{QType::GGML_IQ4_XS, 64}, {QType::GGML_Q6_K, 32}}, GgmlLayout::Tiles16}, rng);
        // The IQ3_S import's dense shapes (Qwen3.8-Flash-Next, hidden 2560): GDN q/k/v/z and attention q/gate/k/v
        // in one mixed launch each, the shared expert's gate/up, the GDN output projection (K = 6144, value
        // heads permuted), the attention output (K = 6144); K = 8192; row-major Q5_K / Q4_K / IQ3_S matrices.
        const std::vector<int> decode = {1, 2, 5, 8, 16, 31};
        run_problem({"gdn q/k/v/z", 2560,
                     {{QType::GGML_Q6_K, 2048}, {QType::GGML_Q6_K, 2048}, {QType::GGML_Q6_K, 6144}, {QType::GGML_IQ4_XS, 6144}},
                     GgmlLayout::Tiles16},
                    rng, decode, {}, {}, Outputs::PerPart);
        run_problem({"attn q/gate/k/v", 2560,
                     {{QType::GGML_IQ4_XS, 6144}, {QType::GGML_IQ4_XS, 6144}, {QType::GGML_Q6_K, 512}, {QType::GGML_Q6_K, 512}},
                     GgmlLayout::Tiles16},
                    rng, decode, {}, {}, Outputs::PerPart);
        run_problem({"shexp gate/up", 2560, {{QType::GGML_Q4_K, 640}, {QType::GGML_Q6_K, 640}}, GgmlLayout::Tiles16}, rng, decode, {}, {},
                    Outputs::PerPart);
        run_problem({"ssm_out (perm)", 6144, {{QType::GGML_Q6_K, 2560}}, GgmlLayout::Tiles16, 128}, rng, decode);
        run_problem({"attn_output", 6144, {{QType::GGML_IQ4_XS, 2560}}, GgmlLayout::Tiles16}, rng, decode);
        run_problem({"k=8192 Q6_K tiled", 8192, {{QType::GGML_Q6_K, 2560}}, GgmlLayout::Tiles16}, rng, {1, 5, 31}, {8});
        run_problem({"k=8192 Q5_K rows", 8192, {{QType::GGML_Q5_K, 2560}}, GgmlLayout::Rows}, rng, {1, 5, 31});
        run_problem({"attn_qkv Q5_K rows", 2560, {{QType::GGML_Q5_K, 10240}}, GgmlLayout::Rows}, rng, decode);
        run_problem({"attn_gate Q4_K rows", 2560, {{QType::GGML_Q4_K, 6144}}, GgmlLayout::Rows}, rng, decode);
        run_problem({"IQ3_S rows 6144", 2560, {{QType::GGML_IQ3_S, 6144}}, GgmlLayout::Rows}, rng, decode);
        if (failures != 0) {
            std::cerr << failures << " ggml_linear a8 checks failed\n";
            return 1;
        }
        std::cout << "ggml_linear a8: every format, layout, width 1-31, mixed launch and output split within the FP32-sum bound "
                     "of the FP64 oracle over the 8-bit codec; deterministic, column-independent at a fixed shape; 32 columns "
                     "keep the other routes' bits\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
