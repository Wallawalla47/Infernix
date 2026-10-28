// ggml_linear's wide 8-bit activation route (LinearRoutes::GgmlA8Wide, `--a8-activations` at prefill widths: LOSSY,
// opt-in; ops/ggml/ggml_a8_tile.cuh) against an FP64 oracle.
//
// The semantic boundary is the activation codec (ops/ggml/ggml_a8.h): the test quantizes every column on the host
// with it (codes q_j and slice scales d_s, input element j read from x[perm(j)]), decodes the weights
// independently (ggml_block_oracle.h, binary64) and sums ref = sum_s d_s sum_(j in s) w_j q_j in binary64.
// Everything else the route does is FP32 rounding: each slice term is rounded once (fl32(P * d * sub), and the
// K-quants' fl32(d * sum q) of the min term) and added by one fma, in K order. With n the row's terms (K / 32,
// twice that for Q6_K / IQ2_S and for Q4_K / Q5_K) and
//
//   M = sum_s |d_s| sum_(j in s) (|w_j| + 2 |o_j|) |q_j|        (o_j: the K-quants' subtracted dmin * min)
//
// bounding the terms' magnitudes, |y - ref| <= (n + 10) 2^-24 M + 2^-36 M (the oracle's own binary64 sums), plus
// 2^-8 (|ref| + that) for a BF16 output's one rounding: the narrow route's criterion (test_ggml_linear_a8.cu). A
// column holding Inf or NaN must give NaN rows.
//
// Coverage: every format on ggml_rows_v1 and the four ggml_tiles16_v1 formats, the scaling classes (one sub-scale
// per slice, Q6_K and IQ2_S's per 16 elements, the K-quants' min term), widths at the route's boundary (32, 33),
// at the CTA tile's (127, 128, 129) and wider, FP32 and BF16 outputs, outputs split at bounds inside segments,
// several formats in one call, the input permutation (16-byte and scalar encoder loads), zero and sub-threshold
// slices, Inf and NaN, a call whose planes need three passes, a call of two column batches, and the IQ3_S import's
// dense shapes at prefill widths (512, 2048, 4096; sampled rows against the oracle, every row for the invariances).
// Every call has a workspace of exactly ggml_linear_workspace_bytes. Invariances, bit for bit: a column's outputs
// are the same at every width of the route (each width against the widest) and with every other column changed;
// repeated calls under a side stream's traffic are identical (op_tester.h verify_repeatable); with GgmlBf16Wide
// also set the route's bits are unchanged (it takes precedence); calls of 31 columns keep the other routes' bits.

#include "ops/op_family.h"
#include "core/device.h"
#include "core/ggml_format.h"
#include "infernix/ops/ggml_linear.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/op_tester.h"
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
bool half_format(QType q) { return q == QType::GGML_Q6_K || q == QType::GGML_IQ2_S; }

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

// Random block bytes (every byte pattern is a valid encoding once the binary16 scales are finite), the scales'
// exponent fields 12-13 (Narrow: the GSQ-RCO import's rows) or 5-20 (Wide).
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

// A ggml_tiles16_v1 matrix of `rows` ggml_rows_v1 rows: tile i at 16 i row_bytes, block b of its row r at (16 b + r) *
// block_bytes, missing rows zero blocks.
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

// BF16 columns [k, columns]: N(0, 1) with every 97th value 64 times larger; every fourth column over magnitudes
// 2^-40 .. 2^12 with zeros, both signs.
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
// (1.5 * 2^-120); 2 a NaN; 3 a +Inf; 4 a -Inf (each makes its column NaN); 5 values near 2^40; 6 a slice of equal
// values (every code 127) and one alternating between +-its maximum.
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
    int columns            = 300; // x's columns: the widest call of the problem
    int sample             = 0;   // > 0: the oracle checks rows 0-31, every sample-th row and the last 32
};

// The codec's codes and slice scales of x's columns (input element j at x[perm(j)]).
struct Quantized {
    std::vector<double> q; // [column][k]
    std::vector<double> d; // [column][k / 32]: NaN for a slice holding Inf or NaN
};

Quantized quantize(const std::vector<std::uint16_t>& x, int k, int columns, const std::vector<std::int32_t>& perm, int pb) {
    Quantized out;
    const int slices = k / 32;
    out.q.resize(static_cast<std::size_t>(columns) * k);
    out.d.resize(static_cast<std::size_t>(columns) * slices);
    parallel_for(columns, [&](int t) {
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
    });
    return out;
}

// The oracle at the checked rows: ref[i][t], mag[i][t] (i: the index of the row in `rows`).
struct Expected {
    std::vector<int> rows;
    std::vector<double> ref, mag; // [row index][column]
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

std::vector<int> checked_rows(int n, int sample) {
    std::vector<int> rows;
    for (int r = 0; r < n; ++r) {
        if (sample <= 0 || r < 32 || r >= n - 32 || r % sample == 0) { rows.push_back(r); }
    }
    return rows;
}

Expected oracle(const Prepared& p, const Quantized& xq, int columns) {
    const int k = p.p.k, slices = k / 32;
    Expected e;
    e.rows = checked_rows(p.n, p.p.sample);
    const int checked = static_cast<int>(e.rows.size());
    e.ref.assign(static_cast<std::size_t>(checked) * columns, 0.0);
    e.mag.assign(e.ref.size(), 0.0);
    e.nan.assign(e.ref.size(), 0);
    parallel_for(checked, [&](int i) {
        const int r = e.rows[static_cast<std::size_t>(i)];
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
            const std::size_t at = static_cast<std::size_t>(i) * columns + t;
            e.ref[at]            = ref;
            e.mag[at]            = mag;
            e.nan[at]            = nan ? 1 : 0;
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
    p.x = random_columns(k, problem.columns, rng);
    if (problem.edges) { make_edges(p.x, k, rng); }
    p.x2 = p.x;
    const auto other = random_columns(k, problem.columns, rng);
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
    p.e = oracle(p, quantize(p.x, k, problem.columns, p.perm, problem.pb), problem.columns);
    return p;
}

// The call's segments into one output (out_offset in the problem's rows): a Rows part as two segments (its halves,
// the second at an unaligned row), a Tiles16 part as one.
std::vector<ops::GgmlSegment> base_segments(const Prepared& p) {
    std::vector<ops::GgmlSegment> segments;
    for (std::size_t i = 0; i < p.p.parts.size(); ++i) {
        const Part& part = p.p.parts[i];
        const auto* data = static_cast<const std::uint8_t*>(p.dev[i].p);
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

// Output bounds: one output, one per part, or three cut inside segments (Rows: around the middle; Tiles16: at 16 and
// 32 rows of the first part, which must hold more than 32).
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

// One call's device outputs and launch (exactly the advertised workspace).
struct Call {
    std::vector<std::int32_t> bounds;
    std::vector<ops::GgmlSegment> segments;
    std::vector<DeviceBuffer> buffers;
    std::vector<Tensor> tensors;
    std::vector<Tensor*> outs;
    Tensor x;
    WorkspaceArena work;
    const std::int32_t* perm = nullptr;
    int pb                   = 0;
    ops::LinearRoutes routes = ops::LinearRoutes::GgmlA8Wide;

    Call(const Prepared& p, bool second_x, int columns, bool f32, Outputs outputs, ops::LinearRoutes r)
        : bounds(output_bounds(p, outputs)),
          x((second_x ? p.dx2 : p.dx).p, DType::BF16, {p.p.k, columns}),
          work(ops::ggml_linear_workspace_bytes(p.p.k, columns)),
          routes(r) {
        const auto base = base_segments(p);
        segments        = bounds.size() > 2 ? ops::ggml_split_segments(base, p.p.k, bounds) : base;
        const std::size_t out_bytes = f32 ? 4 : 2;
        for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
            const int n = bounds[i + 1] - bounds[i];
            buffers.emplace_back(static_cast<std::size_t>(n) * columns * out_bytes);
        }
        for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
            CUDA_CHECK(cudaMemset(buffers[i].p, 0xFF, buffers[i].bytes));
            tensors.push_back(Tensor(buffers[i].p, f32 ? DType::FP32 : DType::BF16, {bounds[i + 1] - bounds[i], columns}));
        }
        for (auto& tensor : tensors) { outs.push_back(&tensor); }
        perm = p.p.pb > 0 ? static_cast<const std::int32_t*>(p.dperm.p) : nullptr;
        pb   = p.p.pb;
    }

    void launch(cudaStream_t stream) {
        if (outs.size() == 1) {
            ops::ggml_linear(x, segments, *outs.front(), perm, pb, work, stream, routes);
        } else {
            ops::ggml_linear(x, segments, std::span<Tensor* const>(outs), perm, pb, work, stream, routes);
        }
    }

    // The outputs [T][n] as raw bits (binary32, or BF16 in the low half), in the problem's row order.
    std::vector<std::uint32_t> bits(int n, int columns, bool f32) const {
        const std::size_t out_bytes = f32 ? 4 : 2;
        std::vector<std::uint32_t> all(static_cast<std::size_t>(columns) * n);
        for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
            const int rows = bounds[i + 1] - bounds[i];
            std::vector<std::uint8_t> raw(buffers[i].bytes);
            buffers[i].copy_to_host(raw.data(), raw.size(), 0);
            for (int t = 0; t < columns; ++t) {
                for (int r = 0; r < rows; ++r) {
                    std::uint32_t v = 0;
                    std::memcpy(&v, raw.data() + (static_cast<std::size_t>(t) * rows + r) * out_bytes, out_bytes);
                    all[static_cast<std::size_t>(t) * n + bounds[i] + r] = v;
                }
            }
        }
        return all;
    }
};

std::vector<std::uint32_t> run(const Prepared& p, bool second_x, int columns, bool f32, Outputs outputs,
                               ops::LinearRoutes routes = ops::LinearRoutes::GgmlA8Wide) {
    Call call(p, second_x, columns, f32, outputs, routes);
    call.launch(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    return call.bits(p.n, columns, f32);
}

// The outputs of one width against the oracle within the FP32-sum bound, at the checked rows.
void check_oracle(Prepared& p, const std::vector<std::uint32_t>& bits, int columns, bool f32, const std::string& label) {
    std::size_t bad = 0, missing_nan = 0, checked = 0;
    for (std::size_t i = 0; i < p.e.rows.size(); ++i) {
        const int r = p.e.rows[i];
        std::size_t part = 0;
        while (part + 1 < p.start.size() && r >= p.start[part + 1]) { ++part; }
        const QType q   = p.p.parts[part].q;
        const int terms = p.p.k / 32 * (half_format(q) || min_format(q) ? 2 : 1);
        for (int t = 0; t < columns; ++t) {
            const std::size_t at = i * static_cast<std::size_t>(p.p.columns) + t;
            const double y       = from_bits(bits[static_cast<std::size_t>(t) * p.n + r], f32);
            ++checked;
            if (p.e.nan[at] != 0) {
                missing_nan += !std::isnan(y);
                continue;
            }
            const double m     = p.e.mag[at];
            const double fp32  = (terms + 10) * std::ldexp(m, -24) + std::ldexp(m, -36);
            const double bound = fp32 + (f32 ? 0.0 : std::ldexp(std::fabs(p.e.ref[at]) + fp32, -8)) + 1e-30;
            const double err   = std::fabs(y - p.e.ref[at]);
            bad += !(err <= bound); // NaN fails
            if (std::isfinite(err)) {
                p.worst     = std::max(p.worst, err / bound);
                p.worst_rel = std::max(p.worst_rel, m > 0 ? err / m : 0.0);
            }
        }
    }
    check(bad == 0, label + ": " + std::to_string(bad) + " of " + std::to_string(checked) + " checked outputs outside the FP32-sum bound");
    check(missing_nan == 0, label + ": " + std::to_string(missing_nan) + " outputs of non-finite columns are not NaN");
}

// Every width of a problem against the oracle; each width's bits against the widest's on their shared columns (a
// column's bits do not depend on the call's width), column 0 with every other column changed, and a second run.
void run_problem(const Problem& problem, std::mt19937& rng, const std::vector<int>& widths, bool f32 = false,
                 Outputs outputs = Outputs::One) {
    Prepared p = prepare(problem, rng);
    const int widest = *std::max_element(widths.begin(), widths.end());
    if (widest > problem.columns) { throw std::logic_error("a width past the problem's columns"); }
    const auto reference = run(p, false, widest, f32, outputs);
    for (const int columns : widths) {
        const std::string label = "a8 wide " + p.p.what + " T=" + std::to_string(columns) + (f32 ? " f32" : "") +
                                  (outputs == Outputs::Three ? " three outputs" : outputs == Outputs::PerPart ? " per-part outputs" : "");
        const auto bits = columns == widest ? reference : run(p, false, columns, f32, outputs);
        check_oracle(p, bits, columns, f32, label);
        check(std::equal(bits.begin(), bits.end(), reference.begin()),
              label + ": a column's bits differ from the same column's in the call of " + std::to_string(widest) + " columns");
    }
    const auto other = run(p, true, widths.front(), f32, outputs);
    const auto first = run(p, false, widths.front(), f32, outputs);
    check(std::equal(first.begin(), first.begin() + p.n, other.begin()),
          "a8 wide " + p.p.what + ": column 0's bits depend on the other columns' values");
    check(first == std::vector<std::uint32_t>(reference.begin(), reference.begin() + static_cast<std::ptrdiff_t>(first.size())),
          "a8 wide " + p.p.what + ": a second run's bits differ");
    std::printf("info a8 wide %s: worst error / bound %.3f, error / M %.2e\n", p.p.what.c_str(), p.worst, p.worst_rel);
}

// The flags around the route: GgmlBf16Wide beside GgmlA8Wide keeps the route's bits; 31 columns keep the exact (and
// with GgmlA8 the narrow 8-bit) route's bits.
void check_flags(const Problem& problem, std::mt19937& rng) {
    const Prepared p = prepare(problem, rng);
    using ops::LinearRoutes;
    check(run(p, false, 64, false, Outputs::One, LinearRoutes::GgmlA8Wide | LinearRoutes::GgmlBf16Wide) ==
              run(p, false, 64, false, Outputs::One, LinearRoutes::GgmlA8Wide),
          "a8 wide " + problem.what + " T=64: GgmlBf16Wide changed the route's bits");
    check(run(p, false, 31, false, Outputs::One, LinearRoutes::GgmlA8Wide) == run(p, false, 31, false, Outputs::One, LinearRoutes::Exact),
          "a8 wide " + problem.what + " T=31: the flag changed the exact route's bits");
    check(run(p, false, 31, false, Outputs::One, LinearRoutes::GgmlA8Wide | LinearRoutes::GgmlA8) ==
              run(p, false, 31, false, Outputs::One, LinearRoutes::GgmlA8),
          "a8 wide " + problem.what + " T=31: the flag changed the narrow 8-bit route's bits");
}

// Repeated calls under a side stream's traffic return the same bits (the cp.async rings, op_tester.h).
void check_repeat(const Problem& problem, std::mt19937& rng, int columns) {
    const Prepared p = prepare(problem, rng);
    Call call(p, false, columns, false, Outputs::One, ops::LinearRoutes::GgmlA8Wide);
    const int bad = test::verify_repeatable("a8 wide " + problem.what + " repeat", 4, nullptr, [&] { call.launch(nullptr); },
                                            {test::OutputRegion{call.buffers.front().p, call.buffers.front().bytes}});
    check(bad == 0, "a8 wide " + problem.what + ": repeated calls differ");
}

// The fixtures make the FP32 order visible: one row's slice terms summed forward and backward differ (so the width
// invariance above could fail).
void check_order_visible(const Problem& problem, std::mt19937& rng) {
    Prepared p             = prepare(problem, rng);
    const int k            = p.p.k;
    const Quantized xq     = quantize(p.x, k, 1, p.perm, p.p.pb);
    const QType q          = p.p.parts.front().q;
    const std::size_t rowb = ops::ggml_row_bytes(q, k), be = ggml_block_elems(q), bb = ggml_block_bytes(q);
    std::vector<float> terms;
    for (int r = 0; r < 8; ++r) {
        std::vector<double> w(static_cast<std::size_t>(k));
        for (std::size_t b = 0; b < static_cast<std::size_t>(k) / be; ++b) { decode_block(q, p.host[0].data() + r * rowb + b * bb, w.data() + b * be); }
        for (int s = 0; s < k / 32; ++s) {
            double dot = 0.0;
            for (int j = 0; j < 32; ++j) { dot += w[static_cast<std::size_t>(32 * s + j)] * xq.q[static_cast<std::size_t>(32 * s + j)]; }
            terms.push_back(static_cast<float>(dot * xq.d[static_cast<std::size_t>(s)]));
        }
    }
    check(test::summation_order_visible(terms, 8, static_cast<std::size_t>(k / 32)),
          "a8 wide " + problem.what + ": the fixture cannot show a summation order");
}

} // namespace

INFERNIX_OP_TEST(ggml_linear_a8_wide) {
    try {
        using ops::GgmlLayout;
        std::mt19937 rng(20261028);
        const std::vector<int> boundary = {32, 33, 64, 127, 128, 129, 300};
        const QType formats[] = {QType::GGML_Q2_0, QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ2_S,
                                 QType::GGML_IQ3_XXS, QType::GGML_IQ3_S, QType::GGML_IQ4_XS, QType::GGML_IQ4_NL};
        const QType tiled[] = {QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ4_XS};

        check_order_visible({"order Q6_K", 2560, {{QType::GGML_Q6_K, 8}}}, rng);
        // ggml_rows_v1, every format: two segments per matrix (an unaligned second half), partial row tiles (70 and 200
        // rows: one and two CTA row pairs, the second half-filled), FP32 and BF16 outputs, three outputs, the
        // permutation by 16-byte (blocks of 128) and scalar (blocks of 4) encoder loads, K of one (640) and many units.
        for (const QType q : formats) {
            const bool small = ggml_block_elems(q) < 256;
            const int k      = small ? 640 : 2560;
            const std::string name = format_name(q);
            run_problem({name + " rows k=" + std::to_string(k), k, {{q, 70}}, GgmlLayout::Rows, 0, Scales::Wide}, rng, boundary);
            run_problem({name + " rows f32", k, {{q, 200}}, GgmlLayout::Rows, 0, Scales::Wide, false, 130}, rng, {32, 130}, true);
            run_problem({name + " rows three", k, {{q, 150}}, GgmlLayout::Rows, 0, Scales::Narrow, false, 64}, rng, {33, 64}, false,
                        Outputs::Three);
            run_problem({name + " rows perm128", k, {{q, 37}}, GgmlLayout::Rows, 128, Scales::Narrow, false, 64}, rng, {40, 64});
            run_problem({name + " rows perm4", k, {{q, 21}}, GgmlLayout::Rows, 4, Scales::Narrow, false, 48}, rng, {48});
            run_problem({name + " rows k=" + std::to_string(small ? 2560 : 6144), small ? 2560 : 6144, {{q, 37}}, GgmlLayout::Rows,
                         0, Scales::Narrow, false, 160},
                        rng, {32, 160});
        }
        // ggml_tiles16_v1: two matrices (the second at an odd output row), partial tiles, three outputs cut at whole
        // tiles, the permutation at K = 6144.
        for (const QType q : tiled) {
            const std::string name = format_name(q);
            run_problem({name + " tiled", 2560, {{q, 93}, {q, 16}}, GgmlLayout::Tiles16, 0, Scales::Wide}, rng, boundary);
            run_problem({name + " tiled three", 2560, {{q, 61}, {q, 20}}, GgmlLayout::Tiles16, 0, Scales::Narrow, false, 64}, rng,
                        {32, 64}, false, Outputs::Three);
            run_problem({name + " tiled perm k=6144", 6144, {{q, 37}}, GgmlLayout::Tiles16, 128, Scales::Wide, false, 130}, rng,
                        {32, 130}, true);
        }
        // Several formats in one call (per-part outputs and one output).
        const std::pair<QType, QType> mixed[] = {{QType::GGML_Q6_K, QType::GGML_IQ4_XS}, {QType::GGML_Q4_K, QType::GGML_Q6_K},
                                                 {QType::GGML_Q5_K, QType::GGML_IQ4_XS}};
        for (const auto& [q0, q1] : mixed) {
            const std::string name = format_name(q0) + "+" + format_name(q1);
            run_problem({name + " rows", 2560, {{q0, 33}, {q1, 70}}, GgmlLayout::Rows, 0, Scales::Narrow, false, 200}, rng, {32, 200},
                        false, Outputs::PerPart);
            run_problem({name + " tiled", 2560, {{q0, 40}, {q1, 136}}, GgmlLayout::Tiles16, 0, Scales::Narrow, false, 200}, rng,
                        {64, 200});
        }
        // Zero and sub-threshold slices, Inf and NaN columns, large values.
        run_problem({"edges Q6_K tiled", 2560, {{QType::GGML_Q6_K, 45}}, GgmlLayout::Tiles16, 0, Scales::Narrow, true, 64}, rng, {32, 64});
        run_problem({"edges Q4_K rows", 2560, {{QType::GGML_Q4_K, 45}}, GgmlLayout::Rows, 0, Scales::Narrow, true, 64}, rng, {33, 64});
        run_problem({"edges IQ4_NL rows", 640, {{QType::GGML_IQ4_NL, 45}}, GgmlLayout::Rows, 0, Scales::Narrow, true, 64}, rng, {64});
        // A call of two column batches (2,048 and 52) and one whose planes take three passes (K = 8192, 16,384 rows:
        // slabs of 5,952 rows), the slab bounds among the checked rows.
        run_problem({"Q4_K two batches", 2560, {{QType::GGML_Q4_K, 300}}, GgmlLayout::Rows, 0, Scales::Narrow, false, 2100}, rng,
                    {32, 2100});
        {
            Problem passes{"Q6_K three passes k=8192", 8192, {{QType::GGML_Q6_K, 16384}}, GgmlLayout::Tiles16, 0, Scales::Narrow, false, 40,
                           93};
            run_problem(passes, rng, {33, 40});
        }
        // The flags around the route, and repeated calls under traffic.
        check_flags({"IQ4_XS+Q6_K tiled flags", 2560, {{QType::GGML_IQ4_XS, 64}, {QType::GGML_Q6_K, 32}}, GgmlLayout::Tiles16, 0,
                     Scales::Narrow, false, 64},
                    rng);
        check_flags({"Q5_K rows flags", 2560, {{QType::GGML_Q5_K, 70}}, GgmlLayout::Rows, 0, Scales::Narrow, false, 64}, rng);
        check_repeat({"gdn q/k/v/z repeat", 2560, {{QType::GGML_Q6_K, 4096}, {QType::GGML_IQ4_XS, 2048}}, GgmlLayout::Tiles16, 0,
                      Scales::Narrow, false, 1024, 211},
                     rng, 1024);
        // The IQ3_S import's dense shapes (Qwen3.8-Flash-Next, hidden 2560) at prefill widths, rows sampled for the
        // oracle: GDN q/k/v/z and attention q/gate/k/v (one call of two formats each), the shared expert's gate/up,
        // the GDN output projection (K = 6144, value heads permuted), the attention output (K = 6144), row-major
        // Q5_K / Q4_K / IQ3_S matrices.
        const std::vector<int> prefill = {512, 2048, 4096};
        run_problem({"gdn q/k/v/z", 2560,
                     {{QType::GGML_Q6_K, 2048}, {QType::GGML_Q6_K, 2048}, {QType::GGML_Q6_K, 6144}, {QType::GGML_IQ4_XS, 6144}},
                     GgmlLayout::Tiles16, 0, Scales::Narrow, false, 4096, 211},
                    rng, prefill, false, Outputs::PerPart);
        run_problem({"attn q/gate/k/v", 2560,
                     {{QType::GGML_IQ4_XS, 6144}, {QType::GGML_IQ4_XS, 6144}, {QType::GGML_Q6_K, 512}, {QType::GGML_Q6_K, 512}},
                     GgmlLayout::Tiles16, 0, Scales::Narrow, false, 2048, 211},
                    rng, {512, 2048}, false, Outputs::PerPart);
        run_problem({"shexp gate/up", 2560, {{QType::GGML_Q4_K, 640}, {QType::GGML_Q6_K, 640}}, GgmlLayout::Tiles16, 0, Scales::Narrow,
                     false, 2048, 17},
                    rng, {512, 2048}, false, Outputs::PerPart);
        run_problem({"ssm_out (perm)", 6144, {{QType::GGML_Q6_K, 2560}}, GgmlLayout::Tiles16, 128, Scales::Narrow, false, 4096, 67}, rng,
                    prefill);
        run_problem({"attn_output", 6144, {{QType::GGML_IQ4_XS, 2560}}, GgmlLayout::Tiles16, 0, Scales::Narrow, false, 2048, 67}, rng,
                    {512, 2048});
        run_problem({"attn_qkv Q5_K rows", 2560, {{QType::GGML_Q5_K, 10240}}, GgmlLayout::Rows, 0, Scales::Narrow, false, 2048, 157}, rng,
                    {512, 2048});
        run_problem({"attn_gate Q4_K rows", 2560, {{QType::GGML_Q4_K, 6144}}, GgmlLayout::Rows, 0, Scales::Narrow, false, 512, 97}, rng,
                    {512});
        run_problem({"IQ3_S rows 6144", 2560, {{QType::GGML_IQ3_S, 6144}}, GgmlLayout::Rows, 0, Scales::Narrow, false, 512, 97}, rng,
                    {512});
        if (failures != 0) {
            std::cerr << failures << " ggml_linear a8 wide checks failed\n";
            return 1;
        }
        std::cout << "ggml_linear a8 wide: every format, layout, scaling class, width at the route's and the tile's bounds and "
                     "prefill widths, output split, batches and plane passes within the FP32-sum bound of the FP64 oracle over "
                     "the 8-bit codec; column bits invariant across widths and other columns, repeatable under traffic; the "
                     "flags keep the other routes' bits\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
