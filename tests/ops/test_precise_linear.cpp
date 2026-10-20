// precise_linear and precise_swiglu (the reference route's projections) against independent oracles.
//
// Weights come from the suites' fixtures and are decoded by their own logical decoders
// (quantized_weight.h logical_weight_fp64; ggml_block_oracle.h for GGUF blocks). Properties:
//  - precision: every output is the correctly rounded value of the exact sum (a double-double
//    compensated dot product), except where the exact sum lies within the route's error bound of a
//    rounding boundary (precise_dot.cuh: (4096 + 2 K) u^2 times the terms' magnitudes, u = 2^-24;
//    such outputs are counted and reported, as are those differing from the binary64 loop
//    acc = fma(w[k], x[k], acc), the route's previous arithmetic);
//  - width invariance: a column's outputs are bit-identical alone, among 9 columns and among 37, and
//    whether the rows are one output or split among three.
// Shapes cover every dense format (BF16 Weight and tensor lists, q2/q4/q5/q6/q8 row-split with a
// partial last group, row-scaled FP8, NVFP4), every GGUF format in both layouts with the input
// permutation and several outputs, BF16 and FP32 outputs, K across the kernel's 256-element slices,
// rows that do not fill a CTA and widths 1, 8, 9 and 37.

#include "core/arena.h"
#include "core/device.h"
#include "core/ggml_format.h"
#include "infernix/ops/precise_linear.h"
#include "ggml_block_oracle.h"
#include "quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace infernix;
namespace qw = infernix::test::quantized_weight;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << what << "\n";
    }
}

std::uint16_t bf16_bits_of_float(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

double bf16_value(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

double f32_value(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// binary64 -> BF16, one round-to-nearest-even: round to odd binary32, then to BF16 (cuda_bf16.hpp).
std::uint16_t bf16_bits_of_double(double x) {
    float f = static_cast<float>(x);
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const double d = f;
    if ((x > 0.0 && d > x) || (x < 0.0 && d < x)) { --u; }
    if (d != x && (u << 1) <= 0xFF000000U) { u |= 1U; }
    std::memcpy(&f, &u, 4);
    return bf16_bits_of_float(f);
}

std::uint32_t f32_bits_of_double(double x) {
    const float f = static_cast<float>(x);
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// The neighbours of an output value in its format (sign-magnitude bits as an ordered integer).
double neighbour(std::uint32_t bits, bool bf16, int step) {
    const std::uint32_t sign = bf16 ? 0x8000U : 0x80000000U;
    std::int64_t ordered = (bits & sign) != 0 ? -static_cast<std::int64_t>(bits & ~sign) : static_cast<std::int64_t>(bits);
    ordered += step;
    const std::uint32_t back = ordered < 0 ? static_cast<std::uint32_t>(-ordered) | sign : static_cast<std::uint32_t>(ordered);
    return bf16 ? bf16_value(static_cast<std::uint16_t>(back)) : f32_value(back);
}

struct Columns {
    int k = 0, columns = 0;
    std::vector<std::uint16_t> bits; // BF16 [K, T] as the Op reads it
};

// Normal columns with every seventh element large and of alternating sign (cancellation) and some
// zeros and tiny values.
Columns random_columns(int k, int columns, std::mt19937& rng) {
    Columns x;
    x.k = k;
    x.columns = columns;
    x.bits.resize(static_cast<std::size_t>(k) * columns);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    for (std::size_t i = 0; i < x.bits.size(); ++i) {
        float v = normal(rng);
        if (i % 7 == 0) { v = (i % 14 == 0 ? 37.0F : -37.0F) * (1.0F + 0.1F * normal(rng)); }
        if (i % 29 == 0) { v = 0.0F; }
        if (i % 31 == 0) { v = std::ldexp(v, -30); }
        x.bits[i] = bf16_bits_of_float(v);
    }
    return x;
}

// The route's error bound (precise_dot.cuh) for a K-term sum whose terms' magnitudes total `magnitude`.
double route_bound(int k, double magnitude) { return (4096.0 + 2.0 * k) * std::ldexp(magnitude, -48); }

// Over the whole run: outputs checked, outputs not correctly rounded but within the route's error of
// a tie, and outputs whose bits differ from the binary64 loop's.
std::size_t g_outputs = 0, g_near_ties = 0, g_loop_differs = 0;

// Checks Op output bits [rows, T] (column-major as stored: column t at t * ld + offset) against the
// precision property. `w` is the decoded weights [rows][K]; x is read through `perm`. `offsets`
// (GGUF K-quants) holds each weight's subtracted term t, which the route applies separately: its
// terms are (w + t) x and t x.
void verify(const std::string& label, const std::vector<double>& w, int rows, const Columns& x,
            const std::vector<std::int32_t>* perm, int pb, const std::vector<std::uint32_t>& got, bool bf16,
            const std::vector<double>* offsets = nullptr) {
    const int k = x.k;
    std::size_t loop_differs = 0, imprecise = 0, near_ties = 0;
    std::vector<double> col(static_cast<std::size_t>(k));
    for (int t = 0; t < x.columns; ++t) {
        for (int j = 0; j < k; ++j) {
            const int src = perm != nullptr ? (*perm)[static_cast<std::size_t>(j / pb)] * pb + j % pb : j;
            col[static_cast<std::size_t>(j)] = bf16_value(x.bits[static_cast<std::size_t>(t) * k + src]);
        }
        for (int r = 0; r < rows; ++r) {
            const double* wr = w.data() + static_cast<std::size_t>(r) * k;
            const double* tr = offsets != nullptr ? offsets->data() + static_cast<std::size_t>(r) * k : nullptr;
            double loop = 0.0, hi = 0.0, lo = 0.0, magnitude = 0.0;
            for (int j = 0; j < k; ++j) {
                const double a = wr[j], b = col[static_cast<std::size_t>(j)];
                loop = std::fma(a, b, loop);
                const double p = a * b, pe = std::fma(a, b, -p); // two-product
                const double s = hi + p, bb = s - hi, se = (hi - (s - bb)) + (p - bb); // two-sum
                hi = s;
                lo += se + pe;
                const double offset = tr != nullptr ? tr[j] : 0.0;
                magnitude += std::fabs((a + offset) * b) + std::fabs(offset * b);
            }
            const std::uint32_t bits = got[static_cast<std::size_t>(t) * rows + r];
            loop_differs += bits != (bf16 ? bf16_bits_of_double(loop) : f32_bits_of_double(loop));
            // Correct rounding of hi + lo: inside [midpoint below, midpoint above] of the output.
            const double y = bf16 ? bf16_value(static_cast<std::uint16_t>(bits)) : f32_value(bits);
            const double below = (neighbour(bits, bf16, -1) + y) / 2, above = (neighbour(bits, bf16, 1) + y) / 2;
            const double lo_side = (hi - std::min(below, above)) + lo, hi_side = (hi - std::max(below, above)) + lo;
            if (lo_side < 0.0 || hi_side > 0.0) {
                const double miss = lo_side < 0.0 ? -lo_side : hi_side;
                ++(miss <= route_bound(k, magnitude) ? near_ties : imprecise);
            }
        }
    }
    g_outputs += static_cast<std::size_t>(x.columns) * rows;
    g_near_ties += near_ties;
    g_loop_differs += loop_differs;
    check(imprecise == 0, label + ": " + std::to_string(imprecise) + " outputs are not correctly rounded");
    if (near_ties != 0 || loop_differs != 0) {
        std::cout << label << ": " << near_ties << " outputs within the route's error of a tie, " << loop_differs
                  << " differ from the binary64 loop\n";
    }
}

std::vector<std::uint32_t> download(const DeviceBuffer& out, std::size_t count, bool bf16) {
    std::vector<std::uint32_t> bits(count);
    if (bf16) {
        std::vector<std::uint16_t> h(count);
        out.copy_to_host(h.data(), count * 2, 0);
        for (std::size_t i = 0; i < count; ++i) { bits[i] = h[i]; }
    } else {
        out.copy_to_host(bits.data(), count * 4, 0);
    }
    return bits;
}

const int kWidths[] = {1, 8, 9, 37};

// A column's outputs among all of x's columns, among the first 9 and alone (column 20) must agree bit
// for bit. launch(x, out) runs the Op on BF16 x [K, T] into out [rows, T].
void check_widths(const std::string& label, const Columns& x, int rows, bool bf16_out,
                  const std::function<void(const Tensor&, Tensor&)>& launch) {
    const int k = x.k;
    DeviceBuffer dx(x.bits.size() * 2);
    dx.copy_from_host(x.bits.data(), x.bits.size() * 2, 0);
    const auto run = [&](int first, int count) {
        DeviceBuffer dout(static_cast<std::size_t>(rows) * count * 4);
        const Tensor tx(static_cast<std::uint16_t*>(dx.p) + static_cast<std::size_t>(first) * k, DType::BF16, {k, count});
        Tensor tout(dout.p, bf16_out ? DType::BF16 : DType::FP32, {rows, count});
        launch(tx, tout);
        CUDA_CHECK(cudaDeviceSynchronize());
        return download(dout, static_cast<std::size_t>(rows) * count, bf16_out);
    };
    const auto all  = run(0, x.columns);
    std::size_t differ = 0;
    const auto compare = [&](int first, int count) {
        const auto part = run(first, count);
        for (int t = 0; t < count; ++t) {
            for (int r = 0; r < rows; ++r) {
                differ += part[static_cast<std::size_t>(t) * rows + r] != all[static_cast<std::size_t>(first + t) * rows + r];
            }
        }
    };
    compare(0, 9);
    compare(20, 1);
    check(differ == 0, label + ": " + std::to_string(differ) + " outputs depend on the call's width");
}

// ------------------------------------------------------------------------------- dense Weight

void run_weight(const std::string& name, const Weight& host_weight, const std::function<double(int, int)>& decode,
                const Weight& device_weight, std::mt19937& rng, bool bf16_out) {
    const int n = host_weight.n, k = host_weight.k;
    std::vector<double> w(static_cast<std::size_t>(n) * k);
    for (int r = 0; r < n; ++r) {
        for (int j = 0; j < k; ++j) { w[static_cast<std::size_t>(r) * k + j] = decode(r, j); }
    }
    for (const int columns : kWidths) {
        const Columns x = random_columns(k, columns, rng);
        DeviceBuffer dx(x.bits.size() * 2), dout(static_cast<std::size_t>(n) * columns * 4);
        dx.copy_from_host(x.bits.data(), x.bits.size() * 2, 0);
        Tensor tx(dx.p, DType::BF16, {k, columns});
        Tensor tout(dout.p, bf16_out ? DType::BF16 : DType::FP32, {n, columns});
        ops::precise_linear(tx, device_weight, tout, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::string label = name + " n=" + std::to_string(n) + " k=" + std::to_string(k) + " T=" +
                                  std::to_string(columns) + (bf16_out ? " bf16" : " fp32");
        verify(label, w, n, x, nullptr, 1, download(dout, static_cast<std::size_t>(n) * columns, bf16_out), bf16_out);
        if (columns > 20) {
            check_widths(label, x, n, bf16_out,
                         [&](const Tensor& in, Tensor& out) { ops::precise_linear(in, device_weight, out, nullptr); });
        }
    }
}

void run_packed(QType qtype, int n, int k, std::uint32_t seed, qw::PatternedWeightOptions options, std::mt19937& rng,
                bool bf16_out = true) {
    const qw::PackedWeight packed = qw::make_patterned_weight(qtype, n, k, seed, options);
    DeviceBuffer payload(packed.payload.size());
    payload.copy_from_host(packed.payload.data(), packed.payload.size(), 0);
    run_weight("format " + std::to_string(static_cast<int>(qtype)), packed.weight,
               [&](int r, int j) { return qw::logical_weight_fp64(packed, r, j); }, packed.device_weight(payload.p), rng,
               bf16_out);
}

void run_bf16_weight(int n, int k, std::mt19937& rng, bool bf16_out) {
    std::vector<std::uint16_t> data(static_cast<std::size_t>(n) * k);
    std::normal_distribution<float> normal(0.0F, 0.05F);
    for (auto& v : data) { v = bf16_bits_of_float(normal(rng)); }
    DeviceBuffer d(data.size() * 2);
    d.copy_from_host(data.data(), data.size() * 2, 0);
    Weight weight;
    weight.qtype = QType::BF16;
    weight.layout = QuantLayout::Contiguous;
    weight.ndim = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = n;
    weight.k = weight.shape[1] = weight.padded_shape[1] = k;
    Weight device = weight;
    device.qdata = d.p;
    device.payload = d.p;
    run_weight("bf16 weight", weight, [&](int r, int j) { return bf16_value(data[static_cast<std::size_t>(r) * k + j]); },
               device, rng, bf16_out);
}

// BF16 tensors [K, N_i] (projection_fp32's form): rows concatenated.
void run_bf16_tensors(int k, std::mt19937& rng) {
    const int parts[] = {70, 129, 3};
    int n = 0;
    for (const int p : parts) { n += p; }
    std::vector<std::uint16_t> data(static_cast<std::size_t>(n) * k);
    std::normal_distribution<float> normal(0.0F, 0.05F);
    for (auto& v : data) { v = bf16_bits_of_float(normal(rng)); }
    std::vector<DeviceBuffer> buffers;
    std::vector<Tensor> tensors;
    int row = 0;
    for (const int p : parts) {
        buffers.emplace_back(static_cast<std::size_t>(p) * k * 2);
        buffers.back().copy_from_host(data.data() + static_cast<std::size_t>(row) * k, static_cast<std::size_t>(p) * k * 2, 0);
        tensors.push_back(Tensor(buffers.back().p, DType::BF16, {k, p}));
        row += p;
    }
    const Tensor* const list[] = {&tensors[0], &tensors[1], &tensors[2]};
    std::vector<double> w(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) { w[i] = bf16_value(data[i]); }
    for (const int columns : kWidths) {
        const Columns x = random_columns(k, columns, rng);
        DeviceBuffer dx(x.bits.size() * 2), dout(static_cast<std::size_t>(n) * columns * 4);
        dx.copy_from_host(x.bits.data(), x.bits.size() * 2, 0);
        Tensor tx(dx.p, DType::BF16, {k, columns});
        Tensor tout(dout.p, DType::FP32, {n, columns});
        ops::precise_linear(tx, list, tout, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        const std::string label = "bf16 tensors k=" + std::to_string(k) + " T=" + std::to_string(columns);
        verify(label, w, n, x, nullptr, 1, download(dout, static_cast<std::size_t>(n) * columns, false), false);
        if (columns > 20) {
            check_widths(label, x, n, false, [&](const Tensor& in, Tensor& out) { ops::precise_linear(in, list, out, nullptr); });
        }
    }
}

// ------------------------------------------------------------------------------------- GGUF

std::vector<std::uint8_t> random_blocks(QType q, int rows, int k, std::mt19937& rng) {
    const int blocks = k / static_cast<int>(ggml_block_elems(q)), size = static_cast<int>(ggml_block_bytes(q));
    std::vector<std::uint8_t> data(static_cast<std::size_t>(rows) * blocks * size);
    for (auto& v : data) { v = static_cast<std::uint8_t>(rng()); }
    const auto scale = [&](std::uint8_t* p) { // finite binary16, exponent fields 5..20
        const auto w = static_cast<std::uint16_t>((rng() & 0x8000U) | (rng() & 0x3FFU) | ((5U + rng() % 16U) << 10));
        p[0] = static_cast<std::uint8_t>(w & 0xFF);
        p[1] = static_cast<std::uint8_t>(w >> 8);
    };
    for (int i = 0; i < rows * blocks; ++i) {
        std::uint8_t* b = data.data() + static_cast<std::size_t>(i) * size;
        if (q == QType::GGML_Q6_K) {
            scale(b + 208);
        } else {
            scale(b);
            if (q == QType::GGML_Q4_K || q == QType::GGML_Q5_K) { scale(b + 2); }
        }
    }
    return data;
}

// ggml_tiles16_v1 of ggml_rows_v1 rows: tile i at 16 i row_bytes, block b of its row r at (16 b + r)
// block bytes, the last tile padded with zero blocks.
std::vector<std::uint8_t> to_tiles16(const std::vector<std::uint8_t>& rows_data, QType q, int rows, int k) {
    const std::size_t size = ggml_block_bytes(q), blocks = static_cast<std::size_t>(k) / ggml_block_elems(q);
    const std::size_t row_bytes = blocks * size, tiles = (static_cast<std::size_t>(rows) + 15) / 16;
    std::vector<std::uint8_t> out(tiles * 16 * row_bytes, 0);
    for (int r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < blocks; ++b) {
            std::memcpy(out.data() + static_cast<std::size_t>(r / 16) * 16 * row_bytes + (16 * b + r % 16) * size,
                        rows_data.data() + static_cast<std::size_t>(r) * row_bytes + b * size, size);
        }
    }
    return out;
}

void run_ggml(QType q, int rows, int k, bool tiles16, bool permute, bool bf16_out, std::mt19937& rng) {
    const std::string label = "ggml format " + std::to_string(static_cast<int>(q)) + " k=" + std::to_string(k) +
                              (tiles16 ? " tiles16" : " rows") + (permute ? " perm" : "") + (bf16_out ? " bf16" : " fp32");
    const std::size_t row_bytes = static_cast<std::size_t>(k) / ggml_block_elems(q) * ggml_block_bytes(q);
    const auto weights = random_blocks(q, rows, k, rng);
    std::vector<double> w(static_cast<std::size_t>(rows) * k), offsets(w.size());
    for (int r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < static_cast<std::size_t>(k) / ggml_block_elems(q); ++b) {
            const std::uint8_t* block = weights.data() + static_cast<std::size_t>(r) * row_bytes + b * ggml_block_bytes(q);
            const std::size_t at      = static_cast<std::size_t>(r) * k + b * ggml_block_elems(q);
            test::ggml_oracle::decode_block(q, block, w.data() + at);
            test::ggml_oracle::block_offsets(q, block, offsets.data() + at);
        }
    }
    const int pb = 128;
    std::vector<std::int32_t> perm(static_cast<std::size_t>(k / pb));
    for (std::size_t i = 0; i < perm.size(); ++i) { perm[i] = static_cast<std::int32_t>(i); }
    if (permute) { std::shuffle(perm.begin(), perm.end(), rng); }
    DeviceBuffer dperm(perm.size() * 4);
    dperm.copy_from_host(perm.data(), perm.size() * 4, 0);
    const auto* perm_device = permute ? static_cast<const std::int32_t*>(dperm.p) : nullptr;
    // Two segments over the rows' halves (a tiles16 cut at a multiple of 16): into one output, and
    // the rows split among three outputs at bounds inside both segments.
    const int half = tiles16 ? rows / 32 * 16 : rows / 2;
    std::vector<DeviceBuffer> stored;
    const auto upload = [&](int first, int count) {
        const std::vector<std::uint8_t> part(weights.begin() + static_cast<std::ptrdiff_t>(first * row_bytes),
                                             weights.begin() + static_cast<std::ptrdiff_t>((first + count) * row_bytes));
        const auto bytes = tiles16 ? to_tiles16(part, q, count, k) : part;
        stored.emplace_back(bytes.size());
        stored.back().copy_from_host(bytes.data(), bytes.size(), 0);
        return static_cast<const std::uint8_t*>(stored.back().p);
    };
    const ops::GgmlLayout layout = tiles16 ? ops::GgmlLayout::Tiles16 : ops::GgmlLayout::Rows;
    const ops::GgmlSegment whole[] = {{q, upload(0, half), half, 0, 0, layout},
                                      {q, upload(half, rows - half), rows - half, half, 0, layout}};
    const std::int32_t bounds[] = {0, half - 3, half + 5, rows};
    for (const int columns : kWidths) {
        const Columns x = random_columns(k, columns, rng);
        DeviceBuffer dx(x.bits.size() * 2), dout(static_cast<std::size_t>(rows) * columns * 4);
        dx.copy_from_host(x.bits.data(), x.bits.size() * 2, 0);
        Tensor tx(dx.p, DType::BF16, {k, columns});
        Tensor tout(dout.p, bf16_out ? DType::BF16 : DType::FP32, {rows, columns});
        Tensor* const one[] = {&tout};
        ops::precise_linear(tx, whole, one, perm_device, pb, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto got = download(dout, static_cast<std::size_t>(rows) * columns, bf16_out);
        verify(label + " T=" + std::to_string(columns), w, rows, x, permute ? &perm : nullptr, pb, got, bf16_out,
               &offsets);
        if (columns > 20) {
            check_widths(label + " T=" + std::to_string(columns), x, rows, bf16_out, [&](const Tensor& in, Tensor& out) {
                Tensor* const target[] = {&out};
                ops::precise_linear(in, whole, target, perm_device, pb, nullptr);
            });
        }
        if (tiles16) { continue; } // tiles16 segments cut only at whole tiles: the split form is the rows layout's

        std::vector<DeviceBuffer> parts;
        std::vector<Tensor> part_tensors;
        for (int i = 0; i < 3; ++i) {
            parts.emplace_back(static_cast<std::size_t>(bounds[i + 1] - bounds[i]) * columns * 4);
            part_tensors.push_back(Tensor(parts.back().p, bf16_out ? DType::BF16 : DType::FP32, {bounds[i + 1] - bounds[i], columns}));
        }
        Tensor* const outs[] = {&part_tensors[0], &part_tensors[1], &part_tensors[2]};
        std::vector<ops::GgmlSegment> split;
        for (const auto& s : whole) {
            for (int i = 0; i < 3; ++i) { // the segment's rows inside output i
                const int begin = std::max(s.out_offset, bounds[i]), end = std::min(s.out_offset + s.rows, bounds[i + 1]);
                if (begin >= end) { continue; }
                ops::GgmlSegment piece = s;
                piece.data       = s.data + static_cast<std::size_t>(begin - s.out_offset) * row_bytes;
                piece.rows       = end - begin;
                piece.out_offset = begin - bounds[i];
                piece.output     = i;
                split.push_back(piece);
            }
        }
        ops::precise_linear(tx, split, outs, perm_device, pb, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<std::uint32_t> joined(static_cast<std::size_t>(rows) * columns);
        for (int i = 0; i < 3; ++i) {
            const int n = bounds[i + 1] - bounds[i];
            const auto part = download(parts[static_cast<std::size_t>(i)], static_cast<std::size_t>(n) * columns, bf16_out);
            for (int t = 0; t < columns; ++t) {
                for (int r = 0; r < n; ++r) {
                    joined[static_cast<std::size_t>(t) * rows + bounds[i] + r] = part[static_cast<std::size_t>(t) * n + r];
                }
            }
        }
        std::size_t differ = 0;
        for (std::size_t i = 0; i < joined.size(); ++i) { differ += joined[i] != got[i]; }
        check(differ == 0, label + " three outputs T=" + std::to_string(columns) + ": " + std::to_string(differ) +
                               " outputs differ from the single output's");
    }
}

// ----------------------------------------------------------------------------------- SwiGLU

void run_swiglu(QType qtype, int channels, int k, std::mt19937& rng) {
    qw::PatternedWeightOptions options;
    options.row_split_scale = qw::RowSplitScalePattern::Small;
    options.row_split_codes = qw::RowSplitCodePattern::Hashed;
    const qw::PackedWeight packed = qw::make_patterned_weight(qtype, 2 * channels, k, 7, options);
    DeviceBuffer payload(packed.payload.size());
    payload.copy_from_host(packed.payload.data(), packed.payload.size(), 0);
    const Weight device = packed.device_weight(payload.p);
    for (const int columns : kWidths) {
        const Columns x = random_columns(k, columns, rng);
        DeviceBuffer dx(x.bits.size() * 2), dout(static_cast<std::size_t>(channels) * columns * 2);
        dx.copy_from_host(x.bits.data(), x.bits.size() * 2, 0);
        Tensor tx(dx.p, DType::BF16, {k, columns});
        Tensor tout(dout.p, DType::BF16, {channels, columns});
        ops::precise_swiglu(tx, device, tout, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto got = download(dout, static_cast<std::size_t>(channels) * columns, true);
        // The output is BF16(SiLU(g) u) of the route's g and u, each within route_bound of the exact
        // sum: where every g and u in those intervals' corners gives the same BF16, the output must be
        // it; otherwise the exact value lies within the route's error of a tie (counted).
        const auto f = [](double g, double u) { return bf16_bits_of_double(g / (1.0 + std::exp(-g)) * u); };
        std::size_t mismatches = 0, near_ties = 0, loop_differs = 0;
        for (int t = 0; t < columns; ++t) {
            for (int i = 0; i < channels; ++i) {
                double sums[2], bounds[2], loops[2];
                for (int m = 0; m < 2; ++m) {
                    double hi = 0.0, lo = 0.0, magnitude = 0.0, loop = 0.0;
                    for (int j = 0; j < k; ++j) {
                        const double a = qw::logical_weight_fp64(packed, m * channels + i, j);
                        const double b = bf16_value(x.bits[static_cast<std::size_t>(t) * k + j]);
                        loop = std::fma(a, b, loop);
                        const double p = a * b, pe = std::fma(a, b, -p);
                        const double s = hi + p, bb = s - hi, se = (hi - (s - bb)) + (p - bb);
                        hi = s;
                        lo += se + pe;
                        magnitude += std::fabs(p);
                    }
                    sums[m]   = hi + lo;
                    bounds[m] = route_bound(k, magnitude) + std::ldexp(std::fabs(sums[m]), -52);
                    loops[m]  = loop;
                }
                const std::uint32_t bits = got[static_cast<std::size_t>(t) * channels + i];
                const std::uint16_t expected = f(sums[0], sums[1]);
                bool determined = true;
                for (const double dg : {-bounds[0], bounds[0]}) {
                    for (const double du : {-bounds[1], bounds[1]}) { determined = determined && f(sums[0] + dg, sums[1] + du) == expected; }
                }
                if (determined) {
                    mismatches += bits != expected;
                } else {
                    ++near_ties;
                }
                loop_differs += bits != f(loops[0], loops[1]);
            }
        }
        const std::string label = "swiglu format " + std::to_string(static_cast<int>(qtype)) + " T=" + std::to_string(columns);
        g_outputs += static_cast<std::size_t>(channels) * columns;
        g_near_ties += near_ties;
        g_loop_differs += loop_differs;
        check(mismatches == 0, label + ": " + std::to_string(mismatches) + " outputs are not correctly rounded");
        if (near_ties != 0 || loop_differs != 0) {
            std::cout << label << ": " << near_ties << " outputs within the route's error of a tie, " << loop_differs
                      << " differ from the binary64 loop\n";
        }
        if (columns > 20) {
            check_widths(label, x, channels, true, [&](const Tensor& in, Tensor& out) { ops::precise_swiglu(in, device, out, nullptr); });
        }
    }
}

// ------------------------------------------------------------ row ranges, residual add, GDN gates

// The exact sum of a weight row and a BF16 column as a binary64 pair (two-product, two-sum) and the
// total magnitude of its terms.
struct ExactSum {
    double hi = 0.0, lo = 0.0, magnitude = 0.0;
};

ExactSum exact_sum(const std::function<double(int)>& w, const std::uint16_t* x, int k) {
    ExactSum s;
    for (int j = 0; j < k; ++j) {
        const double a = w(j), b = bf16_value(x[j]);
        const double p = a * b, pe = std::fma(a, b, -p);
        const double t = s.hi + p, bb = t - s.hi, te = (s.hi - (t - bb)) + (p - bb);
        s.hi = t;
        s.lo += te + pe;
        s.magnitude += std::fabs(p);
    }
    return s;
}

// precise_linear_rows: rows [begin, begin + count) of a weight into rows [out_row, out_row + count) of
// an output equal those rows of precise_linear's whole output, bit for bit.
void run_rows(QType qtype, int n, int k, int begin, int count, int out_row, qw::PatternedWeightOptions options,
              std::mt19937& rng) {
    const qw::PackedWeight packed = qw::make_patterned_weight(qtype, n, k, 13, options);
    DeviceBuffer payload(packed.payload.size());
    payload.copy_from_host(packed.payload.data(), packed.payload.size(), 0);
    const Weight device = packed.device_weight(payload.p);
    const Columns x     = random_columns(k, 9, rng);
    DeviceBuffer dx(x.bits.size() * 2);
    dx.copy_from_host(x.bits.data(), x.bits.size() * 2, 0);
    const Tensor tx(dx.p, DType::BF16, {k, x.columns});
    DeviceBuffer full(static_cast<std::size_t>(n) * x.columns * 4);
    Tensor tfull(full.p, DType::FP32, {n, x.columns});
    ops::precise_linear(tx, device, tfull, nullptr);
    const int rows = out_row + count + 3;
    DeviceBuffer part(static_cast<std::size_t>(rows) * x.columns * 4);
    Tensor tpart(part.p, DType::FP32, {rows, x.columns});
    ops::precise_linear_rows(tx, device, begin, count, tpart, out_row, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto all  = download(full, static_cast<std::size_t>(n) * x.columns, false);
    const auto some = download(part, static_cast<std::size_t>(rows) * x.columns, false);
    std::size_t differ = 0;
    for (int t = 0; t < x.columns; ++t) {
        for (int r = 0; r < count; ++r) {
            differ += some[static_cast<std::size_t>(t) * rows + out_row + r] !=
                      all[static_cast<std::size_t>(t) * n + begin + r];
        }
    }
    check(differ == 0, "rows format " + std::to_string(static_cast<int>(qtype)) + ": " + std::to_string(differ) +
                           " outputs differ from precise_linear's");
}

// precise_linear_add: every output is the correctly rounded BF16 value of residual + the exact sum,
// except within the route's error (plus one binary64 addition) of a tie.
void run_add(QType qtype, int n, int k, qw::PatternedWeightOptions options, std::mt19937& rng) {
    const qw::PackedWeight packed = qw::make_patterned_weight(qtype, n, k, 17, options);
    DeviceBuffer payload(packed.payload.size());
    payload.copy_from_host(packed.payload.data(), packed.payload.size(), 0);
    const Weight device = packed.device_weight(payload.p);
    const Columns x     = random_columns(k, 9, rng);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(n) * x.columns);
    std::normal_distribution<float> normal(0.0F, 4.0F);
    for (auto& r : residual) { r = bf16_bits_of_float(normal(rng)); }
    DeviceBuffer dx(x.bits.size() * 2), dr(residual.size() * 2);
    dx.copy_from_host(x.bits.data(), x.bits.size() * 2, 0);
    dr.copy_from_host(residual.data(), residual.size() * 2, 0);
    const Tensor tx(dx.p, DType::BF16, {k, x.columns});
    Tensor tr(dr.p, DType::BF16, {n, x.columns});
    ops::precise_linear_add(tx, device, tr, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto got = download(dr, residual.size(), true);
    std::size_t imprecise = 0, near_ties = 0;
    for (int t = 0; t < x.columns; ++t) {
        for (int r = 0; r < n; ++r) {
            const ExactSum s = exact_sum([&](int j) { return qw::logical_weight_fp64(packed, r, j); },
                                         x.bits.data() + static_cast<std::size_t>(t) * k, k);
            const double base = bf16_value(residual[static_cast<std::size_t>(t) * n + r]);
            // hi + base + lo, as a pair.
            const double h = s.hi + base, bb = h - s.hi, he = (s.hi - (h - bb)) + (base - bb);
            const double hi = h, lo = s.lo + he;
            const std::uint32_t bits = got[static_cast<std::size_t>(t) * n + r];
            const double y = bf16_value(static_cast<std::uint16_t>(bits));
            const double below = (neighbour(bits, true, -1) + y) / 2, above = (neighbour(bits, true, 1) + y) / 2;
            const double lo_side = (hi - std::min(below, above)) + lo, hi_side = (hi - std::max(below, above)) + lo;
            if (lo_side < 0.0 || hi_side > 0.0) {
                const double miss  = lo_side < 0.0 ? -lo_side : hi_side;
                const double bound = route_bound(k, s.magnitude) + std::ldexp(std::fabs(hi), -52);
                ++(miss <= bound ? near_ties : imprecise);
            }
        }
    }
    g_outputs += static_cast<std::size_t>(n) * x.columns;
    g_near_ties += near_ties;
    check(imprecise == 0, "add format " + std::to_string(static_cast<int>(qtype)) + ": " + std::to_string(imprecise) +
                              " outputs are not correctly rounded");
}

// precise_gdn_gating: g and beta are the FP32 roundings of the formula at the exact sums, except where
// the route's error of a sum moves the rounding (counted as near ties).
void run_gdn_gating(int heads, int k, bool stacked, std::mt19937& rng) {
    std::vector<std::uint16_t> ab(static_cast<std::size_t>(2) * heads * k);
    std::normal_distribution<float> normal(0.0F, 0.03F);
    for (auto& v : ab) { v = bf16_bits_of_float(normal(rng)); }
    std::vector<float> a_log(heads), dt_bias(heads);
    for (int h = 0; h < heads; ++h) {
        a_log[h]   = 0.5F * normal(rng) / 0.03F;
        dt_bias[h] = normal(rng) / 0.03F;
    }
    const Columns x = random_columns(k, 9, rng);
    DeviceBuffer dab(ab.size() * 2), dx(x.bits.size() * 2), dal(heads * 4), ddt(heads * 4);
    dab.copy_from_host(ab.data(), ab.size() * 2, 0);
    dx.copy_from_host(x.bits.data(), x.bits.size() * 2, 0);
    dal.copy_from_host(a_log.data(), heads * 4, 0);
    ddt.copy_from_host(dt_bias.data(), heads * 4, 0);
    const auto bf16_weight = [&](void* data, int rows) {
        Weight w{};
        w.qtype  = QType::BF16;
        w.layout = QuantLayout::Contiguous;
        w.n      = rows;
        w.k      = k;
        w.qdata  = data;
        return w;
    };
    const Weight parent = bf16_weight(dab.p, 2 * heads);
    const Weight a      = bf16_weight(dab.p, heads);
    const Weight b      = bf16_weight(static_cast<std::uint16_t*>(dab.p) + static_cast<std::size_t>(heads) * k, heads);
    const Weight* one[] = {&parent};
    const Weight* two[] = {&a, &b};
    DeviceBuffer dg(static_cast<std::size_t>(heads) * x.columns * 4), dbeta(dg.bytes);
    const Tensor tx(dx.p, DType::BF16, {k, x.columns});
    const Tensor tal(dal.p, DType::FP32, {heads});
    const Tensor tdt(ddt.p, DType::FP32, {heads});
    Tensor tg(dg.p, DType::FP32, {heads, x.columns});
    Tensor tbeta(dbeta.p, DType::FP32, {heads, x.columns});
    if (stacked) {
        ops::precise_gdn_gating(tx, two, tal, tdt, tg, tbeta, nullptr);
    } else {
        ops::precise_gdn_gating(tx, one, tal, tdt, tg, tbeta, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto g    = download(dg, static_cast<std::size_t>(heads) * x.columns, false);
    const auto beta = download(dbeta, static_cast<std::size_t>(heads) * x.columns, false);
    const auto gate = [&](int h, double s) {
        const double v = s + dt_bias[h];
        return f32_bits_of_double(-std::exp(static_cast<double>(a_log[h])) * (v > 40.0 ? v : std::log1p(std::exp(v))));
    };
    const auto sigmoid = [](double s) { return f32_bits_of_double(1.0 / (1.0 + std::exp(-s))); };
    std::size_t mismatches = 0, near_ties = 0;
    for (int t = 0; t < x.columns; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int m = 0; m < 2; ++m) {
                const int row    = m * heads + h;
                const ExactSum s = exact_sum([&](int j) { return bf16_value(ab[static_cast<std::size_t>(row) * k + j]); },
                                             x.bits.data() + static_cast<std::size_t>(t) * k, k);
                const double sum   = s.hi + s.lo;
                const double bound = route_bound(k, s.magnitude) + std::ldexp(std::fabs(sum), -52);
                const auto f       = [&](double v) { return m == 0 ? gate(h, v) : sigmoid(v); };
                const std::uint32_t expected = f(sum);
                const std::uint32_t bits = (m == 0 ? g : beta)[static_cast<std::size_t>(t) * heads + h];
                if (f(sum - bound) == expected && f(sum + bound) == expected) {
                    // binary64 exp/log1p may differ from the host's by an ulp: one FP32 ulp of slack.
                    mismatches += std::fabs(f32_value(bits) - f32_value(expected)) >
                                  std::fabs(f32_value(expected)) * std::ldexp(1.0, -23);
                } else {
                    ++near_ties;
                }
            }
        }
    }
    g_outputs += static_cast<std::size_t>(2) * heads * x.columns;
    g_near_ties += near_ties;
    check(mismatches == 0, std::string("gdn gating ") + (stacked ? "two weights" : "one parent") + ": " +
                               std::to_string(mismatches) + " outputs beyond one FP32 ulp of the formula");
}

// precise_sparse_moe at the 35B-A3B geometry (Q4 routed gate/up, Q6 routed down, Q8 shared expert, BF16
// router): within one BF16 ulp of a binary64 evaluation that rounds the activations to BF16 as the route
// does, almost always the correctly rounded value; and a token's bits are the same alone as among others.
void run_sparse_moe(std::mt19937& rng) {
    constexpr int H = 2048, E = 256, I = 512, T = 8, kTop = 8;
    qw::PatternedWeightOptions row_split;
    row_split.row_split_scale = qw::RowSplitScalePattern::Small;
    row_split.row_split_codes = qw::RowSplitCodePattern::Hashed;
    const qw::PackedWeight gate_up     = qw::make_patterned_weight(QType::Q4_G64_FP16, E * 2 * I, H, 21, row_split);
    const qw::PackedWeight down        = qw::make_patterned_weight(QType::Q6_G64_FP16, E * H, I, 22, row_split);
    const qw::PackedWeight shared_gu   = qw::make_patterned_weight(QType::Q8_G32_FP16, 2 * I, H, 23, row_split);
    const qw::PackedWeight shared_down = qw::make_patterned_weight(QType::Q8_G32_FP16, H, I, 24, row_split);
    std::vector<std::uint16_t> router(static_cast<std::size_t>(E + 1) * H);
    std::normal_distribution<float> normal(0.0F, 0.05F);
    for (auto& v : router) { v = bf16_bits_of_float(normal(rng)); }
    std::vector<DeviceBuffer> payloads;
    const auto upload = [&](const qw::PackedWeight& p) {
        payloads.emplace_back(p.payload.size());
        payloads.back().copy_from_host(p.payload.data(), p.payload.size(), 0);
        return p.device_weight(payloads.back().p);
    };
    ops::SparseMoeWeights weights{};
    DeviceBuffer drouter(router.size() * 2);
    drouter.copy_from_host(router.data(), router.size() * 2, 0);
    weights.router_shared_gate.qtype  = QType::BF16;
    weights.router_shared_gate.layout = QuantLayout::Contiguous;
    weights.router_shared_gate.n      = E + 1;
    weights.router_shared_gate.k      = H;
    weights.router_shared_gate.qdata  = drouter.p;
    weights.routed_gate_up            = upload(gate_up);
    weights.routed_down               = upload(down);
    weights.shared_gate_up            = upload(shared_gu);
    weights.shared_down               = upload(shared_down);

    const Columns x = random_columns(H, T, rng);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(H) * T);
    for (auto& r : residual) { r = bf16_bits_of_float(normal(rng) * 40.0F); }
    const auto run = [&](int first, int count) {
        DeviceBuffer dx(static_cast<std::size_t>(H) * count * 2), dr(static_cast<std::size_t>(H) * count * 2);
        dx.copy_from_host(x.bits.data() + static_cast<std::size_t>(first) * H, dx.bytes, 0);
        dr.copy_from_host(residual.data() + static_cast<std::size_t>(first) * H, dr.bytes, 0);
        DeviceBuffer scratch(ops::precise_sparse_moe_workspace_bytes(count));
        WorkspaceArena arena(DeviceSpan{scratch.p, scratch.bytes});
        const Tensor tx(dx.p, DType::BF16, {H, count});
        Tensor tr(dr.p, DType::BF16, {H, count});
        ops::precise_sparse_moe(tx, weights, tr, arena, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        return download(dr, static_cast<std::size_t>(H) * count, true);
    };
    const auto got   = run(0, T);
    const auto alone = run(5, 1);
    std::size_t differ = 0;
    for (int r = 0; r < H; ++r) { differ += alone[r] != got[static_cast<std::size_t>(5) * H + r]; }
    check(differ == 0, "sparse moe: " + std::to_string(differ) + " outputs of a token alone differ from its batch's");

    // The binary64 evaluation: scores, top 8 (lower id at a tie), softmax, BF16-rounded activations.
    const auto dot = [&](const std::function<double(int)>& w, const double* v, int k) {
        double s = 0.0;
        for (int j = 0; j < k; ++j) { s += w(j) * v[j]; }
        return s;
    };
    std::size_t outside = 0, rounded = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<double> xv(H);
        for (int j = 0; j < H; ++j) { xv[j] = bf16_value(x.bits[static_cast<std::size_t>(t) * H + j]); }
        std::vector<double> scores(E + 1);
        for (int e = 0; e <= E; ++e) {
            scores[e] = dot([&](int j) { return bf16_value(router[static_cast<std::size_t>(e) * H + j]); }, xv.data(), H);
        }
        std::vector<int> ranked(E);
        for (int e = 0; e < E; ++e) { ranked[e] = e; }
        std::sort(ranked.begin(), ranked.end(),
                  [&](int a, int b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); });
        std::vector<double> out(H, 0.0);
        double denominator = 0.0;
        for (int r = 0; r < kTop; ++r) { denominator += std::exp(scores[ranked[r]] - scores[ranked[0]]); }
        const auto expert = [&](const qw::PackedWeight& gu, int gu_row0, const qw::PackedWeight& dn, int dn_row0,
                                double scale) {
            std::vector<double> act(I);
            for (int i = 0; i < I; ++i) {
                const double g = dot([&](int j) { return qw::logical_weight_fp64(gu, gu_row0 + i, j); }, xv.data(), H);
                const double u = dot([&](int j) { return qw::logical_weight_fp64(gu, gu_row0 + I + i, j); }, xv.data(), H);
                act[i] = bf16_value(bf16_bits_of_double(g / (1.0 + std::exp(-g)) * u));
            }
            for (int r = 0; r < H; ++r) {
                out[r] += scale * dot([&](int j) { return qw::logical_weight_fp64(dn, dn_row0 + r, j); }, act.data(), I);
            }
        };
        for (int r = 0; r < kTop; ++r) {
            const int e = ranked[r];
            expert(gate_up, e * 2 * I, down, e * H, std::exp(scores[e] - scores[ranked[0]]) / denominator);
        }
        expert(shared_gu, 0, shared_down, 0, 1.0 / (1.0 + std::exp(-scores[E])));
        for (int r = 0; r < H; ++r) {
            const double value = bf16_value(residual[static_cast<std::size_t>(t) * H + r]) + out[r];
            const auto bits    = static_cast<std::uint16_t>(got[static_cast<std::size_t>(t) * H + r]);
            const double y     = bf16_value(bits);
            const double ulp   = std::fabs(neighbour(bits, true, 1) - y);
            outside += std::fabs(y - value) > ulp;
            rounded += bits == bf16_bits_of_double(value);
        }
    }
    const double share = static_cast<double>(rounded) / (static_cast<double>(H) * T);
    std::cout << "sparse moe: " << 100.0 * share << "% correctly rounded, " << outside << " beyond one BF16 ulp\n";
    check(outside == 0 && share >= 0.99, "sparse moe: outside one BF16 ulp of the evaluation or under 99% rounded");
}

} // namespace

int main() {
    try {
        std::mt19937 rng(20261019);
        run_bf16_weight(200, 2560, rng, true);
        run_bf16_weight(67, 520, rng, false);
        run_bf16_tensors(2560, rng);
        qw::PatternedWeightOptions row_split;
        row_split.row_split_scale = qw::RowSplitScalePattern::Small;
        row_split.row_split_codes = qw::RowSplitCodePattern::Hashed;
        for (const QType q : {QType::Q2_G64_FP16, QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16,
                              QType::Q8_G32_FP16}) {
            run_packed(q, 200, 2560, 3, row_split, rng);
            run_packed(q, 67, 520, 5, row_split, rng, false); // a partial last group
        }
        run_packed(QType::FP8_E4M3FN_ROW_BF16, 200, 2560, 9, {}, rng);
        qw::PatternedWeightOptions nvfp4;
        nvfp4.weight_scale_divisor = 3.0F;
        nvfp4.input_scale_divisor  = 1.0F;
        run_packed(QType::NVFP4, 256, 2560, 11, nvfp4, rng);
        for (const QType q : {QType::GGML_Q2_0, QType::GGML_IQ4_NL, QType::GGML_IQ4_XS, QType::GGML_Q4_K,
                              QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ2_S, QType::GGML_IQ3_XXS,
                              QType::GGML_IQ3_S}) {
            run_ggml(q, 150, 2560, false, true, true, rng);
            if (ggml_block_elems(q) < 256) { run_ggml(q, 150, 384, false, false, false, rng); }
            if (ops::ggml_tiles16_supported(q)) { run_ggml(q, 150, 2560, true, true, true, rng); }
        }
        run_swiglu(QType::Q8_G32_FP16, 70, 2560, rng);
        run_swiglu(QType::Q4_G64_FP16, 70, 520, rng);
        run_rows(QType::NVFP4, 256, 2560, 128, 128, 3, nvfp4, rng);
        run_rows(QType::Q4_G64_FP16, 200, 2560, 37, 50, 5, row_split, rng);
        run_add(QType::Q5_G64_FP16, 200, 2560, row_split, rng);
        run_add(QType::NVFP4, 256, 2560, nvfp4, rng);
        run_gdn_gating(48, 5120, false, rng);
        run_gdn_gating(32, 2048, true, rng);
        run_sparse_moe(rng);
    } catch (const std::exception& e) {
        std::cerr << "FAIL exception: " << e.what() << "\n";
        return 1;
    }
    std::cout << "precise_linear: " << g_outputs << " outputs, " << g_near_ties
              << " within the route's error of a tie, " << g_loop_differs << " differ from the binary64 loop\n";
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "precise_linear: all checks passed\n";
    return 0;
}
