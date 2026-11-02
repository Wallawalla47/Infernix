// bf16_p12 (lossless 12-bit BF16 packing, core/weight.h QuantLayout::P12Rows): the packing's exactness and format,
// and ops::linear over packed weights.
//
// Codec. Matrices of edge values (zeros of both signs, subnormals, infinities, NaNs, the largest and smallest
// normals, rows whose every weight escapes, rows with tied windows, rows at both ends of the exponent range) and
// real-like rows are packed on the device. An independent host decoder of the stored planes must give back every
// weight's bits; each row's base must be the lowest base of a 15-exponent window holding most of its nonzero normal
// weights, its escapes exactly the weights outside that window (zeros and subnormals included), in column order;
// and the device decoder every route shares must agree.
//
// Linear. The packed Flash-Next shapes (each registered route's widths) and an unregistered shape (the skinny GEMV
// in 16-column passes) against the FP64 product of the weights and activations under the BF16 routes' A16
// criterion; repeat calls give the same bits; each BF16 output is its call's FP32 output rounded; a view of rows
// [r0, r1) of a packed parent (escapes before and inside it) meets the oracle; under A16ColumnInvariant a column's
// bits do not depend on the call's width; a graph replay gives the eager bits; and no call takes workspace.
//
// projection_fp32 over a packed head: FP32 logits within the Op's FP32 accumulation bound of FP64, a column's bits
// the same at every width.
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "infernix/ops/bf16_p12.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/projection_fp32.h"
#include "ops/linear/bf16_p12/bf16_p12_dispatch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;

constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

int exponent(std::uint16_t bits) { return (bits >> 7) & 0xFF; }

// ---------------------------------------------------------------------------------------------------------------
// Codec

struct Packed {
    std::int32_t n = 0, k = 0;
    std::uint64_t escapes = 0;
    WeightGeometry geometry;
    DeviceBuffer storage;
    WeightParent parent;
    std::vector<std::uint8_t> host; // the planes as stored

    // The packed [r0, r1) row range as ops::linear takes it (the loader's binding).
    [[nodiscard]] Weight weight(std::int32_t r0, std::int32_t r1) const {
        WeightView view{{static_cast<std::uint64_t>(r1 - r0), static_cast<std::uint64_t>(k)},
                        {{&parent, static_cast<std::uint64_t>(r0) * k, static_cast<std::uint64_t>(r1) * k}}};
        return native_weight(view);
    }
};

Packed pack(const std::vector<std::uint16_t>& bits, std::int32_t n, std::int32_t k) {
    DeviceBuffer source = to_device(bits);
    DeviceBuffer scratch(ops::bf16_p12_pack_scratch_bytes(n)), count(4);
    Packed out;
    out.n = n, out.k = k;
    ops::bf16_p12_count(source.p, n, k, scratch.p, static_cast<std::uint32_t*>(count.p), nullptr);
    cuda_synchronize();
    out.escapes  = from_device<std::uint32_t>(count, 1).front();
    out.geometry = bf16_p12_geometry(n, k, out.escapes);
    out.storage  = DeviceBuffer(out.geometry.bytes);
    ops::bf16_p12_pack(source.p, n, k, out.escapes, scratch.p, out.storage.p, nullptr);
    cuda_synchronize();
    out.host.resize(out.geometry.bytes);
    out.storage.copy_to_host(out.host.data(), out.host.size());
    out.parent.geometry = out.geometry;
    out.parent.data = out.parent.device = static_cast<const std::byte*>(out.storage.p);
    return out;
}

std::uint32_t word_at(const std::vector<std::uint8_t>& bytes, std::uint64_t offset) {
    std::uint32_t value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

// The planes decoded on the host, from their stored layout alone.
std::vector<std::uint16_t> host_decode(const Packed& p) {
    const auto& g = p.geometry;
    std::vector<std::uint16_t> out(static_cast<std::size_t>(p.n) * p.k);
    for (std::int32_t row = 0; row < p.n; ++row) {
        const std::uint32_t word = word_at(p.host, g.scale_offset + 4ULL * row);
        const std::uint32_t next = word_at(p.host, g.scale_offset + 4ULL * (row + 1));
        const int base           = static_cast<int>(word & 0xFF);
        for (std::int32_t col = 0; col < p.k; ++col) {
            const std::uint64_t at = static_cast<std::uint64_t>(row) * p.k + col;
            const std::uint8_t code = p.host[at];
            const int nibble        = (p.host[g.high_offset + at / 2] >> (4 * (col & 1))) & 0xF;
            std::uint16_t bits      = 0xFFFF;
            if (nibble != 15) {
                bits = static_cast<std::uint16_t>(((code & 0x80) << 8) | ((base + nibble) << 7) | (code & 0x7F));
            } else {
                for (std::uint32_t e = word >> 8; e < next >> 8; ++e) {
                    const std::uint32_t escape = word_at(p.host, g.escape_offset + 4ULL * e);
                    if (static_cast<std::int32_t>(escape >> 16) == col) { bits = static_cast<std::uint16_t>(escape); }
                }
            }
            out[at] = bits;
        }
    }
    return out;
}

// The format's rule for a row: the lowest base b in [1, 240] whose window [b, b + 14] holds the most weights.
int expected_base(std::span<const std::uint16_t> row) {
    int histogram[256] = {};
    for (const auto bits : row) { ++histogram[exponent(bits)]; }
    int best = -1, best_base = 1;
    for (int b = 1; b + 14 < 255; ++b) {
        int s = 0;
        for (int e = b; e < b + 15; ++e) { s += histogram[e]; }
        if (s > best) { best = s, best_base = b; }
    }
    return best_base;
}

int check_codec(const char* label, const std::vector<std::uint16_t>& bits, std::int32_t n, std::int32_t k) {
    int failures   = 0;
    const Packed p = pack(bits, n, k);
    if (host_decode(p) != bits) {
        std::cerr << label << ": the stored planes do not decode to the weights\n";
        ++failures;
    }
    std::uint64_t escapes = 0;
    for (std::int32_t row = 0; row < n && failures < 8; ++row) {
        const std::span<const std::uint16_t> weights(bits.data() + static_cast<std::size_t>(row) * k, k);
        const std::uint32_t word = word_at(p.host, p.geometry.scale_offset + 4ULL * row);
        const std::uint32_t next = word_at(p.host, p.geometry.scale_offset + 4ULL * (row + 1));
        const int base           = expected_base(weights);
        std::vector<std::uint32_t> want;
        for (std::int32_t col = 0; col < k; ++col) {
            const int e = exponent(weights[col]);
            if (e == 0 || e < base || e > base + 14) { want.push_back(static_cast<std::uint32_t>(col) << 16 | weights[col]); }
        }
        std::vector<std::uint32_t> got;
        for (std::uint32_t e = word >> 8; e < next >> 8; ++e) { got.push_back(word_at(p.host, p.geometry.escape_offset + 4ULL * e)); }
        if (static_cast<int>(word & 0xFF) != base || got != want || (word >> 8) != escapes) {
            std::cerr << label << ": row " << row << " has base " << (word & 0xFF) << " (want " << base << ") and "
                      << got.size() << " escapes (want " << want.size() << ", in column order)\n";
            ++failures;
        }
        escapes += want.size();
    }
    if (escapes != p.escapes || word_at(p.host, p.geometry.scale_offset + 4ULL * n) >> 8 != escapes) {
        std::cerr << label << ": " << p.escapes << " escapes counted, the rows hold " << escapes << '\n';
        ++failures;
    }
    DeviceBuffer decoded(bits.size() * 2);
    ops::detail::bf16_p12_unpack(p.weight(0, n), decoded.p, nullptr);
    cuda_synchronize();
    if (from_device<std::uint16_t>(decoded, bits.size()) != bits) {
        std::cerr << label << ": the device decoder differs\n";
        ++failures;
    }
    return failures;
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    return static_cast<std::uint16_t>(word >> 16); // truncation keeps the patterns below exact
}

// Real-like rows (N(0, 0.02) with rare outliers), then edge rows.
std::vector<std::uint16_t> codec_matrix(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    std::uniform_int_distribution<int> rare(0, 999);
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(n) * k);
    for (auto& b : bits) {
        float v = normal(rng);
        if (rare(rng) == 0) { v *= 1e4F; }
        if (rare(rng) == 1) { v = 0.0F; }
        b = f32_to_bf16(v);
    }
    const auto row = [&](std::int32_t r) { return bits.data() + static_cast<std::size_t>(r) * k; };
    const std::uint16_t specials[] = {0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x8080, 0x7F7F, 0xFF7F,
                                      0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81, 0x3F80, 0xBF80, 0x0040};
    if (n >= 8) {
        for (std::int32_t c = 0; c < k; ++c) { row(0)[c] = specials[c % 16]; }                   // edge values
        for (std::int32_t c = 0; c < k; ++c) { row(1)[c] = static_cast<std::uint16_t>((c * 37 % 255) << 7 | (c & 0x7F)); }
        for (std::int32_t c = 0; c < k; ++c) { row(2)[c] = c % 2 ? 0x0000 : 0x8001; }           // every weight escapes
        for (std::int32_t c = 0; c < k; ++c) {                                                   // tied windows
            row(3)[c] = static_cast<std::uint16_t>(((c % 2 ? 30 : 60) << 7) | (c & 0x7F) | (c % 3 == 0 ? 0x8000 : 0));
        }
        for (std::int32_t c = 0; c < k; ++c) { row(4)[c] = static_cast<std::uint16_t>((1 + c % 15) << 7 | 0x55); }
        for (std::int32_t c = 0; c < k; ++c) { row(5)[c] = static_cast<std::uint16_t>((240 + c % 15) << 7 | 0x2A); }
        for (std::int32_t c = 0; c < k; ++c) { row(6)[c] = bf16_bits(std::ldexp(1.0F, (c % 40) - 20)); }
        for (std::int32_t c = 0; c < k; ++c) { row(7)[c] = 0x3F80; }                            // one exponent
    }
    return bits;
}

int run_codec() {
    int failures = 0;
    failures += check_codec("bf16_p12 codec [64, 2560]", codec_matrix(64, 2560, 11U), 64, 2560);
    failures += check_codec("bf16_p12 codec [21, 64]", codec_matrix(21, 64, 12U), 21, 64);
    failures += check_codec("bf16_p12 codec [9, 65528]", codec_matrix(9, 65528, 13U), 9, 65528);
    failures += check_codec("bf16_p12 codec [1, 8]", codec_matrix(1, 8, 14U), 1, 8);
    return failures;
}

// ---------------------------------------------------------------------------------------------------------------
// Linear

std::vector<std::uint16_t> linear_weights(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 0.02F);
    std::uniform_int_distribution<int> rare(0, 1999);
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(n) * k);
    for (auto& b : bits) {
        float v = normal(rng);
        if (rare(rng) == 0) { v *= 300.0F; } // outside its row's window: an escape
        if (rare(rng) == 1) { v = 0.0F; }
        b = f32_to_bf16(v);
    }
    return bits;
}

std::vector<std::uint16_t> activations(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    for (auto& b : bits) { b = f32_to_bf16(normal(rng)); }
    return bits;
}

double oracle(const std::vector<std::uint16_t>& w, std::int32_t k, std::int32_t row, const std::vector<std::uint16_t>& x,
              std::int32_t column) {
    double sum = 0;
    for (std::int32_t i = 0; i < k; ++i) {
        sum += static_cast<double>(bf16_to_f32(w[static_cast<std::size_t>(row) * k + i])) *
               static_cast<double>(bf16_to_f32(x[static_cast<std::size_t>(column) * k + i]));
    }
    return sum;
}

std::vector<std::int32_t> sampled(std::int32_t count, std::int32_t picks) {
    std::vector<std::int32_t> out;
    for (std::int32_t i = 0; i < picks; ++i) { out.push_back(static_cast<std::int32_t>((static_cast<std::int64_t>(count - 1) * i) / std::max(1, picks - 1))); }
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

struct Call {
    std::vector<std::uint16_t> bf16;
    std::vector<float> fp32;
};

Call run(const Weight& w, const DeviceBuffer& x, std::int32_t t, ops::LinearPolicy policy, bool fp32) {
    DeviceArena workspace(256);
    Call out;
    const std::size_t elements = static_cast<std::size_t>(w.n) * t;
    DeviceBuffer y(elements * 4);
    Tensor xt(x.p, DType::BF16, {w.k, t});
    Tensor yt(y.p, DType::BF16, {w.n, t});
    ops::linear(xt, w, yt, policy, workspace, nullptr);
    cuda_synchronize();
    out.bf16 = from_device<std::uint16_t>(y, elements);
    if (fp32) {
        Tensor yf(y.p, DType::FP32, {w.n, t});
        ops::linear(xt, w, yf, policy, workspace, nullptr);
        cuda_synchronize();
        out.fp32 = from_device<float>(y, elements);
    }
    return out;
}

int check_linear_shape(std::int32_t n, std::int32_t k, std::span<const std::int32_t> widths, std::uint32_t seed) {
    const std::string label = "bf16_p12 linear [" + std::to_string(n) + "," + std::to_string(k) + "]";
    int failures            = 0;
    const auto weights      = linear_weights(n, k, seed);
    const Packed p          = pack(weights, n, k);
    const Weight w          = p.weight(0, n);
    const std::int32_t max_t = *std::max_element(widths.begin(), widths.end());
    const auto xs           = activations(k, max_t, seed + 1);
    const DeviceBuffer x    = to_device(xs);
    const auto rows         = sampled(n, 48);
    for (const std::int32_t t : widths) {
        const std::string at = label + " T=" + std::to_string(t);
        if (ops::linear_workspace_capacity_bytes(QType::BF16_P12, n, k, ops::LinearPolicy::A16Only, 1, t) != 0) {
            std::cerr << at << ": the workspace query asks for scratch\n";
            ++failures;
        }
        const Call a = run(w, x, t, ops::LinearPolicy::A16Only, t <= 4 || t == max_t);
        const Call b = run(w, x, t, ops::LinearPolicy::A16Only, false);
        if (a.bf16 != b.bf16) {
            std::cerr << at << ": repeat calls differ\n";
            ++failures;
        }
        std::vector<double> got, want;
        for (const std::int32_t column : sampled(t, 6)) {
            for (const std::int32_t row : rows) {
                got.push_back(bf16_to_f32(a.bf16[static_cast<std::size_t>(column) * n + row]));
                want.push_back(oracle(weights, k, row, xs, column));
            }
        }
        failures += verify_reduction(at, got, want, kA16Tolerance);
        if (!a.fp32.empty()) {
            for (std::size_t i = 0; i < a.bf16.size(); ++i) {
                if (f32_to_bf16(a.fp32[i]) != a.bf16[i]) {
                    std::cerr << at << ": BF16 output " << i << " is not its FP32 output rounded\n";
                    ++failures;
                    break;
                }
            }
        }
    }
    // Column invariance: a T = 40 call's columns against one-column calls.
    const Call wide = run(w, x, std::min(max_t, 40), ops::LinearPolicy::A16ColumnInvariant, false);
    for (const std::int32_t column : sampled(std::min(max_t, 40), 5)) {
        const DeviceBuffer one = to_device(std::vector<std::uint16_t>(xs.begin() + static_cast<std::ptrdiff_t>(column) * k,
                                                                      xs.begin() + static_cast<std::ptrdiff_t>(column + 1) * k));
        const Call single = run(w, one, 1, ops::LinearPolicy::A16ColumnInvariant, false);
        if (!std::equal(single.bf16.begin(), single.bf16.end(), wide.bf16.begin() + static_cast<std::ptrdiff_t>(column) * n)) {
            std::cerr << label << ": column " << column << " under A16ColumnInvariant depends on the call's width\n";
            ++failures;
        }
    }
    return failures;
}

// Rows [r0, r1) of a packed parent, escapes before and inside the range.
int check_row_view() {
    constexpr std::int32_t n = 640, k = 2560, r0 = 192, r1 = 448, t = 24;
    auto weights = linear_weights(n, k, 77U);
    for (std::int32_t row = 0; row < n; row += 7) { weights[static_cast<std::size_t>(row) * k + (row * 13) % k] = 0x0000; }
    const Packed p       = pack(weights, n, k);
    const Weight view    = p.weight(r0, r1);
    const auto xs        = activations(k, t, 78U);
    const DeviceBuffer x = to_device(xs);
    const Call call      = run(view, x, t, ops::LinearPolicy::A16Only, false);
    std::vector<double> got, want;
    for (std::int32_t column = 0; column < t; column += 5) {
        for (std::int32_t row = 0; row < r1 - r0; row += 3) {
            got.push_back(bf16_to_f32(call.bf16[static_cast<std::size_t>(column) * (r1 - r0) + row]));
            want.push_back(oracle(weights, k, r0 + row, xs, column));
        }
    }
    return verify_reduction("bf16_p12 linear rows [192, 448) of [640, 2560]", got, want, kA16Tolerance);
}

// projection_fp32 over a packed head: FP32 logits within the FP32 accumulation bound sum_k |w x| * K * 2^-24 of the
// FP64 product, and each column's bits the same at every width (64-column passes beyond 64).
int check_projection() {
    constexpr std::int32_t n = 4096, k = 2560;
    const std::string label = "bf16_p12 projection_fp32 [4096,2560]";
    int failures            = 0;
    const auto weights      = linear_weights(n, k, 61U);
    const Packed p          = pack(weights, n, k);
    const Weight w          = p.weight(0, n);
    constexpr std::int32_t widths[] = {1, 5, 8, 9, 16, 17, 33, 64, 65, 130};
    const auto xs        = activations(k, 130, 62U);
    const DeviceBuffer x = to_device(xs);
    std::vector<float> reference; // the one-column calls' logits, column by column
    for (const std::int32_t t : widths) {
        DeviceBuffer y(static_cast<std::size_t>(n) * t * 4);
        Tensor xt(x.p, DType::BF16, {k, t});
        Tensor yt(y.p, DType::FP32, {n, t});
        ops::projection_fp32(xt, w, yt, nullptr);
        cuda_synchronize();
        const auto logits = from_device<float>(y, static_cast<std::size_t>(n) * t);
        if (t == 1) {
            for (std::int32_t row = 0; row < n; ++row) {
                double sum = 0, bound = 0;
                for (std::int32_t i = 0; i < k; ++i) {
                    const double product = static_cast<double>(bf16_to_f32(weights[static_cast<std::size_t>(row) * k + i])) *
                                           bf16_to_f32(xs[i]);
                    sum += product;
                    bound += std::fabs(product);
                }
                if (std::fabs(logits[row] - sum) > bound * k * std::ldexp(1.0, -24)) {
                    std::cerr << label << ": row " << row << " is " << logits[row] << ", FP64 " << sum << '\n';
                    ++failures;
                    break;
                }
            }
        }
        // Column c's bits against the same column of a one-column call.
        for (const std::int32_t column : sampled(t, 4)) {
            DeviceBuffer one_y(static_cast<std::size_t>(n) * 4);
            Tensor one_x(static_cast<__nv_bfloat16*>(x.p) + static_cast<std::int64_t>(column) * k, DType::BF16, {k, 1});
            Tensor one_out(one_y.p, DType::FP32, {n, 1});
            ops::projection_fp32(one_x, w, one_out, nullptr);
            cuda_synchronize();
            const auto single = from_device<float>(one_y, n);
            if (!std::equal(single.begin(), single.end(), logits.begin() + static_cast<std::ptrdiff_t>(column) * n,
                            [](float a, float b) { return std::memcmp(&a, &b, 4) == 0; })) {
                std::cerr << label << ": column " << column << " of a " << t << "-column call differs from one column\n";
                ++failures;
            }
        }
    }
    return failures;
}

// A graph replay of a registered route gives the eager bits.
int check_graph(std::int32_t n, std::int32_t k, std::int32_t t) {
    const auto weights   = linear_weights(n, k, 91U);
    const Packed p       = pack(weights, n, k);
    const Weight w       = p.weight(0, n);
    const DeviceBuffer x = to_device(activations(k, t, 92U));
    const Call eager     = run(w, x, t, ops::LinearPolicy::A16Only, false);
    DeviceContext context;
    DeviceArena workspace(256);
    DeviceBuffer y(static_cast<std::size_t>(n) * t * 2);
    Tensor xt(x.p, DType::BF16, {k, t});
    Tensor yt(y.p, DType::BF16, {n, t});
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    definition.capture(context.stream, [&] { ops::linear(xt, w, yt, ops::LinearPolicy::A16Only, workspace, context.stream); });
    graph.instantiate(definition);
    graph.launch(context.stream);
    cuda_synchronize();
    if (from_device<std::uint16_t>(y, static_cast<std::size_t>(n) * t) != eager.bf16) {
        std::cerr << "bf16_p12 linear [" << n << "," << k << "] T=" << t << ": the graph replay differs\n";
        return 1;
    }
    return 0;
}

} // namespace

INFERNIX_OP_TEST(linear_bf16_p12) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = run_codec();
        // Every registered route's widths, and the 16-column passes of an unregistered shape.
        const std::int32_t widths[] = {1, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 100, 128, 129, 192, 193, 256, 300, 1000};
        failures += check_linear_shape(16384, 2560, widths, 21U);
        failures += check_linear_shape(13952, 2560, widths, 22U);
        failures += check_linear_shape(2560, 6144, widths, 24U);
        failures += check_projection();
        const std::int32_t generic[] = {1, 8, 16, 17, 40};
        failures += check_linear_shape(96, 2560, generic, 23U);
        failures += check_row_view();
        failures += check_graph(16384, 2560, 3);
        failures += check_graph(16384, 2560, 24);
        failures += check_graph(13952, 2560, 512);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " bf16_p12\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "bf16_p12: " << error.what() << '\n';
        return 1;
    }
}
