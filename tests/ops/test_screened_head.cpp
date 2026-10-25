// screened_head (include/infernix/ops/screened_head.h) at Qwen3.8-Flash-Next's shapes: the MTP proposal
// head [131072, 2560] (q4_g64_fp16, the top form) and the LM head [248320, 2560] (BF16 and q8_g32_fp16,
// the logits form), each with a 2-bit screen (q2_g64_fp16) fitted as the converter fits it (MSE codes,
// each group's scale stretched to keep its norm).
//
// Checks, against an FP64 oracle of the screen (the copy's decoded codes and scales times the column's
// int8 copy as the contract defines it) and against projection_fp32 (the exact head the Op mirrors):
// - logits form: every row outside the selection holds its screen score within the FP32 accumulation
//   bound; every selected (row, column) holds projection_fp32's bits; every row clearly within the margin
//   of the column's best screen score is selected, none clearly outside it, at most `rescored`;
// - top form: the exact head's argmax and its token through the row ids; every exact top row (argmax and
//   top_logprobs over every row) the selection admits among the ranks, judged on the FP64 screen scores
//   (misses the policy explains are reported); rank differences equal to the exact BF16 logit differences;
//   the normalizer within 0.25 nats of the exact log-normalizer, structured and flat columns;
// - the same argmax as the full head on a structured fixture (row-scale spread, outlier dimensions,
//   peaked and near-tie columns);
// - a column's outputs do not depend on T, repeat calls and CUDA Graph replay give the same bits;
// - the selection's radix paths: a margin admitting more rows than `rescored`, and fewer than the ranks.
// Informational: each form's time against the exact head's.

#include "ops/op_family.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/screened_head.h"
#include "ops/linear/linear_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using infernix::DType;
using infernix::QType;
using infernix::QuantLayout;
using infernix::Tensor;
using infernix::Weight;
using infernix::WorkspaceArena;
namespace qw = infernix::test::quantized_weight::detail;

constexpr int K      = 2560;
constexpr int Groups = K / 64;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

void cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

// Runs body(begin, end) over [0, n) on every hardware thread.
void parallel_for(std::size_t n, const std::function<void(std::size_t, std::size_t)>& body) {
    const std::size_t workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t step    = (n + workers - 1) / workers;
    std::vector<std::thread> threads;
    for (std::size_t begin = 0; begin < n; begin += step) {
        threads.emplace_back(body, begin, std::min(n, begin + step));
    }
    for (auto& thread : threads) { thread.join(); }
}

std::uint16_t bf16_bits(float f) {
    const std::uint32_t u = qw::float_bits(f);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

float bf16_round(float f) { return qw::bf16_to_f32(bf16_bits(f)); }

// About N(0, 1): Irwin-Hall of four uniforms, deterministic in `state`.
float normal(std::uint64_t& state) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) {
        state = qw::mix64(state);
        s += static_cast<double>(state >> 11) * 0x1.0p-53;
    }
    return static_cast<float>((s - 2.0) * std::sqrt(3.0));
}

// A head's BF16 rows: row r's scale spreads over [0.6, 1.4] x 0.02, and four hidden dimensions carry four
// times the weight (the outlier dimensions of trained heads).
std::vector<std::uint16_t> make_head(std::int32_t rows, std::uint64_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(rows) * K);
    parallel_for(static_cast<std::size_t>(rows), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) {
            std::uint64_t state = qw::mix64(seed ^ (r * 0x9E3779B97F4A7C15ULL));
            state               = qw::mix64(state);
            const float scale   = 0.02F * (0.6F + 0.8F * static_cast<float>((state >> 11) * 0x1.0p-53));
            for (int k = 0; k < K; ++k) {
                const float outlier = (k == 17 || k == 501 || k == 1023 || k == 1999) ? 4.0F : 1.0F;
                bits[r * K + k]     = bf16_bits(scale * outlier * normal(state));
            }
        }
    });
    return bits;
}

// ---- Row-split packings (docs/maintainer/storage-layouts.md section 3): each plane row-major, one
// group's bytes adjacent; scales one binary16 per group.

struct Packed {
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;
};

// The screen: per 64-group the binary16 scale (of +-max|w| x 0.30..1.00) with the least squared error of
// the codes {-2, -1, 0, 1}, then stretched by sqrt(|w|^2 / <w, w_hat>) (the converter's grouped_mse_norm).
Packed pack_q2(const std::vector<float>& values, std::int32_t rows) {
    Packed p{std::vector<std::uint8_t>(static_cast<std::size_t>(rows) * K / 4),
             std::vector<std::uint16_t>(static_cast<std::size_t>(rows) * Groups)};
    parallel_for(static_cast<std::size_t>(rows), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) {
            for (int g = 0; g < Groups; ++g) {
                const float* w = values.data() + r * K + g * 64;
                float amax     = 0.0F;
                for (int i = 0; i < 64; ++i) { amax = std::max(amax, std::fabs(w[i])); }
                std::uint16_t best_word = 0;
                double best_error       = INFINITY;
                int codes[64]           = {};
                for (int sign = -1; sign <= 1; sign += 2) {
                    for (int f = 0; f <= 14; ++f) {
                        const std::uint16_t word = qw::f32_to_f16(static_cast<float>(sign) * amax * (0.30F + 0.05F * f));
                        const float s            = qw::f16_to_f32(word);
                        if (s == 0.0F || !std::isfinite(s)) { continue; }
                        double error = 0.0;
                        for (int i = 0; i < 64; ++i) {
                            const double c = std::clamp(std::nearbyint(static_cast<double>(w[i]) / s), -2.0, 1.0);
                            error += (w[i] - c * s) * (w[i] - c * s);
                        }
                        if (error < best_error) {
                            best_error = error;
                            best_word  = word;
                        }
                    }
                }
                float s = qw::f16_to_f32(best_word);
                double ww = 0.0, wr = 0.0;
                for (int i = 0; i < 64; ++i) {
                    codes[i] = s == 0.0F ? 0 : static_cast<int>(std::clamp(std::nearbyint(static_cast<double>(w[i]) / s), -2.0, 1.0));
                    ww += static_cast<double>(w[i]) * w[i];
                    wr += static_cast<double>(w[i]) * codes[i] * s;
                }
                if (wr > 0.0) { best_word = qw::f32_to_f16(static_cast<float>(s * std::sqrt(ww / wr))); }
                p.scales[r * Groups + g] = best_word;
                for (int i = 0; i < 64; ++i) {
                    std::uint8_t& byte = p.codes[r * (K / 4) + g * 16 + i / 4];
                    byte = static_cast<std::uint8_t>(byte | ((codes[i] & 3) << (2 * (i % 4))));
                }
            }
        }
    });
    return p;
}

// q4_g64_fp16 (max-abs scales) and q8_g32_fp16 (max-abs scales).
Packed pack_grouped(const std::vector<float>& values, std::int32_t rows, int bits) {
    const int group = bits == 4 ? 64 : 32, qmax = bits == 4 ? 7 : 127, groups = K / group;
    Packed p{std::vector<std::uint8_t>(static_cast<std::size_t>(rows) * K * bits / 8),
             std::vector<std::uint16_t>(static_cast<std::size_t>(rows) * groups)};
    parallel_for(static_cast<std::size_t>(rows), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) {
            for (int g = 0; g < groups; ++g) {
                const float* w = values.data() + r * K + g * group;
                float amax     = 0.0F;
                for (int i = 0; i < group; ++i) { amax = std::max(amax, std::fabs(w[i])); }
                const std::uint16_t word = qw::f32_to_f16(amax / static_cast<float>(qmax));
                const float s            = qw::f16_to_f32(word);
                const float reciprocal   = s > 0.0F ? 1.0F / s : 0.0F;
                p.scales[r * groups + g] = word;
                for (int i = 0; i < group; ++i) {
                    const int c = static_cast<int>(std::clamp(std::nearbyint(w[i] * reciprocal), -static_cast<float>(qmax) - (bits == 4 ? 1.0F : 0.0F),
                                                              static_cast<float>(qmax)));
                    const std::size_t at = r * K + g * group + i;
                    if (bits == 4) {
                        p.codes[at / 2] = static_cast<std::uint8_t>(p.codes[at / 2] | ((c & 0xF) << (4 * (at % 2))));
                    } else {
                        p.codes[at] = static_cast<std::uint8_t>(static_cast<std::int8_t>(c));
                    }
                }
            }
        }
    });
    return p;
}

double q2_code(const Packed& p, std::size_t r, int k) {
    const int u = (p.codes[r * (K / 4) + k / 4] >> (2 * (k % 4))) & 3;
    return u >= 2 ? u - 4 : u;
}

// The represented value of an exact head's weight.
double exact_weight(const Packed& p, int bits, std::size_t r, int k) {
    if (bits == 4) {
        const std::size_t at = r * K + k;
        const int u          = (p.codes[at / 2] >> (4 * (at % 2))) & 0xF;
        return (u >= 8 ? u - 16 : u) * static_cast<double>(qw::f16_to_f32(p.scales[r * (K / 64) + k / 64]));
    }
    return static_cast<std::int8_t>(p.codes[r * K + k]) * static_cast<double>(qw::f16_to_f32(p.scales[r * (K / 32) + k / 32]));
}

struct DeviceMatrix {
    void* codes  = nullptr;
    void* scales = nullptr;
    DeviceMatrix(const Packed& p) {
        cuda(cudaMalloc(&codes, p.codes.size()), "cudaMalloc");
        cuda(cudaMalloc(&scales, p.scales.size() * 2), "cudaMalloc");
        cuda(cudaMemcpy(codes, p.codes.data(), p.codes.size(), cudaMemcpyHostToDevice), "copy");
        cuda(cudaMemcpy(scales, p.scales.data(), p.scales.size() * 2, cudaMemcpyHostToDevice), "copy");
    }
    ~DeviceMatrix() {
        cudaFree(codes);
        cudaFree(scales);
    }
    DeviceMatrix(const DeviceMatrix&)            = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;
};

Weight row_split(QType qtype, std::int32_t rows, const DeviceMatrix& m) {
    Weight w;
    w.qtype           = qtype;
    w.layout          = QuantLayout::RowSplit;
    w.ndim            = 2;
    w.shape[0]        = rows;
    w.shape[1]        = K;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = K;
    w.n               = rows;
    w.k               = K;
    w.group_size      = qtype == QType::Q8_G32_FP16 ? 32 : 64;
    w.group           = static_cast<std::int32_t>(w.group_size);
    w.qdata           = m.codes;
    w.scales          = m.scales;
    return w;
}

template <class T> struct DeviceArray {
    T* p = nullptr;
    std::size_t n = 0;
    explicit DeviceArray(std::size_t count) : n(count) { cuda(cudaMalloc(&p, std::max<std::size_t>(1, count) * sizeof(T)), "cudaMalloc"); }
    DeviceArray(const std::vector<T>& v) : DeviceArray(v.size()) {
        cuda(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "copy");
    }
    ~DeviceArray() { cudaFree(p); }
    DeviceArray(const DeviceArray&)            = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    std::vector<T> host() const {
        std::vector<T> out(n);
        cuda(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "copy back");
        return out;
    }
};

// ---- Columns: structured hidden states. Column t boosts five rows (the first by 6..18 logits, the others
// up to 4 below it: near ties) over a background whose logits spread about 1.5, with two outlier hidden
// dimensions (their int8 groups quantize the rest coarsely); flat columns (`flat`) boost the first by 1..4
// only.
std::vector<std::uint16_t> make_columns(const std::vector<float>& head, std::int32_t rows, std::int32_t columns,
                                        std::uint64_t seed, bool flat) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(K) * columns);
    for (std::int32_t t = 0; t < columns; ++t) {
        std::uint64_t state = qw::mix64(seed + 7919ULL * t);
        std::vector<float> x(K);
        for (auto& v : x) { v = 1.5F * normal(state); }
        x[17] += 15.0F;
        x[1999] -= 12.0F;
        state            = qw::mix64(state);
        const float top  = flat ? 1.0F + 3.0F * static_cast<float>((state >> 11) * 0x1.0p-53)
                                : 6.0F + 12.0F * static_cast<float>((state >> 11) * 0x1.0p-53);
        for (int j = 0; j < 5; ++j) {
            state                = qw::mix64(state);
            const std::size_t r  = state % static_cast<std::uint64_t>(rows);
            state                = qw::mix64(state);
            const float boost    = j == 0 ? top : top - 4.0F * static_cast<float>((state >> 11) * 0x1.0p-53);
            double norm          = 0.0;
            for (int k = 0; k < K; ++k) { norm += static_cast<double>(head[r * K + k]) * head[r * K + k]; }
            for (int k = 0; k < K; ++k) { x[k] += static_cast<float>(boost * head[r * K + k] / norm); }
        }
        for (int k = 0; k < K; ++k) { bits[static_cast<std::size_t>(t) * K + k] = bf16_bits(x[k]); }
    }
    return bits;
}

// The column's int8 copy as the contract defines it: per 64-group m = max|x|, q = rint(x * fl32(127 / m))
// clamped to [-127, 127], scale fl32(m / 127).
struct ColumnCopy {
    std::vector<int> q;
    std::vector<double> scale;
};

ColumnCopy column_copy(const std::uint16_t* x) {
    ColumnCopy c{std::vector<int>(K), std::vector<double>(Groups)};
    for (int g = 0; g < Groups; ++g) {
        float amax = 0.0F;
        for (int i = 0; i < 64; ++i) { amax = std::max(amax, std::fabs(qw::bf16_to_f32(x[g * 64 + i]))); }
        const float inverse = amax > 0.0F ? 127.0F / amax : 0.0F;
        for (int i = 0; i < 64; ++i) {
            const float product = qw::bf16_to_f32(x[g * 64 + i]) * inverse;
            c.q[g * 64 + i]     = std::clamp(static_cast<int>(std::nearbyint(product)), -127, 127);
        }
        c.scale[g] = static_cast<double>(amax / 127.0F);
    }
    return c;
}

// The screen score's FP64 value and the magnitude its FP32 bound scales with.
std::pair<double, double> screen_score(const Packed& screen, std::size_t r, const ColumnCopy& c) {
    double sum = 0.0, magnitude = 0.0;
    for (int g = 0; g < Groups; ++g) {
        long dot = 0;
        for (int i = 0; i < 64; ++i) { dot += static_cast<long>(q2_code(screen, r, g * 64 + i)) * c.q[g * 64 + i]; }
        const double term = static_cast<double>(qw::f16_to_f32(screen.scales[r * Groups + g])) * c.scale[g] * static_cast<double>(dot);
        sum += term;
        magnitude += std::fabs(term);
    }
    return {sum, magnitude};
}

double screen_bound(double magnitude) { return magnitude * 64.0 * std::ldexp(1.0, -24) + 1e-30; }

// One column's FP64 screen scores over every row, each with its FP32 accumulation bound.
struct ColumnScreen {
    std::vector<double> score, bound;
};

ColumnScreen column_screen(const Packed& screen, std::int32_t rows, const std::uint16_t* x) {
    const ColumnCopy copy = column_copy(x);
    ColumnScreen c{std::vector<double>(static_cast<std::size_t>(rows)), std::vector<double>(static_cast<std::size_t>(rows))};
    parallel_for(static_cast<std::size_t>(rows), [&](std::size_t begin, std::size_t end) {
        for (std::size_t r = begin; r < end; ++r) {
            const auto [score, magnitude] = screen_score(screen, r, copy);
            c.score[r]                    = score;
            c.bound[r]                    = screen_bound(magnitude);
        }
    });
    return c;
}

// Whether the selection policy (include/infernix/ops/screened_head.h) alone keeps `row` out of a column's
// P_t, judged on the FP64 screen scores with every comparison clear of both rows' FP32 bounds: at least 8
// rows of its CTA (the screen's grid takes 16-row items round-robin) clearly above it, a score clearly more
// than the margin below the column's best, or at least `rescored` rows clearly above it that are clearly
// candidates within the margin. An exact top row missing from the Op's ranks that none of these explains
// is a selection or rescore fault.
bool selection_excludes(const ColumnScreen& c, int row, int grid, const infernix::ops::ScreenedHeadPolicy& policy) {
    constexpr int kListLength = 8, kItemRows = 16;
    const int rows   = static_cast<int>(c.score.size());
    const auto above = [&](int o, int r) { return c.score[o] - c.bound[o] > c.score[r] + c.bound[r]; };
    const auto cta_rows_above = [&](int r, bool clearly) {
        int count = 0;
        for (int item = (r / kItemRows) % grid; item * kItemRows < rows; item += grid) {
            for (int o = item * kItemRows; o < std::min(rows, (item + 1) * kItemRows); ++o) {
                if (o != r && (clearly ? above(o, r) : !above(r, o))) { ++count; }
            }
        }
        return count;
    };
    if (cta_rows_above(row, true) >= kListLength) { return true; }
    double best_low = -INFINITY, best_high = -INFINITY;
    for (int o = 0; o < rows; ++o) {
        best_low  = std::max(best_low, c.score[o] - c.bound[o]);
        best_high = std::max(best_high, c.score[o] + c.bound[o]);
    }
    if (c.score[row] + c.bound[row] < best_low - policy.margin) { return true; }
    int capped = 0;
    for (int o = 0; o < rows && capped < policy.rescored; ++o) {
        if (o != row && above(o, row) && c.score[o] - c.bound[o] >= best_high - policy.margin &&
            cta_rows_above(o, false) < kListLength) {
            ++capped;
        }
    }
    return capped >= policy.rescored;
}

// Rows of column t ordered by (BF16 logit desc, row asc) among [0, valid): the first n.
std::vector<int> top_rows(const std::vector<float>& logits, std::size_t rows, std::int32_t t, int valid, int n) {
    std::vector<int> index(static_cast<std::size_t>(valid));
    std::iota(index.begin(), index.end(), 0);
    const float* column = logits.data() + static_cast<std::size_t>(t) * rows;
    std::partial_sort(index.begin(), index.begin() + n, index.end(), [&](int a, int b) {
        const float va = bf16_round(column[a]), vb = bf16_round(column[b]);
        return va > vb || (va == vb && a < b);
    });
    index.resize(static_cast<std::size_t>(n));
    return index;
}

double log_normalizer(const std::vector<float>& logits, std::size_t rows, std::int32_t t, int valid) {
    const float* column = logits.data() + static_cast<std::size_t>(t) * rows;
    double peak = -INFINITY;
    for (int r = 0; r < valid; ++r) { peak = std::max(peak, static_cast<double>(bf16_round(column[r]))); }
    double sum = 0.0;
    for (int r = 0; r < valid; ++r) { sum += std::exp(static_cast<double>(bf16_round(column[r])) - peak); }
    return peak + std::log(sum);
}

// The exact head's FP32 logits [rows, T] (projection_fp32).
std::vector<float> exact_logits(const Tensor& x, const Weight* quantized, const Tensor* bf16_head, std::int32_t rows) {
    DeviceArray<float> out(static_cast<std::size_t>(rows) * x.ne[1]);
    Tensor o(out.p, DType::FP32, {rows, x.ne[1]});
    if (quantized != nullptr) {
        infernix::ops::projection_fp32(x, *quantized, o, nullptr);
    } else {
        const Tensor* heads[] = {bf16_head};
        infernix::ops::projection_fp32(x, heads, o, nullptr);
    }
    cuda(cudaDeviceSynchronize(), "projection_fp32");
    return out.host();
}

template <class F> double time_us(F&& f, int repeats = 20) {
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    f();
    cudaEventRecord(begin);
    for (int i = 0; i < repeats; ++i) { f(); }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return ms * 1000.0 / repeats;
}

// ---- The logits form

struct LogitsHead {
    std::string label;
    std::int32_t rows = 0, valid = 0;
    std::vector<float> values; // the head's represented values (BF16, or the dequantized q8 rows)
    Packed screen;
    std::unique_ptr<DeviceMatrix> screen_device;
    Weight screen_weight;
    // Exactly one of: BF16 rows, or a q8 matrix.
    std::unique_ptr<DeviceArray<std::uint16_t>> bf16_rows;
    Tensor bf16_tensor;
    Packed q8;
    std::unique_ptr<DeviceMatrix> q8_device;
    Weight q8_weight;
};

std::vector<float> run_logits(const LogitsHead& head, const Tensor& x, const infernix::ops::ScreenedHeadPolicy& policy,
                              WorkspaceArena& work, cudaStream_t stream = nullptr) {
    DeviceArray<float> out(static_cast<std::size_t>(head.rows) * x.ne[1]);
    Tensor logits(out.p, DType::FP32, {head.rows, x.ne[1]});
    infernix::ops::screened_head_logits(x, head.bf16_rows ? head.bf16_tensor : Tensor{}, head.bf16_rows ? Weight{} : head.q8_weight,
                                        head.screen_weight, head.valid, policy, logits, work, stream);
    cuda(cudaStreamSynchronize(stream), "screened_head_logits");
    return out.host();
}

// Checks one call of the logits form on columns `columns` (all columns' argmax; the selection and every
// row's screen bound on `full` columns, sampled rows elsewhere).
void check_logits_call(const LogitsHead& head, const std::vector<std::uint16_t>& xbits, std::int32_t T,
                       const infernix::ops::ScreenedHeadPolicy& policy, WorkspaceArena& work, const std::string& tag,
                       const std::vector<std::int32_t>& full, int& argmax_differs) {
    DeviceArray<std::uint16_t> dx(std::vector<std::uint16_t>(xbits.begin(), xbits.begin() + static_cast<std::ptrdiff_t>(K) * T));
    const Tensor x(dx.p, DType::BF16, {K, T});
    const auto got   = run_logits(head, x, policy, work);
    const auto exact = exact_logits(x, head.bf16_rows ? nullptr : &head.q8_weight, head.bf16_rows ? &head.bf16_tensor : nullptr,
                                    head.rows);
    const std::size_t R = static_cast<std::size_t>(head.rows);
    long outside = 0, missed = 0, extra = 0;
    double worst = 0.0;
    for (std::int32_t t = 0; t < T; ++t) {
        const ColumnCopy copy = column_copy(xbits.data() + static_cast<std::size_t>(t) * K);
        const float* g        = got.data() + t * R;
        const float* e        = exact.data() + t * R;
        // The argmax over the public rows of the BF16 logits, as the Program's acceptance reads them.
        const auto got_top   = top_rows(got, R, t, head.valid, 1);
        const auto exact_top = top_rows(exact, R, t, head.valid, 1);
        if (got_top[0] != exact_top[0]) { ++argmax_differs; }
        // Rescored rows carry the exact bits; count them. A screen score can equal the exact FP32 bits by
        // coincidence (about one row in a million at the screen's error, so a few per 16-column call
        // over 248K rows): an exact-bits row that is also its FP64 screen score within the FP32 bound
        // is not counted.
        long rescored = 0, coincident = 0;
        for (std::size_t r = 0; r < R; ++r) {
            if (std::memcmp(g + r, e + r, 4) != 0) { continue; }
            const auto [reference, magnitude] = screen_score(head.screen, r, copy);
            const bool screen = std::fabs(static_cast<double>(g[r]) - reference) <= screen_bound(magnitude);
            rescored += screen ? 0 : 1;
            coincident += screen ? 1 : 0;
        }
        if (coincident != 0) {
            std::cout << tag << " column " << t << ": " << coincident
                      << " screen score(s) equal to the exact bits by coincidence\n";
        }
        check(rescored <= policy.rescored, tag + ": more rows rescored than the policy allows (" + std::to_string(rescored) +
                                               " in column " + std::to_string(t) + ")");
        const bool whole = std::find(full.begin(), full.end(), t) != full.end();
        if (whole) {
            // Every row's FP64 screen score: the selection the policy defines, up to rows within the FP32
            // error of its boundaries (the margin below the best score, and the cap). The per-CTA candidate
            // lists (the policy's comment) drop a row only below 8 better rows of its own CTA; the fixture's
            // near-best rows are random rows, so that case does not arise here.
            std::vector<std::pair<double, double>> scores(R);
            parallel_for(R, [&](std::size_t begin, std::size_t end) {
                for (std::size_t r = begin; r < end; ++r) { scores[r] = screen_score(head.screen, r, copy); }
            });
            std::vector<int> ranked(static_cast<std::size_t>(head.valid));
            std::iota(ranked.begin(), ranked.end(), 0);
            std::sort(ranked.begin(), ranked.end(), [&](int a, int b) {
                return scores[a].first > scores[b].first || (scores[a].first == scores[b].first && a < b);
            });
            const double best      = scores[ranked[0]].first;
            const double tolerance = 1e-3 * std::max(1.0, std::fabs(best));
            const double floor     = best - policy.margin;
            const auto cap         = static_cast<std::size_t>(policy.rescored);
            const std::size_t within = static_cast<std::size_t>(
                std::count_if(ranked.begin(), ranked.end(), [&](int r) { return scores[r].first >= floor; }));
            // With more rows within the margin than the cap, the cap's best: its boundary score.
            const double cap_score = within > cap ? scores[ranked[cap - 1]].first : -INFINITY;
            const double next_score = within > cap ? scores[ranked[cap]].first : -INFINITY;
            for (std::size_t i = 0; i < ranked.size(); ++i) {
                const int r           = ranked[i];
                const double s        = scores[r].first;
                const bool exact_bits = std::memcmp(g + r, e + r, 4) == 0;
                const bool inside     = s >= floor + tolerance && (within <= cap || s >= next_score + tolerance);
                const bool out        = s < floor - tolerance || (within > cap && s <= cap_score - tolerance);
                if (inside && !exact_bits) { ++missed; }
                if (out && exact_bits) { ++extra; }
                if (!out) { continue; }
                const double error = std::fabs(static_cast<double>(g[r]) - s);
                worst              = std::max(worst, error / screen_bound(scores[r].second));
                outside += !(error <= screen_bound(scores[r].second));
            }
        } else {
            for (std::size_t r = 0; r < R; r += 977) {
                if (std::memcmp(g + r, e + r, 4) == 0) { continue; }
                const auto [reference, magnitude] = screen_score(head.screen, r, copy);
                const double error = std::fabs(static_cast<double>(g[r]) - reference);
                worst = std::max(worst, error / screen_bound(magnitude));
                outside += !(error <= screen_bound(magnitude));
            }
        }
    }
    std::cout << tag << ": screen scores worst " << worst << " of the bound, " << outside << " outside; " << missed
              << " rows within the margin not rescored, " << extra << " outside it rescored\n";
    check(outside == 0, tag + ": screen scores within the FP32 bound");
    check(missed == 0, tag + ": every row within the margin is rescored with the exact bits");
    check(extra == 0, tag + ": no row clearly outside the margin is rescored");
}

void logits_case(LogitsHead& head, std::uint64_t seed) {
    const std::size_t bytes = infernix::ops::screened_head_workspace_bytes(head.rows, infernix::ops::kScreenedHeadMaxColumns);
    WorkspaceArena work(bytes);
    const infernix::ops::ScreenedHeadPolicy policy{};
    const auto columns = make_columns(head.values, head.rows, 16, seed, false);
    const auto flat    = make_columns(head.values, head.rows, 16, seed + 1, true);
    int differs = 0, flat_differs = 0;
    for (const std::int32_t T : {1, 5, 8, 9, 16}) {
        check_logits_call(head, columns, T, policy, work, head.label + " T=" + std::to_string(T),
                          T >= 8 ? std::vector<std::int32_t>{0, T - 1} : std::vector<std::int32_t>{0}, differs);
    }
    check_logits_call(head, flat, 16, policy, work, head.label + " flat T=16", {0}, flat_differs);
    std::cout << head.label << ": argmax differs from the full head in " << differs << " of 39 structured columns, "
              << flat_differs << " of 16 flat ones\n";
    // A miss needs the exact best row's screen score more than the margin below another's, or below the
    // 64 best: on this fixture at most one column in 39 (the realistic rate comes from the real-artifact
    // check, infernix_qwen4_exp_forward_real_test --screened-heads).
    check(differs <= 1, head.label + ": the screened argmax equals the full head's on the structured columns");

    // Column invariance, repeat determinism and graph replay, on the 16-column call.
    DeviceArray<std::uint16_t> dx(columns);
    const Tensor x16(dx.p, DType::BF16, {K, 16});
    const Tensor x1(dx.p, DType::BF16, {K, 1});
    const auto wide   = run_logits(head, x16, policy, work);
    const auto single = run_logits(head, x1, policy, work);
    check(std::memcmp(wide.data(), single.data(), sizeof(float) * head.rows) == 0,
          head.label + ": a column's logits do not depend on the call's width");
    check(run_logits(head, x16, policy, work) == wide, head.label + ": a repeated call gives the same bits");
    {
        cudaStream_t stream = nullptr;
        cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        DeviceArray<float> out(static_cast<std::size_t>(head.rows) * 16);
        Tensor logits(out.p, DType::FP32, {head.rows, 16});
        cudaGraph_t graph       = nullptr;
        cudaGraphExec_t replay  = nullptr;
        cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
        infernix::ops::screened_head_logits(x16, head.bf16_rows ? head.bf16_tensor : Tensor{},
                                            head.bf16_rows ? Weight{} : head.q8_weight, head.screen_weight, head.valid, policy,
                                            logits, work, stream);
        cuda(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda(cudaGraphInstantiate(&replay, graph, 0), "instantiate");
        for (int i = 0; i < 2; ++i) {
            cuda(cudaMemsetAsync(out.p, 0, out.n * sizeof(float), stream), "clear");
            cuda(cudaGraphLaunch(replay, stream), "replay");
            cuda(cudaStreamSynchronize(stream), "replay sync");
            check(out.host() == wide, head.label + ": CUDA Graph replay gives the eager bits");
        }
        cudaGraphExecDestroy(replay);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    }
    // A margin admitting every row: the `rescored` best screen scores (the radix path), still exact.
    {
        const infernix::ops::ScreenedHeadPolicy wide_margin{.margin = 1000.0F, .rescored = 64};
        int unused = 0;
        check_logits_call(head, columns, 4, wide_margin, work, head.label + " margin 1000 T=4", {0, 3}, unused);
        const auto got   = run_logits(head, Tensor(dx.p, DType::BF16, {K, 1}), wide_margin, work);
        const auto exact = exact_logits(Tensor(dx.p, DType::BF16, {K, 1}), head.bf16_rows ? nullptr : &head.q8_weight,
                                        head.bf16_rows ? &head.bf16_tensor : nullptr, head.rows);
        long rescored = 0;
        for (std::int32_t r = 0; r < head.rows; ++r) { rescored += std::memcmp(&got[r], &exact[r], 4) == 0; }
        check(rescored == 64, head.label + ": a wide margin rescores exactly `rescored` rows");
    }
    // Informational: the 1- and 8-column calls against the exact head.
    for (const std::int32_t T : {1, 8}) {
        const Tensor x(dx.p, DType::BF16, {K, T});
        DeviceArray<float> out(static_cast<std::size_t>(head.rows) * T);
        Tensor logits(out.p, DType::FP32, {head.rows, T});
        const double screened = time_us([&] {
            infernix::ops::screened_head_logits(x, head.bf16_rows ? head.bf16_tensor : Tensor{},
                                                head.bf16_rows ? Weight{} : head.q8_weight, head.screen_weight, head.valid,
                                                policy, logits, work, nullptr);
        });
        const double exact = time_us([&] {
            if (head.bf16_rows) {
                const Tensor* heads[] = {&head.bf16_tensor};
                infernix::ops::projection_fp32(x, heads, logits, nullptr);
            } else {
                infernix::ops::projection_fp32(x, head.q8_weight, logits, nullptr);
            }
        });
        std::cout << head.label << " T=" << T << ": screened " << screened << " us, exact head " << exact << " us\n";
    }
}

// ---- The top form (the proposal head)

void top_case(std::uint64_t seed) {
    constexpr std::int32_t R = 131072;
    const auto bits   = make_head(R, seed);
    std::vector<float> values(bits.size());
    parallel_for(bits.size(), [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) { values[i] = qw::bf16_to_f32(bits[i]); }
    });
    const Packed q4     = pack_grouped(values, R, 4);
    // The converter fits the proposal head and its screen from the same BF16 rows; the columns' boosts
    // follow the proposal head's represented rows.
    std::vector<float> dequantized(values.size());
    parallel_for(static_cast<std::size_t>(R), [&](std::size_t b, std::size_t e) {
        for (std::size_t r = b; r < e; ++r) {
            for (int k = 0; k < K; ++k) { dequantized[r * K + k] = static_cast<float>(exact_weight(q4, 4, r, k)); }
        }
    });
    const Packed screen = pack_q2(values, R);
    const DeviceMatrix q4_device(q4), screen_device(screen);
    const Weight head = row_split(QType::Q4_G64_FP16, R, q4_device);
    const Weight copy = row_split(QType::Q2_G64_FP16, R, screen_device);
    std::vector<std::int32_t> ids(R);
    for (std::int32_t r = 0; r < R; ++r) { ids[r] = (r * 7919 + 13) % 248077; } // distinct (7919 is prime)
    DeviceArray<std::int32_t> row_ids(ids);
    const Tensor row_ids_tensor(row_ids.p, DType::I32, {R});
    WorkspaceArena work(infernix::ops::screened_head_workspace_bytes(R, infernix::ops::kScreenedHeadMaxColumns));
    const infernix::ops::ScreenedHeadPolicy policy{};

    const auto run = [&](const Tensor& x, std::int32_t ranks, bool mapped, const infernix::ops::ScreenedHeadPolicy& p,
                         std::vector<std::int32_t>& rows_out, std::vector<float>& logprobs_out, std::vector<std::int32_t>& tokens_out) {
        const std::int32_t T = x.ne[1];
        DeviceArray<std::int32_t> rows(static_cast<std::size_t>(ranks) * T), tokens(static_cast<std::size_t>(T));
        DeviceArray<float> logprobs(static_cast<std::size_t>(ranks) * T);
        Tensor r(rows.p, DType::I32, {ranks, T}), l(logprobs.p, DType::FP32, {ranks, T});
        Tensor tk = mapped ? Tensor(tokens.p, DType::I32, {T}) : Tensor{};
        infernix::ops::screened_head_top(x, head, copy, R, p, r, l, mapped ? row_ids_tensor : Tensor{}, tk, work, nullptr);
        cuda(cudaDeviceSynchronize(), "screened_head_top");
        rows_out     = rows.host();
        logprobs_out = logprobs.host();
        tokens_out   = mapped ? tokens.host() : std::vector<std::int32_t>{};
    };

    // The screen's grid (one CTA per SM, at most 256) and each fixture column's FP64 screen scores, once.
    int sms = 0;
    cuda(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "SM count");
    const int grid = std::clamp(sms, 1, 256);
    std::vector<std::unique_ptr<ColumnScreen>> oracles(32);
    const auto oracle = [&](const std::vector<std::uint16_t>& xbits, int slot, std::int32_t t) -> const ColumnScreen& {
        auto& c = oracles[static_cast<std::size_t>(slot + t)];
        if (!c) { c = std::make_unique<ColumnScreen>(column_screen(screen, R, xbits.data() + static_cast<std::size_t>(t) * K)); }
        return *c;
    };

    // One call pair (four ranks, and one rank with tokens) over the first T columns. A rank that is not the
    // exact head's is a miss; the contract allows only misses the selection policy explains (the exact row
    // outside P_t), so `unexplained` must stay 0, and the counts of the others are reported.
    struct TopStats {
        int columns = 0, differs = 0, rank_differs = 0, unexplained = 0;
        double worst_difference = 0.0, worst_normalizer = 0.0;
    };
    const auto check_columns = [&](const std::vector<std::uint16_t>& xbits, int slot, std::int32_t T,
                                   const infernix::ops::ScreenedHeadPolicy& p) {
        TopStats stats;
        DeviceArray<std::uint16_t> dx(std::vector<std::uint16_t>(xbits.begin(), xbits.begin() + static_cast<std::ptrdiff_t>(K) * T));
        const Tensor x(dx.p, DType::BF16, {K, T});
        const auto exact = exact_logits(x, &head, nullptr, R);
        std::vector<std::int32_t> rows4, rows1, unused, tokens;
        std::vector<float> logprobs4, logprobs1;
        run(x, 4, false, p, rows4, logprobs4, unused);
        run(x, 1, true, p, rows1, logprobs1, tokens);
        for (std::int32_t t = 0; t < T; ++t) {
            const auto top = top_rows(exact, R, t, R, 4);
            ++stats.columns;
            stats.differs += rows1[t] != top[0] || tokens[t] != ids[top[0]];
            for (int j = 0; j < 4; ++j) { stats.rank_differs += rows4[t * 4 + j] != top[j]; }
            // Every exact top row the Op left out (of its four ranks, or its one) must be one P_t excludes.
            const auto returned = [&](int row, int ranks, const std::vector<std::int32_t>& out) {
                return std::find(out.begin() + t * ranks, out.begin() + (t + 1) * ranks, row) != out.begin() + (t + 1) * ranks;
            };
            for (int j = 0; j < 4; ++j) {
                const bool missing = !returned(top[j], 4, rows4) || (j == 0 && rows1[t] != top[0]);
                if (missing && !selection_excludes(oracle(xbits, slot, t), top[j], grid, p)) { ++stats.unexplained; }
            }
            // Rows the screen returned carry exact logits: their rank differences are the exact BF16
            // differences, whichever rows they are; the normalizer is an approximation.
            const auto at = [&](int row) { return bf16_round(exact[static_cast<std::size_t>(t) * R + row]); };
            const double exact_normalizer = log_normalizer(exact, R, t, R);
            stats.worst_normalizer = std::max(stats.worst_normalizer, std::fabs((at(rows1[t]) - logprobs1[t]) - exact_normalizer));
            for (int j = 1; j < 4; ++j) {
                const double d = std::fabs(static_cast<double>(logprobs4[t * 4 + j] - logprobs4[t * 4]) -
                                           (static_cast<double>(at(rows4[t * 4 + j])) - at(rows4[t * 4])));
                stats.worst_difference = std::max(stats.worst_difference, d / std::max(1.0, std::fabs(exact_normalizer)));
            }
        }
        return stats;
    };
    const auto columns = make_columns(dequantized, R, 16, seed + 5, false);
    const auto flat    = make_columns(dequantized, R, 16, seed + 6, true);
    TopStats total;
    for (const std::int32_t T : {1, 3, 8, 16}) {
        const TopStats s = check_columns(columns, 0, T, policy);
        total.columns += s.columns;
        total.differs += s.differs;
        total.rank_differs += s.rank_differs;
        total.unexplained += s.unexplained;
        total.worst_difference = std::max(total.worst_difference, s.worst_difference);
        total.worst_normalizer = std::max(total.worst_normalizer, s.worst_normalizer);
    }
    // Ranks 2-4 of these columns are often background rows within a nat of dozens of others, which a 2-bit
    // screen (residual about 0.8 nats) cannot order: their misses are reported, and only an exact row the
    // selection admits must come back.
    std::cout << "top form: the draft differs from the exact argmax in " << total.differs << " of " << total.columns
              << " structured columns, " << total.rank_differs << " of " << 4 * total.columns
              << " ranks differ (" << total.unexplained << " not explained by the selection); rank differences worst "
              << total.worst_difference << " (relative); normalizer worst " << total.worst_normalizer << " nats\n";
    check(total.differs <= 1, "top form: the draft equals the exact head's argmax (and its token) on the structured columns");
    check(total.unexplained == 0, "top form: every exact top row the selection admits is among the ranks");
    check(total.worst_difference <= 4.0 * std::ldexp(1.0, -23), "top form: rank differences are the exact BF16 differences");
    check(total.worst_normalizer <= 0.25, "top form: the normalizer approximates the exact log-normalizer");
    const TopStats flat_stats = check_columns(flat, 16, 16, policy);
    std::cout << "top form, flat columns: " << flat_stats.differs << " of 16 drafts and " << flat_stats.rank_differs
              << " of 64 ranks differ (" << flat_stats.unexplained << " not explained), normalizer worst "
              << flat_stats.worst_normalizer << " nats\n";
    check(flat_stats.worst_difference <= 4.0 * std::ldexp(1.0, -23), "top form: flat columns' rank differences are exact");
    check(flat_stats.unexplained == 0, "top form: flat columns' admitted exact top rows are among the ranks");
    check(flat_stats.worst_normalizer <= 0.25, "top form: flat columns' normalizer approximates the exact log-normalizer");
    // The radix paths: margin 0 with four ranks (one row within the margin, four required), and a margin
    // admitting every row (more than `rescored`: the 64 best screen scores). Margin 0 rescores only the
    // screen's four best, so the exact top row is required only where the screen ranks it there.
    const TopStats narrow = check_columns(columns, 0, 8, {.margin = 0.0F, .rescored = 64});
    const TopStats capped = check_columns(columns, 0, 8, {.margin = 1000.0F, .rescored = 64});
    std::cout << "top form, margin 0: " << narrow.differs << " of 8 drafts differ; margin 1000: " << capped.differs
              << " drafts and " << capped.rank_differs << " ranks differ (" << narrow.unexplained << " and "
              << capped.unexplained << " not explained)\n";
    check(narrow.worst_difference <= 4.0 * std::ldexp(1.0, -23), "top form: margin 0 returns exact rank differences");
    check(narrow.unexplained == 0 && capped.unexplained == 0,
          "top form: the radix selections keep every admitted exact top row among the ranks");
    // Column invariance, determinism and graph replay of the 8-column top form.
    {
        DeviceArray<std::uint16_t> dx(std::vector<std::uint16_t>(columns.begin(), columns.begin() + static_cast<std::ptrdiff_t>(K) * 8));
        std::vector<std::int32_t> rows8, rows1, tokens8, tokens1;
        std::vector<float> lp8, lp1;
        run(Tensor(dx.p, DType::BF16, {K, 8}), 4, true, policy, rows8, lp8, tokens8);
        run(Tensor(dx.p, DType::BF16, {K, 1}), 4, true, policy, rows1, lp1, tokens1);
        check(std::equal(rows1.begin(), rows1.end(), rows8.begin()) &&
                  std::memcmp(lp1.data(), lp8.data(), 4 * sizeof(float)) == 0 && tokens1[0] == tokens8[0],
              "top form: a column's ranks do not depend on the call's width");
        std::vector<std::int32_t> again_rows, again_tokens;
        std::vector<float> again_lp;
        run(Tensor(dx.p, DType::BF16, {K, 8}), 4, true, policy, again_rows, again_lp, again_tokens);
        check(again_rows == rows8 && again_lp == lp8 && again_tokens == tokens8, "top form: a repeated call gives the same bits");
        cudaStream_t stream = nullptr;
        cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        DeviceArray<std::int32_t> rows(32), tokens(8);
        DeviceArray<float> logprobs(32);
        Tensor r(rows.p, DType::I32, {4, 8}), l(logprobs.p, DType::FP32, {4, 8}), tk(tokens.p, DType::I32, {8});
        cudaGraph_t graph      = nullptr;
        cudaGraphExec_t replay = nullptr;
        cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
        infernix::ops::screened_head_top(Tensor(dx.p, DType::BF16, {K, 8}), head, copy, R, policy, r, l, row_ids_tensor, tk,
                                         work, stream);
        cuda(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda(cudaGraphInstantiate(&replay, graph, 0), "instantiate");
        for (int i = 0; i < 2; ++i) {
            cuda(cudaGraphLaunch(replay, stream), "replay");
            cuda(cudaStreamSynchronize(stream), "replay sync");
            check(rows.host() == rows8 && logprobs.host() == lp8 && tokens.host() == tokens8,
                  "top form: CUDA Graph replay gives the eager bits");
        }
        cudaGraphExecDestroy(replay);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        // Informational: one draft step's head (T = 1) against the exact proposal head's projection.
        DeviceArray<float> out(static_cast<std::size_t>(R));
        Tensor logits(out.p, DType::FP32, {R, 1});
        Tensor r1(rows.p, DType::I32, {1, 1}), l1(logprobs.p, DType::FP32, {1, 1}), t1(tokens.p, DType::I32, {1});
        const Tensor x1(dx.p, DType::BF16, {K, 1});
        const double screened = time_us([&] { infernix::ops::screened_head_top(x1, head, copy, R, policy, r1, l1, row_ids_tensor, t1, work, nullptr); });
        const double exact    = time_us([&] { infernix::ops::projection_fp32(x1, head, logits, nullptr); });
        std::cout << "top form T=1: screened " << screened << " us, exact head's projection alone " << exact << " us\n";
    }
}

} // namespace

INFERNIX_OP_TEST(screened_head) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        top_case(101);
        {
            // The FP4-Local / FP8 LM head: BF16 rows.
            LogitsHead head;
            head.label = "BF16 head";
            head.rows  = 248320;
            head.valid = 248077;
            const auto bits = make_head(head.rows, 202);
            head.values.resize(bits.size());
            parallel_for(bits.size(), [&](std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e; ++i) { head.values[i] = qw::bf16_to_f32(bits[i]); }
            });
            head.screen        = pack_q2(head.values, head.rows);
            head.screen_device = std::make_unique<DeviceMatrix>(head.screen);
            head.screen_weight = row_split(QType::Q2_G64_FP16, head.rows, *head.screen_device);
            head.bf16_rows     = std::make_unique<DeviceArray<std::uint16_t>>(bits);
            head.bf16_tensor   = Tensor(head.bf16_rows->p, DType::BF16, {K, head.rows});
            logits_case(head, 303);
        }
        {
            // Recipe B / C's 8-bit LM head; the screen fits its represented values.
            LogitsHead head;
            head.label = "Q8 head";
            head.rows  = 248320;
            head.valid = 248077;
            const auto bits = make_head(head.rows, 404);
            std::vector<float> values(bits.size());
            parallel_for(bits.size(), [&](std::size_t b, std::size_t e) {
                for (std::size_t i = b; i < e; ++i) { values[i] = qw::bf16_to_f32(bits[i]); }
            });
            head.q8 = pack_grouped(values, head.rows, 8);
            head.values.resize(values.size());
            parallel_for(static_cast<std::size_t>(head.rows), [&](std::size_t b, std::size_t e) {
                for (std::size_t r = b; r < e; ++r) {
                    for (int k = 0; k < K; ++k) { head.values[r * K + k] = static_cast<float>(exact_weight(head.q8, 8, r, k)); }
                }
            });
            head.screen        = pack_q2(head.values, head.rows);
            head.screen_device = std::make_unique<DeviceMatrix>(head.screen);
            head.screen_weight = row_split(QType::Q2_G64_FP16, head.rows, *head.screen_device);
            head.q8_device     = std::make_unique<DeviceMatrix>(head.q8);
            head.q8_weight     = row_split(QType::Q8_G32_FP16, head.rows, *head.q8_device);
            logits_case(head, 505);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "OK screened_head\n";
    return 0;
}
