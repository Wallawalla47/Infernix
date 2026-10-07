// projection_fp32 (BF16 and q8_g32_fp16 weights) against the FP64 product of the represented
// weights and activations, within the documented FP32 accumulation bound
// sum_k |w x| * K * 2^-24, plus its column invariance: a column's bits do not depend on the batch
// width or the column's position.
#include "infernix/ops/projection_fp32.h"
#include "ops/linear/linear_test_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace infernix::test::linear;
using infernix::DType;
using infernix::Tensor;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

template <class T> T* device(const std::vector<T>& v) {
    T* p = nullptr;
    if (cudaMalloc(&p, v.size() * sizeof(T)) != cudaSuccess ||
        cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("device copy failed");
    }
    return p;
}

std::uint16_t bf16_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}

float bf16_value(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// x: BF16 bits [t][k] and their values.
void random_activations(std::int32_t k, std::int32_t t, std::uint32_t seed, std::vector<std::uint16_t>& bits,
                        std::vector<float>& values) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 1.0F);
    bits.resize(static_cast<std::size_t>(k) * t);
    values.resize(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        bits[i]   = bf16_bits(n(rng));
        values[i] = bf16_value(bits[i]);
    }
}

// Checks out [t][n] against the FP64 product on `rows` (all rows when empty).
// w(r, i): the represented weight value of row r, column i.
using WeightAt = std::function<double(std::int32_t, std::int32_t)>;

void verify(const std::string& tag, const std::vector<float>& out, const WeightAt& w, const std::vector<float>& x, std::int32_t n, std::int32_t k, std::int32_t t,
            const std::vector<std::int32_t>& rows) {
    long bad = 0;
    double worst = 0;
    const auto each = [&](std::int32_t r) {
        for (std::int32_t c = 0; c < t; ++c) {
            double ref = 0, mag = 0;
            for (std::int32_t i = 0; i < k; ++i) {
                const double p = w(r, i) * static_cast<double>(x[static_cast<std::size_t>(c) * k + i]);
                ref += p;
                mag += std::fabs(p);
            }
            const double got   = out[static_cast<std::size_t>(c) * n + r];
            const double bound = mag * k * std::ldexp(1.0, -24) + 1e-30;
            worst              = std::max(worst, std::fabs(got - ref) / bound);
            if (!(std::fabs(got - ref) <= bound)) { ++bad; }
        }
    };
    if (rows.empty()) {
        for (std::int32_t r = 0; r < n; ++r) { each(r); }
    } else {
        for (const auto r : rows) { each(r); }
    }
    std::cout << tag << ": worst error " << worst << " of the bound, " << bad << " outside\n";
    check(bad == 0, tag + " within the FP32 accumulation bound");
}

std::vector<float> run_q8(const infernix::Weight& weight, const std::vector<std::uint16_t>& xbits, std::int32_t k,
                          std::int32_t t) {
    auto* dx = device(xbits);
    float* dout = nullptr;
    cudaMalloc(&dout, sizeof(float) * weight.n * t);
    Tensor x(dx, DType::BF16, {k, t}), out(dout, DType::FP32, {weight.n, t});
    infernix::ops::projection_fp32(x, weight, out, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("projection q8 failed"); }
    std::vector<float> host(static_cast<std::size_t>(weight.n) * t);
    cudaMemcpy(host.data(), dout, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dout);
    return host;
}

void quantized_case(infernix::QType qtype, std::int32_t n, std::int32_t k, std::uint32_t seed, bool sampled) {
    const bool q4     = qtype == infernix::QType::Q4_G64_FP16;
    const auto packed = q4 ? make_q4_g64_fp16_weight(n, k, seed) : make_q8_g32_fp16_weight(n, k, seed);
    void* payload     = nullptr;
    cudaMalloc(&payload, packed.payload.size());
    cudaMemcpy(payload, packed.payload.data(), packed.payload.size(), cudaMemcpyHostToDevice);
    const infernix::Weight weight = packed.device_weight(payload);
    // The patterned fixture holds no dequantized matrix: decode the payload with its stored scales.
    const std::int32_t padded_k = static_cast<std::int32_t>(weight.padded_shape[1]);
    const int group_size = q4 ? 64 : 32;
    const WeightAt w = [&](std::int32_t r, std::int32_t i) {
        const std::size_t group =
            static_cast<std::size_t>(r) * (padded_k / group_size) + static_cast<std::size_t>(i / group_size);
        int code = 0;
        if (q4) {
            const std::uint8_t byte = packed.payload[static_cast<std::size_t>(r) * (padded_k / 2) + i / 2];
            const int u             = (i & 1) ? (byte >> 4) : (byte & 0xF);
            code                    = u >= 8 ? u - 16 : u;
        } else {
            code = static_cast<std::int8_t>(packed.payload[static_cast<std::size_t>(r) * padded_k + i]);
        }
        const float scale =
            infernix::test::quantized_weight::detail::f16_to_f32(infernix::test::quantized_weight::detail::load_u16_le(
                packed.payload, static_cast<std::size_t>(packed.scale_plane_offset) + group * 2));
        return static_cast<double>(code) * static_cast<double>(scale);
    };
    std::vector<std::int32_t> rows;
    if (sampled) {
        for (std::int32_t r = 0; r < n; r += 977) { rows.push_back(r); }
        rows.push_back(n - 1);
    }
    std::vector<std::uint16_t> wide_bits;
    std::vector<float> wide_values;
    random_activations(k, 17, seed + 1, wide_bits, wide_values);
    for (const std::int32_t t : {1, 3, 8, 9, 17}) {
        std::vector<std::uint16_t> bits(wide_bits.begin(), wide_bits.begin() + static_cast<std::ptrdiff_t>(k) * t);
        std::vector<float> values(wide_values.begin(), wide_values.begin() + static_cast<std::ptrdiff_t>(k) * t);
        const auto out = run_q8(weight, bits, k, t);
        verify(std::string(q4 ? "q4" : "q8") + " N=" + std::to_string(n) + " K=" + std::to_string(k) + " T=" + std::to_string(t), out, w, values,
               n, k, t, rows);
        if (t > 1) {
            // Column 0 equals the single-column call bit for bit.
            const auto single = run_q8(weight, std::vector<std::uint16_t>(bits.begin(), bits.begin() + k), k, 1);
            check(std::memcmp(single.data(), out.data(), sizeof(float) * n) == 0,
                  std::string(q4 ? "q4" : "q8") + " column invariance N=" + std::to_string(n) + " T=" + std::to_string(t));
        }
    }
    cudaFree(payload);
}

// Segments of `rows` rows each, concatenated in order (one to four weights).
void bf16_case(const std::vector<std::int32_t>& rows, std::int32_t k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0F, 0.05F);
    std::vector<float> wv;
    std::vector<std::uint16_t*> devices;
    std::vector<Tensor> tensors;
    std::int32_t n = 0;
    std::string label;
    for (const std::int32_t segment : rows) {
        std::vector<std::uint16_t> part(static_cast<std::size_t>(segment) * k);
        for (auto& v : part) { v = bf16_bits(d(rng)); }
        for (auto v : part) { wv.push_back(bf16_value(v)); }
        devices.push_back(device(part));
        tensors.push_back(Tensor(devices.back(), DType::BF16, {k, segment}));
        n += segment;
        label += (label.empty() ? "" : "+") + std::to_string(segment);
    }
    std::vector<const Tensor*> weights;
    for (const auto& t : tensors) { weights.push_back(&t); }
    const WeightAt w = [&](std::int32_t r, std::int32_t i) {
        return static_cast<double>(wv[static_cast<std::size_t>(r) * k + i]);
    };
    std::vector<std::uint16_t> wide_bits;
    std::vector<float> wide_values;
    random_activations(k, 17, seed + 7, wide_bits, wide_values);
    std::vector<float> single;
    // Decode widths (every T up to 8) take the one-row-per-warp mapping, wider calls the four-row
    // one; a column's bits must not depend on either.
    for (const std::int32_t t : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17}) {
        auto* dx    = device(std::vector<std::uint16_t>(wide_bits.begin(), wide_bits.begin() + static_cast<std::ptrdiff_t>(k) * t));
        float* dout = nullptr;
        cudaMalloc(&dout, sizeof(float) * n * t);
        Tensor x(dx, DType::BF16, {k, t}), out(dout, DType::FP32, {n, t});
        infernix::ops::projection_fp32(x, weights, out, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("projection bf16 failed"); }
        std::vector<float> host(static_cast<std::size_t>(n) * t);
        cudaMemcpy(host.data(), dout, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        const std::vector<float> values(wide_values.begin(), wide_values.begin() + static_cast<std::ptrdiff_t>(k) * t);
        verify("bf16 N=" + label + " K=" + std::to_string(k) + " T=" + std::to_string(t),
               host, w, values, n, k, t, {});
        if (t == 1) {
            single = host;
        } else {
            check(std::memcmp(single.data(), host.data(), sizeof(float) * n) == 0,
                  "bf16 column invariance N=" + std::to_string(n) + " T=" + std::to_string(t));
        }
        cudaFree(dx);
        cudaFree(dout);
    }
    for (auto* p : devices) { cudaFree(p); }
}

// Prefill widths take the wide mapping (64 columns and more): every column's bits must equal the
// narrow mapping's for the same column (calls of at most 63 columns), and sampled columns meet the
// FP64 bound. Widths straddle the 128-column CTA range and the 16-column pass.
void bf16_wide_case(const std::vector<std::int32_t>& rows, std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0F, 0.05F);
    std::vector<float> wv;
    std::vector<std::uint16_t*> devices;
    std::vector<Tensor> tensors;
    std::int32_t n = 0;
    for (const std::int32_t segment : rows) {
        std::vector<std::uint16_t> part(static_cast<std::size_t>(segment) * k);
        for (auto& v : part) { v = bf16_bits(d(rng)); }
        for (auto v : part) { wv.push_back(bf16_value(v)); }
        devices.push_back(device(part));
        tensors.push_back(Tensor(devices.back(), DType::BF16, {k, segment}));
        n += segment;
    }
    std::vector<const Tensor*> weights;
    for (const auto& w : tensors) { weights.push_back(&w); }
    std::vector<std::uint16_t> bits;
    std::vector<float> values;
    random_activations(k, t, seed + 3, bits, values);
    auto* dx    = device(bits);
    float* dout = nullptr;
    cudaMalloc(&dout, sizeof(float) * n * t);
    const auto run = [&](std::int32_t first, std::int32_t width) {
        Tensor x(dx + static_cast<std::size_t>(first) * k, DType::BF16, {k, width});
        Tensor out(dout + static_cast<std::size_t>(first) * n, DType::FP32, {n, width});
        infernix::ops::projection_fp32(x, weights, out, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("projection bf16 wide failed"); }
        std::vector<float> host(static_cast<std::size_t>(n) * width);
        cudaMemcpy(host.data(), out.data, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        return host;
    };
    const std::vector<float> wide = run(0, t);
    if (t >= 4096) { // informational: the prefill router's kernel time (not a check)
        Tensor x(dx, DType::BF16, {k, t}), out(dout, DType::FP32, {n, t});
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin);
        for (int i = 0; i < 20; ++i) { infernix::ops::projection_fp32(x, weights, out, nullptr); }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float ms = 0;
        cudaEventElapsedTime(&ms, begin, end);
        std::cout << "wide N=" << n << " K=" << k << " T=" << t << ": " << ms / 20 * 1000 << " us per call\n";
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
    long differing = 0;
    for (std::int32_t first = 0; first < t; first += 63) {
        const std::int32_t width   = std::min(63, t - first);
        const std::vector<float> narrow = run(first, width);
        for (std::int32_t c = 0; c < width; ++c) {
            if (std::memcmp(narrow.data() + static_cast<std::size_t>(c) * n,
                            wide.data() + static_cast<std::size_t>(first + c) * n, sizeof(float) * n) != 0) {
                ++differing;
            }
        }
    }
    const std::string label = "bf16 wide N=" + std::to_string(n) + " K=" + std::to_string(k) + " T=" + std::to_string(t);
    std::cout << label << ": " << differing << " columns differ from the narrow mapping\n";
    check(differing == 0, label + " bitwise equal to the narrow mapping");
    // FP64 bound on the first and last three columns.
    const WeightAt w = [&](std::int32_t r, std::int32_t i) { return static_cast<double>(wv[static_cast<std::size_t>(r) * k + i]); };
    verify(label + " (first columns)", wide, w, values, n, k, 3, {});
    const std::vector<float> tail(wide.end() - static_cast<std::ptrdiff_t>(n) * 3, wide.end());
    const std::vector<float> tail_x(values.end() - static_cast<std::ptrdiff_t>(k) * 3, values.end());
    verify(label + " (last columns)", tail, w, tail_x, n, k, 3, {});
    cudaFree(dx);
    cudaFree(dout);
    for (auto* p : devices) { cudaFree(p); }
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        bf16_case({512, 1}, 2560, 11); // router rows and the shared-expert gate, as Qwen4Exp concatenates them
        bf16_case({1000, 37}, 2560, 13);
        bf16_case({129}, 2560, 14);           // one segment
        bf16_case({7, 300, 1, 64}, 1024, 15); // four segments, one of a single row
        bf16_wide_case({512, 1}, 2560, 64, 31);    // the narrowest wide call
        bf16_wide_case({512, 1}, 2560, 4096, 37);  // a 4096-token prefill chunk's router
        bf16_wide_case({7, 300, 1, 64}, 3072, 301, 41); // K at its limit, partial CTA range and pass
        bf16_wide_case({129}, 1024, 135, 43);
        quantized_case(infernix::QType::Q8_G32_FP16, 1000, 2560, 17, false);
        quantized_case(infernix::QType::Q8_G32_FP16, 248320, 2560, 19, true); // the 8-bit lm_head, sampled rows
        quantized_case(infernix::QType::Q4_G64_FP16, 1000, 2560, 23, false);
        quantized_case(infernix::QType::Q4_G64_FP16, 131072, 2560, 29, true); // the proposal head, sampled rows
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "OK projection_fp32\n";
    return 0;
}
