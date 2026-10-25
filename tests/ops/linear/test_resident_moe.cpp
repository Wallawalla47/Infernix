// resident_moe_experts (the MTP drafter's device-resident experts) against the FP64 formula over
// the exactly decoded Q2_G64_FP16, Q4_G64_FP16 and Q8_G32_FP16 banks, at the Qwen4Exp drafter's shapes
// (H = 2560, I = 640, top-10) and the Qwen3.5-122B-A10B drafter's (H = 3072, I = 1024, top-8). Criterion: the BF16 output rounding plus the FP32 accumulation of
// both projections (h is FP32), sum |D h| * (I + H) * 2^-23 for the accumulation part.
#include "infernix/ops/resident_moe.h"
#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace infernix::test::linear;
namespace qw = infernix::test::quantized_weight;
using infernix::DType;
using infernix::QType;
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

// The decoded value of row r, column i of a patterned row-split bank.
struct Decoder {
    const qw::PackedWeight& packed;
    QType qtype;
    double operator()(std::int64_t r, std::int32_t i) const {
        const std::int32_t padded = packed.weight.padded_shape[1];
        const int group           = qtype == QType::Q8_G32_FP16 ? 32 : 64;
        const std::size_t g       = static_cast<std::size_t>(r) * (padded / group) + static_cast<std::size_t>(i / group);
        const float scale         = qw::detail::f16_to_f32(
            qw::detail::load_u16_le(packed.payload, static_cast<std::size_t>(packed.scale_plane_offset) + g * 2));
        int code = 0;
        if (qtype == QType::Q8_G32_FP16) {
            code = static_cast<std::int8_t>(packed.payload[static_cast<std::size_t>(r) * padded + i]);
        } else if (qtype == QType::Q2_G64_FP16) {
            const std::uint8_t byte = packed.payload[static_cast<std::size_t>(r) * (padded / 4) + i / 4];
            const int u             = (byte >> (2 * (i % 4))) & 3;
            code                    = u >= 2 ? u - 4 : u;
        } else {
            const std::uint8_t byte = packed.payload[static_cast<std::size_t>(r) * (padded / 2) + i / 2];
            const int u             = (i & 1) ? (byte >> 4) : (byte & 0xF);
            code                    = u >= 8 ? u - 16 : u;
        }
        return static_cast<double>(code) * static_cast<double>(scale);
    }
};

// A drafter's expert geometry: Qwen3.8-Flash-Next's (H 2560, I 640, top-10) or Qwen3.5-122B-A10B's
// (H 3072, I 1024, top-8).
struct Shape {
    std::int32_t hidden, intermediate, top_k;
};
constexpr Shape kFlashNext{2560, 640, 10};
constexpr Shape kQwen3_5_122B{3072, 1024, 8};

void run_case(QType qtype, std::int32_t experts, std::int32_t columns, std::uint32_t seed,
              Shape shape = kFlashNext) {
    const std::int32_t H = shape.hidden, I = shape.intermediate, K = shape.top_k;
    const auto gate_up = qw::make_patterned_weight(qtype, experts * 2 * I, H, seed,
                                                   {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
    const auto down    = qw::make_patterned_weight(qtype, experts * H, I, seed + 1,
                                                   {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
    void* gu_payload = nullptr;
    void* dn_payload = nullptr;
    cudaMalloc(&gu_payload, gate_up.payload.size());
    cudaMalloc(&dn_payload, down.payload.size());
    cudaMemcpy(gu_payload, gate_up.payload.data(), gate_up.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dn_payload, down.payload.data(), down.payload.size(), cudaMemcpyHostToDevice);
    const infernix::Weight gu_weight = gate_up.device_weight(gu_payload);
    const infernix::Weight dn_weight = down.device_weight(dn_payload);

    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) { v = bf16_bits(n(rng)); }
    // Distinct experts per column, a duplicate across columns included.
    std::vector<std::int32_t> ids(static_cast<std::size_t>(K) * columns);
    for (std::int32_t t = 0; t < columns; ++t) {
        std::vector<std::int32_t> order(static_cast<std::size_t>(experts));
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (std::int32_t j = 0; j < K; ++j) { ids[static_cast<std::size_t>(t) * K + j] = order[static_cast<std::size_t>(j)]; }
    }
    auto* dx   = device(x);
    auto* dids = device(ids);
    void* dout = nullptr;
    cudaMalloc(&dout, sizeof(std::uint16_t) * H * K * columns);
    const std::size_t ws_bytes = infernix::ops::resident_moe_workspace_bytes(K * columns, I);
    void* ws = nullptr;
    cudaMalloc(&ws, ws_bytes);
    Tensor tx(dx, DType::BF16, {H, columns}), tids(dids, DType::I32, {K, columns});
    Tensor tout(dout, DType::BF16, {H, K * columns});
    infernix::ops::resident_moe_experts(tx, tids, gu_weight, dn_weight, experts, I, ws, ws_bytes, tout, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("resident_moe_experts failed"); }
    std::vector<std::uint16_t> out(static_cast<std::size_t>(H) * K * columns);
    cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost);

    const Decoder gu{gate_up, qtype}, dn{down, qtype};
    long bad = 0;
    double worst = 0;
    for (std::int32_t t = 0; t < columns; ++t) {
        for (std::int32_t j = 0; j < K; ++j) {
            const std::int64_t e = ids[static_cast<std::size_t>(t) * K + j];
            std::vector<double> h(I);
            for (std::int32_t r = 0; r < I; ++r) {
                double g = 0, u = 0;
                for (std::int32_t i = 0; i < H; ++i) {
                    const double xv = bf16_value(x[static_cast<std::size_t>(t) * H + i]);
                    g += gu(e * 2 * I + r, i) * xv;
                    u += gu(e * 2 * I + I + r, i) * xv;
                }
                h[static_cast<std::size_t>(r)] = g / (1.0 + std::exp(-g)) * u;
            }
            // Sampled output rows keep the FP64 oracle affordable.
            for (std::int32_t r = 0; r < H; r += 37) {
                double ref = 0, mag = 0;
                for (std::int32_t i = 0; i < I; ++i) {
                    const double p = dn(e * H + r, i) * h[static_cast<std::size_t>(i)];
                    ref += p;
                    mag += std::fabs(p);
                }
                const double got   = bf16_value(out[(static_cast<std::size_t>(t) * K + j) * H + r]);
                const double bound = std::fabs(ref) * std::ldexp(1.0, -8) + mag * (I + H) * std::ldexp(1.0, -23) + 1e-30;
                worst              = std::max(worst, std::fabs(got - ref) / bound);
                if (!(std::fabs(got - ref) <= bound)) { ++bad; }
            }
        }
    }
    const std::string tag = std::string(qtype == QType::Q8_G32_FP16 ? "q8" : qtype == QType::Q2_G64_FP16 ? "q2" : "q4") +
                            " H=" + std::to_string(H) + " I=" + std::to_string(I) + " E=" + std::to_string(experts) +
                            " T=" + std::to_string(columns);
    std::cout << tag << ": worst error " << worst << " of the bound, " << bad << " outside\n";
    check(bad == 0, tag + " within the bound");
    cudaFree(gu_payload);
    cudaFree(dn_payload);
    cudaFree(dx);
    cudaFree(dids);
    cudaFree(dout);
    cudaFree(ws);
}

// The padding mask (valid_columns): rows of `width` columns, each live for its first valid[row];
// every live entry's output bits equal the unmasked call's, every padding entry's output is zero.
void masked_case(QType qtype, std::int32_t experts, std::int32_t width, const std::vector<std::int32_t>& valid,
                 std::uint32_t seed, Shape shape) {
    const std::int32_t H = shape.hidden, I = shape.intermediate, K = shape.top_k;
    const auto columns   = width * static_cast<std::int32_t>(valid.size());
    const auto gate_up = qw::make_patterned_weight(qtype, experts * 2 * I, H, seed,
                                                   {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
    const auto down    = qw::make_patterned_weight(qtype, experts * H, I, seed + 1,
                                                   {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
    void* gu_payload = nullptr;
    void* dn_payload = nullptr;
    cudaMalloc(&gu_payload, gate_up.payload.size());
    cudaMalloc(&dn_payload, down.payload.size());
    cudaMemcpy(gu_payload, gate_up.payload.data(), gate_up.payload.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dn_payload, down.payload.data(), down.payload.size(), cudaMemcpyHostToDevice);
    const infernix::Weight gu_weight = gate_up.device_weight(gu_payload);
    const infernix::Weight dn_weight = down.device_weight(dn_payload);
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(H) * columns);
    for (auto& v : x) { v = bf16_bits(n(rng)); }
    std::uniform_int_distribution<std::int32_t> pick(0, experts - 1);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(K) * columns);
    for (auto& id : ids) { id = pick(rng); }
    auto* dx    = device(x);
    auto* dids  = device(ids);
    auto* dvalid = device(valid);
    const std::size_t out_bytes = sizeof(std::uint16_t) * H * K * columns;
    void *plain = nullptr, *masked = nullptr, *ws = nullptr;
    cudaMalloc(&plain, out_bytes);
    cudaMalloc(&masked, out_bytes);
    cudaMemset(masked, 0xFF, out_bytes); // padding outputs must be written
    const std::size_t ws_bytes = infernix::ops::resident_moe_workspace_bytes(K * columns, I);
    cudaMalloc(&ws, ws_bytes);
    Tensor tx(dx, DType::BF16, {H, columns}), tids(dids, DType::I32, {K, columns});
    Tensor tplain(plain, DType::BF16, {H, K * columns}), tmasked(masked, DType::BF16, {H, K * columns});
    infernix::ops::resident_moe_experts(tx, tids, gu_weight, dn_weight, experts, I, ws, ws_bytes, tplain, nullptr);
    infernix::ops::resident_moe_experts(tx, tids, gu_weight, dn_weight, experts, I, ws, ws_bytes, tmasked, nullptr,
                                        dvalid, width);
    if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("masked resident_moe_experts failed"); }
    std::vector<std::uint16_t> a(out_bytes / 2), b(out_bytes / 2);
    cudaMemcpy(a.data(), plain, out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), masked, out_bytes, cudaMemcpyDeviceToHost);
    long live_diff = 0, padding_nonzero = 0;
    for (std::int32_t t = 0; t < columns; ++t) {
        const bool live = t % width < valid[static_cast<std::size_t>(t / width)];
        for (std::size_t i = static_cast<std::size_t>(t) * K * H; i < static_cast<std::size_t>(t + 1) * K * H; ++i) {
            if (live) {
                live_diff += a[i] != b[i] ? 1 : 0;
            } else {
                padding_nonzero += b[i] != 0 ? 1 : 0;
            }
        }
    }
    check(live_diff == 0, "masked call: live entries bit-identical to the unmasked call");
    check(padding_nonzero == 0, "masked call: padding entries are BF16 zero");
    // The masked call repeats bit for bit (op_tester.h verify_repeatable).
    check(infernix::test::verify_repeatable(
              "resident_moe masked", 3, nullptr,
              [&] {
                  infernix::ops::resident_moe_experts(tx, tids, gu_weight, dn_weight, experts, I, ws, ws_bytes, tmasked,
                                                      nullptr, dvalid, width);
              },
              {{masked, out_bytes}}) == 0,
          "masked call: repeated calls give the same bits");
    for (void* p : {gu_payload, dn_payload, static_cast<void*>(dx), static_cast<void*>(dids), static_cast<void*>(dvalid),
                    plain, masked, ws}) {
        cudaFree(p);
    }
}

} // namespace

INFERNIX_OP_TEST(resident_moe) {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        run_case(QType::Q4_G64_FP16, 12, 1, 5);
        run_case(QType::Q4_G64_FP16, 12, 3, 7);
        run_case(QType::Q8_G32_FP16, 11, 2, 9);
        run_case(QType::Q2_G64_FP16, 12, 1, 11);
        run_case(QType::Q2_G64_FP16, 10, 3, 13);
        // The Qwen3.5-122B-A10B drafter: Q4 banks (grouped_mse), decode and an MTP verify width.
        run_case(QType::Q4_G64_FP16, 10, 1, 15, kQwen3_5_122B);
        run_case(QType::Q4_G64_FP16, 12, 4, 17, kQwen3_5_122B);
        run_case(QType::Q8_G32_FP16, 9, 2, 19, kQwen3_5_122B);
        // The 122B drafter's alignment: rows of three columns with 1, 3 and 2 live (OR-09's mask).
        masked_case(QType::Q4_G64_FP16, 12, 3, {1, 3, 2}, 21, kQwen3_5_122B);
        masked_case(QType::Q8_G32_FP16, 9, 2, {1, 0}, 23, kQwen3_5_122B);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "OK resident_moe_experts\n";
    return 0;
}
