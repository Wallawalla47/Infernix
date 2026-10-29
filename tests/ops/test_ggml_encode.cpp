// The producers that write a GGUF projection's input encoded (include/infernix/ops/ggml_encode.h) against their
// definition: gated_rmsnorm or sigmoid_mul, then ggml_linear on the BF16 result. ggml_linear_encoded over the
// producer's columns must give the same output bits at every narrow width, with and without the input permutation
// (blocks within and across the cluster's parts), at two, three and four parts a column, over row-major formats with
// and without slice-sum terms and a two-format (mixed) projection, and for columns holding a non-finite value.

#include "ops/op_family.h"
#include "core/device.h"
#include "core/ggml_format.h"
#include "infernix/ops/gated_rmsnorm.h"
#include "infernix/ops/ggml_encode.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/sigmoid_mul.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

std::vector<std::uint8_t> random_blocks(QType q, int rows, int k, std::mt19937& rng) {
    const int blocks = k / static_cast<int>(ggml_block_elems(q)), size = static_cast<int>(ggml_block_bytes(q));
    std::vector<std::uint8_t> data(static_cast<std::size_t>(rows) * blocks * size);
    for (auto& v : data) { v = static_cast<std::uint8_t>(rng()); }
    const auto scale = [&](std::uint8_t* p) { // binary16 scales of exponent fields 5..20
        const std::uint16_t w = static_cast<std::uint16_t>((rng() & 0x83FFU) | ((5 + rng() % 16) << 10));
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

std::vector<std::uint16_t> normal_bf16(std::size_t n, float sigma, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0F, sigma);
    std::vector<float> v(n);
    for (auto& x : v) { x = normal(rng); }
    std::vector<std::uint16_t> bits(n);
    for (std::size_t i = 0; i < n; ++i) { bits[i] = f32_to_bf16(v[i]); }
    return bits;
}

// A projection of `rows` rows of K = k: one segment of `q0`, or (q1 != q0) the rows split between q0 and q1.
struct Projection {
    std::vector<DeviceBuffer> weights;
    std::vector<ops::GgmlSegment> segments;
};

Projection make_projection(QType q0, QType q1, int rows, int k, std::mt19937& rng) {
    Projection p;
    const int first = q1 == q0 ? rows : rows / 2;
    const QType formats[] = {q0, q1};
    const int counts[]    = {first, rows - first};
    for (int i = 0; i < 2 && counts[i] > 0; ++i) {
        const auto blocks = random_blocks(formats[i], counts[i], k, rng);
        p.weights.emplace_back(blocks.size());
        p.weights.back().copy_from_host(blocks.data(), blocks.size(), 0);
        p.segments.push_back({formats[i], static_cast<const std::uint8_t*>(p.weights.back().p), counts[i], i == 0 ? 0 : first});
    }
    return p;
}

int compare(const std::string& label, DeviceBuffer& expected, DeviceBuffer& got, std::size_t n) {
    const auto a = from_device<std::uint16_t>(expected, n), b = from_device<std::uint16_t>(got, n);
    std::size_t differ = 0;
    for (std::size_t i = 0; i < n; ++i) { differ += a[i] != b[i]; }
    if (differ == 0) { return 0; }
    std::cerr << label << ": " << differ << " of " << n << " outputs differ from the composed Ops\n";
    return 1;
}

struct Case {
    QType q0, q1;
    int heads; // rows of 128 a column (gated_rmsnorm), or K / 128 (sigmoid_mul)
    int columns;
    int perm_block; // 0: no permutation
    bool non_finite;
};

int run_case(const Case& c, bool rmsnorm, ops::RmsGate gate, std::mt19937& rng) {
    const int d = 128, k = d * c.heads, t = c.columns, rows = 72;
    const std::string label = std::string(rmsnorm ? "gated_rmsnorm" : "sigmoid_mul") + " q" + std::to_string(static_cast<int>(c.q0)) +
                              "/" + std::to_string(static_cast<int>(c.q1)) + " K=" + std::to_string(k) + " T=" +
                              std::to_string(t) + " perm_block=" + std::to_string(c.perm_block) +
                              (c.non_finite ? " non-finite" : "") + (gate == ops::RmsGate::Sigmoid ? " sigmoid" : "");
    std::vector<std::uint16_t> x = normal_bf16(static_cast<std::size_t>(k) * t, 2.0F, rng);
    const std::vector<std::uint16_t> z = normal_bf16(x.size(), 1.5F, rng), weight = normal_bf16(d, 0.5F, rng);
    if (c.non_finite) { x[static_cast<std::size_t>(k) + 37] = 0x7F80U; } // +Inf in column 1
    std::vector<std::int32_t> perm(c.perm_block > 0 ? static_cast<std::size_t>(k / c.perm_block) : 1, 0);
    for (std::size_t i = 0; i < perm.size(); ++i) { perm[i] = static_cast<std::int32_t>(i); }
    std::shuffle(perm.begin(), perm.end(), rng);
    const Projection p = make_projection(c.q0, c.q1, rows, k, rng);

    DeviceBuffer dx = to_device(x), dz = to_device(z), dw = to_device(weight), dperm = to_device(perm);
    DeviceBuffer staged(x.size() * 2), expected(static_cast<std::size_t>(rows) * t * 2), got(expected.bytes);
    DeviceBuffer words(static_cast<std::size_t>(ops::ggml_encoded_column_words(k)) * t * 4), emax(static_cast<std::size_t>(t) * 4);
    expected.fill(0xff);
    got.fill(0x7f);
    const std::int32_t* input_perm = c.perm_block > 0 ? static_cast<const std::int32_t*>(dperm.p) : nullptr;
    WorkspaceArena work(ops::ggml_linear_workspace_bytes(k, t) + (1 << 20));
    int multiprocessors = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0));
    const DeviceExecutionView execution{nullptr, multiprocessors};

    Tensor tstaged(staged.p, DType::BF16, {k, t});
    Tensor texpected(expected.p, DType::BF16, {rows, t}), tgot(got.p, DType::BF16, {rows, t});
    ops::GgmlEncodedColumns encoded{Tensor(words.p, DType::I32, {ops::ggml_encoded_column_words(k), t}),
                                    Tensor(emax.p, DType::I32, {t}), k};
    if (rmsnorm) {
        const Tensor tx(dx.p, DType::BF16, {d, c.heads * t}), tz(dz.p, DType::BF16, {d, c.heads * t});
        const Tensor tw(dw.p, DType::BF16, {d});
        Tensor normed = tstaged.view({d, c.heads * t});
        ops::gated_rmsnorm(tx, tw, tz, gate, 1.0e-6F, normed, execution);
        ops::gated_rmsnorm_ggml_encoded(tx, tw, tz, gate, 1.0e-6F, input_perm, c.perm_block, encoded, nullptr);
    } else {
        CUDA_CHECK(cudaMemcpy(staged.p, dx.p, dx.bytes, cudaMemcpyDeviceToDevice));
        const Tensor tgate(dz.p, DType::BF16, {k, t}), tx(dx.p, DType::BF16, {k, t});
        ops::sigmoid_mul(tgate, tstaged, nullptr);
        ops::sigmoid_mul_ggml_encoded(tgate, tx, input_perm, c.perm_block, encoded, nullptr);
    }
    ops::ggml_linear(tstaged, p.segments, texpected, input_perm, c.perm_block, work, nullptr, ops::LinearRoutes::Exact);
    Tensor* const outs[] = {&tgot};
    ops::ggml_linear_encoded(encoded, p.segments, outs, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    int failures = compare(label, expected, got, static_cast<std::size_t>(rows) * t);
    if (from_device<std::uint16_t>(dx, x.size()) != x) {
        std::cerr << label << ": the producer wrote its input\n";
        ++failures;
    }
    return failures;
}

// The gated-norm producer and the projection captured in a graph behind a kernel that rewrites the producer's input
// (sigmoid_mul in place), so the producer runs as a programmatic dependent; against the same three Ops eagerly with
// gated_rmsnorm and ggml_linear.
int run_graph_case(std::mt19937& rng) {
    const int d = 128, heads = 48, k = d * heads, t = 4, rows = 72;
    const std::vector<std::uint16_t> x0 = normal_bf16(static_cast<std::size_t>(k) * t, 2.0F, rng);
    const std::vector<std::uint16_t> scale = normal_bf16(x0.size(), 1.0F, rng), z = normal_bf16(x0.size(), 1.5F, rng);
    const std::vector<std::uint16_t> weight = normal_bf16(d, 0.5F, rng);
    const Projection p = make_projection(QType::GGML_IQ3_S, QType::GGML_IQ3_S, rows, k, rng);
    DeviceBuffer dx = to_device(x0), dref = to_device(x0), dscale = to_device(scale), dz = to_device(z), dw = to_device(weight);
    DeviceBuffer normed(x0.size() * 2), expected(static_cast<std::size_t>(rows) * t * 2), got(expected.bytes);
    DeviceBuffer words(static_cast<std::size_t>(ops::ggml_encoded_column_words(k)) * t * 4), emax(static_cast<std::size_t>(t) * 4);
    WorkspaceArena work(ops::ggml_linear_workspace_bytes(k, t) + (1 << 20));
    int multiprocessors = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const Tensor tscale(dscale.p, DType::BF16, {d, heads * t}), tz(dz.p, DType::BF16, {d, heads * t}), tw(dw.p, DType::BF16, {d});
    Tensor tx(dx.p, DType::BF16, {d, heads * t}), tref(dref.p, DType::BF16, {d, heads * t});
    Tensor tnormed(normed.p, DType::BF16, {d, heads * t});
    Tensor texpected(expected.p, DType::BF16, {rows, t}), tgot(got.p, DType::BF16, {rows, t});
    ops::GgmlEncodedColumns encoded{Tensor(words.p, DType::I32, {ops::ggml_encoded_column_words(k), t}),
                                    Tensor(emax.p, DType::I32, {t}), k};
    ops::sigmoid_mul(tscale, tref, stream);
    ops::gated_rmsnorm(tref, tw, tz, ops::RmsGate::Silu, 1.0e-6F, tnormed, DeviceExecutionView{stream, multiprocessors});
    ops::ggml_linear(tnormed.view({k, t}), p.segments, texpected, nullptr, 0, work, stream, ops::LinearRoutes::Exact);
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Tensor* const outs[] = {&tgot};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ops::sigmoid_mul(tscale, tx, stream);
    ops::gated_rmsnorm_ggml_encoded(tx, tw, tz, ops::RmsGate::Silu, 1.0e-6F, nullptr, 0, encoded, stream);
    ops::ggml_linear_encoded(encoded, p.segments, outs, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaMemsetAsync(got.p, 0xff, got.bytes, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return compare("gated_rmsnorm captured behind its input's producer", expected, got, static_cast<std::size_t>(rows) * t);
}

} // namespace

INFERNIX_OP_TEST(ggml_encode) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    std::mt19937 rng(20261029);
    int failures = 0;
    const QType iq3 = QType::GGML_IQ3_S, q6 = QType::GGML_Q6_K, q4 = QType::GGML_Q4_K, iq4 = QType::GGML_IQ4_XS;
    for (const bool rmsnorm : {true, false}) {
        for (const int t : {1, 2, 3, 5, 8, 9, 16, 31}) {
            failures += run_case({iq3, iq3, 48, t, 0, false}, rmsnorm, ops::RmsGate::Silu, rng);
        }
        failures += run_case({q6, q6, 48, 4, 128, false}, rmsnorm, ops::RmsGate::Silu, rng);   // the GDN head order
        failures += run_case({q4, q4, 32, 6, 8, false}, rmsnorm, ops::RmsGate::Sigmoid, rng);  // two parts, fine blocks
        failures += run_case({iq4, q6, 48, 7, 0, false}, rmsnorm, ops::RmsGate::Silu, rng);   // one mixed launch
        failures += run_case({q4, iq3, 64, 3, 256, false}, rmsnorm, ops::RmsGate::Sigmoid, rng); // four parts
        failures += run_case({iq3, iq3, 48, 5, 0, true}, rmsnorm, ops::RmsGate::Silu, rng);
    }
    failures += run_graph_case(rng);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " ggml_encode\n";
    return failures == 0 ? 0 : 1;
}
