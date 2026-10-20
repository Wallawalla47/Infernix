#include "infernix/ops/precise_linear.h"
#include "ops/precise/precise_decode.cuh"
#include "ops/precise/precise_dot.cuh"
#include "core/device.h"
#include "core/ggml_format.h"
#include "core/layout.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::ops {
namespace {

using precise::bf16;

// One thread owns one output row (R weight rows: one, or a SwiGLU channel's gate and up rows, or a GDN
// head's a and b rows) and the CTA's up to kColumns columns (precise_dot.cuh: exact binary32 products
// summed in double-binary32 arithmetic in ascending k); an epilogue turns the R sums of a (row, column)
// into the output, rounding once.
constexpr int kThreads = 64;
constexpr int kColumns = precise::kDotColumns;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("precise_linear: ") + message); }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

// The columns of x (BF16 [K, T]); element j of a column read at perm[j / pb] * pb + j % pb, and column c
// of the launch read at x column column_map[c] when there is a map.
struct Input {
    const bf16* x;
    const std::int32_t* perm;
    int perm_block;
    int k;
    int columns;
    const std::int32_t* column_map;
};

// The weight rows a launch reads: output row r's q-th weight row is row_begin + q * row_stride + r.
struct Rows {
    int count;
    int begin;
    int stride;
};

// out[row, column] at column * ld + offset + row, BF16 or FP32.
struct Output {
    void* data;
    int ld;
    int offset;
    bool bf16_out;
    __device__ void store(int row, int column, double v) const {
        const std::size_t i = static_cast<std::size_t>(column) * ld + offset + row;
        if (bf16_out) {
            static_cast<bf16*>(data)[i] = __double2bfloat16(v);
        } else {
            static_cast<float*>(data)[i] = __double2float_rn(v);
        }
    }
};

// The epilogues (R sums of a row and a column, as binary64 values).
struct Store {
    static constexpr int kRows = 1;
    Output out;
    __device__ void operator()(int row, int column, const double (&v)[1]) const { out.store(row, column, v[0]); }
};

// SiLU(gate) * up, evaluated in binary64.
struct Swiglu {
    static constexpr int kRows = 2;
    Output out;
    __device__ void operator()(int row, int column, const double (&v)[2]) const {
        out.store(row, column, v[0] / (1.0 + exp(-v[0])) * v[1]);
    }
};

// residual = round(residual + sum): linear_add's single rounding of the BF16 residual plus the sum.
struct AddResidual {
    static constexpr int kRows = 1;
    bf16* residual;
    int ld;
    __device__ void operator()(int row, int column, const double (&v)[1]) const {
        bf16& r = residual[static_cast<std::size_t>(column) * ld + row];
        r       = __double2bfloat16(v[0] + static_cast<double>(__bfloat162float(r)));
    }
};

// Gated DeltaNet gates from a head's a and b sums: g = -exp(A_log) * softplus(a + dt_bias) and
// beta = sigmoid(b), in binary64, rounded once to FP32.
struct GdnGates {
    static constexpr int kRows = 2;
    float* g;
    float* beta;
    const float* a_log;
    const float* dt_bias;
    int heads;
    __device__ void operator()(int head, int column, const double (&v)[2]) const {
        const double a        = v[0] + static_cast<double>(dt_bias[head]);
        const double softplus = a > 40.0 ? a : log1p(exp(a));
        const std::size_t i   = static_cast<std::size_t>(column) * heads + head;
        g[i]                  = __double2float_rn(-exp(static_cast<double>(a_log[head])) * softplus);
        beta[i]               = __double2float_rn(1.0 / (1.0 + exp(-v[1])));
    }
};

// acc[row, token] += weight * sum (FP32): an expert's down projection of its tokens into the sparse MoE's
// accumulator (launch column c is token tokens[c], with route weight weights[c]; no map: column c).
struct MoeAccumulate {
    static constexpr int kRows = 1;
    float* acc;
    int ld;
    const std::int32_t* tokens;
    const float* weights;
    __device__ void operator()(int row, int column, const double (&v)[1]) const {
        const int token = tokens != nullptr ? tokens[column] : column;
        float& a        = acc[static_cast<std::size_t>(token) * ld + row];
        a               = __fadd_rn(a, __fmul_rn(weights[column], __double2float_rn(v[0])));
    }
};

template <class Decoder, class Epilogue>
__global__ void __launch_bounds__(kThreads)
    precise_kernel(Input in, Decoder decoder, Rows rows, Epilogue epilogue) {
    constexpr int R = Epilogue::kRows;
    __shared__ precise::DotShared shared;
    const int row     = static_cast<int>(blockIdx.x) * kThreads + static_cast<int>(threadIdx.x);
    const int c0      = static_cast<int>(blockIdx.y) * kColumns;
    const int count   = min(kColumns, in.columns - c0);
    const bool active = row < rows.count;
    const auto column = [&](int c, int chunk, float (&v)[8]) {
        const int j    = 8 * chunk;
        const int src  = in.perm != nullptr ? in.perm[j / in.perm_block] * in.perm_block + j % in.perm_block : j;
        const int col  = in.column_map != nullptr ? in.column_map[c0 + c] : c0 + c;
        const uint4 packed = *reinterpret_cast<const uint4*>(in.x + static_cast<std::size_t>(col) * in.k + src);
        const auto* bits   = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
        for (int e = 0; e < 8; ++e) { v[e] = precise::bf16_float(bits[e]); }
    };
    const auto weights = [&](int q, int chunk, precise::Weights8& w) {
        decoder.decode(rows.begin + q * rows.stride + row, chunk, w);
    };
    double acc[R][kColumns];
    precise::dot<R, Decoder>(in.k, count, column, weights, active, shared, acc);
    if (!active) { return; }
#pragma unroll
    for (int c = 0; c < kColumns; ++c) {
        if (c < count) {
            double v[R];
#pragma unroll
            for (int q = 0; q < R; ++q) { v[q] = acc[q][c]; }
            epilogue(row, c0 + c, v);
        }
    }
}

template <class Epilogue, class Decoder>
void launch(const Input& in, const Decoder& decoder, Rows rows, const Epilogue& epilogue, cudaStream_t stream) {
    const dim3 grid((rows.count + kThreads - 1) / kThreads, (in.columns + kColumns - 1) / kColumns);
    precise_kernel<Decoder, Epilogue><<<grid, kThreads, 0, stream>>>(in, decoder, rows, epilogue);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("precise_linear: ") + cudaGetErrorString(error)); }
}

Input input_of(const Tensor& x) {
    require(x.data != nullptr && x.dtype == DType::BF16 && x.is_contiguous() && x.ne[2] == 1 && x.ne[3] == 1 &&
                aligned16(x.data),
            "x must be contiguous, 16-byte aligned BF16 [K, T]");
    require(x.ne[0] > 0 && x.ne[0] % 8 == 0 && x.ne[1] > 0, "K must be a positive multiple of 8 and T positive");
    return {static_cast<const bf16*>(x.data), nullptr, 1, x.ne[0], x.ne[1], nullptr};
}

Output output_of(Tensor& out, int rows, int columns, int offset = 0) {
    require(out.data != nullptr && (out.dtype == DType::BF16 || out.dtype == DType::FP32) && out.is_contiguous() &&
                out.ne[1] == columns && out.ne[2] == 1 && out.ne[3] == 1 && offset >= 0 && offset + rows <= out.ne[0],
            "out must be contiguous BF16 or FP32 [N, T] covering the rows");
    return {out.data, out.ne[0], offset, out.dtype == DType::BF16};
}

// The decoder of a dense Weight [N, K] (see the header for the formats), run with an epilogue over
// `rows` of its rows.
template <class Epilogue>
void dispatch(const Input& in, const Weight& weight, Rows rows, const Epilogue& epilogue, cudaStream_t stream) {
    const int k = in.k;
    require(weight.k == k && weight.n > 0 && weight.qdata != nullptr && aligned16(weight.qdata), "weight must be [N, K]");
    require(rows.count > 0 && rows.begin >= 0 && rows.stride >= 0 &&
                rows.begin + (Epilogue::kRows - 1) * rows.stride + rows.count <= weight.n,
            "the rows must lie within the weight");
    switch (weight.qtype) {
    case QType::BF16: {
        require(weight.layout == QuantLayout::Contiguous, "BF16 weights must be contiguous");
        precise::Bf16Rows d{};
        d.data[0]  = static_cast<const bf16*>(weight.qdata);
        d.begin[1] = weight.n;
        d.count    = 1;
        d.k        = k;
        launch(in, d, rows, epilogue, stream);
        return;
    }
    case QType::FP8_E4M3FN_ROW_BF16:
        require(weight.layout == QuantLayout::RowScale && weight.scales != nullptr, "FP8 weights must be row-scaled");
        launch(in,
               precise::Fp8Row{static_cast<const std::uint8_t*>(weight.qdata),
                               static_cast<const std::uint16_t*>(weight.scales), k},
               rows, epilogue, stream);
        return;
    case QType::NVFP4:
        require(weight.layout == QuantLayout::BlockScaleK16M128x4 && weight.scales != nullptr && weight.n % 128 == 0 &&
                    k % 64 == 0 && weight.weight_scale_divisor > 0.0F,
                "NVFP4 weights must be block-scaled with a positive divisor");
        launch(in,
               precise::Nvfp4{static_cast<const std::uint8_t*>(weight.qdata),
                              static_cast<const std::uint8_t*>(weight.scales),
                              static_cast<double>(weight.weight_scale_divisor), k},
               rows, epilogue, stream);
        return;
    default:
        break;
    }
    require(weight.layout == QuantLayout::RowSplit && weight.scales != nullptr && weight.padded_shape[1] >= k &&
                weight.padded_shape[1] % 128 == 0,
            "unsupported weight format");
    const auto* codes  = static_cast<const std::uint8_t*>(weight.qdata);
    const auto* high   = static_cast<const std::uint8_t*>(weight.qhigh);
    const auto* scales = static_cast<const __half*>(weight.scales);
    const int padded_k = weight.padded_shape[1];
    switch (weight.qtype) {
    case QType::Q2_G64_FP16:
        launch(in, precise::RowSplit<QType::Q2_G64_FP16>{codes, high, scales, padded_k}, rows, epilogue, stream);
        return;
    case QType::Q4_G64_FP16:
        launch(in, precise::RowSplit<QType::Q4_G64_FP16>{codes, high, scales, padded_k}, rows, epilogue, stream);
        return;
    case QType::Q5_G64_FP16:
        require(high != nullptr, "q5 weights need their high plane");
        launch(in, precise::RowSplit<QType::Q5_G64_FP16>{codes, high, scales, padded_k}, rows, epilogue, stream);
        return;
    case QType::Q6_G64_FP16:
        require(high != nullptr, "q6 weights need their high plane");
        launch(in, precise::RowSplit<QType::Q6_G64_FP16>{codes, high, scales, padded_k}, rows, epilogue, stream);
        return;
    case QType::Q8_G32_FP16:
        launch(in, precise::RowSplit<QType::Q8_G32_FP16>{codes, high, scales, padded_k}, rows, epilogue, stream);
        return;
    default:
        require(false, "unsupported weight format");
    }
}

template <QType F>
void launch_ggml(const Input& in, const GgmlSegment& s, const Output& out, cudaStream_t stream) {
    launch(in, precise::Ggml<F>{s.data, ggml_row_bytes(F, in.k), s.layout == GgmlLayout::Tiles16}, Rows{s.rows, 0, 0},
           Store{out}, stream);
}

// The BF16 rows of up to eight contiguous BF16 weights [N_i, K] stacked in order.
precise::Bf16Rows stacked_bf16(std::span<const Weight* const> weights, int k, int& rows) {
    require(!weights.empty() && weights.size() <= precise::Bf16Rows::kMaxSegments, "between one and eight weights");
    precise::Bf16Rows d{};
    d.count = static_cast<int>(weights.size());
    d.k     = k;
    rows    = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const Weight& w = *weights[i];
        require(w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.k == k && w.n > 0 &&
                    w.qdata != nullptr && aligned16(w.qdata),
                "stacked weights must be contiguous BF16 [N, K]");
        d.data[i]  = static_cast<const bf16*>(w.qdata);
        d.begin[i] = rows;
        rows += w.n;
    }
    d.begin[d.count] = rows;
    return d;
}

__global__ void moe_route_kernel(const float* scores, int experts, int top_k, int tokens, std::int32_t* ids,
                                 float* weights, float* shared_scale) {
    const int t = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (t >= tokens) { return; }
    const float* s = scores + static_cast<std::size_t>(t) * (experts + 1);
    // The top_k experts in descending score, the lower id first at a tie.
    int chosen[8];
    for (int r = 0; r < top_k; ++r) {
        int best = -1;
        for (int e = 0; e < experts; ++e) {
            bool taken = false;
            for (int p = 0; p < r; ++p) { taken |= chosen[p] == e; }
            if (!taken && (best < 0 || s[e] > s[best])) { best = e; }
        }
        chosen[r] = best;
    }
    double denominator = 0.0;
    double w[8];
    for (int r = 0; r < top_k; ++r) {
        w[r] = exp(static_cast<double>(s[chosen[r]]) - static_cast<double>(s[chosen[0]]));
        denominator += w[r];
    }
    for (int r = 0; r < top_k; ++r) {
        ids[static_cast<std::size_t>(t) * top_k + r]     = chosen[r];
        weights[static_cast<std::size_t>(t) * top_k + r] = static_cast<float>(w[r] / denominator);
    }
    shared_scale[t] = static_cast<float>(1.0 / (1.0 + exp(-static_cast<double>(s[experts]))));
}

__global__ void moe_finish_kernel(const float* acc, bf16* residual, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    residual[i] = __double2bfloat16(static_cast<double>(__bfloat162float(residual[i])) + static_cast<double>(acc[i]));
}

constexpr int kMoeExperts      = 256;
constexpr int kMoeTopK         = 8;
constexpr int kMoeHidden       = 2048;
constexpr int kMoeIntermediate = 512;

} // namespace

void precise_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const Input in = input_of(x);
    dispatch(in, weight, Rows{weight.n, 0, 0}, Store{output_of(out, weight.n, in.columns)}, stream);
}

void precise_linear_rows(const Tensor& x, const Weight& weight, std::int32_t row_begin, std::int32_t rows,
                         Tensor& out, std::int32_t out_row, cudaStream_t stream) {
    const Input in = input_of(x);
    dispatch(in, weight, Rows{rows, row_begin, 0}, Store{output_of(out, rows, in.columns, out_row)}, stream);
}

void precise_linear_add(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    const Input in = input_of(x);
    require(residual.data != nullptr && residual.dtype == DType::BF16 && residual.is_contiguous() &&
                residual.ne[0] == weight.n && residual.ne[1] == in.columns && residual.ne[2] == 1 && residual.ne[3] == 1,
            "residual must be contiguous BF16 [N, T]");
    dispatch(in, weight, Rows{weight.n, 0, 0}, AddResidual{static_cast<bf16*>(residual.data), weight.n}, stream);
}

void precise_linear(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out, cudaStream_t stream) {
    const Input in = input_of(x);
    require(!weights.empty() && weights.size() <= precise::Bf16Rows::kMaxSegments, "between one and eight weights");
    precise::Bf16Rows d{};
    d.count = static_cast<int>(weights.size());
    d.k     = in.k;
    int rows = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const Tensor& w = *weights[i];
        require(w.data != nullptr && w.dtype == DType::BF16 && w.is_contiguous() && w.ne[0] == in.k && w.ne[1] > 0 &&
                    w.ne[2] == 1 && w.ne[3] == 1 && aligned16(w.data),
                "each weight must be contiguous, 16-byte aligned BF16 [K, N]");
        d.data[i]  = static_cast<const bf16*>(w.data);
        d.begin[i] = rows;
        rows += w.ne[1];
    }
    d.begin[d.count] = rows;
    launch(in, d, Rows{rows, 0, 0}, Store{output_of(out, rows, in.columns)}, stream);
}

void precise_linear(const Tensor& x, std::span<const GgmlSegment> segments, std::span<Tensor* const> outs,
                    const std::int32_t* input_perm, std::int32_t perm_block, cudaStream_t stream) {
    Input in = input_of(x);
    if (input_perm != nullptr) {
        require(perm_block > 0 && perm_block % 8 == 0 && in.k % perm_block == 0,
                "the input permutation's block must be a multiple of 8 dividing K");
        in.perm       = input_perm;
        in.perm_block = perm_block;
    }
    for (const GgmlSegment& s : segments) {
        require(s.data != nullptr && s.rows > 0 && s.output >= 0 && static_cast<std::size_t>(s.output) < outs.size() &&
                    outs[static_cast<std::size_t>(s.output)] != nullptr,
                "each segment needs rows and an output");
        require(in.k % static_cast<int>(ggml_block_elems(s.format)) == 0, "K must be a multiple of the block");
        require(s.layout == GgmlLayout::Rows || ggml_tiles16_supported(s.format), "tiles16 needs a 256-element format");
        const Output out = output_of(*outs[static_cast<std::size_t>(s.output)], s.rows, in.columns, s.out_offset);
        switch (s.format) {
        case QType::GGML_Q2_0: launch_ggml<QType::GGML_Q2_0>(in, s, out, stream); break;
        case QType::GGML_Q4_K: launch_ggml<QType::GGML_Q4_K>(in, s, out, stream); break;
        case QType::GGML_Q5_K: launch_ggml<QType::GGML_Q5_K>(in, s, out, stream); break;
        case QType::GGML_Q6_K: launch_ggml<QType::GGML_Q6_K>(in, s, out, stream); break;
        case QType::GGML_IQ2_S: launch_ggml<QType::GGML_IQ2_S>(in, s, out, stream); break;
        case QType::GGML_IQ3_XXS: launch_ggml<QType::GGML_IQ3_XXS>(in, s, out, stream); break;
        case QType::GGML_IQ3_S: launch_ggml<QType::GGML_IQ3_S>(in, s, out, stream); break;
        case QType::GGML_IQ4_XS: launch_ggml<QType::GGML_IQ4_XS>(in, s, out, stream); break;
        case QType::GGML_IQ4_NL: launch_ggml<QType::GGML_IQ4_NL>(in, s, out, stream); break;
        default: require(false, "unsupported GGUF format");
        }
    }
}

void precise_swiglu(const Tensor& x, const Weight& gate_up, Tensor& out, cudaStream_t stream) {
    const Input in = input_of(x);
    require(gate_up.n > 0 && gate_up.n % 2 == 0, "gate_up must have an even row count");
    const int channels = gate_up.n / 2;
    require(out.dtype == DType::BF16, "out must be BF16");
    dispatch(in, gate_up, Rows{channels, 0, channels}, Swiglu{output_of(out, channels, in.columns)}, stream);
}

void precise_gdn_gating(const Tensor& x, std::span<const Weight* const> ab, const Tensor& a_log, const Tensor& dt_bias,
                        Tensor& g, Tensor& beta, cudaStream_t stream) {
    const Input in = input_of(x);
    int rows       = 0;
    const precise::Bf16Rows d = stacked_bf16(ab, in.k, rows);
    require(rows % 2 == 0, "the a and b rows must pair");
    const int heads = rows / 2;
    require(a_log.dtype == DType::FP32 && dt_bias.dtype == DType::FP32 && a_log.ne[0] == heads &&
                dt_bias.ne[0] == heads,
            "A_log and dt_bias must be FP32 [H]");
    require(g.dtype == DType::FP32 && beta.dtype == DType::FP32 && g.is_contiguous() && beta.is_contiguous() &&
                g.ne[0] == heads && beta.ne[0] == heads && g.ne[1] == in.columns && beta.ne[1] == in.columns,
            "g and beta must be contiguous FP32 [H, T]");
    launch(in, d, Rows{heads, 0, heads},
           GdnGates{static_cast<float*>(g.data), static_cast<float*>(beta.data),
                    static_cast<const float*>(a_log.data), static_cast<const float*>(dt_bias.data), heads},
           stream);
}

std::size_t precise_sparse_moe_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {kMoeExperts + 1, tokens});      // scores
    (void)layout.alloc(DType::I32, {kMoeTopK, tokens});              // routes
    (void)layout.alloc(DType::FP32, {kMoeTopK, tokens});             // route weights
    (void)layout.alloc(DType::FP32, {tokens});                       // shared scale
    (void)layout.alloc(DType::I32, {kMoeTopK * tokens});             // tokens by expert
    (void)layout.alloc(DType::FP32, {kMoeTopK * tokens});            // their weights
    (void)layout.alloc(DType::FP32, {kMoeHidden, tokens});           // accumulator
    (void)layout.alloc(DType::BF16, {kMoeIntermediate, tokens});     // activations
    return layout.peak_bytes(1);
}

void precise_sparse_moe(const Tensor& x, const SparseMoeWeights& weights, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    const Input in   = input_of(x);
    const int tokens = in.columns;
    require(in.k == kMoeHidden && residual.dtype == DType::BF16 && residual.is_contiguous() &&
                residual.ne[0] == kMoeHidden && residual.ne[1] == tokens,
            "the sparse MoE takes BF16 x and residual [2048, T]");
    require(weights.router_shared_gate.n == kMoeExperts + 1 && weights.routed_gate_up.n == kMoeExperts * 2 * kMoeIntermediate &&
                weights.routed_down.n == kMoeExperts * kMoeHidden && weights.routed_down.k == kMoeIntermediate &&
                weights.shared_gate_up.n == 2 * kMoeIntermediate && weights.shared_down.n == kMoeHidden,
            "the sparse MoE weights have the 35B-A3B geometry");
    auto scope    = workspace.scope();
    Tensor scores = workspace.alloc(DType::FP32, {kMoeExperts + 1, tokens});
    Tensor ids    = workspace.alloc(DType::I32, {kMoeTopK, tokens});
    Tensor route  = workspace.alloc(DType::FP32, {kMoeTopK, tokens});
    Tensor shared = workspace.alloc(DType::FP32, {tokens});
    Tensor order  = workspace.alloc(DType::I32, {kMoeTopK * tokens});
    Tensor order_weights = workspace.alloc(DType::FP32, {kMoeTopK * tokens});
    Tensor acc    = workspace.alloc(DType::FP32, {kMoeHidden, tokens});
    Tensor act    = workspace.alloc(DType::BF16, {kMoeIntermediate, tokens});

    // Router scores (FP32), then each token's top 8 (the lower id at a tie), their softmax and the
    // shared expert's sigmoid gate.
    dispatch(in, weights.router_shared_gate, Rows{kMoeExperts + 1, 0, 0},
             Store{output_of(scores, kMoeExperts + 1, tokens)}, stream);
    moe_route_kernel<<<(tokens + 127) / 128, 128, 0, stream>>>(
        static_cast<const float*>(scores.data), kMoeExperts, kMoeTopK, tokens, static_cast<std::int32_t*>(ids.data),
        static_cast<float*>(route.data), static_cast<float*>(shared.data));
    CUDA_CHECK(cudaGetLastError());
    // Each expert's tokens in ascending order (on the host), so every expert runs once over its own
    // columns and the accumulator receives a token's experts in ascending id.
    std::vector<std::int32_t> host_ids(static_cast<std::size_t>(kMoeTopK) * tokens);
    std::vector<float> host_route(host_ids.size());
    CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), ids.data, host_ids.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_route.data(), route.data, host_route.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int> begin(kMoeExperts + 1, 0);
    for (const std::int32_t e : host_ids) { ++begin[static_cast<std::size_t>(e) + 1]; }
    for (int e = 0; e < kMoeExperts; ++e) { begin[e + 1] += begin[e]; }
    std::vector<std::int32_t> host_order(host_ids.size());
    std::vector<float> host_order_weights(host_ids.size());
    std::vector<int> fill(begin.begin(), begin.end() - 1);
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < kMoeTopK; ++r) {
            const std::size_t i  = static_cast<std::size_t>(t) * kMoeTopK + r;
            const int slot       = fill[static_cast<std::size_t>(host_ids[i])]++;
            host_order[slot]         = t;
            host_order_weights[slot] = host_route[i];
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(order.data, host_order.data(), host_order.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(order_weights.data, host_order_weights.data(), host_order_weights.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(acc.data, 0, acc.bytes(), stream));
    const auto* order_ids  = static_cast<const std::int32_t*>(order.data);
    const auto* order_rate = static_cast<const float*>(order_weights.data);
    for (int e = 0; e < kMoeExperts; ++e) {
        const int count = begin[e + 1] - begin[e];
        if (count == 0) { continue; }
        Input gathered      = in;
        gathered.columns    = count;
        gathered.column_map = order_ids + begin[e];
        Tensor expert_act   = act.slice(1, 0, count);
        dispatch(gathered, weights.routed_gate_up, Rows{kMoeIntermediate, e * 2 * kMoeIntermediate, kMoeIntermediate},
                 Swiglu{output_of(expert_act, kMoeIntermediate, count)}, stream);
        dispatch(input_of(expert_act), weights.routed_down, Rows{kMoeHidden, e * kMoeHidden, 0},
                 MoeAccumulate{static_cast<float*>(acc.data), kMoeHidden, order_ids + begin[e], order_rate + begin[e]},
                 stream);
        // The host vectors must outlive the copies above: they are consumed before the next sync.
    }
    // The shared expert, gated per token, after the routed experts.
    dispatch(in, weights.shared_gate_up, Rows{kMoeIntermediate, 0, kMoeIntermediate},
             Swiglu{output_of(act, kMoeIntermediate, tokens)}, stream);
    dispatch(input_of(act), weights.shared_down, Rows{kMoeHidden, 0, 0},
             MoeAccumulate{static_cast<float*>(acc.data), kMoeHidden, nullptr, static_cast<const float*>(shared.data)},
             stream);
    const std::size_t count = static_cast<std::size_t>(kMoeHidden) * tokens;
    moe_finish_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const float*>(acc.data), static_cast<bf16*>(residual.data), count);
    CUDA_CHECK(cudaGetLastError());
    // The pageable host staging is read by the copies above before this returns.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace infernix::ops
