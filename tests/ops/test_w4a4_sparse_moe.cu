// w4a4_sparse_moe (Inkling 2's routed experts, include/infernix/ops/w4a4_sparse_moe.h) at the real geometry
// [128, 2304, 1152], top 8, in both arithmetics.
//
// Inputs: random expert records written directly in each layout's bytes (NVFP4: random codes, E4M3 scale words of
// realistic magnitude, per-expert amax and the checkpoint's denominator 2688; MXFP8: random non-NaN E4M3 codes, UE8M0
// words 110-120), random BF16 x rows quantized on the host by the canonical rules (the MoE mixer's output), and routing
// of 8 distinct experts a column with gammas summing to 8, some experts shared by many columns (passes of more than 8).
//
// Oracles, per entry (column, slot):
//   - NVFP4, exact: the weights decoded from the layout's specification (docs/maintainer/storage-layouts.md §6) as
//     integers, gate/up and down as exact int64 sums of doubled codes times scaled scales, then the pinned binary32
//     epilogue (rs, softcapped SwiGLU, four-over-six h) through the canonical functions: the outputs must equal it bit
//     for bit;
//   - both formats, FP64 stage by stage at the h quantization (a semantic boundary: a sub-ulp difference upstream
//     flips a 4- or 8-bit code there, so an FP64 chain that re-quantizes its own h is no criterion): the kernel's h
//     against FP64 h from FP64 gate/up sums (NVFP4: its BF16 h within 2^-6 of max|h|; MXFP8: its quantized h within an
//     E4M3 half step, 2^-4 |h|, plus 2^-8 max|h|), and each output against FP64 over the kernel's own quantized h
//     within half a BF16 ulp of the FP64 value plus 2^-16 max|y|;
//   - column invariance: each column run alone (T = 1) gives the bits it gets among the call's 32 columns, and 8
//     sampled columns of a 300-column call (a workspace of that capacity: prefill's width) the bits they get in it;
//   - call state: a call over the same workspace after one whose h amax are 64 times larger gives the same bits;
//   - the single-column launch (moe_single): its routing and outputs equal moe_route + moe_experts byte for byte, for 4
//     columns, eagerly and in graphs interleaving the two paths.
//
// The native NVFP4 core (a4_native_step: the m16n8k64 e2m1 MMA, FP32 chains; the decode and verify kernels' default
// since 2026-11-22, Nvfp4Core::Exact selecting the exact core): criteria fixed 2026-11-22 before its first run. Its
// operand mapping exactly (one-hot cases over every row, K index of either operand and column; every scale word pair
// 0x00-0x7E decoding as canon::e4m3_value, the others reported); then the FP64 criteria above, which already hold
// MXFP8's FP32 chains, unchanged for NVFP4, with column invariance and the single launch = the separate kernels byte
// for byte; "equal to the exact arithmetic" is reported only (a changed bit is the core's FP32 chain).

#include "infernix/ops/w4a4_sparse_moe.h"
#include "ops/common/a4_native.cuh"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/offloaded_sparse_moe/nvfp4_a8.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace canon = infernix::ops::canon;
namespace moe   = infernix::ops::w4a4_moe;
using infernix::test::cuda_check;

namespace {

constexpr int E = 128, H = 2304, I = 1152, K = moe::kTopK;
int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class T>
T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(v.size(), 1) * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}

template <class T>
std::vector<T> download(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

// ---------------------------------------------------------------------------- banks

struct HostBank {
    bool a8 = false;
    std::uint64_t stride = 0, gate_up_bytes = 0;
    std::vector<std::uint8_t> records;
    std::vector<float> amax; // [E][2]
    float den[2] = {2688.0F, 2688.0F};
};

std::uint64_t align4k(std::uint64_t v) { return (v + 4095) / 4096 * 4096; }

HostBank make_bank(bool a8, std::uint32_t seed) {
    HostBank b;
    b.a8                 = a8;
    const int unit       = a8 ? 528 : 144, block = a8 ? 32 : 16;
    b.gate_up_bytes      = std::uint64_t(2 * I / 16) * (H / block) * unit;
    const std::uint64_t record = b.gate_up_bytes + std::uint64_t(H / 16) * (I / block) * unit;
    b.stride             = align4k(record);
    b.records.assign(E * b.stride, 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte(0, 255), scale4(0x28, 0x4E), scale8(110, 120);
    for (int e = 0; e < E; ++e) {
        std::uint8_t* r = b.records.data() + e * b.stride;
        const std::uint64_t units = record / unit;
        for (std::uint64_t u = 0; u < units; ++u) {
            std::uint8_t* p = r + u * unit;
            const int codes = a8 ? 512 : 128;
            for (int i = 0; i < codes; ++i) {
                int c = byte(rng);
                if (a8 && (c & 0x7F) == 0x7F) { c = 0x3C; }
                p[i] = static_cast<std::uint8_t>(c);
            }
            for (int i = 0; i < 16; ++i) { p[codes + i] = static_cast<std::uint8_t>(a8 ? scale8(rng) : scale4(rng)); }
        }
    }
    std::uniform_real_distribution<float> amax(0.05F, 0.33F);
    b.amax.resize(2 * E);
    for (float& a : b.amax) { a = amax(rng); }
    return b;
}

// Decoded weights of expert e, as the layouts' specification defines them: NVFP4 rows of (doubled code, scale word)
// pairs; MXFP8 rows of (code word, UE8M0 word).
struct Matrix {
    int rows = 0, k = 0;
    std::vector<std::uint8_t> code, scale; // [rows][k] codes (NVFP4: the 4-bit code; MXFP8: the E4M3 word), [rows][k / block]
};

Matrix decode(const HostBank& b, int e, bool down) {
    const int unit = b.a8 ? 528 : 144, block = b.a8 ? 32 : 16;
    Matrix m;
    m.rows = down ? H : 2 * I;
    m.k    = down ? I : H;
    m.code.assign(std::size_t(m.rows) * m.k, 0);
    m.scale.assign(std::size_t(m.rows) * (m.k / block), 0);
    const std::uint8_t* base = b.records.data() + e * b.stride + (down ? b.gate_up_bytes : 0);
    const int blocks = m.k / block;
    for (int rg = 0; rg < m.rows / 16; ++rg) {
        for (int bk = 0; bk < blocks; ++bk) {
            const std::uint8_t* u = base + (std::size_t(rg) * blocks + bk) * unit;
            if (!b.a8) {
                // byte 32q + 4j + t: row 16rg + j (low nibble) and 16rg + 8 + j (high nibble) at k = 16b + 4q + t
                for (int q = 0; q < 4; ++q) {
                    for (int j = 0; j < 8; ++j) {
                        for (int t = 0; t < 4; ++t) {
                            const std::uint8_t byte = u[32 * q + 4 * j + t];
                            const int k            = 16 * bk + 4 * q + t;
                            m.code[std::size_t(16 * rg + j) * m.k + k]     = byte & 0xF;
                            m.code[std::size_t(16 * rg + 8 + j) * m.k + k] = byte >> 4;
                        }
                    }
                }
            } else {
                // lane l = 4g + t at 16l: row g k 4t.., row g+8 same k, row g k 16+4t.., row g+8 k 16+4t..
                for (int l = 0; l < 32; ++l) {
                    const int g = l / 4, t = l % 4;
                    for (int part = 0; part < 4; ++part) {
                        const int row = 16 * rg + g + ((part & 1) ? 8 : 0);
                        const int k0  = 32 * bk + ((part & 2) ? 16 : 0) + 4 * t;
                        for (int j = 0; j < 4; ++j) { m.code[std::size_t(row) * m.k + k0 + j] = u[16 * l + 4 * part + j]; }
                    }
                }
            }
            for (int r = 0; r < 16; ++r) { m.scale[std::size_t(16 * rg + r) * blocks + bk] = u[(b.a8 ? 512 : 128) + r]; }
        }
    }
    return m;
}

double e4m3_signed(std::uint8_t w) {
    const double v = canon::e4m3_value(w & 0x7FU);
    return (w & 0x80U) ? -v : v;
}

double ue8m0(std::uint8_t e) { return e == 0 ? std::ldexp(1.0, -127) : std::ldexp(1.0, int(e) - 127); }

// ---------------------------------------------------------------------------- activations

struct QuantRow {
    std::vector<std::uint8_t> bytes; // the row format of moe::row_bytes
    float amax = 0;
};

QuantRow quantize_a4(const std::vector<float>& v) {
    const int n = static_cast<int>(v.size());
    QuantRow q;
    q.bytes.assign(n / 2 + n / 16, 0);
    for (const float x : v) { q.amax = std::max(q.amax, std::fabs(x)); }
    const canon::FourOverSixRow row = canon::four_over_six_row(q.amax);
    for (int b = 0; b < n / 16; ++b) {
        float x[16];
        for (int j = 0; j < 16; ++j) { x[j] = v[16 * b + j]; }
        const canon::FourOverSixBlock blk = canon::four_over_six_block(x, row);
        for (int j = 0; j < 8; ++j) { q.bytes[8 * b + j] = blk.code[2 * j] | (blk.code[2 * j + 1] << 4); }
        q.bytes[n / 2 + b] = blk.scale_word;
    }
    return q;
}

QuantRow quantize_a8(const std::vector<float>& v) {
    const int n = static_cast<int>(v.size());
    QuantRow q;
    q.bytes.assign(n + n / 32, 0);
    for (int b = 0; b < n / 32; ++b) {
        float x[32];
        for (int j = 0; j < 32; ++j) { x[j] = v[32 * b + j]; }
        const canon::Mxfp8Group grp = canon::mxfp8_group(x);
        for (int j = 0; j < 32; ++j) { q.bytes[32 * b + j] = grp.code[j]; }
        q.bytes[n + b] = grp.exponent;
    }
    return q;
}

// The doubled E2M1 value and the scaled scale of element k of an NVFP4 row.
int a4_code2(const std::vector<std::uint8_t>& row, int k) {
    const std::uint8_t byte = row[k / 2];
    return canon::e2m1_x2((k & 1) ? byte >> 4 : byte & 0xF);
}

// ---------------------------------------------------------------------------- oracle

struct EntryResult {
    std::vector<std::uint16_t> exact; // NVFP4: the pinned arithmetic, exactly
    std::vector<double> h64;          // h in FP64 from FP64 gate/up sums
};

// The FP64 down projection of expert e over a quantized h row (the row format of moe::row_bytes; NVFP4 with its amax).
std::vector<double> down64(const HostBank& b, const Matrix& dn, int e, const std::uint8_t* hq, float h_amax) {
    std::vector<double> hv(I);
    for (int k = 0; k < I; ++k) {
        if (!b.a8) {
            const std::uint8_t byte = hq[k / 2];
            hv[k] = 0.5 * canon::e2m1_x2((k & 1) ? byte >> 4 : byte & 0xF) * canon::e4m3_value(hq[I / 2 + k / 16]) *
                    (double(h_amax) / 1536.0);
        } else {
            hv[k] = e4m3_signed(hq[k]) * ue8m0(hq[I + k / 32]);
        }
    }
    std::vector<double> out(H);
    for (int r = 0; r < H; ++r) {
        double d = 0;
        for (int k = 0; k < I; ++k) {
            const std::uint8_t c = dn.code[std::size_t(r) * I + k];
            const double w       = b.a8 ? e4m3_signed(c) * ue8m0(dn.scale[std::size_t(r) * (I / 32) + k / 32])
                                        : 0.5 * canon::e2m1_x2(c) * canon::e4m3_value(dn.scale[std::size_t(r) * (I / 16) + k / 16]) *
                                        (double(b.amax[2 * e + 1]) / b.den[1]);
            d += w * hv[k];
        }
        out[r] = d;
    }
    return out;
}

std::uint16_t swiglu(std::uint16_t g, std::uint16_t u, float gamma) {
    const float gate = canon::bf16_to_f32(g), up = canon::bf16_to_f32(u);
    const float a    = canon::mul_rn(canon::softcap_c(gate, 4.0F), canon::sigmoid_c(gate));
    return canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(a, canon::softcap_c(up, 25.0F)), gamma));
}

double swiglu64(double g, double u, double gamma) {
    return 4.0 * std::tanh(g / 4.0) / (1.0 + std::exp(-g)) * 25.0 * std::tanh(u / 25.0) * gamma;
}

EntryResult oracle(const HostBank& b, const Matrix& gu, const Matrix& dn, int e, const QuantRow& x, float gamma) {
    EntryResult out;
    // gate/up rows: 2i gate, 2i + 1 up. y: the pinned arithmetic (NVFP4) or the FP64 sum rounded (MXFP8, whose FP32
    // accumulation the FP64 checks bound); y64: FP64.
    std::vector<std::uint16_t> y(2 * I);
    std::vector<double> y64(2 * I);
    const float rcp1 = canon::div_rn(1.0F, canon::mul_rn(1536.0F, b.den[0]));
    const float rs1  = canon::mul_rn(canon::mul_rn(x.amax, b.amax[2 * e]), rcp1);
    for (int r = 0; r < 2 * I; ++r) {
        if (!b.a8) {
            long long s = 0;
            for (int blk = 0; blk < H / 16; ++blk) {
                long long p = 0;
                for (int j = 0; j < 16; ++j) {
                    const int k = 16 * blk + j;
                    p += static_cast<long long>(canon::e2m1_x2(gu.code[std::size_t(r) * H + k])) * a4_code2(x.bytes, k);
                }
                s += p * canon::e4m3_scaled(gu.scale[std::size_t(r) * (H / 16) + blk]) *
                     canon::e4m3_scaled(x.bytes[H / 2 + blk]);
            }
            y64[r] = static_cast<double>(s) * std::ldexp(1.0, -20) * (double(x.amax) * b.amax[2 * e] / (1536.0 * b.den[0]));
            y[r]   = canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(s), 1.0F / 1048576.0F), rs1));
        } else {
            double d = 0;
            for (int blk = 0; blk < H / 32; ++blk) {
                double p = 0;
                for (int j = 0; j < 32; ++j) {
                    const int k = 32 * blk + j;
                    p += e4m3_signed(gu.code[std::size_t(r) * H + k]) * e4m3_signed(x.bytes[k]);
                }
                d += p * ue8m0(gu.scale[std::size_t(r) * (H / 32) + blk]) * ue8m0(x.bytes[H + blk]);
            }
            y64[r] = d;
            y[r]   = canon::f32_to_bf16_rn(static_cast<float>(d));
        }
    }
    std::vector<float> h(I);
    out.h64.resize(I);
    for (int i = 0; i < I; ++i) {
        h[i]       = canon::bf16_to_f32(swiglu(y[2 * i], y[2 * i + 1], gamma));
        out.h64[i] = swiglu64(y64[2 * i], y64[2 * i + 1], gamma);
    }
    if (b.a8) { return out; }
    // NVFP4: the exact pinned arithmetic of the down projection over the pinned h's four-over-six.
    const QuantRow hq = quantize_a4(h);
    out.exact.resize(H);
    const float rcp2 = canon::div_rn(1.0F, canon::mul_rn(1536.0F, b.den[1]));
    const float rs2  = canon::mul_rn(canon::mul_rn(hq.amax, b.amax[2 * e + 1]), rcp2);
    for (int r = 0; r < H; ++r) {
        long long s = 0;
        for (int blk = 0; blk < I / 16; ++blk) {
            long long p = 0;
            for (int j = 0; j < 16; ++j) {
                const int k = 16 * blk + j;
                p += static_cast<long long>(canon::e2m1_x2(dn.code[std::size_t(r) * I + k])) * a4_code2(hq.bytes, k);
            }
            s += p * canon::e4m3_scaled(dn.scale[std::size_t(r) * (I / 16) + blk]) * canon::e4m3_scaled(hq.bytes[I / 2 + blk]);
        }
        out.exact[r] = canon::f32_to_bf16_rn(canon::mul_rn(canon::mul_rn(canon::i64_to_f32_rn(s), 1.0F / 1048576.0F), rs2));
    }
    return out;
}

// ---------------------------------------------------------------------------- a call

struct Call {
    int columns = 0;
    std::vector<std::vector<float>> x; // BF16 values
    std::vector<std::int32_t> experts; // [T][K]
    std::vector<float> gamma;          // [T][K]
};

Call make_call(int columns, std::uint32_t seed, float scale) {
    Call c;
    c.columns = columns;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_real_distribution<float> unit(0.2F, 1.0F);
    for (int t = 0; t < columns; ++t) {
        std::vector<float> row(H);
        for (float& v : row) { v = canon::bf16_to_f32(canon::f32_to_bf16_rn(normal(rng) * scale)); }
        c.x.push_back(row);
        // Experts 0-3 in every column (jobs of many columns, passes of 8), the rest random and distinct.
        std::vector<int> chosen = {0, 1, 2, 3};
        while (chosen.size() < K) {
            const int e = static_cast<int>(rng() % E);
            if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) { chosen.push_back(e); }
        }
        std::shuffle(chosen.begin(), chosen.end(), rng);
        float sum = 0;
        std::vector<float> w(K);
        for (float& v : w) { sum += v = unit(rng); }
        for (int j = 0; j < K; ++j) {
            c.experts.push_back(chosen[j]);
            c.gamma.push_back(w[j] / sum * 8.0F);
        }
    }
    return c;
}

struct DeviceBank {
    std::uint8_t* records = nullptr;
    float* amax = nullptr;
    float* den = nullptr;
};

struct Result {
    std::vector<std::uint16_t> out;    // [T * K][H]
    std::vector<std::uint16_t> h;      // NVFP4: [T * K][I] BF16 h
    std::vector<std::uint8_t> h_rows;  // [T * K][row bytes] the quantized h
    std::vector<std::uint32_t> h_amax; // NVFP4: [T * K]
};

// A column's A16 encoding (the arithmetic's definition, design §16.2.1 of Flash-Next): its exponent, X_j and the limb
// row in a16_limbs.cuh's planes (per 64-element tile: 64 H1 = X >> 14, 64 L1 = (X >> 3) & 255, 64 H0 | L0 << 4).
struct LimbRow {
    std::vector<std::uint8_t> bytes;
    std::vector<std::int32_t> X;
    int emax = 0;
};

LimbRow encode_a16(const std::vector<float>& v) {
    const int n = static_cast<int>(v.size());
    std::vector<std::uint16_t> bits(n);
    for (int j = 0; j < n; ++j) { bits[j] = canon::f32_to_bf16_rn(v[j]); }
    LimbRow r;
    r.emax = canon::a16_column_exponent(bits.data(), n);
    r.X.resize(n);
    r.bytes.assign(std::size_t(n) / 64 * 192, 0);
    for (int j = 0; j < n; ++j) {
        const std::int32_t x = r.emax == canon::kA16NonFinite ? 0 : canon::a16_value(bits[j], r.emax);
        r.X[j]               = x;
        const int blk = j / 16, e = j % 16;
        std::uint8_t* at = r.bytes.data() + (blk / 4) * 192 + 16 * (blk % 4) + e;
        at[0]            = static_cast<std::uint8_t>(x >> 14);
        at[64]           = static_cast<std::uint8_t>((x >> 3) & 255);
        at[128]          = static_cast<std::uint8_t>(((x >> 11) & 7) | ((x & 7) << 4));
    }
    return r;
}

__global__ void empty_gate_up_kernel() {}

// A lost count bounds a whole round, not each wait (candidates JM-02): the call's graph with its gate/up pass emptied, so
// no down CTA's job count ever arrives. The first replay waits out one bound (2 s) and sets kErrorDownStalled; replays
// that find the word already set give up after the waits' poll interval (1 ms), not a bound a CTA wave each, which is
// how an emptied producer once spun ~12 h. The word is cleared again afterwards.
void check_stall_bound(cudaGraph_t graph, const moe::Workspace& ws, cudaStream_t stream) {
    static bool checked = false;
    if (checked) { return; }
    checked = true;
    cuda_check(cudaFuncSetAttribute(empty_gate_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024),
               "attribute");
    cudaGraph_t stalled = nullptr;
    cuda_check(cudaGraphClone(&stalled, graph), "clone");
    std::size_t count = 0;
    cuda_check(cudaGraphGetNodes(stalled, nullptr, &count), "nodes");
    std::vector<cudaGraphNode_t> nodes(count);
    cuda_check(cudaGraphGetNodes(stalled, nodes.data(), &count), "nodes");
    int emptied = 0;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        cuda_check(cudaGraphNodeGetType(node, &type), "type");
        if (type != cudaGraphNodeTypeKernel) { continue; }
        cudaKernelNodeParams params{};
        cuda_check(cudaGraphKernelNodeGetParams(node, &params), "params");
        const char* name = nullptr;
        cuda_check(cudaFuncGetName(&name, params.func), "name");
        if (name == nullptr || std::strstr(name, "gate_up") == nullptr) { continue; }
        params.func         = reinterpret_cast<void*>(empty_gate_up_kernel);
        params.kernelParams = nullptr;
        params.extra        = nullptr;
        cuda_check(cudaGraphKernelNodeSetParams(node, &params), "set params");
        ++emptied;
    }
    check(emptied >= 1, "the stall check found no gate/up pass to empty");
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaGraphInstantiate(&exec, stalled, 0), "instantiate");
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    const auto replay_ms = [&] {
        cuda_check(cudaEventRecord(start, stream), "event");
        cuda_check(cudaGraphLaunch(exec, stream), "stalled replay");
        cuda_check(cudaEventRecord(stop, stream), "event");
        cuda_check(cudaEventSynchronize(stop), "stalled replay");
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "event");
        return ms;
    };
    const float first = replay_ms();
    check(download(ws.error, 1)[0] == moe::kErrorDownStalled, "a lost count did not set kErrorDownStalled");
    float later = 0;
    for (int i = 0; i < 2; ++i) { later = std::max(later, replay_ms()); }
    std::printf("stall bound: first replay %.0f ms (one 2 s bound), replays with the word set at most %.1f ms\n", first,
                later);
    check(first >= 1900.0F && first < 8000.0F, "the first stalled replay did not wait about one bound");
    check(later < 200.0F, "a wait that found the error word set still waited out its bound");
    cuda_check(cudaMemsetAsync(ws.error, 0, sizeof(std::uint32_t), stream), "clear");
    cuda_check(cudaStreamSynchronize(stream), "clear");
    cuda_check(cudaEventDestroy(start), "event");
    cuda_check(cudaEventDestroy(stop), "event");
    cuda_check(cudaGraphExecDestroy(exec), "destroy");
    cuda_check(cudaGraphDestroy(stalled), "destroy");
}

Result run(const HostBank& b, const DeviceBank& db, const Call& call, void* workspace,
           int capacity = moe::kMaxColumns, bool a16 = false, moe::Nvfp4Core core = moe::Nvfp4Core::Native,
           bool a8x = false) {
    std::vector<std::uint8_t> rows;
    std::vector<float> amax;
    std::vector<std::int32_t> emax;
    std::vector<std::uint16_t> x16;
    for (const auto& x : call.x) {
        for (const float v : x) { x16.push_back(canon::f32_to_bf16_rn(v)); }
        if ((a16 && b.a8) || a8x) { // MXFP8 A16: the BF16 rows alone; NVFP4 A8: rows encoded on the device below
            amax.push_back(0.0F);
            continue;
        }
        if (a16 && b.a8) {
            amax.push_back(0.0F);
            continue;
        }
        if (a16) {
            const LimbRow l = encode_a16(x);
            rows.insert(rows.end(), l.bytes.begin(), l.bytes.end());
            emax.push_back(l.emax);
            amax.push_back(0.0F);
            continue;
        }
        const QuantRow q = b.a8 ? quantize_a8(x) : quantize_a4(x);
        rows.insert(rows.end(), q.bytes.begin(), q.bytes.end());
        amax.push_back(q.amax);
    }
    if (emax.empty()) { emax.push_back(0); }
    if (rows.empty()) { rows.push_back(0); }
    std::uint8_t* d_rows  = upload(rows);
    float* d_amax         = upload(amax);
    std::int32_t* d_emax  = upload(emax);
    std::uint16_t* d_x16  = upload(x16);
    moe::Activations act{d_rows, d_amax, call.columns};
    act.core = core;
    if (a16) {
        act.activation = moe::ExpertActivation::A16;
        act.rows       = b.a8 ? nullptr : d_rows;
        act.emax       = b.a8 ? nullptr : d_emax;
        act.x16        = d_x16;
    }
    std::uint8_t* d_a8 = nullptr;
    if (a8x) { // the production encoder: the oracle encodes from the codec's definition
        cuda_check(cudaMalloc(&d_a8, std::size_t(call.columns) * moe::kA8XRowBytes), "cudaMalloc");
        moe::moe_encode_a8(d_x16, call.columns, d_a8, nullptr);
        act.activation = moe::ExpertActivation::A8;
        act.rows       = d_a8;
    }
    std::int32_t* d_exp  = upload(call.experts);
    float* d_gamma       = upload(call.gamma);
    std::uint16_t* d_out = nullptr;
    const std::size_t entries = std::size_t(call.columns) * K;
    cuda_check(cudaMalloc(&d_out, entries * H * 2), "cudaMalloc");
    const moe::Workspace ws = moe::carve_workspace(workspace, E, I, capacity,
                                                   a8x   ? moe::ExpertActivation::A8
                                                   : a16 ? moe::ExpertActivation::A16
                                                         : moe::ExpertActivation::Checkpoint);
    const moe::Routing routing{d_exp, d_gamma};
    moe::Bank bank;
    bank.records       = db.records;
    bank.record_stride = b.stride;
    bank.format        = b.a8 ? moe::ExpertFormat::Mxfp8 : moe::ExpertFormat::Nvfp4FourOverSix;
    bank.amax          = db.amax;
    bank.denominators  = db.den;
    bank.experts       = E;
    bank.hidden        = H;
    bank.intermediate  = I;
    moe::moe_dispatch(routing, call.columns, E, ws, nullptr);
    moe::moe_experts(act, routing, bank, ws, d_out, nullptr);
    cuda_check(cudaDeviceSynchronize(), "moe call");
    Result r;
    r.out = download(d_out, entries * H);
    // The same call captured in a CUDA graph, where the down pass is a programmatic dependent of the gate/up pass
    // (its CTAs wait on their jobs' counts while gate/up CTAs still run), replayed three times: the eager bits.
    {
        cudaStream_t stream;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        moe::moe_dispatch(routing, call.columns, E, ws, stream);
        moe::moe_experts(act, routing, bank, ws, d_out, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            cuda_check(cudaMemsetAsync(d_out, 0xFF, entries * H * 2, stream), "clear");
            cuda_check(cudaGraphLaunch(exec, stream), "replay");
            cuda_check(cudaStreamSynchronize(stream), "replay");
            check(download(d_out, entries * H) == r.out, "a graph replay (programmatic down pass) differs from the eager call");
        }
        if (call.columns > 1) { check_stall_bound(graph, ws, stream); }
        cuda_check(cudaGraphExecDestroy(exec), "destroy");
        cuda_check(cudaGraphDestroy(graph), "destroy");
        cuda_check(cudaStreamDestroy(stream), "stream");
        check(download(ws.error, 1)[0] == 0U, "the call's error word is set");
    }
    r.h      = download(ws.h, entries * I);
    r.h_rows = download(ws.h_rows, entries * moe::row_bytes(bank.format, I));
    r.h_amax = download(ws.h_amax, entries);
    for (void* p : {static_cast<void*>(d_rows), static_cast<void*>(d_amax), static_cast<void*>(d_exp),
                    static_cast<void*>(d_gamma), static_cast<void*>(d_out), static_cast<void*>(d_emax),
                    static_cast<void*>(d_x16), static_cast<void*>(d_a8)}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
    return r;
}

// ---------------------------------------------------------------------------- A8 (--a8-activations), NVFP4

// A vector's A8 slices by the codec's definition (ops/ggml/ggml_a8.h): codes q and scales d (a non-finite slice: d NaN).
struct A8Row {
    std::vector<std::int32_t> q;
    std::vector<float> d;
};
A8Row encode_a8(const std::vector<float>& v) {
    namespace a8 = infernix::ggml::a8;
    A8Row r;
    r.q.resize(v.size());
    for (std::size_t s = 0; s < v.size() / 32; ++s) {
        std::uint16_t bits[32];
        unsigned key = 0;
        for (int j = 0; j < 32; ++j) {
            bits[j] = canon::f32_to_bf16_rn(v[32 * s + j]);
            key     = std::max(key, static_cast<unsigned>(bits[j] & 0x7FFFU));
        }
        const a8::SliceScale sc = a8::slice_scale(key);
        r.d.push_back(sc.d);
        for (int j = 0; j < 32; ++j) { r.q[32 * s + j] = a8::code(bits[j], sc.inv); }
    }
    return r;
}

// The A8 outputs of entry (x, e, gamma) by the header's order: per row acc = +0, then per block in order
// acc = fma(P_b * S_b, d_s, acc) (P_b * S_b exact), y = bf16(fl32(fl32(acc * 2^-10) * m)); h = swiglu * gamma, its
// slices encoded again, the down rows likewise.
std::vector<std::uint16_t> oracle_a8(const HostBank& b, const Matrix& gu, const Matrix& dn, int e, const A8Row& x,
                                     float gamma) {
    const float m1 = canon::div_rn(b.amax[2 * e], b.den[0]), m2 = canon::div_rn(b.amax[2 * e + 1], b.den[1]);
    const auto row = [](const Matrix& w, int r, const A8Row& a, float m) {
        float acc = 0.0F;
        for (int blk = 0; blk < w.k / 16; ++blk) {
            int p = 0;
            for (int j = 0; j < 16; ++j) {
                const int k = 16 * blk + j;
                p += canon::e2m1_x2(w.code[std::size_t(r) * w.k + k]) * a.q[k];
            }
            const float ps = static_cast<float>(p) * static_cast<float>(canon::e4m3_scaled(w.scale[std::size_t(r) * (w.k / 16) + blk]));
            acc = canon::fma_rn(ps, a.d[blk / 2], acc);
        }
        return infernix::ops::offloaded_moe::nvfp4_a8::row_output(acc, m);
    };
    std::vector<std::uint16_t> y(2 * I);
    for (int r = 0; r < 2 * I; ++r) { y[r] = row(gu, r, x, m1); }
    std::vector<float> h(I);
    for (int i = 0; i < I; ++i) { h[i] = canon::bf16_to_f32(swiglu(y[2 * i], y[2 * i + 1], gamma)); }
    const A8Row ha = encode_a8(h);
    std::vector<std::uint16_t> out(H);
    for (int r = 0; r < H; ++r) { out[r] = row(dn, r, ha, m2); }
    return out;
}

// Criteria (fixed before the first run): every output equals the order's bits at widths 1, 5, 32, 128 and 300 (graph
// replays included, in run()), a non-finite column gives the canonical NaN in all its outputs, and a column alone gives
// the bits it gets in a wider call. (The order's accuracy: one FP32 chain of exact block terms, K / 16 roundings.)
void a8_case() {
    const HostBank b = make_bank(false, 43U);
    DeviceBank db{upload(b.records), upload(b.amax), upload(std::vector<float>{b.den[0], b.den[1]})};
    constexpr int kCapacity = 300;
    const std::size_t bytes = moe::workspace_bytes(E, I, kCapacity, moe::ExpertActivation::A8);
    void* workspace = nullptr;
    cuda_check(cudaMalloc(&workspace, bytes), "cudaMalloc");
    cuda_check(cudaMemset(workspace, 0, bytes), "cudaMemset");
    std::vector<Matrix> cache_gu(E), cache_dn(E);
    std::vector<bool> decoded(E, false);
    int differ = 0, checked = 0;
    Result wide;
    Call wide_call;
    for (const int columns : {1, 5, 32, 128, kCapacity}) {
        Call call = make_call(columns, 950U + static_cast<std::uint32_t>(columns), columns == 32 ? 16.0F : 1.0F);
        if (columns == 32) { call.x[3][100] = INFINITY; } // a non-finite column
        const Result got = run(b, db, call, workspace, kCapacity, false, moe::Nvfp4Core::Native, true);
        const int entries = columns * K, step = columns >= 128 ? 16 : 1;
        for (int entry = 0; entry < entries; entry += step) {
            const int t = entry / K, e = call.experts[entry];
            if (!decoded[e]) {
                cache_gu[e] = decode(b, e, false);
                cache_dn[e] = decode(b, e, true);
                decoded[e]  = true;
            }
            const std::vector<std::uint16_t> want =
                oracle_a8(b, cache_gu[e], cache_dn[e], e, encode_a8(call.x[t]), call.gamma[entry]);
            for (int i = 0; i < H; ++i) {
                const std::uint16_t g = got.out[std::size_t(entry) * H + i];
                const bool nan = (want[i] & 0x7FFFU) > 0x7F80U;
                differ += (nan ? g != 0x7FC0U : g != want[i]) ? 1 : 0;
            }
            checked += H;
            if (columns == 32 && t == 3) {
                bool all_nan = true;
                for (int i = 0; i < H; ++i) { all_nan = all_nan && got.out[std::size_t(entry) * H + i] == 0x7FC0U; }
                check(all_nan, "A8: a non-finite column's outputs are not the canonical NaN");
            }
        }
        if (columns == kCapacity) {
            wide      = got;
            wide_call = call;
        }
    }
    int alone_differ = 0;
    for (const int t : {0, 9, 150, 299}) {
        Call one;
        one.columns = 1;
        one.x       = {wide_call.x[t]};
        one.experts.assign(wide_call.experts.begin() + t * K, wide_call.experts.begin() + (t + 1) * K);
        one.gamma.assign(wide_call.gamma.begin() + t * K, wide_call.gamma.begin() + (t + 1) * K);
        const Result alone = run(b, db, one, workspace, kCapacity, false, moe::Nvfp4Core::Native, true);
        for (int i = 0; i < K * H; ++i) { alone_differ += alone.out[i] != wide.out[std::size_t(t) * K * H + i] ? 1 : 0; }
    }
    std::printf("NVFP4 A8: %d of %d outputs differ from the order's bits; %d outputs of 4 columns alone differ from the "
                "300-column call\n",
                differ, checked, alone_differ);
    check(differ == 0, "A8 outputs differ from the order's bits");
    check(alone_differ == 0, "an A8 column alone differs from the column in a wider call");
    // The prompt route (moe_experts_prompt: the grouped 8-bit GEMM) on the 300-column call and on one column: the
    // narrow route's bits, every output (the same order and terms), and a repeated call the same.
    const auto run_prompt = [&](const Call& call) {
        std::vector<std::uint16_t> x16;
        for (const auto& x : call.x) {
            for (const float v : x) { x16.push_back(canon::f32_to_bf16_rn(v)); }
        }
        std::uint16_t* d_x16 = upload(x16);
        std::uint8_t* d_a8   = nullptr;
        cuda_check(cudaMalloc(&d_a8, std::size_t(call.columns) * moe::kA8XRowBytes), "cudaMalloc");
        moe::moe_encode_a8(d_x16, call.columns, d_a8, nullptr);
        std::int32_t* d_exp = upload(call.experts);
        float* d_gamma      = upload(call.gamma);
        const std::size_t entries = std::size_t(call.columns) * K;
        std::uint16_t* d_out      = nullptr;
        cuda_check(cudaMalloc(&d_out, entries * H * 2), "cudaMalloc");
        const moe::Workspace ws = moe::carve_workspace(workspace, E, I, kCapacity, moe::ExpertActivation::A8);
        moe::Bank bank;
        bank.records       = db.records;
        bank.record_stride = b.stride;
        bank.format        = moe::ExpertFormat::Nvfp4FourOverSix;
        bank.amax          = db.amax;
        bank.denominators  = db.den;
        bank.experts       = E;
        bank.hidden        = H;
        bank.intermediate  = I;
        const moe::Routing routing{d_exp, d_gamma};
        moe::Activations act{d_a8, nullptr, call.columns};
        act.activation = moe::ExpertActivation::A8;
        moe::moe_dispatch(routing, call.columns, E, ws, nullptr);
        moe::moe_experts_prompt(act, routing, bank, ws, d_out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "A8 prompt call");
        std::vector<std::uint16_t> out = download(d_out, entries * H);
        for (void* p : {static_cast<void*>(d_x16), static_cast<void*>(d_a8), static_cast<void*>(d_exp),
                        static_cast<void*>(d_gamma), static_cast<void*>(d_out)}) {
            cuda_check(cudaFree(p), "cudaFree");
        }
        return out;
    };
    const std::vector<std::uint16_t> prompt = run_prompt(wide_call), prompt_again = run_prompt(wide_call);
    int prompt_differ = 0;
    for (std::size_t i = 0; i < prompt.size(); ++i) { prompt_differ += prompt[i] != wide.out[i] ? 1 : 0; }
    Call one;
    one.columns = 1;
    one.x       = {wide_call.x[150]};
    one.experts.assign(wide_call.experts.begin() + 150 * K, wide_call.experts.begin() + 151 * K);
    one.gamma.assign(wide_call.gamma.begin() + 150 * K, wide_call.gamma.begin() + 151 * K);
    const std::vector<std::uint16_t> prompt_one = run_prompt(one);
    int one_differ = 0;
    for (int i = 0; i < K * H; ++i) { one_differ += prompt_one[i] != wide.out[std::size_t(150) * K * H + i] ? 1 : 0; }
    std::printf("NVFP4 A8 prompt route: %d of %zu outputs of the 300-column call differ from the narrow route, %d of %d "
                "of a 1-column call; a repeated call %s\n",
                prompt_differ, prompt.size(), one_differ, K * H, prompt == prompt_again ? "the same" : "DIFFERS");
    check(prompt_differ == 0 && one_differ == 0, "the A8 prompt route differs from the narrow route");
    check(prompt == prompt_again, "a repeated A8 prompt call differs");
    for (void* p : {static_cast<void*>(db.records), static_cast<void*>(db.amax), static_cast<void*>(db.den), workspace}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
}

// ---------------------------------------------------------------------------- A16 (--a16-activations), NVFP4

// The exact A16 outputs of entry (x, e, gamma): gate/up S = sum_b P_b e4m3_scaled(s_b) over the column's X, y =
// a16_row_output(S, emax, fl32(amax_w / den)), h = swiglu(y_gate, y_up) * gamma (BF16), h encoded the same way, and
// the down outputs likewise.
std::vector<std::uint16_t> oracle_a16(const HostBank& b, const Matrix& gu, const Matrix& dn, int e, const LimbRow& x,
                                      float gamma) {
    const float m1 = canon::div_rn(b.amax[2 * e], b.den[0]), m2 = canon::div_rn(b.amax[2 * e + 1], b.den[1]);
    const auto row_sum = [](const Matrix& w, int r, const std::vector<std::int32_t>& X) {
        long long s = 0;
        for (int blk = 0; blk < w.k / 16; ++blk) {
            long long p = 0;
            for (int j = 0; j < 16; ++j) {
                const int k = 16 * blk + j;
                p += static_cast<long long>(canon::e2m1_x2(w.code[std::size_t(r) * w.k + k])) * X[k];
            }
            s += p * canon::e4m3_scaled(w.scale[std::size_t(r) * (w.k / 16) + blk]);
        }
        return s;
    };
    std::vector<std::uint16_t> y(2 * I);
    for (int r = 0; r < 2 * I; ++r) { y[r] = canon::a16_row_output(row_sum(gu, r, x.X), x.emax, m1); }
    std::vector<float> h(I);
    for (int i = 0; i < I; ++i) { h[i] = canon::bf16_to_f32(swiglu(y[2 * i], y[2 * i + 1], gamma)); }
    const LimbRow hl = encode_a16(h);
    std::vector<std::uint16_t> out(H);
    for (int r = 0; r < H; ++r) { out[r] = canon::a16_row_output(row_sum(dn, r, hl.X), hl.emax, m2); }
    return out;
}

// Criteria (fixed before the first run): every output equals the exact arithmetic bit for bit at widths 1, 5, 32 and
// 128 (graph replays included, in run()), a non-finite column gives the canonical NaN in all its outputs, and a column
// alone gives the bits it gets in a wider call.
void a16_case() {
    const HostBank b = make_bank(false, 41U);
    DeviceBank db{upload(b.records), upload(b.amax), upload(std::vector<float>{b.den[0], b.den[1]})};
    constexpr int kCapacity = 300; // a prefill-width call too
    const std::size_t bytes = moe::workspace_bytes(E, I, kCapacity, moe::ExpertActivation::A16);
    void* workspace = nullptr;
    cuda_check(cudaMalloc(&workspace, bytes), "cudaMalloc");
    cuda_check(cudaMemset(workspace, 0, bytes), "cudaMemset");
    std::vector<Matrix> cache_gu(E), cache_dn(E);
    std::vector<bool> decoded(E, false);
    int differ = 0, checked = 0;
    Result wide;
    Call wide_call;
    for (const int columns : {1, 5, 32, 128, kCapacity}) {
        Call call = make_call(columns, 900U + static_cast<std::uint32_t>(columns), columns == 32 ? 16.0F : 1.0F);
        if (columns == 32) { call.x[3][100] = INFINITY; } // a non-finite column
        const Result got = run(b, db, call, workspace, kCapacity, true);
        // The exact arithmetic over a sample of entries (every entry of the narrow calls, every 16th of the wider).
        const int entries = columns * K, step = columns >= 128 ? 16 : 1;
        for (int entry = 0; entry < entries; entry += step) {
            const int t = entry / K, e = call.experts[entry];
            if (!decoded[e]) {
                cache_gu[e] = decode(b, e, false);
                cache_dn[e] = decode(b, e, true);
                decoded[e]  = true;
            }
            const std::vector<std::uint16_t> want = oracle_a16(b, cache_gu[e], cache_dn[e], e, encode_a16(call.x[t]),
                                                               call.gamma[entry]);
            for (int i = 0; i < H; ++i) { differ += got.out[std::size_t(entry) * H + i] != want[i] ? 1 : 0; }
            checked += H;
            if (columns == 32 && t == 3) {
                bool all_nan = true;
                for (int i = 0; i < H; ++i) { all_nan = all_nan && got.out[std::size_t(entry) * H + i] == 0x7FC0U; }
                check(all_nan, "A16: a non-finite column's outputs are not the canonical NaN");
            }
        }
        if (columns == kCapacity) {
            wide      = got;
            wide_call = call;
        }
    }
    // Columns of the widest call alone.
    int alone_differ = 0;
    for (const int t : {0, 9, 150, 299}) {
        Call one;
        one.columns = 1;
        one.x       = {wide_call.x[t]};
        one.experts.assign(wide_call.experts.begin() + t * K, wide_call.experts.begin() + (t + 1) * K);
        one.gamma.assign(wide_call.gamma.begin() + t * K, wide_call.gamma.begin() + (t + 1) * K);
        const Result alone = run(b, db, one, workspace, kCapacity, true);
        for (int i = 0; i < K * H; ++i) { alone_differ += alone.out[i] != wide.out[std::size_t(t) * K * H + i] ? 1 : 0; }
    }
    std::printf("NVFP4 A16: %d of %d outputs differ from the exact arithmetic; %d outputs of 4 columns alone differ "
                "from the 300-column call\n",
                differ, checked, alone_differ);
    check(differ == 0, "A16 outputs differ from the exact arithmetic");
    check(alone_differ == 0, "an A16 column alone differs from the column in a wider call");
    for (void* p : {static_cast<void*>(db.records), static_cast<void*>(db.amax), static_cast<void*>(db.den), workspace}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
}

// `core`: NVFP4's decode / verify core (both run: the native default under the FP64 criteria, the exact one also equal
// to the pinned arithmetic bit for bit).
void bank_case(bool a8, moe::Nvfp4Core core = moe::Nvfp4Core::Native) {
    const bool exact_core = !a8 && core == moe::Nvfp4Core::Exact;
    const char* name = a8 ? "MXFP8" : exact_core ? "NVFP4 (exact core)" : "NVFP4 (native core)";
    const HostBank b = make_bank(a8, a8 ? 81U : 41U);
    DeviceBank db{upload(b.records), upload(b.amax), upload(std::vector<float>{b.den[0], b.den[1]})};
    void* workspace = nullptr;
    cuda_check(cudaMalloc(&workspace, moe::workspace_bytes(E, I)), "cudaMalloc");
    cuda_check(cudaMemset(workspace, 0, moe::workspace_bytes(E, I)), "cudaMemset");
    const int row_bytes = moe::row_bytes(a8 ? moe::ExpertFormat::Mxfp8 : moe::ExpertFormat::Nvfp4FourOverSix, I);
    const auto go = [&](const Call& c, void* w, int capacity) { return run(b, db, c, w, capacity, false, core); };

    // A call whose h amax are 64 times larger first: the next call must not see its state.
    (void)go(make_call(32, 7, 64.0F), workspace, moe::kMaxColumns);
    const Call call    = make_call(32, 11, 1.0F);
    const Result got   = go(call, workspace, moe::kMaxColumns);
    const Result again = go(call, workspace, moe::kMaxColumns);
    check(got.out == again.out, "a repeated call gives the same bits");

    int exact_differ = 0;
    double worst_h = 0, worst_y = 0;
    std::vector<Matrix> cache_gu(E), cache_dn(E);
    std::vector<bool> decoded(E, false);
    for (int t = 0; t < call.columns; ++t) {
        const QuantRow xq = a8 ? quantize_a8(call.x[t]) : quantize_a4(call.x[t]);
        for (int j = 0; j < K; ++j) {
            const int e = call.experts[t * K + j], entry = t * K + j;
            if (!decoded[e]) {
                cache_gu[e] = decode(b, e, false);
                cache_dn[e] = decode(b, e, true);
                decoded[e]  = true;
            }
            const EntryResult r = oracle(b, cache_gu[e], cache_dn[e], e, xq, call.gamma[entry]);
            const std::uint8_t* hq = got.h_rows.data() + std::size_t(entry) * row_bytes;
            // h against FP64.
            double h_max = 0, h_err = 0;
            for (int i = 0; i < I; ++i) { h_max = std::max(h_max, std::fabs(r.h64[i])); }
            for (int i = 0; i < I; ++i) {
                if (!a8) {
                    h_err = std::max(h_err, std::fabs(canon::bf16_to_f32(got.h[std::size_t(entry) * I + i]) - r.h64[i]) / h_max);
                } else {
                    const double deq   = e4m3_signed(hq[i]) * ue8m0(hq[I + i / 32]);
                    const double bound = std::ldexp(std::fabs(r.h64[i]), -4) + std::ldexp(h_max, -8);
                    h_err = std::max(h_err, std::fabs(deq - r.h64[i]) / bound * std::ldexp(1.0, -6)); // 2^-6 = at the bound
                }
            }
            worst_h = std::max(worst_h, h_err);
            // Outputs against FP64 over the kernel's own quantized h.
            const float h_amax = canon::f32_from_bits(got.h_amax[entry]);
            const std::vector<double> y64 = down64(b, cache_dn[e], e, hq, h_amax);
            double y_max = 0, y_err = 0;
            for (int i = 0; i < H; ++i) { y_max = std::max(y_max, std::fabs(y64[i])); }
            for (int i = 0; i < H; ++i) {
                const std::uint16_t gpu = got.out[std::size_t(entry) * H + i];
                // Half a BF16 ulp of the FP64 value (correct rounding), plus 2^-16 max|y| for the binary32 rescale
                // and accumulation: err / bound <= 1.
                const double half_ulp = y64[i] == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(y64[i]) - 8);
                const double bound    = half_ulp + std::ldexp(y_max, -16);
                y_err = std::max(y_err, std::fabs(canon::bf16_to_f32(gpu) - y64[i]) / bound);
                if (!a8 && gpu != r.exact[i]) { ++exact_differ; }
            }
            worst_y = std::max(worst_y, y_err);
        }
    }
    std::printf("%s: %d entries; h vs FP64 worst %.3e (criterion 2^-6 = %.3e); outputs vs FP64 over the kernel's h worst "
                "%.3f of half a BF16 ulp + 2^-16 max|y| (criterion 1)\n",
                name, call.columns * K, worst_h, std::ldexp(1.0, -6), worst_y);
    check(worst_h <= std::ldexp(1.0, -6), "h is further from FP64 than its criterion");
    check(worst_y <= 1.0, "an output is further than half a BF16 ulp from FP64 over the kernel's h");
    if (!a8) {
        // The exact core must equal the pinned arithmetic; the native core's count is reported (its FP32 chain).
        std::printf("%s: %d outputs differ from the exact pinned arithmetic%s\n", name, exact_differ,
                    exact_core ? "" : " (reported)");
        if (exact_core) { check(exact_differ == 0, "NVFP4 outputs differ from the exact pinned arithmetic"); }
    }

    // Column invariance: each of the first 8 columns alone.
    int invariance_differ = 0;
    for (int t = 0; t < 8; ++t) {
        Call one;
        one.columns = 1;
        one.x       = {call.x[t]};
        one.experts.assign(call.experts.begin() + t * K, call.experts.begin() + (t + 1) * K);
        one.gamma.assign(call.gamma.begin() + t * K, call.gamma.begin() + (t + 1) * K);
        const Result alone = go(one, workspace, moe::kMaxColumns);
        for (int i = 0; i < K * H; ++i) { invariance_differ += alone.out[i] != got.out[std::size_t(t) * K * H + i] ? 1 : 0; }
    }
    std::printf("%s: column invariance: %d outputs differ between T = 1 and T = 32\n", name, invariance_differ);
    check(invariance_differ == 0, "a column's outputs depend on the call's other columns");

    // A prefill-width call (300 columns, a workspace of that capacity: experts 0-3 take a job of 300 entries, its last
    // CTA quantizing every one of their h rows): sampled columns alone give the bits they get in it.
    {
        constexpr int kWide = 300;
        void* wide_ws       = nullptr;
        cuda_check(cudaMalloc(&wide_ws, moe::workspace_bytes(E, I, kWide)), "cudaMalloc");
        cuda_check(cudaMemset(wide_ws, 0, moe::workspace_bytes(E, I, kWide)), "cudaMemset");
        const Call wide   = make_call(kWide, 13, 1.0F);
        const Result all  = go(wide, wide_ws, kWide);
        int wide_differ   = 0;
        for (const int t : {0, 7, 8, 63, 150, 255, 256, 299}) {
            Call one;
            one.columns = 1;
            one.x       = {wide.x[t]};
            one.experts.assign(wide.experts.begin() + t * K, wide.experts.begin() + (t + 1) * K);
            one.gamma.assign(wide.gamma.begin() + t * K, wide.gamma.begin() + (t + 1) * K);
            const Result alone = go(one, workspace, moe::kMaxColumns);
            for (int i = 0; i < K * H; ++i) { wide_differ += alone.out[i] != all.out[std::size_t(t) * K * H + i] ? 1 : 0; }
        }
        std::printf("%s: a 300-column call: %d outputs of 8 sampled columns differ from the columns alone\n", name,
                    wide_differ);
        check(wide_differ == 0, "a column's outputs in a 300-column call differ from the column alone");
        cuda_check(cudaFree(wide_ws), "cudaFree");
    }
    for (void* p : {static_cast<void*>(db.records), static_cast<void*>(db.amax), static_cast<void*>(db.den), workspace}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
}

// ---------------------------------------------------------------------------- A16, MXFP8 (W8A16)

// Criteria (fixed before the first run): h within 2^-6 of max|h| of FP64 h from FP64 gate/up sums over the BF16 x; each
// output within half a BF16 ulp of FP64 over the kernel's own BF16 h plus (K/16 + K/32 + 5) 2^-23 sum|w h| (the FP32
// chain: a block's two MMAs from zero, its scaled fma); columns alone give the bits they get in a 300-column call.
void mx_a16_case() {
    const HostBank b = make_bank(true, 81U);
    DeviceBank db{upload(b.records), upload(b.amax), upload(std::vector<float>{b.den[0], b.den[1]})};
    constexpr int kCapacity = 300;
    const std::size_t bytes = moe::workspace_bytes(E, I, kCapacity, moe::ExpertActivation::A16);
    void* workspace = nullptr;
    cuda_check(cudaMalloc(&workspace, bytes), "cudaMalloc");
    cuda_check(cudaMemset(workspace, 0, bytes), "cudaMemset");
    std::vector<Matrix> cache_gu(E), cache_dn(E);
    std::vector<bool> decoded(E, false);
    const auto weight = [](const Matrix& m, int r, int k) {
        return e4m3_signed(m.code[std::size_t(r) * m.k + k]) * ue8m0(m.scale[std::size_t(r) * (m.k / 32) + k / 32]);
    };
    double worst_h = 0, worst_y = 0;
    Result wide;
    Call wide_call;
    for (const int columns : {1, 32, kCapacity}) {
        const Call call  = make_call(columns, 950U + static_cast<std::uint32_t>(columns), 1.0F);
        const Result got = run(b, db, call, workspace, kCapacity, true);
        const int step   = columns >= 128 ? 16 : 1;
        for (int entry = 0; entry < columns * K; entry += step) {
            const int t = entry / K, e = call.experts[entry];
            if (!decoded[e]) {
                cache_gu[e] = decode(b, e, false);
                cache_dn[e] = decode(b, e, true);
                decoded[e]  = true;
            }
            std::vector<double> h64(I);
            double h_max = 0;
            for (int i = 0; i < I; ++i) {
                double gs = 0, us = 0;
                for (int k = 0; k < H; ++k) {
                    gs += weight(cache_gu[e], 2 * i, k) * call.x[t][k];
                    us += weight(cache_gu[e], 2 * i + 1, k) * call.x[t][k];
                }
                h64[i] = swiglu64(gs, us, call.gamma[entry]);
                h_max  = std::max(h_max, std::fabs(h64[i]));
            }
            const std::uint16_t* hk = got.h.data() + std::size_t(entry) * I;
            for (int i = 0; i < I; ++i) { worst_h = std::max(worst_h, std::fabs(canon::bf16_to_f32(hk[i]) - h64[i]) / h_max); }
            for (int r = 0; r < H; ++r) {
                double y = 0, mag = 0;
                for (int k = 0; k < I; ++k) {
                    const double p = weight(cache_dn[e], r, k) * canon::bf16_to_f32(hk[k]);
                    y += p;
                    mag += std::fabs(p);
                }
                const double half_ulp = y == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(y) - 8);
                const double bound    = half_ulp + (I / 16 + I / 32 + 5) * std::ldexp(mag, -23);
                const double out      = canon::bf16_to_f32(got.out[std::size_t(entry) * H + r]);
                worst_y = std::max(worst_y, bound == 0.0 ? (out == 0.0 ? 0.0 : 1e9) : std::fabs(out - y) / bound);
            }
        }
        if (columns == kCapacity) {
            wide      = got;
            wide_call = call;
        }
    }
    int alone_differ = 0;
    for (const int t : {0, 9, 150, 299}) {
        Call one;
        one.columns = 1;
        one.x       = {wide_call.x[t]};
        one.experts.assign(wide_call.experts.begin() + t * K, wide_call.experts.begin() + (t + 1) * K);
        one.gamma.assign(wide_call.gamma.begin() + t * K, wide_call.gamma.begin() + (t + 1) * K);
        const Result alone = run(b, db, one, workspace, kCapacity, true);
        for (int i = 0; i < K * H; ++i) { alone_differ += alone.out[i] != wide.out[std::size_t(t) * K * H + i] ? 1 : 0; }
    }
    std::printf("MXFP8 A16: h vs FP64 worst %.3e (criterion 2^-6 = %.3e); outputs vs FP64 over the kernel's h worst %.3f of "
                "the bound (criterion 1); %d outputs of 4 columns alone differ from the 300-column call\n",
                worst_h, std::ldexp(1.0, -6), worst_y, alone_differ);
    check(worst_h <= std::ldexp(1.0, -6), "MXFP8 A16: h is further from FP64 than its criterion");
    check(worst_y <= 1.0, "MXFP8 A16: an output is outside its bound");
    check(alone_differ == 0, "MXFP8 A16: a column alone differs from the column in a wider call");
    for (void* p : {static_cast<void*>(db.records), static_cast<void*>(db.amax), static_cast<void*>(db.den), workspace}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
}

// ---------------------------------------------------------------------------- the prompt route (NVFP4)

// moe_experts_prompt at prefill widths. Criteria (fixed before the first run): h within 2^-6 of max|h| of FP64 h from
// FP64 gate/up sums (as the narrow route); each output within half a BF16 ulp of FP64 over the kernel's own quantized
// h, plus the FP32 tensor-core sum's bound (K / 64 + 5) 2^-23 sum |w h| scaled, plus 2^-16 max|y|; every sampled
// column alone (a 1-column prompt call) gives the bits it gets in the call; outputs differing from the exact narrow
// route are counted and reported, not judged (another arithmetic).
void prompt_case(bool a8) {
    const HostBank b = make_bank(a8, a8 ? 81U : 41U);
    DeviceBank db{upload(b.records), upload(b.amax), upload(std::vector<float>{b.den[0], b.den[1]})};
    constexpr int kWide = 300;
    void* workspace     = nullptr;
    const std::size_t bytes = moe::workspace_bytes(E, I, kWide);
    cuda_check(cudaMalloc(&workspace, bytes), "cudaMalloc");
    cuda_check(cudaMemset(workspace, 0, bytes), "cudaMemset");
    const moe::Workspace ws = moe::carve_workspace(workspace, E, I, kWide);
    moe::Bank bank;
    bank.records       = db.records;
    bank.record_stride = b.stride;
    bank.format        = a8 ? moe::ExpertFormat::Mxfp8 : moe::ExpertFormat::Nvfp4FourOverSix;
    bank.amax          = db.amax;
    bank.denominators  = db.den;
    bank.experts       = E;
    bank.hidden        = H;
    bank.intermediate  = I;
    struct Out {
        std::vector<std::uint16_t> out, h;
        std::vector<std::uint8_t> plane;
        std::vector<std::uint32_t> h_amax;
        std::vector<std::int32_t> entries;
    };
    const auto run_prompt = [&](const Call& call, bool narrow) {
        std::vector<std::uint8_t> rows;
        std::vector<float> amax;
        for (const auto& x : call.x) {
            const QuantRow q = a8 ? quantize_a8(x) : quantize_a4(x);
            rows.insert(rows.end(), q.bytes.begin(), q.bytes.end());
            amax.push_back(q.amax);
        }
        std::uint8_t* d_rows = upload(rows);
        float* d_amax        = upload(amax);
        std::int32_t* d_exp  = upload(call.experts);
        float* d_gamma       = upload(call.gamma);
        const std::size_t entries = std::size_t(call.columns) * K;
        std::uint16_t* d_out      = nullptr;
        cuda_check(cudaMalloc(&d_out, entries * H * 2), "cudaMalloc");
        const moe::Routing routing{d_exp, d_gamma};
        moe::moe_dispatch(routing, call.columns, E, ws, nullptr);
        const moe::Activations x{d_rows, d_amax, call.columns};
        narrow ? moe::moe_experts(x, routing, bank, ws, d_out, nullptr)
               : moe::moe_experts_prompt(x, routing, bank, ws, d_out, nullptr);
        cuda_check(cudaDeviceSynchronize(), "prompt call");
        Out o;
        o.out     = download(d_out, entries * H);
        o.h       = download(ws.h, entries * I);
        o.plane   = a8 ? download(ws.x_plane, entries * moe::kPromptXRowBytes) : download(ws.h_plane, entries * moe::kPromptHRowBytes);
        o.h_amax  = download(ws.h_amax, entries);
        o.entries = download(ws.entries, entries);
        for (void* p : {static_cast<void*>(d_rows), static_cast<void*>(d_amax), static_cast<void*>(d_exp),
                        static_cast<void*>(d_gamma), static_cast<void*>(d_out)}) {
            cuda_check(cudaFree(p), "cudaFree");
        }
        return o;
    };
    const Call call   = make_call(kWide, 13, 1.0F);
    const Out got     = run_prompt(call, false);
    const Out again   = run_prompt(call, false);
    const Out narrow  = run_prompt(call, true);
    check(got.out == again.out, "a repeated prompt call gives the same bits");
    // The h plane's rows are in dispatch order: entry -> position.
    std::vector<int> position(std::size_t(kWide) * K);
    for (std::size_t p = 0; p < got.entries.size(); ++p) { position[got.entries[p]] = static_cast<int>(p); }
    double worst_h = 0, worst_y = 0;
    long narrow_differ = 0, checked = 0;
    std::vector<Matrix> cache_gu(E), cache_dn(E);
    std::vector<bool> decoded(E, false);
    for (const int t : {0, 1, 7, 63, 150, 255, 299}) {
        const QuantRow xq = a8 ? quantize_a8(call.x[t]) : quantize_a4(call.x[t]);
        for (int j = 0; j < K; ++j) {
            const int e = call.experts[t * K + j], entry = t * K + j;
            if (!decoded[e]) {
                cache_gu[e] = decode(b, e, false);
                cache_dn[e] = decode(b, e, true);
                decoded[e]  = true;
            }
            const EntryResult r = oracle(b, cache_gu[e], cache_dn[e], e, xq, call.gamma[entry]);
            double h_max = 0, h_err = 0;
            for (int i = 0; i < I; ++i) { h_max = std::max(h_max, std::fabs(r.h64[i])); }
            for (int i = 0; i < I; ++i) {
                h_err = std::max(h_err, std::fabs(canon::bf16_to_f32(got.h[std::size_t(entry) * I + i]) - r.h64[i]) / h_max);
            }
            worst_h = std::max(worst_h, h_err);
            // The kernel's quantized h from its plane row in the natural row format. NVFP4: block b at plane_block(b)
            // (byte j low/high = elements j and 4 + j, byte 4 + j = 8 + j and 12 + j; scales after the 576 code bytes);
            // MXFP8: the x plane's row, already natural (1,152 codes, 36 UE8M0 words).
            const std::uint8_t* row = got.plane.data() + std::size_t(position[entry]) *
                                                             (a8 ? moe::kPromptXRowBytes : moe::kPromptHRowBytes);
            std::vector<std::uint8_t> hq(a8 ? I + I / 32 : I / 2 + I / 16, 0);
            if (a8) { std::copy_n(row, hq.size(), hq.begin()); }
            for (int blk = 0; !a8 && blk < I / 16; ++blk) {
                const int pos = (blk & ~3) | ((blk & 1) << 1) | ((blk & 2) >> 1);
                std::uint8_t c[16];
                for (int jj = 0; jj < 4; ++jj) {
                    c[jj]      = row[8 * pos + jj] & 0xF;
                    c[4 + jj]  = row[8 * pos + jj] >> 4;
                    c[8 + jj]  = row[8 * pos + 4 + jj] & 0xF;
                    c[12 + jj] = row[8 * pos + 4 + jj] >> 4;
                }
                for (int jj = 0; jj < 8; ++jj) { hq[8 * blk + jj] = c[2 * jj] | (c[2 * jj + 1] << 4); }
                hq[I / 2 + blk] = row[I / 2 + pos];
            }
            const float h_amax = canon::f32_from_bits(got.h_amax[entry]);
            const std::vector<double> y64 = down64(b, cache_dn[e], e, hq.data(), h_amax);
            // sum |w h| of each output, for the FP32 sum's bound.
            std::vector<double> mag(H, 0.0);
            for (int rr = 0; rr < H; ++rr) {
                double m = 0;
                for (int k = 0; k < I; ++k) {
                    double hv = 0, w = 0;
                    if (a8) {
                        hv = e4m3_signed(hq[k]) * ue8m0(hq[I + k / 32]);
                        w  = e4m3_signed(cache_dn[e].code[std::size_t(rr) * I + k]) *
                            ue8m0(cache_dn[e].scale[std::size_t(rr) * (I / 32) + k / 32]);
                    } else {
                        const std::uint8_t byte = hq[k / 2];
                        hv = 0.5 * canon::e2m1_x2((k & 1) ? byte >> 4 : byte & 0xF) * canon::e4m3_value(hq[I / 2 + k / 16]) *
                             (double(h_amax) / 1536.0);
                        w = 0.5 * canon::e2m1_x2(cache_dn[e].code[std::size_t(rr) * I + k]) *
                            canon::e4m3_value(cache_dn[e].scale[std::size_t(rr) * (I / 16) + k / 16]) *
                            (double(b.amax[2 * e + 1]) / b.den[1]);
                    }
                    m += std::fabs(w * hv);
                }
                mag[rr] = m;
            }
            double y_max = 0;
            for (int i = 0; i < H; ++i) { y_max = std::max(y_max, std::fabs(y64[i])); }
            for (int i = 0; i < H; ++i) {
                const std::uint16_t gpu = got.out[std::size_t(entry) * H + i];
                const double half_ulp   = y64[i] == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(y64[i]) - 8);
                // The FP32 sum's steps: K / 64 MMAs (NVFP4 m16n8k64) or K / 32 (MXFP8 m16n8k32).
                const double bound = half_ulp + ((a8 ? I / 32 : I / 64) + 5) * std::ldexp(mag[i], -23) + std::ldexp(y_max, -16);
                worst_y = std::max(worst_y, std::fabs(canon::bf16_to_f32(gpu) - y64[i]) / bound);
                narrow_differ += gpu != narrow.out[std::size_t(entry) * H + i] ? 1 : 0;
                ++checked;
            }
        }
    }
    // Column invariance: sampled columns alone.
    int alone_differ = 0;
    for (const int t : {0, 7, 63, 150, 299}) {
        Call one;
        one.columns = 1;
        one.x       = {call.x[t]};
        one.experts.assign(call.experts.begin() + t * K, call.experts.begin() + (t + 1) * K);
        one.gamma.assign(call.gamma.begin() + t * K, call.gamma.begin() + (t + 1) * K);
        const Out alone = run_prompt(one, false);
        for (int i = 0; i < K * H; ++i) { alone_differ += alone.out[i] != got.out[std::size_t(t) * K * H + i] ? 1 : 0; }
    }
    std::printf("%s prompt route (300 columns): h vs FP64 worst %.3e (criterion 2^-6 = %.3e); outputs vs FP64 over the "
                "kernel's h worst %.3f of the bound (criterion 1); %d outputs of 5 columns differ from the columns alone; "
                "%ld of %ld checked outputs differ from the narrow route (reported)\n",
                a8 ? "MXFP8" : "NVFP4", worst_h, std::ldexp(1.0, -6), worst_y, alone_differ, narrow_differ, checked);
    check(worst_h <= std::ldexp(1.0, -6), "prompt route: h is further from FP64 than its criterion");
    check(worst_y <= 1.0, "prompt route: an output is outside its bound from FP64 over the kernel's h");
    check(alone_differ == 0, "prompt route: a column's outputs depend on the call's other columns");
    for (void* p : {static_cast<void*>(db.records), static_cast<void*>(db.amax), static_cast<void*>(db.den), workspace}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
}

// ---------------------------------------------------------------------------- router

void router_case() {
    constexpr int T = 40; // three column tiles of the router's grid
    std::mt19937 rng(23);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> w(std::size_t(E) * H), x(std::size_t(T) * H);
    for (auto& v : w) { v = canon::f32_to_bf16_rn(normal(rng) * 0.02F); }
    for (auto& v : x) { v = canon::f32_to_bf16_rn(normal(rng)); }
    std::vector<float> bias(E);
    for (float& b : bias) { b = normal(rng) * 0.01F; }
    // A tie: experts 5 and 9 get the same weights and bias, so in each column their keys are equal.
    std::copy(w.begin() + 5 * H, w.begin() + 6 * H, w.begin() + 9 * H);
    bias[9] = bias[5];
    std::uint16_t* d_w = upload(w);
    std::uint16_t* d_x = upload(x);
    float* d_bias      = upload(bias);
    void* workspace    = nullptr;
    cuda_check(cudaMalloc(&workspace, moe::workspace_bytes(E, I)), "cudaMalloc");
    cuda_check(cudaMemset(workspace, 0, moe::workspace_bytes(E, I)), "cudaMemset");
    const moe::Workspace ws = moe::carve_workspace(workspace, E, I);
    std::vector<std::int32_t> routed[2];
    std::vector<float> gamma[2];
    for (int call = 0; call < 2; ++call) {
        moe::moe_route(d_x, T, d_w, d_bias, E, H, 8.0F, ws, nullptr);
        cuda_check(cudaDeviceSynchronize(), "router");
        routed[call] = download(ws.routed, std::size_t(T) * K);
        gamma[call]  = download(ws.gamma, std::size_t(T) * K);
    }
    check(routed[0] == routed[1] && gamma[0] == gamma[1], "a repeated router call gives the same routing");
    const auto logits  = download(ws.logits, std::size_t(T) * E);
    const auto offsets = download(ws.offsets, E + 1);
    const auto entries = download(ws.entries, std::size_t(T) * K);
    const auto jobs    = download(ws.jobs, E);
    const int job_count = download(ws.job_count, 1)[0];
    int logit_bad = 0, emulation_differ = 0, fp64_differ = 0, near_ties = 0, gamma_bad = 0;
    double worst_gamma_ulp = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<double> l64(E);
        std::vector<float> keys(E);
        for (int e = 0; e < E; ++e) {
            double d = 0, mag = 0;
            for (int k = 0; k < H; ++k) {
                const double p = double(canon::bf16_to_f32(w[std::size_t(e) * H + k])) * canon::bf16_to_f32(x[std::size_t(t) * H + k]);
                d += p;
                mag += std::fabs(p);
            }
            l64[e] = d;
            const float gpu = logits[std::size_t(t) * E + e];
            if (std::fabs(gpu - d) > std::ldexp(mag, -17)) { ++logit_bad; }
            keys[e] = canon::add_rn(canon::sigmoid_c(gpu), bias[e]);
        }
        // The pinned rule over the GPU's logits, exactly.
        std::vector<int> order(E);
        for (int e = 0; e < E; ++e) { order[e] = e; }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] > keys[b]; });
        float s[K];
        for (int j = 0; j < K; ++j) {
            s[j] = canon::sigmoid_c(logits[std::size_t(t) * E + order[j]]);
            emulation_differ += routed[0][t * K + j] != order[j] ? 1 : 0;
        }
        const float sum = canon::add_rn(canon::add_rn(canon::add_rn(s[0], s[2]), canon::add_rn(s[1], s[3])),
                                        canon::add_rn(canon::add_rn(s[4], s[6]), canon::add_rn(s[5], s[7])));
        double s64[K], sum64 = 0;
        for (int j = 0; j < K; ++j) { sum64 += s64[j] = 1.0 / (1.0 + std::exp(-l64[order[j]])); }
        for (int j = 0; j < K; ++j) {
            const float g = canon::mul_rn(canon::div_rn(s[j], sum), 8.0F);
            gamma_bad += gamma[0][t * K + j] != g ? 1 : 0;
            const double ref = s64[j] / sum64 * 8.0;
            worst_gamma_ulp  = std::max(worst_gamma_ulp, std::fabs(gamma[0][t * K + j] - ref) / std::ldexp(1.0, std::ilogb(ref) - 23));
        }
        // Against FP64 keys: the same set unless the 8th and 9th FP64 keys are within 2^-20.
        std::vector<double> k64(E);
        for (int e = 0; e < E; ++e) { k64[e] = 1.0 / (1.0 + std::exp(-l64[e])) + bias[e]; }
        std::vector<int> order64(E);
        for (int e = 0; e < E; ++e) { order64[e] = e; }
        std::stable_sort(order64.begin(), order64.end(), [&](int a, int b) { return k64[a] > k64[b]; });
        std::vector<int> a(order.begin(), order.begin() + K), b(order64.begin(), order64.begin() + K);
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
            if (std::fabs(k64[order64[K - 1]] - k64[order64[K]]) <= std::ldexp(1.0, -20)) {
                ++near_ties;
            } else {
                ++fp64_differ;
            }
        }
    }
    // The dispatch: every entry once, grouped by expert in ascending order, jobs ascending.
    bool dispatch_ok = offsets[E] == T * K;
    int listed       = 0;
    for (int e = 0; e < E && dispatch_ok; ++e) {
        for (int i = offsets[e]; i < offsets[e + 1]; ++i) {
            dispatch_ok = dispatch_ok && routed[0][entries[i]] == e && (i == offsets[e] || entries[i] > entries[i - 1]);
        }
        if (offsets[e + 1] > offsets[e]) { dispatch_ok = dispatch_ok && listed < job_count && jobs[listed++] == e; }
    }
    dispatch_ok = dispatch_ok && listed == job_count;
    std::printf("router: %d columns; logits outside 2^-17 sum|xw|: %d; routing differs from the pinned rule: %d; gamma "
                "differs: %d (worst %.2f ulp of FP64); sets differ from FP64: %d (+ %d near-ties); dispatch %s\n",
                T, logit_bad, emulation_differ, gamma_bad, worst_gamma_ulp, fp64_differ, near_ties,
                dispatch_ok ? "consistent" : "INCONSISTENT");
    check(logit_bad == 0, "router logits are outside their FP32 summation bound");
    check(emulation_differ == 0 && gamma_bad == 0, "the routing differs from the pinned rule over its logits");
    check(worst_gamma_ulp <= 3.0, "a gamma is more than 3 ulp from FP64");
    check(fp64_differ == 0, "a column selects other experts than FP64 keys beyond a near-tie");
    check(dispatch_ok, "the router's dispatch is inconsistent");

    // The expert prefetch's hint: a call with a record gives the same routing and records its last column's experts in
    // slot order; moe_prefetch_l2 takes ids outside the bank (skipped) and refuses what it cannot issue.
    std::int32_t* d_record = upload(std::vector<std::int32_t>(K, -7));
    moe::moe_route(d_x, T, d_w, d_bias, E, H, 8.0F, ws, nullptr, d_record);
    cuda_check(cudaDeviceSynchronize(), "router with a record");
    const auto routed_rec = download(ws.routed, std::size_t(T) * K);
    const auto record     = download(d_record, K);
    check(routed_rec == routed[0], "a router call with a record changes the routing");
    check(std::equal(record.begin(), record.end(), routed_rec.begin() + std::size_t(T - 1) * K),
          "the record is not the last column's experts in slot order");
    constexpr std::uint64_t kStride = 65536, kBytes = 32768;
    void* records = nullptr;
    cuda_check(cudaMalloc(&records, std::size_t(E) * kStride), "cudaMalloc");
    moe::Bank toy;
    toy.records       = static_cast<const std::uint8_t*>(records);
    toy.record_stride = kStride;
    toy.experts       = E;
    std::int32_t* d_ids = upload(std::vector<std::int32_t>{3, -1, 3, E, 127, 0, 64, 3});
    moe::moe_prefetch_l2(d_ids, 8, toy, kBytes, nullptr);
    moe::moe_prefetch_l2(d_record, K, toy, kBytes, nullptr);
    cuda_check(cudaDeviceSynchronize(), "expert prefetch");
    int refused = 0;
    for (const auto& [count, bytes] : {std::pair<int, std::uint64_t>{0, kBytes}, {17, kBytes}, {8, kBytes + 8}, {8, kStride + 16}}) {
        try {
            moe::moe_prefetch_l2(d_ids, count, toy, bytes, nullptr);
        } catch (const std::invalid_argument&) {
            ++refused;
        }
    }
    std::printf("router record: the last column's %d experts; expert prefetch ran, %d of 4 bad arguments refused\n", K, refused);
    check(refused == 4, "moe_prefetch_l2 accepted arguments it cannot issue");
    for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x), static_cast<void*>(d_bias), workspace,
                    static_cast<void*>(d_record), records, static_cast<void*>(d_ids)}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
}

// ---------------------------------------------------------------------------- one column in one launch

// moe_single against moe_route + moe_experts over the same workspace: routing and outputs byte for byte, for several
// columns; eagerly and in a CUDA graph that interleaves the two paths (single, separate, single: each leaves the
// workspace's state as the other needs it), replayed three times.
void single_case(bool a8, bool a16 = false, moe::Nvfp4Core core = moe::Nvfp4Core::Native, bool a8x = false) {
    const char* name   = a8                              ? "MXFP8"
                         : a16                           ? "NVFP4 A16"
                         : a8x                           ? "NVFP4 A8"
                         : core == moe::Nvfp4Core::Exact ? "NVFP4 (exact core)"
                                                         : "NVFP4 (native core)";
    const auto format  = a8 ? moe::ExpertFormat::Mxfp8 : moe::ExpertFormat::Nvfp4FourOverSix;
    const auto activation = a16 ? moe::ExpertActivation::A16 : a8x ? moe::ExpertActivation::A8 : moe::ExpertActivation::Checkpoint;
    if (!moe::moe_single_supported(format, activation)) {
        std::printf("%s single-column launch: not supported on this device (the separate kernels run instead)\n", name);
        return;
    }
    const HostBank b = make_bank(a8, a8 ? 83U : 43U);
    DeviceBank db{upload(b.records), upload(b.amax), upload(std::vector<float>{b.den[0], b.den[1]})};
    moe::Bank bank;
    bank.records       = db.records;
    bank.record_stride = b.stride;
    bank.format        = format;
    bank.amax          = db.amax;
    bank.denominators  = db.den;
    bank.experts       = E;
    bank.hidden        = H;
    bank.intermediate  = I;
    std::mt19937 rng(a8 ? 97 : 47);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> w(std::size_t(E) * H);
    for (auto& v : w) { v = canon::f32_to_bf16_rn(normal(rng) * 0.02F); }
    std::vector<float> bias(E);
    for (float& v : bias) { v = normal(rng) * 0.01F; }
    std::uint16_t* d_w = upload(w);
    float* d_bias      = upload(bias);
    void* workspace    = nullptr;
    const std::size_t ws_bytes = moe::workspace_bytes(E, I, moe::kMaxColumns, activation);
    cuda_check(cudaMalloc(&workspace, ws_bytes), "cudaMalloc");
    cuda_check(cudaMemset(workspace, 0, ws_bytes), "cudaMemset");
    const moe::Workspace ws = moe::carve_workspace(workspace, E, I, moe::kMaxColumns, activation);
    std::uint16_t *d_ref = nullptr, *d_one = nullptr;
    cuda_check(cudaMalloc(&d_ref, std::size_t(K) * H * 2), "cudaMalloc");
    cuda_check(cudaMalloc(&d_one, std::size_t(K) * H * 2), "cudaMalloc");
    int differ = 0, routing_differ = 0, graph_differ = 0, stall_differ = 0;
    // Column 3 at 8x the scale, column 4 all zeros, column 5 one NaN element (every logit NaN: the selection must still
    // fill its slots, with experts 0-7 in order, and both paths agree).
    constexpr int kColumns = 6;
    int nan_routing_wrong = 0;
    for (int column = 0; column < kColumns; ++column) {
        std::vector<float> xf(H);
        std::vector<std::uint16_t> x(H);
        for (int k = 0; k < H; ++k) {
            x[k]  = column == 4 ? std::uint16_t{0} : canon::f32_to_bf16_rn(normal(rng) * (column == 3 ? 8.0F : 1.0F));
            xf[k] = canon::bf16_to_f32(x[k]);
        }
        if (column == 5) {
            x[100]  = 0x7FC0U;
            xf[100] = canon::bf16_to_f32(x[100]);
        }
        const QuantRow q     = a8 ? quantize_a8(xf) : quantize_a4(xf);
        const LimbRow l      = encode_a16(xf);
        std::uint16_t* d_x   = upload(x);
        std::uint8_t* d_rows = upload(a16 ? l.bytes : q.bytes);
        float* d_amax        = upload(std::vector<float>{q.amax});
        std::int32_t* d_emax = upload(std::vector<std::int32_t>{l.emax});
        moe::Activations rows{d_rows, d_amax, 1};
        rows.core = core;
        if (a16) {
            rows.activation = moe::ExpertActivation::A16;
            rows.emax       = d_emax;
        }
        std::uint8_t* d_a8 = nullptr;
        if (a8x) { // the column's A8 row by the production encoder
            cuda_check(cudaMalloc(&d_a8, moe::kA8XRowBytes), "cudaMalloc");
            moe::moe_encode_a8(d_x, 1, d_a8, nullptr);
            rows.activation = moe::ExpertActivation::A8;
            rows.rows       = d_a8;
            rows.amax       = nullptr;
        }
        const auto separate = [&](cudaStream_t s) {
            moe::moe_route(d_x, 1, d_w, d_bias, E, H, 8.0F, ws, s);
            moe::moe_experts(rows, moe::routed_routing(ws), bank, ws, d_ref, s);
        };
        const auto single = [&](cudaStream_t s) { moe::moe_single(d_x, rows, d_w, d_bias, 8.0F, bank, ws, d_one, s); };
        separate(nullptr);
        cuda_check(cudaDeviceSynchronize(), "separate");
        const auto ref        = download(d_ref, std::size_t(K) * H);
        const auto ref_routed = download(ws.routed, K);
        const auto ref_gamma  = download(ws.gamma, K);
        if (column == 5) {
            for (int j = 0; j < K; ++j) { nan_routing_wrong += ref_routed[j] != j ? 1 : 0; }
        }
        cuda_check(cudaMemset(ws.routed, 0xFF, K * 4), "clear");
        single(nullptr);
        cuda_check(cudaDeviceSynchronize(), "single");
        const auto one = download(d_one, std::size_t(K) * H);
        for (std::size_t i = 0; i < one.size(); ++i) { differ += one[i] != ref[i] ? 1 : 0; }
        // Bitwise (the NaN column's weights are NaN).
        const auto gamma_one = download(ws.gamma, K);
        routing_differ += download(ws.routed, K) != ref_routed ||
                                  std::memcmp(gamma_one.data(), ref_gamma.data(), K * sizeof(float)) != 0
                              ? 1
                              : 0;
        // A graph: single, separate, single (each path after the other), replayed three times.
        cudaStream_t stream;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        single(stream);
        separate(stream);
        single(stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            cuda_check(cudaMemsetAsync(d_one, 0xFF, std::size_t(K) * H * 2, stream), "clear");
            cuda_check(cudaMemsetAsync(d_ref, 0xFF, std::size_t(K) * H * 2, stream), "clear");
            cuda_check(cudaGraphLaunch(exec, stream), "replay");
            cuda_check(cudaStreamSynchronize(stream), "replay");
            graph_differ += download(d_one, std::size_t(K) * H) != ref || download(d_ref, std::size_t(K) * H) != ref ? 1 : 0;
        }
        cuda_check(cudaGraphExecDestroy(exec), "destroy");
        cuda_check(cudaGraphDestroy(graph), "destroy");
        // After a stalled call (its CTA that gave up never counted itself): moe_clear_counters puts the counters back
        // at zero, so the next call passes no wait early, gives the separate path's outputs and leaves them at zero.
        moe::testing::leave_stalled_single_counters(ws, stream);
        moe::moe_clear_counters(ws, stream);
        cuda_check(cudaMemsetAsync(d_one, 0xFF, std::size_t(K) * H * 2, stream), "clear");
        single(stream);
        cuda_check(cudaStreamSynchronize(stream), "after a stall");
        const auto counters = download(ws.single, 2 + K);
        stall_differ += download(d_one, std::size_t(K) * H) != ref ||
                                std::any_of(counters.begin(), counters.end(), [](std::uint32_t c) { return c != 0U; })
                            ? 1
                            : 0;
        cuda_check(cudaStreamDestroy(stream), "stream");
        for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_rows), static_cast<void*>(d_amax),
                        static_cast<void*>(d_emax), static_cast<void*>(d_a8)}) {
            cuda_check(cudaFree(p), "cudaFree");
        }
    }
    const bool error_clear = download(ws.error, 1)[0] == 0U;
    std::printf("%s single-column launch (%d columns): %d of %d outputs differ from moe_route + moe_experts, routing "
                "differs in %d columns, %d graph replays (single, separate, single) differ, error word %s; NaN column's "
                "slots not experts 0-7: %d\n",
                name, kColumns, differ, kColumns * K * H, routing_differ, graph_differ, error_clear ? "clear" : "SET",
                nan_routing_wrong);
    check(differ == 0 && routing_differ == 0, "the single-column launch differs from moe_route + moe_experts");
    check(nan_routing_wrong == 0, "a NaN column's selection did not fill its slots with experts 0-7 in order");
    check(graph_differ == 0, "a graph interleaving the single-column launch and the separate kernels differs");
    check(stall_differ == 0, "after a stalled call and moe_clear_counters, the next call differs or leaves counts");
    check(error_clear, "the single-column launch set the error word");
    for (void* p : {static_cast<void*>(db.records), static_cast<void*>(db.amax), static_cast<void*>(db.den),
                    static_cast<void*>(d_w), static_cast<void*>(d_bias), static_cast<void*>(d_ref),
                    static_cast<void*>(d_one), workspace}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
}

// ---------------------------------------------------------------------------- the native core's operand mapping

// A one-hot case of a4_native_step (one warp): the four units hold one nonzero weight code `wcode` at (row r, K index
// kw) and the eight columns one nonzero code `xcode` in column col at K index kx. Scale words: the mapping cases give
// row rr of unit u the word 0x20 + rr + 16 u and column c's block b 0x28 + c + 8 b (all distinct, all finite); the
// sweep gives the hit's row and column the words sw and sx, every other 0x38.
struct OneHot {
    int r = 0, kw = 0, kx = 0, col = 0;
    unsigned wcode = 0, xcode = 0;
    bool sweep = false;
    unsigned sw = 0, sx = 0;
};

__host__ __device__ unsigned onehot_wscale(const OneHot& o, int row, int u) {
    return o.sweep ? (row == o.r && u == o.kw / 16 ? o.sw : 0x38U) : 0x20U + static_cast<unsigned>(row) + 16U * static_cast<unsigned>(u);
}
__host__ __device__ unsigned onehot_xscale(const OneHot& o, int column, int b) {
    return o.sweep ? (column == o.col && b == o.kx / 16 ? o.sx : 0x38U)
                   : 0x28U + static_cast<unsigned>(column) + 8U * static_cast<unsigned>(b);
}

__device__ void onehot_run(const OneHot& o, std::uint8_t* units, std::uint8_t (*cols)[36], float (&c)[4]) {
    const int lane = static_cast<int>(threadIdx.x) % 32, g = lane / 4, t = lane % 4;
    for (int i = lane; i < 4 * 144; i += 32) {
        const int u = i / 144, off = i % 144;
        units[i] = off < 128 ? std::uint8_t{0} : static_cast<std::uint8_t>(onehot_wscale(o, off - 128, u));
    }
    for (int i = lane; i < 8 * 36; i += 32) {
        const int column = i / 36, off = i % 36;
        cols[column][off] = off < 32 ? std::uint8_t{0} : static_cast<std::uint8_t>(onehot_xscale(o, column, off - 32));
    }
    __syncwarp();
    if (lane == 0) {
        // The record: element e of row r in byte 32 (e / 4) + 4 (r % 8) + e % 4 of its block's unit, rows 0-7 in the
        // low nibbles (docs/maintainer/storage-layouts.md §6); a column: element e of block b in byte 8 b + e / 2.
        const int u = o.kw / 16, e = o.kw % 16;
        std::uint8_t& w = units[u * 144 + 32 * (e / 4) + 4 * (o.r % 8) + e % 4];
        w = static_cast<std::uint8_t>(o.r < 8 ? (w & 0xF0U) | o.wcode : (w & 0x0FU) | (o.wcode << 4));
        const int b = o.kx / 16, ex = o.kx % 16;
        std::uint8_t& x = cols[o.col][8 * b + ex / 2];
        x = static_cast<std::uint8_t>((ex & 1) != 0 ? (x & 0x0FU) | (o.xcode << 4) : (x & 0xF0U) | o.xcode);
    }
    __syncwarp();
    for (int i = lane; i < 64; i += 32) { // the columns' code words in the core's slot order
        auto* word = reinterpret_cast<std::uint32_t*>(cols[i / 8]) + i % 8;
        *word      = infernix::ops::a4_native_order(*word);
    }
    __syncwarp();
    for (float& v : c) { v = 0.0F; }
    infernix::ops::a4_native_step(units, cols[g], *reinterpret_cast<const std::uint32_t*>(cols[g] + 32), g, t, c);
}

__device__ float onehot_want(const OneHot& o, int row, int column, float codes) {
    const bool hit = row == o.r && column == o.col && o.kw == o.kx;
    return hit ? canon::mul_rn(canon::mul_rn(codes, canon::e4m3_value(onehot_wscale(o, o.r, o.kw / 16))),
                               canon::e4m3_value(onehot_xscale(o, o.col, o.kx / 16)))
               : 0.0F;
}

// Mapping: case id = r + 16 (kw + 64 (kx + 64 col)), weight 1.5 (code 3), x 3.0 (code 5).
__global__ void onehot_map_kernel(int* fails, int* first) {
    __shared__ __align__(16) std::uint8_t units[8][4 * 144];
    __shared__ __align__(16) std::uint8_t cols[8][8][36];
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32, g = lane / 4, t = lane % 4;
    const int id = static_cast<int>(blockIdx.x) * 8 + warp;
    OneHot o;
    o.r = id & 15, o.kw = (id >> 4) & 63, o.kx = (id >> 10) & 63, o.col = (id >> 16) & 7;
    o.wcode = 0x3U, o.xcode = 0x5U;
    float c[4];
    onehot_run(o, units[warp], cols[warp], c);
    bool bad = false;
    for (int i = 0; i < 4; ++i) { bad = bad || c[i] != onehot_want(o, g + (i < 2 ? 0 : 8), 2 * t + (i & 1), 4.5F); }
    if (bad && atomicAdd(fails, 1) == 0) { *first = id; }
}

// Scale words: case id = sw + 256 sx at (row 5, K 37, column 3), both codes 1.0 (code 2).
__global__ void onehot_scale_kernel(std::uint8_t* bad) {
    __shared__ __align__(16) std::uint8_t units[8][4 * 144];
    __shared__ __align__(16) std::uint8_t cols[8][8][36];
    const int warp = static_cast<int>(threadIdx.x) / 32, lane = static_cast<int>(threadIdx.x) % 32, g = lane / 4, t = lane % 4;
    const int id = static_cast<int>(blockIdx.x) * 8 + warp;
    OneHot o;
    o.r = 5, o.kw = 37, o.kx = 37, o.col = 3, o.wcode = 0x2U, o.xcode = 0x2U, o.sweep = true;
    o.sw = static_cast<unsigned>(id & 255), o.sx = static_cast<unsigned>(id >> 8);
    float c[4];
    onehot_run(o, units[warp], cols[warp], c);
    bool wrong = false;
    for (int i = 0; i < 4; ++i) { wrong = wrong || c[i] != onehot_want(o, g + (i < 2 ? 0 : 8), 2 * t + (i & 1), 1.0F); }
    if (wrong) { bad[id] = 1; }
}

void native_core_case() {
    int* d_counts = nullptr;
    std::uint8_t* d_bad = nullptr;
    cuda_check(cudaMalloc(&d_counts, 2 * sizeof(int)), "cudaMalloc");
    cuda_check(cudaMalloc(&d_bad, 65536), "cudaMalloc");
    cuda_check(cudaMemset(d_counts, 0, 2 * sizeof(int)), "cudaMemset");
    cuda_check(cudaMemset(d_bad, 0, 65536), "cudaMemset");
    constexpr int kMapCases = 16 * 64 * 64 * 8;
    onehot_map_kernel<<<kMapCases / 8, 256>>>(d_counts, d_counts + 1);
    onehot_scale_kernel<<<65536 / 8, 256>>>(d_bad);
    cuda_check(cudaGetLastError(), "one-hot launches");
    cuda_check(cudaDeviceSynchronize(), "one-hot kernels");
    int counts[2] = {};
    std::vector<std::uint8_t> bad(65536);
    cuda_check(cudaMemcpy(counts, d_counts, sizeof(counts), cudaMemcpyDeviceToHost), "download");
    cuda_check(cudaMemcpy(bad.data(), d_bad, bad.size(), cudaMemcpyDeviceToHost), "download");
    std::printf("native core operand mapping: %d one-hot cases (16 rows x 64 weight K x 64 x K x 8 columns), %d differ%s\n",
                kMapCases, counts[0], counts[0] == 0 ? "" : " (first: see id)");
    if (counts[0] != 0) {
        const int id = counts[1];
        std::printf("  first failing case: row %d, weight K %d, x K %d, column %d\n", id & 15, (id >> 4) & 63,
                    (id >> 10) & 63, (id >> 16) & 7);
    }
    check(counts[0] == 0, "the native core's operand mapping");
    int finite_bad = 0, other_bad = 0, other_total = 0;
    std::vector<int> bad_words;
    for (int id = 0; id < 65536; ++id) {
        const int sw = id & 255, sx = id >> 8;
        const bool finite = sw <= 0x7E && sx <= 0x7E;
        if (finite) {
            finite_bad += bad[static_cast<std::size_t>(id)];
        } else {
            ++other_total;
            other_bad += bad[static_cast<std::size_t>(id)];
        }
    }
    for (int w = 0; w < 256; ++w) { // a word that decodes differently against 0x38 on the other side
        if (bad[static_cast<std::size_t>(w + 256 * 0x38)] != 0) { bad_words.push_back(w); }
    }
    std::printf("native core scale words: %d of %d pairs of 0x00-0x7E differ from canon::e4m3_value (criterion 0); "
                "%d of %d pairs with 0x7F or the sign bit differ (reported; %zu weight words against 0x38 differ:",
                finite_bad, 127 * 127, other_bad, other_total, bad_words.size());
    for (std::size_t i = 0; i < bad_words.size() && i < 12; ++i) { std::printf(" 0x%02X", bad_words[i]); }
    std::printf("%s)\n", bad_words.size() > 12 ? " ..." : "");
    check(finite_bad == 0, "a finite scale word decodes differently in the native core");
    cuda_check(cudaFree(d_counts), "cudaFree");
    cuda_check(cudaFree(d_bad), "cudaFree");
}

} // namespace

INFERNIX_OP_TEST(w4a4_sparse_moe) {
    native_core_case();
    router_case();
    single_case(false);
    single_case(false, false, moe::Nvfp4Core::Exact);
    single_case(true);
    single_case(false, true);
    single_case(false, false, moe::Nvfp4Core::Native, true);
    bank_case(false);
    bank_case(false, moe::Nvfp4Core::Exact);
    bank_case(true);
    a16_case();
    a8_case();
    mx_a16_case();
    prompt_case(false);
    prompt_case(true);
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
