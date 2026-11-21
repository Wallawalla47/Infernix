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
//   - call state: a call over the same workspace after one whose h amax are 64 times larger gives the same bits.

#include "infernix/ops/w4a4_sparse_moe.h"
#include "ops/common/canonical_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
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

Result run(const HostBank& b, const DeviceBank& db, const Call& call, void* workspace,
           int capacity = moe::kMaxColumns) {
    std::vector<std::uint8_t> rows;
    std::vector<float> amax;
    for (const auto& x : call.x) {
        const QuantRow q = b.a8 ? quantize_a8(x) : quantize_a4(x);
        rows.insert(rows.end(), q.bytes.begin(), q.bytes.end());
        amax.push_back(q.amax);
    }
    std::uint8_t* d_rows = upload(rows);
    float* d_amax        = upload(amax);
    std::int32_t* d_exp  = upload(call.experts);
    float* d_gamma       = upload(call.gamma);
    std::uint16_t* d_out = nullptr;
    const std::size_t entries = std::size_t(call.columns) * K;
    cuda_check(cudaMalloc(&d_out, entries * H * 2), "cudaMalloc");
    const moe::Workspace ws = moe::carve_workspace(workspace, E, I, capacity);
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
    moe::moe_experts(moe::Activations{d_rows, d_amax, call.columns}, routing, bank, ws, d_out, nullptr);
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
        moe::moe_experts(moe::Activations{d_rows, d_amax, call.columns}, routing, bank, ws, d_out, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            cuda_check(cudaMemsetAsync(d_out, 0xFF, entries * H * 2, stream), "clear");
            cuda_check(cudaGraphLaunch(exec, stream), "replay");
            cuda_check(cudaStreamSynchronize(stream), "replay");
            check(download(d_out, entries * H) == r.out, "a graph replay (programmatic down pass) differs from the eager call");
        }
        cuda_check(cudaGraphExecDestroy(exec), "destroy");
        cuda_check(cudaGraphDestroy(graph), "destroy");
        cuda_check(cudaStreamDestroy(stream), "stream");
        check(download(ws.error, 1)[0] == 0U, "the call's error word is set");
    }
    r.h      = download(ws.h, entries * I);
    r.h_rows = download(ws.h_rows, entries * moe::row_bytes(bank.format, I));
    r.h_amax = download(ws.h_amax, entries);
    for (void* p : {static_cast<void*>(d_rows), static_cast<void*>(d_amax), static_cast<void*>(d_exp),
                    static_cast<void*>(d_gamma), static_cast<void*>(d_out)}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
    return r;
}

void bank_case(bool a8) {
    const char* name = a8 ? "MXFP8" : "NVFP4";
    const HostBank b = make_bank(a8, a8 ? 81U : 41U);
    DeviceBank db{upload(b.records), upload(b.amax), upload(std::vector<float>{b.den[0], b.den[1]})};
    void* workspace = nullptr;
    cuda_check(cudaMalloc(&workspace, moe::workspace_bytes(E, I)), "cudaMalloc");
    cuda_check(cudaMemset(workspace, 0, moe::workspace_bytes(E, I)), "cudaMemset");
    const int row_bytes = moe::row_bytes(a8 ? moe::ExpertFormat::Mxfp8 : moe::ExpertFormat::Nvfp4FourOverSix, I);

    // A call whose h amax are 64 times larger first: the next call must not see its state.
    (void)run(b, db, make_call(32, 7, 64.0F), workspace);
    const Call call    = make_call(32, 11, 1.0F);
    const Result got   = run(b, db, call, workspace);
    const Result again = run(b, db, call, workspace);
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
        std::printf("%s: %d outputs differ from the exact pinned arithmetic\n", name, exact_differ);
        check(exact_differ == 0, "NVFP4 outputs differ from the exact pinned arithmetic");
    }

    // Column invariance: each of the first 8 columns alone.
    int invariance_differ = 0;
    for (int t = 0; t < 8; ++t) {
        Call one;
        one.columns = 1;
        one.x       = {call.x[t]};
        one.experts.assign(call.experts.begin() + t * K, call.experts.begin() + (t + 1) * K);
        one.gamma.assign(call.gamma.begin() + t * K, call.gamma.begin() + (t + 1) * K);
        const Result alone = run(b, db, one, workspace);
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
        const Result all  = run(b, db, wide, wide_ws, kWide);
        int wide_differ   = 0;
        for (const int t : {0, 7, 8, 63, 150, 255, 256, 299}) {
            Call one;
            one.columns = 1;
            one.x       = {wide.x[t]};
            one.experts.assign(wide.experts.begin() + t * K, wide.experts.begin() + (t + 1) * K);
            one.gamma.assign(wide.gamma.begin() + t * K, wide.gamma.begin() + (t + 1) * K);
            const Result alone = run(b, db, one, workspace);
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
    for (void* p : {static_cast<void*>(d_w), static_cast<void*>(d_x), static_cast<void*>(d_bias), workspace}) {
        cuda_check(cudaFree(p), "cudaFree");
    }
}

} // namespace

INFERNIX_OP_TEST(w4a4_sparse_moe) {
    router_case();
    bank_case(false);
    bank_case(true);
    prompt_case(false);
    prompt_case(true);
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
