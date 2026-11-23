// Inkling 2's visit-chain kernels (include/infernix/ops/inkling2_visit.h) at H = 2304.
//
// Inputs: random BF16 rows of realistic scale (unit-RMS residual sources, small expert outputs), FP32 mix queries of
// standard deviation 2 (so softmax weights spread over the sources), conv taps and norms as the checkpoint's ranges.
// One call holds 22 columns: lane 0 at positions 0-3 (first position 0: history below it is 0), lane 1 at 6-21 (first
// 5: position 6 reads 5 from the ring and 4, 3 as 0), a padding column (lane -1) and lane 2 at 140 (history all from
// the ring). Every ring slot starts random, so a read the rule forbids shows up.
//
// Oracles:
//   - the pinned arithmetic: a host emulation through the same visit_math.h / canonical_math.h definitions; every
//     output, ring slot and activation row must equal it bit for bit (the reproducible route's bits);
//   - FP64, stage by stage from the kernel's own stored upstream values (a BF16 rounding inside a kernel, such as the
//     conv output m, is a semantic boundary: its flip is allowed for): BF16 outputs within half a BF16 ulp of the
//     FP64 value plus 2^-16 of the row's largest magnitude (plus one ulp of an inner BF16 boundary where one lies
//     between), FP32 outputs within 2^-18 of the row's largest magnitude;
//   - the experts' rows: exact, the canonical four-over-six / MXFP8 encoders over the kernel's own BF16 x;
//   - decode against verify: the call's columns run one at a time (each a T = 1 call that advances the ring) give the
//     call's bits, ring included.

#include "infernix/ops/inkling2_visit.h"
#include "ops/common/canonical_math.h"
#include "ops/ggml/ggml_a8.h"
#include "ops/inkling2/visit_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace canon = infernix::ops::canon;
namespace ik    = infernix::ops::inkling2;
namespace moe   = infernix::ops::w4a4_moe;
using infernix::test::cuda_check;

namespace {

constexpr int H = ik::kHidden, K = moe::kTopK, R = ik::kRingPositions;
constexpr int kLanes = 3;
int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class T>
T* dalloc(std::size_t n) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
    return p;
}

template <class T>
T* upload(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}

template <class T>
void upload_to(T* p, const std::vector<T>& v) {
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

template <class T>
std::vector<T> download(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

float bf(float x) { return canon::bf16_to_f32(canon::f32_to_bf16_rn(x)); }
float f(std::uint16_t v) { return canon::bf16_to_f32(v); }
std::uint16_t b16(float x) { return canon::f32_to_bf16_rn(x); }

double ulp_bf16(double r) {
    const double a = std::fabs(r);
    if (a < 0x1p-126) { return 0x1p-133; }
    return std::ldexp(1.0, std::ilogb(a) - 7);
}

// ---------------------------------------------------------------------------- the call's columns

struct Columns {
    std::vector<int> lane, pos;
    std::vector<int> first = {0, 5, 100};
    [[nodiscard]] int size() const { return static_cast<int>(lane.size()); }
};

Columns the_call() {
    Columns c;
    for (int p = 0; p < 4; ++p) { c.lane.push_back(0), c.pos.push_back(p); }
    for (int p = 6; p < 22; ++p) { c.lane.push_back(1), c.pos.push_back(p); }
    c.lane.push_back(-1), c.pos.push_back(0);
    c.lane.push_back(2), c.pos.push_back(140);
    return c;
}

struct DeviceMap {
    int* lane = nullptr;
    int* pos  = nullptr;
    int* first = nullptr;
    explicit DeviceMap(const Columns& c)
        : lane(upload(c.lane)), pos(upload(c.pos)), first(upload(c.first)), columns(c.size()) {}
    ~DeviceMap() {
        cudaFree(lane);
        cudaFree(pos);
        cudaFree(first);
    }
    int columns = 0;
    [[nodiscard]] ik::ColumnMap map() const { return {lane, pos, first, columns}; }
};

// ---------------------------------------------------------------------------- random data

struct Rng {
    std::mt19937 gen;
    explicit Rng(std::uint32_t seed) : gen(seed) {}
    std::vector<std::uint16_t> bf16(std::size_t n, float sd, float mean = 0.0F) {
        std::normal_distribution<float> d(mean, sd);
        std::vector<std::uint16_t> v(n);
        for (auto& x : v) { x = b16(d(gen)); }
        return v;
    }
    std::vector<float> f32(std::size_t n, float sd) {
        std::normal_distribution<float> d(0.0F, sd);
        std::vector<float> v(n);
        for (auto& x : v) { x = d(gen); }
        return v;
    }
};

const float* rowp(const std::vector<float>& v, int t) { return v.data() + static_cast<std::size_t>(t) * H; }

// ---------------------------------------------------------------------------- criteria

struct Tally {
    const char* name;
    double worst = 0;
    long bad     = 0;
    long differ  = 0; // from the pinned emulation
    void report() const {
        std::printf("%-34s worst %.3f of the FP64 criterion; %ld beyond it; %ld differ from the pinned arithmetic\n",
                    name, worst, bad, differ);
        check(bad == 0, name);
        check(differ == 0, name);
    }
};

double row_max(const double* r) {
    double m = 0;
    for (int i = 0; i < H; ++i) { m = std::max(m, std::fabs(r[i])); }
    return m;
}

// BF16 y against FP64 r (the row's values), plus an inner boundary's ulp per element (inner may be null).
void bf16_criterion(Tally& tally, const std::uint16_t* y, const double* r, const double* inner = nullptr) {
    const double slack = 0x1p-16 * row_max(r);
    for (int i = 0; i < H; ++i) {
        const double bound = 0.5 * ulp_bf16(r[i]) + slack + (inner != nullptr ? ulp_bf16(inner[i]) : 0.0);
        const double ratio = std::fabs(f(y[i]) - r[i]) / bound;
        tally.worst        = std::max(tally.worst, ratio);
        tally.bad += ratio > 1.0 ? 1 : 0;
    }
}

void f32_criterion(Tally& tally, const float* y, const double* r) {
    const double bound = 0x1p-18 * std::max(row_max(r), 1e-30);
    for (int i = 0; i < H; ++i) {
        const double ratio = std::fabs(y[i] - r[i]) / bound;
        tally.worst        = std::max(tally.worst, ratio);
        tally.bad += ratio > 1.0 ? 1 : 0;
    }
}

template <class T>
void exact(Tally& tally, const T* a, const T* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) { tally.differ += std::memcmp(a + i, b + i, sizeof(T)) != 0 ? 1 : 0; }
}

// ---------------------------------------------------------------------------- FP64 pieces

void rms64(const double* x, double* out) {
    double s = 0;
    for (int i = 0; i < H; ++i) { s += x[i] * x[i]; }
    const double r = 1.0 / std::sqrt(s / H + 1e-6);
    for (int i = 0; i < H; ++i) { out[i] = x[i] * r; }
}

void mix64(const float* q, const std::vector<const double*>& sources, double* out) {
    std::vector<std::vector<double>> n(sources.size(), std::vector<double>(H));
    std::vector<double> score(sources.size());
    for (std::size_t s = 0; s < sources.size(); ++s) {
        rms64(sources[s], n[s].data());
        double d = 0;
        for (int i = 0; i < H; ++i) { d += double{q[i]} * n[s][i]; }
        score[s] = d / 48.0;
    }
    const double m = *std::max_element(score.begin(), score.end());
    double den     = 0;
    for (auto& s : score) { den += (s = std::exp(s - m)); }
    for (int i = 0; i < H; ++i) {
        double acc = 0;
        for (std::size_t s = 0; s < sources.size(); ++s) { acc += score[s] / den * n[s][i]; }
        out[i] = acc;
    }
}

void norm64(const double* x, const std::uint16_t* w, double* out) {
    rms64(x, out);
    for (int i = 0; i < H; ++i) { out[i] *= f(w[i]); }
}

std::vector<double> d64(const float* x) { return std::vector<double>(x, x + H); }

// ---------------------------------------------------------------------------- conv history (host)

// A host ring [lanes][R][H] of BF16 words.
using HostRing = std::vector<std::uint16_t>;

std::uint16_t* ring_at(HostRing& ring, int lane, int pos) {
    return ring.data() + (static_cast<std::size_t>(lane) * R + (pos % R)) * H;
}

enum class Src { Zero, Call, Ring };
Src history(const Columns& c, int t, int k) {
    const int lane = c.lane[t];
    if (lane < 0 || c.pos[t] - k < c.first[lane]) { return Src::Zero; }
    if (t - k >= 0 && c.lane[t - k] == lane) { return Src::Call; }
    return Src::Ring;
}

// The 4 conv inputs of column t (lag 0 current), given each column's raw input (BF16 words [T][H]) and the
// pre-call ring.
std::vector<std::vector<float>> conv_inputs(const Columns& c, int t, const std::vector<std::vector<float>>& raw,
                                            HostRing& ring_before) {
    std::vector<std::vector<float>> x(4, std::vector<float>(H, 0.0F));
    x[0] = raw[t];
    for (int k = 1; k < 4; ++k) {
        switch (history(c, t, k)) {
        case Src::Zero: break;
        case Src::Call: x[k] = raw[t - k]; break;
        case Src::Ring: {
            const std::uint16_t* r = ring_at(ring_before, c.lane[t], c.pos[t] - k);
            for (int i = 0; i < H; ++i) { x[k][i] = f(r[i]); }
            break;
        }
        }
    }
    return x;
}

// m = bf16(conv_static) per channel, and the FP64 conv value.
void conv_host(const std::vector<std::uint16_t>& taps, const std::vector<std::vector<float>>& x, float* m, double* m64) {
    for (int i = 0; i < H; ++i) {
        const float w[4]  = {f(taps[i]), f(taps[H + i]), f(taps[2 * H + i]), f(taps[3 * H + i])};
        const float xv[4] = {x[0][i], x[1][i], x[2][i], x[3][i]};
        m[i]              = bf(ik::conv_static(w, xv));
        double acc        = 0;
        for (int lag = 0; lag < 4; ++lag) { acc += double{w[lag]} * xv[lag]; }
        m64[i] = (acc + xv[0]) / std::sqrt(5.0);
    }
}

// ---------------------------------------------------------------------------- the experts' rows (host)

// The A16 limb row of a BF16 row (--a16-activations' encoding, Flash-Next design §16.2.1): the exponent
// canon::a16_column_exponent, X_j = a16_value, per 64-element tile 64 H1 = X >> 14, 64 L1 = (X >> 3) & 255 and
// 64 H0 | L0 << 4 bytes.
std::vector<std::uint8_t> limbs_host(const std::uint16_t* x, std::int32_t* emax_out) {
    std::vector<std::uint8_t> row(moe::kA16XRowBytes, 0);
    const int e = canon::a16_column_exponent(x, H);
    *emax_out   = e;
    for (int j = 0; j < H; ++j) {
        const std::int32_t v = e == canon::kA16NonFinite ? 0 : canon::a16_value(x[j], e);
        std::uint8_t* at     = row.data() + (j / 64) * 192 + j % 64;
        at[0]                = static_cast<std::uint8_t>(v >> 14);
        at[64]               = static_cast<std::uint8_t>((v >> 3) & 255);
        at[128]              = static_cast<std::uint8_t>(((v >> 11) & 7) | ((v & 7) << 4));
    }
    return row;
}

// The A8 row of a BF16 row (--a8-activations: ops/ggml/ggml_a8.h's codec by its definition): per 32-slice the scale of
// its largest magnitude and the codes, 2,304 codes then 72 FP32 scales.
std::vector<std::uint8_t> a8_host(const std::uint16_t* x) {
    namespace a8 = infernix::ggml::a8;
    std::vector<std::uint8_t> row(moe::kA8XRowBytes, 0);
    for (int s = 0; s < H / 32; ++s) {
        unsigned key = 0;
        for (int j = 0; j < 32; ++j) { key = std::max(key, static_cast<unsigned>(x[32 * s + j] & 0x7FFFU)); }
        const a8::SliceScale sc = a8::slice_scale(key);
        for (int j = 0; j < 32; ++j) {
            row[32 * s + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(a8::code(x[32 * s + j], sc.inv)));
        }
        std::memcpy(row.data() + H + 4 * s, &sc.d, 4);
    }
    return row;
}

std::vector<std::uint8_t> quantize_host(const std::uint16_t* x, moe::ExpertFormat format, float* amax_out) {
    std::vector<std::uint8_t> row(moe::row_bytes(format, H), 0);
    if (format == moe::ExpertFormat::Nvfp4FourOverSix) {
        float amax = 0;
        for (int i = 0; i < H; ++i) { amax = std::max(amax, std::fabs(f(x[i]))); }
        *amax_out = amax;
        const canon::FourOverSixRow r = canon::four_over_six_row(amax);
        for (int b = 0; b < H / 16; ++b) {
            float v[16];
            for (int j = 0; j < 16; ++j) { v[j] = f(x[16 * b + j]); }
            const canon::FourOverSixBlock q = canon::four_over_six_block(v, r);
            for (int j = 0; j < 8; ++j) {
                row[8 * b + j] = static_cast<std::uint8_t>(q.code[2 * j] | (q.code[2 * j + 1] << 4));
            }
            row[H / 2 + b] = q.scale_word;
        }
    } else {
        for (int g = 0; g < H / 32; ++g) {
            float v[32];
            for (int j = 0; j < 32; ++j) { v[j] = f(x[32 * g + j]); }
            const canon::Mxfp8Group q = canon::mxfp8_group(v);
            for (int j = 0; j < 32; ++j) { row[32 * g + j] = q.code[j]; }
            row[H + g] = q.exponent;
        }
    }
    return row;
}

// ---------------------------------------------------------------------------- embed

void embed_case() {
    Rng rng(11);
    constexpr int V = 997, T = 24;
    const auto table = rng.bf16(static_cast<std::size_t>(V) * H, 0.02F);
    const auto norm  = rng.bf16(H, 0.3F, 1.0F);
    std::vector<int> tokens(T);
    std::uniform_int_distribution<int> pick(0, V - 1);
    for (auto& t : tokens) { t = pick(rng.gen); }
    auto* d_table = upload(table);
    auto* d_norm  = upload(norm);
    auto* d_tok   = upload(tokens);
    auto* d_out   = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * H);
    ik::embed(d_tok, T, d_table, d_norm, d_out, nullptr);
    cuda_check(cudaDeviceSynchronize(), "embed");
    const auto out = download(d_out, static_cast<std::size_t>(T) * H);
    Tally tally{"embed"};
    for (int t = 0; t < T; ++t) {
        std::vector<float> x(H), y(H);
        for (int i = 0; i < H; ++i) { x[i] = f(table[std::size_t(tokens[t]) * H + i]); }
        ik::norm_reference(x.data(), norm.data(), y.data());
        std::vector<std::uint16_t> want(H);
        for (int i = 0; i < H; ++i) { want[i] = b16(y[i]); }
        exact(tally, out.data() + std::size_t(t) * H, want.data(), H);
        std::vector<double> r(H), x64 = d64(x.data());
        norm64(x64.data(), norm.data(), r.data());
        bf16_criterion(tally, out.data() + std::size_t(t) * H, r.data());
    }
    tally.report();
    for (void* p : {static_cast<void*>(d_table), static_cast<void*>(d_norm), static_cast<void*>(d_tok),
                    static_cast<void*>(d_out)}) {
        cudaFree(p);
    }
}

// ---------------------------------------------------------------------------- visit_entry

struct EntryConfig {
    const char* name;
    bool tail;     // the previous visit's MLP tail
    bool restart;  // with a tail, the visit's input given separately (a pass restart)
    bool layer0;   // no mix
    int nq;        // 1 or 3
    int blocks;    // completed blocks
    bool messages;
};

// The data of one visit_entry problem over all of the call's columns.
struct EntryData {
    std::vector<std::uint16_t> experts, conv, input, norms[3], msrc[2];
    std::vector<std::vector<std::uint16_t>> blocks;
    std::vector<float> cand, queries, mqueries;
    HostRing ring;
};

EntryData entry_data(const EntryConfig& cfg, int T, std::uint32_t seed) {
    Rng rng(seed);
    EntryData d;
    d.experts = rng.bf16(static_cast<std::size_t>(T) * K * H, 0.05F);
    d.cand    = rng.f32(static_cast<std::size_t>(T) * H, 1.0F);
    for (auto& v : d.cand) { v = bf(v) + bf(v * 0x1p-9F); } // FP32 sums of two BF16 values, as cand is
    d.conv    = rng.bf16(std::size_t{4} * H, 0.4F);
    d.input   = rng.bf16(static_cast<std::size_t>(T) * H, 1.0F);
    for (int i = 0; i < 3; ++i) { d.norms[i] = rng.bf16(H, 0.2F, 1.0F); }
    for (int s = 0; s < cfg.blocks; ++s) { d.blocks.push_back(rng.bf16(static_cast<std::size_t>(T) * H, 1.0F + 0.5F * s)); }
    d.queries  = rng.f32(static_cast<std::size_t>(cfg.nq) * H, 2.0F);
    d.mqueries = rng.f32(std::size_t{4} * H, 2.0F);
    for (auto& m : d.msrc) { m = rng.bf16(static_cast<std::size_t>(T) * H, 1.5F); }
    d.ring = rng.bf16(static_cast<std::size_t>(kLanes) * R * H, 0.05F);
    return d;
}

struct EntryResult {
    std::vector<std::uint16_t> tail_out, x[3], gate_input;
    std::vector<std::uint8_t> x8[3]; // the x outputs' A8 rows (--a8-activations; non-message calls)
    std::vector<float> mixed, message;
    HostRing ring;
};

// Runs visit_entry over the columns `subset` of the call (in that order) as one call, with the device ring `d_ring`.
EntryResult run_entry(const EntryConfig& cfg, const EntryData& d, const Columns& all, const std::vector<int>& subset,
                      std::uint16_t* d_ring) {
    const int T = static_cast<int>(subset.size());
    Columns c;
    c.first = all.first;
    for (int t : subset) { c.lane.push_back(all.lane[t]), c.pos.push_back(all.pos[t]); }
    const auto pick = [&](const std::vector<std::uint16_t>& v, int width) {
        std::vector<std::uint16_t> out(static_cast<std::size_t>(T) * width);
        for (int i = 0; i < T; ++i) {
            std::copy_n(v.data() + std::size_t(subset[i]) * width, width, out.data() + std::size_t(i) * width);
        }
        return out;
    };
    std::vector<float> cand(static_cast<std::size_t>(T) * H);
    for (int i = 0; i < T; ++i) { std::copy_n(rowp(d.cand, subset[i]), H, cand.data() + std::size_t(i) * H); }
    DeviceMap map(c);
    auto* d_experts = upload(pick(d.experts, K * H));
    auto* d_cand    = upload(cand);
    auto* d_conv    = upload(d.conv);
    auto* d_input   = upload(pick(d.input, H));
    std::vector<std::uint16_t*> d_blocks;
    for (const auto& b : d.blocks) { d_blocks.push_back(upload(pick(b, H))); }
    auto* d_queries  = upload(d.queries);
    auto* d_mqueries = upload(d.mqueries);
    std::uint16_t* d_msrc[2] = {upload(pick(d.msrc[0], H)), upload(pick(d.msrc[1], H))};
    std::uint16_t* d_norms[3];
    for (int i = 0; i < 3; ++i) { d_norms[i] = upload(d.norms[i]); }
    auto* d_tail_out = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * H);
    std::uint16_t* d_x[3];
    for (auto& x : d_x) { x = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * H); }
    auto* d_mixed   = dalloc<float>(std::size_t{3} * T * H);
    auto* d_message = dalloc<float>(std::size_t{3} * T * H);
    auto* d_gate    = dalloc<std::uint16_t>(std::size_t{3} * T * 2 * H);

    ik::EntryArgs a;
    a.columns = T;
    a.map     = map.map();
    if (cfg.tail) {
        a.tail     = {d_experts, d_cand, d_conv, {d_ring, H, std::int64_t{R} * H}};
        a.tail_out = d_tail_out;
    }
    a.input = !cfg.tail || cfg.restart ? d_input : nullptr;
    if (!cfg.layer0) {
        for (int s = 0; s < cfg.blocks; ++s) { a.blocks.rows[s] = d_blocks[s]; }
        a.blocks.count = cfg.blocks;
        a.queries      = d_queries;
        a.nq           = cfg.nq;
    }
    if (cfg.messages) {
        a.message_queries    = d_mqueries;
        a.message_sources[0] = d_msrc[0];
        a.message_sources[1] = d_msrc[1];
        a.mixed              = d_mixed;
        a.message            = d_message;
        a.gate_input         = d_gate;
    } else {
        for (int i = 0; i < 3; ++i) {
            a.norms[i] = d_norms[i];
            a.x[i]     = d_x[i];
        }
    }
    std::uint8_t* d_x8[3] = {};
    if (!cfg.messages) { // the x outputs' A8 rows beside them
        a.x8_stride = moe::kA8XRowBytes / 4;
        for (int i = 0; i < 3; ++i) {
            d_x8[i] = dalloc<std::uint8_t>(static_cast<std::size_t>(T) * moe::kA8XRowBytes);
            a.x8[i] = reinterpret_cast<std::int32_t*>(d_x8[i]);
        }
    }
    ik::visit_entry(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "visit_entry");
    EntryResult r;
    r.tail_out = download(d_tail_out, static_cast<std::size_t>(T) * H);
    for (int i = 0; i < 3; ++i) { r.x[i] = download(d_x[i], static_cast<std::size_t>(T) * H); }
    for (int i = 0; i < 3; ++i) {
        if (d_x8[i] != nullptr) {
            r.x8[i] = download(d_x8[i], static_cast<std::size_t>(T) * moe::kA8XRowBytes);
            cudaFree(d_x8[i]);
        }
    }
    r.mixed      = download(d_mixed, std::size_t{3} * T * H);
    r.message    = download(d_message, std::size_t{3} * T * H);
    r.gate_input = download(d_gate, std::size_t{3} * T * 2 * H);
    r.ring       = download(d_ring, static_cast<std::size_t>(kLanes) * R * H);
    std::vector<void*> frees = {d_experts, d_cand, d_conv, d_input, d_queries, d_mqueries, d_msrc[0], d_msrc[1],
                                d_tail_out, d_mixed, d_message, d_gate};
    for (auto* p : d_blocks) { frees.push_back(p); }
    for (int i = 0; i < 3; ++i) { frees.push_back(d_norms[i]), frees.push_back(d_x[i]); }
    for (void* p : frees) { cudaFree(p); }
    return r;
}

void entry_case(const EntryConfig& cfg, std::uint32_t seed) {
    const Columns call = the_call();
    const int T        = call.size();
    const EntryData d  = entry_data(cfg, T, seed);
    auto* d_ring       = upload(d.ring);
    std::vector<int> all(T);
    for (int t = 0; t < T; ++t) { all[t] = t; }
    const EntryResult got = run_entry(cfg, d, call, all, d_ring);

    // The pinned emulation and the FP64 stages.
    Tally tally{cfg.name}, ftally{"  (FP32 outputs)"};
    HostRing ring_before = d.ring, ring_after = d.ring;
    std::vector<std::vector<float>> combined(T, std::vector<float>(H));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < H; ++i) {
            float acc = 0.0F;
            for (int j = 0; j < K; ++j) { acc = canon::add_rn(acc, f(d.experts[(std::size_t(t) * K + j) * H + i])); }
            combined[t][i] = bf(acc);
        }
    }
    for (int t = 0; t < T; ++t) {
        std::vector<float> h(H);
        if (cfg.tail) {
            const auto x = conv_inputs(call, t, combined, ring_before);
            std::vector<float> m(H);
            std::vector<double> m64(H), out64(H), sum64(H);
            conv_host(d.conv, x, m.data(), m64.data());
            std::vector<std::uint16_t> want(H);
            for (int i = 0; i < H; ++i) {
                h[i]    = bf(canon::add_rn(d.cand[std::size_t(t) * H + i], m[i]));
                want[i] = b16(h[i]);
            }
            exact(tally, got.tail_out.data() + std::size_t(t) * H, want.data(), H);
            if (call.lane[t] >= 0) {
                std::uint16_t* slot = ring_at(ring_after, call.lane[t], call.pos[t]);
                for (int i = 0; i < H; ++i) { slot[i] = b16(combined[t][i]); }
            }
            // FP64: the combine against the exact sum, then out over the kernel's own conv inputs.
            for (int i = 0; i < H; ++i) {
                double s = 0;
                for (int j = 0; j < K; ++j) { s += f(d.experts[(std::size_t(t) * K + j) * H + i]); }
                sum64[i] = s;
                out64[i] = double{d.cand[std::size_t(t) * H + i]} + bf(static_cast<float>(m64[i]));
            }
            std::vector<std::uint16_t> comb(H);
            for (int i = 0; i < H; ++i) { comb[i] = b16(combined[t][i]); }
            bf16_criterion(tally, comb.data(), sum64.data());
            bf16_criterion(tally, got.tail_out.data() + std::size_t(t) * H, out64.data(), m64.data());
        }
        if (!cfg.tail || cfg.restart) {
            for (int i = 0; i < H; ++i) { h[i] = f(d.input[std::size_t(t) * H + i]); }
        }
        // The kernel's own h (for FP64 stages) is the emulated h when the tail matched.
        std::vector<float> outs[6];
        std::vector<std::vector<float>> src;
        for (int s = 0; s < cfg.blocks; ++s) {
            std::vector<float> v(H);
            for (int i = 0; i < H; ++i) { v[i] = f(d.blocks[s][std::size_t(t) * H + i]); }
            src.push_back(v);
        }
        src.push_back(h);
        std::vector<const float*> sp;
        for (const auto& s : src) { sp.push_back(s.data()); }
        std::vector<std::vector<double>> src64;
        for (const auto& s : src) { src64.push_back(d64(s.data())); }
        std::vector<const double*> sp64;
        for (const auto& s : src64) { sp64.push_back(s.data()); }
        const int nq = cfg.layer0 ? 1 : cfg.nq;
        std::vector<std::vector<double>> out64(nq, std::vector<double>(H));
        for (int i = 0; i < nq; ++i) {
            outs[i].resize(H);
            if (cfg.layer0) {
                outs[i] = h;
                out64[i] = d64(h.data());
            } else {
                ik::mix_reference(d.queries.data() + std::size_t(i) * H, sp.data(), static_cast<int>(sp.size()),
                                  outs[i].data());
                mix64(d.queries.data() + std::size_t(i) * H, sp64, out64[i].data());
            }
        }
        if (!cfg.messages) {
            for (int i = 0; i < 3; ++i) {
                const int qi = nq == 1 ? 0 : i;
                std::vector<float> y(H);
                ik::norm_reference(outs[qi].data(), d.norms[qi].data(), y.data());
                std::vector<std::uint16_t> want(H);
                for (int j = 0; j < H; ++j) { want[j] = b16(y[j]); }
                exact(tally, got.x[i].data() + std::size_t(t) * H, want.data(), H);
                std::vector<double> r(H);
                norm64(out64[qi].data(), d.norms[qi].data(), r.data());
                bf16_criterion(tally, got.x[i].data() + std::size_t(t) * H, r.data());
            }
        } else {
            std::vector<float> ms[2] = {std::vector<float>(H), std::vector<float>(H)};
            for (int s = 0; s < 2; ++s) {
                for (int i = 0; i < H; ++i) { ms[s][i] = f(d.msrc[s][std::size_t(t) * H + i]); }
            }
            const float* msp[2] = {ms[0].data(), ms[1].data()};
            const auto ms64a = d64(ms[0].data()), ms64b = d64(ms[1].data());
            const std::vector<const double*> msp64 = {ms64a.data(), ms64b.data()};
            for (int i = 0; i < nq; ++i) {
                const float* y = got.mixed.data() + (std::size_t(i) * T + t) * H;
                exact(tally, y, outs[i].data(), H);
                f32_criterion(ftally, y, out64[i].data());
            }
            for (int k = 0; k < 3; ++k) {
                std::vector<float> m(H);
                ik::mix_reference(d.mqueries.data() + std::size_t(k) * H, msp, 2, m.data());
                std::vector<double> m64(H);
                mix64(d.mqueries.data() + std::size_t(k) * H, msp64, m64.data());
                const float* y = got.message.data() + (std::size_t(k) * T + t) * H;
                exact(tally, y, m.data(), H);
                f32_criterion(ftally, y, m64.data());
                const int xi = nq == 1 ? 0 : k;
                std::vector<float> rx(H), rm(H);
                ik::rms_reference(outs[xi].data(), rx.data());
                ik::rms_reference(m.data(), rm.data());
                std::vector<std::uint16_t> want(2 * H);
                for (int j = 0; j < H; ++j) { want[j] = b16(rx[j]), want[H + j] = b16(rm[j]); }
                const std::uint16_t* gi = got.gate_input.data() + (std::size_t(k) * T + t) * 2 * H;
                exact(tally, gi, want.data(), 2 * H);
                std::vector<double> r1(H), r2(H);
                const auto kx = d64(got.mixed.data() + (std::size_t(xi) * T + t) * H);
                rms64(kx.data(), r1.data());
                rms64(d64(y).data(), r2.data());
                bf16_criterion(tally, gi, r1.data());
                bf16_criterion(tally, gi + H, r2.data());
            }
        }
    }
    if (cfg.tail) { exact(tally, got.ring.data(), ring_after.data(), ring_after.size()); }

    // Decode against verify: each column alone, in call order, from the same initial ring.
    if (cfg.tail) {
        upload_to(d_ring, d.ring);
        Tally inv{"  decode = verify"};
        for (int t = 0; t < T; ++t) {
            const EntryResult one = run_entry(cfg, d, call, {t}, d_ring);
            exact(inv, one.tail_out.data(), got.tail_out.data() + std::size_t(t) * H, H);
            if (!cfg.messages) {
                for (int i = 0; i < 3; ++i) { exact(inv, one.x[i].data(), got.x[i].data() + std::size_t(t) * H, H); }
            } else {
                for (int k = 0; k < 3; ++k) {
                    exact(inv, one.gate_input.data() + std::size_t(k) * 2 * H,
                          got.gate_input.data() + (std::size_t(k) * T + t) * 2 * H, 2 * H);
                }
            }
            if (t == T - 1) { exact(inv, one.ring.data(), got.ring.data(), got.ring.size()); }
        }
        inv.report();
    }
    tally.report();
    if (cfg.messages) { ftally.report(); }
    if (!cfg.messages) { // the A8 rows beside x: the codec of the kernel's own BF16 x, exactly
        Tally a8t{"  x's A8 rows (exact encoder)"};
        for (int i = 0; i < 3; ++i) {
            for (int t = 0; t < T; ++t) {
                const auto q = a8_host(got.x[i].data() + std::size_t(t) * H);
                exact(a8t, got.x8[i].data() + std::size_t(t) * moe::kA8XRowBytes, q.data(), q.size());
            }
        }
        a8t.report();
    }
    cudaFree(d_ring);
}

// ---------------------------------------------------------------------------- mlp_entry and gate_exit

struct MlpConfig {
    const char* name;
    bool boundary;
    int blocks;
    bool messages;
    moe::ExpertFormat format;
    bool a16 = false; // the experts' rows as A16 limb rows and exponents
    bool a8  = false; // the experts' rows as A8 rows
};

void mlp_case(const MlpConfig& cfg, std::uint32_t seed) {
    const Columns call = the_call();
    const int T        = call.size();
    Rng rng(seed);
    const auto o     = rng.bf16(static_cast<std::size_t>(T) * H, 0.5F);
    const auto conv  = rng.bf16(std::size_t{4} * H, 0.4F);
    const auto h     = rng.bf16(static_cast<std::size_t>(T) * H, 1.0F);
    const auto norm  = rng.bf16(H, 0.2F, 1.0F);
    const auto query = rng.f32(H, 2.0F), mquery = rng.f32(H, 2.0F);
    std::vector<std::vector<std::uint16_t>> blocks;
    for (int s = 0; s < cfg.blocks; ++s) { blocks.push_back(rng.bf16(static_cast<std::size_t>(T) * H, 1.0F + 0.5F * s)); }
    const auto ms0 = rng.bf16(static_cast<std::size_t>(T) * H, 1.5F), ms1 = rng.bf16(static_cast<std::size_t>(T) * H, 1.5F);
    const HostRing ring0 = rng.bf16(static_cast<std::size_t>(kLanes) * R * H, 0.5F);

    const int rb = cfg.a16 ? moe::kA16XRowBytes : cfg.a8 ? moe::kA8XRowBytes : moe::row_bytes(cfg.format, H);
    auto* d_emax = dalloc<std::int32_t>(T);
    DeviceMap map(call);
    auto* d_o = upload(o);
    auto* d_conv = upload(conv);
    auto* d_h = upload(h);
    auto* d_norm = upload(norm);
    auto* d_q = upload(query);
    auto* d_mq = upload(mquery);
    auto* d_ring = upload(ring0);
    auto* d_ms0 = upload(ms0);
    auto* d_ms1 = upload(ms1);
    std::vector<std::uint16_t*> d_blocks;
    for (const auto& b : blocks) { d_blocks.push_back(upload(b)); }
    auto* d_cand = dalloc<float>(static_cast<std::size_t>(T) * H);
    auto* d_x = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * H);
    auto* d_rows = dalloc<std::uint8_t>(static_cast<std::size_t>(T) * rb);
    auto* d_amax = dalloc<float>(T);
    auto* d_mixed = dalloc<float>(static_cast<std::size_t>(T) * H);
    auto* d_message = dalloc<float>(static_cast<std::size_t>(T) * H);
    auto* d_gate = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * 2 * H);

    ik::MlpEntryArgs a;
    a.columns  = T;
    a.map      = map.map();
    a.o        = d_o;
    a.conv     = d_conv;
    a.ring     = {d_ring, H, std::int64_t{R} * H};
    a.h        = d_h;
    a.boundary = cfg.boundary;
    a.cand     = d_cand;
    for (int s = 0; s < cfg.blocks; ++s) { a.blocks.rows[s] = d_blocks[s]; }
    a.blocks.count = cfg.blocks;
    a.query        = d_q;
    if (cfg.messages) {
        a.message_query      = d_mq;
        a.message_sources[0] = d_ms0;
        a.message_sources[1] = d_ms1;
        a.mixed              = d_mixed;
        a.message            = d_message;
        a.gate_input         = d_gate;
    } else {
        a.norm = d_norm;
        a.x    = d_x;
        a.rows = cfg.a16  ? ik::ExpertRows{cfg.format, d_rows, nullptr, d_emax}
                 : cfg.a8 ? ik::ExpertRows{cfg.format, d_rows, nullptr, nullptr, true}
                          : ik::ExpertRows{cfg.format, d_rows, d_amax};
    }
    ik::mlp_entry(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "mlp_entry");
    const auto cand = download(d_cand, static_cast<std::size_t>(T) * H);
    const auto x = download(d_x, static_cast<std::size_t>(T) * H);
    const auto rows = download(d_rows, static_cast<std::size_t>(T) * rb);
    const auto amax = download(d_amax, T);
    const auto emax = download(d_emax, T);
    const auto mixed = download(d_mixed, static_cast<std::size_t>(T) * H);
    const auto message = download(d_message, static_cast<std::size_t>(T) * H);
    const auto gate = download(d_gate, static_cast<std::size_t>(T) * 2 * H);
    const auto ring = download(d_ring, ring0.size());

    Tally tally{cfg.name}, ftally{"  (FP32 outputs)"},
        qtally{cfg.a16 ? "  A16 limb rows and exponents (exact encoder)" : "  experts' rows (exact encoders)"};
    HostRing before = ring0, after = ring0;
    std::vector<std::vector<float>> raw(T, std::vector<float>(H));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < H; ++i) { raw[t][i] = f(o[std::size_t(t) * H + i]); }
    }
    for (int t = 0; t < T; ++t) {
        const auto xin = conv_inputs(call, t, raw, before);
        std::vector<float> m(H), c(H);
        std::vector<double> m64(H), c64(H);
        conv_host(conv, xin, m.data(), m64.data());
        for (int i = 0; i < H; ++i) {
            c[i]   = cfg.boundary ? m[i] : canon::add_rn(f(h[std::size_t(t) * H + i]), m[i]);
            c64[i] = (cfg.boundary ? 0.0 : double{f(h[std::size_t(t) * H + i])}) + bf(static_cast<float>(m64[i]));
        }
        exact(tally, cand.data() + std::size_t(t) * H, c.data(), H);
        if (call.lane[t] >= 0) { std::copy_n(o.data() + std::size_t(t) * H, H, ring_at(after, call.lane[t], call.pos[t])); }
        // cand within the inner boundary's flip of FP64 (an FP32 sum of BF16 values: exact over the rounded a).
        {
            const double mx = row_max(c64.data());
            for (int i = 0; i < H; ++i) {
                const double ratio = std::fabs(cand[std::size_t(t) * H + i] - c64[i]) / (ulp_bf16(m64[i]) + 0x1p-20 * mx);
                tally.worst        = std::max(tally.worst, ratio);
                tally.bad += ratio > 1.0 ? 1 : 0;
            }
        }
        std::vector<std::vector<float>> src;
        for (int s = 0; s < cfg.blocks; ++s) {
            std::vector<float> v(H);
            for (int i = 0; i < H; ++i) { v[i] = f(blocks[s][std::size_t(t) * H + i]); }
            src.push_back(v);
        }
        src.push_back(std::vector<float>(cand.begin() + std::size_t(t) * H, cand.begin() + std::size_t(t + 1) * H));
        std::vector<const float*> sp;
        std::vector<std::vector<double>> src64;
        std::vector<const double*> sp64;
        for (const auto& s : src) { sp.push_back(s.data()), src64.push_back(d64(s.data())); }
        for (const auto& s : src64) { sp64.push_back(s.data()); }
        std::vector<float> mm(H);
        std::vector<double> mm64(H);
        ik::mix_reference(query.data(), sp.data(), static_cast<int>(sp.size()), mm.data());
        mix64(query.data(), sp64, mm64.data());
        if (!cfg.messages) {
            std::vector<float> y(H);
            ik::norm_reference(mm.data(), norm.data(), y.data());
            std::vector<std::uint16_t> want(H);
            for (int i = 0; i < H; ++i) { want[i] = b16(y[i]); }
            exact(tally, x.data() + std::size_t(t) * H, want.data(), H);
            std::vector<double> r(H);
            norm64(mm64.data(), norm.data(), r.data());
            bf16_criterion(tally, x.data() + std::size_t(t) * H, r.data());
            if (cfg.a16) {
                std::int32_t e = 0;
                const auto q   = limbs_host(x.data() + std::size_t(t) * H, &e);
                exact(qtally, rows.data() + std::size_t(t) * rb, q.data(), q.size());
                exact(qtally, &emax[t], &e, 1);
            } else if (cfg.a8) {
                const auto q = a8_host(x.data() + std::size_t(t) * H);
                exact(qtally, rows.data() + std::size_t(t) * rb, q.data(), q.size());
            } else {
                float am = 0;
                const auto q = quantize_host(x.data() + std::size_t(t) * H, cfg.format, &am);
                exact(qtally, rows.data() + std::size_t(t) * rb, q.data(), q.size());
                if (cfg.format == moe::ExpertFormat::Nvfp4FourOverSix) { exact(qtally, &amax[t], &am, 1); }
            }
        } else {
            exact(tally, mixed.data() + std::size_t(t) * H, mm.data(), H);
            f32_criterion(ftally, mixed.data() + std::size_t(t) * H, mm64.data());
            std::vector<float> a0(H), a1(H);
            for (int i = 0; i < H; ++i) { a0[i] = f(ms0[std::size_t(t) * H + i]), a1[i] = f(ms1[std::size_t(t) * H + i]); }
            const float* msp[2] = {a0.data(), a1.data()};
            std::vector<float> msg(H);
            ik::mix_reference(mquery.data(), msp, 2, msg.data());
            const auto b0 = d64(a0.data()), b1 = d64(a1.data());
            std::vector<double> msg64(H);
            mix64(mquery.data(), {b0.data(), b1.data()}, msg64.data());
            exact(tally, message.data() + std::size_t(t) * H, msg.data(), H);
            f32_criterion(ftally, message.data() + std::size_t(t) * H, msg64.data());
            std::vector<float> rx(H), rm(H);
            ik::rms_reference(mm.data(), rx.data());
            ik::rms_reference(msg.data(), rm.data());
            std::vector<std::uint16_t> want(2 * H);
            for (int i = 0; i < H; ++i) { want[i] = b16(rx[i]), want[H + i] = b16(rm[i]); }
            exact(tally, gate.data() + std::size_t(t) * 2 * H, want.data(), 2 * H);
        }
    }
    exact(tally, ring.data(), after.data(), after.size());
    tally.report();
    if (cfg.messages) {
        ftally.report();
    } else {
        qtally.report();
    }
    std::vector<void*> frees = {d_o, d_conv, d_h, d_norm, d_q, d_mq, d_ring, d_ms0, d_ms1, d_cand, d_x, d_rows,
                                d_amax, d_mixed, d_message, d_gate, d_emax};
    for (auto* p : d_blocks) { frees.push_back(p); }
    for (void* p : frees) { cudaFree(p); }
}

void gate_case(int kinds, bool shared_x, bool rows_too, moe::ExpertFormat format, const char* name, std::uint32_t seed,
               bool a16 = false, bool a8x = false) {
    constexpr int T = 7;
    Rng rng(seed);
    const int planes = shared_x ? 1 : kinds;
    const auto mixed = rng.f32(static_cast<std::size_t>(planes) * T * H, 1.0F);
    const auto message = rng.f32(static_cast<std::size_t>(kinds) * T * H, 1.0F);
    const auto fg = rng.f32(static_cast<std::size_t>(kinds) * T * 2 * H, 1.2F);
    std::vector<std::uint16_t> norms[3];
    for (auto& n : norms) { n = rng.bf16(H, 0.2F, 1.0F); }
    const int rb = a16 ? moe::kA16XRowBytes : a8x ? moe::kA8XRowBytes : moe::row_bytes(format, H);
    auto* d_emax = dalloc<std::int32_t>(T);
    auto* d_mixed = upload(mixed);
    auto* d_message = upload(message);
    auto* d_fg = upload(fg);
    std::uint16_t* d_norms[3];
    std::uint16_t* d_x[3];
    for (int k = 0; k < 3; ++k) { d_norms[k] = upload(norms[k]), d_x[k] = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * H); }
    auto* d_rows = dalloc<std::uint8_t>(static_cast<std::size_t>(T) * rb);
    auto* d_amax = dalloc<float>(T);
    ik::GateExitArgs a;
    a.columns  = T;
    a.kinds    = kinds;
    a.mixed    = d_mixed;
    a.shared_x = shared_x;
    a.message  = d_message;
    a.fg       = d_fg;
    for (int k = 0; k < kinds; ++k) { a.norms[k] = d_norms[k], a.x[k] = d_x[k]; }
    std::uint8_t* d_x8[3] = {};
    a.x8_stride           = moe::kA8XRowBytes / 4;
    for (int k = 0; k < kinds; ++k) { // each kind's x as A8 rows too (K2's 8-bit inputs)
        d_x8[k] = dalloc<std::uint8_t>(static_cast<std::size_t>(T) * moe::kA8XRowBytes);
        a.x8[k] = reinterpret_cast<std::int32_t*>(d_x8[k]);
    }
    if (rows_too) {
        a.rows = a16   ? ik::ExpertRows{format, d_rows, nullptr, d_emax}
                 : a8x ? ik::ExpertRows{format, d_rows, nullptr, nullptr, true}
                       : ik::ExpertRows{format, d_rows, d_amax};
    }
    ik::gate_exit(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "gate_exit");
    Tally tally{name}, qtally{a16 ? "  A16 limb rows and exponents (exact encoder)" : "  experts' rows (exact encoders)"};
    const auto rows = download(d_rows, static_cast<std::size_t>(T) * rb);
    const auto amax = download(d_amax, T);
    const auto emax = download(d_emax, T);
    Tally a8t{"  x's A8 rows (exact encoder)"};
    for (int k = 0; k < kinds; ++k) {
        const auto x = download(d_x[k], static_cast<std::size_t>(T) * H);
        const auto x8 = download(d_x8[k], static_cast<std::size_t>(T) * moe::kA8XRowBytes);
        cudaFree(d_x8[k]);
        for (int t = 0; t < T; ++t) {
            const auto q8 = a8_host(x.data() + std::size_t(t) * H);
            exact(a8t, x8.data() + std::size_t(t) * moe::kA8XRowBytes, q8.data(), q8.size());
        }
        for (int t = 0; t < T; ++t) {
            const float* xv = mixed.data() + (std::size_t(shared_x ? 0 : k) * T + t) * H;
            const float* mv = message.data() + (std::size_t(k) * T + t) * H;
            const float* fv = fg.data() + (std::size_t(k) * T + t) * 2 * H;
            std::vector<float> g(H), y(H);
            std::vector<double> g64(H), r(H);
            for (int i = 0; i < H; ++i) {
                g[i]   = ik::message_gate(xv[i], mv[i], fv[i], fv[H + i]);
                g64[i] = (1.0 - std::tanh(double{fv[i]})) * xv[i] + std::tanh(double{fv[H + i]}) * mv[i];
            }
            ik::norm_reference(g.data(), norms[k].data(), y.data());
            std::vector<std::uint16_t> want(H);
            for (int i = 0; i < H; ++i) { want[i] = b16(y[i]); }
            exact(tally, x.data() + std::size_t(t) * H, want.data(), H);
            norm64(g64.data(), norms[k].data(), r.data());
            bf16_criterion(tally, x.data() + std::size_t(t) * H, r.data());
            if (rows_too && k == 0 && a16) {
                std::int32_t e = 0;
                const auto q   = limbs_host(x.data() + std::size_t(t) * H, &e);
                exact(qtally, rows.data() + std::size_t(t) * rb, q.data(), q.size());
                exact(qtally, &emax[t], &e, 1);
            } else if (rows_too && k == 0 && a8x) {
                const auto q = a8_host(x.data() + std::size_t(t) * H);
                exact(qtally, rows.data() + std::size_t(t) * rb, q.data(), q.size());
            } else if (rows_too && k == 0) {
                float am = 0;
                const auto q = quantize_host(x.data() + std::size_t(t) * H, format, &am);
                exact(qtally, rows.data() + std::size_t(t) * rb, q.data(), q.size());
                if (format == moe::ExpertFormat::Nvfp4FourOverSix) { exact(qtally, &amax[t], &am, 1); }
            }
        }
    }
    tally.report();
    if (rows_too) { qtally.report(); }
    a8t.report();
    std::vector<void*> frees = {d_mixed, d_message, d_fg, d_rows, d_amax, d_emax};
    for (int k = 0; k < 3; ++k) { frees.push_back(d_norms[k]), frees.push_back(d_x[k]); }
    for (void* p : frees) { cudaFree(p); }
}

void softcap_case() {
    Rng rng(5);
    constexpr std::int64_t N = 200058 * 3 + 5;
    auto v = rng.f32(N, 20.0F);
    v[0] = 0.0F, v[1] = -0.0F, v[2] = 1e-30F, v[3] = 300.0F, v[4] = -1e6F;
    auto* d = upload(v);
    ik::softcap(d, N, 30.0F, nullptr);
    cuda_check(cudaDeviceSynchronize(), "softcap");
    const auto got = download(d, N);
    // Criterion 3 ulp: x / 30 rounds (a non-power-of-two cap: 0.5 ulp, damped by tanh), tanh_c adds its ~1.25 ulp and
    // the product 0.5 ulp, about 2.3 ulp in the result's binade and up to 3 where it sits just above a power of two.
    Tally tally{"softcap 30"};
    for (std::int64_t i = 0; i < N; ++i) {
        const float want = canon::softcap_c(v[i], 30.0F);
        tally.differ += std::memcmp(&want, &got[i], 4) != 0 ? 1 : 0;
        const double r     = 30.0 * std::tanh(double{v[i]} / 30.0);
        const double ratio = std::fabs(got[i] - r) / (3.0 * std::ldexp(1.0, std::ilogb(std::max(std::fabs(r), 1e-38)) - 23));
        tally.worst        = std::max(tally.worst, ratio);
        tally.bad += ratio > 1.0 ? 1 : 0;
    }
    tally.report();
    cudaFree(d);
}

} // namespace

INFERNIX_OP_TEST(inkling2_visit) {
    embed_case();
    entry_case({"visit_entry layer 0", false, false, true, 1, 0, false}, 21);
    entry_case({"visit_entry sliding, 2 blocks", true, false, false, 1, 2, false}, 22);
    entry_case({"visit_entry full, 3 blocks, restart", true, true, false, 3, 3, false}, 23);
    entry_case({"visit_entry sliding messages, 4 blocks", true, false, false, 1, 4, true}, 24);
    entry_case({"visit_entry full messages, 1 block", true, false, false, 3, 1, true}, 25);
    mlp_case({"mlp_entry NVFP4, 2 blocks", false, 2, false, moe::ExpertFormat::Nvfp4FourOverSix}, 31);
    mlp_case({"mlp_entry MXFP8 boundary, 3 blocks", true, 3, false, moe::ExpertFormat::Mxfp8}, 32);
    mlp_case({"mlp_entry messages, 4 blocks", false, 4, true, moe::ExpertFormat::Nvfp4FourOverSix}, 33);
    gate_case(3, false, false, moe::ExpertFormat::Nvfp4FourOverSix, "gate_exit 3 kinds", 41);
    gate_case(3, true, false, moe::ExpertFormat::Nvfp4FourOverSix, "gate_exit 3 kinds, shared x", 42);
    gate_case(1, false, true, moe::ExpertFormat::Nvfp4FourOverSix, "gate_exit mlp NVFP4", 43);
    gate_case(1, false, true, moe::ExpertFormat::Mxfp8, "gate_exit mlp MXFP8", 44);
    mlp_case({"mlp_entry NVFP4 A16, 2 blocks", false, 2, false, moe::ExpertFormat::Nvfp4FourOverSix, true}, 34);
    gate_case(1, false, true, moe::ExpertFormat::Nvfp4FourOverSix, "gate_exit mlp NVFP4 A16", 45, true);
    mlp_case({"mlp_entry NVFP4 A8, 2 blocks", false, 2, false, moe::ExpertFormat::Nvfp4FourOverSix, false, true}, 35);
    gate_case(1, false, true, moe::ExpertFormat::Nvfp4FourOverSix, "gate_exit mlp NVFP4 A8", 46, false, true);
    softcap_case();
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
