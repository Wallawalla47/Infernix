// Inkling 2's decode and verify attention (include/infernix/ops/inkling2_attention.h), sliding and full visits.
//
// One call of 30 columns: lane 0 at positions 1100-1103 (its window starts inside unit 1), lane 1 at 60-81 (crossing
// a unit boundary, history from position 0; two attention groups of 16 and 6), a padding column, lane 2 at 5000
// (sliding: a 17-unit window; full: 79
// units in 27 splits of 3, so the in-split merge runs) and lane 3 at 9000-9001 with its first position 8013 inside
// its lowest unit (keys and conv history below it masked; full visits' tau above 1). Every earlier position of each lane holds random represented
// keys (BF16 K, FP16 V of BF16 values) and the conv rings random inputs. Projections are random BF16 of the
// checkpoint's scales (some query heads scaled so their clip engages; relative tables and sinks of realistic spread).
//
// Oracles:
//   - prologue (no hardware approximation): a host emulation through visit_math.h gives the appended K and V, the
//     final q and r and the rings bit for bit; FP64 stage by stage (k and v within half a BF16 ulp plus the inner conv
//     rounding's ulp, q and r likewise);
//   - attention: FP64 scores over the kernel's stored q, r and represented K/V with the bias table at its BF16
//     boundary (the pinned FP32 fold, itself checked within one BF16 ulp of FP64), FP64 softmax with the sink, the
//     output within half a BF16 ulp of FP64 plus sigmoid(gate) times one ulp of the inner BF16 a plus 2^-16 of the
//     row's largest |v|; ex2.approx makes the bits the kernel's own, so they are pinned by the next check;
//   - decode against verify: the columns run one at a time, in call order, from the same initial state give the call's
//     outputs, K/V pages and rings bit for bit.

#include "infernix/ops/inkling2_attention.h"
#include "ops/common/canonical_math.h"
#include "ops/inkling2/visit_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
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
using infernix::test::cuda_check;

namespace {

constexpr int D = 128, NH = ik::kHeads, NKV = ik::kKvHeads, G = ik::kGroupHeads, RD = ik::kRelDim;
constexpr int KVW = NKV * D, QW = NH * D, RW = NH * RD, R = ik::kRingPositions, U = ik::kKeyUnit;
constexpr int kLanes = 4, kPoolPages = 256, kTableStride = 160;
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

float f(std::uint16_t v) { return canon::bf16_to_f32(v); }
float bf(float x) { return canon::bf16_to_f32(canon::f32_to_bf16_rn(x)); }
std::uint16_t b16(float x) { return canon::f32_to_bf16_rn(x); }
std::uint16_t h16(float x) { return __half_as_ushort(__float2half_rn(x)); }
float hf(std::uint16_t u) { return __half2float(__ushort_as_half(u)); }
double ulp_bf16(double r) {
    const double a = std::fabs(r);
    return a < 0x1p-126 ? 0x1p-133 : std::ldexp(1.0, std::ilogb(a) - 7);
}
double ulp_f16(double r) {
    const double a = std::fabs(r);
    return a < 0x1p-14 ? 0x1p-24 : std::ldexp(1.0, std::ilogb(a) - 10);
}

struct Tally {
    const char* name;
    double worst = 0;
    long bad = 0, differ = 0, n = 0;
    void ratio(double r) {
        worst = std::max(worst, r);
        bad += r > 1.0 ? 1 : 0;
        ++n;
    }
    template <class T>
    void exact(const T* a, const T* b, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) { differ += std::memcmp(a + i, b + i, sizeof(T)) != 0 ? 1 : 0; }
        n += static_cast<long>(count);
    }
    void report() const {
        std::printf("%-40s %8ld checked, worst %.3f of the criterion, %ld beyond it, %ld differ\n", name, n, worst, bad,
                    differ);
        check(bad == 0 && differ == 0, name);
    }
};

// ---------------------------------------------------------------------------- the call

struct Call {
    std::vector<int> lane, pos, first = {0, 0, 0, 8013};
    [[nodiscard]] int size() const { return static_cast<int>(lane.size()); }
};
Call the_call() {
    Call c;
    for (int p = 1100; p < 1104; ++p) { c.lane.push_back(0), c.pos.push_back(p); }
    for (int p = 60; p < 82; ++p) { c.lane.push_back(1), c.pos.push_back(p); } // two groups: 16 + 6 columns
    c.lane.push_back(-1), c.pos.push_back(0);
    c.lane.push_back(2), c.pos.push_back(5000);
    for (int p = 9000; p < 9002; ++p) { c.lane.push_back(3), c.pos.push_back(p); }
    return c;
}

// ---------------------------------------------------------------------------- the problem

struct Problem {
    bool full = false;
    // Projections [T][width].
    std::vector<std::uint16_t> q, r, k, v, sink, gate, ktaps, vtaps;
    // Layer.
    std::vector<std::uint16_t> qscale, rscale, table, kconv, vconv;
    // State: KV (ring [3][8][17][64][128] or pool [8][128 pages][64][128]), its block table, conv rings.
    std::vector<std::uint16_t> kv_k, kv_v, kring, vring;
    std::vector<int> block_table;
};

std::size_t kv_row(const Problem& p, int lane, int head, int pos) {
    if (!p.full) {
        const int page = (pos / U) % ik::kWindowRingPages;
        return ((std::size_t(lane * NKV + head) * ik::kWindowRingPages + page) * U + pos % U) * D;
    }
    const int page = p.block_table[std::size_t(lane) * kTableStride + pos / U];
    return ((std::size_t(head) * kPoolPages + page) * U + pos % U) * D;
}

Problem make_problem(bool full, const Call& call, std::uint32_t seed) {
    std::mt19937 gen(seed);
    const auto normal = [&](std::size_t n, float sd) {
        std::normal_distribution<float> d(0.0F, sd);
        std::vector<std::uint16_t> out(n);
        for (auto& x : out) { x = b16(d(gen)); }
        return out;
    };
    const int T = call.size();
    Problem p;
    p.full  = full;
    p.q     = normal(std::size_t(T) * QW, 1.0F);
    p.r     = normal(std::size_t(T) * RW, 1.0F);
    p.k     = normal(std::size_t(T) * KVW, 1.0F);
    p.v     = normal(std::size_t(T) * KVW, 1.0F);
    p.sink  = normal(std::size_t(T) * NH, 2.0F);
    p.gate  = normal(std::size_t(T) * QW, 1.5F);
    p.ktaps = normal(std::size_t(T) * NKV * 4, 0.3F);
    p.vtaps = normal(std::size_t(T) * NKV * 4, 0.3F);
    p.qscale = normal(std::size_t(NH) * D, 0.15F);
    for (std::size_t i = 0; i < p.qscale.size(); ++i) {
        const int head = static_cast<int>(i / D);
        p.qscale[i]    = b16(f(p.qscale[i]) + (head % 4 == 0 ? 3.0F : 1.0F)); // heads 0, 4, ... clip
    }
    p.rscale = normal(std::size_t(NH) * RD, 0.5F);
    for (auto& x : p.rscale) { x = b16(f(x) + 2.0F); }
    p.table = normal(std::size_t(RD) * ik::kRelativeExtent, 0.4F);
    p.kconv = normal(std::size_t(4) * KVW, 0.4F);
    p.vconv = normal(std::size_t(4) * KVW, 0.4F);
    p.kring = normal(std::size_t(kLanes) * R * KVW, 1.0F);
    p.vring = normal(std::size_t(kLanes) * R * KVW, 1.0F);
    const std::size_t kv_words = full ? std::size_t(NKV) * kPoolPages * U * D
                                      : std::size_t(kLanes) * NKV * ik::kWindowRingPages * U * D;
    p.kv_k = normal(kv_words, 1.0F);
    p.kv_v = normal(kv_words, 1.0F);
    for (auto& x : p.kv_v) { x = h16(f(x)); } // the represented FP16 V of a BF16 value
    if (full) {
        // Lanes' pages in a scattered order.
        std::vector<int> pages(kPoolPages);
        for (int i = 0; i < kPoolPages; ++i) { pages[i] = i; }
        std::shuffle(pages.begin(), pages.end(), gen);
        p.block_table.assign(std::size_t(kLanes) * kTableStride, -1);
        const int need[kLanes] = {1104 / U + 1, 76 / U + 1, 5000 / U + 1, 9002 / U + 1};
        int next = 0;
        for (int l = 0; l < kLanes; ++l) {
            for (int b = 0; b < need[l]; ++b) { p.block_table[std::size_t(l) * kTableStride + b] = pages[next++]; }
        }
        check(next <= kPoolPages, "the test's pool holds its lanes");
    }
    return p;
}

struct Device {
    std::uint16_t *q, *r, *k, *v, *sink, *gate, *ktaps, *vtaps, *qscale, *rscale, *table, *kconv, *vconv;
    std::uint16_t *kv_k, *kv_v, *kring, *vring, *out;
    int *block_table, *lane, *pos, *first, *groups;
    void* ws;
};

// Runs the attention over columns `subset` of the call as one call; the state lives in `state` (kv and rings).
struct Outputs {
    std::vector<std::uint16_t> out, q, r, kv_k, kv_v, kring, vring;
};

Outputs run(const Problem& p, const Call& call, const std::vector<int>& subset, Device& d) {
    const int T = static_cast<int>(subset.size());
    std::vector<int> lane, pos, groups;
    int widest = 1;
    for (int i = 0; i < T; ++i) { lane.push_back(call.lane[subset[i]]), pos.push_back(call.pos[subset[i]]); }
    // Each lane's columns in groups of at most 16, as the forward splits them.
    for (int l = 0; l < kLanes; ++l) {
        int begin = -1, count = 0;
        for (int i = 0; i < T; ++i) {
            if (lane[i] == l) {
                if (begin < 0) { begin = i; }
                ++count;
            }
        }
        for (int done = 0; done < count; done += ik::kMaxAttentionLaneColumns) {
            const int n = std::min(ik::kMaxAttentionLaneColumns, count - done);
            groups.insert(groups.end(), {begin + done, n, l});
            widest = std::max(widest, n);
        }
    }
    const auto pick = [&](const std::vector<std::uint16_t>& v, int width) {
        std::vector<std::uint16_t> o(std::size_t(T) * width);
        for (int i = 0; i < T; ++i) { std::copy_n(v.data() + std::size_t(subset[i]) * width, width, o.data() + std::size_t(i) * width); }
        return o;
    };
    upload_to(d.q, pick(p.q, QW));
    upload_to(d.r, pick(p.r, RW));
    upload_to(d.k, pick(p.k, KVW));
    upload_to(d.v, pick(p.v, KVW));
    upload_to(d.sink, pick(p.sink, NH));
    upload_to(d.gate, pick(p.gate, QW));
    upload_to(d.ktaps, pick(p.ktaps, NKV * 4));
    upload_to(d.vtaps, pick(p.vtaps, NKV * 4));
    upload_to(d.lane, lane);
    upload_to(d.pos, pos);
    upload_to(d.groups, groups);
    ik::AttentionArgs a;
    a.columns           = T;
    a.groups            = static_cast<int>(groups.size() / 3);
    a.max_group_columns = widest;
    a.map               = {d.lane, d.pos, d.first, T};
    a.groups_columns    = d.groups;
    a.in               = {d.q, d.r, d.k, d.v, d.sink, d.gate, p.full ? d.ktaps : nullptr, p.full ? d.vtaps : nullptr};
    a.layer            = {p.full, d.qscale, d.rscale, d.table, d.kconv, d.vconv};
    a.key_ring         = {d.kring, KVW, std::int64_t{R} * KVW};
    a.value_ring       = {d.vring, KVW, std::int64_t{R} * KVW};
    a.kv.k            = d.kv_k;
    a.kv.v            = d.kv_v;
    a.kv.lane_stride  = std::int64_t{NKV} * ik::kWindowRingPages * U * D;
    a.kv.table        = p.full ? d.block_table : nullptr;
    a.kv.table_stride = p.full ? kTableStride : 0;
    a.kv.pool_pages   = p.full ? kPoolPages : 0;
    a.ws               = ik::carve_attention_workspace(d.ws);
    a.out              = d.out;
    ik::attention(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "attention");
    Outputs o;
    o.out   = download(d.out, std::size_t(T) * QW);
    o.q     = download(a.ws.q, std::size_t(T) * QW);
    o.r     = download(a.ws.r, std::size_t(T) * RW);
    o.kv_k  = download(d.kv_k, p.kv_k.size());
    o.kv_v  = download(d.kv_v, p.kv_v.size());
    o.kring = download(d.kring, p.kring.size());
    o.vring = download(d.vring, p.vring.size());
    return o;
}

void reset_state(const Problem& p, Device& d) {
    upload_to(d.kv_k, p.kv_k);
    upload_to(d.kv_v, p.kv_v);
    upload_to(d.kring, p.kring);
    upload_to(d.vring, p.vring);
}

// ---------------------------------------------------------------------------- host prologue emulation

float tau_of(const Problem& p, int pos) { return p.full ? ik::tau(pos) : 1.0F; }

// The conv input of channel ch of column t, k positions back (0, the call's column, or the pre-call ring).
float history(const Call& c, int t, int k, int ch, const std::vector<std::uint16_t>& raw,
              const std::vector<std::uint16_t>& ring) {
    const int lane = c.lane[t];
    if (k == 0) { return f(raw[std::size_t(t) * KVW + ch]); }
    if (lane < 0 || c.pos[t] - k < c.first[lane]) { return 0.0F; }
    if (t - k >= 0 && c.lane[t - k] == lane) { return f(raw[std::size_t(t - k) * KVW + ch]); }
    return f(ring[(std::size_t(lane) * R + (c.pos[t] - k) % R) * KVW + ch]);
}

void prologue_check(const Problem& p, const Call& call, const Outputs& got, const char* name) {
    Tally tally{name};
    const int T = call.size();
    std::vector<std::uint16_t> kv_k = p.kv_k, kv_v = p.kv_v, kring = p.kring, vring = p.vring;
    for (int t = 0; t < T; ++t) {
        const int lane = call.lane[t], pos = call.pos[t];
        if (lane < 0) {
            std::vector<std::uint16_t> zero(QW, 0);
            tally.exact(got.out.data() + std::size_t(t) * QW, zero.data(), QW);
            continue;
        }
        for (int kv = 0; kv < NKV; ++kv) {
            for (int which = 0; which < 2; ++which) {
                const auto& raw  = which == 0 ? p.k : p.v;
                const auto& ring = which == 0 ? p.kring : p.vring;
                const auto& conv = which == 0 ? p.kconv : p.vconv;
                const auto& taps = which == 0 ? p.ktaps : p.vtaps;
                float c[D];
                double c64[D];
                for (int e = 0; e < D; ++e) {
                    const int ch = kv * D + e;
                    float w[4], x[4], tap[4];
                    double acc = 0;
                    for (int lag = 0; lag < 4; ++lag) {
                        w[lag]   = f(conv[std::size_t(lag) * KVW + ch]);
                        x[lag]   = history(call, t, lag, ch, raw, ring);
                        tap[lag] = f(taps[std::size_t(t) * NKV * 4 + kv * 4 + lag]);
                        acc += (double{w[lag]} + (p.full ? double{tap[lag]} : 0.0)) * x[lag];
                    }
                    c[e]   = bf(p.full ? ik::conv_dynamic(w, tap, x) : ik::conv_static(w, x));
                    c64[e] = (acc + x[0]) / (p.full ? 3.0 : std::sqrt(5.0));
                }
                const float sum = ik::head_dot_reference(c, c);
                double s64      = 0;
                for (int e = 0; e < D; ++e) { s64 += double{c[e]} * c[e]; }
                const std::size_t at = kv_row(p, lane, kv, pos);
                for (int e = 0; e < D; ++e) {
                    if (which == 0) {
                        const float y        = bf(canon::mul_rn(c[e], ik::clip_factor(sum, 22.627416997969522F)));
                        kv_k[at + e]         = b16(y);
                        const double y64     = c[e] * std::min(1.0, 22.627416997969522 / (std::sqrt(s64) + 1e-6));
                        tally.ratio(std::fabs(f(got.kv_k[at + e]) - y64) /
                                    (0.5 * ulp_bf16(y64) + ulp_bf16(c64[e]) + 1e-30));
                    } else {
                        const float y    = bf(canon::mul_rn(c[e], ik::rms_factor(sum, 128.0F)));
                        kv_v[at + e]     = h16(y);
                        const double y64 = c[e] / std::sqrt(s64 / 128.0 + 1e-6);
                        tally.ratio(std::fabs(hf(got.kv_v[at + e]) - y64) /
                                    (0.5 * ulp_bf16(y64) + ulp_f16(y64) + ulp_bf16(c64[e]) + 1e-30));
                    }
                }
                auto& out_ring = which == 0 ? kring : vring;
                for (int e = 0; e < D; ++e) {
                    out_ring[(std::size_t(lane) * R + pos % R) * KVW + kv * D + e] = raw[std::size_t(t) * KVW + kv * D + e];
                }
            }
        }
        for (int h = 0; h < NH; ++h) {
            float s[D];
            double s64 = 0;
            for (int e = 0; e < D; ++e) {
                s[e] = canon::mul_rn(f(p.q[std::size_t(t) * QW + h * D + e]), f(p.qscale[h * D + e]));
                s64 += double{s[e]} * s[e];
            }
            const float fq = ik::clip_factor(ik::head_dot_reference(s, s), 22.627416997969522F);
            const double f64 = std::min(1.0, 22.627416997969522 / (std::sqrt(s64) + 1e-6));
            std::uint16_t want[D];
            for (int e = 0; e < D; ++e) {
                float y = bf(canon::mul_rn(s[e], fq));
                double y64 = s[e] * f64;
                if (p.full) {
                    const double inner = y64;
                    y                  = bf(canon::mul_rn(y, tau_of(p, pos)));
                    y64                = bf(static_cast<float>(inner)) * (1.0 + 0.1 * std::log(std::max(1.0, (pos + 1) / 8192.0)));
                }
                want[e] = b16(y);
                tally.ratio(std::fabs(f(got.q[(std::size_t(t) * NH + h) * D + e]) - y64) /
                            (0.5 * ulp_bf16(y64) + (p.full ? ulp_bf16(y64) : 0.0) + 1e-30));
            }
            tally.exact(got.q.data() + (std::size_t(t) * NH + h) * D, want, D);
            float rs[RD];
            float sum = 0;
            double r64 = 0;
            for (int i = 0; i < RD; ++i) {
                rs[i] = canon::mul_rn(f(p.r[std::size_t(t) * RW + h * RD + i]), f(p.rscale[h * RD + i]));
                sum   = canon::fma_rn(rs[i], rs[i], sum);
                r64 += double{rs[i]} * rs[i];
            }
            const float fr = ik::clip_factor(sum, 8.0F);
            std::uint16_t rwant[RD];
            for (int i = 0; i < RD; ++i) {
                float y = bf(canon::mul_rn(rs[i], fr));
                if (p.full) { y = bf(canon::mul_rn(y, tau_of(p, pos))); }
                rwant[i] = b16(y);
                const double y64 = rs[i] * std::min(1.0, 8.0 / (std::sqrt(r64) + 1e-6));
                if (!p.full) { tally.ratio(std::fabs(f(got.r[(std::size_t(t) * NH + h) * RD + i]) - y64) / (0.5 * ulp_bf16(y64) + 1e-30)); }
            }
            tally.exact(got.r.data() + (std::size_t(t) * NH + h) * RD, rwant, RD);
        }
    }
    tally.exact(got.kv_k.data(), kv_k.data(), kv_k.size());
    tally.exact(got.kv_v.data(), kv_v.data(), kv_v.size());
    tally.exact(got.kring.data(), kring.data(), kring.size());
    tally.exact(got.vring.data(), vring.data(), vring.size());
    tally.report();
}

// ---------------------------------------------------------------------------- attention against FP64

void attention_check(const Problem& p, const Call& call, const Outputs& got, const char* name) {
    Tally tally{name}, bias_tally{"  bias table (pinned FP32 fold vs FP64)"};
    const int T = call.size();
    for (int t = 0; t < T; ++t) {
        const int lane = call.lane[t], pos = call.pos[t];
        if (lane < 0) { continue; }
        for (int h = 0; h < NH; ++h) {
            const int kv = h / G;
            const std::uint16_t* q = got.q.data() + (std::size_t(t) * NH + h) * D;
            const std::uint16_t* r = got.r.data() + (std::size_t(t) * NH + h) * RD;
            const int lo = std::max(call.first[lane], p.full ? 0 : std::max(0, pos - (ik::kWindow - 1)));
            std::vector<double> score;
            std::vector<const std::uint16_t*> vrows;
            double vmax = 0;
            for (int j = lo; j <= pos; ++j) {
                const std::size_t at = kv_row(p, lane, kv, j);
                double dot = 0;
                for (int e = 0; e < D; ++e) { dot += double{f(q[e])} * f(got.kv_k[at + e]); }
                const int dist = pos - j;
                double bias = 0;
                if (dist < ik::kRelativeExtent) {
                    float acc = 0;
                    double b64 = 0;
                    for (int i = 0; i < RD; ++i) {
                        const float pv = f(p.table[std::size_t(i) * ik::kRelativeExtent + dist]);
                        acc            = canon::fma_rn(f(r[i]), pv, acc);
                        b64 += double{f(r[i])} * pv;
                    }
                    bias = bf(acc);
                    bias_tally.ratio(std::fabs(bias - b64) / (ulp_bf16(b64) + 1e-30));
                }
                score.push_back(dot / std::sqrt(128.0) + bias);
                vrows.push_back(got.kv_v.data() + at);
                for (int e = 0; e < D; ++e) { vmax = std::max(vmax, std::fabs(double{hf(got.kv_v[at + e])})); }
            }
            const double sink = f(p.sink[std::size_t(t) * NH + h]);
            double m = sink;
            for (double s : score) { m = std::max(m, s); }
            double den = std::exp(sink - m);
            for (double& s : score) { den += (s = std::exp(s - m)); }
            for (int e = 0; e < D; ++e) {
                double acc = 0;
                for (std::size_t j = 0; j < score.size(); ++j) { acc += score[j] * hf(vrows[j][e]); }
                const double a64   = acc / den;
                const double g     = f(p.gate[std::size_t(t) * QW + h * D + e]);
                const double sg    = 1.0 / (1.0 + std::exp(-g));
                const double out64 = a64 * sg;
                const double y     = f(got.out[std::size_t(t) * QW + h * D + e]);
                tally.ratio(std::fabs(y - out64) / (0.5 * ulp_bf16(out64) + sg * ulp_bf16(a64) + 0x1p-16 * vmax));
            }
        }
    }
    bias_tally.report();
    tally.report();
}

void run_case(bool full, std::uint32_t seed) {
    const Call call  = the_call();
    const int T      = call.size();
    const Problem p  = make_problem(full, call, seed);
    Device d{};
    d.q = dalloc<std::uint16_t>(std::size_t(T) * QW), d.r = dalloc<std::uint16_t>(std::size_t(T) * RW);
    d.k = dalloc<std::uint16_t>(std::size_t(T) * KVW), d.v = dalloc<std::uint16_t>(std::size_t(T) * KVW);
    d.sink = dalloc<std::uint16_t>(std::size_t(T) * NH), d.gate = dalloc<std::uint16_t>(std::size_t(T) * QW);
    d.ktaps = dalloc<std::uint16_t>(std::size_t(T) * NKV * 4), d.vtaps = dalloc<std::uint16_t>(std::size_t(T) * NKV * 4);
    d.qscale = upload(p.qscale), d.rscale = upload(p.rscale), d.table = upload(p.table);
    d.kconv = upload(p.kconv), d.vconv = upload(p.vconv);
    d.kv_k = upload(p.kv_k), d.kv_v = upload(p.kv_v), d.kring = upload(p.kring), d.vring = upload(p.vring);
    d.out = dalloc<std::uint16_t>(std::size_t(T) * QW);
    d.block_table = full ? upload(p.block_table) : dalloc<int>(1);
    d.lane = dalloc<int>(T), d.pos = dalloc<int>(T), d.first = upload(call.first), d.groups = dalloc<int>(3 * ik::kMaxAttentionGroups);
    d.ws = dalloc<unsigned char>(ik::attention_workspace_bytes());
    // Garbage in the output, so the padding column's zeros are the kernel's.
    upload_to(d.out, std::vector<std::uint16_t>(std::size_t(T) * QW, 0x7FC0));

    std::vector<int> all(T);
    for (int t = 0; t < T; ++t) { all[t] = t; }
    const Outputs got = run(p, call, all, d);
    prologue_check(p, call, got, full ? "full prologue (emulation, FP64)" : "sliding prologue (emulation, FP64)");
    attention_check(p, call, got, full ? "full attention vs FP64" : "sliding attention vs FP64");

    // A second identical call over the same initial state: the same bits (no state left in the workspace).
    reset_state(p, d);
    const Outputs again = run(p, call, all, d);
    Tally repeat{"  repeated call"};
    repeat.exact(again.out.data(), got.out.data(), got.out.size());
    repeat.report();

    // Decode against verify.
    reset_state(p, d);
    Tally inv{full ? "  full: one column at a time = the call" : "  sliding: one column at a time = the call"};
    Outputs one;
    for (int t = 0; t < T; ++t) {
        one = run(p, call, {t}, d);
        if (call.lane[t] >= 0) { inv.exact(one.out.data(), got.out.data() + std::size_t(t) * QW, QW); }
    }
    inv.exact(one.kv_k.data(), got.kv_k.data(), got.kv_k.size());
    inv.exact(one.kv_v.data(), got.kv_v.data(), got.kv_v.size());
    inv.exact(one.kring.data(), got.kring.data(), got.kring.size());
    inv.exact(one.vring.data(), got.vring.data(), got.vring.size());
    inv.report();

    for (void* x : {static_cast<void*>(d.q), static_cast<void*>(d.r), static_cast<void*>(d.k), static_cast<void*>(d.v),
                    static_cast<void*>(d.sink), static_cast<void*>(d.gate), static_cast<void*>(d.ktaps),
                    static_cast<void*>(d.vtaps), static_cast<void*>(d.qscale), static_cast<void*>(d.rscale),
                    static_cast<void*>(d.table), static_cast<void*>(d.kconv), static_cast<void*>(d.vconv),
                    static_cast<void*>(d.kv_k), static_cast<void*>(d.kv_v), static_cast<void*>(d.kring),
                    static_cast<void*>(d.vring), static_cast<void*>(d.out), static_cast<void*>(d.block_table),
                    static_cast<void*>(d.lane), static_cast<void*>(d.pos), static_cast<void*>(d.first),
                    static_cast<void*>(d.groups), d.ws}) {
        cudaFree(x);
    }
}

// ---------------------------------------------------------------------------- the prompt route

// attention_prompt on one lane's prompt call of 150 columns at positions 1000-1149 (three CTA tiles of up to 64 columns;
// sliding windows across 17 units, full rows over all units from the lane's first position 37). Sliding KV is a
// staging ring of staging_ring_pages(150) pages (the forward's for a wide call); the conv inputs before the call come
// from a history snapshot (as the forward's for a wide call). Criteria (fixed before the first run): each output
// within half a BF16 ulp of FP64 (scores over the kernel's stored q, r and represented K/V, the bias at its pinned
// BF16 boundary, softmax with the sink) plus sigmoid(gate) times (one ulp of the inner BF16 a + 2^-11 of the row's
// largest |v|: p is rounded to FP16 for P x V) plus 2^-16 of the row's largest |v|; the call as calls of 37, 64 and 49
// columns from the same initial state gives the same bits; a repeated call the same bits.
void prompt_case(bool full, std::uint32_t seed) {
    constexpr int T = 150, P0 = 1000, First = 37;
    const int pages = ik::staging_ring_pages(T);
    std::mt19937 gen(seed);
    const auto normal = [&](std::size_t n, float sd) {
        std::normal_distribution<float> d(0.0F, sd);
        std::vector<std::uint16_t> out(n);
        for (auto& x : out) { x = b16(d(gen)); }
        return out;
    };
    // Projections and layer as make_problem's.
    auto q = normal(std::size_t(T) * QW, 1.0F), r = normal(std::size_t(T) * RW, 1.0F);
    auto k = normal(std::size_t(T) * KVW, 1.0F), v = normal(std::size_t(T) * KVW, 1.0F);
    auto sink = normal(std::size_t(T) * NH, 2.0F), gate = normal(std::size_t(T) * QW, 1.5F);
    auto ktaps = normal(std::size_t(T) * NKV * 4, 0.3F), vtaps = normal(std::size_t(T) * NKV * 4, 0.3F);
    auto qscale = normal(std::size_t(NH) * D, 0.15F);
    for (std::size_t i = 0; i < qscale.size(); ++i) { qscale[i] = b16(f(qscale[i]) + ((i / D) % 4 == 0 ? 3.0F : 1.0F)); }
    auto rscale = normal(std::size_t(NH) * RD, 0.5F);
    for (auto& x : rscale) { x = b16(f(x) + 2.0F); }
    auto table = normal(std::size_t(RD) * ik::kRelativeExtent, 0.4F);
    auto kconv = normal(std::size_t(4) * KVW, 0.4F), vconv = normal(std::size_t(4) * KVW, 0.4F);
    auto kring = normal(std::size_t(R) * KVW, 1.0F), vring = normal(std::size_t(R) * KVW, 1.0F);
    const std::size_t kv_words = full ? std::size_t(NKV) * kPoolPages * U * D : std::size_t(NKV) * pages * U * D;
    auto kv_k = normal(kv_words, 1.0F), kv_v = normal(kv_words, 1.0F);
    for (auto& x : kv_v) { x = h16(f(x)); }
    std::vector<int> block_table;
    if (full) {
        std::vector<int> order(kPoolPages);
        for (int i = 0; i < kPoolPages; ++i) { order[i] = i; }
        std::shuffle(order.begin(), order.end(), gen);
        block_table.assign(kTableStride, -1);
        for (int b = 0; b <= (P0 + T - 1) / U; ++b) { block_table[b] = order[b]; }
    }
    const auto row_of = [&](int head, int pos) -> std::size_t {
        if (!full) { return ((std::size_t(head) * pages + (pos / U) % pages) * U + pos % U) * D; }
        return ((std::size_t(head) * kPoolPages + block_table[pos / U]) * U + pos % U) * D;
    };
    // Device state.
    std::uint16_t *dq = upload(q), *dr = upload(r), *dk = upload(k), *dv = upload(v), *dsink = upload(sink);
    std::uint16_t *dgate = upload(gate), *dktaps = upload(ktaps), *dvtaps = upload(vtaps);
    std::uint16_t *dqs = upload(qscale), *drs = upload(rscale), *dtable = upload(table), *dkc = upload(kconv),
                  *dvc = upload(vconv);
    std::uint16_t *dkr = upload(kring), *dvr = upload(vring), *dkk = upload(kv_k), *dvv = upload(kv_v);
    std::uint16_t *dkh = dalloc<std::uint16_t>(std::size_t(ik::kHistorySlots) * KVW),
                  *dvh = dalloc<std::uint16_t>(std::size_t(ik::kHistorySlots) * KVW);
    std::uint16_t* dout = dalloc<std::uint16_t>(std::size_t(T) * QW);
    int* dbt = full ? upload(block_table) : dalloc<int>(1);
    int *dlane = dalloc<int>(T), *dpos = dalloc<int>(T), *dfirst = upload(std::vector<int>{First});
    int* dgroups = dalloc<int>(3 * (T / 16 + 2));
    void* dws = dalloc<unsigned char>(ik::attention_workspace_bytes(T));
    const auto reset = [&] {
        upload_to(dkk, kv_k), upload_to(dvv, kv_v), upload_to(dkr, kring), upload_to(dvr, vring);
    };
    struct Got {
        std::vector<std::uint16_t> out, q, r, kk, vv;
    };
    // Columns [c0, c0 + n) of the call as one prompt call.
    const auto run_prompt = [&](int c0, int n) {
        // The history snapshot: ring rows of positions P0 + c0 - 4 .. - 1 at slots pos & 3.
        const auto kr = download(dkr, kring.size()), vr = download(dvr, vring.size());
        std::vector<std::uint16_t> kh(std::size_t(ik::kHistorySlots) * KVW), vh(kh.size());
        for (int back = 1; back <= ik::kHistorySlots; ++back) {
            const int pos = P0 + c0 - back;
            std::copy_n(kr.data() + std::size_t(pos & (R - 1)) * KVW, KVW, kh.data() + std::size_t(pos & 3) * KVW);
            std::copy_n(vr.data() + std::size_t(pos & (R - 1)) * KVW, KVW, vh.data() + std::size_t(pos & 3) * KVW);
        }
        upload_to(dkh, kh), upload_to(dvh, vh);
        const auto slice = [&](const std::vector<std::uint16_t>& src, int width) {
            return std::vector<std::uint16_t>(src.begin() + std::size_t(c0) * width, src.begin() + std::size_t(c0 + n) * width);
        };
        upload_to(dq, slice(q, QW)), upload_to(dr, slice(r, RW)), upload_to(dk, slice(k, KVW)), upload_to(dv, slice(v, KVW));
        upload_to(dsink, slice(sink, NH)), upload_to(dgate, slice(gate, QW));
        upload_to(dktaps, slice(ktaps, NKV * 4)), upload_to(dvtaps, slice(vtaps, NKV * 4));
        std::vector<int> lane(n, 0), pos(n), groups;
        for (int i = 0; i < n; ++i) { pos[i] = P0 + c0 + i; }
        for (int done = 0; done < n; done += ik::kMaxAttentionLaneColumns) {
            groups.insert(groups.end(), {done, std::min(ik::kMaxAttentionLaneColumns, n - done), 0});
        }
        upload_to(dlane, lane), upload_to(dpos, pos), upload_to(dgroups, groups);
        ik::AttentionArgs a;
        a.columns           = n;
        a.groups            = static_cast<int>(groups.size() / 3);
        a.max_group_columns = std::min(n, ik::kMaxAttentionLaneColumns);
        a.map               = {dlane, dpos, dfirst, n};
        a.groups_columns    = dgroups;
        a.host_groups       = groups.data();
        a.in                = {dq, dr, dk, dv, dsink, dgate, full ? dktaps : nullptr, full ? dvtaps : nullptr};
        a.layer             = {full, dqs, drs, dtable, dkc, dvc};
        a.key_ring          = {dkr, KVW, std::int64_t{R} * KVW, dkh, std::int64_t{ik::kHistorySlots} * KVW};
        a.value_ring        = {dvr, KVW, std::int64_t{R} * KVW, dvh, std::int64_t{ik::kHistorySlots} * KVW};
        a.kv.k              = dkk;
        a.kv.v              = dvv;
        a.kv.ring_pages     = pages;
        a.kv.lane_stride    = std::int64_t{NKV} * pages * U * D;
        a.kv.table          = full ? dbt : nullptr;
        a.kv.table_stride   = full ? kTableStride : 0;
        a.kv.pool_pages     = full ? kPoolPages : 0;
        a.ws                = ik::carve_attention_workspace(dws, T);
        a.out               = dout;
        ik::attention_prompt(a, nullptr);
        cuda_check(cudaDeviceSynchronize(), "attention_prompt");
        Got g;
        g.out = download(dout, std::size_t(n) * QW);
        g.q   = download(a.ws.q, std::size_t(n) * QW);
        g.r   = download(a.ws.r, std::size_t(n) * RW);
        g.kk  = download(dkk, kv_k.size());
        g.vv  = download(dvv, kv_v.size());
        return g;
    };
    reset();
    const Got got = run_prompt(0, T);
    // FP64 over the kernel's own q, r and represented K/V.
    Tally tally{full ? "full prompt attention vs FP64" : "sliding prompt attention vs FP64"};
    for (int t = 0; t < T; ++t) {
        const int pos = P0 + t;
        for (int h = 0; h < NH; ++h) {
            const int kvh = h / G;
            const std::uint16_t* qv = got.q.data() + (std::size_t(t) * NH + h) * D;
            const std::uint16_t* rv = got.r.data() + (std::size_t(t) * NH + h) * RD;
            const int lo = std::max(First, full ? 0 : std::max(0, pos - (ik::kWindow - 1)));
            std::vector<double> score;
            std::vector<std::size_t> rows;
            double vmax = 0;
            for (int j = lo; j <= pos; ++j) {
                const std::size_t at = row_of(kvh, j);
                double dot = 0;
                for (int e = 0; e < D; ++e) { dot += double{f(qv[e])} * f(got.kk[at + e]); }
                const int dist = pos - j;
                double bias    = 0;
                if (dist < ik::kRelativeExtent) {
                    float acc = 0;
                    for (int i = 0; i < RD; ++i) {
                        acc = canon::fma_rn(f(rv[i]), f(table[std::size_t(i) * ik::kRelativeExtent + dist]), acc);
                    }
                    bias = bf(acc);
                }
                score.push_back(dot / std::sqrt(128.0) + bias);
                rows.push_back(at);
                for (int e = 0; e < D; ++e) { vmax = std::max(vmax, std::fabs(double{hf(got.vv[at + e])})); }
            }
            const double s_sink = f(sink[std::size_t(t) * NH + h]);
            double m = s_sink;
            for (double s : score) { m = std::max(m, s); }
            double den = std::exp(s_sink - m);
            for (double& s : score) { den += (s = std::exp(s - m)); }
            for (int e = 0; e < D; ++e) {
                double acc = 0;
                for (std::size_t j = 0; j < score.size(); ++j) { acc += score[j] * hf(got.vv[rows[j] + e]); }
                const double a64 = acc / den;
                const double sg  = 1.0 / (1.0 + std::exp(-double{f(gate[std::size_t(t) * QW + h * D + e])}));
                const double y   = f(got.out[std::size_t(t) * QW + h * D + e]);
                tally.ratio(std::fabs(y - a64 * sg) /
                            (0.5 * ulp_bf16(a64 * sg) + sg * (ulp_bf16(a64) + 0x1p-11 * vmax) + 0x1p-16 * vmax));
            }
        }
    }
    tally.report();
    reset();
    const Got again = run_prompt(0, T);
    Tally repeat{"  repeated prompt call"};
    repeat.exact(again.out.data(), got.out.data(), got.out.size());
    repeat.report();
    reset();
    Tally chunks{full ? "  full: calls of 37, 64, 49 = one call" : "  sliding: calls of 37, 64, 49 = one call"};
    int c0 = 0;
    for (const int n : {37, 64, 49}) {
        const Got part = run_prompt(c0, n);
        chunks.exact(part.out.data(), got.out.data() + std::size_t(c0) * QW, std::size_t(n) * QW);
        c0 += n;
    }
    chunks.report();
    for (void* x : {static_cast<void*>(dq), static_cast<void*>(dr), static_cast<void*>(dk), static_cast<void*>(dv),
                    static_cast<void*>(dsink), static_cast<void*>(dgate), static_cast<void*>(dktaps),
                    static_cast<void*>(dvtaps), static_cast<void*>(dqs), static_cast<void*>(drs), static_cast<void*>(dtable),
                    static_cast<void*>(dkc), static_cast<void*>(dvc), static_cast<void*>(dkr), static_cast<void*>(dvr),
                    static_cast<void*>(dkk), static_cast<void*>(dvv), static_cast<void*>(dkh), static_cast<void*>(dvh),
                    static_cast<void*>(dout), static_cast<void*>(dbt), static_cast<void*>(dlane), static_cast<void*>(dpos),
                    static_cast<void*>(dfirst), static_cast<void*>(dgroups), dws}) {
        cudaFree(x);
    }
}

} // namespace

INFERNIX_OP_TEST(inkling2_attention) {
    run_case(false, 7);
    run_case(true, 8);
    prompt_case(false, 17);
    prompt_case(true, 18);
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
