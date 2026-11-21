// Inkling 2's DSpark drafter Ops (include/infernix/ops/inkling2_dspark.h) against FP64 oracles of the reference
// arithmetic (.local's DSPARK_SPEC). The BF16 roundings the reference defines (mix weights, the head norm before RoPE,
// cos and sin, the Markov head's u, g, state, latent, correction, sum and capped logit) are semantic boundaries the
// oracle applies too; everything else is FP64. Criteria (stated before the run):
//   mix, rotate, add_norm, swiglu: |gpu - ref| <= 2^-8 |ref| + 2^-10 max|ref| (half a BF16 ulp, the FP32 sums, a
//     boundary flip one ulp upstream);
//   attention: |gpu - ref| <= 2^-8 |ref| + 2^-12 max|v|;
//   rotate's values and the residual copy: bit for bit;
//   Markov: each draft equals the oracle's argmax (lowest id on ties) until the oracle's top two capped logits lie
//     within 2 BF16 ulps (a near tie either side may take; later drafts then follow another path), each probability
//     within 2^-6 of the oracle's.

#include "infernix/ops/inkling2_dspark.h"
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
namespace ik    = infernix::ops::inkling2;
using infernix::test::cuda_check;

namespace {

constexpr int H = 2304, D = 128;
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
template <class T>
T* zeros(std::size_t n) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    cuda_check(cudaMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMemset");
    return p;
}

double bf(std::uint16_t v) { return canon::bf16_to_f32(v); }
std::uint16_t rb(double x) { return canon::f32_to_bf16_rn(static_cast<float>(x)); } // via FP32 (as torch's copy_)
double rbd(double x) { return bf(rb(x)); }

std::vector<std::uint16_t> random_bf16(std::size_t n, std::mt19937& rng, double scale, double offset = 0.0) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<std::uint16_t> out(n);
    for (auto& v : out) { v = rb(offset + scale * normal(rng)); }
    return out;
}

// The criterion's worst ratio |gpu - ref| / (rel |ref| + abs).
struct Worst {
    double ratio = 0;
    void add(double gpu, double ref, double rel, double abs) {
        ratio = std::max(ratio, std::fabs(gpu - ref) / (rel * std::fabs(ref) + abs));
    }
};

double rms_inv(const double* x, int n, double eps) {
    double s = 0;
    for (int i = 0; i < n; ++i) { s += x[i] * x[i]; }
    return 1.0 / std::sqrt(s / n + eps);
}

// ---------------------------------------------------------------------------- mix

void mix_case() {
    std::mt19937 rng(11);
    constexpr int C = 5, kRing = 16;
    const auto features = random_bf16(static_cast<std::size_t>(C) * 4 * H, rng, 1.0);
    const auto fuse     = random_bf16(16, rng, 1.0);
    std::vector<std::vector<std::uint16_t>> norms;
    for (int m = 0; m < 4; ++m) { norms.push_back(random_bf16(H, rng, 0.2, 1.0)); }
    const std::vector<std::int32_t> positions = {100, 101, 102, 103, 104}, lanes = {0, 1, 0, 1, 0};
    ik::DSparkMixArgs a;
    a.features    = upload(features);
    a.columns     = C;
    a.fuse_weight = upload(fuse);
    a.count       = 4;
    std::vector<std::uint16_t*> outs;
    for (int m = 0; m < 4; ++m) {
        a.rows[m]  = m;
        a.norms[m] = upload(norms[m]);
        outs.push_back(zeros<std::uint16_t>(m < 3 ? static_cast<std::size_t>(C) * H : 2ULL * kRing * H));
        a.out[m] = outs.back();
    }
    a.ring[3]        = kRing;
    a.positions      = upload(positions);
    a.lanes          = upload(lanes);
    a.lane_stride[3] = static_cast<std::int64_t>(kRing) * H;
    ik::dspark_mix(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "mix");
    Worst worst;
    for (int m = 0; m < 4; ++m) {
        double l[4], top = -1e300, sum = 0, alpha[4];
        for (int j = 0; j < 4; ++j) { top = std::max(top, l[j] = bf(fuse[m * 4 + j])); }
        for (int j = 0; j < 4; ++j) { sum += std::exp(l[j] - top); }
        for (int j = 0; j < 4; ++j) { alpha[j] = rbd(std::exp(l[j] - top) / sum); }
        const auto got = download(outs[m], m < 3 ? static_cast<std::size_t>(C) * H : 2ULL * kRing * H);
        for (int c = 0; c < C; ++c) {
            std::vector<double> mix(H);
            for (int e = 0; e < H; ++e) {
                for (int j = 0; j < 4; ++j) { mix[e] += bf(features[(static_cast<std::size_t>(c) * 4 + j) * H + e]) * alpha[j]; }
            }
            const double inv = rms_inv(mix.data(), H, 1e-6);
            double peak      = 0;
            std::vector<double> ref(H);
            for (int e = 0; e < H; ++e) { peak = std::max(peak, std::fabs(ref[e] = mix[e] * inv * bf(norms[m][e]))); }
            const std::size_t row = m < 3 ? static_cast<std::size_t>(c)
                                          : static_cast<std::size_t>(lanes[c]) * kRing + positions[c] % kRing;
            for (int e = 0; e < H; ++e) { worst.add(bf(got[row * H + e]), ref[e], std::ldexp(1.0, -8), std::ldexp(peak, -10)); }
        }
    }
    std::printf("dspark_mix (5 columns, 3 dense consumers + the Markov ring of 2 lanes): worst %.3f of the bound\n",
                worst.ratio);
    check(worst.ratio <= 1.0, "dspark_mix outside its bound");
}

// ---------------------------------------------------------------------------- rotate

void rotate_reference(const std::uint16_t* x, const std::vector<std::uint16_t>& w, int position, double* out) {
    double v[D];
    for (int d = 0; d < D; ++d) { v[d] = bf(x[d]); }
    const double inv = rms_inv(v, D, 1e-6);
    double n[D];
    for (int d = 0; d < D; ++d) { n[d] = rbd(v[d] * inv * bf(w[d])); }
    for (int i = 0; i < D / 2; ++i) {
        const float inv_f = static_cast<float>(std::pow(1.0e6, -static_cast<double>(i) / (D / 2)));
        const float angle = static_cast<float>(position) * inv_f;
        const double c = rbd(std::cos(static_cast<double>(angle))), s = rbd(std::sin(static_cast<double>(angle)));
        out[i]         = n[i] * c - n[i + 64] * s;
        out[i + 64]    = n[i + 64] * c + n[i] * s;
    }
}

void rotate_case() {
    std::mt19937 rng(12);
    constexpr int R = 8, kRing = 1152;
    // Queries: dense, positions from the round's first (1,000).
    const auto q  = random_bf16(static_cast<std::size_t>(R) * 24 * D, rng, 1.0);
    const auto qw = random_bf16(D, rng, 0.2, 1.0);
    ik::DSparkRound round{1000, 0, 1};
    ik::DSparkRound* d_round = upload(std::vector<ik::DSparkRound>{round});
    float table[D / 2];
    ik::dspark_inv_freq(1.0e6, table);
    const float* inv_freq = upload(std::vector<float>(table, table + D / 2));
    std::uint16_t* dense     = zeros<std::uint16_t>(static_cast<std::size_t>(R) * 24 * D);
    ik::DSparkRotateArgs a;
    a.x        = upload(q);
    a.x_stride = 24 * D;
    a.rows     = R;
    a.heads    = 24;
    a.norm     = upload(qw);
    a.round    = d_round;
    a.inv_freq = inv_freq;
    a.dense    = dense;
    ik::dspark_rotate(a, nullptr);
    // Keys: into lane 1 of a ring at explicit positions (past the ring's wrap), values beside them.
    const auto k = random_bf16(static_cast<std::size_t>(R) * 8 * D, rng, 1.0);
    const auto v = random_bf16(static_cast<std::size_t>(R) * 8 * D, rng, 1.0);
    const auto kw = random_bf16(D, rng, 0.2, 1.0);
    std::vector<std::int32_t> positions(R);
    for (int r = 0; r < R; ++r) { positions[r] = 5000 + r; }
    const std::int64_t lane_words = 8LL * kRing * D;
    std::uint16_t* cache_k = zeros<std::uint16_t>(2 * lane_words);
    std::uint16_t* cache_v = zeros<std::uint16_t>(2 * lane_words);
    ik::DSparkRotateArgs b;
    b.x             = upload(k);
    b.x_stride      = 8 * D;
    b.values        = upload(v);
    b.values_stride = 8 * D;
    b.rows          = R;
    b.heads         = 8;
    b.norm          = upload(kw);
    b.positions     = upload(positions);
    b.inv_freq      = inv_freq;
    b.cache_k       = cache_k;
    b.cache_v       = cache_v;
    b.lane_stride   = lane_words;
    b.lane          = 1;
    b.ring          = kRing;
    ik::dspark_rotate(b, nullptr);
    cuda_check(cudaDeviceSynchronize(), "rotate");
    Worst worst;
    const auto got_q = download(dense, static_cast<std::size_t>(R) * 24 * D);
    double ref[D];
    for (int r = 0; r < R; ++r) {
        for (int h = 0; h < 24; ++h) {
            rotate_reference(q.data() + (static_cast<std::size_t>(r) * 24 + h) * D, qw, 1000 + r, ref);
            double peak = 0;
            for (double x : ref) { peak = std::max(peak, std::fabs(x)); }
            for (int d = 0; d < D; ++d) {
                worst.add(bf(got_q[(static_cast<std::size_t>(r) * 24 + h) * D + d]), ref[d], std::ldexp(1.0, -8),
                          std::ldexp(peak, -10));
            }
        }
    }
    const auto got_k = download(cache_k, 2 * lane_words), got_v = download(cache_v, 2 * lane_words);
    bool values_exact = true;
    for (int r = 0; r < R; ++r) {
        for (int h = 0; h < 8; ++h) {
            rotate_reference(k.data() + (static_cast<std::size_t>(r) * 8 + h) * D, kw, positions[r], ref);
            double peak = 0;
            for (double x : ref) { peak = std::max(peak, std::fabs(x)); }
            const std::size_t at = lane_words + (static_cast<std::size_t>(h) * kRing + positions[r] % kRing) * D;
            for (int d = 0; d < D; ++d) {
                worst.add(bf(got_k[at + d]), ref[d], std::ldexp(1.0, -8), std::ldexp(peak, -10));
                values_exact = values_exact && got_v[at + d] == v[(static_cast<std::size_t>(r) * 8 + h) * D + d];
            }
        }
    }
    std::printf("dspark_rotate (24 query heads dense at 1000-1007; 8 key heads into lane 1's ring at 5000-5007): worst "
                "%.3f of the bound; values %s\n",
                worst.ratio, values_exact ? "bit for bit" : "DIFFER");
    check(worst.ratio <= 1.0, "dspark_rotate outside its bound");
    check(values_exact, "dspark_rotate's values differ from their source");
}

// ---------------------------------------------------------------------------- attention

void attention_case(bool sliding, int first) {
    std::mt19937 rng(sliding ? 13 : 14);
    constexpr int kMaxContext = 2048;
    const int slots  = sliding ? ik::kDSparkRing : (kMaxContext + 16 + 63) / 64 * 64;
    const int splits = sliding ? (ik::kDSparkWindow + 8 + 127) / 128 : (kMaxContext + 8 + 127) / 128;
    const std::int64_t lane_words = 8LL * slots * D;
    const int lo = sliding ? std::max(0, first - ik::kDSparkWindow) : 0, hi = first + 8;
    // Lane 1's keys and values at positions [lo, hi) (each at its slot), the rest of the cache random too.
    const auto k = random_bf16(2 * static_cast<std::size_t>(lane_words), rng, 1.0);
    const auto v = random_bf16(2 * static_cast<std::size_t>(lane_words), rng, 1.0);
    const auto q = random_bf16(8ULL * 24 * D, rng, 0.35);
    ik::DSparkAttentionArgs a;
    a.q           = upload(q);
    a.cache_k     = upload(k);
    a.cache_v     = upload(v);
    a.lane_stride = lane_words;
    a.ring        = sliding ? ik::kDSparkRing : 0;
    a.slots_full  = slots;
    a.splits      = splits;
    a.round       = upload(std::vector<ik::DSparkRound>{{first, 0, 1}});
    a.scratch     = zeros<float>(ik::dspark_attention_scratch_bytes(splits) / 4);
    a.out         = zeros<std::uint16_t>(8ULL * 24 * D);
    ik::dspark_attention(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "attention");
    const auto got = download(a.out, 8ULL * 24 * D);
    double vmax    = 0;
    for (const auto x : v) { vmax = std::max(vmax, std::fabs(bf(x))); }
    Worst worst;
    for (int row = 0; row < 8; ++row) {
        for (int h = 0; h < 24; ++h) {
            const int g = h / 3;
            std::vector<double> s;
            double top = -1e300;
            for (int p = lo; p < hi; ++p) {
                const int slot = sliding ? p % ik::kDSparkRing : p;
                double dot     = 0;
                for (int d = 0; d < D; ++d) {
                    dot += bf(q[(static_cast<std::size_t>(row) * 24 + h) * D + d]) *
                           bf(k[lane_words + (static_cast<std::size_t>(g) * slots + slot) * D + d]);
                }
                s.push_back(dot / std::sqrt(128.0));
                top = std::max(top, s.back());
            }
            double sum = 0;
            for (double& x : s) { sum += x = std::exp(x - top); }
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int p = lo; p < hi; ++p) {
                    const int slot = sliding ? p % ik::kDSparkRing : p;
                    o += s[static_cast<std::size_t>(p - lo)] * bf(v[lane_words + (static_cast<std::size_t>(g) * slots + slot) * D + d]);
                }
                worst.add(bf(got[(static_cast<std::size_t>(row) * 24 + h) * D + d]), o / sum, std::ldexp(1.0, -8),
                          std::ldexp(vmax, -12));
            }
        }
    }
    std::printf("dspark_attention (%s, F = %d: keys %d-%d): worst %.3f of the bound\n", sliding ? "sliding" : "full", first,
                lo, hi - 1, worst.ratio);
    check(worst.ratio <= 1.0, "dspark_attention outside its bound");
}

// ---------------------------------------------------------------------------- add_norm, swiglu

void add_norm_case() {
    std::mt19937 rng(15);
    constexpr int R = 8;
    const auto residual = random_bf16(static_cast<std::size_t>(R) * H, rng, 1.0);
    const auto x        = random_bf16(static_cast<std::size_t>(R) * H, rng, 1.0);
    const auto w        = random_bf16(H, rng, 0.2, 1.0);
    ik::DSparkAddNormArgs a;
    a.residual = upload(residual);
    a.x        = upload(x);
    a.norm     = upload(w);
    a.normed   = zeros<std::uint16_t>(static_cast<std::size_t>(R) * H);
    a.rows     = R;
    ik::dspark_add_norm(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "add_norm");
    const auto got_r = download(a.residual, static_cast<std::size_t>(R) * H);
    const auto got_n = download(a.normed, static_cast<std::size_t>(R) * H);
    Worst worst;
    int residual_off = 0;
    for (int r = 0; r < R; ++r) {
        std::vector<double> s(H);
        for (int e = 0; e < H; ++e) {
            s[e] = bf(residual[static_cast<std::size_t>(r) * H + e]) + bf(x[static_cast<std::size_t>(r) * H + e]);
            const double gpu = bf(got_r[static_cast<std::size_t>(r) * H + e]);
            residual_off += std::fabs(gpu - s[e]) > std::ldexp(std::fabs(s[e]), -8) ? 1 : 0;
        }
        const double inv = rms_inv(s.data(), H, 1e-6);
        double peak      = 0;
        for (int e = 0; e < H; ++e) { peak = std::max(peak, std::fabs(s[e] * inv * bf(w[e]))); }
        for (int e = 0; e < H; ++e) {
            worst.add(bf(got_n[static_cast<std::size_t>(r) * H + e]), s[e] * inv * bf(w[e]), std::ldexp(1.0, -8),
                      std::ldexp(peak, -10));
        }
    }
    std::printf("dspark_add_norm (8 rows): normed worst %.3f of the bound; %d residuals beyond half a BF16 ulp\n",
                worst.ratio, residual_off);
    check(worst.ratio <= 1.0 && residual_off == 0, "dspark_add_norm outside its bound");
}

void swiglu_case() {
    std::mt19937 rng(16);
    constexpr int R = 8, W = ik::kDSparkIntermediate;
    const auto g = random_bf16(static_cast<std::size_t>(R) * W, rng, 2.0);
    const auto u = random_bf16(static_cast<std::size_t>(R) * W, rng, 2.0);
    std::uint16_t* h = zeros<std::uint16_t>(static_cast<std::size_t>(R) * W);
    ik::dspark_swiglu(upload(g), upload(u), R, W, h, nullptr);
    cuda_check(cudaDeviceSynchronize(), "swiglu");
    const auto got = download(h, static_cast<std::size_t>(R) * W);
    Worst worst;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double gv = bf(g[i]), ref = gv / (1.0 + std::exp(-gv)) * bf(u[i]);
        worst.add(bf(got[i]), ref, std::ldexp(1.0, -8), 1e-30);
    }
    std::printf("dspark_swiglu (8 x 9216): worst %.3f of the bound\n", worst.ratio);
    check(worst.ratio <= 1.0, "dspark_swiglu outside its bound");
}

// ---------------------------------------------------------------------------- Markov head

void markov_case() {
    std::mt19937 rng(17);
    constexpr int V = 6000, R3 = 768, K = 5120, kRank = 256, kRing = 16, kSteps = 7;
    const int first = 205, anchor = 4321;
    const auto proj  = random_bf16(static_cast<std::size_t>(R3) * K, rng, 0.02);
    const auto bias  = random_bf16(R3, rng, 0.05);
    const auto block = random_bf16(8ULL * H, rng, 1.0);
    const auto ring  = random_bf16(static_cast<std::size_t>(kRing) * H, rng, 1.0);
    const auto w1    = random_bf16(static_cast<std::size_t>(V) * kRank, rng, 1.0);
    const auto fold  = random_bf16(static_cast<std::size_t>(V) * kRank, rng, 0.15);
    const auto fbias = random_bf16(V, rng, 0.1);
    std::vector<float> base(static_cast<std::size_t>(kSteps) * V);
    {
        std::normal_distribution<float> normal(0.0F, 3.0F);
        for (float& b : base) { b = normal(rng); }
    }
    ik::DSparkMarkovArgs a;
    a.projection    = upload(proj);
    a.bias          = upload(bias);
    a.block         = upload(block);
    a.f_ring        = upload(ring);
    a.f_ring_rows   = kRing;
    a.f_lane_stride = static_cast<std::int64_t>(kRing) * H;
    a.embedding     = upload(w1);
    a.fold          = upload(fold);
    a.fold_bias     = upload(fbias);
    a.base          = upload(base);
    a.vocab         = V;
    a.round         = upload(std::vector<ik::DSparkRound>{{first, anchor, 0}});
    a.tokens        = zeros<std::int32_t>(kSteps);
    a.probabilities = zeros<float>(kSteps);
    a.scratch       = zeros<float>(ik::dspark_markov_scratch_bytes(V) / 4 + 1);
    // Twice: the second call must find the scratch's counters reset and give the same drafts.
    ik::dspark_markov(a, kSteps, nullptr);
    cuda_check(cudaDeviceSynchronize(), "markov");
    const auto tokens = download(a.tokens, kSteps);
    const auto probs  = download(a.probabilities, kSteps);
    ik::dspark_markov(a, kSteps, nullptr);
    cuda_check(cudaDeviceSynchronize(), "markov");
    const bool repeat = download(a.tokens, kSteps) == tokens && download(a.probabilities, kSteps) == probs;
    // The oracle.
    const std::uint16_t* f = ring.data() + static_cast<std::size_t>((first - 1) % kRing) * H;
    std::vector<double> s(kRank, 0.0);
    int prev = anchor, compared = 0;
    bool agree = true, near_tie = false;
    double worst_prob = 0;
    for (int step = 1; step <= kSteps && agree && !near_tie; ++step) {
        std::vector<double> u(R3);
        for (int r = 0; r < R3; ++r) {
            const std::uint16_t* w = proj.data() + static_cast<std::size_t>(r) * K;
            double acc = bf(bias[r]);
            for (int k = 0; k < kRank; ++k) { acc += bf(w[k]) * s[k]; }
            for (int k = 0; k < kRank; ++k) { acc += bf(w[kRank + k]) * bf(w1[static_cast<std::size_t>(prev) * kRank + k]); }
            for (int k = 0; k < H; ++k) { acc += bf(w[2 * kRank + k]) * bf(block[static_cast<std::size_t>(step) * H + k]); }
            for (int k = 0; k < H; ++k) { acc += bf(w[2 * kRank + H + k]) * bf(f[k]); }
            u[r] = rbd(acc);
        }
        std::vector<double> lat(kRank);
        for (int c = 0; c < kRank; ++c) {
            const double g     = rbd(1.0 / (1.0 + std::exp(-u[c])));
            const double kept  = rbd(g * s[c]);
            const double blend = rbd(rbd(1.0 - g) * rbd(std::tanh(u[kRank + c])));
            s[c]               = rbd(kept + blend);
            lat[c]             = rbd(std::tanh(u[2 * kRank + c]));
        }
        std::vector<double> L(V);
        int best = 0;
        for (int vv = 0; vv < V; ++vv) {
            double acc = bf(fbias[vv]);
            for (int c = 0; c < kRank; ++c) { acc += bf(fold[static_cast<std::size_t>(vv) * kRank + c]) * lat[c]; }
            const double z = rbd(rbd(base[static_cast<std::size_t>(step - 1) * V + vv]) + rbd(acc));
            L[vv]          = rbd(30.0 * std::tanh(z / 30.0));
            if (L[vv] > L[best]) { best = vv; }
        }
        double second = -1e300;
        for (int vv = 0; vv < V; ++vv) {
            if (vv != best) { second = std::max(second, L[vv]); }
        }
        double sum = 0;
        for (const double l : L) { sum += std::exp(l - L[best]); }
        const double ulp = std::ldexp(1.0, std::ilogb(std::max(std::fabs(L[best]), 1e-30)) - 7);
        if (L[best] - second <= 2 * ulp) {
            near_tie = true;
            break;
        }
        ++compared;
        agree      = tokens[static_cast<std::size_t>(step - 1)] == best;
        worst_prob = std::max(worst_prob, std::fabs(probs[static_cast<std::size_t>(step - 1)] - 1.0 / sum) / (1.0 / sum));
        prev       = best;
    }
    std::printf("dspark_markov (V = 6000, 7 steps): %d drafts compared%s, %s; worst probability %.2e relative; repeat "
                "%s\n",
                compared, near_tie ? " (then a near tie)" : "", agree ? "all equal the oracle's argmax" : "a draft DIFFERS",
                worst_prob, repeat ? "identical" : "DIFFERS");
    check(agree, "dspark_markov drafts another token than the oracle's argmax");
    check(compared >= 1, "dspark_markov: no draft compared (a near tie at the first step)");
    check(worst_prob <= std::ldexp(1.0, -6), "dspark_markov's probability is outside its bound");
    check(repeat, "a repeated dspark_markov call differs");
}

} // namespace

INFERNIX_OP_TEST(inkling2_dspark) {
    mix_case();
    rotate_case();
    attention_case(true, 1500);
    attention_case(true, 40);
    attention_case(false, 1500);
    attention_case(false, 3);
    add_norm_case();
    swiglu_case();
    markov_case();
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
