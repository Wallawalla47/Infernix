// Inkling 2's DSpark drafter Ops (include/infernix/ops/inkling2_dspark.h) against FP64 oracles of the reference
// arithmetic (.local's DSPARK_SPEC). The BF16 roundings the reference defines (mix weights, the head norm before RoPE,
// cos and sin, the Markov head's u, g, state, latent, correction, sum and capped logit) are semantic boundaries the
// oracle applies too; everything else is FP64. Criteria (stated before the run):
//   mix, rotate, add_norm, swiglu: |gpu - ref| <= 2^-8 |ref| + 2^-10 max|ref| (half a BF16 ulp, the FP32 sums, a
//     boundary flip one ulp upstream);
//   attention: |gpu - ref| <= 2^-8 |ref| + 2^-12 max|v| (sliding with the ring wrapped, full up to 20,008 keys; the
//     full layer in INT8 the same over its represented values bf16(code * s));
//   rotate's values and the residual copy: bit for bit; its INT8 codes and scales the host codec's, bit for bit;
//   Markov: each draft equals the oracle's argmax (lowest id on ties) until the oracle's top two capped logits lie
//     within 2 BF16 ulps (a near tie either side may take; later drafts then follow another path), each probability
//     within 2^-6 of the oracle's.

#include "infernix/ops/inkling2_dspark.h"
#include "ops/common/canonical_math.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
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

// The target pools' int8 row codec (kv_codec.cuh kv_encode_int8), an exact host oracle: per 64-value group s =
// fp16(max |x| / 127) (FP32 division), code = clamp(rint(x * fl32(1 / s)), +-127); 128 codes and two FP16 scales a row.
void int8_encode_row(const float* x, std::uint8_t* codes, std::uint16_t* side) {
    for (int g = 0; g < 2; ++g) {
        float m = 0.0F;
        for (int i = 0; i < 64; ++i) { m = std::max(m, std::fabs(x[64 * g + i])); }
        const __half h      = __float2half_rn(m > 0.0F ? m / 127.0F : 0.0F);
        const float s       = __half2float(h);
        const float inverse = s > 0.0F ? 1.0F / s : 0.0F;
        for (int i = 0; i < 64; ++i) {
            const float product = x[64 * g + i] * inverse;
            const int q         = inverse == 0.0F ? 0 : std::max(-127, std::min(127, static_cast<int>(std::nearbyint(product))));
            codes[64 * g + i]   = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        }
        side[g] = static_cast<__half_raw>(h).x;
    }
}
// An int8 row value's represented word: bf16(code * s) (the product exact in FP32).
double int8_value(std::uint8_t code, std::uint16_t scale_bits) {
    __half_raw raw;
    raw.x = scale_bits;
    return rbd(static_cast<double>(static_cast<std::int8_t>(code)) * static_cast<double>(__half2float(__half(raw))));
}

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

    // A call wider than the ring (a prompt's append): two lanes of 40 and 23 consecutive positions into a 16-row ring.
    // Each row must hold the lane's latest position that maps to it (columns that a later column of the lane overwrites
    // store nothing), the same in every call.
    constexpr int W = 63;
    const auto wide_features = random_bf16(static_cast<std::size_t>(W) * 4 * H, rng, 1.0);
    std::vector<std::int32_t> wide_positions, wide_lanes;
    for (int c = 0; c < W; ++c) {
        wide_lanes.push_back(c < 40 ? 0 : 1);
        wide_positions.push_back(c < 40 ? 300 + c : 1000 + (c - 40));
    }
    ik::DSparkMixArgs w = a;
    w.features  = upload(wide_features);
    w.columns   = W;
    w.count     = 1;
    w.rows[0]   = 3;
    w.norms[0]  = a.norms[3];
    w.ring[0]   = kRing;
    w.out[0]    = zeros<std::uint16_t>(2ULL * kRing * H);
    w.lane_stride[0] = static_cast<std::int64_t>(kRing) * H;
    w.positions = upload(wide_positions);
    w.lanes     = upload(wide_lanes);
    std::vector<std::uint16_t> first_ring;
    bool same = true;
    for (int call = 0; call < 3; ++call) {
        ik::dspark_mix(w, nullptr);
        cuda_check(cudaDeviceSynchronize(), "mix wide");
        const auto ring = download(w.out[0], 2ULL * kRing * H);
        if (call == 0) {
            first_ring = ring;
        } else {
            same = same && ring == first_ring;
        }
    }
    double l[4], top = -1e300, sum = 0, alpha[4];
    for (int j = 0; j < 4; ++j) { top = std::max(top, l[j] = bf(fuse[3 * 4 + j])); }
    for (int j = 0; j < 4; ++j) { sum += std::exp(l[j] - top); }
    for (int j = 0; j < 4; ++j) { alpha[j] = rbd(std::exp(l[j] - top) / sum); }
    Worst wide;
    for (int c = 0; c < W; ++c) {
        const int lane_last = wide_lanes[c] == 0 ? 39 : W - 1;
        if (c + kRing <= lane_last) { continue; } // a later column of the lane owns the row
        std::vector<double> mix(H), ref(H);
        for (int e = 0; e < H; ++e) {
            for (int j = 0; j < 4; ++j) { mix[e] += bf(wide_features[(static_cast<std::size_t>(c) * 4 + j) * H + e]) * alpha[j]; }
        }
        const double inv = rms_inv(mix.data(), H, 1e-6);
        double peak      = 0;
        for (int e = 0; e < H; ++e) { peak = std::max(peak, std::fabs(ref[e] = mix[e] * inv * bf(norms[3][e]))); }
        const std::size_t row = static_cast<std::size_t>(wide_lanes[c]) * kRing + wide_positions[c] % kRing;
        for (int e = 0; e < H; ++e) { wide.add(bf(first_ring[row * H + e]), ref[e], std::ldexp(1.0, -8), std::ldexp(peak, -10)); }
    }
    std::printf("dspark_mix (63 columns of 2 lanes into a 16-row ring): each row its lane's latest position, worst %.3f of "
                "the bound; three calls %s\n",
                wide.ratio, same ? "identical" : "DIFFER");
    check(wide.ratio <= 1.0 && same, "dspark_mix: a ring row does not hold its lane's latest position, or calls differ");
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
    {
        // A call wider than the ring (a long prompt's append): ring + 5 rows of lane 0 at consecutive positions. Every
        // slot holds the latest row that maps to it (the first 5 rows' slots their wrapped successors'), in each call.
        constexpr int RW = kRing + 5;
        const auto wk = random_bf16(static_cast<std::size_t>(RW) * 8 * D, rng, 1.0);
        const auto wv = random_bf16(static_cast<std::size_t>(RW) * 8 * D, rng, 1.0);
        std::vector<std::int32_t> wpos(RW);
        for (int r = 0; r < RW; ++r) { wpos[r] = 6000 + r; }
        ik::DSparkRotateArgs c = b;
        c.x         = upload(wk);
        c.values    = upload(wv);
        c.rows      = RW;
        c.positions = upload(wpos);
        c.lane      = 0;
        bool exact = true, same = true;
        std::vector<std::uint16_t> first_v;
        for (int call = 0; call < 3; ++call) {
            ik::dspark_rotate(c, nullptr);
            cuda_check(cudaDeviceSynchronize(), "rotate wide");
            const auto got = download(cache_v, lane_words);
            if (call == 0) {
                first_v = got;
            } else {
                same = same && got == first_v;
            }
        }
        for (int r = RW - kRing; r < RW; ++r) { // the latest row of each slot
            for (int h = 0; h < 8; ++h) {
                const std::size_t at = (static_cast<std::size_t>(h) * kRing + wpos[r] % kRing) * D;
                for (int d = 0; d < D; ++d) { exact = exact && first_v[at + d] == wv[(static_cast<std::size_t>(r) * 8 + h) * D + d]; }
            }
        }
        std::printf("dspark_rotate (%d rows of one lane into a %d-slot ring): each slot its latest row's values %s; three "
                    "calls %s\n",
                    RW, kRing, exact ? "bit for bit" : "DIFFER", same ? "identical" : "DIFFER");
        check(exact && same, "dspark_rotate: a ring slot does not hold its latest row, or calls differ");
    }
    // The full layer: the same keys into lane 1's pages (a page-major pool, a shuffled block-table row) at 4092-4099,
    // across a page edge: each key head within the bound of its reference, its values bit for bit, every other row
    // untouched.
    constexpr int kStride = 80, kPages = 2 * kStride + 2;
    std::vector<std::int32_t> block_rows(2 * kStride);
    for (std::size_t i = 0; i < block_rows.size(); ++i) { block_rows[i] = static_cast<std::int32_t>(i); }
    std::shuffle(block_rows.begin(), block_rows.end(), rng);
    const std::size_t pool_words = static_cast<std::size_t>(kPages) * 8 * 64 * D;
    std::uint16_t* pool_k = zeros<std::uint16_t>(pool_words);
    std::uint16_t* pool_v = zeros<std::uint16_t>(pool_words);
    for (int r = 0; r < R; ++r) { positions[r] = 4092 + r; }
    ik::DSparkRotateArgs f = b;
    f.positions    = upload(positions);
    f.cache_k      = pool_k;
    f.cache_v      = pool_v;
    f.lane_stride  = 0;
    f.ring         = 0;
    f.table        = upload(block_rows);
    f.table_stride = kStride;
    ik::dspark_rotate(f, nullptr);
    cuda_check(cudaDeviceSynchronize(), "rotate (paged)");
    const auto paged_k = download(pool_k, pool_words), paged_v = download(pool_v, pool_words);
    std::vector<bool> written(pool_words, false);
    bool paged_exact = true;
    for (int r = 0; r < R; ++r) {
        const std::size_t page = static_cast<std::size_t>(block_rows[kStride + positions[r] / 64]);
        for (int h = 0; h < 8; ++h) {
            rotate_reference(k.data() + (static_cast<std::size_t>(r) * 8 + h) * D, kw, positions[r], ref);
            const std::size_t at = ((page * 8 + h) * 64 + positions[r] % 64) * D;
            for (int d = 0; d < D; ++d) {
                written[at + d] = true;
                paged_exact = paged_exact && paged_v[at + d] == v[(static_cast<std::size_t>(r) * 8 + h) * D + d];
            }
            double peak = 0;
            for (double x : ref) { peak = std::max(peak, std::fabs(x)); }
            for (int d = 0; d < D; ++d) { worst.add(bf(paged_k[at + d]), ref[d], std::ldexp(1.0, -8), std::ldexp(peak, -10)); }
        }
    }
    for (std::size_t i = 0; i < pool_words; ++i) {
        paged_exact = paged_exact && (written[i] || (paged_k[i] == 0 && paged_v[i] == 0));
    }
    std::printf("dspark_rotate (8 key heads into lane 1's pages at 4092-4099): worst %.3f of the bound; values and "
                "untouched rows %s\n",
                worst.ratio, paged_exact ? "bit for bit" : "DIFFER");
    check(worst.ratio <= 1.0, "dspark_rotate (paged) outside its bound");
    check(paged_exact, "dspark_rotate (paged) wrote the wrong rows");
    // The full layer in INT8 (DSparkInt8Cache): the same rows' codes and side records equal the host codec's encoding of
    // the BF16 run's keys (its stored rounding) and of the values, bit for bit; every other row untouched (zero).
    const std::size_t pool_rows = static_cast<std::size_t>(kPages) * 8 * 64;
    ik::DSparkRotateArgs e = f;
    e.cache_k              = nullptr;
    e.cache_v              = nullptr;
    e.int8 = {zeros<std::uint8_t>(pool_rows * D), zeros<std::uint8_t>(pool_rows * D), zeros<std::uint8_t>(pool_rows * 4),
              zeros<std::uint8_t>(pool_rows * 4)};
    ik::dspark_rotate(e, nullptr);
    cuda_check(cudaDeviceSynchronize(), "rotate (int8)");
    const auto codes_k = download(e.int8.codes_k, pool_rows * D), codes_v = download(e.int8.codes_v, pool_rows * D);
    const auto side_k = download(e.int8.side_k, pool_rows * 4), side_v = download(e.int8.side_v, pool_rows * 4);
    std::vector<bool> encoded(pool_rows, false);
    bool int8_exact = true;
    for (int r = 0; r < R; ++r) {
        const std::size_t page = static_cast<std::size_t>(block_rows[kStride + positions[r] / 64]);
        for (int h = 0; h < 8; ++h) {
            const std::size_t row = (page * 8 + h) * 64 + positions[r] % 64;
            encoded[row]          = true;
            float key[D], value[D];
            for (int d = 0; d < D; ++d) {
                key[d]   = static_cast<float>(bf(paged_k[row * D + d]));
                value[d] = static_cast<float>(bf(v[(static_cast<std::size_t>(r) * 8 + h) * D + d]));
            }
            std::uint8_t want_k[D], want_v[D];
            std::uint16_t want_ks[2], want_vs[2];
            int8_encode_row(key, want_k, want_ks);
            int8_encode_row(value, want_v, want_vs);
            int8_exact = int8_exact && std::equal(want_k, want_k + D, codes_k.begin() + row * D) &&
                         std::equal(want_v, want_v + D, codes_v.begin() + row * D) &&
                         std::memcmp(want_ks, side_k.data() + row * 4, 4) == 0 &&
                         std::memcmp(want_vs, side_v.data() + row * 4, 4) == 0;
        }
    }
    for (std::size_t row = 0; row < pool_rows; ++row) {
        if (encoded[row]) { continue; }
        for (int i = 0; i < 4; ++i) { int8_exact = int8_exact && side_k[row * 4 + i] == 0 && side_v[row * 4 + i] == 0; }
    }
    std::printf("dspark_rotate (the same rows into INT8 pages): codes and scales the host codec's bit for bit, other rows "
                "untouched: %s\n",
                int8_exact ? "yes" : "NO");
    check(int8_exact, "dspark_rotate (int8) codes or scales differ from the codec's");
}

// ---------------------------------------------------------------------------- attention

void attention_case(bool sliding, int first, int kMaxContext = 2048) {
    std::mt19937 rng(sliding ? 13 : 14);
    const int splits = ik::dspark_attention_splits(sliding, static_cast<std::uint32_t>(kMaxContext));
    // Sliding: each lane's ring [8][ring][128]. Full: a page-major pool [pages][8][64][128] and two lanes' block-table
    // rows, lane 1's pages a shuffled set (so a key's page comes from the table, not its position).
    const int stride = (kMaxContext + 8 + 63) / 64, pages = 2 * stride + 3;
    const std::int64_t lane_words = sliding ? 8LL * ik::kDSparkRing * D : 0;
    std::vector<std::int32_t> table(2 * static_cast<std::size_t>(stride));
    for (std::size_t i = 0; i < table.size(); ++i) { table[i] = static_cast<std::int32_t>(i); }
    std::shuffle(table.begin(), table.end(), rng);
    const auto offset = [&](int g, int p) -> std::size_t {
        if (sliding) { return lane_words + (static_cast<std::size_t>(g) * ik::kDSparkRing + p % ik::kDSparkRing) * D; }
        const std::size_t page = static_cast<std::size_t>(table[static_cast<std::size_t>(stride) + p / 64]);
        return ((page * 8 + g) * 64 + p % 64) * D;
    };
    const std::size_t words = sliding ? 2 * static_cast<std::size_t>(lane_words) : static_cast<std::size_t>(pages) * 8 * 64 * D;
    const int lo = sliding ? std::max(0, first - ik::kDSparkWindow) : 0, hi = first + 8;
    // Lane 1's keys and values at positions [lo, hi) (each at its row), the rest of the cache random too.
    const auto k = random_bf16(words, rng, 1.0);
    const auto v = random_bf16(words, rng, 1.0);
    const auto q = random_bf16(8ULL * 24 * D, rng, 0.35);
    ik::DSparkAttentionArgs a;
    a.q           = upload(q);
    a.cache_k     = upload(k);
    a.cache_v     = upload(v);
    a.lane_stride = lane_words;
    a.ring        = sliding ? ik::kDSparkRing : 0;
    if (!sliding) {
        a.table        = upload(table);
        a.table_stride = stride;
    }
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
                double dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += bf(q[(static_cast<std::size_t>(row) * 24 + h) * D + d]) * bf(k[offset(g, p) + d]);
                }
                s.push_back(dot / std::sqrt(128.0));
                top = std::max(top, s.back());
            }
            double sum = 0;
            for (double& x : s) { sum += x = std::exp(x - top); }
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int p = lo; p < hi; ++p) { o += s[static_cast<std::size_t>(p - lo)] * bf(v[offset(g, p) + d]); }
                worst.add(bf(got[(static_cast<std::size_t>(row) * 24 + h) * D + d]), o / sum, std::ldexp(1.0, -8),
                          std::ldexp(vmax, -12));
            }
        }
    }
    std::printf("dspark_attention (%s, F = %d: keys %d-%d): worst %.3f of the bound\n", sliding ? "sliding" : "full", first,
                lo, hi - 1, worst.ratio);
    check(worst.ratio <= 1.0, "dspark_attention outside its bound");
}

// The full layer in INT8 (DSparkInt8Cache): random codes and FP16 group scales in a page-major pool through a shuffled
// block-table row; the same criterion as attention_case over the represented values bf16(code * s).
void attention_int8_case(int first, int kMaxContext) {
    std::mt19937 rng(static_cast<unsigned>(17 + first));
    const int splits = ik::dspark_attention_splits(false, static_cast<std::uint32_t>(kMaxContext));
    const int stride = (kMaxContext + 8 + 63) / 64, pages = 2 * stride + 3;
    std::vector<std::int32_t> table(2 * static_cast<std::size_t>(stride));
    for (std::size_t i = 0; i < table.size(); ++i) { table[i] = static_cast<std::int32_t>(i); }
    std::shuffle(table.begin(), table.end(), rng);
    const auto row_of = [&](int g, int p) -> std::size_t {
        const std::size_t page = static_cast<std::size_t>(table[static_cast<std::size_t>(stride) + p / 64]);
        return (page * 8 + g) * 64 + p % 64;
    };
    const std::size_t rows = static_cast<std::size_t>(pages) * 8 * 64;
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> magnitude(0.002F, 0.03F);
    std::vector<std::uint8_t> ck(rows * D), cv(rows * D), sk(rows * 4), sv(rows * 4);
    for (auto& c : ck) { c = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng))); }
    for (auto& c : cv) { c = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng))); }
    for (std::size_t r = 0; r < rows * 2; ++r) {
        const std::uint16_t s_k = static_cast<__half_raw>(__float2half_rn(magnitude(rng))).x;
        const std::uint16_t s_v = static_cast<__half_raw>(__float2half_rn(magnitude(rng))).x;
        std::memcpy(sk.data() + 2 * r, &s_k, 2);
        std::memcpy(sv.data() + 2 * r, &s_v, 2);
    }
    const auto key = [&](std::size_t row, int d) {
        std::uint16_t s;
        std::memcpy(&s, sk.data() + row * 4 + (d >= 64 ? 2 : 0), 2);
        return int8_value(ck[row * D + d], s);
    };
    const auto value = [&](std::size_t row, int d) {
        std::uint16_t s;
        std::memcpy(&s, sv.data() + row * 4 + (d >= 64 ? 2 : 0), 2);
        return int8_value(cv[row * D + d], s);
    };
    const int lo = 0, hi = first + 8;
    const auto q = random_bf16(8ULL * 24 * D, rng, 0.35);
    ik::DSparkAttentionArgs a;
    a.q            = upload(q);
    a.int8         = {upload(ck), upload(cv), upload(sk), upload(sv)};
    a.table        = upload(table);
    a.table_stride = stride;
    a.splits       = splits;
    a.round        = upload(std::vector<ik::DSparkRound>{{first, 0, 1}});
    a.scratch      = zeros<float>(ik::dspark_attention_scratch_bytes(splits) / 4);
    a.out          = zeros<std::uint16_t>(8ULL * 24 * D);
    ik::dspark_attention(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "attention (int8)");
    const auto got = download(a.out, 8ULL * 24 * D);
    double vmax    = 0;
    for (int g = 0; g < 8; ++g) {
        for (int p = lo; p < hi; ++p) {
            for (int d = 0; d < D; ++d) { vmax = std::max(vmax, std::fabs(value(row_of(g, p), d))); }
        }
    }
    Worst worst;
    for (int row = 0; row < 8; ++row) {
        for (int h = 0; h < 24; ++h) {
            const int g = h / 3;
            std::vector<double> s;
            double top = -1e300;
            for (int p = lo; p < hi; ++p) {
                double dot = 0;
                for (int d = 0; d < D; ++d) { dot += bf(q[(static_cast<std::size_t>(row) * 24 + h) * D + d]) * key(row_of(g, p), d); }
                s.push_back(dot / std::sqrt(128.0));
                top = std::max(top, s.back());
            }
            double sum = 0;
            for (double& x : s) { sum += x = std::exp(x - top); }
            for (int d = 0; d < D; ++d) {
                double o = 0;
                for (int p = lo; p < hi; ++p) { o += s[static_cast<std::size_t>(p - lo)] * value(row_of(g, p), d); }
                worst.add(bf(got[(static_cast<std::size_t>(row) * 24 + h) * D + d]), o / sum, std::ldexp(1.0, -8),
                          std::ldexp(vmax, -12));
            }
        }
    }
    std::printf("dspark_attention (full, INT8, F = %d: keys %d-%d): worst %.3f of the bound\n", first, lo, hi - 1, worst.ratio);
    check(worst.ratio <= 1.0, "dspark_attention (int8) outside its bound");
}

// INFERNIX_DSPARK_BENCH (a measurement, not a check): the full layer's attention of one row's block at contexts of 2K to
// 200K with the launch sized for a 240,000-token context (the owner's), 100 launches after 20 warm-ups, timed by events.
// INFERNIX_DSPARK_BENCH=int8: the full layer in INT8 (DSparkInt8Cache).
void attention_bench(bool int8) {
    constexpr int kMaxContext = 240000;
    const int splits = ik::dspark_attention_splits(false, kMaxContext);
    const int stride = (kMaxContext + 8 + 63) / 64;
    std::vector<std::int32_t> table(static_cast<std::size_t>(stride));
    for (std::size_t i = 0; i < table.size(); ++i) { table[i] = static_cast<std::int32_t>(i); }
    std::mt19937 rng(21);
    const std::size_t words = static_cast<std::size_t>(stride) * 8 * 64 * D;
    ik::DSparkAttentionArgs a;
    a.q = upload(random_bf16(8ULL * 24 * D, rng, 0.35));
    if (int8) {
        std::vector<std::uint8_t> codes(words), side(words / D * 4);
        for (auto& c : codes) { c = static_cast<std::uint8_t>(rng() % 255U); }
        for (std::size_t i = 0; i < side.size(); i += 2) {
            const std::uint16_t s = static_cast<__half_raw>(__float2half_rn(0.01F)).x;
            std::memcpy(side.data() + i, &s, 2);
        }
        a.int8 = {upload(codes), upload(codes), upload(side), upload(side)};
    } else {
        a.cache_k = upload(random_bf16(words, rng, 1.0));
        a.cache_v = upload(random_bf16(words, rng, 1.0));
    }
    a.table        = upload(table);
    a.table_stride = stride;
    a.splits       = splits;
    a.scratch      = zeros<float>(ik::dspark_attention_scratch_bytes(splits) / 4);
    a.out          = zeros<std::uint16_t>(8ULL * 24 * D);
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    for (const int first : {2048, 8192, 32768, 100000, 160000, 200000}) {
        ik::DSparkRound* round = upload(std::vector<ik::DSparkRound>{{first, 0, 0}});
        a.round                = round;
        for (int i = 0; i < 20; ++i) { ik::dspark_attention(a, nullptr); }
        cuda_check(cudaEventRecord(start, nullptr), "event");
        constexpr int kRuns = 100;
        for (int i = 0; i < kRuns; ++i) { ik::dspark_attention(a, nullptr); }
        cuda_check(cudaEventRecord(stop, nullptr), "event");
        cuda_check(cudaEventSynchronize(stop), "event");
        float ms = 0;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "event");
        const double bytes = 2.0 * (first + 8) * 8 * (int8 ? D + 4 : D * 2); // the keys and values read
        std::printf("bench: full-layer attention (%s) at F = %d (launch for %d): %.4f ms (%.0f GB/s of keys and values)\n",
                    int8 ? "INT8" : "BF16", first, kMaxContext, ms / kRuns, bytes / (ms / kRuns * 1e-3) / 1e9);
        cudaFree(round);
    }
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

// q8: the Fold as Q8_G32_FP16 (q8_g32.h's rule over the BF16 values: scale fl16(m / 127), codes rne(v / scale)); the
// oracle then reads its represented values code * scale. list: a shortlist (--lm-head-draft), every third token left
// out: the tables hold the listed rows in ascending id, and each draft is the oracle's argmax over them as a token id.
void markov_case(bool q8, bool list) {
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
    std::vector<double> fold_value(fold.size());
    std::vector<std::int8_t> codes(q8 ? fold.size() : 0);
    std::vector<std::uint16_t> scales(q8 ? fold.size() / 32 : 0);
    for (std::size_t g = 0; g < fold.size() / 32; ++g) {
        if (!q8) {
            for (int j = 0; j < 32; ++j) { fold_value[32 * g + j] = bf(fold[32 * g + j]); }
            continue;
        }
        float m = 0.0F;
        for (int j = 0; j < 32; ++j) { m = std::max(m, std::fabs(static_cast<float>(bf(fold[32 * g + j])))); }
        const __half sh = __float2half_rn(m / 127.0F);
        const float s   = __half2float(sh);
        scales[g]       = __half_as_ushort(sh);
        for (int j = 0; j < 32; ++j) {
            const float v = static_cast<float>(bf(fold[32 * g + j]));
            const int c   = s == 0.0F ? 0 : std::clamp(static_cast<int>(std::nearbyint(v / s)), -127, 127);
            codes[32 * g + j]      = static_cast<std::int8_t>(c);
            fold_value[32 * g + j] = static_cast<double>(static_cast<float>(c) * s);
        }
    }
    std::vector<float> base(static_cast<std::size_t>(kSteps) * V);
    {
        std::normal_distribution<float> normal(0.0F, 3.0F);
        for (float& b : base) { b = normal(rng); }
    }
    std::vector<std::int32_t> ids;
    for (int v = 0; v < V; ++v) {
        if (!list || v % 3 != 1) { ids.push_back(v); }
    }
    const int R = static_cast<int>(ids.size());
    const auto listed = [&](const auto& table, std::size_t row_words) {
        std::decay_t<decltype(table)> out(static_cast<std::size_t>(R) * row_words);
        for (int r = 0; r < R; ++r) {
            std::copy_n(table.data() + static_cast<std::size_t>(ids[r]) * row_words, row_words,
                        out.data() + static_cast<std::size_t>(r) * row_words);
        }
        return out;
    };
    std::vector<float> listed_base(static_cast<std::size_t>(kSteps) * R);
    for (int step = 0; step < kSteps; ++step) {
        for (int r = 0; r < R; ++r) {
            listed_base[static_cast<std::size_t>(step) * R + r] = base[static_cast<std::size_t>(step) * V + ids[r]];
        }
    }
    ik::DSparkMarkovArgs a;
    a.projection    = upload(proj);
    a.bias          = upload(bias);
    a.block         = upload(block);
    a.f_ring        = upload(ring);
    a.f_ring_rows   = kRing;
    a.f_lane_stride = static_cast<std::int64_t>(kRing) * H;
    a.embedding     = upload(w1);
    if (q8) {
        a.fold_codes  = upload(listed(codes, kRank));
        a.fold_scales = upload(listed(scales, kRank / 32));
    } else {
        a.fold = upload(listed(fold, kRank));
    }
    a.fold_bias     = upload(listed(fbias, 1));
    a.base          = upload(listed_base);
    a.vocab         = R;
    if (list) { a.row_ids = upload(ids); }
    a.round        = upload(std::vector<ik::DSparkRound>{{first, anchor, 0}});
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
        std::vector<double> L(R);
        int best = 0;
        for (int r = 0; r < R; ++r) {
            const int vv = ids[r];
            double acc   = bf(fbias[vv]);
            for (int c = 0; c < kRank; ++c) { acc += fold_value[static_cast<std::size_t>(vv) * kRank + c] * lat[c]; }
            const double z = rbd(rbd(base[static_cast<std::size_t>(step - 1) * V + vv]) + rbd(acc));
            L[r]           = rbd(30.0 * std::tanh(z / 30.0));
            if (L[r] > L[best]) { best = r; }
        }
        double second = -1e300;
        for (int r = 0; r < R; ++r) {
            if (r != best) { second = std::max(second, L[r]); }
        }
        double sum = 0;
        for (const double l : L) { sum += std::exp(l - L[best]); }
        const double ulp = std::ldexp(1.0, std::ilogb(std::max(std::fabs(L[best]), 1e-30)) - 7);
        if (L[best] - second <= 2 * ulp) {
            near_tie = true;
            break;
        }
        ++compared;
        agree      = tokens[static_cast<std::size_t>(step - 1)] == ids[best];
        worst_prob = std::max(worst_prob, std::fabs(probs[static_cast<std::size_t>(step - 1)] - 1.0 / sum) / (1.0 / sum));
        prev       = ids[best];
    }
    std::printf("dspark_markov (%d rows of V = 6000, 7 steps, %s Fold): %d drafts compared%s, %s; worst probability %.2e "
                "relative; repeat %s\n",
                R, q8 ? "8-bit" : "BF16", compared, near_tie ? " (then a near tie)" : "",
                agree ? "all equal the oracle's argmax" : "a draft DIFFERS", worst_prob, repeat ? "identical" : "DIFFERS");
    check(agree, "dspark_markov drafts another token than the oracle's argmax");
    check(compared >= 1, "dspark_markov: no draft compared (a near tie at the first step)");
    check(worst_prob <= std::ldexp(1.0, -6), "dspark_markov's probability is outside its bound");
    check(repeat, "a repeated dspark_markov call differs");
}

// A round of several rows in one launch (block_rows / blocks): three blocks, each its own round (position, anchor,
// lane), data and outputs, give each block's single launch bit for bit: dspark_rotate (dense queries and a sliding
// ring's keys and values), dspark_attention (full and sliding) and dspark_markov (8-bit shortlist Fold).
void blocks_case() {
    std::mt19937 rng(23);
    constexpr int kBlocks = 3, B = 8;
    const std::vector<ik::DSparkRound> rounds{{1500, 4321, 1}, {37, 17, 0}, {70001, 999, 2}};
    bool rotate_ok = true, attention_ok = true, markov_ok = true;
    // Rotate: 24 query heads into a dense buffer, and 8 key heads (with values) into three lanes' rings.
    {
        const auto x  = random_bf16(static_cast<std::size_t>(kBlocks) * B * 24 * D, rng, 1.0);
        const auto kx = random_bf16(static_cast<std::size_t>(kBlocks) * B * 8 * D, rng, 1.0);
        const auto vx = random_bf16(static_cast<std::size_t>(kBlocks) * B * 8 * D, rng, 1.0);
        const auto w  = random_bf16(D, rng, 0.3, 1.0);
        float table[D / 2];
        ik::dspark_inv_freq(1e6, table);
        const std::int64_t lane_words = 8LL * ik::kDSparkRing * D;
        const std::size_t ring_words  = 3 * static_cast<std::size_t>(lane_words);
        const auto run = [&](int b0, int count, std::uint16_t* dense, std::uint16_t* ck, std::uint16_t* cv) {
            const auto* r = upload(std::vector<ik::DSparkRound>(rounds.begin() + b0, rounds.begin() + b0 + count));
            ik::DSparkRotateArgs q;
            q.x          = upload(std::vector<std::uint16_t>(x.begin() + static_cast<std::ptrdiff_t>(b0) * B * 24 * D,
                                                             x.begin() + static_cast<std::ptrdiff_t>(b0 + count) * B * 24 * D));
            q.x_stride   = 24 * D;
            q.rows       = count * B;
            q.heads      = 24;
            q.norm       = upload(w);
            q.round      = r;
            q.block_rows = count > 1 ? B : 0;
            q.inv_freq   = upload(std::vector<float>(table, table + D / 2));
            q.dense      = dense;
            ik::dspark_rotate(q, nullptr);
            ik::DSparkRotateArgs k = q;
            k.x           = upload(std::vector<std::uint16_t>(kx.begin() + static_cast<std::ptrdiff_t>(b0) * B * 8 * D,
                                                              kx.begin() + static_cast<std::ptrdiff_t>(b0 + count) * B * 8 * D));
            k.x_stride    = 8 * D;
            k.values      = upload(std::vector<std::uint16_t>(vx.begin() + static_cast<std::ptrdiff_t>(b0) * B * 8 * D,
                                                              vx.begin() + static_cast<std::ptrdiff_t>(b0 + count) * B * 8 * D));
            k.values_stride = 8 * D;
            k.heads       = 8;
            k.dense       = nullptr;
            k.cache_k     = ck;
            k.cache_v     = cv;
            k.lane_stride = lane_words;
            k.ring        = ik::kDSparkRing;
            ik::dspark_rotate(k, nullptr);
        };
        std::uint16_t* dense_one = zeros<std::uint16_t>(static_cast<std::size_t>(kBlocks) * B * 24 * D);
        std::uint16_t* dense_all = zeros<std::uint16_t>(static_cast<std::size_t>(kBlocks) * B * 24 * D);
        std::uint16_t *k_one = zeros<std::uint16_t>(ring_words), *v_one = zeros<std::uint16_t>(ring_words);
        std::uint16_t *k_all = zeros<std::uint16_t>(ring_words), *v_all = zeros<std::uint16_t>(ring_words);
        for (int b = 0; b < kBlocks; ++b) { run(b, 1, dense_one + static_cast<std::size_t>(b) * B * 24 * D, k_one, v_one); }
        run(0, kBlocks, dense_all, k_all, v_all);
        cuda_check(cudaDeviceSynchronize(), "rotate blocks");
        rotate_ok = download(dense_one, static_cast<std::size_t>(kBlocks) * B * 24 * D) ==
                        download(dense_all, static_cast<std::size_t>(kBlocks) * B * 24 * D) &&
                    download(k_one, ring_words) == download(k_all, ring_words) &&
                    download(v_one, ring_words) == download(v_all, ring_words);
    }
    // Attention: three lanes' caches (sliding rings; the full layer's pages through a shuffled table of three rows).
    for (const bool sliding : {true, false}) {
        // (The full layer's pages at a 4K context: its rounds' positions below it.)
        const int kMaxContext = sliding ? 72000 : 4096;
        const std::vector<ik::DSparkRound> at_rounds =
            sliding ? rounds : std::vector<ik::DSparkRound>{{1500, 4321, 1}, {37, 17, 0}, {3001, 999, 2}};
        const int splits = ik::dspark_attention_splits(sliding, static_cast<std::uint32_t>(kMaxContext));
        const int stride = (kMaxContext + 8 + 63) / 64, pages = 3 * stride;
        const std::int64_t lane_words = sliding ? 8LL * ik::kDSparkRing * D : 0;
        std::vector<std::int32_t> table(3 * static_cast<std::size_t>(stride));
        for (std::size_t i = 0; i < table.size(); ++i) { table[i] = static_cast<std::int32_t>(i); }
        std::shuffle(table.begin(), table.end(), rng);
        const std::size_t words = sliding ? 3 * static_cast<std::size_t>(lane_words) : static_cast<std::size_t>(pages) * 8 * 64 * D;
        const auto* k = upload(random_bf16(words, rng, 1.0));
        const auto* v = upload(random_bf16(words, rng, 1.0));
        const auto q  = random_bf16(static_cast<std::size_t>(kBlocks) * B * 24 * D, rng, 0.35);
        const std::size_t scratch = ik::dspark_attention_scratch_bytes(splits) / 4;
        const auto args = [&](int b0, int count, std::uint16_t* out, float* sc) {
            ik::DSparkAttentionArgs a;
            a.q           = upload(std::vector<std::uint16_t>(q.begin() + static_cast<std::ptrdiff_t>(b0) * B * 24 * D,
                                                              q.begin() + static_cast<std::ptrdiff_t>(b0 + count) * B * 24 * D));
            a.cache_k     = k;
            a.cache_v     = v;
            a.lane_stride = lane_words;
            a.ring        = sliding ? ik::kDSparkRing : 0;
            if (!sliding) {
                a.table        = upload(table);
                a.table_stride = stride;
            }
            a.splits  = splits;
            a.round   = upload(std::vector<ik::DSparkRound>(at_rounds.begin() + b0, at_rounds.begin() + b0 + count));
            a.scratch = sc;
            a.out     = out;
            if (count > 1) {
                a.blocks         = count;
                a.q_stride       = static_cast<std::int64_t>(B) * 24 * D;
                a.out_stride     = static_cast<std::int64_t>(B) * 24 * D;
                a.scratch_stride = static_cast<std::int64_t>(scratch);
            }
            return a;
        };
        std::uint16_t* one = zeros<std::uint16_t>(static_cast<std::size_t>(kBlocks) * B * 24 * D);
        std::uint16_t* all = zeros<std::uint16_t>(static_cast<std::size_t>(kBlocks) * B * 24 * D);
        float* sc = zeros<float>(kBlocks * scratch);
        for (int b = 0; b < kBlocks; ++b) {
            ik::dspark_attention(args(b, 1, one + static_cast<std::size_t>(b) * B * 24 * D, sc), nullptr);
        }
        ik::dspark_attention(args(0, kBlocks, all, sc), nullptr);
        cuda_check(cudaDeviceSynchronize(), "attention blocks");
        attention_ok = attention_ok && download(one, static_cast<std::size_t>(kBlocks) * B * 24 * D) ==
                                           download(all, static_cast<std::size_t>(kBlocks) * B * 24 * D);
    }
    // Markov: an 8-bit shortlist Fold; each block its rows, base, three lanes' context rings and seven steps.
    {
        constexpr int V = 3000, R3 = 768, K = 5120, kRank = 256, kRing = 16, kSteps = 7;
        std::vector<std::int8_t> codes(static_cast<std::size_t>(V) * kRank);
        std::uniform_int_distribution<int> code(-127, 127);
        for (auto& c : codes) { c = static_cast<std::int8_t>(code(rng)); }
        std::vector<std::uint16_t> scales(static_cast<std::size_t>(V) * kRank / 32);
        for (auto& s : scales) { s = __half_as_ushort(__float2half_rn(0.001F)); }
        std::vector<std::int32_t> ids(V);
        for (int i = 0; i < V; ++i) { ids[i] = 2 * i + 1; }
        std::vector<float> base(static_cast<std::size_t>(kBlocks) * B * V);
        std::normal_distribution<float> normal(0.0F, 3.0F);
        for (float& b : base) { b = normal(rng); }
        const auto* proj  = upload(random_bf16(static_cast<std::size_t>(R3) * K, rng, 0.02));
        const auto* bias  = upload(random_bf16(R3, rng, 0.05));
        const auto block  = random_bf16(static_cast<std::size_t>(kBlocks) * B * H, rng, 1.0);
        const auto* ring  = upload(random_bf16(3ULL * kRing * H, rng, 1.0));
        const auto* w1    = upload(random_bf16(static_cast<std::size_t>(2 * V + 2) * kRank, rng, 1.0));
        const auto* fbias = upload(random_bf16(V, rng, 0.1));
        const auto* fcode = upload(codes);
        const auto* fscal = upload(scales);
        const auto* rid   = upload(ids);
        const std::size_t scratch = (ik::dspark_markov_scratch_bytes(V) / 4 + 63) / 64 * 64; // 256-byte slices
        const auto args = [&](int b0, int count, std::int32_t* tokens, float* probs, float* sc) {
            ik::DSparkMarkovArgs a;
            a.projection    = proj;
            a.bias          = bias;
            a.block         = upload(std::vector<std::uint16_t>(block.begin() + static_cast<std::ptrdiff_t>(b0) * B * H,
                                                                block.begin() + static_cast<std::ptrdiff_t>(b0 + count) * B * H));
            a.f_ring        = ring;
            a.f_ring_rows   = kRing;
            a.f_lane_stride = static_cast<std::int64_t>(kRing) * H;
            a.embedding     = w1;
            a.fold_codes    = fcode;
            a.fold_scales   = fscal;
            a.fold_bias     = fbias;
            a.base          = upload(std::vector<float>(base.begin() + static_cast<std::ptrdiff_t>(b0) * B * V,
                                                        base.begin() + static_cast<std::ptrdiff_t>(b0 + count) * B * V));
            a.vocab         = V;
            a.row_ids       = rid;
            a.round         = upload(std::vector<ik::DSparkRound>(rounds.begin() + b0, rounds.begin() + b0 + count));
            a.tokens        = tokens;
            a.probabilities = probs;
            a.scratch       = sc;
            if (count > 1) {
                a.blocks         = count;
                a.block_stride   = static_cast<std::int64_t>(B) * H;
                a.base_stride    = static_cast<std::int64_t>(B) * V;
                a.scratch_stride = static_cast<std::int64_t>(scratch);
            }
            return a;
        };
        std::int32_t* t_one = zeros<std::int32_t>(kBlocks * B);
        std::int32_t* t_all = zeros<std::int32_t>(kBlocks * B);
        float* p_one = zeros<float>(kBlocks * B);
        float* p_all = zeros<float>(kBlocks * B);
        float* sc    = zeros<float>(kBlocks * scratch);
        for (int b = 0; b < kBlocks; ++b) {
            ik::dspark_markov(args(b, 1, t_one + b * B, p_one + b * B, sc), kSteps, nullptr);
        }
        ik::dspark_markov(args(0, kBlocks, t_all, p_all, sc), kSteps, nullptr);
        cuda_check(cudaDeviceSynchronize(), "markov blocks");
        markov_ok = download(t_one, kBlocks * B) == download(t_all, kBlocks * B) &&
                    download(p_one, kBlocks * B) == download(p_all, kBlocks * B);
    }
    std::printf("DSpark blocks (3 rounds in one launch = each alone): rotate %s, attention (sliding, full) %s, markov %s\n",
                rotate_ok ? "bit for bit" : "DIFFER", attention_ok ? "bit for bit" : "DIFFER", markov_ok ? "bit for bit" : "DIFFER");
    check(rotate_ok && attention_ok && markov_ok, "a launch over blocks differs from the blocks' single launches");
}

} // namespace

INFERNIX_OP_TEST(inkling2_dspark) {
    if (const char* bench = std::getenv("INFERNIX_DSPARK_BENCH"); bench != nullptr) {
        attention_bench(std::string(bench) == "int8");
        return 0;
    }
    mix_case();
    rotate_case();
    attention_case(true, 1500);
    attention_case(true, 40);
    attention_case(true, 70001); // the ring's slots wrapped many times, tiles across the wrap
    attention_case(false, 1500);
    attention_case(false, 3);
    // A long context: 42 splits of 15 tiles (the last one partial), the merge over all of them.
    attention_case(false, 20000, 24000);
    attention_int8_case(1500, 2048);
    attention_int8_case(3, 2048);
    attention_int8_case(20000, 24000);
    add_norm_case();
    swiglu_case();
    markov_case(false, false);
    markov_case(true, false);
    markov_case(false, true);
    markov_case(true, true);
    blocks_case();
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
