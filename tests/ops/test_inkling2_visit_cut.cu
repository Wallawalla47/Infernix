// Inkling 2's cut instances of the visit chain (include/infernix/ops/inkling2_visit.h ResidualCut; --ablate, docs/ablate.md):
// the attn lane in mlp_entry (the post-attention conv's output, before it joins the residual) and the mlp lane in
// final_mix / visit_entry's tail (the post-MLP conv's output, before cand + m), against an FP64 oracle over the plain
// instance's own conv output y (read from a plain call: a boundary layer's cand is y itself, and with cand = 0 the
// tail's output is y):
//   y' = y - alpha c d,  c = sum_h d[h] y[h]
//
// Criterion (pointwise): |y' - ref| <= 2^-22 |ref| + alpha |d_h| 2^-16 sum_j |d_j y_j|: the FP32 FMA and the FP32 dot in
// the pinned order over 2304 products (its 2^-16 of the magnitude bounds the fold with room). The tail's output is
// rounded to BF16 once more (the residual add's rounding), which the BF16 criterion adds.
//
// Also: a tiny alpha (1e-30) keeps every plain bit (a cut too small to move an element); the conv ring holds the uncut
// input (the cut acts after the ring's store); a column's bits equal the same column computed alone (one column, its
// own position, with the earlier columns' ring history).

#include "infernix/ops/inkling2_visit.h"
#include "ops/common/canonical_math.h"
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

constexpr int kH = ik::kHidden;
int g_failures   = 0;

void fail(const char* what, int t, int h) {
    if (g_failures++ < 20) { std::fprintf(stderr, "FAIL: %s (column %d, element %d)\n", what, t, h); }
}

template <class T>
T* dalloc(std::size_t n) {
    T* p = nullptr;
    cuda_check(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "cudaMalloc");
    return p;
}

template <class T>
void up(T* p, const std::vector<T>& v) {
    cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

template <class T>
std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

std::vector<std::uint16_t> random_bf16(std::size_t n, float sd, std::mt19937& gen) {
    std::normal_distribution<float> d(0.0F, sd);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) { x = canon::f32_to_bf16_rn(d(gen)); }
    return v;
}

std::vector<float> unit_direction(std::mt19937& gen) {
    std::normal_distribution<float> d(0.0F, 1.0F);
    std::vector<float> v(kH);
    double norm = 0;
    for (auto& x : v) {
        x = d(gen);
        norm += static_cast<double>(x) * x;
    }
    norm = std::sqrt(norm);
    for (auto& x : v) { x = static_cast<float>(x / norm); }
    return v;
}

// One lane's columns at positions 0 .. T - 1: the column map, a zero ring and the other device operands.
struct Fixture {
    int T;
    std::int32_t *lane, *position, *first;
    std::uint16_t *conv, *o, *h, *norm, *experts, *ring, *x, *tail_out;
    float *cand, *query, *direction;
    std::vector<std::uint16_t> o_host;
    std::vector<std::uint16_t> experts_host;

    Fixture(int columns, std::mt19937& gen) : T(columns) {
        lane     = dalloc<std::int32_t>(T);
        position = dalloc<std::int32_t>(T);
        first    = dalloc<std::int32_t>(1);
        up(lane, std::vector<std::int32_t>(T, 0));
        std::vector<std::int32_t> pos(T);
        for (int t = 0; t < T; ++t) { pos[t] = t; }
        up(position, pos);
        up(first, std::vector<std::int32_t>{0});
        conv  = dalloc<std::uint16_t>(4 * kH);
        up(conv, random_bf16(4 * kH, 0.1F, gen));
        o     = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * kH);
        o_host = random_bf16(static_cast<std::size_t>(T) * kH, 1.0F, gen);
        up(o, o_host);
        h     = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * kH);
        up(h, random_bf16(static_cast<std::size_t>(T) * kH, 1.0F, gen));
        norm  = dalloc<std::uint16_t>(kH);
        up(norm, std::vector<std::uint16_t>(kH, canon::f32_to_bf16_rn(1.0F)));
        experts = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * moe::kTopK * kH);
        experts_host = random_bf16(static_cast<std::size_t>(T) * moe::kTopK * kH, 0.5F, gen);
        up(experts, experts_host);
        ring     = dalloc<std::uint16_t>(static_cast<std::size_t>(ik::kRingPositions) * kH);
        x        = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * kH);
        tail_out = dalloc<std::uint16_t>(static_cast<std::size_t>(T) * kH);
        cand     = dalloc<float>(static_cast<std::size_t>(T) * kH);
        query    = dalloc<float>(kH);
        up(query, std::vector<float>(kH, 0.01F));
        direction = dalloc<float>(kH);
        up(direction, unit_direction(gen));
    }
    ~Fixture() {
        for (void* p : {static_cast<void*>(lane), static_cast<void*>(position), static_cast<void*>(first),
                        static_cast<void*>(conv), static_cast<void*>(o), static_cast<void*>(h), static_cast<void*>(norm),
                        static_cast<void*>(experts), static_cast<void*>(ring), static_cast<void*>(x),
                        static_cast<void*>(tail_out), static_cast<void*>(cand), static_cast<void*>(query),
                        static_cast<void*>(direction)}) {
            cudaFree(p);
        }
    }
    void clear_ring() const {
        cuda_check(cudaMemset(ring, 0, static_cast<std::size_t>(ik::kRingPositions) * kH * 2), "ring");
    }
    [[nodiscard]] ik::ColumnMap map() const { return {lane, position, first, T}; }
    [[nodiscard]] ik::Ring ring_view() const { return {ring, kH, std::int64_t{ik::kRingPositions} * kH}; }
};

ik::MlpEntryArgs mlp_args(const Fixture& f, bool boundary) {
    ik::MlpEntryArgs m;
    m.columns  = f.T;
    m.map      = f.map();
    m.o        = f.o;
    m.conv     = f.conv;
    m.ring     = f.ring_view();
    m.h        = f.h;
    m.boundary = boundary;
    m.cand     = f.cand;
    m.query    = f.query;
    m.norm     = f.norm;
    m.x        = f.x;
    return m; // rows.rows null: no expert rows
}

// y' against y - alpha (d . y) d (FP64), for one column.
void check_cut(const float* y, const std::vector<float>& d, float alpha, const float* got, bool bf16, int t) {
    double dot = 0, magnitude = 0;
    for (int h = 0; h < kH; ++h) {
        dot += static_cast<double>(d[h]) * y[h];
        magnitude += std::abs(static_cast<double>(d[h]) * y[h]);
    }
    for (int h = 0; h < kH; ++h) {
        const double ref    = y[h] - static_cast<double>(alpha) * dot * d[h];
        const double fold   = static_cast<double>(alpha) * std::abs(d[h]) * std::ldexp(magnitude, -16);
        const double bound  = std::ldexp(std::abs(ref), -22) + fold + (bf16 ? std::ldexp(std::abs(ref) + fold, -8) : 0.0);
        if (std::abs(got[h] - ref) > bound) { fail("the cut is outside the FP64 criterion", t, h); }
    }
}

void mlp_case(int T, bool boundary, std::uint32_t seed) {
    std::mt19937 gen(seed);
    Fixture f(T, gen);
    const std::vector<float> d = down(f.direction, kH);
    // The plain call at a boundary: cand is the conv's output y itself; the ring its uncut inputs.
    f.clear_ring();
    ik::mlp_entry(mlp_args(f, true), nullptr);
    cuda_check(cudaDeviceSynchronize(), "plain mlp_entry");
    const std::vector<float> y            = down(f.cand, static_cast<std::size_t>(T) * kH);
    const std::vector<std::uint16_t> ring = down(f.ring, static_cast<std::size_t>(ik::kRingPositions) * kH);
    // The plain cand at this case's boundary flag, for the tiny-alpha check.
    f.clear_ring();
    ik::mlp_entry(mlp_args(f, boundary), nullptr);
    cuda_check(cudaDeviceSynchronize(), "plain mlp_entry");
    const std::vector<float> plain = down(f.cand, static_cast<std::size_t>(T) * kH);

    for (const float alpha : {1.0F, 2.5F, 1e-30F}) {
        f.clear_ring();
        ik::MlpEntryCutArgs c;
        static_cast<ik::MlpEntryArgs&>(c) = mlp_args(f, boundary);
        c.cut                              = {f.direction, alpha};
        ik::mlp_entry(c, nullptr);
        cuda_check(cudaDeviceSynchronize(), "mlp_entry cut");
        const std::vector<float> got = down(f.cand, static_cast<std::size_t>(T) * kH);
        if (std::memcmp(down(f.ring, ring.size()).data(), ring.data(), ring.size() * 2) != 0) {
            fail("the ring does not hold the uncut input", -1, -1);
        }
        if (alpha < 1e-20F) {
            for (std::size_t i = 0; i < got.size(); ++i) {
                if (y[i] != 0.0F && std::memcmp(&got[i], &plain[i], 4) != 0) { fail("alpha 1e-30 moved a bit", -1, static_cast<int>(i)); }
            }
            continue;
        }
        if (boundary) {
            for (int t = 0; t < T; ++t) {
                check_cut(y.data() + static_cast<std::size_t>(t) * kH, d, alpha, got.data() + static_cast<std::size_t>(t) * kH,
                          false, t);
            }
        }
    }
}

void tail_case(int T, std::uint32_t seed) {
    std::mt19937 gen(seed);
    Fixture f(T, gen);
    const std::vector<float> d = down(f.direction, kH);
    // cand = 0: the plain tail's output is the conv's BF16 output y.
    cuda_check(cudaMemset(f.cand, 0, static_cast<std::size_t>(T) * kH * 4), "cand");
    ik::FinalArgs a;
    a.columns  = T;
    a.map      = f.map();
    a.tail     = {f.experts, f.cand, f.conv, f.ring_view()};
    a.tail_out = f.tail_out;
    a.query    = f.query;
    a.norm     = f.norm;
    a.out      = f.x;
    f.clear_ring();
    ik::final_mix(a, nullptr);
    cuda_check(cudaDeviceSynchronize(), "plain final_mix");
    const std::vector<std::uint16_t> y16  = down(f.tail_out, static_cast<std::size_t>(T) * kH);
    const std::vector<std::uint16_t> ring = down(f.ring, static_cast<std::size_t>(ik::kRingPositions) * kH);
    std::vector<float> y(y16.size());
    for (std::size_t i = 0; i < y.size(); ++i) { y[i] = canon::bf16_to_f32(y16[i]); }

    for (const float alpha : {1.0F, 2.5F, 1e-30F}) {
        f.clear_ring();
        ik::final_mix(a, ik::ResidualCut{f.direction, alpha}, nullptr);
        cuda_check(cudaDeviceSynchronize(), "final_mix cut");
        const std::vector<std::uint16_t> got16 = down(f.tail_out, y16.size());
        if (std::memcmp(down(f.ring, ring.size()).data(), ring.data(), ring.size() * 2) != 0) {
            fail("the ring does not hold the uncut input", -1, -1);
        }
        if (alpha < 1e-20F) {
            // An element the plain output holds as exactly zero takes the cut's own tiny value (the oracle below).
            for (std::size_t i = 0; i < y16.size(); ++i) {
                if (y[i] != 0.0F && got16[i] != y16[i]) { fail("alpha 1e-30 moved a bit", -1, static_cast<int>(i)); }
            }
        }
        std::vector<float> got(got16.size());
        for (std::size_t i = 0; i < got.size(); ++i) { got[i] = canon::bf16_to_f32(got16[i]); }
        for (int t = 0; t < T; ++t) {
            check_cut(y.data() + static_cast<std::size_t>(t) * kH, d, alpha, got.data() + static_cast<std::size_t>(t) * kH, true, t);
        }
    }
}

// A column's bits are its own: the column computed at T = 1 (position p, its ring history as the wide call left it for
// positions before p) equals its row of the wide call.
void column_case(std::uint32_t seed) {
    constexpr int T = 5;
    std::mt19937 gen(seed);
    Fixture wide(T, gen);
    wide.clear_ring();
    ik::MlpEntryCutArgs c;
    static_cast<ik::MlpEntryArgs&>(c) = mlp_args(wide, true);
    c.cut                              = {wide.direction, 1.0F};
    ik::mlp_entry(c, nullptr);
    cuda_check(cudaDeviceSynchronize(), "mlp_entry cut");
    const std::vector<float> all = down(wide.cand, static_cast<std::size_t>(T) * kH);
    // Column 0 of a one-column call on the same operands (position 0: no history).
    ik::MlpEntryCutArgs one = c;
    one.columns             = 1;
    one.map.columns         = 1;
    wide.clear_ring();
    ik::mlp_entry(one, nullptr);
    cuda_check(cudaDeviceSynchronize(), "mlp_entry cut, one column");
    const std::vector<float> alone = down(wide.cand, static_cast<std::size_t>(kH));
    if (std::memcmp(alone.data(), all.data(), alone.size() * 4) != 0) { fail("a column alone differs", 0, -1); }
}

} // namespace

INFERNIX_OP_TEST(inkling2_visit_cut) {
    std::uint32_t seed = 20261124;
    for (const int T : {1, 3, 17}) {
        mlp_case(T, true, seed++);
        mlp_case(T, false, seed++);
        tail_case(T, seed++);
    }
    column_case(seed++);
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
