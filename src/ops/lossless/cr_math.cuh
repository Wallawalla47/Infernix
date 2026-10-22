#pragma once

// Correctly rounded binary32 (and BF16) functions of binary32 arguments for the --lossless route
// (docs/maintainer/lossless-route.md §3). Each evaluates in double first and accepts that when both ends
// of the evaluation's error interval round to the same value (rounding is monotone, so every value
// between them does too); otherwise it evaluates in double-double (relative error below 2^-94) and
// rounds that, which no binary32 argument's exact value lies close enough to a midpoint to defeat
// (binary32 worst cases are about 2^-60 away). rsqrt settles a near-midpoint case exactly instead.

#include "ops/lossless/noise.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::lossless {

struct DD {
    double hi;
    double lo;
};

__device__ __forceinline__ DD dd_two_sum(double a, double b) {
    const double s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

__device__ __forceinline__ DD dd_fast_two_sum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}

__device__ __forceinline__ DD dd_two_prod(double a, double b) {
    const double p = a * b;
    return {p, fma(a, b, -p)};
}

__device__ __forceinline__ DD dd_add(DD x, DD y) {
    DD s       = dd_two_sum(x.hi, y.hi);
    const DD t = dd_two_sum(x.lo, y.lo);
    s.lo += t.hi;
    s = dd_fast_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    return dd_fast_two_sum(s.hi, s.lo);
}

__device__ __forceinline__ DD dd_mul(DD x, DD y) {
    DD p = dd_two_prod(x.hi, y.hi);
    p.lo += x.hi * y.lo + x.lo * y.hi;
    return dd_fast_two_sum(p.hi, p.lo);
}

__device__ __forceinline__ DD dd_div(DD x, DD y) {
    const double q1 = x.hi / y.hi;
    DD r            = dd_add(x, dd_mul({-q1, 0.0}, y));
    const double q2 = r.hi / y.hi;
    r               = dd_add(r, dd_mul({-q2, 0.0}, y));
    const double q3 = r.hi / y.hi;
    DD q            = dd_fast_two_sum(q1, q2);
    return dd_add(q, {q3, 0.0});
}

// exp(x) for double x, relative error below 2^-94 (the squarings amplify the Taylor sum's 2^-104; no
// overflow or underflow handling: callers keep
// |x| < 700).
__device__ inline DD dd_exp(double x) {
    constexpr double kLn2Hi = 0x1.62e42fefa39efp-1, kLn2Lo = 0x1.abc9e3b39803fp-56;
    const double k = rint(x * 0x1.71547652b82fep0);
    // r = x - k ln2 in double-double.
    DD r = dd_add({x, 0.0}, dd_two_prod(-k, kLn2Hi));
    r    = dd_add(r, dd_two_prod(-k, kLn2Lo));
    // exp(r) = exp(r / 256)^256; Taylor to degree 14 for |r / 256| < 0.0014.
    const DD s = {ldexp(r.hi, -8), ldexp(r.lo, -8)};
    DD term = {1.0, 0.0}, sum = {1.0, 0.0};
#pragma unroll 1
    for (int n = 1; n <= 14; ++n) {
        term = dd_div(dd_mul(term, s), {static_cast<double>(n), 0.0});
        sum  = dd_add(sum, term);
    }
#pragma unroll 1
    for (int i = 0; i < 8; ++i) { sum = dd_mul(sum, sum); }
    const int ki = static_cast<int>(k);
    return {ldexp(sum.hi, ki), ldexp(sum.lo, ki)};
}

// log(a) for a > 0 in double-double: one Newton step on exp from log(a.hi).
__device__ inline DD dd_log(DD a) {
    const double y0 = log(a.hi);
    const DD e      = dd_exp(-y0);
    const DD t      = dd_add(dd_mul(a, e), {-1.0, 0.0});
    return dd_add({y0, 0.0}, t);
}

// ------------------------------------------------------------------------- rounding

// Round to nearest even of a double to BF16 (via round-to-odd binary32, so the double rounding is
// exact).
__device__ __forceinline__ std::uint16_t bf16_rne_of_double(double x) {
    float f         = __double2float_rz(x);
    std::uint32_t u = __float_as_uint(f);
    if (static_cast<double>(f) != x && (u & 0x7F800000U) != 0x7F800000U) { u |= 1U; }
    return __bfloat16_as_ushort(__float2bfloat16_rn(__uint_as_float(u)));
}

__device__ __forceinline__ double bf16_value(std::uint16_t b) {
    return static_cast<double>(__bfloat162float(__ushort_as_bfloat16(b)));
}

// The BF16 neighbour of x one step up (dir > 0) or down (sign-magnitude ordered).
__device__ __forceinline__ std::uint16_t bf16_step(std::uint16_t x, int dir) {
    const bool neg = (x & 0x8000U) != 0;
    int mag        = x & 0x7FFF;
    if ((dir > 0) != neg) {
        ++mag;
    } else if (mag == 0) {
        return static_cast<std::uint16_t>(dir > 0 ? 0x0001U : 0x8001U);
    } else {
        --mag;
    }
    return static_cast<std::uint16_t>((neg ? 0x8000U : 0U) | static_cast<unsigned>(mag));
}

// The noise twin (noise.cuh): r, the nearest BF16 or binary32 to a value r + diff, moved to its other
// neighbour with probability |diff| / gap. Unchanged without a seed, for an exact value, at infinity, or
// where the other neighbour would overflow.
__device__ __forceinline__ std::uint16_t bf16_noise(std::uint16_t r, double diff, std::uint64_t key) {
    if (!noise_active() || diff == 0.0 || (r & 0x7F80U) == 0x7F80U) { return r; }
    const std::uint16_t n = bf16_step(r, diff > 0.0 ? 1 : -1);
    if ((n & 0x7F80U) == 0x7F80U) { return r; }
    return noise_moves(diff / (bf16_value(n) - bf16_value(r)), key) ? n : r;
}

__device__ __forceinline__ float f32_noise(float r, double diff, std::uint64_t key) {
    if (!noise_active() || diff == 0.0 || isinf(r)) { return r; }
    const float n = nextafterf(r, diff > 0.0 ? INFINITY : -INFINITY);
    if (isinf(n)) { return r; }
    return noise_moves(diff / (static_cast<double>(n) - static_cast<double>(r)), key) ? n : r;
}

__device__ __forceinline__ std::uint64_t value_key(double hi, double lo = 0.0) {
    return noise_mix(static_cast<std::uint64_t>(__double_as_longlong(hi))) ^
           static_cast<std::uint64_t>(__double_as_longlong(lo));
}

// A double correctly rounded to BF16 (the noise twin's stochastic rounding of it with a seed).
__device__ __forceinline__ std::uint16_t bf16_of_double(double x) {
    const std::uint16_t b = bf16_rne_of_double(x);
    return bf16_noise(b, x - bf16_value(b), value_key(x));
}

// Whether hi + lo lies beyond the midpoint m (direction dir: +1 above, -1 below), with hi - m exact
// (Sterbenz: hi lies within a binary32 or BF16 step of m) and |lo| below hi's double ulp, so lo
// decides only when hi sits on m exactly.
__device__ __forceinline__ bool beyond(DD v, double m, int dir) {
    const double d = v.hi - m;
    return dir > 0 ? (d > 0.0 || (d == 0.0 && v.lo > 0.0)) : (d < 0.0 || (d == 0.0 && v.lo < 0.0));
}

// Round to nearest even of hi + lo (|lo| <= ulp(hi) / 2) to binary32: rounds hi, then moves to a
// neighbour when hi + lo lies beyond the midpoint toward it.
__device__ inline float f32_rne_of_dd(DD v) {
    float f = __double2float_rn(v.hi);
    if (isinf(f) || v.lo == 0.0) { return f; }
    const double fd = static_cast<double>(f);
    const float up = nextafterf(f, INFINITY), down = nextafterf(f, -INFINITY);
    if (beyond(v, fd + (static_cast<double>(up) - fd) / 2, 1)) { return up; }
    if (beyond(v, fd - (fd - static_cast<double>(down)) / 2, -1)) { return down; }
    return f;
}

__device__ inline float f32_of_dd(DD v) {
    const float f = f32_rne_of_dd(v);
    return f32_noise(f, (v.hi - static_cast<double>(f)) + v.lo, value_key(v.hi, v.lo));
}

__device__ __forceinline__ std::uint16_t bf16_rne_of_dd(DD v) {
    // The BF16 neighbours around the rounded double, decided by hi + lo against their midpoint.
    const std::uint16_t b = bf16_rne_of_double(v.hi);
    if (v.lo == 0.0) { return b; }
    const double bd = bf16_value(b);
    const std::uint16_t up = bf16_step(b, 1), down = bf16_step(b, -1);
    const double up_d = bf16_value(up), down_d = bf16_value(down);
    if (beyond(v, bd + (up_d - bd) / 2, 1)) { return up; }
    if (beyond(v, bd - (bd - down_d) / 2, -1)) { return down; }
    return b;
}

__device__ __forceinline__ std::uint16_t bf16_of_dd(DD v) {
    const std::uint16_t b = bf16_rne_of_dd(v);
    return bf16_noise(b, (v.hi - bf16_value(b)) + v.lo, value_key(v.hi, v.lo));
}

// Whether every value within err of y rounds to the same binary32 (and which). With a noise seed the
// accepted value is the noise twin's rounding of y.
__device__ __forceinline__ bool certain_f32(double y, double err, float& out) {
    const float a = __double2float_rn(y - err), b = __double2float_rn(y + err);
    out           = __double2float_rn(y);
    const bool certain = a == b && __float_as_uint(a) == __float_as_uint(out);
    if (certain) { out = f32_noise(out, y - static_cast<double>(out), value_key(y)); }
    return certain;
}

__device__ __forceinline__ bool certain_bf16(double y, double err, std::uint16_t& out) {
    const std::uint16_t a = bf16_rne_of_double(y - err), b = bf16_rne_of_double(y + err);
    out                   = bf16_rne_of_double(y);
    const bool certain    = a == b && a == out;
    if (certain) { out = bf16_noise(out, y - bf16_value(out), value_key(y)); }
    return certain;
}

// ------------------------------------------------------------------------- functions

// The double evaluations' error bounds (relative): CUDA's double exp and log1p are within 1 ulp and
// the arithmetic around them adds a few roundings; 2^-48 covers every combination below.
constexpr double kDoubleErr = 0x1p-48;

__device__ inline DD dd_exp_f(float x) { return dd_exp(static_cast<double>(x)); }

// sigmoid(x) = 1 / (1 + exp(-x)) in double-double; below 2^-40 in magnitude its series 1/2 + x/4 (the
// next term, x^3 / 48, lies below 2^-122 relative), since 1 + exp(-x) cannot hold x there.
__device__ inline DD dd_sigmoid(float x) {
    if (fabsf(x) < 0x1p-40F) { return {0.5, static_cast<double>(x) * 0.25}; }
    if (x < -700.0F) { return {0.0, 0.0}; }
    const DD e = x > 700.0F ? DD{0.0, 0.0} : dd_exp(-static_cast<double>(x));
    return dd_div({1.0, 0.0}, dd_add({1.0, 0.0}, e));
}

// x sigmoid(x).
__device__ inline DD dd_silu(float x) { return dd_mul({static_cast<double>(x), 0.0}, dd_sigmoid(x)); }

// log(1 + exp(x)) = x + log(1 + exp(-x)) for x > 0.
__device__ inline DD dd_softplus(float x) {
    const double xd = static_cast<double>(x);
    if (x > 0.0F) {
        const DD tail = xd > 700.0 ? DD{0.0, 0.0} : dd_log(dd_add({1.0, 0.0}, dd_exp(-xd)));
        return dd_add({xd, 0.0}, tail);
    }
    if (xd < -700.0) { return {0.0, 0.0}; }
    return dd_log(dd_add({1.0, 0.0}, dd_exp(xd)));
}

__device__ inline float cr_exp(float x) {
    if (isnan(x)) { return x; }
    if (x > 89.0F) { return INFINITY; }
    if (x < -104.0F) { return 0.0F; } // exp(x) < 2^-150: rounds to +0
    const double y = exp(static_cast<double>(x));
    float out;
    if (certain_f32(y, y * kDoubleErr, out)) { return out; }
    return f32_of_dd(dd_exp_f(x));
}

__device__ inline float cr_sigmoid(float x) {
    if (isnan(x)) { return x; }
    const double y = 1.0 / (1.0 + exp(-static_cast<double>(x)));
    float out;
    if (certain_f32(y, y * kDoubleErr, out)) { return out; }
    return f32_of_dd(dd_sigmoid(x));
}

__device__ inline std::uint16_t cr_sigmoid_bf16(float x) {
    if (isnan(x)) { return 0x7FC0U; }
    const double y = 1.0 / (1.0 + exp(-static_cast<double>(x)));
    std::uint16_t out;
    if (certain_bf16(y, y * kDoubleErr, out)) { return out; }
    return bf16_of_dd(dd_sigmoid(x));
}

__device__ inline float cr_silu(float x) {
    if (isnan(x) || x == 0.0F) { return x; } // silu(+-0) = +-0 (the interval below would lose the sign)
    if (x < -110.0F) { return -0.0F; } // |x e^x| < 2^-150
    const double y = static_cast<double>(x) / (1.0 + exp(-static_cast<double>(x)));
    float out;
    if (certain_f32(y, fabs(y) * kDoubleErr, out)) { return out; }
    return f32_of_dd(dd_silu(x));
}

__device__ inline std::uint16_t cr_silu_bf16(float x) {
    if (isnan(x)) { return 0x7FC0U; }
    if (x == 0.0F) { return signbit(x) ? 0x8000U : 0U; } // silu(+-0) = +-0
    if (x < -110.0F) { return 0x8000U; }
    const double y = static_cast<double>(x) / (1.0 + exp(-static_cast<double>(x)));
    std::uint16_t out;
    if (certain_bf16(y, fabs(y) * kDoubleErr, out)) { return out; }
    return bf16_of_dd(dd_silu(x));
}

__device__ inline float cr_softplus(float x) {
    if (isnan(x)) { return x; }
    // x + log1p(exp(-x)) for x > 0, so exp never overflows.
    const double xd = static_cast<double>(x);
    const double y  = x > 0.0F ? xd + log1p(exp(-xd)) : log1p(exp(xd));
    float out;
    if (certain_f32(y, y * kDoubleErr, out)) { return out; }
    return f32_of_dd(dd_softplus(x));
}

// ------------------------------------------------------------------------- cos and sin

// cos(r) and sin(r) for |r| <= pi/4 + 2^-20 in double-double: Taylor to degree 28 / 27 (the remainder
// below 2^-100 of the result).
__device__ inline void dd_cos_sin_reduced(DD r, DD& c, DD& s) {
    const DD r2 = dd_mul(r, r);
    DD cterm = {1.0, 0.0}, csum = {1.0, 0.0}, sterm = r, ssum = r;
#pragma unroll 1
    for (int n = 1; n <= 14; ++n) {
        // cterm_n = -cterm_(n-1) r^2 / ((2n - 1) 2n); sterm_n = -sterm_(n-1) r^2 / (2n (2n + 1)).
        cterm = dd_div(dd_mul(cterm, r2), {-static_cast<double>((2 * n - 1) * (2 * n)), 0.0});
        sterm = dd_div(dd_mul(sterm, r2), {-static_cast<double>((2 * n) * (2 * n + 1)), 0.0});
        csum  = dd_add(csum, cterm);
        ssum  = dd_add(ssum, sterm);
    }
    c = csum;
    s = ssum;
}

// cos(x) and sin(x) for a binary32 |x| < 2^20 in double-double: x - k pi/2 with pi/2 in four pieces
// (fdlibm's: the first three of 33 bits, so k times each is exact for k < 2^20), then the reduced
// Taylor sums by quadrant.
__device__ inline void dd_cos_sin(float x, DD& c, DD& s) {
    constexpr double kPio2_1  = 1.57079632673412561417e+00;
    constexpr double kPio2_2  = 6.07710050630396597660e-11;
    constexpr double kPio2_3  = 2.02226624871116645580e-21;
    constexpr double kPio2_3t = 8.47842766036889956997e-32;
    const double xd = static_cast<double>(x);
    const double k  = rint(xd * 0x1.45f306dc9c883p-1); // x 2 / pi
    DD r            = {xd - k * kPio2_1, 0.0};          // exact (k kPio2_1 exact, the difference by Sterbenz)
    r               = dd_add(r, {-k * kPio2_2, 0.0});    // k kPio2_2 exact
    r               = dd_add(r, {-k * kPio2_3, 0.0});    // exact
    r               = dd_add(r, dd_two_prod(-k, kPio2_3t));
    DD cr, sr;
    dd_cos_sin_reduced(r, cr, sr);
    const int q = static_cast<int>(static_cast<long long>(k) & 3);
    const DD ncr = {-cr.hi, -cr.lo}, nsr = {-sr.hi, -sr.lo};
    c = q == 0 ? cr : q == 1 ? nsr : q == 2 ? ncr : sr;
    s = q == 0 ? sr : q == 1 ? cr : q == 2 ? nsr : ncr;
}

// Correctly rounded binary32 cos(x) and sin(x) for |x| < 2^20 (CUDA's double sincos is within 2 ulp,
// including its argument reduction).
__device__ inline void cr_cos_sin(float x, float& c, float& s) {
    if (x == 0.0F) {
        c = 1.0F;
        s = x;
        return;
    }
    double sd, cd;
    sincos(static_cast<double>(x), &sd, &cd);
    const bool c_ok = certain_f32(cd, fabs(cd) * kDoubleErr, c);
    const bool s_ok = certain_f32(sd, fabs(sd) * kDoubleErr, s);
    if (c_ok && s_ok) { return; }
    DD cdd, sdd;
    dd_cos_sin(x, cdd, sdd);
    if (!c_ok) { c = f32_of_dd(cdd); }
    if (!s_ok) { s = f32_of_dd(sdd); }
}

// fp32(theta^e) for e = i / n (theta and the quotient binary32): exp(e ln theta), certified in double
// and otherwise in double-double.
__device__ inline float cr_pow_ratio(float theta, int i, int n) {
    const double e = static_cast<double>(i) / static_cast<double>(n);
    const double y = pow(static_cast<double>(theta), e);
    float out;
    if (certain_f32(y, y * kDoubleErr, out)) { return out; }
    // ln(theta) e, then exp of that double-double: exp(hi) (1 + lo) (lo^2 below 2^-100 relatively).
    DD l = dd_log({static_cast<double>(theta), 0.0});
    l    = dd_div(dd_mul(l, {static_cast<double>(i), 0.0}), {static_cast<double>(n), 0.0});
    const DD eh = dd_exp(l.hi);
    return f32_of_dd(dd_mul(eh, {1.0, l.lo}));
}

// 1 / sqrt(v), v a binary32: double sqrt and division are correctly rounded, so the double is within
// one double ulp; near a midpoint m, y > m exactly when 1 > m^2 v (m^2 exact, its product with v
// exact in double-double).
__device__ inline float cr_rsqrt(float v) {
    if (isnan(v) || v < 0.0F) { return NAN; }
    if (v == 0.0F) { return __uint_as_float((__float_as_uint(v) & 0x80000000U) | 0x7F800000U); }
    if (isinf(v)) { return 0.0F; }
    const double y = 1.0 / sqrt(static_cast<double>(v));
    float out;
    if (certain_f32(y, y * 0x1p-51, out)) { return out; }
    // The candidates: out and its neighbour on y's side; decide by the midpoint between them.
    const double o    = static_cast<double>(out);
    const float other = y > o ? nextafterf(out, INFINITY) : nextafterf(out, -INFINITY);
    const double m    = (o + static_cast<double>(other)) / 2; // exact: 25 significant bits
    const DD m2       = dd_two_prod(m, m);
    const DD p        = dd_add(dd_two_prod(m2.hi, static_cast<double>(v)), dd_two_prod(m2.lo, static_cast<double>(v)));
    const DD diff     = dd_add({1.0, 0.0}, {-p.hi, -p.lo}); // 1 - m^2 v: positive when 1/sqrt(v) > m
    const bool above  = diff.hi > 0.0 || (diff.hi == 0.0 && diff.lo > 0.0);
    const bool tie    = diff.hi == 0.0 && diff.lo == 0.0;
    const bool other_above = static_cast<double>(other) > o;
    if (tie) { return (__float_as_uint(out) & 1U) == 0 ? out : other; }
    return above == other_above ? other : out;
}

} // namespace infernix::ops::lossless
