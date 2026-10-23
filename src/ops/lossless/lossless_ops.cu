#include "infernix/ops/lossless_ops.h"

#include "core/device.h"
#include "ops/lossless/cr_math.cuh"
#include "ops/lossless/wide.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// The --lossless route's elementwise and row operations (include/infernix/ops/lossless_ops.h). IEEE
// binary32 multiplies and adds are already correctly rounded (__fmul_rn, __fadd_rn); a BF16 rounding of an
// exact binary32 value is __float2bfloat16_rn; sums go through exact fixed-point accumulators
// (wide.cuh) and are rounded once.

namespace infernix::ops {
namespace {

using lossless::bf16_of_double;
using lossless::round_magnitude;
using lossless::Wide;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless op: ") + message); }
}

bool contiguous(const Tensor& t, DType dtype) { return t.dtype == dtype && t.is_contiguous(); }

__device__ __forceinline__ float bf16_bits_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ std::uint16_t bf16_bits_rn(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// --------------------------------------------------------------------------------- RMS rows

// A BF16 value x = M 2^(e - 134) has x^2 = M^2 2^(2 e - 268) (M^2 < 2^16): the square's position above the
// accumulator's unit 2^-268 is 2 e, below 2 * 254 + 16 + 13 bits for rows of up to 8,192.
constexpr int kSquareWords = 18;
constexpr int kSquareUnit  = -268;

// m = fp32(mean(x^2)) of a row (one warp, every lane returns it). The exact sum is shifted up by 64 bits
// before the division, so the quotient's unit (2^-332) lies below every binary32 ulp and its truncation
// shows only as a sticky bit.
__device__ float row_mean_square(const std::uint16_t* row, int width, int lane) {
    Wide<kSquareWords> acc;
    acc.clear();
    for (int i = lane; i < width; i += 32) {
        const std::uint16_t bits = row[i];
        const int field = (bits >> 7) & 0xFF, mant = bits & 0x7F;
        if (field == 255) { continue; } // non-finite rows are handled by the caller
        const long long M = field == 0 ? mant : (128 | mant);
        if (M == 0) { continue; }
        const int e = field == 0 ? 1 : field;
        acc.add(M * M, 2 * e);
    }
    acc.warp_sum();
    std::uint32_t mag[kSquareWords + 2] = {};
    std::uint32_t sum[kSquareWords];
    (void)acc.finish(sum);
#pragma unroll
    for (int i = 0; i < kSquareWords; ++i) { mag[i + 2] = sum[i]; }
    const bool sticky = lossless::divide_small(mag, static_cast<std::uint32_t>(width));
    return __uint_as_float(round_magnitude(mag, kSquareUnit - 64, false, false, sticky));
}

// Whether a row holds a non-finite value (its mean square is then +inf or NaN by IEEE: NaN in -> NaN).
__device__ bool row_non_finite(const std::uint16_t* row, int width, int lane, bool& nan) {
    bool any = false, has_nan = false;
    for (int i = lane; i < width; i += 32) {
        const std::uint16_t bits = row[i];
        if (((bits >> 7) & 0xFF) == 255) {
            any = true;
            if ((bits & 0x7F) != 0) { has_nan = true; }
        }
    }
    nan = __any_sync(0xFFFFFFFFU, has_nan);
    return __any_sync(0xFFFFFFFFU, any);
}

__device__ __forceinline__ float weight_at(const void* w, bool fp32, int i) {
    return fp32 ? static_cast<const float*>(w)[i] : bf16_bits_float(static_cast<const std::uint16_t*>(w)[i]);
}

// One warp per row.
// Row r takes the weight's group r % groups (groups of `width`).
__global__ void rmsnorm_kernel(const std::uint16_t* x, const void* w, bool w_fp32, int groups,
                               float eps, std::uint16_t* out, int width, int rows) {
    const int row  = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (row >= rows) { return; }
    const std::uint16_t* xr = x + static_cast<std::size_t>(row) * width;
    bool nan = false;
    float m;
    if (row_non_finite(xr, width, lane, nan)) {
        m = nan ? NAN : INFINITY;
    } else {
        m = row_mean_square(xr, width, lane);
    }
    const float r = lossless::cr_rsqrt(__fadd_rn(m, eps));
    std::uint16_t* outr = out + static_cast<std::size_t>(row) * width;
    const int w0 = (row % groups) * width;
    for (int i = lane; i < width; i += 32) {
        const float y    = __fmul_rn(bf16_bits_float(xr[i]), r);
        const float gain = __fadd_rn(1.0F, weight_at(w, w_fp32, w0 + i));
        outr[i]          = bf16_bits_rn(__fmul_rn(y, gain));
    }
}

__global__ void rmsnorm_gated_kernel(const std::uint16_t* __restrict__ o, const std::uint16_t* __restrict__ z,
                                     const void* __restrict__ w, bool w_fp32, bool sigmoid_gate, float eps,
                                     std::uint16_t* __restrict__ out, int width, int rows) {
    const int row  = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (row >= rows) { return; }
    const std::uint16_t* orow = o + static_cast<std::size_t>(row) * width;
    const std::uint16_t* zrow = z + static_cast<std::size_t>(row) * width;
    bool nan = false;
    float m;
    if (row_non_finite(orow, width, lane, nan)) {
        m = nan ? NAN : INFINITY;
    } else {
        m = row_mean_square(orow, width, lane);
    }
    const float r = lossless::cr_rsqrt(__fadd_rn(m, eps));
    std::uint16_t* outr = out + static_cast<std::size_t>(row) * width;
    for (int i = lane; i < width; i += 32) {
        const float n  = bf16_bits_float(bf16_bits_rn(__fmul_rn(bf16_bits_float(orow[i]), r)));
        const float wv = weight_at(w, w_fp32, i);
        // `weight * n`: BF16 times BF16 is BF16 (the product exact in binary32, rounded once to BF16); an
        // FP32 weight promotes it to binary32.
        const float y = w_fp32 ? __fmul_rn(wv, n) : bf16_bits_float(bf16_bits_rn(__fmul_rn(wv, n)));
        const float zv = bf16_bits_float(zrow[i]);
        const float s  = sigmoid_gate ? lossless::cr_sigmoid(zv) : lossless::cr_silu(zv);
        outr[i]       = bf16_bits_rn(__fmul_rn(y, s));
    }
}

// --------------------------------------------------------------------------------- elementwise

__global__ void add_kernel(const std::uint16_t* a, const std::uint16_t* b, std::uint16_t* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    // Two BF16 values add exactly in binary64 unless their exponents are more than 45 apart, and then the
    // smaller lies far below the larger's BF16 half ulp: rounding the binary64 sum once to BF16 (via
    // round-to-odd) is the correctly rounded sum either way.
    const double s = static_cast<double>(bf16_bits_float(a[i])) + static_cast<double>(bf16_bits_float(b[i]));
    out[i]         = bf16_of_double(s);
}

__global__ void silu_mul_kernel(const std::uint16_t* gate, const std::uint16_t* up, std::uint16_t* out, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const float act = bf16_bits_float(lossless::cr_silu_bf16(bf16_bits_float(gate[i])));
    out[i]          = bf16_bits_rn(__fmul_rn(act, bf16_bits_float(up[i]))); // the product is exact in binary32
}

__global__ void sigmoid_mul_kernel(const std::uint16_t* x, const std::uint16_t* gate, std::uint16_t* out,
                                   std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const float s = bf16_bits_float(lossless::cr_sigmoid_bf16(bf16_bits_float(gate[i])));
    out[i]        = bf16_bits_rn(__fmul_rn(bf16_bits_float(x[i]), s));
}

__global__ void gdn_gates_kernel(const std::uint16_t* a, const std::uint16_t* b, const float* a_log,
                                 const float* dt_bias, bool a_direct, float* g, std::uint16_t* beta, int heads,
                                 std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const int h      = static_cast<int>(i % static_cast<std::size_t>(heads));
    beta[i]          = lossless::cr_sigmoid_bf16(bf16_bits_float(b[i]));
    const float t    = __fadd_rn(bf16_bits_float(a[i]), dt_bias[h]);
    const float sp   = lossless::cr_softplus(t);
    if (a_direct) { // a GGUF import's A = -exp(A_log), stored
        g[i] = __fmul_rn(a_log[h], sp);
        return;
    }
    const float decay = lossless::cr_exp(a_log[h]);
    g[i]             = -__fmul_rn(decay, sp);
}

// One thread per (channel, column): the K-term sum exact, rounded once to binary32, then SiLU to BF16.
constexpr int kConvWords = 18; // products M_w M_x 2^(e_w + e_x - 268): positions e_w + e_x in [2, 508] + 16 bits
// Columns form runs of `width` (one per sequence); a run's taps before its first column read its source
// slot of `states` (BF16 [C, K - 1, slots]; null: zeros).
__global__ void causal_conv_kernel(const std::uint16_t* x, const std::uint16_t* w, std::uint16_t* out, int channels,
                                   int columns, int taps, int width, const std::uint16_t* states,
                                   const int* source_slots) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(channels) * columns) { return; }
    const int c = static_cast<int>(i % channels), t = static_cast<int>(i / channels);
    const int b = t / width, run = t % width;
    Wide<kConvWords> acc;
    acc.clear();
    bool nan = false, pos_inf = false, neg_inf = false;
    for (int tap = 0; tap < taps; ++tap) {
        const int src = run - taps + 1 + tap;
        if (src < 0 && states == nullptr) { continue; }
        const std::uint16_t xb =
            src >= 0 ? x[(static_cast<std::size_t>(b) * width + src) * channels + c]
                     : states[(static_cast<std::size_t>(source_slots[b]) * (taps - 1) + (taps - 1 + src)) * channels + c];
        const std::uint16_t wb = w[static_cast<std::size_t>(tap) * channels + c];
        const int xf = (xb >> 7) & 0xFF, wf = (wb >> 7) & 0xFF;
        const int xm = xb & 0x7F, wm = wb & 0x7F;
        const bool negative = ((xb ^ wb) & 0x8000U) != 0;
        if (xf == 255 || wf == 255) {
            const bool xz = xf == 0 && xm == 0, wz = wf == 0 && wm == 0;
            if ((xf == 255 && xm != 0) || (wf == 255 && wm != 0) || xz || wz) {
                nan = true;
            } else if (negative) {
                neg_inf = true;
            } else {
                pos_inf = true;
            }
            continue;
        }
        const long long Mx = xf == 0 ? xm : (128 | xm), Mw = wf == 0 ? wm : (128 | wm);
        if (Mx == 0 || Mw == 0) { continue; }
        const int ex = xf == 0 ? 1 : xf, ew = wf == 0 ? 1 : wf;
        acc.add(negative ? -(Mx * Mw) : Mx * Mw, ex + ew);
    }
    float sum;
    if (nan || (pos_inf && neg_inf)) {
        sum = NAN;
    } else if (pos_inf || neg_inf) {
        sum = neg_inf ? -INFINITY : INFINITY;
    } else {
        std::uint32_t mag[kConvWords];
        const bool negative = acc.finish(mag);
        sum = __uint_as_float(round_magnitude(mag, -268, negative, false));
    }
    out[i] = lossless::cr_silu_bf16(sum);
}

// One thread per (channel, sequence): the destination slot's K - 1 columns become the last K - 1 of the
// source slot's followed by the run, in ascending order (an in-place slot reads each old column before
// it is overwritten).
__global__ void conv_state_kernel(const std::uint16_t* x, std::uint16_t* states, const int* source_slots,
                                  const int* destination_slots, int channels, int sequences, int width, int taps) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= channels * sequences) { return; }
    const int c = i % channels, b = i / channels, keep = taps - 1;
    const std::uint16_t* src = states + static_cast<std::size_t>(source_slots[b]) * keep * channels + c;
    std::uint16_t* dst       = states + static_cast<std::size_t>(destination_slots[b]) * keep * channels + c;
    for (int h = 0; h < keep; ++h) {
        const int s = width + h - keep; // the run's column, or (negative) the old state's column h + width
        dst[static_cast<std::size_t>(h) * channels] =
            s >= 0 ? x[(static_cast<std::size_t>(b) * width + s) * channels + c] : src[static_cast<std::size_t>(h + width) * channels];
    }
}

int blocks_for(std::size_t n, int threads) { return static_cast<int>((n + threads - 1) / threads); }

} // namespace

void lossless_rmsnorm_grouped(const Tensor& x, const Tensor& weight, int groups, float eps, Tensor& out,
                              cudaStream_t stream) {
    require(contiguous(x, DType::BF16) && contiguous(out, DType::BF16) && x.numel() == out.numel() && groups > 0 &&
                (contiguous(weight, DType::BF16) || contiguous(weight, DType::FP32)) && weight.numel() % groups == 0 &&
                x.numel() % weight.numel() == 0 && weight.numel() / groups <= 8192,
            "rmsnorm needs BF16 x and out and a BF16 or FP32 weight of whole groups");
    const int width = static_cast<int>(weight.numel() / groups), rows = static_cast<int>(x.numel() / width);
    rmsnorm_kernel<<<blocks_for(static_cast<std::size_t>(rows) * 32, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), weight.data, weight.dtype == DType::FP32, groups, eps,
        static_cast<std::uint16_t*>(out.data), width, rows);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_rmsnorm(const Tensor& x, const Tensor& weight, float eps, Tensor& out, cudaStream_t stream) {
    require(x.dtype == DType::BF16 && x.ne[0] == out.ne[0] && weight.numel() == x.ne[0],
            "rmsnorm needs BF16 x and out [width, rows] and a weight [width]");
    lossless_rmsnorm_grouped(x, weight, 1, eps, out, stream);
}

void lossless_rmsnorm_gated(const Tensor& o, const Tensor& z, const Tensor& weight, float eps, Tensor& out,
                            cudaStream_t stream, bool sigmoid_gate) {
    require(contiguous(o, DType::BF16) && contiguous(z, DType::BF16) && contiguous(out, DType::BF16) &&
                o.numel() == z.numel() && o.numel() == out.numel() &&
                (contiguous(weight, DType::BF16) || contiguous(weight, DType::FP32)) && o.ne[0] <= 8192,
            "gated rmsnorm needs BF16 o, z and out and a BF16 or FP32 weight");
    const int width = weight.numel() == o.ne[0] ? o.ne[0] : static_cast<int>(weight.numel());
    require(o.numel() % width == 0, "gated rmsnorm rows must be whole");
    const int rows = static_cast<int>(o.numel() / width);
    rmsnorm_gated_kernel<<<blocks_for(static_cast<std::size_t>(rows) * 32, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(o.data), static_cast<const std::uint16_t*>(z.data), weight.data,
        weight.dtype == DType::FP32, sigmoid_gate, eps, static_cast<std::uint16_t*>(out.data), width, rows);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream) {
    require(contiguous(a, DType::BF16) && contiguous(b, DType::BF16) && contiguous(out, DType::BF16) &&
                a.numel() == b.numel() && a.numel() == out.numel(),
            "add needs BF16 tensors of one size");
    const std::size_t n = static_cast<std::size_t>(a.numel());
    add_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(a.data),
                                                       static_cast<const std::uint16_t*>(b.data),
                                                       static_cast<std::uint16_t*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    require(contiguous(gate, DType::BF16) && contiguous(up, DType::BF16) && contiguous(out, DType::BF16) &&
                gate.numel() == up.numel() && gate.numel() == out.numel(),
            "silu_mul needs BF16 tensors of one size");
    const std::size_t n = static_cast<std::size_t>(gate.numel());
    silu_mul_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(gate.data),
                                                            static_cast<const std::uint16_t*>(up.data),
                                                            static_cast<std::uint16_t*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_sigmoid_mul(const Tensor& x, const Tensor& gate, Tensor& out, cudaStream_t stream) {
    require(contiguous(x, DType::BF16) && contiguous(gate, DType::BF16) && contiguous(out, DType::BF16) &&
                x.numel() == gate.numel() && x.numel() == out.numel(),
            "sigmoid_mul needs BF16 tensors of one size");
    const std::size_t n = static_cast<std::size_t>(x.numel());
    sigmoid_mul_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(static_cast<const std::uint16_t*>(x.data),
                                                               static_cast<const std::uint16_t*>(gate.data),
                                                               static_cast<std::uint16_t*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_gdn_gates(const Tensor& a, const Tensor& b, const Tensor& a_log, const Tensor& dt_bias, Tensor& g,
                        Tensor& beta, cudaStream_t stream, bool a_direct) {
    require(contiguous(a, DType::BF16) && contiguous(b, DType::BF16) && a.numel() == b.numel() &&
                contiguous(a_log, DType::FP32) && contiguous(dt_bias, DType::FP32) && a_log.numel() == a.ne[0] &&
                dt_bias.numel() == a.ne[0] && contiguous(g, DType::FP32) && contiguous(beta, DType::BF16) &&
                g.numel() == a.numel() && beta.numel() == a.numel(),
            "gdn gates need BF16 a, b [H, T], FP32 A_log and dt_bias [H], FP32 g and BF16 beta [H, T]");
    const std::size_t n = static_cast<std::size_t>(a.numel());
    gdn_gates_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(a.data), static_cast<const std::uint16_t*>(b.data),
        static_cast<const float*>(a_log.data), static_cast<const float*>(dt_bias.data), a_direct,
        static_cast<float*>(g.data), static_cast<std::uint16_t*>(beta.data), a.ne[0], n);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_causal_conv(const Tensor& x, const Tensor& weight, Tensor& out, cudaStream_t stream) {
    require(contiguous(x, DType::BF16) && contiguous(weight, DType::BF16) && contiguous(out, DType::BF16) &&
                weight.ne[0] == x.ne[0] && x.numel() == out.numel() && weight.ne[1] >= 1 && weight.ne[1] <= 8,
            "causal conv needs BF16 x and out [C, T] and a BF16 weight [C, K] (tap-major)");
    const int channels = x.ne[0], columns = static_cast<int>(x.numel() / channels), taps = weight.ne[1];
    const std::size_t n = static_cast<std::size_t>(channels) * columns;
    if (n == 0) { return; }
    causal_conv_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), static_cast<const std::uint16_t*>(weight.data),
        static_cast<std::uint16_t*>(out.data), channels, columns, taps, columns, nullptr, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_causal_conv_slots(const Tensor& x, const Tensor& weight, int sequences, Tensor& states,
                                const Tensor& source_slots, const Tensor& destination_slots, Tensor& out,
                                cudaStream_t stream) {
    require(contiguous(x, DType::BF16) && contiguous(weight, DType::BF16) && contiguous(out, DType::BF16) &&
                weight.ne[0] == x.ne[0] && x.numel() == out.numel() && weight.ne[1] >= 2 && weight.ne[1] <= 8,
            "causal conv needs BF16 x and out [C, T] and a BF16 weight [C, K] (tap-major), K >= 2");
    const int channels = x.ne[0], columns = static_cast<int>(x.numel() / channels), taps = weight.ne[1];
    require(sequences >= 1 && columns % sequences == 0, "the columns are `sequences` equal runs");
    require(contiguous(states, DType::BF16) && states.ne[0] == channels && states.ne[1] == taps - 1,
            "states BF16 [C, K - 1, slots]");
    require(source_slots.dtype == DType::I32 && destination_slots.dtype == DType::I32 &&
                source_slots.is_contiguous() && destination_slots.is_contiguous() &&
                source_slots.numel() >= sequences && destination_slots.numel() >= sequences,
            "source and destination slots I32 [B]");
    if (columns == 0) { return; }
    const std::size_t n = static_cast<std::size_t>(channels) * columns;
    causal_conv_kernel<<<blocks_for(n, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), static_cast<const std::uint16_t*>(weight.data),
        static_cast<std::uint16_t*>(out.data), channels, columns, taps, columns / sequences,
        static_cast<const std::uint16_t*>(states.data), static_cast<const int*>(source_slots.data));
    CUDA_CHECK(cudaGetLastError());
    conv_state_kernel<<<blocks_for(static_cast<std::size_t>(channels) * sequences, 256), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), static_cast<std::uint16_t*>(states.data),
        static_cast<const int*>(source_slots.data), static_cast<const int*>(destination_slots.data), channels,
        sequences, columns / sequences, taps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops

// This unit's copy of the noise twin's seed (ops/lossless/noise.cuh).
INFERNIX_LOSSLESS_NOISE_UNIT(ops)

namespace infernix::ops {
namespace lossless {
// Each rounding unit's setter (INFERNIX_LOSSLESS_NOISE_UNIT).
void set_noise_seed_attention(unsigned long long seed, float scale);
void set_noise_seed_flash(unsigned long long seed, float scale);
void set_noise_seed_gdn(unsigned long long seed, float scale);
void set_noise_seed_moe(unsigned long long seed, float scale);
void set_noise_seed_rope(unsigned long long seed, float scale);
void set_noise_seed_linear(unsigned long long seed, float scale);
void set_noise_seed_linear_ggml_i(unsigned long long seed, float scale);
void set_noise_seed_linear_ggml_k(unsigned long long seed, float scale);
} // namespace lossless

void lossless_set_noise(std::uint64_t seed, std::uint32_t units, float scale) {
    const auto pick = [&](LosslessNoiseUnit unit) -> unsigned long long {
        return (units & static_cast<std::uint32_t>(unit)) != 0 ? seed : 0ULL;
    };
    lossless::set_noise_seed_ops(pick(LosslessNoiseUnit::Elementwise), scale);
    lossless::set_noise_seed_attention(pick(LosslessNoiseUnit::Attention), scale);
    lossless::set_noise_seed_flash(pick(LosslessNoiseUnit::Flash), scale);
    lossless::set_noise_seed_gdn(pick(LosslessNoiseUnit::DeltaRule), scale);
    lossless::set_noise_seed_moe(pick(LosslessNoiseUnit::Moe), scale);
    lossless::set_noise_seed_rope(pick(LosslessNoiseUnit::Rope), scale);
    lossless::set_noise_seed_linear(pick(LosslessNoiseUnit::Linear), scale);
    lossless::set_noise_seed_linear_ggml_i(pick(LosslessNoiseUnit::Gguf), scale);
    lossless::set_noise_seed_linear_ggml_k(pick(LosslessNoiseUnit::Gguf), scale);
}

void lossless_set_noise_seed(std::uint64_t seed) { lossless_set_noise(seed, kAllLosslessNoiseUnits, 1.0F); }

} // namespace infernix::ops
