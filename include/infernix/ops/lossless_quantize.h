#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

// The design reference's activation quantizers (docs/maintainer/lossless-route.md §3b). Each applies a
// rule of the reproducible route exactly to a BF16 activation and writes the quantized values without
// their activation scale, each exact in BF16; lossless_linear applies the scale to the exact sum
// (LosslessActivationScale, LosslessTile::act). x and out are contiguous BF16 [K, T] and may not overlap.

/**
 * Op: lossless_quantize_nvfp4 (Qwen3.5's NVFP4 activations: quantize_nvfp4_k16, the divisor form)
 *
 * Each 16 consecutive K elements v of a column (K a multiple of 16), with the activation divisor D:
 * w = e4m3_rn_satfinite(fp32(fp32(D amax) / 6)), amax their largest magnitude; a zero w makes the block
 * zero; else c = e2m1_rn_satfinite(fp32(fp32(v D) / e4m3(w))) and out = c e4m3(w). The quantized
 * activation is out / D.
 */
void lossless_quantize_nvfp4(const Tensor& x, float divisor, Tensor& out, cudaStream_t stream);

/**
 * Op: lossless_quantize_nvfp4_scaled (Qwen3.8-Flash-Next's W4A4 experts: NVIDIA ModelOpt's rule,
 * quantize_a4_codes)
 *
 * Column c of out takes column in_columns[c] of x (null: c itself; negative: a zero column) and the
 * multiplier g = column_scales[c] (device FP32 [T], positive; its expert's input scale): per 16
 * elements s = e4m3_rn_satfinite(fp32(amax / fp32(6 g))), c = e2m1_rn_satfinite(fp32(v / fp32(e4m3(s) g)))
 * and out = c e4m3(s). The quantized activation is out g. out is [K, T] with T its own columns.
 */
void lossless_quantize_nvfp4_scaled(const Tensor& x, const std::int32_t* in_columns, const float* column_scales,
                                    Tensor& out, cudaStream_t stream);

/**
 * Op: lossless_quantize_fp8_token (the FP8 projections' A8: fp8_a8.cu's per-token rule)
 *
 * Per column t over its whole K: s_t = fp32(amax_t / 448) (0 for a zero column) and
 * c = e4m3_rn_satfinite(fp32(v fp32(1 / s_t))); out = e4m3(c) and scales (device FP32 [T]) s_t. The
 * quantized activation is out s_t.
 */
void lossless_quantize_fp8_token(const Tensor& x, Tensor& out, Tensor& scales, cudaStream_t stream);

/**
 * Op: lossless_quantize_fp8_static (a static A8 site: fp8_e4m3fn_row_f32's calibrated input scale,
 * fp8_a8.cu's static rule)
 *
 * Every element with the input scale s (positive, finite): c = e4m3_rn_satfinite(fp32(v fp32(1 / s)));
 * out = e4m3(c) and scales (device FP32 [T]) s for every column. The quantized activation is out s.
 */
void lossless_quantize_fp8_static(const Tensor& x, float input_scale, Tensor& out, Tensor& scales,
                                  cudaStream_t stream);

/**
 * Op: lossless_round_fp16 (the bf16 KV format's V: FP16 storage)
 *
 * Each element of x (BF16, any contiguous shape), in place: its FP16 rounding (round to nearest even; the
 * subnormals, and infinity from 65520), which is again exact in BF16.
 */
void lossless_round_fp16(Tensor& x, cudaStream_t stream);

} // namespace infernix::ops
