#pragma once

// RTX 5090 (GB202, sm_120a) peaks for the Op benchmarks' roofline percentages, one place for every
// benchmark. The Tensor Core figures are NVIDIA's dense peaks at boost clock from its GB202 table, with
// input and accumulator precision explicit; the read ceiling is this repository's own measurement.
namespace infernix::bench::roofline {

// GDDR7, 28 Gbps on 512 bits.
inline constexpr double kDramSpecGBs = 1792.0;
// The measured pure-read ceiling (tools/hbm_bandwidth_probe.cu): the practical bound of read-dominated points.
inline constexpr double kDramReadGBs = 1674.5;

inline constexpr double kFp8Fp16AccumulateTFLOPs = 838.0;
inline constexpr double kFp8Fp32AccumulateTFLOPs = 419.0;
// Unit-scale block-scaled FP8 takes the full-rate FP32 accumulation path.
inline constexpr double kMxfp8Fp32AccumulateTFLOPs = 838.0;
// Block-scaled FP4 (NVFP4) with FP32 accumulation.
inline constexpr double kNvfp4Fp32AccumulateTFLOPs = 1676.0;
inline constexpr double kBf16Fp32AccumulateTFLOPs  = 209.5;
// FP16 operands with FP16 accumulation (the f16-accumulate-prefill route): twice the FP32-accumulation rate.
inline constexpr double kFp16Fp16AccumulateTFLOPs = 419.0;

} // namespace infernix::bench::roofline
