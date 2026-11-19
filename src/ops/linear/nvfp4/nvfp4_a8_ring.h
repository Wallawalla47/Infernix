#pragma once

// The 8-bit NVFP4 ring's launchers (nvfp4_a8_ring.cuh): one per output, epilogue and row policy whose projection runs
// the 128 x 128 tile, each defined in its family's non-RDC translation unit (the ring's setmaxnreg needs one;
// src/ops/CMakeLists.txt). launch_nvfp4_a8 (nvfp4_a8_mma.cuh) calls the overload its arguments select.

#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

struct Nvfp4A8Operands;
struct Nvfp4SwiGluEpilogue;
struct Nvfp4SwiGluMmaRows;

// ops::linear (nvfp4_a8_linear_ring.cu).
void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16Output output, LinearIdentityEpilogue epilogue,
                          Nvfp4IdentityRows rows, cudaStream_t stream);
void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearFp32Output output, LinearIdentityEpilogue epilogue,
                          Nvfp4IdentityRows rows, cudaStream_t stream);
// ops::linear_swiglu (linear_swiglu/nvfp4/nvfp4_linear_swiglu_a8_ring.cu).
void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16Output output, Nvfp4SwiGluEpilogue epilogue,
                          Nvfp4SwiGluMmaRows rows, cudaStream_t stream);
// ops::linear_add (linear_add/nvfp4/nvfp4_linear_add_a8_ring.cu).
void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16Output output, LinearResidualAddEpilogue epilogue,
                          Nvfp4IdentityRows rows, cudaStream_t stream);
// ops::gdn_input_proj's QKV and Z outputs (gdn_input_proj/nvfp4/nvfp4_gdn_input_a8_ring.cu).
void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16SegmentedOutput<10240, 6144> output,
                          LinearIdentityEpilogue epilogue, Nvfp4IdentityRows rows, cudaStream_t stream);
// ops::attn_input_proj's Q, K, gate and V outputs (attn_input_proj/nvfp4/nvfp4_attn_input_a8_ring.cu).
void launch_nvfp4_a8_ring(const Nvfp4A8Operands& p, LinearBf16SegmentedOutput<6144, 1024, 6144, 1024> output,
                          LinearIdentityEpilogue epilogue, Nvfp4IdentityRows rows, cudaStream_t stream);

} // namespace infernix::ops::detail
