#pragma once

#include "ops/linear/q8/q8_geometry.h"
#include "infernix/ops/linear.h"
#include "ops/linear/q8/q8_launch.h"

namespace infernix::ops::detail {

using Q8N1024K2048   = Q8LinearGeometry<1024, 2048>;
using Q8N1024K5120   = Q8LinearGeometry<1024, 5120>;
using Q8N2048K4096   = Q8LinearGeometry<2048, 4096>;
using Q8N2048K4608   = Q8LinearGeometry<2048, 4608>;
using Q8N2048K16384  = Q8LinearGeometry<2048, 16384>;
using Q8N2560K6144   = Q8LinearGeometry<2560, 6144>;
using Q8N4608K4608   = Q8LinearGeometry<4608, 4608>;
using Q8N5120K4608   = Q8LinearGeometry<5120, 4608>;
using Q8N5120K6144   = Q8LinearGeometry<5120, 6144>;
using Q8N5120K10240  = Q8LinearGeometry<5120, 10240>;
using Q8N5120K17408  = Q8LinearGeometry<5120, 17408>;
using Q8N5120K25600  = Q8LinearGeometry<5120, 25600>;
using Q8N6144K2560   = Q8LinearGeometry<6144, 2560>;
using Q8N6144K5120   = Q8LinearGeometry<6144, 5120>;
using Q8N9216K2048   = Q8LinearGeometry<9216, 2048>;
using Q8N10240K2560  = Q8LinearGeometry<10240, 2560>;
using Q8N12288K2048  = Q8LinearGeometry<12288, 2048>;
using Q8N12288K2560  = Q8LinearGeometry<12288, 2560>;
using Q8N14336K5120  = Q8LinearGeometry<14336, 5120>;
using Q8N16384K2560  = Q8LinearGeometry<16384, 2560>;
using Q8N17408K5120  = Q8LinearGeometry<17408, 5120>;
using Q8N34816K5120  = Q8LinearGeometry<34816, 5120>;
using Q8N248320K5120 = Q8LinearGeometry<248320, 5120>;

// The dispatcher admits N/K and positive T; each shape owns its complete T selection.
[[nodiscard]] Q8Launch select_q8_n1024_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k4096(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2048_k16384(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2560_k6144(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n4608_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k4608(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k10240(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k17408(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n5120_k25600(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n6144_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n9216_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n10240_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n12288_k2048(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n12288_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n16384_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n17408_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n248320_k5120(std::int32_t tokens);
// A16ColumnInvariant for the output head: one 8-K-warp sliced-K family for every T.
[[nodiscard]] Q8Launch select_q8_n248320_k5120_invariant(std::int32_t tokens);
// Qwen3.8-Flash-Next's dense Q8 classes (register-streamed route up to 64 columns).
[[nodiscard]] Q8Launch select_q8_n16384_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n12800_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n13952_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2560_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n1280_k2560(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2560_k6144(std::int32_t tokens);
// The q8-sliced-k fast path (LinearRoutes::Q8SlicedK): NInfer's GEMV, sliced-K and grouped sliced-K
// routes for the same two problems, whose accumulation order differs from the exact routes'.
[[nodiscard]] Q8Launch select_q8_n16384_k2560_fast(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2560_k6144_fast(std::int32_t tokens);
// The [16384,2560] GDN projection on the same routes, rows [0,10240) written to qkv [10240,T] and
// rows [10240,16384) to z [6144,T] (gdn_input_proj).
void launch_q8_n16384_k2560_split(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  LinearRoutes routes, cudaStream_t stream);
// The same projection over x [2560,B] (B = 1..8 sequences of one position each) with the GDN causal
// convolution in its epilogue (gdn_input_proj_conv_snapshot's routes form): the decode routes of
// the selected table, each row's FP32 projection consumed as the current tap.
void launch_q8_n16384_k2560_conv_snapshot(const Tensor& x, const Weight& weight,
                                          const Tensor& conv_weight, Tensor& conv_states,
                                          const Tensor& initial_state_slots,
                                          const Tensor& snapshot_base_slots, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z, LinearRoutes routes,
                                          cudaStream_t stream);
[[nodiscard]] Q8Launch select_q8_n324_k10240(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n320_k10240(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n10240_k320(std::int32_t tokens);
[[nodiscard]] Q8Launch select_q8_n2560_k640(std::int32_t tokens);

} // namespace infernix::ops::detail
