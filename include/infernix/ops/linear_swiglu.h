#pragma once

// infernix::ops - fused gate/up projection followed by SwiGLU.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Returns the transient capacity required by LinearSwiGLU for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype,
                                                                 std::int32_t gate_up_rows,
                                                                 std::int32_t input_rows,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

/**
 * Policy-bearing capacity query. Q4/Q8 use A16 under every policy. NVFP4 uses A16 under
 * A16Only/AllowA8 and accepts every positive T under every policy. Row-scaled FP8 accepts all
 * policies, with A8 permitted by AllowA8/AllowA4.
 * A permissive policy covers whichever qualified route the private resolver selects across the
 * requested interval. `routes` must be the flags the calls pass (linear_swiglu's Routes).
 */
[[nodiscard]] std::size_t
linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                       std::int32_t input_rows, LinearPolicy policy,
                                       std::int32_t min_tokens, std::int32_t max_tokens,
                                       LinearRoutes routes = LinearRoutes::Exact);

/**
 * Whether the registered profile has a dedicated column-invariant route (Q4, NVFP4 and FP8).
 * Without one, LinearPolicy::A16ColumnInvariant runs the exact A16 route once per column.
 */
[[nodiscard]] bool linear_swiglu_column_invariant_route(QType qtype, std::int32_t gate_up_rows,
                                                        std::int32_t input_rows) noexcept;

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = Linear(x, gate_up_weight); M=gate_up_rows/2;
 *   ideal[i,t] = SiLU(gate_up[i,t]) * gate_up[M+i,t].
 *
 * Logical shapes / supported domain:
 *   T may be any positive value. The registered profiles are:
 *   - Q4_G64_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - Q8_G32_FP16 weight [12288,2048], x [2048,T], out [6144,T];
 *   - Q8_G32_FP16 weight [1280,2560], x [2560,T], out [640,T] (Qwen3.8-Flash-Next's shared expert;
 *     at T <= 16 a column's result does not depend on T);
 *   - Q8_G32_FP16 weight [34816,5120], x [5120,T], out [17408,T];
 *   - Q8_G32_FP16 weight [15360,2560], x [2560,T], out [7680,T] (Qwen3.8-Flash-Next's DFlash2 MLP);
 *   - NVFP4 BlockScaleK16M128x4 weight [34816,5120], x [5120,T], out [17408,T];
 *   - FP8_E4M3FN_ROW_BF16 RowScale weight [34816,5120], x [5120,T], out [17408,T];
 *   - FP8_E4M3FN_ROW_F32 RowScale weight [2048,3072], x [3072,T], out [1024,T] (Qwen3.5-122B-A10B's
 *     per-row FP8 shared expert: 16-bit activations under every policy, with a column-invariant route).
 *   Inputs and output are contiguous BF16. Q4/Q8 scales are FP16, NVFP4 scales are E4M3FN, and
 *   row-scaled FP8 has one BF16 (FP32 for FP8_E4M3FN_ROW_F32) multiplier per gate/up parent row. Gate
 *   rows `[0,M)` precede their matching up rows `[M,2M)` (M = 17408, 7680, 6144, 1024 or 640).
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates `ideal` naively in FP64 from the
 *   represented inputs. The BF16 output is promoted and compared directly with that result; output
 *   storage rounding belongs to LinearSwiGLU's named activation-compute criterion, not the oracle.
 *   Production routes may fuse or materialize gate/up and may choose their natural accumulator,
 *   staging, and workspace precision; those private choices are not semantic rounding boundaries.
 *   AllowA8/AllowA4 permit FP8 activation quantization; route thresholds are implementation
 * choices.
 *
 * Routes:
 *   `routes` admits fast-path routes (LinearRoutes); without a flag every profile runs its exact
 *   route. Under LinearRoutes::Q4W4A8 the Q4 profile serves prefill widths (a private column
 *   threshold) on the q4-w4a8 route, whatever `policy`: each token's activation is quantized to
 *   int8 against its own absmax (scale max|x|/127), the dot product of every 64-element weight group
 *   is exact in integers, and the group's FP16 scale and the token scale are applied in FP32. It meets
 *   the int8 per-token activation criterion; narrower calls keep the exact route's bits. A token with
 *   a non-finite element yields non-finite outputs. Other profiles ignore that flag.
 *
 * Effects:
 *   Writes the full output; x/weight and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_swiglu_workspace_capacity_bytes(),
 *   scoped to the call. Q8, NVFP4 A16, and row-scaled FP8 A16 require zero bytes; A4/A8 routes and
 *   the q4-w4a8 route use caller-owned activation storage and may use private projection storage.
 *   There is no persistent state side effect.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream,
                   LinearRoutes routes = LinearRoutes::Exact);

/**
 * A16-only convenience form. Q4/Q8, NVFP4 and row-scaled FP8 retain their complete positive-T
 * domain.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

} // namespace infernix::ops
