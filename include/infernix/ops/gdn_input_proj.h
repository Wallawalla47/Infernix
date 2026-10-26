#pragma once

// infernix::ops - fused GDN Q/K/V/Z input projections.

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/speculative_tree.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * Op: gdn_input_proj
 *
 * Math / indexing:
 *   qkv[:,t] = concat(qk_weight * x[:,t], value_z_weight[0:6144,:] * x[:,t])
 *   z[:,t]   = value_z_weight[6144:12288,:] * x[:,t]
 *
 * Logical shapes:
 *   x [5120,T], qk weight/output rows 4096, value/z rows 6144 each, qkv [10240,T],
 *   z [6144,T]. T may be any positive value. x, qkv, and z are contiguous BF16.
 *   qk_weight is Q4_G64_FP16 RowSplit [4096,5120] and value_z_weight is one
 *   Q5_G64_FP16 RowSplit parent [12288,5120] in [value,z] row order, both with FP16 scales.
 *
 * Numeric:
 *   The oracle exact-decodes both weight parents and evaluates all four logical projections
 *   naively in FP64 from the represented input. The BF16 qkv and z outputs are promoted and
 *   compared directly with those ideal values; final output storage rounding belongs to
 *   GdnInputProj's named A16 criterion, not the oracle. Production routes may choose their
 *   private precision independently; every registered route writes both final allocations.
 *
 * Effects:
 *   Writes the full qkv and z outputs; inputs and outputs must not alias.
 *
 * Workspace:
 *   No transient bytes are required.
 */
void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, cudaStream_t stream);

/**
 * The two-parent Q4/Q5 projection under a compute policy. Every policy computes A16; under
 * A16ColumnInvariant a column's qkv and z do not depend on the call's width or other columns.
 */
void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, LinearPolicy policy, cudaStream_t stream);

/**
 * Single-parent GDN projection. Registered parent forms are:
 *
 * - Q8_G32_FP16 RowSplit [12288,2048], with stored row counts [2048,2048,4096,4096];
 * - Q8_G32_FP16 [16384,2560] (Qwen3.8-Flash-Next), with stored row counts [2048,2048,6144,6144],
 *   on the routes linear selects for that shape, so qkv and z hold exactly the bits of the
 *   corresponding rows of linear's [16384,T] result;
 * - FP8_E4M3FN_G32_E8M0 G32ScaleRows16 [16384,2560] (NVIDIA Qwen3.8-Flash-Next-FP4-Local, MXFP8),
 *   with stored row counts [2048,2048,6144,6144], on linear's route for that shape under every
 *   policy (16-bit activations, column-invariant at every T), so qkv and z hold exactly the bits of
 *   the corresponding rows of linear's [16384,T] result;
 * - NVFP4 BlockScaleK16M128x4 [16384,5120], with stored row counts [2048,2048,6144,6144].
 * - FP8_E4M3FN_ROW_BF16 RowScale [16384,5120], with stored row counts
 *   [2048,2048,6144,6144].
 * - FP8_E4M3FN_ROW_F32 RowScale [20480,3072] (Qwen3.5-122B-A10B: FP32 row multipliers and one
 *   static activation scale s), with stored row counts [2048,2048,8192,8192]. Its A8 route
 *   encodes x as e4m3_rn_satfinite(fl32(x * fl32(1 / s))); a per-token parent (Weight::input_scale
 *   0) takes each column's own s = fl32(max |x| / 448). It has no A8 residual route.
 *
 * The first three ranges are written contiguously to qkv and the final range is written to z.
 * Q8 uses A16 under every policy. Under A16ColumnInvariant a column's outputs do not depend on the
 * call's width (gdn_input_proj_column_invariant_route). NVFP4 uses A16 under A16Only/AllowA8;
 * AllowA4 permits
 * private activation quantization at every positive T. FP8 accepts all policies; AllowA8/AllowA4
 * selects A16 through T=7 and private activation quantization followed by A8 Tensor Core
 * contraction at every T>=8. Every route writes the two independent final allocations directly.
 * The complete projection is evaluated against the same exact-decode/naive-FP64 oracle;
 * activation quantization and the production reduction profile are private effects covered by the
 * selected criterion. x, both persistent weight planes, qkv, z, and the live workspace must be
 * mutually non-overlapping.
 *
 * The policy-bearing form uses caller-owned call-scoped transient storage sized by
 * gdn_input_proj_workspace_capacity_bytes(). A16 requires zero bytes, except that the MXFP8
 * [16384,2560] query also covers its LinearRoutes::F16Accumulate route from that route's admission
 * width (the query takes no route flags). With that flag (and any policy but A16ColumnInvariant), the
 * MXFP8 parent runs linear's f16-accumulate-prefill route for the shape (include/infernix/ops/linear.h)
 * at the same widths, so qkv and z still hold the bits of linear's rows under the same flag; every
 * other parent ignores that flag, and LinearRoutes::Q8SlicedK acts as in the routes overload below.
 * The convenience overload selects A16Only and requires no
 * transient workspace.
 */
[[nodiscard]] std::size_t
gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                        std::int32_t input_rows, LinearPolicy policy,
                                        std::int32_t min_tokens, std::int32_t max_tokens);

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream,
                    LinearRoutes routes = LinearRoutes::Exact);

/**
 * Whether a single parent has a dedicated column-invariant route (NVFP4, FP8, MXFP8 and Q8
 * [12288,2048]).
 * Without one (Q8 [16384,2560]), LinearPolicy::A16ColumnInvariant runs the exact route once per
 * column. The two-parent Q4/Q5 form always has one.
 */
[[nodiscard]] bool gdn_input_proj_column_invariant_route(QType parent_qtype,
                                                         std::int32_t parent_rows,
                                                         std::int32_t input_rows) noexcept;

/**
 * Applies the A16-only single-parent GDN projection without transient workspace. With
 * LinearRoutes::Q8SlicedK, the Q8 [16384,2560] parent runs the q8-sliced-k fast routes linear selects
 * for that problem (include/infernix/ops/linear.h); every other parent runs its exact route.
 */
void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream, LinearRoutes routes = LinearRoutes::Exact);

/**
 * Returns the transient capacity required by the registered two-parent Q4/Q5 or single-parent Q8
 * snapshot profile. `batch_size` is exact and the query covers every W in the inclusive width
 * interval. B=1 preserves the format-specific fused/composed resolver. B=2..8 uses aggregate
 * projection plus one BF16 [C,B*W] projected plane. The query throws for an unregistered row
 * profile or unsupported B/W domain.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * The same query for a compute policy. Under A16ColumnInvariant every B and W composes
 * the projection and the convolution through one BF16 [C,B*W] projected plane.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Returns the transient capacity for a registered [16384,5120] NVFP4 or row-scaled FP8 snapshot
 * profile. `batch_size` is exact and the query covers every W in the inclusive width interval.
 * B=1 preserves the format-specific fused/materialized resolver; B=2..8 covers its aggregate
 * projection mechanism plus any projected BF16 plane selected by the complete-Op plan.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Op: gdn_input_proj_conv_snapshot
 *
 * Math / indexing:
 *   For each row b, let p[:,j,b] be the concatenated q/k/value projection of x[:,j,b], while
 *   z[:,j,b] is the independent z projection. Starting from the BF16 width-three history selected
 *   by initial_state_slots[b], evaluate the width-four depthwise convolution over p, apply SiLU,
 *   and publish its channel ranges to query, key, and value. After valid column j, write the new
 *   width-three history to snapshot_base_slots[b]+j. Z bypasses convolution.
 *
 * Logical shapes:
 *   The 27B registered form has x [5120,W,B], Q4 q/k weight [4096,5120], one Q5 value/z parent
 *   [12288,5120], conv_weight [10240,4], conv_states [10240,3,Slots], query/key [2048,W,B],
 *   value/z [6144,W,B], and I32 selectors [B]. B=1 accepts every positive W; B=2..8 accepts
 *   W=1..64. `valid_columns` is empty for a dense invocation or I32 [B] for a mixed-width batch.
 *   A mixed-width invocation has B>=1 and every valid extent lies in [1,W].
 *
 * Numeric:
 *   The oracle exact-decodes packed weights and evaluates projection, convolution, SiLU, z, and
 *   every snapshot value naively in FP64 from represented inputs. BF16 query/key/value/z and
 *   snapshots are promoted and compared directly with those ideal values; their final storage
 *   rounding belongs to the Op's named A16 criterion, not the oracle. Former unfused projection
 *   tensors are not observable cast boundaries; production routes use their natural private
 *   accumulator and staging precision. This two-parent Q4/Q5 form does not quantize activation;
 *   the single-parent policy-bearing form below defines its own permitted compute profiles.
 *
 * Effects:
 *   Each row writes query/key/value through its valid prefix and exact zero to its invalid tail;
 *   z is projected for all B*W safe input columns. A row writes only its valid destination state
 *   prefix. The caller reserves disjoint complete [base,base+W) intervals, prevents one row from
 *   overwriting another row's initial slot, and may overlap a row's own initial slot with its
 *   destination after that initial history has been loaded. Other slots are unchanged. Newly
 *   projected convolution channels remain private to the call while published snapshots are BF16.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * The two-parent form under a compute policy. Every policy computes A16; A16ColumnInvariant
 * composes the column-invariant projection and the convolution at every B and W, so a column's
 * outputs and snapshots do not depend on the call's width, batch or other columns.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Single-parent form of gdn_input_proj_conv_snapshot. Registered parents are Q8_G32_FP16 RowSplit
 * [12288,2048], NVFP4 BlockScaleK16M128x4 [16384,5120], FP8_E4M3FN_ROW_BF16 RowScale
 * [16384,5120] and FP8_E4M3FN_ROW_F32 RowScale [20480,3072], all in q/k/value/z row order. All
 * policies permit Q8 A16. NVFP4 uses A16 under A16Only/AllowA8; AllowA4 may use A4. FP8 may use A8 under AllowA8/AllowA4. B=1 accepts
 * every positive W for FP8; the batched domain is B=2..8 and W=1..64. Tensor operands, the complete
 * FP8 parent, and live workspace must be mutually non-overlapping, except that the read-only
 * initial_state_slots and snapshot_base_slots selectors may alias each other; same-row state-slot
 * overlap remains governed by the snapshot state contract.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Applies the A16-only single-parent form. FP8 accepts every positive W for dense B=1; the batched
 * domain is B=2..8 and W=1..64.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Routes form of the single-parent snapshot Op for the Q8_G32_FP16 [16384,2560] parent
 * (Qwen3.8-Flash-Next) in q/k/value/z row order, stored row counts [2048,2048,6144,6144], at W=1:
 * x [2560,1,B], conv_weight [10240,4], conv_states [10240,3,Slots], query/key [2048,1,B],
 * value/z [6144,1,B] and I32 [B] selectors, B=1..8. With one column per row there is no
 * valid-column selector. The Math, Effects and state rules are gdn_input_proj_conv_snapshot's;
 * no transient workspace is used.
 *
 * Numeric: the oracle is the snapshot Op's (exact-decode projection, convolution, SiLU and
 * snapshot in FP64 from the represented inputs). `routes` selects the projection's route set as
 * linear does for this problem (LinearRoutes::Q8SlicedK admits the q8-sliced-k routes). The projection
 * is evaluated on those routes, so z and the newest published history value are exactly the bits
 * of the corresponding rows of linear's [16384,B] result under the same routes; the two older
 * published history values are the initial window's BF16 values. The convolution consumes the
 * route's FP32 projected value as its current tap. It therefore differs from the composed
 * gdn_input_proj + causal_conv1d_silu_snapshot_split route, which convolves bf16(projection),
 * only in that one input of query/key/value: this form is the gdn-conv-fused fast path, and the
 * composed route stays the exact one. Operands and the parent are mutually non-overlapping except
 * that the read-only selectors may alias; initial_state_slots[b] may equal snapshot_base_slots[b].
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearRoutes routes,
                                  cudaStream_t stream);

/**
 * Returns the transient capacity for the registered Q4/Q5 or Q8 record-producing profile.
 * `batch_size` is exact, and the inclusive T interval must lie within ReplaySSM's B=1..8,
 * T=2..16 execution domain, extended to T=64 at B=1. These profiles require no transient storage
 * because materialized projection writes directly to caller-owned conv_record.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/// The same query for a compute policy; under A16ColumnInvariant every B and T composes.
[[nodiscard]] std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Returns the transient capacity for a registered [16384,5120] NVFP4 or row-scaled FP8
 * record-producing profile. Fused and materialized A16 routes require no storage. AllowA4/AllowA8
 * returns only the activation-quantization workspace selected by this complete-Op route;
 * conv_record is caller-owned.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Op: gdn_input_proj_conv_record
 *
 * For each batch row, evaluates the registered projection, width-four causal convolution, SiLU,
 * and q/k/value/z split from the BF16 history selected by initial_state_slots. It writes each BF16
 * newest history column to conv_record [C,T,B]. Query, key,
 * and value are zero in each row's invalid tail; z is projected for every physical column.
 *
 * The execution domain is B=1..8 and T=2..16, plus B=1 and T=17..64.
 * valid_columns is empty for dense input or device
 * I32 [B], with each caller-supplied extent in [1,T]. conv_states is a read-only BF16 [C,3,S]
 * state-pool view, and initial_state_slots contains absolute slots in [0,S). Source state is not
 * modified. Only the valid prefix of conv_record is semantically defined. Outputs and valid
 * record columns are bit-identical to the corresponding snapshot execution from the same initial
 * history, inputs, physical T/B, format and policy; records equal its newest history columns.
 * A private convolution intermediate need not be rounded to BF16 before use.
 *
 * The two-parent form registers Q4 q/k [4096,5120] and the Q5 value/z parent [12288,5120]. All
 * tensor operands, outputs, conv_record, source state, and live workspace must be disjoint.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                WorkspaceArena& workspace, cudaStream_t stream);

/// The two-parent form under a compute policy (A16ColumnInvariant composes at every B and T).
void gdn_input_proj_conv_record(const Tensor& x, const Weight& qk_weight,
                                const Weight& value_z_weight, const Tensor& conv_weight,
                                const Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots, Tensor& conv_record,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream);

/**
 * Single-parent record-producing form. Registered parents are Q8_G32_FP16 [12288,2048], NVFP4
 * [16384,5120], FP8_E4M3FN_ROW_BF16 [16384,5120] and FP8_E4M3FN_ROW_F32 [20480,3072]. All
 * policies permit Q8 A16; NVFP4 uses A16 under A16Only/AllowA8 and may use A4 under AllowA4. FP8
 * may use A8 under AllowA8/AllowA4. Record and snapshot share arithmetic route selection. Every
 * tensor operand, the complete FP8 parent, and live workspace must be mutually non-overlapping.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream);

/**
 * Verification-tree form of the single-parent record-producing Op (FP8, NVFP4 and Q8 parents)
 * over a width T in [2,32]. tree_rows is I32 [kSpeculativeTreeRowWords,B]: column c of row b convolves
 * the projected inputs of its three nearest ancestors in that row's tree (conv history past the
 * anchor) and its own, in the chain form's order, so each column sees the inputs of its own root
 * path and a chain row gets the chain form over its valid prefix. Records hold every column's
 * projected input. Projection uses the Op's materialized routes; workspace follows the record
 * capacity query.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                const Tensor& tree_rows, Tensor& conv_record, Tensor& query,
                                Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy,
                                WorkspaceArena& workspace, cudaStream_t stream);

/** Applies the A16-only single-parent record-producing form. */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops
