#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * LoRA adapters (runtime, exact): a projection W x becomes W x + s * B (A x). Two pieces implement it.
 *
 * 1. lora_shrink computes U = A x for every adapter that reads the same input, stacked: A is BF16 [R, K] (R the sum
 *    of the stacked adapters' padded ranks), x BF16 [K, T], U BF16 [R, T].
 * 2. The adapted projection Op takes a LoraAddend and adds s * B U to each output element in FP32 before its BF16
 *    rounding, so each output keeps the Op's one rounding boundary: to gate and up before the SwiGLU; for a single
 *    value, to the value the kernel hands its epilogue where that value is exactly rounded (an accumulator, an
 *    explicitly rounded product), or to the epilogue's FP32 result (after the residual's add) where the kernel hands it
 *    a product the add may fuse into one FMA. The kernel's own arithmetic is then the no-adapter kernel's: B = 0 gives
 *    its bits. The kernels of one column-invariant route take the term in the same place. Two sites run other routes
 *    when adapted, so B = 0 gives those routes' bits rather than the unadapted model's: a Gated DeltaNet projection's
 *    decode and verification (the materialized projection, then the convolution) and an adapted Gated DeltaNet control
 *    (the input norm alone, then gdn_gating_proj's LoRA form over the BF16 h).
 *
 * Canonical arithmetic (every kernel and route computes exactly this, so the adapted model keeps the request
 * invariance of the routes that promise it):
 *
 * - Shrink: K is cut into chunks of kLoraShrinkChunk values and groups of kLoraShrinkGroup chunks. A chunk's partial
 *   is a chain of BF16 m16n8k16 MMA k-steps over its values in ascending k, from +0, accumulating in FP32. A group's
 *   sum adds its chunk partials in ascending order (FP32, round to nearest, starting from the first partial); U's FP32
 *   value adds the group sums likewise; U = bf16_rn of it.
 * - Addend: for output row n of segment s and column t, c = a chain of BF16 m16n8k16 MMA k-steps over j in ascending
 *   order (rank / 16 steps, from +0) of B[n, j] * U[u_s + j, t]; the value v it is added to (2. above) becomes
 *   fl32(v + fl32(c * scale_s)), each operation rounded separately (never contracted into an FMA).
 *
 * The MMA element arithmetic is assumed independent of the element's position in its fragment and of the operand
 * roles (rows as M or as N); tests/ops/lora_mma_position_test checks it on the device.
 */
inline constexpr std::int32_t kLoraShrinkChunk = 256;
inline constexpr std::int32_t kLoraShrinkGroup = 4; // chunks, so K must be a multiple of 1024
inline constexpr std::int32_t kLoraMaxRank     = 128; // padded rank of one adapter (a multiple of 16)

// A's rows for one input site: BF16 [rows, k] row-major (k fastest).
struct LoraShrinkWeight {
    const void* a      = nullptr;
    std::int32_t rows  = 0; // a multiple of 16
    std::int32_t k     = 0; // a multiple of kLoraShrinkChunk * kLoraShrinkGroup
};

/**
 * U = A x under the canonical arithmetic above. x is BF16 [k, T] (any T >= 1), u BF16 [rows, T]; neither may overlap
 * A. The oracle is the represented BF16 A and x in FP64, compared under an FP32-accumulation criterion; equality of the
 * canonical bits across T and column positions is checked separately. No workspace; the Op has no persistent state.
 */
void lora_shrink(const Tensor& x, const LoraShrinkWeight& a, Tensor& u, cudaStream_t stream);

// One row range of an adapted parent: rows [previous segment's end, row_end) take U rows [u_offset, u_offset + rank)
// and `scale`; u_offset < 0 marks rows the adapter does not target (no term is added to them).
struct LoraSegment {
    std::int32_t row_end  = 0;
    std::int32_t u_offset = -1;
    float scale           = 0.0F;
};

/**
 * The low-rank term of one adapted parent weight. B is BF16 [parent rows, rank] row-major (rank fastest), in the
 * parent's physical row order; U is the lora_shrink output of the parent's input (BF16 [u rows, T], the Op's T).
 * Segments cover the parent's rows in order, at most four; every segment boundary is a multiple of 16.
 */
struct LoraAddend {
    const void* b      = nullptr;
    std::int32_t rank  = 0; // padded to a multiple of 16, at most kLoraMaxRank
    const void* u      = nullptr;
    std::int32_t u_rows = 0; // U's rows (its column stride)
    std::int32_t segments = 0;
    std::array<LoraSegment, 4> segment{};

    [[nodiscard]] bool present() const noexcept { return b != nullptr; }
};

// An adapted input of an Op that computes the input itself (the fused-norm projections): its stacked A for the shrink
// and the parent's term, whose `u` the Op sets to the U it computes (u_rows = a.rows).
struct LoraSite {
    LoraShrinkWeight a;
    LoraAddend addend;
};

// Throws std::invalid_argument when `addend` does not describe a term for a parent of `parent_rows` rows over a U of T
// columns (rank, segments, pointers, U shape).
void validate_lora_addend(const LoraAddend& addend, std::int32_t parent_rows, std::int32_t tokens);

// The term for a call over U's columns from `first_column` (a launcher that splits its columns into separate calls).
[[nodiscard]] inline LoraAddend lora_columns_from(const LoraAddend& addend, std::int32_t first_column) {
    LoraAddend out = addend;
    out.u = static_cast<const std::uint16_t*>(addend.u) + static_cast<std::int64_t>(first_column) * addend.u_rows;
    return out;
}

// The term of a parent made of `addend`'s rows [first_row, first_row + rows) (each part of a two-parent projection):
// B starts at first_row and the segments are clipped to the range and shifted.
[[nodiscard]] LoraAddend lora_row_range(const LoraAddend& addend, std::int32_t first_row, std::int32_t rows);

} // namespace infernix::ops
