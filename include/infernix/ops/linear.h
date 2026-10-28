#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit A16, A8 or A4 compute profiles.
    /// Admit only an A16 profile whose arithmetic for one column does not depend on how many
    /// columns the call holds or on their values: column t's output bits are a function of column
    /// t's input and the weight alone, for every T. A projection that does not register such a
    /// route rejects this policy.
    A16ColumnInvariant,
    /// Admit A16 or an A4 residual profile: NVFP4 activations as a sum of NVFP4 terms, each
    /// encoding what the earlier ones left, so the activation error is roughly A4's raised to the
    /// term count. Never one-term A4 or A8. A projection that does not register a residual route
    /// runs A16; one that does needs the A4 activation divisor (as AllowA4 does).
    AllowA4Residual,
    /// Admit A16 or an A8 residual profile: FP8 activations as two E4M3 terms per token, the
    /// second encoding what the first left at a sixteenth of its scale. Never one-term A8 or A4.
    /// A projection that does not register a residual route runs A16 (the FP8 input projections
    /// and gate/up register one; the output and down projections do not).
    AllowA8Residual,
};

/**
 * @brief The fast-path routes a projection may select (a set of flags).
 *
 * Exact (no flag) is each registered problem's reproducible-route arithmetic (deterministic, and
 * column-invariant under A16ColumnInvariant; a new version may change its bits within the Op's criterion). Each
 * flag admits one fast path's routes for the problems that register them, whose arithmetic differs;
 * a Model sets a flag only where its fast path is qualified (docs/maintainer/model-quality.md
 * section 6.7). A problem with no route for a set flag runs its exact route.
 */
enum class LinearRoutes : std::uint16_t {
    Exact     = 0,
    Q8SlicedK = 1, ///< q8-sliced-k: NInfer's Q8 GEMV/sliced-K routes (Flash-Next [16384,2560], [2560,6144]).
    Fp8A8Pair = 2, ///< fp8-a8-pair: FP8 weights with the activation as two E4M3 terms.
    Q4W4A8    = 4, ///< q4-w4a8: groupwise Q4 weights with int8 activations.
    /// batched-decode: NVFP4 linear_add [5120,17408] A16 tiles of 8 K-warps at 5-16 columns (the
    /// exact route keeps the release's 4 K-warp tiles; the K-warp count sets the reduction order).
    WideDownTile = 8,
    /// ggml-bf16-prefill: GGUF projections of 32 or more columns on BF16 tensor cores with FP32 sums
    /// (ops::ggml_linear) instead of the exact A16 integer tiles.
    GgmlBf16Wide = 16,
    /// fp8-a8-pair-block32: fp8-a8-pair's two E4M3 activation terms with a power-of-two scale per
    /// 32 values of K and term, applied by the block-scaled MMA; it takes precedence over Fp8A8Pair. On
    /// MXFP8 weights (the registered FP4-Local problems) the weight's group scales are the MMA's weight
    /// scales; F16Accumulate takes precedence there.
    Fp8A8PairBlock32 = 32,
    /// f16-accumulate-prefill: E4M3-weight projections (row-scaled FP8, MXFP8) of prefill widths on FP16
    /// Tensor Cores, the activation as FP16 under a power-of-two column scale and FP16 sums over 32-value
    /// K chunks with an FP32 carry (twice the FP32-accumulation rate).
    F16Accumulate = 64,
    /// q5-ksplit-mma: Q5 linear_add [5120,6144] and [5120,17408] with eight K warps and the pairwise
    /// reduction (NInfer's K-split residual kernel) at 2-16 columns, and at every width of the
    /// column-invariant policy; the exact route keeps four K warps (a different summation order).
    Q5KSplit = 128,
    /// a8-activations: GGUF projections (ops::ggml_linear) of 1-31 columns take 8-bit activations, each
    /// column quantized per 32 values of K to int8 codes with an FP32 scale (ops/ggml/ggml_a8.h: llama.cpp's
    /// Q8_1 idea), exact int32 slice dots with the weight codes and FP32 sums of the slices' terms, instead
    /// of the exact A16 integer routes. LOSSY and opt-in (`--a8-activations`), never on the reproducible
    /// route; wider calls keep their routes (GgmlBf16Wide or exact). The only dense 8-bit activation
    /// route of decode widths: the other formats' measured slower than their 16-bit decode routes and
    /// were removed (docs/maintainer/qwen3_8-flash-next-design.md §19.3.31).
    GgmlA8 = 256,
    /// a8-activations at prefill widths: GGUF projections (ops::ggml_linear) of 32 or more columns take the
    /// same 8-bit activations (the GgmlA8 codec's codes and slice scales) on int8 tensor cores over decoded
    /// weight planes: exact int32 slice dots, fl32(dot * d * sub) and an FP32 sum in K order per output, so a
    /// column's bits do not depend on the call's width or other columns. LOSSY and opt-in, never on the
    /// reproducible route; it takes precedence over GgmlBf16Wide where both are set, and is independent of
    /// GgmlA8 (which decides the calls of 1-31 columns).
    GgmlA8Wide = 512,
};

[[nodiscard]] constexpr LinearRoutes operator|(LinearRoutes a, LinearRoutes b) noexcept {
    return static_cast<LinearRoutes>(static_cast<std::uint16_t>(static_cast<std::uint16_t>(a) | static_cast<std::uint16_t>(b)));
}

[[nodiscard]] constexpr bool has_route(LinearRoutes routes, LinearRoutes flag) noexcept {
    return (static_cast<std::uint16_t>(routes) & static_cast<std::uint16_t>(flag)) != 0;
}

[[nodiscard]] constexpr bool valid_linear_routes(LinearRoutes routes) noexcept {
    return (static_cast<std::uint16_t>(routes) & ~std::uint16_t{1023}) == 0;
}

[[nodiscard]] constexpr bool valid_linear_policy(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
           policy == LinearPolicy::AllowA4 || policy == LinearPolicy::A16ColumnInvariant ||
           policy == LinearPolicy::AllowA4Residual || policy == LinearPolicy::AllowA8Residual;
}

[[nodiscard]] constexpr bool allows_a4_residual(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA4Residual;
}

[[nodiscard]] constexpr bool allows_a8_residual(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8Residual;
}

// Either residual profile: a projection without that profile's route runs A16.
[[nodiscard]] constexpr bool allows_residual(LinearPolicy policy) noexcept {
    return allows_a4_residual(policy) || allows_a8_residual(policy);
}

[[nodiscard]] constexpr bool column_invariant(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::A16ColumnInvariant;
}

[[nodiscard]] constexpr bool allows_a8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8 || policy == LinearPolicy::AllowA4;
}

[[nodiscard]] constexpr bool allows_a4(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA4;
}

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * Whether the problem registers a dedicated column-invariant route. Without one,
 * LinearPolicy::A16ColumnInvariant runs the exact A16 route once per column.
 */
[[nodiscard]] bool linear_column_invariant_route(QType qtype, std::int32_t output_rows,
                                                 std::int32_t input_rows) noexcept;

/**
 * Whether the format writes an FP32 `out` (a vocabulary head's logits): Q6_G64_FP16, Q8_G32_FP16, BF16,
 * FP8_E4M3FN_ROW_BF16, FP8_E4M3FN_ROW_F32 and nvfp4_rg16_v1. The other formats refuse one.
 */
[[nodiscard]] bool linear_fp32_output(QType qtype) noexcept;

/**
 * The transient bytes LinearRoutes::F16Accumulate needs for a call of `tokens` columns of K =
 * `input_rows` (the FP16-encoded activation and its column scales). linear_workspace_capacity_bytes()
 * and gdn_input_proj_workspace_capacity_bytes() include it wherever an interval reaches the route; a
 * caller that sizes one arena for many projections adds it for the largest K.
 */
[[nodiscard]] std::size_t linear_f16_accumulate_workspace_bytes(std::int32_t input_rows,
                                                                std::int32_t tokens);

/**
 * The transient bytes LinearRoutes::Fp8A8PairBlock32 needs for a call of `tokens` columns of K =
 * `input_rows`, a multiple of 128 (the activation's two E4M3 planes and their per-32-value scales), as
 * linear_f16_accumulate_workspace_bytes() is for its route.
 */
[[nodiscard]] std::size_t linear_fp8_a8_pair_block32_workspace_bytes(std::int32_t input_rows,
                                                                     std::int32_t tokens);

/**
 * @brief Applies a bias-free matrix projection independently to every input column.
 *
 * @details The ideal mathematical result is
 *
 * @f[
 *   \mathrm{ideal}_{n,t} =
 *   \sum_{k=0}^{K-1}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `out` stores a BF16 approximation of this ideal result under the named numerical criterion for
 * the selected private activation-compute path, or, for an FP32 `out`, the FP32 sum itself (the
 * same computation without the final rounding to BF16).
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 or FP32 `[N,T]`. FP32 output serves vocabulary
 * heads (BF16 logits tie near-equal tokens): Q6 and Q8 at every registered problem (Q8 problems other
 * than `[248320,5120]` run the runtime-shape schedules), BF16 on its runtime-shape route, FP8
 * `[248320,5120]` and `[248320,3072]` (FP32 row multipliers), and `nvfp4_rg16_v1`; other formats
 * refuse it. Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses RowSplit Q4_G64_FP16, Q5_G64_FP16, Q6_G64_FP16, or Q8_G32_FP16 weights
 * with FP16 scales, block-scaled NVFP4 weights, row-scaled FP8_E4M3FN_ROW_BF16 weights, group-scaled
 * FP8_E4M3FN_G32_E8M0 (MXFP8) weights in G32ScaleRows16, plus registered contiguous BF16 problems.
 * MXFP8 registers Qwen3.8-Flash-Next's `[16384,2560]`, `[13952,2560]`, `[2560,6144]`, `[1280,2560]`
 * and `[2560,640]` at every positive T, with 16-bit activations under every policy; every route
 * gives a column the same bits at every T, so A16ColumnInvariant needs nothing more. Each format
 * owns a finite registry of exact physical
 * weight problems and selects its kernel internally; a valid encoding and alignment do not imply
 * arbitrary N/K support. FP8 currently registers `[N,K]` in `{[14336,5120], [16384,5120],
 * [34816,5120], [248320,5120], [5120,6144], [5120,17408]}` at every positive T, and with FP32 row
 * multipliers (FP8_E4M3FN_ROW_F32, Qwen3.5-122B-A10B's per-row FP8 shared expert down and head)
 * `[3072,1024]` and `[248320,3072]` at every positive T with 16-bit activations under every policy,
 * each with a column-invariant route. The current NVFP4
 * problems register the five non-vocabulary FP8 geometries and accept every positive T. Q8 also
 * registers `[5120,25600]` at every positive T. BF16 registers the text problems `[14336,5120]`,
 * `[5120,6144]` and `[256,5120]` and the Vision tower problems `[1152,1536]`, `[3456,1152]`,
 * `[1152,1152]`, `[4304,1152]`, `[1152,4304]`, `[4608,4608]`, `[2560,4608]` and `[5120,4608]`,
 * each at every positive T. Text and MTP packed-weight problems accept every positive column
 * extent T. Registered packed-weight Vision problems accept raw-patch P in `{4,8,...,131072}` or
 * merged-token V in `[1,32768]`; a matrix column does not inherently represent a text token. A
 * dense NVFP4_MUL matrix in QuantLayout::Rg16 (`nvfp4_rg16_v1`: Qwen3.5-122B-A10B's head
 * `[248320,3072]`; Qwen3.8-Flash-Next FP4-Local's Vision MLP `[4304,1152]`, `[1152,4304]` and
 * merger `[4608,4608]`, `[2560,4608]`) runs rg16_linear at every positive T under every policy: the
 * canonical W4A16 product with exact sums, so it is column-invariant and needs the workspace
 * overload. FP32 is unsupported.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected
 * activation-compute path; none is copied into the oracle. Kernel, schedule, template instance,
 * host launcher, and T region do not create separate criteria inside one path.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. Every policy permits the existing A16 implementations of BF16 and Q4/Q5/Q6/Q8.
 * FP8 accepts all three policies; AllowA8 and AllowA4 permit the private resolver to select a
 * qualified A16 or A8 route for the registered problem and T. FP8 `[248320,5120]` retains A16
 * compute under every policy at every positive T. NVFP4 uses A16 for A16Only and
 * AllowA8; AllowA4 permits the private resolver to select either a qualified A16 route or
 * activation quantization to NVFP4 at every positive T. The selected route depends only on the
 * registered problem, T and `routes`. A16ColumnInvariant makes each column's result independent of
 * the call's width and other columns, on every registered problem: a problem with a dedicated
 * column-invariant route (linear_column_invariant_route: FP8's text problems, Q8 [248320, 5120],
 * Q6 [248320, 5120], [248320, 2048] and [34816, 5120]) runs one A16 route for every T; any other
 * runs its exact A16 route once per column (T = 1 each).
 *
 * @par Routes
 * Under LinearRoutes::Q8SlicedK, Q8 `[16384,2560]` and `[2560,6144]` (Qwen3.8-Flash-Next's GDN q/k/v/z
 * and GDN/QSA output projections) select from NInfer's GEMV, sliced-K and grouped sliced-K routes,
 * which accumulate in a different order from the exact routes; a column's bits then depend on the
 * call's T. Under LinearRoutes::Fp8A8Pair, FP8 [16384,2560], [13312,2560], [2560,6144], [1280,2560]
 * and [2560,640] at T >= 17, unless a single-term A8 route is selected, carry each activation column
 * as two E4M3 terms with their own scales on the 8-bit Tensor Core path. Under
 * LinearRoutes::Fp8A8PairBlock32 the same problems at the same widths carry the two terms with a
 * power-of-two scale per 32 values of K and term instead of per column (taking precedence where both
 * flags are set). Under LinearRoutes::F16Accumulate, the FP8 problems [16384,2560], [13312,2560],
 * [2560,6144], [1280,2560], [2560,640] and [5120,6144] and every MXFP8 problem, at T >= 256 and where
 * no single-term A8 route is selected, encode each activation column exactly as FP16 under a
 * power-of-two scale that puts its maximum in [1,2) (values more than 2^17 below the column maximum
 * keep fewer bits, those about 2^25 below flush to zero), sum each 32-value K chunk on FP16 Tensor
 * Cores with FP16 accumulation (at most 32 * 448 * 2 in magnitude, so it cannot overflow) and carry
 * the chunks in FP32; it takes precedence over both pair routes where several are set. A column's bits
 * then depend only on the column, but differ from the exact route's. Every other problem runs its
 * exact route under a flag it does not register. Under A16ColumnInvariant every flag is ignored: the
 * column-invariant route is the exact route's. Every route meets the A16 criterion; the workspace
 * query covers the two-term and FP16 routes' scratch.
 *
 * @par Workspace
 * `workspace` is caller-owned call-scoped transient storage sized by
 * linear_workspace_capacity_bytes(). It must not overlap x, any weight plane, or out. Linear does
 * not allocate device memory internally.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 or FP32 output matrix `[N,T]`. It must
 * not overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] workspace Caller-owned transient arena.
 * @param[in] stream CUDA stream on which execution is enqueued.
 * @param[in] routes The fast-path route flags (Exact: none).
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream,
            LinearRoutes routes = LinearRoutes::Exact);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload admits only A16 compute and requires no transient workspace. All tensor,
 * weight, aliasing, and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 or FP32 output matrix `[N,T]`.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

/**
 * The QSA group projection of Qwen3.8-Flash-Next, `[13952,2560]` (query 6,144, gate 6,144, key 512,
 * value 512, index query 512 and index key 128 rows), with each row segment written to its own
 * contiguous BF16 `[rows, T]` tensor: the bits linear() writes to those rows under the same policy and
 * routes. Registered for Q8_G32_FP16 (Dense8) and FP8_E4M3FN_G32_E8M0 (FP4-Local) weights;
 * linear_qsa_group_supported() says whether a weight and segment layout has it (else a caller splits
 * linear()'s output). Its workspace is linear_workspace_capacity_bytes() of `[13952,2560]`.
 */
inline constexpr std::int32_t kQsaGroupSegments = 6;
[[nodiscard]] bool linear_qsa_group_supported(const Weight& w, const Tensor* const* parts,
                                              LinearPolicy policy) noexcept;
void linear_qsa_group(const Tensor& x, const Weight& w, Tensor* const* parts, LinearPolicy policy,
                      WorkspaceArena& workspace, cudaStream_t stream, LinearRoutes routes = LinearRoutes::Exact);

} // namespace infernix::ops
