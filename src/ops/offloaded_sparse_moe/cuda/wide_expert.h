#pragma once

// The wide route of offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md §8.5, §13,
// §16.2, §19.3.7): every routed expert with more than kMaxColumns columns in a call (prefill chunks,
// wide verification batches) is computed by a block-scaled tensor-core grouped GEMM
// (mma.kind::mxf4nvf4.block_scale) instead of the exact integer narrow route.
//
// The arithmetic up to the accumulator is that of §16.2: x quantized to A4 with the expert's own
// gate/up input_scale by the canonical rule, the stored E2M1 codes with their per-16 E4M3 scales,
// h = bf16(silu_c(y_gate) * y_up) on BF16 y, A4 of h with the down input_scale, and
// y = bf16_rn(acc * alpha). Only the sum differs: the tensor core accumulates the exact block
// products in its own FP32 order instead of one exact int64 sum, so the route is qualified against
// the FP64 oracle rather than bitwise. The route is chosen from the column count and the expert's
// stored scales alone, never from placement, so outputs stay placement-invariant.
//
// Experts whose gate and up input scales differ keep the narrow route at every width: one MMA
// tile holds gate and up rows, which then need two different A4 activations. NVIDIA's export never
// produces such experts (24,576 of 24,576 share the scale in Qwen3.8-Flash-Next-NVFP4), and the
// narrow route computes them exactly.
//
// W4A16 experts (§16.2.1) take an exact wide route instead (wide_expert_a16.cuh): the canonical X
// of x and h, split into byte limbs, times the doubled E2M1 codes on integer tensor cores, two MMAs
// per 16-element block, folded into the exact int64 sum with the block scale. It gives the
// narrow route's and the CPU engine's bits, so a W4A16 expert's output never depends on how many
// other columns share its call.

#include "infernix/ops/offloaded_sparse_moe.h"

#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::offloaded_moe {

#if defined(__CUDACC__)
#    define INFERNIX_WIDE_HD __host__ __device__ __forceinline__
#else
#    define INFERNIX_WIDE_HD inline
#endif

// The route of one expert in one call: a function of its column count and its stored scales only.
INFERNIX_WIDE_HD bool wide_route(int columns, const ExpertScales& scales) {
    return columns > kMaxColumns && scales.input_gate == scales.input_up;
}

namespace wide {

inline constexpr int kTileColumns = 64; // columns (routed entries of one expert) per work tile
inline constexpr int kXCodeBytes  = kHidden / 2;  // 1,280 code bytes of one entry's A4(x)
inline constexpr int kXScaleBytes = kHidden / 16; //   160 scale bytes
inline constexpr int kXRowBytes   = kXCodeBytes + kXScaleBytes;
inline constexpr int kHCodeBytes  = kIntermediate / 2;  // 320 code bytes of one entry's A4(h)
inline constexpr int kHScaleBytes = kIntermediate / 16; //  40 scale bytes
// The h scale map spans whole 16-byte boxes; bytes 40-47 of a row are never read by an MMA step.
inline constexpr int kHScaleMapBytes = (kHScaleBytes + 15) / 16 * 16;
// The A4(h) row of a wide entry reuses that entry's narrow-route h block row (unused by the
// narrow kernels for wide experts), so the wide route needs no h workspace of its own.
inline constexpr int kHRowStride = kHBlocks * static_cast<int>(sizeof(canon::A4Block));

static_assert(kXRowBytes % 16 == 0 && kXCodeBytes % 16 == 0, "TMA needs 16-byte rows and bases");
static_assert(kHRowStride % 16 == 0 && kHRowStride >= kHCodeBytes + kHScaleMapBytes && kHCodeBytes % 16 == 0);

// W4A16 limb planes. A row (one token's x, one entry's h) holds the canonical X of elements
// 64t .. 64t+63 per 64-element K tile t as 64 H1 bytes, 64 L1 bytes and 64 H0 | L0 << 4 bytes, with
// X = 2^14 H1 + 2^11 H0 + 8 L1 + L0 (wide_expert_a16.cuh).
inline constexpr int kA16KTile        = 64;
inline constexpr int kA16LimbTileBytes = 3 * kA16KTile;                              // 192
inline constexpr int kA16XRowBytes    = kHidden / kA16KTile * kA16LimbTileBytes;       // 7,680
inline constexpr int kA16HRowBytes    = kIntermediate / kA16KTile * kA16LimbTileBytes; // 1,920
// An entry's h row first holds its BF16 h (narrow and wide gate/up write it); the wide route then
// rewrites a wide entry's row in place as limbs.
inline constexpr int kA16HRowElements = kA16HRowBytes / 2;
static_assert(kHidden % kA16KTile == 0 && kIntermediate % kA16KTile == 0);
static_assert(kA16HRowBytes >= kIntermediate * 2 && kA16HRowBytes % 16 == 0 && kA16XRowBytes % 16 == 0);

// The 16-element A4 block b of a plane row is stored at block position plane_block(b): within each
// group of four blocks (one 64-wide MMA K step) the middle two are swapped, so that the record's
// 144-byte units read for one K step fall in distinct shared-memory banks. Its eight code bytes
// pair elements j and j + 4 of each half block (byte i: code[8(i/4) + i%4] | code[8(i/4) + 4 + i%4]
// << 4), the order in which the record's row-group nibbles enter an MMA register.
INFERNIX_WIDE_HD constexpr int plane_block(int b) { return (b & ~3) | ((b & 1) << 1) | ((b & 2) >> 1); }

// The layout of moe_experts' workspace, the single place that defines it. The bookkeeping comes
// first, so its offsets depend only on max_jobs and entries; the activation planes follow.
struct ExpertsWorkspace {
    const std::uint8_t** job_records; // [max_jobs]
    std::int32_t* cpu_flags;      // [max_jobs]
    void* cpu_call;               // kCpuCallBytes
    std::int32_t* tiles;          // [max_tiles][2]: (job, first column)
    std::int32_t* pass_tiles;     // [max_jobs + 2]
    // W4A4
    canon::A4Block* h_blocks;     // [entries][kHBlocks]; wide entries hold their A4(h) row here
    std::uint8_t* x_plane;        // [entries][kXRowBytes]
    // W4A16
    std::uint8_t* x_limbs;        // [columns][kA16XRowBytes]: every token's x limbs (wide calls)
    std::uint8_t* h_rows;         // [entries][kA16HRowBytes]: BF16 h, rewritten as limbs for wide entries
    std::int32_t* x_emax;         // [columns]: column exponent of each token's x
    std::int32_t* h_emax;         // [entries]: column exponent of each wide entry's h
};

// Bytes of a call's CPU bookkeeping (moe_layer.cu's CpuCall: two words and kMaxCpuJobs job indices).
inline constexpr std::size_t kCpuCallBytes = 2304; // 2 words + 512 indices, rounded to 256

[[nodiscard]] std::int32_t max_tiles(std::int32_t max_jobs, std::int32_t entries);
// `columns` is the call's T (entries = top_k * T); the activation selects the planes.
[[nodiscard]] std::size_t experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns,
                                                  ExpertActivation activation);
[[nodiscard]] ExpertsWorkspace carve_experts_workspace(void* base, std::int32_t max_jobs, std::int32_t entries,
                                                       std::int32_t columns, ExpertActivation activation);

// Device and host state of one moe_experts call's wide route; a plain value, valid until the call's
// workspace is released. `descriptors` points to the call's staged TMA descriptors (W4A4 only).
struct Call {
    MoeDispatch dispatch;
    const ExpertScales* scales                = nullptr;
    const std::uint8_t* const* job_records    = nullptr;
    const std::int32_t* tiles                 = nullptr;
    const std::int32_t* pass_tiles            = nullptr;
    std::uint8_t* h_plane                     = nullptr;
    std::uint16_t* outputs                    = nullptr; // BF16 [H, entries]
    const void* descriptors                   = nullptr;
    std::int32_t passes                       = 0;
    bool a16                                  = false;
    // W4A16 (wide_expert_a16.cuh)
    const std::uint8_t* x_limbs = nullptr; // [T][kA16XRowBytes]
    std::uint8_t* h_rows        = nullptr; // [entries][kA16HRowBytes]
    const std::int32_t* x_emax  = nullptr; // [T]
    std::int32_t* h_emax        = nullptr; // [entries]
    std::int32_t top_k          = 0;
};

// Plans the call's wide work tiles in passes of `pass_jobs` jobs (the passes moe_experts stages),
// quantizes x for every wide entry and stages the TMA descriptors (W4A4), or encodes every token's
// x limbs (W4A16). Enqueued on `stream`.
[[nodiscard]] Call prepare(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                           std::int32_t top_k, std::int32_t max_jobs, std::int32_t pass_jobs,
                           const ExpertsWorkspace& workspace, Tensor& outputs, cudaStream_t stream);

// Computes the wide experts among the jobs of `pass`, whose records job_records holds by then:
// the gate/up GEMM with SwiGLU and the encoding of h (A4, or the W4A16 limbs), then the down GEMM
// into `outputs`.
void run_pass(const Call& call, std::int32_t pass, cudaStream_t stream);

} // namespace wide
} // namespace infernix::ops::offloaded_moe

#undef INFERNIX_WIDE_HD
