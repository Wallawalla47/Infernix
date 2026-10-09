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
static_assert(kXRowBytes >= kIntermediate * 2, "a W4A16 call keeps BF16 h rows in the x plane");
static_assert(kHRowStride % 16 == 0 && kHRowStride >= kHCodeBytes + kHScaleMapBytes && kHCodeBytes % 16 == 0);

// The 16-element A4 block b of a plane row is stored at block position plane_block(b): within each
// group of four blocks (one 64-wide MMA K step) the middle two are swapped, so that the record's
// 144-byte units read for one K step fall in distinct shared-memory banks. Its eight code bytes
// pair elements j and j + 4 of each half block (byte i: code[8(i/4) + i%4] | code[8(i/4) + 4 + i%4]
// << 4), the order in which the record's row-group nibbles enter an MMA register.
INFERNIX_WIDE_HD constexpr int plane_block(int b) { return (b & ~3) | ((b & 1) << 1) | ((b & 2) >> 1); }

// The layout of moe_experts' workspace, the single place that defines it.
struct ExpertsWorkspace {
    canon::A4Block* h_blocks;     // [entries][kHBlocks]; wide entries hold their A4(h) row here
    const std::uint8_t** job_records; // [max_jobs]
    std::int32_t* cpu_flags;      // [max_jobs]
    void* cpu_call;               // kCpuCallBytes
    std::uint8_t* x_plane;        // [entries][kXRowBytes]
    std::int32_t* tiles;          // [max_tiles][2]: (job, first column)
    std::int32_t* pass_tiles;     // [max_jobs + 2]
};

// Bytes of a call's CPU bookkeeping (moe_layer.cu's CpuCall: two words and kMaxCpuJobs job indices).
inline constexpr std::size_t kCpuCallBytes = 2304; // 2 words + 512 indices, rounded to 256

[[nodiscard]] std::int32_t max_tiles(std::int32_t max_jobs, std::int32_t entries);
[[nodiscard]] std::size_t experts_workspace_bytes(std::int32_t max_jobs, std::int32_t entries);
[[nodiscard]] ExpertsWorkspace carve_experts_workspace(void* base, std::int32_t max_jobs, std::int32_t entries);

// Device and host state of one moe_experts call's wide route; a plain value, valid until the call's
// workspace is released. `descriptors` points to the call's staged TMA descriptors (W4A4 only).
// W4A16 calls (wide_expert_a16.cuh) read x by token and keep h in BF16 in the workspace's x plane.
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
    const std::uint16_t* x                    = nullptr; // W4A16: BF16 [H, T]
    std::uint16_t* h16                        = nullptr; // W4A16: BF16 [entries][kIntermediate]
    std::int32_t top_k                        = 0;
};

// Plans the call's wide work tiles in passes of `pass_jobs` jobs (the passes moe_experts stages),
// quantizes x for every wide entry and stages the TMA descriptors. Enqueued on `stream`.
[[nodiscard]] Call prepare(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                           std::int32_t top_k, std::int32_t max_jobs, std::int32_t pass_jobs,
                           const ExpertsWorkspace& workspace, Tensor& outputs, cudaStream_t stream);

// Computes the wide experts among the jobs of `pass`, whose records job_records holds by then:
// the gate/up GEMM with SwiGLU and A4(h), then the down GEMM into `outputs`.
void run_pass(const Call& call, std::int32_t pass, cudaStream_t stream);

} // namespace wide
} // namespace infernix::ops::offloaded_moe

#undef INFERNIX_WIDE_HD
