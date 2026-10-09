#pragma once

// CPU narrow route of offloaded_sparse_moe: one routed expert of Qwen3.8-Flash-Next computed in
// place from its host record, with the canonical W4A4 arithmetic of
// docs/maintainer/qwen3_8-flash-next-design.md §16.2. Every ISA variant returns the same bits,
// and so does the GPU narrow route.

#include "ops/common/canonical_math.h"

#include <cstddef>
#include <cstdint>

namespace infernix::ops::offloaded_moe {

// Record layout `nvfp4_expert_rg16_v1` (design §6.2).
inline constexpr int kHidden           = 2560;
inline constexpr int kIntermediate     = 640;
inline constexpr std::size_t kUnitBytes = 144;
inline constexpr int kGateUpRowGroups  = 2 * kIntermediate / 16; // 80, rows interleaved gate/up
inline constexpr int kGateUpBlocks     = kHidden / 16;           // 160
inline constexpr int kDownRowGroups    = kHidden / 16;           // 160
inline constexpr int kDownBlocks       = kIntermediate / 16;     // 40
inline constexpr int kHBlocks          = kIntermediate / 16;     // 40 units of 16 intermediates
inline constexpr std::size_t kGateUpBytes =
    static_cast<std::size_t>(kGateUpRowGroups) * kGateUpBlocks * kUnitBytes; // 1,843,200
inline constexpr std::size_t kDownBytes =
    static_cast<std::size_t>(kDownRowGroups) * kDownBlocks * kUnitBytes;     // 921,600
inline constexpr std::size_t kRecordBytes = kGateUpBytes + kDownBytes;      // 2,764,800
inline constexpr int kMaxColumns          = 8;                              // narrow route: n <= 8

// Software prefetch distance in bytes along a matrix's unit stream (0 disables it). The best value
// depends on the host's memory latency and core speed; infernix-calibrate measures it on the target
// (design §14.2). This default is only a starting point, not a tuned value.
inline constexpr int kDefaultPrefetchBytes = 2048;

static_assert(kRecordBytes == 675 * 4096);

// Per-expert scalars from the checkpoint (design §6.1). alpha = fl32(weight_scale_2 * input_scale)
// is derived once by the loader, so every route reads the same words.
struct ExpertScales {
    float input_gate;
    float input_up;
    float input_down;
    float alpha_gate;
    float alpha_up;
    float alpha_down;
};

// The activation arithmetic of a layer's experts, fixed by the artifact: W4A4 with each matrix's
// stored input scale (§16.2), or W4A16 for experts stored without activation scales (§16.2.1,
// w4a16_expert.h), whose ExpertScales hold zero input scales and the multipliers in alpha_*.
enum class ExpertActivation : std::uint8_t { kA4, kA16 };

enum class CpuIsa { kScalar, kAvx2, kAvxVnni, kAvx512Vnni };

const char* cpu_isa_name(CpuIsa isa);
bool cpu_isa_supported(CpuIsa isa);
CpuIsa best_cpu_isa();

// Exact int64 row sums S[row][col] for row groups [rg_begin, rg_end) of a matrix stored as
// row-group-major 144-byte units with `blocks` 16-element blocks per row. `acts[col]` points to
// `blocks` quantized activation blocks of column col. `out` has (rg_end - rg_begin) * 16 rows of
// `ncols` values.
void rg16_row_sums(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                   const canon::A4Block* const* acts, int ncols, std::int64_t* out,
                   int prefetch_bytes = kDefaultPrefetchBytes);

// Quantizes a BF16 vector of `n` (a multiple of 16) elements to A4 blocks.
void quantize_a4(const std::uint16_t* v, int n, float input_scale, canon::A4Block* out);

// Phase A for 16-intermediate units [unit_begin, unit_end): gate/up rows, SwiGLU, and A4 of h
// with the down input scale. `x_gate[col]` / `x_up[col]` are x quantized with the gate and up
// input scales (the same pointers when the scales are equal). Writes h_blocks[col][unit].
void gate_up_units(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                   const canon::A4Block* const* x_gate, const canon::A4Block* const* x_up,
                   int ncols, int unit_begin, int unit_end, canon::A4Block* const* h_blocks,
                   int prefetch_bytes = kDefaultPrefetchBytes);

// Phase B for down row groups [rg_begin, rg_end): writes y[col][row] in BF16 for those rows.
void down_rows(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
               const canon::A4Block* const* h_blocks, int ncols, int rg_begin, int rg_end,
               std::uint16_t* const* y, int prefetch_bytes = kDefaultPrefetchBytes);

// Whole expert on one thread: y[col][0..2560) from x[col][0..2560), both BF16.
void expert_forward(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales, int ncols,
                    const std::uint16_t* const* x, std::uint16_t* const* y,
                    int prefetch_bytes = kDefaultPrefetchBytes);

} // namespace infernix::ops::offloaded_moe
