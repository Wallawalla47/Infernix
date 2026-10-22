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

// The expert geometries this Op computes in the `nvfp4_expert_rg16_v1` layout: Qwen3.8-Flash-Next's
// H 2560 / I 640 (every bank and arithmetic; the constants above), and Qwen3.5-122B-A10B's
// H 3072 / I 1024 (W4A16 banks only: the exact narrow, wide and CPU routes and the reference route).
// The record is the gate/up matrix (2I rows, interleaved gate/up, of H) then the down matrix
// (H rows of I), each as row-group-major 144-byte units.
template <int Hidden, int Intermediate>
struct ExpertGeometry {
    static_assert(Hidden % 64 == 0 && Intermediate % 64 == 0);
    static constexpr int kHidden          = Hidden;
    static constexpr int kIntermediate    = Intermediate;
    static constexpr int kGateUpRowGroups = 2 * Intermediate / 16;
    static constexpr int kGateUpBlocks    = Hidden / 16;
    static constexpr int kDownRowGroups   = Hidden / 16;
    static constexpr int kDownBlocks      = Intermediate / 16;
    static constexpr int kHBlocks         = Intermediate / 16;
    static constexpr std::size_t kGateUpBytes =
        static_cast<std::size_t>(kGateUpRowGroups) * kGateUpBlocks * kUnitBytes;
    static constexpr std::size_t kDownBytes =
        static_cast<std::size_t>(kDownRowGroups) * kDownBlocks * kUnitBytes;
    static constexpr std::size_t kRecordBytes = kGateUpBytes + kDownBytes;
};
using FlashNextExperts   = ExpertGeometry<kHidden, kIntermediate>;
using Qwen3_5_122BExperts = ExpertGeometry<3072, 1024>;
static_assert(FlashNextExperts::kRecordBytes == kRecordBytes);
static_assert(Qwen3_5_122BExperts::kRecordBytes == 5308416);

// A bank's expert geometry at run time (MoeExpertSource::shape, the CPU engine's options).
struct ExpertShape {
    int hidden       = kHidden;
    int intermediate = kIntermediate;

    [[nodiscard]] constexpr int gate_up_row_groups() const { return 2 * intermediate / 16; }
    [[nodiscard]] constexpr int gate_up_blocks() const { return hidden / 16; }
    [[nodiscard]] constexpr int down_row_groups() const { return hidden / 16; }
    [[nodiscard]] constexpr int down_blocks() const { return intermediate / 16; }
    [[nodiscard]] constexpr int h_blocks() const { return intermediate / 16; }
    [[nodiscard]] constexpr std::size_t gate_up_bytes() const {
        return static_cast<std::size_t>(gate_up_row_groups()) * gate_up_blocks() * kUnitBytes;
    }
    [[nodiscard]] constexpr std::size_t down_bytes() const {
        return static_cast<std::size_t>(down_row_groups()) * down_blocks() * kUnitBytes;
    }
    [[nodiscard]] constexpr std::size_t record_bytes() const { return gate_up_bytes() + down_bytes(); }
    friend constexpr bool operator==(const ExpertShape&, const ExpertShape&) = default;
};

template <class Geometry>
inline constexpr ExpertShape kShapeOf{Geometry::kHidden, Geometry::kIntermediate};
inline constexpr ExpertShape kFlashNextShape    = kShapeOf<FlashNextExperts>;
inline constexpr ExpertShape kQwen3_5_122BShape = kShapeOf<Qwen3_5_122BExperts>;
// The largest registered hidden size (host buffers sized for any bank).
inline constexpr int kMaxHidden = Qwen3_5_122BExperts::kHidden;

[[nodiscard]] constexpr bool registered_shape(const ExpertShape& shape) {
    return shape == kFlashNextShape || shape == kQwen3_5_122BShape;
}

// Calls f.template operator()<Geometry>() with the geometry of `shape`, which the caller has checked
// is registered (registered_shape).
template <class F>
decltype(auto) with_expert_geometry(const ExpertShape& shape, F&& f) {
    if (shape == kQwen3_5_122BShape) { return f.template operator()<Qwen3_5_122BExperts>(); }
    return f.template operator()<FlashNextExperts>();
}

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

// The wide route of 16-bit-activation experts (design §19.3.23, §19.3.29): kExact, the exact limb
// arithmetic of §16.2.1 / §16.2.2; kA4Residual (NVFP4 W4A16 banks), a fast path that rounds each x and h
// row to two NVFP4 terms (the row's NVFP4 and the NVFP4 of its remainder, one FP32 row scale) and runs
// them on the FP4 Tensor Cores, so its outputs depend on that rounding and on the tensor core's FP32
// order, not only on the weights; kA4Residual3, the same with three terms (the third encodes what two
// left: more precise, slower); kGgmlBf16 (GGUF banks, the ggml-bf16-prefill fast path), BF16 tiles over
// each expert's decoded planes with x and h as stored and FP32 sums (ops/ggml/ggml_bf16_tile.cuh).
// Every other bank ignores the value it does not name. kLossless (every bank, A4 banks too, GPU only)
// is the --lossless route (docs/maintainer/lossless-route.md): x as stored times each weight's exact
// value (an NVFP4 bank's multipliers in ExpertScales::alpha_*), every projection's exact sum correctly
// rounded to BF16, h = bf16(bf16(silu(y_gate)) y_up) (ops::lossless_linear_tiles).
enum class WideA16 : std::uint8_t { kExact, kA4Residual, kA4Residual3, kGgmlBf16, kLossless };

// The design reference's routed-expert activations (kLossless only; docs/maintainer/lossless-route.md §3b):
// none (the lossless route's x as stored), or a W4A4 bank's declared A4 inputs, each column quantized with
// its expert's input scales (ExpertScales::input_*, beside the weights' multipliers in alpha_*): one x for
// the gate and up rows when every expert's two scales are equal, else one per scale (kA4SeparateUp).
enum class LosslessExpertDesign : std::uint8_t { kNone, kA4, kA4SeparateUp };

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
