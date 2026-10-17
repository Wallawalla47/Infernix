#pragma once

#include "core/dtype.h"

#include <cstdint>

namespace infernix {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    // ModelOpt NVFP4: E2M1 codes, E4M3FN scale per 16, FP32 *multiplier* per matrix.
    NVFP4_MUL = 9,
    // GGUF (ggml) block formats stored verbatim (docs/maintainer/tensor-formats.md §ggml);
    // core/ggml_format.h gives their block geometry.
    GGML_Q2_0    = 10,
    GGML_Q4_K    = 11,
    GGML_Q5_K    = 12,
    GGML_Q6_K    = 13,
    GGML_IQ2_S   = 14,
    GGML_IQ3_XXS = 15,
    GGML_IQ3_S   = 16,
    GGML_IQ4_XS  = 17,
    GGML_IQ4_NL  = 18,
    // Block-scaled FP8 (Qwen's official FP8 experts): E4M3FN codes, one BF16 scale per 128 x 128
    // tile (docs/maintainer/tensor-formats.md; tools/artifact/codecs/fp8_b128.py).
    FP8_E4M3FN_B128 = 19,
    // Two-bit grouped codes {-2, -1, 0, 1}, one FP16 scale per 64 (row-split, four codes per byte):
    // draft-only weights (the Qwen4Exp MTP drafter's experts; tensor-formats.md).
    Q2_G64_FP16 = 20,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // nvfp4_expert_rg16_v1: [experts, hidden, intermediate] banks of 4 KiB-aligned expert
    // records in 144-byte row-group units, then an [experts, 3] FP32 multiplier plane.
    ExpertRg16 = 4,
    // ggml_rows_v1: [n, k] as n rows of whole ggml blocks, verbatim and contiguous.
    GgmlRows = 5,
    // ggml_expert_{iq4_nl,q2_0}_v1: [experts, hidden, intermediate] banks of 4 KiB-aligned records
    // (gate rows, up rows in the bank's format, then the down rows in the layout's format).
    GgmlExpertIq4Nl = 6,
    GgmlExpertQ2_0  = 7,
    // fp8_expert_b128_v1: [experts, hidden, intermediate] banks of 4 KiB-aligned records (gate rows,
    // up rows, down rows, one E4M3FN byte per weight, then the three matrices' BF16 tile scales).
    Fp8ExpertB128 = 8,
    // ggml_tiles16_v1: [n, k] of whole 256-element ggml blocks in tiles of 16 rows (tile i at
    // 16 i row_bytes; block b of its row r at (16 b + r) * block_bytes; the last tile padded with
    // zero blocks): ggml_rows_v1's bytes, each tile's block b one contiguous run.
    GgmlTiles16 = 9,
};

// Whether `layout` stores a matrix of GGUF blocks (ggml_rows_v1 or ggml_tiles16_v1).
[[nodiscard]] constexpr bool is_ggml_matrix_layout(QuantLayout layout) noexcept {
    return layout == QuantLayout::GgmlRows || layout == QuantLayout::GgmlTiles16;
}

// Whether `layout` stores routed-expert banks (one 4 KiB-aligned record per expert).
[[nodiscard]] constexpr bool is_expert_bank_layout(QuantLayout layout) noexcept {
    return layout == QuantLayout::ExpertRg16 || layout == QuantLayout::GgmlExpertIq4Nl ||
           layout == QuantLayout::GgmlExpertQ2_0 || layout == QuantLayout::Fp8ExpertB128;
}

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
};

} // namespace infernix
