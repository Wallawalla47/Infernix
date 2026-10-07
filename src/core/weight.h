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
    // E4M3FN codes with one FP32 multiplier per 128 x 128 tile.
    FP8_E4M3FN_BLOCK128_F32 = 10,
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // nvfp4_expert_rg16_v1: [experts, hidden, intermediate] banks of 4 KiB-aligned expert
    // records in 144-byte row-group units, then an [experts, 3] FP32 multiplier plane.
    ExpertRg16 = 4,
    // block128_scale_v1: [..., N, K] row-major codes, then FP32 tile multipliers.
    Block128Scale = 5,
};

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
