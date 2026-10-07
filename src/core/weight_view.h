#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix {

// Existing dense Op ABI uses Weight axes directly; callers establish BF16/FP32 and layout.
inline Tensor as_dense(const Weight& weight) {
    const auto dtype = weight.qtype == QType::FP32 ? DType::FP32 : DType::BF16;
    auto* data       = const_cast<void*>(weight.qdata);
    switch (weight.ndim) {
    case 1:
        return Tensor(data, dtype, {weight.shape[0]});
    case 2:
        return Tensor(data, dtype, {weight.shape[0], weight.shape[1]});
    case 3:
        return Tensor(data, dtype, {weight.shape[0], weight.shape[1], weight.shape[2]});
    default:
        return Tensor(data, dtype,
                      {weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]});
    }
}

// Geometry describes one complete encoded parent. Regions address its logical elements;
// padding and scale words do not belong to that element domain.
struct WeightGeometry {
    QType format       = QType::BF16;
    QuantLayout layout = QuantLayout::Contiguous;
    std::vector<std::uint64_t> shape;
    std::uint64_t elements            = 0;
    std::uint64_t bytes               = 0;
    std::uint64_t alignment           = 256;
    std::uint64_t padded_columns      = 0;
    std::uint64_t group_size          = 0;
    std::uint64_t code_bytes_per_row  = 0;
    std::uint64_t high_bytes_per_row  = 0;
    std::uint64_t scale_bytes_per_row = 0;
    std::uint64_t code_bytes          = 0;
    std::uint64_t high_offset         = 0;
    std::uint64_t high_bytes          = 0;
    std::uint64_t scale_offset        = 0;
    std::uint64_t scale_bytes         = 0;
    std::uint64_t divisor_offset      = 0;
    // ExpertRg16 only: one expert's unit bytes and its 4 KiB-aligned pitch.
    std::uint64_t record_bytes  = 0;
    std::uint64_t record_stride = 0;
};

[[nodiscard]] WeightGeometry weight_geometry(QType format, QuantLayout layout,
                                             std::span<const std::uint64_t> shape);
[[nodiscard]] std::uint64_t weight_element_count(std::span<const std::uint64_t> shape);

struct WeightParent {
    WeightGeometry geometry;
    const std::byte* data      = nullptr;
    float weight_scale_divisor = 0.0F;
};

struct WeightRegion {
    const WeightParent* parent = nullptr;
    std::uint64_t begin        = 0;
    std::uint64_t end          = 0;
};

struct WeightView {
    std::vector<std::uint64_t> shape;
    std::vector<WeightRegion> parts;
};

// Coalesce adjacent logical regions without moving their bytes. Fails for a gather or
// multiple parents; those remain explicit inputs for a consumer supporting that form.
[[nodiscard]] WeightRegion contiguous_weight_region(const WeightView& view);
[[nodiscard]] bool is_complete_weight(const WeightView& view);
[[nodiscard]] std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                                std::uint64_t group);

struct WeightRowPlanes {
    const std::byte* codes        = nullptr;
    const std::byte* high         = nullptr;
    const std::byte* scales       = nullptr;
    std::uint64_t row_begin       = 0;
    std::uint64_t row_count       = 0;
    std::uint64_t code_row_bytes  = 0;
    std::uint64_t high_row_bytes  = 0;
    std::uint64_t scale_row_bytes = 0;
    // NVFP4 keeps the parent scale base; use its row origin and block geometry.
    bool swizzled_scales = false;
};

[[nodiscard]] WeightRowPlanes weight_row_planes(const WeightRegion& region);
[[nodiscard]] Tensor weight_tensor(const WeightView& view,
                                   std::initializer_list<std::int32_t> internal_shape);
// Existing Weight ABI: complete quantized parents, direct regions, and RowSplit row views.
// Arbitrary FP8/NVFP4 regions use their explicit planes until a native consumer supports them.
[[nodiscard]] Weight native_weight(const WeightView& view, float input_divisor = 0.0F);

// A routed-expert bank in nvfp4_expert_rg16_v1 (Qwen3.8-Flash-Next design §6.2). Pointers are in
// the parent's memory space; nothing is dereferenced.
struct ExpertBankPlanes {
    const std::byte* records     = nullptr; // expert e at records + e * record_stride
    const float* multipliers     = nullptr; // [experts][3]: gate, up, down weight_scale_2
    std::uint64_t record_bytes   = 0;
    std::uint64_t record_stride  = 0;
    std::uint64_t gate_up_bytes  = 0; // the down matrix starts here inside a record
    std::uint32_t experts        = 0;
    std::uint32_t hidden         = 0;
    std::uint32_t intermediate   = 0;
};
[[nodiscard]] ExpertBankPlanes expert_bank_planes(const WeightParent& parent);
// The layout of a bank whose records are not resident (read in place by the SSD tier): records is
// null and the multipliers are the caller's copy of the bank's scale tail.
[[nodiscard]] ExpertBankPlanes expert_bank_layout(const WeightGeometry& geometry, const float* multipliers);

// Block-scaled FP8 matrices in block128_scale_v1, leading axes flattened into a batch.
struct Block128Planes {
    const std::byte* codes   = nullptr; // [batch][n][k] E4M3FN
    const float* scales      = nullptr; // [batch][scale_rows][scale_cols] FP32 multipliers
    std::uint64_t batch      = 0;
    std::uint64_t n          = 0;
    std::uint64_t k          = 0;
    std::uint64_t scale_rows = 0;
    std::uint64_t scale_cols = 0;
};
[[nodiscard]] Block128Planes block128_planes(const WeightParent& parent);

} // namespace infernix
