#include "core/weight_view.h"

#include "core/ggml_format.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernix {
namespace {

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error("weight geometry addition overflows u64");
    }
    return a + b;
}

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error("weight geometry multiplication overflows u64");
    }
    return a * b;
}

std::uint64_t aligned(std::uint64_t n, std::uint64_t alignment) {
    return add(n, alignment - 1) / alignment * alignment;
}

std::int32_t dimension(std::uint64_t n) {
    if (!n || n > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("weight dimension is outside the native i32 domain");
    }
    return static_cast<std::int32_t>(n);
}

std::uint64_t word_bytes(QType format) {
    switch (format) {
    case QType::BF16:
        return 2;
    case QType::FP32:
    case QType::INT32:
        return 4;
    default:
        throw std::invalid_argument("direct weight requires BF16, FP32 or INT32");
    }
}

DType direct_dtype(QType format) {
    switch (format) {
    case QType::BF16:
        return DType::BF16;
    case QType::FP32:
        return DType::FP32;
    case QType::INT32:
        return DType::I32;
    default:
        throw std::invalid_argument("quantized weight cannot be bound as a direct tensor");
    }
}

void validate_region(const WeightRegion& region) {
    if (!region.parent || region.begin >= region.end ||
        region.end > region.parent->geometry.elements) {
        throw std::invalid_argument("weight region exceeds its parent element domain");
    }
}

} // namespace

std::uint64_t weight_element_count(std::span<const std::uint64_t> shape) {
    if (shape.size() > 16) { throw std::invalid_argument("weight rank exceeds 16"); }
    std::uint64_t count = 1;
    for (const auto dim : shape) {
        if (!dim) { throw std::invalid_argument("weight dimensions must be positive"); }
        count = mul(count, dim);
    }
    return count;
}

WeightGeometry weight_geometry(QType format, QuantLayout layout,
                               std::span<const std::uint64_t> shape) {
    WeightGeometry out;
    out.format = format;
    out.layout = layout;
    out.shape.assign(shape.begin(), shape.end());
    out.elements = weight_element_count(shape);
    if (layout == QuantLayout::Contiguous) {
        out.bytes = out.code_bytes = mul(out.elements, word_bytes(format));
        if (shape.size() == 2) {
            out.padded_columns     = shape[1];
            out.code_bytes_per_row = mul(shape[1], word_bytes(format));
        }
        return out;
    }
    if (layout == QuantLayout::ExpertRg16) {
        if (format != QType::NVFP4_MUL || shape.size() != 3 || shape[1] % 16 || shape[2] % 16) {
            throw std::invalid_argument(
                "nvfp4_expert_rg16_v1 requires nvfp4_mul [experts, hidden, intermediate] with "
                "hidden and intermediate divisible by 16");
        }
        constexpr std::uint64_t kUnit = 144;
        const auto experts = shape[0], hidden = shape[1], inter = shape[2];
        const auto gate_up = mul(mul(2 * inter / 16, hidden / 16), kUnit);
        out.group_size     = 16;
        out.alignment      = 4096;
        out.record_bytes   = add(gate_up, mul(mul(hidden / 16, inter / 16), kUnit));
        out.record_stride  = aligned(out.record_bytes, 4096);
        out.code_bytes     = mul(experts, out.record_stride);
        out.scale_offset   = out.code_bytes;
        out.scale_bytes    = mul(experts, 12);
        out.bytes          = add(out.scale_offset, out.scale_bytes);
        return out;
    }
    if (is_ggml_expert_layout(layout)) {
        const QType down = ggml_expert_down_format(layout);
        if (!is_ggml(format) || shape.size() != 3 || format == QType::GGML_IQ4_NL || format == QType::GGML_Q2_0 ||
            shape[1] % ggml_block_elems(format) || shape[2] % ggml_block_elems(down)) {
            throw std::invalid_argument(
                "a ggml expert bank requires an IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS [experts, hidden, intermediate] with whole "
                "blocks on both matrices");
        }
        const auto experts = shape[0], hidden = shape[1], inter = shape[2];
        const auto gate_row = mul(hidden / ggml_block_elems(format), ggml_block_bytes(format));
        const auto down_row = mul(inter / ggml_block_elems(down), ggml_block_bytes(down));
        out.group_size     = ggml_block_elems(format);
        out.alignment      = 4096;
        out.code_bytes_per_row = gate_row;
        out.high_bytes_per_row = down_row; // the down rows' bytes (expert banks have no high plane)
        out.record_bytes   = add(mul(2, mul(inter, gate_row)), mul(hidden, down_row));
        out.record_stride  = aligned(out.record_bytes, 4096);
        out.code_bytes     = mul(experts, out.record_stride);
        out.scale_offset   = out.code_bytes;
        out.bytes          = out.code_bytes;
        return out;
    }
    if (layout == QuantLayout::Fp8ExpertB128) {
        constexpr std::uint64_t kTile = 128;
        if (format != QType::FP8_E4M3FN_B128 || shape.size() != 3 || shape[1] % kTile || shape[2] % kTile) {
            throw std::invalid_argument(
                "fp8_expert_b128_v1 requires fp8_e4m3fn_b128 [experts, hidden, intermediate] of whole 128 x 128 tiles");
        }
        const auto experts = shape[0], hidden = shape[1], inter = shape[2];
        const auto tiles   = mul(hidden / kTile, inter / kTile); // per matrix
        out.group_size         = kTile;
        out.alignment          = 4096;
        out.code_bytes_per_row = hidden; // a gate/up row: one byte per weight
        out.high_bytes_per_row = inter;  // a down row
        out.record_bytes       = add(mul(3, mul(hidden, inter)), mul(3, mul(tiles, 2)));
        out.record_stride      = aligned(out.record_bytes, 4096);
        out.code_bytes         = mul(experts, out.record_stride);
        out.scale_offset       = out.code_bytes;
        out.bytes              = out.code_bytes;
        return out;
    }
    if (shape.size() != 2) { throw std::invalid_argument("quantized weight must be a matrix"); }
    if (layout == QuantLayout::GgmlRows || layout == QuantLayout::GgmlTiles16) {
        if (!is_ggml(format) || shape[1] % ggml_block_elems(format)) {
            throw std::invalid_argument("ggml_rows_v1 requires a ggml format and whole blocks per row");
        }
        const bool tiles = layout == QuantLayout::GgmlTiles16;
        if (tiles && ggml_block_elems(format) != 256) {
            throw std::invalid_argument("ggml_tiles16_v1 requires a ggml format of 256-element blocks");
        }
        out.padded_columns     = shape[1];
        out.group_size         = ggml_block_elems(format);
        out.code_bytes_per_row = mul(shape[1] / ggml_block_elems(format), ggml_block_bytes(format));
        // A tiled matrix holds whole tiles of 16 rows (row r's tile at r / 16 * 16 rows).
        out.code_bytes   = mul(tiles ? (shape[0] + 15) / 16 * 16 : shape[0], out.code_bytes_per_row);
        out.scale_offset = out.code_bytes;
        out.bytes        = out.code_bytes;
        return out;
    }
    const auto n       = shape[0];
    const auto k       = shape[1];
    out.padded_columns = k;
    if (layout == QuantLayout::RowSplit) {
        std::uint64_t high_per_group = 0, code_per_group = 32;
        switch (format) {
        case QType::Q2_G64_FP16:
            out.group_size = 64;
            code_per_group = 16; // four 2-bit codes per byte
            break;
        case QType::Q4_G64_FP16:
            out.group_size = 64;
            break;
        case QType::Q5_G64_FP16:
            out.group_size = 64;
            high_per_group = 8;
            break;
        case QType::Q6_G64_FP16:
            out.group_size = 64;
            high_per_group = 16;
            break;
        case QType::Q8_G32_FP16:
            out.group_size = 32;
            break;
        default:
            throw std::invalid_argument("RowSplit requires a grouped integer format");
        }
        out.padded_columns      = aligned(k, 128);
        const auto groups       = out.padded_columns / out.group_size;
        out.code_bytes_per_row  = mul(groups, code_per_group);
        out.high_bytes_per_row  = mul(groups, high_per_group);
        out.scale_bytes_per_row = mul(groups, 2);
        out.code_bytes          = mul(n, out.code_bytes_per_row);
        out.high_offset         = aligned(out.code_bytes, 256);
        out.high_bytes          = mul(n, out.high_bytes_per_row);
        out.scale_offset        = add(out.high_offset, aligned(out.high_bytes, 256));
    } else if (layout == QuantLayout::RowScale) {
        if (!is_fp8_row(format)) { throw std::invalid_argument("RowScale requires row-scaled FP8"); }
        out.group_size          = k;
        out.code_bytes_per_row  = k;
        out.scale_bytes_per_row = format == QType::FP8_E4M3FN_ROW_F32 ? 4 : 2;
        out.code_bytes          = out.elements;
        out.scale_offset        = aligned(out.code_bytes, 256);
    } else if (layout == QuantLayout::Rg16) {
        if (format != QType::NVFP4_MUL || n % 16 || k % 16) {
            throw std::invalid_argument("nvfp4_rg16_v1 requires nvfp4_mul with N%16=0 and K%16=0");
        }
        // 16-row groups of 144-byte units: a group's bytes are (k / 16) * 144, and a row "owns" a
        // sixteenth of them (row planes address whole groups only).
        out.group_size     = 16;
        out.code_bytes     = mul(mul(n / 16, k / 16), 144);
        out.scale_offset   = out.code_bytes; // the scales live inside the units
        out.divisor_offset = aligned(out.code_bytes, 256);
        out.bytes          = add(out.divisor_offset, 4);
        return out;
    } else if (layout == QuantLayout::G32ScaleRows16) {
        if (format != QType::FP8_E4M3FN_G32_E8M0 || n % 16 || k % 32) {
            throw std::invalid_argument("g32_scale_rows16_v1 requires group-scaled FP8 with N%16=0 and K%32=0");
        }
        // A 16-row tile's scales are contiguous, so whole tiles of rows own scale_bytes_per_row each.
        out.group_size          = 32;
        out.code_bytes_per_row  = k;
        out.scale_bytes_per_row = k / 32;
        out.code_bytes          = out.elements;
        out.scale_offset        = aligned(out.code_bytes, 256);
    } else if (layout == QuantLayout::BlockScaleK16M128x4) {
        if (format != QType::NVFP4 || n % 128 || k % 64) {
            throw std::invalid_argument("NVFP4 BlockScale requires N%128=0 and K%64=0");
        }
        out.group_size          = 16;
        out.code_bytes_per_row  = k / 2;
        out.scale_bytes_per_row = k / 16;
        out.code_bytes          = out.elements / 2;
        out.scale_offset        = aligned(out.code_bytes, 256);
    } else {
        throw std::invalid_argument("unknown quantized weight layout");
    }
    out.scale_bytes = mul(n, out.scale_bytes_per_row);
    out.bytes       = add(out.scale_offset, out.scale_bytes);
    if (format == QType::NVFP4) {
        out.divisor_offset = out.bytes;
        out.bytes          = add(out.bytes, 4);
    }
    return out;
}

WeightRegion contiguous_weight_region(const WeightView& view) {
    if (view.parts.empty()) { throw std::invalid_argument("weight view has no regions"); }
    WeightRegion merged = view.parts.front();
    validate_region(merged);
    for (std::size_t i = 1; i < view.parts.size(); ++i) {
        const auto& next = view.parts[i];
        validate_region(next);
        if (next.parent != merged.parent || next.begin != merged.end) {
            throw std::invalid_argument("native input requires one contiguous parent region");
        }
        merged.end = next.end;
    }
    if (merged.end - merged.begin != weight_element_count(view.shape)) {
        throw std::invalid_argument("weight view shape differs from its regions");
    }
    return merged;
}

bool is_complete_weight(const WeightView& view) {
    if (view.parts.empty()) { return false; }
    const auto* parent = view.parts.front().parent;
    if (!parent || view.shape != parent->geometry.shape) { return false; }
    std::uint64_t end = 0;
    for (const auto& part : view.parts) {
        if (part.parent != parent || part.begin != end || part.end <= part.begin) { return false; }
        end = part.end;
    }
    return end == parent->geometry.elements;
}

std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                  std::uint64_t group) {
    if (geometry.shape.size() != 2 || !geometry.group_size || row >= geometry.shape[0] ||
        group >= geometry.padded_columns / geometry.group_size) {
        throw std::invalid_argument("weight scale coordinate exceeds its parent");
    }
    if (geometry.layout == QuantLayout::BlockScaleK16M128x4) {
        const auto inner = row % 128;
        return geometry.scale_offset + (row / 128 * (geometry.shape[1] / 64) + group / 4) * 512 +
               inner % 32 * 16 + inner / 32 * 4 + group % 4;
    }
    return geometry.scale_offset + row * geometry.scale_bytes_per_row + group * 2;
}

WeightRowPlanes weight_row_planes(const WeightRegion& region) {
    validate_region(region);
    const auto& parent = *region.parent;
    const auto& g      = parent.geometry;
    if (is_expert_bank_layout(g.layout) || g.layout == QuantLayout::Rg16) {
        throw std::invalid_argument("expert banks and nvfp4_rg16_v1 matrices have no row planes");
    }
    if (!parent.data || g.shape.size() != 2 || region.begin % g.shape[1] ||
        region.end % g.shape[1]) {
        throw std::invalid_argument("row view requires resident complete logical rows");
    }
    WeightRowPlanes out;
    out.row_begin       = region.begin / g.shape[1];
    out.row_count       = (region.end - region.begin) / g.shape[1];
    out.code_row_bytes  = g.code_bytes_per_row;
    out.high_row_bytes  = g.high_bytes_per_row;
    out.scale_row_bytes = g.scale_bytes_per_row;
    out.codes           = parent.data + out.row_begin * out.code_row_bytes;
    if (g.high_bytes) {
        out.high = parent.data + g.high_offset + out.row_begin * out.high_row_bytes;
    }
    out.swizzled_scales = g.layout == QuantLayout::BlockScaleK16M128x4;
    if (g.scale_bytes) {
        out.scales = parent.data + g.scale_offset +
                     (out.swizzled_scales ? 0 : out.row_begin * out.scale_row_bytes);
    }
    return out;
}

Tensor weight_tensor(const WeightView& view, std::initializer_list<std::int32_t> internal_shape) {
    const auto region = contiguous_weight_region(view);
    const auto& g     = region.parent->geometry;
    if (g.layout != QuantLayout::Contiguous || !region.parent->data) {
        throw std::invalid_argument("direct tensor requires a resident contiguous weight");
    }
    const auto offset = mul(region.begin, word_bytes(g.format));
    Tensor out(const_cast<std::byte*>(region.parent->data + offset), direct_dtype(g.format),
               internal_shape);
    if (static_cast<std::uint64_t>(out.numel()) != region.end - region.begin) {
        throw std::invalid_argument("native tensor shape differs from its logical weight");
    }
    return out;
}

Weight native_weight(const WeightView& view, float input_divisor) {
    const auto region = contiguous_weight_region(view);
    const auto& g     = region.parent->geometry;
    if (view.shape.size() != 2 || !region.parent->data) {
        throw std::invalid_argument("native Weight requires a resident logical matrix");
    }
    if (is_expert_bank_layout(g.layout)) {
        throw std::invalid_argument("expert banks bind through their planes");
    }
    Weight out;
    out.payload          = region.parent->data;
    out.payload_bytes    = g.bytes;
    out.high_plane_bytes = g.high_bytes;
    out.qtype            = g.format;
    out.layout           = g.layout;
    out.ndim             = 2;
    out.n = out.shape[0] = out.padded_shape[0] = dimension(view.shape[0]);
    out.k = out.shape[1] = out.padded_shape[1] = dimension(view.shape[1]);
    if (g.layout == QuantLayout::Contiguous) {
        out.qdata = region.parent->data + mul(region.begin, word_bytes(g.format));
        return out;
    }
    if (g.shape.size() != 2 || view.shape[1] != g.shape[1]) {
        throw std::invalid_argument("quantized native Weight requires unchanged parent K");
    }
    if (g.layout == QuantLayout::Rg16) {
        // The unit plane addresses whole 16-row groups; its consumers take the complete matrix.
        if (!is_complete_weight(view)) {
            throw std::invalid_argument("an nvfp4_rg16_v1 native Weight requires its complete parent");
        }
        out.qdata             = region.parent->data;
        out.group_size        = 16;
        out.group             = 16;
        out.padded_shape[1]   = out.k;
        out.weight_multiplier = region.parent->weight_multiplier;
        return out;
    }
    const auto planes = weight_row_planes(region);
    if ((g.layout == QuantLayout::RowScale || g.layout == QuantLayout::BlockScaleK16M128x4 ||
         g.layout == QuantLayout::G32ScaleRows16) &&
        !is_complete_weight(view)) {
        throw std::invalid_argument(
            "this native Weight input requires a complete FP8/NVFP4 parent");
    }
    out.padded_shape[1]      = dimension(g.padded_columns);
    out.qdata                = planes.codes;
    out.qhigh                = planes.high;
    out.scales               = planes.scales;
    out.group_size           = static_cast<std::uint32_t>(g.group_size);
    out.group                = g.group_size ? dimension(g.group_size) : 0;
    out.weight_scale_divisor = region.parent->weight_scale_divisor;
    out.input_scale_divisor  = input_divisor;
    if (g.layout == QuantLayout::RowSplit) {
        out.scale_dtype = DType::FP16;
        out.scale_ne[0] = dimension(g.padded_columns / g.group_size);
        out.scale_ne[1] = out.n;
        out.scale_nb[0] = 2;
        out.scale_nb[1] = static_cast<std::int64_t>(g.scale_bytes_per_row);
        out.scale_nb[2] = out.scale_nb[3] = out.scale_nb[1] * out.n;
    } else if (g.layout == QuantLayout::RowScale) {
        const std::int64_t word = g.format == QType::FP8_E4M3FN_ROW_F32 ? 4 : 2;
        out.scale_dtype = word == 4 ? DType::FP32 : DType::BF16;
        out.scale_ne[0] = out.n;
        out.scale_nb[0] = word;
        out.scale_nb[1] = out.scale_nb[2] = out.scale_nb[3] = static_cast<std::int64_t>(out.n) * word;
    } else if (g.layout == QuantLayout::BlockScaleK16M128x4) {
        out.scale_dtype = DType::FP8_E4M3FN;
    } else if (g.layout == QuantLayout::G32ScaleRows16) {
        // UE8M0 words: [n / 16][k / 32][16] (the layout's tile order).
        out.scale_dtype = DType::U8;
        out.scale_ne[0] = 16;
        out.scale_ne[1] = out.k / 32;
        out.scale_ne[2] = out.n / 16;
        out.scale_nb[0] = 1;
        out.scale_nb[1] = 16;
        out.scale_nb[2] = static_cast<std::int64_t>(out.k / 32) * 16;
        out.scale_nb[3] = out.scale_nb[2] * (out.n / 16);
    }
    return out;
}

Weight device_native_weight(const WeightView& view, float input_divisor) {
    Weight out           = native_weight(view, input_divisor);
    const auto* parent   = contiguous_weight_region(view).parent;
    if (!parent->device) { throw std::invalid_argument("device native Weight requires a device-visible parent"); }
    const auto rebase = [&](const void* p) -> const void* {
        return p ? parent->device + (static_cast<const std::byte*>(p) - parent->data) : nullptr;
    };
    out.payload = rebase(out.payload);
    out.qdata   = rebase(out.qdata);
    out.qhigh   = rebase(out.qhigh);
    out.scales  = rebase(out.scales);
    return out;
}

ExpertBankPlanes expert_bank_planes(const WeightParent& parent) {
    if (!parent.data) {
        throw std::invalid_argument("expert bank planes require a resident expert-bank parent");
    }
    // Only nvfp4_expert_rg16_v1 has a multiplier tail; the other banks' scales are inside each record.
    const bool tail   = parent.geometry.layout == QuantLayout::ExpertRg16;
    ExpertBankPlanes out = expert_bank_layout(
        parent.geometry, tail ? reinterpret_cast<const float*>(parent.data + parent.geometry.scale_offset) : nullptr);
    out.records        = parent.data;
    out.device_records = parent.device;
    return out;
}

ExpertBankPlanes expert_bank_layout(const WeightGeometry& g, const float* multipliers) {
    if (is_ggml_expert_layout(g.layout) || g.layout == QuantLayout::Fp8ExpertB128) {
        // Row-major gate, up and down rows (a block-FP8 record's tile scales follow the down rows).
        ExpertBankPlanes out;
        out.record_bytes   = g.record_bytes;
        out.record_stride  = g.record_stride;
        out.experts        = static_cast<std::uint32_t>(dimension(g.shape[0]));
        out.hidden         = static_cast<std::uint32_t>(dimension(g.shape[1]));
        out.intermediate   = static_cast<std::uint32_t>(dimension(g.shape[2]));
        out.gate_up_format = g.format;
        out.down_format    = g.layout == QuantLayout::Fp8ExpertB128 ? g.format : ggml_expert_down_format(g.layout);
        out.gate_row_bytes = g.code_bytes_per_row;
        out.down_row_bytes = g.high_bytes_per_row;
        out.gate_up_bytes  = 2ULL * out.intermediate * out.gate_row_bytes;
        return out;
    }
    if (g.layout != QuantLayout::ExpertRg16) {
        throw std::invalid_argument("expert bank layout requires an expert-bank parent");
    }
    ExpertBankPlanes out;
    out.multipliers   = multipliers;
    out.record_bytes  = g.record_bytes;
    out.record_stride = g.record_stride;
    out.experts       = static_cast<std::uint32_t>(dimension(g.shape[0]));
    out.hidden        = static_cast<std::uint32_t>(dimension(g.shape[1]));
    out.intermediate  = static_cast<std::uint32_t>(dimension(g.shape[2]));
    out.gate_up_bytes = 2ULL * out.intermediate / 16 * (out.hidden / 16) * 144;
    return out;
}

} // namespace infernix
