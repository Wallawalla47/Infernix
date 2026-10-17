#pragma once

// Block geometry of the GGUF (ggml) formats Infernix stores verbatim (core/weight.h QType::GGML_*).
// The byte layouts are llama.cpp's (ggml-common.h); tools/artifact/codecs/ggml.py holds the exact
// decoders that the converter and the Op oracles use.

#include "core/weight.h"

#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_GGML_FORMAT_HD __host__ __device__
#else
#    define INFERNIX_GGML_FORMAT_HD
#endif

namespace infernix {

[[nodiscard]] INFERNIX_GGML_FORMAT_HD constexpr bool is_ggml(QType q) noexcept {
    return q >= QType::GGML_Q2_0 && q <= QType::GGML_IQ4_NL;
}

// Elements per block (the K-axis unit of a row).
[[nodiscard]] INFERNIX_GGML_FORMAT_HD constexpr std::uint32_t ggml_block_elems(QType q) noexcept {
    switch (q) {
    case QType::GGML_Q2_0:
        return 64;
    case QType::GGML_IQ4_NL:
        return 32;
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ4_XS:
        return 256;
    default:
        return 0;
    }
}

[[nodiscard]] INFERNIX_GGML_FORMAT_HD constexpr std::uint32_t ggml_block_bytes(QType q) noexcept {
    switch (q) {
    case QType::GGML_Q2_0:
        return 18;
    case QType::GGML_IQ4_NL:
        return 18;
    case QType::GGML_Q4_K:
        return 144;
    case QType::GGML_Q5_K:
        return 176;
    case QType::GGML_Q6_K:
        return 210;
    case QType::GGML_IQ2_S:
        return 82;
    case QType::GGML_IQ3_XXS:
        return 98;
    case QType::GGML_IQ3_S:
        return 110;
    case QType::GGML_IQ4_XS:
        return 136;
    default:
        return 0;
    }
}

// The down matrices' format of a ggml expert-bank layout (QType::FP32 for any other layout).
[[nodiscard]] INFERNIX_GGML_FORMAT_HD constexpr QType ggml_expert_down_format(QuantLayout layout) noexcept {
    return layout == QuantLayout::GgmlExpertIq4Nl ? QType::GGML_IQ4_NL
           : layout == QuantLayout::GgmlExpertQ2_0 ? QType::GGML_Q2_0
                                                    : QType::FP32;
}

[[nodiscard]] INFERNIX_GGML_FORMAT_HD constexpr bool is_ggml_expert_layout(QuantLayout layout) noexcept {
    return layout == QuantLayout::GgmlExpertIq4Nl || layout == QuantLayout::GgmlExpertQ2_0;
}

} // namespace infernix

#undef INFERNIX_GGML_FORMAT_HD
