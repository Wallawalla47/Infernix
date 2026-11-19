#pragma once

#include "core/layout.h"
#include "ops/ggml/ggml_linear_a8.h"

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// The workspace of an NVFP4 projection's 8-bit route (LinearPolicy::A8Only): its encoded columns
// (nvfp4_a8_encode in nvfp4_a8_mma.cuh).
inline std::size_t nvfp4_a8_workspace_capacity_bytes(std::int32_t k, std::int32_t max_tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::I32, {a8_linear_column_words(k, false), max_tokens}, 256);
    return layout.peak_bytes(1);
}

} // namespace infernix::ops::detail
