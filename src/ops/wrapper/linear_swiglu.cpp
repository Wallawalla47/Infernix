#include "core/weight.h"
#include "infernix/ops/linear_swiglu.h"

#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/bf16/bf16_linear_swiglu_kernels.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_plan.h"

#include <cstdint>
#include <stdexcept>

namespace infernix::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
    case LinearPolicy::A16ColumnInvariant:
    case LinearPolicy::AllowA4Residual:
    case LinearPolicy::AllowA8Residual:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

} // namespace

bool linear_swiglu_column_invariant_route(QType qtype, std::int32_t gate_up_rows,
                                          std::int32_t input_rows) noexcept {
    if (qtype == QType::FP8_E4M3FN_ROW_F32) { return gate_up_rows == 2048 && input_rows == 3072; }
    return qtype == QType::Q4_G64_FP16 || qtype == QType::NVFP4 ||
           qtype == QType::FP8_E4M3FN_ROW_BF16;
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens, LinearRoutes routes) {
    validate_policy(policy);
    if (!valid_linear_routes(routes)) {
        throw std::invalid_argument("linear_swiglu workspace: invalid route flags");
    }
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
    if (column_invariant(policy) &&
        !linear_swiglu_column_invariant_route(qtype, gate_up_rows, input_rows)) {
        // One A16 call per column, on the exact route (no route flag applies).
        return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                      LinearPolicy::A16Only, 1, 1);
    }
    if (qtype == QType::Q8_G32_FP16) {
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::q8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::BF16) {
        // The register-streamed route of Qwen3.8-Flash-Next's shared expert, decode widths only.
        if (gate_up_rows != 1280 || input_rows != 2560 || max_tokens > detail::kBf16LinearSwiGluStreamMaxColumns) {
            throw std::invalid_argument("linear_swiglu workspace: BF16 serves gate/up [1280, 2560] at 1..16 columns");
        }
        return 0;
    }
    if (qtype == QType::Q4_G64_FP16) {
        return detail::q4_linear_swiglu_capacity_workspace_bytes(gate_up_rows, gate_up_rows / 2, input_rows,
                                                                 input_rows, min_tokens, max_tokens, routes,
                                                                 column_invariant(policy));
    }
    if (qtype == QType::NVFP4 && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16 && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::fp8_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    // Qwen3.5-122B-A10B's FP8 shared expert: A16 routes, no workspace.
    if (qtype == QType::FP8_E4M3FN_ROW_F32 && gate_up_rows == 2048 && input_rows == 3072) { return 0; }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream, LinearRoutes routes) {
    validate_policy(policy);
    if (!valid_linear_routes(routes)) {
        throw std::invalid_argument("linear_swiglu: invalid route flags");
    }
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    if (column_invariant(policy) &&
        !linear_swiglu_column_invariant_route(gate_up_weight.qtype, gate_up_weight.n,
                                              gate_up_weight.k)) {
        // No dedicated column-invariant route: the exact one-column route, once per column.
        for (std::int32_t column = 0; column < x.ne[1]; ++column) {
            const Tensor x_column = x.slice(1, column, 1);
            Tensor out_column     = out.slice(1, column, 1);
            linear_swiglu(x_column, gate_up_weight, out_column, LinearPolicy::A16Only, ws, stream);
        }
        return;
    }
    const std::int32_t t   = x.ne[1];
    const bool large_shape = x.ne[0] == 5120 && out.ne[0] == 17408 && gate_up_weight.n == 34816 &&
                             gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 34816 &&
                             gate_up_weight.padded_shape[1] == 5120;
    const bool q8_shape = (x.ne[0] == 2048 && out.ne[0] == 6144 && gate_up_weight.n == 12288 &&
                           gate_up_weight.k == 2048 && gate_up_weight.padded_shape[0] == 12288 &&
                           gate_up_weight.padded_shape[1] == 2048) ||
                          (x.ne[0] == 2560 && out.ne[0] == 640 && gate_up_weight.n == 1280 &&
                           gate_up_weight.k == 2560 && gate_up_weight.padded_shape[0] == 1280 &&
                           gate_up_weight.padded_shape[1] == 2560) ||
                          // Qwen3.8-Flash-Next's and Qwen3.5-122B-A10B's DFlash2 MLPs.
                          (x.ne[0] == 2560 && out.ne[0] == 7680 && gate_up_weight.n == 15360 &&
                           gate_up_weight.k == 2560 && gate_up_weight.padded_shape[0] == 15360 &&
                           gate_up_weight.padded_shape[1] == 2560) ||
                          (x.ne[0] == 3072 && out.ne[0] == 9216 && gate_up_weight.n == 18432 &&
                           gate_up_weight.k == 3072 && gate_up_weight.padded_shape[0] == 18432 &&
                           gate_up_weight.padded_shape[1] == 3072);
    // Qwen3.5-122B-A10B's shared expert (fp8_e4m3fn_row_f32, dealignai's Ornith-1.5-122B-UNCENSORED).
    const bool shared_122b = x.ne[0] == 3072 && out.ne[0] == 1024 && gate_up_weight.n == 2048 &&
                             gate_up_weight.k == 3072 && gate_up_weight.padded_shape[0] == 2048 &&
                             gate_up_weight.padded_shape[1] == 3072;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !q8_shape && !shared_122b)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }
    if (gate_up_weight.qtype == QType::BF16) {
        if (!detail::bf16_linear_swiglu_stream_pair_supported(x, gate_up_weight, out)) {
            throw std::invalid_argument("linear_swiglu: BF16 serves gate/up [1280, 2560] at 1..16 columns");
        }
        detail::bf16_linear_swiglu_stream_pair_launch(x, gate_up_weight, out, stream);
        return;
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight = large_shape && gate_up_weight.qtype == QType::Q4_G64_FP16 &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                           common_row_split;
    const bool q8_weight =
        (q8_shape || large_shape) && gate_up_weight.qtype == QType::Q8_G32_FP16 &&
        gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
        gate_up_weight.qhigh == nullptr && gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool nvfp4_weight = large_shape && gate_up_weight.qtype == QType::NVFP4;
    const bool fp8_weight   = (large_shape && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16) ||
                            (shared_122b && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_F32);
    if (!q4_weight && !q8_weight && !nvfp4_weight && !fp8_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (fp8_weight) {
        (void)detail::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (nvfp4_weight) {
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, q8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }

    if (q8_weight) {
        detail::q8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream, routes, column_invariant(policy));
    }
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

} // namespace infernix::ops
