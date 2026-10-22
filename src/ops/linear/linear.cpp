#include "core/weight.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/rg16_linear.h"

#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/mxfp8/mxfp8_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q8/q8_dispatch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
    case LinearPolicy::A16ColumnInvariant:
    case LinearPolicy::AllowA4Residual:
    case LinearPolicy::AllowA8Residual:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy, LinearRoutes routes,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4_G64_FP16:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5_G64_FP16:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6_G64_FP16:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q8_G32_FP16:
        detail::q8_dispatch(x, w, out, policy, routes, stream);
        return;
    case QType::BF16:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16:
        detail::fp8_dispatch(x, w, out, policy, routes, workspace, stream);
        return;
    case QType::FP8_E4M3FN_G32_E8M0:
        detail::mxfp8_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4_MUL:
        // A dense nvfp4_rg16_v1 matrix: the canonical W4A16 projection (exact sums), every policy.
        if (w.layout != QuantLayout::Rg16 || workspace == nullptr) { break; }
        rg16_linear(x, w, out, *workspace, stream);
        return;
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

bool linear_column_invariant_route(QType qtype, std::int32_t output_rows,
                                   std::int32_t input_rows) noexcept {
    if (qtype == QType::Q4_G64_FP16) { return detail::q4_column_invariant_route(output_rows, input_rows); }
    if (qtype == QType::Q6_G64_FP16) { return detail::q6_column_invariant_route(output_rows, input_rows); }
    if (qtype == QType::Q8_G32_FP16) { return detail::q8_column_invariant_route(output_rows, input_rows); }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16) {
        return detail::fp8_linear_column_invariant_route(output_rows, input_rows);
    }
    // nvfp4_rg16_v1 sums exactly: every route is column-invariant.
    if (qtype == QType::NVFP4_MUL) { return true; }
    return false;
}

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }
    if (column_invariant(policy) && !linear_column_invariant_route(qtype, output_rows, input_rows)) {
        // One A16 call per column.
        return linear_workspace_capacity_bytes(qtype, output_rows, input_rows, LinearPolicy::A16Only,
                                               1, 1);
    }

    switch (qtype) {
    case QType::Q4_G64_FP16:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5_G64_FP16:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6_G64_FP16:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q8_G32_FP16:
        (void)detail::select_q8_launch(output_rows, input_rows, min_tokens, policy, LinearRoutes::Exact);
        (void)detail::select_q8_launch(output_rows, input_rows, max_tokens, policy, LinearRoutes::Exact);
        return 0;
    case QType::BF16:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP8_E4M3FN_G32_E8M0:
        return detail::mxfp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy, min_tokens, max_tokens);
    case QType::NVFP4_MUL:
        // The encoded activations grow with the columns.
        return rg16_linear_workspace_bytes(output_rows, input_rows, max_tokens);
    case QType::FP32:
    case QType::INT32:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream, LinearRoutes routes) {
    validate_linear_semantics(x, w, out, policy);
    if (!valid_linear_routes(routes)) {
        throw std::invalid_argument("linear: unsupported route set");
    }
    if (column_invariant(policy) && !linear_column_invariant_route(w.qtype, w.n, w.k)) {
        // No dedicated column-invariant route: the exact one-column route, once per column.
        for (std::int32_t column = 0; column < x.ne[1]; ++column) {
            const Tensor x_column = x.slice(1, column, 1);
            Tensor out_column     = out.slice(1, column, 1);
            linear(x_column, w, out_column, LinearPolicy::A16Only, workspace, stream);
        }
        return;
    }
    dispatch_linear(x, w, out, policy, routes, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, LinearRoutes::Exact, nullptr, stream);
}

} // namespace infernix::ops
