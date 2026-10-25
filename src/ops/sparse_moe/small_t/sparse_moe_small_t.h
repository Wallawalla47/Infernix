#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

inline constexpr std::int32_t kSparseMoeSmallTMin = 2;
// The fixed kernel domain covers the largest codec-specific small-T frontier.
inline constexpr std::int32_t kSparseMoeSmallTMax = 46;

enum class SparseMoeSmallTD3Schedule : std::uint8_t {
    Paths1,
    Paths3,
    Paths9,
};

enum class SparseMoeSmallTD4Schedule : std::uint8_t {
    Rows1,
    Rows2,
    Rows4,
};

struct SparseMoeSmallTPlan {
    std::int32_t tokens                   = 0;
    std::size_t workspace_bytes           = 0;
    SparseMoeSmallTD3Schedule d3_schedule = SparseMoeSmallTD3Schedule::Paths3;
    SparseMoeSmallTD4Schedule d4_schedule = SparseMoeSmallTD4Schedule::Rows1;
    // The router reduces each token as the decode router does (request-invariant exact route).
    bool column_invariant = false;
    // L2 hints (INFERNIX_TMP_SMALL_T_L2_HINTS, candidates §10.3 KW-06): the decode route's next
    // consumer warm from SparseMoeHints for D4's tail, as T = 1 takes it. No numeric effect.
    const void* next_weight_prefetch       = nullptr;
    std::size_t next_weight_prefetch_bytes = 0;
};

struct SparseMoeSmallTWorkspace {
    Tensor token_ids;
    Tensor token_alpha;
    Tensor shared_scale;
    Tensor scratch;
};

template <class Arena>
SparseMoeSmallTWorkspace allocate_sparse_moe_small_t_workspace(Arena& arena, std::int32_t tokens) {
    SparseMoeSmallTWorkspace out;
    const std::int32_t assignments = 8 * tokens;
    out.token_ids                  = arena.alloc(DType::I32, {assignments}, 16);
    out.token_alpha                = arena.alloc(DType::FP32, {assignments}, 16);
    out.shared_scale               = arena.alloc(DType::FP32, {tokens}, 16);
    // S1 uses [T,257,4] partial router scores. After S2, each token reuses
    // its [9,512] region for eight routed and one shared SwiGLU activations.
    out.scratch = arena.alloc(DType::FP32, {512, 9 * tokens}, 256);
    return out;
}

[[nodiscard]] bool sparse_moe_uses_small_t(std::int32_t tokens) noexcept;
[[nodiscard]] std::size_t sparse_moe_small_t_workspace_bytes(std::int32_t tokens);
[[nodiscard]] SparseMoeSmallTPlan
resolve_sparse_moe_small_t_plan(std::int32_t tokens, QType routed_gate_up, QType routed_down,
                                bool column_invariant = false, const SparseMoeHints& hints = {});

void sparse_moe_small_t_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoeSmallTPlan& plan,
                               const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
