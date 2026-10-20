#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/q6/q6_shapes.h"
#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q6Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{248320, 5120, select_q6_n248320_k5120},
    ShapeEntry{34816, 5120, select_q6_n34816_k5120},
    ShapeEntry{248320, 2048, select_q6_n248320_k2048},
    ShapeEntry{1152, 1536, select_q6_n1152_k1536},
};

// The column-invariant routes: every launch is the sliced-K schedule with four K warps (the Q6
// reduction is linear); block rows, token capacity, stages and the token slicing change only staging
// and predication, so a column's bits do not depend on the call's width. Tiles follow the exact
// route's capacities where they share the family (3..32 at K=5120, 2..32 at K=2048).
Q6Launch select_invariant_k5120(std::int32_t tokens) {
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    return launch_q6_a16_sliced_r32_t32_w4_s1;
}

Q6Launch select_invariant_k2048(std::int32_t tokens) {
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    return launch_q6_a16_sliced_r32_t32_w4_s1;
}

// The decode-width projections: the heads and the Q6 27B's fused gate/up.
constexpr std::array kInvariantShapes{
    ShapeEntry{248320, 5120, select_invariant_k5120},
    ShapeEntry{34816, 5120, select_invariant_k5120},
    ShapeEntry{248320, 2048, select_invariant_k2048},
};
} // namespace

bool q6_column_invariant_route(std::int32_t n, std::int32_t k) noexcept {
    for (const auto& entry : kInvariantShapes) {
        if (entry.n == n && entry.k == k) return true;
    }
    return false;
}

Q6Launch select_q6_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q6 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q6 linear: unsupported shape");
}

Q6Launch select_q6_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q6 linear: unsupported policy");
    if (!column_invariant(policy)) return select_q6_a16_launch(n, k, t);
    if (t <= 0) throw std::invalid_argument("q6 linear: T must be positive");
    for (const auto& entry : kInvariantShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q6 linear: the shape registers no column-invariant route");
}

void q6_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    select_q6_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace infernix::ops::detail
