#include "ops/linear/mxfp8/mxfp8_dispatch.h"

#include "ops/linear/mxfp8/mxfp8_format.h"
#include "ops/linear/mxfp8/mxfp8_shapes.h"

#include <array>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

const std::array kShapes{&kMxfp8N16384K2560, &kMxfp8N13952K2560, &kMxfp8N2560K6144, &kMxfp8N1280K2560,
                         &kMxfp8N2560K640};

// Every admitted policy runs the exact A16 route (the format has no activation-quantized route), which
// gives a column the same bits at every T, so it serves A16ColumnInvariant as it is.
const Mxfp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) { throw std::invalid_argument("mxfp8 linear: unsupported policy"); }
    for (const auto* shape : kShapes) {
        if (shape->n == n && shape->k == k) { return *shape; }
    }
    throw std::invalid_argument("mxfp8 linear: unsupported shape");
}

} // namespace

std::size_t mxfp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                  std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("mxfp8 linear workspace: invalid token interval");
    }
    (void)resolve_shape(n, k, policy);
    return 0;
}

void mxfp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, cudaStream_t stream) {
    validate_mxfp8_weight(weight, "mxfp8 linear");
    if (x.ne[1] <= 0) { throw std::invalid_argument("mxfp8 linear: T must be positive"); }
    resolve_shape(weight.n, weight.k, policy).a16(x, weight, out, stream);
}

} // namespace infernix::ops::detail
