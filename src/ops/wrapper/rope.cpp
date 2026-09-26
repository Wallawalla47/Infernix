#include "ninfer/ops/rope.h"

#include "ops/launcher/rope.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kTextHeadDim = 256;
constexpr std::int32_t kVisionDim   = 72;

std::int64_t numel_allow_zero(const Tensor& tensor, const char* label) {
    bool zero      = false;
    std::int64_t n = 1;
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] < 0) {
            throw std::invalid_argument(std::string("rope: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (tensor.ne[dim] == 0) {
            zero = true;
            continue;
        }
        if (n > std::numeric_limits<std::int64_t>::max() / tensor.ne[dim]) {
            throw std::overflow_error("rope: tensor size overflows int64");
        }
        n *= tensor.ne[dim];
    }
    return zero ? 0 : n;
}

int position_axes(const Tensor& positions, std::int32_t tokens) {
    if (positions.ne[0] != tokens || positions.ne[2] != 1 || positions.ne[3] != 1 ||
        (positions.ne[1] != 1 && positions.ne[1] != 2 && positions.ne[1] != 3)) {
        throw std::invalid_argument("rope: positions must have shape [T], [T,2], or [T,3]");
    }
    return positions.ne[1];
}

void require_tensor_layout(const Tensor& tensor, const char* label, std::int32_t head_dim,
                           std::int32_t heads, std::int32_t tokens) {
    if (heads <= 0) {
        throw std::invalid_argument(std::string("rope: ") + label + " must have positive heads");
    }
    if (tensor.ne[0] != head_dim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * head_dim ||
        tensor.nb[2] < elem * static_cast<std::int64_t>(head_dim) * heads ||
        (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string("rope: invalid ") + label + " strides");
    }
}

void require_common(const Tensor& positions, int rotary_dim, float theta,
                    DeviceExecutionView execution) {
    if (execution.multiprocessor_count <= 0) {
        throw std::invalid_argument("rope: positive multiprocessor count required");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope: positions must be I32");
    }
    if (!(theta > 0.0f) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: theta must be positive and finite");
    }
    if (rotary_dim <= 0 || (rotary_dim & 1) != 0) {
        throw std::invalid_argument("rope: rotary_dim must be positive and even");
    }
}

void require_positions_storage(const Tensor& positions) {
    if (!positions.is_contiguous()) {
        throw std::invalid_argument("rope: positions must be contiguous");
    }
    if (positions.data == nullptr) {
        throw std::invalid_argument("rope: positions data must be non-null");
    }
}

void require_model_mode(int axes, int rotary_dim, std::int32_t head_dim) {
    if (axes == 2) {
        if (head_dim != kVisionDim || rotary_dim != kVisionDim) {
            throw std::invalid_argument("rope: 2-D Vision mode requires head_dim=rotary_dim=72");
        }
        return;
    }
    if (axes == 1 && head_dim == 128 && rotary_dim == 128) { return; }
    if (head_dim != kTextHeadDim || rotary_dim > kTextHeadDim) {
        throw std::invalid_argument("rope: Text mode requires D256 or one-dimensional D128/R128");
    }
    if (axes == 3 && rotary_dim != 64) {
        throw std::invalid_argument("rope: 3-D Text MRoPE requires rotary_dim=64");
    }
}

void require_scaling(const RopeScaling& scaling, float theta) {
    if (!std::isfinite(scaling.factor) || scaling.factor < 1.0F || scaling.factor > 4.0F ||
        scaling.original_max_positions == 0) {
        throw std::invalid_argument("rope: YaRN requires factor in [1,4] and positive original positions");
    }
    if (scaling.factor != 1.0F && !(theta > 1.0F)) {
        throw std::invalid_argument("rope: YaRN requires Text mode and theta>1");
    }
}


} // namespace

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          DeviceExecutionView execution) {
    require_common(positions, rotary_dim, theta, execution);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    (void)numel_allow_zero(positions, "positions");
    const std::int64_t q_numel = numel_allow_zero(q, "q");
    (void)numel_allow_zero(k, "k");
    const std::int32_t tokens   = q.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = axes == 2 ? kVisionDim : q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t k_heads  = k.ne[1];
    require_model_mode(axes, rotary_dim, head_dim);
    require_tensor_layout(q, "q", head_dim, q_heads, tokens);
    require_tensor_layout(k, "k", head_dim, k_heads, tokens);
    if (q_numel == 0) { return; }
    require_positions_storage(positions);
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: q/k data must be non-null");
    }
    detail::rope_launch(positions, rotary_dim, theta, q, k, execution);
}

void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          DeviceExecutionView execution) {
    require_common(positions, rotary_dim, theta, execution);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    (void)numel_allow_zero(positions, "positions");
    const std::int64_t x_numel  = numel_allow_zero(x, "tensor");
    const std::int32_t tokens   = x.ne[2];
    const int axes              = position_axes(positions, tokens);
    const std::int32_t head_dim = axes == 2 ? kVisionDim : x.ne[0];
    const std::int32_t heads    = x.ne[1];
    require_model_mode(axes, rotary_dim, head_dim);
    require_tensor_layout(x, "tensor", head_dim, heads, tokens);
    if (x_numel == 0) { return; }
    require_positions_storage(positions);
    if (x.data == nullptr) { throw std::invalid_argument("rope: tensor data must be non-null"); }
    detail::rope_single_launch(positions, rotary_dim, theta, x, execution);
}

void rope(const Tensor& positions, const PreparedRope& prepared, Tensor& q, Tensor& k,
          DeviceExecutionView execution) {
    const int rotary_dim = prepared.rotary_dim;
    const float theta    = prepared.theta;
    if (prepared.factor == 1.0F) { return rope(positions, rotary_dim, theta, q, k, execution); }
    if (positions.ne[1] == 2) {
        throw std::invalid_argument("rope: YaRN does not support Vision");
    }
    require_common(positions, rotary_dim, theta, execution);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope: q/k must be BF16");
    }
    (void)numel_allow_zero(positions, "positions");
    const auto count = numel_allow_zero(q, "q");
    (void)numel_allow_zero(k, "k");
    const int axes = position_axes(positions, q.ne[2]);
    require_model_mode(axes, rotary_dim, q.ne[0]);
    require_tensor_layout(q, "q", q.ne[0], q.ne[1], q.ne[2]);
    require_tensor_layout(k, "k", q.ne[0], k.ne[1], q.ne[2]);
    if (count == 0) { return; }
    require_positions_storage(positions);
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope: q/k data must be non-null");
    }
    detail::rope_prepared_launch(positions, prepared, q, &k, execution);
}

void rope(const Tensor& positions, const PreparedRope& prepared, Tensor& x,
          DeviceExecutionView execution) {
    const int rotary_dim = prepared.rotary_dim;
    const float theta    = prepared.theta;
    if (prepared.factor == 1.0F) { return rope(positions, rotary_dim, theta, x, execution); }
    if (positions.ne[1] == 2) {
        throw std::invalid_argument("rope: YaRN does not support Vision");
    }
    require_common(positions, rotary_dim, theta, execution);
    if (x.dtype != DType::BF16) { throw std::invalid_argument("rope: tensor must be BF16"); }
    (void)numel_allow_zero(positions, "positions");
    const auto count = numel_allow_zero(x, "tensor");
    const int axes = position_axes(positions, x.ne[2]);
    require_model_mode(axes, rotary_dim, x.ne[0]);
    require_tensor_layout(x, "tensor", x.ne[0], x.ne[1], x.ne[2]);
    if (count == 0) { return; }
    require_positions_storage(positions);
    if (x.data == nullptr) { throw std::invalid_argument("rope: tensor data must be non-null"); }
    detail::rope_prepared_launch(positions, prepared, x, nullptr, execution);
}


PreparedRope prepare_rope(int rotary_dim, float theta, const RopeScaling& scaling) {
    require_scaling(scaling, theta);
    if (rotary_dim <= 0 || rotary_dim > 256 || (rotary_dim & 1) != 0 ||
        !(theta > 0.0F) || !std::isfinite(theta)) {
        throw std::invalid_argument("rope: preparation requires even R in [2,256] and finite positive theta");
    }
    PreparedRope prepared;
    prepared.rotary_dim = rotary_dim;
    prepared.theta = theta;
    prepared.factor = scaling.factor;
    if (scaling.factor == 1.0F) { return prepared; }
    constexpr double two_pi = 6.283185307179586476925286766559;
    const auto correction = [&](double beta) {
        return rotary_dim * std::log(scaling.original_max_positions / (two_pi * beta)) /
               (2.0 * std::log(static_cast<double>(theta)));
    };
    const double low = std::max(std::floor(correction(32.0)), 0.0);
    double high = std::min(std::ceil(correction(1.0)), static_cast<double>(rotary_dim - 1));
    if (low == high) { high += 0.001; }
    prepared.attention_scale = static_cast<float>(1.0 + 0.1 * std::log(static_cast<double>(scaling.factor)));
    for (int pair = 0; pair < rotary_dim / 2; ++pair) {
        const double ramp = std::clamp((pair - low) / (high - low), 0.0, 1.0);
        prepared.inverse[pair] = std::pow(static_cast<double>(theta), -2.0 * pair / rotary_dim) *
                                ((1.0 - ramp) + ramp / scaling.factor);
    }
    return prepared;
}

} // namespace ninfer::ops
