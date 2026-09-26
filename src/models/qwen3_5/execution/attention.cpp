#include "models/qwen3_5/execution/attention.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rope.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

// The fused text form is registered for the two text head geometries with a one-dimensional
// position axis. The MRoPE path and any other geometry take the three calls it replaces.
//
// It serves every width. Measured on an RTX 5090 (graph-timed, both geometries), the fused form
// is 44 to 48 % faster than the three calls at 256 tokens and still 13 to 20 % faster at the
// 3584- and 4096-token prefill chunks: it reads and writes q and k once, where the three calls
// pass over them twice. Both branches are the same arithmetic bit for bit.

// The fused text Op's formula fixes the native schedule's constants rather than taking them as
// operands; any other theta, epsilon or YaRN coefficient takes the three calls.
constexpr float kFusedTextRopeTheta = 1.0e7F;
constexpr float kFusedTextNormEpsilon = 1.0e-6F;

bool fused_text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                             const AttentionConfig& attention, float rms_norm_eps,
                             const ops::PreparedRope& prepared) {
    return prepared.factor == 1.0F && prepared.theta == kFusedTextRopeTheta &&
           rms_norm_eps == kFusedTextNormEpsilon && positions.ne[1] == 1 &&
           attention.head_dim == 256 && rope.rotary_dim == 64 &&
           ((attention.num_attention_heads == 16 && attention.num_key_value_heads == 2) ||
            (attention.num_attention_heads == 24 && attention.num_key_value_heads == 4));
}

} // namespace

std::size_t attention_projection_workspace_bytes(const AttentionParameters& parameters,
                                                 std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        const auto& weight = single->weight;
        return ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                             single->policy, first, last);
    }
    return 0;
}

void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::attn_input_proj(hidden, single.weight, query, gate, key, value, single.policy,
                             workspace, stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, const ops::PreparedRope& prepared,
               Tensor& query, DeviceExecutionView execution) {
    require_rope_axes(positions, config);
    ops::rope(positions, prepared, query, execution);
}

void text_qk_norm_rope(const Tensor& positions, const RopeConfig& rope,
                       const AttentionConfig& attention, float rms_norm_eps,
                       const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                       const ops::PreparedRope& prepared,
                       const Tensor& query, const Tensor& key, Tensor& normalized_query,
                       Tensor& normalized_key, DeviceExecutionView execution) {
    require_rope_axes(positions, rope);
    if (fused_text_qk_norm_rope(positions, rope, attention, rms_norm_eps, prepared)) {
        ops::rmsnorm_rope(positions, q_norm_weight, k_norm_weight, query, key, normalized_query,
                          normalized_key, execution.stream);
        return;
    }
    ops::rmsnorm(query, q_norm_weight, rms_norm_eps, true, normalized_query, execution.stream);
    ops::rmsnorm(key, k_norm_weight, rms_norm_eps, true, normalized_key, execution.stream);
    ops::rope(positions, prepared, normalized_query, normalized_key, execution);
}

} // namespace ninfer::models::qwen3_5::execution
