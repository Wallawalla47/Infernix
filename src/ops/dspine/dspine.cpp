// Ported from NInfer-all (https://github.com/iamwavecut/ninfer-all, PRs #341 and #365; Apache-2.0) and modified
// for Infernix (renamed into Infernix's Op contract and its oracle test, the proposal draw sub-keyed by step
// with the Gumbel-max coupled draw: docs/maintainer/dspine.md).
#include "infernix/ops/dspine.h"

#include "core/layout.h"
#include "ops/dspine/launch.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr std::int32_t kMaxRank = 512;

[[noreturn]] void fail(const char* op, const std::string& message) {
    throw std::invalid_argument(std::string(op) + ": " + message);
}

void require(const Tensor& tensor, DType dtype, std::initializer_list<std::int32_t> shape,
             const char* op, const char* name) {
    std::int32_t extents[4] = {1, 1, 1, 1};
    int d                   = 0;
    for (const std::int32_t extent : shape) extents[d++] = extent;
    if (tensor.dtype != dtype || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != extents[0] || tensor.ne[1] != extents[1] || tensor.ne[2] != extents[2] ||
        tensor.ne[3] != extents[3]) {
        fail(op, std::string(name) + " has the wrong type, shape or layout");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & 15U) != 0) {
        fail(op, std::string(name) + " must be 16-byte aligned");
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto first  = reinterpret_cast<std::uintptr_t>(a.data);
    const auto second = reinterpret_cast<std::uintptr_t>(b.data);
    return first < second + b.bytes() && second < first + a.bytes();
}

void require_apart(const char* op, std::initializer_list<const Tensor*> tensors) {
    for (auto i = tensors.begin(); i != tensors.end(); ++i) {
        for (auto j = i + 1; j != tensors.end(); ++j) {
            if (overlaps(**i, **j)) fail(op, "inputs and outputs must not overlap");
        }
    }
}

void require_width(std::int32_t width, std::int32_t batch, const char* op) {
    if (width < 1 || width > 16 || batch < 1 || batch > 8) {
        fail(op, "the block width must be 1..16 and the batch 1..8");
    }
}

void require_positive(float eps, const char* op) {
    if (!(eps > 0.0F) || !std::isfinite(eps)) fail(op, "eps must be positive and finite");
}

} // namespace

void dspine_bridge_read(const Tensor& raw, float eps, Tensor& normalized, Tensor& rms,
                        cudaStream_t stream) {
    constexpr const char* op = "dspine_bridge_read";
    const std::int32_t h = raw.ne[0], n = raw.ne[1];
    if (h <= 0 || h % 8 != 0 || n <= 0 || raw.ne[2] != 1 || raw.ne[3] != 1) {
        fail(op, "raw must be BF16 [H,N] with H a multiple of 8");
    }
    require_positive(eps, op);
    require(raw, DType::BF16, {h, n}, op, "raw");
    require(normalized, DType::BF16, {h, n}, op, "normalized");
    require(rms, DType::FP32, {n}, op, "rms");
    require_apart(op, {&raw, &normalized, &rms});
    detail::dspine_bridge_read_launch(static_cast<const __nv_bfloat16*>(raw.data), h, n, eps,
                                      static_cast<__nv_bfloat16*>(normalized.data),
                                      static_cast<float*>(rms.data), stream);
}

void dspine_bridge_codes(const Tensor& projected, float eps, Tensor& codes, cudaStream_t stream) {
    constexpr const char* op = "dspine_bridge_codes";
    const std::int32_t c = projected.ne[0], n = projected.ne[1];
    if (c <= 0 || c % 8 != 0 || n <= 0 || projected.ne[2] != 1 || projected.ne[3] != 1) {
        fail(op, "projected must be BF16 [C,N] with C a multiple of 8");
    }
    require_positive(eps, op);
    require(projected, DType::BF16, {c, n}, op, "projected");
    require(codes, DType::BF16, {c, n}, op, "codes");
    if (codes.data != projected.data && overlaps(codes, projected)) {
        fail(op, "codes may alias projected exactly, or not overlap it");
    }
    detail::dspine_bridge_codes_launch(static_cast<const __nv_bfloat16*>(projected.data), c, n,
                                       eps, static_cast<__nv_bfloat16*>(codes.data), stream);
}

void dspine_bridge_message(const Tensor& packed, const Tensor& anchors, const Tensor& sender_table,
                           const Tensor& gate_bias, Tensor& message, cudaStream_t stream) {
    constexpr const char* op = "dspine_bridge_message";
    const std::int32_t rank = gate_bias.ne[0];
    const std::int32_t w = packed.ne[1], b = packed.ne[2];
    if (rank <= 0 || rank % 32 != 0 || rank > kMaxRank) {
        fail(op, "the rank must be a multiple of 32 up to 512");
    }
    require_width(w, b, op);
    const std::int32_t vocab = sender_table.ne[1];
    require(packed, DType::BF16, {3 * rank, w, b}, op, "packed");
    require(anchors, DType::I32, {b}, op, "anchors");
    if (vocab <= 0) fail(op, "the sender table needs rows");
    require(sender_table, DType::BF16, {2 * rank, vocab}, op, "sender_table");
    require(gate_bias, DType::BF16, {rank}, op, "gate_bias");
    require(message, DType::BF16, {rank, w, b}, op, "message");
    require_apart(op, {&packed, &anchors, &sender_table, &gate_bias, &message});
    detail::dspine_bridge_message_launch(
        static_cast<const __nv_bfloat16*>(packed.data),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const __nv_bfloat16*>(sender_table.data),
        static_cast<const __nv_bfloat16*>(gate_bias.data), rank, w, b,
        static_cast<__nv_bfloat16*>(message.data), stream);
}

void dspine_bridge_finish(const Tensor& raw, const Tensor& rms, const Tensor& delta,
                          const Tensor& norm_weight, float eps, Tensor& out, Tensor& normed,
                          Tensor* denominator, cudaStream_t stream) {
    constexpr const char* op = "dspine_bridge_finish";
    const std::int32_t h = raw.ne[0], w = raw.ne[1], b = raw.ne[2];
    if (h <= 0 || h % 8 != 0) fail(op, "H must be a positive multiple of 8");
    require_width(w, b, op);
    require_positive(eps, op);
    require(raw, DType::BF16, {h, w, b}, op, "raw");
    require(rms, DType::FP32, {w, b}, op, "rms");
    require(delta, DType::BF16, {h, w, b}, op, "delta");
    require(norm_weight, DType::BF16, {h}, op, "norm_weight");
    require(out, DType::BF16, {h, w, b}, op, "out");
    require(normed, DType::BF16, {h, w, b}, op, "normed");
    if (out.data != raw.data && overlaps(out, raw)) {
        fail(op, "out may alias raw exactly, or not overlap it");
    }
    require_apart(op, {&raw, &rms, &delta, &norm_weight, &normed});
    require_apart(op, {&out, &rms, &delta, &norm_weight, &normed});
    float* denominator_data = nullptr;
    if (denominator != nullptr) {
        require(*denominator, DType::FP32, {w, b}, op, "denominator");
        // out may alias raw, so the denominator is checked against each of them apart.
        require_apart(op, {denominator, &raw, &rms, &delta, &norm_weight, &normed});
        require_apart(op, {denominator, &out});
        denominator_data = static_cast<float*>(denominator->data);
    }
    detail::dspine_bridge_finish_launch(
        static_cast<const __nv_bfloat16*>(raw.data), static_cast<const float*>(rms.data),
        static_cast<const __nv_bfloat16*>(delta.data),
        static_cast<const __nv_bfloat16*>(norm_weight.data), eps, h, w, b,
        static_cast<__nv_bfloat16*>(out.data), static_cast<__nv_bfloat16*>(normed.data),
        denominator_data, stream);
}

void dspine_scale_rows(const Tensor& rows, const Tensor& weight, Tensor& out,
                       cudaStream_t stream) {
    constexpr const char* op = "dspine_scale_rows";
    const std::int32_t h = rows.ne[0], n = rows.ne[1];
    if (h <= 0 || h % 8 != 0 || n <= 0 || rows.ne[2] != 1 || rows.ne[3] != 1) {
        fail(op, "rows must be BF16 [H,N] with H a multiple of 8");
    }
    require(rows, DType::BF16, {h, n}, op, "rows");
    require(weight, DType::BF16, {h}, op, "weight");
    require(out, DType::BF16, {h, n}, op, "out");
    if (out.data != rows.data && overlaps(out, rows)) {
        fail(op, "out may alias rows exactly, or not overlap it");
    }
    require_apart(op, {&weight, &out});
    detail::dspine_scale_rows_launch(static_cast<const __nv_bfloat16*>(rows.data),
                                     static_cast<const __nv_bfloat16*>(weight.data), h, n,
                                     static_cast<__nv_bfloat16*>(out.data), stream);
}

void dspine_store_head_write(const Tensor& logits, std::int32_t first, Tensor& head_write,
                             cudaStream_t stream) {
    constexpr const char* op = "dspine_store_head_write";
    const std::int32_t vocab = logits.ne[0], n = logits.ne[1], rank = head_write.ne[0];
    if (vocab <= 0 || n <= 0 || logits.ne[2] != 1 || logits.ne[3] != 1 ||
        (logits.dtype != DType::BF16 && logits.dtype != DType::FP32) || !logits.is_contiguous() ||
        logits.data == nullptr) {
        fail(op, "logits must be contiguous BF16 or FP32 [V,N]");
    }
    if (rank <= 0 || rank % 32 != 0 || rank > kMaxRank || first < 0 || first + n > rank) {
        fail(op, "the columns must lie inside a rank that is a multiple of 32 up to 512");
    }
    require(head_write, DType::BF16, {rank, vocab}, op, "head_write");
    require_apart(op, {&logits, &head_write});
    detail::dspine_store_head_write_launch(logits.data, logits.dtype == DType::FP32, vocab, n,
                                           rank, first,
                                           static_cast<__nv_bfloat16*>(head_write.data), stream);
}

std::size_t dspine_path_workspace_capacity_bytes(std::int32_t max_steps, std::int32_t max_batch) {
    if (max_steps < 1 || max_steps > 15 || max_batch < 1 || max_batch > 8) {
        fail("dspine_path workspace", "K must be 1..15 and B 1..8");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {kDSpineCandidates, kDSpineCandidates, max_steps, max_batch});
    return layout.peak_bytes(1);
}

void dspine_path(const Tensor& candidate_ids, const Tensor& candidate_scores,
                 const Tensor& denominator, const Tensor& rms, const Tensor& packed,
                 const Tensor& message, const Tensor& head_write, const Tensor& sender_table,
                 const Tensor& gate_bias, const Tensor& anchors, const Tensor& base_positions,
                 const SamplingConfig* configs, Tensor& drafts, Tensor& proposal_q,
                 Tensor* confidence, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "dspine_path";
    const std::int32_t k = candidate_ids.ne[1], b = candidate_ids.ne[2];
    const std::int32_t w = rms.ne[0];
    const std::int32_t rank = gate_bias.ne[0], vocab = head_write.ne[1];
    if (k < 1 || k > 15 || b < 1 || b > 8 || w < k + 1 || w > 16) {
        fail(op, "K must be 1..15, B 1..8 and the block at least K + 1 wide");
    }
    if (rank <= 0 || rank % 32 != 0 || rank > kMaxRank) {
        fail(op, "the rank must be a multiple of 32 up to 512");
    }
    if (configs == nullptr) fail(op, "the sampling configs are required");
    require(candidate_ids, DType::I32, {kDSpineCandidates, k, b}, op, "candidate_ids");
    require(candidate_scores, DType::FP32, {kDSpineCandidates, k, b}, op, "candidate_scores");
    require(denominator, DType::FP32, {w, b}, op, "denominator");
    require(rms, DType::FP32, {w, b}, op, "rms");
    require(packed, DType::BF16, {3 * rank, w, b}, op, "packed");
    require(message, DType::BF16, {rank, w, b}, op, "message");
    if (vocab <= 0) fail(op, "the tables need rows");
    require(head_write, DType::BF16, {rank, vocab}, op, "head_write");
    require(sender_table, DType::BF16, {2 * rank, vocab}, op, "sender_table");
    require(gate_bias, DType::BF16, {rank}, op, "gate_bias");
    require(anchors, DType::I32, {b}, op, "anchors");
    require(base_positions, DType::I32, {b}, op, "base_positions");
    require(drafts, DType::I32, {k, b}, op, "drafts");
    require(proposal_q, DType::FP32, {kDSpineCandidates, k, b}, op, "proposal_q");
    float* confidence_data = nullptr;
    if (confidence != nullptr) {
        require(*confidence, DType::FP32, {k, b}, op, "confidence");
        require_apart(op, {confidence, &drafts, &proposal_q, &candidate_ids, &candidate_scores,
                           &denominator, &rms});
        confidence_data = static_cast<float*>(confidence->data);
    }
    require_apart(op, {&candidate_ids, &candidate_scores, &denominator, &rms, &packed, &message,
                       &head_write, &sender_table, &gate_bias, &anchors, &base_positions, &drafts,
                       &proposal_q});
    auto scope   = workspace.scope();
    Tensor edges = workspace.alloc(DType::FP32, {kDSpineCandidates, kDSpineCandidates, k, b});
    detail::DSpinePathArgs args{
        .candidate_ids    = static_cast<const std::int32_t*>(candidate_ids.data),
        .candidate_scores = static_cast<const float*>(candidate_scores.data),
        .denominator      = static_cast<const float*>(denominator.data),
        .rms              = static_cast<const float*>(rms.data),
        .packed           = static_cast<const __nv_bfloat16*>(packed.data),
        .message          = static_cast<const __nv_bfloat16*>(message.data),
        .head_write       = static_cast<const __nv_bfloat16*>(head_write.data),
        .sender_table     = static_cast<const __nv_bfloat16*>(sender_table.data),
        .gate_bias        = static_cast<const __nv_bfloat16*>(gate_bias.data),
        .anchors          = static_cast<const std::int32_t*>(anchors.data),
        .base_positions   = static_cast<const std::int32_t*>(base_positions.data),
        .configs          = configs,
        .drafts           = static_cast<std::int32_t*>(drafts.data),
        .proposal_q       = static_cast<float*>(proposal_q.data),
        .confidence       = confidence_data,
        .edges            = static_cast<float*>(edges.data),
        .rank             = rank,
        .width            = w,
        .steps            = k,
        .batch            = b,
    };
    detail::dspine_path_launch(args, stream);
}

} // namespace infernix::ops
