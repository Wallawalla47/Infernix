#pragma once

// Ported from NInfer-all (https://github.com/iamwavecut/ninfer-all, PRs #341 and #365; Apache-2.0) and modified
// for Infernix (renamed into Infernix's Op contract and its oracle test, the proposal draw sub-keyed by step
// with the Gumbel-max coupled draw: docs/maintainer/dspine.md).

#include "infernix/ops/sampling.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

void dspine_bridge_read_launch(const __nv_bfloat16* raw, int hidden, int columns, float eps,
                               __nv_bfloat16* normalized, float* rms, cudaStream_t stream);

void dspine_bridge_codes_launch(const __nv_bfloat16* projected, int codes_size, int columns,
                                float eps, __nv_bfloat16* codes, cudaStream_t stream);

void dspine_bridge_message_launch(const __nv_bfloat16* packed, const std::int32_t* anchors,
                                  const __nv_bfloat16* sender_table,
                                  const __nv_bfloat16* gate_bias, int rank, int width, int batch,
                                  __nv_bfloat16* message, cudaStream_t stream);

void dspine_bridge_finish_launch(const __nv_bfloat16* raw, const float* rms,
                                 const __nv_bfloat16* delta, const __nv_bfloat16* norm_weight,
                                 float eps, int hidden, int width, int batch, __nv_bfloat16* out,
                                 __nv_bfloat16* normed, float* denominator, cudaStream_t stream);

void dspine_scale_rows_launch(const __nv_bfloat16* rows, const __nv_bfloat16* weight, int hidden,
                              std::int64_t columns, __nv_bfloat16* out, cudaStream_t stream);

void dspine_store_head_write_launch(const void* logits, bool fp32_logits, int vocab, int columns,
                                    int rank, int first, __nv_bfloat16* head_write,
                                    cudaStream_t stream);

struct DSpinePathArgs {
    const std::int32_t* candidate_ids;
    const float* candidate_scores;
    const float* denominator;
    const float* rms;
    const __nv_bfloat16* packed;
    const __nv_bfloat16* message;
    const __nv_bfloat16* head_write;
    const __nv_bfloat16* sender_table;
    const __nv_bfloat16* gate_bias;
    const std::int32_t* anchors;
    const std::int32_t* base_positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* proposal_q;
    float* confidence; // null: none
    float* edges;      // workspace FP32 [16 (candidate), 16 (sender), K, B]
    int rank, width, steps, batch;
};

void dspine_path_launch(const DSpinePathArgs& args, cudaStream_t stream);

} // namespace infernix::ops::detail
