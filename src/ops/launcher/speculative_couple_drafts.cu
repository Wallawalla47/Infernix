// speculative_couple_drafts (include/infernix/ops/speculative_round.h): the drafts of Gumbel-max rows re-drawn
// under the target's verification keys (candidates ZH-01, QK-04).
// Match: contiguous FP32 or BF16 [physical_rows,T] head logits over more than 256 valid rows (the sampler's
// multi-block route), chunks of at most kSpeculativeSamplerMaxColumns columns through one shared workspace.
// Algorithm assumptions: sample()'s partial top-20 tiles and group merges (ops/kernel/sampling.cuh) on raw logits;
// a column whose row is not coupled leaves every kernel at once, so greedy rows cost two empty launches.

#include "ops/launcher/speculative_round.h"

#include "core/device.h"
#include "infernix/ops/speculative_round.h"
#include "ops/kernel/sampling_device.cuh"

#include <algorithm>
#include <type_traits>

namespace infernix::ops::detail {
namespace {

// A row the target verifies by equality with its Gumbel-max draw.
__device__ __forceinline__ bool coupled_row(const SamplingConfig& cfg) {
    return cfg.gumbel_max && cfg.temperature > 0.0F;
}

// One 512-row tile of column column_base + blockIdx.y: its top `cap` keys (sample()'s order: value descending,
// lower row first) into the column's partial slot.
template <class Logit>
__launch_bounds__(kSamplerBlock) __global__
    void couple_partial_kernel(const Logit* logits, const SamplingConfig* configs,
                               std::int32_t column_base, std::int32_t valid_rows, std::int32_t physical_rows,
                               SamplingWorkspace workspace) {
    const int col             = static_cast<int>(blockIdx.y);
    const int partial         = static_cast<int>(blockIdx.x);
    const int t               = column_base + col;
    const SamplingConfig& cfg = configs[t];
    if (!coupled_row(cfg)) { return; }
    if (partial == 0 && threadIdx.x == 0) { workspace.group_done[col] = 0; }

    // The speculative verifier's tile top-k (warp sorts and one merge warp; BF16 logits on 32-bit tile keys).
    __shared__ SamplingPartialTopKStorage topk_storage;
    const int cap           = sampling_candidate_cap(cfg, valid_rows);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;
    if constexpr (std::is_same_v<Logit, __nv_bfloat16>) {
        unsigned int keys[kSamplerItemsPerThread];
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int tile_index = item * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int v          = tile_start + tile_index;
            keys[item]           = v < valid_rows ? sampling_bf16_tile_sort_key(logits[base + v], tile_index) : 0U;
        }
        sampling_store_bf16_tile_topk(keys, cap, tile_start, workspace, col, partial, topk_storage.bf16);
    } else {
        unsigned long long keys[kSamplerItemsPerThread];
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int v = tile_start + item * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            keys[item]  = v < valid_rows ? sampling_sort_key(sampling_logit(logits[base + v]), v) : 0ULL;
        }
        sampling_store_tile_topk(keys, cap, workspace, col, partial, topk_storage.fp32);
    }
}

// Group `blockIdx.x` merges its partials' keys; the column's last group merges the groups', builds the support
// and re-draws the draft.
template <class Logit>
__launch_bounds__(kSamplerGroupBlock) __global__ void couple_finalize_kernel(
    const Logit* logits, const std::int32_t* token_ids, const SamplingConfig* configs,
    const std::int32_t* positions, std::int32_t offset, std::int32_t column_base,
    std::int32_t valid_rows, std::int32_t physical_rows, std::int32_t partial_blocks, std::int32_t group_count,
    std::int32_t* drafts, float* probability, SamplingWorkspace workspace) {
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);
    const int t     = column_base + col;
    const int row   = t;
    SamplingConfig cfg = configs[row];
    if (!coupled_row(cfg)) { return; }
    cfg.nonfinite = nullptr; // a draft's support without a finite value keeps the argmax draft
    __shared__ SamplingTileTopKStorage topk_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ bool finite;
    unsigned long long keys[kSamplerGroupItemsPerThread];
    const int cap = sampling_candidate_cap(cfg, valid_rows);

    const int group_begin = group * kSamplerPartialsPerGroup;
    const int group_partials = min(max(static_cast<int>(partial_blocks) - group_begin, 0), kSamplerPartialsPerGroup);
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * static_cast<int>(blockDim.x) + tid;
        keys[item]  = p < group_n ? workspace.partial_keys[sampling_partial_offset(
                                       workspace, col, group_begin + p / cap, p - (p / cap) * cap)]
                                  : 0ULL;
    }
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        is_last = atomicAdd(&workspace.group_done[col], 1) + 1 == group_count ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }
    __threadfence();

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * static_cast<int>(blockDim.x) + tid;
        keys[item]  = p < final_n ? workspace.partial_keys[sampling_partial_offset(
                                       workspace, col, partial_blocks + p / cap, p - (p / cap) * cap)]
                                  : 0ULL;
    }
    // The last group's slot now holds the column's top `cap` keys in order.
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();
    if (tid < cap) {
        const unsigned long long key =
            workspace.partial_keys[sampling_partial_offset(workspace, col, partial_blocks + group, tid)];
        cand_val[tid] = sampling_key_float(key);
        cand_idx[tid] = sampling_key_index(key);
    }
    __syncthreads();
    if (tid == 0) { finite = isfinite(cand_val[0]); }
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid != 0) { return; }
    workspace.group_done[col] = 0;
    if (!finite) { return; }
    const int position = positions[row] + offset;
    int best           = 0;
    float best_score   = -CUDART_INF_F;
    for (int j = 0; j < n_support; ++j) {
        const int id      = token_ids != nullptr ? token_ids[cand_idx[j]] : cand_idx[j];
        const float score = sampling_gumbel_score(prob[j], cfg.seed, position, id);
        if (score > best_score) {
            best_score = score;
            best       = j;
        }
    }
    const int chosen = cand_idx[best];
    if (probability != nullptr) {
        const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
        probability[t] *= __expf(sampling_logit(logits[base + chosen]) - sampling_logit(logits[base + drafts[t]]));
    }
    drafts[t] = chosen;
}

template <class Logit>
void couple(const Logit* logits, std::int32_t valid_rows, std::int32_t physical_rows, const std::int32_t* token_ids,
            const SamplingConfig* configs, const std::int32_t* positions, std::int32_t offset,
            std::int32_t columns, std::int32_t* drafts, float* probability, DeviceSpan workspace,
            cudaStream_t stream) {
    const std::int32_t chunk          = std::min(columns, kSpeculativeSamplerMaxColumns);
    const SamplingWorkspaceLayout layout =
        make_sampling_workspace_layout(valid_rows, chunk, kSpeculativeSamplerMaxColumns);
    const SamplingWorkspace scratch   = layout.bind(workspace);
    const std::int32_t partial_blocks = div_up(valid_rows, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks);
    // Chunks run in stream order through the one workspace.
    for (std::int32_t base = 0; base < columns; base += chunk) {
        const auto n = static_cast<unsigned int>(std::min(chunk, columns - base));
        couple_partial_kernel<<<dim3(static_cast<unsigned int>(partial_blocks), n), kSamplerBlock, 0, stream>>>(
            logits, configs, base, valid_rows, physical_rows, scratch);
        CUDA_CHECK(cudaGetLastError());
        couple_finalize_kernel<<<dim3(static_cast<unsigned int>(groups), n), kSamplerGroupBlock, 0, stream>>>(
            logits, token_ids, configs, positions, offset, base, valid_rows, physical_rows, partial_blocks,
            groups, drafts, probability, scratch);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

std::size_t speculative_couple_drafts_workspace_exact_bytes(std::int32_t valid_rows, std::int32_t columns) {
    const SamplingWorkspaceLayout layout = make_sampling_workspace_layout(
        valid_rows, std::min(columns, kSpeculativeSamplerMaxColumns), kSpeculativeSamplerMaxColumns);
    // The kernels have only the multiblock form (no single-block fallback): a vocabulary past its group limit
    // would bind an empty workspace and write through null pointers.
    if (!layout.multiblock) {
        throw std::invalid_argument("speculative_couple_drafts: " + std::to_string(valid_rows) +
                                    " valid rows exceed the multiblock sampler's limit");
    }
    return layout.bytes;
}

void speculative_couple_drafts_launch(const Tensor& logits, std::int32_t valid_rows, const std::int32_t* token_ids,
                                      const SamplingConfig* configs, const Tensor& positions,
                                      std::int32_t offset, Tensor& drafts, Tensor& probability, DeviceSpan workspace,
                                      cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t columns       = logits.ne[1];
    const auto* at                   = static_cast<const std::int32_t*>(positions.data);
    auto* out                        = static_cast<std::int32_t*>(drafts.data);
    auto* p                          = static_cast<float*>(probability.data);
    if (logits.dtype == DType::FP32) {
        couple(static_cast<const float*>(logits.data), valid_rows, physical_rows, token_ids, configs, at, offset,
               columns, out, p, workspace, stream);
    } else {
        couple(static_cast<const __nv_bfloat16*>(logits.data), valid_rows, physical_rows, token_ids, configs, at,
               offset, columns, out, p, workspace, stream);
    }
}

} // namespace infernix::ops::detail
