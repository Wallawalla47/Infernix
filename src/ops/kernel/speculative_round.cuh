#pragma once
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/speculative_tree.h"
#include "core/pdl.cuh"

// Implements: include/ninfer/ops/speculative_round.h
// Match: contiguous request-major state and BF16 verification logits.
// Algorithm assumptions: small vocabularies use one cooperative block; the
// registered full-vocabulary stochastic route uses the sampling partial/group
// pipeline and caller-owned workspace. Sparse acceptance and emission use one warp per request.

#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {


__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* anchors,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* base_positions,
                                                         const std::int32_t* current_extents,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t k) {
    pdl::enter();
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        const int off = row * T + j;
        verify_ids[off] =
            j == 0 ? anchors[row] : (j <= extent ? drafts[row * k + j - 1] : anchors[row]);
        if (positions != nullptr) {
            positions[off] = base_positions[row] + (j <= extent ? j : extent);
        }
    }
}

// One block per row. Rows whose device flag is zero keep the proposal already in place.
__global__ void speculative_overlay_copy_proposals_kernel(
    const std::int32_t* copy_rows, const std::int32_t* copy_drafts,
    const std::int32_t* copy_candidates, const float* copy_q, std::int32_t* drafts,
    std::int32_t* candidates, float* proposal_q, std::int32_t k, std::int32_t slots) {
    pdl::enter();
    const int row = static_cast<int>(blockIdx.x);
    if (copy_rows[row] == 0) { return; }
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        drafts[row * k + j] = copy_drafts[row * k + j];
    }
    if (candidates == nullptr) { return; }
    const int plane = k * slots;
    for (int i = threadIdx.x; i < plane; i += blockDim.x) {
        candidates[row * plane + i] = copy_candidates[row * plane + i];
        proposal_q[row * plane + i] = copy_q[row * plane + i];
    }
}

template <typename T>
__device__ inline T* speculative_workspace_offset(T* ptr, std::size_t byte_offset) {
    return ptr == nullptr
               ? nullptr
               : reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

__device__ inline SamplingWorkspace
speculative_workspace_row(SamplingWorkspace workspace, std::size_t row_stride, std::int32_t row) {
    const std::size_t offset = static_cast<std::size_t>(row) * row_stride;
    workspace.partial_keys   = speculative_workspace_offset(workspace.partial_keys, offset);
    workspace.dist_idx       = speculative_workspace_offset(workspace.dist_idx, offset);
    workspace.dist_prob      = speculative_workspace_offset(workspace.dist_prob, offset);
    workspace.dist_support   = speculative_workspace_offset(workspace.dist_support, offset);
    workspace.group_done     = speculative_workspace_offset(workspace.group_done, offset);
    workspace.speculative_finalize_count =
        speculative_workspace_offset(workspace.speculative_finalize_count, offset);
    return workspace;
}

template <bool UpdateTokenCounts>
__device__ __forceinline__ void
speculative_store_accept_result(const std::int32_t* row_drafts, std::int32_t k, std::int32_t row,
                                std::int32_t accepted_count, std::int32_t terminal_token,
                                std::int32_t* lengths, std::int32_t* anchors,
                                std::int32_t* row_tokens, std::int32_t* licensed_counts,
                                std::int32_t* accepted, const SamplingConfig* config) {
    for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
    for (int i = 0; i < accepted_count; ++i) { row_tokens[i] = row_drafts[i]; }
    row_tokens[accepted_count] = terminal_token;

    const int produced   = accepted_count + 1;
    licensed_counts[row] = produced;
    accepted[row]        = accepted_count;
    anchors[row]         = terminal_token;
    lengths[row] += produced;
    if constexpr (UpdateTokenCounts) {
        if (config->token_counts != nullptr) {
            for (int i = 0; i < produced; ++i) {
                atomicAdd(&config->token_counts[row_tokens[i]], 1);
            }
        }
    }
}

__device__ __forceinline__ float speculative_sparse_probability(const std::int32_t* candidate_ids,
                                                                const float* proposal_q,
                                                                std::int32_t token) {
    float probability = 0.0f;
#pragma unroll
    for (int candidate = 0; candidate < kSparseSpeculativeCandidates; ++candidate) {
        if (candidate_ids[candidate] == token) {
            probability = proposal_q[candidate];
            break;
        }
    }
    return probability;
}

// One warp owns a request. Each draft's acceptance event is independent given the provided
// path and its conditional p/q distributions; the first failed event determines the prefix.
__device__ __forceinline__ void speculative_sparse_warp_store(const int* drafts, int k, int row,
                                                              int accepted_count, int terminal,
                                                              int* lengths, int* anchors,
                                                              int* licensed_tokens,
                                                              int* licensed_counts, int* accepted) {
    const int lane = threadIdx.x & 31;
    for (int column = lane; column <= k; column += 32)
        licensed_tokens[row * (k + 1) + column] = column < accepted_count ? drafts[row * k + column]
                                                  : column == accepted_count ? terminal
                                                                             : 0;
    if (lane == 0) {
        licensed_counts[row] = accepted_count + 1;
        accepted[row]        = accepted_count;
        anchors[row]         = terminal;
        lengths[row] += accepted_count + 1;
    }
}

__device__ __forceinline__ void speculative_sparse_warp_greedy(const int* target_tokens,
                                                               const int* drafts, int* lengths,
                                                               int* anchors, int* licensed_tokens,
                                                               int* licensed_counts, int* accepted,
                                                               int row, int extent, int k) {
    const int lane = threadIdx.x & 31;
    int a          = extent;
    for (int base = 0; base < k; base += 32) {
        const int column = base + lane;
        const bool reject =
            column < extent && target_tokens[row * (k + 1) + column] != drafts[row * k + column];
        const unsigned mask = __ballot_sync(0xffffffffU, reject);
        if (mask) a = min(a, base + __ffs(mask) - 1);
    }
    const int terminal = target_tokens[row * (k + 1) + a];
    speculative_sparse_warp_store(drafts, k, row, a, terminal, lengths, anchors, licensed_tokens,
                                  licensed_counts, accepted);
}

__global__ __launch_bounds__(256) void speculative_accept_sparse_warp_greedy_kernel(
    const int* target_tokens, const int* drafts, const int* current_extents, int* lengths,
    int* anchors, int* licensed_tokens, int* licensed_counts, int* accepted, int k) {
    const int row    = threadIdx.x / 32;
    const int extent = min(k, max(0, current_extents[row]));
    speculative_sparse_warp_greedy(target_tokens, drafts, lengths, anchors, licensed_tokens,
                                   licensed_counts, accepted, row, extent, k);
}

__device__ __forceinline__ void speculative_sparse_warp_accept(
    SamplingWorkspace workspace, const int* drafts, const int* candidate_ids,
    const float* proposal_q, const SamplingConfig& cfg, int k, int row, int extent, bool greedy,
    int* lengths, int* anchors, int* licensed_tokens, int* licensed_counts, int* accepted) {
    const int lane       = threadIdx.x & 31;
    const int old_length = lengths[row];
    int a                = extent;
    // Keep the same per-position RNG keys and one-warp residual CDF at every width.
    for (int base = 0; base < k; base += 32) {
        const int column = base + lane;
        bool reject      = false;
        if (column < extent) {
            const int d = drafts[row * k + column];
            if (greedy)
                reject = workspace.dist_idx[sampling_dist_offset(column, 0)] != d;
            else {
                const int support = workspace.dist_support[column];
                float pd          = 0.0f;
                for (int j = 0; j < support; ++j) {
                    const int at = sampling_dist_offset(column, j);
                    if (workspace.dist_idx[at] == d) {
                        pd = workspace.dist_prob[at];
                        break;
                    }
                }
                const int at = (row * k + column) * kSparseSpeculativeCandidates;
                const float qd =
                    speculative_sparse_probability(candidate_ids + at, proposal_q + at, d);
                const float u = sampling_uniform(cfg.seed, old_length + column + 1,
                                                 kSamplePurposeSpeculativeAccept, 0);
                reject        = !(pd >= qd || u * qd < pd);
            }
        }
        const unsigned failures = __ballot_sync(0xffffffffU, reject);
        if (failures) a = min(a, base + __ffs(failures) - 1);
    }
    int terminal;
    if (greedy)
        terminal = workspace.dist_idx[sampling_dist_offset(a, 0)];
    else {
        const int n  = workspace.dist_support[a];
        int token    = 0;
        float weight = 0.0f;
        if (lane < n) {
            const int at = sampling_dist_offset(a, lane);
            token        = workspace.dist_idx[at];
            weight       = workspace.dist_prob[at];
            if (a < extent) {
                const int q_at = (row * k + a) * kSparseSpeculativeCandidates;
                weight         = fmaxf(weight - speculative_sparse_probability(candidate_ids + q_at,
                                                                               proposal_q + q_at, token),
                                       0.0f);
            }
        }
        float cdf = weight;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const float earlier = __shfl_up_sync(0xffffffffU, cdf, offset);
            if (lane >= offset) cdf += earlier;
        }
        const float mass = __shfl_sync(0xffffffffU, cdf, 31);
        const int purpose =
            a < extent ? kSamplePurposeSpeculativeCorrection : kSamplePurposeSpeculativeBonus;
        const float u           = sampling_uniform(cfg.seed, old_length + a + 1, purpose, 0);
        const float goal        = u * mass;
        const unsigned positive = __ballot_sync(0xffffffffU, lane < n && weight > 0.0f);
        const unsigned candidates =
            __ballot_sync(0xffffffffU, lane < n && weight > 0.0f && goal < cdf);
        const int selected = !(mass > 0.0f) ? 0
                             : candidates   ? __ffs(candidates) - 1
                             : positive     ? 31 - __clz(positive)
                                            : 0;
        terminal           = __shfl_sync(0xffffffffU, token, selected);
    }
    speculative_sparse_warp_store(drafts, k, row, a, terminal, lengths, anchors, licensed_tokens,
                                  licensed_counts, accepted);
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the target length. The greedy branch
// (config temperature <= 0) is bit-identical to the original argmax accept: keep
// the longest draft prefix whose target argmax matches, then take the target
// argmax at the divergence column. The sampling branch (temperature > 0) runs
// distribution-correct speculative rejection sampling over the verify logits with
// a one-hot (greedy) draft: accept drafts[i] with probability p_i(drafts[i]) under
// the truncated target distribution, resample from the masked residual on the
// first rejection, and draw a bonus from the last column when every draft accepts.
// The draft-proposal path stays greedy, so q is one-hot and the accept test
// collapses to `u < p_i(drafts[i])`. Launch with a single block of kSamplerBlock
// threads; only thread 0 performs the sequential accept/commit while the whole
// block cooperates on the per-column truncated-distribution build.
__launch_bounds__(kSamplerBlock) __global__ void speculative_accept_greedy_drafts_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t k) {
    const int tid                   = threadIdx.x;
    const int row                   = static_cast<int>(blockIdx.x);
    const int cols                  = k + 1;
    int extent                      = current_extents[row];
    extent                          = extent < 0 ? 0 : (extent > k ? k : extent);
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const __nv_bfloat16* row_logits =
        logits + static_cast<std::int64_t>(row) * cols * physical_rows;
    const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;

    if (!(cfg.temperature > 0.0f) && !penalties) {
        if (tid == 0) {
            int a = 0;
            while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
            const int t_star = row_targets[a];
            speculative_store_accept_result<true>(row_drafts, k, row, a, t_star, lengths, anchors,
                                                  row_tokens, licensed_counts, accepted, &cfg);
        }
        return;
    }

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int n_support;
    __shared__ int a_sh;
    __shared__ int done_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // No-op when the scratch/group path owns this shape.
    if (sampler_multiblock_ok(token_domain, cols, partial_blocks, group_count,
                              kSpeculativeSamplerMaxColumns)) {
        return;
    }

    if (tid == 0) {
        a_sh     = 0;
        done_sh  = 0;
        tstar_sh = 0;
        L_sh     = lengths[row];
    }
    __syncthreads();

    if (!(cfg.temperature > 0.0f)) {
        for (int i = 0; i <= extent; ++i) {
            const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
            float best_value        = -CUDART_INF_F;
            int best_index          = INT_MAX;
            for (int v = tid; v < token_domain; v += blockDim.x) {
                const float value = sampling_adjusted_logit(__bfloat162float(row_logits[base + v]),
                                                            v, cfg, row_drafts, i);
                if (sampling_better(value, v, best_value, best_index)) {
                    best_value = value;
                    best_index = v;
                }
            }
            red_val[tid] = best_value;
            red_idx[tid] = best_index;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride && sampling_better(red_val[tid + stride], red_idx[tid + stride],
                                                    red_val[tid], red_idx[tid])) {
                    red_val[tid] = red_val[tid + stride];
                    red_idx[tid] = red_idx[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                const int selected = red_idx[0];
                if (i < extent && selected == row_drafts[i]) {
                    a_sh = i + 1;
                } else {
                    tstar_sh = selected;
                    done_sh  = 1;
                }
            }
            __syncthreads();
            if (done_sh) { break; }
        }

        if (tid == 0) {
            const int a     = a_sh;
            const int tstar = tstar_sh;
            speculative_store_accept_result<true>(row_drafts, k, row, a, tstar, lengths, anchors,
                                                  row_tokens, licensed_counts, accepted, &cfg);
        }
        return;
    }

    for (int i = 0; i <= extent; ++i) {
        // Column i is only reached when drafts[0..i-1] were all accepted, so the
        // round-local penalty overlay for this column is exactly those i drafts.
        const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
        if (token_domain <= kSamplerTileItems) {
            sampling_build_truncated_small(row_logits, base, token_domain, cfg, red_val, red_idx,
                                           cand_val, cand_idx, prob, &n_support, row_drafts, i);
        } else {
            sampling_build_truncated_block_fast(row_logits, base, token_domain, cfg, merge_val,
                                                merge_idx, cand_val, cand_idx, prob, &n_support,
                                                row_drafts, i);
        }
        if (tid == 0 && done_sh == 0) {
            const int L = L_sh;
            if (i < extent) {
                const int d = row_drafts[i];
                float pd    = 0.0f;
                for (int j = 0; j < n_support; ++j) {
                    if (cand_idx[j] == d) {
                        pd = prob[j];
                        break;
                    }
                }
                const float u =
                    sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                if (u < pd) {
                    a_sh = i + 1; // accept drafts[i], keep verifying
                } else {
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar_sh       = sampling_pick_from_support(cand_idx, prob, n_support, d, ur);
                    done_sh        = 1;
                }
            } else {
                // Every draft accepted: bonus token from the last verify column.
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, -1, u);
                done_sh  = 1;
            }
        }
        __syncthreads();
        if (done_sh) { break; }
    }

    if (tid == 0) {
        const int a     = a_sh;
        const int tstar = tstar_sh;
        speculative_store_accept_result<true>(row_drafts, k, row, a, tstar, lengths, anchors,
                                              row_tokens, licensed_counts, accepted, &cfg);
    }
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, SamplingWorkspace workspace,
    std::size_t workspace_row_stride) {
    pdl::enter();
    const int row     = static_cast<int>(blockIdx.z);
    const int col     = static_cast<int>(blockIdx.y);
    const int partial = static_cast<int>(blockIdx.x);
    int extent        = current_extents[row];
    extent            = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg = configs[row];
    const bool greedy        = !(cfg.temperature > 0.0f);
    const bool penalties     = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    if ((greedy && !penalties) || token_domain <= kSamplerTileItems) { return; }
    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ SamplingPartialTopKStorage topk_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];

    const int cap                  = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base        = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
    const std::int32_t* row_drafts = drafts + row * k;
    const int tile_start           = partial * kSamplerPartialTileItems;
    if (!penalties) {
        unsigned int keys[kSamplerItemsPerThread];
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int tile_index = item * blockDim.x + threadIdx.x;
            const int v          = tile_start + tile_index;
            keys[item] =
                v < token_domain ? sampling_bf16_tile_sort_key(logits[base + v], tile_index) : 0u;
        }
        sampling_store_bf16_tile_topk(keys, cap, tile_start, workspace, col, partial,
                                      topk_storage.bf16);
        return;
    }

    unsigned long long keys[kSamplerItemsPerThread];
    // Column col's penalty overlay is the first `col` drafts (see accept loop);
    // applying it before top-k selection lets it change the candidate set, not
    // just the post-truncation probabilities.
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const __nv_bfloat16 raw = logits[base + v];
            keys[item]              = sampling_sort_key(
                sampling_adjusted_logit(__bfloat162float(raw), v, cfg, row_drafts, col), v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    sampling_store_tile_topk(keys, cap, workspace, col, partial, topk_storage.fp32);
}

template <bool SparseProposal>
__launch_bounds__(kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts,
    const std::int32_t* candidate_ids, const float* proposal_q, const std::int32_t* current_extents,
    std::int32_t* lengths, std::int32_t* anchors, std::int32_t* licensed_tokens,
    std::int32_t* licensed_counts, std::int32_t* accepted, const SamplingConfig* configs,
    std::int32_t token_domain, std::int32_t cols, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace, std::size_t workspace_row_stride) {
    pdl::enter();
    const int row   = static_cast<int>(blockIdx.z);
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
    const int k     = cols - 1;
    int extent      = current_extents[row];
    extent          = extent < 0 ? 0 : (extent > k ? k : extent);
    if (col > extent) { return; }
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    if (token_domain <= kSamplerTileItems) { return; }
    const bool greedy    = !(cfg.temperature > 0.0f);
    const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;

    if (greedy && !penalties) {
        if constexpr (SparseProposal) {
            if (tid < 32 && col == 0 && group == 0)
                speculative_sparse_warp_greedy(target_tokens, drafts, lengths, anchors,
                                               licensed_tokens, licensed_counts, accepted, row,
                                               extent, k);
        } else {

            if (tid == 0 && col == 0 && group == 0) {
                int a = 0;
                while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
                const int t_star = row_targets[a];
                speculative_store_accept_result<!SparseProposal>(row_drafts, k, row, a, t_star,
                                                                 lengths, anchors, row_tokens,
                                                                 licensed_counts, accepted, &cfg);
            }
        }
        return;
    }


    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);

    __shared__ SamplingTileTopKStorage topk_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    // The preceding partial launch initializes all caller-owned counters. CUDA
    // stream ordering makes those writes visible before this launch begins.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last_group  = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();

    if (tid < cap) {
        const int off = sampling_partial_offset(workspace, col, partial_blocks + group, tid);
        const unsigned long long key = workspace.partial_keys[off];
        cand_val[tid]                = sampling_key_float(key);
        cand_idx[tid]                = sampling_key_index(key);
    }
    __syncthreads();

    if constexpr (SparseProposal) {
        __shared__ int last_column;
        if (greedy) {
            if (tid == 0) {
                workspace.dist_idx[sampling_dist_offset(col, 0)] = cand_idx[0];
                workspace.group_done[col]                        = 0;
                __threadfence();
                last_column = atomicAdd(workspace.speculative_finalize_count, 1) + 1 == extent + 1;
            }
            __syncthreads();
            if (last_column && tid < 32)
                speculative_sparse_warp_accept(workspace, drafts, candidate_ids, proposal_q, cfg, k,
                                               row, extent, true, lengths, anchors, licensed_tokens,
                                               licensed_counts, accepted);
            if (last_column && tid == 0) *workspace.speculative_finalize_count = 0;
            return;
        }
        sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
        if (tid == 0) {
            workspace.dist_support[col] = n_support;
            for (int j = 0; j < n_support; ++j) {
                const int at            = sampling_dist_offset(col, j);
                workspace.dist_idx[at]  = cand_idx[j];
                workspace.dist_prob[at] = prob[j];
            }
            workspace.group_done[col] = 0;
            __threadfence();
            last_column = atomicAdd(workspace.speculative_finalize_count, 1) + 1 == extent + 1;
        }
        __syncthreads();
        if (last_column && tid < 32)
            speculative_sparse_warp_accept(workspace, drafts, candidate_ids, proposal_q, cfg, k,
                                           row, extent, false, lengths, anchors, licensed_tokens,
                                           licensed_counts, accepted);
        if (last_column && tid == 0) *workspace.speculative_finalize_count = 0;
        return;
    } else {

        if (greedy) {
            if (tid == 0) {
                workspace.dist_idx[sampling_dist_offset(col, 0)] = cand_idx[0];
                workspace.group_done[col]                        = 0;
                __threadfence();
                const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
                if (done_cols == extent + 1) {
                    int a     = 0;
                    int tstar = 0;
                    for (int i = 0; i <= extent; ++i) {
                        const int selected = workspace.dist_idx[sampling_dist_offset(i, 0)];
                        if (i < extent && selected == row_drafts[i]) {
                            a = i + 1;
                            continue;
                        }
                        tstar = selected;
                        break;
                    }
                    speculative_store_accept_result<!SparseProposal>(
                        row_drafts, k, row, a, tstar, lengths, anchors, row_tokens, licensed_counts,
                        accepted, &cfg);
                    *workspace.speculative_finalize_count = 0;
                }
            }
            return;
        }

        sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);

        if (tid == 0) {
            workspace.dist_support[col] = n_support;
            for (int j = 0; j < n_support; ++j) {
                const int off            = sampling_dist_offset(col, j);
                workspace.dist_idx[off]  = cand_idx[j];
                workspace.dist_prob[off] = prob[j];
            }
            workspace.group_done[col] = 0;
            __threadfence();
            const int done_cols = atomicAdd(workspace.speculative_finalize_count, 1) + 1;
            if (done_cols == extent + 1) {
                const int L = lengths[row];
                int a       = 0;
                int tstar   = 0;
                for (int i = 0; i <= extent; ++i) {
                    const int n            = workspace.dist_support[i];
                    const int* dist_idx    = workspace.dist_idx + sampling_dist_offset(i, 0);
                    const float* dist_prob = workspace.dist_prob + sampling_dist_offset(i, 0);
                    if (i < extent) {
                        const int d = row_drafts[i];
                        float pd    = 0.0f;
                        for (int j = 0; j < n; ++j) {
                            if (dist_idx[j] == d) {
                                pd = dist_prob[j];
                                break;
                            }
                        }
                        const float u           = sampling_uniform(cfg.seed, L + i + 1,
                                                                   kSamplePurposeSpeculativeAccept, 0u);
                        const bool accept_draft = u < pd;
                        if (accept_draft) {
                            a = i + 1;
                            continue;
                        }
                        const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                          kSamplePurposeSpeculativeCorrection, 0u);

                        tstar = sampling_pick_from_support(dist_idx, dist_prob, n, d, ur);

                        break;
                    }
                    const float u = sampling_uniform(cfg.seed, L + extent + 1,
                                                     kSamplePurposeSpeculativeBonus, 0u);
                    tstar         = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
                }
                speculative_store_accept_result<!SparseProposal>(row_drafts, k, row, a, tstar,
                                                                 lengths, anchors, row_tokens,
                                                                 licensed_counts, accepted, &cfg);
                *workspace.speculative_finalize_count = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Speculative verification tree (ninfer/ops/speculative_tree.h). Every row carries the topology
// its builder wrote this round. A chain row (tree == 0) is verified as the chain prefix of its
// first extent+1 columns; the drafts stride is the round width W - 1 for every row.

__device__ __forceinline__ int speculative_tree_live_columns(const SpeculativeTreeRow& tree,
                                                             int extent, int k) {
    if (tree.tree != 0) { return tree.nodes; }
    extent = extent < 0 ? 0 : (extent > k ? k : extent);
    return extent + 1;
}

// Penalty overlay of a column: the drafts of its ancestors and itself, excluding the anchor. A
// chain prefix is its own ancestry, so a chain column gets the chain overlay drafts[0..column-1]
// (the count is order-free).
__device__ __forceinline__ int speculative_tree_overlay(const SpeculativeTreeRow& tree,
                                                        const std::int32_t* row_drafts, int column,
                                                        std::int32_t* overlay) {
    int n = 0;
    if (tree.tree == 0) {
        for (int a = 0; a < column; ++a) { overlay[n++] = row_drafts[a]; }
        return n;
    }
    for (int a = column; a > 0; a = tree.parent[a]) { overlay[n++] = row_drafts[a - 1]; }
    return n;
}

// Inverse-CDF draw over one warp-resident support (lane < n), identical to the chain's
// correction/bonus draw.
__device__ __forceinline__ int speculative_tree_pick(int token, float weight, int n, float u) {
    const int lane = threadIdx.x & 31;
    float cdf      = weight;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        const float earlier = __shfl_up_sync(0xffffffffU, cdf, offset);
        if (lane >= offset) cdf += earlier;
    }
    const float mass          = __shfl_sync(0xffffffffU, cdf, 31);
    const float goal          = u * mass;
    const unsigned positive   = __ballot_sync(0xffffffffU, lane < n && weight > 0.0f);
    const unsigned candidates = __ballot_sync(0xffffffffU, lane < n && weight > 0.0f && goal < cdf);
    const int selected        = !(mass > 0.0f) ? 0
                                : candidates   ? __ffs(candidates) - 1
                                : positive     ? 31 - __clz(positive)
                                               : 0;
    return __shfl_sync(0xffffffffU, token, selected);
}

// Stores the licensed prefix along the accepted path held one column per lane (path[i] in lane
// i, -1 beyond A), the terminal token and the row counters.
__device__ __forceinline__ void
speculative_tree_warp_store(const int* drafts, int row, int columns, int accepted_count,
                            int path_reg, int terminal, int* lengths, int* anchors,
                            int* licensed_tokens, int* licensed_counts, int* accepted,
                            int* accepted_path, int* accepted_branch) {
    const int lane = threadIdx.x & 31;
    const int k    = columns - 1;
    const int next = __shfl_down_sync(0xffffffffU, path_reg, 1);
    for (int base = 0; base < columns; base += 32) {
        const int i = base + lane;
        if (i >= columns) continue;
        const int c                        = base == 0 ? next : -1;
        licensed_tokens[row * columns + i] = i < accepted_count    ? drafts[row * k + c - 1]
                                             : i == accepted_count ? terminal
                                                                   : 0;
        accepted_path[row * columns + i]   = i <= accepted_count && base == 0 ? path_reg : -1;
    }
    // The path leaves the main chain at its first column c_i != i (path[0] is the anchor, so i>=1).
    const unsigned side = __ballot_sync(0xffffffffU, lane <= accepted_count && path_reg != lane);
    if (lane == 0) {
        licensed_counts[row] = accepted_count + 1;
        accepted[row]        = accepted_count;
        anchors[row]         = terminal;
        lengths[row] += accepted_count + 1;
        accepted_branch[row] = side != 0U ? __ffs(static_cast<int>(side)) - 1 : 0;
    }
}

// Greedy tree walk: accept the child equal to the column's target token, else emit it.
template <class TargetAt>
__device__ __forceinline__ void
speculative_tree_warp_greedy(const SpeculativeTreeRow& tree, TargetAt target_at, const int* drafts,
                             int row, int columns, int* lengths, int* anchors, int* licensed_tokens,
                             int* licensed_counts, int* accepted, int* accepted_path,
                             int* accepted_branch) {
    const int lane = threadIdx.x & 31;
    const int k    = columns - 1;
    int v = 0, a = 0, path_reg = lane == 0 ? 0 : -1, terminal = 0;
    for (;;) {
        const int target = target_at(v);
        int next         = -1;
        for (int s = tree.first_child[v]; s >= 0; s = tree.next_sibling[s]) {
            if (drafts[row * k + s - 1] == target) {
                next = s;
                break;
            }
        }
        if (next < 0) {
            terminal = target;
            break;
        }
        v = next;
        ++a;
        if (lane == a) path_reg = next;
    }
    speculative_tree_warp_store(drafts, row, columns, a, path_reg, terminal, lengths, anchors,
                                licensed_tokens, licensed_counts, accepted, accepted_path,
                                accepted_branch);
}

// Recursive rejection over the tree (see speculative_accept_sparse_tree). Weights stay one warp
// wide over the processed target support of the current column; the first sibling's test, the
// single-child residual and every draw reproduce the chain acceptance bit for bit.
__device__ __forceinline__ void
speculative_tree_warp_accept(SamplingWorkspace workspace, const SpeculativeTreeRow& tree,
                             const int* drafts, const int* candidate_ids, const float* proposal_q,
                             const SamplingConfig& cfg, int row, int columns, int* lengths,
                             int* anchors, int* licensed_tokens, int* licensed_counts,
                             int* accepted, int* accepted_path, int* accepted_branch) {
    const int lane       = threadIdx.x & 31;
    const int k          = columns - 1;
    const int old_length = lengths[row];
    int v = 0, a = 0, path_reg = lane == 0 ? 0 : -1, terminal = 0;
    for (;;) {
        const int n = workspace.dist_support[v];
        int token   = 0;
        float w     = 0.0f;
        if (lane < n) {
            const int at = sampling_dist_offset(v, lane);
            token        = workspace.dist_idx[at];
            w            = workspace.dist_prob[at];
        }
        const int first = tree.first_child[v];
        if (first < 0) {
            const float u = sampling_uniform(cfg.seed, old_length + tree.depth[v] + 1,
                                             kSamplePurposeSpeculativeBonus, 0);
            terminal      = speculative_tree_pick(token, w, n, u);
            break;
        }
        // Every sibling of v stores the same full q_v and candidates.
        const int q_at = (row * k + first - 1) * kSparseSpeculativeCandidates;
        const int cand = lane < kSparseSpeculativeCandidates ? candidate_ids[q_at + lane] : -1;
        const float qv = lane < kSparseSpeculativeCandidates ? proposal_q[q_at + lane] : 0.0f;
        // q_v of this lane's support token (zero outside the candidates).
        float q_token  = 0.0f;
        int token_rank = -1;
#pragma unroll
        for (int c = 0; c < kSparseSpeculativeCandidates; ++c) {
            const int id  = __shfl_sync(0xffffffffU, cand, c);
            const float q = __shfl_sync(0xffffffffU, qv, c);
            if (lane < n && id == token && token_rank < 0) {
                q_token    = q;
                token_rank = c;
            }
        }
        float mass       = 1.0f;
        float taken_mass = 0.0f;
        unsigned taken   = 0U;
        bool advanced    = false;
        int j            = 0;
        for (int s = first; s >= 0; s = tree.next_sibling[s], ++j) {
            const int x = drafts[row * k + s - 1];
            const unsigned x_lane =
                __ballot_sync(0xffffffffU, lane < kSparseSpeculativeCandidates && cand == x);
            const float qx        = x_lane ? __shfl_sync(0xffffffffU, qv, __ffs(x_lane) - 1) : 0.0f;
            const unsigned w_lane = __ballot_sync(0xffffffffU, lane < n && token == x);
            const float wx        = w_lane ? __shfl_sync(0xffffffffU, w, __ffs(w_lane) - 1) : 0.0f;
            const float keep      = 1.0f - taken_mass;
            const float Qx        = j == 0 ? qx : (keep > 0.0f ? qx / keep : 0.0f);
            const float rx        = j == 0 ? wx : (mass > 0.0f ? wx / mass : 0.0f);
            const float u =
                sampling_uniform(cfg.seed, old_length + tree.depth[s],
                                 kSamplePurposeSpeculativeAccept, static_cast<std::uint32_t>(j));
            // The chain test: a zero-probability proposal is always accepted.
            const bool accept_child = rx >= Qx || u * Qx < rx;
            if (accept_child) {
                v = s;
                ++a;
                if (lane == a) path_reg = s;
                advanced = true;
                break;
            }
            // Residual of a rejected sibling: max(r - Q_j, 0), kept unnormalized with its mass.
            if (Qx > 0.0f) {
                const bool untaken = token_rank >= 0 && ((taken >> token_rank) & 1U) == 0U;
                const float q_j    = !untaken ? 0.0f : (j == 0 ? q_token : q_token / keep);
                if (lane < n) w = fmaxf(w - mass * q_j, 0.0f);
                mass = warp_sum(lane < n ? w : 0.0f);
            }
            if (x_lane) taken |= 1U << (__ffs(x_lane) - 1);
            taken_mass += qx;
        }
        if (advanced) continue;
        const float u = sampling_uniform(cfg.seed, old_length + tree.depth[v] + 1,
                                         kSamplePurposeSpeculativeCorrection, 0);
        terminal      = speculative_tree_pick(token, w, n, u);
        break;
    }
    speculative_tree_warp_store(drafts, row, columns, a, path_reg, terminal, lengths, anchors,
                                licensed_tokens, licensed_counts, accepted, accepted_path,
                                accepted_branch);
}

// A chain row of a tree round: the chain acceptance above, then its main-chain path.
__device__ __forceinline__ void speculative_tree_chain_path(int row, int columns,
                                                            const int* accepted, int* accepted_path,
                                                            int* accepted_branch) {
    const int lane = threadIdx.x & 31;
    __syncwarp();
    const int count = accepted[row];
    for (int i = lane; i < columns; i += 32) accepted_path[row * columns + i] = i <= count ? i : -1;
    if (lane == 0) accepted_branch[row] = -1;
}

__global__ __launch_bounds__(256) void speculative_accept_tree_warp_greedy_kernel(
    const int* target_tokens, const int* drafts, const int* current_extents, int* lengths,
    int* anchors, int* licensed_tokens, int* licensed_counts, int* accepted, int* accepted_path,
    int* accepted_branch, const SpeculativeTreeRow* trees, int columns) {
    const int row                  = threadIdx.x / 32;
    const SpeculativeTreeRow& tree = trees[row];
    if (tree.tree != 0) {
        speculative_tree_warp_greedy(
            tree, [&](int column) { return target_tokens[row * columns + column]; }, drafts, row,
            columns, lengths, anchors, licensed_tokens, licensed_counts, accepted, accepted_path,
            accepted_branch);
        return;
    }
    const int chain = min(columns - 1, max(0, current_extents[row]));
    speculative_sparse_warp_greedy(target_tokens, drafts, lengths, anchors, licensed_tokens,
                                   licensed_counts, accepted, row, chain, columns - 1);
    speculative_tree_chain_path(row, columns, accepted, accepted_path, accepted_branch);
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_tree_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    SamplingWorkspace workspace, std::size_t workspace_row_stride, const SpeculativeTreeRow* trees,
    int cols) {
    pdl::enter();
    const int row                  = static_cast<int>(blockIdx.z);
    const int col                  = static_cast<int>(blockIdx.y);
    const int partial              = static_cast<int>(blockIdx.x);
    const int k                    = cols - 1;
    const SpeculativeTreeRow& tree = trees[row];
    if (col >= speculative_tree_live_columns(tree, current_extents[row], k)) { return; }
    const SamplingConfig cfg = configs[row];
    const bool greedy        = !(cfg.temperature > 0.0f);
    const bool penalties     = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;
    if ((greedy && !penalties) || token_domain <= kSamplerTileItems) { return; }
    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ SamplingPartialTopKStorage topk_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];
    __shared__ std::int32_t overlay[kSpeculativeTreeMaxPathLength];
    __shared__ int overlay_len;

    const int cap           = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    const std::int64_t base = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;
    if (!penalties) {
        unsigned int keys[kSamplerItemsPerThread];
#pragma unroll
        for (int item = 0; item < kSamplerItemsPerThread; ++item) {
            const int tile_index = item * blockDim.x + threadIdx.x;
            const int v          = tile_start + tile_index;
            keys[item] =
                v < token_domain ? sampling_bf16_tile_sort_key(logits[base + v], tile_index) : 0u;
        }
        sampling_store_bf16_tile_topk(keys, cap, tile_start, workspace, col, partial,
                                      topk_storage.bf16);
        return;
    }

    if (threadIdx.x == 0)
        overlay_len = speculative_tree_overlay(tree, drafts + row * k, col, overlay);
    __syncthreads();
    unsigned long long keys[kSamplerItemsPerThread];
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain) {
            const __nv_bfloat16 raw = logits[base + v];
            keys[item]              = sampling_sort_key(
                sampling_adjusted_logit(__bfloat162float(raw), v, cfg, overlay, overlay_len), v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    sampling_store_tile_topk(keys, cap, workspace, col, partial, topk_storage.fp32);
}

__launch_bounds__(kSamplerGroupBlock) __global__
    void speculative_tree_sampling_group_finalize_kernel(
        const std::int32_t* target_tokens, const std::int32_t* drafts,
        const std::int32_t* candidate_ids, const float* proposal_q,
        const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
        std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
        std::int32_t* accepted_path, std::int32_t* accepted_branch, const SamplingConfig* configs,
        std::int32_t token_domain, std::int32_t partial_blocks, std::int32_t group_count,
        SamplingWorkspace workspace, std::size_t workspace_row_stride,
        const SpeculativeTreeRow* trees, int cols) {
    pdl::enter();
    const int row                  = static_cast<int>(blockIdx.z);
    const int group                = static_cast<int>(blockIdx.x);
    const int col                  = static_cast<int>(blockIdx.y);
    const int tid                  = threadIdx.x;
    const int k                    = cols - 1;
    const SpeculativeTreeRow& tree = trees[row];
    const int extent               = current_extents[row];
    const bool tree_row            = tree.tree != 0;
    const int live                 = speculative_tree_live_columns(tree, extent, k);
    const int chain                = live - 1;
    if (col >= live) { return; }
    const SamplingConfig cfg = configs[row];
    if (token_domain <= kSamplerTileItems) { return; }
    const bool greedy    = !(cfg.temperature > 0.0f);
    const bool penalties = cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f;

    if (greedy && !penalties) {
        if (tid < 32 && col == 0 && group == 0) {
            if (tree_row) {
                speculative_tree_warp_greedy(
                    tree, [&](int column) { return target_tokens[row * cols + column]; }, drafts,
                    row, cols, lengths, anchors, licensed_tokens, licensed_counts, accepted,
                    accepted_path, accepted_branch);
            } else {
                speculative_sparse_warp_greedy(target_tokens, drafts, lengths, anchors,
                                               licensed_tokens, licensed_counts, accepted, row,
                                               chain, k);
                speculative_tree_chain_path(row, cols, accepted, accepted_path, accepted_branch);
            }
        }
        return;
    }

    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);

    __shared__ SamplingTileTopKStorage topk_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    __shared__ int last_column;
    unsigned long long keys[kSamplerGroupItemsPerThread];

    const int cap = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }
    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        const int done = atomicAdd(&workspace.group_done[col], 1) + 1;
        is_last_group  = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    sampling_store_tile_topk(keys, cap, workspace, col, partial_blocks + group, topk_storage);
    __syncthreads();

    if (tid < cap) {
        const int off = sampling_partial_offset(workspace, col, partial_blocks + group, tid);
        const unsigned long long key = workspace.partial_keys[off];
        cand_val[tid]                = sampling_key_float(key);
        cand_idx[tid]                = sampling_key_index(key);
    }
    __syncthreads();

    if (greedy) {
        if (tid == 0) workspace.dist_idx[sampling_dist_offset(col, 0)] = cand_idx[0];
    } else {
        sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
        if (tid == 0) {
            workspace.dist_support[col] = n_support;
            for (int j = 0; j < n_support; ++j) {
                const int at            = sampling_dist_offset(col, j);
                workspace.dist_idx[at]  = cand_idx[j];
                workspace.dist_prob[at] = prob[j];
            }
        }
    }
    if (tid == 0) {
        workspace.group_done[col] = 0;
        __threadfence();
        last_column = atomicAdd(workspace.speculative_finalize_count, 1) + 1 == live;
    }
    __syncthreads();
    if (last_column && tid < 32) {
        if (tree_row && greedy) {
            speculative_tree_warp_greedy(
                tree,
                [&](int column) { return workspace.dist_idx[sampling_dist_offset(column, 0)]; },
                drafts, row, cols, lengths, anchors, licensed_tokens, licensed_counts, accepted,
                accepted_path, accepted_branch);
        } else if (tree_row) {
            speculative_tree_warp_accept(workspace, tree, drafts, candidate_ids, proposal_q, cfg,
                                         row, cols, lengths, anchors, licensed_tokens,
                                         licensed_counts, accepted, accepted_path, accepted_branch);
        } else {
            speculative_sparse_warp_accept(workspace, drafts, candidate_ids, proposal_q, cfg, k,
                                           row, chain, greedy, lengths, anchors, licensed_tokens,
                                           licensed_counts, accepted);
            speculative_tree_chain_path(row, cols, accepted, accepted_path, accepted_branch);
        }
    }
    if (last_column && tid == 0) *workspace.speculative_finalize_count = 0;
}

// One warp per row. Tree columns take the KV slot F+c and the RoPE position of their depth;
// columns past a row's live nodes repeat its last slot and the anchor's RoPE position. A chain
// row keeps the host's chain RoPE positions.
__global__ void speculative_prepare_tree_verify_inputs_kernel(
    const std::int32_t* anchors, const std::int32_t* drafts, const std::int32_t* base_positions,
    const SpeculativeTreeRow* trees, std::int32_t* verify_ids, std::int32_t* positions,
    std::int32_t* rope_positions, int columns) {
    pdl::enter();
    const int row                  = static_cast<int>(blockIdx.x);
    const int k                    = columns - 1;
    const SpeculativeTreeRow& tree = trees[row];
    const int nodes                = tree.nodes;
    const int base                 = base_positions[row];
    const int rope                 = rope_positions[row * columns];
    __syncwarp();
    for (int j = static_cast<int>(threadIdx.x); j < columns; j += 32) {
        const bool live               = j < nodes;
        verify_ids[row * columns + j] = j == 0 || !live ? anchors[row] : drafts[row * k + j - 1];
        positions[row * columns + j]  = base + (live ? j : nodes - 1);
        if (tree.tree != 0) rope_positions[row * columns + j] = rope + (live ? tree.depth[j] : 0);
    }
}

__global__ void speculative_select_accepted_hidden_kernel(const __nv_bfloat16* hidden,
                                                          const std::int32_t* selectors,
                                                          __nv_bfloat16* out, std::int32_t rows,
                                                          std::int32_t cols) {
    pdl::enter();
    const int batch = static_cast<int>(blockIdx.y);
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = selectors[batch];
    if (col < 0 || col >= cols) { return; }
    out[static_cast<std::int64_t>(batch) * rows + row] =
        hidden[(static_cast<std::int64_t>(batch) * cols + col) * rows + row];
}

__global__ void proposal_remap_token_ids_kernel(std::int32_t* proposal_tokens,
                                                std::int32_t proposal_count,
                                                const std::int32_t* id_map, std::int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= proposal_count) { return; }
    const int idx = proposal_tokens[i];
    if (idx >= 0 && idx < n) { proposal_tokens[i] = id_map[idx]; }
}

} // namespace ninfer::ops
