// speculative_sample_sparse_drafts (include/infernix/ops/speculative_round.h): one thread per column.

#include "ops/launcher/speculative_round.h"

#include "core/device.h"
#include "infernix/ops/speculative_round.h"
#include "ops/kernel/sampling_device.cuh"

namespace infernix::ops::detail {
namespace {

constexpr int kCandidates = kSparseSpeculativeCandidates;

__global__ void speculative_sample_sparse_drafts_kernel(const std::int32_t* rows, const float* logprobs,
                                                        const std::int32_t* token_ids,
                                                        const SamplingConfig* configs,
                                                        const std::int32_t* positions, unsigned int draw, int columns, int count,
                                                        std::int32_t* drafts, std::int32_t* candidates,
                                                        float* proposal_q, float* confidence) {
    const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= columns) { return; }
    const SamplingConfig cfg = configs[t];
    const std::int32_t* row  = rows + t * count;
    const float* lp          = logprobs + t * count;
    std::int32_t* cand       = candidates + t * kCandidates;
    float* q                 = proposal_q + t * kCandidates;
    for (int j = 0; j < count; ++j) {
        cand[j] = token_ids != nullptr ? token_ids[row[j]] : row[j];
        q[j]    = 0.0F;
    }
    // Unused slots: the smallest ids not already candidates.
    int next = 0;
    for (int j = count; j < kCandidates; ++j) {
        bool taken = true;
        while (taken) {
            taken = false;
            for (int i = 0; i < j; ++i) { taken |= cand[i] == next; }
            if (taken) { ++next; }
        }
        cand[j] = next++;
        q[j]    = 0.0F;
    }
    if (confidence != nullptr) { confidence[t] = lp[0]; }
    if (!(cfg.temperature > 0.0F)) {
        q[0]      = 1.0F;
        drafts[t] = cand[0];
        return;
    }
    const int ranks = cfg.top_k > 0 && cfg.top_k < count ? cfg.top_k : count;
    // The weights are recomputed per pass (at most 16 each) rather than held in a local array.
    const float inverse = 1.0F / cfg.temperature;
    const auto weight   = [&](int j) { return __expf((lp[j] - lp[0]) * inverse); };
    float total         = 0.0F;
    for (int j = 0; j < ranks; ++j) { total += weight(j); }
    const float floor  = cfg.min_p > 0.0F ? cfg.min_p : -1.0F; // weight(0) == 1
    const float target = cfg.top_p * total;
    float mass         = 0.0F;
    int support        = 0;
    for (int j = 0; j < ranks; ++j) {
        const float w = weight(j);
        if (w < floor) { break; }
        mass += w;
        support = j + 1;
        if (cfg.top_p < 1.0F && mass >= target) { break; }
    }
    if (support < 1) {
        support = 1;
        mass    = weight(0);
    }
    for (int j = 0; j < support; ++j) { q[j] = weight(j) / mass; }
    const float u = sampling_uniform(cfg.seed, positions[t], kSamplePurposeDFlash2Proposal, draw);
    // A draw past the rounded cumulative mass falls back to the last candidate with q > 0: a tail
    // weight that underflowed has q = 0 and is never drawn (verification would reject it).
    int pick         = 0;
    float cumulative = 0.0F;
    for (int j = 0; j < support; ++j) {
        if (!(q[j] > 0.0F)) { continue; }
        cumulative += q[j];
        pick = j;
        if (u < cumulative) { break; }
    }
    drafts[t] = cand[pick];
}

} // namespace

void speculative_sample_sparse_drafts_launch(const Tensor& rows, const Tensor& logprobs, const std::int32_t* token_ids,
                                             const SamplingConfig* configs, const Tensor& positions, std::uint32_t draw, Tensor& drafts,
                                             Tensor& candidate_ids, Tensor& proposal_q, Tensor& confidence,
                                             cudaStream_t stream) {
    const int columns = static_cast<int>(drafts.numel());
    const int count   = rows.ne[0];
    speculative_sample_sparse_drafts_kernel<<<1, 32, 0, stream>>>(
        static_cast<const std::int32_t*>(rows.data), static_cast<const float*>(logprobs.data), token_ids, configs,
        static_cast<const std::int32_t*>(positions.data), draw, columns, count, static_cast<std::int32_t*>(drafts.data),
        static_cast<std::int32_t*>(candidate_ids.data), static_cast<float*>(proposal_q.data),
        static_cast<float*>(confidence.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
