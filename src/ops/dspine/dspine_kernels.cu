// Ported from NInfer-all (https://github.com/iamwavecut/ninfer-all, PRs #341 and #365; Apache-2.0) and modified
// for Infernix (renamed into Infernix's Op contract and its oracle test, the proposal draw sub-keyed by step
// with the Gumbel-max coupled draw: docs/maintainer/dspine.md).
// The DSpine code bridge and serial refinement kernels (infernix/ops/dspine.h states each formula).
#include "ops/dspine/launch.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kCandidates = 16;
constexpr int kMaxRank    = 512;
constexpr int kVector     = 8; // BF16 values of one 16-byte load

struct alignas(16) Bf16x8 {
    __nv_bfloat162 pair[4];
};

__device__ __forceinline__ void load8(const __nv_bfloat16* source, float (&out)[kVector]) {
    const Bf16x8 vector = *reinterpret_cast<const Bf16x8*>(source);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 value = __bfloat1622float2(vector.pair[i]);
        out[2 * i]         = value.x;
        out[2 * i + 1]     = value.y;
    }
}

__device__ __forceinline__ void store8(__nv_bfloat16* target, const float (&in)[kVector]) {
    Bf16x8 vector;
#pragma unroll
    for (int i = 0; i < 4; ++i) vector.pair[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
    *reinterpret_cast<Bf16x8*>(target) = vector;
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0F / (1.0F + expf(-x)); }

// The block's sum of `value`, visible to every thread.
template <int Threads>
__device__ __forceinline__ float block_total(float value, float* partials) {
    constexpr int Warps = Threads / kWarpSize;
    value               = warp_sum(value);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    if (lane == 0) partials[warp] = value;
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < Warps; ++w) total += partials[w];
    __syncthreads();
    return total;
}

template <int Threads>
__global__ void __launch_bounds__(Threads)
    bridge_read_kernel(const __nv_bfloat16* __restrict__ raw, int hidden, float eps,
                       __nv_bfloat16* __restrict__ normalized, float* __restrict__ rms) {
    __shared__ float partials[Threads / kWarpSize];
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * hidden;
    float squares           = 0.0F;
    for (int h = static_cast<int>(threadIdx.x) * kVector; h < hidden; h += Threads * kVector) {
        float x[kVector];
        load8(raw + base + h, x);
#pragma unroll
        for (int i = 0; i < kVector; ++i) squares = fmaf(x[i], x[i], squares);
    }
    const float total = block_total<Threads>(squares, partials);
    const float value = sqrtf(fmaxf(total / static_cast<float>(hidden), FLT_MIN));
    if (threadIdx.x == 0) rms[blockIdx.x] = value;
    const float denominator = value + eps;
    for (int h = static_cast<int>(threadIdx.x) * kVector; h < hidden; h += Threads * kVector) {
        float x[kVector];
        load8(raw + base + h, x);
#pragma unroll
        for (int i = 0; i < kVector; ++i) x[i] = __fdiv_rn(x[i], denominator);
        store8(normalized + base + h, x);
    }
}

template <int Threads>
__global__ void __launch_bounds__(Threads)
    bridge_codes_kernel(const __nv_bfloat16* projected, int size, float eps,
                        __nv_bfloat16* codes) {
    __shared__ float partials[Threads / kWarpSize];
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * size;
    float squares           = 0.0F;
    for (int c = static_cast<int>(threadIdx.x) * kVector; c < size; c += Threads * kVector) {
        float x[kVector];
        load8(projected + base + c, x);
#pragma unroll
        for (int i = 0; i < kVector; ++i) squares = fmaf(x[i], x[i], squares);
    }
    const float denominator = sqrtf(block_total<Threads>(squares, partials)) + eps;
    const float scale       = sqrtf(static_cast<float>(size));
    for (int c = static_cast<int>(threadIdx.x) * kVector; c < size; c += Threads * kVector) {
        float x[kVector];
        load8(projected + base + c, x);
#pragma unroll
        for (int i = 0; i < kVector; ++i) x[i] = __fdiv_rn(scale * x[i], denominator);
        store8(codes + base + c, x);
    }
}

__global__ void bridge_message_kernel(const __nv_bfloat16* __restrict__ packed,
                                      const std::int32_t* __restrict__ anchors,
                                      const __nv_bfloat16* __restrict__ sender_table,
                                      const __nv_bfloat16* __restrict__ gate_bias, int rank,
                                      int width, __nv_bfloat16* __restrict__ message) {
    const int column = static_cast<int>(blockIdx.x);
    const int b      = static_cast<int>(blockIdx.y);
    __nv_bfloat16* out =
        message + (static_cast<std::int64_t>(b) * width + column) * rank;
    if (column == 0) {
        for (int r = static_cast<int>(threadIdx.x); r < rank; r += blockDim.x)
            out[r] = __float2bfloat16_rn(0.0F);
        return;
    }
    const __nv_bfloat16* receiver =
        packed + (static_cast<std::int64_t>(b) * width + column) * 3 * rank;
    const __nv_bfloat16* sender =
        column == 1 ? sender_table + static_cast<std::int64_t>(anchors[b]) * 2 * rank
                    : packed + (static_cast<std::int64_t>(b) * width + column - 1) * 3 * rank + rank;
    for (int r = static_cast<int>(threadIdx.x); r < rank; r += blockDim.x) {
        const float q    = __bfloat162float(receiver[r]);
        const float s    = __bfloat162float(sender[r]);
        const float v    = __bfloat162float(sender[rank + r]);
        const float gate = sigmoid((q + s) + __bfloat162float(gate_bias[r]));
        out[r]           = __float2bfloat16_rn(gate * v);
    }
}

template <int Threads>
__global__ void __launch_bounds__(Threads)
    bridge_finish_kernel(const __nv_bfloat16* raw, const float* __restrict__ rms,
                         const __nv_bfloat16* __restrict__ delta,
                         const __nv_bfloat16* __restrict__ norm_weight, float eps, int hidden,
                         int width, __nv_bfloat16* out, __nv_bfloat16* __restrict__ normed,
                         float* __restrict__ denominator) {
    __shared__ float partials[Threads / kWarpSize];
    const int column        = static_cast<int>(blockIdx.x);
    const bool anchor       = column % width == 0;
    const float scale       = rms[column];
    const std::int64_t base = static_cast<std::int64_t>(column) * hidden;
    float squares           = 0.0F;
    // Each thread rereads only the values it wrote, so out may alias raw.
    for (int h = static_cast<int>(threadIdx.x) * kVector; h < hidden; h += Threads * kVector) {
        float x[kVector];
        load8(raw + base + h, x);
        if (!anchor) {
            float d[kVector];
            load8(delta + base + h, d);
#pragma unroll
            for (int i = 0; i < kVector; ++i) x[i] = x[i] + scale * d[i];
        }
#pragma unroll
        for (int i = 0; i < kVector; ++i) {
            x[i]    = __bfloat162float(__float2bfloat16_rn(x[i])); // the stored value
            squares = fmaf(x[i], x[i], squares);
        }
        store8(out + base + h, x);
    }
    const float mean  = block_total<Threads>(squares, partials) / static_cast<float>(hidden);
    const float root  = sqrtf(mean + eps);
    const float scale_out = 1.0F / root;
    if (denominator != nullptr && threadIdx.x == 0) denominator[column] = root;
    for (int h = static_cast<int>(threadIdx.x) * kVector; h < hidden; h += Threads * kVector) {
        float x[kVector], w[kVector];
        load8(out + base + h, x);
        load8(norm_weight + h, w);
#pragma unroll
        for (int i = 0; i < kVector; ++i) {
            const float y = __bfloat162float(__float2bfloat16_rn(x[i] * scale_out));
            x[i]          = y * w[i];
        }
        store8(normed + base + h, x);
    }
}

__global__ void scale_rows_kernel(const __nv_bfloat16* rows, const __nv_bfloat16* weight,
                                  int hidden, std::int64_t vectors, __nv_bfloat16* out) {
    const int per_row = hidden / kVector;
    for (std::int64_t v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         v < vectors; v += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const int h = static_cast<int>(v % per_row) * kVector;
        float x[kVector], w[kVector];
        load8(rows + v * kVector, x);
        load8(weight + h, w);
#pragma unroll
        for (int i = 0; i < kVector; ++i) x[i] = x[i] * w[i];
        store8(out + v * kVector, x);
    }
}

// Column j of logits ([vocab, columns], row v of column j at j * vocab + v) is head-write column
// first + j: head_write[v * rank + first + j].
template <typename Logit>
__global__ void store_head_write_kernel(const Logit* __restrict__ logits, int vocab, int columns,
                                        int rank, int first, __nv_bfloat16* __restrict__ head_write) {
    const std::int64_t count = static_cast<std::int64_t>(vocab) * columns;
    for (std::int64_t e = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         e < count; e += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t v = e / columns;
        const int j          = static_cast<int>(e % columns);
        float value;
        if constexpr (sizeof(Logit) == 4) {
            value = logits[static_cast<std::int64_t>(j) * vocab + v];
        } else {
            value = __bfloat162float(logits[static_cast<std::int64_t>(j) * vocab + v]);
        }
        head_write[v * rank + first + j] = __float2bfloat16_rn(value);
    }
}

// One CTA per (draft row i, block b), one warp per candidate (base) and per sender (edges).
constexpr int kLatticeThreads = kCandidates * kWarpSize;

__global__ void __launch_bounds__(kLatticeThreads) path_lattice_kernel(DSpinePathArgs a) {
    extern __shared__ __align__(16) unsigned char lattice_shared[];
    auto* head_write = reinterpret_cast<__nv_bfloat16*>(lattice_shared); // [16][rank]
    __shared__ float base[kCandidates];
    __shared__ std::int32_t candidates[kCandidates], senders[kCandidates];

    const int i = static_cast<int>(blockIdx.x), b = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const std::int64_t cell     = static_cast<std::int64_t>(b) * a.width + i + 1;
    const std::int64_t row_slot = static_cast<std::int64_t>(b) * a.steps + i;
    if (threadIdx.x < kCandidates) {
        candidates[threadIdx.x] = a.candidate_ids[row_slot * kCandidates + threadIdx.x];
        senders[threadIdx.x] =
            i == 0 ? a.anchors[b] : a.candidate_ids[(row_slot - 1) * kCandidates + threadIdx.x];
    }
    __syncthreads();
    const float scale = a.rms[cell];

    // Warp c: its candidate's head-write row into shared memory, then its base, the edge the
    // block's own message gives minus that message's term.
    {
        const int c       = warp;
        const auto* table = a.head_write + static_cast<std::int64_t>(candidates[c]) * a.rank;
        const auto* own   = a.message + cell * a.rank;
        float sum         = 0.0F;
        for (int r = lane * kVector; r < a.rank; r += kWarpSize * kVector) {
            *reinterpret_cast<Bf16x8*>(&head_write[c * a.rank + r]) =
                *reinterpret_cast<const Bf16x8*>(table + r);
            float t[kVector], m[kVector];
            load8(table + r, t);
            load8(own + r, m);
#pragma unroll
            for (int k = 0; k < kVector; ++k) sum = fmaf(m[k], t[k], sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            const float logit = a.candidate_scores[row_slot * kCandidates + c];
            base[c]           = logit == -CUDART_INF_F
                                    ? -CUDART_INF_F
                                    : fmaf(a.denominator[cell], logit, -scale * sum);
        }
    }
    __syncthreads();

    // Warp j: the message of sender j (FP32, rank / 32 values a lane), then its 16 edges.
    const int j               = warp;
    const std::int64_t sender = senders[j];
    const std::int64_t q_base = cell * 3 * a.rank;
    float message[kMaxRank / kWarpSize];
    const int per_lane = a.rank / kWarpSize;
#pragma unroll
    for (int k = 0; k < kMaxRank / kWarpSize; ++k) {
        if (k >= per_lane) break;
        const int r   = lane + k * kWarpSize;
        const float q = __bfloat162float(a.packed[q_base + r]);
        const float s = __bfloat162float(a.sender_table[sender * 2 * a.rank + r]);
        const float v = __bfloat162float(a.sender_table[sender * 2 * a.rank + a.rank + r]);
        const float g = sigmoid((q + s) + __bfloat162float(a.gate_bias[r]));
        message[k]    = g * v;
    }
    float mine = 0.0F;
    for (int c = 0; c < kCandidates; ++c) {
        float partial = 0.0F;
#pragma unroll
        for (int k = 0; k < kMaxRank / kWarpSize; ++k) {
            if (k >= per_lane) break;
            partial = fmaf(message[k],
                           __bfloat162float(head_write[c * a.rank + lane + k * kWarpSize]),
                           partial);
        }
        partial = warp_sum(partial);
        if (lane == c) mine = base[c] == -CUDART_INF_F ? -CUDART_INF_F : fmaf(scale, partial, base[c]);
    }
    if (lane < kCandidates)
        a.edges[(row_slot * kCandidates + j) * kCandidates + lane] = mine;
}

// One warp per block walks its K rows: the sender of row i is the slot row i - 1 selected.
__global__ void path_walk_kernel(DSpinePathArgs a) {
    const int b    = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const bool slot = lane < kCandidates;
    const SamplingConfig config = a.configs[b];
    const int top_k = config.top_k > 0 && config.top_k < kCandidates ? config.top_k : kCandidates;
    int sender      = 0; // row 0: every sender slot is the anchor
    for (int i = 0; i < a.steps; ++i) {
        const std::int64_t row_slot = static_cast<std::int64_t>(b) * a.steps + i;
        float edge = slot ? a.edges[(row_slot * kCandidates + sender) * kCandidates + lane]
                          : -CUDART_INF_F;
        if (!(edge == edge)) edge = -CUDART_INF_F; // NaN takes no part
        const int id = slot ? a.candidate_ids[row_slot * kCandidates + lane] : INT_MAX;
        const float scale   = a.denominator[static_cast<std::int64_t>(b) * a.width + i + 1];
        const float maximum = warp_max(edge);
        int selected        = 0;
        float q             = 0.0F;
        if (config.temperature <= 0.0F || maximum == -CUDART_INF_F) {
            // The greatest edge, the lowest id on a tie.
            int key = slot && edge == maximum ? id : INT_MAX;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                key = min(key, __shfl_xor_sync(kFullWarpMask, key, offset));
            const unsigned winner = __ballot_sync(kFullWarpMask, slot && edge == maximum && id == key);
            selected              = winner == 0 ? 0 : __ffs(winner) - 1;
            q                     = lane == selected ? 1.0F : 0.0F;
        } else {
            const float temperature = config.temperature;
            const float logit_max   = maximum / scale / temperature;
            const float weight =
                slot && edge > -CUDART_INF_F ? expf(edge / scale / temperature - logit_max) : 0.0F;
            const float p = weight / warp_sum(weight);
            // Rank by p, the lower slot first on a tie; the mass ranked before.
            int rank       = 0;
            float before   = 0.0F;
            float p_max    = 0.0F;
            for (int d = 0; d < kCandidates; ++d) {
                const float other = __shfl_sync(kFullWarpMask, p, d);
                p_max             = fmaxf(p_max, other);
                if (other > p || (other == p && d < lane)) {
                    ++rank;
                    before += other;
                }
            }
            // The most probable candidate stays whatever the filters say.
            bool keep = rank < top_k && (config.top_p >= 1.0F || before <= config.top_p);
            if (config.min_p > 0.0F) keep = keep && p >= config.min_p * p_max;
            keep = slot && p > 0.0F && (keep || rank == 0);
            const float kept = keep ? p : 0.0F;
            q                = kept / warp_sum(kept);
            float cumulative = slot ? q : 0.0F;
#pragma unroll
            for (int offset = 1; offset < kWarpSize; offset *= 2) {
                const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
                if (lane >= offset) cumulative += previous;
            }
            const float total = __shfl_sync(kFullWarpMask, cumulative, kCandidates - 1);
            if (config.gumbel_max) {
                // A Gumbel-max row's coupled draw (DFlash2's coupled_rank): the kept candidate whose token scores
                // highest under the noise its verification gives that token at the produced position (base + i +
                // 1), the lower slot on a tie, so the draft is the target's own draw whenever the drafter keeps
                // that token. A draw from q all the same (Gumbel-max over its support).
                const float score = keep && q > 0.0F
                                        ? sampling_gumbel_score(q, config.seed, a.base_positions[b] + i + 1, id)
                                        : -CUDART_INF_F;
                const float best       = warp_max(score);
                const unsigned winners = __ballot_sync(kFullWarpMask, keep && q > 0.0F && score == best);
                selected               = winners == 0 ? 0 : __ffs(winners) - 1;
            } else {
                // Sub-key i, as DFlash2's (speculative_round.h): a later round's draft at the same position
                // has a smaller step, so it never reuses this uniform.
                float uniform = lane == 0 ? sampling_uniform(config.seed, a.base_positions[b] + i,
                                                             kSamplePurposeDSpineProposal, static_cast<unsigned>(i))
                                          : 0.0F;
                uniform                   = __shfl_sync(kFullWarpMask, uniform, 0);
                const float threshold     = uniform * total;
                const unsigned below      = __ballot_sync(kFullWarpMask, slot && cumulative <= threshold);
                const unsigned kept_slots = __ballot_sync(kFullWarpMask, keep);
                const int last            = kept_slots == 0 ? 0 : 31 - __clz(kept_slots);
                selected                  = min(__popc(below), last);
            }
        }
        if (slot) a.proposal_q[row_slot * kCandidates + lane] = q;
        const int drawn = __shfl_sync(kFullWarpMask, id, selected);
        if (lane == 0) a.drafts[row_slot] = drawn;
        if (a.confidence != nullptr) {
            const float weight =
                slot && edge > -CUDART_INF_F && maximum > -CUDART_INF_F
                    ? expf((edge - maximum) / scale)
                    : (slot && maximum == -CUDART_INF_F ? 1.0F : 0.0F);
            const float total = warp_sum(weight);
            const float picked = __shfl_sync(kFullWarpMask, weight, selected);
            if (lane == 0) a.confidence[row_slot] = picked / total;
        }
        sender = selected;
    }
}

} // namespace

void dspine_bridge_read_launch(const __nv_bfloat16* raw, int hidden, int columns, float eps,
                               __nv_bfloat16* normalized, float* rms, cudaStream_t stream) {
    constexpr int Threads = 256;
    bridge_read_kernel<Threads><<<columns, Threads, 0, stream>>>(raw, hidden, eps, normalized, rms);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_bridge_codes_launch(const __nv_bfloat16* projected, int codes_size, int columns,
                                float eps, __nv_bfloat16* codes, cudaStream_t stream) {
    constexpr int Threads = 128;
    bridge_codes_kernel<Threads>
        <<<columns, Threads, 0, stream>>>(projected, codes_size, eps, codes);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_bridge_message_launch(const __nv_bfloat16* packed, const std::int32_t* anchors,
                                  const __nv_bfloat16* sender_table,
                                  const __nv_bfloat16* gate_bias, int rank, int width, int batch,
                                  __nv_bfloat16* message, cudaStream_t stream) {
    bridge_message_kernel<<<dim3(width, batch), 128, 0, stream>>>(packed, anchors, sender_table,
                                                                 gate_bias, rank, width, message);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_bridge_finish_launch(const __nv_bfloat16* raw, const float* rms,
                                 const __nv_bfloat16* delta, const __nv_bfloat16* norm_weight,
                                 float eps, int hidden, int width, int batch, __nv_bfloat16* out,
                                 __nv_bfloat16* normed, float* denominator, cudaStream_t stream) {
    constexpr int Threads = 256;
    bridge_finish_kernel<Threads><<<width * batch, Threads, 0, stream>>>(
        raw, rms, delta, norm_weight, eps, hidden, width, out, normed, denominator);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_scale_rows_launch(const __nv_bfloat16* rows, const __nv_bfloat16* weight, int hidden,
                              std::int64_t columns, __nv_bfloat16* out, cudaStream_t stream) {
    const std::int64_t vectors = columns * (hidden / kVector);
    const int blocks = static_cast<int>(std::min<std::int64_t>((vectors + 255) / 256, 4096));
    scale_rows_kernel<<<blocks, 256, 0, stream>>>(rows, weight, hidden, vectors, out);
    CUDA_CHECK(cudaGetLastError());
}

void dspine_store_head_write_launch(const void* logits, bool fp32_logits, int vocab, int columns,
                                    int rank, int first, __nv_bfloat16* head_write,
                                    cudaStream_t stream) {
    const std::int64_t count = static_cast<std::int64_t>(vocab) * columns;
    const int blocks = static_cast<int>(std::min<std::int64_t>((count + 255) / 256, 8192));
    if (fp32_logits) {
        store_head_write_kernel<float><<<blocks, 256, 0, stream>>>(
            static_cast<const float*>(logits), vocab, columns, rank, first, head_write);
    } else {
        store_head_write_kernel<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits), vocab, columns, rank, first, head_write);
    }
    CUDA_CHECK(cudaGetLastError());
}

void dspine_path_launch(const DSpinePathArgs& args, cudaStream_t stream) {
    const std::size_t shared =
        static_cast<std::size_t>(kCandidates) * args.rank * sizeof(__nv_bfloat16);
    path_lattice_kernel<<<dim3(args.steps, args.batch), kLatticeThreads, shared, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
    path_walk_kernel<<<args.batch, kWarpSize, 0, stream>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
