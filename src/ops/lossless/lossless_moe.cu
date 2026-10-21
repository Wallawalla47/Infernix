#include "infernix/ops/lossless_ops.h"

#include "core/device.h"
#include "ops/lossless/cr_math.cuh"
#include "ops/lossless/wide.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

// The --lossless route's sparse MoE pieces (include/infernix/ops/lossless_ops.h), Hugging Face's
// Qwen3_5MoeSparseMoeBlock operation by operation:
//  - router: p = fp32(softmax(l)) per element (the exact softmax of the BF16 logits, correctly rounded:
//    certified in double, otherwise evaluated in double-double), the top k by p (ties: the lower expert
//    first), w = fp32(p / fp32(sum of the top k)) (the sum exact);
//  - routing: each (slot, column) placed in its expert's run of 64-column tiles (placement within a run
//    is arbitrary: every column's results are its own), the gathered activations, padding zero;
//  - combine: per column, the experts in ascending id, s = bf16(s + bf16(fp32(y_e w_e))) from BF16 zero
//    (index_add_ into zeros_like), then bf16(s + bf16(shared bf16(sigmoid(gate)))), then the decoder's
//    residual bf16(residual + that).

namespace infernix::ops {
namespace {

constexpr int kTile        = 64;
constexpr int kMaxExperts  = 1024;
constexpr int kMaxTopK     = 16;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("lossless_moe: ") + message); }
}

__device__ __forceinline__ float bf16f(std::uint16_t b) { return __uint_as_float(static_cast<std::uint32_t>(b) << 16); }

// ----------------------------------------------------------------------------- router

// One warp per column; lane l holds experts l, l + 32, ...
template <int kPerLane>
__global__ void router_kernel(const std::uint16_t* logits, int logit_rows, int experts, int top_k, int columns,
                              bool bf16_weights, float* weights, std::int32_t* ids) {
    const int warp = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp >= columns) { return; }
    const std::uint16_t* l = logits + static_cast<std::size_t>(warp) * logit_rows;
    const int per_lane     = (experts + 31) / 32;
    float x[kPerLane];
    float m = -INFINITY;
    bool nan = false;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
        const int e = lane + 32 * j;
        x[j]        = j < per_lane && e < experts ? bf16f(l[e]) : -INFINITY;
        nan         = nan || isnan(x[j]);
        m           = fmaxf(m, x[j]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
    nan = __any_sync(0xFFFFFFFFU, nan);
    // Double evaluation: e_j = exp(x_j - m) (the difference exact), S in a fixed order.
    double ev[kPerLane];
    double sum = 0.0;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
        ev[j] = j < per_lane && x[j] != -INFINITY ? exp(static_cast<double>(x[j]) - static_cast<double>(m)) : 0.0;
        sum += ev[j];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFFU, sum, o); }
    // Relative error of p = e / S: exp within 1 ulp, S within experts ulp, the division 1/2 ulp.
    const double err = (static_cast<double>(experts) + 4.0) * 0x1p-53;
    float p[kPerLane];
    bool uncertain = false;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
        if (j >= per_lane) {
            p[j] = -1.0F;
            continue;
        }
        const double y = ev[j] / sum;
        if (!lossless::certain_f32(y, y * err, p[j])) { uncertain = true; }
        if (lane + 32 * j >= experts) { p[j] = -1.0F; }
    }
    if (__any_sync(0xFFFFFFFFU, uncertain)) {
        // Double-double: each exp within 2^-94, the sum and quotient a few roundings more.
        lossless::DD dev[kPerLane];
        lossless::DD dsum = {0.0, 0.0};
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            const bool live = j < per_lane && x[j] != -INFINITY;
            dev[j] = live ? lossless::dd_exp(static_cast<double>(x[j]) - static_cast<double>(m)) : lossless::DD{0.0, 0.0};
            dsum   = lossless::dd_add(dsum, dev[j]);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const lossless::DD other = {__shfl_xor_sync(0xFFFFFFFFU, dsum.hi, o), __shfl_xor_sync(0xFFFFFFFFU, dsum.lo, o)};
            // The same pairs in the same order on both lanes of an exchange: add the lower lane's first.
            dsum = (lane & o) != 0 ? lossless::dd_add(other, dsum) : lossless::dd_add(dsum, other);
        }
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            if (j < per_lane && lane + 32 * j < experts) { p[j] = lossless::f32_of_dd(lossless::dd_div(dev[j], dsum)); }
        }
    }
    if (nan) {
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            if (j < per_lane && lane + 32 * j < experts) { p[j] = NAN; }
        }
    }
    // Top k: the largest p, ties to the lower expert (NaN: the lowest experts).
    float chosen_p[kMaxTopK] = {};
    int chosen[kMaxTopK];
    for (int s = 0; s < top_k; ++s) {
        float best = -2.0F;
        int best_e = 0x7FFFFFFF;
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            const int e = lane + 32 * j;
            if (j >= per_lane || e >= experts) { continue; }
            const float v = isnan(p[j]) ? 2.0F : p[j];
            if (v > best || (v == best && e < best_e)) {
                best   = v;
                best_e = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xFFFFFFFFU, best, o);
            const int oe   = __shfl_xor_sync(0xFFFFFFFFU, best_e, o);
            if (ob > best || (ob == best && oe < best_e)) {
                best   = ob;
                best_e = oe;
            }
        }
        chosen[s] = best_e;
        // Remove it from its lane.
#pragma unroll
        for (int j = 0; j < kPerLane; ++j) {
            if (lane + 32 * j == best_e) {
                chosen_p[s] = p[j];
                p[j]        = -1.5F;
            }
        }
        chosen_p[s] = __shfl_sync(0xFFFFFFFFU, chosen_p[s], best_e % 32);
    }
    if (lane != 0) { return; }
    // fp32 of the exact sum of the top k, then each quotient.
    lossless::F32Sum total;
    total.clear();
    for (int s = 0; s < top_k; ++s) { total.add(chosen_p[s]); }
    const float denominator = total.result();
    for (int s = 0; s < top_k; ++s) {
        const float w = __fdiv_rn(chosen_p[s], denominator);
        weights[static_cast<std::size_t>(warp) * top_k + s] = bf16_weights ? __bfloat162float(__float2bfloat16_rn(w)) : w;
        ids[static_cast<std::size_t>(warp) * top_k + s]     = chosen[s];
    }
}

// ----------------------------------------------------------------------------- routing

__global__ void route_count_kernel(const std::int32_t* ids, int entries, std::int32_t* counts) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < entries) { atomicAdd(counts + ids[i], 1); }
}

// One block: each expert's run padded to whole tiles, exclusive scan, the tiles' experts, and each
// expert's cursor (into counts) at its run's start.
__global__ void route_scan_kernel(std::int32_t* counts, int experts, std::int32_t* tile_experts, int tiles) {
    __shared__ std::int32_t scan[kMaxExperts];
    const int e = static_cast<int>(threadIdx.x);
    const int padded = e < experts ? (counts[e] + kTile - 1) / kTile * kTile : 0;
    scan[e] = padded;
    __syncthreads();
    for (int offset = 1; offset < kMaxExperts; offset <<= 1) {
        const int v = e >= offset ? scan[e - offset] : 0;
        __syncthreads();
        scan[e] += v;
        __syncthreads();
    }
    const int start = scan[e] - padded;
    if (e < experts) {
        for (int t = start / kTile; t < (start + padded) / kTile && t < tiles; ++t) { tile_experts[t] = e; }
        counts[e] = start;
    }
}

__global__ void route_place_kernel(const std::int32_t* ids, int entries, std::int32_t* cursor, std::int32_t* slot_column,
                                   std::int32_t* column_token, int top_k) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= entries) { return; }
    const int c     = atomicAdd(cursor + ids[i], 1);
    slot_column[i]  = c;
    column_token[c] = i / top_k;
}

// One block per gathered column (tiles without an expert skipped); padding columns zero.
__global__ void gather_kernel(const std::uint16_t* x, int width, const std::int32_t* column_token,
                              const std::int32_t* tile_experts, std::uint16_t* gathered) {
    const int c = static_cast<int>(blockIdx.x);
    if (tile_experts[c / kTile] < 0) { return; }
    const int t = column_token[c];
    std::uint16_t* dst = gathered + static_cast<std::size_t>(c) * width;
    for (int h = static_cast<int>(threadIdx.x); h < width; h += static_cast<int>(blockDim.x)) {
        dst[h] = t >= 0 ? x[static_cast<std::size_t>(t) * width + h] : 0;
    }
}

// ----------------------------------------------------------------------------- combine

__device__ __forceinline__ std::uint16_t bf16_bits(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }

// One block per column, threads over the hidden width.
__global__ void combine_kernel(const std::uint16_t* expert_out, int width, const float* weights, const std::int32_t* ids,
                               const std::int32_t* slot_column, int top_k, const std::uint16_t* shared_out,
                               const std::uint16_t* shared_logits, int shared_rows, int shared_row, bool add,
                               std::uint16_t* residual) {
    __shared__ int order[kMaxTopK];
    __shared__ float gate;
    const int t = static_cast<int>(blockIdx.x);
    if (threadIdx.x == 0) {
        // The slots by ascending expert id (HF's loop over the experts hit).
        for (int s = 0; s < top_k; ++s) { order[s] = s; }
        for (int a = 1; a < top_k; ++a) {
            const int v = order[a];
            int b       = a - 1;
            while (b >= 0 && ids[static_cast<std::size_t>(t) * top_k + order[b]] > ids[static_cast<std::size_t>(t) * top_k + v]) {
                order[b + 1] = order[b];
                --b;
            }
            order[b + 1] = v;
        }
        gate = shared_out != nullptr
                   ? bf16f(lossless::cr_sigmoid_bf16(bf16f(shared_logits[static_cast<std::size_t>(t) * shared_rows + shared_row])))
                   : 0.0F;
    }
    __syncthreads();
    for (int h = static_cast<int>(threadIdx.x); h < width; h += static_cast<int>(blockDim.x)) {
        float s = 0.0F;
        for (int i = 0; i < top_k; ++i) {
            const int slot = order[i];
            const std::size_t at = static_cast<std::size_t>(t) * top_k + slot;
            const std::size_t column = slot_column != nullptr ? static_cast<std::size_t>(slot_column[at]) : at;
            const float y = bf16f(expert_out[column * width + h]);
            const float term = __bfloat162float(__float2bfloat16_rn(__fmul_rn(y, weights[at])));
            s = bf16f(lossless::bf16_of_double(static_cast<double>(s) + static_cast<double>(term)));
        }
        if (shared_out != nullptr) {
            const float sh = __bfloat162float(
                __float2bfloat16_rn(__fmul_rn(bf16f(shared_out[static_cast<std::size_t>(t) * width + h]), gate)));
            s = bf16f(lossless::bf16_of_double(static_cast<double>(s) + static_cast<double>(sh)));
        }
        std::uint16_t& r = residual[static_cast<std::size_t>(t) * width + h];
        r = add ? lossless::bf16_of_double(static_cast<double>(bf16f(r)) + static_cast<double>(s)) : bf16_bits(s);
    }
}

} // namespace

void lossless_router(const Tensor& logits, int experts, int top_k, Tensor& weights, Tensor& ids, cudaStream_t stream,
                     bool bf16_weights) {
    require(logits.dtype == DType::BF16 && logits.is_contiguous() && experts > 0 && experts <= kMaxExperts &&
                logits.ne[0] >= experts && top_k > 0 && top_k <= kMaxTopK && top_k <= experts,
            "logits BF16 [>= experts, T], experts <= 1024, top_k <= 16");
    const int columns = logits.ne[1];
    require(weights.dtype == DType::FP32 && ids.dtype == DType::I32 && weights.is_contiguous() && ids.is_contiguous() &&
                weights.ne[0] == top_k && weights.ne[1] == columns && ids.ne[0] == top_k && ids.ne[1] == columns,
            "weights FP32 and ids I32 [top_k, T]");
    const int threads = 128;
    const int grid    = (columns * 32 + threads - 1) / threads;
    const auto* l     = static_cast<const std::uint16_t*>(logits.data);
    auto* w           = static_cast<float*>(weights.data);
    auto* i           = static_cast<std::int32_t*>(ids.data);
    if (experts <= 256) {
        router_kernel<8><<<grid, threads, 0, stream>>>(l, logits.ne[0], experts, top_k, columns, bf16_weights, w, i);
    } else if (experts <= 512) {
        router_kernel<16><<<grid, threads, 0, stream>>>(l, logits.ne[0], experts, top_k, columns, bf16_weights, w, i);
    } else {
        router_kernel<32><<<grid, threads, 0, stream>>>(l, logits.ne[0], experts, top_k, columns, bf16_weights, w, i);
    }
    CUDA_CHECK(cudaGetLastError());
}

std::int32_t lossless_moe_columns(std::int32_t columns, std::int32_t top_k, std::int32_t experts) {
    return (columns * top_k + kTile - 1) / kTile * kTile + experts * kTile;
}

void lossless_moe_route(const Tensor& ids, int experts, Tensor& counts, Tensor& tile_experts, Tensor& slot_column,
                        Tensor& column_token, cudaStream_t stream) {
    require(ids.dtype == DType::I32 && ids.is_contiguous() && experts > 0 && experts <= kMaxExperts,
            "ids I32 [top_k, T], experts <= 1024");
    const int top_k = ids.ne[0], columns = ids.ne[1], entries = top_k * columns;
    const int gathered = lossless_moe_columns(columns, top_k, experts);
    require(counts.dtype == DType::I32 && counts.numel() >= experts, "counts I32 [experts]");
    require(tile_experts.dtype == DType::I32 && tile_experts.numel() == gathered / kTile,
            "tile_experts I32 [lossless_moe_columns / 64]");
    require(slot_column.dtype == DType::I32 && slot_column.numel() == entries, "slot_column I32 [top_k, T]");
    require(column_token.dtype == DType::I32 && column_token.numel() == gathered, "column_token I32 [lossless_moe_columns]");
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, static_cast<std::size_t>(experts) * 4, stream));
    CUDA_CHECK(cudaMemsetAsync(tile_experts.data, 0xFF, static_cast<std::size_t>(gathered / kTile) * 4, stream));
    CUDA_CHECK(cudaMemsetAsync(column_token.data, 0xFF, static_cast<std::size_t>(gathered) * 4, stream));
    const auto* id = static_cast<const std::int32_t*>(ids.data);
    auto* cnt      = static_cast<std::int32_t*>(counts.data);
    route_count_kernel<<<(entries + 255) / 256, 256, 0, stream>>>(id, entries, cnt);
    route_scan_kernel<<<1, kMaxExperts, 0, stream>>>(cnt, experts, static_cast<std::int32_t*>(tile_experts.data),
                                                     gathered / kTile);
    route_place_kernel<<<(entries + 255) / 256, 256, 0, stream>>>(id, entries, cnt,
                                                                 static_cast<std::int32_t*>(slot_column.data),
                                                                 static_cast<std::int32_t*>(column_token.data), top_k);
    CUDA_CHECK(cudaGetLastError());
}

void lossless_moe_gather(const Tensor& x, const Tensor& column_token, const Tensor& tile_experts, Tensor& gathered,
                         cudaStream_t stream) {
    require(x.dtype == DType::BF16 && x.is_contiguous() && gathered.dtype == DType::BF16 && gathered.is_contiguous() &&
                gathered.ne[0] == x.ne[0] && gathered.ne[1] == column_token.numel() &&
                tile_experts.numel() * kTile == column_token.numel(),
            "x BF16 [H, T], gathered BF16 [H, C]");
    gather_kernel<<<static_cast<unsigned>(gathered.ne[1]), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(x.data), x.ne[0], static_cast<const std::int32_t*>(column_token.data),
        static_cast<const std::int32_t*>(tile_experts.data), static_cast<std::uint16_t*>(gathered.data));
    CUDA_CHECK(cudaGetLastError());
}

void lossless_moe_combine(const Tensor& expert_out, const Tensor& weights, const Tensor& ids, const Tensor& slot_column,
                          const Tensor* shared_out, const Tensor* shared_logits, int shared_row, Tensor& residual,
                          cudaStream_t stream, bool add) {
    const int width = residual.ne[0], columns = residual.ne[1], top_k = ids.ne[0];
    require(residual.dtype == DType::BF16 && residual.is_contiguous() && expert_out.dtype == DType::BF16 &&
                expert_out.ne[0] == width && weights.ne[1] == columns && ids.ne[1] == columns && top_k <= kMaxTopK,
            "expert_out BF16 [H, C], weights and ids [top_k, T], residual BF16 [H, T]");
    require((shared_out == nullptr) == (shared_logits == nullptr), "the shared expert's output and gate come together");
    if (shared_out != nullptr) {
        require(shared_out->dtype == DType::BF16 && shared_out->ne[0] == width && shared_out->ne[1] == columns &&
                    shared_logits->dtype == DType::BF16 && shared_logits->ne[1] == columns &&
                    shared_row < shared_logits->ne[0],
                "shared_out BF16 [H, T], shared_logits BF16 [R, T]");
    }
    combine_kernel<<<static_cast<unsigned>(columns), 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(expert_out.data), width, static_cast<const float*>(weights.data),
        static_cast<const std::int32_t*>(ids.data), static_cast<const std::int32_t*>(slot_column.data), top_k,
        shared_out != nullptr ? static_cast<const std::uint16_t*>(shared_out->data) : nullptr,
        shared_logits != nullptr ? static_cast<const std::uint16_t*>(shared_logits->data) : nullptr,
        shared_logits != nullptr ? shared_logits->ne[0] : 0, shared_row, add, static_cast<std::uint16_t*>(residual.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
