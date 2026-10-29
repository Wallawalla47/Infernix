// Producers that write a GGUF projection's input encoded for ggml_linear_encoded (include/infernix/ops/ggml_encode.h).
// A cluster of K / 2,048 CTAs takes a column (blockIdx.y): each computes its 2,048 of the column's BF16 outputs into
// shared memory with its Op's arithmetic (rmsnorm_warp_bf16x2_kernel's gated rows, one a warp; sigmoid_gate_mul_pair's
// packs), the cluster takes the column exponent from every CTA's largest key (distributed shared memory), and each
// CTA encodes a 2,048-element share of the column's chunks as ggml_linear's encode_kernel does
// (ops/ggml/ggml_encode.cuh), reading each chunk's permuted source from the CTA that holds it. So the projection reads
// the same words without the encoding launch. (One CTA a column, every row of the column in it, was slower than the
// separate kernels: its rows and its encode wait in one CTA, and the column's slowest CTA sets the projection's start;
// design §19.3.38.)

#include "infernix/ops/ggml_encode.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/ggml/ggml_encode.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/kernel/sigmoid_gate_mul_pair.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

namespace cg = cooperative_groups;

constexpr int kThreads      = 512;
constexpr int kWarps        = kThreads / 32;
constexpr int kPartElements = 2048; // a CTA's share of the column
constexpr int kMaxParts     = 8;    // the portable cluster size
static_assert(kPartElements == kWarps * 128 && kGgmlEncodeMaxElements <= kMaxParts * kPartElements);

// The cluster's encode of column blockIdx.y once every CTA has its part in shared memory and its largest |bits| in
// `part_key`: the exponent from every part's key (rank 0 writes it to emax), then this CTA's chunks, each read from
// the part that holds its source (perm: 16-byte chunks, perm_block a multiple of 8).
__device__ __forceinline__ void encode_cluster_share(cg::cluster_group& cluster, std::uint16_t* part,
                                                     unsigned* part_key, int k, const std::int32_t* __restrict__ perm,
                                                     int perm_block, std::int32_t* __restrict__ words,
                                                     std::int32_t* __restrict__ emax) {
    __shared__ int column_e;
    const int rank = static_cast<int>(cluster.block_rank()), parts = static_cast<int>(cluster.num_blocks());
    cluster.sync(); // every part and key
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int r = 0; r < parts; ++r) { m = max(m, *cluster.map_shared_rank(part_key, r)); }
        column_e = ggml_encode::column_exponent(m);
        if (rank == 0) { emax[blockIdx.y] = column_e; }
    }
    __syncthreads();
    const int e = column_e, slices = k / 32;
    std::int32_t* column = words + static_cast<std::size_t>(blockIdx.y) * ggml::fast::encoded_column_words(k);
    constexpr int kShare = kPartElements / 8; // chunks a CTA encodes: whole warps
    if (static_cast<int>(threadIdx.x) < kShare) {
        const int c = rank * kShare + static_cast<int>(threadIdx.x);
        const int j = 8 * c;
        const int s = perm == nullptr ? j : perm[j / perm_block] * perm_block + j % perm_block;
        const uint4 u = *reinterpret_cast<const uint4*>(cluster.map_shared_rank(part, s / kPartElements) + s % kPartElements);
        ggml_encode::finish_chunk(column, slices, c, true, ggml_encode::encode_chunk(column, c, e, u));
    }
    cluster.sync(); // no CTA leaves while another reads its part
}

// The largest |bits| of the CTA's part into *part_key: `key` is the thread's own.
__device__ __forceinline__ void part_key_of(unsigned key, unsigned* warp_key, unsigned* part_key) {
    key = __reduce_max_sync(0xFFFFFFFFU, key);
    if (threadIdx.x % kWarpSize == 0) { warp_key[threadIdx.x / kWarpSize] = key; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned m = 0;
        for (int w = 0; w < kWarps; ++w) { m = max(m, warp_key[w]); }
        *part_key = m;
    }
}

__device__ __forceinline__ unsigned pair_key(__nv_bfloat162 y) {
    const unsigned bits = *reinterpret_cast<const unsigned*>(&y);
    return max(bits & 0x7FFFU, (bits >> 16) & 0x7FFFU);
}

// rmsnorm_warp_bf16x2_kernel's gated rows of D = 128 (the route gated_rmsnorm takes there; a lane's pairs lane and
// lane + 32 added in that order), one a warp: CTA rank r holds rows 16 r .. 16 r + 15 of the column.
constexpr int kNormPairsPerLane = 2;

template <RmsEpilogue Epilogue>
__launch_bounds__(kThreads) __global__
    void gated_rmsnorm_encode_kernel(const __nv_bfloat162* __restrict__ x, const __nv_bfloat162* __restrict__ weight,
                                     const __nv_bfloat162* __restrict__ z, std::int32_t d, std::int32_t heads, float eps,
                                     const std::int32_t* __restrict__ perm, int perm_block, std::int32_t* __restrict__ words,
                                     std::int32_t* __restrict__ emax) {
    __shared__ __align__(16) __nv_bfloat162 part[kPartElements / 2];
    __shared__ unsigned warp_key[kWarps];
    __shared__ unsigned part_key;
    cg::cluster_group cluster = cg::this_cluster();
    const int lane  = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp  = static_cast<int>(threadIdx.x) / kWarpSize;
    const int pairs = d / 2;
    __nv_bfloat162 weights[kNormPairsPerLane];
#pragma unroll
    for (int k = 0; k < kNormPairsPerLane; ++k) { weights[k] = weight[lane + k * kWarpSize]; }
    pdl::enter();
    const std::int64_t row =
        static_cast<std::int64_t>(blockIdx.y) * heads + static_cast<std::int64_t>(cluster.block_rank()) * kWarps + warp;
    const std::int64_t row_base = row * pairs;
    __nv_bfloat162 values[kNormPairsPerLane], gates[kNormPairsPerLane];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kNormPairsPerLane; ++k) {
        values[k]       = x[row_base + lane + k * kWarpSize];
        gates[k]        = z[row_base + lane + k * kWarpSize];
        const float2 xf = __bfloat1622float2(values[k]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }
    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(d) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);
    unsigned key = 0;
#pragma unroll
    for (int k = 0; k < kNormPairsPerLane; ++k) {
        const float2 xf        = __bfloat1622float2(values[k]);
        const float2 wf        = __bfloat1622float2(weights[k]);
        const float2 zf        = __bfloat1622float2(gates[k]);
        const __nv_bfloat162 y = __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                                       rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        part[warp * pairs + lane + k * kWarpSize] = y;
        key = max(key, pair_key(y));
    }
    part_key_of(key, warp_key, &part_key);
    encode_cluster_share(cluster, reinterpret_cast<std::uint16_t*>(part), &part_key, heads * d, perm, perm_block, words,
                         emax);
}

// sigmoid_gate_mul_pair over the CTA's 256 packs of the column.
__launch_bounds__(kThreads) __global__
    void sigmoid_mul_encode_kernel(const Bf16x8Pack* __restrict__ gate, const Bf16x8Pack* __restrict__ x, std::int32_t k,
                                   const std::int32_t* __restrict__ perm, int perm_block, std::int32_t* __restrict__ words,
                                   std::int32_t* __restrict__ emax) {
    constexpr int kPacks = kPartElements / 8;
    __shared__ __align__(16) Bf16x8Pack part[kPacks];
    __shared__ unsigned warp_key[kWarps];
    __shared__ unsigned part_key;
    cg::cluster_group cluster = cg::this_cluster();
    pdl::enter();
    unsigned key = 0;
    if (static_cast<int>(threadIdx.x) < kPacks) {
        const std::int64_t i = static_cast<std::int64_t>(blockIdx.y) * (k / 8) +
                               static_cast<std::int64_t>(cluster.block_rank()) * kPacks + threadIdx.x;
        const Bf16x8Pack gv = load_vec<Bf16x8Pack>(gate + i);
        Bf16x8Pack xv       = load_vec<Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            xv.pair[pair] = sigmoid_gate_mul_pair(gv.pair[pair], xv.pair[pair]);
            key           = max(key, pair_key(xv.pair[pair]));
        }
        part[threadIdx.x] = xv;
    }
    part_key_of(key, warp_key, &part_key);
    encode_cluster_share(cluster, reinterpret_cast<std::uint16_t*>(part), &part_key, k, perm, perm_block, words, emax);
}

void require(bool condition, const char* op, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

bool contiguous_bf16(const Tensor& t, std::uintptr_t alignment) {
    return t.data != nullptr && t.dtype == DType::BF16 && t.is_contiguous() &&
           reinterpret_cast<std::uintptr_t>(t.data) % alignment == 0;
}

// The encoded output of `columns` columns of out.k elements (a whole number of parts) and the permutation.
void validate_encoded_output(const char* op, const GgmlEncodedColumns& out, std::int32_t columns,
                             const std::int32_t* input_perm, std::int32_t perm_block) {
    const std::int32_t k = out.k;
    require(columns > 0 && k > 0 && k % kPartElements == 0 && k <= kGgmlEncodeMaxElements, op,
            "the column length must be a multiple of 2048 up to kGgmlEncodeMaxElements");
    require(out.words.data != nullptr && out.words.dtype == DType::I32 && out.words.is_contiguous() &&
                out.words.ne[0] == ggml_encoded_column_words(k) && out.words.ne[1] == columns && out.words.ne[2] == 1 &&
                out.emax.data != nullptr && out.emax.dtype == DType::I32 && out.emax.is_contiguous() &&
                out.emax.ne[0] == columns && out.emax.ne[1] == 1,
            op, "out must be I32 words [ggml_encoded_column_words(K), T] and exponents [T]");
    require(input_perm == nullptr || (perm_block > 0 && perm_block % 8 == 0 && k % perm_block == 0), op,
            "the input permutation needs whole blocks of a multiple of 8 elements");
}

// A cluster of K / 2048 CTAs a column, launched as pdl::launch_consumer launches (a programmatic dependent while a
// graph is captured).
template <class... KernelArgs, class... Args>
void launch_columns(std::int32_t k, std::int32_t columns, cudaStream_t stream, void (*kernel)(KernelArgs...),
                    Args... args) {
    const unsigned parts = static_cast<unsigned>(k / kPartElements);
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id               = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = parts;
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;
    unsigned count                 = 1;
    if (pdl::graph_dependent(stream)) {
        attributes[1].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
        attributes[1].val.programmaticStreamSerializationAllowed = 1;
        count                                                    = 2;
    }
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(parts, static_cast<unsigned>(columns));
    config.blockDim = dim3(kThreads);
    config.stream   = stream;
    config.attrs    = attributes;
    config.numAttrs = count;
    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gated_rmsnorm_ggml_encoded(const Tensor& x, const Tensor& weight, const Tensor& z, RmsGate gate, float eps,
                                const std::int32_t* input_perm, std::int32_t perm_block, GgmlEncodedColumns& out,
                                cudaStream_t stream) {
    constexpr const char* kOp = "gated_rmsnorm_ggml_encoded";
    const std::int32_t d = x.ne[0], columns = out.emax.ne[0];
    require(d == 128, kOp, "rows of 128 elements");
    require(contiguous_bf16(x, 4) && contiguous_bf16(z, 4) && contiguous_bf16(weight, 4) && x.ne[2] == 1 && x.ne[3] == 1,
            kOp, "x, z and weight must be 4-byte aligned contiguous BF16");
    for (int i = 0; i < 4; ++i) { require(z.ne[i] == x.ne[i], kOp, "z must have x's shape"); }
    require(weight.ne[0] == d && weight.ne[1] == 1 && weight.ne[2] == 1 && weight.ne[3] == 1, kOp, "weight must be [D]");
    require(eps > 0.0f && std::isfinite(eps), kOp, "eps must be positive and finite");
    require(columns > 0 && x.ne[1] % columns == 0 && out.k == d * (x.ne[1] / columns), kOp,
            "x's rows must form out's columns");
    validate_encoded_output(kOp, out, columns, input_perm, perm_block);
    const std::int32_t heads = x.ne[1] / columns;
    const auto launch = [&](auto kernel) {
        launch_columns(out.k, columns, stream, kernel, static_cast<const __nv_bfloat162*>(x.data),
                       static_cast<const __nv_bfloat162*>(weight.data), static_cast<const __nv_bfloat162*>(z.data), d,
                       heads, eps, input_perm, perm_block, static_cast<std::int32_t*>(out.words.data),
                       static_cast<std::int32_t*>(out.emax.data));
    };
    if (gate == RmsGate::Sigmoid) {
        launch(gated_rmsnorm_encode_kernel<RmsEpilogue::SigmoidGated>);
    } else {
        launch(gated_rmsnorm_encode_kernel<RmsEpilogue::Gated>);
    }
}

void sigmoid_mul_ggml_encoded(const Tensor& gate, const Tensor& x, const std::int32_t* input_perm,
                              std::int32_t perm_block, GgmlEncodedColumns& out, cudaStream_t stream) {
    constexpr const char* kOp = "sigmoid_mul_ggml_encoded";
    const std::int32_t columns = out.emax.ne[0];
    require(contiguous_bf16(x, 16) && contiguous_bf16(gate, 16) && x.ne[2] == 1 && x.ne[3] == 1, kOp,
            "gate and x must be 16-byte aligned contiguous BF16 [K, T]");
    for (int i = 0; i < 4; ++i) { require(gate.ne[i] == x.ne[i], kOp, "gate must have x's shape"); }
    require(x.ne[0] == out.k && x.ne[1] == columns, kOp, "x must be [out.k, T]");
    validate_encoded_output(kOp, out, columns, input_perm, perm_block);
    launch_columns(out.k, columns, stream, sigmoid_mul_encode_kernel, static_cast<const Bf16x8Pack*>(gate.data),
                   static_cast<const Bf16x8Pack*>(x.data), out.k, input_perm, perm_block,
                   static_cast<std::int32_t*>(out.words.data), static_cast<std::int32_t*>(out.emax.data));
}

} // namespace infernix::ops
