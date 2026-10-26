// Row gathers over GGUF block tables: the token embedding and the PLE n-gram rows
// (include/infernix/ops/ggml_rows.h).

#include "infernix/ops/ggml_rows.h"

#include "core/ggml_format.h"
#include "ops/ggml/ggml_exact.h"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ggml rows: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("ggml rows ") + what + ": " + cudaGetErrorString(error)); }
}

__device__ __forceinline__ float f16_value(std::uint16_t word) {
    return __half2float(__ushort_as_half(word));
}

// The exact value of one chunk element, kappa * d * sub * code, for formats without a min term
// whose |sub * code| < 2^13 (IQ4_XS, IQ4_NL, Q8_0): with d's 11 significant bits it is exact in binary32.
template <QType F>
__device__ __forceinline__ void chunk_values(const std::uint8_t* row, int index, float (&v)[8]) {
    const ggml::Chunk c = ggml::row_chunk<F>(row, index);
    const float d = f16_value(c.d);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float m = static_cast<float>(c.sub * c.code[j]);
        v[j] = ggml::scale_pow2(__fmul_rn(d, m), ggml::kappa_log2(F));
    }
}

template <QType F>
__global__ void __launch_bounds__(kThreads)
    embedding_kernel(const std::int32_t* __restrict__ ids, const std::uint8_t* __restrict__ table, int vocab, int hidden,
                     int row_bytes, std::uint16_t* __restrict__ out) {
    const int t  = blockIdx.x;
    const int id = ids[t];
    if (id < 0 || id >= vocab) { __trap(); }
    const std::uint8_t* row = table + static_cast<std::size_t>(id) * row_bytes;
    for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x) {
        float v[8];
        chunk_values<F>(row, i, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[static_cast<std::size_t>(t) * hidden + 8 * i + j] = canon::f32_to_bf16_rn(v[j]);
        }
    }
}

template <QType F>
__global__ void __launch_bounds__(kThreads)
    ple_rows_kernel(const std::uint8_t* __restrict__ rows, int row_bytes, int heads, int width,
                    std::uint16_t* __restrict__ out) {
    const int t = blockIdx.x;
    const int per_head = width / 8;
    for (int i = threadIdx.x; i < heads * per_head; i += blockDim.x) {
        const int h = i / per_head, chunk = i % per_head;
        const std::uint8_t* row = rows + (static_cast<std::size_t>(t) * heads + h) * row_bytes;
        float v[8];
        chunk_values<F>(row, chunk, v);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[static_cast<std::size_t>(t) * heads * width + static_cast<std::size_t>(h) * width + 8 * chunk + j] =
                canon::f32_to_bf16_rn(v[j]);
        }
    }
}

} // namespace

void ggml_embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream) {
    require(ids.data != nullptr && ids.dtype == DType::I32 && ids.is_contiguous(), "ids must be contiguous I32 [T]");
    require(out.data != nullptr && out.dtype == DType::BF16 && out.is_contiguous(), "out must be contiguous BF16 [H, T]");
    require(table.layout == QuantLayout::GgmlRows && table.qdata != nullptr, "table must be a ggml_rows_v1 Weight");
    const int vocab = table.n, hidden = table.k, columns = static_cast<int>(ids.numel());
    require(out.ne[0] == hidden && out.ne[1] == columns && columns > 0 && hidden % 8 == 0, "embedding shapes disagree");
    const int row_bytes = static_cast<int>(hidden / ggml_block_elems(table.qtype) * ggml_block_bytes(table.qtype));
    const auto* data    = static_cast<const std::uint8_t*>(table.qdata);
    auto* o             = static_cast<std::uint16_t*>(out.data);
    const auto* id      = static_cast<const std::int32_t*>(ids.data);
    switch (table.qtype) {
    case QType::GGML_IQ4_XS:
        embedding_kernel<QType::GGML_IQ4_XS><<<columns, kThreads, 0, stream>>>(id, data, vocab, hidden, row_bytes, o);
        break;
    default:
        // Other formats' products can exceed 24 significant bits (Q6_K), so they would need the
        // int128 path; only IQ4_XS tables are registered (a GGUF's Q8_0 embedding is imported as
        // q8_g32_fp16, which ops::embedding gathers).
        throw std::invalid_argument("ggml_embedding: table format has no embedding route (IQ4_XS)");
    }
    check_launch("embedding");
}

namespace {

// The PLE n-gram rows of a 32-element block format (IQ4_NL, Q8_0).
template <QType F>
void ple_embed_rows(const Tensor& rows, Tensor& out, cudaStream_t stream, const char* label) {
    constexpr int kBlockBytes = static_cast<int>(ggml_block_bytes(F));
    require(rows.data != nullptr && rows.dtype == DType::U8 && rows.is_contiguous(), "rows must be contiguous U8");
    require(out.data != nullptr && out.dtype == DType::BF16 && out.is_contiguous(), "out must be contiguous BF16");
    const int row_bytes = rows.ne[0], heads = rows.ne[1], columns = rows.ne[2];
    require(row_bytes % kBlockBytes == 0 && heads > 0 && columns > 0, "rows must be whole 32-element blocks");
    const int width = row_bytes / kBlockBytes * 32;
    require(out.ne[0] == heads * width && out.ne[1] == columns, "ple embed shapes disagree");
    ple_rows_kernel<F><<<columns, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(rows.data), row_bytes, heads,
                                                         width, static_cast<std::uint16_t*>(out.data));
    check_launch(label);
}

} // namespace

void ple_embed_iq4_nl(const Tensor& rows, Tensor& out, cudaStream_t stream) {
    ple_embed_rows<QType::GGML_IQ4_NL>(rows, out, stream, "ple iq4_nl");
}

void ple_embed_q8_0(const Tensor& rows, Tensor& out, cudaStream_t stream) {
    ple_embed_rows<QType::GGML_Q8_0>(rows, out, stream, "ple q8_0");
}

} // namespace infernix::ops
