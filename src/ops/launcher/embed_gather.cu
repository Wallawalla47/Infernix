// infernix::ops - embedding launcher: variant grid/block/stream setup.
#include "core/weight.h"
#include "ops/launcher/embed_gather.h"

#include "ops/common/math.h"
#include "ops/kernel/embed_gather.cuh"
#include "ops/linear/bf16_p12/bf16_p12_kernels.cuh"
#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"

#include <algorithm>
#include <cstdint>

namespace infernix::ops::detail {
namespace {

// bf16_p12 rows (BF16 packed at load, core/weight.h P12Rows): a CTA per token and a thread per 8-weight chunk, so a
// table in pinned host memory costs one round trip of independent reads (the row's words, codes and nibbles; a chunk
// with escapes then reads the row's escape list). Each chunk decodes to the stored BF16 words.
template <bool Vector>
__global__ void embed_gather_p12_kernel(EmbedIds ids, const uint2* __restrict__ codes,
                                        const unsigned* __restrict__ nibbles, const unsigned* __restrict__ words,
                                        const unsigned* __restrict__ escapes, __nv_bfloat16* __restrict__ out,
                                        std::int32_t d) {
    const auto t              = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t row    = ids.row(t);
    const std::int32_t chunks = d / 8;
    __nv_bfloat16* column     = out + static_cast<std::int64_t>(t) * d;
    const std::int64_t base   = static_cast<std::int64_t>(row < 0 ? 0 : row) * chunks;
    const P12Row state        = row < 0 ? P12Row{} : p12_row(words, row);
    for (auto c = static_cast<std::int32_t>(threadIdx.x); c < chunks; c += static_cast<std::int32_t>(blockDim.x)) {
        const uint4 w = row < 0 ? make_uint4(0, 0, 0, 0) : p12_decode_chunk(codes[base + c], nibbles[base + c], state, c, escapes);
        if constexpr (Vector) {
            reinterpret_cast<uint4*>(column)[c] = w;
        } else {
            const unsigned parts[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                column[8 * c + 2 * i]     = __ushort_as_bfloat16(static_cast<unsigned short>(parts[i] & 0xFFFFU));
                column[8 * c + 2 * i + 1] = __ushort_as_bfloat16(static_cast<unsigned short>(parts[i] >> 16));
            }
        }
    }
}

constexpr int kBlock          = 128;
constexpr int kQ6GroupedBlock = kEmbedGatherQ6Group * kEmbedGatherQ6GroupsPerBlock;
constexpr int kQ8GroupedBlock = 32;
constexpr int kQ8RowBlock     = 256;

template <int BlocksPerToken, int Threads>
void launch_fp8(const Tensor& ids, const EmbedIds& guard, const Weight& table, Tensor& out, cudaStream_t stream) {
    const int grid = ids.ne[0] * BlocksPerToken;
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(grid), dim3(Threads), 0, stream}, embed_gather_fp8_kernel<BlocksPerToken, Threads>,
        guard, static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const __nv_bfloat16*>(table.scales), static_cast<__nv_bfloat16*>(out.data)));
}

template <int Blocks, int Threads>
void launch_q8_packed(const Tensor& ids, const EmbedIds& guard, const Weight& table, Tensor& out,
                      cudaStream_t stream) {
    const auto launch = [&]<bool PairStore>() {
        embed_gather_q8_packed_5120_kernel<Blocks, Threads, PairStore>
            <<<ids.ne[0] * Blocks, Threads, 0, stream>>>(
                guard,
                static_cast<const std::uint8_t*>(table.qdata),
                static_cast<const std::uint8_t*>(table.scales),
                static_cast<__nv_bfloat16*>(out.data));
    };
    if (reinterpret_cast<std::uintptr_t>(out.data) % 4 == 0)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
}

int grid_for(std::int64_t n) {
    return static_cast<int>(
        std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
}

int grid_for_q6_grouped(std::int32_t d, std::int32_t T) {
    const std::int32_t kg           = d / kEmbedGatherQ6Group;
    const std::int32_t group_blocks = div_up(kg, kEmbedGatherQ6GroupsPerBlock);
    return static_cast<int>(std::max<std::int64_t>(1, static_cast<std::int64_t>(T) *
                                                          static_cast<std::int64_t>(group_blocks)));
}

} // namespace

const char* q8_embed_route_name(Q8EmbedRoute route) {
    switch (route) {
    case Q8EmbedRoute::Auto:
        return "auto";
    case Q8EmbedRoute::Grouped:
        return "grouped-b32";
    case Q8EmbedRoute::Row:
        return "row-b256";
    }
    return "unknown";
}

void embed_gather_q8_2048_launch(const Tensor& ids, const Weight& table, Tensor& out,
                                 Q8EmbedRoute route, cudaStream_t stream, DeviceFault fault) {
    const EmbedIds guard{static_cast<const std::int32_t*>(ids.data), table.shape[0], fault};
    const std::int32_t T = ids.ne[0];
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (route == Q8EmbedRoute::Auto) { route = T <= 6 ? Q8EmbedRoute::Grouped : Q8EmbedRoute::Row; }
    // The row kernel loads eight codes as one uint2 and stores BF16 pairs; every row and column
    // offset it adds is a multiple of 16 bytes, so the base pointers decide. A code plane off an
    // 8-byte boundary or an output off a 4-byte one takes the grouped kernel: same bits.
    const bool row_aligned = reinterpret_cast<std::uintptr_t>(codes) % alignof(uint2) == 0 &&
                             reinterpret_cast<std::uintptr_t>(out.data) % alignof(__nv_bfloat162) == 0;
    if (route == Q8EmbedRoute::Row && !row_aligned) { route = Q8EmbedRoute::Grouped; }
    if (route == Q8EmbedRoute::Grouped) {
        const int grid = T * kEmbedGatherQ8Groups;
        embed_gather_q8_grouped_2048_kernel<<<grid, kQ8GroupedBlock, 0, stream>>>(
            guard, codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
    } else {
        embed_gather_q8_row_2048_kernel<<<T, kQ8RowBlock, 0, stream>>>(
            guard, codes, scales,
            static_cast<__nv_bfloat16*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_dense_launch(const Tensor& ids, const Tensor& table, std::int32_t vocab, Tensor& out,
                               cudaStream_t stream, DeviceFault fault) {
    const EmbedIds guard{static_cast<const std::int32_t*>(ids.data), vocab, fault};
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_dense_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        guard, static_cast<const __nv_bfloat16*>(table.data),
        static_cast<__nv_bfloat16*>(out.data), d, T);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_q6_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream, DeviceFault fault) {
    const EmbedIds guard{static_cast<const std::int32_t*>(ids.data), table.shape[0], fault};
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* high     = static_cast<const std::uint8_t*>(table.qhigh);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (d == table.padded_shape[1] && d % kEmbedGatherQ6Group == 0) {
        embed_gather_q6_grouped_kernel<<<grid_for_q6_grouped(d, T), kQ6GroupedBlock, 0, stream>>>(
            guard, codes, high, scales,
            static_cast<__nv_bfloat16*>(out.data), d, T);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    embed_gather_q6_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        guard, codes, high, scales,
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_q8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                            cudaStream_t stream, DeviceFault fault) {
    const EmbedIds guard{static_cast<const std::int32_t*>(ids.data), table.shape[0], fault};
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const auto* codes    = static_cast<const std::uint8_t*>(table.qdata);
    const auto* scales   = static_cast<const std::uint8_t*>(table.scales);
    if (d == kEmbedGatherQ8D && table.padded_shape[1] == kEmbedGatherQ8D) {
        embed_gather_q8_2048_launch(ids, table, out, Q8EmbedRoute::Auto, stream, fault);
        return;
    }

    // Four signed codes share their exact group scale. Keep byte-addressed tables on the
    // scalar reader and preserve two-byte output alignment through the packed kernel's stores.
    if (d == 5120 && table.padded_shape[1] == 5120 &&
        reinterpret_cast<std::uintptr_t>(table.qdata) % 4 == 0) {
        if (T <= 128)
            launch_q8_packed<10, 128>(ids, guard, table, out, stream);
        else
            launch_q8_packed<5, 128>(ids, guard, table, out, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n = static_cast<std::int64_t>(d) * T;
    embed_gather_q8_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        guard, codes, scales,
        static_cast<__nv_bfloat16*>(out.data), d, T, table.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_p12_launch(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream,
                             DeviceFault fault) {
    const EmbedIds guard{static_cast<const std::int32_t*>(ids.data), table.n, fault};
    const std::int32_t d       = out.ne[0];
    const std::int32_t threads = std::min(1024, static_cast<std::int32_t>(div_up(d / 8, 32)) * 32);
    const auto* codes          = static_cast<const uint2*>(table.qdata);
    const auto* nibbles        = static_cast<const unsigned*>(table.qhigh);
    const auto* words          = static_cast<const unsigned*>(table.scales);
    const auto* escapes        = static_cast<const unsigned*>(table.escapes);
    auto* column               = static_cast<__nv_bfloat16*>(out.data);
    if (reinterpret_cast<std::uintptr_t>(out.data) % 16 == 0) {
        embed_gather_p12_kernel<true><<<ids.ne[0], threads, 0, stream>>>(guard, codes, nibbles, words, escapes, column, d);
    } else {
        embed_gather_p12_kernel<false><<<ids.ne[0], threads, 0, stream>>>(guard, codes, nibbles, words, escapes, column, d);
    }
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_fp8_launch(const Tensor& ids, const Weight& table, Tensor& out,
                             cudaStream_t stream, DeviceFault fault) {
    const EmbedIds guard{static_cast<const std::int32_t*>(ids.data), table.n, fault};
    const std::int32_t T = ids.ne[0];
    if (T <= 176)
        launch_fp8<10, 128>(ids, guard, table, out, stream);
    else
        launch_fp8<5, 128>(ids, guard, table, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
