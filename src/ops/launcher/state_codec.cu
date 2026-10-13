#include "ops/launcher/state_codec.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/state_codec.cuh"

#include <algorithm>
#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kBlock   = 256;
constexpr int kGridCap = 4096;

int codec_grid(std::int64_t groups) {
    return static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(div_up(groups, static_cast<std::int64_t>(kBlock)), kGridCap)));
}

} // namespace

void pack_fp32_state_launch(const Tensor& source, void* packed, PackedStateFormat format,
                            cudaStream_t stream) {
    const std::int64_t groups = source.numel() / 4;
    const auto* values        = static_cast<const float4*>(source.data);
    if (format == PackedStateFormat::Fp24) {
        pack_fp24_kernel<<<codec_grid(groups), kBlock, 0, stream>>>(
            values, static_cast<std::uint32_t*>(packed), groups);
    } else {
        pack_bf16_kernel<<<codec_grid(groups), kBlock, 0, stream>>>(
            values, static_cast<__nv_bfloat162*>(packed), groups);
    }
    CUDA_CHECK(cudaGetLastError());
}

void unpack_fp32_state_launch(const void* packed, Tensor& destination, PackedStateFormat format,
                              cudaStream_t stream) {
    const std::int64_t groups = destination.numel() / 4;
    auto* values              = static_cast<float4*>(destination.data);
    if (format == PackedStateFormat::Fp24) {
        unpack_fp24_kernel<<<codec_grid(groups), kBlock, 0, stream>>>(
            static_cast<const std::uint32_t*>(packed), values, groups);
    } else {
        unpack_bf16_kernel<<<codec_grid(groups), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(packed), values, groups);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
