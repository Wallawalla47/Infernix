// infernix::ops - rope launcher: private token-count tuning and generic fallback.
#include "ops/launcher/rope.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/rope.cuh"

#include <type_traits>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kLargeBlock               = 256;
constexpr int kFullChunkBlock           = 192;
constexpr int kSmallBlock               = 128;
constexpr int kDefaultChunkTargetTokens = 1024;
// Qualified residency of this 256-thread instance on the sm_120a target.
constexpr int kLargeBlockResidentCtasPerSm = 6;

template <RopeKernelMode Mode>
inline constexpr bool kTextMode =
    Mode == RopeKernelMode::Text1D || Mode == RopeKernelMode::TextMrope ||
    Mode == RopeKernelMode::DflashText1D;

// Prepared (YaRN) coefficients carry their own inverse frequencies, so the fixed instances
// accept them at any theta; native coefficients are only instantiated for the model's theta.
template <class Coefficients>
inline constexpr bool kPrepared = std::is_same_v<Coefficients, PreparedRopeCoefficients>;

std::int64_t token_stride(const Tensor* tensor) {
    return tensor == nullptr ? 0 : tensor->nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

bool bf16x2_aligned(const Tensor& tensor) {
    return (reinterpret_cast<std::uintptr_t>(tensor.data) & (alignof(__nv_bfloat162) - 1)) == 0 &&
           tensor.nb[2] % static_cast<std::int64_t>(alignof(__nv_bfloat162)) == 0;
}

template <RopeKernelMode Mode, int QHeads, int KHeads, class Coefficients = NativeRopeCoefficients>
void launch_fixed_block(const Tensor& positions, Tensor* q, Tensor* k, int block,
                        cudaStream_t stream, const Coefficients& coefficients = {}) {
    const int tokens = positions.ne[0];
    rope_fixed_kernel<Mode, QHeads, KHeads, Coefficients><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
        token_stride(k), coefficients);
}

template <RopeKernelMode Mode, int QHeads, int KHeads, class Coefficients = NativeRopeCoefficients>
void launch_fixed(const Tensor& positions, Tensor* q, Tensor* k, DeviceExecutionView execution,
                  const Coefficients& coefficients = {}) {
    const int tokens = positions.ne[0];
    int block        = kSmallBlock;
    if constexpr (kTextMode<Mode>) {
        if (tokens <= 6) {
            block = (QHeads + KHeads) * 32;
        } else if (tokens <= static_cast<std::int64_t>(execution.multiprocessor_count) *
                                 kLargeBlockResidentCtasPerSm) {
            block = kLargeBlock;
        } else if (tokens <= kDefaultChunkTargetTokens) {
            block = kFullChunkBlock;
        }
        const int head_warps = (QHeads + KHeads) * 32;
        if (block > head_warps) { block = head_warps; }
        if (block > 1024) { block = 1024; }
    }
    launch_fixed_block<Mode, QHeads, KHeads>(positions, q, k, block, execution.stream,
                                             coefficients);
}

template <int HeadsPerBlock, int QHeads, int KHeads, class Coefficients = NativeRopeCoefficients>
void launch_dflash_split(const Tensor& positions, Tensor* q, Tensor* k, cudaStream_t stream,
                         const Coefficients& coefficients = {}) {
    constexpr int kGroups = (QHeads + KHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int kBlock  = HeadsPerBlock <= 2 ? 64 : HeadsPerBlock * 32;
    const int tokens      = positions.ne[0];
    rope_fixed_split_kernel<RopeKernelMode::DflashText1D, QHeads, KHeads, HeadsPerBlock,
                            Coefficients><<<tokens * kGroups, kBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), tokens, token_stride(q),
        token_stride(k), coefficients);
}

template <class Coefficients = NativeRopeCoefficients>
bool launch_fixed_pair(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                       DeviceExecutionView execution, const Coefficients& coefficients = {}) {
    const auto stream = execution.stream;
    if (!bf16x2_aligned(q) || !bf16x2_aligned(k)) { return false; }
    const int axes        = positions.ne[1];
    const bool text_theta = kPrepared<Coefficients> || theta == 1.0e7F;
    if (axes == 1 && q.ne[0] == 128 && rotary_dim == 128 && text_theta && q.ne[1] == 32 &&
        k.ne[1] == 8) {
        const int tokens = positions.ne[0];
        if (tokens <= 16) {
            launch_dflash_split<5, 32, 8>(positions, &q, &k, stream, coefficients);
        } else if (tokens <= 400) {
            launch_dflash_split<8, 32, 8>(positions, &q, &k, stream, coefficients);
        } else {
            launch_fixed_block<RopeKernelMode::DflashText1D, 32, 8>(positions, &q, &k, 160,
                                                                    stream, coefficients);
        }
        return true;
    }
    if (rotary_dim == 64 && text_theta) {
        if (q.ne[1] == 24 && k.ne[1] == 4) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 24, 4>(positions, &q, &k, execution,
                                                            coefficients);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 24, 4>(positions, &q, &k, execution,
                                                               coefficients);
                return true;
            }
        }
        if (q.ne[1] == 16 && k.ne[1] == 2) {
            if (axes == 1) {
                launch_fixed<RopeKernelMode::Text1D, 16, 2>(positions, &q, &k, execution,
                                                            coefficients);
                return true;
            }
            if (axes == 3) {
                launch_fixed<RopeKernelMode::TextMrope, 16, 2>(positions, &q, &k, execution,
                                                               coefficients);
                return true;
            }
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && q.ne[1] == 16 && k.ne[1] == 16) {
        launch_fixed<RopeKernelMode::Vision2D, 16, 16>(positions, &q, &k, execution, coefficients);
        return true;
    }
    return false;
}

template <RopeKernelMode Mode, int Heads, class Coefficients = NativeRopeCoefficients>
void launch_fixed_single(const Tensor& positions, Tensor& x, DeviceExecutionView execution,
                         const Coefficients& coefficients = {}) {
    launch_fixed<Mode, Heads, 0>(positions, &x, nullptr, execution, coefficients);
}

template <int Heads, class Coefficients = NativeRopeCoefficients>
bool launch_text_single(const Tensor& positions, int axes, Tensor& x,
                        DeviceExecutionView execution, const Coefficients& coefficients = {}) {
    if (x.ne[1] != Heads) { return false; }
    if (axes == 1) {
        launch_fixed_single<RopeKernelMode::Text1D, Heads>(positions, x, execution, coefficients);
        return true;
    }
    if (axes == 3) {
        launch_fixed_single<RopeKernelMode::TextMrope, Heads>(positions, x, execution,
                                                              coefficients);
        return true;
    }
    return false;
}

template <class Coefficients = NativeRopeCoefficients>
bool launch_fixed_single_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                                  DeviceExecutionView execution,
                                  const Coefficients& coefficients = {}) {
    if (!bf16x2_aligned(x)) { return false; }
    const int axes        = positions.ne[1];
    const bool text_theta = kPrepared<Coefficients> || theta == 1.0e7F;
    if (axes == 1 && x.ne[0] == 128 && rotary_dim == 128 && text_theta) {
        if (x.ne[1] == 32) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 32>(positions, x, execution,
                                                                  coefficients);
            return true;
        }
        if (x.ne[1] == 8) {
            launch_fixed_single<RopeKernelMode::DflashText1D, 8>(positions, x, execution,
                                                                 coefficients);
            return true;
        }
    }
    if (rotary_dim == 64 && text_theta) {
        if (launch_text_single<24>(positions, axes, x, execution, coefficients) ||
            launch_text_single<4>(positions, axes, x, execution, coefficients) ||
            launch_text_single<16>(positions, axes, x, execution, coefficients) ||
            launch_text_single<2>(positions, axes, x, execution, coefficients)) {
            return true;
        }
    }
    if (axes == 2 && rotary_dim == 72 && theta == 10'000.0F && x.ne[1] == 16) {
        launch_fixed_single<RopeKernelMode::Vision2D, 16>(positions, x, execution, coefficients);
        return true;
    }
    return false;
}

template <class Coefficients = NativeRopeCoefficients>
void launch_generic(const Tensor& positions, int rotary_dim, float theta, Tensor* q, Tensor* k,
                    cudaStream_t stream, const Coefficients& coefficients = {}) {
    constexpr int block = 128;
    Tensor& sample      = q != nullptr ? *q : *k;
    const int tokens    = sample.ne[2];
    rope_generic_kernel<Coefficients><<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), positions.ne[1],
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        k == nullptr ? nullptr : static_cast<__nv_bfloat16*>(k->data), sample.ne[0], rotary_dim,
        theta, q == nullptr ? 0 : q->ne[1], k == nullptr ? 0 : k->ne[1], tokens, token_stride(q),
        token_stride(k), coefficients);
}

} // namespace

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 DeviceExecutionView execution) {
    if (!launch_fixed_pair(positions, rotary_dim, theta, q, k, execution)) {
        launch_generic(positions, rotary_dim, theta, &q, &k, execution.stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        DeviceExecutionView execution) {
    if (!launch_fixed_single_dispatch(positions, rotary_dim, theta, x, execution)) {
        launch_generic(positions, rotary_dim, theta, &x, nullptr, execution.stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void rope_prepared_launch(const Tensor& positions, const PreparedRope& prepared, Tensor& q,
                          Tensor* k, DeviceExecutionView execution) {
    const PreparedRopeCoefficients coefficients{prepared};
    const bool fixed =
        k != nullptr ? launch_fixed_pair(positions, prepared.rotary_dim, prepared.theta, q, *k,
                                         execution, coefficients)
                     : launch_fixed_single_dispatch(positions, prepared.rotary_dim,
                                                    prepared.theta, q, execution, coefficients);
    if (!fixed) {
        launch_generic(positions, prepared.rotary_dim, prepared.theta, &q, k, execution.stream,
                       coefficients);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
