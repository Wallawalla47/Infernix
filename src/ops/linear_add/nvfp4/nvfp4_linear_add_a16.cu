#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear/nvfp4/nvfp4_a16_slice_ring_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
template <int K>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& residual, bool wide_k, cudaStream_t stream) {
    const int tokens  = x.ne[1];
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto epilogue =
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}};
    if constexpr (K == 6144) {
        if (tokens <= 8) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 6144>>(p, output, epilogue,
                                                                           stream);
            return;
        }
        if (tokens <= 16) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 6144>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 24) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 1>, 6144>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, 6144>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 48) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 2, 2>, 6144>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 2, 2>, 6144>>(p, output, epilogue,
                                                                            stream);
            return;
        }
        if (tokens <= 128) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 128, 32, 16, 2, 1>, 6144>>(
                p, output, epilogue, stream);
            return;
        }
        {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 6144>>(
                p, output, epilogue, stream);
            return;
        }
    } else {
        // N=5120 gives these tiles 320 CTAs of 16 rows, so their time is the weight stream each
        // keeps in flight: 8 K-warps stream 512 K per stage where 4 streamed 256. Measured on RTX
        // 5090 (cold weights, two interleaved passes): 35.6-38.0 us at 5-8 tokens against 41.6
        // (SIMT, 5) and 58.5-58.7 (4 K-warps, 6-8), and 44.4-51.8 us at 9-14 against 58.0-60.8.
        // The K-warp count sets each column's reduction order, so the 8 K-warp tiles run only under
        // LinearRoutes::WideDownTile (`wide_k`: batched-decode's decode and verification rounds,
        // where they were gated); the exact route and prefill keep the 4 K-warp tiles of the release.
        if (tokens <= 8) {
            if (wide_k) {
                launch_nvfp4_a16_sliced_k_mma<
                    Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 17408>>(p, output, epilogue, stream);
            } else {
                launch_nvfp4_a16_sliced_k_mma<
                    Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 4, 2>, 17408>>(p, output, epilogue, stream);
            }
            return;
        }
        if (tokens <= 16) {
            if (wide_k) {
                launch_nvfp4_a16_sliced_k_mma<
                    Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 17408>>(p, output, epilogue, stream);
            } else {
                launch_nvfp4_a16_sliced_k_mma<
                    Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 4, 2>, 17408>>(p, output, epilogue, stream);
            }
            return;
        }
        if (tokens <= 24) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, 17408>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 32) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 17408>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 48) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 17408>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 64) {
            launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 2, 2>, 17408>>(p, output, epilogue,
                                                                             stream);
            return;
        }
        if (tokens <= 128) {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>, 17408>>(
                p, output, epilogue, stream);
            return;
        }
        {
            launch_nvfp4_a16_mma<
                Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 17408>>(
                p, output, epilogue, stream);
            return;
        }
    }
}
} // namespace

void nvfp4_linear_add_invariant_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                       cudaStream_t stream) {
    // The slice-major ring tile, bit-identical to the sliced-K tiles, where it is faster (2026-10-25): [5120,6144]
    // from 17 columns (0.45-0.68x), [5120,17408] from 32 (1.23x at 17, 1.06x at 24, 0.93x at 32, 0.52-0.85x
    // beyond); the reproducible route's DFlash2 decode +19 %.
    if (x.ne[1] >= (weight.k == 17408 ? 32 : 17)) {
        nvfp4_linear_add_invariant_ring_launch(x, weight, residual, stream);
        return;
    }
    nvfp4_linear_add_invariant_sliced_launch(x, weight, residual, stream);
}

namespace {
template <int KSlices, int K>
void launch_invariant_ring(const Tensor& x, const Weight& weight, Tensor& residual,
                           cudaStream_t stream) {
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto epilogue =
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}};
    // N=5120: 32-row tiles give 160 CTAs a token tile, two an SM; above 64 columns 64-row tiles
    // keep 160 CTAs over two token tiles with half the activation reads.
    if (x.ne[1] <= 32) {
        launch_nvfp4_a16_slice_ring_mma<
            Nvfp4ScheduleInstance<Nvfp4A16SliceRingSchedule<KSlices, 32, 32, 2, 2, 4, 2>, K>>(
            p, output, epilogue, stream);
    } else if (x.ne[1] <= 64) {
        launch_nvfp4_a16_slice_ring_mma<
            Nvfp4ScheduleInstance<Nvfp4A16SliceRingSchedule<KSlices, 32, 64, 2, 4, 4, 2>, K>>(
            p, output, epilogue, stream);
    } else {
        launch_nvfp4_a16_slice_ring_mma<
            Nvfp4ScheduleInstance<Nvfp4A16SliceRingSchedule<KSlices, 64, 64, 4, 2, 4, 2>, K>>(
            p, output, epilogue, stream);
    }
}
} // namespace

void nvfp4_linear_add_invariant_ring_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                            cudaStream_t stream) {
    // The K-warp counts of the sliced-K tiles below: 8 at K=6144, 4 at K=17408.
    if (x.ne[1] < 17) throw std::invalid_argument("nvfp4 linear_add ring tile: T must be 17 or more");
    if (weight.k == 6144) return launch_invariant_ring<8, 6144>(x, weight, residual, stream);
    if (weight.k == 17408) return launch_invariant_ring<4, 17408>(x, weight, residual, stream);
    throw std::invalid_argument("nvfp4 linear_add: unsupported column-invariant problem");
}

void nvfp4_linear_add_invariant_sliced_launch(const Tensor& x, const Weight& weight,
                                              Tensor& residual, cudaStream_t stream) {
    // Tiles of one K-warp count walk the same K slices and reduce them in the same order, and the
    // MMA's columns are independent: the tile width only sets how many columns share a launch,
    // and wider calls run as 64-column slices.
    const int tokens  = x.ne[1];
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = LinearBf16Output{static_cast<__nv_bfloat16*>(residual.data), weight.n};
    const auto epilogue =
        LinearResidualAddEpilogue{{static_cast<__nv_bfloat16*>(residual.data), weight.n}};
    if (weight.k == 6144) {
        if (tokens <= 8)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 6144>>(p, output, epilogue,
                                                                           stream);
        if (tokens <= 16)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 6144>>(p, output, epilogue,
                                                                            stream);
        return launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 8, 1>, 6144>>(p, output, epilogue,
                                                                        stream);
    }
    if (weight.k == 17408) {
        if (tokens <= 8)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 4, 2>, 17408>>(p, output, epilogue,
                                                                            stream);
        if (tokens <= 16)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 4, 2>, 17408>>(p, output, epilogue,
                                                                             stream);
        if (tokens <= 32)
            return launch_nvfp4_a16_sliced_k_mma<
                Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, 17408>>(p, output, epilogue,
                                                                             stream);
        return launch_nvfp4_a16_sliced_k_mma<
            Nvfp4ScheduleInstance<Nvfp4SlicedInstance<64, 4, 2>, 17408>>(p, output, epilogue,
                                                                         stream);
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported column-invariant problem");
}

void nvfp4_linear_add_a16_launch(const Tensor& x, const Weight& weight, Tensor& residual, bool wide_k,
                                 cudaStream_t stream) {
    if (x.ne[1] == 1) {
        nvfp4_linear_add_decode_launch(x, weight, residual, stream);
        return;
    }
    // At K=17408 the SIMT kernel ties the 8 K-warp tile at 3-4 tokens (34-36 us) and loses at 5; it
    // keeps 5 tokens beside the 4 K-warp tiles.
    if (x.ne[1] <= (weight.k == 6144 ? 2 : (wide_k ? 4 : 5))) {
        nvfp4_linear_add_small_t_launch(x, weight, residual, stream);
        return;
    }
    if (weight.k == 6144)
        launch_matrix<6144>(x, weight, residual, wide_k, stream);
    else
        launch_matrix<17408>(x, weight, residual, wide_k, stream);
}
} // namespace infernix::ops::detail
