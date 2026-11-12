#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

#include <stdexcept>
#include <type_traits>

namespace infernix::ops::detail {
void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4PromptWeightPromotion>, 5120>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{output, p.rows / 2},
                               SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}

void launch_nvfp4_linear_swiglu_a4_tma_planes(const Nvfp4A4Operands& p, Nvfp4A4Workspace product,
                                              float down_input_scale_divisor, Nvfp4ScaleLayout product_layout,
                                              cudaStream_t stream, int row_major_from) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1, kNvfp4PromptWeightPromotion>, 5120>;
    const auto run = [&]<Nvfp4ScaleLayout Layout>(std::integral_constant<Nvfp4ScaleLayout, Layout>) {
        SwiGluA4PlanesEpilogue<1, Layout> epilogue{};
        epilogue.row_major_from      = row_major_from;
        epilogue.codes               = product.codes;
        epilogue.scales              = product.scales;
        epilogue.code_plane          = 0;
        epilogue.scale_plane         = 0;
        epilogue.input_scale_divisor = down_input_scale_divisor;
        epilogue.down_input_rows     = p.rows / 2;
        launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{nullptr, p.rows / 2}, epilogue, stream,
                                   SwiGluTokenMajorMmaRows<S>{});
    };
    if (product.terms != 1 ||
        product.scale_bytes < static_cast<std::size_t>(nvfp4_a4_padded_tokens(p.tokens)) * (p.rows / 32)) {
        throw std::invalid_argument("nvfp4 linear_swiglu TMA planes: one plane of the padded tokens' scales");
    }
    if (product_layout == Nvfp4ScaleLayout::Tiled256) {
        run(std::integral_constant<Nvfp4ScaleLayout, Nvfp4ScaleLayout::Tiled256>{});
    } else if (product_layout == Nvfp4ScaleLayout::RowMajor) {
        run(std::integral_constant<Nvfp4ScaleLayout, Nvfp4ScaleLayout::RowMajor>{});
    } else {
        throw std::invalid_argument("nvfp4 linear_swiglu TMA planes: the down projection reads row-major or "
                                    "256-token tiled scales");
    }
}

void launch_nvfp4_linear_swiglu_a4_residual_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                                cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A4TmaMmaSchedule<128, 3, 1, CU_TENSOR_MAP_L2_PROMOTION_NONE, kNvfp4A4ResidualTerms>,
        5120>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{output, p.rows / 2},
                               SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}
} // namespace infernix::ops::detail
