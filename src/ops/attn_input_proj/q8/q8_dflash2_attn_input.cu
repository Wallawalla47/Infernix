#include "ops/linear/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/attn_input_proj/q8/q8_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace infernix::ops::detail {
namespace {

// The DFlash2 drafters' Q/K/V parent [6144, H]: H = 5120 (Qwen3.8-27B) or 2560
// (Qwen3.8-Flash-Next).
constexpr std::int32_t kQueryRows       = 4096;
constexpr std::int32_t kKvRows          = 1024;
constexpr std::int32_t kLastSmallTokens = 48;
using Output                            = LinearBf16SegmentedOutput<kQueryRows, kKvRows, kKvRows>;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, cudaStream_t);

// Exact input-load unrolling matters in the first two MMA column tiles. Wider blocks use
// runtime live columns with one specialization per accumulator capacity. The narrowest columns
// split K across 16 warps where H divides into 1024-wide blocks (5120), 8 otherwise (2560).
template <int Hidden, int Columns, bool Exact>
void launch_small(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                  cudaStream_t stream) {
    using Geometry         = Q8LinearGeometry<6144, Hidden>;
    constexpr int Capacity = (Columns + 7) / 8 * 8;
    constexpr int Wide     = Hidden % 1024 == 0 ? 16 : 8;
    constexpr int Warps    = Columns <= 4 ? Wide : Columns <= 16 ? 8 : 4;
    constexpr auto Scales  = Columns <= 4 || (Columns > 8 && Columns <= 16) ? Q8ScaleAccess::Direct
                                                                            : Q8ScaleAccess::Shared;
    using Schedule         = Q8A16SlicedKMmaSchedule<Capacity, Warps, 1, 2, Scales>;
    static_assert((kQueryRows % Schedule::kBlockRows) == 0);
    static_assert((kKvRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);

    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, Columns, Exact>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream, {}, pdl::Dependency::Programmatic);
    CUDA_CHECK(cudaGetLastError());
}

template <int Hidden, bool Exact, std::size_t... I>
constexpr auto make_small_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{
        &launch_small<Hidden,
                      Exact ? 1 + static_cast<int>(I) : 24 + 8 * static_cast<int>(I), Exact>...};
}

template <int Hidden>
struct SmallLaunchers {
    static constexpr auto exact = make_small_launchers<Hidden, true>(std::make_index_sequence<16>{});
    static constexpr auto tile  = make_small_launchers<Hidden, false>(std::make_index_sequence<4>{});
};

template <int Hidden>
void small_t(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
             cudaStream_t stream) {
    if (x.ne[1] <= 16)
        SmallLaunchers<Hidden>::exact[x.ne[1] - 1](x, weight, q, k, v, stream);
    else
        SmallLaunchers<Hidden>::tile[(x.ne[1] - 17) / 8](x, weight, q, k, v, stream);
}

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k, Tensor& v,
                cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
                        static_cast<__nv_bfloat16*>(v.data)};
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                stream);
}

} // namespace

void q8_dflash2_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                          Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > kLastSmallTokens) {
        throw std::invalid_argument("Q8 DFlash2 attention input small-T: unsupported T");
    }
    switch (x.ne[0]) {
    case 5120:
        return small_t<5120>(x, weight, q, k, v, stream);
    case 2560:
        return small_t<2560>(x, weight, q, k, v, stream);
    }
    throw std::invalid_argument("Q8 DFlash2 attention input small-T: unsupported H");
}

void q8_dflash2_attn_input_mma_r32_c64_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                              Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
    launch_mma<Schedule>(x, weight, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r64_c128_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                               Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
    launch_mma<Schedule>(x, weight, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r16_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<16, 64, 128, 16, 16, 1, 1>;
    // This route owns only the partial 49..63-column tile.
    launch_mma<Schedule>(x, w, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r32_c32_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    using Schedule = Q8A16MmaSchedule<32, 32, 128, 16, 16, 1, 1>;
    launch_mma<Schedule>(x, w, q, k, v, stream);
}

void q8_dflash2_attn_input_mma_r32_c64_k128_launch(const Tensor& x, const Weight& w, Tensor& q,
                                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    // Three-block launch bounds reduce register usage and keep all 384 decode CTAs in one wave.
    using Schedule = Q8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>;
    launch_mma<Schedule>(x, w, q, k, v, stream);
}

} // namespace infernix::ops::detail
