#include "ops/linear/q5/q5_instances.cuh"
#include "ops/linear/q4/q4_instances.cuh"
#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/q4/q4_simt_launch.cuh"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q5/q5_simt_launch.cuh"
#include "ops/linear/q5/q5_gemv_launch.cuh"
#include "ops/linear/q5/q5_sliced_k_launch.cuh"
#include "ops/lora/lora_device.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kHidden    = 5120;
// The parents' rows in the term's logical order [q, k, value, z]: the Q4 parent's first, the Q5 parent's after.
constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueZRows = 12288;

using Q4GdnSimtR8T4Schedule = Q4A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8T8Schedule = Q4A16SimtSchedule<8, 8, 1, 16, 2, Cache::ca, 1>;

auto q4_projection_output(Tensor& out) {
    return LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(out.data),
                                   static_cast<std::int64_t>(out.nb[1] / sizeof(__nv_bfloat16)), 0};
}

// `lora` below: the parent's own term (lora_row_range of the call's), or null.
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                    const LoraAddend* lora) {
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_q4_a16_gemv<q4_instances::GemvR1W8K5120>(q4_linear_operands(x, weight), q4_projection_output(out),
                                                        epilogue, stream);
    });
}

template <class Schedule>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                    const LoraAddend* lora) {
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_q4_a16_simt<Schedule>(q4_linear_operands(x, weight), q4_projection_output(out), epilogue, stream);
    });
}

template <std::int32_t Capacity>
void launch_q4_sliced_exact(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream, const LoraAddend* lora) {
    using Schedule = Q4A16SlicedKMmaSchedule<16, (Capacity + 7) / 8 * 8, 8, 1, Cache::cg, Cache::ca,
                                             6, kHidden, Capacity>;
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_q4_a16_sliced_k_mma<Schedule>(q4_linear_operands(x, weight), q4_projection_output(out), epilogue,
                                             stream);
    });
}

void launch_q4_sliced_band(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream, const LoraAddend* lora) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("Q4/Q5 GDN K-split requires padded K == hidden");
    }
    switch (x.ne[1]) {
    case 7:
        launch_q4_sliced_exact<7>(x, weight, out, stream, lora);
        return;
    case 8:
        launch_q4_sliced_exact<8>(x, weight, out, stream, lora);
        return;
    case 9:
        launch_q4_sliced_exact<9>(x, weight, out, stream, lora);
        return;
    case 10:
        launch_q4_sliced_exact<10>(x, weight, out, stream, lora);
        return;
    case 11:
        launch_q4_sliced_exact<11>(x, weight, out, stream, lora);
        return;
    case 12:
        launch_q4_sliced_exact<12>(x, weight, out, stream, lora);
        return;
    default:
        throw std::invalid_argument("Q4/Q5 GDN K-split band covers T in [7,12]");
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
               const LoraAddend* lora) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv(x, weight, out, stream, lora);
        return;
    case 2:
    case 3:
    case 4:
        launch_q4_simt<Q4GdnSimtR8T4Schedule>(x, weight, out, stream, lora);
        return;
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
        // The K-split MMA arms all 8 warps of a CTA onto K instead of waiting out the weight stream
        // of a row. Complete-op measurement (both parents, one graph, one probe run per column
        // count) at T=9..12: the split form with this parent is 101.6-105.7 us against 120.1 us for
        // the grouped kernel R6 chose, while at T=13 the grouped kernel wins again (120.1 against
        // 126.2), so the band ends at 12.
        launch_q4_sliced_band(x, weight, out, stream, lora);
        return;
    default:
        if (x.ne[1] <= 15) {
            launch_q4_simt<Q4GdnSimtR8T8Schedule>(x, weight, out, stream, lora);
            return;
        }
        throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
    }
}

auto q5_projection_output(Tensor& first, Tensor& second) {
    return LinearBf16SplitOutput2<kValueRows>{
        {static_cast<__nv_bfloat16*>(first.data), std::int64_t(first.nb[1] / sizeof(__nv_bfloat16)),
         0},
        {static_cast<__nv_bfloat16*>(second.data),
         std::int64_t(second.nb[1] / sizeof(__nv_bfloat16)), 0}};
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream, const LoraAddend* lora) {
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_q5_a16_gemv<q5_instances::GemvR16W1G16S2XK5120>(q5_linear_operands(x, weight),
                                                               q5_projection_output(value, z), epilogue, stream);
    });
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream, const LoraAddend* lora) {
    using Schedule = Q5A16DirectSimtSchedule<1, Cols, 4, 4, 10, kHidden, true>;
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_q5_a16_direct_simt<Schedule>(q5_linear_operands(x, weight), q5_projection_output(value, z), epilogue,
                                            stream);
    });
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                            cudaStream_t stream, const LoraAddend* lora) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, value, z, stream, lora);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, value, z, stream, lora);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, value, z, stream, lora);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, value, z, stream, lora);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, value, z, stream, lora);
        return;
    case 7:
        launch_q5_split4<7>(x, weight, value, z, stream, lora);
        return;
    case 8:
        launch_q5_split4<8>(x, weight, value, z, stream, lora);
        return;
    case 9:
        launch_q5_split4<9>(x, weight, value, z, stream, lora);
        return;
    case 10:
        launch_q5_split4<10>(x, weight, value, z, stream, lora);
        return;
    default:
        throw std::invalid_argument("GDN Q5 split4 requires T in [2,10]");
    }
}

template <int kColsPerTile>
void launch_q5_simt_cols(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                         cudaStream_t stream, const LoraAddend* lora) {
    using Schedule = Q5A16SimtSchedule<8, kColsPerTile, 1, 16, 2, Cache::ca, 1>;
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_q5_a16_simt<Schedule>(q5_linear_operands(x, weight), q5_projection_output(value, z), epilogue,
                                     stream);
    });
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
               cudaStream_t stream, const LoraAddend* lora) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, value, z, stream, lora);
        return;
    }
    if (x.ne[1] <= 10) {
        // Split4: one CTA owns one output row, its four warps split the K dimension and reduce
        // their partial sums through shared memory. The column count is a compile-time template
        // argument, so the kernel covers exactly the live columns. The fused projections below
        // (T=2..6) use the same shape, so the Q5 parent has one mechanism from 2 to 10; the band
        // end is the measured crossover against the c4 tile at 11..12, not a shape limit.
        launch_q5_split4_exact(x, weight, value, z, stream, lora);
        return;
    }
    if (x.ne[1] <= 12) {
        // c4 SIMT: one output row per warp, up to four columns per column tile, with the quantized
        // weight planes staged in shared memory and activations read from the input tensor.
        // Retained in this interval on complete-Op measurements; both shapes are legal at every
        // count in [2,15].
        launch_q5_simt_cols<4>(x, weight, value, z, stream, lora);
        return;
    }
    if (x.ne[1] <= 15) {
        launch_q5_simt_cols<8>(x, weight, value, z, stream, lora);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
}

// No term only: the Q4 launch joins the Q5 producer (JoinPdl), and a joining launch reads its inputs before its
// wait, which a term's U may not be read before; a term runs the same two schedules as separate launches.
void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Q4Schedule = Q4GdnSimtR8T4Schedule;
    using Q5Schedule = Q5A16DirectSimtSchedule<1, 4, 4, 4, 10, kHidden, true>;
    // Q5 and Q4 publish disjoint rows; Q4 joins the Q5 producer at kernel exit.
    launch_q5_a16_direct_simt<Q5Schedule, true>(q5_linear_operands(x, value_z_weight),
                                                q5_projection_output(value, z),
                                                LinearIdentityEpilogue{}, stream);
    launch_q4_a16_simt<Q4Schedule, false, true, true>(
        q4_linear_operands(x, qk_weight),
        LinearBf16StridedOutput{static_cast<__nv_bfloat16*>(qk.data),
                                static_cast<std::int64_t>(qk.nb[1] / sizeof(__nv_bfloat16)), 0},
        LinearIdentityEpilogue{}, stream);
}

// The column-invariant routes. Each parent runs one sliced-K reduction at every width: the Q4
// parent the exact route's 8 K-warp linear schedule of T=7..12 (token capacities 1..12, then 16,
// 24 and 32 with wider calls sliced by 32), the Q5 parent the 4 K-warp linear schedule of
// linear_add's exact route. Capacity, block rows, stages and token slicing change only staging and
// predication, so a column's bits do not depend on the call's width.
void launch_q4_invariant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream,
                         const LoraAddend* lora) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("Q4/Q5 GDN K-split requires padded K == hidden");
    }
    const std::int32_t cols = x.ne[1];
    if (cols <= 1) return launch_q4_sliced_exact<1>(x, weight, out, stream, lora);
    if (cols <= 2) return launch_q4_sliced_exact<2>(x, weight, out, stream, lora);
    if (cols <= 4) return launch_q4_sliced_exact<4>(x, weight, out, stream, lora);
    if (cols <= 6) return launch_q4_sliced_exact<6>(x, weight, out, stream, lora);
    if (cols <= 12) return launch_q4_sliced_band(x, weight, out, stream, lora);
    if (cols <= 16) return launch_q4_sliced_exact<16>(x, weight, out, stream, lora);
    if (cols <= 24) return launch_q4_sliced_exact<24>(x, weight, out, stream, lora);
    launch_q4_sliced_exact<32>(x, weight, out, stream, lora);
}

template <class Schedule>
void launch_q5_sliced(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream, const LoraAddend* lora) {
    static_assert(Schedule::kKWarps == 4 && Schedule::kReduction == Q5SlicedKReduction::Linear);
    with_lora_epilogue(lora, LinearIdentityEpilogue{}, [&](auto epilogue) {
        launch_q5_a16_sliced_k_mma<Schedule>(q5_linear_operands(x, weight), q5_projection_output(value, z),
                                             epilogue, stream);
    });
}

void launch_q5_invariant(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                         cudaStream_t stream, const LoraAddend* lora) {
    const std::int32_t cols = x.ne[1];
    if (cols <= 8) return launch_q5_sliced<q5_instances::SlicedR16T8W4S2>(x, weight, value, z, stream, lora);
    if (cols <= 16) return launch_q5_sliced<q5_instances::SlicedR16T16W4S2>(x, weight, value, z, stream, lora);
    if (cols <= 24) return launch_q5_sliced<q5_instances::SlicedR16T24W4S2>(x, weight, value, z, stream, lora);
    launch_q5_sliced<q5_instances::SlicedR32T32W4S2>(x, weight, value, z, stream, lora);
}

// Each parent's term: the call's rows [0, 4096) (q, k) and [4096, 16384) (value, z).
struct ParentTerms {
    LoraAddend first, second;
    const LoraAddend* first_ptr  = nullptr;
    const LoraAddend* second_ptr = nullptr;
};

ParentTerms parent_terms(const LoraAddend* lora) {
    ParentTerms terms;
    if (lora != nullptr && lora->present()) {
        terms.first      = lora_row_range(*lora, 0, kQkRows);
        terms.second     = lora_row_range(*lora, kQkRows, kValueZRows);
        terms.first_ptr  = &terms.first;
        terms.second_ptr = &terms.second;
    }
    return terms;
}

} // namespace

void q4_q5_gdn_input_column_invariant_launch(const Tensor& x, const Weight& qk_weight,
                                             const Weight& value_z_weight, Tensor& qk,
                                             Tensor& value, Tensor& z, cudaStream_t stream,
                                             const LoraAddend* lora) {
    if (x.ne[1] < 1) {
        throw std::invalid_argument("Q4/Q5 GDN column-invariant route requires T >= 1");
    }
    const ParentTerms terms = parent_terms(lora);
    launch_q4_invariant(x, qk_weight, qk, stream, terms.first_ptr);
    launch_q5_invariant(x, value_z_weight, value, z, stream, terms.second_ptr);
}

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream, const LoraAddend* lora) {
    const ParentTerms terms = parent_terms(lora);
    if (x.ne[1] == 4 && terms.first_ptr == nullptr) {
        launch_t4_pdl(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    launch_q4(x, qk_weight, qk, stream, terms.first_ptr);
    launch_q5(x, value_z_weight, value, z, stream, terms.second_ptr);
}

} // namespace infernix::ops::detail
