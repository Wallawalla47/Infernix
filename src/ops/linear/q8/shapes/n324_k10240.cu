#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next hyper-connection down + inject: the register-streamed route at 1-3 and 9-64 columns (2-7x
// faster than the MMA tile there), the few-row SIMT route at 4-8 (M1 sweep). K1: 2 row(s) per warp, 8 warps
// splitting each row's K, x read through L1.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<10240, 2, 8, 8, MaxCols, ColsPerPass, Q8StreamX::Global, 2>;
} // namespace

Q8Launch select_q8_n324_k10240(std::int32_t tokens) {
    if (tokens <= 3) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 4) return launch_q8_a16_simt_r1_t4_w8;
    if (tokens <= 8) return launch_q8_a16_simt_r1_t8_w8;
    if (tokens <= 64) return launch_q8_a16_stream_linear<Stream<16, 8>>;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    // Beyond 96 columns (prefill chunks): the MMA tile sweep of 2026-10-05 (fn/rigs/layer/q8tiles2.bat,
    // RTX 5090). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical (checked at every swept point); only speed differs.
    // Past 1,024 columns the tile follows the SM waves: the r48_t64 tile everywhere lost up to 20 % where its last
    // wave was nearly empty (candidates NQ-08). infernix_linear_bench --qtype q8 --n 324 --k 10240, cold L2, two
    // interleaved rounds every 64 columns from 1,088 to 16,384 over r32_t64 / r48_t64 / r64_t96 / r64_t128 /
    // r96_t96 (a first pass at 512-column steps also had r32_t96, r32_t128 and r128_t64, never best): a tile takes
    // an interval where it beat r48_t64 by 2 % or more, the seams at the last measured column; the sweep's summed
    // time x0.922, the intervals' gains 3-20 % (2026-11-04).
    if (tokens <= 1920) return launch_q8_a16_mma_r32_t64;   // -5.3..-9.0 % past 1,024
    if (tokens <= 3072) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 3584) return launch_q8_a16_mma_r64_t128;  // -15.8..-16.7 %
    if (tokens <= 4032) return launch_q8_a16_mma_r96_t96;   // -2.7..-3.8 %
    if (tokens <= 4608) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 5376) return launch_q8_a16_mma_r64_t96;   // -16.7..-19.7 %
    if (tokens <= 7168) return launch_q8_a16_mma_r64_t128;  // -6.5..-14.4 %
    if (tokens <= 8128) return launch_q8_a16_mma_r96_t96;   // -3.4..-15.2 %
    if (tokens <= 9280) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 10880) return launch_q8_a16_mma_r64_t128; // -8.8..-11.9 %
    if (tokens <= 12160) return launch_q8_a16_mma_r96_t96;  // -3.8..-6.3 %
    if (tokens <= 12416) return launch_q8_a16_mma_r48_t64;
    if (tokens <= 14464) return launch_q8_a16_mma_r64_t128; // -6.8..-15.6 %
    if (tokens <= 16320) return launch_q8_a16_mma_r96_t96;  // -4.8..-10.1 %
    return launch_q8_a16_mma_r64_t128;                      // 16,384: -4.6 %
}

namespace {
// The q8-sliced-k fast path's routes past 64 columns (candidates NQ-01): sliced-K MMA tiles, whose warps split each
// tile's K and sum their partials, so the 21 row tiles fill the GPU where the exact MMA tiles' 11 run 22-44 CTAs.
template <class Schedule>
void hc_sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Instance = typename Schedule::template with_problem<10240, Schedule::kBlockTokens>;
    launch_q8_a16_sliced_k_mma<Instance>(q8_linear_operands(x, weight),
                                         LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                         LinearIdentityEpilogue{}, stream, {}, pdl::Dependency::Programmatic);
}
using S16W8   = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared>;
using S32W4   = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared>;
using S48W4   = Q8A16SlicedKMmaSchedule<48, 4, 1, 1, Q8ScaleAccess::Shared>;
using S32W4R2 = Q8A16SlicedKMmaSchedule<32, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::ActiveOnly, 0, 32, false, 2>;
} // namespace

// Sweep 2026-11-03 (infernix_linear_bench --routes fast, RTX 5090, cold L2, two runs, 64-column steps): against the
// exact routes 128 columns 98.9 -> 20.9 us, 256 96.9 -> 31.7, 512 96.9 -> 50.3, 768 97.4 -> 62.5, 1,024 101.5 ->
// 90.8, 1,280 115.3 -> 107.1; from 1,408 columns no sliced route beat the exact ones (the 32-row tiles past
// 340 CTAs take a second wave). Dense8 prefill (BF16 KV, -p N -n 1, two rotated runs, both shapes' routes against the
// exact ones): pp128 +0.7 %, pp256 +0.5 %, pp512 -0.3 %, pp1024 0.0 %, pp2048 +1.2 %, every difference inside the
// runs' spread (the expert stream bounds those prompts); kept by the op gain.
Q8Launch select_q8_n324_k10240_fast(std::int32_t tokens) {
    if (tokens <= 64) return select_q8_n324_k10240(tokens);
    if (tokens <= 256) return hc_sliced<S16W8>;
    if (tokens <= 512) return hc_sliced<S32W4>;
    if (tokens <= 960) return hc_sliced<S32W4R2>;
    if (tokens <= 1152) return hc_sliced<S48W4>;
    if (tokens <= 1344) return hc_sliced<S32W4R2>;
    return select_q8_n324_k10240(tokens);
}

} // namespace infernix::ops::detail
