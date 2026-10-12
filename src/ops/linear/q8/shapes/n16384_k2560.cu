#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next GDN q/k/v/z projection (packed): the register-streamed route up to 8 columns (5-28 %
// faster than SIMT), the MMA tiles beyond (M1 sweep). K1: 2 row(s) per warp, x staged in shared memory per
// pass.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<2560, 2, 1, 8, MaxCols, ColsPerPass, Q8StreamX::Shared, 2>;

// One route table for both outputs: the packed [16384,T] linear result, and the q/k/v rows and z rows
// written straight to their two destinations (gdn_input_proj). The output map changes only where a
// value is stored, so both produce the same bits.
template <class Output>
void launch_routes(const Q8LinearOperands& operands, Output output, cudaStream_t stream) {
    const LinearIdentityEpilogue epilogue{};
    const std::int32_t tokens = operands.tokens;
    if (tokens <= 4) return launch_q8_a16_stream<Stream<4, 4>>(operands, output, epilogue, stream);
    if (tokens <= 8) return launch_q8_a16_stream<Stream<8, 8>>(operands, output, epilogue, stream);
    if (tokens <= 64) return launch_q8_a16_mma<q8_instances::MmaR32T64>(operands, output, epilogue, stream);
    if (tokens <= 96) return launch_q8_a16_mma<q8_instances::MmaR32T96>(operands, output, epilogue, stream);
    // Beyond 96 columns (prefill chunks): the MMA tile sweep of 2026-10-05 (fn/rigs/layer/q8tiles2.bat,
    // RTX 5090). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical (checked at every swept point); only speed differs.
    launch_q8_a16_mma<q8_instances::MmaR64T128>(operands, output, epilogue, stream);
}

void launch_linear(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_routes(q8_linear_operands(x, weight),
                  LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, stream);
}
} // namespace

Q8Launch select_q8_n16384_k2560(std::int32_t) { return launch_linear; }

void launch_q8_n16384_k2560_split(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  cudaStream_t stream) {
    launch_routes(q8_linear_operands(x, weight),
                  LinearBf16SegmentedOutput<10240, 6144>{
                      {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}},
                  stream);
}

} // namespace infernix::ops::detail
