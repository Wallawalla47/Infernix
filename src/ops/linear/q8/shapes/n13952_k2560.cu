#include "ops/linear/common/output.cuh"
#include "ops/linear/q8/q8_f16acc.h"
#include "ops/linear/q8/q8_instances.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_stream_launch.cuh"
#include "ops/linear/qsa_group.h"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next MTP attention group: the register-streamed route up to 8 columns (4-28 % faster than
// SIMT), the MMA tiles beyond (M1 sweep). K1: 2 row(s) per warp, x staged in shared memory per pass.
template <int MaxCols, int ColsPerPass>
using Stream = Q8A16StreamSchedule<2560, 2, 1, 8, MaxCols, ColsPerPass, Q8StreamX::Shared, 2>;

// select_q8_n13952_k2560's routes into any output (the QSA group's six segments for linear_qsa_group):
// the output map changes only where a value is stored, so both give the same bits.
template <class Output>
void launch_routes(const Q8LinearOperands& operands, Output output, cudaStream_t stream) {
    const LinearIdentityEpilogue epilogue{};
    const std::int32_t tokens = operands.tokens;
    if (tokens <= 4) return launch_q8_a16_stream<Stream<4, 4>>(operands, output, epilogue, stream);
    if (tokens <= 8) return launch_q8_a16_stream<Stream<8, 8>>(operands, output, epilogue, stream);
    if (tokens <= 64) return launch_q8_a16_mma<q8_instances::MmaR32T64>(operands, output, epilogue, stream);
    if (tokens <= 96) return launch_q8_a16_mma<q8_instances::MmaR32T96>(operands, output, epilogue, stream);
    launch_q8_a16_mma<q8_instances::MmaR32T128>(operands, output, epilogue, stream);
}
} // namespace

Q8Launch select_q8_n13952_k2560(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 8) return launch_q8_a16_stream_linear<Stream<8, 8>>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    return launch_q8_a16_mma_r32_t128;
}

void q8_qsa_group(const Tensor& x, const Weight& w, const QsaGroupParts& parts, LinearRoutes routes,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    if (has_route(routes, LinearRoutes::F16Accumulate) && q8_f16acc_admits(w.n, w.k, x.ne[1]) &&
        w.padded_shape[1] == w.k) {
        if (workspace == nullptr) {
            throw std::invalid_argument("q8 qsa group: the f16-accumulate route requires caller workspace");
        }
        q8_qsa_group_f16acc(x, w, parts, *workspace, stream);
        return;
    }
    const LinearBf16SegmentedOutput<6144, 6144, 512, 512, 512, 128> output{
        {parts.data[0], parts.data[1], parts.data[2], parts.data[3], parts.data[4], parts.data[5]}};
    launch_routes(q8_linear_operands(x, w), output, stream);
}

} // namespace infernix::ops::detail
