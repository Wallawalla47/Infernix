// f16-accumulate-prefill on Q8 (q8_f16acc.h): Qwen3.8-Flash-Next Dense8's dense problems on the FP16-accumulate
// GEMM with FP16 group scales. The tiles are the row-scaled FP8 route's (cp.async rings; the consumers read
// the group scales from global memory), not yet swept for Q8.
#include "ops/linear/q8/q8_f16acc.h"

#include "core/temporary_switches.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/f16acc/f16acc_mma.cuh"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/qsa_group.h"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

using RingT128R128 = Fp8A16RingMmaSchedule<128, 128, 4, 4, 2, 1>;

template <std::int32_t N, std::int32_t K, class Output>
void run(const Tensor& x, const Weight& weight, Output output, F16AccumulateWorkspace scratch, cudaStream_t stream) {
    using Geometry = Fp8Geometry<N, K>;
    if (x.ne[1] <= 3072) return q8_f16acc<Geometry, Fp8A16RingT64R128>(x, weight, output, scratch, stream);
    q8_f16acc<Geometry, RingT128R128>(x, weight, output, scratch, stream);
}

template <class Output>
void dispatch(const Tensor& x, const Weight& weight, Output output, F16AccumulateWorkspace scratch,
              cudaStream_t stream) {
    const auto shape = [&](std::int32_t n, std::int32_t k) { return weight.n == n && weight.k == k; };
    if (shape(16384, 2560)) return run<16384, 2560>(x, weight, output, scratch, stream);
    if (shape(13952, 2560)) return run<13952, 2560>(x, weight, output, scratch, stream);
    if (shape(12800, 2560)) return run<12800, 2560>(x, weight, output, scratch, stream);
    if (shape(2560, 2560)) return run<2560, 2560>(x, weight, output, scratch, stream);
    if (shape(1280, 2560)) return run<1280, 2560>(x, weight, output, scratch, stream);
    if (shape(2560, 6144)) return run<2560, 6144>(x, weight, output, scratch, stream);
    if (shape(2560, 640)) return run<2560, 640>(x, weight, output, scratch, stream);
    throw std::invalid_argument("q8 f16-accumulate: unregistered problem");
}

void require_route(const Tensor& x, const Weight& weight) {
    if (weight.qtype != QType::Q8_G32_FP16 || !q8_f16acc_problem(weight.n, weight.k) || x.ne[1] < kQ8F16MinTokens ||
        weight.padded_shape[1] != weight.k) {
        throw std::invalid_argument("q8 f16-accumulate: the call is outside the route's problems or widths");
    }
}

} // namespace

bool q8_f16acc_problem(std::int32_t n, std::int32_t k) noexcept {
    return (k == 2560 && (n == 16384 || n == 13952 || n == 12800 || n == 2560 || n == 1280)) ||
           (n == 2560 && (k == 6144 || k == 640));
}

bool q8_f16acc_admits(std::int32_t n, std::int32_t k, std::int32_t tokens) noexcept {
    if (!q8_f16acc_problem(n, k) || tokens < kQ8F16MinTokens) { return false; }
    static const bool measured = temporary_switches::flag("INFERNIX_TMP_Q8_F16ACC_ADMIT");
    if (measured && tokens > 8192) {
        return (n == 13952 && k == 2560) || (n == 2560 && (k == 2560 || k == 640));
    }
    return true;
}

std::size_t q8_f16acc_workspace_capacity_bytes(std::int32_t n, std::int32_t k, std::int32_t max_tokens) {
    return q8_f16acc_problem(n, k) && max_tokens >= kQ8F16MinTokens
               ? f16_accumulate_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void q8_linear_f16acc(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                      cudaStream_t stream) {
    require_route(x, weight);
    auto scope = workspace.scope();
    dispatch(x, weight, LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n},
             allocate_f16_accumulate_workspace(workspace, x.ne[1], weight.k), stream);
}

void q8_gdn_input_f16acc(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    require_route(x, weight);
    if (weight.n != 16384 || weight.k != 2560) { throw std::invalid_argument("q8 f16-accumulate gdn_input_proj: shape"); }
    auto scope = workspace.scope();
    const LinearBf16SegmentedOutput<10240, 6144> output{
        {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}};
    run<16384, 2560>(x, weight, output, allocate_f16_accumulate_workspace(workspace, x.ne[1], weight.k), stream);
}

void q8_qsa_group_f16acc(const Tensor& x, const Weight& weight, const QsaGroupParts& parts, WorkspaceArena& workspace,
                         cudaStream_t stream) {
    require_route(x, weight);
    if (weight.n != 13952 || weight.k != 2560) { throw std::invalid_argument("q8 f16-accumulate qsa group: shape"); }
    auto scope = workspace.scope();
    const LinearBf16SegmentedOutput<6144, 6144, 512, 512, 512, 128> output{
        {parts.data[0], parts.data[1], parts.data[2], parts.data[3], parts.data[4], parts.data[5]}};
    run<13952, 2560>(x, weight, output, allocate_f16_accumulate_workspace(workspace, x.ne[1], weight.k), stream);
}

} // namespace infernix::ops::detail
