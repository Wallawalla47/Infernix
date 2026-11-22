// The generic Q4 linear's LoRA form for an Op that materializes a projection through it (linear_swiglu's
// Materialized route of the [34816,5120] gate/up parent): the instance the release route table runs at the call's
// width (select_q4_a16_launch), instantiated with the term added before its BF16 store (the adapter epilogue of the
// term's class: narrow or wide).

#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_instance_launch.cuh"
#include "ops/lora/lora_device.cuh"

#include <stdexcept>

namespace infernix::ops::detail {
namespace {

template <class Epilogue>
void q4_lora_dispatch_as(const Tensor& x, const Weight& w, Tensor& out, const Epilogue& epilogue,
                         cudaStream_t stream) {
    const auto operands = q4_linear_operands(x, w);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), w.n};
    const Q4Launch release = select_q4_a16_launch(w.n, w.k, x.ne[1]);
    if (release == &launch_q4_a16_sliced_r32_t32_w4_s1) {
        launch_q4_a16_sliced_k_mma<q4_instances::SlicedR32T32W4S1>(operands, output, epilogue, stream);
    } else if (release == &launch_q4_a16_mma_r64_t64_k128_s2_a1) {
        launch_q4_a16_mma<q4_instances::MmaR64T64K128S2A1>(operands, output, epilogue, stream);
    } else if (release == &launch_q4_a16_mma_r64_t80) {
        launch_q4_a16_mma<q4_instances::MmaR64T80K64Wr16Wt40S2A2B1>(operands, output, epilogue, stream);
    } else if (release == &launch_q4_a16_mma_r64_t96) {
        launch_q4_a16_mma<q4_instances::MmaR64T96K64Wr32Wt16S2A2B1>(operands, output, epilogue, stream);
    } else if (release == &launch_q4_a16_mma_r64_t112 && x.ne[1] % 112 == 0) {
        launch_q4_a16_mma<q4_instances::MmaR64T112K64Wr16Wt112S2A2B1>(operands, output, epilogue, stream);
    } else {
        throw std::logic_error("q4 linear: this route has no LoRA form (linear_swiglu's Materialized widths only)");
    }
}

} // namespace

void q4_lora_dispatch(const Tensor& x, const Weight& w, Tensor& out, const LoraAddend& lora, cudaStream_t stream) {
    if (lora_narrow(lora)) {
        q4_lora_dispatch_as(x, w, out,
                            LoraEpilogue<LinearIdentityEpilogue>{{}, lora_device_addend<LoraDeviceAddend>(lora)}, stream);
    } else {
        q4_lora_dispatch_as(
            x, w, out, LoraEpilogueWide<LinearIdentityEpilogue>{{}, lora_device_addend<LoraDeviceAddendWide>(lora)},
            stream);
    }
}

} // namespace infernix::ops::detail
