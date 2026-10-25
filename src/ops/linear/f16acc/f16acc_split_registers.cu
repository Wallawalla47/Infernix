// The split-register f16-accumulate launches (f16acc_mma.cuh F16AccSplitRegisters, chosen per call by
// f16acc_split_registers), each defined once here. This translation unit is in the non-RDC
// archive, where ptxas keeps the producers' setmaxnreg hand-over to the consumers; the routes' own
// translation units see only the declarations at the end of f16acc_mma.cuh.
#include "ops/linear/f16acc/f16acc_mma.cuh"

namespace infernix::ops::detail {

#define INFERNIX_F16ACC_SPLIT_DEFINE(SCHEDULE, SCALES, OUTPUT)                                             \
    template void launch_f16acc_mma<F16AccSplitRegisters<SCHEDULE>, SCALES, __nv_bfloat16, OUTPUT,         \
                                    LinearIdentityEpilogue>(const F16AccOperands<SCALES, __nv_bfloat16>&, \
                                                            OUTPUT, LinearIdentityEpilogue, cudaStream_t);
INFERNIX_F16ACC_SPLIT_LAUNCHES(INFERNIX_F16ACC_SPLIT_DEFINE)
#undef INFERNIX_F16ACC_SPLIT_DEFINE

} // namespace infernix::ops::detail
