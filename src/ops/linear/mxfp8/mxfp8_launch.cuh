#pragma once

// The plain-linear wrapper of the MXFP8 A16 route (mxfp8_a16_tma_mma.cuh), the one route of every
// width, and the schedules the shapes choose from: the FP8 family's TMA schedules (fp8_schedule.cuh)
// instantiated for a shape's static K (Fp8ScheduleInstance), and 16-token tiles for decode and
// verification calls. Every schedule accumulates a column in the same order, so a shape may change
// schedule at any width without changing bits.

#include "core/device.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_shared.cuh"
#include "ops/linear/mxfp8/mxfp8_a16_tma_mma.cuh"

namespace infernix::ops::detail {

// One 16-token MMA tile per CTA (a T = 1 call streams the weights once): rows per CTA trade the
// grid's size against its re-reads of the activation tile. With the cp.async producer the 16-row
// tile is the output projection's fastest to 12 columns (bench/ops/mxfp8_flash_next_sweep.cu,
// 2026-10-20; under the TMA producer it was 1.2-2.5x slower everywhere).
using Mxfp8TmaT16R16  = Fp8A16RingMmaSchedule<16, 16, 1, 2, 4, 3>;
using Mxfp8TmaT16R32  = Fp8A16RingMmaSchedule<16, 32, 1, 4, 3, 3>;
using Mxfp8TmaT16R64  = Fp8A16RingMmaSchedule<16, 64, 1, 4, 3, 2>;
using Mxfp8TmaT16R128 = Fp8A16RingMmaSchedule<16, 128, 1, 8, 3, 1>;
using Mxfp8TmaT32R32  = Fp8A16RingMmaSchedule<32, 32, 1, 4, 4, 2>;
using Mxfp8TmaT64R64  = Fp8A16RingMmaSchedule<64, 64, 2, 4, 2, 2>;
using Mxfp8TmaT64R256 = Fp8A16RingMmaSchedule<64, 256, 2, 8, 2, 1>;
using Mxfp8TmaT128R128 = Fp8A16RingMmaSchedule<128, 128, 4, 4, 2, 1>;

// A schedule's cp.async producer spread over two or four warps (Mxfp8ProducerWarps): a full 16-token
// tile and the prefill tiles issue more copies per stage than one warp keeps in flight.
template <class Schedule>
using Mxfp8P2 = Mxfp8ProducerWarps<Schedule, 2>;
template <class Schedule>
using Mxfp8P4 = Mxfp8ProducerWarps<Schedule, 4>;
// A schedule staged by TMA tensor loads (Mxfp8TmaProducer), the faster producer of prefill widths.
template <class Schedule>
using Mxfp8Tma = Mxfp8TmaProducer<Schedule>;

// One schedule into any linear output (a plain [N,T] tensor, or segments of the rows).
template <class Geometry, class Schedule, class Output>
void mxfp8_tma(const Mxfp8Operands& p, Output output, cudaStream_t stream) {
    launch_mxfp8_a16_tma_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(p, output, LinearIdentityEpilogue{},
                                                                                 stream);
}

template <class Geometry, class Schedule>
void mxfp8_linear_a16_tma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    mxfp8_tma<Geometry, Schedule>(mxfp8_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), w.n},
                                  stream);
}

} // namespace infernix::ops::detail
