// Qwen3.8-Flash-Next shared expert down projection, row-scaled FP8 with BF16 activations (W8A16, recipe C).
// K = 640 is not a whole number of the SIMT, GEMV or sliced-K tiles, so every width takes the TMA tensor-core
// tiles; selected by bench/ops/fp8_flash_next_sweep.cu on the RTX 5090. All of them give one column's bits at
// every width (design §11.3).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry    = Fp8Geometry<2560, 640>;
using TmaT64R256  = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>;
using TmaT128R128 = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>;
using MmaR64T128K64 = Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 9) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64>(x, weight, out, stream);
    if (tokens <= 48) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64>(x, weight, out, stream);
    if (tokens <= 384) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 512) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 1024) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 2048) return fp8_linear_a16_mma<Geometry, MmaR64T128K64>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, TmaT128R128>(x, weight, out, stream);
}

// A16 only: the checkpoint is weight-only, served with BF16 activations.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N2560K640{2560, 640, launch_a16, nullptr, uses_a8, nullptr};
} // namespace infernix::ops::detail
