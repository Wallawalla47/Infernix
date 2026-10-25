// Qwen3.8-Flash-Next shared expert down projection, row-scaled FP8 with BF16 activations (W8A16, recipe C).
// K = 640 is not a whole number of the SIMT, GEMV or sliced-K tiles, so every width takes the ring tensor-core
// tiles; selected by bench/ops/fp8_flash_next_sweep.cu on the RTX 5090. All of them give one column's bits at
// every width (design §11.3).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"
#include "ops/linear/f16acc/f16acc_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry    = Fp8Geometry<2560, 640>;
using RingT64R256  = Fp8A16RingMmaSchedule<64, 256, 2, 8, 2, 1>;
using RingT128R128 = Fp8A16RingMmaSchedule<128, 128, 4, 4, 2, 1>;
using MmaR64T128K64 = Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 9) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64S3>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64>(x, weight, out, stream);
    if (tokens <= 48) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64S3>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64>(x, weight, out, stream);
    if (tokens <= 384) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64S3>(x, weight, out, stream);
    if (tokens <= 512) return fp8_linear_a16_ring<Geometry, Fp8A16RingT64R128>(x, weight, out, stream);
    if (tokens <= 1024) return fp8_linear_a16_ring<Geometry, RingT64R256>(x, weight, out, stream);
    if (tokens <= 2048) return fp8_linear_a16_mma<Geometry, MmaR64T128K64>(x, weight, out, stream);
    fp8_linear_a16_ring<Geometry, RingT128R128>(x, weight, out, stream);
}

// No single-term A8 route: the checkpoint is weight-only, served with BF16 activations. The
// fp8-a8-pair fast path carries them as two E4M3 terms from T = 513 (five 128-byte K
// tiles per stage loop).
bool uses_a8(std::int32_t, std::int32_t) { return false; }
// The pair route's admission (2026-10-15 sweep, RTX 5090): 1.03-1.61 times the exact route up to
// T=512, 0.73-0.90 beyond.
constexpr std::int32_t kPairMinTokens = 513;

// f16-accumulate-prefill: the exact route's ring tiles at the same widths (its 1,025-2,048-column
// tiles are not ring tiles; not yet swept for this route).
void launch_f16(const Tensor& x, const Weight& weight, Tensor& out, F16AccumulateWorkspace scratch,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 384) return fp8_linear_f16acc<Geometry, Fp8A16RingT32R64S3>(x, weight, out, scratch, stream);
    if (tokens <= 512) return fp8_linear_f16acc<Geometry, Fp8A16RingT64R128>(x, weight, out, scratch, stream);
    if (tokens <= 2048) return fp8_linear_f16acc<Geometry, RingT64R256>(x, weight, out, scratch, stream);
    fp8_linear_f16acc<Geometry, RingT128R128>(x, weight, out, scratch, stream);
}
} // namespace

const Fp8LinearShape kFp8N2560K640{2560,    640, launch_a16, nullptr, uses_a8, nullptr,
                                   /*invariant*/ nullptr, launch_fp8_a8_pair<Geometry>, kPairMinTokens,
                                   launch_fp8_a8_pair_block32<Geometry>, launch_f16, kFp8F16MinTokens};
} // namespace infernix::ops::detail
