// Qwen3.8-Flash-Next shared expert gate and up projection, row-scaled FP8 with BF16 activations (W8A16,
// recipe C). Selected by bench/ops/fp8_flash_next_sweep.cu on the RTX 5090; the GEMV and SIMT routes
// through T = 8 both reduce a column as 16 values per lane in one accumulator chain, so a column has
// the same bits at every width (design §11.3).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<1280, 2560>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
using Simt     = Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                    Fp8SimtBlockOrder::RowsContiguous, 1>;
using RingT64R256 = Fp8A16RingMmaSchedule<64, 256, 2, 8, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, Simt>(x, weight, out, stream);
    if (tokens <= 24) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    if (tokens <= 96) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 2>>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64S3>(x, weight, out, stream);
    if (tokens <= 512) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R128>(x, weight, out, stream);
    if (tokens <= 768) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64S3>(x, weight, out, stream);
    if (tokens <= 1024) return fp8_linear_a16_ring<Geometry, Fp8A16RingT64R128>(x, weight, out, stream);
    if (tokens <= 1536) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64S3>(x, weight, out, stream);
    if (tokens <= 2048) return fp8_linear_a16_ring<Geometry, RingT64R256>(x, weight, out, stream);
    if (tokens <= 3072) return fp8_linear_a16_ring<Geometry, Fp8A16RingT64R128>(x, weight, out, stream);
    fp8_linear_a16_ring<Geometry, RingT64R256>(x, weight, out, stream);
}

// No single-term A8 route: the checkpoint is weight-only, served with BF16 activations. The
// fp8-a8-pair fast path carries them as two E4M3 terms from T = 1025.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
// The pair route's admission (2026-10-15 sweep, RTX 5090): 1.08-2.43 times the exact route up to
// T=1024, 0.69-1.00 beyond.
constexpr std::int32_t kPairMinTokens = 1025;
} // namespace

const Fp8LinearShape kFp8N1280K2560{1280,    2560, launch_a16, nullptr, uses_a8, nullptr,
                                    /*invariant*/ nullptr, launch_fp8_a8_pair<Geometry>, kPairMinTokens};
} // namespace infernix::ops::detail
