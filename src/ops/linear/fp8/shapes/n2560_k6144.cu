// Qwen3.8-Flash-Next GDN output and QSA output projections, row-scaled FP8 with BF16 activations (W8A16,
// recipe C). Selected by bench/ops/fp8_flash_next_sweep.cu on the RTX 5090; the GEMV and SIMT routes
// through T = 8 both reduce a column as 16 values per lane in one accumulator chain, so a column has
// the same bits at every width (design §11.3).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"
#include "ops/linear/fp8/fp8_a8_pair_tma_mma.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<2560, 6144>;
using Gemv     = Fp8A16GemvSchedule<8, 1, 16, 1, Fp8CodeCache::Default, 1, 2>;
using Simt     = Fp8A16SimtSchedule<4, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                    Fp8SimtBlockOrder::RowsContiguous, 1>;
using RingT128R128 = Fp8A16RingMmaSchedule<128, 128, 4, 4, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, Simt>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    if (tokens <= 24) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 8, 2>>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64S3>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R128>(x, weight, out, stream);
    if (tokens <= 384) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R64S3>(x, weight, out, stream);
    if (tokens <= 512) return fp8_linear_a16_ring<Geometry, Fp8A16RingT64R128>(x, weight, out, stream);
    if (tokens <= 768) return fp8_linear_a16_ring<Geometry, Fp8A16RingT32R128>(x, weight, out, stream);
    if (tokens <= 1024) return fp8_linear_a16_ring<Geometry, RingT128R128>(x, weight, out, stream);
    if (tokens <= 1536) return fp8_linear_a16_ring<Geometry, Fp8A16RingT64R128>(x, weight, out, stream);
    fp8_linear_a16_ring<Geometry, RingT128R128>(x, weight, out, stream);
}

// No single-term A8 route: the checkpoint is weight-only, served with BF16 activations. The
// fp8-a8-pair fast path carries them as two E4M3 terms from T = 513.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
// The pair route's admission (2026-10-15 sweep, RTX 5090): 1.05-1.97 times the exact route up to
// T=512, 0.64-1.00 beyond except T=1040-1088 (1.18-1.19).
constexpr std::int32_t kPairMinTokens = 513;
} // namespace

const Fp8LinearShape kFp8N2560K6144{2560,    6144, launch_a16, nullptr, uses_a8, nullptr,
                                    /*invariant*/ nullptr, launch_fp8_a8_pair<Geometry>, kPairMinTokens};
} // namespace infernix::ops::detail
