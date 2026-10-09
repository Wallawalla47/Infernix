// Qwen3.8-Flash-Next QSA query/gate/key/value projection (the BF16 indexer runs apart), row-scaled FP8 with
// BF16 activations (W8A16, recipe C). Selected by bench/ops/fp8_flash_next_sweep.cu on the RTX 5090; the
// GEMV and SIMT routes through T = 8 both reduce a column as 16 values per lane in one accumulator
// chain, so a column has the same bits at every width (design §11.3).
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<13312, 2560>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
using Simt     = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                    Fp8SimtBlockOrder::RowsContiguous, 1>;
using TmaT64R256  = Fp8A16TmaMmaSchedule<64, 256, 2, 8, 2, 1>;
using TmaT128R128 = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, Simt>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 1>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R128>(x, weight, out, stream);
    if (tokens <= 64) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 96) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R128>(x, weight, out, stream);
    if (tokens <= 128) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 192) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 384) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 512) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 1024) return fp8_linear_a16_tma<Geometry, TmaT64R256>(x, weight, out, stream);
    if (tokens <= 1536) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, TmaT128R128>(x, weight, out, stream);
}

// A16 only: the checkpoint is weight-only, served with BF16 activations.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N13312K2560{13312, 2560, launch_a16, nullptr, uses_a8, nullptr};
} // namespace infernix::ops::detail
