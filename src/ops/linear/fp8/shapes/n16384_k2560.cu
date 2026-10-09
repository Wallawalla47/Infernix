// Qwen3.8-Flash-Next GDN query/key/value/z projection, row-scaled FP8 with BF16 activations (W8A16, recipe C).
// Selected by bench/ops/fp8_flash_next_sweep.cu on the RTX 5090. The GEMV at T = 1 and the SIMT tile
// through T = 8 both reduce a column as 16 values per lane in one accumulator chain, so a column has
// the same bits at every width (column invariance, design §11.3); the tensor-core tiles beyond.
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

namespace infernix::ops::detail {
namespace {
using Geometry = Fp8Geometry<16384, 2560>;
using Gemv     = Fp8A16GemvSchedule<8, 2, 16, 1, Fp8CodeCache::Default, 2, 2>;
using Simt     = Fp8A16SimtSchedule<4, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Streaming, 1,
                                    Fp8SimtBlockOrder::RowsContiguous, 1>;
using TmaT128R128 = Fp8A16TmaMmaSchedule<128, 128, 4, 4, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv>(x, weight, out, stream);
    if (tokens <= 8) return fp8_linear_a16_simt<Geometry, 8, Simt>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<16, 4, 1>>(x, weight, out, stream);
    if (tokens <= 32) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 64) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    if (tokens <= 256) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT32R64S3>(x, weight, out, stream);
    if (tokens <= 4096) return fp8_linear_a16_tma<Geometry, Fp8A16TmaT64R128>(x, weight, out, stream);
    fp8_linear_a16_tma<Geometry, TmaT128R128>(x, weight, out, stream);
}

// A16 only: the checkpoint is weight-only, served with BF16 activations.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N16384K2560{16384, 2560, launch_a16, nullptr, uses_a8, nullptr};
} // namespace infernix::ops::detail
