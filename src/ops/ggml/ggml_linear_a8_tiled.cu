// The 8-bit route's mma kernels over ggml_tiles16_v1 matrices (GgmlLayout::Tiles16; ops/ggml/ggml_linear_a8.cuh):
// Q4_K, Q5_K, Q6_K and IQ4_XS at 1-31 columns (one-column Q6_K and IQ4_XS calls of one format take the rows
// kernel instead, ggml_linear_a8.cu), one format a launch (FP32 or BF16 outputs) or two in one (BF16 outputs,
// every narrow width, as the exact route's tiled mixed launches). A unit is one block of the tile's 16 rows:
// one contiguous, 16-byte aligned run (the exact tiled_mma_body's ring).

#include "ops/ggml/ggml_linear_a8.cuh"

namespace infernix::ops::a8_linear {
namespace {

template <QType F>
inline constexpr bool kTiledFormat = ggml_tiles16_supported(F);

// Two CTAs of 256 threads per SM, as rows_mma_kernel (ggml_linear_a8_rows.cu). Unmeasured.
template <QType F, bool OutF32, int NT>
__global__ void __launch_bounds__(kMmaMaxThreads, 2)
    tiled_mma_kernel(A8Group grp, int slices, const std::int32_t* __restrict__ encoded, int stride, int columns, int split,
                     int stages, int slot_bytes) {
    extern __shared__ __align__(16) unsigned char a8_tiled_shared[];
    mma_body<F, OutF32, NT, true>(slot_at(grp, static_cast<int>(blockIdx.x)), slices, encoded, stride, columns, split, stages,
                                  slot_bytes, a8_tiled_shared);
}

// Two tiled formats in one launch, each CTA running its segment's (A8Group::format 0: F0, 1: F1).
template <QType F0, QType F1, int NT>
__global__ void __launch_bounds__(kMmaMaxThreads, 2)
    tiled_mma_mixed_kernel(A8Group grp, int slices, const std::int32_t* __restrict__ encoded, int stride, int columns,
                           int split, int stages, int slot_bytes) {
    extern __shared__ __align__(16) unsigned char a8_tiled_mixed_shared[];
    const Slot seg = slot_at(grp, static_cast<int>(blockIdx.x));
    if (seg.format == 0) {
        mma_body<F0, false, NT, true>(seg, slices, encoded, stride, columns, split, stages, slot_bytes, a8_tiled_mixed_shared);
    } else {
        mma_body<F1, false, NT, true>(seg, slices, encoded, stride, columns, split, stages, slot_bytes, a8_tiled_mixed_shared);
    }
}

template <QType F, bool OutF32>
void launch_format(const A8Group& g, const Columns& in, cudaStream_t stream) {
    if constexpr (!kTiledFormat<F>) {
        throw std::logic_error("ggml_linear (a8): a tiled segment's format has no tiled kernels");
    } else {
        constexpr int kSlot = kSlotBytes<F, true>;
        switch (column_tiles_for(in.columns)) {
        case 1: launch_mma(tiled_mma_kernel<F, OutF32, 1>, g, in, kSlot, 1, stream); break;
        case 2: launch_mma(tiled_mma_kernel<F, OutF32, 2>, g, in, kSlot, 2, stream); break;
        default: launch_mma(tiled_mma_kernel<F, OutF32, 4>, g, in, kSlot, 4, stream); break;
        }
    }
}

template <QType F0, QType F1>
void launch_pair(const A8Group& g, const Columns& in, cudaStream_t stream) {
    constexpr int kSlot = std::max(kSlotBytes<F0, true>, kSlotBytes<F1, true>);
    switch (column_tiles_for(in.columns)) {
    case 1: launch_mma(tiled_mma_mixed_kernel<F0, F1, 1>, g, in, kSlot, 1, stream); break;
    case 2: launch_mma(tiled_mma_mixed_kernel<F0, F1, 2>, g, in, kSlot, 2, stream); break;
    default: launch_mma(tiled_mma_mixed_kernel<F0, F1, 4>, g, in, kSlot, 4, stream); break;
    }
}

} // namespace

void launch_mma_tiled(QType format, const A8Group& g, const Columns& in, bool out_f32, cudaStream_t stream) {
#define INFERNIX_GGML_A8_TILED(Q)                                                                                      \
    case Q:                                                                                                            \
        if (out_f32) {                                                                                                 \
            launch_format<Q, true>(g, in, stream);                                                                     \
        } else {                                                                                                       \
            launch_format<Q, false>(g, in, stream);                                                                    \
        }                                                                                                              \
        break;
    switch (format) {
        INFERNIX_GGML_A8_FORMATS(INFERNIX_GGML_A8_TILED)
    default:
        throw std::invalid_argument("ggml_linear (a8): segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_A8_TILED
}

bool launch_mma_tiled_mixed(QType f0, QType f1, const A8Group& g, const Columns& in, cudaStream_t stream) {
#define INFERNIX_GGML_A8_PAIR(A, B)                                                                                    \
    if (f0 == QType::A && f1 == QType::B) {                                                                            \
        launch_pair<QType::A, QType::B>(g, in, stream);                                                                \
        return true;                                                                                                   \
    }
    INFERNIX_GGML_A8_PAIR(GGML_Q4_K, GGML_Q5_K)
    INFERNIX_GGML_A8_PAIR(GGML_Q4_K, GGML_Q6_K)
    INFERNIX_GGML_A8_PAIR(GGML_Q4_K, GGML_IQ4_XS)
    INFERNIX_GGML_A8_PAIR(GGML_Q5_K, GGML_Q6_K)
    INFERNIX_GGML_A8_PAIR(GGML_Q5_K, GGML_IQ4_XS)
    INFERNIX_GGML_A8_PAIR(GGML_Q6_K, GGML_IQ4_XS)
#undef INFERNIX_GGML_A8_PAIR
    return false;
}

} // namespace infernix::ops::a8_linear
