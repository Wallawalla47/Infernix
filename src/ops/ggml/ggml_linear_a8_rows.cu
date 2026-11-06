// The 8-bit route's mma kernels over ggml_rows_v1 matrices (GgmlLayout::Rows; ops/ggml/ggml_linear_a8.cuh):
// every format at 3-31 columns, one format a launch (FP32 or BF16 outputs) or two of mixed_format's in one
// (BF16 outputs). A unit of 256 elements is each row's bytes of it, copied from the row's 16-byte aligned
// start (the exact route's staged mma rows, ggml_mma_rows.cuh); a matrix whose K is not whole units (formats
// of 32- and 64-element blocks) has a short last unit.

#include "ops/ggml/ggml_linear_a8.cuh"

namespace infernix::ops::a8_linear {
namespace {

// At two CTAs of 256 threads per SM (128 registers): a lane's state is 4 NT FP32 accumulators and per slice
// one int32 dot per output, where the exact wide tiles carry int128 outputs, three int32 limb sums and the
// K-quants' int64 min sums (and spilled at two CTAs for Q4_K, Q5_K and Q6_K: kWideMinBlocks). Unmeasured.
template <QType F, bool OutF32, int NT>
__global__ void __launch_bounds__(kMmaMaxThreads, 2)
    rows_mma_kernel(A8Group grp, int slices, const std::int32_t* __restrict__ encoded, int stride, int columns, int split,
                    int stages, int slot_bytes) {
    extern __shared__ __align__(16) unsigned char a8_rows_shared[];
    mma_body<F, OutF32, NT, false>(slot_at(grp, static_cast<int>(blockIdx.x)), slices, encoded, stride, columns, split,
                                   stages, slot_bytes, a8_rows_shared);
}

// Two formats in one launch, each CTA running its segment's (A8Group::format 0: F0, 1: F1).
template <QType F0, QType F1, int NT>
__global__ void __launch_bounds__(kMmaMaxThreads, 2)
    rows_mma_mixed_kernel(A8Group grp, int slices, const std::int32_t* __restrict__ encoded, int stride, int columns,
                          int split, int stages, int slot_bytes) {
    extern __shared__ __align__(16) unsigned char a8_rows_mixed_shared[];
    const Slot seg = slot_at(grp, static_cast<int>(blockIdx.x));
    if (seg.format == 0) {
        mma_body<F0, false, NT, false>(seg, slices, encoded, stride, columns, split, stages, slot_bytes, a8_rows_mixed_shared);
    } else {
        mma_body<F1, false, NT, false>(seg, slices, encoded, stride, columns, split, stages, slot_bytes, a8_rows_mixed_shared);
    }
}

template <QType F, bool OutF32>
void launch_format(const A8Group& g, const Columns& in, cudaStream_t stream) {
    constexpr int kSlot = kSlotBytes<F, false>;
    switch (column_tiles_for(in.columns)) {
    case 1: launch_mma(rows_mma_kernel<F, OutF32, 1>, g, in, kSlot, 1, stream); break;
    case 2: launch_mma(rows_mma_kernel<F, OutF32, 2>, g, in, kSlot, 2, stream); break;
    default: launch_mma(rows_mma_kernel<F, OutF32, 4>, g, in, kSlot, 4, stream); break;
    }
}

template <QType F0, QType F1>
void launch_pair(const A8Group& g, const Columns& in, cudaStream_t stream) {
    constexpr int kSlot = std::max(kSlotBytes<F0, false>, kSlotBytes<F1, false>);
    switch (column_tiles_for(in.columns)) {
    case 1: launch_mma(rows_mma_mixed_kernel<F0, F1, 1>, g, in, kSlot, 1, stream); break;
    case 2: launch_mma(rows_mma_mixed_kernel<F0, F1, 2>, g, in, kSlot, 2, stream); break;
    default: launch_mma(rows_mma_mixed_kernel<F0, F1, 4>, g, in, kSlot, 4, stream); break;
    }
}

} // namespace

void launch_mma_rows(QType format, const A8Group& g, const Columns& in, bool out_f32, cudaStream_t stream) {
#define INFERNIX_GGML_A8_ROWS(Q)                                                                                       \
    case Q:                                                                                                            \
        if (out_f32) {                                                                                                 \
            launch_format<Q, true>(g, in, stream);                                                                     \
        } else {                                                                                                       \
            launch_format<Q, false>(g, in, stream);                                                                    \
        }                                                                                                              \
        break;
    switch (format) {
        INFERNIX_GGML_A8_FORMATS(INFERNIX_GGML_A8_ROWS)
    default:
        throw std::invalid_argument("ggml_linear (a8): segment format is not a ggml block format");
    }
#undef INFERNIX_GGML_A8_ROWS
}

bool launch_mma_rows_mixed(QType f0, QType f1, const A8Group& g, const Columns& in, cudaStream_t stream) {
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
