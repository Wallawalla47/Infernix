#include "ops/lossless/linear_impl.cuh"

// The --lossless route's GGUF projections: IQ2_S, IQ3_XXS, IQ3_S, Q2_0 (ops/lossless/linear_impl.cuh).

namespace infernix::ops::lossless_linear_impl {

bool launch_ggml_i(QType q, const Call& c, std::int32_t first, std::int32_t count, std::uint8_t* base, const Layout& l,
            cudaStream_t stream, const ChunkStage* stage) {
    switch (q) {
    case QType::GGML_IQ2_S: launch_chunk_stage<GgmlFormat<QType::GGML_IQ2_S>>(c, first, count, base, l, stream, stage); break;
    case QType::GGML_IQ3_XXS: launch_chunk_stage<GgmlFormat<QType::GGML_IQ3_XXS>>(c, first, count, base, l, stream, stage); break;
    case QType::GGML_IQ3_S: launch_chunk_stage<GgmlFormat<QType::GGML_IQ3_S>>(c, first, count, base, l, stream, stage); break;
    case QType::GGML_Q2_0: launch_chunk_stage<GgmlFormat<QType::GGML_Q2_0>>(c, first, count, base, l, stream, stage); break;
    default: return false;
    }
    return true;
}

} // namespace infernix::ops::lossless_linear_impl
