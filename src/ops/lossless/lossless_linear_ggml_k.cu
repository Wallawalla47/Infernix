#include "ops/lossless/linear_impl.cuh"

// The --lossless route's GGUF projections: Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL (ops/lossless/linear_impl.cuh).

namespace infernix::ops::lossless_linear_impl {

bool launch_ggml_k(QType q, const Call& c, std::int32_t first, std::int32_t count, std::uint8_t* base, const Layout& l,
            cudaStream_t stream, const ChunkStage* stage) {
    switch (q) {
    case QType::GGML_Q4_K: launch_chunk_stage<GgmlFormat<QType::GGML_Q4_K>>(c, first, count, base, l, stream, stage); break;
    case QType::GGML_Q5_K: launch_chunk_stage<GgmlFormat<QType::GGML_Q5_K>>(c, first, count, base, l, stream, stage); break;
    case QType::GGML_Q6_K: launch_chunk_stage<GgmlFormat<QType::GGML_Q6_K>>(c, first, count, base, l, stream, stage); break;
    case QType::GGML_IQ4_XS: launch_chunk_stage<GgmlFormat<QType::GGML_IQ4_XS>>(c, first, count, base, l, stream, stage); break;
    case QType::GGML_IQ4_NL: launch_chunk_stage<GgmlFormat<QType::GGML_IQ4_NL>>(c, first, count, base, l, stream, stage); break;
    default: return false;
    }
    return true;
}

} // namespace infernix::ops::lossless_linear_impl

// This unit's copy of the noise twin's seed (ops/lossless/noise.cuh).
INFERNIX_LOSSLESS_NOISE_UNIT(linear_ggml_k)
