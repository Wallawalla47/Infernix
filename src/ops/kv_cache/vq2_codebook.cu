// infernix::ops - the device copies of the VQ2 KV-cache codebook.
#include "ops/kv_cache/append/vq2_kernel.cuh"
#include "ops/kv_cache/vq2_codec.cuh"

namespace infernix::ops {

// The encoder and the decoders stage this table into shared memory.
__device__ std::int8_t g_kv_cache_vq2_codebook[kKVCacheVq2Patterns * 8] = {
#include "ops/kv_cache/vq2_codebook.inc"
};

} // namespace infernix::ops
