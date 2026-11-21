// hyper_connection_repack_q8 (include/infernix/ops/hyper_connection.h): a BF16 Contiguous hyper-connection
// matrix rewritten in its own bytes as Q8_G32_FP16 RowSplit for `--a8-activations` (LOSSY, opt-in), the
// layout hyper_connection_mix's Q8 routes read (hyper_connection_mix_fused.cu's Q8Codec: codes at
// row * padded_k + k, the FP16 scale of group g at row * (padded_k / 32) * 2 + 2 g past the codes). The
// quantization is the shared Op's (include/infernix/ops/q8_g32.h), run from a scratch copy into the weight's bytes.

#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/q8_g32.h"

#include <stdexcept>

namespace infernix::ops {

std::size_t hyper_connection_repack_q8_scratch_bytes(const Weight& w) {
    if (!q8_g32_quantizable(w)) { return 0; }
    const std::size_t bf16 = static_cast<std::size_t>(w.n) * static_cast<std::size_t>(w.k) * 2;
    return q8_g32_bytes(w.n, w.k) <= bf16 ? bf16 : 0; // the Q8 matrix must fit in the BF16 matrix's bytes
}

void hyper_connection_repack_q8(Weight& w, void* scratch, cudaStream_t stream) {
    if (hyper_connection_repack_q8_scratch_bytes(w) == 0 || scratch == nullptr) {
        throw std::invalid_argument("hyper_connection_repack_q8: needs a 16-byte aligned BF16 Contiguous [n, k] weight "
                                    "(k a multiple of 32) whose Q8 form fits in its bytes, and scratch");
    }
    void* base = const_cast<void*>(w.qdata);
    const std::size_t bf16_bytes = static_cast<std::size_t>(w.n) * w.k * 2;
    if (cudaMemcpyAsync(scratch, base, bf16_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        throw std::runtime_error("hyper_connection_repack_q8: copy to scratch failed");
    }
    Weight copy  = w;
    copy.qdata   = scratch;
    auto* scales = static_cast<std::uint8_t*>(base) + static_cast<std::size_t>(w.n) * q8_g32_padded_columns(w.k);
    w            = quantize_bf16_q8_g32(copy, base, scales, stream);
}

} // namespace infernix::ops
