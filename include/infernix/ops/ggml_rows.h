#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace infernix::ops {

/**
 * Token embedding from a GGUF block table (docs/maintainer/qwen3_8-flash-next-design.md §16.2.2):
 * out[:, t] = bf16(row ids[t] of `table`), each element's exact value (a product of at most 24
 * significant bits, exact in binary32) rounded once to BF16. `table` is a ggml_rows_v1 Weight
 * [V, H] whose rows the device reads in place (pinned host memory: about 1.4 KB per token for
 * IQ4_XS). ids: I32 [T]; out: BF16 [H, T]. An id outside [0, V) traps.
 */
void ggml_embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);

/**
 * The PLE n-gram rows of an IQ4_NL table (design §12.5): rows is U8
 * [row_bytes, heads, T] (row_bytes = W / 32 * 18) and out BF16 [heads * W, T] with
 * out[h * W + j, t] = bf16(fp16(d) * kvalues_iq4nl[nibble]), the exact product rounded once.
 */
void ple_embed_iq4_nl(const Tensor& rows, Tensor& out, cudaStream_t stream);

} // namespace infernix::ops
