#pragma once

// TEMPORARY (INFERNIX_TMP_KV_DUMP=<dir>, candidates §23.2 PB-01 step 1): writes a prompt's BF16 KV pages
// after its prefill, for tools/kv_lossless_count to count the blocks a 12-bit lossless page form would
// hold. Diagnostic only: removed with the switch once PB-01's go/no-go is decided.

#include "models/qwen3_5/config.h"
#include "models/qwen3_5/state/decoder_state.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace infernix::models::qwen3_5::detail {

// The directory INFERNIX_TMP_KV_DUMP names; empty when unset.
[[nodiscard]] const std::string& kv_dump_directory();

// Writes positions [0, tokens) of the sequence `view` maps (every attention layer of `cache`, K and V)
// to kv_dump_directory(): per layer one K and one V file holding the sequence's pages in logical
// order, each page as stored ([kv head][slot][dim], 16-bit little-endian values), and one JSON file
// with the layout, written last. The caller has synchronized the prefill's work. A cache that does
// not store BF16 K is reported once on stderr and not written.
void dump_prompt_kv(const PagedKVCache& cache, const PagedKVCacheView& view, std::uint32_t tokens,
                    const TextConfig& text, std::string_view model_name, cudaStream_t stream);

} // namespace infernix::models::qwen3_5::detail
