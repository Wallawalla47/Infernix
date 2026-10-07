#pragma once

// The fast NVFP4 prompt kernel: the MXFP8 tiled kernel (mxfp8_tiled_mma.cuh) with NVFP4 keys. QK
// runs on block-scaled FP4 Tensor Cores straight from the stored K codes with a two-term NVFP4 Q,
// each 64-key V tile is decoded once per CTA into an FP16 arena, PV accumulates per tile on FP16
// Tensor Cores (or, with pv8, block-scaled E4M3), and the split partials merge with the inverse
// rotation. envelope.fast_prompt_pv8 selects the E4M3 PV form, and the key splits keep their FP32
// partials within envelope.prompt_split_workspace_bytes.

#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/softmax_attention.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

#include <cstdint>

namespace infernix::ops::detail {

// Whether an NVFP4 prompt-route launch runs the fast kernel: over 768 visible keys. Forced at
// every visible count (3 passes, H24 and H16 geometries, 256-3584 columns over 0-8192 keys) the
// fast kernel is equal to the tiled kernel at 512-768 visible keys, up to 1.6x faster from 1024
// on and up to 3.4x at 8192, and up to 1.6x slower for 256-512-column launches over an empty
// context (the tiled kernel's launch is cheaper there). Picking the fast kernel above 768 keys is
// within 0.03 % of the per-shape best in both geometries; the 2048 threshold of 2026-09, from a
// kernel without the one-decode V arena, costs 2.3-2.7 % overall and 56-64 % in its worst cell.
inline bool nvfp4_fast_prompt_applies(std::uint32_t max_visible_keys) {
    return max_visible_keys > 768;
}

void nvfp4_kv_fast_prompt_attention(const CausalAttentionOperands&, Nvfp4KvReadView,
                                    CausalAttentionExecutionEnvelope, WorkspaceArena&,
                                    DeviceExecutionView);

} // namespace infernix::ops::detail
