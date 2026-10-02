#pragma once
#include "ops/softmax_attention/common/causal_geometry.h"

namespace ninfer::ops::detail {
template <int Tokens, int Warps, int Keys, int MinBlocks, bool Compact, bool Dynamic = true>
struct Nvfp4KvGroupedMmaSchedule {
    static_assert(Tokens > 0 && Warps > 0 && Warps <= 16 && MinBlocks > 0);
    static_assert(Keys == 32 || Keys == 64);
    static constexpr int kTokenTile = Tokens, kWarps = Warps, kThreads = Warps * 32;
    static constexpr int kKeyRows = Keys, kMinBlocks = MinBlocks;
    static constexpr bool kCompactKVStage = Compact, kDynamicArena = Dynamic;
    static constexpr int kArenaBytes = (Compact ? 3 : 5) * Keys * 256;
};

// Four consumer warps each own 16 query rows; producer warps decode K/V concurrently.
// Register redistribution requires this kernel's non-RDC translation unit.
template <int Keys = 64, int ProducerWarps = 12, int ProducerRegisters = 40,
          int ConsumerRegisters = 232>
struct Nvfp4KvTiledMmaSchedule {
    static_assert(Keys == 32 || Keys == 64);
    static_assert(ProducerWarps >= 4 && ProducerWarps <= 12 && ProducerWarps % 4 == 0);
    static_assert(ProducerRegisters >= 24 && ProducerRegisters % 8 == 0);
    static_assert(ConsumerRegisters >= 128 && ConsumerRegisters <= 256 &&
                  ConsumerRegisters % 8 == 0);
    static_assert(ProducerWarps * 32 * ProducerRegisters + 128 * ConsumerRegisters <= 65536);
    static constexpr int kQueryRows = 64, kKeyRows = Keys;
    static constexpr int kProducerWarps = ProducerWarps, kConsumerWarps = 4;
    static constexpr int kProducerThreads = ProducerWarps * 32, kConsumerThreads = 128;
    static constexpr int kThreads           = kProducerThreads + kConsumerThreads;
    static constexpr int kProducerRegisters = ProducerRegisters,
                         kConsumerRegisters = ConsumerRegisters;
    static constexpr int kSharedBytes       = (kQueryRows + 2 * Keys) * 256 * 2 + 32;
};

// The MXFP8 tiled kernel over NVFP4 KV (Nvfp4TiledKeys): eight warps of 16 query rows, one
// 64-key page per tile, V decoded once per tile into an FP16 arena.
struct Nvfp4KvArenaTiledSchedule {
    static constexpr int kQueryRows    = 128;
    static constexpr int kKeyRows      = 64;
    static constexpr int kWarps        = kQueryRows / 16;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kMaxRegisters = 255;
    static constexpr int kQBytes       = kQueryRows * 256; // two terms, 128 B each
    static constexpr int kQScaleBytes  = 2 * kQueryRows * 16 + kQueryRows * 4;
    static constexpr int kKBytes       = kKeyRows * 128;
    static constexpr int kVBytes       = kKeyRows * 128;
    static constexpr int kVStageBytes  = kKeyRows * 256 * 2;
    static constexpr int kScaleBytes   = kKeyRows * (16 + 16);
    static constexpr int kSharedBytes =
        kQBytes + kQScaleBytes + kKBytes + kVBytes + kVStageBytes + kScaleBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

struct Nvfp4KvMergeSchedule {
    static constexpr int kDChunk = 256, kThreads = 256;
};
} // namespace ninfer::ops::detail
