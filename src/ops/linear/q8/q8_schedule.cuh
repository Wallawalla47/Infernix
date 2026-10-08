#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_rowsplit_storage.cuh"

namespace infernix::ops::detail {

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage, int Stages,
          Cache CodeCache, int MinBlocksPerSm, bool Predicated = false>
struct Q8A16SimtSchedule {
    static_assert(BlockRows > 0 && BlockRows <= 32);
    static_assert(BlockTokens > 0 && BlockTokens <= 8);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(GroupsPerWarpStage > 0 && GroupsPerWarpStage % 2 == 0);
    static_assert(Stages >= 2 && Stages <= 8);
    static_assert(MinBlocksPerSm > 0);
    static constexpr bool kPredicated        = Predicated;
    static constexpr int kBlockRows          = BlockRows;
    static constexpr int kBlockTokens        = BlockTokens;
    static constexpr int kWarpsPerRow        = WarpsPerRow;
    static constexpr int kGroupsPerWarpStage = GroupsPerWarpStage;
    static constexpr int kStages             = Stages;
    static constexpr Cache kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm     = MinBlocksPerSm;
    static constexpr int kWarps              = BlockRows * WarpsPerRow;
    static constexpr int kThreads            = kWarps * 32;
    static constexpr int kStageK             = GroupsPerWarpStage * Q8RowSplitStorage::kGroupK;
    static constexpr int kCodeVecsPerStage   = GroupsPerWarpStage * 2;
    static constexpr int kScalePairsPerStage = GroupsPerWarpStage / 2;
    static constexpr int kCodePhases         = (GroupsPerWarpStage + 7) / 8;
    static constexpr int kStagingBytes =
        kWarps * Stages * GroupsPerWarpStage *
        (Q8RowSplitStorage::kCodeBytesPerGroup + Q8RowSplitStorage::kScaleBytesPerGroup);
    static constexpr int kPartialBytes = WarpsPerRow > 1 ? kWarps* BlockTokens * sizeof(float) : 0;
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024, "Q8 SIMT shared memory exceeds 48 KiB");
};

template <int BlockRows, int WarpsPerRow, int MinBlocksPerSm, int StaticK = 0, int Unroll = 8>
struct Q8A16GemvSchedule {
    static_assert(BlockRows > 0 && WarpsPerRow > 0);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(MinBlocksPerSm > 0 && Unroll > 0);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % 256 == 0));
    static constexpr int kBlockRows      = BlockRows;
    static constexpr int kBlockTokens    = 1;
    static constexpr int kWarpsPerRow    = WarpsPerRow;
    static constexpr int kThreads        = BlockRows * WarpsPerRow * 32;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kStaticK        = StaticK;
    static constexpr int kUnroll         = Unroll;
    static_assert(kThreads <= 1024);
};

// x of the register-streamed route: staged once per CTA and pass into shared memory, or read by
// each warp through L1 (long K split over a CTA's warps, where a whole-K tile would not fit).
enum class Q8StreamX : std::uint8_t { Shared, Global };

// Register-streamed decode schedule (q8_a16_stream.cuh). A warp holds RowsPerWarp rows and
// KSplit warps share each row group, splitting K into contiguous scale-pair ranges; a CTA covers up
// to MaxCols columns in passes of ColsPerPass over the register-resident weights.
template <int K, int RowsPerWarp, int KSplit, int Warps, int MaxCols, int ColsPerPass, Q8StreamX XStage,
          int MinBlocksPerSm>
struct Q8A16StreamSchedule {
    static_assert(K > 0 && K % 64 == 0);
    static_assert(KSplit == 1 || KSplit == 2 || KSplit == 4 || KSplit == 8);
    static_assert(K % (64 * KSplit) == 0, "every K warp owns the same whole scale pairs");
    static_assert(RowsPerWarp > 0 && Warps > 0 && Warps % KSplit == 0);
    static_assert(ColsPerPass > 0 && ColsPerPass <= 8 && MaxCols % ColsPerPass == 0 && MaxCols <= 16);
    static_assert(MinBlocksPerSm > 0);
    static constexpr int kK              = K;
    static constexpr int kRowsPerWarp    = RowsPerWarp;
    static constexpr int kKSplit         = KSplit;
    static constexpr int kWarps          = Warps;
    static constexpr int kThreads        = Warps * 32;
    static constexpr int kRowGroups      = Warps / KSplit;
    static constexpr int kBlockRows      = kRowGroups * RowsPerWarp;
    static constexpr int kMaxCols        = MaxCols;
    static constexpr int kColsPerPass    = ColsPerPass;
    static constexpr int kPasses         = MaxCols / ColsPerPass;
    static constexpr Q8StreamX kXStage   = XStage;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kWarpK          = K / KSplit;
    static constexpr int kChunks         = (kWarpK + 255) / 256;
    static constexpr int kXTileBytes     = XStage == Q8StreamX::Shared ? ColsPerPass * K * 2 : 0;
    static constexpr int kPartialBytes   = KSplit > 1 ? kRowGroups * KSplit * RowsPerWarp * ColsPerPass * 4 : 0;
    static constexpr int kSharedBytes    = (kXTileBytes + kPartialBytes + 15) / 16 * 16 > 0
                                               ? (kXTileBytes + kPartialBytes + 15) / 16 * 16
                                               : 16;
    static_assert(kThreads <= 1024 && kSharedBytes <= 99 * 1024);
};

enum class Q8MmaFragmentPipeline { Serial, PingPong };

// Quant codes are prefetched while MMA consumes the decoded BF16 weight tile.
// Eight G32 scales are cached per row. Activations may use one or two buffers.
// Padding-free problems can fix K and specialize the outer-loop unroll factor.
template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int ActivationStages, int MinBlocksPerSm,
          Q8MmaFragmentPipeline FragmentPipeline = Q8MmaFragmentPipeline::PingPong,
          Cache WeightCache = Cache::cg, Cache ActivationCache = Cache::cg,
          Cache PredicatedCache = Cache::ca, bool Predicated = false, int StaticK = 0,
          int KLoopUnroll = 4>
struct Q8A16MmaSchedule {
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kBlockK            = BlockK;
    static constexpr int kWarpRows          = WarpRows;
    static constexpr int kWarpTokens        = WarpTokens;
    static constexpr int kActivationStages  = ActivationStages;
    static constexpr int kStages            = 2;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kFragmentPipeline = FragmentPipeline;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kPredicatedCache  = PredicatedCache;
    static constexpr bool kPredicated       = Predicated;
    static constexpr int kStaticK           = StaticK;
    static constexpr int kKLoopUnroll       = KLoopUnroll;
    static constexpr int kWarpGridRows      = BlockRows / WarpRows;
    static constexpr int kWarpGridTokens    = BlockTokens / WarpTokens;
    static constexpr int kWarps             = kWarpGridRows * kWarpGridTokens;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kMmaRows           = WarpRows / 16;
    static constexpr int kMmaTokens         = WarpTokens / 8;
    static constexpr int kMmaKSteps         = BlockK / 16;
    static constexpr int kScaleCacheBytes   = 16;
    static constexpr int kSharedBytes       = BlockRows * BlockK * 3 +
                                        ActivationStages * BlockTokens * BlockK * 2 +
                                        BlockRows * kScaleCacheBytes;
    static_assert(BlockRows > 0 && BlockTokens > 0 && WarpRows > 0 && WarpTokens > 0);
    static_assert(BlockRows % WarpRows == 0 && BlockTokens % WarpTokens == 0);
    static_assert(WarpRows % 16 == 0 && WarpTokens % 8 == 0);
    static_assert(BlockK == 64 || BlockK == 128);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % BlockK == 0));
    static_assert(KLoopUnroll > 0);
    static_assert(ActivationStages == 1 || ActivationStages == 2);
    static_assert(kThreads <= 1024 && MinBlocksPerSm > 0);
    static_assert(kSharedBytes <= 99 * 1024);
};

enum class Q8ScaleAccess : std::uint8_t { Direct, Shared };
enum class Q8ActivationStage : std::uint8_t {
    ActiveOnly,    // Stage the compile-time capacity, zero filling inactive tokens.
    PaddedZero,    // Also initialize MMA columns beyond the capacity.
    RuntimeActive, // Stage only live columns; the epilogue discards inactive columns.
};

// RowTiles 16-row MMA tiles share one CTA's staged activation and B fragments, halving the
// activation's L2 re-reads per weight row where they outweigh the weight stream.
template <int BlockTokens, int KWarps, int Stages, int MinBlocksPerSm, Q8ScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          Q8ActivationStage ActivationStage = Q8ActivationStage::ActiveOnly, int StaticK = 0,
          int TokenCapacity = BlockTokens, bool ExactTokens = false, int RowTiles = 1>
struct Q8A16SlicedKMmaSchedule {
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(BlockTokens >= 8 && BlockTokens % 8 == 0);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
    static_assert(TokenCapacity > 0 && TokenCapacity <= BlockTokens);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % (KWarps * 64) == 0));
    static_assert(RowTiles == 1 || RowTiles == 2);
    template <int K, int Capacity, bool Exact = false>
    using with_problem =
        Q8A16SlicedKMmaSchedule<BlockTokens, KWarps, Stages, MinBlocksPerSm, ScaleAccess,
                                ActivationCache, WeightCache, ActivationStage, K, Capacity, Exact,
                                RowTiles>;
    static constexpr int kRowTiles          = RowTiles;
    static constexpr int kBlockRows         = 16 * RowTiles;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kTokenCapacity     = TokenCapacity;
    static constexpr int kStaticK           = StaticK;
    static constexpr bool kExactTokens      = ExactTokens;
    static constexpr int kKWarps            = KWarps;
    static constexpr int kStages            = Stages;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kWarpK             = 64;
    static constexpr int kBlockK            = KWarps * kWarpK;
    static constexpr int kRowsPerLoaderWarp = kBlockRows / KWarps;
    static constexpr int kScaleBytesPerRow  = kBlockK / 16;
    static constexpr int kStagingBytes =
        Stages * (kBlockRows * kBlockK + KWarps * BlockTokens * kWarpK * 2 +
                  kBlockRows * (ScaleAccess == Q8ScaleAccess::Shared ? kScaleBytesPerRow : 1));
    static constexpr int kPartialBytes = KWarps * RowTiles * (BlockTokens / 8) * 32 * 16;
    static constexpr int kSharedBytes =
        kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

template <int BlockTokens, int KWarps, int TokenGroups, int Stages, int MinBlocksPerSm,
          int StaticK = 0, Cache WeightCache = Cache::cg, Cache ActivationCache = Cache::cg,
          bool TiledTokens = false, bool ReadOnlyScales = true>
struct Q8A16GroupedSlicedKMmaSchedule {
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8);
    static_assert(TokenGroups > 0 && BlockTokens > 0 && BlockTokens % TokenGroups == 0);
    static_assert((BlockTokens / TokenGroups) % 8 == 0);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % (KWarps * 64) == 0));
    static constexpr int kBlockRows        = 16;
    static constexpr int kBlockTokens      = BlockTokens;
    static constexpr int kKWarps           = KWarps;
    static constexpr int kTokenGroups      = TokenGroups;
    static constexpr int kWarpTokens       = BlockTokens / TokenGroups;
    static constexpr int kWarps            = KWarps * TokenGroups;
    static constexpr int kThreads          = kWarps * 32;
    static constexpr int kBlockK           = KWarps * 64;
    static constexpr int kStages           = Stages;
    static constexpr int kMinBlocksPerSm   = MinBlocksPerSm;
    static constexpr int kStaticK          = StaticK;
    static constexpr auto kWeightCache     = WeightCache;
    static constexpr auto kActivationCache = ActivationCache;
    static constexpr bool kTiledTokens     = TiledTokens;
    static constexpr bool kReadOnlyScales  = ReadOnlyScales;
    static constexpr int kSharedBytes = Stages * (16 * kBlockK + kWarps * kWarpTokens * 64 * 2);
    static_assert(kThreads <= 1024 && kSharedBytes <= 99 * 1024);
};

} // namespace infernix::ops::detail
