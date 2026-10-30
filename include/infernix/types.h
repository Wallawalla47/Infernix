#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes  = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes   = 2ULL << 30;
// Physical Host memory Infernix leaves free for the OS and other programs after its weights, experts,
// n-gram table and caches, when --ram-free-mib is not given (8 GiB since 2026-10-26: with 2 GiB a desktop paged
// once Qwen3.8-Flash-Next's experts and n-gram table took the rest; Basalt's RAM-low guard sits at ~8.3 GiB).
inline constexpr std::uint64_t kDefaultRamFreeBytes = 8ULL << 30;
// Prompt-attention split workspace when --prefill-split-workspace-mib is not given.
inline constexpr std::uint32_t kDefaultPrefillSplitWorkspaceMiB = 256;
inline constexpr std::uint32_t kMaximumPrefillSplitWorkspaceMiB = 16384;
// Pinned Host slab pool of the hybrid prefix cache when --host-context-mib is not given, for every
// architecture.
inline constexpr std::size_t kDefaultHybridHostCacheBytes = 8ULL << 30;
// How long the prefix cache holds a conversation's state after a reply ending in a tool call,
// when --cache-tool-hold-secs is not given (Qwen3.5): longer than most tools take to run.
inline constexpr std::chrono::milliseconds kDefaultToolCallHold{30'000};
// How long the Engine must have been idle before it prefills a finished turn's history for the
// conversation's next request, when --cache-next-turn-ms is not given (TensorFold's measured
// quiet period: a request sent the moment a reply ended was not delayed).
inline constexpr std::chrono::milliseconds kDefaultNextTurnPrefillQuiet{150};

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Fp8E4M3Row256,
    Nvfp4Group16,
    Fp8KeyNvfp4Value,
    // Rotated 2-bit eight-value vector codes for K and V plus an exact INT8 recent-key window.
    Vq2,
    // Rotated 4-bit Lloyd-Max K codes, 2-bit vector V codes and the same exact recent-key window.
    Q4KeyVq2Value,
};

// The vocabulary head's logits (EngineOptions::logits_dtype), which the sampler, greedy argmax, draft
// acceptance, constraints and scoring read.
enum class LogitsDtype : std::uint8_t {
    // The head's sums without a final rounding (the default).
    FP32,
    // Each logit rounded once to BF16, the dtype Hugging Face Transformers returns them in: near-equal
    // logits become exact ties, which selection breaks by token id. For comparisons.
    BF16,
};
[[nodiscard]] inline const char* kv_cache_cli_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    case KvCacheStorage::Vq2:
        return "vq2";
    case KvCacheStorage::Q4KeyVq2Value:
        return "k4v2";
    }
    return "unknown";
}

// A fast path: a route whose arithmetic differs from the default route's, run only under
// EngineOptions::fast_paths and only for the configurations it is qualified for
// (docs/maintainer/model-quality.md section 6.7 lists each with its evidence).
enum class FastPath : std::uint8_t {
    // Prompt attention runs P*V in 8 bits where the KV format's prompt kernel has that form: the
    // fast INT8 kernel rounds each probability times its key's V group scale to a u8 code per
    // 64-key tile; the fast NVFP4 kernel (over more than 768 visible keys) and K8V4's tiled kernel
    // round probabilities and decoded V to E4M3; the VQ2 and K4V2 prompt kernel uses its integer
    // form over the tile's INT8 V rows. The default route runs FP16 P*V.
    PromptPv8,
    // INT8 KV with 24 query heads: single-column decode rows split their keys from 64 rather than
    // 128 (a different softmax merge order).
    DecodeSplit64,
    // Qwen3.8-Flash-Next with Q8 dense weights: the GDN q/k/v/z [16384,2560] and the GDN and QSA
    // output [2560,6144] projections run NInfer's GEMV, sliced-K and grouped sliced-K routes
    // (ops::LinearRoutes::Q8SlicedK), which accumulate in a different order, so a column's bits depend on
    // how many columns share the call: it couples a request's decode to the other requests of its
    // round (LoadSummary::request_invariant is false with it).
    Q8SlicedK,
    // Qwen3.8-Flash-Next with Q8 dense weights: the GDN projection feeds its causal convolution the
    // current token's FP32 q/k/v instead of their BF16 rounding (one fused kernel; the convolution
    // history stays the BF16 state).
    GdnConvFused,
    // Qwen3.8-Flash-Next: one kernel per QSA layer normalizes and rotates Q, K and the index query,
    // appends K/V and the index keys, keeping FP32 between those steps instead of BF16.
    QsaPrepareFused,
    // FP8 weights whose permission is 16-bit activations: prefill-width projections carry the
    // activation as two E4M3 terms on the 8-bit Tensor Core path (ops::LinearRoutes::Fp8A8Pair).
    Fp8A8Pair,
    // The same projections and widths as fp8-a8-pair, with each term's power-of-two scale per 32
    // values of K instead of per column, applied by the block-scaled MMA
    // (ops::LinearRoutes::Fp8A8PairBlock32): every value within 2^14 of its block's maximum is carried
    // exactly. It takes precedence where both are selected. It also serves MXFP8 dense weights (NVIDIA
    // FP4-Local), whose group scales are the MMA's weight scales.
    Fp8A8PairBlock32,
    // Groupwise Q4 gate/up weights: prefill-width projections quantize activations to int8 per token
    // and fold each group's weight scale into its accumulator (ops::LinearRoutes::Q4W4A8).
    Q4W4A8,
    // Qwen3.8-Flash-Next MTP, and Qwen3.5 MTP with resident experts and no n-gram copies: a sampled
    // request's drafts are drawn from the draft head's four most probable tokens under the request's
    // sampler and verified by speculative rejection sampling against that proposal
    // (ops::speculative_sample_sparse_drafts), and its rounds verify every draft (no --spec-min-p
    // cut). The output distribution stays the sampler's; a seeded sampled request's text changes,
    // greedy requests are unchanged. Without it the drafts are the head's argmax (one-hot proposals).
    // Qualified on Flash-Next only; on Qwen3.5 it runs as a --fast-path candidate.
    SampledMtpDrafts,
    // Chunked Gated DeltaNet calls (prefill and other calls of 16 or more tokens) store the
    // normalized Q/K in FP16 and run the state products with FP16 operands, accumulating each pair
    // of k16 products in FP16 before the FP32 carry (ops::GdnChunkArithmetic::Fp16Accumulate).
    // Without it they keep BF16/TF32 operands with FP32 accumulators.
    GdnF16Accumulate,
    // Groupwise Q5 output and MLP down projections (linear_add [5120,6144] and [5120,17408]):
    // decode and verification calls run eight K warps with the pairwise reduction (NInfer's K-split
    // residual kernel, ops::LinearRoutes::Q5KSplit) instead of the exact route's four, a different
    // FP32 summation order. The column-invariant route stays column-invariant (one reduction at every
    // width). Without it they keep the exact route's tiles.
    Q5KSplitMma,
    // The paths below couple a request's arithmetic to the other requests of the Engine. Without
    // them a request's outputs depend only on the request, the artifact, the options and the build
    // (LoadSummary::request_invariant); they apply where the exact route is request-invariant.
    //
    // Decode and verify projections choose their route by the round's column count (single-column
    // GEMVs, small-T SIMT, MMA tiles), all at 16-bit activations, and the GDN input projection
    // rounds to BF16 at more than one row. Without it every width runs one column-invariant
    // 16-bit-activation route per projection.
    BatchedDecode,
    // With batched-decode, wide decode and verify rounds of FP8 projections take the artifact's
    // 8-bit activations (from 5 columns, or 17 in wide verification). Without it they stay at 16
    // bits.
    BatchedA8Decode,
    // With batched-decode, wide decode and verify rounds of NVFP4 projections take the artifact's
    // 4-bit activations (the MLP gate/up from 5 columns, down from 8). Without it they stay at 16 bits.
    BatchedA4Decode,
    // With batched-decode, decode and verify rounds of 9 or more columns take the A4 residual
    // profile in the NVFP4 MLP projections where the artifact permits 4-bit activations: each
    // column as a sum of NVFP4 terms, each encoding what the earlier ones left
    // (ops::LinearPolicy::AllowA4Residual). batched-a4-decode takes precedence. Without either they
    // stay at 16 bits.
    BatchedA4ResidualDecode,
    // With batched-decode, wide decode and verify rounds (17+ columns) of the FP8 attention and GDN
    // input projections and gate/up take the A8 residual profile where the artifact permits 8-bit
    // activations: each token as two E4M3 terms, the second encoding what the first left
    // (ops::LinearPolicy::AllowA8Residual); output and down projections stay at 16 bits.
    // batched-a8-decode takes precedence. Without either they stay at 16 bits.
    BatchedA8ResidualDecode,
    // With batched-decode, wide decode and verify rounds take the artifact's one-term 8-bit
    // activations in a subset of the FP8 projections only: rounds of 65+ columns, the attention and
    // GDN input projections and the dense MLP gate/up of every layer (qwen3_5 kA8SubsetMinColumns,
    // execution::precision_decode_text). The others keep their route
    // (the A8 residual profile where batched-a8-residual-decode is selected, else 16 bits).
    // batched-a8-decode takes precedence. The narrower rung of batched-a8-decode, which failed its
    // decode gate (docs/maintainer/model-quality.md, 6.7).
    BatchedA8SubsetDecode,
    // Without batched-decode: decode and verify calls of a single column (one request's plain
    // decode step alone in its round) run batched-decode's single-column routes, while every wider
    // round keeps the column-invariant route. A request's step then depends on whether another
    // request shares its round. The drafters keep their column-invariant copies. Where
    // batched-decode is selected it changes nothing.
    SingleColumnDecode,
    // W4A16 experts (weight-only exports: Qwen3.8-Flash-Next's, and Qwen3.5-122B-A10B's offloaded routed and
    // shared experts): calls of more than eight columns per
    // expert (prefill, wide verification) take each x and h row as the A4 residual profile of two
    // NVFP4 terms with an FP32 row scale, on the FP4 Tensor Cores, instead of the exact integer limb
    // route. A column's result then depends on its expert's column count in the call, which other
    // requests' columns share in batched rounds. Without it W4A16 experts are exact on every route.
    ExpertA4Residual,
    // The more precise form of expert-a4-residual: three NVFP4 terms per x and h row instead of two,
    // about a quarter of the two-term activation error for roughly a quarter of its prefill gain. It
    // takes precedence where both are selected.
    ExpertA4Residual3,
    // Qwen3.8-Flash-Next W4A16 experts: experts of at most eight columns in a call (decode and
    // verification, short prompts) take each x and h row as expert-a4-residual's two NVFP4 terms with
    // an FP32 row scale, their products summed exactly (int64) on the GPU narrow kernels and in the CPU
    // expert engine alike, instead of the exact integer limb route. A column's result then depends on
    // its expert's column count in the call (wider experts follow expert-a4-residual or stay exact).
    ExpertA4ResidualDecode,
    // Decode attention sizes each row's key partition from the round's batch. Without it a row's
    // partition depends only on its own keys.
    BatchSplitAttention,
    // Speculative rounds size their verify width and topology from all rows' proposals: DFlash2
    // tree tables and controller, the round's longest n-gram copy. Without it each round verifies
    // the configured chain at a fixed width.
    AdaptiveSpeculation,
    // With adaptive-speculation and --draft-tokens auto (draft_tokens_auto), Qwen3.8-27B / 35B MTP
    // rounds choose their draft length from the rows' learned acceptance and the round costs (one
    // common length for several rows). Without it they draft the configured length.
    AdaptiveMtp,
    // With adaptive-speculation and --draft-tokens auto, Qwen3.8-27B / 35B DFlash and DFlash2
    // all-neural rounds propose the whole block, then verify the width the drafts' calibrated
    // probabilities make fastest (one width for several rows, each row its own extent). Without it
    // they verify the configured length.
    AdaptiveDFlash,
    // Qwen3.8-Flash-Next with a DFlash2 drafter (--spec dflash2): rounds of two decoding rows
    // speculate. One proposal serves both rows and the round verifies one width for both (at most
    // two drafts per row), with --draft-tokens auto the width that maximizes both rows' expected
    // tokens from their drafts' calibrated probabilities per round time. The shared width couples
    // the rows (adaptive-speculation's mechanism for the MTP drafter). Without it two rows decode
    // plain, as three or more do.
    DFlash2Pairs,
    // The prefix cache resumes from any tap or endpoint and splits prefill at taps and sibling
    // boundaries. Without it prefill splits and the cache publishes and resumes only on the
    // prefill chunk grid, and decode-computed endpoints are not published.
    FreeCacheResume,
    // KV pages grow with each unit and a short pool pauses residents, which replay their generated
    // tokens as prefill. Without it a request holds its whole KV extent from admission.
    Preemption,
    // Prefill steps take a wider chunk while no request is waiting to decode (the chunk boundaries,
    // and with them the prefill arithmetic, depend on load).
    IdleWidePrefill,
    // Qwen3.8-Flash-Next GGUF imports: dense projections of prefill calls of 32 or more columns run on
    // BF16 tensor cores (each code exact in BF16, x as stored, FP32 sums of each sub-block scaled by
    // d * sub; ops::LinearRoutes::GgmlBf16Wide) instead of the exact A16 integer tiles, and the GGUF
    // expert banks' MoE calls of 4,096 or more columns the same arithmetic, each K stage of an expert's
    // codes decoded into shared memory (offloaded_moe::wide::kGgmlStageDecodeMinColumns).
    // A prompt's arithmetic then depends on where its calls start and how the layer walk groups them
    // (a narrower tail call stays exact), which prefix-cache resumes share between requests. Without
    // it every GGUF projection and expert is exact.
    GgmlBf16Prefill,
    // Qwen3.8-Flash-Next with row-FP8 (recipe C) or MXFP8 (NVIDIA FP4-Local) dense weights: dense
    // projections of prefill calls of 256 or more columns run on FP16 Tensor Cores at twice the
    // FP32-accumulation rate (ops::LinearRoutes::F16Accumulate): each E4M3 code exact in FP16, each
    // activation column exact in FP16 under a power-of-two scale (apart from values more than 2^17 below
    // its maximum), FP16 sums over 32-value K chunks carried in FP32. Like ggml-bf16-prefill, a prompt's
    // arithmetic then depends on where its calls start (a narrower tail call stays exact). Without it
    // every FP8 and MXFP8 projection is exact.
    F16AccumulatePrefill,
    // Qwen3.5 BF16 KV: prompt-route attention (calls of more than 256 columns) splits every query
    // tile's keys into the equal shares that fill whole waves of the SMs and merges their FP32
    // partials (a different softmax merge order). Without it each query tile runs all its keys in one
    // CTA.
    Bf16PromptSplit,
    // Qwen3.8-Flash-Next: the MoE router logits of prefill calls of 256 or more columns (the routed
    // experts' rows) on BF16 Tensor Cores with FP32 accumulation (ops::projection_fp32_router_mma; capy-flash
    // e0c0a5b5), exact products summed in the tile's order. Like ggml-bf16-prefill, a prompt's routing then
    // depends on where its calls start. Without it the router runs the FP32 SIMT dot products.
    RouterMma,
    // Qwen3.8-Flash-Next artifacts with a BF16 or 8-bit LM head converted with its 2-bit screen
    // (--screen-heads): a decode or verification round of at most 16 columns whose every row is
    // greedy, without penalties, a grammar or a token constraint, scores the whole vocabulary from the screen and
    // computes only the rows within 5 logits of each column's best screen score (at most 64) with the
    // exact head's arithmetic (ops::screened_head_logits). Those rows' logits carry the exact head's
    // bits and the others the screen's scores, so a column's token is the exact head's unless the
    // screen misses its row. Other rounds run the exact head, so a request's tokens depend on the
    // rows it shares its rounds with. Without it every round runs the exact head.
    ScreenedLmHead,
    // Every GDN model: the state pool stores each request's GDN recurrent state as FP16 instead of FP32,
    // halving the state's DRAM traffic in every decode, verification and prefill call (and its pool and
    // prefix-cache images). Each call widens the stored state exactly, runs its FP32 arithmetic and
    // rounds the state it publishes once to nearest FP16 (saturating at its largest finite value), so a
    // request's state carries one rounding per call. FP16 keeps 10 mantissa bits where BF16 keeps 7
    // (DAMP, arXiv 2608.27513: Qwen3.6-35B-A3B AIME 84.58 with an FP16 state, 79.71 with BF16, 85.46
    // FP32). Calls follow the request alone on the reproducible route, so its output stays
    // request-invariant. Without it the state is FP32.
    Fp16GdnState,
    // Speculative rounds whose drafts carry their proposal laws (DFlash2, sampled-mtp-drafts): a
    // sampled row's chain is accepted by block verification (Sun et al., arXiv 2403.10444;
    // ops::speculative_accept_sparse_drafts) instead of token by token. Its output distribution is the
    // target's either way and its expected accepted length is at least the token rule's, but a
    // seeded request draws other text. Greedy rows, tree rows and the reproducible route's Gumbel
    // rows are unchanged; a row's acceptance reads only that row.
    BlockVerification,
    // Tree rounds of a fixed --draft-tree-nodes table whose drafter builds its own trees (DFlash2,
    // DFlash): sampled rows' trees take each node's top ranks and at most one draw from the rest of
    // its law, and are verified by UniVer (Weng, Hu, Yairi, arXiv 2605.04543) instead of recursive
    // rejection without replacement. The output distribution is the target's either way; the drawn
    // text differs. Greedy rows, chain rows and automatic widths are unchanged.
    UniverTrees,
    // Tree rounds (any drafter, fixed or automatic widths): each GDN layer's verification record
    // computes every tree column in closed form from the row's initial state and its ancestors'
    // deltas, one solve level per depth (ops::GdnTreeRecordRoute::ClosedForm; TreeWY, arXiv
    // 2608.20961), instead of walking the tree column by column. FP32 in another order: outputs
    // within rounding of the walk, not bit-identical; records unchanged.
    GdnTreeClosedForm,
};

inline constexpr std::array<FastPath, 38> kFastPaths{
    FastPath::PromptPv8,           FastPath::DecodeSplit64,
    FastPath::Q8SlicedK,           FastPath::GdnConvFused,
    FastPath::QsaPrepareFused,     FastPath::Fp8A8Pair,
    FastPath::Fp8A8PairBlock32,
    FastPath::Q4W4A8,              FastPath::SampledMtpDrafts,
    FastPath::GdnF16Accumulate,    FastPath::Q5KSplitMma,
    FastPath::BatchedDecode,       FastPath::BatchedA8Decode,
    FastPath::BatchedA4Decode,     FastPath::BatchedA4ResidualDecode,
    FastPath::BatchedA8ResidualDecode, FastPath::BatchedA8SubsetDecode,
    FastPath::SingleColumnDecode,
    FastPath::ExpertA4Residual,    FastPath::ExpertA4Residual3,
    FastPath::ExpertA4ResidualDecode,
    FastPath::BatchSplitAttention, FastPath::AdaptiveSpeculation,
    FastPath::AdaptiveMtp,         FastPath::AdaptiveDFlash,
    FastPath::DFlash2Pairs,        FastPath::FreeCacheResume,
    FastPath::Preemption,          FastPath::IdleWidePrefill,
    FastPath::GgmlBf16Prefill,     FastPath::F16AccumulatePrefill,
    FastPath::Bf16PromptSplit,     FastPath::RouterMma,
    FastPath::ScreenedLmHead,      FastPath::Fp16GdnState,
    FastPath::BlockVerification,   FastPath::UniverTrees,
    FastPath::GdnTreeClosedForm};

[[nodiscard]] constexpr const char* fast_path_name(FastPath path) noexcept {
    switch (path) {
    case FastPath::PromptPv8: return "prompt-pv8";
    case FastPath::DecodeSplit64: return "decode-split64";
    case FastPath::BatchedDecode: return "batched-decode";
    case FastPath::BatchedA8Decode: return "batched-a8-decode";
    case FastPath::BatchedA4Decode: return "batched-a4-decode";
    case FastPath::BatchedA4ResidualDecode: return "batched-a4-residual-decode";
    case FastPath::BatchedA8ResidualDecode: return "batched-a8-residual-decode";
    case FastPath::BatchedA8SubsetDecode: return "batched-a8-subset-decode";
    case FastPath::SingleColumnDecode: return "single-column-decode";
    case FastPath::ExpertA4Residual: return "expert-a4-residual";
    case FastPath::ExpertA4Residual3: return "expert-a4-residual-3";
    case FastPath::ExpertA4ResidualDecode: return "expert-a4-residual-decode";
    case FastPath::BatchSplitAttention: return "batch-split-attention";
    case FastPath::AdaptiveSpeculation: return "adaptive-speculation";
    case FastPath::AdaptiveMtp: return "adaptive-mtp";
    case FastPath::AdaptiveDFlash: return "adaptive-dflash";
    case FastPath::DFlash2Pairs: return "dflash2-pairs";
    case FastPath::FreeCacheResume: return "free-cache-resume";
    case FastPath::Preemption: return "preemption";
    case FastPath::Q8SlicedK: return "q8-sliced-k";
    case FastPath::GdnConvFused: return "gdn-conv-fused";
    case FastPath::QsaPrepareFused: return "qsa-prepare-fused";
    case FastPath::IdleWidePrefill: return "idle-wide-prefill";
    case FastPath::Fp8A8Pair: return "fp8-a8-pair";
    case FastPath::Fp8A8PairBlock32: return "fp8-a8-pair-block32";
    case FastPath::Q4W4A8: return "q4-w4a8";
    case FastPath::SampledMtpDrafts: return "sampled-mtp-drafts";
    case FastPath::GdnF16Accumulate: return "gdn-f16-accumulate";
    case FastPath::Q5KSplitMma: return "q5-ksplit-mma";
    case FastPath::GgmlBf16Prefill: return "ggml-bf16-prefill";
    case FastPath::F16AccumulatePrefill: return "f16-accumulate-prefill";
    case FastPath::Bf16PromptSplit: return "bf16-prompt-split";
    case FastPath::RouterMma: return "router-mma";
    case FastPath::ScreenedLmHead: return "screened-lm-head";
    case FastPath::Fp16GdnState: return "fp16-gdn-state";
    case FastPath::BlockVerification: return "block-verification";
    case FastPath::UniverTrees: return "univer-trees";
    case FastPath::GdnTreeClosedForm: return "gdn-tree-closed-form";
    }
    return "unknown";
}

// The fast path named by fast_path_name, or empty.
[[nodiscard]] constexpr std::optional<FastPath> parse_fast_path(std::string_view name) noexcept {
    for (const FastPath path : kFastPaths) {
        if (name == fast_path_name(path)) { return path; }
    }
    return std::nullopt;
}

enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

// Qwen3.5's automatic KV capacity leaves this much device memory free unless
// EngineOptions::vram_headroom_bytes says otherwise.
inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode           = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens = 2048;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy automatic() noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

// Where the Vision tower's weights live between encodes (--vision-offload).
enum class VisionOffload : std::uint8_t { Auto, On, Off };

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
    DFlash2,
};

// Automatic tree verification considers, besides the chain, trees of these many columns
// (anchor included) for all-neural rounds of at most kDraftTreeAutoMaxBatch rows.
inline constexpr std::uint32_t kDraftTreeAutoMaxBatch = 4;

[[nodiscard]] constexpr std::array<std::uint32_t, 2>
draft_tree_auto_widths(std::uint32_t draft_tokens) noexcept {
    return {draft_tokens + 5U, draft_tokens + 9U};
}

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // --spec auto (backend None until the Engine resolves it; requires draft_tokens_auto and no
    // draft_min_p): startup chooses, among the drafters the artifact can run, the one measured fastest
    // overall: Qwen3.8-Flash-Next DFlash2 with a draft_model, else MTP; a Qwen3.5-family artifact with
    // offloaded experts MTP; any other Qwen3.5-family artifact DFlash2, then MTP, then DFlash, as
    // bundled. Without one the Engine decodes plain. Engine::options() then reports false and the
    // chosen backend.
    bool backend_auto = false;
    // Startup-fixed K: MTP 1..7 (Qwen3.5/3.6 1..5); DFlash and DFlash2 1..15 (query width K+1).
    // With draft_tokens_auto it is the most a round may draft; 0 there takes the model's most
    // (Qwen3.8-Flash-Next 7, Qwen3.5/3.6 5), which Engine::options() reports.
    std::uint32_t draft_tokens = 0;
    // MTP (--draft-tokens auto): each round chooses its draft length, 0..draft_tokens, from the
    // request's learned acceptance; Qwen3.8-Flash-Next's one-row rounds also verify only the drafts
    // before their cumulative acceptance cut. DFlash and DFlash2: each all-neural round proposes the whole
    // block of draft_tokens and verifies as many as the drafts' calibrated probabilities make
    // fastest. On Qwen3.5/3.6 these are the adaptive-mtp and adaptive-dflash fast paths; where
    // they do not run, startup makes it the configuration's fastest fixed length (Engine::options()
    // then reports false and that length). False: every round drafts and verifies draft_tokens
    // (fewer only near the output limit; draft_min_p defaults to 0).
    bool draft_tokens_auto = false;
    ProposalHead proposal_head = ProposalHead::Full;
    // Qwen3.8-Flash-Next DFlash2 (--draft-model): the companion artifact holding the drafter
    // (tools.convert --components dflash2), loaded beside the main artifact. Qwen3.5/3.6 artifacts
    // bundle their DFlash and DFlash2 drafters and refuse it.
    std::filesystem::path draft_model;
    // Optional target-verified copy proposals with MTP, DFlash or DFlash2.
    // Zero disables the proposer; enabled draft width is 1..63 and minimum match 4..64.
    std::uint32_t ngram_draft_tokens = 0;
    std::uint32_t ngram_min_match    = 12;
    // Copy proposals of rounds of several requests hold at most this many drafts (0: up to
    // ngram_draft_tokens); --ngram-draft-tokens auto sets it where a wide copy costs a batch's other
    // rows more than it saves. Qwen3.5 family only.
    std::uint32_t ngram_batch_draft_tokens = 0;
    // --ngram-draft-tokens auto (Qwen3.8-27B / Qwen3.6-35B-A3B): startup sets ngram_draft_tokens
    // and ngram_min_match to the configuration's measured copy window and minimum match for its
    // drafter, or 0 (no copies) where none was measured faster, including wherever speculation is
    // fixed (every round would verify the n-gram window). Engine::options() then reports false and
    // the resolved values.
    bool ngram_draft_tokens_auto = false;
    // --spec-min-p (MTP, Qwen3.8-Flash-Next DFlash2): a one-row round verifies its drafts up to the
    // first whose drafter probability is below this value; 0 verifies every draft. In [0, 1]; unset
    // is 0. Qwen3.8-Flash-Next MTP takes it with a fixed draft length only (with draft_tokens_auto its
    // argmax chains stop at a cumulative acceptance cut, and startup refuses it); other backends
    // refuse it.
    std::optional<float> draft_min_p;
    // DFlash2, DFlash or MTP tree verification by batch size: entry c-1 is the column count (anchor
    // included, draft_tokens+2..32) a neural round of c rows verifies per row as a draft tree built
    // from the drafter's candidates (DFlash2's lattice, DFlash's most probable tokens per
    // position, leaves beside MTP's sampled drafts); zero keeps that batch size on chain
    // verification. All zero disables trees.
    std::array<std::uint32_t, kMaximumConcurrency> draft_tree_nodes{};
    // Automatic tree verification (draft_tree_nodes all zero): each all-neural round
    // verifies a chain or one of draft_tree_auto_widths, whichever measured round time and
    // same-text acceptance favour at its batch size and context length. A target whose GDN input
    // projections cannot verify trees resolves this to false (chain verification).
    bool draft_tree_auto = false;
    // Most root-to-leaf paths a tree row may hold, 2..8.
    std::uint32_t draft_tree_paths = 8;
    // CPU-only retention, separate from KV. Zero keeps request-local drafting.
    std::size_t ngram_archive_bytes = 0;
    std::size_t ngram_session_bytes = 128ULL << 20;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    // A companion drafter's weights (--draft-model), read after the model's own.
    DraftWeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostContextPin,
    CudaGraphPrepare,
    PrefixCacheLoad,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

enum class DiagnosticLevel : std::uint8_t {
    Debug,
    Info,
    Warning,
    Error,
};

// One runtime diagnostic from the Engine worker or its shutdown.
struct Diagnostic {
    DiagnosticLevel level = DiagnosticLevel::Info;
    std::string message;
};

struct DiagnosticObserver {
    // Receives every diagnostic, from the worker thread, the Engine's helper threads (the startup
    // constraint warm-up, expert offload workers) or the destructor, so it must be thread-safe; the
    // product decides
    // which levels to show. Without a callback, Info and above go to stderr. Callback exceptions
    // are ignored so a logging failure cannot disturb execution.
    std::function<void(const Diagnostic& diagnostic)> callback;
};

// Abandons the Host tier save of a stopping Engine (HybridPrefixCacheOptions::persistent_file)
// from any thread, also while the Engine is being destroyed. Copies share one state: the product
// keeps a copy from the options it passes, beyond the Engine's lifetime.
class PrefixCacheSaveControl {
public:
    enum class Abandon : std::uint8_t {
        // Nothing replaced the previous file; a save in progress deleted its unfinished file.
        Unsaved,
        // The save had already completed.
        Saved,
        // The save did not stop within the timeout; its unfinished file may remain.
        StillWriting,
    };

    PrefixCacheSaveControl();

    // Stops a save in progress, or one not begun yet, from replacing the previous file. Waits at
    // most `timeout` for a save in progress to stop and delete its unfinished file.
    Abandon abandon(std::chrono::milliseconds timeout) const noexcept;

    // The Engine's side. A save writes only after begin() returns true, polls abandoned() while
    // it writes, and then calls end() with whether it replaced the file.
    [[nodiscard]] bool begin() const noexcept;
    [[nodiscard]] bool abandoned() const noexcept;
    void end(bool saved) const noexcept;

private:
    struct State;
    std::shared_ptr<State> state_;
};

// How a snapshot's linear-attention recurrent state (FP32 on the Device) is stored on the Host
// tier. Fp32 is exact; Fp24 (sign, exponent and the 15 high mantissa bits, rounded to nearest
// even) and Bf16 shrink the Host image, and a snapshot restored from the Host then resumes from a
// rounded state. Device snapshots stay exact.
enum class HybridHostStatePrecision : std::uint8_t { Fp32, Fp24, Bf16 };

// Prefix-cache tuning. Every field is optional: Engine construction derives the unset ones from
// max_concurrency, prefill_chunk and whether a Host tier exists (host_capacity_bytes, default
// kDefaultHybridHostCacheBytes, 0 disables it), and Engine::options() reports the effective
// values.
struct HybridPrefixCacheOptions {
    // Device StateImage slots holding inactive snapshots (and tap/endpoint staging). Total Device
    // StateImage capacity is max_concurrency + device_snapshot_slots. Default: one per request
    // lane plus one staging slot with a Host tier, plus two without one (Device slots are then
    // the only snapshot storage).
    std::optional<std::uint32_t> device_snapshot_slots;
    // New prefill state snapshots one request may create. Default 8 with a Host tier, 2 without.
    std::optional<std::uint32_t> max_new_taps;
    // Geometric ladder base G: ladder taps target prompt_tokens - G * 2^k. Default
    // max(4096, 2 * prefill_chunk): ladder taps land on prefill chunk boundaries.
    std::optional<std::uint32_t> tap_ladder_tokens;
    // Ladder taps closer than this to another snapshot on the same path are skipped. Default
    // max(1024, prefill_chunk).
    std::optional<std::uint32_t> tap_min_gap_tokens;
    // Opt-in persistence of the Host tier: saved to this file when the Engine shuts down cleanly
    // and restored from it at startup, but only when it was written for the same artifact, KV and
    // state formats and `persistent_identity` (the product binary's build). Empty disables it.
    std::filesystem::path persistent_file;
    std::string persistent_identity;
    // Lets the product abandon the save at shutdown (infernix-serve: Ctrl+C during the stop).
    PrefixCacheSaveControl persistent_save;
    // With `persistent_file`: also save the Host tier this often while the Engine runs (zero: only
    // at shutdown). A save runs at the first worker boundary once the interval has passed, also
    // while requests run (it pauses generation for its duration), and only when the Host tier has
    // been written since the last save. Like the shutdown save it writes `<file>.tmp` and replaces
    // the file only once complete; stopping the Engine abandons a periodic save in progress.
    std::chrono::seconds persistent_save_interval{0};
    // Queue holds: the snapshot each queued request would resume from, and its path, are evicted
    // after every other cache entry, the request furthest back in the queue first; new snapshots
    // and Host block writes that would need a held snapshot's slabs are skipped instead. False
    // restores plain LRU/GDSF eviction.
    bool queue_holds = true;
    // Tool-call holds: after a request ends with a tool call, the snapshot its next turn resumes
    // from and its path are held like a queued request's (after every queued one) for this long,
    // since the client usually returns within the tool's run time. Zero disables them. Default
    // kDefaultToolCallHold on Qwen3.5; Qwen3.8-Flash-Next has none.
    std::optional<std::chrono::milliseconds> tool_call_hold;
    // Idle-time prefill of a finished turn (hybrid-prefix-cache-spec §7.9): once the Engine has had
    // nothing to run for this long after Engine::prefill_next_turn, it prefills that conversation's
    // history as the next request will render it, as an internal request that yields to any real one
    // at its next unit boundary. Zero disables it. Default kDefaultNextTurnPrefillQuiet with the cache.
    std::optional<std::chrono::milliseconds> next_turn_prefill_quiet;
    // Host form of snapshot recurrent state (Qwen3.5). Changing it changes the Host image size.
    HybridHostStatePrecision host_state_precision = HybridHostStatePrecision::Fp32;
};

// The prefix cache: a content-addressed block tree with sparse state snapshots
// (docs/maintainer/hybrid-prefix-cache-spec.md).
struct ContextCacheOptions {
    // Controls cross-request history reads and writes. Disabled, every request starts from the
    // root and a preempted request recovers by Replay, as it does with the cache on.
    bool enabled = true;
    HybridPrefixCacheOptions hybrid;
    // The pinned Host slab pool that cached KV blocks and state snapshots share (default
    // kDefaultHybridHostCacheBytes, 0 disables the Host tier; 0 when the cache is disabled). This
    // does not bound total process RAM. Engine::options() returns the resolved capacity.
    std::optional<std::size_t> host_capacity_bytes;
    // --host-context-mib auto (host_capacity_bytes unset): Qwen3.8-Flash-Next sizes the Host tier after its
    // experts and n-gram table from the RAM they leave beyond ram_free_bytes, min(8 GiB, 0.4 x left) in 256 MiB
    // steps and at least 1 GiB (after Strata v0.2.6's rule); other models take kDefaultHybridHostCacheBytes.
    // Engine::options() reports the resolved size in host_capacity_bytes.
    bool host_capacity_auto = false;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

// Where Qwen3.8-Flash-Next's n-gram table lives while serving.
enum class NgramResidency : std::uint8_t {
    Auto,   // the whole table in RAM when the RAM ledger still keeps every expert resident, else Volume
    Ram,    // the whole table read into pinned RAM at startup
    Volume, // rows read from the volume on demand through a small row cache
};

// The CLI spelling: auto, ram or volume.
inline NgramResidency parse_ngram_residency(std::string_view value) {
    if (value == "auto") { return NgramResidency::Auto; }
    if (value == "ram") { return NgramResidency::Ram; }
    if (value == "volume") { return NgramResidency::Volume; }
    throw std::invalid_argument("--ngram-residency must be auto, ram or volume");
}

// Which offloaded routed experts (Qwen3.8-Flash-Next, the Qwen3.5-122B-A10B) keep a copy in Host RAM when VRAM
// cannot hold them all.
enum class ExpertRamLayout : std::uint8_t {
    Exclusive, // only the experts VRAM does not hold, plus shadows for what a prefill borrows (the SSD tier)
    Full,      // every expert, whenever the RAM ledger allows it (else the SSD tier with the ledger's share)
};

// The CLI spelling: exclusive or full.
inline ExpertRamLayout parse_expert_ram_layout(std::string_view value) {
    if (value == "exclusive") { return ExpertRamLayout::Exclusive; }
    if (value == "full") { return ExpertRamLayout::Full; }
    throw std::invalid_argument("--expert-ram-layout must be exclusive or full");
}

inline const char* expert_ram_layout_name(ExpertRamLayout layout) {
    return layout == ExpertRamLayout::Exclusive ? "exclusive" : "full";
}

// The layout without --expert-ram-layout, by model: Qwen3.8-Flash-Next exclusive (Full's decode speed with 4-6 GiB
// less RAM, design §19.3.7, 2026-10-29), the Qwen3.5-122B-A10B full (exclusive's 3.9 GiB less RAM cost 1.0 % of its
// decode after a 32K prompt and 0.4 % of prefill, 2026-10-30; its design §11.3).
inline constexpr const char* kExpertRamLayoutDefaults = "exclusive for Qwen3.8-Flash-Next, full for the 122B";

struct EngineOptions {
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    // Qwen3.8-Flash-Next's per-layer n-gram embedding volume. Empty uses `<artifact>.ngram`.
    std::filesystem::path ngram_volume_path;
    NgramResidency ngram_residency = NgramResidency::Auto;
    // Qwen3.8-Flash-Next's saved expert-cache state: read at startup to fill the VRAM expert cache
    // before the first request (warm start), written at stop and every 10 minutes between requests.
    // Without a usable file (none given, a first start, another artifact's) the cache is seeded from the
    // built-in calibrated expert order instead, and only a given file is saved.
    std::filesystem::path expert_state_path;
    // --expert-state off: no file and no seed, so the expert cache starts empty (cold benchmarks).
    bool expert_state_off = false;
    // Qwen3.8-Flash-Next's expert stripe (tools/flash_next/expert_stripe.py): the tail of every routed
    // expert record on a second drive, read beside each record's head in the artifact when the SSD
    // expert tier serves the experts. Empty: whole records from the artifact.
    std::filesystem::path expert_stripe_path;
    // Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B): writes the internal per-round route
    // trace (runtime/expert_offload/route_trace.h) to this file, truncating it at startup. Empty
    // (default): no trace is written and the run is numerically unchanged. Programs without
    // offloaded experts ignore it.
    std::filesystem::path route_trace_path;
    std::size_t grammar_cache_bytes    = 256ULL * 1024 * 1024;
    // Message the model receives when it hits its thinking budget, before the canonical
    // </think> close the frontend appends when the message lacks it. Empty preserves the
    // model's built-in end-of-thinking control suffix.
    std::string thinking_budget_message;
    EnginePurpose purpose              = EnginePurpose::Generation;
    int device                         = 0;
    std::uint32_t max_context          = 2048; // Logical ceiling of one request or score window.
    // Startup-fixed runtime YaRN override, finite [1,4]; 1 preserves native RoPE.
    // Extends only the allowed ceiling, never max_context itself or artifact metadata.
    float rope_yarn_factor             = 1.0F;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    // Device memory startup sizing leaves free; empty selects the model's automatic value.
    // Qwen3.5: what --kv-capacity auto leaves after the KV cache (requires it; automatic 1 GiB).
    // Qwen3.8-Flash-Next: what the expert cache leaves for the display and other programs once
    // every fixed allocation is made (automatic: 1 GiB with a display attached, 256 MiB headless).
    std::optional<std::size_t> vram_headroom_bytes;
    // Qwen3.8-Flash-Next on Windows: also size the expert cache into the physical VRAM the OS
    // budget withholds (~0.6 GiB on an RTX 5090), measured at startup (--vram-past-budget). That
    // memory may be moved to system memory when another program needs VRAM.
    bool vram_past_budget = false;
    // Qwen3.8-Flash-Next, approximate (--expert-tail-skip R, 1..9; 0 off): in decode and verification
    // rounds, a routed expert not resident in VRAM that every token of the round routes at top-10
    // rank R or lower (0-based, the smallest router weights) is neither copied nor computed; it adds
    // zero and the other experts keep their weights. Faster decode when experts miss VRAM, at the
    // cost of changed and slightly worse output that depends on the expert cache's contents. Strata
    // does the same by default at rank 7 (STRATA_ROUTE_TAIL_SKIP=7).
    std::uint32_t expert_tail_skip = 0;
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    // Prompt tokens per prefill call, a multiple of 128; 0 takes the model's default: 16,384 for
    // Qwen3.8-Flash-Next (its exact-route prefill measured identical across chunks), 1,024
    // otherwise.
    std::uint32_t prefill_chunk        = 0;
    // INT8 KV prefills with the fast prompt-attention kernel and rounds prefill_chunk down to whole
    // prompt-attention waves. True selects the original INT8 prompt kernel at the requested chunk;
    // it requires the INT8 KV cache.
    bool original_int8_prefill_kernel  = false;
    // Workspace (MiB, [0,16384]) the FP32 partials of one prompt-attention launch may take when
    // it splits its keys across SMs (the INT8 and NVFP4 fast prompt kernels, FP8 and K8V4).
    // Larger budgets speed up prompt chunks of about 1-2K tokens over long cached context and take
    // device memory from the KV cache; BF16 KV and the original INT8/NVFP4 kernels ignore it.
    std::uint32_t prefill_split_workspace_mib = kDefaultPrefillSplitWorkspaceMiB;
    // NVFP4 KV prefills over more than 768 visible keys with the fast prompt-attention kernel
    // (block-scaled FP4 QK). True selects the tiled NVFP4 prompt kernel; it requires the NVFP4
    // KV cache.
    bool original_nvfp4_prefill_kernel = false;
    // The lossless route (--lossless; docs/maintainer/lossless-route.md), the quality gates'
    // reference on Qwen3.5/3.6/3.8 (Qwen3_5ForCausalLM and Qwen3_5MoeForCausalLM): the model's
    // Hugging Face forward operation by operation, each correctly rounded to the dtype it has
    // there (weights decoded exactly, sums exact, transcendental functions correctly rounded), so
    // every output is a function of the request and the artifact alone; logits in logits_dtype (each
    // exact sum correctly rounded to it), no fast path.
    // CausalScoring scores each window in one pass (prefill_chunk covering max_context); Generation
    // (Qwen3.5/3.6/3.8 and Qwen3.8-Flash-Next) takes text requests with bf16 KV (V stored as BF16) and
    // no speculation.
    bool lossless                      = false;
    // The design reference (--reference; docs/maintainer/lossless-route.md §3b), the quality gate's second
    // reference: the lossless route (it requires `lossless`) with the configuration's declared lossy steps
    // applied exactly: each projection input's activation quantization that its artifact's Use permits (and
    // a routed-expert bank's W4A4, FP4-Local's with expert_a4_activations), and the KV format's codec.
    // CausalScoring only; its output is a function of the request, the artifact and those steps.
    bool reference                     = false;
    // The noise twin (IQS-2's noise floor, docs/maintainer/model-quality.md §6): nonzero makes every
    // rounding of the lossless route (or the design reference) a stochastic rounding of the same exact value,
    // seeded by this value: an ideal BF16 implementation off by at most one unit in the last place at each
    // rounding. CausalScoring with `lossless` only; 0 (the default) is the lossless route itself. The rounding
    // mode is process-wide device state, so one process runs one seed.
    std::uint64_t lossless_noise_seed  = 0;
    // The attribution of the noise (ops::LosslessNoiseUnit): the bit mask of the operation groups the noise
    // applies to (default all) and its strength in (0, 1] (default full), both with a nonzero seed only.
    std::uint32_t lossless_noise_units = 0xFFU;
    float lossless_noise_scale         = 1.0F;
    // Qwen3.8-Flash-Next artifacts whose routed experts offer 4-bit activations as an option (only
    // nvidia/Qwen3.8-Flash-Next-FP4-Local, whose experts NVIDIA calibrated for them): run the routed
    // experts W4A4 with their stored activation scales (--expert-a4-activations) instead of the
    // default 16-bit activations. Other artifacts refuse it, as does the lossless route without the
    // design reference.
    bool expert_a4_activations         = false;
    // The 8-bit activation routes (--a8-activations; approximate, opt-in, never a default), each where it
    // measured faster than the route it replaces (or is being measured; docs/maintainer/qwen3_8-flash-next-design.md
    // §19.3.31). Decode and verification calls: int8 codes with an FP32 scale per 32 values (llama.cpp's Q8_1
    // form) in a GGUF import's dense projections (calls of 1-31 columns) and in routed experts (GGUF and NVFP4
    // W4A16 banks, calls of at most 128 columns) with their CPU-served misses (the CPU returns the GPU's bits).
    // Prefill calls: on the same codec, a GGUF import's dense projections (calls of 32 or more columns) and the
    // same routed experts (calls of more than 128 columns, in place of expert-a4-residual's A4 terms where that
    // runs); a groupwise Q4 MLP gate/up's int8 activations per token (q4-w4a8's route, calls of 256 or more
    // columns); MXFP8 dense projections' two E4M3 terms per 32 values (fp8-a8-pair-block32's route, in place of
    // f16-accumulate-prefill). Every other call keeps its route. An artifact with none of these (the
    // NVFP4 Qwen3.8-27B, whose FP8 and NVFP4 projections already take the activations their artifact permits, the
    // 35B-A3B, Flash-Next Dense8 and the official FP8 import), the reproducible route and the lossless route
    // refuse it.
    bool a8_activations                = false;
    // Qwen3.5-family artifacts (--embedding-host): keep the token embedding in pinned host memory and
    // gather its rows zero-copy, as artifacts with offloaded experts already do, so its VRAM (1.18 GiB on
    // the 27B) goes to the KV and prefix caches. Same bits; a prefill chunk reads its rows over the link.
    bool embedding_host                = false;
    // Run every fast path qualified for this configuration (model geometry, activation precision
    // and KV format): routes whose arithmetic differs from the reproducible route but whose measured cost
    // stays within Infernix's numerical acceptance policy (docs/maintainer/model-quality.md
    // section 6; the registry is 6.7). LoadSummary::fast_paths reports the selection. False
    // (--reproducible) runs the reproducible route: on a given engine build and GPU a request's output
    // depends only on the request (with its seed), the artifact and these options, not on other requests,
    // concurrency, cache state or resource pressure; other engine versions may give other bits. The
    // lossless route (`lossless`), whose bits never change, requires it.
    bool fast_paths                    = true;
    // Qualification only (the quality tools, never a product route): also run these fast paths
    // whether or not they are qualified for this configuration, so their cost can be measured.
    // Each must apply to the configuration (its kernel must run in it).
    std::vector<FastPath> fast_path_candidates;
    KvCacheStorage kv_cache            = KvCacheStorage::BFloat16;
    // The head's logits on every route (--logits-dtype): FP32, or BF16 for comparisons with engines and
    // references that round them; a startup choice (the round buffers and CUDA Graphs are planned for it).
    LogitsDtype logits_dtype           = LogitsDtype::FP32;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // --ram-free-mib: physical Host memory everything Infernix plans in RAM (weights, experts, the n-gram table,
    // the prefix cache's Host tier, buffers) leaves free for the OS and other programs; startup refuses a lock
    // that would take it, the n-gram table goes to RAM only beside it, and an automatic Host tier sizes from
    // what is left beyond it.
    std::uint64_t ram_free_bytes = kDefaultRamFreeBytes;
    // Qwen3.8-Flash-Next (--expert-ram-mib): at most this much Host RAM for the routed experts.
    // Less than every expert needs selects the SSD tier: the experts not in RAM are read from the
    // artifact when used. Empty: what the RAM ledger leaves after the reserve and every other pin.
    std::optional<std::uint64_t> expert_ram_bytes;
    // Offloaded experts (--expert-ram-layout), without expert_ram_bytes: Exclusive keeps in RAM only the experts VRAM
    // does not hold (plus shadows for a prefill's loan and the KV growth), Full a copy of every expert. Empty: the
    // model's default (kExpertRamLayoutDefaults).
    std::optional<ExpertRamLayout> expert_ram_layout;
    // Qwen3.8-Flash-Next (--expert-pin-mib): at most this much of the experts' RAM page-locked for
    // the GPU; the rest stays in RAM unpinned and the CPU serves it. Empty: what Windows lets the
    // process page-lock (the GPU's shared memory) after every other pin; no limit elsewhere.
    std::optional<std::uint64_t> expert_pin_bytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    // Vision offload keeps the vision tower in pinned system RAM instead of device memory and
    // streams it through borrowed device memory per encode window. Auto: on with offloaded experts
    // (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B: the expert cache lends the window's memory), off for
    // the other Qwen3.5 artifacts.
    VisionOffload vision_offload = VisionOffload::Auto;
    // Upper bound on merged tokens per vision item; zero leaves the compiled limit.
    std::uint32_t vision_max_merged_tokens = 32768;
    bool use_cuda_graph                    = true;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    StartupObserver startup_observer;
    DiagnosticObserver diagnostic_observer;
};

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 20;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

// What the reasoning-loop guard does when the open thinking keeps repeating whole passages (every
// 512 thinking tokens, the share of the last 2,000 words inside a 12-word passage seen three times
// reaches 25 %).
enum class ReasoningLoopAction : std::uint8_t {
    Off,      // no check
    Stop,     // end the request there (FinishReason::OutputLimit)
    Conclude, // inject the thinking-close control (as a reached thinking budget does) and let the
              // model answer; Stop when the remaining output budget cannot hold it and one token
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the Qwen thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
    ReasoningLoopAction loop = ReasoningLoopAction::Off;
};

// Most tokens one prompt readout may name.
inline constexpr std::uint32_t kMaximumReadoutTokens = 256;

// Constrained generation limits: output steps, tokens one step may permit, and distinct
// multi-token sets one request may use (a one-token step forces that token and needs no set).
inline constexpr std::uint32_t kMaximumConstraintSteps   = 64;
inline constexpr std::uint32_t kMaximumConstraintChoices = 16;
inline constexpr std::uint32_t kMaximumConstraintSets    = 4;
// Most tokens one constrained step may observe.
inline constexpr std::uint32_t kMaximumObservedTokens    = 256;

// Output step i may only produce a token of steps[i]; a one-token step forces it. Generation
// stops after the last step at the latest, so requested_output_tokens must not exceed the step
// count. Sampling still chooses among the permitted tokens (greedy picks the most probable).
struct TokenConstraint {
    std::vector<std::vector<TokenId>> steps;
    // Empty, or one list per step: distinct tokens (at most kMaximumObservedTokens) whose
    // log-probabilities under the step's whole next-token distribution its ConstrainedDraw
    // reports. Measurement only: observing never changes what a step may produce.
    std::vector<std::vector<TokenId>> observe;

    [[nodiscard]] bool empty() const noexcept { return steps.empty(); }
};

// One constrained output token and the model's distribution over its step's permitted tokens.
struct ConstrainedDraw {
    TokenId token = 0;
    // Renormalized over the step's permitted tokens, in the step's order (temperature 1, before
    // penalties).
    std::vector<float> probabilities;
    // The permitted tokens' share of the whole next-token distribution.
    float mass = 0.0F;
    // For a step with an observe list: the log-probabilities of its tokens in their order, and
    // the step's largest log-probability (both over the whole distribution, temperature 1, from
    // the logits the sampler reads, before penalties). Empty and 0 otherwise.
    std::vector<float> observed_logprobs;
    float top_logprob = 0.0F;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
    // Distinct public tokens whose log-probabilities at the prompt's next position the request
    // reports as GenerationResult::readout, at most kMaximumReadoutTokens. Empty requests none.
    std::vector<TokenId> readout_tokens;
    // Restricts each output token to its step's permitted set. Empty constrains nothing.
    TokenConstraint constraint;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds only Qwen's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
    // Opt-in recovery of complete Qwen calls whose wrapper or suffix markup is malformed; see the
    // tool-call parser tolerant mode. Off preserves the strict all-or-nothing behavior.
    bool tolerant_tool_calls = false;
};

enum class OutputConstraintKind : std::uint8_t { Grammar, JsonObject, JsonSchema, Choice, Regex };

// Constrains generated content; Chat reasoning retains the model's framing. Source is owning
// GBNF, JSON Schema or regex text. Choice owns literal alternatives; JsonObject has no payload.
struct OutputConstraint {
    OutputConstraintKind kind = OutputConstraintKind::Grammar;
    std::string source;
    std::vector<std::string> choices;

    [[nodiscard]] static OutputConstraint grammar(std::string source) {
        return {OutputConstraintKind::Grammar, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint json_object() {
        return {OutputConstraintKind::JsonObject, {}, {}};
    }

    [[nodiscard]] static OutputConstraint json_schema(std::string source) {
        return {OutputConstraintKind::JsonSchema, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint choice(std::vector<std::string> values) {
        return {OutputConstraintKind::Choice, {}, std::move(values)};
    }

    [[nodiscard]] static OutputConstraint regex(std::string pattern) {
        return {OutputConstraintKind::Regex, std::move(pattern), {}};
    }

    bool operator==(const OutputConstraint&) const = default;
};

enum class ToolChoiceMode : std::uint8_t { Auto, None, Required };
// Basic protects every tool call. Automatic applies constraints only for strict tools or an
// explicit selection/count policy.
enum class ToolConstraintMode : std::uint8_t { Automatic, Basic };

// Declarations belong to PromptOptions. Selection and cardinality affect this generation only.
struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::optional<std::vector<std::string>> allowed_names;
    bool parallel                  = true;
    ToolConstraintMode constraints = ToolConstraintMode::Basic;
};

struct NgramSessionHints {
    // Explicit local conversation identity, at most 256 bytes. Never inferred from
    // prompt similarity or the KV session key. Empty means request-local drafting.
    std::string key;
    bool reset = false;
    std::string parent;
    std::uint64_t parent_generation = 0;
    // Retention-enabled requests mix the sampling seed with fresh request entropy:
    // prior generated proposals must not reuse the random draws that produced them.
};

struct RequestOptions {
    std::optional<OutputConstraint> constraint;
    ToolChoice tool_choice;
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
    NgramSessionHints ngram_session;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Model-origin structured output. Protocol adapters own any wire-level call identifier.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
};

// Diagnostics for model-origin tool-call markup. Free generation uses schema-guided value
// normalization and can return malformed markup as text. Constrained calls use the model's
// parameter encoding contract and publish only complete calls.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    DuplicateParameter,
    InvalidToolName,
    UndeclaredTool,
    TrailingContent,
    // Tolerant recovery discarded a trailing suffix after an otherwise complete call, or kept a
    // call whose closing tags were cut off at the region end. A structured response was still
    // produced, so this is surfaced for transparency, not as a fallback-to-text failure.
    TruncatedTail,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::DuplicateParameter:
        return "duplicate_parameter";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    case ToolCallParseFallbackReason::TrailingContent:
        return "trailing_content";
    case ToolCallParseFallbackReason::TruncatedTail:
        return "truncated_tail";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen                              = false;
    std::uint32_t structured_call_count           = 0;
    std::uint32_t empty_arguments_omitted         = 0;
    std::uint32_t schema_mismatch_arguments       = 0;
    std::uint32_t duplicate_parameters_repaired   = 0;
    ToolCallParseFallbackReason fallback_reason   = ToolCallParseFallbackReason::None;
    // Tolerant tool-call mode turned output the strict parser rejects into structured calls.
    bool tolerant_recovered = false;
    // Tolerant tool-call mode took the structured calls from a turn that ended inside its thinking.
    bool recovered_from_reasoning = false;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

// Whether the active chat template renders an instruction role at each conversation position,
// found by rendering each position once while the template loads. Agent clients legitimately send
// instructions where a checkpoint template refuses them: Claude Code appends a mid-conversation
// system entry, and OpenAI clients carry a leading `developer` turn. Preparation then renders the
// turn's text under a role the template does render, never changing text, part count or position.
//
// The two instruction roles are treated differently on purpose. `system` is the role the Qwen
// templates themselves use for instructions, so a template that refuses it at every position is
// stating its own rule and the request fails as that template asks; only a positional rule (the
// checkpoint guard, which allows a system turn first and nowhere else) is worked around.
// `developer` is an instruction alias from the client protocols that many Qwen checkpoint
// templates have no branch for at all, so it is always rendered under a role the template has.
struct ChatTemplateRoleSupport {
    // Renders a System turn as the first message. False together with a false late_system is a
    // template's own rule about instructions, which preparation leaves alone.
    bool leading_system = true;
    // Renders a System turn after the first message. Otherwise, and only while it does render one
    // first, preparation renders that turn as a User turn.
    bool late_system = true;
    // Renders a developer turn after the first message. Otherwise preparation renders it as a User
    // turn.
    bool late_developer = true;
    // Renders a developer turn as the first message. Otherwise preparation renders it as a System
    // turn when the template opens with one, and as a User turn otherwise.
    bool leading_developer = true;

    // True when preparation lowers a System turn that arrived after the first message.
    [[nodiscard]] constexpr bool lowers_late_system() const noexcept {
        return !late_system && leading_system;
    }

    [[nodiscard]] friend constexpr bool
    operator==(const ChatTemplateRoleSupport&, const ChatTemplateRoleSupport&) noexcept = default;
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
    // The conversation's history as its next request will render it: the messages end with the
    // finished assistant reply, which renders as history before the next turn (a tool result when
    // the reply calls tools, else a user turn), and the prompt ends with that turn's <|im_start|>.
    // Prefilled by the Engine's idle-time pass (Engine::prefill_next_turn); text only.
    NextTurnHistory,
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    // Declarations the prompt does not list (deferred tools). Tool-call parsing, selection and
    // constraints cover them after tool_jsons, in this order, while the template's tool section
    // omits them; a message that loads one carries its definition in its own content.
    std::vector<std::string> deferred_tool_jsons;
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

// Where the rendered prompt has reusable structure: the prefix cache places state snapshots
// (taps) at these frontiers.
struct ContextCacheHints {
    std::vector<PromptCacheMarker> markers;
    // Protocols with their own automatic/explicit write policy disable the Engine's structural
    // candidates (the tool-definition boundary).
    bool allow_engine_automatic_shared_prefixes = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

// A chat rendered with the artifact's template and tokenized. generated[i] is 1 when token i belongs to an
// assistant message (after its "<|im_start|>assistant" header, through its "<|im_end|>"), the tokens the model
// itself writes, and 0 for the system prompt, tool schemas, user turns and tool responses: text the model is
// never trained to predict and whose predictions are meaningless (and numerically fragile).
struct ChatTokens {
    std::vector<TokenId> ids;
    std::vector<std::uint8_t> generated;
};

enum class RequestErrorKind : std::uint8_t {
    InvalidToolConstraint,
    InvalidGrammar,
    InvalidChoice,
    InvalidRegex,
    InvalidJsonSchema,
    UnsupportedJsonSchema,
    UnsatisfiableJsonSchema,
    ConstraintDeadEnd,
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

enum class RequestErrorSource : std::uint8_t { OutputConstraint, Tools };

// A prompt longer than the Engine's max_context. Counting stops at a bound past the limit, so
// prompt_tokens is exact only when `exact`; otherwise the prompt has more than prompt_tokens.
// With `vision`, the counts are the prompt's images and videos together (their vision tokens)
// against the processor's per-prompt vision budget: an aggregate overflow a client recovers from by
// compacting, as from a long prompt (a single item over Vision capacity stays MediaBudgetExceeded).
struct ContextLengthDetails {
    std::uint64_t prompt_tokens = 0;
    std::uint32_t max_context   = 0;
    bool exact                  = true;
    bool vision                 = false;
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message, std::string pointer = {},
                 RequestErrorSource source = RequestErrorSource::OutputConstraint)
        : std::invalid_argument(std::move(message)), kind_(kind), pointer_(std::move(pointer)),
          source_(source) {}

    // RequestErrorKind::ContextLengthExceeded with its counts.
    explicit RequestError(ContextLengthDetails details)
        : std::invalid_argument("prepared prompt has " + std::string(details.exact ? "" : "more than ") +
                                std::to_string(details.prompt_tokens) +
                                (details.vision ? " vision tokens, exceeding the prompt vision budget "
                                                : " tokens, exceeding Engine max_context ") +
                                std::to_string(details.max_context)),
          kind_(RequestErrorKind::ContextLengthExceeded), source_(RequestErrorSource::OutputConstraint),
          context_length_(details) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

    [[nodiscard]] RequestErrorSource source() const noexcept { return source_; }

    [[nodiscard]] const std::string& pointer() const noexcept { return pointer_; }

    [[nodiscard]] const std::optional<ContextLengthDetails>& context_length() const noexcept {
        return context_length_;
    }

private:
    RequestErrorKind kind_;
    std::string pointer_;
    RequestErrorSource source_;
    std::optional<ContextLengthDetails> context_length_;
};

// A submitted image's or video's display size in pixels, before any Vision resizing.
struct MediaGeometry {
    std::int32_t width  = 0;
    std::int32_t height = 0;

    [[nodiscard]] friend bool operator==(MediaGeometry, MediaGeometry) noexcept = default;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
    // One entry per media item, in prompt order.
    std::vector<MediaGeometry> media;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
    // Instruction turns whose role preparation lowered because the active chat template cannot
    // render them where they arrived (see ChatTemplateRoleSupport). 0 when the template renders
    // every instruction position it received.
    std::size_t instruction_role_lowerings = 0;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

// One step of a tool call as the constrained tool route generates it (calls of free tool output
// arrive whole, in GenerationResult::tool_calls). A call's first delta names its tool; every delta
// carries the next bytes of its arguments JSON, and the call's last one is marked complete. A
// call's deltas concatenate to the arguments_json the result holds for it; a call cut by the output
// limit, the context or a cancellation never completes and stays out of the result.
struct ToolCallDelta {
    std::uint32_t index = 0; // the call's position among the turn's calls
    std::string name;        // the tool's name, on the call's first delta only
    std::string arguments;
    bool complete = false;
};

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputDelta, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputDelta for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputDelta delta)                 = 0;
    // Published after the content that precedes the call, in generation order.
    virtual void tool_call(ToolCallDelta delta)             = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

enum class GenerationSchedulingTransition : std::uint8_t {
    PauseStarted,
    Paused,
    RestoreStarted,
    Restored,
    ReplayComplete,
    RecoveryComplete,
    Terminal,
};

// A paused request recovers by Replay from the deepest state the prefix cache holds.
enum class GenerationRecoveryRoute : std::uint8_t { None, Replay };

// Sparse lifecycle observations captured by the Engine worker and delivered by wait() on the
// consumer thread. Counter deltas between boundaries, less the request's own deltas, measure
// other requests' completed work during recovery. Decode includes forced control tokens.
struct GenerationSchedulingObservation {
    GenerationSchedulingTransition transition = GenerationSchedulingTransition::PauseStarted;
    GenerationRecoveryRoute route             = GenerationRecoveryRoute::None;
    std::uint64_t engine_request_id           = 0;
    std::uint64_t steady_ns                   = 0;
    std::uint64_t elapsed_ns                  = 0;
    std::uint64_t preemption_index            = 0;
    std::uint64_t global_prefill_tokens       = 0;
    std::uint64_t global_decode_tokens        = 0;
    std::uint64_t global_replayed_tokens      = 0;
    std::uint64_t request_prefill_tokens      = 0;
    std::uint64_t request_decode_tokens       = 0;
    std::uint64_t request_replayed_tokens     = 0;
};

using GenerationSchedulingObserver = std::function<void(const GenerationSchedulingObservation&)>;

// Emitted once on the first accepted model/control token, delivered by wait() in either consumer
// mode. Submission elapsed includes queueing and binding; preparation is reported separately.
struct GenerationFirstTokenObservation {
    double prepare_seconds              = 0.0;
    double elapsed_since_submit_seconds = 0.0;
    double queue_wait_seconds           = 0.0;
};

using GenerationFirstTokenObserver = std::function<void(const GenerationFirstTokenObservation&)>;

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live output timings and prompt progress require a
// Streaming consumer; phase timings and scheduling observations also support Aggregate consumers.
struct GenerationObservationOptions {
    bool phase_timings   = false;
    bool live_timings    = false;
    bool prompt_progress = false;
    // Optional for either consumer mode. No scheduling events are retained when unset.
    GenerationSchedulingObserver scheduling;
    GenerationFirstTokenObserver first_token;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

// Request-stage wall timings retained for end-to-end latency/rate reporting. Prefill/decode are
// Program execution elapsed time and include Device completion waits; total also includes queueing
// and other request lifetime. They are not Host-work phases. GenerationEngineTiming below is the
// direct, mutually-exclusive Host observation contract.
struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    // decode_seconds with each compact-batch unit's elapsed time divided among the requests it
    // served. Unlike decode_seconds, summing it over concurrent requests counts each unit once, so
    // summed decode tokens over summed shares is aggregate decode throughput.
    double decode_share_seconds = 0.0;
    // Prompt wall time begins at the successful initial binding attempt and ends at the first
    // accepted output token. Generation spans the first through last accepted output token and
    // therefore has N-1 token intervals.
    double prompt_wall_seconds     = 0.0;
    double generation_wall_seconds = 0.0;
    double total_seconds           = 0.0;
    // Vision offload: duration and traffic of the encode window that streamed the vision tower
    // from system RAM through borrowed device memory.
    double vision_offload_window_seconds       = 0.0;
    double vision_offload_evict_seconds        = 0.0;
    double vision_offload_restore_seconds      = 0.0;
    std::uint64_t vision_offload_evicted_bytes = 0;
    std::uint64_t vision_offload_staged_bytes  = 0;
};

// Wall elapsed time directly observed in Engine-owned regions. "Exposed" values are latency
// exposure: every active request delayed by one compact-batch unit observes that unit's full
// elapsed time, so values from concurrent requests must not be summed. Device wait is reported
// separately from Host-active work.
struct GenerationEngineTiming {
    double queue_wait_seconds                    = 0.0;
    double engine_boundary_exposed_seconds       = 0.0;
    double program_submit_exposed_seconds        = 0.0;
    double program_post_exposed_seconds          = 0.0;
    double engine_commit_output_exposed_seconds  = 0.0;
    double engine_maintenance_exposed_seconds    = 0.0;
    double device_wait_exposed_seconds           = 0.0;
    double constraint_draft_wait_exposed_seconds = 0.0;
    double decode_host_exposed_seconds           = 0.0;
    double decode_device_wait_exposed_seconds    = 0.0;
    std::uint64_t prefill_units                  = 0;
    std::uint64_t decode_rounds                  = 0;
    std::uint64_t control_units                  = 0;
};

// Request-owned scheduling observations. Restore counters count completed restorations: a
// snapshot restore resumed from a prefix-cache snapshot at the paused frontier itself, a replay
// restore recomputed past the deepest cached state. replayed_tokens counts tokens actually
// recomputed, including previously generated output; these tokens are separate from initial
// prompt prefill and delivered output. paused_ns covers pause preparation, waiting and binding
// restoration, excluding replay computation.
struct GenerationSchedulingStats {
    std::uint64_t preemptions       = 0;
    std::uint64_t snapshot_restores = 0;
    std::uint64_t replay_restores   = 0;
    std::uint64_t replayed_tokens   = 0;
    std::uint64_t paused_ns         = 0;
};

// Request-owned execution work. GPU stream intervals overlap Host submission and completion
// waits; they are evidence about Device work, not an additional wall-time component.
struct GenerationWorkTiming {
    double submit_seconds = 0.0;
    double wait_seconds   = 0.0;
    double post_seconds   = 0.0;
    double gpu_seconds    = 0.0;
};

// Frozen immediately before the first nonempty public output delta is published. This boundary
// differs from the first accepted model token and from the client's first HTTP output. Elapsed
// starts at Engine submission, excluding Frontend preparation. Initial queue, initial binding,
// paused time and the remaining resident interval partition this Engine wall time.
struct GenerationFirstOutputTiming {
    double elapsed_seconds         = 0.0;
    double initial_binding_seconds = 0.0;
    GenerationEngineTiming engine;
    GenerationWorkTiming prefill;
    GenerationWorkTiming replay;
    GenerationSchedulingStats scheduling;
    std::uint32_t computed_prefill_tokens = 0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
    // Qwen3.5-family MTP, DFlash and DFlash2: entry k-1 counts the all-neural rounds (fallback
    // steps excluded) that verified k drafts. Every one verifies the configured window (fewer near
    // the output limit) unless adaptive-mtp or adaptive-dflash chooses it per round. Empty for other
    // models.
    std::vector<std::uint64_t> rounds_by_draft_length;
    std::uint64_t ngram_rounds                  = 0;
    std::uint64_t ngram_drafted_tokens          = 0;
    std::uint64_t ngram_accepted_tokens         = 0;
    std::uint64_t ngram_archive_rounds          = 0;
    std::uint64_t ngram_archive_drafted_tokens  = 0;
    std::uint64_t ngram_archive_accepted_tokens = 0;
    // Tree-verified rounds, those whose accepted path left the main chain, and the drafts those
    // paths accepted after leaving it (what the main chain alone would not have accepted).
    std::uint64_t tree_rounds               = 0;
    std::uint64_t tree_side_rounds          = 0;
    std::uint64_t tree_side_accepted_tokens = 0;
    // Every verified round in order (fallback steps excluded): the drafts it verified, how many it
    // accepted and whether they were n-gram (copy) drafts. Qwen3.5-family MTP and DFlash rounds and
    // Qwen3.8-Flash-Next rounds; empty for other models.
    struct RoundStep {
        std::uint16_t drafted  = 0;
        std::uint16_t accepted = 0;
        bool ngram             = false;
    };
    std::vector<RoundStep> round_steps;
};

struct ThinkingBudgetStats {
    std::optional<std::uint32_t> configured_budget;
    // Model-origin tokens accepted while capped thinking remained open.
    std::uint32_t model_thinking_tokens = 0;
    // Complete tokenizer-derived target-control suffix committed by Engine.
    std::uint32_t injected_tokens = 0;
    bool applied                  = false;
    // The reasoning-loop guard fired: after this many model-origin thinking tokens, at this
    // repeated-passage coverage (ThinkingControlOptions::loop decided the action).
    bool loop_detected               = false;
    std::uint32_t loop_thinking_tokens = 0;
    float loop_coverage              = 0.0F;
};

enum class ConstraintCacheAccess : std::uint8_t { Hit, Built, Waited };
enum class ConstraintOutputBranch : std::uint8_t { Undecided, Content, Tools };

// State describes the committed output language, including an assistant continuation prefix.
// Work includes speculative lookahead that was subsequently rolled back. Times are subintervals
// of existing request timings, not extra latency to add to them.
struct ConstraintObservation {
    ConstraintOutputBranch branch   = ConstraintOutputBranch::Undecided;
    bool complete                   = false;
    bool terminated                 = false;
    ConstraintCacheAccess cache     = ConstraintCacheAccess::Hit;
    bool timings_collected          = false;
    double prepare_seconds          = 0.0;
    double mask_seconds             = 0.0;
    double matcher_seconds          = 0.0;
    std::uint64_t mask_positions    = 0;
    std::uint64_t mask_upload_bytes = 0;
};

// A root start, or a prefix-cache snapshot: a previous generation's endpoint, or a prefill tap.
enum class PrefixReusePath : std::uint8_t {
    Root,
    HybridEndpoint,
    HybridSnapshot,
};

enum class AdmissionFallbackReason : std::uint8_t {
    None,
    SourceInvalid,
    CapacityLimit,
    IsolatedCapacity,
};

struct GenerationAdmissionStats {
    std::uint32_t preferred_reused_tokens = 0;
    // Subset of initial queue wait, not additional TTFT. Includes waiting for a lane
    // after a useful source has been selected.
    double source_wait_seconds              = 0.0;
    AdmissionFallbackReason fallback_reason = AdmissionFallbackReason::None;
};

struct NgramArchiveStats {
    bool enabled              = false;
    bool bound                = false;
    bool published            = false;
    std::uint64_t generation  = 0;
    std::size_t sources       = 0;
    std::size_t session_bytes = 0;
    std::size_t total_bytes   = 0;
    // Effective seed after request-domain separation; archive state is also needed for replay.
    std::optional<std::uint64_t> sampling_seed;
};

// The next-token distribution after the whole prompt, read before the first output token.
struct PromptReadout {
    // log p(readout_tokens[i] | prompt) over the model's public tokens, in request order.
    std::vector<float> logprobs;
    // The unrestricted most probable next token and its log-probability.
    TokenId top_token = 0;
    float top_logprob = 0.0F;
};

// Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B): one request's routed-expert lookups and
// those its VRAM expert cache served.
struct ExpertCacheStats {
    std::uint64_t routed = 0;
    std::uint64_t hits   = 0;
    // Of those, the lookups of decode and verification rounds.
    std::uint64_t decode_routed = 0;
    std::uint64_t decode_hits   = 0;
};

// Qwen3.8-Flash-Next: the VRAM expert cache's cumulative counters since startup and its current
// size (RuntimeStats::expert_cache, exported by /metrics). Absent for models without one.
struct ExpertCacheTotals {
    std::uint32_t frames     = 0; // frames backed now, one routed expert each
    std::uint64_t routed     = 0; // routed (layer, expert) uses
    std::uint64_t hits       = 0; // of those, served from a frame
    std::uint64_t promotions = 0; // experts copied into frames by the cache policy
    std::uint64_t landed     = 0; // staged misses adopted into frames
    std::uint64_t cpu_served = 0; // misses computed by the CPU expert team
};

struct GenerationResult {
    // Unique within this Engine instance; diagnostic correlation only.
    std::uint64_t engine_request_id = 0;
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    std::uint32_t reused_prompt_tokens = 0;
    // Completed initial prefill, excluding cached tokens and separately counted replay work.
    std::uint32_t computed_prefill_tokens = 0;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::Root;
    GenerationTimings timings;
    GenerationEngineTiming engine_timing;
    GenerationSchedulingStats scheduling;
    GenerationAdmissionStats admission;
    std::optional<GenerationFirstOutputTiming> first_output_timing;
    SpeculativeStats speculative;
    NgramArchiveStats ngram_archive;
    ThinkingBudgetStats thinking;
    // Present for models that cache routed experts in VRAM (Qwen3.8-Flash-Next, Qwen3.5-122B-A10B)
    // and requests that finished normally.
    std::optional<ExpertCacheStats> expert_cache;
    // With the prefix cache, for a request that finished normally: the prompt tokens past its reused
    // prefix whose state its snapshots made reusable for a later request (Anthropic's
    // cache_creation_input_tokens). Absent without the cache or for a prompt that does not publish.
    std::optional<std::uint32_t> cache_created_prompt_tokens;
    // Present exactly when the request named readout tokens and its prompt completed.
    std::optional<PromptReadout> readout;
    // One entry per generated token of a constrained request, in output order.
    std::vector<ConstrainedDraw> constrained_draws;
    std::optional<ConstraintObservation> constraint;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

// Logical regions within the one physical workspace allocation. These byte values describe
// layout and live extents and must not be added to workspace.capacity_bytes.
struct VisionWorkspaceMemorySummary {
    std::uint32_t aggregate_prompt_tokens = 0;
    std::uint32_t max_item_tokens         = 0;
    std::size_t general_capacity_bytes    = 0;
    std::size_t encode_peak_bytes         = 0;
    std::size_t handoff_offset_bytes      = 0;
    std::size_t handoff_capacity_bytes    = 0;
    std::size_t handoff_active_bytes      = 0;
    std::size_t handoff_peak_bytes        = 0;
};

// Most tokens Engine::score_tokens returns per scored position for each distribution output.
inline constexpr std::uint32_t kMaximumScoreTopTokens = 256;

// Distribution outputs of Engine::score_tokens beyond every target token's log-probability. All
// log-probabilities are over the model's public tokens.
struct ScoreOptions {
    // The top_k most probable tokens of every scored position, in [0,kMaximumScoreTopTokens].
    std::uint32_t top_k = 0;
    // The log-probabilities of candidates_per_position given tokens at every scored position, in
    // [0,kMaximumScoreTopTokens]: candidates holds them position-major (scored positions x
    // candidates_per_position).
    std::uint32_t candidates_per_position = 0;
    std::vector<TokenId> candidates;
    // Scored positions (0-based within the scored range, strictly ascending, at most
    // kMaximumScoreLogitPositions) whose whole logits vector ScoreResult::logits returns, for exact
    // comparisons (IQS-2 reference conformance).
    std::vector<std::uint32_t> logits_positions;
};

inline constexpr std::uint32_t kMaximumScoreLogitPositions = 256;

struct ScoreResult {
    // log p(tokens[i] | tokens[0..i)) for every scored position i.
    std::vector<float> logprobs;
    // Position-major (scored positions x top_k): most probable first, equal logits by ascending
    // token id.
    std::vector<TokenId> top_ids;
    std::vector<float> top_logprobs;
    // Position-major (scored positions x candidates_per_position), in the candidates' order.
    std::vector<float> candidate_logprobs;
    // One vocabulary-sized row per ScoreOptions::logits_positions entry, in that order: the FP32 logits
    // as scoring computed them.
    std::vector<float> logits;
};

// Offloaded routed experts (Qwen3.5-122B-A10B): the VRAM expert cache's frames now and at startup,
// the staging slots misses are copied through, and the host RAM holding expert records (with the SSD
// tier the rest are read from the artifact on use).
struct ExpertCacheMemorySummary {
    std::uint32_t frames         = 0;
    std::uint32_t startup_frames = 0;
    std::uint64_t routed_experts = 0; // layers x experts
    std::uint64_t frame_bytes    = 0; // one frame
    std::uint64_t staging_bytes  = 0;
    bool ssd_tier                = false;
    std::uint64_t host_ram_bytes = 0;
    // An elastic text KV (it grows from the frames): its backed and total pages and the device bytes
    // mapped for them now; 0 pages when the KV is a fixed part of the sequence memory.
    std::uint32_t kv_backed_pages   = 0;
    std::uint32_t kv_capacity_pages = 0;
    std::uint64_t kv_backed_bytes   = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    std::optional<VisionWorkspaceMemorySummary> vision_workspace;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t vram_headroom_bytes               = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    // Device memory CUDA Graph preparation actually took at startup (free memory before minus
    // after instantiating, uploading and launching every executable), in the driver's allocation
    // granularity: 0 without CUDA Graphs, and also when the graphs fit in memory the driver already
    // held (StartupPhase::CudaGraphPrepare reports the preparation itself).
    std::size_t cuda_graph_measured_bytes         = 0;
    std::size_t kv_payload_bytes                  = 0;
    // With offloaded experts (beside runtime_reservation_bytes, which excludes them).
    std::optional<ExpertCacheMemorySummary> expert_cache;
};

// A count and total of durations, and the longest since Engine::reset_execution_maxima().
struct DurationCounter {
    std::uint64_t count    = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t max_ns   = 0;

    void add(std::uint64_t ns) noexcept {
        ++count;
        total_ns += ns;
        if (ns > max_ns) { max_ns = ns; }
    }
};

// The Engine worker's execution units by kind, each from its call to its return (Host work and
// Device waits together): the maximum shows stalls the totals hide, and decode_by_batch[b - 1]
// (decode rounds of b rows) the cost of the batch size.
struct RuntimeExecutionLatency {
    DurationCounter prefill; // one prefill step of a request
    DurationCounter replay;  // one Replay step
    DurationCounter control; // one control batch
    DurationCounter decode;  // one decode round
    std::array<DurationCounter, kMaximumConcurrency> decode_by_batch;
};

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// device_wait_ns is blocked wall time and is intentionally excluded from their sum. Detail values
// are subsets of a top-level phase and must not be added to Host-active time again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns       = 0;
    std::uint64_t program_submit_ns        = 0;
    std::uint64_t program_post_ns          = 0;
    std::uint64_t engine_commit_output_ns  = 0;
    std::uint64_t engine_maintenance_ns    = 0;
    std::uint64_t device_wait_ns           = 0;
    std::uint64_t constraint_draft_wait_ns = 0;

    std::uint64_t decode_host_ns         = 0;
    std::uint64_t decode_device_wait_ns  = 0;
    std::uint64_t prefill_host_ns        = 0;
    std::uint64_t prefill_device_wait_ns = 0;
    std::uint64_t control_host_ns        = 0;
    std::uint64_t control_device_wait_ns = 0;
    std::uint64_t prefill_units          = 0;
    std::uint64_t control_units          = 0;

    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    RuntimeExecutionLatency execution;
    // Full prompt counted once when initial binding succeeds; includes reused tokens.
    std::uint64_t prompt_tokens = 0;
    // All committed output tokens, including the first token and Engine-injected control tokens.
    std::uint64_t generated_tokens = 0;
    // Native verification work, using the same acceptance semantics as SpeculativeStats.
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    // Initial prompt tokens evaluated by prefill. Reused prefix-cache tokens and replay
    // recomputation are excluded; replayed_tokens separately counts that additional model work.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t paused_requests           = 0;
    std::uint32_t replaying_requests        = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t preemptions               = 0;
    std::uint64_t snapshot_restores         = 0;
    std::uint64_t replay_restores           = 0;
    std::uint64_t replayed_tokens           = 0;

    // Initial bindings from the root and from a prefix-cache snapshot.
    std::uint64_t root_selections               = 0;
    std::uint64_t prefix_selections             = 0;
    std::uint64_t reused_prompt_tokens          = 0;
    std::uint32_t last_selected_frontier_tokens = 0;

    // StateImage copies from a snapshot into a lane (state_forks; at initial binding too:
    // materialization_state_forks), Host snapshot restores and private partial-tail page copies.
    std::uint64_t state_forks                 = 0;
    std::uint64_t materialization_state_forks = 0;
    std::uint64_t state_restores              = 0;
    std::uint64_t partial_tail_cow_pages      = 0;

    std::uint32_t device_state_occupied_slots      = 0;
    std::uint32_t device_main_kv_occupied_pages    = 0;
    std::uint32_t device_backend_kv_occupied_pages = 0;

    // The VRAM expert cache of an offloaded-expert model (Qwen3.8-Flash-Next); absent otherwise.
    std::optional<ExpertCacheTotals> expert_cache;

    // The prefix cache; zero when it is disabled. Block and snapshot gauges are absolute; the
    // rest are cumulative event counters.
    std::uint32_t hybrid_cached_blocks           = 0; // Device-resident tree blocks
    std::uint32_t hybrid_evictable_blocks        = 0;
    std::uint32_t hybrid_tree_blocks             = 0; // Device or Host
    std::uint32_t hybrid_snapshots               = 0;
    std::uint64_t hybrid_host_capacity_bytes     = 0;
    std::uint64_t hybrid_host_used_bytes         = 0;
    std::uint64_t hybrid_snapshot_hits           = 0;
    std::uint64_t hybrid_reused_tokens           = 0;
    std::uint64_t hybrid_blocks_inserted         = 0;
    std::uint64_t hybrid_blocks_reattached       = 0;
    std::uint64_t hybrid_blocks_duplicate        = 0;
    std::uint64_t hybrid_taps_created            = 0;
    std::uint64_t hybrid_taps_skipped            = 0;
    std::uint64_t hybrid_endpoints_created       = 0;
    // Publication refusals: taps skipped for want of a Device snapshot slot (of taps_skipped),
    // endpoints that passed their gates but were not published, snapshot images and blocks
    // refused Host slabs (they stay Device-only), Device snapshot slots taken from an owner, slot
    // requests refused, and the slots free now.
    std::uint64_t hybrid_taps_skipped_no_slot      = 0;
    std::uint64_t hybrid_endpoints_skipped         = 0;
    std::uint64_t hybrid_host_image_write_refusals = 0;
    std::uint64_t hybrid_host_block_write_refusals = 0;
    std::uint64_t hybrid_device_slot_evictions     = 0;
    std::uint64_t hybrid_device_slot_refusals      = 0;
    std::uint32_t hybrid_free_device_snapshot_slots = 0;
    std::uint64_t hybrid_host_image_writes       = 0;
    std::uint64_t hybrid_host_block_writes       = 0;
    std::uint64_t hybrid_host_image_restores     = 0;
    std::uint64_t hybrid_host_block_restores     = 0;
    // Of those, blocks copied for a waiting FIFO head before its admission.
    std::uint64_t hybrid_prefetched_blocks       = 0;
    std::uint64_t hybrid_host_write_bytes        = 0;
    std::uint64_t hybrid_host_restore_bytes      = 0;
    // The copy calls behind those bytes (Qwen3.5's Host tier; zero for other caches).
    std::uint64_t hybrid_host_write_copies       = 0;
    std::uint64_t hybrid_host_restore_copies     = 0;
    std::uint64_t hybrid_evicted_blocks          = 0;
    std::uint64_t hybrid_host_snapshot_evictions = 0;
    std::uint64_t hybrid_host_dead_reclaims      = 0;
    std::uint64_t hybrid_unbacked_node_losses    = 0;
    // Queue holds (HybridPrefixCacheOptions::queue_holds): the snapshots held for queued
    // requests now, held entries whose Device copy was evicted, held snapshots lost, and Host
    // writes refused because only held snapshots were left to evict.
    std::uint32_t hybrid_held_snapshots          = 0;
    std::uint64_t hybrid_held_device_evictions   = 0;
    std::uint64_t hybrid_held_snapshot_losses    = 0;
    std::uint64_t hybrid_held_host_refusals      = 0;
    // Holds placed after a reply ending in a tool call (HybridPrefixCacheOptions::tool_call_hold).
    std::uint64_t hybrid_tool_call_holds         = 0;
    // Idle-time prefill of finished turns (HybridPrefixCacheOptions::next_turn_prefill_quiet): passes
    // started, run to their end, yielded to a request at a unit boundary, skipped because the cache
    // already held all but a few of their tokens, and turns dropped before a pass started (a
    // request or a newer turn came first, or preparation failed); and the prompt tokens the passes
    // computed (also in computed_prefill_tokens).
    std::uint64_t next_turn_prefills_started          = 0;
    std::uint64_t next_turn_prefills_completed        = 0;
    std::uint64_t next_turn_prefills_yielded          = 0;
    std::uint64_t next_turn_prefills_skipped          = 0;
    std::uint64_t next_turn_prefills_dropped          = 0;
    std::uint64_t next_turn_prefill_tokens            = 0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

// Static facts about the loaded model, independent of the current request context and memory
// layout. The Model supplies the architecture dimension facts (vocab_size, embedding_size,
// native_context) and its registered identity; the Engine adds the artifact-measured facts
// (parameters, weight_bytes). Serving renders these into the OpenAI /v1/models model object and
// its llama.cpp-compatible `meta` field. The effective per-request context ceiling is a property
// of the running instance, not of the model, and is reported separately (meta n_ctx).
struct ModelMetadata {
    std::string model_id;    // Registered model identity (serving public alias default).
    std::string weights_id;  // Dominant loaded weights format name (meta ftype).
    std::uint64_t vocab_size     = 0; // Tokenizer token domain (meta n_vocab).
    std::uint64_t embedding_size = 0; // Model embedding width (meta n_embd).
    std::uint64_t native_context = 0; // Model native/training context (meta n_ctx_train).
    std::uint64_t parameters     = 0; // Total logical weight elements (meta n_params).
    std::uint64_t weight_bytes   = 0; // Encoded weight payload bytes (meta size).
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t device_object_count    = 0;
    std::size_t host_object_count      = 0;
    ContextCostSummary context_cost;
    // The configuration fast paths are qualified by (for example "24 heads, 4-bit activations,
    // int8 KV"), the fast paths this Engine runs (fast_path_name; empty when none is qualified or
    // they are disabled), and whether EngineOptions::fast_paths was off.
    std::string fast_path_configuration;
    std::vector<std::string> fast_paths;
    bool fast_paths_disabled = false;
    // The temporary A/B switches (INFERNIX_TMP_*) this process read, as "NAME=value | ...".
    std::string temporary_switches;
    // Whether a request's output depends only on the request, the artifact, the options and the
    // build: no request-coupling fast path (FastPath::BatchedDecode and after) runs, and no
    // Qwen3.5 DFlash or DFlash2 drafter (whose batched draft calls are not yet column-invariant).
    bool request_invariant          = false;
    // Text projections whose format and shape register no dedicated column-invariant route, so the
    // request-invariant decode and verification run each once per column (much slower, same output
    // guarantee), as "role format [rows,cols] xN" (Qwen3.5 family; empty when every one has one).
    // A nonempty list marks an unsupported artifact; startup logs it as a warning.
    std::vector<std::string> column_by_column_projections;
    // Effective prefill chunk (tokens), and the width a prefill step takes while no request waits
    // to decode when idle-wide-prefill runs: the widest that fit the memory plan, or the chunk when
    // none wider fit. Zero when idle-wide-prefill is not selected (Qwen3.5 family only).
    std::uint32_t prefill_chunk      = 0;
    std::uint32_t idle_prefill_width = 0;

    // Hybrid prefix cache restored from its persistent file at startup.
    struct PrefixCacheRestore {
        bool attempted = false;
        bool restored  = false;
        // Why nothing was restored (no file yet, incompatible file, I/O error).
        std::string message;
        std::uint64_t blocks    = 0;
        std::uint64_t snapshots = 0;
        std::uint64_t bytes     = 0;
        double seconds          = 0.0;
        // What the file holds and the Host tier bytes all of it takes, against this Engine's
        // tier. When the tier is smaller, only the snapshots it values most were restored.
        std::uint64_t saved_blocks        = 0;
        std::uint64_t saved_snapshots     = 0;
        std::uint64_t required_host_bytes = 0;
        std::uint64_t host_bytes          = 0;
    } prefix_cache;
};

} // namespace infernix
