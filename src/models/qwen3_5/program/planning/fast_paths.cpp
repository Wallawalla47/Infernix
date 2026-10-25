// Fast-path selection (EngineOptions::fast_paths): which numerics-changing routes run for a loaded
// model, from the registry of configurations each passed the acceptance gate in
// (docs/maintainer/model-quality.md section 6.7).
#include "models/qwen3_5/program/planning/fast_paths.h"

#include "infernix/ops/linear.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace infernix::models::qwen3_5 {
namespace {

struct Qualification {
    FastPath path;
    FastPathConfiguration configuration;
};

// Every configuration a fast path passed the acceptance gate in; the evidence is in
// docs/maintainer/model-quality.md section 6.7. A path runs only in a listed configuration.
constexpr std::array kQualified{
    Qualification{FastPath::DecodeSplit64, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // The request-coupling paths, qualified jointly per KV format at concurrency 8, DFlash2 with
    // n-gram copies at concurrency 4, and off-grid and endpoint cache resumes (2026-10-14).
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::BatchSplitAttention, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchSplitAttention, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchSplitAttention,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchSplitAttention, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveSpeculation,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // Two-term A4 residual MLP activations in wide rounds, jointly with the coupling paths above
    // (DFlash2 with n-gram copies at concurrency 4, 2026-10-14).
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // Two-term A8 residual activations in the FP8 input projections' wide rounds: gated on INT8
    // and NVFP4 KV (the least headroom), K8V4 and K4V2 by their bounds plus INT8's increment
    // (2026-10-15).
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // Wider prefill steps while no lane waits to decode: the idle width (3,584) against 896- and
    // 1,792-token chunks, prefill gate, quick corpus (2026-10-15; BF16, FP8 and VQ2 2026-10-16).
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // Preemption with Replay under a two-lane pool at concurrency 8 (2026-10-14).
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // VQ2 KV: every request-coupling path and both residual profiles, jointly in the plain, spec,
    // cache and preemption arms (2026-10-15).
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::BatchSplitAttention, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::BatchedA4ResidualDecode, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::BatchedA8ResidualDecode, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    // BF16 and FP8 KV: the same paths, jointly in the spec, cache and preemption arms at 512
    // segments (the preemption arm also covers eight requests per round; 2026-10-15).
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::BatchSplitAttention, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::BatchedDecode, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::BatchSplitAttention,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveSpeculation,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::BatchedA4ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::BatchedA8ResidualDecode,
                  {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    // FP16-accumulate chunked GDN, jointly with the paths above: prefill gate (quick corpus, 896
    // and 1,792) on every KV format, decode (plain, 256 segments) on INT8 and BF16 (2026-10-17).
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // The groupwise Qwen3.8-27B (24 heads, 16-bit activations, groupwise Q4 gate/up) by composition
    // (model-quality.md 6.1, 2026-10-18): prompt-pv8 changes only the prompt attention kernels it
    // shares, at the same 24-head shapes and with no added cost, with the NVFP4 27B, whose gates it
    // passed with a clear prefill gain (+1.6-5.7 %), so it takes that verdict and timing.
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A16, KvCacheStorage::Nvfp4Group16, true}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value, true}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A16, KvCacheStorage::Vq2, true}},
    Qualification{FastPath::PromptPv8, {24, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value, true}},
    // The groupwise 27B, its own evidence (2026-10-18). idle-wide-prefill with prompt-pv8 and gdn-f16-accumulate:
    // prefill gate (quick corpus, INT8, 896/1,792 against 3,584) passes; pp16384 +3.31 % and +0.12 %,
    // every run ahead; the other KV formats take the NVFP4 27B's per-format verdicts (6.1). decode-split64:
    // the NVFP4 27B's kernel and shapes, +0.25 % at 512-token prompts here, level at 4,096.
    Qualification{FastPath::DecodeSplit64, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A16, KvCacheStorage::BFloat16, true}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256, true}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A16, KvCacheStorage::Nvfp4Group16, true}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value, true}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A16, KvCacheStorage::Vq2, true}},
    Qualification{FastPath::IdleWidePrefill, {24, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value, true}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A16, KvCacheStorage::BFloat16, true}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256, true}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A16, KvCacheStorage::Nvfp4Group16, true}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value, true}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A16, KvCacheStorage::Vq2, true}},
    Qualification{FastPath::GdnF16Accumulate, {24, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value, true}},
    // Qwen3.6-35B-A3B (16 heads, 16-bit activations), its own evidence
    // (2026-10-18): the same prefill gate passes; idle-wide-prefill pp16384 +16.9 % (other KV formats
    // from the NVFP4 27B's verdicts), prompt-pv8 +0.24 % on INT8 KV (its 16-head shapes are its own,
    // so each other format is timed before it is listed). gdn-f16-accumulate's +0.05 % was within
    // run-to-run spread, so it is not listed.
    Qualification{FastPath::IdleWidePrefill, {16, ActivationClass::A16, KvCacheStorage::BFloat16}},
    Qualification{FastPath::IdleWidePrefill, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::IdleWidePrefill, {16, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::IdleWidePrefill, {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::IdleWidePrefill, {16, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::IdleWidePrefill, {16, ActivationClass::A16, KvCacheStorage::Vq2}},
    Qualification{FastPath::IdleWidePrefill, {16, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::PromptPv8, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}},
    // The 35B's prompt-pv8 on the other formats it serves, each timed (pp16384, 2026-10-18): NVFP4
    // +1.51 %, K8V4 +1.59 %, VQ2 +0.15 %, K4V2 +0.27 %, every run ahead.
    Qualification{FastPath::PromptPv8, {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::PromptPv8, {16, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::PromptPv8, {16, ActivationClass::A16, KvCacheStorage::Vq2}},
    Qualification{FastPath::PromptPv8, {16, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value}},
    // The groupwise 27B's and the 35B-A3B's request-coupling paths, once their reproducible routes became
    // request-invariant (2026-10-22): each timed faster alone (ABBA, serve: free-cache-resume second-turn
    // TTFT -95.6 % and -58.8 %, preemption +203 % and +109 % aggregate, adaptive-speculation +1.18 % and
    // +7.96 % with MTP), then gated jointly per KV format (concurrency 4, prefix reuse, a 16,640-token pool,
    // DFlash2 with n-gram copies and MTP, 256 segments against the lossless decode references).
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A16, KvCacheStorage::BFloat16, true}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256, true}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A16, KvCacheStorage::Nvfp4Group16, true}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value, true}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A16, KvCacheStorage::Vq2, true}},
    Qualification{FastPath::AdaptiveSpeculation, {24, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value, true}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A16, KvCacheStorage::BFloat16, true}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256, true}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A16, KvCacheStorage::Nvfp4Group16, true}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value, true}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A16, KvCacheStorage::Vq2, true}},
    Qualification{FastPath::FreeCacheResume, {24, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value, true}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A16, KvCacheStorage::BFloat16, true}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256, true}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A16, KvCacheStorage::Nvfp4Group16, true}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value, true}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A16, KvCacheStorage::Vq2, true}},
    Qualification{FastPath::Preemption, {24, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value, true}},
    // The groupwise 27B's one-column rounds on batched-decode's routes (2026-10-22): one request decodes
    // +14.5 % (end to end +13.9 %), four +0.05 % (ABBA, INT8, pp512 + tg512), gated per KV format on plain
    // one-request decode (every round one column), 256 segments against the lossless decode reference.
    Qualification{FastPath::SingleColumnDecode, {24, ActivationClass::A16, KvCacheStorage::BFloat16, true}},
    Qualification{FastPath::SingleColumnDecode, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::SingleColumnDecode, {24, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256, true}},
    Qualification{FastPath::SingleColumnDecode, {24, ActivationClass::A16, KvCacheStorage::Nvfp4Group16, true}},
    Qualification{FastPath::SingleColumnDecode, {24, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value, true}},
    Qualification{FastPath::SingleColumnDecode, {24, ActivationClass::A16, KvCacheStorage::Vq2, true}},
    Qualification{FastPath::SingleColumnDecode, {24, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value, true}},
    Qualification{FastPath::AdaptiveSpeculation, {16, ActivationClass::A16, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveSpeculation, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveSpeculation, {16, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveSpeculation, {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveSpeculation, {16, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveSpeculation, {16, ActivationClass::A16, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveSpeculation, {16, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::FreeCacheResume, {16, ActivationClass::A16, KvCacheStorage::BFloat16}},
    Qualification{FastPath::FreeCacheResume, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::FreeCacheResume, {16, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::FreeCacheResume, {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::FreeCacheResume, {16, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::FreeCacheResume, {16, ActivationClass::A16, KvCacheStorage::Vq2}},
    Qualification{FastPath::FreeCacheResume, {16, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::Preemption, {16, ActivationClass::A16, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Preemption, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::Preemption, {16, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Preemption, {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Preemption, {16, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Preemption, {16, ActivationClass::A16, KvCacheStorage::Vq2}},
    Qualification{FastPath::Preemption, {16, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value}},
    // --draft-tokens auto's policy with the learned round cost (2026-10-18, INT8 KV). It changes no
    // kernel: each round's verify width is one a fixed --draft-tokens K already runs, through routes
    // qualified at every such width (6.1, item 3), so it is decided by speed alone. Against the best
    // single fixed length overall (NVFP4 27B: 5; 35B-A3B: 3), geometric mean over the default route at
    // concurrency 1, 2 and 4, sampled CLI text and (35B) the exact route: +2.1 % and +2.0 %; the
    // exceptions are in model-quality.md 6.7. The KV format changes only the round cost the policy
    // learns, so every format takes the INT8 timing.
    Qualification{FastPath::AdaptiveMtp, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveMtp, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveMtp, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveMtp, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveMtp, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveMtp, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveMtp, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::AdaptiveMtp, {16, ActivationClass::A16, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveMtp, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveMtp, {16, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveMtp, {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveMtp, {16, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveMtp, {16, ActivationClass::A16, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveMtp, {16, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value}},
    // --draft-tokens auto's policy for the block drafters (2026-10-22, INT8 KV). Like adaptive-mtp it
    // changes no kernel: each round verifies leading drafts of a block a fixed --draft-tokens K
    // proposes, at widths a fixed K verifies (6.1, item 3). Against the configuration's best single
    // fixed length (NVFP4 27B DFlash2 7, 35B-A3B DFlash 3) at concurrency 1, 2 and 4 on code and
    // sampled docs, geometric means +1.6 % and +6.1 % (model-quality.md 6.7, with the exceptions).
    // The KV format changes only the round's absolute time, so every format takes the INT8 timing.
    Qualification{FastPath::AdaptiveDFlash, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveDFlash, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveDFlash, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveDFlash, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveDFlash, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveDFlash, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveDFlash, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::AdaptiveDFlash, {16, ActivationClass::A16, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveDFlash, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveDFlash, {16, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveDFlash, {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveDFlash, {16, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveDFlash, {16, ActivationClass::A16, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveDFlash, {16, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value}},
    // batched-a8-decode (wide rounds' FP8 projections at the artifact's single-term 8-bit activations,
    // ahead of the two-term residual profile) by the owner's decision (2026-10-24), although it failed its
    // decode gate in the speculative arm on INT8 (model-quality.md 6.7): every KV format.
    Qualification{FastPath::BatchedA8Decode, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::BatchedA8Decode, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::BatchedA8Decode, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::BatchedA8Decode, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::BatchedA8Decode, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::BatchedA8Decode, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::BatchedA8Decode, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    // On by the owner's decision of 2026-10-25 (a new path measured faster goes on by default before its
    // quality gate; the owner may turn it off once the gating regime is settled), in the configuration
    // measured, each against the default route (ABBA, RTX 5090, 2026-10-25, MTP 3, pp1280+tg256):
    // groupwise 27B, INT8 KV: q5-ksplit-mma decode +4.5 % (3 of 3 blocks, +4.48 to +4.54 %), fp16-gdn-state
    // decode +0.7 % (3 of 3, at least +0.70 %); pp4096 neutral for both. NVFP4 27B, INT8 KV: fp16-gdn-state
    // decode +0.7 % (3 of 3, at least +0.67 %), pp8192 neutral. Not the 35B-A3B (fp16-gdn-state one block
    // -3 %, q5-ksplit-mma neutral).
    Qualification{FastPath::Q5KSplitMma, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::Fp16GdnState, {24, ActivationClass::A16, KvCacheStorage::Int8Group64, true}},
    Qualification{FastPath::Fp16GdnState, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    // gdn-tree-closed-form by the owner's rule of 2026-10-25 (a path measured faster goes on before its gate), where
    // measured (ABBA, RTX 5090, INT8 KV, greedy pp1280+tg256, fixed trees of 16, 6 blocks): NVFP4 27B DFlash2
    // decode +1.19 % (6 of 6, at least +1.01 %; rounds 1.0-1.3 % shorter), 35B-A3B DFlash rounds 1.0-1.7 % shorter in
    // 6 of 6 (decode +1.6 % where the greedy text matched; elsewhere the changed text moved the round count). Only
    // the GDN record changes, so every KV format.
    Qualification{FastPath::GdnTreeClosedForm, {24, ActivationClass::A4, KvCacheStorage::BFloat16}},
    Qualification{FastPath::GdnTreeClosedForm, {24, ActivationClass::A4, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GdnTreeClosedForm, {24, ActivationClass::A4, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::GdnTreeClosedForm, {24, ActivationClass::A4, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::GdnTreeClosedForm, {24, ActivationClass::A4, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::GdnTreeClosedForm, {24, ActivationClass::A4, KvCacheStorage::Vq2}},
    Qualification{FastPath::GdnTreeClosedForm, {24, ActivationClass::A4, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::GdnTreeClosedForm, {16, ActivationClass::A16, KvCacheStorage::BFloat16}},
    Qualification{FastPath::GdnTreeClosedForm, {16, ActivationClass::A16, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GdnTreeClosedForm, {16, ActivationClass::A16, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::GdnTreeClosedForm, {16, ActivationClass::A16, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::GdnTreeClosedForm, {16, ActivationClass::A16, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::GdnTreeClosedForm, {16, ActivationClass::A16, KvCacheStorage::Vq2}},
    Qualification{FastPath::GdnTreeClosedForm, {16, ActivationClass::A16, KvCacheStorage::Q4KeyVq2Value}},
};

const char* kv_name(KvCacheStorage kv) {
    switch (kv) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    case KvCacheStorage::Vq2: return "vq2";
    case KvCacheStorage::Q4KeyVq2Value: return "k4v2";
    }
    return "unknown";
}

const char* activation_name(ActivationClass activations) {
    switch (activations) {
    case ActivationClass::A4: return "4-bit";
    case ActivationClass::A8: return "8-bit";
    case ActivationClass::A16: return "16-bit";
    }
    return "unknown";
}

} // namespace

std::string describe(const FastPathConfiguration& configuration) {
    return std::to_string(configuration.query_heads) + " heads, " +
           activation_name(configuration.activations) + " activations, " +
           kv_name(configuration.kv) + " KV" + (configuration.q4_gate_up ? ", Q4 gate/up" : "") +
           (configuration.offloaded_experts ? ", offloaded experts" : "");
}

FastPathConfiguration fast_path_configuration(const Model& model, const EngineOptions& options) {
    ActivationClass activations = ActivationClass::A16;
    // Text-decoder weights only: drafter and vision weights change proposals and encodings, not
    // the precision of the decoder's own arithmetic.
    for (const BoundWeight& weight : model.weight_data()) {
        if (!std::string_view(weight.name).starts_with("text/")) { continue; }
        for (const WeightUse& use : weight.uses) {
            if (ops::allows_a4(use.policy)) {
                activations = ActivationClass::A4;
            } else if (ops::allows_a8(use.policy) && activations == ActivationClass::A16) {
                activations = ActivationClass::A8;
            }
        }
    }
    // A text layer whose dense MLP gate and up are both RowSplit Q4_G64_FP16 (the groupwise recipes).
    const auto groupwise_q4 = [&](WeightId id) {
        const WeightView& view = model.weight(id).view;
        return !view.parts.empty() &&
               std::all_of(view.parts.begin(), view.parts.end(), [](const WeightRegion& part) {
                   return part.parent != nullptr &&
                          part.parent->geometry.format == QType::Q4_G64_FP16 &&
                          part.parent->geometry.layout == QuantLayout::RowSplit;
               });
    };
    bool q4_gate_up = false;
    for (const BlockWeights& layer : model.weights().text.layers) {
        if (const auto* dense = std::get_if<DenseWeights>(&layer.ffn)) {
            q4_gate_up = q4_gate_up || (groupwise_q4(dense->gate) && groupwise_q4(dense->up));
        }
    }
    const auto& attention = model.config().text.attention;
    return {.query_heads       = attention ? static_cast<std::uint32_t>(attention->num_attention_heads) : 0U,
            .activations       = activations,
            .kv                = options.kv_cache,
            .q4_gate_up        = q4_gate_up,
            .offloaded_experts = model.offloaded_experts()};
}

bool fast_path_applies(FastPath path, const FastPathConfiguration& configuration,
                       const EngineOptions& options) {
    switch (path) {
    case FastPath::BatchedDecode:
    case FastPath::SingleColumnDecode:
    case FastPath::BatchSplitAttention:
    case FastPath::Preemption: return true;
    case FastPath::BatchedA8Decode:
    case FastPath::BatchedA8ResidualDecode:
    case FastPath::BatchedA8SubsetDecode: return configuration.activations != ActivationClass::A16;
    case FastPath::BatchedA4Decode:
    case FastPath::BatchedA4ResidualDecode: return configuration.activations == ActivationClass::A4;
    case FastPath::AdaptiveSpeculation: return options.speculative.backend != SpeculativeBackend::None;
    case FastPath::FreeCacheResume: return options.context_cache.enabled;
    case FastPath::PromptPv8:
        switch (configuration.kv) {
        case KvCacheStorage::Int8Group64: return !options.original_int8_prefill_kernel;
        case KvCacheStorage::Nvfp4Group16: return !options.original_nvfp4_prefill_kernel;
        case KvCacheStorage::Fp8KeyNvfp4Value:
        case KvCacheStorage::Vq2:
        case KvCacheStorage::Q4KeyVq2Value: return true;
        case KvCacheStorage::BFloat16:
        case KvCacheStorage::Fp8E4M3Row256: return false;
        }
        return false;
    case FastPath::DecodeSplit64:
        return configuration.kv == KvCacheStorage::Int8Group64 && configuration.query_heads == 24;
    case FastPath::Q8SlicedK:
    case FastPath::GdnConvFused:
    case FastPath::QsaPrepareFused: return false; // Qwen3.8-Flash-Next kernels only
    // It is --draft-tokens auto's policy, and runs only where speculation is adaptive
    // (adaptive-speculation selected), see ExecutionSchedule::adaptive_mtp. A fixed --draft-tokens N
    // drafts N.
    case FastPath::AdaptiveMtp:
        return options.speculative.backend == SpeculativeBackend::Mtp && options.speculative.draft_tokens_auto;
    // The same for the block drafters: --draft-tokens auto's DFlash and DFlash2 policy, acting only
    // where speculation is adaptive (ExecutionSchedule::adaptive_dflash).
    case FastPath::AdaptiveDFlash:
        return (options.speculative.backend == SpeculativeBackend::DFlash ||
                options.speculative.backend == SpeculativeBackend::DFlash2) &&
               options.speculative.draft_tokens_auto;
    // Causal scoring has no idle steps (each window's chunks are the caller's); the path is a no-op
    // there, which lets the gate name it while its fast run prefills at --fast-chunk.
    case FastPath::IdleWidePrefill: return true;
    // The FP8 sites of the Qwen3.8-27B recipes already permit single-term A8 at prefill widths.
    case FastPath::Fp8A8Pair:
    case FastPath::Fp8A8PairBlock32: return false;
    // The W4A8 gate/up kernel runs only on groupwise Q4 weights (the text MLP's linear_swiglu).
    case FastPath::Q4W4A8: return configuration.q4_gate_up;
    // The offloaded experts' wide calls (Qwen3.5-122B-A10B: every offloaded bank is NVFP4 W4A16, its
    // shared expert one more job of the call). Not qualified for any Qwen3.5 configuration: it runs
    // only as a --fast-path candidate.
    case FastPath::ExpertA4Residual:
    case FastPath::ExpertA4Residual3: return configuration.offloaded_experts;
    // The narrow form is Flash-Next's geometry only (its CPU engine and narrow A4 kernels).
    case FastPath::ExpertA4ResidualDecode: return false;
    // Qwen3.8-Flash-Next's two-row DFlash2 rounds; Qwen3.5's several-row DFlash2 rounds are
    // adaptive-speculation's.
    case FastPath::DFlash2Pairs: return false;
    // MTP rounds with resident experts and no n-gram copies (its sparse proposal rides the MTP
    // drafts' window; --ngram-draft-tokens auto keeps copies off beside MTP); qualified nowhere on
    // Qwen3.5 yet, so it runs only as a candidate.
    case FastPath::SampledMtpDrafts:
        return options.speculative.backend == SpeculativeBackend::Mtp && !configuration.offloaded_experts &&
               options.speculative.ngram_draft_tokens == 0;
    // Every Text GDN layer normalizes Q/K, and its chunked calls serve prefill.
    case FastPath::GdnF16Accumulate: return true;
    // The groupwise recipes' Q5 output and down projections; their artifacts compute with 16-bit
    // activations (on an A16 artifact without Q5 linear_add weights the path changes nothing).
    case FastPath::Q5KSplitMma: return configuration.activations == ActivationClass::A16;
    case FastPath::GgmlBf16Prefill: return false; // Qwen3.8-Flash-Next's GGUF imports only
    // Qwen3.8-Flash-Next's FP8 and MXFP8 dense projections only (the Qwen3.8-27B FP8 sites already take
    // single-term A8 at prefill widths).
    case FastPath::F16AccumulatePrefill: return false;
    // BF16 KV prompt attention; qualified nowhere yet, so it runs only as a candidate.
    case FastPath::Bf16PromptSplit: return configuration.kv == KvCacheStorage::BFloat16;
    // The offloaded MoE layers' BF16 router (the Qwen3.5-122B-A10B: 256 experts plus the shared expert's
    // score at H 3,072); the resident MoE models route their prefill through SparseMoe's own MMA router.
    case FastPath::RouterMma: return configuration.offloaded_experts;
    case FastPath::ScreenedLmHead: return false; // Qwen3.8-Flash-Next's screened head only
    // Every Qwen3.5 artifact has GDN layers; qualified on the 27Bs with INT8 KV (kQualified).
    case FastPath::Fp16GdnState: return true;
    // Chains whose drafts carry their proposal laws: DFlash2's, and MTP's under sampled-mtp-drafts
    // (without it MTP drafts argmax chains, which the block rule cannot improve and the greedy-draft
    // verifier serves). Qualified nowhere yet, so it runs only as a candidate.
    case FastPath::BlockVerification:
        return options.speculative.backend == SpeculativeBackend::DFlash2 ||
               options.speculative.backend == SpeculativeBackend::Mtp;
    // Fixed-table tree rounds whose drafter builds the tree (DFlash2, DFlash): automatic widths
    // derive a narrower width's tokens from the wider tree's walk, which UniVer's decision does not
    // share, and MTP's chain is drawn before its siblings. Qualified nowhere yet: a candidate only.
    case FastPath::UniverTrees:
        return (options.speculative.backend == SpeculativeBackend::DFlash2 ||
                options.speculative.backend == SpeculativeBackend::DFlash) &&
               !options.speculative.draft_tree_auto &&
               std::any_of(options.speculative.draft_tree_nodes.begin(),
                           options.speculative.draft_tree_nodes.end(), [](std::uint32_t n) { return n != 0; });
    // Every engine with tree rounds, fixed or automatic: only their GDN records change. Qualified on the
    // NVFP4 27B and the 35B-A3B (kQualified).
    case FastPath::GdnTreeClosedForm:
        return options.speculative.draft_tree_auto ||
               std::any_of(options.speculative.draft_tree_nodes.begin(),
                           options.speculative.draft_tree_nodes.end(), [](std::uint32_t n) { return n != 0; });
    }
    return false;
}

bool fast_path_qualified(FastPath path, const FastPathConfiguration& configuration) {
    return std::any_of(kQualified.begin(), kQualified.end(), [&](const Qualification& entry) {
        return entry.path == path && entry.configuration == configuration;
    });
}

bool auto_drafts_nothing(bool offloaded_experts, SpeculativeBackend backend) noexcept {
    // Qwen3.5-122B-A10B (the Qwen3.5 artifact with offloaded experts): each verify column brings its
    // own expert misses, about 10 ms a column against 13.8 ms a plain token (2026-10-22, one request,
    // bf16 KV, warm cache, 256 tokens of one prompt): DFlash2 1/2/4/7/15 65.8/61.3/57.1/35.2/19.2 tok/s
    // against plain 72.5 with the drafter loaded; without one the expert cache also keeps the drafter's
    // VRAM. MTP's confidence gate verifies only the drafts worth their misses, and its drafts beat plain
    // decode (auto_fixed_draft_tokens). DFlash2's adaptive-dflash (forced; 2026-10-25, Ornith-1.5-122B, INT8
    // KV) was 13.5 % faster than plain decode at one request (geometric mean of eight bench-corpus pairs)
    // but 15.6 % slower at two (sampled docs, decode tok/s; batched rounds plain at one draft 18.3 %), so it
    // decodes without its drafter.
    return offloaded_experts && backend != SpeculativeBackend::Mtp;
}

std::uint32_t auto_fixed_draft_tokens(const FastPathConfiguration& configuration,
                                      SpeculativeBackend backend) noexcept {
    // Offloaded experts (Qwen3.5-122B-A10B, Ornith-1.5-122B): MTP drafts two behind its confidence gate,
    // 2026-10-25 (RTX 5090, INT8 KV, one request, 256 tokens of three bench-corpus prompts, ABBA, greedy
    // and temperature 1.0): against plain decode two gated drafts were a geometric mean 14.0 % faster over
    // eight pairs (greedy +11 % to +26 %, sampled -1.1 % to +25 %), one draft 8.7 %, adaptive-mtp at a
    // ceiling of 2 tied (13.9 %; greedy worst +1.2 %) and of 3 13.3 %; with two requests on sampled docs two
    // gated drafts were about 6 % slower than plain decode (each row's verify columns bring their own
    // misses; skipping the drafter in batched rounds would need a round kind of its own). DFlash2 decodes
    // without its drafter
    // (auto_drafts_nothing); a fixed length is reached only with a forced policy: one draft, its fastest.
    if (configuration.offloaded_experts) { return backend == SpeculativeBackend::Mtp ? 2 : 1; }
    // The block drafters' fastest single fixed lengths overall (code and sampled docs; 2026-10-20 at
    // one request, the NVFP4 27B 2026-10-21/22 also at two and four): the 27B's DFlash2 7 (15 is
    // 12.4 % faster than 7 on code at one request but 1-19 % slower batched, a geometric mean 6.1 %
    // behind), the 35B-A3B's DFlash 3.
    if (backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2) {
        return configuration.query_heads == 24 ? 7 : 3;
    }
    // 2026-10-18 sweeps (INT8 KV; tg512 at concurrency 1, 2 and 4 and sampled text, model-quality.md
    // 6.7): the NVFP4 27B is fastest overall at 5, the groupwise 27B and the 35B-A3B at 3. The KV
    // format changes the round cost only slightly, so every format takes these. The groupwise 27B's
    // adaptive-mtp, re-timed 2026-10-25 on code and docs at one, two and four requests, won on code
    // (+10 % at one request) and lost on docs and at four requests (-1.5 %, -4.8 %): it keeps 3.
    if (configuration.query_heads == 24 && configuration.activations == ActivationClass::A4) { return 5; }
    return 3;
}

std::uint32_t dflash_auto_ceiling(const FastPathConfiguration& configuration, SpeculativeBackend backend) noexcept {
    (void)backend;
    return configuration.query_heads == 24 ? 15U : 7U;
}

detail::DFlashWidthCosts dflash_width_costs(const FastPathConfiguration& configuration,
                                    SpeculativeBackend backend) noexcept {
    (void)backend;
    // Measured 2026-10-20 (RTX 5090, INT8 KV, one request; fixed lengths 3, 7 and 15 against plain
    // decoding, then the policy's cost tried around the fit on code and sampled docs): the NVFP4
    // 27B verifies a draft at about 0.01 of a plain round (0.02 chose best), the 35B-A3B's experts
    // make a draft 0.13-0.15. The groupwise 27B's (nearly free to 7 drafts and 0.05 past them after
    // a block proposal of 0.69) serve a forced --fast-path adaptive-dflash only: it is not
    // registered there.
    if (configuration.query_heads == 24 && configuration.activations == ActivationClass::A4) {
        return {.proposal = 0.18, .per_draft = 0.02, .per_draft_wide = 0.02};
    }
    if (configuration.query_heads == 24) { return {.proposal = 0.69, .per_draft = 0.003, .per_draft_wide = 0.05}; }
    // Offloaded experts (Qwen3.5-122B-A10B; 2026-10-22, one request, bf16 KV): a verified draft
    // brings its own expert misses, ~10 ms against a 13.8 ms plain round (0.72), and DFlash2's block
    // proposal ~0.12 (its one-draft rounds at 25.4 ms). Not registered there: these serve a forced
    // adaptive-dflash (the A/B of design §12.2).
    if (configuration.offloaded_experts) { return {.proposal = 0.12, .per_draft = 0.72, .per_draft_wide = 0.72}; }
    return {.proposal = 0.2, .per_draft = 0.15, .per_draft_wide = 0.15};
}

std::uint32_t dflash_batch_drafts(const FastPathConfiguration& configuration, SpeculativeBackend backend) noexcept {
    (void)backend;
    // Measured 2026-10-21/22 (RTX 5090, INT8 KV, two and four requests, code and sampled docs): the
    // NVFP4 27B's split rounds (a 15-draft proposal, a host wait, then the chosen width) ran 1-8 %
    // slower than fixed 7-draft rounds whatever widths they chose, the proposal and the wait costing
    // 0.5-0.9 ms a batched round, so its batched rounds run plain at 7. The 35B-A3B's chosen batch
    // widths ran -1.2 to +11.2 % against plain rounds of 3 (+3.8 % geometric mean) and 3-36 % ahead
    // of plain rounds of 5, so it keeps them.
    if (configuration.query_heads == 24 && configuration.activations == ActivationClass::A4) { return 7; }
    return 0;
}

NgramSettings auto_ngram_settings(const FastPathConfiguration& configuration, SpeculativeBackend backend,
                                  bool adaptive_drafts) noexcept {
    // Measured 2026-10-22 (RTX 5090, INT8 KV; 24 repository files rewritten with one identifier
    // renamed, 512 greedy tokens each, against code and sampled docs): beside the NVFP4 27B's
    // adaptive DFlash2 rounds, copies of up to 63 drafts lifted one request from 612 to 1,304 tok/s
    // (31 drafts: 1,088), within the spread on code and docs; with two requests 63 and 31 copied
    // alike (1,405 and 1,396 against 776) but 63-draft copy rounds cost the batch's other rows 6.3 %
    // on code (31: 1.6 %), so batched copies stop at 31. Minimum matches 8 and 16 moved the copy
    // workload +4.1 % and -3.5 % with the opposite effect on code's wrong copies, so it stays 12.
    // The 35B-A3B's DFlash copies did not pay (15 drafts +0.5 %, 31 drafts -4.1 %).
    if (configuration.query_heads == 24 && configuration.activations == ActivationClass::A4 &&
        backend == SpeculativeBackend::DFlash2 && adaptive_drafts) {
        return {.window = 63, .batch_window = 31, .min_match = 12};
    }
    return {};
}

bool FastPathSelection::selected(FastPath path) const noexcept {
    return std::find(paths.begin(), paths.end(), path) != paths.end();
}

bool FastPathSelection::request_invariant() const noexcept {
    return std::none_of(paths.begin(), paths.end(), couples_requests);
}

FastPathSelection select_fast_paths(const Model& model, const EngineOptions& options) {
    FastPathSelection out{.configuration = fast_path_configuration(model, options), .paths = {}};
    for (const FastPath path : options.fast_path_candidates) {
        if (!fast_path_applies(path, out.configuration, options)) {
            throw std::invalid_argument(std::string("fast path ") + fast_path_name(path) +
                                        " does not apply to " + describe(out.configuration));
        }
    }
    for (const FastPath path : kFastPaths) {
        const bool candidate = std::find(options.fast_path_candidates.begin(),
                                         options.fast_path_candidates.end(),
                                         path) != options.fast_path_candidates.end();
        const bool qualified = options.fast_paths &&
                               fast_path_applies(path, out.configuration, options) &&
                               fast_path_qualified(path, out.configuration);
        if (candidate || qualified) { out.paths.push_back(path); }
    }
    return out;
}

} // namespace infernix::models::qwen3_5
