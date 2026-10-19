// Qwen3.8-Flash-Next fast-path selection (EngineOptions::fast_paths): which numerics-changing routes
// run for a loaded artifact, from the registry of configurations each passed the acceptance gate in
// (docs/maintainer/model-quality.md section 6.7).
#include "models/qwen4_exp/program/fast_paths.h"

#include "models/qwen4_exp/execution/parameters.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <variant>

namespace infernix::models::qwen4_exp {
namespace {

struct Qualification {
    FastPath path;
    FastPathConfiguration configuration;
};

// Every configuration a fast path passed the acceptance gate in; the evidence is in
// docs/maintainer/model-quality.md section 6.7. A path runs only in a listed configuration.
constexpr std::array kQualified{
    // NVIDIA NVFP4 Dense8, INT8 KV: MTP at concurrency 2, and off-grid and endpoint cache resumes
    // (2026-10-14).
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Q8, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Q8, true, KvCacheStorage::Int8Group64}},
    // Preemption with Replay under a 388-page pool that certainly runs short (2026-10-14).
    Qualification{FastPath::Preemption, {DenseKernels::Q8, true, KvCacheStorage::Int8Group64}},
    // The other formats (2026-10-16/17, 256 segments): MTP at concurrency 2 with every request resuming
    // from a warm 3,000-token entry, against the exact route; preemption by its increment over the
    // qualified paths under the same short pool (model-quality.md 6.1). FP8 and K8V4 failed the
    // joint arm narrowly at 256 segments and passed it at 1,024 (2026-10-17).
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Q8, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Q8, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Preemption, {DenseKernels::Q8, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Q8, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Q8, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Preemption, {DenseKernels::Q8, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Q8, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Q8, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Preemption, {DenseKernels::Q8, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Q8, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Q8, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Preemption, {DenseKernels::Q8, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Q8, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Q8, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::Preemption, {DenseKernels::Q8, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Q8, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Q8, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::Preemption, {DenseKernels::Q8, true, KvCacheStorage::Q4KeyVq2Value}},
    // NInfer's Q8 dense tables, fused GDN convolution and fused QSA prepare, jointly: prefill (quick
    // corpus, four chunk sizes) and decode (plain at concurrency 8, MTP at 2) (2026-10-15).
    Qualification{FastPath::Q8SlicedK, {DenseKernels::Q8, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GdnConvFused, {DenseKernels::Q8, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::QsaPrepareFused, {DenseKernels::Q8, true, KvCacheStorage::Int8Group64}},
    // The same three on the other formats qsa_prepare serves: prefill (quick corpus, 896 and 7,168
    // tokens) on each, decode plain on BF16 (the lowest floor) and NVFP4 (the lossiest KV) (2026-10-15).
    Qualification{FastPath::Q8SlicedK, {DenseKernels::Q8, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Q8SlicedK, {DenseKernels::Q8, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Q8SlicedK, {DenseKernels::Q8, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Q8SlicedK, {DenseKernels::Q8, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::GdnConvFused, {DenseKernels::Q8, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::GdnConvFused, {DenseKernels::Q8, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::GdnConvFused, {DenseKernels::Q8, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::GdnConvFused, {DenseKernels::Q8, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::QsaPrepareFused, {DenseKernels::Q8, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::QsaPrepareFused, {DenseKernels::Q8, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::QsaPrepareFused, {DenseKernels::Q8, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::QsaPrepareFused, {DenseKernels::Q8, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    // VQ2 and K4V2 keep the composed QSA prepare; the dense paths leave attention and the KV cache alone,
    // so they are registered from the INT8 increments (model-quality.md 6.1, step 3).
    Qualification{FastPath::Q8SlicedK, {DenseKernels::Q8, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::Q8SlicedK, {DenseKernels::Q8, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::GdnConvFused, {DenseKernels::Q8, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::GdnConvFused, {DenseKernels::Q8, true, KvCacheStorage::Q4KeyVq2Value}},
    // Recipe C (FP8 dense, W4A16 experts): two-term A4 expert activations. Prefill (quick corpus,
    // 896 and 7,168 tokens) on every format, decode (MTP at concurrency 2, 256 segments) on INT8, the
    // passing format with the lowest floor; the path leaves attention and the KV cache alone, so the
    // others are registered from the INT8 increments (model-quality.md 6.1, step 3). BF16 KV fails:
    // there recipe C's exact route is the reference itself (floor 0) (2026-10-15).
    Qualification{FastPath::ExpertA4Residual, {DenseKernels::Fp8Row, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::ExpertA4Residual, {DenseKernels::Fp8Row, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::ExpertA4Residual, {DenseKernels::Fp8Row, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::ExpertA4Residual, {DenseKernels::Fp8Row, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::ExpertA4Residual, {DenseKernels::Fp8Row, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::ExpertA4Residual, {DenseKernels::Fp8Row, false, KvCacheStorage::Q4KeyVq2Value}},
    // Recipe C: fp8-a8-pair (power-of-two column scales) beside two-term expert-a4-residual,
    // decode gated on each format directly (MTP at concurrency 2, 2026-10-17), since INT8, the
    // lowest floor, fails its tail. Prefill passes on every format but BF16; FP8, NVFP4 and K4V2 by
    // their 1,024-segment decode gates, K8V4 and VQ2 by 256 (model-quality.md 6.7).
    Qualification{FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::Fp8A8Pair, {DenseKernels::Fp8Row, false, KvCacheStorage::Q4KeyVq2Value}},
    // NVIDIA NVFP4 Dense8: sampled MTP drafts (2026-10-17). They change no logit, so the gate is the
    // sampler's distribution (2,000 seeded requests per arm, chi-square per position against plain
    // decoding, INT8 KV) and greedy identity (INT8, NVFP4), which no KV format affects; speed on
    // INT8 (MTP 4, T 1.0 / top-p 0.95 / top-k 20, geomean +2.7 %, 5 of 6 cells).
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Q8, true, KvCacheStorage::Q4KeyVq2Value}},
    // GSQ-RCO IQ3_S (GGUF dense, 16-bit expert activations): the coupling paths by the kernel rule
    // (real tests on the exact route, one INT8 joint decode gate at 256 segments, 2026-10-17; the other
    // formats take Dense8's verdicts), timed on INT8 (2026-10-18): second-turn resume 1.65 -> 0.11 s,
    // two-row MTP 160 -> 200 tok/s. Sampled drafts change no logit (6.1): math +18.8 %, story +22.4 %,
    // code +4.1 % (overlapping). FP16 GDN (qualified on the 27B, every Flash-Next artifact): pp16384
    // +0.21 %, every run ahead; it touches no attention or KV. Not with BF16 KV (2026-10-19): there the
    // IQ3_S exact route is the gate's reference (floor 0) and FP16 GDN alone fails the prefill gate (KL
    // bound 7.5e-3 of 1.5e-3) and the decode gate (TV bound 4.6e-3 of 4e-3), as ggml-bf16-prefill does.
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Ggml, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Ggml, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Ggml, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Ggml, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Ggml, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Ggml, false, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Ggml, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Ggml, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Ggml, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Ggml, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Ggml, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Ggml, false, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Ggml, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Ggml, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Ggml, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Ggml, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Ggml, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Ggml, false, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Ggml, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Ggml, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Ggml, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Ggml, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Ggml, false, KvCacheStorage::Q4KeyVq2Value}},
    // GGUF imports: the BF16 prefill tiles (2026-10-19). Prefill gate (quick corpus, 896 and 7,168
    // tokens) on every format jointly with the qualified paths: KL excess bounds 3.7e-4 to 7.8e-4 on
    // INT8, FP8, NVFP4, K8V4, VQ2 and K4V2; decode plain on INT8 (the passing format with the lowest
    // floor), 256 segments: TV bound 9.3e-4. BF16 KV fails, alone as with gdn-f16-accumulate: there the
    // exact route is the reference (floor 0), and either path alone moves prefill by KL 5.5-5.9e-3 and
    // decode by TV 4.0-4.2e-3 (2026-10-19). Timed on INT8 against the exact route: pp32768 +33 %,
    // pp65536 +44 % (design §19.3.29); no KV format enters its kernels.
    Qualification{FastPath::GgmlBf16Prefill, {DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GgmlBf16Prefill, {DenseKernels::Ggml, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::GgmlBf16Prefill, {DenseKernels::Ggml, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::GgmlBf16Prefill, {DenseKernels::Ggml, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::GgmlBf16Prefill, {DenseKernels::Ggml, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::GgmlBf16Prefill, {DenseKernels::Ggml, false, KvCacheStorage::Q4KeyVq2Value}},
    // Official FP8 (BF16 dense, 16-bit expert activations): the same tests (2026-10-17/18): second-turn
    // resume 3.9-5.7 -> 0.26-0.51 s; sampled drafts math +20.4 %, story +15.6 % (every run ahead), code
    // -13.0 % (overlapping: one exact run at 52.5 against 39.7-40.1 for the other three); FP16 GDN
    // pp16384 +0.33 %, every run ahead (narrowly).
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, false, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, false, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Bf16, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Bf16, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Bf16, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Bf16, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Bf16, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Bf16, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::GdnF16Accumulate, {DenseKernels::Bf16, false, KvCacheStorage::Q4KeyVq2Value}},
    // Preemption on IQ3_S and FP8, timed where it acts (2026-10-18): eight brief requests at once asking
    // for 4,096 tokens each under an 8,192-token pool, IQ3_S 118 -> 131 tok/s (+11.3 %), FP8 26.7 -> 28.3
    // (+5.9 %), every run ahead; with requests that use their whole max_tokens under a short pool it costs
    // about 3 % on IQ3_S (Replay redoes work). FP8 adaptive-speculation: three ABBA blocks, +2.6, +3.8
    // and +7.3 %, every block ahead (FP8 decode is SSD-bound and noisy, so twelve runs).
    Qualification{FastPath::Preemption, {DenseKernels::Ggml, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Preemption, {DenseKernels::Ggml, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::Preemption, {DenseKernels::Ggml, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Preemption, {DenseKernels::Ggml, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Preemption, {DenseKernels::Ggml, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Preemption, {DenseKernels::Ggml, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::Preemption, {DenseKernels::Ggml, false, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, false, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, false, KvCacheStorage::Q4KeyVq2Value}},
    // Recipe A (BF16 dense, NVFP4 W4A4 experts), registered without its own runs at the owner's request
    // (2026-10-18) from the configurations closest to it: free-cache-resume from FP8's timing (same dense
    // kernels), adaptive-speculation from Dense8's and IQ3_S's (4-bit experts streamed as Dense8's; the
    // gain is MTP kept at two rows), preemption from IQ3_S's and FP8's (requests admitted under a short
    // pool), sampled drafts by their distribution test (6.1). Its exact-route equivalences and joint gate
    // were not run on the artifact itself.
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::AdaptiveSpeculation, {DenseKernels::Bf16, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::FreeCacheResume, {DenseKernels::Bf16, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::Preemption, {DenseKernels::Bf16, true, KvCacheStorage::Q4KeyVq2Value}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, true, KvCacheStorage::BFloat16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, true, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, true, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, true, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, true, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, true, KvCacheStorage::Vq2}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Bf16, true, KvCacheStorage::Q4KeyVq2Value}},
    // Recipe C (FP8-row dense): sampled drafts change no logit, so its exact-route request-invariance
    // defect (which keeps the coupling paths off it) does not bear on them; speed from IQ3_S's and FP8's
    // timing (the gain is the drafts' acceptance, not the dense kernels).
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Fp8Row, false, KvCacheStorage::BFloat16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Fp8Row, false, KvCacheStorage::Int8Group64}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Fp8Row, false, KvCacheStorage::Fp8E4M3Row256}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Fp8Row, false, KvCacheStorage::Nvfp4Group16}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Fp8Row, false, KvCacheStorage::Fp8KeyNvfp4Value}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Fp8Row, false, KvCacheStorage::Vq2}},
    Qualification{FastPath::SampledMtpDrafts, {DenseKernels::Fp8Row, false, KvCacheStorage::Q4KeyVq2Value}},
};

const char* dense_name(DenseKernels dense) {
    switch (dense) {
    case DenseKernels::Q8: return "Q8";
    case DenseKernels::Bf16: return "BF16";
    case DenseKernels::Fp8Row: return "FP8";
    case DenseKernels::Nvfp4: return "NVFP4";
    case DenseKernels::Ggml: return "GGUF";
    case DenseKernels::Other: return "other";
    }
    return "other";
}

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

} // namespace

DenseKernels dense_kernels(QType qtype) noexcept {
    switch (qtype) {
    case QType::Q8_G32_FP16: return DenseKernels::Q8;
    case QType::BF16: return DenseKernels::Bf16;
    case QType::FP8_E4M3FN_ROW_BF16: return DenseKernels::Fp8Row;
    case QType::NVFP4: return DenseKernels::Nvfp4;
    case QType::GGML_Q2_0:
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K:
    case QType::GGML_Q6_K:
    case QType::GGML_IQ2_S:
    case QType::GGML_IQ3_XXS:
    case QType::GGML_IQ3_S:
    case QType::GGML_IQ4_XS:
    case QType::GGML_IQ4_NL: return DenseKernels::Ggml;
    default: return DenseKernels::Other;
    }
}

std::string describe(const FastPathConfiguration& configuration) {
    return std::string(dense_name(configuration.dense)) + " dense, " +
           (configuration.a4_experts ? "4-bit" : "16-bit") + " expert activations, " +
           kv_name(configuration.kv) + " KV";
}

FastPathConfiguration fast_path_configuration(const execution::Parameters& parameters,
                                              const EngineOptions& options) {
    FastPathConfiguration out{.dense      = DenseKernels::Bf16,
                              .a4_experts = !parameters.model.config().text.moe.a16_experts,
                              .kv         = options.kv_cache};
    // The dense layers share one kernel family; the GDN q/k/v/z projection is the one the fast
    // routes serve.
    for (const execution::BlockParameters& block : parameters.layers) {
        if (const auto* gdn = std::get_if<execution::GdnParameters>(&block.mixer)) {
            out.dense = dense_kernels(gdn->projection.weight.qtype);
            break;
        }
    }
    return out;
}

bool fast_path_applies(FastPath path, const FastPathConfiguration& configuration,
                       const EngineOptions& options) {
    switch (path) {
    // The request-coupling behaviours: two-row speculative rounds, free cache resume, preemption.
    // Flash-Next's decode is already request-invariant at every batch, so the batched-decode and
    // attention paths have nothing to change.
    case FastPath::AdaptiveSpeculation: return options.speculative.backend == SpeculativeBackend::Mtp;
    case FastPath::FreeCacheResume: return options.context_cache.enabled;
    case FastPath::Preemption: return true;
    case FastPath::BatchedDecode:
    case FastPath::BatchedA8Decode:
    case FastPath::BatchedA4Decode:
    case FastPath::BatchedA4ResidualDecode:
    case FastPath::BatchedA8ResidualDecode:
    case FastPath::BatchSplitAttention: return false;
    case FastPath::PromptPv8:
    case FastPath::DecodeSplit64: return false; // Qwen3.8-27B attention kernels only
    // Flash-Next chooses its draft lengths and prefill widths itself; groupwise Q4 is not a dense format.
    case FastPath::AdaptiveMtp:
    case FastPath::IdleWidePrefill:
    case FastPath::Q4W4A8: return false;
    // The drafter's sampled proposals (the program checks the sparse acceptance's token domain).
    case FastPath::SampledMtpDrafts: return options.speculative.backend == SpeculativeBackend::Mtp;
    // The GDN layers normalize Q/K; their chunked calls serve prefill.
    case FastPath::GdnF16Accumulate: return true;
    // Recipe C's row-FP8 dense projections run 16-bit activations (A16Only).
    case FastPath::Fp8A8Pair: return configuration.dense == DenseKernels::Fp8Row;
    case FastPath::Q8SlicedK:
    case FastPath::GdnConvFused: return configuration.dense == DenseKernels::Q8;
    // The wide route of NVFP4 W4A16 experts (weight-only exports): recipe C, the one configuration
    // with them (FP8-row dense). GGUF and block-FP8 experts also take 16-bit activations but have no
    // such kernel.
    case FastPath::ExpertA4Residual: return !configuration.a4_experts && configuration.dense == DenseKernels::Fp8Row;
    // The GGUF imports' dense projections (ops::ggml_linear's BF16 prefill tiles).
    case FastPath::GgmlBf16Prefill: return configuration.dense == DenseKernels::Ggml;
    case FastPath::QsaPrepareFused:
        // ops::qsa_prepare_supports's KV formats (bf16, int8, fp8, nvfp4, k8v4); VQ2 and K4V2 append
        // inside qsa_attention and keep the composed prepare.
        return configuration.kv == KvCacheStorage::Int8Group64 || configuration.kv == KvCacheStorage::BFloat16 ||
               configuration.kv == KvCacheStorage::Fp8E4M3Row256 || configuration.kv == KvCacheStorage::Nvfp4Group16 ||
               configuration.kv == KvCacheStorage::Fp8KeyNvfp4Value;
    }
    return false;
}

bool fast_path_qualified(FastPath path, const FastPathConfiguration& configuration) {
    return std::any_of(kQualified.begin(), kQualified.end(), [&](const Qualification& entry) {
        return entry.path == path && entry.configuration == configuration;
    });
}

bool FastPathSelection::selected(FastPath path) const noexcept {
    return std::find(paths.begin(), paths.end(), path) != paths.end();
}

bool FastPathSelection::request_invariant() const noexcept {
    return std::none_of(paths.begin(), paths.end(), [](FastPath path) {
        // q8-sliced-k picks its reduction order by the round's column count, so a decode column's bits
        // depend on the other requests of its round.
        // expert-a4-residual routes an expert, and fp8-a8-pair a dense problem, by its column count in
        // the call, which batched rounds share between requests. ggml-bf16-prefill routes a prompt's
        // GGUF projections by its calls' widths, whose boundaries prefix-cache resumes move.
        return path == FastPath::AdaptiveSpeculation || path == FastPath::FreeCacheResume ||
               path == FastPath::Preemption || path == FastPath::Q8SlicedK ||
               path == FastPath::ExpertA4Residual || path == FastPath::Fp8A8Pair ||
               path == FastPath::GgmlBf16Prefill;
    });
}

execution::ForwardFastPaths FastPathSelection::forward() const noexcept {
    ops::LinearRoutes routes = ops::LinearRoutes::Exact;
    if (selected(FastPath::Q8SlicedK)) { routes = routes | ops::LinearRoutes::Q8SlicedK; }
    if (selected(FastPath::Fp8A8Pair)) { routes = routes | ops::LinearRoutes::Fp8A8Pair; }
    if (selected(FastPath::GgmlBf16Prefill)) { routes = routes | ops::LinearRoutes::GgmlBf16Wide; }
    return {.dense_routes       = routes,
            .gdn_conv_fused     = selected(FastPath::GdnConvFused),
            .qsa_prepare_fused  = selected(FastPath::QsaPrepareFused),
            .expert_a4_residual = selected(FastPath::ExpertA4Residual),
            .gdn_arithmetic     = selected(FastPath::GdnF16Accumulate) ? ops::GdnChunkArithmetic::Fp16Accumulate
                                                                       : ops::GdnChunkArithmetic::Exact};
}

FastPathSelection select_fast_paths(const execution::Parameters& parameters, const EngineOptions& options) {
    FastPathSelection out{.configuration = fast_path_configuration(parameters, options), .paths = {}};
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
        const bool qualified = options.fast_paths && fast_path_applies(path, out.configuration, options) &&
                               fast_path_qualified(path, out.configuration);
        if (candidate || qualified) { out.paths.push_back(path); }
    }
    return out;
}

} // namespace infernix::models::qwen4_exp
