#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_request.h"
#include "ops/offloaded_sparse_moe/cpu/miss_request.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace infernix::ops {

/**
 * Routed experts of one MoE layer with experts held outside device memory
 * (docs/maintainer/qwen3_8-flash-next-design.md §8.4-8.5, §16.2). For each column t:
 *
 *   logits     = FP32 [router; shared gate] x[:, t]             -> projection_fp32
 *   p          = softmax(router logits[0:E, t]);  ids = top-k of p (lower id wins exact ties)
 *   w_i        = p[ids_i] / sum_j p[ids_j]                      -> moe_route
 *   s          = sigmoid(logits[E, t])  (the shared-expert gate)  -> moe_route
 *   y_i        = expert ids_i applied to x[:, t] with the W4A4 arithmetic of design §16.2
 *   y[:, t]    = bf16(sum_i w_i * y_i + s * y_shared[:, t])       -> moe_combine
 *
 * Expert placement is an execution resource, not a semantic input: an expert's record is read
 * from a device frame when `frames[e] >= 0` and from the pinned host bank otherwise, and its
 * BF16 output bits are identical either way. An expert with at most eight columns in a call takes
 * the narrow route, whose products are exact integer sums: no batch shape, split or order changes
 * a bit of its output, and the CPU engine (ops/offloaded_sparse_moe/cpu/w4a4_expert.h) returns
 * the same bits. An expert with more columns (and one shared gate/up input scale) takes the wide
 * route: the same A4 activations and BF16 boundaries, with the block products summed by FP32
 * tensor cores (qualified against FP64; design §8.5, §13). Both routes depend only on the column
 * count and the stored scales, never on placement. W4A16 experts (§16.2.1) are exact on every
 * route, the wide one included (integer tensor cores), so their bits depend on neither placement
 * nor column count. The route reads each router logit as its model's forward does
 * (RouterLogits): FP32 for Qwen3.8-Flash-Next, or rounded once to BF16 for Qwen3.5-122B-A10B,
 * whose router projection writes BF16 (Transformers' Qwen3_5MoeSparseMoeBlock; ties at that
 * precision go to the lower id, as in the resident Qwen3.5 sparse MoE). Each logit is one FP32 dot
 * product whose order depends only on H, so a column's routing is identical in every batch shape.
 * Routing weights, the shared gate and the combine are FP32 evaluations rounded once; their oracle
 * is the formula in FP64 over the logits as read.
 */

/// An optional column mask of a call laid out as rows of `width` columns (a verification round's
/// [width, rows] columns): column t is live when t % width < valid[t / width] (device I32 [T /
/// width], read at execution, so a captured call follows it). A masked column (a round's padding past
/// a row's drafts) routes nowhere: it adds no dispatch entry, routed or shared, so no job, miss or
/// prefetch, its ids, weights, gate and route-log ids are still the router's, and the combine writes
/// y = 0 for it. Every live column's bits are those of the unmasked call. A call with a mask may have
/// a shared job (MoeDispatch::shared_first) or not (a dense shared expert computed beside the call).
struct MoeColumnMask {
    const std::int32_t* valid = nullptr; // null: every column is live
    std::int32_t width        = 0;
};

struct MoeRouting {
    Tensor ids;         // I32 [k, T]
    Tensor weights;     // FP32 [k, T]
    Tensor shared_gate; // FP32 [T]
    MoeColumnMask mask; // read by the dispatch and the combine
};

/// How the route reads the FP32 router logits: as stored, or each rounded once to BF16 (round to
/// nearest even) before selection, the routing weights and the shared gate.
enum class RouterLogits : std::uint8_t { kFp32, kBf16 };

/// logits: FP32 [E + 1, T], the router rows followed by the shared-expert gate row.
void moe_route(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, cudaStream_t stream,
               RouterLogits precision = RouterLogits::kFp32);

/// Device-side dispatch of the routed (column, slot) entries by expert. Every array is
/// caller-owned device memory sized for `max_entries = k * T` entries and E experts.
struct MoeDispatch {
    std::int32_t* counts  = nullptr; // [E]
    std::int32_t* offsets = nullptr; // [E + 1]
    std::int32_t* cursor  = nullptr; // [E] scratch
    std::int32_t* entries = nullptr; // [k*T]: t * k + slot, grouped by expert
    std::int32_t* jobs    = nullptr; // [E]: experts with entries, ascending id
    std::int32_t* job_count = nullptr; // [1]
    // A dispatch with a shared job (shared_job of moe_dispatch): expert E (one past the routed ones)
    // holds entries shared_first + t for every column t, after the k*T routed entries
    // (shared_first = k*T); the arrays are then sized for E + 1 experts and k*T + T entries.
    // Set by the dispatch call; 0x7FFFFFFF without a shared job.
    std::int32_t shared_first = 0x7FFFFFFF;
};

[[nodiscard]] std::size_t moe_dispatch_bytes(std::int32_t experts, std::int32_t entries);
[[nodiscard]] MoeDispatch carve_moe_dispatch(void* base, std::int32_t experts, std::int32_t entries);
/// Fills `dispatch` from routing.ids: counts, offsets (exclusive scan, offsets[E] = k*T), jobs in
/// ascending expert id, job_count, and entries grouped by expert (their order within an expert is
/// unspecified). When `route_log` is not null it also receives a copy of the k*T ids. The arrays need
/// no clearing between calls, so a captured call replays correctly. With `shared_job` the layer's
/// shared expert is the last job: expert E with entries k*T + t (MoeDispatch::shared_first; the
/// route log keeps the routed ids only).
void moe_dispatch(const MoeRouting& routing, std::int32_t experts, MoeDispatch& dispatch,
                  std::int32_t* route_log, cudaStream_t stream, bool shared_job = false);
/// moe_route then moe_dispatch (experts = E of the logits), in one kernel for decode and
/// verification widths.
void moe_route_dispatch(const Tensor& logits, std::int32_t top_k, MoeRouting& routing, MoeDispatch& dispatch,
                        std::int32_t* route_log, cudaStream_t stream,
                        RouterLogits precision = RouterLogits::kFp32, bool shared_job = false);

/// The layer's routed expert records: expert e's record is at frame_base + frames[e] *
/// record_stride when frames[e] >= 0, else at host_records + e * record_stride (pinned host
/// memory). scales: device [E] ExpertScales of the layer.
///
/// Non-resident records are first copied to `staging_slots` device slots (staging_base + i *
/// record_stride) by a few CTAs that keep their reads within a compact address window, then read
/// from there: pinned memory read by many CTAs at scattered offsets, as the expert kernels would,
/// runs at under half the copy rate on the RTX 5090 (design section 8.6). The jobs run in passes
/// of staging_slots jobs, so every miss is staged; staging_slots == 0 reads misses zero-copy.
/// The staging slots are scratch of this call and are shared by every layer.
/// CPU-served misses (design section 10.3). When enabled for a call (max_jobs > 0 and the call
/// has at most max_columns <= kMaxCpuCallColumns columns), up to max_jobs <= kMaxCpuJobs of its
/// non-resident experts with at most max_job_columns columns, the fewest-column ones first, are
/// published as a request in mapped host memory, with the x columns they read (compacted: the
/// request's column k is x's k-th column read by a chosen job, in column order), and computed by
/// the host's expert engine (offloaded_moe::CpuMissService) while the GPU computes the other jobs.
/// Calls of at least wide_from columns (prefill; 0: none) take wide_jobs as their cap instead.
/// moe_experts returns after placing the host's outputs. The arithmetic is the same exact W4A4,
/// so where an expert is computed never changes a bit. The wait lasts as long as the host's
/// heartbeat word keeps changing; when it stops for MoeExpertSource::heartbeat_timeout_ns the call writes
/// kErrorHostSilent to the source's error word and leaves its CPU outputs unwritten (without an
/// error word, or without a heartbeat after 2 s, it traps).
struct MoeCpuChannel {
    offloaded_moe::MissRequest* request = nullptr; // mapped host memory
    std::uint16_t* x                    = nullptr; // mapped BF16 [H, kMaxCpuXColumns]
    const std::uint16_t* y              = nullptr; // mapped BF16 [H, max(max_jobs, wide_jobs) * kMaxCpuColumns]
    const std::uint32_t* done           = nullptr; // mapped; the host writes the answered sequence
    // Mapped; written before `done`: 0, or an errno value when an expert could not be computed (its
    // record could not be read). Null for services without one.
    const std::uint32_t* status         = nullptr;
    const std::uint32_t* heartbeat      = nullptr; // mapped; changes at least every few ms while the host is alive
    std::uint32_t* sequence             = nullptr; // device counter of published requests
    std::int32_t layer                  = 0;
    std::int32_t max_jobs               = 0;       // 0 disables
    std::int32_t max_columns            = 0;
    // misses / pcie_divisor of a call's misses stay on the GPU stage (0: the CPU takes up to
    // max_jobs of them), so the CPU and the PCIe stage share the call's misses.
    std::int32_t pcie_divisor           = 3;
    // When > 0, (misses * pcie_share) >> 8 of them stay on the GPU stage instead of misses /
    // pcie_divisor (a share in 256ths: Strata's measured link share, a temporary A/B of the Flash-Next
    // design §19.3.12 divisor rule). One-column calls with a one-column divisor keep theirs.
    std::int32_t pcie_share             = 0;
    // One-column calls (plain decode) use this divisor instead (0: pcie_divisor). When
    // one_column_divisor_device is set, they read it there at execution instead (a device I32 the
    // caller rewrites between rounds, so captured graphs follow it: Qwen4Exp's plain decode by
    // context; 0 or less: pcie_divisor).
    std::int32_t one_column_divisor     = 0;
    const std::int32_t* one_column_divisor_device = nullptr;
    std::int32_t max_job_columns        = offloaded_moe::kMaxCpuColumns; // 1..kMaxCpuColumns
    // Prefill CPU assist (design §19.3.1 P7): calls of at least wide_from columns (0: none) take up
    // to wide_jobs <= kMaxCpuJobs CPU-served misses.
    std::int32_t wide_from              = 0;
    std::int32_t wide_jobs              = 0;
    // The arithmetic the service computes W4A16 jobs with (CpuMissService::Options::narrow_a16);
    // moe_experts requires it to equal the source's narrow_a16, so a CPU-served miss keeps the GPU's bits.
    offloaded_moe::NarrowA16 narrow_a16 = offloaded_moe::NarrowA16::kExact;
};

/// Device memory a call warms into L2 while it waits for its CPU-served misses (design §19.3.3
/// Phase 2): typically the next layer's first weights. Spans start 16-byte aligned; a span's bytes
/// are rounded down to 16. Warming changes no result.
struct MoeL2Warm {
    static constexpr int kSpans = 12;
    const void* ptr[kSpans]   = {};
    std::size_t bytes[kSpans] = {};
};

/// The SSD expert tier's fetch channel (design §19.3.7; protocol in
/// ops/offloaded_sparse_moe/cpu/fetch_request.h). With a host table, a call's SSD-only experts that
/// no CPU job takes (and no F2 stream holds) are fetch-served: the call moves them to the end of
/// dispatch.jobs (a stable partition; outputs are indexed by entry, so no bit changes), publishes
/// them as one request before any pass, and its stage kernels copy each record from the host's
/// answer as soon as it lands, so a pass waits only for its own records and the reads run under
/// the earlier passes. One request is outstanding at a time (layers run in stream order).
struct MoeFetchChannel {
    offloaded_moe::FetchRequest* request        = nullptr; // mapped host memory
    const offloaded_moe::FetchResponse* response = nullptr; // mapped host memory
    std::uint64_t* consumed                     = nullptr; // mapped: sequence << 32 | records copied
    std::uint32_t* sequence                     = nullptr; // device counter of published requests
    const std::uint32_t* heartbeat              = nullptr; // mapped; changes while the host is alive
    std::int32_t layer                          = 0;
};

/// The record geometry of an exact-arithmetic expert bank: a GGUF (ggml) bank
/// (docs/maintainer/qwen3_8-flash-next-design.md §19.3.25, storage layouts §8: gate rows, up rows in gate_up_format, then
/// down rows in down_format, every row verbatim ggml blocks) or a block-FP8 bank (fp8_expert_b128_v1,
/// both formats FP8_E4M3FN_B128: the rows, one byte per weight, then their BF16 tile scales;
/// storage layouts §9). gate_up_format == NVFP4_MUL selects the NVFP4 record and its
/// routes. Both banks run exact A16 arithmetic on every column count, so their outputs depend on
/// neither placement nor batch.
// TEMPORARY (A/B of the GGUF banks' prefill expert tiles, docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.29): the defaults are the measured routing; from_environment() reads the switches once.
// - INFERNIX_TMP_IQ_STAGE_DECODE_MIN_COLUMNS / _MAX_COLUMNS: ggml-bf16-prefill calls
//   (WideA16::kGgmlBf16) of [min, max) columns run the BF16 stage-decode tiles (each K stage's codes
//   decoded into shared memory, no planes: the planes' bits) instead of the exact tiles or the
//   planes. Unset: no call (min = max = INT32_MAX).
// - INFERNIX_TMP_IQ_PLANE_MIN_COLUMNS: ggml-bf16-prefill calls of at least this many columns carve
//   the planes (the workspace layout reads from_environment()'s, every call the same) and, unless
//   stage_decode takes them, run on them; below it the exact tiles. Default 14,336
//   (offloaded_moe::wide::kGgmlBf16MinColumns); lower only to time the planes at narrower calls.
// - INFERNIX_TMP_GGML_SHARED_CODEBOOKS=1: the exact gate/up tiles and the plane decode read the
//   codebooks from shared memory (the same bits).
struct MoeGgmlTiles {
    std::int32_t stage_decode_min_columns = INT32_MAX;
    std::int32_t stage_decode_max_columns = INT32_MAX;
    std::int32_t plane_min_columns        = 14336;
    bool shared_codebooks                 = false;

    [[nodiscard]] bool stage_decode(std::int32_t columns) const noexcept {
        return columns >= stage_decode_min_columns && columns < stage_decode_max_columns;
    }
    // Read once (core/temporary_switches.h, which the startup log lists).
    [[nodiscard]] static const MoeGgmlTiles& from_environment();
};

struct MoeGgmlBank {
    QType gate_up_format       = QType::NVFP4_MUL;
    QType down_format          = QType::NVFP4_MUL;
    std::uint32_t gate_row_bytes = 0;
    std::uint32_t down_row_bytes = 0;
    std::uint64_t record_bytes = 0; // bytes a stage copies: the record rounded up to 16 (within its stride)
    MoeGgmlTiles tiles;             // the prefill tiles' TEMPORARY switches (default: the measured routing)
};

struct MoeExpertSource {
    const std::uint8_t* frame_base   = nullptr;
    const std::int32_t* frames       = nullptr; // device [E]
    // A non-resident expert's host record: host_table[e] when host_table is set (device [E] of
    // host pointers, the SSD tier's RAM slots, design §19.3.7; a null entry is an SSD-only expert:
    // CPU jobs take those first and read them through the service's RecordProvider, and the fetch
    // channel serves the rest), else host_records + e *
    // record_stride. A record is read from a frame, a host pointer or a landing; its bits are
    // identical in each.
    const std::uint8_t* host_records = nullptr;
    const std::uint8_t* const* host_table = nullptr;
    // Optional mapped word: a call that cannot serve an expert (an SSD-only record with no CPU job
    // and no fetch channel, a record the host could not read, or a host whose heartbeat stopped)
    // writes layer << 16 | code there (offloaded_moe::kError* or an errno); that call's outputs are
    // undefined and the caller must fail the round. The word is the caller's to clear.
    std::uint32_t* error = nullptr;
    // How long the CPU wait and the fetch poller wait on a heartbeat that stands still (GPU time)
    // before they write kErrorHostSilent / kErrorFetchSilent.
    std::uint64_t heartbeat_timeout_ns = offloaded_moe::kHeartbeatTimeoutNs;
    // TEMPORARY (INFERNIX_TMP_REPLAY_BENCH): optional device word; each call's CPU wait adds the GPU
    // nanoseconds it waited for the host's answer (the replay bench's waits, subtracted from its
    // round time).
    unsigned long long* wait_ns = nullptr;
    std::uint64_t record_stride      = 0;
    const offloaded_moe::ExpertScales* scales = nullptr;
    // The layer's expert arithmetic: W4A4 with stored input scales (§16.2), or W4A16 for experts
    // stored without activation scales (§16.2.1; scales then hold the multipliers in alpha_*). The
    // CPU channel's service must compute with the same arithmetic.
    offloaded_moe::ExpertActivation activation = offloaded_moe::ExpertActivation::kA4;
    // W4A16 experts' wide route: exact (default) or the two-term A4 fast path (design §19.3.23); the
    // workspace must be sized for the same choice. Narrow routes and W4A4 experts ignore it.
    offloaded_moe::WideA16 wide_a16 = offloaded_moe::WideA16::kExact;
    // W4A16 experts' narrow route (experts of at most kMaxColumns columns in the call): exact (default)
    // or expert-a4-residual-decode's two-term A4 with exact sums (offloaded_moe::NarrowA16; NVFP4 banks
    // at Flash-Next's geometry, any wide route but kLossless). The workspace is the one wide_a16 sizes;
    // a CPU channel must carry the same choice. W4A4 experts ignore it.
    offloaded_moe::NarrowA16 narrow_a16 = offloaded_moe::NarrowA16::kExact;
    // With kLossless: the design reference's expert activations (offloaded_moe::LosslessExpertDesign).
    offloaded_moe::LosslessExpertDesign lossless_design = offloaded_moe::LosslessExpertDesign::kNone;
    std::uint8_t* staging_base  = nullptr;
    std::int32_t staging_slots  = 0;
    MoeCpuChannel cpu;
    MoeFetchChannel fetch; // with host_table only; needs staging_slots > 0 and max_jobs <= kMaxFetch
    // Optional overlap of staging with compute for calls of several passes (prefill chunks): the
    // slots are split in two halves and pass p+1 is staged on `overlap_stream` while pass p
    // computes. The stream and the five events (start, staged[2], consumed[2]) are caller-owned;
    // the stream is otherwise idle. Not for calls captured into a CUDA graph.
    cudaStream_t overlap_stream = nullptr;
    cudaEvent_t overlap_events[5] = {};
    // Optional fork for calls of one pass (decode and verification widths; may be captured into a
    // CUDA graph): the pass's misses are staged and computed on `fork_stream` while the calling
    // stream computes the resident experts. The calling stream joins the fork in
    // moe_experts_cpu_wait (immediately when wait_for_cpu is true). The stream and both events
    // (fork, join) are caller-owned; the stream is otherwise idle.
    cudaStream_t fork_stream = nullptr;
    cudaEvent_t fork_events[2] = {};
    // Optional landing for forked calls (design §19.3.5 S4; ignored by every other route): the n-th
    // staged miss in job order (CPU-served misses excluded) is copied into frame landing[n]
    // (frame_base + landing[n] * record_stride) instead of a staging slot when n < landing_slots
    // and landing[n] >= 0, and stage block 0 writes landed[n] = its expert; other entries of
    // `landed` are left as they are. Device arrays of landing_slots <= kMaxLandingSlots entries.
    // The frames must be free (no expert, not read by this call); the caller adopts the landed
    // experts afterwards. Results are unchanged: a landed record is read exactly as a staged one.
    const std::int32_t* landing = nullptr;
    std::int32_t* landed        = nullptr;
    std::int32_t landing_slots  = 0;
    // Prefetched into L2 by extra CTAs of the CPU wait while the host computes this call's misses
    // (only when a request was published).
    MoeL2Warm l2_warm;
    // Optional streamed records (design §19.3.8 F2): prefetched (device [E]) gives, for a
    // non-resident expert, the slot whose record the caller has already copied to prefetch_base +
    // slot * prefetch_stride (ordered before this call), or -1. Such an expert is read there like a
    // resident one: never staged and never CPU-served (the CPU takes misses the stream left out).
    const std::int32_t* prefetched     = nullptr;
    const std::uint8_t* prefetch_base  = nullptr;
    std::uint64_t prefetch_stride      = 0; // 0 means frame_stride's
    // The pitch of device frames, staging slots and landings; 0 means record_stride. The caller's
    // frames of a layer's bank have that bank's record stride (a frame pool per record size).
    std::uint64_t frame_stride = 0;
    MoeGgmlBank ggml;
    // The bank's expert geometry (offloaded_moe::ExpertShape): Qwen3.8-Flash-Next's H 2560 / I 640
    // for every bank, or Qwen3.5-122B-A10B's H 3072 / I 1024 for NVFP4 W4A16 banks on the exact, A4
    // residual and lossless routes. x and outputs have shape.hidden rows; the CPU channel's service serves the
    // same geometry.
    offloaded_moe::ExpertShape shape = offloaded_moe::kFlashNextShape;
    // The layer's shared expert as the call's last job (design §8.6; NVFP4 W4A16 banks): expert
    // index shared_expert (= E, one past the routed experts; -1: none) of a dispatch made with
    // shared_job, its record shared_record (device memory, the bank's record layout: always
    // resident, never staged or CPU-served) and scales[E] its multipliers (so scales holds E + 1).
    // Its outputs are the last T columns of outputs, which has k*T + T columns.
    std::int32_t shared_expert         = -1;
    const std::uint8_t* shared_record  = nullptr;
    // Approximate tail skip (0: off; 1..top_k-1): in a CPU-served call (moe_experts' CPU plan runs), an
    // expert neither resident nor prefetched, of at most kMaxColumns columns, that every column of the
    // call routes at top-k slot >= tail_skip_rank (the smallest router weights) is neither staged,
    // fetched nor computed: its outputs are zero and the other experts keep their weights (no
    // renormalisation). It changes the output; the routing and route log are unchanged, so the
    // expert is still credited as routed.
    std::int32_t tail_skip_rank = 0;
};

/// Workspace of one moe_experts call of `columns` (T) columns and `entries` (k*T) routed entries
/// with the layer's expert geometry (MoeExpertSource::shape) and arithmetic.
[[nodiscard]] std::size_t moe_experts_workspace_bytes(const offloaded_moe::ExpertShape& shape, std::int32_t max_jobs,
                                                      std::int32_t entries, std::int32_t columns,
                                                      offloaded_moe::ExpertActivation activation,
                                                      offloaded_moe::WideA16 wide_a16 = offloaded_moe::WideA16::kExact);

/// x: BF16 [H, T]. outputs: BF16 [H, k*T], column t*k + slot receives expert ids[slot, t]'s
/// output for column t. max_jobs bounds the job count the grid covers (min(E, k*T)).
///
/// With wait_for_cpu == false the call returns before placing CPU-served outputs; the caller may
/// enqueue unrelated work (the shared expert) and must call moe_experts_cpu_wait with the same
/// arguments before reading `outputs`. The workspace stays reserved until then.
void moe_experts(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                 std::int32_t top_k, std::int32_t max_jobs, void* workspace, Tensor& outputs,
                 cudaStream_t stream, bool wait_for_cpu = true, bool cpu_planned = false);

/// moe_route_dispatch and, for a CPU-served call of decode or verification width, the CPU plan of
/// the moe_experts call with these x, source, max_jobs, workspace and outputs, in one kernel.
/// Returns true when it planned; that moe_experts call then passes cpu_planned = true.
[[nodiscard]] bool moe_route_dispatch_plan(const Tensor& logits, std::int32_t top_k, MoeRouting& routing,
                                           MoeDispatch& dispatch, std::int32_t* route_log, const Tensor& x,
                                           const MoeExpertSource& source, std::int32_t max_jobs, void* workspace,
                                           const Tensor& outputs, cudaStream_t stream,
                                           RouterLogits precision = RouterLogits::kFp32,
                                           bool shared_job = false);

/// Waits for the host's answer to a moe_experts call made with wait_for_cpu == false and places
/// its outputs; a no-op when that call published no request.
void moe_experts_cpu_wait(const Tensor& x, const MoeDispatch& dispatch, const MoeExpertSource& source,
                          std::int32_t max_jobs, void* workspace, Tensor& outputs, cudaStream_t stream);

/// y = bf16(sum_slot w * outputs + shared_gate * shared), BF16 [H, T]; 0 in a column
/// routing.mask masks.
void moe_combine(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared, Tensor& y,
                 cudaStream_t stream);
/// moe_combine followed by hyper_connection_inject(y, inject, residual) without materializing y:
/// residual[s*H+d, t] = bf16(residual[s*H+d, t] + inject[s, t] * y[d, t]), y the BF16 combine of
/// column t, with the same bits as the two calls. inject: FP32 [S, T]; residual: BF16 [S*H, T].
void moe_combine_inject(const Tensor& outputs, const MoeRouting& routing, const Tensor& shared,
                        const Tensor& inject, Tensor& residual, cudaStream_t stream);

} // namespace infernix::ops
