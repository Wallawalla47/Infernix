#pragma once

// The Qwen4Exp forward pass (docs/maintainer/qwen3_8-flash-next-design.md §2, §8): embedding,
// the PLE injection, 48 hyper-connection blocks of GDN or QSA plus the offloaded MoE, the final
// mixer and the LM head. One call runs `batch` sequences of `width` consecutive positions: a
// prefill chunk is one sequence of many positions, a decode round several sequences of one.

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "models/qwen4_exp/execution/expert_stream.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/qsa.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

// Columns of one MTP call inside a prefill chunk (bounds the drafter's share of the workspace);
// the Program stages the chunk's RoPE positions per sub-chunk of this many cells.
inline constexpr std::int32_t kMtpChunkColumns = 512;

// Program-owned recurrent state of every slot.
struct ForwardState {
    LinearAttentionStatePool* gdn = nullptr; // one layer per GDN block
    Tensor ple_conv;                         // BF16 [S*H, span, slots]
    std::vector<Tensor> qsa_tails;           // per attention block: BF16 [Di, R - 1, slots]
    Tensor mtp_tails;                        // the MTP block's tails (empty without MTP)
    Tensor mtp_ones;                         // FP32 [S, columns] of 1: broadcasts the embedding
};

// Program-owned paged KV of the attention blocks.
struct ForwardKV {
    std::vector<ops::QsaKVLayer> layers; // per attention block
    std::optional<ops::QsaKVLayer> mtp;  // the MTP block's layer, through the same tables
    Tensor block_tables;                 // I32 [pages per row, rows]
};

// The CPU's share of a gated prefill call's experts (design §19.3.12): from the call's columns per
// expert in one layer, marks in `cpu` the experts the CPU serves, which must be non-resident and
// have at most ops::offloaded_moe::kMaxCpuColumns columns (the narrow route, placement-invariant),
// and returns their number. Owned by the Program, which knows the link and the CPU's rates.
class CpuSplitPolicy {
public:
    virtual ~CpuSplitPolicy() = default;
    virtual std::uint32_t choose(std::uint32_t layer, std::span<const std::int32_t> columns,
                                 std::span<std::uint8_t> cpu) const = 0;
};

// Where each layer's routed experts live: device frames for resident experts, the pinned host
// bank otherwise.
struct ForwardExperts {
    const std::uint8_t* frame_base = nullptr;
    std::uint64_t frame_stride     = 0;
    std::vector<const std::int32_t*> frames; // per layer, device I32 [E]: frame or -1
    // Optional route log for the expert cache: layer l's routed ids (I32 [k, T]) are copied to
    // route_log + l * route_stride.
    std::int32_t* route_log  = nullptr;
    std::size_t route_stride = 0;
    // Device slots the MoE stages non-resident records into before computing them (shared by
    // every layer; see ops::MoeExpertSource).
    std::uint8_t* staging_base  = nullptr;
    std::int32_t staging_slots  = 0;
    // Per layer: the channel to the host expert engine for CPU-served misses (empty disables).
    std::vector<ops::MoeCpuChannel> cpu;
    // SSD tier (design §19.3.7; empty in full mode): per layer, the device table of host record
    // pointers (null: SSD-only; replaces the pinned bank) and the fetch channel its SSD-only
    // experts without a CPU job are read through.
    std::vector<const std::uint8_t* const*> host_tables;
    std::vector<ops::MoeFetchChannel> fetch;
    // Mapped word a call that could not serve an expert writes (ops::MoeExpertSource::error); the
    // Program checks it after each synchronization. Null: none.
    std::uint32_t* error = nullptr;
    // Prefill chunks stage their next pass of misses on this stream while one pass computes.
    cudaStream_t overlap_stream = nullptr;
    std::array<cudaEvent_t, 5> overlap_events{};
    // Landing frames of decode and verification calls (design §19.3.5 S4; see
    // ops::MoeExpertSource::landing): layer l uses landing + l * landing_slots and landed + l *
    // landing_slots. Null disables.
    const std::int32_t* landing = nullptr;
    std::int32_t* landed        = nullptr;
    std::int32_t landing_slots  = 0;
    // Prefill chunks read the experts this stream copied ahead while it is active (design §19.3.8
    // F2; owned by the Program, which starts it per chunk).
    ExpertStream* stream = nullptr;
    // A gated stream's split of each layer's experts between the link and the CPU (§19.3.12).
    const CpuSplitPolicy* split = nullptr;
};

// A speculative verification call (design §11): `batch` sequences of `width` >= 2 positions. It
// leaves every recurrent state, PLE history and QSA tail unchanged and records what the commit of
// an accepted prefix needs instead; K/V and pooled keys are written as usual, since positions past
// the accepted prefix are rewritten before any later query reads them.
struct ForwardVerify {
    GdnReplayRecords gdn; // narrowed to the call's width; physical record row b = sequence b
    Tensor ple_inputs;    // BF16 [S*H, W, B]: the PLE convolution inputs
    Tensor qsa_keys;      // BF16 [Di, W, B, attention layers]: the raw index keys
};

// Visual embeddings of a call's image columns (design §19.3.2): they replace those columns' token
// embeddings before the stream expand (transformers' masked_scatter precedes repeat).
struct VisionInput {
    Tensor embeddings; // BF16 [H, m]: the columns' embeddings, in column order
    Tensor columns;    // I32 [m]: call-local columns, ascending; empty (no data) when m = 0
};

// The MTP cells of a prefill or forced-token chunk (one sequence), written from the chunk's own
// residuals while they are live: cell c pairs the residual at position c with the token at c + 1.
// The columns are `saved` (the residual of the position before the chunk) when `prepend`, then
// the chunk's first `columns` residuals; the chunk's last residual then replaces `saved`. Cells
// run in sub-chunks of kMtpChunkColumns; sub-chunk k's RoPE positions are the [n_k, 3] block at
// word 3 * kMtpChunkColumns * k of `rope_positions`, its pooled-block start the three words at
// 3 * k of `block_start_rope`.
struct MtpChunk {
    Tensor saved;     // BF16 [S*H]: the sequence's pending residual
    bool prepend = false;
    std::int32_t columns = 0;
    Tensor ids;       // I32 [prepend + columns]: the token after each cell
    Tensor positions; // I32 [prepend + columns]: the cells
    std::int32_t* rope_positions   = nullptr; // device words, per sub-chunk [n_k, 3]
    std::int32_t* block_start_rope = nullptr; // device words, [3] per sub-chunk
    // Per sub-chunk: the visual embeddings of cells whose next token is an image token (empty
    // columns when none); null for a prompt without media.
    const VisionInput* vision = nullptr;
};

// A call whose n-gram rows the host writes after the launch (design §12.3, n-gram S2): before
// ple_embed, the rows are copied from `pinned_rows` into ForwardBatch::ngram_rows once the pinned
// `ready` word equals the device word `expected` (core upload_pinned_when).
struct NgramRowGate {
    const void* pinned_rows        = nullptr;
    const std::uint32_t* ready     = nullptr; // pinned
    const std::uint32_t* expected  = nullptr; // device
    std::uint64_t* wait_stats      = nullptr; // device U64 [rows, 2]: wait ns and waits, per row
    const std::int32_t* wait_row   = nullptr; // device: the row to add to
};

struct ForwardBatch {
    Tensor ids;        // I32 [T]
    Tensor positions;  // I32 [T]: KV position of each column
    Tensor rope_positions;   // I32 [T, 3] axis-major: RoPE position of each column
    Tensor block_start_rope; // I32 [batch, 3] axis-major: RoPE position of each sequence's first
                             // pooled-block token (ops::QsaBatch::block_start_rope)
    Tensor ngram_rows; // U8 [row bytes, heads, T]: each column's FP8 n-gram rows
    const NgramRowGate* ngram_gate = nullptr; // when set: ngram_rows are written by the gate
    Tensor slots;      // I32 [batch]: state slot of each sequence, updated in place
    Tensor table_rows; // I32 [batch]: KV block-table row of each sequence
    Tensor logit_columns; // I32 [n]: the columns whose logits are produced (each sequence's last
                          // for generation, every column for teacher-forced scoring)
    std::span<const std::int32_t> host_slots;      // the same slots on the host
    std::span<const std::int32_t> host_table_rows; // the same rows on the host
    std::int32_t batch = 0;
    std::int32_t width = 0;
    const ForwardVerify* verify = nullptr; // set for a speculative verification call
    Tensor residual_out;                   // when set: BF16 [S*H, T] copy of the final residual
    const MtpChunk* mtp_chunk = nullptr;   // when set: the chunk's MTP cells (one sequence)
    // Prefix-cache copies of the call's lane in flight (a restore into it, a copy-out of it): each
    // empty or one event per decoder layer, then one for the MTP block. Layer l waits for entry l
    // before it reads or writes its state or KV planes; the MTP block waits for the last entry.
    std::array<std::span<const cudaEvent_t>, 2> layer_waits{};
    const VisionInput* vision = nullptr;   // when set: image columns of a prefill call
    // Prefill expert streaming (ExpertStream): the call's MoE reads the active stream's ring, and
    // with `release_stream` frees each layer's half once its experts are enqueued. A layer walk sets
    // release_stream on the span's last chunk only (every chunk reads the same layer copies).
    bool stream         = false;
    bool release_stream = true;
    // With a gated stream (ExpertStream::gated): each layer's routing decides, through
    // ForwardExperts::split, which experts the stream copies and which the CPU serves.
    bool split          = false;
};

// One call of the MTP drafter (design §11.2). Cell c of a sequence pairs a residual at position c
// (the main model's, or the drafter's own output for a chained step) with the token at c + 1, at
// the target's RoPE position of c; the block's output predicts the token at c + 2.
struct MtpCall {
    Tensor residuals;  // BF16 [S*H, T]
    Tensor ids;        // I32 [T]
    Tensor positions;  // I32 [T]: the cells
    Tensor rope_positions;   // I32 [T, 3] axis-major
    Tensor block_start_rope; // I32 [batch, 3] axis-major
    Tensor slots;      // I32 [batch]
    Tensor table_rows; // I32 [batch]
    std::int32_t batch = 0, width = 0;
    // K/V and index keys only (prompt, forced and verified cells). The index-key tails advance,
    // unless key_records ([Di, W, B]) receives the keys for a later qsa_commit_tails.
    bool kv_only = true;
    Tensor key_records;
    // A full step (one column per sequence, tails unchanged): the block's output residual and the
    // draft head's argmax token of each column.
    Tensor residual_out; // BF16 [S*H, T]
    Tensor drafts;       // I32 [T]
    // Cells whose next token is an image token take its visual embedding (prompt cells only).
    const VisionInput* vision = nullptr;
};

// Optional observation of every block, for reference comparison.
struct ForwardTap {
    std::vector<Tensor>* residuals = nullptr; // receives BF16 [S*H, T] copies, one per block
    std::vector<Tensor>* routes    = nullptr; // receives I32 [top k, T] routed expert ids, one per block
    // BF16 [H, T] copies per block of the mixer (GDN/QSA) input and output and the MoE input
    // and output, so each op chain can be checked against the reference on identical inputs.
    std::vector<Tensor>* mixer_inputs  = nullptr;
    std::vector<Tensor>* mixer_outputs = nullptr;
    std::vector<Tensor>* moe_inputs    = nullptr;
    std::vector<Tensor>* moe_outputs   = nullptr;
};

// The QSA geometry of the configuration.
[[nodiscard]] ops::QsaGeometry qsa_geometry(const TextConfig& config);

class Forward {
public:
    Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work,
            ForwardState state, ForwardKV kv, ForwardExperts experts, std::int32_t max_context);
    ~Forward();
    Forward(const Forward&)            = delete;
    Forward& operator=(const Forward&) = delete;

    // FP32 logits [V, n] of batch.logit_columns.
    void run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap = nullptr);

    // The steps run() composes, for a layer walk over a prefill span (design §19.3.8 F4): the
    // same chunks executed layer-major instead of chunk-major. Each step of a chunk starts with
    // begin_call (it resets the workspace); `residual` (BF16 [S*H, T]) lives outside the
    // workspace. embed writes the chunk's expanded embedding into it; layer runs one decoder
    // layer; finish runs the chunk's MTP cells and, when logits is set, the final mixer and the
    // LM head. Outputs are run()'s bit for bit: every step's kernels, inputs and order within the
    // chunk are the same.
    void begin_call(const ForwardBatch& batch, const Tensor* logits);
    void embed(const ForwardBatch& batch, Tensor& residual);
    void layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, const ForwardTap* tap = nullptr);
    void finish(const ForwardBatch& batch, Tensor& residual, Tensor* logits);

    // Runs the MTP drafter (requires MTP parameters and state).
    void run_mtp(const MtpCall& call);

    [[nodiscard]] static std::size_t workspace_bytes(const TextConfig& config, std::int32_t columns,
                                                     std::int32_t max_context, KvCacheStorage storage);
    // Extra workspace of the MTP drafter: K/V-only calls over up to `kv_columns` columns (chunks
    // run them inside run()) and full steps over `draft_columns` sequences.
    [[nodiscard]] static std::size_t mtp_workspace_bytes(const TextConfig& config, std::int32_t kv_columns,
                                                         std::int32_t draft_columns, std::int32_t vocabulary_rows,
                                                         std::int32_t max_context, KvCacheStorage storage);

private:
    void ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch);
    Tensor mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject);
    Tensor gdn(const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch);
    // One QSA block's attention through `layer` and `tails`. Index keys go to `key_records` when
    // set (verification and MTP catch-up: tails unchanged) or advance the tails when update_tails;
    // kv_only stops after the K/V and index-key writes and returns an empty tensor.
    struct AttentionCall {
        const ops::QsaKVLayer& layer;
        Tensor& tails;
        Tensor key_records;
        bool update_tails = true;
        bool kv_only      = false;
        const Tensor& positions;
        const Tensor& rope_positions;
        const Tensor& block_start_rope;
        const Tensor& slots;
        const Tensor& table_rows;
        std::int32_t batch = 0, width = 0;
    };
    Tensor attention(const AttentionParameters& p, const Tensor& x, const AttentionCall& call);
    Tensor moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap,
               const ForwardBatch* batch = nullptr);
    void wait_layer(const ForwardBatch& batch, std::size_t entry);
    void mtp_block(const MtpCall& call, Tensor& residual);

    const Parameters& parameters_;
    const TextConfig& config_;
    DeviceContext& device_;
    WorkspaceArena& work_;
    ForwardState state_;
    ForwardKV kv_;
    ForwardExperts experts_;
    std::int32_t max_context_;
    bool eager_chunk_ = false;
    // A gated call's layer routing on the host: the dispatch offsets (pinned I32 [E + 1]), the
    // event that lands them, and the per-expert columns and CPU marks derived from them.
    PinnedHostBuffer split_offsets_{1};
    cudaEvent_t split_ready_ = nullptr;
    std::vector<std::int32_t> split_columns_;
    std::vector<std::uint8_t> split_cpu_;
    // The CPU-served and streamed experts of gated calls (diagnostics).
    std::uint64_t split_cpu_experts_ = 0, split_streamed_experts_ = 0;

public:
    struct SplitStats {
        std::uint64_t cpu_experts      = 0;
        std::uint64_t streamed_experts = 0;
    };
    [[nodiscard]] SplitStats split_stats() const noexcept { return {split_cpu_experts_, split_streamed_experts_}; }
};

} // namespace ninfer::models::qwen4_exp::execution
