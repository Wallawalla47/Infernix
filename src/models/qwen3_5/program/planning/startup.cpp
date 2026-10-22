#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/lossless.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/offloaded_moe.h"
#include "models/qwen3_5/program/planning/fast_paths.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/planning/idle_prefill.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/prefix/hybrid_host_layout.h"
#include "models/qwen3_5/program/speculative/dflash_width_policy.h"
#include "models/qwen3_5/program/speculative/mtp_draft_policy.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/execution/workspace.h"
#include "core/device.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/candidate_selector.h"
#include "infernix/ops/context_kv_materialize.h"
#include "infernix/ops/dynamic_grouped_conv.h"
#include "infernix/ops/linear_topk.h"
#include "infernix/ops/gdn_gating_proj.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/linear_add.h"
#include "infernix/ops/linear_swiglu.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/sliding_window_attention.h"
#include "infernix/ops/softmax_attention.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/speculative_tree.h"
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::qwen3_5::detail {
namespace {

using execution::dimension;
namespace workspace = execution::workspace;

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

bool tree_table_enabled(const std::array<std::uint32_t, kMaximumConcurrency>& nodes) {
    return std::any_of(nodes.begin(), nodes.end(), [](std::uint32_t n) { return n != 0; });
}

// Whether every GDN layer's input projection is a single FP8 or NVFP4 parent: tree convolution
// reads each column's ancestors from that parent's materialized projection.
bool tree_verification_supported(const execution::Parameters& parameters) {
    for (const auto& block : parameters.text.layers) {
        const auto* gdn = std::get_if<execution::GdnParameters>(&block.mixer);
        if (gdn == nullptr) { continue; }
        const auto* single = std::get_if<ops::SingleProjectionWeight>(&gdn->projection);
        if (single == nullptr || (single->weight.qtype != QType::NVFP4 &&
                                  !is_fp8_row(single->weight.qtype))) {
            return false;
        }
    }
    return true;
}

// The tree widths the options select: their fixed table, or the automatic widths for rounds of
// up to kDraftTreeAutoMaxBatch rows. Automatic mode on a target that cannot verify trees resolves
// to chain verification.
TreeWidthPlan resolve_tree_widths(const execution::Parameters& parameters,
                                  const EngineOptions& options) {
    TreeWidthPlan plan{.fixed = options.speculative.draft_tree_nodes};
    if (options.speculative.draft_tree_auto && tree_verification_supported(parameters)) {
        const auto widths = draft_tree_auto_widths(options.speculative.draft_tokens);
        plan.automatic.assign(widths.begin(), widths.end());
        plan.automatic_max_batch = std::min(kDraftTreeAutoMaxBatch, options.max_concurrency);
    }
    return plan;
}

bool plan_has_split(const SequencePlanImpl& plan) {
    return std::any_of(plan.round_shapes.begin(), plan.round_shapes.end(),
                       [](const SpeculativeRoundShape& shape) { return shape.split; });
}

bool plan_has_tree(const SequencePlanImpl& plan) {
    return std::any_of(plan.round_shapes.begin(), plan.round_shapes.end(),
                       [](const SpeculativeRoundShape& shape) {
                           return shape.kind == SpeculativeRoundKind::Tree;
                       });
}

// The speculative round families an engine captures. Neural rounds verify at the drafter's window
// and n-gram copy rounds at the n-gram window, for every backend. With adaptive-mtp every shorter
// MTP draft length has its own neural family; lengths outside the batch lengths serve one row. With
// adaptive-dflash the neural rounds instead verify, after a proposal of the whole block, one of the
// policy's widths (dflash_width_policy.h), each a split family; widths outside the batch widths
// serve one row. With a batch length (batch_drafts) there are no batch widths: rounds of several
// rows run a plain neural family of that length.
std::vector<SpeculativeRoundShape> speculative_round_shapes(const EngineOptions& options,
                                                            const TreeWidthPlan& tree_widths,
                                                            bool adaptive_mtp, bool adaptive_dflash,
                                                            std::uint32_t batch_drafts) {
    const SpeculativeOptions& spec = options.speculative;
    std::vector<SpeculativeRoundShape> shapes;
    if (spec.backend == SpeculativeBackend::None) { return shapes; }
    if (adaptive_dflash && is_masked_draft_backend(spec.backend)) {
        const bool plain_batches = batch_drafts != 0 && options.max_concurrency > 1;
        const std::vector<std::uint32_t> batch =
            plain_batches ? std::vector<std::uint32_t>{} : dflash_policy_widths(spec.draft_tokens, true);
        for (const std::uint32_t width : dflash_policy_widths(spec.draft_tokens, false)) {
            const bool batches = std::find(batch.begin(), batch.end(), width) != batch.end();
            shapes.push_back({.kind          = SpeculativeRoundKind::Neural,
                              .verify_drafts = width - 1U,
                              .single_row    = options.max_concurrency > 1 && !batches,
                              .split         = true});
        }
        if (plain_batches) {
            shapes.push_back({.kind          = SpeculativeRoundKind::Neural,
                              .verify_drafts = std::min(batch_drafts, spec.draft_tokens),
                              .batch_only    = true});
        }
    } else {
        shapes.push_back({SpeculativeRoundKind::Neural, spec.draft_tokens});
    }
    if (adaptive_mtp && spec.backend == SpeculativeBackend::Mtp) {
        for (std::uint32_t k = 1; k < spec.draft_tokens; ++k) {
            shapes.push_back({SpeculativeRoundKind::Neural, k,
                              options.max_concurrency > 1 &&
                                  !mtp_batch_draft_length(k, spec.draft_tokens)});
        }
    }
    if (spec.ngram_draft_tokens != 0) {
        // Copy rounds take the narrowest n-gram family that holds their longest copy, so short
        // copies do not pay for the whole window. Narrower families lie strictly between the
        // neural proposal a copy must exceed (ProgramImpl::ngram_copy_floor: the neural window,
        // or adaptive-dflash's batch length) and the n-gram window.
        std::uint32_t copy_floor = spec.draft_tokens;
        if (adaptive_dflash && is_masked_draft_backend(spec.backend) && batch_drafts != 0 &&
            options.max_concurrency > 1) {
            copy_floor = std::min(copy_floor, batch_drafts);
        }
        for (const std::uint32_t narrow : {7U, 15U, 31U}) {
            if (narrow > copy_floor && narrow < spec.ngram_draft_tokens) {
                shapes.push_back({SpeculativeRoundKind::Ngram, narrow});
            }
        }
        // Where batched copies stop at a narrower window, the widest family serves one row.
        const bool narrow_batches = spec.ngram_batch_draft_tokens != 0 &&
                                    spec.ngram_batch_draft_tokens < spec.ngram_draft_tokens;
        if (narrow_batches && std::none_of(shapes.begin(), shapes.end(), [&](const SpeculativeRoundShape& s) {
                return s.kind == SpeculativeRoundKind::Ngram && s.verify_drafts == spec.ngram_batch_draft_tokens;
            })) {
            shapes.push_back({SpeculativeRoundKind::Ngram, spec.ngram_batch_draft_tokens});
        }
        shapes.push_back({.kind          = SpeculativeRoundKind::Ngram,
                          .verify_drafts = spec.ngram_draft_tokens,
                          .single_row    = narrow_batches && options.max_concurrency > 1});
    }
    // One tree family per distinct tree width.
    for (const std::uint32_t nodes : tree_widths.widths()) {
        const SpeculativeRoundShape shape{SpeculativeRoundKind::Tree, nodes - 1U};
        if (std::none_of(shapes.begin(), shapes.end(), [&](const SpeculativeRoundShape& other) {
                return other.kind == shape.kind && other.verify_drafts == shape.verify_drafts;
            })) {
            shapes.push_back(shape);
        }
    }
    return shapes;
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// The Main KV pages a plan may hold. Without a prefix cache only the C active lanes' windows
// hold pages, so C * L pages bound it. The prefix cache keeps every Device page no active lease
// holds as cached blocks, so only the token capacity representation (pages * page size in int32)
// bounds it and automatic sizing spends the free VRAM on cache.
std::uint64_t maximum_main_page_groups(std::uint32_t concurrency, std::uint32_t logical_pages,
                                       const ContextCacheOptions& cache) {
    std::uint64_t maximum = static_cast<std::uint64_t>(concurrency) * logical_pages;
    if (cache.enabled) {
        maximum = std::max<std::uint64_t>(
            maximum, static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) /
                         static_cast<std::uint64_t>(kPagedKVPageSize));
    }
    if (maximum > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    return maximum;
}

// Device memory the driver takes for CUDA Graph decode executables. Every topology class of a
// graph family is instantiated as one executable per batch size, uploaded and launched once at
// startup, so graph memory scales with the executable count. Measured on an RTX 5090 (CUDA 13.4,
// Qwen3.8-27B NVFP4, as the drop in free Device memory after startup against --no-cuda-graph at
// the same KV capacity): 2.2-2.9 MiB per executable for no speculation, MTP, DFlash2 and DFlash2
// with n-gram drafting (16 MiB for DFlash2 at max concurrency 1, 274 MiB for DFlash2 with n-gram
// at 8), rising to 4.1 MiB only where MTP verifies a 15-wide n-gram window at batch 4-8 (198 MiB
// at max concurrency 8, the tightest case against this allowance). The allowance keeps 4 MiB per
// executable plus a fixed 64 MiB for driver and module state.
constexpr std::size_t kGraphExecutableAllowance = 4ULL * kMiB;
constexpr std::size_t kGraphDriverAllowance     = 64ULL * kMiB;

// Number of executables one family instantiates for one batch size: one per topology class.
std::uint32_t graph_topology_classes(const std::vector<GraphExecutionProfile>& profiles) {
    std::vector<std::uint32_t> classes;
    for (const GraphExecutionProfile profile : profiles) {
        if (std::find(classes.begin(), classes.end(), profile.topology_class) == classes.end()) {
            classes.push_back(profile.topology_class);
        }
    }
    return static_cast<std::uint32_t>(classes.size());
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;

    // One StateImage per lane, plus the prefix cache's Device snapshot slots.
    if (!plan.context_cache.hybrid.device_snapshot_slots) {
        throw std::logic_error("Qwen3.5 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) +
            *plan.context_cache.hybrid.device_snapshot_slots,
        "Qwen3.5 StateImage slot count exceeds int32");
    // Prefill buffers hold the widest step: the idle-wide-prefill width, else the chunk.
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.idle_prefill_width, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
    // Offloaded experts take the VRAM the Program leaves, so a generation Program's text KV pages grow
    // from them on demand (elastic KV): planned in their own region, backed beyond kElasticKvBaseTokens
    // as requests need. (Every capacity plans the region the same way, so the plan stays affine in it.)
    const bool elastic_kv = parameters.model.offloaded_experts() && !plan.causal_scoring;
    LayoutBuilder text_pages;
    out.decoder = qwen3_5::plan_decoder_state(
        builder, qwen3_5::DecoderStateSpec{
                     .full_attention_layers     = config.full_attention_layers,
                     .mtp_layers                = 1,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = dimension(config.attention->num_key_value_heads),
                     .attention_head_dim        = dimension(config.attention->head_dim),
                     .kv_storage                = plan.kv_storage,
                     .lossless                  = plan.execution_schedule.lossless,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                 },
        elastic_kv ? &text_pages : nullptr);
    if (elastic_kv) {
        out.text_kv_region_bytes = text_pages.finish(kArenaAlign, "elastic text KV pages");
        out.text_kv_base_pages   = std::min(physical_pages, page_count(kElasticKvBaseTokens));
    }
    qwen3_5::StateImageSpec state_image_spec{
        .linear =
            {
                .layers        = config.linear_attention_layers,
                .conv_channels = (config.gdn ? dimension(config.gdn->conv_channels()) : 0),
                .conv_width  = (config.gdn ? dimension(config.gdn->linear_conv_kernel_dim - 1) : 0),
                .value_heads = (config.gdn ? dimension(config.gdn->linear_num_value_heads) : 0),
                .value_head_dim = (config.gdn ? dimension(config.gdn->linear_value_head_dim) : 0),
                .key_head_dim   = (config.gdn ? dimension(config.gdn->linear_key_head_dim) : 0),
                .slot_count     = state_image_slots,
                .conv_dtype     = DType::BF16,
            },
        .hidden = dimension(config.hidden_size),
    };
    // Only the prefix cache's Host tier holds the Host form, so without it the exact form stays.
    if (plan.context_cache.enabled) {
        switch (plan.context_cache.hybrid.host_state_precision) {
        case HybridHostStatePrecision::Fp32:
            state_image_spec.recurrent_host = qwen3_5::RecurrentHostFormat::Fp32;
            break;
        case HybridHostStatePrecision::Fp24:
            state_image_spec.recurrent_host = qwen3_5::RecurrentHostFormat::Fp24;
            break;
        case HybridHostStatePrecision::Bf16:
            state_image_spec.recurrent_host = qwen3_5::RecurrentHostFormat::Bf16;
            break;
        }
    }
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (plan.features.masked_draft()) {
            state_image_spec.dflash_local = qwen3_5::DFlashLocalStateSpec{
                .layers   = draft->local_layer_count(),
                .capacity = draft->sliding_window.value_or(0),
                .kv_heads = dimension(draft->attention.num_key_value_heads),
                .head_dim = dimension(draft->attention.head_dim),
            };
        }
    }
    if (kv_storage_has_exact_window(plan.kv_storage)) {
        state_image_spec.kv_window = qwen3_5::KVWindowStateSpec{
            .layers   = config.full_attention_layers + (plan.features.mtp() ? 1U : 0U),
            .kv_heads = dimension(config.attention->num_key_value_heads),
        };
    }
    out.state_images = qwen3_5::plan_state_image_device_pool(builder, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder,
            GdnReplayRecordSpec{
                .layers          = dimension(config.linear_attention_layers),
                .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                .conv_channels   = (config.gdn ? dimension(config.gdn->conv_channels()) : 0),
                .qk_heads        = (config.gdn ? dimension(config.gdn->linear_num_key_heads) : 0),
                .value_heads     = (config.gdn ? dimension(config.gdn->linear_num_value_heads) : 0),
                .key_dim         = (config.gdn ? dimension(config.gdn->linear_key_head_dim) : 0),
                .value_dim       = (config.gdn ? dimension(config.gdn->linear_value_head_dim) : 0),
            });
    }
    {
        const auto* draft =
            parameters.model.config().draft ? &*parameters.model.config().draft : nullptr;
        if (plan.features.masked_draft()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            if (draft->full_layer_count() != 0) {
                const PagedKVStorageLayout full_storage = paged_kv_storage_layout(
                    KvCacheStorage::BFloat16, dimension(draft->attention.head_dim));
                KVPageGeometry full_geometry{
                    .page_tokens        = kPagedKVPageSize,
                    .device_plane_order = PagedKVPlaneOrder::HeadMajor,
                    .planes =
                        {
                            {full_storage.key.data_dtype, full_storage.key.data_leading_extent,
                             dimension(draft->attention.num_key_value_heads), 256},
                            {full_storage.value.data_dtype, full_storage.value.data_leading_extent,
                             dimension(draft->attention.num_key_value_heads), 256},
                        },
                };
                const auto planes = full_geometry.planes;
                for (std::uint32_t layer = 1; layer < draft->full_layer_count(); ++layer) {
                    full_geometry.planes.insert(full_geometry.planes.end(), planes.begin(),
                                                planes.end());
                }
                dflash.full = qwen3_5::PagedKVCacheLayout{
                    .pages = plan_device_kv_page_pool(
                        builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages,
                                                      .geometry = std::move(full_geometry)}),
                    .execution_tables = plan_kv_execution_tables(
                        builder,
                        KVExecutionTableSpec{
                            .logical_page_capacity = logical_pages,
                            .table_rows = static_cast<std::int32_t>(plan.max_concurrency),
                        }),
                    .layers        = draft->full_layer_count(),
                    .max_context   = plan.capacity,
                    .kv_heads      = dimension(draft->attention.num_key_value_heads),
                    .layer_storage = full_storage,
                };
            }
            dflash.prefill_features = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                 effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(
                builder, DType::BF16,
                {dimension(config.hidden_size * std::uint64_t(draft->target_layer_ids.size())),
                  static_cast<std::int32_t>(plan.draft_window + 1U),
                  static_cast<std::int32_t>(plan.max_concurrency)},
                "DFlash pending target features");
        }
    }

    out.round = qwen3_5::begin_round_state_layout(
        builder, qwen3_5::RoundStateSpec{.hidden         = dimension(config.hidden_size),
                                         .output_rows    = dimension(config.vocab_size),
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .backend        = plan.speculative_backend,
                                         .causal_scoring = plan.causal_scoring,
                                         .fp32_logits    = plan.execution_schedule.lossless,
                                         .block_drafts   = plan_has_split(plan) ? plan.neural_draft_window : 0U});
    out.prefill_hidden =
        add_tensor(builder, DType::BF16, {dimension(config.hidden_size), effective_prefill_chunk},
                   "step prefill hidden");
    if (plan.causal_scoring) {
        out.score_hidden =
            add_tensor(builder, DType::BF16,
                       {dimension(config.hidden_size), static_cast<std::int32_t>(kCausalScoreTile)},
                       "causal score hidden staging");
    }
    qwen3_5::complete_round_state_layout(builder, out.round);
    if (!plan.causal_scoring) {
        out.grammar_masks =
            add_tensor(builder, DType::I32,
                       {dimension((parameters.model.resources().public_token_count + 31) / 32),
                        static_cast<std::int32_t>(plan.draft_window + 1),
                        static_cast<std::int32_t>(plan.max_concurrency)},
                       "grammar token masks");
        out.token_counts        = add_tensor(builder, DType::I32,
                                             {dimension(parameters.model.resources().public_token_count),
                                              static_cast<std::int32_t>(plan.max_concurrency)},
                                             "sampling token counts");
        const auto config_words = static_cast<std::int32_t>(
            (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
        out.sampling_config = add_tensor(
            builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
            "sampling config");
    }
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const DeviceExecutionView device_execution{nullptr, plan.multiprocessor_count};
    const auto& parameters = *plan.parameters;
    const auto& config     = parameters.model.config().text;

    // Prefill passes are as wide as the widest step: the idle-wide-prefill width, else the chunk.
    const std::uint32_t chunk_u32 = std::min(plan.idle_prefill_width, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    // Offloaded experts (122B design §10): a pass wider than kOffloadedMoeStaticColumns runs on a
    // workspace the expert cache lends for the whole pass (ProgramOffload::begin_pass), so the
    // Program's own workspace plans prefill to that width and *_prefill_wide records a full pass's.
    // The lossless route's layers keep their own path (execution/lossless.cpp) and lend nothing.
    const bool lent_prefill =
        !plan.execution_schedule.lossless && chunk > execution::kOffloadedMoeStaticColumns &&
        std::holds_alternative<execution::OffloadedMoeParameters>(parameters.text.layers.front().ffn);
    const std::int32_t prefill_width = lent_prefill ? execution::kOffloadedMoeStaticColumns : chunk;
    // Every launch takes the schedule's fast-path permission, which sizes partials (see
    // execution/text.cpp).
    const ops::CausalAttentionExecutionEnvelope text_envelope =
        plan.execution_schedule.apply({1, plan.capacity});
    // Prefill chunks run the selected prompt kernel, which may split keys into workspace.
    const ops::CausalAttentionExecutionEnvelope prefill_envelope =
        plan.execution_schedule.envelope(1, plan.capacity);
    // Target verification only: the drafters' attention runs text_envelope, since proposals need
    // no column invariance, and its workspace must be planned with the envelope it runs.
    const ops::CausalAttentionExecutionEnvelope verify_envelope =
        plan.execution_schedule.verify({1, plan.capacity});

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace::text_prefill_roots(layout, config, tokens, plan.features.vision ? 3 : 0,
                                            plan.features.vision ? tokens : 0);
    };
    const auto linear_scratch = [&](WorkspaceLayoutBuilder& layout,
                                    const execution::LinearParameters& p, int first, int last) {
        scratch(layout, ops::linear_workspace_capacity_bytes(p.weight.qtype, p.weight.n, p.weight.k,
                                                             p.policy, first, last));
    };
    const auto add_scratch = [&](WorkspaceLayoutBuilder& layout,
                                 const execution::LinearParameters& p, int first, int last,
                                 bool wide_verification = false) {
        // A round's narrower families may resolve the ordinary policy at the same columns.
        const auto bytes = [&](bool wide) {
            return ops::linear_add_workspace_capacity_bytes(
                p.weight.qtype, p.weight.n, p.weight.k,
                execution::residual_projection_policy(p, wide), first, last);
        };
        scratch(layout, wide_verification ? std::max(bytes(true), bytes(false)) : bytes(false));
    };
    const auto target_layers = [&](WorkspaceLayoutBuilder& layout,
                                   const execution::TextParameters& text, std::int32_t first,
                                   std::int32_t last, TextPhase phase, GdnWorkspacePath path,
                                   std::int32_t batch_size, std::int32_t min_width,
                                   std::int32_t max_width,
                                   ops::CausalAttentionExecutionEnvelope envelope) {
        const bool wide_verification = wide_residual_verification(phase, min_width, max_width);
        for (const auto& block : text.layers) {
            {
                auto stage = layout.scope();
                if (const auto* attention =
                        std::get_if<execution::AttentionParameters>(&block.mixer)) {
                    (void)workspace::text_attention_projection(layout, config, last);
                    scratch(layout, execution::attention_projection_workspace_bytes(*attention,
                                                                                    first, last));
                    (void)workspace::text_attention_results(layout, config, last);
                    scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                        {dimension(config.attention->head_dim),
                                         dimension(config.attention->num_attention_heads),
                                         dimension(config.attention->num_key_value_heads)},
                                        plan.kv_storage, envelope, batch_size, min_width, max_width,
                                        device_execution));
                    add_scratch(layout, attention->output, first, last, wide_verification);
                } else {
                    const auto& gdn = std::get<execution::GdnParameters>(block.mixer);
                    const auto* control = std::get_if<execution::LinearParameters>(&gdn.control);
                    (void)workspace::gdn_control(layout, config, last);
                    scratch(layout, ops::gdn_norm_gating_proj_workspace_capacity_bytes(
                                        dimension(config.gdn->linear_num_value_heads),
                                        dimension(config.hidden_size),
                                        control != nullptr ? control->policy
                                                           : ops::LinearPolicy::A16Only,
                                        first, last));
                    (void)workspace::gdn_projection(layout, config, last);
                    if (path == GdnWorkspacePath::Snapshot) {
                        scratch(layout, execution::gdn_snapshot_workspace_bytes(
                                            gdn, *config.gdn, batch_size, min_width, max_width));
                    } else if (path == GdnWorkspacePath::ReplayRecord) {
                        scratch(layout, execution::gdn_record_workspace_bytes(
                                            gdn, *config.gdn, batch_size, min_width, max_width,
                                            plan_has_tree(plan)));
                    } else {
                        (void)workspace::gdn_prefill_conv(layout, config, last);
                        scratch(layout,
                                execution::gdn_projection_workspace_bytes(gdn, first, last));
                    }
                    (void)workspace::gdn_recurrent_output(layout, config, last);
                    if (path == GdnWorkspacePath::Prefill) {
                        scratch(layout,
                                ops::gated_delta_net_workspace_capacity_bytes(
                                    dimension(config.gdn->linear_num_key_heads),
                                    dimension(config.gdn->linear_num_value_heads), first, last));
                    }
                    (void)workspace::gdn_normalized_output(layout, config, last);
                    add_scratch(layout, gdn.output, first, last, wide_verification);
                }
            }
            auto stage = layout.scope();
            (void)workspace::post_mixer_hidden(layout, config, last);
            const ops::LinearRoutes ffn_routes = plan.execution_schedule.linear_routes;
            scratch(layout,
                    std::max(execution::ffn_workspace_bytes(block.ffn, first, last, false,
                                                            wide_verification, ffn_routes),
                             execution::ffn_workspace_bytes(block.ffn, first, last, false, false,
                                                            ffn_routes)));
        }
        if (!plan.causal_scoring) { linear_scratch(layout, text.output_head, first, last); }
    };
    // Decode and verification calls run Parameters::decode_text: the column-invariant copy unless
    // batched_columns, which single-column-decode grants to one-column calls only, so a family that
    // can run one column also sizes the batched route there.
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width,
                                 ops::CausalAttentionExecutionEnvelope envelope) {
        if (phase != TextPhase::Verify) {
            target_layers(layout, parameters.text, first, last, phase, path, batch_size, min_width,
                          max_width, envelope);
            return;
        }
        const auto& schedule = plan.execution_schedule;
        const execution::DecodePrecision precision{schedule.a8_decode, schedule.a4_decode,
                                                   schedule.a8_residual_decode,
                                                   schedule.a4_residual_decode};
        target_layers(layout, parameters.decode_text(!schedule.batched_columns(last), precision), first,
                      last, phase, path, batch_size, min_width, max_width, envelope);
        if (first == 1 && schedule.batched_columns(1) != schedule.batched_columns(last)) {
            target_layers(layout, parameters.decode_text(false, precision), 1, 1, phase, path,
                          batch_size, min_width, max_width, envelope);
        }
    };
    // The drafter the MTP sizing below plans: its own routes (the prompt's drafts), or its
    // column-invariant copy (Parameters::invariant_mtp), which the exact route's batched draft
    // calls run.
    const execution::MtpParameters* drafter = parameters.mtp ? &*parameters.mtp : nullptr;
    const execution::LinearParameters* drafter_proposal =
        parameters.proposal ? &parameters.proposal->head : nullptr;
    const auto mtp_post_mixer = [&](WorkspaceLayoutBuilder& layout, int first, int last) {
        linear_scratch(layout, drafter->output, first, last);
        scratch(layout, execution::ffn_workspace_bytes(drafter->ffn, first, last, true));
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, dimension(parameters.proposal->rows), columns);
            linear_scratch(layout, *drafter_proposal, columns, columns);
        } else {
            linear_scratch(layout, drafter->output_head, columns, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace::mtp_stem(layout, config, tokens, !preembedded);
        linear_scratch(layout, drafter->input_projection, 1, tokens);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace::mtp_attention_projection(layout, config, tokens);
        scratch(layout, execution::mtp_projection_workspace_bytes(drafter->projection, tokens,
                                                                  tokens));
        (void)workspace::mtp_attention_results(layout, config, tokens);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, envelope, 1, tokens, tokens, device_execution));
        (void)workspace::mtp_post_attention(layout, config, tokens);
        mtp_post_mixer(layout, tokens, tokens);
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
            scratch(layout, execution::mtp_kv_workspace_bytes(drafter->projection,
                                                              *config.attention, first, last));
            matrix(layout, DType::BF16, dimension(config.attention->key_width()), last);
        }
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, execution::mtp_query_gate_workspace_bytes(drafter->projection,
                                                                  *config.attention, 1, 1));
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, dimension(config.attention->query_width()), 1);
        scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                            {dimension(config.attention->head_dim),
                             dimension(config.attention->num_attention_heads),
                             dimension(config.attention->num_key_value_heads)},
                            plan.kv_storage, text_envelope, 1, 1, 1, device_execution));
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        matrix(layout, DType::BF16, dimension(config.hidden_size), 1);
        mtp_post_mixer(layout, 1, 1);
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    // A text prefill pass of `width` columns (its roots, layers, head and first-token sampling).
    const auto plan_text_prefill = [&](std::int32_t width) {
        WorkspaceLayoutBuilder text_prefill;
        text_common_root(text_prefill, width);
        // The lossless route's cache form (generation): its sequences and key capacity size the attention.
        execution::LosslessCache lossless_shape{.sequences    = 1,
                                                .key_capacity = plan.execution_schedule.lossless_keys};
        const execution::LosslessCache* lossless_cache =
            plan.execution_schedule.lossless_keys != 0 ? &lossless_shape : nullptr;
        if (plan.execution_schedule.lossless) {
            // The chunk's rotary table and (cache form) state slots, then the widest layer
            // (execution/lossless.cpp's own allocations), the hidden output and the head's FP32 logits.
            (void)text_prefill.alloc_bytes(execution::lossless_rope_window_bytes(config, width));
            if (lossless_cache != nullptr) { matrix(text_prefill, DType::I32, 2, 1); }
            for (std::size_t layer = 0; layer < parameters.text.layers.size(); ++layer) {
                scratch(text_prefill, execution::lossless_layer_workspace_bytes(
                                          parameters.text.layers[layer], config, config.layer_types[layer], width,
                                          lossless_cache, plan.execution_schedule.lossless_design));
            }
            matrix(text_prefill, DType::BF16, dimension(config.hidden_size), width);
            scratch(text_prefill, execution::lossless_head_workspace_bytes(parameters.text.output_head.weight, 1));
        } else {
            target_body(text_prefill, 1, width, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                        1, width, prefill_envelope);
        }
        if (!plan.causal_scoring) {
            scratch(text_prefill,
                    ops::sampling_workspace_capacity_bytes(
                        dimension(parameters.model.resources().public_token_count), 1, 1));
        }
        return finish(text_prefill);
    };
    out.text_prefill      = plan_text_prefill(prefill_width);
    out.text_prefill_wide = lent_prefill ? plan_text_prefill(chunk) : 0;

    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder causal_score;
        // The lossless route's logits are FP32.
        matrix(causal_score, plan.execution_schedule.lossless ? DType::FP32 : DType::BF16,
               dimension(config.vocab_size), static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::I32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::FP32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        // Top tokens and candidates: ids and log-probabilities of each.
        for (const DType dtype : {DType::I32, DType::FP32, DType::I32, DType::FP32}) {
            matrix(causal_score, dtype, static_cast<std::int32_t>(kMaximumScoreTopTokens),
                   static_cast<std::int32_t>(kCausalScoreTile));
        }
        if (plan.execution_schedule.lossless) {
            scratch(causal_score, execution::lossless_head_workspace_bytes(
                                      parameters.text.output_head.weight, static_cast<std::int32_t>(kCausalScoreTile)));
        } else {
            linear_scratch(causal_score, parameters.text.output_head, 1, kCausalScoreTile);
        }
        out.causal_score = finish(causal_score);
    }

    if (!plan.causal_scoring) {
        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            WorkspaceLayoutBuilder ordinary;
            matrix(ordinary, DType::BF16, dimension(config.hidden_size), batch);
            if (plan.execution_schedule.lossless) {
                // TextContext::ordinary_decode_batch's lossless allocations: the rotary table, the widest
                // layer of the cache form over `batch` sequences, the head.
                (void)ordinary.alloc_bytes(execution::lossless_rope_window_bytes(config, batch));
                const execution::LosslessCache shape{.sequences    = batch,
                                                     .key_capacity = plan.execution_schedule.lossless_keys};
                for (std::size_t layer = 0; layer < parameters.text.layers.size(); ++layer) {
                    scratch(ordinary, execution::lossless_layer_workspace_bytes(
                                          parameters.text.layers[layer], config, config.layer_types[layer], batch,
                                          &shape));
                }
                scratch(ordinary,
                        execution::lossless_head_workspace_bytes(parameters.text.output_head.weight, batch));
            } else {
                target_body(ordinary, batch, batch, qwen3_5::TextPhase::Verify,
                            GdnWorkspacePath::Snapshot, batch, 1, 1, text_envelope);
            }
            scratch(ordinary,
                    ops::sampling_workspace_capacity_bytes(
                        dimension(parameters.model.resources().public_token_count), batch, batch));
            out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
        }
    }

    if (plan.features.mtp()) {
        const auto plan_mtp_prefill = [&](std::int32_t width) {
            WorkspaceLayoutBuilder mtp_prefill;
            text_common_root(mtp_prefill, width);
            target_body(mtp_prefill, 1, width, qwen3_5::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                        1, 1, width, prefill_envelope);
            matrix(mtp_prefill, DType::I32, 1, width);
            if (plan.features.vision) {
                matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), width);
                (void)workspace::visual_scatter_indices(mtp_prefill, width);
            }
            mtp_prefill_chunk(mtp_prefill, 1, width, plan.features.vision);
            for (std::int32_t i = 1; i < drafts; ++i) {
                matrix(mtp_prefill, DType::BF16, dimension(config.hidden_size), 1);
                mtp_full_call(mtp_prefill, 1, text_envelope, true);
            }
            return finish(mtp_prefill);
        };
        out.mtp_prefill      = plan_mtp_prefill(prefill_width);
        out.mtp_prefill_wide = lent_prefill ? plan_mtp_prefill(chunk) : 0;

        // A round's draft calls run either drafter copy: each is sized.
        std::vector<std::pair<const execution::MtpParameters*, const execution::LinearParameters*>>
            drafters{{drafter, drafter_proposal}};
        if (plan.execution_schedule.column_invariant_decode && parameters.invariant_mtp) {
            drafters.emplace_back(&*parameters.invariant_mtp,
                                  parameters.invariant_proposal_head
                                      ? &*parameters.invariant_proposal_head
                                      : nullptr);
        }
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            dimension(parameters.model.resources().public_token_count), drafts, drafts, 1, 1);
        out.mtp_round = accept;
        for (const auto& [mtp, proposal] : drafters) {
            drafter          = mtp;
            drafter_proposal = proposal;
            WorkspaceLayoutBuilder mtp_batch;
            mtp_full_call(mtp_batch, verify, text_envelope, false);
            WorkspaceLayoutBuilder mtp_ar;
            mtp_full_call(mtp_ar, 1, text_envelope, true);
            WorkspaceLayoutBuilder mtp_align;
            mtp_full_call(mtp_align, 1, text_envelope, false);
            WorkspaceLayoutBuilder mtp_proposal;
            proposal_scratch(mtp_proposal, 1);
            out.mtp_round =
                std::max({out.mtp_round, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
            out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));
        }

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
            target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, verify_envelope);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    dimension(parameters.model.resources().public_token_count), drafts, drafts,
                    batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), batch_accept});

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace::mtp_attention_projection(layout, config, tokens);
                scratch(layout, execution::mtp_projection_workspace_bytes(drafter->projection,
                                                                          tokens, tokens));
                (void)workspace::mtp_attention_results(layout, config, tokens);
                scratch(layout, ops::causal_softmax_attention_workspace_capacity_bytes(
                                    {dimension(config.attention->head_dim),
                                     dimension(config.attention->num_attention_heads),
                                     dimension(config.attention->num_key_value_heads)},
                                    plan.kv_storage, text_envelope, batch, width, width,
                                    device_execution));
                (void)workspace::mtp_post_attention(layout, config, tokens);
                mtp_post_mixer(layout, tokens, tokens);
            };
            for (const auto& [mtp, proposal_head] : drafters) {
                drafter          = mtp;
                drafter_proposal = proposal_head;
                WorkspaceLayoutBuilder alignment;
                mtp_decode_core(alignment, verify);
                WorkspaceLayoutBuilder ar;
                mtp_decode_core(ar, 1);
                WorkspaceLayoutBuilder proposal;
                proposal_scratch(proposal, batch);
                out.mtp_round = std::max(
                    {out.mtp_round, finish(alignment), finish(ar), finish(proposal)});
            }
        }
        drafter          = &*parameters.mtp;
        drafter_proposal = parameters.proposal ? &parameters.proposal->head : nullptr;
    }

    if (plan.features.masked_draft()) {
        {
            const auto* draft                  = &*parameters.model.config().draft;
            const auto dflash_context_capacity = [&](std::int32_t width, std::int32_t batch,
                                                     bool compact_input) {
                const auto tokens = width * batch;
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16,
                           dimension(config.hidden_size *
                                     std::uint64_t(draft->target_layer_ids.size())),
                           tokens);
                }
                if (draft->dflash2.has_value()) {
                    const auto local_width =
                        std::min(width, dimension(draft->sliding_window.value_or(0)));
                    (void)workspace::dflash_context(layout, config, *draft, local_width * batch);
                    linear_scratch(layout, parameters.draft->feature_projection,
                                   local_width * batch, local_width * batch);
                    scratch(layout, ops::context_kv_materialize_workspace_capacity_bytes(
                                        static_cast<std::int32_t>(draft->num_hidden_layers), batch,
                                        local_width, local_width));
                    return finish(layout);
                }
                (void)workspace::dflash_context(layout, config, *draft, tokens);
                linear_scratch(layout, parameters.draft->feature_projection, tokens, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace::dflash_context_layer(layout, config, *draft, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens          = width * batch;
                const std::int32_t proposal_drafts = width - 1;
                matrix(layout, DType::BF16, dimension(config.hidden_size), tokens);
                if (draft->dflash2.has_value()) {
                    const auto prepare = [&] {
                        (void)workspace::dflash2_branch(layout, config, *draft, width, batch);
                        scratch(layout,
                                ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                    dimension(config.hidden_size), width, width, batch, batch));
                    };
                    {
                        auto attention = layout.scope();
                        prepare();
                        matrix(layout, DType::BF16, dimension(draft->attention.query_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.key_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.key_width()),
                               tokens);
                        matrix(layout, DType::BF16, dimension(draft->attention.query_width()),
                               tokens);
                        scratch(layout, ops::sliding_window_attention_workspace_capacity_bytes(
                                            {dimension(draft->attention.head_dim),
                                             dimension(draft->attention.num_attention_heads),
                                             dimension(draft->attention.num_key_value_heads)},
                                            dimension(draft->sliding_window.value_or(0)),
                                            {0, plan.capacity}, width, width, batch));
                        scratch(layout,
                                ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                    dimension(config.hidden_size), dimension(draft->attention.query_width()), width, width, batch,
                                    batch));
                    }
                    {
                        auto mlp = layout.scope();
                        prepare();
                        matrix(layout, DType::BF16, dimension(draft->intermediate_size), tokens);
                        for (const auto& block : parameters.draft->layers) {
                            const auto& p = block.mlp.gate_up;
                            scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                                p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                                tokens, tokens));
                        }
                        scratch(
                            layout,
                            ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                                dimension(config.hidden_size), dimension(draft->intermediate_size), width, width, batch, batch));
                    }
                    const auto mask_columns = proposal_drafts * batch;
                    matrix(layout, DType::BF16, dimension(config.hidden_size), mask_columns);
                    // A proposal landing in another (wider) frame keeps the lattice's candidates
                    // apart from the frame's column candidates.
                    matrix(layout, DType::I32, dimension(draft->dflash2->selector_top_k),
                           mask_columns);
                    matrix(layout, DType::FP32, dimension(draft->dflash2->selector_top_k),
                           mask_columns);
                    const auto& head = plan.proposal_head == ProposalHead::Optimized
                                           ? parameters.proposal->head
                                           : parameters.draft->output_head;
                    scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                                        head.weight.qtype, head.weight.n, head.weight.k,
                                        mask_columns, mask_columns));
                    matrix(layout, DType::BF16, dimension(draft->dflash2->selector_rank),
                           mask_columns);
                    linear_scratch(layout, parameters.draft->selector->hidden_projection,
                                   mask_columns, mask_columns);
                    // Staged chain drafts and laws before their copy into another frame.
                    matrix(layout, DType::I32, 1, mask_columns);
                    matrix(layout, DType::FP32, dimension(draft->dflash2->selector_top_k),
                           mask_columns);
                    scratch(layout, ops::candidate_selector_path_workspace_capacity_bytes(
                                        proposal_drafts, proposal_drafts, batch, batch));
                    if (plan_has_tree(plan)) {
                        // The tree build always runs over the full lattice.
                        scratch(layout, ops::candidate_selector_tree_workspace_capacity_bytes(
                                            proposal_drafts, batch, batch));
                    }
                    return finish(layout);
                }
                {
                    auto attention = layout.scope();
                    (void)workspace::dflash_attention(layout, config, *draft, tokens);
                    scratch(layout,
                            std::max(ops::sliding_window_attention_workspace_capacity_bytes(
                                         {dimension(draft->attention.head_dim),
                                          dimension(draft->attention.num_attention_heads),
                                          dimension(draft->attention.num_key_value_heads)},
                                         dimension(draft->sliding_window.value_or(0)),
                                         {0, plan.capacity}, width, width, batch),
                                     ops::context_softmax_attention_workspace_capacity_bytes(
                                         {dimension(draft->attention.head_dim),
                                          dimension(draft->attention.num_attention_heads),
                                          dimension(draft->attention.num_key_value_heads)},
                                         {0, plan.capacity}, width, width, batch)));
                    for (const auto& block : parameters.draft->layers) {
                        add_scratch(layout, block.output, tokens, tokens);
                    }
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace::dflash_mlp(layout, config, *draft, tokens);
                    for (const auto& block : parameters.draft->layers) {
                        const auto& p = block.mlp.gate_up;
                        scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                            p.weight.qtype, p.weight.n, p.weight.k, p.policy,
                                            tokens, tokens));
                    }
                    for (const auto& block : parameters.draft->layers) {
                        add_scratch(layout, block.mlp.down, tokens, tokens);
                    }
                }
                matrix(layout, DType::BF16, dimension(config.hidden_size), proposal_drafts * batch);
                matrix(layout, DType::BF16, dimension(config.hidden_size), proposal_drafts * batch);
                // Staged drafts before their copy into another round's frame.
                matrix(layout, DType::I32, 1, proposal_drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, dimension(parameters.proposal->rows),
                           proposal_drafts * batch);
                } else {
                    matrix(layout, DType::BF16, dimension(config.vocab_size),
                           proposal_drafts * batch);
                }
                const auto& head = plan.proposal_head == ProposalHead::Optimized
                                       ? parameters.proposal->head
                                       : parameters.draft->output_head;
                linear_scratch(layout, head, proposal_drafts * batch, proposal_drafts * batch);
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, 1, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, dimension(config.hidden_size), aggregate);
                target_body(target, aggregate, aggregate, qwen3_5::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, verify_envelope);
                std::size_t accept =
                    draft->dflash2.has_value()
                        ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
                              dimension(parameters.model.resources().public_token_count), {false},
                              drafts, drafts, batch, batch)
                        : ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                              dimension(parameters.model.resources().public_token_count), drafts,
                              drafts, batch, batch);
                for (const SpeculativeRoundShape& shape : plan.round_shapes) {
                    if (shape.kind != SpeculativeRoundKind::Tree) { continue; }
                    accept = std::max(
                        accept,
                        ops::speculative_accept_sparse_tree_workspace_capacity_bytes(
                            dimension(parameters.model.resources().public_token_count),
                            static_cast<std::int32_t>(shape.verify_drafts + 1U), batch, batch));
                }
                const std::size_t proposal = dflash_proposal_capacity(
                    static_cast<std::int32_t>(plan.neural_draft_window) + 1, batch);
                out.dflash_round =
                    std::max({out.dflash_round, finish(target), accept,
                              dflash_context_capacity(verify, batch, true), proposal});
            }
        }
    }

    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.causal_score});
    out.capacity = out.general_capacity;
    if (plan.features.vision) {
        const std::uint64_t capped =
            std::min<std::uint64_t>(plan.capacity, kMaximumVisionItemTokens);
        const std::uint32_t merged = static_cast<std::uint32_t>(
            std::min(capped, static_cast<std::uint64_t>(plan.features.vision_max_merged_tokens)));
        out.vision = execution::VisionContext::plan_workspace(
            *parameters.model.config().vision, *parameters.vision, merged, out.general_capacity);
        if (plan.features.overlay_vision()) {
            // Overlay: the resident reservation carries only the output handoff; the encode
            // workspace comes from the evicted weight tail during each overlay window.
            out.vision->capacity_bytes =
                out.vision->handoff_offset_bytes + out.vision->handoff_capacity_bytes;
        }
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

void validate_target_options(const execution::Parameters& parameters, DeviceContext& device,
                             const EngineOptions& options) {
    if (!parameters.model.config().text.attention ||
        parameters.model.config().text.full_attention_layers == 0) {
        throw std::invalid_argument("Qwen3.5 Program requires at least one full-attention layer");
    }
    // The RAM ledger, not the options, decides whether offloaded banks stay in the artifact (the SSD
    // tier): the loaded model's choice stands.
    auto expected           = models::load_options(options, Architecture::Qwen3_5);
    expected.stream_experts = parameters.model.options().stream_experts;
    if (parameters.model.options() != expected) {
        throw std::invalid_argument(
            "loaded components do not match the requested execution options");
    }
    if (parameters.draft &&
        options.max_context > rope_context_ceiling(
            parameters.model.config().draft->max_position_embeddings, options.rope_yarn_factor)) {
        throw std::invalid_argument("max_context exceeds the selected draft position capacity");
    }
    if (options.max_context == 0 ||
        options.max_context > rope_context_ceiling(
            parameters.model.config().text.max_position_embeddings, options.rope_yarn_factor)) {
        throw std::invalid_argument("max_context exceeds the configured position capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    const std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    const std::uint64_t maximum_pages64 =
        maximum_main_page_groups(options.max_concurrency, logical_pages, options.context_cache);
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens < options.max_context) {
            throw std::invalid_argument("kv_capacity must be at least max_context");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash:
    case SpeculativeBackend::DFlash2:
        if (!parameters.draft || (options.speculative.backend == SpeculativeBackend::DFlash2) !=
                                     parameters.model.config().draft->dflash2.has_value()) {
            throw std::invalid_argument(
                "selected masked draft backend is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 || options.speculative.draft_tokens > 15) {
            throw std::invalid_argument("masked draft window must be in [1,15]");
        }
        break;
    }
    const bool tree_table = tree_table_enabled(options.speculative.draft_tree_nodes);
    if (tree_table || options.speculative.draft_tree_auto) {
        if (options.speculative.backend != SpeculativeBackend::DFlash2) {
            throw std::invalid_argument("DFlash2 tree verification requires the DFlash2 backend");
        }
        if (tree_table && options.speculative.draft_tree_auto) {
            throw std::invalid_argument("automatic DFlash2 tree widths take no fixed table");
        }
        std::vector<std::uint32_t> widths(options.speculative.draft_tree_nodes.begin(),
                                          options.speculative.draft_tree_nodes.end());
        if (options.speculative.draft_tree_auto) {
            const auto automatic = draft_tree_auto_widths(options.speculative.draft_tokens);
            widths.assign(automatic.begin(), automatic.end());
        }
        for (const std::uint32_t nodes : widths) {
            if (nodes == 0) { continue; }
            ops::validate_speculative_tree_shape(
                {static_cast<std::int32_t>(nodes),
                 static_cast<std::int32_t>(options.speculative.draft_tokens),
                 static_cast<std::int32_t>(options.speculative.draft_tree_paths)});
        }
        // A fixed table needs trees; automatic mode falls back to chains (resolve_tree_widths).
        if (tree_table && !tree_verification_supported(parameters)) {
            throw std::invalid_argument("DFlash2 tree verification requires single FP8 or NVFP4 "
                                        "GDN input projections");
        }
    }
    if (device.compute_capability() != 120) {
        throw std::invalid_argument("Qwen3.5 family runtime requires compute capability 12.0");
    }
    if (options.speculative.ngram_draft_tokens != 0 &&
        ((options.speculative.backend != SpeculativeBackend::DFlash2 &&
          options.speculative.backend != SpeculativeBackend::DFlash &&
          options.speculative.backend != SpeculativeBackend::Mtp) ||
         options.speculative.ngram_draft_tokens > 63 || options.speculative.ngram_min_match < 4 ||
         options.speculative.ngram_min_match > 64)) {
        throw std::invalid_argument("ngram requires MTP/DFlash/DFlash2, K1..63 and match 4..64");
    }
    if (options.speculative.ngram_batch_draft_tokens > options.speculative.ngram_draft_tokens) {
        throw std::invalid_argument("the batched ngram window must lie within the ngram window");
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                  = std::make_unique<SequencePlanImpl>();
    impl->parameters           = inputs.parameters;
    impl->capacity             = inputs.capacity;
    impl->main_page_groups     = main_page_groups;
    impl->kv_capacity          = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency      = inputs.max_concurrency;
    impl->prefill_chunk        = inputs.prefill_chunk;
    impl->idle_prefill_width   = inputs.idle_prefill_width;
    impl->execution_schedule   = inputs.execution_schedule;
    impl->draft_window         = inputs.draft_window;
    impl->neural_draft_window  = inputs.neural_draft_window;
    impl->round_shapes         = inputs.round_shapes;
    impl->dflash_width_costs   = inputs.dflash_width_costs;
    impl->tree_widths          = inputs.tree_widths;
    impl->draft_tree_paths     = inputs.draft_tree_paths;
    impl->ngram_draft_window   = inputs.ngram_draft_window;
    impl->ngram_batch_window   = inputs.ngram_batch_window;
    impl->ngram_min_match      = inputs.ngram_min_match;
    impl->speculative_backend  = inputs.speculative_backend;
    impl->proposal_head        = inputs.proposal_head;
    impl->features             = inputs.features;
    impl->use_cuda_graph       = inputs.use_cuda_graph;
    impl->causal_scoring       = inputs.causal_scoring;
    impl->device               = inputs.device;
    impl->multiprocessor_count = inputs.multiprocessor_count;
    impl->context_cache        = inputs.context_cache;
    impl->kv_storage           = inputs.kv_storage;
    impl->persistent           = persistent_layout(*impl);
    if (impl->context_cache.enabled) {
        // The whole Host budget is the slab pool that KV blocks and state snapshots share
        // (docs/maintainer/hybrid-prefix-cache-spec.md §5.4); a paused request recovers by
        // Replay. A budget too small for one snapshot is rejected here, before anything is
        // allocated. The backend pool is the MTP KV
        // pool, or the DFlash draft's paged full-attention pool.
        const KVPageGeometry* backend = nullptr;
        if (impl->speculative_backend == SpeculativeBackend::Mtp &&
            impl->persistent.decoder.mtp_kv) {
            backend = &impl->persistent.decoder.mtp_kv->pages.spec.geometry;
        } else if (impl->persistent.dflash && impl->persistent.dflash->full) {
            backend = &impl->persistent.dflash->full->pages.spec.geometry;
        }
        const HybridHostLayout host =
            plan_hybrid_host_layout(impl->persistent.decoder.text_kv.pages.spec.geometry, backend,
                                    impl->persistent.state_images.host.image_bytes);
        if (!impl->context_cache.host_capacity_bytes) {
            impl->context_cache.host_capacity_bytes = kDefaultHybridHostCacheBytes;
        }
        (void)hybrid_host_slabs(host, *impl->context_cache.host_capacity_bytes);
    } else {
        impl->context_cache.host_capacity_bytes = 0U;
    }
    impl->workspace = build_workspace_plan(*impl);
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each topology class, per batch size, of each family prepare_graphs() captures.
        std::uint64_t executables = 0;
        if (impl->speculative_backend == SpeculativeBackend::None) {
            executables = static_cast<std::uint64_t>(
                              graph_topology_classes(ordinary_graph_profiles(impl->capacity))) *
                          impl->max_concurrency;
        } else {
            // Each speculative round family's profiles are captured at the family's own width for
            // every batch size, on the one frame viewed at that width, as a Forward/Finish pair.
            // MTP Finish keeps the Forward profiles; DFlash Finish has one executable per exact B.
            // Every MTP round proposes the next round's drafts at its neural family's draft length
            // (copy rounds at the configured neural depth).
            const bool mtp      = impl->speculative_backend == SpeculativeBackend::Mtp;
            bool split          = false;
            bool split_batches  = false;
            for (const SpeculativeRoundShape& shape : impl->round_shapes) {
                const std::uint32_t ar_depth = mtp_proposal_drafts(shape, impl->neural_draft_window);
                split                        = split || shape.split;
                split_batches                = split_batches || (shape.split && !shape.single_row);
                const auto profiles =
                    mtp           ? mtp_graph_profiles(impl->capacity, shape.verify_drafts, ar_depth)
                    : shape.split ? dflash_verify_graph_profiles(impl->capacity, shape.verify_drafts)
                                  : dflash_graph_profiles(impl->speculative_backend, impl->capacity,
                                                          shape.verify_drafts);
                // A tree family serves only the batch sizes that may verify its width. Chain
                // families serve every batch size: a round with a constrained row verifies the
                // chain whatever the tree table selects. A single-row family serves one row, a
                // batch-only family every other batch size.
                std::uint32_t batch_sizes = 0;
                for (std::uint32_t b = 1; b <= impl->max_concurrency; ++b) {
                    batch_sizes += (shape.kind != SpeculativeRoundKind::Tree ||
                                    impl->tree_widths.tree(b, shape.verify_drafts + 1U)) &&
                                   (!shape.single_row || b == 1U) && (!shape.batch_only || b > 1U);
                }
                const std::uint64_t forward = graph_topology_classes(profiles);
                executables += (forward + (mtp ? forward : 1U)) * batch_sizes;
            }
            // adaptive-dflash's proposal graphs: the drafter's profiles at the whole block, for every
            // batch size a split family serves.
            if (split) {
                executables += static_cast<std::uint64_t>(graph_topology_classes(dflash_graph_profiles(
                                   impl->speculative_backend, impl->capacity, impl->neural_draft_window))) *
                               (split_batches ? impl->max_concurrency : 1U);
            }
        }
        impl->graph_allowance_bytes = checked_add(
            kGraphDriverAllowance,
            checked_mul(kGraphExecutableAllowance, executables, "CUDA Graph executable allowance"),
            "CUDA Graph allowance");
    }

    // An elastic text KV region counts whole: the frames give it back as it grows.
    impl->device_reservation_bytes = checked_add(
        checked_add(checked_add(impl->persistent.bytes, impl->persistent.text_kv_region_bytes, "elastic KV region"),
                    impl->workspace.capacity, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

namespace {
// INT8 KV prefills with the fast prompt kernel unless the original kernel was selected.
bool uses_fast_int8_prefill(const EngineOptions& options) {
    return options.kv_cache == KvCacheStorage::Int8Group64 && !options.original_int8_prefill_kernel;
}

static_assert(std::size_t{kDefaultPrefillSplitWorkspaceMiB} << 20 ==
              ops::kCausalPromptSplitWorkspaceDefaultBytes);

// INT8 and NVFP4 KV prefill with their fast prompt kernels unless the original was selected.
// Prompt P*V runs in FP16 unless the prompt-pv8 fast path is selected, and single-column decode
// keeps 128-key splits unless decode-split64 is (fast_paths.h). The MLP gate/up takes the q4-w4a8
// route at prefill widths when that path is selected.
ExecutionSchedule execution_schedule(const execution::Parameters& parameters,
                                     const EngineOptions& options) {
    // The lossless route: causal scoring takes each window in one pass (its attention reads the window's
    // own keys); generation runs text through bf16 KV (V stored as BF16), without speculation.
    if (options.lossless) {
        if (options.purpose == EnginePurpose::CausalScoring && options.prefill_chunk < options.max_context) {
            throw std::invalid_argument("the Qwen3.5 lossless route scores a window in one pass: its prefill_chunk "
                                        "must cover max_context");
        }
        if (options.purpose == EnginePurpose::Generation &&
            (options.kv_cache != KvCacheStorage::BFloat16 ||
             options.speculative.backend != SpeculativeBackend::None || options.enable_vision)) {
            throw std::invalid_argument("Qwen3.5 lossless generation takes text requests with bf16 KV and no "
                                        "speculation");
        }
        if (options.reference && options.kv_cache != KvCacheStorage::BFloat16) {
            throw std::invalid_argument("the Qwen3.5 design reference scores with bf16 KV (its other KV codecs are "
                                        "not implemented yet)");
        }
    }
    // The exact route is request-invariant (docs/maintainer/model-quality.md, 6.1); each request-coupling
    // fast path gives up one part of it, and each part holds whenever its coupling path is not selected.
    const FastPathSelection fast_paths = select_fast_paths(parameters.model, options);
    return {.fast = uses_fast_int8_prefill(options) ||
                    (options.kv_cache == KvCacheStorage::Nvfp4Group16 &&
                     !options.original_nvfp4_prefill_kernel),
            .pv8                     = fast_paths.selected(FastPath::PromptPv8),
            .split_workspace_bytes   = std::size_t{options.prefill_split_workspace_mib} << 20,
            .decode_split64          = fast_paths.selected(FastPath::DecodeSplit64),
            .column_invariant_decode = fast_paths.invariant(FastPath::BatchedDecode),
            .single_column_decode    = fast_paths.selected(FastPath::SingleColumnDecode),
            .a8_decode               = fast_paths.selected(FastPath::BatchedA8Decode),
            .a4_decode               = fast_paths.selected(FastPath::BatchedA4Decode),
            .a8_residual_decode      = fast_paths.selected(FastPath::BatchedA8ResidualDecode),
            .a4_residual_decode      = fast_paths.selected(FastPath::BatchedA4ResidualDecode),
            .row_invariant_partition = !fast_paths.selected(FastPath::BatchSplitAttention),
            .fixed_speculation       = !fast_paths.selected(FastPath::AdaptiveSpeculation),
            .grid_cache              = !fast_paths.selected(FastPath::FreeCacheResume),
            .held_extent             = !fast_paths.selected(FastPath::Preemption),
            .adaptive_mtp            = fast_paths.selected(FastPath::AdaptiveMtp) &&
                                       fast_paths.selected(FastPath::AdaptiveSpeculation),
            .adaptive_dflash         = fast_paths.selected(FastPath::AdaptiveDFlash) &&
                                       fast_paths.selected(FastPath::AdaptiveSpeculation),
            // Only offloaded experts make an unverified draft column free (the verification's
            // masked padding routes no expert); elsewhere trimming drafts saves nothing.
            .mtp_draft_confidence    = options.speculative.backend == SpeculativeBackend::Mtp &&
                                       parameters.model.offloaded_experts(),
            .linear_routes         = fast_paths.selected(FastPath::Q4W4A8) ? ops::LinearRoutes::Q4W4A8
                                                                             : ops::LinearRoutes::Exact,
            .gdn_arithmetic          = fast_paths.selected(FastPath::GdnF16Accumulate)
                                           ? ops::GdnChunkArithmetic::Fp16Accumulate
                                           : ops::GdnChunkArithmetic::Exact,
            .lossless                = options.lossless,
            .lossless_keys           = options.lossless && options.purpose == EnginePurpose::Generation
                                           ? static_cast<std::int32_t>(options.max_context)
                                           : 0,
            .lossless_design         = {.active = options.reference, .kv = options.kv_cache},
            .gumbel_sampling         = !options.fast_paths || options.lossless};
}
// The fast INT8, NVFP4, VQ2 and K4V2 prompt kernels and the MXFP8 tiled kernel of FP8 and K8V4 KV
// run one 128-row CTA of one query head per SM; BF16 KV and the original INT8 and NVFP4 kernels use
// other tiles.
bool prefill_attention_runs_waves(const EngineOptions& options) {
    switch (options.kv_cache) {
    case KvCacheStorage::Int8Group64: return !options.original_int8_prefill_kernel;
    case KvCacheStorage::Nvfp4Group16: return !options.original_nvfp4_prefill_kernel;
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Fp8KeyNvfp4Value:
    case KvCacheStorage::Vq2:
    case KvCacheStorage::Q4KeyVq2Value: return true;
    case KvCacheStorage::BFloat16: return false;
    }
    return false;
}
} // namespace

// Every chunk but a prompt's last one has the effective width, so with a prompt-attention kernel of
// whole-SM row blocks it is rounded down to whole prompt-attention waves, keeping each full chunk's
// attention free of a partial last wave.
std::uint32_t prefill_step_width(const execution::Parameters& parameters,
                                 const EngineOptions& options, std::uint32_t requested) {
    requested = std::min(requested, options.max_context);
    // The lossless route's attention is its own (no prompt waves): its window form needs the whole
    // window in one pass.
    if (options.lossless || !prefill_attention_runs_waves(options)) { return requested; }
    const auto& attention = *parameters.model.config().text.attention;
    const auto wave = static_cast<std::uint32_t>(ops::causal_softmax_attention_prompt_wave_tokens(
        {static_cast<std::int32_t>(attention.head_dim),
         static_cast<std::int32_t>(attention.num_attention_heads),
         static_cast<std::int32_t>(attention.num_key_value_heads)}));
    return requested < wave ? requested : requested / wave * wave;
}

std::uint32_t effective_prefill_chunk(const execution::Parameters& parameters,
                                      const EngineOptions& options) {
    return prefill_step_width(parameters, options, options.prefill_chunk);
}

std::vector<std::uint32_t> idle_prefill_widths_impl(const execution::Parameters& parameters,
                                                    const EngineOptions& options) {
    if (options.purpose != EnginePurpose::Generation ||
        !select_fast_paths(parameters.model, options)
             .selected(FastPath::IdleWidePrefill)) {
        return {};
    }
    // Each width is taken as the chunk is (prefill_step_width), then kept on the chunk's grid.
    std::array<std::uint32_t, kIdlePrefillRequestedWidths.size()> steps{};
    for (std::size_t i = 0; i < steps.size(); ++i) {
        steps[i] = prefill_step_width(parameters, options, kIdlePrefillRequestedWidths[i]);
    }
    return idle_prefill_width_ladder(effective_prefill_chunk(parameters, options), steps);
}

std::uint32_t resolve_auto_draft_length_impl(const execution::Parameters& parameters,
                                             EngineOptions& options) {
    // A block drafter's automatic ceiling, unless given: the configuration's block.
    if (options.speculative.draft_tokens_auto && is_masked_draft_backend(options.speculative.backend) &&
        options.speculative.draft_tokens == 0) {
        options.speculative.draft_tokens = dflash_auto_ceiling(
            select_fast_paths(parameters.model, options).configuration, options.speculative.backend);
    }
    const ExecutionSchedule schedule = execution_schedule(parameters, options);
    if (!options.speculative.draft_tokens_auto || schedule.adaptive_mtp || schedule.adaptive_dflash) {
        return 0;
    }
    // The policies act only where speculation is adaptive; a qualification run that names one must
    // also name adaptive-speculation.
    for (const FastPath policy : {FastPath::AdaptiveMtp, FastPath::AdaptiveDFlash}) {
        if (std::find(options.fast_path_candidates.begin(), options.fast_path_candidates.end(), policy) !=
            options.fast_path_candidates.end()) {
            throw std::invalid_argument(std::string("--fast-path ") + fast_path_name(policy) +
                                        " acts only where speculation is adaptive: add --fast-path "
                                        "adaptive-speculation");
        }
    }
    const FastPathSelection fast =
        select_fast_paths(parameters.model, options);
    options.speculative.draft_tokens_auto = false;
    options.speculative.draft_tokens =
        std::min(options.speculative.draft_tokens,
                 auto_fixed_draft_tokens(fast.configuration, options.speculative.backend));
    return options.speculative.draft_tokens;
}

std::uint32_t resolve_auto_ngram_impl(const Model& model, EngineOptions& options) {
    SpeculativeOptions& spec = options.speculative;
    if (!spec.ngram_draft_tokens_auto) { return spec.ngram_draft_tokens; }
    const FastPathSelection fast = select_fast_paths(model, options);
    // Fixed speculation verifies every round at the n-gram window (decode_dflash_batch,
    // decode_mtp_batch): auto keeps copies off there.
    const bool adaptive        = fast.selected(FastPath::AdaptiveSpeculation);
    const bool adaptive_drafts = spec.draft_tokens_auto && (fast.selected(FastPath::AdaptiveMtp) ||
                                                            fast.selected(FastPath::AdaptiveDFlash));
    const NgramSettings settings =
        adaptive ? auto_ngram_settings(fast.configuration, spec.backend, adaptive_drafts) : NgramSettings{};
    spec.ngram_draft_tokens_auto  = false;
    spec.ngram_draft_tokens       = settings.window;
    spec.ngram_batch_draft_tokens = settings.window != 0 ? settings.batch_window : 0U;
    if (settings.window != 0) { spec.ngram_min_match = settings.min_match; }
    return settings.window;
}

std::unique_ptr<qwen3_5::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options, std::uint32_t idle_prefill_width) {
    validate_target_options(parameters, device, options);
    if (idle_prefill_width != 0) {
        const std::vector<std::uint32_t> widths = idle_prefill_widths_impl(parameters, options);
        if (std::find(widths.begin(), widths.end(), idle_prefill_width) == widths.end()) {
            throw std::invalid_argument("idle prefill width is not an idle-wide-prefill width of "
                                        "these options");
        }
    }
    TreeWidthPlan tree_widths = resolve_tree_widths(parameters, options);
    // Fixed speculation (the exact route, and any route without adaptive-speculation) verifies the chain at a
    // fixed width every round and never picks a tree (decode_mtp/dflash rounds), so no tree family is
    // planned or captured: its column-invariant attention workspace does not hold the tree route.
    if (execution_schedule(parameters, options).fixed_speculation) { tree_widths = TreeWidthPlan{}; }
    // Fixed speculation verifies every copy round at the n-gram window, which a narrower batched
    // window would leave without a family for its batches.
    if (execution_schedule(parameters, options).fixed_speculation &&
        options.speculative.ngram_batch_draft_tokens != 0 &&
        options.speculative.ngram_batch_draft_tokens != options.speculative.ngram_draft_tokens) {
        throw std::invalid_argument("a batched ngram window needs adaptive speculation");
    }
    const bool adaptive_mtp    = execution_schedule(parameters, options).adaptive_mtp;
    const bool adaptive_dflash = execution_schedule(parameters, options).adaptive_dflash;
    // adaptive-dflash verifies its policy's chain widths. Automatic trees of K + 5 and K + 9 columns
    // on its 15-draft block ran slower than the chain they replace (2026-10-22, NVFP4 27B, one
    // request: 20 and 24 columns -1.0 to -1.9 % by the same-text estimate), and the tree controller
    // times the chain only when the policy verifies the whole block, so automatic widths resolve to
    // chains there.
    if (adaptive_dflash) {
        tree_widths.automatic.clear();
        tree_widths.automatic_max_batch = 0;
    }
    // resolve_auto_draft_length made --draft-tokens auto a fixed length wherever its policy does
    // not act; without it the rounds would silently draft the ceiling.
    if (options.speculative.draft_tokens_auto && !adaptive_mtp && !adaptive_dflash) {
        throw std::logic_error("--draft-tokens auto reached planning without its policy; "
                               "resolve_auto_draft_length was not applied");
    }
    const FastPathConfiguration configuration = select_fast_paths(parameters.model, options).configuration;
    const std::vector<SpeculativeRoundShape> round_shapes = speculative_round_shapes(
        options, tree_widths, adaptive_mtp, adaptive_dflash,
        dflash_batch_drafts(configuration, options.speculative.backend));
    std::uint32_t draft_window = 0;
    for (const SpeculativeRoundShape& shape : round_shapes) {
        draft_window = std::max(draft_window, shape.verify_drafts);
    }
    SequencePlanningInputs inputs{
        .parameters          = &parameters,
        .neural_draft_window = options.speculative.draft_tokens,
        .round_shapes        = round_shapes,
        .dflash_width_costs  = dflash_width_costs(configuration, options.speculative.backend),
        .tree_widths         = std::move(tree_widths),
        .draft_tree_paths    = options.speculative.draft_tree_paths,
        .ngram_draft_window  = options.speculative.ngram_draft_tokens,
        .ngram_batch_window  = options.speculative.ngram_batch_draft_tokens,
        .ngram_min_match     = options.speculative.ngram_min_match,
        .capacity            = options.max_context,
        .max_concurrency     = options.max_concurrency,
        .prefill_chunk       = effective_prefill_chunk(parameters, options),
        .idle_prefill_width  = idle_prefill_width != 0 ? idle_prefill_width
                                                       : effective_prefill_chunk(parameters, options),
        .execution_schedule   = execution_schedule(parameters, options),
        .draft_window         = draft_window,
        .speculative_backend  = options.speculative.backend,
        .kv_storage           = options.kv_cache,
        .proposal_head        = options.speculative.proposal_head,
        // The loaded model's options (validate_target_options checked them against these options): they
        // carry the RAM ledger's tier choice.
        .features             = parameters.model.options(),
        // Lossless decode rounds are captured too: nothing on their path waits for the host, and the
        // attention's launches depend on the round's shape, not on its key bound.
        .use_cuda_graph       = options.use_cuda_graph,
        .causal_scoring       = options.purpose == EnginePurpose::CausalScoring,
        .device               = options.device,
        .multiprocessor_count = device.multiprocessor_count(),
        .context_cache        = options.context_cache,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    const std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    const auto maximum_pages          = static_cast<std::uint32_t>(
        maximum_main_page_groups(inputs.max_concurrency, logical_pages, inputs.context_cache));

    auto planner     = std::make_unique<qwen3_5::detail::SequencePlannerImpl>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.5 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_5::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.5 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.5 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace infernix::models::qwen3_5::detail
