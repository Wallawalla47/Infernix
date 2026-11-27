#pragma once

// Inkling2Loop (Thinking Machines Inkling 2 Mini) instance configuration as recorded by the converter
// (tools/convert/inkling2_loop.py). The mathematics is fixed by this implementation
// (docs/maintainer/inkling2_mini-design.md §2); the config supplies dimensions, the loop schedule, the layer kinds
// and each layer's expert arithmetic.

#include "models/load_options.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace infernix::artifact {
struct Directory;
}

namespace infernix::models::inkling2 {

enum class AttentionKind : std::uint8_t { Sliding, Full };

// A layer's routed-expert arithmetic, part of the trained model (the artifact's Uses permit it): NVFP4 weights
// with per-token four-over-six NVFP4 activations (W4A4), or MXFP8 weights with MXFP8 activations (W8A8).
enum class ExpertFormat : std::uint8_t { Nvfp4FourOverSix, Mxfp8 };

struct LoopConfig {
    std::uint32_t core_start   = 0; // first layer of the looped core
    std::uint32_t core_end     = 0; // one past its last layer
    std::uint32_t passes       = 0; // times the core runs
    std::uint32_t message_rank = 0; // the loop message gates' rank

    [[nodiscard]] std::uint32_t core_layers() const noexcept { return core_end - core_start; }
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0; // the real vocabulary: embedding and head rows
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t heads                   = 0;
    std::uint32_t kv_heads                = 0;
    std::uint32_t head_dim                = 0;
    std::vector<AttentionKind> layer_types;
    std::uint32_t sliding_window    = 0;
    std::uint32_t relative_dim      = 0;
    std::uint32_t relative_extent   = 0;
    float qk_clip_threshold         = 0;
    float relative_clip_threshold   = 0;
    float log_scaling_alpha         = 0;
    std::uint32_t log_scaling_floor = 0;
    std::uint32_t conv_kernel       = 0;
    std::uint32_t residual_block    = 0; // layers per attention-residual block
    LoopConfig loop;
    std::uint32_t experts      = 0;
    std::uint32_t top_k        = 0;
    std::uint32_t intermediate = 0;
    float route_scale          = 0;
    float gate_softcap         = 0;
    float up_softcap           = 0;
    std::vector<ExpertFormat> expert_formats;
    float final_logit_softcap = 0;
    float rms_norm_eps        = 0;
    std::int32_t eos_token_id = 0;

    [[nodiscard]] std::uint32_t query_width() const noexcept { return heads * head_dim; }
    [[nodiscard]] std::uint32_t key_width() const noexcept { return kv_heads * head_dim; }
    [[nodiscard]] std::uint32_t relative_width() const noexcept { return heads * relative_dim; }
    // Layer visits of one forward: the prelude, the core `passes` times, the coda.
    [[nodiscard]] std::uint32_t visits() const noexcept {
        return num_hidden_layers + loop.core_layers() * (loop.passes - 1);
    }
    // The layer a visit runs and the pass it belongs to (0 outside the core).
    [[nodiscard]] std::uint32_t visit_layer(std::uint32_t visit) const noexcept {
        if (visit < loop.core_start) { return visit; }
        const std::uint32_t core_visits = loop.core_layers() * loop.passes;
        if (visit < loop.core_start + core_visits) { return loop.core_start + (visit - loop.core_start) % loop.core_layers(); }
        return visit - core_visits + loop.core_layers();
    }
    [[nodiscard]] std::uint32_t visit_pass(std::uint32_t visit) const noexcept {
        if (visit < loop.core_start || visit >= loop.core_start + loop.core_layers() * loop.passes) { return 0; }
        return (visit - loop.core_start) / loop.core_layers();
    }
    [[nodiscard]] bool in_core(std::uint32_t layer) const noexcept {
        return layer >= loop.core_start && layer < loop.core_end;
    }
};

// The DSpark drafter (design §13): DFlash-style block layers over context features, then the Markov RNN head.
struct DSparkConfig {
    std::uint32_t layers       = 0;
    std::uint32_t heads        = 0;
    std::uint32_t kv_heads     = 0;
    std::uint32_t head_dim     = 0;
    std::uint32_t intermediate = 0;
    std::vector<AttentionKind> layer_types;
    std::uint32_t sliding_window = 0;
    float rope_theta             = 0;
    float rms_norm_eps           = 0;
    std::uint32_t block_size     = 0; // the anchor and block_size - 1 drafted positions
    std::uint32_t markov_rank    = 0;
    // Inputs of these visits (0-based), then the final state (== TextConfig::visits()).
    std::vector<std::uint32_t> target_layer_ids;
};

// The vision tower (design §15): a pre-norm ViT over (patch x patch x 3) patches with a class token, the learned
// position table bilinearly resized to the image's patch grid, axial rotary on interleaved pairs, layer scale, then
// merge x merge patch rows per image token and the two-layer projector into the text width. One image token per
// (patch * merge)-pixel cell, its row replacing the image placeholder's embedding.
struct VisionConfig {
    std::uint32_t hidden_size       = 0;
    std::uint32_t layers            = 0;
    std::uint32_t heads             = 0;
    std::uint32_t intermediate      = 0;
    std::uint32_t patch_size        = 0;
    std::uint32_t merge             = 0;
    std::uint32_t position_grid     = 0; // the learned table's side
    std::uint32_t projection_hidden = 0;
    std::uint32_t output_size       = 0; // the text width
    float layer_norm_eps            = 0;
    float rope_theta                = 0;
    std::string image_token;       // one per cell
    std::string image_start_token; // the chat template's image part marker

    [[nodiscard]] std::uint32_t head_dim() const noexcept { return hidden_size / heads; }
    [[nodiscard]] std::uint32_t cell() const noexcept { return patch_size * merge; }
    [[nodiscard]] std::uint32_t patch_width() const noexcept { return 3 * patch_size * patch_size; }
    [[nodiscard]] std::uint32_t merged_width() const noexcept { return merge * merge * hidden_size; }
};

// A DFlash2 drafter of a companion artifact (--spec dflash2 --draft-model; docs/maintainer/dflash.md, design §13.8): the
// DFlash2 backbone (sliding-window layers with dynamic convolutions over a per-lane context ring) conditioned on the
// DSpark features (the same visits), drafting from the target's head over the real vocabulary with the conditional
// candidate selector.
struct DFlash2Config {
    std::uint32_t layers       = 0;
    std::uint32_t heads        = 0;
    std::uint32_t kv_heads     = 0;
    std::uint32_t head_dim     = 0;
    std::uint32_t intermediate = 0;
    std::uint32_t window       = 0; // the sliding window and context ring capacity
    float rms_norm_eps         = 0;
    float rope_theta           = 0; // full-head 1-D RoPE at absolute positions
    std::uint32_t block_size   = 0; // the trained block: the anchor and block_size - 1 drafts
    std::uint32_t mask_token_id  = 0;
    std::uint32_t conv_kernel    = 0;
    std::uint32_t conv_group     = 0;
    std::uint32_t selector_rank  = 0;
    std::uint32_t selector_top_k = 0;
    // The selector's logit softcap (0: none), on the candidates' logits (before the correction) or on each edge's
    // corrected score (after it: incoai's Inkling drafter).
    float softcap               = 0;
    bool softcap_after_correction = false;
    // Inputs of these visits (0-based), then the final state (== TextConfig::visits()): DSpark's taps.
    std::vector<std::uint32_t> target_layer_ids;

    [[nodiscard]] std::uint32_t query_width() const noexcept { return heads * head_dim; }
    [[nodiscard]] std::uint32_t key_width() const noexcept { return kv_heads * head_dim; }
    [[nodiscard]] std::uint32_t feature_width(std::uint32_t hidden) const noexcept {
        return hidden * static_cast<std::uint32_t>(target_layer_ids.size());
    }
};

struct Config {
    TextConfig text;
    std::optional<DSparkConfig> dspark;   // the drafter is bound (--spec dspark)
    std::optional<DFlash2Config> dflash2; // the companion's drafter is bound (--spec dflash2 --draft-model)
    std::optional<VisionConfig> vision;   // the tower is bound (--vision)

    // The bound drafter's feature visits (the Forward's taps); empty without one.
    [[nodiscard]] const std::vector<std::uint32_t>& draft_taps() const noexcept {
        static const std::vector<std::uint32_t> none;
        return dspark ? dspark->target_layer_ids : dflash2 ? dflash2->target_layer_ids : none;
    }
};

[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);
// The DFlash2 companion's config (one dflash2 component, no text), checked against the target it binds to.
[[nodiscard]] DFlash2Config parse_dflash2_config(const artifact::Directory& companion, const TextConfig& target);
// Whether `companion` is a DFlash2 drafter of the Inkling 2 model of `target` (the Engine's companion discovery).
[[nodiscard]] bool dflash2_companion_of(const artifact::Directory& companion, const artifact::Directory& target);
// Refuses a drafter outside the fused DFlash2 Ops' registered geometry (execution/dflash2.cpp): 5 sliding layers, 32
// query and 8 KV heads of 128, MLP 6912, window 2048, block 8, two taps of groups of 16, top 16 of rank 256, eps 1e-6.
void require_dflash2_kernel_geometry(const DFlash2Config& drafter, const TextConfig& text);

// Refuses (ArtifactError) a drafter the DSpark kernels and buffers are not built for: they are fixed to Inkling 2 Mini's
// (include/infernix/ops/inkling2_dspark.h; execution/dspark.cpp sizes its buffers by those constants and its kernels
// take the norm epsilon as a constant), while the binder would bind another drafter's weights at their own shapes.
// parse_config applies it where the drafter is bound (--spec dspark), so a load without the drafter is unaffected.
void require_dspark_kernel_geometry(const DSparkConfig& drafter);

} // namespace infernix::models::inkling2
