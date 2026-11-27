#include "models/inkling2/config.h"

#include "artifact/schema.h"
#include "infernix/ops/inkling2_dspark.h"

#include <cmath>
#include <limits>
#include <string>

namespace infernix::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) { return integer(config.at(name), name); }

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

std::vector<AttentionKind> layer_types(const Json& value, std::uint32_t layers, std::string_view label) {
    if (!value.is_array() || value.size() != layers) {
        throw ArtifactError(std::string(label) + ": layer_types must name every layer");
    }
    std::vector<AttentionKind> out;
    for (const Json& kind : value) {
        if (kind == "sliding_attention") {
            out.push_back(AttentionKind::Sliding);
        } else if (kind == "full_attention") {
            out.push_back(AttentionKind::Full);
        } else {
            throw ArtifactError(std::string(label) + ": layer types are sliding_attention or full_attention");
        }
    }
    return out;
}

TextConfig text(const Json& value) {
    require_members(value,
                    {"architectures", "model_type", "hidden_size", "num_hidden_layers", "vocab_size",
                     "num_attention_heads", "num_key_value_heads", "head_dim", "layer_types", "sliding_window",
                     "relative_dim", "relative_extent", "qk_clip_threshold", "relative_clip_threshold",
                     "log_scaling_alpha", "log_scaling_floor", "conv_kernel_size", "attention_residual_block", "loop",
                     "num_experts", "num_experts_per_tok", "moe_intermediate_size", "route_scale", "gate_softcap",
                     "up_softcap", "expert_formats", "final_logit_softcapping", "rms_norm_eps",
                     "max_position_embeddings", "eos_token_id"},
                    {}, "Inkling2Loop text config");
    if (value.at("architectures") != Json::array({"Inkling2LoopForCausalLM"}) ||
        value.at("model_type") != "inkling2_loop_text") {
        throw ArtifactError("the text component is not Inkling2LoopForCausalLM");
    }
    TextConfig out;
    out.hidden_size             = dimension(value, "hidden_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.vocab_size              = dimension(value, "vocab_size");
    out.heads                   = dimension(value, "num_attention_heads");
    out.kv_heads                = dimension(value, "num_key_value_heads");
    out.head_dim                = dimension(value, "head_dim");
    out.layer_types             = layer_types(value.at("layer_types"), out.num_hidden_layers, "text");
    out.sliding_window          = dimension(value, "sliding_window");
    out.relative_dim            = dimension(value, "relative_dim");
    out.relative_extent         = dimension(value, "relative_extent");
    out.qk_clip_threshold       = positive_float(value, "qk_clip_threshold");
    out.relative_clip_threshold = positive_float(value, "relative_clip_threshold");
    out.log_scaling_alpha       = positive_float(value, "log_scaling_alpha");
    out.log_scaling_floor       = dimension(value, "log_scaling_floor");
    out.conv_kernel             = dimension(value, "conv_kernel_size");
    out.residual_block          = dimension(value, "attention_residual_block");
    out.experts                 = dimension(value, "num_experts");
    out.top_k                   = dimension(value, "num_experts_per_tok");
    out.intermediate            = dimension(value, "moe_intermediate_size");
    out.route_scale             = positive_float(value, "route_scale");
    out.gate_softcap            = positive_float(value, "gate_softcap");
    out.up_softcap              = positive_float(value, "up_softcap");
    out.final_logit_softcap     = positive_float(value, "final_logit_softcapping");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.eos_token_id            = static_cast<std::int32_t>(integer(value.at("eos_token_id"), "eos_token_id", false));

    const Json& loop = value.at("loop");
    require_members(loop, {"core_start", "core_end", "passes", "message_rank"}, {}, "Inkling2Loop loop");
    out.loop.core_start   = dimension(loop, "core_start");
    out.loop.core_end     = dimension(loop, "core_end");
    out.loop.passes       = dimension(loop, "passes");
    out.loop.message_rank = dimension(loop, "message_rank");

    const Json& formats = value.at("expert_formats");
    if (!formats.is_array() || formats.size() != out.num_hidden_layers) {
        throw ArtifactError("expert_formats must name every layer's expert arithmetic");
    }
    for (const Json& format : formats) {
        if (format == "nvfp4_four_over_six") {
            out.expert_formats.push_back(ExpertFormat::Nvfp4FourOverSix);
        } else if (format == "mxfp8") {
            out.expert_formats.push_back(ExpertFormat::Mxfp8);
        } else {
            throw ArtifactError("expert formats are nvfp4_four_over_six or mxfp8");
        }
    }

    // The shapes the mathematics and its Ops assume (design §2): grouped heads, whole 32-element expert blocks,
    // the loop core made of whole attention-residual blocks, top-k within the experts.
    if (out.heads % out.kv_heads || out.hidden_size % 32 || out.intermediate % 32 || out.top_k > out.experts ||
        out.loop.passes < 2 || out.loop.core_start == 0 || out.loop.core_end <= out.loop.core_start ||
        out.loop.core_end > out.num_hidden_layers || out.loop.core_start % out.residual_block ||
        out.loop.core_end % out.residual_block || out.relative_extent > out.sliding_window ||
        out.layer_types.front() != AttentionKind::Sliding) {
        throw ArtifactError("the Inkling2Loop text config's dimensions are inconsistent");
    }
    return out;
}

DSparkConfig dspark(const Json& value, const TextConfig& text) {
    require_members(value,
                    {"architectures", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                     "head_dim", "intermediate_size", "layer_types", "sliding_window", "rope_theta", "rms_norm_eps",
                     "block_size", "markov_rank", "target_layer_ids"},
                    {}, "DSpark config");
    if (value.at("architectures") != Json::array({"Inkling2DSpark"}) ||
        dimension(value, "hidden_size") != text.hidden_size) {
        throw ArtifactError("the dspark component is not a drafter of this Inkling 2 model");
    }
    DSparkConfig out;
    out.layers         = dimension(value, "num_hidden_layers");
    out.heads          = dimension(value, "num_attention_heads");
    out.kv_heads       = dimension(value, "num_key_value_heads");
    out.head_dim       = dimension(value, "head_dim");
    out.intermediate   = dimension(value, "intermediate_size");
    out.layer_types    = layer_types(value.at("layer_types"), out.layers, "dspark");
    out.sliding_window = dimension(value, "sliding_window");
    out.rope_theta     = positive_float(value, "rope_theta");
    out.rms_norm_eps   = positive_float(value, "rms_norm_eps");
    out.block_size     = dimension(value, "block_size");
    out.markov_rank    = dimension(value, "markov_rank");
    const Json& targets = value.at("target_layer_ids");
    if (!targets.is_array() || targets.size() < 2) { throw ArtifactError("DSpark needs its feature visits"); }
    for (const Json& target : targets) { out.target_layer_ids.push_back(integer(target, "target_layer_ids", false)); }
    if (out.target_layer_ids.back() != text.visits() || out.heads % out.kv_heads || out.block_size < 2) {
        throw ArtifactError("the DSpark config does not fit this Inkling 2 model");
    }
    for (std::size_t i = 0; i + 1 < out.target_layer_ids.size(); ++i) {
        if (out.target_layer_ids[i] >= text.visits()) { throw ArtifactError("a DSpark feature visit is out of range"); }
    }
    return out;
}

VisionConfig vision(const Json& value, const TextConfig& text) {
    require_members(value,
                    {"architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                     "intermediate_size", "patch_size", "spatial_merge_size", "position_grid",
                     "projection_hidden_size", "output_size", "layer_norm_eps", "rope_theta", "image_token",
                     "image_start_token", "image_mean", "image_std", "rescale_factor"},
                    {}, "vision config");
    if (value.at("architectures") != Json::array({"Inkling2Vision"}) || value.at("model_type") != "inkling2_vision" ||
        dimension(value, "output_size") != text.hidden_size) {
        throw ArtifactError("the vision component is not a tower of this Inkling 2 model");
    }
    // The processor the frontend implements (Inkling2ImageProcessor): v / 255, then (x - 0.5) / 0.5.
    if (value.at("image_mean") != Json::array({0.5, 0.5, 0.5}) || value.at("image_std") != Json::array({0.5, 0.5, 0.5}) ||
        !value.at("rescale_factor").is_number() || value.at("rescale_factor").get<double>() != 1.0 / 255.0) {
        throw ArtifactError("the vision component's pixel normalization is not Inkling2ImageProcessor's");
    }
    VisionConfig out;
    out.hidden_size       = dimension(value, "hidden_size");
    out.layers            = dimension(value, "num_hidden_layers");
    out.heads             = dimension(value, "num_attention_heads");
    out.intermediate      = dimension(value, "intermediate_size");
    out.patch_size        = dimension(value, "patch_size");
    out.merge             = dimension(value, "spatial_merge_size");
    out.position_grid     = dimension(value, "position_grid");
    out.projection_hidden = dimension(value, "projection_hidden_size");
    out.output_size       = dimension(value, "output_size");
    out.layer_norm_eps    = positive_float(value, "layer_norm_eps");
    out.rope_theta        = positive_float(value, "rope_theta");
    out.image_token       = value.at("image_token").get<std::string>();
    out.image_start_token = value.at("image_start_token").get<std::string>();
    // The shapes the tower's Ops cover: 16 heads of 64 (the packed attention's D64 profile), even widths.
    if (out.heads != 16 || out.hidden_size != 16U * 64U || out.intermediate % 2 || out.projection_hidden % 2 ||
        out.image_token.empty() || out.image_start_token.empty()) {
        throw ArtifactError("the Inkling 2 vision config's dimensions are outside the tower's Ops");
    }
    return out;
}

} // namespace

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.purpose != EnginePurpose::Generation && options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        if (options.speculative != SpeculativeBackend::None && options.speculative != SpeculativeBackend::DSpark &&
            options.speculative != SpeculativeBackend::DFlash2) {
            throw ArtifactError("Inkling 2 drafts with DSpark (--spec dspark) or a DFlash2 companion (--spec dflash2 "
                                "--draft-model FILE)");
        }
        // The DFlash2 drafter is the companion's (plan_load parses it); DSpark stays unbound beside it.
        if (options.speculative == SpeculativeBackend::DFlash2 && options.draft_model.empty()) {
            throw ArtifactError("--spec dflash2 needs the drafter's companion artifact (--draft-model FILE)");
        }
        if (options.speculative != SpeculativeBackend::DFlash2 && !options.draft_model.empty()) {
            throw ArtifactError("--draft-model is the DFlash2 drafter's companion (--spec dflash2)");
        }
        Config out;
        out.text = text(directory.component("text").config);
        if (directory.components.contains("vision")) {
            VisionConfig tower = vision(directory.component("vision").config, out.text);
            if (options.vision) { out.vision = std::move(tower); }
        } else if (options.vision) {
            throw ArtifactError("--vision: the artifact has no vision component (convert with --components "
                                "text,dspark,vision)");
        }
        if (options.vision_offload && !options.vision) {
            throw ArtifactError("--vision-offload on needs --vision");
        }
        // A stored drafter must fit the model even when it is not bound.
        if (directory.components.contains("dspark")) {
            DSparkConfig drafter = dspark(directory.component("dspark").config, out.text);
            if (options.speculative == SpeculativeBackend::DSpark) {
                require_dspark_kernel_geometry(drafter);
                out.dspark = std::move(drafter);
            }
        } else if (options.speculative == SpeculativeBackend::DSpark) {
            throw ArtifactError("--spec dspark: the artifact has no dspark component (convert with --components "
                                "text,dspark)");
        }
        return out;
    } catch (const ArtifactError&) {
        throw;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Inkling2Loop config: ") + error.what());
    }
}

DFlash2Config parse_dflash2_config(const artifact::Directory& companion, const TextConfig& target) {
    try {
        if (!companion.companion || companion.components.size() != 1 || !companion.components.contains("dflash2")) {
            throw ArtifactError("the draft model must be a DFlash2 companion artifact (one dflash2 component, no text)");
        }
        const Json& value = companion.component("dflash2").config;
        require_members(value,
                        {"architectures", "model_type", "hidden_size", "intermediate_size", "num_attention_heads",
                         "num_key_value_heads", "head_dim", "num_hidden_layers", "max_position_embeddings",
                         "rms_norm_eps", "rope_parameters", "layer_types", "sliding_window", "dflash_config",
                         "target_config"},
                        {}, "DFlash2 config");
        if (value.at("architectures") != Json::array({"DFlash2DraftModel"}) || value.at("model_type") != "qwen3") {
            throw ArtifactError("the draft model is not a DFlash2DraftModel");
        }
        // The target it was converted for (tools/convert/inkling2_loop.py DFLASH2_TARGET_FIELDS).
        const Json& t = value.at("target_config");
        require_members(t, {"architectures", "hidden_size", "vocab_size", "num_hidden_layers", "visits"}, {},
                        "DFlash2 target");
        if (t.at("architectures") != Json::array({"Inkling2LoopForCausalLM"}) ||
            dimension(t, "hidden_size") != target.hidden_size || dimension(t, "vocab_size") != target.vocab_size ||
            dimension(t, "num_hidden_layers") != target.num_hidden_layers || dimension(t, "visits") != target.visits() ||
            dimension(value, "hidden_size") != target.hidden_size) {
            throw ArtifactError("the DFlash2 drafter was converted for another target (" + t.dump() + ")");
        }
        DFlash2Config out;
        out.layers       = dimension(value, "num_hidden_layers");
        out.heads        = dimension(value, "num_attention_heads");
        out.kv_heads     = dimension(value, "num_key_value_heads");
        out.head_dim     = dimension(value, "head_dim");
        out.intermediate = dimension(value, "intermediate_size");
        out.window       = dimension(value, "sliding_window");
        out.rms_norm_eps = positive_float(value, "rms_norm_eps");
        require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "DFlash2 RoPE");
        out.rope_theta = positive_float(value.at("rope_parameters"), "rope_theta");
        for (const AttentionKind kind : layer_types(value.at("layer_types"), out.layers, "dflash2")) {
            if (kind != AttentionKind::Sliding) { throw ArtifactError("the DFlash2 drafter's layers are all sliding"); }
        }
        const Json& d = value.at("dflash_config");
        require_members(d,
                        {"target_layer_ids", "mask_token_id", "block_size", "conv_kernel_size", "conv_group_size",
                         "selector_rank", "selector_top_k", "final_logit_softcapping", "selector_softcap_mode"},
                        {}, "DFlash2 dflash_config");
        const Json& taps = d.at("target_layer_ids");
        if (!taps.is_array() || taps.size() < 2) { throw ArtifactError("DFlash2 needs its feature visits"); }
        for (const Json& tap : taps) {
            const std::uint32_t visit = integer(tap, "target_layer_ids", false);
            if (!out.target_layer_ids.empty() && visit <= out.target_layer_ids.back()) {
                throw ArtifactError("DFlash2 target_layer_ids must ascend");
            }
            out.target_layer_ids.push_back(visit);
        }
        if (out.target_layer_ids.back() != target.visits()) {
            throw ArtifactError("DFlash2 target_layer_ids must end at the final state (visit " +
                                std::to_string(target.visits()) + ")");
        }
        out.mask_token_id  = integer(d.at("mask_token_id"), "mask_token_id", false);
        out.block_size     = dimension(d, "block_size");
        out.conv_kernel    = dimension(d, "conv_kernel_size");
        out.conv_group     = dimension(d, "conv_group_size");
        out.selector_rank  = dimension(d, "selector_rank");
        out.selector_top_k = dimension(d, "selector_top_k");
        if (!d.at("final_logit_softcapping").is_number()) { throw ArtifactError("DFlash2 softcap must be a number"); }
        out.softcap = d.at("final_logit_softcapping").get<float>();
        if (!std::isfinite(out.softcap) || out.softcap < 0) { throw ArtifactError("DFlash2 softcap must be >= 0"); }
        const Json& mode = d.at("selector_softcap_mode");
        if (mode != "before_correction" && mode != "after_correction") {
            throw ArtifactError("DFlash2 selector_softcap_mode must be before_correction or after_correction");
        }
        out.softcap_after_correction = mode == "after_correction";
        if (out.mask_token_id >= target.vocab_size || out.heads % out.kv_heads || out.head_dim % 2 ||
            out.block_size < 2 || target.hidden_size % out.conv_group) {
            throw ArtifactError("the DFlash2 config does not fit this Inkling 2 model");
        }
        return out;
    } catch (const ArtifactError&) {
        throw;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Inkling 2 DFlash2 config: ") + error.what());
    }
}

bool dflash2_companion_of(const artifact::Directory& companion, const artifact::Directory& target) {
    try {
        (void)parse_dflash2_config(companion, text(target.component("text").config));
        return true;
    } catch (...) {
        return false;
    }
}

void require_dflash2_kernel_geometry(const DFlash2Config& d, const TextConfig& text) {
    const bool ok = text.hidden_size == 2304 && d.layers == 5 && d.heads == 32 && d.kv_heads == 8 &&
                    d.head_dim == 128 && d.intermediate == 6912 && d.window == 2048 && d.block_size == 8 &&
                    d.conv_kernel == 2 && d.conv_group == 16 && d.selector_rank == 256 && d.selector_top_k == 16 &&
                    d.rms_norm_eps == 1e-6F && d.rope_theta > 1.0F &&
                    d.target_layer_ids.size() == static_cast<std::size_t>(ops::inkling2::kDSparkFeatures) &&
                    // The selector applies a softcap after its correction (candidate_selector_path's edge_softcap);
                    // one on the candidates' logits before it is not built.
                    (d.softcap == 0.0F || d.softcap_after_correction);
    if (!ok) {
        throw ArtifactError("the DFlash2 Ops are registered for Inkling 2 Mini's drafter (5 sliding layers; 32 query "
                            "and 8 KV heads of 128; MLP 6912; window 2048; block 8; two conv taps in groups of 16; "
                            "top 16 of rank 256; eps 1e-6; 4 features; a selector softcap after its correction); "
                            "this companion states another");
    }
}

void require_dspark_kernel_geometry(const DSparkConfig& d) {
    namespace k   = ops::inkling2;
    const auto is = [](std::uint32_t value, std::int32_t want) { return value == static_cast<std::uint32_t>(want); };
    const bool layers = d.layer_types.size() == 3 && d.layer_types[0] == AttentionKind::Sliding &&
                        d.layer_types[1] == AttentionKind::Full && d.layer_types[2] == AttentionKind::Sliding;
    if (!layers || !is(d.heads, k::kDSparkHeads) || !is(d.kv_heads, k::kDSparkKvHeads) ||
        !is(d.head_dim, k::kDSparkHeadDim) || !is(d.intermediate, k::kDSparkIntermediate) ||
        !is(d.sliding_window, k::kDSparkWindow) || !is(d.block_size, k::kDSparkBlock) ||
        !is(d.markov_rank, k::kMarkovRank) || d.target_layer_ids.size() != static_cast<std::size_t>(k::kDSparkFeatures) ||
        d.rms_norm_eps != k::DSparkRotateArgs{}.eps) {
        throw ArtifactError("the DSpark drafter's kernels are built for Inkling 2 Mini's drafter (3 layers sliding, "
                            "full, sliding; 24 query and 8 KV heads of 128; MLP 9216; window 1024; block 8; Markov "
                            "rank 256; 4 features; rms_norm_eps 1e-6); the artifact's dspark config states another");
    }
}

} // namespace infernix::models::inkling2
