#include "models/qwen4_exp/config.h"

#include "artifact/schema.h"
#include "models/qwen3_5/vision_config.h"
#include "models/registry.h"

#include <cmath>
#include <limits>
#include <string>

namespace infernix::models::qwen4_exp {
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

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

std::uint64_t wide(const Json& config, const char* name) {
    return artifact::require_u64(config.at(name), name, true);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

RopeConfig rope(const Json& value, std::uint32_t head_dim) {
    require_members(value, {"rope_theta", "partial_rotary_factor", "mrope_section"}, {},
                    "text RoPE");
    RopeConfig out;
    out.theta            = positive_float(value, "rope_theta");
    const float factor   = positive_float(value, "partial_rotary_factor");
    out.rotary_dim       = static_cast<std::uint32_t>(double(head_dim) * factor);
    if (factor > 1 || !out.rotary_dim || out.rotary_dim % 2) {
        throw ArtifactError("rotary dimension must be positive, even and within the head");
    }
    const auto& sections = value.at("mrope_section");
    if (!sections.is_array() || sections.size() != 3) {
        throw ArtifactError("MRoPE requires three sections");
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        out.mrope_section[i] = integer(sections[i], "MRoPE section", false);
        sum += out.mrope_section[i];
    }
    if (sum != out.rotary_dim / 2) { throw ArtifactError("MRoPE sections differ from the rotary width"); }
    std::array<std::uint32_t, 3> counts{};
    for (std::uint32_t pair = 0; pair < out.rotary_dim / 2; ++pair) {
        const std::uint8_t axis = pair % 3 == 1 && pair < 3ULL * out.mrope_section[1]   ? 1
                                  : pair % 3 == 2 && pair < 3ULL * out.mrope_section[2] ? 2
                                                                                        : 0;
        out.pair_axes.push_back(axis);
        ++counts[axis];
    }
    if (counts != out.mrope_section) {
        throw ArtifactError("MRoPE sections exceed interleaved axis ranges");
    }
    return out;
}

NgramTableConfig ngram_table(const Json& value, std::uint64_t hash_rows, std::uint32_t head_width) {
    require_members(value,
                    {"format", "rows", "row_bytes", "rows_per_block", "block_bytes",
                     "header_bytes", "blocks", "file_bytes", "volume_id"},
                    {}, "n-gram table");
    NgramTableConfig out;
    if (value.at("format") == "fp8_e4m3fn") {
        out.format = NgramRowFormat::Fp8E4m3;
    } else if (value.at("format") == "iq4_nl") {
        // A GGUF table's rows verbatim: IQ4_NL blocks of 32 (fp16 scale and 16 nibble bytes).
        out.format = NgramRowFormat::Iq4Nl;
        if (head_width % 32) { throw ArtifactError("an IQ4_NL n-gram row must be whole 32-element blocks"); }
        head_width = head_width / 32 * 18;
    } else if (value.at("format") == "bf16") {
        // The unquantized table's BF16 words (an exact conversion bound to a BF16 volume).
        out.format = NgramRowFormat::Bf16;
        head_width = 2 * head_width;
    } else if (value.at("format") == "nvfp4") {
        // NVIDIA FP4-Local's rows verbatim: W / 2 E2M1 code bytes, then W / 16 E4M3FN block scales.
        out.format = NgramRowFormat::Nvfp4;
        if (head_width % 16) { throw ArtifactError("an NVFP4 n-gram row must be whole 16-element blocks"); }
        head_width = head_width / 2 + head_width / 16;
    } else {
        throw ArtifactError("n-gram table must hold FP8 E4M3FN, IQ4_NL, BF16 or NVFP4 rows");
    }
    out.rows           = wide(value, "rows");
    out.row_bytes      = dimension(value, "row_bytes");
    out.rows_per_block = dimension(value, "rows_per_block");
    out.block_bytes    = dimension(value, "block_bytes");
    out.header_bytes   = dimension(value, "header_bytes");
    out.blocks         = wide(value, "blocks");
    out.file_bytes     = wide(value, "file_bytes");
    if (out.rows != hash_rows) {
        throw ArtifactError("n-gram table rows differ from the hash domain");
    }
    if (out.row_bytes != head_width || out.block_bytes != 4096 || out.header_bytes != 4096 ||
        out.rows_per_block != out.block_bytes / out.row_bytes ||
        out.blocks != (out.rows + out.rows_per_block - 1) / out.rows_per_block ||
        out.file_bytes != std::uint64_t(out.header_bytes) + out.blocks * out.block_bytes) {
        throw ArtifactError("n-gram table geometry is inconsistent");
    }
    const auto id = value.at("volume_id");
    if (!id.is_string() || id.get<std::string>().size() != 32) {
        throw ArtifactError("n-gram volume_id must be 32 hex digits");
    }
    const std::string hex = id.get<std::string>();
    for (std::size_t i = 0; i < 16; ++i) {
        out.volume_id[i] = static_cast<std::uint8_t>(std::stoul(hex.substr(2 * i, 2), nullptr, 16));
    }
    return out;
}

TextConfig text(const Json& value) {
    require_members(
        value,
        {"architectures", "model_type", "hidden_size", "vocab_size", "num_hidden_layers",
         "max_position_embeddings", "tie_word_embeddings", "rms_norm_eps", "layer_types",
         "num_attention_heads", "num_key_value_heads", "head_dim", "rope_parameters",
         "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
         "linear_value_head_dim", "linear_conv_kernel_dim", "output_gate_type", "num_experts",
         "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
         "norm_topk_prob", "hc_count", "hc_lowrank", "indexer_n_heads", "indexer_kv_heads",
         "indexer_head_dim", "indexer_budget", "indexer_compress_ratio", "ple_layer_ids",
         "ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
         "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "seed",
         "split_ngram_parts", "eos_token_id", "ngram_table"},
        {"gdn_output_order", "gdn_a", "expert_a4_activations"}, "text config");
    const auto& names = value.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    if (resolve_architecture(artifact::require_id(names[0], "architecture"),
                             artifact::require_id(value.at("model_type"), "model_type")) !=
        Architecture::Qwen4Exp) {
        throw ArtifactError("text component is not Qwen4Exp");
    }
    if (value.at("tie_word_embeddings") != false || value.at("norm_topk_prob") != true) {
        throw ArtifactError("Qwen4Exp requires untied embeddings and normalized top-k weights");
    }
    TextConfig out;
    out.hidden_size             = dimension(value, "hidden_size");
    out.vocab_size              = dimension(value, "vocab_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    out.eos_token_id            = static_cast<std::int32_t>(integer(value.at("eos_token_id"), "eos", false));
    if (static_cast<std::uint32_t>(out.eos_token_id) >= out.vocab_size) {
        throw ArtifactError("eos_token_id exceeds the vocabulary");
    }
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must cover every text block");
    }
    for (const auto& layer : layers) {
        if (layer == "full_attention") {
            out.layer_types.push_back(MixerKind::Attention);
            out.compact_layer_indices.push_back(out.attention_layers++);
        } else if (layer == "linear_attention") {
            out.layer_types.push_back(MixerKind::Gdn);
            out.compact_layer_indices.push_back(out.gdn_layers++);
        } else {
            throw ArtifactError("unknown text layer_type");
        }
    }
    out.attention = {dimension(value, "num_attention_heads"), dimension(value, "num_key_value_heads"),
                     dimension(value, "head_dim")};
    if (out.attention.heads % out.attention.kv_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    out.rope = rope(value.at("rope_parameters"), out.attention.head_dim);
    out.gdn  = {dimension(value, "linear_num_key_heads"), dimension(value, "linear_key_head_dim"),
                dimension(value, "linear_num_value_heads"), dimension(value, "linear_value_head_dim"),
                dimension(value, "linear_conv_kernel_dim")};
    if (out.gdn.value_heads % out.gdn.key_heads) {
        throw ArtifactError("GDN value heads must be divisible by key heads");
    }
    const auto& gate = value.at("output_gate_type");
    if (gate == "sigmoid") {
        out.gdn.output_gate = GateActivation::Sigmoid;
    } else if (gate == "silu") {
        out.gdn.output_gate = GateActivation::Silu;
    } else {
        throw ArtifactError("output_gate_type must be sigmoid or silu");
    }
    // A GGUF import: the output projection's columns in ggml's tiled value-head order, and the
    // decay as A = -exp(A_log) itself (tools/convert/qwen4_exp_gguf.py).
    if (value.contains("gdn_output_order")) {
        if (value.at("gdn_output_order") != "ggml_tiled") { throw ArtifactError("gdn_output_order must be ggml_tiled"); }
        out.gdn.output_ggml_tiled = true;
    }
    if (value.contains("gdn_a")) {
        if (value.at("gdn_a") != "direct") { throw ArtifactError("gdn_a must be direct"); }
        out.gdn.a_direct = true;
    }
    if (out.gdn.a_direct != out.gdn.output_ggml_tiled) {
        throw ArtifactError("gdn_a and gdn_output_order describe one GGUF import and come together");
    }
    out.ggml_weights = out.gdn.a_direct;
    out.moe = {dimension(value, "num_experts"), dimension(value, "num_experts_per_tok"),
               dimension(value, "moe_intermediate_size"),
               dimension(value, "shared_expert_intermediate_size")};
    if (out.moe.top_k > out.moe.experts) { throw ArtifactError("selected experts exceed expert count"); }
    if (value.contains("expert_a4_activations")) {
        if (value.at("expert_a4_activations") != "optional") {
            throw ArtifactError("expert_a4_activations must be optional");
        }
        out.moe.a4_optional = true;
    }
    out.hc = {dimension(value, "hc_count"), dimension(value, "hc_lowrank")};
    if (out.hc.streams < 2) { throw ArtifactError("hc_count must exceed one"); }
    out.qsa = {dimension(value, "indexer_n_heads"), dimension(value, "indexer_head_dim"),
               dimension(value, "indexer_budget"), dimension(value, "indexer_compress_ratio")};
    if (dimension(value, "indexer_kv_heads") != 1 || out.qsa.budget % out.qsa.compress_ratio ||
        out.rope.rotary_dim > out.qsa.index_head_dim) {
        throw ArtifactError("invalid QSA indexer geometry");
    }
    const auto& ple = value.at("ple_layer_ids");
    if (!ple.is_array() || ple.size() != 1) {
        throw ArtifactError("Qwen4Exp implements exactly one PLE layer");
    }
    const auto ple_layer = integer(ple[0], "PLE layer id");
    if (ple_layer > out.num_hidden_layers || out.layer_types[ple_layer - 1] != MixerKind::Gdn) {
        throw ArtifactError("the PLE layer must be a linear-attention layer");
    }
    out.ple.layer         = ple_layer - 1;
    out.ple.embed_dim     = dimension(value, "ple_embed_dim");
    out.ple.conv_kernel   = dimension(value, "ple_conv_kernel_size");
    out.ple.split_parts   = dimension(value, "split_ngram_parts");
    out.ple.ngram         = NgramConfig{
                .vocab_size                         = out.vocab_size,
                .eos_token_id                       = out.eos_token_id,
                .ngram_size                         = dimension(value, "ngram_size"),
                .heads_per_ngram                    = dimension(value, "heads_per_ngram"),
                .ngram_vocab_size_base              = wide(value, "ngram_vocab_size_base"),
                .make_ngram_vocab_size_divisible_by = wide(value, "make_ngram_vocab_size_divisible_by"),
                .seed = artifact::require_u64(value.at("seed"), "seed", false)};
    out.ple.conv_dilation = out.ple.ngram.ngram_size;
    if (out.ple.ngram.ngram_size < 2 || out.ple.embed_dim % out.ple.heads()) {
        throw ArtifactError("invalid PLE n-gram geometry");
    }
    const NgramHash hash(out.ple.ngram, 0);
    out.ple.table = ngram_table(value.at("ngram_table"), hash.table_rows(), out.ple.head_width());
    return out;
}

} // namespace

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.purpose != EnginePurpose::Generation &&
            options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        if (options.speculative != SpeculativeBackend::None && !options.mtp() && !options.dflash2()) {
            throw ArtifactError("Qwen4Exp speculative decoding uses its MTP drafter (--spec mtp) or a DFlash2 "
                                "drafter (--spec dflash2 --draft-model FILE)");
        }
        if (options.dflash2() && options.draft_model.empty()) {
            throw ArtifactError("--spec dflash2 needs the DFlash2 companion artifact (--draft-model FILE)");
        }
        if (!options.dflash2() && !options.draft_model.empty()) {
            throw ArtifactError("--draft-model is the DFlash2 drafter: it needs --spec dflash2");
        }
        if (!std::isfinite(options.rope_yarn_factor) || options.rope_yarn_factor < 1.0F ||
            options.rope_yarn_factor > 4.0F) {
            throw ArtifactError("rope_yarn_factor must be finite and in [1,4]");
        }
        Config out;
        out.text = text(directory.component("text").config);
        out.text.rope.yarn_factor = options.rope_yarn_factor;
        // A weight-only export stores its banks without activation scales (bind_moe checks every
        // layer); banks whose 4-bit activations are optional run them only when asked.
        const bool scales = directory.bindings.contains("text/layers/0/moe/expert_input_scales");
        if (out.text.moe.a4_optional && !scales) {
            throw ArtifactError("optional 4-bit expert activations need the banks' activation scales");
        }
        if (options.expert_a4_activations && !out.text.moe.a4_optional) {
            throw ArtifactError("--expert-a4-activations needs an artifact whose routed experts offer 4-bit "
                                "activations as an option (nvidia/Qwen3.8-Flash-Next-FP4-Local)");
        }
        out.text.lossless         = options.lossless;
        out.text.design_reference = options.lossless && options.reference;
        out.text.moe.a16_experts = out.text.moe.a4_optional ? !options.expert_a4_activations : !scales;
        if (options.rope_yarn_factor > 1.0F && out.text.rope.theta <= 1.0F) {
            throw ArtifactError("YaRN needs a RoPE theta above 1");
        }
        if (options.vision) {
            if (!directory.components.contains("vision")) { throw ArtifactError("the artifact has no vision component"); }
            constexpr std::string_view types[] = {"qwen4_exp_vision"};
            out.vision = qwen3_5::parse_vision_config(directory.component("vision").config, types);
            if (out.vision->hidden_size / out.vision->num_heads != 72) {
                throw ArtifactError("the Vision tower must have 72-wide heads");
            }
            // Image tokens rotate by three axes: the text RoPE must be interleaved M-RoPE.
            for (std::size_t pair = 0; pair < out.text.rope.pair_axes.size(); ++pair) {
                if (out.text.rope.pair_axes[pair] != pair % 3) {
                    throw ArtifactError("Qwen4Exp vision needs interleaved M-RoPE (pair i on axis i % 3)");
                }
            }
        }
        if (options.purpose == EnginePurpose::Generation && (options.mtp() || options.dflash2())) {
            if (options.mtp() && !directory.components.contains("mtp")) {
                throw ArtifactError("the artifact has no MTP component");
            }
            out.mtp = options.mtp();
            if (options.proposal_head == ProposalHead::Optimized) {
                const auto& proposal = directory.component("text").proposal;
                if (!proposal || !proposal->indexed || proposal->rows == 0 ||
                    proposal->rows > out.text.vocab_size) {
                    throw ArtifactError("--lm-head-draft needs an artifact converted with --proposal");
                }
                out.proposal_rows = static_cast<std::uint32_t>(proposal->rows);
            }
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Qwen4Exp config: ") + error.what());
    }
}

DFlash2Config parse_dflash2_config(const artifact::Directory& companion, const TextConfig& target) {
    try {
        if (!companion.companion || companion.components.size() != 1 || !companion.components.contains("dflash2")) {
            throw ArtifactError("the draft model must be a DFlash2 companion artifact (one dflash2 component, no text)");
        }
        const Json& value = companion.component("dflash2").config;
        require_members(value,
                        {"architectures", "model_type", "intermediate_size", "num_attention_heads",
                         "num_key_value_heads", "head_dim", "num_hidden_layers", "max_position_embeddings",
                         "rms_norm_eps", "rope_parameters", "layer_types", "sliding_window", "dflash_config",
                         "target_config"},
                        {}, "DFlash2 config");
        const auto& names = value.at("architectures");
        if (!names.is_array() || names.size() != 1 || names[0] != "DFlash2DraftModel" || value.at("model_type") != "qwen3") {
            throw ArtifactError("the draft model is not a DFlash2DraftModel");
        }
        // The target it was converted for: this architecture and geometry.
        const Json& t = value.at("target_config");
        require_members(t, {"architectures", "hidden_size", "vocab_size", "num_hidden_layers"}, {}, "DFlash2 target");
        if (!t.at("architectures").is_array() || t.at("architectures").size() != 1 ||
            t.at("architectures")[0] != "Qwen4ExpForCausalLM" || dimension(t, "hidden_size") != target.hidden_size ||
            dimension(t, "vocab_size") != target.vocab_size ||
            dimension(t, "num_hidden_layers") != target.num_hidden_layers) {
            throw ArtifactError("the DFlash2 drafter was converted for another target (" + t.dump() + ")");
        }
        DFlash2Config out;
        out.layers                = dimension(value, "num_hidden_layers");
        out.attention.heads       = dimension(value, "num_attention_heads");
        out.attention.kv_heads    = dimension(value, "num_key_value_heads");
        out.attention.head_dim    = dimension(value, "head_dim");
        out.intermediate          = dimension(value, "intermediate_size");
        out.window                = dimension(value, "sliding_window");
        out.rms_norm_eps          = positive_float(value, "rms_norm_eps");
        require_members(value.at("rope_parameters"), {"rope_theta"}, {}, "DFlash2 RoPE");
        out.rope_theta = positive_float(value.at("rope_parameters"), "rope_theta");
        if (out.attention.heads % out.attention.kv_heads || out.attention.head_dim % 2) {
            throw ArtifactError("invalid DFlash2 attention geometry");
        }
        const auto& layers = value.at("layer_types");
        if (!layers.is_array() || layers.size() != out.layers) { throw ArtifactError("DFlash2 layer_types length mismatch"); }
        for (const auto& layer : layers) {
            if (layer != "sliding_attention") {
                throw ArtifactError("the Qwen4Exp DFlash2 drafter has sliding-window layers only");
            }
        }
        const Json& d = value.at("dflash_config");
        require_members(d,
                        {"target_layer_ids", "mask_token_id", "conv_kernel_size", "conv_group_size", "selector_rank",
                         "selector_top_k"},
                        {}, "DFlash2 dflash_config");
        const auto& taps = d.at("target_layer_ids");
        if (!taps.is_array() || taps.empty()) { throw ArtifactError("target_layer_ids must be nonempty"); }
        for (const auto& tap : taps) {
            const auto layer = integer(tap, "target layer", false);
            if (layer >= target.num_hidden_layers ||
                (!out.target_layer_ids.empty() && layer <= out.target_layer_ids.back())) {
                throw ArtifactError("target_layer_ids must be ascending target layers");
            }
            out.target_layer_ids.push_back(layer);
        }
        out.mask_token_id  = integer(d.at("mask_token_id"), "mask token", false);
        out.conv_kernel    = dimension(d, "conv_kernel_size");
        out.conv_group     = dimension(d, "conv_group_size");
        out.selector_rank  = dimension(d, "selector_rank");
        out.selector_top_k = dimension(d, "selector_top_k");
        if (out.mask_token_id >= target.vocab_size) { throw ArtifactError("the mask token exceeds the embedding rows"); }
        if (target.hidden_size % out.conv_group) {
            throw ArtifactError("the DFlash2 conv group does not divide the hidden width");
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("DFlash2 companion config: ") + error.what());
    }
}

} // namespace infernix::models::qwen4_exp
