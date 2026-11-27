#include "launcher/option_support.h"

#include "launcher/util.h"

#include <algorithm>
#include <array>

namespace infernix::launcher {
namespace {

constexpr unsigned bit(ModelKind kind) { return 1U << static_cast<unsigned>(kind); }

constexpr unsigned kDense    = bit(ModelKind::Qwen35Dense);
constexpr unsigned kMoe      = bit(ModelKind::Qwen35Moe);
constexpr unsigned kMoeOff   = bit(ModelKind::Qwen35MoeOffloaded);
constexpr unsigned kFlash    = bit(ModelKind::FlashNext);
constexpr unsigned kInkling  = bit(ModelKind::Inkling2);
constexpr unsigned kQwen35   = kDense | kMoe | kMoeOff;
constexpr unsigned kQwen     = kQwen35 | kFlash;
constexpr unsigned kOffload  = kMoeOff | kFlash;

struct Rule {
    std::string_view flag;
    unsigned kinds;
};

// The options that do not apply to every model. Why, for the model types left out:
constexpr std::array kRules = {
    // Adapters and weights.
    Rule{"--lora", kDense | kInkling},      // the Qwen MoE models refuse it (validate_engine_options, bind_lora)
    Rule{"--q8-weights", kInkling},         // normalize_engine_options refuses it elsewhere
    // Routed experts held off the GPU.
    Rule{"--expert-ram-mib", kOffload},
    Rule{"--expert-ram-layout", kOffload},
    Rule{"--expert-csf", kOffload},
    Rule{"--expert-pin-mib", kOffload},
    Rule{"--expert-state", kOffload},
    Rule{"--cpu-rates", kFlash},             // only Flash-Next's start reads the stored rates
    Rule{"--expert-stripe", kOffload},
    Rule{"--expert-tail-skip", kOffload},   // refused without offloaded experts
    Rule{"--route-trace", kOffload},
    // Memory.
    Rule{"--kv-window-dtype", kInkling},    // Inkling's sliding windows; normalize_engine_options refuses it elsewhere
    Rule{"--ram-resize", kQwen | kInkling}, // the prefix cache's Host tier (and Flash-Next's expert RAM)
    Rule{"--embedding-host", kDense | kMoe | kInkling}, // Flash-Next does not read it; offloaded Qwen pins it anyway
    Rule{"--prefill-split-workspace-mib", kQwen35},
    Rule{"--use-original-int8-prefill-kernel", kQwen35},
    Rule{"--use-original-nvfp4-prefill-kernel", kQwen35},
    // Numerics and routes.
    Rule{"--rope-yarn-factor", kQwen},      // Inkling has no RoPE and refuses it
    Rule{"--logits-dtype", kQwen},          // Inkling's logits are FP32 and it refuses another
    Rule{"--fast-path", kQwen},             // Inkling registers no selectable fast path
    Rule{"--no-fast-path", kQwen},
    Rule{"--no-cuda-graph", kQwen35 | kInkling}, // Flash-Next does not read it
    // The prefix cache.
    Rule{"--cache-host-state", kQwen35},    // Flash-Next and Inkling refuse everything but fp32
    Rule{"--device-snapshot-slots", kQwen35}, // refused by Flash-Next and Inkling
    Rule{"--cache-tool-hold-secs", kQwen35},  // refused by Flash-Next and Inkling
    Rule{"--context-cost-presets", kQwen},  // Inkling's prefix cache has no cost model
    // Speculative decoding.
    Rule{"--draft-model", kFlash | kDense | kInkling}, // DFlash2 companions (Flash-Next, Inkling 2), DSpine (dense)
    Rule{"--draft-tree-nodes", kQwen35},    // Flash-Next verifies neural drafts as chains and refuses it
    Rule{"--draft-tree-paths", kQwen},      // Flash-Next: its n-gram trees' paths
    Rule{"--spec-min-p", kQwen},
    Rule{"--ngram-draft-tokens", kQwen},    // Inkling refuses n-gram drafting
    Rule{"--ngram-min-match", kQwen},
    Rule{"--ngram-tree-nodes", kQwen},
    Rule{"--ngram-archive-mib", kQwen35},   // Flash-Next refuses the archive
    Rule{"--ngram-session-mib", kQwen35},
    Rule{"--ngram-native-sessions", kQwen35},
    Rule{"--ngram-volume", kFlash},
    Rule{"--ngram-residency", kFlash},
    // Thinking and tools: Inkling's template has an effort instead of preserve_thinking, its output session has no loop
    // guard or tolerant tool-call parsing, and its constrained tool route has no parameter-name policy (infernix-serve
    // refuses each: serve::refuse_unsupported_options). Its thinking budgets work as every model's.
    Rule{"--preserve-thinking", kQwen},
    Rule{"--tolerant-tool-calls", kQwen},
    Rule{"--tool-parameter-names", kQwen},
    Rule{"--reasoning-loop", kQwen},
    // Vision's media pipeline (the Qwen frontends'; with `needs_vision` below).
    Rule{"--media-cache-mib", kQwen},
    Rule{"--media-live-mib", kQwen},
    Rule{"--media-preprocess-threads", kQwen},
};

// Options that also need something of the artifact.
constexpr std::array<std::string_view, 6> kNeedsVision = {"--vision",          "--vision-offload",  "--vision-max-merged",
                                                          "--media-cache-mib", "--media-live-mib", "--media-preprocess-threads"};

bool contains_text(const std::string& text, std::string_view needle) { return text.find(needle) != std::string::npos; }

std::string recipe_of(const ScannedModel& model) { return lower(model.recipe.value_or("")); }

// nvidia/Qwen3.8-Flash-Next-FP4-Local, by its recipe or names.
bool fp4_local(const ScannedModel& model) {
    const std::string text = recipe_of(model) + " " + lower(model.name) + " " + lower(model.file_name);
    return contains_text(text, "fp4-local") || contains_text(text, "fp4_local") || contains_text(text, "fp4local");
}

// --a16-activations has something to change: Inkling's routed experts, the NVFP4 and FP8 layers of NVIDIA's NVFP4 27B,
// and Flash-Next's W4A4 experts (other artifacts already run 16-bit: the option's help).
bool takes_a16(const ScannedModel& model) {
    const std::string formats = recipe_of(model) + " " + lower(model.quantization.value_or(""));
    switch (model_kind(model)) {
    case ModelKind::Inkling2:
        return true;
    case ModelKind::Qwen35Dense:
        return contains_text(formats, "nvfp4");
    case ModelKind::FlashNext: {
        const std::string named = recipe_of(model) + " " + lower(model.name);
        return contains_text(formats, "nvfp4") && !fp4_local(model) && !contains_text(named, "orcarouter") &&
               !contains_text(named, "uncensored");
    }
    default:
        return false;
    }
}

} // namespace

ModelKind model_kind(const ScannedModel& model) {
    if (model.model_type == "qwen3_5_text") { return ModelKind::Qwen35Dense; }
    if (model.model_type == "qwen3_5_moe_text") {
        return model.offloaded_experts ? ModelKind::Qwen35MoeOffloaded : ModelKind::Qwen35Moe;
    }
    if (model.model_type == "qwen4_exp_text") { return ModelKind::FlashNext; }
    if (model.model_type == "inkling2_loop_text") { return ModelKind::Inkling2; }
    return ModelKind::Unknown;
}

bool supports_option(const ScannedModel& model, std::string_view flag) {
    const ModelKind kind = model_kind(model);
    if (kind == ModelKind::Unknown) { return true; }
    if (flag == "--expert-a4-activations") { return kind == ModelKind::FlashNext && fp4_local(model); }
    if (flag == "--a16-activations") { return takes_a16(model); }
    if (flag == "--lm-head-draft") {
        // On Inkling its proposal/token_rank parameter, and a drafter it shortlists for (DSpark or a DFlash2 companion:
        // the Engine refuses the flag without one).
        return model.has_proposal && (kind != ModelKind::Inkling2 || model.has_dspark || model.has_draft_companion);
    }
    if (std::find(kNeedsVision.begin(), kNeedsVision.end(), flag) != kNeedsVision.end() && !model.has_vision) {
        return false;
    }
    const auto rule = std::find_if(kRules.begin(), kRules.end(), [&](const Rule& r) { return r.flag == flag; });
    return rule == kRules.end() || (rule->kinds & bit(kind)) != 0;
}

std::vector<std::string> spec_values(const ScannedModel& model) {
    std::vector<std::string> values = {"off", "auto"};
    if (model.has_mtp) { values.emplace_back("mtp"); }
    if (model.has_dflash) { values.emplace_back("dflash"); }
    // A drafter artifact is a DFlash2 companion except on the Qwen3.8-27B, whose is a DSpine drafter (drafter_fits).
    const bool dspine_target = model.family == "qwen3.8-27b";
    if (model.has_dflash2 || (model.accepts_draft_model && !dspine_target)) { values.emplace_back("dflash2"); }
    if (model.has_dspark) { values.emplace_back("dspark"); }
    if (model.accepts_draft_model && dspine_target) { values.emplace_back("dspine"); }
    return values;
}

bool supports_context_auto(const ScannedModel& model) {
    const ModelKind kind = model_kind(model);
    return kind == ModelKind::Unknown || (!model.offloaded_experts && kind != ModelKind::FlashNext);
}

std::vector<std::string> strip_unsupported_options(Args& args, const ScannedModel& model, const TakesValue& takes_value) {
    std::vector<std::string> removed;
    if (model_kind(model) == ModelKind::Unknown) { return removed; }
    const auto drop = [&](const std::string& flag) {
        remove_flag(args, flag, takes_value);
        removed.push_back(flag);
    };
    for (std::size_t i = 0; i < args.size();) {
        if (!args[i].starts_with("--")) {
            ++i;
            continue;
        }
        const std::string flag = args[i].substr(0, args[i].find('='));
        if (!supports_option(model, flag)) {
            drop(flag); // every occurrence goes, so the walk continues at the same index
        } else {
            ++i;
        }
    }
    if (const std::optional<std::string> spec = flag_value(args, "--spec")) {
        const std::vector<std::string> values = spec_values(model);
        if (std::find(values.begin(), values.end(), *spec) == values.end()) { drop("--spec"); }
    }
    if (flag_value(args, "--max-context").value_or("") == "auto" && !supports_context_auto(model)) {
        drop("--max-context");
    }
    return removed;
}

nlohmann::json option_support_json(const ScannedModel& model) {
    nlohmann::json unsupported = nlohmann::json::array();
    const auto note = [&](std::string_view flag) {
        if (!supports_option(model, flag)) { unsupported.push_back(std::string(flag)); }
    };
    for (const Rule& rule : kRules) { note(rule.flag); }
    for (const std::string_view flag : kNeedsVision) {
        if (std::find_if(kRules.begin(), kRules.end(), [&](const Rule& r) { return r.flag == flag; }) == kRules.end()) {
            note(flag);
        }
    }
    for (const std::string_view flag : {"--expert-a4-activations", "--a16-activations", "--lm-head-draft"}) { note(flag); }
    return {{"unsupported", unsupported}, {"spec_values", spec_values(model)}, {"context_auto", supports_context_auto(model)}};
}

} // namespace infernix::launcher
