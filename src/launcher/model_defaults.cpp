#include "launcher/model_defaults.h"

#include "launcher/option_support.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace infernix::launcher {
namespace {

// An NVFP4 conversion, by its recipe or its weight formats ("NVFP4 + FP8").
bool nvfp4_weights(const ScannedModel& model) {
    std::string text = model.recipe.value_or("") + " " + model.quantization.value_or("");
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text.find("nvfp4") != std::string::npos;
}

std::string stem_of(const ScannedModel& model) {
    std::string stem = path_utf8(model.path.stem());
    const std::size_t of = stem.rfind("-of-");
    if (of != std::string::npos && of >= 6 && stem[of - 6] == '-') { stem.resize(of - 6); }
    return stem;
}

} // namespace

const ScannedDrafter* best_drafter(const LibraryScan& scan, const ScannedModel& model) {
    const ScannedDrafter* best = nullptr;
    for (const ScannedDrafter& drafter : scan.drafters) {
        if (std::find(drafter.compatible_models.begin(), drafter.compatible_models.end(), model.key) ==
            drafter.compatible_models.end()) {
            continue;
        }
        // DSpine is an opt-in for long agent and coding sessions (4-10 % slower on short chats): never a default.
        if (drafter.kind == "dspine") { continue; }
        // Several copies of compatible drafters: the newest file.
        if (best == nullptr || drafter.modified > best->modified) { best = &drafter; }
    }
    return best;
}

const ScannedNgramVolume* best_ngram_volume(const LibraryScan& scan, const ScannedModel& model) {
    const ScannedNgramVolume* best = nullptr;
    int best_rank                  = 0;
    for (const ScannedNgramVolume& volume : scan.ngram_volumes) {
        const std::optional<int> rank = ngram_rank(model, volume);
        if (!rank) { continue; }
        if (best == nullptr || *rank < best_rank) {
            best      = &volume;
            best_rank = *rank;
        }
    }
    return best;
}

bool defaults_use_proposal_head(const ScannedModel& model) {
    // The proposal head pays off with DFlash/DFlash2, the small-model MTP drafters and Inkling 2's DSpark (its
    // shortlist decoded 2.4-5.8 % faster on nine prompts, the output unchanged), not on the 122B whose MTP rounds
    // measured no gain from it. On Inkling 2 the option needs the artifact's ranking and DSpark drafter. On the
    // Qwen3.8-27B NVFP4 conversions DFlash2 drafts from the screened full head instead (RV-41): against the proposal
    // head it decoded 1.7 % faster over code and sampled docs at one to eight requests on NVIDIA's NVFP4 and 1.8 % on
    // NVFP4-QAD (2026-11-27, README "Recommended settings per model").
    if (!model.has_proposal || model.family == "qwen3.5-122b-a10b") { return false; }
    return !(model.family == "qwen3.8-27b" && model.has_dflash2 && nvfp4_weights(model));
}

Args default_args(const LibraryScan& scan, const ScannedModel& model) {
    Args args;
    // Only what this model takes (option_support.h).
    const auto add = [&](std::string_view flag, std::string_view value = {}) {
        if (!supports_option(model, flag)) { return; }
        args.emplace_back(flag);
        if (!value.empty()) { args.emplace_back(value); }
    };
    const bool inkling               = model_kind(model) == ModelKind::Inkling2;
    const bool has_drafter_component = model.has_mtp || model.has_dflash || model.has_dflash2;

    if (model.family == "qwen3.8-flash-next") {
        add("--max-context", "262144");
    } else if (model.family == "qwen3.8-27b" || model.family == "qwen3.6-27b" || inkling) {
        // Every KV page stays on the GPU: infernix-serve sizes the context at startup to the largest the KV
        // cache holds beside the weights and the headroom (2026-11-14, a headless 32 GB RTX 5090 with the 22 GB
        // NVIDIA NVFP4 artifact, INT8 KV, two lanes, DFlash2: 256,896 tokens, the whole KV pool). Inkling 2's lanes
        // share its pool the same way, and a fixed default could exceed the card.
        add("--max-context", "auto");
    } else if (model.native_context != 0) {
        add("--max-context", std::to_string(std::min<std::uint64_t>(model.native_context, 262144)));
    }
    add("--max-concurrency", "2");
    add("--kv-dtype", "int8");

    // Speculative decoding: the Engine picks the drafter measured fastest for the artifact.
    // Inkling 2's DFlash2 companion only for an artifact without the bundled DSpark drafter, which won the agentic replay
    // against it (2026-11-26, docs/inkling2-mini.md); the user may still choose it (--draft-model).
    const ScannedDrafter* drafter =
        model.accepts_draft_model && !(inkling && model.has_dspark) ? best_drafter(scan, model) : nullptr;
    // Inkling 2: auto drafts with DSpark when the artifact bundles it, else with a companion, else plainly.
    if (has_drafter_component || drafter != nullptr || inkling) {
        add("--spec", "auto");
        add("--draft-tokens", "auto");
        if (defaults_use_proposal_head(model)) { add("--lm-head-draft"); }
    }
    if (drafter != nullptr) { add("--draft-model", path_utf8(drafter->path)); }
    if (model.family == "qwen3.8-27b" && model.has_dflash2) { add("--ngram-draft-tokens", "auto"); }

    if (model.needs_ngram_volume && !model.own_ngram_volume) {
        if (const ScannedNgramVolume* volume = best_ngram_volume(scan, model)) {
            add("--ngram-volume", path_utf8(volume->path));
        }
    }

    add("--preserve-thinking");
    add("--tolerant-tool-calls");
    // Agent clients queue several requests; let them wait for a lane instead of failing.
    add("--pending-timeout-ms", "900000");
    return args;
}

std::string default_alias(const LibraryScan& scan, const ScannedModel& model) {
    const std::string base = stem_of(model);
    int index              = 1;
    for (const ScannedModel& other : scan.models) {
        if (&other == &model || other.key == model.key) { break; }
        if (iequals(stem_of(other), base)) { ++index; }
    }
    return index == 1 ? base : base + "-" + std::to_string(index);
}

} // namespace infernix::launcher
