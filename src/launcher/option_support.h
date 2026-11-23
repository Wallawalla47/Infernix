#pragma once

// Which infernix-serve options each model type takes: the one owner of that question in the launcher. The Configure
// form reads it through the library's `option_support`, and the default arguments, the launch, the preview and the fit
// plan apply it, so they cannot disagree. The table follows what the Engine accepts, refuses or reads for each
// architecture (normalize_inkling2_options, validate_engine_options, normalize_engine_options, the Qwen4Exp checks and
// which code reads each option); an option no entry names applies to every model. A model type the launcher does not
// know is not restricted: infernix-serve is the one to say.

#include "launcher/args.h"
#include "launcher/artifact_scan.h"

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace infernix::launcher {

enum class ModelKind {
    Qwen35Dense,        // qwen3_5_text: the 27Bs
    Qwen35Moe,          // qwen3_5_moe_text with resident experts: the 35B-A3B
    Qwen35MoeOffloaded, // qwen3_5_moe_text with offloaded experts: the 122B-A10B
    FlashNext,          // qwen4_exp_text
    Inkling2,           // inkling2_loop_text
    Unknown,
};

[[nodiscard]] ModelKind model_kind(const ScannedModel& model);

// Whether this model takes the option at all (its type, and what the artifact holds: vision, a proposal head, ...).
[[nodiscard]] bool supports_option(const ScannedModel& model, std::string_view flag);

// The --spec values this model takes (off and auto always; each drafter it bundles or, on Flash-Next, can be given).
[[nodiscard]] std::vector<std::string> spec_values(const ScannedModel& model);

// Whether this model takes --max-context auto (offloaded experts share the VRAM the KV pool would take).
[[nodiscard]] bool supports_context_auto(const ScannedModel& model);

// Removes every option this model does not take, and a --spec or --max-context value it refuses (the server's default
// then applies). Returns the removed options, each once and in order.
std::vector<std::string> strip_unsupported_options(Args& args, const ScannedModel& model,
                                                   const TakesValue& takes_value = {});

// The form's view: {unsupported: [flags of the table this model lacks], spec_values: [...], context_auto: bool}.
[[nodiscard]] nlohmann::json option_support_json(const ScannedModel& model);

} // namespace infernix::launcher
