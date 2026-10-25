#pragma once

#include "infernix/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace infernix::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::filesystem::path ngram_volume_path;
    NgramResidency ngram_residency = NgramResidency::Auto;
    // --expert-state FILE|off: absent uses the artifact path + .expert-state, "off" disables.
    std::optional<std::filesystem::path> expert_state;
    // --expert-stripe FILE: the experts' record tails on a second drive (EngineOptions::expert_stripe_path).
    std::filesystem::path expert_stripe_path;
    std::uint64_t ram_headroom_bytes = kDefaultRamHeadroomBytes;
    std::optional<std::uint64_t> expert_ram_bytes; // --expert-ram-mib; empty: the RAM ledger's share
    std::optional<std::uint64_t> expert_pin_bytes; // --expert-pin-mib; empty: what the OS lets it pin
    std::filesystem::path grammar_path;
    std::filesystem::path json_schema_path;
    std::optional<std::string> regex;
    std::vector<std::string> choices;
    bool json_object = false;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    float rope_yarn_factor       = 1.0F;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::optional<std::size_t> vram_headroom_bytes; // empty: the model's automatic headroom
    bool vram_past_budget = false;
    std::uint32_t expert_tail_skip = 0; // EngineOptions::expert_tail_skip
    bool expert_a4_activations     = false; // EngineOptions::expert_a4_activations
    bool embedding_host            = false; // EngineOptions::embedding_host
    std::uint32_t prefill_chunk  = 0; // 0: the model's default (EngineOptions)
    int device                   = 0;

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    LogitsDtype logits_dtype = LogitsDtype::FP32;
    // INT8 KV prefills with the fast prompt kernel unless the original kernel is selected.
    bool original_int8_prefill_kernel = false;
    std::uint32_t prefill_split_workspace_mib = kDefaultPrefillSplitWorkspaceMiB;
    bool original_nvfp4_prefill_kernel = false;
    // Off with --reproducible or --lossless (EngineOptions::fast_paths).
    bool fast_paths = true;
    // --lossless (EngineOptions::lossless).
    bool lossless = false;
    // --fast-path NAME (repeatable): also run these, qualified or not (qualification runs;
    // EngineOptions::fast_path_candidates).
    std::vector<FastPath> fast_path_candidates;
    SpeculativeOptions speculative;
    bool enable_vision                     = false;
    VisionOffload vision_offload                    = VisionOffload::Auto;
    std::uint32_t vision_max_merged_tokens = 32768;
    bool use_cuda_graph                    = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    std::optional<bool> log_colours; // --log-colours on|off (unset = on when stderr is a terminal)
    std::optional<bool> enable_thinking;
    std::optional<std::uint32_t> thinking_budget;
    infernix::ReasoningLoopAction reasoning_loop = infernix::ReasoningLoopAction::Off;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy                 = false;
    product::LogLevel log_level = product::LogLevel::Info;

    // --config FILE and the options it gave (provenance for the startup log).
    std::optional<std::string> config_path;
    std::vector<std::string> config_arguments;
};

// argv[1] is the model path and the rest are options. `--config FILE` reads further options, and the
// model path when the command line has none, from a TOML file (product/config/config_file.h);
// options on the command line override the file.
[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace infernix::cli
