#pragma once

#include "ninfer/types.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::filesystem::path ngram_volume_path;
    // --expert-state FILE|off: absent uses the artifact path + .expert-state, "off" disables.
    std::optional<std::filesystem::path> expert_state;
    std::uint64_t ram_headroom_bytes = kDefaultRamHeadroomBytes;
    std::optional<std::uint64_t> expert_ram_bytes; // --expert-ram-mib; empty: the RAM ledger's share
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    float rope_yarn_factor       = 1.0F;
    std::uint32_t max_context    = 2048;
    KvCapacityPolicy kv_capacity = KvCapacityPolicy::explicit_capacity(2048);
    std::optional<std::size_t> vram_headroom_bytes; // empty: the model's automatic headroom
    bool vram_past_budget = false;
    std::uint32_t prefill_chunk  = 1024;
    int device                   = 0;

    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    // INT8 KV prefills with the fast prompt kernel unless the original kernel is selected.
    bool original_int8_prefill_kernel = false;
    PrefillPv8 prefill_8bit_pv         = PrefillPv8::Auto;
    std::uint32_t prefill_split_workspace_mib = kDefaultPrefillSplitWorkspaceMiB;
    bool original_nvfp4_prefill_kernel = false;
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
    ninfer::ReasoningLoopAction reasoning_loop = ninfer::ReasoningLoopAction::Off;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy                 = false;
    product::LogLevel log_level = product::LogLevel::Info;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
