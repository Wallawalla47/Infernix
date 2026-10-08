#include "serve/serve_options.h"
#include "serve/translate.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace infernix::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures       = 0;
    failures += check(parse({"infernix-serve", "model.ninfer"}).rope_yarn_factor == 1.0F,
                      "serving YaRN must default off");
    for (const auto* factor : {"1", "2.5", "4"}) {
        const auto yarn = parse({"infernix-serve", "model.ninfer", "--rope-yarn-factor", factor});
        failures += check(yarn.rope_yarn_factor == std::stof(factor) && yarn.max_context == 8192 &&
                              yarn.kv_capacity.mode == infernix::KvCapacityMode::Automatic &&
                              !yarn.vram_headroom_bytes,
                          "YaRN must not grow serving context or KV defaults");
    }
    for (const auto* factor : {"0", "0.99", "4.01", "-1", "nan", "inf", "-inf", "1e999", "2x", ""}) {
        bool rejected = false;
        try {
            (void)parse({"infernix-serve", "model.ninfer", "--rope-yarn-factor", factor});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid serving YaRN factor accepted");
    }
    bool missing_yarn_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--rope-yarn-factor"});
    } catch (const std::invalid_argument&) { missing_yarn_rejected = true; }
    failures += check(missing_yarn_rejected, "missing serving YaRN factor accepted");
    failures += check(serve_usage_text("infernix-serve").find("--rope-yarn-factor") != std::string::npos,
                      "serving help omits YaRN");
    const auto archive = parse({"infernix-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens",
                                "5", "--ngram-draft-tokens", "63", "--ngram-archive-mib", "512",
                                "--ngram-session-mib", "128", "--ngram-native-sessions"});
    failures += check(archive.speculative.ngram_archive_bytes == (512ULL << 20) &&
                          archive.speculative.ngram_session_bytes == (128ULL << 20) &&
                          archive.ngram_native_sessions,
                      "archive capacities or native opt-in not preserved");
    for (const auto& extra : std::vector<std::vector<std::string>>{
             {"--ngram-native-sessions"},
             {"--ngram-archive-mib", "64"},
             {"--ngram-archive-mib", "512", "--ngram-session-mib", "0"},
             {"--ngram-archive-mib", "18446744073709551615"}}) {
        std::vector<std::string> args{"infernix-serve",
                                      "model.ninfer",
                                      "--spec",
                                      "mtp",
                                      "--draft-tokens",
                                      "5",
                                      "--ngram-draft-tokens",
                                      "63"};
        args.insert(args.end(), extra.begin(), extra.end());
        bool rejected = false;
        try {
            (void)parse(args);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "invalid archive contract admitted");
    }
    const auto ngram = parse({"infernix-serve", "model.ninfer", "--spec", "dflash2", "--draft-tokens",
                              "5", "--ngram-draft-tokens", "15", "--ngram-min-match", "12"});
    failures +=
        check(ngram.speculative.ngram_draft_tokens == 15 && ngram.speculative.draft_tokens == 5 &&
                  ngram.speculative.ngram_min_match == 12,
              "ngram and neural widths were not kept separate");
    for (const auto& tail : std::vector<std::vector<std::string>>{{"--ngram-draft-tokens", "64"},
                                                                  {"--ngram-min-match", "3"},
                                                                  {"--spec", "none"}}) {
        std::vector<std::string> arguments{"infernix-serve",
                                           "model.ninfer",
                                           "--spec",
                                           "dflash2",
                                           "--draft-tokens",
                                           "5",
                                           "--ngram-draft-tokens",
                                           "15"};
        arguments.insert(arguments.end(), tail.begin(), tail.end());
        bool rejected = false;
        try {
            (void)parse(arguments);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "unsupported ngram configuration admitted");
    }
    // ngram with concurrency>1 is admitted for every backend at every width 1..63.
    for (const std::string backend : {"mtp", "dflash", "dflash2"}) {
        for (const std::string width : {"15", "31", "63"}) {
            const auto concurrent =
                parse({"infernix-serve", "model.ninfer", "--spec", backend, "--draft-tokens", "3",
                       "--ngram-draft-tokens", width, "--max-concurrency", "4"});
            failures += check(concurrent.speculative.ngram_draft_tokens ==
                                      static_cast<std::uint32_t>(std::stoul(width)) &&
                                  concurrent.max_concurrency == 4,
                              "ngram with concurrency>1 was not admitted");
        }
    }

    const auto mtp_ngram = parse({"infernix-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens",
                                  "3", "--ngram-draft-tokens", "15"});
    failures += check(mtp_ngram.speculative.draft_tokens == 3 &&
                          mtp_ngram.speculative.ngram_draft_tokens == 15,
                      "MTP and ngram widths were not kept separate");
    for (const std::string backend : {"mtp", "dflash", "dflash2"}) {
        const int neural_limit = backend == "mtp" ? 5 : 15;
        for (int neural = 1; neural <= neural_limit; ++neural) {
            for (int lookup = 1; lookup <= 63; ++lookup) {
                const auto options =
                    parse({"infernix-serve", "model.ninfer", "--spec", backend, "--draft-tokens",
                           std::to_string(neural), "--ngram-draft-tokens", std::to_string(lookup)});
                failures += check(options.speculative.draft_tokens == neural &&
                                      options.speculative.ngram_draft_tokens == lookup,
                                  "valid neural/ngram width pair changed");
            }
        }
        for (const auto& tail : std::vector<std::vector<std::string>>{
                 {"--draft-tokens", std::to_string(neural_limit + 1)},
                 {"--ngram-draft-tokens", "-1"},
                 {"--ngram-draft-tokens", "64"},
                 {"--ngram-min-match", "65"}}) {
            std::vector<std::string> arguments{"infernix-serve",
                                               "model.ninfer",
                                               "--spec",
                                               backend,
                                               "--draft-tokens",
                                               "3",
                                               "--ngram-draft-tokens",
                                               "15"};
            arguments.insert(arguments.end(), tail.begin(), tail.end());
            bool rejected = false;
            try {
                (void)parse(arguments);
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, "unsupported neural/ngram pair admitted");
        }
        const auto wide_single =
            parse({"infernix-serve", "model.ninfer", "--spec", backend, "--draft-tokens", "3",
                   "--ngram-draft-tokens", "63", "--max-concurrency", "1"});
        failures += check(wide_single.speculative.ngram_draft_tokens == 63,
                          "wide ngram rejected at concurrency one");
        const auto disabled =
            parse({"infernix-serve", "model.ninfer", "--spec", backend, "--draft-tokens", "3",
                   "--ngram-draft-tokens", "0", "--max-concurrency", "8"});
        failures += check(disabled.speculative.ngram_draft_tokens == 0,
                          "disabled ngram changed multi-slot configuration");
    }
    // DFlash2 tree table: entry c applies to rounds of c rows and the last entry to larger batches.
    using TreeTable  = std::array<std::uint32_t, infernix::kMaximumConcurrency>;
    const auto table = parse({"infernix-serve", "model.ninfer", "--spec", "dflash2", "--draft-tokens",
                              "7", "--draft-tree-nodes", "20,12,10,0", "--max-concurrency", "8"});
    failures += check(table.speculative.draft_tree_nodes == TreeTable{20, 12, 10, 0, 0, 0, 0, 0} &&
                          table.speculative.draft_tree_paths == 8,
                      "tree table or default path cap changed");
    const auto single =
        parse({"infernix-serve", "model.ninfer", "--spec", "dflash2", "--draft-tokens", "7",
               "--draft-tree-nodes", "9", "--draft-tree-paths", "2"});
    failures += check(single.speculative.draft_tree_nodes == TreeTable{9, 9, 9, 9, 9, 9, 9, 9} &&
                          single.speculative.draft_tree_paths == 2,
                      "a single tree entry did not apply to every batch size");
    const auto tree_auto = parse({"infernix-serve", "model.ninfer", "--spec", "dflash2",
                                  "--draft-tokens", "7", "--draft-tree-nodes", "auto"});
    failures += check(tree_auto.speculative.draft_tree_auto &&
                          tree_auto.speculative.draft_tree_nodes == TreeTable{},
                      "auto did not select automatic tree widths");
    const auto replaced =
        parse({"infernix-serve", "model.ninfer", "--spec", "dflash2", "--draft-tokens", "7",
               "--draft-tree-nodes", "auto", "--draft-tree-nodes", "12"});
    failures += check(!replaced.speculative.draft_tree_auto &&
                          replaced.speculative.draft_tree_nodes ==
                              TreeTable{12, 12, 12, 12, 12, 12, 12, 12},
                      "a later tree table did not replace auto");
    for (const auto& tail : std::vector<std::vector<std::string>>{
             {"--draft-tree-nodes", "8"}, // below draft tokens + 2
             {"--draft-tree-nodes", "33"},
             {"--draft-tree-nodes", "12,,0"},
             {"--draft-tree-nodes", "12,12,12,12,12,12,12,12,12"},
             {"--draft-tree-nodes", "12x"},
             {"--draft-tree-nodes", "12", "--draft-tree-paths", "1"},
             {"--draft-tree-nodes", "12", "--draft-tree-paths", "9"},
             {"--draft-tree-nodes", "12", "--spec", "dflash"},
             {"--draft-tree-nodes", "auto", "--spec", "dflash"},
             {"--draft-tree-nodes", "Auto"}}) {
        std::vector<std::string> arguments{"infernix-serve", "model.ninfer",   "--spec",
                                           "dflash2",      "--draft-tokens", "7"};
        arguments.insert(arguments.end(), tail.begin(), tail.end());
        bool rejected = false;
        try {
            (void)parse(arguments);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "unsupported tree table admitted");
    }
    const ServeOptions defaults = parse({"infernix-serve", "model.ninfer"});
    failures += check(defaults.speculative.ngram_draft_tokens == 0,
                      "ngram is unexpectedly enabled by default");
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.vision_offload == infernix::VisionOffload::Auto, "Vision offload is not auto by default");
    failures += check(defaults.vision_max_merged_tokens == 32768,
                      "merged Vision token default mismatch");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.context_cost_presets.empty(),
                      "external context-cost presets are unexpectedly configured by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.media_cache_bytes == infernix::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == infernix::kDefaultMediaLiveBytes &&
                          defaults.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.context_cache.enabled &&
                          defaults.context_cache.mode == infernix::ContextCacheMode::Hybrid,
                      "the hybrid prefix cache is not the default");
    failures += check(defaults.kv_capacity.mode == infernix::KvCapacityMode::Automatic &&
                          !defaults.vram_headroom_bytes,
                      "default KV capacity is not sized to free VRAM for the hybrid cache");
    failures += check(!defaults.context_cache.host_capacity_bytes.has_value(),
                      "default Host context capacity was resolved before Engine startup");
    failures += check(defaults.speculative.backend == infernix::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.default_thinking_budget,
                      "thinking budget is unexpectedly limited by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions fp8 = parse({"infernix-serve", "model.ninfer", "--kv-dtype", "fp8"});
    failures += check(fp8.kv_cache == infernix::KvCacheStorage::Fp8E4M3Row256,
                      "--kv-dtype fp8 did not select row-scaled E4M3 KV");
    const ServeOptions nvfp4 = parse({"infernix-serve", "model.ninfer", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == infernix::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ServeOptions k8v4 = parse({"infernix-serve", "model.ninfer", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == infernix::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    failures += check(parse({"infernix-serve", "model.ninfer", "--kv-dtype", "vq2"}).kv_cache ==
                          infernix::KvCacheStorage::Vq2,
                      "--kv-dtype vq2 did not select 2-bit vector KV");
    failures += check(parse({"infernix-serve", "model.ninfer", "--kv-dtype", "k4v2"}).kv_cache ==
                          infernix::KvCacheStorage::Q4KeyVq2Value,
                      "--kv-dtype k4v2 did not select 4-bit K / 2-bit V KV");
    const std::string kv_help = serve_usage_text("infernix-serve");
    failures += check(kv_help.find("nvfp4") != std::string::npos &&
                          kv_help.find("k8v4") != std::string::npos &&
                          kv_help.find("vq2") != std::string::npos &&
                          kv_help.find("k4v2") != std::string::npos,
                      "serve help omits a production KV storage mode");

    const ServeOptions model_alias =
        parse({"infernix-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    const ServeOptions context_cost =
        parse({"infernix-serve", "model.ninfer", "--context-cost-presets", "local-costs.json"});
    failures += check(context_cost.context_cost_presets == "local-costs.json",
                      "--context-cost-presets did not preserve its path");

    const ServeOptions thinking_budget =
        parse({"infernix-serve", "model.ninfer", "--default-thinking-budget", "37"});
    failures += check(thinking_budget.default_thinking_budget == 37,
                      "--default-thinking-budget did not preserve its positive value");
    bool zero_thinking_budget_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--default-thinking-budget", "0"});
    } catch (const std::invalid_argument&) { zero_thinking_budget_rejected = true; }
    failures += check(zero_thinking_budget_rejected, "zero --default-thinking-budget was accepted");
    failures += check(defaults.thinking_budget_message.empty(),
                      "thinking budget message is unexpectedly configured by default");
    const ServeOptions thinking_message =
        parse({"infernix-serve", "model.ninfer", "--thinking-budget-message",
               "Time to stop thinking. I must act now:"});
    failures += check(thinking_message.thinking_budget_message ==
                          "Time to stop thinking. I must act now:",
                      "--thinking-budget-message did not preserve its value");
    bool empty_thinking_message_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--thinking-budget-message", ""});
    } catch (const std::invalid_argument&) { empty_thinking_message_rejected = true; }
    failures += check(empty_thinking_message_rejected, "empty --thinking-budget-message accepted");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"infernix-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == infernix::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == infernix::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto options = parse({"infernix-serve", "model.ninfer", "--spec", "dflash2",
                                    "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures += check(options.speculative.backend == infernix::SpeculativeBackend::DFlash2 &&
                              options.speculative.draft_tokens == k &&
                              options.speculative.proposal_head == infernix::ProposalHead::Optimized,
                          "serve options did not preserve DFlash2 configuration");
    }

    const ServeOptions dflash_vision = parse(
        {"infernix-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "15", "--vision"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == infernix::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 15,
                      "serve options did not preserve combined DFlash and Vision features");

    const ServeOptions offload = parse({"infernix-serve", "model.ninfer", "--vision",
                                        "--vision-offload", "on", "--vision-max-merged", "512"});
    failures += check(offload.enable_vision && offload.vision_offload == infernix::VisionOffload::On,
                      "--vision-offload on did not reach serving options");
    failures += check(offload.vision_max_merged_tokens == 512,
                      "--vision-max-merged did not preserve its value");
    failures +=
        check(parse({"infernix-serve", "model.ninfer", "--vision-offload", "off"}).vision_offload ==
                  infernix::VisionOffload::Off,
              "--vision-offload off did not reach serving options");
    failures += check(parse({"infernix-serve", "model.ninfer", "--vision-offload", "auto"}).vision_offload ==
                          infernix::VisionOffload::Auto,
                      "--vision-offload auto (no --vision) is accepted");
    bool offload_without_vision_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--vision-offload", "on"});
    } catch (const std::invalid_argument&) { offload_without_vision_rejected = true; }
    failures +=
        check(offload_without_vision_rejected, "--vision-offload on without --vision was accepted");
    bool bad_offload_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--vision", "--vision-offload", "overlay"});
    } catch (const std::invalid_argument&) { bad_offload_rejected = true; }
    failures +=
        check(bad_offload_rejected, "--vision-offload accepted a value other than on or off");
    bool low_merged_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--vision-max-merged", "33"});
    } catch (const std::invalid_argument&) { low_merged_rejected = true; }
    failures += check(low_merged_rejected, "--vision-max-merged below 64 was accepted");
    bool high_merged_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--vision-max-merged", "40000"});
    } catch (const std::invalid_argument&) { high_merged_rejected = true; }
    failures += check(high_merged_rejected, "--vision-max-merged above 32768 was accepted");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse({"infernix-serve",
                                           "model.ninfer",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--usage-chunk-choice",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--log-stats-panel",
                                           "off",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(!configured.context_cache.enabled &&
                          configured.context_cache.host_capacity_bytes ==
                              defaults.context_cache.host_capacity_bytes &&
                          configured.context_cache.device_state_slots ==
                              defaults.context_cache.device_state_slots,
                      "--no-prefix-reuse changed context capacities or retained cache enablement");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures += check(configured.usage_chunk_choice,
                      "--usage-chunk-choice did not reach serving options");
    failures += check(configured.preserve_thinking == true,
                      "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == infernix::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(!configured.log_stats_panel &&
                          parse({"infernix-serve", "model.ninfer"}).log_stats_panel,
                      "--log-stats-panel did not reach serving options or is not on by default");
    bool invalid_panel_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--log-stats-panel", "yes"});
    } catch (const std::invalid_argument&) { invalid_panel_rejected = true; }
    failures +=
        check(invalid_panel_rejected, "--log-stats-panel accepted a value other than on|off");
    failures += check(configured.media_cache_bytes == (256ULL << 20) &&
                          configured.media_live_bytes == (512ULL << 20) &&
                          configured.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions logging = parse({"infernix-serve", "model.ninfer", "--log-level", "debug"});
    failures += check(logging.log_level == infernix::product::LogLevel::Debug,
                      "log level did not reach serving options");

    // The original prefix cache and its capacities.
    const ServeOptions original = parse({"infernix-serve", "model.ninfer", "--max-context", "16384",
                                         "--use-original-prefix-caching"});
    failures += check(original.context_cache.enabled &&
                          original.context_cache.mode == infernix::ContextCacheMode::Original &&
                          original.kv_capacity.mode == infernix::KvCapacityMode::Explicit &&
                          original.kv_capacity.explicit_tokens == 16384,
                      "the original prefix cache must keep the explicit max-context KV capacity");
    const ServeOptions context_cache =
        parse({"infernix-serve", "model.ninfer", "--use-original-prefix-caching",
               "--device-state-slots", "3", "--host-context-mib", "64"});
    failures += check(context_cache.context_cache.enabled &&
                          context_cache.context_cache.mode == infernix::ContextCacheMode::Original &&
                          context_cache.context_cache.device_state_slots == 3 &&
                          context_cache.context_cache.host_capacity_bytes == (64ULL << 20),
                      "context-cache capacities did not reach serving options");

    const ServeOptions zero_host_context =
        parse({"infernix-serve", "model.ninfer", "--host-context-mib", "0"});
    failures += check(zero_host_context.context_cache.enabled &&
                          zero_host_context.context_cache.host_capacity_bytes == 0,
                      "zero Host context capacity was not retained as an explicit budget");

    for (const auto& [mib, bytes] : std::vector<std::pair<std::string, std::size_t>>{
             {"146.822265625", 153954304},
             {"587.2890625", 615817216},
             {"186.822265625", 195897344},
             {"0.00000095367431640625", 1},
             {"0.000000953674316406250000", 1},
             {"000146.822265625000", 153954304},
             {"64.0000", 64ULL << 20},
             {"000.0000", 0},
         }) {
        const ServeOptions exact_host_context =
            parse({"infernix-serve", "model.ninfer", "--host-context-mib", mib});
        failures += check(exact_host_context.context_cache.host_capacity_bytes == bytes,
                          ("Host context MiB did not preserve exact bytes: " + mib).c_str());
    }

    // --vram-headroom-mib N|auto sets the headroom startup sizing leaves (Qwen3.5: after automatic
    // KV capacity, its Engine refusing it beside an explicit capacity; Qwen3.8-Flash-Next: after the
    // expert cache). 0 leaves none.
    failures += check(parse({"infernix-serve", "model.ninfer", "--vram-headroom-mib", "0"}).vram_headroom_bytes ==
                          std::optional<std::size_t>(0),
                      "--vram-headroom-mib 0 must leave no sizing headroom");

    constexpr std::size_t bytes_per_mib  = 1ULL << 20;
    constexpr std::size_t max_host_bytes = std::numeric_limits<std::size_t>::max();
    std::string maximum_host_mib         = std::to_string(max_host_bytes / bytes_per_mib);
    std::size_t fractional_bytes         = max_host_bytes % bytes_per_mib;
    if (fractional_bytes != 0) { maximum_host_mib += '.'; }
    while (fractional_bytes != 0) {
        fractional_bytes *= 10;
        maximum_host_mib += static_cast<char>('0' + fractional_bytes / bytes_per_mib);
        fractional_bytes %= bytes_per_mib;
    }
    const ServeOptions maximum_host_context =
        parse({"infernix-serve", "model.ninfer", "--host-context-mib", maximum_host_mib});
    failures += check(maximum_host_context.context_cache.host_capacity_bytes == max_host_bytes,
                      "maximum size_t Host context capacity was not accepted exactly");

    bool overflowing_host_context_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--host-context-mib",
                     std::to_string(std::numeric_limits<std::size_t>::max() / (1ULL << 20) + 1)});
    } catch (const std::invalid_argument&) { overflowing_host_context_rejected = true; }
    failures +=
        check(overflowing_host_context_rejected, "overflowing Host context capacity was accepted");

    for (const std::string mib : {"0.1", "0.0000001", "0.000000476837158203125", "-1", "-0", "+1",
                                  "nan", "inf", "1e3", "1.", ".5", "", " 1", "1 ", "1.2.3"}) {
        bool invalid_host_context_rejected = false;
        try {
            (void)parse({"infernix-serve", "model.ninfer", "--host-context-mib", mib});
        } catch (const std::invalid_argument&) { invalid_host_context_rejected = true; }
        failures +=
            check(invalid_host_context_rejected,
                  ("invalid or fractional-byte Host context MiB was accepted: " + mib).c_str());
    }

    for (const std::string option : {"--max-request-mib", "--media-cache-mib", "--media-live-mib",
                                     "--response-store-max-mib"}) {
        bool fractional_mib_rejected = false;
        try {
            (void)parse({"infernix-serve", "model.ninfer", option, "1.5"});
        } catch (const std::invalid_argument&) { fractional_mib_rejected = true; }
        failures += check(fractional_mib_rejected,
                          ("integer-only MiB option accepted a fraction: " + option).c_str());
    }

    for (const bool disable_first : {false, true}) {
        std::vector<std::string> arguments = {"infernix-serve", "model.ninfer"};
        if (disable_first) { arguments.push_back("--no-prefix-reuse"); }
        arguments.insert(arguments.end(),
                         {"--device-state-slots", "3", "--host-context-mib", "64"});
        if (!disable_first) { arguments.push_back("--no-prefix-reuse"); }
        const ServeOptions disabled_cache = parse(std::move(arguments));
        failures +=
            check(!disabled_cache.allow_prefix_reuse && !disabled_cache.context_cache.enabled &&
                      disabled_cache.context_cache.device_state_slots == 3 &&
                      disabled_cache.context_cache.host_capacity_bytes == (64ULL << 20),
                  "--no-prefix-reuse did not preserve independently configured capacities");
    }

    // Hybrid prefix cache capacities.
    const ServeOptions hybrid =
        parse({"infernix-serve", "model.ninfer", "--host-context-mib", "4096",
               "--device-snapshot-slots", "3", "--cache-taps-per-request", "5",
               "--cache-tap-ladder", "8192", "--cache-tap-min-gap", "512"});
    failures += check(hybrid.context_cache.enabled &&
                          hybrid.context_cache.mode == infernix::ContextCacheMode::Hybrid &&
                          hybrid.context_cache.host_capacity_bytes == (4096ULL << 20) &&
                          hybrid.context_cache.hybrid.device_snapshot_slots == 3 &&
                          hybrid.context_cache.hybrid.max_new_taps == 5 &&
                          hybrid.context_cache.hybrid.tap_ladder_tokens == 8192 &&
                          hybrid.context_cache.hybrid.tap_min_gap_tokens == 512,
                      "hybrid prefix-cache options did not reach serving options");
    // The default is a complete configuration: the KV pool defaults to free VRAM (the Device
    // block cache) and every hybrid tuning value is left for the Engine to derive.
    const ServeOptions hybrid_minimal = parse({"infernix-serve", "model.ninfer"});
    failures += check(hybrid_minimal.kv_capacity.mode == infernix::KvCapacityMode::Automatic &&
                          !hybrid_minimal.vram_headroom_bytes &&
                          !hybrid_minimal.context_cache.host_capacity_bytes &&
                          !hybrid_minimal.context_cache.hybrid.device_snapshot_slots &&
                          !hybrid_minimal.context_cache.hybrid.max_new_taps &&
                          !hybrid_minimal.context_cache.hybrid.tap_ladder_tokens &&
                          !hybrid_minimal.context_cache.hybrid.tap_min_gap_tokens &&
                          hybrid_minimal.context_cache.hybrid.queue_holds,
                      "the hybrid default must size KV automatically and leave tuning derived");
    failures += check(!parse({"infernix-serve", "model.ninfer", "--no-queue-holds"})
                           .context_cache.hybrid.queue_holds,
                      "--no-queue-holds did not turn queue holds off");
    const ServeOptions hybrid_headroom =
        parse({"infernix-serve", "model.ninfer", "--vram-headroom-mib", "2048"});
    failures += check(hybrid_headroom.kv_capacity.mode == infernix::KvCapacityMode::Automatic &&
                          hybrid_headroom.vram_headroom_bytes == (2048ULL << 20),
                      "hybrid automatic KV capacity must accept --vram-headroom-mib");
    // The headroom also applies to Qwen3.8-Flash-Next's expert cache, so the parser accepts it
    // with an explicit capacity (Qwen3.5's Engine refuses that combination) and accepts auto.
    const ServeOptions explicit_headroom =
        parse({"infernix-serve", "model.ninfer", "--kv-capacity", "8192", "--vram-headroom-mib", "512"});
    failures += check(explicit_headroom.kv_capacity.mode == infernix::KvCapacityMode::Explicit &&
                          explicit_headroom.vram_headroom_bytes == (512ULL << 20),
                      "--vram-headroom-mib must parse beside an explicit --kv-capacity");
    failures += check(!parse({"infernix-serve", "model.ninfer", "--vram-headroom-mib", "512",
                              "--vram-headroom-mib", "auto"})
                           .vram_headroom_bytes,
                      "--vram-headroom-mib auto must select the automatic headroom");
    failures += check(parse({"infernix-serve", "model.ninfer", "--expert-ram-mib", "32768"}).expert_ram_bytes == (32768ULL << 20) &&
                          !parse({"infernix-serve", "model.ninfer", "--expert-ram-mib", "32768", "--expert-ram-mib", "auto"})
                               .expert_ram_bytes &&
                          !parse({"infernix-serve", "model.ninfer"}).expert_ram_bytes,
                      "--expert-ram-mib must parse a MiB count and auto");
    for (const auto* bad : {"0", "-1", "12x", ""}) {
        bool rejected = false;
        try {
            (void)parse({"infernix-serve", "model.ninfer", "--expert-ram-mib", bad});
        } catch (const std::exception&) { rejected = true; }
        failures += check(rejected, "--expert-ram-mib must reject a zero or malformed size");
    }
    const ServeOptions hybrid_explicit_kv =
        parse({"infernix-serve", "model.ninfer", "--max-context", "8192", "--kv-capacity", "16384",
               "--host-context-mib", "0"});
    failures += check(hybrid_explicit_kv.kv_capacity.mode == infernix::KvCapacityMode::Explicit &&
                          hybrid_explicit_kv.kv_capacity.explicit_tokens == 16384 &&
                          hybrid_explicit_kv.context_cache.host_capacity_bytes == 0U,
                      "hybrid mode must keep an explicit KV capacity and a zero Host budget");
    for (const std::vector<std::string>& legacy_flag :
         std::vector<std::vector<std::string>>{{"--device-state-slots", "2"}}) {
        std::vector<std::string> arguments{"infernix-serve", "model.ninfer"};
        arguments.insert(arguments.end(), legacy_flag.begin(), legacy_flag.end());
        bool rejected = false;
        try {
            (void)parse(std::move(arguments));
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an original prefix-cache flag was accepted without the mode");
    }
    // The cache file resolves to an absolute path at launch, so the shutdown save writes where
    // startup read. A path with backslashes and a drive, as a Windows shell passes a quoted
    // argument, names the same file.
    const auto cache_file = [&](std::vector<std::string> extra) {
        std::vector<std::string> arguments{"infernix-serve", "model.ninfer"};
        arguments.insert(arguments.end(), extra.begin(), extra.end());
        return parse(std::move(arguments)).context_cache.hybrid.persistent_file;
    };
    failures +=
        check(cache_file({"--prefix-cache-file", "file.cache"}) ==
                  (std::filesystem::current_path() / "file.cache").lexically_normal(),
              "a relative --prefix-cache-file did not resolve against the launch directory");
    const std::filesystem::path absolute =
        (std::filesystem::temp_directory_path() / "infernix-serve-options.cache").lexically_normal();
    failures += check(cache_file({"--prefix-cache-file", absolute.string()}) == absolute,
                      "an absolute --prefix-cache-file was not kept as given");
    const auto rejected_cache_file = [&](std::vector<std::string> extra) {
        try {
            (void)cache_file(std::move(extra));
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    failures += check(rejected_cache_file({"--prefix-cache-file",
                                           (std::filesystem::temp_directory_path() /
                                            "infernix-missing-directory-for-test" / "file.cache")
                                               .string()}),
                      "--prefix-cache-file accepted a file in a directory that does not exist");
    failures += check(rejected_cache_file(
                          {"--prefix-cache-file", std::filesystem::temp_directory_path().string()}),
                      "--prefix-cache-file accepted a directory");
    failures +=
        check(rejected_cache_file({"--prefix-cache-file", "file.cache", "--host-context-mib", "0"}),
              "--prefix-cache-file was accepted without a Host tier to save");
    for (const std::vector<std::string>& hybrid_flag :
         std::vector<std::vector<std::string>>{{"--prefix-cache-file", "file.cache"},
                                               {"--device-snapshot-slots", "2"},
                                               {"--cache-taps-per-request", "2"},
                                               {"--cache-tap-ladder", "2048"},
                                               {"--cache-tap-min-gap", "64"},
                                               {"--no-queue-holds"}}) {
        for (const char* disabling : {"--use-original-prefix-caching", "--no-prefix-reuse"}) {
            std::vector<std::string> arguments{"infernix-serve", "model.ninfer", disabling};
            arguments.insert(arguments.end(), hybrid_flag.begin(), hybrid_flag.end());
            bool rejected = false;
            try {
                (void)parse(std::move(arguments));
            } catch (const std::invalid_argument&) { rejected = true; }
            failures += check(rejected, "a hybrid prefix-cache flag was accepted without the mode");
        }
    }
    // --no-prefix-reuse disables whichever cache is selected by default and keeps the KV pool at
    // --max-context; naming the original cache alongside it is contradictory.
    const ServeOptions without_reuse =
        parse({"infernix-serve", "model.ninfer", "--max-context", "16384", "--no-prefix-reuse"});
    failures += check(!without_reuse.context_cache.enabled &&
                          without_reuse.kv_capacity.mode == infernix::KvCapacityMode::Explicit &&
                          without_reuse.kv_capacity.explicit_tokens == 16384,
                      "--no-prefix-reuse did not disable the default prefix cache");
    bool original_without_reuse_rejected = false;
    try {
        (void)parse(
            {"infernix-serve", "model.ninfer", "--use-original-prefix-caching", "--no-prefix-reuse"});
    } catch (const std::invalid_argument&) { original_without_reuse_rejected = true; }
    failures += check(original_without_reuse_rejected,
                      "original prefix cache was accepted together with --no-prefix-reuse");

    const ServeOptions response_store =
        parse({"infernix-serve", "model.ninfer", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"infernix-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "20", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 20 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");
    bool oversized_top_k_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--top-k", "21"});
    } catch (const std::invalid_argument&) { oversized_top_k_rejected = true; }
    failures += check(oversized_top_k_rejected,
                      "server accepted top_k beyond the executable candidate domain");

    GenerationRequest request;
    request.max_tokens   = 1;
    const auto semantics = resolve_prompt_semantics(request, defaults);
    failures += check(!semantics.reasoning_effort && semantics.enable_thinking == true,
                      "omitted thinking did not resolve to the enabled template default");
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const infernix::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.allow_prefix_reuse)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking = resolve_prompt_semantics(request, thinking_budget);
    failures += check(!non_thinking.reasoning_effort,
                      "disabled thinking retained an effective reasoning effort");
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.allow_prefix_reuse)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    request.reasoning_effort   = RequestedReasoningEffort::Low;
    const auto explicit_effort = resolve_prompt_semantics(request, defaults);
    failures += check(explicit_effort.reasoning_effort == infernix::ReasoningEffort::Low &&
                          explicit_effort.enable_thinking == true,
                      "explicit reasoning effort did not remain the effective effort");
    request.reasoning_effort.reset();
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == true,
                      "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == false,
                      "request preserve-thinking override did not win");

    failures +=
        check(serve_usage_text("infernix-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures +=
        check(serve_usage_text("infernix-serve").find("--host-context-mib") != std::string::npos,
              "serve help omits context-cache capacities");
    failures += check(serve_usage_text("infernix-serve").find("--use-original-prefix-caching") !=
                          std::string::npos,
                      "serve help does not name the original prefix-cache selection");
    failures +=
        check(serve_usage_text("infernix-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("infernix-serve").find("--default-thinking-budget") !=
                          std::string::npos,
                      "serve help omits --default-thinking-budget");
    failures += check(serve_usage_text("infernix-serve").find("--thinking-budget-message") !=
                          std::string::npos,
                      "serve help omits --thinking-budget-message");
    failures += check(serve_usage_text("infernix-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures += check(serve_usage_text("infernix-serve").find("--usage-chunk-choice") !=
                          std::string::npos,
                      "serve help omits --usage-chunk-choice");
    failures +=
        check(serve_usage_text("infernix-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("infernix-serve").find("--log-stats-panel") != std::string::npos,
                      "serve help omits --log-stats-panel");
    failures += check(serve_usage_text("infernix-serve").find("--log-level") != std::string::npos,
                      "serve help omits the log-level control");
    failures += check(serve_usage_text("infernix-serve").find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(serve_usage_text("infernix-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures += check(serve_usage_text("infernix-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("infernix-serve").find("--context-cost-presets") != std::string::npos,
              "serve help omits external context-cost presets");
    failures += check(serve_usage_text("infernix-serve").find("metadata.name") != std::string::npos,
                      "serve help omits the artifact-derived model id default");

    const ServeOptions inherited = parse({"infernix-serve", "model.ninfer", "--max-context", "16384",
                                          "--use-original-prefix-caching"});
    failures += check(inherited.kv_capacity.mode == infernix::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");

    const ServeOptions automatic = parse({"infernix-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == infernix::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          !automatic.vram_headroom_bytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions logged = parse({"infernix-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("infernix-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    failures += check(logged.request_log_rotation.max_bytes == 0 &&
                          logged.request_log_rotation.keep == kDefaultRequestLogKeep,
                      "the request log must not rotate unless --request-log-max-mib is given");
    const ServeOptions rotating =
        parse({"infernix-serve", "model.ninfer", "--request-log-jsonl", "requests.jsonl",
               "--request-log-max-mib", "256", "--request-log-keep", "0"});
    failures += check(rotating.request_log_rotation.max_bytes == (256ULL << 20) &&
                          rotating.request_log_rotation.keep == 0,
                      "--request-log-max-mib/--request-log-keep did not set the rotation");
    for (const std::vector<std::string>& arguments : std::vector<std::vector<std::string>>{
             {"--request-log-jsonl", "requests.jsonl", "--request-log-max-mib", "0"},
             {"--request-log-jsonl", "requests.jsonl", "--request-log-max-mib", "-1"},
             {"--request-log-jsonl", "requests.jsonl", "--request-log-max-mib", "17592186044416"},
             {"--request-log-max-mib", "64"},
             {"--request-log-jsonl", "requests.jsonl", "--request-log-keep", "2"},
             {"--request-log-jsonl", "requests.jsonl", "--request-log-max-mib", "64",
              "--request-log-keep", "1001"},
             {"--request-log-jsonl", "requests.jsonl", "--request-log-max-mib", "64",
              "--request-log-keep", "two"}}) {
        std::vector<std::string> command{"infernix-serve", "model.ninfer"};
        command.insert(command.end(), arguments.begin(), arguments.end());
        bool rejected = false;
        try {
            (void)parse(command);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an invalid request-log rotation option was accepted");
    }
    failures +=
        check(serve_usage_text("infernix-serve").find("--request-log-max-mib") != std::string::npos &&
                  serve_usage_text("infernix-serve").find("--request-log-keep") != std::string::npos,
              "serve help omits the request-log rotation options");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    failures += check(!archive.original_int8_prefill_kernel,
                      "the original INT8 prefill kernel must default off");
    failures += check(parse({"infernix-serve", "model.ninfer", "--use-original-int8-prefill-kernel"})
                          .original_int8_prefill_kernel,
                      "--use-original-int8-prefill-kernel was not preserved");
    failures += check(serve_usage_text("infernix-serve").find("--use-original-int8-prefill-kernel") !=
                          std::string::npos,
                      "serve help omits --use-original-int8-prefill-kernel");
    failures += check(!archive.original_nvfp4_prefill_kernel,
                      "the original NVFP4 prefill kernel must default off");
    failures += check(parse({"infernix-serve", "model.ninfer", "--use-original-nvfp4-prefill-kernel"})
                          .original_nvfp4_prefill_kernel,
                      "--use-original-nvfp4-prefill-kernel was not preserved");
    failures += check(serve_usage_text("infernix-serve").find("--use-original-nvfp4-prefill-kernel") !=
                          std::string::npos,
                      "serve help omits --use-original-nvfp4-prefill-kernel");
    failures += check(archive.prefill_8bit_pv == infernix::PrefillPv8::Auto,
                      "8-bit prefill P*V must default to auto");
    failures += check(parse({"infernix-serve", "model.ninfer", "--no-prefill-8bit-pv"}).prefill_8bit_pv ==
                          infernix::PrefillPv8::Off,
                      "--no-prefill-8bit-pv was not preserved");
    failures += check(parse({"infernix-serve", "model.ninfer", "--prefill-8bit-pv"}).prefill_8bit_pv ==
                          infernix::PrefillPv8::On,
                      "--prefill-8bit-pv was not preserved");
    failures += check(serve_usage_text("infernix-serve").find("--no-prefill-8bit-pv") !=
                              std::string::npos &&
                          serve_usage_text("infernix-serve").find("--prefill-8bit-pv") !=
                              std::string::npos,
                      "serve help omits the 8-bit prefill P*V flags");
    failures += check(archive.prefill_split_workspace_mib == infernix::kDefaultPrefillSplitWorkspaceMiB,
                      "the prompt split workspace must default to its product default");
    failures += check(parse({"infernix-serve", "model.ninfer", "--prefill-split-workspace-mib", "0"})
                              .prefill_split_workspace_mib == 0,
                      "--prefill-split-workspace-mib 0 was not preserved");
    failures += check(serve_usage_text("infernix-serve").find("--prefill-split-workspace-mib") !=
                          std::string::npos,
                      "serve help omits --prefill-split-workspace-mib");
    bool split_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--prefill-split-workspace-mib", "16385"});
    } catch (const std::invalid_argument&) { split_rejected = true; }
    failures += check(split_rejected, "a prompt split workspace above 16384 MiB was accepted");
    bool fast_flag_rejected = false;
    try {
        (void)parse({"infernix-serve", "model.ninfer", "--fast-prefill-kernel"});
    } catch (const std::invalid_argument&) { fast_flag_rejected = true; }
    failures += check(fast_flag_rejected, "the removed --fast-prefill-kernel was accepted");

    // --expert-state: absent keeps the default (the artifact path + .expert-state), "off" disables.
    failures += check(!parse({"infernix-serve", "model.ninfer"}).expert_state.has_value(),
                      "serve expert state must default to the artifact's file");
    const auto state_off = parse({"infernix-serve", "model.ninfer", "--expert-state", "off"});
    failures += check(state_off.expert_state.has_value() && state_off.expert_state->empty(),
                      "serve --expert-state off must disable the saved state");
    const auto state_file = parse({"infernix-serve", "model.ninfer", "--expert-state", "cache.state"});
    failures += check(state_file.expert_state && *state_file.expert_state == "cache.state",
                      "serve --expert-state FILE must name the file");
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
