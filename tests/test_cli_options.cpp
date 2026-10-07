#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

infernix::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return infernix::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int run_tests() {
    int failures = 0;
    failures += check(infernix::EngineOptions{}.rope_yarn_factor == 1.0F,
                      "Engine YaRN must default off");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "x"}).rope_yarn_factor == 1.0F,
                      "CLI YaRN must default off");
    for (const auto* factor : {"1", "2.5", "4"}) {
        const auto yarn = parse({"infernix-cli", "model.ninfer", "--prompt", "x",
                                 "--rope-yarn-factor", factor});
        failures += check(yarn.rope_yarn_factor == std::stof(factor) && yarn.max_context == 2048 &&
                              yarn.kv_capacity.explicit_tokens == 2048,
                          "YaRN must preserve factor without growing default context or KV");
    }
    for (const auto* factor : {"0", "0.99", "4.01", "-1", "nan", "inf", "-inf", "1e999", "2x", ""}) {
        failures += check(rejects([&] {
            (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--rope-yarn-factor", factor});
        }), "invalid YaRN factor accepted");
    }
    failures += check(rejects([] {
        (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--rope-yarn-factor"});
    }), "missing YaRN factor accepted");
    failures += check(infernix::cli::usage_text("infernix").find("--rope-yarn-factor") != std::string::npos,
                      "CLI help omits YaRN");
    for (const auto* backend : {"mtp", "dflash", "dflash2"}) {
        for (unsigned width = 0; width <= 63; ++width) {
            const auto mixed =
                parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", backend,
                       "--draft-tokens", "5", "--ngram-draft-tokens", std::to_string(width)});
            failures += check(mixed.speculative.draft_tokens == 5 &&
                                  mixed.speculative.ngram_draft_tokens == width,
                              "CLI failed to preserve independent neural/ngram widths");
        }
        failures +=
            check(rejects([&] {
                      (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", backend,
                                   "--draft-tokens", "5", "--ngram-draft-tokens", "64"});
                  }),
                  "CLI admitted unsupported ngram width");
    }
    failures += check(infernix::cli::usage_text("infernix-cli").find("--ngram-draft-tokens 1..63") !=
                          std::string::npos,
                      "CLI help has a stale ngram width limit");
    failures +=
        check(parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--ngram-draft-tokens", "0"})
                      .speculative.ngram_draft_tokens == 0,
              "explicit ngram-off must not require a neural backend");
    const auto ngram = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", "dflash2",
                              "--draft-tokens", "5", "--ngram-draft-tokens", "15"});
    failures +=
        check(ngram.speculative.ngram_draft_tokens == 15 && ngram.speculative.draft_tokens == 5,
              "CLI ngram widths changed");
    const auto mtp_ngram = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                  "--draft-tokens", "5", "--ngram-draft-tokens", "1"});
    failures += check(mtp_ngram.speculative.draft_tokens == 5 &&
                          mtp_ngram.speculative.ngram_draft_tokens == 1,
                      "CLI narrow ngram/wide MTP pair changed");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec",
                                       "mtp", "--draft-tokens", "6", "--ngram-draft-tokens", "15"});
                      }),
                      "CLI admitted unsupported MTP neural width");
    const infernix::cli::Options configured =
        parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(infernix::cli::usage_text("infernix-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const infernix::cli::Options with_effort =
        parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const infernix::cli::Options dflash_vision =
        parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--vision", "--spec", "dflash",
               "--draft-tokens", "7"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == infernix::SpeculativeBackend::DFlash &&
                          dflash_vision.speculative.draft_tokens == 7,
                      "CLI did not preserve the combined DFlash and Vision startup features");
    const infernix::cli::Options offload =
        parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--vision", "--vision-offload",
               "on", "--vision-max-merged", "512"});
    failures += check(offload.enable_vision && offload.vision_offload == infernix::VisionOffload::On,
                      "--vision-offload on did not reach CLI options");
    failures += check(offload.vision_max_merged_tokens == 512,
                      "--vision-max-merged did not preserve its value");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--vision",
                             "--vision-offload", "on", "--vision-offload", "off"})
                              .vision_offload == infernix::VisionOffload::Off,
                      "--vision-offload off did not reach CLI options");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello"}).vision_offload ==
                          infernix::VisionOffload::Auto,
                      "--vision-offload does not default to auto");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--vision-offload", "off",
                             "--vision-offload", "auto"})
                              .vision_offload == infernix::VisionOffload::Auto,
                      "--vision-offload auto did not reach CLI options");
    failures +=
        check(infernix::cli::usage_text("infernix-cli").find("--vision-offload") != std::string::npos,
              "CLI help omits --vision-offload");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello",
                                       "--vision-offload", "overlay"});
                      }),
                      "--vision-offload accepted a value other than on or off");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello",
                                       "--vision-max-merged", "33"});
                      }),
                      "--vision-max-merged below 64 was accepted");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello",
                                       "--vision-max-merged", "40000"});
                      }),
                      "--vision-max-merged above 32768 was accepted");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == infernix::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const infernix::cli::Options nvfp4 =
        parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == infernix::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const infernix::cli::Options k8v4 =
        parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == infernix::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "vq2"})
                              .kv_cache == infernix::KvCacheStorage::Vq2,
                      "--kv-dtype vq2 did not select 2-bit vector KV");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k4v2"})
                              .kv_cache == infernix::KvCacheStorage::Q4KeyVq2Value,
                      "--kv-dtype k4v2 did not select 4-bit K / 2-bit V KV");
    const std::string help = infernix::cli::usage_text("infernix-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos &&
                  help.find("vq2") != std::string::npos && help.find("k4v2") != std::string::npos,
              "CLI help omits a production KV storage mode");
    failures += check(!k8v4.original_int8_prefill_kernel,
                      "the CLI original INT8 prefill kernel must default off");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype",
                             "int8", "--use-original-int8-prefill-kernel"})
                          .original_int8_prefill_kernel,
                      "--use-original-int8-prefill-kernel was not parsed");
    failures += check(help.find("--use-original-int8-prefill-kernel") != std::string::npos,
                      "CLI help omits --use-original-int8-prefill-kernel");
    failures += check(!k8v4.original_nvfp4_prefill_kernel,
                      "the CLI original NVFP4 prefill kernel must default off");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype",
                             "nvfp4", "--use-original-nvfp4-prefill-kernel"})
                          .original_nvfp4_prefill_kernel,
                      "--use-original-nvfp4-prefill-kernel was not parsed");
    failures += check(help.find("--use-original-nvfp4-prefill-kernel") != std::string::npos,
                      "CLI help omits --use-original-nvfp4-prefill-kernel");
    failures += check(k8v4.prefill_8bit_pv == infernix::PrefillPv8::Auto,
                      "the CLI 8-bit prefill P*V must default to auto");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype",
                             "k8v4", "--no-prefill-8bit-pv"})
                          .prefill_8bit_pv == infernix::PrefillPv8::Off,
                      "--no-prefill-8bit-pv was not parsed");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype",
                             "int8", "--prefill-8bit-pv"})
                          .prefill_8bit_pv == infernix::PrefillPv8::On,
                      "--prefill-8bit-pv was not parsed");
    failures += check(help.find("--no-prefill-8bit-pv") != std::string::npos &&
                          help.find("--prefill-8bit-pv") != std::string::npos,
                      "CLI help omits the 8-bit prefill P*V flags");
    failures +=
        check(k8v4.prefill_split_workspace_mib == infernix::kDefaultPrefillSplitWorkspaceMiB,
              "the CLI prompt split workspace must default to its product default");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "hello",
                             "--prefill-split-workspace-mib", "384"})
                              .prefill_split_workspace_mib == 384,
                      "--prefill-split-workspace-mib was not parsed");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello",
                                       "--prefill-split-workspace-mib", "16385"});
                      }),
                      "a CLI prompt split workspace above 16384 MiB was accepted");
    failures += check(help.find("--prefill-split-workspace-mib") != std::string::npos,
                      "CLI help omits --prefill-split-workspace-mib");
    const infernix::cli::Options logging =
        parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == infernix::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(help.find("--log-colours") != std::string::npos,
                      "CLI help omits the log-colours control");
    failures += check(help.find("--vram-headroom-mib") != std::string::npos,
                      "CLI help omits the VRAM headroom control");
    failures += check(help.find("--expert-ram-mib") != std::string::npos, "CLI help omits the expert RAM control");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-ram-mib", "40960"}).expert_ram_bytes ==
                              (40960ULL << 20) &&
                          !parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-ram-mib", "auto"}).expert_ram_bytes,
                      "--expert-ram-mib must parse a MiB count and auto");
    failures += check(rejects([] { (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-ram-mib", "0"}); }),
                      "--expert-ram-mib must reject zero");
    const char* const cli_help_sections[] = {"CONTEXT", "KV CACHE", "SPECULATIVE DECODING",
                                             "SAMPLING", "VISION", "LOGGING"};
    for (const char* section : cli_help_sections) {
        failures += check(help.find(section) != std::string::npos, "CLI help omits a category");
    }
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures +=
        check(rejects([] {
                  (void)parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    // --expert-state: absent keeps the default (the artifact path + .expert-state), "off" disables.
    failures += check(!parse({"infernix-cli", "model.ninfer", "--prompt", "x"}).expert_state.has_value(),
                      "CLI expert state must default to the artifact's file");
    const auto state_off = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-state", "off"});
    failures += check(state_off.expert_state.has_value() && state_off.expert_state->empty(),
                      "CLI --expert-state off must disable the saved state");
    const auto state_file = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-state", "cache.state"});
    failures += check(state_file.expert_state && *state_file.expert_state == "cache.state",
                      "CLI --expert-state FILE must name the file");
    return failures == 0 ? 0 : 1;
}

int main() {
    try {
        return run_tests();
    } catch (const std::exception& error) {
        std::cerr << "Unexpected CLI test exception: " << error.what() << '\n';
        return 1;
    }
}
