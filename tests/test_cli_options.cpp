#include "test_temp_path.h"
#include "options.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Replaces the scratch --config file's contents and returns its path.
std::string write_config(const std::string& text) {
    const std::filesystem::path path = infernix::test::unique_temp_path("infernix-cli-options-config.toml");
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << text;
    return path.string();
}

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
    int failures      = 0;
    const auto choice = parse({"infernix", "model.ninfer", "--prompt", "hello", "--choice", "",
                               "--choice", "a|b", "--choice", "你好"});
    failures += check(choice.choices == std::vector<std::string>{"", "a|b", "你好"},
                      "choice flags changed literal text or order");
    failures +=
        check(parse({"infernix", "model.ninfer", "--prompt", "hello", "--regex", ""}).regex == "",
              "empty regex must be distinct from no regex");
    failures += check(rejects([] {
                          (void)parse({"infernix", "model.ninfer", "--prompt", "hello", "--regex", "",
                                       "--choice", "yes"});
                      }),
                      "regex and choice were accepted together");
    failures += check(rejects([] {
                          (void)parse({"infernix", "model.ninfer", "--prompt", "hello", "--regex",
                                       "a", "--json-object"});
                      }),
                      "regex and JSON were accepted together");
    failures += check(parse({"infernix", "model.ninfer", "--prompt", "hello", "--expert-tail-skip", "8"})
                              .expert_tail_skip == 8,
                      "--expert-tail-skip did not parse");
    failures += check(parse({"infernix", "model.ninfer", "--prompt", "hello", "--expert-a4-activations"})
                                  .expert_a4_activations &&
                          !parse({"infernix", "model.ninfer", "--prompt", "hello"}).expert_a4_activations,
                      "--expert-a4-activations did not parse (default off)");
    const auto gated = parse({"infernix", "model.ninfer", "--prompt", "hello", "--spec", "mtp",
                              "--draft-tokens", "3", "--spec-min-p", "0.7"});
    failures += check(gated.speculative.draft_min_p && *gated.speculative.draft_min_p == 0.7F,
                      "--spec-min-p lost its probability");
    failures += check(rejects([] {
                          (void)parse({"infernix", "model.ninfer", "--prompt", "hello", "--spec",
                                       "dflash", "--draft-tokens", "3", "--spec-min-p", "0.5"});
                      }),
                      "--spec-min-p was accepted without MTP");
    failures += check(rejects([] {
                          (void)parse({"infernix", "model.ninfer", "--prompt", "hello", "--spec", "mtp",
                                       "--draft-tokens", "3", "--spec-min-p", "2"});
                      }),
                      "--spec-min-p accepted a value above 1");
    failures += check(infernix::cli::usage_text("infernix").find("--spec-min-p") != std::string::npos,
                      "CLI help omits --spec-min-p");
    failures += check(
        parse({"infernix", "model.ninfer", "--prompt", "hello", "--grammar-file", "answer.gbnf"})
                .grammar_path == "answer.gbnf",
        "grammar file argument was lost");
    failures +=
        check(parse({"infernix", "model.ninfer", "--prompt", "hello", "--json-object"}).json_object,
              "JSON object CLI option lost");
    failures += check(
        parse({"infernix", "model.ninfer", "--prompt", "hello", "--json-schema-file", "answer.json"})
                .json_schema_path == "answer.json",
        "schema file CLI option lost");
    failures += check(rejects([] {
                          (void)parse({"infernix", "model.ninfer", "--prompt", "hello",
                                       "--json-object", "--grammar-file", "a.gbnf"});
                      }),
                      "conflicting CLI constraints accepted");
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
    // auto leaves the copy window to the Engine; it needs a neural drafter.
    const auto ngram_auto = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", "dflash2",
                                   "--draft-tokens", "auto", "--ngram-draft-tokens", "auto"});
    failures += check(ngram_auto.speculative.ngram_draft_tokens_auto &&
                          ngram_auto.speculative.ngram_draft_tokens == 0,
                      "CLI did not select the automatic ngram window");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x",
                                       "--ngram-draft-tokens", "auto"});
                      }),
                      "CLI admitted an automatic ngram window without a neural drafter");
    const auto mtp_ngram = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                  "--draft-tokens", "5", "--ngram-draft-tokens", "1"});
    failures += check(mtp_ngram.speculative.draft_tokens == 5 &&
                          mtp_ngram.speculative.ngram_draft_tokens == 1,
                      "CLI narrow ngram/wide MTP pair changed");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec",
                                       "mtp", "--draft-tokens", "8", "--ngram-draft-tokens", "15"});
                      }),
                      "CLI admitted unsupported MTP neural width");
    // auto leaves the count to the Engine (the model's most); a fixed N stays fixed.
    const auto mtp_auto = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                 "--draft-tokens", "auto"});
    failures += check(mtp_auto.speculative.draft_tokens_auto && mtp_auto.speculative.draft_tokens == 0,
                      "CLI did not select the automatic MTP draft length");
    const auto mtp_fixed = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                  "--draft-tokens", "7"});
    failures += check(!mtp_fixed.speculative.draft_tokens_auto && mtp_fixed.speculative.draft_tokens == 7,
                      "CLI did not keep a fixed MTP draft length of 7");
    // The block drafters' automatic verify width (DFlash and DFlash2), the ceiling left to the Engine.
    const auto dflash_auto = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", "dflash",
                                    "--draft-tokens", "auto"});
    failures += check(dflash_auto.speculative.draft_tokens_auto && dflash_auto.speculative.draft_tokens == 0,
                      "CLI did not select the automatic DFlash draft length");
    failures += check(rejects([&] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--spec", "mtp",
                                       "--draft-tokens", "Auto"});
                      }),
                      "CLI admitted --draft-tokens Auto");
    failures += check(rejects([&] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--draft-tokens", "auto"});
                      }),
                      "CLI admitted --draft-tokens auto without --spec");
    // Qwen3.8-Flash-Next's DFlash2 drafter: a companion artifact by path, and the automatic length.
    const auto companion = parse({"infernix-cli", "model.infernix", "--prompt", "x", "--spec", "dflash2",
                                  "--draft-tokens", "auto", "--draft-model", "drafter.infernix", "--lm-head-draft"});
    failures += check(companion.speculative.draft_model == std::filesystem::path("drafter.infernix") &&
                          companion.speculative.draft_tokens_auto &&
                          companion.speculative.proposal_head == infernix::ProposalHead::Optimized,
                      "CLI did not keep --draft-model, --draft-tokens auto and --lm-head-draft with dflash2");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.infernix", "--prompt", "x", "--spec", "mtp",
                                       "--draft-tokens", "3", "--draft-model", "drafter.infernix"});
                      }),
                      "CLI admitted --draft-model without --spec dflash2");
    failures += check(infernix::cli::usage_text("infernix-cli").find("--draft-model FILE") != std::string::npos,
                      "CLI help omits --draft-model");
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
    failures += check(k8v4.fast_paths, "the CLI fast paths must default on");
    failures += check(!parse({"infernix-cli", "model.ninfer", "--prompt", "hello", "--reproducible"})
                           .fast_paths,
                      "--reproducible was not parsed");
    failures += check(help.find("--reproducible") != std::string::npos,
                      "CLI help omits --reproducible");
    failures += check(help.find("prefill-8bit-pv") == std::string::npos,
                      "CLI help still lists the removed 8-bit prefill P*V flags");
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
    failures += check(help.find("--expert-pin-mib") != std::string::npos, "CLI help omits the expert pin control");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-pin-mib", "24576"}).expert_pin_bytes ==
                              (24576ULL << 20) &&
                          !parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-pin-mib", "auto"}).expert_pin_bytes,
                      "--expert-pin-mib must parse a MiB count and auto");
    failures += check(rejects([] { (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-pin-mib", "0"}); }),
                      "--expert-pin-mib must reject zero");
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
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "x"}).expert_stripe_path.empty(),
                      "CLI expert stripe must default to none");
    failures += check(parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--expert-stripe", "D:/m.stripe"})
                              .expert_stripe_path == "D:/m.stripe",
                      "CLI --expert-stripe FILE must name the file");
    // --fast-path NAME: a qualification candidate, repeatable; unknown names are refused.
    const auto candidate = parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--fast-path", "sampled-mtp-drafts",
                                  "--fast-path", "preemption"});
    failures += check(candidate.fast_path_candidates ==
                          std::vector<infernix::FastPath>{infernix::FastPath::SampledMtpDrafts,
                                                          infernix::FastPath::Preemption},
                      "CLI --fast-path must collect each named candidate");
    failures += check(rejects([] {
                          (void)parse({"infernix-cli", "model.ninfer", "--prompt", "x", "--fast-path", "no-such-path"});
                      }),
                      "CLI accepted an unknown fast path");

    // --config FILE: the model and options from a TOML file; [cli] overrides the top level, [serve]
    // is infernix-serve's and is not read here, repeatable options take arrays, and the command
    // line overrides the file.
    {
        const std::string config = write_config("model = \"from-file.infernix\"\n"
                                                "max-context = 4096\n"
                                                "kv-dtype = \"int8\"\n"
                                                "stop = [\"END\", \"STOP\"]\n"
                                                "choice = [\"yes\", \"no\"]\n"
                                                "greedy = true\n"
                                                "[cli]\n"
                                                "prompt = \"hello there\"\n"
                                                "max-context = 8192\n"
                                                "[serve]\n"
                                                "port = 9000\n");
        const auto from_file = parse({"infernix", "--config", config});
        failures += check(from_file.artifact_path == "from-file.infernix" && from_file.prompt == "hello there" &&
                              from_file.max_context == 8192 &&
                              from_file.kv_cache == infernix::KvCacheStorage::Int8Group64 &&
                              from_file.stop_strings.size() == 2 && from_file.stop_strings[1].text == "STOP" &&
                              from_file.choices == std::vector<std::string>{"yes", "no"} && from_file.greedy &&
                              from_file.config_path == config,
                          "CLI --config values did not reach the options");
        const auto example = parse({"infernix", "--config", std::string(INFERNIX_SOURCE_DIR) + "/infernix.example.toml"});
        failures += check(example.artifact_path.extension() == ".infernix" && example.max_context == 32768 &&
                              example.max_new == 1024 && !example.prompt.empty(),
                          "infernix.example.toml does not configure infernix as written");
        const auto overridden =
            parse({"infernix", "cmd.infernix", "--config", config, "--max-context", "1024", "--stop", "HALT"});
        failures += check(overridden.artifact_path == "cmd.infernix" && overridden.max_context == 1024 &&
                              overridden.stop_strings.size() == 3,
                          "the CLI command line did not override --config, or its repeated values replaced the file's");
        for (const auto& [text, expected] : std::vector<std::pair<std::string, std::string>>{
                 {"prompt = \"x\"\nport = 9000", "unknown option 'port'"},
                 {"prompt = \"x\"\nmax-new = [1, 2]", "'max-new' takes one value, not an array"},
                 {"prompt = \"x\"\n[cli]\nmodel = 1", "'model' takes the artifact path as a string"},
                 {"prompt = \"x\"", "a model path (.infernix or .ninfer) is required"}}) {
            std::string error;
            try {
                (void)parse({"infernix", "--config", write_config(text)});
            } catch (const std::exception& exception) { error = exception.what(); }
            failures += check(error.find(expected) != std::string::npos,
                              ("CLI --config did not reject \"" + text + "\" with: " + expected).c_str());
        }

        // Every option the help lists is a --config key with the command line's arity.
        const std::string help = infernix::cli::usage_text("infernix");
        std::set<std::string> listed;
        for (std::size_t at = help.find("--"); at != std::string::npos; at = help.find("--", at + 2)) {
            std::size_t end = at + 2;
            while (end < help.size() && ((help[end] >= 'a' && help[end] <= 'z') ||
                                         (help[end] >= '0' && help[end] <= '9') || help[end] == '-')) {
                ++end;
            }
            listed.insert(help.substr(at, end - at));
        }
        listed.erase("--config");
        failures += check(listed.contains("--prompt") && listed.contains("--greedy") && listed.contains("--stop"),
                          "the CLI help lists no options to compare with --config");
        for (const std::string& option : listed) {
            std::string direct;
            try {
                (void)parse({"infernix", "model.ninfer", option});
            } catch (const std::exception& exception) { direct = exception.what(); }
            const bool takes_value = direct.ends_with(option + " needs a value");
            const std::string key  = option.substr(2);
            std::string error;
            try {
                (void)parse({"infernix", "model.ninfer", "--config",
                             write_config(takes_value ? key + " = \"1\"\n" : key + " = true\n")});
            } catch (const std::exception& exception) { error = exception.what(); }
            failures += check(error.find("unknown option") == std::string::npos &&
                                  error.find("is a flag") == std::string::npos &&
                                  error.find("takes a string") == std::string::npos &&
                                  error.find("invalid TOML") == std::string::npos &&
                                  error.find("has no parser") == std::string::npos,
                              ("CLI --config does not accept the option " + option).c_str());
        }
        std::error_code ignored;
        std::filesystem::remove(config, ignored);
    }
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
