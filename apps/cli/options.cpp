#include "options.h"
#include "product/config/config_file.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace infernix::cli {
namespace {

std::uint64_t parse_u64(const char* text, std::string_view label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t parse_u32(const char* text, std::string_view label, bool allow_zero = false) {
    const std::uint64_t value = parse_u64(text, label);
    if ((!allow_zero && value == 0) || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<std::uint32_t>(value);
}

int parse_device(const char* text) {
    const std::uint64_t value = parse_u64(text, "device");
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid device: ") + text);
    }
    return static_cast<int>(value);
}

float parse_float(const char* text, std::string_view label, float minimum, float maximum) {
    errno              = 0;
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(value) ||
        value < static_cast<double>(minimum) || value > static_cast<double>(maximum)) {
        throw std::invalid_argument("invalid " + std::string(label) + ": " + text);
    }
    return static_cast<float>(value);
}

KvCacheStorage parse_kv_cache(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    if (text == "vq2") { return KvCacheStorage::Vq2; }
    if (text == "k4v2") { return KvCacheStorage::Q4KeyVq2Value; }
    throw std::invalid_argument("invalid kv-dtype: " + std::string(text));
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    return KvCapacityPolicy::explicit_capacity(parse_u32(text, "kv-capacity"));
}

// Every option parse_options accepts after the model path, whether it takes a value and whether it
// repeats. --config maps a file's keys through this table, so a config file accepts exactly the
// command line's options with the same arity.
using CliOption = product::ConfigOption;

constexpr std::array kCliOptions{
    CliOption{"--prompt"},
    CliOption{"--messages"},
    CliOption{"--chat-template"},
    CliOption{"--ngram-volume"},
    CliOption{"--ngram-residency"},
    CliOption{"--expert-state"},
    CliOption{"--expert-stripe"},
    CliOption{"--grammar-file"},
    CliOption{"--json-object", false},
    CliOption{"--json-schema-file"},
    CliOption{"--regex"},
    CliOption{"--choice", true, true},
    CliOption{"--max-new"},
    CliOption{"--rope-yarn-factor"},
    CliOption{"--max-context"},
    CliOption{"--kv-capacity"},
    CliOption{"--vram-past-budget", false},
    CliOption{"--expert-tail-skip"},
    CliOption{"--vram-headroom-mib"},
    CliOption{"--prefill-chunk"},
    CliOption{"--device"},
    CliOption{"--ram-headroom-mib"},
    CliOption{"--expert-ram-mib"},
    CliOption{"--expert-pin-mib"},
    CliOption{"--kv-dtype"},
    CliOption{"--spec"},
    CliOption{"--draft-tokens"},
    CliOption{"--draft-model"},
    CliOption{"--spec-min-p"},
    CliOption{"--ngram-draft-tokens"},
    CliOption{"--ngram-min-match"},
    CliOption{"--draft-tree-nodes"},
    CliOption{"--draft-tree-paths"},
    CliOption{"--lm-head-draft", false},
    CliOption{"--raw-output", false},
    CliOption{"--print-token-ids", false},
    CliOption{"--log-colours"},
    CliOption{"--no-thinking", false},
    CliOption{"--thinking-budget"},
    CliOption{"--reasoning-loop"},
    CliOption{"--reasoning-effort"},
    CliOption{"--vision", false},
    CliOption{"--vision-offload"},
    CliOption{"--vision-max-merged"},
    CliOption{"--no-cuda-graph", false},
    CliOption{"--use-original-int8-prefill-kernel", false},
    CliOption{"--prefill-split-workspace-mib"},
    CliOption{"--use-original-nvfp4-prefill-kernel", false},
    CliOption{"--reproducible", false},
    CliOption{"--lossless", false},
    CliOption{"--fast-path", true, true},
    CliOption{"--expert-a4-activations", false},
    CliOption{"--stop-token-id", true, true},
    CliOption{"--stop", true, true},
    CliOption{"--reasoning-stop", true, true},
    CliOption{"--temperature"},
    CliOption{"--top-p"},
    CliOption{"--top-k"},
    CliOption{"--min-p"},
    CliOption{"--presence-penalty"},
    CliOption{"--frequency-penalty"},
    CliOption{"--seed"},
    CliOption{"--greedy", false},
    CliOption{"--log-level"},
};

const CliOption* find_option(std::string_view name) noexcept {
    const auto found = std::find_if(kCliOptions.begin(), kCliOptions.end(),
                                    [&](const CliOption& option) { return option.name == name; });
    return found == kCliOptions.end() ? nullptr : &*found;
}

// infernix's --config: top-level keys and the [cli] table; [serve] is infernix-serve's.
const product::ConfigProgram& cli_config_program() {
    static const product::ConfigProgram program{
        .table = "cli", .other_table = "serve", .find_option = [](std::string_view name) { return find_option(name); }};
    return program;
}

ReasoningEffort parse_reasoning_effort(std::string_view text) {
    if (text == "none") { return ReasoningEffort::None; }
    if (text == "minimal") { return ReasoningEffort::Minimal; }
    if (text == "high") { return ReasoningEffort::High; }
    if (text == "max") { return ReasoningEffort::Max; }
    if (text == "low") { return ReasoningEffort::Low; }
    if (text == "medium") { return ReasoningEffort::Medium; }
    if (text == "xhigh") { return ReasoningEffort::XHigh; }
    throw std::invalid_argument("invalid reasoning-effort: " + std::string(text));
}

} // namespace

std::string usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.infernix> (--prompt <text>|--messages <messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "       [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2] [--spec mtp|dflash|dflash2 --draft-tokens "
           "N|auto]\n"
           "       [--draft-model FILE]\n"
           "       [--lm-head-draft] [--spec-min-p P] [--draft-tree-nodes auto|N] [--draft-tree-paths N]\n"
           "       [--ngram-draft-tokens 1..63|auto] [--ngram-min-match 4..64]\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--chat-template FILE] [--ngram-volume FILE] [--ngram-residency auto|ram|volume]\n"
           "       [--expert-state FILE|off] [--expert-stripe FILE] [--expert-a4-activations]\n"
           "       [--ram-headroom-mib N] [--expert-ram-mib N|auto] [--expert-pin-mib N|auto]\n"
           "       [--grammar-file FILE | --json-object | --json-schema-file FILE |\n"
           "        --regex PATTERN | --choice TEXT ...]\n"
           "       [--raw-output] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-loop off|stop|conclude]\n"
           "       [--reasoning-effort none|minimal|low|medium|high|xhigh|max]\n"
           "       [--vision] [--vision-offload auto|on|off] [--vision-max-merged N]\n"
           "       [--no-cuda-graph]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "       " + argv0 + " --config FILE [options]   (FILE gives the model)\n"
           "\n"
           "Runs one generation: answer content streams to stdout, reasoning and\n"
           "diagnostics to stderr. Sampling defaults come from the loaded model and\n"
           "thinking mode; flags override individual fields.\n"
           "\n"
           "  --config FILE            read options from a TOML file keyed by the long option\n"
           "                           names without the dashes, e.g. max-context = 32768,\n"
           "                           kv-dtype = \"int8\", greedy = true (a flag takes true or\n"
           "                           false; a repeatable option an array); model = \"PATH\"\n"
           "                           names the model; keys in a [cli] table apply only here,\n"
           "                           [serve] only to infernix-serve. Options on the command\n"
           "                           line override the file\n"
           "\n"
           "CONTEXT\n"
           "  --max-context N          max context tokens (default 2048)\n"
           "  --rope-yarn-factor F     startup-fixed YaRN, finite [1,4] (default 1);\n"
           "                           extends allowed ceiling only, not --max-context\n"
           "  --prefill-chunk N        prefill chunk size in tokens, multiple of 128\n"
           "                           (default 16384 with offloaded experts, 1024 otherwise)\n"
           "  --max-new N              cap on generated tokens\n"
           "  --device N               CUDA device ordinal (default 0)\n"
           "  --ram-headroom-mib N     physical Host RAM in MiB that pinned weights and\n"
           "                           caches leave free for the system (default " +
           std::to_string(kDefaultRamHeadroomBytes >> 20) +
           ")\n"
           "  --expert-ram-mib N|auto  Offloaded experts: Host RAM in MiB for the routed\n"
           "                           experts (auto: what --ram-headroom-mib and the other\n"
           "                           pins leave); less than all of them need reads the\n"
           "                           rest from the artifact (the SSD tier)\n"
           "  --expert-pin-mib N|auto  Offloaded experts: at most this much of the experts'\n"
           "                           RAM page-locked for the GPU; the rest stays in RAM\n"
           "                           and the CPU serves it (auto: what Windows allows,\n"
           "                           the GPU's shared memory)\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto     KV-cache capacity in tokens\n"
           "                           (default = --max-context; auto sizes to free\n"
           "                           VRAM less --vram-headroom-mib; offloaded experts:\n"
           "                           auto = --max-context)\n"
           "  --vram-past-budget       Qwen3.8-Flash-Next on Windows: also use the VRAM the\n"
           "                           OS budget withholds (~0.6 GiB more expert cache on\n"
           "                           an RTX 5090; may move to system memory if another\n"
           "                           program needs VRAM)\n"
           "  --expert-tail-skip R     Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B):\n"
           "                           trade quality for decode speed by skipping a routed\n"
           "                           expert missing from VRAM that every token of a decode\n"
           "                           round routes at top-k rank R or lower (1 to k-1; k is\n"
           "                           10 on Flash-Next, 8 on the 122B). Changes the output; not\n"
           "                           with --reproducible. Default off; Strata's\n"
           "                           default is 7\n"
           "  --vram-headroom-mib N|auto\n"
           "                           VRAM in MiB startup sizing leaves free: Qwen3.5,\n"
           "                           after --kv-capacity auto (auto " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           "); Qwen3.8-\n"
           "                           Flash-Next, after the expert cache, for the display\n"
           "                           and other programs (auto: 1024 with a display on\n"
           "                           the GPU, else 256)\n"
           "  --kv-dtype T             bf16 (default) | int8 | fp8 | nvfp4 | k8v4 | vq2 | k4v2;\n"
           "                           bf16 stores K as BF16 and V as FP16 (exact for\n"
           "                           2^-17 <= |v| < 2^16), with --lossless V as BF16\n"
           "  --use-original-int8-prefill-kernel\n"
           "                           prefill INT8 KV with the original prompt kernel\n"
           "                           (default: the fast kernel)\n"
           "  --use-original-nvfp4-prefill-kernel\n"
           "                           prefill NVFP4 KV with the original prompt kernel\n"
           "                           (default: the fast kernel)\n"
           "  --prefill-split-workspace-mib N\n"
           "                           memory for splitting prompt attention across SMs\n"
           "                           (default " +
           std::to_string(kDefaultPrefillSplitWorkspaceMiB) +
           "; 128-384 recommended). Less frees KV cache but\n"
           "                           slows 1-2K-token chunks over long context (64: up\n"
           "                           to 27% slower attention; 0: no splitting)\n"
           "  --reproducible           run the reproducible route: the same bits on every\n"
           "                           run and release, whatever else is running.\n"
           "                           Without it, every path qualified and measured\n"
           "                           faster for this artifact's kernels and KV type\n"
           "                           runs (within Infernix's numerical quality policy;\n"
           "                           startup lists them; docs/cli.md)\n"
           "  --lossless               run the lossless route: every operation of the\n"
           "                           model correctly rounded at its own BF16 points\n"
           "                           (FP32 logits), so the output depends on the\n"
           "                           request and the artifact alone; text with\n"
           "                           --kv-dtype bf16, no speculation; slower\n"
           "                           (docs/maintainer/lossless-route.md)\n"
           "  --fast-path NAME         also run this fast path, qualified or not\n"
           "                           (qualification runs; repeatable)\n"
           "  --expert-a4-activations  nvidia/Qwen3.8-Flash-Next-FP4-Local only: run its\n"
           "                           routed experts with 4-bit (NVFP4) activations, using\n"
           "                           the activation scales NVIDIA calibrated for them.\n"
           "                           Changes the output. No other artifact supports it\n"
           "                           (default: 16-bit activations, as NVIDIA serves it)\n"
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2 speculative backend\n"
           "  --draft-tokens N|auto    drafts per round: mtp 1-7 (Qwen3.5/3.6 1-5),\n"
           "                           dflash/dflash2 1-15 (Qwen3.8-Flash-Next dflash2 1-7)\n"
           "                             N     every round drafts and verifies N\n"
           "                             auto  mtp: each round picks 0-7 from recent\n"
           "                                   acceptance; dflash/dflash2: each round drafts\n"
           "                                   its whole block (Qwen3.5/3.6 15, Flash-Next 7)\n"
           "                                   and verifies as many as the drafts'\n"
           "                                   probabilities favour (Qwen3.5/3.6: where\n"
           "                                   qualified, else the fastest fixed length)\n"
           "  --draft-model FILE       Qwen3.8-Flash-Next dflash2: the DFlash2 drafter's\n"
           "                           companion artifact (Qwen3.5/3.6 artifacts bundle theirs)\n"
           "  --lm-head-draft          use the optimized proposal head\n"
           "  --spec-min-p P           Qwen3.8-Flash-Next mtp: verify drafts only up to the\n"
           "                           first below probability P, 0-1 (default 0.5 with\n"
           "                           --draft-tokens auto, else 0 = all)\n"
           "  --draft-tree-nodes auto|N\n"
           "                           dflash2: verify a small tree of alternative drafts each\n"
           "                           round instead of a single draft, so more drafted tokens\n"
           "                           are accepted (same output distribution).\n"
           "                           auto (recommended): measures speed as it runs and uses a\n"
           "                           tree only when it is faster; artifacts without tree\n"
           "                           support keep single drafts (with a warning).\n"
           "                           N: a fixed tree of N tokens, draft tokens + 2 to 32 (a\n"
           "                           single draft is draft tokens + 1; 16 suits\n"
           "                           --draft-tokens 7). Seeded runs repeat exactly only with N\n"
           "  --draft-tree-paths N     most branches per tree (2-8, default 8)\n"
           "  --no-cuda-graph          disable CUDA-graph decode rounds\n"
           "\n"
           "SAMPLING\n"
           "  --temperature F          sampling temperature (0-2)\n"
           "  --top-p F                nucleus probability (0-1)\n"
           "  --top-k N                keep the top N tokens\n"
           "  --min-p F                minimum token probability (0-1)\n"
           "  --presence-penalty F     -2 to 2\n"
           "  --frequency-penalty F    -2 to 2\n"
           "  --seed N                 fixed random seed\n"
           "  --greedy                 force temperature 0 (exact argmax)\n"
           "  --stop-token-id N...     stop token ids\n"
           "  --stop <text>...         stop on this text\n"
           "  --reasoning-stop <text>  stop reasoning on this text\n"
           "  --reasoning-effort E     none | minimal | low | medium | high | xhigh | max\n"
           "  --no-thinking            disable the thinking mode\n"
           "  --thinking-budget N      cap model-origin thinking tokens\n"
           "  --reasoning-loop MODE    off (default) | stop | conclude: when the thinking keeps\n"
           "                           repeating whole passages, end the reply or close the\n"
           "                           thinking and let the model answer\n"
           "  --raw-output             emit raw content without framing\n"
           "  --print-token-ids        also print generated token ids\n"
           "\n"
           "VISION (off by default)\n"
           "  --vision                 enable image/video input\n"
           "  --vision-offload auto|on|off  keep the vision tower in pinned system RAM\n"
           "                           instead of VRAM; on adds no steady-state VRAM (default\n"
           "                           auto: on with offloaded experts, else off)\n"
           "  --vision-max-merged N    max merged vision tokens per item (default 32768);\n"
           "                           oversized media downscales at preprocessing\n"
           "\n"
           "LOGGING\n"
           "  --log-level L            trace|debug|info|warning|error|critical|off\n"
           "  --log-colours on|off     colour the stats output on stderr (on by default when\n"
           "                           stderr is a terminal; off forces plain output)\n"
           "\n"
           "Structured message content accepts text, image/image_url, and video/video_url\n"
           "parts; media sources may be local paths, HTTP(S) URLs, or base64 data URIs.\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (argc >= 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    product::ProgramArguments program = product::read_program_arguments(argc, argv, cli_config_program());
    if (program.model.empty()) {
        throw std::invalid_argument("a model path (.infernix or .ninfer) is required: give it first on "
                                    "the command line or as `model` in the --config file");
    }
    options.artifact_path     = program.model;
    options.config_path       = std::move(program.config_path);
    options.config_arguments  = program.config_arguments;
    bool kv_capacity_explicit = false;

    const std::vector<std::string>& arguments = program.arguments;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string_view arg(arguments[i]);
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= arguments.size()) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return arguments[i].c_str();
        };
        if (find_option(arg) == nullptr) { throw std::invalid_argument("unknown argument: " + std::string(arg)); }

        if (arg == "--prompt") {
            options.prompt = value(arg);
        } else if (arg == "--chat-template") {
            options.chat_template_path = value(arg);
        } else if (arg == "--ngram-volume") {
            options.ngram_volume_path = value(arg);
        } else if (arg == "--ngram-residency") {
            options.ngram_residency = parse_ngram_residency(value(arg));
        } else if (arg == "--expert-state") {
            const std::string path = value(arg);
            options.expert_state   = path == "off" ? std::filesystem::path{} : std::filesystem::path(path);
        } else if (arg == "--expert-stripe") {
            options.expert_stripe_path = value(arg);
        } else if (arg == "--grammar-file") {
            options.grammar_path = value(arg);
        } else if (arg == "--json-object") {
            options.json_object = true;
        } else if (arg == "--json-schema-file") {
            options.json_schema_path = value(arg);
        } else if (arg == "--regex") {
            options.regex = value(arg);
        } else if (arg == "--choice") {
            options.choices.emplace_back(value(arg));
        } else if (arg == "--messages") {
            options.messages_path = value(arg);
        } else if (arg == "--max-new") {
            options.max_new = parse_u32(value(arg), "max-new");
        } else if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor = parse_float(value(arg), "rope-yarn-factor", 1.0F, 4.0F);
        } else if (arg == "--max-context") {
            options.max_context = parse_u32(value(arg), "max-context");
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(value(arg));
            kv_capacity_explicit = true;
        } else if (arg == "--vram-past-budget") {
            options.vram_past_budget = true;
        } else if (arg == "--expert-tail-skip") {
            options.expert_tail_skip = parse_u32(value(arg), "expert-tail-skip", true);
        } else if (arg == "--vram-headroom-mib") {
            const char* text = value(arg);
            if (std::string_view(text) == "auto") {
                options.vram_headroom_bytes.reset();
            } else {
                const std::uint64_t mib = parse_u64(text, "vram-headroom-mib");
                if (mib > (std::numeric_limits<std::size_t>::max() >> 20)) {
                    throw std::invalid_argument("--vram-headroom-mib is out of range");
                }
                options.vram_headroom_bytes = static_cast<std::size_t>(mib) << 20;
            }
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
            if (options.prefill_chunk == 0) {
                throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
            }
        } else if (arg == "--device") {
            options.device = parse_device(value(arg));
        } else if (arg == "--ram-headroom-mib") {
            const std::uint64_t mib = parse_u64(value(arg), "ram-headroom-mib");
            if (mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                throw std::invalid_argument("--ram-headroom-mib is out of range");
            }
            options.ram_headroom_bytes = mib << 20;
        } else if (arg == "--expert-ram-mib") {
            const std::string text = value(arg);
            if (text == "auto") {
                options.expert_ram_bytes.reset();
            } else {
                const std::uint64_t mib = parse_u64(text.c_str(), "expert-ram-mib");
                if (mib == 0 || mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                    throw std::invalid_argument("--expert-ram-mib is out of range");
                }
                options.expert_ram_bytes = mib << 20;
            }
        } else if (arg == "--expert-pin-mib") {
            const std::string text = value(arg);
            if (text == "auto") {
                options.expert_pin_bytes.reset();
            } else {
                const std::uint64_t mib = parse_u64(text.c_str(), "expert-pin-mib");
                if (mib == 0 || mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                    throw std::invalid_argument("--expert-pin-mib is out of range");
                }
                options.expert_pin_bytes = mib << 20;
            }
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value(arg));
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            product::apply_draft_tokens(options.speculative, value(arg));
        } else if (arg == "--draft-model") {
            options.speculative.draft_model = std::filesystem::path(std::string(value(arg)));
        } else if (arg == "--spec-min-p") {
            options.speculative.draft_min_p = product::parse_draft_min_p(value(arg));
        } else if (arg == "--ngram-draft-tokens") {
            product::apply_ngram_draft_tokens(options.speculative, value(arg));
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match = parse_u32(value(arg), "ngram-min-match");
        } else if (arg == "--draft-tree-nodes") {
            product::apply_draft_tree_nodes(options.speculative, value(arg));
        } else if (arg == "--draft-tree-paths") {
            options.speculative.draft_tree_paths = parse_u32(value(arg), "draft-tree-paths");
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--raw-output") {
            options.raw_output = true;
        } else if (arg == "--print-token-ids") {
            options.print_token_ids = true;
        } else if (arg == "--log-colours") {
            const std::string_view mode = value(arg);
            if (mode == "on") {
                options.log_colours = true;
            } else if (mode == "off") {
                options.log_colours = false;
            } else {
                throw std::invalid_argument("--log-colours accepts on or off");
            }
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--thinking-budget") {
            options.thinking_budget = parse_u32(value(arg), "thinking-budget");
        } else if (arg == "--reasoning-loop") {
            const std::string mode = value(arg);
            if (mode == "off") {
                options.reasoning_loop = infernix::ReasoningLoopAction::Off;
            } else if (mode == "stop") {
                options.reasoning_loop = infernix::ReasoningLoopAction::Stop;
            } else if (mode == "conclude") {
                options.reasoning_loop = infernix::ReasoningLoopAction::Conclude;
            } else {
                throw std::invalid_argument("--reasoning-loop must be off, stop or conclude");
            }
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-offload") {
            const std::string_view mode = value(arg);
            if (mode == "auto") {
                options.vision_offload = VisionOffload::Auto;
            } else if (mode == "on") {
                options.vision_offload = VisionOffload::On;
            } else if (mode == "off") {
                options.vision_offload = VisionOffload::Off;
            } else {
                throw std::invalid_argument("--vision-offload accepts auto, on or off");
            }
        } else if (arg == "--vision-max-merged") {
            const std::uint32_t merged = parse_u32(value(arg), "vision-max-merged");
            if (merged < 64 || merged > 32768) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 32768]");
            }
            options.vision_max_merged_tokens = merged;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--use-original-int8-prefill-kernel") {
            options.original_int8_prefill_kernel = true;
        } else if (arg == "--prefill-split-workspace-mib") {
            options.prefill_split_workspace_mib =
                parse_u32(value(arg), "prefill-split-workspace-mib", true);
        } else if (arg == "--use-original-nvfp4-prefill-kernel") {
            options.original_nvfp4_prefill_kernel = true;
        } else if (arg == "--reproducible") {
            options.fast_paths = false;
        } else if (arg == "--lossless") {
            options.lossless = true;
        } else if (arg == "--fast-path") {
            const std::string name(value(arg));
            const std::optional<FastPath> path = parse_fast_path(name);
            if (!path) { throw std::invalid_argument("unknown fast path: " + name); }
            options.fast_path_candidates.push_back(*path);
        } else if (arg == "--expert-a4-activations") {
            options.expert_a4_activations = true;
        } else if (arg == "--stop-token-id") {
            const std::uint32_t token = parse_u32(value(arg), "stop-token-id", true);
            if (token > static_cast<std::uint32_t>(std::numeric_limits<TokenId>::max())) {
                throw std::invalid_argument("--stop-token-id exceeds the token domain");
            }
            options.stop_token_ids.push_back(static_cast<TokenId>(token));
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            std::string text = value(arg);
            if (text.empty()) {
                throw std::invalid_argument(std::string(arg) + " must not be empty");
            }
            options.stop_strings.push_back(StopString{
                .text    = std::move(text),
                .channel = arg == "--stop" ? OutputChannel::Content : OutputChannel::Reasoning,
            });
        } else if (arg == "--temperature") {
            options.sampling.temperature = parse_float(value(arg), "temperature", 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            options.sampling.top_p = parse_float(value(arg), "top-p", 0.0F, 1.0F);
        } else if (arg == "--top-k") {
            const std::uint32_t top_k = parse_u32(value(arg), "top-k", true);
            if (top_k > 20) { throw std::invalid_argument("--top-k must be in [0,20]"); }
            options.sampling.top_k = static_cast<std::int32_t>(top_k);
        } else if (arg == "--min-p") {
            options.sampling.min_p = parse_float(value(arg), "min-p", 0.0F, 1.0F);
        } else if (arg == "--presence-penalty") {
            options.sampling.presence_penalty =
                parse_float(value(arg), "presence-penalty", -2.0F, 2.0F);
        } else if (arg == "--frequency-penalty") {
            options.sampling.frequency_penalty =
                parse_float(value(arg), "frequency-penalty", -2.0F, 2.0F);
        } else if (arg == "--seed") {
            options.sampling.seed = parse_u64(value(arg), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(value(arg));
        } else {
            throw std::logic_error("CLI option " + std::string(arg) + " has no parser");
        }
    }

    if (int(!options.grammar_path.empty()) + int(!options.json_schema_path.empty()) +
            int(options.json_object) + int(options.regex.has_value()) +
            int(!options.choices.empty()) >
        1)
        throw std::invalid_argument("select only one of --grammar-file, --json-object, "
                                    "--json-schema-file, --regex or --choice");
    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }

    const bool has_prompt   = !options.prompt.empty();
    const bool has_messages = !options.messages_path.empty();
    if (has_prompt == has_messages) {
        throw std::invalid_argument("pass exactly one of --prompt or --messages");
    }
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a multiple of 128");
    }
    if (options.prefill_split_workspace_mib > kMaximumPrefillSplitWorkspaceMiB) {
        throw std::invalid_argument("--prefill-split-workspace-mib must be in [0,16384]");
    }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.enable_thinking == false && options.reasoning_effort &&
        *options.reasoning_effort != ReasoningEffort::None) {
        throw std::invalid_argument("--reasoning-effort cannot be combined with --no-thinking");
    }
    if (options.reasoning_effort == ReasoningEffort::None) options.enable_thinking = false;
    if (options.enable_thinking == false && options.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    if (options.greedy) { options.sampling.temperature = 0.0F; }
    return options;
}

} // namespace infernix::cli
