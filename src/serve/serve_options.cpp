#include "serve/serve_options.h"
#include "product/config/config_file.h"
#include "product/logits_options.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace infernix::serve {
namespace {

// Every option parse_serve_options accepts after the artifact path, whether it takes a value and
// whether it repeats. The argument loop accepts nothing outside this table, and --config maps its
// keys through it, so a config file accepts exactly the command line's options with the same arity.
using ServeOptionArity = product::ConfigOption;

constexpr std::array kServeOptionArities{
    ServeOptionArity{"--host"},
    ServeOptionArity{"--port"},
    ServeOptionArity{"--api-key"},
    ServeOptionArity{"--model-id"},
    ServeOptionArity{"--ngram-volume"},
    ServeOptionArity{"--ngram-residency"},
    ServeOptionArity{"--expert-stripe"},
    ServeOptionArity{"--expert-state"},
    ServeOptionArity{"--chat-template"},
    ServeOptionArity{"--claude-code-prompt"},
    ServeOptionArity{"--rope-yarn-factor"},
    ServeOptionArity{"--max-context"},
    ServeOptionArity{"--kv-capacity"},
    ServeOptionArity{"--vram-headroom-mib"},
    ServeOptionArity{"--vram-past-budget", false},
    ServeOptionArity{"--expert-tail-skip"},
    ServeOptionArity{"--max-concurrency"},
    ServeOptionArity{"--max-pending-requests"},
    ServeOptionArity{"--pending-timeout-ms"},
    ServeOptionArity{"--prefill-chunk"},
    ServeOptionArity{"--use-original-int8-prefill-kernel", false},
    ServeOptionArity{"--prefill-split-workspace-mib"},
    ServeOptionArity{"--use-original-nvfp4-prefill-kernel", false},
    ServeOptionArity{"--reproducible", false},
    ServeOptionArity{"--lossless", false},
    ServeOptionArity{"--fast-path", true, true},
    ServeOptionArity{"--expert-a4-activations", false},
    ServeOptionArity{"--a8-activations", false},
    ServeOptionArity{"--embedding-host", false},
    ServeOptionArity{"--context-cost-presets"},
    ServeOptionArity{"--log-stats-interval-ms"},
    ServeOptionArity{"--log-colours"},
    ServeOptionArity{"--log-stats-panel"},
    ServeOptionArity{"--max-request-mib"},
    ServeOptionArity{"--media-cache-mib"},
    ServeOptionArity{"--media-live-mib"},
    ServeOptionArity{"--media-preprocess-threads"},
    ServeOptionArity{"--device-snapshot-slots"},
    ServeOptionArity{"--cache-taps-per-request"},
    ServeOptionArity{"--cache-tap-ladder"},
    ServeOptionArity{"--prefix-cache-file"},
    ServeOptionArity{"--prefix-cache-save-mins"},
    ServeOptionArity{"--no-queue-holds", false},
    ServeOptionArity{"--cache-tool-hold-secs"},
    ServeOptionArity{"--cache-next-turn-ms"},
    ServeOptionArity{"--cache-host-state"},
    ServeOptionArity{"--cache-tap-min-gap"},
    ServeOptionArity{"--host-context-mib"},
    ServeOptionArity{"--request-log-jsonl"},
    ServeOptionArity{"--request-log-max-mib"},
    ServeOptionArity{"--request-log-keep"},
    ServeOptionArity{"--route-trace"},
    ServeOptionArity{"--response-store-max-records"},
    ServeOptionArity{"--response-store-max-mib"},
    ServeOptionArity{"--device"},
    ServeOptionArity{"--ram-free-mib"},
    ServeOptionArity{"--expert-ram-mib"},
    ServeOptionArity{"--expert-ram-layout"},
    ServeOptionArity{"--expert-pin-mib"},
    ServeOptionArity{"--kv-dtype"},
    ServeOptionArity{"--logits-dtype"},
    ServeOptionArity{"--spec"},
    ServeOptionArity{"--draft-tokens"},
    ServeOptionArity{"--draft-model"},
    ServeOptionArity{"--spec-min-p"},
    ServeOptionArity{"--draft-tree-nodes"},
    ServeOptionArity{"--draft-tree-paths"},
    ServeOptionArity{"--ngram-draft-tokens"},
    ServeOptionArity{"--ngram-min-match"},
    ServeOptionArity{"--ngram-archive-mib"},
    ServeOptionArity{"--ngram-session-mib"},
    ServeOptionArity{"--ngram-native-sessions", false},
    ServeOptionArity{"--default-max-tokens"},
    ServeOptionArity{"--default-thinking-budget"},
    ServeOptionArity{"--thinking-budget-message"},
    ServeOptionArity{"--no-thinking-answer-room", false},
    ServeOptionArity{"--reasoning-loop"},
    ServeOptionArity{"--vision", false},
    ServeOptionArity{"--vision-offload"},
    ServeOptionArity{"--vision-max-merged"},
    ServeOptionArity{"--no-cuda-graph", false},
    ServeOptionArity{"--no-prefix-reuse", false},
    ServeOptionArity{"--lm-head-draft", false},
    ServeOptionArity{"--no-thinking", false},
    ServeOptionArity{"--preserve-thinking", false},
    ServeOptionArity{"--tolerant-tool-calls", false},
    ServeOptionArity{"--restore-tool-reasoning", false},
    ServeOptionArity{"--any-model-name", false},
    ServeOptionArity{"--cors", false},
    ServeOptionArity{"--usage-chunk-choice", false},
    ServeOptionArity{"--temperature"},
    ServeOptionArity{"--top-p"},
    ServeOptionArity{"--top-k"},
    ServeOptionArity{"--min-p"},
    ServeOptionArity{"--presence-penalty"},
    ServeOptionArity{"--frequency-penalty"},
    ServeOptionArity{"--seed"},
    ServeOptionArity{"--greedy", false},
    ServeOptionArity{"--log-level"},
};

const ServeOptionArity* find_option(std::string_view name) noexcept {
    const auto found =
        std::find_if(kServeOptionArities.begin(), kServeOptionArities.end(),
                     [&](const ServeOptionArity& option) { return option.name == name; });
    return found == kServeOptionArities.end() ? nullptr : &*found;
}

// infernix-serve's --config: top-level keys and the [serve] table; [cli] is infernix's.
const product::ConfigProgram& serve_config_program() {
    static const product::ConfigProgram program{.table       = "serve",
                                                .other_table = "cli",
                                                .find_option = [](std::string_view name) {
                                                    return find_option(name);
                                                }};
    return program;
}

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

std::size_t parse_host_context_mib(const char* text) {
    constexpr std::size_t bytes_per_mib = 1ULL << 20;
    constexpr std::size_t maximum       = std::numeric_limits<std::size_t>::max();
    const std::string_view value(text);
    const std::size_t point      = value.find('.');
    const std::string_view whole = value.substr(0, point);
    std::string_view fraction =
        point == std::string_view::npos ? std::string_view{} : value.substr(point + 1);
    if (whole.empty() || whole.find_first_not_of("0123456789") != std::string_view::npos ||
        (point != std::string_view::npos &&
         (fraction.empty() ||
          fraction.find_first_not_of("0123456789") != std::string_view::npos))) {
        throw std::invalid_argument("--host-context-mib requires a nonnegative decimal MiB value");
    }

    std::size_t whole_mib = 0;
    for (const char character : whole) {
        const std::size_t digit = static_cast<std::size_t>(character - '0');
        if (whole_mib > (maximum / bytes_per_mib - digit) / 10) {
            throw std::invalid_argument("--host-context-mib is out of range");
        }
        whole_mib = whole_mib * 10 + digit;
    }

    // A whole-byte MiB value has at most 20 fractional decimal digits (1 MiB = 2^20 bytes).
    // Use exact integer arithmetic so decimal parsing cannot round the requested budget upward.
    while (!fraction.empty() && fraction.back() == '0') { fraction.remove_suffix(1); }
    if (fraction.size() > 20) {
        throw std::invalid_argument("--host-context-mib must resolve to a whole number of bytes");
    }
    // Horner from the last digit: every partial value is below 10 MiB and, when the whole
    // fraction resolves to whole bytes, every division is exact (5^k divides each suffix).
    std::size_t fractional_bytes = 0;
    for (auto it = fraction.rbegin(); it != fraction.rend(); ++it) {
        const std::size_t scaled =
            fractional_bytes + static_cast<std::size_t>(*it - '0') * bytes_per_mib;
        if (scaled % 10 != 0) {
            throw std::invalid_argument("--host-context-mib must resolve to a whole number of bytes");
        }
        fractional_bytes = scaled / 10;
    }
    const std::size_t whole_bytes      = whole_mib * bytes_per_mib;
    if (fractional_bytes > maximum - whole_bytes) {
        throw std::invalid_argument("--host-context-mib is out of range");
    }
    return whole_bytes + fractional_bytes;
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    if (value == "vq2") { return KvCacheStorage::Vq2; }
    if (value == "k4v2") { return KvCacheStorage::Q4KeyVq2Value; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

// N MiB, or auto (empty: the model's automatic headroom).
std::optional<std::size_t> parse_vram_headroom(const char* text) {
    if (std::string_view(text) == "auto") { return std::nullopt; }
    const std::uint64_t mib = parse_u64(text, "vram-headroom-mib");
    if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
        throw std::invalid_argument("--vram-headroom-mib is out of range");
    }
    return static_cast<std::size_t>(mib) << 20;
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
           " <model.infernix> [options]\n"
           "       " + argv0 + " --config FILE [options]   (FILE gives the model)\n"
           "\n"
           "Serves the OpenAI Responses/Chat Completions and Anthropic Messages APIs.\n"
           "  --help, -h                 show this help and exit\n"
           "  --config FILE              read options from a TOML file keyed by the long option\n"
           "                             names without the dashes, e.g. port = 8080,\n"
           "                             kv-dtype = \"int8\", greedy = true (a flag takes true or\n"
           "                             false; a repeatable option an array); model = \"PATH\"\n"
           "                             names the artifact; keys in a [serve] table apply only\n"
           "                             here, [cli] only to infernix. Options on the command\n"
           "                             line override the file\n"
           "\n"
           "MODEL & CONTEXT\n"
           "  --max-context N            max context tokens per request (default 8192)\n"
           "  --max-concurrency N        max concurrent sequences, 1-8 (default 1)\n"
           "  --prefill-chunk N          prefill chunk size in tokens, multiple of 128\n"
           "                             (default 16384 with offloaded experts, 1024 otherwise)\n"
           "  --use-original-int8-prefill-kernel\n"
           "                             prefill INT8 KV with the original prompt kernel at\n"
           "                             the requested chunk (default: the fast kernel, chunk\n"
           "                             rounded down to whole attention waves)\n"
           "  --use-original-nvfp4-prefill-kernel\n"
           "                             prefill NVFP4 KV with the tiled prompt kernel\n"
           "                             (default: the fast kernel)\n"
           "  --prefill-split-workspace-mib N\n"
           "                             memory for splitting prompt attention across SMs\n"
           "                             (default " +
           std::to_string(kDefaultPrefillSplitWorkspaceMiB) +
           "; 128-384 recommended). Less frees KV cache but\n"
           "                             slows 1-2K-token chunks over long context (64: up\n"
           "                             to 27% slower attention; 0: no splitting)\n"
           "  --reproducible             run the reproducible route (the same bits on every\n"
           "                             run of this build, whatever else is running):\n"
           "                             without it, every path qualified and measured\n"
           "                             faster for this artifact's kernels and KV type\n"
           "                             runs (within Infernix's numerical quality\n"
           "                             policy; startup lists them; docs/serving.md)\n"
           "  --lossless                 run the lossless route: every operation of the\n"
           "                             model correctly rounded at its own BF16 points\n"
           "                             (logits as --logits-dtype), so the output depends on the\n"
           "                             request and the artifact alone; text with\n"
           "                             --kv-dtype bf16, no speculation; slower\n"
           "                             (docs/maintainer/lossless-route.md)\n"
           "  --fast-path NAME           also run this fast path, qualified or not\n"
           "                             (qualification and timing runs; repeatable)\n"
           "  --expert-a4-activations    nvidia/Qwen3.8-Flash-Next-FP4-Local only: run its\n"
           "                             routed experts with 4-bit (NVFP4) activations, using\n"
           "                             the activation scales NVIDIA calibrated for them.\n"
           "                             Changes the output. No other artifact supports it\n"
           "                             (default: 16-bit activations, as NVIDIA serves it)\n"
           "  --a8-activations           8-bit activations where they measured faster: decode\n"
           "                             and verification calls of GGUF dense projections and\n"
           "                             routed experts (GGUF and NVFP4 W4A16 banks, CPU-served\n"
           "                             misses included; int8 with one scale per 32 values, as\n"
           "                             llama.cpp's Q8_1), and prefill calls of GGUF dense\n"
           "                             projections and those experts (the same codec), a\n"
           "                             groupwise Q4 MLP gate/up (int8 per token) and MXFP8\n"
           "                             dense projections (two E4M3 terms per 32 values).\n"
           "                             Other calls keep their routes. Approximate: changes the\n"
           "                             output; --reproducible and artifacts without such a\n"
           "                             route refuse it (default: off)\n"
           "  --embedding-host           Qwen3.5-family artifacts: keep the token embedding in\n"
           "                             pinned host memory, read zero-copy, so its VRAM\n"
           "                             (1.18 GiB on the 27B) goes to the KV and prefix caches.\n"
           "                             Same output; prompts read their rows over PCIe\n"
           "  --no-cuda-graph            disable CUDA-graph decode rounds (on by default)\n"
           "  --default-max-tokens N     default max_tokens when a request omits it\n"
           "                             (default " +
           std::to_string(kDefaultMaxTokens) +
           ")\n"
           "  --default-thinking-budget N  cap model-origin thinking for enabled\n"
           "                             requests; control tokens count toward the\n"
           "                             request output limit\n"
           "  --no-thinking-answer-room  without a budget, thinking may use all of the\n"
           "                             output limit (default: from a limit of 1024, thinking\n"
           "                             closes early enough to leave a quarter for the answer;\n"
           "                             the limit is max_tokens or the context the prompt\n"
           "                             leaves, whichever is less)\n"
           "  --thinking-budget-message S  message fed to the model when it hits its thinking\n"
           "                             budget, replacing the built-in end-of-thinking notice\n"
           "                             (wrap the message in double quotes, e.g.\n"
           "                             --thinking-budget-message \"Time to stop thinking. I must "
           "act\n"
           "                             now:\")\n"
           "  --reasoning-loop MODE      off (default), stop or conclude: every 512 thinking\n"
           "                             tokens, check whether 25% of the last 2,000 words\n"
           "                             repeat 12-word passages seen 3 times; stop ends the\n"
           "                             reply as length, conclude closes the thinking with\n"
           "                             the thinking-budget message and lets the model answer\n"
           "  --model-id ID              override the artifact metadata.name reported by\n"
           "                             the server\n"
           "  --any-model-name           serve OpenAI requests whatever model name they\n"
           "                             give, reporting the served model's name (off: any\n"
           "                             other name is 404 model_not_found)\n"
           "  --chat-template FILE       replace the artifact frontend chat template at\n"
           "                             startup; must match a template the target accepts\n"
           "  --claude-code-prompt as-sent|local\n"
           "                             Anthropic Messages from Claude Code: keep its system\n"
           "                             prompt (as-sent, default) or replace its Claude-\n"
           "                             specific parts with a short local-agent preamble\n"
           "  --ngram-volume FILE        Qwen3.8-Flash-Next n-gram embedding volume: the\n"
           "                             artifact's, or any FP8, IQ4_NL, BF16 or NVFP4 volume\n"
           "                             of the table in place of its own\n"
           "                             (default: the artifact path + .ngram)\n"
           "  --ngram-residency auto|ram|volume\n"
           "                             where the n-gram table lives: ram reads it all into\n"
           "                             RAM (locked for the CPU) at startup, volume reads rows\n"
           "                             on demand;\n"
           "                             auto (default) takes RAM while every expert still\n"
           "                             fits in RAM beside it\n"
           "  --expert-state FILE|off    Qwen3.8-Flash-Next expert-cache state: read at\n"
           "                             startup to fill the VRAM expert cache, written at\n"
           "                             stop and every 10 minutes between requests\n"
           "                             (default: the artifact path + .expert-state)\n"
           "  --expert-stripe FILE       Qwen3.8-Flash-Next expert stripe on a second drive:\n"
           "                             the SSD expert tier reads each record's tail from it\n"
           "                             beside its head in the artifact\n"
           "  --context-cost-presets F   runtime context-cost preset file (overrides\n"
           "                             matching compiled-in values)\n"
           "  --rope-yarn-factor F       runtime YaRN context extension factor, finite [1,4]\n"
           "                             (default 1); startup-fixed, extends the allowed\n"
           "                             ceiling only, not --max-context\n"
           "  --device N                 CUDA device ordinal (default 0)\n"
           "  --ram-free-mib N           physical Host RAM in MiB that Infernix's weights,\n"
           "                             experts, n-gram table and caches leave free for\n"
           "                             other programs (default " +
           std::to_string(kDefaultRamFreeBytes >> 20) +
           ")\n"
           "  --expert-ram-mib N|auto    Offloaded experts: Host RAM in MiB for the routed\n"
           "                             experts (auto: what --ram-free-mib and the other\n"
           "                             pins leave); less than all of them need reads the\n"
           "                             rest from the artifact (the SSD tier)\n"
           "  --expert-ram-layout exclusive|full\n"
           "                             Qwen3.8-Flash-Next experts VRAM cannot hold: exclusive\n"
           "                             keeps in RAM only the experts VRAM does not hold;\n"
           "                             full keeps a RAM copy of every expert (more RAM)\n"
           "                             (default " + std::string(expert_ram_layout_name(EngineOptions{}.expert_ram_layout)) + ")\n"
           "  --expert-pin-mib N|auto    Offloaded experts: at most this much of the experts'\n"
           "                             RAM page-locked for the GPU; the rest stays in RAM\n"
           "                             and the CPU serves it (auto: what Windows allows,\n"
           "                             the GPU's shared memory)\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto       KV-cache capacity in tokens (default auto, or\n"
           "                             --max-context with --no-prefix-reuse; auto sizes\n"
           "                             to free VRAM less --vram-headroom-mib;\n"
           "                             offloaded experts:\n"
           "                             auto = --max-context x --max-concurrency)\n"
           "  --vram-headroom-mib N|auto VRAM in MiB startup sizing leaves free: Qwen3.5,\n"
           "                             after --kv-capacity auto (auto " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           "); Qwen3.8-Flash-\n"
           "                             Next, after the expert cache, for the display and\n"
           "                             other programs (auto: 1024 with a display on the\n"
           "                             GPU, else 256)\n"
           "  --vram-past-budget         Qwen3.8-Flash-Next on Windows: also use the VRAM\n"
           "                             the OS budget withholds (~0.6 GiB more expert\n"
           "                             cache on an RTX 5090; may move to system memory\n"
           "                             if another program needs VRAM)\n"
           "  --expert-tail-skip R       Offloaded experts (Qwen3.8-Flash-Next, Qwen3.5-122B):\n"
           "                             trade quality for decode speed by skipping a routed\n"
           "                             expert missing from VRAM that every token of a decode\n"
           "                             round routes at top-k rank R or lower (1 to k-1; k is\n"
           "                             10 on Flash-Next, 8 on the 122B). Changes the output; not\n"
           "                             with --reproducible. Default off; Strata's\n"
           "                             default is 7\n"
           "  --kv-dtype T               KV storage: bf16 (default) | int8 | fp8 | nvfp4 | k8v4 |\n"
           "                             vq2 | k4v2; bf16 stores K as BF16 and V as FP16 (exact\n"
           "                             for 2^-17 <= |v| < 2^16), with --lossless V as BF16\n"
           "  --logits-dtype T           fp32 (default) | bf16: the head's logits the sampler,\n"
           "                             greedy choice, draft acceptance and constraints read;\n"
           "                             bf16 rounds each once, as Transformers returns them\n"
           "                             (near-equal logits then tie), for comparisons\n"
           "  --no-prefix-reuse          disable the prefix cache: every request starts from\n"
           "                             the root and a preempted request recovers by\n"
           "                             replay; cannot be combined with any option below\n"
           "\n"
           "PREFIX CACHE\n"
           "  Content-addressed 64-token KV blocks shared across requests plus sparse model\n"
           "  state snapshots. It configures itself: free VRAM becomes Device block cache\n"
           "  (--kv-capacity defaults to auto) and every option below is optional.\n"
           "  --host-context-mib N|auto  pinned Host pool in decimal MiB (whole bytes) that\n"
           "                             KV blocks and state snapshots share (default 8192;\n"
           "                             0 = GPU only; Qwen3.8-Flash-Next: must be positive;\n"
           "                             auto: Qwen3.8-Flash-Next sizes it from the RAM its\n"
           "                             experts and n-gram table leave, up to 8192)\n"
           "  --prefix-cache-file PATH   restore the host tier from PATH at startup when\n"
           "                             present (written by this binary for the same\n"
           "                             artifact and KV format) and save it there on clean\n"
           "                             shutdown (Ctrl+C twice); relative paths resolve\n"
           "                             against the launch directory (default off:\n"
           "                             nothing is saved)\n"
           "  --prefix-cache-save-mins N also save the host tier to --prefix-cache-file\n"
           "                             every N minutes while serving, when it changed;\n"
           "                             requests wait while it is written (default off:\n"
           "                             only at shutdown). The file is replaced only once\n"
           "                             the new one is complete. WARNING: each save\n"
           "                             rewrites the whole file (up to --host-context-mib,\n"
           "                             several GB), so a short interval writes a lot to\n"
           "                             the drive and wears an SSD faster; prefer 15-60\n"
           "                             minutes\n"
           "  --device-snapshot-slots N  device state snapshot slots (default concurrency\n"
           "                             + 1; + 2 without a host tier). Qwen3.8-Flash-\n"
           "                             Next has none\n"
           "  --cache-taps-per-request N new prefill snapshots per request (default 8;\n"
           "                             2 without a host tier)\n"
           "  --cache-tap-ladder N       ladder base in tokens for history snapshots\n"
           "                             (default max(4096, 2x prefill chunk))\n"
           "  --cache-tap-min-gap N      minimum tokens between ladder snapshots\n"
           "                             (default max(1024, prefill chunk))\n"
           "  --no-queue-holds           do not keep waiting requests' cached prefixes\n"
           "                             ahead of other cache entries (default on: what a\n"
           "                             queued request resumes from is evicted last, the\n"
           "                             request furthest back first)\n"
           "  --cache-tool-hold-secs N   after a reply that ends in a tool call, keep that\n"
           "                             conversation's cached prefix ahead of other cache\n"
           "                             entries for N seconds (default 30, 0 turns it off;\n"
           "                             Qwen3.8-Flash-Next has none)\n"
           "  --cache-next-turn-ms N     once the server has been idle for N ms after a\n"
           "                             reply, prefill that conversation's history as its\n"
           "                             next request will render it (yields to any request\n"
           "                             at its next unit; never changes output). Default\n"
           "                             150, 0 turns it off; Qwen3.8-Flash-Next: default 0,\n"
           "                             refused with --reproducible and --lossless\n"
           "  --cache-host-state fp32|fp24|bf16\n"
           "                             host tier form of snapshot recurrent state: fp24\n"
           "                             and bf16 shrink each 27B snapshot by 36 and 72 MiB\n"
           "                             but resume host-restored snapshots from a rounded\n"
           "                             state (default fp32: exact)\n"
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2    speculative decoding backend\n"
           "  --draft-tokens N|auto      drafts per round: mtp 1-7 (Qwen3.5/3.6 1-5),\n"
           "                             dflash/dflash2 1-15 (Qwen3.8-Flash-Next\n"
           "                             dflash2 1-7)\n"
           "                               N     every round drafts and verifies N\n"
           "                               auto  mtp: each round picks 0-7 from recent\n"
           "                                     acceptance; dflash/dflash2: each round\n"
           "                                     drafts its whole block (Qwen3.5/3.6 15,\n"
           "                                     Flash-Next 7) and verifies as many as the\n"
           "                                     drafts' probabilities favour (Qwen3.5/3.6:\n"
           "                                     where qualified, else the fastest fixed\n"
           "                                     length)\n"
           "  --draft-model FILE         Qwen3.8-Flash-Next dflash2: the DFlash2 drafter's\n"
           "                             companion artifact (Qwen3.5/3.6 artifacts bundle\n"
           "                             theirs)\n"
           "  --lm-head-draft            use the optimized proposal head\n"
           "  --spec-min-p P             mtp (and Qwen3.8-Flash-Next dflash2): one-request\n"
           "                             rounds verify drafts only up to the first below\n"
           "                             probability P, 0-1 (Flash-Next default 0.5 with\n"
           "                             --draft-tokens auto, else 0 = all; Qwen3.5/3.6\n"
           "                             default 0)\n"
           "  --draft-tree-nodes auto|LIST\n"
           "                             dflash2: verify a small tree of alternative drafts\n"
           "                             each round instead of a single draft, so more drafted\n"
           "                             tokens are accepted (same output distribution).\n"
           "                             auto (recommended): measures speed while serving and\n"
           "                             uses a tree only where it is faster, typically with\n"
           "                             one to three active requests below ~64K context; its\n"
           "                             choices follow timing, so seeded output can vary\n"
           "                             between runs. Artifacts without tree support keep\n"
           "                             single drafts (startup warning).\n"
           "                             LIST: fixed tree sizes in tokens by number of active\n"
           "                             requests, e.g. 16,12,12,0 = 16 for one request, 12\n"
           "                             for two or three, single drafts from four; sizes are\n"
           "                             draft tokens + 2 to 32, 0 = single draft (draft\n"
           "                             tokens + 1)\n"
           "  --draft-tree-paths N       most branches per tree, 2-8 (default 8)\n"
           "  --ngram-draft-tokens N     propose N verified ngram copies per round, 1-63 (0 off);\n"
           "                             auto: the configuration's measured window and minimum\n"
           "                             match (Qwen3.8-27B / Qwen3.6-35B-A3B)\n"
           "  --ngram-min-match N        minimum ngram match length, 4-64\n"
           "  --ngram-archive-mib N      MiB of retained source archive for ngram proposals\n"
           "  --ngram-session-mib N      MiB session-scoped ngram source budget (default 128);\n"
           "                             used only with --ngram-archive-mib\n"
           "  --ngram-native-sessions    retain ngram sources across compaction; requires\n"
           "                             --ngram-archive-mib\n"
           "\n"
           "VISION (off by default)\n"
           "  --vision                   enable media and load the Vision GPU allocations\n"
           "  --vision-offload auto|on|off  keep the vision tower in pinned system RAM\n"
           "                             instead of VRAM; on adds no steady-state VRAM and\n"
           "                             borrows device memory only while encoding an image\n"
           "                             (default auto: on with offloaded experts, else\n"
           "                             off); on requires --vision\n"
           "  --vision-max-merged N      bound merged vision tokens, 64-32768 (default 32768)\n"
           "  --media-cache-mib N        retained decoded-media cache\n"
           "                             (default 1024; 0 disables)\n"
           "  --media-live-mib N         cap on live BF16 patch payloads (default 2048)\n"
           "  --media-preprocess-threads N  decode/preprocess workers (default 0 = auto,\n"
           "                             at most 16 workers; explicit max 64)\n"
           "\n"
           "SAMPLING (defaults come from the loaded model + thinking mode; server flags\n"
           "and request fields override individual values)\n"
           "  --temperature F            sampling temperature (0-2)\n"
           "  --top-p F                  nucleus probability (0-1)\n"
           "  --top-k N                  keep the top N tokens (0-20)\n"
           "  --min-p F                  minimum token probability (0-1)\n"
           "  --presence-penalty F       -2 to 2\n"
           "  --frequency-penalty F      -2 to 2\n"
           "  --seed N                   fixed random seed\n"
           "  --greedy                   force temperature 0 (exact argmax)\n"
           "  --no-thinking              disable the thinking mode (enabled by default)\n"
           "  --preserve-thinking        retain closed-turn assistant reasoning\n"
           "                             in later prompts\n"
           "  --tolerant-tool-calls      recover complete Qwen calls with malformed wrapper or\n"
           "                             suffix output, keep a final call cut by the output\n"
           "                             budget and an undeclared name, and return complete\n"
           "                             calls stranded in unclosed thinking (strict\n"
           "                             all-or-nothing by default)\n"
           "  --restore-tool-reasoning   give a Chat Completions tool-call turn sent back\n"
           "                             without its reasoning the reasoning this server\n"
           "                             generated for it, so a tool loop keeps its\n"
           "                             prefix-cache match (off by default)\n"
           "\n"
           "NETWORKING & RESOURCES\n"
           "  --host H                   listen address (default 127.0.0.1)\n"
           "  --port P                   listen port (default 8080)\n"
           "  --api-key KEY              require this key on every request (default: none)\n"
           "  --max-request-mib N        max request body in MiB (default 384; enforced\n"
           "                             pre-parse)\n"
           "  --max-pending-requests N   max queued requests (default 16)\n"
           "  --pending-timeout-ms N     queue timeout in ms (default 30000)\n"
           "  --request-log-jsonl FILE   append full-precision server/request records\n"
           "  --request-log-max-mib N    rotate FILE when it reaches N MiB: it becomes\n"
           "                             FILE.1, older files shift up (default: one file\n"
           "                             without a size limit)\n"
           "  --request-log-keep N       rotated files kept, 0-1000 (default 4)\n"
           "  --route-trace FILE         write the model's internal per-round expert route\n"
           "                             trace to FILE (offloaded experts only)\n"
           "  --response-store-max-records N Responses-state record cap (default 1024)\n"
           "  --response-store-max-mib N      Responses-state byte cap in MiB (default 256)\n"
           "  --log-stats-interval-ms N  throughput-log interval in ms\n"
           "                             (default 5000; 0 disables)\n"
           "  --log-colours on|off       colour the console stats lines (default off; on\n"
           "                             colours the command-window log only, never file\n"
           "                             logs)\n"
           "  --log-stats-panel on|off   pin session averages (TTFT, cache hit, prefill,\n"
           "                             decode, MTP/DFlash and n-gram acceptance) beneath\n"
           "                             the console log (default on; interactive\n"
           "                             terminals only)\n"
           "  --log-level L              pretty stderr verbosity (trace|debug|info|warning|\n"
           "                             error|critical|off; default info)\n"
           "  --cors                     send permissive CORS headers for browser UIs\n"
           "  --usage-chunk-choice       give the streamed usage chunk a zero-delta choice so\n"
           "                             strict parsers that reject choices:[] accept it\n"
           "\n"
           "NOTES\n"
           "  Qwen3.5 applies --vram-headroom-mib only with --kv-capacity auto (the\n"
           "  default with the new prefix caching system).\n"
           "  Options of the two prefix caching systems cannot be mixed.\n"
           "  --vision-offload on requires --vision.\n"
           "  --ngram-native-sessions requires --ngram-archive-mib.\n"
           "  --request-log-max-mib requires --request-log-jsonl, and --request-log-keep\n"
           "  requires --request-log-max-mib.\n"
           "  --rope-yarn-factor is startup-fixed, finite [1,4] (default 1); it extends the\n"
           "  allowed ceiling only, not --max-context.\n"
           "  sampler defaults come from the loaded model and resolved thinking mode;\n"
           "  server flags and request fields override individual values.\n"
           "  --greedy forces temperature 0 (exact argmax).\n";
}

const char* reasoning_loop_name(infernix::ReasoningLoopAction action) noexcept {
    switch (action) {
    case infernix::ReasoningLoopAction::Stop: return "stop";
    case infernix::ReasoningLoopAction::Conclude: return "conclude";
    case infernix::ReasoningLoopAction::Off: break;
    }
    return "off";
}

bool is_loopback_host(std::string_view host) noexcept {
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') { host = host.substr(1, host.size() - 2); }
    return host == "localhost" || host == "::1" || host.starts_with("127.");
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;
    bool request_log_keep_explicit = false;
    bool host_context_explicit     = false;
    // Last prefix-cache flag seen, for the --no-prefix-reuse error.
    const char* cache_option_flag = nullptr;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    product::ProgramArguments program = product::read_program_arguments(argc, argv, serve_config_program());
    if (program.model.empty()) {
        throw std::invalid_argument("artifact path is required: give it first on the command line or "
                                    "as `model` in the --config file");
    }
    options.artifact_path    = std::move(program.model);
    options.config_path      = std::move(program.config_path);
    options.config_arguments = product::redact_option_values(std::move(program.config_arguments), {"--api-key"});
    const std::vector<std::string>& arguments = program.arguments;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const std::string arg    = arguments[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= arguments.size()) {
                throw std::invalid_argument(std::string(flag) + " needs a value");
            }
            return arguments[i].c_str();
        };
        if (find_option(arg) == nullptr) { throw std::invalid_argument("unknown argument: " + arg); }
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--ngram-volume") {
            options.ngram_volume_path = require_value("--ngram-volume");
        } else if (arg == "--ngram-residency") {
            options.ngram_residency = parse_ngram_residency(require_value("--ngram-residency"));
        } else if (arg == "--expert-state") {
            const std::string path = require_value("--expert-state");
            options.expert_state   = path == "off" ? std::filesystem::path{} : std::filesystem::path(path);
        } else if (arg == "--expert-stripe") {
            options.expert_stripe_path = require_value("--expert-stripe");
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
            if (options.chat_template_path.empty()) {
                throw std::invalid_argument("--chat-template must not be empty");
            }
        } else if (arg == "--claude-code-prompt") {
            const std::string_view value = require_value("--claude-code-prompt");
            if (value == "as-sent") {
                options.claude_code_prompt = ClaudeCodePrompt::AsSent;
            } else if (value == "local") {
                options.claude_code_prompt = ClaudeCodePrompt::Local;
            } else {
                throw std::invalid_argument("--claude-code-prompt accepts as-sent or local");
            }
        } else if (arg == "--rope-yarn-factor") {
            options.rope_yarn_factor =
                parse_float_in(require_value("--rope-yarn-factor"), "rope-yarn-factor", 1.0F, 4.0F);
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--vram-headroom-mib") {
            options.vram_headroom_bytes =
                parse_vram_headroom(require_value("--vram-headroom-mib"));
        } else if (arg == "--vram-past-budget") {
            options.vram_past_budget = true;
        } else if (arg == "--expert-tail-skip") {
            options.expert_tail_skip = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--expert-tail-skip"), "expert-tail-skip"));
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
            if (options.prefill_chunk == 0) {
                throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
            }
        } else if (arg == "--use-original-int8-prefill-kernel") {
            options.original_int8_prefill_kernel = true;
        } else if (arg == "--prefill-split-workspace-mib") {
            options.prefill_split_workspace_mib = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--prefill-split-workspace-mib"), "prefill-split-workspace-mib"));
        } else if (arg == "--use-original-nvfp4-prefill-kernel") {
            options.original_nvfp4_prefill_kernel = true;
        } else if (arg == "--reproducible") {
            options.fast_paths = false;
        } else if (arg == "--lossless") {
            options.lossless = true;
        } else if (arg == "--fast-path") {
            const std::string name(require_value("--fast-path"));
            const std::optional<FastPath> path = parse_fast_path(name);
            if (!path) { throw std::invalid_argument("unknown fast path: " + name); }
            options.fast_path_candidates.push_back(*path);
        } else if (arg == "--expert-a4-activations") {
            options.expert_a4_activations = true;
        } else if (arg == "--a8-activations") {
            options.a8_activations = true;
        } else if (arg == "--embedding-host") {
            options.embedding_host = true;
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--log-colours") {
            const std::string_view value = require_value("--log-colours");
            if (value == "on") {
                options.log_colours = true;
            } else if (value == "off") {
                options.log_colours = false;
            } else {
                throw std::invalid_argument("--log-colours accepts on or off");
            }
        } else if (arg == "--log-stats-panel") {
            const std::string_view value = require_value("--log-stats-panel");
            if (value == "on") {
                options.log_stats_panel = true;
            } else if (value == "off") {
                options.log_stats_panel = false;
            } else {
                throw std::invalid_argument("--log-stats-panel accepts on or off");
            }
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-snapshot-slots") {
            options.context_cache.hybrid.device_snapshot_slots =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--device-snapshot-slots"), "device-snapshot-slots"));
            cache_option_flag = "--device-snapshot-slots";
        } else if (arg == "--cache-taps-per-request") {
            options.context_cache.hybrid.max_new_taps =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--cache-taps-per-request"), "cache-taps-per-request"));
            cache_option_flag = "--cache-taps-per-request";
        } else if (arg == "--cache-tap-ladder") {
            options.context_cache.hybrid.tap_ladder_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-ladder"), "cache-tap-ladder"));
            cache_option_flag = "--cache-tap-ladder";
        } else if (arg == "--prefix-cache-file") {
            options.context_cache.hybrid.persistent_file = require_value("--prefix-cache-file");
            if (options.context_cache.hybrid.persistent_file.empty()) {
                throw std::invalid_argument("--prefix-cache-file must not be empty");
            }
            cache_option_flag = "--prefix-cache-file";
        } else if (arg == "--prefix-cache-save-mins") {
            const int minutes = parse_nonnegative_int(require_value("--prefix-cache-save-mins"), "prefix-cache-save-mins");
            if (minutes == 0) { throw std::invalid_argument("--prefix-cache-save-mins must be positive"); }
            options.context_cache.hybrid.persistent_save_interval = std::chrono::minutes(minutes);
            cache_option_flag = "--prefix-cache-save-mins";
        } else if (arg == "--no-queue-holds") {
            options.context_cache.hybrid.queue_holds = false;
            cache_option_flag                        = "--no-queue-holds";
        } else if (arg == "--cache-tool-hold-secs") {
            const int seconds = parse_nonnegative_int(require_value("--cache-tool-hold-secs"),
                                                      "cache-tool-hold-secs");
            if (seconds > 3600) {
                throw std::invalid_argument("--cache-tool-hold-secs must be in [0,3600]");
            }
            options.context_cache.hybrid.tool_call_hold = std::chrono::seconds(seconds);
            cache_option_flag                           = "--cache-tool-hold-secs";
        } else if (arg == "--cache-next-turn-ms") {
            const int milliseconds = parse_nonnegative_int(require_value("--cache-next-turn-ms"),
                                                           "cache-next-turn-ms");
            if (milliseconds > 3'600'000) {
                throw std::invalid_argument("--cache-next-turn-ms must be in [0,3600000]");
            }
            options.context_cache.hybrid.next_turn_prefill_quiet = std::chrono::milliseconds(milliseconds);
            cache_option_flag                               = "--cache-next-turn-ms";
        } else if (arg == "--cache-host-state") {
            const std::string value = require_value("--cache-host-state");
            if (value == "fp32") {
                options.context_cache.hybrid.host_state_precision = HybridHostStatePrecision::Fp32;
            } else if (value == "fp24") {
                options.context_cache.hybrid.host_state_precision = HybridHostStatePrecision::Fp24;
            } else if (value == "bf16") {
                options.context_cache.hybrid.host_state_precision = HybridHostStatePrecision::Bf16;
            } else {
                throw std::invalid_argument("--cache-host-state must be fp32, fp24 or bf16");
            }
            cache_option_flag = "--cache-host-state";
        } else if (arg == "--cache-tap-min-gap") {
            options.context_cache.hybrid.tap_min_gap_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-min-gap"), "cache-tap-min-gap"));
            cache_option_flag = "--cache-tap-min-gap";
        } else if (arg == "--host-context-mib") {
            const char* text = require_value("--host-context-mib");
            if (std::string_view(text) == "auto") {
                options.context_cache.host_capacity_bytes.reset();
                options.context_cache.host_capacity_auto = true;
            } else {
                options.context_cache.host_capacity_bytes = parse_host_context_mib(text);
                options.context_cache.host_capacity_auto  = false;
            }
            host_context_explicit = true;
            cache_option_flag     = "--host-context-mib";
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--request-log-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--request-log-max-mib"), "request-log-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::uint64_t>::max() >> 20) {
                throw std::invalid_argument("--request-log-max-mib is out of range");
            }
            options.request_log_rotation.max_bytes = mib << 20;
        } else if (arg == "--request-log-keep") {
            const int keep =
                parse_nonnegative_int(require_value("--request-log-keep"), "request-log-keep");
            if (keep > static_cast<int>(kMaximumRequestLogKeep)) {
                throw std::invalid_argument("--request-log-keep must be in [0,1000]");
            }
            options.request_log_rotation.keep = static_cast<std::uint32_t>(keep);
            request_log_keep_explicit         = true;
        } else if (arg == "--route-trace") {
            options.route_trace = require_value("--route-trace");
            if (options.route_trace.empty()) {
                throw std::invalid_argument("--route-trace must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--ram-free-mib") {
            const std::uint64_t mib = parse_u64(require_value("--ram-free-mib"), "ram-free-mib");
            if (mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                throw std::invalid_argument("--ram-free-mib is out of range");
            }
            options.ram_free_bytes = mib << 20;
        } else if (arg == "--expert-ram-mib") {
            const std::string text = require_value("--expert-ram-mib");
            if (text == "auto") {
                options.expert_ram_bytes.reset();
            } else {
                const std::uint64_t mib = parse_u64(text.c_str(), "expert-ram-mib");
                if (mib == 0 || mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                    throw std::invalid_argument("--expert-ram-mib is out of range");
                }
                options.expert_ram_bytes = mib << 20;
            }
        } else if (arg == "--expert-ram-layout") {
            options.expert_ram_layout = parse_expert_ram_layout(require_value("--expert-ram-layout"));
        } else if (arg == "--expert-pin-mib") {
            const std::string text = require_value("--expert-pin-mib");
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
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--logits-dtype") {
            options.logits_dtype = product::parse_logits_dtype(require_value("--logits-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            product::apply_draft_tokens(options.speculative, require_value("--draft-tokens"));
        } else if (arg == "--draft-model") {
            options.speculative.draft_model = std::filesystem::path(std::string(require_value("--draft-model")));
        } else if (arg == "--spec-min-p") {
            options.speculative.draft_min_p = product::parse_draft_min_p(require_value("--spec-min-p"));
        } else if (arg == "--draft-tree-nodes") {
            product::apply_draft_tree_nodes(options.speculative,
                                            require_value("--draft-tree-nodes"));
        } else if (arg == "--draft-tree-paths") {
            options.speculative.draft_tree_paths = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tree-paths"), "draft-tree-paths"));
        } else if (arg == "--ngram-draft-tokens") {
            product::apply_ngram_draft_tokens(options.speculative, require_value("--ngram-draft-tokens"));
        } else if (arg == "--ngram-min-match") {
            options.speculative.ngram_min_match = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--ngram-min-match"), "ngram-min-match"));
        } else if (arg == "--ngram-archive-mib" || arg == "--ngram-session-mib") {
            const auto mib = parse_u64(require_value(arg.c_str()), arg.c_str());
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("ngram archive capacity is out of range");
            }
            auto& bytes = arg == "--ngram-archive-mib" ? options.speculative.ngram_archive_bytes
                                                       : options.speculative.ngram_session_bytes;
            bytes       = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--ngram-native-sessions") {
            options.ngram_native_sessions = true;
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--no-thinking-answer-room") {
            options.thinking_answer_room = false;
        } else if (arg == "--thinking-budget-message") {
            options.thinking_budget_message = require_value("--thinking-budget-message");
            if (options.thinking_budget_message.empty()) {
                throw std::invalid_argument("--thinking-budget-message must not be empty");
            }
        } else if (arg == "--reasoning-loop") {
            const std::string_view value = require_value("--reasoning-loop");
            if (value == "off") {
                options.reasoning_loop = infernix::ReasoningLoopAction::Off;
            } else if (value == "stop") {
                options.reasoning_loop = infernix::ReasoningLoopAction::Stop;
            } else if (value == "conclude") {
                options.reasoning_loop = infernix::ReasoningLoopAction::Conclude;
            } else {
                throw std::invalid_argument("--reasoning-loop must be off, stop or conclude");
            }
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-offload") {
            const std::string_view value = require_value("--vision-offload");
            if (value == "auto") {
                options.vision_offload = VisionOffload::Auto;
            } else if (value == "on") {
                options.vision_offload = VisionOffload::On;
            } else if (value == "off") {
                options.vision_offload = VisionOffload::Off;
            } else {
                throw std::invalid_argument("--vision-offload accepts auto, on or off");
            }
        } else if (arg == "--vision-max-merged") {
            const std::uint32_t merged =
                parse_nonnegative_int(require_value("--vision-max-merged"), "vision-max-merged");
            if (merged < 64 || merged > 32768) {
                throw std::invalid_argument("--vision-max-merged must be in [64, 32768]");
            }
            options.vision_max_merged_tokens = merged;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--tolerant-tool-calls") {
            options.tolerant_tool_calls = true;
        } else if (arg == "--restore-tool-reasoning") {
            options.restore_tool_reasoning = true;
        } else if (arg == "--any-model-name") {
            options.any_model_name = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--usage-chunk-choice") {
            options.usage_chunk_choice = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::logic_error("serve option " + arg + " has no parser");
        }
    }
    if (!kv_capacity_explicit) {
        // The prefix cache turns every Device page no active request holds into block cache,
        // so it sizes the KV pool to free VRAM unless a capacity is given. Without a prefix cache
        // pages beyond the active requests would sit unused.
        options.kv_capacity = options.allow_prefix_reuse
                                  ? KvCapacityPolicy::automatic()
                                  : KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (!options.allow_prefix_reuse) {
        if (cache_option_flag != nullptr) {
            throw std::invalid_argument(std::string(cache_option_flag) +
                                        " configures the prefix cache and cannot be combined "
                                        "with --no-prefix-reuse");
        }
    } else {
        std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        if (file.empty() && options.context_cache.hybrid.persistent_save_interval.count() != 0) {
            throw std::invalid_argument("--prefix-cache-save-mins saves to --prefix-cache-file, which is not set");
        }
        if (!file.empty()) {
            if (host_context_explicit && options.context_cache.host_capacity_bytes == 0) {
                throw std::invalid_argument(
                    "--prefix-cache-file saves the Host tier, which --host-context-mib 0 removes");
            }
            // Resolved now, so the save at shutdown writes where startup read, and checked now,
            // so an unusable location fails at launch rather than after a session of caching.
            file = std::filesystem::absolute(file).lexically_normal();
            std::error_code error;
            if (std::filesystem::is_directory(file, error)) {
                throw std::invalid_argument("--prefix-cache-file " + file.string() +
                                            " is a directory; name a file in it");
            }
            if (!std::filesystem::is_directory(file.parent_path(), error)) {
                throw std::invalid_argument("--prefix-cache-file " + file.string() +
                                            ": the directory " + file.parent_path().string() +
                                            " does not exist");
            }
        }
    }
    options.context_cache.enabled = options.allow_prefix_reuse;
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    // 0: the model's default (EngineOptions::prefill_chunk).
    if (options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (options.prefill_split_workspace_mib > kMaximumPrefillSplitWorkspaceMiB) {
        throw std::invalid_argument("--prefill-split-workspace-mib must be in [0,16384]");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.vision_offload == VisionOffload::On && !options.enable_vision) {
        throw std::invalid_argument("--vision-offload on requires --vision");
    }
    if (options.ngram_native_sessions && options.speculative.ngram_archive_bytes == 0) {
        throw std::invalid_argument("--ngram-native-sessions requires --ngram-archive-mib");
    }
    if (options.request_log_rotation.max_bytes != 0 && options.request_log_jsonl.empty()) {
        throw std::invalid_argument("--request-log-max-mib requires --request-log-jsonl");
    }
    if (request_log_keep_explicit && options.request_log_rotation.max_bytes == 0) {
        throw std::invalid_argument("--request-log-keep requires --request-log-max-mib");
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    return std::string(artifact_model_name);
}

} // namespace infernix::serve
