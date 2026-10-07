#include "serve/serve_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::serve {
namespace {

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
           " <model.ninfer> [options]\n"
           "\n"
           "Serves the OpenAI Responses/Chat Completions and Anthropic Messages APIs.\n"
           "  --help, -h                 show this help and exit\n"
           "\n"
           "MODEL & CONTEXT\n"
           "  --max-context N            max context tokens per request (default 8192)\n"
           "  --max-concurrency N        max concurrent sequences, 1-8 (default 1)\n"
           "  --prefill-chunk N          prefill chunk size in tokens, multiple of 128\n"
           "                             (default 1024)\n"
           "  --use-original-int8-prefill-kernel\n"
           "                             prefill INT8 KV with the original prompt kernel at\n"
           "                             the requested chunk (default: the fast kernel, chunk\n"
           "                             rounded down to whole attention waves)\n"
           "  --prefill-8bit-pv          INT8 and K4V2 KV: run prompt attention's P*V on\n"
           "                             8-bit Tensor Cores (up to 5 % (INT8) and 7 % (K4V2)\n"
           "                             faster long-prompt prefill, but a measurable KL\n"
           "                             divergence from a BF16 KV reference: twice FP16\n"
           "                             P*V's for INT8, 1.09x for K4V2, so FP16 P*V is\n"
           "                             the default)\n"
           "  --no-prefill-8bit-pv       NVFP4, K8V4 and VQ2 KV: run prompt attention's\n"
           "                             P*V in FP16 instead of 8 bits (5-7 % slower long-\n"
           "                             prompt prefill; the default 8-bit form stays\n"
           "                             within 1.1x FP16's KL divergence from BF16 KV)\n"
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
           "  --no-cuda-graph            disable CUDA-graph decode rounds (on by default)\n"
           "  --default-max-tokens N     default max_tokens when a request omits it\n"
           "                             (default " +
           std::to_string(kDefaultMaxTokens) +
           ")\n"
           "  --default-thinking-budget N  cap model-origin thinking for enabled\n"
           "                             requests; control tokens count toward the\n"
           "                             request output limit\n"
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
           "  --chat-template FILE       replace the artifact frontend chat template at\n"
           "                             startup; must match a template the target accepts\n"
           "  --ngram-volume FILE        Qwen3.8-Flash-Next n-gram embedding volume\n"
           "                             (default: the artifact path + .ngram)\n"
           "  --expert-state FILE|off    Qwen3.8-Flash-Next expert-cache state: read at\n"
           "                             startup to fill the VRAM expert cache, written at\n"
           "                             stop and every 10 minutes between requests\n"
           "                             (default: the artifact path + .expert-state)\n"
           "  --context-cost-presets F   runtime context-cost preset file (overrides\n"
           "                             matching compiled-in values)\n"
           "  --rope-yarn-factor F       runtime YaRN context extension factor, finite [1,4]\n"
           "                             (default 1); startup-fixed, extends the allowed\n"
           "                             ceiling only, not --max-context\n"
           "  --device N                 CUDA device ordinal (default 0)\n"
           "  --ram-headroom-mib N       physical Host RAM in MiB that pinned weights and\n"
           "                             caches leave free for the system (default " +
           std::to_string(kDefaultRamHeadroomBytes >> 20) +
           ")\n"
           "  --expert-ram-mib N|auto    Qwen3.8-Flash-Next: Host RAM in MiB for the routed\n"
           "                             experts (auto: what --ram-headroom-mib and the other\n"
           "                             pins leave); less than all of them need reads the\n"
           "                             rest from the artifact (the SSD tier)\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto       KV-cache capacity in tokens (default auto, or\n"
           "                             --max-context with the original prefix caching\n"
           "                             system or --no-prefix-reuse; auto sizes to free\n"
           "                             VRAM less --vram-headroom-mib; Qwen3.8-Flash-Next:\n"
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
           "  --kv-dtype T               KV storage: bf16 (default) | int8 | fp8 | nvfp4 | k8v4 |\n"
           "                             vq2 | k4v2\n"
           "  --host-context-mib N       pinned Host budget in decimal MiB that resolve to\n"
           "                             whole bytes. Hybrid system: the pool KV blocks and\n"
           "                             state snapshots share (default 8192; 0 = GPU only).\n"
           "                             Original system: the shared budget for StateImages,\n"
           "                             KV and pause snapshots (default 8192 MiB + eight\n"
           "                             native StateImages; 0 disables Host context backing).\n"
           "                             Qwen3.8-Flash-Next: default 4096, must be positive\n"
           "  --no-prefix-reuse          disable prefix caching in either system below;\n"
           "                             request pause/replay resources remain available;\n"
           "                             cannot be combined with any prefix-cache option\n"
           "\n"
           "NEW PREFIX CACHING SYSTEM (hybrid; the default)\n"
           "  Content-addressed 64-token KV blocks shared across requests plus sparse model\n"
           "  state snapshots. It configures itself: free VRAM becomes Device block cache\n"
           "  (--kv-capacity defaults to auto) and every option below is optional.\n"
           "  --prefix-cache-file PATH   restore the host tier from PATH at startup when\n"
           "                             present (written by this binary for the same\n"
           "                             artifact and KV format) and save it there on clean\n"
           "                             shutdown (Ctrl+C twice); relative paths resolve\n"
           "                             against the launch directory (default off:\n"
           "                             nothing is saved)\n"
           "  --device-snapshot-slots N  device state snapshot slots (default concurrency\n"
           "                             + 1; + 2 without a host tier). Qwen3.8-Flash-\n"
           "                             Next has none\n"
           "  --cache-taps-per-request N new prefill snapshots per request (default 8;\n"
           "                             2 without a host tier)\n"
           "  --cache-tap-ladder N       ladder base in tokens for history snapshots\n"
           "                             (default max(4096, 2x prefill chunk))\n"
           "  --cache-tap-min-gap N      minimum tokens between ladder snapshots\n"
           "                             (default max(1024, prefill chunk))\n"
           "\n"
           "ORIGINAL PREFIX CACHING SYSTEM (upstream's continuation/checkpoint cache;\n"
           "Qwen3.5 artifacts only)\n"
           "  --use-original-prefix-caching\n"
           "                             use this system instead of the new one; the\n"
           "                             option below requires it. --kv-capacity then\n"
           "                             defaults to --max-context\n"
           "  --device-state-slots N     extra Device StateImages beyond active lanes\n"
           "                             (default = --max-concurrency)\n"
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2    speculative decoding backend\n"
           "  --draft-tokens N           draft tokens per round (mtp 1-5; dflash/dflash2 1-15)\n"
           "  --lm-head-draft            use the optimized proposal head\n"
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
           "  --ngram-draft-tokens N     propose N verified ngram copies per round, 1-63 (0 off)\n"
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
           "                             (default auto: on for Qwen3.8-Flash-Next, off for\n"
           "                             Qwen3.5); on requires --vision\n"
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
           "                             budget and an undeclared name (strict all-or-nothing\n"
           "                             by default)\n"
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

const char* reasoning_loop_name(ninfer::ReasoningLoopAction action) noexcept {
    switch (action) {
    case ninfer::ReasoningLoopAction::Stop: return "stop";
    case ninfer::ReasoningLoopAction::Conclude: return "conclude";
    case ninfer::ReasoningLoopAction::Off: break;
    }
    return "off";
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
    bool host_context_explicit   = false;
    bool original_cache_selected = false;
    // Last flag seen that belongs to only one prefix-cache mode, for the cross-mode error.
    const char* original_cache_flag = nullptr;
    const char* hybrid_option_flag  = nullptr;
    // The hybrid prefix cache is the server default; --use-original-prefix-caching selects the
    // original one.
    options.context_cache.mode = ContextCacheMode::Hybrid;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
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
        } else if (arg == "--expert-state") {
            const std::string path = require_value("--expert-state");
            options.expert_state   = path == "off" ? std::filesystem::path{} : std::filesystem::path(path);
        } else if (arg == "--chat-template") {
            options.chat_template_path = require_value("--chat-template");
            if (options.chat_template_path.empty()) {
                throw std::invalid_argument("--chat-template must not be empty");
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
        } else if (arg == "--use-original-int8-prefill-kernel") {
            options.original_int8_prefill_kernel = true;
        } else if (arg == "--prefill-8bit-pv") {
            options.prefill_8bit_pv = PrefillPv8::On;
        } else if (arg == "--no-prefill-8bit-pv") {
            options.prefill_8bit_pv = PrefillPv8::Off;
        } else if (arg == "--prefill-split-workspace-mib") {
            options.prefill_split_workspace_mib = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--prefill-split-workspace-mib"), "prefill-split-workspace-mib"));
        } else if (arg == "--use-original-nvfp4-prefill-kernel") {
            options.original_nvfp4_prefill_kernel = true;
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
        } else if (arg == "--use-original-prefix-caching") {
            options.context_cache.mode = ContextCacheMode::Original;
            original_cache_selected    = true;
        } else if (arg == "--device-snapshot-slots") {
            options.context_cache.hybrid.device_snapshot_slots =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--device-snapshot-slots"), "device-snapshot-slots"));
            hybrid_option_flag = "--device-snapshot-slots";
        } else if (arg == "--cache-taps-per-request") {
            options.context_cache.hybrid.max_new_taps =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--cache-taps-per-request"), "cache-taps-per-request"));
            hybrid_option_flag = "--cache-taps-per-request";
        } else if (arg == "--cache-tap-ladder") {
            options.context_cache.hybrid.tap_ladder_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-ladder"), "cache-tap-ladder"));
            hybrid_option_flag = "--cache-tap-ladder";
        } else if (arg == "--prefix-cache-file") {
            options.context_cache.hybrid.persistent_file = require_value("--prefix-cache-file");
            if (options.context_cache.hybrid.persistent_file.empty()) {
                throw std::invalid_argument("--prefix-cache-file must not be empty");
            }
            hybrid_option_flag = "--prefix-cache-file";
        } else if (arg == "--cache-tap-min-gap") {
            options.context_cache.hybrid.tap_min_gap_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--cache-tap-min-gap"), "cache-tap-min-gap"));
            hybrid_option_flag = "--cache-tap-min-gap";
        } else if (arg == "--device-state-slots") {
            original_cache_flag                      = "--device-state-slots";
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
        } else if (arg == "--host-context-mib") {
            options.context_cache.host_capacity_bytes =
                parse_host_context_mib(require_value("--host-context-mib"));
            host_context_explicit = true;
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
        } else if (arg == "--ram-headroom-mib") {
            const std::uint64_t mib = parse_u64(require_value("--ram-headroom-mib"), "ram-headroom-mib");
            if (mib > (std::numeric_limits<std::uint64_t>::max() >> 20)) {
                throw std::invalid_argument("--ram-headroom-mib is out of range");
            }
            options.ram_headroom_bytes = mib << 20;
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
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--draft-tree-nodes") {
            product::apply_draft_tree_nodes(options.speculative,
                                            require_value("--draft-tree-nodes"));
        } else if (arg == "--draft-tree-paths") {
            options.speculative.draft_tree_paths = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tree-paths"), "draft-tree-paths"));
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--ngram-draft-tokens"), "ngram-draft-tokens"));
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
        } else if (arg == "--thinking-budget-message") {
            options.thinking_budget_message = require_value("--thinking-budget-message");
            if (options.thinking_budget_message.empty()) {
                throw std::invalid_argument("--thinking-budget-message must not be empty");
            }
        } else if (arg == "--reasoning-loop") {
            const std::string_view value = require_value("--reasoning-loop");
            if (value == "off") {
                options.reasoning_loop = ninfer::ReasoningLoopAction::Off;
            } else if (value == "stop") {
                options.reasoning_loop = ninfer::ReasoningLoopAction::Stop;
            } else if (value == "conclude") {
                options.reasoning_loop = ninfer::ReasoningLoopAction::Conclude;
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
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!kv_capacity_explicit) {
        // The hybrid cache turns every Device page no active request holds into block cache, so
        // it sizes the KV pool to free VRAM unless a capacity is given. Without a prefix cache
        // pages beyond the active requests would sit unused.
        options.kv_capacity =
            options.allow_prefix_reuse && options.context_cache.mode == ContextCacheMode::Hybrid
                ? KvCapacityPolicy::automatic()
                : KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (!options.allow_prefix_reuse) {
        if (original_cache_selected) {
            throw std::invalid_argument(
                "--use-original-prefix-caching cannot be combined with --no-prefix-reuse");
        }
        if (hybrid_option_flag != nullptr) {
            throw std::invalid_argument(std::string(hybrid_option_flag) +
                                        " configures the hybrid prefix cache and cannot be "
                                        "combined with --no-prefix-reuse");
        }
        // Without reuse the original system's manager runs disabled: it still owns request
        // pause/replay resources (--device-state-slots, --host-context-mib).
        options.context_cache.mode = ContextCacheMode::Original;
    } else if (options.context_cache.mode == ContextCacheMode::Hybrid) {
        if (original_cache_flag != nullptr) {
            throw std::invalid_argument(std::string(original_cache_flag) +
                                        " configures the original prefix cache and requires "
                                        "--use-original-prefix-caching");
        }
        std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
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
    } else if (hybrid_option_flag != nullptr) {
        throw std::invalid_argument(std::string(hybrid_option_flag) +
                                    " configures the hybrid prefix cache and cannot be combined "
                                    "with --use-original-prefix-caching");
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
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
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

} // namespace ninfer::serve
