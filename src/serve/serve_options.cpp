#include "serve/serve_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

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
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
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
           "  --no-cuda-graph            disable CUDA-graph decode rounds (on by default)\n"
           "  --default-max-tokens N     default max_tokens when a request omits it\n"
           "                             (default " +
           std::to_string(kDefaultMaxTokens) + ")\n"
           "  --default-thinking-budget N  cap model-origin thinking for enabled\n"
           "                             requests; control tokens count toward the\n"
           "                             request output limit\n"
           "  --model-id ID              override the artifact metadata.name reported by\n"
           "                             the server\n"
           "  --chat-template FILE       replace the artifact frontend chat template at\n"
           "                             startup; must match a template the target accepts\n"
           "  --context-cost-presets F   runtime context-cost preset file (overrides\n"
           "                             matching compiled-in values)\n"
           "  --rope-yarn-factor F       runtime YaRN context extension factor, finite [1,4]\n"
           "                             (default 1); startup-fixed, extends the allowed\n"
           "                             ceiling only, not --max-context\n"
           "  --device N                 CUDA device ordinal (default 0)\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto       KV-cache capacity in tokens (default =\n"
           "                             --max-context; auto sizes to free VRAM,\n"
           "                             leaving " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of headroom; configurable via\n"
           "                             --kv-headroom-mib)\n"
           "                             (default " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           ")\n"
           "  --kv-dtype T               KV storage: bf16 (default) | int8 | fp8 | nvfp4 | k8v4\n"
           "  --device-state-slots N     extra Device StateImages beyond active lanes\n"
           "                             (default = --max-concurrency)\n"
           "  --host-context-mib N       shared pinned Host budget for StateImages, KV and\n"
           "                             pause snapshots, in decimal MiB that resolve to\n"
           "                             whole bytes (default 8192 MiB + eight native\n"
           "                             StateImages; 0 disables Host context backing)\n"
           "  --no-prefix-reuse          disable cross-request history; request pause/replay\n"
           "                             resources remain available\n"
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2    speculative decoding backend\n"
           "  --draft-tokens N           draft tokens per round (mtp 1-5; dflash/dflash2 1-15)\n"
           "  --lm-head-draft            use the optimized proposal head\n"
           "  --ngram-draft-tokens N     propose N verified ngram copies per round, 1-63 (0 off);\n"
           "                             above 15 requires --max-concurrency 1\n"
           "  --ngram-min-match N        minimum ngram match length, 4-64\n"
           "  --ngram-archive-mib N      MiB of retained source archive for ngram proposals\n"
           "  --ngram-session-mib N      MiB session-scoped ngram source budget (default 128);\n"
           "                             used only with --ngram-archive-mib\n"
           "  --ngram-native-sessions    retain ngram sources across compaction; requires\n"
           "                             --ngram-archive-mib\n"
           "\n"
           "VISION (off by default)\n"
           "  --vision                   enable media and load the Vision GPU allocations\n"
           "  --vision-offload on|off    keep the vision tower in pinned system RAM instead of\n"
           "                             VRAM (default off); on adds no steady-state VRAM and\n"
           "                             borrows device memory only while encoding an image;\n"
           "                             requires --vision\n"
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
           "  --response-store-max-records N Responses-state record cap (default 1024)\n"
           "  --response-store-max-mib N      Responses-state byte cap in MiB (default 256)\n"
           "  --log-stats-interval-ms N  throughput-log interval in ms\n"
           "                             (default 5000; 0 disables)\n"
           "  --log-level L              pretty stderr verbosity (trace|debug|info|warning|\n"
           "                             error|critical|off; default info)\n"
           "  --cors                     send permissive CORS headers for browser UIs\n"
           "  --usage-chunk-choice       give the streamed usage chunk a zero-delta choice so\n"
           "                             strict parsers that reject choices:[] accept it\n"
           "\n"
           "NOTES\n"
           "  --vision-offload on requires --vision.\n"
           "  --ngram-draft-tokens above 15 requires --max-concurrency 1.\n"
           "  --ngram-native-sessions requires --ngram-archive-mib.\n"
           "  --rope-yarn-factor is startup-fixed, finite [1,4] (default 1); it extends the\n"
           "  allowed ceiling only, not --max-context.\n"
           "  sampler defaults come from the loaded model and resolved thinking mode;\n"
           "  server flags and request fields override individual values.\n"
           "  --greedy forces temperature 0 (exact argmax).\n";
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
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
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
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
        } else if (arg == "--host-context-mib") {
            options.context_cache.host_capacity_bytes =
                parse_host_context_mib(require_value("--host-context-mib"));
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
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
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
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
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-offload") {
            const std::string_view value = require_value("--vision-offload");
            if (value == "on") {
                options.vision_offload = true;
            } else if (value == "off") {
                options.vision_offload = false;
            } else {
                throw std::invalid_argument("--vision-offload accepts on or off");
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
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
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
    product::validate_speculative_cli_options(options.speculative);
    if (options.vision_offload && !options.enable_vision) {
        throw std::invalid_argument("--vision-offload on requires --vision");
    }
    // A speculative decode frame is allocated at the wider of the neural and ngram draft windows
    // and cannot be narrowed for a multi-request batch. The GDN conv-record workspace admits at
    // most 16 verification columns when the batch holds more than one request, so a wider ngram
    // proposal is admitted only for a single active request.
    if (options.speculative.ngram_draft_tokens > 15 && options.max_concurrency != 1) {
        throw std::invalid_argument("--ngram-draft-tokens above 15 requires --max-concurrency 1");
    }
    if (options.ngram_native_sessions && options.speculative.ngram_archive_bytes == 0) {
        throw std::invalid_argument("--ngram-native-sessions requires --ngram-archive-mib");
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
