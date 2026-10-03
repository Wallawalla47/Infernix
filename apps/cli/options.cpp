#include "options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ninfer::cli {
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
           " <model.ninfer> (--prompt <text>|--messages <messages.json>)\n"
           "       [--max-context N] [--kv-capacity N|auto] [--prefill-chunk N] [--max-new N]\n"
           "       [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2] [--spec mtp|dflash|dflash2 --draft-tokens "
           "N]\n"
           "       [--lm-head-draft] [--draft-tree-nodes auto|N] [--draft-tree-paths N]\n"
           "       [--ngram-draft-tokens 1..63] [--ngram-min-match 4..64]\n"
           "       [--temperature F] [--top-p F] [--top-k N] [--min-p F]\n"
           "       [--presence-penalty F] [--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--stop-token-id N]... [--stop <text>]... [--reasoning-stop <text>]...\n"
           "       [--chat-template FILE]\n"
           "       [--raw-output] [--print-token-ids] [--no-thinking] [--thinking-budget N]\n"
           "       [--reasoning-effort none|minimal|low|medium|high|xhigh|max]\n"
           "       [--vision] [--vision-offload on|off] [--vision-max-merged N]\n"
           "       [--no-cuda-graph]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "\n"
           "Runs one generation: answer content streams to stdout, reasoning and\n"
           "diagnostics to stderr. Sampling defaults come from the loaded model and\n"
           "thinking mode; flags override individual fields.\n"
           "\n"
           "CONTEXT\n"
           "  --max-context N          max context tokens (default 2048)\n"
           "  --rope-yarn-factor F     startup-fixed YaRN, finite [1,4] (default 1);\n"
           "                           extends allowed ceiling only, not --max-context\n"
           "  --prefill-chunk N        prefill chunk size in tokens, multiple of 128\n"
           "  --max-new N              cap on generated tokens\n"
           "  --device N               CUDA device ordinal (default 0)\n"
           "\n"
           "KV CACHE\n"
           "  --kv-capacity N|auto     KV-cache capacity in tokens\n"
           "                           (default = --max-context; auto sizes to free\n"
           "                           VRAM, leaving " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB headroom; configurable\n"
           "                           via --vram-headroom-mib)\n"
           "  --vram-headroom-mib N    VRAM headroom in MiB left by --kv-capacity auto\n"
           "                           (default " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           ")\n"
           "  --kv-dtype T             bf16 (default) | int8 | fp8 | nvfp4 | k8v4 | vq2 | k4v2\n"
           "  --use-original-int8-prefill-kernel\n"
           "                           prefill INT8 KV with the original prompt kernel\n"
           "                           (default: the fast kernel)\n"
           "  --prefill-8bit-pv        INT8 and K4V2 KV: run prompt attention's P*V on 8-bit\n"
           "                           Tensor Cores (up to 5-7 % faster long-prompt prefill,\n"
           "                           but a measurable KL divergence from a BF16 KV\n"
           "                           reference, so FP16 P*V is the default)\n"
           "  --no-prefill-8bit-pv     NVFP4, K8V4 and VQ2 KV: run prompt attention's P*V in\n"
           "                           FP16 instead of 8 bits (5-7 % slower long-prompt\n"
           "                           prefill; the default 8-bit form stays within 1.1x\n"
           "                           FP16's KL divergence from BF16 KV)\n"
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
           "\n"
           "SPECULATIVE DECODING (off by default)\n"
           "  --spec mtp|dflash|dflash2 speculative backend\n"
           "  --draft-tokens N         draft tokens per round (mtp 1-5; dflash 1-15)\n"
           "  --lm-head-draft          use the optimized proposal head\n"
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
           "  --raw-output             emit raw content without framing\n"
           "  --print-token-ids        also print generated token ids\n"
           "\n"
           "VISION (off by default)\n"
           "  --vision                 enable image/video input\n"
           "  --vision-offload on|off  keep the vision tower in pinned system RAM instead of\n"
           "                           VRAM (default off; on adds no steady-state VRAM)\n"
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
    if (argc < 2) { throw std::invalid_argument(".ninfer model path is required"); }
    options.artifact_path     = argv[1];
    bool kv_capacity_explicit = false;
    std::optional<std::size_t> vram_headroom_mib;

    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto value = [&](std::string_view flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };

        if (arg == "--prompt") {
            options.prompt = value(arg);
        } else if (arg == "--chat-template") {
            options.chat_template_path = value(arg);
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
        } else if (arg == "--vram-headroom-mib") {
            const std::uint64_t mib = parse_u64(value(arg), "vram-headroom-mib");
            if (mib > (std::numeric_limits<std::size_t>::max() >> 20)) {
                throw std::invalid_argument("--vram-headroom-mib is out of range");
            }
            vram_headroom_mib = static_cast<std::size_t>(mib);
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = parse_u32(value(arg), "prefill-chunk");
        } else if (arg == "--device") {
            options.device = parse_device(value(arg));
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_cache(value(arg));
        } else if (arg == "--spec") {
            options.speculative.backend = product::parse_speculative_backend(value(arg));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = parse_u32(value(arg), "draft-tokens");
        } else if (arg == "--ngram-draft-tokens") {
            options.speculative.ngram_draft_tokens =
                parse_u32(value(arg), "ngram-draft-tokens", true);
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
        } else if (arg == "--reasoning-effort") {
            options.reasoning_effort = parse_reasoning_effort(value(arg));
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-offload") {
            const std::string_view mode = value(arg);
            if (mode == "on") {
                options.vision_offload = true;
            } else if (mode == "off") {
                options.vision_offload = false;
            } else {
                throw std::invalid_argument("--vision-offload accepts on or off");
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
        } else if (arg == "--prefill-8bit-pv") {
            options.prefill_8bit_pv = PrefillPv8::On;
        } else if (arg == "--no-prefill-8bit-pv") {
            options.prefill_8bit_pv = PrefillPv8::Off;
        } else if (arg == "--prefill-split-workspace-mib") {
            options.prefill_split_workspace_mib =
                parse_u32(value(arg), "prefill-split-workspace-mib", true);
        } else if (arg == "--use-original-nvfp4-prefill-kernel") {
            options.original_nvfp4_prefill_kernel = true;
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
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }

    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (vram_headroom_mib.has_value()) {
        if (options.kv_capacity.mode != KvCapacityMode::Automatic) {
            throw std::invalid_argument("--vram-headroom-mib requires --kv-capacity auto");
        }
        options.kv_capacity = KvCapacityPolicy::automatic(*vram_headroom_mib << 20);
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

} // namespace ninfer::cli
