#pragma once

#include "infernix/types.h"
#include "product/logging/logging.h"
#include "serve/request.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::serve {

// Protocol default when the client omits max_tokens. Engine independently
// clamps the request to its effective context capacity.
inline constexpr int kDefaultMaxTokens                    = 8192;
// ServeOptions::thinking_answer_room's thresholds (as Strata 0.1.45's answer room).
inline constexpr int kThinkingAnswerRoomMinTokens         = 1024;
inline constexpr int kThinkingAnswerRoomMinReserve        = 256;
inline constexpr std::size_t kDefaultMaxRequestBytes      = 384ULL << 20;
inline constexpr std::size_t kDefaultResponseStoreRecords = 1024;
inline constexpr std::size_t kDefaultResponseStoreBytes   = 256ULL << 20;
inline constexpr std::uint32_t kDefaultRequestLogKeep     = 4;
inline constexpr std::uint32_t kMaximumRequestLogKeep     = 1000;
inline constexpr std::uint32_t kMaximumPendingRequests    = 1024;
// Seconds an idle connection stays open past the Keep-Alive timeout the server advertises, so a
// client that retires its pooled connection at the advertised deadline always closes first.
inline constexpr std::uint32_t kKeepAliveBufferSec = 10;
// Seconds a response write may wait for the client to take bytes (its socket buffers full): a
// streaming client that stops reading keeps its request this long. cpp-httplib's 5 s default
// ended such streams, with no finish, after a 5 s pause (Flare #199). A peer that stops
// acknowledging is still cut sooner by the TCP retransmission limit (http_transport.cpp).
inline constexpr std::uint32_t kWriteTimeoutSec = 60;

// Size-based rotation of the request JSONL log.
struct RequestLogRotation {
    std::uint64_t max_bytes = 0;                      // 0 => one file that grows without bound
    std::uint32_t keep      = kDefaultRequestLogKeep; // rotated files kept: PATH.1 .. PATH.keep
};

struct ServeOptions {
    bool help_requested = false;
    // --plan FILE: size a start with these options without loading the model (Engine::plan), write
    // the plan to FILE as JSON (serve/plan_report.h) and exit.
    std::filesystem::path plan_path;
    std::string artifact_path;
    std::filesystem::path chat_template_path;
    std::filesystem::path ngram_volume_path;
    NgramResidency ngram_residency = NgramResidency::Auto;
    // --expert-state FILE|off: absent uses the artifact path + .expert-state, "off" disables.
    std::optional<std::filesystem::path> expert_state;
    // --expert-stripe FILE: the experts' record tails on a second drive (EngineOptions::expert_stripe_path).
    std::filesystem::path expert_stripe_path;
    std::string host = "127.0.0.1";
    int port         = 8080;
    std::string api_key;                          // empty => no auth
    std::optional<std::string> model_id_override; // unset => artifact metadata.name
    std::string request_log_jsonl;                // empty => structured request logging disabled
    RequestLogRotation request_log_rotation;
    // --route-trace FILE: writes the model's internal per-round route trace (route_trace.h).
    std::filesystem::path route_trace;            // empty => no trace written
    float rope_yarn_factor             = 1.0F;
    std::uint32_t max_context          = 8192;
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(8192);
    std::optional<std::size_t> vram_headroom_bytes; // empty: the model's automatic headroom
    bool vram_past_budget = false;
    std::uint32_t expert_tail_skip = 0; // EngineOptions::expert_tail_skip
    bool expert_a4_activations     = false; // EngineOptions::expert_a4_activations
    bool expert_a16_activations    = false; // EngineOptions::expert_a16_activations (--a16-activations)
    bool a8_activations            = false; // EngineOptions::a8_activations
    bool embedding_host            = false; // EngineOptions::embedding_host
    std::uint32_t max_concurrency      = 1;
    // At most kMaximumPendingRequests: the HTTP pool starts a worker for each admitted or pending request.
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    // HTTP keep-alive: how long a finished response's connection stays open for the next
    // request (the HTTP server's own pool, not TCP keepalive), advertised as
    // `Keep-Alive: timeout=`; the close follows it by kKeepAliveBufferSec. The cpp-httplib
    // default of 5 s closes pooled connections between agent turns, so long-idle reuse pays a
    // reconnect and can race the close. The timeout close itself is graceful (FIN-first drain),
    // and a request that does land re-arms the window — nothing else needs to retire a
    // connection deliberately. A client that probes a pooled socket before reuse can still
    // write a request between its probe and a close in flight, and loses it (a POST is not
    // retried), so the server must not close first: clients that read the Keep-Alive header
    // retire before the buffered close; clients that do not are outlasted by the window, which
    // must exceed their pools' idle timeout (Bun's fetch, opencode's client: 5 minutes on a
    // minute wheel, so 4-5). An open connection holds a request worker, bounded by
    // max_http_clients.
    std::uint32_t keep_alive_timeout_sec = 360;
    // Ceiling on simultaneous open HTTP connections. cpp-httplib runs each connection's
    // keep-alive loop as a pool task, so an open connection — idle or not — holds a worker;
    // the pool already enforced this bound with max_n hardwired to the admission size. Base
    // (admission-sized) workers stay permanently reachable; connections up to this ceiling
    // hold a worker each; beyond it, connections queue (max_pending_requests deep) and are
    // then refused. Values below the admission requirement are raised to it at startup.
    std::uint32_t max_http_clients = 64;
    std::uint32_t prefill_chunk        = 0; // 0: the model's default (EngineOptions)
    bool original_int8_prefill_kernel  = false;
    std::uint32_t prefill_split_workspace_mib = kDefaultPrefillSplitWorkspaceMiB;
    bool original_nvfp4_prefill_kernel = false;
    bool fast_paths                    = true; // off with --reproducible or --lossless (EngineOptions)
    bool lossless                      = false; // --lossless (EngineOptions)
    // --fast-path NAME (repeatable): also run these, qualified or not (qualification and timing
    // runs; EngineOptions::fast_path_candidates).
    std::vector<FastPath> fast_path_candidates;
    // --no-fast-path NAME (repeatable): leave these selected paths out (leave-one-out timing runs;
    // EngineOptions::fast_path_exclusions).
    std::vector<FastPath> fast_path_exclusions;
    std::filesystem::path context_cost_presets;
    std::uint32_t log_stats_interval_ms    = 5000; // 0 disables periodic Engine throughput logs
    std::size_t max_request_bytes          = kDefaultMaxRequestBytes;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint64_t ram_free_bytes           = kDefaultRamFreeBytes;
    bool ram_resize                        = true; // --ram-resize auto|off (infernix::EngineOptions::ram_resize)
    std::optional<std::uint64_t> expert_ram_bytes; // --expert-ram-mib; empty: the RAM ledger's share
    std::optional<ExpertRamLayout> expert_ram_layout; // --expert-ram-layout (empty: the model's default)
    bool expert_csf = false;                          // --expert-csf
    std::optional<std::uint64_t> expert_pin_bytes; // --expert-pin-mib; empty: what the OS lets it pin
    std::uint32_t media_preprocess_threads = 0;
    std::size_t response_store_max_records = kDefaultResponseStoreRecords;
    std::size_t response_store_max_bytes   = kDefaultResponseStoreBytes;
    int device                             = 0;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    LogitsDtype logits_dtype               = LogitsDtype::FP32;
    SpeculativeOptions speculative;
    bool ngram_native_sessions = false;
    ContextCacheOptions context_cache;
    bool enable_vision                     = false;
    VisionOffload vision_offload                    = VisionOffload::Auto;
    std::uint32_t vision_max_merged_tokens = 32768;
    bool use_cuda_graph                    = true;
    bool allow_prefix_reuse                = true;
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
    // Recover complete Qwen calls with malformed wrapper/suffix output (opt-in; strict by default).
    bool tolerant_tool_calls = false;
    // --restore-reasoning: a Chat Completions assistant message whose client dropped the reasoning of a
    // response this server produced gets it back (serve/reasoning_store.h): a tool-call message by its call
    // ids, an answer by its text and the conversation before it. The history then renders as generated and
    // keeps its prefix-cache match. Off by default: it changes the history the client sent.
    bool restore_reasoning = false;
    // --any-model-name: the OpenAI routes serve a request whatever non-empty model name it gives (a
    // client's hard-coded or default name) and report the served model's name; off, any name but the
    // public model ID is a 404 model_not_found.
    bool any_model_name = false;
    // --claude-code-prompt as-sent|local: Anthropic Messages from Claude Code keep its System prompt,
    // or have its Claude-specific parts replaced by a short local-agent preamble.
    ClaudeCodePrompt claude_code_prompt = ClaudeCodePrompt::AsSent;
    std::optional<std::uint32_t> default_thinking_budget;
    // The answer room: from kThinkingAnswerRoomMinTokens of output limit (max_tokens, or the context
    // the prompt leaves when that is less), thinking closes early enough to leave a quarter of that
    // limit (at least kThinkingAnswerRoomMinReserve, at most half) for the answer, whether the
    // request has no budget, the server's default_thinking_budget, or a budget of its own at or above
    // the limit: the limit counts the thinking, so a long thought would otherwise end the request
    // with no answer. A request's own budget below the limit stands. See settle_thinking_budget.
    // --no-thinking-answer-room turns it off.
    bool thinking_answer_room = true;
    // End-of-thinking message fed to the model when it hits the thinking budget; empty
    // preserves the model's built-in control suffix.
    std::string thinking_budget_message;
    // --reasoning-loop off|stop|conclude: the reasoning-loop guard for thinking requests without a
    // token constraint (off by default).
    infernix::ReasoningLoopAction reasoning_loop = infernix::ReasoningLoopAction::Off;
    // --end-in-thinking refuse|allow: whether a turn may end while its thinking is open (refuse by default:
    // ThinkingControlOptions::end_turn_in_thinking off).
    bool end_turn_in_thinking = false;
    int default_max_tokens = kDefaultMaxTokens;
    bool enable_cors       = false; // send permissive CORS headers for browser UIs
    bool log_colours       = false; // --log-colours on|off: colour the console stats lines
    // --log-stats-panel on|off: pin the session statistics beneath the console log (terminal only).
    bool log_stats_panel = true;
    // --usage-chunk-choice: emit the streaming usage chunk with a zero-delta choice instead of the
    // OpenAI-conformant empty choices array. Strict client parsers (GitHub Copilot) reject the
    // empty array as "Response contained no choices"; the extra choice is inert for other clients.
    bool usage_chunk_choice = false;
    // --claude-code-usage-compat: Anthropic streaming only. An affected Claude Code version keeps the
    // positive `message_start.usage.input_tokens` it saw first even when the terminal
    // `message_delta` corrects that field to 0 (anthropics/claude-code#103901), so a prompt this
    // request wrote into the prefix cache is counted twice. Here the start's `input_tokens` is a
    // provisional 0 instead, so the exact uncached count then travels only in the terminal
    // `message_delta`: a stream the generation breaks off, or a transfer the client closes before
    // it, carries that count nowhere in either mode. The start's cache-read split and the Engine's
    // own accounting stay exact. Off by default: a client that reads the start value for early
    // context warnings or rate limits then sees the prompt's uncached tokens only once the stream
    // ends.
    bool claude_code_usage_compat = false;
    // Process-level explicit overrides layered between registered model/mode defaults and request
    // fields. An omitted seed is replaced per request with a fresh random seed.
    SamplingOverrides sampling_overrides;
    bool greedy                 = false; // --greedy: force temperature 0 (exact argmax)
    product::LogLevel log_level = product::LogLevel::Info;
    std::string log_file; // --log-file PATH: duplicate log records to a plain file beside the console

    // Exact process argv for the server-start record. Secret-bearing option values are redacted
    // while parsing; this is provenance only and never affects execution.
    std::vector<std::string> startup_argv;
    // --config FILE and the options it gave, secret-bearing values redacted (provenance only).
    std::optional<std::string> config_path;
    std::vector<std::string> config_arguments;
};

// argv[1] is the artifact path and the rest are options. `--config FILE` reads further options, and
// the artifact path when the command line has none, from a TOML file keyed by their long names
// without the dashes (product/config/config_file.h); options on the command line override the
// file, and the file overrides the defaults.
ServeOptions parse_serve_options(int argc, char** argv);
// "off", "stop" or "conclude".
const char* reasoning_loop_name(infernix::ReasoningLoopAction action) noexcept;
std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_name);
std::string serve_usage_text(const char* argv0);
// Whether a --host value accepts connections from this machine only (127.0.0.0/8, ::1, localhost).
bool is_loopback_host(std::string_view host) noexcept;

} // namespace infernix::serve
