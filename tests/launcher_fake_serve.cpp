// A stand-in infernix-serve for the launcher's own checks (docs/launcher.md, "Web UI development"):
// it answers the routes the launcher uses (/health, /metrics?format=json with live progress and
// labelled request rows, the OpenAI and Anthropic generation routes with thinking, prompt progress
// and llama-server timings, --help, --plan) with plausible, slowly varying figures, logs in the
// server's record format, and misbehaves on request so the watchdog can be exercised without a GPU:
//
//   POST /fake/crash?code=N   exit at once with status N (default: an access violation)
//   POST /fake/hang           stop answering anything (the process stays alive)
//   POST /fake/stall          keep answering, but running requests stop producing tokens
//   --fake-load-ms N          take N ms to "load" (/health answers 503 meanwhile)
//   --fake-fail-load TEXT     fail the load with TEXT as its fatal record (exit status 1)
//   --fake-hog-mib N          while loading, take and touch N MiB of RAM (the RAM-floor check)
//   --fake-experts            report an expert cache (offloaded-expert models)
//   --plan FILE               write a plan (fits up to --max-context 225280) and exit
//   INFERNIX_FAKE_PROBE=TEXT  logged at startup (a configuration's environment reaches the server)
//   INFERNIX_FAKE_BIND_FAIL_ONCE=FILE  refuse the port once, as infernix-serve does when another
//                             program holds it, and note it in FILE (later starts bind normally)
//
// A port that is really held is refused with infernix-serve's own message.
//
// Everything else on the command line (the artifact, launch parameters) is accepted and ignored,
// except --host, --port, --model-id and --max-context.

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#endif

namespace {

using Json  = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr double kPrefillRate = 3200.0; // tokens/s
constexpr auto kTokenEvery    = std::chrono::microseconds(6700);

std::mutex g_log_mutex;

void log(const char* level, const std::string& message) {
    const auto now         = std::chrono::system_clock::now();
    const std::time_t secs = std::chrono::system_clock::to_time_t(now);
    const auto millis      = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &secs);
#else
    localtime_r(&secs, &local);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &local);
    std::lock_guard lock(g_log_mutex);
    std::printf("%s.%03d  %-5s  %s\n", stamp, static_cast<int>(millis), level, message.c_str());
    std::fflush(stdout);
}

double unix_now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

// One finished request's figures (the server's session_stats rows sum them; `requests` rows list them).
struct Sample {
    std::uint64_t id = 0;
    double time      = 0;
    std::string label;
    std::string protocol;
    double ttft_s          = 0;
    std::uint64_t prompt   = 0;
    std::uint64_t cached   = 0;
    double prefill_s       = 0;
    std::uint64_t output   = 0;
    double decode_s        = 0;
    std::uint64_t drafted  = 0;
    std::uint64_t accepted = 0;
    std::uint64_t lookups  = 0;
    std::uint64_t hits     = 0;
};

// A request in the server: its progress, as live.active reports it.
struct Active {
    std::uint64_t id = 0;
    std::string label;
    std::string protocol;
    std::string phase = "prefill";
    std::uint64_t prompt = 0, cached = 0, processed = 0, output = 0;
    Clock::time_point start = Clock::now();
};

struct State {
    std::mutex mutex;
    std::condition_variable cv;
    bool ready   = false;
    bool hung    = false;
    bool stalled = false;
    bool experts = false;
    std::uint64_t next_id        = 1;
    std::uint64_t completed      = 0;
    std::uint64_t prefill_tokens = 0;
    std::uint64_t output_tokens  = 0;
    std::map<std::uint64_t, Active> active;
    std::deque<Sample> samples;
    std::string model = "fake-model";
    std::mt19937 random{12345};
    double start_unix = 0;
    std::string instance_id; // unique per process, like the server's
};

State g_state;

// Blocks for ever once /fake/hang was called (the hang the watchdog must notice).
void hang_point() {
    std::unique_lock lock(g_state.mutex);
    g_state.cv.wait(lock, [] { return !g_state.hung; });
}

Json stats_row(const std::deque<Sample>& samples, std::size_t last) {
    Sample sum;
    std::size_t count = 0;
    for (auto it = samples.rbegin(); it != samples.rend() && count < last; ++it, ++count) {
        sum.ttft_s += it->ttft_s;
        sum.prompt += it->prompt;
        sum.cached += it->cached;
        sum.prefill_s += it->prefill_s;
        sum.output += it->output;
        sum.decode_s += it->decode_s;
        sum.drafted += it->drafted;
        sum.accepted += it->accepted;
        sum.lookups += it->lookups;
        sum.hits += it->hits;
    }
    const auto ratio = [](double a, double b) { return b > 0 ? Json(a / b) : Json(nullptr); };
    const std::uint64_t computed = sum.prompt - sum.cached;
    return {{"requests", count},
            {"ttft_s", count ? Json(sum.ttft_s / static_cast<double>(count)) : Json(nullptr)},
            {"prompt_tokens", sum.prompt},
            {"cached_tokens", sum.cached},
            {"cache_hit_rate", ratio(static_cast<double>(sum.cached), static_cast<double>(sum.prompt))},
            {"prefill_tokens", computed},
            {"prefill_s", sum.prefill_s},
            {"prefill_tok_s", ratio(static_cast<double>(computed), sum.prefill_s)},
            {"decode_tokens", sum.output},
            {"decode_s", sum.decode_s},
            {"decode_tok_s", ratio(static_cast<double>(sum.output), sum.decode_s)},
            {"batch", count ? Json(1.0) : Json(nullptr)},
            {"drafted", sum.drafted},
            {"accepted", sum.accepted},
            {"acceptance", ratio(static_cast<double>(sum.accepted), static_cast<double>(sum.drafted))},
            {"accepted_per_round", ratio(static_cast<double>(sum.accepted), static_cast<double>(sum.drafted) / 7.0)},
            {"ngram_rounds", 0},
            {"ngram_drafted", 0},
            {"ngram_accepted", 0},
            {"ngram_acceptance", nullptr},
            {"archive_drafted", 0},
            {"archive_accepted", 0},
            {"archive_acceptance", nullptr},
            {"expert_lookups", sum.lookups},
            {"expert_hits", sum.hits},
            {"expert_hit_rate", ratio(static_cast<double>(sum.hits), static_cast<double>(sum.lookups))}};
}

Json metrics_json() {
    std::lock_guard lock(g_state.mutex);
    const Json recent = stats_row(g_state.samples, 10);
    Json active       = Json::array();
    for (const auto& [id, request] : g_state.active) {
        active.push_back({{"id", id},
                          {"label", request.label.empty() ? Json(nullptr) : Json(request.label)},
                          {"protocol", request.protocol},
                          {"phase", request.phase},
                          {"prompt_tokens", request.prompt},
                          {"cached_tokens", request.cached},
                          {"processed_tokens", request.processed},
                          {"output_tokens", request.output},
                          {"elapsed_ms", std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - request.start).count()}});
    }
    Json rows = Json::array();
    // As infernix-serve: the last 128 settled requests, oldest first.
    const std::size_t first = g_state.samples.size() > 128 ? g_state.samples.size() - 128 : 0;
    for (auto it = g_state.samples.begin() + static_cast<std::ptrdiff_t>(first); it != g_state.samples.end(); ++it) {
        rows.push_back({{"id", it->id},
                        {"time", it->time},
                        {"protocol", it->protocol},
                        {"finish", "stop"},
                        {"label", it->label.empty() ? Json(nullptr) : Json(it->label)},
                        {"prompt_tokens", it->prompt},
                        {"reused", it->cached},
                        {"output_tokens", it->output},
                        {"ttft_ms", it->ttft_s * 1000.0},
                        {"prompt_ms", it->prefill_s * 1000.0},
                        {"decode_ms", it->decode_s * 1000.0},
                        // As infernix-serve: tokens after the first over the decode time.
                        {"decode_tok_s", it->decode_s > 0.0
                                             ? Json(static_cast<double>(it->output > 0 ? it->output - 1 : 0) / it->decode_s)
                                             : Json(nullptr)}});
    }
    const std::uint64_t running = g_state.active.size();
    return {{"schema", 1},
            {"time", unix_now()},
            {"engine",
             {{"name", "infernix"},
              {"model", g_state.model},
              {"architecture", "fake"},
              {"weights", "fake"},
              {"kv", "int8"},
              {"speculative_backend", "dflash2"},
              {"max_context", 262144},
              {"max_concurrency", 2}}},
            {"live",
             {{"state", running > 0 ? "generating" : "idle"},
              {"running", running},
              {"waiting", 0},
              {"prefilling", 0},
              {"decode_ready", running},
              {"paused", 0},
              {"replaying", 0},
              {"materializing", 0},
              {"prompt_tokens", nullptr},
              {"tok_s", recent["decode_tok_s"]},
              {"tok_s_mean", recent["decode_tok_s"]},
              {"prefill_tok_s_mean", recent["prefill_tok_s"]},
              {"start_time", g_state.start_unix},
              {"instance_id", g_state.instance_id},
              {"active", active}}},
            {"totals",
             {{"requests", g_state.completed},
              {"completed", g_state.completed},
              {"cancelled", 0},
              {"failed", 0},
              {"rejected", 0},
              {"prompt_tokens", g_state.prefill_tokens},
              {"computed_prefill_tokens", g_state.prefill_tokens},
              {"output_tokens", g_state.output_tokens},
              {"decode_tokens", g_state.output_tokens},
              {"replayed_tokens", 0}}},
            {"requests", rows},
            {"session_stats",
             {{"recent_window", 10},
              {"drafter", "dflash2"},
              {"ngram_enabled", false},
              {"completed", g_state.completed},
              {"failed", 0},
              {"cancelled", 0},
              {"rejected", 0},
              {"session", stats_row(g_state.samples, g_state.samples.size())},
              {"recent", recent}}}};
}

std::string sse(const std::string& event, const Json& data) {
    return (event.empty() ? std::string() : "event: " + event + "\n") + "data: " + data.dump() + "\n\n";
}

// One generation, step by step: prompt (in 100 ms slices, with progress), thinking, answer.
struct Generation {
    std::string protocol;
    bool stream = false, progress = false, per_token = false, usage = false;
    std::uint64_t id = 0;
    std::uint64_t prompt = 0, cached = 0;
    int thinking = 0, answer = 0, emitted = 0;
    Clock::time_point start = Clock::now();
    Clock::time_point first_token{};
    double prefill_s = 0;
    bool prompt_done = false, finished = false;
    std::string text, reasoning;

    // Advances the prompt by up to 100 ms of work; true once it is done.
    bool prefill_step() {
        const auto total = std::chrono::duration<double>(prefill_s);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const double done = std::min(1.0, std::chrono::duration<double>(Clock::now() - start) / total);
        std::lock_guard lock(g_state.mutex);
        Active& active   = g_state.active[id];
        active.processed = cached + static_cast<std::uint64_t>(static_cast<double>(prompt - cached) * done);
        if (done >= 1.0) {
            active.phase = "decode";
            prompt_done  = true;
        }
        return prompt_done;
    }

    // The next token (a word of thinking or of the answer); waits while the server is stalled.
    std::string next_token(bool& is_reasoning) {
        for (;;) {
            {
                std::lock_guard lock(g_state.mutex);
                if (!g_state.stalled) { break; }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::this_thread::sleep_for(kTokenEvery);
        if (emitted == 0) { first_token = Clock::now(); }
        is_reasoning = emitted < thinking;
        const int index = is_reasoning ? emitted : emitted - thinking;
        ++emitted;
        std::lock_guard lock(g_state.mutex);
        ++g_state.active[id].output;
        ++g_state.output_tokens;
        // The answer exercises the chat's Markdown: a table, inline and display maths, code.
        static const char* kThoughts[] = {"Let", " me", " think", " about", " this", " carefully", "."};
        static const char* kWords[]    = {"Here", " is", " **a", " table**", ":\n\n", "| Planet | Moons |\n", "|---|---:|\n",
                                          "| Earth | 1 |\n", "| Mars | 2 |\n", "| Jupiter | 95 |\n\n", "Energy", " is",
                                          " $E", " = mc^2$", ", and\n\n", "$$\\int_0^1 x^2\\,dx = \\tfrac{1}{3}$$\n\n",
                                          "```python\n", "def hello():\n", "    print(\"hello", " world\")\n", "```\n\n",
                                          "Use", " `hello()`", ".\n\n"};
        constexpr int kWordCount = sizeof(kWords) / sizeof(kWords[0]);
        return is_reasoning ? kThoughts[index % 7] : kWords[index % kWordCount];
    }

    [[nodiscard]] Json timings() const {
        const double prompt_ms = prefill_s * 1000.0;
        const double gen_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - (emitted > 0 ? first_token : Clock::now())).count();
        const std::uint64_t computed = prompt - cached;
        return {{"cache_n", cached},
                {"prompt_n", computed},
                {"prompt_ms", prompt_ms},
                {"prompt_per_token_ms", computed ? prompt_ms / static_cast<double>(computed) : 0.0},
                {"prompt_per_second", prompt_ms > 0 ? static_cast<double>(computed) * 1000.0 / prompt_ms : 0.0},
                {"predicted_n", emitted},
                {"predicted_ms", gen_ms},
                {"predicted_per_token_ms", emitted > 1 ? gen_ms / (emitted - 1) : 0.0},
                {"predicted_per_second", gen_ms > 0 && emitted > 1 ? (emitted - 1) * 1000.0 / gen_ms : 0.0},
                {"draft_n", emitted * 2},
                {"draft_n_accepted", emitted * 13 / 10}};
    }

    void finish() {
        if (finished) { return; }
        finished = true;
        Sample sample;
        sample.id        = id;
        sample.time      = unix_now();
        sample.protocol  = protocol;
        sample.prompt    = prompt;
        sample.cached    = cached;
        sample.prefill_s = prefill_s;
        sample.ttft_s    = std::chrono::duration<double>(first_token - start).count();
        sample.output    = static_cast<std::uint64_t>(emitted);
        sample.decode_s  = std::max(std::chrono::duration<double>(Clock::now() - first_token).count(), 0.001);
        sample.drafted   = sample.output * 2;
        sample.accepted  = sample.output * 13 / 10;
        std::lock_guard lock(g_state.mutex);
        sample.label = g_state.active[id].label;
        if (g_state.experts) {
            sample.lookups = sample.output * 40 * 8;
            sample.hits    = sample.lookups * (70 + g_state.random() % 20) / 100;
        }
        g_state.active.erase(id);
        ++g_state.completed;
        g_state.prefill_tokens += prompt - cached;
        g_state.samples.push_back(sample);
        while (g_state.samples.size() > 1000) { g_state.samples.pop_front(); }
        log("INFO", "req#" + std::to_string(id) + " done | " + protocol + " | prompt " + std::to_string(prompt) +
                        " | cache " + std::to_string(cached) + " | output " + std::to_string(emitted));
    }
};

// A member of the expected type, else the fallback (clients send null for "the default").
template <typename T>
T member(const Json& object, const char* name, T fallback) {
    if (!object.is_object()) { return fallback; }
    const auto it = object.find(name);
    if (it == object.end()) { return fallback; }
    if constexpr (std::is_same_v<T, bool>) {
        return it->is_boolean() ? it->template get<bool>() : fallback;
    } else {
        return it->is_number() ? it->template get<T>() : fallback;
    }
}

std::shared_ptr<Generation> begin_generation(const std::string& protocol, const Json& body, const std::string& label) {
    auto g       = std::make_shared<Generation>();
    g->protocol  = protocol;
    g->stream    = member(body, "stream", false);
    g->progress  = member(body, "return_progress", false);
    g->per_token = member(body, "timings_per_token", false);
    g->usage     = body.contains("stream_options") && member(body["stream_options"], "include_usage", false);
    const int limit =
        std::clamp(member(body, "max_tokens", member(body, "max_output_tokens", member(body, "max_completion_tokens", 64))), 1, 4096);
    const bool think = body.contains("chat_template_kwargs") ? member(body["chat_template_kwargs"], "enable_thinking", true) : true;
    g->thinking = think ? std::min(24, limit / 3) : 0;
    g->answer   = limit - g->thinking;
    std::lock_guard lock(g_state.mutex);
    g->id     = g_state.next_id++;
    // The prompt: about one token per four characters of the request, plus a few thousand.
    g->prompt = body.dump().size() / 4 + 400 + g_state.random() % 4000;
    g->cached = g->prompt * (g_state.random() % 85) / 100;
    g->prefill_s = std::min(20.0, static_cast<double>(g->prompt - g->cached) / kPrefillRate);
    Active active;
    active.id       = g->id;
    active.label    = label;
    active.protocol = protocol;
    active.prompt   = g->prompt;
    active.cached   = g->cached;
    active.processed = g->cached;
    g_state.active[g->id] = active;
    return g;
}

Json final_answer(const Generation& g) {
    if (g.protocol == "anthropic") {
        return {{"id", "msg_fake"}, {"type", "message"}, {"role", "assistant"}, {"model", g_state.model},
                {"content", Json::array({{{"type", "text"}, {"text", g.text}}})}, {"stop_reason", "end_turn"},
                {"usage", {{"input_tokens", g.prompt}, {"output_tokens", g.emitted}}}};
    }
    if (g.protocol == "responses") {
        return {{"id", "resp_fake"}, {"object", "response"}, {"status", "completed"}, {"model", g_state.model},
                {"output", Json::array({{{"type", "message"}, {"role", "assistant"},
                                         {"content", Json::array({{{"type", "output_text"}, {"text", g.text}}})}}})}};
    }
    return {{"id", "chatcmpl-fake"}, {"object", "chat.completion"}, {"model", g_state.model},
            {"choices", Json::array({{{"index", 0},
                                      {"message", {{"role", "assistant"}, {"content", g.text}, {"reasoning_content", g.reasoning}}},
                                      {"finish_reason", "stop"}}})},
            {"usage", {{"prompt_tokens", g.prompt}, {"completion_tokens", g.emitted},
                       {"prompt_tokens_details", {{"cached_tokens", g.cached}}}}},
            {"timings", g.timings()}};
}

Json chat_chunk(const Json& delta) {
    return {{"object", "chat.completion.chunk"}, {"model", g_state.model},
            {"choices", Json::array({{{"index", 0}, {"delta", delta}}})}};
}

void generate(const std::string& protocol, const Json& body, const std::string& label, httplib::Response& response) {
    hang_point();
    auto g = begin_generation(protocol, body, label);
    if (!g->stream) {
        while (!g->prefill_step()) {}
        for (int i = 0; i < g->thinking + g->answer; ++i) {
            bool reasoning     = false;
            const std::string token = g->next_token(reasoning);
            (reasoning ? g->reasoning : g->text) += token;
        }
        g->finish();
        response.set_content(final_answer(*g).dump(), "application/json");
        return;
    }
    response.set_chunked_content_provider("text/event-stream", [g](size_t, httplib::DataSink& sink) {
        std::string chunk;
        if (!g->prompt_done) {
            const bool done = g->prefill_step();
            if (g->progress && g->protocol == "openai") {
                Json progress = chat_chunk(Json::object());
                std::lock_guard lock(g_state.mutex);
                const Active& active      = g_state.active[g->id];
                progress["prompt_progress"] = {{"total", g->prompt},
                                               {"cache", g->cached},
                                               {"processed", active.processed},
                                               {"time_ms", std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g->start).count()}};
                chunk = sse("", progress);
            }
            if (!done) { return chunk.empty() || sink.write(chunk.data(), chunk.size()); }
            if (g->protocol == "anthropic") {
                chunk += sse("message_start", {{"type", "message_start"}, {"message", {{"id", "msg_fake"}}}});
            }
            if (!chunk.empty() && !sink.write(chunk.data(), chunk.size())) { return false; }
            return true;
        }
        if (g->emitted < g->thinking + g->answer) {
            bool reasoning          = false;
            const std::string token = g->next_token(reasoning);
            (reasoning ? g->reasoning : g->text) += token;
            if (g->protocol == "anthropic") {
                chunk = reasoning ? sse("content_block_delta", {{"type", "content_block_delta"}, {"index", 0},
                                                                {"delta", {{"type", "thinking_delta"}, {"thinking", token}}}})
                                  : sse("content_block_delta", {{"type", "content_block_delta"}, {"index", 1},
                                                                {"delta", {{"type", "text_delta"}, {"text", token}}}});
            } else if (g->protocol == "responses") {
                chunk = reasoning ? sse("response.reasoning_text.delta", {{"type", "response.reasoning_text.delta"}, {"delta", token}})
                                  : sse("response.output_text.delta", {{"type", "response.output_text.delta"}, {"delta", token}});
            } else {
                Json payload = chat_chunk(reasoning ? Json{{"reasoning_content", token}} : Json{{"content", token}});
                if (g->per_token) { payload["timings"] = g->timings(); }
                chunk = sse("", payload);
            }
            return sink.write(chunk.data(), chunk.size());
        }
        g->finish();
        if (g->protocol == "anthropic") {
            chunk = sse("message_delta", {{"type", "message_delta"}, {"delta", {{"stop_reason", "end_turn"}}},
                                          {"usage", {{"output_tokens", g->emitted}}}}) +
                    sse("message_stop", {{"type", "message_stop"}});
        } else if (g->protocol == "responses") {
            chunk = sse("response.completed", {{"type", "response.completed"},
                                               {"response", {{"id", "resp_fake"}, {"status", "completed"}}}});
        } else {
            Json last = chat_chunk(Json::object());
            last["choices"][0]["finish_reason"] = "stop";
            last["timings"]                    = g->timings();
            chunk = sse("", last);
            if (g->usage) {
                Json usage       = {{"object", "chat.completion.chunk"}, {"model", g_state.model}, {"choices", Json::array()}};
                usage["usage"]   = {{"prompt_tokens", g->prompt}, {"completion_tokens", g->emitted},
                                    {"prompt_tokens_details", {{"cached_tokens", g->cached}}}};
                chunk += sse("", usage);
            }
            chunk += "data: [DONE]\n\n";
        }
        sink.write(chunk.data(), chunk.size());
        sink.done();
        return true;
    }, [g](bool) {
        // A client that went away: the request leaves the server, as a cancelled one would.
        if (g->finished) { return; }
        std::lock_guard lock(g_state.mutex);
        g_state.active.erase(g->id);
        log("INFO", "req#" + std::to_string(g->id) + " cancelled | client disconnected");
    });
}

void print_help() {
    std::printf(
        "usage: infernix-serve ARTIFACT [options]   (the launcher's stand-in for tests)\n\n"
        "SERVER\n"
        "  --host ADDR                listen address\n"
        "  --port N                   listen port\n"
        "  --model-id NAME            public model name\n"
        "  --any-model-name           accept any model name\n"
        "  --log-stats-panel on|off   console statistics panel\n"
        "  --log-colours on|off       coloured console\n"
        "  --plan FILE                write the sizing plan to FILE and exit\n"
        "MODEL & CONTEXT\n"
        "  --max-context N            max context tokens per request (default 8192)\n"
        "  --max-concurrency N        max concurrent sequences, 1-8 (default 1)\n"
        "  --kv-dtype T               KV storage\n"
        "  --vram-headroom-mib N|auto VRAM left free\n"
        "  --ram-free-mib N           RAM left free\n"
        "  --expert-ram-mib N|auto    expert RAM\n"
        "  --host-context-mib N|auto  pinned Host pool\n"
        "  --expert-state FILE|off    expert cache state\n"
        "  --vision                   load the vision tower\n"
        "FAKE\n"
        "  --fake-load-ms N           loading time\n"
        "  --fake-fail-load TEXT      fail the load\n"
        "  --fake-hog-mib N           take RAM while loading\n"
        "  --fake-experts             report an expert cache\n");
}

// --plan: a plan that fits up to 225,280 tokens of context (2.03 GB + 33,792 bytes per token
// beside 21.4 GiB of weights on a 32 GiB card), the shape infernix-serve writes.
int write_plan(const std::string& file, std::uint64_t context) {
    constexpr double kGiB          = 1024.0 * 1024.0 * 1024.0;
    const std::uint64_t total      = static_cast<std::uint64_t>(32.0 * kGiB);
    const std::uint64_t free_bytes = static_cast<std::uint64_t>(31.4 * kGiB);
    const std::uint64_t weights    = static_cast<std::uint64_t>(21.4 * kGiB);
    const std::uint64_t headroom   = 1ull << 30;
    const std::uint64_t minimum    = 2030000000ull + context * 33792ull;
    const std::uint64_t planned    = weights + minimum + headroom;
    const bool fits                = planned <= free_bytes;
    const std::uint64_t max_context = (free_bytes - weights - headroom - 2030000000ull) / 33792ull / 4096 * 4096;
    const Json plan = {
        {"schema", 1},
        {"fits", fits},
        {"estimated", true},
        {"architecture", "fake"},
        {"error", fits ? Json(nullptr)
                       : Json("automatic KV capacity requires " + std::to_string(minimum + headroom) + " bytes total (" +
                              std::to_string(minimum) + " bytes minimum Engine runtime reservation + " +
                              std::to_string(headroom) + " bytes automatic headroom), but only " +
                              std::to_string(free_bytes - weights) + " bytes are available after weights")},
        {"gpu", {{"name", "Fake GPU"}, {"total_bytes", total}, {"free_bytes", free_bytes}, {"budget_bytes", nullptr}}},
        {"vram",
         {{"weights_bytes", weights},
          {"kv_bytes", context * 32768ull},
          {"kv_tokens", context},
          {"minimum_reservation_bytes", minimum},
          {"bytes_per_page_group", 8650752},
          {"page_group_tokens", 256},
          {"workspace_bytes", 1500000000ull},
          {"graph_allowance_bytes", 530000000ull},
          {"expert_cache_bytes", nullptr},
          {"expert_frames", nullptr},
          {"headroom_bytes", headroom},
          {"planned_bytes", planned},
          {"left_bytes", fits ? free_bytes - planned : 0}}},
        {"ram",
         {{"total_bytes", static_cast<std::uint64_t>(96.0 * kGiB)},
          {"available_bytes", static_cast<std::uint64_t>(80.0 * kGiB)},
          {"ram_free_bytes", 8192ull << 20},
          {"experts_bytes", nullptr},
          {"pinned_bytes", 4096ull << 20},
          {"host_context_bytes", 4096ull << 20},
          {"ngram_bytes", nullptr},
          {"ngram_residency", nullptr},
          {"tier", nullptr},
          {"planned_bytes", 8192ull << 20},
          {"left_bytes", static_cast<std::uint64_t>(72.0 * kGiB)}}},
        {"max_context", max_context}};
    std::ofstream(file, std::ios::binary | std::ios::trunc) << plan.dump(2) << "\n";
    return fits ? 0 : 3;
}

#ifdef _WIN32
BOOL WINAPI on_console(DWORD event) {
    if (event == CTRL_BREAK_EVENT || event == CTRL_C_EVENT || event == CTRL_CLOSE_EVENT) {
        log("INFO", "stopping | running and queued requests are cancelled");
        log("INFO", "server stopped");
        std::_Exit(0);
    }
    return FALSE;
}
#else
void on_signal(int) {
    log("INFO", "server stopped");
    std::_Exit(0);
}
#endif

} // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1", fail_load, plan_file;
    int port = 8080, load_ms = 1500, hog_mib = 0;
    std::uint64_t context = 8192;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        }
        if (arg == "--host") {
            host = value();
        } else if (arg == "--port") {
            port = std::atoi(value().c_str());
        } else if (arg == "--model-id") {
            g_state.model = value();
        } else if (arg == "--max-context") {
            context = std::strtoull(value().c_str(), nullptr, 10);
        } else if (arg == "--plan") {
            plan_file = value();
        } else if (arg == "--fake-load-ms") {
            load_ms = std::atoi(value().c_str());
        } else if (arg == "--fake-fail-load") {
            fail_load = value();
        } else if (arg == "--fake-hog-mib") {
            hog_mib = std::atoi(value().c_str());
        } else if (arg == "--fake-experts") {
            g_state.experts = true;
        }
    }
    if (!plan_file.empty()) { return write_plan(plan_file, context); }
#ifdef _WIN32
    SetConsoleCtrlHandler(on_console, TRUE);
#else
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);
#endif
    g_state.start_unix = unix_now();
#ifdef _WIN32
    const unsigned long process_id = GetCurrentProcessId();
#else
    const unsigned long process_id = static_cast<unsigned long>(getpid());
#endif
    g_state.instance_id = "fake-" + std::to_string(process_id) + "-" +
                          std::to_string(static_cast<long long>(g_state.start_unix * 1e6));
    log("INFO", "build fake");
    if (const char* probe = std::getenv("INFERNIX_FAKE_PROBE")) {
        // The launcher's checks see a configuration's environment here.
        log("INFO", std::string("environment INFERNIX_FAKE_PROBE=") + probe);
    }
    log("INFO", "starting engine");

    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(8, 64); };
    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        hang_point();
        std::lock_guard lock(g_state.mutex);
        response.status = g_state.ready ? 200 : 503;
        response.set_content(Json{{"status", g_state.ready ? "ok" : "unavailable"}}.dump(), "application/json");
    });
    server.Get("/metrics", [](const httplib::Request&, httplib::Response& response) {
        hang_point();
        response.set_content(metrics_json().dump(), "application/json");
    });
    const auto route = [](const std::string& protocol) {
        return [protocol](const httplib::Request& request, httplib::Response& response) {
            Json body = Json::parse(request.body, nullptr, false);
            if (!body.is_object()) { body = Json::object(); }
            generate(protocol, body, request.get_header_value("X-Infernix-Request-Label"), response);
        };
    };
    server.Post(R"((?:/v1)?/v1/chat/completions)", route("openai"));
    server.Post(R"((?:/v1)?/v1/completions)", route("openai"));
    server.Post(R"((?:/v1)?/v1/responses)", route("responses"));
    server.Post(R"((?:/v1)?/v1/messages)", route("anthropic"));
    server.Post(R"((?:/v1)?/v1/messages/count_tokens)", [](const httplib::Request&, httplib::Response& response) {
        response.set_content(R"({"input_tokens":56})", "application/json");
    });
    // As infernix-serve (docs/serving.md, Resource behavior): compaction is refused, and this
    // stand-in stores no Response, so every id is unknown to cancel.
    server.Post(R"((?:/v1)?/v1/responses/compact)", [](const httplib::Request&, httplib::Response& response) {
        response.status = 400;
        response.set_content(R"({"error":{"message":"Responses compaction is not supported","type":"invalid_request_error",)"
                             R"("param":"context_management","code":"compaction_not_supported"}})",
                             "application/json");
    });
    server.Post(R"((?:/v1)?/v1/responses/([^/]+)/cancel)", [](const httplib::Request& request, httplib::Response& response) {
        response.status = 404;
        response.set_content(Json{{"error",
                                   {{"message", "No response found with id '" + request.matches[1].str() + "'"},
                                    {"type", "invalid_request_error"},
                                    {"param", nullptr},
                                    {"code", "response_not_found"}}}}
                                 .dump(),
                             "application/json");
    });
    server.Post("/fake/crash", [](const httplib::Request& request, httplib::Response&) {
        const int code = request.has_param("code") ? std::atoi(request.get_param_value("code").c_str())
#ifdef _WIN32
                                                   : static_cast<int>(0xC0000005u);
        log("ERROR", "fake crash");
        TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#else
                                                   : 0;
        log("ERROR", "fake crash");
        if (code == 0) { std::raise(SIGSEGV); }
        std::_Exit(code);
#endif
    });
    server.Post("/fake/hang", [](const httplib::Request&, httplib::Response& response) {
        log("WARN", "fake hang: no more answers");
        std::lock_guard lock(g_state.mutex);
        g_state.hung = true;
        response.set_content("{}", "application/json");
    });
    server.Post("/fake/stall", [](const httplib::Request&, httplib::Response& response) {
        log("WARN", "fake stall: requests stop producing tokens");
        std::lock_guard lock(g_state.mutex);
        g_state.stalled = true;
        response.set_content("{}", "application/json");
    });

    // infernix-serve's refusal when its port is held (it binds exclusively and exits before loading).
    const auto port_refused = [&host, port] {
        log("ERROR", "cannot bind " + host + ":" + std::to_string(port) +
                         ": the port is in use (another server, an earlier infernix-serve say, holds it): stop it or "
                         "choose another --port");
        return 1;
    };
    if (const char* once = std::getenv("INFERNIX_FAKE_BIND_FAIL_ONCE"); once && !std::filesystem::exists(once)) {
        // Another program took the port first, this once (the launcher picks another and starts again).
        std::ofstream(once) << "port taken once\n";
        return port_refused();
    }
    if (!server.bind_to_port(host, port)) { return port_refused(); }
    std::thread loader([load_ms, hog_mib, fail_load] {
        std::vector<std::unique_ptr<char[]>> hog;
        log("INFO", "loading weights | 21.4 GiB");
        for (int taken = 0; taken < hog_mib; taken += 256) {
            // RAM taken and touched in 256 MiB steps over about three seconds.
            auto block = std::make_unique<char[]>(256u << 20);
            for (std::size_t at = 0; at < (256u << 20); at += 4096) { block[at] = 1; }
            hog.push_back(std::move(block));
            std::this_thread::sleep_for(std::chrono::milliseconds(3000 / std::max(1, hog_mib / 256)));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(load_ms));
        if (!fail_load.empty()) {
            log("ERROR", "startup failed | finalizing target | 37.5 ms");
            log("FATAL", "server failed during startup | " + fail_load);
            std::_Exit(1);
        }
        hog.clear();
        log("INFO", "engine ready | " + g_state.model + " | total 1.5s");
        std::lock_guard lock(g_state.mutex);
        g_state.ready = true;
    });
    loader.detach();
    server.listen_after_bind();
    return 0;
}
