#include "serve/metrics_json.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace infernix::serve;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool near(double actual, double expected, double tolerance = 1e-9) {
    return std::abs(actual - expected) <= tolerance;
}

} // namespace

int main() {
    int failures = 0;

    MetricsJsonSnapshot snapshot;
    snapshot.instance_id         = "instance-1";
    snapshot.model               = "qwen3.8-27b-nvfp4";
    snapshot.architecture        = "Qwen3_5ForCausalLM";
    snapshot.weights             = "nvfp4";
    snapshot.kv                  = "nvfp4";
    snapshot.speculative_backend = "dflash2";
    snapshot.time_unix           = 1000.5;
    snapshot.start_time_unix     = 900.0;
    snapshot.max_context         = 200000;
    snapshot.max_concurrency     = 2;
    snapshot.stats.running_requests           = 1;
    snapshot.stats.prompt_tokens              = 3000;
    snapshot.stats.reused_prompt_tokens       = 2500;
    snapshot.stats.computed_prefill_tokens    = 500;
    snapshot.stats.generated_tokens           = 120;
    snapshot.stats.committed_decode_tokens    = 110;
    snapshot.stats.speculative_draft_tokens   = 80;
    snapshot.stats.speculative_accepted_tokens = 60;
    snapshot.counts.completed                 = 3;
    snapshot.counts.cancelled                 = 1;
    snapshot.counts.rejected                  = 2;

    RequestLogContext context;
    context.id       = 41;
    context.protocol = "openai_chat_completions";
    context.label    = "launcher-7"; // the client's X-Infernix-Request-Label

    GenerationOutcome outcome;
    outcome.prompt_tokens                       = 1200;
    outcome.completion_tokens                   = 65;
    outcome.finish_reason                       = infernix::FinishReason::StopToken;
    outcome.timings.first_token_seconds                = 0.05;
    outcome.timings.prefill_seconds             = 0.09;
    outcome.timings.decode_share_seconds        = 1.60;
    outcome.timings.total_seconds               = 1.75;
    outcome.computed_prefill_tokens     = 200;
    outcome.reused_prompt_tokens     = 1000;
    outcome.speculative.drafted_tokens    = 80;
    outcome.speculative.accepted_tokens = 60;

    snapshot.requests.push_back(
        make_metrics_json_request(41, context, outcome, 1000.123456));

    const std::string body = render_metrics_json(snapshot);
    const Json root        = Json::parse(body);

    failures += check(root["schema"] == "infernix.telemetry/1", "schema id");
    failures += check(root["time"].get<double>() == 1000.5, "top-level time");
    failures += check(root["engine"]["name"] == "infernix", "engine.name");
    failures += check(root["engine"]["model"] == "qwen3.8-27b-nvfp4", "engine.model");
    failures += check(root["engine"]["kv"] == "nvfp4", "engine.kv");
    failures += check(root["engine"]["max_context"] == 200000, "engine.max_context");
    failures += check(root["live"]["state"] == "generating", "live.state");
    failures += check(root["live"]["tok_s"].is_null(), "unmeasured rate is null");
    failures += check(root["totals"]["requests"] == 6, "totals include rejected");
    failures += check(root["totals"]["prompt_tokens"] == 3000, "totals prompt tokens");
    failures += check(root["totals"]["reused"] == 2500, "totals reused tokens");
    failures += check(root["live"]["instance_id"] == "instance-1", "live instance id");

    const Json row = root["requests"][0];
    failures += check(row["id"] == 41, "row id");
    failures += check(row["finish"] == "stop_token", "row finish string");
    failures += check(row["prompt_tokens"] == 1200, "row prompt tokens");
    failures += check(row["reused"] == 1000, "row cached tokens");
    failures += check(row["output_tokens"] == 65, "row output tokens");
    failures += check(row["prompt_ms"] == 90.0, "row prompt milliseconds");
    failures += check(row["decode_ms"] == 1600.0, "row decode milliseconds");
    failures += check(row["ttft_ms"] == 50.0, "row ttft milliseconds");
    failures += check(row["decode_tok_s"] == 40.0, "row decode rate excludes first token");
    failures += check(near(row["hit_rate"].get<double>(), 1000.0 / 1200.0), "row hit rate");
    failures += check(row["duration_s"] == 1.75, "row duration");
    failures += check(row["drafts_offered"] == 80 && row["drafts_accepted"] == 60,
                      "row speculative counts");
    // The launcher matches rows to its clients by the label it sent.
    failures += check(row["label"] == "launcher-7", "row carries the request's label");

    // The file may not grow a text field: this key set is the whole row contract (the label is the
    // client's own header value, validated printable ASCII of at most 128 bytes).
    std::vector<std::string> keys;
    for (const auto& item : row.items()) { keys.push_back(item.key()); }
    std::sort(keys.begin(), keys.end());
    const std::vector<std::string> expected_keys{
        "decode_ms", "decode_tok_s", "drafts_accepted", "drafts_offered", "duration_s",
        "expert_hits", "expert_lookups", "finish", "hit_rate", "id", "label", "ngram_accepted",
        "ngram_drafts", "output_tokens", "prefill_tok_s", "prompt_ms", "prompt_tokens",
        "protocol", "reused", "time", "ttft_ms"};
    failures += check(keys == expected_keys, "row must carry no prompt or output text");
    failures += check(root["live"]["active"].is_array() && root["live"]["active"].empty(),
                      "live.active is an empty array with nothing in flight");

    // The monitor's content sniffer sends any body containing these substrings to its Prometheus
    // parser. The JSON representation must never contain them.
    failures += check(body.find("infernix_") == std::string::npos,
                      "body must not trip the Prometheus content sniffer");
    failures += check(body.find("ninfer_") == std::string::npos,
                      "body must not trip the legacy content sniffer");

    failures += check(select_metrics_format("", "") == MetricsFormat::Prometheus,
                      "no format keeps Prometheus");
    failures += check(select_metrics_format("json", "") == MetricsFormat::Json, "format=json");
    failures += check(select_metrics_format("", "application/json") == MetricsFormat::Json,
                      "Accept: application/json selects JSON");
    failures += check(select_metrics_format("", "text/plain;version=0.0.4") ==
                          MetricsFormat::Prometheus,
                      "Prometheus Accept keeps text");
    failures += check(select_metrics_format("prometheus", "application/json") ==
                          MetricsFormat::Prometheus,
                      "an explicit format beats the Accept header");
    failures += check(select_metrics_format("yaml", "") == MetricsFormat::Invalid,
                      "unknown format is invalid");

    RequestHistory history;
    history.done(context, outcome);
    history.failure(context, true);
    history.failure(context, false);
    history.set_rates(12.5, 3400.0);
    const auto rows = history.rows();
    failures += check(rows.size() == 3, "history keeps terminal rows");
    failures += check(rows[0].finish == "stop_token" && rows[1].finish == "cancelled" &&
                          rows[2].finish == "failed",
                      "history classifies terminal outcomes");
    failures += check(rows[0].label == "launcher-7" && rows[1].label == "launcher-7",
                      "completed and failed rows keep the label");
    double decode  = 0.0;
    double prefill = 0.0;
    failures += check(history.rates(decode, prefill) && decode == 12.5 && prefill == 3400.0,
                      "history carries the measured rates");

    // session_stats: the console panel's aggregates as numbers (what the launcher's dashboard
    // shows), formed from summed tokens and seconds exactly as the panel forms them.
    {
        GenerationOutcome first = outcome;
        first.timings.decode_seconds      = 3.2; // two requests shared the rounds: batch 2
        first.speculative.backend         = infernix::SpeculativeBackend::DFlash2;
        first.speculative.rounds          = 20;
        infernix::ExpertCacheStats experts;
        experts.routed     = 1000;
        experts.hits       = 750;
        first.expert_cache = experts;
        GenerationOutcome second       = first;
        second.prompt_tokens           = 800;
        second.reused_prompt_tokens    = 0;
        second.timings.prefill_seconds = 0.31; // the other request's 800 computed tokens

        ConsoleStatsPanel stats(nullptr, false);
        stats.request_done(first);
        stats.request_done(second);
        MetricsJsonSnapshot with_stats = snapshot;
        with_stats.session_stats       = stats.snapshot();
        const Json session_root        = Json::parse(render_metrics_json(with_stats));
        failures += check(!root.contains("session_stats"), "session_stats only when supplied");
        const Json& block   = session_root["session_stats"];
        const Json& session = block["session"];
        failures += check(block["drafter"] == "dflash2" && block["completed"] == 2 &&
                              block["recent_window"] == 10 && block["ngram_enabled"] == false,
                          "session_stats header");
        failures += check(session["requests"] == 2 && near(session["ttft_s"].get<double>(), 0.05),
                          "session TTFT is the mean");
        failures += check(near(session["cache_hit_rate"].get<double>(), 1000.0 / 2000.0),
                          "cache hit rate over summed prompt tokens");
        failures += check(near(session["prefill_tok_s"].get<double>(), (200.0 + 800.0) / (0.09 + 0.31)),
                          "prefill rate over summed computed tokens and seconds");
        failures += check(near(session["decode_tok_s"].get<double>(), 128.0 / 3.2),
                          "decode rate excludes each first token");
        failures += check(near(session["batch"].get<double>(), 2.0), "time-weighted batch");
        failures += check(near(session["acceptance"].get<double>(), 0.75) &&
                              near(session["accepted_per_round"].get<double>(), 3.0),
                          "drafter acceptance and accepted per round");
        failures += check(near(session["expert_hit_rate"].get<double>(), 0.75),
                          "expert VRAM cache hit rate");
        failures += check(session["ngram_acceptance"].is_null(), "no n-gram drafts: null, not zero");
        failures += check(block["recent"]["requests"] == 2, "recent row covers the last requests");
    }

    RequestHistory bounded;
    for (std::uint64_t i = 0; i < kMetricsRequestLimit + 5; ++i) {
        RequestLogContext each;
        each.id = i;
        bounded.failure(each, false);
    }
    const auto bounded_rows = bounded.rows();
    failures += check(bounded_rows.size() == kMetricsRequestLimit, "history ring is bounded");
    failures += check(bounded_rows.front().id == 5, "history drops the oldest rows first");
    {
        MetricsJsonSnapshot unlabelled = snapshot;
        unlabelled.requests            = bounded_rows;
        const Json unlabelled_root     = Json::parse(render_metrics_json(unlabelled));
        failures += check(unlabelled_root["requests"][0]["label"].is_null(),
                          "a request without a label renders null");
    }

    // live.active: the requests in flight, fed by the Engine's progress observations, from queued
    // through prefill to decode, and gone once they end.
    {
        using Clock = LiveRequests::Clock;
        const Clock::time_point t0 = Clock::now();
        LiveRequests live;
        live.begin(12, "launcher-3", "anthropic_messages", 50, t0);
        live.begin(9, "launcher-7", "openai_chat_completions", 1200, t0);
        live.begin(15, "", "decide", 30, Clock::time_point{}); // no start time: now

        auto active = live.rows(t0 + std::chrono::milliseconds(250));
        failures += check(active.size() == 3 && active[0].id == 9 && active[1].id == 12 && active[2].id == 15,
                          "live rows are listed oldest first");
        failures += check(active[0].phase == ActiveRequestPhase::Queued && active[0].prompt_tokens == 1200 &&
                              active[0].cached_tokens == 0 && active[0].processed_tokens == 0 &&
                              active[0].output_tokens == 0 && near(active[0].elapsed_ms, 250.0, 1e-6),
                          "a queued request shows its prepared prompt and its wait");
        failures += check(active[2].elapsed_ms >= 0.0, "an unknown start time counts from registration");

        // Admitted with 1,000 cached tokens, then a prefill unit, then decoding.
        live.progress(9, {.prompt_tokens = 1200, .reused_prompt_tokens = 1000, .processed_prompt_tokens = 1000});
        active = live.rows(t0);
        failures += check(active[0].phase == ActiveRequestPhase::Prefill && active[0].cached_tokens == 1000 &&
                              active[0].processed_tokens == 1000 && active[0].output_tokens == 0,
                          "an admitted request is in prefill with its cached tokens processed");
        live.progress(9, {.prompt_tokens           = 1200,
                          .reused_prompt_tokens    = 1000,
                          .processed_prompt_tokens = 1100,
                          .generated_tokens        = 0});
        failures += check(live.rows(t0)[0].processed_tokens == 1100, "prefill progress advances");
        live.progress(9, {.prompt_tokens           = 1200,
                          .reused_prompt_tokens    = 1000,
                          .processed_prompt_tokens = 1100,
                          .generated_tokens        = 3});
        active = live.rows(t0);
        failures += check(active[0].phase == ActiveRequestPhase::Decode && active[0].processed_tokens == 1200 &&
                              active[0].output_tokens == 3,
                          "a generating request's prompt is done");

        MetricsJsonSnapshot with_live = snapshot;
        with_live.active              = active;
        const Json live_root          = Json::parse(render_metrics_json(with_live));
        const Json& first             = live_root["live"]["active"][0];
        failures += check(first["id"] == 9 && first["label"] == "launcher-7" &&
                              first["protocol"] == "openai_chat_completions" && first["phase"] == "decode" &&
                              first["prompt_tokens"] == 1200 && first["cached_tokens"] == 1000 &&
                              first["processed_tokens"] == 1200 && first["output_tokens"] == 3 &&
                              first["elapsed_ms"].is_number(),
                          "live.active row fields");
        failures += check(live_root["live"]["active"][1]["phase"] == "queued" &&
                              live_root["live"]["active"][2]["label"].is_null(),
                          "phase names and an absent label");
        std::vector<std::string> active_keys;
        for (const auto& item : first.items()) { active_keys.push_back(item.key()); }
        std::sort(active_keys.begin(), active_keys.end());
        const std::vector<std::string> expected_active_keys{
            "cached_tokens", "elapsed_ms",    "id",           "label",        "output_tokens",
            "phase",         "processed_tokens", "prompt_tokens", "protocol"};
        failures += check(active_keys == expected_active_keys, "live.active rows carry no request text");

        live.end(9);
        live.progress(9, {.prompt_tokens = 1200, .generated_tokens = 4}); // a late observation
        active = live.rows(t0);
        failures += check(active.size() == 2 && active[0].id == 12, "an ended request leaves live.active");
        live.end(9); // ending twice is harmless
        live.end(12);
        live.end(15);
        failures += check(live.rows(t0).empty(), "every request ended");
    }

    return failures == 0 ? 0 : 1;
}
