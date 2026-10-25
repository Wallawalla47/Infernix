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

    // The file may not grow a text field: this key set is the whole row contract.
    std::vector<std::string> keys;
    for (const auto& item : row.items()) { keys.push_back(item.key()); }
    std::sort(keys.begin(), keys.end());
    const std::vector<std::string> expected_keys{
        "decode_ms", "decode_tok_s", "drafts_accepted", "drafts_offered", "duration_s",
        "expert_hits", "expert_lookups", "finish", "hit_rate", "id", "ngram_accepted",
        "ngram_drafts", "output_tokens", "prefill_tok_s", "prompt_ms", "prompt_tokens",
        "protocol", "reused", "time", "ttft_ms"};
    failures += check(keys == expected_keys, "row must carry no prompt or output text");

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
    double decode  = 0.0;
    double prefill = 0.0;
    failures += check(history.rates(decode, prefill) && decode == 12.5 && prefill == 3400.0,
                      "history carries the measured rates");

    RequestHistory bounded;
    for (std::uint64_t i = 0; i < kMetricsRequestLimit + 5; ++i) {
        RequestLogContext each;
        each.id = i;
        bounded.failure(each, false);
    }
    const auto bounded_rows = bounded.rows();
    failures += check(bounded_rows.size() == kMetricsRequestLimit, "history ring is bounded");
    failures += check(bounded_rows.front().id == 5, "history drops the oldest rows first");

    return failures == 0 ? 0 : 1;
}
