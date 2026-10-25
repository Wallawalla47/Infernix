#include "serve/metrics.h"

#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace infernix;
using namespace infernix::serve;

int main() {
    int failures     = 0;
    const auto check = [&](bool pass, const char* message) {
        if (!pass) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    EngineOptions options;
    options.max_concurrency                  = 2;
    options.context_cache.hybrid.device_snapshot_slots = 1;
    MemorySummary memory;
    memory.kv_capacity_page_groups = 256;
    RuntimeStats baseline;
    baseline.generated_tokens         = 4;
    baseline.speculative_draft_tokens = 3;
    baseline.expert_cache             = ExpertCacheTotals{.frames     = 100,
                                                          .routed     = 1000,
                                                          .hits       = 900,
                                                          .promotions = 50,
                                                          .landed     = 5,
                                                          .cpu_served = 60};
    Metrics metrics;
    metrics.configure("custom\"model\\name\nline", options, memory, baseline);
    RuntimeStats running = baseline;
    running.generated_tokens += 5;
    running.speculative_draft_tokens += 6;
    running.speculative_accepted_tokens = 3;
    running.running_requests            = 1;
    running.prefix_selections           = 2;
    running.expert_cache                = ExpertCacheTotals{.frames     = 120,
                                                            .routed     = 1480,
                                                            .hits       = 1332,
                                                            .promotions = 80,
                                                            .landed     = 9,
                                                            .cpu_served = 100};
    metrics.first_token({.prepare_seconds = 0.05, .elapsed_since_submit_seconds = 0.2});
    const auto live = metrics.render(running, true);
    check(live.find("infernix_generation_tokens_total 5\n") != std::string::npos,
          "live generated count must exclude startup warmup");
    check(live.find("infernix_spec_decode_draft_tokens_total 6\n") != std::string::npos,
          "speculative work must be observable before request completion");
    check(live.find("infernix_expert_cache_frames 120\n") != std::string::npos &&
              live.find("infernix_expert_cache_routed_total 480\n") != std::string::npos &&
              live.find("infernix_expert_cache_hits_total 432\n") != std::string::npos &&
              live.find("infernix_expert_cache_promotions_total 30\n") != std::string::npos &&
              live.find("infernix_expert_cache_landed_total 4\n") != std::string::npos &&
              live.find("infernix_expert_cache_cpu_served_total 40\n") != std::string::npos,
          "expert-cache counters start after warmup and the frames gauge is absolute");
    {
        RuntimeStats dense = running;
        dense.expert_cache.reset();
        check(metrics.render(dense, true).find("infernix_expert_cache_") == std::string::npos,
              "no expert-cache metrics without an expert cache");
    }
    {
        // Warm-up executions (the baseline) are excluded from counts and totals; maxima are
        // absolute. The durations are binary fractions of a second, which the 17-digit output
        // prints exactly.
        RuntimeStats warm                 = baseline;
        warm.execution.decode             = {.count = 3, .total_ns = 375'000'000, .max_ns = 0};
        warm.execution.decode_by_batch[0] = warm.execution.decode;
        Metrics timed;
        timed.configure("m", options, memory, warm);
        RuntimeStats now = warm;
        now.execution.decode.add(250'000'000);
        now.execution.decode_by_batch[1].add(250'000'000);
        now.execution.prefill.add(500'000'000);
        const auto text = timed.render(now, true);
        check(text.find("infernix_executions_total{kind=\"decode\"} 1\n") != std::string::npos &&
                  text.find("infernix_execution_seconds_total{kind=\"decode\"} 0.25\n") !=
                      std::string::npos &&
                  text.find("infernix_execution_max_seconds{kind=\"prefill\"} 0.5\n") !=
                      std::string::npos &&
                  text.find("infernix_decode_batch_rounds_total{batch=\"1\"} 0\n") !=
                      std::string::npos &&
                  text.find("infernix_decode_batch_rounds_total{batch=\"2\"} 1\n") !=
                      std::string::npos &&
                  text.find("infernix_decode_batch_max_seconds{batch=\"2\"} 0.25\n") !=
                      std::string::npos &&
                  text.find("batch=\"3\"") == std::string::npos,
              "execution units by kind and decode rounds by batch size, the warm-up excluded");
    }
    check(live.find("infernix_time_to_first_token_seconds_count 1\n") != std::string::npos &&
              live.find("infernix_requests_total{outcome=\"completed\"} 0\n") != std::string::npos,
          "TTFT must be visible before completion");
    check(live.find("_bucket{le=\"0.25\"} 1\n") != std::string::npos &&
              live.find("infernix_time_to_first_token_seconds_sum 0.25\n") != std::string::npos,
          "TTFT must include preparation and use inclusive histogram buckets");
    check(live.find("model_name=\"custom\\\"model\\\\name\\nline\"") != std::string::npos,
          "model label must escape quotes, backslashes and newlines");
    check(live.find("kv=\"bf16\"") != std::string::npos,
          "model_info must identify the KV cache format");
    check(metrics.render(running, true) == live, "scrapes must not consume or reset counters");
    check(live.find("infernix_prefix_selections_total 2\n") != std::string::npos &&
              live.find("infernix_device_state_capacity_slots 3\n") != std::string::npos,
          "prefix-cache selections and StateImage capacity must be observable");

    // Host-tier occupancy is absolute; its eviction and copy counters start after warmup.
    RuntimeStats tiered                    = running;
    tiered.hybrid_tree_blocks              = 40;
    tiered.hybrid_snapshots                = 3;
    tiered.hybrid_host_used_bytes          = 3ULL << 30;
    tiered.hybrid_host_capacity_bytes      = 8ULL << 30;
    tiered.hybrid_host_snapshot_evictions  = 2;
    tiered.hybrid_host_restore_bytes       = 1ULL << 20;
    const auto host = metrics.render(tiered, true);
    check(host.find("infernix_prefix_cache_blocks 40\n") != std::string::npos &&
              host.find("infernix_prefix_cache_snapshots 3\n") != std::string::npos &&
              host.find("infernix_host_context_used_bytes 3221225472\n") != std::string::npos &&
              host.find("infernix_host_context_capacity_bytes 8589934592\n") != std::string::npos,
          "prefix-cache and Host-tier occupancy must be observable");
    check(host.find("infernix_host_context_snapshot_evictions_total 2\n") != std::string::npos &&
              host.find("infernix_host_context_restore_bytes_total 1048576\n") != std::string::npos &&
              host.find("infernix_host_context_write_bytes_total 0\n") != std::string::npos,
          "Host-tier evictions and copy volume must be counted");

    GenerationOutcome outcome;
    outcome.finish_reason                            = FinishReason::OutputLimit;
    outcome.timings.total_seconds                    = 2.0;
    outcome.engine_timing.queue_wait_seconds         = 0.1;
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([&] {
            for (int n = 0; n < 100; ++n) { metrics.done(outcome); }
        });
    }
    for (auto& writer : writers) { writer.join(); }
    outcome.finish_reason = FinishReason::Cancelled;
    metrics.done(outcome);
    metrics.failed(true);
    metrics.failed(false);
    metrics.rejected();
    const auto final = metrics.render(running, false);
    check(final.find("infernix_engine_ready 0\n") != std::string::npos,
          "unavailable engine must remain observable");
    check(final.find("infernix_requests_total{outcome=\"completed\"} 400\n") != std::string::npos &&
              final.find("infernix_requests_total{outcome=\"cancelled\"} 2\n") != std::string::npos &&
              final.find("infernix_requests_total{outcome=\"failed\"} 1\n") != std::string::npos &&
              final.find("infernix_requests_total{outcome=\"rejected\"} 1\n") != std::string::npos,
          "concurrent settlements and terminal classifications must be preserved");
    check(final.find("infernix_request_duration_seconds_bucket{le=\"+Inf\"} 401\n") !=
                  std::string::npos &&
              final.find("infernix_request_duration_seconds_count 401\n") != std::string::npos &&
              final.find("infernix_request_duration_seconds_sum 802\n") != std::string::npos,
          "histogram counts and sums must include cancelled outcomes without fabricating failure "
          "durations");
    check(final.find("infernix_time_to_first_token_seconds_count 1\n") != std::string::npos,
          "request settlement must not count the first token again");
    const Metrics::RequestCounts counts = metrics.counts();
    check(counts.completed == 400 && counts.cancelled == 2 && counts.failed == 1 &&
              counts.rejected == 1,
          "terminal counts must be readable without a scrape");
    outcome.constraint = ConstraintObservation{.complete          = true,
                                               .terminated        = false,
                                               .cache             = ConstraintCacheAccess::Built,
                                               .mask_positions    = 5,
                                               .mask_upload_bytes = 128};
    metrics.done(outcome);
    const auto constrained = metrics.render(running, false);
    check(constrained.find(
              "infernix_constraint_requests_total{outcome=\"complete_interrupted\"} 1\n") !=
                  std::string::npos &&
              constrained.find("infernix_constraint_mask_positions_total 5\n") != std::string::npos &&
              constrained.find("infernix_constraint_mask_upload_bytes_total 128\n") !=
                  std::string::npos,
          "interrupted complete constraint lost its state or actual work");
    return failures == 0 ? 0 : 1;
}
