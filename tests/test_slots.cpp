#include "serve/slots.h"

#include <iostream>
#include <string>

namespace {

using namespace infernix::serve;
using infernix::LaneActivity;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

constexpr std::uint64_t kNow = 1792000000000ULL; // 2026-10-14T17:46:40Z

// Three general lanes and one decision lane: an agent decoding on lane 0, an agent prefilling on
// lane 1 (cache hit of 30,000 of 40,000 tokens, 4,000 computed so far), lane 2 idle, a decision
// prefilling on the decision lane.
infernix::RuntimeStats four_lanes() {
    infernix::RuntimeStats stats;
    stats.lane_count = 4;
    stats.lanes[0]   = {.engine_request_id      = 11,
                        .activity               = LaneActivity::Decode,
                        .prompt_tokens          = 12000,
                        .reused_prompt_tokens   = 11500,
                        .binding_known          = true,
                        .computed_prompt_tokens = 500,
                        .generated_tokens       = 300};
    stats.lanes[1]   = {.engine_request_id      = 12,
                        .activity               = LaneActivity::Prefill,
                        .prompt_tokens          = 40000,
                        .reused_prompt_tokens   = 30000,
                        .binding_known          = true,
                        .computed_prompt_tokens = 4000};
    stats.lanes[3]   = {.engine_request_id = 13,
                        .activity          = LaneActivity::Materializing,
                        .decision_lane     = true,
                        .decision          = true,
                        .prompt_tokens     = 350};
    return stats;
}

int test_identifiers() {
    int failures = 0;
    for (const char* ok : {"a", "np-0199a3f2.x:y_z", "REQ-1"}) {
        failures += check(valid_client_id(ok), "a valid client id was refused");
    }
    for (const char* bad : {"", "has space", "slash/no", "quote\"", "\xc3\xa9"}) {
        failures += check(!valid_client_id(bad), "an invalid client id was kept");
    }
    failures += check(!valid_client_id(std::string(129, 'a')) && valid_client_id(std::string(128, 'a')),
                      "the 128-character limit is wrong");
    failures += check(std::string(slot_phase(LaneActivity::Replay)) == "prefill" &&
                          std::string(slot_phase(LaneActivity::Materializing)) == "prefill" &&
                          std::string(slot_phase(LaneActivity::Control)) == "decode" &&
                          std::string(slot_phase(LaneActivity::Pausing)) == "queued" &&
                          std::string(slot_phase(LaneActivity::Idle)) == "idle",
                      "lane activities map to the wrong slot phases");
    return failures;
}

int test_lanes() {
    int failures = 0;
    SlotRegistry registry;
    registry.begin(11, SlotIdentity{.request_id      = "gw-1",
                                    .upstream_id     = "chatcmpl-1",
                                    .session_id      = "netpi-7",
                                    .protocol        = "openai_chat",
                                    .model           = "qwen3.8-27b",
                                    .stream          = true,
                                    .started_unix_ms = kNow - 5000});
    registry.begin(12, SlotIdentity{.upstream_id = "resp_2", .protocol = "openai_responses",
                                    .started_unix_ms = kNow - 1000});
    const auto slots = registry.render({.max_context = 262144, .speculative = true}, four_lanes(), kNow);

    failures += check(slots.is_array() && slots.size() == 4, "not one slot per lane");
    const auto& decode = slots[0];
    failures += check(decode["id"] == 0 && decode["is_processing"] == true && decode["phase"] == "decode" &&
                          decode["phase_source"] == "engine" && decode["id_task"] == 11,
                      "the decoding lane lost its phase or task");
    // llama.cpp occupancy: prompt plus decoded; processed counts only computed prompt tokens.
    failures += check(decode["n_prompt_tokens"] == 12300 && decode["n_prompt_tokens_cache"] == 11500 &&
                          decode["n_prompt_tokens_processed"] == 500 &&
                          decode["next_token"][0]["n_decoded"] == 300,
                      "the decoding lane's counters are not llama.cpp's");
    failures += check(decode["request_id"] == "gw-1" && decode["upstream_id"] == "chatcmpl-1" &&
                          decode["session_id"] == "netpi-7" && decode["protocol"] == "openai_chat" &&
                          decode["stream"] == true && decode["elapsed_ms"] == 5000 &&
                          decode["started_at"] == "2026-10-14T17:46:35.000Z",
                      "the client's identity or the elapsed time was lost");
    failures += check(decode["n_ctx"] == 262144 && decode["speculative"] == true && !decode.contains("decision"),
                      "a general lane lost its facts or was marked a decision lane");
    failures += check(decode["decode_tps"].is_null(), "a first sample invented a rate");

    const auto& prefill = slots[1];
    failures += check(prefill["phase"] == "prefill" && prefill["n_prompt_tokens"] == 40000 &&
                          prefill["n_prompt_tokens_cache"] == 30000 &&
                          prefill["n_prompt_tokens_processed"] == 4000,
                      "the prefilling lane's progress is wrong");
    failures += check(prefill["request_id"] == "resp_2" && prefill["session_id"].is_null(),
                      "without X-Request-Id the request id is not the server's own id");

    failures += check(slots[2]["is_processing"] == false && slots[2]["phase"] == "idle" &&
                          !slots[2].contains("id_task"),
                      "an idle lane reported a request");

    const auto& decision = slots[3];
    failures += check(decision["decision"] == true && decision["is_decision"] == true &&
                          decision["phase"] == "prefill" && decision["activity"] == "materializing",
                      "the decision lane is not marked or not binding");
    failures += check(decision["n_prompt_tokens_cache"].is_null() &&
                          decision["n_prompt_tokens_processed"].is_null() &&
                          decision["request_id"].is_null() && decision["elapsed_ms"].is_null(),
                      "a lane with no binding yet or no identity invented counters");
    return failures;
}

int test_rates_and_release() {
    int failures = 0;
    SlotRegistry registry;
    registry.begin(11, SlotIdentity{.request_id = "gw-1", .started_unix_ms = kNow - 5000});
    registry.begin(12, SlotIdentity{.request_id = "gw-2", .started_unix_ms = kNow - 1000});
    auto stats = four_lanes();
    (void)registry.render({}, stats, kNow);

    // 2 s later: 160 more tokens out on lane 0, 9,000 more prompt tokens on lane 1.
    stats.lanes[0].generated_tokens       = 460;
    stats.lanes[1].computed_prompt_tokens = 13000;
    auto slots = registry.render({}, stats, kNow + 2000);
    failures += check(slots[0]["decode_tps"] == 80.0 && slots[0]["prefill_tps"].is_null(),
                      "the decode rate is not tokens out per second since the last sample");
    failures += check(slots[1]["prefill_tps"] == 4500.0 && slots[1]["decode_tps"].is_null(),
                      "the prefill rate is not computed prompt tokens per second");

    // A poll 100 ms later keeps the last rate instead of dividing by a sliver.
    stats.lanes[0].generated_tokens = 470;
    slots = registry.render({}, stats, kNow + 2100);
    failures += check(slots[0]["decode_tps"] == 80.0, "a close second poll changed the rate");

    // The request finished: its identity and rate are released, the lane reads idle.
    registry.end(11);
    stats.lanes[0] = infernix::LaneStats{};
    slots          = registry.render({}, stats, kNow + 3000);
    failures += check(slots[0]["phase"] == "idle" && registry.live() == 1,
                      "a finished request was not released");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_identifiers();
    failures += test_lanes();
    failures += test_rates_and_release();
    if (failures != 0) {
        std::cerr << failures << " slots check(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
