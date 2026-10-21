#include "serve/slots.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <set>
#include <utility>

namespace infernix::serve {
namespace {

// Rates are sampled no more often than this: two polls closer together share the last rate.
constexpr std::uint64_t kRateIntervalMs = 250;

std::string iso8601_utc(std::uint64_t unix_ms) {
    const std::time_t seconds = static_cast<std::time_t>(unix_ms / 1000);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    char text[32];
    std::snprintf(text, sizeof(text), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", utc.tm_year + 1900,
                  utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min, utc.tm_sec,
                  static_cast<int>(unix_ms % 1000));
    return text;
}

RequestJson optional_string(const std::string& value) {
    return value.empty() ? RequestJson(nullptr) : RequestJson(value);
}

RequestJson optional_rate(const std::optional<double>& value) {
    return value ? RequestJson(*value) : RequestJson(nullptr);
}

} // namespace

bool valid_client_id(std::string_view value) noexcept {
    if (value.empty() || value.size() > 128) { return false; }
    for (const char c : value) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                        c == '.' || c == '_' || c == ':' || c == '-';
        if (!ok) { return false; }
    }
    return true;
}

const char* slot_phase(infernix::LaneActivity activity) noexcept {
    switch (activity) {
    case infernix::LaneActivity::Idle:
        return "idle";
    case infernix::LaneActivity::Materializing:
    case infernix::LaneActivity::Prefill:
    case infernix::LaneActivity::Replay:
        return "prefill";
    case infernix::LaneActivity::Decode:
    case infernix::LaneActivity::Control:
    case infernix::LaneActivity::Finished:
        return "decode";
    case infernix::LaneActivity::Pausing:
        return "queued";
    }
    return "idle";
}

const char* lane_activity_name(infernix::LaneActivity activity) noexcept {
    switch (activity) {
    case infernix::LaneActivity::Idle:
        return "idle";
    case infernix::LaneActivity::Materializing:
        return "materializing";
    case infernix::LaneActivity::Prefill:
        return "prefill";
    case infernix::LaneActivity::Replay:
        return "replay";
    case infernix::LaneActivity::Decode:
        return "decode";
    case infernix::LaneActivity::Control:
        return "control";
    case infernix::LaneActivity::Pausing:
        return "pausing";
    case infernix::LaneActivity::Finished:
        return "finished";
    }
    return "idle";
}

void SlotRegistry::begin(std::uint64_t engine_request_id, SlotIdentity identity) {
    if (engine_request_id == 0) { return; }
    std::lock_guard lock(mutex_);
    identities_[engine_request_id] = std::move(identity);
}

void SlotRegistry::end(std::uint64_t engine_request_id) {
    if (engine_request_id == 0) { return; }
    std::lock_guard lock(mutex_);
    identities_.erase(engine_request_id);
    rates_.erase(engine_request_id);
}

std::size_t SlotRegistry::live() const {
    std::lock_guard lock(mutex_);
    return identities_.size();
}

RequestJson SlotRegistry::render(const SlotFacts& facts, const infernix::RuntimeStats& stats,
                                 std::uint64_t now_unix_ms) {
    std::lock_guard lock(mutex_);
    RequestJson slots = RequestJson::array();
    std::set<std::uint64_t> resident;
    const std::uint32_t lanes = std::min<std::uint32_t>(stats.lane_count, infernix::kMaximumConcurrency);
    for (std::uint32_t lane = 0; lane < lanes; ++lane) {
        const infernix::LaneStats& view = stats.lanes[lane];
        RequestJson slot;
        slot["id"]            = lane;
        slot["n_ctx"]         = facts.max_context;
        slot["speculative"]   = facts.speculative;
        slot["is_processing"] = false;
        if (view.decision_lane) { slot["decision"] = true; }
        if (view.activity == infernix::LaneActivity::Idle || view.engine_request_id == 0) {
            slot["phase"] = "idle";
            slots.push_back(std::move(slot));
            continue;
        }
        resident.insert(view.engine_request_id);

        // Token rates from the previous sample of this request (any lane: a resumed request may
        // come back on another one).
        RateSample& sample = rates_[view.engine_request_id];
        if (sample.at_unix_ms == 0) {
            sample = RateSample{.at_unix_ms             = now_unix_ms,
                                .generated_tokens       = view.generated_tokens,
                                .computed_prompt_tokens = view.computed_prompt_tokens};
        } else if (now_unix_ms >= sample.at_unix_ms + kRateIntervalMs) {
            const double seconds = static_cast<double>(now_unix_ms - sample.at_unix_ms) / 1000.0;
            const auto rate      = [seconds](std::uint32_t now, std::uint32_t before) {
                return now > before ? static_cast<double>(now - before) / seconds : 0.0;
            };
            sample.decode_tps  = rate(view.generated_tokens, sample.generated_tokens);
            sample.prefill_tps = rate(view.computed_prompt_tokens, sample.computed_prompt_tokens);
            sample.at_unix_ms             = now_unix_ms;
            sample.generated_tokens       = view.generated_tokens;
            sample.computed_prompt_tokens = view.computed_prompt_tokens;
        }

        const char* phase = slot_phase(view.activity);
        slot["is_processing"] = true;
        slot["id_task"]       = view.engine_request_id;
        // llama.cpp semantics: occupancy (prompt plus tokens decoded so far); processed counts only
        // computed (uncached) prompt tokens.
        slot["n_prompt_tokens"] =
            static_cast<std::uint64_t>(view.prompt_tokens) + view.generated_tokens;
        if (view.binding_known) {
            slot["n_prompt_tokens_cache"]     = view.reused_prompt_tokens;
            slot["n_prompt_tokens_processed"] = view.computed_prompt_tokens;
        } else {
            slot["n_prompt_tokens_cache"]     = nullptr;
            slot["n_prompt_tokens_processed"] = nullptr;
        }
        slot["next_token"] = RequestJson::array({RequestJson{{"has_next_token", true},
                                                             {"has_new_line", false},
                                                             {"n_remain", -1},
                                                             {"n_decoded", view.generated_tokens}}});
        slot["phase"]        = phase;
        slot["phase_source"] = "engine";
        slot["activity"]     = lane_activity_name(view.activity);
        slot["is_decision"]  = view.decision;
        slot["decode_tps"]   = std::string_view(phase) == "decode" ? optional_rate(sample.decode_tps)
                                                                   : RequestJson(nullptr);
        slot["prefill_tps"]  = std::string_view(phase) == "prefill" ? optional_rate(sample.prefill_tps)
                                                                    : RequestJson(nullptr);
        const auto identity = identities_.find(view.engine_request_id);
        if (identity != identities_.end()) {
            const SlotIdentity& who = identity->second;
            slot["request_id"]      = optional_string(who.request_id.empty() ? who.upstream_id
                                                                             : who.request_id);
            slot["upstream_id"]     = optional_string(who.upstream_id);
            slot["session_id"]      = optional_string(who.session_id);
            slot["protocol"]        = who.protocol;
            slot["model"]           = who.model;
            slot["stream"]          = who.stream;
            slot["started_at"]      = iso8601_utc(who.started_unix_ms);
            slot["elapsed_ms"] =
                now_unix_ms > who.started_unix_ms ? now_unix_ms - who.started_unix_ms : 0;
        } else {
            // Not (yet) registered by an HTTP handler: the lane is still reported, anonymously.
            slot["request_id"] = nullptr;
            slot["started_at"] = nullptr;
            slot["elapsed_ms"] = nullptr;
        }
        slots.push_back(std::move(slot));
    }
    // A request no longer resident (finished, paused) keeps no stale rate sample.
    std::erase_if(rates_, [&](const auto& entry) { return !resident.contains(entry.first); });
    return slots;
}

} // namespace infernix::serve
