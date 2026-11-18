// The launcher's proxy activity (Launcher members, launcher.h): one record per public request, the
// server's progress and usage matched to them, and the list the web UI shows.

#include "launcher/launcher.h"

#include "launcher/route_support.h"

#include <algorithm>
#include <set>

namespace infernix::launcher {
namespace {

constexpr std::size_t kActivityLimit = 100;

} // namespace

std::uint64_t Launcher::activity_begin(const Request& request, const ApiCall& call) {
    std::uint64_t id = 0;
    {
        std::lock_guard lock(activity_mutex_);
        ProxyRecord record;
        record.id        = next_activity_++;
        record.t         = now_ms();
        record.method    = request.method;
        record.path      = request.path;
        record.model     = call.model;
        record.model_key = call.key;
        record.stream    = call.stream;
        record.client    = request.remote_addr;
        record.key_name  = call.key_name;
        record.routed_by = call.routed_by;
        if (request.has_header("X-Infernix-Request-Label")) {
            record.label = request.get_header_value("X-Infernix-Request-Label");
        }
        id = record.id;
        activity_.push_back(std::move(record));
        while (activity_.size() > kActivityLimit) { activity_.pop_front(); }
    }
    publish("activity", activity_json());
    return id;
}

void Launcher::activity_update(std::uint64_t id, const std::string& state, int status,
                               const std::optional<std::string>& error) {
    {
        std::lock_guard lock(activity_mutex_);
        for (ProxyRecord& record : activity_) {
            if (record.id != id) { continue; }
            record.state  = state;
            record.status = status;
            if (error) { record.error = error; }
            if (state == "done" || state == "failed") {
                record.duration_ms = now_ms() - record.t;
                record.progress    = nullptr;
            }
        }
    }
    publish("activity", activity_json());
}

void Launcher::absorb_metrics(const Json& metrics) {
    if (!metrics.is_object()) { return; }
    // The launcher's record of a server row by its label: launcher-<n>, or a client's own label
    // (the newest record that sent it). The activity lock held.
    const auto activity_of = [this](const Json& label) -> std::optional<std::uint64_t> {
        if (!label.is_string()) { return std::nullopt; }
        const std::string& text = label.get_ref<const std::string&>();
        if (text.starts_with("launcher-")) {
            try {
                return std::stoull(text.substr(9));
            } catch (const std::exception&) { return std::nullopt; }
        }
        for (auto it = activity_.rbegin(); it != activity_.rend(); ++it) {
            if (it->label == text) { return it->id; }
        }
        return std::nullopt;
    };
    bool changed = false;
    // Running requests' progress.
    const Json live = metrics.value("live", Json::object());
    {
        std::lock_guard lock(activity_mutex_);
        std::set<std::uint64_t> active;
        for (const Json& item : live.value("active", Json::array())) {
            const std::optional<std::uint64_t> id = activity_of(item.value("label", Json(nullptr)));
            if (!id) { continue; }
            active.insert(*id);
            for (ProxyRecord& record : activity_) {
                if (record.id != *id || (record.state != "running" && record.state != "waiting")) { continue; }
                const Json progress = {{"phase", item.value("phase", "queued")},
                                       {"prompt_tokens", item.value("prompt_tokens", 0)},
                                       {"cached_tokens", item.value("cached_tokens", 0)},
                                       {"processed_tokens", item.value("processed_tokens", 0)},
                                       {"output_tokens", item.value("output_tokens", 0)},
                                       {"elapsed_ms", item.value("elapsed_ms", 0)}};
                if (record.progress != progress) {
                    record.progress = progress;
                    changed         = true;
                }
            }
        }
        for (ProxyRecord& record : activity_) {
            if (!record.progress.is_null() && !active.contains(record.id)) {
                record.progress = nullptr;
                changed         = true;
            }
        }
    }
    // Finished requests' usage, once each (the server keeps its last rows; each poll repeats them).
    const InstanceSnapshot snapshot = supervisor_->snapshot();
    // Row ids restart with each server; an older server without instance_id is told apart by the
    // launcher's own instance number.
    std::string instance = live.value("instance_id", "");
    if (instance.empty()) { instance = "launcher-" + std::to_string(snapshot.instance); }
    const std::string serving = snapshot.model_key;
    std::vector<UsageSample> samples;
    {
        std::lock_guard lock(activity_mutex_);
        for (const Json& row : metrics.value("requests", Json::array())) {
            const std::string seen = instance + ":" + std::to_string(row.value("id", std::uint64_t{0}));
            if (!counted_rows_.insert(seen).second) { continue; }
            counted_order_.push_back(seen);
            UsageSample sample;
            sample.t = static_cast<std::int64_t>(row.value("time", 0.0) * 1000.0);
            if (sample.t == 0) { sample.t = now_ms(); }
            const std::string finish = row.value("finish", "");
            sample.failed            = finish == "failed" || finish == "cancelled";
            sample.prompt_tokens     = row.value("prompt_tokens", std::uint64_t{0});
            sample.cached_tokens     = row.value("reused", std::uint64_t{0});
            sample.prefill_tokens    = sample.prompt_tokens - std::min(sample.cached_tokens, sample.prompt_tokens);
            sample.output_tokens     = row.value("output_tokens", std::uint64_t{0});
            sample.prefill_s         = row.value("prompt_ms", 0.0) / 1000.0;
            sample.decode_s          = row.value("decode_ms", 0.0) / 1000.0;
            sample.config_id         = serving;
            sample.client            = "unknown";
            const Json label         = row.value("label", Json(nullptr));
            if (const std::optional<std::uint64_t> id = activity_of(label)) {
                for (ProxyRecord& record : activity_) {
                    if (record.id != *id) { continue; }
                    if (record.model_key) { sample.config_id = *record.model_key; }
                    sample.client = record.key_name ? *record.key_name : record.client;
                }
            }
            samples.push_back(std::move(sample));
        }
        // Bounded: the server repeats at most its last 128 rows, so the oldest marks can go.
        while (counted_order_.size() > 4096) {
            counted_rows_.erase(counted_order_.front());
            counted_order_.pop_front();
        }
    }
    if (!samples.empty()) {
        std::lock_guard lock(library_mutex_);
        for (UsageSample& sample : samples) {
            if (const std::optional<ConfigRef> ref = find_config_locked(sample.config_id)) {
                sample.model = effective_alias(ref->id, *ref->model, library_);
            } else {
                sample.model = sample.config_id;
            }
        }
    }
    for (const UsageSample& sample : samples) { usage_.add(sample); }
    if (changed) { publish("activity", activity_json()); }
}

Json Launcher::activity_json() const {
    Json recent = Json::array();
    {
        // The newest 50, oldest first.
        std::lock_guard lock(activity_mutex_);
        const std::size_t skip = activity_.size() > 50 ? activity_.size() - 50 : 0;
        for (auto it = activity_.begin() + static_cast<std::ptrdiff_t>(skip); it != activity_.end(); ++it) {
            const ProxyRecord& r = *it;
            recent.push_back({{"id", r.id},
                              {"t", r.t},
                              {"method", r.method},
                              {"path", r.path},
                              {"model", r.model ? Json(*r.model) : Json(nullptr)},
                              {"model_key", r.model_key ? Json(*r.model_key) : Json(nullptr)},
                              {"stream", r.stream},
                              {"status", r.status},
                              {"state", r.state},
                              {"duration_ms", r.duration_ms ? Json(*r.duration_ms) : Json(nullptr)},
                              {"error", r.error ? Json(*r.error) : Json(nullptr)},
                              {"routed_by", r.routed_by ? Json(*r.routed_by) : Json(nullptr)},
                              {"client", r.client},
                              {"key_name", r.key_name ? Json(*r.key_name) : Json(nullptr)},
                              {"progress", r.progress}});
        }
    }
    return {{"in_flight", router_ ? router_->in_flight() : 0}, {"waiting", router_ ? router_->waiting() : 0},
            {"recent", recent}};
}

} // namespace infernix::launcher
