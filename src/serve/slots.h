#pragma once

// GET /slots: one llama.cpp slot object per resident lane, in the shape llama.cpp's server and
// NInfer report, so supervisors and dashboards written for them read Infernix unchanged.
// The lane facts come from the Engine's published lane view (RuntimeStats::lanes); the request
// identity (the client's X-Request-Id and X-Session-Id, the protocol) from the serve layer, joined
// by the Engine request id. No prompt or output text is ever reported.

#include "infernix/types.h"
#include "serve/request_json.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace infernix::serve {

// Client identifiers are kept only when they are 1-128 characters of [A-Za-z0-9._:-].
[[nodiscard]] bool valid_client_id(std::string_view value) noexcept;

struct SlotIdentity {
    std::string request_id;  // the client's X-Request-Id when valid, else upstream_id
    std::string upstream_id; // Infernix's own response id (x-request-id, Anthropic request-id)
    std::string session_id;  // the client's X-Session-Id when valid
    std::string protocol;
    std::string model;
    bool stream                  = false;
    std::uint64_t started_unix_ms = 0;
};

// Static facts every slot object repeats.
struct SlotFacts {
    std::uint32_t max_context = 0;
    bool speculative          = false;
};

// The serve-side half: who each Engine request is, and the last counter sample of each lane for
// its token rates. Thread-safe; HTTP workers register, release and render concurrently.
class SlotRegistry {
public:
    void begin(std::uint64_t engine_request_id, SlotIdentity identity);
    void end(std::uint64_t engine_request_id);

    // The slot array for one published lane view at `now_unix_ms`.
    [[nodiscard]] RequestJson render(const SlotFacts& facts, const infernix::RuntimeStats& stats,
                                     std::uint64_t now_unix_ms);

    // Requests registered and not yet released (tests, diagnostics).
    [[nodiscard]] std::size_t live() const;

private:
    struct RateSample {
        std::uint64_t at_unix_ms             = 0;
        std::uint32_t generated_tokens       = 0;
        std::uint32_t computed_prompt_tokens = 0;
        std::optional<double> decode_tps;
        std::optional<double> prefill_tps;
    };

    mutable std::mutex mutex_;
    std::map<std::uint64_t, SlotIdentity> identities_;
    std::map<std::uint64_t, RateSample> rates_; // keyed by Engine request id
};

// The phase word of a lane activity: idle, prefill (binding, prompt or replay), decode (generating
// or committing control tokens) or queued (giving its lane back under resource pressure).
[[nodiscard]] const char* slot_phase(infernix::LaneActivity activity) noexcept;
[[nodiscard]] const char* lane_activity_name(infernix::LaneActivity activity) noexcept;

} // namespace infernix::serve
