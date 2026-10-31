#pragma once

// Why a model did not fit, and what to change. A load that ran out of GPU memory (infernix-serve's
// own sizing errors: the KV reservation beside the weights, a device allocation) or system RAM
// (pinning, the SSD expert tier, or the launcher stopping it at the RAM floor) is classified from
// the server's message, and the suggestions are concrete launch-parameter changes for that model,
// computed from the figures the message gives where it gives them. A server that could not bind its
// port gets advice instead (the launcher sets the port).

#include "launcher/args.h"
#include "launcher/artifact_scan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::launcher {

struct LoadSuggestion {
    std::string title;
    std::string detail;
    // The model's launch parameters with the change made; none for advice without a flag.
    std::optional<Args> args;
};

[[nodiscard]] Json to_json(const LoadSuggestion& suggestion);

// What the machine looked like when the load failed.
struct ResourceFacts {
    std::uint64_t ram_total      = 0;
    std::uint64_t ram_available  = 0;
    std::uint64_t vram_total     = 0;
    std::uint64_t vram_used      = 0; // after the server ended: what other programs hold
};

// "vram" or "ram" when `error` says the model did not fit, "port" when infernix-serve could not
// bind its port; nullopt for other failures.
[[nodiscard]] std::optional<std::string> classify_load_failure(std::string_view error);

// What to do when infernix-serve could not bind `port` (`fixed`: the launcher's Server port
// setting chose it). Advice only: the launcher, not the launch parameters, sets the port.
[[nodiscard]] std::vector<LoadSuggestion> suggest_port_fixes(int port, bool fixed);

[[nodiscard]] std::vector<LoadSuggestion> suggest_load_fixes(const std::string& kind, const ScannedModel& model,
                                                             const Args& args, std::string_view error,
                                                             const ResourceFacts& facts);

} // namespace infernix::launcher
