#pragma once

// Temporary A/B switches (INFERNIX_TMP_*): the one place that reads them. Every switch is registered
// in temporary_switches.cpp (name, the models it acts on, what it does), all of them are read
// together once per process from the environment, and the default (unset) is the behaviour without
// the switch. The Engine records describe() for its model in LoadSummary::temporary_switches, whose
// set switches the startup log prints as one `temporary switches | ...` line (every switch's value
// at debug), so an A/B arm can be checked from its log; op benches print describe() themselves. A switch stays only until its A/B is decided:
// then its registration, its readers and the losing arm go together.
//
// Readers parse the raw value. They throw std::invalid_argument on a malformed value (a bad arm
// fails at startup, never silently runs the default) and std::logic_error on an unregistered name.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::temporary_switches {

// The models whose startup line lists a switch.
enum class Scope : std::uint8_t {
    Qwen3_5  = 1, // Qwen3_5ForCausalLM / Qwen3_5MoeForCausalLM Programs (and their kernels)
    Qwen4Exp = 2, // Qwen3.8-Flash-Next
    All      = 3, // both (the Engine core, the shared expert offload stack)
};

// The raw value, or nullopt when unset or empty.
[[nodiscard]] std::optional<std::string> value(std::string_view name);
// A 0/1 switch: unset, empty or 0 is false, 1 is true.
[[nodiscard]] bool flag(std::string_view name);
// An integer in [lo, hi], `fallback` when unset.
[[nodiscard]] std::int64_t integer(std::string_view name, std::int64_t fallback, std::int64_t lo, std::int64_t hi);
// A finite number in [lo, hi], `fallback` when unset.
[[nodiscard]] double number(std::string_view name, double fallback, double lo, double hi);

// The registered switches `scope` lists, as NAME=value (unset: "unset") separated by " | ", followed
// by the expert offload switches' ignored values when there are any (OffloadSwitches::errors).
[[nodiscard]] std::string describe(Scope scope = Scope::All);

// ---------------------------------------------------------------------------------------------
// Switches with a structured value, parsed once here for every reader.

// INFERNIX_TMP_GW_SHAPE=<rows>x<stages>: the groupwise sliced-K pipelined kernel's forced block rows
// (16, 32 or 64) and ring stages (1-4) instead of choose_sliced_k_shape's (ops/linear/common/
// sliced_k_route.h); unset (0x0) keeps the rule.
struct GwShape {
    int block_rows = 0;
    int stages     = 0;
};
[[nodiscard]] GwShape gw_shape();

// The expert offload stack's switches (Qwen3.8-Flash-Next's Program and runtime::expert_offload's
// ExpertOffload, which serves Qwen3.5's offloaded Programs). They change where a miss is served, which
// threads run where, which experts the cache admits and how many staging slots a call gets: placement
// only, so every output bit is unchanged (Flash-Next design §16.2: a CPU-served expert equals the
// GPU's). Unlike the other switches, a malformed value is ignored (the default runs) and listed by
// describe(); runtime/expert_offload/expert_offload.h applies them (link_share_256, cpu_placement).
//
//   INFERNIX_TMP_LINK_SHARE       auto | <share in (0, 1]>   Strata's measured link share instead of
//                                 the divisor rule (Flash-Next §19.3.12): the link keeps
//                                 floor(M s) of a call's M host-memory misses, s = L / (L + 1.85 R)
//                                 (L the startup link probe, R the CPU-rate probe's experts/s times
//                                 the record bytes), rounded to 0.05 within [0.10, 0.60] as Strata
//                                 does; a number fixes s.
//   INFERNIX_TMP_CPU_PLACEMENT    physical | physical-set [,<workers>]   the CPU expert team one worker
//                                 per physical core, P-cores first, no SMT siblings: "physical" pins
//                                 worker i to core i's first logical CPU, "physical-set" lets every
//                                 worker run on any core's first logical CPU. <workers> overrides
//                                 the worker count (P-1 + 8 E on an i9-13900K: physical,15).
//   INFERNIX_TMP_STAGING_SLOTS    8..512     the staging slots instead of the width rule (a call whose
//                                 jobs exceed them is not forked).
struct OffloadSwitches {
    enum class LinkShare : std::uint8_t { kDivisor, kAuto, kFixed };
    LinkShare link_share    = LinkShare::kDivisor;
    double link_share_fixed = 0.0;

    enum class Placement : std::uint8_t { kDefault, kPhysical, kPhysicalSet };
    Placement placement   = Placement::kDefault;
    std::uint32_t workers = 0; // 0: the configured count

    std::int32_t staging_slots = 0; // 0: the width rule

    // The values that could not be parsed (NAME=value), which run the default.
    std::vector<std::string> errors;
};
[[nodiscard]] const OffloadSwitches& offload();

} // namespace infernix::temporary_switches
