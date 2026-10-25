#pragma once

// Temporary A/B switches (INFERNIX_TMP_*): the one place that reads them. Every switch is registered
// in temporary_switches.cpp (name, the models it acts on, what it does), all of them are read
// together once per process from the environment, and the default (unset) is the behaviour without
// the switch. The Engine records describe() for its model in LoadSummary::temporary_switches, which
// the startup log prints as one `temporary switches | ...` line, so an A/B arm can be checked from
// its log; op benches print describe() themselves. A switch stays only until its A/B is decided:
// then its registration, its readers and the losing arm go together.
//
// Readers parse the raw value. They throw std::invalid_argument on a malformed value (a bad arm
// fails at startup, never silently runs the default) and std::logic_error on an unregistered name.

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
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

// INFERNIX_TMP_Q56_FP32_SCALE=1: the Q5/Q6 prefill MMA kernels (q5/q6_a16_mma and the sparse-MoE
// qx_down) feed the integer codes to the MMA and apply each 64-group scale to that group's FP32
// partial sum, as the Q4 gate/up kernel does, instead of rounding q x scale to BF16. Default off.
// It changes reproducible-route prefill bits on every Q5/Q6 artifact and needs a gate.
[[nodiscard]] bool q56_fp32_scale();

// INFERNIX_TMP_GW_SHAPE=<rows>x<stages>: the groupwise sliced-K pipelined kernel's forced block rows
// (16, 32 or 64) and ring stages (1-4) instead of choose_sliced_k_shape's (ops/linear/common/
// sliced_k_route.h); unset (0x0) keeps the rule.
struct GwShape {
    int block_rows = 0;
    int stages     = 0;
};
[[nodiscard]] GwShape gw_shape();

// The projection classes of a Qwen3.5 text layer (and the vocabulary head) that
// FastPath::BatchedA8SubsetDecode may give one-term 8-bit activations.
enum class A8SubsetClass : std::uint8_t {
    Input,  // attention and GDN input projections
    GateUp, // dense MLP gate/up
    Output, // attention and GDN output projections
    Down,   // dense MLP down
    Head,   // the vocabulary head
};
inline constexpr std::size_t kA8SubsetClasses = 5;

// Where batched-a8-subset-decode acts:
//   INFERNIX_TMP_A8_SUBSET_MIN_COLUMNS=N  rounds of at least N columns (default 65: the two-term
//                                         tiles step up from 64 columns, one term does not)
//   INFERNIX_TMP_A8_SUBSET_CLASSES=a,b    input, gate_up, output, down, head (default input,gate_up:
//                                         where one term beats A16 by the most per op)
//   INFERNIX_TMP_A8_SUBSET_LAYERS=a-b,c   text layers, inclusive ranges (default every layer)
struct A8Subset {
    std::int32_t min_columns = 65;
    std::array<bool, kA8SubsetClasses> classes{true, true, false, false, false};
    // Inclusive layer ranges; empty means every layer.
    std::vector<std::pair<std::int32_t, std::int32_t>> layers;

    // Whether a projection of `kind` in text layer `layer` (-1 for the head) takes one term.
    [[nodiscard]] bool covers(A8SubsetClass kind, std::int32_t layer) const noexcept;
};
[[nodiscard]] const A8Subset& a8_subset();

// The expert offload stack's switches (Qwen3.8-Flash-Next's Program and runtime::expert_offload's
// ExpertOffload, which serves Qwen3.5's offloaded Programs). They change where a miss is served, which
// threads run where, which experts the cache admits and how many staging slots a call gets: placement
// only, so every output bit is unchanged (Flash-Next design §16.2: a CPU-served expert equals the
// GPU's). Unlike the other switches, a malformed value is ignored (the default runs) and listed by
// describe(); runtime/expert_offload/expert_offload.h applies them (link_share_256, plain_divisor,
// cpu_placement).
//
//   INFERNIX_TMP_LINK_SHARE       auto | <share in (0, 1]>   Strata's measured link share instead of
//                                 the divisor rule (Flash-Next §19.3.12): the link keeps
//                                 floor(M s) of a call's M host-memory misses, s = L / (L + 1.85 R)
//                                 (L the startup link probe, R the CPU-rate probe's experts/s times
//                                 the record bytes), rounded to 0.05 within [0.10, 0.60] as Strata
//                                 does; a number fixes s.
//   INFERNIX_TMP_PLAIN_DIVISOR_ALL on | <short>,<long>       plain decode's one-column divisor by
//                                 context (d = short below 1,536 context tokens, long from there;
//                                 "on": 1, 2, the GGUF rule) for every bank, not only GGUF imports.
//   INFERNIX_TMP_CPU_PLACEMENT    physical | physical-set [,<workers>]   the CPU expert team one worker
//                                 per physical core, P-cores first, no SMT siblings: "physical" pins
//                                 worker i to core i's first logical CPU, "physical-set" lets every
//                                 worker run on any core's first logical CPU. <workers> overrides
//                                 the worker count (P-1 + 8 E on an i9-13900K: physical,15 with
//                                 the engine isolated).
//   INFERNIX_TMP_ENGINE_ISOLATE   1    the engine thread on the last P-core (both its logical CPUs),
//                                 which no CPU expert worker may use.
//   INFERNIX_TMP_PROMPT_SEED      <credit>[,<budget>]   prompt seeding's credit (today 256; 0 turns
//                                 it off) and promotions per layer at the prompt's end (today 0).
//   INFERNIX_TMP_STAGING_SLOTS    64..512    the staging slots instead of the width rule.
struct OffloadSwitches {
    enum class LinkShare : std::uint8_t { kDivisor, kAuto, kFixed };
    LinkShare link_share    = LinkShare::kDivisor;
    double link_share_fixed = 0.0;

    bool plain_divisor_all   = false;
    std::int32_t plain_short = 1;
    std::int32_t plain_long  = 2;

    enum class Placement : std::uint8_t { kDefault, kPhysical, kPhysicalSet };
    Placement placement   = Placement::kDefault;
    std::uint32_t workers = 0; // 0: the configured count
    bool engine_isolate   = false;

    double prompt_credit        = -1.0; // < 0: the Program's kPromptSeedCredit (256)
    std::uint32_t prompt_budget = 0;
    [[nodiscard]] double credit_or(double rule) const noexcept { return prompt_credit >= 0.0 ? prompt_credit : rule; }

    std::int32_t staging_slots = 0; // 0: the width rule

    // The values that could not be parsed (NAME=value), which run the default.
    std::vector<std::string> errors;
};
[[nodiscard]] const OffloadSwitches& offload();

} // namespace infernix::temporary_switches
