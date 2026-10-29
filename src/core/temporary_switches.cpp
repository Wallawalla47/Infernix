#include "core/temporary_switches.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace infernix::temporary_switches {
namespace {

// Flag: a 0/1 switch, which describe() validates; Value: parsed by its reader.
enum class Kind : std::uint8_t { Flag, Value };

struct Registration {
    std::string_view name;
    Scope scope;
    Kind kind;
};

// Every switch, grouped by the change that added it; the reader named beside each parses it.
constexpr Registration kSwitches[] = {
    // Groupwise Q4/Q5/Q6 sliced-K kernels (ops/linear/common/sliced_k_route.cpp): a forced <rows>x<stages>
    // shape of the one-wave pipelined kernel. Same bits.
    {"INFERNIX_TMP_GW_SHAPE", Scope::Qwen3_5, Kind::Value},
    // Qwen3.5 MTP drafting (models/qwen3_5/program/planning/startup.cpp, mtp_draft_switches): calibrated
    // extents and a per-round draft log path. Draft-only.
    {"INFERNIX_TMP_QWEN35_MTP_CALIBRATED", Scope::Qwen3_5, Kind::Flag},
    {"INFERNIX_TMP_QWEN35_MTP_DRAFT_LOG", Scope::Qwen3_5, Kind::Value},
    // The scheduler's decode share (runtime/engine/engine_core.h, decode_share_from_environment): a
    // prefill unit's share of decode batches and the remaining tokens below which it stops. Same bits.
    {"INFERNIX_TMP_DECODE_SHARE", Scope::All, Kind::Value},
    {"INFERNIX_TMP_DECODE_SHARE_MIN_TOKENS", Scope::All, Kind::Value},
    // Flash-Next's prefill walk step while another lane decodes (models/qwen4_exp/program/prefill_walk.cpp).
    {"INFERNIX_TMP_DECODE_SHARE_WALK_TOKENS", Scope::Qwen4Exp, Kind::Value},
    // Diagnostic: the CPU team's rates on 32 records in three memories, logged at startup
    // (runtime/expert_offload/expert_offload.cpp, cpu_probe_detail; candidates §11.2 OR-08).
    {"INFERNIX_TMP_CPU_PROBE_DETAIL", Scope::All, Kind::Flag},
    // The expert offload stack's placement switches (offload()).
    {"INFERNIX_TMP_LINK_SHARE", Scope::All, Kind::Value},
    {"INFERNIX_TMP_CPU_PLACEMENT", Scope::All, Kind::Value},
    {"INFERNIX_TMP_STAGING_SLOTS", Scope::All, Kind::Value},

    // Debug diagnostic: the prefill projection inputs' FP16 headroom scan, written to this path
    // (models/qwen4_exp/execution/activation_scan.cpp).
    {"INFERNIX_TMP_ACTIVATION_SCAN", Scope::Qwen4Exp, Kind::Value},
    // Offloaded experts: =<n> (1-64) runs a routing-sequence predictor beside the expert cache whose next
    // n experts per layer and lane each eviction spares while another resident is left
    // (runtime/expert_offload/route_predictor.h; candidates §13 PUB-22; A/B only after its replay
    // gains 3 points). Same bits (placement only).
    {"INFERNIX_TMP_ROUTE_PROTECT", Scope::All, Kind::Value},
    // =<rounds>: the replay bench (candidates §8.5 CF-03). The expert residency freezes after that many
    // rounds (its loads landed, then no admission, eviction, landing or prompt seed); Flash-Next's MTP
    // width cost keeps its prior, so greedy rounds repeat across runs; its release log reports the
    // decode rounds' device time and the CPU waits inside it (core/round_clock.h). Same bits.
    {"INFERNIX_TMP_REPLAY_BENCH", Scope::All, Kind::Value},
    // Flash-Next elastic KV: the idle shrink also runs while requests decode (after kKvIdleShrink without
    // an admission or release), releasing idle conversations' host-backed device blocks so the free top
    // returns to the expert frames (models/qwen4_exp/program/program_impl.h apply_vram_target;
    // candidates §8.3 CF-06). Same bits.
    {"INFERNIX_TMP_KV_SHRINK_BUSY", Scope::Qwen4Exp, Kind::Flag},
    // Offloaded-expert caches: =<fraction> (0, 1]; a prompt whose reused prefix covers at least that much
    // of it neither scores its prefill routes nor seeds its experts (Flash-Next: models/qwen4_exp/program/
    // program_impl.h apply_prompt_routes; the 122B: models/qwen3_5/program/offload.cpp end_pass; candidates
    // CF-01). Same bits (placement only).
    {"INFERNIX_TMP_CONTINUING_PROMPT", Scope::All, Kind::Value},
    // Diagnostic: =<dir>; after each prompt's prefill on a BF16 KV cache, its K and V pages are written
    // there with their layout (models/qwen3_5/program/kv_dump.cpp; tools/kv_lossless_count counts the
    // 12-bit lossless blocks of candidates §23.2 PB-01). Output-neutral.
    {"INFERNIX_TMP_KV_DUMP", Scope::Qwen3_5, Kind::Value},
    // Diagnostic (2026-10-29): =<MB/s>; the artifact read queues (core/direct_read_queue.cpp) hold each finished read
    // until a device of that rate would have delivered it, to measure the SSD tier on a slower drive. Same bits.
    {"INFERNIX_TMP_SSD_MBPS", Scope::All, Kind::Value},
};
constexpr std::size_t kCount = std::size(kSwitches);

const char* environment(const char* name) {
#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4996) // std::getenv: read once, before any thread could change it
#endif
    return std::getenv(name);
#ifdef _MSC_VER
#    pragma warning(pop)
#endif
}

// Every registered switch's raw value, read together once.
const std::array<std::optional<std::string>, kCount>& snapshot() {
    static const auto values = [] {
        std::array<std::optional<std::string>, kCount> out;
        for (std::size_t i = 0; i < kCount; ++i) {
            if (const char* v = environment(std::string(kSwitches[i].name).c_str()); v != nullptr && *v != '\0') {
                out[i] = std::string(v);
            }
        }
        return out;
    }();
    return values;
}

std::size_t index_of(std::string_view name) {
    for (std::size_t i = 0; i < kCount; ++i) {
        if (kSwitches[i].name == name) { return i; }
    }
    throw std::logic_error("temporary switch " + std::string(name) + " is not registered (core/temporary_switches.cpp)");
}

[[noreturn]] void malformed(std::string_view name, const std::string& text, const char* expected) {
    throw std::invalid_argument(std::string(name) + "=" + text + ": expected " + expected);
}

bool parse_long(const std::string& text, long lo, long hi, long& out) {
    if (text.empty()) { return false; }
    char* end = nullptr;
    errno     = 0;
    out       = std::strtol(text.c_str(), &end, 10);
    return errno == 0 && end != nullptr && *end == '\0' && out >= lo && out <= hi;
}

bool parse_double(const std::string& text, double& out) {
    if (text.empty()) { return false; }
    char* end = nullptr;
    errno     = 0;
    out       = std::strtod(text.c_str(), &end);
    return errno == 0 && end != nullptr && *end == '\0' && std::isfinite(out);
}

// "a,b" -> a and b (no comma: b empty).
std::pair<std::string, std::string> split_comma(const std::string& text) {
    const auto comma = text.find(',');
    if (comma == std::string::npos) { return {text, {}}; }
    return {text.substr(0, comma), text.substr(comma + 1)};
}

GwShape read_gw_shape() {
    const auto v = value("INFERNIX_TMP_GW_SHAPE");
    if (!v) { return {}; }
    const auto bad = [&] { malformed("INFERNIX_TMP_GW_SHAPE", *v, "<rows>x<stages>, rows 16, 32 or 64 and stages 1-4 (for example 32x3)"); };
    const std::size_t x = v->find('x');
    long rows = 0, stages = 0;
    if (x == std::string::npos || !parse_long(v->substr(0, x), 16, 64, rows) || !parse_long(v->substr(x + 1), 1, 4, stages) ||
        (rows != 16 && rows != 32 && rows != 64)) {
        bad();
    }
    return {static_cast<int>(rows), static_cast<int>(stages)};
}

OffloadSwitches read_offload() {
    OffloadSwitches s;
    const auto reject = [&](const char* name, const std::string& v) { s.errors.push_back(std::string(name) + "=" + v); };
    if (const auto v = value("INFERNIX_TMP_LINK_SHARE")) {
        double share = 0.0;
        if (*v == "auto") {
            s.link_share = OffloadSwitches::LinkShare::kAuto;
        } else if (parse_double(*v, share) && share > 0.0 && share <= 1.0) {
            s.link_share       = OffloadSwitches::LinkShare::kFixed;
            s.link_share_fixed = share;
        } else {
            reject("INFERNIX_TMP_LINK_SHARE", *v);
        }
    }
    if (const auto v = value("INFERNIX_TMP_CPU_PLACEMENT")) {
        const auto [mode, count] = split_comma(*v);
        long workers             = 0;
        const bool counted       = count.empty() || parse_long(count, 1, 40, workers);
        if (counted && (mode == "default" || mode == "physical" || mode == "physical-set")) {
            s.placement = mode == "physical"       ? OffloadSwitches::Placement::kPhysical
                          : mode == "physical-set" ? OffloadSwitches::Placement::kPhysicalSet
                                                   : OffloadSwitches::Placement::kDefault;
            s.workers   = static_cast<std::uint32_t>(workers);
        } else {
            reject("INFERNIX_TMP_CPU_PLACEMENT", *v);
        }
    }
    if (const auto v = value("INFERNIX_TMP_STAGING_SLOTS")) {
        long slots = 0;
        if (parse_long(*v, 8, 512, slots)) {
            s.staging_slots = static_cast<std::int32_t>(slots);
        } else {
            reject("INFERNIX_TMP_STAGING_SLOTS", *v);
        }
    }
    return s;
}

} // namespace

std::optional<std::string> value(std::string_view name) { return snapshot()[index_of(name)]; }

bool flag(std::string_view name) {
    const auto v = value(name);
    if (!v || *v == "0") { return false; }
    if (*v == "1") { return true; }
    malformed(name, *v, "0 or 1");
}

std::int64_t integer(std::string_view name, std::int64_t fallback, std::int64_t lo, std::int64_t hi) {
    const auto v = value(name);
    if (!v) { return fallback; }
    char* end = nullptr;
    errno     = 0;
    const long long out = std::strtoll(v->c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || out < lo || out > hi) {
        malformed(name, *v, ("an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]").c_str());
    }
    return out;
}

double number(std::string_view name, double fallback, double lo, double hi) {
    const auto v = value(name);
    if (!v) { return fallback; }
    double out = 0.0;
    if (!parse_double(*v, out) || !(out >= lo && out <= hi)) {
        malformed(name, *v, ("a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]").c_str());
    }
    return out;
}

GwShape gw_shape() {
    static const GwShape shape = read_gw_shape();
    return shape;
}

const OffloadSwitches& offload() {
    static const OffloadSwitches switches = read_offload();
    return switches;
}

std::string describe(Scope scope) {
    const auto& values = snapshot();
    // A malformed arm fails here, at startup, rather than at its first use.
    if ((static_cast<unsigned>(scope) & static_cast<unsigned>(Scope::Qwen3_5)) != 0) {
        (void)gw_shape();
    }
    std::string out;
    for (std::size_t i = 0; i < kCount; ++i) {
        if ((static_cast<unsigned>(kSwitches[i].scope) & static_cast<unsigned>(scope)) == 0) { continue; }
        if (kSwitches[i].kind == Kind::Flag) { (void)flag(kSwitches[i].name); }
        if (!out.empty()) { out += " | "; }
        out += std::string(kSwitches[i].name) + "=" + (values[i] ? *values[i] : std::string("unset"));
    }
    if (const OffloadSwitches& offloaded = offload(); !offloaded.errors.empty()) {
        out += "; ignored (invalid):";
        for (const auto& error : offloaded.errors) { out += " " + error; }
    }
    return out;
}

} // namespace infernix::temporary_switches
