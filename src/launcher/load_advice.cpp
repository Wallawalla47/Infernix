#include "launcher/load_advice.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <regex>

namespace infernix::launcher {
namespace {

constexpr std::uint64_t kStep          = 4096;  // contexts are suggested in whole 4,096-token steps
constexpr std::uint64_t kMinContext    = 8192;
constexpr std::uint64_t kDefaultRamFree = 4096; // MiB; infernix-serve's --ram-free-mib default
constexpr double kGiB                  = 1024.0 * 1024.0 * 1024.0;

std::string gib(double bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f GiB", bytes / kGiB);
    return text;
}

std::string thousands(std::uint64_t value) {
    std::string digits = std::to_string(value);
    for (int at = static_cast<int>(digits.size()) - 3; at > 0; at -= 3) { digits.insert(static_cast<std::size_t>(at), ","); }
    return digits;
}

std::optional<std::uint64_t> number_flag(const Args& args, std::string_view flag) {
    const std::optional<std::string> value = flag_value(args, flag);
    if (!value) { return std::nullopt; }
    try {
        return std::stoull(*value);
    } catch (const std::exception&) { return std::nullopt; }
}

Args with_flag(Args args, std::string_view flag, std::string_view value) {
    set_flag(args, flag, value);
    return args;
}

// The KV reservation figures of infernix-serve's sizing errors (bytes).
struct KvShortfall {
    double minimum   = 0; // the Engine's minimum runtime reservation at this context
    double headroom  = 0; // the automatic headroom kept free
    double available = 0; // VRAM left after the weights
};

std::optional<KvShortfall> parse_kv_shortfall(std::string_view error) {
    const std::string text(error);
    std::smatch match;
    static const std::regex automatic(R"(\((\d+) bytes minimum Engine runtime reservation \+ (\d+) bytes automatic headroom\), but only (\d+) bytes are available after weights)");
    if (std::regex_search(text, match, automatic)) {
        return KvShortfall{std::stod(match[1]), std::stod(match[2]), std::stod(match[3])};
    }
    static const std::regex explicit_capacity(R"(requested Engine runtime reservation requires (\d+) bytes, but only (\d+) bytes are available after weights)");
    if (std::regex_search(text, match, explicit_capacity)) {
        return KvShortfall{std::stod(match[1]), 0.0, std::stod(match[2])};
    }
    return std::nullopt;
}

// The context whose reservation fits `budget`, from the reservation at `context`. Scaling through
// zero overstates the fit (part of the reservation does not grow with the context), so 90 % of it.
std::uint64_t fitting_context(std::uint64_t context, double reservation, double budget) {
    if (reservation <= 0 || budget <= 0) { return 0; }
    const double tokens = static_cast<double>(context) * budget / reservation * 0.9;
    return static_cast<std::uint64_t>(tokens) / kStep * kStep;
}

void vram_suggestions(const ScannedModel& model, const Args& args, std::string_view error, const ResourceFacts& facts,
                      std::vector<LoadSuggestion>& out) {
    const std::uint64_t context     = context_length(args);
    const std::uint64_t lanes       = number_flag(args, "--max-concurrency").value_or(1);
    const std::optional<KvShortfall> kv = parse_kv_shortfall(error);
    if (kv) {
        const std::uint64_t fit = fitting_context(context, kv->minimum, kv->available - kv->headroom);
        if (fit >= kMinContext && fit < context) {
            out.push_back({"Lower the context to " + thousands(fit) + " tokens",
                           "At " + thousands(context) + " tokens the KV cache and its workspace need " + gib(kv->minimum) +
                               " beside the weights; " + gib(kv->available - kv->headroom) +
                               " is free after the headroom (--max-context " + std::to_string(fit) + ").",
                           with_flag(args, "--max-context", std::to_string(fit))});
        }
        if (kv->headroom > 0) {
            const bool enough = kv->minimum <= kv->available;
            out.push_back({"Use the GPU's reserved headroom",
                           "Startup keeps " + gib(kv->headroom) + " of VRAM free for the display and other programs; "
                               "--vram-headroom-mib 0 gives it to the model" +
                               std::string(enough ? " (enough on its own here)" : "") +
                               ". Only when nothing else draws on this GPU.",
                           with_flag(args, "--vram-headroom-mib", "0")});
        }
    } else if (context > kMinContext) {
        const std::uint64_t half = std::max<std::uint64_t>(context / 2 / kStep * kStep, kMinContext);
        out.push_back({"Lower the context to " + thousands(half) + " tokens",
                       "The KV cache grows with --max-context" +
                           std::string(model.offloaded_experts ? " (with offloaded experts it is --max-context x "
                                                                 "--max-concurrency, and the expert cache takes the rest)"
                                                               : "") +
                           ".",
                       with_flag(args, "--max-context", std::to_string(half))});
    }
    if (lanes > 1) {
        out.push_back({"Serve one request at a time",
                       "Each of the " + std::to_string(lanes) +
                           " lanes keeps its own state and draft buffers (--max-concurrency 1).",
                       with_flag(args, "--max-concurrency", "1")});
    }
    const std::optional<std::string> kv_dtype = flag_value(args, "--kv-dtype");
    if (!kv_dtype || *kv_dtype == "bf16") {
        out.push_back({"Store the KV cache as INT8",
                       "Half the size of the default BF16 cache, and the format Infernix recommends (--kv-dtype int8).",
                       with_flag(args, "--kv-dtype", "int8")});
    }
    if (facts.vram_used > (1ull << 30)) {
        out.push_back({"Close other programs using the GPU",
                       "Other programs hold " + gib(static_cast<double>(facts.vram_used)) + " of the GPU's " +
                           gib(static_cast<double>(facts.vram_total)) + ".",
                       std::nullopt});
    }
}

void ram_suggestions(const ScannedModel& model, const Args& args, const ResourceFacts& facts,
                     std::vector<LoadSuggestion>& out) {
    if (model.offloaded_experts) {
        const std::uint64_t free_now = number_flag(args, "--ram-free-mib").value_or(kDefaultRamFree);
        const std::uint64_t more     = free_now + 8192;
        Args changed                 = with_flag(args, "--ram-free-mib", std::to_string(more));
        set_flag(changed, "--expert-ram-mib", "auto");
        out.push_back({"Leave more RAM free",
                       "--ram-free-mib " + std::to_string(more) + " with --expert-ram-mib auto: the experts then take " +
                           "8 GiB less RAM and the rest are read from the SSD as needed (slower on misses).",
                       changed});
    }
    const std::optional<std::uint64_t> host = number_flag(args, "--host-context-mib");
    if (host && *host > 4096) {
        const std::uint64_t smaller = model.offloaded_experts ? 4096 : std::max<std::uint64_t>(*host / 2 / 1000 * 1000, 4096);
        out.push_back({"Shrink the Host prefix cache",
                       "--host-context-mib " + std::to_string(smaller) + ": it pins " + std::to_string(*host) +
                           " MiB of RAM for cached conversations now.",
                       with_flag(args, "--host-context-mib", std::to_string(smaller))});
    }
    if (facts.ram_total > 0) {
        out.push_back({"Close other programs",
                       gib(static_cast<double>(facts.ram_available)) + " of " + gib(static_cast<double>(facts.ram_total)) +
                           " of RAM is free now.",
                       std::nullopt});
    }
}

} // namespace

Json to_json(const LoadSuggestion& suggestion) {
    return {{"title", suggestion.title},
            {"detail", suggestion.detail},
            {"args", suggestion.args ? Json(*suggestion.args) : Json(nullptr)}};
}

std::optional<std::string> classify_load_failure(std::string_view error) {
    const std::string text(error);
    // infernix-serve binds its port exclusively and exits before loading when it cannot
    // ("cannot bind HOST:PORT: the port is in use ..." or a port Windows refuses).
    if (text.find("cannot bind ") != std::string::npos) { return std::string("port"); }
    static const std::regex vram(
        R"(automatic KV capacity requires|Engine runtime reservation requires|does not fit in device memory|cudaErrorMemoryAllocation|CUDA out of memory|\bVRAM\b|device memory)",
        std::regex::icase);
    static const std::regex ram(
        R"(\bRAM\b|page-lock|pinned|host memory|bad_alloc|commit limit|paging file)", std::regex::icase);
    if (std::regex_search(text, vram)) { return std::string("vram"); }
    if (std::regex_search(text, ram)) { return std::string("ram"); }
    return std::nullopt;
}

std::vector<LoadSuggestion> suggest_port_fixes(int port, bool fixed) {
    const std::string number = std::to_string(port);
    std::vector<LoadSuggestion> out;
    if (fixed) {
        out.push_back({"Let the launcher choose the server's port",
                       "Settings → infernix-serve → Server port 0 picks a free port for every start; port " + number +
                           " is held by another program.",
                       std::nullopt});
    }
#ifdef _WIN32
    const std::string owners = "netstat -ano lists the owner's process id";
#else
    const std::string owners = "ss -ltnp names the owner";
#endif
    out.push_back({"Stop the program that holds port " + number,
                   "Another server, an infernix-serve started from a script say, is listening on it (" + owners + ").",
                   std::nullopt});
#ifdef _WIN32
    out.push_back({"Check Windows' reserved port ranges",
                   "A port inside a range Windows reserves cannot be used even when free: netsh int ipv4 show "
                   "excludedportrange protocol=tcp.",
                   std::nullopt});
#endif
    return out;
}

std::vector<LoadSuggestion> suggest_load_fixes(const std::string& kind, const ScannedModel& model, const Args& args,
                                               std::string_view error, const ResourceFacts& facts) {
    std::vector<LoadSuggestion> out;
    if (kind == "vram") { vram_suggestions(model, args, error, facts, out); }
    if (kind == "ram") { ram_suggestions(model, args, facts, out); }
    return out;
}

} // namespace infernix::launcher
