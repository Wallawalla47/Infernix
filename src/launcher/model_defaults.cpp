#include "launcher/model_defaults.h"

#include <algorithm>
#include <cmath>

namespace infernix::launcher {
namespace {

std::string stem_of(const ScannedModel& model) {
    std::string stem = path_utf8(model.path.stem());
    const std::size_t of = stem.rfind("-of-");
    if (of != std::string::npos && of >= 6 && stem[of - 6] == '-') { stem.resize(of - 6); }
    return stem;
}

} // namespace

std::uint64_t fitting_27b_context(const ScannedModel& model, const HostFacts& host) {
    // Measured on the official NVFP4 artifact (RTX 5090 32 GB, INT8 KV, two lanes, DFlash2 and
    // n-gram drafts; 2026-10-31): the Engine's minimum reservation is 2.03 GB plus 33,792 bytes per
    // context token, and about 0.74 GB beside the file's bytes went to the CUDA context, workspaces
    // and the display (1 GB is assumed, leaving room for other programs). The Engine's automatic
    // headroom is 256 MiB on a GPU without a display and 1 GiB with one (or when NVML cannot tell).
    // A file's unloaded parts (vision without --vision) make this lower than what fits, never higher.
    constexpr std::uint64_t kUnknownGpu = 188416;
    constexpr double kPerToken          = 33792.0;
    constexpr double kFixed             = 2.03e9;
    constexpr double kBeside            = 1.0e9;
    constexpr std::uint64_t kStep       = 4096;
    if (host.vram_total_bytes == 0) { return kUnknownGpu; }
    const double headroom = host.display == false ? 268435456.0 : 1073741824.0;
    const double budget =
        static_cast<double>(host.vram_total_bytes) - static_cast<double>(model.bytes) - kBeside - headroom - kFixed;
    const auto tokens = budget <= 0.0 ? 0 : static_cast<std::uint64_t>(budget / kPerToken) / kStep * kStep;
    return std::clamp<std::uint64_t>(tokens, 32768, 240000);
}

const ScannedDrafter* best_drafter(const LibraryScan& scan, const ScannedModel& model) {
    const ScannedDrafter* best = nullptr;
    for (const ScannedDrafter& drafter : scan.drafters) {
        if (std::find(drafter.compatible_models.begin(), drafter.compatible_models.end(), model.key) ==
            drafter.compatible_models.end()) {
            continue;
        }
        // Several copies of compatible drafters: the newest file.
        if (best == nullptr || drafter.modified > best->modified) { best = &drafter; }
    }
    return best;
}

const ScannedNgramVolume* best_ngram_volume(const LibraryScan& scan, const ScannedModel& model) {
    const ScannedNgramVolume* best = nullptr;
    int best_rank                  = 0;
    for (const ScannedNgramVolume& volume : scan.ngram_volumes) {
        const std::optional<int> rank = ngram_rank(model, volume);
        if (!rank) { continue; }
        if (best == nullptr || *rank < best_rank) {
            best      = &volume;
            best_rank = *rank;
        }
    }
    return best;
}

Args default_args(const LibraryScan& scan, const ScannedModel& model, const HostFacts& host) {
    Args args;
    const auto add = [&](std::string_view flag, std::string_view value = {}) {
        args.emplace_back(flag);
        if (!value.empty()) { args.emplace_back(value); }
    };
    const bool has_drafter_component = model.has_mtp || model.has_dflash || model.has_dflash2;

    if (model.family == "qwen3.8-flash-next") {
        add("--max-context", "262144");
    } else if (model.family == "qwen3.8-27b" || model.family == "qwen3.6-27b") {
        add("--max-context", std::to_string(fitting_27b_context(model, host)));
    } else if (model.native_context != 0) {
        add("--max-context", std::to_string(std::min<std::uint64_t>(model.native_context, 262144)));
    }
    add("--max-concurrency", "2");
    add("--kv-dtype", "int8");

    // Speculative decoding: the Engine picks the drafter measured fastest for the artifact.
    const ScannedDrafter* drafter = model.accepts_draft_model ? best_drafter(scan, model) : nullptr;
    if (has_drafter_component || drafter != nullptr) {
        add("--spec", "auto");
        add("--draft-tokens", "auto");
        // The proposal head pays off with DFlash/DFlash2 and the small-model MTP drafters, not on the
        // 122B whose MTP rounds measured no gain from it.
        if (model.has_proposal && model.family != "qwen3.5-122b-a10b") { add("--lm-head-draft"); }
    }
    if (drafter != nullptr) { add("--draft-model", path_utf8(drafter->path)); }
    if (model.family == "qwen3.8-27b" && model.has_dflash2) { add("--ngram-draft-tokens", "auto"); }

    if (model.needs_ngram_volume && !model.own_ngram_volume) {
        if (const ScannedNgramVolume* volume = best_ngram_volume(scan, model)) {
            add("--ngram-volume", path_utf8(volume->path));
        }
    }

    if (model.family == "qwen3.5-122b-a10b") {
        // The experts take most of the RAM; a 4 GiB Host prefix cache leaves it to them.
        add("--host-context-mib", "4096");
    } else if (!model.offloaded_experts && host.ram_total_bytes >= (32ull << 30)) {
        // Dense models leave RAM free: a larger Host tier keeps more agent sessions cached.
        const double mib = static_cast<double>(host.ram_total_bytes) / (1 << 20) * 0.55;
        const auto rounded = static_cast<std::uint64_t>(std::floor(mib / 1000.0) * 1000.0);
        add("--host-context-mib", std::to_string(std::min<std::uint64_t>(rounded, 52000)));
    }
    if (model.family == "qwen3.8-27b") { add("--prefill-chunk", "4096"); }

    add("--preserve-thinking");
    add("--tolerant-tool-calls");
    // Agent clients queue several requests; let them wait for a lane instead of failing.
    add("--pending-timeout-ms", "900000");
    return args;
}

std::string default_alias(const LibraryScan& scan, const ScannedModel& model) {
    const std::string base = stem_of(model);
    int index              = 1;
    for (const ScannedModel& other : scan.models) {
        if (&other == &model || other.key == model.key) { break; }
        if (iequals(stem_of(other), base)) { ++index; }
    }
    return index == 1 ? base : base + "-" + std::to_string(index);
}

} // namespace infernix::launcher
