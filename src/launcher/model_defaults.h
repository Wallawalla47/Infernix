#pragma once

// The launcher's recommended launch parameters for each model family (README "Recommended settings
// per model"), the default public model id, and the companions it picks: the DFlash2 drafter and
// the n-gram volume found in the model folders.

#include "launcher/args.h"
#include "launcher/artifact_scan.h"

#include <cstdint>
#include <optional>
#include <string>

namespace infernix::launcher {

struct HostFacts {
    std::uint64_t ram_total_bytes  = 0;
    std::uint64_t vram_total_bytes = 0; // the first GPU's; 0 when unknown
};

// Qwen3.x-27B keeps all of its KV on the GPU: the --max-context that fits beside the artifact.
[[nodiscard]] std::uint64_t fitting_27b_context(const ScannedModel& model, const HostFacts& host);

// The best compatible DFlash2 drafter and n-gram volume for a model; nullptr when none.
[[nodiscard]] const ScannedDrafter* best_drafter(const LibraryScan& scan, const ScannedModel& model);
[[nodiscard]] const ScannedNgramVolume* best_ngram_volume(const LibraryScan& scan, const ScannedModel& model);

// Launch parameters, without the artifact path and the managed flags.
[[nodiscard]] Args default_args(const LibraryScan& scan, const ScannedModel& model, const HostFacts& host);

// The artifact's file name without its part numbering and extension; unique within the library
// (a numeric suffix separates two artifacts that would share one).
[[nodiscard]] std::string default_alias(const LibraryScan& scan, const ScannedModel& model);

} // namespace infernix::launcher
