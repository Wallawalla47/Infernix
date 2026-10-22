#pragma once

// The MTP proposal head a real-artifact test selects: the optimized head where the artifact ships one
// (the Qwen3.8-27B artifacts bind `proposal/head`), else the full output head (for example the
// Qwen3.5-122B-A10B's).

#include "artifact/reader.h"
#include "infernix/types.h"

#include <filesystem>

namespace infernix::test {

[[nodiscard]] inline ProposalHead available_proposal_head(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    return reader.directory().bindings.contains("proposal/head") ? ProposalHead::Optimized : ProposalHead::Full;
}

} // namespace infernix::test
