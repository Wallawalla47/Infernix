#pragma once

// The drafter Inkling 2's real tests draft with: the artifact's DSpark drafter, or with INFERNIX_INKLING2_DRAFTER=dflash2
// the DFlash2 companion INFERNIX_INKLING2_DFLASH2 names (--spec dflash2 --draft-model), so every drafting check runs
// under either drafter.

#include "infernix/types.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace infernix::test::inkling2 {

inline bool dflash2_drafter() {
    const char* drafter = std::getenv("INFERNIX_INKLING2_DRAFTER");
    return drafter != nullptr && std::string(drafter) == "dflash2";
}

inline void use_drafter(SpeculativeOptions& speculative) {
    if (!dflash2_drafter()) {
        speculative.backend = SpeculativeBackend::DSpark;
        return;
    }
    const char* companion = std::getenv("INFERNIX_INKLING2_DFLASH2");
    if (companion == nullptr || *companion == '\0') {
        throw std::runtime_error("INFERNIX_INKLING2_DRAFTER=dflash2 needs INFERNIX_INKLING2_DFLASH2 (the companion)");
    }
    speculative.backend     = SpeculativeBackend::DFlash2;
    speculative.draft_model = companion;
}

} // namespace infernix::test::inkling2
