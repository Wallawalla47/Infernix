#pragma once

#include "infernix/types.h"

#include <memory>

namespace spdlog {
class logger;
}

namespace infernix::product {

// Renders Engine runtime diagnostics as ordinary operational records ("engine | ...") at the level
// the Engine assigned, so --log-level filters them and a pinned terminal panel stays beneath them.
[[nodiscard]] DiagnosticObserver engine_diagnostic_observer(std::shared_ptr<spdlog::logger> logger);

} // namespace infernix::product
