#pragma once

#include "infernix/types.h"

namespace infernix {

// Installs the process-wide observer for CUDA diagnostics and returns the previous one, so the
// owner can restore it. Empty installs nothing; the bare stderr line is always written.
DiagnosticObserver set_cuda_diagnostic_observer(DiagnosticObserver observer);

} // namespace infernix
