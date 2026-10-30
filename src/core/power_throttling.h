#pragma once

// Windows power throttling (EcoQoS) runs a process it considers to be in the background at reduced
// clock speed and steers its threads to efficiency cores: a server whose console is not the
// foreground window, or a job launched without a visible window. An inference engine's host threads
// (the CPU expert workers, the thread that submits CUDA graphs and polls the device) then slow the
// GPU down with them. Measured on an i9-13900K + RTX 5090 with Qwen3.8-Flash-Next, a hidden run
// decoded 37.6 tok/s at 8K context throttled and 98.7 tok/s exempt (2.6x); serving at two and four
// concurrent requests gained 69-81 %. Outputs are unchanged: only where and how fast threads run.

#include <optional>

namespace infernix {

// Opts the calling process out of execution-speed power throttling, once per process. A no-op on
// platforms without it (Linux). Returns the Windows error code when Windows refused the request
// (every call returns the first call's outcome), so the caller can say the process may be
// throttled: a background window measured 66 -> 4 tok/s on a hybrid CPU (Basalt #93).
std::optional<unsigned long> exempt_process_from_power_throttling() noexcept;

} // namespace infernix
