#pragma once

#include <cstdint>

namespace infernix::ops {

// The CUDA driver reads the host copy of a module's kernel images when the module's first kernel launches, not when the
// context is created. The Ops' module is the large one: about 4.5 GiB of RAM and 2.4 s on the 2026-11-13 build (RTX
// 5090, driver 621.01, CUDA 13.4, lazy module loading, which still loads each function's device code at its first
// use; eager loading took 450 MiB more device memory). load_kernel_images launches an empty kernel of that module on
// the current device so this happens now: a start does it before any RAM plan reads the host's free memory.
void load_kernel_images();

// What a RAM plan made before load_kernel_images (Engine::plan, which launches no kernel) counts for the images: the
// footprint above until they are loaded, then 0.
[[nodiscard]] std::uint64_t kernel_images_pending_bytes() noexcept;

} // namespace infernix::ops
