#pragma once

#include <cstdint>

namespace infernix::ops {

// The CUDA driver reads the host copy of a module's kernel images when the module's first kernel launches, not when the
// context is created. The Ops' module is the large one: about 1.5 GiB of RAM and 0.85 s (RTX 5090, driver 621.01, CUDA
// 13.4, lazy module loading, which still loads each function's device code at its first use; eager loading took
// 450 MiB more device memory); a build with CUDA line information (INFERNIX_CUDA_LINEINFO) reads 4.5 GiB in 2.8 s. load_kernel_images launches an empty kernel of that module on
// the current device so this happens now (a start's KernelImagesLoad runs it beside the weight read).
void load_kernel_images();

// What a RAM plan made before load_kernel_images counts for the images (every start's plan: they load after it, beside
// the weight read; Engine::plan launches no kernel): the footprint above until they are loaded, then 0.
[[nodiscard]] std::uint64_t kernel_images_pending_bytes() noexcept;

// A start's load of the kernel images on a thread of its own, from construction to finish() (destruction joins and
// drops any error). A start opens it after its RAM plan, which counted the images as the process's growth
// (kernel_images_pending_bytes), and finishes it once its weights are on the device: the time the driver spends
// reading the Ops module overlaps the weight read instead of standing before it (GitHub #88, RTX 5090, three starts each:
// Qwen3.8-Flash-Next engine ready 30.4 -> 29.5 s, Qwen3.8-27B 15.1 -> 14.2 s). Does nothing when the images are loaded.
class KernelImagesLoad {
public:
    explicit KernelImagesLoad(int device);
    ~KernelImagesLoad();
    KernelImagesLoad(const KernelImagesLoad&)            = delete;
    KernelImagesLoad& operator=(const KernelImagesLoad&) = delete;
    // Joins the thread and rethrows its error; the load's seconds (0 when nothing ran).
    double finish();

private:
    struct State;
    State* state_ = nullptr;
};

} // namespace infernix::ops
