#pragma once

#include "infernix/types.h"

#include <atomic>
#include <memory>

namespace infernix::test {

// Whether an Engine's startup prepared its CUDA Graphs (StartupPhase::CudaGraphPrepare completed).
// MemorySummary::cuda_graph_measured_bytes cannot tell: it is the device's free-memory drop across
// the preparation, in the driver's allocation granularity, so graphs that fit in memory the driver
// already held read 0 (the 35B-A3B's DFlash Engine, 2026-10-22).
class GraphPrepareProbe {
public:
    // `options` reporting to this probe (it replaces their startup observer).
    [[nodiscard]] EngineOptions attached(EngineOptions options) const {
        *prepared_                        = false;
        options.startup_observer.callback = [prepared = prepared_](const StartupEvent& event) {
            if (event.phase == StartupPhase::CudaGraphPrepare && event.status == StartupStatus::Complete) {
                *prepared = true;
            }
        };
        return options;
    }
    [[nodiscard]] bool prepared() const noexcept { return *prepared_; }

private:
    std::shared_ptr<std::atomic<bool>> prepared_ = std::make_shared<std::atomic<bool>>(false);
};

} // namespace infernix::test
