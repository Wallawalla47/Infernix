#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <stdexcept>

namespace infernix {

// Enqueues on `stream` a one-thread kernel that sets the CUDA Graph conditional `handle` to 0 when
// `*value < threshold` and to 1 otherwise (NaN included), the float widened to binary64 as a host
// comparison of a float with a double does. Only inside a graph that owns `handle`.
void launch_graph_gate(const float* value, double threshold, cudaGraphConditionalHandle handle, cudaStream_t stream);

// A CUDA Graph executable found no device memory to instantiate or upload into. The device stays
// usable (the error is not sticky), so a caller may free memory and retry; every other CUDA error
// of these calls aborts as CUDA_CHECK does.
class GraphMemoryExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    // Captures `steps` bodies as one chain in which body j > 0 runs only while the gates pass:
    // `gate(j, handle)` enqueues on `stream`, after body j, the kernel that sets `handle`
    // (launch_graph_gate), and body j + 1 is captured into the IF node that handle controls, so a
    // gate that clears it skips every later body. Bodies hold only what a conditional body may:
    // kernels, device copies and sets (no event, host or host-memory copy nodes).
    void capture_gated_chain(cudaStream_t stream, std::int32_t steps, const std::function<void(std::int32_t)>& body,
                             const std::function<void(std::int32_t, cudaGraphConditionalHandle)>& gate);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
};

} // namespace infernix
