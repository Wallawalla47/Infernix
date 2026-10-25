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
// The same gate with a device step limit: `handle` is 1 only when the comparison passes and
// step + 1 < *limit, so a chain gated after each step j runs steps 0..*limit-1 at most.
void launch_graph_gate(const float* value, double threshold, const std::int32_t* limit, std::int32_t step,
                       cudaGraphConditionalHandle handle, cudaStream_t stream);
// A gate over `rows` rows before the step that computes draft `draft`: `handle` is 1 when some row
// has draft < limits[row] and, unless `values` is null, values[row] (its previous draft's
// probability) widened to binary64 is not below *threshold (a device word; NaN passes).
void launch_graph_gate_rows(const float* values, const double* threshold, const std::int32_t* limits,
                            std::int32_t rows, std::int32_t draft, cudaGraphConditionalHandle handle,
                            cudaStream_t stream);

// A binned-product gate (TEMPORARY with INFERNIX_TMP_MTP_CUMULATIVE, candidates §7 D8): each value
// maps to the rate of the first bin whose edge it is below (edges ascending; at or past the last edge,
// or NaN, the last rate), and a chain's steps 0..step multiply their rates in step order.
// graph_gate_product is that product on host and device alike (the same comparisons and binary32
// multiplications in the same order), so a host cut and a device gate agree on every chain.
struct GraphGateBins {
    static constexpr int kEdges = 7;
    float edges[kEdges];
    float rates[kEdges + 1];
};

__host__ __device__ inline float graph_gate_rate(float value, const GraphGateBins& bins) {
    for (int i = 0; i < GraphGateBins::kEdges; ++i) {
        if (value < bins.edges[i]) { return bins.rates[i]; }
    }
    return bins.rates[GraphGateBins::kEdges];
}

// Step i's value at values[i * stride].
__host__ __device__ inline float graph_gate_product(const float* values, std::int32_t stride, std::int32_t step,
                                                    const GraphGateBins& bins) {
    float product = 1.0F;
    for (std::int32_t i = 0; i <= step; ++i) { product = product * graph_gate_rate(values[i * stride], bins); }
    return product;
}

// `handle` is 1 when graph_gate_product(values, stride, step, bins) is not below *threshold (a device
// word), 0 otherwise.
void launch_graph_gate_product(const float* values, std::int32_t stride, std::int32_t step, const GraphGateBins& bins,
                               const float* threshold, cudaGraphConditionalHandle handle, cudaStream_t stream);

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
    // kernels, device copies and sets (no event, host or host-memory copy nodes). `tail`, when
    // given, is captured into the chain's own graph after the whole chain (after the first IF node,
    // which completes with every nested body), so it may hold what a body may not, such as a
    // device-to-host copy of the chain's results.
    void capture_gated_chain(cudaStream_t stream, std::int32_t steps, const std::function<void(std::int32_t)>& body,
                             const std::function<void(std::int32_t, cudaGraphConditionalHandle)>& gate,
                             const std::function<void()>& tail = {});
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
