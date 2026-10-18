#include "core/decode_graph.h"

#include "core/device.h"
#include "core/nvtx.h"

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace infernix {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_graph_exec(cudaGraphExec_t& exec) noexcept {
    if (exec != nullptr) {
        log_cuda_error("cudaGraphExecDestroy", cudaGraphExecDestroy(exec));
        exec = nullptr;
    }
}

void destroy_graph(cudaGraph_t& graph) noexcept {
    if (graph != nullptr) {
        log_cuda_error("cudaGraphDestroy", cudaGraphDestroy(graph));
        graph = nullptr;
    }
}

void discard_capture(cudaStream_t stream) noexcept {
    cudaGraph_t discard = nullptr;
    log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &discard));
    destroy_graph(discard);
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_) {
    other.graph_ = nullptr;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    graph_ = other.graph_;

    other.graph_ = nullptr;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    reset();

    // Thread-local mode: only this thread's capture-unsafe calls (synchronous copies, allocation,
    // event queries on the captured stream) invalidate the capture. Other threads of the process
    // (a Program's SSD read agent, VRAM monitor and n-gram volume reader) are not checked, so they
    // must never use the captured stream or its events; none does.
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));

    try {
        body();
    } catch (...) {
        discard_capture(stream);
        throw;
    }

    cudaGraph_t graph = nullptr;

    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess) {
        destroy_graph(graph);
        CUDA_CHECK(err);
    }

    graph_ = graph;
}

void DecodeGraphDefinition::capture_gated_chain(
    cudaStream_t stream, std::int32_t steps, const std::function<void(std::int32_t)>& body,
    const std::function<void(std::int32_t, cudaGraphConditionalHandle)>& gate) {
    nvtx::ScopedRange capture_range(nvtx::Name::CudaGraphCapture, nvtx::Category::Graph);
    if (steps < 1) { throw std::invalid_argument("a gated graph chain needs at least one step"); }
    reset();

    cudaGraph_t chain = nullptr;
    CUDA_CHECK(cudaGraphCreate(&chain, 0));
    // Body j is captured into `target`: the chain itself for j = 0, else the IF body that the gate
    // after body j - 1 controls (owned by its conditional node, so by the chain).
    cudaGraph_t target = chain;
    try {
        for (std::int32_t j = 0; j < steps; ++j) {
            CUDA_CHECK(cudaStreamBeginCaptureToGraph(stream, target, nullptr, nullptr, 0,
                                                     cudaStreamCaptureModeThreadLocal));
            cudaGraph_t next = nullptr;
            try {
                body(j);
                if (j + 1 < steps) {
                    cudaGraphConditionalHandle handle{};
                    CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, target, 0, cudaGraphCondAssignDefault));
                    gate(j, handle);
                    cudaStreamCaptureStatus status   = cudaStreamCaptureStatusNone;
                    cudaGraph_t captured             = nullptr;
                    const cudaGraphNode_t* leaves    = nullptr;
                    const cudaGraphEdgeData* edges   = nullptr;
                    std::size_t count                = 0;
                    CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &captured, &leaves, &edges, &count));
                    cudaGraphNodeParams params{};
                    params.type               = cudaGraphNodeTypeConditional;
                    params.conditional.handle = handle;
                    params.conditional.type   = cudaGraphCondTypeIf;
                    params.conditional.size   = 1;
                    cudaGraphNode_t node      = nullptr;
                    CUDA_CHECK(cudaGraphAddNode(&node, captured, leaves, edges, count, &params));
                    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream, &node, nullptr, 1,
                                                                   cudaStreamSetCaptureDependencies));
                    next = params.conditional.phGraph_out[0];
                }
            } catch (...) {
                // The capture's graph is `target`, which the chain owns: end it without destroying it.
                cudaGraph_t ended = nullptr;
                log_cuda_error("cudaStreamEndCapture(discard)", cudaStreamEndCapture(stream, &ended));
                throw;
            }
            cudaGraph_t captured = nullptr;
            CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
            target = next;
        }
    } catch (...) {
        destroy_graph(chain);
        throw;
    }
    graph_ = chain;
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

void DecodeGraphDefinition::reset() noexcept { destroy_graph(graph_); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_) {
    other.exec_ = nullptr;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    reset();
    exec_       = other.exec_;
    other.exec_ = nullptr;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange instantiate_range(nvtx::Name::CudaGraphInstantiate, nvtx::Category::Graph);
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    reset();

    cudaGraphExec_t exec  = nullptr;
    const cudaError_t err = cudaGraphInstantiate(&exec, definition.graph_, 0);
    if (err != cudaSuccess) {
        destroy_graph_exec(exec);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            throw GraphMemoryExhausted("CUDA Graph instantiation: out of device memory");
        }
        CUDA_CHECK(err);
    }
    exec_ = exec;
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    nvtx::ScopedRange update_range(nvtx::Name::CudaGraphUpdate, nvtx::Category::Graph);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
            " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    nvtx::ScopedRange upload_range(nvtx::Name::CudaGraphUpload, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    const cudaError_t err = cudaGraphUpload(exec_, stream);
    if (err == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        throw GraphMemoryExhausted("CUDA Graph upload: out of device memory");
    }
    CUDA_CHECK(err);
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    // This range executes for every replay; ranges in the captured body execute only at capture.
    nvtx::ScopedRange launch_range(nvtx::Name::CudaGraphLaunch, nvtx::Category::Graph);
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

void DecodeGraphExecutable::reset() noexcept { destroy_graph_exec(exec_); }

} // namespace infernix
