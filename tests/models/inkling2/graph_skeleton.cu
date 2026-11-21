// A captured round's launch skeleton for the forward test's decode attribution (INFERNIX_INKLING2_SKELETON): every
// kernel node of a graph replaced by an empty kernel of the same grid, block and dynamic shared memory, the edges
// (programmatic ones included) unchanged, so replaying it times the chain of launches and dependency waits alone.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Waits for its programmatic dependency (a no-op without one) and exits: the dependent of every node launches when
// it completes, as when a kernel triggers at its end (the chain fully serialized: an upper bound).
__global__ void empty_kernel() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
// The same, triggering its dependents at entry (as short kernels do): the chain's launches overlap (a lower bound).
__global__ void empty_trigger_kernel() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) { throw std::runtime_error(std::string("graph skeleton: ") + what + ": " + cudaGetErrorString(error)); }
}

} // namespace

// Every kernel node's (mangled) name and node count, in first-seen order: the names the filters below match.
void print_graph_kernels(cudaGraph_t graph) {
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count), "nodes");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count), "nodes");
    std::vector<std::pair<std::string, int>> names;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        check(cudaGraphNodeGetType(node, &type), "type");
        if (type != cudaGraphNodeTypeKernel) { continue; }
        cudaKernelNodeParams params{};
        check(cudaGraphKernelNodeGetParams(node, &params), "params");
        const char* name = nullptr;
        check(cudaFuncGetName(&name, params.func), "name");
        const std::string key = name != nullptr ? name : "?";
        auto it = names.begin();
        while (it != names.end() && it->first != key) { ++it; }
        if (it == names.end()) {
            names.emplace_back(key, 1);
        } else {
            ++it->second;
        }
    }
    for (const auto& [name, nodes_of] : names) { std::printf("kernel %4d x %s\n", nodes_of, name.c_str()); }
}

// `filter`: only kernels whose name contains one of its '|'-separated parts (all kernels when empty).
int skeletonize_graph(cudaGraph_t graph, bool trigger, const std::string& filter) {
    static const bool configured = [] {
        check(cudaFuncSetAttribute(empty_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024), "attribute");
        check(cudaFuncSetAttribute(empty_trigger_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024),
              "attribute");
        return true;
    }();
    (void)configured;
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count), "nodes");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count), "nodes");
    int replaced = 0;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        check(cudaGraphNodeGetType(node, &type), "type");
        if (type != cudaGraphNodeTypeKernel) { continue; }
        cudaKernelNodeParams params{};
        check(cudaGraphKernelNodeGetParams(node, &params), "params");
        if (!filter.empty()) {
            const char* name = nullptr;
            check(cudaFuncGetName(&name, params.func), "name");
            bool match = false;
            for (std::size_t from = 0; from <= filter.size() && !match;) {
                const std::size_t bar  = filter.find('|', from);
                const std::string part = filter.substr(from, bar == std::string::npos ? std::string::npos : bar - from);
                match                  = !part.empty() && name != nullptr && std::string(name).find(part) != std::string::npos;
                if (bar == std::string::npos) { break; }
                from = bar + 1;
            }
            if (!match) { continue; }
        }
        params.func         = trigger ? reinterpret_cast<void*>(empty_trigger_kernel) : reinterpret_cast<void*>(empty_kernel);
        params.kernelParams = nullptr;
        params.extra        = nullptr;
        check(cudaGraphKernelNodeSetParams(node, &params), "set params");
        ++replaced;
    }
    return replaced;
}
