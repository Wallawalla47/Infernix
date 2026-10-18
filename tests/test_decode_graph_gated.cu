// A gated graph chain (DecodeGraphDefinition::capture_gated_chain): body j > 0 runs only while every
// gate before it passed, the gate applies the host's binary64 comparison of a float with a double
// threshold, and bodies may hold programmatic dependent kernels and device copies, as the
// Qwen3.8-Flash-Next draft steps do. Replays with different values take different paths.

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/pdl.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr int kSteps = 5;

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

// A step's two dependent kernels: the first marks the step, the second turns the scripted value it
// was copied into the step's confidence (a device copy between them, as a draft step has).
__global__ void mark_kernel(int* trace, int step) {
    infernix::pdl::enter();
    if (threadIdx.x == 0) { trace[step] += 1; }
}

__global__ void confidence_kernel(const float* staged, float* confidence, int step) {
    infernix::pdl::enter();
    if (threadIdx.x == 0) { confidence[step] = staged[step]; }
}

void enqueue_step(cudaStream_t stream, int* trace, const float* script, float* staged, float* confidence, int step) {
    const infernix::pdl::LaunchConfig launch{dim3(1), dim3(32), 0, stream};
    CUDA_CHECK(infernix::pdl::launch_consumer(launch, mark_kernel, trace, step));
    CUDA_CHECK(cudaMemcpyAsync(staged + step, script + step, sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(infernix::pdl::launch_consumer(launch, confidence_kernel, static_cast<const float*>(staged), confidence, step));
}

// The host's cut: steps run up to and including the first whose value is below the threshold.
std::vector<int> expected_trace(const std::vector<float>& script, double threshold) {
    std::vector<int> out(kSteps, 0);
    for (int j = 0; j < kSteps; ++j) {
        out[j] = 1;
        if (script[j] < threshold) { break; }
    }
    return out;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    try {
        infernix::DeviceContext device(0);
        const cudaStream_t s = device.stream;
        infernix::DeviceArena trace_memory(sizeof(int) * kSteps);
        infernix::DeviceArena floats(sizeof(float) * kSteps * 3);
        auto* trace      = static_cast<int*>(trace_memory.base());
        auto* script     = static_cast<float*>(floats.base());
        float* staged    = script + kSteps;
        float* confidence = script + 2 * kSteps;
        // log(0.5) as --spec-min-p's cut: a float rounding of it lies below the double threshold.
        const double threshold = std::log(0.5);

        infernix::DecodeGraphDefinition definition;
        definition.capture_gated_chain(
            s, kSteps, [&](std::int32_t j) { enqueue_step(s, trace, script, staged, confidence, j); },
            [&](std::int32_t j, cudaGraphConditionalHandle next) {
                infernix::launch_graph_gate(confidence + j, threshold, next, s);
            });
        infernix::DecodeGraphExecutable executable;
        executable.instantiate(definition);

        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::vector<std::vector<float>> scripts{
            {0.0F, 0.0F, 0.0F, 0.0F, 0.0F},                              // every gate passes
            {-5.0F, 0.0F, 0.0F, 0.0F, 0.0F},                             // only the first step
            {0.0F, -1.0F, 0.0F, 0.0F, 0.0F},                             // two steps
            {0.0F, 0.0F, 0.0F, -2.0F, 0.0F},                             // four steps
            {nan, 0.0F, static_cast<float>(threshold), 0.0F, 0.0F},      // NaN passes; the float log(0.5) is below
            {static_cast<float>(threshold) + 1e-6F, 0.0F, 0.0F, 0.0F, -9.0F}, // just above passes
        };
        std::mt19937 rng(17);
        std::uniform_real_distribution<float> values(-1.5F, 0.0F);
        for (int i = 0; i < 64; ++i) {
            std::vector<float> random(kSteps);
            for (float& v : random) { v = values(rng); }
            scripts.push_back(random);
        }
        int failures = 0;
        for (std::size_t i = 0; i < scripts.size(); ++i) {
            CUDA_CHECK(cudaMemcpyAsync(script, scripts[i].data(), sizeof(float) * kSteps, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaMemsetAsync(trace, 0, sizeof(int) * kSteps, s));
            executable.launch(s);
            std::vector<int> got(kSteps);
            CUDA_CHECK(cudaMemcpyAsync(got.data(), trace, sizeof(int) * kSteps, cudaMemcpyDeviceToHost, s));
            device.synchronize();
            if (got != expected_trace(scripts[i], threshold)) {
                std::cerr << "script " << i << ": steps run";
                for (int v : got) { std::cerr << ' ' << v; }
                std::cerr << '\n';
                ++failures;
            }
        }
        std::cout << scripts.size() << " gated chains replayed, " << failures << " mismatching\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "gated decode graph test failed: " << error.what() << '\n';
        return 1;
    }
}
