#include "core/weight.h"
#include "infernix/ops/dynamic_grouped_conv.h"

#include "infernix_bench_common.h"
#include "ops/dynamic_grouped_conv/q8/q8_dynamic_grouped_conv_add_plan.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace infernix;
using namespace infernix::bench;

namespace {

constexpr std::int32_t kMaximumWidth     = 16;
constexpr std::int32_t kTaps             = 2;
constexpr std::int32_t kSides            = 2;
constexpr std::size_t kDefaultFlushBytes = 256ULL << 20;

// The registered drafters' hidden widths H (output rows, H / 16 groups) and their two input widths C: the attention
// output and the MLP down projections of Qwen3.8-27B (5120), Qwen3.5-122B-A10B (3072) and Qwen3.8-Flash-Next (2560).
std::array<std::int32_t, 2> input_widths(std::int32_t hidden) {
    if (hidden == 5120) { return {4096, 17408}; }
    if (hidden == 3072) { return {4096, 9216}; }
    if (hidden == 2560) { return {4096, 7680}; }
    throw std::invalid_argument("hidden must be 5120, 3072 or 2560");
}

struct Options {
    std::int32_t hidden     = 5120;
    int width               = 0;
    std::int32_t input_rows = 0;
    std::int32_t batch_size = 0;
    int warmup              = 8;
    int repeat              = 40;
    std::size_t flush_bytes = kDefaultFlushBytes;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const auto next = [&](const char* label) -> const char* {
            if (index + 1 >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[++index];
        };
        if (!std::strcmp(argv[index], "--width")) {
            options.width = std::atoi(next("width"));

        } else if (!std::strcmp(argv[index], "--warmup")) {
            options.warmup = std::atoi(next("warmup"));
        } else if (!std::strcmp(argv[index], "--repeat")) {
            options.repeat = std::atoi(next("repeat"));
        } else if (!std::strcmp(argv[index], "--k")) {
            options.input_rows = std::atoi(next("K"));
        } else if (!std::strcmp(argv[index], "--hidden")) {
            options.hidden = std::atoi(next("hidden"));
        } else if (!std::strcmp(argv[index], "--batch")) {
            options.batch_size = std::atoi(next("batch"));
        } else if (!std::strcmp(argv[index], "--flush-mib")) {
            const long mib = std::strtol(next("flush MiB"), nullptr, 10);
            if (mib <= 0) { throw std::invalid_argument("flush MiB must be positive"); }
            options.flush_bytes = static_cast<std::size_t>(mib) << 20;
        } else if (!std::strcmp(argv[index], "--help") || !std::strcmp(argv[index], "-h")) {
            std::printf("usage: %s [--hidden 5120|3072|2560] [--width 2..16] [--k C] [--batch 1..8] [--warmup N] "
                        "[--repeat N] [--flush-mib N]\n"
                        "  C: 4096 or 17408 at hidden 5120, 4096 or 9216 at 3072, 4096 or 7680 at 2560\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argv[index]));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repeat positive");
    }
    const auto widths = input_widths(options.hidden);
    if (options.input_rows != 0 && options.input_rows != widths[0] && options.input_rows != widths[1]) {
        throw std::invalid_argument("K must be one of the hidden width's two registered input widths");
    }
    if ((options.width && (options.width < 2 || options.width > 16)) || options.batch_size < 0 ||
        options.batch_size > 8) {
        throw std::invalid_argument("batch must be in [1,8]");
    }
    return options;
}

void run_profile(std::int32_t input_rows, const Options& options, bench::L2FlushBuffer& flush,
                 cudaStream_t stream) {
    const std::int32_t hidden = options.hidden, groups = options.hidden / 16;
    DeviceBuffer input = make_bf16(static_cast<std::size_t>(input_rows) * kMaximumWidth * 8, 101U);
    DeviceBuffer base  = make_bf16(static_cast<std::size_t>(hidden) * kTaps * kSides, 103U);
    DeviceBuffer delta =
        make_bf16(static_cast<std::size_t>(groups) * kTaps * kMaximumWidth * 8, 105U);
    DeviceBuffer residual = make_bf16(static_cast<std::size_t>(hidden) * kMaximumWidth * 8, 107U);
    SavedBuffer residual_initial(residual);
    const auto restore = [&](cudaStream_t s) { residual_initial.restore(s); };
    PackedQuantizedWeight packed =
        make_row_split_weight(QType::Q8_G32_FP16, hidden, input_rows, input_rows, 501U);
    const std::size_t capacity =
        ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(hidden, input_rows, 2, 16, 1, 8);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    Tensor base_kernel(base.p, DType::BF16, {hidden, kTaps, kSides});

    for (int width = 2; width <= 16; ++width) {
        if (options.width && options.width != width) continue;
        for (std::int32_t batch_size = 1; batch_size <= 8; ++batch_size) {
            if (options.batch_size != 0 && batch_size != options.batch_size) { continue; }
            const std::int32_t cols = width * batch_size;
            Tensor x(input.p, DType::BF16, {input_rows, width, batch_size});
            Tensor finish_delta(delta.p, DType::BF16, {groups, kTaps, width, batch_size});
            Tensor residual_view(residual.p, DType::BF16, {hidden, width, batch_size});
            const double flops = 2.0 * static_cast<double>(hidden) * input_rows * cols;
            const double bytes = static_cast<double>(packed.model_weight_bytes()) +
                                 2.0 * static_cast<double>(input_rows) * cols +
                                 2.0 * static_cast<double>(hidden) * kTaps * kSides +
                                 2.0 * static_cast<double>(groups) * kTaps * cols +
                                 4.0 * static_cast<double>(hidden) * cols;
            const auto measure = [&](const char* route, auto&& launch) {
                TimedGraph graph;
                workspace.reset();
                workspace.reset_peak();
                graph.capture(stream, launch);
                const ColdTiming timing = measure_cold_graph_prepared(
                    restore, graph, flush, stream, options.warmup, options.repeat);
                const double tflops    = flops / timing.median_us / 1.0e6;
                const double bandwidth = bytes / timing.median_us / 1.0e3;
                std::printf("%d,%d,%d,%d,%s,%.3f,%.3f,%.3f,%.2f,%.1f,%zu,%zu\n", input_rows, width,
                            batch_size, cols, route, timing.median_us, timing.min_us, timing.p95_us,
                            tflops, bandwidth, graph.nodes(), workspace.peak_used());
            };
            const char* route = ops::detail::q8_linear_dynamic_grouped_conv_add_route_name(hidden,
                input_rows, width, batch_size);
            measure(route, [&](cudaStream_t capture_stream) {
                ops::linear_dynamic_grouped_conv_add(x, packed.weight, base_kernel, finish_delta,
                                                     residual_view, workspace, capture_stream);
            });
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        const Options options = parse_args(argc, argv);
        bench::L2FlushBuffer flush(options.flush_bytes);
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::printf("C,W,B,T,route,median_us,min_us,p95_us,effective_tflops,effective_gbs,"
                    "graph_nodes,workspace_bytes\n");
        for (const std::int32_t input_rows : input_widths(options.hidden)) {
            if (options.input_rows != 0 && input_rows != options.input_rows) { continue; }
            run_profile(input_rows, options, flush, stream);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_linear_dynamic_grouped_conv_add_bench: %s\n", error.what());
        return 1;
    }
}
