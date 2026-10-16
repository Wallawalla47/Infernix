#include "infernix/ops/add_bias.h"
#include "infernix_bench_common.h"
#include "ops/common/bf16_vector.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace infernix;
using namespace infernix::bench;

namespace {

// The production route's traffic: one 16-byte pack per thread over the whole tensor.
__global__ void add_bias_payload_control(const uint4* bias, uint4* x, std::uint32_t packs,
                                         std::uint32_t total) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) { return; }
    const uint4 b = bias[index % packs];
    uint4 value   = x[index];
    value.x ^= b.x;
    value.y ^= b.y;
    value.z ^= b.z;
    value.w ^= b.w;
    x[index] = value;
}

void run(int d, int columns, bool control) {
    const std::size_t n = static_cast<std::size_t>(d) * static_cast<std::size_t>(columns);
    DeviceBuffer x      = make_bf16(n, 101U);
    DeviceBuffer bias   = make_bf16(d, 103U);
    SavedBuffer initial(x);
    const auto restore = [&](cudaStream_t stream) { initial.restore(stream); };
    Tensor tx(x.p, DType::BF16, {d, columns});
    Tensor tb(bias.p, DType::BF16, {d});

    const Result result = bench_loop_prepared(
        restore,
        [&](cudaStream_t stream) {
            if (control) {
                constexpr unsigned block   = 256;
                const auto total           = static_cast<std::uint32_t>(n / 8);
                add_bias_payload_control<<<(total + block - 1) / block, block, 0, stream>>>(
                    static_cast<const uint4*>(bias.p), static_cast<uint4*>(x.p),
                    static_cast<std::uint32_t>(d / 8), total);
            } else {
                ops::add_bias(tb, tx, stream);
            }
        },
        static_cast<double>(n) * 4.0);

    char tag[80];
    std::snprintf(tag, sizeof(tag), "%s [%d,%-5d]", control ? "control" : "add_bias", d, columns);
    print_result(tag, result);
}

void run_matrix(bool control) {
    constexpr std::array<int, 5> patches{8, 256, 4096, 49152, 65536};
    constexpr std::array<int, 5> merged{2, 64, 1024, 12288, 16384};
    for (int d : {1152, 3456, 4304}) {
        for (int p : patches) { run(d, p, control); }
    }
    for (int d : {2048, 4608}) {
        for (int v : merged) { run(d, v, control); }
    }
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    int selected_d       = 0;
    int selected_columns = 0;
    bool control         = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            selected_d = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--columns") && i + 1 < argc) {
            selected_columns = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--control")) {
            control = true;
        } else {
            std::fprintf(stderr, "usage: %s [--d D --columns C] [--control]\n", argv[0]);
            return 2;
        }
    }
    if ((selected_d == 0) != (selected_columns == 0) || selected_d < 0 || selected_columns < 0 ||
        (selected_d > 0 && selected_d % 8 != 0)) {
        std::fprintf(stderr, "d and columns must be supplied together; d must be divisible by 8\n");
        return 2;
    }

    if (selected_d > 0) {
        run(selected_d, selected_columns, control);
        return 0;
    }
    run_matrix(control);
    return 0;
}
