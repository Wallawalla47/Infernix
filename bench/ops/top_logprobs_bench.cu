// The two top-K selections of ops::top_logprobs (ops/kernel/top_logprobs.cuh) at the vocabulary and the K of
// its callers: the K-pass selection and the radix selection, timed on the same logits (warm, as a head's
// logits are when the selection runs), with their outputs compared bit for bit (the contract: the same rows in
// the same order and the same values).
//
//   infernix_top_logprobs_bench [--k 4,16,20] [--columns 1,8,16,32] [--dtype fp32|bf16|both]
//       [--warmup N] [--repeat N]
#include "core/device.h"
#include "infernix_bench_common.h"
#include "ops/kernel/top_logprobs.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;
using namespace infernix::bench;

namespace {

constexpr std::int32_t kPhysicalRows = 248320;
constexpr std::int32_t kValidRows    = 248077;

std::vector<int> parse_list(std::string_view text) {
    std::vector<int> values;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(',', start);
        const std::string item(text.substr(start, end == std::string_view::npos ? text.size() - start : end - start));
        if (item.empty()) throw std::invalid_argument("empty list item");
        values.push_back(std::stoi(item));
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return values;
}

template <class Logit>
void run(const char* dtype, const DeviceBuffer& logits, int k, int columns, int warmup, int repeat) {
    DeviceBuffer ids_serial(static_cast<std::size_t>(k) * columns * sizeof(std::int32_t));
    DeviceBuffer out_serial(static_cast<std::size_t>(k) * columns * sizeof(float));
    DeviceBuffer ids_select(ids_serial.bytes);
    DeviceBuffer out_select(out_serial.bytes);
    const auto* values = static_cast<const Logit*>(logits.p);
    const auto serial  = [&](cudaStream_t stream) {
        ops::top_logprobs_kernel<ops::kTopLogprobsBlock><<<columns, ops::kTopLogprobsBlock, 0, stream>>>(
            values, static_cast<std::int32_t*>(ids_serial.p), static_cast<float*>(out_serial.p), kValidRows,
            kPhysicalRows, k);
    };
    const auto select = [&](cudaStream_t stream) {
        ops::top_logprobs_select_kernel<ops::kTopLogprobsBlock><<<columns, ops::kTopLogprobsBlock, 0, stream>>>(
            values, static_cast<std::int32_t*>(ids_select.p), static_cast<float*>(out_select.p), kValidRows,
            kPhysicalRows, k);
    };
    const ColdTiming a = measure_launch(serial, nullptr, warmup, repeat);
    const ColdTiming b = measure_launch(select, nullptr, warmup, repeat);
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned char> h1(ids_serial.bytes), h2(ids_serial.bytes), o1(out_serial.bytes), o2(out_serial.bytes);
    read_back(h1.data(), ids_serial.p, h1.size());
    read_back(h2.data(), ids_select.p, h2.size());
    read_back(o1.data(), out_serial.p, o1.size());
    read_back(o2.data(), out_select.p, o2.size());
    const bool same = h1 == h2 && o1 == o2;
    std::printf("%-4s K=%-4d C=%-3d k-pass %9.2f us   radix %9.2f us   radix/k-pass %.3f   %s\n", dtype, k, columns,
                a.median_us, b.median_us, b.median_us / a.median_us, same ? "same bits" : "DIFFERENT");
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        std::vector<int> ks{4, 16, 20}, columns{1, 8, 16, 32};
        std::string dtype = "both";
        int warmup = 5, repeat = 30;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            const auto next = [&]() -> std::string_view {
                if (i + 1 >= argc) throw std::invalid_argument(std::string(arg) + " needs a value");
                return argv[++i];
            };
            if (arg == "--k") ks = parse_list(next());
            else if (arg == "--columns") columns = parse_list(next());
            else if (arg == "--dtype") dtype = std::string(next());
            else if (arg == "--warmup") warmup = std::stoi(std::string(next()));
            else if (arg == "--repeat") repeat = std::stoi(std::string(next()));
            else throw std::invalid_argument("unknown argument " + std::string(arg));
        }
        int widest = 1;
        for (const int c : columns) widest = c > widest ? c : widest;
        for (const int k : ks) {
            if (k < 1 || k > ops::kTopLogprobsSelectMaximum) throw std::invalid_argument("--k out of range");
        }
        const std::size_t count = static_cast<std::size_t>(kPhysicalRows) * widest;
        if (dtype == "fp32" || dtype == "both") {
            const DeviceBuffer logits = make_f32(count, 1234, -12.0F, 12.0F);
            for (const int k : ks)
                for (const int c : columns) run<float>("fp32", logits, k, c, warmup, repeat);
        }
        if (dtype == "bf16" || dtype == "both") {
            const DeviceBuffer logits = make_bf16(count, 1234, -12.0F, 12.0F);
            for (const int k : ks)
                for (const int c : columns) run<__nv_bfloat16>("bf16", logits, k, c, warmup, repeat);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_top_logprobs_bench: %s\n", error.what());
        return 1;
    }
}
