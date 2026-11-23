// Performance bench for lora_shrink (U = A x, include/infernix/ops/lora.h) at Qwen3.8-27B's adapter sites: the
// stacked q/k/v A (3 r rows over K 5120), the Gated DeltaNet input's in_proj_qkv / z / a / b (4 r over 5120), gate/up
// (2 r over 5120), o_proj and out_proj (r over 6144) and down_proj (r over 17408), r the padded rank. The GB/s it
// prints is informational (A + x read, U written). The call takes the schedule lora_shrink picks for its shape.
//   ./infernix_lora_shrink_bench [--rank 32] [--tokens 1,16,1024,4096] [--site q/k/v|gdn-in|gate/up|o_proj|down]
#include "infernix/ops/lora.h"
#include "infernix_bench_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::bench;

namespace {

std::vector<int> parse_list(const char* raw) {
    std::vector<int> result;
    const std::string text(raw);
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        const int value       = std::stoi(text.substr(begin, end == std::string::npos ? end : end - begin));
        if (value <= 0) { throw std::invalid_argument("values must be positive"); }
        result.push_back(value);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("the list must not be empty"); }
    return result;
}

void run(const char* site, int rows, int k, int tokens) {
    DeviceBuffer a = make_bf16(static_cast<std::size_t>(rows) * k, 211U, -0.03F, 0.03F);
    DeviceBuffer x = make_bf16(static_cast<std::size_t>(k) * tokens, 223U);
    DeviceBuffer u = make_zeros(static_cast<std::size_t>(rows) * tokens * 2);
    const ops::LoraShrinkWeight weight{a.p, rows, k};
    Tensor tx(x.p, DType::BF16, {k, tokens});
    Tensor tu(u.p, DType::BF16, {rows, tokens});
    const double bytes = 2.0 * (static_cast<double>(rows) * k + static_cast<double>(k) * tokens +
                                static_cast<double>(rows) * tokens);
    const Result r = bench_loop([&](cudaStream_t s) { ops::lora_shrink(tx, weight, tu, s); }, bytes);
    char tag[128];
    std::snprintf(tag, sizeof(tag), "lora_shrink %s [%d,%d] T=%d", site, rows, k, tokens);
    print_result(tag, r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    int rank                = 32;
    std::vector<int> tokens = {1, 2, 8, 16, 64, 256, 1024, 2048, 3072, 4096, 6144, 8192};
    std::string only; // --site: one site (a profiler's target)
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--rank") && i + 1 < argc) {
            rank = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) {
            tokens = parse_list(argv[++i]);
        } else if (!std::strcmp(argv[i], "--site") && i + 1 < argc) {
            only = argv[++i];
        } else {
            std::fprintf(stderr,
                         "usage: %s [--rank R (padded, multiple of 16)] [--tokens T[,T...]] "
                         "[--site q/k/v|gdn-in|gate/up|o_proj|down]\n",
                         argv[0]);
            return 2;
        }
    }
    if (rank <= 0 || rank % 16 != 0 || rank > ops::kLoraMaxRank) {
        std::fprintf(stderr, "--rank must be a multiple of 16 up to %d\n", ops::kLoraMaxRank);
        return 2;
    }
    try {
        const auto site = [&](const char* name, int rows, int k, int t) {
            if (only.empty() || only == name) { run(name, rows, k, t); }
        };
        for (const int t : tokens) {
            site("q/k/v", 3 * rank, 5120, t);
            site("gdn-in", 4 * rank, 5120, t); // in_proj_qkv / z / a / b stacked
            site("gate/up", 2 * rank, 5120, t);
            site("o_proj", rank, 6144, t);
            site("down", rank, 17408, t);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "lora_shrink bench: %s\n", error.what());
        return 1;
    }
    return 0;
}
