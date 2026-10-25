#pragma once

// INFERNIX_TMP_ACTIVATION_SCAN=<path> (temporary diagnostic, read once per Forward): before each dense
// projection of a prefill call (eager calls only), scan its input the way f16-accumulate-prefill would
// encode it (ops/linear/f16acc/f16acc_plan.h) and, after every prefill call, rewrite <path> with one
// line per layer and projection input: max |x|, the smallest column maximum, the widest column dynamic
// range (column max / column min nonzero), the largest 32-value chunk sum of encoded magnitudes (times
// 448, the E4M3 maximum, it bounds an FP16 chunk partial; FP16's limit is 65,504), and the fractions of
// nonzero values the encoding rounds and flushes. Off (no allocation, no scan) when the variable is
// unset or empty. Run it under the perplexity tool or a CLI prompt on the artifact in question, with or
// without --fast-path f16-accumulate-prefill (the scan reads the BF16 inputs either way).

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/linear/f16acc/f16acc_plan.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace infernix::models::qwen4_exp::execution {

class ActivationScan {
public:
    // The projection inputs a layer scans (a GDN layer the first two, a QSA layer the next two, every
    // layer's shared expert the last two).
    enum class Site : std::uint8_t { GdnInput, GdnOutput, QsaInput, QsaOutput, SharedInput, SharedDown };
    static constexpr std::size_t kSites = 6;

    // The scan the environment asks for over `layers` layers, or null.
    [[nodiscard]] static std::unique_ptr<ActivationScan> from_environment(std::uint32_t layers);

    void scan(const Tensor& x, std::uint32_t layer, Site site, cudaStream_t stream);
    // Waits for `stream` and rewrites the report.
    void write(cudaStream_t stream);

private:
    ActivationScan(std::string path, std::uint32_t layers);

    std::string path_;
    std::uint32_t layers_;
    DeviceBuffer records_;
    std::vector<ops::detail::F16AccumulateScanRecord> host_;
    std::uint64_t calls_ = 0;
};

} // namespace infernix::models::qwen4_exp::execution
