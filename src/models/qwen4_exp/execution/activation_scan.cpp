#include "models/qwen4_exp/execution/activation_scan.h"

#include "core/device.h"
#include "core/temporary_switches.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace infernix::models::qwen4_exp::execution {
namespace {

constexpr const char* kSiteNames[ActivationScan::kSites] = {"gdn_input",  "gdn_output",   "qsa_input",
                                                            "qsa_output", "shared_input", "shared_down"};

float from_bits(unsigned bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

std::unique_ptr<ActivationScan> ActivationScan::from_environment(std::uint32_t layers) {
    // The path is in the startup log's temporary switches line.
    auto path = temporary_switches::value("INFERNIX_TMP_ACTIVATION_SCAN");
    if (!path) { return nullptr; }
    return std::unique_ptr<ActivationScan>(new ActivationScan(std::move(*path), layers));
}

ActivationScan::ActivationScan(std::string path, std::uint32_t layers)
    : path_(std::move(path)), layers_(layers),
      records_(static_cast<std::size_t>(layers) * kSites * sizeof(ops::detail::F16AccumulateScanRecord)),
      host_(static_cast<std::size_t>(layers) * kSites) {
    records_.copy_from_host(host_.data(), host_.size() * sizeof(ops::detail::F16AccumulateScanRecord));
}

void ActivationScan::scan(const Tensor& x, std::uint32_t layer, Site site, cudaStream_t stream) {
    if (layer >= layers_) { throw std::invalid_argument("activation scan: layer out of range"); }
    auto* records = static_cast<ops::detail::F16AccumulateScanRecord*>(records_.p);
    ops::detail::launch_f16_accumulate_scan(x, records + static_cast<std::size_t>(layer) * kSites +
                                                   static_cast<std::size_t>(site),
                                            stream);
}

void ActivationScan::write(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    records_.copy_to_host(host_.data(), host_.size() * sizeof(ops::detail::F16AccumulateScanRecord));
    ++calls_;
    std::ofstream out(path_, std::ios::trunc);
    out << "# f16-accumulate-prefill activation scan, " << calls_ << " prefill calls\n"
        << "# layer site columns values max_abs min_column_max max_log2_range max_chunk_sum chunk_bound_x448"
           " inexact_fraction flushed_fraction\n";
    for (std::uint32_t layer = 0; layer < layers_; ++layer) {
        for (std::size_t site = 0; site < kSites; ++site) {
            const auto& r = host_[static_cast<std::size_t>(layer) * kSites + site];
            if (r.columns == 0) { continue; }
            const double nonzero = r.nonzero == 0 ? 1.0 : static_cast<double>(r.nonzero);
            const float range    = from_bits(r.max_range_bits);
            char line[512];
            std::snprintf(line, sizeof(line), "%u %s %llu %llu %.6g %.6g %.2f %.6g %.6g %.3e %.3e\n", layer,
                          kSiteNames[site], r.columns, r.values, from_bits(r.max_bits),
                          from_bits(r.min_column_max_bits), range > 0.0F ? std::log2(range) : 0.0,
                          from_bits(r.max_chunk_bits), 448.0 * from_bits(r.max_chunk_bits),
                          static_cast<double>(r.inexact) / nonzero, static_cast<double>(r.flushed) / nonzero);
            out << line;
        }
    }
}

} // namespace infernix::models::qwen4_exp::execution
