#include "models/qwen4_exp/load/ablate.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::qwen4_exp {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("ablate-sidecar: " + what); }

// "8-47" for a contiguous run of cut layers, "8-20, 30" otherwise; "none" when the lane is uncut.
std::string layer_ranges(const artifact::AblateSidecar& sidecar, const artifact::AblateSidecar::Lane lane) {
    std::string out;
    std::uint32_t l = 0;
    while (l < sidecar.n_layers) {
        if (sidecar.alpha_for(l, lane) == 0.0F) {
            ++l;
            continue;
        }
        const std::uint32_t first = l;
        while (l < sidecar.n_layers && sidecar.alpha_for(l, lane) != 0.0F) { ++l; }
        if (!out.empty()) { out += ", "; }
        out += std::to_string(first);
        if (l - 1 != first) { out += "-" + std::to_string(l - 1); }
    }
    return out.empty() ? "none" : out;
}

} // namespace

AblateBinding bind_ablate(const std::filesystem::path& path, const TextConfig& text) {
    AblateBinding binding;
    binding.sidecar = artifact::AblateSidecar::load(path, text.num_hidden_layers, text.hidden_size);
    binding.path    = path;
    binding.enabled = true;
    return binding;
}

void AblateBinding::upload() {
    if (!enabled) { return; }
    const std::size_t floats = static_cast<std::size_t>(sidecar.rows()) * sidecar.hidden_size;
    if (sidecar.d_host.size() != floats) { fail("internal: direction size does not match the geometry"); }
    if (d_device_ == nullptr) {
        if (cudaMalloc(reinterpret_cast<void**>(&d_device_), floats * sizeof(float)) != cudaSuccess) {
            d_device_ = nullptr;
            fail("cudaMalloc for the ablation direction");
        }
    }
    if (cudaMemcpy(d_device_, sidecar.d_host.data(), floats * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        release();
        fail("cudaMemcpy for the ablation direction");
    }
}

void AblateBinding::release() noexcept {
    if (d_device_ != nullptr) {
        cudaFree(d_device_);
        d_device_ = nullptr;
    }
}

std::string AblateBinding::describe() const {
    using artifact::AblateSidecar;
    char text[256];
    std::snprintf(text, sizeof(text), "%s, %s direction, attn lane %u layers (%s), mlp lane %u layers (%s)",
                  sidecar.format_name(), sidecar.per_layer_direction ? "per-layer" : "one shared",
                  sidecar.layers_cut(AblateSidecar::Lane::Attn), layer_ranges(sidecar, AblateSidecar::Lane::Attn).c_str(),
                  sidecar.layers_cut(AblateSidecar::Lane::Mlp), layer_ranges(sidecar, AblateSidecar::Lane::Mlp).c_str());
    std::string line = text;
    if (!sidecar.generation.empty()) { line += "; " + sidecar.generation; }
    return line;
}

AblateBinding::AblateBinding(AblateBinding&& other) noexcept
    : enabled(other.enabled), path(std::move(other.path)), sidecar(std::move(other.sidecar)),
      d_device_(std::exchange(other.d_device_, nullptr)) {
    other.enabled = false;
}

AblateBinding& AblateBinding::operator=(AblateBinding&& other) noexcept {
    if (this != &other) {
        release();
        enabled       = std::exchange(other.enabled, false);
        path          = std::move(other.path);
        sidecar       = std::move(other.sidecar);
        d_device_     = std::exchange(other.d_device_, nullptr);
    }
    return *this;
}

} // namespace infernix::models::qwen4_exp
