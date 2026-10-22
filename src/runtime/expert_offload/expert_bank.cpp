#include "runtime/expert_offload/expert_bank.h"

#include <cmath>
#include <stdexcept>

namespace infernix::runtime::expert_offload {

namespace moe = ops::offloaded_moe;

std::vector<moe::ExpertScales> a16_expert_scales(const ExpertBankPlanes& planes, const std::string& name) {
    if (planes.multipliers == nullptr) { throw std::invalid_argument(name + ": an NVFP4 bank needs its multipliers"); }
    std::vector<moe::ExpertScales> out(planes.experts);
    for (std::uint32_t e = 0; e < planes.experts; ++e) {
        const float* m = planes.multipliers + 3 * static_cast<std::size_t>(e);
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(m[i]) || m[i] <= 0) {
                throw std::invalid_argument(name + ": expert multipliers must be positive finite FP32");
            }
        }
        out[e] = {0.0F, 0.0F, 0.0F, m[0], m[1], m[2]};
    }
    return out;
}

namespace {

const WeightParent& a16_nvfp4_parent(const WeightView& view, const std::string& name) {
    if (view.parts.size() != 1 || !is_complete_weight(view)) {
        throw std::invalid_argument(name + ": an expert bank must be one complete parent");
    }
    const WeightParent& parent = *view.parts.front().parent;
    if (parent.geometry.layout != QuantLayout::ExpertRg16 || parent.geometry.format != QType::NVFP4_MUL) {
        throw std::invalid_argument(name + ": the bank must be nvfp4_mul in nvfp4_expert_rg16_v1");
    }
    return parent;
}

ExpertBank a16_nvfp4_bank(ExpertBankPlanes planes, std::uint32_t experts, std::uint32_t hidden,
                          std::uint32_t intermediate, const std::string& name) {
    ExpertBank bank;
    bank.planes = planes;
    if (bank.planes.experts != experts || bank.planes.hidden != hidden || bank.planes.intermediate != intermediate) {
        throw std::invalid_argument(name + ": expert bank geometry differs from the config");
    }
    if (!moe::registered_shape(bank.shape())) {
        throw std::invalid_argument(name + ": no expert routes are registered at this geometry");
    }
    bank.activation = moe::ExpertActivation::kA16;
    bank.scales     = a16_expert_scales(bank.planes, name);
    return bank;
}

} // namespace

ExpertBank a16_nvfp4_bank(const WeightView& view, std::uint32_t experts, std::uint32_t hidden,
                          std::uint32_t intermediate, const std::string& name) {
    return a16_nvfp4_bank(expert_bank_planes(a16_nvfp4_parent(view, name)), experts, hidden, intermediate, name);
}

ExpertBank a16_nvfp4_streamed_bank(const WeightView& view, const float* multipliers, std::uint32_t experts,
                                   std::uint32_t hidden, std::uint32_t intermediate, const std::string& name) {
    const WeightParent& parent = a16_nvfp4_parent(view, name);
    return a16_nvfp4_bank(expert_bank_layout(parent.geometry, multipliers), experts, hidden, intermediate, name);
}

ExpertBank a16_nvfp4_device_bank(const WeightView& view, const float* multipliers, std::uint32_t experts,
                                 std::uint32_t hidden, std::uint32_t intermediate, const std::string& name) {
    const WeightParent& parent = a16_nvfp4_parent(view, name);
    if (parent.device == nullptr) { throw std::invalid_argument(name + ": a device bank needs device records"); }
    ExpertBankPlanes planes = expert_bank_layout(parent.geometry, multipliers);
    planes.device_records   = parent.device;
    return a16_nvfp4_bank(planes, experts, hidden, intermediate, name);
}

} // namespace infernix::runtime::expert_offload
