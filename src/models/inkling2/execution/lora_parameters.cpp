#include "models/inkling2/execution/lora_parameters.h"

#include "infernix/ops/lossless_lora.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::inkling2 {
namespace {

// Round to the nearest BF16, ties to even (torch's float32 -> bfloat16 cast); NaN stays NaN.
std::uint16_t bf16_rn(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040U);
    }
    bits += 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

// A module's two factor regions in the buffer: A at `a`, B at `b` (byte offsets).
struct Regions {
    std::uint64_t a = 0, b = 0;
};

// The lossless route's regions in module order (LoraBinding::device_bytes(true) counts the same).
std::vector<Regions> lossless_layout(const LoraBinding& binding, std::uint64_t& total) {
    const std::uint64_t r = binding.padded_rank;
    std::vector<Regions> out;
    out.reserve(binding.adapter->modules().size());
    std::uint64_t at = 0;
    for (const auto& m : binding.adapter->modules()) {
        Regions regions;
        regions.a = at;
        at += lora_region_bytes(r * m.in * 6);
        regions.b = at;
        at += lora_region_bytes(m.out * r * 4);
        out.push_back(regions);
    }
    total = at;
    return out;
}

// The fast routes' factors of module `index`, uploaded at its group's rank: A's rows then zero rows to it, B's rows with
// zero columns to it and zero rows to a multiple of 16.
LoraTerm upload_fast(const LoraBinding& binding, std::size_t index, const LoraFastModule& regions, DeviceBuffer& factors) {
    const auto& m         = binding.module(index);
    const std::uint64_t r = regions.rank;
    const auto* base      = static_cast<const std::uint8_t*>(factors.p);
    {
        const std::vector<float> a = binding.adapter->file().read_f32(m.a_tensor);
        std::vector<std::uint16_t> stage(r * m.in, 0);
        for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(m.rank) * m.in; ++i) { stage[i] = bf16_rn(a[i]); }
        factors.copy_from_host(stage.data(), stage.size() * 2, regions.a);
    }
    {
        const std::vector<float> b = binding.adapter->file().read_f32(m.b_tensor);
        std::vector<std::uint16_t> stage(lora_padded_rows(m.out) * r, 0);
        for (std::uint64_t row = 0; row < m.out; ++row) {
            for (std::uint32_t j = 0; j < m.rank; ++j) { stage[row * r + j] = bf16_rn(b[row * m.rank + j]); }
        }
        factors.copy_from_host(stage.data(), stage.size() * 2, regions.b);
    }
    LoraTerm term;
    term.a     = reinterpret_cast<const std::uint16_t*>(base + regions.a);
    term.b     = reinterpret_cast<const std::uint16_t*>(base + regions.b);
    term.rows  = static_cast<std::int32_t>(m.out);
    term.scale = m.scale;
    term.rank  = static_cast<std::int32_t>(r);
    return term;
}

// The lossless route's: A's three exact planes [r, 3 in] (zero rows to the padded rank) and B in FP32 [out, r].
LosslessLoraTerm upload_lossless(const LoraBinding& binding, std::size_t index, const Regions& regions,
                                 DeviceBuffer& factors) {
    const auto& m         = binding.module(index);
    const std::uint64_t r = binding.padded_rank;
    const auto* base      = static_cast<const std::uint8_t*>(factors.p);
    const auto refuse     = [&](const char* factor) {
        throw std::invalid_argument("--lora --lossless: " + m.module + "'s " + factor +
                                    " holds a nonzero value below 2^-110, which the lossless route cannot split exactly");
    };
    {
        const std::vector<float> a = binding.adapter->file().read_f32(m.a_tensor);
        std::vector<std::uint16_t> planes;
        if (!ops::lossless_lora_split3(a.data(), m.rank, static_cast<std::int64_t>(m.in), planes)) { refuse("lora_A"); }
        std::vector<std::uint16_t> stage(r * 3 * m.in, 0);
        std::memcpy(stage.data(), planes.data(), planes.size() * 2);
        factors.copy_from_host(stage.data(), stage.size() * 2, regions.a);
    }
    {
        const std::vector<float> b = binding.adapter->file().read_f32(m.b_tensor);
        std::vector<std::uint16_t> check;
        if (!ops::lossless_lora_split3(b.data(), static_cast<std::int64_t>(m.out), m.rank, check)) { refuse("lora_B"); }
        std::vector<float> stage(m.out * r, 0.0F);
        for (std::uint64_t row = 0; row < m.out; ++row) {
            std::memcpy(stage.data() + row * r, b.data() + row * m.rank, static_cast<std::size_t>(m.rank) * 4);
        }
        factors.copy_from_host(stage.data(), stage.size() * 4, regions.b);
    }
    LosslessLoraTerm term;
    term.a3    = base + regions.a;
    term.b     = reinterpret_cast<const float*>(base + regions.b);
    term.rows  = static_cast<std::int32_t>(m.out);
    term.scale = m.scale;
    return term;
}

// Calls place(module slot, term slot) for every projection the two layer structures name.
template <class Terms, class Place>
void for_each_slot(const LoraLayer& layer, Terms& terms, Place&& place) {
    place(layer.query, terms.query);
    place(layer.relative, terms.relative);
    place(layer.gate, terms.gate);
    place(layer.sink, terms.sink);
    place(layer.key, terms.key);
    place(layer.value, terms.value);
    place(layer.key_taps, terms.key_taps);
    place(layer.value_taps, terms.value_taps);
    place(layer.output, terms.output);
    for (std::size_t k = 0; k < layer.gates.size(); ++k) {
        place(layer.gates[k].down, terms.gates[k].down);
        place(layer.gates[k].forget, terms.gates[k].forget);
        place(layer.gates[k].message, terms.gates[k].message);
    }
}

} // namespace

std::optional<LoraParameters> prepare_lora(const LoraBinding* binding, const TextConfig& text, bool lossless) {
    if (binding == nullptr) { return std::nullopt; }
    if (binding->layers.size() != text.num_hidden_layers) {
        throw std::logic_error("--lora: the adapter does not match the model it was bound to");
    }
    LoraParameters out;
    out.rank     = static_cast<std::int32_t>(binding->padded_rank);
    out.lossless = lossless;
    std::uint64_t total = 0;
    std::vector<Regions> regions;
    LoraFastLayout fast;
    if (lossless) {
        regions = lossless_layout(*binding, total);
    } else {
        fast  = binding->fast_layout();
        total = fast.bytes;
    }
    if (total != binding->device_bytes(lossless)) {
        throw std::logic_error("--lora: the factors' layout disagrees with its planned size");
    }
    out.factors = DeviceBuffer(total);
    if (lossless) {
        out.lossless_layers.resize(binding->layers.size());
    } else {
        out.layers.resize(binding->layers.size());
    }
    for (std::size_t l = 0; l < binding->layers.size(); ++l) {
        if (lossless) {
            for_each_slot(binding->layers[l], out.lossless_layers[l],
                          [&](const std::optional<std::size_t>& module, LosslessLoraTerm& term) {
                              if (module) { term = upload_lossless(*binding, *module, regions[*module], out.factors); }
                          });
        } else {
            for_each_slot(binding->layers[l], out.layers[l],
                          [&](const std::optional<std::size_t>& module, LoraTerm& term) {
                              if (module) { term = upload_fast(*binding, *module, fast.modules[*module], out.factors); }
                          });
        }
    }
    return out;
}

} // namespace infernix::models::inkling2
