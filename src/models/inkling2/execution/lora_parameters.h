#pragma once

// --lora (docs/lora.md): the device operands of an Inkling 2 adapter, prepared once from the Model's binding. One
// DeviceBuffer holds every factor, 256-byte aligned per region. The fast routes (default, --q8-weights, --a8-activations,
// --reproducible) use the BF16 form, the lossless route the exact form; a Program runs one of them.

#include "core/arena.h"
#include "models/inkling2/model.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace infernix::models::inkling2 {

// One adapted projection on the fast routes: A BF16 [rank][K] and B BF16 [rows rounded up to 16][rank], the padded rows
// and columns zero, `rank` its term group's (LoraBinding::fast_layout).
struct LoraTerm {
    const std::uint16_t* a = nullptr;
    const std::uint16_t* b = nullptr;
    std::int32_t rows      = 0; // the projection's output rows
    float scale            = 0.0F;
    std::int32_t rank      = 0;
    [[nodiscard]] bool present() const noexcept { return b != nullptr; }
};

// One adapted projection on the lossless route: A's three exact BF16 planes [rank][3 K] (ops::lossless_lora_split3) and
// B FP32 [rows][rank].
struct LosslessLoraTerm {
    const void* a3   = nullptr;
    const float* b   = nullptr;
    std::int32_t rows = 0;
    float scale       = 0.0F;
    [[nodiscard]] bool present() const noexcept { return b != nullptr; }
};

// One message gate's projections (q, k, v, mlp).
template <class Term>
struct LoraGateTerms {
    Term down, forget, message;
    [[nodiscard]] bool any() const noexcept { return down.present() || forget.present() || message.present(); }
    [[nodiscard]] bool outputs() const noexcept { return forget.present() || message.present(); }
};

// One layer's adapted projections: the K2 outputs (query, relative, gate and sink read x_q; key and key_taps x_k; value
// and value_taps x_v), W_o, and the message gates.
template <class Term>
struct LoraLayerTerms {
    Term query, relative, gate, sink, key, value, key_taps, value_taps;
    Term output;
    std::array<LoraGateTerms<Term>, 4> gates;

    [[nodiscard]] bool attention_input() const noexcept {
        return query.present() || relative.present() || gate.present() || sink.present() || key.present() ||
               value.present() || key_taps.present() || value_taps.present();
    }
    [[nodiscard]] bool any() const noexcept {
        return attention_input() || output.present() || gates[0].any() || gates[1].any() || gates[2].any() ||
               gates[3].any();
    }
};

struct LoraParameters {
    DeviceBuffer factors;
    std::int32_t rank = 0;  // the adapter's largest padded rank (the lossless route's, and the bound of U's workspace)
    bool lossless     = false;
    std::vector<LoraLayerTerms<LoraTerm>> layers;                  // the fast routes: [layers], else empty
    std::vector<LoraLayerTerms<LosslessLoraTerm>> lossless_layers; // the lossless route: [layers], else empty
};

// The device factors of `binding` (nullopt without one), in the route's form. Reads one module's factors at a time.
[[nodiscard]] std::optional<LoraParameters> prepare_lora(const LoraBinding* binding, const TextConfig& text,
                                                         bool lossless);

} // namespace infernix::models::inkling2
