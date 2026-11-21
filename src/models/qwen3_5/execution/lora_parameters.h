#pragma once

#include "core/arena.h"
#include "models/qwen3_5/execution/parameters.h"

#include <cstdint>

namespace infernix::models::qwen3_5::execution {

// --lora: reads the adapter's factors (Model::lora) from its file, rounds them to BF16 and uploads them into one
// allocation (`factors`) laid out for the kernels:
// - per input site, A stacked over the adapters that read it (attention input: q_proj, k_proj, v_proj; MLP input:
//   gate_proj, up_proj), each padded to LoraBinding::padded_rank rows;
// - per parent, B [parent rows, padded rank] in the parent's physical row order: the attention input's single-parent
//   order query, key, gate, value (q_proj's rows split per head into query and gate), gate/up's gate then up.
// It sets the text layers' LoraSiteParameters. A plan-only model (Engine::plan) reads and allocates nothing: the
// return value is then the bytes a start would allocate (0 otherwise).
[[nodiscard]] std::uint64_t prepare_lora(const Model& model, TextParameters& text, DeviceBuffer& factors);

} // namespace infernix::models::qwen3_5::execution
