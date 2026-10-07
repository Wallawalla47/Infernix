#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>

namespace infernix::runtime {

// Automatic capacity takes what `available_runtime_bytes` leaves after `automatic_headroom_bytes`;
// explicit capacity ignores the headroom.
[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes,
                                                       std::size_t automatic_headroom_bytes);

} // namespace infernix::runtime
