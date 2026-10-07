#pragma once

#include "core/weight_view.h"

namespace infernix::artifact {

struct TensorObject;

// Interpret and validate a requested tensor's complete encoded parent.
[[nodiscard]] WeightGeometry describe_tensor(const TensorObject& object);

} // namespace infernix::artifact
