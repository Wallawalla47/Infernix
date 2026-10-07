#pragma once
#include "ops/softmax_attention/common/causal_operands.h"

namespace infernix::ops::detail {
template <bool Writable>
using Nvfp4KvCacheView =
    QuantizedCausalCacheView<std::uint8_t, std::uint8_t, std::uint8_t, Writable>;
using Nvfp4KvReadView = Nvfp4KvCacheView<false>;

} // namespace infernix::ops::detail
