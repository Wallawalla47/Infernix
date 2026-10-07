#pragma once
#include "ops/softmax_attention/common/causal_operands.h"

namespace infernix::ops::detail {
template <bool Writable>
using Int8KvCacheView = QuantizedCausalCacheView<std::int8_t, __half, __half, Writable>;
using Int8KvReadView  = Int8KvCacheView<false>;

} // namespace infernix::ops::detail
