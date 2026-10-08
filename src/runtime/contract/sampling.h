#pragma once

#include "infernix/types.h"

namespace infernix::runtime {

// Resolves one request at the Engine boundary. The registered preset supplies every omitted
// model-owned field; an omitted seed remains deterministic for direct Engine callers.
// `constrained_output` (a content constraint is active) drops the preset's presence and frequency
// penalties: the constrained language repeats its structure by construction (quotes, separators,
// keys), and a penalty on those tokens makes the sampler pick a legal neighbour that changes the
// content instead (a JSON string opened by the `",` token). Penalties the caller sets still apply.
[[nodiscard]] ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                                          SamplingMode mode,
                                                          const SamplingOverrides& overrides,
                                                          bool constrained_output);

} // namespace infernix::runtime
