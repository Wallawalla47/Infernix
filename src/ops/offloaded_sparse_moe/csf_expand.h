#pragma once

// Host interface of the GPU's in-place expansion of CSF records (csf_record.h) after copy-engine landings: the
// prefill stream's ring and frame promotions DMA a compressed record's bytes [0, table[rgs]) into its destination,
// then expand_in_place restores the expanded record there.

#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::offloaded_moe::csf {

// One record: `record` (device) holds the record's compressed bytes and receives its expanded record; `table` is
// the record's table where the device reads it (the bank's mapped side array).
struct InPlaceJob {
    std::uint8_t* record       = nullptr;
    const std::uint32_t* table = nullptr;
};

// The most records one expand_in_place launch takes (its kernel parameter, within the classic 4 KiB).
inline constexpr int kMaxInPlaceJobs = 128;
struct InPlaceJobs {
    int count = 0;
    InPlaceJob job[kMaxInPlaceJobs];
};

// Expands every job's record in place on `stream`, one CTA a record, chunks in descending order (csf_record.h: no
// region starts after its expanded offset). The compressed bytes must have landed in stream order.
void expand_in_place(const ExpertShape& shape, const InPlaceJobs& jobs, cudaStream_t stream);

} // namespace infernix::ops::offloaded_moe::csf
