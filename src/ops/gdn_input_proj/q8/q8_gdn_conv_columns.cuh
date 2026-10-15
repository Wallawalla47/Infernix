#pragma once

#include "ops/common/math.cuh"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

// Output map of the single-parent [query,key,value,z] GDN projection with the width-four causal
// convolution and SiLU in its epilogue, for a call whose contraction columns are B independent
// sequences of one position each (gdn_input_proj_conv_snapshot at W = 1). Contraction column b is
// sequence b. Rows [0,10240) are convolution channels: each consumes the FP32 projected value p
// as its current tap, with the three history taps read from the BF16 window in
// initial_state_slots[b], writes BF16 SiLU(conv) to query, key or value column b, and publishes
// the window (history1, history2, bf16(p)) to snapshot_base_slots[b]. Rows [10240,16384) are z,
// stored as bf16(p). The convolution's FMA order is the snapshot kernels' (oldest tap first, from
// zero), so the convolution differs from the composed projection + snapshot route only in that
// the current tap is p rather than bf16(p); the published window and z are those routes' bits.
struct GdnConvSnapshotColumnsOutput {
    const __nv_bfloat16* conv_weight;        // [C,4]: tap j of channel c at j*C+c, oldest first
    __nv_bfloat16* conv_states;              // [C,3,Slots]: history oldest first
    const std::int32_t* initial_state_slots; // [B]
    const std::int32_t* snapshot_base_slots; // [B]
    __nv_bfloat16* query;                    // [2048,B]
    __nv_bfloat16* key;                      // [2048,B]
    __nv_bfloat16* value;                    // [6144,B]
    __nv_bfloat16* z;                        // [6144,B]

    // One fully reduced row over contraction columns [column_begin, column_begin + active). Every
    // history the row needs is loaded before the row publishes any window, so a sequence's
    // snapshot slot may be its own initial slot (the decode configuration). Distinct sequences
    // name distinct slots, and distinct rows touch distinct channels, so no other thread reads a
    // location this one writes.
    template <int Columns>
    __device__ __forceinline__ void store_row(int row, int column_begin,
                                              const float (&projected)[Columns],
                                              int active) const {
        if (row >= kGdnChannels) {
#pragma unroll
            for (int column = 0; column < Columns; ++column) {
                if (column < active) {
                    z[static_cast<std::int64_t>(column_begin + column) * kGdnZRows + row -
                      kGdnChannels] = __float2bfloat16_rn(projected[column]);
                }
            }
            return;
        }

        constexpr std::int64_t kChannels  = kGdnChannels;
        constexpr std::int64_t kSlotWidth = 3 * kChannels;
        const float w0 = __bfloat162float(conv_weight[row]);
        const float w1 = __bfloat162float(conv_weight[kChannels + row]);
        const float w2 = __bfloat162float(conv_weight[2 * kChannels + row]);
        const float w3 = __bfloat162float(conv_weight[3 * kChannels + row]);

        __nv_bfloat16 history[Columns][3];
        std::int32_t snapshot_slot[Columns];
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            if (column < active) {
                const int sequence = column_begin + column;
                const __nv_bfloat16* initial =
                    conv_states + static_cast<std::int64_t>(initial_state_slots[sequence]) * kSlotWidth;
                history[column][0]    = initial[row];
                history[column][1]    = initial[kChannels + row];
                history[column][2]    = initial[2 * kChannels + row];
                snapshot_slot[column] = snapshot_base_slots[sequence];
            }
        }

        __nv_bfloat16* destination;
        std::int32_t destination_rows;
        std::int32_t local_row;
        if (row < kGdnQueryRows) {
            destination      = query;
            destination_rows = kGdnQueryRows;
            local_row        = row;
        } else if (row < kGdnQueryRows + kGdnKeyRows) {
            destination      = key;
            destination_rows = kGdnKeyRows;
            local_row        = row - kGdnQueryRows;
        } else {
            destination      = value;
            destination_rows = kGdnValueRows;
            local_row        = row - kGdnQueryRows - kGdnKeyRows;
        }

#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            if (column < active) {
                const int sequence = column_begin + column;
                const float p      = projected[column];
                float conv         = fmaf(w0, __bfloat162float(history[column][0]), 0.0F);
                conv               = fmaf(w1, __bfloat162float(history[column][1]), conv);
                conv               = fmaf(w2, __bfloat162float(history[column][2]), conv);
                conv               = fmaf(w3, p, conv);
                destination[static_cast<std::int64_t>(sequence) * destination_rows + local_row] =
                    __float2bfloat16_rn(silu(conv));

                __nv_bfloat16* snapshot =
                    conv_states + static_cast<std::int64_t>(snapshot_slot[column]) * kSlotWidth;
                snapshot[row]                 = history[column][1];
                snapshot[kChannels + row]     = history[column][2];
                snapshot[2 * kChannels + row] = __float2bfloat16_rn(p);
            }
        }
    }

    __device__ __forceinline__ void store(int row, int column, float projected) const {
        const float values[1]{projected};
        store_row(row, column, values, 1);
    }
};

// Hands each fully reduced row to its output map's row store (every Q8 decode contraction finishes
// a row in one thread), so the convolution issues all of a row's history loads before its first
// store instead of one dependent round trip per column.
struct GdnConvSnapshotRowEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const { return value; }

    template <class Output, int Columns>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int column_begin,
                                              const float (&projected)[Columns], int active) const {
        output.store_row(row, column_begin, projected, active);
    }
};

} // namespace infernix::ops::detail
