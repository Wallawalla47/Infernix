#pragma once

#include "ops/common/bf16_vector.cuh"
#include "core/pdl.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"
#include "ops/linear_attention/gated_delta_net/launch.h"
#include "infernix/ops/speculative_tree.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail::gated_delta_net {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const float* base,
                                             std::uint32_t dqk_base) {
    store_vec(reg, load_vec<float4>(base + dqk_base));
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], float* base,
                                              std::uint32_t dqk_base) {
    store_vec(base + dqk_base, load_vec<float4>(reg));
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawValueLane {
    __nv_bfloat16 bits;
    float value;
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo = bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(float (&value)[kQkPerLane], int lane) {
    if constexpr (Normalize) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += value[i] * value[i]; }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { value[i] *= inv; }
    }
}

__device__ __forceinline__ RawValueLane load_value_lane(const __nv_bfloat16* base, int lane,
                                                        std::uint32_t dv_base) {
    RawValueLane out{__float2bfloat16(0.0f), 0.0f};
    if (lane < kDvPerWarp) {
        out.bits  = base[dv_base + lane];
        out.value = __bfloat162float(out.bits);
    }
    return out;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    const float g_value    = g[offset];
    const float beta_value = beta[offset];
    return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
}

__device__ __forceinline__ RawGatePair load_record_gate(const uint2* gate, std::int64_t offset) {
    const uint2 bits = load_vec<uint2>(gate + offset);
    return {bits, __uint_as_float(bits.x), __uint_as_float(bits.y)};
}

__device__ __forceinline__ void apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane],
                                                     const float (&key)[kQkPerLane], float v_local,
                                                     float g, float beta) {
    const float alpha = expf(g);

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * key[c]; }
        partial = warp_sum<kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
        const float delta = beta * (v_r - alpha * partial);

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    RawQkLane q = load_raw_qk_lane(query, dqk_base);
    normalize_qk_lane<Normalize>(q.value, lane);

    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * q.value[c]; }
        partial = warp_sum<kWarpSize>(partial);
        if (lane == r) { attn_val = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
}

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 head_map heads) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

struct OutputEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access&, const RecurrentCoordinates&,
                                                       std::int32_t, const RawQkLane&) {}

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access&, const RecurrentCoordinates&, std::int32_t,
                       const RawValueLane&, const RawGatePair&) {}

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                           access.output_ptr(coord, token), coord.dqk_base,
                                           coord.dv_base, coord.lane, access.scale);
    }
};

struct RecordEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access& access,
                                                       const RecurrentCoordinates& coord,
                                                       std::int32_t token, const RawQkLane& key) {
        access.store_key(coord, token, key);
    }

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access& access, const RecurrentCoordinates& coord, std::int32_t token,
                       const RawValueLane& value, const RawGatePair& gate) {
        access.store_value(coord, token, value);
        access.store_gate(coord, token, gate);
    }

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                   const RecurrentCoordinates& coord, std::int32_t token) {
        OutputEffects::publish_output<NormalizeInputs>(state, access, coord, token);
    }
};

struct FoldEffects {
    template <class Access>
    __device__ __forceinline__ static void observe_key(const Access&, const RecurrentCoordinates&,
                                                       std::int32_t, const RawQkLane&) {}

    template <class Access>
    __device__ __forceinline__ static void
    observe_value_gate(const Access&, const RecurrentCoordinates&, std::int32_t,
                       const RawValueLane&, const RawGatePair&) {}

    template <bool NormalizeInputs, class Access>
    __device__ __forceinline__ static void
    publish_output(float (&)[kDvPerWarp][kQkPerLane], const Access&, const RecurrentCoordinates&,
                   std::int32_t) {}
};

struct DirectAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const float* state_read;
    float* state_write;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(0, 0, static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)coord;
        return token;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return state_read + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ float* state_write_base(const RecurrentCoordinates& coord) const {
        return state_write + static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

struct BatchUpdateAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const float* states_read;
    float* states_write;
    const std::int32_t* source_state_slots;
    const std::int32_t* destination_state_slots;
    __nv_bfloat16* out;
    head_map heads;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        (void)token;
        return coord.batch;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states_read +
               static_cast<std::int64_t>(source_state_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ float* state_write_base(const RecurrentCoordinates& coord) const {
        return states_write +
               static_cast<std::int64_t>(destination_state_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }
};

template <bool Masked>
struct RecordAccess {
    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const float* states;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void store_key(const RecurrentCoordinates& coord, std::int32_t token,
                                              const RawQkLane& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 &&
            static_cast<int>(coord.value_head) % heads.group_size() == 0) {
            __nv_bfloat16* destination =
                key_record + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
            store_vec(destination + coord.dqk_base, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value(const RecurrentCoordinates& coord,
                                                std::int32_t token, const RawValueLane& raw) const {
        if (coord.lane < kDvPerWarp) {
            __nv_bfloat16* destination =
                value_record + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
            destination[coord.dv_base + coord.lane] = raw.bits;
        }
    }

    __device__ __forceinline__ void store_gate(const RecurrentCoordinates& coord,
                                               std::int32_t token, const RawGatePair& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 && coord.lane == 0) {
            gate_record[column(coord, token) * heads.H_v + coord.value_head] = raw.bits;
        }
    }
};

template <int Layers, int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr int kLayers       = Layers;
    static constexpr int kQkHeads      = QkHeads;
    static constexpr int kValueHeads   = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
    static_assert(ValueHeads % QkHeads == 0);
    static_assert(ConvChannels % 128 == 0);
};

using FoldGeometry48x48 = FoldGeometry<48, 16, 48, 10240>;
using FoldGeometry30x32 = FoldGeometry<30, 16, 32, 8192>;
// Qwen3.8-Flash-Next: 36 GDN layers with the 48x48 head and channel geometry.
using FoldGeometry36x48 = FoldGeometry<36, 16, 48, 10240>;

template <class Geometry>
struct FoldAccess {
    const __nv_bfloat16* key_record;
    const __nv_bfloat16* value_record;
    const uint2* gate_record;
    const __nv_bfloat16* conv_record;
    float* recurrent_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t record_capacity;
    std::int32_t width;
    GdnReplayFoldKernelRows rows;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch       = static_cast<std::int32_t>(blockIdx.y);
        const std::int32_t layer_tile  = static_cast<std::int32_t>(blockIdx.z);
        const int lane                 = threadIdx.x;
        const int warp                 = threadIdx.y;
        const std::int32_t state_tile  = layer_tile & 7;
        const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
        constexpr std::uint32_t kGroup = Geometry::kValueHeads / Geometry::kQkHeads;
        const std::uint32_t qk_head    = value_head / kGroup;
        const std::uint32_t dv_base =
            static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
        return {lane,
                warp,
                batch,
                layer_tile >> 3,
                state_tile,
                value_head,
                qk_head,
                dv_base,
                static_cast<std::uint32_t>(lane * kQkPerLane)};
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        return rows.row[coord.batch].commit_columns;
    }

    __device__ __forceinline__ std::int64_t record_outer(const RecurrentCoordinates& coord) const {
        return static_cast<std::int64_t>(coord.layer) * record_capacity + coord.batch;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].source_state_slot) * slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ float* state_write_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].destination_state_slot) *
                   slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return key_record + (column * Geometry::kQkHeads + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return value_record + (column * Geometry::kValueHeads + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + token;
        return load_record_gate(gate_record, column * Geometry::kValueHeads + coord.value_head);
    }

    __device__ __forceinline__ void
    store_final_state(const RecurrentCoordinates& coord,
                      const float (&state)[kDvPerWarp][kQkPerLane]) const {
        float* destination = state_write_base(coord);
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(state[r],
                          destination + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                          coord.dqk_base);
        }
    }

    __device__ __forceinline__ void publish_final_conv_history(const RecurrentCoordinates& coord,
                                                               std::int32_t commit) const {
        const std::int32_t tile_block =
            static_cast<std::int32_t>(coord.value_head) * 8 + coord.state_tile;
        if (tile_block >= Geometry::kConvChannels / 128) { return; }

        const std::int32_t tid     = coord.warp * kWarpSize + coord.lane;
        const std::int32_t channel = tile_block * 128 + tid;
        const __nv_bfloat16* source_history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].source_state_slot) *
                (3LL * Geometry::kConvChannels) +
            channel;
        __nv_bfloat16* destination_history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].destination_state_slot) *
                (3LL * Geometry::kConvChannels) +
            channel;
        const __nv_bfloat16* record =
            conv_record + record_outer(coord) * width * Geometry::kConvChannels + channel;

        __nv_bfloat16 h0;
        __nv_bfloat16 h1;
        __nv_bfloat16 h2;
        if (commit == 1) {
            h0 = source_history[Geometry::kConvChannels];
            h1 = source_history[2LL * Geometry::kConvChannels];
            h2 = record[0];
        } else if (commit == 2) {
            h0 = source_history[2LL * Geometry::kConvChannels];
            h1 = record[0];
            h2 = record[Geometry::kConvChannels];
        } else {
            h0 = record[static_cast<std::int64_t>(commit - 3) * Geometry::kConvChannels];
            h1 = record[static_cast<std::int64_t>(commit - 2) * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(commit - 1) * Geometry::kConvChannels];
        }
        destination_history[0]                             = h0;
        destination_history[Geometry::kConvChannels]       = h1;
        destination_history[2LL * Geometry::kConvChannels] = h2;
    }
};

__device__ __forceinline__ void load_state_tile(float (&state)[kDvPerWarp][kQkPerLane],
                                                const float* base,
                                                const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                     coord.dqk_base);
    }
}

__device__ __forceinline__ void store_state_tile(const float (&state)[kDvPerWarp][kQkPerLane],
                                                 float* base, const RecurrentCoordinates& coord) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r], base + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                      coord.dqk_base);
    }
}

template <bool NormalizeInputs, class Effects, class Access>
__device__ __forceinline__ void
run_recurrent_sequence(float (&state)[kDvPerWarp][kQkPerLane], const Access& access,
                       const RecurrentCoordinates& coord, std::int32_t valid) {
    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    Effects::observe_key(access, coord, 0, key);
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);

    for (std::int32_t token = 0; token < valid; ++token) {
        const RawGatePair gate = access.load_gate(coord, token);
        const RawValueLane value =
            load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);
        Effects::observe_value_gate(access, coord, token, value, gate);

        apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);

        if (token + 1 < valid) {
            key = load_raw_qk_lane(access.key_ptr(coord, token + 1), coord.dqk_base);
            Effects::observe_key(access, coord, token + 1, key);
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
        }

        Effects::template publish_output<NormalizeInputs>(state, access, coord, token);
    }
}

template <class Access>
__device__ __forceinline__ void zero_output_suffix(const Access& access,
                                                   const RecurrentCoordinates& coord,
                                                   std::int32_t valid, std::int32_t width) {
    if (coord.lane < kDvPerWarp) {
        for (std::int32_t token = valid; token < width; ++token) {
            access.output_ptr(coord, token)[coord.dv_base + coord.lane] = __float2bfloat16(0.0f);
        }
    }
}

template <bool NormalizeInputs>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_direct_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 const float* __restrict__ state_read,
                                 float* __restrict__ state_write, __nv_bfloat16* __restrict__ out,
                                 std::int32_t width, head_map heads, float scale) {
    const DirectAccess access{q, k, v, g, beta, state_read, state_write, out, heads, width, scale};
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<NormalizeInputs, OutputEffects>(state, access, coord, width);
    store_state_tile(state, access.state_write_base(coord), coord);
}

template <bool NormalizeInputs>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_batch_update_kernel(BatchUpdateAccess access) {
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<NormalizeInputs, OutputEffects>(state, access, coord, 1);
    store_state_tile(state, access.state_write_base(coord), coord);
}

template <bool Masked>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_record_kernel(RecordAccess<Masked> access) {
    const RecurrentCoordinates coord = access.coordinates();
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    // Verification leaves the persistent state unchanged: it was last written by the preceding
    // round's fold, which completes before this round's graph starts, and its slot comes from the
    // round's ingress. The tile therefore loads before waiting on the projection that feeds q/k/v.
    load_state_tile(state, access.state_read_base(coord), coord);
    pdl::enter_streaming();
    const std::int32_t valid = access.active_columns(coord);
    run_recurrent_sequence<true, RecordEffects>(state, access, coord, valid);
    zero_output_suffix(access, coord, valid, access.width);
}

// One step of a verification-tree walk: the column it computes, where its starting state comes
// from (the state the previous step left, a shared-memory slot, or the state after its parent
// rebuilt from the initial state) and the slot that keeps its state for later children.
struct TreeWalkStep {
    std::int8_t column;
    std::int8_t restore;
    std::int8_t save;
    std::int8_t unused;
};

inline constexpr std::int8_t kTreeWalkContinue = -1;
inline constexpr std::int8_t kTreeWalkReplay   = -2;
inline constexpr std::int8_t kTreeWalkNoSlot   = -1;

// Orders a tree row's columns depth first, every node's later-drawn children before its first
// child (a main column's main-chain child), so a branch node holds a slot only while its side
// subtrees run: a main column's slot is released when the walk returns to the main chain, and a
// row with s side columns nests at most (s+1)/2 slots. A branch node beyond the Slots free slots
// is rebuilt from the initial state when the walk returns to it. Returns the step count (nodes).
template <int Slots>
__device__ int build_tree_walk(const SpeculativeTreeRow& row, TreeWalkStep* steps) {
    struct Pending {
        int node;
        int next;
        int slot;
    };

    // A pending node has an unvisited child, so it is an inner node with at least two children:
    // a row has at most leaves-1 <= kSpeculativeTreeMaxPaths-1 of them.
    Pending stack[kSpeculativeTreeMaxPaths];
    int depth           = 0;
    int count           = 0;
    unsigned free_slots = (1U << Slots) - 1U;
    int node            = 0;
    std::int8_t restore = kTreeWalkContinue;
    while (true) {
        const int first  = row.first_child[node];
        const int second = first >= 0 ? row.next_sibling[first] : -1;
        TreeWalkStep step{static_cast<std::int8_t>(node), restore, kTreeWalkNoSlot, 0};
        restore = kTreeWalkContinue;
        if (second >= 0) {
            int slot = kTreeWalkNoSlot;
            if (free_slots != 0U) {
                slot = __ffs(static_cast<int>(free_slots)) - 1;
                free_slots &= free_slots - 1U;
            }
            step.save       = static_cast<std::int8_t>(slot);
            steps[count++]  = step;
            const int after = row.next_sibling[second];
            stack[depth++]  = {node, after >= 0 ? after : first, slot};
            node            = second;
            continue;
        }
        steps[count++] = step;
        if (first >= 0) {
            node = first;
            continue;
        }
        if (depth == 0) { return count; }
        Pending& top = stack[depth - 1];
        node         = top.next;
        restore      = top.slot >= 0 ? static_cast<std::int8_t>(top.slot) : kTreeWalkReplay;
        if (node == row.first_child[top.node]) {
            // The last child: the slot is read by this step before any later step saves into it.
            if (top.slot >= 0) { free_slots |= 1U << top.slot; }
            --depth;
        } else {
            const int after = row.next_sibling[node];
            top.next        = after >= 0 ? after : row.first_child[top.node];
        }
    }
}

template <int Slots>
struct TreeWalkShared {
    TreeWalkStep steps[kSpeculativeTreeMaxNodes];
    int count;
    // Slot s holds every thread's state tile, [s][r][thread] for row r of its tile.
    float4 saved[Slots][kDvPerWarp][kWarpSize * kNumWarps];
};

__device__ __forceinline__ void save_state_tile(const float (&state)[kDvPerWarp][kQkPerLane],
                                                float4 (&slot)[kDvPerWarp][kWarpSize * kNumWarps],
                                                int thread) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        slot[r][thread] = make_float4(state[r][0], state[r][1], state[r][2], state[r][3]);
    }
}

__device__ __forceinline__ void
restore_state_tile(float (&state)[kDvPerWarp][kQkPerLane],
                   const float4 (&slot)[kDvPerWarp][kWarpSize * kNumWarps], int thread) {
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        const float4 value = slot[r][thread];
        state[r][0]        = value.x;
        state[r][1]        = value.y;
        state[r][2]        = value.z;
        state[r][3]        = value.w;
    }
}

// Rebuilds the state after `node` from the initial state along its root path: the same
// transitions the walk applied there, without effects.
__device__ void replay_root_path(float (&state)[kDvPerWarp][kQkPerLane],
                                 const RecordAccess<true>& access,
                                 const RecurrentCoordinates& coord, const SpeculativeTreeRow& row,
                                 int node) {
    int path[kSpeculativeTreeMaxPathLength];
    int length = 0;
    for (int at = node; at >= 0; at = row.parent[at]) { path[length++] = at; }
    load_state_tile(state, access.state_read_base(coord), coord);
    for (int i = length - 1; i >= 0; --i) {
        RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, path[i]), coord.dqk_base);
        normalize_qk_lane<true>(key.value, coord.lane);
        const RawGatePair gate = access.load_gate(coord, path[i]);
        const RawValueLane value =
            load_value_lane(access.value_ptr(coord, path[i]), coord.lane, coord.dv_base);
        apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);
    }
}

// Verification-tree record: one CTA per (request row, value head, state tile) walks its row's tree
// once (build_tree_walk), so every live column is computed exactly once, by the chain recurrence
// of its root path from the row's initial state, and publishes its output and records. A chain
// row of a tree round walks its valid prefix.
template <int Slots>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 4)
    recurrent_tree_walk_kernel(const __grid_constant__ RecordAccess<true> access,
                               const SpeculativeTreeRow* __restrict__ trees) {
    __shared__ TreeWalkShared<Slots> shared;
    const RecurrentCoordinates coord = access.coordinates();
    const SpeculativeTreeRow& row    = trees[coord.batch];
    const int thread                 = coord.warp * kWarpSize + coord.lane;
    // Tree rows and valid columns come from the round's ingress and tree build, which precede the
    // verification forward, so the walk is ordered before waiting on the projection.
    if (thread == 0) {
        if (row.tree != 0) {
            shared.count = build_tree_walk<Slots>(row, shared.steps);
        } else {
            const std::int32_t valid = access.valid_columns[coord.batch];
            for (int c = 0; c < valid; ++c) {
                shared.steps[c] = {static_cast<std::int8_t>(c), kTreeWalkContinue, kTreeWalkNoSlot,
                                   0};
            }
            shared.count = valid < 0 ? 0 : valid;
        }
    }
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    __syncthreads();
    pdl::enter_streaming();
    const int count = shared.count;
    if (count > 0) {
        TreeWalkStep step = shared.steps[0];
        RawQkLane key     = load_raw_qk_lane(access.key_ptr(coord, step.column), coord.dqk_base);
        RecordEffects::observe_key(access, coord, step.column, key);
        normalize_qk_lane<true>(key.value, coord.lane);
        for (int i = 0; i < count; ++i) {
            const int column = step.column;
            if (step.restore >= 0) {
                restore_state_tile(state, shared.saved[step.restore], thread);
            } else if (step.restore == kTreeWalkReplay) {
                replay_root_path(state, access, coord, row, row.parent[column]);
            }
            const RawGatePair gate = access.load_gate(coord, column);
            const RawValueLane value =
                load_value_lane(access.value_ptr(coord, column), coord.lane, coord.dv_base);
            RecordEffects::observe_value_gate(access, coord, column, value, gate);
            apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);
            if (step.save >= 0) { save_state_tile(state, shared.saved[step.save], thread); }
            if (i + 1 < count) {
                step = shared.steps[i + 1];
                key  = load_raw_qk_lane(access.key_ptr(coord, step.column), coord.dqk_base);
                RecordEffects::observe_key(access, coord, step.column, key);
                normalize_qk_lane<true>(key.value, coord.lane);
            }
            RecordEffects::publish_output<true>(state, access, coord, column);
        }
    }
    const std::int32_t live = row.tree != 0 ? row.nodes : count;
    zero_output_suffix(access, coord, live, access.width);
}

template <class Geometry>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_fold_kernel(const __grid_constant__ FoldAccess<Geometry> access) {
    const RecurrentCoordinates coord = access.coordinates();
    const std::int32_t valid         = access.active_columns(coord);
    if (valid == 0) { return; }
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    run_recurrent_sequence<true, FoldEffects>(state, access, coord, valid);
    access.store_final_state(coord, state);
    access.publish_final_conv_history(coord, valid);
}

} // namespace infernix::ops::detail::gated_delta_net
