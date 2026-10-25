#pragma once

// FastPath::GdnTreeClosedForm (candidates §13 PUB-05; TreeWY, arXiv 2608.20961; Bole, 2608.01651):
// a verification tree's GDN record in closed form instead of the walk (recurrent_tree_walk_kernel).
//
// Along column c's root path P(c) = (0, .., c), the transition s_c = a_c s_p + u_c k_c^T (a = exp(g),
// p the parent, the row's initial state s_0 before column 0) unrolls to
//   s_c = exp(G_c) s_0 + sum_{a in P(c)} exp(G_c - G_a) u_a k_a^T,   G_c = sum of g over P(c),
// so each column's delta
//   u_c = beta_c (v_c - exp(G_c) s_0 k_c - sum_{a in P(c), a != c} exp(G_c - G_a) (k_a . k_c) u_a)
// depends only on its ancestors' (a lower-triangular solve in column order, one level per depth), and
// its output is
//   o_c = scale (exp(G_c) s_0 q_c + sum_{a in P(c)} exp(G_c - G_a) (k_a . q_c) u_a).
// The dense work is s_0 k_c and s_0 q_c, each CTA's 16 state rows against every column; every other
// product follows the tree's ancestor pairs, at most kSpeculativeTreeMaxPathLength per tree column
// (a chain row's column has up to 32). The
// walk's W dependent steps become one parallel pass plus one short step per depth level. Everything
// is FP32, in another order than the walk, so outputs match the walk and the FP64 oracle within
// rounding but not bit for bit: a fast path, never the reproducible or lossless route. Records are the
// walk's raw copies, bit for bit.

#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

namespace infernix::ops::detail::gated_delta_net {

// A root path's columns: a tree's at most kSpeculativeTreeMaxPathLength, a chain row's up to its width.
inline constexpr int kTreeClosedFormPath = kSpeculativeTreeMaxNodes;
static_assert(kBlockDv == 16, "the transposed reductions sum one value per state row of a tile");
static_assert(kTreeClosedFormPath % 16 == 0, "the ancestor reductions sum sixteen path entries at a time");

struct TreeClosedFormShared {
    float key[kSpeculativeTreeMaxNodes][kStateDim];          // normalized keys
    float value[kSpeculativeTreeMaxNodes][kBlockDv];         // the tile's value rows
    float sk[kSpeculativeTreeMaxNodes][kBlockDv];            // exp(G_c) s_0 k_c
    float sq[kSpeculativeTreeMaxNodes][kBlockDv];            // exp(G_c) s_0 q_c
    float u[kSpeculativeTreeMaxNodes][kBlockDv];             // deltas
    float kk[kSpeculativeTreeMaxNodes][kTreeClosedFormPath]; // exp(G_c - G_a) k_a . k_c, a = path[c][i], i < depth
    float qk[kSpeculativeTreeMaxNodes][kTreeClosedFormPath]; // exp(G_c - G_a) k_a . q_c, i <= depth
    float log_gamma[kSpeculativeTreeMaxNodes];               // g_c, then G_c
    float beta[kSpeculativeTreeMaxNodes];
    std::int8_t path[kSpeculativeTreeMaxNodes][kTreeClosedFormPath]; // root path, anchor first, c last
    std::int8_t depth[kSpeculativeTreeMaxNodes];
    int count;
    int levels;
};

// Sums each of a lane's sixteen values over the warp in sixteen shuffles: lane L returns the total of
// value (L >> 1) & 15 (lanes L and L ^ 1 return the same total).
__device__ __forceinline__ float transpose_reduce16(float (&v)[16], int lane) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool upper = (lane & 16) != 0;
        const float send = upper ? v[i] : v[i + 8];
        const float keep = upper ? v[i + 8] : v[i];
        v[i]             = keep + __shfl_xor_sync(kFullWarpMask, send, 16);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool upper = (lane & 8) != 0;
        const float send = upper ? v[i] : v[i + 4];
        const float keep = upper ? v[i + 4] : v[i];
        v[i]             = keep + __shfl_xor_sync(kFullWarpMask, send, 8);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool upper = (lane & 4) != 0;
        const float send = upper ? v[i] : v[i + 2];
        const float keep = upper ? v[i + 2] : v[i];
        v[i]             = keep + __shfl_xor_sync(kFullWarpMask, send, 4);
    }
    {
        const bool upper = (lane & 2) != 0;
        const float send = upper ? v[0] : v[1];
        const float keep = upper ? v[1] : v[0];
        v[0]             = keep + __shfl_xor_sync(kFullWarpMask, send, 2);
    }
    return v[0] + __shfl_xor_sync(kFullWarpMask, v[0], 1);
}

// One CTA per (request row, value head, state tile of 16 rows), as the walk's grid; 33 KiB of shared
// memory, two CTAs per SM. A chain row of a tree round is the chain over its valid prefix.
template <class State>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_tree_closed_form_kernel(const __grid_constant__ RecordAccess<true, State> access,
                                      const SpeculativeTreeRow* __restrict__ trees) {
    __shared__ TreeClosedFormShared shared;
    const RecurrentCoordinates coord = access.coordinates();
    const SpeculativeTreeRow& row    = trees[coord.batch];
    const int thread                 = coord.warp * kWarpSize + coord.lane;
    const int lane                   = coord.lane;
    const auto tile_base             = static_cast<std::uint32_t>(coord.state_tile * kBlockDv);
    // The tree and valid columns come from the round's ingress and tree build, and the initial state
    // from the preceding round's fold, so both are read before waiting on the projection.
    if (thread == 0) {
        const int count = row.tree != 0 ? row.nodes : max(0, access.valid_columns[coord.batch]);
        int levels      = 0;
        for (int c = 0; c < count; ++c) {
            const int parent = row.tree != 0 ? row.parent[c] : c - 1;
            const int depth  = parent < 0 ? 0 : shared.depth[parent] + 1;
            shared.depth[c]  = static_cast<std::int8_t>(depth);
            for (int i = 0; i < depth; ++i) { shared.path[c][i] = shared.path[parent][i]; }
            shared.path[c][depth] = static_cast<std::int8_t>(c);
            levels                = max(levels, depth + 1);
        }
        shared.count  = count;
        shared.levels = levels;
    }
    // Every warp holds the tile's whole initial state: lane l its four key dims of each of the 16 rows.
    __align__(16) float initial[kBlockDv][kQkPerLane];
    const State* initial_base = access.state_read_base(coord);
#pragma unroll
    for (int r = 0; r < kBlockDv; ++r) {
        load_qk_lane(initial[r], initial_base + static_cast<std::int64_t>(tile_base + r) * kStateDim, coord.dqk_base);
    }
    __syncthreads();
    pdl::enter_streaming();
    const int count               = shared.count;
    const bool key_records        = coord.state_tile == 0 && static_cast<int>(coord.value_head) % access.heads.group_size() == 0;
    // A: each column's raw inputs and records, its normalized key and its gate.
    for (int c = coord.warp; c < count; c += kNumWarps) {
        const std::int64_t column = access.column(coord, c);
        RawQkLane key             = load_raw_qk_lane(access.key_ptr(coord, c), coord.dqk_base);
        if (key_records) {
            store_vec(access.key_record + (column * access.heads.H_qk + coord.qk_head) * kStateDim + coord.dqk_base,
                      key.bits);
        }
        normalize_qk_lane<true>(key.value, lane);
        *reinterpret_cast<float4*>(&shared.key[c][coord.dqk_base]) =
            make_float4(key.value[0], key.value[1], key.value[2], key.value[3]);
        if (lane < kBlockDv) {
            const __nv_bfloat16 bits = access.value_ptr(coord, c)[tile_base + lane];
            access.value_record[(column * access.heads.H_v + coord.value_head) * kStateDim + tile_base + lane] = bits;
            shared.value[c][lane] = __bfloat162float(bits);
        }
        if (lane == 0) {
            const RawGatePair gate = access.load_gate(coord, c);
            if (coord.state_tile == 0) { access.gate_record[column * access.heads.H_v + coord.value_head] = gate.bits; }
            shared.log_gamma[c] = gate.g;
            shared.beta[c]      = gate.beta;
        }
    }
    __syncthreads();
    if (thread == 0) {
        for (int c = 0; c < count; ++c) {
            const int depth = shared.depth[c];
            if (depth > 0) { shared.log_gamma[c] += shared.log_gamma[shared.path[c][depth - 1]]; }
        }
    }
    __syncthreads();
    // B: the dense products with the initial state and the ancestor products, scaled by their decays.
    for (int c = coord.warp; c < count; c += kNumWarps) {
        const int depth    = shared.depth[c];
        const float gamma  = shared.log_gamma[c];
        const float4 key4  = *reinterpret_cast<const float4*>(&shared.key[c][coord.dqk_base]);
        const float key[4] = {key4.x, key4.y, key4.z, key4.w};
        RawQkLane query    = load_raw_qk_lane(access.query_ptr(coord, c), coord.dqk_base);
        normalize_qk_lane<true>(query.value, lane);
        float partial[16];
#pragma unroll
        for (int r = 0; r < kBlockDv; ++r) {
            partial[r] = initial[r][0] * key[0] + initial[r][1] * key[1] + initial[r][2] * key[2] +
                         initial[r][3] * key[3];
        }
        const float sk = transpose_reduce16(partial, lane);
#pragma unroll
        for (int r = 0; r < kBlockDv; ++r) {
            partial[r] = initial[r][0] * query.value[0] + initial[r][1] * query.value[1] +
                         initial[r][2] * query.value[2] + initial[r][3] * query.value[3];
        }
        const float sq    = transpose_reduce16(partial, lane);
        const float decay = expf(gamma);
        const int index   = (lane >> 1) & 15;
        if ((lane & 1) == 0) {
            shared.sk[c][index] = decay * sk;
            shared.sq[c][index] = decay * sq;
        }
        // Sixteen root-path entries at a time (a tree's path fits one chunk).
        float with_key[16];
        for (int first = 0; first <= depth; first += 16) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int i = first + j;
                partial[j]  = 0.0f;
                with_key[j] = 0.0f;
                if (i <= depth) {
                    const float4 a =
                        *reinterpret_cast<const float4*>(&shared.key[shared.path[c][i]][coord.dqk_base]);
                    partial[j] = a.x * query.value[0] + a.y * query.value[1] + a.z * query.value[2] +
                                 a.w * query.value[3];
                    if (i < depth) { with_key[j] = a.x * key[0] + a.y * key[1] + a.z * key[2] + a.w * key[3]; }
                }
            }
            const float qk = transpose_reduce16(partial, lane);
            const float kk = transpose_reduce16(with_key, lane);
            const int i    = first + index;
            if ((lane & 1) == 0 && i <= depth) {
                const float ratio = expf(gamma - shared.log_gamma[shared.path[c][i]]);
                shared.qk[c][i]   = ratio * qk;
                if (i < depth) { shared.kk[c][i] = ratio * kk; }
            }
        }
    }
    __syncthreads();
    // C: the deltas, one depth level at a time (a column's ancestors lie on earlier levels).
    const int d    = thread % kBlockDv;
    const int slot = thread / kBlockDv;
    constexpr int kSlots = kWarpSize * kNumWarps / kBlockDv;
    for (int level = 0; level < shared.levels; ++level) {
        for (int c = slot; c < count; c += kSlots) {
            if (shared.depth[c] != level) { continue; }
            float acc = shared.value[c][d] - shared.sk[c][d];
            for (int i = 0; i < level; ++i) { acc -= shared.kk[c][i] * shared.u[shared.path[c][i]][d]; }
            shared.u[c][d] = shared.beta[c] * acc;
        }
        __syncthreads();
    }
    // D: the outputs, and exact zeros past the live columns.
    for (int c = slot; c < count; c += kSlots) {
        float acc       = shared.sq[c][d];
        const int depth = shared.depth[c];
        for (int i = 0; i <= depth; ++i) { acc += shared.qk[c][i] * shared.u[shared.path[c][i]][d]; }
        access.output_ptr(coord, c)[tile_base + d] = __float2bfloat16(acc * access.scale);
    }
    for (int c = count + slot; c < access.width; c += kSlots) {
        access.output_ptr(coord, c)[tile_base + d] = __float2bfloat16(0.0f);
    }
}

} // namespace infernix::ops::detail::gated_delta_net
