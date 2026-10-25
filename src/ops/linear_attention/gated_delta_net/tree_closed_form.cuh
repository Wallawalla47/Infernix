#pragma once

// FastPath::GdnTreeClosedForm (candidates §13 PUB-05, §15; TreeWY, arXiv 2608.20961; Bole, 2608.01651):
// a verification tree's GDN record in closed form instead of the walk (recurrent_tree_walk_kernel), for rows of
// at most kTreeClosedFormColumns columns. Wider rows (K = 15 trees) walk: there the closed form's solve is up to
// 16 levels deep and measured 10-66 % slower than the walk at 2-4 rows (gdn_replay_bench --component tree).
//
// Along column c's root path P(c) = (0, .., c), the transition s_c = a_c s_p + u_c k_c^T (a = exp(g),
// p the parent, the row's initial state s_0 before column 0) unrolls to
//   s_c = exp(G_c) s_0 + sum_{a in P(c)} exp(G_c - G_a) u_a k_a^T,   G_c = sum of g over P(c),
// so each column's delta
//   u_c = beta_c (v_c - exp(G_c) s_0 k_c - sum_{a in P(c), a != c} exp(G_c - G_a) (k_a . k_c) u_a)
// depends only on its ancestors' (a lower-triangular solve in column order, one level per depth), and
// its output is
//   o_c = scale (exp(G_c) s_0 q_c + sum_{a in P(c)} exp(G_c - G_a) (k_a . q_c) u_a).
// The dense work is s_0 k_c and s_0 q_c, the tile's 16 state rows against every column; every other product
// follows the tree's ancestor pairs. The walk's W dependent steps become one parallel pass plus one short step per
// depth level. Everything is FP32, in another order than the walk, so outputs match the walk and the FP64 oracle
// within rounding but not bit for bit: a fast path. Records are the walk's raw copies, bit for bit.
//
// The grid is the walk's, one CTA per (request row, value head, state tile of 16 rows), and so is the occupancy
// it needs: the walk fits five CTAs per SM, and a closed form with fewer lost wherever its grid took one more wave
// (768 CTAs: the 27B at two rows, the 35B at three). Warp w therefore holds only rows 4w..4w+3 of the tile's initial
// state (the walk's 16 registers): it forms s_0 k_c and s_0 q_c for its rows and every column with one 32-value
// transposed reduction per eight columns, and solves its rows' deltas on its own (rows do not mix: one __syncwarp per
// depth level, no CTA barrier). Keys and queries stay in shared memory as their raw BF16 bits beside each column's
// normalization scale, from which a lane rebuilds the normalized value with normalize_qk_lane's one multiply: 11.7
// KiB, six CTAs per SM. Root paths and the summed gates take one thread per column.

#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

namespace infernix::ops::detail::gated_delta_net {

// The closed form's widest row; wider rows walk.
inline constexpr int kTreeClosedFormColumns = 16;
static_assert(kBlockDv == 16 && kDvPerWarp == 4, "four rows per warp, eight columns per 32-value reduction");
static_assert(kTreeClosedFormColumns % 8 == 0 && kTreeClosedFormColumns <= kWarpSize,
              "eight-column chunks and one-lane-per-column scans");

struct TreeClosedFormShared {
    // Raw key and query bits, lane L's four values of column c at [c][L], and each column's normalization scale.
    Bf16x4Pack key[kTreeClosedFormColumns][kWarpSize];
    Bf16x4Pack query[kTreeClosedFormColumns][kWarpSize];
    float key_scale[kTreeClosedFormColumns];
    float query_scale[kTreeClosedFormColumns];
    // exp(G_c - G_a) k_a . k_c (i < depth) and k_a . q_c (i <= depth) for a = path[c][i]; one word of padding
    // keeps the eight columns a warp's lanes read in one step on distinct banks.
    float kk[kTreeClosedFormColumns][kTreeClosedFormColumns + 1];
    float qk[kTreeClosedFormColumns][kTreeClosedFormColumns + 1];
    float u[kTreeClosedFormColumns][kBlockDv]; // deltas; warp w writes the tile's rows 4w..4w+3
    float g[kTreeClosedFormColumns];
    float gamma[kTreeClosedFormColumns]; // G_c
    float beta[kTreeClosedFormColumns];
    std::int8_t path[kTreeClosedFormColumns][kTreeClosedFormColumns]; // root path, anchor first, c last
    std::int8_t depth[kTreeClosedFormColumns];
};

// A lane's four normalized values of a column: its raw bits times the column's scale, as normalize_qk_lane
// computes them.
__device__ __forceinline__ void normalized_lane(const Bf16x4Pack& bits, float scale, float (&value)[kQkPerLane]) {
    const float2 lo = bf16x2_to_float2(bits.pair[0]);
    const float2 hi = bf16x2_to_float2(bits.pair[1]);
    value[0]        = lo.x * scale;
    value[1]        = lo.y * scale;
    value[2]        = hi.x * scale;
    value[3]        = hi.y * scale;
}

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

// Sums each of a lane's 32 values over the warp in 31 shuffles: lane L returns the total of value L.
// The lanes combine in transpose_reduce16's order (xor 16, 8, 4, 2, 1).
__device__ __forceinline__ float transpose_reduce32(float (&v)[32], int lane) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const bool upper = (lane & 16) != 0;
        const float send = upper ? v[i] : v[i + 16];
        const float keep = upper ? v[i + 16] : v[i];
        v[i]             = keep + __shfl_xor_sync(kFullWarpMask, send, 16);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool upper = (lane & 8) != 0;
        const float send = upper ? v[i] : v[i + 8];
        const float keep = upper ? v[i + 8] : v[i];
        v[i]             = keep + __shfl_xor_sync(kFullWarpMask, send, 8);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool upper = (lane & 4) != 0;
        const float send = upper ? v[i] : v[i + 4];
        const float keep = upper ? v[i + 4] : v[i];
        v[i]             = keep + __shfl_xor_sync(kFullWarpMask, send, 4);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool upper = (lane & 2) != 0;
        const float send = upper ? v[i] : v[i + 2];
        const float keep = upper ? v[i + 2] : v[i];
        v[i]             = keep + __shfl_xor_sync(kFullWarpMask, send, 2);
    }
    const bool upper = (lane & 1) != 0;
    const float send = upper ? v[0] : v[1];
    const float keep = upper ? v[1] : v[0];
    return keep + __shfl_xor_sync(kFullWarpMask, send, 1);
}

// Rows of at most kTreeClosedFormColumns columns (the launcher walks wider ones). A chain row of a tree round is the
// chain over its valid prefix.
template <class State>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 6)
    recurrent_tree_closed_form_kernel(const __grid_constant__ RecordAccess<true, State> access,
                                      const SpeculativeTreeRow* __restrict__ trees) {
    constexpr int kChunks = kTreeClosedFormColumns / 8;
    __shared__ TreeClosedFormShared shared;
    const RecurrentCoordinates coord = access.coordinates();
    const SpeculativeTreeRow& row    = trees[coord.batch];
    const int thread                 = coord.warp * kWarpSize + coord.lane;
    const int lane                   = coord.lane;
    const bool tree                  = row.tree != 0;
    const int count                  = tree ? row.nodes : max(0, access.valid_columns[coord.batch]);
    // The tree and valid columns come from the round's ingress and tree build, and the initial state
    // from the preceding round's fold, so both are read before waiting on the projection.
    if (thread < count) {
        const int depth = tree ? row.depth[thread] : thread;
        int at          = thread;
        for (int i = depth; i >= 0; --i) {
            shared.path[thread][i] = static_cast<std::int8_t>(at);
            at                     = tree ? row.parent[at] : at - 1;
        }
        shared.depth[thread] = static_cast<std::int8_t>(depth);
    }
    __align__(16) float state[kDvPerWarp][kQkPerLane];
    load_state_tile(state, access.state_read_base(coord), coord);
    __syncthreads();
    pdl::enter_streaming();
    // A: each column's raw inputs and records, its key and query bits with their scales, and its gate.
    const bool key_records =
        coord.state_tile == 0 && static_cast<int>(coord.value_head) % access.heads.group_size() == 0;
    for (int c = coord.warp; c < count; c += kNumWarps) {
        const std::int64_t column = access.column(coord, c);
        const RawQkLane key       = load_raw_qk_lane(access.key_ptr(coord, c), coord.dqk_base);
        if (key_records) {
            store_vec(access.key_record + (column * access.heads.H_qk + coord.qk_head) * kStateDim + coord.dqk_base,
                      key.bits);
        }
        const float key_scale     = qk_inverse_norm(key.value, lane);
        shared.key[c][lane]       = key.bits;
        const RawQkLane query     = load_raw_qk_lane(access.query_ptr(coord, c), coord.dqk_base);
        const float query_scale   = qk_inverse_norm(query.value, lane);
        shared.query[c][lane]     = query.bits;
        if (lane == 0) {
            shared.key_scale[c]    = key_scale;
            shared.query_scale[c]  = query_scale;
            const RawGatePair gate = access.load_gate(coord, c);
            if (coord.state_tile == 0) { access.gate_record[column * access.heads.H_v + coord.value_head] = gate.bits; }
            shared.g[c]    = gate.g;
            shared.beta[c] = gate.beta;
        }
    }
    __syncthreads();
    // G_c, summed from the root as the walk's prefix order adds it (g_c + G_parent at every step).
    if (thread < count) {
        const int depth = shared.depth[thread];
        float gamma     = shared.g[shared.path[thread][0]];
        for (int i = 1; i <= depth; ++i) { gamma = shared.g[shared.path[thread][i]] + gamma; }
        shared.gamma[thread] = gamma;
    }
    // B1: s_0 k_c and s_0 q_c for the warp's four rows and every column, eight columns per reduction;
    // lane L ends with column 8m + L/4, row L%4 of chunk m, the lane the solve below gives that pair.
    float sk[kChunks];
    float sq[kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
        sk[m] = 0.0f;
        sq[m] = 0.0f;
        if (8 * m >= count) { continue; }
        float partial[32];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c  = 8 * m + j;
            float key[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (c < count) { normalized_lane(shared.key[c][lane], shared.key_scale[c], key); }
#pragma unroll
            for (int r = 0; r < kDvPerWarp; ++r) {
                partial[j * kDvPerWarp + r] =
                    state[r][0] * key[0] + state[r][1] * key[1] + state[r][2] * key[2] + state[r][3] * key[3];
            }
        }
        sk[m] = transpose_reduce32(partial, lane);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c    = 8 * m + j;
            float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (c < count) { normalized_lane(shared.query[c][lane], shared.query_scale[c], query); }
#pragma unroll
            for (int r = 0; r < kDvPerWarp; ++r) {
                partial[j * kDvPerWarp + r] = state[r][0] * query[0] + state[r][1] * query[1] +
                                              state[r][2] * query[2] + state[r][3] * query[3];
            }
        }
        sq[m] = transpose_reduce32(partial, lane);
    }
    __syncthreads();
    // B2: the ancestor products, scaled by their decays (a path holds at most 16 entries).
    for (int c = coord.warp; c < count; c += kNumWarps) {
        const int depth   = shared.depth[c];
        const float gamma = shared.gamma[c];
        float key[4];
        float query[4];
        normalized_lane(shared.key[c][lane], shared.key_scale[c], key);
        normalized_lane(shared.query[c][lane], shared.query_scale[c], query);
        const int index = (lane >> 1) & 15;
        float partial[16];
        float with_key[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            partial[i]  = 0.0f;
            with_key[i] = 0.0f;
            if (i <= depth) {
                const int a = shared.path[c][i];
                float ancestor[4];
                normalized_lane(shared.key[a][lane], shared.key_scale[a], ancestor);
                partial[i] =
                    ancestor[0] * query[0] + ancestor[1] * query[1] + ancestor[2] * query[2] + ancestor[3] * query[3];
                if (i < depth) {
                    with_key[i] =
                        ancestor[0] * key[0] + ancestor[1] * key[1] + ancestor[2] * key[2] + ancestor[3] * key[3];
                }
            }
        }
        const float qk = transpose_reduce16(partial, lane);
        const float kk = transpose_reduce16(with_key, lane);
        if ((lane & 1) == 0 && index <= depth) {
            const float ratio   = expf(gamma - shared.gamma[shared.path[c][index]]);
            shared.qk[c][index] = ratio * qk;
            if (index < depth) { shared.kk[c][index] = ratio * kk; }
        }
    }
    __syncthreads();
    // C: the warp's rows' deltas, one depth level at a time (a column's ancestors lie on earlier levels
    // and in the same rows), lane (slot, r) for columns slot + 8m and tile row 4w + r.
    const int slot         = lane >> 2;
    const int tile_row     = coord.warp * kDvPerWarp + (lane & 3);
    const std::uint32_t dv = coord.dv_base + static_cast<std::uint32_t>(lane & 3);
    const int levels       = __reduce_max_sync(kFullWarpMask, lane < count ? shared.depth[lane] : -1) + 1;
    float value[kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
        const int c = slot + 8 * m;
        value[m]    = 0.0f;
        if (c < count) {
            const __nv_bfloat16 bits = access.value_ptr(coord, c)[dv];
            access.value_record[(access.column(coord, c) * access.heads.H_v + coord.value_head) * kStateDim + dv] = bits;
            value[m] = __bfloat162float(bits);
        }
    }
    for (int level = 0; level < levels; ++level) {
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int c = slot + 8 * m;
            if (c >= count || shared.depth[c] != level) { continue; }
            // exp(G_c) s_0 k_c rounds once before the subtraction.
            float acc = value[m] - __fmul_rn(expf(shared.gamma[c]), sk[m]);
            for (int i = 0; i < level; ++i) { acc -= shared.kk[c][i] * shared.u[shared.path[c][i]][tile_row]; }
            shared.u[c][tile_row] = shared.beta[c] * acc;
        }
        __syncwarp();
    }
    // D: the outputs, and exact zeros past the live columns.
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
        const int c = slot + 8 * m;
        if (c >= count) { continue; }
        float acc       = __fmul_rn(expf(shared.gamma[c]), sq[m]);
        const int depth = shared.depth[c];
        for (int i = 0; i <= depth; ++i) { acc += shared.qk[c][i] * shared.u[shared.path[c][i]][tile_row]; }
        access.output_ptr(coord, c)[dv] = __float2bfloat16(acc * access.scale);
    }
    for (int c = count + slot; c < access.width; c += 8) { access.output_ptr(coord, c)[dv] = __float2bfloat16(0.0f); }
}

} // namespace infernix::ops::detail::gated_delta_net
