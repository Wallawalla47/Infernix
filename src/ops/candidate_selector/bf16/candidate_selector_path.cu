#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include "infernix/ops/speculative_tree.h"
#include <cuda_bf16.h>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {
constexpr int kCandidates = 16, kRank = 256;

struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    const __nv_bfloat16* predecessor;
    const __nv_bfloat16* successor;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    int steps;
    float* confidence; // [K, B] or null: the selected rank's temperature-1 softmax
};

struct alignas(16) SelectorShared {
    __nv_bfloat16 successors[kCandidates * kRank];
    float product[kRank];
    float edge[kCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

// The probabilities written here are the same FP32 values consumed by the draw. Draft `step` of a
// round at base position `base` draws with (seed, base + step, DFlash2Proposal, step): the step keeps
// a later round, whose drafts at the same position have smaller steps, from meeting the uniform again
// (block verification reads every draft of its round), as sampled MTP drafts' `draw` does.
__device__ int draw_rank(float edge, float temperature, unsigned long long seed, int base, int step,
                         float* q) {
    const int position = base + step;
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, static_cast<unsigned>(step))
                  : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits = __ballot_sync(kFullWarpMask, lane < kCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kCandidates - 1;
}

// Lane 0 writes the temperature-1 softmax over the 16 edges at `selected` (a warp-wide call).
__device__ __forceinline__ void write_confidence(float edge, int selected, float* out) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    const float weight  = lane < kCandidates ? __expf(edge - maximum) : 0.0F;
    const float total   = warp_sum(weight);
    const float chosen  = __shfl_sync(kFullWarpMask, weight, selected);
    if (lane == 0) *out = chosen / total;
}

__device__ void score_row(const DeviceArgs& a, int column, SelectorShared& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int token = lane == 0 ? a.ids[column * kCandidates + warp] : 0;
    token     = __shfl_sync(kFullWarpMask, token, 0);
    cp_async<16, Cache::cg>(&shared.successors[warp * kRank + lane * 8],
                            a.successor + static_cast<std::int64_t>(token) * kRank + lane * 8);
    cp_commit();
    // Publish the preceding draw before reading its token. Successor prefetch is independent.
    __syncthreads();
    if (tid < kRank)
        shared.product[tid] =
            __bfloat162float(
                a.predecessor[static_cast<std::int64_t>(shared.predecessor) * kRank + tid]) *
            __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kRank + tid]);
    cp_wait<0>();
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
#pragma unroll
        for (int r = lane; r < kRank; r += 32)
            sum = fmaf(shared.product[r], __bfloat162float(shared.successors[c * kRank + r]), sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = a.unary[column * kCandidates + c] + sum;
    }
    __syncthreads();
}

__global__ __launch_bounds__(512, 1) void selector_walk_kernel(DeviceArgs a) {
    pdl::enter();
    __shared__ SelectorShared shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor   = a.anchors[batch];
        shared.base_position = a.positions[batch];
        shared.temperature   = a.configs[batch].temperature;
        shared.seed          = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        score_row(a, column, shared);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed, shared.base_position, step,
                                           a.q + column * kCandidates);
            if (a.confidence != nullptr) { write_confidence(edge, selected, a.confidence + column); }
            if (lane == 0) {
                shared.predecessor = a.ids[column * kCandidates + selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

__global__ __launch_bounds__(512, 2) void selector_lattice_kernel(DeviceArgs a, float* edges) {
    pdl::enter();
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if (threadIdx.x == 0)
        shared.predecessor = step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
    score_row(a, column, shared);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

__global__ __launch_bounds__(32) void selector_lattice_walk_kernel(DeviceArgs a,
                                                                   const float* edges) {
    pdl::enter();
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kCandidates
                ? edges[(static_cast<std::int64_t>(column) * kCandidates + predecessor_rank) *
                            kCandidates +
                        lane]
                : -CUDART_INF_F;
        int selected = draw_rank(edge, temperature, seed, position, step, a.q + column * kCandidates);
        if (a.confidence != nullptr) { write_confidence(edge, selected, a.confidence + column); }
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kCandidates + selected];
    }
}

struct TreeArgs {
    const std::int32_t* ids;
    const std::int32_t* positions;
    const std::int32_t* extents;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    std::int32_t* column_candidates;
    float* q;
    SpeculativeTreeRow* rows;
    std::uint32_t* masks;
    std::int32_t* valid_columns;
    int steps;
    int width;
    int max_paths;
    bool transport; // SpeculativeTreeShape::transport
};

__device__ void publish_tree_row(const TreeArgs& a, int row, int nodes, bool tree_row,
                                 const int* parent, const int* depth, const int* first_child,
                                 const int* next_sibling, const int* sibling, bool transport = false,
                                 std::uint32_t sampled = 0U);

// The law a node's children are drawn from and scored by, one candidate rank per lane: the
// proposal q at the row's temperature (bit-identical to draw_rank's) for sampled rows, and the
// temperature-1 softmax of the edges for greedy rows, which only use it to rank branches.
__device__ __forceinline__ float child_law(float edge, float temperature) {
    const int lane          = threadIdx.x & 31;
    const float scale       = temperature > 0.0F ? temperature : 1.0F;
    const float maximum     = warp_max(edge);
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / scale) : 0.0F;
    const float probability = weight / warp_sum(weight);
    return lane < kCandidates ? probability : 0.0F;
}

// Expected law value of a node's next child given the ranks its earlier children took: the
// next draw is the best untaken rank for greedy rows and a without-replacement draw from the
// untaken law for sampled rows. The expectation depends only on earlier draws, so choosing to
// add that child never depends on the child itself.
__device__ __forceinline__ float next_child_expectation(float law, unsigned taken, bool greedy) {
    const int lane       = threadIdx.x & 31;
    const bool available = lane < kCandidates && ((taken >> lane) & 1U) == 0U;
    if (greedy) {
        const float best = warp_max(available ? law : -CUDART_INF_F);
        return best > -CUDART_INF_F ? best : -CUDART_INF_F;
    }
    const float mass = warp_sum(available ? law : 0.0F);
    if (!(mass > 0.0F)) return -CUDART_INF_F;
    return warp_sum(available ? law * law : 0.0F) / mass;
}

// Without-replacement draw of a node's next child from its law restricted to the untaken ranks.
// Greedy rows take the lowest untaken rank attaining the untaken edge maximum.
__device__ __forceinline__ int draw_untaken(float edge, float law, unsigned taken, bool greedy,
                                            unsigned long long seed, int position, unsigned index) {
    const int lane       = threadIdx.x & 31;
    const bool available = lane < kCandidates && ((taken >> lane) & 1U) == 0U;
    if (greedy) {
        const float maximum    = warp_max(available ? edge : -CUDART_INF_F);
        const unsigned winners = __ballot_sync(kFullWarpMask, available && edge == maximum);
        return winners == 0 ? 0 : __ffs(winners) - 1;
    }
    const float remaining = available ? law : 0.0F;
    const float mass      = warp_sum(remaining);
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, index) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    const float goal = uniform * mass;
    float cumulative = remaining;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned open = __ballot_sync(kFullWarpMask, available && remaining > 0.0F);
    const unsigned hits =
        __ballot_sync(kFullWarpMask, available && remaining > 0.0F && goal < cumulative);
    if (hits) return __ffs(hits) - 1;
    if (open) return 31 - __clz(open);
    const unsigned any = __ballot_sync(kFullWarpMask, available);
    return any ? __ffs(any) - 1 : 0;
}

// Builds one row's verification tree over the K-position lattice (one warp per row).
//
// The main chain repeats selector_lattice_walk_kernel's draws bit for bit (columns 1..K). A tree
// row then adds side columns best first until the row holds `width` columns: lane v holds the
// priority of node v's next child, score(v) * E[law value of that child], where score is the
// product of the law values on v's path. The winner's next child is drawn without replacement
// from its law with a counter index unique to (parent, sibling), so draws at one position never
// share a uniform. Children never exceed depth K and leaves never exceed max_paths; a row stops
// early when no node can take another child.
//
// Independent: `edges` is FP32 [16,K,B], one edge row per position whatever the predecessor's rank
// (candidate_tree), instead of the lattice's [16,16,K,B].
//
// Transport (a sampled tree row of a.transport, UniVer): the shape is built first as a greedy row's
// is (each node's best untaken rank, under the row's sampled law), so it follows no draw. The
// tokens are then filled parents first: a node's children but the last take its law's top ranks in
// rank order and the last is drawn from its law restricted to the other ranks (the sampled bit),
// with the chain's counter key when it is a main column's only child and (node+1)<<5 | sibling
// otherwise. Every child stores its parent's full law.
template <bool Independent>
__global__ __launch_bounds__(32) void selector_tree_build_kernel(TreeArgs a, const float* edges) {
    pdl::enter();
    constexpr int kMax = kSpeculativeTreeMaxNodes;
    const int lane = threadIdx.x, row = blockIdx.x;
    const int k = a.steps, width = a.width, drafts = width - 1;
    __shared__ float law[kMax][kCandidates];
    __shared__ int rank[kMax], depth[kMax], parent[kMax], children[kMax];
    __shared__ int first_child[kMax], last_child[kMax], next_sibling[kMax], sibling[kMax];
    __shared__ unsigned taken[kMax];
    __shared__ float score[kMax];
    // The row's whole lattice, so the serial build reads every edge from shared memory.
    constexpr int kEdgeSteps       = kSpeculativeTreeMaxPathLength - 1;
    constexpr int kEdgePredecessor = Independent ? 1 : kCandidates;
    __shared__ __align__(16) float edge_rows[kEdgeSteps][kEdgePredecessor][kCandidates];
    {
        const float4* source = reinterpret_cast<const float4*>(
            edges + static_cast<std::int64_t>(row) * k * kEdgePredecessor * kCandidates);
        float4* destination = reinterpret_cast<float4*>(&edge_rows[0][0][0]);
        for (int i = lane; i < k * kEdgePredecessor * kCandidates / 4; i += 32)
            destination[i] = source[i];
    }
    rank[lane]                = 0;
    depth[lane]               = 0;
    parent[lane]              = -1;
    children[lane]            = 0;
    first_child[lane]         = -1;
    last_child[lane]          = -1;
    next_sibling[lane]        = -1;
    sibling[lane]             = 0;
    taken[lane]               = 0U;
    score[lane]               = 0.0F;
    const auto seed           = a.configs[row].seed;
    const float temperature   = a.configs[row].temperature;
    const bool greedy         = !(temperature > 0.0F);
    const int position        = a.positions[row];
    const bool tree_row       = a.extents[row] == drafts;
    const bool transport      = a.transport && !greedy && tree_row;
    // A transport row's shape takes each node's best untaken rank, as a greedy row's does.
    const bool shape_greedy   = greedy || transport;
    const std::int64_t q_base = static_cast<std::int64_t>(row) * drafts;
    __syncwarp();
    if (lane == 0) score[0] = 1.0F;
    __syncwarp();

    const auto lattice_edge = [&](int step, int predecessor_rank) {
        return lane < kCandidates ? edge_rows[step][Independent ? 0 : predecessor_rank][lane]
                                  : -CUDART_INF_F;
    };
    const auto publish_column = [&](int column, int step, int selected, const float* q_row) {
        const int lattice = row * k + step;
        float* q          = a.q + (q_base + column - 1) * kCandidates;
        if (lane < kCandidates) {
            a.column_candidates[(q_base + column - 1) * kCandidates + lane] =
                a.ids[lattice * kCandidates + lane];
            if (q_row != nullptr) q[lane] = q_row[lane];
        }
        if (lane == 0) a.drafts[q_base + column - 1] = a.ids[lattice * kCandidates + selected];
    };

    // Main chain.
#pragma unroll 1
    for (int step = 0; step < k; ++step) {
        const int column = step + 1, from = step;
        const float edge = lattice_edge(step, rank[from]);
        const float p    = child_law(edge, temperature);
        if (lane < kCandidates) law[from][lane] = p;
        int selected = 0;
        if (transport) {
            selected = draw_untaken(edge, p, 0U, true, seed, position + step, 0U); // published below
        } else {
            selected = draw_rank(edge, temperature, seed, position, step, a.q + (q_base + column - 1) * kCandidates);
            publish_column(column, step, selected, nullptr);
        }
        __syncwarp();
        if (lane == 0) {
            rank[column]      = selected;
            depth[column]     = column;
            parent[column]    = from;
            children[from]    = 1;
            first_child[from] = column;
            last_child[from]  = column;
            taken[from]       = 1U << selected;
            score[column]     = score[from] * law[from][selected];
        }
        __syncwarp();
    }

    int nodes = k + 1;
    if (tree_row) {
        // Lane v keeps node v's next-child priority.
        float priority = -CUDART_INF_F;
#pragma unroll 1
        for (int v = 0; v < k; ++v) {
            const float e = next_child_expectation(lane < kCandidates ? law[v][lane] : 0.0F, taken[v],
                                                   shape_greedy);
            if (lane == v) priority = e > -CUDART_INF_F ? score[v] * e : -CUDART_INF_F;
        }
        int leaves = 1;
#pragma unroll 1
        while (nodes < width) {
            // A child of a node that already has children adds a leaf.
            const bool open     = lane < nodes && depth[lane] < k && children[lane] < kCandidates &&
                                  (children[lane] == 0 || leaves < a.max_paths);
            const float value   = open ? priority : -CUDART_INF_F;
            const float best    = warp_max(value);
            const unsigned wins = __ballot_sync(kFullWarpMask, open && value == best);
            if (!(best > -CUDART_INF_F) || wins == 0U) break;
            const int from   = __ffs(wins) - 1;
            const int column = nodes;
            const int step   = depth[from];
            const int index  = children[from];
            const float edge = lattice_edge(step, rank[from]);
            const float p    = lane < kCandidates ? law[from][lane] : 0.0F;
            const unsigned id =
                (static_cast<unsigned>(from + 1) << 5) | static_cast<unsigned>(index);
            const int selected =
                draw_untaken(edge, p, taken[from], shape_greedy, seed, position + step, id);
            if (!transport) { publish_column(column, step, selected, greedy ? nullptr : law[from]); }
            if (greedy && lane < kCandidates) {
                a.q[(q_base + column - 1) * kCandidates + lane] = lane == selected ? 1.0F : 0.0F;
            }
            __syncwarp();
            if (lane == 0) {
                if (children[from] > 0) {
                    ++leaves;
                    next_sibling[last_child[from]] = column;
                } else {
                    first_child[from] = column;
                }
                last_child[from] = column;
                sibling[column]  = children[from];
                children[from] += 1;
                taken[from] |= 1U << selected;
                rank[column]   = selected;
                depth[column]  = step + 1;
                parent[column] = from;
                score[column]  = score[from] * law[from][selected];
            }
            __syncwarp();
            leaves = __shfl_sync(kFullWarpMask, leaves, 0);
            // The parent's next child, then the new column's first child.
            const float again = next_child_expectation(p, taken[from], shape_greedy);
            if (lane == from)
                priority = again > -CUDART_INF_F ? score[from] * again : -CUDART_INF_F;
            float first = -CUDART_INF_F;
            if (depth[column] < k) {
                const float child_edge = lattice_edge(depth[column], selected);
                const float c          = child_law(child_edge, temperature);
                if (lane < kCandidates) law[column][lane] = c;
                const float e = next_child_expectation(c, 0U, shape_greedy);
                first         = e > -CUDART_INF_F ? score[column] * e : -CUDART_INF_F;
            }
            if (lane == column) priority = first;
            ++nodes;
            __syncwarp();
        }
    } else {
        const int extent = a.extents[row];
        nodes            = (extent < 0 ? 0 : (extent > k ? k : extent)) + 1;
    }
    std::uint32_t sampled = 0U;
    if (transport) {
        // Token fill over the built shape, parents before children (parent < child): the laws come
        // from the filled ranks, so a node's children follow its own token.
#pragma unroll 1
        for (int v = 0; v < nodes; ++v) {
            if (children[v] == 0) continue; // uniform: shared
            const int step   = depth[v];
            const float edge = lattice_edge(step, rank[v]);
            const float p    = child_law(edge, temperature);
            if (lane < kCandidates) law[v][lane] = p;
            __syncwarp();
            unsigned used = 0U;
            int j         = 0;
#pragma unroll 1
            for (int c = first_child[v]; c >= 0; c = next_sibling[c], ++j) {
                const bool last = j == children[v] - 1;
                int selected    = 0;
                if (last) {
                    // A main column's only child keeps the chain's key (its step, draw_rank's sub-key);
                    // a node's other last children take the sibling key, so no two draws share a uniform.
                    const unsigned id = j == 0 && c == v + 1 && v < k
                                            ? static_cast<unsigned>(step)
                                            : (static_cast<unsigned>(v + 1) << 5) | static_cast<unsigned>(j);
                    selected = draw_untaken(edge, p, used, false, seed, position + step, id);
                    sampled |= 1U << c;
                } else {
                    selected = draw_untaken(edge, p, used, true, seed, position + step, 0U);
                }
                publish_column(c, step, selected, law[v]);
                used |= 1U << selected;
                __syncwarp();
                if (lane == 0) rank[c] = selected;
                __syncwarp();
            }
        }
    }
    publish_tree_row(a, row, nodes, tree_row, parent, depth, first_child, next_sibling, sibling, transport,
                     sampled);
}

// A row's built tree: topology, ancestor masks and root-to-leaf paths (one warp; the arrays are the
// builder's shared node tables).
__device__ void publish_tree_row(const TreeArgs& a, int row, int nodes, bool tree_row,
                                 const int* parent, const int* depth, const int* first_child,
                                 const int* next_sibling, const int* sibling, bool transport,
                                 std::uint32_t sampled) {
    constexpr int kMax = kSpeculativeTreeMaxNodes;
    const int lane = threadIdx.x, k = a.steps, width = a.width;
    SpeculativeTreeRow& out = a.rows[row];
    if (lane == 0) {
        out.nodes       = nodes;
        out.main_depth  = k;
        out.tree        = tree_row ? 1 : 0;
        out.transport   = transport ? 1 : 0;
        out.sampled     = transport ? sampled : 0U;
        out.reserved[0] = 0;
        out.reserved[1] = 0;
        if (tree_row) a.valid_columns[row] = nodes;
    }
    for (int c = lane; c < kMax; c += 32) {
        const bool live = c < nodes;
        out.parent[c]   = static_cast<std::int8_t>(live ? parent[c] : -1);
        out.depth[c]    = static_cast<std::int8_t>(live ? depth[c] : 0);
        out.first_child[c] =
            static_cast<std::int8_t>(live && first_child[c] < nodes ? first_child[c] : -1);
        out.next_sibling[c] =
            static_cast<std::int8_t>(live && next_sibling[c] < nodes ? next_sibling[c] : -1);
        out.sibling_index[c] = static_cast<std::int8_t>(live ? sibling[c] : 0);
    }
    __syncwarp();
    // Ancestor mask of column c: its own bit and those of every ancestor.
    std::uint32_t mask = 0U;
    if (lane < nodes) {
        for (int c = lane; c >= 0; c = parent[c]) mask |= 1U << c;
    } else if (lane < width) {
        mask = 1U << lane;
    }
    if (lane < width) a.masks[static_cast<std::int64_t>(row) * width + lane] = mask;
    if (lane == 0) {
        std::uint32_t covered = 0U;
        int paths             = 0;
        for (int c = 0; c < nodes; ++c) {
            const bool leaf = tree_row ? first_child[c] < 0 : c == nodes - 1;
            if (!leaf || paths == kSpeculativeTreeMaxPaths) continue;
            const int length = depth[c] + 1;
            int owned_from   = 0;
            for (int at = c, i = length - 1; at >= 0; at = parent[at], --i) {
                out.path_columns[paths][i] = static_cast<std::int8_t>(at);
                if (owned_from == 0 && ((covered >> at) & 1U) != 0U) owned_from = i + 1;
            }
            for (int at = c; at >= 0; at = parent[at]) covered |= 1U << at;
            out.path_length[paths] = static_cast<std::int8_t>(length);
            out.owned_from[paths]  = static_cast<std::int8_t>(owned_from);
            ++paths;
        }
        for (int p = paths; p < kSpeculativeTreeMaxPaths; ++p) {
            out.path_length[p] = 0;
            out.owned_from[p]  = 0;
        }
        out.paths = paths;
    }
}

struct ChainTreeArgs {
    TreeArgs tree;      // ids: each step's candidates; drafts: the chain in, side columns out
    const float* laws;  // FP32 [16,K,B]: the law each chain draft was drawn from
    const float* scores; // FP32 [score_ranks,K,B]: greedy rows' ranking log-probabilities
    int score_ranks;
};

// Extends one row's proposed chain to a verification tree of leaves (one warp per row): the chain
// is columns 1..K as proposed, at its drafts' ranks among their step's candidates, and side columns
// are siblings of chain columns that take no children, added best first as the lattice builder adds
// them. A sampled row's law at step i is the given law its chain draft was drawn from; a greedy
// row ranks its alternatives by the temperature-1 softmax of its scores. A tree row whose chain
// draft is missing from its candidates keeps its chain alone (a tree without side columns).
__global__ __launch_bounds__(32) void chain_tree_build_kernel(ChainTreeArgs c) {
    pdl::enter();
    const TreeArgs& a  = c.tree;
    constexpr int kMax = kSpeculativeTreeMaxNodes;
    const int lane = threadIdx.x, row = blockIdx.x;
    const int k = a.steps, width = a.width, drafts = width - 1;
    __shared__ float law[kMax][kCandidates];
    __shared__ int depth[kMax], parent[kMax], children[kMax];
    __shared__ int first_child[kMax], last_child[kMax], next_sibling[kMax], sibling[kMax];
    __shared__ unsigned taken[kMax];
    __shared__ float score[kMax];
    depth[lane]        = 0;
    parent[lane]       = -1;
    children[lane]     = 0;
    first_child[lane]  = -1;
    last_child[lane]   = -1;
    next_sibling[lane] = -1;
    sibling[lane]      = 0;
    taken[lane]        = 0U;
    score[lane]        = 0.0F;
    const auto seed           = a.configs[row].seed;
    const bool greedy         = !(a.configs[row].temperature > 0.0F);
    const int position        = a.positions[row];
    const bool tree_row       = a.extents[row] == drafts;
    bool extend               = tree_row;
    const std::int64_t q_base = static_cast<std::int64_t>(row) * drafts;
    __syncwarp();
    if (lane == 0) score[0] = 1.0F;
    __syncwarp();

    // The chain, as proposed.
#pragma unroll 1
    for (int step = 0; step < k; ++step) {
        const int column = step + 1, from = step;
        const std::int64_t lattice = static_cast<std::int64_t>(row) * k + step;
        const int id      = lane < kCandidates ? a.ids[lattice * kCandidates + lane] : -1;
        const int draft   = a.drafts[q_base + step];
        const unsigned at = __ballot_sync(kFullWarpMask, lane < kCandidates && id == draft);
        const int rank    = at != 0U ? __ffs(at) - 1 : -1;
        extend            = extend && rank >= 0;
        const float given = lane < kCandidates ? c.laws[lattice * kCandidates + lane] : 0.0F;
        float p           = given;
        if (greedy) {
            const float s = lane < c.score_ranks
                                ? c.scores[lattice * c.score_ranks + lane]
                                : -CUDART_INF_F;
            p = child_law(s, 0.0F);
        }
        if (lane < kCandidates) {
            law[from][lane] = p;
            a.column_candidates[(q_base + column - 1) * kCandidates + lane] = id;
            a.q[(q_base + column - 1) * kCandidates + lane]                 = given;
        }
        __syncwarp();
        if (lane == 0) {
            const int r       = rank < 0 ? 0 : rank;
            depth[column]     = column;
            parent[column]    = from;
            children[from]    = 1;
            first_child[from] = column;
            last_child[from]  = column;
            taken[from]       = rank < 0 ? 0U : 1U << rank;
            score[column]     = score[from] * law[from][r];
        }
        __syncwarp();
    }

    int nodes = k + 1;
    if (extend) {
        // Lane v keeps chain node v's next-child priority; side columns take no children.
        float priority = -CUDART_INF_F;
#pragma unroll 1
        for (int v = 0; v < k; ++v) {
            const float e =
                next_child_expectation(lane < kCandidates ? law[v][lane] : 0.0F, taken[v], greedy);
            if (lane == v) priority = e > -CUDART_INF_F ? score[v] * e : -CUDART_INF_F;
        }
        int leaves = 1;
#pragma unroll 1
        while (nodes < width && leaves < a.max_paths) {
            const bool open     = lane < k && children[lane] < kCandidates;
            const float value   = open ? priority : -CUDART_INF_F;
            const float best    = warp_max(value);
            const unsigned wins = __ballot_sync(kFullWarpMask, open && value == best);
            if (!(best > -CUDART_INF_F) || wins == 0U) break;
            const int from   = __ffs(wins) - 1;
            const int column = nodes;
            const int index  = children[from];
            const float p    = lane < kCandidates ? law[from][lane] : 0.0F;
            const unsigned id =
                (static_cast<unsigned>(from + 1) << 5) | static_cast<unsigned>(index);
            const int selected =
                draw_untaken(p, p, taken[from], greedy, seed, position + from, id);
            const std::int64_t lattice = static_cast<std::int64_t>(row) * k + from;
            const int candidate = lane < kCandidates ? a.ids[lattice * kCandidates + lane] : -1;
            if (lane < kCandidates) {
                a.column_candidates[(q_base + column - 1) * kCandidates + lane] = candidate;
                a.q[(q_base + column - 1) * kCandidates + lane] =
                    greedy ? (lane == selected ? 1.0F : 0.0F) : p;
            }
            if (lane == selected) a.drafts[q_base + column - 1] = candidate;
            __syncwarp();
            if (lane == 0) {
                ++leaves;
                next_sibling[last_child[from]] = column;
                last_child[from]               = column;
                sibling[column]                = children[from];
                children[from] += 1;
                taken[from] |= 1U << selected;
                depth[column]  = from + 1;
                parent[column] = from;
            }
            __syncwarp();
            leaves = __shfl_sync(kFullWarpMask, leaves, 0);
            const float again = next_child_expectation(p, taken[from], greedy);
            if (lane == from)
                priority = again > -CUDART_INF_F ? score[from] * again : -CUDART_INF_F;
            ++nodes;
            __syncwarp();
        }
    } else if (!tree_row) {
        const int extent = a.extents[row];
        nodes            = (extent < 0 ? 0 : (extent > k ? k : extent)) + 1;
    }
    publish_tree_row(a, row, nodes, tree_row, parent, depth, first_child, next_sibling, sibling);
}

} // namespace

void candidate_selector_tree_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                    const Tensor& projected_hidden, const Tensor& anchors,
                                    const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const Tensor& current_extents, const SamplingConfig* configs,
                                    SpeculativeTreeShape shape, Tensor& drafts,
                                    Tensor& column_candidates, Tensor& proposal_q,
                                    Tensor& tree_rows, Tensor& tree_masks, Tensor& valid_columns,
                                    const SelectorWorkspace& workspace, cudaStream_t stream) {
    const DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                          static_cast<const float*>(unary_scores.data),
                          static_cast<const __nv_bfloat16*>(projected_hidden.data),
                          static_cast<const std::int32_t*>(anchors.data),
                          static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
                          static_cast<const __nv_bfloat16*>(successor_codebook.data),
                          static_cast<const std::int32_t*>(base_positions.data),
                          configs,
                          nullptr,
                          nullptr,
                          candidate_ids.ne[1],
                          nullptr};
    auto* edges = static_cast<float*>(workspace.edges.data);
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(args.steps * candidate_ids.ne[2], kCandidates), dim3(512), 0, stream},
        selector_lattice_kernel, args, edges));
    CUDA_CHECK(cudaGetLastError());
    const TreeArgs tree_args{static_cast<const std::int32_t*>(candidate_ids.data),
                             static_cast<const std::int32_t*>(base_positions.data),
                             static_cast<const std::int32_t*>(current_extents.data),
                             configs,
                             static_cast<std::int32_t*>(drafts.data),
                             static_cast<std::int32_t*>(column_candidates.data),
                             static_cast<float*>(proposal_q.data),
                             static_cast<SpeculativeTreeRow*>(tree_rows.data),
                             static_cast<std::uint32_t*>(tree_masks.data),
                             static_cast<std::int32_t*>(valid_columns.data),
                             candidate_ids.ne[1],
                             shape.nodes,
                             shape.paths,
                             shape.transport};
    CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(32), 0, stream},
                                    selector_tree_build_kernel<false>, tree_args,
                                    static_cast<const float*>(edges)));
    CUDA_CHECK(cudaGetLastError());
}

void candidate_tree_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                           const Tensor& base_positions, const Tensor& current_extents,
                           const SamplingConfig* configs, SpeculativeTreeShape shape,
                           Tensor& drafts, Tensor& column_candidates, Tensor& proposal_q,
                           Tensor& tree_rows, Tensor& tree_masks, Tensor& valid_columns,
                           cudaStream_t stream) {
    const TreeArgs tree_args{static_cast<const std::int32_t*>(candidate_ids.data),
                             static_cast<const std::int32_t*>(base_positions.data),
                             static_cast<const std::int32_t*>(current_extents.data),
                             configs,
                             static_cast<std::int32_t*>(drafts.data),
                             static_cast<std::int32_t*>(column_candidates.data),
                             static_cast<float*>(proposal_q.data),
                             static_cast<SpeculativeTreeRow*>(tree_rows.data),
                             static_cast<std::uint32_t*>(tree_masks.data),
                             static_cast<std::int32_t*>(valid_columns.data),
                             candidate_ids.ne[1],
                             shape.nodes,
                             shape.paths,
                             shape.transport};
    CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(32), 0, stream},
                                    selector_tree_build_kernel<true>, tree_args,
                                    static_cast<const float*>(unary_scores.data)));
    CUDA_CHECK(cudaGetLastError());
}

void chain_tree_launch(const Tensor& candidate_ids, const Tensor& laws, const Tensor& scores,
                       const Tensor& base_positions, const Tensor& current_extents,
                       const SamplingConfig* configs, SpeculativeTreeShape shape, Tensor& drafts,
                       Tensor& column_candidates, Tensor& proposal_q, Tensor& tree_rows,
                       Tensor& tree_masks, Tensor& valid_columns, cudaStream_t stream) {
    const ChainTreeArgs args{{static_cast<const std::int32_t*>(candidate_ids.data),
                              static_cast<const std::int32_t*>(base_positions.data),
                              static_cast<const std::int32_t*>(current_extents.data),
                              configs,
                              static_cast<std::int32_t*>(drafts.data),
                              static_cast<std::int32_t*>(column_candidates.data),
                              static_cast<float*>(proposal_q.data),
                              static_cast<SpeculativeTreeRow*>(tree_rows.data),
                              static_cast<std::uint32_t*>(tree_masks.data),
                              static_cast<std::int32_t*>(valid_columns.data),
                              candidate_ids.ne[1],
                              shape.nodes,
                              shape.paths,
                              false}, // a proposed chain was drawn before its siblings: no transport
                             static_cast<const float*>(laws.data),
                             static_cast<const float*>(scores.data),
                             scores.ne[0]};
    CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(32), 0, stream},
                                    chain_tree_build_kernel, args));
    CUDA_CHECK(cudaGetLastError());
}

void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, Tensor* confidence,
                                    const SelectorWorkspace& workspace, cudaStream_t stream) {
    const DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                          static_cast<const float*>(unary_scores.data),
                          static_cast<const __nv_bfloat16*>(projected_hidden.data),
                          static_cast<const std::int32_t*>(anchors.data),
                          static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
                          static_cast<const __nv_bfloat16*>(successor_codebook.data),
                          static_cast<const std::int32_t*>(base_positions.data),
                          configs,
                          static_cast<std::int32_t*>(drafts.data),
                          static_cast<float*>(proposal_q.data),
                          candidate_ids.ne[1],
                          confidence != nullptr ? static_cast<float*>(confidence->data) : nullptr};
    if (route == SelectorRoute::Direct) {
        CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(512), 0, stream},
                                        selector_walk_kernel, args));
    } else {
        auto* edges = static_cast<float*>(workspace.edges.data);
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(args.steps * candidate_ids.ne[2], kCandidates), dim3(512), 0, stream},
            selector_lattice_kernel, args, edges));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(32), 0, stream},
                                        selector_lattice_walk_kernel, args, edges));
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace infernix::ops::detail
