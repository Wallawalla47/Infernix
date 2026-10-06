// QSA block selection (docs/maintainer/qwen3_8-flash-next-design.md §8.10, §19.3.6 item 2, steps
// Q1/Q2): the production selection, ops::detail::qsa_select, against three oracles:
//   - the one-CTA-per-column kernel it replaced, kept below verbatim as the reference: counts and
//     selected ids bitwise equal in every case, FP32 scores bitwise equal where both expose them;
//   - the exact top-k of the production kernel's own FP32 scores: the selected blocks are the
//     `top` highest scores, equal scores taken in ascending block order, written ascending;
//   - an FP64 oracle of the scores from the same BF16 inputs: FP32 scores within an accumulation
//     bound, and no unselected block above a selected one in FP64 beyond both bounds.
// Cases: 0.5K-131K blocks per column; both sides of the 512-block budget (dense columns write
// count -1), also inside one row; widths 1-1024 and 1-9 rows, groups beyond 128 columns; index
// heads 1-4; ratios 4, 8, 64 and 128 (a pooled key across two pages); exact ties across the
// threshold, zero-score ties, bins larger than the candidate buffer, near ties; and CUDA Graph
// capture replayed with new positions over garbage scratch.
#include "ninfer/ops/qsa.h"
#include "core/paged_kv_cache.h"
#include "ops/host_parallel.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/op_tester.h"
#include "ops/qsa/qsa_select.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

using ninfer::DeviceBuffer;
using ninfer::DType;
using ninfer::kPagedKVPageSize;
using ninfer::Tensor;
using ninfer::ops::kPagedKVPageMask;
using ninfer::ops::kPagedKVPageShift;
using ninfer::ops::QsaBatch;
using ninfer::ops::QsaGeometry;
using ninfer::test::cuda_check;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// ------------------------------------------------------------------------------------ reference
// The selection kernel of qsa.cu before step Q1/Q2 (commit 85fa90b5c), verbatim: one CTA per
// column scores every block (one warp per block), then an 8-bit radix select over the scores and
// an ascending scan. Groups of 32 columns share its score scratch.
namespace reference {

using bf16 = __nv_bfloat16;

constexpr int kIndexDim      = 128;
constexpr int kSelectThreads = 512;
constexpr int kSelectGroup   = 32; // columns whose scores share one scratch pass

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFU, v, o); }
    return v;
}

__device__ __forceinline__ std::int64_t pooled_offset(const std::int32_t* table, int position,
                                                      int slot_width, int lane) {
    const int page = table[position >> kPagedKVPageShift];
    return static_cast<std::int64_t>(slot_width) * kPagedKVPageSize * page +
           static_cast<std::int64_t>(slot_width) * (position & kPagedKVPageMask) + lane;
}

__device__ __forceinline__ std::uint32_t sortable(float x) {
    const std::uint32_t u = __float_as_uint(x);
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}

// One CTA per column of a group: block scores into scratch, then the exact top-k by radix select,
// written in ascending block order. Columns that select every block write count = -1 (dense).
__global__ void __launch_bounds__(kSelectThreads)
    select_kernel(const bf16* __restrict__ index_q, int index_heads, const bf16* __restrict__ pooled,
                  const std::int32_t* __restrict__ tables, int table_stride,
                  const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ positions,
                  int width, int ratio, int top_blocks, int column_begin, float* __restrict__ scores,
                  int score_stride, std::int32_t* __restrict__ selected,
                  std::int32_t* __restrict__ counts) {
    const int t = column_begin + blockIdx.x;
    const int p = positions[t];
    const int blocks = (p + 1) / ratio;
    if (blocks <= top_blocks) {
        if (threadIdx.x == 0) { counts[t] = -1; }
        return;
    }
    __shared__ float q[4 * kIndexDim];
    __shared__ unsigned histogram[256];
    __shared__ std::uint32_t prefix_shared, mask_shared;
    __shared__ int remaining_shared, scan[kSelectThreads];
    for (int i = threadIdx.x; i < index_heads * kIndexDim; i += blockDim.x) {
        q[i] = __bfloat162float(index_q[static_cast<std::size_t>(t) * index_heads * kIndexDim + i]);
    }
    __syncthreads();
    const int sequence        = t / width;
    const std::int32_t* table = tables + static_cast<std::int64_t>(table_rows[sequence]) * table_stride;
    const int slot_width      = kIndexDim / ratio;
    float* column_scores      = scores + static_cast<std::size_t>(blockIdx.x) * score_stride;
    const float scale         = rsqrtf(static_cast<float>(kIndexDim));
    // Scores: one warp per block.
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
    for (int b = warp; b < blocks; b += warps) {
        float key[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = lane + 32 * j;
            key[j] = __bfloat162float(pooled[pooled_offset(table, ratio * b + d / slot_width, slot_width,
                                                           d % slot_width)]);
        }
        float score = 0.0F;
        for (int h = 0; h < index_heads; ++h) {
            float dot = 0.0F;
#pragma unroll
            for (int j = 0; j < 4; ++j) { dot += q[h * kIndexDim + lane + 32 * j] * key[j]; }
            score += fmaxf(warp_sum(dot), 0.0F);
        }
        if (lane == 0) { column_scores[b] = score * scale; }
    }
    __syncthreads();
    // Radix select of the top_blocks-th largest key, eight bits at a time.
    if (threadIdx.x == 0) {
        prefix_shared    = 0;
        mask_shared      = 0;
        remaining_shared = top_blocks;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) { histogram[i] = 0; }
        __syncthreads();
        const std::uint32_t prefix = prefix_shared, mask = mask_shared;
        for (int b = threadIdx.x; b < blocks; b += blockDim.x) {
            const std::uint32_t key = sortable(column_scores[b]);
            if ((key & mask) == prefix) { atomicAdd(&histogram[(key >> shift) & 255U], 1U); }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int remaining = remaining_shared;
            for (int digit = 255; digit >= 0; --digit) {
                const int count = static_cast<int>(histogram[digit]);
                if (count >= remaining) {
                    prefix_shared = prefix | (static_cast<std::uint32_t>(digit) << shift);
                    mask_shared   = mask | (255U << shift);
                    break;
                }
                remaining -= count;
            }
            remaining_shared = remaining; // how many of the threshold value to take
        }
        __syncthreads();
    }
    const std::uint32_t threshold = prefix_shared;
    const int take_equal          = remaining_shared;
    // Ascending block order: everything above the threshold, and the first take_equal ties.
    int out = 0, equal_seen = 0;
    for (int begin = 0; begin < blocks; begin += blockDim.x) {
        const int b = begin + threadIdx.x;
        const std::uint32_t key = b < blocks ? sortable(column_scores[b]) : 0;
        const int equal = b < blocks && key == threshold ? 1 : 0;
        scan[threadIdx.x] = equal;
        __syncthreads();
        for (int step = 1; step < blockDim.x; step <<= 1) {
            const int v = threadIdx.x >= step ? scan[threadIdx.x - step] : 0;
            __syncthreads();
            scan[threadIdx.x] += v;
            __syncthreads();
        }
        const int equal_rank = equal_seen + scan[threadIdx.x] - equal;
        const int total_eq   = scan[blockDim.x - 1];
        __syncthreads();
        const int take = b < blocks && (key > threshold || (equal && equal_rank < take_equal)) ? 1 : 0;
        scan[threadIdx.x] = take;
        __syncthreads();
        for (int step = 1; step < blockDim.x; step <<= 1) {
            const int v = threadIdx.x >= step ? scan[threadIdx.x - step] : 0;
            __syncthreads();
            scan[threadIdx.x] += v;
            __syncthreads();
        }
        if (take) { selected[static_cast<std::size_t>(t) * top_blocks + out + scan[threadIdx.x] - 1] = b; }
        const int taken = scan[blockDim.x - 1];
        __syncthreads();
        out += taken;
        equal_seen += total_eq;
    }
    if (threadIdx.x == 0) { counts[t] = out; }
}

// The replaced launch loop of qsa_attention. scores: kSelectGroup x (max_context / ratio + 1).
void select(const Tensor& index_q, const Tensor& pooled, const QsaBatch& batch, const QsaGeometry& g,
            int max_context, float* scores, std::int32_t* selected, std::int32_t* counts, cudaStream_t stream) {
    const int columns    = batch.batch * batch.width;
    const int top_blocks = g.budget / g.ratio;
    for (int begin = 0; begin < columns; begin += kSelectGroup) {
        const int n = std::min(kSelectGroup, columns - begin);
        select_kernel<<<n, kSelectThreads, 0, stream>>>(
            static_cast<const bf16*>(index_q.data), g.index_heads, static_cast<const bf16*>(pooled.data),
            static_cast<const std::int32_t*>(batch.block_tables.data), static_cast<int>(batch.block_tables.ne[0]),
            static_cast<const std::int32_t*>(batch.table_rows.data),
            static_cast<const std::int32_t*>(batch.positions.data), batch.width, g.ratio, top_blocks, begin, scores,
            max_context / g.ratio + 1, selected, counts);
        cuda_check(cudaGetLastError(), "reference select");
    }
}

} // namespace reference

// -------------------------------------------------------------------------------------- fixture

constexpr int kDi = 128;

enum class Keys {
    Random,       // independent N(0, 1) keys
    Ties,         // block b repeats key b % period: equal scores across the threshold
    ZeroHeavy,    // `extra` blocks score > 0, every other block exactly 0 (all heads' dots < 0)
    Concentrated, // one key plus 2 % noise: the threshold bin holds (nearly) every block
    NearTies,     // block 2i+1 repeats block 2i with one element one BF16 ulp up
};

struct Case {
    std::string name;
    int index_heads = 4, ratio = 4, budget = 2048;
    int batch = 1, width = 1;
    std::vector<int> first; // first position of each row; the row's columns are first..first+width-1
    Keys keys        = Keys::Random;
    int extra        = 0;   // Ties: period; ZeroHeavy: positive blocks per row
    int max_context  = 262144 + 1024;
    std::uint32_t seed = 1;
};

QsaGeometry geometry_of(const Case& c) {
    return {.heads = 24, .kv_heads = 2, .head_dim = 256, .index_heads = c.index_heads, .index_head_dim = kDi,
            .rotary_dim = 64, .budget = c.budget, .ratio = c.ratio, .theta = 1.0e7F, .eps = 1.0e-6F};
}

std::uint16_t to_bf16(float f) { return ninfer::test::f32_to_bf16(f); }
float from_bf16(std::uint16_t h) { return ninfer::test::bf16_to_f32(h); }

struct Fixture {
    QsaGeometry g{};
    int batch = 0, width = 0, columns = 0, max_context = 0, top = 0;
    int pages_per_row = 0;
    std::vector<int> row_blocks;          // blocks each row's keys cover (the last column's count)
    std::vector<std::vector<float>> keys; // per row: [row_blocks, 128], BF16 values
    std::vector<float> q;                 // [columns, index_heads, 128], BF16 values
    std::vector<std::int32_t> positions, table_rows, tables, tail_slots;
    DeviceBuffer d_q, d_pooled, d_positions, d_table_rows, d_tables, d_tail_slots;

    QsaBatch batch_view() const {
        return {.block_tables = Tensor(d_tables.p, DType::I32, {pages_per_row, batch}),
                .table_rows   = Tensor(d_table_rows.p, DType::I32, {batch}),
                .positions    = Tensor(d_positions.p, DType::I32, {columns}),
                .tail_slots   = Tensor(d_tail_slots.p, DType::I32, {batch}),
                .batch        = batch,
                .width        = width};
    }
    Tensor index_q() const { return Tensor(d_q.p, DType::BF16, {kDi, g.index_heads, columns}); }
    Tensor pooled() const {
        return Tensor(d_pooled.p, DType::BF16, {kDi / g.ratio, kPagedKVPageSize, 1, pages_per_row * batch});
    }
    void set_positions(const std::vector<int>& first) {
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < width; ++j) { positions[static_cast<std::size_t>(b) * width + j] = first[b] + j; }
        }
    }
};

std::vector<float> random_vector(std::mt19937& rng, int n, float mean_scale, const std::vector<float>* base,
                                 float noise) {
    std::normal_distribution<float> d(0.0F, 1.0F);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        const float x = base != nullptr ? mean_scale * (*base)[i] + noise * d(rng) : d(rng);
        v[i]          = from_bf16(to_bf16(x));
    }
    return v;
}

// `key_first`: per row, the first position whose keys the fixture must hold (the largest the row
// is used with); `first`: the positions the case starts with.
Fixture build(const Case& c, const std::vector<int>& key_first) {
    Fixture f;
    f.g           = geometry_of(c);
    f.batch       = c.batch;
    f.width       = c.width;
    f.columns     = c.batch * c.width;
    f.max_context = c.max_context;
    f.top         = c.budget / c.ratio;
    std::mt19937 rng(c.seed);
    // Queries; ZeroHeavy aligns every head with one direction u.
    const std::vector<float> u = random_vector(rng, kDi, 0, nullptr, 0);
    f.q.reserve(static_cast<std::size_t>(f.columns) * c.index_heads * kDi);
    for (int t = 0; t < f.columns * c.index_heads; ++t) {
        const auto v = c.keys == Keys::ZeroHeavy ? random_vector(rng, kDi, 1.0F, &u, 0.1F)
                                                 : random_vector(rng, kDi, 0, nullptr, 0);
        f.q.insert(f.q.end(), v.begin(), v.end());
    }
    // Keys per row.
    f.row_blocks.resize(c.batch);
    f.keys.resize(c.batch);
    int max_pages = 1;
    for (int b = 0; b < c.batch; ++b) {
        const int blocks = (key_first[b] + c.width) / c.ratio;
        f.row_blocks[b]  = blocks;
        max_pages        = std::max(max_pages, (blocks * c.ratio + kPagedKVPageSize - 1) / kPagedKVPageSize);
        auto& k          = f.keys[b];
        k.resize(static_cast<std::size_t>(blocks) * kDi);
        std::vector<std::vector<float>> period_keys;
        std::vector<float> base = random_vector(rng, kDi, 0, nullptr, 0);
        std::vector<char> positive(blocks, 0);
        if (c.keys == Keys::ZeroHeavy) {
            std::vector<int> ids(blocks);
            for (int i = 0; i < blocks; ++i) { ids[i] = i; }
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int i = 0; i < std::min(c.extra, blocks); ++i) { positive[ids[i]] = 1; }
        }
        if (c.keys == Keys::Ties) {
            for (int i = 0; i < c.extra; ++i) { period_keys.push_back(random_vector(rng, kDi, 0, nullptr, 0)); }
        }
        for (int blk = 0; blk < blocks; ++blk) {
            std::vector<float> v;
            switch (c.keys) {
            case Keys::Random: v = random_vector(rng, kDi, 0, nullptr, 0); break;
            case Keys::Ties: v = period_keys[blk % c.extra]; break;
            case Keys::ZeroHeavy: v = random_vector(rng, kDi, positive[blk] ? 1.0F : -1.0F, &u, 0.3F); break;
            case Keys::Concentrated: v = random_vector(rng, kDi, 1.0F, &base, 0.02F); break;
            case Keys::NearTies:
                if (blk % 2 == 0) {
                    v = random_vector(rng, kDi, 0, nullptr, 0);
                } else {
                    v = std::vector<float>(k.begin() + static_cast<std::ptrdiff_t>(blk - 1) * kDi,
                                           k.begin() + static_cast<std::ptrdiff_t>(blk) * kDi);
                    const int e = (blk / 2) % kDi;
                    v[e]        = from_bf16(static_cast<std::uint16_t>(to_bf16(v[e]) + 1));
                }
                break;
            }
            std::copy(v.begin(), v.end(), k.begin() + static_cast<std::ptrdiff_t>(blk) * kDi);
        }
    }
    // Paged layout: each row's pages are a shuffled run of physical pages; the rows use the table
    // rows in reverse order.
    f.pages_per_row = max_pages + 1;
    const int total_pages = f.pages_per_row * c.batch;
    std::vector<int> physical(total_pages);
    for (int i = 0; i < total_pages; ++i) { physical[i] = i; }
    std::shuffle(physical.begin(), physical.end(), rng);
    f.tables.assign(static_cast<std::size_t>(total_pages), 0);
    for (int i = 0; i < total_pages; ++i) { f.tables[i] = physical[i]; }
    f.table_rows.resize(c.batch);
    f.tail_slots.resize(c.batch);
    for (int b = 0; b < c.batch; ++b) {
        f.table_rows[b] = c.batch - 1 - b;
        f.tail_slots[b] = b;
    }
    const int slot_width = kDi / c.ratio;
    std::vector<std::uint16_t> pooled(static_cast<std::size_t>(total_pages) * kPagedKVPageSize * slot_width, 0);
    for (int b = 0; b < c.batch; ++b) {
        const int* table = f.tables.data() + static_cast<std::size_t>(f.table_rows[b]) * f.pages_per_row;
        for (int blk = 0; blk < f.row_blocks[b]; ++blk) {
            for (int d = 0; d < kDi; ++d) {
                const int token = c.ratio * blk + d / slot_width;
                const std::size_t at =
                    static_cast<std::size_t>(slot_width) * kPagedKVPageSize * table[token / kPagedKVPageSize] +
                    static_cast<std::size_t>(slot_width) * (token % kPagedKVPageSize) + d % slot_width;
                pooled[at] = to_bf16(f.keys[b][static_cast<std::size_t>(blk) * kDi + d]);
            }
        }
    }
    std::vector<std::uint16_t> q16(f.q.size());
    for (std::size_t i = 0; i < f.q.size(); ++i) { q16[i] = to_bf16(f.q[i]); }
    f.positions.resize(f.columns);
    f.set_positions(c.first);
    f.d_q          = DeviceBuffer(q16.size() * 2);
    f.d_pooled     = DeviceBuffer(pooled.size() * 2);
    f.d_positions  = DeviceBuffer(f.positions.size() * 4);
    f.d_table_rows = DeviceBuffer(f.table_rows.size() * 4);
    f.d_tables     = DeviceBuffer(f.tables.size() * 4);
    f.d_tail_slots = DeviceBuffer(f.tail_slots.size() * 4);
    f.d_q.copy_from_host(q16.data(), q16.size() * 2);
    f.d_pooled.copy_from_host(pooled.data(), pooled.size() * 2);
    f.d_positions.copy_from_host(f.positions.data(), f.positions.size() * 4);
    f.d_table_rows.copy_from_host(f.table_rows.data(), f.table_rows.size() * 4);
    f.d_tables.copy_from_host(f.tables.data(), f.tables.size() * 4);
    f.d_tail_slots.copy_from_host(f.tail_slots.data(), f.tail_slots.size() * 4);
    return f;
}

int blocks_of(const Fixture& f, int t) { return (f.positions[t] + 1) / f.g.ratio; }

// ---------------------------------------------------------------------------------------- oracle

struct Fp64Scores {
    std::vector<double> score, bound; // per block
};

// score_b = (1/sqrt(Di)) sum_h relu(<q_h, k_b>) in FP64; bound_b = 2^-19 (1/sqrt(Di)) sum |q_hd k_bd|,
// above the FP32 accumulation error (4-term chains, a 5-level tree, 4 head additions, the scale).
Fp64Scores fp64_scores(const Fixture& f, int t) {
    const int n = blocks_of(f, t), H = f.g.index_heads, row = t / f.width;
    Fp64Scores out;
    out.score.resize(n);
    out.bound.resize(n);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kDi));
    const float* q     = f.q.data() + static_cast<std::size_t>(t) * H * kDi;
    ninfer::test::parallel_ranges(n, ninfer::test::threads_for_rows(n), [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t b = begin; b < end; ++b) {
            const float* k = f.keys[row].data() + b * kDi;
            double s = 0, a = 0;
            for (int h = 0; h < H; ++h) {
                double dot = 0;
                for (int d = 0; d < kDi; ++d) {
                    const double p = static_cast<double>(q[h * kDi + d]) * k[d];
                    dot += p;
                    a += std::fabs(p);
                }
                s += std::max(dot, 0.0);
            }
            out.score[b] = s * scale;
            out.bound[b] = std::ldexp(a * scale, -19);
        }
    });
    return out;
}

// ------------------------------------------------------------------------------------------ runs

struct Selection {
    std::vector<std::int32_t> selected, counts;
    std::vector<float> scores; // [group columns, stride] of the last group (empty if not kept)
    int score_stride = 0, score_begin = 0;
};

Selection run_reference(const Fixture& f, cudaStream_t stream) {
    const int stride = f.max_context / f.g.ratio + 1;
    DeviceBuffer scores(sizeof(float) * reference::kSelectGroup * static_cast<std::size_t>(stride));
    DeviceBuffer selected(sizeof(std::int32_t) * static_cast<std::size_t>(f.columns) * f.top);
    DeviceBuffer counts(sizeof(std::int32_t) * static_cast<std::size_t>(f.columns));
    selected.fill(0x7F);
    counts.fill(0x7F);
    reference::select(f.index_q(), f.pooled(), f.batch_view(), f.g, f.max_context, static_cast<float*>(scores.p),
                      static_cast<std::int32_t*>(selected.p), static_cast<std::int32_t*>(counts.p), stream);
    cuda_check(cudaStreamSynchronize(stream), "reference select");
    Selection out;
    out.selected.resize(static_cast<std::size_t>(f.columns) * f.top);
    out.counts.resize(f.columns);
    selected.copy_to_host(out.selected.data(), out.selected.size() * 4);
    counts.copy_to_host(out.counts.data(), out.counts.size() * 4);
    out.score_begin  = (f.columns - 1) / reference::kSelectGroup * reference::kSelectGroup;
    out.score_stride = stride;
    out.scores.resize(static_cast<std::size_t>(f.columns - out.score_begin) * stride);
    scores.copy_to_host(out.scores.data(), out.scores.size() * 4);
    return out;
}

struct ProductionBuffers {
    ninfer::test::GuardedDeviceBuffer scratch, selected, counts;
    ProductionBuffers(const Fixture& f)
        : scratch(ninfer::ops::detail::qsa_select_scratch_bytes(f.g, f.columns, f.max_context)),
          selected(sizeof(std::int32_t) * static_cast<std::size_t>(f.columns) * f.top),
          counts(sizeof(std::int32_t) * static_cast<std::size_t>(f.columns)) {}
};

void launch_production(const Fixture& f, ProductionBuffers& buffers, cudaStream_t stream) {
    ninfer::ops::detail::qsa_select(f.index_q(), f.pooled(), f.batch_view(), f.g, f.max_context,
                                    buffers.scratch.data(), buffers.scratch.bytes(),
                                    {static_cast<std::int32_t*>(buffers.selected.data()),
                                     static_cast<std::int32_t*>(buffers.counts.data())},
                                    stream);
}

Selection read_production(const Fixture& f, ProductionBuffers& buffers, const std::string& tag) {
    Selection out;
    out.selected.resize(static_cast<std::size_t>(f.columns) * f.top);
    out.counts.resize(f.columns);
    buffers.selected.copy_to_host(out.selected.data(), out.selected.size() * 4);
    buffers.counts.copy_to_host(out.counts.data(), out.counts.size() * 4);
    const auto scores = ninfer::ops::detail::qsa_select_scores(buffers.scratch.data(), f.g, f.max_context);
    const int group   = ninfer::ops::detail::kQsaSelectGroupColumns;
    out.score_begin   = (f.columns - 1) / group * group;
    out.score_stride  = scores.stride;
    out.scores.resize(static_cast<std::size_t>(f.columns - out.score_begin) * scores.stride);
    cuda_check(cudaMemcpy(out.scores.data(), scores.scores, out.scores.size() * 4, cudaMemcpyDeviceToHost),
               "copy production scores");
    check(buffers.scratch.verify_guards(tag + " scratch") == 0, tag + ": scratch guards intact");
    check(buffers.selected.verify_guards(tag + " selected") == 0, tag + ": selected guards intact");
    check(buffers.counts.verify_guards(tag + " counts") == 0, tag + ": counts guards intact");
    return out;
}

const float* column_scores(const Selection& s, int t) {
    if (t < s.score_begin) { return nullptr; }
    return s.scores.data() + static_cast<std::size_t>(t - s.score_begin) * s.score_stride;
}

// Production against the reference, bit for bit, and against the exact top-k of its own scores.
void compare(const Fixture& f, const Selection& ref, const Selection& got, const std::string& tag) {
    int mismatched_columns = 0;
    for (int t = 0; t < f.columns; ++t) {
        const int n = blocks_of(f, t);
        const std::string where = tag + " column " + std::to_string(t) + " (" + std::to_string(n) + " blocks)";
        const int expected_count = n <= f.top ? -1 : f.top;
        check(ref.counts[t] == expected_count, where + ": reference count " + std::to_string(ref.counts[t]));
        if (got.counts[t] != ref.counts[t]) {
            check(false, where + ": count " + std::to_string(got.counts[t]) + " vs reference " +
                             std::to_string(ref.counts[t]));
            continue;
        }
        if (got.counts[t] < 0) { continue; }
        const std::int32_t* a = ref.selected.data() + static_cast<std::size_t>(t) * f.top;
        const std::int32_t* b = got.selected.data() + static_cast<std::size_t>(t) * f.top;
        if (!std::equal(a, a + f.top, b)) {
            if (++mismatched_columns <= 4) {
                int i = 0;
                while (a[i] == b[i]) { ++i; }
                check(false, where + ": selected[" + std::to_string(i) + "] = " + std::to_string(b[i]) +
                                 ", reference " + std::to_string(a[i]));
            }
        }
        // Scores bitwise against the reference where both kept them.
        const float* rs = column_scores(ref, t);
        const float* gs = column_scores(got, t);
        if (rs != nullptr && gs != nullptr && std::memcmp(rs, gs, sizeof(float) * n) != 0) {
            int i = 0;
            while (std::memcmp(rs + i, gs + i, 4) == 0) { ++i; }
            check(false, where + ": score of block " + std::to_string(i) + " " + std::to_string(gs[i]) +
                             " differs from the reference's " + std::to_string(rs[i]));
        }
        // The exact top-k (score descending, lower id first) of the production's own scores.
        if (gs != nullptr) {
            bool ascending = true;
            for (int i = 0; i < f.top; ++i) {
                ascending = ascending && b[i] >= 0 && b[i] < n && (i == 0 || b[i] > b[i - 1]);
            }
            check(ascending, where + ": selected ids ascend inside [0, n)");
            if (!ascending) { continue; }
            std::vector<char> in(n, 0);
            float threshold = INFINITY;
            for (int i = 0; i < f.top; ++i) {
                in[b[i]]  = 1;
                threshold = std::min(threshold, gs[b[i]]);
            }
            bool exact = true;
            bool ties_open = true; // blocks at the threshold score are taken in ascending order
            for (int i = 0; i < n && exact; ++i) {
                if (gs[i] > threshold) { exact = in[i] != 0; }
                if (gs[i] < threshold) { exact = in[i] == 0; }
                if (gs[i] == threshold) {
                    if (in[i] && !ties_open) { exact = false; }
                    if (!in[i]) { ties_open = false; }
                }
            }
            check(exact, where + ": selection is the exact top-k of its FP32 scores (ties to the lower id)");
        }
    }
    if (mismatched_columns > 4) {
        check(false, tag + ": " + std::to_string(mismatched_columns) + " columns differ from the reference in all");
    }
}

// FP64: scores within the bound, and no unselected block above a selected one beyond both bounds.
void check_fp64(const Fixture& f, const Selection& got, int t, const Fp64Scores& o, const std::string& tag) {
    const int n = blocks_of(f, t);
    const std::string where = tag + " column " + std::to_string(t) + " FP64";
    if (got.counts[t] < 0) { return; }
    if (const float* gs = column_scores(got, t)) {
        int worst = -1;
        double worst_ratio = 0;
        for (int b = 0; b < n; ++b) {
            const double ratio = std::fabs(gs[b] - o.score[b]) / std::max(o.bound[b], 1e-300);
            if (ratio > worst_ratio) {
                worst_ratio = ratio;
                worst       = b;
            }
        }
        check(worst_ratio <= 1.0, where + ": FP32 score of block " + std::to_string(worst) + " off by " +
                                      std::to_string(worst_ratio) + " x its bound");
    }
    const std::int32_t* ids = got.selected.data() + static_cast<std::size_t>(t) * f.top;
    std::vector<char> in(n, 0);
    for (int i = 0; i < f.top; ++i) {
        if (ids[i] >= 0 && ids[i] < n) { in[ids[i]] = 1; }
    }
    double lowest_selected = INFINITY, highest_unselected = -INFINITY;
    for (int b = 0; b < n; ++b) {
        if (in[b]) {
            lowest_selected = std::min(lowest_selected, o.score[b] + o.bound[b]);
        } else {
            highest_unselected = std::max(highest_unselected, o.score[b] - o.bound[b]);
        }
    }
    check(highest_unselected <= lowest_selected, where + ": an unselected block outranks a selected one beyond "
                                                         "rounding");
}

std::vector<int> fp64_columns(const Fixture& f) {
    std::vector<int> out;
    const int want = std::min(f.columns, 8);
    for (int i = 0; i < want; ++i) {
        out.push_back(want == 1 ? 0 : static_cast<int>(static_cast<std::int64_t>(f.columns - 1) * i / (want - 1)));
    }
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

void run_case(const Case& c) {
    Fixture f = build(c, c.first);
    const Selection ref = run_reference(f, nullptr);
    std::vector<std::pair<int, Fp64Scores>> oracle;
    for (const int t : fp64_columns(f)) { oracle.emplace_back(t, fp64_scores(f, t)); }
    const std::string& tag = c.name;
    ProductionBuffers buffers(f);
    buffers.scratch.fill(0xA5);
    buffers.selected.fill(0x7F);
    buffers.counts.fill(0x7F);
    launch_production(f, buffers, nullptr);
    cuda_check(cudaStreamSynchronize(nullptr), "production select");
    const Selection got = read_production(f, buffers, tag);
    compare(f, ref, got, tag);
    for (const auto& [t, o] : oracle) { check_fp64(f, got, t, o, tag); }
    std::printf("  %-44s %5d columns, %6d..%6d blocks\n", c.name.c_str(), f.columns, blocks_of(f, 0),
                blocks_of(f, f.columns - 1));
}

// One capture, replayed with new positions over garbage scratch and outputs.
void run_graph_case() {
    Case c;
    c.name  = "graph B=2 W=5";
    c.batch = 2;
    c.width = 5;
    c.first = {100, 200};
    c.seed  = 77;
    const std::vector<std::vector<int>> replays = {
        {100, 200}, {2045, 9000}, {40000, 2047}, {130000, 60000}, {2051, 2046}, {9000, 130000}};
    Fixture f = build(c, {130000, 130000});
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "graph stream");
    ProductionBuffers buffers(f);
    cudaGraph_t graph     = nullptr;
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    launch_production(f, buffers, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (std::size_t r = 0; r < replays.size(); ++r) {
        const std::string tag = c.name + " replay " + std::to_string(r);
        f.set_positions(replays[r]);
        cuda_check(cudaMemcpyAsync(f.d_positions.p, f.positions.data(), f.positions.size() * 4,
                                   cudaMemcpyHostToDevice, stream),
                   "upload positions");
        cuda_check(cudaMemsetAsync(buffers.scratch.data(), 0x5A + static_cast<int>(r), buffers.scratch.bytes(), stream),
                   "garbage scratch");
        cuda_check(cudaMemsetAsync(buffers.selected.data(), 0x7F, buffers.selected.bytes(), stream),
                   "garbage selected");
        cuda_check(cudaMemsetAsync(buffers.counts.data(), 0x7F, buffers.counts.bytes(), stream), "garbage counts");
        cuda_check(cudaGraphLaunch(exec, stream), "graph launch");
        cuda_check(cudaStreamSynchronize(stream), "graph replay");
        const Selection got = read_production(f, buffers, tag);
        const Selection ref = run_reference(f, stream);
        compare(f, ref, got, tag);
    }
    cuda_check(cudaGraphExecDestroy(exec), "destroy graph exec");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    cuda_check(cudaStreamDestroy(stream), "destroy graph stream");
    std::printf("  %-44s %zu replays\n", c.name.c_str(), replays.size());
}

std::vector<Case> cases() {
    std::vector<Case> out;
    auto add = [&](std::string name, int batch, int width, std::vector<int> first, Keys keys = Keys::Random,
                   int extra = 0) {
        Case c;
        c.name  = std::move(name);
        c.batch = batch;
        c.width = width;
        c.first = std::move(first);
        c.keys  = keys;
        c.extra = extra;
        c.seed  = static_cast<std::uint32_t>(out.size() * 7919 + 13);
        out.push_back(std::move(c));
        return &out.back();
    };
    // Budget boundary (top 512 blocks = 2,048 tokens): 511, 512 dense; 513, 514 selected.
    add("boundary W=1 n=511", 1, 1, {2044 - 1});
    add("boundary W=1 n=512", 1, 1, {2048 - 1});
    add("boundary W=1 n=513", 1, 1, {2052 - 1});
    add("boundary W=1 n=514", 1, 1, {2056 - 1});
    add("boundary straddle W=16", 1, 16, {2040});
    add("boundary rows B=4 W=3", 4, 3, {2045, 2049, 2060, 100});
    // Decode and verify shapes over 0.5K-65K blocks.
    add("decode B=1 W=1 2K blocks", 1, 1, {8191});
    add("decode B=1 W=1 8K blocks", 1, 1, {32767});
    add("decode B=1 W=1 32K blocks", 1, 1, {131071});
    add("decode B=1 W=1 65K blocks", 1, 1, {262143});
    add("decode B=8 W=1 mixed contexts", 8, 1, {2399, 8191, 19999, 32767, 80001, 131071, 160003, 262143});
    add("verify B=1 W=4 33K blocks", 1, 4, {131500});
    add("verify B=3 W=5 mixed", 3, 5, {32000, 1000, 262000});
    add("verify B=8 W=16 (128 columns)", 8, 16, {4000, 9000, 16000, 2041, 33000, 20000, 60000, 12000});
    add("verify B=9 W=16 (two groups)", 9, 16, {3000, 7000, 11000, 2049, 5000, 9000, 13000, 2500, 30000});
    add("verify B=2 W=13", 2, 13, {50000, 6000});
    // Prefill groups (128 columns each).
    add("prefill W=1024 across the budget", 1, 1024, {1500});
    add("prefill W=300 at 3K blocks", 1, 300, {12000});
    add("prefill W=200 at 30K blocks", 1, 200, {120000});
    // Wide rows (the register-tiled scoring): partial tiles, two rows, the edge key families,
    // fewer index heads, ratio 8, and a full serving chunk deep in the context.
    add("prefill W=33 partial tiles", 1, 33, {9000});
    add("prefill B=2 W=40 two rows", 2, 40, {20000, 6000});
    add("prefill W=64 zero scores at the threshold", 1, 64, {2100}, Keys::ZeroHeavy, 470);
    add("prefill W=48 zero scores, all blocks", 1, 48, {40000}, Keys::ZeroHeavy, 0);
    add("prefill W=40 ties period 97", 1, 40, {16000}, Keys::Ties, 97);
    add("prefill W=36 near ties", 1, 36, {20000}, Keys::NearTies);
    auto* wide_h1 = add("prefill W=48 index heads 1", 1, 48, {12000});
    wide_h1->index_heads = 1;
    auto* wide_h3 = add("prefill W=40 index heads 3", 1, 40, {20000});
    wide_h3->index_heads = 3;
    auto* wide_r8 = add("prefill W=36 ratio 8, budget 512", 1, 36, {12000});
    wide_r8->ratio  = 8;
    wide_r8->budget = 512;
    add("prefill W=4096 at 27K blocks", 1, 4096, {110000});
    // Ties and near ties.
    add("ties period 97, 4K blocks", 2, 3, {16000, 16100}, Keys::Ties, 97);
    add("ties period 3, 1K blocks", 1, 2, {4000}, Keys::Ties, 3);
    add("ties period 701, 40K blocks", 1, 1, {160000}, Keys::Ties, 701);
    add("zero scores at the threshold, 520 blocks", 1, 2, {2078}, Keys::ZeroHeavy, 470);
    add("zero scores, bin beyond the buffer", 1, 2, {80000}, Keys::ZeroHeavy, 300);
    add("zero scores, all blocks", 1, 1, {40000}, Keys::ZeroHeavy, 0);
    add("concentrated, bin beyond the buffer", 1, 1, {120000}, Keys::Concentrated);
    add("concentrated, 6K blocks", 2, 2, {24000, 25000}, Keys::Concentrated);
    add("near ties, 5K blocks", 1, 4, {20000}, Keys::NearTies);
    // Beyond 262K tokens.
    auto* far = add("beyond: 131K blocks", 1, 2, {524286});
    far->max_context = 524288 + 1024;
    // Geometries: index heads 1 and 3; ratios 8 (top 64), 64 (top 32), 128 (blocks span two pages).
    auto* h1 = add("index heads 1", 2, 3, {12000, 3000});
    h1->index_heads = 1;
    auto* h3 = add("index heads 3", 1, 4, {20000});
    h3->index_heads = 3;
    auto* r8 = add("ratio 8, budget 512", 2, 2, {12000, 600});
    r8->ratio  = 8;
    r8->budget = 512;
    auto* r64 = add("ratio 64, budget 2048", 1, 3, {64 * 900});
    r64->ratio = 64;
    auto* r128 = add("ratio 128, budget 1024", 1, 2, {128 * 300});
    r128->ratio  = 128;
    r128->budget = 1024;
    return out;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const Case& c : cases()) { run_case(c); }
        run_graph_case();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all qsa selection checks passed\n");
    return 0;
}
