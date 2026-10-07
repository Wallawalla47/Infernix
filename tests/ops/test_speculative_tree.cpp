// Speculative verification-tree Ops: per-row verify inputs, the lattice tree builder (main chain
// bit-identical to the path walk, best-first priorities and without-replacement draws replayed
// against a host oracle), tree acceptance (exact greedy walk, chain equivalence, recursive
// rejection output distribution), path-parallel GDN convolution and record replay, and
// accepted-path compaction of record planes and of every paged KV storage format.
#include "infernix/ops/candidate_selector.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/speculative_tree.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_storage.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr int kPhysicalRows = 248320;
constexpr int kTokenDomain  = 248077;
constexpr int kCandidates   = 16;
constexpr int kRank         = 256;

int expect(bool condition, const std::string& label) {
    if (condition) return 0;
    std::cerr << label << '\n';
    return 1;
}

template <typename T>
std::vector<T> read_buffer(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), count * sizeof(T));
    return values;
}

template <typename T>
void write_buffer(GuardedDeviceBuffer& buffer, const std::vector<T>& values) {
    buffer.copy_from_host(values.data(), values.size() * sizeof(T));
}

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t i = 0; i < count; ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// ---------------------------------------------------------------------------------------------
// Host tree oracle

// A tree row from its parent array: depth, children linked in column (draw) order, and the
// root-to-leaf paths in leaf column order, each owning its columns no earlier path holds.
ops::SpeculativeTreeRow host_row(const std::vector<int>& parent, int main_depth, bool tree) {
    ops::SpeculativeTreeRow row{};
    const int n    = static_cast<int>(parent.size());
    row.nodes      = n;
    row.main_depth = main_depth;
    row.tree       = tree ? 1 : 0;
    std::array<int, ops::kSpeculativeTreeMaxNodes> last{};
    for (int c = 0; c < ops::kSpeculativeTreeMaxNodes; ++c) {
        row.parent[c]        = static_cast<std::int8_t>(c < n ? parent[c] : -1);
        row.depth[c]         = 0;
        row.first_child[c]   = -1;
        row.next_sibling[c]  = -1;
        row.sibling_index[c] = 0;
        last[c]              = -1;
    }
    for (int c = 1; c < n; ++c) {
        const int p  = parent[c];
        row.depth[c] = static_cast<std::int8_t>(row.depth[p] + 1);
        if (last[p] < 0) {
            row.first_child[p] = static_cast<std::int8_t>(c);
        } else {
            row.next_sibling[last[p]] = static_cast<std::int8_t>(c);
            row.sibling_index[c]      = static_cast<std::int8_t>(row.sibling_index[last[p]] + 1);
        }
        last[p] = c;
    }
    std::uint32_t covered = 0;
    for (int c = 0; c < n; ++c) {
        if (row.first_child[c] >= 0) continue;
        const int path   = row.paths++;
        const int length = row.depth[c] + 1;
        int owned_from   = 0;
        for (int at = c, i = length - 1; at >= 0; at = parent[at], --i) {
            row.path_columns[path][i] = static_cast<std::int8_t>(at);
            if (owned_from == 0 && ((covered >> at) & 1U) != 0U) owned_from = i + 1;
        }
        for (int at = c; at >= 0; at = parent[at]) covered |= 1U << at;
        row.path_length[path] = static_cast<std::int8_t>(length);
        row.owned_from[path]  = static_cast<std::int8_t>(owned_from);
    }
    return row;
}

std::vector<int> chain_parents(int extent) {
    std::vector<int> parent(static_cast<std::size_t>(extent) + 1);
    for (int c = 0; c <= extent; ++c) parent[static_cast<std::size_t>(c)] = c - 1;
    return parent;
}

ops::SpeculativeTreeRow chain_row(int extent, int main_depth) {
    return host_row(chain_parents(extent), main_depth, false);
}

// Main chain 0..k, then side chains of the given lengths hung below the given main columns.
std::vector<int> branched_parents(int k, const std::vector<std::pair<int, int>>& branches) {
    auto parent = chain_parents(k);
    for (const auto& [from, length] : branches) {
        int at = from;
        for (int i = 0; i < length; ++i) {
            parent.push_back(at);
            at = static_cast<int>(parent.size()) - 1;
        }
    }
    return parent;
}

std::vector<std::uint32_t> host_masks(const ops::SpeculativeTreeRow& row, int width) {
    std::vector<std::uint32_t> masks(static_cast<std::size_t>(width));
    for (int c = 0; c < width; ++c) {
        if (c >= row.nodes) {
            masks[static_cast<std::size_t>(c)] = 1U << c;
            continue;
        }
        for (int a = c; a >= 0; a = row.parent[a]) masks[static_cast<std::size_t>(c)] |= 1U << a;
    }
    return masks;
}

bool same_row(const ops::SpeculativeTreeRow& a, const ops::SpeculativeTreeRow& b) {
    if (a.nodes != b.nodes || a.main_depth != b.main_depth || a.tree != b.tree ||
        a.paths != b.paths)
        return false;
    for (int c = 0; c < a.nodes; ++c)
        if (a.parent[c] != b.parent[c] || a.depth[c] != b.depth[c] ||
            a.first_child[c] != b.first_child[c] || a.next_sibling[c] != b.next_sibling[c] ||
            a.sibling_index[c] != b.sibling_index[c])
            return false;
    for (int p = 0; p < a.paths; ++p) {
        if (a.path_length[p] != b.path_length[p] || a.owned_from[p] != b.owned_from[p])
            return false;
        for (int i = 0; i < a.path_length[p]; ++i)
            if (a.path_columns[p][i] != b.path_columns[p][i]) return false;
    }
    return true;
}

GuardedDeviceBuffer upload_rows(const std::vector<ops::SpeculativeTreeRow>& rows) {
    GuardedDeviceBuffer buffer(sizeof(ops::SpeculativeTreeRow) * rows.size());
    buffer.copy_from_host(rows.data(), sizeof(ops::SpeculativeTreeRow) * rows.size());
    return buffer;
}

Tensor rows_tensor(GuardedDeviceBuffer& buffer, int batch) {
    return Tensor(buffer.data(), DType::I32, {ops::kSpeculativeTreeRowWords, batch});
}

// ---------------------------------------------------------------------------------------------
// Verify inputs

int prepare_inputs_case() {
    constexpr int k = 4, width = 10, batch = 4;
    const std::vector<ops::SpeculativeTreeRow> rows{
        host_row(branched_parents(k, {{0, 2}, {0, 2}, {2, 1}}), k, true), // 10 columns
        chain_row(2, k),
        chain_row(0, k),
        host_row(branched_parents(k, {{1, 3}}), k, true), // 8 of 10 columns live
    };
    std::vector<int> anchors{100, 200, 300, 400}, base{10, 64, 1000, 5};
    std::vector<int> drafts(static_cast<std::size_t>(width - 1) * batch);
    for (std::size_t i = 0; i < drafts.size(); ++i) drafts[i] = 5000 + static_cast<int>(i) * 7;
    std::vector<int> rope(static_cast<std::size_t>(width) * batch);
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < width; ++c)
            rope[static_cast<std::size_t>(b) * width + c] = 3000 + 100 * b + c;
    GuardedDeviceBuffer da(batch * 4), dd(drafts.size() * 4), db(batch * 4),
        ids(static_cast<std::size_t>(width) * batch * 4),
        pos(static_cast<std::size_t>(width) * batch * 4), drope(rope.size() * 4);
    auto drows = upload_rows(rows);
    write_buffer(da, anchors);
    write_buffer(dd, drafts);
    write_buffer(db, base);
    write_buffer(drope, rope);
    ids.fill(0xff);
    pos.fill(0xff);
    Tensor ta(da.data(), DType::I32, {batch}), td(dd.data(), DType::I32, {width - 1, batch}),
        tb(db.data(), DType::I32, {batch});
    Tensor tids(ids.data(), DType::I32, {width, batch}),
        tpos(pos.data(), DType::I32, {width, batch}),
        trope(drope.data(), DType::I32, {width, batch});
    ops::speculative_prepare_tree_verify_inputs(ta, td, tb, rows_tensor(drows, batch), tids, tpos,
                                                trope, nullptr);
    cuda_synchronize();
    const auto got_ids  = read_buffer<int>(ids, static_cast<std::size_t>(width) * batch);
    const auto got_pos  = read_buffer<int>(pos, static_cast<std::size_t>(width) * batch);
    const auto got_rope = read_buffer<int>(drope, static_cast<std::size_t>(width) * batch);
    int failures        = 0;
    for (int b = 0; b < batch; ++b) {
        const auto& row = rows[static_cast<std::size_t>(b)];
        for (int c = 0; c < width; ++c) {
            const auto at   = static_cast<std::size_t>(b) * width + c;
            const bool live = c < row.nodes;
            const int id    = c == 0 || !live
                                  ? anchors[static_cast<std::size_t>(b)]
                                  : drafts[static_cast<std::size_t>(b) * (width - 1) + c - 1];
            const int slot  = base[static_cast<std::size_t>(b)] + (live ? c : row.nodes - 1);
            // Tree columns take the anchor's RoPE position plus their depth; chain rows keep the
            // host's positions.
            const int rope_position =
                row.tree ? rope[static_cast<std::size_t>(b) * width] + (live ? row.depth[c] : 0)
                         : rope[at];
            failures += expect(
                got_ids[at] == id && got_pos[at] == slot && got_rope[at] == rope_position,
                "tree verify input row " + std::to_string(b) + " column " + std::to_string(c));
        }
    }
    failures += ids.verify_guards("tree verify ids") + pos.verify_guards("tree verify positions") +
                drope.verify_guards("tree verify rope");
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Compaction

struct PathRow {
    std::vector<int> path; // c_0 = 0, c_1..c_A
};

void encode_paths(const std::vector<PathRow>& rows, int n, std::vector<int>& accepted_path,
                  std::vector<int>& accepted) {
    accepted_path.assign(static_cast<std::size_t>(n) * rows.size(), -1);
    accepted.assign(rows.size(), 0);
    for (std::size_t b = 0; b < rows.size(); ++b) {
        for (std::size_t i = 0; i < rows[b].path.size(); ++i)
            accepted_path[b * static_cast<std::size_t>(n) + i] = rows[b].path[i];
        accepted[b] = static_cast<int>(rows[b].path.size()) - 1;
    }
}

int compact_columns_case(bool with_row_outer) {
    constexpr int n      = 9;
    constexpr int inner  = 40; // 80 bytes per column
    constexpr int width  = 12;
    constexpr int layers = 3;
    const int batch      = 3;
    const int rows_per   = with_row_outer ? 4 : batch;
    const std::vector<int> row_outer{2, 0, 3};
    // Side path from the anchor, the main chain, and a side path below main column 1.
    const std::vector<PathRow> rows{{{0, 5, 6}}, {{0, 1, 2, 3}}, {{0, 1, 7, 8}}};
    std::vector<int> accepted_path, accepted;
    encode_paths(rows, n, accepted_path, accepted);
    const std::size_t elements = static_cast<std::size_t>(inner) * width * layers * rows_per;
    auto data                  = random_bf16(elements, 91U, -1.0F, 1.0F);
    auto expected              = data;
    for (int layer = 0; layer < layers; ++layer)
        for (int b = 0; b < batch; ++b) {
            const int outer =
                layer * rows_per + (with_row_outer ? row_outer[static_cast<std::size_t>(b)] : b);
            const auto& path = rows[static_cast<std::size_t>(b)].path;
            for (std::size_t i = 1; i < path.size(); ++i) {
                if (path[i] == static_cast<int>(i)) continue;
                for (int e = 0; e < inner; ++e)
                    expected[(static_cast<std::size_t>(outer) * width + i) * inner + e] =
                        data[(static_cast<std::size_t>(outer) * width + path[i]) * inner + e];
            }
        }
    GuardedDeviceBuffer dd(elements * 2), dp(accepted_path.size() * 4), da(accepted.size() * 4),
        dr(row_outer.size() * 4);
    write_buffer(dd, data);
    write_buffer(dp, accepted_path);
    write_buffer(da, accepted);
    write_buffer(dr, row_outer);
    Tensor td(dd.data(), DType::BF16, {inner, width, layers * rows_per});
    Tensor tp(dp.data(), DType::I32, {n, batch}), ta(da.data(), DType::I32, {batch});
    Tensor tr = with_row_outer ? Tensor(dr.data(), DType::I32, {batch}) : Tensor{};
    ops::speculative_tree_compact_columns(td, rows_per, tr, tp, ta, nullptr);
    cuda_synchronize();
    const std::string label =
        std::string("tree compact columns") + (with_row_outer ? " (row outer)" : "");
    return verify_exact(label.c_str(), read_buffer<std::uint16_t>(dd, elements), expected) +
           dd.verify_guards(label);
}

// Every stored plane of a paged layer moves with its position: codes and scales of K and V in
// every KV storage format, through a fragmented block table.
int compact_kv_case(KvCacheStorage storage) {
    constexpr int n                   = 9;
    constexpr int heads               = 4;
    constexpr int pages               = 6;
    constexpr int stride              = 3;
    constexpr int page_size           = 64;
    constexpr int head_dim            = 256;
    constexpr int layer_count         = 2;
    const PagedKVStorageLayout layout = paged_kv_storage_layout(storage, head_dim);

    struct PlaneSpec {
        int bytes;
        DType dtype;
        int leading;
    };

    std::vector<PlaneSpec> plane_specs{
        {layout.key.data_leading_extent * static_cast<int>(dtype_size(layout.key.data_dtype)),
         layout.key.data_dtype, layout.key.data_leading_extent},
        {layout.value.data_leading_extent * static_cast<int>(dtype_size(layout.value.data_dtype)),
         layout.value.data_dtype, layout.value.data_leading_extent},
        {layout.key.scale_leading_extent * static_cast<int>(dtype_size(layout.key.scale_dtype)),
         layout.key.scale_dtype, layout.key.scale_leading_extent},
        {layout.value.scale_leading_extent * static_cast<int>(dtype_size(layout.value.scale_dtype)),
         layout.value.scale_dtype, layout.value.scale_leading_extent}};
    const std::vector<int> tables{4, 1, 5, 0, 3, 2}; // [stride, 2 table rows]
    const std::vector<int> table_rows{1, 0};
    const std::vector<PathRow> rows{{{0, 5, 6}}, {{0, 1, 7, 8}}};
    const std::vector<int> bases{60, 120};
    std::vector<int> positions(static_cast<std::size_t>(n) * 2);
    for (int b = 0; b < 2; ++b)
        for (int c = 0; c < n; ++c)
            positions[static_cast<std::size_t>(b) * n + c] = bases[static_cast<std::size_t>(b)] + c;
    std::vector<int> accepted_path, accepted;
    encode_paths(rows, n, accepted_path, accepted);

    GuardedDeviceBuffer dt(tables.size() * 4);
    write_buffer(dt, tables);
    std::vector<std::vector<std::uint8_t>> planes, expected;
    std::vector<GuardedDeviceBuffer> buffers;
    buffers.reserve(layer_count * 4);
    std::vector<PagedKVBatchLayerView> views;
    for (int layer = 0; layer < layer_count; ++layer) {
        PagedKVBatchLayerView view;
        view.block_tables = Tensor(dt.data(), DType::I32, {stride, 2});
        view.head_dim     = head_dim;
        view.num_kv_heads = heads;
        view.storage      = storage;
        for (int p = 0; p < 4; ++p) {
            const PlaneSpec spec  = plane_specs[static_cast<std::size_t>(p)];
            const std::size_t len = static_cast<std::size_t>(pages) * heads * page_size *
                                    static_cast<std::size_t>(std::max(spec.bytes, 1));
            std::vector<std::uint8_t> bytes(spec.bytes == 0 ? 0 : len);
            std::mt19937 rng(700U + static_cast<unsigned>(layer * 4 + p));
            for (auto& byte : bytes) byte = static_cast<std::uint8_t>(rng());
            buffers.emplace_back(std::max<std::size_t>(bytes.size(), 16));
            if (!bytes.empty()) write_buffer(buffers.back(), bytes);
            Tensor tensor = spec.bytes == 0 ? Tensor{}
                                            : Tensor(buffers.back().data(), spec.dtype,
                                                     {spec.leading, page_size, heads, pages});
            (p == 0   ? view.k_pages
             : p == 1 ? view.v_pages
             : p == 2 ? view.k_scale_pages
                      : view.v_scale_pages) = tensor;
            planes.push_back(bytes);
        }
        views.push_back(view);
    }
    expected = planes;
    for (int b = 0; b < 2; ++b) {
        const int* table = tables.data() + table_rows[static_cast<std::size_t>(b)] * stride;
        const auto& path = rows[static_cast<std::size_t>(b)].path;
        for (std::size_t i = 1; i < path.size(); ++i) {
            if (path[i] == static_cast<int>(i)) continue;
            const int from = bases[static_cast<std::size_t>(b)] + path[i];
            const int to   = bases[static_cast<std::size_t>(b)] + static_cast<int>(i);
            for (int plane = 0; plane < layer_count * 4; ++plane) {
                const int bytes = plane_specs[static_cast<std::size_t>(plane % 4)].bytes;
                if (bytes == 0) continue;
                for (int head = 0; head < heads; ++head) {
                    const auto at = [&](int position) {
                        return static_cast<std::size_t>(bytes) *
                               (static_cast<std::size_t>(page_size) *
                                    (head + static_cast<std::size_t>(heads) *
                                                table[position / page_size]) +
                                static_cast<std::size_t>(position % page_size));
                    };
                    std::copy_n(planes[static_cast<std::size_t>(plane)].begin() +
                                    static_cast<std::ptrdiff_t>(at(from)),
                                bytes,
                                expected[static_cast<std::size_t>(plane)].begin() +
                                    static_cast<std::ptrdiff_t>(at(to)));
                }
            }
        }
    }
    GuardedDeviceBuffer dpos(positions.size() * 4), drows(table_rows.size() * 4),
        dpath(accepted_path.size() * 4), dacc(accepted.size() * 4);
    write_buffer(dpos, positions);
    write_buffer(drows, table_rows);
    write_buffer(dpath, accepted_path);
    write_buffer(dacc, accepted);
    ops::speculative_tree_compact_kv(
        views.data(), layer_count, Tensor(dpos.data(), DType::I32, {n, 2}),
        Tensor(drows.data(), DType::I32, {2}), Tensor(dpath.data(), DType::I32, {n, 2}),
        Tensor(dacc.data(), DType::I32, {2}), nullptr);
    cuda_synchronize();
    int failures = 0;
    for (int plane = 0; plane < layer_count * 4; ++plane) {
        const auto& want = expected[static_cast<std::size_t>(plane)];
        if (want.empty()) continue;
        const std::string label = "tree compact KV " + std::to_string(static_cast<int>(storage)) +
                                  " plane " + std::to_string(plane);
        failures += verify_exact(
            label.c_str(),
            read_buffer<std::uint8_t>(buffers[static_cast<std::size_t>(plane)], want.size()), want);
        failures += buffers[static_cast<std::size_t>(plane)].verify_guards(label);
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Acceptance

struct AcceptBuffers {
    int n, k, batch;
    GuardedDeviceBuffer targets, logits, drafts, candidates, q, extents, lengths, anchors, licensed,
        counts, accepted, path, branch, configs;

    AcceptBuffers(int nodes, int rows)
        : n(nodes), k(nodes - 1), batch(rows), targets(sizeof(int) * nodes * rows),
          logits(static_cast<std::size_t>(kPhysicalRows) * nodes * rows * 2),
          drafts(sizeof(int) * (nodes - 1) * rows),
          candidates(sizeof(int) * kCandidates * (nodes - 1) * rows),
          q(sizeof(float) * kCandidates * (nodes - 1) * rows), extents(sizeof(int) * rows),
          lengths(sizeof(int) * rows), anchors(sizeof(int) * rows),
          licensed(sizeof(int) * nodes * rows), counts(sizeof(int) * rows),
          accepted(sizeof(int) * rows), path(sizeof(int) * nodes * rows),
          branch(sizeof(int) * rows), configs(sizeof(ops::SamplingConfig) * rows) {}

    Tensor t_targets() { return Tensor(targets.data(), DType::I32, {n, batch}); }

    Tensor t_logits() { return Tensor(logits.data(), DType::BF16, {kPhysicalRows, n, batch}); }

    Tensor t_drafts() { return Tensor(drafts.data(), DType::I32, {k, batch}); }

    Tensor t_candidates() { return Tensor(candidates.data(), DType::I32, {kCandidates, k, batch}); }

    Tensor t_q() { return Tensor(q.data(), DType::FP32, {kCandidates, k, batch}); }

    Tensor t_extents() { return Tensor(extents.data(), DType::I32, {batch}); }

    Tensor t_lengths() { return Tensor(lengths.data(), DType::I32, {batch}); }

    Tensor t_anchors() { return Tensor(anchors.data(), DType::I32, {batch}); }

    Tensor t_licensed() { return Tensor(licensed.data(), DType::I32, {n, batch}); }

    Tensor t_counts() { return Tensor(counts.data(), DType::I32, {batch}); }

    Tensor t_accepted() { return Tensor(accepted.data(), DType::I32, {batch}); }

    Tensor t_path() { return Tensor(path.data(), DType::I32, {n, batch}); }

    Tensor t_branch() { return Tensor(branch.data(), DType::I32, {batch}); }

    const ops::SamplingConfig* t_configs() {
        return static_cast<const ops::SamplingConfig*>(configs.data());
    }
};

struct AcceptOutputs {
    std::vector<int> licensed, counts, accepted, lengths, anchors, path, branch;
};

AcceptOutputs read_outputs(AcceptBuffers& b) {
    const auto n = static_cast<std::size_t>(b.n), rows = static_cast<std::size_t>(b.batch);
    return {read_buffer<int>(b.licensed, n * rows), read_buffer<int>(b.counts, rows),
            read_buffer<int>(b.accepted, rows),     read_buffer<int>(b.lengths, rows),
            read_buffer<int>(b.anchors, rows),      read_buffer<int>(b.path, n * rows),
            read_buffer<int>(b.branch, rows)};
}

void run_tree_accept(AcceptBuffers& b, GuardedDeviceBuffer& tree_rows, GuardedDeviceBuffer& ws,
                     cudaStream_t stream) {
    WorkspaceArena workspace(DeviceSpan{ws.data(), ws.bytes()});
    Tensor lengths = b.t_lengths(), anchors = b.t_anchors(), licensed = b.t_licensed(),
           counts = b.t_counts(), accepted = b.t_accepted(), path = b.t_path(),
           branch = b.t_branch();
    ops::speculative_accept_sparse_tree(b.t_targets(), b.t_logits(), b.t_drafts(), b.t_candidates(),
                                        b.t_q(), b.t_extents(), rows_tensor(tree_rows, b.batch),
                                        lengths, anchors, licensed, counts, accepted, path, branch,
                                        kTokenDomain, b.t_configs(), workspace, stream);
}

// Greedy rows take the tree path whose drafts match the target tokens; each row has its own
// tree.
int greedy_tree_case() {
    constexpr int k = 4, n = 9, batch = 5;
    const auto fanout = host_row(branched_parents(k, {{0, 2}, {0, 2}}), k, true);
    // Branches below main columns 1 and 2; only 8 of 9 columns live.
    const auto deep = host_row(branched_parents(k, {{1, 2}, {2, 1}}), k, true);
    const std::vector<ops::SpeculativeTreeRow> rows{fanout, fanout, fanout, deep, chain_row(2, k)};
    AcceptBuffers b(n, batch);
    std::vector<int> drafts(static_cast<std::size_t>(k + 4) * batch);
    for (int r = 0; r < batch; ++r)
        for (int c = 1; c < n; ++c)
            drafts[static_cast<std::size_t>(r) * (n - 1) + c - 1] = 1000 * (r + 1) + c;
    const auto d = [&](int r, int c) {
        return drafts[static_cast<std::size_t>(r) * (n - 1) + c - 1];
    };
    std::vector<int> targets(static_cast<std::size_t>(n) * batch, 7);
    const auto set = [&](int r, int c, int token) {
        targets[static_cast<std::size_t>(r) * n + c] = token;
    };
    // Row 0: whole main chain, then the bonus.
    for (int c = 0; c < 4; ++c) set(0, c, d(0, c + 1));
    set(0, 4, 777);
    // Row 1: side branch 1 to its leaf, then the bonus at column 6.
    set(1, 0, d(1, 5));
    set(1, 5, d(1, 6));
    set(1, 6, 888);
    // Row 2: no child matches.
    set(2, 0, 999);
    // Row 3: main column 1, its side child 5 and grandchild 6, then the bonus at column 6.
    set(3, 0, d(3, 1));
    set(3, 1, d(3, 5));
    set(3, 5, d(3, 6));
    set(3, 6, 555);
    // Row 4: a chain row of extent 2 rejecting its second draft.
    set(4, 0, d(4, 1));
    set(4, 1, 444);
    const std::vector<int> extents{n - 1, n - 1, n - 1, n - 1, 2};
    const std::vector<PathRow> expected_paths{
        {{0, 1, 2, 3, 4}}, {{0, 5, 6}}, {{0}}, {{0, 1, 5, 6}}, {{0, 1}}};
    const std::vector<int> terminals{777, 888, 999, 555, 444};
    // Depth at which the accepted path leaves the main chain.
    const std::vector<int> branches{0, 1, 0, 2, -1};

    std::vector<int> candidates(static_cast<std::size_t>(kCandidates) * (n - 1) * batch);
    std::vector<float> q(candidates.size(), 0.0F);
    for (std::size_t i = 0; i < drafts.size(); ++i)
        for (int c = 0; c < kCandidates; ++c) {
            candidates[i * kCandidates + c] =
                c == 0 ? drafts[i] : 200000 + static_cast<int>(i) * 16 + c;
            q[i * kCandidates + c] = c == 0 ? 1.0F : 0.0F;
        }
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    for (auto& config : configs) config.temperature = 0.0F;
    const std::vector<int> lengths(static_cast<std::size_t>(batch), 40),
        anchors(static_cast<std::size_t>(batch), 3);
    b.logits.fill(0);
    write_buffer(b.targets, targets);
    write_buffer(b.drafts, drafts);
    write_buffer(b.candidates, candidates);
    write_buffer(b.q, q);
    write_buffer(b.extents, extents);
    write_buffer(b.lengths, lengths);
    write_buffer(b.anchors, anchors);
    write_buffer(b.configs, configs);
    b.licensed.fill(0xff);
    b.path.fill(0xff);
    auto tree_rows = upload_rows(rows);
    const auto capacity =
        ops::speculative_accept_sparse_tree_workspace_capacity_bytes(kTokenDomain, n, batch, batch);
    GuardedDeviceBuffer ws(capacity);
    run_tree_accept(b, tree_rows, ws, nullptr);
    cuda_synchronize();
    const auto out = read_outputs(b);
    int failures   = 0;
    for (int r = 0; r < batch; ++r) {
        const auto& path = expected_paths[static_cast<std::size_t>(r)].path;
        const int a      = static_cast<int>(path.size()) - 1;
        bool ok = out.accepted[r] == a && out.counts[r] == a + 1 && out.lengths[r] == 40 + a + 1 &&
                  out.anchors[r] == terminals[static_cast<std::size_t>(r)] &&
                  out.branch[r] == branches[static_cast<std::size_t>(r)];
        for (int i = 0; i < n; ++i) {
            const auto at    = static_cast<std::size_t>(r) * n + i;
            const int token  = i < a    ? d(r, path[static_cast<std::size_t>(i) + 1])
                               : i == a ? terminals[static_cast<std::size_t>(r)]
                                        : 0;
            const int column = i <= a ? path[static_cast<std::size_t>(i)] : -1;
            ok               = ok && out.licensed[at] == token && out.path[at] == column;
        }
        failures += expect(ok, "greedy tree acceptance row " + std::to_string(r));
    }
    failures +=
        b.licensed.verify_guards("greedy tree licensed") + ws.verify_guards("greedy tree ws");
    return failures;
}

// Fills one row's proposal (16 distinct candidates, normalized q, a draft drawn from q) and the
// target logits of every live column: candidates of the column's proposal get high logits.
void fill_random_round(int n, int batch, std::uint32_t seed, std::vector<std::uint16_t>& logits,
                       std::vector<int>& drafts, std::vector<int>& candidates,
                       std::vector<float>& q, std::vector<int>& targets) {
    const int k = n - 1;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0F, 1.0F);
    logits.assign(static_cast<std::size_t>(kPhysicalRows) * n * batch, 0);
    drafts.assign(static_cast<std::size_t>(k) * batch, 0);
    candidates.assign(static_cast<std::size_t>(kCandidates) * k * batch, 0);
    q.assign(candidates.size(), 0.0F);
    targets.assign(static_cast<std::size_t>(n) * batch, 0);
    for (int r = 0; r < batch; ++r) {
        for (int c = 0; c < n; ++c) {
            const std::size_t base = (static_cast<std::size_t>(r) * n + c) * kPhysicalRows;
            for (int token = 0; token < kTokenDomain; ++token)
                logits[base + token] = f32_to_bf16(-8.0F + 4.0F * uniform(rng));
        }
        for (int c = 0; c < k; ++c) {
            const std::size_t at = static_cast<std::size_t>(r) * k + c;
            std::set<int> used;
            float total = 0.0F;
            for (int j = 0; j < kCandidates; ++j) {
                int id = 0;
                do {
                    id = static_cast<int>(rng() % 4096U) + 10000;
                } while (!used.insert(id).second);
                candidates[at * kCandidates + j] = id;
                q[at * kCandidates + j]          = 0.02F + uniform(rng);
                total += q[at * kCandidates + j];
            }
            for (int j = 0; j < kCandidates; ++j) q[at * kCandidates + j] /= total;
            float u = uniform(rng), cumulative = 0.0F;
            int drawn = kCandidates - 1;
            for (int j = 0; j < kCandidates; ++j) {
                cumulative += q[at * kCandidates + j];
                if (u < cumulative) {
                    drawn = j;
                    break;
                }
            }
            drafts[at] = candidates[at * kCandidates + drawn];
            // Target column c (which verifies draft c) prefers the proposal's candidates.
            const std::size_t base = (static_cast<std::size_t>(r) * n + c) * kPhysicalRows;
            for (int j = 0; j < kCandidates; ++j)
                logits[base + candidates[at * kCandidates + j]] = f32_to_bf16(3.0F * uniform(rng));
        }
        for (int c = 0; c < n; ++c) {
            const std::size_t base = (static_cast<std::size_t>(r) * n + c) * kPhysicalRows;
            int best               = 0;
            for (int token = 1; token < kTokenDomain; ++token)
                if (bf16_to_f32(logits[base + token]) > bf16_to_f32(logits[base + best]))
                    best = token;
            targets[static_cast<std::size_t>(r) * n + c] = best;
        }
    }
}

// A tree row whose tree is a chain walks single children: the recursive rejection must reproduce
// the chain Op bit for bit (same keys, same residual, same draws); chain rows are the chain Op.
int chain_equivalence_case() {
    constexpr int kDrafts = 5, batch = 8, n = kDrafts + 1;
    const std::vector<int> extents{kDrafts, kDrafts, kDrafts, kDrafts, kDrafts, kDrafts, 3, 0};
    std::vector<ops::SpeculativeTreeRow> rows;
    for (const int extent : extents)
        rows.push_back(host_row(chain_parents(extent), kDrafts, extent == kDrafts));
    auto tree_rows = upload_rows(rows);
    AcceptBuffers tree(n, batch), chain(n, batch);
    const auto tree_capacity =
        ops::speculative_accept_sparse_tree_workspace_capacity_bytes(kTokenDomain, n, batch, batch);
    const auto chain_capacity = ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(
        kTokenDomain, {false}, kDrafts, kDrafts, batch, batch);
    GuardedDeviceBuffer tree_ws(tree_capacity), chain_ws(chain_capacity);
    int failures = 0;
    for (std::uint32_t trial = 0; trial < 12; ++trial) {
        std::vector<std::uint16_t> logits;
        std::vector<int> drafts, candidates, targets;
        std::vector<float> q;
        fill_random_round(n, batch, 1234U + trial, logits, drafts, candidates, q, targets);
        std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
        const std::array<float, batch> temperature{0.8F, 1.0F, 1.3F, 0.0F, 1.0F, 0.7F, 1.0F, 0.9F};
        const std::array<int, batch> top_k{0, 40, 0, 0, 20, 0, 0, 5};
        const std::array<float, batch> top_p{1.0F, 0.9F, 1.0F, 1.0F, 1.0F, 0.95F, 1.0F, 1.0F};
        const std::array<float, batch> min_p{0.0F, 0.0F, 0.05F, 0.0F, 0.0F, 0.0F, 0.1F, 0.0F};
        // Rows 1, 3 and 5 also apply presence/frequency penalties over the drafts on their path.
        const std::array<float, batch> presence{0.0F, 0.5F, 0.0F, 0.4F, 0.0F, 0.3F, 0.0F, 0.0F};
        const std::array<float, batch> frequency{0.0F, 0.25F, 0.0F, 0.0F, 0.0F, 0.6F, 0.0F, 0.0F};
        for (int r = 0; r < batch; ++r) {
            auto& c             = configs[static_cast<std::size_t>(r)];
            c.temperature       = temperature[static_cast<std::size_t>(r)];
            c.top_k             = top_k[static_cast<std::size_t>(r)];
            c.top_p             = top_p[static_cast<std::size_t>(r)];
            c.min_p             = min_p[static_cast<std::size_t>(r)];
            c.presence_penalty  = presence[static_cast<std::size_t>(r)];
            c.frequency_penalty = frequency[static_cast<std::size_t>(r)];
            c.seed = 0x9e3779b97f4a7c15ULL * (trial + 1) + static_cast<std::uint64_t>(r);
        }
        std::vector<int> lengths(static_cast<std::size_t>(batch)),
            anchors(static_cast<std::size_t>(batch), 11);
        for (int r = 0; r < batch; ++r)
            lengths[static_cast<std::size_t>(r)] = 100 + 37 * r + static_cast<int>(trial);
        for (AcceptBuffers* side : {&tree, &chain}) {
            write_buffer(side->logits, logits);
            write_buffer(side->targets, targets);
            write_buffer(side->drafts, drafts);
            write_buffer(side->candidates, candidates);
            write_buffer(side->q, q);
            write_buffer(side->extents, extents);
            write_buffer(side->lengths, lengths);
            write_buffer(side->anchors, anchors);
            write_buffer(side->configs, configs);
            side->licensed.fill(0xff);
        }
        run_tree_accept(tree, tree_rows, tree_ws, nullptr);
        {
            WorkspaceArena workspace(DeviceSpan{chain_ws.data(), chain_ws.bytes()});
            Tensor lengths_t = chain.t_lengths(), anchors_t = chain.t_anchors(),
                   licensed_t = chain.t_licensed(), counts_t = chain.t_counts(),
                   accepted_t = chain.t_accepted();
            ops::speculative_accept_sparse_drafts(
                chain.t_targets(), chain.t_logits(), chain.t_drafts(), chain.t_candidates(),
                chain.t_q(), chain.t_extents(), lengths_t, anchors_t, licensed_t, counts_t,
                accepted_t, kTokenDomain, chain.t_configs(), {false}, workspace, nullptr);
        }
        cuda_synchronize();
        const auto got          = read_outputs(tree);
        const auto expected     = read_outputs(chain);
        const std::string label = "tree/chain acceptance trial " + std::to_string(trial);
        failures += verify_exact((label + " licensed").c_str(), got.licensed, expected.licensed);
        failures += verify_exact((label + " counts").c_str(), got.counts, expected.counts);
        failures += verify_exact((label + " accepted").c_str(), got.accepted, expected.accepted);
        failures += verify_exact((label + " lengths").c_str(), got.lengths, expected.lengths);
        failures += verify_exact((label + " anchors").c_str(), got.anchors, expected.anchors);
        for (int r = 0; r < batch; ++r) {
            bool ok = got.branch[r] == (extents[static_cast<std::size_t>(r)] == kDrafts ? 0 : -1);
            for (int i = 0; i < n; ++i)
                ok = ok && got.path[static_cast<std::size_t>(r) * n + i] ==
                               (i <= got.accepted[r] ? i : -1);
            failures += expect(ok, label + " path row " + std::to_string(r));
        }
    }
    failures += tree_ws.verify_guards("tree acceptance workspace");
    return failures;
}

// A root fan-out of m children drawn from q without replacement: the first emitted token must be
// distributed exactly as the processed target distribution p of the anchor column.
int distribution_case(int m) {
    const int n = m + 1, k = m, batch = 8;
    std::vector<int> parent(static_cast<std::size_t>(n), 0);
    parent[0] = -1;
    std::vector<ops::SpeculativeTreeRow> rows(static_cast<std::size_t>(batch),
                                              host_row(parent, 1, true));
    auto tree_rows = upload_rows(rows);
    AcceptBuffers b(n, batch);
    const std::array<int, 6> support{1000, 1001, 1002, 1003, 1004, 1005};
    const std::array<float, 6> target_logits{2.0F, 1.5F, 1.0F, 0.5F, 0.0F, -0.5F};
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(kPhysicalRows) * n * batch,
                                      f32_to_bf16(-100.0F));
    for (int r = 0; r < batch; ++r)
        for (int c = 0; c < n; ++c)
            for (std::size_t s = 0; s < support.size(); ++s)
                logits[(static_cast<std::size_t>(r) * n + c) * kPhysicalRows + support[s]] =
                    f32_to_bf16(target_logits[s]);
    std::array<double, 6> p{};
    double z = 0.0;
    for (std::size_t s = 0; s < support.size(); ++s) z += std::exp(double(target_logits[s]));
    for (std::size_t s = 0; s < support.size(); ++s) p[s] = std::exp(double(target_logits[s])) / z;

    // Candidates: the six support tokens (with a q unlike p) and ten tokens p excludes.
    std::array<int, kCandidates> ids{};
    std::array<float, kCandidates> q{};
    const std::array<float, 6> support_q{0.05F, 0.2F, 0.1F, 0.15F, 0.02F, 0.08F};
    float total = 0.0F;
    for (int j = 0; j < kCandidates; ++j) {
        ids[static_cast<std::size_t>(j)] = j < 6 ? support[static_cast<std::size_t>(j)] : 2000 + j;
        q[static_cast<std::size_t>(j)]   = j < 6 ? support_q[static_cast<std::size_t>(j)] : 0.04F;
        total += q[static_cast<std::size_t>(j)];
    }
    for (auto& value : q) value /= total;
    std::vector<int> candidates(static_cast<std::size_t>(kCandidates) * k * batch);
    std::vector<float> proposal(candidates.size());
    for (std::size_t column = 0; column < static_cast<std::size_t>(k) * batch; ++column)
        for (int j = 0; j < kCandidates; ++j) {
            candidates[column * kCandidates + j] = ids[static_cast<std::size_t>(j)];
            proposal[column * kCandidates + j]   = q[static_cast<std::size_t>(j)];
        }
    write_buffer(b.logits, logits);
    write_buffer(b.candidates, candidates);
    write_buffer(b.q, proposal);
    b.targets.fill(0);
    write_buffer(b.extents, std::vector<int>(static_cast<std::size_t>(batch), k));
    const auto capacity =
        ops::speculative_accept_sparse_tree_workspace_capacity_bytes(kTokenDomain, n, batch, batch);
    GuardedDeviceBuffer ws(capacity);

    std::mt19937_64 rng(4242U + static_cast<unsigned>(m));
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    constexpr int kLaunches = 2000;
    std::array<std::int64_t, 6> observed{};
    std::int64_t outside = 0, inconsistent = 0;
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    std::vector<int> drafts(static_cast<std::size_t>(k) * batch);
    for (int launch = 0; launch < kLaunches; ++launch) {
        for (int r = 0; r < batch; ++r) {
            // Children in sibling order are a without-replacement sample of q.
            std::array<bool, kCandidates> taken{};
            for (int child = 0; child < k; ++child) {
                double mass = 0.0;
                for (int j = 0; j < kCandidates; ++j)
                    if (!taken[static_cast<std::size_t>(j)]) mass += q[static_cast<std::size_t>(j)];
                double goal = uniform(rng) * mass;
                int drawn   = -1;
                for (int j = 0; j < kCandidates; ++j) {
                    if (taken[static_cast<std::size_t>(j)]) continue;
                    drawn = j;
                    goal -= q[static_cast<std::size_t>(j)];
                    if (goal < 0.0) break;
                }
                taken[static_cast<std::size_t>(drawn)] = true;
                drafts[static_cast<std::size_t>(r) * k + child] =
                    ids[static_cast<std::size_t>(drawn)];
            }
            auto& config       = configs[static_cast<std::size_t>(r)];
            config.temperature = 1.0F;
            config.seed =
                0xD1B54A32D192ED03ULL * static_cast<std::uint64_t>(launch * batch + r + 1);
        }
        write_buffer(b.drafts, drafts);
        write_buffer(b.configs, configs);
        write_buffer(b.lengths, std::vector<int>(static_cast<std::size_t>(batch), 50));
        write_buffer(b.anchors, std::vector<int>(static_cast<std::size_t>(batch), 1));
        run_tree_accept(b, tree_rows, ws, nullptr);
        cuda_synchronize();
        const auto licensed = read_buffer<int>(b.licensed, static_cast<std::size_t>(n) * batch);
        const auto path     = read_buffer<int>(b.path, static_cast<std::size_t>(n) * batch);
        const auto accepted = read_buffer<int>(b.accepted, static_cast<std::size_t>(batch));
        for (int r = 0; r < batch; ++r) {
            const int token = licensed[static_cast<std::size_t>(r) * n];
            const auto it   = std::find(support.begin(), support.end(), token);
            if (it == support.end()) {
                ++outside;
            } else {
                ++observed[static_cast<std::size_t>(it - support.begin())];
            }
            if (accepted[static_cast<std::size_t>(r)] == 1) {
                const int column = path[static_cast<std::size_t>(r) * n + 1];
                if (column < 1 || column > k ||
                    drafts[static_cast<std::size_t>(r) * k + column - 1] != token)
                    ++inconsistent;
            }
        }
    }
    const double trials = static_cast<double>(kLaunches) * batch;
    double chi2         = 0.0;
    for (std::size_t s = 0; s < support.size(); ++s) {
        const double expected = trials * p[s];
        chi2 += (observed[s] - expected) * (observed[s] - expected) / expected;
    }
    // Five degrees of freedom; 30 is a tail probability near 1.5e-5.
    const std::string label = "tree root distribution m=" + std::to_string(m);
    std::cout << label << " chi2=" << chi2 << " trials=" << trials << '\n';
    int failures = expect(chi2 < 30.0, label + ": chi-square " + std::to_string(chi2));
    failures += expect(outside == 0, label + ": emitted a token outside the target support");
    failures += expect(inconsistent == 0, label + ": accepted path disagrees with its token");
    failures += ws.verify_guards(label);
    return failures;
}

// ---------------------------------------------------------------------------------------------
// Tree builder

std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float pattern(std::uint32_t first, std::uint32_t second, std::uint32_t seed, float scale) {
    const std::uint32_t mixed = mix32(first * 0x9e3779b9U ^ second * 0x85ebca6bU ^ seed);
    int centered              = static_cast<int>((mixed >> 8) & 0xffU) - 128;
    if (centered == 0) centered = ((first + second) & 1U) == 0 ? 1 : -1;
    return bf16_to_f32(f32_to_bf16(static_cast<float>(centered) * scale));
}

double host_uniform(std::uint64_t seed, int position, int purpose, std::uint32_t sub) {
    const auto splitmix = [](std::uint64_t value) {
        value += 0x9E3779B97F4A7C15ull;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
        return value ^ (value >> 31);
    };
    std::uint64_t key = seed;
    key = splitmix(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                          0xD1B54A32D192ED03ull));
    key = splitmix(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(purpose)) << 21) ^
                   (static_cast<std::uint64_t>(sub) * 0x2545F4914F6CDD1Dull));
    return static_cast<double>(static_cast<std::uint32_t>(key >> 40)) / 16777216.0;
}

// The builder against the path walk (main chain, bit for bit) and a host replay of its
// best-first choices: every side column's parent had the highest priority (within FP32 rounding)
// when the column was added, its rank is the without-replacement draw of its parent's law with the
// column's counter key, and a row that stops short has no node left that may take a child.
int selector_case(int width, int max_paths) {
    constexpr int kSteps = 7, batch = 5;
    std::vector<int> ids(static_cast<std::size_t>(kCandidates) * kSteps * batch);
    std::vector<float> unary(ids.size());
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(kRank) * kSteps * batch);
    for (int r = 0; r < batch; ++r)
        for (int s = 0; s < kSteps; ++s)
            for (int c = 0; c < kCandidates; ++c) {
                const std::size_t at = (static_cast<std::size_t>(r) * kSteps + s) * kCandidates + c;
                ids[at]              = 1000 + r * 20000 + s * 257 + c;
                unary[at] = static_cast<float>(
                                static_cast<int>(
                                    (mix32(static_cast<std::uint32_t>(at) + 307U) >> 10) & 0xffU) -
                                128) /
                                64.0F +
                            static_cast<float>(kCandidates - c) * 0.0625F;
            }
    for (std::size_t i = 0; i < hidden.size(); ++i)
        hidden[i] =
            f32_to_bf16(pattern(static_cast<std::uint32_t>(i / kRank),
                                static_cast<std::uint32_t>(i % kRank), 401U, 1.0F / 512.0F));
    const std::vector<int> anchors{220000, 220101, 220202, 220303, 220404};
    const std::vector<int> positions{10, 500, 4096, 77, 9000};
    // Row 4 is a chain row of the round.
    const std::vector<int> extents{width - 1, width - 1, width - 1, width - 1, 4};
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    const std::array<float, batch> temperature{1.0F, 0.0F, 0.6F, 1.4F, 1.0F};
    for (int r = 0; r < batch; ++r) {
        configs[static_cast<std::size_t>(r)].temperature = temperature[static_cast<std::size_t>(r)];
        configs[static_cast<std::size_t>(r)].seed        = 77U + 13U * static_cast<unsigned>(r);
    }
    const auto pred_value = [](int token, int rank) {
        return static_cast<double>(pattern(static_cast<std::uint32_t>(token),
                                           static_cast<std::uint32_t>(rank), 101U, 1.0F / 256.0F));
    };
    const auto succ_value = [](int token, int rank) {
        return static_cast<double>(pattern(static_cast<std::uint32_t>(token),
                                           static_cast<std::uint32_t>(rank), 211U, 1.0F / 256.0F));
    };
    DeviceBuffer predecessor(static_cast<std::size_t>(kRank) * kPhysicalRows * 2),
        successor(static_cast<std::size_t>(kRank) * kPhysicalRows * 2);
    predecessor.fill();
    successor.fill();
    std::set<int> tokens(anchors.begin(), anchors.end());
    tokens.insert(ids.begin(), ids.end());
    std::vector<std::uint16_t> code(kRank);
    for (const int token : tokens) {
        for (int rank = 0; rank < kRank; ++rank)
            code[static_cast<std::size_t>(rank)] =
                f32_to_bf16(static_cast<float>(pred_value(token, rank)));
        predecessor.copy_from_host(code.data(), code.size() * 2,
                                   static_cast<std::size_t>(token) * kRank * 2);
        for (int rank = 0; rank < kRank; ++rank)
            code[static_cast<std::size_t>(rank)] =
                f32_to_bf16(static_cast<float>(succ_value(token, rank)));
        successor.copy_from_host(code.data(), code.size() * 2,
                                 static_cast<std::size_t>(token) * kRank * 2);
    }
    DeviceBuffer dids = to_device(ids), dunary = to_device(unary), dhidden = to_device(hidden),
                 danchors = to_device(anchors), dpositions = to_device(positions),
                 dextents = to_device(extents), dconfigs = to_device(configs);
    Tensor tids(dids.p, DType::I32, {kCandidates, kSteps, batch});
    Tensor tunary(dunary.p, DType::FP32, {kCandidates, kSteps, batch});
    Tensor thidden(dhidden.p, DType::BF16, {kRank, kSteps, batch});
    Tensor tanchors(danchors.p, DType::I32, {batch}), tpositions(dpositions.p, DType::I32, {batch}),
        textents(dextents.p, DType::I32, {batch});
    Tensor tpred(predecessor.p, DType::BF16, {kRank, kPhysicalRows});
    Tensor tsucc(successor.p, DType::BF16, {kRank, kPhysicalRows});
    const auto* cfg = static_cast<const ops::SamplingConfig*>(dconfigs.p);

    // Reference: the path walk.
    GuardedDeviceBuffer path_drafts(sizeof(int) * kSteps * batch),
        path_q(sizeof(float) * kCandidates * kSteps * batch);
    {
        GuardedDeviceBuffer ws(std::max<std::size_t>(
            ops::candidate_selector_path_workspace_capacity_bytes(kSteps, kSteps, batch, batch),
            1));
        WorkspaceArena workspace(DeviceSpan{ws.data(), ws.bytes()});
        Tensor drafts(path_drafts.data(), DType::I32, {kSteps, batch});
        Tensor q(path_q.data(), DType::FP32, {kCandidates, kSteps, batch});
        ops::candidate_selector_path(tids, tunary, thidden, tanchors, tpred, tsucc, tpositions, cfg,
                                     drafts, q, workspace, nullptr);
        cuda_synchronize();
    }
    const auto ref_drafts = read_buffer<int>(path_drafts, static_cast<std::size_t>(kSteps) * batch);
    const auto ref_q =
        read_buffer<float>(path_q, static_cast<std::size_t>(kCandidates) * kSteps * batch);

    const int columns = width - 1;
    GuardedDeviceBuffer drafts_buffer(sizeof(int) * columns * batch),
        cand_buffer(sizeof(int) * kCandidates * columns * batch),
        q_buffer(sizeof(float) * kCandidates * columns * batch),
        rows_buffer(sizeof(ops::SpeculativeTreeRow) * batch),
        masks_buffer(sizeof(std::uint32_t) * width * batch), valid_buffer(sizeof(int) * batch);
    std::vector<int> valid_in(static_cast<std::size_t>(batch));
    for (int r = 0; r < batch; ++r)
        valid_in[static_cast<std::size_t>(r)] = extents[static_cast<std::size_t>(r)] + 1;
    write_buffer(valid_buffer, valid_in);
    const auto capacity =
        ops::candidate_selector_tree_workspace_capacity_bytes(kSteps, batch, batch);
    GuardedDeviceBuffer ws(std::max<std::size_t>(capacity, 1));
    {
        WorkspaceArena workspace(DeviceSpan{ws.data(), ws.bytes()});
        Tensor drafts(drafts_buffer.data(), DType::I32, {columns, batch});
        Tensor column_candidates(cand_buffer.data(), DType::I32, {kCandidates, columns, batch});
        Tensor q(q_buffer.data(), DType::FP32, {kCandidates, columns, batch});
        Tensor tree_rows = rows_tensor(rows_buffer, batch);
        Tensor masks(masks_buffer.data(), DType::I32, {width, batch});
        Tensor valid(valid_buffer.data(), DType::I32, {batch});
        ops::candidate_selector_tree(tids, tunary, thidden, tanchors, tpred, tsucc, tpositions,
                                     textents, cfg, {width, kSteps, max_paths}, drafts,
                                     column_candidates, q, tree_rows, masks, valid, workspace,
                                     nullptr);
        cuda_synchronize();
    }
    const auto got_drafts =
        read_buffer<int>(drafts_buffer, static_cast<std::size_t>(columns) * batch);
    const auto got_cand =
        read_buffer<int>(cand_buffer, static_cast<std::size_t>(kCandidates) * columns * batch);
    const auto got_q =
        read_buffer<float>(q_buffer, static_cast<std::size_t>(kCandidates) * columns * batch);
    const auto got_masks =
        read_buffer<std::uint32_t>(masks_buffer, static_cast<std::size_t>(width) * batch);
    const auto got_valid = read_buffer<int>(valid_buffer, static_cast<std::size_t>(batch));
    std::vector<ops::SpeculativeTreeRow> got_rows(static_cast<std::size_t>(batch));
    rows_buffer.copy_to_host(got_rows.data(), sizeof(ops::SpeculativeTreeRow) * batch);

    const std::string label = "candidate_selector_tree W=" + std::to_string(width) +
                              " paths=" + std::to_string(max_paths);
    int failures            = 0;
    for (int r = 0; r < batch; ++r) {
        const std::string row_label = label + " row " + std::to_string(r);
        const auto& row             = got_rows[static_cast<std::size_t>(r)];
        const bool tree_row         = extents[static_cast<std::size_t>(r)] == columns;
        const bool greedy           = !(temperature[static_cast<std::size_t>(r)] > 0.0F);
        const double law_temperature =
            greedy ? 1.0 : static_cast<double>(temperature[static_cast<std::size_t>(r)]);
        const auto draft = [&](int c) {
            return got_drafts[static_cast<std::size_t>(r) * columns + c - 1];
        };
        const auto column_q = [&](int c, int j) {
            return got_q[(static_cast<std::size_t>(r) * columns + c - 1) * kCandidates + j];
        };
        // Main chain: the path walk, bit for bit.
        for (int c = 1; c <= kSteps; ++c) {
            const std::size_t at      = static_cast<std::size_t>(r) * columns + c - 1;
            const std::size_t lattice = static_cast<std::size_t>(r) * kSteps + c - 1;
            bool ok                   = got_drafts[at] == ref_drafts[lattice];
            for (int j = 0; j < kCandidates; ++j)
                ok = ok && got_cand[at * kCandidates + j] == ids[lattice * kCandidates + j] &&
                     std::memcmp(&got_q[at * kCandidates + j], &ref_q[lattice * kCandidates + j],
                                 4) == 0;
            failures += expect(ok, row_label + " main column " + std::to_string(c));
        }
        // The published topology is the host derivation of its parent array, and the masks and
        // live count follow it.
        const int nodes = row.nodes;
        bool shape_ok   = nodes >= 1 && nodes <= width && row.main_depth == kSteps &&
                          row.tree == (tree_row ? 1 : 0);
        std::vector<int> parent;
        for (int c = 0; shape_ok && c < nodes; ++c) parent.push_back(row.parent[c]);
        shape_ok = shape_ok && same_row(row, host_row(parent, kSteps, tree_row));
        failures += expect(shape_ok, row_label + " topology");
        if (!shape_ok) continue;
        const auto masks = host_masks(row, width);
        bool masks_ok    = true;
        for (int c = 0; c < width; ++c)
            masks_ok = masks_ok && got_masks[static_cast<std::size_t>(r) * width + c] ==
                                       masks[static_cast<std::size_t>(c)];
        failures += expect(masks_ok, row_label + " ancestor masks");
        failures += expect(got_valid[static_cast<std::size_t>(r)] ==
                               (tree_row ? nodes : valid_in[static_cast<std::size_t>(r)]),
                           row_label + " live columns");
        if (!tree_row) {
            failures += expect(nodes == extents[static_cast<std::size_t>(r)] + 1,
                               row_label + " chain row length");
            continue;
        }
        bool layout_ok = true;
        for (int c = 1; c < nodes; ++c) {
            if (c <= kSteps) {
                layout_ok = layout_ok && row.parent[c] == c - 1;
            } else {
                layout_ok = layout_ok && row.parent[c] < c && row.depth[c] <= kSteps;
            }
        }
        failures += expect(layout_ok && row.paths <= max_paths, row_label + " tree layout");

        // Host laws: the lattice edges recomputed in FP64 from the same operands.
        const auto token_of = [&](int c) {
            return c == 0 ? anchors[static_cast<std::size_t>(r)] : draft(c);
        };
        const auto law_of = [&](int node) {
            const int step = row.depth[node];
            std::array<double, kCandidates> edge{}, law{};
            double maximum            = -std::numeric_limits<double>::infinity();
            const std::size_t lattice = static_cast<std::size_t>(r) * kSteps + step;
            for (int c = 0; c < kCandidates; ++c) {
                double sum          = unary[lattice * kCandidates + c];
                const int candidate = ids[lattice * kCandidates + c];
                for (int rank = 0; rank < kRank; ++rank)
                    sum += pred_value(token_of(node), rank) *
                           static_cast<double>(bf16_to_f32(hidden[lattice * kRank + rank])) *
                           succ_value(candidate, rank);
                edge[static_cast<std::size_t>(c)] = sum;
                maximum                           = std::max(maximum, sum);
            }
            double total = 0.0;
            for (int c = 0; c < kCandidates; ++c) {
                law[static_cast<std::size_t>(c)] =
                    std::exp((edge[static_cast<std::size_t>(c)] - maximum) / law_temperature);
                total += law[static_cast<std::size_t>(c)];
            }
            for (auto& value : law) value /= total;
            return law;
        };
        const auto rank_of = [&](int c) {
            const std::size_t lattice = static_cast<std::size_t>(r) * kSteps + row.depth[c] - 1;
            for (int j = 0; j < kCandidates; ++j)
                if (ids[lattice * kCandidates + j] == draft(c)) return j;
            return -1;
        };
        std::vector<std::array<double, kCandidates>> laws(static_cast<std::size_t>(nodes));
        std::vector<double> score(static_cast<std::size_t>(nodes), 1.0);
        for (int v = 0; v < nodes; ++v) {
            if (row.depth[v] < kSteps) laws[static_cast<std::size_t>(v)] = law_of(v);
            if (v > 0)
                score[static_cast<std::size_t>(v)] =
                    score[static_cast<std::size_t>(row.parent[v])] *
                    laws[static_cast<std::size_t>(row.parent[v])]
                        [static_cast<std::size_t>(rank_of(v))];
        }
        // Replay the side columns in creation order.
        std::vector<unsigned> taken(static_cast<std::size_t>(nodes), 0U);
        std::vector<int> children(static_cast<std::size_t>(nodes), 0);
        for (int c = 1; c <= kSteps; ++c) {
            taken[static_cast<std::size_t>(c - 1)] |= 1U << rank_of(c);
            children[static_cast<std::size_t>(c - 1)] = 1;
        }
        int leaves          = 1;
        const auto priority = [&](int v) {
            if (row.depth[v] >= kSteps || children[static_cast<std::size_t>(v)] >= kCandidates ||
                (children[static_cast<std::size_t>(v)] > 0 && leaves >= max_paths))
                return -std::numeric_limits<double>::infinity();
            const auto& law = laws[static_cast<std::size_t>(v)];
            double best = 0.0, mass = 0.0, square = 0.0;
            for (int j = 0; j < kCandidates; ++j) {
                if ((taken[static_cast<std::size_t>(v)] >> j) & 1U) continue;
                best = std::max(best, law[static_cast<std::size_t>(j)]);
                mass += law[static_cast<std::size_t>(j)];
                square += law[static_cast<std::size_t>(j)] * law[static_cast<std::size_t>(j)];
            }
            const double expectation = greedy ? best : (mass > 0.0 ? square / mass : 0.0);
            return score[static_cast<std::size_t>(v)] * expectation;
        };
        for (int c = kSteps + 1; c <= nodes; ++c) {
            double best = -std::numeric_limits<double>::infinity();
            for (int v = 0; v < c; ++v) best = std::max(best, priority(v));
            if (c == nodes) {
                // A short row has no node left that may take a child.
                failures += expect(nodes == width || !(best > 0.0), row_label + " stopped early");
                break;
            }
            const int from       = row.parent[c];
            const double chosen  = priority(from);
            const std::string at = row_label + " side column " + std::to_string(c);
            failures += expect(chosen >= best * (1.0 - 1e-4) - 1e-12, at + " priority");
            const int rank = rank_of(c);
            bool draw_ok =
                rank >= 0 && ((taken[static_cast<std::size_t>(from)] >> rank) & 1U) == 0U;
            for (int j = 0; draw_ok && j < kCandidates; ++j)
                draw_ok =
                    got_cand[(static_cast<std::size_t>(r) * columns + c - 1) * kCandidates + j] ==
                    ids[(static_cast<std::size_t>(r) * kSteps + row.depth[c] - 1) * kCandidates +
                        j];
            if (draw_ok && greedy) {
                // The best untaken rank (ties either way).
                double top = 0.0;
                for (int j = 0; j < kCandidates; ++j)
                    if (((taken[static_cast<std::size_t>(from)] >> j) & 1U) == 0U)
                        top = std::max(
                            top, laws[static_cast<std::size_t>(from)][static_cast<std::size_t>(j)]);
                draw_ok = laws[static_cast<std::size_t>(from)][static_cast<std::size_t>(rank)] >=
                          top * (1.0 - 1e-6);
            } else if (draw_ok) {
                // The untaken-restricted inverse CDF of the parent's published law with the
                // column's counter key; accept either side of a rounding boundary.
                double mass = 0.0, before = 0.0;
                for (int j = 0; j < kCandidates; ++j) {
                    if ((taken[static_cast<std::size_t>(from)] >> j) & 1U) continue;
                    mass += column_q(c, j);
                    if (j < rank) before += column_q(c, j);
                }
                const std::uint32_t key =
                    (static_cast<std::uint32_t>(from + 1) << 5) |
                    static_cast<std::uint32_t>(children[static_cast<std::size_t>(from)]);
                const double goal =
                    host_uniform(configs[static_cast<std::size_t>(r)].seed,
                                 positions[static_cast<std::size_t>(r)] + row.depth[from], 5, key) *
                    mass;
                draw_ok = goal >= before - 1e-5 && goal < before + column_q(c, rank) + 1e-5;
                // Every child of a sampled parent carries the parent's whole law.
                for (int j = 0; j < kCandidates; ++j)
                    draw_ok =
                        draw_ok &&
                        std::fabs(
                            column_q(c, j) -
                            laws[static_cast<std::size_t>(from)][static_cast<std::size_t>(j)]) <
                            1e-4;
            }
            failures += expect(draw_ok, at + " draw");
            if (children[static_cast<std::size_t>(from)] > 0) ++leaves;
            taken[static_cast<std::size_t>(from)] |= 1U << std::max(rank, 0);
            children[static_cast<std::size_t>(from)] += 1;
        }
    }
    failures += drafts_buffer.verify_guards(label) + cand_buffer.verify_guards(label) +
                q_buffer.verify_guards(label) + rows_buffer.verify_guards(label) +
                masks_buffer.verify_guards(label) + ws.verify_guards(label);
    return failures;
}

// ---------------------------------------------------------------------------------------------
// GDN convolution and record replay: every root path of a tree row equals the chain Op on that
// path's columns alone, bit for bit; chain rows equal the chain Op.

std::vector<ops::SpeculativeTreeRow> gdn_rows(int k, int n) {
    return {host_row(branched_parents(k, {{0, 2}, {0, 2}}), k, true), chain_row(3, k),
            host_row(branched_parents(k, {{1, 2}, {2, 1}}), k, true)};
    (void)n;
}

int gdn_conv_case() {
    constexpr int channels = 10240, qrows = 2048, krows = 2048, vrows = 6144, slots = 4;
    constexpr int k = 4, n = 9, batch = 3;
    const auto rows = gdn_rows(k, n);
    std::vector<int> valid;
    for (const auto& row : rows) valid.push_back(row.nodes);
    const std::vector<int> initial{1, 3, 2};
    const auto projected =
        random_bf16(static_cast<std::size_t>(channels) * n * batch, 11U, -1.0F, 1.0F);
    const auto weight = random_bf16(static_cast<std::size_t>(channels) * 4, 12U, -0.5F, 0.5F);
    const auto states =
        random_bf16(static_cast<std::size_t>(channels) * 3 * slots, 13U, -1.0F, 1.0F);
    DeviceBuffer dweight = to_device(weight), dstates = to_device(states);
    auto tree_rows = upload_rows(rows);
    const auto run = [&](const std::vector<std::uint16_t>& input, int width, int rows_count,
                         const std::vector<int>& row_valid, const std::vector<int>& row_initial,
                         bool tree) {
        DeviceBuffer in = to_device(input), init = to_device(row_initial),
                     rv = to_device(row_valid);
        DeviceBuffer q(static_cast<std::size_t>(qrows) * width * rows_count * 2),
            kk(static_cast<std::size_t>(krows) * width * rows_count * 2),
            v(static_cast<std::size_t>(vrows) * width * rows_count * 2);
        Tensor tin(in.p, DType::BF16, {channels, width, rows_count});
        Tensor tweight(dweight.p, DType::BF16, {channels, 4});
        Tensor tstates(dstates.p, DType::BF16, {channels, 3, slots});
        Tensor tvalid(rv.p, DType::I32, {rows_count});
        Tensor tinit(init.p, DType::I32, {rows_count});
        Tensor tq(q.p, DType::BF16, {qrows, width, rows_count}),
            tk(kk.p, DType::BF16, {krows, width, rows_count}),
            tv(v.p, DType::BF16, {vrows, width, rows_count});
        if (tree)
            ops::detail::gdn_projected_conv_record_tree_launch(tin, tweight, tstates, tvalid, tinit,
                                                               rows_tensor(tree_rows, batch), tq,
                                                               tk, tv, nullptr);
        else
            ops::detail::gdn_projected_conv_record_launch(tin, tweight, tstates, tvalid, tinit, tq,
                                                          tk, tv, nullptr);
        cuda_synchronize();
        std::vector<std::uint16_t> out(static_cast<std::size_t>(channels) * width * rows_count);
        const auto qv =
            from_device<std::uint16_t>(q.p, static_cast<std::size_t>(qrows) * width * rows_count);
        const auto kv =
            from_device<std::uint16_t>(kk.p, static_cast<std::size_t>(krows) * width * rows_count);
        const auto vv =
            from_device<std::uint16_t>(v.p, static_cast<std::size_t>(vrows) * width * rows_count);
        for (std::size_t column = 0; column < static_cast<std::size_t>(width) * rows_count;
             ++column) {
            std::copy_n(qv.begin() + static_cast<std::ptrdiff_t>(column * qrows), qrows,
                        out.begin() + static_cast<std::ptrdiff_t>(column * channels));
            std::copy_n(kv.begin() + static_cast<std::ptrdiff_t>(column * krows), krows,
                        out.begin() + static_cast<std::ptrdiff_t>(column * channels + qrows));
            std::copy_n(vv.begin() + static_cast<std::ptrdiff_t>(column * vrows), vrows,
                        out.begin() +
                            static_cast<std::ptrdiff_t>(column * channels + qrows + krows));
        }
        return out;
    };
    const auto tree_out  = run(projected, n, batch, valid, initial, true);
    int failures         = 0;
    const auto column_of = [&](const std::vector<std::uint16_t>& data, int width, int row,
                               int column) {
        const auto begin = (static_cast<std::size_t>(row) * width + column) * channels;
        return std::vector<std::uint16_t>(data.begin() + static_cast<std::ptrdiff_t>(begin),
                                          data.begin() +
                                              static_cast<std::ptrdiff_t>(begin + channels));
    };
    const std::vector<std::uint16_t> zeros(channels, 0);
    for (int b = 0; b < batch; ++b) {
        const auto& t = rows[static_cast<std::size_t>(b)];
        for (int p = 0; p < t.paths; ++p) {
            const int length = t.path_length[p];
            std::vector<std::uint16_t> gathered;
            for (int i = 0; i < length; ++i) {
                const auto source = column_of(projected, n, b, t.path_columns[p][i]);
                gathered.insert(gathered.end(), source.begin(), source.end());
            }
            const auto chain_out =
                run(gathered, length, 1, {length}, {initial[static_cast<std::size_t>(b)]}, false);
            for (int i = 0; i < length; ++i)
                failures += verify_exact(("tree conv row " + std::to_string(b) + " path " +
                                          std::to_string(p) + " column " + std::to_string(i))
                                             .c_str(),
                                         column_of(tree_out, n, b, t.path_columns[p][i]),
                                         column_of(chain_out, length, 0, i));
        }
        for (int c = t.nodes; c < n; ++c)
            failures += verify_exact(
                ("tree conv row " + std::to_string(b) + " idle column " + std::to_string(c))
                    .c_str(),
                column_of(tree_out, n, b, c), zeros);
    }
    return failures;
}

// Rows whose walks nest: a two-level side branch and a four-child root (k=4), a three-level
// nesting that overflows the walk's two state slots and rebuilds its innermost branch state (k=4),
// a chain row, and a 16-column K=7 tree with two nested side branches.
std::vector<ops::SpeculativeTreeRow> nested_gdn_rows() {
    return {host_row({-1, 0, 1, 2, 3, 0, 0, 0, 2, 8, 8}, 4, true),
            host_row({-1, 0, 1, 2, 3, 1, 5, 5, 7, 7}, 4, true), chain_row(2, 4),
            host_row({-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 8, 10, 10, 12, 5, 14}, 7, true)};
}

int gdn_record_case(int value_heads, const std::vector<ops::SpeculativeTreeRow>& rows, int n) {
    constexpr int dim = 128, qk_heads = 16, slots = 4;
    const int batch = static_cast<int>(rows.size());
    std::vector<int> valid, initial;
    for (const auto& row : rows) {
        valid.push_back(row.nodes);
        initial.push_back((static_cast<int>(initial.size()) * 3 + 1) % slots);
    }
    const std::size_t qk_column = static_cast<std::size_t>(dim) * qk_heads;
    const std::size_t v_column  = static_cast<std::size_t>(dim) * value_heads;
    const std::size_t columns   = static_cast<std::size_t>(n) * batch;
    const auto q                = random_bf16(qk_column * columns, 21U, -0.08F, 0.08F);
    const auto kv               = random_bf16(qk_column * columns, 22U, -0.08F, 0.08F);
    const auto v                = random_bf16(v_column * columns, 23U, -0.08F, 0.08F);
    std::vector<float> g(static_cast<std::size_t>(value_heads) * columns), beta(g.size());
    fill_uniform(g, 24U, -1.2F, -0.02F);
    fill_uniform(beta, 25U, 0.02F, 0.98F);
    std::vector<float> states(static_cast<std::size_t>(dim) * dim * value_heads * slots);
    fill_uniform(states, 26U, -0.03F, 0.03F);
    DeviceBuffer dstates = to_device(states);
    auto tree_rows       = upload_rows(rows);
    const float scale    = 1.0F / std::sqrt(128.0F);

    struct Result {
        std::vector<std::uint16_t> out, key_record, value_record;
        std::vector<float> gate_record;
    };

    const auto gather = [](const auto& data, std::size_t column_elements, int width, int row,
                           const std::vector<int>& cols) {
        std::decay_t<decltype(data)> result;
        for (const int c : cols) {
            const auto begin = (static_cast<std::size_t>(row) * width + c) * column_elements;
            result.insert(result.end(), data.begin() + static_cast<std::ptrdiff_t>(begin),
                          data.begin() + static_cast<std::ptrdiff_t>(begin + column_elements));
        }
        return result;
    };
    const auto run = [&](const std::vector<std::uint16_t>& hq, const std::vector<std::uint16_t>& hk,
                         const std::vector<std::uint16_t>& hv, const std::vector<float>& hg,
                         const std::vector<float>& hbeta, int width, int rows_count,
                         const std::vector<int>& row_valid, const std::vector<int>& row_initial,
                         bool tree) {
        DeviceBuffer dq = to_device(hq), dk = to_device(hk), dv = to_device(hv), dg = to_device(hg),
                     dbeta = to_device(hbeta), dinit = to_device(row_initial),
                     dvalid    = to_device(row_valid);
        const std::size_t cols = static_cast<std::size_t>(width) * rows_count;
        DeviceBuffer out(v_column * cols * 2), key_record(qk_column * cols * 2),
            value_record(v_column * cols * 2),
            gate_record(static_cast<std::size_t>(2) * value_heads * cols * 4);
        out.fill(0xff);
        key_record.fill(0xff);
        value_record.fill(0xff);
        gate_record.fill(0xff);
        Tensor tq(dq.p, DType::BF16, {dim, qk_heads, width, rows_count}),
            tk(dk.p, DType::BF16, {dim, qk_heads, width, rows_count});
        Tensor tv(dv.p, DType::BF16, {dim, value_heads, width, rows_count});
        Tensor tg(dg.p, DType::FP32, {value_heads, width, rows_count}),
            tbeta(dbeta.p, DType::FP32, {value_heads, width, rows_count});
        Tensor tstates(dstates.p, DType::FP32, {dim, dim, value_heads, slots});
        Tensor tvalid(dvalid.p, DType::I32, {rows_count});
        Tensor tinit(dinit.p, DType::I32, {rows_count});
        Tensor tout(out.p, DType::BF16, {dim, value_heads, width, rows_count});
        Tensor tkr(key_record.p, DType::BF16, {dim, qk_heads, width, rows_count});
        Tensor tvr(value_record.p, DType::BF16, {dim, value_heads, width, rows_count});
        Tensor tgr(gate_record.p, DType::FP32, {2, value_heads, width, rows_count});
        if (tree)
            ops::gated_delta_net_replay_record(tq, tk, tv, tg, tbeta, scale, tstates, tvalid, tinit,
                                               rows_tensor(tree_rows, batch), tkr, tvr, tgr, tout,
                                               nullptr);
        else
            ops::gated_delta_net_replay_record(tq, tk, tv, tg, tbeta, scale, tstates, tvalid, tinit,
                                               tkr, tvr, tgr, tout, nullptr);
        cuda_synchronize();
        return Result{
            from_device<std::uint16_t>(out.p, v_column * cols),
            from_device<std::uint16_t>(key_record.p, qk_column * cols),
            from_device<std::uint16_t>(value_record.p, v_column * cols),
            from_device<float>(gate_record.p, static_cast<std::size_t>(2) * value_heads * cols)};
    };
    const auto tree = run(q, kv, v, g, beta, n, batch, valid, initial, true);
    int failures    = 0;
    const std::string label =
        "tree record Hv=" + std::to_string(value_heads) + " width=" + std::to_string(n);
    for (int b = 0; b < batch; ++b) {
        const auto& t = rows[static_cast<std::size_t>(b)];
        for (int p = 0; p < t.paths; ++p) {
            const std::vector<int> cols(t.path_columns[p], t.path_columns[p] + t.path_length[p]);
            const int width = static_cast<int>(cols.size());
            const auto chain =
                run(gather(q, qk_column, n, b, cols), gather(kv, qk_column, n, b, cols),
                    gather(v, v_column, n, b, cols),
                    gather(g, static_cast<std::size_t>(value_heads), n, b, cols),
                    gather(beta, static_cast<std::size_t>(value_heads), n, b, cols), width, 1,
                    {width}, {initial[static_cast<std::size_t>(b)]}, false);
            for (int i = 0; i < width; ++i) {
                const int c        = cols[static_cast<std::size_t>(i)];
                const auto tree_at = [&](std::size_t elements) {
                    return (static_cast<std::size_t>(b) * n + c) * elements;
                };
                const auto chain_at = [&](std::size_t elements) {
                    return static_cast<std::size_t>(i) * elements;
                };
                const std::vector<std::uint16_t> got(
                    tree.out.begin() + static_cast<std::ptrdiff_t>(tree_at(v_column)),
                    tree.out.begin() + static_cast<std::ptrdiff_t>(tree_at(v_column) + v_column));
                const std::vector<std::uint16_t> want(
                    chain.out.begin() + static_cast<std::ptrdiff_t>(chain_at(v_column)),
                    chain.out.begin() + static_cast<std::ptrdiff_t>(chain_at(v_column) + v_column));
                const std::string where = label + " row " + std::to_string(b) + " path " +
                                          std::to_string(p) + " column " + std::to_string(c);
                failures += verify_exact((where + " out").c_str(), got, want);
                // Records are raw copies of every live column's own inputs.
                const std::vector<std::uint16_t> key_got(
                    tree.key_record.begin() + static_cast<std::ptrdiff_t>(tree_at(qk_column)),
                    tree.key_record.begin() +
                        static_cast<std::ptrdiff_t>(tree_at(qk_column) + qk_column));
                const std::vector<std::uint16_t> key_want(
                    kv.begin() + static_cast<std::ptrdiff_t>(tree_at(qk_column)),
                    kv.begin() + static_cast<std::ptrdiff_t>(tree_at(qk_column) + qk_column));
                failures += verify_exact((where + " key record").c_str(), key_got, key_want);
                const std::vector<std::uint16_t> value_got(
                    tree.value_record.begin() + static_cast<std::ptrdiff_t>(tree_at(v_column)),
                    tree.value_record.begin() +
                        static_cast<std::ptrdiff_t>(tree_at(v_column) + v_column));
                const std::vector<std::uint16_t> value_want(
                    v.begin() + static_cast<std::ptrdiff_t>(tree_at(v_column)),
                    v.begin() + static_cast<std::ptrdiff_t>(tree_at(v_column) + v_column));
                failures += verify_exact((where + " value record").c_str(), value_got, value_want);
                bool gates = true;
                for (int h = 0; h < value_heads; ++h) {
                    const std::size_t at = (static_cast<std::size_t>(b) * n + c) * value_heads + h;
                    gates = gates && std::memcmp(&tree.gate_record[at * 2], &g[at], 4) == 0 &&
                            std::memcmp(&tree.gate_record[at * 2 + 1], &beta[at], 4) == 0;
                }
                failures += expect(gates, where + " gate record");
            }
        }
        // Columns past a row's live ones publish zero.
        bool idle = true;
        for (int c = t.nodes; c < n; ++c)
            for (std::size_t e = 0; e < v_column; ++e)
                idle = idle && tree.out[(static_cast<std::size_t>(b) * n + c) * v_column + e] == 0;
        failures += expect(idle, label + " row " + std::to_string(b) + " idle columns");
    }
    failures += verify_exact((label + " states unchanged").c_str(),
                             from_device<float>(dstates.p, states.size()), states);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += prepare_inputs_case();
        failures += compact_columns_case(false);
        failures += compact_columns_case(true);
        for (const KvCacheStorage storage :
             {KvCacheStorage::Int8Group64, KvCacheStorage::Fp8KeyNvfp4Value,
              KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8E4M3Row256,
              KvCacheStorage::BFloat16})
            failures += compact_kv_case(storage);
        failures += greedy_tree_case();
        failures += chain_equivalence_case();
        for (int m = 1; m <= 4; ++m) failures += distribution_case(m);
        failures += selector_case(16, 4);
        failures += selector_case(32, 8);
        failures += selector_case(24, 2);
        failures += gdn_conv_case();
        failures += gdn_record_case(48, gdn_rows(4, 9), 9);
        failures += gdn_record_case(32, gdn_rows(4, 9), 9);
        failures += gdn_record_case(48, nested_gdn_rows(), 16);
    } catch (const std::exception& error) {
        std::cerr << "speculative tree test threw: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " speculative tree check(s) failed\n";
        return 1;
    }
    std::cout << "speculative tree tests passed\n";
    return 0;
}
