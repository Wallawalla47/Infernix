// Speculative verification-tree Ops: per-row verify inputs, the lattice tree builder (main chain
// bit-identical to the path walk, best-first priorities and without-replacement draws replayed
// against a host oracle) and the independent-position builder (the same replay, its main chain
// against the host argmax and inverse CDF), the UniVer (transport) builds of both (shape replay,
// top ranks then one residual draw per node), tree acceptance (exact greedy walk, chain equivalence,
// recursive rejection output distribution, UniVer's first-token and per-column continuation
// distributions), path-parallel GDN convolution and record replay, and accepted-path compaction of
// record planes and of every paged KV storage format.
#include "ops/op_family.h"
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
#include <string_view>
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

// The verification logits' dtype: FP32 under --fp32-logits (the fixtures' BF16 values widened exactly),
// else BF16.
DType g_logits_dtype = DType::BF16;

void write_logits(GuardedDeviceBuffer& buffer, const std::vector<std::uint16_t>& bits) {
    if (g_logits_dtype == DType::BF16) {
        write_buffer(buffer, bits);
        return;
    }
    std::vector<float> wide(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { wide[i] = bf16_to_f32(bits[i]); }
    write_buffer(buffer, wide);
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
          logits(static_cast<std::size_t>(kPhysicalRows) * nodes * rows * sizeof(float)),
          drafts(sizeof(int) * (nodes - 1) * rows),
          candidates(sizeof(int) * kCandidates * (nodes - 1) * rows),
          q(sizeof(float) * kCandidates * (nodes - 1) * rows), extents(sizeof(int) * rows),
          lengths(sizeof(int) * rows), anchors(sizeof(int) * rows),
          licensed(sizeof(int) * nodes * rows), counts(sizeof(int) * rows),
          accepted(sizeof(int) * rows), path(sizeof(int) * nodes * rows),
          branch(sizeof(int) * rows), configs(sizeof(ops::SamplingConfig) * rows) {}

    Tensor t_targets() { return Tensor(targets.data(), DType::I32, {n, batch}); }

    Tensor t_logits() { return Tensor(logits.data(), g_logits_dtype, {kPhysicalRows, n, batch}); }

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
        // Proposal misses (candidates are 10000..14095): row 0's first draft has a small p > 0
        // (top-k off), row 4's has p = 0 (top-k 20). With q = 0 both must be rejected.
        constexpr std::array<int, 2> kMissRows{0, 4};
        for (const int r : kMissRows) drafts[static_cast<std::size_t>(r) * kDrafts] = 9000 + r;
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
            write_logits(side->logits, logits);
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
        for (const int r : kMissRows)
            failures += expect(got.accepted[static_cast<std::size_t>(r)] == 0,
                               label + " proposal miss rejected row " + std::to_string(r));
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
    write_logits(b.logits, logits);
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

// UniVer (transport rows): a two-level tree built by its greedy strategy, verified many times with
// fresh draws. The first emitted token must follow the anchor column's target, and the token emitted
// after a child's draft must follow that child column's target (its subtree's continuation), at
// every child that has one. Tree (columns): the anchor 0 has its law's top two ranks (1, the main
// chain, and 3) and a residual draw 4; column 1 has its top rank (2, the main chain) and a residual
// draw 5; column 4 has one residual draw 6 from its whole law. Every column's target puts mass on
// six candidate tokens, with column-dependent logits, and on one token outside the candidates.
int transport_distribution_case() {
    constexpr int n = 7, k = n - 1, batch = 8, kSupport = 7;
    const std::vector<int> parent{-1, 0, 1, 0, 0, 1, 4};
    ops::SpeculativeTreeRow row = host_row(parent, 2, true);
    row.transport               = 1;
    row.sampled                 = (1U << 4) | (1U << 5) | (1U << 6);
    const std::vector<ops::SpeculativeTreeRow> rows(static_cast<std::size_t>(batch), row);
    auto tree_rows = upload_rows(rows);
    AcceptBuffers b(n, batch);
    const std::array<int, kSupport> support{1000, 1001, 1002, 1003, 1004, 1005, 3000};
    // Column c's target logits: a column-dependent tilt over the support.
    const auto logit = [](int c, int s) {
        return 1.2F - 0.35F * static_cast<float>(s) + 0.25F * static_cast<float>(c) * static_cast<float>((s + c) % 3) -
               (s == kSupport - 1 ? 0.6F : 0.0F);
    };
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(kPhysicalRows) * n * batch, f32_to_bf16(-100.0F));
    std::array<std::array<double, kSupport>, n> target{};
    for (int c = 0; c < n; ++c) {
        double z = 0.0;
        for (int s = 0; s < kSupport; ++s) z += std::exp(static_cast<double>(bf16_to_f32(f32_to_bf16(logit(c, s)))));
        for (int s = 0; s < kSupport; ++s) {
            target[static_cast<std::size_t>(c)][static_cast<std::size_t>(s)] =
                std::exp(static_cast<double>(bf16_to_f32(f32_to_bf16(logit(c, s))))) / z;
        }
        for (int r = 0; r < batch; ++r)
            for (int s = 0; s < kSupport; ++s)
                logits[(static_cast<std::size_t>(r) * n + c) * kPhysicalRows + support[static_cast<std::size_t>(s)]] =
                    f32_to_bf16(logit(c, s));
    }
    // Every node's law over the same 16 candidates (support tokens 1000..1005, then ten tokens no
    // target holds); the laws differ per node and from the targets.
    std::array<int, kCandidates> ids{};
    for (int j = 0; j < kCandidates; ++j) ids[static_cast<std::size_t>(j)] = j < 6 ? 1000 + j : 2000 + j;
    const auto law = [&](int node) {
        std::array<double, kCandidates> q{};
        double total = 0.0;
        for (int j = 0; j < kCandidates; ++j) {
            q[static_cast<std::size_t>(j)] =
                j < 6 ? 0.30 / (1.0 + std::abs(j - (node * 2) % 6)) + 0.02 * j : 0.012 + 0.002 * (j % 4);
            total += q[static_cast<std::size_t>(j)];
        }
        for (auto& value : q) value /= total;
        return q;
    };
    const std::array<std::array<double, kCandidates>, 3> laws{law(0), law(1), law(4)}; // nodes 0, 1, 4
    // Each child column stores its parent's law and candidates.
    const std::array<int, n> law_of_column{-1, 0, 1, 0, 0, 1, 2};
    std::vector<int> candidates(static_cast<std::size_t>(kCandidates) * k * batch);
    std::vector<float> proposal(candidates.size());
    for (int r = 0; r < batch; ++r)
        for (int c = 1; c < n; ++c)
            for (int j = 0; j < kCandidates; ++j) {
                const std::size_t at = (static_cast<std::size_t>(r) * k + c - 1) * kCandidates + j;
                candidates[at]       = ids[static_cast<std::size_t>(j)];
                proposal[at] = static_cast<float>(
                    laws[static_cast<std::size_t>(law_of_column[static_cast<std::size_t>(c)])][static_cast<std::size_t>(j)]);
            }
    write_buffer(b.logits, logits);
    write_buffer(b.candidates, candidates);
    write_buffer(b.q, proposal);
    b.targets.fill(0);
    write_buffer(b.extents, std::vector<int>(static_cast<std::size_t>(batch), k));
    GuardedDeviceBuffer ws(ops::speculative_accept_sparse_tree_workspace_capacity_bytes(kTokenDomain, n, batch, batch));

    // The greedy strategy's ranks: the top `count` ranks of a law (lowest rank on ties), then one
    // draw from the rest of it.
    const auto top_ranks = [](const std::array<double, kCandidates>& q, int count) {
        std::vector<int> order(kCandidates);
        for (int j = 0; j < kCandidates; ++j) order[static_cast<std::size_t>(j)] = j;
        std::stable_sort(order.begin(), order.end(), [&](int x, int y) {
            return q[static_cast<std::size_t>(x)] > q[static_cast<std::size_t>(y)];
        });
        order.resize(static_cast<std::size_t>(count));
        return order;
    };
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const auto residual_draw = [&](const std::array<double, kCandidates>& q, const std::vector<int>& taken) {
        double mass = 0.0;
        for (int j = 0; j < kCandidates; ++j)
            if (std::find(taken.begin(), taken.end(), j) == taken.end()) mass += q[static_cast<std::size_t>(j)];
        double goal = uniform(rng) * mass;
        int drawn   = -1;
        for (int j = 0; j < kCandidates; ++j) {
            if (std::find(taken.begin(), taken.end(), j) != taken.end()) continue;
            drawn = j;
            goal -= q[static_cast<std::size_t>(j)];
            if (goal < 0.0) break;
        }
        return drawn;
    };
    const auto root_top = top_ranks(laws[0], 2), one_top = top_ranks(laws[1], 1);

    constexpr int kLaunches = 2500;
    // first[s]: the first token; after[c][s]: the token emitted after column c's draft.
    std::array<std::int64_t, kSupport> first{};
    std::array<std::array<std::int64_t, kSupport>, n> after{};
    std::array<std::int64_t, n> reached{};
    std::int64_t outside = 0, inconsistent = 0;
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    std::vector<int> drafts(static_cast<std::size_t>(k) * batch);
    const auto index_of = [&](int token) {
        const auto it = std::find(support.begin(), support.end(), token);
        return it == support.end() ? -1 : static_cast<int>(it - support.begin());
    };
    for (int launch = 0; launch < kLaunches; ++launch) {
        for (int r = 0; r < batch; ++r) {
            int* d = drafts.data() + static_cast<std::size_t>(r) * k;
            d[0]   = ids[static_cast<std::size_t>(root_top[0])];                         // column 1
            d[2]   = ids[static_cast<std::size_t>(root_top[1])];                         // column 3
            d[3]   = ids[static_cast<std::size_t>(residual_draw(laws[0], root_top))];    // column 4
            d[1]   = ids[static_cast<std::size_t>(one_top[0])];                          // column 2
            d[4]   = ids[static_cast<std::size_t>(residual_draw(laws[1], one_top))];     // column 5
            d[5]   = ids[static_cast<std::size_t>(residual_draw(laws[2], {}))];          // column 6
            auto& config       = configs[static_cast<std::size_t>(r)];
            config.temperature = 1.0F;
            config.seed        = 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(launch * batch + r + 7);
        }
        write_buffer(b.drafts, drafts);
        write_buffer(b.configs, configs);
        write_buffer(b.lengths, std::vector<int>(static_cast<std::size_t>(batch), 60));
        write_buffer(b.anchors, std::vector<int>(static_cast<std::size_t>(batch), 1));
        run_tree_accept(b, tree_rows, ws, nullptr);
        cuda_synchronize();
        const auto licensed = read_buffer<int>(b.licensed, static_cast<std::size_t>(n) * batch);
        const auto path     = read_buffer<int>(b.path, static_cast<std::size_t>(n) * batch);
        const auto accepted = read_buffer<int>(b.accepted, static_cast<std::size_t>(batch));
        for (int r = 0; r < batch; ++r) {
            const int a = accepted[static_cast<std::size_t>(r)];
            for (int i = 0; i <= a; ++i) {
                const int s = index_of(licensed[static_cast<std::size_t>(r) * n + i]);
                if (s < 0) {
                    ++outside;
                    continue;
                }
                if (i == 0) {
                    ++first[static_cast<std::size_t>(s)];
                } else {
                    const int c = path[static_cast<std::size_t>(r) * n + i];
                    ++after[static_cast<std::size_t>(c)][static_cast<std::size_t>(s)];
                    ++reached[static_cast<std::size_t>(c)];
                }
            }
            // Path column i holds licensed token i-1 and descends from column i-1.
            for (int i = 1; i <= a; ++i) {
                const int c = path[static_cast<std::size_t>(r) * n + i];
                if (c < 1 || c >= n || parent[static_cast<std::size_t>(c)] != path[static_cast<std::size_t>(r) * n + i - 1] ||
                    drafts[static_cast<std::size_t>(r) * k + c - 1] != licensed[static_cast<std::size_t>(r) * n + i - 1])
                    ++inconsistent;
            }
        }
    }
    const auto chi2_of = [&](const std::array<std::int64_t, kSupport>& observed, std::int64_t trials,
                             const std::array<double, kSupport>& p) {
        double chi2 = 0.0;
        for (int s = 0; s < kSupport; ++s) {
            const double expected = static_cast<double>(trials) * p[static_cast<std::size_t>(s)];
            const double diff     = static_cast<double>(observed[static_cast<std::size_t>(s)]) - expected;
            chi2 += diff * diff / expected;
        }
        return chi2;
    };
    const std::string label = "transport tree distribution";
    int failures            = 0;
    // Six degrees of freedom; 35 is a tail probability near 4e-6.
    const double root_chi2 = chi2_of(first, static_cast<std::int64_t>(kLaunches) * batch, target[0]);
    std::cout << label << " anchor chi2=" << root_chi2 << '\n';
    failures += expect(root_chi2 < 35.0, label + ": the first token follows the anchor's target");
    for (int c = 1; c < n; ++c) {
        if (reached[static_cast<std::size_t>(c)] < 500) continue; // too few continuations to test
        const double chi2 = chi2_of(after[static_cast<std::size_t>(c)], reached[static_cast<std::size_t>(c)],
                                    target[static_cast<std::size_t>(c)]);
        std::cout << label << " column " << c << " chi2=" << chi2 << " over " << reached[static_cast<std::size_t>(c)]
                  << '\n';
        failures += expect(chi2 < 35.0, label + ": the token after column " + std::to_string(c) +
                                            " follows its target");
    }
    failures += expect(reached[1] >= 500 && reached[4] >= 500, label + ": both root branches continue");
    failures += expect(outside == 0, label + ": emitted a token outside the target support");
    failures += expect(inconsistent == 0, label + ": accepted path disagrees with its tokens");
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

// One tree build's inputs as the host checks see them: [16,K,B] candidate ids and per-row base
// positions, extents, valid columns before the build and sampling configs.
struct TreeBuildCase {
    int width = 0, max_paths = 0, steps = 0, batch = 0;
    std::vector<int> ids, positions, extents, valid_in;
    std::vector<ops::SamplingConfig> configs;
};

struct TreeBuildOutputs {
    std::vector<int> drafts, candidates, valid;
    std::vector<float> q;
    std::vector<std::uint32_t> masks;
    std::vector<ops::SpeculativeTreeRow> rows;
};

using CandidateLaw = std::array<double, kCandidates>;
// The FP64 law of a node's children (at the row's temperature, or 1 for greedy rows).
using NodeLaw = std::function<CandidateLaw(int row, const ops::SpeculativeTreeRow& tree, int node)>;

// A sampled tree row built for UniVer (SpeculativeTreeShape::transport), whose laws do not depend
// on the node's token (independent positions): the shape is the greedy best-first build in which
// sibling j of node v takes v's j-th best rank (replayed as check_tree_rows replays priorities, under
// those ranks' scores); then every node's children but the last take its top ranks in order, and its
// last child is the inverse CDF of its law over the other ranks at the counter key (the chain's key
// for a main column's only child), marked in the row's sampled bits. Every child stores its parent's
// law.
//
// replay_shape: the shape replay assumes a node's law does not depend on its token, which holds
// for independent positions only; a lattice row checks the token fill alone (against the laws of
// its filled tokens).
int check_transport_row(const std::string& label, const TreeBuildCase& t, const TreeBuildOutputs& got, int r,
                        const ops::SpeculativeTreeRow& row, const NodeLaw& law_of, bool replay_shape) {
    const int columns = t.width - 1, kSteps = t.steps, width = t.width, max_paths = t.max_paths;
    const int nodes   = row.nodes;
    int failures      = expect(row.transport == 1, label + " transport flag");
    const auto column_q = [&](int c, int j) {
        return got.q[(static_cast<std::size_t>(r) * columns + c - 1) * kCandidates + j];
    };
    const auto rank_of = [&](int c) {
        const std::size_t lattice = static_cast<std::size_t>(r) * kSteps + row.depth[c] - 1;
        for (int j = 0; j < kCandidates; ++j)
            if (t.ids[lattice * kCandidates + j] == got.drafts[static_cast<std::size_t>(r) * columns + c - 1]) return j;
        return -1;
    };
    std::vector<CandidateLaw> laws(static_cast<std::size_t>(nodes));
    std::vector<std::vector<int>> order(static_cast<std::size_t>(nodes));
    for (int v = 0; v < nodes; ++v) {
        if (row.depth[v] >= kSteps) continue;
        laws[static_cast<std::size_t>(v)] = law_of(r, row, v);
        auto& o                           = order[static_cast<std::size_t>(v)];
        for (int j = 0; j < kCandidates; ++j) o.push_back(j);
        std::stable_sort(o.begin(), o.end(), [&](int x, int y) {
            return laws[static_cast<std::size_t>(v)][static_cast<std::size_t>(x)] >
                   laws[static_cast<std::size_t>(v)][static_cast<std::size_t>(y)];
        });
    }
    // Shape scores: sibling j of v contributes law_v(j-th best rank).
    std::vector<double> score(static_cast<std::size_t>(nodes), 1.0);
    for (int c = 1; c < nodes; ++c) {
        const int p = row.parent[c];
        score[static_cast<std::size_t>(c)] =
            score[static_cast<std::size_t>(p)] *
            laws[static_cast<std::size_t>(p)][static_cast<std::size_t>(
                order[static_cast<std::size_t>(p)][static_cast<std::size_t>(row.sibling_index[c])])];
    }
    std::vector<int> children(static_cast<std::size_t>(nodes), 0);
    for (int c = 1; c <= kSteps; ++c) children[static_cast<std::size_t>(c - 1)] = 1;
    int leaves          = 1;
    const auto priority = [&](int v) {
        const int taken = children[static_cast<std::size_t>(v)];
        if (row.depth[v] >= kSteps || taken >= kCandidates || (taken > 0 && leaves >= max_paths))
            return -std::numeric_limits<double>::infinity();
        return score[static_cast<std::size_t>(v)] *
               laws[static_cast<std::size_t>(v)][static_cast<std::size_t>(
                   order[static_cast<std::size_t>(v)][static_cast<std::size_t>(taken)])];
    };
    for (int c = kSteps + 1; replay_shape && c <= nodes; ++c) {
        double best = -std::numeric_limits<double>::infinity();
        for (int v = 0; v < c; ++v) best = std::max(best, priority(v));
        if (c == nodes) {
            failures += expect(nodes == width || !(best > 0.0), label + " stopped early");
            break;
        }
        const int from = row.parent[c];
        failures += expect(priority(from) >= best * (1.0 - 1e-4) - 1e-12,
                           label + " side column " + std::to_string(c) + " priority");
        if (children[static_cast<std::size_t>(from)] > 0) ++leaves;
        children[static_cast<std::size_t>(from)] += 1;
    }
    // Token fill.
    std::uint32_t expected_sampled = 0U;
    for (int v = 0; v < nodes; ++v) {
        std::vector<int> kids;
        for (int c = row.first_child[v]; c >= 0; c = row.next_sibling[c]) kids.push_back(c);
        const int m = static_cast<int>(kids.size());
        for (int j = 0; j < m; ++j) {
            const int c          = kids[static_cast<std::size_t>(j)];
            const int rank       = rank_of(c);
            const std::string at = label + " column " + std::to_string(c);
            bool law_ok          = true;
            for (int i = 0; i < kCandidates; ++i)
                law_ok = law_ok && std::fabs(column_q(c, i) - laws[static_cast<std::size_t>(v)][static_cast<std::size_t>(i)]) <
                                       1e-4;
            failures += expect(law_ok, at + " stores its parent's law");
            if (j < m - 1) {
                failures += expect(rank == order[static_cast<std::size_t>(v)][static_cast<std::size_t>(j)],
                                   at + " takes its parent's next top rank");
                continue;
            }
            expected_sampled |= 1U << c;
            // The inverse CDF over the ranks the deterministic siblings left; either side of a
            // rounding boundary.
            std::uint32_t taken = 0U;
            for (int i = 0; i < m - 1; ++i)
                taken |= 1U << order[static_cast<std::size_t>(v)][static_cast<std::size_t>(i)];
            double mass = 0.0, before = 0.0;
            for (int i = 0; i < kCandidates; ++i) {
                if ((taken >> i) & 1U) continue;
                mass += column_q(c, i);
                if (i < rank) before += column_q(c, i);
            }
            // A main-chain draft (step v) carries its step; a side child its (parent, sibling) index.
            const std::uint32_t key = j == 0 && c == v + 1 && v < kSteps
                                          ? static_cast<std::uint32_t>(v)
                                          : (static_cast<std::uint32_t>(v + 1) << 5) | static_cast<std::uint32_t>(j);
            const double goal = host_uniform(t.configs[static_cast<std::size_t>(r)].seed,
                                             t.positions[static_cast<std::size_t>(r)] + row.depth[v], 5, key) *
                                mass;
            const bool draw_ok = rank >= 0 && ((taken >> rank) & 1U) == 0U && goal >= before - 1e-5 &&
                                 goal < before + column_q(c, rank) + 1e-5;
            failures += expect(draw_ok, at + " is its parent's residual draw");
        }
    }
    failures += expect(row.sampled == expected_sampled, label + " sampled bits");
    return failures;
}

// Every row of a build past its main chain (which the caller checks): the published topology is
// the host derivation of its parent array, the masks and live count follow it, and a host replay
// of its best-first choices under `law_of` holds: every side column's parent had the highest
// priority (within FP32 rounding) when the column was added, its rank is the without-replacement
// draw of its parent's law with the column's counter key, and a row that stops short has no node
// left that may take a child. With leaf_sides only main-chain nodes take children; a row in
// bare_rows is a tree row without side columns. With transport, sampled tree rows are checked by
// check_transport_row instead (and every other row carries no transport marks).
int check_tree_rows(const std::string& label, const TreeBuildCase& t, const TreeBuildOutputs& got,
                    const NodeLaw& law_of, bool leaf_sides = false,
                    const std::vector<int>& bare_rows = {}, bool transport = false,
                    bool independent = true) {
    const int columns = t.width - 1, kSteps = t.steps, width = t.width, max_paths = t.max_paths;
    int failures      = 0;
    for (int r = 0; r < t.batch; ++r) {
        const std::string row_label = label + " row " + std::to_string(r);
        const auto& row             = got.rows[static_cast<std::size_t>(r)];
        const bool tree_row         = t.extents[static_cast<std::size_t>(r)] == columns;
        const bool greedy           = !(t.configs[static_cast<std::size_t>(r)].temperature > 0.0F);
        const auto draft            = [&](int c) {
            return got.drafts[static_cast<std::size_t>(r) * columns + c - 1];
        };
        const auto column_q = [&](int c, int j) {
            return got.q[(static_cast<std::size_t>(r) * columns + c - 1) * kCandidates + j];
        };
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
            masks_ok = masks_ok && got.masks[static_cast<std::size_t>(r) * width + c] ==
                                       masks[static_cast<std::size_t>(c)];
        failures += expect(masks_ok, row_label + " ancestor masks");
        failures += expect(got.valid[static_cast<std::size_t>(r)] ==
                               (tree_row ? nodes : t.valid_in[static_cast<std::size_t>(r)]),
                           row_label + " live columns");
        if (!tree_row) {
            failures += expect(nodes == t.extents[static_cast<std::size_t>(r)] + 1,
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
        if (std::find(bare_rows.begin(), bare_rows.end(), r) != bare_rows.end()) {
            failures += expect(nodes == kSteps + 1, row_label + " has no side column");
            continue;
        }
        if (transport && !greedy) {
            failures += check_transport_row(row_label, t, got, r, row, law_of, independent);
            continue;
        }
        failures += expect(row.transport == 0 && row.sampled == 0U, row_label + " carries no transport marks");

        const auto rank_of = [&](int c) {
            const std::size_t lattice = static_cast<std::size_t>(r) * kSteps + row.depth[c] - 1;
            for (int j = 0; j < kCandidates; ++j)
                if (t.ids[lattice * kCandidates + j] == draft(c)) return j;
            return -1;
        };
        std::vector<CandidateLaw> laws(static_cast<std::size_t>(nodes));
        std::vector<double> score(static_cast<std::size_t>(nodes), 1.0);
        for (int v = 0; v < nodes; ++v) {
            if (row.depth[v] < kSteps) laws[static_cast<std::size_t>(v)] = law_of(r, row, v);
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
            if ((leaf_sides && v > kSteps) || row.depth[v] >= kSteps ||
                children[static_cast<std::size_t>(v)] >= kCandidates ||
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
                    got.candidates[(static_cast<std::size_t>(r) * columns + c - 1) * kCandidates +
                                   j] ==
                    t.ids[(static_cast<std::size_t>(r) * kSteps + row.depth[c] - 1) * kCandidates +
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
                    host_uniform(t.configs[static_cast<std::size_t>(r)].seed,
                                 t.positions[static_cast<std::size_t>(r)] + row.depth[from], 5,
                                 key) *
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
    return failures;
}

// Device buffers for one tree build's outputs, read back after the build.
struct TreeBuildBuffers {
    GuardedDeviceBuffer drafts, candidates, q, rows, masks, valid;
    int width, batch;
    TreeBuildBuffers(int width_, int batch_, const std::vector<int>& valid_in)
        : drafts(sizeof(int) * (width_ - 1) * batch_),
          candidates(sizeof(int) * kCandidates * (width_ - 1) * batch_),
          q(sizeof(float) * kCandidates * (width_ - 1) * batch_),
          rows(sizeof(ops::SpeculativeTreeRow) * batch_),
          masks(sizeof(std::uint32_t) * width_ * batch_), valid(sizeof(int) * batch_),
          width(width_), batch(batch_) {
        write_buffer(valid, valid_in);
    }
    TreeBuildOutputs read() {
        const auto columns = static_cast<std::size_t>(width - 1);
        TreeBuildOutputs out;
        out.drafts     = read_buffer<int>(drafts, columns * batch);
        out.candidates = read_buffer<int>(candidates, kCandidates * columns * batch);
        out.q          = read_buffer<float>(q, kCandidates * columns * batch);
        out.masks      = read_buffer<std::uint32_t>(masks, static_cast<std::size_t>(width) * batch);
        out.valid      = read_buffer<int>(valid, static_cast<std::size_t>(batch));
        out.rows.resize(static_cast<std::size_t>(batch));
        rows.copy_to_host(out.rows.data(), sizeof(ops::SpeculativeTreeRow) * batch);
        return out;
    }
    int verify_guards(const std::string& label) {
        return drafts.verify_guards(label) + candidates.verify_guards(label) +
               q.verify_guards(label) + rows.verify_guards(label) + masks.verify_guards(label) +
               valid.verify_guards(label);
    }
};

// The builder against the path walk (main chain, bit for bit) and the host replay of
// check_tree_rows over FP64 lattice laws. With transport the sampled tree rows are UniVer trees:
// check_transport_row checks their token fill against the laws of their filled tokens.
int selector_case(int width, int max_paths, bool transport = false) {
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
                                     drafts, q, nullptr, workspace, nullptr);
        cuda_synchronize();
    }
    const auto ref_drafts = read_buffer<int>(path_drafts, static_cast<std::size_t>(kSteps) * batch);
    const auto ref_q =
        read_buffer<float>(path_q, static_cast<std::size_t>(kCandidates) * kSteps * batch);

    TreeBuildCase t{width, max_paths, kSteps, batch, ids, positions, extents, {}, configs};
    t.valid_in.resize(static_cast<std::size_t>(batch));
    for (int r = 0; r < batch; ++r)
        t.valid_in[static_cast<std::size_t>(r)] = extents[static_cast<std::size_t>(r)] + 1;
    const int columns = width - 1;
    TreeBuildBuffers out(width, batch, t.valid_in);
    const auto capacity =
        ops::candidate_selector_tree_workspace_capacity_bytes(kSteps, batch, batch);
    GuardedDeviceBuffer ws(std::max<std::size_t>(capacity, 1));
    {
        WorkspaceArena workspace(DeviceSpan{ws.data(), ws.bytes()});
        Tensor drafts(out.drafts.data(), DType::I32, {columns, batch});
        Tensor column_candidates(out.candidates.data(), DType::I32, {kCandidates, columns, batch});
        Tensor q(out.q.data(), DType::FP32, {kCandidates, columns, batch});
        Tensor tree_rows = rows_tensor(out.rows, batch);
        Tensor masks(out.masks.data(), DType::I32, {width, batch});
        Tensor valid(out.valid.data(), DType::I32, {batch});
        ops::candidate_selector_tree(tids, tunary, thidden, tanchors, tpred, tsucc, tpositions,
                                     textents, cfg, {width, kSteps, max_paths, transport}, drafts,
                                     column_candidates, q, tree_rows, masks, valid, workspace,
                                     nullptr);
        cuda_synchronize();
    }
    const TreeBuildOutputs got = out.read();

    const std::string label = std::string(transport ? "transport " : "") + "candidate_selector_tree W=" +
                              std::to_string(width) + " paths=" + std::to_string(max_paths);
    int failures            = 0;
    // Main chain: the path walk, bit for bit (a UniVer row's main chain is its token fill).
    for (int r = 0; r < batch; ++r) {
        if (transport && temperature[static_cast<std::size_t>(r)] > 0.0F &&
            extents[static_cast<std::size_t>(r)] == columns)
            continue;
        for (int c = 1; c <= kSteps; ++c) {
            const std::size_t at      = static_cast<std::size_t>(r) * columns + c - 1;
            const std::size_t lattice = static_cast<std::size_t>(r) * kSteps + c - 1;
            bool ok                   = got.drafts[at] == ref_drafts[lattice];
            for (int j = 0; j < kCandidates; ++j)
                ok = ok && got.candidates[at * kCandidates + j] == ids[lattice * kCandidates + j] &&
                     std::memcmp(&got.q[at * kCandidates + j], &ref_q[lattice * kCandidates + j],
                                 4) == 0;
            failures += expect(ok, label + " row " + std::to_string(r) + " main column " +
                                       std::to_string(c));
        }
    }
    // Host laws: the lattice edges recomputed in FP64 from the same operands.
    const NodeLaw law_of = [&](int r, const ops::SpeculativeTreeRow& row, int node) {
        const bool greedy = !(temperature[static_cast<std::size_t>(r)] > 0.0F);
        const double law_temperature =
            greedy ? 1.0 : static_cast<double>(temperature[static_cast<std::size_t>(r)]);
        const int token = node == 0 ? anchors[static_cast<std::size_t>(r)]
                                    : got.drafts[static_cast<std::size_t>(r) * columns + node - 1];
        const int step  = row.depth[node];
        CandidateLaw edge{}, law{};
        double maximum            = -std::numeric_limits<double>::infinity();
        const std::size_t lattice = static_cast<std::size_t>(r) * kSteps + step;
        for (int c = 0; c < kCandidates; ++c) {
            double sum          = unary[lattice * kCandidates + c];
            const int candidate = ids[lattice * kCandidates + c];
            for (int rank = 0; rank < kRank; ++rank)
                sum += pred_value(token, rank) *
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
    failures += check_tree_rows(label, t, got, law_of, false, {}, transport, false);
    failures += out.verify_guards(label) + ws.verify_guards(label);
    return failures;
}

// The DFlash builder over independent positions: each position's edges are its candidates'
// scores whatever the predecessor, so every node's law at depth i-1 is the softmax of position
// i's scores. The main chain is checked against the host directly (a greedy row's lowest rank
// attaining the maximum score with a one-hot law; a sampled row's FP32 softmax of the scores at
// its temperature and the inverse CDF of its counter key), the rest by check_tree_rows' replay
// over the FP64 independent laws. With transport the sampled tree rows are UniVer trees, checked
// by check_transport_row (their main chain included); greedy and chain rows are unchanged.
int candidate_tree_case(int width, int max_paths, bool transport = false) {
    constexpr int kSteps = 7, batch = 6;
    std::vector<int> ids(static_cast<std::size_t>(kCandidates) * kSteps * batch);
    std::vector<float> scores(ids.size());
    for (int r = 0; r < batch; ++r)
        for (int s = 0; s < kSteps; ++s) {
            // top_logprobs order: descending log-probabilities, a tie at ranks 2 and 3 of row 5.
            float value = -0.25F - 0.03125F * static_cast<float>(
                                                  (mix32(static_cast<std::uint32_t>(r * 31 + s)) >>
                                                   12) & 0x1fU);
            for (int c = 0; c < kCandidates; ++c) {
                const std::size_t at = (static_cast<std::size_t>(r) * kSteps + s) * kCandidates + c;
                ids[at]              = 3000 + r * 30011 + s * 409 + c * 7;
                scores[at]           = value;
                const bool tie       = r == 5 && c == 2;
                if (!tie)
                    value -= 0.0625F + 0.0625F * static_cast<float>(
                                                     (mix32(static_cast<std::uint32_t>(at) + 91U) >>
                                                      9) & 0x1fU);
            }
        }
    const std::vector<int> positions{10, 500, 4096, 77, 9000, 33};
    // Row 4 is a chain row of the round.
    const std::vector<int> extents{width - 1, width - 1, width - 1, width - 1, 4, width - 1};
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    const std::array<float, batch> temperature{1.0F, 0.0F, 0.6F, 1.4F, 1.0F, 0.0F};
    for (int r = 0; r < batch; ++r) {
        configs[static_cast<std::size_t>(r)].temperature = temperature[static_cast<std::size_t>(r)];
        configs[static_cast<std::size_t>(r)].seed        = 91U + 17U * static_cast<unsigned>(r);
    }
    DeviceBuffer dids = to_device(ids), dscores = to_device(scores),
                 dpositions = to_device(positions), dextents = to_device(extents),
                 dconfigs = to_device(configs);
    Tensor tids(dids.p, DType::I32, {kCandidates, kSteps, batch});
    Tensor tscores(dscores.p, DType::FP32, {kCandidates, kSteps, batch});
    Tensor tpositions(dpositions.p, DType::I32, {batch}), textents(dextents.p, DType::I32, {batch});
    const auto* cfg = static_cast<const ops::SamplingConfig*>(dconfigs.p);

    TreeBuildCase t{width, max_paths, kSteps, batch, ids, positions, extents, {}, configs};
    t.valid_in.resize(static_cast<std::size_t>(batch));
    for (int r = 0; r < batch; ++r)
        t.valid_in[static_cast<std::size_t>(r)] = extents[static_cast<std::size_t>(r)] + 1;
    const int columns = width - 1;
    TreeBuildBuffers out(width, batch, t.valid_in);
    {
        Tensor drafts(out.drafts.data(), DType::I32, {columns, batch});
        Tensor column_candidates(out.candidates.data(), DType::I32, {kCandidates, columns, batch});
        Tensor q(out.q.data(), DType::FP32, {kCandidates, columns, batch});
        Tensor tree_rows = rows_tensor(out.rows, batch);
        Tensor masks(out.masks.data(), DType::I32, {width, batch});
        Tensor valid(out.valid.data(), DType::I32, {batch});
        ops::candidate_tree(tids, tscores, tpositions, textents, cfg, {width, kSteps, max_paths, transport},
                            drafts, column_candidates, q, tree_rows, masks, valid, nullptr);
        cuda_synchronize();
    }
    const TreeBuildOutputs got = out.read();

    const std::string label = std::string(transport ? "transport " : "") + "candidate_tree W=" +
                              std::to_string(width) + " paths=" + std::to_string(max_paths);
    // The FP64 law of position `step`'s candidates at `law_temperature`.
    const auto position_law = [&](int r, int step, double law_temperature) {
        CandidateLaw law{};
        const std::size_t at = (static_cast<std::size_t>(r) * kSteps + step) * kCandidates;
        double maximum       = -std::numeric_limits<double>::infinity();
        for (int c = 0; c < kCandidates; ++c)
            maximum = std::max(maximum, static_cast<double>(scores[at + c]));
        double total = 0.0;
        for (int c = 0; c < kCandidates; ++c) {
            law[static_cast<std::size_t>(c)] =
                std::exp((static_cast<double>(scores[at + c]) - maximum) / law_temperature);
            total += law[static_cast<std::size_t>(c)];
        }
        for (auto& value : law) value /= total;
        return law;
    };
    int failures = 0;
    for (int r = 0; r < batch; ++r) {
        const bool greedy = !(temperature[static_cast<std::size_t>(r)] > 0.0F);
        if (transport && !greedy && extents[static_cast<std::size_t>(r)] == columns) continue;
        for (int c = 1; c <= kSteps; ++c) {
            const std::size_t at      = static_cast<std::size_t>(r) * columns + c - 1;
            const std::size_t lattice = static_cast<std::size_t>(r) * kSteps + c - 1;
            bool ok                   = true;
            for (int j = 0; j < kCandidates; ++j)
                ok = ok && got.candidates[at * kCandidates + j] == ids[lattice * kCandidates + j];
            int rank = -1;
            for (int j = 0; j < kCandidates; ++j)
                if (ids[lattice * kCandidates + j] == got.drafts[at]) rank = j;
            ok = ok && rank >= 0;
            if (ok && greedy) {
                // The lowest rank attaining the maximum score, with its exact one-hot law.
                int best = 0;
                for (int j = 1; j < kCandidates; ++j)
                    if (scores[lattice * kCandidates + j] > scores[lattice * kCandidates + best])
                        best = j;
                ok = rank == best;
                for (int j = 0; j < kCandidates; ++j)
                    ok = ok && got.q[at * kCandidates + j] == (j == best ? 1.0F : 0.0F);
            } else if (ok) {
                // The softmax at the row's temperature and its inverse CDF at the column's key.
                const auto law =
                    position_law(r, c - 1, static_cast<double>(temperature[static_cast<std::size_t>(r)]));
                double before = 0.0, mass = 0.0;
                for (int j = 0; j < kCandidates; ++j) {
                    ok = ok && std::fabs(got.q[at * kCandidates + j] -
                                         law[static_cast<std::size_t>(j)]) < 1e-5;
                    mass += got.q[at * kCandidates + j];
                    if (j < rank) before += got.q[at * kCandidates + j];
                }
                // Main-chain draft c - 1 carries its step as the sub-key.
                const double goal =
                    host_uniform(configs[static_cast<std::size_t>(r)].seed,
                                 positions[static_cast<std::size_t>(r)] + c - 1, 5, static_cast<std::uint32_t>(c - 1));
                ok = ok && goal >= before / mass - 1e-5 &&
                     goal < (before + got.q[at * kCandidates + rank]) / mass + 1e-5;
            }
            failures += expect(ok, label + " row " + std::to_string(r) + " main column " +
                                       std::to_string(c));
        }
    }
    const NodeLaw law_of = [&](int r, const ops::SpeculativeTreeRow& row, int node) {
        const bool greedy = !(temperature[static_cast<std::size_t>(r)] > 0.0F);
        return position_law(r, row.depth[node],
                            greedy ? 1.0 : static_cast<double>(temperature[static_cast<std::size_t>(r)]));
    };
    failures += check_tree_rows(label, t, got, law_of, false, {}, transport);
    failures += out.verify_guards(label);
    return failures;
}

// The leaf extension of a proposed chain (MTP's): the chain columns as given (drafts, candidates and
// laws bit for bit), then check_tree_rows' replay with leaf-only side columns over the given laws
// (sampled rows) or the temperature-1 softmax of the scores (greedy rows). Row 4 is a chain row;
// row 5's chain draft at step 2 is not among its candidates, so it takes no side column.
int chain_tree_case(int width, int max_paths) {
    constexpr int kSteps = 5, batch = 6, kRanks = 4;
    const int columns    = width - 1;
    std::vector<int> ids(static_cast<std::size_t>(kCandidates) * kSteps * batch);
    std::vector<float> laws(ids.size(), 0.0F);
    std::vector<float> scores(static_cast<std::size_t>(kRanks) * kSteps * batch);
    std::vector<int> drafts_in(static_cast<std::size_t>(columns) * batch, 0);
    const std::array<float, batch> temperature{1.0F, 0.0F, 0.7F, 0.0F, 1.0F, 1.3F};
    for (int r = 0; r < batch; ++r) {
        const bool greedy = !(temperature[static_cast<std::size_t>(r)] > 0.0F);
        for (int s = 0; s < kSteps; ++s) {
            const std::size_t step = static_cast<std::size_t>(r) * kSteps + s;
            for (int c = 0; c < kCandidates; ++c)
                ids[step * kCandidates + c] = 5000 + r * 40009 + s * 523 + c * 11;
            // Descending log-probabilities of ranks 0..3.
            float logprob = -0.1F - 0.05F * static_cast<float>(
                                                (mix32(static_cast<std::uint32_t>(r * 7 + s)) >> 9) & 7U);
            for (int j = 0; j < kRanks; ++j) {
                scores[step * kRanks + j] = logprob;
                logprob -= 0.25F + 0.125F * static_cast<float>(
                                                 (mix32(static_cast<std::uint32_t>(r * 13 + s * 3 + j)) >> 7) &
                                                 7U);
            }
            int chain_rank = 0;
            if (greedy) {
                laws[step * kCandidates] = 1.0F;
            } else {
                // The sampler's law over the four ranks at the row's temperature.
                double total = 0.0;
                std::array<double, kRanks> weight{};
                for (int j = 0; j < kRanks; ++j) {
                    weight[static_cast<std::size_t>(j)] =
                        std::exp(static_cast<double>(scores[step * kRanks + j]) /
                                 static_cast<double>(temperature[static_cast<std::size_t>(r)]));
                    total += weight[static_cast<std::size_t>(j)];
                }
                for (int j = 0; j < kRanks; ++j)
                    laws[step * kCandidates + j] =
                        static_cast<float>(weight[static_cast<std::size_t>(j)] / total);
                chain_rank = (s + r) % 3;
            }
            drafts_in[static_cast<std::size_t>(r) * columns + s] = ids[step * kCandidates + chain_rank];
        }
    }
    drafts_in[static_cast<std::size_t>(5) * columns + 2] = 999999;
    const std::vector<int> positions{20, 700, 4096, 81, 9000, 55};
    const std::vector<int> extents{columns, columns, columns, columns, 3, columns};
    std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
    for (int r = 0; r < batch; ++r) {
        configs[static_cast<std::size_t>(r)].temperature = temperature[static_cast<std::size_t>(r)];
        configs[static_cast<std::size_t>(r)].seed        = 313U + 29U * static_cast<unsigned>(r);
    }
    DeviceBuffer dids = to_device(ids), dlaws = to_device(laws), dscores = to_device(scores),
                 dpositions = to_device(positions), dextents = to_device(extents),
                 dconfigs = to_device(configs);
    Tensor tids(dids.p, DType::I32, {kCandidates, kSteps, batch});
    Tensor tlaws(dlaws.p, DType::FP32, {kCandidates, kSteps, batch});
    Tensor tscores(dscores.p, DType::FP32, {kRanks, kSteps, batch});
    Tensor tpositions(dpositions.p, DType::I32, {batch}), textents(dextents.p, DType::I32, {batch});
    const auto* cfg = static_cast<const ops::SamplingConfig*>(dconfigs.p);

    TreeBuildCase t{width, max_paths, kSteps, batch, ids, positions, extents, {}, configs};
    t.valid_in.resize(static_cast<std::size_t>(batch));
    for (int r = 0; r < batch; ++r)
        t.valid_in[static_cast<std::size_t>(r)] = extents[static_cast<std::size_t>(r)] + 1;
    TreeBuildBuffers out(width, batch, t.valid_in);
    write_buffer(out.drafts, drafts_in);
    {
        Tensor drafts(out.drafts.data(), DType::I32, {columns, batch});
        Tensor column_candidates(out.candidates.data(), DType::I32, {kCandidates, columns, batch});
        Tensor q(out.q.data(), DType::FP32, {kCandidates, columns, batch});
        Tensor tree_rows = rows_tensor(out.rows, batch);
        Tensor masks(out.masks.data(), DType::I32, {width, batch});
        Tensor valid(out.valid.data(), DType::I32, {batch});
        ops::candidate_chain_tree(tids, tlaws, tscores, tpositions, textents, cfg,
                                  {width, kSteps, max_paths}, drafts, column_candidates, q,
                                  tree_rows, masks, valid, nullptr);
        cuda_synchronize();
    }
    const TreeBuildOutputs got = out.read();

    const std::string label =
        "candidate_chain_tree W=" + std::to_string(width) + " paths=" + std::to_string(max_paths);
    int failures = 0;
    // The chain columns as given.
    for (int r = 0; r < batch; ++r) {
        for (int c = 1; c <= kSteps; ++c) {
            const std::size_t at   = static_cast<std::size_t>(r) * columns + c - 1;
            const std::size_t step = static_cast<std::size_t>(r) * kSteps + c - 1;
            bool ok                = got.drafts[at] == drafts_in[at];
            for (int j = 0; j < kCandidates; ++j)
                ok = ok && got.candidates[at * kCandidates + j] == ids[step * kCandidates + j] &&
                     std::memcmp(&got.q[at * kCandidates + j], &laws[step * kCandidates + j], 4) == 0;
            failures += expect(ok, label + " row " + std::to_string(r) + " chain column " +
                                       std::to_string(c));
        }
    }
    const NodeLaw law_of = [&](int r, const ops::SpeculativeTreeRow& row, int node) {
        const std::size_t step = static_cast<std::size_t>(r) * kSteps + row.depth[node];
        CandidateLaw law{};
        if (temperature[static_cast<std::size_t>(r)] > 0.0F) {
            for (int j = 0; j < kCandidates; ++j)
                law[static_cast<std::size_t>(j)] = static_cast<double>(laws[step * kCandidates + j]);
            return law;
        }
        double total = 0.0;
        for (int j = 0; j < kRanks; ++j) {
            law[static_cast<std::size_t>(j)] = std::exp(static_cast<double>(scores[step * kRanks + j]) -
                                                        static_cast<double>(scores[step * kRanks]));
            total += law[static_cast<std::size_t>(j)];
        }
        for (auto& value : law) value /= total;
        return law;
    };
    failures += check_tree_rows(label, t, got, law_of, true, {5});
    // A tree row without side columns keeps the rest of its drafts.
    bool untouched = true;
    for (int c = kSteps; c < columns; ++c)
        untouched = untouched && got.drafts[static_cast<std::size_t>(5) * columns + c] == 0;
    failures += expect(untouched, label + " row 5 drafts past its chain");
    failures += out.verify_guards(label);
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

// value_rows 6144 is the 27B's geometry, 4096 the 35B-A3B's (Q8 parent).
int gdn_conv_case(int vrows) {
    constexpr int qrows = 2048, krows = 2048, slots = 4;
    const int channels  = qrows + krows + vrows;
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
                failures += verify_exact(("tree conv " + std::to_string(channels) + " row " +
                                          std::to_string(b) + " path " +
                                          std::to_string(p) + " column " + std::to_string(i))
                                             .c_str(),
                                         column_of(tree_out, n, b, t.path_columns[p][i]),
                                         column_of(chain_out, length, 0, i));
        }
        for (int c = t.nodes; c < n; ++c)
            failures += verify_exact(
                ("tree conv " + std::to_string(channels) + " row " + std::to_string(b) +
                 " idle column " + std::to_string(c))
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

// Rows wider than 16 columns, which the closed form's route walks: a 24-column K=15 tree with side
// chains below main columns 0, 2 and 5 and below a side column, and a 20-column chain row.
std::vector<ops::SpeculativeTreeRow> wide_gdn_rows() {
    return {host_row(branched_parents(15, {{0, 3}, {2, 2}, {5, 1}, {16, 2}}), 15, true), chain_row(19, 15)};
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
                         bool tree, ops::GdnTreeRecordRoute route = ops::GdnTreeRecordRoute::Walk) {
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
                                               nullptr, route);
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

    // gdn-tree-closed-form (ops::GdnTreeRecordRoute::ClosedForm): the walk's records and idle zeros bit
    // for bit, and every live output within one BF16 ulp of the FP64 recurrence along its root path
    // (normalized q and k, a = exp(g), s <- a s + beta (v - a s k) k^T, o = scale s q).
    const auto closed = run(q, kv, v, g, beta, n, batch, valid, initial, true, ops::GdnTreeRecordRoute::ClosedForm);
    failures += verify_exact((label + " closed form key records").c_str(), closed.key_record, tree.key_record);
    failures += verify_exact((label + " closed form value records").c_str(), closed.value_record, tree.value_record);
    // Bitwise: an idle column's record words keep the 0xff fill, a NaN that never equals itself.
    const auto words = [](const std::vector<float>& values) {
        std::vector<std::uint32_t> bits(values.size());
        std::memcpy(bits.data(), values.data(), values.size() * sizeof(float));
        return bits;
    };
    failures += verify_exact((label + " closed form gate records").c_str(), words(closed.gate_record),
                             words(tree.gate_record));
    const auto normalized = [&](const std::vector<std::uint16_t>& data, std::size_t at) {
        std::array<double, dim> x{};
        double sum = 0.0;
        for (int i = 0; i < dim; ++i) {
            x[static_cast<std::size_t>(i)] = bf16_to_f32(data[at + static_cast<std::size_t>(i)]);
            sum += x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
        }
        const double inverse = 1.0 / std::sqrt(sum + 1e-6);
        for (double& e : x) { e *= inverse; }
        return x;
    };
    int differing = 0, outputs = 0;
    double worst = 0.0;
    for (int b = 0; b < batch; ++b) {
        const auto& t = rows[static_cast<std::size_t>(b)];
        for (int c = 0; c < t.nodes; ++c) {
            std::vector<int> path;
            for (int at = c; at >= 0; at = t.tree != 0 ? t.parent[at] : at - 1) { path.insert(path.begin(), at); }
            for (int h = 0; h < value_heads; ++h) {
                const int qk_head = h / (value_heads / qk_heads);
                std::vector<double> s(static_cast<std::size_t>(dim) * dim);
                const std::size_t base =
                    (static_cast<std::size_t>(initial[static_cast<std::size_t>(b)]) * value_heads + h) * dim * dim;
                for (std::size_t e = 0; e < s.size(); ++e) { s[e] = states[base + e]; }
                for (const int a : path) {
                    const std::size_t column = static_cast<std::size_t>(b) * n + a;
                    const auto key   = normalized(kv, (column * qk_heads + qk_head) * dim);
                    const double gate = std::exp(static_cast<double>(g[column * value_heads + h]));
                    const double bet  = beta[column * value_heads + h];
                    for (int d = 0; d < dim; ++d) {
                        double* row  = &s[static_cast<std::size_t>(d) * dim];
                        double dot   = 0.0;
                        for (int i = 0; i < dim; ++i) { dot += row[i] * key[static_cast<std::size_t>(i)]; }
                        const double value = bf16_to_f32(v[(column * value_heads + h) * dim + static_cast<std::size_t>(d)]);
                        const double delta = bet * (value - gate * dot);
                        for (int i = 0; i < dim; ++i) { row[i] = gate * row[i] + delta * key[static_cast<std::size_t>(i)]; }
                    }
                }
                const std::size_t column = static_cast<std::size_t>(b) * n + c;
                const auto query         = normalized(q, (column * qk_heads + qk_head) * dim);
                for (int d = 0; d < dim; ++d) {
                    double dot = 0.0;
                    for (int i = 0; i < dim; ++i) {
                        dot += s[static_cast<std::size_t>(d) * dim + i] * query[static_cast<std::size_t>(i)];
                    }
                    const double want       = scale * dot;
                    const std::size_t at    = (column * value_heads + h) * dim + static_cast<std::size_t>(d);
                    const double got        = bf16_to_f32(closed.out[at]);
                    // A BF16 rounding (at most 2^-8 of the value) plus FP32 slack.
                    const double tolerance  = std::abs(want) * 0x1p-8 * 1.01 + 1e-7;
                    worst                   = std::max(worst, std::abs(got - want) / std::max(std::abs(want), 1e-7));
                    differing += closed.out[at] != tree.out[at] ? 1 : 0;
                    ++outputs;
                    if (std::abs(got - want) > tolerance) {
                        ++failures;
                        if (failures < 8) {
                            std::cerr << label << " closed form row " << b << " column " << c << " head " << h << " dv "
                                      << d << ": " << got << " vs FP64 " << want << '\n';
                        }
                    }
                }
            }
        }
        bool idle = true;
        for (int c = t.nodes; c < n; ++c)
            for (std::size_t e = 0; e < v_column; ++e)
                idle = idle && closed.out[(static_cast<std::size_t>(b) * n + c) * v_column + e] == 0;
        failures += expect(idle, label + " closed form row " + std::to_string(b) + " idle columns");
    }
    std::cout << label << " closed form: " << differing << " of " << outputs
              << " outputs differ from the walk's bits; worst relative error against FP64 " << worst << '\n';
    // Rows wider than the closed form's 16 columns walk under its route.
    if (n > 16) {
        failures += verify_exact((label + " closed form route above 16 columns walks").c_str(), closed.out, tree.out);
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(speculative_tree) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // --fp32-logits: the acceptance cases on FP32 verification logits.
    if (argc == 2 && std::string_view(argv[1]) == "--fp32-logits") {
        g_logits_dtype = DType::FP32;
        try {
            failures += greedy_tree_case();
            failures += chain_equivalence_case();
            for (int m = 1; m <= 4; ++m) failures += distribution_case(m);
        } catch (const std::exception& error) {
            std::cerr << "speculative tree FP32 test threw: " << error.what() << '\n';
            return 1;
        }
        std::cout << (failures == 0 ? "speculative tree FP32 acceptance passed\n"
                                    : "speculative tree FP32 acceptance failed\n");
        return failures == 0 ? 0 : 1;
    }
    if (argc != 1) {
        std::cerr << "usage: infernix_speculative_tree_test [--fp32-logits]\n";
        return 2;
    }
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
        failures += selector_case(16, 8, true);
        failures += candidate_tree_case(16, 4);
        failures += candidate_tree_case(32, 8);
        failures += candidate_tree_case(24, 2);
        failures += candidate_tree_case(12, 8, true);
        failures += candidate_tree_case(32, 3, true);
        failures += transport_distribution_case();
        failures += chain_tree_case(16, 8);
        failures += chain_tree_case(9, 3);
        failures += chain_tree_case(32, 8);
        failures += gdn_conv_case(6144);
        failures += gdn_conv_case(4096);
        failures += gdn_record_case(48, gdn_rows(4, 9), 9);
        failures += gdn_record_case(32, gdn_rows(4, 9), 9);
        failures += gdn_record_case(48, nested_gdn_rows(), 16);
        failures += gdn_record_case(48, wide_gdn_rows(), 24);
        failures += gdn_record_case(32, wide_gdn_rows(), 24);
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
