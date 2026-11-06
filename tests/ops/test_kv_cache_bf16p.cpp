// Packed BF16 KV storage (KvCacheStorage::Bf16Packed, ops/kv_cache/bf16p_codec.cuh).
//
// Codec: K and V rows appended to a packed cache - normal rows and planted ones (zeros, negative zeros, subnormals,
// infinities and NaN, values far above and far below a row's bulk, an all-zero row, a wide row whose best window is
// not at its top, rows spanning exactly 14 binades and holding 6 and 7 escapes) - are read back and decoded on the host
// from the stored 384 bytes and side records alone (the exact oracle). A row has a packed form exactly when the window
// rule leaves it at most 6 escapes (zeros and subnormals are not escapes); such a row decodes to its BF16 bits (K and V
// alike) and its side record follows the format (base; escape count and exponent-0 bit; the escapes' exponents in
// column order; zero padding). A row with more escapes keeps its first 6 and sets the sequence's flag
// (PagedKVLayerView::unpackable); an append of packable rows leaves it clear.
//
// Attention: the same calls on a bf16 cache and a packed cache (physical pages in a scrambled order) on every BF16
// route - a 700-column prompt (tiled), a 300-column prompt with the bf16-prompt-split envelope, single columns
// (grouped decode), 4 and 64 columns (grouped), a batch of two sequences with valid columns, a verification tree, 65-96
// query rows (the packed storage's 96-row tile) with and without a verification tree, and
// the cached Op on its grouped, tiled and split routes - for the 24/4, 16/2 and 32/2 geometries; and past 16,384 keys
// (the packed prompt route's second segment: suspended and resumed CTA state, with and without the prompt split; the
// grouped producer warps over hundreds of pages; a 96-row tile whose first 64-row plan tile ends inside a split) for
// 24/4 and 16/2. The packed outputs meet the bf16 storage's
// criterion against the FP64 oracle over the represented K (BF16) and V (FP16 of the BF16 input), and equal the bf16
// storage's outputs bit for bit, and no append sets the flag.

#include "ops/op_family.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "infernix/ops/kv_cache_append.h"
#include "infernix/ops/softmax_attention.h"
#include "ops/host_parallel.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr int kD = 256, kPage = 64, kRowBytes = 384, kSideBytes = 8, kEscapes = 6, kWindow = 14;
constexpr float kScale = 0.0625F;
// The bf16 storage's criterion against the FP64 oracle (tests/ops/softmax_attention/causal_cache.cpp).
constexpr ReductionCriterion kCriterion{2.8e-3, 1.0e-3, 2.7e-3};
int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// The window rule (bf16p_codec.cuh): the window [base, base + 13] of normal exponents, the row's escapes (normal
// values outside it, infinities and NaN) and whether a value has exponent 0 (a zero or a subnormal).
struct Window {
    int base    = 1;
    int escapes = 0;
    bool zeros  = false;
};

Window window_of(const std::uint16_t* row) {
    int lowest = 255, highest = 0;
    std::array<int, 256> histogram{};
    for (int d = 0; d < kD; ++d) {
        const int e = (row[d] >> 7) & 0xFF;
        if (e != 0 && e != 255) {
            lowest  = std::min(lowest, e);
            highest = std::max(highest, e);
            ++histogram[e];
        }
    }
    Window out;
    if (highest < lowest) {
        out.base = 1;
    } else if (highest - lowest < kWindow) {
        out.base = std::clamp(lowest, 1, 240);
    } else {
        int best = -1;
        for (int b = 1; b <= 240; ++b) {
            int count = 0;
            for (int e = b; e < b + kWindow; ++e) { count += histogram[e]; }
            if (count > best) {
                best     = count;
                out.base = b;
            }
        }
    }
    for (int d = 0; d < kD; ++d) {
        const int e = (row[d] >> 7) & 0xFF;
        out.zeros   = out.zeros || e == 0;
        out.escapes += (e != 0 && (e == 255 || e < out.base || e >= out.base + kWindow)) ? 1 : 0;
    }
    return out;
}

// A row decoded from its 384 bytes and side record alone: nibbles 0-13 are base + nibble, 14 exponent 0, and 15 takes
// the next escape exponent.
std::vector<std::uint16_t> decode_packed(const std::uint8_t* row, const std::uint8_t* side) {
    std::vector<std::uint16_t> out(kD);
    int escape = 0;
    for (int d = 0; d < kD; ++d) {
        const int code     = row[d];
        const int nibble   = (row[256 + d / 2] >> (4 * (d & 1))) & 0xF;
        const int exponent = nibble == 15   ? (escape < kEscapes ? side[2 + escape++] : 0)
                             : nibble == 14 ? 0
                                            : side[0] + nibble;
        out[d]             = static_cast<std::uint16_t>(((code & 0x80) << 8) | (exponent << 7) | (code & 0x7F));
    }
    return out;
}

// A paged cache of one storage: planes [extent, 64, H, pages], a block table [logical, rows] of physical pages in a
// scrambled order (the same for every storage of a seed), every byte poisoned; a pinned unpackable flag per table row.
struct Cache {
    KvCacheStorage storage;
    int heads, logical, physical, rows;
    PagedKVStorageLayout layout;
    DeviceBuffer k, v, ks, vs, table;
    std::vector<std::int32_t> host_table;
    std::int32_t* flags = nullptr;

    Cache(const Cache&)            = delete;
    Cache& operator=(const Cache&) = delete;
    ~Cache() { cudaFreeHost(flags); }

    // Whether every table row's flag is clear.
    [[nodiscard]] bool all_packed() const { return std::all_of(flags, flags + rows, [](std::int32_t f) { return f == 0; }); }

    Cache(KvCacheStorage s, int kv_heads, int logical_pages, int table_rows, std::uint32_t seed)
        : storage(s), heads(kv_heads), logical(logical_pages), physical(logical_pages * table_rows),
          rows(table_rows), layout(paged_kv_storage_layout(s, kD)) {
        if (cudaMallocHost(&flags, sizeof(std::int32_t) * static_cast<std::size_t>(table_rows)) != cudaSuccess) {
            throw std::runtime_error("cannot allocate the pinned unpackable flags");
        }
        std::fill(flags, flags + table_rows, 0);
        const auto bytes = [&](const PagedKVVectorLayout& vector, bool scale) {
            const int extent  = scale ? vector.scale_leading_extent : vector.data_leading_extent;
            const DType dtype = scale ? vector.scale_dtype : vector.data_dtype;
            return static_cast<std::size_t>(extent) * dtype_size(dtype) * kPage * heads * physical;
        };
        k = DeviceBuffer(bytes(layout.key, false));
        v = DeviceBuffer(bytes(layout.value, false));
        k.fill(0xA5);
        v.fill(0x5A);
        if (layout.key.has_scale()) {
            ks = DeviceBuffer(bytes(layout.key, true));
            ks.fill(0xEE);
        }
        if (layout.value.has_scale()) {
            vs = DeviceBuffer(bytes(layout.value, true));
            vs.fill(0xEE);
        }
        host_table.resize(static_cast<std::size_t>(physical));
        std::iota(host_table.begin(), host_table.end(), 0);
        std::shuffle(host_table.begin(), host_table.end(), std::mt19937(seed));
        table = DeviceBuffer(host_table.size() * 4);
        table.copy_from_host(host_table.data(), host_table.size() * 4);
    }

    Tensor plane(const DeviceBuffer& buffer, const PagedKVVectorLayout& vector, bool scale) const {
        if (scale && !vector.has_scale()) { return Tensor{}; }
        return Tensor(buffer.p, scale ? vector.scale_dtype : vector.data_dtype,
                      {scale ? vector.scale_leading_extent : vector.data_leading_extent, kPage, heads, physical});
    }

    PagedKVBatchLayerView batch() const {
        PagedKVBatchLayerView out;
        out.k_pages       = plane(k, layout.key, false);
        out.v_pages       = plane(v, layout.value, false);
        out.k_scale_pages = plane(ks, layout.key, true);
        out.v_scale_pages = plane(vs, layout.value, true);
        out.block_tables  = Tensor(table.p, DType::I32, {logical, rows});
        out.head_dim      = kD;
        out.num_kv_heads  = heads;
        out.storage       = storage;
        out.unpackable    = flags;
        return out;
    }

    PagedKVLayerView single(int row) const {
        PagedKVLayerView out;
        out.k_pages       = plane(k, layout.key, false);
        out.v_pages       = plane(v, layout.value, false);
        out.k_scale_pages = plane(ks, layout.key, true);
        out.v_scale_pages = plane(vs, layout.value, true);
        out.block_table   = Tensor(static_cast<std::int32_t*>(table.p) + static_cast<std::size_t>(row) * logical,
                                   DType::I32, {logical});
        out.head_dim      = kD;
        out.num_kv_heads  = heads;
        out.storage       = storage;
        out.unpackable    = flags + row;
        return out;
    }

    // (table row, position, head)'s element index in a plane: its data bytes start at the data extent times it, its
    // side record at 8 times it.
    std::size_t element(int table_row, int position, int head) const {
        const int page = host_table[static_cast<std::size_t>(table_row) * logical + position / kPage];
        return (static_cast<std::size_t>(page) * heads + head) * kPage + position % kPage;
    }
};

std::vector<std::uint8_t> download(const DeviceBuffer& buffer) { return from_device<std::uint8_t>(buffer, buffer.bytes); }

// K or V rows [256, H, T] of BF16 bits: N(0, 1) values with planted rows at tokens 1..6 (H >= 2). `extreme` adds the
// non-finite row, plants far outside a row's bulk (they would swamp attention's softmax) and a row with no packed form
// (codec only).
std::vector<std::uint16_t> make_rows(int heads, int tokens, std::uint32_t seed, bool extreme) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(kD) * heads * tokens);
    for (auto& x : out) { x = f32_to_bf16(normal(rng)); }
    const auto row = [&](int token, int head) {
        return out.data() + (static_cast<std::size_t>(token) * heads + head) * kD;
    };
    if (tokens < 7) { return out; }
    // Token 1: zeros (head 0, three) and negative zeros (head 1, four).
    for (int i = 0; i < 3; ++i) { row(1, 0)[17 * i] = 0x0000; }
    for (int i = 0; i < 4; ++i) { row(1, 1)[13 * i + 1] = 0x8000; }
    // Token 2: subnormals, seven (head 0) and six (head 1): exponent 0, no escapes.
    for (int i = 0; i < 7; ++i) { row(2, 0)[35 * i + 3] = static_cast<std::uint16_t>(0x0001 + i); }
    for (int i = 0; i < 6; ++i) { row(2, 1)[41 * i + 5] = static_cast<std::uint16_t>(0x8011 + i); }
    // Token 3: an all-zero row; with `extreme`, a row holding both infinities and a NaN (3 escapes).
    std::fill(row(3, 0), row(3, 0) + kD, std::uint16_t{0});
    if (extreme) {
        row(3, 1)[0] = 0x7F80;
        row(3, 1)[1] = 0xFF80;
        row(3, 1)[2] = 0x7FC1;
    }
    // Token 4: one value far above the bulk and one far below (the window stays at the bulk: two escapes).
    row(4, 0)[100] = f32_to_bf16(extreme ? 1.0e30F : 64.0F);
    row(4, 0)[200] = f32_to_bf16(extreme ? 1.0e-30F : 0x1p-60F);
    // Token 5: a wide row whose best window does not start at its lowest exponent - its bulk held to 2^-10 <= |x| (12
    // binades) and 6 values 2^20 times smaller (exactly 6 escapes), or with `extreme` every other value 2^-20 times
    // smaller (128 escapes: no packed form).
    if (extreme) {
        for (int d = 0; d < kD; d += 2) { row(5, 0)[d] = f32_to_bf16(bf16_to_f32(row(5, 0)[d]) * 0x1p-20F); }
    } else {
        for (int d = 0; d < kD; ++d) {
            const float x = bf16_to_f32(row(5, 0)[d]);
            if (std::fabs(x) < 0x1p-10F) { row(5, 0)[d] = f32_to_bf16(std::copysign(0x1p-10F, x)); }
        }
        for (int i = 0; i < 6; ++i) { row(5, 0)[41 * i + 7] = f32_to_bf16(bf16_to_f32(row(5, 0)[41 * i + 7]) * 0x1p-20F); }
    }
    // Token 6: magnitudes over exactly 14 binades (no escape), and the same with values one binade above them: 6 (a full
    // side record) or, with `extreme`, 7 (no packed form).
    const int top = extreme ? 0 : -14;
    for (int d = 0; d < kD; ++d) {
        row(6, 0)[d] = f32_to_bf16(std::ldexp(d % 3 == 0 ? -1.0F : 1.0F, top + d % kWindow));
        row(6, 1)[d] = f32_to_bf16(std::ldexp(1.0F, top + d % kWindow));
    }
    for (int i = 0; i < (extreme ? 7 : 6); ++i) { row(6, 1)[37 * i + 2] = f32_to_bf16(std::ldexp(1.0F, top + kWindow)); }
    return out;
}

void run_codec(int heads) {
    const int tokens = 64 + 7;
    const auto k = make_rows(heads, tokens, 11, true), v = make_rows(heads, tokens, 12, true);
    Cache packed(KvCacheStorage::Bf16Packed, heads, 4, 1, 77);
    std::vector<std::int32_t> positions(tokens);
    std::iota(positions.begin(), positions.end(), 3); // a call starting inside a page and crossing into the next
    const DeviceBuffer kd = to_device(k), vd = to_device(v), pd = to_device(positions);
    const Tensor kt(kd.p, DType::BF16, {kD, heads, tokens}), vt(vd.p, DType::BF16, {kD, heads, tokens});
    const Tensor pt(pd.p, DType::I32, {tokens});
    ops::kv_cache_append(kt, vt, pt, packed.single(0), nullptr);
    cuda_synchronize();
    const auto pk = download(packed.k), pv = download(packed.v), sk = download(packed.ks), sv = download(packed.vs);
    int packed_rows = 0, unpackable_rows = 0, zero_rows = 0;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            for (int role = 0; role < 2; ++role) {
                const std::uint16_t* source = (role ? v : k).data() + (static_cast<std::size_t>(t) * heads + h) * kD;
                const std::size_t at        = packed.element(0, positions[t], h);
                const std::uint8_t* row     = (role ? pv : pk).data() + at * kRowBytes;
                const std::uint8_t* side    = (role ? sv : sk).data() + at * kSideBytes;
                const Window w              = window_of(source);
                const std::string where     = "codec H=" + std::to_string(heads) + (role ? " V" : " K") +
                                          " token " + std::to_string(t) + " head " + std::to_string(h);
                const int stored = std::min(w.escapes, kEscapes);
                check(side[0] == w.base && side[1] == (stored | (w.zeros ? 0x80 : 0)),
                      where + ": base, escape count and exponent-0 bit");
                bool zero = true;
                for (int b = 2 + stored; b < kSideBytes; ++b) { zero = zero && side[b] == 0; }
                check(zero, where + ": unused side bytes zero");
                if (w.escapes <= kEscapes) {
                    ++packed_rows;
                    zero_rows += w.zeros ? 1 : 0;
                    const auto decoded = decode_packed(row, side);
                    check(std::equal(decoded.begin(), decoded.end(), source), where + ": decodes to its BF16 bits");
                } else {
                    ++unpackable_rows;
                }
            }
        }
    }
    check(unpackable_rows >= 4 && zero_rows >= 4 && packed_rows > unpackable_rows,
          "codec: the planted rows reach exponent-0 codes and rows with no packed form");
    check(packed.flags[0] == 1, "codec: a row with no packed form sets the sequence's flag");
    // Rows that all pack leave a cleared flag clear.
    packed.flags[0]   = 0;
    const auto normal = make_rows(heads, tokens, 13, false);
    const DeviceBuffer nd = to_device(normal);
    const Tensor nt(nd.p, DType::BF16, {kD, heads, tokens});
    ops::kv_cache_append(nt, nt, pt, packed.single(0), nullptr);
    cuda_synchronize();
    check(packed.flags[0] == 0, "codec: packable rows leave the flag clear");
    std::printf("codec H=%d: %d packed rows (%d with exponent-0 values), %d with no packed form\n", heads, packed_rows,
                zero_rows, unpackable_rows);
}

// A bf16 cache and a packed cache taking the same calls, and the host copy of what every live column appended: K as
// its BF16 value, V as the FP16 value the bf16 storage represents.
struct Pair {
    ops::AttentionHeadGeometry geometry;
    int capacity; // positions a table row holds
    Cache plain, packed;
    std::vector<double> keys, values; // [row][position][head][d]

    Pair(int q_heads, int kv_heads, int logical_pages = 24, int table_rows = 2)
        : geometry{kD, q_heads, kv_heads}, capacity(logical_pages * kPage),
          plain(KvCacheStorage::BFloat16, kv_heads, logical_pages, table_rows, 5),
          packed(KvCacheStorage::Bf16Packed, kv_heads, logical_pages, table_rows, 5),
          keys(static_cast<std::size_t>(table_rows) * capacity * kv_heads * kD), values(keys.size()) {}

    std::size_t at(int row, int position, int head) const {
        return ((static_cast<std::size_t>(row) * capacity + position) * geometry.kv_heads + head) * kD;
    }
};

ops::CausalAttentionExecutionEnvelope envelope_of(const Pair& pair, bool split) {
    ops::CausalAttentionExecutionEnvelope envelope{1, static_cast<std::uint32_t>(pair.capacity)};
    envelope.bf16_prompt_split = split;
    return envelope;
}

// One call per storage, run by `call(cache, workspace, out)`: the packed output of the live columns must meet the
// criterion against the FP64 oracle, and every output bit must equal the bf16 storage's.
template <class Call>
void compare(Pair& pair, std::size_t outputs, const std::vector<double>& oracle, const std::vector<int>& live,
             int width, bool split, int batch, const std::string& label, DeviceExecutionView execution,
             const Call& call) {
    std::vector<std::uint16_t> bits[2];
    for (int which = 0; which < 2; ++which) {
        Cache& cache            = which ? pair.packed : pair.plain;
        const std::size_t bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
            pair.geometry, cache.storage, envelope_of(pair, split), batch, width, width, execution);
        DeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
        WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
        DeviceBuffer out(outputs * 2);
        out.fill(0x7F);
        call(cache, workspace, out);
        cuda_synchronize(execution.stream);
        bits[which] = from_device<std::uint16_t>(out, outputs);
    }
    const std::size_t column = static_cast<std::size_t>(kD) * pair.geometry.query_heads;
    std::vector<double> got;
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < live[b]; ++j) {
            const std::size_t base = (static_cast<std::size_t>(b) * width + j) * column;
            for (std::size_t i = 0; i < column; ++i) { got.push_back(bf16_to_f32(bits[1][base + i])); }
        }
    }
    failures += verify_reduction(label + " (packed vs FP64 oracle)", got, oracle, kCriterion);
    check(bits[0] == bits[1], label + ": packed outputs equal the bf16 storage's bits");
    check(pair.packed.all_packed(), label + ": every appended row packed");
}

// The FP64 oracle of the live columns: column j of batch row b at positions[b][j] over its table row's appended
// keys; a tree column only over the keys before the call's first position and its ancestors'.
std::vector<double> oracle(const Pair& pair, const std::vector<std::uint16_t>& q, int width, int batch,
                           const std::vector<int>& table_rows, const std::vector<std::int32_t>& positions,
                           const std::vector<int>& live, const std::vector<std::uint32_t>& tree) {
    const int qh = pair.geometry.query_heads;
    std::vector<double> out;
    for (int b = 0; b < batch; ++b) {
        const std::size_t first_column = static_cast<std::size_t>(b) * width;
        const int row = table_rows[b], first = positions[first_column], columns = live[b];
        const int visible = positions[first_column + columns - 1] + 1;
        std::vector<double> part(static_cast<std::size_t>(kD) * qh * columns);
        naive_dense_softmax_attention(
            pair.geometry, columns, visible, kScale,
            [&](int d, int head, int j) {
                return double(bf16_to_f32(q[((first_column + j) * qh + head) * kD + d]));
            },
            [&](int d, int head, int key) { return pair.keys[pair.at(row, key, head) + d]; },
            [&](int d, int head, int key) { return pair.values[pair.at(row, key, head) + d]; },
            [&](int j, int key) {
                if (key > positions[first_column + j]) { return false; }
                const int offset = key - first;
                return tree.empty() || offset < 0 || ((tree[first_column + j] >> offset) & 1U) != 0U;
            },
            [&](int d, int head, int j, double value) {
                part[(static_cast<std::size_t>(j) * qh + head) * kD + d] = value;
            },
            static_cast<int>(host_thread_count()));
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::vector<std::uint16_t> random_bits(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> out(count);
    for (auto& x : out) { x = f32_to_bf16(normal(rng)); }
    return out;
}

// Appends W columns to each of B table rows (row b's live columns at starts[b], starts[b] + 1, ...; its masked tail
// repeating its last live position) and attends.
void append_attend(Pair& pair, int width, const std::vector<int>& table_rows, const std::vector<int>& starts,
                   const std::vector<int>& valid, const std::vector<std::uint32_t>& tree, bool split,
                   std::uint32_t seed, const std::string& label, DeviceExecutionView execution) {
    const int batch = static_cast<int>(table_rows.size()), qh = pair.geometry.query_heads;
    const int kvh   = pair.geometry.kv_heads;
    const auto q    = random_bits(static_cast<std::size_t>(kD) * qh * width * batch, seed);
    std::vector<std::uint16_t> k, v;
    for (int b = 0; b < batch; ++b) {
        const auto kb = make_rows(kvh, width, seed + 100 + b, false), vb = make_rows(kvh, width, seed + 200 + b, false);
        k.insert(k.end(), kb.begin(), kb.end());
        v.insert(v.end(), vb.begin(), vb.end());
    }
    std::vector<int> live(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> positions(static_cast<std::size_t>(width) * batch);
    for (int b = 0; b < batch; ++b) {
        live[b] = valid.empty() ? width : valid[b];
        for (int t = 0; t < width; ++t) {
            positions[static_cast<std::size_t>(b) * width + t] = starts[b] + std::min(t, live[b] - 1);
        }
        for (int t = 0; t < live[b]; ++t) {
            for (int h = 0; h < kvh; ++h) {
                const std::size_t source = ((static_cast<std::size_t>(b) * width + t) * kvh + h) * kD;
                const std::size_t target = pair.at(table_rows[b], starts[b] + t, h);
                for (int d = 0; d < kD; ++d) {
                    pair.keys[target + d]   = bf16_to_f32(k[source + d]);
                    pair.values[target + d] = fp16_to_f32(f32_to_fp16_saturated(bf16_to_f32(v[source + d])));
                }
            }
        }
    }
    const auto reference = oracle(pair, q, width, batch, table_rows, positions, live, tree);
    const DeviceBuffer qd = to_device(q), kd = to_device(k), vd = to_device(v), pd = to_device(positions),
                       rd = to_device(table_rows), valid_d = to_device(valid.empty() ? std::vector<int>{0} : valid),
                       tree_d = to_device(tree.empty() ? std::vector<std::uint32_t>{0} : tree);
    const Tensor qt(qd.p, DType::BF16, {kD, qh, width, batch});
    const Tensor kt(kd.p, DType::BF16, {kD, kvh, width, batch}), vt(vd.p, DType::BF16, {kD, kvh, width, batch});
    const Tensor pt(pd.p, DType::I32, {width, batch}), rt(rd.p, DType::I32, {batch});
    const Tensor valid_t = valid.empty() ? Tensor{} : Tensor(valid_d.p, DType::I32, {batch});
    const Tensor tree_t  = tree.empty() ? Tensor{} : Tensor(tree_d.p, DType::I32, {width, batch});
    compare(pair, q.size(), reference, live, width, split, batch, label, execution,
            [&](Cache& cache, WorkspaceArena& workspace, DeviceBuffer& out) {
                Tensor ot(out.p, DType::BF16, {kD, qh, width, batch});
                ops::causal_softmax_attention(qt, kt, vt, pt, valid_t, rt, tree_t, pair.geometry, kScale,
                                              cache.batch(), envelope_of(pair, split), workspace, ot, execution);
            });
}

// The cached Op over table row `row`: W columns at position, position + 1, ...
void cached(Pair& pair, int row, int position, int width, bool split, std::uint32_t seed, const std::string& label,
            DeviceExecutionView execution) {
    const int qh = pair.geometry.query_heads;
    const auto q = random_bits(static_cast<std::size_t>(kD) * qh * width, seed);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
    std::iota(positions.begin(), positions.end(), position);
    const auto reference  = oracle(pair, q, width, 1, {row}, positions, {width}, {});
    const DeviceBuffer qd = to_device(q), pd = to_device(positions);
    const Tensor qt(qd.p, DType::BF16, {kD, qh, width}), pt(pd.p, DType::I32, {width});
    compare(pair, q.size(), reference, {width}, width, split, 1, label, execution,
            [&](Cache& cache, WorkspaceArena& workspace, DeviceBuffer& out) {
                Tensor ot(out.p, DType::BF16, {kD, qh, width});
                ops::causal_softmax_attention_cached(qt, pt, pair.geometry, kScale, cache.single(row),
                                                     envelope_of(pair, split), workspace, ot, execution);
            });
}

void run_attention(int q_heads, int kv_heads, DeviceExecutionView execution) {
    const std::string g = std::to_string(q_heads) + "/" + std::to_string(kv_heads) + " ";
    Pair pair(q_heads, kv_heads);
    // Table row 0: a 700-column prompt (tiled), single columns (grouped decode), 4 and 64 columns (grouped).
    append_attend(pair, 700, {0}, {0}, {}, {}, false, 1, g + "prompt 700", execution);
    append_attend(pair, 1, {0}, {700}, {}, {}, false, 2, g + "decode", execution);
    append_attend(pair, 1, {0}, {701}, {}, {}, false, 3, g + "decode again", execution);
    append_attend(pair, 4, {0}, {702}, {}, {}, false, 4, g + "4 columns", execution);
    append_attend(pair, 64, {0}, {706}, {}, {}, false, 5, g + "64 columns", execution);
    // Both table rows in one batch (row 1 starting empty), then rows of 8 and 5 valid columns.
    append_attend(pair, 3, {0, 1}, {770, 0}, {3, 3}, {}, false, 6, g + "batch of two", execution);
    append_attend(pair, 8, {0, 1}, {773, 3}, {8, 5}, {}, false, 7, g + "batch of two, 8 and 5 valid", execution);
    // A verification tree of 8 columns: a chain of 4, then columns hung below columns 3, 3, 1 and 1.
    const std::vector<std::uint32_t> tree{0x1, 0x3, 0x7, 0xF, 0x1F, 0x2F, 0x43, 0x83};
    append_attend(pair, 8, {0}, {781}, {8}, tree, false, 8, g + "tree of 8", execution);
    // 65-96 query rows (the packed storage's 96-row tile): 96 rows, then 78 or so, then a 96-row verification tree
    // (a chain of half its columns, the rest hung one below each of them).
    const int group = q_heads / kv_heads, wide = 96 / group, narrower = std::max(65 / group + 1, (wide * 13) / 16);
    append_attend(pair, wide, {0}, {789}, {}, {}, false, 13, g + std::to_string(wide * group) + " rows", execution);
    append_attend(pair, narrower, {0}, {789 + wide}, {}, {}, false, 14,
                  g + std::to_string(narrower * group) + " rows", execution);
    std::vector<std::uint32_t> wide_tree(static_cast<std::size_t>(wide));
    for (int i = 0; i < wide; ++i) {
        wide_tree[static_cast<std::size_t>(i)] =
            i < wide / 2 ? (2U << i) - 1U : wide_tree[static_cast<std::size_t>(i - wide / 2)] | (1U << i);
    }
    append_attend(pair, wide, {0}, {789 + wide + narrower}, {wide}, wide_tree, false, 15,
                  g + "tree of " + std::to_string(wide), execution);
    // Table row 1: a 300-column prompt with the bf16-prompt-split envelope.
    append_attend(pair, 300, {1}, {8}, {}, {}, true, 9, g + "prompt 300, prompt split", execution);
    // The cached Op: grouped, tiled, tiled with the prompt split.
    cached(pair, 0, 780, 5, false, 10, g + "cached 5", execution);
    cached(pair, 0, 400, 300, false, 11, g + "cached 300", execution);
    cached(pair, 1, 100, 200, true, 12, g + "cached 200, prompt split", execution);
}

// Appends positions [first, first + count) of table row 0 to both caches (no attention), in calls of up to 4,096.
void fill(Pair& pair, int first, int count, std::uint32_t seed) {
    const int kvh = pair.geometry.kv_heads;
    for (int begin = first; begin < first + count; begin += 4096) {
        const int width = std::min(4096, first + count - begin);
        const auto k = make_rows(kvh, width, seed + begin, false), v = make_rows(kvh, width, seed + begin + 1, false);
        std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
        std::iota(positions.begin(), positions.end(), begin);
        for (int t = 0; t < width; ++t) {
            for (int h = 0; h < kvh; ++h) {
                const std::size_t source = (static_cast<std::size_t>(t) * kvh + h) * kD;
                const std::size_t target = pair.at(0, begin + t, h);
                for (int d = 0; d < kD; ++d) {
                    pair.keys[target + d]   = bf16_to_f32(k[source + d]);
                    pair.values[target + d] = fp16_to_f32(f32_to_fp16_saturated(bf16_to_f32(v[source + d])));
                }
            }
        }
        const DeviceBuffer kd = to_device(k), vd = to_device(v), pd = to_device(positions);
        const Tensor kt(kd.p, DType::BF16, {kD, kvh, width}), vt(vd.p, DType::BF16, {kD, kvh, width});
        const Tensor pt(pd.p, DType::I32, {width});
        ops::kv_cache_append(kt, vt, pt, pair.plain.single(0), nullptr);
        ops::kv_cache_append(kt, vt, pt, pair.packed.single(0), nullptr);
        cuda_synchronize();
    }
}

// Past 16,384 keys: the packed prompt route runs two segments, its CTAs' state suspended and resumed between them.
void run_long(int q_heads, int kv_heads, DeviceExecutionView execution) {
    const std::string g = std::to_string(q_heads) + "/" + std::to_string(kv_heads) + " long ";
    Pair pair(q_heads, kv_heads, 272, 1);
    fill(pair, 0, 16700, 40);
    cached(pair, 0, 16400, 280, false, 41, g + "cached 280 over two segments", execution);
    cached(pair, 0, 16200, 280, true, 42, g + "cached 280 over two segments, prompt split", execution);
    append_attend(pair, 280, {0}, {16700}, {}, {}, false, 43, g + "prompt 280 over two segments", execution);
    append_attend(pair, 1, {0}, {16980}, {}, {}, false, 44, g + "decode", execution);
    append_attend(pair, 4, {0}, {16981}, {}, {}, false, 45, g + "4 columns", execution);
    // 96 query rows (the packed storage's 96-row tile) where the first 64-row plan tile's keys end on a key-tile
    // boundary inside a split of many key tiles: its rows must stop there, as that tile's own CTA does (the next key
    // tile, masked for all their keys, would still rescale their state by exp2 of a rounding residual).
    const int group = q_heads / kv_heads, wide = 96 / group, plan_tokens = (64 + group - 1) / group;
    int at = 16985;
    while ((at + plan_tokens) % 32 != 0) ++at;
    fill(pair, 16985, at - 16985, 46);
    append_attend(pair, wide, {0}, {at}, {}, {}, false, 47, g + "96 rows, plan tile ending on a key tile", execution);
}

} // namespace

INFERNIX_OP_TEST(kv_cache_bf16p) {
    if (cuda_unavailable()) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    try {
        int sms = 0;
        cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "SM count");
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        const DeviceExecutionView execution{stream, sms};
        run_codec(4);
        run_codec(2);
        run_attention(24, 4, execution);
        run_attention(16, 2, execution);
        run_attention(32, 2, execution);
        run_long(24, 4, execution);
        run_long(16, 2, execution);
        cuda_check(cudaStreamDestroy(stream), "stream");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "kv_cache_bf16p checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
