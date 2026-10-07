// Causal attention over the vector-quantized KV formats (Vq2, Q4KeyVq2Value) on every route:
// grouped decode (W <= 8, one and several rows, masked columns), parallel verification widths,
// verification trees, the wide prompt kernel with and without key splits, and the cached entry.
//
// The FP64 oracle reads the cache as the device stored it and applies the format's read rule per
// query position q (softmax_attention.h): a sink key or a key at most kKVWindowRecentTokens before
// q is read exactly (the call's own keys as the INT8-G64 rows of their rotated inputs, older keys
// and a cached call's keys from their window slot when its tag matches the stored codes); every
// other key, including a call's own keys far enough before q, reads its stored codes (codebook or
// 4-bit level times the row scale). Q is rotated exactly and the output inverse-rotated, in FP64.
// One case leaves stale window slots behind on purpose. After a wide call, the window committed
// from staging is checked byte for byte.

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "kv_cache_vq_reference.h"
#include "infernix/ops/kv_cache_append.h"
#include "infernix/ops/softmax_attention.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

// The launch view of the current device on the legacy stream.
DeviceExecutionView test_execution() {
    static const int multiprocessors = [] {
        int device = 0;
        int count  = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
        return count;
    }();
    return {nullptr, multiprocessors};
}

constexpr int kDim   = 256;
constexpr int kPage  = 64;
constexpr int kSlots = kKVWindowSlots;

// Native INT8 Q encoding dominates the kernels' error, as for the INT8-G64 cache.
constexpr ReductionCriterion kCriterion{2.0e-2, 1.1e-3, 4.0e-2};

struct Geometry {
    const char* name;
    int q_heads;
    int kv_heads;
};

constexpr Geometry kGeometries[] = {{"h24-kv4", 24, 4}, {"h16-kv2", 16, 2}};

struct Snapshot {
    std::vector<std::uint8_t> k, v;
    std::vector<std::uint16_t> ks, vs;
    std::vector<std::int8_t> wk, wv;
    std::vector<std::uint16_t> wks, wvs;
    std::vector<std::uint32_t> tags;
};

class VqCache {
public:
    VqCache(KvCacheStorage storage, int kv_heads, int max_context, int rows)
        : storage_(storage), kv_heads_(kv_heads), key_bytes_(storage == KvCacheStorage::Vq2 ? 64 : 128),
          logical_((max_context + kPage - 1) / kPage), rows_(rows),
          physical_(rows * logical_ + 1), tables_(static_cast<std::size_t>(rows) * logical_),
          k_(plane(key_bytes_)), v_(plane(64)), ks_(plane(2)), vs_(plane(2)),
          wk_(window(kDim)), wv_(window(kDim)), wks_(window(8)), wvs_(window(8)), wtags_(window(8)) {
        // Row r's logical page p lives on physical page 1 + r * logical + (logical - 1 - p).
        for (int r = 0; r < rows; ++r)
            for (int p = 0; p < logical_; ++p)
                tables_[static_cast<std::size_t>(r) * logical_ + p] = 1 + r * logical_ + (logical_ - 1 - p);
        d_tables_ = to_device(tables_);
        // The window is sequence state indexed by state slot, not by KV table row: row r's
        // window lives in slot rows - 1 - r.
        std::vector<std::int32_t> slots(static_cast<std::size_t>(rows));
        for (int r = 0; r < rows; ++r) slots[static_cast<std::size_t>(r)] = window_slot(r);
        d_slots_ = to_device(slots);
        // Unwritten rows hold NaN-patterned bytes: no visible key may ever read them.
        for (auto* buffer : {&k_, &v_, &ks_, &vs_}) buffer->fill(0xff);
        for (auto* buffer : {&wk_, &wv_, &wks_, &wvs_, &wtags_}) buffer->fill(0);
    }

    PagedKVBatchLayerView batch() {
        return {
            .k_pages       = Tensor(k_.data(), DType::U8, {key_bytes_, kPage, kv_heads_, physical_}),
            .v_pages       = Tensor(v_.data(), DType::U8, {64, kPage, kv_heads_, physical_}),
            .k_scale_pages = Tensor(ks_.data(), DType::FP16, {1, kPage, kv_heads_, physical_}),
            .v_scale_pages = Tensor(vs_.data(), DType::FP16, {1, kPage, kv_heads_, physical_}),
            .block_tables  = Tensor(d_tables_.p, DType::I32, {logical_, rows_}),
            .head_dim      = kDim,
            .num_kv_heads  = kv_heads_,
            .storage       = storage_,
            .window        = all_windows(),
        };
    }

    // The window planes of every slot, indexed by batch row b through slots[b] (table rows are
    // the identity in these calls).
    PagedKVWindowView all_windows() {
        PagedKVWindowView view = window_view(-1);
        view.slots             = Tensor(d_slots_.p, DType::I32, {rows_});
        return view;
    }

    // `with_window` false writes pages only (to leave stale window slots behind).
    PagedKVLayerView single(int row, bool with_window = true) {
        const auto all = batch();
        return {
            .k_pages       = all.k_pages,
            .v_pages       = all.v_pages,
            .k_scale_pages = all.k_scale_pages,
            .v_scale_pages = all.v_scale_pages,
            .block_table   = Tensor(static_cast<std::int32_t*>(d_tables_.p) + row * logical_,
                                    DType::I32, {logical_}),
            .head_dim      = kDim,
            .num_kv_heads  = kv_heads_,
            .storage       = storage_,
            .window        = with_window ? window_view(window_slot(row)) : PagedKVWindowView{},
        };
    }

    Snapshot snapshot() const {
        Snapshot s;
        s.k    = read<std::uint8_t>(k_, plane(key_bytes_));
        s.v    = read<std::uint8_t>(v_, plane(64));
        s.ks   = read<std::uint16_t>(ks_, plane(2) / 2);
        s.vs   = read<std::uint16_t>(vs_, plane(2) / 2);
        s.wk   = read<std::int8_t>(wk_, window(kDim));
        s.wv   = read<std::int8_t>(wv_, window(kDim));
        s.wks  = read<std::uint16_t>(wks_, window(8) / 2);
        s.wvs  = read<std::uint16_t>(wvs_, window(8) / 2);
        s.tags = read<std::uint32_t>(wtags_, window(8) / 4);
        return s;
    }

    // Rotated value of coordinate d of the stored row (role 0 = K) at `position` of `row`.
    std::array<double, kDim> stored_row(const Snapshot& s, int row, int role, int head,
                                        int position) const {
        const std::size_t r = page_row(row, head, position);
        const bool q4       = role == 0 && storage_ == KvCacheStorage::Q4KeyVq2Value;
        const int bytes     = role == 0 ? key_bytes_ : 64;
        const auto& codes   = role ? s.v : s.k;
        const double scale  = vq::f16_to_f32((role ? s.vs : s.ks)[r]);
        std::array<double, kDim> out{};
        for (int w = 0; w < vq::kWords; ++w) {
            for (int j = 0; j < 8; ++j) {
                int level = 0;
                if (q4) {
                    const int n = (codes[r * bytes + 4 * w + j / 2] >> (4 * (j & 1))) & 0xf;
                    level       = n >= 8 ? vq::q4_level(n - 8) : -vq::q4_level(7 - n);
                } else {
                    const std::uint32_t code = codes[r * bytes + 2 * w] |
                                               (static_cast<std::uint32_t>(codes[r * bytes + 2 * w + 1]) << 8);
                    const unsigned low7  = (code >> 9) & 0x7fu;
                    const unsigned signs = low7 | ((std::popcount(low7) & 1u) << 7);
                    const int magnitude  = infernix::ops::kKVCacheVq2Codebook[8 * (code & 0x1ffu) + j];
                    level                = ((signs >> j) & 1u) ? -magnitude : magnitude;
                }
                out[8 * w + j] = static_cast<double>(static_cast<float>(level) *
                                                     static_cast<float>(scale));
            }
        }
        return out;
    }

    // The slot's INT8 values when its tag matches the stored codes of `position`.
    bool slot_row(const Snapshot& s, int row, int role, int head, int position,
                  std::array<double, kDim>& out) const {
        const std::size_t r = page_row(row, head, position);
        const int bytes     = role == 0 ? key_bytes_ : 64;
        const auto& codes   = role ? s.v : s.k;
        std::vector<std::uint8_t> stored(codes.begin() + r * bytes, codes.begin() + (r + 1) * bytes);
        const std::uint16_t scale = (role ? s.vs : s.ks)[r];
        const std::size_t slot =
            (static_cast<std::size_t>(window_slot(row)) * kv_heads_ + head) * kSlots +
            vq::window_slot(position);
        if (s.tags[slot * 2 + role] != vq::window_tag(position, stored, scale)) return false;
        const auto& win    = role ? s.wv : s.wk;
        const auto& scales = role ? s.wvs : s.wks;
        for (int d = 0; d < kDim; ++d) {
            out[d] = static_cast<double>(static_cast<float>(win[slot * kDim + d]) *
                                         vq::f16_to_f32(scales[slot * 4 + d / 64]));
        }
        return true;
    }

    int kv_heads() const { return kv_heads_; }
    int window_slot(int row) const { return rows_ - 1 - row; }

private:
    template <class T>
    static std::vector<T> read(const GuardedDeviceBuffer& buffer, std::size_t count) {
        std::vector<T> out(count);
        buffer.copy_to_host(out.data(), count * sizeof(T));
        return out;
    }
    std::size_t plane(int bytes) const {
        return static_cast<std::size_t>(bytes) * kPage * kv_heads_ * physical_;
    }
    std::size_t window(int bytes) const {
        return static_cast<std::size_t>(bytes) * kSlots * kv_heads_ * rows_;
    }
    std::size_t page_row(int row, int head, int position) const {
        const int page = tables_[static_cast<std::size_t>(row) * logical_ + position / kPage];
        return (static_cast<std::size_t>(page) * kv_heads_ + head) * kPage + position % kPage;
    }
    PagedKVWindowView window_view(int row) {
        const int base  = std::max(row, 0);
        const int count = row >= 0 ? 1 : rows_;
        const auto at   = [&](GuardedDeviceBuffer& buffer, std::size_t row_bytes) {
            return static_cast<std::uint8_t*>(buffer.data()) + row_bytes * base;
        };
        const std::size_t code_row  = static_cast<std::size_t>(kDim) * kSlots * kv_heads_;
        const std::size_t scale_row = static_cast<std::size_t>(8) * kSlots * kv_heads_;
        return {
            .k_codes  = Tensor(at(wk_, code_row), DType::I8, {kDim, kSlots, kv_heads_, count}),
            .v_codes  = Tensor(at(wv_, code_row), DType::I8, {kDim, kSlots, kv_heads_, count}),
            .k_scales = Tensor(at(wks_, scale_row), DType::FP16, {4, kSlots, kv_heads_, count}),
            .v_scales = Tensor(at(wvs_, scale_row), DType::FP16, {4, kSlots, kv_heads_, count}),
            .tags     = Tensor(at(wtags_, scale_row), DType::I32, {2, kSlots, kv_heads_, count}),
        };
    }

    KvCacheStorage storage_;
    int kv_heads_, key_bytes_, logical_, rows_, physical_;
    std::vector<std::int32_t> tables_;
    DeviceBuffer d_tables_, d_slots_;
    GuardedDeviceBuffer k_, v_, ks_, vs_, wk_, wv_, wks_, wvs_, wtags_;
};

std::vector<float> random_bf16(std::size_t count, std::uint32_t seed, float amplitude) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -amplitude, amplitude);
    round_to_bf16(values);
    return values;
}

// The read rule (vq_exact_key): query position `query` reads `key` from its exact INT8-G64 row
// when the key is a sink or at most kKVWindowRecentTokens before the query.
bool exact_read(int key, int query) {
    return key < kKVWindowSinkTokens || key >= query - kKVWindowRecentTokens;
}

std::array<double, kDim> rotate64(std::array<double, kDim> x) {
    for (int h = 1; h < kDim; h <<= 1)
        for (int i = 0; i < kDim; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const double a = x[j], b = x[j + h];
                x[j]           = a + b;
                x[j + h]       = a - b;
            }
    for (double& value : x) value /= 16.0;
    return x;
}

// Appends history [0, length) of `row` in chunks through the single-sequence append.
void write_history(VqCache& cache, int row, int length, std::uint32_t seed, int chunk = 1500) {
    const int heads = cache.kv_heads();
    for (int begin = 0; begin < length; begin += chunk) {
        const int count = std::min(chunk, length - begin);
        const auto k    = random_bf16(static_cast<std::size_t>(kDim) * heads * count, seed + begin, 2.0f);
        const auto v    = random_bf16(static_cast<std::size_t>(kDim) * heads * count, seed + 7 + begin, 1.0f);
        std::vector<std::int32_t> positions(static_cast<std::size_t>(count));
        for (int t = 0; t < count; ++t) positions[t] = begin + t;
        DeviceBuffer dk = to_device_bf16(k), dv = to_device_bf16(v), dp = to_device(positions);
        ops::kv_cache_append(Tensor(dk.p, DType::BF16, {kDim, heads, count}),
                             Tensor(dv.p, DType::BF16, {kDim, heads, count}),
                             Tensor(dp.p, DType::I32, {count}), cache.single(row), nullptr);
        cuda_synchronize();
    }
}

struct Call {
    int width                    = 1;
    std::vector<int> prefix;           // per batch row: history length = first position
    std::vector<int> valid;            // per batch row valid columns (empty: all)
    std::vector<std::uint32_t> tree;   // [B][W] ancestor masks (empty: causal)
    std::uint32_t seed           = 1;
    bool cached                  = false; // causal_softmax_attention_cached (B = 1, no append)
    int oracle_columns           = 64;    // query columns checked per row (evenly spaced)
    int envelope_keys            = 0;     // visible-key envelope (0: the cache capacity); above
                                          // 32K it selects the 8-column verification CTAs
    bool prompt_pv8              = false; // the prompt kernel's 8-bit PV form
};

int run_call(const Geometry& g, KvCacheStorage storage, const Call& c, const std::string& name,
             bool stale_window = false) {
    const int batch      = static_cast<int>(c.prefix.size());
    const int width      = c.width;
    const int max_prefix = *std::max_element(c.prefix.begin(), c.prefix.end());
    const int capacity   = std::max(max_prefix + width + 70, c.envelope_keys);
    VqCache cache(storage, g.kv_heads, capacity, batch);
    for (int b = 0; b < batch; ++b) write_history(cache, b, c.prefix[b], c.seed * 131 + b);
    if (stale_window) {
        // Rewrite pages [prefix - 500, prefix - 100) of row 0 without touching its window.
        const int begin = c.prefix[0] - 500, count = 400;
        const auto k = random_bf16(static_cast<std::size_t>(kDim) * g.kv_heads * count, 99, 2.0f);
        const auto v = random_bf16(static_cast<std::size_t>(kDim) * g.kv_heads * count, 98, 1.0f);
        std::vector<std::int32_t> positions(static_cast<std::size_t>(count));
        for (int t = 0; t < count; ++t) positions[t] = begin + t;
        DeviceBuffer dk = to_device_bf16(k), dv = to_device_bf16(v), dp = to_device(positions);
        ops::kv_cache_append(Tensor(dk.p, DType::BF16, {kDim, g.kv_heads, count}),
                             Tensor(dv.p, DType::BF16, {kDim, g.kv_heads, count}),
                             Tensor(dp.p, DType::I32, {count}), cache.single(0, false), nullptr);
        cuda_synchronize();
    }
    Snapshot before = cache.snapshot();

    const std::size_t q_count  = static_cast<std::size_t>(kDim) * g.q_heads * width * batch;
    const std::size_t kv_count = static_cast<std::size_t>(kDim) * g.kv_heads * width * batch;
    const auto q = random_bf16(q_count, c.seed + 1, 0.5f);
    const auto k = random_bf16(kv_count, c.seed + 2, 2.0f);
    const auto v = random_bf16(kv_count, c.seed + 3, 1.0f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(width) * batch), rows(batch);
    for (int b = 0; b < batch; ++b) {
        rows[b] = b;
        for (int t = 0; t < width; ++t) positions[static_cast<std::size_t>(b) * width + t] = c.prefix[b] + t;
    }
    DeviceBuffer dq = to_device_bf16(q), dk = to_device_bf16(k), dv = to_device_bf16(v);
    DeviceBuffer dp = to_device(positions), drows = to_device(rows);
    DeviceBuffer dvalid = to_device(c.valid.empty() ? std::vector<std::int32_t>(1, 0) : c.valid);
    DeviceBuffer dtree  = to_device(c.tree.empty() ? std::vector<std::uint32_t>(1, 0) : c.tree);
    GuardedDeviceBuffer dout(q_count * 2);
    dout.fill(0x7f);
    const ops::AttentionHeadGeometry geometry{kDim, g.q_heads, g.kv_heads};
    ops::CausalAttentionExecutionEnvelope envelope{1, static_cast<std::uint32_t>(capacity)};
    envelope.fast_prompt_kernel = true;
    envelope.fast_prompt_pv8    = c.prompt_pv8;
    const std::size_t workspace_bytes = ops::causal_softmax_attention_workspace_capacity_bytes(
        geometry, storage, envelope, batch, width, width, test_execution());
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});
    Tensor tout(dout.data(), DType::BF16, {kDim, g.q_heads, width, batch});
    if (c.cached) {
        // The call's own keys are already in the cache (appended with the window slots).
        ops::kv_cache_append(Tensor(dk.p, DType::BF16, {kDim, g.kv_heads, width}),
                             Tensor(dv.p, DType::BF16, {kDim, g.kv_heads, width}),
                             Tensor(dp.p, DType::I32, {width}), cache.single(0), nullptr);
        cuda_synchronize();
        // A cached call reads the window as that append left it (a wide append may reuse the
        // ring slots of earlier keys).
        before = cache.snapshot();
        Tensor out3(dout.data(), DType::BF16, {kDim, g.q_heads, width});
        ops::causal_softmax_attention_cached(Tensor(dq.p, DType::BF16, {kDim, g.q_heads, width}),
                                             Tensor(dp.p, DType::I32, {width}), geometry, 0.0625f,
                                             cache.single(0), envelope, workspace, out3,
                                             test_execution());
    } else {
        ops::causal_softmax_attention(
            Tensor(dq.p, DType::BF16, {kDim, g.q_heads, width, batch}),
            Tensor(dk.p, DType::BF16, {kDim, g.kv_heads, width, batch}),
            Tensor(dv.p, DType::BF16, {kDim, g.kv_heads, width, batch}),
            Tensor(dp.p, DType::I32, {width, batch}),
            c.valid.empty() ? Tensor{} : Tensor(dvalid.p, DType::I32, {batch}),
            Tensor(drows.p, DType::I32, {batch}),
            c.tree.empty() ? Tensor{} : Tensor(dtree.p, DType::I32, {width, batch}), geometry,
            0.0625f, cache.batch(), envelope, workspace, tout, test_execution());
    }
    cuda_synchronize();
    std::vector<std::uint16_t> out(q_count);
    dout.copy_to_host(out.data(), q_count * 2);
    const Snapshot after = cache.snapshot();

    // Oracle over evenly spaced valid columns of every row.
    std::vector<double> got, ref;
    for (int b = 0; b < batch; ++b) {
        const int valid = c.valid.empty() ? width : c.valid[b];
        const int first = c.prefix[b];
        // Both representations of every visible key of this row, per kv head and role: exact
        // (where some query reads it exactly) and coded (where some query reads its codes).
        const int keys       = first + valid;
        const int last_query = first + valid - 1;
        const std::size_t count = static_cast<std::size_t>(keys) * g.kv_heads * 2;
        std::vector<std::array<double, kDim>> exact(count), coded(count);
        for (int head = 0; head < g.kv_heads; ++head) {
            for (int key = 0; key < keys; ++key) {
                for (int role = 0; role < 2; ++role) {
                    const std::size_t index =
                        (static_cast<std::size_t>(key) * g.kv_heads + head) * 2 + role;
                    // Codes as the append left them (the call's own rows included).
                    if (!exact_read(key, last_query))
                        coded[index] = cache.stored_row(after, b, role, head, key);
                    if (!exact_read(key, first)) continue;
                    auto& r = exact[index];
                    if (key >= first && !c.cached) {
                        // The call's own row: INT8-G64 of its rotated input.
                        std::array<float, kDim> y{};
                        const auto& source = role ? v : k;
                        for (int d = 0; d < kDim; ++d)
                            y[d] = source[(static_cast<std::size_t>(b) * width + (key - first)) *
                                              kDim * g.kv_heads +
                                          static_cast<std::size_t>(head) * kDim + d];
                        vq::hadamard(y);
                        const auto row8 = vq::encode_int8(y);
                        for (int d = 0; d < kDim; ++d)
                            r[d] = static_cast<double>(static_cast<float>(row8.codes[d]) *
                                                       vq::f16_to_f32(row8.scales[d / 64]));
                        continue;
                    }
                    // Older keys, and a cached call's own keys, read their window slot when its
                    // tag matches the stored codes, and the codes otherwise.
                    if (!cache.slot_row(before, b, role, head, key, r))
                        r = cache.stored_row(after, b, role, head, key);
                }
            }
        }
        const int step = std::max(1, valid / std::max(1, c.oracle_columns));
        for (int j = 0; j < valid; j += step) {
            const int qpos = first + j;
            for (int h = 0; h < g.q_heads; ++h) {
                const int kv = h / (g.q_heads / g.kv_heads);
                std::array<double, kDim> qr{};
                for (int d = 0; d < kDim; ++d)
                    qr[d] = q[((static_cast<std::size_t>(b) * width + j) * g.q_heads + h) * kDim + d];
                qr = rotate64(qr);
                std::vector<double> scores(static_cast<std::size_t>(qpos) + 1, -1e300);
                double m = -1e300;
                for (int key = 0; key <= qpos; ++key) {
                    if (!c.tree.empty() && key >= first && key - first < 32 &&
                        !((c.tree[static_cast<std::size_t>(b) * width + j] >> (key - first)) & 1u))
                        continue;
                    const auto& reps = exact_read(key, qpos) ? exact : coded;
                    const auto& kr   = reps[(static_cast<std::size_t>(key) * g.kv_heads + kv) * 2];
                    double s = 0.0;
                    for (int d = 0; d < kDim; ++d) s += qr[d] * kr[d];
                    scores[key] = s * 0.0625;
                    m           = std::max(m, scores[key]);
                }
                std::array<double, kDim> o{};
                double l = 0.0;
                for (int key = 0; key <= qpos; ++key) {
                    if (scores[key] <= -1e299) continue;
                    const double p = std::exp(scores[key] - m);
                    l += p;
                    const auto& reps = exact_read(key, qpos) ? exact : coded;
                    const auto& vr = reps[(static_cast<std::size_t>(key) * g.kv_heads + kv) * 2 + 1];
                    for (int d = 0; d < kDim; ++d) o[d] += p * vr[d];
                }
                for (double& value : o) value /= l;
                o = rotate64(o);
                for (int d = 0; d < kDim; ++d) {
                    ref.push_back(o[d]);
                    got.push_back(bf16_to_f32(
                        out[((static_cast<std::size_t>(b) * width + j) * g.q_heads + h) * kDim + d]));
                }
            }
        }
        // Columns past the valid count publish zeros.
        for (int j = valid; j < width; ++j) {
            for (std::size_t e = 0; e < static_cast<std::size_t>(g.q_heads) * kDim; ++e) {
                ref.push_back(0.0);
                got.push_back(bf16_to_f32(out[(static_cast<std::size_t>(b) * width + j) * g.q_heads * kDim + e]));
            }
        }
    }
    const std::string label = name + " " + g.name + " " +
                              (storage == KvCacheStorage::Vq2 ? "vq2" : "k4v2");
    int failures = verify_reduction(label, got, ref, kCriterion);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace exceeded its query\n";
        ++failures;
    }

    // Every call leaves its sinks and final kKVWindowRingTokens columns in the window.
    if (!c.cached) {
        int window_failures = 0;
        for (int b = 0; b < batch; ++b) {
            const int valid = c.valid.empty() ? width : c.valid[b];
            const int first = c.prefix[b];
            const int last  = first + valid - 1;
            for (int key = first; key <= last; ++key) {
                if (key >= kKVWindowSinkTokens && key <= last - kKVWindowRingTokens) continue;
                for (int head = 0; head < g.kv_heads; ++head)
                    for (int role = 0; role < 2; ++role) {
                        std::array<double, kDim> slot{};
                        if (!cache.slot_row(after, b, role, head, key, slot)) ++window_failures;
                    }
            }
        }
        if (window_failures != 0) {
            std::cerr << label << ": " << window_failures << " window slots not written\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (cuda_unavailable()) {
        std::cout << "SKIP no CUDA device\n";
        return 77;
    }
    const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
    int failures     = 0;
    for (const auto storage : {KvCacheStorage::Vq2, KvCacheStorage::Q4KeyVq2Value}) {
        for (const auto& g : kGeometries) {
            // Grouped decode: sinks only, inside the window, beyond the window, page crossings.
            for (const int w : {1, 2, 5, 8})
                for (const int prefix : {0, 40, 700, 1500, 3200})
                    failures += run_call(g, storage, Call{.width = w, .prefix = {prefix}, .seed = 11u + w},
                                         "decode W=" + std::to_string(w) + " prefix=" + std::to_string(prefix));
            // Several rows with masked columns.
            failures += run_call(g, storage,
                                 Call{.width = 4, .prefix = {300, 2100, 60}, .valid = {4, 2, 3}, .seed = 21},
                                 "batched decode W=4 B=3 masked");
            // Parallel verification widths.
            for (const int w : {9, 33, 64})
                failures += run_call(g, storage, Call{.width = w, .prefix = {1300}, .seed = 31u + w},
                                     "verify W=" + std::to_string(w));
            failures += run_call(g, storage,
                                 Call{.width = 40, .prefix = {900, 2000}, .valid = {40, 23}, .seed = 37},
                                 "verify W=40 B=2 masked");
            failures += run_call(g, storage,
                                 Call{.width = 33, .prefix = {1300, 700}, .valid = {33, 20}, .seed = 38,
                                      .envelope_keys = 40000},
                                 "verify W=33 B=2 masked 8-column");
            // Verification trees (ancestor masks), grouped and parallel.
            for (const int w : {8, 16}) {
                Call c{.width = w, .prefix = {1100, 64}, .valid = {w, w - 3}, .seed = 41u + w};
                std::mt19937 rng(w);
                for (int b = 0; b < 2; ++b) {
                    std::vector<int> parent(w, -1);
                    for (int j = 1; j < w; ++j) parent[j] = static_cast<int>(rng() % j);
                    for (int j = 0; j < w; ++j) {
                        std::uint32_t mask = 0;
                        for (int a = j; a >= 0; a = parent[a]) mask |= 1u << a;
                        c.tree.push_back(mask);
                    }
                }
                failures += run_call(g, storage, c, "tree W=" + std::to_string(w));
                c.envelope_keys = 40000;
                failures += run_call(g, storage, c, "tree W=" + std::to_string(w) + " 8-column");
            }
            // Stale window slots must fall back to the stored codes.
            failures += run_call(g, storage, Call{.width = 3, .prefix = {1500}, .seed = 51},
                                 "stale window W=3", true);
            failures += run_call(g, storage, Call{.width = 300, .prefix = {1500}, .seed = 52, .oracle_columns = 24},
                                 "stale window prompt W=300", true);
            // Cached entry (the MTP final-token form) and a cached prompt.
            failures += run_call(g, storage, Call{.width = 1, .prefix = {2500}, .seed = 61, .cached = true},
                                 "cached W=1");
            if (!quick) {
                failures += run_call(g, storage,
                                     Call{.width = 400, .prefix = {1800}, .seed = 62, .cached = true, .oracle_columns = 16},
                                     "cached prompt W=400");
                failures += run_call(g, storage,
                                     Call{.width = 400, .prefix = {5200}, .seed = 63, .cached = true, .oracle_columns = 16},
                                     "cached prompt W=400 prefix=5200");
                failures += run_call(g, storage, Call{.width = 300, .prefix = {5000}, .seed = 53, .oracle_columns = 24},
                                     "stale window prompt W=300 prefix=5000", true);
                // Wide prompt calls: fresh, after a cached prefix, with key splits, wider than the
                // ring (the window commits only the final kKVWindowRingTokens columns).
                // Calls seeing 1536 keys take 64-key tiles with one INT8 tile; fewer take 32.
                for (const auto& [w, prefix] :
                     std::vector<std::pair<int, int>>{{257, 0}, {300, 1000}, {300, 3000}, {1100, 900},
                                                      {2048, 40}, {300, 5000}, {1100, 4500}})
                    failures += run_call(g, storage,
                                         Call{.width = w, .prefix = {prefix}, .seed = 71u + w, .oracle_columns = 24},
                                         "prompt W=" + std::to_string(w) + " prefix=" + std::to_string(prefix));
                // The prompt kernel's 8-bit PV form: its V rows stay the stored INT8 codes and
                // groups, so the case criterion is the format's own. Both tile widths (a call
                // under 1536 visible keys takes 32-key tiles, above it 64) and the exact-window
                // band are covered, with and without key splits and through the cached entry.
                for (const auto& [w, prefix, stale] :
                     std::vector<std::array<int, 3>>{{300, 0, 0}, {1100, 900, 0}, {300, 3000, 0},
                                                     {2048, 40, 0}, {300, 5000, 0}, {300, 1500, 1}}) {
                    failures += run_call(
                        g, storage,
                        Call{.width = w, .prefix = {prefix}, .seed = 91u + w, .oracle_columns = 24,
                             .prompt_pv8 = true},
                        "prompt pv8 W=" + std::to_string(w) + " prefix=" + std::to_string(prefix),
                        stale != 0);
                }
                failures += run_call(
                    g, storage,
                    Call{.width = 400, .prefix = {1800}, .seed = 92, .cached = true,
                         .oracle_columns = 16, .prompt_pv8 = true},
                    "cached prompt pv8 W=400");
            }
        }
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " vq attention (" << failures
              << " failures)\n";
    return failures == 0 ? 0 : 1;
}
