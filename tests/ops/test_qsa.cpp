// QSA (include/ninfer/ops/qsa.h; docs/maintainer/qwen3_8-flash-next-design.md §8.10) against
// independent oracles (docs/maintainer/op-development.md §6), at the real Qwen3.8-Flash-Next
// geometry (24 query heads, 2 KV heads of 256, a 4 x 128 indexer, 64 rotary dimensions, budget
// 2048 in blocks of 4):
//
//   - qsa_index_query: FP64 RMSNorm with the unit-offset weight, then the half-split RoPE of the
//     first rotary_dim dimensions at each column's three-axis RoPE position, pair i on axis i % 3
//     (distinct random axes).
//   - qsa_pool_keys: the pooled key of every block a call completes, read back from the paged
//     plane, against FP64 RoPE(norm(bf16(mean))) at the RoPE position of the block's first token
//     (the BF16 mean is the contract's boundary). Every sequence holds an image span with real
//     M-RoPE positions (text: three equal axes; image: (cur, cur + y, cur + x); text after it
//     shifted by max(h, w)), so blocks rotate by distinct axes, also when their first token
//     precedes the call (block starts). The slots of blocks no call completed stay untouched;
//     pooled keys are bitwise the same however a sequence is split into calls.
//   - The raw-key tails, exactly: after a call or commit ending at position `last`, slot q % R
//     holds the raw key of every position q <= last of the latest block with q % R < R - 1, also
//     when `last` completes that block. Rewriting position `last` of a completed block (the MTP
//     drafter's prepend and draft step 0) re-pools it from the tails, so its pooled key must come
//     back unchanged whether the block completed inside one call, across calls or in a commit.
//   - qsa_commit_tails: the tails of every layer after committing n of W verified columns.
//   - qsa_attention: block selection by exact scores (the top budget / R blocks, lower ids on
//     equal scores; every token while the visible context is at most budget + R - 1) and FP64
//     softmax attention over the K/V decoded from their stored codes, for each implemented KV
//     storage (bf16; int8-g64 with Hadamard-rotated keys), at decode, verification and prefill
//     widths, across the dense/selected boundary and page boundaries.
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/attention_geometry.h"
#include "ninfer/ops/qsa.h"
#include "ops/host_parallel.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace ops = ninfer::ops;
namespace t   = ninfer::test;
using ninfer::DeviceBuffer;
using ninfer::DType;
using ninfer::KvCacheStorage;
using ninfer::Tensor;

constexpr int kDi               = 128; // index head dimension
constexpr int kR                = 4;   // tokens per pooled block
constexpr int kPage             = 64;  // positions per KV page
constexpr int kD                = 256; // attention head dimension
constexpr int kHeads            = 24;
constexpr int kKvHeads          = 2;
constexpr int kIndexHeads       = 4;
constexpr int kRotary           = 64;
constexpr int kBudget           = 2048;
constexpr int kTopBlocks        = kBudget / kR;
constexpr int kSlotWidth        = kDi / kR; // pooled-plane lanes per token slot
constexpr std::uint16_t kCanary = 0x7fc1;

const ops::QsaGeometry kGeometry{.heads          = kHeads,
                                 .kv_heads       = kKvHeads,
                                 .head_dim       = kD,
                                 .index_heads    = kIndexHeads,
                                 .index_head_dim = kDi,
                                 .rotary_dim     = kRotary,
                                 .budget         = kBudget,
                                 .ratio          = kR,
                                 .theta          = 1.0e7F,
                                 .eps            = 1.0e-6F};

// Index queries and pooled keys are normalized, rotated BF16 vectors. The Op rounds the normalized
// vector to BF16 before rotating it (the reference RMSNorm's output) and rounds the output: two
// roundings of at most 2^-9 relative each (relative RMS about 2^-9 * sqrt(2/3) = 1.6e-3), plus FP32
// rotation angles (positions below 2^13 here, at most ~5e-4 rad). A rotated element errs by at most
// 2^-9 * (|a| + |b| + |out|) <= 3 * 2^-9 = 5.9e-3 of the row's largest magnitude, plus the angle.
constexpr t::ReductionCriterion kKeyCriterion{
    /*relative_l2*/ 4.0e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 8.0e-3,
};

// QSA attention evaluates FP32 scores, an online softmax and FP32 sums over K/V decoded exactly
// from their stored codes (Q is not quantized for either storage), then rounds the output to
// BF16, so both storages share the Softmax Attention suite's bf16 decode criterion.
constexpr t::ReductionCriterion kAttentionCriterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// ----------------------------------------------------------------------------------- values

double from_bf16(std::uint16_t h) { return t::bf16_to_f32(h); }

// The nearest BF16 of a double, ties to even (no intermediate FP32 rounding).
std::uint16_t to_bf16(double v) {
    if (v == 0.0 || !std::isfinite(v)) { return t::f32_to_bf16(static_cast<float>(v)); }
    int exponent          = 0;
    const double mantissa = std::frexp(v, &exponent); // |mantissa| in [0.5, 1)
    const double rounded  = std::nearbyint(std::ldexp(mantissa, 8));
    return t::f32_to_bf16(static_cast<float>(std::ldexp(rounded, exponent - 8)));
}

double from_fp16(std::uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return static_cast<double>(__half2float(h));
}

std::uint16_t to_fp16(float v) {
    const __half h = __float2half_rn(v);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

std::vector<std::uint16_t> random_bf16(std::mt19937& rng, std::size_t n, double scale) {
    std::normal_distribution<double> d(0.0, scale);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) { x = to_bf16(d(rng)); }
    return v;
}

// BF16 multiples of 2^-4 in [-1, 1]. A block score over them is a sum of products that are
// multiples of 2^-8 with |dot| <= 128 per index head and <= 512 in all, so the Op's FP32 sums equal
// the oracle's exactly; its final FP32 scaling keeps their order (distinct scores differ by at
// least 2^-17 relative) and their ties. Selection is then decided exactly, including the
// lower-id rule on equal scores, with no near-tie guard.
std::vector<std::uint16_t> grid_bf16(std::mt19937& rng, std::size_t n) {
    std::uniform_int_distribution<int> step(-16, 16);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) { x = to_bf16(step(rng) / 16.0); }
    return v;
}

template <class T>
DeviceBuffer upload(const std::vector<T>& v) {
    return t::to_device(v);
}

template <class T>
std::vector<T> download(const DeviceBuffer& d, std::size_t n) {
    return t::from_device<T>(d, n);
}

void synchronize(const char* what) { t::cuda_check(cudaDeviceSynchronize(), what); }

// --------------------------------------------------------------------------- FP64 key oracle

// RMSNorm with the unit-offset weight.
std::vector<double> normalize(const std::vector<double>& x,
                              const std::vector<std::uint16_t>& weight) {
    double squares = 0.0;
    for (const double v : x) { squares += v * v; }
    const double inv = 1.0 / std::sqrt(squares / kDi + static_cast<double>(kGeometry.eps));
    std::vector<double> out(kDi);
    for (int d = 0; d < kDi; ++d) { out[d] = x[d] * inv * (1.0 + from_bf16(weight[d])); }
    return out;
}

// A token's RoPE position: three axes (temporal, height, width).
using Rope = std::array<int, 3>;

// The half-split rotation of the first kRotary dimensions at RoPE position `rope`, pair i by
// axis i % 3 (interleaved M-RoPE).
void rotate(std::vector<double>& x, const Rope& rope) {
    constexpr int half = kRotary / 2;
    for (int i = 0; i < half; ++i) {
        const double inv   = std::pow(static_cast<double>(kGeometry.theta), -2.0 * i / kRotary);
        const double angle = rope[i % 3] * inv;
        const double c = std::cos(angle), s = std::sin(angle);
        const double a = x[i], b = x[i + half];
        x[i]        = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

// An image span inside a sequence: grid rows x cols tokens from position `start` (rows = 0: none).
struct Image {
    int start = 0, rows = 0, cols = 0;
};

// RoPE positions of `n` tokens with one image, as the frontend assigns them: text tokens before the
// image take their index on every axis, image token (y, x) takes (cur, cur + y, cur + x) with cur
// its start, and text after it continues from cur + max(rows, cols).
std::vector<Rope> rope_map(int n, const Image& image) {
    std::vector<Rope> out(n);
    const int end = image.start + image.rows * image.cols;
    for (int q = 0; q < n; ++q) {
        if (image.rows == 0 || q < image.start) {
            out[q] = {q, q, q};
        } else if (q < end) {
            const int k = q - image.start;
            out[q]      = {image.start, image.start + k / image.cols, image.start + k % image.cols};
        } else {
            const int v = image.start + std::max(image.rows, image.cols) + (q - end);
            out[q]      = {v, v, v};
        }
    }
    return out;
}

// Pooled key of block b of a raw-key sequence ([positions][kDi] BF16) with RoPE positions `rope`.
std::vector<double> pooled_oracle(const std::vector<std::uint16_t>& raw,
                                  const std::vector<std::uint16_t>& weight,
                                  const std::vector<Rope>& rope, int block) {
    std::vector<double> mean(kDi);
    for (int d = 0; d < kDi; ++d) {
        double sum = 0.0;
        for (int j = 0; j < kR; ++j) {
            sum += from_bf16(raw[static_cast<std::size_t>(kR * block + j) * kDi + d]);
        }
        mean[d] = from_bf16(to_bf16(sum / kR));
    }
    auto out = normalize(mean, weight);
    rotate(out, rope[kR * block]);
    return out;
}

// ------------------------------------------------------------------------------ index query

void test_index_query(int columns, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const auto q = random_bf16(rng, static_cast<std::size_t>(kDi) * kIndexHeads * columns, 2.0);
    const auto weight = random_bf16(rng, kDi, 0.3);
    // Distinct axes per column (axis a of column c at word a * columns + c).
    std::uniform_int_distribution<int> position(0, 8191);
    std::vector<std::int32_t> rope(3 * static_cast<std::size_t>(columns));
    for (auto& p : rope) { p = position(rng); }
    DeviceBuffer dq = upload(q), dw = upload(weight), dp = upload(rope);
    Tensor tq(dq.p, DType::BF16, {kDi, kIndexHeads, columns});
    ops::qsa_index_query(tq, Tensor(dw.p, DType::BF16, {kDi}), Tensor(dp.p, DType::I32, {columns, 3}),
                         kGeometry, nullptr);
    synchronize("index query");
    const auto got = t::from_device_bf16(dq, q.size());
    std::vector<double> ref;
    ref.reserve(q.size());
    for (int c = 0; c < columns; ++c) {
        for (int h = 0; h < kIndexHeads; ++h) {
            std::vector<double> x(kDi);
            for (int d = 0; d < kDi; ++d) {
                x[d] = from_bf16(q[(static_cast<std::size_t>(c) * kIndexHeads + h) * kDi + d]);
            }
            auto out = normalize(x, weight);
            rotate(out, {rope[c], rope[columns + c], rope[2 * static_cast<std::size_t>(columns) + c]});
            ref.insert(ref.end(), out.begin(), out.end());
        }
    }
    g_failures += t::verify_reduction("qsa_index_query T=" + std::to_string(columns), got, ref,
                                      kKeyCriterion);
}

// ---------------------------------------------------------------------- pooled keys and tails

// A sequence of calls over `rows` sequences of two layers. Row r's positions before first[r] are
// already in the state (their latest block's raw keys preset in the tails). A step is a call of
// `width` columns per row at each row's frontier: a prefill/decode call (tails advance) or a
// verification call (tails unchanged) followed by a commit of commit[r] columns.
struct Step {
    int width   = 0;
    bool verify = false;
    std::vector<int> commit;
};

struct PoolCase {
    const char* name;
    std::vector<int> first;
    std::vector<Step> steps;
    std::vector<Image> images; // one per row
};

constexpr int kLayers = 2;

class PoolHarness {
public:
    PoolHarness(const PoolCase& c, std::uint32_t seed)
        : case_(c), rows_(static_cast<int>(c.first.size())) {
        std::mt19937 rng(seed);
        int extent = 0;
        for (int r = 0; r < rows_; ++r) {
            int end = c.first[r], hi = c.first[r];
            for (const auto& step : c.steps) {
                hi = std::max(hi, end + step.width);
                end += step.verify ? step.commit.at(r) : step.width;
            }
            extent = std::max(extent, hi);
        }
        positions_     = (extent + kPage - 1) / kPage * kPage;
        pages_per_row_ = positions_ / kPage;
        // Rows 0..rows-1 run the steps; rows rows..2*rows-1 pool the same sequences in one call.
        table_rows_     = 2 * rows_;
        slots_          = table_rows_ + 1;
        const int pages = table_rows_ * pages_per_row_;
        std::vector<std::int32_t> page_ids(pages);
        std::iota(page_ids.begin(), page_ids.end(), 0);
        std::shuffle(page_ids.begin(), page_ids.end(), rng);
        tables_host_ = page_ids; // row-major [table row][logical page]
        tables_      = upload(tables_host_);
        for (int l = 0; l < kLayers; ++l) {
            weight_[l]  = random_bf16(rng, kDi, 0.3);
            weights_[l] = upload(weight_[l]);
            for (int r = 0; r < rows_; ++r) {
                raw_[l].push_back(
                    random_bf16(rng, static_cast<std::size_t>(positions_) * kDi, 1.0));
            }
            if (l == 0) {
                for (int r = 0; r < rows_; ++r) { rope_.push_back(rope_map(positions_, c.images.at(r))); }
            }
            pooled_[l] = upload(std::vector<std::uint16_t>(
                static_cast<std::size_t>(kSlotWidth) * kPage * pages, kCanary));
        }
        // Tails start random, except each row's latest block before its first position.
        tails_host_ =
            random_bf16(rng, static_cast<std::size_t>(kDi) * (kR - 1) * slots_ * kLayers, 1.0);
        for (int r = 0; r < rows_; ++r) {
            for (const int row : {r, rows_ + r}) { preset_tails(row, r, c.first[r]); }
        }
        tails_    = upload(tails_host_);
        frontier_ = c.first;
        written_  = c.first; // one past the highest position any call wrote
    }

    void run() {
        for (std::size_t s = 0; s < case_.steps.size(); ++s) { step(s, case_.steps[s]); }
        compare_single_call();
    }

private:
    int slot_of(int table_row) const { return slots_ - 1 - table_row; } // not the identity

    std::size_t tail_index(int layer, int slot, int j, int d) const {
        return ((static_cast<std::size_t>(layer) * slots_ + slot) * (kR - 1) + j) * kDi + d;
    }

    void preset_tails(int table_row, int r, int first) {
        if (first == 0) { return; }
        const int last = first - 1, base = last - last % kR;
        for (int l = 0; l < kLayers; ++l) {
            for (int q = base; q <= std::min(last, base + kR - 2); ++q) {
                for (int d = 0; d < kDi; ++d) {
                    tails_host_[tail_index(l, slot_of(table_row), q % kR, d)] =
                        raw_[l][r][static_cast<std::size_t>(q) * kDi + d];
                }
            }
        }
    }

    std::uint16_t* tails_layer(int l) const {
        return static_cast<std::uint16_t*>(tails_.p) +
               static_cast<std::size_t>(l) * kDi * (kR - 1) * slots_;
    }

    ops::QsaKVLayer layer_view(int l) const {
        ops::QsaKVLayer layer;
        layer.pooled_pages =
            Tensor(pooled_[l].p, DType::BF16, {kSlotWidth, kPage, 1, table_rows_ * pages_per_row_});
        return layer;
    }

    // Pools `width` positions of each listed table row starting at starts[i], from host raw keys
    // [kDi, width, rows, kLayers] uploaded here; returns the device copy for a commit.
    DeviceBuffer pool(const std::vector<int>& table_rows, const std::vector<int>& sources,
                      const std::vector<int>& starts, int width, bool update_tails,
                      DeviceBuffer* positions_out = nullptr) {
        const int batch = static_cast<int>(table_rows.size());
        std::vector<std::uint16_t> keys(static_cast<std::size_t>(kDi) * width * batch * kLayers);
        const std::size_t columns = static_cast<std::size_t>(width) * batch;
        std::vector<std::int32_t> positions(columns), rows(batch), slots(batch), rope(3 * columns),
            block_rope(3 * static_cast<std::size_t>(batch));
        for (int b = 0; b < batch; ++b) {
            rows[b]  = table_rows[b];
            slots[b] = slot_of(table_rows[b]);
            const Rope& block_start = rope_[sources[b]][starts[b] / kR * kR];
            for (int a = 0; a < 3; ++a) { block_rope[static_cast<std::size_t>(a) * batch + b] = block_start[a]; }
            for (int j = 0; j < width; ++j) {
                const std::size_t column = static_cast<std::size_t>(b) * width + j;
                positions[column]        = starts[b] + j;
                for (int a = 0; a < 3; ++a) { rope[a * columns + column] = rope_[sources[b]][starts[b] + j][a]; }
                for (int l = 0; l < kLayers; ++l) {
                    std::memcpy(
                        &keys[((static_cast<std::size_t>(l) * batch + b) * width + j) * kDi],
                        &raw_[l][sources[b]][static_cast<std::size_t>(starts[b] + j) * kDi],
                        kDi * 2);
                }
            }
        }
        DeviceBuffer dkeys = upload(keys), dpos = upload(positions), drows = upload(rows),
                     dslots = upload(slots), drope = upload(rope), dblock = upload(block_rope);
        const ops::QsaBatch qsa_batch{
            .block_tables     = Tensor(tables_.p, DType::I32, {pages_per_row_, table_rows_}),
            .table_rows       = Tensor(drows.p, DType::I32, {batch}),
            .positions        = Tensor(dpos.p, DType::I32, {batch * width}),
            .tail_slots       = Tensor(dslots.p, DType::I32, {batch}),
            .rope_positions   = Tensor(drope.p, DType::I32, {batch * width, 3}),
            .block_start_rope = Tensor(dblock.p, DType::I32, {batch, 3}),
            .batch            = batch,
            .width            = width,
            .update_tails     = update_tails};
        for (int l = 0; l < kLayers; ++l) {
            Tensor raw(static_cast<std::uint16_t*>(dkeys.p) +
                           static_cast<std::size_t>(l) * kDi * width * batch,
                       DType::BF16, {kDi, batch * width});
            Tensor tails(tails_layer(l), DType::BF16, {kDi, kR - 1, slots_});
            ops::qsa_pool_keys(raw, Tensor(weights_[l].p, DType::BF16, {kDi}), tails, layer_view(l),
                               qsa_batch, kGeometry, nullptr);
        }
        synchronize("pool");
        if (positions_out != nullptr) {
            *positions_out = std::move(dpos);
            commit_slots_  = std::move(dslots);
        }
        return dkeys;
    }

    std::vector<std::uint16_t> plane(int l) const {
        return download<std::uint16_t>(pooled_[l], static_cast<std::size_t>(kSlotWidth) * kPage *
                                                       table_rows_ * pages_per_row_);
    }

    // The 128 stored values of block b of table row `row`, from the paged plane.
    std::vector<std::uint16_t> block_key(const std::vector<std::uint16_t>& plane, int row,
                                         int block) const {
        std::vector<std::uint16_t> out(kDi);
        for (int d = 0; d < kDi; ++d) {
            const int token = kR * block + d / kSlotWidth;
            const int page =
                tables_host_[static_cast<std::size_t>(row) * pages_per_row_ + token / kPage];
            out[d] = plane[(static_cast<std::size_t>(page) * kPage + token % kPage) * kSlotWidth +
                           d % kSlotWidth];
        }
        return out;
    }

    void step(std::size_t index, const Step& s) {
        std::vector<int> rows(rows_), starts = frontier_;
        std::iota(rows.begin(), rows.end(), 0);
        DeviceBuffer positions;
        DeviceBuffer keys =
            pool(rows, rows, starts, s.width, !s.verify, s.verify ? &positions : nullptr);
        if (s.verify) {
            std::vector<std::int32_t> commit(s.commit.begin(), s.commit.end());
            DeviceBuffer dcommit = upload(commit);
            Tensor tails(tails_.p, DType::BF16, {kDi, kR - 1, slots_, kLayers});
            ops::qsa_commit_tails(Tensor(keys.p, DType::BF16, {kDi, s.width, rows_, kLayers}),
                                  Tensor(positions.p, DType::I32, {s.width, rows_}),
                                  Tensor(dcommit.p, DType::I32, {rows_}), tails,
                                  Tensor(commit_slots_.p, DType::I32, {rows_}), kGeometry, nullptr);
        }
        synchronize("pool step");
        for (int r = 0; r < rows_; ++r) {
            written_[r] = std::max(written_[r], starts[r] + s.width);
            frontier_[r] += s.verify ? s.commit[r] : s.width;
        }
        const std::string tag = std::string(case_.name) + " step " + std::to_string(index) +
                                (s.verify ? " verify W=" : " call W=") + std::to_string(s.width);
        check_tails(tag);
        check_planes(tag);
        for (int r = 0; r < rows_; ++r) {
            const int f = frontier_[r];
            if (f > case_.first[r] && f % kR == 0 && f != starts[r]) {
                // The block ending at f - 1 completed in this step; `inside` when the step wrote at
                // least two of its positions (the tails needed for a rewrite came from this step).
                rewrite(tag, r, starts[r] <= f - 2);
            }
        }
    }

    void check_tails(const std::string& tag) {
        const auto tails = download<std::uint16_t>(tails_, tails_host_.size());
        for (int r = 0; r < rows_; ++r) {
            if (frontier_[r] == 0) { continue; }
            const int last = frontier_[r] - 1, base = last - last % kR;
            bool ok = true;
            for (int l = 0; l < kLayers; ++l) {
                for (int q = base; q <= std::min(last, base + kR - 2); ++q) {
                    ok &= std::memcmp(&tails[tail_index(l, slot_of(r), q % kR, 0)],
                                      &raw_[l][r][static_cast<std::size_t>(q) * kDi], kDi * 2) == 0;
                }
            }
            expect(ok, tag + " row " + std::to_string(r) +
                           ": tails hold the raw keys of positions " + std::to_string(base) + ".." +
                           std::to_string(std::min(last, base + kR - 2)));
        }
    }

    // Every block a call completed against the oracle; every other slot of the row untouched.
    void check_planes(const std::string& tag) {
        for (int l = 0; l < kLayers; ++l) {
            const auto stored = plane(l);
            for (int r = 0; r < rows_; ++r) {
                std::vector<double> got, ref;
                bool untouched = true;
                for (int b = 0; b < positions_ / kR; ++b) {
                    const auto key = block_key(stored, r, b);
                    if (kR * b + kR - 1 >= case_.first[r] && kR * b + kR - 1 < written_[r]) {
                        for (const auto v : key) { got.push_back(from_bf16(v)); }
                        const auto oracle = pooled_oracle(raw_[l][r], weight_[l], rope_[r], b);
                        ref.insert(ref.end(), oracle.begin(), oracle.end());
                    } else {
                        untouched &= std::all_of(key.begin(), key.end(),
                                                 [](std::uint16_t v) { return v == kCanary; });
                    }
                }
                const std::string label =
                    tag + " layer " + std::to_string(l) + " row " + std::to_string(r);
                if (!got.empty()) {
                    g_failures +=
                        t::verify_reduction(label + " pooled keys", got, ref, kKeyCriterion);
                }
                expect(untouched, label + ": slots of blocks no call completed are untouched");
            }
        }
    }

    // Rewrites the last position of row r's just-completed block alone, as the drafter's draft step
    // 0 (tails unchanged) and its prepend (tails advance) do: the pooled key and the tails must not
    // change.
    void rewrite(const std::string& tag, int r, bool inside) {
        const int position = frontier_[r] - 1, block = position / kR;
        std::vector<std::vector<std::uint16_t>> before(kLayers);
        for (int l = 0; l < kLayers; ++l) { before[l] = block_key(plane(l), r, block); }
        const auto tails_before = download<std::uint16_t>(tails_, tails_host_.size());
        for (const bool update : {false, true}) {
            (void)pool({r}, {r}, {position}, 1, update);
            synchronize("rewrite");
            bool same = true;
            for (int l = 0; l < kLayers; ++l) {
                same &= block_key(plane(l), r, block) == before[l];
            }
            const std::string label =
                tag + " row " + std::to_string(r) + ": rewriting position " +
                std::to_string(position) + (update ? " (prepend)" : " (draft step 0)") +
                (inside ? ", block completed inside the step," : ", block completed across steps,");
            expect(same, label + " re-pools the same key from the tails");
            expect(download<std::uint16_t>(tails_, tails_host_.size()) == tails_before,
                   label + " leaves the tails");
        }
    }

    // The steps' pooled keys equal one call over the same positions, bit for bit.
    void compare_single_call() {
        std::vector<int> rows(rows_), sources(rows_);
        for (int r = 0; r < rows_; ++r) {
            rows[r]    = rows_ + r;
            sources[r] = r;
            (void)pool({rows[r]}, {r}, {case_.first[r]}, written_[r] - case_.first[r], true);
        }
        synchronize("single call");
        for (int l = 0; l < kLayers; ++l) {
            const auto stored = plane(l);
            for (int r = 0; r < rows_; ++r) {
                bool same = true;
                for (int b = 0; b < positions_ / kR; ++b) {
                    if (kR * b + kR - 1 >= case_.first[r] && kR * b + kR - 1 < written_[r]) {
                        same &= block_key(stored, r, b) == block_key(stored, rows_ + r, b);
                    }
                }
                expect(same, std::string(case_.name) + " layer " + std::to_string(l) + " row " +
                                 std::to_string(r) +
                                 ": pooled keys equal one call over the same positions");
            }
        }
    }

    const PoolCase& case_;
    int rows_ = 0, positions_ = 0, pages_per_row_ = 0, table_rows_ = 0, slots_ = 0;
    std::vector<std::int32_t> tables_host_;
    DeviceBuffer tables_, tails_, commit_slots_;
    std::vector<std::uint16_t> tails_host_;
    std::vector<std::uint16_t> weight_[kLayers];
    DeviceBuffer weights_[kLayers], pooled_[kLayers];
    std::vector<std::vector<std::uint16_t>> raw_[kLayers]; // [row][position * kDi + d]
    std::vector<std::vector<Rope>> rope_;                  // [row][position]
    std::vector<int> frontier_, written_;
};

void test_pool(std::uint32_t seed) {
    // Row 1 starts after two positions already in the state, so its blocks complete at other
    // steps. Steps place block starts before the call (start = 1, 2, 3 mod 4), end calls and
    // commits at block ends with the block completed inside the step and across steps, commit
    // 0..W columns, and cross page boundaries.
    const PoolCase mixed{"pool",
                         {0, 2},
                         {{7, false, {}},
                          {5, false, {}},
                          {1, false, {}},
                          {17, false, {}},
                          {2, false, {}},
                          {5, true, {4, 2}},
                          {8, true, {3, 0}},
                          {4, true, {1, 4}},
                          {64, false, {}},
                          {3, false, {}},
                          {5, true, {5, 1}},
                          {130, false, {}},
                          {2, false, {}}},
                         {{9, 6, 10}, {41, 5, 8}}};
    PoolHarness(mixed, seed).run();
    // One sequence, the MTP drafter's pattern: prompt chunks, then verification commits of one to
    // five cells (draft step 0 rewrites the last committed cell).
    const PoolCase drafter{"drafter",
                           {0},
                           {{512, false, {}},
                            {6, true, {4}},
                            {6, true, {1}},
                            {6, true, {3}},
                            {6, true, {5}},
                            {6, true, {2}},
                            {6, true, {5}},
                            {1, false, {}},
                            {6, true, {6}}},
                           {{100, 16, 16}}};
    PoolHarness(drafter, seed + 1).run();
    // Vision (VT2): a 1,024-column prefill call through 32 x 24 image grids that begin inside it
    // (row 1 at a position that is not a multiple of 4), then decode W = 1 and verification
    // W = 5 / 8 at B = 2, so blocks whose first token precedes a call rotate by its block start.
    const PoolCase image{"image",
                         {0, 3},
                         {{1024, false, {}},
                          {1, false, {}},
                          {2, false, {}},
                          {5, true, {3, 5}},
                          {8, true, {8, 2}},
                          {1, false, {}},
                          {8, true, {1, 7}}},
                         {{200, 32, 24}, {777, 24, 32}}};
    PoolHarness(image, seed + 2).run();
}

// --------------------------------------------------------------------------------- attention

struct AttentionCase {
    const char* name;
    int width;
    std::vector<int> starts; // per row: the position of its first column
};

// Element (d, head, token) of a PageMajor plane [leading, 64, heads, pages].
std::size_t plane_index(int leading, int page, int head, int token, int d) {
    return ((static_cast<std::size_t>(page) * kKvHeads + head) * kPage + token % kPage) * leading +
           d;
}

// The normalized Sylvester transform of one 256-vector (its own inverse).
void hadamard(std::vector<double>& x) {
    for (int span = 1; span < kD; span <<= 1) {
        for (int base = 0; base < kD; base += 2 * span) {
            for (int i = base; i < base + span; ++i) {
                const double a = x[i], b = x[i + span];
                x[i]        = a + b;
                x[i + span] = a - b;
            }
        }
    }
    for (auto& v : x) { v /= 16.0; }
}

void test_attention(KvCacheStorage storage, const AttentionCase& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const bool int8 = storage == KvCacheStorage::Int8Group64;
    const int rows = static_cast<int>(c.starts.size()), width = c.width, columns = rows * width;
    int context = 0;
    for (const int s : c.starts) { context = std::max(context, s + width); }
    const int pages_per_row = (context + kPage - 1) / kPage, pages = rows * pages_per_row;
    const int max_context = pages_per_row * kPage;
    std::vector<std::int32_t> tables(pages);
    std::iota(tables.begin(), tables.end(), 0);
    std::shuffle(tables.begin(), tables.end(), rng);

    // Stored K/V: BF16 keys and FP16 values, or INT8 codes with one FP16 scale per 64 elements
    // (keys in the normalized Hadamard domain).
    const std::size_t elements = static_cast<std::size_t>(kD) * kPage * kKvHeads * pages;
    const std::size_t groups   = static_cast<std::size_t>(kD / 64) * kPage * kKvHeads * pages;
    std::vector<std::uint16_t> k16, v16, k_scale, v_scale;
    std::vector<std::int8_t> k8, v8;
    if (int8) {
        std::uniform_int_distribution<int> code(-127, 127);
        std::uniform_real_distribution<float> scale(0.004F, 0.012F);
        k8.resize(elements);
        v8.resize(elements);
        for (auto& x : k8) { x = static_cast<std::int8_t>(code(rng)); }
        for (auto& x : v8) { x = static_cast<std::int8_t>(code(rng)); }
        k_scale.resize(groups);
        v_scale.resize(groups);
        for (auto& x : k_scale) { x = to_fp16(scale(rng)); }
        for (auto& x : v_scale) { x = to_fp16(scale(rng)); }
    } else {
        k16 = random_bf16(rng, elements, 1.0);
        std::normal_distribution<float> value(0.0F, 1.0F);
        v16.resize(elements);
        for (auto& x : v16) { x = to_fp16(value(rng)); }
    }
    // Pooled keys fill the whole plane; selection reads only complete blocks. Pooled keys and index
    // queries lie on the exact grid, so the oracle's selection is the contract's (grid_bf16).
    const auto pooled = grid_bf16(rng, static_cast<std::size_t>(kSlotWidth) * kPage * pages);
    const auto q      = random_bf16(rng, static_cast<std::size_t>(kD) * kHeads * columns, 1.0);
    const auto iq     = grid_bf16(rng, static_cast<std::size_t>(kDi) * kIndexHeads * columns);
    std::vector<std::int32_t> positions(columns), table_rows(rows), tail_slots(rows, 0);
    for (int r = 0; r < rows; ++r) {
        table_rows[r] = r;
        for (int j = 0; j < width; ++j) {
            positions[static_cast<std::size_t>(r) * width + j] = c.starts[r] + j;
        }
    }

    DeviceBuffer dk = int8 ? upload(k8) : upload(k16), dv = int8 ? upload(v8) : upload(v16);
    DeviceBuffer dks, dvs;
    if (int8) {
        dks = upload(k_scale);
        dvs = upload(v_scale);
    }
    DeviceBuffer dpooled = upload(pooled), dq = upload(q), diq = upload(iq),
                 dtables = upload(tables);
    DeviceBuffer dpos = upload(positions), drows = upload(table_rows), dslots = upload(tail_slots);
    ops::QsaKVLayer layer;
    layer.kv.storage      = storage;
    layer.kv.head_dim     = kD;
    layer.kv.num_kv_heads = kKvHeads;
    layer.kv.k_pages = Tensor(dk.p, int8 ? DType::I8 : DType::BF16, {kD, kPage, kKvHeads, pages});
    layer.kv.v_pages = Tensor(dv.p, int8 ? DType::I8 : DType::FP16, {kD, kPage, kKvHeads, pages});
    if (int8) {
        layer.kv.k_scale_pages = Tensor(dks.p, DType::FP16, {kD / 64, kPage, kKvHeads, pages});
        layer.kv.v_scale_pages = Tensor(dvs.p, DType::FP16, {kD / 64, kPage, kKvHeads, pages});
    }
    layer.pooled_pages = Tensor(dpooled.p, DType::BF16, {kSlotWidth, kPage, 1, pages});
    const ops::QsaBatch batch{.block_tables = Tensor(dtables.p, DType::I32, {pages_per_row, rows}),
                              .table_rows   = Tensor(drows.p, DType::I32, {rows}),
                              .positions    = Tensor(dpos.p, DType::I32, {columns}),
                              .tail_slots   = Tensor(dslots.p, DType::I32, {rows}),
                              .batch        = rows,
                              .width        = width};
    const std::size_t workspace_bytes =
        ops::qsa_attention_workspace_bytes(kGeometry, columns, max_context);
    DeviceBuffer workspace(workspace_bytes),
        dout(static_cast<std::size_t>(kD) * kHeads * columns * 2);
    Tensor out(dout.p, DType::BF16, {kD, kHeads, columns});
    ops::qsa_attention(Tensor(dq.p, DType::BF16, {kD, kHeads, columns}),
                       Tensor(diq.p, DType::BF16, {kDi, kIndexHeads, columns}), layer, batch,
                       kGeometry, 1.0F / 16.0F, max_context, workspace.p, workspace_bytes, out,
                       nullptr);
    synchronize("attention");
    const auto got = t::from_device_bf16(dout, static_cast<std::size_t>(kD) * kHeads * columns);
    // Width invariance inside the unsplit class: the row's last 100 columns as a call of their own
    // (also unsplit) give the same bits; the prefix cache resumes rely on it.
    if (rows == 1 && width >= 186) {
        constexpr int kSub = 100;
        const int first    = width - kSub;
        const ops::QsaBatch sub{.block_tables = Tensor(dtables.p, DType::I32, {pages_per_row, rows}),
                                .table_rows   = Tensor(drows.p, DType::I32, {1}),
                                .positions = Tensor(static_cast<std::int32_t*>(dpos.p) + first, DType::I32, {kSub}),
                                .tail_slots = Tensor(dslots.p, DType::I32, {1}),
                                .batch      = 1,
                                .width      = kSub};
        DeviceBuffer dsub(static_cast<std::size_t>(kD) * kHeads * kSub * 2);
        Tensor sub_out(dsub.p, DType::BF16, {kD, kHeads, kSub});
        ops::qsa_attention(Tensor(static_cast<std::uint16_t*>(dq.p) + static_cast<std::size_t>(first) * kD * kHeads,
                                  DType::BF16, {kD, kHeads, kSub}),
                           Tensor(static_cast<std::uint16_t*>(diq.p) +
                                      static_cast<std::size_t>(first) * kDi * kIndexHeads,
                                  DType::BF16, {kDi, kIndexHeads, kSub}),
                           layer, sub, kGeometry, 1.0F / 16.0F, max_context, workspace.p, workspace_bytes, sub_out,
                           nullptr);
        synchronize("attention sub-call");
        const auto part = t::from_device_bf16(dsub, static_cast<std::size_t>(kD) * kHeads * kSub);
        bool same = true;
        for (std::size_t i = 0; i < part.size() && same; ++i) {
            same = part[i] == got[static_cast<std::size_t>(first) * kD * kHeads + i];
        }
        expect(same, std::string("qsa_attention ") + (int8 ? "int8 " : "bf16 ") + c.name +
                         ": a 100-column call equals the same columns of the wider call bit for bit");
    }

    // Page spaces (design §19.3.11): the same KV with every page whose id is 1 mod 3 in a mapped host
    // plane and every page 2 mod 3 in a separate device ("lent") plane, their pool copies zeroed so
    // only the translated reads can find them; selection and attention must give the same bits.
    {
        const std::size_t kv_page    = static_cast<std::size_t>(kD) * kPage * kKvHeads * (int8 ? 1 : 2);
        const std::size_t sc_page    = static_cast<std::size_t>(kD / 64) * kPage * kKvHeads * 2;
        const std::size_t pool_page  = static_cast<std::size_t>(kSlotWidth) * kPage * 2;
        std::vector<int> host_of(pages, -1), lent_of(pages, -1);
        int host_pages = 0, lent_pages = 0;
        for (int p = 0; p < pages; ++p) {
            if (p % 3 == 1) { host_of[p] = host_pages++; }
            if (p % 3 == 2) { lent_of[p] = lent_pages++; }
        }
        const auto planes = [&](std::size_t page_bytes, int count) { return page_bytes * std::max(count, 1); };
        void* host_mem = nullptr;
        const std::size_t host_bytes = 2 * planes(kv_page, host_pages) + (int8 ? 2 * planes(sc_page, host_pages) : 0);
        expect(cudaHostAlloc(&host_mem, host_bytes, cudaHostAllocMapped) == cudaSuccess, "host planes allocate");
        auto* host_k  = static_cast<std::byte*>(host_mem);
        auto* host_v  = host_k + planes(kv_page, host_pages);
        auto* host_ks = host_v + planes(kv_page, host_pages);
        auto* host_vs = host_ks + planes(sc_page, host_pages);
        DeviceBuffer lent_k(planes(kv_page, lent_pages)), lent_v(planes(kv_page, lent_pages));
        DeviceBuffer lent_ks(planes(sc_page, lent_pages)), lent_vs(planes(sc_page, lent_pages));
        DeviceBuffer spaces_pooled(planes(pool_page, host_pages + lent_pages));
        const auto move = [&](const DeviceBuffer& pool, std::size_t page_bytes, int p, void* host_plane,
                              const DeviceBuffer& lent_plane) {
            auto* source = static_cast<std::byte*>(pool.p) + page_bytes * p;
            void* target = host_of[p] >= 0 ? static_cast<std::byte*>(host_plane) + page_bytes * host_of[p]
                                           : static_cast<std::byte*>(lent_plane.p) + page_bytes * lent_of[p];
            expect(cudaMemcpy(target, source, page_bytes, cudaMemcpyDefault) == cudaSuccess, "page copy");
            expect(cudaMemset(source, 0, page_bytes) == cudaSuccess, "pool page cleared");
        };
        std::vector<std::int32_t> moved(tables.size());
        for (int p = 0; p < pages; ++p) {
            if (host_of[p] < 0 && lent_of[p] < 0) { continue; }
            move(dk, kv_page, p, host_k, lent_k);
            move(dv, kv_page, p, host_v, lent_v);
            if (int8) {
                move(dks, sc_page, p, host_ks, lent_ks);
                move(dvs, sc_page, p, host_vs, lent_vs);
            }
            const int slot = host_of[p] >= 0 ? host_of[p] : host_pages + lent_of[p];
            auto* source   = static_cast<std::byte*>(dpooled.p) + pool_page * p;
            expect(cudaMemcpy(static_cast<std::byte*>(spaces_pooled.p) + pool_page * slot, source, pool_page,
                              cudaMemcpyDeviceToDevice) == cudaSuccess,
                   "pooled page copy");
            expect(cudaMemset(source, 0, pool_page) == cudaSuccess, "pool pooled page cleared");
        }
        for (std::size_t i = 0; i < tables.size(); ++i) {
            const int p = tables[i];
            moved[i]    = host_of[p] >= 0 ? pages + host_of[p] : lent_of[p] >= 0 ? pages + host_pages + lent_of[p] : p;
        }
        DeviceBuffer dmoved = upload(moved);
        ops::QsaKVLayer spaced = layer;
        spaced.spaces = ops::QsaPageSpaces{.device_pages = pages,
                                           .host_pages   = host_pages,
                                           .host_k       = host_k,
                                           .host_v       = host_v,
                                           .host_k_scale = int8 ? host_ks : nullptr,
                                           .host_v_scale = int8 ? host_vs : nullptr,
                                           .lent_k       = lent_k.p,
                                           .lent_v       = lent_v.p,
                                           .lent_k_scale = int8 ? lent_ks.p : nullptr,
                                           .lent_v_scale = int8 ? lent_vs.p : nullptr,
                                           .pooled       = spaces_pooled.p};
        ops::QsaBatch moved_batch = batch;
        moved_batch.block_tables  = Tensor(dmoved.p, DType::I32, {pages_per_row, rows});
        DeviceBuffer dspaced(static_cast<std::size_t>(kD) * kHeads * columns * 2);
        Tensor spaced_out(dspaced.p, DType::BF16, {kD, kHeads, columns});
        ops::qsa_attention(Tensor(dq.p, DType::BF16, {kD, kHeads, columns}),
                           Tensor(diq.p, DType::BF16, {kDi, kIndexHeads, columns}), spaced, moved_batch, kGeometry,
                           1.0F / 16.0F, max_context, workspace.p, workspace_bytes, spaced_out, nullptr);
        synchronize("attention over page spaces");
        const auto spaced_got = t::from_device_bf16(dspaced, static_cast<std::size_t>(kD) * kHeads * columns);
        expect(spaced_got == got, std::string("qsa_attention ") + (int8 ? "int8 " : "bf16 ") + c.name +
                                      ": pages in host and lent spaces give the pool's bits");
        cudaFreeHost(host_mem);
    }
    std::vector<double> ref(got.size());
    const std::string tag = std::string("qsa_attention ") + (int8 ? "int8" : "bf16") + " " + c.name;
    for (int r = 0; r < rows; ++r) {
        const int keys     = c.starts[r] + width;
        const auto page_of = [&](int token) {
            return tables[static_cast<std::size_t>(r) * pages_per_row + token / kPage];
        };
        // Decoded keys and values of the row, [head][token][d].
        std::vector<double> key(static_cast<std::size_t>(kKvHeads) * keys * kD), value(key.size());
        t::parallel_ranges(
            static_cast<std::int64_t>(kKvHeads) * keys, t::threads_for_rows(kKvHeads * keys),
            [&](std::int64_t begin, std::int64_t end) {
                std::vector<double> rotated(kD);
                for (std::int64_t i = begin; i < end; ++i) {
                    const int head = static_cast<int>(i / keys), token = static_cast<int>(i % keys);
                    const int page = page_of(token);
                    for (int d = 0; d < kD; ++d) {
                        const std::size_t at = plane_index(kD, page, head, token, d);
                        if (int8) {
                            const std::size_t g = plane_index(kD / 64, page, head, token, d / 64);
                            rotated[d]          = k8[at] * from_fp16(k_scale[g]);
                            value[static_cast<std::size_t>(i) * kD + d] =
                                v8[at] * from_fp16(v_scale[g]);
                        } else {
                            key[static_cast<std::size_t>(i) * kD + d]   = from_bf16(k16[at]);
                            value[static_cast<std::size_t>(i) * kD + d] = from_fp16(v16[at]);
                        }
                    }
                    if (int8) {
                        hadamard(rotated);
                        std::copy(rotated.begin(), rotated.end(),
                                  key.begin() + static_cast<std::ptrdiff_t>(i) * kD);
                    }
                }
            });
        // Attended tokens of each column: every token while dense, else the selected blocks and the
        // incomplete block's positions.
        std::vector<std::vector<char>> attended(width, std::vector<char>(keys, 0));
        for (int j = 0; j < width; ++j) {
            const int p = c.starts[r] + j, complete = (p + 1) / kR;
            auto& visible = attended[j];
            if (complete <= kTopBlocks) {
                std::fill(visible.begin(), visible.begin() + p + 1, 1);
                continue;
            }
            const std::size_t column = static_cast<std::size_t>(r) * width + j;
            std::vector<std::pair<double, int>> scores(complete);
            for (int b = 0; b < complete; ++b) {
                double score = 0.0;
                for (int h = 0; h < kIndexHeads; ++h) {
                    double dot = 0.0;
                    for (int d = 0; d < kDi; ++d) {
                        const int token = kR * b + d / kSlotWidth;
                        const std::size_t at =
                            (static_cast<std::size_t>(page_of(token)) * kPage + token % kPage) *
                                kSlotWidth +
                            d % kSlotWidth;
                        dot += from_bf16(iq[(column * kIndexHeads + h) * kDi + d]) *
                               from_bf16(pooled[at]);
                    }
                    score += std::max(dot, 0.0);
                }
                scores[b] = {score / std::sqrt(static_cast<double>(kDi)), b};
            }
            std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            for (int i = 0; i < kTopBlocks; ++i) {
                std::fill(visible.begin() + kR * scores[i].second,
                          visible.begin() + kR * scores[i].second + kR, 1);
            }
            std::fill(visible.begin() + kR * complete, visible.begin() + p + 1, 1);
        }
        t::naive_dense_softmax_attention(
            ops::AttentionHeadGeometry{.head_dim = kD, .query_heads = kHeads, .kv_heads = kKvHeads},
            width, keys, 1.0 / 16.0,
            [&](int d, int h, int j) {
                return from_bf16(
                    q[((static_cast<std::size_t>(r) * width + j) * kHeads + h) * kD + d]);
            },
            [&](int d, int h, int token) {
                return key[(static_cast<std::size_t>(h) * keys + token) * kD + d];
            },
            [&](int d, int h, int token) {
                return value[(static_cast<std::size_t>(h) * keys + token) * kD + d];
            },
            [&](int j, int token) { return attended[j][token] != 0; },
            [&](int d, int h, int j, double v) {
                ref[((static_cast<std::size_t>(r) * width + j) * kHeads + h) * kD + d] = v;
            });
    }
    g_failures += t::verify_reduction(tag, got, ref, kAttentionCriterion);
}

} // namespace

int main() {
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        for (const int columns : {1, 5, 64, 1024}) { test_index_query(columns, 100U + columns); }
        test_pool(7);
        // Decode across the dense/selected boundary (2,051 visible tokens is the last dense one),
        // verification across a page boundary, prefill columns crossing the boundary inside one
        // call (one attention split), and a split-K prefill width.
        // Calls the FP32 kernel would run unsplit (86+ columns here) take the Tensor Core prompt
        // route: dense columns, the dense/selected boundary inside the call, selected blocks with
        // tails over several pages, two rows, a list length off the 16-token tile, and the class's
        // narrowest call. Every column of a prompt-route call must also equal, bit for bit, the
        // same column computed in a call of another width of the class (checked below).
        const AttentionCase cases[] = {{"decode W=1 B=2", 1, {2050, 2051}},
                                       {"verify W=5 B=2", 5, {3001, 4093}},
                                       {"prefill W=192 B=1", 192, {1900}},
                                       {"prefill W=40 B=1", 40, {5000}},
                                       {"prompt W=256 B=1 dense", 256, {0}},
                                       {"prompt W=300 B=1 boundary", 300, {1901}},
                                       {"prompt W=256 B=2 selected", 256, {6001, 9013}},
                                       {"prompt W=257 B=1 selected", 257, {4093}},
                                       {"prompt W=86 B=1 class edge", 86, {7000}}};
        for (const auto storage : {KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64}) {
            std::uint32_t seed = 31;
            for (const auto& c : cases) { test_attention(storage, c, seed++); }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all qsa checks passed\n");
    return 0;
}
