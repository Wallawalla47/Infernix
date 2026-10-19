// QSA (include/infernix/ops/qsa.h; docs/maintainer/qwen3_8-flash-next-design.md §8.10) against
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
//     storage (bf16; int8-g64 and fp8-row with Hadamard-rotated keys; nvfp4 and k8v4 with rotated
//     keys and values, the output rotated back; vq2 and k4v2 appended by the Op itself, with the
//     exact recent-key window: slots with matching and stale tags, and a staged wide call), at
//     decode, verification and prefill
//     widths, across the dense/selected boundary and page boundaries.
//   - qsa_prepare (the qsa-prepare-fused fast path): Q, the index query, the stored keys (decoded
//     exactly from their codes and scales) and the pooled keys against the FP64 chain norm -> RoPE
//     (-> Hadamard) under named one-rounding / codec criteria; values against kv_cache_append_batch's
//     bytes; the tails exactly; every other cache byte untouched; and bit for bit: W = 8 against
//     eight W = 1 calls, B = 4 against four B = 1 calls, K/V-only and in-place calls, page spaces and
//     CUDA Graph replay over new device rows and positions.
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "infernix/ops/attention_geometry.h"
#include "infernix/ops/kv_cache_append.h"
#include "infernix/ops/qsa.h"
#include "ops/host_parallel.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace ops = infernix::ops;
namespace t   = infernix::test;
using infernix::DeviceBuffer;
using infernix::DType;
using infernix::KvCacheStorage;
using infernix::Tensor;

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

// The geometry of the index-query, pool and tail tests: kGeometry, or kGeometry with YaRN.
ops::QsaGeometry g_index_geometry = kGeometry;

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
// from their stored codes (Q is not quantized for any storage; rotated values are rotated back
// in FP32), then rounds the output to BF16, so every storage shares the Softmax Attention
// suite's bf16 decode criterion.
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
// axis i % 3 (interleaved M-RoPE). Under YaRN (g_index_geometry.yarn_factor f > 1, original span L),
// written from its definition: d(beta) = R ln(L / (2 pi beta)) / (2 ln theta), low = max(floor(d(32)), 0),
// high = min(ceil(d(1)), R - 1) (+0.001 if equal), ramp_i = clamp((i - low) / (high - low), 0, 1); pair i
// turns by theta^(-2i/R) ((1 - ramp_i) + ramp_i / f), and cos and sin are scaled by 1 + 0.1 ln f.
void rotate(std::vector<double>& x, const Rope& rope) {
    constexpr int half  = kRotary / 2;
    const double theta  = g_index_geometry.theta, f = g_index_geometry.yarn_factor;
    const double span   = g_index_geometry.original_positions;
    const auto d        = [&](double beta) { return kRotary * std::log(span / (2.0 * std::acos(-1.0) * beta)) / (2.0 * std::log(theta)); };
    const double low    = f > 1.0 ? std::max(std::floor(d(32.0)), 0.0) : 0.0;
    double high         = f > 1.0 ? std::min(std::ceil(d(1.0)), static_cast<double>(kRotary - 1)) : 1.0;
    if (low == high) { high += 0.001; }
    const double scale  = f > 1.0 ? 1.0 + 0.1 * std::log(f) : 1.0;
    for (int i = 0; i < half; ++i) {
        const double ramp  = f > 1.0 ? std::clamp((i - low) / (high - low), 0.0, 1.0) : 0.0;
        const double inv   = std::pow(theta, -2.0 * i / kRotary) * ((1.0 - ramp) + ramp / f);
        const double angle = rope[i % 3] * inv;
        const double c = scale * std::cos(angle), s = scale * std::sin(angle);
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

void test_index_query(int columns, std::uint32_t seed, int last_position = 8191) {
    std::mt19937 rng(seed);
    const auto q = random_bf16(rng, static_cast<std::size_t>(kDi) * kIndexHeads * columns, 2.0);
    const auto weight = random_bf16(rng, kDi, 0.3);
    // Distinct axes per column (axis a of column c at word a * columns + c).
    std::uniform_int_distribution<int> position(0, last_position);
    std::vector<std::int32_t> rope(3 * static_cast<std::size_t>(columns));
    for (auto& p : rope) { p = position(rng); }
    DeviceBuffer dq = upload(q), dw = upload(weight), dp = upload(rope);
    Tensor tq(dq.p, DType::BF16, {kDi, kIndexHeads, columns});
    ops::qsa_index_query(tq, Tensor(dw.p, DType::BF16, {kDi}), Tensor(dp.p, DType::I32, {columns, 3}),
                         g_index_geometry, nullptr);
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
    g_failures += t::verify_reduction(std::string("qsa_index_query") +
                                          (g_index_geometry.yarn_factor > 1.0F
                                               ? " yarn L=" + std::to_string(g_index_geometry.original_positions)
                                               : std::string()) +
                                          " T=" + std::to_string(columns) +
                                          " positions<=" + std::to_string(last_position),
                                      got, ref,
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
                               qsa_batch, g_index_geometry, nullptr);
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
                                  Tensor(commit_slots_.p, DType::I32, {rows_}), g_index_geometry, nullptr);
        }
        synchronize("pool step");
        for (int r = 0; r < rows_; ++r) {
            written_[r] = std::max(written_[r], starts[r] + s.width);
            frontier_[r] += s.verify ? s.commit[r] : s.width;
        }
        const std::string tag = std::string(case_.name) + (g_index_geometry.yarn_factor > 1.0F ? " yarn" : "") +
                                " step " + std::to_string(index) + (s.verify ? " verify W=" : " call W=") +
                                std::to_string(s.width);
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

const char* storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    default: return "?";
    }
}

std::uint8_t to_e4m3(float v) { return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }

double from_e4m3(std::uint8_t code) {
    __nv_fp8_e4m3 v;
    v.__x = code;
    return static_cast<double>(static_cast<float>(v));
}

double from_e2m1(unsigned code) {
    static constexpr double kMagnitude[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (code & 8U) != 0 ? -kMagnitude[code & 7U] : kMagnitude[code & 7U];
}

// One K or V vector plane pair of a storage: data bytes and scale bytes per token, stored
// [leading bytes, 64 tokens, heads, pages] (paged_kv_storage_layout), with random codes whose
// decoded values are O(1), and its exact FP64 decode.
struct StoredVector {
    infernix::PagedKVVectorLayout layout;
    std::vector<std::uint8_t> data, scales;
    enum class Kind { Bf16, Fp16, Int8, Fp8, Nvfp4 } kind;

    [[nodiscard]] int data_bytes() const {
        return layout.data_leading_extent * static_cast<int>(infernix::dtype_size(layout.data_dtype));
    }
    [[nodiscard]] int scale_bytes() const {
        return layout.scale_leading_extent * static_cast<int>(infernix::dtype_size(layout.scale_dtype));
    }

    StoredVector(infernix::PagedKVVectorLayout l, int pages, std::mt19937& rng) : layout(l) {
        const std::size_t rows = static_cast<std::size_t>(kPage) * kKvHeads * pages;
        data.resize(rows * data_bytes());
        scales.resize(rows * scale_bytes());
        std::normal_distribution<float> normal(0.0F, 1.0F);
        const auto put16 = [](std::vector<std::uint8_t>& v, std::size_t i, std::uint16_t bits) {
            std::memcpy(v.data() + 2 * i, &bits, 2);
        };
        switch (layout.data_dtype) {
        case DType::BF16:
            kind = Kind::Bf16;
            for (std::size_t i = 0; i < data.size() / 2; ++i) { put16(data, i, to_bf16(normal(rng))); }
            break;
        case DType::FP16:
            kind = Kind::Fp16;
            for (std::size_t i = 0; i < data.size() / 2; ++i) { put16(data, i, to_fp16(normal(rng))); }
            break;
        case DType::I8: {
            kind = Kind::Int8;
            std::uniform_int_distribution<int> code(-127, 127);
            std::uniform_real_distribution<float> scale(0.004F, 0.012F);
            for (auto& x : data) { x = static_cast<std::uint8_t>(static_cast<std::int8_t>(code(rng))); }
            for (std::size_t i = 0; i < scales.size() / 2; ++i) { put16(scales, i, to_fp16(scale(rng))); }
            break;
        }
        case DType::FP8_E4M3FN: {
            // Codes of N(0, 100^2), one FP16 row scale near 1/100.
            kind = Kind::Fp8;
            std::uniform_real_distribution<float> scale(0.006F, 0.014F);
            for (auto& x : data) { x = to_e4m3(100.0F * normal(rng)); }
            for (std::size_t i = 0; i < scales.size() / 2; ++i) { put16(scales, i, to_fp16(scale(rng))); }
            break;
        }
        default: {
            // Packed E2M1 codes (every nibble), E4M3 group scales in [0.15, 0.4].
            kind = Kind::Nvfp4;
            std::uniform_int_distribution<int> byte(0, 255);
            std::uniform_real_distribution<float> scale(0.15F, 0.4F);
            for (auto& x : data) { x = static_cast<std::uint8_t>(byte(rng)); }
            for (auto& x : scales) { x = to_e4m3(scale(rng)); }
            break;
        }
        }
    }

    // Element d of (page, head, token), exactly.
    [[nodiscard]] double decode(int page, int head, int token, int d) const {
        const std::size_t row = (static_cast<std::size_t>(page) * kKvHeads + head) * kPage + token % kPage;
        const std::uint8_t* bytes = data.data() + row * data_bytes();
        const std::uint8_t* s     = scales.data() + row * scale_bytes();
        std::uint16_t bits        = 0;
        switch (kind) {
        case Kind::Bf16: std::memcpy(&bits, bytes + 2 * d, 2); return from_bf16(bits);
        case Kind::Fp16: std::memcpy(&bits, bytes + 2 * d, 2); return from_fp16(bits);
        case Kind::Int8:
            std::memcpy(&bits, s + 2 * (d / 64), 2);
            return static_cast<std::int8_t>(bytes[d]) * from_fp16(bits);
        case Kind::Fp8: std::memcpy(&bits, s, 2); return from_e4m3(bytes[d]) * from_fp16(bits);
        case Kind::Nvfp4: return from_e2m1((bytes[d / 2] >> (4 * (d % 2))) & 0xFU) * from_e4m3(s[d / 16]);
        }
        return 0.0;
    }

    [[nodiscard]] std::size_t data_page_bytes() const {
        return static_cast<std::size_t>(data_bytes()) * kPage * kKvHeads;
    }
    [[nodiscard]] std::size_t scale_page_bytes() const {
        return static_cast<std::size_t>(scale_bytes()) * kPage * kKvHeads;
    }
};

// The VQ2 codebook (ops/kv_cache/vq2_codec.cuh): magnitudes, pattern-major, eight per pattern.
constexpr std::int8_t kVq2Codebook[512 * 8] = {
#include "ops/kv_cache/vq2_codebook.inc"
};

// Attended tokens of row r's columns (positions starts[r] + j): every token while dense, else the
// selected blocks (exact scores, lower ids on equal scores) and the incomplete block's positions.
std::vector<std::vector<char>> attended_tokens(int r, int width, int start, const std::vector<std::uint16_t>& iq,
                                               const std::vector<std::uint16_t>& pooled,
                                               const std::function<int(int)>& page_of) {
    const int keys = start + width;
    std::vector<std::vector<char>> attended(width, std::vector<char>(keys, 0));
    for (int j = 0; j < width; ++j) {
        const int p = start + j, complete = (p + 1) / kR;
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
                    const int token      = kR * b + d / kSlotWidth;
                    const std::size_t at = (static_cast<std::size_t>(page_of(token)) * kPage + token % kPage) *
                                               kSlotWidth +
                                           d % kSlotWidth;
                    dot += from_bf16(iq[(column * kIndexHeads + h) * kDi + d]) * from_bf16(pooled[at]);
                }
                score += std::max(dot, 0.0);
            }
            scores[b] = {score / std::sqrt(static_cast<double>(kDi)), b};
        }
        std::sort(scores.begin(), scores.end(),
                  [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        for (int i = 0; i < kTopBlocks; ++i) {
            std::fill(visible.begin() + kR * scores[i].second, visible.begin() + kR * scores[i].second + kR, 1);
        }
        std::fill(visible.begin() + kR * complete, visible.begin() + p + 1, 1);
    }
    return attended;
}

// ------------------------------------------------------------- vector-quantized storages (vq2, k4v2)

// The exact window's slot tag (ops/kv_cache/kv_window.cuh): FNV-1a over the row's code words and its
// FP16 scale bits, finalized and forced odd.
std::uint32_t window_tag(int position, const std::uint8_t* codes, int code_bytes, std::uint16_t scale_bits) {
    std::uint32_t hash = (2166136261u ^ static_cast<std::uint32_t>(position)) * 16777619u;
    for (int i = 0; i < code_bytes / 4; ++i) {
        std::uint32_t word;
        std::memcpy(&word, codes + 4 * i, 4);
        hash = (hash ^ word) * 16777619u;
    }
    hash = (hash ^ scale_bits) * 16777619u;
    hash ^= hash >> 15;
    hash *= 0x2c1b3c6du;
    hash ^= hash >> 12;
    return hash | 1u;
}

int window_slot(int position) {
    return position < infernix::kKVWindowSinkTokens
               ? position
               : infernix::kKVWindowSinkTokens + (position & (infernix::kKVWindowRingTokens - 1));
}

// Coordinate i of a stored code row, as the signed integer the codec multiplies by the row scale:
// VQ2 (64 bytes, 16-bit word codes: pattern, seven signs, even parity) or Q4 (128 bytes of nibbles).
int vq_code(const std::uint8_t* row, bool q4, int i) {
    if (q4) {
        static constexpr int kLevels[8] = {4, 12, 21, 30, 40, 52, 66, 87};
        const unsigned n = (row[i / 2] >> (4 * (i % 2))) & 0xFU;
        return n >= 8 ? kLevels[n ^ 8U] : -kLevels[n ^ 7U];
    }
    std::uint16_t word;
    std::memcpy(&word, row + 2 * (i / 8), 2);
    const unsigned pattern = word & 0x1FFU, seven = (word >> 9) & 0x7FU;
    const unsigned signs   = seven | ((std::popcount(seven) & 1U) << 7);
    const int magnitude    = kVq2Codebook[pattern * 8 + i % 8];
    return ((signs >> (i % 8)) & 1U) != 0 ? -magnitude : magnitude;
}

struct VqCase {
    const char* name;
    int width;
    std::vector<int> starts;
    bool window;
};

// The Op appends the call's K/V itself. Before the call, every row's earlier positions hold random
// code rows; with a window, each slot of a position the queries may read exactly holds an INT8 row
// whose tag matches the stored codes for three of four positions and is stale for the fourth (those
// keys must read their codes). After the call the paged rows and window are read back: a query reads
// an earlier key exactly from the pre-call slot when its tag matched, and a call key from the call's
// own exact row (its committed slot: every case's call fits the final kKVWindowRingTokens).
void test_vq_attention(KvCacheStorage storage, const VqCase& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const bool q4    = storage == KvCacheStorage::Q4KeyVq2Value;
    const auto layout = infernix::paged_kv_storage_layout(storage, kD);
    const int kb = layout.key.data_leading_extent, vb = layout.value.data_leading_extent; // code bytes
    const int rows = static_cast<int>(c.starts.size()), width = c.width, columns = rows * width;
    int context = 0;
    for (const int s : c.starts) { context = std::max(context, s + width); }
    const int pages_per_row = (context + kPage - 1) / kPage, pages = rows * pages_per_row;
    const int max_context   = pages_per_row * kPage;
    std::vector<std::int32_t> tables(pages);
    std::iota(tables.begin(), tables.end(), 0);
    std::shuffle(tables.begin(), tables.end(), rng);
    const auto page_of = [&](int r, int token) { return tables[static_cast<std::size_t>(r) * pages_per_row + token / kPage]; };
    const auto row_of  = [&](int page, int head, int token) {
        return (static_cast<std::size_t>(page) * kKvHeads + head) * kPage + token % kPage;
    };

    // Paged code rows and FP16 row scales (random codes everywhere; the call's rows are rewritten).
    const std::size_t paged_rows = static_cast<std::size_t>(kPage) * kKvHeads * pages;
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.004F, 0.012F);
    std::vector<std::uint8_t> kc(paged_rows * kb), vc(paged_rows * vb);
    std::vector<std::uint16_t> ks(paged_rows), vs(paged_rows);
    for (auto& x : kc) { x = static_cast<std::uint8_t>(byte(rng)); }
    for (auto& x : vc) { x = static_cast<std::uint8_t>(byte(rng)); }
    for (auto& x : ks) { x = to_fp16(scale(rng)); }
    for (auto& x : vs) { x = to_fp16(scale(rng)); }

    // The window before the call: [256, slots, Hkv, rows] codes, [4, ...] scales, [2, ...] tags.
    const std::size_t slot_rows = static_cast<std::size_t>(infernix::kKVWindowSlots) * kKvHeads * rows;
    std::vector<std::int8_t> wk(slot_rows * kD), wv(slot_rows * kD);
    std::vector<std::uint16_t> wks(slot_rows * 4), wvs(slot_rows * 4);
    std::vector<std::int32_t> wtags(slot_rows * 2, 0);
    std::vector<char> stale(static_cast<std::size_t>(rows) * kKvHeads * context, 0);
    const auto slot_row = [&](int r, int head, int slot) {
        return (static_cast<std::size_t>(r) * kKvHeads + head) * infernix::kKVWindowSlots + slot;
    };
    if (c.window) {
        std::uniform_int_distribution<int> code(-127, 127);
        std::uniform_real_distribution<float> group_scale(0.006F, 0.012F);
        for (auto& x : wk) { x = static_cast<std::int8_t>(code(rng)); }
        for (auto& x : wv) { x = static_cast<std::int8_t>(code(rng)); }
        for (auto& x : wks) { x = to_fp16(group_scale(rng)); }
        for (auto& x : wvs) { x = to_fp16(group_scale(rng)); }
        for (int r = 0; r < rows; ++r) {
            const int start = c.starts[r];
            for (int key = 0; key < start; ++key) {
                if (key >= infernix::kKVWindowSinkTokens && key < start - infernix::kKVWindowRingTokens) { continue; }
                for (int head = 0; head < kKvHeads; ++head) {
                    const std::size_t prow = row_of(page_of(r, key), head, key);
                    const std::size_t srow = slot_row(r, head, window_slot(key));
                    const bool is_stale    = (key + head) % 4 == 3;
                    stale[(static_cast<std::size_t>(r) * kKvHeads + head) * context + key] = is_stale;
                    wtags[srow * 2] = static_cast<std::int32_t>(window_tag(key, &kc[prow * kb], kb, ks[prow]) ^
                                                                (is_stale ? 2U : 0U));
                    wtags[srow * 2 + 1] = static_cast<std::int32_t>(window_tag(key, &vc[prow * vb], vb, vs[prow]) ^
                                                                    (is_stale ? 2U : 0U));
                }
            }
        }
    }

    const auto pooled = grid_bf16(rng, static_cast<std::size_t>(kSlotWidth) * kPage * pages);
    const auto q      = random_bf16(rng, static_cast<std::size_t>(kD) * kHeads * columns, 1.0);
    const auto iq     = grid_bf16(rng, static_cast<std::size_t>(kDi) * kIndexHeads * columns);
    const auto call_k = random_bf16(rng, static_cast<std::size_t>(kD) * kKvHeads * columns, 1.0);
    const auto call_v = random_bf16(rng, static_cast<std::size_t>(kD) * kKvHeads * columns, 1.0);
    std::vector<std::int32_t> positions(columns), table_rows(rows), tail_slots(rows), window_rows(rows);
    for (int r = 0; r < rows; ++r) {
        table_rows[r] = r;
        tail_slots[r] = 0;
        window_rows[r] = r;
        for (int j = 0; j < width; ++j) { positions[static_cast<std::size_t>(r) * width + j] = c.starts[r] + j; }
    }

    DeviceBuffer dk = upload(kc), dv = upload(vc), dks = upload(ks), dvs = upload(vs);
    DeviceBuffer dwk = upload(wk), dwv = upload(wv), dwks = upload(wks), dwvs = upload(wvs), dwtags = upload(wtags),
                 dwrows = upload(window_rows);
    DeviceBuffer dpooled = upload(pooled), dq = upload(q), diq = upload(iq), dtables = upload(tables);
    DeviceBuffer dpos = upload(positions), drows = upload(table_rows), dslots = upload(tail_slots);
    DeviceBuffer dck = upload(call_k), dcv = upload(call_v);
    ops::QsaKVLayer layer;
    layer.kv.storage      = storage;
    layer.kv.head_dim     = kD;
    layer.kv.num_kv_heads = kKvHeads;
    layer.kv.k_pages       = Tensor(dk.p, DType::U8, {kb, kPage, kKvHeads, pages});
    layer.kv.v_pages       = Tensor(dv.p, DType::U8, {vb, kPage, kKvHeads, pages});
    layer.kv.k_scale_pages = Tensor(dks.p, DType::FP16, {1, kPage, kKvHeads, pages});
    layer.kv.v_scale_pages = Tensor(dvs.p, DType::FP16, {1, kPage, kKvHeads, pages});
    if (c.window) {
        const int slots = infernix::kKVWindowSlots;
        layer.kv.window = infernix::PagedKVWindowView{
            .k_codes  = Tensor(dwk.p, DType::I8, {kD, slots, kKvHeads, rows}),
            .v_codes  = Tensor(dwv.p, DType::I8, {kD, slots, kKvHeads, rows}),
            .k_scales = Tensor(dwks.p, DType::FP16, {4, slots, kKvHeads, rows}),
            .v_scales = Tensor(dwvs.p, DType::FP16, {4, slots, kKvHeads, rows}),
            .tags     = Tensor(dwtags.p, DType::I32, {2, slots, kKvHeads, rows}),
            .slots    = Tensor(dwrows.p, DType::I32, {rows})};
    }
    layer.pooled_pages = Tensor(dpooled.p, DType::BF16, {kSlotWidth, kPage, 1, pages});
    const ops::QsaBatch batch{.block_tables = Tensor(dtables.p, DType::I32, {pages_per_row, rows}),
                              .table_rows   = Tensor(drows.p, DType::I32, {rows}),
                              .positions    = Tensor(dpos.p, DType::I32, {columns}),
                              .tail_slots   = Tensor(dslots.p, DType::I32, {rows}),
                              .batch        = rows,
                              .width        = width};
    const ops::QsaAppend append{.k = Tensor(dck.p, DType::BF16, {kD, kKvHeads, width, rows}),
                                .v = Tensor(dcv.p, DType::BF16, {kD, kKvHeads, width, rows})};
    const std::size_t workspace_bytes = ops::qsa_attention_workspace_bytes(kGeometry, columns, max_context, storage);
    DeviceBuffer workspace(workspace_bytes), dout(static_cast<std::size_t>(kD) * kHeads * columns * 2);
    Tensor out(dout.p, DType::BF16, {kD, kHeads, columns});
    ops::qsa_attention(Tensor(dq.p, DType::BF16, {kD, kHeads, columns}),
                       Tensor(diq.p, DType::BF16, {kDi, kIndexHeads, columns}), layer, batch, kGeometry, 1.0F / 16.0F,
                       max_context, workspace.p, workspace_bytes, out, nullptr, &append);
    synchronize("vq attention");
    const auto got = t::from_device_bf16(dout, static_cast<std::size_t>(kD) * kHeads * columns);
    // The stored state after the call: paged rows, and the window holding the call's exact rows.
    const auto kc_after  = download<std::uint8_t>(dk, kc.size());
    const auto vc_after  = download<std::uint8_t>(dv, vc.size());
    const auto ks_after  = download<std::uint16_t>(dks, ks.size());
    const auto vs_after  = download<std::uint16_t>(dvs, vs.size());
    const auto wk_after  = download<std::int8_t>(dwk, wk.size());
    const auto wv_after  = download<std::int8_t>(dwv, wv.size());
    const auto wks_after = download<std::uint16_t>(dwks, wks.size());
    const auto wvs_after = download<std::uint16_t>(dwvs, wvs.size());
    const auto wt_after  = download<std::int32_t>(dwtags, wtags.size());

    const std::string tag = std::string("qsa_attention ") + (q4 ? "k4v2 " : "vq2 ") + c.name;
    std::vector<double> ref(got.size());
    for (int r = 0; r < rows; ++r) {
        const int start = c.starts[r], keys = start + width;
        // A call key's exact rows must be in its committed slot, tagged with its stored codes.
        if (c.window) {
            for (int key = start; key < keys; ++key) {
                for (int head = 0; head < kKvHeads; ++head) {
                    const std::size_t prow = row_of(page_of(r, key), head, key);
                    const std::size_t srow = slot_row(r, head, window_slot(key));
                    const bool kt = static_cast<std::uint32_t>(wt_after[srow * 2]) ==
                                    window_tag(key, &kc_after[prow * kb], kb, ks_after[prow]);
                    const bool vt = static_cast<std::uint32_t>(wt_after[srow * 2 + 1]) ==
                                    window_tag(key, &vc_after[prow * vb], vb, vs_after[prow]);
                    expect(kt && vt, tag + ": the call's window rows are committed with their codes' tags");
                    if (!(kt && vt)) { return; }
                }
            }
        }
        const auto attended = attended_tokens(r, width, start, iq, pooled, [&](int token) { return page_of(r, token); });
        std::vector<double> kx(kD), vx(kD);
        for (int j = 0; j < width; ++j) {
            const int p = start + j;
            // This query's keys and values in the model's domain, [token][head][d].
            std::vector<double> k_all(static_cast<std::size_t>(keys) * kKvHeads * kD), v_all(k_all.size());
            for (int token = 0; token < keys; ++token) {
                if (attended[j][token] == 0) { continue; }
                for (int head = 0; head < kKvHeads; ++head) {
                    const std::size_t prow = row_of(page_of(r, token), head, token);
                    const bool exact_rule =
                        c.window && (token < infernix::kKVWindowSinkTokens || token >= p - infernix::kKVWindowRecentTokens);
                    const bool in_call = token >= start;
                    const bool exact =
                        exact_rule &&
                        (in_call || stale[(static_cast<std::size_t>(r) * kKvHeads + head) * context + token] == 0);
                    const std::size_t srow = slot_row(r, head, window_slot(token));
                    for (int d = 0; d < kD; ++d) {
                        if (exact) {
                            const auto& wkk = in_call ? wk_after : wk;
                            const auto& wvv = in_call ? wv_after : wv;
                            const auto& wks_ = in_call ? wks_after : wks;
                            const auto& wvs_ = in_call ? wvs_after : wvs;
                            kx[d] = wkk[srow * kD + d] * from_fp16(wks_[srow * 4 + d / 64]);
                            vx[d] = wvv[srow * kD + d] * from_fp16(wvs_[srow * 4 + d / 64]);
                        } else {
                            kx[d] = vq_code(&kc_after[prow * kb], q4, d) * from_fp16(ks_after[prow]);
                            vx[d] = vq_code(&vc_after[prow * vb], false, d) * from_fp16(vs_after[prow]);
                        }
                    }
                    hadamard(kx);
                    hadamard(vx);
                    std::copy(kx.begin(), kx.end(), k_all.begin() + (static_cast<std::ptrdiff_t>(token) * kKvHeads + head) * kD);
                    std::copy(vx.begin(), vx.end(), v_all.begin() + (static_cast<std::ptrdiff_t>(token) * kKvHeads + head) * kD);
                }
            }
            t::naive_dense_softmax_attention(
                ops::AttentionHeadGeometry{.head_dim = kD, .query_heads = kHeads, .kv_heads = kKvHeads}, 1, keys,
                1.0 / 16.0,
                [&](int d, int h, int) {
                    return from_bf16(q[((static_cast<std::size_t>(r) * width + j) * kHeads + h) * kD + d]);
                },
                [&](int d, int h, int token) {
                    return k_all[(static_cast<std::size_t>(token) * kKvHeads + h) * kD + d];
                },
                [&](int d, int h, int token) {
                    return v_all[(static_cast<std::size_t>(token) * kKvHeads + h) * kD + d];
                },
                [&](int, int token) { return attended[j][token] != 0; },
                [&](int d, int h, int, double v) {
                    ref[((static_cast<std::size_t>(r) * width + j) * kHeads + h) * kD + d] = v;
                });
        }
    }
    g_failures += t::verify_reduction(tag, got, ref, kAttentionCriterion);
}

void test_attention(KvCacheStorage storage, const AttentionCase& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const auto layout        = infernix::paged_kv_storage_layout(storage, kD);
    const bool rotated_keys   = storage != KvCacheStorage::BFloat16;
    const bool rotated_values = storage == KvCacheStorage::Nvfp4Group16 ||
                                storage == KvCacheStorage::Fp8KeyNvfp4Value;
    const int rows = static_cast<int>(c.starts.size()), width = c.width, columns = rows * width;
    int context = 0;
    for (const int s : c.starts) { context = std::max(context, s + width); }
    const int pages_per_row = (context + kPage - 1) / kPage, pages = rows * pages_per_row;
    const int max_context = pages_per_row * kPage;
    std::vector<std::int32_t> tables(pages);
    std::iota(tables.begin(), tables.end(), 0);
    std::shuffle(tables.begin(), tables.end(), rng);

    // Stored K/V in the storage's planes (rotated vectors in the normalized Hadamard domain).
    const StoredVector key_plane(layout.key, pages, rng), value_plane(layout.value, pages, rng);
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

    DeviceBuffer dk = upload(key_plane.data), dv = upload(value_plane.data);
    DeviceBuffer dks, dvs;
    if (layout.key.has_scale()) { dks = upload(key_plane.scales); }
    if (layout.value.has_scale()) { dvs = upload(value_plane.scales); }
    DeviceBuffer dpooled = upload(pooled), dq = upload(q), diq = upload(iq),
                 dtables = upload(tables);
    DeviceBuffer dpos = upload(positions), drows = upload(table_rows), dslots = upload(tail_slots);
    ops::QsaKVLayer layer;
    layer.kv.storage      = storage;
    layer.kv.head_dim     = kD;
    layer.kv.num_kv_heads = kKvHeads;
    const auto plane      = [&](const DeviceBuffer& b, DType dtype, int leading) {
        return Tensor(b.p, dtype, {leading, kPage, kKvHeads, pages});
    };
    layer.kv.k_pages = plane(dk, layout.key.data_dtype, layout.key.data_leading_extent);
    layer.kv.v_pages = plane(dv, layout.value.data_dtype, layout.value.data_leading_extent);
    if (layout.key.has_scale()) {
        layer.kv.k_scale_pages = plane(dks, layout.key.scale_dtype, layout.key.scale_leading_extent);
    }
    if (layout.value.has_scale()) {
        layer.kv.v_scale_pages = plane(dvs, layout.value.scale_dtype, layout.value.scale_leading_extent);
    }
    layer.pooled_pages = Tensor(dpooled.p, DType::BF16, {kSlotWidth, kPage, 1, pages});
    const ops::QsaBatch batch{.block_tables = Tensor(dtables.p, DType::I32, {pages_per_row, rows}),
                              .table_rows   = Tensor(drows.p, DType::I32, {rows}),
                              .positions    = Tensor(dpos.p, DType::I32, {columns}),
                              .tail_slots   = Tensor(dslots.p, DType::I32, {rows}),
                              .batch        = rows,
                              .width        = width};
    const std::size_t workspace_bytes =
        ops::qsa_attention_workspace_bytes(kGeometry, columns, max_context, storage);
    DeviceBuffer workspace(workspace_bytes),
        dout(static_cast<std::size_t>(kD) * kHeads * columns * 2);
    Tensor out(dout.p, DType::BF16, {kD, kHeads, columns});
    ops::qsa_attention(Tensor(dq.p, DType::BF16, {kD, kHeads, columns}),
                       Tensor(diq.p, DType::BF16, {kDi, kIndexHeads, columns}), layer, batch,
                       kGeometry, 1.0F / 16.0F, max_context, workspace.p, workspace_bytes, out,
                       nullptr);
    synchronize("attention");
    const auto got = t::from_device_bf16(dout, static_cast<std::size_t>(kD) * kHeads * columns);
    const std::string tag = std::string("qsa_attention ") + storage_name(storage) + " " + c.name;
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
        expect(same, tag + ": a 100-column call equals the same columns of the wider call bit for bit");
    }
    // Width invariance of decode and verification calls (at most 8 columns): the last row alone gives
    // the bits it gives beside the other rows, so concurrent sequences decode as they do alone (the
    // preemption and prefix-cache resumes rely on it).
    if (rows > 1 && columns <= 8) {
        const int r     = rows - 1;
        const int first = r * width;
        const ops::QsaBatch sub{.block_tables = Tensor(dtables.p, DType::I32, {pages_per_row, rows}),
                                .table_rows   = Tensor(static_cast<std::int32_t*>(drows.p) + r, DType::I32, {1}),
                                .positions = Tensor(static_cast<std::int32_t*>(dpos.p) + first, DType::I32, {width}),
                                .tail_slots = Tensor(static_cast<std::int32_t*>(dslots.p) + r, DType::I32, {1}),
                                .batch      = 1,
                                .width      = width};
        DeviceBuffer dsub(static_cast<std::size_t>(kD) * kHeads * width * 2);
        Tensor sub_out(dsub.p, DType::BF16, {kD, kHeads, width});
        ops::qsa_attention(Tensor(static_cast<std::uint16_t*>(dq.p) + static_cast<std::size_t>(first) * kD * kHeads,
                                  DType::BF16, {kD, kHeads, width}),
                           Tensor(static_cast<std::uint16_t*>(diq.p) +
                                      static_cast<std::size_t>(first) * kDi * kIndexHeads,
                                  DType::BF16, {kDi, kIndexHeads, width}),
                           layer, sub, kGeometry, 1.0F / 16.0F, max_context, workspace.p, workspace_bytes, sub_out,
                           nullptr);
        synchronize("attention single row");
        const auto part = t::from_device_bf16(dsub, static_cast<std::size_t>(kD) * kHeads * width);
        bool same = true;
        for (std::size_t i = 0; i < part.size() && same; ++i) {
            same = part[i] == got[static_cast<std::size_t>(first) * kD * kHeads + i];
        }
        expect(same, tag + ": the last row alone equals its columns of the batched call bit for bit");
    }

    // Page spaces (design §19.3.11): the same KV with every page whose id is 1 mod 3 in a mapped host
    // plane and every page 2 mod 3 in a separate device ("lent") plane, their pool copies zeroed so
    // only the translated reads can find them; selection and attention must give the same bits.
    {
        const std::size_t k_page = key_plane.data_page_bytes(), v_page = value_plane.data_page_bytes();
        const std::size_t ks_page = key_plane.scale_page_bytes(), vs_page = value_plane.scale_page_bytes();
        const std::size_t pool_page = static_cast<std::size_t>(kSlotWidth) * kPage * 2;
        std::vector<int> host_of(pages, -1), lent_of(pages, -1);
        int host_pages = 0, lent_pages = 0;
        for (int p = 0; p < pages; ++p) {
            if (p % 3 == 1) { host_of[p] = host_pages++; }
            if (p % 3 == 2) { lent_of[p] = lent_pages++; }
        }
        const auto planes = [&](std::size_t page_bytes, int count) {
            return std::max<std::size_t>(page_bytes * std::max(count, 1), 16);
        };
        void* host_mem = nullptr;
        const std::size_t host_bytes = planes(k_page, host_pages) + planes(v_page, host_pages) +
                                       planes(ks_page, host_pages) + planes(vs_page, host_pages);
        expect(cudaHostAlloc(&host_mem, host_bytes, cudaHostAllocMapped) == cudaSuccess, "host planes allocate");
        auto* host_k  = static_cast<std::byte*>(host_mem);
        auto* host_v  = host_k + planes(k_page, host_pages);
        auto* host_ks = host_v + planes(v_page, host_pages);
        auto* host_vs = host_ks + planes(ks_page, host_pages);
        DeviceBuffer lent_k(planes(k_page, lent_pages)), lent_v(planes(v_page, lent_pages));
        DeviceBuffer lent_ks(planes(ks_page, lent_pages)), lent_vs(planes(vs_page, lent_pages));
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
            move(dk, k_page, p, host_k, lent_k);
            move(dv, v_page, p, host_v, lent_v);
            if (layout.key.has_scale()) { move(dks, ks_page, p, host_ks, lent_ks); }
            if (layout.value.has_scale()) { move(dvs, vs_page, p, host_vs, lent_vs); }
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
                                           .host_k_scale = layout.key.has_scale() ? host_ks : nullptr,
                                           .host_v_scale = layout.value.has_scale() ? host_vs : nullptr,
                                           .lent_k       = lent_k.p,
                                           .lent_v       = lent_v.p,
                                           .lent_k_scale = layout.key.has_scale() ? lent_ks.p : nullptr,
                                           .lent_v_scale = layout.value.has_scale() ? lent_vs.p : nullptr,
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
        expect(spaced_got == got, tag + ": pages in host and lent spaces give the pool's bits");
        cudaFreeHost(host_mem);
    }
    std::vector<double> ref(got.size());
    for (int r = 0; r < rows; ++r) {
        const int keys     = c.starts[r] + width;
        const auto page_of = [&](int token) {
            return tables[static_cast<std::size_t>(r) * pages_per_row + token / kPage];
        };
        // Decoded keys and values of the row in the model's domain, [head][token][d]: a rotated
        // vector is decoded, then rotated back (the normalized transform is its own inverse).
        std::vector<double> key(static_cast<std::size_t>(kKvHeads) * keys * kD), value(key.size());
        t::parallel_ranges(
            static_cast<std::int64_t>(kKvHeads) * keys, t::threads_for_rows(kKvHeads * keys),
            [&](std::int64_t begin, std::int64_t end) {
                std::vector<double> k(kD), v(kD);
                for (std::int64_t i = begin; i < end; ++i) {
                    const int head = static_cast<int>(i / keys), token = static_cast<int>(i % keys);
                    const int page = page_of(token);
                    for (int d = 0; d < kD; ++d) {
                        k[d] = key_plane.decode(page, head, token, d);
                        v[d] = value_plane.decode(page, head, token, d);
                    }
                    if (rotated_keys) { hadamard(k); }
                    if (rotated_values) { hadamard(v); }
                    std::copy(k.begin(), k.end(), key.begin() + static_cast<std::ptrdiff_t>(i) * kD);
                    std::copy(v.begin(), v.end(), value.begin() + static_cast<std::ptrdiff_t>(i) * kD);
                }
            });
        const auto attended = attended_tokens(r, width, c.starts[r], iq, pooled, page_of);
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

// ----------------------------------------------------------------------------------- prepare

// Q, the index query, BF16-stored keys and pooled keys: one BF16 rounding of an FP32 result (the
// fused prepare keeps FP32 through norm and rotation). An RNE rounding errs by at most 2^-8 of the
// element (3.9e-3 of the largest), and uniform rounding errors give a relative L2 of at most
// 2^-8 / sqrt(3) = 2.3e-3; FP32 normalization adds ~1e-6, and FP32 angles at positions below 2^13 err
// by at most ~3e-4 rad (one ulp of position x frequency), at most ~1e-3 of the largest element.
constexpr t::ReductionCriterion kPrepareCriterion{
    /*relative_l2*/ 3.0e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 5.5e-3,
};

// Quantized stored keys, decoded exactly and compared in the codec's (Hadamard-rotated) domain with
// the FP64 key, add the codec's rounding to that FP32 noise (1e-3 of the largest element):
// - INT8-G64: |error| <= s / 2 with s = FP16(group max / 127), at most 3.95e-3 of the largest
//   element; uniform errors s / sqrt(12) against groups whose max is at most 3.5 RMS give a relative
//   L2 of at most 3.5 / (127 sqrt(12)) = 8.0e-3.
// - FP8 E4M3 rows (fp8, k8v4 keys): three fraction bits, |error| <= 2^-4 |x| (6.25e-2 of the largest);
//   uniform errors within a binade give a relative L2 of at most 2^-3 / sqrt(12) = 3.6e-2.
// - NVFP4-G16: E2M1 levels times the group scale (max / target, target 4..6, E4M3-rounded): half the
//   level spacing is at most a quarter of the element's magnitude or of the scale, so |error| <=
//   max / 4 (1 + 2^-4) and uniform errors give a relative L2 of at most 1 / (4 sqrt(3)) = 0.144.
constexpr t::ReductionCriterion kStoredInt8Criterion{8.5e-3, 1.0e-3, 5.0e-3};
constexpr t::ReductionCriterion kStoredFp8Criterion{3.7e-2, 1.0e-3, 6.4e-2};
constexpr t::ReductionCriterion kStoredNvfp4Criterion{1.45e-1, 1.0e-3, 2.7e-1};

const t::ReductionCriterion& stored_key_criterion(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::Int8Group64: return kStoredInt8Criterion;
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Fp8KeyNvfp4Value: return kStoredFp8Criterion;
    case KvCacheStorage::Nvfp4Group16: return kStoredNvfp4Criterion;
    default: return kPrepareCriterion;
    }
}

// Element d of a stored key row (row: its code bytes, scales: its scale bytes), decoded exactly.
double stored_key(KvCacheStorage storage, const std::uint8_t* row, const std::uint8_t* scales, int d) {
    std::uint16_t bits = 0;
    switch (storage) {
    case KvCacheStorage::BFloat16: std::memcpy(&bits, row + 2 * d, 2); return from_bf16(bits);
    case KvCacheStorage::Int8Group64:
        std::memcpy(&bits, scales + 2 * (d / 64), 2);
        return static_cast<std::int8_t>(row[d]) * from_fp16(bits);
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Fp8KeyNvfp4Value: std::memcpy(&bits, scales, 2); return from_e4m3(row[d]) * from_fp16(bits);
    default: return from_e2m1((row[d / 2] >> (4 * (d % 2))) & 0xFU) * from_e4m3(scales[d / 16]);
    }
}

// The projection rows of one (sequence, position), deterministic in the seed: harnesses built from
// one seed see the same inputs whatever calls they split them into.
enum class Rows : std::uint32_t { Query, Key, Value, IndexQuery, IndexKey };

std::vector<std::uint16_t> projection_rows(std::uint32_t seed, Rows kind, int sequence, int position) {
    static constexpr std::size_t kCount[] = {std::size_t{kD} * kHeads, std::size_t{kD} * kKvHeads,
                                             std::size_t{kD} * kKvHeads, std::size_t{kDi} * kIndexHeads,
                                             std::size_t{kDi}};
    static constexpr double kScale[] = {2.0, 2.0, 1.0, 2.0, 1.0};
    const auto k = static_cast<std::uint32_t>(kind);
    std::seed_seq words{seed, k, static_cast<std::uint32_t>(sequence), static_cast<std::uint32_t>(position)};
    std::mt19937 rng(words);
    return random_bf16(rng, kCount[k], kScale[k]);
}

// Position q's RoPE position in a sequence with one image (rope_map's assignment).
Rope rope_at(int q, const Image& image) {
    const int end = image.start + image.rows * image.cols;
    if (image.rows == 0 || q < image.start) { return {q, q, q}; }
    if (q < end) {
        const int k = q - image.start;
        return {image.start, image.start + k / image.cols, image.start + k % image.cols};
    }
    const int v = image.start + std::max(image.rows, image.cols) + (q - end);
    return {v, v, v};
}

// RMSNorm with the unit-offset weight over the whole row, in FP64.
std::vector<double> normalize_row(const std::vector<double>& x, const std::vector<std::uint16_t>& weight) {
    double squares = 0.0;
    for (const double v : x) { squares += v * v; }
    const double inv = 1.0 / std::sqrt(squares / static_cast<double>(x.size()) + static_cast<double>(kGeometry.eps));
    std::vector<double> out(x.size());
    for (std::size_t d = 0; d < x.size(); ++d) { out[d] = x[d] * inv * (1.0 + from_bf16(weight[d])); }
    return out;
}

// RoPE(norm(row)) of one head of `dim` BF16 values.
std::vector<double> head_oracle(const std::uint16_t* row, int dim, const std::vector<std::uint16_t>& weight,
                                const Rope& rope) {
    std::vector<double> x(dim);
    for (int d = 0; d < dim; ++d) { x[d] = from_bf16(row[d]); }
    auto out = normalize_row(x, weight);
    rotate(out, rope);
    return out;
}

std::vector<std::uint8_t> bytes_of(const DeviceBuffer& buffer) {
    return buffer.p == nullptr ? std::vector<std::uint8_t>{} : download<std::uint8_t>(buffer, buffer.bytes);
}

template <class T>
void put(DeviceBuffer& buffer, const std::vector<T>& host) {
    if (buffer.p == nullptr) {
        buffer = upload(host);
    } else {
        buffer.copy_from_host(host.data(), host.size() * sizeof(T));
    }
}

enum class Launch { Eager, InPlace, KvOnly, Graph };

// The device inputs and outputs of one call; a Graph call keeps them and its graph, so a later call
// of the same batch and width uploads new rows, positions and table rows and replays it.
struct PrepareCall {
    DeviceBuffer q, k, v, iq, ik, positions, rows, slots, rope, block_rope, q_out, iq_out;
    cudaStream_t stream  = nullptr;
    cudaGraphExec_t exec = nullptr;

    PrepareCall() = default;
    PrepareCall(const PrepareCall&)            = delete;
    PrepareCall& operator=(const PrepareCall&) = delete;
    ~PrepareCall() {
        if (exec != nullptr) { (void)cudaGraphExecDestroy(exec); }
        if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
    }
};

// The KV planes, pooled keys and tails of `first.size()` sequences and the qsa_prepare calls over
// them. Sequence r's positions before first[r] are in the state (its latest block's raw keys in the
// tails); its RoPE positions hold one image. Two harnesses of one seed start from the same bytes.
// With `spaces`, every page wholly before a sequence's first position moves to a host or lent space
// (alternately); the call's pages stay in the pool.
class PrepareHarness {
public:
    PrepareHarness(KvCacheStorage storage, std::vector<int> first, std::vector<Image> images, int extent,
                   std::uint32_t seed, bool spaces = false)
        : storage_(storage), layout_(infernix::paged_kv_storage_layout(storage, kD)), first_(std::move(first)),
          images_(std::move(images)), seed_(seed), rows_(static_cast<int>(first_.size())) {
        std::mt19937 rng(seed);
        pages_per_row_ = (extent + kPage - 1) / kPage;
        pages_         = rows_ * pages_per_row_;
        tables_host_.resize(static_cast<std::size_t>(pages_));
        std::iota(tables_host_.begin(), tables_host_.end(), 0);
        std::shuffle(tables_host_.begin(), tables_host_.end(), rng);
        wq_  = random_bf16(rng, kD, 0.3);
        wk_  = random_bf16(rng, kD, 0.3);
        wiq_ = random_bf16(rng, kDi, 0.3);
        wik_ = random_bf16(rng, kDi, 0.3);
        dwq_  = upload(wq_);
        dwk_  = upload(wk_);
        dwiq_ = upload(wiq_);
        dwik_ = upload(wik_);
        // Random bytes in every plane, so an unwritten row's bytes are recognizable.
        std::uniform_int_distribution<int> byte(0, 255);
        const auto random_plane = [&](int row_bytes) {
            std::vector<std::uint8_t> v(static_cast<std::size_t>(row_bytes) * kPage * kKvHeads * pages_);
            for (auto& x : v) { x = static_cast<std::uint8_t>(byte(rng)); }
            return v;
        };
        k0_  = random_plane(key_bytes());
        v0_  = random_plane(value_bytes());
        ks0_ = random_plane(key_scale_bytes());
        vs0_ = random_plane(value_scale_bytes());
        for (auto [device, reference, host] : {std::tuple{&dk_, &rk_, &k0_}, std::tuple{&dv_, &rv_, &v0_},
                                               std::tuple{&dks_, &rks_, &ks0_}, std::tuple{&dvs_, &rvs_, &vs0_}}) {
            if (!host->empty()) {
                *device    = upload(*host);
                *reference = upload(*host);
            }
        }
        dpooled_ = upload(std::vector<std::uint16_t>(static_cast<std::size_t>(kSlotWidth) * kPage * pages_, kCanary));
        // Tails: random, except each sequence's latest block before its first position.
        tails_host_ = random_bf16(rng, static_cast<std::size_t>(kDi) * (kR - 1) * rows_, 1.0);
        tail_last_.assign(rows_, -1);
        for (int r = 0; r < rows_; ++r) {
            if (first_[r] == 0) { continue; }
            const int last = first_[r] - 1, base = last - last % kR;
            for (int q = base; q <= std::min(last, base + kR - 2); ++q) {
                const auto raw = projection_rows(seed_, Rows::IndexKey, r, q);
                std::copy(raw.begin(), raw.end(), tails_host_.begin() + tail_index(slot_of(r), q % kR));
            }
            tail_last_[r] = last;
        }
        dtails_ = upload(tails_host_);
        if (spaces) { move_to_spaces(); }
        dtables_ = upload(tables_host_);
    }

    PrepareHarness(const PrepareHarness&)            = delete;
    PrepareHarness& operator=(const PrepareHarness&) = delete;
    ~PrepareHarness() {
        for (void* p : host_planes_) { (void)cudaFreeHost(p); }
    }

    // One qsa_prepare call: `width` columns of each listed sequence from starts[i].
    void call(const std::vector<int>& rows, const std::vector<int>& starts, int width, bool update_tails,
              Launch mode = Launch::Eager, PrepareCall* persistent = nullptr) {
        PrepareCall local;
        PrepareCall& b    = persistent != nullptr ? *persistent : local;
        const int batch   = static_cast<int>(rows.size());
        const int columns = batch * width;
        std::vector<std::uint16_t> q, k, v, iq, ik;
        std::vector<std::int32_t> positions(columns), table_rows(batch), slots(batch), rope(3 * std::size_t(columns)),
            block_rope(3 * std::size_t(batch));
        const auto append = [](std::vector<std::uint16_t>& to, const std::vector<std::uint16_t>& from) {
            to.insert(to.end(), from.begin(), from.end());
        };
        for (int i = 0; i < batch; ++i) {
            const int r   = rows[i];
            table_rows[i] = r;
            slots[i]      = slot_of(r);
            const Rope block_start = rope_at(starts[i] / kR * kR, images_[r]);
            for (int a = 0; a < 3; ++a) { block_rope[static_cast<std::size_t>(a) * batch + i] = block_start[a]; }
            for (int j = 0; j < width; ++j) {
                const int column = i * width + j, p = starts[i] + j;
                positions[column] = p;
                const Rope at     = rope_at(p, images_[r]);
                for (int a = 0; a < 3; ++a) { rope[static_cast<std::size_t>(a) * columns + column] = at[a]; }
                append(q, projection_rows(seed_, Rows::Query, r, p));
                append(k, projection_rows(seed_, Rows::Key, r, p));
                append(v, projection_rows(seed_, Rows::Value, r, p));
                append(iq, projection_rows(seed_, Rows::IndexQuery, r, p));
                append(ik, projection_rows(seed_, Rows::IndexKey, r, p));
            }
        }
        put(b.q, q);
        put(b.k, k);
        put(b.v, v);
        put(b.iq, iq);
        put(b.ik, ik);
        put(b.positions, positions);
        put(b.rows, table_rows);
        put(b.slots, slots);
        put(b.rope, rope);
        put(b.block_rope, block_rope);
        if (b.q_out.p == nullptr) {
            b.q_out  = DeviceBuffer(q.size() * 2);
            b.iq_out = DeviceBuffer(iq.size() * 2);
        }
        Tensor tq(b.q.p, DType::BF16, {kD, kHeads, columns});
        Tensor tiq(b.iq.p, DType::BF16, {kDi, kIndexHeads, columns});
        const ops::QsaPrepareInputs inputs{.q       = tq,
                                           .k       = Tensor(b.k.p, DType::BF16, {kD, kKvHeads, columns}),
                                           .v       = Tensor(b.v.p, DType::BF16, {kD, kKvHeads, columns}),
                                           .index_q = tiq,
                                           .index_k = Tensor(b.ik.p, DType::BF16, {kDi, columns})};
        const ops::QsaPrepareNorms norms{.query       = Tensor(dwq_.p, DType::BF16, {kD}),
                                         .key         = Tensor(dwk_.p, DType::BF16, {kD}),
                                         .index_query = Tensor(dwiq_.p, DType::BF16, {kDi}),
                                         .index_key   = Tensor(dwik_.p, DType::BF16, {kDi})};
        Tensor tails(dtails_.p, DType::BF16, {kDi, kR - 1, rows_});
        const ops::QsaBatch qsa_batch{.block_tables     = Tensor(dtables_.p, DType::I32, {pages_per_row_, rows_}),
                                      .table_rows       = Tensor(b.rows.p, DType::I32, {batch}),
                                      .positions        = Tensor(b.positions.p, DType::I32, {columns}),
                                      .tail_slots       = Tensor(b.slots.p, DType::I32, {batch}),
                                      .rope_positions   = Tensor(b.rope.p, DType::I32, {columns, 3}),
                                      .block_start_rope = Tensor(b.block_rope.p, DType::I32, {batch, 3}),
                                      .batch            = batch,
                                      .width            = width,
                                      .update_tails     = update_tails};
        Tensor q_out  = mode == Launch::KvOnly    ? Tensor{}
                        : mode == Launch::InPlace ? tq
                                                  : Tensor(b.q_out.p, DType::BF16, {kD, kHeads, columns});
        Tensor iq_out = mode == Launch::KvOnly    ? Tensor{}
                        : mode == Launch::InPlace ? tiq
                                                  : Tensor(b.iq_out.p, DType::BF16, {kDi, kIndexHeads, columns});
        const ops::QsaKVLayer kv_layer = layer();
        if (mode == Launch::Graph) {
            if (b.exec == nullptr) {
                t::cuda_check(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking), "prepare stream");
                cudaGraph_t graph = nullptr;
                t::cuda_check(cudaStreamBeginCapture(b.stream, cudaStreamCaptureModeThreadLocal), "prepare capture");
                ops::qsa_prepare(inputs, norms, tails, kv_layer, qsa_batch, g_index_geometry, q_out, iq_out, b.stream);
                t::cuda_check(cudaStreamEndCapture(b.stream, &graph), "prepare end capture");
                t::cuda_check(cudaGraphInstantiate(&b.exec, graph, 0), "prepare graph instantiate");
                t::cuda_check(cudaGraphDestroy(graph), "prepare graph destroy");
            }
            t::cuda_check(cudaGraphLaunch(b.exec, b.stream), "prepare graph launch");
            t::cuda_check(cudaStreamSynchronize(b.stream), "prepare graph replay");
        } else {
            ops::qsa_prepare(inputs, norms, tails, kv_layer, qsa_batch, g_index_geometry, q_out, iq_out, nullptr);
        }
        synchronize("prepare");
        if (mode != Launch::KvOnly) {
            const auto qs  = download<std::uint16_t>(mode == Launch::InPlace ? b.q : b.q_out, q.size());
            const auto iqs = download<std::uint16_t>(mode == Launch::InPlace ? b.iq : b.iq_out, iq.size());
            for (int i = 0; i < batch; ++i) {
                for (int j = 0; j < width; ++j) {
                    const std::size_t column = static_cast<std::size_t>(i) * width + j;
                    const auto key           = std::pair{rows[i], starts[i] + j};
                    q_[key].assign(qs.begin() + column * kHeads * kD, qs.begin() + (column + 1) * kHeads * kD);
                    iq_[key].assign(iqs.begin() + column * kIndexHeads * kDi,
                                    iqs.begin() + (column + 1) * kIndexHeads * kDi);
                }
            }
        }
        // The same K/V through kv_cache_append_batch into the reference planes (the values must match).
        infernix::PagedKVBatchLayerView reference;
        reference.k_pages      = plane_tensor(rk_, layout_.key, false);
        reference.v_pages      = plane_tensor(rv_, layout_.value, false);
        reference.k_scale_pages = plane_tensor(rks_, layout_.key, true);
        reference.v_scale_pages = plane_tensor(rvs_, layout_.value, true);
        reference.block_tables = Tensor(dtables_.p, DType::I32, {pages_per_row_, rows_});
        reference.head_dim     = kD;
        reference.num_kv_heads = kKvHeads;
        reference.storage      = storage_;
        ops::kv_cache_append_batch(Tensor(b.k.p, DType::BF16, {kD, kKvHeads, width, batch}),
                                   Tensor(b.v.p, DType::BF16, {kD, kKvHeads, width, batch}),
                                   Tensor(b.positions.p, DType::I32, {width, batch}), Tensor(b.rows.p, DType::I32, {batch}),
                                   reference, nullptr);
        synchronize("reference append");
        for (int i = 0; i < batch; ++i) {
            for (int j = 0; j < width; ++j) { written_.insert({rows[i], starts[i] + j}); }
            if (update_tails) { tail_last_[rows[i]] = starts[i] + width - 1; }
        }
    }

    // Every recorded column and stored row against the oracles, and every other byte untouched.
    void check(const std::string& tag) const {
        // Queries and index queries.
        std::vector<double> got, ref, got_index, ref_index;
        for (const auto& [key, rows] : q_) {
            const auto [r, p] = key;
            const Rope at     = rope_at(p, images_[r]);
            const auto in     = projection_rows(seed_, Rows::Query, r, p);
            for (int h = 0; h < kHeads; ++h) {
                const auto oracle = head_oracle(in.data() + static_cast<std::size_t>(h) * kD, kD, wq_, at);
                ref.insert(ref.end(), oracle.begin(), oracle.end());
                for (int d = 0; d < kD; ++d) { got.push_back(from_bf16(rows[static_cast<std::size_t>(h) * kD + d])); }
            }
            const auto index_in = projection_rows(seed_, Rows::IndexQuery, r, p);
            const auto& index   = iq_.at(key);
            for (int h = 0; h < kIndexHeads; ++h) {
                const auto oracle = head_oracle(index_in.data() + static_cast<std::size_t>(h) * kDi, kDi, wiq_, at);
                ref_index.insert(ref_index.end(), oracle.begin(), oracle.end());
                for (int d = 0; d < kDi; ++d) {
                    got_index.push_back(from_bf16(index[static_cast<std::size_t>(h) * kDi + d]));
                }
            }
        }
        if (!got.empty()) {
            g_failures += t::verify_reduction(tag + " q", got, ref, kPrepareCriterion);
            g_failures += t::verify_reduction(tag + " index q", got_index, ref_index, kPrepareCriterion);
        }
        // Stored keys (decoded) against the FP64 key, in the codec's domain; unwritten rows untouched.
        const auto kd = bytes_of(dk_), ks = bytes_of(dks_), vd = bytes_of(dv_), vs = bytes_of(dvs_);
        const int kb = key_bytes(), ksb = key_scale_bytes();
        std::vector<char> written_row(static_cast<std::size_t>(kPage) * kKvHeads * pages_, 0);
        std::vector<double> got_key, ref_key;
        for (const auto& [r, p] : written_) {
            const int page  = page_of(r, p);
            const auto in   = projection_rows(seed_, Rows::Key, r, p);
            const Rope at   = rope_at(p, images_[r]);
            for (int h = 0; h < kKvHeads; ++h) {
                const std::size_t row = (static_cast<std::size_t>(page) * kKvHeads + h) * kPage + p % kPage;
                written_row[row]      = 1;
                auto oracle           = head_oracle(in.data() + static_cast<std::size_t>(h) * kD, kD, wk_, at);
                if (storage_ != KvCacheStorage::BFloat16) { hadamard(oracle); }
                ref_key.insert(ref_key.end(), oracle.begin(), oracle.end());
                for (int d = 0; d < kD; ++d) {
                    got_key.push_back(stored_key(storage_, kd.data() + row * kb,
                                                 ksb > 0 ? ks.data() + row * ksb : nullptr, d));
                }
            }
        }
        if (!got_key.empty()) {
            g_failures += t::verify_reduction(tag + " stored keys", got_key, ref_key, stored_key_criterion(storage_));
        }
        bool untouched = true;
        for (std::size_t row = 0; row < written_row.size(); ++row) {
            if (written_row[row] != 0) { continue; }
            untouched &= std::memcmp(kd.data() + row * kb, k0_.data() + row * kb, kb) == 0;
            if (ksb > 0) { untouched &= std::memcmp(ks.data() + row * ksb, ks0_.data() + row * ksb, ksb) == 0; }
        }
        expect(untouched, tag + ": key rows of no call's position are untouched");
        expect(vd == bytes_of(rv_) && vs == bytes_of(rvs_),
               tag + ": values (and every unwritten value byte) equal kv_cache_append_batch's");
        // Pooled keys: every block a call completed against the oracle, every other slot untouched.
        const auto pooled = download<std::uint16_t>(dpooled_, static_cast<std::size_t>(kSlotWidth) * kPage * pages_);
        std::vector<double> got_pooled, ref_pooled;
        bool canaries = true;
        for (int r = 0; r < rows_; ++r) {
            for (int block = 0; block < pages_per_row_ * kPage / kR; ++block) {
                const int page = page_of(r, kR * block);
                if (page >= pages_) { continue; } // a moved page: never written
                std::vector<std::uint16_t> key(kDi);
                for (int d = 0; d < kDi; ++d) {
                    const int token = kR * block + d / kSlotWidth;
                    key[d] = pooled[(static_cast<std::size_t>(page) * kPage + token % kPage) * kSlotWidth + d % kSlotWidth];
                }
                if (written_.count({r, kR * block + kR - 1}) != 0) {
                    const auto oracle = pooled_at(r, block);
                    ref_pooled.insert(ref_pooled.end(), oracle.begin(), oracle.end());
                    for (const auto bits : key) { got_pooled.push_back(from_bf16(bits)); }
                } else {
                    canaries &= std::all_of(key.begin(), key.end(), [](std::uint16_t x) { return x == kCanary; });
                }
            }
        }
        if (!got_pooled.empty()) {
            g_failures += t::verify_reduction(tag + " pooled keys", got_pooled, ref_pooled, kPrepareCriterion);
        }
        expect(canaries, tag + ": pooled slots of blocks no call completed are untouched");
        // Tails: the raw keys of the latest block up to each sequence's last tail-updating position.
        const auto tails = download<std::uint16_t>(dtails_, tails_host_.size());
        for (int r = 0; r < rows_; ++r) {
            if (tail_last_[r] < 0) { continue; }
            const int last = tail_last_[r], base = last - last % kR;
            bool ok = true;
            for (int q = base; q <= std::min(last, base + kR - 2); ++q) {
                const auto raw = projection_rows(seed_, Rows::IndexKey, r, q);
                ok &= std::memcmp(&tails[tail_index(slot_of(r), q % kR)], raw.data(), kDi * 2) == 0;
            }
            expect(ok, tag + " row " + std::to_string(r) + ": tails hold the raw keys of the latest block");
        }
        // Page spaces: nothing outside the pool is written.
        if (!host_planes_.empty()) {
            bool clean = true;
            for (std::size_t i = 0; i < host_planes_.size(); ++i) {
                const auto* p = static_cast<const std::uint8_t*>(host_planes_[i]);
                clean &= std::all_of(p, p + host_bytes_[i], [](std::uint8_t x) { return x == kSpaceByte; });
            }
            for (const DeviceBuffer* d : {&lent_k_, &lent_v_, &lent_ks_, &lent_vs_, &spaces_pooled_}) {
                const auto bytes = bytes_of(*d);
                clean &= std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t x) { return x == kSpaceByte; });
            }
            expect(clean, tag + ": host and lent pages are not written");
        }
    }

    // Outputs of the same columns, bit for bit (both harnesses recorded them).
    bool same_outputs(const PrepareHarness& other) const { return q_ == other.q_ && iq_ == other.iq_; }

    // Every pool-plane and pooled-key byte, bit for bit, and every tail byte with `tails` (calls that
    // end at different positions leave different stale slots; check() checks the defined ones).
    bool same_state(const PrepareHarness& other, bool tails = true) const {
        return bytes_of(dk_) == bytes_of(other.dk_) && bytes_of(dv_) == bytes_of(other.dv_) &&
               bytes_of(dks_) == bytes_of(other.dks_) && bytes_of(dvs_) == bytes_of(other.dvs_) &&
               bytes_of(dpooled_) == bytes_of(other.dpooled_) &&
               (!tails || bytes_of(dtails_) == bytes_of(other.dtails_));
    }

private:
    static constexpr std::uint8_t kSpaceByte = 0x5A;

    int key_bytes() const { return row_bytes(layout_.key, false); }
    int value_bytes() const { return row_bytes(layout_.value, false); }
    int key_scale_bytes() const { return row_bytes(layout_.key, true); }
    int value_scale_bytes() const { return row_bytes(layout_.value, true); }
    static int row_bytes(const infernix::PagedKVVectorLayout& v, bool scale) {
        return scale ? v.scale_leading_extent * static_cast<int>(infernix::dtype_size(v.scale_dtype))
                     : v.data_leading_extent * static_cast<int>(infernix::dtype_size(v.data_dtype));
    }

    Tensor plane_tensor(const DeviceBuffer& buffer, const infernix::PagedKVVectorLayout& v, bool scale) const {
        if (buffer.p == nullptr) { return Tensor{}; }
        return Tensor(buffer.p, scale ? v.scale_dtype : v.data_dtype,
                      {scale ? v.scale_leading_extent : v.data_leading_extent, kPage, kKvHeads, pages_});
    }

    ops::QsaKVLayer layer() const {
        ops::QsaKVLayer out;
        out.kv.storage       = storage_;
        out.kv.head_dim      = kD;
        out.kv.num_kv_heads  = kKvHeads;
        out.kv.k_pages       = plane_tensor(dk_, layout_.key, false);
        out.kv.v_pages       = plane_tensor(dv_, layout_.value, false);
        out.kv.k_scale_pages = plane_tensor(dks_, layout_.key, true);
        out.kv.v_scale_pages = plane_tensor(dvs_, layout_.value, true);
        out.pooled_pages     = Tensor(dpooled_.p, DType::BF16, {kSlotWidth, kPage, 1, pages_});
        out.spaces           = spaces_;
        return out;
    }

    int slot_of(int r) const { return rows_ - 1 - r; } // not the identity
    int page_of(int r, int position) const {
        return tables_host_[static_cast<std::size_t>(r) * pages_per_row_ + position / kPage];
    }
    std::size_t tail_index(int slot, int j) const { return (static_cast<std::size_t>(slot) * (kR - 1) + j) * kDi; }

    // FP64 RoPE(norm(bf16(mean))) of sequence r's block, at its first token's RoPE position.
    std::vector<double> pooled_at(int r, int block) const {
        std::vector<double> mean(kDi, 0.0);
        for (int j = 0; j < kR; ++j) {
            const auto raw = projection_rows(seed_, Rows::IndexKey, r, kR * block + j);
            for (int d = 0; d < kDi; ++d) { mean[d] += from_bf16(raw[d]); }
        }
        for (auto& m : mean) { m = from_bf16(to_bf16(m / kR)); }
        auto out = normalize_row(mean, wik_);
        rotate(out, rope_at(kR * block, images_[r]));
        return out;
    }

    void move_to_spaces() {
        std::vector<std::pair<std::size_t, bool>> moves; // (table entry, to the host space)
        for (int r = 0; r < rows_; ++r) {
            for (int lp = 0; (lp + 1) * kPage <= first_[r]; ++lp) {
                moves.emplace_back(static_cast<std::size_t>(r) * pages_per_row_ + lp, moves.size() % 2 == 0);
            }
        }
        int host_pages = 0, lent_pages = 0;
        for (const auto& move : moves) { (move.second ? host_pages : lent_pages) += 1; }
        int h = 0, l = 0;
        for (const auto& [entry, to_host] : moves) {
            tables_host_[entry] = to_host ? pages_ + h++ : pages_ + host_pages + l++;
        }
        const auto page_bytes = [](int row_bytes) { return static_cast<std::size_t>(row_bytes) * kPage * kKvHeads; };
        const int row_sizes[] = {key_bytes(), value_bytes(), key_scale_bytes(), value_scale_bytes()};
        void* host[4]         = {nullptr, nullptr, nullptr, nullptr};
        DeviceBuffer* lent[4] = {&lent_k_, &lent_v_, &lent_ks_, &lent_vs_};
        for (int i = 0; i < 4; ++i) {
            if (row_sizes[i] == 0) { continue; }
            const std::size_t bytes = page_bytes(row_sizes[i]) * std::max(host_pages, 1);
            t::cuda_check(cudaHostAlloc(&host[i], bytes, cudaHostAllocMapped), "host plane");
            std::memset(host[i], kSpaceByte, bytes);
            host_planes_.push_back(host[i]);
            host_bytes_.push_back(bytes);
            *lent[i] = DeviceBuffer(page_bytes(row_sizes[i]) * std::max(lent_pages, 1));
            lent[i]->fill(kSpaceByte);
        }
        spaces_pooled_ = DeviceBuffer(static_cast<std::size_t>(kSlotWidth) * kPage * 2 *
                                      std::max(host_pages + lent_pages, 1));
        spaces_pooled_.fill(kSpaceByte);
        spaces_ = ops::QsaPageSpaces{.device_pages = pages_,
                                     .host_pages   = host_pages,
                                     .host_k       = host[0],
                                     .host_v       = host[1],
                                     .host_k_scale = host[2],
                                     .host_v_scale = host[3],
                                     .lent_k       = lent_k_.p,
                                     .lent_v       = lent_v_.p,
                                     .lent_k_scale = lent_ks_.p,
                                     .lent_v_scale = lent_vs_.p,
                                     .pooled       = spaces_pooled_.p};
    }

    KvCacheStorage storage_;
    infernix::PagedKVStorageLayout layout_;
    std::vector<int> first_;
    std::vector<Image> images_;
    std::uint32_t seed_;
    int rows_ = 0, pages_per_row_ = 0, pages_ = 0;
    std::vector<std::int32_t> tables_host_;
    std::vector<std::uint16_t> wq_, wk_, wiq_, wik_, tails_host_;
    std::vector<std::uint8_t> k0_, v0_, ks0_, vs0_;
    DeviceBuffer dwq_, dwk_, dwiq_, dwik_, dtables_, dk_, dv_, dks_, dvs_, rk_, rv_, rks_, rvs_, dpooled_, dtails_;
    ops::QsaPageSpaces spaces_{};
    std::vector<void*> host_planes_;
    std::vector<std::size_t> host_bytes_;
    DeviceBuffer lent_k_, lent_v_, lent_ks_, lent_vs_, spaces_pooled_;
    std::vector<int> tail_last_;
    std::set<std::pair<int, int>> written_;                               // (sequence, position)
    std::map<std::pair<int, int>, std::vector<std::uint16_t>> q_, iq_; // the latest outputs of a column
};

void test_prepare(KvCacheStorage storage, std::uint32_t seed) {
    const std::string name = std::string("qsa_prepare ") + storage_name(storage) +
                             (g_index_geometry.yarn_factor > 1.0F ? " yarn" : "");
    // Four sequences whose first positions fall on every residue mod 4 (a block carried into the
    // call from the tails, or none), one at a page's end; images give the blocks distinct axes.
    const std::vector<int> first        = {2049, 3006, 4093, 126};
    const std::vector<Image> images     = {{2040, 4, 8}, {3000, 8, 4}, {4090, 2, 8}, {0, 0, 0}};
    constexpr int kExtent               = 4160;
    const auto harness = [&](std::uint32_t s, bool spaces = false) {
        return std::make_unique<PrepareHarness>(storage, first, images, kExtent, s, spaces);
    };
    {
        // Decode of four rows; a verification of two (tails unchanged); its commit as a narrower
        // call over the same positions; a call across a page boundary; two rows completing their
        // carried blocks at their first column.
        auto h = harness(seed);
        h->call({0, 1, 2, 3}, {2049, 3006, 4093, 126}, 1, true);
        h->check(name + " decode W=1 B=4");
        h->call({0, 1}, {2050, 3007}, 5, false);
        h->check(name + " verify W=5 B=2");
        h->call({0, 1}, {2050, 3007}, 3, true);
        h->check(name + " commit W=3 B=2");
        h->call({2}, {4094}, 8, true);
        h->check(name + " W=8 B=1 across a page");
        h->call({3, 0}, {127, 2053}, 4, true);
        h->check(name + " W=4 B=2");
    }
    {
        // A column's bits do not depend on its call: one W = 8 call against eight W = 1 calls.
        auto wide = harness(seed + 1), narrow = harness(seed + 1);
        wide->call({2}, {4093}, 8, true);
        for (int j = 0; j < 8; ++j) { narrow->call({2}, {4093 + j}, 1, true); }
        wide->check(name + " W=8");
        narrow->check(name + " eight W=1");
        expect(wide->same_outputs(*narrow) && wide->same_state(*narrow, /*tails=*/false),
               name + ": a W=8 call equals eight W=1 calls bit for bit");
    }
    {
        // ... at prefill widths too (one route serves every width, so a decode append is a prefill
        // append's arithmetic): one W = 896 chunk against 896 W = 1 calls.
        auto wide = harness(seed + 6), narrow = harness(seed + 6);
        wide->call({3}, {126}, 896, true);
        for (int j = 0; j < 896; ++j) { narrow->call({3}, {126 + j}, 1, true); }
        wide->check(name + " W=896");
        expect(wide->same_outputs(*narrow) && wide->same_state(*narrow, /*tails=*/false),
               name + ": a W=896 call equals 896 W=1 calls bit for bit");
    }
    {
        // ... and not on the other rows of its call: B = 4 against four B = 1 calls.
        auto batched = harness(seed + 2), alone = harness(seed + 2);
        batched->call({0, 1, 2, 3}, first, 2, true);
        for (int r = 0; r < 4; ++r) { alone->call({r}, {first[r]}, 2, true); }
        batched->check(name + " W=2 B=4");
        expect(batched->same_outputs(*alone) && batched->same_state(*alone),
               name + ": a B=4 call equals four B=1 calls bit for bit");
    }
    {
        // In place and K/V only (the MTP drafter's chunks) give the eager call's bits.
        auto eager = harness(seed + 3), in_place = harness(seed + 3), kv_only = harness(seed + 3);
        eager->call({0, 3}, {2049, 126}, 3, true);
        in_place->call({0, 3}, {2049, 126}, 3, true, Launch::InPlace);
        kv_only->call({0, 3}, {2049, 126}, 3, true, Launch::KvOnly);
        expect(in_place->same_outputs(*eager) && in_place->same_state(*eager),
               name + ": in-place queries give the separate outputs' bits");
        expect(kv_only->same_state(*eager), name + ": a K/V-only call writes the same state");
        kv_only->check(name + " K/V only");
    }
    {
        // Page spaces: pages before the calls live in host and lent spaces; the calls write only the
        // pool and give the bits of the all-pool layout.
        auto pool = harness(seed + 4), spaced = harness(seed + 4, true);
        for (auto* h : {pool.get(), spaced.get()}) {
            h->call({0, 1, 2, 3}, first, 1, true);
            h->call({1, 2}, {3007, 4094}, 6, false);
        }
        spaced->check(name + " page spaces");
        expect(spaced->same_outputs(*pool) && spaced->same_state(*pool),
               name + ": pages in host and lent spaces give the pool layout's bits");
    }
    {
        // CUDA Graph: captured once, replayed with new rows, positions and table rows (swapped
        // sequences) uploaded into the same buffers.
        auto graph = harness(seed + 5), eager = harness(seed + 5);
        PrepareCall captured;
        graph->call({0, 1}, {2049, 3006}, 4, true, Launch::Graph, &captured);
        graph->call({1, 0}, {3010, 2053}, 4, true, Launch::Graph, &captured);
        eager->call({0, 1}, {2049, 3006}, 4, true);
        eager->call({1, 0}, {3010, 2053}, 4, true);
        graph->check(name + " graph replay");
        expect(graph->same_outputs(*eager) && graph->same_state(*eager),
               name + ": graph replays give the eager calls' bits");
    }
}

// The index queries and pooled keys (plain RoPE, YaRN over a short span and Flash-Next's YaRN x4),
// and the fused prepare under YaRN.
void run_index_part() {
    for (const int columns : {1, 5, 64, 1024}) { test_index_query(columns, 100U + columns); }
    test_pool(7);
    // Under YaRN (factor 4 over an original span of 4096, so the ramp falls inside the rotary
    // pairs and positions reach past the span).
    g_index_geometry.yarn_factor        = 4.0F;
    g_index_geometry.original_positions = 4096;
    for (const int columns : {1, 64}) { test_index_query(columns, 200U + columns); }
    test_pool(9);
    // Qwen3.8-Flash-Next's YaRN x4: an original span of 262144, positions up to the last of 1M.
    g_index_geometry.original_positions = 262144;
    for (const int columns : {1, 64}) { test_index_query(columns, 300U + columns, 1048575); }
    g_index_geometry = kGeometry;
    expect(!ops::qsa_prepare_supports(KvCacheStorage::Vq2) &&
               !ops::qsa_prepare_supports(KvCacheStorage::Q4KeyVq2Value),
           "qsa_prepare: the vector-quantized storages keep the composed prepare");
    g_index_geometry.yarn_factor        = 4.0F;
    g_index_geometry.original_positions = 4096;
    test_prepare(KvCacheStorage::Int8Group64, 601);
    g_index_geometry = kGeometry;
}

// Attention over one scalar KV storage, and its fused prepare (qsa-prepare-fused).
void run_storage_part(KvCacheStorage storage, std::uint32_t prepare_seed) {
    // Decode across the dense/selected boundary (2,051 visible tokens is the last dense one),
    // verification across a page boundary, prefill columns crossing the boundary inside one
    // call (one attention split), and a split-K prefill width.
    // Calls the FP32 kernel would run unsplit (columns x KV heads of at least half the SMs: 43+
    // columns on an RTX 5090) take the Tensor Core prompt
    // route: dense columns, the dense/selected boundary inside the call, selected blocks with
    // tails over several pages, two rows, a list length off the 16-token tile, and the class's
    // narrowest call. Every column of a prompt-route call must also equal, bit for bit, the
    // same column computed in a call of another width of the class (checked below).
    const AttentionCase cases[] = {{"decode W=1 B=2", 1, {2050, 2051}},
                                   {"decode W=1 B=4", 1, {2600, 3100, 4093, 2051}},
                                   {"decode W=1 B=8", 1, {2051, 2300, 2700, 3001, 3500, 4000, 4500, 5003}},
                                   {"verify W=4 B=2", 4, {2500, 3700}},
                                   {"verify W=5 B=2", 5, {3001, 4093}},
                                   {"prefill W=192 B=1", 192, {1900}},
                                   {"prefill W=40 B=1", 40, {5000}},
                                   {"prompt W=256 B=1 dense", 256, {0}},
                                   {"prompt W=300 B=1 boundary", 300, {1901}},
                                   {"prompt W=256 B=2 selected", 256, {6001, 9013}},
                                   {"prompt W=257 B=1 selected", 257, {4093}},
                                   {"prompt W=86 B=1 selected", 86, {7000}},
                                   {"prompt W=43 B=1 class edge (170 SMs)", 43, {7000}}};
    std::uint32_t seed = 31;
    for (const auto& c : cases) { test_attention(storage, c, seed++); }
    expect(ops::qsa_prepare_supports(storage), std::string("qsa_prepare supports ") + storage_name(storage));
    test_prepare(storage, prepare_seed);
}

// Vector-quantized storages: the Op's own append; decode, verification and prefill with the
// exact window (the prompt route from 86 columns of one row or 43 of two; a staged call wider
// than kKVWindowInlineWidth), and decode and a prompt call without one.
void run_vq_part(KvCacheStorage storage) {
    const VqCase vq_cases[] = {{"decode W=1 B=2 window", 1, {2050, 3001}, true},
                               {"verify W=5 B=1 window", 5, {4093}, true},
                               {"prefill W=40 B=1 window", 40, {2500}, true},
                               {"prompt W=100 B=2 window", 100, {3000, 7000}, true},
                               {"prompt W=200 B=1 window", 200, {6000}, true},
                               {"prompt W=300 B=1 window staged", 300, {5000}, true},
                               {"prompt W=300 B=1 codes only", 300, {5000}, false},
                               {"decode W=1 B=1 codes only", 1, {3000}, false}};
    std::uint32_t seed = 71;
    for (const auto& c : vq_cases) { test_vq_attention(storage, c, seed++); }
}

} // namespace

int main(int argc, char** argv) {
    // --part selects one independent part, so CTest runs them concurrently: `index` (index queries,
    // pooled keys, the YaRN prepare), one KV storage (`bf16`, `int8`, `fp8`, `nvfp4`, `k8v4`:
    // attention and the fused prepare) or one vector-quantized storage (`vq2`, `k4v2`). Each case
    // keeps its seed, so the parts together run exactly the cases of one unsharded invocation.
    std::string part = "all";
    if (argc == 3 && std::string(argv[1]) == "--part") {
        part = argv[2];
    } else if (argc != 1) {
        std::fprintf(stderr,
                     "usage: infernix_qsa_test [--part index|bf16|int8|fp8|nvfp4|k8v4|vq2|k4v2]\n");
        return 2;
    }
    const std::array<std::pair<const char*, KvCacheStorage>, 5> scalar_storages{{
        {"bf16", KvCacheStorage::BFloat16},
        {"int8", KvCacheStorage::Int8Group64},
        {"fp8", KvCacheStorage::Fp8E4M3Row256},
        {"nvfp4", KvCacheStorage::Nvfp4Group16},
        {"k8v4", KvCacheStorage::Fp8KeyNvfp4Value},
    }};
    const bool known = part == "all" || part == "index" || part == "vq2" || part == "k4v2" ||
                       std::any_of(scalar_storages.begin(), scalar_storages.end(),
                                   [&](const auto& entry) { return part == entry.first; });
    if (!known) {
        std::fprintf(stderr, "unknown --part %s\n", part.c_str());
        return 2;
    }
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    const auto selected = [&](const char* name) { return part == "all" || part == name; };
    try {
        if (selected("index")) { run_index_part(); }
        std::uint32_t prepare_seed = 501;
        for (const auto& [name, storage] : scalar_storages) {
            if (selected(name)) { run_storage_part(storage, prepare_seed); }
            prepare_seed += 10;
        }
        if (selected("vq2")) { run_vq_part(KvCacheStorage::Vq2); }
        if (selected("k4v2")) { run_vq_part(KvCacheStorage::Q4KeyVq2Value); }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all qsa checks passed (%s)\n", part.c_str());
    return 0;
}
