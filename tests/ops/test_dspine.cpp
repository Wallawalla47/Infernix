// Ported from NInfer-all (https://github.com/iamwavecut/ninfer-all, PRs #341 and #365; Apache-2.0) and modified
// for Infernix (renamed into Infernix's Op contract and its oracle test, the proposal draw sub-keyed by step
// with the Gumbel-max coupled draw: docs/maintainer/dspine.md).
// The DSpine code-bridge Ops against naive FP64 oracles (infernix/ops/dspine.h).
#include "infernix/ops/dspine.h"

#include "core/arena.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

// One BF16 rounding of an FP32 evaluation: half a unit in the last place, plus FP32 slack.
constexpr PointwiseCriterion kOneRounding{.absolute = 1e-30, .relative = 1.0 / 256.0};
// Two roundings in a chain (the norm's y, then its product with the weight).
constexpr PointwiseCriterion kTwoRoundings{.absolute = 1e-30, .relative = 2.0 / 256.0};
constexpr PointwiseCriterion kFp32{.absolute = 1e-6, .relative = 2e-5};
// Probabilities from FP32 edges, exponentials and sums.
constexpr PointwiseCriterion kProbability{.absolute = 1e-5, .relative = 1e-4};

std::vector<float> uniform(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) out[i] = f32_to_bf16(values[i]);
    return out;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int test_bridge_read(int h, int n) {
    auto raw = uniform(std::size_t(h) * n, 11U + h, -3.0F, 3.0F);
    std::fill(raw.begin(), raw.begin() + h, 0.0F); // a zero column: rms is sqrt(FLT_MIN)
    const double eps = 1e-6;
    std::vector<double> normalized(raw.size()), rms(n);
    for (int c = 0; c < n; ++c) {
        double squares = 0;
        for (int i = 0; i < h; ++i) squares += double(raw[std::size_t(c) * h + i]) * raw[std::size_t(c) * h + i];
        rms[c] = std::sqrt(std::max(squares / h, double(std::numeric_limits<float>::min())));
        for (int i = 0; i < h; ++i)
            normalized[std::size_t(c) * h + i] = raw[std::size_t(c) * h + i] / (rms[c] + eps);
    }
    DeviceBuffer d_raw = to_device(bits(raw));
    DeviceBuffer d_normalized(raw.size() * 2), d_rms(std::size_t(n) * 4);
    Tensor raw_t(d_raw.p, DType::BF16, {h, n}), normalized_t(d_normalized.p, DType::BF16, {h, n}),
        rms_t(d_rms.p, DType::FP32, {n});
    ops::dspine_bridge_read(raw_t, float(eps), normalized_t, rms_t, nullptr);
    cuda_synchronize();
    const std::string label = "dspine_bridge_read H=" + std::to_string(h);
    return verify_pointwise(label + " normalized", from_device_bf16(d_normalized, raw.size()),
                            normalized, kOneRounding) +
           verify_pointwise(label + " rms", from_device_f32(d_rms, std::size_t(n)), rms, kFp32);
}

int test_bridge_codes(int c, int n, bool in_place) {
    const auto projected = uniform(std::size_t(c) * n, 23U + c, -0.7F, 0.7F);
    const double eps = 1e-6;
    std::vector<double> codes(projected.size());
    for (int col = 0; col < n; ++col) {
        double squares = 0;
        for (int i = 0; i < c; ++i)
            squares += double(projected[std::size_t(col) * c + i]) * projected[std::size_t(col) * c + i];
        const double norm = std::sqrt(squares);
        for (int i = 0; i < c; ++i)
            codes[std::size_t(col) * c + i] =
                std::sqrt(double(c)) * projected[std::size_t(col) * c + i] / (norm + eps);
    }
    DeviceBuffer d_projected = to_device(bits(projected));
    DeviceBuffer d_codes(projected.size() * 2);
    Tensor projected_t(d_projected.p, DType::BF16, {c, n});
    Tensor codes_t(in_place ? d_projected.p : d_codes.p, DType::BF16, {c, n});
    ops::dspine_bridge_codes(projected_t, float(eps), codes_t, nullptr);
    cuda_synchronize();
    return verify_pointwise("dspine_bridge_codes C=" + std::to_string(c) +
                                (in_place ? " in place" : ""),
                            from_device_bf16(codes_t.data, projected.size()), codes, kOneRounding);
}

int test_bridge_message(int rank, int w, int b, int vocab) {
    const auto packed = uniform(std::size_t(3) * rank * w * b, 31U, -2.0F, 2.0F);
    const auto table  = uniform(std::size_t(2) * rank * vocab, 37U, -2.0F, 2.0F);
    const auto bias   = uniform(std::size_t(rank), 41U, -0.5F, 0.5F);
    std::vector<std::int32_t> anchors(b);
    for (int i = 0; i < b; ++i) anchors[i] = (i * 7919 + 13) % vocab;
    std::vector<double> expected(std::size_t(rank) * w * b, 0.0);
    for (int batch = 0; batch < b; ++batch) {
        for (int col = 1; col < w; ++col) {
            const std::size_t receiver = (std::size_t(batch) * w + col) * 3 * rank;
            for (int r = 0; r < rank; ++r) {
                double s = 0, v = 0;
                if (col == 1) {
                    s = table[std::size_t(anchors[batch]) * 2 * rank + r];
                    v = table[std::size_t(anchors[batch]) * 2 * rank + rank + r];
                } else {
                    const std::size_t sender = (std::size_t(batch) * w + col - 1) * 3 * rank;
                    s = packed[sender + rank + r];
                    v = packed[sender + 2 * rank + r];
                }
                expected[(std::size_t(batch) * w + col) * rank + r] =
                    sigmoid(double(packed[receiver + r]) + s + bias[r]) * v;
            }
        }
    }
    DeviceBuffer d_packed = to_device(bits(packed)), d_table = to_device(bits(table)),
                 d_bias = to_device(bits(bias)), d_anchors = to_device(anchors);
    DeviceBuffer d_message(expected.size() * 2);
    Tensor packed_t(d_packed.p, DType::BF16, {3 * rank, w, b}),
        anchors_t(d_anchors.p, DType::I32, {b}), table_t(d_table.p, DType::BF16, {2 * rank, vocab}),
        bias_t(d_bias.p, DType::BF16, {rank}), message_t(d_message.p, DType::BF16, {rank, w, b});
    ops::dspine_bridge_message(packed_t, anchors_t, table_t, bias_t, message_t, nullptr);
    cuda_synchronize();
    return verify_pointwise("dspine_bridge_message R=" + std::to_string(rank) +
                                " W=" + std::to_string(w) + " B=" + std::to_string(b),
                            from_device_bf16(d_message, expected.size()), expected, kOneRounding);
}

int test_bridge_finish(int h, int w, int b, bool in_place) {
    const std::size_t count = std::size_t(h) * w * b;
    const auto raw    = uniform(count, 43U, -4.0F, 4.0F);
    const auto delta  = uniform(count, 47U, -0.5F, 0.5F);
    const auto weight = uniform(std::size_t(h), 53U, 0.2F, 1.8F);
    std::vector<float> rms(std::size_t(w) * b);
    fill_uniform(rms, 59U, 0.3F, 2.5F);
    const double eps = 1e-6;
    std::vector<double> out(count), normed(count), denominator(rms.size());
    for (std::size_t col = 0; col < rms.size(); ++col) {
        double squares = 0;
        for (int i = 0; i < h; ++i) {
            const std::size_t at = col * h + i;
            // The stored out is a BF16 rounding of an FP32 sum; the norm reads the stored value.
            const float stored =
                col % w == 0 ? raw[at] : bf16_to_f32(f32_to_bf16(raw[at] + rms[col] * delta[at]));
            out[at] = col % w == 0 ? raw[at] : double(raw[at]) + double(rms[col]) * delta[at];
            squares += double(stored) * stored;
        }
        denominator[col] = std::sqrt(squares / h + eps);
        for (int i = 0; i < h; ++i) {
            const std::size_t at = col * h + i;
            const float stored =
                col % w == 0 ? raw[at] : bf16_to_f32(f32_to_bf16(raw[at] + rms[col] * delta[at]));
            normed[at] = double(stored) / denominator[col] * weight[i];
        }
    }
    DeviceBuffer d_raw = to_device(bits(raw)), d_delta = to_device(bits(delta)),
                 d_weight = to_device(bits(weight)), d_rms = to_device(rms);
    DeviceBuffer d_out(count * 2), d_normed(count * 2), d_denominator(rms.size() * 4);
    Tensor raw_t(d_raw.p, DType::BF16, {h, w, b}), rms_t(d_rms.p, DType::FP32, {w, b}),
        delta_t(d_delta.p, DType::BF16, {h, w, b}), weight_t(d_weight.p, DType::BF16, {h}),
        out_t(in_place ? d_raw.p : d_out.p, DType::BF16, {h, w, b}),
        normed_t(d_normed.p, DType::BF16, {h, w, b}),
        denominator_t(d_denominator.p, DType::FP32, {w, b});
    ops::dspine_bridge_finish(raw_t, rms_t, delta_t, weight_t, float(eps), out_t, normed_t,
                              &denominator_t, nullptr);
    cuda_synchronize();
    const std::string label = "dspine_bridge_finish H=" + std::to_string(h) +
                              (in_place ? " in place" : "");
    return verify_pointwise(label + " out", from_device_bf16(out_t.data, count), out, kOneRounding) +
           verify_pointwise(label + " normed", from_device_bf16(d_normed, count), normed,
                            kTwoRoundings) +
           verify_pointwise(label + " denominator", from_device_f32(d_denominator, rms.size()),
                            denominator, kFp32);
}

int test_scale_rows(int h, int n) {
    const auto rows   = uniform(std::size_t(h) * n, 61U, -0.2F, 0.2F);
    const auto weight = uniform(std::size_t(h), 67U, 0.1F, 2.0F);
    std::vector<double> expected(rows.size());
    for (std::size_t i = 0; i < rows.size(); ++i) expected[i] = double(rows[i]) * weight[i % h];
    DeviceBuffer d_rows = to_device(bits(rows)), d_weight = to_device(bits(weight)),
                 d_out(rows.size() * 2);
    Tensor rows_t(d_rows.p, DType::BF16, {h, n}), weight_t(d_weight.p, DType::BF16, {h}),
        out_t(d_out.p, DType::BF16, {h, n});
    ops::dspine_scale_rows(rows_t, weight_t, out_t, nullptr);
    cuda_synchronize();
    return verify_pointwise("dspine_scale_rows", from_device_bf16(d_out, rows.size()), expected,
                            kOneRounding);
}

int test_store_head_write(DType dtype, int vocab, int rank, int first, int n) {
    std::vector<float> logits(std::size_t(vocab) * n);
    fill_uniform(logits, 71U + n, -30.0F, 30.0F);
    if (dtype == DType::BF16) round_to_bf16(logits);
    const auto before = bits(uniform(std::size_t(rank) * vocab, 73U, -1.0F, 1.0F));
    // Round-to-nearest-even of each FP32 value; a BF16 value is its own rounding. The other
    // entries keep their bits.
    std::vector<std::uint16_t> expected = before;
    for (int v = 0; v < vocab; ++v) {
        for (int j = 0; j < n; ++j) {
            expected[std::size_t(v) * rank + first + j] =
                f32_to_bf16(logits[std::size_t(j) * vocab + v]);
        }
    }
    DeviceBuffer d_logits = dtype == DType::BF16 ? to_device(bits(logits)) : to_device(logits);
    DeviceBuffer d_table  = to_device(before);
    Tensor logits_t(d_logits.p, dtype, {vocab, n}), table_t(d_table.p, DType::BF16, {rank, vocab});
    ops::dspine_store_head_write(logits_t, first, table_t, nullptr);
    cuda_synchronize();
    return verify_exact("dspine_store_head_write",
                        from_device<std::uint16_t>(d_table, before.size()), expected);
}

// --- dspine_path -------------------------------------------------------------------------------

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

double oracle_uniform(std::uint64_t seed, std::int32_t position, std::uint32_t sub) {
    std::uint64_t key = seed;
    key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                            0xD1B54A32D192ED03ULL));
    key = splitmix64(key ^ (static_cast<std::uint64_t>(ops::kSamplePurposeDSpineProposal) << 21) ^
                     (static_cast<std::uint64_t>(sub) * 0x2545F4914F6CDD1DULL));
    const auto top = static_cast<std::uint32_t>(key >> 40);
    return static_cast<float>(top) * (1.0F / 16777216.0F);
}

struct PathCase {
    int rank = 512, vocab = 2048, steps = 15, width = 16, batch = 3;
    std::uint32_t seed = 1;
    bool gumbel = false; // sampled rows are gumbel_max rows (the coupled draw)
};

// sampling_gumbel_score's uniform: (seed, position, kSamplePurposeGumbel, token), held below 1.
double oracle_gumbel_uniform(std::uint64_t seed, std::int32_t position, std::int32_t token) {
    std::uint64_t key = seed;
    key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                            0xD1B54A32D192ED03ULL));
    key = splitmix64(key ^ (static_cast<std::uint64_t>(ops::kSamplePurposeGumbel) << 21) ^
                     (static_cast<std::uint64_t>(static_cast<std::uint32_t>(token)) * 0x2545F4914F6CDD1DULL));
    const auto top = static_cast<std::uint32_t>(key >> 40);
    return std::min(static_cast<double>(static_cast<float>(top) * (1.0F / 16777216.0F)) + 1.0 / 33554432.0,
                    static_cast<double>(0x1.fffffep-1F));
}

struct PathData {
    std::vector<std::int32_t> ids, anchors, positions;
    std::vector<float> scores, rms, packed, message, denominator, head_write, sender, bias;
    std::vector<ops::SamplingConfig> configs;
};

PathData make_path(const PathCase& p) {
    PathData d;
    std::mt19937 random(p.seed);
    d.ids.resize(std::size_t(16) * p.steps * p.batch);
    for (std::size_t row = 0; row < d.ids.size() / 16; ++row) {
        std::vector<std::int32_t> all(p.vocab);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), random);
        std::copy(all.begin(), all.begin() + 16, d.ids.begin() + row * 16);
    }
    for (int b = 0; b < p.batch; ++b) {
        d.anchors.push_back(int(random() % p.vocab));
        d.positions.push_back(int(random() % 100000));
    }
    d.scores.resize(std::size_t(16) * p.steps * p.batch);
    fill_uniform(d.scores, p.seed + 1, -4.0F, 4.0F);
    // A slot candidate_topk could not fill: id 0 at -inf takes no part.
    d.ids[16 * (d.ids.size() / 16 - 1) + 15]    = 0;
    d.scores[16 * (d.ids.size() / 16 - 1) + 15] = -std::numeric_limits<float>::infinity();
    d.message    = uniform(std::size_t(p.rank) * p.width * p.batch, p.seed + 2, -1.0F, 1.0F);
    d.packed     = uniform(std::size_t(3) * p.rank * p.width * p.batch, p.seed + 4, -1.5F, 1.5F);
    d.head_write = uniform(std::size_t(p.rank) * p.vocab, p.seed + 5, -0.06F, 0.06F);
    d.sender     = uniform(std::size_t(2) * p.rank * p.vocab, p.seed + 6, -2.0F, 2.0F);
    d.bias       = uniform(std::size_t(p.rank), p.seed + 7, -0.5F, 0.5F);
    d.rms.resize(std::size_t(p.width) * p.batch);
    d.denominator.resize(d.rms.size());
    fill_uniform(d.rms, p.seed + 8, 0.5F, 2.0F);
    fill_uniform(d.denominator, p.seed + 9, 0.5F, 2.0F);
    const float temperatures[3] = {0.0F, 0.8F, 1.3F};
    for (int b = 0; b < p.batch; ++b) {
        ops::SamplingConfig config;
        config.temperature = temperatures[b % 3];
        config.seed        = 1000003ULL * (b + 1) + p.seed;
        config.gumbel_max  = p.gumbel && config.temperature > 0.0F;
        if (b % 3 == 2) {
            config.top_k = 6;
            config.top_p = 0.8F;
            config.min_p = 0.02F;
        }
        d.configs.push_back(config);
    }
    return d;
}

struct PathOracle {
    std::vector<std::int32_t> drafts;
    std::vector<double> q, confidence;
    double margin = std::numeric_limits<double>::infinity(); // the closest decision
};

PathOracle path_oracle(const PathCase& p, const PathData& d) {
    PathOracle o;
    o.drafts.resize(std::size_t(p.steps) * p.batch);
    o.q.assign(std::size_t(16) * p.steps * p.batch, 0.0);
    o.confidence.resize(o.drafts.size());
    for (int b = 0; b < p.batch; ++b) {
        const auto& config = d.configs[b];
        std::int64_t sender = d.anchors[b];
        for (int i = 0; i < p.steps; ++i) {
            const std::size_t row = std::size_t(b) * p.steps + i;
            const int column      = i + 1;
            const std::size_t at  = std::size_t(b) * p.width + column;
            std::vector<double> message(p.rank);
            for (int r = 0; r < p.rank; ++r) {
                const double q = d.packed[at * 3 * p.rank + r];
                message[r] = sigmoid(q + d.sender[std::size_t(sender) * 2 * p.rank + r] + d.bias[r]) *
                             d.sender[std::size_t(sender) * 2 * p.rank + p.rank + r];
            }
            double edge[16];
            for (int c = 0; c < 16; ++c) {
                const std::int64_t id = d.ids[row * 16 + c];
                if (d.scores[row * 16 + c] == -std::numeric_limits<float>::infinity()) {
                    edge[c] = -std::numeric_limits<double>::infinity();
                    continue;
                }
                double dot = 0;
                for (int r = 0; r < p.rank; ++r)
                    dot += (message[r] - double(d.message[at * p.rank + r])) *
                           d.head_write[std::size_t(id) * p.rank + r];
                edge[c] = double(d.denominator[at]) * d.scores[row * 16 + c] +
                          double(d.rms[at]) * dot;
            }
            const double scale = d.denominator[at];
            int selected        = 0;
            const double maximum = *std::max_element(edge, edge + 16);
            if (config.temperature <= 0.0F) {
                for (int c = 0; c < 16; ++c)
                    if (edge[c] == maximum && (d.ids[row * 16 + c] < d.ids[row * 16 + selected] ||
                                               edge[selected] != maximum))
                        selected = c;
                double second = -std::numeric_limits<double>::infinity();
                for (int c = 0; c < 16; ++c)
                    if (c != selected) second = std::max(second, edge[c]);
                if (second > -std::numeric_limits<double>::infinity())
                    o.margin = std::min(o.margin, (maximum - second) / scale);
                o.q[row * 16 + selected] = 1.0;
            } else {
                double weight[16], total = 0;
                for (int c = 0; c < 16; ++c) {
                    weight[c] = std::exp((edge[c] - maximum) / scale / config.temperature);
                    total += weight[c];
                }
                double prob[16], p_max = 0;
                for (int c = 0; c < 16; ++c) p_max = std::max(p_max, prob[c] = weight[c] / total);
                const int top_k = config.top_k > 0 && config.top_k < 16 ? config.top_k : 16;
                bool keep[16];
                double kept_total = 0;
                for (int c = 0; c < 16; ++c) {
                    int rank      = 0;
                    double before = 0;
                    for (int e = 0; e < 16; ++e) {
                        if (prob[e] > prob[c] || (prob[e] == prob[c] && e < c)) {
                            ++rank;
                            before += prob[e];
                        }
                    }
                    bool k = rank < top_k && (config.top_p >= 1.0F || before <= config.top_p);
                    if (config.min_p > 0.0F) k = k && prob[c] >= config.min_p * p_max;
                    keep[c] = prob[c] > 0 && (k || rank == 0);
                    if (keep[c]) kept_total += prob[c];
                    // A kept decision within reach of rounding would make the test ambiguous.
                    if (config.top_p < 1.0F) o.margin = std::min(o.margin, std::abs(before - config.top_p));
                    if (config.min_p > 0.0F)
                        o.margin = std::min(o.margin, std::abs(prob[c] - config.min_p * p_max));
                }
                double cumulative = 0, threshold = 0;
                const double u = oracle_uniform(config.seed, d.positions[b] + i, static_cast<std::uint32_t>(i));
                threshold      = u * 1.0;
                int last       = 0;
                selected       = -1;
                double score[16];
                for (int c = 0; c < 16; ++c) {
                    const double qc = keep[c] ? prob[c] / kept_total : 0.0;
                    o.q[row * 16 + c] = qc;
                    if (keep[c]) last = c;
                    cumulative += qc;
                    if (keep[c] && !config.gumbel_max) o.margin = std::min(o.margin, std::abs(cumulative - threshold));
                    if (selected < 0 && cumulative > threshold) selected = c;
                    const double g = oracle_gumbel_uniform(config.seed, d.positions[b] + i + 1, d.ids[row * 16 + c]);
                    score[c] = keep[c] && qc > 0 ? std::log(qc) - std::log(-std::log(g))
                                                 : -std::numeric_limits<double>::infinity();
                }
                if (selected < 0 || selected > last) selected = last;
                if (config.gumbel_max) {
                    // The coupled draw: the kept candidate of the greatest Gumbel score, the lower slot on a tie.
                    selected = 0;
                    for (int c = 1; c < 16; ++c)
                        if (score[c] > score[selected]) selected = c;
                    for (int c = 0; c < 16; ++c)
                        if (c != selected && score[c] > -std::numeric_limits<double>::infinity())
                            o.margin = std::min(o.margin, score[selected] - score[c]);
                }
            }
            double total = 0;
            for (int c = 0; c < 16; ++c) total += std::exp((edge[c] - maximum) / scale);
            o.confidence[row]   = std::exp((edge[selected] - maximum) / scale) / total;
            o.drafts[row]       = d.ids[row * 16 + selected];
            sender              = d.ids[row * 16 + selected];
        }
    }
    return o;
}

int test_path(PathCase p) {
    // Choose inputs whose every decision is far from a tie at FP32 precision.
    PathData d;
    PathOracle o;
    for (int attempt = 0;; ++attempt) {
        d = make_path(p);
        o = path_oracle(p, d);
        if (o.margin > 1e-4) break;
        if (attempt == 20) {
            std::cerr << "dspine_path: no unambiguous input in 20 attempts\n";
            return 1;
        }
        ++p.seed;
    }
    DeviceBuffer d_ids = to_device(d.ids), d_scores = to_device(d.scores),
                 d_message = to_device(bits(d.message)),
                 d_rms = to_device(d.rms), d_packed = to_device(bits(d.packed)),
                 d_denominator = to_device(d.denominator), d_head = to_device(bits(d.head_write)),
                 d_sender = to_device(bits(d.sender)), d_bias = to_device(bits(d.bias)),
                 d_anchors = to_device(d.anchors), d_positions = to_device(d.positions),
                 d_configs = to_device(d.configs);
    const std::size_t rows = std::size_t(p.steps) * p.batch;
    DeviceBuffer d_drafts(rows * 4), d_q(rows * 16 * 4), d_confidence(rows * 4);
    Tensor ids_t(d_ids.p, DType::I32, {16, p.steps, p.batch}),
        scores_t(d_scores.p, DType::FP32, {16, p.steps, p.batch}),
        message_t(d_message.p, DType::BF16, {p.rank, p.width, p.batch}),
        rms_t(d_rms.p, DType::FP32, {p.width, p.batch}),
        packed_t(d_packed.p, DType::BF16, {3 * p.rank, p.width, p.batch}),
        denominator_t(d_denominator.p, DType::FP32, {p.width, p.batch}),
        head_t(d_head.p, DType::BF16, {p.rank, p.vocab}),
        sender_t(d_sender.p, DType::BF16, {2 * p.rank, p.vocab}),
        bias_t(d_bias.p, DType::BF16, {p.rank}), anchors_t(d_anchors.p, DType::I32, {p.batch}),
        positions_t(d_positions.p, DType::I32, {p.batch}),
        drafts_t(d_drafts.p, DType::I32, {p.steps, p.batch}),
        q_t(d_q.p, DType::FP32, {16, p.steps, p.batch}),
        confidence_t(d_confidence.p, DType::FP32, {p.steps, p.batch});
    DeviceArena workspace(ops::dspine_path_workspace_capacity_bytes(p.steps, p.batch));
    ops::dspine_path(ids_t, scores_t, denominator_t, rms_t, packed_t, message_t, head_t,
                     sender_t, bias_t, anchors_t, positions_t,
                     static_cast<const ops::SamplingConfig*>(d_configs.p), drafts_t, q_t,
                     &confidence_t, workspace, nullptr);
    cuda_synchronize();
    const std::string label = "dspine_path K=" + std::to_string(p.steps) +
                              " W=" + std::to_string(p.width) + " B=" + std::to_string(p.batch);
    int failures =
        verify_exact((label + " drafts").c_str(), from_device<std::int32_t>(d_drafts, rows), o.drafts);
    failures +=
        verify_pointwise(label + " proposal_q", from_device_f32(d_q, rows * 16), o.q, kProbability);
    failures += verify_pointwise(label + " confidence", from_device_f32(d_confidence, rows),
                                 o.confidence, kProbability);
    if (workspace.peak_used() > ops::dspine_path_workspace_capacity_bytes(p.steps, p.batch)) {
        std::cerr << label << ": workspace beyond its capacity\n";
        ++failures;
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(dspine) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += test_bridge_read(5120, 32);
        failures += test_bridge_read(2816, 3);
        failures += test_bridge_codes(1024, 32, false);
        failures += test_bridge_codes(1024, 7, true);
        failures += test_bridge_message(512, 16, 3, 1000);
        failures += test_bridge_message(64, 5, 8, 300);
        failures += test_bridge_finish(5120, 16, 2, false);
        failures += test_bridge_finish(2048, 9, 3, true);
        failures += test_scale_rows(5120, 33);
        failures += test_store_head_write(DType::BF16, 1000, 512, 7, 64);
        failures += test_store_head_write(DType::FP32, 248320, 512, 0, 32);
        failures += test_path({});
        failures += test_path({.gumbel = true});
        failures += test_path({.rank = 256, .vocab = 4096, .steps = 1, .width = 2, .batch = 1,
                               .seed = 7});
        failures += test_path({.rank = 512, .vocab = 3000, .steps = 7, .width = 16, .batch = 8,
                               .seed = 19});
    } catch (const std::exception& error) {
        std::cerr << "dspine threw: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << "dspine failures=" << failures << '\n';
        return 1;
    }
    std::cout << "dspine: PASS\n";
    return 0;
}
