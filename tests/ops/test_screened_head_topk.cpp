// ops::screened_head_fit and ops::screened_head_topk (the 5,120-wide screened head, candidates RV-41) against host
// oracles, on an FP8 row-scaled (BF16 scales) and a row-split q8_g32_fp16 head of 8,192 x 5,120 (512 items over the
// device's screen CTAs, so each CTA streams several), valid_rows = 8,176.
//
// Fit: the screen's bytes against a host replay of the documented fit (every step an explicitly rounded FP32
// operation, binary16 by round to nearest even), for every 13th row and the planted rows, all 80 groups.
//
// Top sixteen, at T = 1, 5, 8, 9 and 16 (both screen tile widths, full and ragged):
// - columns with planted rows (even columns: 16 rows aligned with the column's signs, at distinct magnitudes, so
//   their exact scores exceed every other valid row's by far): the candidates equal the FP64 top sixteen exactly;
//   a planted row past valid_rows with a still larger score never appears;
// - every column: each candidate's score is its row's FP64 dot within the FP32 summation bound
//   |s - e| <= 2^-12 sum |w x| + 1e-30, the candidates are distinct valid rows in descending score order (equal
//   scores in ascending row order);
// - column invariance: a column's candidates and scores are the same bits in every call that holds it (T = 1 .. 16);
// - a CUDA Graph replay of the T = 16 call gives the eager call's bits.
// The recall of the FP64 top sixteen on the unplanted columns is reported (the screen's quality on this synthetic
// head), not asserted: the contract promises the exact top sixteen only for rows the screen selects.
#include "ops/op_family.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "infernix/ops/screened_head.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::input_projection;

namespace {

namespace qw = quantized_weight;

constexpr int kRows    = 8192;
constexpr int kK       = 5120;
constexpr int kValid   = kRows - 16;
constexpr int kColumns = 16;
constexpr int kGroups  = kK / 64;
constexpr int kTop     = 16;

int failures = 0;

void fail(const std::string& what) {
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }

// x: BF16 [5120, 16] in [-1, 1].
std::vector<std::uint16_t> make_x() {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(kK) * kColumns);
    std::uint32_t state = 0x5EEDU;
    for (auto& v : x) {
        const float unit = static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) /
                           static_cast<float>(1 << 23);
        v = f32_to_bf16(unit);
    }
    return x;
}

float x_value(const std::vector<std::uint16_t>& x, int column, int k) {
    return bf16_to_f32(x[static_cast<std::size_t>(column) * kK + k]);
}

int planted_row(int column, int j) { return 64 + column * 200 + j * 7; }
constexpr int kInvalidPlant = kRows - 3; // past valid_rows

// Rows aligned with even columns' signs, at decreasing code magnitudes; row kInvalidPlant aligned with column 0 at
// the largest magnitude (and, FP8, a doubled row scale).
void plant(qw::PackedWeight& head, const std::vector<std::uint16_t>& x) {
    const bool fp8 = head.weight.qtype == QType::FP8_E4M3FN_ROW_BF16;
    const auto put = [&](int row, int column, int j) {
        for (int k = 0; k < kK; ++k) {
            const float v = x_value(x, column, k);
            std::uint8_t code;
            if (fp8) {
                const std::uint8_t magnitude = static_cast<std::uint8_t>(0x7e - j); // 448, 416, 384, ...
                code = v == 0.0F ? 0 : static_cast<std::uint8_t>(v < 0.0F ? 0x80 | magnitude : magnitude);
            } else {
                const int magnitude = 127 - 6 * j;
                code = static_cast<std::uint8_t>(static_cast<std::int8_t>(v == 0.0F ? 0 : (v < 0.0F ? -magnitude : magnitude)));
            }
            head.payload[static_cast<std::size_t>(row) * kK + k] = code;
        }
    };
    for (int column = 0; column < kColumns; column += 2) {
        for (int j = 0; j < kTop; ++j) { put(planted_row(column, j), column, j); }
    }
    put(kInvalidPlant, 0, 0);
    if (fp8) {
        qw::detail::store_u16_le(head.payload, head.scale_plane_offset + static_cast<std::size_t>(kInvalidPlant) * 2,
                                 0x3c00U); // 2^-7: above every fixture row scale (2^-9 x [1, 1.75])
    }
}

// The head's dequantized FP32 values of one row, as the Op decodes them (FP8 code times the BF16 row scale, or the
// q8 code times its group's binary16 scale, each one rounded FP32 product).
std::vector<float> row_values(const qw::PackedWeight& head, int row) {
    std::vector<float> w(kK);
    if (head.weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        const float scale = qw::detail::bf16_to_f32(
            qw::detail::load_u16_le(head.payload, head.scale_plane_offset + static_cast<std::size_t>(row) * 2));
        for (int k = 0; k < kK; ++k) {
            w[k] = static_cast<float>(qw::detail::decode_e4m3fn(head.payload[static_cast<std::size_t>(row) * kK + k])) *
                   scale;
        }
    } else {
        for (int k = 0; k < kK; ++k) {
            const float scale = qw::detail::f16_to_f32(qw::detail::load_u16_le(
                head.payload, head.scale_plane_offset + (static_cast<std::size_t>(row) * (kK / 32) + k / 32) * 2));
            w[k] = static_cast<float>(static_cast<std::int8_t>(head.payload[static_cast<std::size_t>(row) * kK + k])) *
                   scale;
        }
    }
    return w;
}

// ---- The fit's host replay

float half_round(float v) { return qw::detail::f16_to_f32(qw::detail::f32_to_f16(v)); }

int q2_code(float w, float reciprocal) {
    const float product = w * reciprocal;
    return std::max(-2, std::min(1, static_cast<int>(std::nearbyint(product))));
}

float q2_reciprocal(float scale) { return scale != 0.0F ? 1.0F / scale : 0.0F; }

float q2_error(const float* w, float scale) {
    const float reciprocal = q2_reciprocal(scale);
    float sse              = 0.0F;
    for (int i = 0; i < 64; ++i) {
        const float e = std::fma(-static_cast<float>(q2_code(w[i], reciprocal)), scale, w[i]);
        sse           = std::fma(e, e, sse);
    }
    return sse;
}

// The group's 16 code bytes and binary16 scale word.
void fit_group(const float* w, std::uint8_t (&codes)[16], std::uint16_t& scale_word) {
    float amax = 0.0F, energy = 0.0F;
    for (int i = 0; i < 64; ++i) {
        amax   = std::max(amax, std::fabs(w[i]));
        energy = std::fma(w[i], w[i], energy);
    }
    float scale = 0.0F;
    if (amax > 0.0F) {
        float best_error    = INFINITY;
        const auto consider = [&](float candidate) {
            const float rounded = half_round(candidate);
            if (rounded == 0.0F || !std::isfinite(rounded)) { return; }
            const float error = q2_error(w, rounded);
            if (error < best_error) {
                best_error = error;
                scale      = rounded;
            }
        };
        for (int sign = 1; sign >= -1; sign -= 2) {
            for (int i = 0; i < 29; ++i) {
                const float step   = 0.025F * static_cast<float>(i);
                const float factor = 0.300F + step;
                const float base   = static_cast<float>(sign) * amax;
                consider(base * factor);
            }
        }
        const float canonical = half_round(amax);
        if (canonical > 0.0F && std::isfinite(canonical)) {
            const float base = 1.0F / canonical;
            float overlap = 0.0F, norm = 0.0F;
            for (int i = 0; i < 64; ++i) {
                const float c = static_cast<float>(q2_code(w[i], base));
                overlap       = std::fma(w[i], c, overlap);
                norm          = std::fma(c, c, norm);
            }
            if (norm > 0.0F) { consider(overlap / norm); }
        }
    }
    const float reciprocal = q2_reciprocal(scale);
    float overlap          = 0.0F;
    for (int b = 0; b < 16; ++b) {
        std::uint32_t byte = 0;
        for (int f = 0; f < 4; ++f) {
            const int c          = q2_code(w[4 * b + f], reciprocal);
            const float product  = static_cast<float>(c) * scale;
            overlap              = std::fma(w[4 * b + f], product, overlap);
            byte |= (static_cast<std::uint32_t>(c) & 3U) << (2 * f);
        }
        codes[b] = static_cast<std::uint8_t>(byte);
    }
    float stretched = scale;
    if (overlap > 0.0F) {
        const float ratio     = energy / overlap;
        const float candidate = scale * std::sqrt(ratio);
        if (std::isfinite(half_round(candidate))) { stretched = candidate; }
    }
    scale_word = qw::detail::f32_to_f16(stretched);
}

void check_fit(const std::string& label, const qw::PackedWeight& head, const Weight& screen) {
    const auto codes  = from_device<std::uint8_t>(screen.qdata, static_cast<std::size_t>(kRows) * (kK / 4));
    const auto scales = from_device<std::uint16_t>(screen.scales, static_cast<std::size_t>(kRows) * kGroups);
    std::vector<int> rows;
    for (int row = 0; row < kRows; row += 13) { rows.push_back(row); }
    for (int column = 0; column < kColumns; column += 2) {
        for (int j = 0; j < kTop; ++j) { rows.push_back(planted_row(column, j)); }
    }
    rows.push_back(kInvalidPlant);
    for (const int row : rows) {
        const std::vector<float> w = row_values(head, row);
        for (int group = 0; group < kGroups; ++group) {
            std::uint8_t want[16];
            std::uint16_t want_scale = 0;
            fit_group(w.data() + group * 64, want, want_scale);
            const std::size_t at = static_cast<std::size_t>(row) * (kK / 4) + group * 16;
            if (!std::equal(want, want + 16, codes.begin() + static_cast<std::ptrdiff_t>(at)) ||
                scales[static_cast<std::size_t>(row) * kGroups + group] != want_scale) {
                fail(label + ": fit row " + std::to_string(row) + " group " + std::to_string(group) + " (scale " +
                     std::to_string(scales[static_cast<std::size_t>(row) * kGroups + group]) + ", expected " +
                     std::to_string(want_scale) + ")");
                return;
            }
        }
    }
}

// ---- The top sixteen

struct Exact {
    std::vector<double> score;     // [column][row]
    std::vector<double> magnitude; // sum |w x|
};

Exact exact_scores(const qw::PackedWeight& head, const std::vector<std::uint16_t>& x) {
    Exact out{std::vector<double>(static_cast<std::size_t>(kColumns) * kRows),
              std::vector<double>(static_cast<std::size_t>(kColumns) * kRows)};
    for (int row = 0; row < kRows; ++row) {
        const std::vector<float> w = row_values(head, row);
        for (int column = 0; column < kColumns; ++column) {
            double sum = 0.0, magnitude = 0.0;
            for (int k = 0; k < kK; ++k) {
                const double p = static_cast<double>(w[k]) * x_value(x, column, k);
                sum += p;
                magnitude += std::fabs(p);
            }
            out.score[static_cast<std::size_t>(column) * kRows + row]     = sum;
            out.magnitude[static_cast<std::size_t>(column) * kRows + row] = magnitude;
        }
    }
    return out;
}

std::vector<int> oracle_top(const Exact& exact, int column) {
    std::vector<int> rows(kValid);
    std::iota(rows.begin(), rows.end(), 0);
    const double* s = exact.score.data() + static_cast<std::size_t>(column) * kRows;
    std::partial_sort(rows.begin(), rows.begin() + kTop, rows.end(),
                      [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    rows.resize(kTop);
    return rows;
}

struct Result {
    std::vector<std::int32_t> ids;
    std::vector<float> scores;
};

Result run_topk(const Weight& head, const Weight& screen, const DeviceBuffer& x, int columns, DeviceArena& workspace,
                cudaStream_t stream = nullptr) {
    DeviceBuffer ids(static_cast<std::size_t>(kTop) * columns * 4), scores(static_cast<std::size_t>(kTop) * columns * 4);
    Tensor xt(x.p, DType::BF16, {kK, columns});
    Tensor id_t(ids.p, DType::I32, {kTop, columns});
    Tensor score_t(scores.p, DType::FP32, {kTop, columns});
    ops::screened_head_topk(xt, head, screen, kValid, ops::ScreenedHeadPolicy{}, id_t, score_t, workspace, stream);
    cuda_synchronize();
    return {from_device<std::int32_t>(ids, static_cast<std::size_t>(kTop) * columns),
            from_device<float>(scores, static_cast<std::size_t>(kTop) * columns)};
}

void check_head(const std::string& label, qw::PackedWeight packed, const std::vector<std::uint16_t>& x_bits) {
    plant(packed, x_bits);
    DevicePackedWeight head_buffer(std::move(packed));
    const Weight head = head_buffer.view();
    DeviceBuffer screen_storage(ops::screened_head_wide_screen_bytes(kRows));
    const Weight screen = ops::screened_head_wide_screen(screen_storage.p, kRows);
    ops::screened_head_fit(head, screen, nullptr);
    cuda_synchronize();
    check_fit(label, head_buffer.host, screen);

    const Exact exact     = exact_scores(head_buffer.host, x_bits);
    const DeviceBuffer x  = to_device(x_bits);
    DeviceArena workspace(ops::screened_head_topk_workspace_bytes(kColumns));
    std::vector<Result> by_width;
    double recall_sum = 0.0;
    int recall_columns = 0;
    for (const int T : {16, 1, 5, 8, 9}) {
        const Result r = run_topk(head, screen, x, T, workspace);
        const std::string call = label + " T=" + std::to_string(T);
        for (int column = 0; column < T; ++column) {
            const std::string where = call + " column " + std::to_string(column);
            const std::int32_t* ids = r.ids.data() + static_cast<std::size_t>(column) * kTop;
            const float* scores     = r.scores.data() + static_cast<std::size_t>(column) * kTop;
            for (int j = 0; j < kTop; ++j) {
                const int row = ids[j];
                if (row < 0 || row >= kValid) {
                    fail(where + ": candidate " + std::to_string(row) + " is not a valid row");
                    return;
                }
                for (int i = 0; i < j; ++i) {
                    if (ids[i] == row) {
                        fail(where + ": candidate repeated");
                        return;
                    }
                }
                const std::size_t at = static_cast<std::size_t>(column) * kRows + row;
                if (!(std::fabs(scores[j] - exact.score[at]) <= std::ldexp(exact.magnitude[at], -12) + 1e-30)) {
                    fail(where + ": score of row " + std::to_string(row) + " is " + std::to_string(scores[j]) +
                         ", exact " + std::to_string(exact.score[at]));
                    return;
                }
                if (j > 0 && !(scores[j - 1] > scores[j] || (scores[j - 1] == scores[j] && ids[j - 1] < row))) {
                    fail(where + ": candidates out of order");
                    return;
                }
            }
            const std::vector<int> want = oracle_top(exact, column);
            if (column % 2 == 0) {
                if (!std::equal(want.begin(), want.end(), ids)) {
                    fail(where + ": planted column's candidates differ from the exact top sixteen");
                    return;
                }
            } else if (T == 16) {
                int hits = 0;
                for (const int row : want) { hits += std::find(ids, ids + kTop, row) != ids + kTop; }
                recall_sum += static_cast<double>(hits) / kTop;
                ++recall_columns;
            }
        }
        // Column invariance against the T = 16 call (the first).
        if (!by_width.empty()) {
            const Result& full = by_width.front();
            for (int column = 0; column < T; ++column) {
                for (int j = 0; j < kTop; ++j) {
                    const std::size_t i = static_cast<std::size_t>(column) * kTop + j;
                    if (r.ids[i] != full.ids[i] || std::memcmp(&r.scores[i], &full.scores[i], 4) != 0) {
                        fail(call + ": column " + std::to_string(column) + " differs from the T=16 call");
                        return;
                    }
                }
            }
        }
        by_width.push_back(r);
    }
    std::printf("%s: recall of the exact top sixteen on unplanted columns %.3f (%d columns)\n", label.c_str(),
                recall_columns > 0 ? recall_sum / recall_columns : 0.0, recall_columns);

    // A CUDA Graph replay of the T = 16 call.
    {
        DeviceBuffer ids(static_cast<std::size_t>(kTop) * kColumns * 4), scores(static_cast<std::size_t>(kTop) * kColumns * 4);
        Tensor xt(x.p, DType::BF16, {kK, kColumns});
        Tensor id_t(ids.p, DType::I32, {kTop, kColumns});
        Tensor score_t(scores.p, DType::FP32, {kTop, kColumns});
        DeviceContext device;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(device.stream, [&] {
            ops::screened_head_topk(xt, head, screen, kValid, ops::ScreenedHeadPolicy{}, id_t, score_t, workspace,
                                    device.stream);
        });
        graph.instantiate(definition);
        graph.launch(device.stream);
        cuda_synchronize(device.stream);
        const auto got_ids    = from_device<std::int32_t>(ids, static_cast<std::size_t>(kTop) * kColumns);
        const auto got_scores = from_device<float>(scores, static_cast<std::size_t>(kTop) * kColumns);
        if (got_ids != by_width.front().ids ||
            std::memcmp(got_scores.data(), by_width.front().scores.data(), got_scores.size() * 4) != 0) {
            fail(label + ": graph replay differs");
        }
    }
    if (workspace.used() != 0) { fail(label + ": workspace scope not released"); }
    failures += head_buffer.verify_preserved(label + " head");
}

} // namespace

INFERNIX_OP_TEST(screened_head_topk) {
    try {
        if (cuda_unavailable()) { return 77; }
        const std::vector<std::uint16_t> x = make_x();
        check_head("fp8 row head", qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kRows, kK, 0x41U), x);
        qw::PatternedWeightOptions q8;
        q8.row_split_codes = qw::RowSplitCodePattern::Hashed;
        check_head("q8 head", qw::make_patterned_weight(QType::Q8_G32_FP16, kRows, kK, 0x42U, q8), x);
    } catch (const std::exception& error) {
        std::cerr << "screened_head_topk: " << error.what() << '\n';
        return 1;
    }
    std::printf("%s screened_head_topk\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
