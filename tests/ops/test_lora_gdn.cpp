// LoRA on the Gated DeltaNet input projection (include/infernix/ops/gdn_input_proj.h, include/infernix/ops/lora.h) of
// the Qwen3.8-27B parents: NVFP4 (divisor and multiplier forms) and row-scaled FP8 (BF16 and FP32 row words)
// [16384,5120], and the two-parent Q4/Q5 form; every check runs on non-zero patterned weights.
//
// 1. gdn_input_proj with a term against the no-term call plus the FP64 term (sum of B U times the segment's scale), at
//    every route boundary of the three formats and every policy the parents take, with a narrow (rank 32) and a wide
//    (rank 128) adapter. The term's segments leave the key rows unadapted (their bits must be the no-term call's) and
//    put a segment boundary inside a row tile and across the qkv/z split. A sensitivity count guards the criterion: most
//    adapted values must lie far outside the tolerance a dropped term would get.
// 2. A zero B (PEFT's initial state) gives the no-term call's bits on every route.
// 3. A16ColumnInvariant with a term: every column's bits are its bits in the widest call.
// 4. The snapshot and record forms (and the record's tree form) with a term, exact by construction: query, key, value,
//    z, the published states and the record are bit for bit gdn_input_proj with the term and the same policy over the
//    call's W * B columns, followed by the convolution launcher the materialized route runs on its BF16 output (the
//    NVFP4 post kernels for one row under a policy that is not column-invariant, the generic projected convolution
//    otherwise). The call's workspace is the lora=true capacity query's for exactly its B and W.
// 5. Other parents refuse a term, and the capacity queries refuse lora=true for them.

#include "ops/host_parallel.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include "core/weight.h"
#include "infernix/ops/gdn_input_proj.h"
#include "infernix/ops/lora.h"
#include "infernix/ops/speculative_tree.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
namespace qw = test::quantized_weight;
using P = ops::LinearPolicy;

constexpr std::int32_t kHidden = 5120, kRows = 16384, kQueryRows = 2048, kKeyRows = 2048, kValueRows = 6144,
                       kZRows = 6144, kChannels = kQueryRows + kKeyRows + kValueRows;
static_assert(kChannels + kZRows == kRows);

// Every route boundary of the formats: NVFP4 A16 (GEMV 1, SIMT 2, sliced-K 3-8 / 9-16 / 17-24 / 25-32, MMA 33-64 /
// 65-96 / 97+), A4 (MMA 64 / 96 / 128 columns, the 256-token TMA tile from 129), the 8-bit tiles (sliced-K through 64,
// the ring above); FP8 A16 (GEMV 1, SIMT 2-4, sliced-K 5-16 / 17-24 / 25-32, MMA 33-64 / 65-96 / 97+) and A8 from 17
// (32 / 64 / 128 / 192, the split-K tile at 193-256 and 321-384, 64x256 at 257-320, 96x256 at 385-480, MidBulk at
// 481-512, Bulk from 513); Q4/Q5 (the independent parents 1-12 with Q4's SIMT 2-4 and 5-6, its K-split 7-12 and Q5's
// split4 2-10 / c4 11-12, the grouped tiles 13-32 / 33-64, the ring from 65).
const std::int32_t kWidths[] = {1,  2,  4,   5,   7,   12,  13,  16,  17,  24,  32,  33,  64,
                                65, 96, 97,  128, 129, 193, 250, 300, 350, 400, 500, 513, 2048};
constexpr std::int32_t kMaxWidth = 2048;

std::vector<std::uint16_t> random_bf16(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<std::uint16_t> out(count);
    for (auto& v : out) { v = test::f32_to_bf16(dist(rng)); }
    return out;
}

double bf16(std::uint16_t bits) { return static_cast<double>(test::bf16_to_f32(bits)); }

// Every element's bits; prints how many differ and the first.
int compare_bits(const std::string& label, const std::uint16_t* got, const std::uint16_t* ref, std::size_t count) {
    std::size_t differ = 0, first = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (got[i] != ref[i] && differ++ == 0) { first = i; }
    }
    if (differ == 0) { return 0; }
    std::cerr << label << ": " << differ << " of " << count << " differ; first at " << first << " (" << got[first]
              << " vs " << ref[first] << ")\n";
    return 1;
}

int compare_bits(const std::string& label, const std::vector<std::uint16_t>& got,
                 const std::vector<std::uint16_t>& ref) {
    if (got.size() != ref.size()) {
        std::cerr << label << ": " << got.size() << " values against " << ref.size() << '\n';
        return 1;
    }
    return compare_bits(label, got.data(), ref.data(), got.size());
}

const char* policy_name(P policy) {
    switch (policy) {
    case P::A16Only: return "A16Only";
    case P::AllowA8: return "AllowA8";
    case P::AllowA4: return "AllowA4";
    case P::A16ColumnInvariant: return "A16ColumnInvariant";
    case P::AllowA4Residual: return "AllowA4Residual";
    case P::AllowA8Residual: return "AllowA8Residual";
    case P::A8Only: return "A8Only";
    }
    return "?";
}

std::unique_ptr<test::GuardedDeviceBuffer> device_copy(const void* data, std::size_t bytes) {
    auto buffer = std::make_unique<test::GuardedDeviceBuffer>(std::max<std::size_t>(bytes, 16));
    if (bytes != 0) { buffer->copy_from_host(data, bytes); }
    return buffer;
}

std::vector<std::uint16_t> read_bf16(const test::GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<std::uint16_t> out(count);
    buffer.copy_to_host(out.data(), count * 2);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// The parents.
enum class Parent { Nvfp4, Nvfp4Multiplier, Fp8Bf16, Fp8F32, Q4Q5 };

const char* parent_name(Parent parent) {
    switch (parent) {
    case Parent::Nvfp4: return "nvfp4";
    case Parent::Nvfp4Multiplier: return "nvfp4 multiplier";
    case Parent::Fp8Bf16: return "fp8 bf16";
    case Parent::Fp8F32: return "fp8 f32";
    case Parent::Q4Q5: return "q4/q5";
    }
    return "?";
}

struct ParentWeights {
    Parent parent{};
    QType qtype{}; // the single parent's
    std::vector<std::unique_ptr<test::GuardedDeviceBuffer>> buffers;
    Weight single{}, qk{}, value_z{};

    [[nodiscard]] bool two_parent() const { return parent == Parent::Q4Q5; }
};

ParentWeights make_parent(Parent parent, std::uint32_t seed) {
    ParentWeights w;
    w.parent         = parent;
    const auto upload = [&](const qw::PackedWeight& packed) {
        w.buffers.push_back(device_copy(packed.payload.data(), packed.payload.size()));
        return packed.device_weight(w.buffers.back()->data());
    };
    switch (parent) {
    case Parent::Nvfp4:
        // Non-unit divisors, so a term added after the projection's scale (instead of before) shows.
        w.qtype  = QType::NVFP4;
        w.single = upload(qw::make_patterned_weight(QType::NVFP4, kRows, kHidden, seed,
                                                    {.weight_scale_divisor = 0.125F, .input_scale_divisor = 3.5F}));
        break;
    case Parent::Nvfp4Multiplier:
        // The QAD recipe's multiplier form (ModelOpt's FP32 multiplier, run as NVFP4).
        w.qtype  = QType::NVFP4;
        w.single = upload(qw::make_patterned_weight(QType::NVFP4, kRows, kHidden, seed,
                                                    {.activation_scale = 0.0625F, .weight_multiplier = 3.0F}));
        break;
    case Parent::Fp8Bf16:
        w.qtype  = QType::FP8_E4M3FN_ROW_BF16;
        w.single = upload(qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, seed));
        break;
    case Parent::Fp8F32:
        // The QAD recipe's FP8 row-F32 parent, with its static activation scale.
        w.qtype  = QType::FP8_E4M3FN_ROW_F32;
        w.single = upload(qw::make_patterned_weight(QType::FP8_E4M3FN_ROW_F32, kRows, kHidden, seed,
                                                    {.activation_scale = 1.0F / 256.0F}));
        break;
    case Parent::Q4Q5: {
        // Group scales of a trained checkpoint's size (the unit pattern's projections would swamp the term's).
        const qw::PatternedWeightOptions scales{.row_split_scale = qw::RowSplitScalePattern::Small};
        w.qtype   = QType::Q4_G64_FP16;
        w.qk      = upload(qw::make_patterned_weight(QType::Q4_G64_FP16, kQueryRows + kKeyRows, kHidden, seed, scales));
        w.value_z =
            upload(qw::make_patterned_weight(QType::Q5_G64_FP16, kValueRows + kZRows, kHidden, seed + 1, scales));
        break;
    }
    }
    return w;
}

std::size_t projection_capacity(const ParentWeights& w, P policy, std::int32_t min_tokens, std::int32_t max_tokens) {
    if (w.two_parent()) { return 0; }
    return ops::gdn_input_proj_workspace_capacity_bytes(w.qtype, kRows, kHidden, policy, min_tokens, max_tokens);
}

void project(const ParentWeights& w, const Tensor& x, Tensor& qkv, Tensor& z, P policy, WorkspaceArena& workspace,
             const ops::LoraAddend* term) {
    if (w.two_parent()) {
        ops::gdn_input_proj(x, w.qk, w.value_z, qkv, z, policy, nullptr, term);
    } else {
        ops::gdn_input_proj(x, w.single, qkv, z, policy, workspace, nullptr, ops::LinearRoutes::Exact, term);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The term: B [16384, rank] and U [2 rank, T] (BF16). in_proj_qkv's rows take U rows 0 .. rank - 1, in_proj_z's the
// next rank; the key rows are unadapted, and the third segment ends 48 rows before the qkv/z split (inside a row
// tile), so the last value rows take z's U rows and scale across the split.
constexpr std::int32_t kValueSegmentEnd = kChannels - 48;

struct Term {
    std::int32_t rank = 32, u_rows = 0;
    std::unique_ptr<test::GuardedDeviceBuffer> b_d, u_d;
    ops::LoraAddend addend;
    // The FP64 term of every row and token (column-major), 0 on the unadapted rows; empty for a zero B.
    std::vector<double> values;

    [[nodiscard]] double value(std::int32_t row, std::int32_t token) const {
        return values[static_cast<std::size_t>(token) * kRows + row];
    }
};

[[nodiscard]] bool adapted_row(std::int32_t row) { return row < kQueryRows || row >= kQueryRows + kKeyRows; }

Term make_term(std::int32_t rank, std::uint32_t seed, bool zero_b) {
    Term term;
    term.rank   = rank;
    term.u_rows = 2 * rank;
    const auto b = zero_b ? std::vector<std::uint16_t>(static_cast<std::size_t>(kRows) * rank, 0)
                          : random_bf16(static_cast<std::size_t>(kRows) * rank, seed, -0.1F, 0.1F);
    const auto u = random_bf16(static_cast<std::size_t>(term.u_rows) * kMaxWidth, seed + 1, -3.0F, 3.0F);
    term.b_d     = device_copy(b.data(), b.size() * 2);
    term.u_d     = device_copy(u.data(), u.size() * 2);
    term.addend.b        = term.b_d->data();
    term.addend.rank     = rank;
    term.addend.u        = term.u_d->data();
    term.addend.u_rows   = term.u_rows;
    term.addend.segments = 4;
    term.addend.segment  = {ops::LoraSegment{kQueryRows, 0, 4.0F}, ops::LoraSegment{kQueryRows + kKeyRows, -1, 0.0F},
                            ops::LoraSegment{kValueSegmentEnd, 0, 4.0F}, ops::LoraSegment{kRows, rank, 3.0F}};
    if (zero_b) { return term; }
    std::vector<double> b_value(b.size()), u_value(u.size());
    for (std::size_t i = 0; i < b.size(); ++i) { b_value[i] = bf16(b[i]); }
    for (std::size_t i = 0; i < u.size(); ++i) { u_value[i] = bf16(u[i]); }
    term.values.assign(static_cast<std::size_t>(kRows) * kMaxWidth, 0.0);
    test::parallel_ranges(kMaxWidth, test::host_thread_count(), [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t token = begin; token < end; ++token) {
            int s = 0;
            for (std::int32_t row = 0; row < kRows; ++row) {
                while (s + 1 < term.addend.segments && row >= term.addend.segment[s].row_end) { ++s; }
                const ops::LoraSegment& seg = term.addend.segment[s];
                if (seg.u_offset < 0) { continue; }
                const double* b_row = &b_value[static_cast<std::size_t>(row) * rank];
                const double* u_col = &u_value[static_cast<std::size_t>(token) * term.u_rows + seg.u_offset];
                double sum          = 0.0;
                for (std::int32_t j = 0; j < rank; ++j) { sum += b_row[j] * u_col[j]; }
                term.values[static_cast<std::size_t>(token) * kRows + row] = sum * seg.scale;
            }
        }
    });
    return term;
}

// ---------------------------------------------------------------------------------------------------------------
// One projection call's outputs (column-major).
struct Outputs {
    std::vector<std::uint16_t> qkv, z;

    [[nodiscard]] std::uint16_t at(std::int32_t row, std::int32_t column) const {
        return row < kChannels ? qkv[static_cast<std::size_t>(column) * kChannels + row]
                               : z[static_cast<std::size_t>(column) * kZRows + row - kChannels];
    }
};

struct ProjectionBuffers {
    std::unique_ptr<test::GuardedDeviceBuffer> x, qkv, z;
};

ProjectionBuffers make_projection_buffers() {
    const auto x = random_bf16(static_cast<std::size_t>(kHidden) * kMaxWidth, 0x6d1U, -0.25F, 0.25F);
    ProjectionBuffers buffers;
    buffers.x   = device_copy(x.data(), x.size() * 2);
    buffers.qkv = std::make_unique<test::GuardedDeviceBuffer>(static_cast<std::size_t>(kChannels) * kMaxWidth * 2);
    buffers.z   = std::make_unique<test::GuardedDeviceBuffer>(static_cast<std::size_t>(kZRows) * kMaxWidth * 2);
    return buffers;
}

Outputs run_projection(const ParentWeights& w, ProjectionBuffers& buffers, std::int32_t t, P policy,
                       WorkspaceArena& workspace, const ops::LoraAddend* term) {
    Tensor x(buffers.x->data(), DType::BF16, {kHidden, t});
    Tensor qkv(buffers.qkv->data(), DType::BF16, {kChannels, t});
    Tensor z(buffers.z->data(), DType::BF16, {kZRows, t});
    project(w, x, qkv, z, policy, workspace, term);
    test::cuda_synchronize();
    return {read_bf16(*buffers.qkv, static_cast<std::size_t>(kChannels) * t),
            read_bf16(*buffers.z, static_cast<std::size_t>(kZRows) * t)};
}

// With-term outputs against the no-term outputs plus the FP64 term: |got - ref| <= 1e-4 + 2^-7 max(|p|, |term|, |ref|)
// (p the no-term BF16 value, ref = p + term; the BF16 roundings of p and of the sum, and the term's FP32 / BF16-U
// arithmetic). The unadapted rows keep the no-term bits. At least half the adapted values must exceed twice the
// tolerance a dropped term would get, or the check could not see one.
int check_term_outputs(const std::string& label, const Outputs& none, const Outputs& with, const Term& term,
                       std::int32_t t) {
    std::mutex mutex;
    std::int64_t failures = 0, unadapted_differ = 0, sensitive = 0, adapted = 0;
    double worst_excess = -1.0, worst_got = 0.0, worst_ref = 0.0;
    std::int64_t worst_row = 0, worst_column = 0;
    test::parallel_ranges(t, test::host_thread_count(), [&](std::int64_t begin, std::int64_t end) {
        std::int64_t local_failures = 0, local_unadapted = 0, local_sensitive = 0, local_adapted = 0;
        double local_worst = -1.0, local_got = 0.0, local_ref = 0.0;
        std::int64_t local_row = 0, local_column = 0;
        for (std::int64_t c = begin; c < end; ++c) {
            const auto column = static_cast<std::int32_t>(c);
            for (std::int32_t row = 0; row < kRows; ++row) {
                const std::uint16_t base_bits = none.at(row, column), got_bits = with.at(row, column);
                if (!adapted_row(row)) {
                    local_unadapted += base_bits != got_bits ? 1 : 0;
                    continue;
                }
                const double p = bf16(base_bits), value = term.value(row, column), ref = p + value;
                const double got   = bf16(got_bits);
                const double limit = 1.0e-4 + std::max({std::fabs(p), std::fabs(value), std::fabs(ref)}) / 128.0;
                const double excess = std::isfinite(got) ? std::fabs(got - ref) - limit : 1e300;
                local_failures += excess > 0.0 ? 1 : 0;
                local_sensitive += std::fabs(value) > 2.0 * limit ? 1 : 0;
                ++local_adapted;
                if (excess > local_worst) {
                    local_worst  = excess;
                    local_got    = got;
                    local_ref    = ref;
                    local_row    = row;
                    local_column = column;
                }
            }
        }
        const std::lock_guard<std::mutex> lock(mutex);
        failures += local_failures;
        unadapted_differ += local_unadapted;
        sensitive += local_sensitive;
        adapted += local_adapted;
        if (local_worst > worst_excess) {
            worst_excess = local_worst;
            worst_got    = local_got;
            worst_ref    = local_ref;
            worst_row    = local_row;
            worst_column = local_column;
        }
    });
    int result = 0;
    if (failures != 0) {
        std::cerr << label << ": " << failures << " of " << adapted << " adapted values outside tolerance; worst at row "
                  << worst_row << " column " << worst_column << " got " << worst_got << " reference " << worst_ref
                  << '\n';
        result = 1;
    }
    if (unadapted_differ != 0) {
        std::cerr << label << ": " << unadapted_differ << " unadapted values differ from the no-term call\n";
        result = 1;
    }
    if (2 * sensitive < adapted) {
        std::cerr << label << ": only " << sensitive << " of " << adapted
                  << " adapted values would show a dropped term; the check is insensitive\n";
        result = 1;
    }
    return result;
}

// Checks 1 and 2 on one parent: every width and policy, with the term against the no-term call (1) and a zero B
// against the no-term bits (2).
int check_terms(const ParentWeights& w, const std::vector<P>& policies, const Term& term, const Term& zero) {
    int failures     = 0;
    auto buffers     = make_projection_buffers();
    for (const P policy : policies) {
        WorkspaceArena workspace(std::max<std::size_t>(projection_capacity(w, policy, 1, kMaxWidth), 256));
        for (const std::int32_t t : kWidths) {
            const std::string label = std::string("gdn_input_proj ") + parent_name(w.parent) + " " +
                                      policy_name(policy) + " r=" + std::to_string(term.rank) +
                                      " T=" + std::to_string(t);
            const Outputs none = run_projection(w, buffers, t, policy, workspace, nullptr);
            const Outputs with = run_projection(w, buffers, t, policy, workspace, &term.addend);
            failures += check_term_outputs(label + " term", none, with, term, t);
            const Outputs zeroed = run_projection(w, buffers, t, policy, workspace, &zero.addend);
            failures += compare_bits(label + " zero-B qkv", zeroed.qkv, none.qkv);
            failures += compare_bits(label + " zero-B z", zeroed.z, none.z);
        }
    }
    return failures;
}

// Check 3: under A16ColumnInvariant every column's bits with the term are its bits in the widest call (each output is
// column-major, so a call of t columns is the widest's first rows * t values).
int check_invariance(const ParentWeights& w, const Term& term) {
    constexpr std::int32_t widths[] = {2048, 513, 300, 129, 128, 97, 65, 64, 33, 32, 24, 17, 16, 13, 12, 7, 5, 4, 2, 1};
    int failures = 0;
    auto buffers = make_projection_buffers();
    WorkspaceArena workspace(std::max<std::size_t>(projection_capacity(w, P::A16ColumnInvariant, 1, kMaxWidth), 256));
    Outputs widest;
    for (const std::int32_t t : widths) {
        const Outputs out = run_projection(w, buffers, t, P::A16ColumnInvariant, workspace, &term.addend);
        if (t == widths[0]) {
            widest = out;
            continue;
        }
        const std::string label = std::string("gdn_input_proj invariance ") + parent_name(w.parent) +
                                  " r=" + std::to_string(term.rank) + " T=" + std::to_string(t);
        failures += compare_bits(label + " qkv", out.qkv.data(), widest.qkv.data(), out.qkv.size());
        failures += compare_bits(label + " z", out.z.data(), widest.z.data(), out.z.size());
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------------------------
// Check 4: the convolution forms.
struct ConvCase {
    std::int32_t width;
    std::vector<std::int32_t> valid; // one extent a row (B = its size); empty: one dense row
    bool tree = false;               // the record's tree form (single parents)

    [[nodiscard]] std::int32_t batch() const { return valid.empty() ? 1 : static_cast<std::int32_t>(valid.size()); }
};

// Row b's tree: row 0 branches (anchor 0; main chain 0-1-2; a second path 0-3-4), the others are chains of their
// extent.
std::vector<ops::SpeculativeTreeRow> make_tree_rows(const ConvCase& c) {
    std::vector<ops::SpeculativeTreeRow> rows(static_cast<std::size_t>(c.batch()));
    for (std::int32_t b = 0; b < c.batch(); ++b) {
        ops::SpeculativeTreeRow& row = rows[static_cast<std::size_t>(b)];
        row                          = {};
        const std::int32_t nodes     = c.valid[static_cast<std::size_t>(b)];
        for (int i = 0; i < ops::kSpeculativeTreeMaxNodes; ++i) {
            row.parent[i] = row.first_child[i] = row.next_sibling[i] = -1;
            row.depth[i] = row.sibling_index[i] = 0;
        }
        row.nodes = nodes;
        row.tree  = 1;
        if (b == 0 && nodes == 5) {
            const std::int8_t parent[5] = {-1, 0, 1, 0, 3}, depth[5] = {0, 1, 2, 1, 2};
            for (int i = 0; i < 5; ++i) {
                row.parent[i] = parent[i];
                row.depth[i]  = depth[i];
            }
            row.first_child[0]   = 1;
            row.first_child[1]   = 2;
            row.first_child[3]   = 4;
            row.next_sibling[1]  = 3;
            row.sibling_index[3] = 1;
            row.main_depth       = 2;
            row.paths            = 2;
            row.path_length[0]   = 3;
            row.path_length[1]   = 3;
            row.owned_from[0]    = 0;
            row.owned_from[1]    = 1;
            const std::int8_t path0[3] = {0, 1, 2}, path1[3] = {0, 3, 4};
            for (int i = 0; i < 3; ++i) {
                row.path_columns[0][i] = path0[i];
                row.path_columns[1][i] = path1[i];
            }
            continue;
        }
        for (int i = 0; i < nodes; ++i) {
            row.parent[i]      = static_cast<std::int8_t>(i - 1);
            row.depth[i]       = static_cast<std::int8_t>(i);
            row.first_child[i] = static_cast<std::int8_t>(i + 1 < nodes ? i + 1 : -1);
            row.path_columns[0][i] = static_cast<std::int8_t>(i);
        }
        row.main_depth     = nodes - 1;
        row.paths          = 1;
        row.path_length[0] = static_cast<std::int8_t>(nodes);
    }
    return rows;
}

int check_conv_case(const ParentWeights& w, P policy, const ConvCase& c, const Term& term, std::uint32_t seed) {
    const std::int32_t width = c.width, batch = c.batch(), columns = width * batch;
    const std::int32_t slots = 2 + columns;
    const std::string label  = std::string(parent_name(w.parent)) + " " + policy_name(policy) +
                              " B=" + std::to_string(batch) + " W=" + std::to_string(width) + (c.tree ? " tree" : "");
    int failures = 0;

    const auto x_bits      = random_bf16(static_cast<std::size_t>(kHidden) * columns, seed, -0.25F, 0.25F);
    const auto conv_bits   = random_bf16(static_cast<std::size_t>(kChannels) * 4, seed + 1, -0.5F, 0.5F);
    const auto state_bits  = random_bf16(static_cast<std::size_t>(kChannels) * 3 * slots, seed + 2, -1.0F, 1.0F);
    std::vector<std::int32_t> initial(static_cast<std::size_t>(batch)), base(static_cast<std::size_t>(batch));
    for (std::int32_t b = 0; b < batch; ++b) {
        initial[static_cast<std::size_t>(b)] = b;                  // slots 0 .. B - 1
        base[static_cast<std::size_t>(b)]    = batch + b * width;  // then each row's W destinations
    }
    const auto x_d       = device_copy(x_bits.data(), x_bits.size() * 2);
    const auto conv_d    = device_copy(conv_bits.data(), conv_bits.size() * 2);
    const auto initial_d = device_copy(initial.data(), initial.size() * 4);
    const auto base_d    = device_copy(base.data(), base.size() * 4);
    std::unique_ptr<test::GuardedDeviceBuffer> valid_d, tree_d;
    if (!c.valid.empty()) { valid_d = device_copy(c.valid.data(), c.valid.size() * 4); }
    if (c.tree) {
        const std::vector<ops::SpeculativeTreeRow> tree_rows = make_tree_rows(c);
        tree_d = device_copy(tree_rows.data(), tree_rows.size() * sizeof(ops::SpeculativeTreeRow));
    }
    Tensor x(x_d->data(), DType::BF16, {kHidden, width, batch});
    Tensor conv(conv_d->data(), DType::BF16, {kChannels, 4});
    Tensor initial_slots(initial_d->data(), DType::I32, {batch});
    Tensor base_slots(base_d->data(), DType::I32, {batch});
    const Tensor valid = valid_d ? Tensor(valid_d->data(), DType::I32, {batch}) : Tensor{};
    const Tensor tree  = tree_d ? Tensor(tree_d->data(), DType::I32, {ops::kSpeculativeTreeRowWords, batch}) : Tensor{};

    struct Planes {
        std::unique_ptr<test::GuardedDeviceBuffer> states, record, q, k, v, z;
    };
    const auto make_planes = [&] {
        Planes planes;
        planes.states = device_copy(state_bits.data(), state_bits.size() * 2);
        planes.record = std::make_unique<test::GuardedDeviceBuffer>(static_cast<std::size_t>(kChannels) * columns * 2);
        planes.q      = std::make_unique<test::GuardedDeviceBuffer>(static_cast<std::size_t>(kQueryRows) * columns * 2);
        planes.k      = std::make_unique<test::GuardedDeviceBuffer>(static_cast<std::size_t>(kKeyRows) * columns * 2);
        planes.v      = std::make_unique<test::GuardedDeviceBuffer>(static_cast<std::size_t>(kValueRows) * columns * 2);
        planes.z      = std::make_unique<test::GuardedDeviceBuffer>(static_cast<std::size_t>(kZRows) * columns * 2);
        return planes;
    };
    const auto compare_planes = [&](const std::string& form, const Planes& got, const Planes& ref, bool record) {
        int f = 0;
        f += compare_bits(label + " " + form + " query", read_bf16(*got.q, static_cast<std::size_t>(kQueryRows) * columns),
                          read_bf16(*ref.q, static_cast<std::size_t>(kQueryRows) * columns));
        f += compare_bits(label + " " + form + " key", read_bf16(*got.k, static_cast<std::size_t>(kKeyRows) * columns),
                          read_bf16(*ref.k, static_cast<std::size_t>(kKeyRows) * columns));
        f += compare_bits(label + " " + form + " value", read_bf16(*got.v, static_cast<std::size_t>(kValueRows) * columns),
                          read_bf16(*ref.v, static_cast<std::size_t>(kValueRows) * columns));
        f += compare_bits(label + " " + form + " z", read_bf16(*got.z, static_cast<std::size_t>(kZRows) * columns),
                          read_bf16(*ref.z, static_cast<std::size_t>(kZRows) * columns));
        f += compare_bits(label + " " + form + " states", read_bf16(*got.states, state_bits.size()),
                          read_bf16(*ref.states, state_bits.size()));
        if (record) {
            f += compare_bits(label + " " + form + " record",
                              read_bf16(*got.record, static_cast<std::size_t>(kChannels) * columns),
                              read_bf16(*ref.record, static_cast<std::size_t>(kChannels) * columns));
            f += compare_bits(label + " " + form + " states unchanged", read_bf16(*got.states, state_bits.size()),
                              state_bits);
        }
        return f;
    };
    const auto tensor = [&](test::GuardedDeviceBuffer& buffer, std::int32_t rows) {
        return Tensor(buffer.data(), DType::BF16, {rows, width, batch});
    };
    const auto states_of = [&](test::GuardedDeviceBuffer& buffer) {
        return Tensor(buffer.data(), DType::BF16, {kChannels, 3, slots});
    };
    // The reference: the projection Op with the term over the call's columns, into `projected` ([C, W, B]: the
    // record, or the snapshot's plane), then the convolution launcher the route runs.
    const bool nvfp4_one_row = !w.two_parent() && w.qtype == QType::NVFP4 && batch == 1 && !ops::column_invariant(policy);
    const auto reference_projection = [&](Planes& ref) {
        WorkspaceArena workspace(std::max<std::size_t>(projection_capacity(w, policy, columns, columns), 256));
        Tensor x_flat(x_d->data(), DType::BF16, {kHidden, columns});
        Tensor projected_flat(ref.record->data(), DType::BF16, {kChannels, columns});
        Tensor z_flat(ref.z->data(), DType::BF16, {kZRows, columns});
        project(w, x_flat, projected_flat, z_flat, policy, workspace, &term.addend);
    };

    if (!c.tree) {
        // Snapshot.
        Planes got = make_planes(), ref = make_planes();
        {
            const std::size_t bytes =
                w.two_parent() ? ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                                     kQueryRows, kKeyRows, kValueRows, policy, batch, width, width, true)
                               : ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                                     w.qtype, kRows, kHidden, policy, batch, width, width, true);
            WorkspaceArena workspace(std::max<std::size_t>(bytes, 256));
            Tensor states = states_of(*got.states), q = tensor(*got.q, kQueryRows), k = tensor(*got.k, kKeyRows),
                   v = tensor(*got.v, kValueRows), z = tensor(*got.z, kZRows);
            if (w.two_parent()) {
                ops::gdn_input_proj_conv_snapshot(x, w.qk, w.value_z, conv, states, valid, initial_slots, base_slots, q,
                                                  k, v, z, policy, workspace, nullptr, &term.addend);
            } else {
                ops::gdn_input_proj_conv_snapshot(x, w.single, conv, states, valid, initial_slots, base_slots, q, k, v,
                                                  z, policy, workspace, nullptr, &term.addend);
            }
            test::cuda_synchronize();
        }
        {
            reference_projection(ref);
            Tensor states = states_of(*ref.states), q = tensor(*ref.q, kQueryRows), k = tensor(*ref.k, kKeyRows),
                   v = tensor(*ref.v, kValueRows);
            if (nvfp4_one_row) {
                Tensor projected(ref.record->data(), DType::BF16, {kChannels, width});
                ops::detail::nvfp4_gdn_snapshot_post_launch(projected, conv, states, valid, initial_slots, base_slots,
                                                            q, k, v, nullptr);
            } else {
                Tensor projected = tensor(*ref.record, kChannels);
                ops::detail::gdn_projected_conv_snapshot_launch(projected, conv, states, valid, initial_slots,
                                                                base_slots, q, k, v, nullptr);
            }
            test::cuda_synchronize();
        }
        failures += compare_planes("snapshot", got, ref, false);
    }

    if (width >= 2) {
        // Record (the tree form for a tree case).
        Planes got = make_planes(), ref = make_planes();
        {
            const std::size_t bytes =
                w.two_parent() ? ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                                     kQueryRows, kKeyRows, kValueRows, policy, batch, width, width, true)
                               : ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                                     w.qtype, kRows, kHidden, policy, batch, width, width, true);
            WorkspaceArena workspace(std::max<std::size_t>(bytes, 256));
            Tensor states = states_of(*got.states), record = tensor(*got.record, kChannels),
                   q = tensor(*got.q, kQueryRows), k = tensor(*got.k, kKeyRows), v = tensor(*got.v, kValueRows),
                   z = tensor(*got.z, kZRows);
            if (w.two_parent()) {
                ops::gdn_input_proj_conv_record(x, w.qk, w.value_z, conv, states, valid, initial_slots, record, q, k,
                                                v, z, policy, workspace, nullptr, &term.addend);
            } else if (c.tree) {
                ops::gdn_input_proj_conv_record(x, w.single, conv, states, valid, initial_slots, tree, record, q, k, v,
                                                z, policy, workspace, nullptr, &term.addend);
            } else {
                ops::gdn_input_proj_conv_record(x, w.single, conv, states, valid, initial_slots, record, q, k, v, z,
                                                policy, workspace, nullptr, &term.addend);
            }
            test::cuda_synchronize();
        }
        {
            reference_projection(ref);
            Tensor states = states_of(*ref.states), record = tensor(*ref.record, kChannels),
                   q = tensor(*ref.q, kQueryRows), k = tensor(*ref.k, kKeyRows), v = tensor(*ref.v, kValueRows);
            if (c.tree) {
                ops::detail::gdn_projected_conv_record_tree_launch(record, conv, states, valid, initial_slots, tree, q,
                                                                   k, v, nullptr);
            } else if (nvfp4_one_row) {
                Tensor projected(ref.record->data(), DType::BF16, {kChannels, width});
                ops::detail::nvfp4_gdn_record_post_launch(projected, conv, states, valid, initial_slots, q, k, v,
                                                          nullptr);
            } else {
                ops::detail::gdn_projected_conv_record_launch(record, conv, states, valid, initial_slots, q, k, v,
                                                              nullptr);
            }
            test::cuda_synchronize();
        }
        failures += compare_planes(c.tree ? "tree record" : "record", got, ref, true);
    }
    return failures;
}

int check_conv_forms(const ParentWeights& w, const std::vector<P>& policies, const Term& term) {
    // One dense row at the decode and verification widths (the fused routes' 1-4 and the materialized 16 and 24), and
    // two rows with a short second row (its invalid tail is zero and publishes nothing).
    const std::vector<ConvCase> cases = {{1, {}},     {2, {}},     {4, {}},           {16, {}},
                                         {24, {}},    {4, {4, 2}}, {16, {16, 9}},      {5, {5, 3}, true}};
    int failures = 0;
    std::uint32_t seed = 0x3c1U;
    for (const P policy : policies) {
        for (const ConvCase& c : cases) {
            if (c.tree && w.two_parent()) { continue; } // the tree form is the single parent's
            failures += check_conv_case(w, policy, c, term, seed += 17U);
        }
    }
    return failures;
}

// ---------------------------------------------------------------------------------------------------------------
// Check 5: refusals (host-side, before any launch).
int check_refusals(const Term& term) {
    int failures = 0;
    const auto expect_refusal = [&](const std::string& label, const auto& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            return;
        }
        std::cerr << label << ": a term was not refused\n";
        ++failures;
    };
    struct Unsupported {
        const char* name;
        QType qtype;
        std::int32_t rows, k;
    };
    for (const Unsupported& u : {Unsupported{"q8 [12288,2048]", QType::Q8_G32_FP16, 12288, 2048},
                                 Unsupported{"q8 [16384,2560]", QType::Q8_G32_FP16, 16384, 2560},
                                 Unsupported{"mxfp8 [16384,2560]", QType::FP8_E4M3FN_G32_E8M0, 16384, 2560},
                                 Unsupported{"fp8 f32 [20480,3072]", QType::FP8_E4M3FN_ROW_F32, 20480, 3072}}) {
        if (ops::gdn_input_proj_lora_supported(u.qtype, u.rows, u.k)) {
            std::cerr << u.name << ": reported as taking a term\n";
            ++failures;
        }
        Weight weight{};
        weight.qtype = u.qtype;
        weight.n     = u.rows;
        weight.k     = u.k;
        Tensor x(nullptr, DType::BF16, {u.k, 1});
        Tensor qkv(nullptr, DType::BF16, {u.rows - 4096, 1}), z(nullptr, DType::BF16, {4096, 1});
        WorkspaceArena workspace(256);
        expect_refusal(std::string(u.name) + " gdn_input_proj", [&] {
            ops::gdn_input_proj(x, weight, qkv, z, P::A16Only, workspace, nullptr, ops::LinearRoutes::Exact,
                                &term.addend);
        });
        if (u.qtype != QType::FP8_E4M3FN_G32_E8M0 && !(u.qtype == QType::Q8_G32_FP16 && u.k == 2560)) {
            expect_refusal(std::string(u.name) + " snapshot capacity", [&] {
                (void)ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(u.qtype, u.rows, u.k, P::A16Only, 1,
                                                                                 1, 4, true);
            });
            expect_refusal(std::string(u.name) + " record capacity", [&] {
                (void)ops::gdn_input_proj_conv_record_workspace_capacity_bytes(u.qtype, u.rows, u.k, P::A16Only, 1, 2,
                                                                               4, true);
            });
        }
    }
    expect_refusal("q8 [12288,2048] row-profile snapshot capacity", [&] {
        (void)ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 4096, P::A16Only, 1, 1, 4, true);
    });
    expect_refusal("q8 [12288,2048] row-profile record capacity", [&] {
        (void)ops::gdn_input_proj_conv_record_workspace_capacity_bytes(2048, 2048, 4096, P::A16Only, 1, 2, 4, true);
    });
    for (const QType qtype : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16, QType::FP8_E4M3FN_ROW_F32}) {
        if (!ops::gdn_input_proj_lora_supported(qtype, kRows, kHidden)) {
            std::cerr << "qtype " << static_cast<int>(qtype) << " [16384,5120]: reported as taking no term\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(lora_gdn) {
    if (test::cuda_unavailable()) { return 0; }
    int failures = 0;
    // Each check counts an exception as one failure and the run goes on (a refused profile names itself).
    const auto run = [&](const std::string& name, const auto& check) {
        try {
            failures += check();
        } catch (const std::exception& error) {
            std::cerr << "lora_gdn test " << name << ": " << error.what() << '\n';
            ++failures;
        }
    };
    const Term narrow = make_term(32, 0x51U, false), wide = make_term(128, 0x53U, false);
    const Term narrow_zero = make_term(32, 0x55U, true), wide_zero = make_term(128, 0x57U, true);
    run("refusals", [&] { return check_refusals(narrow); });

    struct ParentCase {
        Parent parent;
        std::vector<P> policies, conv_policies;
    };
    // The policies each parent takes (the 27B's: A16Only, AllowA4 for NVFP4, AllowA8 for FP8, A16ColumnInvariant), and
    // every other route a policy reaches: NVFP4's 8-bit tiles (A8Only) and residual profile (AllowA4Residual: A16),
    // FP8's one- and two-term 8-bit routes (A8Only, AllowA8Residual; a static scale has no residual profile).
    const ParentCase parents[] = {
        {Parent::Nvfp4,
         {P::A16Only, P::AllowA4, P::A16ColumnInvariant, P::A8Only, P::AllowA4Residual},
         {P::A16Only, P::AllowA4, P::A16ColumnInvariant, P::A8Only}},
        {Parent::Nvfp4Multiplier, {P::A16Only, P::AllowA4, P::A16ColumnInvariant}, {P::AllowA4}},
        {Parent::Fp8Bf16,
         {P::A16Only, P::AllowA8, P::A16ColumnInvariant, P::A8Only, P::AllowA8Residual},
         {P::A16Only, P::AllowA8, P::A16ColumnInvariant, P::A8Only, P::AllowA8Residual}},
        {Parent::Fp8F32,
         {P::A16Only, P::AllowA8, P::A16ColumnInvariant, P::A8Only},
         {P::A16Only, P::AllowA8, P::A16ColumnInvariant, P::A8Only}},
        {Parent::Q4Q5, {P::A16Only, P::A16ColumnInvariant}, {P::A16Only, P::A16ColumnInvariant}},
    };
    std::uint32_t seed = 0x700U;
    for (const ParentCase& parent : parents) {
        const std::string name = parent_name(parent.parent);
        ParentWeights weights;
        try {
            weights = make_parent(parent.parent, seed += 11U);
        } catch (const std::exception& error) {
            std::cerr << "lora_gdn test " << name << " weights: " << error.what() << '\n';
            ++failures;
            continue;
        }
        run(name + " terms r32", [&] { return check_terms(weights, parent.policies, narrow, narrow_zero); });
        run(name + " terms r128", [&] { return check_terms(weights, parent.policies, wide, wide_zero); });
        run(name + " invariance r32", [&] { return check_invariance(weights, narrow); });
        run(name + " invariance r128", [&] { return check_invariance(weights, wide); });
        run(name + " conv forms", [&] { return check_conv_forms(weights, parent.conv_policies, narrow); });
    }
    return failures == 0 ? 0 : 1;
}
