#include "ops/offloaded_sparse_moe/cpu/a4r_expert.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace infernix::ops::offloaded_moe {
namespace {

constexpr int kTerms = canon::kA4ResidualTerms;
// Columns per rg16_row_sums call: each takes kTerms of its kMaxColumns slots, so one weight pass serves
// four columns (an eight-column job reads its units twice, the second time from cache).
constexpr int kColumnsPerPass = kMaxColumns / kTerms;
static_assert(kMaxColumns % kTerms == 0);

} // namespace

float a4r_row_scale_of(const std::uint16_t* v, int n) {
    unsigned key = 0;
    for (int j = 0; j < n; ++j) { key = std::max(key, static_cast<unsigned>(v[j] & 0x7FFFU)); }
    return canon::a4r_row_scale(key);
}

void encode_a4r_blocks(const std::uint16_t* v, int blocks, float row_scale, int block_begin, int block_end,
                       canon::A4Block* terms) {
    const float multiplier = canon::a4r_multiplier(row_scale);
    for (int b = block_begin; b < block_end; ++b) {
        canon::A4Codes codes[kTerms];
        canon::a4r_block_terms<kTerms>(v + 16 * b, multiplier, codes);
        for (int t = 0; t < kTerms; ++t) {
            terms[static_cast<std::size_t>(t) * blocks + b] = canon::a4_block_of_codes(codes[t]);
        }
    }
}

float encode_a4r(const std::uint16_t* v, int n, canon::A4Block* terms) {
    if (n <= 0 || n % 16 != 0) { throw std::invalid_argument("offloaded_moe: A4 residual rows hold whole blocks"); }
    const float row_scale = a4r_row_scale_of(v, n);
    encode_a4r_blocks(v, n / 16, row_scale, 0, n / 16, terms);
    return row_scale;
}

void rg16_row_sums_a4r(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                       const canon::A4Block* const* acts, int ncols, std::int64_t* out, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    if (rg_end - rg_begin < 1 || rg_end - rg_begin > 2) {
        throw std::invalid_argument("offloaded_moe: A4 residual row sums take one or two row groups");
    }
    const int rows = (rg_end - rg_begin) * 16;
    std::int64_t part[32 * kMaxColumns]; // no allocation on the miss path
    for (int c0 = 0; c0 < ncols; c0 += kColumnsPerPass) {
        const int n = std::min(kColumnsPerPass, ncols - c0);
        const canon::A4Block* terms[kMaxColumns];
        for (int c = 0; c < n; ++c) {
            for (int t = 0; t < kTerms; ++t) { terms[c * kTerms + t] = acts[c0 + c] + static_cast<std::size_t>(t) * blocks; }
        }
        rg16_row_sums(isa, matrix, blocks, rg_begin, rg_end, terms, n * kTerms, part, prefetch_bytes);
        // Exact integers: the term order of the sum changes no bit.
        for (int row = 0; row < rows; ++row) {
            for (int c = 0; c < n; ++c) {
                std::int64_t s = 0;
                for (int t = 0; t < kTerms; ++t) { s += part[static_cast<std::size_t>(row) * (n * kTerms) + c * kTerms + t]; }
                out[static_cast<std::size_t>(row) * ncols + c0 + c] = s;
            }
        }
    }
}

void gate_up_units_a4r(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                       const canon::A4Block* const* x, const float* x_scale, int ncols, int unit_begin, int unit_end,
                       std::uint16_t* const* h, int prefetch_bytes) {
    std::int64_t s[32 * kMaxColumns]; // one unit's sums
    for (int u = unit_begin; u < unit_end; ++u) {
        // Unit u is row groups 2u, 2u+1: rows 32u..32u+31, gate_i at even rows, up_i at odd rows.
        rg16_row_sums_a4r(isa, record, kGateUpBlocks, 2 * u, 2 * u + 2, x, ncols, s, prefetch_bytes);
        for (int c = 0; c < ncols; ++c) {
            for (int i = 0; i < 16; ++i) {
                const std::uint16_t g =
                    canon::a4r_row_output(s[static_cast<std::size_t>(2 * i) * ncols + c], x_scale[c], scales.alpha_gate);
                const std::uint16_t p =
                    canon::a4r_row_output(s[static_cast<std::size_t>(2 * i + 1) * ncols + c], x_scale[c], scales.alpha_up);
                h[c][16 * u + i] = canon::swiglu_bf16(g, p);
            }
        }
    }
}

void down_rows_a4r(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales,
                   const canon::A4Block* const* h, const float* h_scale, int ncols, int rg_begin, int rg_end,
                   std::uint16_t* const* y, int prefetch_bytes) {
    std::int64_t s[16 * kMaxColumns]; // one row group at a time
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        rg16_row_sums_a4r(isa, record + kGateUpBytes, kDownBlocks, rg, rg + 1, h, ncols, s, prefetch_bytes);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < ncols; ++c) {
                y[c][rg * 16 + r] = canon::a4r_row_output(s[r * ncols + c], h_scale[c], scales.alpha_down);
            }
        }
    }
}

void expert_forward_a4r(CpuIsa isa, const std::uint8_t* record, const ExpertScales& scales, int ncols,
                        const std::uint16_t* const* x, std::uint16_t* const* y, int prefetch_bytes) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_moe: ncols must be in [1, 8]"); }
    constexpr std::size_t kXTerms = static_cast<std::size_t>(kTerms) * kGateUpBlocks;
    constexpr std::size_t kHTerms = static_cast<std::size_t>(kTerms) * kHBlocks;
    std::vector<canon::A4Block> xt(static_cast<std::size_t>(ncols) * kXTerms);
    std::vector<canon::A4Block> ht(static_cast<std::size_t>(ncols) * kHTerms);
    std::vector<std::uint16_t> hv(static_cast<std::size_t>(ncols) * kIntermediate);
    const canon::A4Block* x_ptr[kMaxColumns];
    const canon::A4Block* h_ptr[kMaxColumns];
    std::uint16_t* hv_ptr[kMaxColumns];
    float x_scale[kMaxColumns], h_scale[kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        x_scale[c] = encode_a4r(x[c], kHidden, &xt[static_cast<std::size_t>(c) * kXTerms]);
        x_ptr[c]   = &xt[static_cast<std::size_t>(c) * kXTerms];
        hv_ptr[c]  = &hv[static_cast<std::size_t>(c) * kIntermediate];
        h_ptr[c]   = &ht[static_cast<std::size_t>(c) * kHTerms];
    }
    gate_up_units_a4r(isa, record, scales, x_ptr, x_scale, ncols, 0, kHBlocks, hv_ptr, prefetch_bytes);
    for (int c = 0; c < ncols; ++c) {
        h_scale[c] = encode_a4r(hv_ptr[c], kIntermediate, &ht[static_cast<std::size_t>(c) * kHTerms]);
    }
    down_rows_a4r(isa, record, scales, h_ptr, h_scale, ncols, 0, kDownRowGroups, y, prefetch_bytes);
}

} // namespace infernix::ops::offloaded_moe
