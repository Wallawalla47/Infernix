#pragma once

// Expert records and activations shared by the CPU and GPU narrow-route tests of
// offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md §16.2).

#include "ops/common/canonical_math.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace ninfer::test::offloaded_moe {

namespace canon = ninfer::ops::canon;
namespace moe   = ninfer::ops::offloaded_moe;

inline std::uint16_t to_bf16(float x) { return canon::f32_to_bf16_rn(x); }

struct Expert {
    std::vector<std::uint8_t> record;
    moe::ExpertScales scales;
};

// Random codes; scale bytes in the realistic range plus extremes (zero, subnormal, 448).
inline Expert random_expert(std::mt19937& rng, bool split_input_scales) {
    Expert e;
    e.record.resize(moe::kRecordBytes);
    std::uniform_int_distribution<int> byte(0, 255);
    for (std::size_t i = 0; i < e.record.size(); ++i) { e.record[i] = static_cast<std::uint8_t>(byte(rng)); }
    std::uniform_int_distribution<int> scale(40, 80), rare(0, 99);
    auto fix_scales = [&](std::size_t begin, int row_groups, int blocks) {
        for (int u = 0; u < row_groups * blocks; ++u) {
            std::uint8_t* s = &e.record[begin + static_cast<std::size_t>(u) * moe::kUnitBytes + 128];
            for (int r = 0; r < 16; ++r) {
                const int pick = rare(rng);
                s[r] = static_cast<std::uint8_t>(pick == 0 ? 0 : pick == 1 ? 3 : pick == 2 ? 0x7E : scale(rng));
            }
        }
    };
    fix_scales(0, moe::kGateUpRowGroups, moe::kGateUpBlocks);
    fix_scales(moe::kGateUpBytes, moe::kDownRowGroups, moe::kDownBlocks);
    std::uniform_real_distribution<float> lg(-11.0F, -7.0F);
    e.scales.input_gate = std::exp2(lg(rng));
    e.scales.input_up   = split_input_scales ? std::exp2(lg(rng)) : e.scales.input_gate;
    e.scales.input_down = std::exp2(lg(rng));
    // alpha = fl32(weight_scale_2 * input_scale), as the loader derives it.
    e.scales.alpha_gate = std::exp2(lg(rng)) * e.scales.input_gate;
    e.scales.alpha_up   = std::exp2(lg(rng)) * e.scales.input_up;
    e.scales.alpha_down = std::exp2(lg(rng)) * e.scales.input_down;
    return e;
}

inline std::vector<std::uint16_t> random_activations(std::mt19937& rng, int ncols) {
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(ncols) * moe::kHidden);
    for (auto& v : x) { v = to_bf16(n(rng) * 0.05F); }
    // An outlier per column, as real activations have.
    for (int c = 0; c < ncols; ++c) { x[static_cast<std::size_t>(c) * moe::kHidden + 17 * (c + 1)] = to_bf16(9.0F); }
    return x;
}

// ---------------------------------------------------------------------------- golden case

// A fixed expert and input built from an integer generator (portable across standard libraries).
// Every CPU ISA, every compiler and the GPU narrow route must reproduce these output hashes.
inline constexpr std::uint64_t kGolden1 = 0x8e4b822b07293949ULL; // ncols = 1
inline constexpr std::uint64_t kGolden4 = 0x947202285aeb28bdULL; // ncols = 4

inline std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct GoldenCase {
    Expert expert;
    std::vector<std::uint16_t> x; // [ncols][kHidden]
};

inline GoldenCase golden_case(int ncols) {
    std::uint64_t s = 2026;
    GoldenCase g;
    Expert& e = g.expert;
    e.record.resize(moe::kRecordBytes);
    for (auto& b : e.record) { b = static_cast<std::uint8_t>(splitmix(s)); }
    auto fix = [&](std::size_t begin, int units) {
        for (int u = 0; u < units; ++u) {
            for (int r = 0; r < 16; ++r) {
                e.record[begin + static_cast<std::size_t>(u) * moe::kUnitBytes + 128 + r] =
                    static_cast<std::uint8_t>(32 + splitmix(s) % 64);
            }
        }
    };
    fix(0, moe::kGateUpRowGroups * moe::kGateUpBlocks);
    fix(moe::kGateUpBytes, moe::kDownRowGroups * moe::kDownBlocks);
    e.scales = {0.00390625F * 0.75F, 0.00390625F * 0.75F, 0.0078125F * 0.625F, 3.0e-6F, 3.0e-6F, 5.0e-6F};
    g.x.resize(static_cast<std::size_t>(ncols) * moe::kHidden);
    for (auto& v : g.x) {
        // BF16 with exponent in [2^-9, 2^-2] and a random sign and mantissa.
        const std::uint64_t r = splitmix(s);
        v = static_cast<std::uint16_t>(((r & 1) << 15) | ((118 + (r >> 1) % 8) << 7) | ((r >> 8) & 0x7F));
    }
    return g;
}

// FNV-1a over the BF16 outputs.
inline std::uint64_t output_hash(const std::vector<std::uint16_t>& y) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::uint16_t v : y) { h = (h ^ v) * 1099511628211ULL; }
    return h;
}

} // namespace ninfer::test::offloaded_moe
