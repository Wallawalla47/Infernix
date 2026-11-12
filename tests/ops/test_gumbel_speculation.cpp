// Gumbel-max sampling (SamplingConfig::gumbel_max, the reproducible and lossless routes): a sampled
// request's speculative verification returns exactly the tokens plain sampling draws at the same
// positions, through the chain acceptance (single-block and multi-block routes) and the sparse
// acceptance; and the draws follow the truncated distribution (top-k, top-p).
//
// Plain sampling draws column c of a row with logical position (the input token's) L + c under
// kSamplePurposeDecode; the verification of a round with frontier L keys column c at L + c + 1. Both
// name the produced token's position, so the same Gumbel noise decides both.

#include "ops/op_family.h"
#include "infernix/ops/candidate_selector.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_round.h"
#include "ops/op_tester.h"
#include "core/device.h"

#include <algorithm>
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

float bf16_round(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    u = ((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16) << 16;
    float r;
    std::memcpy(&r, &u, 4);
    return r;
}

ops::SamplingConfig gumbel_config(unsigned long long seed) {
    ops::SamplingConfig c;
    c.temperature = 0.8F;
    c.top_k       = 20;
    c.top_p       = 0.95F;
    c.seed        = seed;
    c.gumbel_max  = true;
    return c;
}

// Plain sampling of one [rows, B] column set over the first `vocab` rows: every row's token at its logical
// position.
std::vector<std::int32_t> plain(const std::vector<float>& column_logits, int rows, int vocab, int batch,
                                const std::vector<ops::SamplingConfig>& configs, const std::vector<std::int32_t>& positions) {
    DeviceBuffer logits = to_device_bf16(column_logits);
    DeviceBuffer cfg    = to_device(configs);
    DeviceBuffer pos    = to_device(positions);
    DeviceBuffer out(static_cast<std::size_t>(batch) * 4);
    Tensor tl(logits.p, DType::BF16, {rows, batch}), to(out.p, DType::I32, {batch}), tp(pos.p, DType::I32, {batch});
    DeviceBuffer scratch(std::max<std::size_t>(ops::sampling_workspace_capacity_bytes(vocab, batch, batch), 1));
    WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
    ops::sample(tl, to, vocab, static_cast<const ops::SamplingConfig*>(cfg.p), tp, ops::kSamplePurposeDecode, workspace,
                nullptr);
    cuda_synchronize();
    return from_device<std::int32_t>(out, static_cast<std::size_t>(batch));
}

// Chain verification of K drafts per row against the plain tokens: row b's first accept[b] drafts are
// the plain tokens, the next (if any) is not; the round must license plain's tokens. The logits have
// `rows` rows, of which the first `vocab` are the token domain.
int verification_case(int vocab, int rows, int k, int batch, bool sparse, std::mt19937& rng) {
    const int cols = k + 1;
    std::normal_distribution<float> normal(0.0F, 3.0F);
    std::vector<float> logits(static_cast<std::size_t>(rows) * cols * batch);
    for (auto& v : logits) { v = bf16_round(normal(rng)); }
    std::vector<ops::SamplingConfig> configs;
    std::vector<std::int32_t> lengths(batch);
    for (int b = 0; b < batch; ++b) {
        configs.push_back(gumbel_config(1000 + 17 * b));
        lengths[b] = 300 + 41 * b;
    }
    // Plain tokens per column (input position L + c).
    std::vector<std::vector<std::int32_t>> y(batch, std::vector<std::int32_t>(cols));
    for (int c = 0; c < cols; ++c) {
        std::vector<float> column(static_cast<std::size_t>(rows) * batch);
        std::vector<std::int32_t> positions(batch);
        for (int b = 0; b < batch; ++b) {
            std::copy_n(logits.begin() + (static_cast<std::size_t>(b) * cols + c) * rows, rows,
                        column.begin() + static_cast<std::size_t>(b) * rows);
            positions[b] = lengths[b] + c;
        }
        const auto tokens = plain(column, rows, vocab, batch, configs, positions);
        for (int b = 0; b < batch; ++b) { y[b][c] = tokens[b]; }
    }
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(k) * batch), extents(batch, k), accept(batch);
    for (int b = 0; b < batch; ++b) {
        accept[b] = b % 3 == 0 ? 0 : (b % 3 == 1 ? k / 2 : k);
        for (int j = 0; j < k; ++j) {
            drafts[static_cast<std::size_t>(b) * k + j] =
                j < accept[b] ? y[b][j] : (j == accept[b] ? (y[b][j] + 1) % vocab : static_cast<std::int32_t>(rng() % vocab));
        }
    }
    std::vector<std::int32_t> candidates(static_cast<std::size_t>(ops::kSparseSpeculativeCandidates) * k * batch);
    std::vector<float> q(candidates.size(), 0.0F);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const std::size_t draft = i / ops::kSparseSpeculativeCandidates;
        const int slot          = static_cast<int>(i % ops::kSparseSpeculativeCandidates);
        candidates[i]           = slot == 0 ? drafts[draft] : static_cast<std::int32_t>((drafts[draft] + 1 + slot) % vocab);
        q[i]                    = slot == 0 ? 0.6F : 0.4F / (ops::kSparseSpeculativeCandidates - 1);
    }
    DeviceBuffer d_logits = to_device_bf16(logits), d_drafts = to_device(drafts), d_extents = to_device(extents);
    DeviceBuffer d_lengths = to_device(lengths), d_cfg = to_device(configs);
    DeviceBuffer d_targets = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(cols) * batch, 0));
    DeviceBuffer d_cand = to_device(candidates), d_q = to_device(q);
    DeviceBuffer d_anchors(static_cast<std::size_t>(batch) * 4), d_licensed(static_cast<std::size_t>(cols) * batch * 4),
        d_counts(static_cast<std::size_t>(batch) * 4), d_accepted(static_cast<std::size_t>(batch) * 4);
    Tensor tt(d_targets.p, DType::I32, {cols, batch}), tl(d_logits.p, DType::BF16, {rows, cols, batch});
    Tensor td(d_drafts.p, DType::I32, {k, batch}), te(d_extents.p, DType::I32, {batch});
    Tensor tlen(d_lengths.p, DType::I32, {batch}), ta(d_anchors.p, DType::I32, {batch});
    Tensor tlic(d_licensed.p, DType::I32, {cols, batch}), tc(d_counts.p, DType::I32, {batch});
    Tensor tacc(d_accepted.p, DType::I32, {batch});
    Tensor tcand(d_cand.p, DType::I32, {ops::kSparseSpeculativeCandidates, k, batch});
    Tensor tq(d_q.p, DType::FP32, {ops::kSparseSpeculativeCandidates, k, batch});
    const ops::SpeculativeAcceptExecutionEnvelope envelope{false};
    const std::size_t bytes =
        sparse ? ops::speculative_accept_sparse_drafts_workspace_capacity_bytes(vocab, envelope, k, k, batch, batch)
               : ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(vocab, k, k, batch, batch);
    DeviceBuffer scratch(std::max<std::size_t>(bytes, 1));
    WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
    const auto* cfg = static_cast<const ops::SamplingConfig*>(d_cfg.p);
    if (sparse) {
        ops::speculative_accept_sparse_drafts(tt, tl, td, tcand, tq, te, tlen, ta, tlic, tc, tacc, vocab, cfg, envelope,
                                              workspace, nullptr);
    } else {
        ops::speculative_accept_greedy_drafts(tt, tl, td, te, tlen, ta, tlic, tc, tacc, vocab, cfg, workspace, nullptr);
    }
    cuda_synchronize();
    const auto licensed = from_device<std::int32_t>(d_licensed, static_cast<std::size_t>(cols) * batch);
    const auto accepted = from_device<std::int32_t>(d_accepted, static_cast<std::size_t>(batch));
    const auto out_len  = from_device<std::int32_t>(d_lengths, static_cast<std::size_t>(batch));
    int failures = 0;
    for (int b = 0; b < batch; ++b) {
        bool ok = accepted[b] == accept[b] && out_len[b] == lengths[b] + accept[b] + 1;
        for (int j = 0; j <= accept[b]; ++j) { ok = ok && licensed[static_cast<std::size_t>(b) * cols + j] == y[b][j]; }
        if (!ok) {
            ++failures;
            std::cerr << "FAIL " << (sparse ? "sparse" : "chain") << " V=" << vocab << " row " << b << ": accepted "
                      << accepted[b] << " (want " << accept[b] << ")\n";
        }
    }
    std::cout << (sparse ? "sparse" : "chain") << " verification V=" << vocab << " K=" << k << " B=" << batch << ": "
              << (failures == 0 ? "plain tokens" : "MISMATCH") << "\n";
    return failures;
}

// The draws follow the truncated distribution: one column, many seeds.
int distribution_case(std::mt19937& rng) {
    constexpr int kVocab = 200, kSeeds = 20000;
    std::normal_distribution<float> normal(0.0F, 1.5F);
    std::vector<float> column(kVocab);
    for (auto& v : column) { v = bf16_round(normal(rng)); }
    std::vector<float> logits(static_cast<std::size_t>(kVocab) * kSeeds);
    std::vector<ops::SamplingConfig> configs(kSeeds);
    std::vector<std::int32_t> positions(kSeeds, 77);
    for (int s = 0; s < kSeeds; ++s) {
        std::copy(column.begin(), column.end(), logits.begin() + static_cast<std::size_t>(s) * kVocab);
        configs[s] = gumbel_config(static_cast<unsigned long long>(s) * 7919ULL + 1);
    }
    const auto tokens = plain(logits, kVocab, kVocab, kSeeds, configs, positions);
    // The expected support: the top 20 by logit, weights exp((x - x0) / T), the shortest prefix whose
    // weight reaches top_p of the total, renormalized.
    std::vector<int> order(kVocab);
    for (int i = 0; i < kVocab; ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return column[a] > column[b] || (column[a] == column[b] && a < b); });
    std::vector<double> w(20);
    double total = 0.0;
    for (int j = 0; j < 20; ++j) {
        w[j] = std::exp((column[order[j]] - column[order[0]]) / 0.8);
        total += w[j];
    }
    int support = 0;
    double cum = 0.0;
    for (int j = 0; j < 20; ++j) {
        cum += w[j];
        support = j + 1;
        if (cum >= 0.95 * total) { break; }
    }
    double mass = 0.0;
    for (int j = 0; j < support; ++j) { mass += w[j]; }
    std::vector<int> count(kVocab, 0);
    for (const int t : tokens) { ++count[t]; }
    double worst = 0.0;
    int outside = 0;
    for (int v = 0; v < kVocab; ++v) {
        const int rank = static_cast<int>(std::find(order.begin(), order.end(), v) - order.begin());
        const double p = rank < support ? w[rank] / mass : 0.0;
        if (p == 0.0) {
            outside += count[v];
            continue;
        }
        worst = std::max(worst, std::fabs(count[v] / static_cast<double>(kSeeds) - p));
    }
    std::cout << "distribution: support " << support << ", largest frequency error " << worst << ", draws outside "
              << outside << "\n";
    if (outside != 0 || worst > 0.015) {
        std::cerr << "FAIL Gumbel-max draws do not follow the truncated distribution\n";
        return 1;
    }
    return 0;
}

// Coupled drafts (candidates ZH-01): candidate_selector_path's gumbel_max rows draw draft i with the Gumbel
// key its verification column gives each candidate, (seed, L + i + 1, token). With a drafter whose law at every
// position equals the target's (zero codebooks, so edge = unary score; the target's logits are those scores on
// the same 16 candidates, its support those 16), every draft must then be the target's own draw: the chain
// accepts all K. The same drafter drawing with its proposal uniform (a row without gumbel_max at the selector)
// is a draw from the same law that the verification accepts only by chance; its count is printed for
// comparison, not checked.
int coupled_case(int k, int batch, std::mt19937& rng) {
    constexpr int kRows = 248320, kVocab = 248077, kRank = 256, kCandidates = 16;
    std::normal_distribution<float> normal(0.0F, 1.5F);
    const std::size_t lattice = static_cast<std::size_t>(kCandidates) * k * batch;
    std::vector<std::int32_t> ids(lattice);
    std::vector<float> unary(lattice);
    for (std::size_t column = 0; column < static_cast<std::size_t>(k) * batch; ++column) {
        const auto first = static_cast<std::int32_t>(rng() % (kVocab - 40 * kCandidates));
        for (int c = 0; c < kCandidates; ++c) {
            ids[column * kCandidates + c]   = first + 37 * c;
            unary[column * kCandidates + c] = bf16_round(normal(rng));
        }
    }
    std::vector<ops::SamplingConfig> coupled, plain_rows;
    std::vector<std::int32_t> lengths(batch), anchors(batch, 5);
    for (int b = 0; b < batch; ++b) {
        ops::SamplingConfig c = gumbel_config(4000 + 29 * b);
        c.top_k               = kCandidates; // the target's support: exactly the drafter's candidates
        c.top_p               = 1.0F;
        coupled.push_back(c);
        c.gumbel_max = false;
        plain_rows.push_back(c);
        lengths[b] = 500 + 13 * b;
    }
    // The target's logits: -30 off the candidates, the candidates' scores at their column; the bonus
    // column repeats the last position's.
    const int cols = k + 1;
    std::vector<float> logits(static_cast<std::size_t>(kRows) * cols * batch, -30.0F);
    for (int b = 0; b < batch; ++b) {
        for (int column = 0; column < cols; ++column) {
            const std::size_t at = static_cast<std::size_t>(b) * k + std::min(column, k - 1);
            for (int c = 0; c < kCandidates; ++c) {
                logits[(static_cast<std::size_t>(b) * cols + column) * kRows + ids[at * kCandidates + c]] =
                    unary[at * kCandidates + c];
            }
        }
    }
    DeviceBuffer d_ids = to_device(ids), d_unary = to_device(unary), d_anchors = to_device(anchors);
    DeviceBuffer d_base = to_device(lengths);
    DeviceBuffer d_hidden(static_cast<std::size_t>(kRank) * k * batch * 2);
    DeviceBuffer d_predecessor(static_cast<std::size_t>(kRank) * kRows * 2), d_successor(d_predecessor.bytes);
    cuda_check(cudaMemset(d_hidden.p, 0, d_hidden.bytes), "cudaMemset");
    cuda_check(cudaMemset(d_predecessor.p, 0, d_predecessor.bytes), "cudaMemset");
    cuda_check(cudaMemset(d_successor.p, 0, d_successor.bytes), "cudaMemset");
    DeviceBuffer d_logits = to_device_bf16(logits);
    const auto accepted_with = [&](const std::vector<ops::SamplingConfig>& selector_configs) {
        DeviceBuffer d_sel = to_device(selector_configs), d_cfg = to_device(coupled);
        DeviceBuffer d_drafts(static_cast<std::size_t>(k) * batch * 4), d_q(lattice * 4);
        Tensor tids(d_ids.p, DType::I32, {kCandidates, k, batch}), tunary(d_unary.p, DType::FP32, {kCandidates, k, batch});
        Tensor thidden(d_hidden.p, DType::BF16, {kRank, k, batch}), tanchors(d_anchors.p, DType::I32, {batch});
        Tensor tpred(d_predecessor.p, DType::BF16, {kRank, kRows}), tsucc(d_successor.p, DType::BF16, {kRank, kRows});
        Tensor tbase(d_base.p, DType::I32, {batch});
        Tensor tdrafts(d_drafts.p, DType::I32, {k, batch}), tq(d_q.p, DType::FP32, {kCandidates, k, batch});
        DeviceBuffer selector_scratch(std::max<std::size_t>(ops::candidate_selector_path_workspace_capacity_bytes(k, k, batch, batch), 1));
        WorkspaceArena selector_workspace(DeviceSpan{selector_scratch.p, selector_scratch.bytes});
        ops::candidate_selector_path(tids, tunary, thidden, tanchors, tpred, tsucc, tbase,
                                     static_cast<const ops::SamplingConfig*>(d_sel.p), tdrafts, tq, nullptr,
                                     selector_workspace, nullptr);
        DeviceBuffer d_extents = to_device(std::vector<std::int32_t>(batch, k)), d_lengths = to_device(lengths);
        DeviceBuffer d_targets = to_device(std::vector<std::int32_t>(static_cast<std::size_t>(cols) * batch, 0));
        DeviceBuffer d_round_anchors(static_cast<std::size_t>(batch) * 4), d_licensed(static_cast<std::size_t>(cols) * batch * 4),
            d_counts(static_cast<std::size_t>(batch) * 4), d_accepted(static_cast<std::size_t>(batch) * 4);
        Tensor tt(d_targets.p, DType::I32, {cols, batch}), tl(d_logits.p, DType::BF16, {kRows, cols, batch});
        Tensor te(d_extents.p, DType::I32, {batch}), tlen(d_lengths.p, DType::I32, {batch});
        Tensor ta(d_round_anchors.p, DType::I32, {batch}), tlic(d_licensed.p, DType::I32, {cols, batch});
        Tensor tc(d_counts.p, DType::I32, {batch}), tacc(d_accepted.p, DType::I32, {batch});
        DeviceBuffer scratch(std::max<std::size_t>(
            ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(kVocab, k, k, batch, batch), 1));
        WorkspaceArena workspace(DeviceSpan{scratch.p, scratch.bytes});
        ops::speculative_accept_greedy_drafts(tt, tl, tdrafts, te, tlen, ta, tlic, tc, tacc, kVocab,
                                              static_cast<const ops::SamplingConfig*>(d_cfg.p), workspace, nullptr);
        cuda_synchronize();
        return from_device<std::int32_t>(d_accepted, static_cast<std::size_t>(batch));
    };
    const auto coupled_accepted = accepted_with(coupled);
    const auto plain_accepted   = accepted_with(plain_rows);
    int failures = 0, coupled_total = 0, plain_total = 0;
    for (int b = 0; b < batch; ++b) {
        coupled_total += coupled_accepted[b];
        plain_total += plain_accepted[b];
        if (coupled_accepted[b] != k) {
            ++failures;
            std::cerr << "FAIL coupled drafts row " << b << ": accepted " << coupled_accepted[b] << " of " << k << "\n";
        }
    }
    std::cout << "coupled drafts K=" << k << " B=" << batch << ": accepted " << coupled_total << " of " << k * batch
              << " (proposal-uniform drafts of the same law: " << plain_total << ")\n";
    return failures;
}

} // namespace

INFERNIX_OP_TEST(gumbel_speculation) {
    if (cuda_unavailable()) {
        std::cout << "gumbel_speculation: SKIP (CUDA unavailable)\n";
        return 77;
    }
    std::mt19937 rng(20261021);
    int failures = 0;
    try {
        // The sparse acceptance serves Qwen's public token domain (248,077 of 248,320 logit rows) only.
        failures += verification_case(200, 200, 4, 6, false, rng);
        failures += verification_case(248077, 248320, 4, 6, false, rng);
        failures += verification_case(248077, 248320, 7, 3, true, rng);
        failures += distribution_case(rng);
        failures += coupled_case(7, 6, rng);
        failures += coupled_case(15, 2, rng);
    } catch (const std::exception& e) {
        std::cerr << "FAIL exception: " << e.what() << "\n";
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "gumbel_speculation: verification returns plain sampling's tokens\n";
    return 0;
}
