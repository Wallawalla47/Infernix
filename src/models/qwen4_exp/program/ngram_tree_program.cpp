// N-gram trees on Qwen3.8-Flash-Next (docs/maintainer/tree-verification.md, "N-gram trees"). QSA's block-sparse
// attention has no tree form (a tree column's visible keys, block count and the pooled key of a block completing
// inside the round all follow its own path), so a tree round verifies each root-to-leaf path of the row's tree as
// a verification row of its own over the lane's state, with every chain kernel: the rows read the lane's
// recurrent state, PLE history and QSA tails (verification leaves them in place) and the lane's KV pages, and
// each side row writes its columns' K/V and pooled keys into copies of the lane's last pages through a table row
// of its own. The tree's columns' targets are gathered from the rows' logits (a column shared by several paths
// from the first path holding it) and verified by tree acceptance, exactly as a tree row elsewhere. The accepted
// path's row then becomes row 0 of an ordinary one-row round of the round's width (its records, drafter inputs
// and, for a side row, its pages copied into place), which commit_verified commits as any other.

#include "models/qwen4_exp/program/ngram_tree_testing.h"
#include "models/qwen4_exp/program/program_impl.h"

#include "core/copy_batch.h"
#include "core/dtype.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/speculative_tree.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infernix::models::qwen4_exp::testing {
namespace {
std::atomic<bool> policy_override{false};
} // namespace
void set_ngram_policy_override(bool override_policy) noexcept { policy_override.store(override_policy); }
bool ngram_policy_override() noexcept { return policy_override.load(); }
} // namespace infernix::models::qwen4_exp::testing

namespace infernix::models::qwen4_exp::detail {

qwen3_5::detail::NgramTree ProgramImpl::ngram_tree_for(std::uint32_t index, const runtime::RoundBudget& budget) const {
    const Lane& lane = lanes_[index];
    const auto& main = drafts_[0];
    if (!lane.proposer || main.empty() || plan_.tree_paths < 2) { return {}; }
    // Depth: a draft fewer than the row may still emit (its bonus), at most the copy the round chose.
    const std::uint32_t emit = budget.generated_tokens_remaining > 1 ? budget.generated_tokens_remaining - 1U : 0U;
    const auto copy          = static_cast<std::uint32_t>(main.size());
    const auto widest        = static_cast<std::uint32_t>(max_width_);
    // The other sources, best first, to the deepest tree's depth (two paths').
    const std::uint32_t deepest = std::min({widest / 2U - 1U, emit, copy});
    if (deepest < 2) { return {}; }
    const auto history = std::span<const std::int32_t>(lane.history)
                             .last(std::min<std::size_t>(qwen3_5::kNgramMatchHistoryTokens, lane.history.size()));
    std::vector<qwen3_5::NgramMatch> candidates{qwen3_5::NgramMatch{.tokens = main, .matched = 0}};
    for (auto& other : lane.proposer->propose_candidates(history, deepest, options_.ngram_min_match,
                                                         2U * static_cast<std::uint32_t>(plan_.tree_paths))) {
        candidates.push_back(std::move(other));
    }
    if (candidates.size() < 2) { return {}; }
    // A tree's rows together verify at most a one-row round's columns (paths x its longest path <= max_width_), so
    // it needs no records, logits or io beyond a one-row round's. Of the trees of 2..tree_paths paths, the one with
    // the most expected tokens per measured round time, if it beats the copy alone (tree_score, copy_choice).
    const int bucket = ngram_bucket_[0];
    const double p   = bucket >= 0 ? lane.ngram_accept[static_cast<std::size_t>(bucket)] : 0.0;
    double best_score = testing::ngram_policy_override()
                            ? -1.0
                            : copy_tokens(p, static_cast<std::int32_t>(copy)) /
                                  cycle_seconds(CycleKind::Copy, 1 + static_cast<std::int32_t>(copy));
    qwen3_5::detail::NgramTree best;
    for (std::uint32_t paths = 2; paths <= static_cast<std::uint32_t>(plan_.tree_paths); ++paths) {
        const std::uint32_t depth = std::min({widest / paths - 1U, emit, copy,
                                              static_cast<std::uint32_t>(ops::kSpeculativeTreeMaxPathLength - 1)});
        if (depth < 2) { break; }
        const std::uint32_t nodes = std::min(static_cast<std::uint32_t>(plan_.tree_nodes), paths * (depth + 1U));
        if (nodes < depth + 2U) { continue; }
        qwen3_5::detail::NgramTree tree = qwen3_5::detail::build_ngram_tree(
            candidates, {.depth = depth, .nodes = nodes, .paths = paths}, token_domain_);
        if (!tree.tree()) { continue; }
        const double score = tree_score(lane, tree, bucket);
        if (score > best_score) {
            best_score = score;
            best       = std::move(tree);
        }
    }
    return best;
}

void ProgramImpl::promote_path_row(std::int32_t row, std::int32_t rows, std::int32_t W) {
    const cudaStream_t s = device_.stream;
    const auto layers    = static_cast<std::size_t>(c_.attention_layers);
    // QSA key records: the verification's [Di, W, rows, L] into a one-row round's [Di, W, 1, L] (layer l's row 0
    // at l W Di), through a staging copy (the layers' regions overlap in place).
    {
        auto scope            = work_->scope();
        const std::size_t run = 2ULL * static_cast<std::size_t>(di_) * static_cast<std::size_t>(W);
        Tensor staged         = work_->alloc(DType::BF16, {di_, W, dim(layers)});
        auto* records         = static_cast<std::byte*>(qsa_records_.p);
        CUDA_CHECK(cudaMemcpy2DAsync(staged.data, run, records + run * static_cast<std::size_t>(row), run * rows, run,
                                     layers, cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(records, staged.data, run * layers, cudaMemcpyDeviceToDevice, s));
    }
    // GDN replay records: the verification recorded `rows` rows of W columns a layer (layer l's row r at outer index
    // l * rows + r, verify_paths); the commit folds the one-row round's view of the storage (l * capacity). Row
    // `row` of each layer moves to its place layer by layer in increasing order: rows >= capacity, so every block
    // moves down onto storage no later layer still reads.
    const auto capacity = static_cast<std::size_t>(records_.spec.record_capacity);
    if (row != 0 || static_cast<std::size_t>(rows) != capacity) {
        const GdnReplayRecords view = records_.narrowed(W, rows);
        const auto gdn_layers       = static_cast<std::size_t>(records_.spec.layers);
        const auto verified         = static_cast<std::size_t>(rows);
        if (capacity > verified) { throw std::logic_error("Qwen4Exp: an n-gram tree round has fewer rows than the records"); }
        for (const Tensor* plane : {&view.conv, &view.key, &view.value, &view.gate}) {
            const std::size_t run = plane->bytes() / (verified * gdn_layers);
            auto* base            = static_cast<std::byte*>(plane->data);
            for (std::size_t l = 0; l < gdn_layers; ++l) {
                std::byte* to        = base + run * l * capacity;
                const std::byte* src = base + run * (l * verified + static_cast<std::size_t>(row));
                if (to != src) { CUDA_CHECK(cudaMemcpyAsync(to, src, run, cudaMemcpyDeviceToDevice, s)); }
            }
        }
    }
    if (row == 0) { return; }
    // PLE convolution inputs [width, W, rows]: row `row` onto row 0.
    {
        const std::size_t run = 2ULL * static_cast<std::size_t>(width_) * static_cast<std::size_t>(W);
        auto* base            = static_cast<std::byte*>(ple_records_.p);
        CUDA_CHECK(cudaMemcpyAsync(base, base + run * static_cast<std::size_t>(row), run, cudaMemcpyDeviceToDevice, s));
    }
    // The drafters' inputs: MTP's residual columns, DFlash2's target features (column row * W + j onto j).
    if (mtp_) {
        const std::size_t run = 2ULL * static_cast<std::size_t>(width_) * static_cast<std::size_t>(W);
        auto* base            = static_cast<std::byte*>(mtp_residuals_.p);
        CUDA_CHECK(cudaMemcpyAsync(base, base + run * static_cast<std::size_t>(row), run, cudaMemcpyDeviceToDevice, s));
    }
    if (dflash2_) {
        const Tensor to   = dflash_feature_columns(0, W);
        const Tensor from = dflash_feature_columns(row * W, W);
        CUDA_CHECK(cudaMemcpyAsync(to.data, from.data, to.bytes(), cudaMemcpyDeviceToDevice, s));
    }
}

PendingBatch ProgramImpl::verify_paths(std::span<const SequenceHandle> sequences, std::uint32_t index,
                                       std::int32_t position, const qwen3_5::detail::NgramTree& tree,
                                       Clock::time_point start) {
    const cudaStream_t s = device_.stream;
    Lane& lane           = lanes_[index];
    const ops::SpeculativeTreeRow& row = tree.row;
    const std::int32_t R = row.paths, N = static_cast<std::int32_t>(tree.columns), p = position;
    if (sequences.size() != 1 || R < 2 || R > plan_.tree_paths || N > plan_.tree_nodes || N < 3) {
        throw std::logic_error("Qwen4Exp: an n-gram tree round is outside its plan");
    }
    // The copy the round chose (drafts_[0], before the paths' drafts replace it): the tree's gain is over it.
    const auto copy_drafts = static_cast<std::int32_t>(drafts_[0].size());
    // A copy round drafted nothing it verifies (verify() consumes the same flags).
    dflash_round_       = false;
    mtp_sampled_round_  = false;
    // The round's width: its longest path (shorter rows are padded with their last draft), so every written
    // position lies within the row's reservation (reserve_units: its frontier plus max_width_).
    std::int32_t W = 2;
    for (std::int32_t r = 0; r < R; ++r) { W = std::max<std::int32_t>(W, row.path_length[r]); }
    if (R * W > max_width_) { throw std::logic_error("Qwen4Exp: an n-gram tree's rows exceed a one-row round's columns"); }

    // Each path's drafts (verify_token's rows), and its side rows' copy-on-write pages: the lane's pages of the
    // round's positions, copied to the row's own pages, which its table row maps in their place.
    for (std::int32_t r = 0; r < R; ++r) {
        auto& drafts = drafts_[static_cast<std::size_t>(r)];
        drafts.clear();
        for (std::int32_t i = 1; i < row.path_length[r]; ++i) {
            drafts.push_back(tree.drafts[static_cast<std::size_t>(row.path_columns[r][i] - 1)]);
        }
    }
    const auto first_page = static_cast<std::uint32_t>(p) / static_cast<std::uint32_t>(kPagedKVPageSize);
    const auto last_page  = static_cast<std::uint32_t>(p + W - 1) / static_cast<std::uint32_t>(kPagedKVPageSize);
    const std::uint32_t pages = last_page - first_page + 1U;
    if (first_page < lane.prefix.page_base ||
        last_page - lane.prefix.page_base >= static_cast<std::uint32_t>(lane.pages.size())) {
        throw std::logic_error("Qwen4Exp: an n-gram tree round's positions lie outside its lane's pages");
    }
    std::vector<std::pair<DeviceKVPageHandle, DeviceKVPageHandle>> cow;
    for (std::int32_t r = 1; r < R; ++r) {
        const KVExecutionRowHandle side = tree_rows_[static_cast<std::size_t>(r - 1)].handle();
        // The lane's mapping through its last page, then the side row's own pages in place of the round's.
        const Tensor from = tables_->row(lane.row.handle());
        const Tensor to   = tables_->row(side);
        CUDA_CHECK(cudaMemcpyAsync(to.data, from.data, 4ULL * (last_page + 1U), cudaMemcpyDeviceToDevice, s));
        const auto own = std::span<const DeviceKVPageLease>(tree_pages_).subspan(2U * static_cast<std::size_t>(r - 1), pages);
        tables_->publish(side, first_page, own, s);
        for (std::uint32_t k = 0; k < pages; ++k) {
            cow.emplace_back(lane.pages[first_page + k - lane.prefix.page_base].handle(), own[k].handle());
        }
    }
    (void)pool_->copy_pages(cow, s);

    // The rows' io: every row the lane's slot (its state, read in place) at the same positions; row 0 the lane's
    // table row, side row r its own.
    auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
    auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
    auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
    auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
    auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
    std::array<std::uint32_t, kNgramTreeMaximumPaths> round_lanes{};
    for (std::int32_t r = 0; r < R; ++r) {
        for (std::int32_t j = 0; j < W; ++j) {
            ids[r * W + j]     = verify_token(lane, static_cast<std::size_t>(r), p, p + j);
            pos[r * W + j]     = p + j;
            columns[r * W + j] = r * W + j;
        }
        slots[r] = static_cast<std::int32_t>(index);
        rows[r]  = r == 0 ? static_cast<std::int32_t>(index)
                          : tree_rows_[static_cast<std::size_t>(r - 1)].handle().row_index();
        host_lanes_[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(index);
        round_lanes[static_cast<std::size_t>(r)] = index;
        stage_call_rope(lane, static_cast<std::uint32_t>(p), W, R * W, r * W, R, r);
        stage_verify_ngram(lane, static_cast<std::size_t>(r), p, W, 0, W);
    }
    const std::span<const std::uint32_t> path_lanes(round_lanes.data(), static_cast<std::size_t>(R));
    // The n-gram rows are staged: the gate opens before the launch.
    publish_pinned_word(gate_ready(), next_gate_word());
    // The commit is the accepted path's one-row round, launched by commit_verified.
    if (!c_.lossless) { stage_commit_gate(1, W); }
    upload_pinned(io_device_.p, io_host_.data(), io_layout_.ngram, s);
    upload_pinned(spec_device_.p, spec_host_.data(), 4ULL * spec_layout_.licensed, s);
    if (cpu_service_) { cpu_service_->wake(); }
    settle_round();
    residency_->before_round(s, /*landing=*/true, /*walking=*/walk_.active);
    const bool screened = screened_round(path_lanes, R * W);
    {
        // Eager: a tree round is rare, and captured graphs of every row count and width cost every round (FP4-Local
        // with two lanes: 37 more graphs, 133 MiB, decode +1.3 % where no tree formed). The records view holds R rows
        // of W columns a layer, within the one-row round's storage (promote_path_row re-lays the accepted one).
        const execution::NgramRowGate gate = row_gate();
        execution::ForwardVerify view;
        view.gdn        = records_.narrowed(W, R);
        view.ple_inputs = Tensor(ple_records_.p, DType::BF16, {width_, W, R});
        view.qsa_keys   = Tensor(qsa_records_.p, DType::BF16, {di_, W, R, dim(c_.attention_layers)});
        forward_call(R, W, R * W, &view, nullptr, nullptr, &gate, false, /*decode_round=*/true, screened);
    }
    residency_->enqueue_route_download(s, R * W);

    // The tree's columns' logits: column c from the first path holding it (path q at depth d: row q's column d).
    Tensor logits = read_logits(R * W);
    std::vector<std::int32_t> source(static_cast<std::size_t>(N), -1);
    for (std::int32_t r = 0; r < R; ++r) {
        for (std::int32_t i = 0; i < row.path_length[r]; ++i) {
            auto& at = source[static_cast<std::size_t>(row.path_columns[r][i])];
            if (at < 0) { at = r * W + i; }
        }
    }
    const std::size_t column_bytes = static_cast<std::size_t>(vocab_) * dtype_size(logits.dtype);
    // The gathered columns live in the workspace until the acceptance has read them.
    auto gather_scope        = work_->scope();
    const Tensor tree_logits = work_->alloc(logits.dtype, {vocab_, N, 1});
    {
        CopyBatch gather(s);
        for (std::int32_t c = 0; c < N; ++c) {
            gather.add(static_cast<std::byte*>(tree_logits.data) + column_bytes * static_cast<std::size_t>(c),
                       static_cast<const std::byte*>(logits.data) + column_bytes * static_cast<std::size_t>(source[c]),
                       column_bytes);
        }
        gather.flush();
    }
    const TreeIo& io = plan_.tree_io;
    const auto tree_word = [&](std::size_t offset) { return static_cast<std::int32_t*>(tree_device_.p) + offset; };
    auto* host_words     = static_cast<std::int32_t*>(tree_host_.data());
    Tensor target(tree_word(io.target), DType::I32, {N, 1});
    {
        const Tensor columns_of_tree = tree_logits.view({vocab_, N});
        Tensor flat_target           = target.view({N});
        ops::argmax(columns_of_tree, flat_target, token_domain_, s);
    }

    // The row's sampler (an unconstrained row: trees never verify a grammar's or a constraint's row).
    auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
    configs[0]    = lane.sampling;
    reasoning_ban(0, configs[0]);
    upload_pinned(configs_.p, configs, sizeof(ops::SamplingConfig), s);
    // The tree, its drafts and their parents' laws, the extent, round length and anchor.
    constexpr std::int32_t C = ops::kSparseSpeculativeCandidates;
    std::copy(tree.drafts.begin(), tree.drafts.end(), host_words + io.drafts);
    std::copy(tree.candidate_ids.begin(), tree.candidate_ids.end(), host_words + io.candidates);
    std::copy(tree.proposal_q.begin(), tree.proposal_q.end(), reinterpret_cast<float*>(host_words + io.q));
    host_words[io.extents] = N - 1;
    host_words[io.lengths] = p;
    host_words[io.anchors] = lane.history.back();
    std::memcpy(host_words + io.row, &row, sizeof(ops::SpeculativeTreeRow));
    upload_pinned(tree_word(io.drafts), host_words + io.drafts, 4ULL * (io.licensed - io.drafts), s);
    Tensor drafts(tree_word(io.drafts), DType::I32, {N - 1, 1});
    Tensor candidates(tree_word(io.candidates), DType::I32, {C, N - 1, 1});
    Tensor q(tree_word(io.q), DType::FP32, {C, N - 1, 1});
    Tensor extents(tree_word(io.extents), DType::I32, {1});
    Tensor lengths(tree_word(io.lengths), DType::I32, {1});
    Tensor anchors(tree_word(io.anchors), DType::I32, {1});
    Tensor tree_rows(tree_word(io.row), DType::I32, {ops::kSpeculativeTreeRowWords, 1});
    Tensor licensed(tree_word(io.licensed), DType::I32, {N, 1});
    Tensor counts(tree_word(io.counts), DType::I32, {1});
    Tensor accepted(tree_word(io.accepted), DType::I32, {1});
    Tensor path(tree_word(io.path), DType::I32, {N, 1});
    Tensor branch(tree_word(io.branch), DType::I32, {1});
    {
        auto scope = work_->scope();
        ops::speculative_accept_sparse_tree(target, tree_logits, drafts, candidates, q, extents, tree_rows, lengths,
                                            anchors, licensed, counts, accepted, path, branch, token_domain_,
                                            static_cast<const ops::SamplingConfig*>(configs_.p), *work_, s);
    }
    // The sparse acceptance leaves token counts to the caller (the next round's penalties).
    ops::increment_licensed_token_counts(licensed, counts, static_cast<const ops::SamplingConfig*>(configs_.p), s);
    CUDA_CHECK(cudaMemcpyAsync(host_words + io.licensed, tree_word(io.licensed), 4ULL * (io.words - io.licensed),
                               cudaMemcpyDeviceToHost, s));
    device_.synchronize();
    check_device_error();

    const std::int32_t A = host_words[io.accepted], L = host_words[io.counts];
    if (L < 1 || L > W || A != L - 1) { throw std::logic_error("Qwen4Exp: tree acceptance returned an invalid extent"); }
    // The accepted path's row: the first path through the path's last column.
    const std::int32_t last = host_words[io.path + static_cast<std::size_t>(A)];
    std::int32_t taken      = -1;
    for (std::int32_t r = 0; r < R && taken < 0; ++r) {
        if (row.path_length[r] > A && row.path_columns[r][A] == last) { taken = r; }
    }
    if (taken < 0) { throw std::logic_error("Qwen4Exp: tree acceptance returned a path of no verification row"); }
    // The accepted row becomes row 0 of a one-row round: its records and drafter inputs, then for a side row its
    // pages (its columns' K/V, MTP cells and pooled keys) over the lane's; the commit reads row 0's RoPE positions.
    promote_path_row(taken, R, W);
    if (taken != 0) {
        std::vector<std::pair<DeviceKVPageHandle, DeviceKVPageHandle>> back;
        const auto own = std::span<const DeviceKVPageLease>(tree_pages_).subspan(2U * static_cast<std::size_t>(taken - 1), pages);
        for (std::uint32_t k = 0; k < pages; ++k) {
            back.emplace_back(own[k].handle(), lane.pages[first_page + k - lane.prefix.page_base].handle());
        }
        (void)pool_->copy_pages(back, s);
    }
    stage_call_rope(lane, static_cast<std::uint32_t>(p), W, W, 0, 1, 0);
    rows[0] = static_cast<std::int32_t>(index);
    upload_pinned(io_device_.p, io_host_.data(), io_layout_.ngram, s);

    // The pending round: the accepted path's tokens, its columns live in the routes (row `taken`'s columns).
    for (std::int32_t j = 0; j < W; ++j) {
        pending_tokens_[static_cast<std::size_t>(j)] = j < L ? host_words[io.licensed + static_cast<std::size_t>(j)] : 0;
    }
    std::fill_n(live_.begin(), static_cast<std::size_t>(R * W), std::uint8_t{0});
    for (std::int32_t j = 0; j <= A; ++j) { live_[static_cast<std::size_t>(taken * W + j)] = 1; }
    pending_counts_[0] = L;
    // Statistics: a copy round of the tree's drafts, of which the accepted path's.
    const std::int32_t branch_at = host_words[io.branch];
    auto& stats                  = lane.speculative;
    ++stats.rounds;
    stats.drafted_tokens += static_cast<std::uint64_t>(N - 1);
    stats.accepted_tokens += static_cast<std::uint64_t>(A);
    ++stats.ngram_rounds;
    stats.ngram_drafted_tokens += static_cast<std::uint64_t>(N - 1);
    stats.ngram_accepted_tokens += static_cast<std::uint64_t>(A);
    ++stats.ngram_tree_rounds;
    if (branch_at > 0) {
        if (branch_at > A) { throw std::logic_error("Qwen4Exp: tree acceptance returned an invalid branch"); }
        ++stats.ngram_tree_side_rounds;
        stats.ngram_tree_side_accepted_tokens += static_cast<std::uint64_t>(A - branch_at + 1);
    }
    // The copy's bucket learns its main chain's acceptance (the drafts it held before the path left it).
    const std::int32_t main_taken = branch_at > 0 ? branch_at - 1 : A;
    double accept                 = 0.0;
    if (ngram_bucket_[0] >= 0) {
        double& a = lane.ngram_accept[static_cast<std::size_t>(ngram_bucket_[0])];
        for (std::int32_t j = 0; j < row.main_depth && j <= main_taken; ++j) {
            const double hit = j < main_taken ? 1.0 : 0.0;
            a                = (1.0 - kAcceptanceWeight) * a + kAcceptanceWeight * hit;
        }
        accept = a;
    }
    // The tree's gain over the copy the round would have verified (copy_choice's): the copy alone accepts the
    // main chain's drafts up to where the path left it, or, where the path took the whole main chain, the
    // expected rest of the copy past it.
    {
        double chain = 1.0 + static_cast<double>(main_taken);
        if (branch_at <= 0 && A == row.main_depth && copy_drafts > row.main_depth) {
            chain += copy_tokens(accept, copy_drafts - row.main_depth) - 1.0;
        }
        const double gain = static_cast<double>(L) / chain;
        lane.tree_gain    = (1.0 - kAcceptanceWeight) * lane.tree_gain + kAcceptanceWeight * gain;
    }
    last_kind_    = CycleKind::Tree;
    last_columns_ = R * W;
    for (std::int32_t j = 0; j < A; ++j) { ++stats.accepted_per_position[static_cast<std::size_t>(j)]; }
    stats.round_steps.push_back(
        {.drafted = static_cast<std::uint16_t>(N - 1), .accepted = static_cast<std::uint16_t>(A), .ngram = true});
    defer_round(R * W, static_cast<std::uint32_t>(L), true, W, path_lanes);
    round_width_ = W;
    runtime::ExecutionTiming timing;
    timing.submit_host_ns = elapsed_ns(start);
    lane.decode_ns += timing.submit_host_ns;
    lane.decode_share_ns += timing.submit_host_ns;
    pending_transaction_ = ++next_transaction_;
    auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                {pending_tokens_.data(), static_cast<std::size_t>(W)}, timing,
                                                {pending_counts_.data(), 1}, static_cast<std::uint32_t>(W));
    mark_row_failures(pending, {pending_counts_.data(), 1});
    return pending;
}

} // namespace infernix::models::qwen4_exp::detail
