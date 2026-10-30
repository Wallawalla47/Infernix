// Qwen3.8-Flash-Next's CUDA graph set captured at the end of startup (candidates §23 NU-04, design §19.3.37). At
// its first use (ProgramImpl::replay, §19.3.34) a round's graph would put a 13-14 ms cudaGraphInstantiate on the
// request path of every new batch size, width and draft length. Measured (2026-10-27, ABBA 6 pairs, first request
// in a fresh process): Dense8 MTP one request decode +5.1 % (6 of 6), four requests flat, IQ3_S DFlash2 +1.8 %
// (4 of 6); startup +2.4 s (17-53 graphs), 44-76 MiB of device memory. Here every graph a round may replay is captured
// from the same body functions the rounds call (decode_graph_body, verify_graph_body, commit_body,
// mtp_catch_body, run_mtp_draft_steps, the DFlash2 proposal and context calls), instantiated and uploaded, and
// nothing runs: a graph reads its inputs (the io, the slots and rows, the commit counts, the drafter's io) by
// address when it is launched, so no lane, request or KV state is staged for it, and the first round of a shape
// launches the same executable a first-use capture would have made.

#include "models/qwen4_exp/program/graph_set.h"
#include "models/qwen4_exp/program/program_impl.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::qwen4_exp {

namespace testing {
namespace {
std::atomic<int> startup_capture_override{-1}; // -1: the switch decides
} // namespace

void set_startup_capture(std::optional<bool> capture) noexcept {
    startup_capture_override.store(capture ? (*capture ? 1 : 0) : -1);
}

std::optional<bool> startup_capture() noexcept {
    const int value = startup_capture_override.load();
    if (value < 0) { return std::nullopt; }
    return value == 1;
}
} // namespace testing

namespace detail {

std::string ProgramImpl::graph_label(const DecodeGraph& graph) const {
    // Each family's slots and its second index's stride: batch x width (decode 1), batch x draft steps.
    struct Family {
        const char* name;
        std::span<const DecodeGraph> slots;
        std::int32_t stride;
    };
    const Family families[] = {
        {"decode", graphs_, 1},
        {"screened decode", screened_graphs_, 1},
        {"verify", verify_graphs_, max_width_},
        {"screened verify", screened_verify_graphs_, max_width_},
        {"commit", commit_graphs_, max_width_},
        {"MTP draft", mtp_draft_graphs_, mtp_k_},
        {"sampled MTP draft", mtp_sampled_graphs_, mtp_k_},
        {"gated MTP draft", mtp_gated_graphs_, mtp_k_},
        {"MTP catch-up", mtp_catch_graphs_, max_width_},
        {"DFlash2 proposal", dflash_propose_graphs_, 1},
        {"DFlash2 context", dflash_context_graphs_, max_width_},
    };
    for (const Family& family : families) {
        const std::ptrdiff_t at = runtime::expert_offload::route_trace_index(family.slots, &graph);
        if (at < 0 || family.stride <= 0) { continue; }
        const auto stride = static_cast<std::ptrdiff_t>(family.stride);
        return std::string(family.name) + " " + std::to_string(at / stride + 1) + "x" + std::to_string(at % stride + 1);
    }
    return "(unknown slot)";
}

void ProgramImpl::capture_graph_set() {
    if (!testing::startup_capture().value_or(true)) { return; }
    // The lossless route runs its rounds eagerly (and takes no speculation); a scoring Program runs no rounds.
    if (c_.lossless || options_.causal_scoring) { return; }
    const auto began = std::chrono::steady_clock::now();
    device_.synchronize();
    std::size_t free_before = 0, free_after = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

    const std::int32_t lanes = plan_.lanes;
    // The rounds that verify, by rows, with their widest verification (decode()): one row up to max_width_
    // (its drafts, or a longer n-gram copy); with pair speculation and a neural drafter two rows, at one shared
    // draft length of at most kPairMaxDrafts (n-gram copies included). Every other round decodes plain.
    std::vector<std::pair<std::int32_t, std::int32_t>> verified;
    if (max_width_ > 1) {
        verified.emplace_back(1, max_width_);
        const std::int32_t pair_width = 1 + std::min(kPairMaxDrafts, neural_k());
        if (options_.pair_speculation && pair_width > 1 && lanes >= 2 && (mtp_ || dflash2_)) {
            verified.emplace_back(2, pair_width);
        }
    }

    const DecodeGraph* current = nullptr;
    const auto take = [&](DecodeGraph& graph, const std::function<void()>& body) {
        current = &graph;
        replay(graph, body);
    };
    graph_set_.capturing = true;
    try {
        // Plain rounds: every batch size (a screened round's columns at most the screened head's), and with a
        // drafter their one-column catch-up (commit(): MTP's, else DFlash2's).
        for (std::int32_t batch = 1; batch <= lanes; ++batch) {
            take(decode_graph(batch, false), [&] { decode_graph_body(batch, false); });
            if (screened_head_ && batch <= ops::kScreenedHeadMaxColumns) {
                take(decode_graph(batch, true), [&] { decode_graph_body(batch, true); });
            }
            if (mtp_) {
                take(mtp_catch_graph(batch, 1), [&] { mtp_catch_body(batch, 1); });
            } else if (dflash2_) {
                take(dflash_context_graphs_[static_cast<std::size_t>(batch - 1) * max_width_],
                     [&] { drafter_->materialize(dflash2_context_call(batch, 1)); });
            }
        }
        // Verification rounds and their commits (commit_verified): the forward, the commit graph, the MTP
        // catch-up, and with both drafters bound DFlash2's context outside the commit graph.
        for (const auto& shape : verified) {
            const std::int32_t batch = shape.first, widest = shape.second;
            for (std::int32_t W = 2; W <= widest; ++W) {
                take(verify_graph(batch, W, false), [&] { verify_graph_body(batch, W, false); });
                if (screened_head_ && batch * W <= ops::kScreenedHeadMaxColumns) {
                    take(verify_graph(batch, W, true), [&] { verify_graph_body(batch, W, true); });
                }
                take(commit_graph(batch, W), [&] { commit_body(batch, W); });
                if (mtp_) { take(mtp_catch_graph(batch, W), [&] { mtp_catch_body(batch, W); }); }
                if (mtp_ && dflash2_) {
                    take(dflash_context_graphs_[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)],
                         [&] { drafter_->materialize(dflash2_context_call(batch, W)); });
                }
            }
        }
        // The drafters: MTP's draft steps per verifying row count and length (a one-row argmax round's gated
        // chain, a sampled round's own graph), or DFlash2's proposal per verifying row count.
        for (const auto& shape : verified) {
            const std::int32_t batch = shape.first;
            if (mtp_) {
                const std::int32_t longest = batch == 1 ? mtp_k_ : std::min(kPairMaxDrafts, mtp_k_);
                for (std::int32_t steps = 1; steps <= longest; ++steps) {
                    for (const bool sampled : {false, true}) {
                        if (sampled && !sampled_drafts_) { continue; }
                        current = &mtp_draft_graph(batch, steps, sampled);
                        run_mtp_draft_steps(batch, steps, sampled);
                    }
                }
            } else if (dflash2_) {
                const execution::DFlash2ProposalCall call = dflash2_proposal_call(batch);
                take(dflash_propose_graphs_[static_cast<std::size_t>(batch - 1)], [&] { drafter_->propose(call); });
            }
        }
        device_.synchronize(); // the uploads
    } catch (...) {
        graph_set_.capturing = false;
        diagnostic("the startup capture of CUDA graph " + (current != nullptr ? graph_label(*current) : std::string("(none)")) +
                       " failed",
                   DiagnosticLevel::Error);
        throw;
    }
    graph_set_.capturing  = false;
    graph_set_.at_startup = true;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    graph_set_.bytes = static_cast<std::int64_t>(free_before) - static_cast<std::int64_t>(free_after);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    char line[256];
    std::snprintf(line, sizeof(line),
                  "CUDA graphs: captured %u at startup in %.2f s, %.0f MiB of device memory",
                  graph_set_.captured, seconds, static_cast<double>(graph_set_.bytes) / (1 << 20));
    diagnostic(line, DiagnosticLevel::Debug);
}

} // namespace detail
} // namespace infernix::models::qwen4_exp
