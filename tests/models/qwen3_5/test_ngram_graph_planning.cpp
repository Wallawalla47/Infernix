#include "core/device.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "models/qwen3_5/program/program.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string_view>

namespace {
namespace qwen = ninfer::models::qwen3_5;
using qwen::detail::mtp_graph_profiles;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

void verify_profiles() {
    for (const unsigned capacity : {2U, 128U, 4090U, 4096U, 8192U, 32768U, 262144U}) {
        for (unsigned neural = 1; neural <= 5; ++neural) {
            for (unsigned width = 1; width <= 63; ++width) {
                const auto profiles = mtp_graph_profiles(capacity, width, neural);
                require(!profiles.empty(), "valid verification width needs graph profiles");
                unsigned frontier = 0;
                std::map<unsigned, unsigned> classes;
                for (const auto& profile : profiles) {
                    require(profile.min == frontier && profile.max >= profile.min &&
                                profile.max < capacity,
                            "graph profile coverage has a gap");
                    frontier = profile.max + 1;
                    if (width != neural) {
                        require(classes.emplace(profile.topology_class, 1).second,
                                "mixed-width MTP graphs cannot alias");
                    }
                }
                require(frontier == capacity, "graph profiles must cover full capacity");
            }
        }
        for (const auto backend :
             {ninfer::SpeculativeBackend::DFlash, ninfer::SpeculativeBackend::DFlash2}) {
            for (unsigned width = 1; width <= 63; ++width) {
                const auto profiles =
                    qwen::detail::dflash_graph_profiles(backend, capacity, width, 1);
                unsigned frontier = 0;
                for (const auto& profile : profiles) {
                    require(profile.min == frontier && profile.max >= profile.min,
                            "masked draft profiles must be contiguous");
                    frontier = profile.max + 1;
                }
                require(frontier == capacity, "masked draft profiles must cover full capacity");
            }
        }
    }
    for (const auto phase : {qwen::TextPhase::Prefill, qwen::TextPhase::Verify}) {
        for (int batch : {0, 1, 2, 4, 8}) {
            for (int first : {1, 6, 16, 17, 21, 22, 24, 25, 31, 32, 33, 48, 63, 64, 65, 96}) {
                for (int last : {first, 64, 65}) {
                    const bool expected = phase == qwen::TextPhase::Verify && batch == 1 &&
                                          first >= 17 && first <= last && last <= 64;
                    require(qwen::wide_residual_verification(phase, batch, first, last) == expected,
                            "wide residual precision escaped its phase/batch/width domain");
                }
            }
        }
    }
    for (auto type :
         {ninfer::QType::FP8_E4M3FN_ROW_BF16, ninfer::QType::NVFP4, ninfer::QType::Q8_G32_FP16}) {
        for (auto policy : {ninfer::ops::LinearPolicy::A16Only, ninfer::ops::LinearPolicy::AllowA8,
                            ninfer::ops::LinearPolicy::AllowA4}) {
            qwen::execution::LinearParameters p;
            p.weight.qtype = type;
            p.policy       = policy;
            require(qwen::execution::residual_projection_policy(p, false) == policy,
                    "ordinary residual policy must remain unchanged");
            require(qwen::execution::residual_projection_policy(p, true) ==
                        (type == ninfer::QType::FP8_E4M3FN_ROW_BF16
                             ? ninfer::ops::LinearPolicy::A16Only
                             : policy),
                    "wide residual precision must be scoped to FP8");
        }
    }
}

std::size_t expected_family_bytes(unsigned capacity, unsigned verify, unsigned neural) {
    std::map<std::uint32_t, std::size_t> classes;
    for (const auto& profile : mtp_graph_profiles(capacity, verify, neural)) {
        const auto visible = std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + verify + neural);
        classes[profile.topology_class] = std::max<std::size_t>(
            classes[profile.topology_class], (visible <= 4096 ? 12ULL : 82ULL) << 20);
    }
    std::size_t total = 0;
    for (const auto& [identity, bytes] : classes) {
        (void)identity;
        total += bytes;
    }
    return total;
}

void verify_real_plan(const char* artifact) {
    ninfer::DeviceContext device;
    ninfer::models::LoadOptions selected;
    selected.vision        = false;
    selected.speculative   = ninfer::SpeculativeBackend::Mtp;
    selected.proposal_head = ninfer::ProposalHead::Optimized;
    auto model             = qwen::load_model(artifact, selected, device);
    const qwen::execution::Parameters parameters(*model);
    unsigned cases = 0;
    for (unsigned capacity : {128U, 4090U, 4096U, 8192U, 32768U, 262144U}) {
        for (unsigned neural : {1U, 2U, 3U, 4U, 5U}) {
            for (unsigned ngram : {0U, 1U, 2U, 3U, 4U, 5U, 15U, 31U, 32U, 33U, 47U, 63U}) {
                std::array<std::size_t, 2> bytes{};
                for (unsigned graphs = 0; graphs < 2; ++graphs) {
                    ninfer::EngineOptions options;
                    options.max_context     = capacity;
                    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(capacity);
                    options.max_concurrency = 1;
                    options.prefill_chunk   = 1024;
                    options.enable_vision   = false;
                    options.kv_cache        = ninfer::KvCacheStorage::Nvfp4Group16;
                    options.speculative.backend              = ninfer::SpeculativeBackend::Mtp;
                    options.speculative.draft_tokens         = neural;
                    options.speculative.ngram_draft_tokens   = ngram;
                    options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
                    options.context_cache.device_state_slots = 1;
                    options.use_cuda_graph                   = graphs != 0;
                    auto planner     = qwen::make_sequence_planner(parameters, device, options);
                    const auto pages = planner.capacity_curve().minimum_main_page_groups;
                    bytes[graphs] = std::move(planner).finalize(pages).device_reservation_bytes();
                }
                const auto expected = expected_family_bytes(capacity, neural, neural) +
                                      (ngram ? expected_family_bytes(capacity, ngram, neural) : 0);
                require(bytes[1] >= bytes[0] && bytes[1] - bytes[0] == expected,
                        "planned graph allowance does not cover both provider families");
                ++cases;
            }
        }
    }
    std::cout << "Graph planning: " << cases << " real-parameter cases passed\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            verify_profiles();
            std::cout << "Ngram graph coverage and residual policy passed\n";
            return 0;
        }
        require(argc == 2 && std::string_view(argv[1]) == "--real", "expected optional --real");
        const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
        if (!artifact || !*artifact) { return 77; }
        verify_real_plan(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
