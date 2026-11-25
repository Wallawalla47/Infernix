#pragma once

#include "artifact/binder.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"

#include <map>
#include <span>
#include <string>
#include <vector>

namespace infernix::models::qwen3_5::loading {

[[nodiscard]] FrontendResources bind_resources(artifact::Binder& binder, const Config& config);

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

class Bindings {
public:
    // The lossless route binds every Use A16Only (LoadOptions::lossless); `a8_only` (LoadOptions::a8_dense) binds
    // the text Uses that permit 4- or 8-bit activations A8Only. `index_base`: the index of this binder's first weight
    // in the Model's weight array, where a second artifact's weights (a DSpine drafter's) follow the main artifact's.
    Bindings(artifact::Binder& binder, bool a16_only, bool a8_only = false, std::size_t index_base = 0)
        : binder(binder), a16_only(a16_only), a8_only(a8_only), index_base(index_base) {}

    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {},
                                     artifact::Residency residency = artifact::Residency::Device);
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  QType format = QType::BF16,
                                  artifact::Residency residency = artifact::Residency::Device);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index - index_base); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    artifact::Binder& binder;
    bool a16_only = false;
    bool a8_only  = false;
    std::size_t index_base = 0;
    std::vector<PendingWeight> weights;

private:
    std::map<std::string, WeightId, std::less<>> parameters_;
};

[[nodiscard]] AttentionWeights bind_attention(Bindings& bindings, const TextConfig& config,
                                              const std::string& prefix);
[[nodiscard]] DenseWeights bind_dense(Bindings& bindings, std::uint64_t hidden,
                                      std::uint64_t intermediate, const std::string& prefix,
                                      bool draft = false);
// `stream_experts`: an offloaded bank stays in the artifact (LoadOptions::stream_experts).
// `resident_experts`: a resident MoE's routed experts' residency (HostPinned: the MTP drafter behind the drafter
// pool, Config::mtp_expert_slots).
[[nodiscard]] BlockWeights bind_block(Bindings& bindings, const TextConfig& config,
                                      const std::string& prefix, MixerKind mixer,
                                      bool stream_experts                = false,
                                      artifact::Residency resident_experts = artifact::Residency::Device);
// Whether the block at `prefix` stores its routed experts as one bank (`moe/experts`), which binds
// the offloaded form.
[[nodiscard]] bool offloaded_expert_bank(const Bindings& bindings, const std::string& prefix);
[[nodiscard]] TextWeights bind_text(Bindings& bindings, const TextConfig& config,
                                    const LoadOptions& options);
[[nodiscard]] VisionWeights bind_vision(Bindings& bindings, const VisionConfig& config,
                                        const TextConfig& target, const LoadOptions& options);
[[nodiscard]] MtpWeights bind_mtp(Bindings& bindings, const TextConfig& config,
                                  const TextWeights& target,
                                  artifact::Residency experts = artifact::Residency::Device);
[[nodiscard]] DraftWeights bind_draft(Bindings& bindings, const DraftConfig& config,
                                      const TextConfig& target, const TextWeights& weights,
                                      const std::string& component);
void bind_dflash2(Bindings& bindings, DraftWeights& weights, const DraftConfig& config,
                  const TextConfig& target);
// The DSpine drafter's code bridge, from the drafter artifact's binder (`bindings`).
void bind_dspine(Bindings& bindings, DraftWeights& weights, const DraftConfig& config,
                 const TextConfig& target);
// The input through which a DSpine drafter reads the target's head: the artifact's DFlash2 drafter's
// use of it where declared, else the text's (the drafter artifact declares no use of the target's head).
[[nodiscard]] std::string dspine_head_input(const artifact::Directory& target);
[[nodiscard]] ProposalWeights bind_proposal(Bindings& bindings, const artifact::Proposal& proposal,
                                            const TextConfig& target, const LoadOptions& options,
                                            std::uint32_t public_tokens);
[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized);

} // namespace infernix::models::qwen3_5::loading
