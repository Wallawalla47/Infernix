#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/load/vision_overlay.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace infernix::models::qwen3_5 {

// Evictable ranks: higher ranks sit closer to the arena end and are evicted first, so the
// low-traffic endpoints (MTP, draft head, embedding, LM head) form the borrowable tail that
// an overlay vision window streams the vision tower through.
constexpr std::uint32_t kEvictRankMtp        = 400;
constexpr std::uint32_t kEvictRankDraftHead  = 500;
constexpr std::uint32_t kEvictRankEmbedding  = 600;
constexpr std::uint32_t kEvictRankLmHead     = 700;

std::uint32_t overlay_evict_rank(std::string_view name) {
    if (name == "text/token_embedding") { return kEvictRankEmbedding; }
    if (name == "text/output_head") { return kEvictRankLmHead; }
    if (name == "proposal/head") { return kEvictRankDraftHead; }
    if (name.rfind("mtp/", 0) == 0) { return kEvictRankMtp; }
    return 0;
}

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    std::optional<VisionOverlayLayout> vision_overlay;
    FrontendResources resources;
    InstanceInfo info;
    std::uint64_t embedding_bytes = 0; // offloaded experts: the token embedding's stored bytes (pinned)
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

void LoadPlan::set_host_reserve(std::uint64_t reserve_bytes, std::uint64_t later_pinned_bytes) {
    impl_->materialization.host_reserve_bytes = reserve_bytes;
    impl_->materialization.later_pinned_bytes = later_pinned_bytes;
}

bool LoadPlan::offloaded_experts() const { return impl_->config.text.offloaded_experts; }

std::uint64_t LoadPlan::pinned_other_bytes() const {
    std::uint64_t bytes = impl_->embedding_bytes;
    if (impl_->vision_overlay) {
        const auto& v = *impl_->vision_overlay;
        bytes += v.prelude.bytes + v.merger.bytes;
        for (const auto& layer : v.layers) { bytes += layer.bytes; }
    }
    return bytes;
}

std::uint64_t LoadPlan::pinned_expert_bytes() const {
    const std::uint64_t pinned = impl_->materialization.pinned_capacity_bytes;
    const std::uint64_t other  = pinned_other_bytes();
    return pinned > other ? pinned - other : 0;
}

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const std::optional<VisionOverlayLayout>& LoadPlan::vision_overlay_layout() const {
    return impl_->vision_overlay;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    // The lossless route binds every Use A16Only; the design reference keeps the declared permissions.
    loading::Bindings bindings(binder, options.lossless && !options.reference);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    // Offloaded experts are all or nothing: every text block binds the bank form, or none does.
    {
        std::size_t offloaded = 0;
        for (const auto& layer : out->weights.text.layers) {
            offloaded += std::holds_alternative<OffloadedMoeWeights>(layer.ffn) ? 1 : 0;
        }
        if (offloaded != 0 && offloaded != out->weights.text.layers.size()) {
            throw artifact::ArtifactError("Qwen3.5: every text MoE layer must store its experts as one bank, or none");
        }
        out->config.text.offloaded_experts = offloaded != 0;
        if (offloaded != 0) {
            // The embedding is pinned beside the banks (bind_text): a term of the RAM ledger.
            std::vector<artifact::ObjectHandle> objects;
            for (const auto& part : bindings.weights.at(out->weights.text.token_embedding.index).reference.binding.parts) {
                if (std::find(objects.begin(), objects.end(), part.object) == objects.end()) {
                    objects.push_back(part.object);
                }
            }
            for (const auto object : objects) {
                out->embedding_bytes += artifact::object_bytes(reader.directory().object(object));
            }
        }
    }
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text, options);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending = std::move(bindings.weights);
    // The overlay window borrows the evictable tail of the device weights, or with offloaded experts
    // expert frames (design D8), which needs no ladder.
    const bool ladder = options.overlay_vision() && !out->config.text.offloaded_experts;
    if (ladder) {
        for (auto& pending : out->pending) {
            const auto rank = overlay_evict_rank(pending.reference.name);
            if (rank == 0) { continue; }
            for (const auto& part : pending.reference.binding.parts) {
                binder.mark_device_evictable(part.object, rank);
            }
        }
    }
    out->materialization = std::move(binder).finish(ladder ? EvictableWeightPool::kChunkBytes : 1);
    if (options.overlay_vision()) {
        out->vision_overlay =
            compute_vision_overlay_layout(out->weights, out->pending, out->materialization);
    }
    out->info.name = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data = std::move(plan.impl_);
    std::unique_ptr<EvictableWeightPool> pool;
    const bool ladder = data->options.overlay_vision() && !data->config.text.offloaded_experts;
    if (ladder) {
        const auto& layout = *data->vision_overlay;
        if (layout.staging_bytes == 0) {
            throw std::logic_error("overlay vision requires a staged overlay window");
        }
        if (data->materialization.evictable_tail_bytes == 0) {
            throw std::logic_error("overlay vision requires an evictable weight ladder");
        }
        if (!EvictableWeightPool::supported(device.device)) {
            throw std::invalid_argument(
                "this device does not support virtual memory management, so "
                "--vision-offload on cannot be used; select off");
        }
        pool = std::make_unique<EvictableWeightPool>(EvictableWeightPool::Config{
            .arena_bytes          = data->materialization.device_capacity_bytes,
            .evictable_tail_bytes = data->materialization.evictable_tail_bytes,
            .overlay_bytes        = layout.staging_bytes,
            .device               = device.device});
    }
    const artifact::Reader& reader = *data->materialization.source;
    auto backing = artifact::materialize(reader, std::move(data->materialization), device, observer,
                                         std::move(pool));
    std::optional<VisionOverlayAssets> overlay;
    if (data->options.overlay_vision() && !ladder) {
        if (data->vision_overlay->staging_bytes == 0) {
            throw std::logic_error("overlay vision requires a staged overlay window");
        }
        const auto pinned = backing.pinned_bytes_range();
        if (pinned.empty()) { throw std::logic_error("overlay vision requires a pinned materialization"); }
        overlay = VisionOverlayAssets{.pool         = nullptr,
                                      .pinned_block = pinned.data(),
                                      .pinned_bytes = pinned.size(),
                                      .ladder_bytes = 0,
                                      .layout       = std::move(*data->vision_overlay)};
    } else if (data->options.overlay_vision()) {
        EvictableWeightPool* eviction = backing.eviction_pool();
        if (eviction == nullptr) {
            throw std::logic_error("overlay vision lost its evictable pool");
        }
        eviction->capture_tail_mirror(device.transfer_stream);
        const auto pinned = backing.pinned_bytes_range();
        if (pinned.empty()) {
            throw std::logic_error("overlay vision requires a pinned materialization");
        }
        overlay = VisionOverlayAssets{
            .pool         = eviction,
            .pinned_block = pinned.data(),
            .pinned_bytes = pinned.size(),
            .ladder_bytes = eviction->evictable_tail_bytes(),
            .layout       = std::move(*data->vision_overlay)};
    }
    // Streamed banks (the SSD tier, LoadOptions::stream_experts): their objects, before the pending
    // weights resolve.
    const bool streamed = data->config.text.offloaded_experts && data->options.stream_experts;
    std::vector<artifact::ObjectHandle> bank_objects, shared_objects;
    if (data->config.text.offloaded_experts) {
        for (const auto& layer : data->weights.text.layers) {
            const auto& w = std::get<OffloadedMoeWeights>(layer.ffn);
            if (streamed) {
                bank_objects.push_back(data->pending.at(w.experts.index).reference.binding.parts.front().object);
            }
            if (w.shared_expert) {
                shared_objects.push_back(data->pending.at(w.shared_expert->index).reference.binding.parts.front().object);
            }
        }
    }
    auto bound = loading::resolve_weights(std::move(data->pending), backing);
    // Offloaded MoE layers: the banks' planes and scalars. A pinned routed bank's multipliers (its scale
    // tail) are read in place; a streamed bank's records stay in the artifact and a shared expert's are in
    // device memory only, so their scale tails are read from the artifact into Model memory.
    std::vector<runtime::expert_offload::ExpertBank> banks, shared_banks;
    std::vector<std::uint64_t> bank_strides;
    std::vector<std::vector<float>> multipliers, shared_multipliers;
    const auto read_scale_tail = [&](const artifact::ObjectHandle& object, const WeightView& view) {
        const auto& geometry = view.parts.front().parent->geometry;
        std::vector<float> words(geometry.scale_bytes / sizeof(float));
        reader.read_into(artifact::object_offset(reader.directory().object(object)) + geometry.scale_offset,
                         std::as_writable_bytes(std::span(words)));
        return words;
    };
    if (data->config.text.offloaded_experts) {
        const auto& moe = std::get<MoeConfig>(data->config.text.ffn);
        for (std::size_t i = 0; i < data->weights.text.layers.size(); ++i) {
            const auto& w = std::get<OffloadedMoeWeights>(data->weights.text.layers[i].ffn);
            const auto& routed = bound.at(w.experts.index);
            try {
                if (streamed) {
                    std::vector<float> words = read_scale_tail(bank_objects[i], routed.view);
                    banks.push_back(runtime::expert_offload::a16_nvfp4_streamed_bank(
                        routed.view, words.data(), moe.num_experts, data->config.text.hidden_size,
                        moe.moe_intermediate_size, routed.name));
                    bank_strides.push_back(banks.back().planes.record_stride);
                    multipliers.push_back(std::move(words)); // the buffer moves: the planes' pointer holds
                } else {
                    banks.push_back(runtime::expert_offload::a16_nvfp4_bank(
                        routed.view, moe.num_experts, data->config.text.hidden_size,
                        moe.moe_intermediate_size, routed.name));
                }
                // A dense shared expert (an FP8 checkpoint's) binds as ordinary device projections.
                if (w.shared_expert) {
                    const auto& shared = bound.at(w.shared_expert->index);
                    std::vector<float> words = read_scale_tail(shared_objects[shared_banks.size()], shared.view);
                    shared_banks.push_back(runtime::expert_offload::a16_nvfp4_device_bank(
                        shared.view, words.data(), 1, data->config.text.hidden_size,
                        moe.shared_expert_intermediate_size, shared.name));
                    shared_multipliers.push_back(std::move(words));
                }
            } catch (const std::invalid_argument& error) {
                throw artifact::ArtifactError(error.what());
            }
        }
    }
    std::optional<runtime::expert_offload::ExpertStore> store;
    if (streamed) {
        store.emplace(backing.stream_source(), bank_objects, std::move(bank_strides),
                      std::get<MoeConfig>(data->config.text.ffn).num_experts, std::move(multipliers));
    }
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(data->resources), std::move(data->info), std::move(overlay),
        std::move(backing), std::move(banks), std::move(shared_banks), std::move(shared_multipliers),
        std::move(store)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace infernix::models::qwen3_5
