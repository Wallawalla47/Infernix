// Qwen3.5's binding of the prefix cache file (runtime/prefix_cache/persist_file.h;
// docs/maintainer/hybrid-prefix-cache-spec.md §5.5): its file kind and its Host geometry. A state image
// carries everything a resume needs, so no word per block or meta per snapshot is kept beside the index.

#include "models/qwen3_5/program/prefix/hybrid_cache.h"

#include <array>
#include <stdexcept>

namespace infernix::models::qwen3_5::detail {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::array<char, 8> kMagic = {'N', 'I', 'N', 'F', 'H', 'P', 'C', '1'};

template <class T>
void append(std::vector<std::byte>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

} // namespace

pc::PersistBinding HybridPrefixCache::persist_binding() {
    pc::PersistBinding binding;
    binding.magic = kMagic;
    append(binding.geometry, static_cast<std::uint64_t>(host_layout_.slab_bytes));
    append(binding.geometry, static_cast<std::uint64_t>(host_layout_.image_bytes));
    append(binding.geometry, static_cast<std::uint64_t>(host_layout_.block_payload_bytes));
    append(binding.geometry, static_cast<std::uint64_t>(host_layout_.text.page_stride));
    append(binding.geometry, static_cast<std::uint64_t>(host_layout_.backend ? host_layout_.backend->page_stride : 0U));
    append(binding.geometry, static_cast<std::uint64_t>(host_layout_.backend_offset));
    append(binding.geometry, static_cast<std::uint32_t>(host_layout_.image_slabs));
    binding.slab_bytes  = host_layout_.slab_bytes;
    binding.image_slabs = host_layout_.image_slabs;
    binding.host_slabs  = config_.host_slabs;
    binding.slab        = [this](std::uint32_t id) { return slab(id); };
    binding.ssd         = ssd_ && ssd_->kept() ? ssd_.get() : nullptr;
    return binding;
}

pc::PersistResult HybridPrefixCache::save(const std::filesystem::path& path, std::string_view fingerprint,
                                          const CancellationView& abandoned) {
    if (!host_tier()) {
        pc::PersistResult out;
        out.message = "the hybrid prefix cache has no Host tier to save";
        return out;
    }
    if (!pending_.empty() || restore_.open || !landing_.empty()) {
        throw std::logic_error("hybrid prefix cache save requires idle transfers");
    }
    // A kept SSD tier's copies are named by the file: every transfer lands before (the file then makes
    // them durable).
    if (ssd_ && ssd_->kept()) { ssd_->drain(*index_); }
    return pc::save_prefix_cache(*index_, persist_binding(), path, fingerprint, abandoned);
}

pc::PersistResult HybridPrefixCache::load(const std::filesystem::path& path, std::string_view fingerprint,
                                          const StartupObserver& observer) {
    if (!host_tier()) {
        pc::PersistResult out;
        out.message = "the hybrid prefix cache has no Host tier to load into";
        return out;
    }
    if (index_->stats().nodes != 0 || index_->stats().snapshots != 0) {
        throw std::logic_error("hybrid prefix cache load requires an empty cache");
    }
    pc::PersistResult out = pc::load_prefix_cache(*index_, persist_binding(), path, fingerprint, observer);
    // Partially restored entries may hold unread slab bytes: drop them all.
    if (!out.ok && (index_->stats().nodes != 0 || index_->stats().snapshots != 0)) { clear(); }
    return out;
}

} // namespace infernix::models::qwen3_5::detail
