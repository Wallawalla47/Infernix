// Qwen4Exp's binding of the prefix cache file (runtime/prefix_cache/persist_file.h; docs/maintainer/
// hybrid-prefix-cache-spec.md §5.5 and §17; design §19.3.1, X12): its file kind, its Host geometry (slab,
// page record and state-image layout), each block's MTP continuation token (`mtp_next`, rule MR5/MR6)
// and each snapshot's meta (StateImageHeader), which also sits in the image's header slab: a Host
// restore checks the two agree, and an SSD-only snapshot takes the saved meta, since its header slab is
// read only when it is promoted.

#include "models/qwen4_exp/program/prefix/prefix_cache.h"

#include <array>
#include <cstring>
#include <stdexcept>

namespace infernix::models::qwen4_exp::prefix {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::array<char, 8> kMagic = {'N', 'I', 'N', 'F', 'Q', '4', 'P', 'C'};

template <class T>
void append(std::vector<std::byte>& out, const T& value) {
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

} // namespace

pc::PersistBinding PrefixCache::persist_binding() {
    pc::PersistBinding binding;
    binding.magic = kMagic;
    append(binding.geometry, static_cast<std::uint64_t>(host_.slab_bytes));
    append(binding.geometry, static_cast<std::uint64_t>(host_.image_bytes));
    append(binding.geometry, static_cast<std::uint64_t>(host_.page.page_stride));
    append(binding.geometry, static_cast<std::uint64_t>(lanes_->layout().fingerprint));
    append(binding.geometry, static_cast<std::uint32_t>(host_.image_slabs));
    binding.slab_bytes  = host_.slab_bytes;
    binding.image_slabs = host_.image_slabs;
    binding.host_slabs  = index_->host_slab_limit(); // in service now: a load fills no more
    binding.slab        = [this](std::uint32_t id) { return slab(id); };
    binding.block_word  = [this](pc::NodeRef node) { return node_mtp_next_[node.index]; };
    binding.set_block_word = [this](pc::NodeRef node, std::int32_t word) { node_mtp_next_[node.index] = word; };
    binding.meta_bytes     = sizeof(StateImageHeader);
    binding.snapshot_meta  = [this](pc::SnapshotRef snapshot, std::byte* out) {
        std::memcpy(out, &snapshot_meta_[snapshot.index], sizeof(StateImageHeader));
    };
    binding.restore_meta = [this](pc::SnapshotRef snapshot, const std::byte* saved, bool host_read) {
        StateImageHeader meta;
        std::memcpy(&meta, saved, sizeof(meta));
        if (host_read) {
            // The image's header slab carries the snapshot's meta (MTP cell state, lineage).
            const std::optional<StateImageHeader> header =
                read_state_image_header(lanes_->layout(), slab(index_->snapshot(snapshot).host_slabs[0]));
            if (!header || header->frontier != meta.frontier) { return false; }
        }
        if (meta.frontier != index_->snapshot(snapshot).frontier) { return false; }
        snapshot_meta_[snapshot.index] = meta;
        return true;
    };
    binding.ssd = ssd_ && ssd_->kept() ? ssd_.get() : nullptr;
    return binding;
}

pc::PersistResult PrefixCache::save(const std::filesystem::path& path, std::string_view fingerprint,
                                    const CancellationView& abandoned) {
    if (!host_tier()) {
        pc::PersistResult out;
        out.message = "the prefix cache has no Host tier to save";
        return out;
    }
    if (!captures_.empty() || !writes_.empty() || restore_.open || !landing_.empty()) {
        throw std::logic_error("Qwen4Exp prefix cache save requires idle transfers");
    }
    // A kept SSD tier's copies are named by the file: every transfer lands before (the file then
    // makes them durable).
    if (ssd_ && ssd_->kept()) { ssd_->drain(*index_); }
    return pc::save_prefix_cache(*index_, persist_binding(), path, fingerprint, abandoned);
}

pc::PersistResult PrefixCache::load(const std::filesystem::path& path, std::string_view fingerprint,
                                    const StartupObserver& observer) {
    if (!host_tier()) {
        pc::PersistResult out;
        out.message = "the prefix cache has no Host tier to load into";
        return out;
    }
    if (index_->stats().nodes != 0 || index_->stats().snapshots != 0) {
        throw std::logic_error("Qwen4Exp prefix cache load requires an empty cache");
    }
    pc::PersistResult out = pc::load_prefix_cache(*index_, persist_binding(), path, fingerprint, observer);
    // Partially restored entries may hold unread slab bytes: drop them all.
    if (!out.ok && (index_->stats().nodes != 0 || index_->stats().snapshots != 0)) { clear(); }
    return out;
}

} // namespace infernix::models::qwen4_exp::prefix
