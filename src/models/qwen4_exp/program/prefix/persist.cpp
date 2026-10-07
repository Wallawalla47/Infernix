// Host tier persistence of Qwen4Exp's prefix cache (docs/maintainer/hybrid-prefix-cache-spec.md §5.5 and
// §17; design §19.3.1, X12). The file holds what a restarted Engine can resume from: every Host-backed
// snapshot and the Host-resident block path it anchors on, with the slab bytes, each block's MTP
// continuation token (`mtp_next`, rule MR5/MR6) and, inside each image's header slab, the snapshot's
// meta. Nothing in it is trusted unless the fingerprint (artifact, KV and drafter formats, binary)
// and the Host geometry (slab, page record and state-image layout) match; a damaged or foreign file
// loads nothing.
//
// Layout (version 1): the header, the block table (parents first), the snapshot table, then the
// slab bytes of every block and of every snapshot in table order, then the footer. The tables come
// first so a Host tier smaller than the file can choose what to restore before reading any slab.

#include "models/qwen4_exp/program/prefix/prefix_cache.h"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infernix::models::qwen4_exp::prefix {

namespace pc = runtime::prefix_cache;

namespace {

constexpr std::array<char, 8> kMagic  = {'N', 'I', 'N', 'F', 'Q', '4', 'P', 'C'};
constexpr std::array<char, 8> kFooter = {'N', 'I', 'N', 'F', 'Q', '4', 'E', 'D'};
constexpr std::uint32_t kVersion      = 1;
constexpr std::int32_t kRootIndex     = -1;
constexpr std::uint64_t kFooterBytes  = kFooter.size() + 2U * sizeof(std::uint32_t);
constexpr const char* kSaveAbandoned  = "abandoned; the previous file is kept";

struct Geometry {
    std::uint64_t slab_bytes        = 0;
    std::uint64_t image_bytes       = 0;
    std::uint64_t page_stride       = 0;
    std::uint64_t image_fingerprint = 0;
    std::uint32_t image_slabs       = 0;
    std::uint32_t reserved          = 0;

    [[nodiscard]] friend bool operator==(const Geometry&, const Geometry&) = default;
};

Geometry geometry_of(const PrefixHostLayout& host, const StateImageLayout& image) {
    return Geometry{.slab_bytes        = host.slab_bytes,
                    .image_bytes       = host.image_bytes,
                    .page_stride       = host.page.page_stride,
                    .image_fingerprint = image.fingerprint,
                    .image_slabs       = host.image_slabs};
}

struct SavedBlock {
    std::int32_t parent   = kRootIndex;
    std::uint64_t hash    = 0;
    std::uint64_t extra   = 0;
    std::int32_t mtp_next = -1;
    std::array<TokenId, pc::kBlockTokens> tokens{};
};

struct SavedSnapshot {
    std::int32_t anchor    = kRootIndex;
    std::uint32_t frontier = 0;
    std::uint32_t tail_len = 0;
    std::array<TokenId, pc::kBlockTokens> tail{};
    pc::SnapshotKind kind = pc::SnapshotKind::Tap;
    std::uint32_t hits    = 0;
    std::uint32_t slabs   = 0;
    // First slab of this snapshot, counted from the start of the slab bytes.
    std::uint64_t first_slab = 0;
};

class Writer {
public:
    explicit Writer(const std::filesystem::path& path) : out_(path, std::ios::binary | std::ios::trunc) {
        if (!out_) { throw std::runtime_error("cannot create " + path.string()); }
    }

    template <class T>
    void value(const T& value) {
        bytes(&value, sizeof(T));
    }

    void bytes(const void* data, std::size_t count) {
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(count));
        if (!out_) { throw std::runtime_error("prefix cache file write failed"); }
        written_ += count;
    }

    void finish() {
        out_.flush();
        if (!out_) { throw std::runtime_error("prefix cache file flush failed"); }
        out_.close();
    }

    [[nodiscard]] std::uint64_t written() const noexcept { return written_; }

private:
    std::ofstream out_;
    std::uint64_t written_ = 0;
};

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : in_(path, std::ios::binary) {}

    [[nodiscard]] bool open() const noexcept { return static_cast<bool>(in_); }

    template <class T>
    [[nodiscard]] T value() {
        T out{};
        bytes(&out, sizeof(T));
        return out;
    }

    void bytes(void* data, std::size_t count) {
        in_.read(static_cast<char*>(data), static_cast<std::streamsize>(count));
        if (!in_) { throw std::runtime_error("prefix cache file is truncated"); }
        read_ += count;
        position_ += count;
    }

    void seek(std::uint64_t offset) {
        if (offset == position_) { return; }
        in_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        if (!in_) { throw std::runtime_error("prefix cache file is truncated"); }
        position_ = offset;
    }

    // Bytes read, not counting what seeks passed over.
    [[nodiscard]] std::uint64_t read() const noexcept { return read_; }
    [[nodiscard]] std::uint64_t position() const noexcept { return position_; }

private:
    std::ifstream in_;
    std::uint64_t read_     = 0;
    std::uint64_t position_ = 0;
};

double seconds_since(std::chrono::steady_clock::time_point started) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

} // namespace

PersistResult PrefixCache::save(const std::filesystem::path& path, std::string_view fingerprint,
                                const CancellationView& abandoned) const {
    PersistResult out;
    const auto started = std::chrono::steady_clock::now();
    if (!host_tier()) {
        out.message = "the prefix cache has no Host tier to save";
        return out;
    }
    if (!captures_.empty() || !writes_.empty() || restore_.open || !landing_.empty()) {
        throw std::logic_error("Qwen4Exp prefix cache save requires idle transfers");
    }
    if (abandoned.requested()) {
        out.message = kSaveAbandoned;
        return out;
    }
    const auto check_abandoned = [&] {
        if (abandoned.requested()) { throw std::runtime_error(kSaveAbandoned); }
    };
    std::vector<pc::NodeRef> nodes;
    std::vector<pc::SnapshotRef> snapshots;
    index_->collect_persistable(nodes, snapshots);

    std::filesystem::path temporary = path;
    temporary += ".tmp";
    try {
        Writer writer(temporary);
        writer.bytes(kMagic.data(), kMagic.size());
        writer.value(kVersion);
        writer.value(static_cast<std::uint32_t>(fingerprint.size()));
        writer.bytes(fingerprint.data(), fingerprint.size());
        writer.value(geometry_of(host_, lanes_->layout()));
        writer.value(static_cast<std::uint32_t>(nodes.size()));
        writer.value(static_cast<std::uint32_t>(snapshots.size()));

        std::unordered_map<std::uint32_t, std::int32_t> file_index;
        file_index.reserve(nodes.size());
        for (const pc::NodeRef node : nodes) {
            const pc::BlockIdentity identity = index_->block_identity(node);
            writer.value(identity.parent.valid() ? file_index.at(identity.parent.index) : kRootIndex);
            writer.value(identity.lookup_hash);
            writer.value(identity.extra);
            writer.value(node_mtp_next_[node.index]);
            writer.bytes(identity.tokens.data(), identity.tokens.size_bytes());
            file_index.emplace(node.index, static_cast<std::int32_t>(file_index.size()));
        }
        for (const pc::SnapshotRef snapshot : snapshots) {
            const pc::SnapshotView view = index_->snapshot(snapshot);
            if (view.host_slabs.size() != host_.image_slabs + (view.tail_len != 0 ? 1U : 0U)) {
                throw std::logic_error("a saved snapshot's Host slabs do not match its tail");
            }
            writer.value(view.anchor.valid() ? file_index.at(view.anchor.index) : kRootIndex);
            writer.value(view.frontier);
            writer.value(view.tail_len);
            writer.bytes(view.tail.data(), view.tail.size_bytes());
            writer.value(static_cast<std::uint8_t>(view.kind));
            writer.value(view.hits);
        }
        for (const pc::NodeRef node : nodes) {
            check_abandoned();
            writer.bytes(slab(index_->node(node).host_slab), host_.slab_bytes);
        }
        for (const pc::SnapshotRef snapshot : snapshots) {
            for (const std::uint32_t slab_id : index_->snapshot(snapshot).host_slabs) {
                check_abandoned();
                writer.bytes(slab(slab_id), host_.slab_bytes);
            }
        }
        writer.bytes(kFooter.data(), kFooter.size());
        writer.value(static_cast<std::uint32_t>(nodes.size()));
        writer.value(static_cast<std::uint32_t>(snapshots.size()));
        writer.finish();
        out.bytes = writer.written();
        // Abandoned after the last slab: the previous file still stays.
        check_abandoned();
        std::filesystem::rename(temporary, path);
    } catch (const std::exception& error) {
        // The writer is closed by now, so the temporary file can be deleted on every platform.
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        out.message = error.what();
        return out;
    }
    out.ok        = true;
    out.blocks    = nodes.size();
    out.snapshots = snapshots.size();
    out.seconds   = seconds_since(started);
    return out;
}

PersistResult PrefixCache::load(const std::filesystem::path& path, std::string_view fingerprint,
                                const StartupObserver& observer) {
    PersistResult out;
    const auto started = std::chrono::steady_clock::now();
    if (!host_tier()) {
        out.message = "the prefix cache has no Host tier to load into";
        return out;
    }
    if (index_->stats().nodes != 0 || index_->stats().snapshots != 0) {
        throw std::logic_error("Qwen4Exp prefix cache load requires an empty cache");
    }
    Reader reader(path);
    if (!reader.open()) {
        out.message = "no saved prefix cache at " + path.string() + " yet; it is written at shutdown";
        return out;
    }
    // Opened once the header matches, so a missing or foreign file shows only the reason it was
    // not restored. A file damaged part way through completes it with the bytes that were read.
    std::optional<StartupPhaseScope> phase;
    try {
        std::array<char, 8> magic{};
        reader.bytes(magic.data(), magic.size());
        if (magic != kMagic || reader.value<std::uint32_t>() != kVersion) {
            out.message = "not a prefix cache file of this format version";
            return out;
        }
        const auto fingerprint_bytes = reader.value<std::uint32_t>();
        if (fingerprint_bytes > (1U << 20U)) {
            out.message = "prefix cache file header is damaged";
            return out;
        }
        std::string saved(fingerprint_bytes, '\0');
        reader.bytes(saved.data(), saved.size());
        if (saved != fingerprint) {
            out.message = "saved prefix cache belongs to a different model, KV format or build";
            return out;
        }
        if (reader.value<Geometry>() != geometry_of(host_, lanes_->layout())) {
            out.message = "saved prefix cache has a different Host geometry";
            return out;
        }
        const std::uint64_t file_bytes = std::filesystem::file_size(path);
        const std::uint64_t slab_bytes = host_.slab_bytes;
        const auto node_count          = reader.value<std::uint32_t>();
        const auto snapshot_count      = reader.value<std::uint32_t>();
        // Every entry takes at least one slab of the file: bounds the tables before allocating.
        if (static_cast<std::uint64_t>(node_count) + snapshot_count > file_bytes / slab_bytes) {
            throw std::runtime_error("prefix cache file header is damaged");
        }

        std::vector<SavedBlock> blocks(node_count);
        std::vector<std::int32_t> parents(node_count);
        for (std::uint32_t index = 0; index < node_count; ++index) {
            SavedBlock& block = blocks[index];
            block.parent      = reader.value<std::int32_t>();
            block.hash        = reader.value<std::uint64_t>();
            block.extra       = reader.value<std::uint64_t>();
            block.mtp_next    = reader.value<std::int32_t>();
            reader.bytes(block.tokens.data(), sizeof(block.tokens));
            if (block.parent >= static_cast<std::int32_t>(index) || block.parent < kRootIndex) {
                throw std::runtime_error("prefix cache file block order is damaged");
            }
            parents[index] = block.parent;
        }
        std::vector<SavedSnapshot> snapshots(snapshot_count);
        std::vector<pc::SavedSnapshotShape> shapes(snapshot_count);
        std::uint64_t slabs = node_count;
        for (std::uint32_t index = 0; index < snapshot_count; ++index) {
            SavedSnapshot& snapshot = snapshots[index];
            snapshot.anchor         = reader.value<std::int32_t>();
            snapshot.frontier       = reader.value<std::uint32_t>();
            snapshot.tail_len       = reader.value<std::uint32_t>();
            if (snapshot.tail_len >= pc::kBlockTokens || snapshot.anchor >= static_cast<std::int32_t>(node_count) ||
                snapshot.anchor < kRootIndex) {
                throw std::runtime_error("prefix cache file snapshot is damaged");
            }
            reader.bytes(snapshot.tail.data(), static_cast<std::size_t>(snapshot.tail_len) * sizeof(TokenId));
            const auto kind_byte = reader.value<std::uint8_t>();
            if (kind_byte > static_cast<std::uint8_t>(pc::SnapshotKind::Boundary)) {
                throw std::runtime_error("prefix cache file snapshot kind is damaged");
            }
            snapshot.kind       = static_cast<pc::SnapshotKind>(kind_byte);
            snapshot.hits       = reader.value<std::uint32_t>();
            snapshot.slabs      = host_.image_slabs + (snapshot.tail_len != 0 ? 1U : 0U);
            snapshot.first_slab = slabs;
            slabs += snapshot.slabs;
            shapes[index] = pc::SavedSnapshotShape{.anchor   = snapshot.anchor,
                                                   .frontier = snapshot.frontier,
                                                   .slabs    = snapshot.slabs,
                                                   .hits     = snapshot.hits};
        }
        const std::uint64_t payload = reader.position();
        if (file_bytes != payload + slabs * slab_bytes + kFooterBytes) {
            throw std::runtime_error("prefix cache file is truncated or damaged");
        }
        out.saved_blocks        = node_count;
        out.saved_snapshots     = snapshot_count;
        out.required_host_bytes = slabs * slab_bytes;
        out.host_bytes          = static_cast<std::uint64_t>(host_.slabs) * slab_bytes;

        const pc::HostRestorePlan plan = pc::plan_host_restore(parents, shapes, host_.slabs, index_->config());
        const std::uint64_t planned    = reader.read() + plan.slabs * slab_bytes + kFooterBytes;
        phase.emplace(observer, StartupPhase::PrefixCacheLoad, StartupProgressUnit::Bytes, planned);
        phase->progress(reader.read(), planned);

        // Restored index ids of the file's blocks; absent when a block was not chosen or did not
        // restore, so its descendants and the snapshots on it are skipped too.
        std::vector<std::optional<pc::NodeRef>> restored(node_count);
        for (std::uint32_t index = 0; index < node_count; ++index) {
            const SavedBlock& saved_block = blocks[index];
            if (!plan.blocks[index] ||
                (saved_block.parent != kRootIndex && !restored[static_cast<std::size_t>(saved_block.parent)])) {
                continue;
            }
            const pc::NodeRef parent =
                saved_block.parent == kRootIndex ? pc::NodeRef{} : *restored[static_cast<std::size_t>(saved_block.parent)];
            const std::optional<pc::RestoredBlock> block =
                index_->restore_host_block(parent, saved_block.hash, saved_block.tokens, saved_block.extra);
            if (!block) { continue; }
            reader.seek(payload + static_cast<std::uint64_t>(index) * slab_bytes);
            reader.bytes(slab(block->slab), slab_bytes);
            node_mtp_next_[block->node.index] = saved_block.mtp_next;
            restored[index]                   = block->node;
            ++out.blocks;
            phase->progress(reader.read(), planned);
        }
        for (std::uint32_t index = 0; index < snapshot_count; ++index) {
            const SavedSnapshot& saved_snapshot = snapshots[index];
            if (!plan.snapshots[index] ||
                (saved_snapshot.anchor != kRootIndex && !restored[static_cast<std::size_t>(saved_snapshot.anchor)])) {
                continue;
            }
            const pc::NodeRef anchor = saved_snapshot.anchor == kRootIndex
                                           ? pc::NodeRef{}
                                           : *restored[static_cast<std::size_t>(saved_snapshot.anchor)];
            const std::optional<pc::SnapshotRef> snapshot = index_->restore_host_snapshot(
                anchor, saved_snapshot.frontier,
                std::span<const TokenId>(saved_snapshot.tail.data(), saved_snapshot.tail_len), saved_snapshot.kind,
                saved_snapshot.hits);
            if (!snapshot) { continue; }
            reader.seek(payload + saved_snapshot.first_slab * slab_bytes);
            const std::span<const std::uint32_t> slab_ids = index_->snapshot(*snapshot).host_slabs;
            for (const std::uint32_t slab_id : slab_ids) { reader.bytes(slab(slab_id), slab_bytes); }
            // The image's header slab carries the snapshot's meta (MTP cell state, lineage).
            const std::optional<StateImageHeader> header = read_state_image_header(lanes_->layout(), slab(slab_ids[0]));
            if (!header || header->frontier != saved_snapshot.frontier) {
                throw std::runtime_error("prefix cache file snapshot meta is damaged");
            }
            snapshot_meta_[snapshot->index] = *header;
            ++out.snapshots;
            phase->progress(reader.read(), planned);
        }
        reader.seek(payload + slabs * slab_bytes);
        std::array<char, 8> footer{};
        reader.bytes(footer.data(), footer.size());
        if (footer != kFooter || reader.value<std::uint32_t>() != node_count ||
            reader.value<std::uint32_t>() != snapshot_count) {
            throw std::runtime_error("prefix cache file footer is damaged");
        }
        phase->complete(reader.read());
    } catch (const std::exception& error) {
        if (phase) { phase->complete(reader.read()); }
        // Partially restored entries may hold unread slab bytes: drop them all.
        clear();
        out         = PersistResult{};
        out.message = error.what();
        return out;
    }
    out.ok      = true;
    out.bytes   = reader.read();
    out.seconds = seconds_since(started);
    return out;
}

} // namespace infernix::models::qwen4_exp::prefix
